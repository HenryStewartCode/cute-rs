// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   02_gemm_simple_ts/01_fp16_bf16_gemm_3.py (its configuration, the resources
//   GmemAbResource, SmemAbResource, TmemCResource and GmemDResource with their
//   work methods, the schedules load_schedule, mma_schedule, store_schedule and
//   padding_schedule, the tasks LoadTask, MmaTask, StoreTask and PaddingTask with
//   their register budgets, the allocator and TaskManager calls of
//   `_create_gemm_pipeline`, the kernel's descriptor prefetch and the host
//   function's TMA boxes and grid),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The single-CTA Task Scheduling tcgen05 GEMM, ported from
//! `02_gemm_simple_ts/01_fp16_bf16_gemm_3.py`: `D = A B^T` with A `(M, K)`
//! and B `(N, K)` K-major FP16 or BF16, an FP32 accumulator in tensor memory
//! and D `(M, N)` row-major in the input type. Each CTA computes one
//! `128 x 256` tile, walking K in 64-element stages. The task graph is
//! linear, `GmemAb -> SmemAb -> TmemC -> GmemD`:
//!
//! * `GmemAb` (no storage, no pipeline) turns the CTA and the K stage into
//!   the TMA coordinates `coord_k`, `coord_m`, `coord_n`;
//! * `SmemAb` holds three stages of A and B behind one `TmaUmma` pipeline
//!   (the transaction count is one A tile plus one B tile). Its producer
//!   work `tma_load` issues both TMA loads from an elected lane; its consumer
//!   work `build_descriptors` returns the stage's two tcgen05 shared-memory
//!   descriptors (128-byte swizzle, LBO 16, SBO 1024) as task-local values;
//! * `TmemC` holds two accumulator stages (512 TMEM columns, laid out by the
//!   `TmemAllocator`) behind a `UmmaAsync` pipeline. Its producer work `mma`
//!   issues four `128x256x16` `tcgen05.mma`s per K stage, advancing both
//!   descriptors by 32 bytes and accumulating after the first (`scale_d`, a
//!   per-tile flag reset by `init_work_tile_state`); its consumer work
//!   `load_subtile` reads a 32-column subtile (`tcgen05.ld.32x32b.x32`) into
//!   a register vector;
//! * `GmemD` stores each subtile, converted to the input type, with 16-byte
//!   vector stores and a scalar tail at the ragged right edge.
//!
//! `LoadTask` (warp 4) and `MmaTask` (warp 5) run the K loop; `StoreTask`
//! (warps 0-3) waits for the committed accumulator after an empty pacing
//! loop and drains it in eight subtiles, whose index is a compile-time work
//! argument; `PaddingTask` (warps 6-7) completes the second warpgroup. The
//! register budgets are the tutorial's: 160 for the store warpgroup, 40 for
//! the other (shared by three tasks, so the TaskManager sets it once per
//! warpgroup).
//!
//! Differences from the tutorial: the TaskManager's TMEM allocator
//! allocates, publishes and frees the columns (the tutorial calls
//! `tcgen05_alloc`, a named barrier and `tcgen05_dealloc` around `run()`
//! itself); the `SmemAllocator` also places TmemC's barriers (cute-rs's
//! TaskManager requires every pipeline's barriers to be placed); the A and
//! B buffers are 1024-byte aligned (the 128-byte
//! swizzle's atom) rather than 128; the TMA loads are the `cute` TMA atoms
//! (`make_tiled_tma_atom` over the swizzled shared-memory layout) rather
//! than raw tensor maps; and `M`, `N`, `K` need not be multiples of the tile:
//! TMA zero-fills what lies outside A and B, and the store skips rows and
//! columns outside D (K must be a multiple of 8, for TMA's 16-byte strides).
//!
//! No Blackwell GPU is available, so the kernel runs on the simulator (the
//! tests also compile it for sm_100a and assemble the PTX with `ptxas`).
//! Run with `cargo run --release --example ts_gemm_simple`.

use std::cell::RefCell;
use std::rc::Rc;

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::prims::{
    self, Tcgen05InstrDesc, Tcgen05MmaKind, Tcgen05MmaOptions, Tcgen05SmemDesc, Tcgen05SmemDescFields,
    Tcgen05SmemSwizzle,
};
use cute::task_scheduling::*;
use cute::tcgen05::{self, CtaGroup, Ld32x32bOp, Repetition, SmemLayoutAtomKind};

/// The per-instruction tcgen05 MMA shape (`mma_inst_shape_mnk`).
pub const MMA_INST_SHAPE_MNK: (usize, usize, usize) = (128, 256, 16);
/// The CTA tile and the K step of one TMA stage (`mma_tiler_mnk`).
pub const MMA_TILER_MNK: (usize, usize, usize) = (128, 256, 64);
/// Shared-memory A/B stages (`ab_stages`).
pub const AB_STAGES: usize = 3;
/// TMEM accumulator stages (`acc_stages`).
pub const ACC_STAGES: usize = 2;
/// Threads of the store warpgroup (`threads_in_epilogue`).
pub const THREADS_IN_EPILOGUE: u32 = 128;
/// Columns of one T2R load (`t2r_inst_repx`).
pub const T2R_INST_REPX: usize = 32;
/// Threads per CTA: eight warps.
pub const THREADS: u32 = 256;
/// Elements of one 16-byte vector store of D.
const VSIZE: usize = 8;
/// One TMEM lane in a TMEM address (the row field).
const TMEM_LANE: i32 = 1 << 16;

/// The problem `main` runs (the tutorial's default, 8192^3, is too large
/// to simulate).
pub const MNK: (usize, usize, usize) = (256, 512, 256);

/// The swizzled K-major shared-memory layout of one stage of an `(mn, 64)`
/// operand tile: 8-row atoms of 1024 bytes, rows of 128 bytes.
fn smem_layout<T: Numeric>(mn: usize) -> ComposedLayout {
    let atom = tcgen05::make_smem_layout_atom::<T>(SmemLayoutAtomKind::K_SW128);
    tile_to_shape_swizzled(&atom, (mn, MMA_TILER_MNK.2))
}

type Cell<T> = Rc<RefCell<Option<T>>>;

/// The tutorial's I/O types (`_DTYPE_MAP`): FP16 and BF16, with their host
/// conversions.
pub trait IoDtype: Numeric {
    fn from_f32(x: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl IoDtype for F16 {
    fn from_f32(x: f32) -> Self {
        F16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        F16::to_f32(self)
    }
}

impl IoDtype for BF16 {
    fn from_f32(x: f32) -> Self {
        BF16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        BF16::to_f32(self)
    }
}

fn get<T: Clone>(c: &Cell<T>) -> T {
    c.borrow().clone().expect("the task's auxiliary init work ran first")
}

/// The GEMM kernel for an `m x n x k` problem in `T` (`F16` or `BF16`),
/// with IKET ranges per task and step when `iket` is given.
pub fn build<T: IoDtype>(m: usize, n: usize, k: usize, iket: Option<IketOptions>) -> Kernel {
    assert!(m > 0 && n > 0 && k > 0 && k.is_multiple_of(8), "K must be a positive multiple of 8 (TMA strides)");
    let (bm, bn, bk) = MMA_TILER_MNK;
    let num_k_tiles = k.div_ceil(bk);
    let name = format!("ts_gemm_simple_{}", T::DTYPE).to_lowercase();
    // One CTA per SM (`.minnctapersm 1`), so that ptxas knows the register
    // count the `setmaxnreg`s start from.
    let mut options = Kernel::builder(&name).block_dim([THREADS, 1, 1]).min_blocks_per_sm(1).target(Arch::Sm100a);
    if let Some(o) = iket {
        options = options.iket(o);
    }
    options.build(move || {
        let a = param_tensor_aligned::<T>("a", Layout::new((m, k), (k, 1)), 16);
        let b = param_tensor_aligned::<T>("b", Layout::new((n, k), (k, 1)), 16);
        let c = param_tensor_aligned::<T>("c", Layout::new((m, n), (n, 1)), 16);

        // The host function's TMA descriptors: (128, 64) and (256, 64) boxes
        // in the 128-byte swizzle.
        let (a_layout, b_layout) = (smem_layout::<T>(bm), smem_layout::<T>(bn));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (tma_atom_a, tma_a) = cpasync::make_tiled_tma_atom(g2s, &a, a_layout.clone(), (bm, bk), 1).into();
        let (tma_atom_b, tma_b) = cpasync::make_tiled_tma_atom(g2s, &b, b_layout.clone(), (bn, bk), 1).into();

        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(4), || {
            cpasync::prefetch_descriptor(&tma_atom_a);
            cpasync::prefetch_descriptor(&tma_atom_b);
        });

        // ---------------------------------------------------------------
        // GmemAbResource: the coordinates of the next TMA loads.
        // ---------------------------------------------------------------
        let coord_k = TaskLocalVariable::<Int32>::scalar("coord_k", 0, "K coordinate of the next TMA tile");
        let coord_m = TaskLocalVariable::<Int32>::scalar("coord_m", 0, "M coordinate of the next TMA tile");
        let coord_n = TaskLocalVariable::<Int32>::scalar("coord_n", 0, "N coordinate of the next TMA tile");
        let gmem_ab = MemoryResource::new("GmemAb").work(
            Work::consumer("compute_coords").returns(&coord_k).returns(&coord_m).returns(&coord_n).run({
                let (ck, cm, cn) = (coord_k.clone(), coord_m.clone(), coord_n.clone());
                move |si, _| {
                    let bi = block_idx();
                    WorkResults::of(&ck, si.loop_offset * bk as i32)
                        .and(&cm, bi.x * bm as i32)
                        .and(&cn, bi.y * bn as i32)
                }
            }),
        );

        // ---------------------------------------------------------------
        // SmemAbResource: three stages of A and B behind a TmaUmma pipeline.
        // ---------------------------------------------------------------
        let (sa_elems, sb_elems) = (bm * bk, bn * bk);
        let elem_bytes = T::DTYPE.bits() / 8;
        let smem_ab_cfg = PipelineConfig::create_tma_umma_pipeline_cfg(
            AB_STAGES,
            ((sa_elems + sb_elems) * elem_bytes) as u32,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 1),
            Layout::col_major((1usize, 1usize, 1usize, 1usize)),
        );
        let alloc_a = SmemAllocation::new::<T>("smem_a", sa_elems * AB_STAGES, 1024);
        let alloc_b = SmemAllocation::new::<T>("smem_b", sb_elems * AB_STAGES, 1024);
        let desc_a_base = TaskLocalVariable::<Value<u64>>::scalar("desc_a_base", 0, "SMEM descriptor of A's stage");
        let desc_b_base = TaskLocalVariable::<Value<u64>>::scalar("desc_b_base", 0, "SMEM descriptor of B's stage");
        let shared_smem: Cell<(Ptr<T>, Ptr<T>)> = Rc::default();
        // The typed views over the allocator's block (`_init_smem_state`).
        let init_smem_state = {
            let (alloc_a, alloc_b, views) = (alloc_a.clone(), alloc_b.clone(), shared_smem.clone());
            move |si: &StageInfo, _: &WorkArgs| {
                *views.borrow_mut() = Some((alloc_a.ptr::<T>(&si.context), alloc_b.ptr::<T>(&si.context)));
                WorkResults::none()
            }
        };
        let smem_ab = MemoryResource::new("SmemAb")
            .with_pipeline_config(smem_ab_cfg)
            .smem_requirement(alloc_a)
            .smem_requirement(alloc_b)
            .work(Work::producer("init_load_state").auxiliary().run(init_smem_state.clone()))
            .work(Work::consumer("init_descriptors").auxiliary().run(init_smem_state))
            .work(Work::consumer("build_descriptors").returns(&desc_a_base).returns(&desc_b_base).run({
                let (da, db, views) = (desc_a_base.clone(), desc_b_base.clone(), shared_smem.clone());
                move |si, _| {
                    let stage = si.stage_idx.expect("a pipelined stage");
                    let (sa, sb) = get(&views);
                    let fields = Tcgen05SmemDescFields {
                        leading_byte_offset: 16,
                        stride_byte_offset: 1024,
                        swizzle: Tcgen05SmemSwizzle::Swizzle128B,
                        ..Default::default()
                    };
                    let desc = |p: Ptr<T>| Tcgen05SmemDesc::build(&p, &fields).unwrap_or_else(|e| panic!("{e}"));
                    let a = desc(sa.offset(stage * sa_elems as i32)).value();
                    let b = desc(sb.offset(stage * sb_elems as i32)).value();
                    WorkResults::of(&da, a).and(&db, b)
                }
            }))
            .work(
                Work::producer("tma_load")
                    .input::<Int32>("coord_k")
                    .input::<Int32>("coord_m")
                    .input::<Int32>("coord_n")
                    .run({
                        let views = shared_smem.clone();
                        move |si, args| {
                            let (ck, cm, cn): (Int32, Int32, Int32) =
                                (args.get("coord_k"), args.get("coord_m"), args.get("coord_n"));
                            let stage = si.stage_idx.expect("a pipelined stage");
                            let bar = si.barrier().expect("the stage's full barrier");
                            let (sa, sb) = get(&views);
                            let load =
                                |atom: &CopyAtom<T>, tma: &Tensor<Ident>, s: Ptr<T>, layout, rows, row: Int32| {
                                    let s = make_tensor(&s, layout);
                                    let g = local_tile(tma, (rows, bk), (row / rows as i32, ck / bk as i32));
                                    let (ts, tg) = cpasync::tma_partition(
                                        atom,
                                        0,
                                        &make_layout(1, 0),
                                        &s.group_modes(0, 2),
                                        &g.group_modes(0, 2),
                                    );
                                    (ts, tg)
                                };
                            let (tsa, tga) =
                                load(&tma_atom_a, &tma_a, sa.offset(stage * sa_elems as i32), a_layout.clone(), bm, cm);
                            let (tsb, tgb) =
                                load(&tma_atom_b, &tma_b, sb.offset(stage * sb_elems as i32), b_layout.clone(), bn, cn);
                            if_(prims::elect_sync(u32::MAX), || {
                                copy_with(&tma_atom_a, &tga, &tsa, CopyArgs::new().tma_bar_ptr(&bar));
                                copy_with(&tma_atom_b, &tgb, &tsb, CopyArgs::new().tma_bar_ptr(&bar));
                            });
                            WorkResults::none()
                        }
                    }),
            );

        // ---------------------------------------------------------------
        // TmemCResource: two accumulator stages behind a UmmaAsync pipeline.
        // ---------------------------------------------------------------
        let tmem_c_cfg = PipelineConfig::create_umma_async_pipeline_cfg(
            ACC_STAGES,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, THREADS_IN_EPILOGUE),
            Layout::col_major((1usize, 1usize, 1usize, 1usize)),
        );
        let alloc_acc = TmemAllocation::new("tmem_acc", bn * ACC_STAGES);
        let t2r_rmem = TaskLocalVariable::<Vector<f32>>::new(
            "t2r_rmem",
            || vector::full(T2R_INST_REPX, 0.0f32),
            "a register subtile loaded from TMEM for the epilogue",
        );
        // The instruction descriptor: A and B in `T`, D in FP32, 128 x 256,
        // both K-major.
        let idesc = Tcgen05InstrDesc::for_mma(
            UmmaDataKind::F16,
            T::DTYPE,
            T::DTYPE,
            DType::F32,
            (MMA_INST_SHAPE_MNK.0 as u32, MMA_INST_SHAPE_MNK.1 as u32),
            CtaGroup::ONE,
            OperandMajorMode::K,
            OperandMajorMode::K,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let tmem_base: Cell<Ptr<f32>> = Rc::default();
        let scale_d: Cell<Rc<Var<Boolean>>> = Rc::default();
        // The accumulator's base address (`_init_tmem_state`).
        let init_tmem_state = {
            let (alloc, base) = (alloc_acc.clone(), tmem_base.clone());
            move |si: &StageInfo, _: &WorkArgs| {
                *base.borrow_mut() = Some(alloc.ptr::<f32>(&si.context));
                WorkResults::none()
            }
        };
        let tmem_c = MemoryResource::new("TmemC")
            .with_pipeline_config(tmem_c_cfg)
            .tmem_requirement(alloc_acc)
            .work(Work::producer("init_accumulator_state").auxiliary().run(init_tmem_state.clone()))
            .work(Work::consumer("init_store_state").auxiliary().run(init_tmem_state))
            .work(Work::producer("init_work_tile_state").auxiliary().run({
                let scale_d = scale_d.clone();
                // The first MMA of a tile overwrites the accumulator.
                move |_, _| {
                    *scale_d.borrow_mut() = Some(Rc::new(Var::new(Boolean::constant(false))));
                    WorkResults::none()
                }
            }))
            .work(Work::consumer("load_subtile").constexpr_input("subtile_idx").returns(&t2r_rmem).run({
                let (v, base) = (t2r_rmem.clone(), tmem_base.clone());
                move |si, args| {
                    let subtile_idx = args.constexpr("subtile_idx") as i32;
                    let stage = si.stage_idx.expect("an accumulator stage");
                    let warp = make_warp_uniform(warp_idx());
                    // The warp's 32 TMEM lanes, from the subtile's first column.
                    let col = stage * bn as i32 + subtile_idx * T2R_INST_REPX as i32;
                    let ptr = get(&base) + (warp % 4) * (32 * TMEM_LANE) + col;
                    let view = make_tensor(ptr, Layout::from_modes([Layout::new((T2R_INST_REPX, 32), (1, TMEM_LANE))]));
                    let frag = make_fragment::<f32>(T2R_INST_REPX);
                    copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32)), &view, &frag);
                    fence_view_async_tmem_load();
                    WorkResults::of(&v, Vector::new(T2R_INST_REPX, frag.values()))
                }
            }))
            .work(Work::producer("mma").input::<Value<u64>>("desc_a_base").input::<Value<u64>>("desc_b_base").run({
                let (base, scale_d) = (tmem_base.clone(), scale_d.clone());
                move |si, args| {
                    let (desc_a_base, desc_b_base): (Value<u64>, Value<u64>) =
                        (args.get("desc_a_base"), args.get("desc_b_base"));
                    let stage = si.stage_idx.expect("an accumulator stage");
                    let d = get(&base).offset(stage * bn as i32);
                    let scale_d = get(&scale_d);
                    let num_k_blocks = bk / MMA_INST_SHAPE_MNK.2;
                    for k_block_idx in 0..num_k_blocks {
                        // 16 elements of K are 32 bytes: 2 in the
                        // descriptors' 16-byte start-address units.
                        let inc_bytes_per_iter = MMA_INST_SHAPE_MNK.2 * elem_bytes;
                        let increment = ((inc_bytes_per_iter * k_block_idx) >> 4) as u64;
                        let desc_a = desc_a_base + increment;
                        let desc_b = desc_b_base + increment;
                        let enable_input_d = scale_d.get();
                        if_(prims::elect_sync(u32::MAX), || {
                            prims::tcgen05_mma(
                                Tcgen05MmaKind::F16,
                                CtaGroup::ONE,
                                &d,
                                desc_a,
                                desc_b,
                                idesc,
                                enable_input_d,
                                Tcgen05MmaOptions::default(),
                            );
                        });
                        scale_d.set(Boolean::constant(true));
                    }
                    WorkResults::none()
                }
            }));

        // ---------------------------------------------------------------
        // GmemDResource: the output sink.
        // ---------------------------------------------------------------
        let gmem_d = MemoryResource::new("GmemD").work(
            Work::producer("store").input::<Vector<f32>>("t2r_rmem").constexpr_input("subtile_idx").run(
                move |_, args| {
                    let t2r_rmem: Vector<f32> = args.get("t2r_rmem");
                    let subtile_idx = args.constexpr("subtile_idx") as i32;
                    let bi = block_idx();
                    let (coordc_m, coordc_n) = (bi.x * bm as i32, bi.y * bn as i32);
                    let row = coordc_m + thread_idx().x;
                    let col = coordc_n + subtile_idx * T2R_INST_REPX as i32;
                    let (num_rows, num_cols) = (m as i32, n as i32);
                    let gc = c.ptr();
                    if_(row.lt(num_rows), || {
                        let row_offset = row * num_cols;
                        for j in 0..T2R_INST_REPX / VSIZE {
                            let chunk_col = col + (j * VSIZE) as i32;
                            let v: Vec<Value<T>> = (0..VSIZE).map(|i| t2r_rmem.get(j * VSIZE + i).to::<T>()).collect();
                            let scalar_tail = || {
                                for (i, &x) in v.iter().enumerate() {
                                    let col_idx = chunk_col + i as i32;
                                    if_(col_idx.lt(num_cols), || gc.offset(row_offset + col_idx).store(x));
                                }
                            };
                            if n.is_multiple_of(VSIZE) {
                                // Rows start 16-byte aligned: vectorise all but
                                // the ragged edge.
                                let vec: [Value<T>; VSIZE] = v.clone().try_into().expect("VSIZE values");
                                if_else(
                                    (chunk_col + VSIZE as i32).le(num_cols),
                                    || store_vector(&gc.offset(row_offset + chunk_col), vec, StoreOptions::default()),
                                    scalar_tail,
                                );
                            } else {
                                scalar_tail();
                            }
                        }
                    });
                    WorkResults::none()
                },
            ),
        );

        // ---------------------------------------------------------------
        // Allocators, tasks, graph and the TaskManager
        // (`_create_gemm_pipeline`).
        // ---------------------------------------------------------------
        let num_epilogue_warps = THREADS_IN_EPILOGUE / 32;
        let (num_load_warps, num_mma_warps) = (1, 1);
        let mut allocator = SmemAllocator::new();
        allocator.add_resource(&smem_ab);
        // cute-rs's TaskManager requires every pipeline's barriers to be
        // placed (the tutorial leaves TmemC's to the pipeline itself).
        allocator.add_resource(&tmem_c);
        allocator.compute_layout().unwrap_or_else(|e| panic!("{e}"));
        let mut tmem_allocator = TmemAllocator::new();
        tmem_allocator.add_resource(&tmem_c);
        tmem_allocator.compute_layout().unwrap_or_else(|e| panic!("{e}"));
        let total_num_warps = num_epilogue_warps + num_load_warps + num_mma_warps;
        let num_padding_warps = total_num_warps.div_ceil(4) * 4 - total_num_warps;

        let k_tiles = num_k_tiles as i64;
        let load_schedule = schedule("load_schedule", &[&gmem_ab, &smem_ab], |s, r| {
            let (gmem_ab, smem_ab) = (r[0], r[1]);
            smem_ab.call("init_load_state");
            s.domain_loop(0, k_tiles, 1, 1, |_| {
                let coords = gmem_ab.call("compute_coords");
                smem_ab.try_acquire();
                smem_ab.acquire();
                smem_ab.call_with(
                    "tma_load",
                    [
                        ("coord_k", coords.named("coord_k")),
                        ("coord_m", coords.named("coord_m")),
                        ("coord_n", coords.named("coord_n")),
                    ],
                );
                smem_ab.commit();
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let mma_schedule = schedule("mma_schedule", &[&smem_ab, &tmem_c], |s, r| {
            let (smem_ab, tmem_c) = (r[0], r[1]);
            smem_ab.call("init_descriptors");
            tmem_c.call("init_accumulator_state");
            tmem_c.call("init_work_tile_state");
            tmem_c.try_acquire();
            tmem_c.acquire();
            s.domain_loop(0, k_tiles, 1, 1, |_| {
                smem_ab.try_wait();
                smem_ab.wait();
                let descs = smem_ab.call("build_descriptors");
                tmem_c.call_with(
                    "mma",
                    [("desc_a_base", descs.named("desc_a_base")), ("desc_b_base", descs.named("desc_b_base"))],
                );
                smem_ab.release();
            });
            tmem_c.commit();
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let subtile_cnt = bn / T2R_INST_REPX;
        let store_schedule = schedule("store_schedule", &[&tmem_c, &gmem_d], |s, r| {
            let (tmem_c, gmem_d) = (r[0], r[1]);
            tmem_c.call("init_store_state");
            // Empty on purpose: the stores run in the tail, after the MMA
            // task commits the whole accumulator.
            s.domain_loop(0, k_tiles, 1, 1, |_| {});
            tmem_c.try_wait();
            tmem_c.wait();
            for subtile_idx in 0..subtile_cnt as i64 {
                let t2r_rmem =
                    tmem_c.call_with("load_subtile", [("subtile_idx", ArgSource::constexpr(subtile_idx))]).one();
                gmem_d.call_with(
                    "store",
                    [("t2r_rmem", ArgSource::from(t2r_rmem)), ("subtile_idx", ArgSource::constexpr(subtile_idx))],
                );
            }
            tmem_c.release();
        })
        .unwrap_or_else(|e| panic!("{e}"));

        let mut task_list = vec![
            Task::new("LoadTask", &[&gmem_ab], &[&smem_ab], 4, num_load_warps, load_schedule).setmaxnreg_decrease(40),
            Task::new("MmaTask", &[&smem_ab], &[&tmem_c], 5, num_mma_warps, mma_schedule).setmaxnreg_decrease(40),
            Task::new("StoreTask", &[&tmem_c], &[&gmem_d], 0, num_epilogue_warps, store_schedule)
                .setmaxnreg_decrease(160),
        ];
        if num_padding_warps > 0 {
            let padding_schedule = schedule("padding_schedule", &[], |s, _| s.domain_loop(0, k_tiles, 1, 1, |_| {}))
                .unwrap_or_else(|e| panic!("{e}"));
            task_list.push(
                Task::new("PaddingTask", &[], &[], total_num_warps, num_padding_warps, padding_schedule)
                    .setmaxnreg_decrease(40),
            );
        }
        let resource_dependency_graph = ResourceDependencyGraph::new()
            .depends_on(&smem_ab, [&gmem_ab])
            .depends_on(&tmem_c, [&smem_ab])
            .depends_on(&gmem_d, [&tmem_c]);
        let mut task_manager = TaskManager::new(task_list, resource_dependency_graph)
            .resource(gmem_ab)
            .resource(smem_ab)
            .resource(tmem_c)
            .resource(gmem_d)
            .smem_allocator(allocator)
            .tmem_allocator(tmem_allocator)
            .iket_enable_profiling(iket.is_some());
        task_manager.verify().unwrap_or_else(|e| panic!("{e}"));
        task_manager.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        task_manager.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

/// One CTA per `128 x 256` tile of D, eight warps.
pub fn launch(m: usize, n: usize) -> LaunchConfig {
    let (bm, bn, _) = MMA_TILER_MNK;
    LaunchConfig::new([m.div_ceil(bm) as u32, n.div_ceil(bn) as u32, 1], [THREADS, 1, 1])
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// `len` integers from -2 to 1 (the tutorial's `random_(-2, 2)`), exact in
/// FP16 and BF16 and in every FP32 partial sum.
pub fn small_ints(len: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..len).map(|_| (r() % 4) as f32 - 2.0).collect()
}

/// `len` values uniform in `[-1, 1)`.
pub fn uniform(len: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..len).map(|_| (r() % (1 << 24)) as f32 / (1 << 23) as f32 - 1.0).collect()
}

/// `x` rounded to `T` and back.
pub fn round<T: IoDtype>(x: f32) -> f32 {
    T::from_f32(x).to_f32()
}

/// `C = A B^T` in f64 (A `m x k`, B `n x k`, row-major), with `sum |a b|`
/// per element.
pub fn reference(av: &[f32], bv: &[f32], m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut c, mut abs) = (vec![0f64; m * n], vec![0f64; m * n]);
    for r in 0..m {
        for col in 0..n {
            for kk in 0..k {
                let p = av[r * k + kk] as f64 * bv[col * k + kk] as f64;
                c[r * n + col] += p;
                abs[r * n + col] += p.abs();
            }
        }
    }
    (c, abs)
}

/// Run the GEMM on the simulator on `av` and `bv` (already representable
/// in `T`) and return D as f32.
pub fn simulate<T: IoDtype>(kernel: &Kernel, av: &[f32], bv: &[f32], m: usize, n: usize) -> Vec<f32> {
    let to_t = |v: &[f32]| v.iter().map(|&x| T::from_f32(x)).collect::<Vec<T>>();
    let mut args = vec![
        KernelArg::buffer(&to_t(av)),
        KernelArg::buffer(&to_t(bv)),
        KernelArg::buffer(&vec![T::from_f32(0.0); m * n]),
    ];
    kernel.simulate(launch(m, n), &mut args).unwrap_or_else(|e| panic!("{e}"));
    args[2].to_vec::<T>().into_iter().map(T::to_f32).collect()
}

/// Simulate an `m x n x k` problem in `T` on small integers and check D bit
/// for bit against plain Rust.
pub fn run_exact<T: IoDtype>(m: usize, n: usize, k: usize) {
    let (av, bv) = (small_ints(m * k, 3), small_ints(n * k, 7));
    let got = simulate::<T>(&build::<T>(m, n, k, None), &av, &bv, m, n);
    let (want, _) = reference(&av, &bv, m, n, k);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(*g, round::<T>(*w as f32), "D[{}, {}] of {m}x{n}x{k} {}", i / n, i % n, T::DTYPE);
    }
}

/// Simulate an `m x n x k` problem in `T` on random data and check D within
/// the bound of an FP32 sum of `k` products rounded to `T`:
/// `|d - c| <= u_T |c| + gamma_k sum |a b|` (plus `u_T`'s share of the
/// accumulation error).
pub fn run_random<T: IoDtype>(m: usize, n: usize, k: usize) {
    let to_t = |v: Vec<f32>| v.into_iter().map(round::<T>).collect::<Vec<f32>>();
    let (av, bv) = (to_t(uniform(m * k, 11)), to_t(uniform(n * k, 13)));
    let got = simulate::<T>(&build::<T>(m, n, k, None), &av, &bv, m, n);
    let (want, abs) = reference(&av, &bv, m, n, k);
    let u_t = match T::DTYPE {
        DType::BF16 => 2f64.powi(-8),
        _ => 2f64.powi(-11),
    };
    let u32_ = 2f64.powi(-24);
    let gamma = k as f64 * u32_ / (1.0 - k as f64 * u32_);
    for (i, ((g, w), a)) in got.iter().zip(&want).zip(&abs).enumerate() {
        let acc_err = gamma * a;
        let bound = u_t * (w.abs() + acc_err) + acc_err + f64::MIN_POSITIVE;
        let err = (*g as f64 - w).abs();
        assert!(err <= bound, "D[{}, {}] of {m}x{n}x{k} {}: {g} vs {w} (bound {bound})", i / n, i % n, T::DTYPE);
    }
}

#[allow(dead_code)]
fn main() {
    let (m, n, k) = MNK;
    run_exact::<F16>(m, n, k);
    run_exact::<BF16>(m, n, k);
    println!("ts_gemm_simple: {m}x{n}x{k} FP16 and BF16 GEMMs through a TaskManager on the simulator: PASS");
}

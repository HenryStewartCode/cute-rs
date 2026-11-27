// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   07_group_pipeline_ts/01_merge_copy.py (the resources GmemAResource,
//   GmemBResource, SmemAResource, SmemBResource and StoreResource, the Merge
//   PipelineGroup `ab_sync`, the schedules load_a_schedule, load_b_schedule,
//   store_schedule and padding_schedule, the tasks and the TaskManager calls of
//   `merge_copy_kernel`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A merge copy through a pipeline group, ported from `01_merge_copy.py`:
//! two producers fill two 4-stage shared-memory resources and one consumer
//! drains both, releasing them with one arrival through a `Merge`
//! [`PipelineGroup`]:
//!
//! * `smem_a` (`AsyncAsync`): `LoadA` (warp 4) copies 128 elements of A per
//!   iteration with ordinary loads and stores;
//! * `smem_b` (`TmaAsync`, or `AsyncAsync` with `tma_for_b = false`):
//!   `LoadB` (warp 5) loads 128 elements of B with one TMA copy (or ordinary
//!   copies);
//! * `ab_sync` merges them: each keeps its full barriers, the empty barriers
//!   are shared, and `StoreTask` (warps 0-3) waits on both members, stores
//!   both tiles and releases them once (`group.release()`), every consumer
//!   thread arriving (the TMA member's consumer group counts warps, the
//!   other's threads; the group checks they agree).
//!
//! The SMEM of both members and the group's 12 barriers are laid out by the
//! declarative `SmemAllocator`. The kernel targets sm_90a (TMA), so it runs
//! on the simulator here; the tests also check its MLIR and PTX. Run with
//! `cargo run --release --example ts_merge_copy`.

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::task_scheduling::*;

/// Elements of a tile of A (`tile_m`) and of B (`tile_n`).
pub const TILE_M: usize = 128;
pub const TILE_N: usize = 128;
/// Stages of both members.
pub const NUM_STAGES: usize = 4;
pub const NUM_STORE_WARPS: u32 = 4;
/// Elements `main` and the tests copy (the tutorial's default is 1024).
pub const N_ELEMENTS: usize = 1024;

/// The merge copy of `n_elements` elements of A and of B (a multiple of the
/// tiles), B through TMA when `tma_for_b`.
pub fn build(n_elements: usize, tma_for_b: bool) -> Kernel {
    assert!(n_elements.is_multiple_of(TILE_M) && n_elements.is_multiple_of(TILE_N));
    let num_iters = (n_elements / TILE_M) as i64;
    let name = if tma_for_b { "ts_merge_copy_tma" } else { "ts_merge_copy" };
    Kernel::builder(name).block_dim([256, 1, 1]).target(Arch::Sm90a).build(move || {
        let ga = param_tensor::<F16>("a", Layout::col_major(n_elements));
        let gb = param_tensor_aligned::<F16>("b", Layout::new((1usize, n_elements), (n_elements, 1usize)), 16);
        let gda = param_tensor::<F16>("da", Layout::col_major(n_elements));
        let gdb = param_tensor::<F16>("db", Layout::col_major(n_elements));
        let tile_b = Layout::new((1usize, TILE_N), (TILE_N, 1usize));
        let tma = tma_for_b.then(|| {
            let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
            cpasync::make_tiled_tma_atom(g2s, &gb, tile_b.clone(), (1, TILE_N), 1).into()
        });
        // Prefetch B's TMA descriptor, as the tutorial does.
        if let Some((atom, _)) = &tma {
            if_(elect_one(), || cpasync::prefetch_descriptor(atom));
        }

        // GmemA / GmemB: the offsets of this iteration's tiles.
        let offset_var = |name: &str| TaskLocalVariable::<Int32>::scalar(name, 0, "the GMEM offset of the tile");
        let (off_a, off_b) = (offset_var("gmem_offset"), offset_var("gmem_offset"));
        let gmem = |name: &str, var: &TaskLocalVariable<Int32>, tile: usize| {
            let v = var.clone();
            MemoryResource::new(name).work(
                Work::consumer("compute_offset")
                    .returns(var)
                    .run(move |si, _| WorkResults::of(&v, si.loop_offset * tile as i32)),
            )
        };
        let gmem_a = gmem("gmem_a", &off_a, TILE_M);
        let gmem_b = gmem("gmem_b", &off_b, TILE_N);

        // SmemA: ordinary copies, AsyncAsync.
        let elem = 2;
        let alloc_a = SmemAllocation::bytes("smem_a", TILE_M * NUM_STAGES * elem);
        let cfg_a = PipelineConfig::create_async_async_pipeline_cfg(
            NUM_STAGES,
            CooperativeGroup::new(Agent::Thread, 32),
            CooperativeGroup::new(Agent::Thread, NUM_STORE_WARPS * 32),
            Some(Layout::col_major((1usize, 1usize, 1usize, 1usize))),
        );
        let nothing = |_: &StageInfo, _: &WorkArgs| WorkResults::none();
        let smem_a = MemoryResource::new("smem_a")
            .with_pipeline_config(cfg_a)
            .smem_requirement(alloc_a.clone())
            .work(Work::producer("init_load_state").auxiliary().run(nothing))
            .work(Work::consumer("init_store_state").auxiliary().run(nothing))
            .work(Work::consumer("consume_tile").run(nothing))
            .work(Work::producer("load").input::<Int32>("gmem_offset").run({
                let alloc = alloc_a.clone();
                move |si, args| {
                    let off: Int32 = args.get("gmem_offset");
                    let s = alloc.ptr::<F16>(&si.context).offset(si.stage_idx.expect("staged") * TILE_M as i32);
                    let per_thread = (TILE_M / 32) as i32;
                    for i in 0..per_thread {
                        let idx = lane_idx() * per_thread + i;
                        s.offset(idx).store(ga.load(off + idx));
                    }
                    WorkResults::none()
                }
            }));

        // SmemB: one TMA copy per tile (or ordinary copies).
        let alloc_b = SmemAllocation::bytes("smem_b", TILE_N * NUM_STAGES * elem);
        let cfg_b = if tma_for_b {
            PipelineConfig::create_tma_async_pipeline_cfg(
                NUM_STAGES,
                (TILE_N * elem) as u32,
                CooperativeGroup::new(Agent::Thread, 1),
                CooperativeGroup::new(Agent::Thread, NUM_STORE_WARPS),
            )
        } else {
            PipelineConfig::create_async_async_pipeline_cfg(
                NUM_STAGES,
                CooperativeGroup::new(Agent::Thread, 32),
                CooperativeGroup::new(Agent::Thread, NUM_STORE_WARPS * 32),
                Some(Layout::col_major((1usize, 1usize, 1usize, 1usize))),
            )
        };
        let smem_b = MemoryResource::new("smem_b")
            .with_pipeline_config(cfg_b)
            .smem_requirement(alloc_b.clone())
            .work(Work::producer("init_load_state").auxiliary().run(nothing))
            .work(Work::consumer("init_store_state").auxiliary().run(nothing))
            .work(Work::consumer("consume_tile").run(nothing))
            .work(Work::producer("load").input::<Int32>("gmem_offset").run({
                let alloc = alloc_b.clone();
                move |si, args| {
                    let off: Int32 = args.get("gmem_offset");
                    let s = alloc.ptr::<F16>(&si.context).offset(si.stage_idx.expect("staged") * TILE_N as i32);
                    match &tma {
                        Some((atom, tma_b)) => {
                            let dst = make_tensor(&s, tile_b.clone());
                            let g = local_tile(tma_b, (1, TILE_N), (0, off / TILE_N as i32));
                            let (ts, tg) = cpasync::tma_partition(
                                atom,
                                0,
                                &make_layout(1, 0),
                                &dst.group_modes(0, 2),
                                &g.group_modes(0, 2),
                            );
                            let bar = si.barrier().expect("the stage's full barrier");
                            if_(elect_one(), || copy_with(atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&bar)));
                        }
                        None => {
                            let per_thread = (TILE_N / 32) as i32;
                            for i in 0..per_thread {
                                let idx = lane_idx() * per_thread + i;
                                s.offset(idx).store(gb.load((0, off + idx)));
                            }
                        }
                    }
                    WorkResults::none()
                }
            }));

        // ab_sync: one release covers both members' (shared) empty barriers.
        let ab_sync = PipelineGroup::new("ab_sync", &[&smem_a, &smem_b], PipelineGroupMode::Merge);

        // StoreResource reads both members' current stage.
        let store_res = MemoryResource::new("store_res").work(Work::producer("store").run(move |si, _| {
            let tx = thread_idx().x;
            let sidx = si.loop_offset % NUM_STAGES as i32;
            let sa = alloc_a.ptr::<F16>(&si.context).offset(sidx * TILE_M as i32);
            let sb = alloc_b.ptr::<F16>(&si.context).offset(sidx * TILE_N as i32);
            let total = (NUM_STORE_WARPS * 32) as i32;
            for i in 0..(TILE_M as i32 / total) {
                let idx = tx + i * total;
                gda.store(si.loop_offset * TILE_M as i32 + idx, sa.offset(idx).load());
            }
            for i in 0..(TILE_N as i32 / total) {
                let idx = tx + i * total;
                gdb.store(si.loop_offset * TILE_N as i32 + idx, sb.offset(idx).load());
            }
            WorkResults::none()
        }));

        let mut allocator = SmemAllocator::new();
        allocator.add_resource(&smem_a);
        allocator.add_resource(&smem_b);
        allocator.add_pipeline_group(&ab_sync);

        let load = |name: &str, gmem: &MemoryResource, smem: &MemoryResource| {
            schedule(name, &[gmem, smem], |s, r| {
                let (gmem, smem) = (r[0], r[1]);
                smem.call("init_load_state");
                s.domain_loop(0, num_iters, 1, 1, |_| {
                    let gmem_offset = gmem.call("compute_offset").one();
                    smem.try_acquire();
                    smem.acquire();
                    smem.call_with("load", [("gmem_offset", &gmem_offset)]);
                    smem.commit();
                });
            })
            .unwrap_or_else(|e| panic!("{e}"))
        };
        // The register budgets are the tutorial's `num_registers` (warpgroup 1 shares one count).
        let load_a_task = Task::new("LoadA", &[&gmem_a], &[&smem_a], 4, 1, load("load_a_schedule", &gmem_a, &smem_a))
            .setmaxnreg_decrease(40);
        let load_b_task = Task::new("LoadB", &[&gmem_b], &[&smem_b], 5, 1, load("load_b_schedule", &gmem_b, &smem_b))
            .setmaxnreg_decrease(40);
        let store_schedule = schedule("store_schedule", &[&smem_a, &smem_b, &store_res, &ab_sync], |s, r| {
            let (smem_a_ref, smem_b_ref, store, group) = (r[0], r[1], r[2], r[3]);
            smem_a_ref.call("init_store_state");
            smem_b_ref.call("init_store_state");
            s.domain_loop(0, num_iters, 1, 1, |_| {
                smem_a_ref.try_wait();
                smem_a_ref.wait();
                smem_b_ref.try_wait();
                smem_b_ref.wait();
                smem_a_ref.call("consume_tile");
                smem_b_ref.call("consume_tile");
                store.call("store");
                group.release();
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let store_task = Task::new("StoreTask", &[&smem_a, &smem_b], &[&store_res], 0, NUM_STORE_WARPS, store_schedule)
            .setmaxnreg_decrease(160);
        let padding_schedule = schedule("padding_schedule", &[], |s, _| s.domain_loop(0, num_iters, 1, 1, |_| {}))
            .unwrap_or_else(|e| panic!("{e}"));
        let padding_task = Task::new("PaddingTask", &[], &[], 6, 2, padding_schedule).setmaxnreg_decrease(40);

        let resource_dependency_graph = ResourceDependencyGraph::new()
            .depends_on(&smem_a, [&gmem_a])
            .depends_on(&smem_b, [&gmem_b])
            .depends_on(&store_res, [&smem_a, &smem_b]);
        allocator.compute_layout().unwrap_or_else(|e| panic!("{e}"));
        let mut task_manager =
            TaskManager::new(vec![load_a_task, load_b_task, store_task, padding_task], resource_dependency_graph)
                .resource(gmem_a)
                .resource(gmem_b)
                .resource(smem_a)
                .resource(smem_b)
                .resource(ab_sync)
                .resource(store_res)
                .smem_allocator(allocator);
        task_manager.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        task_manager.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

/// One CTA of 8 warps.
pub fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [256, 1, 1])
}

/// The inputs A and B: distinct small values, exact in f16.
pub fn inputs(n: usize) -> (Vec<F16>, Vec<F16>) {
    let a = (0..n).map(|i| F16::from_f32((i % 1000) as f32 - 500.0)).collect();
    let b = (0..n).map(|i| F16::from_f32(((i * 37) % 999) as f32)).collect();
    (a, b)
}

/// The kernel's arguments for `n` elements.
pub fn args(n: usize) -> Vec<KernelArg> {
    let (a, b) = inputs(n);
    let zero = vec![F16::from_f32(0.0); n];
    vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&zero), KernelArg::buffer(&zero)]
}

/// Copy on the simulator and check both outputs against plain Rust (DA ==
/// A, DB == B).
pub fn run(n: usize, tma_for_b: bool) {
    let mut a = args(n);
    build(n, tma_for_b).simulate(launch(), &mut a).unwrap_or_else(|e| panic!("{e}"));
    let (ra, rb) = inputs(n);
    assert_eq!(a[2].to_vec::<F16>(), ra);
    assert_eq!(a[3].to_vec::<F16>(), rb);
}

fn main() {
    run(N_ELEMENTS, true);
    run(N_ELEMENTS, false);
    println!("ts_merge_copy: {N_ELEMENTS} elements of A and B through a Merge group (TMA and plain B): PASS");
}

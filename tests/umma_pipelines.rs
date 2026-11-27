// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The warp-specialised mainloop's structure (a TMA warp feeding an MMA warp through a
// PipelineTmaUmma, the MMA warp handing accumulator stages in TMEM to four epilogue warps
// through a PipelineUmmaAsync, and the TmemAllocator's allocate / wait_for_alloc /
// retrieve_ptr / relinquish_alloc_permit / free sequence) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The Blackwell pipelines and the TMEM allocator on the
//! simulator: a warp-specialised 1-CTA GEMM (`PipelineTmaUmma` with 4 stages of TMA loads,
//! `PipelineUmmaAsync` with 2 accumulator stages, `TmemAllocator` and `TmemBufferPool`)
//! and its `PipelineAsyncUmma` variant with `cp.async` producer warps, both against plain
//! Rust; the 2-CTA allocator's dealloc handshake; the simulator's reports for the
//! pipelines' misuses; the column math and TMEM copy introspection; and the sm_100a PTX.

use cute::memory::{
    TmemAllocator, TmemAllocatorOptions, TmemBufferPool, compute_tmem_cols_from_layout, get_num_tmem_alloc_cols,
};
use cute::pipeline::{
    Agent, CooperativeGroup, CtaGroup as PipeCtaGroup, NamedBarrier, Pipeline, PipelineAsyncUmma, PipelineBarriers,
    PipelineProducer, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions,
    PipelineUserType, agent_sync, make_pipeline_state,
};
use cute::prelude::*;
use cute::tcgen05::{
    self, CtaGroup, Field, Ld16x64bOp, Ld32x32bOp, MmaF16BF16Op, OperandSource, Pack, Repetition, SmemLayoutAtomKind,
    St16x32bx2Op, St32x32bOp, TmemPacking, Unpack, make_tmem_copy,
};

/// Rows of C (one 128-row MMA tile).
const M: usize = 128;
/// Columns of an output tile.
const NT: usize = 64;
/// Output tiles along N: two passes over the accumulator stages.
const N_TILES: usize = 4;
/// K of a k-tile (four k-blocks of 16).
const KT: usize = 64;
/// k-tiles per output tile.
const K_TILES: usize = 3;
const N: usize = NT * N_TILES;
const K: usize = KT * K_TILES;
/// Stages of the A/B pipeline.
const AB_STAGES: usize = 4;
/// Stages of the accumulator pipeline.
const ACC_STAGES: usize = 2;
/// Epilogue threads (warps 0 to 3), the MMA warp (4), then the producers.
const EPI: usize = 128;
const MMA_WARP: i32 = 4;

/// How the stages are filled.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Producer {
    /// One TMA warp (warp 5), through `PipelineTmaUmma`.
    Tma,
    /// Two `cp.async` warps (5 and 6), through `PipelineAsyncUmma`.
    CpAsync,
}

impl Producer {
    fn threads(self) -> usize {
        match self {
            Producer::Tma => EPI + 64,
            Producer::CpAsync => EPI + 96,
        }
    }
}

/// What a kernel gets wrong on purpose.
#[derive(Clone, Copy, Debug, Default)]
struct Faults {
    no_ab_release: bool,
    wrong_initial_phase: bool,
    small_tx_count: bool,
    commit_without_elect: bool,
    read_before_wait: bool,
    release_before_fence: bool,
    no_free: bool,
    alloc_after_relinquish: bool,
    no_proxy_fence: bool,
}

fn umma_op() -> TiledMma {
    let op = MmaF16BF16Op::new(
        DType::F16,
        DType::F32,
        (M, NT),
        CtaGroup::ONE,
        OperandSource::SMEM,
        OperandMajorMode::K,
        OperandMajorMode::K,
    );
    make_tiled_mma(op, Layout::col_major((1, 1, 1)))
}

fn staged(mn: usize) -> ComposedLayout {
    let atom = tcgen05::make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
    tile_to_shape_swizzled(&atom, (mn, KT, AB_STAGES))
}

/// The MMA warp's side: for each output tile, acquire an accumulator stage, wait for each
/// k-tile, issue its four k-blocks and release the stage with a commit, then commit the
/// accumulator.
fn mma_warp<P>(
    ab: &P,
    acc_pipe: &PipelineUmmaAsync,
    mma: &TiledMma,
    sa: &Tensor<F16>,
    sb: &Tensor<F16>,
    acc0: &Tensor<f32>,
    stage_cols: usize,
    faults: Faults,
) where
    P: PipelineBarriers,
{
    let thr = mma.get_slice(0);
    let (_, ab_c) = ab.make_participants();
    let (acc_p, _) = acc_pipe.make_participants();
    let (_, mut acc_p) = for_range_carry(0, N_TILES as i32, 1, (ab_c, acc_p), |_, (mut abc, mut accp)| {
        let acc_h = accp.acquire_and_advance(None);
        let acc = make_tensor(acc0.ptr().offset(acc_h.index() * stage_cols as i32), acc0.layout().clone());
        let mut m = mma.clone();
        m.set(Field::ACCUMULATE, false);
        for _ in 0..K_TILES {
            let h = abc.wait_and_advance(None);
            let fa = mma.make_fragment_a(&thr.partition_a(&sa.slice((.., .., h.index()))));
            let fb = mma.make_fragment_b(&thr.partition_b(&sb.slice((.., .., h.index()))));
            for kb in 0..KT / 16 {
                gemm(&m, &acc, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), &acc);
                m.set(Field::ACCUMULATE, true);
            }
            if faults.commit_without_elect {
                tcgen05::commit(&h.barrier(), None, CtaGroup::ONE);
            } else if !faults.no_ab_release {
                h.release();
            }
        }
        acc_h.commit();
        (abc, accp)
    });
    acc_p.tail();
}

/// The epilogue warps' side: wait for each accumulator stage, load it from TMEM
/// (`tcgen05.ld.32x32b.x64`), wait for the loads, release the stage and store the tile.
fn epilogue(acc_pipe: &PipelineUmmaAsync, c: &Tensor<f32>, acc0: &Tensor<f32>, stage_cols: usize, faults: Faults) {
    let t = thread_idx().x;
    let (_, acc_c) = acc_pipe.make_participants();
    for_range_carry(0, N_TILES as i32, 1, acc_c, |n, mut cc| {
        let h = if faults.read_before_wait {
            let h = cc.current_handle();
            cc.advance(1);
            h
        } else {
            cc.wait_and_advance(None)
        };
        let acc = make_tensor(acc0.ptr().offset(h.index() * stage_cols as i32), acc0.layout().clone());
        let block = acc.layout().mode(0);
        let tile = make_tensor(acc.ptr(), Layout::from_modes([block.mode(0), block.mode(1)]));
        let ctile = local_tile(c, (M, NT), (0, n));
        let tiled = make_tmem_copy(&make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X64)), &tile);
        let thr = tiled.get_slice(t);
        let (src, dst) = (thr.partition_s(&tile), thr.partition_d(&ctile));
        let frag = make_fragment_like::<f32, _>(&dst);
        copy(&tiled, &src, &frag);
        if faults.release_before_fence {
            h.release();
            fence_view_async_tmem_load();
        } else {
            fence_view_async_tmem_load();
            h.release();
        }
        basic_copy(&frag, &dst);
        cc
    });
}

/// The warp-specialised GEMM `C = A B^T` (A `(M, K)`, B `(N, K)`, both K-major f16; C
/// `(M, N)` row-major f32) in output tiles of `(M, NT)`.
fn gemm_kernel(name: &'static str, producer: Producer, faults: Faults) -> Kernel {
    let threads = producer.threads();
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).target(Arch::Sm100a).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((M, K), (K, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((N, K), (K, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((M, N), (N, 1)));
        let mma = umma_op();
        let (a_staged, b_staged) = (staged(M), staged(NT));
        let tma = (producer == Producer::Tma).then(|| {
            let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
            let (atom_a, tma_a) = cpasync::make_tiled_tma_atom(g2s, &a, a_staged.clone(), (M, KT), 1).into();
            let (atom_b, tma_b) = cpasync::make_tiled_tma_atom(g2s, &b, b_staged.clone(), (NT, KT), 1).into();
            (atom_a, tma_a, atom_b, tma_b)
        });
        let sa = make_smem_tensor_aligned::<F16>(a_staged, 1024);
        let sb = make_smem_tensor_aligned::<F16>(b_staged, 1024);

        // TMEM: two accumulator stages, allocated by warp 0; the MMA and epilogue warps
        // (160 threads) wait on named barrier 1 for the address.
        let acc_layout = mma.make_fragment_c::<f32>(&mma.partition_shape_c((M, NT))).layout().clone();
        let stage_cols = compute_tmem_cols_from_layout(&acc_layout, DType::F32);
        let cols = ACC_STAGES * stage_cols;
        let mut tmem = TmemAllocator::new(
            NamedBarrier::new(1, EPI + 32),
            TmemAllocatorOptions { arch: Arch::Sm100a, ..Default::default() },
        );
        if faults.alloc_after_relinquish {
            tmem.relinquish_alloc_permit();
        }
        tmem.allocate(cols);

        let one = CooperativeGroup::new(Agent::Thread, 1);
        let acc_pipe = PipelineUmmaAsync::create(
            ACC_STAGES,
            one,
            CooperativeGroup::new(Agent::Thread, EPI as i32),
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        );
        let tx = (M + NT) * KT * 2 - if faults.small_tx_count { 128 } else { 0 };
        let acc_view = |tmem: &TmemAllocator| {
            tmem.wait_for_alloc();
            let mut pool = TmemBufferPool::new(&tmem.retrieve_ptr::<f32>(), cols);
            let acc0 = pool.allocate_tensor::<f32>(&acc_layout);
            let _acc1 = pool.allocate::<f32>(&acc_layout);
            assert_eq!(pool.remaining_cols(), 0);
            acc0
        };
        match producer {
            Producer::Tma => {
                let (atom_a, tma_a, atom_b, tma_b) = tma.unwrap();
                let ab = PipelineTmaUmma::create(AB_STAGES, one, one, tx, PipelineTmaUmmaOptions::default());
                let ga = local_tile(&tma_a, (M, KT), (0, ..));
                let gb = local_tile(&tma_b, (NT, KT), (.., ..));
                let lay = make_layout(1, 0);
                let (ts_a, tg_a) =
                    cpasync::tma_partition(&atom_a, 0, &lay, &sa.group_modes(0, 2), &ga.group_modes(0, 2));
                let (ts_b, tg_b) =
                    cpasync::tma_partition(&atom_b, 0, &lay, &sb.group_modes(0, 2), &gb.group_modes(0, 2));
                if_(warp_idx().eq(5), || {
                    let producer = if faults.wrong_initial_phase {
                        PipelineProducer::new(&ab, make_pipeline_state(PipelineUserType::Consumer, AB_STAGES), None)
                    } else {
                        ab.make_producer()
                    };
                    let mut p = for_range_carry(0, N_TILES as i32, 1, producer, |n, mut p| {
                        for kt in 0..K_TILES {
                            let h = p.acquire_and_advance(None);
                            let bar = CopyArgs::new().tma_bar_ptr(&h.barrier());
                            copy_with(&atom_a, &tg_a.slice((.., kt)), &ts_a.slice((.., h.index())), bar.clone());
                            copy_with(&atom_b, &tg_b.slice((.., n, kt)), &ts_b.slice((.., h.index())), bar);
                            h.commit();
                        }
                        p
                    });
                    p.tail();
                });
                if_(warp_idx().eq(MMA_WARP), || {
                    let acc0 = acc_view(&tmem);
                    mma_warp(&ab, &acc_pipe, &mma, &sa, &sb, &acc0, stage_cols, faults);
                });
            }
            Producer::CpAsync => {
                let ab = PipelineAsyncUmma::create(
                    AB_STAGES,
                    CooperativeGroup::new(Agent::Thread, 64),
                    one,
                    PipelineUmmaOptions { defer_sync: true, ..Default::default() },
                );
                sync_threads();
                if_(warp_idx().ge(5), || {
                    let pt = thread_idx().x - (EPI + 32) as i32;
                    let atom = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
                    let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 8)), &Layout::row_major((1, 8)));
                    let thr = tiled.get_slice(pt);
                    let ga = local_tile(&a, (M, KT), (0, ..));
                    let gb = local_tile(&b, (NT, KT), (.., ..));
                    let (ta_g, ta_s) = (thr.partition_s(&ga), thr.partition_d(&sa));
                    let (tb_g, tb_s) = (thr.partition_s(&gb), thr.partition_d(&sb));
                    let mut p = for_range_carry(0, N_TILES as i32, 1, ab.make_producer(), |n, mut p| {
                        for kt in 0..K_TILES {
                            let h = p.acquire_and_advance(None);
                            copy(&tiled, &ta_g.slice((.., .., .., kt)), &ta_s.slice((.., .., .., h.index())));
                            copy(&tiled, &tb_g.slice((.., .., .., n, kt)), &tb_s.slice((.., .., .., h.index())));
                            cp_async_commit_group();
                            cp_async_wait_group(0);
                            if !faults.no_proxy_fence {
                                fence_view_async_shared();
                            }
                            h.commit();
                        }
                        p
                    });
                    p.tail();
                });
                if_(warp_idx().eq(MMA_WARP), || {
                    let acc0 = acc_view(&tmem);
                    mma_warp(&ab, &acc_pipe, &mma, &sa, &sb, &acc0, stage_cols, faults);
                });
            }
        }
        if_(warp_idx().lt(4), || {
            let acc0 = acc_view(&tmem);
            epilogue(&acc_pipe, &c, &acc0, stage_cols, faults);
            tmem.relinquish_alloc_permit();
            NamedBarrier::new(2, EPI).arrive_and_wait();
            if !faults.no_free {
                tmem.free(&acc0.ptr(), 0);
            }
        });
    })
}

/// Small integers from -4 to 4, the same on every run.
fn small_ints(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) % 9) as f32 - 4.0
        })
        .collect()
}

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<KernelArg>) {
    let (av, bv) = (small_ints(M * K, 11), small_ints(N * K, 13));
    let a: Vec<F16> = av.iter().map(|&x| F16::from_f32(x)).collect();
    let b: Vec<F16> = bv.iter().map(|&x| F16::from_f32(x)).collect();
    let args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; M * N])];
    (av, bv, args)
}

/// `C = A B^T` in plain Rust (exact: small integers).
fn reference(av: &[f32], bv: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; M * N];
    for r in 0..M {
        for col in 0..N {
            out[r * N + col] = (0..K).map(|k| av[r * K + k] * bv[col * K + k]).sum();
        }
    }
    out
}

fn cfg(producer: Producer) -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [producer.threads() as u32, 1, 1])
}

fn run(name: &'static str, producer: Producer) {
    let kernel = gemm_kernel(name, producer, Faults::default());
    let (av, bv, mut args) = inputs();
    let report = kernel
        .simulate_all_orders(cfg(producer), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.tcgen05_mma_instructions, (N_TILES * K_TILES * KT / 16) as u64, "{name}");
    assert_eq!(args[2].to_vec::<f32>(), reference(&av, &bv), "{name}");
}

/// (a) A TMA warp, an MMA warp and four epilogue warps through `PipelineTmaUmma` (4
/// stages, the ring passing three times) and `PipelineUmmaAsync` (2 accumulator stages,
/// twice), with the `TmemAllocator`: C matches plain Rust under every built-in schedule.
#[test]
fn tma_umma_mainloop_matches_rust() {
    run("umma_pipe_tma", Producer::Tma);
}

/// (b) `PipelineAsyncUmma` with two `cp.async` producer warps.
#[test]
fn async_umma_cp_async_producers_match_rust() {
    run("umma_pipe_cp_async", Producer::CpAsync);
}

/// The simulator's report for the GEMM with `faults`.
fn report_for(producer: Producer, faults: Faults) -> String {
    let kernel = gemm_kernel("umma_pipe_misuse", producer, faults);
    let (_, _, mut args) = inputs();
    match kernel.simulate(cfg(producer), &mut args) {
        Ok(_) => panic!("expected a simulator report for {faults:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn pipeline_misuses_are_reported() {
    let tma = |f: Faults| report_for(Producer::Tma, f);
    let e = tma(Faults { no_ab_release: true, ..Default::default() });
    assert!(e.contains("deadlock"), "missing release: {e}");
    let e = tma(Faults { wrong_initial_phase: true, ..Default::default() });
    assert!(e.contains("deadlock"), "wrong phase: {e}");
    let e = tma(Faults { small_tx_count: true, ..Default::default() });
    assert!(e.contains("transaction count"), "small tx_count: {e}");
    let e = tma(Faults { commit_without_elect: true, ..Default::default() });
    assert!(e.contains("elect_one"), "commit without elect_one: {e}");
    let e = tma(Faults { read_before_wait: true, ..Default::default() });
    assert!(
        e.contains("has not observed the MMA's completion") || e.contains("nothing wrote") || e.contains("in flight"),
        "TMEM read before consumer_wait: {e}"
    );
    let e = tma(Faults { release_before_fence: true, ..Default::default() });
    assert!(e.contains("before releasing the TMEM"), "release before fence_view_async_tmem_load: {e}");
    let e = tma(Faults { no_free: true, ..Default::default() });
    assert!(e.contains("freed") || e.contains("dealloc"), "missing free: {e}");
    let e = tma(Faults { alloc_after_relinquish: true, ..Default::default() });
    assert!(e.contains("relinquish"), "allocate after relinquish: {e}");
    let e = report_for(Producer::CpAsync, Faults { no_proxy_fence: true, ..Default::default() });
    assert!(e.contains("fence.proxy.async"), "no proxy fence: {e}");
}

/// (c) The allocator in 2-CTA mode on a `(2, 1, 1)` cluster: each CTA's warp 0 allocates
/// for the pair (`reserve`), both read the same columns, and `free` runs the dealloc
/// handshake on the peers' mbarriers before `tcgen05.dealloc.cta_group::2`.
fn two_cta_kernel(skip_free_on_rank_1: bool) -> Kernel {
    Kernel::builder("tmem_two_cta").block_dim([64, 1, 1]).target(Arch::Sm100a).build(move || {
        let out = param_tensor::<i32>("out", Layout::col_major(2));
        let mut tmem = TmemAllocator::new(
            NamedBarrier::new(1, 64),
            TmemAllocatorOptions { is_two_cta: true, arch: Arch::Sm100a, ..Default::default() },
        );
        assert!(tmem.two_cta_tmem_dealloc_mbar_ptr().is_some());
        agent_sync(Agent::ThreadBlockCluster, true);
        let pool = tmem.reserve(64);
        let rank = block_idx_in_cluster();
        if_(thread_idx().x.eq(0), || out.store(rank, pool.base_ptr().toint_i32()));
        // Every thread is done with the columns before the allocator warp frees them.
        sync_threads();
        tmem.relinquish_alloc_permit();
        let base = pool.base_ptr().clone();
        if skip_free_on_rank_1 {
            if_(rank.eq(0), || tmem.clone().free(&base, 0));
        } else {
            tmem.free(&base, 0);
        }
    })
}

#[test]
fn two_cta_allocator_frees_after_the_handshake() {
    let cfg = LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    let kernel = two_cta_kernel(false);
    let ir = kernel.to_mlir();
    assert!(ir.contains("tcgen05_group<cta_2>"), "{ir}");
    let mut args = vec![KernelArg::buffer(&[-1i32; 2])];
    kernel.simulate_all_orders(cfg, &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    let got = args[0].to_vec::<i32>();
    assert_eq!(got[0], got[1], "both CTAs of the pair see the same columns");
    // One CTA never frees: the other waits on its dealloc mbarrier for ever, or (by
    // schedule) the first CTA's exit with the pair's columns allocated is reported.
    let mut args = vec![KernelArg::buffer(&[-1i32; 2])];
    let e = two_cta_kernel(true).simulate(cfg, &mut args).unwrap_err().to_string();
    assert!(e.contains("deadlock") || e.contains("still allocated"), "{e}");
}

/// The column math, the pool's bookkeeping, the allocator's column checks and the TMEM
/// copy introspection, on the host (inside a kernel being built where a TMEM tensor is
/// needed).
#[test]
fn column_math_pool_and_copy_properties() {
    // 128 x 64 f32 accumulator columns, recast through 32-bit cells; 16-bit and Boolean.
    let acc = Layout::new((128, 64), (65536, 1));
    assert_eq!(compute_tmem_cols_from_layout(&acc, DType::F32), 64);
    // A 16-bit (8-bit) TMEM tensor's lane stride is 2^16 cells of 2 (4) elements.
    assert_eq!(compute_tmem_cols_from_layout(&Layout::new((128, 64), (131072, 1)), DType::F16), 32);
    assert_eq!(compute_tmem_cols_from_layout(&Layout::new((128, 64), (262144, 1)), DType::Bool), 16);
    Kernel::builder("tmem_cols").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let slot = alloc_smem::<u32>(1, 4);
        let base = retrieve_tmem_ptr::<f32>(16, &slot);
        let t1 = make_tensor(base.clone(), acc.clone());
        let t2 = make_tensor(base.clone(), Layout::new((128, 40), (65536, 1)));
        let t3 = make_tensor(recast_ptr::<F16, f32>(&base, None), Layout::new((128, 64), (131072, 1)));
        assert_eq!(get_num_tmem_alloc_cols(&[&t1], true, Arch::Sm100), 64);
        assert_eq!(get_num_tmem_alloc_cols(&[&t1, &t2], false, Arch::Sm100), 104);
        assert_eq!(get_num_tmem_alloc_cols(&[&t1, &t2], true, Arch::Sm100), 128);
        assert_eq!(get_num_tmem_alloc_cols(&[&t3], true, Arch::Sm100), 32);
        assert_eq!(
            get_num_tmem_alloc_cols(
                &[&make_tensor(base.clone(), Layout::new((128, 8), (65536, 1)))],
                true,
                Arch::Sm100
            ),
            32
        );
        // Tests the deprecated names, which still have to work.
        #[allow(deprecated)]
        let old = cute::utils::get_num_tmem_alloc_cols(&[&t1, &t2], true);
        assert_eq!(old, 128);
        assert_eq!(tcgen05::find_tmem_tensor_col_offset(&t2), 40);
        let big = make_tensor(base.clone(), Layout::new((128, 576), (65536, 1)));
        let e = panic_text(|| {
            get_num_tmem_alloc_cols(&[&big], true, Arch::Sm100);
        });
        assert!(e.contains("1024"), "{e}");
        assert_eq!(get_num_tmem_alloc_cols(&[&big], false, Arch::Sm107a), 576);

        // The pool: offsets are build-time columns.
        let mut pool = TmemBufferPool::new(&base, 128);
        let p0 = pool.allocate::<f32>(64usize);
        let t = pool.allocate_tensor::<F16>(&Layout::new((128, 32), (131072, 1)));
        assert_eq!((pool.current_offset(), pool.remaining_cols(), pool.total_cols()), (80, 48, 128));
        assert_eq!(p0.memspace(), AddressSpace::Tmem);
        assert_eq!(t.ptr().memspace(), AddressSpace::Tmem);
        let e = panic_text(move || {
            pool.allocate::<f32>(64usize);
        });
        assert!(e.contains("64 columns asked for, 48 left"), "{e}");

        // The allocator's column checks.
        let alloc =
            |arch| TmemAllocator::new(NamedBarrier::new(1, 128), TmemAllocatorOptions { arch, ..Default::default() });
        let a100 = alloc(Arch::Sm100);
        assert!(a100.check_valid_num_columns(32) && a100.check_valid_num_columns(512));
        assert!(!a100.check_valid_num_columns(96) && !a100.check_valid_num_columns(48));
        assert!(!a100.check_valid_num_columns(576) && !a100.check_valid_num_columns(1024));
        let a107 = alloc(Arch::Sm107a);
        assert!(a107.check_valid_num_columns(544) && a107.check_valid_num_columns(576));
        assert!(!a107.check_valid_num_columns(96) && !a107.check_valid_num_columns(608));
        let e = panic_text(|| alloc(Arch::Sm100).allocate(96));
        assert!(e.contains("96 columns"), "{e}");
        let mut a = alloc(Arch::Sm100);
        a.allocate(256);
        a.allocate(256);
        assert_eq!(a.num_allocated_columns(), 512);
        let e = panic_text(move || a.allocate(32));
        assert!(e.contains("544 columns allocated in all"), "{e}");
    });

    // TMEM copy introspection.
    let ld = make_copy_atom::<F16>(Ld16x64bOp::new(Repetition::X8).with_pack(Pack::Pack16bIn32b));
    assert!(tcgen05::is_tmem_load(&ld) && !tcgen05::is_tmem_store(&ld));
    assert_eq!(tcgen05::get_tmem_copy_properties(&ld), (16, 64, 8, TmemPacking::Pack(Pack::Pack16bIn32b)));
    let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X128));
    assert_eq!(tcgen05::get_tmem_copy_properties(&ld), (32, 32, 128, TmemPacking::Pack(Pack::None)));
    let st = make_copy_atom::<F16>(St16x32bx2Op::new(Repetition::X2).with_unpack(Unpack::Unpack32bIn16b));
    assert!(tcgen05::is_tmem_store(&st) && !tcgen05::is_tmem_load(&st));
    assert_eq!(tcgen05::get_tmem_copy_properties(&st), (16, 32, 2, TmemPacking::Unpack(Unpack::Unpack32bIn16b)));
    let st = make_copy_atom::<f32>(St32x32bOp::new(Repetition::X1));
    assert_eq!(st.op().tmem_copy_properties(), Some((32, 32, 1, TmemPacking::Unpack(Unpack::None))));
    let other = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    assert!(!tcgen05::is_tmem_load(&other) && !tcgen05::is_tmem_store(&other));
    let e = panic_text(move || {
        tcgen05::get_tmem_copy_properties(&other);
    });
    assert!(e.contains("not a TMEM load or store"), "{e}");
}

fn panic_text(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// The pipelines' build-time properties: the CTA group from the cluster's V mode, the
/// leader flag, the per-acquire byte count, and what each side emits.
#[test]
fn pipeline_properties_and_ops() {
    use cute::ir::Op;
    let kernel = Kernel::builder("umma_pipe_props").block_dim([64, 1, 1]).target(Arch::Sm100a).build(|| {
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let p = PipelineTmaUmma::create(2, one, one, 4096, PipelineTmaUmmaOptions::default());
        assert_eq!(p.cta_group(), PipeCtaGroup::One);
        assert_eq!(p.is_leader_cta().as_constant(), Some(true));
        assert!(p.consumer_mask().is_none());
        assert_eq!(p.sync_object_full().tx_count().as_static(), Some(4096));
        let (prod, cons) = p.make_participants();
        // A per-acquire byte count, then a release (a commit).
        let h = prod.acquire_with(None, Some(Int32::from(1024)));
        h.commit();
        cons.wait(None).release();
        let q = PipelineUmmaAsync::create(
            2,
            one,
            CooperativeGroup::new(Agent::Thread, 64),
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(q.cta_group(), PipeCtaGroup::One);
        q.make_producer().acquire(None).commit();
        let r = PipelineAsyncUmma::create(
            2,
            CooperativeGroup::new(Agent::Warp, 1),
            one,
            PipelineUmmaOptions { defer_sync: true, name: "ab".into(), ..Default::default() },
        );
        assert_eq!(r.sync_object_full().name(), "ab.full");
        assert_eq!(r.sync_object_full().arrive_count().as_static(), Some(32));
    });
    let ir = kernel.ir();
    // Two tcgen05.commits: PipelineTmaUmma's release and PipelineUmmaAsync's commit.
    assert_eq!(ir.count_ops(|op| matches!(op, Op::Tcgen05Commit { .. })), 2);
    let text = kernel.to_mlir();
    assert!(text.contains("1024"), "{text}");
    // A V = 2 cluster makes the commits cta_group::2.
    Kernel::builder("umma_pipe_v2").block_dim([64, 1, 1]).target(Arch::Sm100a).build(|| {
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let vmnk = Layout::col_major((2usize, 1usize, 1usize, 1usize));
        let o = PipelineUmmaOptions { cta_layout_vmnk: Some(vmnk.clone()), defer_sync: true, ..Default::default() };
        assert_eq!(PipelineUmmaAsync::create(2, one, one, o.clone()).cta_group(), PipeCtaGroup::Two);
        let r = PipelineAsyncUmma::create(2, one, one, o);
        assert_eq!(r.cta_group(), PipeCtaGroup::Two);
        assert!(r.producer_mask().is_some() && r.consumer_mask().is_some());
        let t = PipelineTmaUmma::create(
            2,
            one,
            one,
            128,
            PipelineTmaUmmaOptions { cta_layout_vmnk: Some(vmnk), defer_sync: true, ..Default::default() },
        );
        assert_eq!(t.cta_group(), PipeCtaGroup::Two);
        assert!(t.is_leader_cta().as_constant().is_none());
    });
}

/// The sm_100a PTX of the mainloop: the stage releases and accumulator commits are
/// `tcgen05.commit.cta_group::1.mbarrier::arrive::one`; assembled by `ptxas -arch=sm_100a`
/// when it is on PATH.
#[test]
fn umma_pipelines_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for (name, producer) in [("umma_pipe_ptx_tma", Producer::Tma), ("umma_pipe_ptx_cp_async", Producer::CpAsync)] {
        let kernel = gemm_kernel(name, producer, Faults::default());
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        for w in [
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one",
            "tcgen05.alloc.cta_group::1",
            "tcgen05.dealloc.cta_group::1",
            "tcgen05.relinquish_alloc_permit.cta_group::1",
            "tcgen05.wait::ld",
            "mbarrier.try_wait",
        ] {
            assert!(ptx.contains(w), "{name}: no `{w}` in the PTX");
        }
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-umma-pipe-{}-{name}", std::process::id()));
        let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

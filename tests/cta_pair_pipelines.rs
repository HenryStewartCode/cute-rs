// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The 2-CTA mainloop's structure (the cluster layout from tiled_divide of the cluster by
// the MMA's thr_id, the leader CTA alone waiting and issuing the MMAs, the pipelines' cluster
// arguments, the producer tails after the epilogue) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py,
// and the multi-consumer kernel's roles (one TMA warp feeding both the MMA warp and
// ordinary consumer warps from the same stage) follow
//   examples/python/CuTeDSL/cute/blackwell/kernel/attention/mamba2_ssd/mamba2_ssd.py,
//     Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file is cute-rs's
// own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! CTA-pair pipelines on the simulator:
//!
//! * the `blackwell_fp16_gemm` example in `fp16_gemm_1.py`'s configuration (a CTA pair,
//!   `256x256x16` MMAs, 7 stages, multicast TMA loads) and on larger clusters, bit for bit
//!   against the CPU;
//! * the 2-CTA masks of the Blackwell pipelines on `(2, 2, 1)` and `(4, 1, 1)` clusters;
//! * a 2-CTA warp-specialised kernel through `PipelineTmaUmma` and `PipelineUmmaAsync`, and
//!   the simulator's reports when the odd CTA arms the full barrier and when the TMEM-sync
//!   mask lacks the peer;
//! * a Mamba2-style kernel through `PipelineTmaMultiConsumersAsync` (a TMA warp, the MMA
//!   warp and two async consumer warps reading each stage), with one-lane and per-lane
//!   signalling, and the report when async consumers counted per warp arrive from every
//!   lane;
//! * the TMA load selection for a cluster shape and an MMA (`cluster_shape_to_tma_atom_*`);
//! * the sm_100a PTX, assembled by `ptxas` when it is on PATH.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/blackwell_fp16_gemm.rs"]
mod blackwell_fp16_gemm;

use blackwell_fp16_gemm::{FP16_GEMM_0, FP16_GEMM_1, Fp16GemmConfig};
use cute::cpasync;
use cute::memory::{TmemAllocator, TmemAllocatorOptions};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    self, Agent, CooperativeGroup, CtaGroup as PipeCtaGroup, NamedBarrier, Pipeline, PipelineAsyncUmma, PipelineOp,
    PipelineTmaMultiConsumersAsync, PipelineTmaMultiConsumersAsyncOptions, PipelineTmaUmma, PipelineTmaUmmaOptions,
    PipelineUmmaAsync, PipelineUmmaOptions, agent_sync, init_empty_barrier_arrive_signal_2sm,
};
use cute::prelude::*;
use cute::tcgen05::{self, CtaGroup, Field, Ld32x32bOp, MmaF16BF16Op, OperandSource, Repetition, SmemLayoutAtomKind};
use cute::utils::sm100::{
    Sm100TmaLoadOp, cluster_shape_to_tma_atom_a, cluster_shape_to_tma_atom_b, cluster_shape_to_tma_atom_sfb,
    make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma,
};

// ---------------------------------------------------------------------------------------
// The example in fp16_gemm_1's configuration.
// ---------------------------------------------------------------------------------------

/// `fp16_gemm_1` on a 256x256x512 problem: one CTA pair, 8 k-tiles (the 7-stage ring
/// wraps). C matches the CPU bit for bit; the even CTA issues all 32 MMAs; each CTA issues
/// its own two TMA loads per k-tile; A and B are read from global memory exactly once.
#[test]
fn fp16_gemm_1_matches_the_reference() {
    let (m, n, k) = (256, 256, 512);
    let report = blackwell_fp16_gemm::run(&FP16_GEMM_1, m, n, k);
    let k_tiles = k / 64;
    assert_eq!(report.blocks, 2, "{report}");
    assert_eq!(report.tcgen05_mma_instructions as usize, k_tiles * 4, "{report}");
    assert_eq!(report.tma_instructions as usize, 2 * k_tiles * 2, "{report}");
    assert_eq!(report.global_bytes_read as usize, (m + n) * k * 2, "{report}");
    assert_eq!(report.global_bytes_written as usize, m * n * 2, "{report}");
}

/// Multicast in both modes: the example on `(2, 2, 1)` clusters, with the CTA pair (A
/// multicast across N) and with 1-CTA MMAs (A across N and B across M), and with the pair
/// on a `(4, 1, 1)` cluster (B multicast across the two pairs). Each multicast tile is read
/// from global memory once per cluster.
#[test]
fn fp16_gemm_on_larger_clusters() {
    // (config, m, n, k, bytes of A and B read)
    let pair_2x2 = Fp16GemmConfig { cluster_shape_mnk: (2, 2, 1), ..FP16_GEMM_1 };
    let one_2x2 = Fp16GemmConfig { cluster_shape_mnk: (2, 2, 1), ..FP16_GEMM_0 };
    let pair_4x1 = Fp16GemmConfig { cluster_shape_mnk: (4, 1, 1), ..FP16_GEMM_1 };
    let cases = [
        // One 256-row A tile shared by both N tiles; B's two tiles read once.
        (pair_2x2, 256, 512, 256, (256 + 512) * 256 * 2),
        // Two 128-row A tiles, each shared across N; two B tiles, each shared across M.
        (one_2x2, 256, 512, 256, (256 + 512) * 256 * 2),
        // Two A tiles; one B tile shared by both pairs.
        (pair_4x1, 512, 256, 256, (512 + 256) * 256 * 2),
    ];
    for (cfg, m, n, k, read) in cases {
        let report = blackwell_fp16_gemm::run(&cfg, m, n, k);
        let what = format!("{:?} {}", cfg.cluster_shape_mnk, cfg.name);
        assert_eq!(report.blocks, 4, "{what}: {report}");
        assert_eq!(report.global_bytes_read as usize, read, "{what}: {report}");
    }
}

#[test]
fn fp16_gemm_1_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let kernel = FP16_GEMM_1.build();
    let ptx = compile_ptx(&tc, &kernel);
    for w in [
        "tcgen05.mma.cta_group::2.kind::f16",
        ".multicast::cluster",
        "tcgen05.commit.cta_group::2",
        "tcgen05.alloc.cta_group::2",
        "tcgen05.dealloc.cta_group::2",
    ] {
        assert!(ptx.contains(w), "no `{w}` in the PTX");
    }
    ptxas_check("fp16_gemm_1", &ptx);
}

// ---------------------------------------------------------------------------------------
// The pipelines' 2-CTA masks.
// ---------------------------------------------------------------------------------------

/// The cluster layout of a cluster shape for a CTA pair's MMA: `tiled_divide` of the
/// column-major cluster by the MMA's `thr_id` (`2:1`).
fn pair_vmnk(cluster: (usize, usize, usize)) -> Layout {
    Layout::col_major(cluster).tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(Layout::new(2, 1)))]))
}

/// Each CTA of the cluster writes its pipelines' 2-CTA values (8 per CTA) and each lane its
/// async signalling destination and flag under `mcast` (2 per lane).
fn masks_kernel(cluster: (usize, usize, usize), mcast: (u32, u32)) -> Kernel {
    Kernel::builder("pair_masks").block_dim([32, 1, 1]).target(Arch::Sm100a).build(move || {
        let ctas = cluster.0 * cluster.1 * cluster.2;
        let out = param_tensor::<i32>("out", Layout::row_major((ctas, 8)));
        let sig = param_tensor::<i32>("sig", Layout::row_major((ctas, 32, 2)));
        let vmnk = pair_vmnk(cluster);
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let tma = PipelineTmaUmma::create(
            1,
            one,
            one,
            16,
            PipelineTmaUmmaOptions { cta_layout_vmnk: Some(vmnk.clone()), mcast_mode_mn: mcast, ..Default::default() },
        );
        let opts = PipelineUmmaOptions { cta_layout_vmnk: Some(vmnk.clone()), ..Default::default() };
        let acc = PipelineUmmaAsync::create(1, one, one, opts.clone());
        let au = PipelineAsyncUmma::create(1, one, one, opts);
        let multi = PipelineTmaMultiConsumersAsync::create(
            1,
            one,
            one,
            one,
            16,
            PipelineTmaMultiConsumersAsyncOptions {
                cta_layout_vmnk: Some(vmnk.clone()),
                mcast_mode_mn: mcast,
                ..Default::default()
            },
        );
        let r = block_idx_in_cluster();
        let vals = [
            tma.is_leader_cta().to::<i32>(),
            tma.consumer_mask().unwrap(),
            acc.producer_mask().unwrap(),
            acc.consumer_mask().unwrap(),
            au.producer_mask().unwrap(),
            au.consumer_mask().unwrap(),
            multi.consumer_mask().unwrap(),
            multi.is_leader_cta().to::<i32>(),
        ];
        if_(thread_idx().x.eq(0), || {
            for (i, v) in vals.iter().enumerate() {
                out.store((r, i as i32), *v);
            }
        });
        let (dst, signals) = init_empty_barrier_arrive_signal_2sm(&vmnk, thread_idx().x, mcast);
        sig.store((r, thread_idx().x, 0), dst);
        sig.store((r, thread_idx().x, 1), signals.to::<i32>());
        agent_sync(Agent::ThreadBlockCluster, false);
    })
}

/// CuTe DSL's formulas on `(2, 2, 1)` (cluster layout `(2, 1, 2, 1)`) and `(4, 1, 1)`
/// (`(2, 2, 1, 1)`) clusters: the leader is the even CTA; `PipelineTmaUmma`'s release
/// mask is the CTAs sharing this CTA's or its peer's tiles; `PipelineUmmaAsync` commits
/// to the pair and releases to `rank / 2 * 2`; `PipelineAsyncUmma` arrives on the leading
/// CTA and commits to the pair; async signalling lanes match `(M, L)` or `(N, L)` without
/// `V`.
#[test]
fn pair_masks_follow_cute_dsl() {
    // (cluster, mcast, the M and N coordinate of rank r, expected TmaUmma mask of rank r)
    type Coords = fn(i32) -> (i32, i32);
    type Case = ((usize, usize, usize), (u32, u32), Coords, fn(i32) -> i32);
    let cases: [Case; 3] = [
        ((2, 2, 1), (1, 1), |r| (0, r / 2), |_| 0b1111),
        ((2, 2, 1), (0, 1), |r| (0, r / 2), |r| 0b11 << (r / 2 * 2)),
        ((4, 1, 1), (1, 0), |r| (r / 2, 0), |r| 0b11 << (r / 2 * 2)),
    ];
    for (cluster, mcast, coords, tma_mask) in cases {
        let kernel = masks_kernel(cluster, mcast);
        let mut args = vec![KernelArg::buffer(&[0i32; 4 * 8]), KernelArg::buffer(&[0i32; 4 * 64])];
        let launch = LaunchConfig::new([cluster.0 as u32, cluster.1 as u32, 1], [32, 1, 1]).with_cluster([
            cluster.0 as u32,
            cluster.1 as u32,
            cluster.2 as u32,
        ]);
        kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{cluster:?}: {e}"));
        let (out, sig) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for r in 0..4i32 {
            let pair = 0b11 << (r / 2 * 2);
            let want = [
                i32::from(r % 2 == 0),
                tma_mask(r),
                pair,
                r / 2 * 2,
                r / 2 * 2,
                pair,
                tma_mask(r),
                i32::from(r % 2 == 0),
            ];
            let got = &out[r as usize * 8..r as usize * 8 + 8];
            assert_eq!(got, want, "{cluster:?} {mcast:?} rank {r}");
            let (m, n) = coords(r);
            for lane in 0..32i32 {
                let d = lane % 4;
                let (dm, dn) = coords(d);
                let shares = match mcast {
                    (1, 0) => dm == m,
                    (0, 1) => dn == n,
                    _ => dm == m || dn == n,
                };
                let at = (r * 64 + lane * 2) as usize;
                assert_eq!(sig[at], d, "{cluster:?} rank {r} lane {lane}: destination");
                assert_eq!(sig[at + 1], i32::from(lane < 4 && shares), "{cluster:?} rank {r} lane {lane}: signals");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// A 2-CTA warp-specialised kernel and its misuses.
// ---------------------------------------------------------------------------------------

/// K of the pair kernel: two k-tiles of 64.
const PK: usize = 128;
const PM: usize = 256;
const PN: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PairFault {
    None,
    /// The full barrier is armed on the odd CTA (a wrong leader computation).
    OddArms,
    /// The accumulator commit names only this CTA, not its peer.
    NoPeerBit,
}

/// `C = A B^T` (A `(256, 128)`, B `(128, 128)` K-major f16, C row-major f32) by a CTA pair
/// on a `(2, 1, 1)` cluster: warp 0 of each CTA loads its halves through a 2-stage
/// `PipelineTmaUmma`, the even CTA's warp 0 issues `256x128x16` MMAs and commits the
/// accumulator through `PipelineUmmaAsync`, and all warps of both CTAs read their half
/// of D from TMEM.
fn pair_pipeline_kernel(fault: PairFault) -> Kernel {
    Kernel::builder("pair_pipelines").block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let tiler = (PM, PN, 64);
        let a = param_tensor_aligned::<F16>("a", Layout::new((PM, PK), (PK, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((PN, PK), (PK, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((PM, PN), (PN, 1)));
        let mma = make_trivial_tiled_mma(
            DType::F16,
            DType::F16,
            OperandMajorMode::K,
            OperandMajorMode::K,
            DType::F32,
            CtaGroup::TWO,
            (PM, PN),
            OperandSource::SMEM,
        );
        let cluster = (2, 1, 1);
        let vmnk = Layout::col_major(cluster).tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(mma.thr_id().clone()))]));
        let sa_layout = make_smem_layout_a(&mma, tiler, DType::F16, 2, None);
        let sb_layout = make_smem_layout_b(&mma, tiler, DType::F16, 2, None);
        let op_a = cluster_shape_to_tma_atom_a(cluster, mma.thr_id()).unwrap();
        let op_b = cluster_shape_to_tma_atom_b(cluster, mma.thr_id()).unwrap();
        let (atom_a, tma_a) = make_tiled_tma_atom_a(op_a, &a, sa_layout.select(&[0, 1, 2]), tiler, &mma, None).into();
        let (atom_b, tma_b) = make_tiled_tma_atom_b(op_b, &b, sb_layout.select(&[0, 1, 2]), tiler, &mma, None).into();
        let sa = make_smem_tensor_aligned::<F16>(sa_layout.clone(), 1024);
        let sb = make_smem_tensor_aligned::<F16>(sb_layout.clone(), 1024);

        // (Made before the pipelines, whose cluster barrier publishes its dealloc mbarrier.)
        let mut tmem = TmemAllocator::new(
            NamedBarrier::new(1, 128),
            TmemAllocatorOptions { is_two_cta: true, arch: Arch::Sm100a, ..Default::default() },
        );
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let tx = size_in_bytes::<F16>(&sa_layout.select(&[0, 1, 2])).as_static().unwrap()
            + size_in_bytes::<F16>(&sb_layout.select(&[0, 1, 2])).as_static().unwrap();
        let tx = (tx * 2) as i32;
        let ab = PipelineTmaUmma::create(
            2,
            one,
            one,
            tx,
            PipelineTmaUmmaOptions { cta_layout_vmnk: Some(vmnk.clone()), ..Default::default() },
        );
        let acc = PipelineUmmaAsync::create(
            1,
            one,
            CooperativeGroup::new(Agent::Thread, 256),
            PipelineUmmaOptions { cta_layout_vmnk: Some(vmnk.clone()), ..Default::default() },
        );
        tmem.allocate(PN);

        let v = block_idx().x % 2;
        let leader = v.eq(0);
        let thr = mma.get_slice(v);
        let coord = (Int32::from(0), Int32::from(0), ..);
        let ga = local_tile_proj(&tma_a, tiler, coord, &[true, false, true]);
        let gb = local_tile_proj(&tma_b, tiler, coord, &[false, true, true]);
        let (tcga, tcgb) = (thr.partition_a(&ga), thr.partition_b(&gb));
        let lay = make_layout(1, 0);
        let (tas, tag) = cpasync::tma_partition(&atom_a, 0, &lay, &sa.group_modes(0, 3), &tcga.group_modes(0, 3));
        let (tbs, tbg) = cpasync::tma_partition(&atom_b, 0, &lay, &sb.group_modes(0, 3), &tcgb.group_modes(0, 3));
        tmem.wait_for_alloc();
        let tmem_ptr = tmem.retrieve_ptr::<f32>();
        let acc_t = make_tensor(
            tmem_ptr.clone(),
            mma.make_fragment_c::<f32>(&mma.partition_shape_c((PM, PN))).layout().clone(),
        );
        let (fa, fb) = (mma.make_fragment_a(&sa), mma.make_fragment_b(&sb));
        let warp = make_warp_uniform(warp_idx());

        if_(warp.eq(0), || {
            let (mut p, mut cons) = ab.make_participants();
            let (mut accp, _) = acc.make_participants();
            let acc_h = accp.current_handle();
            accp.advance(1);
            if_(leader, || acc.producer_acquire(&acc_h.state(), None));
            for kt in 0..PK / 64 {
                let h = p.current_handle();
                p.advance(1);
                let s = h.state();
                if fault == PairFault::OddArms {
                    ab.sync_object_empty().wait(s.index(), s.phase());
                    if_(v.eq(1), || ab.sync_object_full().arrive_and_expect_tx(s.index(), tx));
                } else {
                    ab.producer_acquire(&s, None);
                }
                let bar = CopyArgs::new().tma_bar_ptr(&h.barrier());
                copy_with(&atom_a, &tag.slice((.., kt)), &tas.slice((.., h.index())), bar.clone());
                copy_with(&atom_b, &tbg.slice((.., kt)), &tbs.slice((.., h.index())), bar);
                let ch = cons.current_handle();
                cons.advance(1);
                if_(leader, || {
                    ab.consumer_wait(&ch.state(), None);
                    let mut m = mma.clone();
                    for kb in 0..4 {
                        m.set(Field::ACCUMULATE, kt > 0 || kb > 0);
                        let k = (.., .., kb, ch.index());
                        gemm(&m, &acc_t, &fa.slice(k), &fb.slice(k), &acc_t);
                    }
                    ab.consumer_release(&ch.state());
                });
            }
            if_(leader, || {
                if fault == PairFault::NoPeerBit {
                    let own = Int32::from(1) << block_idx_in_cluster();
                    acc.sync_object_full().arrive(acc_h.index(), Some(own), Some(PipeCtaGroup::Two));
                } else {
                    acc.producer_commit(&acc_h.state());
                }
            });
        });

        // Epilogue: every warp of both CTAs reads its 32 TMEM lanes (rows v * 128 + lane).
        let (_, mut acc_c) = acc.make_participants();
        let h = acc_c.wait_and_advance(None);
        let lane_stride = 1usize << 16;
        let row = v * 128 + (warp % 4) * 32 + lane_idx();
        let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..PN / 32 {
            let layout = Layout::from_modes([Layout::new((32, 32), (1, lane_stride))]);
            let view =
                make_tensor(tmem_ptr.clone() + (warp % 4) * (32 * lane_stride) as i32 + (32 * chunk) as i32, layout);
            let frag = make_fragment::<f32>(32);
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                c.store((row, (32 * chunk + e) as i32), frag.load(e));
            }
        }
        h.release();

        // The producers' tails, then free the pair's TMEM.
        if_(warp.eq(0), || {
            let (mut p, _) = ab.make_participants();
            p.advance(PK / 64);
            p.tail();
            if_(leader, || {
                let (mut accp, _) = acc.make_participants();
                accp.advance(1);
                accp.tail();
            });
        });
        tmem.relinquish_alloc_permit();
        pipeline::sync(1);
        tmem.free(&tmem_ptr, 0);
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

/// f16 operands of `rows x k` and their f32 values.
fn operand(rows: usize, k: usize, seed: u64) -> (Vec<F16>, Vec<f32>) {
    let v = small_ints(rows * k, seed);
    (v.iter().map(|&x| F16::from_f32(x)).collect(), v)
}

/// `A B^T` in plain Rust (exact on small integers).
fn product(av: &[f32], bv: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            out[r * n + col] = (0..k).map(|i| av[r * k + i] * bv[col * k + i]).sum();
        }
    }
    out
}

fn pair_launch() -> LaunchConfig {
    LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1])
}

fn pair_args() -> (Vec<KernelArg>, Vec<f32>) {
    let ((a, av), (b, bv)) = (operand(PM, PK, 5), operand(PN, PK, 7));
    let want = product(&av, &bv, PM, PN, PK);
    (vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; PM * PN])], want)
}

/// The pair kernel with the library's 2-CTA pipelines: C matches plain Rust under every
/// built-in schedule.
#[test]
fn pair_pipelines_match_the_reference() {
    let kernel = pair_pipeline_kernel(PairFault::None);
    let (mut args, want) = pair_args();
    let report =
        kernel.simulate_all_orders(pair_launch(), &mut args, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.tcgen05_mma_instructions, 8, "{report}");
    assert_eq!(args[2].to_vec::<f32>(), want);
}

fn pair_report(fault: PairFault) -> String {
    let (mut args, _) = pair_args();
    match pair_pipeline_kernel(fault).simulate(pair_launch(), &mut args) {
        Ok(_) => panic!("{fault:?}: expected a simulator report"),
        Err(e) => e.to_string(),
    }
}

/// The odd CTA arming the full barrier (a wrong leader computation): the pair's four loads
/// report to the even CTA's barrier, which no arrival arms, so the leader's MMA warp never
/// sees the stage full: a deadlock with the copies in flight.
#[test]
fn tx_armed_on_the_odd_cta_is_reported() {
    let e = pair_report(PairFault::OddArms);
    assert!(e.contains("deadlock") && e.contains("4 bulk copies in flight"), "{e}");
}

/// A TMEM-sync mask without the peer's bit: the commit arrives on the even CTA only, and the
/// odd CTA's epilogue waits forever.
#[test]
fn a_tmem_sync_mask_without_the_peer_deadlocks() {
    let e = pair_report(PairFault::NoPeerBit);
    assert!(e.contains("deadlock") && e.contains("block (1,0,0)"), "{e}");
}

// ---------------------------------------------------------------------------------------
// PipelineTmaMultiConsumersAsync: a Mamba2-style kernel.
// ---------------------------------------------------------------------------------------

const XM: usize = 128;
const XN: usize = 64;
const XKT: usize = 64;
const XK_TILES: usize = 3;
const XK: usize = XKT * XK_TILES;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Signalling {
    /// One lane per async warp arrives; the async group counts 2 (warps).
    OneLane,
    /// Every async thread arrives; the async group counts 64 (the legacy form, chosen
    /// by default for that count).
    PerLane,
    /// The group counts warps, but every lane arrives (bypassing the signalling lanes).
    EveryLaneFault,
}

/// One TMA warp (5) fills a 2-stage ring of A `(128, 64)` and B `(64, 64)` k-tiles; the MMA
/// warp (4) accumulates `A B^T` into TMEM; two async warps (0 and 1) read every stage of A
/// and sum each row; then warps 0 to 3 read the accumulator. Outputs: C `(128, 64)` f32 and
/// the row sums of A (`128` f32).
fn multi_consumer_kernel(signalling: Signalling) -> Kernel {
    Kernel::builder("multi_consumer").block_dim([192, 1, 1]).target(Arch::Sm100a).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((XM, XK), (XK, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((XN, XK), (XK, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((XM, XN), (XN, 1)));
        let sums = param_tensor::<f32>("sums", Layout::col_major(XM));
        let op = MmaF16BF16Op::new(
            DType::F16,
            DType::F32,
            (XM, XN),
            CtaGroup::ONE,
            OperandSource::SMEM,
            OperandMajorMode::K,
            OperandMajorMode::K,
        );
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let atom = tcgen05::make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
        let (a_staged, b_staged) =
            (tile_to_shape_swizzled(&atom, (XM, XKT, 2)), tile_to_shape_swizzled(&atom, (XN, XKT, 2)));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom_a, tma_a) = cpasync::make_tiled_tma_atom(g2s, &a, a_staged.clone(), (XM, XKT), 1).into();
        let (atom_b, tma_b) = cpasync::make_tiled_tma_atom(g2s, &b, b_staged.clone(), (XN, XKT), 1).into();
        let sa = make_smem_tensor_aligned::<F16>(a_staged, 1024);
        let sb = make_smem_tensor_aligned::<F16>(b_staged, 1024);

        let mut tmem = TmemAllocator::new(
            NamedBarrier::new(1, 160),
            TmemAllocatorOptions { arch: Arch::Sm100a, ..Default::default() },
        );
        tmem.allocate(XN);
        let one = CooperativeGroup::new(Agent::Thread, 1);
        let async_group = CooperativeGroup::new(Agent::Thread, if signalling == Signalling::PerLane { 64 } else { 2 });
        let pipe = PipelineTmaMultiConsumersAsync::create(
            2,
            one,
            one,
            async_group,
            ((XM + XN) * XKT * 2) as i32,
            PipelineTmaMultiConsumersAsyncOptions { defer_sync: true, ..Default::default() },
        );
        assert_eq!(pipe.per_lane_signaling(), signalling == Signalling::PerLane);
        let acc = PipelineUmmaAsync::create(
            1,
            one,
            CooperativeGroup::new(Agent::Thread, 128),
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        );
        mbarrier_init_fence();
        sync_threads();
        let warp = make_warp_uniform(warp_idx());
        // Warps 0 to 4 use the TMEM (named barrier 1): wait for its address before warps 0
        // and 1 turn to the stages.
        if_(warp.lt(5), || tmem.wait_for_alloc());

        // The TMA warp.
        if_(warp.eq(5), || {
            let ga = local_tile(&tma_a, (XM, XKT), (0, ..));
            let gb = local_tile(&tma_b, (XN, XKT), (0, ..));
            let lay = make_layout(1, 0);
            let (ts_a, tg_a) = cpasync::tma_partition(&atom_a, 0, &lay, &sa.group_modes(0, 2), &ga.group_modes(0, 2));
            let (ts_b, tg_b) = cpasync::tma_partition(&atom_b, 0, &lay, &sb.group_modes(0, 2), &gb.group_modes(0, 2));
            let mut p = pipe.make_producer();
            for kt in 0..XK_TILES {
                let h = p.acquire_and_advance(None);
                let bar = CopyArgs::new().tma_bar_ptr(&h.barrier());
                copy_with(&atom_a, &tg_a.slice((.., kt)), &ts_a.slice((.., h.index())), bar.clone());
                copy_with(&atom_b, &tg_b.slice((.., kt)), &ts_b.slice((.., h.index())), bar);
            }
            p.tail();
        });

        // The MMA warp: the UMMA consumer.
        if_(warp.eq(4), || {
            let acc_t = make_tensor(
                tmem.retrieve_ptr::<f32>(),
                mma.make_fragment_c::<f32>(&mma.partition_shape_c((XM, XN))).layout().clone(),
            );
            let thr = mma.get_slice(0);
            let (_, mut cons) = pipe.make_participants();
            let (mut accp, _) = acc.make_participants();
            let acc_h = accp.acquire_and_advance(None);
            for kt in 0..XK_TILES {
                let h = cons.wait_and_advance(None);
                let fa = mma.make_fragment_a(&thr.partition_a(&sa.slice((.., .., h.index()))));
                let fb = mma.make_fragment_b(&thr.partition_b(&sb.slice((.., .., h.index()))));
                let mut m = mma.clone();
                for kb in 0..XKT / 16 {
                    m.set(Field::ACCUMULATE, kt > 0 || kb > 0);
                    gemm(&m, &acc_t, &fa.slice((.., .., kb)), &fb.slice((.., .., kb)), &acc_t);
                }
                h.release_op(PipelineOp::TCGen05Mma);
            }
            acc_h.commit();
            accp.tail();
        });

        // The async consumers: thread t sums rows 2t and 2t + 1 of A, stage by stage.
        if_(warp.lt(2), || {
            let t = thread_idx().x;
            let mut cons = pipe.make_consumer();
            let mut s0 = Float32::from(0.0);
            let mut s1 = Float32::from(0.0);
            for _ in 0..XK_TILES {
                let h = cons.wait_and_advance(None);
                for k in 0..XKT as i32 {
                    s0 = s0 + sa.load((t * 2, k, h.index())).to::<f32>();
                    s1 = s1 + sa.load((t * 2 + 1, k, h.index())).to::<f32>();
                }
                sync_warp();
                if signalling == Signalling::EveryLaneFault {
                    pipe.sync_object_empty_async().arrive(h.index(), None, None);
                } else {
                    h.release_op(PipelineOp::AsyncThread);
                }
            }
            sums.store((t * 2,), s0);
            sums.store((t * 2 + 1,), s1);
        });

        // The epilogue: warps 0 to 3 read the accumulator (rows warp * 32 + lane).
        if_(warp.lt(4), || {
            let tmem_ptr = tmem.retrieve_ptr::<f32>();
            let (_, mut acc_c) = acc.make_participants();
            let h = acc_c.wait_and_advance(None);
            let lane_stride = 1usize << 16;
            let row = warp * 32 + lane_idx();
            let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
            for chunk in 0..XN / 32 {
                let layout = Layout::from_modes([Layout::new((32, 32), (1, lane_stride))]);
                let view =
                    make_tensor(tmem_ptr.clone() + warp * (32 * lane_stride) as i32 + (32 * chunk) as i32, layout);
                let frag = make_fragment::<f32>(32);
                copy(&ld, &view, &frag);
                fence_view_async_tmem_load();
                for e in 0..32 {
                    c.store((row, (32 * chunk + e) as i32), frag.load(e));
                }
            }
            h.release();
            tmem.relinquish_alloc_permit();
            NamedBarrier::new(2, 128).arrive_and_wait();
            tmem.free(&tmem_ptr, 0);
        });
    })
}

fn multi_args() -> (Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    let ((a, av), (b, bv)) = (operand(XM, XK, 21), operand(XN, XK, 23));
    let want_c = product(&av, &bv, XM, XN, XK);
    let want_sums = (0..XM).map(|r| av[r * XK..(r + 1) * XK].iter().sum()).collect();
    let args = vec![
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::buffer(&vec![0f32; XM * XN]),
        KernelArg::buffer(&vec![0f32; XM]),
    ];
    (args, want_c, want_sums)
}

fn multi_launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [192, 1, 1])
}

/// One TMA producer, the MMA warp and two async consumer warps reading the same stages:
/// C and the row sums match plain Rust under every built-in schedule, with one-lane and
/// with per-lane signalling.
#[test]
fn multi_consumer_pipeline_matches_the_reference() {
    for signalling in [Signalling::OneLane, Signalling::PerLane] {
        let kernel = multi_consumer_kernel(signalling);
        let (mut args, want_c, want_sums) = multi_args();
        let report = kernel
            .simulate_all_orders(multi_launch(), &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{signalling:?}: {e}"));
        assert_eq!(report.tcgen05_mma_instructions as usize, XK_TILES * 4, "{signalling:?}");
        assert_eq!(args[2].to_vec::<f32>(), want_c, "{signalling:?}: C");
        assert_eq!(args[3].to_vec::<f32>(), want_sums, "{signalling:?}: row sums");
    }
}

/// Async consumers counted one per warp that arrive from every lane: more arrivals than
/// the empty barrier counts.
#[test]
fn async_consumers_arriving_from_every_lane_are_reported() {
    let (mut args, _, _) = multi_args();
    let e = match multi_consumer_kernel(Signalling::EveryLaneFault).simulate(multi_launch(), &mut args) {
        Ok(_) => panic!("expected a simulator report"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("more arrivals than it was initialised with"), "{e}");
}

/// The pipeline's checks when it is made.
#[test]
fn multi_consumer_create_checks() {
    let build = |f: fn()| {
        std::panic::catch_unwind(|| {
            Kernel::builder("multi_checks").block_dim([64, 1, 1]).target(Arch::Sm100a).build(f);
        })
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())))
        .unwrap_or_default()
    };
    // Different agents.
    let e = build(|| {
        PipelineTmaMultiConsumersAsync::create(
            1,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Warp, 1),
            16,
            Default::default(),
        );
    });
    assert!(e.contains("same kind of agent"), "{e}");
    // Per-lane signalling in a cluster.
    let e = build(|| {
        PipelineTmaMultiConsumersAsync::create(
            1,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 64),
            16,
            PipelineTmaMultiConsumersAsyncOptions {
                cta_layout_vmnk: Some(Layout::col_major((2usize, 1usize, 1usize, 1usize))),
                ..Default::default()
            },
        );
    });
    assert!(e.contains("single CTA"), "{e}");
    // Multicast signalling: one arrival per warp and partner; the empty barrier counts both.
    let e = build(|| {
        let p = PipelineTmaMultiConsumersAsync::create(
            1,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 32),
            CooperativeGroup::new(Agent::Thread, 64),
            16,
            PipelineTmaMultiConsumersAsyncOptions { enable_multicast_signaling: true, ..Default::default() },
        );
        assert!(!p.per_lane_signaling());
        assert_eq!(p.sync_object_empty().arrive_count().as_static(), Some(3));
        assert_eq!(p.sync_object_empty_umma().op(), PipelineOp::TCGen05Mma);
        assert_eq!(p.sync_object_empty_async().op(), PipelineOp::AsyncThread);
        panic!("checked");
    });
    assert!(e.contains("checked"), "{e}");
}

#[test]
fn multi_consumer_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptx = compile_ptx(&tc, &multi_consumer_kernel(Signalling::OneLane));
    for w in ["tcgen05.mma.cta_group::1.kind::f16", "tcgen05.commit.cta_group::1", "mbarrier.arrive"] {
        assert!(ptx.contains(w), "no `{w}` in the PTX");
    }
    ptxas_check("multi_consumer", &ptx);
    let ptx = compile_ptx(&tc, &pair_pipeline_kernel(PairFault::None));
    ptxas_check("pair_pipelines", &ptx);
}

// ---------------------------------------------------------------------------------------
// TMA load selection.
// ---------------------------------------------------------------------------------------

/// Every branch of `cluster_shape_to_tma_atom_A/B/SFB`.
#[test]
fn tma_atom_selection() {
    use CtaGroup::{ONE, TWO};
    let one = Layout::new(1, 0);
    let two = Layout::new(2, 1);
    let tile = |g| Sm100TmaLoadOp::Tile(cpasync::CopyBulkTensorTileG2SOp { cta_group: g });
    let mc = |g| Sm100TmaLoadOp::Multicast(cpasync::CopyBulkTensorTileG2SMulticastOp { cta_group: g });
    // A: multicast when the cluster's N extent is above 1.
    assert_eq!(cluster_shape_to_tma_atom_a((1, 1, 1), &one), Ok(tile(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_a((2, 1, 1), &one), Ok(tile(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_a((1, 2, 1), &one), Ok(mc(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_a((2, 1, 1), &two), Ok(tile(TWO)));
    assert_eq!(cluster_shape_to_tma_atom_a((2, 4, 1), &two), Ok(mc(TWO)));
    // B: multicast when the cluster's M extent differs from the SM count.
    assert_eq!(cluster_shape_to_tma_atom_b((1, 4, 1), &one), Ok(tile(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_b((2, 1, 1), &one), Ok(mc(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_b((2, 2, 1), &two), Ok(tile(TWO)));
    assert_eq!(cluster_shape_to_tma_atom_b((4, 1, 1), &two), Ok(mc(TWO)));
    // SFB: always multicast for a pair; for one SM when the cluster's M is above 1.
    assert_eq!(cluster_shape_to_tma_atom_sfb((2, 1, 1), &two), Ok(mc(TWO)));
    assert_eq!(cluster_shape_to_tma_atom_sfb((1, 2, 1), &one), Ok(tile(ONE)));
    assert_eq!(cluster_shape_to_tma_atom_sfb((2, 1, 1), &one), Ok(mc(ONE)));
    // The op's accessors and its CopyOp.
    let op = mc(TWO);
    assert!(op.is_multicast() && op.cta_group() == TWO && !tile(ONE).is_multicast());
    // Errors: M not a multiple of the SM count, an MMA of 4 SMs, dynamic shapes.
    type Select = fn(Shape, &Layout) -> Result<Sm100TmaLoadOp, String>;
    let selects: [Select; 3] = [
        |s, t| cluster_shape_to_tma_atom_a(s, t),
        |s, t| cluster_shape_to_tma_atom_b(s, t),
        |s, t| cluster_shape_to_tma_atom_sfb(s, t),
    ];
    for f in selects {
        let e = f((1, 2, 1).into(), &two).unwrap_err();
        assert!(e.contains("not a multiple"), "{e}");
        let e = f((4, 1, 1).into(), &Layout::new(4, 1)).unwrap_err();
        assert!(e.contains("1 or 2"), "{e}");
    }
    let mut dynamic = None;
    Kernel::builder("tma_select_dynamic").build(|| {
        let x = param_extent("x", 1);
        let shape: Shape =
            Tree::Branch(vec![Tree::Leaf(IndexExpr::from(x)), Tree::Leaf(1usize.into()), Tree::Leaf(1usize.into())]);
        dynamic = Some(cluster_shape_to_tma_atom_a(shape, &one).unwrap_err());
    });
    assert!(dynamic.unwrap().contains("static"));
}

// ---------------------------------------------------------------------------------------
// PTX helpers.
// ---------------------------------------------------------------------------------------

fn compile_ptx(tc: &Toolchain, kernel: &Kernel) -> String {
    let name = kernel.ir().name.clone();
    kernel
        .compile_with(tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .ptx()
        .to_string()
}

/// Assemble `ptx` with `ptxas -arch=sm_100a` when `ptxas` is on PATH.
fn ptxas_check(name: &str, ptx: &str) {
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let Some(ptxas) = ptxas else { return };
    let base = std::env::temp_dir().join(format!("cute-pair-pipelines-{}-{name}", std::process::id()));
    let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
    std::fs::write(&src, ptx).unwrap();
    let out = std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&cubin);
    assert!(
        out.status.success(),
        "{name}: ptxas: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

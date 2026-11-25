// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernel's structure (a CTA pair's tiled MMA sliced by the CTA's rank in its pair,
// MMA-aware TMA atoms with cta_group TWO whose loads report to the leader CTA's mbarrier,
// the leader alone expecting both CTAs' bytes and issuing the MMAs, a commit multicast to
// both CTAs of the pair) follows NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's own,
// under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! CTA pairs (roadmap part m10-cta-pair): a `(2, 1, 1)` cluster computes one `M x N` tile
//! with `tcgen05.mma.cta_group::2`. Each CTA loads its half of A (`M / 2` rows) and of B
//! (`N / 2` rows) with 2-CTA TMA loads that report to the leader CTA's mbarrier, the leader
//! expects both CTAs' bytes, waits, issues the MMAs and commits with a mask naming both
//! CTAs, and every warp of both CTAs waits on its own CTA's mbarrier and reads its half of
//! D with `tcgen05.ld`. Checked against plain Rust for M = 256 (Layout A) and M = 128
//! (Layout B), N = 128 and 256, f16 and bf16; then the simulator's reports for the misuses
//! the PTX ISA forbids, the atoms' 2-CTA split, and the sm_100a PTX (assembled by `ptxas`
//! when it is on PATH).

use cute::cpasync;
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::prelude::*;
use cute::tcgen05::{self, CtaGroup, Field, Ld32x32bOp, OperandSource, Repetition};
use cute::utils::sm100::{make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma};

/// K of the tile: four k-blocks of 16.
const K: usize = 64;

/// What a kernel gets wrong on purpose.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Fault {
    #[default]
    None,
    /// The odd CTA issues the MMAs.
    OddIssues,
    /// The commit names only the leader, so the peer's waiters wait forever.
    NoPeerBit,
    /// TMEM is allocated with `cta_group::1` while the MMAs use `cta_group::2`.
    MixedGroups,
    /// Only the leader CTA allocates the pair's TMEM.
    LeaderAllocOnly,
    /// The peer overwrites its half of A while the pair's MMAs may read it.
    PeerWritesA,
}

fn tiled_mma<T: Numeric>(m: usize, n: usize) -> TiledMma {
    make_trivial_tiled_mma(
        T::DTYPE,
        T::DTYPE,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F32,
        CtaGroup::TWO,
        (m, n),
        OperandSource::SMEM,
    )
}

/// The `(m, n)` tile `C = A B^T` (A `(m, K)` and B `(n, K)` K-major, C row-major Float32)
/// by a CTA pair of four warps each.
fn pair_kernel<T: Numeric>(name: &'static str, m: usize, n: usize, fault: Fault) -> Kernel {
    Kernel::builder(name).block_dim([128, 1, 1]).target(Arch::Sm100a).build(move || {
        let tiler = (m, n, K);
        let a = param_tensor_aligned::<T>("a", Layout::new((m, K), (K, 1)), 16);
        let b = param_tensor_aligned::<T>("b", Layout::new((n, K), (K, 1)), 16);
        let c = param_tensor::<f32>("c", Layout::new((m, n), (n, 1)));
        let mut mma = tiled_mma::<T>(m, n);
        let sa_layout = make_smem_layout_a(&mma, tiler, T::DTYPE, 1, None);
        let sb_layout = make_smem_layout_b(&mma, tiler, T::DTYPE, 1, None);
        let op = cpasync::CopyBulkTensorTileG2SOp { cta_group: CtaGroup::TWO };
        let (atom_a, tma_a) = make_tiled_tma_atom_a(op, &a, sa_layout.select(&[0, 1, 2]), tiler, &mma, None).into();
        let (atom_b, tma_b) = make_tiled_tma_atom_b(op, &b, sb_layout.select(&[0, 1, 2]), tiler, &mma, None).into();
        let sa = make_smem_tensor_aligned::<T>(sa_layout.clone(), 1024);
        let sb = make_smem_tensor_aligned::<T>(sb_layout.clone(), 1024);
        let slot = alloc_smem::<u32>(1, 4);
        let tma_bar = alloc_smem::<u64>(1, 8);
        let mma_bar = alloc_smem::<u64>(1, 8);
        let issued = alloc_smem::<u64>(1, 8);
        let written = alloc_smem::<u64>(1, 8);
        let rank = block_idx_in_cluster();
        let leader = rank.eq(0);
        let warp = make_warp_uniform(warp_idx());
        let cols = n.next_power_of_two().max(32) as i32;
        let two = fault != Fault::MixedGroups;

        // TMEM for the pair (one warp of each CTA), and the mbarriers, published to the
        // cluster (the peer's loads report to the leader's mbarrier and the leader's commit
        // arrives on the peer's) by a cluster barrier, which also orders the allocation's
        // address before every warp reads it.
        let allocates = if fault == Fault::LeaderAllocOnly { warp.eq(0) & leader } else { warp.eq(0) };
        if_(allocates, || alloc_tmem(cols, &slot, Some(two), Arch::Sm100a));
        if_(thread_idx().x.eq(0), || {
            mbarrier_init(&tma_bar, 1);
            mbarrier_init(&mma_bar, 1);
            mbarrier_init(&issued, 1);
            mbarrier_init(&written, 1);
        });
        mbarrier_init_fence();
        cluster_arrive();
        cluster_wait();

        // Each CTA's half of A and B: the pair's MMA partitioned by the CTA's rank in it.
        let thr = mma.get_slice(rank % 2);
        let coord = (Int32::from(0), Int32::from(0), ..);
        let ga = local_tile_proj(&tma_a, tiler, coord, &[true, false, true]);
        let gb = local_tile_proj(&tma_b, tiler, coord, &[false, true, true]);
        let (tcga, tcgb) = (thr.partition_a(&ga), thr.partition_b(&gb));
        let (tas, tag) =
            cpasync::tma_partition(&atom_a, 0, &make_layout(1, 0), &sa.group_modes(0, 3), &tcga.group_modes(0, 3));
        let (tbs, tbg) =
            cpasync::tma_partition(&atom_b, 0, &make_layout(1, 0), &sb.group_modes(0, 3), &tcgb.group_modes(0, 3));
        let bytes = ((m + n) * K * T::DTYPE.bytes()) as i32;
        if_(warp.eq(0), || {
            if_(leader & elect_one(), || mbarrier_arrive_and_expect_tx(&tma_bar, bytes));
            let bar = CopyArgs::new().tma_bar_ptr(&tma_bar);
            copy_with(&atom_a, &tag.slice((.., 0)), &tas.slice((.., 0)), bar.clone());
            copy_with(&atom_b, &tbg.slice((.., 0)), &tbs.slice((.., 0)), bar);
        });

        let tmem_ptr = retrieve_tmem_ptr::<f32>(16, &slot);
        let acc_shape = mma.partition_shape_c((m, n));
        let acc = make_tensor(tmem_ptr.clone(), mma.make_fragment_c::<f32>(&acc_shape).layout().clone());
        let (fa, fb) = (mma.make_fragment_a(&sa), mma.make_fragment_b(&sb));
        let issuer = if fault == Fault::OddIssues { rank.eq(1) } else { leader };
        if_(warp.eq(0) & issuer, || {
            // (An odd issuer's own mbarrier hears of no load: it goes straight to the MMA.)
            if fault != Fault::OddIssues {
                mbarrier_wait(&tma_bar, 0);
            }
            mma.set(Field::ACCUMULATE, false);
            for kb in 0..K / 16 {
                gemm(&mma, &acc, &fa.slice((.., .., kb, 0)), &fb.slice((.., .., kb, 0)), &acc);
                mma.set(Field::ACCUMULATE, true);
            }
            let mask: i16 = if fault == Fault::NoPeerBit { 0b01 } else { 0b11 };
            let group = if two { CtaGroup::TWO } else { CtaGroup::ONE };
            if fault == Fault::PeerWritesA {
                // Tell the peer the MMAs are issued (and, through the leader's wait, that
                // its loads have landed), and commit them only once it has written.
                let peer = MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(Int32::from(1)),
                    scope: Some(MemScopeKind::Cluster),
                    ..Default::default()
                };
                if_(elect_one(), || mbarrier_arrive_with(&issued, 1, peer));
                mbarrier_wait(&written, 0);
            }
            if_(elect_one(), || tcgen05::commit(&mma_bar, Some(Int16::from(mask)), group));
        });
        if fault == Fault::PeerWritesA {
            if_(rank.eq(1), || {
                mbarrier_wait(&issued, 0);
                sa.store((thread_idx().x % (m / 2) as i32, 0, 0, 0), a.load((0, 0)));
                let leader = MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(Int32::from(0)),
                    scope: Some(MemScopeKind::Cluster),
                    ..Default::default()
                };
                if_(thread_idx().x.eq(0), || mbarrier_arrive_with(&written, 1, leader));
            });
        }

        // Every warp of both CTAs: wait for the MMAs, then read this CTA's half of D. Warp w
        // reads TMEM lanes 32 w to 32 w + 31: for M = 256 rows rank * 128 + lane of all N
        // columns; for M = 128 rows rank * 64 + lane % 64 of the column half lane / 64.
        mbarrier_wait(&mma_bar, 0);
        let lane_stride = 1usize << 16;
        let l = lane_idx();
        let tmem_lane = (warp % 4) * 32 + l;
        let (row, col0, chunks) = if m == 256 {
            (rank * 128 + tmem_lane, Int32::from(0), n / 32)
        } else {
            (rank * 64 + tmem_lane % 64, (tmem_lane / 64) * (n / 2) as i32, n / 64)
        };
        let ld = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X32));
        for chunk in 0..chunks {
            let layout = Layout::from_modes([Layout::new((32, 32), (1, lane_stride))]);
            let view =
                make_tensor(tmem_ptr.clone() + (warp % 4) * (32 * lane_stride) as i32 + (32 * chunk) as i32, layout);
            let frag = make_fragment::<f32>(32);
            copy(&ld, &view, &frag);
            fence_view_async_tmem_load();
            for e in 0..32 {
                c.store((row, col0 + (32 * chunk + e) as i32), frag.load(e));
            }
        }

        // Free the pair's TMEM once both CTAs are done with it, and keep both CTAs (and
        // their mbarriers) alive until then.
        fence_view_async_tmem_load();
        cluster_arrive();
        cluster_wait();
        if_(warp.eq(0), || {
            relinquish_tmem_alloc_permit(Some(two));
            dealloc_tmem(&tmem_ptr, cols, Some(two), Arch::Sm100a);
        });
    })
}

/// Host conversions of the operand types.
trait HostFloat: Numeric + Copy {
    fn host(x: f32) -> Self;
}

impl HostFloat for F16 {
    fn host(x: f32) -> Self {
        F16::from_f32(x)
    }
}

impl HostFloat for BF16 {
    fn host(x: f32) -> Self {
        BF16::from_f32(x)
    }
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

fn launch() -> LaunchConfig {
    LaunchConfig::new([2, 1, 1], [128, 1, 1]).with_cluster([2, 1, 1])
}

/// The operands and an empty C, with the reference product (exact on small integers).
fn operands<T: HostFloat>(m: usize, n: usize) -> (Vec<KernelArg>, Vec<f32>) {
    let (av, bv) = (small_ints(m * K, 3 + m as u64), small_ints(n * K, 11 + n as u64));
    let a: Vec<T> = av.iter().map(|&x| T::host(x)).collect();
    let b: Vec<T> = bv.iter().map(|&x| T::host(x)).collect();
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            want[r * n + col] = (0..K).map(|k| av[r * K + k] * bv[col * K + k]).sum();
        }
    }
    (vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0f32; m * n])], want)
}

fn run_pair<T: HostFloat>(name: &'static str, m: usize, n: usize, all_orders: bool) {
    let kernel = pair_kernel::<T>(name, m, n, Fault::None);
    let (mut args, want) = operands::<T>(m, n);
    let report = if all_orders {
        kernel.simulate_all_orders(launch(), &mut args, &SimOptions::default())
    } else {
        kernel.simulate(launch(), &mut args)
    }
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(report.tcgen05_mma_instructions, (K / 16) as u64, "{name}: one pair MMA per k-block");
    let got = args[2].to_vec::<f32>();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{name}: C({}, {})", i / n, i % n);
    }
}

#[test]
fn pair_tiles_match_the_rust_reference() {
    run_pair::<F16>("pair_256x128_f16", 256, 128, true);
    run_pair::<BF16>("pair_256x256_bf16", 256, 256, false);
    run_pair::<F16>("pair_128x256_f16", 128, 256, false);
    run_pair::<BF16>("pair_128x128_bf16", 128, 128, true);
}

/// The simulator's report for the 256x128 f16 pair kernel with `fault`.
fn report_for(fault: Fault) -> String {
    let kernel = pair_kernel::<F16>("pair_misuse", 256, 128, fault);
    let (mut args, _) = operands::<F16>(256, 128);
    match kernel.simulate(launch(), &mut args) {
        Ok(_) => panic!("expected a simulator report"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn an_mma_issued_by_the_odd_cta_is_reported() {
    let e = report_for(Fault::OddIssues);
    assert!(e.contains("an odd CTA") && e.contains("cta_group::2"), "{e}");
}

#[test]
fn a_commit_without_the_peer_bit_deadlocks_the_peer() {
    let e = report_for(Fault::NoPeerBit);
    assert!(e.contains("deadlock"), "{e}");
}

#[test]
fn mixed_cta_groups_are_reported() {
    let e = report_for(Fault::MixedGroups);
    assert!(e.contains("cta_group::2 after a tcgen05 instruction with .cta_group::1"), "{e}");
}

#[test]
fn a_pair_allocation_by_one_cta_is_reported() {
    let e = report_for(Fault::LeaderAllocOnly);
    assert!(e.contains("deadlock") && e.contains("tcgen05.alloc"), "{e}");
}

#[test]
fn the_peer_writing_its_operand_during_the_mma_is_reported() {
    let e = report_for(Fault::PeerWritesA);
    assert!(e.contains("still reads"), "{e}");
}

/// The atoms of a CTA pair: each CTA's box is its half of A's rows (and of B's), as
/// `make_tiled_tma_atom` gives for the half tile; on a `(2, 2)` cluster the multicast
/// form shares A across the N mode. CtaGroup::TWO on an sm_90a kernel is refused.
#[test]
fn pair_atoms_split_the_tile_between_the_ctas() {
    let mut out = None;
    Kernel::builder("pair_atoms").target(Arch::Sm100a).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((512, 64), (64, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((512, 64), (64, 1)), 16);
        let mma = tiled_mma::<F16>(256, 256);
        let tiler = (256, 256, 64);
        let sa = make_smem_layout_a(&mma, tiler, DType::F16, 2, None);
        let sb = make_smem_layout_b(&mma, tiler, DType::F16, 2, None);
        assert_eq!(mma.partition_shape_a((256, 64)).to_string(), "((128,16),1,4)");
        let op = cpasync::CopyBulkTensorTileG2SOp { cta_group: CtaGroup::TWO };
        let ia = make_tiled_tma_atom_a(op, &a, sa.select(&[0, 1, 2]), tiler, &mma, None);
        let ib = make_tiled_tma_atom_b(op, &b, sb.select(&[0, 1, 2]), tiler, &mma, None);
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp { cta_group: CtaGroup::TWO };
        let vmnk: Shape = (2usize, 1usize, 2usize, 1usize).into();
        let ma = make_tiled_tma_atom_a(mc, &a, sa.select(&[0, 1, 2]), tiler, &mma, Some(vmnk));
        let box_of = |i: &cpasync::TmaInfo<F16>| i.atom.tma_info().unwrap().spec().box_dims.clone();
        out = Some((box_of(&ia), box_of(&ib), ma.atom.tma_info().unwrap().num_multicast(), box_of(&ma)));
    });
    let (ba, bb, mca, bma) = out.unwrap();
    assert_eq!(ba, vec![64, 128], "A: 128 of the pair's 256 rows per CTA");
    assert_eq!(bb, vec![64, 128], "B: 128 of the pair's 256 rows per CTA");
    assert_eq!(mca, 2);
    assert_eq!(bma, vec![64, 64], "the multicast splits each CTA's half again");
    let e = std::panic::catch_unwind(|| {
        Kernel::builder("pair_sm90").target(Arch::Sm90a).build(|| {
            let a = param_tensor_aligned::<F16>("a", Layout::new((512, 64), (64, 1)), 16);
            let op = cpasync::CopyBulkTensorTileG2SOp { cta_group: CtaGroup::TWO };
            cpasync::make_tiled_tma_atom(op, &a, Layout::new((64, 64), (64, 1)), (64, 64), 1);
        });
    })
    .unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("sm_90a"), "{e}");
}

/// The tiled MMA of a pair: two "threads" (the CTAs), each partitioning its half.
#[test]
fn pair_tiled_mma_partitions_halves() {
    Kernel::builder("pair_mma").target(Arch::Sm100a).build(|| {
        for (m, n) in [(128, 128), (256, 64)] {
            let mma = tiled_mma::<F16>(m, n);
            assert_eq!(mma.num_threads(), 2);
            assert_eq!(mma.partition_shape_c((m, n)).to_string(), format!("(({},{n}),1,1)", m / 2));
            assert_eq!(mma.partition_shape_b((n, 64)).to_string(), format!("(({},16),1,4)", n / 2));
            // The second CTA's first value of A is row M / 2.
            assert_eq!(mma.layout_a_tv().call(1), m / 2);
        }
    });
}

#[test]
fn pair_kernels_need_sm100a() {
    let kernel = pair_kernel::<F16>("pair_target", 256, 128, Fault::None);
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("tcgen05") && e.contains("sm_90a"), "{e}");
}

#[test]
fn pair_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    for kernel in [
        pair_kernel::<F16>("pair_ptx_256", 256, 128, Fault::None),
        pair_kernel::<BF16>("pair_ptx_128", 128, 256, Fault::None),
    ] {
        let name = kernel.ir().name.clone();
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptx = compiled.ptx();
        for w in [
            "tcgen05.mma.cta_group::2.kind::f16",
            ".cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
            "tcgen05.commit.cta_group::2",
            "tcgen05.alloc.cta_group::2",
            "tcgen05.dealloc.cta_group::2",
        ] {
            assert!(ptx.contains(w), "{name}: no `{w}` in the PTX");
        }
        let Some(ptxas) = &ptxas else { continue };
        let dir = std::env::temp_dir().join(format!("cute-pair-{}-{name}", std::process::id()));
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

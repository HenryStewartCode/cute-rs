// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hopper dense GEMM port: the simulator
//! against a plain Rust GEMM for clusters of 1x1, 2x1, 1x2 and 2x2 CTAs,
//! Float16, BFloat16, FP8 and 8-bit integer inputs, every major mode the
//! example allows, partial tiles and batches; the configuration it derives;
//! and the sm_90a PTX with `ptxas` (register counts, no serialised WGMMA)
//! where installed.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/hopper_gemm.rs"]
mod hopper_gemm;

use cute::prelude::*;
use hopper_gemm::{Data, HopperWgmmaGemmKernel, Majors, Problem, run};

fn gemm(
    ab: (DType, DType),
    c: DType,
    acc: DType,
    majors: Majors,
    tile_mn: (usize, usize),
    cluster_mn: (usize, usize),
) -> HopperWgmmaGemmKernel {
    HopperWgmmaGemmKernel::new(acc, tile_mn, cluster_mn).setup(ab.0, ab.1, c, majors)
}

fn f16(majors: Majors, tile_mn: (usize, usize), cluster_mn: (usize, usize)) -> HopperWgmmaGemmKernel {
    gemm((DType::F16, DType::F16), DType::F16, DType::F32, majors, tile_mn, cluster_mn)
}

/// Build once, then simulate and check each problem.
fn check_all(gemm: &HopperWgmmaGemmKernel, problems: &[(usize, usize, usize, usize)]) {
    let kernel = gemm.build();
    for &(m, n, k, l) in problems {
        run(gemm, &kernel, &Problem::new(m, n, k, l));
    }
}

/// The example's default tile (128x128) in each cluster shape the brief
/// asks for, on whole and partial tiles (M, N and K not multiples of the
/// tile), and a K of one k-tile and of more k-tiles than stages.
#[test]
fn float16_clusters_and_partial_tiles() {
    check_all(&f16(Majors::KKN, (128, 128), (1, 1)), &[(128, 128, 64, 1), (200, 136, 72, 1), (128, 64, 512, 1)]);
    check_all(&f16(Majors::KKN, (128, 128), (2, 1)), &[(256, 128, 128, 1), (136, 120, 200, 1)]);
    check_all(&f16(Majors::KKN, (128, 128), (1, 2)), &[(128, 256, 128, 1), (72, 136, 136, 1)]);
    check_all(&f16(Majors::KKN, (64, 64), (2, 2)), &[(128, 128, 128, 1), (200, 72, 64, 1)]);
}

/// Every major mode of A, B and C that Float16 takes, with multicast of
/// both operands.
#[test]
fn float16_every_major_mode() {
    for a in ['k', 'm'] {
        for b in ['k', 'n'] {
            for c in ['n', 'm'] {
                let majors = Majors::new(a, b, c);
                check_all(&f16(majors, (64, 128), (2, 1)), &[(128, 136, 72, 1)]);
                check_all(&f16(majors, (128, 64), (1, 2)), &[(136, 128, 64, 1)]);
            }
        }
    }
}

/// Batches (L along the grid's z).
#[test]
fn batched() {
    check_all(&f16(Majors::new('m', 'k', 'n'), (64, 64), (2, 1)), &[(128, 64, 64, 3)]);
    check_all(&f16(Majors::KKN, (128, 128), (1, 1)), &[(136, 72, 80, 2)]);
}

/// The two-warpgroup ("cooperative") kernel the example picks for tiles
/// wider than 128: 256 threads, the epilogue tile `(128, 32)`.
#[test]
fn two_warpgroups() {
    let g = f16(Majors::KKN, (128, 256), (1, 1));
    assert_eq!((g.atom_layout_mnk, g.threads_per_cta, g.epi_tile), ((2, 1, 1), 256, (128, 32)));
    check_all(&g, &[(128, 256, 64, 1), (200, 264, 72, 1)]);
    check_all(&f16(Majors::new('m', 'n', 'm'), (128, 256), (2, 1)), &[(256, 256, 64, 1)]);
}

#[test]
fn bfloat16() {
    let bf16 = (DType::BF16, DType::BF16);
    check_all(&gemm(bf16, DType::BF16, DType::F32, Majors::KKN, (128, 128), (2, 1)), &[(256, 128, 128, 1)]);
    check_all(&gemm(bf16, DType::BF16, DType::F32, Majors::new('m', 'n', 'm'), (64, 128), (1, 2)), &[(72, 200, 64, 1)]);
}

/// Float16 accumulators, and Float32 C (the universal register-to-shared
/// copy instead of `stmatrix`), both C majors.
#[test]
fn float16_accumulators_and_float32_c() {
    let h = (DType::F16, DType::F16);
    check_all(&gemm(h, DType::F16, DType::F16, Majors::KKN, (128, 128), (1, 2)), &[(128, 200, 128, 1)]);
    for c in ['n', 'm'] {
        check_all(&gemm(h, DType::F32, DType::F32, Majors::new('k', 'k', c), (128, 128), (2, 1)), &[(200, 128, 64, 1)]);
    }
}

/// FP8: E4M3 and E5M2 in each pairing (K-major only),
/// a K tile of 128, Float16, Float32 and E5M2 C.
#[test]
fn float8() {
    use DType::{F8E4M3FN as E4, F8E5M2 as E5};
    check_all(&gemm((E4, E4), DType::F16, DType::F32, Majors::KKN, (128, 128), (1, 1)), &[(128, 128, 256, 1)]);
    check_all(&gemm((E4, E5), DType::F32, DType::F32, Majors::KKN, (128, 128), (2, 1)), &[(256, 128, 144, 1)]);
    check_all(
        &gemm((E5, E4), DType::F16, DType::F16, Majors::new('k', 'k', 'm'), (64, 128), (1, 2)),
        &[(64, 256, 128, 1)],
    );
    check_all(&gemm((E5, E5), E5, DType::F32, Majors::KKN, (128, 64), (2, 1)), &[(200, 64, 128, 1)]);
    let g = gemm((E4, E4), DType::F16, DType::F32, Majors::KKN, (128, 128), (1, 1));
    assert_eq!(g.tile_shape_mnk, (128, 128, 128));
}

/// 8-bit integers with Int32 accumulators: Int32 C (stored with the
/// universal copy) and Float16 C from mixed-signedness inputs.
#[test]
fn int8() {
    use DType::{I8, U8};
    check_all(&gemm((I8, I8), DType::I32, DType::I32, Majors::KKN, (128, 128), (2, 1)), &[(256, 128, 128, 1)]);
    check_all(&gemm((U8, I8), DType::F16, DType::I32, Majors::KKN, (64, 64), (1, 2)), &[(64, 128, 160, 1)]);
}

/// Fractions, whose sums round: within the tolerance [`hopper_gemm::check`]
/// documents.
#[test]
fn fractions_within_tolerance() {
    let cases = [
        f16(Majors::KKN, (128, 128), (2, 1)),
        gemm((DType::BF16, DType::BF16), DType::BF16, DType::F32, Majors::KKN, (128, 128), (1, 2)),
        gemm((DType::F16, DType::F16), DType::F16, DType::F16, Majors::new('m', 'n', 'n'), (128, 64), (1, 1)),
        gemm((DType::F8E4M3FN, DType::F8E5M2), DType::F32, DType::F32, Majors::KKN, (128, 128), (1, 1)),
    ];
    for g in &cases {
        let kernel = g.build();
        let problem = Problem { data: Data::Fractions, ..Problem::new(136, 256, 256, 1) };
        run(g, &kernel, &problem);
    }
}

/// A Float16 accumulator over a long K (64 MMAs): its roundings add up to more than the
/// output's own, which the check allows for ([`hopper_gemm::f16_accumulator_rounding`]).
/// An H100 gives these bits, and a Float32 accumulator stays within the plain tolerance.
#[test]
fn float16_accumulator_over_a_long_k() {
    for acc in [DType::F16, DType::F32] {
        let g = gemm((DType::F16, DType::F16), DType::F16, acc, Majors::new('m', 'n', 'n'), (128, 64), (1, 1));
        let problem = Problem { data: Data::Fractions, ..Problem::new(128, 64, 1024, 1) };
        run(&g, &g.build(), &problem);
    }
}

/// The CTA swizzle: with more than eight clusters along M, the first eight
/// columns of clusters go through the regular groups and the rest through
/// the tail; every tile is still written once.
#[test]
fn cta_swizzle_regular_groups_and_tail() {
    check_all(&f16(Majors::KKN, (64, 64), (1, 1)), &[(64 * 10 - 8, 128, 64, 1)]);
    check_all(&f16(Majors::KKN, (64, 64), (2, 1)), &[(64 * 18, 64, 64, 1)]);
}

/// The 2x1 cluster under every built-in schedule of the simulator: the
/// mbarrier protocol holds whatever order the CTAs and warps run in.
#[test]
fn cluster_under_every_schedule() {
    let g = f16(Majors::KKN, (64, 64), (2, 1));
    let kernel = g.build();
    let problem = Problem::new(128, 64, 256, 1);
    let mut args = hopper_gemm::inputs(&g, &problem);
    kernel
        .simulate_all_orders(g.launch(128, 64, 1), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    hopper_gemm::check(&g, &problem, &args);
}

/// The empty barriers and the peer-CTA releases, which the tests above
/// barely reach (their K fits in the stages the example computes, so no
/// stage is reused): two stages and nine k-tiles, so every stage is
/// refilled four times, in each cluster shape under every built-in
/// schedule and seeded random ones. Too few releases hang, and too many
/// (a CTA releasing a stage it does not share, which the 2x2 cluster has)
/// let a producer refill a stage a sharer is still reading, which shows as
/// a race, a phase error or a wrong C.
#[test]
fn stage_reuse_across_clusters() {
    for cluster in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        let mut g = f16(Majors::KKN, (64, 64), cluster);
        g.ab_stage = 2;
        let kernel = g.build();
        let (m, n) = (64 * cluster.0, 64 * cluster.1);
        let problem = Problem::new(m, n, 64 * 9, 1);
        let mut args = hopper_gemm::inputs(&g, &problem);
        kernel
            .simulate_schedules(g.launch(m, n, 1), &mut args, &SimOptions::default(), &[1, 2, 3])
            .unwrap_or_else(|e| panic!("{} {cluster:?}: {e}", kernel.name()));
        hopper_gemm::check(&g, &problem, &args);
    }
}

/// What `_setup_attributes` derives, against the example's numbers.
#[test]
fn configuration_follows_the_example() {
    let g = f16(Majors::KKN, (128, 128), (2, 1));
    assert_eq!(g.tile_shape_mnk, (128, 128, 64));
    // (227 KiB - 1 KiB) / 32 KiB per stage.
    assert_eq!((g.smem_capacity, g.ab_stage, g.epi_stage), ((228 - 1) * 1024, 7, 4));
    assert_eq!((g.num_mcast_ctas_a, g.num_mcast_ctas_b, g.is_a_mcast, g.is_b_mcast), (1, 2, false, true));
    assert_eq!((g.atom_layout_mnk, g.threads_per_cta, g.epi_tile), ((1, 1, 1), 128, (64, 32)));
    let g = gemm((DType::F8E4M3FN, DType::F8E4M3FN), DType::F8E4M3FN, DType::F32, Majors::KKN, (64, 64), (1, 1));
    assert_eq!((g.tile_shape_mnk, g.epi_tile, g.ab_stage), ((64, 64, 128), (64, 64), 14));
    assert_eq!(
        HopperWgmmaGemmKernel::compute_grid((200, 136, 3), (128, 64, 64), (2, 2)),
        [2, 4, 3],
        "tiles rounded up to whole clusters"
    );
    // is_valid_dtypes and is_valid_tensor_alignment.
    use DType::*;
    assert!(HopperWgmmaGemmKernel::is_valid_dtypes(F16, F16, F32, F16, 'm', 'n'));
    assert!(!HopperWgmmaGemmKernel::is_valid_dtypes(F16, BF16, F32, F16, 'k', 'k'));
    assert!(!HopperWgmmaGemmKernel::is_valid_dtypes(F8E4M3FN, F8E4M3FN, F32, F16, 'm', 'k'));
    assert!(!HopperWgmmaGemmKernel::is_valid_dtypes(I8, F8E4M3FN, F32, F16, 'k', 'k'));
    assert!(!HopperWgmmaGemmKernel::is_valid_dtypes(I8, I8, F32, F16, 'k', 'k'));
    assert!(HopperWgmmaGemmKernel::is_valid_dtypes(U8, I8, I32, I32, 'k', 'k'));
    assert!(HopperWgmmaGemmKernel::is_valid_tensor_alignment(200, 136, 72, 1, F16, F16, Majors::KKN));
    assert!(!HopperWgmmaGemmKernel::is_valid_tensor_alignment(200, 136, 68, 1, F16, F16, Majors::KKN));
    assert!(!HopperWgmmaGemmKernel::is_valid_tensor_alignment(201, 136, 72, 1, F16, F16, Majors::new('m', 'k', 'n')));
    assert!(HopperWgmmaGemmKernel::is_valid_tensor_alignment(208, 136, 80, 1, F8E4M3FN, F16, Majors::KKN));
    assert!(!HopperWgmmaGemmKernel::is_valid_tensor_alignment(208, 136, 72, 1, F8E4M3FN, F16, Majors::KKN));
}

/// The shared storage struct: the mbarriers, then A and B (and C, when it
/// cannot alias A) at 1024-byte alignment.
#[test]
fn shared_storage_layout() {
    let l = hopper_gemm::SharedStorage::<F16, F16, F16>::layout(7, 7 * 128 * 64, 7 * 128 * 64, 0);
    assert_eq!(
        l.field_table(),
        ["mainloop_pipeline_array_ptr:112:0", "s_a:114688:1024", "s_b:114688:115712", "s_c:0:230400"]
    );
    assert_eq!(l.align_of(), 1024);
    // 8-bit A with 16-bit C: C's staging tiles get a member of their own.
    let g = gemm((DType::F8E4M3FN, DType::F8E4M3FN), DType::F16, DType::F32, Majors::KKN, (128, 128), (1, 1));
    assert!(!hopper_gemm::epi_reuses_smem_a(g.a_dtype, g.c_dtype));
}

/// Misuse the example reports: a
/// consumer count in threads instead of warps hangs (a consumer release is
/// one arrival per warp), a stage count beyond the shared memory fails the
/// target check, and a K that breaks TMA's alignment is refused before
/// launch.
#[test]
fn consumer_count_in_threads_hangs() {
    let mut g = f16(Majors::KKN, (64, 64), (1, 1));
    g.ab_stage = 2;
    g.consumers_in_threads = true;
    let kernel = g.build();
    let problem = Problem::new(64, 64, 64 * 10, 1);
    let mut args = hopper_gemm::inputs(&g, &problem);
    let e = kernel.simulate(g.launch(64, 64, 1), &mut args).expect_err("a hang").to_string();
    assert!(e.contains("the kernel did not terminate"), "{e}");
    assert!(e.contains("waits at mbarrier.try_wait"), "{e}");
}

/// Building the kernel checks it against its target.
#[test]
#[should_panic(expected = "bytes of shared memory, more than the 232448 bytes")]
fn stages_beyond_the_shared_memory_are_refused() {
    let mut g = f16(Majors::KKN, (128, 128), (1, 1));
    g.ab_stage += 2;
    g.build();
}

#[test]
fn misaligned_k_is_refused_before_launch() {
    let e = HopperWgmmaGemmKernel::can_implement(128, 128, 68, 1, DType::F16, DType::F16, DType::F16, Majors::KKN)
        .unwrap_err()
        .to_string();
    assert!(e.contains("[alignment]") && e.contains("K=68"), "{e}");
    // The contiguous mode of each operand is the one that counts.
    HopperWgmmaGemmKernel::can_implement(
        128,
        128,
        68,
        1,
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::new('m', 'n', 'n'),
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "CTA tile shape M must be 64/128")]
fn tile_m_must_be_64_or_128() {
    f16(Majors::KKN, (192, 128), (1, 1));
}

/// The kernel's instructions: WGMMA, multicast TMA loads, stmatrix, the
/// TMA stores, and the mbarrier arrivals on peer CTAs; it needs sm_90a.
#[test]
fn kernel_ir_and_target() {
    use cute::ir::Op;
    let kernel = f16(Majors::KKN, (128, 128), (2, 2)).build();
    let ir = kernel.ir();
    assert!(ir.count_ops(|op| matches!(op, Op::Wgmma { .. })) > 0);
    assert!(ir.count_ops(|op| matches!(op, Op::StMatrix { .. })) > 0);
    assert!(ir.count_ops(|op| matches!(op, Op::MbarrierArrive { peer: Some(_), .. })) > 0);
    let e = cute::validate::check_target(ir, Arch::Sm90).unwrap_err().to_string();
    assert!(e.contains("sm_90a"), "{e}");
    cute::validate::check_target(ir, Arch::Sm90a).unwrap();
}

/// The sm_90a PTX of a kernel per type family and cluster shape with each
/// installed `mlir-opt`, and, when `ptxas` is installed, its assembly for
/// sm_90a with `-v`: no warning that the WGMMAs were serialised, no spills,
/// and the register counts printed.
#[test]
fn ptx_and_ptxas_for_sm_90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    use DType::*;
    let kernels = [
        f16(Majors::KKN, (128, 128), (1, 1)),
        f16(Majors::KKN, (128, 128), (2, 1)),
        f16(Majors::new('m', 'n', 'm'), (128, 128), (1, 2)),
        f16(Majors::KKN, (128, 256), (2, 2)),
        gemm((BF16, BF16), BF16, F32, Majors::KKN, (128, 128), (2, 1)),
        gemm((F8E4M3FN, F8E5M2), F32, F32, Majors::KKN, (128, 128), (1, 2)),
        gemm((I8, I8), I32, I32, Majors::KKN, (128, 128), (2, 1)),
    ];
    for g in &kernels {
        let kernel = g.build();
        let name = kernel.name().to_string();
        let ptx = kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{name}: {e}"));
        let multicast = g.is_a_mcast || g.is_b_mcast;
        for (pattern, want) in [
            ("wgmma.mma_async.sync.aligned", true),
            (".multicast::cluster", multicast),
            ("stmatrix.sync.aligned", g.c_dtype.bits() == 16),
            ("fence.proxy.async.shared::cta", true),
            ("cp.async.bulk.commit_group", true),
            ("cp.async.bulk.wait_group.read", true),
            ("mbarrier.try_wait.parity", true),
            ("cp.async.bulk.tensor.3d.global.shared::cta", true),
            ("barrier.cluster.wait", multicast),
        ] {
            assert_eq!(ptx.contains(pattern), want, "{name}: {pattern}");
        }
        let Some(ptxas) = find_ptxas() else { continue };
        let path = std::env::temp_dir().join(format!("cute-hopper-gemm-{}-{name}.ptx", std::process::id()));
        std::fs::write(&path, &ptx).unwrap();
        let out = std::process::Command::new(&ptxas)
            .args(["-arch=sm_90a", "-v", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("running ptxas");
        let _ = std::fs::remove_file(&path);
        let log = String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{name}: ptxas failed:\n{log}");
        assert!(!log.contains("serialized"), "{name}: ptxas serialised the warpgroup MMAs:\n{log}");
        // No local memory: the accumulators stay in registers.
        assert!(log.contains("0 bytes spill stores, 0 bytes spill loads"), "{name}: ptxas spilled:\n{log}");
        let info = |w: &str| log.lines().filter(|l| l.contains(w)).map(str::trim).collect::<Vec<_>>().join(" ");
        eprintln!("{name}: {} {}", info("registers"), info("spill"));
        eprintln!(
            "{name}: PTX per kernel: {} wgmma.mma_async, {} multicast loads, {} stmatrix, {} TMA stores, {} try_wait",
            ptx.matches("wgmma.mma_async.sync.aligned").count(),
            ptx.matches(".multicast::cluster").count(),
            ptx.matches("stmatrix.sync.aligned").count(),
            ptx.matches("cp.async.bulk.tensor.3d.global.shared::cta").count(),
            ptx.matches("mbarrier.try_wait.parity").count(),
        );
    }
}

/// `ptxas` on `PATH`.
fn find_ptxas() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file())
}

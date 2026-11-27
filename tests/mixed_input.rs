//! The Blackwell mixed-input GEMM helpers: the host selectors
//! and validity checks, the Int4 conversions on the simulator against plain Rust, and the
//! reduced `mixed_input_gemm.py` port (`examples/blackwell_mixed_input_gemm.rs`) against a CPU
//! reference, with its sm_100a PTX assembled by `ptxas` when it is on PATH.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/blackwell_mixed_input_gemm.rs"]
mod blackwell_mixed_input_gemm;

use blackwell_mixed_input_gemm::{
    MIXED_K_MAJOR_CONVERT, MIXED_K_MAJOR_SCALE, MIXED_M_MAJOR_2CTA, MIXED_M_MAJOR_2CTA_SCALE, MIXED_M_MAJOR_CONVERT,
    MIXED_M_MAJOR_SCALE, MixedInputGemm, run, run_with,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, OperandSource, Repetition, St16x256bOp, St32x32bOp};
use cute::utils::LayoutEnum;
use cute::utils::mixed_input_helpers::{self, *};
use cute::utils::sm100::{Sm100TmaLoadOp, make_trivial_tiled_mma};

const K: OperandMajorMode = OperandMajorMode::K;
const MN: OperandMajorMode = OperandMajorMode::MN;

fn panic_text(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map(|_| ()).unwrap_err();
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// Selectors and checks (host).
// ---------------------------------------------------------------------------------------

#[test]
fn transform_source_mode_and_tma_kind_selectors() {
    assert_eq!(get_transform_a_source(K), OperandSource::TMEM);
    assert_eq!(get_transform_a_source(MN), OperandSource::SMEM);
    assert_eq!(TransformMode::from_granularity(0, 0), TransformMode::ConvertOnly);
    assert_eq!(TransformMode::from_granularity(1, 128), TransformMode::ConvertScale);
    // 2-CTA only for B; multicast as asked.
    for mcast in [false, true] {
        for use_2cta in [false, true] {
            for is_b in [false, true] {
                let op = get_tma_atom_kind(mcast, use_2cta, is_b);
                assert_eq!(op.is_multicast(), mcast, "{op:?}");
                let want = if use_2cta && is_b { CtaGroup::TWO } else { CtaGroup::ONE };
                assert_eq!(op.cta_group(), want, "mcast {mcast} 2cta {use_2cta} b {is_b}");
            }
        }
    }
    assert!(matches!(get_tma_atom_kind(false, true, true), Sm100TmaLoadOp::Tile(_)));
    assert!(SHUFFLE_MIN_PTX_ISA == 91 && shuffle_supported_by(91) && shuffle_supported_by(92));
    assert!(!shuffle_supported_by(90) && !shuffle_supported_by(88));
}

#[test]
fn copy_atom_a_transform_selection() {
    let shape = |m: usize| Shape::from(((m, 16), 1, 4));
    let st16: CopyOp = St16x256bOp::new(Repetition::X1).into();
    let st32: CopyOp = St32x32bOp::new(Repetition::X8).into();
    assert_eq!(select_copy_op_a_transform(false, OperandSource::TMEM, &shape(64)), st16);
    assert_eq!(select_copy_op_a_transform(true, OperandSource::TMEM, &shape(64)), st32);
    assert_eq!(select_copy_op_a_transform(false, OperandSource::TMEM, &shape(128)), st32);
    assert_eq!(select_copy_op_a_transform(true, OperandSource::TMEM, &shape(128)), st32);
    for m in [64, 128] {
        assert_eq!(select_copy_op_a_transform(false, OperandSource::SMEM, &shape(m)), CopyOp::Universal { bits: 32 });
    }
    // The typed atoms: the MMA's type into TMEM, A's into shared memory.
    match get_copy_atom_a_transform::<BF16, I4>(false, OperandSource::TMEM, &shape(128)) {
        TransformCopyAtom::Tmem(a) => assert_eq!(a.op(), st32),
        other => panic!("{other:?}"),
    }
    match get_copy_atom_a_transform::<BF16, I4>(false, OperandSource::SMEM, &shape(128)) {
        TransformCopyAtom::Smem(a) => assert_eq!(a.op(), CopyOp::Universal { bits: 32 }),
        other => panic!("{other:?}"),
    }
    // The shape must be A's partition shape.
    let e = panic_text(|| {
        select_copy_op_a_transform(false, OperandSource::TMEM, &Shape::from(128));
    });
    assert!(e.contains("partition shape"), "{e}");
}

#[test]
fn scale_granularity_and_shuffle_checks() {
    // 8-bit A: no scales.
    assert!(is_valid_scale_granularity(0, 0, DType::I8, 4096, 64));
    assert!(!is_valid_scale_granularity(1, 128, DType::I8, 4096, 64));
    assert!(!is_valid_scale_granularity(0, 128, DType::U8, 4096, 64));
    // 4-bit A: one scale per row, a K granule dividing K and a multiple of the tile's K.
    assert!(is_valid_scale_granularity(1, 256, DType::I4, 6144, 128));
    assert!(!is_valid_scale_granularity(2, 256, DType::I4, 6144, 128));
    assert!(!is_valid_scale_granularity(1, 0, DType::I4, 6144, 128));
    assert!(!is_valid_scale_granularity(1, 256, DType::I4, 6000, 128));
    assert!(!is_valid_scale_granularity(1, 192, DType::I4, 6144, 128));
    // Other widths are not checked.
    assert!(is_valid_scale_granularity(3, 5, DType::BF16, 7, 64));

    let bf = DType::BF16;
    assert!(is_shuffle_a(K, 256, DType::I4, bf, 128, true));
    assert!(!is_shuffle_a(K, 256, DType::I4, bf, 128, false), "not without toolchain support");
    assert!(!is_shuffle_a(MN, 256, DType::I4, bf, 128, true));
    assert!(!is_shuffle_a(K, 260, DType::I4, bf, 128, true));
    assert!(!is_shuffle_a(K, 256, DType::I8, bf, 128, true));
    assert!(!is_shuffle_a(K, 256, DType::I4, DType::F16, 128, true));
    assert!(!is_shuffle_a(K, 256, DType::I4, bf, 4, true));
}

#[test]
fn tensor_alignment_tiler_cluster_and_divisibility_checks() {
    let ok = |mnk, a_major, c: LayoutEnum, tiler, use_2cta, cluster, gm, gk| {
        is_valid_tensor_alignment(
            mnk,
            DType::I4,
            DType::BF16,
            DType::BF16,
            DType::BF16,
            a_major,
            K,
            c,
            tiler,
            use_2cta,
            cluster,
            gm,
            gk,
        )
    };
    let row = LayoutEnum::ROW_MAJOR;
    assert!(ok((256, 512, 8192), K, row, (128, 128, 64), false, (1, 1), 0, 0));
    // Int4 A takes 32 contiguous values per 16 bytes; BF16 8.
    assert!(!ok((256, 512, 8176), K, row, (128, 128, 64), false, (1, 1), 0, 0));
    assert!(ok((256, 512, 8176), MN, row, (128, 128, 64), false, (1, 1), 0, 0) == (8176 % 8 == 0));
    assert!(!ok((240, 512, 8192), MN, row, (128, 128, 64), false, (1, 1), 0, 0));
    assert!(!ok((256, 508, 8192), K, row, (128, 128, 64), false, (1, 1), 0, 0), "C rows unaligned");
    assert!(ok((256, 508, 8192), K, LayoutEnum::COL_MAJOR, (128, 128, 64), false, (1, 1), 0, 0));
    // The scales: (m, k / gk) M-major; and 128 bytes of scale rows per CTA.
    assert!(ok((1024, 512, 6144), K, row, (256, 128, 128), true, (2, 1), 1, 256));
    assert!(!ok((1028, 512, 6144), K, row, (256, 128, 128), true, (2, 1), 1, 256));
    assert!(ok((1024, 512, 6144), K, row, (128, 128, 128), false, (1, 1), 1, 256));
    assert!(ok((1024, 512, 6144), K, row, (128, 128, 128), false, (1, 2), 1, 256), "64 rows x 2 bytes");
    assert!(!ok((1024, 512, 6144), K, row, (128, 128, 128), false, (1, 4), 1, 256), "32 rows x 2 bytes");
    assert!(!ok((1024, 512, 6144), K, row, (64, 128, 128), false, (1, 2), 1, 256));

    assert!(is_valid_mma_tiler_and_cluster_shape((128, 128, 64), (1, 1), false));
    assert!(is_valid_mma_tiler_and_cluster_shape((64, 128, 64), (1, 1), false));
    assert!(is_valid_mma_tiler_and_cluster_shape((256, 128, 64), (2, 1), true));
    assert!(is_valid_mma_tiler_and_cluster_shape((128, 128, 64), (2, 2), true));
    assert!(!is_valid_mma_tiler_and_cluster_shape((256, 128, 64), (1, 1), true));
    assert!(!is_valid_mma_tiler_and_cluster_shape((256, 128, 64), (1, 1), false));
    assert!(!is_valid_mma_tiler_and_cluster_shape((512, 128, 64), (2, 1), true));

    assert_eq!(mixed_input_helpers::get_divisibility(4096, 128), 128);
    assert_eq!(mixed_input_helpers::get_divisibility(96, 128), 32);
    assert_eq!(mixed_input_helpers::get_divisibility(6, 128), 2);
    assert_eq!(mixed_input_helpers::get_divisibility(7, 128), 1);
    assert_eq!(mixed_input_helpers::get_divisibility(1, 128), 1);
    assert_eq!(mixed_input_helpers::get_divisibility(64, 16), 16);
    assert_eq!(mixed_input_helpers::get_divisibility(2, 128), 2);
}

#[test]
fn scale_layouts_and_smem_layouts() {
    // (M, K, L) = (256, 1024, 2), one scale per row and 128 K values.
    let g = get_gmem_layout_scale((256, 1024, 2), 1, 128, MN);
    let sc =
        |m: usize, k: usize, l: usize| g.offset(&Coord::from(((0, m), (k % 128, k / 128), l))).as_static().unwrap();
    for (m, k, l) in [(0, 0, 0), (5, 0, 0), (5, 127, 0), (5, 128, 0), (255, 1023, 1), (17, 300, 1)] {
        assert_eq!(sc(m, k, l), m + 256 * (k / 128) + 256 * 8 * l, "({m}, {k}, {l})");
    }
    assert_eq!(g.size(), 256 * 1024 * 2);
    assert_eq!(g.cosize(), 256 * 8 * 2);
    let gk = get_gmem_layout_scale((256, 1024, 1), 1, 128, K);
    let off = gk.offset(&Coord::from(((0, 7), (3, 2), 0))).as_static().unwrap();
    assert_eq!(off, 7 * 8 + 2, "K-major granules");

    // One CTA's (128, 64) tile with granules (1, 128): 128 scales per stage, 256 bytes.
    let (tile, per_stage, staged) = get_smem_layout_scale((128, 128, 64), false, 1, 128, MN, DType::BF16, 3);
    assert_eq!(tile, (128, 64));
    assert_eq!(per_stage.outer().to_string(), "((1,128),(64,1)):((0,1),(0,128))");
    assert_eq!(staged.outer().to_string(), "((1,128),(64,1),3):((0,1),(0,128),128)");
    let (tile, per_stage, _) = get_smem_layout_scale((256, 128, 128), true, 1, 64, MN, DType::BF16, 2);
    assert_eq!(tile, (128, 128));
    assert_eq!(per_stage.outer().to_string(), "((1,128),(64,2)):((0,1),(0,128))");
    let (_, per_stage, _) = get_smem_layout_scale((128, 128, 64), false, 1, 128, K, DType::BF16, 2);
    assert_eq!(per_stage.outer().to_string(), "((1,128),(64,1)):((0,1),(0,1))");
    // Under 128 bytes of scales per stage.
    let e = panic_text(|| {
        get_smem_layout_scale((64, 128, 64), false, 1, 64, MN, DType::I8, 2);
    });
    assert!(e.contains("under 128"), "{e}");

    // A as loaded (Int4), A transformed (the MMA's BFloat16) and B, with their stages.
    let mma = make_trivial_tiled_mma(
        DType::BF16,
        DType::BF16,
        K,
        K,
        DType::F32,
        CtaGroup::ONE,
        (128, 64),
        OperandSource::TMEM,
    );
    let (a, at, b) = compute_smem_layout(&mma, (128, 64, 64), DType::I4, DType::BF16, 3, 2);
    assert_eq!(a.outer().shape().mode(3).size(), 3);
    assert_eq!(at.outer().shape().mode(3).size(), 2);
    assert_eq!(b.outer().shape().mode(3).size(), 3);
    assert_eq!(size_in_bytes_dtype(DType::I4, &a).as_static(), Some(128 * 64 / 2 * 3));
    assert_eq!(size_in_bytes_dtype(DType::BF16, &at).as_static(), Some(128 * 64 * 2 * 2));
}

#[test]
fn host_shuffle_order() {
    let v: Vec<u8> = (0..16).collect();
    assert_eq!(shuffle_a_host(&v), [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15]);
}

// ---------------------------------------------------------------------------------------
// The conversions (simulator against plain Rust).
// ---------------------------------------------------------------------------------------

/// Values per thread: two groups of eight.
const PER: usize = 16;
const THREADS: usize = 32;

/// Thread `t`'s 16 Int4 values: all 16, rotated by `t`.
fn sweep() -> Vec<i8> {
    (0..THREADS).flat_map(|t| (0..PER).map(move |i| ((i + t) % 16) as i8 - 8)).collect()
}

/// A kernel of 32 threads, each converting its 16 Int4 values of `src` with `cvt` into the
/// 16 `D` of `dst`.
fn cvt_kernel<D: Numeric>(name: &str, cvt: impl Fn(&Tensor<I4>) -> TensorSsa<D> + 'static) -> Kernel {
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).target(Arch::Sm100a).build(move || {
        let src = param_tensor::<I4>("src", Layout::row_major((THREADS, PER)));
        let dst = param_tensor::<D>("dst", Layout::row_major((THREADS, PER)));
        let t = thread_idx().x;
        let frag = make_fragment::<I4>(PER);
        autovec_copy(&src.slice((t, ..)), &frag);
        let out = make_fragment::<D>(PER);
        out.store_ssa(&cvt(&frag));
        autovec_copy(&out, &dst.slice((t, ..)));
    })
}

fn simulate_cvt<D: Numeric + Copy + Default>(kernel: &Kernel, src: &[I4]) -> Vec<D> {
    let mut args = vec![KernelArg::buffer(src), KernelArg::buffer(&[D::default(); THREADS * PER])];
    kernel
        .simulate(LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    args[1].to_vec::<D>()
}

#[test]
fn cvt_tensor_a_to_bf16_matches_rust_for_every_int4() {
    let vals = sweep();
    let src: Vec<I4> = vals.iter().map(|&x| I4::new(x)).collect();
    let kernel = cvt_kernel::<BF16>("mixed_cvt_a_bf16", |f| cvt_tensor_a::<BF16>(f, false));
    let got = simulate_cvt::<BF16>(&kernel, &src);
    for (i, (g, &x)) in got.iter().zip(&vals).enumerate() {
        assert_eq!(*g, BF16::from_f32(x as f32), "value {i} ({x})");
    }
    // And to Float32 (the plain conversion).
    let kernel = cvt_kernel::<f32>("mixed_cvt_a_f32", |f| cvt_tensor_a::<f32>(f, false));
    let got = simulate_cvt::<f32>(&kernel, &src);
    assert!(got.iter().zip(&vals).all(|(g, &x)| *g == x as f32));
}

/// The shuffled path converts each group of eight from the order `0, 2, 1, 3, 4, 6, 5, 7`:
/// the host shuffle of the values gives them back in natural order.
#[test]
fn cvt_tensor_a_shuffled_matches_rust_in_natural_order() {
    let vals = sweep();
    let shuffled: Vec<I4> = shuffle_a_host(&vals).into_iter().map(I4::new).collect();
    let kernel = cvt_kernel::<BF16>("mixed_cvt_a_shuffle", |f| cvt_tensor_a::<BF16>(f, true));
    let got = simulate_cvt::<BF16>(&kernel, &shuffled);
    for (i, (g, &x)) in got.iter().zip(&vals).enumerate() {
        assert_eq!(*g, BF16::from_f32(x as f32), "value {i} ({x})");
    }
    // Unshuffled data through the shuffled path come out permuted.
    let plain: Vec<I4> = vals.iter().map(|&x| I4::new(x)).collect();
    let got = simulate_cvt::<BF16>(&kernel, &plain);
    let permuted = shuffle_a_host(&vals);
    assert!(got.iter().zip(&permuted).all(|(g, &x)| *g == BF16::from_f32(x as f32)));
    // The scaled .s2f6x2 cvt is in the kernel.
    assert!(kernel.to_mlir().contains("s2f6x2"), "the shuffled path uses the scaled cvt");
    let e = panic_text(|| {
        cvt_kernel::<F16>("mixed_cvt_a_shuffle_f16", |f| cvt_tensor_a::<F16>(f, true));
    });
    assert!(e.contains("BFloat16"), "{e}");
}

#[test]
fn cvt_tensor_a_mxf8_matches_rust_e4m3() {
    let vals = sweep();
    let src: Vec<I4> = vals.iter().map(|&x| I4::new(x)).collect();
    let kernel = cvt_kernel::<F8E4M3FN>("mixed_cvt_a_mxf8", |f| cvt_tensor_a_mxf8::<F8E4M3FN>(f));
    let got = simulate_cvt::<F8E4M3FN>(&kernel, &src);
    for (i, (g, &x)) in got.iter().zip(&vals).enumerate() {
        assert_eq!(g.0, F8E4M3FN::from_f32(x as f32).0, "value {i} ({x})");
    }
}

/// The bias encoding: nibble `u` (the value `u - 8`) becomes the byte `u ^ 8`.
#[test]
fn cvt_tensor_a_biased_matches_rust_bits() {
    let vals = sweep();
    let src: Vec<I4> = vals.iter().map(|&x| I4::new(x)).collect();
    let nibble = |x: i8| (x as u8) & 0xf;
    let kernel = cvt_kernel::<i8>("mixed_cvt_a_biased", |f| cvt_tensor_a_biased::<i8>(f));
    let got = simulate_cvt::<i8>(&kernel, &src);
    for (i, (g, &x)) in got.iter().zip(&vals).enumerate() {
        assert_eq!(*g as u8, nibble(x) ^ 8, "value {i} (nibble {:#x})", nibble(x));
    }
    // Bias-encoded data (u = x + 8) give x's two's-complement nibble.
    let biased: Vec<I4> = vals.iter().map(|&x| I4::wrapping(x as i64 + 8)).collect();
    let got = simulate_cvt::<i8>(&kernel, &biased);
    assert!(got.iter().zip(&vals).all(|(g, &x)| *g as u8 == nibble(x)));
    // Read as E4M3 bytes.
    let kernel = cvt_kernel::<F8E4M3FN>("mixed_cvt_a_biased_e4m3", |f| cvt_tensor_a_biased::<F8E4M3FN>(f));
    let got = simulate_cvt::<F8E4M3FN>(&kernel, &src);
    assert!(got.iter().zip(&vals).all(|(g, &x)| g.0 == nibble(x) ^ 8));
    // The lop3 and prmt of the encoding are in the kernel.
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("lop3") && mlir.contains("prmt"), "lop3/prmt expected");
}

// ---------------------------------------------------------------------------------------
// The reduced mixed-input GEMM.
// ---------------------------------------------------------------------------------------

/// Each configuration against the CPU, bit for bit, and its MMA and TMA counts: K-major A in
/// TMEM and M-major A in shared memory, convert only and scaled, one CTA and a CTA pair
/// (B with the 2-CTA load).
#[test]
fn mixed_input_gemm_matches_the_reference() {
    for (cfg, mnk) in [
        (MIXED_K_MAJOR_CONVERT, (128, 64, 128)),
        (MIXED_K_MAJOR_SCALE, (256, 128, 256)),
        (MIXED_M_MAJOR_CONVERT, (128, 128, 192)),
        (MIXED_M_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_2CTA, (256, 64, 128)),
        (MIXED_M_MAJOR_2CTA_SCALE, (256, 128, 256)),
    ] {
        let (m, n, k) = mnk;
        let report = run(&cfg, mnk);
        let v = if cfg.use_2cta_instrs { 2 } else { 1 };
        let mma_tiles = (m / cfg.mma_tiler_mnk.0) * (n / cfg.mma_tiler_mnk.1);
        let k_tiles = k / cfg.mma_tiler_mnk.2;
        // Four 16-deep MMAs per k-tile, on the leader CTA of each MMA tile.
        assert_eq!(report.tcgen05_mma_instructions as usize, mma_tiles * k_tiles * 4, "{}: {report}", cfg.name);
        // Per CTA and k-tile, one load of A and one of B (half of the pair's B tile each);
        // per scale granule, one load of scales.
        let scale_loads = k.checked_div(cfg.scale_granularity_k).unwrap_or(0);
        assert_eq!(
            report.tma_instructions as usize,
            mma_tiles * v * (2 * k_tiles + scale_loads),
            "{}: {report}",
            cfg.name
        );
    }
}

/// The shuffled path in the GEMM: the host shuffles A's K-major rows and the transform warps
/// convert them back (simulated; compiling it needs PTX ISA 9.1).
#[test]
fn mixed_input_gemm_with_shuffled_a() {
    let cfg = MixedInputGemm { name: "blackwell_mixed_input_gemm_k_shuffle", shuffle_a: true, ..MIXED_K_MAJOR_SCALE };
    run(&cfg, (128, 64, 256));
    assert!(cfg.build((128, 64, 256)).to_mlir().contains("s2f6x2"));
    // Not for M-major A.
    let bad = MixedInputGemm { shuffle_a: true, ..MIXED_M_MAJOR_SCALE };
    assert!(!bad.can_implement((128, 64, 256)));
}

/// The race detector stays clean and the results do not depend on the schedule.
#[test]
fn mixed_input_gemm_is_race_free_under_several_schedules() {
    let opts = SimOptions { detect_races: true, ..Default::default() };
    for (cfg, mnk) in [
        (MIXED_K_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_2CTA_SCALE, (256, 64, 256)),
    ] {
        run_with(&cfg, mnk, &opts, &[3, 11]);
    }
}

#[test]
fn mixed_input_gemm_refuses_what_it_cannot_implement() {
    assert!(!MIXED_K_MAJOR_CONVERT.can_implement((128, 64, 100)), "K not a tile multiple");
    assert!(!MIXED_K_MAJOR_SCALE.can_implement((128, 64, 192)), "K not a whole number of granules");
    assert!(!MIXED_M_MAJOR_2CTA.can_implement((128, 64, 128)), "M under the pair's tile");
    let e = panic_text(|| {
        MIXED_K_MAJOR_CONVERT.build((128, 64, 100));
    });
    assert!(e.contains("cannot implement"), "{e}");
}

/// The MLIR and the sm_100a PTX of each configuration (TMA loads, the TS-UMMA from TMEM or the
/// SS-UMMA, `tcgen05.st` of the transformed A, the 2-CTA B load), assembled by
/// `ptxas -arch=sm_100a` when it is on PATH.
#[test]
fn mixed_input_gemm_in_mlir_and_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for (cfg, mnk) in [
        (MIXED_K_MAJOR_CONVERT, (128, 64, 128)),
        (MIXED_K_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_CONVERT, (128, 64, 128)),
        (MIXED_M_MAJOR_2CTA_SCALE, (256, 64, 256)),
    ] {
        let kernel = cfg.build(mnk);
        let mlir = kernel.to_mlir();
        assert!(mlir.contains("tcgen05"), "{}: no tcgen05 in the MLIR", cfg.name);
        let compiled =
            kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
        let ptx = compiled.ptx();
        let group = if cfg.use_2cta_instrs { 2 } else { 1 };
        let mut wanted = vec![
            "cp.async.bulk.tensor".to_string(),
            format!("tcgen05.mma.cta_group::{group}.kind::f16"),
            format!("tcgen05.commit.cta_group::{group}"),
            "tcgen05.ld.sync.aligned".to_string(),
            "fma.rn.bf16x2".to_string(),
        ];
        if cfg.a_major == OperandMajorMode::K {
            wanted.push("tcgen05.st.sync.aligned.32x32b.x8".to_string());
            wanted.push("tcgen05.wait::st".to_string());
        } else {
            wanted.push("fence.proxy.async.shared::cta".to_string());
        }
        if cfg.use_2cta_instrs {
            wanted.push("cta_group::2".to_string());
        }
        for w in &wanted {
            assert!(ptx.contains(w.as_str()), "{}: no `{w}` in the PTX", cfg.name);
        }
        let ptxas = std::env::var_os("PATH")
            .and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
        let Some(ptxas) = ptxas else { continue };
        let base = std::env::temp_dir().join(format!("cute-mixed-input-{}-{}", std::process::id(), cfg.name));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, ptx).unwrap();
        let out =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{}: ptxas: {}{}",
            cfg.name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{}: ptxas -arch=sm_100a OK", cfg.name);
    }
    // The shuffled conversion needs PTX ISA 9.1, which LLVM 23 emits.
    if tc.llvm_version().is_some_and(|v| v >= 23) {
        let cfg =
            MixedInputGemm { name: "blackwell_mixed_input_gemm_k_shuffle", shuffle_a: true, ..MIXED_K_MAJOR_SCALE };
        let compiled = cfg
            .build((128, 64, 256))
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.ptx().contains("s2f6x2"), "the shuffled conversion's cvt");
        assert!(shuffle_supported_by(compiled.ptx_version()));
    }
}

/// The epilogue's global partitions: the TMA store's (shared and global, per epilogue tile)
/// and the SIMT store's (per thread), from an MMA-partitioned C.
#[test]
fn epilog_gmem_copy_and_partition_shapes() {
    let _ = Kernel::builder("mixed_epilog_partitions").block_dim([128, 1, 1]).target(Arch::Sm100a).build(|| {
        let c = param_tensor_aligned::<f32>("c", Layout::new((256, 128), (128, 1)), 32);
        let s_c = make_smem_tensor_aligned::<f32>(Layout::new((64, 32), (32, 1)), 1024);
        let (tma_atom_c, tma_tensor_c): (CopyAtom<f32>, Tensor<Ident>) = cute::cpasync::make_tiled_tma_atom(
            cute::cpasync::CopyBulkTensorTileS2GOp,
            &c,
            s_c.layout().clone(),
            (64, 32),
            1,
        )
        .into();
        let mma = make_trivial_tiled_mma(
            DType::BF16,
            DType::BF16,
            K,
            K,
            DType::F32,
            CtaGroup::ONE,
            (128, 64),
            OperandSource::SMEM,
        );
        let thr = mma.get_slice(0);
        // ((bM, bN), 1, 1, RestM, RestN)
        let t_cg_c_tma = thr.partition_c(&local_tile(&tma_tensor_c, (128, 64), (.., ..)));
        let t_cg_c = thr.partition_c(&local_tile(&c, (128, 64), (.., ..)));
        let t2r = make_tiled_copy_tv(
            &make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }),
            &Layout::row_major((64, 2)),
            &Layout::row_major((1, 16)),
        );
        let epi_tile: Tiler =
            Tree::Branch(vec![Tree::Leaf(Some(Layout::col_major(64))), Tree::Leaf(Some(Layout::col_major(32)))]);
        let p = epilog_gmem_copy_and_partition(
            thread_idx().x,
            Some(&tma_atom_c),
            Some(&t2r),
            Some(&t_cg_c_tma),
            Some(&t_cg_c),
            &epi_tile,
            &s_c,
        );
        // Per epilogue tile (2 x 2 of a 128 x 64 tile), per CTA tile (2 x 2 of C).
        let g = p.b_sg_g_c.expect("a TMA partition");
        assert_eq!(g.layout().mode(0).size(), 64 * 32);
        assert_eq!(g.size(), 256 * 128);
        assert_eq!(p.b_sg_s_c.expect("a TMA partition").size(), 64 * 32);
        assert!(p.tma_atom_c.is_some() && p.simt_atom.is_some());
        // Each of the 128 threads stores 16 values per epilogue tile.
        assert_eq!(p.t_tr_g_c.expect("a SIMT partition").size(), 256 * 128 / 128);
    });
}

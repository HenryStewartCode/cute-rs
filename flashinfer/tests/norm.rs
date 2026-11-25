// SPDX-License-Identifier: Apache-2.0

//! RMSNorm and fused add + RMSNorm on the simulator (with cute-rs's race and alignment
//! checks) against the f64 reference, for Float16 and BFloat16, and the PTX they lower to.

use cute::prelude::*;
use flashinfer::norm::*;

/// Rows that are a whole number of rounds (`d = 1024`, 128 threads), a partial last round
/// (`d = 1528`), one warp (`d = 8`, `d = 256`) and several warps.
#[test]
fn matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, d) in [(3usize, 8usize), (2, 256), (3, 1024), (2, 1528), (2, 2056)] {
            simulate(&Problem::new(dtype, batch, d, 5));
        }
    }
}

/// The multi-row geometry (FlashInfer's CuTe DSL one) and the one-block-per-row one agree
/// with the reference on both, with a batch that leaves a partial last block of rows, rows
/// of one warp (`tpr = 32`), several warps, and a partial last vector round.
#[test]
fn both_geometries_match_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, d) in [(5usize, 256usize), (3, 1528), (3, 4104), (2, 8192)] {
            for config in [Config::for_d(d), Config::row(d)] {
                let mut p = Problem::new(dtype, batch, d, 9);
                p.config = config;
                simulate(&p);
            }
        }
    }
    assert!(matches!(Config::for_d(3072), Config::Rows { tpr: 32, threads: 128, nvb: 12 }));
    assert!(matches!(Config::for_d(4096), Config::Rows { tpr: 64, threads: 128, nvb: 8 }));
    assert!(matches!(Config::for_d(8192), Config::Rows { tpr: 128, threads: 128, nvb: 8 }));
}

/// The tuned choices of `Config::choose` (wide rows at 8192, `tpr = 64..512` for the
/// quantising kernel) and the kernels they build agree with the reference.
#[test]
fn tuned_geometries_match_reference() {
    let rows = |tpr: usize, d: usize| Config::Rows { tpr, threads: tpr.max(128), nvb: (d / 8).div_ceil(tpr) };
    assert_eq!(Config::choose(Variant::Rms, 99, 8192), rows(512, 8192));
    assert_eq!(Config::choose(Variant::Rms, 19, 8192), rows(256, 8192));
    assert_eq!(Config::choose(Variant::Rms, 99, 3072), rows(128, 3072));
    assert_eq!(Config::choose(Variant::FusedAddQuant, 989, 3072), rows(64, 3072));
    assert_eq!(Config::choose(Variant::FusedAddQuant, 99, 4096), rows(256, 4096));
    assert_eq!(Config::choose(Variant::FusedAddQuant, 1, 3072), rows(512, 3072));
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (variant, batch, d) in [(Variant::Rms, 3usize, 1032usize), (Variant::FusedAddQuant, 3, 1032)] {
            for tpr in [64usize, 256, 512] {
                let mut p = Problem::new(dtype, batch, d, 13);
                p.config = rows(tpr, d);
                simulate_variant(&p, variant);
            }
        }
    }
}

#[test]
fn block_size_follows_flashinfer() {
    assert_eq!(num_warps(8), 1);
    assert_eq!(num_warps(3072), 12);
    assert_eq!(num_warps(4096), 16);
    assert_eq!(num_warps(8192), 32);
    assert_eq!(num_warps(16384), 32);
    assert_eq!(smem_bytes(Variant::FusedAdd, 4096), (16 + 4096) * 4);
}

/// 16-byte global accesses (`ld.global.v4.b32`) and shuffles.
#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for variant in [Variant::Rms, Variant::FusedAdd] {
        let compiled = Config::row(4096)
            .kernel(Dtype::F16, variant)
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
            .unwrap();
        let ptx = compiled.ptx();
        assert!(ptx.contains("ld.global.v4.b32"), "{ptx}");
        assert!(ptx.contains("shfl.sync.bfly"), "{ptx}");
        assert!(ptx.contains("rsqrt.approx"), "{ptx}");
    }
}

const NEW: [Variant; 3] = [Variant::GemmaRms, Variant::GemmaFusedAdd, Variant::LayerNorm];

/// `gemma_rmsnorm`, `gemma_fused_add_rmsnorm` and `layernorm` against their f64 references in
/// both geometries, on rows of one warp, several whole rounds and a partial last round.
#[test]
fn new_variants_match_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, d) in [(3usize, 8usize), (2, 256), (3, 1528), (2, 2056), (3, 4104)] {
            for config in [Config::for_d(d), Config::row(d)] {
                for variant in NEW {
                    let mut p = Problem::new(dtype, batch, d, 13);
                    p.config = config;
                    simulate_variant(&p, variant);
                }
            }
        }
    }
}

/// The Gemma kernels differ from the plain ones only by `weight + 1`: with a zero weight,
/// `gemma_rmsnorm` is plain normalisation.
#[test]
fn gemma_adds_one_to_the_weight() {
    let mut p = Problem::new(Dtype::F16, 2, 256, 4);
    p.weight = vec![0.0; 256];
    p.config = Config::row(256);
    let k = Config::row(256).kernel(Dtype::F16, Variant::GemmaRms);
    let mut args = p.args(Variant::GemmaRms);
    k.simulate(p.launch(Variant::GemmaRms), &mut args).unwrap();
    for (r, x) in p.dtype.read(&args[2]).chunks(256).zip(p.input.chunks(256)) {
        let rms = (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / 256.0 + p.eps as f64).sqrt();
        for (&a, &b) in r.iter().zip(x) {
            assert!((a as f64 - b as f64 / rms).abs() < 2e-3 * (1.0 + a.abs() as f64));
        }
    }
}

/// A constant row has zero variance: `layernorm` returns `beta` (the epsilon keeps it finite).
#[test]
fn layernorm_of_a_constant_row_is_beta() {
    let mut p = Problem::new(Dtype::BF16, 2, 256, 6);
    p.input = vec![1.5; 512];
    for config in [Config::row(256), Config::for_d(256)] {
        p.config = config;
        let k = config.kernel(p.dtype, Variant::LayerNorm);
        let mut args = p.args(Variant::LayerNorm);
        k.simulate(p.launch(Variant::LayerNorm), &mut args).unwrap();
        let out = p.dtype.read(&args[3]);
        let beta = p.beta();
        for (i, &o) in out.iter().enumerate() {
            assert!((o - p.dtype.round(beta[i % 256])).abs() < 1e-2, "{o} vs {}", beta[i % 256]);
        }
    }
}

#[test]
fn new_variants_lower_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for variant in NEW {
        for config in [Config::row(4096), Config::for_d(4096)] {
            let compiled = config
                .kernel(Dtype::BF16, variant)
                .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                .unwrap();
            let ptx = compiled.ptx();
            assert!(ptx.contains("ld.global.v4.b32"), "{variant:?} {ptx}");
            assert!(ptx.contains("st.global.v4.b32"), "{variant:?} {ptx}");
            assert!(ptx.contains("rsqrt.approx"), "{variant:?} {ptx}");
        }
    }
}

const QUANT: [Variant; 2] = [Variant::RmsQuant, Variant::FusedAddQuant];

/// `rmsnorm_quant` and `fused_add_rmsnorm_quant` against the f64 reference in both geometries
/// (the scale is small enough that some values saturate at 448).
#[test]
fn quant_variants_match_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, d) in [(3usize, 8usize), (2, 256), (3, 1528), (2, 2056), (3, 4104)] {
            for config in [Config::for_d(d), Config::row(d)] {
                for variant in QUANT {
                    let mut p = Problem::new(dtype, batch, d, 17);
                    p.config = config;
                    simulate_variant(&p, variant);
                }
            }
        }
    }
}

/// A huge input saturates to +-448 rather than becoming NaN, and the sign survives.
#[test]
fn quant_saturates() {
    let mut p = Problem::new(Dtype::F16, 1, 256, 2);
    p.scale = 1e-4;
    p.config = Config::row(256);
    let k = p.config.kernel(p.dtype, Variant::RmsQuant);
    let mut args = p.args(Variant::RmsQuant);
    k.simulate(p.launch(Variant::RmsQuant), &mut args).unwrap();
    let out = args[2].to_f32();
    assert!(out.iter().all(|v| v.abs() <= 448.0));
    assert!(out.iter().any(|v| v.abs() == 448.0));
}

#[test]
fn qk_geometry() {
    assert!(matches!(qk_config(128), Config::Rows { tpr: 16, threads: 128, nvb: 1 }));
    assert!(matches!(qk_config(64), Config::Rows { tpr: 8, threads: 128, nvb: 1 }));
    assert!(matches!(qk_config(520), Config::Rows { tpr: 32, threads: 128, nvb: 3 }));
}

/// `QKRMSNorm` on packed-QKV input (token stride `(heads + 2) * d`), with heads of one round,
/// a partial round and several rounds, and a job count that is not a multiple of the warps.
#[test]
fn qk_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, heads, d) in
            [(3usize, 5usize, 128usize), (2, 3, 64), (2, 2, 520), (1, 7, 8), (5, 9, 24), (2, 4, 1032)]
        {
            let p = Problem::qk(dtype, batch, heads, d, 21);
            simulate_variant(&p, Variant::QkRms);
        }
    }
}

#[test]
fn quant_and_qk_lower_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for variant in [Variant::RmsQuant, Variant::FusedAddQuant, Variant::QkRms] {
        let configs = if variant == Variant::QkRms {
            vec![qk_config(128), qk_config(1024)]
        } else {
            vec![Config::row(4096), Config::for_d(4096)]
        };
        for config in configs {
            let ptx = config
                .kernel(Dtype::BF16, variant)
                .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                .unwrap()
                .ptx()
                .to_string();
            assert!(ptx.contains("ld.global.v4.b32"), "{variant:?} {ptx}");
            assert!(ptx.contains("rsqrt.approx"), "{variant:?} {ptx}");
            if variant != Variant::QkRms {
                assert!(ptx.contains("e4m3x2"), "{variant:?} {ptx}");
            }
        }
    }
}

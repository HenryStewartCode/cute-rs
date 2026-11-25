// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `rmsnorm` and `fused_add_rmsnorm` ([`flashinfer::norm`]): simulated on small
//! rows, then lowered to PTX if a toolchain is found. With `--features cuda` and `GPU=1` it
//! runs on the GPU instead: correctness checks against the f64 reference, then CUDA-graph
//! timings on the shapes of FlashInfer's `benchmarks/bench_fused_add_rmsnorm.py` (batch
//! 1, 19, 99, 989; hidden 3072, 4096, 8192), with GB/s as that script counts it.
//!
//! `rmsnorm_quant` and `fused_add_rmsnorm_quant` (E4M3 out, scale 0.02) and `QKRMSNorm` (3-D input,
//! head_dim 128) run too. `gemma_rmsnorm`, `gemma_fused_add_rmsnorm` and `layernorm` (bf16 `input`, f32 `gamma` and
//! `beta`) run on the same shapes; `norm_flashinfer_bench.py` times FlashInfer's.
//!
//! `cargo run -p cute-flashinfer --example norm`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example norm`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::norm::gpu::{Kernels, run};
use flashinfer::norm::*;

#[cfg(feature = "cuda")]
const ALL: [Variant; 7] = [
    Variant::Rms,
    Variant::FusedAdd,
    Variant::GemmaRms,
    Variant::GemmaFusedAdd,
    Variant::LayerNorm,
    Variant::RmsQuant,
    Variant::FusedAddQuant,
];

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    // `NEW_ONLY=1` runs just the quantising and QK variants (a full run compiles ~300 kernels).
    let all: &[Variant] = if std::env::var_os("NEW_ONLY").is_some() { &ALL[5..] } else { &ALL };
    println!("device: {}", ctx.name());
    // `BENCH_ONLY=1` skips the correctness checks and the QKRMSNorm timings.
    let bench_only = std::env::var_os("BENCH_ONLY").is_some();
    for dtype in [Dtype::F16, Dtype::BF16].into_iter().filter(|_| !bench_only) {
        for (batch, d) in
            [(1usize, 8usize), (5, 1024), (7, 1528), (19, 3072), (99, 4096), (33, 8192), (99, 8192), (4, 16384)]
        {
            for &variant in all {
                for config in [Config::for_d(d), Config::row(d), Config::choose(variant, batch, d)] {
                    let mut p = Problem::new(dtype, batch, d, 11);
                    p.config = config;
                    let (err, _) = run(&ctx, &mut ks, &p, variant, true, false)?;
                    println!("check {} {variant:?} batch {batch} d {d}: max scaled error {err:.2e}", dtype.name());
                }
            }
        }
    }
    // QKRMSNorm (rmsnorm of a 3-D input): packed-QKV input, head_dim 128, Q and K head counts
    // of Llama-3-8B (32, 8) and 70B (64, 8), strided input, dense output.
    for dtype in [Dtype::F16, Dtype::BF16].into_iter().filter(|_| !bench_only) {
        for (batch, heads, d) in [(5usize, 32usize, 128usize), (3, 7, 520), (2, 8, 64)] {
            let p = Problem::qk(dtype, batch, heads, d, 11);
            let (err, _) = run(&ctx, &mut ks, &p, Variant::QkRms, true, false)?;
            println!("check {} QkRms batch {batch} heads {heads} d {d}: max scaled error {err:.2e}", dtype.name());
        }
    }
    println!("variant    dtype  batch  hidden  geometry   time(us)    GB/s");
    for dtype in [Dtype::F16, Dtype::BF16].into_iter().filter(|_| !bench_only) {
        for (heads, d) in [(32usize, 128usize), (8, 128), (64, 128)] {
            for batch in [1usize, 19, 99, 989] {
                let p = Problem::qk(dtype, batch, heads, d, 3);
                let (_, t) = run(&ctx, &mut ks, &p, Variant::QkRms, false, true)?;
                let t = t.unwrap();
                let bytes = (batch * heads * d * 4 + d * 2) as f64;
                println!(
                    "QkRms      {:5}  {batch:5}  {heads:2}x{d}  {:8}  {t:8.2}  {:7.1}",
                    dtype.name(),
                    "rows",
                    bytes / (t * 1e-6) / 1e9
                );
            }
        }
    }
    println!("variant    dtype  batch  hidden  geometry   time(us)    GB/s");
    // `BENCH_VARIANT=FusedAddQuant`, `BENCH_BATCH=989`, `BENCH_D=3072` narrow the timings.
    let only = |name: &str, v: &str| std::env::var(name).map_or(true, |s| s == v);
    for &variant in all.iter().filter(|v| only("BENCH_VARIANT", &format!("{v:?}"))) {
        for dtype in [Dtype::F16, Dtype::BF16] {
            for batch in [1usize, 19, 99, 989].into_iter().filter(|b| only("BENCH_BATCH", &b.to_string())) {
                for d in [3072usize, 4096, 8192].into_iter().filter(|d| only("BENCH_D", &d.to_string())) {
                    let mut p = Problem::new(dtype, batch, d, 3);
                    let (geo, config) = ("auto", Config::choose(variant, batch, d));
                    {
                        p.config = config;
                        let (_, t) = run(&ctx, &mut ks, &p, variant, false, true)?;
                        let t = t.unwrap();
                        let rows = (batch * d) as f64 * 2.0;
                        // The bytes bench_fused_add_rmsnorm.py counts: input and residual read
                        // and written, plus the weight; rmsnorm: input read, output written.
                        let bytes = match variant {
                            Variant::FusedAdd | Variant::GemmaFusedAdd => rows * 4.0 + (d * 2) as f64,
                            Variant::Rms | Variant::GemmaRms => rows * 2.0 + (d * 2) as f64,
                            // input (2 B) read, E4M3 (1 B) written; the fused one also moves the residual.
                            Variant::RmsQuant => rows * 1.5 + (d * 2) as f64,
                            Variant::FusedAddQuant => rows * 2.5 + (d * 2) as f64,
                            Variant::QkRms => unreachable!(),
                            Variant::LayerNorm => rows * 2.0 + (d * 8) as f64,
                        };
                        println!(
                            "{:9}  {:5}  {batch:5}  {d:6}  {geo:8}  {t:8.2}  {:7.1}",
                            format!("{variant:?}"),
                            dtype.name(),
                            bytes / (t * 1e-6) / 1e9
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (batch, d) in [(3usize, 8usize), (2, 1528), (2, 2056)] {
            let (a, b, c) = simulate(&Problem::new(dtype, batch, d, 7));
            println!(
                "{} batch {batch} d {d}: rmsnorm {a:.2e}, fused output {b:.2e}, fused residual {c:.2e}",
                dtype.name()
            );
            for v in [
                Variant::GemmaRms,
                Variant::GemmaFusedAdd,
                Variant::LayerNorm,
                Variant::RmsQuant,
                Variant::FusedAddQuant,
            ] {
                let (e, r) = simulate_variant(&Problem::new(dtype, batch, d, 7), v);
                println!("  {v:?}: output {e:.2e}, residual {r:.2e}");
            }
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            for variant in [
                Variant::Rms,
                Variant::FusedAdd,
                Variant::GemmaRms,
                Variant::GemmaFusedAdd,
                Variant::LayerNorm,
                Variant::RmsQuant,
                Variant::FusedAddQuant,
                Variant::QkRms,
            ] {
                let config = if variant == Variant::QkRms { qk_config(128) } else { Config::row(4096) };
                let k = config.kernel(Dtype::F16, variant);
                let ptx = k
                    .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .ptx()
                    .to_string();
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} 16-byte loads, {} shfl, {} fma",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.v4.b32").count(),
                    ptx.matches("shfl.sync").count(),
                    ptx.matches("fma.rn.f32").count()
                );
            }
            let (e, _) = simulate_variant(&Problem::qk(Dtype::F16, 3, 5, 128, 7), Variant::QkRms);
            println!("f16 QkRms 3 tokens x 5 heads x 128: output {e:.2e}");
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

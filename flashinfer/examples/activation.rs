// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `silu_and_mul`, `gelu_and_mul` and `gelu_tanh_and_mul`
//! ([`flashinfer::activation`]): simulated on small rows, then lowered to PTX if a
//! toolchain is found. With `--features cuda` and `GPU=1` it runs on the GPU instead:
//! correctness checks against the f64 reference, then CUDA-graph timings (one L2-resident
//! buffer pair) on tokens {1, 64, 4096} and output widths d {4096, 14336} (the range of
//! `benchmarks/bench_act_and_mul.py`), with GB/s as that script counts it.
//!
//! `cargo run -p cute-flashinfer --example activation`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example activation`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::{CudaContext, HardwareInfo};
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::activation::gpu::{Kernels, run};
use flashinfer::activation::*;

const ACTS: [Act; 3] = [Act::Silu, Act::Gelu, Act::GeluTanh];

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {} ({sms} SMs)", ctx.name());
    for dtype in [Dtype::F16, Dtype::BF16] {
        for act in ACTS {
            for (tokens, d) in [(1usize, 8usize), (1, 4096), (3, 1544), (5, 14336), (64, 4096), (100, 2056)] {
                let mut p = Problem::new(dtype, act, tokens, d, 11);
                p.num_sms = sms;
                let (err, _) = run(&ctx, &mut ks, &p, true, false)?;
                println!("check {} {} tokens {tokens} d {d}: max scaled error {err:.2e}", act.name(), dtype.name());
            }
        }
    }
    println!("act        dtype  tokens  d      split unroll  time(us)    GB/s");
    for act in ACTS {
        for dtype in [Dtype::F16, Dtype::BF16] {
            for d in [4096usize, 14336] {
                for tokens in [1usize, 64, 4096] {
                    let mut p = Problem::new(dtype, act, tokens, d, 3);
                    p.num_sms = sms;
                    let plan = p.plan();
                    let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
                    let t = t.unwrap();
                    println!(
                        "{:9}  {:5}  {tokens:6}  {d:5}  {:5} {:6}  {t:8.2}  {:7.1}",
                        act.name(),
                        dtype.name(),
                        plan.split,
                        plan.unroll,
                        p.bytes() / (t * 1e-6) / 1e9
                    );
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
        for act in ACTS {
            for (tokens, d) in [(3usize, 8usize), (2, 1544)] {
                let e = simulate(&Problem::new(dtype, act, tokens, d, 7));
                println!("{} {} tokens {tokens} d {d}: max scaled error {e:.2e}", act.name(), dtype.name());
            }
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            for act in ACTS {
                let k = kernel(Dtype::F16, act, 256, 2);
                let ptx = k
                    .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .ptx()
                    .to_string();
                println!(
                    "PTX {}: {} bytes, {} 16-byte loads, {} 16-byte stores, {} ex2, {} tanh",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.v4.b32").count(),
                    ptx.matches("st.global.v4.b32").count(),
                    ptx.matches("ex2.approx").count(),
                    ptx.matches("tanh.approx").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

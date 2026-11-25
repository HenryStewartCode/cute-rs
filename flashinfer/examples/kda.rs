// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's Kimi Delta Attention decode ([`flashinfer::kda`]): simulated on a small problem,
//! then lowered to PTX if a toolchain is found. With `--features cuda` and `GPU=1` it runs on
//! the GPU instead: correctness checks against the f64 reference, then CUDA-graph timings on
//! the shapes of FlashInfer's `benchmarks/bench_recurrent_kda.py` (16 q/k heads, 32 v heads,
//! K = V = 128, bf16 inputs and state, precomputed gate; T = 1 and T = 4 with per-token
//! checkpoint slots) and on Kimi-Linear's equal 32 heads. GB/s counts the state read and
//! written plus the inputs. Compare with `kda_flashinfer_bench.py`.
//!
//! `cargo run -p cute-flashinfer --example kda`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example kda`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::{CudaContext, HardwareInfo};
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::kda::gpu::{Kernels, run};
use flashinfer::kda::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {} ({sms} SMs)", ctx.name());
    let base = Options::default();
    if std::env::var_os("SWEEP").is_some() {
        for state in [StateDtype::F32, StateDtype::BF16] {
            for batch in [1usize, 16, 128] {
                for rpb in [128usize, 64, 32, 16, 8] {
                    for unroll in [1usize, 2, 4, 8] {
                        if !(rpb / WARPS).is_multiple_of(unroll) {
                            continue;
                        }
                        let mut p = Problem::new(Dtype::BF16, state, base, (batch, 16, 32, 128, 128), 3);
                        p.plan_override = Some((rpb, unroll));
                        let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
                        println!("sweep {} b{batch} rpb {rpb} unroll {unroll}: {:.2} us", state.name(), t.unwrap());
                    }
                }
            }
        }
        return Ok(());
    }
    let soft = Options { gate: Gate::Softplus, has_dt_bias: true, beta_is_logit: true, ..base };
    let lower = Options { gate: Gate::LowerBound, has_dt_bias: true, beta_is_logit: true, ..base };
    for input in [Dtype::F16, Dtype::BF16] {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            for (opts, shape) in [
                (base, (2usize, 4usize, 8usize, 128usize, 128usize)),
                (soft, (64, 16, 32, 128, 128)),
                (lower, (33, 4, 8, 128, 64)),
                (Options { has_indices: true, ..soft }, (33, 4, 8, 128, 64)),
                (Options { has_indices: true, has_out_indices: true, ..lower }, (13, 2, 8, 256, 32)),
                (Options { l2norm: false, ..base }, (5, 2, 4, 64, 12)),
                (Options { tokens: 4, has_indices: true, per_token_slots: true, ..lower }, (9, 2, 4, 128, 32)),
                (Options { tokens: 3, update_state: false, ..soft }, (7, 2, 4, 128, 32)),
            ] {
                let mut p = Problem::new(input, state, opts, shape, 11);
                p.num_sms = sms;
                let ((es, eo), _) = run(&ctx, &mut ks, &p, true, false)?;
                println!(
                    "check {} {} {shape:?} gate {:?} tokens {} idx {} out_idx {} per_token {}: state {es:.2e} out {eo:.2e}",
                    input.name(),
                    state.name(),
                    opts.gate,
                    opts.tokens,
                    opts.has_indices,
                    opts.has_out_indices,
                    opts.per_token_slots
                );
            }
        }
    }
    println!("heads(q/v)  T  batch  rows/blk unroll  time(us)    GB/s");
    for (hq, hv) in [(16usize, 32usize), (32, 32)] {
        for tokens in [1usize, 4] {
            for batch in [1usize, 16, 64, 128] {
                let opts = if tokens == 1 {
                    base
                } else {
                    Options { tokens, has_indices: true, per_token_slots: true, ..base }
                };
                let mut p = Problem::new(Dtype::BF16, StateDtype::BF16, opts, (batch, hq, hv, 128, 128), 3);
                p.num_sms = sms;
                let plan = p.plan();
                let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
                let t = t.unwrap();
                println!(
                    "{hq:3}/{hv:<3}   {tokens:3}  {batch:5}  {:8} {:6}  {t:8.2}  {:7.1}",
                    plan.rows_per_block,
                    plan.unroll,
                    p.bytes() / (t * 1e-6) / 1e9
                );
            }
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        if let Err(e) = gpu_main() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    let opts =
        Options { gate: Gate::Softplus, has_dt_bias: true, beta_is_logit: true, tokens: 2, ..Options::default() };
    let p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (3, 2, 4, 128, 16), 1);
    let (es, eo) = simulate(&p);
    println!("simulated KDA decode (2 tokens, softplus gate) matches the f64 reference: state {es:.2e}, out {eo:.2e}");
    if let Some(tc) = Toolchain::discover() {
        let ptx = p
            .kernel()
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
            .unwrap_or_else(|e| panic!("{e}"))
            .ptx()
            .to_string();
        println!("lowered to {} bytes of PTX", ptx.len());
    }
}

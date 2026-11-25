// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's Gated Delta Net decode and prefill ([`flashinfer::gdn`]): simulated on small problems,
//! then lowered to PTX if a toolchain is found. With `--features cuda` and `GPU=1` it runs on
//! the GPU instead: correctness checks against the f64 reference, then CUDA-graph timings on
//! Qwen3-Next's shape (16 q/k heads, 32 v heads, K = V = 128, bf16 inputs; f32 and bf16
//! state), GB/s counting the state read and written plus the inputs. Compare with
//! `gdn_flashinfer_bench.py`.
//!
//! `cargo run -p cute-flashinfer --example gdn`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example gdn`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::{CudaContext, HardwareInfo};
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::gdn::gpu::{Kernels, run};
use flashinfer::gdn::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {} ({sms} SMs)", ctx.name());
    if std::env::var("ONLY").is_ok_and(|s| s == "prefill") {
        return prefill_gpu(&ctx, &mut ks);
    }
    let base = Options::default();
    if std::env::var("ONLY").is_ok_and(|s| s == "extras") {
        return extras_gpu(&ctx, &mut ks, sms);
    }
    if std::env::var_os("SWEEP").is_some() {
        for state in [StateDtype::F32, StateDtype::BF16] {
            for batch in [1usize, 4, 16] {
                for rpb in [128usize, 64, 32, 16, 8] {
                    for unroll in [1usize, 2, 4, 8] {
                        if !(rpb / WARPS).is_multiple_of(unroll)
                            || (std::env::var_os("SWEEP_KV").is_some() && !unroll.is_multiple_of(16 / state.bytes()))
                        {
                            continue;
                        }
                        let tokens: usize = std::env::var("SWEEP_T").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
                        let opts = Options {
                            tokens,
                            update_state: tokens == 1,
                            kv_layout: std::env::var_os("SWEEP_KV").is_some(),
                            ..base
                        };
                        let mut p = Problem::new(Dtype::BF16, state, opts, (batch, 16, 32, 128, 128), 3);
                        p.plan_override = Some((rpb, unroll));
                        let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
                        println!("sweep {} b{batch} rpb {rpb} unroll {unroll}: {:.2} us", state.name(), t.unwrap());
                    }
                }
            }
        }
        return Ok(());
    }
    for input in [Dtype::F16, Dtype::BF16] {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            for (opts, shape) in [
                (base, (2usize, 4usize, 8usize, 128usize, 128usize)),
                (base, (64, 16, 32, 128, 128)),
                (Options { has_indices: true, ..base }, (33, 4, 8, 128, 64)),
                (Options { has_indices: true, has_out_indices: true, ..base }, (13, 2, 8, 256, 32)),
                (Options { l2norm: false, ..base }, (5, 2, 4, 64, 12)),
            ] {
                let mut p = Problem::new(input, state, opts, shape, 11);
                p.num_sms = sms;
                let ((es, eo), _) = run(&ctx, &mut ks, &p, true, false)?;
                println!(
                    "check {} {} {shape:?} idx {} out_idx {}: state {es:.2e} out {eo:.2e}",
                    input.name(),
                    state.name(),
                    opts.has_indices,
                    opts.has_out_indices
                );
            }
        }
    }
    // Multi-token checks (one input/state pair per state dtype, to keep the build time down).
    for (input, state) in
        [(Dtype::BF16, StateDtype::F32), (Dtype::F16, StateDtype::BF16), (Dtype::F16, StateDtype::F16)]
    {
        for (opts, shape) in [
            (Options { tokens: 4, cache_steps: 4, ..base }, (3, 4, 8, 128, 64)),
            (Options { tokens: 3, cache_steps: 5, update_state: false, has_indices: true, ..base }, (9, 2, 4, 128, 32)),
            (Options { tokens: 6, has_indices: true, has_out_indices: true, ..base }, (13, 2, 8, 256, 32)),
        ] {
            let mut p = Problem::new(input, state, opts, shape, 11);
            p.num_sms = sms;
            let ((es, eo), _) = run(&ctx, &mut ks, &p, true, false)?;
            println!(
                "check {} {} {shape:?} tokens {} cache {} update {}: state {es:.2e} out {eo:.2e}",
                input.name(),
                state.name(),
                opts.tokens,
                opts.cache_steps,
                opts.update_state
            );
        }
    }
    println!("state  batch  rows/blk unroll  time(us)    GB/s");
    for state in [StateDtype::F32, StateDtype::BF16] {
        for batch in [1usize, 4, 16, 64, 128] {
            let mut p = Problem::new(Dtype::BF16, state, base, (batch, 16, 32, 128, 128), 3);
            p.num_sms = sms;
            let plan = p.plan();
            let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
            let t = t.unwrap();
            println!(
                "{:5}  {batch:5}  {:8} {:6}  {t:8.2}  {:7.1}",
                state.name(),
                plan.rows_per_block,
                plan.unroll,
                p.bytes() / (t * 1e-6) / 1e9
            );
        }
    }
    // Multi-token (FlashInfer's `gated_delta_rule_mtp`): verify (no update), verify with the
    // intermediate-state cache, and update.
    println!("mode           T  batch  rows/blk unroll  time(us)    GB/s");
    for (mode, update_state, cached) in
        [("verify", false, false), ("verify+cache", false, true), ("update", true, false)]
    {
        for tokens in [2usize, 4] {
            for batch in [1usize, 16, 64] {
                let opts = Options { tokens, update_state, cache_steps: if cached { tokens } else { 0 }, ..base };
                let mut p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (batch, 16, 32, 128, 128), 3);
                p.num_sms = sms;
                let plan = p.plan();
                let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
                let t = t.unwrap();
                println!(
                    "{mode:12} {tokens:3}  {batch:5}  {:8} {:6}  {t:8.2}  {:7.1}",
                    plan.rows_per_block,
                    plan.unroll,
                    p.bytes() / (t * 1e-6) / 1e9
                );
            }
        }
    }
    extras_gpu(&ctx, &mut ks, sms)?;
    prefill_gpu(&ctx, &mut ks)
}

/// The non-transposed `[K, V]` state (`gated_delta_rule_decode`) and the per-token pool scatter
/// (`gated_delta_rule_mtp` with `ssm_state_indices`): checks, then CUDA-graph timings on
/// Qwen3-Next heads (f32 state, as FlashInfer's).
#[cfg(feature = "cuda")]
fn extras_gpu(ctx: &CudaContext, ks: &mut Kernels<'_>, sms: usize) -> Result<(), Box<dyn Error>> {
    let base = Options::default();
    let kv = Options { kv_layout: true, ..base };
    for state in [StateDtype::F32, StateDtype::BF16] {
        for (opts, shape) in [
            (kv, (5usize, 4usize, 8usize, 128usize, 128usize)),
            (Options { tokens: 3, has_indices: true, ..kv }, (9, 2, 4, 128, 64)),
            (Options { tokens: 4, per_token_slots: true, ..base }, (9, 2, 4, 128, 64)),
            (
                Options { tokens: 3, per_token_slots: true, has_indices: true, kv_layout: true, ..base },
                (9, 2, 4, 128, 64),
            ),
        ] {
            let mut p = Problem::new(Dtype::BF16, state, opts, shape, 11);
            p.num_sms = sms;
            let ((es, eo), _) = run(ctx, ks, &p, true, false)?;
            println!(
                "check {} {shape:?} tokens {} kv {} scatter: {}: state {es:.2e} out {eo:.2e}",
                state.name(),
                opts.tokens,
                opts.kv_layout,
                opts.per_token_slots
            );
        }
    }
    println!("mode            T  batch  rows/blk unroll  time(us)    GB/s");
    let scatter = |tokens| Options { tokens, per_token_slots: true, ..base };
    for (mode, tokens, opts) in
        [("kv [K,V]", 1usize, kv), ("kv [K,V]", 4, Options { tokens: 4, ..kv }), ("scatter", 4, scatter(4))]
    {
        for batch in [1usize, 4, 16, 64] {
            let mut p = Problem::new(Dtype::BF16, StateDtype::F32, opts, (batch, 16, 32, 128, 128), 3);
            p.num_sms = sms;
            let plan = p.plan();
            let (_, t) = run(ctx, ks, &p, false, true)?;
            let t = t.unwrap();
            println!(
                "{mode:12} {tokens:3}  {batch:5}  {:8} {:6}  {t:8.2}  {:7.1}",
                plan.rows_per_block,
                plan.unroll,
                p.bytes() / (t * 1e-6) / 1e9
            );
        }
    }
    Ok(())
}

/// Prefill (`chunk_gated_delta_rule`): checks, then CUDA-graph timings on Qwen3-Next's heads
/// (16 q/k, 32 v, K = V = 128, bf16 q/k/v, f32 gates and state, L2 norm, initial and final
/// state) at 1k, 4k and 16k tokens in one and in four sequences. `SKIP_CHECKS` skips the
/// checks.
#[cfg(feature = "cuda")]
fn prefill_gpu(ctx: &CudaContext, ks: &mut Kernels<'_>) -> Result<(), Box<dyn Error>> {
    use flashinfer::gdn::gpu::run_prefill;
    let base = PrefillOptions::default();
    if std::env::var_os("SKIP_CHECKS").is_none() {
        for input in [Dtype::F16, Dtype::BF16] {
            for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
                for (opts, lens, heads) in [
                    (base, vec![300usize, 0, 64, 7, 129], (2usize, 4usize, 128usize)),
                    (PrefillOptions { has_indices: true, has_out_indices: true, ..base }, vec![65, 1, 200], (2, 2, 64)),
                    (PrefillOptions { l2norm: false, has_initial_state: false, ..base }, vec![100, 33], (1, 2, 128)),
                ] {
                    let p = PrefillProblem::new(input, state, opts, &lens, heads, 11);
                    let ((es, eo), _) = run_prefill(ctx, ks, &p, true, false)?;
                    println!(
                        "check prefill {} {} {lens:?} {heads:?} l2 {} init {} idx {}: state {es:.2e} out {eo:.2e}",
                        input.name(),
                        state.name(),
                        opts.l2norm,
                        opts.has_initial_state,
                        opts.has_indices
                    );
                }
            }
        }
        // Qwen3-Next's heads, 1k tokens in four sequences.
        let p = PrefillProblem::new(Dtype::BF16, StateDtype::F32, base, &[256; 4], (16, 32, 128), 5);
        let ((es, eo), _) = run_prefill(ctx, ks, &p, true, false)?;
        println!("check prefill bf16 f32 4x256 (16, 32, 128): state {es:.2e} out {eo:.2e}");
    }
    println!("prefill tokens seqs  time(us)    GB/s  TFLOPS");
    for total in [1024usize, 4096, 16384] {
        for seqs in [1usize, 4] {
            let lens = vec![total / seqs; seqs];
            let p = PrefillProblem::new(Dtype::BF16, StateDtype::F32, base, &lens, (16, 32, 128), 3);
            let (_, t) = run_prefill(ctx, ks, &p, false, true)?;
            let t = t.unwrap();
            println!(
                "prefill {total:6} {seqs:4}  {t:8.1}  {:6.1}  {:6.2}",
                p.bytes() / (t * 1e-6) / 1e9,
                p.flops() / (t * 1e-6) / 1e12
            );
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
    let opts = Options { has_indices: true, has_out_indices: true, ..Options::default() };
    for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
        let mut p = Problem::new(Dtype::BF16, state, opts, (6, 2, 4, 128, 16), 7);
        p.num_sms = 2;
        let (es, eo) = simulate(&p);
        println!("bf16 {} pool: state error {es:.2e}, out error {eo:.2e}", state.name());
    }
    let p = PrefillProblem::new(Dtype::BF16, StateDtype::F32, PrefillOptions::default(), &[100, 9], (1, 2, 64), 7);
    let (es, eo) = simulate_prefill(&p);
    println!("bf16 f32 prefill {:?}: state error {es:.2e}, out error {eo:.2e}", p.seqlens);
    match Toolchain::discover() {
        Some(tc) => {
            let pk = prefill_kernel(Dtype::BF16, StateDtype::F32, PrefillOptions::default(), 128);
            let ptx = pk
                .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                .unwrap_or_else(|e| panic!("{e}"))
                .ptx()
                .to_string();
            if std::env::var_os("DUMP_PREFILL").is_some() {
                println!("{ptx}");
            }
            println!(
                "PTX {}: {} bytes, {} mma, {} ldmatrix, {} cp.async, {} bar, {} ex2, {} local",
                pk.name(),
                ptx.len(),
                ptx.matches("mma.sync").count(),
                ptx.matches("ldmatrix").count(),
                ptx.matches("cp.async.cg").count() + ptx.matches("cp.async.ca").count(),
                ptx.matches("bar.sync").count(),
                ptx.matches("ex2.approx").count(),
                ptx.matches(".local").count()
            );
            let p = Problem::new(Dtype::BF16, StateDtype::F32, Options::default(), (32, 16, 32, 128, 128), 1);
            let k = p.kernel();
            let ptx = k
                .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                .unwrap_or_else(|e| panic!("{e}"))
                .ptx()
                .to_string();
            if std::env::var_os("DUMP").is_some() {
                println!("{ptx}");
            }
            println!(
                "PTX {}: {} bytes, {} 16-byte loads, {} 16-byte stores, {} ex2, {} shfl, {} bar",
                k.name(),
                ptx.len(),
                ptx.matches("ld.global.v4.b32").count(),
                ptx.matches("st.global.v4.b32").count(),
                ptx.matches("ex2.approx").count(),
                ptx.matches("shfl.sync").count(),
                ptx.matches("bar.sync").count()
            );
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

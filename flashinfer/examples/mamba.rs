// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `selective_state_update` ([`flashinfer::mamba`]): simulated on small
//! problems, then lowered to PTX if a toolchain is found. With `--features cuda` and
//! `GPU=1` it runs on the GPU instead: correctness checks against the f64 reference, then
//! CUDA-graph timings on Mamba-2-like shapes (nheads 128, dim 64, dstate 128, ngroups 8,
//! bf16 inputs and f32 `dt`/`A`/`D`, with `dt_softplus`), GB/s counting the state read and
//! written plus the inputs. Compare with `mamba_flashinfer_bench.py`.
//!
//! `cargo run -p cute-flashinfer --example mamba`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example mamba`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::{CudaContext, HardwareInfo};
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::mamba::gpu::{Kernels, run};
use flashinfer::mamba::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let sms = HardwareInfo::new(0)?.get_device_multiprocessor_count()? as usize;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {} ({sms} SMs)", ctx.name());
    let base = Options::default();
    // `BENCH_ONLY=1` skips the checks and the f16/f32 state rows.
    let bench_only = std::env::var_os("BENCH_ONLY").is_some();
    for input in if bench_only { vec![] } else { vec![Dtype::F16, Dtype::BF16] } {
        for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
            for (opts, shape) in [
                (base, (2usize, 8usize, 64usize, 128usize, 2usize)),
                (Options { has_z: true, ..base }, (64, 32, 64, 128, 8)),
                (Options { has_indices: true, has_z: true, ..base }, (33, 16, 32, 256, 4)),
                (Options { has_d: false, has_dt_bias: false, dt_softplus: false, ..base }, (5, 4, 12, 64, 1)),
                (Options { tokens: 6, ..base }, (3, 8, 64, 128, 2)),
                (Options { tokens: 4, has_z: true, has_indices: true, ..base }, (33, 16, 32, 256, 4)),
            ] {
                let mut p = Problem::new(input, state, opts, shape, 11);
                p.num_sms = sms;
                let ((es, eo), _) = run(&ctx, &mut ks, &p, true, false)?;
                println!(
                    "check {} {} {shape:?} z {} idx {}: state {es:.2e} out {eo:.2e}",
                    input.name(),
                    state.name(),
                    opts.has_z,
                    opts.has_indices
                );
            }
        }
    }
    println!("state  batch  rows/blk unroll  time(us)    GB/s");
    let states: &[StateDtype] =
        if bench_only { &[StateDtype::BF16] } else { &[StateDtype::BF16, StateDtype::F16, StateDtype::F32] };
    for &state in states {
        for batch in [1usize, 16, 128] {
            let mut p = Problem::new(Dtype::BF16, state, base, (batch, 128, 64, 128, 8), 3);
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
    // Multi-token (FlashInfer's `cache_steps = 6`, no intermediate states).
    for &state in if bench_only { &[StateDtype::BF16][..] } else { &[StateDtype::BF16, StateDtype::F32][..] } {
        for batch in [1usize, 16, 128] {
            let opts = Options { tokens: 6, ..base };
            let mut p = Problem::new(Dtype::BF16, state, opts, (batch, 128, 64, 128, 8), 3);
            p.num_sms = sms;
            let plan = p.plan();
            let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
            let t = t.unwrap();
            println!(
                "{:5}  {batch:5}  {:8} {:6}  {t:8.2}  {:7.1}  (6 tokens)",
                state.name(),
                plan.rows_per_block,
                plan.unroll,
                p.bytes() / (t * 1e-6) / 1e9
            );
        }
    }
    // The options beyond the plain update: checked on a small shape, then timed at batch 16
    // (bf16 state, 6 tokens where there are several), against `mamba_flashinfer_bench.py`.
    let variants: [(&str, Options); 6] = [
        ("dst (1 token)", Options { has_indices: true, has_dst: true, ..base }),
        ("varlen", Options { varlen: true, tokens: 6, ..base }),
        ("accepted", Options { has_indices: true, has_accepted: true, tokens: 6, ..base }),
        ("istate", Options { intermediate: true, istate_indices: true, tokens: 6, ..base }),
        ("dst (6 tokens)", Options { has_indices: true, has_dst: true, tokens: 6, ..base }),
        ("int16", Options { int16: true, ..base }),
    ];
    println!("option           time(us)    GB/s");
    for (name, opts) in variants {
        let mut p = Problem::new(Dtype::BF16, StateDtype::F16, opts, (6, 8, 32, 128, 2), 5);
        p.num_sms = sms;
        let ((es, eo), _) = run(&ctx, &mut ks, &p, true, false)?;
        let state = if opts.int16 { StateDtype::F16 } else { StateDtype::BF16 };
        let mut p = Problem::new(Dtype::BF16, state, opts, (16, 128, 64, 128, 8), 3);
        p.num_sms = sms;
        // No pad slots when timing (FlashInfer's script has none): distinct sources, then
        // distinct destinations.
        let n = p.indices.len() as i32;
        p.indices = (0..n).collect();
        p.dst = (0..p.dst.len() as i32).map(|i| n + i).collect();
        let (_, t) = run(&ctx, &mut ks, &p, false, true)?;
        let t = t.unwrap();
        println!("{name:15} {t:8.2}  {:7.1}  (check: state {es:.2e}, out {eo:.2e})", p.bytes() / (t * 1e-6) / 1e9);
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    let opts = Options { has_z: true, has_indices: true, ..Options::default() };
    for state in [StateDtype::F16, StateDtype::BF16, StateDtype::F32] {
        let mut p = Problem::new(Dtype::BF16, state, opts, (6, 4, 16, 128, 2), 7);
        p.num_sms = 2;
        let (es, eo) = simulate(&p);
        println!("bf16 {} z + indices: state error {es:.2e}, out error {eo:.2e}", state.name());
    }
    match Toolchain::discover() {
        Some(tc) => {
            let p = Problem::new(Dtype::BF16, StateDtype::BF16, Options::default(), (128, 128, 64, 128, 8), 1);
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
                "PTX {}: {} bytes, {} 8-byte loads, {} 8-byte stores, {} ex2, {} shfl, {} bar",
                k.name(),
                ptx.len(),
                ptx.matches("ld.global.v2.b32").count(),
                ptx.matches("st.global.v2.b32").count(),
                ptx.matches("ex2.approx").count(),
                ptx.matches("shfl.sync").count(),
                ptx.matches("bar.sync").count()
            );
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

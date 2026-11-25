// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's Blackwell (SM100) ragged prefill ([`flashinfer::prefill_sm100`]): small
//! varlen problems on the simulator against the f64 reference, then, with `--features cuda`
//! and `GPU=1` on a B200, checked on the GPU and benchmarked from CUDA-graph replays on
//! batch 1/8, qo_len = kv_len 1K/4K/16K, 32 query / 8 KV heads, head dim 128, causal and not
//! (the shapes of `prefill_sm100_flashinfer_bench.py`).
//!
//! `cargo run -p cute-flashinfer --example prefill_sm100`; on a B200,
//! `GPU=1 cargo run --release -p cute-flashinfer --features cuda --example prefill_sm100`
//! (add `BENCH=1` for the timings).

use flashinfer::prefill_sm100::{Problem, simulate};

fn sim_main() {
    println!("simulator: ragged prefill, head dim 128, max |o - ref| and |lse - ref| (f64 reference)");
    for &(causal, bf16) in &[(false, false), (true, false), (false, true), (true, true)] {
        let p = Problem::new(&[300, 70], &[300, 200], 2, 1, causal, bf16, 7);
        let (eo, el) = simulate(&p, 3);
        println!(
            "  {} causal={causal:<5} qo [300, 70] kv [300, 200] heads 2/1: o {eo:.2e} lse {el:.2e}",
            if bf16 { "bf16" } else { "f16 " }
        );
    }
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn std::error::Error>> {
    use cute::cuda::CudaContext;
    use cute::prelude::*;
    use flashinfer::prefill_sm100::gpu::{Kernels, run};

    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt (set CUTE_MLIR_OPT)")?;
    println!("{}", ctx.name());
    let (hq, hkv) = (32, 8);
    // `PTX=file`: write the bf16 causal kernel's PTX (for `ptxas -arch=sm_100a -v`).
    if let Some(path) = std::env::var_os("PTX") {
        let cfg = Problem::new(&[128], &[128], hq, hkv, true, true, 0).config();
        let opts = CompileOptions::new().gpu_arch(Arch::Sm100a);
        std::fs::write(path, cfg.kernel().compile_with(&tc, &opts)?.ptx())?;
    }
    // Correctness on varlen batches (every 7th row checked against f64).
    for &bf16 in &[false, true] {
        for &causal in &[false, true] {
            let p = Problem::new(&[1000, 77, 4096, 513], &[1000, 300, 4096, 1700], hq, hkv, causal, bf16, 3);
            let ks = Kernels::new(&ctx, &tc, p.config())?;
            let (err, _) = run(&ctx, &ks, &p, 7, false)?;
            let (eo, el) = err.unwrap();
            println!(
                "check {} causal={causal:<5}: max |o - ref| {eo:.2e}, |lse - ref| {el:.2e}",
                if bf16 { "bf16" } else { "f16" }
            );
            assert!(eo < 2e-2 && el < 1e-2, "GPU result differs from the reference");
        }
    }
    if std::env::var_os("BENCH").is_none() {
        return Ok(());
    }
    println!("dtype causal batch  seq   gpu(us)  TFLOPS");
    for &bf16 in &[false, true] {
        for &causal in &[false, true] {
            let cfg = Problem::new(&[128], &[128], hq, hkv, causal, bf16, 0).config();
            let ks = Kernels::new(&ctx, &tc, cfg)?;
            for &batch in &[1usize, 8] {
                for &n in &[1024usize, 4096, 16384] {
                    let p = Problem::new(&vec![n; batch], &vec![n; batch], hq, hkv, causal, bf16, 1);
                    let (_, t) = run(&ctx, &ks, &p, 0, true)?;
                    let t = t.unwrap();
                    println!(
                        "{:5} {causal:6} {batch:5} {n:5} {t:9.1} {:7.1}",
                        if bf16 { "bf16" } else { "f16" },
                        p.flops() / (t * 1e-6) / 1e12
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
        gpu_main().unwrap();
        return;
    }
    sim_main();
}

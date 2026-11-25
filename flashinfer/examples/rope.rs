// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's RoPE with a cos/sin cache and its paged-KV append ([`flashinfer::rope`]):
//! simulated on small problems, then lowered to PTX if a toolchain is found. With
//! `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks, then
//! CUDA-graph timings (Llama-8B-like: 32 query heads, 8 KV heads, head dim 128) to compare
//! with `rope_flashinfer_bench.py`.
//!
//! `cargo run -p cute-flashinfer --example rope`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example rope`. `KV=e4m3` (or
//! `e5m2`) times the append into an FP8 cache instead (compare with
//! `rope_append_fp8_flashinfer_bench.py`).

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::rope::gpu::{AppendKernels, FreqKernels, RopeKernels, run_append, run_freq, run_rope};
use flashinfer::rope::*;

#[cfg(feature = "cuda")]
fn rope_row<T: Half, I: IndexType>(
    ctx: &CudaContext,
    ks: &RopeKernels<'_, T, I>,
    nnz: usize,
    nq: usize,
    nk: usize,
    hd: usize,
    interleave: bool,
) -> Result<(), Box<dyn Error>> {
    let p = Problem::<T, I>::new(nnz, nq, nk, hd, interleave, 4096.max(nnz), 9);
    let (pl, eq, ek, t) = run_rope(ctx, ks, &p, true, true)?;
    if !(eq < T::TOL && ek < T::TOL) {
        return Err(format!("rope out of tolerance: q {eq:.2e} k {ek:.2e}").into());
    }
    let t = t.unwrap();
    // q and k read and written in place; the cos/sin rows (4 bytes per element) are extra.
    let bytes = (2 * nnz * (nq + nk) * hd * 2) as f64;
    let with_cache = bytes + (nnz * hd * 4) as f64;
    println!(
        "{:5} {:5} {:4} {:4} {:3} {:>9} {:>10?}  {t:9.2}  {:8.1}  {:8.1}",
        T::NAME,
        if interleave { "gptj" } else { "neox" },
        nnz,
        nq,
        nk,
        format!("{:?}", pl.variant),
        pl.grid[..2].to_vec(),
        bytes / (t * 1e-6) / 1e9,
        with_cache / (t * 1e-6) / 1e9
    );
    Ok(())
}

/// `apply_rope_pos_ids_inplace` / `apply_llama31_rope_pos_ids_inplace`: checks, then timings
/// to compare with `rope_flashinfer_bench.py`.
#[cfg(feature = "cuda")]
fn freq_gpu(ctx: &CudaContext, tc: &Toolchain) -> Result<(), Box<dyn Error>> {
    println!("\nfrequency RoPE (pos_ids), f16, head dim 128, heads 32/8 (CUDA-graph replays)");
    println!("kind             rotary interleave   nnz  variant  graph(us)  |q|      |k|");
    for (label, rd, inter, cfg) in [
        ("rope", 128usize, false, FreqConfig::new(128)),
        ("rope", 128, true, FreqConfig::new(128)),
        ("rope rotary 64", 64, false, FreqConfig::new(64)),
        ("llama31", 128, false, FreqConfig::llama31(128)),
        ("llama31", 128, true, FreqConfig::llama31(128)),
    ] {
        let ks = FreqKernels::<F16, i32>::new(ctx, tc, 128, inter, cfg)?;
        for nnz in [1usize, 16, 512, 4096] {
            let p = Problem::<F16, i32>::new(nnz, 32, 8, 128, inter, 4096.max(nnz), 13);
            let (pl, eq, ek, t) = run_freq(ctx, &ks, &p, true, true)?;
            if !(eq < F16::TOL && ek < F16::TOL) {
                return Err(format!("{label} out of tolerance: q {eq:.2e} k {ek:.2e}").into());
            }
            println!("{label:16} {rd:6} {inter:9} {nnz:5}  {:>8?}  {:9.2}  {eq:.1e}  {ek:.1e}", pl.variant, t.unwrap());
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    // Correctness: both layouts, both dtypes, i32 and i64 positions, both head dims, with
    // token counts that pick each launch shape.
    macro_rules! check {
        ($t:ty, $i:ty, $hd:expr, $inter:expr) => {{
            let ks = RopeKernels::<$t, $i>::new(&ctx, &tc, $hd, $inter)?;
            for (nnz, nq, nk) in [(1usize, 32usize, 8usize), (13, 4, 2), (700, 32, 8), (5000, 32, 8), (4096, 8, 8)] {
                let p = Problem::<$t, $i>::new(nnz, nq, nk, $hd, $inter, 5000, 11);
                let (pl, eq, ek, _) = run_rope(&ctx, &ks, &p, true, false)?;
                println!(
                    "check {} {} hd {} {} nnz {nnz} heads {nq}/{nk}: {:?}, |q| {eq:.2e} |k| {ek:.2e}",
                    <$t as Half>::NAME,
                    <$i as IndexType>::NAME,
                    $hd,
                    if $inter { "gptj" } else { "neox" },
                    pl.variant
                );
                if !(eq < <$t as Half>::TOL && ek < <$t as Half>::TOL) {
                    return Err("out of tolerance".into());
                }
            }
        }};
    }
    check!(F16, i32, 128, false);
    check!(F16, i32, 128, true);
    check!(BF16, i64, 128, false);
    check!(BF16, i64, 128, true);
    check!(F16, i64, 64, false);
    check!(BF16, i32, 64, true);

    println!("\nRoPE in place, Llama-8B-like heads 32/8, head dim 128 (CUDA-graph replays)");
    println!("dtype layout  nnz   nq   nk   variant  grid(x,y)   graph(us)  GB/s(q,k)  GB/s(+cache)");
    let ks = RopeKernels::<F16, i32>::new(&ctx, &tc, 128, false)?;
    let ksj = RopeKernels::<F16, i32>::new(&ctx, &tc, 128, true)?;
    let ksb = RopeKernels::<BF16, i32>::new(&ctx, &tc, 128, false)?;
    for nnz in [1usize, 16, 512, 4096] {
        rope_row(&ctx, &ks, nnz, 32, 8, 128, false)?;
        rope_row(&ctx, &ksj, nnz, 32, 8, 128, true)?;
        rope_row(&ctx, &ksb, nnz, 32, 8, 128, false)?;
    }
    println!("FlashInfer's benchmarks/bench_rope.py shape: heads 32/32, head dim 128, batch 2");
    for nnz in [2usize * 128, 2 * 1024, 2 * 4096] {
        rope_row(&ctx, &ks, nnz, 32, 32, 128, false)?;
    }

    freq_gpu(&ctx, &tc)?;

    // Paged append: correctness (exact), then FlashInfer's bench_append_paged_kv_cache.py shapes.
    println!();
    let aks = AppendKernels::<F16>::new(&ctx, &tc, 8, 128)?;
    for (seq, app, page) in [
        (vec![5usize, 17, 1, 40], vec![5usize, 3, 1, 9], 4usize),
        (vec![16, 33, 2], vec![1, 1, 1], 16),
        (vec![1000, 777], vec![1000, 700], 16),
    ] {
        let p = AppendProblem::<F16>::new(&seq, &app, page, 8, 128, 2);
        let (bad, _) = run_append(&ctx, &aks, &p, true, false)?;
        println!("check append batch {} page {page}: {bad} mismatching elements", seq.len());
        if bad != 0 {
            return Err("append mismatch".into());
        }
    }
    let aks_bf = AppendKernels::<BF16>::new(&ctx, &tc, 8, 128)?;
    let p = AppendProblem::<BF16>::new(&[100, 50], &[100, 20], 16, 8, 128, 2);
    let (bad, _) = run_append(&ctx, &aks_bf, &p, true, false)?;
    println!("check append bf16: {bad} mismatching elements");
    if bad != 0 {
        return Err("append mismatch".into());
    }
    // FP8 caches (FP8 inputs): exact.
    let aks_e4 = AppendKernels::<F8E4M3FN>::new(&ctx, &tc, 8, 128)?;
    let aks_e5 = AppendKernels::<F8E5M2>::new(&ctx, &tc, 8, 128)?;
    let bad =
        run_append(&ctx, &aks_e4, &AppendProblem::<F8E4M3FN>::new(&[1000, 50], &[999, 1], 16, 8, 128, 2), true, false)?
            .0
            + run_append(
                &ctx,
                &aks_e5,
                &AppendProblem::<F8E5M2>::new(&[100, 50], &[100, 20], 16, 8, 128, 2),
                true,
                false,
            )?
            .0;
    println!("check append e4m3, e5m2: {bad} mismatching elements");
    if bad != 0 {
        return Err("append mismatch".into());
    }
    match std::env::var("KV").as_deref() {
        Ok("e4m3") => append_timings(&ctx, &aks_e4),
        Ok("e5m2") => append_timings(&ctx, &aks_e5),
        _ => append_timings(&ctx, &aks),
    }
}

/// FlashInfer's bench_append_paged_kv_cache.py shapes, 8 KV heads, head dim 128, page 16.
#[cfg(feature = "cuda")]
fn append_timings<T: CacheElem>(ctx: &CudaContext, aks: &AppendKernels<'_, T>) -> Result<(), Box<dyn Error>> {
    println!("\nAppend to the paged cache, 8 KV heads, head dim 128, {}, page 16 (CUDA-graph replays)", T::CACHE_NAME);
    println!("batch  tokens  grid  graph(us)  GB/s (K,V read + write)");
    let mut shapes: Vec<(Vec<usize>, String)> = vec![];
    for n in [1usize, 16, 512, 4096] {
        shapes.push((vec![n], format!("{n} tokens, 1 request")));
    }
    let seqlen = 5000usize;
    shapes.push((vec![1; 8], "decode x8".into()));
    shapes.push((std::iter::once(seqlen - 7).chain(std::iter::repeat_n(1, 7)).collect(), "prefill + 7 decode".into()));
    shapes.push((vec![seqlen / 8; 8], "8 x 625".into()));
    for (app, label) in shapes {
        let p = AppendProblem::<T>::new(&app, &app, 16, 8, 128, 4);
        let (_, t) = run_append(ctx, aks, &p, false, true)?;
        let t = t.unwrap();
        let bytes = (p.nnz() * 8 * 128 * 2 * 2 * T::BYTES) as f64;
        println!(
            "{:5}  {:6}  {:4}  {t:9.2}  {:8.1}   ({label})",
            app.len(),
            p.nnz(),
            aks.grid(p.nnz()),
            bytes / (t * 1e-6) / 1e9
        );
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for interleave in [false, true] {
        let p = Problem::<F16, i32>::new(19, 4, 2, 128, interleave, 300, 5);
        for (variant, split) in [(Variant::HeadParallel, 1), (Variant::Unsplit, 1), (Variant::Split, 3)] {
            let (eq, ek) = simulate(&p, variant, split);
            println!(
                "rope f16 hd 128 {} {variant:?}: |q| {eq:.2e} |k| {ek:.2e}",
                if interleave { "gptj" } else { "neox" }
            );
            assert!(eq < F16::TOL && ek < F16::TOL, "out of tolerance");
        }
    }
    for (rd, cfg) in [(128, FreqConfig::new(128)), (64, FreqConfig::new(64)), (128, FreqConfig::llama31(128))] {
        let p = Problem::<F16, i32>::new(19, 4, 2, 128, false, 300, 5);
        let (eq, ek) = simulate_freq(&p, &cfg, Variant::Unsplit, 1);
        println!(
            "freq rope f16 rotary {rd} {}: |q| {eq:.2e} |k| {ek:.2e}",
            if cfg.llama31.is_some() { "llama31" } else { "" }
        );
        assert!(eq < F16::TOL && ek < F16::TOL, "out of tolerance");
    }
    let p = AppendProblem::<F16>::new(&[5, 17, 1, 40], &[5, 3, 1, 9], 4, 2, 128, 3);
    let bad = simulate_append(&p, 3);
    println!("append (4 requests, {} tokens): {bad} mismatching elements", p.nnz());
    assert_eq!(bad, 0);
    match Toolchain::discover() {
        Some(tc) => {
            let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
            let kernels = [
                kernel::<F16, i32>(128, false, Variant::Unsplit),
                kernel::<F16, i32>(128, true, Variant::Split),
                kernel::<F16, i32>(128, false, Variant::HeadParallel),
                append_kernel::<F16>(8, 128),
            ];
            for k in kernels {
                let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} ld.global.v4, {} st.global.v4, {} fma",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.v4").count(),
                    ptx.matches("st.global.v4").count(),
                    ptx.matches("fma.rn.f32").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's batch decode over a paged KV cache ([`flashinfer::decode`]): simulated on
//! two small problems, split and unsplit, then lowered to PTX if a toolchain is found.
//! With `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks
//! against the f64 reference, then timings on shapes from FlashInfer's
//! `benchmarks/bench_batch_decode.py`.
//!
//! `cargo run -p cute-flashinfer --example batch_decode`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example batch_decode`.
//! `KV=e4m3` or `KV=e5m2` picks an FP8 K/V cache (Float16 by default); compare with
//! `batch_decode_fp8_flashinfer_bench.py`. With a Float16 cache it then checks and times
//! the attention variants (soft-cap 30, window 1024; compare with
//! `attention_variants_flashinfer_bench.py`). `HD=64` or `HD=256` picks another head
//! dimension and `DT=bf16` BFloat16 queries, cache and output (compare with
//! `batch_decode_hd_flashinfer_bench.py`); the variants then run only with `VARIANTS=1`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::decode::gpu::{Kernels, run};
use flashinfer::decode::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let kv = kv_dtype();
    let (hd, bf16) = shape();
    let ks = Kernels::with_shape(&ctx, &tc, STAGES, false, kv, Variant::NONE, hd, bf16)?;
    // Problems of this shape: head dimension `hd`, BFloat16 if asked, the cache quantised to `kv`.
    let make = |lens: &[usize], ps: usize, h: usize, seed: u64| {
        let p = Problem::new_hd(lens, ps, h, hd, seed);
        (if bf16 { p.bf16() } else { p }).quantized(kv, 1.0, 1.0)
    };
    println!(
        "device: {}, K/V {kv:?}, head dim {hd}, {}, max concurrent decode CTAs: {}",
        ctx.name(),
        if bf16 { "bf16" } else { "f16" },
        ks.max_grid
    );
    let tol_o = if bf16 { 8e-3 } else { 2e-3 };
    // Correctness, split and unsplit, including realistic sizes.
    for (lens, ps, h) in [
        (vec![1usize, 7, 21, 40], 4usize, 2usize),
        (vec![16, 3, 33], 16, 1),
        ((0..8).map(|i| 300 + 397 * i).collect::<Vec<_>>(), 16, 8),
        (vec![4096], 16, 8),
        ((0..64).map(|i| 50 + 31 * i).collect::<Vec<_>>(), 16, 8),
    ] {
        let p = make(&lens, ps, h, 11);
        let (plan, eo, el, _) = run(&ctx, &ks, &p, true, false)?;
        println!(
            "check batch {} page {ps} kv heads {h}: split {} ({} CTAs x {h}), max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
            lens.len(),
            plan.split,
            plan.num_ctas()
        );
        if !(eo < tol_o && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    if kv != KvDtype::F16 {
        // FlashInfer's k_scale (folded into the softmax scale) and v_scale.
        let p = Problem::new(&[700, 33, 2049], 16, 8, 5).quantized(kv, 0.5, 2.0);
        let (plan, eo, el, _) = run(&ctx, &ks, &p, true, false)?;
        println!("check k_scale 0.5 v_scale 2: split {}, |o - ref| = {eo:.2e}, |lse - ref| = {el:.2e}", plan.split);
        if !(eo < 4e-3 && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    // Timings: page size 16, 8 KV heads x 4 (32 query heads).
    println!("batch  kv_len  split  ctas   time(us)  graph(us)  decode(us)   GB/s (K+V read, graph)");
    for batch in [1usize, 16, 64] {
        for kv_len in [1024usize, 4096] {
            let p = make(&vec![kv_len; batch], 16, 8, 3);
            let (plan, _, _, t) = run(&ctx, &ks, &p, false, true)?;
            let (t, tg, td) = t.unwrap();
            let bytes = (2 * batch * kv_len * 8 * hd * kv.bytes()) as f64;
            println!(
                "{batch:5}  {kv_len:6}  {:5}  {:4}  {t:9.1}  {tg:9.1}  {td:9.1}  {:8.1}",
                plan.split,
                plan.num_ctas(),
                bytes / (tg * 1e-6) / 1e9
            );
        }
    }
    // The variants run in the default shape, or in another with `VARIANTS=1`.
    if kv == KvDtype::F16 && ((hd == HEAD_DIM && !bf16) || std::env::var_os("VARIANTS").is_some()) {
        variants(&ctx, &tc, hd, bf16)?;
    }
    Ok(())
}

/// The attention variants on the GPU: checks against the f64 reference, then CUDA-graph
/// timings on the shapes above (soft-cap 30, window 1024).
#[cfg(feature = "cuda")]
fn variants(ctx: &CudaContext, tc: &Toolchain, hd: usize, bf16: bool) -> Result<(), Box<dyn Error>> {
    let new = |lens: &[usize], ps: usize, h: usize, seed: u64| {
        let p = Problem::new_hd(lens, ps, h, hd, seed);
        if bf16 { p.bf16() } else { p }
    };
    let tol_o = if bf16 { 8e-3 } else { 2e-3 };
    type Make = fn(Problem) -> Problem;
    let cases: [(&str, Make); 4] = [
        ("soft-cap 30", |p| p.with_soft_cap(30.0)),
        ("window 1024", |p| p.with_window(1024)),
        ("alibi", |p| p.with_alibi()),
        ("soft-cap 5 + window 100", |p| p.with_soft_cap(5.0).with_window(100)),
    ];
    for (name, make) in cases {
        let ks = Kernels::with_shape(
            ctx,
            tc,
            STAGES,
            false,
            KvDtype::F16,
            make(Problem::new(&[1], 16, 1, 1)).variant,
            hd,
            bf16,
        )?;
        for (lens, ps, h) in [
            ((0..8).map(|i| 300 + 397 * i).collect::<Vec<_>>(), 16usize, 8usize),
            (vec![4096], 16, 8),
            (vec![1, 7, 21, 1500], 4, 2),
        ] {
            let p = make(new(&lens, ps, h, 11));
            let (plan, eo, el, _) = run(ctx, &ks, &p, true, false)?;
            println!(
                "check {name}, batch {}: split {}, |o - ref| = {eo:.2e}, |lse - ref| = {el:.2e}",
                lens.len(),
                plan.split
            );
            if !(eo < tol_o && el < 1e-3) {
                return Err("out of tolerance".into());
            }
        }
        if name.contains('+') || name == "alibi" {
            continue;
        }
        println!("{name}: batch  kv_len  split  ctas  graph(us)");
        for batch in [1usize, 16, 64] {
            for kv_len in [1024usize, 4096] {
                let p = make(new(&vec![kv_len; batch], 16, 8, 3));
                let (plan, _, _, t) = run(ctx, &ks, &p, false, true)?;
                println!(
                    "{name}: {batch:5}  {kv_len:6}  {:5}  {:4}  {:9.1}",
                    plan.split,
                    plan.num_ctas(),
                    t.unwrap().1
                );
            }
        }
    }
    Ok(())
}

/// The head dimension from `HD` (128 by default) and whether `DT=bf16`.
#[cfg(feature = "cuda")]
fn shape() -> (usize, bool) {
    let hd = std::env::var("HD").ok().map_or(HEAD_DIM, |v| v.parse().expect("HD is 64, 128 or 256"));
    (hd, std::env::var("DT").is_ok_and(|v| v == "bf16"))
}

/// The K/V cache type from `KV` (`f16`, `e4m3` or `e5m2`).
#[cfg(feature = "cuda")]
fn kv_dtype() -> KvDtype {
    match std::env::var("KV").as_deref() {
        Ok("e4m3") => KvDtype::E4M3,
        Ok("e5m2") => KvDtype::E5M2,
        _ => KvDtype::F16,
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    // Lengths cover a single token, partial and exact last pages, and longer requests;
    // each problem runs unsplit (a large grid) and split (a small one).
    for (lens, page_size, heads) in [(vec![1usize, 7, 21, 40], 4usize, 2usize), (vec![16, 3, 33], 16, 1)] {
        let p = Problem::new(&lens, page_size, heads, 7);
        // A large grid leaves requests whole; a small one with 1-page minimum chunks splits.
        for (max_grid, min_chunk) in [(1000, 32), (12, 1)] {
            let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
            println!(
                "batch {:?}, page size {page_size}, {heads} KV heads x {GROUP}, split {} (chunk {} pages, {} CTAs): \
                 max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
                lens,
                plan.split,
                plan.chunk_pages,
                plan.num_ctas()
            );
            assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
        }
        // The same problem over an E4M3 cache (scaled), against the dequantised cache.
        let p = Problem::new(&lens, page_size, heads, 7).quantized(KvDtype::E4M3, 0.5, 2.0);
        let (plan, eo, el) = simulate(&p, 12, 1);
        println!(
            "  E4M3 cache, k_scale 0.5, v_scale 2, split {}: max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
            plan.split
        );
        assert!(eo < 4e-3 && el < 1e-3, "out of tolerance");
    }
    // Head dimensions 64 and 256, and BFloat16, split and unsplit.
    for (hd, bf16) in [(64usize, false), (256, false), (128, true), (64, true), (256, true)] {
        let p = Problem::new_hd(&[1, 7, 21, 40], 4, 2, hd, 7);
        let p = if bf16 { p.bf16() } else { p };
        let (plan, eo, el) = simulate(&p, 12, 1);
        println!(
            "  head dim {hd}, {}, split {}: max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
            if bf16 { "bf16" } else { "f16" },
            plan.split
        );
        assert!(eo < if bf16 { 8e-3 } else { 2e-3 } && el < 1e-3, "out of tolerance");
    }
    match Toolchain::discover() {
        Some(tc) => {
            for k in [
                kernel(false),
                kernel(true),
                merge_kernel(),
                kernel_variant_kv(false, STAGES, false, KvDtype::E4M3),
                kernel_variant_kv(true, STAGES, false, KvDtype::E4M3),
            ] {
                let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
                let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
                // `PTX_DIR=dir` also writes each kernel's PTX there, for `ptxas -v` or SASS.
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} cp.async, {} ex2.approx, {} fma",
                    k.name(),
                    ptx.len(),
                    ptx.matches("cp.async.cg").count(),
                    ptx.matches("ex2.approx").count(),
                    ptx.matches("fma.rn.f32").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

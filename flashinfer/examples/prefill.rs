// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's batch prefill over a paged KV cache ([`flashinfer::prefill`]): simulated on
//! small ragged problems (each query tile size, causal and not, split and unsplit), then
//! lowered to PTX if a toolchain is found. With `--features cuda` and `GPU=1` it runs on
//! the GPU instead: correctness checks against the f64 reference, then CUDA-graph timings
//! on the shapes of `prefill_flashinfer_bench.py` (batch {1, 8} x qo_len = kv_len {512,
//! 2048, 8192}, causal, page 16, 32 query / 8 KV heads, f16).
//!
//! `SKIP_CHECKS=1` skips the reference checks (use `--release` for long timing runs) and
//! `VARIANTS_ONLY=1` skips the plain-causal timings.
//!
//! `cargo run -p cute-flashinfer --example prefill`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example prefill`. On the GPU it
//! then checks and times the attention variants (causal with window 1024, causal with
//! soft-cap 30, a causal custom mask; compare with `attention_variants_flashinfer_bench.py`),
//! and an FP8 (E4M3 and E5M2) KV cache against Float16 on the shapes above (compare with
//! `prefill_fp8_flashinfer_bench.py`); `FP8_ONLY=1` runs only the FP8 part.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::prefill::gpu::{Kernels, run};
use flashinfer::prefill::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new(&ctx, tc)?;
    println!("device: {}, planner grid: {} CTAs", ctx.name(), ks.max_grid);
    if std::env::var_os("FP8_ONLY").is_some() {
        return fp8(&ks);
    }
    // Correctness: every tile size, causal and not, GQA and 1:1, split and unsplit.
    for (qo, kv, page, heads, group, causal) in [
        (vec![3usize, 1, 4], vec![70usize, 9, 130], 8usize, 2usize, 4usize, true),
        (vec![9, 13, 30], vec![37, 13, 300], 16, 8, 4, false),
        (vec![9, 13, 30], vec![37, 13, 300], 16, 8, 4, true),
        (vec![150, 70, 333], vec![160, 75, 400], 16, 8, 1, true),
        (vec![256, 100], vec![256, 1000], 16, 8, 4, true),
        (vec![512], vec![512], 16, 8, 4, false),
    ] {
        if std::env::var_os("SKIP_CHECKS").is_some() {
            break;
        }
        let p = Problem::new(&qo, &kv, page, heads, group, causal, 11);
        let (plan, eo, el, _) = run(&ks, &p, true, None)?;
        println!(
            "check qo {qo:?} kv {kv:?} page {page} heads {}/{heads} causal {causal}: tile {} split {} ({} CTAs x {heads}), \
             max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
            heads * group,
            plan.tile.cta_tile_q,
            plan.split,
            plan.num_ctas()
        );
        if !(eo < 2e-3 && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    // Timings: causal, page size 16, 8 KV heads x 4 (32 query heads), head dim 128, f16.
    println!("batch  qo=kv  causal  tile  split  ctas     gpu(us)   TFLOPS");
    let timings = std::env::var_os("VARIANTS_ONLY").is_none();
    for batch in if timings { vec![1usize, 8] } else { vec![] } {
        for n in [512usize, 2048, 8192] {
            let p = Problem::new(&vec![n; batch], &vec![n; batch], 16, 8, 4, true, 3);
            let iterations = if batch * n * n <= 8 * 2048 * 2048 { 100 } else { 20 };
            let (plan, _, _, t) = run(&ks, &p, false, Some(iterations))?;
            let t = t.unwrap();
            println!(
                "{batch:5}  {n:5}  {:6}  {:4}  {:5}  {:4}  {t:10.1}  {:7.2}",
                true,
                plan.tile.cta_tile_q,
                plan.split,
                plan.num_ctas(),
                p.flops() / (t * 1e-6) / 1e12
            );
        }
    }
    variants(&ks)?;
    fp8(&ks)
}

/// An FP8 KV cache (`k_scale` 0.5, `v_scale` 1.5) with Float16 queries: checks against the
/// f64 reference on the dequantised cache, paged and ragged, then CUDA-graph timings of
/// Float16, E4M3 and E5M2 caches on the shapes above.
#[cfg(feature = "cuda")]
fn fp8(ks: &Kernels<'_>) -> Result<(), Box<dyn Error>> {
    let skip = std::env::var_os("SKIP_CHECKS").is_some();
    for kv in [KvDtype::E4M3, KvDtype::E5M2] {
        for (qo, kvl, causal) in [
            (vec![3usize, 1, 4], vec![70usize, 9, 1300], true),
            (vec![9, 13, 30], vec![37, 13, 300], false),
            (vec![150, 70, 333], vec![1600, 75, 400], true),
            (vec![256, 100], vec![256, 1000], true),
        ] {
            if skip {
                continue;
            }
            for ragged in [false, true] {
                let p = if ragged {
                    Problem::new_ragged(&qo, &kvl, 8, 4, causal, 11)
                } else {
                    Problem::new(&qo, &kvl, 16, 8, 4, causal, 11)
                };
                let p = p.quantized(kv, 0.5, 1.5);
                let (plan, eo, el, _) = run(ks, &p, true, None)?;
                println!(
                    "check {kv:?} {} qo {qo:?} causal {causal}: tile {} kv tile {} split {}, |o - ref| = {eo:.2e}, |lse - ref| = {el:.2e}",
                    if ragged { "ragged" } else { "paged" },
                    plan.tile.cta_tile_q,
                    plan.tile.cta_tile_kv(),
                    plan.split
                );
                if !(eo < 3e-3 && el < 1e-3) {
                    return Err("out of tolerance".into());
                }
            }
        }
    }
    println!("kv      batch  qo=kv  tile  kvtile  split  ctas     gpu(us)   TFLOPS");
    for batch in [1usize, 8] {
        for n in [512usize, 2048, 8192] {
            for kv in [KvDtype::F16, KvDtype::E4M3, KvDtype::E5M2] {
                let p = Problem::new(&vec![n; batch], &vec![n; batch], 16, 8, 4, true, 3).quantized(kv, 0.5, 1.5);
                let iterations = if batch * n * n <= 8 * 2048 * 2048 { 100 } else { 20 };
                let (plan, _, _, t) = run(ks, &p, false, Some(iterations))?;
                let t = t.unwrap();
                println!(
                    "{:6}  {batch:5}  {n:5}  {:4}  {:6}  {:5}  {:4}  {t:10.1}  {:7.2}",
                    format!("{kv:?}"),
                    plan.tile.cta_tile_q,
                    plan.tile.cta_tile_kv(),
                    plan.split,
                    plan.num_ctas(),
                    p.flops() / (t * 1e-6) / 1e12
                );
            }
        }
    }
    Ok(())
}

/// The attention variants on the GPU: checks against the f64 reference, then CUDA-graph
/// timings on the shapes above.
#[cfg(feature = "cuda")]
fn variants(ks: &Kernels<'_>) -> Result<(), Box<dyn Error>> {
    type Make = fn(Problem) -> Problem;
    let cases: [(&str, Make); 5] = [
        ("window 1024", |p| p.with_window(1024)),
        ("soft-cap 30", |p| p.with_soft_cap(30.0)),
        ("mask", |p| p.with_custom_mask(5)),
        ("alibi", |p| p.with_alibi()),
        ("mask + window 70 + cap 5", |p| p.with_custom_mask(6).with_window(70).with_soft_cap(5.0)),
    ];
    let skip = std::env::var_os("SKIP_CHECKS").is_some();
    for (name, make) in cases {
        for (qo, kv, causal) in [
            (vec![3usize, 1, 4], vec![70usize, 9, 1300], true),
            (vec![9, 13, 30], vec![37, 13, 300], false),
            (vec![150, 70, 333], vec![1600, 75, 400], true),
            (vec![256, 100], vec![256, 1000], true),
        ] {
            if skip {
                continue;
            }
            let p = make(Problem::new(&qo, &kv, 16, 8, 4, causal, 11));
            let (plan, eo, el, _) = run(ks, &p, true, None)?;
            println!(
                "check {name} qo {qo:?} causal {causal}: tile {} split {}, |o - ref| = {eo:.2e}, |lse - ref| = {el:.2e}",
                plan.tile.cta_tile_q, plan.split
            );
            if !(eo < 2e-3 && el < 1e-3) {
                return Err("out of tolerance".into());
            }
        }
    }
    // Timings: causal with window 1024, causal with soft-cap 30, and a causal pattern as a
    // custom mask (a non-causal kernel reading the bits; up to 2048 tokens).
    let causal_mask = |n: usize, batch: usize| -> Vec<bool> {
        (0..batch).flat_map(|_| (0..n).flat_map(move |i| (0..n).map(move |t| t <= i))).collect()
    };
    println!("variant      batch  qo=kv  tile  split  ctas     gpu(us)");
    for (name, make) in cases.iter().take(3) {
        for batch in [1usize, 8] {
            for n in [512usize, 2048, 8192] {
                let mask = *name == "mask";
                if mask && n > 2048 {
                    continue;
                }
                let base = Problem::new(&vec![n; batch], &vec![n; batch], 16, 8, 4, !mask, 3);
                let p = if mask { base.with_mask_bits(causal_mask(n, batch)) } else { make(base) };
                let iterations = if batch * n * n <= 8 * 2048 * 2048 { 100 } else { 20 };
                let (plan, _, _, t) = run(ks, &p, false, Some(iterations))?;
                println!(
                    "{name:11}  {batch:5}  {n:5}  {:4}  {:5}  {:4}  {:10.1}",
                    plan.tile.cta_tile_q,
                    plan.split,
                    plan.num_ctas(),
                    t.unwrap()
                );
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
    // Each query tile size (16, 64, 128 rows), causal and not; each problem runs whole
    // (one CTA per KV head of grid, which never splits) and split (a large grid with
    // 1-page chunks, then the merge).
    for (qo, kv, page, heads, group) in [
        (vec![3usize, 1, 4], vec![70usize, 9, 130], 8usize, 2usize, 4usize),
        (vec![9, 13], vec![37, 13], 4, 1, 4),
        (vec![150, 70], vec![160, 75], 16, 1, 1),
    ] {
        for causal in [false, true] {
            let p = Problem::new(&qo, &kv, page, heads, group, causal, 7);
            for grid in [heads, 1000] {
                let (plan, eo, el) = simulate(&p, grid, 1);
                println!(
                    "qo {qo:?} kv {kv:?} page {page} heads {}/{heads} causal {causal}: tile {} split {} \
                     (chunk {} tokens, {} CTAs): max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
                    heads * group,
                    plan.tile.cta_tile_q,
                    plan.split,
                    plan.kv_chunk_size,
                    plan.num_ctas()
                );
                assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
            }
        }
    }
    // An FP8 (E4M3) KV cache with scales, whole and split.
    for kv in [KvDtype::E4M3, KvDtype::E5M2] {
        let p = Problem::new(&[9, 13], &[37, 13], 4, 1, 4, true, 7).quantized(kv, 0.5, 1.5);
        for grid in [1, 1000] {
            let (plan, eo, el) = simulate(&p, grid, 1);
            println!(
                "{kv:?} kv: tile {} split {}: max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
                plan.tile.cta_tile_q, plan.split
            );
            assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            for cta_tile_q in [128, 64, 16] {
                let spec = KernelSpec {
                    tile: TileConfig::sm89(cta_tile_q),
                    group: 4,
                    page_size: 16,
                    causal: true,
                    partition: false,
                    variant: Variant::NONE,
                };
                let k = kernel(&spec);
                let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
                let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
                // `PTX_DIR=dir` also writes each kernel's PTX there, for `ptxas -v` or SASS.
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} mma.sync, {} ldmatrix, {} cp.async, {} ex2.approx, {} B shared",
                    k.name(),
                    ptx.len(),
                    ptx.matches("mma.sync").count(),
                    ptx.matches("ldmatrix").count(),
                    ptx.matches("cp.async.cg").count(),
                    ptx.matches("ex2.approx").count(),
                    spec.tile.smem_bytes()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

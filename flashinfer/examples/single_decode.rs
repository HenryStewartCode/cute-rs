// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `single_decode_with_kv_cache` ([`flashinfer::decode`]): the decode kernel
//! over contiguous K/V, under FlashInfer's single-decode split rule, simulated on small
//! problems; with `--features cuda` and `GPU=1`, checked on the GPU against the f64
//! reference and timed from CUDA graphs at kv_len {1024, 8192, 32768}, 32 query / 8 KV
//! heads, head dim 128, f16, NHD (as `single_decode_flashinfer_bench.py`).
//!
//! `cargo run -p cute-flashinfer --example single_decode`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example single_decode`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::decode::gpu::{Kernels, launch, single_decode_with_kv_cache as gpu_single};
use flashinfer::decode::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new_ragged(&ctx, &tc)?;
    println!("device: {}, max concurrent decode CTAs: {}", ctx.name(), ks.max_grid);
    for (kv_len, heads) in [(1usize, 1usize), (200, 2), (1000, 8), (4097, 8), (32768, 8)] {
        let p = Problem::new_ragged(&[kv_len], heads, 11);
        let (o, lse) = gpu_single(&ctx, &ks, p.q.clone(), p.k_cache.clone(), p.v_cache.clone(), heads)?;
        let (want_o, want_lse) = p.reference();
        let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        let plan = p.single_plan(ks.max_grid);
        println!("check kv_len {kv_len} kv heads {heads}: {} chunks: |o| {eo:.2e}, |lse| {el:.2e}", plan.num_ctas());
        if !(eo < 2e-3 && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    println!("kv_len  chunks   graph(us)  decode(us)   GB/s (K+V read, graph)");
    for kv_len in [1024usize, 8192, 32768] {
        let p = Problem::new_ragged(&[kv_len], 8, 3);
        let plan = p.single_plan(ks.max_grid);
        let (_, t) = launch(&ctx, &ks, &p, &plan, false, true)?;
        let (_, tg, td) = t.unwrap();
        let bytes = (2 * kv_len * 8 * HEAD_DIM * 2) as f64;
        println!("{kv_len:6}  {:6}  {tg:9.1}  {td:9.1}  {:8.1}", plan.num_ctas(), bytes / (tg * 1e-6) / 1e9);
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for (kv_len, heads) in [(1usize, 1usize), (200, 2), (600, 1)] {
        let p = Problem::new_ragged(&[kv_len], heads, 3);
        let (o, lse) = single_decode_with_kv_cache(p.q.clone(), p.k_cache.clone(), p.v_cache.clone(), heads, 1000);
        let (want_o, want_lse) = p.reference();
        let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        println!(
            "kv_len {kv_len} kv heads {heads} ({} chunks): |o| {eo:.2e}, |lse| {el:.2e}",
            p.single_plan(1000).num_ctas()
        );
        assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
    }
}

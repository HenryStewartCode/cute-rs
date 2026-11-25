// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's block-sparse attention ([`flashinfer::sparse`]): simulated on small BSR
//! and variable-block problems, then, with `--features cuda` and `GPU=1`, checked on the
//! GPU against the f64 reference and timed (CUDA-graph replays) on the shapes of
//! `sparse_flashinfer_bench.py`:
//!
//! * `BlockSparseAttentionWrapper`: M = N = 4K, 8K, 16K in 64 x 64 blocks, 32 query / 8 KV
//!   heads, head dim 128 (and 64 at 8K), 50% and 90% block sparsity, and a custom mask at
//!   4K;
//! * `VariableBlockSparseAttentionWrapper`: FlashInfer's `bench_block_sparse_attention.py`
//!   shapes, 32 / 32 heads, head dim 128, seq 8K and 16K in 20 x 50 blocks, densities 0.5
//!   and 0.1.
//!
//! Block patterns come from [`keep_block`], which the Python script shares, so both sides
//! run the same blocks. `SKIP_CHECKS=1` skips the reference checks (use `--release` for
//! timing runs), `BSR_ONLY=1` / `VARIABLE_ONLY=1` time one wrapper, `HEAD_DIM_ONLY=64` only
//! the BSR shapes of one head dim.
//!
//! `cargo run -p cute-flashinfer --example sparse`; on a GPU,
//! `GPU=1 cargo run --release -p cute-flashinfer --features cuda --example sparse`.

use flashinfer::sparse::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn std::error::Error>> {
    use cute::cuda::CudaContext;
    use cute::prelude::*;
    use flashinfer::sparse::gpu::{Kernels, run};

    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new(&ctx, tc)?;
    println!("device: {}, planner grid: {} CTAs", ctx.name(), ks.max_grid);
    let env = |k: &str| std::env::var_os(k).is_some();
    if !env("SKIP_CHECKS") {
        let checks = [
            ("bsr 64x64 g4 d128", Problem::random_bsr((512, 1024, 64, 64), 0.5, 8, 2, 128, false, 1)),
            ("bsr 16x16 g4 d64", Problem::random_bsr((256, 512, 16, 16), 0.3, 8, 2, 64, false, 2)),
            ("bsr 1x16 g1 d128", Problem::random_bsr((32, 512, 1, 16), 0.5, 2, 2, 128, false, 3)),
            ("bsr 64x64 mask d128", Problem::random_bsr((512, 512, 64, 64), 0.5, 8, 2, 128, true, 4)),
            ("bsr 32x32 mask d64", Problem::random_bsr((256, 512, 32, 32), 0.5, 4, 4, 64, true, 5)),
            ("variable 4x8 d128", Problem::random_variable(700, (4, 8), 0.5, 4, 4, 128, false, 6)),
            ("variable jitter g2 d64", Problem::random_variable(600, (5, 9), 0.4, 4, 2, 64, true, 7)),
            ("variable sparse 1:1", Problem::random_variable(2048, (20, 50), 0.1, 2, 2, 128, false, 8)),
        ];
        for (what, p) in checks {
            let (plan, eo, el, _) = run(&ks, &p, true, None)?;
            println!(
                "check {what}: tile {} split {} ({} CTAs), max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
                plan.tile.cta_tile_q,
                plan.split,
                plan.num_ctas()
            );
            if !(eo < 2e-3 && el < 1e-3) {
                return Err("out of tolerance".into());
            }
        }
    }
    let iters = |p: &Problem| ((4e11 / p.flops()) as usize).clamp(5, 100);
    if !env("VARIABLE_ONLY") {
        println!("BlockSparseAttentionWrapper, 64 x 64 blocks, 32/8 heads");
        println!("  seq   hd  density  mask   nnz   tile  split  ctas    gpu(us)   TFLOPS");
        let mut shapes = vec![];
        for seq in [4096usize, 8192, 16384] {
            for density in [0.5, 0.1] {
                shapes.push((seq, 128usize, density, false));
            }
        }
        shapes.extend([(8192, 64, 0.5, false), (8192, 64, 0.1, false), (4096, 128, 0.5, true)]);
        let only_hd: Option<usize> = std::env::var("HEAD_DIM_ONLY").ok().and_then(|s| s.parse().ok());
        for (seq, hd, density, mask) in shapes.into_iter().filter(|s| only_hd.is_none_or(|h| h == s.1)) {
            let p = Problem::random_bsr((seq, seq, 64, 64), density, 32, 8, hd, mask, 3);
            let (plan, _, _, t) = run(&ks, &p, false, Some(iters(&p)))?;
            let t = t.unwrap();
            println!(
                "{seq:6} {hd:4}  {density:7}  {mask:5} {:5}  {:4}  {:5}  {:4} {t:10.1}  {:7.2}",
                p.pages().iter().sum::<usize>(),
                plan.tile.cta_tile_q,
                plan.split,
                plan.num_ctas(),
                p.flops() / (t * 1e-6) / 1e12
            );
        }
    }
    if !env("BSR_ONLY") {
        println!("VariableBlockSparseAttentionWrapper, 20 x 50 blocks, 32/32 heads, head dim 128");
        println!("  seq  density     kv tokens  tile  split  ctas    gpu(us)   TFLOPS");
        for seq in [8192usize, 16384] {
            for density in [0.5, 0.1] {
                let p = Problem::random_variable(seq, (20, 50), density, 32, 32, 128, false, 3);
                let (plan, _, _, t) = run(&ks, &p, false, Some(iters(&p)))?;
                let t = t.unwrap();
                println!(
                    "{seq:6}  {density:7}  {:12}  {:4}  {:5}  {:4} {t:10.1}  {:7.2}",
                    p.kv_lens.iter().sum::<usize>(),
                    plan.tile.cta_tile_q,
                    plan.split,
                    plan.num_ctas(),
                    p.flops() / (t * 1e-6) / 1e12
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
    for (what, p) in [
        ("bsr 16x16, 4/1 heads, d128", Problem::random_bsr((64, 96, 16, 16), 0.5, 4, 1, 128, false, 1)),
        ("bsr 64x16, 2/1 heads, d64", Problem::random_bsr((128, 160, 64, 16), 0.5, 2, 1, 64, false, 2)),
        ("bsr 16x16 + mask, 4/2 heads, d64", Problem::random_bsr((64, 96, 16, 16), 0.5, 4, 2, 64, true, 3)),
        ("variable 4x6 per-head sizes, 4/2 heads, d64", Problem::random_variable(96, (4, 6), 0.5, 4, 2, 64, true, 4)),
    ] {
        for (grid, min_chunk) in [(p.num_kv_heads, 1), (4000, 1)] {
            let (plan, eo, el) = simulate(&p, grid, min_chunk);
            println!(
                "{what}: tile {} split {}: max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
                plan.tile.cta_tile_q, plan.split
            );
            assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
        }
    }
}

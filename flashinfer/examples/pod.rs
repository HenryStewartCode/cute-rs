// SPDX-License-Identifier: Apache-2.0

//! POD attention ([`flashinfer::pod`]): a prefill and a decode batch in one launch,
//! simulated on small problems (whole and split-KV), then lowered to PTX if a toolchain is
//! found. With `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks
//! against the f64 reference, then CUDA-graph timings on the shapes of FlashInfer's
//! `benchmarks/bench_mixed_attention.py` (single-prefill configurations: 128 decodes of
//! 2K-8K tokens next to a causal prefill of 512-6000 queries, 32 query / 8 KV heads, head
//! dim 128, page size 1, f16; compare with `pod_flashinfer_bench.py`). Each shape is timed
//! as the POD launch, as the same two halves in two launches, and as cute-rs's
//! `single_prefill_with_kv_cache` plus its CUDA-core batch decode ([`flashinfer::decode`]).
//!
//! `SKIP_CHECKS=1` skips the reference checks (use `--release` for timing runs);
//! `NO_SEPARATE=1` skips the separate prefill + decode timings.
//!
//! `cargo run -p cute-flashinfer --example pod`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --release --features cuda --example pod`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
use flashinfer::pod::*;

/// FlashInfer's `bench_mixed_attention.py` single-prefill configurations: `(prefill
/// qo_len, prefill kv_len, decode kv_len)`, 128 decodes each.
pub const SHAPES: [(usize, usize, usize); 6] = [
    (512, 512, 2048),
    (1536, 1536, 2048),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (6000, 7000, 8192),
];
/// Decodes per shape.
pub const DECODES: usize = 128;

/// A benchmark problem: page size 1 with the pages in order (FlashInfer's benchmark uses
/// `arange` page tables), 8 KV heads x 4.
fn bench_problem(qo: usize, kv: usize, dkv: usize) -> Problem {
    let mut p = Problem::new(qo, kv, &vec![dkv; DECODES], 1, 8, 4, true, 3);
    p.decode.kv_indices.sort_unstable();
    p
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    use flashinfer::pod::gpu::{Kernels, run};
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new(&ctx, tc.clone())?;
    println!("device: {}, planner grid: {} CTAs", ctx.name(), ks.max_grid);
    let skip = std::env::var_os("SKIP_CHECKS").is_some();
    // Correctness: each prefill tile, causal and not, split and whole, paged decodes with
    // shuffled page tables, up to a realistic size.
    for (qo, kv, decodes, page, heads, causal) in [
        (3usize, 70usize, vec![9usize, 37, 70], 8usize, 2usize, false),
        (40, 75, vec![1, 200, 33], 16, 2, true),
        (200, 300, vec![500; 6], 16, 8, true),
        (512, 512, vec![2048; 16], 1, 8, true),
        (1000, 1500, (0..40).map(|i| 100 + 97 * i).collect(), 16, 8, true),
    ] {
        if skip {
            break;
        }
        let p = Problem::new(qo, kv, &decodes, page, heads, 4, causal, 11);
        let (plans, e, _) = run(&ks, &p, true, None)?;
        println!(
            "check prefill {qo}x{kv} causal {causal} + {} decodes, page {page}, heads {}/{heads}: tiles {}/{}, split {}/{}, \
             ctas {:?} x {heads}, errors o_p {:.1e} lse_p {:.1e} o_d {:.1e} lse_d {:.1e}",
            decodes.len(),
            4 * heads,
            plans.prefill.tile.cta_tile_q,
            plans.decode.tile.cta_tile_q,
            plans.prefill.split,
            plans.decode.split,
            plans.ctas(),
            e[0],
            e[1],
            e[2],
            e[3]
        );
        if !(e[0] < 2e-3 && e[1] < 1e-3 && e[2] < 2e-3 && e[3] < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    // Timings (CUDA-graph replays, microseconds).
    let separate = std::env::var_os("NO_SEPARATE").is_none();
    let dks = if separate {
        Some(flashinfer::decode::gpu::Kernels::new(&ctx, &tc, flashinfer::decode::STAGES)?)
    } else {
        None
    };
    println!("prefill qo/kv  decode kv  ctas(p,d)    pod(us)  two-launch(us)  prefill+decode(us)  pod GB/s");
    for (qo, kv, dkv) in SHAPES {
        let p = bench_problem(qo, kv, dkv);
        let iterations = if dkv >= 8192 { 20 } else { 50 };
        let (plans, _, t) = run(&ks, &p, false, Some(iterations))?;
        let t = t.unwrap();
        let bytes = p.bytes();
        drop(p);
        // cute-rs's single prefill (same plan) and CUDA-core batch decode, timed apart.
        let sep = match &dks {
            Some(dks) => {
                let pp = flashinfer::prefill::Problem::new_ragged(&[qo], &[kv], 8, 4, true, 3);
                let plan = pp.single_plan(ks.max_grid);
                let (_, _, tp) = flashinfer::prefill::gpu::run_plan(&ks.prefill, &pp, &plan, false, Some(iterations))?;
                drop(pp);
                let mut dp = flashinfer::decode::Problem::new(&vec![dkv; DECODES], 1, 8, 5);
                dp.kv_indices.sort_unstable();
                let (_, _, _, td) = flashinfer::decode::gpu::run(&ctx, dks, &dp, false, true)?;
                format!("{:.1}", tp.unwrap() + td.unwrap().1)
            }
            None => "-".into(),
        };
        println!(
            "{qo:5}/{kv:<5}  {dkv:9}  {:>9}  {:9.1}  {:14.1}  {sep:>18}  {:8.1}",
            format!("{:?}", plans.ctas()),
            t.pod,
            t.two_launches,
            bytes / (t.pod * 1e-6) / 1e9
        );
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
    for (p, grid, min_chunk) in [
        (Problem::new(40, 75, &[9, 37, 70, 16], 8, 1, 4, true, 3), 1, 128),
        (Problem::new(3, 600, &[300, 5, 130], 4, 2, 4, false, 7), 1000, 1),
    ] {
        let (plans, e) = simulate(&p, grid, min_chunk);
        println!(
            "simulated prefill {}x{} + {} decodes: tiles {}/{}, split {}/{}, errors {:?}",
            p.prefill.qo_lens[0],
            p.prefill.kv_lens[0],
            p.decode.qo_lens.len(),
            plans.prefill.tile.cta_tile_q,
            plans.decode.tile.cta_tile_q,
            plans.prefill.split,
            plans.decode.split,
            e.map(|x| format!("{x:.1e}"))
        );
        assert!(e[0] < 2e-3 && e[1] < 1e-3 && e[2] < 2e-3 && e[3] < 1e-3, "out of tolerance");
    }
    match Toolchain::discover() {
        Some(tc) => {
            let p = bench_problem(SHAPES[0].0, SHAPES[0].1, 16);
            let spec = p.spec(&p.plans(44));
            let k = kernel(&spec);
            let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
            let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
            // `PTX_DIR=dir` also writes the PTX there, for `ptxas -v` or SASS.
            if let Some(dir) = std::env::var_os("PTX_DIR") {
                std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            println!(
                "PTX {}: {} bytes, {} mma.sync, {} ldmatrix, {} cp.async, {} atom, {} B shared",
                k.name(),
                ptx.len(),
                ptx.matches("mma.sync").count(),
                ptx.matches("ldmatrix").count(),
                ptx.matches("cp.async.cg").count(),
                ptx.matches("atom.").count(),
                spec.smem_bytes()
            );
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

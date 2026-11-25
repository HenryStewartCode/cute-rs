// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's persistent `BatchAttention` ([`flashinfer::attention`]): mixed prefill and
//! decode batches in one persistent launch, simulated on small problems (written through
//! and split-KV), then lowered to PTX if a toolchain is found. With `--features cuda` and
//! `GPU=1` it runs on the GPU instead: checks against the f64 reference, then CUDA-graph
//! timings on the mixes of FlashInfer's `benchmarks/bench_batch_attention.py` (decodes of
//! one query next to a causal prefill chunk, Qwen-8B 32/8 and Llama-3.1-70B 64/8 heads,
//! head dim 128, page size 1, f16; compare with `attention_flashinfer_bench.py`). Each mix
//! is also timed as cute-rs's paged batch prefill over the whole batch, as the prefill and
//! decode requests in two separate batch-prefill launches (the decodes on the 16-row
//! tensor-core tile; the times summed, as the benchmark's "Decode + Prefill"), and, with
//! one prefill request, as cute-rs's POD ([`flashinfer::pod`], ragged prefill).
//!
//! `SKIP_CHECKS=1` skips the reference checks (use `--release` for timing runs);
//! `NO_BASELINES=1` times the persistent kernel alone; `ALTERNATE=1` builds it with
//! [`AttentionSpec::alternate`]; `MODEL=Qwen-8B` and `MIX=0,2` pick rows; `PAGE=16` sets
//! the page size.
//!
//! `cargo run -p cute-flashinfer --example attention`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --release --features cuda --example attention`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
use flashinfer::attention::*;
use flashinfer::prefill::TileConfig;

/// Mixes after `bench_batch_attention.py`: `(decodes, decode kv_len, prefills, prefill
/// chunk (queries), prefill kv_len)`; the first is its default.
pub const MIXES: [(usize, usize, usize, usize, usize); 5] = [
    (128, 8192, 1, 4096, 8192),
    (128, 2048, 1, 1024, 2048),
    (32, 8192, 1, 4096, 8192),
    (128, 8192, 0, 0, 0),
    (64, 4096, 2, 2048, 4096),
];
/// `(name, num_qo_heads, num_kv_heads)` of the benchmark's head-dim-128 models.
pub const MODELS: [(&str, usize, usize); 2] = [("Qwen-8B", 32, 8), ("Llama-3.1-70B", 64, 8)];

#[cfg(feature = "cuda")]
fn lens(mix: (usize, usize, usize, usize, usize)) -> (Vec<usize>, Vec<usize>) {
    let (nd, dkv, np, pq, pkv) = mix;
    let mut qo = vec![1; nd];
    let mut kv = vec![dkv; nd];
    qo.extend(vec![pq; np]);
    kv.extend(vec![pkv; np]);
    (qo, kv)
}

#[cfg(feature = "cuda")]
/// A benchmark problem: the pages in order (the benchmark's `arange` page table).
fn bench_batch(qo: &[usize], kv: &[usize], page: usize, kv_heads: usize, group: usize) -> flashinfer::prefill::Problem {
    let mut p = flashinfer::prefill::Problem::new(qo, kv, page, kv_heads, group, true, 3);
    p.kv_indices.sort_unstable();
    p
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    use flashinfer::attention::gpu::{Kernels, run};
    use flashinfer::prefill;
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, tc.clone())?;
    // `ALTERNATE=1`: task 1 first in half the grid (not FlashInfer's order).
    ks.alternate = std::env::var_os("ALTERNATE").is_some();
    println!("device: {}, persistent grid: {} CTAs", ctx.name(), 2 * ks.num_sm);
    let skip = std::env::var_os("SKIP_CHECKS").is_some();
    for (qo, kv, page, heads, group, causal) in [
        (vec![1usize, 40, 1, 1], vec![37usize, 75, 9, 70], 16usize, 2usize, 4usize, true),
        (vec![1, 300, 1, 7, 1], vec![2000, 300, 513, 900, 64], 16, 8, 4, true),
        (
            vec![1; 40].into_iter().chain([512]).collect(),
            (0..40).map(|i| 100 + 97 * i).chain([1500]).collect(),
            16,
            8,
            4,
            true,
        ),
        (vec![3, 1, 64, 1], vec![1000, 4000, 700, 77], 16, 4, 8, false),
        (vec![1; 16].into_iter().chain([256]).collect(), vec![2048; 17], 1, 8, 4, true),
    ] {
        if skip {
            break;
        }
        let p = Problem::new(&qo, &kv, page, heads, group, causal, 11);
        let (plan, eo, el, _) = run(&ks, &p, true, None)?;
        println!(
            "check {} requests, page {page}, heads {}/{heads}, causal {causal}: works {}/{}, chunks {}/{}, merge rows {}, \
             errors o {eo:.1e} lse {el:.1e}",
            qo.len(),
            heads * group,
            plan.tasks[0].num_works(),
            plan.tasks[1].num_works(),
            plan.tasks[0].len_kv_chunk,
            plan.tasks[1].len_kv_chunk,
            plan.num_merge()
        );
        if !(eo < 2e-3 && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    let baselines = std::env::var_os("NO_BASELINES").is_none();
    let pks = prefill::gpu::Kernels::new(&ctx, tc.clone())?;
    let pod_ks = flashinfer::pod::gpu::Kernels::new(&ctx, tc.clone())?;
    let page = std::env::var("PAGE").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize);
    println!("page size {page}; times in us (CUDA-graph replays)");
    println!(
        "model           decodes  dkv    prefill  pkv   | persistent  GB/s  | batch-prefill  prefill+decode  pod  | works  merge"
    );
    // `MODEL=name` and `MIX=i,j,...` pick models and mixes.
    let model = std::env::var("MODEL").ok();
    let picks: Option<Vec<usize>> =
        std::env::var("MIX").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    for (name, qo_heads, kv_heads) in MODELS {
        if model.as_deref().is_some_and(|m| m != name) {
            continue;
        }
        let group = qo_heads / kv_heads;
        for (i, mix) in MIXES.into_iter().enumerate() {
            if picks.as_ref().is_some_and(|p| !p.contains(&i)) {
                continue;
            }
            let (qo, kv) = lens(mix);
            let iterations = 20;
            let p = Problem::from_batch(bench_batch(&qo, &kv, page, kv_heads, group));
            let (plan, _, _, t) = run(&ks, &p, false, Some(iterations))?;
            let t = t.unwrap();
            let bytes = p.bytes();
            let (mut whole, mut sep, mut pod) = ("-".to_string(), "-".to_string(), "-".to_string());
            if baselines {
                let plan_w = p.batch.plan(pks.max_grid);
                let (_, tw) = prefill::gpu::launch(&pks, &p.batch, &plan_w, false, Some(iterations))?;
                whole = format!("{:.1}", tw.unwrap());
                drop(p);
                let (nd, dkv, np, pq, pkv) = mix;
                let mut total = 0.0;
                for (qs, ks_) in [(vec![1; nd], vec![dkv; nd]), (vec![pq; np], vec![pkv; np])] {
                    if qs.is_empty() {
                        continue;
                    }
                    let sub = bench_batch(&qs, &ks_, page, kv_heads, group);
                    let plan_s = sub.plan(pks.max_grid);
                    let (_, ts) = prefill::gpu::launch(&pks, &sub, &plan_s, false, Some(iterations))?;
                    total += ts.unwrap();
                }
                sep = format!("{total:.1}");
                if np == 1 && nd > 0 {
                    let mut pp = flashinfer::pod::Problem::new(pq, pkv, &vec![dkv; nd], page, kv_heads, group, true, 3);
                    pp.decode.kv_indices.sort_unstable();
                    let (_, _, tp) = flashinfer::pod::gpu::run(&pod_ks, &pp, false, Some(iterations))?;
                    pod = format!("{:.1}", tp.unwrap().pod);
                }
            }
            println!(
                "{name:15} {:7}  {:5}  {:3}x{:<5} {:5} | {:10.1}  {:5.1} | {whole:>13}  {sep:>14}  {pod:>5} | {:>3}/{:<4} {}",
                mix.0,
                mix.1,
                mix.2,
                mix.3,
                mix.4,
                t,
                bytes / (t * 1e-6) / 1e9,
                plan.tasks[0].num_works(),
                plan.tasks[1].num_works(),
                plan.num_merge()
            );
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
    for (qo, kv, page, heads, group, causal, num_sm) in [
        (vec![1usize, 40, 1, 1], vec![37usize, 75, 9, 70], 16usize, 1usize, 4usize, true, 1usize),
        (vec![1, 48, 1, 3], vec![300, 200, 520, 260], 16, 2, 4, true, 8),
        (vec![2, 1, 20, 1], vec![700, 40, 150, 90], 16, 1, 8, false, 4),
    ] {
        let p = Problem::new(&qo, &kv, page, heads, group, causal, 3);
        let (plan, eo, el) = simulate(&p, num_sm);
        println!(
            "simulated {} requests (causal {causal}, {num_sm} SMs): works {}/{}, chunks {}/{}, merge rows {}, errors o {eo:.1e} lse {el:.1e}",
            qo.len(),
            plan.tasks[0].num_works(),
            plan.tasks[1].num_works(),
            plan.tasks[0].len_kv_chunk,
            plan.tasks[1].len_kv_chunk,
            plan.num_merge()
        );
        assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
    }
    match Toolchain::discover() {
        Some(tc) => {
            // The split-test kernel, and the benchmark's (page size 1, unsplit) with both
            // tasks and with each task alone.
            let p = Problem::new(&[1, 48, 1, 3], &[300, 200, 520, 260], 16, 2, 4, true, 7);
            let split = p.spec(&p.plan(8));
            let task = |q: usize| Some(TaskSpec { tile: TileConfig::sm89(q), split: false, unsplit: true });
            let bench =
                AttentionSpec { tasks: [task(128), task(16)], group: 4, page_size: 1, causal: true, alternate: false };
            let specs = [
                split,
                bench,
                AttentionSpec { tasks: [task(128), None], ..bench },
                AttentionSpec { tasks: [None, task(16)], ..bench },
            ];
            for spec in specs {
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
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

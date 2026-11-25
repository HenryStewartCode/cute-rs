// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's MLA paged attention ([`flashinfer::mla`]): simulated on small decode,
//! prefill and split problems, then lowered to PTX if a toolchain is found. With
//! `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks against
//! the f64 reference, then CUDA-graph timings on the shapes of FlashInfer's
//! `benchmarks/bench_deepseek_mla.py` (decode, page size 1; the same shapes as
//! `examples/mla_flashinfer_bench.py`).
//!
//! `cargo run -p cute-flashinfer --example mla`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example mla`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::mla::gpu::{Kernels, run};
use flashinfer::mla::*;

/// `(qo_lens, kv_lens, page_size, num_heads, causal)` checked on the simulator.
type Case = (Vec<usize>, Vec<usize>, usize, usize, bool);

/// See [`Case`].
fn sim_cases() -> Vec<Case> {
    vec![
        (vec![1; 4], vec![1, 7, 21, 40], 4, 16, false),
        (vec![1, 1], vec![5, 33], 16, 128, false),
        (vec![3, 5], vec![10, 37], 4, 16, true),
        (vec![1, 1], vec![1024, 9], 16, 16, false),
        (vec![4], vec![150], 4, 16, true),
    ]
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let ks = Kernels::new(&ctx, tc)?;
    println!("device: {}, {} SMs", ctx.name(), ks.num_sm);
    // Correctness: the simulator's cases, then realistic decode and prefill sizes.
    let mut cases = sim_cases();
    cases.extend([
        (vec![1; 8], (0..8).map(|i| 300 + 397 * i).collect(), 64, 128, false),
        (vec![1; 3], vec![4096, 1000, 17], 1, 128, false),
        (vec![1; 5], vec![2048, 64, 999, 1, 3000], 16, 16, false),
        (vec![2; 4], vec![700, 1500, 33, 2500], 32, 64, true),
        (vec![37, 128], vec![300, 900], 16, 16, true),
        (vec![17], vec![600], 1, 128, true),
    ]);
    // `MLA_SKIP_CHECKS=1` skips them (the f64 reference of the larger cases is slow).
    if std::env::var_os("MLA_SKIP_CHECKS").is_some() {
        cases.clear();
    }
    for (qo, kv, ps, h, causal) in cases {
        let p = Problem::new(&qo, &kv, ps, h, causal, 11);
        let (plan, eo, el, _) = run(&ks, &p, true, None)?;
        println!(
            "check qo {qo:?} kv {:?}{} page {ps} heads {h} causal {causal}: {} works, split {}, max |o - ref| = \
             {eo:.2e}, max |lse - ref| = {el:.2e}",
            &kv[..kv.len().min(4)],
            if kv.len() > 4 { " .." } else { "" },
            plan.num_works(),
            plan.split()
        );
        if !(eo < 2e-3 && el < 1e-3) {
            return Err("out of tolerance".into());
        }
    }
    // Timings: bench_deepseek_mla.py's decode shapes (page size 1, one query per request).
    println!("batch  seq_len  heads  works  split   gpu(us)    GB/s  TFLOPS");
    let mut shapes = vec![];
    for seq_len in [1024usize, 2048, 8192] {
        for batch in [64usize, 128, 768] {
            for heads in [64usize, 128] {
                // 768 x 8192 tokens would address more than 2^30 cache elements.
                if batch * seq_len * HEAD_DIM_CKV < 1 << 30 {
                    shapes.push((batch, seq_len, heads));
                }
            }
        }
    }
    shapes.extend([(64, 1024, 16), (64, 8192, 16)]);
    for (batch, seq_len, heads) in shapes {
        let p = Problem::bench(batch, seq_len, heads);
        let reps = if batch * seq_len <= 128 * 2048 { 50 } else { 10 };
        let (plan, _, _, t) = run(&ks, &p, false, Some(reps))?;
        let t = t.unwrap();
        let io = 2 * (p.q_nope.len() + p.q_pe.len() + p.ckv.len() + p.kpe.len() + p.q_nope.len());
        let flops = 2 * batch * heads * (2 * HEAD_DIM_CKV + HEAD_DIM_KPE) * seq_len;
        println!(
            "{batch:5}  {seq_len:7}  {heads:5}  {:5}  {:5}  {t:8.1}  {:6.1}  {:6.2}",
            plan.num_works(),
            plan.split(),
            io as f64 / (t * 1e-6) / 1e9,
            flops as f64 / (t * 1e-6) / 1e12
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
    for (qo, kv, ps, h, causal) in sim_cases() {
        let p = Problem::new(&qo, &kv, ps, h, causal, 7);
        let (plan, eo, el) = simulate(&p, 8);
        println!(
            "qo {qo:?} kv {kv:?} page {ps} heads {h} causal {causal}: cluster {} x {}, {} works, split {}: \
             max |o - ref| = {eo:.2e}, max |lse - ref| = {el:.2e}",
            plan.cluster_size,
            plan.num_clusters,
            plan.num_works(),
            plan.split()
        );
        assert!(eo < 2e-3 && el < 1e-3, "out of tolerance");
    }
    match Toolchain::discover() {
        Some(tc) => {
            let spec = KernelSpec { num_heads: 128, page_size: 1, causal: false };
            let k = kernel(&spec);
            let ptx = k
                .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                .unwrap_or_else(|e| panic!("{e}"))
                .ptx()
                .to_string();
            // `PTX_DIR=dir` also writes the PTX there, for `ptxas -v` or SASS.
            if let Some(dir) = std::env::var_os("PTX_DIR") {
                std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            println!(
                "PTX {}: {} bytes, {} cp.async, {} ldmatrix, {} mma, {} ex2.approx",
                k.name(),
                ptx.len(),
                ptx.matches("cp.async.cg").count(),
                ptx.matches("ldmatrix").count(),
                ptx.matches("mma.sync").count(),
                ptx.matches("ex2.approx").count()
            );
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

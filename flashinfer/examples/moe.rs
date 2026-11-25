// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `cutlass_fused_moe` (bf16/f16, SwiGLU, precomputed top-k routing;
//! [`flashinfer::moe`]): simulated on small shapes, then lowered to PTX if a toolchain is
//! found. With `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks
//! against the f64 reference, then CUDA-graph timings on Mixtral 8x7B (hidden 4096, inter
//! 14336, 8 experts, top-2) and the two configs of FlashInfer's
//! `benchmarks/bench_cutlass_fused_moe.py` (hidden 7168, top-8; 256 experts of inter 256 and
//! 32 experts of inter 2048), at 1, 16, 128 and 1024 tokens. The routing kernels
//! ([`route_kernel`]) are checked the same way and timed on Mixtral (softmax-top-2 of 8),
//! Qwen3-MoE (128 experts, top-8, Renormalize and softmax-top-k-renormalize), DeepSeek-V3
//! (256 experts, 8 groups, top-4 groups, top-8) and Kimi K2 (384 experts, ungrouped, top-8)
//! at 1 to 8192 tokens.
//!
//! `TILE1=bm,bn,bk,stages,wm,wn` and `TILE2=...` override the GEMM tiles; `CONFIGS=mixtral,..`,
//! `TOKENS=1,16,..`, `DTYPE=f16` and `NOCHECK=1` pick what runs; `MOE_STEPS=2` times only
//! some of the five launches; `ROUTES=0` skips the routing kernels and `ROUTE_TOKENS=..` picks
//! their token counts.
//!
//! `cargo run -p cute-flashinfer --example moe`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example moe`.

use cute::prelude::*;
use flashinfer::moe::*;

#[allow(dead_code)]
fn env_tile(var: &str) -> Option<Tile> {
    let v: Vec<usize> = std::env::var(var).ok()?.split(',').map(|x| x.trim().parse().unwrap()).collect();
    Some(Tile { bm: v[0], bn: v[1], bk: v[2], stages: v[3], wm: v[4], wn: v[5] })
}

#[allow(dead_code)]
const CONFIGS: [(&str, MoeShape); 3] = [
    ("mixtral", MoeShape::new(4096, 14336, 8, 2)),
    ("e256_i256", MoeShape::new(7168, 256, 256, 8)),
    ("e32_i2048", MoeShape::new(7168, 2048, 32, 8)),
];

#[allow(dead_code)]
const ROUTES: [(&str, Routing, usize, usize); 5] = [
    ("mixtral_softmax", Routing::SoftmaxTopK { renormalize: false }, 8, 2),
    ("qwen3_renormalize", Routing::Renormalize, 128, 8),
    ("qwen3_softmax_renorm", Routing::SoftmaxTopK { renormalize: true }, 128, 8),
    ("deepseek_v3", Routing::DeepSeekV3 { n_group: 8, topk_group: 4 }, 256, 8),
    ("kimi_k2", Routing::DeepSeekV3 { n_group: 1, topk_group: 1 }, 384, 8),
];

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn std::error::Error>> {
    use cute::cuda::CudaContext;
    use flashinfer::moe::gpu::{Kernels, bench, check, random_weights, route};
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    let ks = Kernels::new(&ctx, tc)?;
    let dtype = if std::env::var("DTYPE").is_ok_and(|d| d == "f16") { Dtype::F16 } else { Dtype::BF16 };
    if std::env::var_os("NOCHECK").is_none() {
        for dt in [Dtype::F16, Dtype::BF16] {
            for (shape, tokens) in [
                (MoeShape::new(256, 384, 4, 2), 37),
                (MoeShape::new(1024, 512, 8, 2), 3),
                (MoeShape::new(512, 256, 64, 8), 9),
                (MoeShape::new(512, 512, 8, 2), 300),
            ] {
                let p = Problem::new(dt, shape, tokens, 5);
                let err = check(&ks, &p)?;
                println!(
                    "check {} {shape:?} tokens {tokens} tiles {} / {}: max scaled error {err:.2e}",
                    dt.name(),
                    p.tile1.name(),
                    p.tile2.name()
                );
                assert!(err <= 1.0);
            }
        }
        for (name, routing, e, k) in ROUTES {
            for score in [ScoreType::F32, ScoreType::BF16, ScoreType::F16] {
                let p = RouteProblem::new(routing, score, 37, e, k, 3);
                let (err, _) = route(&ks, &p, true, false)?;
                println!("check route {name} {}: max scaled error {err:.2e}", score.name());
                assert!(err <= 1.0);
            }
        }
    }
    if std::env::var("ROUTES").map_or(true, |v| v != "0") {
        let tokens: Vec<usize> = std::env::var("ROUTE_TOKENS")
            .map_or(vec![1, 16, 128, 1024, 8192], |v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
        println!("route                 scores experts top_k tokens  time(us)");
        for (name, routing, e, k) in ROUTES {
            for score in [ScoreType::F32, ScoreType::BF16] {
                for &n in &tokens {
                    let p = RouteProblem::new(routing, score, n, e, k, 1);
                    let t = route(&ks, &p, false, true)?.1.unwrap();
                    println!("{name:21} {:6} {e:7} {k:5} {n:6} {t:9.2}", score.name());
                }
            }
        }
    }
    let only = std::env::var("CONFIGS").ok();
    let tokens: Vec<usize> = std::env::var("TOKENS")
        .map_or(vec![1, 16, 128, 1024], |v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    println!("config        hidden  inter experts top_k tokens  time(us)  TFLOPS  tiles");
    for (name, shape) in CONFIGS {
        if only.as_ref().is_some_and(|o| !o.split(',').any(|c| c == name)) {
            continue;
        }
        let w = random_weights(&ks, dtype, shape)?;
        for &n in &tokens {
            let (mut t1, mut t2) = pick_tiles(n * shape.top_k, shape.num_experts);
            t1 = env_tile("TILE1").unwrap_or(t1);
            t2 = env_tile("TILE2").unwrap_or(t2);
            let t = bench(&ks, dtype, shape, &w, n, (t1, t2), 20)?;
            let MoeShape { hidden, inter, num_experts, top_k } = shape;
            println!(
                "{name:12} {hidden:7} {inter:6} {num_experts:7} {top_k:5} {n:6} {t:9.1} {:7.2}  {} / {}",
                shape.flops(n) / t * 1e-6,
                t1.name(),
                t2.name()
            );
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
    for (shape, tokens) in
        [(MoeShape::new(128, 128, 4, 2), 5), (MoeShape::new(128, 128, 16, 1), 3), (MoeShape::new(128, 128, 4, 2), 50)]
    {
        for dtype in [Dtype::F16, Dtype::BF16] {
            let p = Problem::new(dtype, shape, tokens, 1);
            let e = simulate(&p);
            println!(
                "{} {shape:?} tokens {tokens} tiles {} / {}: max scaled error {e:.2e}",
                dtype.name(),
                p.tile1.name(),
                p.tile2.name()
            );
        }
    }
    for (name, routing, e, k) in ROUTES {
        for score in [ScoreType::F32, ScoreType::BF16] {
            let p = RouteProblem::new(routing, score, 7, e, k, 2);
            println!("route {name} {}: max scaled error {:.2e}", score.name(), simulate_route(&p));
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            let shape = MoeShape::new(4096, 14336, 8, 2);
            for k in [gemm1_kernel(Dtype::BF16, shape, DECODE_TILE), gemm2_kernel(Dtype::BF16, shape, DECODE_TILE)] {
                let ptx = k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
                println!(
                    "PTX {}: {} mma.sync, {} ldmatrix, {} cp.async",
                    k.name(),
                    ptx.matches("mma.sync.aligned").count(),
                    ptx.matches("ldmatrix.sync").count(),
                    ptx.matches("cp.async.cg").count()
                );
            }
        }
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

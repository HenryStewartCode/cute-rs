// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's segment GEMM (`SegmentGEMMWrapper.run`) and `bmm_fp8` ([`flashinfer::gemm`]):
//! simulated on small shapes, then lowered to PTX if a toolchain is found. With
//! `--features cuda` and `GPU=1` it runs on the GPU instead: correctness checks against the
//! f64 reference, then CUDA-graph timings on the shapes of FlashInfer's
//! `benchmarks/bench_hopper_grouped_gemm.py` (bf16, column-major weights, d 4096) and of its
//! `bmm_fp8` sample test list (`benchmarks/samples/sample_testlist.txt`).
//!
//! `SEG_TILE=bm,bn,bk,stages,wm,wn` and `FP8_TILE=...` override the tiles; `BATCH=..`, `TOKENS=..`,
//! `D8192=1`, `NOFP8=1`, `FP8_M=..` and `NOCHECK=1` pick the timed shapes.
//!
//! `cargo run -p cute-flashinfer --example gemm`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example gemm`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::gemm::gpu::{run_bmm_fp8, run_segment};
use flashinfer::gemm::*;

#[allow(dead_code)]
fn env_tile(var: &str) -> Option<Tile> {
    let v: Vec<usize> = std::env::var(var).ok()?.split(',').map(|x| x.trim().parse().unwrap()).collect();
    Some(Tile { bm: v[0], bn: v[1], bk: v[2], stages: v[3], wm: v[4], wn: v[5] })
}

#[allow(dead_code)]
fn env_list(var: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(var).map_or(default.to_vec(), |v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    let seg_tile = env_tile("SEG_TILE");
    let fp8_tile = env_tile("FP8_TILE");
    if std::env::var_os("NOCHECK").is_none() {
        for dtype in [Dtype::F16, Dtype::BF16] {
            for col_major in [true, false] {
                for indices in [None, Some(vec![1, 0, 1])] {
                    let mut p = SegmentProblem::new(dtype, &[7, 130, 64], 256, 384, col_major, 3, indices, true, 5);
                    if let Some(t) = seg_tile {
                        p.tile = t;
                    }
                    let (err, _) = run_segment(&ctx, &tc, &p, true, false)?;
                    println!("check segment {} col_major={col_major}: max scaled error {err:.2e}", dtype.name());
                    assert!(err <= 1.0);
                }
            }
        }
        for fp8 in [Fp8::E4M3, Fp8::E5M2] {
            for out in [Dtype::BF16, Dtype::F16] {
                for (b, m, n, k) in [(4usize, 4usize, 256usize, 1024usize), (1, 64, 256, 512), (2, 200, 256, 256)] {
                    let mut p = BmmFp8Problem::new(fp8, out, b, m, n, k, 3);
                    if let Some(t) = fp8_tile {
                        p.tile = t;
                    }
                    let (err, _) = run_bmm_fp8(&ctx, &tc, &p, true, false)?;
                    println!(
                        "check bmm_fp8 {} -> {} {b}x{m}x{n}x{k}: max scaled error {err:.2e}",
                        fp8.name(),
                        out.name()
                    );
                    assert!(err <= 1.0);
                }
            }
        }
    }
    println!("segment_gemm bf16 col-major  batch tokens  d_in  d_out  time(us)  TFLOPS");
    let d_list: Vec<usize> = if std::env::var_os("D8192").is_some() { vec![4096, 8192] } else { vec![4096] };
    for &d in &d_list {
        for batch in env_list("BATCH", &[1, 3, 8, 16]) {
            for tokens in env_list("TOKENS", &[32, 128, 512]) {
                let mut p = SegmentProblem::new(Dtype::BF16, &vec![tokens; batch], d, d, true, batch, None, false, 1);
                if let Some(t) = seg_tile {
                    p.tile = t;
                }
                let t = run_segment(&ctx, &tc, &p, false, true)?.1.unwrap();
                println!("{:28} {batch:5} {tokens:6} {d:5} {d:6} {t:9.1} {:7.2}", "", p.flops() / t * 1e-6);
            }
        }
    }
    println!("bmm_fp8 e4m3 -> bf16   batch     m     n     k  time(us)  TFLOPS");
    let fp8_shapes = [(64usize, 4usize, 1024usize, 7168usize), (1, 64, 4096, 4096)];
    let fp8_m = env_list("FP8_M", &[4, 64]);
    for &(b, m, n, k) in fp8_shapes.iter().filter(|s| std::env::var_os("NOFP8").is_none() && fp8_m.contains(&s.1)) {
        let mut p = BmmFp8Problem::new(Fp8::E4M3, Dtype::BF16, b, m, n, k, 1);
        if let Some(t) = fp8_tile {
            p.tile = t;
        }
        let t = run_bmm_fp8(&ctx, &tc, &p, false, true)?.1.unwrap();
        println!("{:22} {b:5} {m:5} {n:5} {k:5} {t:9.1} {:7.2}", "", p.flops() / t * 1e-6);
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    let small = Tile { bm: 32, bn: 64, bk: 32, stages: 3, wm: 1, wn: 2 };
    for dtype in [Dtype::F16, Dtype::BF16] {
        for col_major in [true, false] {
            let mut p =
                SegmentProblem::new(dtype, &[5, 40, 0, 33], 64, 128, col_major, 3, Some(vec![2, 0, 1, 2]), true, 7);
            p.tile = small;
            let e = simulate_segment(&p);
            println!("segment {} col_major={col_major}: max scaled error {e:.2e}", dtype.name());
        }
    }
    for fp8 in [Fp8::E4M3, Fp8::E5M2] {
        let mut p = BmmFp8Problem::new(fp8, Dtype::BF16, 2, 20, 64, 128, 5);
        p.tile = Tile { bm: 32, bn: 32, bk: 64, stages: 2, wm: 1, wn: 2 };
        println!("bmm_fp8 {}: max scaled error {:.2e}", fp8.name(), simulate_bmm_fp8(&p));
    }
    match Toolchain::discover() {
        Some(tc) => {
            let k = segment_gemm_kernel(Dtype::BF16, 4096, 4096, true, Tile::SEGMENT);
            let ptx = k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
            println!(
                "PTX {}: {} mma.sync, {} ldmatrix, {} cp.async",
                k.name(),
                ptx.matches("mma.sync.aligned").count(),
                ptx.matches("ldmatrix.sync").count(),
                ptx.matches("cp.async.cg").count()
            );
        }
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

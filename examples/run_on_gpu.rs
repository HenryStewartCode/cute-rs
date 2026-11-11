//! Run the tensor-core GEMM on an NVIDIA GPU (SM80 or newer) and check it
//! against the CPU.
//!
//! Needs the `cuda` feature, an NVIDIA driver, and an `mlir-opt` to compile
//! the kernel to PTX:
//!
//! ```text
//! cargo run --release --features cuda --example run_on_gpu
//! ```

#[allow(dead_code)]
#[path = "tensorop_gemm.rs"]
mod tensorop_gemm;

use cute::cuda::CudaContext;
use cute::prelude::*;
use tensorop_gemm::GemmConfig;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability();
    println!("device: {} (compute capability {major}.{minor})", ctx.name());
    let toolchain = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;

    let cfg = GemmConfig::new(1024, 1024, 1024);
    let kernel = tensorop_gemm::build(cfg);
    let a: Vec<f32> = (0..cfg.m * cfg.k).map(|i| ((i * 7 + 3) % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..cfg.n * cfg.k).map(|i| ((i * 5 + 1) % 5) as f32 - 2.0).collect();
    let mut args =
        vec![KernelArg::f16_from_f32(&a), KernelArg::f16_from_f32(&b), KernelArg::f32(vec![0.0; cfg.m * cfg.n])];
    let launch = LaunchConfig::new([(cfg.m / cfg.bm) as u32, (cfg.n / cfg.bn) as u32, 1], [128, 1, 1]);
    kernel.run_cuda(&ctx, &toolchain, launch, &mut args)?;

    let c = args[2].as_f32();
    let mut mismatches = 0;
    for i in (0..cfg.m).step_by(37) {
        for j in (0..cfg.n).step_by(29) {
            let expected: f32 = (0..cfg.k).map(|l| a[i * cfg.k + l] * b[j * cfg.k + l]).sum();
            if c[i * cfg.n + j] != expected {
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        return Err(format!("{mismatches} sampled elements of C are wrong").into());
    }
    println!("{}x{}x{} tensor-core GEMM on the GPU: correct", cfg.m, cfg.n, cfg.k);
    Ok(())
}

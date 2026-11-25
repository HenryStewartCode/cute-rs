// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `per_token_group_quant_8bit` ([`flashinfer::quantization::group_quant_kernel`]):
//! simulated on small inputs, then lowered to PTX if a toolchain is found. With
//! `--features cuda` and `GPU=1` it runs on the GPU instead: exact checks (bytes and scale
//! bits against the f32 reference), then CUDA-graph timings to compare with
//! `fp8_quant_flashinfer_bench.py`.
//!
//! `cargo run -p cute-flashinfer --example quantization_fp8`; on a GPU,
//! `GPU=1 cargo run --release -p cute-flashinfer --features cuda --example quantization_fp8`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
use flashinfer::quantization::*;

const ROW: QuantMode = QuantMode { column_major: false, ue8m0: false };
const COL: QuantMode = QuantMode { column_major: true, ue8m0: false };
const UE8M0: QuantMode = QuantMode { column_major: true, ue8m0: true };

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    for dtype in [Dtype::F16, Dtype::BF16] {
        for out in [Out8::E4M3, Out8::E5M2, Out8::Int8] {
            for mode in [ROW, COL, UE8M0] {
                for gs in [64usize, 128, 256] {
                    let p = GroupQuantProblem::new(dtype, out, 37, 1024, gs, mode, 3);
                    let (bad, _) = gpu::run_group_quant(&ctx, &tc, &p, false)?;
                    if bad != 0 {
                        return Err(format!("{dtype:?} {out:?} {mode:?} g{gs}: {bad} wrong").into());
                    }
                }
            }
        }
    }
    println!("check: all dtypes, output types, layouts and group sizes bit-exact");
    println!("\nper_token_group_quant_8bit, bf16 -> e4m3, group 128 (CUDA-graph replays)");
    println!("mode     tokens x hidden   graph(us)  GB/s (in + out)");
    for (name, mode) in [("row", ROW), ("col", COL), ("ue8m0", UE8M0)] {
        for hidden in [4096usize, 8192] {
            for tokens in [1usize, 16, 128, 1024, 8192] {
                let p = GroupQuantProblem::new(Dtype::BF16, Out8::E4M3, tokens, hidden, 128, mode, 1);
                let (bad, t) = gpu::run_group_quant(&ctx, &tc, &p, true)?;
                assert_eq!(bad, 0);
                let t = t.unwrap();
                let bytes = (tokens * hidden * 3 + p.num_groups() * 4) as f64;
                println!("{name:<8} {tokens:5} x {hidden:<6}  {t:9.2}  {:8.1}", bytes / (t * 1e-6) / 1e9);
            }
        }
    }
    println!("\nfp16 -> e5m2 / int8, row scales, 8192 x 4096");
    for out in [Out8::E5M2, Out8::Int8] {
        let p = GroupQuantProblem::new(Dtype::F16, out, 8192, 4096, 128, ROW, 1);
        let (bad, t) = gpu::run_group_quant(&ctx, &tc, &p, true)?;
        assert_eq!(bad, 0);
        println!("{:<5} {:9.2} us", out.name(), t.unwrap());
    }
    Ok(())
}

fn main() {
    if std::env::var_os("PTX").is_some() {
        let tc = Toolchain::discover().unwrap();
        let k = group_quant_kernel(Dtype::BF16, Out8::E4M3, 128, ROW);
        println!("{}", k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx());
        return;
    }
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for out in [Out8::E4M3, Out8::E5M2, Out8::Int8] {
        for mode in [ROW, COL, UE8M0] {
            let p = GroupQuantProblem::new(Dtype::BF16, out, 5, 512, 128, mode, 1);
            let bad = simulate_group_quant(&p);
            println!("per_token_group_quant_8bit {out:?} {mode:?}: {bad} wrong");
            assert_eq!(bad, 0);
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
            let k = group_quant_kernel(Dtype::BF16, Out8::E4M3, 128, UE8M0);
            let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
            println!(
                "PTX {}: {} bytes, {} ld.global.v4, {} st.global.v2, {} shfl",
                k.name(),
                ptx.len(),
                ptx.matches("ld.global.v4").count(),
                ptx.matches("st.global.v2").count(),
                ptx.matches("shfl.sync").count()
            );
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

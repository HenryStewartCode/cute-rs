// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `packbits` and `segment_packbits` ([`flashinfer::quantization`]): simulated
//! on small inputs, then lowered to PTX if a toolchain is found. With `--features cuda`
//! and `GPU=1` it runs on the GPU instead: exact checks, then CUDA-graph timings to compare
//! with `quantization_flashinfer_bench.py` (FlashInfer has no packbits benchmark: the shapes
//! are attention-mask sizes, `qo_len * kv_len` bits).
//!
//! `cargo run -p cute-flashinfer --example quantization`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example quantization`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
use flashinfer::quantization::*;

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    println!("device: {}", ctx.name());
    for order in [Order::Big, Order::Little] {
        let ks = gpu::Kernels::new(&ctx, &tc, order)?;
        for lens in [vec![1usize], vec![13], vec![5000], vec![1 << 20], vec![4, 3, 4], vec![17, 0, 9000, 2048, 31]] {
            let p = Problem::new(&lens, 5);
            let segmented = lens.len() > 1;
            let (bad, _) = gpu::run(&ctx, &ks, order, &p, segmented, false)?;
            println!("check {order:?} {} {lens:?}: {bad} wrong bytes", if segmented { "segment" } else { "pack" });
            if bad != 0 {
                return Err("packbits mismatch".into());
            }
        }
    }
    let ks = gpu::Kernels::new(&ctx, &tc, Order::Big)?;
    println!("\npackbits, big bit order (CUDA-graph replays)");
    println!("bits        graph(us)  GB/s (in + out)");
    for log in [16u32, 20, 24, 26] {
        let n = 1usize << log;
        let p = Problem::new(&[n], 1);
        let (bad, t) = gpu::run(&ctx, &ks, Order::Big, &p, false, true)?;
        assert_eq!(bad, 0);
        let t = t.unwrap();
        println!("2^{log:<9} {t:9.2}  {:8.1}", (n + n / 8) as f64 / (t * 1e-6) / 1e9);
    }
    println!("\nsegment_packbits, big bit order (CUDA-graph replays)");
    println!("segments x len   graph(us)  GB/s (in + out)");
    for (batch, len) in [(8usize, 4096usize), (64, 16385), (512, 4097), (64, 65536)] {
        let p = Problem::new(&vec![len; batch], 2);
        let (bad, t) = gpu::run(&ctx, &ks, Order::Big, &p, true, true)?;
        assert_eq!(bad, 0);
        let t = t.unwrap();
        let n = batch * len;
        println!("{batch:4} x {len:<8} {t:9.2}  {:8.1}", (n + n / 8) as f64 / (t * 1e-6) / 1e9);
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    for order in [Order::Big, Order::Little] {
        let p = Problem::new(&[5000], 1);
        let bad = simulate_pack(&p, order);
        println!("packbits {order:?}: {bad} wrong bytes");
        assert_eq!(bad, 0);
        let p = Problem::new(&[4, 3, 4, 3000, 17], 2);
        let bad = simulate_segment(&p, order);
        println!("segment_packbits {order:?}: {bad} wrong bytes");
        assert_eq!(bad, 0);
    }
    match Toolchain::discover() {
        Some(tc) => {
            let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
            for k in [pack_kernel(Order::Big), segment_kernel(Order::Big)] {
                let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).ptx().to_string();
                println!(
                    "PTX {}: {} bytes, {} ld.global.b64, {} ld.global.u8, {} mul.lo",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.b64").count() + ptx.matches("ld.global.u64").count(),
                    ptx.matches("ld.global.u8").count(),
                    ptx.matches("mul.lo").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

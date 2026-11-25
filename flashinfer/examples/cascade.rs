// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's `merge_state`, `merge_state_in_place` and `merge_states`
//! ([`flashinfer::cascade`]): simulated on small shapes, then lowered to PTX if a toolchain
//! is found. With `--features cuda` and `GPU=1` it runs on the GPU instead: correctness
//! checks against the f64 reference, then CUDA-graph timings (one L2-resident set of
//! buffers) on seq {1, 64, 4096} and heads {32, 64} of head dimension 128 (`heads * 128`
//! is 4096 and 8192; FlashInfer has no cascade benchmark, and its block of
//! `(head_dim / 8, heads)` threads caps `heads` at 64), merge_states of 4 and 64 states (64 only up to seq 64: seq 4096 would be 4 GB of states).
//!
//! `cargo run -p cute-flashinfer --example cascade`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example cascade`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::cascade::gpu::{Kernels, run};
use flashinfer::cascade::*;

const OPS: [Op; 3] = [Op::MergeState, Op::InPlace { masked: false }, Op::MergeStates];

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {}", ctx.name());
    // `QUICK=1` skips the checks and the shapes of seq 4096 (a few minutes in a debug build).
    let quick = std::env::var_os("QUICK").is_some();
    for dtype in if quick { vec![] } else { vec![Dtype::F16, Dtype::BF16] } {
        for (seq, heads, hd, sets) in [
            (5usize, 32usize, 128usize, 2usize),
            (7, 3, 64, 6),
            (3, 2, 256, 2),
            (9, 32, 128, 4),
            (4, 32, 128, 64),
            (1, 64, 128, 100),
        ] {
            let p = Problem::new(dtype, seq, heads, hd, sets, 11);
            for op in [OPS[0], OPS[1], Op::InPlace { masked: true }, OPS[2]] {
                let ((ev, es), _) = run(&ctx, &mut ks, &p, op, true, false)?;
                println!(
                    "check {} {:28} seq {seq} heads {heads} d {hd} sets {sets}: v {ev:.2e}, lse {es:.2e}",
                    dtype.name(),
                    op.name()
                );
            }
        }
    }
    println!("op                    sets  seq   heads  time(us)    GB/s");
    for op in OPS {
        for heads in [32usize, 64] {
            for sets in if op == Op::MergeStates { vec![4usize, 64] } else { vec![2] } {
                for seq in if sets == 64 || quick { vec![1usize, 64] } else { vec![1, 64, 4096] } {
                    let p = Problem::new(Dtype::F16, seq, heads, 128, sets, 3);
                    let (_, t) = run(&ctx, &mut ks, &p, op, false, true)?;
                    let t = t.unwrap();
                    println!(
                        "{:21} {sets:4}  {seq:5} {heads:5}  {t:8.2}  {:7.1}",
                        op.name(),
                        p.bytes(op) / (t * 1e-6) / 1e9
                    );
                }
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
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (seq, heads, hd, sets) in [(3usize, 2usize, 128usize, 2usize), (4, 2, 64, 5), (5, 2, 128, 3)] {
            let p = Problem::new(dtype, seq, heads, hd, sets, 7);
            for op in OPS {
                let (ev, es) = simulate(&p, op);
                println!(
                    "{} {:21} seq {seq} heads {heads} d {hd} sets {sets}: v {ev:.2e}, lse {es:.2e}",
                    dtype.name(),
                    op.name()
                );
            }
        }
    }
    match Toolchain::discover() {
        Some(tc) => {
            let ks = [
                merge_state_kernel(Dtype::F16, 128, 32),
                merge_state_in_place_kernel(Dtype::F16, 128, 32, false),
                merge_states_kernel(Dtype::F16, 128, 32),
                merge_states_large_kernel(Dtype::F16, 128),
            ];
            for k in ks {
                let ptx = k
                    .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .ptx()
                    .to_string();
                println!(
                    "PTX {}: {} bytes, {} 16-byte loads, {} cp.async, {} ex2, {} lg2",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.v4.b32").count(),
                    ptx.matches("cp.async").count(),
                    ptx.matches("ex2.approx").count(),
                    ptx.matches("lg2.approx").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

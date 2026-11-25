// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/red.py (the kernel
//   `kernel`, the launch of `host` and the host data of `run`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Reductions into global memory, ported from CuTe DSL's
//! `experimental/primitives/red.py`: each of `n` threads adds its element
//! of `src` into `out[0]` with `red` (relaxed at GPU scope; the example's
//! `prims.red`), and for Int32 also takes the maximum into `out[1]`. PTX has
//! no `red.max.f32`, so Float32 has the sum alone. `red` combines with what
//! the cells hold, so the host initialises them: 0 for the sum, the
//! smallest Int32 for the maximum.
//!
//! Run with `cargo run --release --example red`.

use cute::prelude::*;

/// The example's default element count (`_DEFAULT_N`) and block size
/// (`_DEFAULT_THREADS`).
pub const DEFAULT_N: usize = 4096;
pub const DEFAULT_THREADS: usize = 128;

/// The kernel for `n` elements of `src`, Int32 (`out` is `[sum, max]`) or
/// Float32 (`out` is `[sum]`), in blocks of `threads`.
pub fn build<T: Numeric>(n: usize, threads: usize) -> Kernel {
    let int = T::DTYPE == DType::I32;
    assert!(int || T::DTYPE == DType::F32, "red.py reduces Int32 or Float32");
    let name = if int { "red_s32" } else { "red_f32" };
    Kernel::builder(name).block_dim([threads as u32, 1, 1]).build(move || {
        let src = param_tensor::<T>("src", Layout::col_major(n));
        let out = param_tensor::<T>("out", Layout::col_major(if int { 2 } else { 1 }));
        let tidx = thread_idx().x;
        let bidx = block_idx().x;
        let idx = bidx * threads as i32 + tidx;
        let relaxed_gpu = AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Gpu);
        if_(idx.lt(n as i32), || {
            let value = src.load(idx);
            if int {
                red(&out.ptr(), value, RedOp::Add, AtomicType::S32, relaxed_gpu);
                red(&out.ptr().offset(1usize), value, RedOp::Max, AtomicType::S32, relaxed_gpu);
            } else {
                red(&out.ptr(), value, RedOp::Add, AtomicType::F32, relaxed_gpu);
            }
        });
    })
}

/// Enough blocks of `threads` for `n` elements (`host`).
pub fn launch(n: usize, threads: usize) -> LaunchConfig {
    LaunchConfig::new([n.div_ceil(threads) as u32, 1, 1], [threads as u32, 1, 1])
}

/// The example's data, `(i % 23) - 11`, and the initialised outputs.
pub fn s32_args(n: usize) -> Vec<KernelArg> {
    let src: Vec<i32> = (0..n).map(|i| (i % 23) as i32 - 11).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&[0, i32::MIN])]
}

pub fn f32_args(n: usize) -> Vec<KernelArg> {
    let src: Vec<f32> = (0..n).map(|i| (i % 23) as f32 - 11.0).collect();
    vec![KernelArg::f32(src), KernelArg::f32(vec![0.0])]
}

/// Check `args` after a run (`expected`): the Int32 sum and maximum, or the
/// Float32 sum, which small integers make exact in any order.
pub fn check(args: &[KernelArg]) {
    match &args[1] {
        KernelArg::F32(out) => {
            let src = args[0].as_f32();
            assert_eq!(out[..], [src.iter().sum::<f32>()], "the Float32 sum");
        }
        _ => {
            let (src, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
            let sum = src.iter().fold(0i32, |a, &b| a.wrapping_add(b));
            assert_eq!(out, [sum, *src.iter().max().unwrap()], "the Int32 sum and maximum");
        }
    }
}

/// Build, simulate and check both variants.
pub fn run(n: usize, threads: usize) -> (SimReport, SimReport) {
    let mut args = s32_args(n);
    let s32 = build::<i32>(n, threads)
        .simulate_all_orders(launch(n, threads), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    let mut args = f32_args(n);
    let f32 = build::<f32>(n, threads)
        .simulate_all_orders(launch(n, threads), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    (s32, f32)
}

#[allow(dead_code)]
fn main() {
    run(DEFAULT_N, DEFAULT_THREADS);
    println!("red: correct for s32 (sum, max) and f32 (sum), n = {DEFAULT_N}");
    match Toolchain::discover() {
        Some(tc) => match build::<i32>(DEFAULT_N, DEFAULT_THREADS).compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let reds: Vec<&str> = ptx.lines().map(str::trim).filter(|l| l.starts_with("red.")).collect();
                println!("PTX: {reds:?}");
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

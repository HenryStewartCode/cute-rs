// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/dsl_tutorials/call_bypass_dlpack.py (the wrapper
//   `tensor_op_gemm_wrapper`: pointer parameters, `m` and `n` assumed multiples of 8, the
//   ordered layouts of A, B and C over the pointers; and `run_tensor_op_gemm_wrapper`'s
//   permuted tensors and `make_ptr(Float16, data_ptr, gmem, assumed_align=32)` pointers),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Calling a kernel on raw pointers, bypassing DLPack, ported from CuTe DSL's
//! `dsl_tutorials/call_bypass_dlpack.py`: the kernel takes three Float16 pointers assumed
//! 32-byte aligned and the extents `m`, `n`, `k`, `l`, and builds its tensors itself, A
//! `(m, k, l)` and B `(n, k, l)` column-major and C `(m, n, l)` with `n` innermost, as the
//! tutorial's permuted PyTorch tensors are. The tutorial then calls the Ampere
//! `TensorOpGemm`; here each thread computes one element of `C[m, n, l] = sum_k A[m, k, l]
//! B[n, k, l]` in Float32, which keeps the point of the example, the wrapper over raw
//! pointers, in view.
//!
//! On the simulator the pointers are host buffers; on the GPU they are raw device addresses
//! (`DevicePtr::from_raw(DType::F16, address, Some(32), bytes)`, CuTe DSL's `make_ptr`),
//! which the launch checks against the parameters' 32-byte alignment.
//!
//! Run with `cargo run --release --example call_bypass_dlpack`.

use cute::prelude::*;

/// The wrapper's parameters: A, B and C, then `m`, `n`, `k` and `l`.
pub type Params = (Ptr<F16>, Ptr<F16>, Ptr<F16>, Int32, Int32, Int32, Int32);

/// Threads per block.
pub const THREADS: u32 = 128;

/// The kernel: `tensor_op_gemm_wrapper` with a one-element-per-thread GEMM.
pub fn build() -> TypedKernel<Params> {
    let ptr = PtrSpec { assumed_align: 32 };
    let s = ScalarSpec::Plain;
    Kernel::builder("call_bypass_dlpack").block_dim([THREADS, 1, 1]).jit(
        (ptr, ptr, ptr, s, s, s, s),
        |(a_ptr, b_ptr, c_ptr, m, n, k, l): Params| {
            // The tutorial assumes these to call its GEMM.
            let m = assume(m, Some(8), false);
            let n = assume(n, Some(8), false);
            let a = make_tensor(&a_ptr, make_ordered_layout((m, k, l), &[0, 1, 2]));
            let b = make_tensor(&b_ptr, make_ordered_layout((n, k, l), &[0, 1, 2]));
            let c = make_tensor(&c_ptr, make_ordered_layout((m, n, l), &[1, 0, 2]));
            let i = block_idx().x * THREADS as i32 + thread_idx().x;
            if_(i.lt(m * n * l), || {
                let (row, rest) = (i % m, i / m);
                let (col, batch) = (rest % n, rest / n);
                let acc = for_range_carry(0, k, 1, Float32::from(0.0f32), |kk, acc| {
                    acc + a.load((row, kk, batch)).to::<f32>() * b.load((col, kk, batch)).to::<f32>()
                });
                c.store((row, col, batch), acc.to::<F16>());
            });
        },
    )
}

/// The grid for an `(m, n, l)` C.
pub fn launch(m: usize, n: usize, l: usize) -> LaunchConfig {
    LaunchConfig::new([((m * n * l) as u32).div_ceil(THREADS), 1, 1], [THREADS, 1, 1])
}

/// A, B (small integers, exact in Float16 and in the sums) for `(m, n, k, l)`, in the
/// tutorial's memory orders: A and B column-major, C row-major per batch.
pub fn inputs(m: usize, n: usize, k: usize, l: usize) -> (Vec<F16>, Vec<F16>) {
    let a = (0..m * k * l).map(|i| F16::from_f32(((i * 7) % 5) as f32 - 2.0)).collect();
    let b = (0..n * k * l).map(|i| F16::from_f32(((i * 3) % 7) as f32 - 3.0)).collect();
    (a, b)
}

/// The reference C, in C's memory order (`(m, n, l)` with strides `(n, 1, m n)`).
pub fn reference(a: &[F16], b: &[F16], m: usize, n: usize, k: usize, l: usize) -> Vec<F16> {
    let mut c = vec![F16::from_f32(0.0); m * n * l];
    for bt in 0..l {
        for r in 0..m {
            for col in 0..n {
                let s: f32 =
                    (0..k).map(|kk| a[r + kk * m + bt * m * k].to_f32() * b[col + kk * n + bt * n * k].to_f32()).sum();
                c[col + r * n + bt * m * n] = F16::from_f32(s);
            }
        }
    }
    c
}

/// Simulate on host buffers and check against the reference; returns C.
pub fn run(m: usize, n: usize, k: usize, l: usize) -> Vec<F16> {
    let (a, b) = inputs(m, n, k, l);
    let kernel = build();
    let mut host = (
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::buffer(&vec![F16::from_f32(0.0); m * n * l]),
        m as i32,
        n as i32,
        k as i32,
        l as i32,
    );
    kernel.simulate(launch(m, n, l), &mut host).unwrap_or_else(|e| panic!("{e}"));
    let c = host.2.to_vec::<F16>();
    assert_eq!(c, reference(&a, &b, m, n, k, l));
    c
}

fn main() {
    run(64, 32, 16, 2);
    println!("call_bypass_dlpack: C = A B on raw pointers, checked against the reference");
}

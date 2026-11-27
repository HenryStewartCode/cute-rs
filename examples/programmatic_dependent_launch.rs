// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/dsl_tutorials/programmatic_dependent_launch.py (the kernel
//   `elementwise_add_kernel`, the host function `elementwise_add` and the two launches of
//   `run_pdl_example`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Programmatic dependent launch (PDL), ported from CuTe DSL's
//! `dsl_tutorials/programmatic_dependent_launch.py`: two element-wise additions back to
//! back in one stream, `w = u + v` then `y = w + x`. The second kernel is launched with
//! `use_pdl`, so it may start while the first still runs: it loads `x`, which the first
//! kernel does not touch, before [`griddepcontrol_wait`], and loads `w` (the first kernel's
//! output) after it. The first kernel calls [`griddepcontrol_launch_dependents`] once it has
//! loaded its inputs, a hint that lets the second be scheduled early. As in the tutorial,
//! each copy runs ten times to make the kernels long enough to overlap.
//!
//! The griddepcontrol instructions need sm_90, so the SM89 test GPU cannot run the pair; the
//! simulator runs it through a [`SimStream`], whose PDL model checks that nothing the second
//! kernel does before its wait conflicts with what the first reads and writes, and `ptxas`
//! assembles both kernels for sm_90.
//!
//! Run with `cargo run --release --example programmatic_dependent_launch`.

use cute::prelude::*;

/// The tutorial's default problem (`--M 256 --N 256`).
pub const M: usize = 256;
pub const N: usize = 256;

/// The kernel parameters: A, B and C.
pub type Params = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// An `m x n` row-major tensor over `values`, every extent and the row stride dynamic
/// (`from_dlpack(t).mark_layout_dynamic()`).
pub fn tensor(values: &[f32], m: usize, n: usize) -> HostTensor {
    HostTensor::from_slice(values, Layout::row_major((m, n)))
        .mark_layout_dynamic(None)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The thread and value layouts: 4 x 32 threads (row-major) by 4 x 4 values.
fn tv() -> (Layout, Layout) {
    (make_ordered_layout((4, 32), &[1, 0]), make_ordered_layout((4, 128 / 32), &[1, 0]))
}

/// `elementwise_add` with `is_first_kernel` (a build-time constant, so each kernel is built
/// separately, as the tutorial compiles each).
pub fn build(is_first_kernel: bool) -> TypedKernel<Params> {
    let host = (tensor(&[0.0; 4], 2, 2), tensor(&[0.0; 4], 2, 2), tensor(&[0.0; 4], 2, 2));
    let name = if is_first_kernel { "pdl_first_kernel" } else { "pdl_second_kernel" };
    Kernel::builder(name).block_dim([128, 1, 1]).jit_for(
        &host,
        move |m_a: Tensor<f32>, m_b: Tensor<f32>, m_c: Tensor<f32>| {
            let (thr_layout, val_layout) = tv();
            let (tiler_mn, _) = make_layout_tv(&thr_layout, &val_layout);
            let g_a = zipped_divide(&m_a, &tiler_mn);
            let g_b = zipped_divide(&m_b, &tiler_mn);
            let g_c = zipped_divide(&m_c, &tiler_mn);
            let c_c = zipped_divide(&make_identity_tensor(m_c.shape()), &tiler_mn);
            let shape = m_c.shape().clone();

            let tidx = thread_idx().x;
            let blk_coord = ((.., ..), block_idx().x);
            let (blk_a, blk_b, blk_c, blk_crd) =
                (g_a.slice(blk_coord), g_b.slice(blk_coord), g_c.slice(blk_coord), c_c.slice(blk_coord));
            let copy_atom_load = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let copy_atom_store = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let tiled_copy = make_tiled_copy_tv(&copy_atom_load, &thr_layout, &val_layout);
            let thr = tiled_copy.get_slice(tidx);
            let (thr_a, thr_b, thr_c) = (thr.partition_s(&blk_a), thr.partition_s(&blk_b), thr.partition_s(&blk_c));
            let frg_a = make_fragment_like::<f32, _>(&thr_a);
            let frg_b = make_fragment_like::<f32, _>(&thr_b);
            let frg_c = make_fragment_like::<f32, _>(&thr_c);
            let thr_crd = thr.partition_s(&blk_crd);
            let frg_pred = make_fragment::<bool>(thr_crd.shape().clone());
            for i in 0..frg_pred.size() {
                frg_pred.store(i, elem_less(thr_crd.coord(i), shape.clone()));
            }

            if is_first_kernel {
                for _ in 0..10 {
                    copy_if(&copy_atom_load, &frg_pred, &thr_a, &frg_a);
                    copy_if(&copy_atom_load, &frg_pred, &thr_b, &frg_b);
                }
                // A hint that the next kernel may start: no effect on correctness.
                griddepcontrol_launch_dependents();
            } else {
                // B does not depend on the previous kernel: load it while that kernel finishes.
                for _ in 0..10 {
                    copy_if(&copy_atom_load, &frg_pred, &thr_b, &frg_b);
                }
                // A is the previous kernel's output: wait for it to complete first.
                griddepcontrol_wait();
                for _ in 0..10 {
                    copy_if(&copy_atom_load, &frg_pred, &thr_a, &frg_a);
                }
            }
            for _ in 0..10 {
                frg_c.store_ssa(&(frg_a.load_ssa() + frg_b.load_ssa()));
            }
            copy_if(&copy_atom_store, &frg_pred, &frg_c, &thr_c);
        },
    )
}

/// The launch of an `m x n` problem: one block per tile (`size(gC, mode=[1])`), with `use_pdl`.
pub fn launch(m: usize, n: usize, use_pdl: bool) -> LaunchConfig {
    let (thr_layout, val_layout) = tv();
    let (tiler_mn, _) = make_layout_tv(&thr_layout, &val_layout);
    let tiles = size(&zipped_divide(&Layout::row_major((m, n)), &tiler_mn), &[1]).as_static().expect("static");
    LaunchConfig::new([tiles as u32, 1, 1], [128, 1, 1]).with_pdl(use_pdl)
}

/// The arguments of `kernel` on the stream buffers `a`, `b` and `c` (`m x n` each): the
/// buffers, then the packed dynamic extents and strides.
pub fn stream_args(
    kernel: &TypedKernel<Params>,
    stream: &SimStream,
    [a, b, c]: [SimBuffer; 3],
    m: usize,
    n: usize,
) -> Vec<StreamArg> {
    let host = |s: SimBuffer| tensor(&stream.get(s).to_vec::<f32>(), m, n);
    let packed = kernel.pack(&(host(a), host(b), host(c))).unwrap_or_else(|e| panic!("{e}"));
    let mut buffers = [a, b, c].into_iter();
    packed
        .into_iter()
        .zip(&kernel.ir().args)
        .map(|(arg, p)| match p {
            cute::ir::Param::Buffer(_) => buffers.next().expect("three buffers").into(),
            _ => arg.into(),
        })
        .collect()
}

/// The inputs `u`, `v` and `x` of an `m x n` problem.
pub fn inputs(m: usize, n: usize) -> [Vec<f32>; 3] {
    let f = |k: usize| (0..m * n).map(|i| ((i * k) % 23) as f32 * 0.25 - 2.0).collect();
    [f(1), f(7), f(11)]
}

/// Run both kernels through a [`SimStream`], the second with `use_pdl`, and check
/// `y = u + v + x`; returns `y`.
pub fn run(m: usize, n: usize) -> Vec<f32> {
    let (first, second) = (build(true), build(false));
    let [u, v, x] = inputs(m, n);
    let mut stream = SimStream::new();
    let [u_b, v_b, x_b] = [&u, &v, &x].map(|d| stream.buffer(KernelArg::buffer(d)));
    let w_b = stream.buffer(KernelArg::buffer(&vec![0.0f32; m * n]));
    let y_b = stream.buffer(KernelArg::buffer(&vec![0.0f32; m * n]));
    let args = stream_args(&first, &stream, [u_b, v_b, w_b], m, n);
    stream.launch(&first, launch(m, n, true), &args).unwrap_or_else(|e| panic!("{e}"));
    let args = stream_args(&second, &stream, [w_b, x_b, y_b], m, n);
    stream.launch(&second, launch(m, n, true), &args).unwrap_or_else(|e| panic!("{e}"));
    let y = stream.get(y_b).to_vec::<f32>();
    for i in 0..m * n {
        assert_eq!(y[i], u[i] + v[i] + x[i], "y[{i}]");
    }
    y
}

fn main() {
    run(M, N);
    println!("programmatic_dependent_launch: y = u + v + x on {M}x{N}, the second kernel launched with use_pdl");
}

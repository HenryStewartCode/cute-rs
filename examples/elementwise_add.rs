// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py
//   (the tiled and dynamic kernels),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb (which the
//   naive kernel follows), a notebook with no header of its own: the README.md
//   of its folder carries
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and CUTLASS's LICENSE.txt
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Element-wise addition, ported from the CuTe DSL example
//! `elementwise_add.py` and its `elementwise_add` notebook.
//!
//! Three kernels compute `C = A + B` on an `M x N` row-major matrix:
//!
//! 1. `naive`: one element per thread, plain indexing.
//! 2. `tiled`: each block owns a 16x128 tile built from a thread layout and a
//!    value layout (`make_layout_tv`). Every thread copies a 4x4 block with
//!    128-bit vector loads, adds in registers and stores the result.
//!    `M` is deliberately not a multiple of the tile height, so the last row
//!    of tiles is predicated with an identity tensor, as in the example.
//!
//! 3. `dynamic`: the example as `elementwise_add.py` has it, on host tensors
//!    marked with `mark_layout_dynamic()`, so `M`, `N` and the row stride
//!    are launch-time values and one compiled kernel serves every size.
//!    The tensors are `zipped_divide`d by the TV tiler, each block takes
//!    the tile of its linear index in the rest mode, an identity tensor
//!    over `mC.shape` predicates the copies with `elem_less` against the
//!    runtime shape, the sum is a `TensorSsa` `a + b`, and the host sizes
//!    the grid as `size(gC, mode=[1])`. As in the example, the copies are
//!    one element wide: nothing is known of the runtime row stride's
//!    alignment.
//!
//! All three kernels are run in the simulator and checked against the CPU;
//! the simulator's sector counts show the effect of vectorisation. If an
//! `mlir-opt` is installed, the tiled kernel is also compiled to PTX.
//!
//! Run with `cargo run --example elementwise_add`.

use cute::prelude::*;

const M: usize = 300;
const N: usize = 512;

pub fn naive() -> Kernel {
    Kernel::builder("elementwise_add_naive").block_dim([256, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((M, N)));
        let b = param_tensor::<f32>("b", Layout::row_major((M, N)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));

        let idx = block_idx().x * 256 + thread_idx().x;
        let (m, n) = (idx / N as i32, idx % N as i32);
        if_(idx.lt((M * N) as i32), || {
            c.store((m, n), a.load((m, n)) + b.load((m, n)));
        });
    })
}

pub fn tiled() -> Kernel {
    // 4 x 32 threads (row-major), each owning 4 x 4 values (row-major):
    // a 16 x 128 tile per block.
    let thr_layout = Layout::row_major((4, 32));
    let val_layout = Layout::row_major((4, 4));
    let (tiler_mn, _) = make_layout_tv(&thr_layout, &val_layout);

    Kernel::builder("elementwise_add_tiled").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::row_major((M, N)));
        let b = param_tensor::<f32>("b", Layout::row_major((M, N)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let tidx = thread_idx().x;
        let bid = block_idx();

        // This block's tile of each tensor, plus the matching coordinates.
        let tile = |t: &Tensor<f32>| local_tile(t, &tiler_mn, (bid.x, bid.y));
        let (blk_a, blk_b, blk_c) = (tile(&a), tile(&b), tile(&c));
        let blk_coord = local_tile(&make_identity_tensor((M, N)), &tiler_mn, (bid.x, bid.y));

        // 128-bit loads and stores: four f32 per instruction.
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        let tiled_copy = make_tiled_copy_tv(&atom, &thr_layout, &val_layout);
        let thr = tiled_copy.get_slice(tidx);

        let thr_a = thr.partition_s(&blk_a);
        let thr_b = thr.partition_s(&blk_b);
        let thr_c = thr.partition_d(&blk_c);
        let thr_coord = thr.partition_s(&blk_coord);

        // Rows past M belong to the partial last tile and are skipped.
        let pred = make_predicate(&thr_coord, |crd| elem_less(crd, (M, N)));

        let frg_a = make_fragment_like::<f32, _>(&thr_a);
        let frg_b = make_fragment_like::<f32, _>(&thr_b);
        copy_if(&tiled_copy, &pred, &thr_a, &frg_a);
        copy_if(&tiled_copy, &pred, &thr_b, &frg_b);

        let frg_c = make_fragment_like::<f32, _>(&thr_c);
        for i in 0..frg_c.size() {
            frg_c.store(i, frg_a.load(i) + frg_b.load(i));
        }
        copy_if(&tiled_copy, &pred, &frg_c, &thr_c);
    })
}

/// The kernel parameters of [`dynamic`]: A, B and C.
pub type DynParams = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// The host tensors of [`dynamic`]: A, B and C, `m x n` row-major, with
/// every extent and the row stride dynamic (`mark_layout_dynamic()`).
pub fn dynamic_host(a: &[f32], b: &[f32], m: usize, n: usize) -> (HostTensor, HostTensor, HostTensor) {
    let tensor = |values: &[f32]| {
        HostTensor::from_slice(values, Layout::row_major((m, n)))
            .mark_layout_dynamic(None)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    (tensor(a), tensor(b), tensor(&vec![0.0; m * n]))
}

/// The thread and value layouts of [`dynamic`] (`copy_bits = 128` over
/// 32-bit elements): 4 x 32 threads by 4 x 4 values, both row-major.
fn dynamic_tv() -> (Layout, Layout) {
    (make_ordered_layout((4, 32), &[1, 0]), make_ordered_layout((4, 128 / 32), &[1, 0]))
}

/// The kernel of `elementwise_add.py`, for tensors of any `M x N`.
pub fn dynamic() -> TypedKernel<DynParams> {
    let host = dynamic_host(&[0.0; 6], &[0.0; 6], 2, 3);
    Kernel::builder("elementwise_add_dynamic").block_dim([128, 1, 1]).jit_for(
        &host,
        |m_a: Tensor<f32>, m_b: Tensor<f32>, m_c: Tensor<f32>| {
            // The host function (`elementwise_add`): tile by the TV layout.
            let (thr_layout, val_layout) = dynamic_tv();
            let (tiler_mn, _tv_layout) = make_layout_tv(&thr_layout, &val_layout);
            let g_a = zipped_divide(&m_a, &tiler_mn); // ((TileM,TileN),(RestM,RestN))
            let g_b = zipped_divide(&m_b, &tiler_mn);
            let g_c = zipped_divide(&m_c, &tiler_mn);
            let c_c = zipped_divide(&make_identity_tensor(m_c.shape()), &tiler_mn);
            let shape = m_c.shape().clone();

            // The kernel (`elementwise_add_kernel`).
            let tidx = thread_idx().x;
            let bidx = block_idx().x;
            let blk_coord = ((.., ..), bidx);
            let blk_a = g_a.slice(blk_coord); // (TileM,TileN)
            let blk_b = g_b.slice(blk_coord);
            let blk_c = g_c.slice(blk_coord);
            let blk_crd = c_c.slice(blk_coord);

            let copy_atom_load = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let copy_atom_store = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let tiled_copy_a = make_tiled_copy_tv(&copy_atom_load, &thr_layout, &val_layout);
            let tiled_copy_b = make_tiled_copy_tv(&copy_atom_load, &thr_layout, &val_layout);
            let tiled_copy_c = make_tiled_copy_tv(&copy_atom_store, &thr_layout, &val_layout);
            let thr_a = tiled_copy_a.get_slice(tidx).partition_s(&blk_a);
            let thr_b = tiled_copy_b.get_slice(tidx).partition_s(&blk_b);
            let thr_c = tiled_copy_c.get_slice(tidx).partition_s(&blk_c);
            let frg_a = make_fragment_like::<f32, _>(&thr_a);
            let frg_b = make_fragment_like::<f32, _>(&thr_b);
            let frg_c = make_fragment_like::<f32, _>(&thr_c);

            let thr_crd = tiled_copy_c.get_slice(tidx).partition_s(&blk_crd);
            let frg_pred = make_fragment::<bool>(thr_crd.shape().clone());
            for i in 0..frg_pred.size() {
                frg_pred.store(i, elem_less(thr_crd.coord(i), shape.clone()));
            }

            copy_if(&copy_atom_load, &frg_pred, &thr_a, &frg_a);
            copy_if(&copy_atom_load, &frg_pred, &thr_b, &frg_b);
            let result = frg_a.load_ssa() + frg_b.load_ssa();
            frg_c.store_ssa(&result);
            copy_if(&copy_atom_store, &frg_pred, &frg_c, &thr_c);
        },
    )
}

/// The grid of [`dynamic`] for an `m x n` C, computed on the host as the
/// example does: `size(gC, mode=[1])`, the number of tiles.
pub fn dynamic_grid(m: usize, n: usize) -> [u32; 3] {
    let (thr_layout, val_layout) = dynamic_tv();
    let (tiler_mn, _) = make_layout_tv(&thr_layout, &val_layout);
    let g_c = zipped_divide(&Layout::row_major((m, n)), &tiler_mn);
    [size(&g_c, &[1]).as_static().expect("a static size") as u32, 1, 1]
}

/// The inputs A and B of an `m x n` problem of [`dynamic`].
pub fn dynamic_inputs(m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * n).map(|i| (i % 97) as f32 * 0.5 + 0.125).collect();
    let b = (0..m * n).map(|i| (i % 13) as f32 - 6.0).collect();
    (a, b)
}

/// Simulate [`dynamic`] on an `m x n` problem and check C; returns the
/// report and C.
pub fn run_dynamic(kernel: &TypedKernel<DynParams>, m: usize, n: usize) -> (SimReport, Vec<f32>) {
    let (a, b) = dynamic_inputs(m, n);
    let mut host = dynamic_host(&a, &b, m, n);
    let cfg = LaunchConfig::new(dynamic_grid(m, n), [128, 1, 1]);
    let report = kernel.simulate(cfg, &mut host).unwrap_or_else(|e| panic!("{m}x{n}: {e}"));
    let c = host.2.to_vec::<f32>();
    for i in 0..m * n {
        assert_eq!(c[i], a[i] + b[i], "{m}x{n}: mismatch at {i}");
    }
    (report, c)
}

pub fn run(kernel: &Kernel, grid: [u32; 3], block: [u32; 3]) -> SimReport {
    let a: Vec<f32> = (0..M * N).map(|i| (i % 97) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..M * N).map(|i| (i % 13) as f32 - 6.0).collect();
    let mut args = vec![KernelArg::f32(a.clone()), KernelArg::f32(b.clone()), KernelArg::f32(vec![0.0; M * N])];
    let report = kernel
        .simulate_all_orders(LaunchConfig::new(grid, block), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let c = args[2].as_f32();
    for i in 0..M * N {
        assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
    }
    report
}

pub const NAIVE_GRID: [u32; 3] = [(M * N).div_ceil(256) as u32, 1, 1];
pub const TILED_GRID: [u32; 3] = [M.div_ceil(16) as u32, N.div_ceil(128) as u32, 1];

#[allow(dead_code)]
fn main() {
    let naive = naive();
    let report = run(&naive, NAIVE_GRID, [256, 1, 1]);
    println!("{}: correct\n{report}\n", naive.name());

    let tiled = tiled();
    let report = run(&tiled, TILED_GRID, [128, 1, 1]);
    println!("{}: correct\n{report}\n", tiled.name());

    let dynamic = dynamic();
    for (m, n) in [(300, 512), (3, 12), (17, 130), (64, 128)] {
        let (report, _) = run_dynamic(&dynamic, m, n);
        println!("{} {m}x{n}: correct, {} blocks", dynamic.name(), report.blocks);
    }
    println!();

    match Toolchain::discover() {
        Some(tc) => {
            for kernel in [&tiled, dynamic.kernel()] {
                match kernel.compile_ptx(&tc, Arch::Sm80) {
                    Ok(ptx) => {
                        let vector_loads = ptx.lines().filter(|l| l.contains("ld.global.v")).count();
                        println!(
                            "PTX for {}: {} lines, {vector_loads} vector loads",
                            kernel.name(),
                            ptx.lines().count()
                        );
                    }
                    Err(e) => eprintln!("PTX compilation failed: {e}"),
                }
            }
        }
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

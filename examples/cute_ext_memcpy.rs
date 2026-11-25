// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute_ext/ampere/memcpy_simt_universal_copy.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A memory copy with automatic predication, ported from the CuTe DSL
//! example `cute_ext/ampere/memcpy_simt_universal_copy.py`.
//!
//! `D = A + addend` over an `M x N x L` Float16 tensor whose M mode is
//! contiguous. The extents are runtime values (M a multiple of 8): one
//! compiled kernel serves every size. Each block takes a 128 x 64 tile;
//! each of its 128 threads copies eight 128-bit vectors of 8 elements into a
//! register buffer from `experimental::allocate`, adds, and copies back.
//! The thread's part of the tile comes from `experimental::partition` with
//! an explicit TV layout, and `experimental::copy` predicates the vectors
//! against the shape of the kernel's argument tensors without being told
//! to: the default shape (136, 7, 9) leaves partial tiles in M and N.
//!
//! Run with `cargo run --example cute_ext_memcpy [M N L]`.

use cute::experimental;
use cute::prelude::*;

/// The kernel parameters: A, D and the addend.
pub type Params = (Tensor<F16>, Tensor<F16>, Float16);

/// The example's default problem.
pub const DEFAULT_MNL: (usize, usize, usize) = (136, 7, 9);

/// The tile each block copies.
const TILE_M: usize = 128;
const TILE_N: usize = 64;

/// An `m x n x l` tensor with M contiguous (`(m, n, l):(1, m, m*n)`, the
/// example's permuted PyTorch tensor), marked as the example marks it:
/// 16-byte aligned, every mode dynamic and M's extent a multiple of 8.
pub fn host_tensor(values: &[F16], (m, n, l): (usize, usize, usize)) -> HostTensor {
    HostTensor::from_slice(values, Layout::new((m, n, l), (1, m, m * n)))
        .with_assumed_align(16)
        .mark_layout_dynamic(Some(0))
        .and_then(|t| t.mark_compact_shape_dynamic(0, Some(&[2, 1, 0]), 8))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The kernel (`memcpy_simt_universal_copy_kernel`).
pub fn kernel() -> TypedKernel<Params> {
    let shape = (8, 2, 2);
    let zeros = vec![F16::from_f32(0.0); 32];
    let host = (host_tensor(&zeros, shape), host_tensor(&zeros, shape), F16::from_f32(0.0));
    Kernel::builder("cute_ext_memcpy").block_dim([128, 1, 1]).jit_for(
        &host,
        |m_a: Tensor<F16>, m_d: Tensor<F16>, addend: Float16| {
            let tile_mn = Tiler::from(&IntTuple::from((TILE_M, TILE_N)));
            let g_a = zipped_divide(&m_a, tile_mn.clone()); // ((128,64),(RestM,RestN,L))
            let g_d = zipped_divide(&m_d, tile_mn);

            let bid = block_idx();
            let tid = thread_idx().x;
            let g_a_tile = g_a.slice(((.., ..), (bid.x, bid.y, bid.z)));
            let g_d_tile = g_d.slice(((.., ..), (bid.x, bid.y, bid.z)));

            // Eight vectors of eight elements per thread.
            let buffer =
                experimental::allocate::<F16>(AddressSpace::Rmem, Layout::new(((8, 1), (1, 8)), ((1, 8), (1, 8))), 16);

            // 16 x 8 threads, each owning 8 consecutive elements of M, over a
            // 128 x 8 tile.
            let layout_tv = Layout::new(((16, 8), (8, 1)), ((8, 128), (1, 1024)));
            let tiler = Tiler::from(&IntTuple::from((128, 8)));
            let t_g_a = experimental::partition(&g_a_tile, tid, &layout_tv, tiler.clone());
            let t_g_d = experimental::partition(&g_d_tile, tid, &layout_tv, tiler);

            let atom = make_copy_atom::<F16>(CopyOp::Universal { bits: 128 });
            experimental::copy(&t_g_a, &buffer, &atom);
            buffer.store_ssa(&(buffer.load_ssa() + addend));
            experimental::copy(&buffer, &t_g_d, &atom);
        },
    )
}

/// The grid for an `m x n x l` problem (the host function
/// `memcpy_simt_universal_copy`: the rest modes of `tiled_divide(src,
/// (128, 64))`).
pub fn grid((m, n, l): (usize, usize, usize)) -> [u32; 3] {
    [m.div_ceil(TILE_M) as u32, n.div_ceil(TILE_N) as u32, l as u32]
}

/// The launch for an `m x n x l` problem.
pub fn launch(mnl: (usize, usize, usize)) -> LaunchConfig {
    LaunchConfig::new(grid(mnl), [128, 1, 1])
}

/// The addend of the example.
pub const ADDEND: f32 = 5.0;

/// The host tensors of an `m x n x l` problem: A from a fixed sequence of
/// values, D filled with a marker that every element must lose.
pub fn problem(mnl: (usize, usize, usize)) -> (HostTensor, HostTensor, F16) {
    let (m, n, l) = mnl;
    let len = m * n * l;
    let a: Vec<F16> = (0..len).map(|i| F16::from_f32(((i * 37) % 1001) as f32 * 0.25 - 125.0)).collect();
    let d = vec![F16::from_f32(-7777.0); len];
    (host_tensor(&a, mnl), host_tensor(&d, mnl), F16::from_f32(ADDEND))
}

/// The expected D for A: `A + addend`, rounded to Float16.
pub fn reference(a: &[F16]) -> Vec<F16> {
    a.iter().map(|x| F16::from_f32(x.to_f32() + ADDEND)).collect()
}

/// Simulate an `m x n x l` problem and check D bit for bit; returns the
/// report and D.
pub fn run(kernel: &TypedKernel<Params>, mnl: (usize, usize, usize)) -> (SimReport, Vec<F16>) {
    let mut host = problem(mnl);
    let a = host.0.to_vec::<F16>();
    let report = kernel.simulate(launch(mnl), &mut host).unwrap_or_else(|e| panic!("{mnl:?}: {e}"));
    let d = host.1.to_vec::<F16>();
    let want = reference(&a);
    for (i, (got, want)) in d.iter().zip(&want).enumerate() {
        assert_eq!(got.0, want.0, "{mnl:?}: D[{i}] is {}, expected {}", got.to_f32(), want.to_f32());
    }
    (report, d)
}

#[allow(dead_code)]
fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("M N L")).collect();
    let mnl = match args.as_slice() {
        [m, n, l] => (*m, *n, *l),
        [] => DEFAULT_MNL,
        _ => panic!("usage: cute_ext_memcpy [M N L]"),
    };
    let kernel = kernel();
    let (report, _) = run(&kernel, mnl);
    println!("{} {mnl:?}: correct, {} blocks", kernel.name(), report.blocks);
    match Toolchain::discover() {
        Some(tc) => match kernel.kernel().compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let v4 = ptx.lines().filter(|l| l.contains("ld.global.v4") || l.contains("st.global.v4")).count();
                println!("PTX: {} lines, {v4} 128-bit global accesses", ptx.lines().count());
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
    println!("PASS");
}

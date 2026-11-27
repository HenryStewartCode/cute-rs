// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute_ext/ampere/sgemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A single-warp SM80 tensor-core GEMM in the fragment-free style, ported
//! from the CuTe DSL example `cute_ext/ampere/sgemm.py`.
//!
//! `D = A * B^T` with Float16 `A` (`M x K x L`, K-major), `B` (`N x K x L`,
//! K-major) and Float32 `D` (`M x N x L`, N-major). Each block is one warp
//! computing a 16 x 8 tile of D with the 16x8x8 `mma.sync` atom:
//!
//! - shared buffers for the A and B tiles and register buffers for the MMA
//!   fragments come from `experimental::allocate`;
//! - each K tile is copied global -> shared with 32-bit `cp.async`, through
//!   `experimental::partition` by the tiled copies' TV layouts and
//!   `experimental::copy`; the per-thread global views are marked with
//!   `predicated_tensor_origin`, so (being statically in bounds) they copy
//!   unpredicated;
//! - `experimental::partition_and_copy` loads the fragments with `ldmatrix`
//!   (`LdMatrix8x8x16bOp`, one matrix);
//! - `experimental::dot` accumulates with the tiled MMA;
//! - the epilogue is a universal copy of the accumulators to D, which
//!   `partition_and_copy` predicates against D's shape.
//!
//! As in the example, M, N and K are multiples of the tile (16, 8 and 8).
//! The example's epilogue atom copies 16 bits of a Float32 accumulator;
//! here it is 32 bits, one element (`partition_and_copy` picks the vector
//! width of a register -> global copy itself). The example aligns the
//! global pointers to 4 bytes (`iterator.align(4)`); here the static
//! layouts prove it.
//!
//! Run with `cargo run --example cute_ext_sgemm [M N K L]`.

use cute::experimental::{self, partition, partition_and_copy, predicated_tensor_origin};
use cute::prelude::*;

/// The example's default problem (M, N, K, L).
pub const DEFAULT_MNKL: (usize, usize, usize, usize) = (64, 32, 32, 2);

/// Build the kernel (`sgemm_kernel`) for an `m x n x k x l` problem.
pub fn build((m, n, k, l): (usize, usize, usize, usize)) -> Kernel {
    assert!(
        m.is_multiple_of(16) && n.is_multiple_of(8) && k.is_multiple_of(8),
        "cute_ext_sgemm: M, N and K must be multiples of 16, 8 and 8"
    );
    Kernel::builder("cute_ext_sgemm").block_dim([32, 1, 1]).build(|| {
        let m_a = param_tensor::<F16>("a", Layout::new((m, k, l), (k, 1, m * k)));
        let m_b = param_tensor::<F16>("b", Layout::new((n, k, l), (k, 1, n * k)));
        let m_d = param_tensor::<f32>("d", Layout::new((m, n, l), (n, 1, m * n)));

        let tiler = |a: usize, b: usize| Tiler::from(&IntTuple::from((a, b)));
        let g_a = zipped_divide(&m_a, tiler(16, 8)); // ((16,8),(RestM,RestK,L))
        let g_b = zipped_divide(&m_b, tiler(8, 8));
        let g_d = zipped_divide(&m_d, tiler(16, 8));
        let k_tiles = size(&g_a, &[1, 1]).as_static().expect("a static K");

        let bid = block_idx();
        let tid = thread_idx().x;
        let g_a_tile = g_a.slice(((.., ..), (bid.x, .., bid.z))); // (16,8,RestK)
        let g_b_tile = g_b.slice(((.., ..), (bid.y, .., bid.z)));
        let g_d_tile = g_d.slice(((.., ..), (bid.x, bid.y, bid.z))); // (16,8)

        let s_a = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((16, 8), (8, 1)), 16);
        let s_b = experimental::allocate::<F16>(AddressSpace::Smem, Layout::new((8, 8), (8, 1)), 16);

        let g2s_atom = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 4, bypass_l1: false });
        let g2s_a = make_tiled_copy_tv(&g2s_atom, &Layout::new((16, 2), (2, 1)), &Layout::col_major((1, 4)));
        let g2s_b = make_tiled_copy_tv(&g2s_atom, &Layout::new((8, 4), (4, 1)), &Layout::col_major((1, 2)));
        let t_a_s_a = partition(&s_a, tid, &g2s_a.layout_dst_tv_tiled(), g2s_a.tiler_mn().clone());
        let t_b_s_b = partition(&s_b, tid, &g2s_b.layout_dst_tv_tiled(), g2s_b.tiler_mn().clone());

        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 8)), Layout::col_major((1, 1, 1)));
        let r_a =
            experimental::allocate::<F16>(AddressSpace::Rmem, Layout::col_major(mma.partition_shape_a((16, 8))), 16);
        let r_b =
            experimental::allocate::<F16>(AddressSpace::Rmem, Layout::col_major(mma.partition_shape_b((8, 8))), 16);
        let r_c =
            experimental::allocate::<f32>(AddressSpace::Rmem, Layout::col_major(mma.partition_shape_c((16, 8))), 16);
        r_c.fill(0.0f32);

        let ldsm = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 1, trans: false });
        let s2r_a = make_tiled_copy_a(&ldsm, &mma);
        let s2r_b = make_tiled_copy_b(&ldsm, &mma);

        for kt in 0..k_tiles {
            let g_a_k = g_a_tile.slice((.., .., kt));
            let g_b_k = g_b_tile.slice((.., .., kt));
            let t_a_g_a = partition(&g_a_k, tid, &g2s_a.layout_src_tv_tiled(), g2s_a.tiler_mn().clone());
            let t_a_g_a = predicated_tensor_origin(&t_a_g_a);
            let t_b_g_b = partition(&g_b_k, tid, &g2s_b.layout_src_tv_tiled(), g2s_b.tiler_mn().clone());
            let t_b_g_b = predicated_tensor_origin(&t_b_g_b);

            experimental::copy(&t_a_g_a, &t_a_s_a, &g2s_atom);
            experimental::copy(&t_b_g_b, &t_b_s_b, &g2s_atom);
            cp_async_commit_group();
            cp_async_wait_group(0);
            sync_threads();

            partition_and_copy(&s2r_a.get_slice(tid), &s_a, &r_a);
            partition_and_copy(&s2r_b.get_slice(tid), &s_b, &r_b);
            experimental::dot(&mma, &r_a, &r_b, &r_c);
            sync_threads();
        }

        let r2g = make_tiled_copy_c(&make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }), &mma);
        partition_and_copy(&r2g.get_slice(tid), &r_c, &g_d_tile);
    })
}

/// The launch for an `m x n x k x l` problem (the host function `sgemm`:
/// one block per 16 x 8 tile of D).
pub fn launch((m, n, _, l): (usize, usize, usize, usize)) -> LaunchConfig {
    LaunchConfig::new([(m / 16) as u32, (n / 8) as u32, l as u32], [32, 1, 1])
}

/// The inputs of a problem: `A` and `B` as Float32 values that Float16
/// holds exactly, small integers or random in `[-1, 1)`.
pub fn inputs((m, n, k, l): (usize, usize, usize, usize), small: bool, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let x = (state >> 40) as u32;
        if small { (x % 9) as f32 - 4.0 } else { F16::from_f32(x as f32 / (1u32 << 23) as f32 - 1.0).to_f32() }
    };
    let a = (0..m * k * l).map(|_| next()).collect();
    let b = (0..n * k * l).map(|_| next()).collect();
    (a, b)
}

/// The kernel arguments for `a` and `b`, and a D filled with a marker.
pub fn args(a: &[f32], b: &[f32], len_d: usize) -> Vec<KernelArg> {
    vec![KernelArg::f16_from_f32(a), KernelArg::f16_from_f32(b), KernelArg::f32(vec![-7777.0; len_d])]
}

/// `D = A * B^T` per batch, summed in f64.
pub fn reference((m, n, k, l): (usize, usize, usize, usize), a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut d = vec![0.0; m * n * l];
    for b_ in 0..l {
        for i in 0..m {
            for j in 0..n {
                d[b_ * m * n + i * n + j] =
                    (0..k).map(|p| a[b_ * m * k + i * k + p] as f64 * b[b_ * n * k + j * k + p] as f64).sum::<f64>();
            }
        }
    }
    d.to_vec()
}

/// The bound on a Float32 sum of `k` products against the exact value: the
/// usual `k * eps * sum |a_i b_i|` (the products of two Float16 values are
/// exact in Float32).
pub fn within_bound(mnkl: (usize, usize, usize, usize), a: &[f32], b: &[f32], got: &[f32]) -> Result<(), String> {
    let (m, n, k, l) = mnkl;
    let want = reference(mnkl, a, b);
    for b_ in 0..l {
        for i in 0..m {
            for j in 0..n {
                let idx = b_ * m * n + i * n + j;
                let mag: f64 =
                    (0..k).map(|p| (a[b_ * m * k + i * k + p] as f64 * b[b_ * n * k + j * k + p] as f64).abs()).sum();
                let bound = k as f64 * f32::EPSILON as f64 * mag + 1e-30;
                let err = (got[idx] as f64 - want[idx]).abs();
                if err > bound {
                    return Err(format!("D[{idx}] = {} differs from {} by {err}, beyond {bound}", got[idx], want[idx]));
                }
            }
        }
    }
    Ok(())
}

/// Simulate a problem and check D (exactly for small integers, within the
/// summation bound otherwise); returns D.
pub fn run(kernel: &Kernel, mnkl: (usize, usize, usize, usize), small: bool) -> Vec<f32> {
    let (m, n, _, l) = mnkl;
    let (a, b) = inputs(mnkl, small, 1111);
    let mut args = args(&a, &b, m * n * l);
    kernel.simulate(launch(mnkl), &mut args).unwrap_or_else(|e| panic!("{mnkl:?}: {e}"));
    let d = args[2].as_f32();
    if small {
        let want = reference(mnkl, &a, &b);
        for (i, (&got, &want)) in d.iter().zip(&want).enumerate() {
            assert_eq!(got as f64, want, "{mnkl:?}: D[{i}]");
        }
    } else {
        within_bound(mnkl, &a, &b, d).unwrap_or_else(|e| panic!("{mnkl:?}: {e}"));
    }
    d.to_vec()
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("M N K L")).collect();
    let mnkl = match args.as_slice() {
        [m, n, k, l] => (*m, *n, *k, *l),
        [] => DEFAULT_MNKL,
        _ => panic!("usage: cute_ext_sgemm [M N K L]"),
    };
    let kernel = build(mnkl);
    run(&kernel, mnkl, true);
    run(&kernel, mnkl, false);
    println!("{} {mnkl:?}: correct", kernel.name());
    match Toolchain::discover() {
        Some(tc) => match kernel.compile_ptx(&tc, Arch::Sm80) {
            Ok(ptx) => {
                let count = |s: &str| ptx.lines().filter(|l| l.contains(s)).count();
                println!(
                    "PTX: {} lines, {} cp.async, {} ldmatrix, {} mma",
                    ptx.lines().count(),
                    count("cp.async.ca"),
                    count("ldmatrix"),
                    count("mma.sync")
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
    println!("PASS");
}

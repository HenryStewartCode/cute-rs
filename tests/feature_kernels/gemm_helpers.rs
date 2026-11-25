// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The epilogue kernels follow the design of NVIDIA CUTLASS 4.8.1's
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! An accumulator fragment stored to shared memory through
//! the atom `utils::sm90::get_smem_store_op` selects, `stmatrix` for a 16-bit D (sm_90) and
//! the universal copy for f32 D (any target).

use cute::prelude::*;
use cute::utils::sm90::get_smem_store_op;

use super::Case;

const M: usize = 32;
const N: usize = 32;
const K: usize = 16;

/// A `2 x 2`-warp SM80 m16n8k16 TiledMma computes `A B^T` of a 32 x 32 tile; its accumulator
/// goes to a shared tile of `layout_d`'s majorness through `get_smem_store_op` (converted to
/// `D` first), and the tile is copied to global memory.
fn epilogue<D: Numeric>(name: &'static str, layout_d: LayoutEnum, check: fn(&[KernelArg])) -> Case {
    let kernel = Kernel::builder(name).block_dim([128, 1, 1]).build(move || {
        let a = param_tensor::<F16>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F16>("b", Layout::row_major((N, K)));
        let c_layout = match layout_d {
            LayoutEnum::ROW_MAJOR => Layout::row_major((M, N)),
            LayoutEnum::COL_MAJOR => Layout::col_major((M, N)),
        };
        let c = param_tensor::<D>("c", c_layout.clone());
        let t = thread_idx().x;
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)))
            .with_permutation_mnk([None, Some(Layout::new(32, 1)), None]);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) = (thr_mma.partition_fragment_a::<F16, _>(&a), thr_mma.partition_fragment_b::<F16, _>(&b));
        let (pa, pb) = (thr_mma.partition_a(&a), thr_mma.partition_b(&b));
        for i in 0..fa.size() {
            fa.store(i, pa.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, pb.load(i));
        }
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let accd = make_fragment_like::<D, f32>(&acc);
        for i in 0..acc.size() {
            accd.store(i, acc.load(i).to::<D>());
        }
        let atom = get_smem_store_op::<D>(layout_d, DType::F32);
        // The tile the epilogue covers per copy comes from a wider atom when the store op moves
        // one element: two accumulators per thread (a row pair of the MMA's fragment).
        let c_atom = if atom.num_val_src() == 1 {
            make_tiled_copy_c_atom(&make_copy_atom::<D>(CopyOp::Universal { bits: 2 * D::DTYPE.bits() }), &mma)
        } else {
            make_tiled_copy_c_atom(&atom, &mma)
        };
        let r2s = make_tiled_copy_s(&atom, &c_atom);
        let sc = make_smem_tensor::<D>(c_layout);
        copy(&r2s, &r2s.retile(&accd), &r2s.get_slice(t).partition_d(&sc));
        sync_threads();
        for i in 0..M * N / 128 {
            let k = t + 128 * i as i32;
            c.store((k % M as i32, k / M as i32), sc.load((k % M as i32, k / M as i32)));
        }
    });
    Case { name, kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check }
}

fn reference(args: &[KernelArg]) -> Vec<f32> {
    let (a, b) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
    (0..M * N)
        .map(|k| {
            let (i, j) = (k % M, k / M);
            (0..K).map(|l| a[i * K + l].to_f32() * b[j * K + l].to_f32()).sum()
        })
        .collect()
}

/// `stmatrix.x4` for an N-major (row-major) f16 D.
pub fn gemm_helper_stmatrix_n_major() -> Case {
    fn check(args: &[KernelArg]) {
        let c = args[2].to_vec::<F16>();
        for (k, want) in reference(args).into_iter().enumerate() {
            let (i, j) = (k % M, k / M);
            assert_eq!(c[i * N + j], F16::from_f32(want), "C[{i},{j}]");
        }
    }
    epilogue::<F16>("gemm_helper_stmatrix_n_major", LayoutEnum::ROW_MAJOR, check)
}

/// `stmatrix.x4.trans` for an M-major (column-major) f16 D.
pub fn gemm_helper_stmatrix_m_major() -> Case {
    fn check(args: &[KernelArg]) {
        let c = args[2].to_vec::<F16>();
        for (k, want) in reference(args).into_iter().enumerate() {
            assert_eq!(c[k], F16::from_f32(want), "C[{},{}]", k % M, k / M);
        }
    }
    epilogue::<F16>("gemm_helper_stmatrix_m_major", LayoutEnum::COL_MAJOR, check)
}

/// The universal copy for an f32 D.
pub fn gemm_helper_universal_f32() -> Case {
    fn check(args: &[KernelArg]) {
        let c = args[2].to_vec::<f32>();
        for (k, want) in reference(args).into_iter().enumerate() {
            let (i, j) = (k % M, k / M);
            assert_eq!(c[i * N + j], want, "C[{i},{j}]");
        }
    }
    epilogue::<f32>("gemm_helper_universal_f32", LayoutEnum::ROW_MAJOR, check)
}

/// The cases that need sm_90 (`stmatrix`).
pub fn gemm_helper_cases_sm90() -> Vec<Case> {
    vec![gemm_helper_stmatrix_n_major(), gemm_helper_stmatrix_m_major()]
}

/// The cases for any target, the SM89 GPU included.
pub fn gemm_helper_cases_sm89() -> Vec<Case> {
    vec![gemm_helper_universal_f32()]
}

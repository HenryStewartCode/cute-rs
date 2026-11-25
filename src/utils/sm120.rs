// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm120.hpp (the SFALayout and SFBLayout
//   scale-factor atoms),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! CuTe DSL's SM120 block-scaled helpers from `cutlass.utils`
//! (`blackwell_helpers`): the tiled-MMA
//! permutation, the scale-factor thread partitions that give each thread
//! the scales a warp-level block-scaled `mma.sync`
//! ([`warp::MmaMXF4Op`](crate::warp::MmaMXF4Op) and the others) expects,
//! their thread-value layouts for tiled copies, and the epilogue's
//! shared-memory store atom. The shared-memory scale-factor layouts are
//! [`sm120_make_smem_layout_sfa`](crate::utils::sm120_make_smem_layout_sfa)
//! and [`sm120_make_smem_layout_sfb`](crate::utils::sm120_make_smem_layout_sfb).
//!
//! A scale-factor tensor is an `(M, K)` (or `(N, K)`) view whose layout
//! repeats each scale over its vector of K elements (stride 0), as
//! [`tile_atom_to_shape_sf`](crate::utils::tile_atom_to_shape_sf) and the
//! shared-memory layouts give. Its atom (CuTe's `SFALayout`) gives lanes
//! `4 g` and `4 g + 2` row `g`'s scales and lanes `4 g + 1` and `4 g + 3`
//! row `g + 8`'s (`SFBLayout`: every lane of quad `g` column `g`'s), one
//! value per K element of the instruction.

use crate::atom::copy::{CopyAtom, CopyOp, make_copy_atom};
use crate::atom::mma::{MmaOp, Operand, ThrMma, TiledMma};
use crate::ir::DType;
use crate::layout::Layout;
use crate::tensor::{Tensor, make_fragment_like};
use crate::utils::LayoutEnum;
use crate::value::Numeric;

/// CuTe DSL's `get_permutation_mnk(tile_shape_mnk, sf_vec_size,
/// use_mxf8f6f4)`: the tiled MMA's permutation for a CTA tile, M `min(M,
/// 128)`, N `(8,2,2):(1,16,8)` (32 columns, interleaved so a thread's
/// scale factors for B are adjacent) and K 32 (`kind::mxf8f6f4`) or 64
/// (FP4), for [`TiledMma::with_permutation_mnk`]. Panics unless
/// `sf_vec_size` is 16 or 32.
#[track_caller]
pub fn get_permutation_mnk(
    tile_shape_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    use_mxf8f6f4: bool,
) -> [Option<Layout>; 3] {
    assert!(
        sf_vec_size == 16 || sf_vec_size == 32,
        "cute: get_permutation_mnk: 'sf_vec_size' is {sf_vec_size}; SM120 block-scaled MMAs take 16 or 32"
    );
    let perm_m = tile_shape_mnk.0.min(128);
    let perm_k = if use_mxf8f6f4 { 32 } else { 64 };
    [Some(Layout::new(perm_m, 1)), Some(Layout::new((8, 2, 2), (1, 16, 8))), Some(Layout::new(perm_k, 1))]
}

fn sm120_kind(tiled_mma: &TiledMma, what: &str) -> usize {
    match tiled_mma.atom().op() {
        MmaOp::Sm120(kind) => kind.shape_mnk().2,
        op => panic!("cute: {what}: {op:?} is not an SM120 block-scaled warp MMA"),
    }
}

/// CuTe's `SFALayout` for an atom of K extent `k` (32 or 64): `(T32, V_K)
/// -> (M16, K)`. For K 32 the value mode is `(32, 1)`, so that FP8
/// fragments have the rank of FP4 ones.
fn sfa_atom(k: usize) -> Layout {
    if k == 32 {
        Layout::new(((2, 2, 8), (32, 1)), ((8, 0, 1), (16, 0)))
    } else {
        Layout::new(((2, 2, 8), 64), ((8, 0, 1), 16))
    }
}

/// CuTe's `SFBLayout`: `(T32, V_K) -> (N8, K)`.
fn sfb_atom(k: usize) -> Layout {
    if k == 32 { Layout::new(((4, 8), (32, 1)), ((0, 1), (8, 0))) } else { Layout::new(((4, 8), 64), ((0, 1), 8)) }
}

/// CuTe DSL's `thrfrg_SFA(sfa, tiled_mma)` of a layout: `((ThrV, (ThrM,
/// ThrK)), (FrgV, (RestM, RestK)))` over an `(M, K)` scale-factor layout.
#[track_caller]
pub fn thrfrg_sfa(sfa: &Layout, tiled_mma: &TiledMma) -> Layout {
    let k = sm120_kind(tiled_mma, "thrfrg_SFA");
    tiled_mma.thrfrg_with_atom(sfa, Operand::A, &sfa_atom(k), 16, k)
}

/// CuTe DSL's `thrfrg_SFB(sfb, tiled_mma)`: `((ThrV, (ThrN, ThrK)), (FrgV,
/// (RestN, RestK)))` over an `(N, K)` scale-factor layout.
#[track_caller]
pub fn thrfrg_sfb(sfb: &Layout, tiled_mma: &TiledMma) -> Layout {
    let k = sm120_kind(tiled_mma, "thrfrg_SFB");
    tiled_mma.thrfrg_with_atom(sfb, Operand::B, &sfb_atom(k), 8, k)
}

/// CuTe DSL's `partition_fragment_SFA(sfa, thr_mma, tidx)` (the thread is
/// `thr_mma`'s): a register fragment for this thread's scale factors of
/// the `(M, K)` scale-factor tile `sfa` (static), `(MMA_SF, MMA_M, MMA_K)`
/// with the residual K modes grouped (as CuTe DSL kernels group them
/// afterwards), keeping the stride-0 repeats so that it holds one register
/// per scale. It is what [`gemm`](crate::gemm) takes with A, `(&a, &sfa)`.
#[track_caller]
pub fn partition_fragment_sfa<S: Numeric>(sfa: &Tensor<S>, thr_mma: &ThrMma) -> Tensor<S> {
    let l = thrfrg_sfa(sfa.layout(), thr_mma.tiled_mma());
    make_fragment_like::<S, S>(&group_rest(thr_mma.slice_thrfrg(sfa, l, Operand::A)))
}

/// CuTe DSL's `partition_fragment_SFB(sfb, thr_mma, tidx)`: as
/// [`partition_fragment_sfa`] for the `(N, K)` scale factors of B,
/// `(MMA_SF, MMA_N, MMA_K)`.
#[track_caller]
pub fn partition_fragment_sfb<S: Numeric>(sfb: &Tensor<S>, thr_mma: &ThrMma) -> Tensor<S> {
    let l = thrfrg_sfb(sfb.layout(), thr_mma.tiled_mma());
    make_fragment_like::<S, S>(&group_rest(thr_mma.slice_thrfrg(sfb, l, Operand::B)))
}

/// This thread's partition of an `(M, K)` scale-factor tile (shared memory,
/// say), shaped as [`partition_fragment_sfa`]'s fragment: the source of a
/// per-element copy into it.
#[track_caller]
pub fn partition_sfa<S>(sfa: &Tensor<S>, thr_mma: &ThrMma) -> Tensor<S> {
    let l = thrfrg_sfa(sfa.layout(), thr_mma.tiled_mma());
    group_rest(thr_mma.slice_thrfrg(sfa, l, Operand::A))
}

/// As [`partition_sfa`] for the `(N, K)` scale factors of B.
#[track_caller]
pub fn partition_sfb<S>(sfb: &Tensor<S>, thr_mma: &ThrMma) -> Tensor<S> {
    let l = thrfrg_sfb(sfb.layout(), thr_mma.tiled_mma());
    group_rest(thr_mma.slice_thrfrg(sfb, l, Operand::B))
}

/// `(FrgV, RestMN, RestK...)` with the residual K modes grouped into one.
fn group_rest<S>(t: Tensor<S>) -> Tensor<S> {
    let r = t.layout().rank();
    if r > 3 { t.group_modes(2, r) } else { t }
}

/// CuTe DSL's `get_layoutSFA_TV(tiled_mma)`: `(thread index, value) ->`
/// index into the `(tile_m, tile_k)` scale-factor tile, for a tiled copy of
/// SFA into [`partition_fragment_sfa`]'s fragments (`make_tiled_copy(atom,
/// &get_layout_sfa_tv(&mma), (tile_m, tile_k))`).
#[track_caller]
pub fn get_layout_sfa_tv(tiled_mma: &TiledMma) -> Layout {
    let reference = Layout::col_major((tiled_mma.tile_size(0), tiled_mma.tile_size(2)));
    tiled_mma.layout_tv_of(thrfrg_sfa(&reference, tiled_mma), Operand::A)
}

/// CuTe DSL's `get_layoutSFB_TV(tiled_mma)`: as [`get_layout_sfa_tv`] over
/// the `(tile_n, tile_k)` tile.
#[track_caller]
pub fn get_layout_sfb_tv(tiled_mma: &TiledMma) -> Layout {
    let reference = Layout::col_major((tiled_mma.tile_size(1), tiled_mma.tile_size(2)));
    tiled_mma.layout_tv_of(thrfrg_sfb(&reference, tiled_mma), Operand::B)
}

/// CuTe DSL's `sm120_get_smem_store_op(layout_d, elem_ty_d, elem_ty_acc)`
/// (of its two definitions, the second, which wins): `stmatrix.x2` (with
/// `.trans` for an M-major D) for a 16-bit D, else a universal copy. The
/// accumulator type does not change the choice.
#[track_caller]
pub fn sm120_get_smem_store_op<D: Numeric>(layout_d: LayoutEnum, elem_ty_acc: DType) -> CopyAtom<D> {
    let _ = elem_ty_acc;
    if D::DTYPE.bits() == 16 {
        make_copy_atom::<D>(CopyOp::StMatrix { num: 2, trans: layout_d.is_m_major_c() })
    } else {
        make_copy_atom::<D>(CopyOp::Universal { bits: D::DTYPE.bits().max(8) })
    }
}

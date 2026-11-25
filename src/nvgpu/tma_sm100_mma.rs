// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The CTA value maps of the MMA-aware TMA atoms (the tile's identity layout partitioned
// by the MMA's thrfrg_A/B/C, keeping the value mode) and the multicast counts taken from
// the cluster's N mode for A and M mode for B are ported from NVIDIA CUTLASS 4.8.1's
//   include/cute/atom/copy_traits_sm100_tma.hpp
//   (make_tma_atom_A_sm100, make_tma_atom_B_sm100, make_tma_copy_C_sm100),
//     Copyright (c) 2021 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// That portion is under the BSD-3-Clause licence, and the rest of the file is cute-rs's
// own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The MMA-aware TMA atoms of SM100 (CuTe DSL's
//! `cute.nvgpu.make_tiled_tma_atom_A/B/C`): the CTA tile of the atom is the part of
//! the MMA tile (`mma_tiler_mnk`) one CTA's tcgen05 MMA reads or writes, laid out as the
//! MMA partitions it, `(MMA, MMA_M, MMA_K)` for A, so that one stage of the shared-memory
//! layout from [`make_smem_layout_a`](crate::utils::sm100::make_smem_layout_a) (or `_b`,
//! `_c`) and the MMA-partitioned global tile ([`ThrMma::partition_a`](crate::ThrMma))
//! share their coordinates, as [`tma_partition`](crate::cpasync::tma_partition) needs.

use std::panic::Location;

use crate::atom::copy::CopyOp;
use crate::atom::mma::{Operand, TiledMma};
use crate::builder::user_panic;
use crate::layout::{Layout, Shape, Tiler, Tree};
use crate::nvgpu::tma_atom::{TmaAtomOptions, TmaInfo, TmaTile, make_tma_atom_core, tiler_v_map};
use crate::tensor::{SmemLayout, Tensor};
use crate::value::Numeric;

/// The CTA value map of operand `which`: the `(i, j)` tile of `gmem` (its identity
/// layout composed with the tile), partitioned by the MMA (`thrfrg`), its value mode
/// flattened to `(FrgV, Rest_i, Rest_j)`.
fn mma_v_map(tiled_mma: &TiledMma, glayout: &Layout, tile: (usize, usize), which: Operand) -> Result<Layout, String> {
    let tiler: Tiler = tile.into();
    if glayout.rank() < 2 {
        return Err(format!("the global tensor {glayout} has rank {}, not at least 2", glayout.rank()));
    }
    Ok(mma_v_map_of(tiled_mma, &tiler_v_map(glayout, &tiler)?, which))
}

/// The value mode of `tiled_mma`'s partition of the tile `g_tile` (a layout of global
/// coordinates), flattened to `(FrgV, Rest_i, Rest_j)`: C++ CuTe's
/// `layout<1>(mma.thrfrg_X(g_tile))(_, repeat<rank(g_tile)>(_))`.
pub(crate) fn mma_v_map_of(tiled_mma: &TiledMma, g_tile: &Layout, which: Operand) -> Layout {
    let frg = tiled_mma.thrfrg(g_tile, which).mode(1);
    let rest = frg.mode(1);
    Layout::from_modes([frg.mode(0), rest.mode(0), rest.mode(1)])
}

/// The static size of mode `i` of a cluster shape `(V, M, N, K)`.
fn cluster_mode(shape: &Shape, i: usize) -> Result<usize, String> {
    let modes = match shape {
        Tree::Branch(m) if m.len() == 4 => m,
        _ => return Err(format!("cluster_shape_vmnk {shape} has not four modes (V, M, N, K)")),
    };
    modes[i]
        .flat_leaves()
        .iter()
        .map(|e| e.as_static())
        .product::<Option<usize>>()
        .ok_or_else(|| format!("cluster_shape_vmnk {shape} has a runtime extent; the multicast count is static"))
}

#[allow(clippy::too_many_arguments)]
fn make_ab<T: Numeric>(
    loc: &'static Location<'static>,
    what: &str,
    which: Operand,
    op: CopyOp,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    cluster_shape_vmnk: Option<Shape>,
    options: &TmaAtomOptions,
) -> TmaInfo<T> {
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: {what}({op:?}): {m}")) };
    if !tiled_mma.atom().is_tcgen05() {
        fail(format!(
            "the tiled MMA's atom {:?} is not a tcgen05 MMA: the atom's CTA tile is the value mode of a tcgen05 \
             partition (one CTA, or a CTA pair, per instruction)",
            tiled_mma.atom().op()
        ));
    }
    let CopyOp::TmaLoad { multicast, .. } = op else {
        fail(
            "the operation is not a TMA load (cpasync::CopyBulkTensorTileG2SOp or CopyBulkTensorTileG2SMulticastOp)"
                .to_string(),
        )
    };
    let (m, n, k) = mma_tiler_mnk;
    let (tile, cluster_mode_idx) = if which == Operand::A { ((m, k), 2) } else { ((n, k), 1) };
    let num_multicast = match (&cluster_shape_vmnk, multicast) {
        (Some(c), true) => cluster_mode(c, cluster_mode_idx).unwrap_or_else(|e| fail(e)),
        _ => 1,
    };
    let map = mma_v_map(tiled_mma, gmem_tensor.layout(), tile, which).unwrap_or_else(|e| fail(e));
    let tile = TmaTile::VMap { map, rank: 3 };
    make_tma_atom_core(loc, what, op, gmem_tensor, smem_layout.into_composed(), tile, num_multicast, options)
}

/// A TMA load atom for the A operand of `tiled_mma` (CuTe DSL's
/// `cute.nvgpu.make_tiled_tma_atom_A(op, gmem_tensor, smem_layout, mma_tiler_mnk,
/// tiled_mma, cluster_shape_vmnk=None)`), with its TMA tensor. See
/// [`make_tiled_tma_atom_a_with`].
#[track_caller]
pub fn make_tiled_tma_atom_a<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    cluster_shape_vmnk: Option<Shape>,
) -> TmaInfo<T> {
    let o = TmaAtomOptions::new();
    let loc = Location::caller();
    let what = "make_tiled_tma_atom_A";
    let op = op.into();
    make_ab(loc, what, Operand::A, op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, cluster_shape_vmnk, &o)
}

/// [`make_tiled_tma_atom_a`] with CuTe DSL's keyword argument `internal_type`
/// ([`TmaAtomOptions::internal_type`]).
///
/// - `op` is `CopyBulkTensorTileG2SOp` or `CopyBulkTensorTileG2SMulticastOp`.
/// - `gmem_tensor` is the `(M, K, ...)` global tensor, on a kernel buffer parameter.
/// - `smem_layout` is one stage of A's shared-memory layout, `(MMA, MMA_M, MMA_K)`, or
///   the staged layout with a fourth (stage) mode, as
///   [`make_smem_layout_a`](crate::utils::sm100::make_smem_layout_a) gives it.
/// - The atom's CTA tile is the `(TILE_M, TILE_K)` tile of `mma_tiler_mnk` as
///   `tiled_mma` (a tcgen05 MMA) partitions it: the value mode of its A partition,
///   `(MMA, MMA_M, MMA_K)`.
/// - With the multicast operation, the box is split across the cluster's N mode
///   (`cluster_shape_vmnk`'s mode 2): the CTAs that share an A tile. Otherwise, or
///   without a cluster shape, one CTA loads the whole box.
#[track_caller]
pub fn make_tiled_tma_atom_a_with<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    cluster_shape_vmnk: Option<Shape>,
    options: &TmaAtomOptions,
) -> TmaInfo<T> {
    let loc = Location::caller();
    let what = "make_tiled_tma_atom_A";
    let op = op.into();
    make_ab(loc, what, Operand::A, op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, cluster_shape_vmnk, options)
}

/// A TMA load atom for the B operand of `tiled_mma` (CuTe DSL's
/// `cute.nvgpu.make_tiled_tma_atom_B(...)`): [`make_tiled_tma_atom_a`] for the
/// `(N, K, ...)` tensor and the `(TILE_N, TILE_K)` tile, the multicast across the
/// cluster's M mode (`cluster_shape_vmnk`'s mode 1).
#[track_caller]
pub fn make_tiled_tma_atom_b<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    cluster_shape_vmnk: Option<Shape>,
) -> TmaInfo<T> {
    let o = TmaAtomOptions::new();
    let loc = Location::caller();
    let what = "make_tiled_tma_atom_B";
    let op = op.into();
    make_ab(loc, what, Operand::B, op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, cluster_shape_vmnk, &o)
}

/// [`make_tiled_tma_atom_b`] with `internal_type` ([`TmaAtomOptions`]).
#[track_caller]
pub fn make_tiled_tma_atom_b_with<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    cluster_shape_vmnk: Option<Shape>,
    options: &TmaAtomOptions,
) -> TmaInfo<T> {
    let loc = Location::caller();
    let what = "make_tiled_tma_atom_B";
    let op = op.into();
    make_ab(loc, what, Operand::B, op, gmem_tensor, smem_layout, mma_tiler_mnk, tiled_mma, cluster_shape_vmnk, options)
}

/// A TMA store (or reduce) atom for the C operand of `tiled_mma` (CuTe DSL's
/// `cute.nvgpu.make_tiled_tma_atom_C(op, gmem_tensor, smem_layout, mma_tiler_mnk,
/// tiled_mma)`): `op` is `CopyBulkTensorTileS2GOp` or `CopyReduceBulkTensorTileS2GOp`,
/// `gmem_tensor` the `(M, N, ...)` tensor, `smem_layout` one stage of C's layout
/// `(MMA, MMA_M, MMA_N)` (or staged), and the CTA tile the `(TILE_M, TILE_N)` tile as
/// the MMA partitions C. Stores do not multicast.
#[track_caller]
pub fn make_tiled_tma_atom_c<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
) -> TmaInfo<T> {
    let loc = Location::caller();
    let op = op.into();
    let what = "make_tiled_tma_atom_C";
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: {what}({op:?}): {m}")) };
    if !tiled_mma.atom().is_tcgen05() {
        fail(format!("the tiled MMA's atom {:?} is not a tcgen05 MMA", tiled_mma.atom().op()));
    }
    if !matches!(op, CopyOp::TmaStore | CopyOp::TmaReduce(_)) {
        fail(
            "the operation is not a TMA store (cpasync::CopyBulkTensorTileS2GOp or CopyReduceBulkTensorTileS2GOp)"
                .to_string(),
        );
    }
    let (m, n, _) = mma_tiler_mnk;
    let map = mma_v_map(tiled_mma, gmem_tensor.layout(), (m, n), Operand::C).unwrap_or_else(|e| fail(e));
    let tile = TmaTile::VMap { map, rank: 3 };
    make_tma_atom_core(loc, what, op, gmem_tensor, smem_layout.into_composed(), tile, 1, &TmaAtomOptions::new())
}

/// An im2col TMA load atom for the A operand of `tiled_mma` (CuTe DSL's
/// `cute.nvgpu.make_im2col_tma_atom_A(op, gmem_tensor, smem_layout, mma_tiler_mnk,
/// tiled_mma, filter_trs, upper_padding_dhw, lower_padding_dhw, stride_dhw, dilation_dhw,
/// cluster_shape_vmnk=None)`): [`make_im2col_tma_atom`](crate::cpasync::make_im2col_tma_atom)
/// whose CTA tile is the `(TILE_M, TILE_K)` tile of the GEMM's `(pixels, channels)` as the
/// tcgen05 MMA partitions it, `smem_layout` one stage (or the staged layout) of A's
/// `(MMA, MMA_M, MMA_K)` SMEM layout.
///
/// - `op` is `CopyBulkTensorIm2ColG2SOp` or `CopyBulkTensorIm2ColG2SMulticastOp`.
/// - `conv` has the convolution's parameters:
///   [`Im2colOptions::conv`](crate::cpasync::Im2colOptions::conv)`(filter_srt,
///   lower_padding_whd, upper_padding_whd, stride_whd, dilation_whd)`, CuTe DSL's
///   `filter_trs` and `*_dhw` tuples reversed (W first), and the `internal_type`. Its
///   `num_multicast` is replaced: with the multicast operation the cluster's N mode
///   (`cluster_shape_vmnk`'s mode 2), otherwise 1.
#[track_caller]
pub fn make_im2col_tma_atom_a<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    mma_tiler_mnk: (usize, usize, usize),
    tiled_mma: &TiledMma,
    conv: &crate::cpasync::Im2colOptions,
    cluster_shape_vmnk: Option<Shape>,
) -> TmaInfo<T> {
    use crate::cpasync::TmaOpMode;
    use crate::nvgpu::tma_im2col::{Im2colTile, make_im2col_core};
    let loc = Location::caller();
    let op = op.into();
    let what = "make_im2col_tma_atom_A";
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: {what}({op:?}): {m}")) };
    if !tiled_mma.atom().is_tcgen05() {
        fail(format!("the tiled MMA's atom {:?} is not a tcgen05 MMA", tiled_mma.atom().op()));
    }
    let CopyOp::TmaLoadMode { mode: TmaOpMode::Im2col, multicast, .. } = op else {
        fail(
            "the operation is not an im2col TMA load (cpasync::CopyBulkTensorIm2ColG2SOp or \
             CopyBulkTensorIm2ColG2SMulticastOp)"
                .to_string(),
        )
    };
    let num_multicast = match (&cluster_shape_vmnk, multicast) {
        (Some(c), true) => cluster_mode(c, 2).unwrap_or_else(|e| fail(e)),
        _ => 1,
    };
    let mut options = conv.clone();
    options.num_multicast = num_multicast;
    let (m, _, k) = mma_tiler_mnk;
    let tile = Im2colTile::MmaA { tiled_mma, tile_mk: (m, k) };
    make_im2col_core(loc, what, op, gmem_tensor, smem_layout.into_composed(), tile, &options)
}

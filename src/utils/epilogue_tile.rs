// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/epilogue/collective/builders/sm100_builder.inl
//   (the by-mode tiler of sm100_dense_compute_tile_shape_or_override),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! CuTe DSL's `cutlass.utils.compute_epilogue_tile_shape`.

use super::LayoutEnum;
use super::sm100::{compute_epilogue_tile_size, default_tmem_warp_shape_mn, epilogue_n_minimums};
use crate::ir::DType;
use crate::layout::{Layout, Tiler, Tree};

/// The epilogue tiler of an SM100 GEMM (CuTe DSL's
/// `cutlass.utils.compute_epilogue_tile_shape(cta_tile_shape,
/// use_2cta_instrs, layout_d, elem_ty_d, layout_c=, elem_ty_c=,
/// tmem_warp_shape_mn=)`): the subtile of
/// [`compute_epilogue_tile_size`](super::sm100::compute_epilogue_tile_size)
/// as a by-mode tiler `(tile_m, coalesce((tile_n / warp_n, warp_n):(1,
/// cta_n / warp_n)))`, so that each of the `warp_n` warps along N takes its
/// own slice of the CTA tile's N. `cta_tile_shape` is `(M, N)` or `(M, N,
/// K)`; `layout_c`/`elem_ty_c` describe the source C (`None`: none), and C
/// counts as M-major when its layout is not given.
///
/// Panics with "CTA tile too small" when `cta_tile_shape`'s N is below
/// the TMA store's minimum N for D or C.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn compute_epilogue_tile_shape(
    cta_tile_shape: &[usize],
    use_2cta_instrs: bool,
    layout_d: LayoutEnum,
    elem_ty_d: DType,
    layout_c: Option<LayoutEnum>,
    elem_ty_c: Option<DType>,
    tmem_warp_shape_mn: Option<(usize, usize)>,
) -> Tiler {
    assert!(cta_tile_shape.len() >= 2, "compute_epilogue_tile_shape: the CTA tile shape {cta_tile_shape:?} has no N");
    let (cta_m, cta_n) = (cta_tile_shape[0], cta_tile_shape[1]);
    let d_is_m_major = layout_d.is_m_major_c();
    let c_is_m_major = layout_c.is_none_or(|l| l.is_m_major_c());
    let (d_width, c_width) = (elem_ty_d.bits(), elem_ty_c.map(|t| t.bits()));
    let (tile_m, tile_n) = compute_epilogue_tile_size(
        cta_m,
        cta_n,
        use_2cta_instrs,
        d_width,
        c_width,
        d_is_m_major,
        c_is_m_major,
        tmem_warp_shape_mn,
    );
    let (_, warp_n) = tmem_warp_shape_mn.unwrap_or_else(|| default_tmem_warp_shape_mn(cta_m, use_2cta_instrs));
    let (n_min_d, n_min_c) = epilogue_n_minimums(warp_n, d_width, c_width, d_is_m_major, c_is_m_major);
    assert!(
        cta_n >= n_min_c && cta_n >= n_min_d,
        "compute_epilogue_tile_shape: CTA tile too small: {cta_tile_shape:?} (N must be at least {} for D and {} for \
         C)",
        n_min_d,
        n_min_c
    );
    let n = Layout::new((tile_n / warp_n, warp_n), (1, cta_n / warp_n)).coalesce();
    Tree::Branch(vec![Tree::Leaf(Some(Layout::new(tile_m, 1))), Tree::Leaf(Some(n))])
}

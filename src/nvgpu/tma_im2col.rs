// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/copy_traits_sm90_im2col.hpp (make_tma_atom_im2col and
//   make_im2col_tma_copy_desc: the channels and pixels of the box from the SMEM layout's
//   contiguous runs, the descriptor's extents, strides and corners, and the GEMM extents
//   q/p/z = (w/h/d + upper corner - lower corner - 1) / stride + 1 and
//   s/r/t = 1 + (upper padding - upper corner) / dilation),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Im2col TMA atoms: [`make_im2col_tma_atom`] and what
//! their copies issue.
//!
//! An implicit-GEMM convolution reads its activation tensor (`NWC`, `NHWC` or `NDHWC`,
//! given as `((w, (h, (d,)) n), c)`) as the GEMM operand `A` of shape `(M, K)`, `M` the
//! output pixels `(q, p, z, n)` and `K` the filter taps `(c, s, r, t)`. TMA's im2col
//! mode does the rearrangement: one instruction loads a column of pixels (`M`) of some
//! channels (`K`), walking a bounding box of filter positions and adding the filter
//! tap's offsets. The atom's TMA tensor has the GEMM's shape `(M, (C, S[, R[, T]]))` with
//! coordinate strides, `M` linear as in C++ CuTe (whose TMA tensor composes its
//! multi-mode `(q, p, z, n)` with a linear layout): a copy turns each box's GEMM
//! coordinate into the instruction's operands, `(c, w, h, d, n)` with
//! `w = lower_corner_w + q * stride_w` and the offsets `lower_srt + s * stride_srt`.
//! Stores (`CopyBulkTensorIm2ColS2GOp`, `.im2col_no_offs`) take the output tensor
//! `((q, p, z, n), k)` with no corners or offsets.

use std::panic::Location;

use crate::atom::copy::CopyOp;
use crate::builder::{self, user_panic};
use crate::cpasync::{CtaGroup, TmaInfo, TmaOpMode};
use crate::ir::{DType, Param, ValueId};
use crate::layout::{IndexExpr, Layout, ScaledBasis, StrideLeaf, Tiler, Tree, make_identity_layout};
use crate::nvgpu::tma_atom::{TmaAtomInfo, TmaAtomMode, tma_swizzle};
use crate::tensor::{Engine, SmemLayout, Tensor, TensorInner, make_coord_tensor};
use crate::tma::{Im2colParams, TmaDescSpec};
use crate::value::{IntoValue, Numeric};

/// The im2col keyword arguments of CuTe DSL's `cpasync.make_im2col_tma_atom`, one entry
/// per spatial dimension, W first (`whd`) or S first (`srt`): the bounding box's corners
/// relative to the tensor (`lower_corner_whd`, `upper_corner_whd`), the convolution's
/// padding (`lower_padding_whd` before the data, `upper_padding_whd` after), the traversal
/// (convolution) stride `stride_whd`, and the filter offsets' start and step
/// (`lower_srt`, `stride_srt`, the dilation). Loads need them all; stores take none.
/// [`Im2colOptions::conv`] derives them from a convolution's filter, padding, stride and
/// dilation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Im2colOptions {
    pub lower_corner_whd: Option<Vec<i32>>,
    pub upper_corner_whd: Option<Vec<i32>>,
    pub lower_padding_whd: Option<Vec<i32>>,
    pub upper_padding_whd: Option<Vec<i32>>,
    pub stride_whd: Option<Vec<i32>>,
    pub lower_srt: Option<Vec<i32>>,
    pub stride_srt: Option<Vec<i32>>,
    /// The CTAs that share each box (1 without multicast).
    pub num_multicast: usize,
    /// `internal_type=`: the type the descriptor moves the elements as.
    pub internal_type: Option<DType>,
}

impl Im2colOptions {
    /// No im2col parameters and no multicast: a store's options.
    pub fn new() -> Im2colOptions {
        Im2colOptions { num_multicast: 1, ..Im2colOptions::default() }
    }

    /// The parameters of a forward convolution with filter extents `filter_srt`
    /// (`S, R, T`, W's first), padding `lower_padding_whd` before and
    /// `upper_padding_whd` after the data, convolution stride `stride_whd` and dilation
    /// `dilation_whd`: the bounding box spans the filter's first tap over the output,
    /// from `-lower_padding` to `extent - 1 + upper_padding - (filter - 1) * dilation`,
    /// and the filter offsets start at 0 with the dilation as their step (as CuTe DSL's
    /// convolution examples set them).
    pub fn conv(
        filter_srt: &[i32],
        lower_padding_whd: &[i32],
        upper_padding_whd: &[i32],
        stride_whd: &[i32],
        dilation_whd: &[i32],
    ) -> Im2colOptions {
        let n = filter_srt.len();
        let upper: Vec<i32> = (0..n).map(|i| upper_padding_whd[i] - (filter_srt[i] - 1) * dilation_whd[i]).collect();
        Im2colOptions {
            lower_corner_whd: Some(lower_padding_whd.iter().map(|p| -p).collect()),
            upper_corner_whd: Some(upper),
            lower_padding_whd: Some(lower_padding_whd.to_vec()),
            upper_padding_whd: Some(upper_padding_whd.to_vec()),
            stride_whd: Some(stride_whd.to_vec()),
            lower_srt: Some(vec![0; n]),
            stride_srt: Some(dilation_whd.to_vec()),
            num_multicast: 1,
            internal_type: None,
        }
    }

    pub fn num_multicast(mut self, n: usize) -> Self {
        self.num_multicast = n;
        self
    }

    pub fn internal_type(mut self, t: DType) -> Self {
        self.internal_type = Some(t);
        self
    }
}

/// A run's global mode and its static step, from the SMEM index -> global mode map.
type BasisRun = (Vec<usize>, Option<usize>);

/// What an im2col atom's copies need to turn a GEMM coordinate into the instruction's
/// operands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Im2colGeometry {
    /// Spatial dimensions (1 to 3).
    pub spatial: usize,
    /// The output extents `q, p, z` (the bounding box's filter positions per dimension).
    pub gemm_mn: Vec<IndexExpr>,
    pub lower_corner: Vec<i32>,
    pub stride_whd: Vec<i32>,
    pub lower_srt: Vec<i32>,
    pub stride_srt: Vec<i32>,
    /// `.im2col` (loads: offsets) or `.im2col_no_offs` (stores).
    pub offsets: bool,
}

impl Im2colGeometry {
    /// The instruction's coordinates `(c, w, (h, (d,)) n)` and offsets for the GEMM
    /// coordinate `leaves` (`m, c, s, (r, (t,))`), as i32 and i16 values.
    #[track_caller]
    pub(crate) fn operands(&self, leaves: &[IndexExpr]) -> (Vec<ValueId>, Vec<ValueId>) {
        let n = self.spatial;
        // Trailing taps of extent 1 (a store's, or a 1-wide filter) may be absent.
        assert!(
            (2..=2 + n).contains(&leaves.len()),
            "an im2col atom's TMA tensor gives (m, (c, s, r, t)) coordinates, not {leaves:?}: the coordinate tensor is \
             not the atom's TMA tensor"
        );
        let mut leaves = leaves.to_vec();
        leaves.resize(2 + n, IndexExpr::Static(0));
        let mut rest = leaves[0].clone();
        let mut coords = vec![leaves[1].clone()];
        for i in 0..n {
            let q = IndexExpr::remainder(rest.clone(), self.gemm_mn[i].clone());
            rest = IndexExpr::quotient(rest, self.gemm_mn[i].clone());
            let step = IndexExpr::product(q, IndexExpr::from_signed(i64::from(self.stride_whd[i])));
            coords.push(IndexExpr::sum(IndexExpr::from_signed(i64::from(self.lower_corner[i])), step));
        }
        coords.push(rest);
        let coords = coords.into_iter().map(|c| IntoValue::<i32>::into_value(c).checked()).collect();
        let offsets = if self.offsets {
            (0..n)
                .map(|i| {
                    let o = IndexExpr::sum(
                        IndexExpr::from_signed(i64::from(self.lower_srt[i])),
                        IndexExpr::product(
                            leaves[2 + i].clone(),
                            IndexExpr::from_signed(i64::from(self.stride_srt[i])),
                        ),
                    );
                    IntoValue::<i32>::into_value(o).to::<i16>().checked()
                })
                .collect()
        } else {
            Vec::new()
        };
        (coords, offsets)
    }
}

/// An im2col TMA atom (CuTe DSL's `cpasync.make_im2col_tma_atom(op, gmem_tensor,
/// smem_layout, cta_tiler, lower_corner_whd, upper_corner_whd, lower_padding_whd,
/// upper_padding_whd, stride_whd, lower_srt, stride_srt, num_multicast,
/// internal_type=)`), with its TMA tensor.
///
/// - `op` is [`CopyBulkTensorIm2ColG2SOp`](crate::cpasync::CopyBulkTensorIm2ColG2SOp)
///   (`num_multicast` 1), its multicast form, or
///   [`CopyBulkTensorIm2ColS2GOp`](crate::cpasync::CopyBulkTensorIm2ColS2GOp) (1).
/// - `gmem_tensor` is a global tensor on a kernel buffer parameter laid out
///   `((w, (h, (d,)) n), c)`: one to three spatial dimensions and the images in mode 0,
///   the channels, contiguous, in mode 1 (a loaded activation tensor, or a stored
///   output `((q, p, z, n), k)`). Its extents and strides may be scalar parameters.
/// - `smem_layout` is one tile (or staged tiles) of `(pixels, channels)`, channels
///   contiguous (K-major), plain or with a TMA swizzle; `cta_tiler` its tile of the
///   tensor's `(pixels, channels)`.
/// - `options` has the im2col parameters (all of them for a load), the multicast and
///   the internal type.
///
/// One instruction moves a box of the SMEM layout's contiguous run of channels (the
/// descriptor's channels per pixel, at most 256) for its next run of pixels (pixels per
/// column, at most 1024; split among the CTAs of a multicast). The TMA tensor has the
/// GEMM's shape `(M, (C, S[, R[, T]]))`, `M` the bounding box's filter positions
/// (`q * p * z * n`), `S, R, T` the filter's taps; tile it with the same tiler, partition
/// it with [`tma_partition`](crate::cpasync::tma_partition) and copy as for a tiled
/// atom. A load's box past the last image is filled with zeros, as is the padding.
#[track_caller]
pub fn make_im2col_tma_atom<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    cta_tiler: impl Into<Tiler>,
    options: &Im2colOptions,
) -> TmaInfo<T> {
    let loc = Location::caller();
    let tile = Im2colTile::Tiler(cta_tiler.into());
    make_im2col_core(loc, "make_im2col_tma_atom", op.into(), gmem_tensor, smem_layout.into_composed(), tile, options)
}

/// How an im2col atom's CTA tile is given: by a tiler of the tensor's `(pixels,
/// channels)`, or as a tcgen05 MMA partitions its `(TILE_M, TILE_K)` tile of them (CuTe
/// DSL's `make_im2col_tma_atom_A`).
pub(crate) enum Im2colTile<'a> {
    Tiler(Tiler),
    MmaA { tiled_mma: &'a crate::atom::mma::TiledMma, tile_mk: (usize, usize) },
}

/// The body of [`make_im2col_tma_atom`], for a tile given either way.
pub(crate) fn make_im2col_core<T: Numeric>(
    loc: &'static Location<'static>,
    what: &str,
    op: CopyOp,
    gmem_tensor: &Tensor<T>,
    staged: crate::layout::ComposedLayout,
    tile: Im2colTile<'_>,
    options: &Im2colOptions,
) -> TmaInfo<T> {
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: {what}({op:?}): {m}")) };
    let num_multicast = options.num_multicast;
    let load = match op {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Im2col, multicast, cta_group } => {
            if cta_group == CtaGroup::TWO
                && let Err(why) = crate::nvgpu::tma_atom::check_pair_load_target()
            {
                fail(why);
            }
            if !multicast && num_multicast != 1 {
                fail(format!(
                    "CopyBulkTensorIm2ColG2SOp loads for one CTA, so num_multicast is 1, not {num_multicast} (use \
                     CopyBulkTensorIm2ColG2SMulticastOp)"
                ));
            }
            if num_multicast == 0 {
                fail("num_multicast is 0; it counts the CTAs sharing a box, at least 1".to_string());
            }
            true
        }
        CopyOp::TmaStoreMode { mode: TmaOpMode::Im2col } => {
            if num_multicast != 1 {
                fail(format!("TMA stores do not multicast, so num_multicast is 1, not {num_multicast}"));
            }
            false
        }
        _ => fail(
            "the operation is not an im2col TMA operation (cpasync::CopyBulkTensorIm2Col*Op); tiled ones take \
             make_tiled_tma_atom"
                .to_string(),
        ),
    };

    // The global tensor: a buffer parameter, ((w, h, d, n), c).
    let Engine::Memory { buffer, space, .. } = gmem_tensor.engine else {
        fail("the global tensor is not in memory".to_string())
    };
    let (is_param, decl) = builder::with(|b| {
        (b.params().iter().any(|p| matches!(p, Param::Buffer(x) if *x == buffer)), b.buffer(buffer).clone())
    });
    if space != crate::ir::MemSpace::Global || !is_param {
        fail(format!(
            "`{}` is not a global tensor on a kernel buffer parameter (param_tensor, or make_tensor on param_ptr)",
            decl.name
        ));
    }
    if !matches!(gmem_tensor.inner, TensorInner::Identity) {
        fail(format!("`{}` is swizzled or composed: TMA describes a plain strided tensor", decl.name));
    }
    let glayout = gmem_tensor.layout();
    let shape_txt = glayout.to_string();
    if glayout.rank() != 2 || glayout.mode(1).shape().flat_leaves().len() != 1 {
        fail(format!(
            "`{}` is laid out {shape_txt}; an im2col tensor is ((w, h, d, n), c): the spatial dimensions and the \
             images in mode 0, the channels in mode 1",
            decl.name
        ));
    }
    let pix_shape = glayout.mode(0).shape().flat_leaves();
    let pix_stride = glayout.mode(0).stride().flat_leaves();
    let spatial = pix_shape.len().wrapping_sub(1);
    if !(1..=3).contains(&spatial) {
        fail(format!(
            "`{}` is laid out {shape_txt}, whose mode 0 has {} modes; it is (w, n), (w, h, n) or (w, h, d, n): one to \
             three spatial dimensions and the images",
            decl.name,
            pix_shape.len()
        ));
    }
    let int = |s: &StrideLeaf| match s {
        StrideLeaf::Int(e) => e.clone(),
        StrideLeaf::Basis(_) => fail(format!("`{}` has a basis stride; TMA needs integer strides", decl.name)),
    };
    let c_extent = glayout.mode(1).shape().flat_leaves()[0].clone();
    let c_stride = int(&glayout.mode(1).stride().flat_leaves()[0]);
    if c_stride.as_static() != Some(1) {
        fail(format!(
            "`{}`'s channels (mode 1) have stride {c_stride:?}, not 1: TMA's dimension 0 (C) is contiguous",
            decl.name
        ));
    }

    // The im2col parameters, one per spatial dimension.
    let get = |v: &Option<Vec<i32>>, name: &str, default: i32| -> Vec<i32> {
        match v {
            Some(v) if load => {
                if v.len() != spatial {
                    fail(format!("{name} has {} entries; the tensor has {spatial} spatial dimensions", v.len()));
                }
                v.clone()
            }
            None if load => fail(format!(
                "an im2col load needs {name} (and lower_corner_whd, upper_corner_whd, lower_padding_whd, \
                 upper_padding_whd, stride_whd, lower_srt, stride_srt: Im2colOptions::conv derives them)"
            )),
            _ => vec![default; spatial],
        }
    };
    let lower_corner = get(&options.lower_corner_whd, "lower_corner_whd", 0);
    let upper_corner = get(&options.upper_corner_whd, "upper_corner_whd", 0);
    let lower_padding = get(&options.lower_padding_whd, "lower_padding_whd", 0);
    let upper_padding = get(&options.upper_padding_whd, "upper_padding_whd", 0);
    let stride_whd = get(&options.stride_whd, "stride_whd", 1);
    let lower_srt = get(&options.lower_srt, "lower_srt", 0);
    let stride_srt = get(&options.stride_srt, "stride_srt", 1);
    if let Some(s) = stride_whd.iter().find(|&&s| !(1..=8).contains(&s)) {
        fail(format!("stride_whd {s} is outside 1 to 8 (the descriptor's traversal strides)"));
    }
    if stride_srt.contains(&0) {
        fail("stride_srt (the dilation) has a 0".to_string());
    }
    // The filter taps per dimension (C++ CuTe: 1 + (upper padding - upper corner) / dilation,
    // or from the lower side for a negative dilation).
    let taps: Vec<i32> = (0..spatial)
        .map(|i| {
            let pad =
                if stride_srt[i] > 0 { upper_padding[i] - upper_corner[i] } else { lower_corner[i] - lower_padding[i] };
            1 + pad / stride_srt[i]
        })
        .collect();
    if let Some(i) = (0..spatial).find(|&i| taps[i] < 1) {
        fail(format!(
            "spatial dimension {i}: upper_padding_whd - upper_corner_whd is {} with stride_srt {}, which leaves no \
             filter tap",
            upper_padding[i] - upper_corner[i],
            stride_srt[i]
        ));
    }
    // The instruction's offsets are unsigned and of the rank's width: check the range the
    // filter's taps reach.
    let rank = spatial + 2;
    let max_off = i64::from(crate::tma::im2col_offset_max(rank));
    if load {
        for i in 0..spatial {
            let (a, b) =
                (i64::from(lower_srt[i]), i64::from(lower_srt[i]) + i64::from(taps[i] - 1) * i64::from(stride_srt[i]));
            if a.min(b) < 0 || a.max(b) > max_off {
                fail(format!(
                    "spatial dimension {i}'s filter offsets run from {a} to {b} (lower_srt + tap * stride_srt), \
                     outside the 0 to {max_off} a rank-{rank} im2col instruction's offsets take"
                ));
            }
        }
    }

    // One stage of the SMEM layout.
    let tiler_rank = match &tile {
        Im2colTile::Tiler(tiler) => {
            if tiler.is_leaf() {
                1
            } else {
                tiler.rank()
            }
        }
        Im2colTile::MmaA { .. } => 3,
    };
    let smem = if staged.rank() == tiler_rank + 1 {
        staged.select(&(0..tiler_rank).collect::<Vec<_>>())
    } else if staged.rank() == tiler_rank {
        staged.clone()
    } else {
        fail(format!(
            "the SMEM layout {staged} has rank {}, but the CTA tiler has rank {tiler_rank}: the layout is one tile or \
             staged tiles",
            staged.rank()
        ))
    };
    if tiler_rank != 2 && matches!(tile, Im2colTile::Tiler(_)) {
        fail(format!("the CTA tiler has rank {tiler_rank}; an im2col tile is (pixels, channels)"));
    }
    let Some(sw) = smem.swizzle() else {
        fail(format!("the SMEM layout {smem} is composed with a layout, not a swizzle, which TMA cannot write"))
    };
    let swizzle = tma_swizzle(sw, T::DTYPE.bits()).unwrap_or_else(|e| fail(e));
    let slayout = smem.outer.clone();
    if !slayout.is_static() || slayout.has_scaled_basis() {
        fail(format!("the SMEM layout {slayout} is not static with integer strides"));
    }

    // C++ CuTe's make_tma_atom_im2col: the SMEM index -> (pixel, channel) map over the
    // tile of the tensor's (pixels, channels); its first run walks the channels, its
    // second the pixels.
    let m_extent = pix_shape.iter().cloned().reduce(IndexExpr::product).expect("one or more modes");
    let identity = make_identity_layout(Tree::Branch(vec![Tree::Leaf(m_extent), Tree::Leaf(c_extent.clone())]));
    let cta_v_map = match &tile {
        Im2colTile::Tiler(tiler) => identity.compose_tiler(tiler),
        Im2colTile::MmaA { tiled_mma, tile_mk } => crate::nvgpu::tma_sm100_mma::mma_v_map_of(
            tiled_mma,
            &identity.compose_tiler(&(*tile_mk).into()),
            crate::atom::mma::Operand::A,
        ),
    };
    let v_size: Option<usize> = cta_v_map.shape().flat_leaves().iter().map(|e| e.as_static()).product();
    if v_size != Some(slayout.size()) {
        fail(format!(
            "the SMEM layout {slayout} has {} elements, but the CTA tile {} has {v_size:?}",
            slayout.size(),
            cta_v_map.shape()
        ));
    }
    let sidx2gmode = cta_v_map.compose(&slayout.right_inverse()).coalesce();
    let runs: Vec<(usize, Option<BasisRun>)> = sidx2gmode
        .static_shape()
        .flat_leaves()
        .into_iter()
        .zip(sidx2gmode.stride().flat_leaves())
        .map(|(n, s)| match s {
            StrideLeaf::Basis(b) => (n, Some((b.mode().to_vec(), b.value().as_int().and_then(|v| v.as_static())))),
            StrideLeaf::Int(_) => (n, None),
        })
        .collect();
    let unit = |k: usize, mode: usize| {
        runs.get(k).is_some_and(|(_, r)| matches!(r, Some((m, Some(1))) if m.as_slice() == [mode]))
    };
    if !unit(0, 1) {
        fail(format!(
            "the SMEM layout {slayout}'s contiguous run does not walk the channels (its SMEM index -> (pixel, channel) \
             map is {sidx2gmode}): im2col copies each pixel's channels contiguously, so the layout is K-major"
        ));
    }
    let channels = runs[0].0;
    let pixels = if unit(1, 0) { runs[1].0 } else { 1 };
    if channels > 256 {
        fail(format!(
            "the SMEM layout's contiguous run is {channels} channels, more than an im2col box's 256: use a swizzle \
             atom or a narrower tile"
        ));
    }
    if pixels > 1024 * num_multicast {
        fail(format!("the SMEM layout's run of {pixels} pixels is more than an im2col box's 1024 per CTA"));
    }
    if !pixels.is_multiple_of(num_multicast) {
        fail(format!("the box's {pixels} pixels cannot be split among {num_multicast} CTAs"));
    }
    let ppc = pixels / num_multicast;
    let elem_bytes = T::DTYPE.bytes();
    let share = channels * ppc * elem_bytes;
    if num_multicast > 1 && !share.is_multiple_of(128) {
        fail(format!(
            "each of the {num_multicast} CTAs' share of the box is {share} bytes, so some CTA's would start at an \
             address that is not 128-byte aligned, as TMA's shared-memory destination must be"
        ));
    }

    // The descriptor: dimensions (C, W, H, D, N).
    let data_type =
        crate::nvgpu::tma_atom::resolve_internal_type(T::DTYPE, options.internal_type).unwrap_or_else(|e| fail(e));
    let elem = IndexExpr::Static(elem_bytes);
    let mut global_dims = vec![c_extent];
    global_dims.extend(pix_shape.iter().cloned());
    let global_strides: Vec<IndexExpr> = pix_stride.iter().map(|s| IndexExpr::product(int(s), elem.clone())).collect();
    let pad3 = |v: &[i32]| {
        let mut a = [0; 3];
        a[..v.len()].copy_from_slice(v);
        a
    };
    let mut element_strides = vec![1u32];
    element_strides.extend(stride_whd.iter().map(|&s| s as u32));
    element_strides.push(1);
    let spec = TmaDescSpec {
        data_type,
        global_dims: global_dims.clone(),
        global_strides,
        box_dims: vec![0; rank],
        element_strides,
        interleave: crate::tma::TensorMapInterleave::none,
        swizzle,
        l2_promotion: crate::tma::TensorMapL2Promotion::l2_128b,
        oob_fill: crate::tma::TensorMapFloatOOBFill::none,
        base_offset: IndexExpr::product(gmem_tensor.offset.clone(), elem),
        im2col: Some(Im2colParams {
            lower_corner: pad3(&lower_corner),
            upper_corner: pad3(&upper_corner),
            channels_per_pixel: channels as u32,
            pixels_per_column: ppc as u32,
        }),
    };
    let descriptor = crate::tma::declare_hidden_descriptor(buffer, &decl, spec).unwrap_or_else(|e| fail(e));

    // The GEMM extents: q/p/z = (w/h/d + upper corner - lower corner - 1) / stride + 1.
    let gemm_mn: Vec<IndexExpr> = (0..spatial)
        .map(|i| {
            let span = IndexExpr::sum(
                pix_shape[i].clone(),
                IndexExpr::from_signed(i64::from(upper_corner[i]) - i64::from(lower_corner[i]) - 1),
            );
            IndexExpr::sum(IndexExpr::quotient(span, IndexExpr::Static(stride_whd[i] as usize)), IndexExpr::Static(1))
        })
        .collect();
    if let Some(i) = gemm_mn.iter().position(|q| q.as_static_signed().is_some_and(|q| q < 1)) {
        fail(format!("spatial dimension {i}'s bounding box holds no filter position"));
    }
    let m = gemm_mn.iter().cloned().chain(std::iter::once(pix_shape[spatial].clone())).reduce(IndexExpr::product);
    let basis = |j: usize| StrideLeaf::Basis(ScaledBasis::new(IndexExpr::Static(1), vec![j]));
    let mut k_shape = vec![Tree::Leaf(global_dims[0].clone())];
    let mut k_stride = vec![Tree::Leaf(basis(1))];
    for (i, &t) in taps.iter().enumerate() {
        k_shape.push(Tree::Leaf(IndexExpr::Static(t as usize)));
        k_stride.push(Tree::Leaf(basis(2 + i)));
    }
    let tma_layout = Layout::new(
        Tree::Branch(vec![Tree::Leaf(m.expect("one or more modes")), Tree::Branch(k_shape)]),
        Tree::Branch(vec![Tree::Leaf(basis(0)), Tree::Branch(k_stride)]),
    );
    let tma_tensor = make_coord_tensor(0, tma_layout);

    let geometry = Im2colGeometry { spatial, gemm_mn, lower_corner, stride_whd, lower_srt, stride_srt, offsets: load };
    let gbasis = Layout::new(
        (channels, pixels),
        Tree::Branch(vec![Tree::Leaf(crate::layout::E(vec![1])), Tree::Leaf(crate::layout::E(vec![0]))]),
    );
    let info = TmaAtomInfo {
        descriptor,
        gbasis,
        num_multicast,
        swizzle,
        smem_layout: smem,
        dim_modes: Vec::new(),
        flat_modes: pix_shape.len() + 1,
        mode: TmaAtomMode::Im2col(geometry),
    };
    let atom = crate::atom::copy::CopyAtom::tma(op, channels * pixels, info);
    TmaInfo { atom, tma_tensor, smem_layout: staged }
}

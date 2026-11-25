// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py
//   (the uses of the helpers in `MixedInputGemmKernel`: the transform partition, the scale
//   partition and its TMA partition, which this file packages),
//   examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_host_utils.py
//   (the int4 shuffle order of the host data, in shuffle_a_host),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The helpers of a Blackwell mixed-input GEMM (CuTe DSL's
//! `cutlass.utils.mixed_input_helpers`). In such a GEMM, transform warps convert a narrow A
//! operand (Int4 or 8-bit, optionally scaled) to the MMA's type and store it into tensor
//! memory (A K-major, for a TS-UMMA) or shared memory (A MN-major), from where the MMA warp
//! consumes it. `examples/blackwell_mixed_input_gemm.rs` puts the helpers together.
//!
//! - Selection, on the host: [`TransformMode`], [`get_transform_a_source`],
//!   [`get_tma_atom_kind`] (2-CTA loads only for B), [`get_copy_atom_a_transform`]
//!   ([`select_copy_op_a_transform`]).
//! - Validity checks, on the host: [`is_valid_scale_granularity`],
//!   [`is_valid_tensor_alignment`], [`is_valid_mma_tiler_and_cluster_shape`],
//!   [`get_divisibility`], [`is_shuffle_a`].
//! - Layouts: [`get_gmem_layout_scale`], [`get_smem_layout_scale`], [`compute_smem_layout`].
//! - Partitions, while a kernel is built: [`scale_tma_partition`], [`transform_partition`],
//!   [`scale_partition`] and the epilogue's [`epilog_gmem_copy_and_partition`],
//!   [`epilog_smem_copy_and_partition`] and [`epilog_tmem_copy_and_partition`].
//! - Conversions of the A tile: [`cvt_tensor_a`] (Int4 to BFloat16, optionally from the
//!   shuffled order), [`cvt_tensor_a_mxf8`] (Int4 to E4M3 bytes), [`cvt_tensor_a_biased`]
//!   (bias-encoded Int4 to bytes, with [`cvt_i4x8_to_mxf8x8_biased`]), and
//!   [`store_transformed_a`].
//!
//! # The shuffled Int4 conversion
//!
//! CuTe DSL converts Int4 to BFloat16 with the scaled `.s2f6x2` cvt when its target CUDA is
//! 13.1 or newer and A is in the shuffled order (each group of eight K values stored as
//! `0, 2, 1, 3, 4, 6, 5, 7`, which [`shuffle_a_host`] produces), and with `lop3`/`prmt`/
//! `fma.rn.bf16x2` otherwise. cute-rs has no target CUDA version: the condition is the PTX ISA
//! the toolchain emits, 9.1 or newer ([`SHUFFLE_MIN_PTX_ISA`], LLVM 23;
//! [`shuffle_supported_by`]), and callers pass the outcome to [`is_shuffle_a`] and
//! [`cvt_tensor_a`]. **By default nothing is shuffled**: the example's configurations set
//! `shuffle_a: false`, which every toolchain cute-rs supports compiles, and the shuffled path
//! is opt-in for an LLVM 23 toolchain.

use crate::arch::{cvt_i4_bf16_intrinsic, cvt_i4_mxf8_intrinsic, lop3, prmt};
use crate::atom::copy::{
    CopyAtom, CopyOp, TiledCopy, autovec_copy, copy, make_copy_atom, make_cotiled_copy, make_tiled_copy_s,
};
use crate::atom::mma::TiledMma;
use crate::ir::DType;
use crate::layout::{
    ComposedLayout, IndexExpr, Layout, Shape, StrideLeaf, Tileable, Tiler, Tree, flat_divide, logical_product,
    make_composed_layout, make_swizzle, size_in_bytes_dtype,
};
use crate::numeric::I4;
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::{CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp, CtaGroup};
use crate::nvgpu::tcgen05::{OperandSource, Repetition, St16x256bOp, St32x32bOp, make_tmem_copy};
use crate::tensor::{AddressSpace, Tensor, make_rmem_tensor, make_tensor};
use crate::tensor_ssa::TensorSsa;
use crate::utils::LayoutEnum;
use crate::utils::sm100::{Sm100TmaLoadOp, make_smem_layout_a, make_smem_layout_b};
use crate::value::{Int32, Numeric, Uint32, Value, pack, unpack};

// The grouped variants' search (ported earlier, in `utils`), under this module's name too.
pub use super::{
    ContiguousGGSearchState, ContiguousGroupWorkTileInfo, contiguous_group_search,
    create_initial_contiguous_group_search_state, make_contiguous_group_work_tile_info,
};

/// The transform of a mixed-input GEMM (CuTe DSL's `TransformMode`): A converted to the
/// MMA's type, or converted and multiplied by a scale tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TransformMode {
    ConvertOnly,
    ConvertScale,
}

impl TransformMode {
    /// The mode of the scale granularities, as the example picks it: [`ConvertOnly`]
    /// when both are 0, [`ConvertScale`] otherwise.
    ///
    /// [`ConvertOnly`]: TransformMode::ConvertOnly
    /// [`ConvertScale`]: TransformMode::ConvertScale
    pub fn from_granularity(scale_granularity_m: usize, scale_granularity_k: usize) -> TransformMode {
        if scale_granularity_m == 0 && scale_granularity_k == 0 {
            TransformMode::ConvertOnly
        } else {
            TransformMode::ConvertScale
        }
    }
}

/// The PTX ISA version (`10 * major + minor`) from which cute-rs takes the shuffled Int4
/// conversion as available: 9.1, CUDA 13.1's (CuTe DSL's `target_version(min_version="13.1")`).
pub const SHUFFLE_MIN_PTX_ISA: u32 = 91;

/// Whether a toolchain emitting PTX ISA `ptx_isa` (`10 * major + minor`) assembles the
/// shuffled Int4 conversion's `.s2f6x2` cvt: 9.1 or newer ([`SHUFFLE_MIN_PTX_ISA`]).
pub fn shuffle_supported_by(ptx_isa: u32) -> bool {
    ptx_isa >= SHUFFLE_MIN_PTX_ISA
}

/// Where the transformed A lives (CuTe DSL's `get_transform_a_source(a_major_mode)`): tensor
/// memory for K-major A (the MMA reads A from TMEM), shared memory otherwise.
pub fn get_transform_a_source(a_major_mode: OperandMajorMode) -> OperandSource {
    if a_major_mode == OperandMajorMode::K { OperandSource::TMEM } else { OperandSource::SMEM }
}

/// The TMA load of an operand (CuTe DSL's `get_tma_atom_kind(mcast, use_2cta_instrs, is_b)`):
/// the multicast form when `mcast`, with [`CtaGroup::TWO`] only for B of a CTA pair (A is
/// loaded by each CTA for its own transform warps).
pub fn get_tma_atom_kind(mcast: bool, use_2cta_instrs: bool, is_b: bool) -> Sm100TmaLoadOp {
    let cta_group = if use_2cta_instrs && is_b { CtaGroup::TWO } else { CtaGroup::ONE };
    if mcast {
        Sm100TmaLoadOp::Multicast(CopyBulkTensorTileG2SMulticastOp { cta_group })
    } else {
        Sm100TmaLoadOp::Tile(CopyBulkTensorTileG2SOp { cta_group })
    }
}

/// The copy operation that stores the transformed A (the selection of CuTe DSL's
/// `get_copy_atom_a_transform`): into TMEM, `tcgen05.st.16x256b.x1` when A's partition shape
/// `a_smem_shape` (`((MMA_M, MMA_K), ...)`) has 64 rows and the MMA is 1-CTA, otherwise
/// `tcgen05.st.32x32b.x8`; into shared memory, a 32-bit universal copy.
#[track_caller]
pub fn select_copy_op_a_transform(
    use_2cta_instrs: bool,
    transform_a_source: OperandSource,
    a_smem_shape: &Shape,
) -> CopyOp {
    if transform_a_source == OperandSource::TMEM {
        let rows = match a_smem_shape {
            Tree::Branch(modes) if !modes.is_empty() => match &modes[0] {
                Tree::Branch(inner) if !inner.is_empty() => inner[0].size(),
                other => panic!("cute: get_copy_atom_a_transform: A's MMA mode {other} is not (MMA_M, MMA_K)"),
            },
            other => panic!("cute: get_copy_atom_a_transform: {other} is not A's partition shape ((M, K), ...)"),
        };
        if rows == 64 && !use_2cta_instrs {
            St16x256bOp::new(Repetition::X1).into()
        } else {
            St32x32bOp::new(Repetition::X8).into()
        }
    } else {
        CopyOp::Universal { bits: 32 }
    }
}

/// The copy atom of [`get_copy_atom_a_transform`]: a TMEM store of the MMA's type `M`, or a
/// shared-memory copy of A's type `A`.
#[derive(Clone, Debug)]
pub enum TransformCopyAtom<M, A> {
    Tmem(CopyAtom<M>),
    Smem(CopyAtom<A>),
}

/// The copy atom that stores the transformed A (CuTe DSL's
/// `get_copy_atom_a_transform(mma_dtype, use_2cta_instrs, transform_a_source, a_smem_shape,
/// a_dtype)`): [`select_copy_op_a_transform`]'s operation on the MMA's type `M` into TMEM, or
/// on A's type `A` into shared memory.
#[track_caller]
pub fn get_copy_atom_a_transform<M: Numeric, A: Numeric>(
    use_2cta_instrs: bool,
    transform_a_source: OperandSource,
    a_smem_shape: &Shape,
) -> TransformCopyAtom<M, A> {
    let op = select_copy_op_a_transform(use_2cta_instrs, transform_a_source, a_smem_shape);
    if transform_a_source == OperandSource::TMEM {
        TransformCopyAtom::Tmem(make_copy_atom::<M>(op))
    } else {
        TransformCopyAtom::Smem(make_copy_atom::<A>(op))
    }
}

/// Whether the scale granularities suit A's type (CuTe DSL's `is_valid_scale_granularity`):
/// none for 8-bit A (both 0); for 4-bit A, one scale per row (`scale_granularity_m` 1) and
/// a K granularity that divides `k` and is a multiple of the MMA tile's K.
pub fn is_valid_scale_granularity(
    scale_granularity_m: usize,
    scale_granularity_k: usize,
    a_dtype: DType,
    k: usize,
    mma_tiler_k: usize,
) -> bool {
    match a_dtype.bits() {
        8 => scale_granularity_m == 0 && scale_granularity_k == 0,
        4 => {
            scale_granularity_m == 1
                && scale_granularity_k != 0
                && k.is_multiple_of(scale_granularity_k)
                && scale_granularity_k.is_multiple_of(mma_tiler_k)
        }
        _ => true,
    }
}

/// Whether A is converted from the shuffled order (CuTe DSL's `is_shuffle_a`): K-major Int4
/// A, `k` a multiple of 8, a BFloat16 MMA and a K granularity of at least 8, on a toolchain
/// that supports it (`shuffle_supported`, see [`shuffle_supported_by`]).
pub fn is_shuffle_a(
    a_major: OperandMajorMode,
    k: usize,
    a_dtype: DType,
    mma_dtype: DType,
    scale_granularity_k: usize,
    shuffle_supported: bool,
) -> bool {
    let shuffle_a = a_major == OperandMajorMode::K
        && a_dtype == DType::I4
        && k.is_multiple_of(8)
        && mma_dtype == DType::BF16
        && scale_granularity_k >= 8;
    shuffle_a && shuffle_supported
}

/// Whether the tensors are aligned for the kernel's TMA loads and stores (CuTe DSL's
/// `is_valid_tensor_alignment`): the contiguous extents of A `(m, k)`, B `(n, k)` and C
/// `(m, n)` (and of the scale tensor, `(m, k / scale_granularity_k)` contiguous along M,
/// counted in B's type) hold whole 16-byte runs; and with an M granularity, each CTA's scale
/// rows (`cta_m / cluster_n / scale_granularity_m` of `scale_dtype`) take at least 128 bytes.
#[allow(clippy::too_many_arguments)]
pub fn is_valid_tensor_alignment(
    (m, n, k): (usize, usize, usize),
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    scale_dtype: DType,
    a_major: OperandMajorMode,
    b_major: OperandMajorMode,
    c_layout: LayoutEnum,
    mma_tiler_mnk: (usize, usize, usize),
    use_2cta_instrs: bool,
    cluster_shape_mn: (usize, usize),
    scale_granularity_m: usize,
    scale_granularity_k: usize,
) -> bool {
    let aligned = |dtype: DType, mode0_major: bool, (d0, d1): (usize, usize)| {
        let major = if mode0_major { d0 } else { d1 };
        major.is_multiple_of(128 / dtype.bits())
    };
    let scale_ok = scale_granularity_k == 0 || aligned(b_dtype, true, (m, k / scale_granularity_k));
    if !(aligned(a_dtype, a_major == OperandMajorMode::MN, (m, k))
        && aligned(b_dtype, b_major == OperandMajorMode::MN, (n, k))
        && aligned(c_dtype, c_layout.is_m_major_c(), (m, n))
        && scale_ok)
    {
        return false;
    }
    let cta_m = mma_tiler_mnk.0 / if use_2cta_instrs { 2 } else { 1 };
    !(scale_granularity_m > 0 && (cta_m / cluster_shape_mn.1 / scale_granularity_m) * (scale_dtype.bits() / 8) < 128)
}

/// Whether the MMA tile and the cluster shape suit the kernel (CuTe DSL's
/// `is_valid_mma_tiler_and_cluster_shape`): the cluster's M extent is a multiple of the
/// CTAs per MMA and each CTA's share of the tile's M is 64 or 128.
pub fn is_valid_mma_tiler_and_cluster_shape(
    mma_tiler: (usize, usize, usize),
    cluster_shape_mn: (usize, usize),
    use_2cta_instrs: bool,
) -> bool {
    let v = if use_2cta_instrs { 2 } else { 1 };
    cluster_shape_mn.0.is_multiple_of(v) && matches!(mma_tiler.0 / v, 64 | 128)
}

/// The largest power of two from 2 up that divides `contiguous_dim_size`, at most
/// `upper_bound`, or 1 (CuTe DSL's `get_divisibility(contiguous_dim_size, upper_bound=128)`):
/// the alignment, in elements, a tensor with that contiguous extent may assume.
pub fn get_divisibility(contiguous_dim_size: usize, upper_bound: usize) -> usize {
    if contiguous_dim_size == 0 {
        return 1;
    }
    let top = contiguous_dim_size.ilog2();
    (1..=top)
        .rev()
        .map(|i| 1usize << i)
        .find(|p| contiguous_dim_size.is_multiple_of(*p))
        .map_or(1, |p| p.min(upper_bound))
}

/// The global layout of the scale tensor (CuTe DSL's `get_gmem_layout_scale(scale_shape_mkl,
/// scale_granularity_m, scale_granularity_k, scale_major_mode)`): A's `(M, K, L)` coordinates
/// to the offset of their scale, `((gm, M / gm), (gk, K / gk), L)` with stride 0 within a
/// granule, the granules M-major (MN mode) or K-major, and L after the `(M, K)` scales.
pub fn get_gmem_layout_scale(
    scale_shape_mkl: (usize, usize, usize),
    scale_granularity_m: usize,
    scale_granularity_k: usize,
    scale_major_mode: OperandMajorMode,
) -> Layout {
    let (m, k, l) = scale_shape_mkl;
    let (gm, gk) = (scale_granularity_m, scale_granularity_k);
    let (rm, rk) = (m.div_ceil(gm), k.div_ceil(gk));
    let layout_mk = if scale_major_mode == OperandMajorMode::MN {
        Layout::new(((gm, rm), (gk, rk)), ((0, 1), (0, rm)))
    } else {
        Layout::new(((gm, rm), (gk, rk)), ((0, rk), (0, 1)))
    };
    let cos = layout_mk.cosize();
    Layout::from_modes([layout_mk.mode(0), layout_mk.mode(1), Layout::new(l, cos)])
}

/// The scale tile and its shared-memory layouts (CuTe DSL's `get_smem_layout_scale(mma_tiler,
/// use_2cta_instrs, scale_granularity_m, scale_granularity_k, scale_major_mode, a_scale_dtype,
/// num_scale_load2trans_stage)`): the `(MN, K)` scale tile of a CTA (half the MMA tile's M for
/// a CTA pair), one stage's layout `((smem_mn, MN / smem_mn), (smem_k, K / smem_k))` (the
/// granules, clipped to the tile, at stride 0, behind an identity swizzle) and the layout with
/// `num_scale_load2trans_stage` stages. Panics if the tile is not a whole number of granules
/// or one stage is under 128 bytes.
#[track_caller]
pub fn get_smem_layout_scale(
    mma_tiler: (usize, usize, usize),
    use_2cta_instrs: bool,
    scale_granularity_m: usize,
    scale_granularity_k: usize,
    scale_major_mode: OperandMajorMode,
    a_scale_dtype: DType,
    num_scale_load2trans_stage: usize,
) -> ((usize, usize), ComposedLayout, ComposedLayout) {
    let what = "cute: get_smem_layout_scale";
    let scale_tile_shape = (if use_2cta_instrs { mma_tiler.0 / 2 } else { mma_tiler.0 }, mma_tiler.2);
    let (size_mn, size_k) = scale_tile_shape;
    let smem_size_mn = scale_granularity_m.min(size_mn);
    let smem_size_k = scale_granularity_k.min(size_k);
    assert!(smem_size_mn > 0 && smem_size_k > 0, "{what}: the granularities must be positive for a scale tensor");
    let div_mn = size_mn.div_ceil(smem_size_mn);
    let div_k = size_k.div_ceil(smem_size_k);
    let shape = ((smem_size_mn, div_mn), (smem_size_k, div_k));
    let outer = if scale_major_mode == OperandMajorMode::MN {
        Layout::new(shape, ((0, 1), (0, div_mn)))
    } else {
        Layout::new(shape, ((0, div_k), (0, 1)))
    };
    let per_stage = make_composed_layout(make_swizzle(0, 4, 3), 0, outer.clone());
    assert!(
        mma_tiler.0.is_multiple_of(outer.mode(0).size()) && mma_tiler.2.is_multiple_of(outer.mode(1).size()),
        "{what}: one stage {outer} must tile the MMA tile {mma_tiler:?}"
    );
    let bytes = size_in_bytes_dtype(a_scale_dtype, &per_stage).as_static().expect("a static layout");
    assert!(bytes >= 128, "{what}: one stage of scales takes {bytes} bytes, under 128");
    let staged_outer = outer.append(Layout::new(num_scale_load2trans_stage, outer.cosize()));
    let staged = make_composed_layout(make_swizzle(0, 4, 3), 0, staged_outer);
    (scale_tile_shape, per_stage, staged)
}

/// The staged shared-memory layouts of A as loaded, A transformed (in the MMA's A type) and
/// B (CuTe DSL's `compute_smem_layout(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
/// load2trans_stage_count, trans2mma_stage_count)`).
#[track_caller]
pub fn compute_smem_layout(
    tiled_mma: &TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    a_dtype: DType,
    b_dtype: DType,
    load2trans_stage_count: usize,
    trans2mma_stage_count: usize,
) -> (ComposedLayout, ComposedLayout, ComposedLayout) {
    let mma_a_dtype = tiled_mma.atom().dtypes().0;
    let smem_layout_a = make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, load2trans_stage_count, None);
    let smem_layout_a_transform =
        make_smem_layout_a(tiled_mma, mma_tiler_mnk, mma_a_dtype, trans2mma_stage_count, None);
    let smem_layout_b = make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, load2trans_stage_count, None);
    (smem_layout_a, smem_layout_a_transform, smem_layout_b)
}

/// `((x, 1), rest...)` with stride `((d, 0), rest...)`: a unit sub-mode under mode 0.
fn nest_mode0<T>(t: &Tensor<T>) -> Tensor<T> {
    let l = t.layout();
    let m0 = l.mode(0);
    let nested = Layout::from_modes([m0, Layout::new(1, 0)]);
    let modes: Vec<Layout> = std::iter::once(nested).chain((1..l.rank()).map(|i| l.mode(i))).collect();
    t.with_layout(Layout::from_modes(modes))
}

/// The TMA partition of the scale tensor (CuTe DSL's `scale_tma_partition(tCsS, tCgS,
/// tma_atom_s, block_in_cluster_coord_vmnk, scale_cta_layout)`): [`tma_partition`] of the
/// MMA-partitioned shared and global scale tensors (their first three modes grouped) at the
/// CTA's N coordinate in the cluster, with mode 0 of each result nested as `(x, 1)`.
/// Returns `(tSsS, tSgS)`.
///
/// [`tma_partition`]: crate::cpasync::tma_partition
#[track_caller]
pub fn scale_tma_partition<T: Numeric, S, G>(
    t_cs_s: &Tensor<S>,
    t_cg_s: &Tensor<G>,
    tma_atom_s: &CopyAtom<T>,
    block_in_cluster_coord_vmnk: &crate::layout::Coord,
    scale_cta_layout: &Layout,
) -> (Tensor<S>, Tensor<G>) {
    let coord_n = match block_in_cluster_coord_vmnk {
        Tree::Branch(modes) if modes.len() > 2 => modes[2].clone(),
        other => panic!("cute: scale_tma_partition: {other:?} is not a (V, M, N, L) coordinate"),
    };
    // The scales repeat within a granule (stride 0); the TMA box holds each once, so the
    // grouped tile is taken without its stride-0 modes (one TMA copy per stage, not one per
    // repeat).
    let distinct = |l: &Layout| {
        let modes: Vec<Layout> =
            std::iter::once(l.mode(0).filter_zeros()).chain((1..l.rank()).map(|i| l.mode(i))).collect();
        Layout::from_modes(modes)
    };
    let s = t_cs_s.group_modes(0, 3);
    let g = t_cg_s.group_modes(0, 3);
    let (t_ss_s, t_sg_s) = crate::cpasync::tma_partition(
        tma_atom_s,
        coord_n,
        scale_cta_layout,
        &s.with_layout(distinct(s.layout())),
        &g.with_layout(distinct(g.layout())),
    );
    (nest_mode0(&t_ss_s), nest_mode0(&t_sg_s))
}

/// The transform warps' partition of A (from [`transform_partition`]).
#[derive(Clone, Debug)]
pub struct TransformPartition<A, M> {
    /// The tiled copy that loads A (and the scales) from shared memory, laid out as the
    /// store: only for [`TransformMode::ConvertScale`].
    pub src_copy_a: Option<TiledCopy<A>>,
    /// The TMEM store of the transformed A (`None` for shared memory, which
    /// [`store_transformed_a`] writes with `autovec_copy`).
    pub dst_copy_a: Option<TiledCopy<M>>,
    /// This thread's part of A as loaded: `(CPY, ..., STAGE)`.
    pub t_as_a_input: Tensor<A>,
    /// This thread's part of the transformed A: `(CPY, ..., STAGE)`.
    pub t_a_transform: Tensor<M>,
}

/// The transform warps' partition of A (CuTe DSL's `transform_partition(transform_a_source,
/// scale_mode, copy_atom_a_input, copy_atom_a_transform, sA_input, A_transform,
/// transform_local_tidx)`): A's staged input `s_a_input` and its transformed copy
/// `a_transform` (the TMEM fragment `(MMA, MMA_M, MMA_K, STAGE)` or the shared-memory tensor
/// of the same shape) partitioned for the transform thread `transform_local_tidx` (0 to 127).
///
/// - Into TMEM: the TMEM store atom tiled over one stage of `a_transform`
///   ([`make_tmem_copy`]) partitions both, A's 64-row input repeated twice along M for a
///   128-row store.
/// - Into shared memory: a 128-thread copy of 8 values per thread over one stage of
///   `a_transform`'s layout ([`make_cotiled_copy`]) partitions both.
///
/// With [`TransformMode::ConvertScale`], `src_copy_a` is `copy_atom_a_input` (required then)
/// tiled as that store's source ([`make_tiled_copy_s`]).
#[track_caller]
pub fn transform_partition<A: Numeric, M: Numeric>(
    transform_a_source: OperandSource,
    scale_mode: TransformMode,
    copy_atom_a_input: Option<&CopyAtom<A>>,
    copy_atom_a_transform: &TransformCopyAtom<M, A>,
    s_a_input: &Tensor<A>,
    a_transform: &Tensor<M>,
    transform_local_tidx: Int32,
) -> TransformPartition<A, M> {
    let what = "cute: transform_partition";
    let src_atom = || {
        copy_atom_a_input
            .unwrap_or_else(|| panic!("{what}: TransformMode::ConvertScale loads A with copy_atom_a_input; pass one"))
    };
    let stage0 = |t: &Tensor<M>| t.slice((.., .., .., 0));
    match (transform_a_source, copy_atom_a_transform) {
        (OperandSource::TMEM, TransformCopyAtom::Tmem(atom)) => {
            assert_eq!(a_transform.memspace(), AddressSpace::Tmem, "{what}: the transformed A is not in TMEM");
            let rows = |t: &Layout| t.mode(0).mode(0).size();
            let tensor_input = if rows(a_transform.layout()) == 128 && rows(s_a_input.layout()) == 64 {
                let tiler: Tiler = Tree::Branch(vec![
                    Tree::Branch(vec![Tree::Leaf(Some(Layout::new(2, 0))), Tree::Leaf(None)]),
                    Tree::Leaf(None),
                    Tree::Leaf(None),
                    Tree::Leaf(None),
                ]);
                make_tensor(s_a_input.ptr().clone(), logical_product(s_a_input.layout(), tiler))
            } else {
                s_a_input.clone()
            };
            let reg2tmem = make_tmem_copy(atom, &stage0(a_transform));
            let thr = reg2tmem.get_slice(transform_local_tidx);
            let t_as_a_input = thr.partition_s(&tensor_input);
            let t_a_transform = thr.partition_d(a_transform);
            let src_copy_a =
                (scale_mode == TransformMode::ConvertScale).then(|| make_tiled_copy_s(src_atom(), &reg2tmem));
            TransformPartition { src_copy_a, dst_copy_a: Some(reg2tmem), t_as_a_input, t_a_transform }
        }
        (OperandSource::SMEM, TransformCopyAtom::Smem(atom)) => {
            let reg2smem = make_cotiled_copy(atom, &Layout::new((128, 8), (8, 1)), stage0(a_transform).layout());
            let thr = reg2smem.get_slice(transform_local_tidx);
            let t_as_a_input = thr.partition_s(s_a_input);
            let t_a_transform = thr.partition_d(a_transform);
            let src_copy_a =
                (scale_mode == TransformMode::ConvertScale).then(|| make_tiled_copy_s(src_atom(), &reg2smem));
            TransformPartition { src_copy_a, dst_copy_a: None, t_as_a_input, t_a_transform }
        }
        (source, _) => panic!("{what}: the copy atom does not store into {source:?}"),
    }
}

/// The transform warps' partition of the scales (from [`scale_partition`]).
#[derive(Clone, Debug)]
pub struct ScalePartition<A, M> {
    /// The source copy's slice for this thread.
    pub smem_thr_copy_s: crate::atom::copy::ThrCopy<A>,
    /// This thread's scales in shared memory: `(CPY, ..., STAGE)`.
    pub t_ss_s_trans: Tensor<M>,
    /// The registers the distinct scales of one stage are copied to.
    pub t_sr_s_copy: Tensor<M>,
    /// Those registers viewed with the stage's shape, broadcast where the scales repeat.
    pub t_sr_s: Tensor<M>,
}

/// The transform warps' partition of the scales (CuTe DSL's `scale_partition(src_copy_a, tCsS,
/// transform_local_tidx, mma_dtype)`): the MMA-partitioned staged scales `t_cs_s` partitioned
/// by `src_copy_a` for thread `transform_local_tidx`, registers for one stage's distinct
/// scales (its layout without the stride-0 modes) and a view of them with the stage's shape.
#[track_caller]
pub fn scale_partition<A: Numeric, M: Numeric>(
    src_copy_a: &TiledCopy<A>,
    t_cs_s: &Tensor<M>,
    transform_local_tidx: Int32,
) -> ScalePartition<A, M> {
    let smem_thr_copy_s = src_copy_a.get_slice(transform_local_tidx);
    let t_ss_s_trans = smem_thr_copy_s.partition_s(t_cs_s);
    let per_stage = t_ss_s_trans.slice((.., .., .., .., 0)).layout().clone();
    let filtered = per_stage.filter_zeros();
    let t_sr_s_copy = make_rmem_tensor::<M>(Layout::col_major(filtered.shape().clone()));
    // The register strides where the stage's scales vary, 0 where they repeat.
    let zero = StrideLeaf::Int(IndexExpr::Static(0));
    let copy_strides = t_sr_s_copy.layout().stride().flat_leaves();
    let stage_strides = per_stage.stride().flat_leaves();
    let strides: Vec<StrideLeaf> = stage_strides
        .iter()
        .zip(&copy_strides)
        .map(|(s, c)| if s == &zero { zero.clone() } else { (*c).clone() })
        .collect();
    let stride = Tree::unflatten(&strides, per_stage.stride());
    let t_sr_s = make_tensor(t_sr_s_copy.ptr().clone(), Layout::new(per_stage.shape().clone(), stride));
    ScalePartition { smem_thr_copy_s, t_ss_s_trans, t_sr_s_copy, t_sr_s }
}

/// What [`epilog_gmem_copy_and_partition`] returns: the TMA store's atom and partitions and
/// the SIMT store's atom and partition, each when asked for.
#[derive(Clone, Debug)]
pub struct EpilogGmemPartition<C> {
    pub tma_atom_c: Option<CopyAtom<C>>,
    /// `((ATOM_V, REST_V), EPI_M, EPI_N)` of shared memory.
    pub b_sg_s_c: Option<Tensor<C>>,
    /// `((ATOM_V, REST_V), EPI_M, EPI_N, ...)` of global memory.
    pub b_sg_g_c: Option<Tensor<crate::tensor::Ident>>,
    pub simt_atom: Option<CopyAtom<C>>,
    /// This thread's part of C for the SIMT store.
    pub t_tr_g_c: Option<Tensor<C>>,
}

/// The global partitions of the epilogue's store (CuTe DSL's `epilog_gmem_copy_and_partition`):
/// with `tma_atom_c`, the epilogue tiles of the TMA tensor `g_c_mnl_tma` (`((bM, bN), 0, 0,
/// ...)` divided by `epi_tile`) and `s_c` partitioned for the TMA store; with
/// `tiled_copy_t2r`, those of the global tensor `g_c_mnl_simt` partitioned for thread `tidx`,
/// and a universal copy atom of C's type.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn epilog_gmem_copy_and_partition<C: Numeric, Acc: Numeric>(
    tidx: Int32,
    tma_atom_c: Option<&CopyAtom<C>>,
    tiled_copy_t2r: Option<&TiledCopy<Acc>>,
    g_c_mnl_tma: Option<&Tensor<crate::tensor::Ident>>,
    g_c_mnl_simt: Option<&Tensor<C>>,
    epi_tile: &Tiler,
    s_c: &Tensor<C>,
) -> EpilogGmemPartition<C> {
    let first_tile = |l: &Layout| {
        // ((bM, bN), 0, 0, rest...) -> (bM, bN, rest...)
        let m0 = l.mode(0);
        let modes: Vec<Layout> = [m0.mode(0), m0.mode(1)].into_iter().chain((3..l.rank()).map(|i| l.mode(i))).collect();
        Layout::from_modes(modes)
    };
    let (mut b_sg_s_c, mut b_sg_g_c) = (None, None);
    if let Some(atom) = tma_atom_c {
        let g = g_c_mnl_tma.expect("cute: epilog_gmem_copy_and_partition: a TMA store needs gC_mnl_tma");
        let g = g.with_layout(first_tile(g.layout()));
        let g_epi = flat_divide(&g, epi_tile.clone());
        let (s, gg) = crate::cpasync::tma_partition(
            atom,
            0,
            &Layout::new(1, 0),
            &s_c.group_modes(0, 2),
            &g_epi.group_modes(0, 2),
        );
        b_sg_s_c = Some(s);
        b_sg_g_c = Some(gg);
    }
    let (mut simt_atom, mut t_tr_g_c) = (None, None);
    if let Some(t2r) = tiled_copy_t2r {
        let g = g_c_mnl_simt.expect("cute: epilog_gmem_copy_and_partition: a SIMT store needs gC_mnl_simt");
        let g = g.with_layout(first_tile(g.layout()));
        let g_epi = flat_divide(&g, epi_tile.clone());
        t_tr_g_c = Some(t2r.get_slice(tidx).partition_d(&g_epi));
        simt_atom = Some(make_copy_atom::<C>(CopyOp::Universal { bits: C::DTYPE.bits() }));
    }
    EpilogGmemPartition { tma_atom_c: tma_atom_c.cloned(), b_sg_s_c, b_sg_g_c, simt_atom, t_tr_g_c }
}

/// The shared-memory partitions of the epilogue (CuTe DSL's `epilog_smem_copy_and_partition`):
/// [`epilogue_smem_copy_and_partition`](crate::utils::gemm::sm100::epilogue_smem_copy_and_partition).
pub use super::sm100_epilogue::epilogue_smem_copy_and_partition as epilog_smem_copy_and_partition;

/// The TMEM partitions of the epilogue (CuTe DSL's `epilog_tmem_copy_and_partition`):
/// [`epilogue_tmem_copy_and_partition`](crate::utils::gemm::sm100::epilogue_tmem_copy_and_partition).
pub use super::sm100_epilogue::epilogue_tmem_copy_and_partition as epilog_tmem_copy_and_partition;

/// A's tile loaded and converted to `D` (CuTe DSL's `cvt_tensor_a(src, dtype, shuffle)`): with
/// `shuffle` (a BFloat16 `D`, the tile's size a multiple of 8, each group of eight in the
/// shuffled order `0, 2, 1, 3, 4, 6, 5, 7`) by [`cvt_i4_bf16_intrinsic`]'s shuffled path,
/// otherwise by [`TensorSsa::to`]. Both are exact; the result is in natural order. See the
/// module documentation for when to shuffle; callers pass the decision (CuTe DSL also checks
/// the target CUDA version here, which cute-rs leaves to [`is_shuffle_a`]).
#[track_caller]
pub fn cvt_tensor_a<D: Numeric>(src: &Tensor<I4>, shuffle: bool) -> TensorSsa<D> {
    let rst = src.load_ssa();
    if !shuffle {
        return rst.to::<D>();
    }
    assert_eq!(
        D::DTYPE,
        DType::BF16,
        "cute: cvt_tensor_a: the shuffled conversion produces BFloat16, not {}",
        D::DTYPE
    );
    let n = rst.size();
    assert!(n.is_multiple_of(8), "cute: cvt_tensor_a: the shuffled conversion takes groups of 8 values, not {n}");
    let values: Vec<Value<D>> =
        cvt_i4_bf16_intrinsic(rst.values(), n, true).into_iter().map(|v| v.bitcast::<D>()).collect();
    TensorSsa::new(rst.shape().clone(), values)
}

/// A's tile loaded and converted to E4M3 bytes read as `F` (CuTe DSL's
/// `cvt_tensor_a_mxf8(src, dtype)`): [`cvt_i4_mxf8_intrinsic`], eight values at a time.
#[track_caller]
pub fn cvt_tensor_a_mxf8<F: Numeric>(src: &Tensor<I4>) -> TensorSsa<F> {
    let rst = src.load_ssa();
    let values = cvt_i4_mxf8_intrinsic::<F>(rst.values(), rst.size());
    TensorSsa::new(rst.shape().clone(), values)
}

/// Eight bias-encoded Int4 values to bytes read as `F` (CuTe DSL's
/// `cvt_i4x8_to_mxf8x8_biased`): the nibble `u` of each value (its bits) becomes the byte
/// `u ^ 8`, zero-extended, which for the encoding `u = x + 8` is `x`'s two's-complement
/// nibble. `lop3` (LUT `0x28`, `(a ^ b) & c`) flips the top bit of the even nibbles and clears
/// the odd ones, and likewise for the word shifted right by 4; two `prmt`s (`0x5140`,
/// `0x7362`) interleave the bytes back into element order.
#[track_caller]
pub fn cvt_i4x8_to_mxf8x8_biased<F: Numeric>(src_vec8: [Value<I4>; 8]) -> [Value<F>; 8] {
    assert_eq!(F::DTYPE.bits(), 8, "cute: cvt_i4x8_to_mxf8x8_biased: the bytes are 8-bit, not {}", F::DTYPE);
    let src: Uint32 = pack(src_vec8);
    let lo = lop3::<u32>(src, 0x8888_8888u32, 0x0F0F_0F0Fu32, 0x28);
    let hi = lop3::<u32>(src >> 4u32, 0x0808_0808u32, 0x0F0F_0F0Fu32, 0x28);
    let out0 = prmt::<u32>(lo, hi, 0x5140u32);
    let out1 = prmt::<u32>(lo, hi, 0x7362u32);
    let bytes: Vec<Value<i8>> = [out0, out1].into_iter().flat_map(unpack::<i8, u32, 4>).collect();
    std::array::from_fn(|k| bytes[k].bitcast::<F>())
}

/// The first `length` bias-encoded Int4 values of `vec_i4` as bytes read as `F`, eight at a
/// time with [`cvt_i4x8_to_mxf8x8_biased`] (CuTe DSL's `cvt_i4_mxf8_biased_intrinsic`).
/// `length` is a positive multiple of 8, at most `vec_i4.len()`.
#[track_caller]
pub fn cvt_i4_mxf8_biased_intrinsic<F: Numeric>(vec_i4: &[Value<I4>], length: usize) -> Vec<Value<F>> {
    assert!(
        length > 0 && length.is_multiple_of(8) && length <= vec_i4.len(),
        "cute: cvt_i4_mxf8_biased_intrinsic: length {length} must be a multiple of 8 from 8 to {}",
        vec_i4.len()
    );
    vec_i4[..length].chunks(8).flat_map(|c| cvt_i4x8_to_mxf8x8_biased::<F>(std::array::from_fn(|k| c[k]))).collect()
}

/// A's bias-encoded tile loaded and converted to bytes read as `F` (CuTe DSL's
/// `cvt_tensor_a_biased(src, dtype)`), with [`cvt_i4_mxf8_biased_intrinsic`].
#[track_caller]
pub fn cvt_tensor_a_biased<F: Numeric>(src: &Tensor<I4>) -> TensorSsa<F> {
    let rst = src.load_ssa();
    let values = cvt_i4_mxf8_biased_intrinsic::<F>(rst.values(), rst.size());
    TensorSsa::new(rst.shape().clone(), values)
}

/// Store the transformed A (CuTe DSL's `store_transformed_a(src_a, dst_a, copy_atom_a)`): with
/// the TMEM store `copy_a`, or with `autovec_copy` into shared memory.
#[track_caller]
pub fn store_transformed_a<M: Numeric>(src_a: &Tensor<M>, dst_a: &Tensor<M>, copy_a: Option<&TiledCopy<M>>) {
    match copy_a {
        Some(c) => copy(c, src_a, dst_a),
        None => autovec_copy(src_a, dst_a),
    }
}

/// The host side of the shuffled order (`mixed_input_host_utils.py`): each group of eight
/// consecutive values of `a` (A's K-major rows, `k` a multiple of 8) reordered `0, 2, 1, 3, 4,
/// 6, 5, 7`, the order [`cvt_tensor_a`]'s shuffled path converts from.
pub fn shuffle_a_host<T: Copy>(a: &[T]) -> Vec<T> {
    const PERM: [usize; 8] = [0, 2, 1, 3, 4, 6, 5, 7];
    assert!(a.len().is_multiple_of(8), "shuffle_a_host: {} values are not groups of 8", a.len());
    a.chunks(8).flat_map(|g| PERM.map(|p| g[p])).collect()
}

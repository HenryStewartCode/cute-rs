// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/copy_traits_sm90_tma.hpp (coalesce_256, construct_tma_gbasis,
//   fill_tma_gmem_shape_stride, make_tma_copy_desc, make_tma_copy_atom, make_tma_atom and
//   tma_partition), include/cute/atom/copy_traits_sm90_tma_swizzle.hpp
//   (get_tma_swizzle_bits) and include/cute/arch/copy_sm90_desc.hpp (the FP4/FP6 cases of
//   TMA::to_CUtensorMapDataType, in resolve_internal_type),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! TMA copy atoms: their construction from a global tensor, a shared-memory
//! layout and a CTA tiler ([`make_tiled_tma_atom`]), and the partitioning of
//! tiles into per-instruction boxes ([`tma_partition`]).
//!
//! The construction follows C++ CuTe's `make_tma_atom`. The shared-memory
//! layout (one stage of it) is inverted to find its largest contiguous run,
//! and composed with the CTA tile's map into the global tensor's modes
//! (`cta_v_map`, an identity layout of the tensor's shape composed with the
//! tiler) to give, for each run of shared memory, the global mode it walks.
//! The leading runs that walk a global mode with a unit step form the box;
//! consecutive ones contiguous in global memory are merged up to TMA's 256
//! elements per box dimension, the tensor's other modes are appended as
//! dimensions of extent 1 in the box, and trailing dimensions are folded into
//! the fifth when there are more than five. The descriptor then has the
//! global extents and byte strides of those dimensions, the box (split
//! across the CTAs of a multicast), and the swizzle of the shared-memory
//! layout. The TMA tensor ([`TmaInfo::tma_tensor`]) is a coordinate tensor
//! over the global tensor's shape whose scaled-basis strides give, for each
//! global coordinate, the coordinates of the TMA dimensions that the
//! instructions take.

use std::fmt;
use std::panic::Location;

use crate::atom::copy::{CopyAtom, CopyOp};
use crate::builder::{self, user_panic};
use crate::cpasync::{CtaGroup, TmaOpMode};
use crate::ir::{DType, Param};
use crate::layout::{
    ComposedLayout, Coord, IndexExpr, Layout, ScaledBasis, Shape, Stride, StrideLeaf, Swizzle, Tileable, Tiler, Tree,
    make_identity_layout,
};
use crate::tensor::{Engine, Ident, SmemLayout, Tensor, TensorInner, domain_offset_non_negative, make_coord_tensor};
use crate::tma::{MAX_BOX_DIM, MAX_RANK, TensorMapSwizzle, TmaDescSpec, TmaDescriptor};
use crate::value::Numeric;

/// What a TMA atom carries besides its thread-value layouts: the hidden
/// descriptor parameter it issues with, how the box maps onto the global
/// tensor, and the multicast split.
#[derive(Clone, Debug)]
pub struct TmaAtomInfo {
    pub(crate) descriptor: TmaDescriptor,
    pub(crate) gbasis: Layout,
    pub(crate) num_multicast: usize,
    pub(crate) swizzle: TensorMapSwizzle,
    pub(crate) smem_layout: ComposedLayout,
    /// For each TMA dimension, the flat modes of the global tensor it walks,
    /// and the global tensor's number of flat modes (what
    /// [`update_tma_descriptor`](crate::cpasync::update_tma_descriptor)
    /// rewrites the descriptor from).
    pub(crate) dim_modes: Vec<Vec<usize>>,
    pub(crate) flat_modes: usize,
    /// What an atom in another access mode needs when it issues.
    pub(crate) mode: TmaAtomMode,
}

/// The issue-time geometry of a TMA atom's access mode.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TmaAtomMode {
    /// `.tile`: the TMA tensor's coordinates are the instruction's.
    Tile,
    /// `.im2col` and `.im2col_no_offs`.
    Im2col(crate::nvgpu::tma_im2col::Im2colGeometry),
    /// `.tile::gather4` and `.tile::scatter4`: the index tensor's gather
    /// mode is the TMA tensor's flat mode `gather_mode`.
    Rows4 { gather_mode: usize },
    /// `.override::*`: the instructions take the global tensor's tile and a
    /// residue tensor.
    Override,
}

impl TmaAtomInfo {
    /// The element type the descriptor moves the global tensor's elements as
    /// (its `internal_type` or the tensor's own).
    pub fn data_type(&self) -> crate::tma::TensorMapDataType {
        self.spec().data_type
    }
}

impl TmaAtomInfo {
    /// For each TMA dimension, the flat modes of the global tensor it walks
    /// (one, or several folded into one dimension).
    pub fn dim_modes(&self) -> &[Vec<usize>] {
        &self.dim_modes
    }

    /// The number of flat modes of the global tensor the atom was made for.
    pub fn flat_modes(&self) -> usize {
        self.flat_modes
    }

    /// The hidden tensor-map parameter of the kernel that the atom's
    /// instructions take ([`TmaDescriptor::get_ptr`]).
    pub fn descriptor(&self) -> &TmaDescriptor {
        &self.descriptor
    }

    /// How the descriptor is computed at each launch.
    pub fn spec(&self) -> &TmaDescSpec {
        self.descriptor.spec().expect("a TMA atom's descriptor is a hidden one")
    }

    /// C++ CuTe's `TmaGmemBasis`: one mode per TMA dimension, its shape the
    /// box extents (before the multicast split) and its strides the bases of
    /// the global tensor's modes it walks (`(64,128):(1@1,1@0)` for a K-major
    /// 128x64 tile of an `(M,K)` tensor).
    pub fn tma_gbasis(&self) -> &Layout {
        &self.gbasis
    }

    /// The CTAs that share each box (1 without multicast).
    pub fn num_multicast(&self) -> usize {
        self.num_multicast
    }

    /// The descriptor's shared-memory swizzle.
    pub fn swizzle(&self) -> TensorMapSwizzle {
        self.swizzle
    }

    /// The shared-memory layout of one stage that the box was derived from.
    pub fn smem_layout(&self) -> &ComposedLayout {
        &self.smem_layout
    }
}

/// What [`make_tiled_tma_atom`] returns (CuTe DSL's `TmaInfo`): the atom,
/// the TMA tensor (a coordinate tensor over the global tensor's shape giving
/// TMA coordinates) and the shared-memory layout the atom was made for (as
/// given, with its stage mode). `let (atom, tma_tensor) = info.into();`
/// unpacks it as CuTe DSL's tuple unpacking does.
#[derive(Clone, Debug)]
pub struct TmaInfo<T> {
    pub atom: CopyAtom<T>,
    pub tma_tensor: Tensor<Ident>,
    pub smem_layout: ComposedLayout,
}

impl<T> TmaInfo<T> {
    /// The atom and the TMA tensor.
    pub fn into_parts(self) -> (CopyAtom<T>, Tensor<Ident>) {
        (self.atom, self.tma_tensor)
    }
}

impl<T> From<TmaInfo<T>> for (CopyAtom<T>, Tensor<Ident>) {
    fn from(info: TmaInfo<T>) -> Self {
        info.into_parts()
    }
}

/// The keyword arguments of CuTe DSL's `make_tiled_tma_atom` beyond the
/// multicast: the index tensor of the
/// gather4/scatter4 modes, and an internal type to move the elements as (an
/// 8-bit one unpacks FP4 and FP6 elements into 16-byte slots).
#[derive(Clone, Debug, Default)]
pub struct TmaAtomOptions {
    pub gmem_coord_tensor: Option<Tensor<i32>>,
    pub internal_type: Option<DType>,
}

impl TmaAtomOptions {
    pub fn new() -> TmaAtomOptions {
        TmaAtomOptions::default()
    }

    pub fn gmem_coord_tensor(mut self, t: &Tensor<i32>) -> Self {
        self.gmem_coord_tensor = Some(t.clone());
        self
    }

    pub fn internal_type(mut self, t: DType) -> Self {
        self.internal_type = Some(t);
        self
    }
}

/// A TMA atom copying tiles of `gmem_tensor` to or from shared memory laid
/// out as `smem_layout` (CuTe DSL's `cpasync.make_tiled_tma_atom(op,
/// gmem_tensor, smem_layout, cta_tiler, num_multicast)`), with its TMA
/// tensor. See [`make_tiled_tma_atom_with`].
#[track_caller]
pub fn make_tiled_tma_atom<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    cta_tiler: impl Into<Tiler>,
    num_multicast: usize,
) -> TmaInfo<T> {
    make_tiled_tma_atom_with(op, gmem_tensor, smem_layout, cta_tiler, num_multicast, &TmaAtomOptions::new())
}

/// [`make_tiled_tma_atom`] with CuTe DSL's keyword arguments
/// `gmem_coord_tensor` and `internal_type` ([`TmaAtomOptions`]).
///
/// - `op` is one of the TMA tile operations of [`cpasync`](crate::cpasync):
///   `CopyBulkTensorTileG2SOp` (`num_multicast` 1),
///   `CopyBulkTensorTileG2SMulticastOp` (any `num_multicast`),
///   `CopyBulkTensorTileS2GOp` or `CopyReduceBulkTensorTileS2GOp` (1).
/// - `gmem_tensor` is a global tensor on a kernel buffer parameter
///   ([`param_tensor`](crate::param_tensor), or [`make_tensor`](crate::make_tensor)
///   on a [`param_ptr`](crate::param_ptr)), unswizzled, with integer
///   strides; its extents and strides may be the kernel's scalar parameters.
/// - `smem_layout` is a plain or swizzled layout in units of elements, of
///   rank `rank(cta_tiler)`, or `rank(cta_tiler) + 1` with a stage mode last
///   (the first stage is used). Its swizzle, restated on bytes, must be
///   `Swizzle<1|2|3,4,3>` (the 32-, 64- and 128-byte swizzles of TMA).
/// - `cta_tiler` gives the tile of the tensor's leading modes one CTA
///   copies; it has as many elements as the layout.
///
/// The atom registers a hidden tensor-map parameter of the kernel, encoded
/// at each launch from the tensor's address and the launch's scalar
/// arguments. One thread issues an instruction (`thr_id` `1:0`), which
/// moves the box (`num_val_src` elements: the whole box, of which each CTA
/// of a multicast loads its share). What the build can check fails here;
/// extents and strides known only at launch fail the launch.
#[track_caller]
pub fn make_tiled_tma_atom_with<T: Numeric>(
    op: impl Into<CopyOp>,
    gmem_tensor: &Tensor<T>,
    smem_layout: impl SmemLayout,
    cta_tiler: impl Into<Tiler>,
    num_multicast: usize,
    options: &TmaAtomOptions,
) -> TmaInfo<T> {
    let loc = Location::caller();
    let tile = TmaTile::Tiler(cta_tiler.into());
    make_tma_atom_core(
        loc,
        "make_tiled_tma_atom",
        op.into(),
        gmem_tensor,
        smem_layout.into_composed(),
        tile,
        num_multicast,
        options,
    )
}

/// How a TMA atom's CTA tile is given: by a tiler of the tensor's leading
/// modes, or by a `cta_v_map` (CTA value index -> global mode) whose
/// first `rank` modes match one stage of the SMEM layout (the SM100
/// MMA-aware atoms).
pub(crate) enum TmaTile {
    Tiler(Tiler),
    VMap { map: Layout, rank: usize },
}

/// The body of [`make_tiled_tma_atom_with`], for a tile given either way;
/// `what` names the user-facing function in messages.
/// A TMA load for a CTA pair (`cta_group=CtaGroup::TWO`,
/// `.cta_group::2`) needs a kernel target with tcgen05's CTA pairs (sm_100a,
/// sm_101a, sm_110a, or their `f` families from PTX 8.8); SM90 has none.
pub(crate) fn check_pair_load_target() -> Result<(), String> {
    match crate::builder::target_arch() {
        Some(arch) if crate::validate::TCGEN05_ALLOC.ptx_on(arch).is_none() => Err(format!(
            "cta_group=CtaGroup::TWO loads for the CTA pair of SM100's 2-CTA tcgen05 MMAs, which the kernel's target \
             {arch} does not have (sm_100a, sm_101a, sm_110a or their f families)"
        )),
        _ => Ok(()),
    }
}

pub(crate) fn make_tma_atom_core<T: Numeric>(
    loc: &'static Location<'static>,
    what: &str,
    op: CopyOp,
    gmem_tensor: &Tensor<T>,
    staged: ComposedLayout,
    tile: TmaTile,
    num_multicast: usize,
    options: &TmaAtomOptions,
) -> TmaInfo<T> {
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: {what}({op:?}): {m}")) };
    // The gather4/scatter4 and override operations check as their
    // tiled forms do.
    let (base_op, mode) = match op {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Im2col, .. } | CopyOp::TmaStoreMode { mode: TmaOpMode::Im2col } => {
            fail("im2col atoms come from make_im2col_tma_atom".to_string())
        }
        CopyOp::TmaLoadMode { mode, multicast, cta_group } => (CopyOp::TmaLoad { multicast, cta_group }, Some(mode)),
        CopyOp::TmaStoreMode { mode } => (CopyOp::TmaStore, Some(mode)),
        op => (op, None),
    };
    if mode == Some(TmaOpMode::Rows4) && num_multicast != 1 {
        fail(format!(
            "a gather4 atom's instructions each move four whole rows, which cute-rs does not split among CTAs: \
             num_multicast is 1, not {num_multicast} (the multicast form writes each box to every CTA of its mask)"
        ));
    }
    match base_op {
        CopyOp::TmaLoad { multicast, cta_group } => {
            if cta_group == CtaGroup::TWO {
                // SM100's 2-CTA loads (CuTe DSL's TiledTmaLoadEnum.sm_100_2sm).
                if let Err(why) = check_pair_load_target() {
                    fail(why);
                }
                if mode.is_some() {
                    fail("cute-rs models cta_group=CtaGroup::TWO for tile and im2col loads only".to_string());
                }
            }
            if !multicast && num_multicast != 1 {
                fail(format!(
                    "CopyBulkTensorTileG2SOp loads for one CTA, so num_multicast is 1, not {num_multicast} (use \
                     CopyBulkTensorTileG2SMulticastOp)"
                ));
            }
            if num_multicast == 0 {
                fail("num_multicast is 0; it counts the CTAs sharing a box, at least 1".to_string());
            }
        }
        CopyOp::TmaStore | CopyOp::TmaReduce(_) => {
            if num_multicast != 1 {
                fail(format!("TMA stores do not multicast, so num_multicast is 1, not {num_multicast}"));
            }
        }
        _ => fail("the operation is not a TMA tile operation (cpasync::CopyBulkTensorTile*Op)".to_string()),
    }
    match (&options.gmem_coord_tensor, mode) {
        (Some(_), Some(TmaOpMode::Rows4)) | (None, None | Some(TmaOpMode::Override { .. } | TmaOpMode::Im2col)) => {}
        (None, Some(TmaOpMode::Rows4)) => fail(
            "a gather4/scatter4 atom needs gmem_coord_tensor, the index tensor whose layout drives its TMA tensor \
             (TmaAtomOptions::gmem_coord_tensor)"
                .to_string(),
        ),
        (Some(_), _) => fail("gmem_coord_tensor is for the gather4/scatter4 modes (CopyBulkTensor2DGather4G2SOp, CopyBulkTensor2DScatter4S2GOp)".to_string()),
    }

    // The global tensor: a buffer parameter of the kernel.
    let Engine::Memory { buffer, space, .. } = gmem_tensor.engine else {
        fail("the global tensor is not in memory (a register fragment or a coordinate tensor)".to_string())
    };
    let (is_param, decl) = builder::with(|b| {
        (b.params().iter().any(|p| matches!(p, Param::Buffer(x) if *x == buffer)), b.buffer(buffer).clone())
    });
    if space != crate::ir::MemSpace::Global {
        fail(format!("`{}` is in {} memory, not global memory", decl.name, crate::tensor::AddressSpace::from(space)));
    }
    if !is_param {
        fail(format!(
            "`{}` is not on a kernel buffer parameter (param_tensor, or make_tensor on param_ptr): the descriptor is \
             encoded on the host from the parameter's address",
            decl.name
        ));
    }
    if !matches!(gmem_tensor.inner, TensorInner::Identity) {
        fail(format!("`{}` is swizzled or composed: TMA describes a plain strided tensor", decl.name));
    }
    let data_type = resolve_internal_type(T::DTYPE, options.internal_type).unwrap_or_else(|e| fail(e));

    // One stage of the shared-memory layout.
    let tiler_rank = match &tile {
        TmaTile::Tiler(tiler) => {
            if tiler.is_leaf() {
                1
            } else {
                tiler.rank()
            }
        }
        TmaTile::VMap { rank, .. } => *rank,
    };
    let smem_rank = staged.rank();
    let smem = if smem_rank == tiler_rank + 1 {
        staged.select(&(0..tiler_rank).collect::<Vec<_>>())
    } else if smem_rank == tiler_rank {
        staged.clone()
    } else {
        fail(format!(
            "the SMEM layout {staged} has rank {smem_rank}, but the CTA tiler has rank {tiler_rank}: the layout is \
             one tile (the tiler's rank) or staged tiles (one more mode)"
        ))
    };

    let merge = mode != Some(TmaOpMode::Rows4);
    let cta_v_map = match &tile {
        TmaTile::Tiler(tiler) => tiler_v_map(gmem_tensor.layout(), tiler),
        TmaTile::VMap { map, .. } => Ok(map.clone()),
    };
    let mut plan = cta_v_map
        .and_then(|m| plan_tma_v(gmem_tensor.layout(), T::DTYPE.bits(), &smem, &m, num_multicast, merge))
        .unwrap_or_else(|e| fail(format!("`{}`: {e}", decl.name)));
    // Byte strides and offsets: sub-byte elements packed.
    let bits = T::DTYPE.bits();
    let to_bytes = |e: IndexExpr| {
        if bits.is_multiple_of(8) {
            IndexExpr::product(e, IndexExpr::Static(bits / 8))
        } else {
            IndexExpr::quotient(IndexExpr::product(e, IndexExpr::Static(bits)), IndexExpr::Static(8))
        }
    };
    let atom_mode = match mode {
        None => TmaAtomMode::Tile,
        Some(TmaOpMode::Rows4) => {
            let index = options.gmem_coord_tensor.as_ref().expect("checked");
            let gather_mode = crate::nvgpu::tma_sm100::check_rows4(gmem_tensor.layout(), index.layout(), &mut plan)
                .unwrap_or_else(|e| fail(format!("`{}`: {e}", decl.name)));
            TmaAtomMode::Rows4 { gather_mode }
        }
        Some(TmaOpMode::Override { .. }) => {
            if let Some(j) = plan.box_dims.iter().position(|&b| b > 255) {
                fail(format!(
                    "box dimension {j} is {} elements; an override copy gives each extent in 8 bits, so its box \
                     dimensions are at most 255",
                    plan.box_dims[j]
                ));
            }
            if plan.dim_modes.iter().any(|m| m.len() > 1) {
                fail("an override copy's TMA dimensions are the tensor's modes, unfolded".to_string());
            }
            TmaAtomMode::Override
        }
        Some(TmaOpMode::Im2col) => unreachable!("refused above"),
    };
    let spec = TmaDescSpec {
        data_type,
        global_dims: plan.global_dims.clone(),
        global_strides: plan.global_strides.iter().map(|s| to_bytes(s.clone())).collect(),
        box_dims: plan.box_dims.clone(),
        element_strides: vec![1; plan.box_dims.len()],
        interleave: crate::tma::TensorMapInterleave::none,
        swizzle: plan.swizzle,
        // C++ CuTe's choice for its TMA descriptors.
        l2_promotion: crate::tma::TensorMapL2Promotion::l2_128b,
        oob_fill: crate::tma::TensorMapFloatOOBFill::none,
        base_offset: to_bytes(gmem_tensor.offset.clone()),
        im2col: None,
    };
    let descriptor = crate::tma::declare_hidden_descriptor(buffer, &decl, spec).unwrap_or_else(|e| fail(e));
    let info = TmaAtomInfo {
        descriptor,
        gbasis: plan.gbasis,
        num_multicast,
        swizzle: plan.swizzle,
        smem_layout: smem,
        dim_modes: plan.dim_modes,
        flat_modes: plan.flat_modes,
        mode: atom_mode,
    };
    let atom = CopyAtom::tma(op, plan.box_elements, info);
    let tma_tensor = make_coord_tensor(0, Layout::new(gmem_tensor.layout().shape().clone(), plan.tma_stride));
    TmaInfo { atom, tma_tensor, smem_layout: staged }
}

/// The encoding of a descriptor for a tensor of `dtype` moved as CuTe DSL's
/// `internal_type` (CuTe DSL's `use_unpack` and C++
/// CuTe's `TMA::to_CUtensorMapDataType`): without one, the type's own
/// ([`TensorMapDataType::for_dtype`](crate::tma::TensorMapDataType::for_dtype): the
/// packed `f416u4_align8b` for FP4, the padded `f416u6_align16b` for FP6); an 8-bit
/// internal type for FP4 or FP6 elements unpacks them into 16-byte slots
/// (`f416u4_align16b`, `f416u6_align16b`); any other internal type of the element's
/// width is moved as that type's encoding.
pub(crate) fn resolve_internal_type(
    dtype: DType,
    internal: Option<DType>,
) -> Result<crate::tma::TensorMapDataType, String> {
    use crate::tma::TensorMapDataType as T;
    let Some(it) = internal else {
        return crate::tensor_map::resolve_format(dtype, None);
    };
    let narrow_float = matches!(dtype, DType::F4E2M1FN | DType::F6E2M3FN | DType::F6E3M2FN);
    if it.bits() == 8 && narrow_float {
        return Ok(if dtype == DType::F4E2M1FN { T::f416u4_align16b } else { T::f416u6_align16b });
    }
    if it.bits() != dtype.bits() {
        return Err(format!(
            "internal_type={} is {} bits, but the tensor's {} elements are {}: the internal type moves the same \
             bits (or, 8-bit, unpacks FP4 and FP6 values into 16-byte slots)",
            it.cute_name(),
            it.bits(),
            dtype.cute_name(),
            dtype.bits()
        ));
    }
    T::for_dtype(it).ok_or_else(|| format!("internal_type={} has no tensor-map encoding either", it.cute_name()))
}

/// A TMA dimension: the runs of the box along it, each `(extent in the box,
/// flat index of the global mode it walks)`.
type TmaMode = Vec<(usize, usize)>;

/// What [`plan_tma`] derives: everything the descriptor and the TMA tensor
/// need.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TmaPlan {
    /// The TMA dimensions' global extents.
    pub global_dims: Vec<IndexExpr>,
    /// The strides of dimensions 1 and up, in elements.
    pub global_strides: Vec<IndexExpr>,
    /// The box, split across the multicast.
    pub box_dims: Vec<u32>,
    /// Elements of the whole box (one instruction's values).
    pub box_elements: usize,
    pub swizzle: TensorMapSwizzle,
    /// The TMA tensor's strides, congruent with the global shape.
    pub tma_stride: Stride,
    /// C++ CuTe's `TmaGmemBasis`.
    pub gbasis: Layout,
    /// For each TMA dimension, the flat global modes it walks (one, or
    /// several folded), and the number of flat global modes.
    pub dim_modes: Vec<Vec<usize>>,
    pub flat_modes: usize,
}

/// The descriptor's swizzle for a shared-memory layout swizzled by `sw` on
/// the offsets of `elem_bits`-bit elements (C++ CuTe's
/// `get_tma_swizzle_bits` on the swizzle restated on bytes): none, or
/// `Swizzle<1|2|3,4,3>` on bytes.
pub(crate) fn tma_swizzle(sw: Swizzle, elem_bits: usize) -> Result<TensorMapSwizzle, String> {
    if sw.is_identity() {
        return Ok(TensorMapSwizzle::none);
    }
    let bytes = sw.to_bytes(elem_bits);
    match (bytes.bits, bytes.base, bytes.shift) {
        (1, 4, 3) => Ok(TensorMapSwizzle::s32b),
        (2, 4, 3) => Ok(TensorMapSwizzle::s64b),
        (3, 4, 3) => Ok(TensorMapSwizzle::s128b),
        // The 128-byte swizzle of 32-byte atoms (the SM100
        // UMMA layout `SW128_32B` that make_smem_layout_epi picks for M-major Float32
        // tiles), sm_100 and newer.
        (2, 5, 2) => match crate::builder::declared_target() {
            Some(arch) if arch.sm() < 100 => Err(format!(
                "the SMEM layout's swizzle {sw} on {elem_bits}-bit elements is {bytes} on bytes, TMA's \
                 s128b_atom_32b mode, which needs sm_100 or newer; the kernel targets {}",
                arch.chip()
            )),
            _ => Ok(TensorMapSwizzle::s128b_atom_32b),
        },
        _ => Err(format!(
            "the SMEM layout's swizzle {sw} on {elem_bits}-bit elements is {bytes} on bytes, which TMA cannot \
             express: TMA swizzles 16-byte chunks within 32-, 64- or 128-byte rows, S<1,4,3>, S<2,4,3> or S<3,4,3> \
             on bytes (and, on sm_100, 32-byte atoms within 128-byte rows, S<2,5,2>)"
        )),
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn leaf_paths<X>(t: &Tree<X>, at: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    match t {
        Tree::Leaf(_) => out.push(at.clone()),
        Tree::Branch(cs) => {
            for (i, c) in cs.iter().enumerate() {
                at.push(i);
                leaf_paths(c, at, out);
                at.pop();
            }
        }
    }
}

fn fmt_path(p: &[usize]) -> String {
    if p.is_empty() { "the whole tensor".to_string() } else { format!("mode {p:?}") }
}

/// C++ CuTe's `construct_tma_gbasis`, `fill_tma_gmem_shape_stride` and the
/// descriptor and TMA-stride parts of `make_tma_copy_desc`, for the global
/// layout `glayout` (integer strides, possibly runtime extents and strides),
/// one stage `smem` of the shared-memory layout (in elements of `elem_bits`
/// bits) and the CTA tiler.
#[cfg(test)]
pub(crate) fn plan_tma(
    glayout: &Layout,
    elem_bits: usize,
    smem: &ComposedLayout,
    tiler: &Tiler,
    num_multicast: usize,
) -> Result<TmaPlan, String> {
    plan_tma_v(glayout, elem_bits, smem, &tiler_v_map(glayout, tiler)?, num_multicast, true)
}

/// cta_v_map: tile coordinate -> global mode, the identity layout of the
/// tensor's shape composed with the tiler. As in C++ CuTe, the tensor's
/// modes past the tiler's rank are dropped.
pub(crate) fn tiler_v_map(glayout: &Layout, tiler: &Tiler) -> Result<Layout, String> {
    let identity = make_identity_layout(glayout.shape().clone());
    Ok(match tiler {
        Tree::Branch(ts) => {
            if ts.len() > identity.rank() {
                return Err(format!(
                    "the CTA tiler has {} modes, but the global tensor only {}",
                    ts.len(),
                    identity.rank()
                ));
            }
            identity.select(&(0..ts.len()).collect::<Vec<_>>()).compose_tiler(tiler)
        }
        Tree::Leaf(_) => identity.compose_tiler(tiler),
    })
}

/// The plan for a given `cta_v_map` (CTA value index -> global mode, C++
/// CuTe's `cta_v_map`): from a tiler ([`tiler_v_map`]), or from an MMA's
/// partitioning of the tile (the SM100 MMA-aware atoms). Runs contiguous in global memory merge into one
/// TMA dimension only with `merge` (gather4/scatter4 keep the tensor's two
/// modes apart).
pub(crate) fn plan_tma_v(
    glayout: &Layout,
    elem_bits: usize,
    smem: &ComposedLayout,
    cta_v_map: &Layout,
    num_multicast: usize,
    merge: bool,
) -> Result<TmaPlan, String> {
    // The swizzle.
    let sw = match smem.swizzle() {
        Some(sw) => sw,
        None => {
            return Err(format!(
                "the SMEM layout {smem} is composed with a layout, not a swizzle, which TMA cannot write"
            ));
        }
    };
    if smem.offset.leaves().iter().any(|o| o.as_static() != Some(0)) {
        return Err(format!(
            "the SMEM layout {smem} has an offset; TMA writes boxes from the start of a swizzle pattern"
        ));
    }
    let swizzle = tma_swizzle(sw, elem_bits)?;
    let slayout = smem.outer.clone();
    if !slayout.is_static() || slayout.has_scaled_basis() {
        return Err(format!("the SMEM layout {slayout} is not static with integer strides"));
    }

    // The global tensor's flat modes.
    let gshape = glayout.shape();
    let mut paths = Vec::new();
    leaf_paths(gshape, &mut Vec::new(), &mut paths);
    let extents = gshape.flat_leaves();
    let mut strides = Vec::new();
    for (k, s) in glayout.stride().flat_leaves().into_iter().enumerate() {
        match s {
            StrideLeaf::Int(e) => strides.push(e),
            StrideLeaf::Basis(b) => {
                return Err(format!("{} has the basis stride {b}; TMA needs integer strides", fmt_path(&paths[k])));
            }
        }
    }

    let v_size = cta_v_map.shape().flat_leaves().iter().map(|e| e.as_static()).collect::<Option<Vec<_>>>();
    let Some(v_size) = v_size.map(|v| v.iter().product::<usize>()) else {
        return Err(format!("the CTA tile {} has a runtime extent; TMA boxes are static", cta_v_map.shape()));
    };
    if v_size != slayout.size() {
        return Err(format!(
            "the SMEM layout {slayout} has {} elements, but the CTA tile {} has {v_size}",
            slayout.size(),
            cta_v_map.shape()
        ));
    }

    // smem index -> global mode, over the largest contiguous run of smem.
    let inv = slayout.right_inverse();
    let sidx2gmode = cta_v_map.compose(&inv).coalesce();
    let s_shapes = sidx2gmode.static_shape().flat_leaves();
    let s_strides = sidx2gmode.stride().flat_leaves();
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (n, s) in s_shapes.iter().zip(&s_strides) {
        let StrideLeaf::Basis(b) = s else { break };
        if b.value().as_int().and_then(|v| v.as_static()) != Some(1) {
            break;
        }
        let Some(k) = paths.iter().position(|p| p.as_slice() == b.mode()) else {
            return Err(format!("the tile walks the global {} which is not a mode of the tensor", fmt_path(b.mode())));
        };
        kept.push((*n, k));
    }
    if kept.is_empty() {
        return Err(format!(
            "the SMEM layout {slayout} has no contiguous run along a unit step of a global mode (its smem index -> \
             global mode map is {sidx2gmode}): a TMA box is contiguous in shared memory, so the layout's element at \
             offset 1 must be the tile's next element along some global mode"
        ));
    }

    // coalesce_256: merge runs contiguous in global memory while the box
    // dimension stays within 256 elements.
    let mut modes: Vec<TmaMode> = Vec::new();
    let mut back: Option<(usize, Option<usize>)> = None; // (merged extent, static stride of its first run)
    for (i, &(n, k)) in kept.iter().enumerate() {
        if i > 0 && n == 1 {
            continue;
        }
        let d = strides[k].as_static();
        // Runs fold into one dimension only over static extents (a
        // scale-factor tile's 32-row run and its runtime-extent K-chunk mode are contiguous
        // in global memory, but stay two dimensions).
        let foldable = |j: usize| extents[j].as_static().is_some() && strides[j].as_static().is_some();
        let merge = merge && foldable(k) && modes.last().is_some_and(|m| m.iter().all(|&(_, j)| foldable(j)));
        match back {
            Some((1, _)) => {
                modes.last_mut().unwrap().clear();
                modes.last_mut().unwrap().push((n, k));
                back = Some((n, d));
            }
            Some((bn, Some(bd))) if merge && d == Some(bn * bd) && n * bn <= MAX_BOX_DIM as usize => {
                modes.last_mut().unwrap().push((n, k));
                back = Some((n * bn, Some(bd)));
            }
            _ => {
                modes.push(vec![(n, k)]);
                back = Some((n, d));
            }
        }
    }
    if modes.len() == 1 && modes[0].iter().all(|(n, _)| *n == 1) {
        return Err("the tile is a single element; a TMA box is at least 16 bytes".to_string());
    }

    // The tensor's modes the box does not walk, as dimensions of extent 1 in
    // the box (unless static extent 1 or stride 0).
    for k in 0..paths.len() {
        if extents[k].as_static() == Some(1) || strides[k].as_static() == Some(0) {
            continue;
        }
        if modes.iter().any(|m| m.iter().any(|&(_, j)| j == k)) {
            continue;
        }
        modes.push(vec![(1, k)]);
    }
    // At most five dimensions: the trailing ones fold into the fifth.
    let full_rank = modes.len();
    if full_rank > MAX_RANK {
        let tail: Vec<(usize, usize)> = modes.drain(MAX_RANK - 1..).flatten().collect();
        modes.push(tail);
    }

    // Global extents and strides (in elements) of each dimension.
    let mut global_dims = Vec::new();
    let mut gstrides = Vec::new();
    let mut box_dims = Vec::new();
    for (j, m) in modes.iter().enumerate() {
        let boxj: usize = m.iter().map(|(n, _)| n).product();
        if m.len() == 1 {
            let k = m[0].1;
            global_dims.push(extents[k].clone());
            gstrides.push(strides[k].clone());
        } else {
            let mut shape: u64 = 1;
            let mut stride: u64 = 0;
            for &(_, k) in m {
                let (Some(sj), Some(dj)) = (extents[k].as_static(), strides[k].as_static()) else {
                    return Err(format!(
                        "TMA dimension {j} folds global modes {:?}, and folding modes with runtime extents or strides \
                         is not supported yet",
                        m.iter().map(|&(_, k)| paths[k].clone()).collect::<Vec<_>>()
                    ));
                };
                let (sj, dj) = (sj as u64, dj as u64);
                let old = stride;
                stride = gcd(stride, dj);
                shape = match old.checked_div(stride) {
                    Some(ratio) => (shape - 1) * ratio + (sj - 1) * (dj / stride) + 1,
                    None => sj,
                };
            }
            // The box walks the folded dimension one step at a time: that is the
            // tile's run only if the runs chain with the step the gcd gives.
            let runs: Vec<(usize, u64)> =
                m.iter().filter(|(n, _)| *n > 1).map(|&(n, k)| (n, strides[k].as_static().unwrap() as u64)).collect();
            for t in 0..boxj as u64 {
                let (mut rest, mut off) = (t, 0u64);
                for &(n, d) in &runs {
                    off += (rest % n as u64) * d;
                    rest /= n as u64;
                }
                if off != stride * t {
                    return Err(if full_rank > MAX_RANK {
                        format!(
                            "the tile needs {full_rank} TMA dimensions, and folding the last {} into TMA's fifth \
                             cannot express its box: TMA has at most {MAX_RANK} dimensions",
                            full_rank - MAX_RANK + 1
                        )
                    } else {
                        format!("TMA dimension {j} folds global modes whose runs do not chain")
                    });
                }
            }
            global_dims.push(IndexExpr::Static(shape as usize));
            gstrides.push(IndexExpr::Static(stride as usize));
        }
        if boxj > MAX_BOX_DIM as usize {
            return Err(format!(
                "the box's dimension {j} would hold {boxj} elements (a contiguous run of the SMEM layout along global \
                 {}), more than TMA's {MAX_BOX_DIM}; C++ CuTe refuses it too: use a layout whose contiguous runs are \
                 at most {MAX_BOX_DIM} elements (a swizzle atom, or a narrower tile)",
                fmt_path(&paths[m[0].1])
            ));
        }
        box_dims.push(boxj as u32);
    }
    if gstrides[0].as_static() != Some(1) {
        let k = modes[0][0].1;
        return Err(format!(
            "TMA's dimension 0 is contiguous, but the SMEM layout's contiguous run walks global {} whose stride is \
             {:?}, not 1: the SMEM layout's majorness does not match the global tensor's",
            fmt_path(&paths[k]),
            strides[k]
        ));
    }
    // A tile mode of stride 0 (a broadcast) would be a TMA dimension of
    // stride 0, which the driver refuses at the launch: say so here.
    if let Some(j) = (1..modes.len()).find(|&j| gstrides[j].as_static() == Some(0)) {
        return Err(format!(
            "the tile walks global {} whose stride is 0 (a broadcast mode), and a TMA dimension's stride cannot be \
             0: tile only the tensor's non-broadcast modes",
            fmt_path(&paths[modes[j][0].1])
        ));
    }
    let box_elements: usize = box_dims.iter().map(|&b| b as usize).product();

    // Split the box across the multicast, from the outermost dimension.
    let mut mc = num_multicast;
    let mut i = box_dims.len();
    while mc > 1 {
        if i == 0 {
            return Err(format!("the box {box_dims:?} cannot be split across {num_multicast} CTAs"));
        }
        i -= 1;
        let b = box_dims[i] as usize;
        if !(b.is_multiple_of(mc) || mc.is_multiple_of(b)) {
            return Err(format!(
                "box dimension {i} of {b} elements cannot be split across {mc} CTAs (one must divide the other)"
            ));
        }
        let next = mc.div_ceil(b);
        box_dims[i] = b.div_ceil(mc) as u32;
        mc = next;
    }
    // Each CTA's instruction writes its share at the box's start plus the
    // shares before it, and TMA's shared-memory destination is 128-byte
    // aligned (C++ CuTe does not check this; the instruction would fault).
    let share_bytes = box_elements / num_multicast * elem_bits / 8;
    if num_multicast > 1 && !share_bytes.is_multiple_of(128) {
        return Err(format!(
            "each of the {num_multicast} CTAs' share of the box {box_dims:?} is {share_bytes} bytes, so some CTA's \
             would start at an address that is not 128-byte aligned, as TMA's shared-memory destination must be: \
             use fewer CTAs or a layout with a larger box"
        ));
    }

    // The TMA tensor's strides: each global mode's basis in the TMA
    // coordinates.
    let elem_bytes = (elem_bits / 8) as u64;
    let mut leaves = Vec::new();
    for k in 0..paths.len() {
        let zero = StrideLeaf::Int(IndexExpr::Static(0));
        if extents[k].as_static() == Some(1) || strides[k].as_static() == Some(0) {
            leaves.push(zero);
            continue;
        }
        let Some(j) = modes.iter().position(|m| m.iter().any(|&(_, x)| x == k)) else {
            leaves.push(zero);
            continue;
        };
        let leaf = if j == 0 {
            StrideLeaf::Basis(ScaledBasis::new(strides[k].clone(), vec![0]))
        } else if modes[j].len() == 1 {
            StrideLeaf::Basis(ScaledBasis::new(IndexExpr::Static(1), vec![j]))
        } else {
            let di = strides[k].as_static().unwrap() as u64;
            let sb = gstrides[j].as_static().unwrap() as u64 * elem_bytes;
            let scale = (di * elem_bits as u64 / sb.max(16)).div_ceil(8);
            StrideLeaf::Basis(ScaledBasis::new(IndexExpr::Static(scale as usize), vec![j]))
        };
        leaves.push(leaf);
    }
    let tma_stride: Stride = Tree::unflatten(&leaves, gshape);

    // C++ CuTe's TmaGmemBasis, for inspection.
    let one = |n: usize, k: usize| (Shape::Leaf(IndexExpr::Static(n)), crate::layout::E(paths[k].clone()));
    let gbasis_modes: Vec<(Shape, Stride)> = modes
        .iter()
        .map(|m| {
            if m.len() == 1 {
                let (s, d) = one(m[0].0, m[0].1);
                (s, Tree::Leaf(d))
            } else {
                let (s, d): (Vec<_>, Vec<_>) =
                    m.iter().map(|&(n, k)| one(n, k)).map(|(s, d)| (s, Tree::Leaf(d))).unzip();
                (Tree::Branch(s), Tree::Branch(d))
            }
        })
        .collect();
    let (gs, gd): (Vec<_>, Vec<_>) = gbasis_modes.into_iter().unzip();
    let gbasis = Layout::new(Tree::Branch(gs), Tree::Branch(gd));

    Ok(TmaPlan {
        global_strides: gstrides[1..].to_vec(),
        global_dims,
        box_dims,
        box_elements,
        swizzle,
        tma_stride,
        gbasis,
        dim_modes: modes.iter().map(|m| m.iter().map(|&(_, k)| k).collect()).collect(),
        flat_modes: paths.len(),
    })
}

impl fmt::Display for TmaAtomInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.spec();
        write!(
            f,
            "TMA atom: gbasis {}, box {:?}, multicast {}, swizzle {}, descriptor `{}`",
            self.gbasis,
            s.box_dims,
            self.num_multicast,
            self.swizzle,
            self.descriptor.name()
        )
    }
}

/// Partition a shared-memory tile and the matching tile of a TMA tensor for
/// `atom` (CuTe DSL's `cpasync.tma_partition(atom, cta_coord, cta_layout,
/// smem_tensor, gmem_tensor)`, C++ CuTe's `tma_partition`), returning
/// `(tXsX, tXgX)`.
///
/// Both tensors have the tile in mode 0 (group the tile's modes first:
/// [`Tensor::group_modes`]`(0, 2)`) and any rest modes after it (stages,
/// k tiles). Mode 0 becomes `(TMA, TMA_Iter)`: the box of one instruction
/// (`atom.num_val_src()` elements, in the order of the shared-memory
/// layout's contiguous run) and the instructions that cover the tile. With a
/// multicast, `cta_layout` maps `cta_coord` (this CTA's coordinate among
/// those sharing the box, `make_layout(n)` for `n` CTAs) to its share, and
/// both views start at that share; without one, pass `0` and
/// `make_layout(1)`. The layout's cosize must be the atom's
/// `num_multicast`, and the coordinate's rank among those CTAs non-negative
/// and (when static) within them. Element `((0, i), rest...)` of the
/// TMA-tensor view is the TMA coordinate tuple of instruction `i`, and of
/// the shared-memory view the start of its box.
#[track_caller]
pub fn tma_partition<T: Numeric, S, G>(
    atom: &CopyAtom<T>,
    cta_coord: impl Into<Coord>,
    cta_layout: &Layout,
    smem_tensor: &Tensor<S>,
    gmem_tensor: &Tensor<G>,
) -> (Tensor<S>, Tensor<G>) {
    let loc = Location::caller();
    let fail = |m: String| -> ! { user_panic(loc, format!("cute: tma_partition: {m}")) };
    let Some(info) = atom.tma_info() else {
        fail(format!("{:?} is not a TMA atom (make one with cpasync::make_tiled_tma_atom)", atom.op()))
    };
    if matches!(smem_tensor.inner, TensorInner::Layout { .. }) {
        fail("the SMEM tensor is over a layout inner; TMA writes plain or swizzled shared memory".to_string());
    }
    // The shared-memory tile's contiguous run, repeated over the tile.
    let s0 = smem_tensor.layout().mode(0);
    if !s0.is_static() {
        fail(format!("the SMEM tile {s0} is not static"));
    }
    let size0 = s0.size();
    let inv = s0.right_inverse();
    let reps = size0 / inv.size();
    if reps * inv.size() != size0 {
        fail(format!("the SMEM tile {s0} is not a whole number of its contiguous runs {inv}"));
    }
    let rest = inv.complement(IndexExpr::Static(size0)).compose(&Layout::new(reps, 1));
    let layout_v = Layout::from_modes([inv, rest]).coalesce();
    let n = atom.num_val_src();
    if !size0.is_multiple_of(n) {
        fail(format!("the tile has {size0} elements, not a whole number of the atom's {n}-element boxes"));
    }
    let layout_vv = layout_v.compose(&Layout::new((n, size0 / n), (1, n)));
    // This CTA's share of the box.
    let cosize = cta_layout.cosize();
    if cosize == 0 || !n.is_multiple_of(cosize) {
        fail(format!("the box of {n} elements cannot be shared among the {cosize} CTAs of {cta_layout}"));
    }
    // The descriptor's box is the atom's share of the whole box: a CTA layout of
    // another size would offset the views by the wrong share (C++ CuTe does not
    // check this; a mismatch places the tile wrongly).
    let mc = info.num_multicast();
    if cosize != mc {
        fail(format!(
            "the CTA layout {cta_layout} spans {cosize} CTAs, but the atom's box is split across {mc} (its \
             num_multicast): pass make_layout({mc}) and this CTA's coordinate among them"
        ));
    }
    let rank = crate::layout::crd2idx(cta_coord, cta_layout);
    if !rank.is_non_negative() {
        fail(format!(
            "the CTA coordinate gives the multicast rank {rank:?}, which may be negative (it holds a signed value); \
             pass the CTA's non-negative rank among the {mc} CTAs"
        ));
    }
    if let Some(r) = rank.as_static()
        && r >= mc
    {
        fail(format!("the CTA coordinate gives the multicast rank {r}, outside the {mc} CTAs of {cta_layout}"));
    }
    let share = rank * IndexExpr::Static(n / cosize);

    fn part<X>(t: &Tensor<X>, layout_vv: &Layout, share: &IndexExpr, what: &str) -> Result<Tensor<X>, String> {
        let r = t.rank();
        let size0 = t.layout().mode(0).size();
        if size0 != layout_vv.size() {
            return Err(format!(
                "the {what} tensor's mode 0 has {size0} elements, the SMEM tile {}: group the tile's modes into mode 0",
                layout_vv.size()
            ));
        }
        let mut tiler: Vec<Tree<Option<Layout>>> = vec![Tree::Leaf(Some(layout_vv.clone()))];
        tiler.resize(r, Tree::Leaf(None));
        let tv = t.compose(Tree::Branch(tiler));
        let modes = tv.layout().modes();
        let m0 = Layout::from_modes([modes[0].mode(0).coalesce(), modes[0].mode(1).coalesce()]);
        let layout = Layout::from_modes(std::iter::once(m0).chain(modes[1..].iter().cloned()));
        let tv = tv.with_layout(layout);
        let mut coord: Vec<Coord> =
            vec![Tree::Branch(vec![Tree::Leaf(share.clone()), Tree::Leaf(IndexExpr::Static(0))])];
        coord.resize(r, Tree::Leaf(IndexExpr::Static(0)));
        Ok(domain_offset_non_negative(Tree::Branch(coord), &tv))
    }
    let s = part(smem_tensor, &layout_vv, &share, "SMEM").unwrap_or_else(|e| fail(e));
    let g = part(gmem_tensor, &layout_vv, &share, "GMEM").unwrap_or_else(|e| fail(e));
    (s, g)
}

/// The list form of CuTe DSL's `tma_partition` (`[coord_tensor, index_tensor]`,
/// giving `(tXsX, tXgX, tXgI)`) for the gather4/scatter4 modes.
pub use crate::nvgpu::tma_sm100::tma_partition_indexed;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{make_composed_layout, tile_to_shape_swizzled};

    #[test]
    fn swizzles_restated_on_bytes() {
        // Swizzle<B,4,3> on bytes is Swizzle<B,4,3> on u8, <B,3,3> on 16-bit and <B,2,3>
        // on 32-bit elements.
        for (b, want) in [(1, TensorMapSwizzle::s32b), (2, TensorMapSwizzle::s64b), (3, TensorMapSwizzle::s128b)] {
            assert_eq!(tma_swizzle(Swizzle::new(b, 4, 3), 8), Ok(want));
            assert_eq!(tma_swizzle(Swizzle::new(b, 3, 3), 16), Ok(want));
            assert_eq!(tma_swizzle(Swizzle::new(b, 2, 3), 32), Ok(want));
            assert!(tma_swizzle(Swizzle::new(b, 4, 3), 16).is_err());
            assert!(tma_swizzle(Swizzle::new(b, 3, 3), 32).is_err());
        }
        assert_eq!(tma_swizzle(Swizzle::identity(), 16), Ok(TensorMapSwizzle::none));
        // S<2,5,2> on bytes is S<2,3,2> on Float32.
        assert_eq!(tma_swizzle(Swizzle::new(2, 3, 2), 32), Ok(TensorMapSwizzle::s128b_atom_32b));
        assert_eq!(TensorMapSwizzle::s128b_atom_32b.to_byte_swizzle(), Swizzle::new(2, 5, 2));
        for a in 0..2048usize {
            let sw = Swizzle::new(2, 5, 2);
            assert_eq!(TensorMapSwizzle::s128b_atom_32b.apply(a as u64), sw.apply(a) as u64);
        }
        let e = tma_swizzle(Swizzle::new(3, 4, 4), 8).unwrap_err();
        assert!(e.contains("S<3,4,4> on 8-bit elements is S<3,4,4> on bytes"), "{e}");
    }

    #[test]
    fn plan_of_a_k_major_128b_tile() {
        // (512, 256) K-major f16, 128x64 tile over the 128B atom: box (64, 128).
        let g = Layout::new((512, 256), (256, 1));
        let s = tile_to_shape_swizzled(
            &make_composed_layout(Swizzle::new(3, 3, 3), 0, Layout::new((8, 64), (64, 1))),
            (128, 64),
        );
        let tiler: Tiler = (128, 64).into();
        let p = plan_tma(&g, 16, &s, &tiler, 1).unwrap();
        assert_eq!(p.box_dims, vec![64, 128]);
        assert_eq!(p.box_elements, 8192);
        assert_eq!(p.global_dims, vec![IndexExpr::Static(256), IndexExpr::Static(512)]);
        assert_eq!(p.global_strides, vec![IndexExpr::Static(256)]);
        assert_eq!(p.swizzle, TensorMapSwizzle::s128b);
        assert_eq!(p.tma_stride.to_string(), "(1@1,1@0)");
        let p = plan_tma(&g, 16, &s, &tiler, 8).unwrap();
        assert_eq!(p.box_dims, vec![64, 16]);
        // 256 CTAs would take 64 bytes each, not a 128-byte-aligned destination.
        let e = plan_tma(&g, 16, &s, &tiler, 256).unwrap_err();
        assert!(e.contains("is 64 bytes") && e.contains("not 128-byte aligned"), "{e}");
        assert!(plan_tma(&g, 16, &s, &tiler, 16384).unwrap_err().contains("cannot be split"));
        // A split past the outer dimension: an f32 box [64, 4] over 8 CTAs gives the outer
        // dimension's 4 rows one per pair of CTAs, and the pairs halve the inner one (C++
        // CuTe's loop: 8 / 4 = 2 left for dimension 0), 128 bytes each.
        let s = make_composed_layout(Swizzle::identity(), 0, Layout::new((4, 64), (64, 1)));
        let p = plan_tma(&g, 32, &s, &(4, 64).into(), 8).unwrap();
        assert_eq!(p.box_dims, vec![32, 1]);
    }
}

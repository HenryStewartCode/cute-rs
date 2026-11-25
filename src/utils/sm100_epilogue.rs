// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/epilogue/collective/builders/sm100_builder.inl
//   (sm100_get_tmem_load_op: the choice of the TMEM load shape, its packing and
//   its repetition from the datapaths and columns of a warp's epilogue tile, the
//   accumulator and D widths and D's majorness; sm100_get_smem_store_op: the
//   stmatrix variants compatible with the load; sm100_get_epilogue_smem_swizzle_layout_atom:
//   the swizzle atom from the contiguous run TMA can use) and
//   include/cute/atom/copy_traits_sm100.hpp (TMEM::op_repeater: the widest
//   repetition the columns allow),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The epilogue helpers of the Blackwell GEMMs (roadmap part
//! m10-epilogue-helpers): CuTe DSL's `cutlass.utils.sm100.get_tmem_load_op`,
//! `get_smem_store_op`, `get_smem_layout_atom_epi` and `make_smem_layout_epi`,
//! and `cutlass.utils.gemm.sm100`'s `transform_partitioned_tensor_layout`,
//! `epilogue_tmem_copy_and_partition`, `epilogue_smem_copy_and_partition`,
//! `epilogue_tma_store` and `epilogue`.
//!
//! An epilogue drains the accumulator tile in TMEM `epi_tile` by `epi_tile`:
//!
//! 1. [`get_tmem_load_op`] picks the `tcgen05.ld` of the tile (its shape,
//!    packing and repetition follow from the datapaths and columns of each
//!    of the four TMEM warps' share of the tile, the accumulator and D
//!    widths, and D's majorness);
//! 2. [`get_smem_store_op`] picks the widest `stmatrix` (or a plain store)
//!    whose fragments are the ones that load leaves in registers, so that the
//!    registers go to shared memory with no shuffling;
//! 3. [`make_smem_layout_epi`] gives the swizzled, staged shared-memory
//!    layout of the tile ([`get_smem_layout_atom_epi`] picks the atom).
//!
//! [`epilogue_tma_store`] runs the loop with a TMA store of each subtile
//! (under a [`PipelineTmaStore`]); [`epilogue`] stores straight to global
//! memory, predicated when given the coordinates and shape of C. CuTe DSL
//! passes the kernel object to both and reads attributes off it; here
//! [`EpilogueConfig`] holds those attributes.

use super::LayoutEnum;
use super::sm100::{default_tmem_warp_shape_mn, get_smem_layout_atom_ab};
use crate::arch::{FenceProxyKind, FenceProxyOptions, SharedSpace, fence_proxy_with};
use crate::atom::copy::{
    CopyAtom, CopyElem, CopyOp, R2GAttrs, TiledCopy, copy, copy_if, make_copy_atom, make_tiled_copy_d,
};
use crate::control::if_;
use crate::ir::{CacheEvictionPriority, DType, TmemShape};
use crate::layout::{
    ComposedLayout, IndexExpr, Layout, Shape, Tileable, Tiler, Tree, flat_divide, make_layout, tile_to_shape_ordered,
};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync;
use crate::nvgpu::tcgen05::{
    Ld16x32bx2Op, Ld16x64bOp, Ld16x128bOp, Ld16x256bOp, Ld32x32bOp, Pack, Repetition, SmemLayoutAtomKind,
    make_smem_layout_atom_dtype, make_tmem_copy,
};
use crate::pipeline::{NamedBarrier, Pipeline, PipelineState, PipelineTmaStore};
use crate::tensor::{Ident, Tensor, elem_less, make_fragment};
use crate::tensor_ssa::TensorSsa;
use crate::value::{Int32, Numeric};

// ---------------------------------------------------------------------------------------
// The choice of the TMEM load.
// ---------------------------------------------------------------------------------------

/// What [`get_tmem_load_op`] picks: the `tcgen05.ld` shape, its repetition and
/// whether it packs 16-bit values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmemLoadChoice {
    pub shape: TmemShape,
    pub repeat: Repetition,
    pub pack: bool,
}

impl TmemLoadChoice {
    /// The copy operation.
    pub fn op(self) -> CopyOp {
        CopyOp::TmemLoad { shape: self.shape, num: self.repeat.value(), pack: self.pack }
    }
}

/// `(product of the extents of each mode)` of an epilogue tile, which is a
/// by-mode tiler (`(tile_m, tile_n)` or the layouts of
/// [`compute_epilogue_tile_shape`](super::compute_epilogue_tile_shape)).
#[track_caller]
fn epi_tile_shape(epi_tile: &Tiler, what: &str) -> (usize, usize) {
    let Tree::Branch(modes) = epi_tile else {
        panic!("cute: {what}: the epilogue tile must have two modes (M, N), not a single {epi_tile:?}")
    };
    assert!(modes.len() == 2, "cute: {what}: the epilogue tile must have two modes (M, N), not {}", modes.len());
    let size = |t: &Tree<Option<Layout>>| match t {
        Tree::Leaf(Some(l)) => l.size(),
        other => panic!("cute: {what}: the epilogue tile's mode {other:?} is not a layout"),
    };
    (size(&modes[0]), size(&modes[1]))
}

/// The repetition of `n` as a [`Repetition`].
fn repetition(n: u32) -> Repetition {
    match n {
        1 => Repetition::X1,
        2 => Repetition::X2,
        4 => Repetition::X4,
        8 => Repetition::X8,
        16 => Repetition::X16,
        32 => Repetition::X32,
        64 => Repetition::X64,
        128 => Repetition::X128,
        _ => unreachable!("a repetition is a power of two up to 128"),
    }
}

/// The choice of [`get_tmem_load_op`] as data: see there. `cta_tile_shape` is
/// `(M, N)` or `(M, N, K)` of the CTA's tile (per CTA of a pair), `epi_tile`
/// the epilogue's tiler. Panics for 8 datapaths (or any count but 16 and 32)
/// per warp, for an epilogue tile the TMEM warp layout does not divide, for
/// D of 6 or 4 bits with 16 datapaths, and for a pair of widths no
/// instruction serves.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn select_tmem_load_op(
    cta_tile_shape: &[usize],
    layout_d: LayoutEnum,
    elem_ty_d: DType,
    elem_ty_acc: DType,
    epi_tile: &Tiler,
    use_2cta_instrs: bool,
    tmem_warp_shape_mn: Option<(usize, usize)>,
) -> TmemLoadChoice {
    const WHAT: &str = "get_tmem_load_op";
    assert!(!cta_tile_shape.is_empty(), "cute: {WHAT}: the CTA tile shape is empty");
    let (acc_bits, d_bits) = (elem_ty_acc.bits(), elem_ty_d.bits());
    let is_m_major = layout_d.is_m_major_c();
    let (warp_m, warp_n) =
        tmem_warp_shape_mn.unwrap_or_else(|| default_tmem_warp_shape_mn(cta_tile_shape[0], use_2cta_instrs));
    let (tile_m, tile_n) = epi_tile_shape(epi_tile, WHAT);
    assert!(
        warp_m > 0 && warp_n > 0 && tile_m.is_multiple_of(warp_m) && tile_n.is_multiple_of(warp_n),
        "cute: {WHAT}: the TMEM warp layout {warp_m}x{warp_n} does not divide the epilogue tile {tile_m}x{tile_n}"
    );
    // The datapaths and columns of one warp's part of the tile.
    let num_dp = tile_m / warp_m;
    assert!(
        num_dp == 16 || num_dp == 32,
        "cute: {WHAT}: a warp's part of the {tile_m}x{tile_n} epilogue tile has {num_dp} datapaths, but a TMEM warp \
         reaches 16 or 32 (with the CTA tile {cta_tile_shape:?} and 2-CTA MMA {use_2cta_instrs})"
    );
    let num_col_bits = tile_n / warp_n * acc_bits;

    // (datapaths, bits per datapath of one repetition, 16-bit packing)
    let (dp, bit, pack): (usize, usize, bool) = match (acc_bits, d_bits) {
        (32, 32) => match (num_dp, is_m_major) {
            (16, true) => (16, 256, false),
            (16, false) => (16, 128, false),
            _ => (32, 32, false),
        },
        (32, 16) => match (num_dp, is_m_major) {
            (_, true) | (16, false) => (16, 256, false),
            _ => (32, 32, false),
        },
        (32, 8) => match (num_dp, is_m_major) {
            (_, true) => (16, 256, false),
            (16, false) => (16, 32, false),
            _ => (32, 32, false),
        },
        (16, 16) => match (num_dp, is_m_major) {
            (_, true) | (16, false) => (16, 128, true),
            _ => (32, 32, true),
        },
        (32, 6) | (32, 4) => {
            assert!(num_dp == 32, "cute: {WHAT}: D of {d_bits} bits needs 32 datapaths per warp, not {num_dp}");
            (32, 32, false)
        }
        _ => panic!("cute: {WHAT}: no TMEM load serves {elem_ty_acc} accumulators with {elem_ty_d} D"),
    };
    let (shape, unit, max_rep) = match (dp, bit) {
        (16, 256) => (TmemShape::S16x256b, 256, 32),
        (16, 128) => (TmemShape::S16x128b, 128, 64),
        (16, 32) => (TmemShape::S16x32bx2, 64, 128),
        _ => (TmemShape::S32x32b, 32, 128),
    };
    // The widest repetition the columns span a whole number of times.
    let mut rep = max_rep;
    while rep > 1 && !num_col_bits.is_multiple_of(unit * rep) {
        rep /= 2;
    }
    assert!(
        num_col_bits.is_multiple_of(unit * rep),
        "cute: {WHAT}: a warp's {num_col_bits} bits of columns are not a whole number of the {unit}-bit units the \
         TMEM load repeats"
    );
    TmemLoadChoice { shape, repeat: repetition(rep as u32), pack }
}

/// A performant TMEM load for an epilogue tile (CuTe DSL's
/// `get_tmem_load_op(cta_tile_shape, layout_d, elem_ty_d, elem_ty_acc,
/// epi_tile, use_2cta_instrs, *, tmem_warp_shape_mn=None)`, CuTe C++'s
/// `sm100_get_tmem_load_op`), as a copy atom of the accumulator type `Acc`.
///
/// The four warps of the epilogue are laid out `tmem_warp_shape_mn` over the
/// tile (`(2, 2)` for a pair's MMA with 64 rows per CTA, else `(4, 1)`), each
/// taking `epi_tile / warp_shape` of it, which must have 16 or 32 datapaths
/// (rows). With 32 datapaths the narrower `32x32b` is preferred (more
/// throughput); with 16 only `16x*` forms reach. By the accumulator and D bit
/// widths and D's majorness:
///
/// | acc, D | 16 datapaths, M-major | 16, N-major | 32, M-major | 32, N-major |
/// |---|---|---|---|---|
/// | 32, 32 | `16x256b` | `16x128b` | `32x32b` | `32x32b` |
/// | 32, 16 | `16x256b` | `16x256b` | `16x256b` | `32x32b` |
/// | 32, 8 | `16x256b` | `16x32bx2` | `16x256b` | `32x32b` |
/// | 16, 16 (packed) | `16x128b` | `16x128b` | `16x128b` | `32x32b` |
/// | 32, 6 or 4 | refused | refused | `32x32b` | `32x32b` |
///
/// The repetition is the largest the columns' bits allow (`.x32` at most
/// for `16x256b`, `.x64` for `16x128b`, `.x128` otherwise). Panics when
/// nothing serves (see [`select_tmem_load_op`]).
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn get_tmem_load_op<Acc: Numeric>(
    cta_tile_shape: &[usize],
    layout_d: LayoutEnum,
    elem_ty_d: DType,
    epi_tile: &Tiler,
    use_2cta_instrs: bool,
    tmem_warp_shape_mn: Option<(usize, usize)>,
) -> CopyAtom<Acc> {
    let choice = select_tmem_load_op(
        cta_tile_shape,
        layout_d,
        elem_ty_d,
        Acc::DTYPE,
        epi_tile,
        use_2cta_instrs,
        tmem_warp_shape_mn,
    );
    let pack = if choice.pack { Pack::Pack16bIn32b } else { Pack::None };
    match choice.shape {
        TmemShape::S16x64b => make_copy_atom::<Acc>(Ld16x64bOp::new(choice.repeat).with_pack(pack)),
        TmemShape::S16x128b => make_copy_atom::<Acc>(Ld16x128bOp::new(choice.repeat).with_pack(pack)),
        TmemShape::S16x256b => make_copy_atom::<Acc>(Ld16x256bOp::new(choice.repeat).with_pack(pack)),
        TmemShape::S16x32bx2 => make_copy_atom::<Acc>(Ld16x32bx2Op::new(choice.repeat).with_pack(pack)),
        TmemShape::S32x32b => make_copy_atom::<Acc>(Ld32x32bOp::new(choice.repeat).with_pack(pack)),
    }
}

// ---------------------------------------------------------------------------------------
// The shared-memory store and layout.
// ---------------------------------------------------------------------------------------

/// The register-to-shared-memory store [`get_smem_store_op`] picks for D
/// under a given TMEM load, as a copy operation.
#[track_caller]
pub fn select_smem_store_op(layout_d: LayoutEnum, d_bits: usize, acc_bits: usize, tmem_load: &CopyOp) -> CopyOp {
    let (is_m_major, is_n_major) = (layout_d.is_m_major_c(), layout_d.is_n_major_c());
    let universal = CopyOp::Universal { bits: d_bits.max(8) };
    let CopyOp::TmemLoad { shape, num, pack } = *tmem_load else { return universal };
    let (num_dp, num_bits) = match shape {
        TmemShape::S16x64b => (16, 64),
        TmemShape::S16x128b => (16, 128),
        TmemShape::S16x256b => (16, 256),
        TmemShape::S16x32bx2 => (16, 32),
        TmemShape::S32x32b => (32, 32),
    };
    let repeated = |reps: &[u32]| reps.contains(&num);
    // The loads whose thread-value ownership is that of a 4-matrix stmatrix,
    // and (a single repetition lacks the values for four) of a 2-matrix one.
    let m8n8_4x = (acc_bits == 32
        && d_bits == 32
        && is_n_major
        && num_dp == 16
        && num_bits == 128
        && repeated(&[2, 4, 8, 16, 32, 64])
        && !pack)
        || (acc_bits == 32 && d_bits == 16 && num_dp == 16 && num_bits == 256 && repeated(&[2, 4, 8, 16, 32]) && !pack)
        || (acc_bits == 16
            && d_bits == 16
            && num_dp == 16
            && num_bits == 128
            && repeated(&[2, 4, 8, 16, 32, 64])
            && pack);
    let m8n8_2x =
        (acc_bits == 32 && d_bits == 32 && is_n_major && num_dp == 16 && num_bits == 128 && num == 1 && !pack)
            || (acc_bits == 32 && d_bits == 16 && num_dp == 16 && num_bits == 256 && num == 1 && !pack)
            || (acc_bits == 16 && d_bits == 16 && num_dp == 16 && num_bits == 128 && num == 1 && pack);
    let m16n8_matrices = if acc_bits == 32 && d_bits == 8 && is_m_major && num_dp == 16 && num_bits == 256 && !pack {
        match num {
            4 | 8 | 16 | 32 => Some(4),
            2 => Some(2),
            1 => Some(1),
            _ => None,
        }
    } else {
        None
    };
    if m8n8_4x {
        CopyOp::StMatrix { num: 4, trans: is_m_major }
    } else if m8n8_2x {
        CopyOp::StMatrix { num: 2, trans: is_m_major }
    } else if let Some(n) = m16n8_matrices {
        CopyOp::StMatrix16x8x8b { num: n, trans: true }
    } else {
        universal
    }
}

/// The widest shared-memory store of D that suits a TMEM load (CuTe DSL's
/// `get_smem_store_op(layout_d, elem_ty_d, elem_ty_acc, tiled_tmem_load)`,
/// CuTe C++'s `sm100_get_smem_store_op`): the `stmatrix` of 8x8 16-bit
/// matrices ([`CopyOp::StMatrix`], transposed for M-major D, four matrices
/// when the load's repetition gives the values and two for `x1`), or of
/// 16x8 8-bit ones ([`CopyOp::StMatrix16x8x8b`], 8-bit M-major D), whenever
/// the load's thread-value ownership is that of the `stmatrix`
/// fragments:
///
/// * `16x256b` loads (not packed) for 32-bit accumulators and 16-bit D;
/// * `16x128b` loads for 32-bit accumulators and N-major 32-bit D
///   (whose `stmatrix` stores the 32-bit values as pairs of 16-bit ones),
///   and packed `16x128b` loads of 16-bit accumulators for 16-bit D;
/// * `16x256b` loads for 32-bit accumulators and M-major 8-bit D.
///
/// Anything else, and a `tiled_tmem_load` that is not a TMEM load, gets a
/// plain one-element store ([`CopyOp::Universal`]).
#[track_caller]
pub fn get_smem_store_op<D: Numeric, A: Numeric>(
    layout_d: LayoutEnum,
    elem_ty_acc: DType,
    tiled_tmem_load: &TiledCopy<A>,
) -> CopyAtom<D> {
    make_copy_atom::<D>(select_smem_store_op(
        layout_d,
        D::DTYPE.bits(),
        elem_ty_acc.bits(),
        &tiled_tmem_load.atom().op(),
    ))
}

/// The shared-memory layout atom for the epilogue tile of C or D (CuTe DSL's
/// `get_smem_layout_atom_epi(layout, element_type, epi_tile)`, CuTe C++'s
/// `sm100_get_epilogue_smem_swizzle_layout_atom`): the atom of the SM100
/// operand selector for the M-major ([`OperandMajorMode::MN`]) or N-major
/// (K) layout, over the largest tile TMA can move contiguously: each mode of
/// `epi_tile` that is a layout counts as the extent of its right inverse
/// (`(tile_n / warp_n, warp_n):(1, cta_n / warp_n)` gives `tile_n / warp_n`
/// when the warps' slices are apart).
#[track_caller]
pub fn get_smem_layout_atom_epi(layout: LayoutEnum, element_type: DType, epi_tile: &Tiler) -> SmemLayoutAtomKind {
    let Tree::Branch(modes) = epi_tile else {
        panic!("cute: get_smem_layout_atom_epi: the epilogue tile must have two modes (M, N)")
    };
    assert!(modes.len() == 2, "cute: get_smem_layout_atom_epi: the epilogue tile must have two modes (M, N)");
    let tma = |t: &Tree<Option<Layout>>| match t {
        Tree::Leaf(Some(l)) => l.right_inverse().coalesce().size(),
        other => panic!("cute: get_smem_layout_atom_epi: the epilogue tile's mode {other:?} is not a layout"),
    };
    let shape = (tma(&modes[0]), tma(&modes[1]));
    let major = if layout.is_m_major_c() { OperandMajorMode::MN } else { OperandMajorMode::K };
    get_smem_layout_atom_ab(major, element_type, shape, 1)
}

/// The staged shared-memory layout of the epilogue tile (CuTe DSL's
/// `make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage)`):
/// [`get_smem_layout_atom_epi`]'s atom tiled to the tile's `(M, N)` extents
/// and `epi_stage` stages, in the order `(1, 0, 2)` for M-major C or D and
/// `(0, 1, 2)` otherwise. A Boolean tensor is stored as Int8.
#[track_caller]
pub fn make_smem_layout_epi(
    epi_dtype: DType,
    epi_layout: LayoutEnum,
    epi_tile: &Tiler,
    epi_stage: usize,
) -> ComposedLayout {
    let epi_dtype = if epi_dtype == DType::Bool { DType::I8 } else { epi_dtype };
    let (m, n) = epi_tile_shape(epi_tile, "make_smem_layout_epi");
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom_epi(epi_layout, epi_dtype, epi_tile), epi_dtype);
    let order: &[usize] = if epi_layout.is_n_major_c() { &[0, 1, 2] } else { &[1, 0, 2] };
    tile_to_shape_ordered(&atom, (m, n, epi_stage), order)
}

// ---------------------------------------------------------------------------------------
// The packaged epilogues.
// ---------------------------------------------------------------------------------------

/// What the epilogue helpers read off the GEMM kernel (CuTe DSL passes the
/// kernel object, which they duck-type: `cta_tile_shape_mnk`, `c_layout`,
/// `c_dtype`, `acc_dtype`, `use_2cta_instrs`, `tmem_warp_shape_mn`,
/// `epilog_sync_bar_id`, `epilogue_warp_id`, `num_c_stage`,
/// `iter_acc_early_release_in_epilogue`).
#[derive(Clone, Debug)]
pub struct EpilogueConfig {
    /// The CTA tile `(M, N, K)` (per CTA of a pair).
    pub cta_tile_shape_mnk: (usize, usize, usize),
    /// C's (and D's) majorness in global memory.
    pub c_layout: LayoutEnum,
    pub c_dtype: DType,
    pub acc_dtype: DType,
    pub use_2cta_instrs: bool,
    /// Overrides the default TMEM warp arrangement.
    pub tmem_warp_shape_mn: Option<(usize, usize)>,
    /// The named barrier of the epilogue warps (TMA store).
    pub epilog_sync_bar_id: u32,
    /// The warps of the epilogue (the first issues the TMA stores); their
    /// threads are the barrier's.
    pub epilogue_warp_id: Vec<u32>,
    /// The shared-memory stages of C (TMA store).
    pub num_c_stage: usize,
    /// With an overlapping accumulator, the subtile after which the
    /// accumulator stage is released early (the one whose TMEM columns the
    /// next tile's scale factors reuse).
    pub iter_acc_early_release_in_epilogue: usize,
}

impl EpilogueConfig {
    /// A configuration with 4 epilogue warps (0 to 3) on named barrier 2,
    /// one C stage and no early release.
    pub fn new(
        cta_tile_shape_mnk: (usize, usize, usize),
        c_layout: LayoutEnum,
        c_dtype: DType,
        acc_dtype: DType,
        use_2cta_instrs: bool,
    ) -> EpilogueConfig {
        EpilogueConfig {
            cta_tile_shape_mnk,
            c_layout,
            c_dtype,
            acc_dtype,
            use_2cta_instrs,
            tmem_warp_shape_mn: None,
            epilog_sync_bar_id: 2,
            epilogue_warp_id: vec![0, 1, 2, 3],
            num_c_stage: 1,
            iter_acc_early_release_in_epilogue: 0,
        }
    }

    fn cta_tile(&self) -> [usize; 3] {
        let (m, n, k) = self.cta_tile_shape_mnk;
        [m, n, k]
    }

    #[track_caller]
    fn check_types<Acc: Numeric, C: Numeric>(&self, what: &str) {
        assert!(
            Acc::DTYPE == self.acc_dtype && C::DTYPE == self.c_dtype,
            "cute: {what}: the tensors are {} accumulators and {} C, but the configuration says {} and {}",
            Acc::DTYPE,
            C::DTYPE,
            self.acc_dtype,
            self.c_dtype
        );
    }
}

/// Regroup an MMA-partitioned tensor's modes from `((MMA_ATOM_M, MMA_ATOM_N),
/// MMA_M, MMA_N, ...rest)` to `((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N),
/// ...rest)` (CuTe DSL's `transform_partitioned_tensor_layout`), so that
/// the tile's M and N are each one mode. Panics if mode 0 is not a pair.
#[track_caller]
pub fn transform_partitioned_tensor_layout<T>(tensor: &Tensor<T>) -> Tensor<T> {
    let l = tensor.layout();
    // The partition of a TMEM accumulator carries a trailing mode of extent 1 after (M, N).
    let atom = l.mode(0);
    let mut atom_modes = atom.modes();
    while atom_modes.len() > 2 && atom_modes.last().is_some_and(|m| m.size() == 1) {
        atom_modes.pop();
    }
    assert!(
        l.rank() >= 3 && atom_modes.len() == 2,
        "cute: transform_partitioned_tensor_layout: the layout {l} is not ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...)"
    );
    let mut modes = vec![
        Layout::from_modes([atom_modes[0].clone(), l.mode(1)]),
        Layout::from_modes([atom_modes[1].clone(), l.mode(2)]),
    ];
    modes.extend((3..l.rank()).map(|i| l.mode(i)));
    Tileable::with_layout(tensor, Layout::from_modes(modes))
}

/// `t[(.., .., front..., idx...)]`: slice `nones` leading modes whole and fix
/// the next ones.
fn slice_after<T>(t: &Tensor<T>, nones: usize, fixed: &[IndexExpr]) -> Tensor<T> {
    let mut coords: Vec<Tree<Option<IndexExpr>>> = vec![Tree::Leaf(None); nones];
    coords.extend(fixed.iter().cloned().map(|i| Tree::Leaf(Some(i))));
    t.slice(Tree::Branch(coords))
}

/// The pieces of the TMEM copy of an epilogue: the tiled copy (CuTe DSL's
/// `tiled_copy_t2r`), the TMEM partition `tTR_tAcc` of the staged
/// accumulator `(T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)` and the register
/// array `tTR_rAcc` of one subtile's values.
pub struct TmemCopyPartition<Acc> {
    pub tiled_copy_t2r: TiledCopy<Acc>,
    pub t_tr_t_acc: Tensor<Acc>,
    pub t_tr_r_acc: Tensor<Acc>,
}

/// The TMEM load of the epilogue and its partitions (CuTe DSL's
/// `epilogue_tmem_copy_and_partition(gemm_kernel, tidx, tAcc, tCgC, epi_tile,
/// use_2cta_instrs)`). `t_acc` is the accumulator `((M), (N), STAGE)` and
/// `t_cg_c` C's `((M), (N), RestM, RestN, ...)` after
/// [`transform_partitioned_tensor_layout`]; the load is
/// [`get_tmem_load_op`]'s, tiled over the four warps of the epilogue by
/// [`make_tmem_copy`]. Thread `tidx` is the thread's index in the epilogue
/// warpgroup.
#[track_caller]
pub fn epilogue_tmem_copy_and_partition<Acc: Numeric, G>(
    cfg: &EpilogueConfig,
    tidx: impl Into<IndexExpr>,
    t_acc: &Tensor<Acc>,
    t_cg_c: &Tensor<G>,
    epi_tile: &Tiler,
    use_2cta_instrs: bool,
) -> TmemCopyPartition<Acc> {
    let atom = get_tmem_load_op::<Acc>(
        &cfg.cta_tile(),
        cfg.c_layout,
        cfg.c_dtype,
        epi_tile,
        use_2cta_instrs,
        cfg.tmem_warp_shape_mn,
    );
    // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
    let t_acc_epi = flat_divide(t_acc, epi_tile.clone());
    // (EPI_TILE_M, EPI_TILE_N)
    let tiled_copy_t2r = make_tmem_copy(&atom, &slice_after(&t_acc_epi, 2, &[0.into(), 0.into(), 0.into()]));
    let thr_copy_t2r = tiled_copy_t2r.get_slice(tidx);
    // (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
    let t_tr_t_acc = thr_copy_t2r.partition_s(&t_acc_epi);
    // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    let t_cg_c_epi = flat_divide(t_cg_c, epi_tile.clone());
    // (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    let t_tr_g_c = thr_copy_t2r.partition_d(&t_cg_c_epi);
    let rest = t_tr_g_c.rank() - 3;
    let one_subtile = slice_after(&t_tr_g_c, 3, &vec![IndexExpr::from(0); rest]);
    // (T2R, T2R_M, T2R_N)
    let t_tr_r_acc = make_fragment::<Acc>(one_subtile.shape().clone());
    TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc, t_tr_r_acc }
}

/// The shared-memory store of the epilogue and its partitions (CuTe DSL's
/// `epilogue_smem_copy_and_partition(gemm_kernel, tiled_copy_t2r, tTR_rC,
/// tidx, sC)`): the store [`get_smem_store_op`] picks, tiled like the TMEM
/// load's registers (`make_tiled_copy_D`), `(R2S, R2S_M, R2S_N)` registers
/// (the C array retiled) and `(R2S, R2S_M, R2S_N, PIPE_D)` shared memory.
pub struct SmemCopyPartition<C> {
    pub tiled_copy_r2s: TiledCopy<C>,
    pub t_rs_r_c: Tensor<C>,
    pub t_rs_s_c: Tensor<C>,
}

/// See [`SmemCopyPartition`].
#[track_caller]
pub fn epilogue_smem_copy_and_partition<Acc: Numeric, C: Numeric>(
    cfg: &EpilogueConfig,
    tiled_copy_t2r: &TiledCopy<Acc>,
    t_tr_r_c: &Tensor<C>,
    tidx: impl Into<IndexExpr>,
    s_c: &Tensor<C>,
) -> SmemCopyPartition<C> {
    let atom = get_smem_store_op::<C, Acc>(cfg.c_layout, cfg.acc_dtype, tiled_copy_t2r);
    let tiled_copy_r2s = make_tiled_copy_d(&atom, tiled_copy_t2r);
    // (R2S, R2S_M, R2S_N, PIPE_D)
    let thr_copy_r2s = tiled_copy_r2s.get_slice(tidx);
    let t_rs_s_c = thr_copy_r2s.partition_d(s_c);
    // (R2S, R2S_M, R2S_N)
    let t_rs_r_c = tiled_copy_r2s.retile(t_tr_r_c);
    SmemCopyPartition { tiled_copy_r2s, t_rs_r_c, t_rs_s_c }
}

/// Drain a TMEM accumulator through shared memory to global memory with TMA
/// stores (CuTe DSL's `cutlass.utils.gemm.sm100.epilogue_tma_store`).
///
/// `t_ct_acc_base` is the accumulator `((MMA_ATOM_M, MMA_ATOM_N), MMA_M,
/// MMA_N, STAGE)` in TMEM, `t_cg_c_base` C's TMA tensor partitioned for the
/// MMA (`((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, RestM, RestN, ...)`),
/// `s_c` the staged shared-memory tile ([`make_smem_layout_epi`]) and
/// `tma_atom_c` the TMA store atom of it. For each subtile of the stage the
/// accumulator `acc_consumer_state` names (after
/// `acc_pipeline.consumer_wait`):
///
/// 1. `tcgen05.ld` (the load of [`get_tmem_load_op`]) and a wait for it;
/// 2. the conversion to C's type and `epilogue_op` (a closure over the
///    [`TensorSsa`] of the subtile's registers);
/// 3. the store to shared memory stage `(num_tiles_executed * subtiles +
///    subtile) % num_c_stage` ([`get_smem_store_op`]);
/// 4. `fence.proxy.async.shared::cta` and the epilogue warps' named barrier
///    (`epilog_sync_bar_id`, 32 threads per warp), then the first epilogue
///    warp issues the TMA store, commits it and waits (`c_pipeline`)
///    until the next stage's earlier store has read its shared memory, and
///    the barrier again.
///
/// Last, one thread of each warp releases the accumulator stage and the
/// state advances: the returned state is the next. `mma_tile_coord_mnl`
/// are the tile's coordinates in C's rest modes. The caller runs
/// `c_pipeline.producer_tail()` (on the issuing warp) before the block exits.
///
/// Differences: the simulator and PTX need the wait for each TMEM load
/// explicit (`fence_view_async_tmem_load`), which this inserts; the
/// consumer arrival of `acc_pipeline` is one per warp (elected), so the
/// pipeline's consumer group counts the epilogue warps.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn epilogue_tma_store<Acc: Numeric, C: Numeric, G: CopyElem<C>, P: Pipeline>(
    cfg: &EpilogueConfig,
    epi_tidx: impl Into<IndexExpr> + Clone,
    warp_idx: Int32,
    tma_atom_c: &CopyAtom<C>,
    t_ct_acc_base: &Tensor<Acc>,
    s_c: &Tensor<C>,
    t_cg_c_base: &Tensor<G>,
    epi_tile: &Tiler,
    num_tiles_executed: Int32,
    epilogue_op: &dyn Fn(TensorSsa<C>) -> TensorSsa<C>,
    mma_tile_coord_mnl: &[Int32],
    acc_consumer_state: &PipelineState,
    acc_pipeline: &P,
    c_pipeline: &PipelineTmaStore,
) -> PipelineState {
    cfg.check_types::<Acc, C>("epilogue_tma_store");
    // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), RestM, RestN, ...)
    let t_cg_c = transform_partitioned_tensor_layout(t_cg_c_base);
    // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    let t_ct_acc = transform_partitioned_tensor_layout(t_ct_acc_base);

    let TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc: t_tr_t_acc_base, t_tr_r_acc } =
        epilogue_tmem_copy_and_partition(cfg, epi_tidx.clone(), &t_ct_acc, &t_cg_c, epi_tile, cfg.use_2cta_instrs);

    let t_tr_r_c = make_fragment::<C>(t_tr_r_acc.shape().clone());
    let SmemCopyPartition { tiled_copy_r2s, t_rs_r_c, t_rs_s_c } =
        epilogue_smem_copy_and_partition(cfg, &tiled_copy_t2r, &t_tr_r_c, epi_tidx, s_c);

    // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, ...)
    let t_cg_c_epi = flat_divide(&t_cg_c, epi_tile.clone());
    // ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, ...)
    let (b_sg_s_c, b_sg_g_c_partitioned) = cpasync::tma_partition(
        tma_atom_c,
        0,
        &make_layout(1, 1),
        &s_c.group_modes(0, 2),
        &t_cg_c_epi.group_modes(0, 2),
    );

    let epilog_sync_barrier = NamedBarrier::new(cfg.epilog_sync_bar_id, 32 * cfg.epilogue_warp_id.len());

    // Slice to the MMA tile: ((ATOM_V, REST_V), EPI_M, EPI_N)
    let coords: Vec<IndexExpr> = mma_tile_coord_mnl.iter().map(|&c| c.into()).collect();
    let b_sg_g_c = slice_after(&b_sg_g_c_partitioned, 3, &coords);

    // The accumulator stage of this tile: (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    let t_tr_t_acc = slice_after(&t_tr_t_acc_base, 5, &[acc_consumer_state.index().into()]);

    // Wait for the accumulator buffer to be full.
    acc_pipeline.consumer_wait(acc_consumer_state, None);

    let t_tr_t_acc = t_tr_t_acc.group_modes(3, t_tr_t_acc.rank());
    let b_sg_g_c = b_sg_g_c.group_modes(1, b_sg_g_c.rank());

    // Store the accumulator to global memory in subtiles.
    let subtile_cnt = t_tr_t_acc.layout().mode(3).size();
    let num_prev_subtiles = num_tiles_executed * (subtile_cnt as i32);
    for subtile_idx in 0..subtile_cnt {
        // Load the accumulator from tensor memory to registers.
        let t_tr_t_acc_mn = slice_after(&t_tr_t_acc, 3, &[IndexExpr::from(subtile_idx)]);
        copy(&tiled_copy_t2r, &t_tr_t_acc_mn, &t_tr_r_acc);
        crate::fence_view_async_tmem_load();

        // Convert to C's type and apply the elementwise operation.
        let acc_vec = tiled_copy_r2s.retile(&t_tr_r_acc).load_ssa();
        let acc_vec = epilogue_op(acc_vec.to::<C>());
        t_rs_r_c.store_ssa(&acc_vec);

        // Store C to shared memory.
        let c_buffer = (num_prev_subtiles + subtile_idx as i32) % (cfg.num_c_stage as i32);
        copy(&tiled_copy_r2s, &t_rs_r_c, &slice_after(&t_rs_s_c, 3, &[c_buffer.into()]));
        // The shared memory stores must be visible to the TMA store.
        fence_proxy_with(
            FenceProxyKind::AsyncShared,
            FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
        );
        epilog_sync_barrier.arrive_and_wait();

        // TMA store C to global memory.
        let first_warp = cfg.epilogue_warp_id[0] as i32;
        if_(warp_idx.eq(first_warp), || {
            copy(
                tma_atom_c,
                &slice_after(&b_sg_s_c, 1, &[c_buffer.into()]),
                &slice_after(&b_sg_g_c, 1, &[IndexExpr::from(subtile_idx)]),
            );
            c_pipeline.producer_commit();
            c_pipeline.producer_acquire();
        });
        epilog_sync_barrier.arrive_and_wait();
    }
    epilog_sync_barrier.arrive_and_wait();

    // Arrive: the accumulator buffer is empty.
    if_(crate::elect_one(), || acc_pipeline.consumer_release(acc_consumer_state));
    let mut next = *acc_consumer_state;
    next.advance();
    next
}

/// Drain a TMEM accumulator straight to global memory (CuTe DSL's
/// `cutlass.utils.gemm.sm100.epilogue`, for kernels without TMA stores).
///
/// The arguments are those of [`epilogue_tma_store`] without the shared
/// memory, TMA and store pipeline. Each thread keeps its values of a subtile
/// in registers, converts them to C's type, applies `epilogue_op` and stores
/// them with `st.global` vectors (the widest of at most 128 bits that the
/// pointer's alignment and the registers' and C's common contiguous run
/// allow). With `t_cc_c_base` (C's coordinate tensor, partitioned like
/// `t_cg_c_base`) and `m_c_mnl` (C, whose shape bounds it), the stores are
/// predicated: a vector is stored when its first element's coordinate is
/// inside C's shape, so the shape's extent along the contiguous mode must be
/// a whole number of vectors. `overlapping_accum` reads the accumulator
/// stage from the state's phase, takes the subtiles in reverse on phase 0,
/// and releases the stage early, after subtile
/// `cfg.iter_acc_early_release_in_epilogue` (and a TMEM-load fence), instead of
/// at the last.
///
/// The stage is released by one thread of each warp, as in
/// [`epilogue_tma_store`]. Returns the next accumulator state.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn epilogue<Acc: Numeric, C: Numeric, P: Pipeline>(
    cfg: &EpilogueConfig,
    epi_tidx: impl Into<IndexExpr> + Clone,
    t_ct_acc_base: &Tensor<Acc>,
    t_cg_c_base: &Tensor<C>,
    epi_tile: &Tiler,
    epilogue_op: &dyn Fn(TensorSsa<C>) -> TensorSsa<C>,
    mma_tile_coord_mnl: &[Int32],
    acc_consumer_state: &PipelineState,
    acc_pipeline: &P,
    t_cc_c_base: Option<&Tensor<Ident>>,
    m_c_mnl: Option<&Tensor<C>>,
    overlapping_accum: bool,
) -> PipelineState {
    cfg.check_types::<Acc, C>("epilogue");
    // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), RestM, RestN, ...)
    let t_cg_c = transform_partitioned_tensor_layout(t_cg_c_base);
    // ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    let t_ct_acc = transform_partitioned_tensor_layout(t_ct_acc_base);

    let TmemCopyPartition { tiled_copy_t2r, t_tr_t_acc: t_tr_t_acc_base, t_tr_r_acc } =
        epilogue_tmem_copy_and_partition(cfg, epi_tidx.clone(), &t_ct_acc, &t_cg_c, epi_tile, cfg.use_2cta_instrs);

    let g_c_epi = flat_divide(&t_cg_c, epi_tile.clone());
    // (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, ...)
    let thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx);
    let t_tr_g_c_partitioned = thr_copy_t2r.partition_d(&g_c_epi);
    let rest = t_tr_g_c_partitioned.rank() - 3;
    // (T2R, T2R_M, T2R_N)
    let first_subtile = slice_after(&t_tr_g_c_partitioned, 3, &vec![IndexExpr::from(0); rest]);
    let t_tr_r_c = make_fragment::<C>(first_subtile.shape().clone());

    // The widest vector store: the pointer's alignment, the contiguous run the registers and C
    // share, and the 128 bits an R2G atom takes.
    let mcl_d = crate::layout::max_common_layout(&t_tr_r_c, &first_subtile);
    let alignment_bits = t_tr_g_c_partitioned.ptr().max_alignment() * 8;
    let c_width = C::DTYPE.bits();
    let mut num_bits = alignment_bits.min(mcl_d.size() * c_width).min(128).max(c_width);
    while !num_bits.is_power_of_two() || !num_bits.is_multiple_of(c_width) {
        num_bits -= 1;
    }
    let vec_len = num_bits / c_width;
    let simt_atom = make_copy_atom::<C>(CopyOp::R2G {
        bits: num_bits,
        attrs: R2GAttrs { l1c_evict_priority: CacheEvictionPriority::NoAllocate, ..Default::default() },
    });
    let use_predication = t_cc_c_base.is_some() && m_c_mnl.is_some();

    // The coordinates of C, laid out as the stores.
    let t_tr_c_c = use_predication.then(|| {
        let t_cc_c = transform_partitioned_tensor_layout(t_cc_c_base.expect("coordinates"));
        let c_epi = flat_divide(&t_cc_c, epi_tile.clone());
        let part = thr_copy_t2r.partition_d(&c_epi);
        let coords: Vec<Option<IndexExpr>> = (0..part.rank())
            .map(|i| i.checked_sub(5).and_then(|j| mma_tile_coord_mnl.get(j)).map(|&c| c.into()))
            .collect();
        let sliced = part.slice(Tree::Branch(coords.into_iter().map(Tree::Leaf).collect()));
        sliced.group_modes(3, sliced.rank())
    });

    // The accumulator stage index.
    let stage = if overlapping_accum { acc_consumer_state.phase() } else { acc_consumer_state.index() };
    // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    let t_tr_t_acc = slice_after(&t_tr_t_acc_base, 5, &[stage.into()]);
    // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    let coords: Vec<Option<IndexExpr>> = (0..t_tr_g_c_partitioned.rank())
        .map(|i| i.checked_sub(5).and_then(|j| mma_tile_coord_mnl.get(j)).map(|&c| c.into()))
        .collect();
    let t_tr_g_c = t_tr_g_c_partitioned.slice(Tree::Branch(coords.into_iter().map(Tree::Leaf).collect()));

    // Wait for the accumulator buffer to be full.
    acc_pipeline.consumer_wait(acc_consumer_state, None);

    let t_tr_t_acc = t_tr_t_acc.group_modes(3, t_tr_t_acc.rank());
    let t_tr_g_c = t_tr_g_c.group_modes(3, t_tr_g_c.rank());

    let mut state = *acc_consumer_state;
    let subtile_cnt = t_tr_t_acc.layout().mode(3).size();
    for subtile_idx in 0..subtile_cnt {
        // Overlapping accumulators on phase 0 are drained backwards: the columns
        // the next tile reuses come first.
        let reversed = if overlapping_accum { Some(acc_consumer_state.phase().eq(0)) } else { None };
        let real_subtile = |idx: usize| -> IndexExpr {
            match reversed {
                None => IndexExpr::from(idx),
                Some(rev) => {
                    let r = (subtile_cnt - 1 - idx) as i32;
                    let f = idx as i32;
                    crate::select(rev, Int32::from(r), Int32::from(f)).into()
                }
            }
        };
        let real = real_subtile(subtile_idx);
        // The destination for this subtile.
        let t_tr_g_c_subtile = slice_after(&t_tr_g_c, 3, std::slice::from_ref(&real));
        // Load the accumulator from tensor memory to registers.
        copy(&tiled_copy_t2r, &slice_after(&t_tr_t_acc, 3, std::slice::from_ref(&real)), &t_tr_r_acc);

        // Arrive: the accumulator buffer is empty.
        let release = if overlapping_accum {
            subtile_idx == cfg.iter_acc_early_release_in_epilogue
        } else {
            subtile_idx == subtile_cnt - 1
        };
        crate::fence_view_async_tmem_load();
        if release {
            if_(crate::elect_one(), || acc_pipeline.consumer_release(acc_consumer_state));
            state.advance();
        }

        // Convert to C's type and apply the elementwise operation.
        let acc_vec = epilogue_op(t_tr_r_acc.load_ssa().to::<C>());
        t_tr_r_c.store_ssa(&acc_vec);

        if let (Some(t_tr_c_c), Some(m_c)) = (&t_tr_c_c, m_c_mnl) {
            // One predicate per vector: its first element's coordinate is inside C.
            let c_subtile = slice_after(t_tr_c_c, 3, &[real]);
            let chunks = t_tr_r_c.layout().mode(0).size() / vec_len;
            let rest_shape: Vec<usize> = (1..c_subtile.rank()).map(|i| c_subtile.layout().mode(i).size()).collect();
            let pred_shape: Shape =
                Tree::tuple(std::iter::once(chunks).chain(rest_shape.iter().copied()).map(Shape::from));
            let pred = make_fragment::<bool>(pred_shape);
            let shape = m_c.shape().clone();
            for m_idx in 0..rest_shape.first().copied().unwrap_or(1) {
                for n_idx in 0..rest_shape.get(1).copied().unwrap_or(1) {
                    for c in 0..chunks {
                        let crd = c_subtile.coord((c * vec_len, m_idx, n_idx));
                        pred.store((c, m_idx, n_idx), elem_less(crd, shape.clone()));
                    }
                }
            }
            copy_if(&simt_atom, &pred, &t_tr_r_c, &t_tr_g_c_subtile);
        } else {
            copy(&simt_atom, &t_tr_r_c, &t_tr_g_c_subtile);
        }
    }
    state
}

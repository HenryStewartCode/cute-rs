// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/copy_sm90_desc.hpp (the L2 cache-policy values of
//   TMA::CacheHintSm90),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! CuTe DSL's `cute.nvgpu.cpasync`: asynchronous copies. So far the bulk
//! copy from shared to global memory, [`CopyBulkS2GOp`], and
//! [`group_bulk_copy_modes`], and the TMA tile operations
//! ([`CopyBulkTensorTileG2SOp`], [`CopyBulkTensorTileG2SMulticastOp`],
//! [`CopyBulkTensorTileS2GOp`], [`CopyReduceBulkTensorTileS2GOp`]) with the
//! construction of their atoms ([`make_tiled_tma_atom`]) and the
//! partitioning of tiles for them ([`tma_partition`]). [`copy`](crate::copy)
//! issues TMA loads ([`CopyBulkTensorTileG2SOp`], with its mbarrier given as
//! [`CopyArgs::tma_bar_ptr`](crate::CopyArgs::tma_bar_ptr)), multicast loads
//! ([`CopyBulkTensorTileG2SMulticastOp`], with
//! [`CopyArgs::mcast_mask`](crate::CopyArgs::mcast_mask) from
//! [`create_tma_multicast_mask`]), TMA stores and reduce stores, and the
//! non-tensor bulk copies ([`CopyBulkG2SOp`], [`CopyBulkG2SMulticastOp`],
//! [`CopyBulkS2SOp`]). [`prefetch_descriptor`] prefetches a TMA atom's
//! descriptor, and [`CacheHintSm90`] has the L2 cache policies the copies'
//! `cache_policy` takes.

use crate::atom::copy::CopyOp;
use crate::tensor::Tensor;

/// CuTe DSL's `cpasync.CopyBulkS2GOp()`: a bulk copy from shared to global
/// memory (`cp.async.bulk.global.shared::cta.bulk_group`, SM90 and newer).
/// `make_copy_atom::<T>(CopyBulkS2GOp)` copies the whole of mode 0 with
/// each instruction (CuTe DSL's `num_bits_per_copy=0`);
/// [`CopyBulkS2GOp::num_bits_per_copy`] gives a fixed size. It is not
/// elected: run [`copy`](crate::copy) on one thread, then commit and wait
/// for the thread's bulk async-group
/// ([`cp_async_bulk_commit_group`](crate::cp_async_bulk_commit_group),
/// [`cp_async_bulk_wait_group`](crate::cp_async_bulk_wait_group)). Shared
/// memory written by ordinary stores needs
/// [`fence_view_async_shared`](crate::fence_view_async_shared) before the
/// copy reads it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkS2GOp;

impl CopyBulkS2GOp {
    /// The operation moving `bits` bits (a multiple of 128) per
    /// instruction: `make_copy_atom(CopyBulkS2GOp(), dtype,
    /// num_bits_per_copy=bits)`.
    pub fn num_bits_per_copy(self, bits: usize) -> CopyOp {
        CopyOp::BulkS2G { bits }
    }
}

impl From<CopyBulkS2GOp> for CopyOp {
    fn from(_: CopyBulkS2GOp) -> CopyOp {
        CopyOp::BulkS2G { bits: 0 }
    }
}

/// The source and destination with all their modes grouped into mode 0
/// (CuTe DSL's `cpasync.group_bulk_copy_modes(src, dst)`), so that one
/// bulk copy moves each whole tensor.
pub fn group_bulk_copy_modes<S, D>(src: &Tensor<S>, dst: &Tensor<D>) -> (Tensor<S>, Tensor<D>) {
    (src.group_modes(0, src.rank()), dst.group_modes(0, dst.rank()))
}

// TMA (bulk tensor) copy atoms (roadmap part m7-tma-atom).

pub use super::tma_atom::{
    TmaAtomInfo, TmaAtomOptions, TmaInfo, make_tiled_tma_atom, make_tiled_tma_atom_with, tma_partition,
    tma_partition_indexed,
};

/// How many CTAs one TMA instruction serves (CuTe DSL's `CtaGroup`, the
/// `.cta_group` qualifier): `ONE`, the CTA that issues it, or `TWO`, the
/// CTA pair of SM100's 2-CTA MMAs.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CtaGroup {
    #[default]
    ONE = 1,
    TWO = 2,
}

/// The reduction a TMA reduce store applies to global memory (CuTe DSL's
/// `ReductionKind`, PTX's `.redOp` of `cp.reduce.async.bulk.tensor`).
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ReductionKind {
    #[default]
    ADD,
    MIN,
    MAX,
    INC,
    DEC,
    AND,
    OR,
    XOR,
}

impl ReductionKind {
    /// PTX's name of the operation (`add`, `min`, ...).
    pub fn ptx_name(self) -> &'static str {
        match self {
            ReductionKind::ADD => "add",
            ReductionKind::MIN => "min",
            ReductionKind::MAX => "max",
            ReductionKind::INC => "inc",
            ReductionKind::DEC => "dec",
            ReductionKind::AND => "and",
            ReductionKind::OR => "or",
            ReductionKind::XOR => "xor",
        }
    }
}

/// CuTe DSL's `cpasync.CopyBulkTensorTileG2SOp(cta_group)`: a TMA load of a
/// tile from global to shared memory (`cp.async.bulk.tensor` in `.tile`
/// mode, SM90 and newer). Atoms of it come from [`make_tiled_tma_atom`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileG2SOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)`: a TMA
/// load whose tile is shared among the CTAs of a cluster, each loading its
/// share and multicasting it to the others.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileG2SMulticastOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensorTileS2GOp()`: a TMA store of a tile
/// from shared to global memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileS2GOp;

/// CuTe DSL's `cpasync.CopyReduceBulkTensorTileS2GOp(reduction_kind)`: a
/// TMA store that combines the tile with global memory by
/// `reduction_kind`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyReduceBulkTensorTileS2GOp {
    pub reduction_kind: ReductionKind,
}

impl From<CopyBulkTensorTileG2SOp> for CopyOp {
    fn from(op: CopyBulkTensorTileG2SOp) -> CopyOp {
        CopyOp::TmaLoad { multicast: false, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensorTileG2SMulticastOp> for CopyOp {
    fn from(op: CopyBulkTensorTileG2SMulticastOp) -> CopyOp {
        CopyOp::TmaLoad { multicast: true, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensorTileS2GOp> for CopyOp {
    fn from(_: CopyBulkTensorTileS2GOp) -> CopyOp {
        CopyOp::TmaStore
    }
}

impl From<CopyReduceBulkTensorTileS2GOp> for CopyOp {
    fn from(op: CopyReduceBulkTensorTileS2GOp) -> CopyOp {
        CopyOp::TmaReduce(op.reduction_kind)
    }
}

// ---- m7-tma-store and m7-cluster-copies ----

pub use super::tma_copy::prefetch_descriptor;
// In-kernel descriptor copies and updates (m7-tensormap-update).
pub use super::tensormap::{
    TensormapField, copy_tensormap, cp_fence_tma_desc_release, fence_tma_desc_acquire, fence_tma_desc_release,
    tensormap_replace, update_tma_descriptor,
};
pub use crate::layout::image_mask::create_tma_multicast_mask;

/// The L2 cache policies C++ CuTe passes as a TMA copy's cache hint
/// (`TMA::CacheHintSm90` in `include/cute/arch/copy_sm90_desc.hpp`, the
/// 64-bit values `createpolicy.fractional.L2::evict_*` gives for a fraction
/// of 1.0): the `cache_policy` of
/// [`CopyArgs`](crate::CopyArgs::cache_policy). The value is opaque to the
/// copy (`.L2::cache_hint`); CuTe DSL has no `createpolicy` wrapper.
#[derive(Clone, Copy, Debug)]
pub struct CacheHintSm90;

impl CacheHintSm90 {
    /// `EVICT_NORMAL`.
    pub const EVICT_NORMAL: i64 = 0x1000_0000_0000_0000;
    /// `EVICT_FIRST`.
    pub const EVICT_FIRST: i64 = 0x12F0_0000_0000_0000;
    /// `EVICT_LAST`.
    pub const EVICT_LAST: i64 = 0x14F0_0000_0000_0000;
}

/// CuTe DSL's `cpasync.CopyBulkG2SOp()`: a bulk copy from global to shared
/// memory (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`,
/// SM90), its bytes counting down the mbarrier given as
/// [`CopyArgs::mbar_ptr`](crate::CopyArgs::mbar_ptr), with an optional
/// `cache_policy`. `make_copy_atom::<T>(CopyBulkG2SOp)` moves the whole of
/// mode 0 per instruction; [`CopyBulkG2SOp::num_bits_per_copy`] a fixed
/// size. It is not elected: one thread issues it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkG2SOp;

impl CopyBulkG2SOp {
    /// The operation moving `bits` bits (a multiple of 128) per instruction.
    pub fn num_bits_per_copy(self, bits: usize) -> CopyOp {
        CopyOp::BulkG2S { bits, multicast: false }
    }
}

impl From<CopyBulkG2SOp> for CopyOp {
    fn from(_: CopyBulkG2SOp) -> CopyOp {
        CopyOp::BulkG2S { bits: 0, multicast: false }
    }
}

/// CuTe DSL's `cpasync.CopyBulkG2SMulticastOp()`: [`CopyBulkG2SOp`] with
/// `.multicast::cluster`: the data lands at the same offset in every CTA of
/// [`CopyArgs::mcast_mask`](crate::CopyArgs::mcast_mask), counting down the
/// mbarrier at the same offset in each. Not elected.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkG2SMulticastOp;

impl CopyBulkG2SMulticastOp {
    /// The operation moving `bits` bits (a multiple of 128) per instruction.
    pub fn num_bits_per_copy(self, bits: usize) -> CopyOp {
        CopyOp::BulkG2S { bits, multicast: true }
    }
}

impl From<CopyBulkG2SMulticastOp> for CopyOp {
    fn from(_: CopyBulkG2SMulticastOp) -> CopyOp {
        CopyOp::BulkG2S { bits: 0, multicast: true }
    }
}

/// CuTe DSL's `cpasync.CopyBulkS2SOp()`: a bulk copy from this CTA's shared
/// memory to the same offset in CTA
/// [`CopyArgs::cta_rank`](crate::CopyArgs::cta_rank) of the cluster,
/// counting down that CTA's mbarrier at the offset of
/// [`CopyArgs::mbar_ptr`](crate::CopyArgs::mbar_ptr)
/// (`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`,
/// SM90). Not elected.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkS2SOp;

impl CopyBulkS2SOp {
    /// The operation moving `bits` bits (a multiple of 128) per instruction.
    pub fn num_bits_per_copy(self, bits: usize) -> CopyOp {
        CopyOp::BulkS2S { bits }
    }
}

impl From<CopyBulkS2SOp> for CopyOp {
    fn from(_: CopyBulkS2SOp) -> CopyOp {
        CopyOp::BulkS2S { bits: 0 }
    }
}

// ---- m7-tma-im2col and m7-tma-sm100-modes ----

pub use super::tma_im2col::{Im2colOptions, make_im2col_tma_atom};
pub use super::tma_sm100::{prefetch_atom_aux, tma_partition_override};

/// The access mode of a TMA operation beyond the tiled one (part of
/// [`CopyOp::TmaLoadMode`] and [`CopyOp::TmaStoreMode`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmaOpMode {
    /// `.im2col` loads and `.im2col_no_offs` stores (roadmap part m7-tma-im2col).
    Im2col,
    /// `.tile::gather4` loads and `.tile::scatter4` stores (roadmap part m7-tma-sm100-modes).
    Rows4,
    /// `.tile` with `.override::global_address` and `.override::global_dim_stride`
    /// (roadmap part m7-tma-sm100-modes).
    Override { no_fully_oob_tile: bool },
}

/// CuTe DSL's `cpasync.CopyBulkTensorIm2ColG2SOp(cta_group)`: a TMA load in im2col
/// mode (`cp.async.bulk.tensor.{3,4,5}d.shared::cluster.global.im2col`, sm_90), the
/// activation loads of an implicit-GEMM convolution. Atoms of it come from
/// [`make_im2col_tma_atom`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorIm2ColG2SOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group)`: the multicast
/// form of [`CopyBulkTensorIm2ColG2SOp`], each CTA of the mask loading its share of the
/// pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorIm2ColG2SMulticastOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensorIm2ColS2GOp()`: a TMA store through an im2col
/// descriptor without offsets
/// (`cp.async.bulk.tensor.{3,4,5}d.global.shared::cta.im2col_no_offs.bulk_group`, sm_90):
/// a convolution's output tile of pixels and channels. Atoms of it come from
/// [`make_im2col_tma_atom`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorIm2ColS2GOp;

/// CuTe DSL's `cpasync.CopyBulkTensor2DGather4G2SOp(cta_group)`: a TMA load of four
/// rows of a 2-D tensor whose indices come from an index tensor
/// (`cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4`, sm_100a). Atoms of it
/// come from [`make_tiled_tma_atom_with`] with
/// [`TmaAtomOptions::gmem_coord_tensor`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensor2DGather4G2SOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensor2DGather4G2SMulticastOp(cta_group)`: the
/// multicast form of [`CopyBulkTensor2DGather4G2SOp`] (cute-rs: each instruction's four
/// rows land in every CTA of the mask; the box is not split among them, so
/// `num_multicast` is 1).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensor2DGather4G2SMulticastOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensor2DScatter4S2GOp()`: a TMA store of four rows to the
/// rows an index tensor gives (`cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4`,
/// sm_100a and sm_110a).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensor2DScatter4S2GOp;

/// CuTe DSL's `cpasync.CopyBulkTensorTileG2SOverrideOp(cta_group)`: a TMA tile load
/// whose global address, extents and strides are given when it is issued
/// (`.override::global_address.override::global_dim_stride`, the sm_107f family, PTX
/// 9.4). cute-rs makes its atoms with [`make_tiled_tma_atom`] and partitions with
/// [`tma_partition_override`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileG2SOverrideOp {
    pub cta_group: CtaGroup,
}

/// CuTe DSL's `cpasync.CopyBulkTensorTileG2SMulticastOverrideOp(cta_group,
/// no_fully_oob_tile)`: the multicast form of [`CopyBulkTensorTileG2SOverrideOp`].
/// `no_fully_oob_tile` promises that no box is wholly outside the tensor (recorded; the
/// copy is the same).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileG2SMulticastOverrideOp {
    pub cta_group: CtaGroup,
    pub no_fully_oob_tile: bool,
}

/// CuTe DSL's `cpasync.CopyBulkTensorTileS2GOverrideOp(no_fully_oob_tile)`: a TMA tile
/// store with overrides.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkTensorTileS2GOverrideOp {
    pub no_fully_oob_tile: bool,
}

/// CuTe DSL's `cpasync.CopyBulkS2GByteMaskOp()`: a bulk copy from shared to global
/// memory writing only the bytes of each 16-byte chunk that the copy's `byte_mask`
/// selects (bit `i` for byte `i`; `cp.async.bulk.global.shared::cta.bulk_group.cp_mask`,
/// sm_100 and newer). As [`CopyBulkS2GOp`], one thread issues it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CopyBulkS2GByteMaskOp;

impl CopyBulkS2GByteMaskOp {
    /// The operation moving `bits` bits (a multiple of 128) per instruction.
    pub fn num_bits_per_copy(self, bits: usize) -> CopyOp {
        CopyOp::BulkS2GByteMask { bits }
    }
}

impl From<CopyBulkS2GByteMaskOp> for CopyOp {
    fn from(_: CopyBulkS2GByteMaskOp) -> CopyOp {
        CopyOp::BulkS2GByteMask { bits: 0 }
    }
}

impl From<CopyBulkTensorIm2ColG2SOp> for CopyOp {
    fn from(op: CopyBulkTensorIm2ColG2SOp) -> CopyOp {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Im2col, multicast: false, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensorIm2ColG2SMulticastOp> for CopyOp {
    fn from(op: CopyBulkTensorIm2ColG2SMulticastOp) -> CopyOp {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Im2col, multicast: true, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensorIm2ColS2GOp> for CopyOp {
    fn from(_: CopyBulkTensorIm2ColS2GOp) -> CopyOp {
        CopyOp::TmaStoreMode { mode: TmaOpMode::Im2col }
    }
}

impl From<CopyBulkTensor2DGather4G2SOp> for CopyOp {
    fn from(op: CopyBulkTensor2DGather4G2SOp) -> CopyOp {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Rows4, multicast: false, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensor2DGather4G2SMulticastOp> for CopyOp {
    fn from(op: CopyBulkTensor2DGather4G2SMulticastOp) -> CopyOp {
        CopyOp::TmaLoadMode { mode: TmaOpMode::Rows4, multicast: true, cta_group: op.cta_group }
    }
}

impl From<CopyBulkTensor2DScatter4S2GOp> for CopyOp {
    fn from(_: CopyBulkTensor2DScatter4S2GOp) -> CopyOp {
        CopyOp::TmaStoreMode { mode: TmaOpMode::Rows4 }
    }
}

impl From<CopyBulkTensorTileG2SOverrideOp> for CopyOp {
    fn from(op: CopyBulkTensorTileG2SOverrideOp) -> CopyOp {
        CopyOp::TmaLoadMode {
            mode: TmaOpMode::Override { no_fully_oob_tile: false },
            multicast: false,
            cta_group: op.cta_group,
        }
    }
}

impl From<CopyBulkTensorTileG2SMulticastOverrideOp> for CopyOp {
    fn from(op: CopyBulkTensorTileG2SMulticastOverrideOp) -> CopyOp {
        CopyOp::TmaLoadMode {
            mode: TmaOpMode::Override { no_fully_oob_tile: op.no_fully_oob_tile },
            multicast: true,
            cta_group: op.cta_group,
        }
    }
}

impl From<CopyBulkTensorTileS2GOverrideOp> for CopyOp {
    fn from(op: CopyBulkTensorTileS2GOverrideOp) -> CopyOp {
        CopyOp::TmaStoreMode { mode: TmaOpMode::Override { no_fully_oob_tile: op.no_fully_oob_tile } }
    }
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/copy_atom.hpp, include/cute/atom/copy_traits.hpp,
//   include/cute/atom/copy_traits_sm75.hpp, include/cute/atom/copy_traits_sm80.hpp,
//   include/cute/atom/copy_traits_sm90.hpp,
//   include/cute/atom/copy_traits_sm100.hpp, include/cute/arch/copy_sm100.hpp and
//   include/cute/layout.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Copy atoms and tiled copies.
//!
//! A [`CopyAtom`] describes one copy instruction as CuTe does: which threads
//! take part (`thr_id`) and which elements each thread reads (`src`) and
//! receives (`dst`), as `(thread, value)` layouts over the atom's reference
//! element space. A [`TiledCopy`] tiles an atom over a thread-value layout so
//! that a whole tile is copied by a group of threads; [`ThrCopy`] is one
//! thread's view, which partitions source and destination tensors.

use std::panic::Location;

use crate::arch::SharedSpace;
use crate::atom::mma::TiledMma;
use crate::builder;
use crate::control::{for_range, if_, if_else_yield};
use crate::ir::{
    BulkCompletion, BulkDirection, CacheEvictionPriority, CacheOperator, DType, L2PrefetchSize, LdMatrixForm, Literal,
    MemAttrs, MemSpace, MemoryOrder, MemoryScope, Op, StMatrixForm, ValueId,
};
use crate::layout::{
    IndexExpr, IntTuple, Layout, SliceCoord, Tileable, Tiler, Tree, logical_divide, make_layout_tv, max_common_layout,
};
use crate::tensor::{AddressSpace, Ident, Ptr, Tensor, load_vector_with, store_vector_with};
use crate::validate::{Access, check_mem_attrs};
use crate::value::{Boolean, Int16, Int32, Int64, IntoValue, Numeric, Value};

/// The instruction a copy atom issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyOp {
    /// Plain loads and stores moving `bits` bits per instruction (a multiple
    /// of the element size, at most 128, or 256 for global memory accesses of
    /// 32- or 64-bit elements from a 32-byte-aligned buffer, PTX's
    /// `ld.global.v8.b32`, which LLVM splits in two below sm_100).
    Universal { bits: usize },
    /// `cp.async` from global to shared memory, 4, 8 or 16 bytes per thread.
    /// `bypass_l1` selects `.cg` (16 bytes only); otherwise `.ca`.
    /// Predicated copies zero-fill the destination.
    CpAsync { bytes: usize, bypass_l1: bool },
    /// `ldmatrix.m8n8.x{num}` (`num` in 1, 2, 4) from shared memory into
    /// registers, optionally transposed. 16-bit elements only.
    LdMatrix { num: usize, trans: bool },
    /// CuTe DSL's `cute.nvgpu.warp.LdMatrix8x16x8bOp(num_matrices,
    /// unpack_bits)`: `num` (1, 2 or 4) matrices of 8 rows of 16 bytes of
    /// 8-bit elements from shared memory into registers, each lane supplying
    /// the address of one row and receiving 4 consecutive bytes of each
    /// matrix. Without unpacking this is `ldmatrix.m8n8.b16` over the bytes
    /// (CuTe: "LdMatrix8x16x8b without unpacking ... is equivalent to
    /// LdMatrix8x8x16b"), the form CuTe's Ampere GEMM uses for K-major FP8
    /// operands. With `unpack_bits` 4 or 6 it is
    /// `ldmatrix.m8n16.x{num}.shared.b8x16.b4x16_p64` or `.b6x16_p32`: each
    /// 16-byte row holds 16 4-bit (6-bit) values packed into its low 8 (12)
    /// bytes, the rest padding (what TMA's `16U4_ALIGN16B` and
    /// `16U6_ALIGN16B` formats write), and each value arrives zero-extended
    /// in an 8-bit container (a 4-bit value in bits 0 to 3, a 6-bit one in
    /// bits 0 to 5). `mma.kind::f8f6f4` takes a 6-bit value there but a
    /// 4-bit one in bits 2 to 5, so CuTe C++ shifts 4-bit operands left by
    /// two before the MMA (`fp4_shift_A`, `fp4_shift_B`); the source tensor
    /// has 16 8-bit elements per row and only its row addresses matter. CuTe C++'s
    /// `SM100_SU4_DU8x16_x{1,2,4}_LDSM_N` and `SM100_SU6_DU8x16_*`; sm_100a or
    /// sm_120a.
    LdMatrix8x16x8b { num: usize, unpack_bits: Option<u32> },
    /// CuTe DSL's `cute.nvgpu.warp.LdMatrix16x16x8bOp(num_matrices,
    /// transpose=True, unpack_bits)`:
    /// `ldmatrix.m16n16.x{num}.trans.shared.b8` (`num` 1 or 2) of 16x16
    /// matrices of 8-bit elements, transposed (`trans` must be true, as the
    /// PTX ISA requires), or with `unpack_bits` 4 or 6 the padded sub-byte
    /// rows of [`CopyOp::LdMatrix8x16x8b`] (`.b8x16.b4x16_p64`,
    /// `.b8x16.b6x16_p32`). Lanes 16 `m` to 16 `m` + 15 supply the 16 row
    /// addresses of matrix `m`; lane `l` receives in its registers `2m` and
    /// `2m + 1` byte `l / 4` and `l / 4 + 8` of rows `4 (l % 4)` to `4 (l %
    /// 4) + 3` (the PTX ISA's fragment of the transposed matrix). sm_100a or
    /// sm_120a.
    LdMatrix16x16x8b { num: usize, trans: bool, unpack_bits: Option<u32> },
    /// CuTe DSL's `cute.nvgpu.warp.LdMatrix16x8x8bOp(num_matrices,
    /// transpose=True, unpack_bits)`: `num` (2 or 4) 16x8 matrices of 8-bit
    /// elements, transposed, laid out like `stmatrix.m16n8.trans` so that
    /// Ampere-style 8x8 thread-value layouts vectorise. PTX has no such
    /// shape: it issues `ldmatrix.m16n16.x{num / 2}.trans` with the address
    /// and value permutations of CuTe C++'s `SM100_U8x8_LDSM_T` and
    /// `SM100_U8x16_LDSM_T`. `trans` must be true; `unpack_bits` as for
    /// [`CopyOp::LdMatrix16x16x8b`]. sm_100a or sm_120a.
    LdMatrix16x8x8b { num: usize, trans: bool, unpack_bits: Option<u32> },
    /// CuTe DSL's `cute.nvgpu.warp.StMatrix8x8x16bOp(transpose,
    /// num_matrices)`: `stmatrix.sync.aligned.m8n8.x{num}[.trans].shared.b16`
    /// (`num` 1, 2 or 4; sm_90, PTX 7.8), the inverse of [`CopyOp::LdMatrix`]:
    /// each lane holds two 16-bit elements of each 8x8 matrix and supplies the
    /// address of one 16-byte row of shared memory. The register-to-shared
    /// atom of Hopper epilogues. Its thread-value layouts are CuTe C++'s
    /// `SM90_U32x{1,2,4}_STSM_N` and `SM90_U16x{2,4,8}_STSM_T` (those of
    /// `ldmatrix` with source and destination swapped). A 32-bit element
    /// type goes one element to a register, untransposed only (Blackwell
    /// epilogues of Float32 D).
    StMatrix { num: usize, trans: bool },
    /// CuTe DSL's `cute.nvgpu.warp.StMatrix16x8x8bOp(transpose=True,
    /// num_matrices)`: `stmatrix.sync.aligned.m16n8.x{num}.trans.shared.b8`
    /// (`num` 1, 2 or 4): each lane holds four 8-bit elements of each 16x8
    /// matrix in one register (the `mma.m16n8` accumulator fragment), stored
    /// transposed as 8 rows of 16 bytes. `trans` must be true. CuTe C++'s
    /// `SM100_U8x{4,8,16}_STSM_T`; sm_100a, sm_110a or sm_120a.
    /// [`CopyOp::LdMatrix16x8x8b`] loads it back.
    StMatrix16x8x8b { num: usize, trans: bool },
    /// Loads from global memory into registers (a fragment or a
    /// `make_rmem_tensor`) carrying memory attributes (CuTe DSL's
    /// `cute.nvgpu.CopyG2ROp`): `bits` bits per load (a multiple of the
    /// element size, at most 128), or 0 for the widest vector that each copy
    /// can prove aligned and contiguous, as [`autovec_copy`] chooses (one
    /// element at a time under a predicate, and for MMIO). The loads are
    /// [`load_vector`](crate::load_vector)s, so an ordered or volatile one of
    /// 8- or 16-bit elements keeps their type.
    G2R { bits: usize, attrs: G2RAttrs },
    /// Stores from registers (a fragment or a `make_rmem_tensor`) to global
    /// memory carrying memory attributes (CuTe DSL's `cute.nvgpu.CopyR2GOp`),
    /// `bits` as for [`CopyOp::G2R`].
    R2G { bits: usize, attrs: R2GAttrs },
    /// A bulk copy from shared to global memory
    /// (`cp.async.bulk.global.shared::cta.bulk_group`, SM90 and newer; CuTe
    /// DSL's `cpasync.CopyBulkS2GOp`, which
    /// [`cpasync::CopyBulkS2GOp`](crate::cpasync::CopyBulkS2GOp) converts
    /// into): one instruction moves `bits` bits (a multiple of 128), or, with
    /// 0, the whole of mode 0 of the tensors
    /// ([`group_bulk_copy_modes`](crate::cpasync::group_bulk_copy_modes)
    /// makes that the whole tensor). Each copy's elements are contiguous in
    /// both tensors, a multiple of 16 bytes, at 16-byte-aligned addresses,
    /// and the source is not swizzled. Every thread that executes the copy
    /// issues it (elect one), and its bulk async-groups track it.
    BulkS2G { bits: usize },
    /// A bulk copy from global to shared memory reporting to an mbarrier
    /// (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`,
    /// SM90; CuTe DSL's `cpasync.CopyBulkG2SOp`, or with `multicast`
    /// `cpasync.CopyBulkG2SMulticastOp`, `.multicast::cluster`): `bits` as
    /// for [`CopyOp::BulkS2G`]. It takes `mbar_ptr` (and with `multicast`
    /// `mcast_mask`) and an optional `cache_policy`. Not elected.
    BulkG2S { bits: usize, multicast: bool },
    /// A bulk copy from this CTA's shared memory to a peer CTA's
    /// (`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`,
    /// SM90; CuTe DSL's `cpasync.CopyBulkS2SOp`): it takes `mbar_ptr` and
    /// `cta_rank`, maps the destination and the mbarrier to that CTA
    /// (`mapa`) and counts its bytes down the peer's mbarrier. Not elected.
    BulkS2S { bits: usize },
    /// Loads from shared memory into registers carrying memory attributes
    /// (CuTe DSL's `cute.nvgpu.CopyS2ROp`): `bits` as
    /// for [`CopyOp::G2R`]. With [`SharedSpace::Cluster`] the source is
    /// another CTA's shared memory (a tensor from a
    /// [`map_dsmem_ptr`](crate::map_dsmem_ptr) pointer: `ld.shared::cluster`,
    /// sm_90), otherwise this CTA's.
    S2R { bits: usize, attrs: S2RAttrs },
    /// Stores from registers into shared memory carrying memory attributes
    /// (CuTe DSL's `cute.nvgpu.CopyR2SOp`), as [`CopyOp::S2R`] loads.
    R2S { bits: usize, attrs: R2SAttrs },
    /// CuTe DSL's `cpasync.CopyDsmemStoreOp`: an
    /// asynchronous store from registers into another CTA's shared memory,
    /// `st.async.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.b32`
    /// (sm_90, PTX 8.1), whose bytes count down that CTA's mbarrier. `bits`
    /// is 32, 64 or 128 per store, or 0 for the widest of those the copy can
    /// prove aligned and contiguous. The destination is a tensor from a
    /// [`map_dsmem_ptr`](crate::map_dsmem_ptr) pointer and the copy takes
    /// `mbar_ptr`, the peer's mbarrier (mapped to the same CTA). Every thread
    /// that executes the copy stores (not elected).
    DsmemStore { bits: usize },
    /// A TMA load of a tile from global to shared memory
    /// ([`cpasync::CopyBulkTensorTileG2SOp`](crate::cpasync::CopyBulkTensorTileG2SOp),
    /// or with `multicast`
    /// [`cpasync::CopyBulkTensorTileG2SMulticastOp`](crate::cpasync::CopyBulkTensorTileG2SMulticastOp)).
    /// Its atoms come from [`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom),
    /// which builds the descriptor; [`copy_with`] issues the plain load with its
    /// mbarrier ([`CopyArgs::tma_bar_ptr`]).
    TmaLoad { multicast: bool, cta_group: crate::cpasync::CtaGroup },
    /// A TMA store of a tile from shared to global memory
    /// ([`cpasync::CopyBulkTensorTileS2GOp`](crate::cpasync::CopyBulkTensorTileS2GOp)).
    TmaStore,
    /// A TMA reduce store
    /// ([`cpasync::CopyReduceBulkTensorTileS2GOp`](crate::cpasync::CopyReduceBulkTensorTileS2GOp)).
    TmaReduce(crate::cpasync::ReductionKind),
    /// A TMA load in another access mode: im2col
    /// ([`cpasync::CopyBulkTensorIm2ColG2SOp`](crate::cpasync::CopyBulkTensorIm2ColG2SOp),
    /// its multicast form), gather4
    /// ([`cpasync::CopyBulkTensor2DGather4G2SOp`](crate::cpasync::CopyBulkTensor2DGather4G2SOp),
    /// its multicast form) or with overrides
    /// ([`cpasync::CopyBulkTensorTileG2SOverrideOp`](crate::cpasync::CopyBulkTensorTileG2SOverrideOp),
    /// its multicast form).
    TmaLoadMode { mode: crate::cpasync::TmaOpMode, multicast: bool, cta_group: crate::cpasync::CtaGroup },
    /// A TMA store in another access mode: im2col without offsets
    /// ([`cpasync::CopyBulkTensorIm2ColS2GOp`](crate::cpasync::CopyBulkTensorIm2ColS2GOp)),
    /// scatter4 ([`cpasync::CopyBulkTensor2DScatter4S2GOp`](crate::cpasync::CopyBulkTensor2DScatter4S2GOp))
    /// or with overrides ([`cpasync::CopyBulkTensorTileS2GOverrideOp`](crate::cpasync::CopyBulkTensorTileS2GOverrideOp)).
    TmaStoreMode { mode: crate::cpasync::TmaOpMode },
    /// CuTe DSL's `cpasync.CopyBulkS2GByteMaskOp`: [`CopyOp::BulkS2G`] writing only the bytes of each
    /// 16-byte chunk that [`CopyArgs::byte_mask`] selects
    /// (`cp.async.bulk.global.shared::cta.bulk_group.cp_mask`, sm_100).
    BulkS2GByteMask { bits: usize },
    /// A TMEM load: `tcgen05.ld` of `shape`,
    /// `.x<num>`, `.pack::16b` with `pack`, from a TMEM tensor into a
    /// register fragment ([`tcgen05::Ld32x32bOp`](crate::nvgpu::tcgen05::Ld32x32bOp)
    /// and its peers). Warp-collective; the registers are undefined until
    /// [`fence_view_async_tmem_load`](crate::fence_view_async_tmem_load).
    TmemLoad { shape: crate::ir::TmemShape, num: u32, pack: bool },
    /// A TMEM load with a row reduction:
    /// `tcgen05.ld.red` of `.16x32bx2` or `.32x32b`, `.x<num>`, loading
    /// Float32 values into a register fragment and reducing each thread's
    /// values with `red` (PTX `min`/`max`, `.abs`, and `.NaN` with `nan`) into
    /// a second register. Issued with [`copy_atom_call`], which takes both
    /// destinations ([`tcgen05::LdRed32x32bOp`](crate::nvgpu::tcgen05::LdRed32x32bOp)
    /// and its peer). PTX ISA 8.8: sm_110a and the sm_103f, sm_110f families.
    /// `half_split_off` is the immediate column offset of `.16x32bx2`'s second
    /// access (0: as many columns as the first spans).
    TmemLoadRed { shape: crate::ir::TmemShape, num: u32, red: crate::ir::TmemLoadRedOp, nan: bool, half_split_off: u32 },
    /// A TMEM load with 2:4 sparse compression: `tcgen05.ld{.red}.spcompress.32x32b.x<num>`
    /// (`with_red` for the `.red` form, which also reduces the thread's
    /// values with `red`), loading Float32 values and keeping two of every
    /// four as `red` prefers (`nan`: a NaN is preferred). Issued with
    /// [`copy_atom_call_sp`], which takes the compressed data, the metadata
    /// and the reduction as destinations
    /// ([`tcgen05::LdSPCompress32x32bOp`](crate::nvgpu::tcgen05::LdSPCompress32x32bOp)
    /// and [`LdRedSPCompress32x32bOp`](crate::nvgpu::tcgen05::LdRedSPCompress32x32bOp)).
    /// PTX ISA 9.4: sm_107a.
    TmemLoadSpCompress { num: u32, red: crate::ir::TmemLoadRedOp, with_red: bool, nan: bool },
    /// A TMEM store: `tcgen05.st` of `shape`, `.x<num>`, `.unpack::16b`
    /// with `unpack`, from registers into a TMEM tensor
    /// ([`tcgen05::St32x32bOp`](crate::nvgpu::tcgen05::St32x32bOp) and its
    /// peers). Warp-collective; in flight until
    /// [`fence_view_async_tmem_store`](crate::fence_view_async_tmem_store).
    TmemStore { shape: crate::ir::TmemShape, num: u32, unpack: bool },
    /// A shared-memory-to-TMEM copy:
    /// `tcgen05.cp` of `shape`, `.cta_group::2` with `two_cta`
    /// ([`tcgen05::Cp128x256bOp`](crate::nvgpu::tcgen05::Cp128x256bOp) and
    /// its peers). The source is a tensor of shared-memory descriptors
    /// ([`tcgen05::get_s2t_smem_desc_tensor`](crate::nvgpu::tcgen05::get_s2t_smem_desc_tensor)),
    /// the destination a TMEM tensor; a converged warp calls the copy and
    /// one elected lane issues it.
    TmemCp { shape: crate::ir::TmemCpShape, two_cta: bool },
}

impl CopyOp {
    /// Whether this is one of the TMA (bulk tensor) operations, whose atoms
    /// [`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom) (or
    /// [`make_im2col_tma_atom`](crate::cpasync::make_im2col_tma_atom)) makes.
    pub fn is_tma(&self) -> bool {
        matches!(
            self,
            CopyOp::TmaLoad { .. }
                | CopyOp::TmaStore
                | CopyOp::TmaReduce(_)
                | CopyOp::TmaLoadMode { .. }
                | CopyOp::TmaStoreMode { .. }
        )
    }

    /// Whether this is one of the TMA loads, whatever the mode.
    pub fn is_tma_load(&self) -> bool {
        matches!(self, CopyOp::TmaLoad { .. } | CopyOp::TmaLoadMode { .. })
    }
}

/// How the loads of a [`CopyOp::G2R`] atom use the caches (CuTe DSL's
/// `cute.nvgpu.LoadCacheMode`). A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum LoadCacheMode {
    /// `ALWAYS` (the default): cache at every level, PTX's default (`.ca`),
    /// so no qualifier is written and an L1 eviction priority may be given.
    #[default]
    Always,
    /// `GLOBAL`: cache in L2, not L1 (`.cg`).
    Global,
    /// `STREAMING`: likely to be read once (`.cs`).
    Streaming,
    /// `LAST_USE`: the last read of the line (`.lu`).
    LastUse,
    /// `NONE`: no cache operator, as `Always`.
    None,
}

impl LoadCacheMode {
    /// The cache operator the loads carry.
    pub fn cop(self) -> Option<CacheOperator> {
        match self {
            LoadCacheMode::Always | LoadCacheMode::None => None,
            LoadCacheMode::Global => Some(CacheOperator::Cg),
            LoadCacheMode::Streaming => Some(CacheOperator::Cs),
            LoadCacheMode::LastUse => Some(CacheOperator::Lu),
        }
    }
}

/// How the stores of a [`CopyOp::R2G`] atom use the caches (CuTe DSL's
/// `cute.nvgpu.StoreCacheMode`). A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StoreCacheMode {
    /// `WRITE_BACK` (the default): PTX's default (`.wb`), so no qualifier is
    /// written.
    #[default]
    WriteBack,
    /// `GLOBAL`: cache in L2, not L1 (`.cg`).
    Global,
    /// `STREAMING`: likely to be accessed once (`.cs`).
    Streaming,
    /// `WRITE_THROUGH`: write through to system memory (`.wt`).
    WriteThrough,
    /// `NONE`: no cache operator, as `WriteBack`.
    None,
}

impl StoreCacheMode {
    /// The cache operator the stores carry.
    pub fn cop(self) -> Option<CacheOperator> {
        match self {
            StoreCacheMode::WriteBack | StoreCacheMode::None => None,
            StoreCacheMode::Global => Some(CacheOperator::Cg),
            StoreCacheMode::Streaming => Some(CacheOperator::Cs),
            StoreCacheMode::WriteThrough => Some(CacheOperator::Wt),
        }
    }
}

/// The memory attributes of a [`CopyOp::G2R`] atom: the keyword arguments
/// of CuTe DSL's `make_copy_atom(CopyG2ROp(), dtype, memory_order=,
/// memory_scope=, l2_prefetch_size=, l1c_evict_priority=, load_cache_mode=,
/// shared_space=, invariant=)`. The default is CuTe DSL's (a weak load with
/// no hints); give the fields that differ, as in `G2RAttrs { memory_order:
/// MemoryOrder::Acquire, memory_scope: MemoryScope::Gpu,
/// ..Default::default() }`. The loads follow the rules of
/// [`load`](crate::load), which the atom checks when it is made.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct G2RAttrs {
    /// `WEAK` by default. `CONSTANT` reads through the non-coherent cache, as
    /// `invariant` does.
    pub memory_order: MemoryOrder,
    /// `CTA` by default.
    pub memory_scope: MemoryScope,
    /// `NONE` (`None`) by default.
    pub l2_prefetch_size: Option<L2PrefetchSize>,
    /// `EVICT_NORMAL` by default, which writes no qualifier.
    pub l1c_evict_priority: CacheEvictionPriority,
    /// `ALWAYS` by default.
    pub load_cache_mode: LoadCacheMode,
    /// `CTA` by default; `CLUSTER` (another CTA's shared memory) is for
    /// [`CopyOp::S2R`] and [`CopyOp::R2S`], not global memory.
    pub shared_space: SharedSpace,
    /// Read through the non-coherent cache (`ld.global.nc`): only for memory
    /// no thread writes while the kernel runs, which the simulator checks.
    pub invariant: bool,
}

impl Default for G2RAttrs {
    fn default() -> Self {
        G2RAttrs {
            memory_order: MemoryOrder::Weak,
            memory_scope: MemoryScope::Cta,
            l2_prefetch_size: None,
            l1c_evict_priority: CacheEvictionPriority::EvictNormal,
            load_cache_mode: LoadCacheMode::Always,
            shared_space: SharedSpace::Cta,
            invariant: false,
        }
    }
}

impl G2RAttrs {
    /// The attributes of the atom's loads.
    pub fn mem_attrs(&self) -> MemAttrs {
        let evict = self.l1c_evict_priority;
        MemAttrs {
            order: self.memory_order,
            scope: self.memory_scope,
            l1_eviction: (evict != CacheEvictionPriority::EvictNormal).then_some(evict),
            cop: self.load_cache_mode.cop(),
            l2_prefetch: self.l2_prefetch_size,
            non_coherent: self.invariant,
            cache_policy: None,
            peer: None,
        }
    }
}

/// The memory attributes of a [`CopyOp::R2G`] atom: the keyword arguments
/// of CuTe DSL's `make_copy_atom(CopyR2GOp(), dtype, memory_order=,
/// memory_scope=, l1c_evict_priority=, store_cache_mode=, shared_space=)`,
/// with its defaults. The stores follow the rules of
/// [`store`](crate::store).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct R2GAttrs {
    /// `WEAK` by default.
    pub memory_order: MemoryOrder,
    /// `CTA` by default.
    pub memory_scope: MemoryScope,
    /// `EVICT_NORMAL` by default, which writes no qualifier.
    pub l1c_evict_priority: CacheEvictionPriority,
    /// `WRITE_BACK` by default.
    pub store_cache_mode: StoreCacheMode,
    /// `CTA` by default; `CLUSTER` (another CTA's shared memory) is for
    /// [`CopyOp::S2R`] and [`CopyOp::R2S`], not global memory.
    pub shared_space: SharedSpace,
}

impl Default for R2GAttrs {
    fn default() -> Self {
        R2GAttrs {
            memory_order: MemoryOrder::Weak,
            memory_scope: MemoryScope::Cta,
            l1c_evict_priority: CacheEvictionPriority::EvictNormal,
            store_cache_mode: StoreCacheMode::WriteBack,
            shared_space: SharedSpace::Cta,
        }
    }
}

impl R2GAttrs {
    /// The attributes of the atom's stores.
    pub fn mem_attrs(&self) -> MemAttrs {
        let evict = self.l1c_evict_priority;
        MemAttrs {
            order: self.memory_order,
            scope: self.memory_scope,
            l1_eviction: (evict != CacheEvictionPriority::EvictNormal).then_some(evict),
            cop: self.store_cache_mode.cop(),
            l2_prefetch: None,
            non_coherent: false,
            cache_policy: None,
            peer: None,
        }
    }
}

/// The memory attributes of a [`CopyOp::S2R`] atom: the keyword arguments
/// of CuTe DSL's `make_copy_atom(CopyS2ROp(), dtype, num_bits_per_copy=,
/// memory_order=, memory_scope=, shared_space=)`, with its defaults (a weak
/// load of this CTA's shared memory). The loads follow the rules of
/// [`load`](crate::load).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct S2RAttrs {
    /// `WEAK` by default.
    pub memory_order: MemoryOrder,
    /// `CTA` by default.
    pub memory_scope: MemoryScope,
    /// `CTA` (this CTA's shared memory) by default; `CLUSTER` reads a
    /// [`map_dsmem_ptr`](crate::map_dsmem_ptr) tensor.
    pub shared_space: SharedSpace,
}

impl Default for S2RAttrs {
    fn default() -> Self {
        S2RAttrs { memory_order: MemoryOrder::Weak, memory_scope: MemoryScope::Cta, shared_space: SharedSpace::Cta }
    }
}

impl S2RAttrs {
    /// The attributes of the atom's loads (the peer is the tensor's).
    pub fn mem_attrs(&self) -> MemAttrs {
        MemAttrs { order: self.memory_order, scope: self.memory_scope, ..MemAttrs::default() }
    }
}

/// The memory attributes of a [`CopyOp::R2S`] atom, as [`S2RAttrs`] for
/// CuTe DSL's `CopyR2SOp`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct R2SAttrs {
    /// `WEAK` by default.
    pub memory_order: MemoryOrder,
    /// `CTA` by default.
    pub memory_scope: MemoryScope,
    /// `CTA` by default; `CLUSTER` writes a
    /// [`map_dsmem_ptr`](crate::map_dsmem_ptr) tensor.
    pub shared_space: SharedSpace,
}

impl Default for R2SAttrs {
    fn default() -> Self {
        R2SAttrs { memory_order: MemoryOrder::Weak, memory_scope: MemoryScope::Cta, shared_space: SharedSpace::Cta }
    }
}

impl R2SAttrs {
    /// The attributes of the atom's stores (the peer is the tensor's).
    pub fn mem_attrs(&self) -> MemAttrs {
        MemAttrs { order: self.memory_order, scope: self.memory_scope, ..MemAttrs::default() }
    }
}

/// One copy instruction's thread/value mapping.
#[derive(Clone, Debug)]
pub struct CopyAtom<T> {
    op: CopyOp,
    thr_id: Layout,
    src: Layout,
    dst: Layout,
    reference: Layout,
    /// The descriptor and box of a TMA atom.
    tma: Option<std::rc::Rc<crate::cpasync::TmaAtomInfo>>,
    /// The runtime fields bound by [`CopyAtom::with_`].
    bound: Option<std::rc::Rc<AtomFields>>,
    _t: std::marker::PhantomData<fn() -> T>,
}

impl<T: Numeric> CopyAtom<T> {
    /// The atom for `op` on elements of type `T`.
    #[track_caller]
    pub fn new(op: CopyOp) -> CopyAtom<T> {
        let bits = T::DTYPE.storage_bits();
        let words = |b: usize| {
            assert!(
                !T::DTYPE.is_word_packed() || b.is_multiple_of(32),
                "cute: a copy of {b} bits of {} values: sub-byte tensors are copied in vectors of a multiple of 32 bits",
                T::DTYPE.cute_name()
            );
        };
        let simple = |vals: usize| {
            let l = Layout::new((1, vals), (0, 1));
            (Layout::new(1, 0), l.clone(), l.clone(), l)
        };
        let (thr_id, src, dst, reference) = match op {
            CopyOp::TmaLoad { .. }
            | CopyOp::TmaStore
            | CopyOp::TmaReduce(_)
            | CopyOp::TmaLoadMode { .. }
            | CopyOp::TmaStoreMode { .. } => panic!(
                "cute: make_copy_atom({op:?}): a TMA atom owns a descriptor of the tensor it copies; make it with \
                 cpasync::make_tiled_tma_atom"
            ),
            CopyOp::TmemLoad { .. }
            | CopyOp::TmemLoadRed { .. }
            | CopyOp::TmemLoadSpCompress { .. }
            | CopyOp::TmemStore { .. } => crate::nvgpu::tcgen05::copy::tmem_atom_layouts(op, T::DTYPE),
            CopyOp::TmemCp { .. } => crate::nvgpu::tcgen05::s2t::s2t_atom_layouts(op, T::DTYPE),
            CopyOp::Universal { bits: b } => {
                words(b);
                assert!(
                    b % bits == 0 && b <= 256 && b >= bits,
                    "Universal copy of {b} bits is not a whole number of {} elements (max 256 bits)",
                    T::DTYPE
                );
                simple(b / bits)
            }
            CopyOp::G2R { bits: b, attrs } => {
                words(b);
                let name = "CopyG2ROp";
                let values = global_copy_values(name, b, bits, attrs.shared_space);
                if let Err(why) = check_mem_attrs(&attrs.mem_attrs(), Access::Load, MemSpace::Global, values, T::DTYPE)
                {
                    panic!("cute: make_copy_atom({name}, {}): {why}", T::DTYPE);
                }
                simple(values)
            }
            CopyOp::R2G { bits: b, attrs } => {
                words(b);
                let name = "CopyR2GOp";
                let values = global_copy_values(name, b, bits, attrs.shared_space);
                if let Err(why) = check_mem_attrs(&attrs.mem_attrs(), Access::Store, MemSpace::Global, values, T::DTYPE)
                {
                    panic!("cute: make_copy_atom({name}, {}): {why}", T::DTYPE);
                }
                simple(values)
            }
            CopyOp::S2R { bits: b, attrs } => {
                words(b);
                let values = shared_copy_values("CopyS2ROp", b, bits);
                if let Err(why) = check_mem_attrs(&attrs.mem_attrs(), Access::Load, MemSpace::Shared, values, T::DTYPE)
                {
                    panic!("cute: make_copy_atom(CopyS2ROp, {}): {why}", T::DTYPE);
                }
                simple(values)
            }
            CopyOp::R2S { bits: b, attrs } => {
                words(b);
                let values = shared_copy_values("CopyR2SOp", b, bits);
                if let Err(why) = check_mem_attrs(&attrs.mem_attrs(), Access::Store, MemSpace::Shared, values, T::DTYPE)
                {
                    panic!("cute: make_copy_atom(CopyR2SOp, {}): {why}", T::DTYPE);
                }
                simple(values)
            }
            CopyOp::DsmemStore { bits: b } => {
                assert!(
                    matches!(b, 0 | 32 | 64 | 128) && (b == 0 || (b % bits == 0 && b >= bits)),
                    "cute: make_copy_atom(CopyDsmemStoreOp, {}): num_bits_per_copy {b} is not 0, 32, 64 or 128 (a \
                     whole number of {bits}-bit elements: st.async stores one, two or four 32-bit words)",
                    T::DTYPE
                );
                assert!(
                    !T::DTYPE.is_word_packed() && bits <= 32,
                    "cute: make_copy_atom(CopyDsmemStoreOp, {}): st.async stores elements of 8, 16 or 32 bits packed \
                     into 32-bit words",
                    T::DTYPE
                );
                simple(if b == 0 { 1 } else { b / bits })
            }
            CopyOp::BulkS2G { bits: b } | CopyOp::BulkS2GByteMask { bits: b } => {
                assert!(
                    b % 128 == 0 && b % bits == 0,
                    "cute: make_copy_atom(CopyBulkS2GOp, {}): num_bits_per_copy {b} is not a multiple of 128 (bulk \
                     copies move multiples of 16 bytes) and of the element's {bits} bits, or 0 for the whole of mode 0",
                    T::DTYPE
                );
                simple(if b == 0 { 1 } else { b / bits })
            }
            CopyOp::BulkG2S { bits: b, .. } | CopyOp::BulkS2S { bits: b } => {
                assert!(
                    b % 128 == 0 && b % bits == 0,
                    "cute: make_copy_atom({op:?}, {}): num_bits_per_copy {b} is not a multiple of 128 (bulk copies \
                     move multiples of 16 bytes) and of the element's {bits} bits, or 0 for the whole of mode 0",
                    T::DTYPE
                );
                simple(if b == 0 { 1 } else { b / bits })
            }
            CopyOp::CpAsync { bytes, bypass_l1 } => {
                assert!(matches!(bytes, 4 | 8 | 16), "cp.async copies 4, 8 or 16 bytes, not {bytes}");
                assert!(!bypass_l1 || bytes == 16, "cp.async.cg (bypass_l1) requires 16 bytes");
                assert!(bytes * 8 >= bits, "cp.async of {bytes} bytes is smaller than one {}", T::DTYPE);
                simple(bytes * 8 / bits)
            }
            CopyOp::LdMatrix { num, trans } => {
                assert_eq!(T::DTYPE.bits(), 16, "ldmatrix moves 16-bit elements, not {}", T::DTYPE);
                let src = match num {
                    1 => Layout::new(((8, 4), 8), ((8, 0), 1)),
                    2 => Layout::new(((16, 2), 8), ((8, 0), 1)),
                    4 => Layout::new((32, 8), (8, 1)),
                    _ => panic!("ldmatrix loads 1, 2 or 4 matrices, not {num}"),
                };
                let dst = match (num, trans) {
                    (1, false) => Layout::new((32, 2), (2, 1)),
                    (2, false) => Layout::new((32, (2, 2)), (2, (1, 64))),
                    (4, false) => Layout::new((32, (2, 4)), (2, (1, 64))),
                    (1, true) => Layout::new(((4, 8), 2), ((16, 1), 8)),
                    (2, true) => Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    (4, true) => Layout::new(((4, 8), (2, 4)), ((16, 1), (8, 64))),
                    _ => unreachable!(),
                };
                (Layout::new(32, 1), src, dst.clone(), dst)
            }
            CopyOp::StMatrix { num, trans } if T::DTYPE.bits() == 32 => {
                // CuTe C++'s SM90_U32x{1,2,4}_STSM_N on 32-bit elements, one per
                // register: each lane holds one element of each 8x4 matrix (a row of an 8x8 matrix of
                // 16-bit elements is four 32-bit ones). A transposed store moves 16-bit elements
                // across rows, which 32-bit elements do not survive.
                assert!(
                    !trans,
                    "cute: StMatrix8x8x16bOp stores 32-bit elements ({}) only untransposed: a transposed stmatrix moves 16-bit halves",
                    T::DTYPE
                );
                let shared = match num {
                    1 => Layout::new(((8, 4), 4), ((4, 0), 1)),
                    2 => Layout::new(((16, 2), 4), ((4, 0), 1)),
                    4 => Layout::new((32, 4), (4, 1)),
                    _ => panic!("cute: StMatrix8x8x16bOp stores 1, 2 or 4 matrices, not {num}"),
                };
                let regs = Layout::new((32, num), (1, 32));
                (Layout::new(32, 1), regs.clone(), shared, regs)
            }
            CopyOp::StMatrix { num, trans } => {
                assert_eq!(
                    T::DTYPE.bits(),
                    16,
                    "cute: StMatrix8x8x16bOp stores 16-bit elements (or 32-bit ones, untransposed), not {}",
                    T::DTYPE
                );
                // CuTe C++'s SM90 STSM traits: the LDSM layouts, source and destination swapped.
                let shared = match num {
                    1 => Layout::new(((8, 4), 8), ((8, 0), 1)),
                    2 => Layout::new(((16, 2), 8), ((8, 0), 1)),
                    4 => Layout::new((32, 8), (8, 1)),
                    _ => panic!("cute: StMatrix8x8x16bOp stores 1, 2 or 4 matrices, not {num}"),
                };
                let regs = match (num, trans) {
                    (1, false) => Layout::new((32, 2), (2, 1)),
                    (2, false) => Layout::new((32, (2, 2)), (2, (1, 64))),
                    (4, false) => Layout::new((32, (2, 4)), (2, (1, 64))),
                    (1, true) => Layout::new(((4, 8), 2), ((16, 1), 8)),
                    (2, true) => Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    (4, true) => Layout::new(((4, 8), (2, 4)), ((16, 1), (8, 64))),
                    _ => unreachable!(),
                };
                (Layout::new(32, 1), regs.clone(), shared, regs)
            }
            CopyOp::StMatrix16x8x8b { num, trans } => {
                assert!(trans, "cute: StMatrix16x8x8bOp must be transposed (trans: true, CuTe DSL's transpose=True)");
                assert_eq!(T::DTYPE.bits(), 8, "cute: StMatrix16x8x8bOp stores 8-bit elements, not {}", T::DTYPE);
                // CuTe C++'s SM100_U8x{4,8,16}_STSM_T, from bits to bytes: lane `l` byte `q`
                // of register `m` is memory row `2 (l % 4) + q % 2`, byte `l / 4 + 8 (q / 2)`,
                // of matrix `m` (128 bytes each).
                let regs = match num {
                    1 => Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))),
                    2 => Layout::new(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
                    4 => Layout::new(((4, 8), (2, 2, 4)), ((32, 1), (16, 8, 128))),
                    _ => panic!("cute: StMatrix16x8x8bOp stores 1, 2 or 4 matrices, not {num}"),
                };
                let shared = match num {
                    1 => Layout::new(((8, 4), 16), ((16, 0), 1)),
                    2 => Layout::new(((16, 2), 16), ((16, 0), 1)),
                    _ => Layout::new((32, 16), (16, 1)),
                };
                (Layout::new(32, 1), regs.clone(), shared, regs)
            }
            CopyOp::LdMatrix8x16x8b { num, unpack_bits } => {
                check_unpack_bits("LdMatrix8x16x8bOp", unpack_bits);
                assert_eq!(T::DTYPE.bits(), 8, "LdMatrix8x16x8bOp moves 8-bit elements, not {}", T::DTYPE);
                // The 16-bit layouts of LdMatrix8x8x16b in bytes (CuTe's
                // SM75_U32x{1,2,4}_LDSM_N viewed as 8-bit values): lane `l`
                // supplies row `l` (16 bytes) and receives bytes `4 * (l %
                // 4)` to `4 * (l % 4) + 3` of row `l / 4` of each matrix.
                // Unpacking keeps them (CuTe C++'s
                // SM100_SU{4,6}_DU8x16_x{1,2,4}_LDSM_N, in bits).
                let src = match num {
                    1 => Layout::new(((8, 4), 16), ((16, 0), 1)),
                    2 => Layout::new(((16, 2), 16), ((16, 0), 1)),
                    4 => Layout::new((32, 16), (16, 1)),
                    _ => panic!("ldmatrix loads 1, 2 or 4 matrices, not {num}"),
                };
                let dst = match num {
                    1 => Layout::new((32, 4), (4, 1)),
                    2 => Layout::new((32, (4, 2)), (4, (1, 128))),
                    _ => Layout::new((32, (4, 4)), (4, (1, 128))),
                };
                (Layout::new(32, 1), src, dst.clone(), dst)
            }
            CopyOp::LdMatrix16x16x8b { num, trans, unpack_bits } => {
                assert!(trans, "cute: LdMatrix16x16x8bOp must be transposed (trans: true, CuTe DSL's transpose=True)");
                assert!(matches!(num, 1 | 2), "cute: LdMatrix16x16x8bOp loads 1 or 2 matrices, not {num}");
                check_unpack_bits("LdMatrix16x16x8bOp", unpack_bits);
                assert_eq!(T::DTYPE.bits(), 8, "LdMatrix16x16x8bOp moves 8-bit elements, not {}", T::DTYPE);
                // Offsets are 16 * row + byte of 16-byte rows, matrix `m`
                // from 256 m. Lane `l` supplies row `l % 16` of matrix `l /
                // 16`; its value `j + 4 h + 8 m` is byte `l / 4 + 8 h` of
                // row `4 (l % 4) + j` of matrix `m` (the PTX ISA's Figure
                // 108, of the transposed matrix).
                let src = match num {
                    1 => Layout::new(((16, 2), 16), ((16, 0), 1)),
                    _ => Layout::new((32, 16), (16, 1)),
                };
                let dst = match num {
                    1 => Layout::new(((4, 8), (4, 2)), ((64, 1), (16, 8))),
                    _ => Layout::new(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
                };
                (Layout::new(32, 1), src, dst.clone(), dst)
            }
            CopyOp::LdMatrix16x8x8b { num, trans, unpack_bits } => {
                assert!(trans, "cute: LdMatrix16x8x8bOp must be transposed (trans: true, CuTe DSL's transpose=True)");
                assert!(matches!(num, 2 | 4), "cute: LdMatrix16x8x8bOp loads 2 or 4 matrices, not {num}");
                check_unpack_bits("LdMatrix16x8x8bOp", unpack_bits);
                assert_eq!(T::DTYPE.bits(), 8, "LdMatrix16x8x8bOp moves 8-bit elements, not {}", T::DTYPE);
                // CuTe C++'s SM100_U8x8_LDSM_T (num 2) and
                // SM100_U8x16_LDSM_T (num 4), from bits to bytes.
                let src = match num {
                    2 => Layout::new(((2, 2, 4, 2), 16), ((16, 128, 32, 0), 1)),
                    _ => Layout::new(((2, 2, 4, 2), 16), ((16, 128, 32, 256), 1)),
                };
                let dst = match num {
                    2 => Layout::new(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
                    _ => Layout::new(((4, 8), (2, 2, 4)), ((32, 1), (16, 8, 128))),
                };
                (Layout::new(32, 1), src, dst.clone(), dst)
            }
        };
        CopyAtom { op, thr_id, src, dst, reference, tma: None, bound: None, _t: std::marker::PhantomData }
    }

    /// A TMA atom (C++ CuTe's TMA `Copy_Traits`): one thread issues it
    /// (`thr_id` `1:0`) and one application moves `values` elements, the
    /// whole box before any multicast split.
    pub(crate) fn tma(op: CopyOp, values: usize, info: crate::cpasync::TmaAtomInfo) -> CopyAtom<T> {
        let l = Layout::new((1, values), (0, 1));
        CopyAtom {
            op,
            thr_id: Layout::new(1, 0),
            src: l.clone(),
            dst: l.clone(),
            reference: l,
            tma: Some(std::rc::Rc::new(info)),
            bound: None,
            _t: std::marker::PhantomData,
        }
    }

    /// The atom with runtime fields bound (CuTe DSL's `atom.with_(**kwargs)`,
    /// which makes an executable TMA atom): every [`copy`] with it uses
    /// them, as if they were given to [`copy_with`]. `args` sets atom
    /// fields only (not `pred` or `unroll_factor`, which belong to one
    /// copy), the ones this atom takes ([`CopyArgs`]); fields already bound
    /// are replaced. Values made inside a loop or branch are bound there:
    /// use the atom where they are defined.
    #[track_caller]
    pub fn with_(&self, args: impl Into<CopyArgs<'static>>) -> CopyAtom<T> {
        let args: CopyArgs<'static> = args.into();
        assert!(
            args.pred.is_none() && args.unroll_factor.is_none(),
            "with_: pred and unroll_factor are arguments of one copy, not fields of the atom: pass them to copy_with"
        );
        let (takes, _) = atom_field_rules(self.op);
        if let Some(f) = args
            .atom_fields()
            .iter()
            .find(|f| !takes.contains(f) || (**f == "cache_policy" && !self.takes_cache_policy()))
        {
            panic!("with_: the {:?} atom takes no `{f}` field", self.op);
        }
        let mut merged = self.bound.as_deref().cloned().unwrap_or_default();
        let args = AtomFields::of(&args);
        macro_rules! over {
            ($($f:ident),*) => {$( if args.$f.is_some() { merged.$f = args.$f.clone(); } )*};
        }
        over!(cache_policy, tma_bar_ptr, mbar_ptr, mcast_mask, tma_desc_ptr, byte_mask, cta_rank);
        CopyAtom { bound: Some(std::rc::Rc::new(merged)), ..self.clone() }
    }

    /// The runtime fields bound by [`with_`](Self::with_), if any, as copy
    /// arguments.
    pub fn bound_fields(&self) -> Option<CopyArgs<'static>> {
        self.bound.as_deref().map(|b| b.as_args())
    }

    fn takes_cache_policy(&self) -> bool {
        matches!(
            self.op,
            CopyOp::G2R { .. }
                | CopyOp::R2G { .. }
                | CopyOp::TmaLoad { .. }
                | CopyOp::TmaStore
                | CopyOp::TmaReduce(_)
                | CopyOp::TmaLoadMode { .. }
                | CopyOp::TmaStoreMode { .. }
                | CopyOp::BulkS2G { .. }
                | CopyOp::BulkS2GByteMask { .. }
                | CopyOp::BulkG2S { .. }
        )
    }

    /// The descriptor, box and multicast of a TMA atom, `None` for other
    /// atoms.
    pub fn tma_info(&self) -> Option<&crate::cpasync::TmaAtomInfo> {
        self.tma.as_deref()
    }

    /// The instruction this atom issues.
    pub fn op(&self) -> CopyOp {
        self.op
    }

    /// The element type the layouts count in (CuTe DSL's `value_type`).
    pub fn value_type(&self) -> DType {
        T::DTYPE
    }

    /// Atom thread id to thread index (`1:0` for one thread, `32:1` for a
    /// warp).
    pub fn thr_id(&self) -> &Layout {
        &self.thr_id
    }

    /// `(thread, value) ->` offset in the atom's elements of the values each
    /// thread supplies (CuTe DSL's `layout_src_tv`).
    pub fn layout_src_tv(&self) -> &Layout {
        &self.src
    }

    /// `(thread, value) ->` offset in the atom's elements of the values each
    /// thread receives (CuTe DSL's `layout_dst_tv`).
    pub fn layout_dst_tv(&self) -> &Layout {
        &self.dst
    }

    /// The reference `(thread, value)` layout that tiled copies are
    /// expressed in (the destination for `ldmatrix`).
    pub fn layout_ref_tv(&self) -> &Layout {
        &self.reference
    }

    /// Threads that cooperate in one instruction (1, or 32 for `ldmatrix`).
    pub fn num_threads(&self) -> usize {
        self.thr_id.size()
    }

    /// Source elements each thread supplies per instruction.
    pub fn num_val_src(&self) -> usize {
        self.src.mode(1).size()
    }

    /// Destination elements each thread receives per instruction.
    pub fn num_val_dst(&self) -> usize {
        self.dst.mode(1).size()
    }

    fn num_val_ref(&self) -> usize {
        self.reference.mode(1).size()
    }
}

/// `unpack_bits` of an 8-bit `ldmatrix` atom: none, 4 or 6.
#[track_caller]
fn check_unpack_bits(name: &str, unpack_bits: Option<u32>) {
    assert!(
        matches!(unpack_bits, None | Some(4 | 6)),
        "cute: {name} unpacks 4- or 6-bit elements (unpack_bits Some(4) or Some(6)) or none, not unpack_bits={}",
        unpack_bits.unwrap_or(0)
    );
}

/// The values per thread of a G2R or R2G atom of `bits` bits (0: one, and
/// the copy vectorises) on elements of `elem_bits` bits.
#[track_caller]
fn global_copy_values(name: &str, bits: usize, elem_bits: usize, shared_space: SharedSpace) -> usize {
    assert!(
        shared_space == SharedSpace::Cta,
        "cute: make_copy_atom({name}): shared_space=SharedSpace::Cluster addresses another CTA's shared memory \
         (DSMEM), which this global-memory copy does not reach (CopyS2ROp and CopyR2SOp do)"
    );
    if bits == 0 {
        return 1;
    }
    assert!(
        bits.is_multiple_of(elem_bits) && bits <= 128 && bits >= elem_bits,
        "cute: make_copy_atom({name}): num_bits_per_copy {bits} is not a whole number of {elem_bits}-bit elements \
         (at most 128 bits, or 0 to vectorise automatically)"
    );
    bits / elem_bits
}

/// The values per thread of an S2R or R2S atom of `bits` bits (0: one, and
/// the copy vectorises) on elements of `elem_bits` bits.
#[track_caller]
fn shared_copy_values(name: &str, bits: usize, elem_bits: usize) -> usize {
    global_copy_values(name, bits, elem_bits, SharedSpace::Cta)
}

/// `make_copy_atom(op)` with the element type as a type parameter. `op` is
/// a [`CopyOp`] or an operation that converts into one
/// ([`cpasync::CopyBulkS2GOp`](crate::cpasync::CopyBulkS2GOp)).
#[track_caller]
pub fn make_copy_atom<T: Numeric>(op: impl Into<CopyOp>) -> CopyAtom<T> {
    let op = op.into();
    CopyAtom::new(op)
}

/// A copy atom tiled across threads and values.
#[derive(Clone, Debug)]
pub struct TiledCopy<T> {
    atom: CopyAtom<T>,
    /// `(thread, value) -> ` colexicographic index into the tile.
    layout_tv: Layout,
    /// The tile one application of the tiled copy covers: a shape (a `n:1`
    /// layout per mode), or a layout per mode selecting scattered
    /// coordinates (`make_tiled_copy_c_atom`).
    tiler: Tiler,
}

/// The shape of the tile a tiler selects (CuTe's `shape(Tiler_MN)`).
fn tiler_shape(tiler: &Tiler) -> IntTuple {
    match tiler {
        Tree::Leaf(Some(l)) => l.static_shape(),
        Tree::Leaf(None) => panic!("cute: a tiled copy's tiler cannot leave a mode untiled"),
        Tree::Branch(ts) => Tree::Branch(ts.iter().map(tiler_shape).collect()),
    }
}

/// A tiled copy from an explicit `(thread, value) ->` tile-coordinate layout
/// and the tile it covers (CuTe DSL's `make_tiled_copy(atom, layout_tv,
/// tiler_mn)`). `tiler_mn` is a shape such as `(32, 16)` or a [`Tiler`] of
/// one layout per mode. The thread count must be a multiple of the atom's
/// threads and each thread's values a multiple of the atom's values; the
/// atom's threads are the first, consecutive thread indices.
#[track_caller]
pub fn make_tiled_copy<T: Numeric>(atom: &CopyAtom<T>, layout_tv: &Layout, tiler_mn: impl Into<Tiler>) -> TiledCopy<T> {
    let tiler = tiler_mn.into();
    layout_tv.require_static("make_tiled_copy: the thread-value layout", "thread-value layouts are static");
    for layout in tiler.leaves().into_iter().flatten() {
        layout.require_static("make_tiled_copy: the tiler", "a tiled copy covers a static tile");
    }
    assert_eq!(layout_tv.rank(), 2, "cute: make_tiled_copy needs a (thread, value) layout, not {layout_tv}");
    let (threads, values) = (layout_tv.mode(0).size(), layout_tv.mode(1).size());
    assert_eq!(
        threads % atom.num_threads(),
        0,
        "cute: a tiled copy of {threads} threads cannot be built from a {:?} atom of {} threads",
        atom.op,
        atom.num_threads()
    );
    assert_eq!(
        values % atom.num_val_ref(),
        0,
        "cute: each thread's {values} values must be a multiple of the {:?} atom's {}",
        atom.op,
        atom.num_val_ref()
    );
    let tile = tiler_shape(&tiler).size();
    assert!(
        layout_tv.cosize() <= tile,
        "cute: layout_tv {layout_tv} reaches index {} outside the {tile}-element tile",
        layout_tv.cosize() - 1
    );
    TiledCopy { atom: atom.clone(), layout_tv: layout_tv.clone(), tiler }
}

/// Tile `atom` so that each thread of `thr_layout` (a compact `(m, n) ->
/// thread` layout) copies a `val_layout`-shaped block of values (CuTe DSL's
/// `make_tiled_copy_tv`). Warp-level atoms such as `ldmatrix` work as long
/// as each thread's source values end up contiguous in memory.
#[track_caller]
pub fn make_tiled_copy_tv<T: Numeric>(atom: &CopyAtom<T>, thr_layout: &Layout, val_layout: &Layout) -> TiledCopy<T> {
    let (tiler, layout_tv) = make_layout_tv(thr_layout, val_layout);
    make_tiled_copy(atom, &layout_tv, &tiler)
}

/// A tiled copy that produces exactly the A-operand fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_a<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_a_tv(), (mma.tile_size(0), mma.tile_size(2)))
}

/// A tiled copy that produces exactly the B-operand fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_b<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_b_tv(), (mma.tile_size(1), mma.tile_size(2)))
}

/// A tiled copy matching the C-operand (accumulator) fragments of `mma`.
#[track_caller]
pub fn make_tiled_copy_c<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    make_tiled_copy(atom, &mma.layout_c_tv(), (mma.tile_size(0), mma.tile_size(1)))
}

/// A tiled copy with `atom` that partitions like the source side of
/// `tiled_copy` (CuTe DSL's `make_tiled_copy_S`).
#[track_caller]
pub fn make_tiled_copy_s<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tiled_copy: &TiledCopy<U>) -> TiledCopy<T> {
    make_tiled_copy(atom, &tiled_copy.layout_src_tv_tiled(), tiled_copy.tiler_mn())
}

/// A tiled copy with `atom` that partitions like the destination side of
/// `tiled_copy` (CuTe DSL's `make_tiled_copy_D`).
#[track_caller]
pub fn make_tiled_copy_d<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tiled_copy: &TiledCopy<U>) -> TiledCopy<T> {
    make_tiled_copy(atom, &tiled_copy.layout_dst_tv_tiled(), tiled_copy.tiler_mn())
}

/// A minimal tiled copy for moving `mma`'s accumulators in pieces of `atom`'s
/// size, for epilogues that store the C tile in pieces (CuTe DSL's
/// `make_tiled_copy_C_atom`). Each thread takes the first `atom` values of
/// its accumulator fragment, in the fragment's order; the tiler picks out
/// the C coordinates those values cover.
#[track_caller]
pub fn make_tiled_copy_c_atom<T: Numeric>(atom: &CopyAtom<T>, mma: &TiledMma) -> TiledCopy<T> {
    // Truncate the value mode to one atom, keeping the order of the values.
    let layout_c_tv = mma.layout_c_tv();
    let copy_v = atom.num_val_src();
    let mma_v = layout_c_tv.mode(1).size();
    assert!(
        copy_v <= mma_v,
        "cute: the copy atom's {copy_v} source values are more than the {mma_v} accumulators per thread of the MMA"
    );
    let layout_tv = layout_c_tv.compose(&Layout::col_major((layout_c_tv.mode(0).size(), copy_v)));
    // A tiler per mode selecting the C coordinates those values touch.
    let mma_tiler = (mma.tile_size(0), mma.tile_size(1));
    let tiler_0 = Layout::new(mma_tiler, (1, 0)).compose(&layout_tv).filter();
    let tiler_1 = Layout::new(mma_tiler, (0, 1)).compose(&layout_tv).filter();
    let tiler: Tiler = Tree::Branch(vec![Tree::Leaf(Some(tiler_0)), Tree::Leaf(Some(tiler_1))]);
    // Re-express the values as coordinates into the selected tile.
    let tile2mma = Layout::col_major(mma_tiler).compose_tiler(&tiler);
    let layout_tv = tile2mma.left_inverse().compose(&layout_tv);
    make_tiled_copy(atom, &layout_tv, tiler)
}

/// A tiled copy made from a `(thread, value)` layout that gives addresses in
/// the data, while the returned copy's layout gives coordinates of a tile, as
/// every [`TiledCopy`]'s does (CuTe DSL's
/// `make_cotiled_copy(atom, atom_layout_tv, data_layout)`):
/// `atom_layout_tv` maps `(thread, value)` to an address of the data and
/// `data_layout` maps a data coordinate to its address. Thread `t` copies
/// the elements at addresses `atom_layout_tv(t, ..)`, in that order, so the
/// caller fixes how far apart each thread's values are in memory (and so
/// how wide its accesses can be) and leaves the coordinates to
/// `data_layout`. The tiler selects, mode by mode, the coordinates of
/// `data_layout` those addresses hold, so the tiled copy partitions tensors
/// shaped like `data_layout`. Both layouts must have static strides.
///
/// It panics if any thread's value is at an address that `data_layout`
/// does not hold. CuTe checks only the value mode (the offsets of one
/// thread's values), so it accepts a thread mode that reaches beyond the
/// data, and the threads there then wrap around onto other threads'
/// elements; cute-rs checks every thread.
///
/// Ported from CuTe's `make_cotiled_copy` (`include/cute/atom/copy_atom.hpp`
/// in NVIDIA CUTLASS 4.8.1, BSD-3-Clause), with the composition that keeps
/// the appended `1:0` modes (`detail::coalesce_x` in `include/cute/layout.hpp`).
#[track_caller]
pub fn make_cotiled_copy<T: Numeric>(
    atom: &CopyAtom<T>,
    atom_layout_tv: &Layout,
    data_layout: &Layout,
) -> TiledCopy<T> {
    assert!(
        atom_layout_tv.is_static() && data_layout.is_static(),
        "cute: make_cotiled_copy: both layouts need static strides, but atom_layout_tv is {atom_layout_tv} and \
         data_layout is {data_layout}"
    );
    assert_eq!(
        atom_layout_tv.rank(),
        2,
        "cute: make_cotiled_copy needs a (thread, value) -> address layout, not {atom_layout_tv}"
    );
    // Every value must be at an address of the data. (CuTe checks this for
    // the value mode below; a thread mode reaching beyond the data would
    // wrap around onto other threads' elements.)
    let held: std::collections::HashSet<usize> = (0..data_layout.size()).map(|c| data_layout.call(c)).collect();
    let threads = atom_layout_tv.mode(0).size();
    if let Some(i) = (0..atom_layout_tv.size()).find(|&i| !held.contains(&atom_layout_tv.call(i))) {
        panic!(
            "cute: make_cotiled_copy: every value of atom_layout_tv must be at an address that data_layout holds, \
             but value {} of thread {} of {atom_layout_tv} is at address {}, which data_layout {data_layout} does \
             not hold",
            i / threads,
            i % threads,
            atom_layout_tv.call(i)
        );
    }
    // data address -> data coordinate, with 1:0 appended so that addresses
    // past its end map with stride 0.
    let inv_data_layout = Layout::from_modes([data_layout.left_inverse(), Layout::new(1, 0)]);
    // (thread, value) -> data coordinate
    let layout_tv_data = coalesce_x(&inv_data_layout).compose(atom_layout_tv);
    // The addresses of the values must be in the data: going back through
    // data_layout (with 1:0 appended, so that coordinates past its end map
    // with stride 0) must give them again.
    let atom_values = atom_layout_tv.mode(1).coalesce();
    let data_with_zero = Layout::from_modes([data_layout.clone(), Layout::new(1, 0)]);
    let data_values = coalesce_x(&data_with_zero).compose(&layout_tv_data.mode(1)).coalesce();
    assert!(
        data_values == atom_values,
        "cute: make_cotiled_copy: a thread's values must keep their addresses when mapped to data coordinates and \
         back through data_layout, but the values of {atom_layout_tv} are at addresses {atom_values}, and \
         data_layout {data_layout} gives {data_values}"
    );
    // Tiler: for each mode of the data, the coordinates the threads' values
    // hold in it (a by-mode tiler, which keeps the data's modes).
    let flat_data_shape = IntTuple::tuple(data_layout.shape().modes().iter().map(|m| Tree::Leaf(m.size())));
    let rank = flat_data_shape.rank();
    let tiler: Tiler = Tree::tuple((0..rank).map(|i| {
        let unit = IntTuple::tuple((0..rank).map(|j| Tree::Leaf(usize::from(j == i))));
        Tree::Leaf(Some(Layout::new(flat_data_shape.clone(), unit).compose(&layout_tv_data).filter()))
    }));
    // (thread, value) -> tile coordinate: the tile's coordinates in the data
    // (tile_coord -> data_coord), inverted, after layout_tv_data.
    let tile2data = Layout::col_major(flat_data_shape).compose_tiler(&tiler);
    let layout_tv = tile2data.left_inverse().compose(&layout_tv_data);
    make_tiled_copy(atom, &layout_tv, tiler)
}

/// `layout` flattened and coalesced into the same function outside its
/// domain as well as inside (CuTe's `detail::coalesce_x`, which CuTe 4.8's
/// composition applies to its left operand): a last mode `1:d`, such as an
/// appended `1:0`, becomes `2:d`, so that indices past the end step by `d`
/// rather than by the stride of the mode before it.
fn coalesce_x(layout: &Layout) -> Layout {
    let mut modes = layout.flat_modes();
    if let Some(last) = modes.last_mut()
        && last.0 == 1
    {
        last.0 = 2;
    }
    Layout::from_flat(&modes).coalesce()
}

impl<T: Numeric> TiledCopy<T> {
    /// The tiled atom.
    pub fn atom(&self) -> &CopyAtom<T> {
        &self.atom
    }

    /// `(thread, value) ->` index into the tile (CuTe DSL's
    /// `layout_tv_tiled`).
    pub fn layout_tv_tiled(&self) -> &Layout {
        &self.layout_tv
    }

    /// Same as [`layout_tv_tiled`](Self::layout_tv_tiled).
    pub fn layout_tv(&self) -> &Layout {
        &self.layout_tv
    }

    /// The tile one application covers (CuTe DSL's `tiler_mn`).
    pub fn tiler_mn(&self) -> &Tiler {
        &self.tiler
    }

    /// The shape of that tile.
    pub fn tile_shape(&self) -> IntTuple {
        tiler_shape(&self.tiler)
    }

    /// Threads taking part in the tiled copy (CuTe DSL's `size`).
    pub fn size(&self) -> usize {
        self.layout_tv.mode(0).size()
    }

    /// Same as [`size`](Self::size).
    pub fn num_threads(&self) -> usize {
        self.size()
    }

    /// The atom's element type.
    pub fn value_type(&self) -> DType {
        T::DTYPE
    }

    /// The tiled copy with its atom's runtime fields bound
    /// ([`CopyAtom::with_`]; CuTe DSL's `tiled_copy.with_(**kwargs)`).
    #[track_caller]
    pub fn with_(&self, args: impl Into<CopyArgs<'static>>) -> TiledCopy<T> {
        TiledCopy { atom: self.atom.with_(args), ..self.clone() }
    }

    /// The atom's thread id layout.
    pub fn thr_id(&self) -> &Layout {
        self.atom.thr_id()
    }

    /// The atom's source `(thread, value)` layout.
    pub fn layout_src_tv(&self) -> &Layout {
        self.atom.layout_src_tv()
    }

    /// The atom's destination `(thread, value)` layout.
    pub fn layout_dst_tv(&self) -> &Layout {
        self.atom.layout_dst_tv()
    }

    /// `(thread, value) ->` tile index of the values each thread supplies
    /// (CuTe DSL's `layout_src_tv_tiled`).
    pub fn layout_src_tv_tiled(&self) -> Layout {
        self.layout_trg_tv_tiled(true)
    }

    /// `(thread, value) ->` tile index of the values each thread receives
    /// (CuTe DSL's `layout_dst_tv_tiled`).
    pub fn layout_dst_tv_tiled(&self) -> Layout {
        self.layout_trg_tv_tiled(false)
    }

    fn layout_trg_tv_tiled(&self, src: bool) -> Layout {
        let reference = Layout::from_modes([Layout::col_major(self.tile_shape()), Layout::new(1, 0)]);
        let l = self.tile2thrfrg(&reference, &self.ref2trg(src));
        Layout::from_modes([l.mode(0), l.mode(1)])
    }

    /// Thread `thr_idx`'s view.
    pub fn get_slice(&self, thr_idx: impl Into<IndexExpr>) -> ThrCopy<T> {
        ThrCopy { tiled: self.clone(), thr_idx: thr_idx.into() }
    }

    /// View a thread's fragment (a register tensor whose mode 0 holds that
    /// thread's values in the order of this copy's value layout, such as an
    /// MMA fragment from `partition_fragment_a/b/c`) as a source or
    /// destination of this copy: `((atom_values, rest), rest_m, rest_n, ...)`,
    /// with the values of one copy tile gathered from the fragment's modes
    /// (CuTe's `retile`).
    #[track_caller]
    pub fn retile<U>(&self, tensor: &Tensor<U>) -> Tensor<U> {
        if let Some((path, leaf)) = tensor.shape().first_dynamic() {
            panic!(
                "retile: tensor {} has the dynamic extent {leaf:?} at mode {path:?}; retile regroups the values of a \
                 thread's fragment, whose extents are static",
                tensor.layout()
            );
        }
        let v = tensor.layout().mode(0).size();
        // (m, n) -> which block of `v` values of a thread holds the element.
        let frg_layout_mn = self.layout_tv.right_inverse().with_shape(self.tile_shape()).upcast(self.size() * v);
        // (atom values, rest values) -> (v, m, n).
        let frg_layout_v = crate::layout::logical_product(&Layout::new(v, 1), frg_layout_mn.right_inverse())
            .zipped_divide(&Layout::new(self.atom.num_val_ref(), 1).into());
        let mut tile = vec![v];
        match frg_layout_mn.static_shape().product_each() {
            Tree::Leaf(n) => tile.push(n),
            Tree::Branch(ns) => tile.extend(ns.iter().map(IntTuple::size)),
        }
        let mut t = tensor.clone();
        if t.rank() < tile.len() {
            assert!(
                tile[t.rank()..].iter().all(|&n| n == 1),
                "cute: cannot retile a rank-{} tensor {t:?} for a copy over {} modes",
                t.rank(),
                tile.len() - 1
            );
            let padded = t.layout().pad_to_rank(tile.len());
            t = Tileable::with_layout(&t, padded);
        }
        let r = t.rank();
        let tiled = crate::layout::zipped_divide(&t, Tiler::from(&IntTuple::from(tile)));
        let viewed = tiled.compose(Tree::Branch(vec![Tree::Leaf(Some(frg_layout_v)), Tree::Leaf(None)]));
        let mut rest = vec![Tree::Leaf(Some(IndexExpr::Static(0)))];
        rest.extend((1..r).map(|_| Tree::Leaf(None)));
        viewed.slice(Tree::Branch(vec![Tree::Leaf(None), Tree::Branch(rest)]))
    }

    /// `(source or destination thread, value) ->` reference `(thread, value)`.
    fn ref2trg(&self, src: bool) -> Layout {
        let target = if src { &self.atom.src } else { &self.atom.dst };
        self.atom.reference.right_inverse().compose(target)
    }

    /// `((tile...), (rest...))` to `(thread, value, (rest...))` (CuTe's
    /// `tile2thrfrg`).
    fn tile2thrfrg(&self, tiled: &Layout, ref2trg: &Layout) -> Layout {
        let atom = &self.atom;
        // The threads and values of one atom, and how the atoms tile.
        let atom_tv = self.layout_tv.zipped_divide(&(atom.num_threads(), atom.num_val_ref()).into());
        let trg_tv = atom_tv.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(ref2trg.clone())), Tree::Leaf(None)]));
        let zipped = Layout::from_modes([
            Layout::from_modes([trg_tv.get(&[0, 0]), trg_tv.get(&[1, 0])]),
            Layout::from_modes([trg_tv.get(&[0, 1]), trg_tv.get(&[1, 1])]),
        ]);
        let thrval2mn = zipped.coalesce_profile(&(1, (1, 1)).into());
        let tv = tiled.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(thrval2mn)), Tree::Leaf(None)]));
        Layout::from_modes([tv.get(&[0, 0]), tv.get(&[0, 1]), tv.mode(1)])
    }

    /// `(thread, value, rest...)` view of a tensor layout, for the source
    /// (`src == true`) or destination side of the atom.
    fn tidfrg(&self, layout: &Layout, src: bool) -> Layout {
        self.tile2thrfrg(&layout.zipped_divide(&self.tiler), &self.ref2trg(src))
    }
}

/// One thread's slice of a [`TiledCopy`].
#[derive(Clone, Debug)]
pub struct ThrCopy<T> {
    tiled: TiledCopy<T>,
    thr_idx: IndexExpr,
}

impl<T: Numeric> ThrCopy<T> {
    /// The tiled copy this is a slice of.
    pub fn tiled_copy(&self) -> &TiledCopy<T> {
        &self.tiled
    }

    /// The thread whose slice this is.
    pub fn thr_idx(&self) -> &IndexExpr {
        &self.thr_idx
    }

    #[track_caller]
    fn partition<U>(&self, t: &Tensor<U>, src: bool) -> Tensor<U> {
        let tiled = self.tiled.clone();
        let layout = tiled.tidfrg(t.layout(), src);
        let viewed = Tileable::map_layout(t, layout, move |l| tiled.tidfrg(l, src));
        let rest = viewed.layout().mode(2).rank();
        viewed.slice(Tree::Branch(vec![
            Tree::Leaf(Some(self.thr_idx.clone())),
            Tree::Leaf(None),
            Tree::Branch(vec![Tree::Leaf(None); rest]),
        ]))
    }

    /// This thread's part of a source tile: `(CPY, CPY_M, CPY_N, ...)`. The
    /// tensor may have runtime extents: `CPY` has the tiled copy's static
    /// extents, and the modes after it count the copy's tiles.
    #[track_caller]
    pub fn partition_s<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, true)
    }

    /// This thread's part of a destination tile: `(CPY, CPY_M, CPY_N, ...)`,
    /// as in [`partition_s`](Self::partition_s).
    #[track_caller]
    pub fn partition_d<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, false)
    }

    /// View a fragment (such as an MMA fragment from `partition_fragment_a/b/c`)
    /// as a destination of this copy ([`TiledCopy::retile`]).
    #[track_caller]
    pub fn retile_d<U>(&self, frag: &Tensor<U>) -> Tensor<U> {
        self.tiled.retile(frag)
    }

    /// View a fragment as a source of this copy ([`TiledCopy::retile`]).
    #[track_caller]
    pub fn retile_s<U>(&self, frag: &Tensor<U>) -> Tensor<U> {
        self.tiled.retile(frag)
    }
}

/// Anything that can drive [`copy`]: an atom or a tiled copy.
pub trait AsCopyAtom<T> {
    fn copy_atom(&self) -> &CopyAtom<T>;
}

impl<T> AsCopyAtom<T> for CopyAtom<T> {
    fn copy_atom(&self) -> &CopyAtom<T> {
        self
    }
}

impl<T> AsCopyAtom<T> for TiledCopy<T> {
    fn copy_atom(&self) -> &CopyAtom<T> {
        &self.atom
    }
}

/// The element types a copy operand may have for an atom on `T` elements:
/// `T` itself (data), or [`Ident`] (a coordinate tensor, the TMA tensor's
/// partition that a TMA load reads its coordinates from).
pub trait CopyElem<T>: 'static {
    /// Whether the operand holds coordinates rather than data.
    #[doc(hidden)]
    const COORDINATES: bool;
    /// Whether the operand is the one-byte-per-value shared memory of a padded
    /// FP4/FP6 TMA load.
    #[doc(hidden)]
    const UNPACKED: bool = false;
}

// A TMA load of FP4 or FP6 values through an unpacking format
// (`internal_type` 8-bit) writes 16-byte slots of 16 values, so its shared-memory operand
// is a `u8` tensor with one element per value.
impl CopyElem<crate::numeric::F4E2M1FN> for u8 {
    const COORDINATES: bool = false;
    const UNPACKED: bool = true;
}

impl CopyElem<crate::numeric::F6E2M3FN> for u8 {
    const COORDINATES: bool = false;
    const UNPACKED: bool = true;
}

impl CopyElem<crate::numeric::F6E3M2FN> for u8 {
    const COORDINATES: bool = false;
    const UNPACKED: bool = true;
}

impl<T: Numeric> CopyElem<T> for T {
    const COORDINATES: bool = false;
}

impl<T: Numeric> CopyElem<T> for Ident {
    const COORDINATES: bool = true;
}

/// Copy `src` to `dst` with the instruction of `atom` (CuTe DSL's
/// `cute.copy(atom, src, dst)`). Both tensors are split into atom-sized
/// chunks along their first mode; the remaining modes are iterated. The
/// chunks are unrolled when the kernel is built, so the extents of both
/// tensors must be static; a tensor in global memory with runtime extents
/// is copied one tile at a time, its rest modes sliced away (by a
/// [`for_range`] over a runtime number of tiles), or with
/// [`CopyArgs::unroll_factor`], which loops over the rest modes. Strides
/// and offsets may be runtime values.
///
/// The operands are data tensors of the atom's element type, except for a
/// TMA load ([`cpasync::CopyBulkTensorTileG2SOp`](crate::cpasync::CopyBulkTensorTileG2SOp)),
/// whose source is a coordinate tensor (the TMA tensor's partition from
/// [`tma_partition`](crate::cpasync::tma_partition)) and which needs its
/// mbarrier: copy it with [`copy_with`] and [`CopyArgs::tma_bar_ptr`], or
/// bind the mbarrier first with [`CopyAtom::with_`].
#[track_caller]
pub fn copy<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(atom: &impl AsCopyAtom<T>, src: &Tensor<S>, dst: &Tensor<D>) {
    copy_front(atom.copy_atom(), src, dst, &CopyArgs::default(), "copy");
}

/// [`copy`] with a predicate per element of `src` (only the first element
/// of each atom-sized chunk is consulted), or one per application of the
/// atom (CuTe DSL's profile `(ATOM_REST, REST...)`: as many flags as atom
/// chunks of the source). Predicated-off register and memory destinations
/// are left unchanged, except for `cp.async`, which zero-fills them.
#[track_caller]
pub fn copy_if<T: Numeric>(atom: &impl AsCopyAtom<T>, pred: &Tensor<bool>, src: &Tensor<T>, dst: &Tensor<T>) {
    copy_front(atom.copy_atom(), src, dst, &CopyArgs::new().pred(pred), "copy_if");
}

/// The keyword arguments of CuTe DSL's `cute.copy(atom, src, dst, *, pred=,
/// unroll_factor=, **kwargs)`: `cute.copy`'s own (`pred`, `unroll_factor`)
/// and the fields of the atom's runtime state that the call sets (CuTe
/// DSL's `Trait.unpack` keywords, the fields below from `cache_policy` to
/// `cta_rank`). Build it with
/// the methods (`CopyArgs::new().tma_bar_ptr(&mbar)`) or as a struct with
/// `..Default::default()`.
///
/// Each atom takes some of the fields, and needs some:
///
/// | atom | takes | needs |
/// |---|---|---|
/// | TMA load ([`cpasync::CopyBulkTensorTileG2SOp`](crate::cpasync::CopyBulkTensorTileG2SOp)) | `tma_bar_ptr`, `cache_policy`, `tma_desc_ptr`, `mcast_mask` (ignored, as in CuTe DSL, so that one body serves multicast and plain loads) | `tma_bar_ptr` |
/// | multicast TMA load ([`cpasync::CopyBulkTensorTileG2SMulticastOp`](crate::cpasync::CopyBulkTensorTileG2SMulticastOp)) | the same | `tma_bar_ptr`, `mcast_mask` |
/// | TMA store and reduce store | `cache_policy`, `tma_desc_ptr` | |
/// | [`CopyOp::BulkG2S`] | `mbar_ptr`, `cache_policy`, and with `multicast` `mcast_mask` | `mbar_ptr` (and `mcast_mask`) |
/// | [`CopyOp::BulkS2S`] | `mbar_ptr`, `cta_rank` | both |
/// | [`CopyOp::G2R`], [`CopyOp::R2G`], [`CopyOp::BulkS2G`] | `cache_policy` | |
/// | the others | none | |
///
/// `byte_mask` belongs to [`CopyBulkS2GByteMaskOp`](crate::cpasync::CopyBulkS2GByteMaskOp),
/// which needs it. A field an atom does not take
/// panics when the kernel is built.
#[derive(Clone, Debug, Default)]
pub struct CopyArgs<'a> {
    /// `pred=`: a flag per element of the source, or per application of
    /// the atom, as for [`copy_if`].
    pub pred: Option<&'a Tensor<bool>>,
    /// `unroll_factor=`: copy the rest modes with a loop unrolled this many
    /// times (`LoopUnroll(count)`), rather than unrolling every
    /// application when the kernel is built. Both operands must be memory
    /// or coordinate tensors (a runtime loop cannot index a register
    /// fragment); otherwise the copy is unrolled fully, as CuTe DSL's
    /// fragments are. The rest modes may then have runtime extents.
    pub unroll_factor: Option<u32>,
    /// `cache_policy=`: an L2 cache policy (a 64-bit value from
    /// `createpolicy`, such as [`CacheHintSm90`](crate::cpasync::CacheHintSm90)'s;
    /// sm_80) for the loads of a [`CopyOp::G2R`] atom, the stores of a
    /// [`CopyOp::R2G`] atom (`.L2::cache_hint`), a TMA load, store or
    /// reduction, or a bulk copy to or from global memory.
    pub cache_policy: Option<Int64>,
    /// `tma_bar_ptr=`: the mbarrier a TMA load's bytes count down.
    pub tma_bar_ptr: Option<Ptr<u64>>,
    /// `mbar_ptr=`: the mbarrier of a non-tensor bulk copy to shared memory.
    pub mbar_ptr: Option<Ptr<u64>>,
    /// `mcast_mask=`: the CTAs of the cluster a multicast load writes.
    pub mcast_mask: Option<Int16>,
    /// `tma_desc_ptr=`: the generic address of a tensor map to use instead
    /// of the atom's own descriptor (64-byte aligned, 128 bytes), such as
    /// [`TmaDescriptor::get_ptr`](crate::TmaDescriptor::get_ptr) of another
    /// descriptor of the same box.
    pub tma_desc_ptr: Option<Int64>,
    /// `byte_mask=`: the bytes of each 16-byte chunk a byte-masked bulk store
    /// writes.
    pub byte_mask: Option<Int16>,
    /// `cta_rank=`: the CTA of the cluster a peer bulk copy addresses.
    pub cta_rank: Option<Int32>,
}

/// The name of CuTe DSL's `cute.copy` keyword arguments before 4.8's
/// generalisation; the same struct.
pub type CopyOptions<'a> = CopyArgs<'a>;

impl<'a, 'b> From<&'b CopyArgs<'a>> for CopyArgs<'a> {
    fn from(args: &'b CopyArgs<'a>) -> Self {
        args.clone()
    }
}

impl<'a> CopyArgs<'a> {
    pub fn new() -> CopyArgs<'a> {
        CopyArgs::default()
    }

    pub fn pred(mut self, pred: &'a Tensor<bool>) -> Self {
        self.pred = Some(pred);
        self
    }

    pub fn unroll_factor(mut self, count: u32) -> Self {
        self.unroll_factor = Some(count);
        self
    }

    pub fn cache_policy(mut self, policy: impl IntoValue<i64>) -> Self {
        self.cache_policy = Some(policy.into_value());
        self
    }

    pub fn tma_bar_ptr(mut self, mbarrier: &Ptr<u64>) -> Self {
        self.tma_bar_ptr = Some(mbarrier.clone());
        self
    }

    pub fn mbar_ptr(mut self, mbarrier: &Ptr<u64>) -> Self {
        self.mbar_ptr = Some(mbarrier.clone());
        self
    }

    pub fn mcast_mask(mut self, mask: impl IntoValue<i16>) -> Self {
        self.mcast_mask = Some(mask.into_value());
        self
    }

    pub fn tma_desc_ptr(mut self, address: impl IntoValue<i64>) -> Self {
        self.tma_desc_ptr = Some(address.into_value());
        self
    }

    pub fn byte_mask(mut self, mask: impl IntoValue<i16>) -> Self {
        self.byte_mask = Some(mask.into_value());
        self
    }

    pub fn cta_rank(mut self, rank: impl IntoValue<i32>) -> Self {
        self.cta_rank = Some(rank.into_value());
        self
    }

    /// CuTe DSL's `tma_multicast=` keyword (`cluster_shape`, `multicast_dim`)
    /// belongs to the block API ([`block_copy`](crate::block::block_copy)),
    /// whose multicast is not supported yet: this panics.
    #[track_caller]
    pub fn tma_multicast(self, cluster_shape: (u32, u32), multicast_dim: char) -> Self {
        panic!(
            "cute: copy: tma_multicast=(cluster_shape={cluster_shape:?}, multicast_dim={multicast_dim:?}) is the \
             block API's multicast (cute::block::block_copy), which is not supported yet; for a multicast TMA load \
             use a multicast atom and pass mcast_mask"
        )
    }

    /// The atom fields that are set, by CuTe DSL's keyword names.
    fn atom_fields(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        let mut note = |set: bool, name: &'static str| {
            if set {
                v.push(name)
            }
        };
        note(self.tma_bar_ptr.is_some(), "tma_bar_ptr");
        note(self.mbar_ptr.is_some(), "mbar_ptr");
        note(self.mcast_mask.is_some(), "mcast_mask");
        note(self.cache_policy.is_some(), "cache_policy");
        note(self.tma_desc_ptr.is_some(), "tma_desc_ptr");
        note(self.byte_mask.is_some(), "byte_mask");
        note(self.cta_rank.is_some(), "cta_rank");
        v
    }

    /// These arguments with the atom's bound fields: a field given both
    /// ways panics.
    #[track_caller]
    fn with_bound(&self, bound: &AtomFields, func: &str) -> CopyArgs<'a> {
        fn pick<X: Clone>(call: &Option<X>, bound: &Option<X>, name: &str, func: &str) -> Option<X> {
            match (call, bound) {
                (Some(_), Some(_)) => panic!(
                    "{func}: `{name}` was bound to the atom by with_ and is given again to the copy: give it once"
                ),
                (Some(x), None) | (None, Some(x)) => Some(x.clone()),
                (None, None) => None,
            }
        }
        CopyArgs {
            pred: self.pred,
            unroll_factor: self.unroll_factor,
            cache_policy: pick(&self.cache_policy, &bound.cache_policy, "cache_policy", func),
            tma_bar_ptr: pick(&self.tma_bar_ptr, &bound.tma_bar_ptr, "tma_bar_ptr", func),
            mbar_ptr: pick(&self.mbar_ptr, &bound.mbar_ptr, "mbar_ptr", func),
            mcast_mask: pick(&self.mcast_mask, &bound.mcast_mask, "mcast_mask", func),
            tma_desc_ptr: pick(&self.tma_desc_ptr, &bound.tma_desc_ptr, "tma_desc_ptr", func),
            byte_mask: pick(&self.byte_mask, &bound.byte_mask, "byte_mask", func),
            cta_rank: pick(&self.cta_rank, &bound.cta_rank, "cta_rank", func),
        }
    }
}

/// The atom fields of [`CopyArgs`] that [`CopyAtom::with_`] binds.
#[derive(Clone, Debug, Default)]
struct AtomFields {
    cache_policy: Option<Int64>,
    tma_bar_ptr: Option<Ptr<u64>>,
    mbar_ptr: Option<Ptr<u64>>,
    mcast_mask: Option<Int16>,
    tma_desc_ptr: Option<Int64>,
    byte_mask: Option<Int16>,
    cta_rank: Option<Int32>,
}

impl AtomFields {
    fn of(a: &CopyArgs) -> AtomFields {
        AtomFields {
            cache_policy: a.cache_policy,
            tma_bar_ptr: a.tma_bar_ptr.clone(),
            mbar_ptr: a.mbar_ptr.clone(),
            mcast_mask: a.mcast_mask,
            tma_desc_ptr: a.tma_desc_ptr,
            byte_mask: a.byte_mask,
            cta_rank: a.cta_rank,
        }
    }

    fn as_args(&self) -> CopyArgs<'static> {
        CopyArgs {
            cache_policy: self.cache_policy,
            tma_bar_ptr: self.tma_bar_ptr.clone(),
            mbar_ptr: self.mbar_ptr.clone(),
            mcast_mask: self.mcast_mask,
            tma_desc_ptr: self.tma_desc_ptr,
            byte_mask: self.byte_mask,
            cta_rank: self.cta_rank,
            ..CopyArgs::default()
        }
    }
}

/// The atom fields `op` takes, and those it needs, by CuTe DSL's keyword
/// names. A cache policy on the other atoms is refused where they copy.
fn atom_field_rules(op: CopyOp) -> (&'static [&'static str], &'static [&'static str]) {
    match op {
        CopyOp::TmaLoad { multicast: false, .. } => {
            (&["tma_bar_ptr", "cache_policy", "tma_desc_ptr", "mcast_mask"], &["tma_bar_ptr"])
        }
        CopyOp::TmaLoad { multicast: true, .. } => {
            (&["tma_bar_ptr", "cache_policy", "tma_desc_ptr", "mcast_mask"], &["tma_bar_ptr", "mcast_mask"])
        }
        CopyOp::TmaStore | CopyOp::TmaReduce(_) => (&["cache_policy", "tma_desc_ptr"], &[]),
        // m7-tma-im2col and m7-tma-sm100-modes.
        CopyOp::TmaLoadMode { multicast: false, .. } => {
            (&["tma_bar_ptr", "cache_policy", "tma_desc_ptr", "mcast_mask"], &["tma_bar_ptr"])
        }
        CopyOp::TmaLoadMode { multicast: true, .. } => {
            (&["tma_bar_ptr", "cache_policy", "tma_desc_ptr", "mcast_mask"], &["tma_bar_ptr", "mcast_mask"])
        }
        CopyOp::TmaStoreMode { .. } => (&["cache_policy", "tma_desc_ptr"], &[]),
        CopyOp::BulkS2GByteMask { .. } => (&["cache_policy", "byte_mask"], &["byte_mask"]),
        CopyOp::BulkG2S { multicast: false, .. } => (&["mbar_ptr", "cache_policy"], &["mbar_ptr"]),
        CopyOp::BulkG2S { multicast: true, .. } => {
            (&["mbar_ptr", "mcast_mask", "cache_policy"], &["mbar_ptr", "mcast_mask"])
        }
        CopyOp::BulkS2S { .. } => (&["mbar_ptr", "cta_rank"], &["mbar_ptr", "cta_rank"]),
        CopyOp::DsmemStore { .. } => (&["mbar_ptr"], &["mbar_ptr"]),
        _ => (&["cache_policy"], &[]),
    }
}

/// Panic if `args` sets an atom field `op` does not take, or misses one it
/// needs.
#[track_caller]
fn check_atom_fields(op: CopyOp, args: &CopyArgs, func: &str) {
    let (takes, needs) = atom_field_rules(op);
    let given = args.atom_fields();
    if let Some(f) = given.iter().find(|f| !takes.contains(f)) {
        let why = match *f {
            "tma_bar_ptr" => "it is a TMA load's field",
            "tma_desc_ptr" => "it is a TMA copy's field",
            "mcast_mask" => "it is a multicast load's field",
            "mbar_ptr" => {
                "it is the mbarrier of a bulk copy to shared memory (CopyBulkG2SOp, CopyBulkS2SOp) or of a \
                 CopyDsmemStoreOp"
            }
            "byte_mask" => "it is a byte-masked bulk store's field (CopyBulkS2GByteMaskOp)",
            "cache_policy" => "it takes no L2 cache hint",
            _ => "it is a peer bulk copy's field (CopyBulkS2SOp)",
        };
        panic!("{func}: the {op:?} atom takes no `{f}` argument: {why}");
    }
    for f in needs {
        if !given.contains(f) {
            let how = match *f {
                "tma_bar_ptr" => {
                    "a TMA load signals its completion on an mbarrier: CopyArgs::new().tma_bar_ptr(&mbarrier)"
                }
                "mbar_ptr" => {
                    "a bulk copy to shared memory signals its completion on an mbarrier: \
                               CopyArgs::new().mbar_ptr(&mbarrier)"
                }
                "mcast_mask" => "a multicast copy writes the CTAs of its mask: CopyArgs::new().mcast_mask(mask)",
                "byte_mask" => {
                    "a byte-masked bulk store writes the bytes of each 16-byte chunk its mask selects: \
                     CopyArgs::new().byte_mask(mask)"
                }
                _ => {
                    "a peer copy writes the shared memory of another CTA of the cluster: CopyArgs::new().cta_rank(rank)"
                }
            };
            panic!("{func}: the {op:?} atom needs `{f}` ({how}), or bind it with atom.with_(..)");
        }
    }
}

/// [`copy`] with CuTe DSL's keyword arguments ([`CopyArgs`], or a
/// reference to one): a predicate, an unroll factor and the atom's runtime
/// fields, such as a TMA load's mbarrier
/// (`copy_with(&tma_atom, &tAgA.slice((.., k)), &tAsA.slice((.., s)),
/// CopyArgs::new().tma_bar_ptr(&mbar))`).
#[track_caller]
pub fn copy_with<'a, T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &impl AsCopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: impl Into<CopyArgs<'a>>,
) {
    copy_front(atom.copy_atom(), src, dst, &args.into(), "copy_with");
}

/// An auxiliary operand of a copy (CuTe DSL passes `[primary, aux...]` as
/// `src` or `dst`): a tensor of any element type or of coordinates, such as
/// the index tensor of a TMA gather4 load.
#[derive(Clone, Debug)]
pub struct AuxOperand {
    layout: Layout,
    /// The element type, or `None` for coordinates.
    dtype: Option<DType>,
    /// The tensor itself, for the operands an atom reads: an
    /// `Int32` index tensor, or a coordinate tensor.
    index: Option<Tensor<i32>>,
    coords: Option<Tensor<Ident>>,
}

impl<T: Numeric> From<&Tensor<T>> for AuxOperand {
    fn from(t: &Tensor<T>) -> Self {
        let index = (T::DTYPE == DType::I32).then(|| t.retyped());
        AuxOperand { layout: t.layout().clone(), dtype: Some(T::DTYPE), index, coords: None }
    }
}

impl From<&Tensor<Ident>> for AuxOperand {
    fn from(t: &Tensor<Ident>) -> Self {
        AuxOperand { layout: t.layout().clone(), dtype: None, index: None, coords: Some(t.clone()) }
    }
}

/// [`copy_with`] with auxiliary operands after the primary source and
/// destination (CuTe DSL's `cute.copy(atom, [src, aux...], [dst, aux...])`,
/// as TMA gather4 loads take `[tAgA, tAgI]`). The auxiliary operands are
/// canonicalised as the primary ones are (rank 2, the rest modes grouped)
/// and must have as many rest elements. The gather4 and scatter4 TMA atoms
/// take the index tensor's partition (`(&[tAgA], [&tAgI])` as the source of
/// a gather4 load, the destination of a scatter4 store), and the override
/// atoms the residue's (with the global tile's partition, in memory, as the
/// primary operand); roadmap part m7-tma-sm100-modes.
#[track_caller]
pub fn copy_aux<'a, T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &impl AsCopyAtom<T>,
    src: (&Tensor<S>, &[AuxOperand]),
    dst: (&Tensor<D>, &[AuxOperand]),
    args: impl Into<CopyArgs<'a>>,
) {
    let atom = atom.copy_atom();
    let _args: CopyArgs = args.into();
    let rest = |l: &Layout| canonical_layout(l).mode(1).shape().clone();
    for (primary, aux, role) in [(src.0.layout(), src.1, "source"), (dst.0.layout(), dst.1, "destination")] {
        for (k, a) in aux.iter().enumerate() {
            let (p, q) = (rest(primary), rest(&a.layout));
            if p.is_static() && q.is_static() && p.size() != q.size() {
                panic!(
                    "copy: auxiliary {role} operand {k} ({}) has {} rest elements (mode 1 after grouping), but the \
                     primary {role} {primary} has {}",
                    a.dtype.map_or("coordinates".to_string(), |t| t.cute_name().to_string()),
                    q.size(),
                    p.size()
                );
            }
        }
    }
    if src.1.is_empty() && dst.1.is_empty() {
        copy_front(atom, src.0, dst.0, &_args, "copy_aux");
        return;
    }
    // Gather4, scatter4 and the overrides.
    if let CopyOp::TmaLoadMode { mode, .. } | CopyOp::TmaStoreMode { mode } = atom.op
        && mode != crate::cpasync::TmaOpMode::Im2col
    {
        aux_tma_copy(atom, src, dst, &_args, mode);
        return;
    }
    panic!(
        "copy: the {:?} atom takes no auxiliary operands (the gather4/scatter4 TMA atoms take an index tensor, the \
         override ones a residue tensor)",
        atom.op
    );
}

/// A gather4/scatter4 or override copy: one
/// auxiliary operand on the global side.
#[track_caller]
fn aux_tma_copy<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: (&Tensor<S>, &[AuxOperand]),
    dst: (&Tensor<D>, &[AuxOperand]),
    args: &CopyArgs,
    mode: crate::cpasync::TmaOpMode,
) {
    let func = "copy_aux";
    let args = match &atom.bound {
        Some(bound) => args.with_bound(bound, func),
        None => args.clone(),
    };
    check_atom_fields(atom.op, &args, func);
    let load = atom.op.is_tma_load();
    let (global, aux, smem_is_src) =
        if load { (src.0.retyped::<T>(), src.1, false) } else { (dst.0.retyped::<T>(), dst.1, true) };
    let smem: Tensor<T> = if smem_is_src { src.0.retyped() } else { dst.0.retyped() };
    let others = if load { dst.1 } else { src.1 };
    assert!(
        aux.len() == 1 && others.is_empty(),
        "{func}: the {:?} atom takes one auxiliary operand, on its global side ({})",
        atom.op,
        if load { "the source" } else { "the destination" }
    );
    assert!(args.pred.is_none(), "{func}: TMA copies are not predicated");
    match mode {
        crate::cpasync::TmaOpMode::Rows4 => {
            let coords_side = if load { S::COORDINATES } else { D::COORDINATES };
            assert!(coords_side, "{func}: the global side of a gather4/scatter4 copy is the TMA tensor's partition");
            let Some(index) = &aux[0].index else {
                panic!("{func}: a gather4/scatter4 copy's auxiliary operand is the Int32 index tensor's partition")
            };
            let coords: Tensor<Ident> = global.retyped();
            crate::nvgpu::tma_sm100::rows4_copy(atom, &coords, index, &smem, &args, load, func);
        }
        crate::cpasync::TmaOpMode::Override { .. } => {
            let Some(residue) = &aux[0].coords else {
                panic!("{func}: an override copy's auxiliary operand is the residue tensor's partition (coordinates)")
            };
            crate::nvgpu::tma_sm100::override_copy(atom, &global, residue, &smem, &args, load, func);
        }
        crate::cpasync::TmaOpMode::Im2col => unreachable!("im2col copies take no auxiliary operand"),
    }
}

/// A layout canonicalised as `cute.copy` canonicalises its operands: size-1
/// modes appended up to rank 2, and modes 1 and up grouped into mode 1.
fn canonical_layout(l: &Layout) -> Layout {
    let l = l.pad_to_rank(2);
    if l.rank() == 2 {
        l
    } else {
        Layout::from_modes([l.mode(0), Layout::from_modes((1..l.rank()).map(|i| l.mode(i)))])
    }
}

/// A tensor canonicalised as [`canonical_layout`] does.
fn canonical<X>(t: &Tensor<X>) -> Tensor<X> {
    Tileable::with_layout(t, canonical_layout(t.layout()))
}

/// What [`copy`], [`copy_if`], [`copy_with`] and [`copy_aux`] share: the
/// atom's bound fields, the fields' checks, the unroll factor's loop, and
/// the TMA load or the data copy.
#[track_caller]
fn copy_front<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: &CopyArgs,
    func: &str,
) {
    let args = match &atom.bound {
        Some(bound) => args.with_bound(bound, func),
        None => args.clone(),
    };
    check_atom_fields(atom.op, &args, func);
    if let Some(n) = args.unroll_factor {
        assert!(n >= 1, "{func}: unroll_factor={n}: the loop is unrolled 1 or more times");
        // A runtime loop indexes memory and coordinates, not register fragments: those copy
        // fully unrolled, as CuTe DSL's do.
        let fragment = src.is_register() || dst.is_register() || args.pred.is_some_and(|p| p.is_register());
        if !fragment {
            let (s, d) = (canonical(src), canonical(dst));
            let pred = args.pred.map(canonical);
            let (sr, dr) = (s.layout().mode(1).shape().product(), d.layout().mode(1).shape().product());
            if let (Some(a), Some(b)) = (sr.as_static(), dr.as_static()) {
                assert_eq!(
                    a,
                    b,
                    "{func}: the source {} and destination {} have different numbers of rest elements (mode 1 \
                     after grouping): {a} and {b}",
                    src.layout(),
                    dst.layout()
                );
            }
            let opts = crate::ir::LoopOptions::unroll(n);
            crate::control::for_range_with(0, sr, 1, opts, |i| {
                let coord: SliceCoord = Tree::Branch(vec![Tree::Leaf(None), Tree::Leaf(Some(IndexExpr::from(i)))]);
                let p = pred.as_ref().map(|p| p.slice(coord.clone()));
                let inner = CopyArgs { pred: p.as_ref(), unroll_factor: None, ..args.clone() };
                copy_body(atom, &s.slice(coord.clone()), &d.slice(coord), &inner, func);
            });
            return;
        }
    }
    copy_body(atom, src, dst, &args, func);
}

/// One copy, its rest modes unrolled: a TMA load, or a data copy.
#[track_caller]
fn copy_body<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: &CopyArgs,
    func: &str,
) {
    if atom.op.is_tma() {
        crate::nvgpu::tma_copy::tma_copy(atom, src, dst, args, func);
        return;
    }
    assert!(
        !S::UNPACKED && !D::UNPACKED,
        "{func}: a u8 tensor stands for unpacked FP4/FP6 values only in a TMA load through an unpacking format, not \
         a {:?} copy",
        atom.op
    );
    for (coords, role) in [(S::COORDINATES, "source"), (D::COORDINATES, "destination")] {
        assert!(
            !coords,
            "{func}: the {role} is a coordinate tensor, which only TMA copies take (as the TMA tensor's \
             partition); a {:?} atom copies data",
            atom.op
        );
    }
    let (src, dst): (Tensor<T>, Tensor<T>) = (src.retyped(), dst.retyped());
    if matches!(
        atom.op,
        CopyOp::BulkS2G { .. } | CopyOp::BulkS2GByteMask { .. } | CopyOp::BulkG2S { .. } | CopyOp::BulkS2S { .. }
    ) {
        bulk_atom_copy(atom.op, &src, &dst, args, func);
        return;
    }
    if let CopyOp::DsmemStore { bits } = atom.op {
        dsmem_store_copy(bits, &src, &dst, args, func);
        return;
    }
    // m10-s2t-copies.
    if let CopyOp::TmemCp { shape, two_cta } = atom.op {
        s2t_copy(shape, two_cta, &src, &dst, args, func);
        return;
    }
    match args.pred {
        Some(pred) => copy_pred(atom, pred, &src, &dst, args.cache_policy, func),
        None => copy_impl(atom, &src, &dst, None, args.cache_policy, func),
    }
}

/// [`copy_if`] and [`copy_with`] with a predicate: the predicate has one
/// flag per source element, and the extents the copy unrolls are static.
#[track_caller]
fn copy_pred<T: Numeric>(
    atom: &impl AsCopyAtom<T>,
    pred: &Tensor<bool>,
    src: &Tensor<T>,
    dst: &Tensor<T>,
    cache_policy: Option<Int64>,
    func: &str,
) {
    let op = atom.copy_atom().op;
    let src_size = copy_extents(src, func, "source", &op);
    let flags = copy_extents(pred, func, "predicate", &op);
    let apps = src_size / atom.copy_atom().num_val_src().max(1);
    assert!(
        flags == src_size || flags == apps,
        "{func}: the predicate tensor {} has {flags} flags, but needs one per source element ({src_size}) or one per \
         application of the {op:?} atom ({apps}, CuTe DSL's (ATOM_REST, REST...) profile)",
        pred.layout()
    );
    copy_impl(atom.copy_atom(), src, dst, Some(pred), cache_policy, func);
}

/// The keyword arguments of CuTe DSL's `cute.autovec_copy(src, dst, *,
/// l1c_evict_priority=)`, for [`autovec_copy_with`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AutovecOptions {
    /// `EVICT_NORMAL` by default, which writes no qualifier. Another
    /// priority makes the copy a [`CopyOp::G2R`] (global memory to
    /// registers) or [`CopyOp::R2G`] (registers to global memory) carrying
    /// it (`ld.global.L1::evict_first`, say; sm_70); the other pairs of
    /// memory spaces refuse it.
    pub l1c_evict_priority: CacheEvictionPriority,
}

/// Element-wise copy with the widest vector loads and stores both tensors
/// allow (CuTe DSL's `autovec_copy`). Every element is copied by
/// instructions unrolled when the kernel is built, so the extents must be
/// static, and the sizes equal.
///
/// As in CuTe DSL, the vector is `max_common_layout(src, dst)`, the longest
/// run of elements contiguous in both tensors, whatever their shapes: a
/// row-major tile and a row-major fragment ([`make_fragment_like`]) copy
/// whole rows. Its width is bounded by both layouts'
/// [`max_alignment`](Layout::max_alignment), by the alignment of both
/// pointers ([`Ptr::max_alignment`], the swizzle's contiguous run counted),
/// and by 256 bits: 32-byte accesses (PTX's `ld.global.v8.b32`, sm_100) to
/// global memory need a 32-byte-aligned buffer
/// ([`param_tensor_aligned`](crate::param_tensor_aligned)) of 32- or 64-bit
/// elements, and LLVM splits them into two 16-byte accesses for older
/// targets; shared memory, per-thread memory and the other element types
/// stop at 128 bits. The offset of each vector is then proved aligned (as
/// with runtime strides and offsets), halving the width until it is. When
/// nothing vectorizes, the copy is scalar ([`basic_copy`]).
///
/// [`make_fragment_like`]: crate::make_fragment_like
/// [`Ptr::max_alignment`]: crate::Ptr::max_alignment
#[track_caller]
pub fn autovec_copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>) {
    autovec_impl(src, dst, AutovecOptions::default(), "autovec_copy");
}

/// [`autovec_copy`] with CuTe DSL's keyword arguments: an L1 eviction
/// priority for the global-memory accesses of a copy between global memory
/// and registers.
#[track_caller]
pub fn autovec_copy_with<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, opts: AutovecOptions) {
    autovec_impl(src, dst, opts, "autovec_copy_with");
}

#[track_caller]
fn autovec_impl<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, opts: AutovecOptions, func: &str) {
    let src_size = unrolled_copy_size(src, func, "source");
    let dst_size = unrolled_copy_size(dst, func, "destination");
    assert_eq!(
        src_size,
        dst_size,
        "{func}: tensors have different sizes: the source {} has {src_size} elements and the destination {} has \
         {dst_size}",
        src.layout(),
        dst.layout()
    );
    let registers = |t: &Tensor<T>| t.is_register() || t.memory_space() == Some(MemSpace::Local);
    let evict = opts.l1c_evict_priority;
    let attrs = if evict == CacheEvictionPriority::EvictNormal {
        None
    } else if src.memory_space() == Some(MemSpace::Global) && registers(dst) {
        Some(true)
    } else if registers(src) && dst.memory_space() == Some(MemSpace::Global) {
        Some(false)
    } else {
        let space = |t: &Tensor<T>| if t.is_register() { AddressSpace::Rmem } else { t.memspace() };
        panic!(
            "{func}: an L1 eviction priority ({}) is for a copy between global memory and registers (CuTe's \
             CopyG2ROp and CopyR2GOp), but this copy is from {} to {}",
            evict.name(),
            space(src),
            space(dst)
        )
    };
    // The 256-bit vectors are plain `ld.global`/`st.global`s of 32- or
    // 64-bit elements; inline PTX with attributes stops at 128 bits.
    let eb = T::DTYPE.storage_bits();
    let wide = attrs.is_none()
        && matches!(eb, 32 | 64)
        && [src, dst].iter().all(|t| t.is_register() || t.memory_space() == Some(MemSpace::Global))
        && [src, dst].iter().any(|t| t.memory_space() == Some(MemSpace::Global));
    let (s, d, v) = if T::DTYPE.is_word_packed() {
        // Word-packed elements (sub-byte, FP6x4) move in whole 32-bit words: the word path, or a
        // refusal.
        let (s, d) = (as_1d(src), as_1d(dst));
        let v = widest_vector::<T>(&s, &d, func);
        (s, d, v)
    } else if T::DTYPE == DType::Bool {
        // Booleans are bytes in memory but `i1` in MLIR, whose vectors LLVM
        // stores bit-packed: they move one at a time.
        (as_1d(src), as_1d(dst), 1)
    } else {
        match autovec_plan(src, dst, if wide { 256 } else { 128 }) {
            Some(plan) => plan,
            None => (as_1d(src), as_1d(dst), 1),
        }
    };
    let bits = v * eb;
    let op = match attrs {
        None => CopyOp::Universal { bits },
        Some(true) => CopyOp::G2R { bits, attrs: G2RAttrs { l1c_evict_priority: evict, ..Default::default() } },
        Some(false) => CopyOp::R2G { bits, attrs: R2GAttrs { l1c_evict_priority: evict, ..Default::default() } },
    };
    copy_impl(&CopyAtom::<T>::new(op), &s, &d, None, None, func);
}

/// CuTe DSL's `autovec_copy` plan: the tensors divided so that mode 0 is
/// one vector of `v` elements, and `v`, or `None` if nothing wider than one
/// element is possible. `max_common_layout(src, dst)` is the common run;
/// its width is bounded by the layouts' and the pointers' alignments and by
/// `cap_bits`, then halved until every vector's offset is proved aligned.
pub(crate) fn autovec_plan<T: Numeric>(
    src: &Tensor<T>,
    dst: &Tensor<T>,
    cap_bits: usize,
) -> Option<(Tensor<T>, Tensor<T>, usize)> {
    let eb = T::DTYPE.storage_bits();
    let common = max_common_layout(src, dst);
    let n = common.size();
    if n <= 1 {
        return None;
    }
    let mut bits = gcd(gcd(src.layout().max_alignment(), dst.layout().max_alignment()), n) * eb;
    for t in [src, dst] {
        if let Some(align) = t.iterator_alignment(eb / 8) {
            bits = gcd(bits, align * 8);
        }
    }
    bits = gcd(bits, cap_bits);
    if bits < eb {
        return None;
    }
    let mut v = bits / eb;
    let divide = |t: &Tensor<T>, v: usize| {
        let by_common = logical_divide(t, Tree::Leaf(Some(common.clone())));
        logical_divide(&by_common, Tree::Leaf(Some(Layout::new(v, 1))))
    };
    while v > 1 {
        let (s, d) = (divide(src, v), divide(dst, v));
        if aligned_vectors(&s, v) && aligned_vectors(&d, v) {
            return Some((s, d, v));
        }
        v /= 2;
    }
    None
}

/// Whether mode 0 of `t`, `v` elements, is contiguous, within the swizzle's
/// runs, and starts at an offset provably a multiple of `v` at every
/// position of mode 1. A register fragment is always.
fn aligned_vectors<T>(t: &Tensor<T>, v: usize) -> bool {
    if t.memory_space().is_none() {
        return true;
    }
    let (m0, rest) = (t.layout().mode(0), t.layout().mode(1));
    if !contiguous(&m0) {
        return false;
    }
    if t.contiguous_run().is_some_and(|run| v > run) {
        return false;
    }
    (0..rest.size()).all(|c| (t.base_offset().clone() + rest.offset_at(c)).is_multiple_of(v))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Element-wise scalar copy (CuTe DSL's `cute.basic_copy`): `dst[i] =
/// src[i]` for every `i`. When both shapes are static, the copy is unrolled
/// when the kernel is built (one scalar atom of the element's width per
/// element), and the sizes must match. Otherwise it is a runtime loop over
/// `size(dst)` that indexes both tensors with the loop counter, as CuTe
/// DSL's `scf.for` does; the sizes are then assumed to match, unchecked.
/// A register fragment is indexed with static coordinates only, so it
/// cannot take part in the loop: copy into a memory tensor
/// ([`make_rmem_tensor`](crate::make_rmem_tensor) for per-thread storage),
/// or copy a tile of static shape.
#[track_caller]
pub fn basic_copy<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>) {
    if !(src.shape().is_static() && dst.shape().is_static()) {
        for (t, role) in [(src.is_register(), "source"), (dst.is_register(), "destination")] {
            if t {
                dynamic_loop_over_fragment("basic_copy", role, src, dst);
            }
        }
        for_range(0, crate::layout::size(dst, &[]), 1, |i| dst.store(i, src.load(i)));
        return;
    }
    assert_eq!(src.size(), dst.size(), "basic_copy: tensors have different sizes");
    let bits = T::DTYPE.bits().max(8);
    copy_impl(&CopyAtom::<T>::new(CopyOp::Universal { bits }), &as_1d(src), &as_1d(dst), None, None, "basic_copy");
}

/// Panic: the `role` tensor of `func`, a copy of runtime size over `src`
/// and `dst`, is a register fragment, which a runtime loop cannot index.
#[track_caller]
fn dynamic_loop_over_fragment<S, D>(func: &str, role: &str, src: &Tensor<S>, dst: &Tensor<D>) -> ! {
    panic!(
        "{func}: the {role} tensor is a register fragment, but the source {} and destination {} do not both have \
         static shapes, so {func} copies with a runtime loop over size(dst) that indexes both with the loop counter, \
         and a register fragment is indexed with static coordinates only: copy through a memory tensor \
         (make_rmem_tensor for per-thread storage), or copy one tile of static shape",
        src.layout(),
        dst.layout()
    )
}

/// Element-wise predicated copy (CuTe DSL's `cute.basic_copy_if`): `dst[i]
/// = src[i]` for every `i` where `pred[i]` is true; elsewhere `dst` keeps
/// its value. The element types may differ but must have the same width
/// (the bits are copied, a `bool` counting as 8 bits). When the three
/// shapes are static, the copy is unrolled when the kernel is built, and
/// the sizes must match: with one element type it is [`copy_if`] with a
/// one-element scalar atom. Otherwise it is a runtime loop over `size(dst)`
/// with a branch per element, as CuTe DSL's `scf.for` and `scf.if` are; a
/// register fragment cannot take part in that loop (see [`basic_copy`]),
/// so the predicate is then a memory tensor such as one from
/// [`make_rmem_tensor`](crate::make_rmem_tensor).
#[track_caller]
pub fn basic_copy_if<S: Numeric, D: Numeric>(pred: &Tensor<bool>, src: &Tensor<S>, dst: &Tensor<D>) {
    let (sb, db) = (S::DTYPE.bits().max(8), D::DTYPE.bits().max(8));
    assert!(
        sb == db,
        "basic_copy_if: the source elements ({}, {sb} bits) and the destination elements ({}, {db} bits) differ in \
         width; basic_copy_if copies the bits of elements of the same width",
        S::DTYPE,
        D::DTYPE
    );
    let same = S::DTYPE == D::DTYPE;
    let value = |i: crate::layout::Coord| -> Value<D> {
        let v = src.load(i);
        if same { Value::from_id(v.checked()) } else { v.bitcast::<D>() }
    };
    if !(pred.shape().is_static() && src.shape().is_static() && dst.shape().is_static()) {
        for (t, role) in
            [(pred.is_register(), "predicate"), (src.is_register(), "source"), (dst.is_register(), "destination")]
        {
            if t {
                dynamic_loop_over_fragment("basic_copy_if", role, src, dst);
            }
        }
        for_range(0, crate::layout::size(dst, &[]), 1, |i| {
            if_(pred.load(i), || dst.store(i, value(i.into())));
        });
        return;
    }
    let n = src.size();
    assert_eq!(n, dst.size(), "basic_copy_if: the source and destination tensors have different sizes");
    assert_eq!(n, pred.size(), "basic_copy_if: the predicate tensor must have one flag per source element");
    if same {
        let dst_s: Tensor<S> = dst.retyped();
        let atom = CopyAtom::<S>::new(CopyOp::Universal { bits: sb });
        copy_pred(&atom, &as_1d(pred), &as_1d(src), &as_1d(&dst_s), None, "basic_copy_if");
        return;
    }
    for i in 0..n {
        let p = pred.load(i);
        if dst.is_register() {
            let slot = dst.element_offset(&i.into());
            let old = old_register_value(dst, &slot);
            let v = if_else_yield(p, || value(i.into()), || old);
            dst.store(i, v);
        } else {
            if_(p, || dst.store(i, value(i.into())));
        }
    }
}

/// The number of elements of `t`, the `role` tensor (source, destination,
/// predicate) of `func`, which copies them one atom-sized chunk of mode 0
/// at a time, unrolled when the kernel is built. Panics, naming the leaf,
/// if mode 0 (each thread's chunk of values) or a mode after it (the rest
/// modes, iterated) has a runtime extent.
#[track_caller]
fn copy_extents<T>(t: &Tensor<T>, func: &str, role: &str, op: &CopyOp) -> usize {
    let layout = t.layout();
    if let Some((path, leaf)) = layout.shape().first_dynamic() {
        // Mode 0 comes first depth first (and a leaf shape's path is `[0]`).
        let message = if path.first() == Some(&0) {
            format!(
                "{func}: the {role} tensor {layout} has the dynamic extent {leaf:?} at mode {path:?}, in its mode 0, \
                 the chunk of values each thread copies: {func} splits mode 0 into the {op:?} atom's chunks when the \
                 kernel is built, so it must be static (partition a tile with a tiled copy, whose per-thread chunks \
                 are static)"
            )
        } else {
            format!(
                "{func}: the {role} tensor {layout} has the dynamic extent {leaf:?} at mode {path:?}, a rest mode that \
                 {func} unrolls when the kernel is built (one {op:?} copy for each of its elements), so it must be \
                 static: for a runtime number of tiles, write a for_range over that mode and copy one slice of it per \
                 iteration"
            )
        };
        panic!("{message}");
    }
    t.size()
}

/// The number of elements of `t`, the `role` tensor of `func`, which copies
/// every element by instructions unrolled when the kernel is built: panics,
/// naming the leaf, if an extent is a runtime value.
#[track_caller]
fn unrolled_copy_size<T>(t: &Tensor<T>, func: &str, role: &str) -> usize {
    let layout = t.layout();
    if let Some((path, leaf)) = layout.shape().first_dynamic() {
        panic!(
            "{func}: the {role} tensor {layout} has the dynamic extent {leaf:?} at mode {path:?}, but {func} copies \
             every element with instructions unrolled when the kernel is built, so the extents must be static: for a \
             runtime number of tiles, write a for_range over the rest mode and copy one tile per iteration"
        );
    }
    t.size()
}

/// `cp.async`, `ldmatrix` and the bulk copies address buffers: a tensor over
/// a raw pointer ([`make_ptr`](crate::make_ptr)) is refused with a clear
/// message.
#[track_caller]
fn refuse_raw<T>(src: &Tensor<T>, dst: &Tensor<T>, what: &str) {
    for (t, role) in [(src, "source"), (dst, "destination")] {
        if let Some(MemSpace::Raw(space)) = t.memory_space() {
            panic!(
                "{what}: the {role} is a tensor over a raw {space} pointer (make_ptr, inttoptr, tospace, Ptr::align \
                 or a masked pointer), which {what} does not take yet: it addresses buffers (a parameter, \
                 alloc_smem or make_smem_tensor); raw pointers take loads, stores, atomics and the copies built on \
                 them"
            );
        }
    }
}

fn as_1d<T>(t: &Tensor<T>) -> Tensor<T> {
    let flat = t.layout().compose(&Layout::new(t.size(), 1));
    crate::layout::Tileable::with_layout(t, Layout::from_modes([flat]))
}

/// The widest vector (in elements, at most 128 bits) every chunk of both
/// one-mode tensors allows. Word-packed elements (sub-byte, FP6x4) move in whole 32-bit words,
/// so for them the narrowest vector is one word, which the tensors must
/// allow.
#[track_caller]
fn widest_vector<T: Numeric>(src: &Tensor<T>, dst: &Tensor<T>, func: &str) -> usize {
    let bits = T::DTYPE.storage_bits();
    let narrowest = if T::DTYPE.is_word_packed() { crate::tensor::word_group(T::DTYPE).0 } else { 1 };
    // The widest power-of-two multiple of the narrowest within 128 bits.
    let mut v = narrowest;
    while 2 * v * bits <= 128 {
        v *= 2;
    }
    while v > narrowest && !(vectorizable(src, v) && vectorizable(dst, v)) {
        v /= 2;
    }
    assert!(
        narrowest == 1 || (vectorizable(src, v) && vectorizable(dst, v)),
        "{func}: sub-byte tensors are copied in whole 32-bit words ({v} {} values), but {} and {} are not \
         contiguous and aligned in runs of {v}",
        T::DTYPE.cute_name(),
        src.layout(),
        dst.layout()
    );
    v
}

/// Whether every `v`-element chunk of mode 0 is contiguous and aligned.
fn vectorizable<T>(t: &Tensor<T>, v: usize) -> bool {
    if t.is_register() {
        return true;
    }
    let m0 = t.layout().mode(0);
    if !m0.size().is_multiple_of(v) {
        return false;
    }
    if t.contiguous_run().is_some_and(|run| v > run) {
        return false;
    }
    let rest = t.size() / m0.size();
    let all = t.layout();
    (0..rest).all(|r| {
        (0..m0.size() / v).all(|c| {
            let start = all.offset_at(r * m0.size() + c * v);
            (0..v).all(|i| all.offset_at(r * m0.size() + c * v + i) == start.clone() + i)
                && (t.base_offset().clone() + start).is_multiple_of(v)
        })
    })
}

/// Split mode 0 of `layout` into `v`-element atom chunks: returns the
/// per-chunk element layout and the static offset of every chunk.
#[track_caller]
fn chunks(layout: &Layout, v: usize, what: &str) -> (Layout, Vec<IndexExpr>) {
    let m0 = layout.mode(0);
    assert_eq!(m0.size() % v, 0, "copy: {what} mode 0 ({m0}) is not a multiple of the atom's {v} values");
    let divided = m0.logical_divide(&Tree::Leaf(Some(Layout::new(v, 1))));
    let atom = divided.mode(0);
    let mut rest_modes = vec![divided.mode(1)];
    rest_modes.extend((1..layout.rank()).map(|i| layout.mode(i)));
    let rest = Layout::from_modes(rest_modes);
    (atom, (0..rest.size()).map(|c| rest.offset_at(c)).collect())
}

fn contiguous(atom: &Layout) -> bool {
    atom.size() == 1 || atom.coalesce() == Layout::new(atom.size(), 1)
}

#[track_caller]
fn check_vector_access<T>(t: &Tensor<T>, atom: &Layout, off: &IndexExpr, what: &str, op: &CopyOp) {
    let v = atom.size();
    if v == 1 {
        return;
    }
    assert!(
        contiguous(atom),
        "copy: {op:?} needs {v} contiguous {what} elements per thread, but they are laid out as {atom}"
    );
    if let Some(sw) = t.swizzle() {
        assert!(v <= sw.contiguous_run(), "copy: {v}-element vectors are broken up by swizzle {sw}");
    }
    if let Some(inner) = t.inner_layout() {
        panic!(
            "copy: {op:?} moves {v} contiguous {what} elements per thread, but the tensor is over a composed layout \
             with the inner layout {inner}, whose elements are not known to be contiguous; copy element by element"
        );
    }
    assert!(
        off.is_multiple_of(v),
        "copy: cannot prove that the {what} offset {off:?} is aligned to {v} elements for {op:?}{}",
        builder::var_value_note(off)
    );
}

/// `func` is the public function that copies, as a refusal names it.
#[track_caller]
fn copy_impl<T: Numeric>(
    atom: &CopyAtom<T>,
    src: &Tensor<T>,
    dst: &Tensor<T>,
    pred: Option<&Tensor<bool>>,
    cache_policy: Option<Int64>,
    func: &str,
) {
    let loc = Location::caller();
    assert!(!atom.op.is_tma(), "{func}: TMA atoms ({:?}) copy through the TMA path, not element by element", atom.op);
    assert!(
        !matches!(atom.op, CopyOp::TmemLoadRed { .. } | CopyOp::TmemLoadSpCompress { .. }),
        "{func}: a TMEM load-reduce or sparse-compress atom ({:?}) has several destinations: issue a load-reduce \
         with copy_atom_call(atom, src, &[&data, &stat]) and a sparse-compress with copy_atom_call_sp",
        atom.op
    );
    if matches!(
        atom.op,
        CopyOp::BulkS2G { .. } | CopyOp::BulkS2GByteMask { .. } | CopyOp::BulkG2S { .. } | CopyOp::BulkS2S { .. }
    ) {
        let args = CopyArgs { pred, cache_policy, ..CopyArgs::default() };
        bulk_atom_copy(atom.op, src, dst, &args, func);
        return;
    }
    let auto = match atom.op {
        CopyOp::G2R { bits: 0, attrs } => Some((attrs.memory_order, CopyOp::G2R { bits: 0, attrs })),
        CopyOp::R2G { bits: 0, attrs } => Some((attrs.memory_order, CopyOp::R2G { bits: 0, attrs })),
        CopyOp::S2R { bits: 0, attrs } => Some((attrs.memory_order, CopyOp::S2R { bits: 0, attrs })),
        CopyOp::R2S { bits: 0, attrs } => Some((attrs.memory_order, CopyOp::R2S { bits: 0, attrs })),
        CopyOp::G2R { .. } | CopyOp::R2G { .. } => None,
        CopyOp::S2R { .. } | CopyOp::R2S { .. } => {
            assert!(cache_policy.is_none(), "copy: a cache policy is for global memory, not {:?}", atom.op);
            None
        }
        _ => {
            assert!(
                cache_policy.is_none(),
                "copy: a cache policy is for CopyG2ROp and CopyR2GOp atoms (CopyOp::G2R, CopyOp::R2G), not {:?}",
                atom.op
            );
            None
        }
    };
    // `num_bits_per_copy = 0`: the widest vector both tensors allow, one
    // element at a time under a predicate (which each element has) and for
    // MMIO (which is scalar).
    if let Some((order, op)) = auto
        && pred.is_none()
        && order != MemoryOrder::Mmio
    {
        assert_eq!(
            unrolled_copy_size(src, func, "source"),
            unrolled_copy_size(dst, func, "destination"),
            "copy: tensors have different sizes"
        );
        let (bits, (src1, dst1, v)) = if T::DTYPE.is_word_packed() {
            // Word-packed elements (sub-byte, FP6x4) move in whole 32-bit words.
            let (s, d) = (as_1d(src), as_1d(dst));
            let v = widest_vector::<T>(&s, &d, func);
            (T::DTYPE.storage_bits(), (s, d, v))
        } else {
            (T::DTYPE.bits().max(8), autovec_plan(src, dst, 128).unwrap_or_else(|| (as_1d(src), as_1d(dst), 1)))
        };
        let op = match op {
            CopyOp::G2R { attrs, .. } => CopyOp::G2R { bits: v * bits, attrs },
            CopyOp::R2G { attrs, .. } => CopyOp::R2G { bits: v * bits, attrs },
            CopyOp::S2R { attrs, .. } => CopyOp::S2R { bits: v * bits, attrs },
            CopyOp::R2S { attrs, .. } => CopyOp::R2S { bits: v * bits, attrs },
            _ => unreachable!("G2R, R2G, S2R or R2S"),
        };
        copy_impl(&CopyAtom::<T>::new(op), &src1, &dst1, None, cache_policy, func);
        return;
    }
    let (sv, dv) = (atom.num_val_src(), atom.num_val_dst());
    copy_extents(src, func, "source", &atom.op);
    copy_extents(dst, func, "destination", &atom.op);
    let (src_atom, src_chunks) = chunks(src.layout(), sv, "source");
    let (dst_atom, dst_chunks) = chunks(dst.layout(), dv, "destination");
    assert_eq!(
        src_chunks.len(),
        dst_chunks.len(),
        "copy: source {:?} and destination {:?} split into different numbers of {:?} chunks",
        src,
        dst,
        atom.op
    );
    // One flag per source element (the first of each chunk is consulted), or one per
    // application, in the order of the chunks.
    let pred_chunks = pred.map(|p| {
        if p.size() == src_chunks.len() && p.size() != src.size() {
            (0..p.size()).map(|a| p.layout().offset_at(a)).collect()
        } else {
            let (_, offs) = chunks(p.layout(), sv, "predicate");
            offs
        }
    });

    for (c, (so, d_o)) in src_chunks.iter().zip(&dst_chunks).enumerate() {
        let s_base = src.base_offset().clone() + so;
        let d_base = dst.base_offset().clone() + d_o;
        let p = match (pred, &pred_chunks) {
            (Some(p), Some(offs)) => Some(p.load_at(p.base_offset().clone() + &offs[c])),
            _ => None,
        };
        emit_chunk(atom, src, &src_atom, s_base, dst, &dst_atom, d_base, p, cache_policy, loc);
    }
}

/// A copy with a bulk-copy atom ([`CopyOp::BulkS2G`], [`CopyOp::BulkG2S`],
/// [`CopyOp::BulkS2S`]): one bulk copy per `bits`-bit chunk of mode 0 (the
/// whole of mode 0 for 0), with the atom's fields from `args`.
#[track_caller]
fn bulk_atom_copy<T: Numeric>(op: CopyOp, src: &Tensor<T>, dst: &Tensor<T>, args: &CopyArgs, func: &str) {
    let loc = Location::caller();
    let (name, bits, direction, from, to) = match op {
        CopyOp::BulkS2G { bits } => ("CopyBulkS2GOp", bits, BulkDirection::S2G, MemSpace::Shared, MemSpace::Global),
        CopyOp::BulkS2GByteMask { bits } => {
            ("CopyBulkS2GByteMaskOp", bits, BulkDirection::S2G, MemSpace::Shared, MemSpace::Global)
        }
        CopyOp::BulkG2S { bits, multicast: false } => {
            ("CopyBulkG2SOp", bits, BulkDirection::G2S, MemSpace::Global, MemSpace::Shared)
        }
        CopyOp::BulkG2S { bits, multicast: true } => {
            ("CopyBulkG2SMulticastOp", bits, BulkDirection::G2S, MemSpace::Global, MemSpace::Shared)
        }
        CopyOp::BulkS2S { bits } => ("CopyBulkS2SOp", bits, BulkDirection::S2S, MemSpace::Shared, MemSpace::Shared),
        _ => unreachable!("a bulk-copy atom"),
    };
    assert!(args.pred.is_none(), "{func}: {name} copies cannot be predicated");
    let space = |t: &Tensor<T>| if t.is_register() { AddressSpace::Rmem } else { t.memspace() };
    assert!(
        src.memory_space() == Some(from) && dst.memory_space() == Some(to),
        "{func}: {name} copies {} memory to {} memory, not {} to {}",
        AddressSpace::from(from),
        AddressSpace::from(to),
        space(src),
        space(dst)
    );
    for (t, role) in [(src, "source"), (dst, "destination")] {
        if let Some(sw) = t.swizzle() {
            panic!(
                "{func}: {name} copies plain shared memory: a bulk copy does not apply the swizzle {sw} of the {role} \
                 {}",
                t.layout()
            );
        }
        if let Some(inner) = t.inner_layout() {
            panic!("{func}: {name} copies plain memory, not the {role} over the inner layout {inner}");
        }
    }
    let n = copy_extents(src, func, "source", &op);
    assert_eq!(n, copy_extents(dst, func, "destination", &op), "{func}: tensors have different sizes");
    let v = if bits == 0 { src.layout().mode(0).size() } else { bits / T::DTYPE.bits() };
    let (src_atom, src_chunks) = chunks(src.layout(), v, "source");
    let (dst_atom, dst_chunks) = chunks(dst.layout(), v, "destination");
    assert_eq!(
        src_chunks.len(),
        dst_chunks.len(),
        "{func}: source and destination split into different numbers of bulk copies"
    );
    for (atom, role) in [(&src_atom, "source"), (&dst_atom, "destination")] {
        assert!(
            contiguous(atom),
            "{func}: a {name} copy moves {v} contiguous elements, but the {role}'s are laid out as {atom}"
        );
    }
    let bytes = v * T::DTYPE.bytes();
    assert!(
        bytes.is_multiple_of(16),
        "{func}: a {name} copy of {v} {} elements is {bytes} bytes, not a multiple of 16",
        T::DTYPE
    );
    let mbar = match &args.mbar_ptr {
        Some(m) => {
            assert!(
                m.memory_space() == MemSpace::Shared,
                "{func}: mbar_ptr points to {:?} memory; an mbarrier lives in shared memory",
                m.memory_space()
            );
            Some(m.parts())
        }
        None => None,
    };
    let extras = crate::arch::BulkExtras {
        cache_policy: args.cache_policy.map(|p| p.checked()),
        mcast_mask: args.mcast_mask.map(|m| m.checked()),
        peer: args.cta_rank.map(|r| r.checked()),
        byte_mask: args.byte_mask.map(|m| m.checked()),
    };
    let (s, d) = (src.ptr(), dst.ptr());
    for (so, d_o) in src_chunks.iter().zip(&dst_chunks) {
        let completion = || match &mbar {
            Some((buffer, offset)) => {
                BulkCompletion::Mbarrier { buffer: *buffer, index: crate::tensor::index_operand(offset.clone()) }
            }
            None => BulkCompletion::BulkGroup,
        };
        crate::arch::bulk_copy(
            &s.offset(so.clone()),
            &d.offset(d_o.clone()),
            bytes as i32,
            direction,
            completion,
            extras,
            loc,
        );
    }
}

/// [`prefetch_atom`](crate::prefetch_atom) with a bulk-copy atom: one
/// `cp.async.bulk.prefetch.L2.global` per `bits`-bit chunk of mode 0 of the
/// global tensor `src` (the whole of mode 0 for 0).
#[track_caller]
pub(crate) fn bulk_prefetch<T: Numeric>(src: &Tensor<T>, bits: usize, loc: crate::ir::Loc, func: &str) {
    assert!(
        src.memory_space() == Some(MemSpace::Global),
        "{func}: a bulk prefetch reads global memory, not {}",
        src.memspace()
    );
    let op = CopyOp::BulkG2S { bits, multicast: false };
    copy_extents(src, func, "source", &op);
    let v = if bits == 0 { src.layout().mode(0).size() } else { bits / T::DTYPE.bits() };
    let (atom, offsets) = chunks(src.layout(), v, "source");
    assert!(
        contiguous(&atom),
        "{func}: a bulk prefetch reads {v} contiguous elements, but the source's are laid out as {atom}"
    );
    let bytes = v * T::DTYPE.bytes();
    assert!(bytes.is_multiple_of(16), "{func}: a bulk prefetch of {bytes} bytes is not a multiple of 16");
    let p = src.ptr();
    for o in offsets {
        let (buffer, offset) = p.offset(o).parts();
        let index = crate::tensor::index_operand(offset);
        let bytes = crate::value::IntoValue::<i32>::into_value(bytes as i32).checked();
        builder::with(|b| b.push(Op::BulkPrefetch { buffer, index, elem: T::DTYPE, bytes }, loc));
    }
}

fn physical(t: &Tensor<impl Sized>, linear: IndexExpr) -> IndexExpr {
    t.physical_offset(linear)
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn emit_chunk<T: Numeric>(
    atom: &CopyAtom<T>,
    src: &Tensor<T>,
    src_atom: &Layout,
    s_base: IndexExpr,
    dst: &Tensor<T>,
    dst_atom: &Layout,
    d_base: IndexExpr,
    pred: Option<Value<bool>>,
    cache_policy: Option<Int64>,
    loc: crate::ir::Loc,
) {
    let op = atom.op;
    match op {
        CopyOp::Universal { .. }
        | CopyOp::G2R { .. }
        | CopyOp::R2G { .. }
        | CopyOp::S2R { .. }
        | CopyOp::R2S { .. } => {
            // The loads of a G2R atom and the stores of an R2G one carry its
            // memory attributes, located at the user's copy. The register
            // side is a fragment or a `make_rmem_tensor` (per-thread memory,
            // CuTe DSL's register tensor), whose accesses stay plain.
            let _at = builder::LocationScope::new();
            let policy = cache_policy.map(|p| p.checked());
            let registers = |t: &Tensor<T>| t.is_register() || t.memory_space() == Some(MemSpace::Local);
            let storage = |t: &Tensor<T>| match t.memory_space() {
                Some(space) => format!("{space:?} memory"),
                None => "an identity tensor".to_string(),
            };
            let (load_attrs, store_attrs) = match op {
                CopyOp::G2R { attrs, .. } => {
                    assert_eq!(src.memory_space(), Some(MemSpace::Global), "copy: CopyG2ROp reads global memory");
                    assert!(
                        registers(dst),
                        "copy: CopyG2ROp writes registers (a fragment or make_rmem_tensor), not {}",
                        storage(dst)
                    );
                    (MemAttrs { cache_policy: policy, ..attrs.mem_attrs() }, MemAttrs::default())
                }
                CopyOp::R2G { attrs, .. } => {
                    assert!(
                        registers(src),
                        "copy: CopyR2GOp reads registers (a fragment or make_rmem_tensor), not {}",
                        storage(src)
                    );
                    assert_eq!(dst.memory_space(), Some(MemSpace::Global), "copy: CopyR2GOp writes global memory");
                    (MemAttrs::default(), MemAttrs { cache_policy: policy, ..attrs.mem_attrs() })
                }
                CopyOp::S2R { attrs, .. } => {
                    check_shared_side(src, attrs.shared_space, "CopyS2ROp", "reads");
                    assert!(
                        registers(dst),
                        "copy: CopyS2ROp writes registers (a fragment or make_rmem_tensor), not {}",
                        storage(dst)
                    );
                    (attrs.mem_attrs(), MemAttrs::default())
                }
                CopyOp::R2S { attrs, .. } => {
                    assert!(
                        registers(src),
                        "copy: CopyR2SOp reads registers (a fragment or make_rmem_tensor), not {}",
                        storage(src)
                    );
                    check_shared_side(dst, attrs.shared_space, "CopyR2SOp", "writes");
                    (MemAttrs::default(), attrs.mem_attrs())
                }
                _ => (MemAttrs::default(), MemAttrs::default()),
            };
            // A tensor in another CTA's shared memory: its accesses
            // carry the peer.
            let (load_attrs, store_attrs) = (
                MemAttrs { peer: src.buffer_and_peer().and_then(|(_, p)| p), ..load_attrs },
                MemAttrs { peer: dst.buffer_and_peer().and_then(|(_, p)| p), ..store_attrs },
            );
            let v = src_atom.size();
            // Addresses are computed before any predicate so that branches
            // only contain the memory access itself.
            let src_index = src.memory_space().map(|_| {
                check_vector_access(src, src_atom, &s_base, "source", &op);
                crate::tensor::access_index::<T>(physical(src, s_base.clone()), v, "copy source")
            });
            let dst_index = dst.memory_space().map(|_| {
                check_vector_access(dst, dst_atom, &d_base, "destination", &op);
                crate::tensor::access_index::<T>(physical(dst, d_base.clone()), v, "copy destination")
            });
            let read = || -> Vec<ValueId> {
                match src_index {
                    Some(i) => load_vector_with::<T>(src.buffer_and_peer().unwrap().0, i, v, load_attrs),
                    None => (0..v).map(|k| src.load_at(s_base.clone() + src_atom.call(k)).checked()).collect(),
                }
            };
            match dst_index {
                Some(i) => {
                    let write = || store_vector_with(dst.buffer_and_peer().unwrap().0, i, &read(), store_attrs);
                    match pred {
                        Some(p) => if_(p, write),
                        None => write(),
                    }
                }
                None => {
                    let slots: Vec<IndexExpr> = (0..v).map(|k| d_base.clone() + dst_atom.call(k)).collect();
                    let values: Vec<Value<T>> = match pred {
                        None => read().into_iter().map(Value::from_id).collect(),
                        Some(p) => {
                            let old: Vec<Value<T>> = slots.iter().map(|s| old_register_value(dst, s)).collect();
                            if_else_yield(p, || read().into_iter().map(Value::from_id).collect::<Vec<_>>(), || old)
                        }
                    };
                    for (s, v) in slots.into_iter().zip(values) {
                        dst.store_at(s, v);
                    }
                }
            }
        }
        CopyOp::CpAsync { bypass_l1, .. } => {
            refuse_raw(src, dst, "cp.async");
            assert_eq!(src.memory_space(), Some(MemSpace::Global), "cp.async reads global memory");
            assert_eq!(dst.memory_space(), Some(MemSpace::Shared), "cp.async writes shared memory");
            check_vector_access(src, src_atom, &s_base, "source", &op);
            check_vector_access(dst, dst_atom, &d_base, "destination", &op);
            let v = src_atom.size();
            let src_index = crate::tensor::access_index::<T>(physical(src, s_base), v, "cp.async source");
            let dst_index = crate::tensor::access_index::<T>(physical(dst, d_base), v, "cp.async destination");
            let pred = pred.map(|p| p.checked());
            let (s, d) = (src.buffer().unwrap(), dst.buffer().unwrap());
            // Sub-byte elements move as the 32-bit words they are packed in.
            let (elem, elems) =
                if T::DTYPE.is_word_packed() { (DType::U32, v * T::DTYPE.bits() / 32) } else { (T::DTYPE, v) };
            builder::with(|b| {
                b.push(
                    Op::CpAsync { src: s, src_index, dst: d, dst_index, elem, elems: elems as u32, bypass_l1, pred },
                    loc,
                )
            });
        }
        CopyOp::TmemLoad { shape, num, pack } => {
            tmem_load_chunk(atom, src, src_atom, s_base, dst, dst_atom, d_base, pred, shape, num, pack, None, loc)
        }
        CopyOp::TmemLoadRed { .. } | CopyOp::TmemLoadSpCompress { .. } => {
            unreachable!(
                "refused by copy_impl: copy_atom_call issues tcgen05.ld.red and copy_atom_call_sp the spcompress forms"
            )
        }
        CopyOp::TmemStore { shape, num, unpack } => {
            tmem_store_chunk(atom, src, src_atom, s_base, dst, dst_atom, d_base, pred, shape, num, unpack, loc)
        }
        CopyOp::DsmemStore { .. } => unreachable!("CopyDsmemStoreOp copies are issued by dsmem_store_copy"),
        CopyOp::TmemCp { .. } => unreachable!("S2T copies are issued by s2t_copy"),
        CopyOp::BulkS2G { .. } | CopyOp::BulkS2GByteMask { .. } | CopyOp::BulkG2S { .. } | CopyOp::BulkS2S { .. } => {
            unreachable!("bulk copies are issued by bulk_atom_copy")
        }
        CopyOp::TmaLoad { .. }
        | CopyOp::TmaStore
        | CopyOp::TmaReduce(_)
        | CopyOp::TmaLoadMode { .. }
        | CopyOp::TmaStoreMode { .. } => unreachable!("refused by copy_impl"),
        CopyOp::LdMatrix { num, trans } => {
            refuse_raw(src, dst, "ldmatrix");
            assert!(pred.is_none(), "ldmatrix copies cannot be predicated");
            assert_eq!(src.memory_space(), Some(MemSpace::Shared), "ldmatrix reads shared memory");
            assert!(dst.is_register(), "ldmatrix writes registers");
            check_vector_access(src, src_atom, &s_base, "source", &op);
            let index = crate::tensor::index_operand(physical(src, s_base));
            let buffer = src.buffer().unwrap();
            let results = builder::with(|b| {
                let results: Vec<ValueId> = (0..2 * num).map(|_| b.new_value(T::DTYPE)).collect();
                let form = LdMatrixForm::M8n8B16;
                b.push(Op::LdMatrix { results: results.clone(), buffer, index, num: num as u32, trans, form }, loc);
                results
            });
            for (i, r) in results.into_iter().enumerate() {
                dst.store_at(d_base.clone() + dst_atom.call(i), Value::from_id(r));
            }
        }
        CopyOp::StMatrix { num, trans } | CopyOp::StMatrix16x8x8b { num, trans } => {
            let form =
                if matches!(op, CopyOp::StMatrix { .. }) { StMatrixForm::M8n8B16 } else { StMatrixForm::M16n8B8 };
            refuse_raw(src, dst, "stmatrix");
            assert!(pred.is_none(), "stmatrix copies cannot be predicated");
            assert!(src.is_register(), "stmatrix reads registers (a fragment)");
            assert_eq!(dst.memory_space(), Some(MemSpace::Shared), "stmatrix writes shared memory");
            check_vector_access(dst, dst_atom, &d_base, "destination", &op);
            // The index counts 16-bit elements, or bytes (32-bit elements count twice).
            let wide = if matches!(op, CopyOp::StMatrix { .. }) && T::DTYPE.bits() == 32 { 2 } else { 1 };
            let index = crate::tensor::index_operand(physical(dst, d_base) * wide);
            let buffer = dst.buffer().unwrap();
            let per = 32 / T::DTYPE.bits();
            let elems: Vec<ValueId> =
                (0..per * num).map(|k| src.load_at(s_base.clone() + src_atom.call(k)).checked()).collect();
            builder::with(|b| {
                // One 32-bit register per matrix, its first element in the low bits.
                let values: Vec<ValueId> =
                    elems
                        .chunks(per)
                        .map(|c| {
                            if per == 1 {
                                b.bitcast(c[0], DType::I32, loc)
                            } else {
                                b.pack(c.to_vec(), DType::I32, loc)
                            }
                        })
                        .collect();
                b.push(Op::StMatrix { values, buffer, index, num: num as u32, trans, form }, loc);
            });
        }
        CopyOp::LdMatrix8x16x8b { num, unpack_bits: Some(bits) } => {
            let form = LdMatrixForm::M8n16B8 { unpack: bits as u8 };
            ldmatrix_b8(src, src_atom, s_base, dst, dst_atom, d_base, pred, op, num, false, form, loc, |w, q| (w, q));
        }
        CopyOp::LdMatrix16x16x8b { num, unpack_bits, .. } => {
            let form = LdMatrixForm::M16n16B8 { unpack: unpack_bits.map(|b| b as u8) };
            ldmatrix_b8(src, src_atom, s_base, dst, dst_atom, d_base, pred, op, num, true, form, loc, |w, q| (w, q));
        }
        CopyOp::LdMatrix16x8x8b { num, unpack_bits, .. } => {
            // ldmatrix.m16n16.x{num / 2}; then, as CuTe C++'s
            // SM100_U8x{8,16}_LDSM_T, destination register `w` byte `q` is
            // byte `2 (w % 2) + q % 2` of register `2 (w / 2) + q / 2`.
            let form = LdMatrixForm::M16n16B8 { unpack: unpack_bits.map(|b| b as u8) };
            let permute = |w: usize, q: usize| (2 * (w / 2) + q / 2, 2 * (w % 2) + q % 2);
            ldmatrix_b8(src, src_atom, s_base, dst, dst_atom, d_base, pred, op, num / 2, true, form, loc, permute);
        }
        CopyOp::LdMatrix8x16x8b { num, unpack_bits: None } => {
            assert!(pred.is_none(), "ldmatrix copies cannot be predicated");
            assert_eq!(src.memory_space(), Some(MemSpace::Shared), "ldmatrix reads shared memory");
            assert!(dst.is_register(), "ldmatrix writes registers");
            check_vector_access(src, src_atom, &s_base, "source", &op);
            // ldmatrix.m8n8.b16 on a 16-bit view of the bytes: the row
            // offset is a multiple of 16 bytes (checked above), so half of
            // it indexes the view; each 16-bit result holds two bytes, the
            // first in its low half.
            let index = crate::tensor::index_operand(physical(src, s_base) / 2);
            let buffer = src.buffer().unwrap();
            let results = builder::with(|b| {
                let halves: Vec<ValueId> = (0..2 * num).map(|_| b.new_value(DType::U16)).collect();
                let form = LdMatrixForm::M8n8B16;
                let op = Op::LdMatrix { results: halves.clone(), buffer, index, num: num as u32, trans: false, form };
                b.push(op, loc);
                halves.into_iter().flat_map(|h| b.unpack(h, T::DTYPE, 2, loc)).collect::<Vec<_>>()
            });
            for (i, r) in results.into_iter().enumerate() {
                dst.store_at(d_base.clone() + dst_atom.call(i), Value::from_id(r));
            }
        }
    }
}

/// Whether `t` is a tensor in tensor memory.
fn is_tmem<T>(t: &Tensor<T>) -> bool {
    t.memory_space() == Some(MemSpace::Raw(AddressSpace::Tmem))
}

/// The TMEM address (an Int32) of element offset `offset` of the TMEM
/// tensor `t`, checking that its chunk `atom` has the layout the TMEM atom
/// `op` needs (CuTe C++'s TMEM `copy_unpack` assertion).
#[track_caller]
fn tmem_chunk_address<T: Numeric>(t: &Tensor<T>, atom: &Layout, offset: IndexExpr, op: CopyOp, role: &str) -> ValueId {
    let want = crate::nvgpu::tcgen05::copy::tmem_val_layout(op, T::DTYPE.storage_bits());
    assert!(
        atom.coalesce() == want,
        "copy: the TMEM {role}'s chunk has the layout {atom}, but the {op:?} atom addresses TMEM as {want} (TMEM lanes \
         at a stride of {}, one warp's lane quadrant): partition the TMEM tensor with make_tmem_copy",
        (1usize << 21) / T::DTYPE.storage_bits()
    );
    let ptr = Ptr::<T> { offset: physical(t, offset), ..t.ptr() };
    ptr.toint_i32().checked()
}

/// One `tcgen05.ld` of a TMEM load atom: the chunk at `s_base` of the TMEM
/// tensor `src` into the registers of `dst` (a register fragment).
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn tmem_load_chunk<T: Numeric>(
    atom: &CopyAtom<T>,
    src: &Tensor<T>,
    src_atom: &Layout,
    s_base: IndexExpr,
    dst: &Tensor<T>,
    dst_atom: &Layout,
    d_base: IndexExpr,
    pred: Option<Value<bool>>,
    shape: crate::ir::TmemShape,
    num: u32,
    pack: bool,
    stat: Option<&Tensor<T>>,
    loc: crate::ir::Loc,
) {
    assert!(pred.is_none(), "copy: tcgen05.ld copies cannot be predicated (the instruction is warp-collective)");
    assert!(
        is_tmem(src),
        "copy: a TMEM load atom ({:?}) reads tensor memory, but the source is not a TMEM tensor",
        atom.op
    );
    assert!(
        dst.is_register(),
        "copy: a TMEM load atom writes registers: the destination is a register fragment (make_fragment or \
         make_fragment_like), not {}",
        dst.memory_space().map_or("a coordinate tensor".to_string(), |s| format!("{s:?} memory"))
    );
    let taddr = tmem_chunk_address(src, src_atom, s_base, atom.op, "source");
    let n = shape.registers(num) as usize * (32 / T::DTYPE.storage_bits());
    let (elems, red) = builder::with(|b| {
        let results: Vec<ValueId> = (0..n).map(|_| b.new_value(T::DTYPE)).collect();
        let red = match atom.op {
            CopyOp::TmemLoadRed { red, nan, half_split_off, .. } => {
                let half_split =
                    shape.half_split(num, false).map(|d| if half_split_off == 0 { d } else { half_split_off });
                Some(crate::ir::TmemLdRed { result: b.new_value(DType::F32), op: red, nan, half_split })
            }
            _ => None,
        };
        b.push(Op::Tcgen05Ld { results: results.clone(), taddr, shape, num, pack, red, sp: None }, loc);
        (results, red)
    });
    if let (Some(red), Some(stat)) = (red, stat) {
        stat.store_at(IndexExpr::Static(0), Value::from_id(red.result));
    }
    assert_eq!(
        elems.len(),
        dst_atom.size(),
        "copy: the TMEM load's registers and its destination chunk differ in size"
    );
    for (i, e) in elems.into_iter().enumerate() {
        dst.store_at(d_base.clone() + dst_atom.call(i), Value::from_id(e));
    }
}

/// One call of a TMEM load-reduce atom (CuTe DSL's
/// `cute.copy_atom_call(atom, src, [data, stat])` on a
/// [`tcgen05::LdRed32x32bOp`](crate::nvgpu::tcgen05::LdRed32x32bOp) or
/// [`LdRed16x32bx2Op`](crate::nvgpu::tcgen05::LdRed16x32bx2Op) atom): `tcgen05.ld.red` of the atom's TMEM
/// view `src` (the layout [`make_tmem_copy`](crate::nvgpu::tcgen05::make_tmem_copy)
/// partitions give one warp, with its address in the pointer) into `dsts`:
/// the register fragment of the loaded Float32 values, then a register
/// fragment whose first element receives the thread's reduction (the
/// minimum or maximum of the values it loaded, as the atom's reduction and
/// `nan` say). Both are undefined until
/// [`fence_view_async_tmem_load`](crate::fence_view_async_tmem_load).
/// Warp-collective.
#[track_caller]
pub fn copy_atom_call<T: Numeric>(atom: &CopyAtom<T>, src: &Tensor<T>, dsts: &[&Tensor<T>]) {
    let loc = Location::caller();
    let CopyOp::TmemLoadRed { shape, num, .. } = atom.op else {
        panic!(
            "cute: copy_atom_call({:?}): only the TMEM load-reduce atoms (tcgen05.LdRed32x32bOp, LdRed16x32bx2Op) \
             take several destinations; copy other atoms with copy",
            atom.op
        )
    };
    assert_eq!(
        dsts.len(),
        2,
        "cute: copy_atom_call({:?}) takes two destinations, the data and the reduction, not {}",
        atom.op,
        dsts.len()
    );
    assert_eq!(T::DTYPE, DType::F32, "cute: copy_atom_call: tcgen05.ld.red reads Float32 values, not {}", T::DTYPE);
    let (data, stat) = (dsts[0], dsts[1]);
    assert!(
        stat.is_register() && stat.layout().size() >= 1,
        "cute: copy_atom_call: the reduction goes to a register fragment"
    );
    tmem_load_chunk(
        atom,
        src,
        src.layout(),
        src.offset.clone(),
        data,
        data.layout(),
        IndexExpr::Static(0),
        None,
        shape,
        num,
        false,
        Some(stat),
        loc,
    );
}

/// One call of a TMEM sparse-compress atom (CuTe DSL's
/// `cute.copy_atom_call(atom, src, [cdata, mdata{, red}])` on a
/// [`tcgen05::LdSPCompress32x32bOp`](crate::nvgpu::tcgen05::LdSPCompress32x32bOp)
/// or [`LdRedSPCompress32x32bOp`](crate::nvgpu::tcgen05::LdRedSPCompress32x32bOp)
/// atom): `tcgen05.ld{.red}.spcompress.32x32b`
/// of the atom's TMEM view `src` (one `.x<num>` warp-wide chunk, the layout
/// [`make_tmem_copy`](crate::nvgpu::tcgen05::make_tmem_copy) partitions give
/// one warp). Each thread loads `num` Float32 values and keeps two of every
/// four, as the atom's row operation prefers (the larger for `Max`, the
/// smaller for `Min`, by absolute value for the `Abs` forms; a NaN is the
/// least preferred, or with `nan` the most): the `num / 2` kept values, in
/// order, go to `cdata` (a Float32 register fragment), and the 2:4
/// metadata to `mdata` (a Uint32 register fragment of `max(num / 32, 1)`
/// registers: the 4-bit field of each group of four, the position of the
/// first kept value in bits 0-1 and of the second in bits 2-3, the format
/// of the sparse MMAs, at bit `4 (g % 8)` of register `g / 8`). The `.red`
/// form also writes the thread's reduction over all its loaded values (as
/// [`copy_atom_call`] does) to the first element of `red`. All are undefined
/// until [`fence_view_async_tmem_load`](crate::fence_view_async_tmem_load).
/// Warp-collective. PTX ISA 9.4, sm_107a: cute-rs's reading of the
/// instruction (no installed assembler knows it).
#[track_caller]
pub fn copy_atom_call_sp(
    atom: &CopyAtom<f32>,
    src: &Tensor<f32>,
    cdata: &Tensor<f32>,
    mdata: &Tensor<u32>,
    red: Option<&Tensor<f32>>,
) {
    let loc = Location::caller();
    let CopyOp::TmemLoadSpCompress { num, red: red_op, with_red, nan } = atom.op else {
        panic!(
            "cute: copy_atom_call_sp({:?}): only the TMEM sparse-compress atoms (tcgen05.LdSPCompress32x32bOp, \
             LdRedSPCompress32x32bOp) take this call",
            atom.op
        )
    };
    assert_eq!(
        red.is_some(),
        with_red,
        "cute: copy_atom_call_sp({:?}): the reduction destination is given exactly for the .red form",
        atom.op
    );
    assert!(is_tmem(src), "cute: copy_atom_call_sp: the source is not a TMEM tensor");
    for (what, ok) in [
        ("cdata", cdata.is_register() && cdata.layout().size() == num as usize / 2),
        ("mdata", mdata.is_register() && mdata.layout().size() == (num as usize / 32).max(1)),
        ("red", red.is_none_or(|r| r.is_register() && r.layout().size() >= 1)),
    ] {
        assert!(
            ok,
            "cute: copy_atom_call_sp({:?}): {what} must be a register fragment of {} elements (cdata: num / 2; mdata: \
             max(num / 32, 1); red: at least 1)",
            atom.op,
            match what {
                "cdata" => num as usize / 2,
                "mdata" => (num as usize / 32).max(1),
                _ => 1,
            }
        );
    }
    let taddr = tmem_chunk_address(src, src.layout(), src.offset.clone(), atom.op, "source");
    let (data_ids, meta_ids, red_id) = builder::with(|b| {
        let data: Vec<ValueId> = (0..num as usize / 2).map(|_| b.new_value(DType::F32)).collect();
        let meta: Vec<ValueId> = (0..(num as usize / 32).max(1)).map(|_| b.new_value(DType::I32)).collect();
        let red = with_red.then(|| crate::ir::TmemLdRed {
            result: b.new_value(DType::F32),
            op: red_op,
            nan,
            half_split: None,
        });
        b.push(
            Op::Tcgen05Ld {
                results: data.clone(),
                taddr,
                shape: crate::ir::TmemShape::S32x32b,
                num,
                pack: false,
                red,
                sp: Some(crate::ir::TmemLdSpCompress { mdata: meta.clone(), op: red_op, nan }),
            },
            loc,
        );
        (data, meta, red)
    });
    for (i, e) in data_ids.into_iter().enumerate() {
        cdata.store_at(IndexExpr::Static(0) + cdata.layout().call(i), Value::from_id(e));
    }
    for (i, e) in meta_ids.into_iter().enumerate() {
        mdata.store_at(IndexExpr::Static(0) + mdata.layout().call(i), Value::from_id(e));
    }
    if let (Some(red), Some(r)) = (red, red_id) {
        red.store_at(IndexExpr::Static(0), Value::from_id(r.result));
    }
}

/// One `tcgen05.st` of a TMEM store atom: the registers of `src` into the
/// chunk at `d_base` of the TMEM tensor `dst`.
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn tmem_store_chunk<T: Numeric>(
    atom: &CopyAtom<T>,
    src: &Tensor<T>,
    src_atom: &Layout,
    s_base: IndexExpr,
    dst: &Tensor<T>,
    dst_atom: &Layout,
    d_base: IndexExpr,
    pred: Option<Value<bool>>,
    shape: crate::ir::TmemShape,
    num: u32,
    unpack: bool,
    loc: crate::ir::Loc,
) {
    assert!(pred.is_none(), "copy: tcgen05.st copies cannot be predicated (the instruction is warp-collective)");
    assert!(
        is_tmem(dst),
        "copy: a TMEM store atom ({:?}) writes tensor memory, but the destination is not a TMEM tensor",
        atom.op
    );
    assert!(
        src.is_register() || src.memory_space() == Some(MemSpace::Local),
        "copy: a TMEM store atom reads registers: the source is a register fragment or make_rmem_tensor, not {}",
        src.memory_space().map_or("a coordinate tensor".to_string(), |s| format!("{s:?} memory"))
    );
    let taddr = tmem_chunk_address(dst, dst_atom, d_base, atom.op, "destination");
    let per = 32 / T::DTYPE.storage_bits();
    let elems: Vec<ValueId> =
        (0..src_atom.size()).map(|k| src.load_at(s_base.clone() + src_atom.call(k)).checked()).collect();
    assert_eq!(
        elems.len(),
        shape.registers(num) as usize * per,
        "copy: the TMEM store's source chunk and registers differ in size"
    );
    builder::with(|b| b.push(Op::Tcgen05St { values: elems, taddr, shape, num, unpack }, loc));
}

/// An 8-bit `ldmatrix` (`.m16n16` or `.m8n16`) into `dst`: the 32-bit
/// registers it yields, split into bytes; destination register `w` byte
/// `q` is byte `permute(w, q).1` of result `permute(w, q).0`.
#[allow(clippy::too_many_arguments)]
fn ldmatrix_b8<T: Numeric>(
    src: &Tensor<T>,
    src_atom: &Layout,
    s_base: IndexExpr,
    dst: &Tensor<T>,
    dst_atom: &Layout,
    d_base: IndexExpr,
    pred: Option<Value<bool>>,
    op: CopyOp,
    num: usize,
    trans: bool,
    form: LdMatrixForm,
    loc: crate::ir::Loc,
    permute: impl Fn(usize, usize) -> (usize, usize),
) {
    assert!(pred.is_none(), "ldmatrix copies cannot be predicated");
    assert_eq!(src.memory_space(), Some(MemSpace::Shared), "ldmatrix reads shared memory");
    assert!(dst.is_register(), "ldmatrix writes registers");
    check_vector_access(src, src_atom, &s_base, "source", &op);
    // The index counts the source's 8-bit elements, which are bytes.
    let index = crate::tensor::index_operand(physical(src, s_base));
    let buffer = src.buffer().unwrap();
    let regs = (form.results_per_matrix() as usize) * num;
    let bytes = builder::with(|b| {
        let words: Vec<ValueId> = (0..regs).map(|_| b.new_value(DType::I32)).collect();
        let op = Op::LdMatrix { results: words.clone(), buffer, index, num: num as u32, trans, form };
        b.push(op, loc);
        words.into_iter().map(|w| b.unpack(w, T::DTYPE, 4, loc)).collect::<Vec<_>>()
    });
    for w in 0..regs {
        for q in 0..4 {
            let (r, k) = permute(w, q);
            dst.store_at(d_base.clone() + dst_atom.call(4 * w + q), Value::from_id(bytes[r][k]));
        }
    }
}

/// The current value of a register slot, or zero if it was never written.
/// This does not count as reading the register in a runtime loop (which
/// the builder reports when the loop then writes it): a predicated copy
/// into a fragment created outside a loop keeps the value from before the
/// loop in the registers it skips, where CuTe DSL keeps the previous
/// iteration's. That differs only if the predicate changes between
/// iterations and the kernel uses the skipped registers, while reporting
/// it would reject the usual pattern of computing on every register and
/// storing under the same predicate.
fn old_register_value<T: Numeric>(t: &Tensor<T>, slot: &IndexExpr) -> Value<T> {
    let i = slot.as_static().expect("register tensors are indexed statically");
    let existing = t.register_storage().and_then(|s| s.peek(i));
    match existing {
        Some(v) => Value::from_id(v),
        None => {
            let loc = Location::caller();
            Value::from_id(builder::with(|b| b.constant(Literal::zero(T::DTYPE), loc)))
        }
    }
}

/// A boolean register fragment shaped like `coords`, filled by `f(coord)`
/// for each element of `coords` (typically a partitioned identity tensor).
#[track_caller]
pub fn make_predicate<F>(coords: &Tensor<crate::tensor::Ident>, mut f: F) -> Tensor<bool>
where
    F: FnMut(&crate::layout::Coord) -> Boolean,
{
    let pred = crate::tensor::make_fragment::<bool>(coords.shape().clone());
    for i in 0..coords.size() {
        let c = coords.coord(i);
        pred.store(i, f(&c));
    }
    pred
}

/// The shared-memory side of a [`CopyOp::S2R`] or [`CopyOp::R2S`] copy:
/// this CTA's shared memory for [`SharedSpace::Cta`], another CTA's (a
/// [`map_dsmem_ptr`](crate::map_dsmem_ptr) tensor) for
/// [`SharedSpace::Cluster`].
#[track_caller]
fn check_shared_side<T: Numeric>(t: &Tensor<T>, ss: SharedSpace, name: &str, verb: &str) {
    let space = if t.is_register() { AddressSpace::Rmem } else { t.memspace() };
    match ss {
        SharedSpace::Cta => assert!(
            space == AddressSpace::Smem,
            "copy: {name} with shared_space=SharedSpace::Cta {verb} this CTA's shared memory, not {space} memory{}",
            if space == AddressSpace::Dsmem { " (another CTA's: shared_space=SharedSpace::Cluster)" } else { "" }
        ),
        SharedSpace::Cluster => assert!(
            space == AddressSpace::Dsmem,
            "copy: {name} with shared_space=SharedSpace::Cluster {verb} another CTA's shared memory (a tensor made from \
             a map_dsmem_ptr pointer), not {space} memory"
        ),
    }
}

/// A [`CopyOp::DsmemStore`] copy: registers to the
/// distributed-shared-memory tensor `dst`, one `st.async` of `bits` bits
/// (0: the widest of 128, 64 and 32 the tensors allow) per chunk, each
/// counting its bytes down the mbarrier `args.mbar_ptr` (a
/// [`map_dsmem_ptr`](crate::map_dsmem_ptr) pointer of the same CTA).
#[track_caller]
fn dsmem_store_copy<T: Numeric>(bits: usize, src: &Tensor<T>, dst: &Tensor<T>, args: &CopyArgs, func: &str) {
    let loc = Location::caller();
    let name = "CopyDsmemStoreOp";
    assert!(args.pred.is_none(), "{func}: {name} copies cannot be predicated");
    assert!(src.is_register(), "{func}: {name} stores registers (a fragment), not {:?}", src.memory_space());
    let Some((buffer, Some(peer))) = dst.buffer_and_peer() else {
        panic!(
            "{func}: {name} stores into another CTA's shared memory: the destination must be a tensor made from a \
             map_dsmem_ptr pointer, not {} memory",
            if dst.is_register() { AddressSpace::Rmem } else { dst.memspace() }
        )
    };
    let mbar = args.mbar_ptr.as_ref().expect("checked: CopyDsmemStoreOp takes mbar_ptr");
    let (mbar_buffer, mbar_offset, mbar_peer) = mbar.parts_dsmem();
    let same_peer = mbar_peer == Some(peer)
        || matches!((mbar_peer.and_then(builder::try_constant), builder::try_constant(peer)), (Some(a), Some(b)) if a == b);
    assert!(
        mbar_peer.is_some() && same_peer,
        "{func}: {name}'s mbar_ptr is the mbarrier of the CTA the store writes to: map it with map_dsmem_ptr to the \
         same rank as the destination ({})",
        if mbar_peer.is_none() { "it is this CTA's" } else { "it is mapped to another rank value" }
    );
    for (t, role) in [(src, "source"), (dst, "destination")] {
        if let Some(inner) = t.inner_layout() {
            panic!("{func}: {name} copies plain memory, not the {role} over the inner layout {inner}");
        }
    }
    let n = copy_extents(src, func, "source", &CopyOp::DsmemStore { bits });
    assert_eq!(
        n,
        copy_extents(dst, func, "destination", &CopyOp::DsmemStore { bits }),
        "{func}: tensors have different sizes"
    );
    let eb = T::DTYPE.bits();
    let (src, dst, v) = if bits == 0 {
        let (s, d, v) = autovec_plan(src, dst, 128).unwrap_or_else(|| (as_1d(src), as_1d(dst), 1));
        assert!(
            v * eb >= 32,
            "{func}: {name} with num_bits_per_copy=0 found runs of {v} contiguous {} elements ({} bits), but st.async \
             stores at least 32 bits",
            T::DTYPE,
            v * eb
        );
        (s, d, v)
    } else {
        (src.clone(), dst.clone(), bits / eb)
    };
    let (src_atom, src_chunks) = chunks(src.layout(), v, "source");
    let (dst_atom, dst_chunks) = chunks(dst.layout(), v, "destination");
    assert_eq!(
        src_chunks.len(),
        dst_chunks.len(),
        "{func}: source and destination split into different numbers of stores"
    );
    assert!(
        contiguous(&dst_atom),
        "{func}: a {name} store writes {v} contiguous elements, but the destination's are laid out as {dst_atom}"
    );
    let words = v * eb / 32;
    let mbar_index = crate::tensor::index_operand(mbar_offset);
    for (so, d_o) in src_chunks.iter().zip(&dst_chunks) {
        let values: Vec<ValueId> =
            (0..v).map(|k| src.load_at(src.base_offset().clone() + so.clone() + src_atom.call(k)).checked()).collect();
        let off = dst.physical_offset(dst.base_offset().clone() + d_o.clone());
        assert!(
            off.is_multiple_of(v),
            "{func}: cannot prove that the {name} destination offset {off:?} is aligned to the store's {v} elements{}",
            builder::var_value_note(&off)
        );
        // The index counts 32-bit words.
        let index = crate::tensor::index_operand(off * eb / 32);
        builder::with(|b| {
            // The elements packed into 32-bit words, the first in the low bits.
            let per = 32 / eb;
            let packed: Vec<ValueId> = values
                .chunks(per)
                .map(|w| if per == 1 { b.bitcast(w[0], DType::I32, loc) } else { b.pack(w.to_vec(), DType::I32, loc) })
                .collect();
            debug_assert_eq!(packed.len(), words);
            b.push(
                Op::StAsync { buffer, index, elem: DType::I32, values: packed, mbar: mbar_buffer, mbar_index, peer },
                loc,
            );
        });
    }
}

// Shared-memory-to-TMEM copies.

/// A copy with an S2T atom ([`CopyOp::TmemCp`]): `src` is a tensor of
/// shared-memory descriptors ([`get_s2t_smem_desc_tensor`](crate::nvgpu::tcgen05::get_s2t_smem_desc_tensor)),
/// `dst` the partitioned TMEM tensor. Mode 0 of `dst` is split into the
/// atom's TMEM footprints, mode 0 of `src` into as many parts, and the rest
/// modes are unrolled: one `tcgen05.cp` per footprint, from the descriptor at
/// the first element of its part, issued by a lane the copy elects (once per
/// call) of the converged warp that calls it.
#[track_caller]
fn s2t_copy<T: Numeric>(
    shape: crate::ir::TmemCpShape,
    two_cta: bool,
    src: &Tensor<T>,
    dst: &Tensor<T>,
    args: &CopyArgs,
    func: &str,
) {
    let loc = Location::caller();
    let name = crate::nvgpu::tcgen05::s2t::s2t_op_name(shape);
    assert!(
        args.pred.is_none(),
        "{func}: an S2T copy ({name}) cannot be predicated (a converged warp calls it, and one lane issues it)"
    );
    assert!(args.cache_policy.is_none(), "{func}: an S2T copy ({name}) takes no cache policy");
    assert!(
        src.engine_is_desc(),
        "{func}: the source of an S2T copy ({name}) is the tensor of shared-memory descriptors \
         tcgen05::get_s2t_smem_desc_tensor makes of the partitioned shared-memory tensor, not {src:?}"
    );
    assert!(
        is_tmem(dst),
        "{func}: the destination of an S2T copy ({name}) is a TMEM tensor (make_s2t_copy's partition_d), not {dst:?}"
    );
    let bits = T::DTYPE.storage_bits();
    let want = crate::nvgpu::tcgen05::s2t::s2t_val_layout(shape, bits);
    let v = want.size();
    copy_extents(src, func, "source", &CopyOp::TmemCp { shape, two_cta });
    copy_extents(dst, func, "destination", &CopyOp::TmemCp { shape, two_cta });
    let (dst_atom, dst_chunks) = chunks(dst.layout(), v, "destination");
    assert!(
        dst_atom.coalesce() == want,
        "{func}: the TMEM destination's chunk has the layout {dst_atom}, but the {name} atom writes TMEM as {want}: \
         partition the TMEM tensor with make_s2t_copy"
    );
    let per = dst.layout().mode(0).size() / v;
    let s0 = src.layout().mode(0).size();
    assert!(
        s0.is_multiple_of(per),
        "{func}: the source's mode 0 ({}) does not split into the destination's {per} atoms",
        src.layout().mode(0)
    );
    let (_, src_chunks) = chunks(src.layout(), s0 / per, "source");
    assert_eq!(
        src_chunks.len(),
        dst_chunks.len(),
        "{func}: the source {} and destination {} hold different numbers of {name} atoms",
        src.layout(),
        dst.layout()
    );
    let crate::tensor::Engine::SmemDesc { desc } = src.engine else { unreachable!("checked above") };
    let elect = crate::arch::elect_one().checked();
    for (so, d_o) in src_chunks.iter().zip(&dst_chunks) {
        let off = src.offset.clone() + so;
        let base = Value::<u64>::from_id(desc);
        let desc = if off.as_static() == Some(0) {
            base
        } else {
            let off = builder::with(|b| {
                let x = b.materialize_at(&off, 64, loc);
                b.cast(x, DType::U64, loc)
            });
            base + Value::<u64>::from_id(off)
        }
        .checked();
        let ptr = Ptr::<T> { offset: physical(dst, dst.base_offset().clone() + d_o), ..dst.ptr() };
        let taddr = ptr.toint_i32().checked();
        builder::with(|b| b.push(Op::Tcgen05Cp { shape, two_cta, taddr, desc, elect }, loc));
    }
}

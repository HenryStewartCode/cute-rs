// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's `cute.nvgpu.tcgen05` (Blackwell, SM100 and its family): so
//! far (roadmap part m10-umma-layouts) the build-time pieces of the UMMA
//! operands: the shared-memory layout atoms ([`SmemLayoutAtomKind`],
//! [`make_smem_layout_atom`], [`make_sparse_smem_layout_atom`]),
//! [`tile_to_mma_shape`], and the shared-memory descriptors
//! ([`make_umma_smem_desc`], [`SmemDescriptor`], [`UmmaOperandMap`]).
//! m10-tmem-ld-st adds the TMEM load and store copy atoms
//! ([`Ld32x32bOp`] and its peers) and [`make_tmem_copy`]. m10-umma-f16
//! adds the F16/BF16 MMA ([`MmaF16BF16Op`], with [`CtaGroup`],
//! [`OperandSource`] and [`Field`]) and [`commit`]. m10-umma-dense-kinds
//! adds [`MmaTF32Op`], [`MmaI8Op`], [`MmaF8F6F4Op`] and [`MmaFP8Op`].
//! m10-umma-kinds adds A from tensor memory ([`OperandSource::TMEM`]) and
//! the runtime fields `NEGATE_A`, `NEGATE_B` and `DISABLE_OUTPUT_LANE`.
//! m10-s2t-copies adds the shared-memory-to-TMEM copies ([`Cp128x256bOp`]
//! and its peers, [`make_s2t_copy`], [`get_s2t_smem_desc_tensor`],
//! [`append_s2t_broadcast_mode`]). m10-umma-blockscaled adds the
//! block-scaled MMAs ([`MmaMXF8F6F4Op`], [`MmaMXF4Op`], [`MmaMXF4NVF4Op`],
//! [`SM103MmaMXF4Op`], [`SM103MmaMXF4NVF4Op`]) and [`tmem_sf_frg_layout`].
//! m10-sparse-umma adds the 2:4 structured-sparse MMAs
//! ([`MmaF16BF16SparseOp`], [`MmaFP8SparseOp`]). m10-rubin-umma adds the
//! Rubin (sm_107) MMAs ([`SM107MmaFP8Op`], [`SM107BlockScaledMmaMXF8F6F4Op`],
//! [`SM107MmaMXF4NVF4Op`]) and their collector buffers ([`CollectorOp`]).

pub(crate) mod copy;
mod layouts;
// m10-umma-f16: tcgen05 MMAs and commits.
pub(crate) mod mma;
// m10-umma-dense-kinds: the TF32, I8 and F8F6F4 MMAs.
mod mma_dense;
// m10-umma-blockscaled: the block-scaled MMAs.
pub(crate) mod mma_blockscaled;
// m10-sparse-umma: the sparse MMAs.
mod mma_sparse;
// m10-rubin-umma: the Rubin (sm_107) MMAs.
mod mma_sm107;
mod smem_desc;
// m10-s2t-copies: SMEM-to-TMEM copies.
pub(crate) mod s2t;

// m10-tmem-ld-st: TMEM load and store copy atoms.
pub use copy::{
    Ld16x32bx2Op, Ld16x64bOp, Ld16x128bOp, Ld16x256bOp, Ld32x32bOp, Pack, Repetition, St16x32bx2Op, St16x64bOp,
    St16x128bOp, St16x256bOp, St32x32bOp, Unpack, make_tmem_copy,
};
// m10-ldred-packed-math: TMEM loads with a row reduction.
pub use copy::{LdRed16x32bx2Op, LdRed32x32bOp, TmemLoadRedOp};
// m10-rubin-extras: TMEM loads with sparse compression.
pub use copy::{LdRedSPCompress32x32bOp, LdSPCompress32x32bOp};

// m10-umma-f16.
pub use mma::{CtaGroup, Field, MmaF16BF16Op, OperandSource, commit};
// m10-umma-dense-kinds.
pub use mma_dense::{MmaF8F6F4Op, MmaFP8Op, MmaI8Op, MmaTF32Op};
// m10-umma-blockscaled.
pub use mma_blockscaled::{
    MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op, MmaMXF8Op, SM103MmaMXF4NVF4Op, SM103MmaMXF4Op, tmem_sf_frg_layout,
};
// m10-sparse-umma.
pub use crate::atom::mma::SparseMetadataFormat;
pub use mma_sparse::{MmaF16BF16SparseOp, MmaFP8SparseOp};
// m10-rubin-umma.
pub use crate::ir::CollectorOp;
pub use mma_sm107::{SM107BlockScaledMmaMXF8F6F4Op, SM107MmaFP8Op, SM107MmaMXF4NVF4Op};
// m10-umma-pipelines: TMEM copy introspection.
pub use crate::arch::find_tmem_tensor_col_offset;
pub use copy::{TmemPacking, get_tmem_copy_properties, is_tmem_load, is_tmem_store};

pub use super::common::OperandMajorMode;
pub use layouts::{
    SmemLayoutAtomKind, make_smem_layout_atom, make_smem_layout_atom_dtype, make_sparse_smem_layout_atom,
    make_sparse_smem_layout_atom_dtype, tile_to_mma_shape,
};
pub use smem_desc::{
    SmemDescriptor, UmmaDescEncoding, UmmaLayoutType, UmmaOperandMap, int_to_smem_descriptor, make_umma_smem_desc,
    smem_descriptor_to_int, umma_desc_fields,
};

// m10-s2t-copies.
pub use s2t::{
    Cp2x64x128b0123Op, Cp2x64x128b0213Op, Cp4x32x128bOp, Cp4x256bOp, Cp128x128bOp, Cp128x256bOp,
    append_s2t_broadcast_mode, get_s2t_smem_desc_tensor, make_s2t_copy,
};

/// CuTe DSL's `tcgen05.helpers`: the descriptor casts.
pub mod helpers {
    pub use super::mma::commit;
    pub use super::smem_desc::{int_to_smem_descriptor, smem_descriptor_to_int};
    // m10-s2t-copies.
    pub use super::s2t::{append_s2t_broadcast_mode, get_s2t_smem_desc_tensor, make_s2t_copy};
}

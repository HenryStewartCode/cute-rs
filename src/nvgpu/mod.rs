// SPDX-License-Identifier: MIT OR Apache-2.0

//! The operations of CuTe DSL's `cute.nvgpu`, by its submodules: the
//! structs are named like CuTe DSL's classes and convert into the
//! IR-facing [`CopyOp`](crate::CopyOp) the copy functions take. So far
//! [`cpasync`], the common major modes ([`OperandMajorMode`],
//! [`OutputMajorMode`]) and [`warpgroup`]'s shared-memory layout atoms and
//! F16/BF16 warpgroup MMA (m8-wgmma-core); the `warp` operations follow
//! with the roadmap's later parts.

mod common;
pub mod cpasync;
pub mod tensormap;
pub(crate) mod tma_atom;
pub(crate) mod tma_copy;
// m7-tma-im2col.
pub(crate) mod tma_im2col;
// m7-tma-sm100-modes.
pub(crate) mod tma_sm100;
// m10-port-fp16-gemm: the MMA-aware TMA atoms.
mod tma_sm100_mma;
pub mod warpgroup;
// m10-umma-layouts.
pub mod tcgen05;
// m10-sm120-blockscaled: cute.nvgpu.warp's block-scaled MMAs.
pub mod warp;

pub use common::{OperandMajorMode, OutputMajorMode};
pub use tma_sm100_mma::{
    make_im2col_tma_atom_a, make_tiled_tma_atom_a, make_tiled_tma_atom_a_with, make_tiled_tma_atom_b,
    make_tiled_tma_atom_b_with, make_tiled_tma_atom_c,
};

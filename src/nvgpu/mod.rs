// SPDX-License-Identifier: MIT OR Apache-2.0

//! The operations of CuTe DSL's `cute.nvgpu`, by its submodules: the
//! structs are named like CuTe DSL's classes and convert into the
//! IR-facing [`CopyOp`](crate::CopyOp) the copy functions take. It holds
//! [`cpasync`], the common major modes ([`OperandMajorMode`],
//! [`OutputMajorMode`]) and [`warpgroup`]'s shared-memory layout atoms and
//! warpgroup MMAs, with [`tcgen05`] and the `warp` operations.

mod common;
pub mod cpasync;
pub mod tensormap;
pub(crate) mod tma_atom;
pub(crate) mod tma_copy;
// The im2col TMA atoms.
pub(crate) mod tma_im2col;
// The SM100 TMA modes.
pub(crate) mod tma_sm100;
// The MMA-aware TMA atoms.
mod tma_sm100_mma;
pub mod warpgroup;
// The tcgen05 atoms.
pub mod tcgen05;
// Cute.nvgpu.warp's block-scaled MMAs.
pub mod warp;

pub use common::{OperandMajorMode, OutputMajorMode};
pub use tma_sm100_mma::{
    make_im2col_tma_atom_a, make_tiled_tma_atom_a, make_tiled_tma_atom_a_with, make_tiled_tma_atom_b,
    make_tiled_tma_atom_b_with, make_tiled_tma_atom_c,
};

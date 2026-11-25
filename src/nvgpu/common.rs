// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's `cute.nvgpu` common types: the major modes of MMA operands
//! ([`OperandMajorMode`]) and of the output ([`OutputMajorMode`]).

use std::fmt;
use std::str::FromStr;

/// Which mode of an MMA operand is contiguous (CuTe DSL's
/// `cute.nvgpu.OperandMajorMode`, passed as `a_major_mode` /
/// `b_major_mode` to the warpgroup and tcgen05 MMA operations). `K`: the
/// reduction mode is contiguous (A row-major, B column-major); `MN`: the M
/// mode of A or the N mode of B is.
///
/// The mode also picks the shared-memory layout atom
/// ([`SmemLayoutAtomKind`](crate::nvgpu::warpgroup::SmemLayoutAtomKind)'s
/// `MN_*` or `K_*` kinds). WGMMA takes `MN` only for 16-bit operands in
/// shared memory, and a register A is always `K`; those rules belong to the
/// MMA operations that take the mode. `"k"` and `"mn"` parse to it, in
/// either case, as CuTe DSL's enum accepts strings.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperandMajorMode {
    MN,
    K,
}

impl fmt::Display for OperandMajorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperandMajorMode::MN => f.write_str("OperandMajorMode.MN"),
            OperandMajorMode::K => f.write_str("OperandMajorMode.K"),
        }
    }
}

impl FromStr for OperandMajorMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_uppercase().as_str() {
            "MN" => Ok(OperandMajorMode::MN),
            "K" => Ok(OperandMajorMode::K),
            _ => Err(format!("OperandMajorMode: {s:?} is neither \"MN\" nor \"K\"")),
        }
    }
}

/// Which mode of the MMA output `D(M, N)` is contiguous (CuTe DSL's
/// `cute.nvgpu.OutputMajorMode`): `M`, column-major, stride `(1, M)`; `N`,
/// row-major, stride `(N, 1)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OutputMajorMode {
    M,
    N,
}

impl fmt::Display for OutputMajorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputMajorMode::M => f.write_str("OutputMajorMode.M"),
            OutputMajorMode::N => f.write_str("OutputMajorMode.N"),
        }
    }
}

impl FromStr for OutputMajorMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_uppercase().as_str() {
            "M" => Ok(OutputMajorMode::M),
            "N" => Ok(OutputMajorMode::N),
            _ => Err(format!("OutputMajorMode: {s:?} is neither \"M\" nor \"N\"")),
        }
    }
}

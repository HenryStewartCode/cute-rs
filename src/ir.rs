//! The kernel intermediate representation.
//!
//! A kernel is a tree of structured regions (like MLIR's `scf` dialect) over
//! SSA scalar values. Memory is addressed through buffers: global buffers are
//! kernel parameters, shared buffers are per-block allocations and local
//! buffers are per-thread arrays. Register tensors never appear here: the
//! builder keeps them as arrays of SSA values, which is exactly what they
//! become after register allocation.

use std::fmt;
use std::panic::Location;

/// Source location of the user code that created an instruction.
pub type Loc = &'static Location<'static>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BufferId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

mod types;
pub use types::*;
mod scalar;
pub use scalar::*;
mod memory;
pub use memory::*;
mod mma;
pub use mma::*;
mod numeric;
pub use numeric::*;
mod asm;
pub use asm::*;
mod op;
pub use op::*;
mod tcgen05;
pub use tcgen05::*;
mod kernel;
pub use kernel::*;
mod externs;
pub use externs::*;

//! Swizzle functions and swizzled (composed) layouts.
//!
//! Shared memory is split into 32 four-byte banks. Tiles read column-wise
//! (as `ldmatrix` does for row-major operands) hit the same bank from many
//! threads. A swizzle XORs some higher offset bits into lower ones so that
//! such accesses spread across banks, while keeping short runs of
//! consecutive elements together (so vector accesses still work).

use std::fmt;

use super::index::IndexExpr;
use super::layout::{Layout, tile_to_shape};
use super::tree::IntTuple;

/// CuTe's `Swizzle<B, M, S>`: XOR the `bits` bits starting at bit
/// `base + shift` into the `bits` bits starting at bit `base`. The low `base`
/// bits are never changed, so runs of `2^base` elements stay contiguous.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Swizzle {
    pub bits: u32,
    pub base: u32,
    pub shift: u32,
}

impl Swizzle {
    /// `Swizzle<bits, base, shift>`.
    pub fn new(bits: u32, base: u32, shift: u32) -> Swizzle {
        assert!(shift >= bits, "Swizzle<{bits},{base},{shift}>: the shift must be at least the number of bits");
        Swizzle { bits, base, shift }
    }

    fn source_mask(&self) -> usize {
        ((1usize << self.bits) - 1) << (self.base + self.shift)
    }

    pub fn apply(&self, offset: usize) -> usize {
        offset ^ ((offset & self.source_mask()) >> self.shift)
    }

    pub fn apply_expr(&self, offset: IndexExpr) -> IndexExpr {
        if self.bits == 0 {
            return offset;
        }
        if let Some(v) = offset.as_static() {
            return IndexExpr::Static(self.apply(v));
        }
        let moved = IndexExpr::shift_right(
            IndexExpr::and(offset.clone(), IndexExpr::Static(self.source_mask())),
            IndexExpr::Static(self.shift as usize),
        );
        IndexExpr::xor(offset, moved)
    }

    /// Number of consecutive elements guaranteed to stay contiguous.
    pub fn contiguous_run(&self) -> usize {
        1 << self.base
    }
}

impl fmt::Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sw<{},{},{}>", self.bits, self.base, self.shift)
    }
}

/// `Swizzle<bits, base, shift>` (CuTe DSL's `make_swizzle`).
pub fn make_swizzle(bits: u32, base: u32, shift: u32) -> Swizzle {
    Swizzle::new(bits, base, shift)
}

/// A layout followed by a swizzle: `offset = swizzle(layout(coord))`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ComposedLayout {
    pub swizzle: Swizzle,
    pub layout: Layout,
}

impl ComposedLayout {
    pub fn call(&self, idx: usize) -> usize {
        self.swizzle.apply(self.layout.call(idx))
    }

    pub fn shape(&self) -> &IntTuple {
        self.layout.shape()
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Codomain size, rounded up so swizzled offsets stay in range.
    pub fn cosize(&self) -> usize {
        let span = 1usize << (self.swizzle.base + self.swizzle.shift + self.swizzle.bits);
        self.layout.cosize().next_multiple_of(span)
    }
}

impl fmt::Display for ComposedLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} o {}", self.swizzle, self.layout)
    }
}

/// `swizzle ∘ layout`, for swizzled shared-memory tensors.
pub fn make_composed_layout(swizzle: Swizzle, layout: Layout) -> ComposedLayout {
    ComposedLayout { swizzle, layout }
}

/// [`tile_to_shape`] for a swizzled layout atom.
pub fn tile_to_shape_swizzled(atom: &ComposedLayout, shape: impl Into<IntTuple>) -> ComposedLayout {
    ComposedLayout { swizzle: atom.swizzle, layout: tile_to_shape(&atom.layout, shape) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swizzle_is_a_permutation_within_its_span() {
        let sw = Swizzle::new(3, 3, 3);
        let mut seen = vec![false; 512];
        for i in 0..512 {
            let j = sw.apply(i);
            assert!(j < 512);
            assert!(!seen[j]);
            seen[j] = true;
            assert_eq!(j & 7, i & 7, "low bits are preserved");
        }
        // Bit 6 (the lowest source bit) is XORed into bit 3.
        assert_eq!(sw.apply(64), 64 + 8);
    }
}

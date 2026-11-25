// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/swizzle.hpp, include/cute/swizzle_layout.hpp and include/cute/layout_composed.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and cutlass_compiler/cutegen/include/cutegen/composed_layout.hpp,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Swizzle functions and swizzled (composed) layouts.
//!
//! Shared memory is split into 32 four-byte banks. Tiles read column-wise
//! (as `ldmatrix` does for row-major operands) hit the same bank from many
//! threads. A swizzle XORs some higher offset bits into lower ones so that
//! such accesses spread across banks, while keeping short runs of
//! consecutive elements together (so vector accesses still work).

use std::fmt;

use super::basis::tuple_sum;
use super::index::IndexExpr;
use super::layout::{
    Coord, HasCosize, Layout, LayoutSource, Shape, SliceCoord, Tileable, tile_to_shape, tile_to_shape_ordered,
};
use super::tree::{HasShape, Tree};

/// CuTe's `Swizzle<B, M, S>`: XOR the `bits` bits starting at bit
/// `base + max(shift, 0)` (the "Y" bits) into the `bits` bits starting at
/// bit `base + max(-shift, 0)` (the "Z" bits). The low `base` bits are never
/// changed, so runs of `2^base` elements stay contiguous. A negative shift
/// XORs low bits into high ones; `|shift| >= bits`, as in CuTe, so the two
/// sets of bits do not overlap.
///
/// A swizzle acts on whatever offsets it is given: element offsets in a
/// [`ComposedLayout`] (and in the tensors built from one, as
/// `make_composed_layout` gives them), byte addresses in a pointer's swizzle
/// ([`recast_ptr`](crate::recast_ptr)), as in CuTe DSL. For elements of
/// `2^k` bytes, `Swizzle<B,M,S>` on elements is `Swizzle<B,M+k,S>` on bytes
/// ([`Swizzle::to_bytes`], [`Swizzle::to_elements`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Swizzle {
    pub bits: u32,
    pub base: u32,
    pub shift: i32,
}

impl Swizzle {
    /// `Swizzle<bits, base, shift>`. `|shift|` must be at least `bits`.
    #[track_caller]
    pub fn new(bits: u32, base: u32, shift: i32) -> Swizzle {
        assert!(
            shift.unsigned_abs() >= bits,
            "Swizzle<{bits},{base},{shift}>: |shift| must be at least the number of bits, so the bits XORed in do \
             not overlap the bits they change"
        );
        assert!(
            bits + base + shift.unsigned_abs() < usize::BITS,
            "Swizzle<{bits},{base},{shift}>: the swizzle reaches past bit {}",
            usize::BITS - 1
        );
        Swizzle { bits, base, shift }
    }

    /// The identity, as CuTe DSL spells it: `Swizzle<0,4,3>`.
    pub fn identity() -> Swizzle {
        Swizzle { bits: 0, base: 4, shift: 3 }
    }

    /// Whether the swizzle changes nothing (`bits == 0`).
    pub fn is_identity(&self) -> bool {
        self.bits == 0
    }

    /// The bits read (CuTe's `yyy_msk`).
    pub fn y_mask(&self) -> usize {
        ((1usize << self.bits) - 1) << (self.base + self.shift.max(0) as u32)
    }

    /// The bits changed (CuTe's `zzz_msk`).
    pub fn z_mask(&self) -> usize {
        ((1usize << self.bits) - 1) << (self.base + (-self.shift).max(0) as u32)
    }

    fn source_mask(&self) -> usize {
        self.y_mask()
    }

    pub fn apply(&self, offset: usize) -> usize {
        let moved = offset & self.source_mask();
        if self.shift >= 0 { offset ^ (moved >> self.shift) } else { offset ^ (moved << -self.shift) }
    }

    pub fn apply_expr(&self, offset: IndexExpr) -> IndexExpr {
        if self.bits == 0 {
            return offset;
        }
        if let Some(v) = offset.as_static() {
            return IndexExpr::Static(self.apply(v));
        }
        let picked = IndexExpr::and(offset.clone(), IndexExpr::Static(self.source_mask()));
        let moved = if self.shift >= 0 {
            IndexExpr::shift_right(picked, IndexExpr::Static(self.shift as usize))
        } else {
            picked * IndexExpr::Static(1usize << -self.shift)
        };
        IndexExpr::xor(offset, moved)
    }

    /// CuTe's `max_alignment` of a swizzle: `2^base`, the runs it never
    /// breaks up.
    pub fn max_alignment(&self) -> usize {
        1 << self.base
    }

    /// Number of consecutive elements guaranteed to stay contiguous.
    pub fn contiguous_run(&self) -> usize {
        1 << self.base
    }

    /// The span the swizzle permutes within, `2^(bits + base + |shift|)`:
    /// a swizzle maps every aligned block of this many offsets onto itself,
    /// so a swizzle on absolute addresses needs a base aligned to it.
    pub fn period(&self) -> usize {
        1usize << (self.bits + self.base + self.shift.unsigned_abs())
    }

    /// This swizzle on byte offsets, read as one on the offsets of elements
    /// of `elem_bits` bits (a power of two): `Swizzle<B,M,S>` on bytes is
    /// `Swizzle<B,M-k,S>` on elements of `2^k` bytes, and `Swizzle<B,M+j,S>`
    /// on elements of `8 / 2^j` bits. It fails for an element width that is
    /// not a power of two, or a byte swizzle finer than the element (`M <
    /// k`), which would split elements.
    pub fn to_elements(&self, elem_bits: usize) -> Result<Swizzle, String> {
        if self.bits == 0 {
            return Ok(*self);
        }
        if !elem_bits.is_power_of_two() {
            return Err(format!(
                "a pointer swizzle acts on bytes, and {elem_bits}-bit elements are not a power-of-two fraction or \
                 multiple of a byte"
            ));
        }
        if elem_bits >= 8 {
            let k = (elem_bits / 8).trailing_zeros();
            if self.base < k {
                return Err(format!(
                    "the byte swizzle {self} leaves only the low {} bytes of each run alone, finer than the {} bytes \
                     of an element, so it would split elements",
                    1 << self.base,
                    elem_bits / 8
                ));
            }
            Ok(Swizzle { base: self.base - k, ..*self })
        } else {
            Ok(Swizzle { base: self.base + (8 / elem_bits).trailing_zeros(), ..*self })
        }
    }

    /// This swizzle on the offsets of elements of `elem_bits` bits (a power
    /// of two, at least 8 or a fraction of a byte) as one on byte offsets,
    /// the inverse of [`to_elements`](Swizzle::to_elements).
    #[track_caller]
    pub fn to_bytes(&self, elem_bits: usize) -> Swizzle {
        if self.bits == 0 {
            return *self;
        }
        assert!(elem_bits.is_power_of_two(), "Swizzle::to_bytes: {elem_bits}-bit elements are not a power of two");
        if elem_bits >= 8 {
            Swizzle { base: self.base + (elem_bits / 8).trailing_zeros(), ..*self }
        } else {
            let j = (8 / elem_bits).trailing_zeros();
            assert!(self.base >= j, "Swizzle::to_bytes: {self} on {elem_bits}-bit elements splits bytes");
            Swizzle { base: self.base - j, ..*self }
        }
    }
}

/// `S<b,m,s>`, as CuTe prints a swizzle.
impl fmt::Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S<{},{},{}>", self.bits, self.base, self.shift)
    }
}

/// `Swizzle<bits, base, shift>` (CuTe DSL's `make_swizzle`). A swizzle of no
/// bits is the identity, which CuTe DSL spells `S<0,4,3>`, so `make_swizzle(0,
/// m, s)` gives that whatever `m` and `s` are.
#[track_caller]
pub fn make_swizzle(bits: u32, base: u32, shift: i32) -> Swizzle {
    if bits == 0 {
        return Swizzle::identity();
    }
    Swizzle::new(bits, base, shift)
}

/// The inner function of a [`ComposedLayout`]: a swizzle or a layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Inner {
    Swizzle(Swizzle),
    Layout(Layout),
}

impl From<Swizzle> for Inner {
    fn from(sw: Swizzle) -> Inner {
        Inner::Swizzle(sw)
    }
}

impl From<Layout> for Inner {
    fn from(l: Layout) -> Inner {
        Inner::Layout(l)
    }
}

impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inner::Swizzle(sw) => sw.fmt(f),
            Inner::Layout(l) => l.fmt(f),
        }
    }
}

/// CuTe's composed layout `inner o offset o outer`: the function `c ->
/// inner(offset + outer(c))`. `outer` gives the shape (it is the layout the
/// algebra acts on: composition, the divides and products, `tile_to_shape`,
/// slicing, `coalesce`, `size`); `inner` is a [`Swizzle`] (swizzled shared
/// memory) or a [`Layout`]; `offset` is added in between, and is where a
/// slice's offset goes, since a swizzle does not distribute over `+`.
///
/// A composed layout whose inner is the identity swizzle and whose offset
/// is 0 is *normal* ([`is_normal`](ComposedLayout::is_normal)): it is its
/// outer layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ComposedLayout {
    pub inner: Inner,
    pub offset: Coord,
    pub outer: Layout,
}

fn is_zero(c: &Coord) -> bool {
    c.leaves().iter().all(|l| l.as_static() == Some(0))
}

impl ComposedLayout {
    /// `swizzle o 0 o outer`.
    #[track_caller]
    pub fn swizzled(swizzle: Swizzle, outer: Layout) -> ComposedLayout {
        make_composed_layout(swizzle, 0, outer)
    }

    /// The swizzle, if the inner function is one.
    pub fn swizzle(&self) -> Option<Swizzle> {
        match &self.inner {
            Inner::Swizzle(sw) => Some(*sw),
            Inner::Layout(_) => None,
        }
    }

    /// The inner function (CuTe DSL's `ComposedLayout.inner`).
    pub fn inner(&self) -> &Inner {
        &self.inner
    }

    /// The offset between the two (CuTe DSL's `ComposedLayout.offset`).
    pub fn offset(&self) -> &Coord {
        &self.offset
    }

    /// The outer layout (CuTe DSL's `ComposedLayout.outer`).
    pub fn outer(&self) -> &Layout {
        &self.outer
    }

    /// Whether the composed layout is its outer layout: the identity swizzle
    /// with offset 0 (CuTe DSL's `ComposedLayout.is_normal`).
    pub fn is_normal(&self) -> bool {
        matches!(self.inner, Inner::Swizzle(sw) if sw.is_identity()) && is_zero(&self.offset)
    }

    /// The outer layout of a normal composed layout (one that decays to it),
    /// or `None`.
    pub fn as_layout(&self) -> Option<&Layout> {
        self.is_normal().then_some(&self.outer)
    }

    fn with_outer(&self, outer: Layout) -> ComposedLayout {
        ComposedLayout { inner: self.inner.clone(), offset: self.offset.clone(), outer }
    }

    /// `inner(x)` for a value `x` of the outer layout plus the offset.
    #[track_caller]
    fn apply_inner(&self, x: &Coord, what: &str) -> Coord {
        match &self.inner {
            Inner::Swizzle(sw) => {
                let Some(v) = x.as_leaf() else {
                    panic!("{what}: the swizzle {sw} of {self} takes an offset, not the tuple {x}")
                };
                Tree::Leaf(sw.apply_expr(v.clone()))
            }
            Inner::Layout(l) => l.eval_checked(x, what),
        }
    }

    /// Evaluate at `coord`: `inner(offset + outer(coord))`, a tuple when the
    /// inner layout has basis strides.
    #[track_caller]
    pub fn eval(&self, coord: &Coord) -> Coord {
        let x = tuple_sum(&self.offset, &self.outer.eval_checked(coord, "ComposedLayout::eval"));
        self.apply_inner(&x, "ComposedLayout::eval")
    }

    /// Evaluate at `coord` as an offset (CuTe DSL's `crd2idx` and calling
    /// the layout).
    #[track_caller]
    pub fn offset_of(&self, coord: &Coord) -> IndexExpr {
        match self.eval(coord) {
            Tree::Leaf(v) => v,
            t => panic!("ComposedLayout::offset_of: {self} maps {coord} to the tuple {t}, not an offset"),
        }
    }

    /// Evaluate at the 1-D index `idx`; panics if the offset is not static.
    #[track_caller]
    pub fn call(&self, idx: usize) -> usize {
        let off = self.offset_of(&Tree::Leaf(IndexExpr::Static(idx)));
        off.as_static().unwrap_or_else(|| panic!("ComposedLayout::call: {self} maps {idx} to the runtime {off:?}"))
    }

    pub fn shape(&self) -> &Shape {
        self.outer.shape()
    }

    /// Size of the domain; panics if it depends on a runtime extent.
    #[track_caller]
    pub fn size(&self) -> usize {
        self.outer.size()
    }

    /// The span a shared-memory allocation of a swizzled layout covers: the
    /// outer layout's cosize plus the offset, rounded up to whole swizzle
    /// periods so swizzled offsets stay in range. For a layout inner, the
    /// inner layout's cosize.
    #[track_caller]
    pub(crate) fn allocation_size(&self) -> usize {
        match &self.inner {
            Inner::Swizzle(sw) => {
                let off = self.offset.as_leaf().and_then(IndexExpr::as_static).unwrap_or_else(|| {
                    panic!("make_smem_tensor: the offset {} of {self} must be a static integer", self.offset)
                });
                let span = self.outer.cosize() + off;
                if sw.is_identity() { span } else { span.next_multiple_of(sw.period()) }
            }
            Inner::Layout(l) => l.cosize(),
        }
    }

    /// Codomain size (CuTe's `cosize`): for a swizzled layout, its outer
    /// layout's, as in C++ CuTe (`swizzle_layout.hpp`) and the CuTe
    /// dialect. Panics for a layout inner, whose cosize CuTe leaves
    /// undefined, or if it depends on a runtime value ([`cosize`](super::cosize)
    /// gives an `IndexExpr`).
    #[track_caller]
    pub fn cosize(&self) -> usize {
        self.require_swizzle_inner("ComposedLayout::cosize");
        self.outer.cosize()
    }

    #[track_caller]
    fn require_swizzle_inner(&self, what: &str) {
        if let Inner::Layout(_) = self.inner {
            panic!(
                "{what}: {self} has a layout as its inner function, and the cosize of such a composed layout is not \
                 defined (as in the CuTe dialect); take the cosize of its inner or outer layout"
            );
        }
    }

    /// CuTe's `max_alignment` of a composed layout: the gcd of the inner's
    /// and the outer layout's.
    pub fn max_alignment(&self) -> usize {
        let a = match &self.inner {
            Inner::Swizzle(sw) if sw.is_identity() => return self.outer.max_alignment(),
            Inner::Swizzle(sw) => sw.max_alignment(),
            Inner::Layout(l) => l.max_alignment(),
        };
        let b = self.outer.max_alignment();
        let mut g = (a, b);
        while g.1 != 0 {
            g = (g.1, g.0 % g.1);
        }
        g.0
    }

    /// Number of top-level modes of the outer layout.
    pub fn rank(&self) -> usize {
        self.outer.rank()
    }

    /// Nesting depth of the outer layout's shape.
    pub fn depth(&self) -> usize {
        self.outer.depth()
    }

    /// Mode `i` (CuTe DSL's `layout[i]`): the same inner and offset over
    /// mode `i` of the outer layout.
    pub fn mode(&self, i: usize) -> ComposedLayout {
        self.with_outer(self.outer.mode(i))
    }

    /// The mode at a path (CuTe DSL's `cute.get(layout, mode=...)`).
    pub fn get(&self, path: &[usize]) -> ComposedLayout {
        self.with_outer(self.outer.get(path))
    }

    /// Select (and reorder) top-level modes (CuTe DSL's `cute.select`).
    pub fn select(&self, modes: &[usize]) -> ComposedLayout {
        self.with_outer(self.outer.select(modes))
    }

    /// Group modes `[begin, end)` into one (CuTe DSL's `cute.group_modes`).
    #[track_caller]
    pub fn group_modes(&self, begin: usize, end: usize) -> ComposedLayout {
        self.with_outer(self.outer.group(begin, end))
    }

    /// Merge contiguous modes of the outer layout (CuTe DSL's
    /// `cute.coalesce`).
    pub fn coalesce(&self) -> ComposedLayout {
        self.with_outer(self.outer.coalesce())
    }

    /// Slice with `coord` (CuTe DSL's `cute.slice_`): the kept modes of the
    /// outer layout, with the offset of the fixed ones moved into the
    /// composed offset, since the swizzle does not distribute over `+`.
    #[track_caller]
    pub fn slice(&self, coord: impl Into<SliceCoord>) -> ComposedLayout {
        let (outer, off) = self.outer.slice_and_offset_in(&coord.into(), "ComposedLayout::slice");
        ComposedLayout { inner: self.inner.clone(), offset: tuple_sum(&self.offset, &off), outer }
    }

    /// Slice with `coord` and return the result with an offset, as the CuTe
    /// dialect's `slice_and_offset` does: `inner(offset + outer(c))` for a
    /// fixed `c` is the returned offset plus the sliced layout at the kept
    /// coordinates. With every mode fixed, the sliced layout is empty and the
    /// offset is the value. A swizzled layout whose slice the swizzle acts on
    /// affinely (static, and checked at every coordinate) decays to a normal
    /// composed layout, the affine function, and the offset of its first
    /// element, as cutegen's slices of `S<3,4,4> o 0 o (64,128):(1,64)` at
    /// `(_,16)` do; otherwise the slice's offset moves into the composed
    /// offset and the returned one is 0.
    #[track_caller]
    pub fn slice_and_offset(&self, coord: impl Into<SliceCoord>) -> (ComposedLayout, Coord) {
        let coord = coord.into();
        let sliced = self.slice(coord.clone());
        if sliced.outer.rank() == 0 || !coord.has_underscore() {
            let value = sliced.apply_inner(&sliced.offset, "ComposedLayout::slice_and_offset");
            return (sliced, value);
        }
        if let Some((affine, start)) = sliced.decay() {
            return (ComposedLayout::swizzled(Swizzle::identity(), affine), Tree::Leaf(IndexExpr::Static(start)));
        }
        (sliced, Tree::Leaf(IndexExpr::Static(0)))
    }

    /// The affine layout (and its start) that a static swizzled composed
    /// layout equals, if one does: `f(c) = f(0) + sum_i c_i * (f(e_i) -
    /// f(0))`, checked at every coordinate.
    fn decay(&self) -> Option<(Layout, usize)> {
        let Inner::Swizzle(sw) = self.inner else { return None };
        let off = self.offset.as_leaf()?.as_static()?;
        if !self.outer.is_static() || self.outer.has_scaled_basis() || self.outer.size() > 1 << 16 {
            return None;
        }
        let f = |i: usize| sw.apply(off + self.outer.call(i)) as i64;
        let f0 = f(0);
        let shape = self.outer.shape().flat_leaves();
        let mut strides = Vec::with_capacity(shape.len());
        let mut unit = 1;
        for s in &shape {
            let s = s.as_static()?;
            strides.push(if s > 1 { f(unit) - f0 } else { 0 });
            unit *= s;
        }
        let stride = Tree::unflatten(
            &strides.iter().map(|&d| super::basis::StrideLeaf::Int(IndexExpr::from_signed(d))).collect::<Vec<_>>(),
            self.outer.shape(),
        );
        let affine = Layout::new(self.outer.shape().clone(), stride);
        let ok = (0..self.outer.size()).all(|i| affine.offset_at(i).as_static_signed() == Some(f(i) - f0));
        ok.then_some((affine, f0 as usize))
    }
}

/// A composed layout's layout-like functions read its outer layout, as CuTe's
/// `make_layout_like` and `make_fragment_like` do.
impl LayoutSource for ComposedLayout {
    fn source_layout(&self) -> &Layout {
        &self.outer
    }
}

/// The layout algebra acts on the outer layout: [`composition`](super::composition),
/// the divides and products, [`tile_to_shape`] and [`coalesce`](super::coalesce)
/// keep the inner and the offset.
impl Tileable for ComposedLayout {
    fn layout_ref(&self) -> &Layout {
        &self.outer
    }

    fn with_layout(&self, layout: Layout) -> Self {
        self.with_outer(layout)
    }
}

/// The outer layout's cosize at the mode path, as in CuTe (the inner and the
/// offset do not count). Panics for a layout inner.
impl HasCosize for ComposedLayout {
    #[track_caller]
    fn cosize_at(&self, mode: &[usize]) -> IndexExpr {
        self.require_swizzle_inner("cosize");
        self.outer.cosize_at(mode)
    }
}

/// `inner o offset o outer`, as CuTe prints it: `S<3,4,3> o 0 o (8,64):(64,1)`.
impl fmt::Display for ComposedLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} o {} o {}", self.inner, self.offset, self.outer)
    }
}

impl HasShape for ComposedLayout {
    type Leaf = IndexExpr;

    fn shape(&self) -> &Shape {
        self.outer.shape()
    }
}

/// CuTe's composed layout `inner o offset o outer` (CuTe DSL's
/// `make_composed_layout(inner, offset, outer)`): `c -> inner(offset +
/// outer(c))`. `inner` is a [`Swizzle`] or a [`Layout`]. A swizzle needs an
/// outer layout of integer strides (it permutes offsets, which a layout with
/// basis strides does not give) and an integer offset.
#[track_caller]
pub fn make_composed_layout(inner: impl Into<Inner>, offset: impl Into<Coord>, outer: Layout) -> ComposedLayout {
    let inner = inner.into();
    let offset = offset.into();
    if let Inner::Swizzle(sw) = &inner {
        if let Some((path, leaf)) = outer.stride().first_basis() {
            panic!(
                "make_composed_layout: invalid composition {sw} o {offset} o {outer}: a swizzle permutes offsets, \
                 and the outer layout has the basis stride {leaf} at mode {path:?}, so it gives tuples"
            );
        }
        assert!(
            offset.as_leaf().is_some(),
            "make_composed_layout: the offset {offset} of a swizzle o offset o {outer} must be an integer"
        );
    }
    ComposedLayout { inner, offset, outer }
}

/// The swizzle of a layout (CuTe DSL's `get_swizzle_portion`): a composed
/// layout's inner swizzle, or the identity `S<0,4,3>` for a plain layout or
/// a layout inner.
pub fn get_swizzle_portion(layout: &impl SwizzlePortion) -> Swizzle {
    layout.swizzle_portion()
}

/// The layout without its swizzle (CuTe DSL's `get_nonswizzle_portion`): a
/// layout itself, or a composed layout's outer layout.
pub fn get_nonswizzle_portion(layout: &impl SwizzlePortion) -> Layout {
    layout.nonswizzle_portion()
}

/// Layouts that [`get_swizzle_portion`] and [`get_nonswizzle_portion`] split.
pub trait SwizzlePortion {
    fn swizzle_portion(&self) -> Swizzle;
    fn nonswizzle_portion(&self) -> Layout;
}

impl SwizzlePortion for Layout {
    fn swizzle_portion(&self) -> Swizzle {
        Swizzle::identity()
    }

    fn nonswizzle_portion(&self) -> Layout {
        self.clone()
    }
}

impl SwizzlePortion for ComposedLayout {
    fn swizzle_portion(&self) -> Swizzle {
        self.swizzle().unwrap_or_else(Swizzle::identity)
    }

    fn nonswizzle_portion(&self) -> Layout {
        self.outer.clone()
    }
}

/// [`tile_to_shape`] for a swizzled layout atom: the outer layout is tiled
/// column-major, the inner and the offset kept.
#[track_caller]
pub fn tile_to_shape_swizzled(atom: &ComposedLayout, shape: impl Into<Shape>) -> ComposedLayout {
    tile_to_shape(atom, shape)
}

/// [`tile_to_shape_ordered`] for a swizzled layout atom (CuTe DSL's
/// `tile_to_shape` of a composed atom): it tiles the outer layout in the
/// mode order `order` and keeps the inner and the offset.
#[track_caller]
pub fn tile_to_shape_swizzled_ordered(
    atom: &ComposedLayout,
    shape: impl Into<Shape>,
    order: &[usize],
) -> ComposedLayout {
    tile_to_shape_ordered(atom, shape, order)
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

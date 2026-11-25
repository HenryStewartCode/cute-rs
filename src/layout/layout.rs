// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/layout.hpp, include/cute/stride.hpp, include/cute/int_tuple.hpp,
//   include/cute/underscore.hpp, include/cute/algorithm/tuple_algorithms.hpp
//   and include/cute/atom/copy_atom.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   cutlass_compiler/cutegen/include/cutegen/layout.hpp,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and media/docs/cpp/cute/02_layout_algebra.md (the logical_product formula),
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and from include/cute/layout.hpp of an earlier release, for formulas that later
// releases changed: CUTLASS 3.8.0 (left_inverse and upcast),
//     Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Layouts and the CuTe layout algebra.
//!
//! A [`Layout`] is a pair of congruent hierarchical tuples, a shape and a
//! stride. It is a function from coordinates to offsets: a coordinate
//! `(c0, c1, ...)` maps to `sum(ci * di)`. A 1-D index is first split into a
//! coordinate in colexicographic (first-mode-fastest) order, so every layout
//! is also a function from `[0, size)` to offsets.
//!
//! The algebra (composition, complement, divides, products, inverses) follows
//! CuTe's definitions; see the CuTe documentation for the theory. Every
//! operation is evaluated when the kernel is built.
//!
//! # Static and dynamic leaves
//!
//! As in CuTe DSL, each leaf of a shape or a stride is either a static
//! integer or a value only known at run time, printed `?`: an
//! [`IndexExpr`]. So [`Shape`] and [`Stride`] are the same tree type as
//! [`Coord`], and the algebra works on layouts such as `(?,?):(?,1)`. Static
//! leaves fold as the algebra runs; only the dynamic remainder becomes
//! instructions, when an offset is finally computed. [`IntTuple`] stays for
//! tuples that are static by nature (tile shapes, the shapes of register
//! fragments and thread-value layouts); `Shape::from` converts an `IntTuple`,
//! and [`Shape::as_static`] converts back when every leaf is static.
//!
//! [`Layout::new`] and [`make_layout`] take any mix of static and runtime
//! leaves (`usize`, [`Int32`](crate::Int32), `IndexExpr`), so
//! `make_layout((m, 8), (8, 1))` with `m` an `Int32` works as in CuTe.
//!
//! Sizes come in two forms. The `usize` accessors ([`Layout::size`],
//! [`Layout::cosize`], [`Shape::size`], and so the size of a mode) are what
//! Rust loops unroll over; they panic, naming the dynamic leaf, when the
//! answer is only known at run time. [`size`] and [`cosize`] (CuTe DSL's
//! `cute.size` and `cute.cosize`, with a mode path) return an `IndexExpr`
//! that is static whenever it can be, and [`shape`] (`cute.shape`) the
//! sub-shape at a mode path.
//!
//! The algebra follows CuTe's rules for dynamic leaves:
//!
//! * Composition and the divides assume that the dynamic extents of the
//!   layout being composed or divided (the left-hand side) are large and
//!   divisible: where one meets a static extent or stride, it is taken to be
//!   a multiple of it, at least as large, and it is divided by it rounding
//!   up (C++ CuTe's `shape_div`, which skips the divisibility check for
//!   dynamic leaves). So `zipped_divide` of an `(M, K)` layout with dynamic
//!   extents by `(128, 32)` has static `(128,32)` tiles and rest modes of
//!   extent `ceil_div(M, 128)` and `ceil_div(K, 32)`, and dividing all of an
//!   `(M, N)` layout by `32` splits the rest over the modes as C++ CuTe does
//!   (it takes the complement of the tiler in the layout's shape), giving
//!   rest modes of extent `ceil_div(M, 32)` and `N`. The right-hand side's
//!   runtime extents and strides (a tiler's, say) are not assumed to be
//!   anything: where one meets a static extent of the left-hand side and is
//!   not provably a multiple of it (from its static factors, or a
//!   [`param_extent`](crate::param_extent)'s `multiple_of`), and wherever two
//!   runtime values are not provably multiples of one another, composition
//!   computes the extents at run time with a `ceil_div` and a `min`, as C++
//!   CuTe does. So `(4,8):(8,1) ∘ m:1` is `(min(4,m),ceil_div(m,4)):(8,1)`,
//!   which is `2:8` for `m = 2` and `(4,2):(8,1)` for `m = 8`, the static
//!   results. As in CuTe, the result is only right for values that satisfy
//!   the divisibility conditions of the static algebra, on the modes that
//!   composition sees (runtime strides keep modes from merging).
//! * A runtime stride on the right-hand side of a composition, or in the
//!   tiler of a divide or a product, must not be 0 at launch: the extents
//!   computed at run time divide by it, and nothing checks it, as in C++
//!   CuTe, which has the same precondition. A static stride of 0 is fine
//!   (`a ∘ s:0` is `s:0`). Where a kernel uses such extents with a stride of
//!   0, the simulator reports the division by zero; on a GPU the result is
//!   undefined.
//! * Static composition follows C++ CuTe's `composition_impl`: a mode with
//!   at least as many elements as are left to place holds them, whether or
//!   not they divide it, so `(6,8):(8,1) ∘ 4:1` is `4:8`. It refuses the
//!   cases where C++ CuTe's weak divisibility condition lets a stride split
//!   a mode it does not divide (`(6,8):(8,1) ∘ (4,12):(1,4)`, the divide of
//!   `(6,8):(8,1)` by 4, say), because the layout C++ CuTe builds then is
//!   often not the composition.
//! * [`Layout::coalesce`] merges two modes only when it can prove the merge
//!   condition (`s0 * d0 == d1`), so `(4,5,?,3,2):(1,4,?,1,3)` coalesces to
//!   `(20,?,6):(1,?,1)`. It does prove some conditions that involve runtime
//!   values (`(m,4):(1,m)` coalesces to `(m*4):1`), which CuTe leaves
//!   unmerged.
//! * [`Layout::complement`] takes a dynamic cotarget, and a runtime stride
//!   only in a layout with a single mode (after filtering), so a divide
//!   takes a tiler with a runtime stride only if it has a single mode;
//!   [`Layout::right_inverse`] follows only static strides, ignoring modes
//!   with runtime strides as C++ CuTe does; [`Layout::left_inverse`] needs
//!   static strides.
//! * The products replicate a static layout; the tiler may be dynamic.
//! * What needs concrete numbers stays static and says so: [`Layout::upcast`],
//!   [`make_layout_tv`], register fragments, and the buffer sizes of shared
//!   memory and per-thread arrays. A tensor in global memory takes runtime
//!   extents (see [`crate::tensor`]); what a kernel unrolls over it when it
//!   is built (a copy's per-thread chunks and rest modes) stays static.
//! * Dynamic leaves are compared structurally, so `m * 2` and `m + m` are
//!   different leaves; equality is never decided by evaluating them. A
//!   dynamic extent is assumed to be at least 1.

//!
//! # Scaled-basis strides
//!
//! A stride leaf may also be a scaled basis element ([`StrideLeaf::Basis`],
//! printed `v@m`): such a layout maps a coordinate to a tuple of integers, an
//! arithmetic tuple, rather than to an offset. [`make_identity_layout`](super::make_identity_layout)
//! builds the one that maps every coordinate to itself, and TMA coordinate
//! layouts are of this kind. [`Layout::eval`] and [`Layout::slice_and_offset`]
//! give the tuple; [`Layout::offset`], [`crd2idx`] and [`Layout::slice`],
//! which give offsets, refuse basis strides. The algebra carries them as
//! C++ CuTe and cutegen do: composition scales them (and a basis stride on
//! the right-hand side picks a mode of the left-hand side), coalesce merges
//! only strides of the same mode that chain, the complement is taken per
//! leaf of the coprofile, and the divides and products follow; `cosize` is the size of the
//! coshape. Swizzles, `upcast`, `make_layout_like` and the inverses need
//! integer strides. A layout whose strides are all integers is untouched by
//! any of this.

use std::fmt;
use std::ops::{Deref, RangeFull};

use super::basis::{StrideLeaf, scale_coord, tuple_add};
use super::index::IndexExpr;
use super::tree::{HasShape, IntTuple, StaticLeaf, Tree, tree_from_tuple};

/// A coordinate whose leaves may be static or runtime values.
pub type Coord = Tree<IndexExpr>;

/// A shape: a tuple of extents, each a static integer or a runtime value
/// (printed `?`). The same tree type as [`Coord`].
pub type Shape = Tree<IndexExpr>;

/// A stride: a tuple of strides, each an integer (static or a runtime
/// value) or a scaled basis element ([`StrideLeaf`]). A layout whose strides
/// are all integers maps coordinates to offsets; one with basis elements maps
/// them to tuples (see [`Layout::eval`]).
pub type Stride = Tree<StrideLeaf>;

/// A coordinate for slicing: `None` leaves (written `..` when converting from
/// tuples) keep the corresponding mode, like CuTe's `_`.
pub type SliceCoord = Tree<Option<IndexExpr>>;

impl<T> Tree<Option<T>> {
    /// Whether a leaf is `None`, the `_` (written `..`) that keeps a mode
    /// (CuTe DSL's `has_underscore`), for a [`SliceCoord`] or a [`Tiler`].
    pub fn has_underscore(&self) -> bool {
        self.any_leaf(&mut |l| l.is_none())
    }
}

/// A tiler for divides and by-mode composition. A leaf layout applies to the
/// whole layout, a tuple applies mode by mode, and `None` (written `..`)
/// leaves a mode untouched, like CuTe's `_`.
pub type Tiler = Tree<Option<Layout>>;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    stride: Stride,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn st(v: usize) -> IndexExpr {
    IndexExpr::Static(v)
}

fn is_one(e: &IndexExpr) -> bool {
    e.as_static() == Some(1)
}

fn is_zero(e: &IndexExpr) -> bool {
    e.as_static() == Some(0)
}

/// An extent that is statically 0 or 1, the modes that coalescing,
/// complements and inverses skip.
fn is_trivial(e: &IndexExpr) -> bool {
    matches!(e.as_static(), Some(0 | 1))
}

impl Tree<IndexExpr> {
    /// The tuple as static integers, or `None` if a leaf is a runtime value.
    pub fn as_static(&self) -> Option<IntTuple> {
        match self {
            Tree::Leaf(e) => e.as_static().map(Tree::Leaf),
            Tree::Branch(children) => {
                children.iter().map(Tree::<IndexExpr>::as_static).collect::<Option<Vec<_>>>().map(Tree::Branch)
            }
        }
    }

    /// Whether every leaf is a static integer.
    pub fn is_static(&self) -> bool {
        self.leaves().iter().all(|e| e.is_static())
    }

    /// The first leaf that is not static, depth first, and its path.
    pub(crate) fn first_dynamic(&self) -> Option<(Vec<usize>, &IndexExpr)> {
        let path = self.find_if(|e, _| !e.is_static())?;
        let leaf = self.get(&path).as_leaf().expect("find_if returns the path of a leaf");
        Some((path, leaf))
    }

    /// The product of the leaves (CuTe's `product`, the size of a shape),
    /// static when every leaf is.
    pub fn product(&self) -> IndexExpr {
        self.leaves().into_iter().fold(st(1), |acc, e| acc * e)
    }

    /// The number of elements of a shape, for code that needs it when the
    /// kernel is built (to unroll a loop, say). Panics, naming the dynamic
    /// extent, when it is only known at run time; [`size`] (or
    /// [`product`](Tree::product)) gives an [`IndexExpr`] instead.
    #[track_caller]
    pub fn size(&self) -> usize {
        // The panics are in the body, not in closures, so that they report
        // the caller's location.
        if let Some(size) = self.product().as_static() {
            return size;
        }
        let Some((path, leaf)) = self.first_dynamic() else { unreachable!("a dynamic product has a dynamic leaf") };
        panic!(
            "size: shape {self} has the dynamic extent {leaf:?} at mode {path:?}, so its size is only known at run \
             time; use cute::size(&shape, &[]), an IndexExpr, where a static size is not needed"
        )
    }

    /// The size of each top-level mode, as a flat tuple (a leaf stays a leaf).
    pub fn product_each(&self) -> Shape {
        match self {
            Tree::Leaf(_) => self.clone(),
            Tree::Branch(children) => Tree::Branch(children.iter().map(|c| Tree::Leaf(c.product())).collect()),
        }
    }

    /// Compact column-major (CuTe `LayoutLeft`) strides with the same nesting:
    /// the exclusive prefix product of the leaves.
    pub fn compact_col_major(&self) -> Shape {
        let mut running = st(1);
        self.map(&mut |s| {
            let stride = running.clone();
            running = running.clone() * s;
            stride
        })
    }

    /// Compact row-major (CuTe `LayoutRight`) strides with the same nesting:
    /// the exclusive suffix product of the leaves.
    pub fn compact_row_major(&self) -> Shape {
        let shapes = self.flat_leaves();
        let mut strides = vec![st(0); shapes.len()];
        let mut running = st(1);
        for i in (0..shapes.len()).rev() {
            strides[i] = running.clone();
            running = running * &shapes[i];
        }
        Tree::unflatten(&strides, self)
    }
}

impl Layout {
    /// A layout of `shape` and `stride`, congruent tuples whose leaves may be
    /// static integers or runtime values, mixed freely:
    /// `Layout::new((m, 8), (8, 1))` with `m` an [`Int32`](crate::Int32).
    #[track_caller]
    pub fn new(shape: impl Into<Shape>, stride: impl Into<Stride>) -> Layout {
        let shape = shape.into();
        let stride = stride.into();
        assert!(shape.congruent(&stride), "layout shape {shape} and stride {stride} are not congruent");
        Layout { shape, stride }
    }

    /// [`Layout::new`], named for its common use with runtime strides:
    /// `Layout::with_strides((MAX_M, MAX_K), (k, 1))` is a row-major matrix
    /// whose row stride `k` is a runtime value, with static upper bounds as
    /// its extents (or runtime extents: `(m, k)`, which is
    /// `Layout::row_major((m, k))`).
    #[track_caller]
    pub fn with_strides(shape: impl Into<Shape>, stride: impl Into<Stride>) -> Layout {
        Layout::new(shape, stride)
    }

    /// Compact column-major layout of `shape` (CuTe's default). A dynamic
    /// extent makes the strides after it dynamic: `(m, n)` gets `(1, m)`.
    pub fn col_major(shape: impl Into<Shape>) -> Layout {
        let shape = shape.into();
        let stride = Stride::from(shape.compact_col_major());
        Layout { shape, stride }
    }

    /// Compact row-major layout of `shape`.
    pub fn row_major(shape: impl Into<Shape>) -> Layout {
        let shape = shape.into();
        let stride = Stride::from(shape.compact_row_major());
        Layout { shape, stride }
    }

    /// Concatenate layouts as the modes of a new layout.
    pub fn from_modes(modes: impl IntoIterator<Item = Layout>) -> Layout {
        let (shapes, strides): (Vec<_>, Vec<_>) = modes.into_iter().map(|l| (l.shape, l.stride)).unzip();
        Layout { shape: Tree::Branch(shapes), stride: Tree::Branch(strides) }
    }

    /// A flat layout from `(shape, stride)` pairs: no modes becomes `1:0`, one
    /// mode stays a single mode.
    pub fn from_flat(modes: &[(usize, usize)]) -> Layout {
        match modes {
            [] => Layout::new(1, 0),
            [(s, d)] => Layout::new(*s, *d),
            _ => Layout::from_modes(modes.iter().map(|&(s, d)| Layout::new(s, d))),
        }
    }

    /// The hierarchical shape (domain), whose leaves may be runtime values.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The hierarchical stride, congruent with the shape.
    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    /// The shape as integers. Panics, naming it, if an extent is dynamic.
    #[track_caller]
    pub fn static_shape(&self) -> IntTuple {
        if let Some(shape) = self.shape.as_static() {
            return shape;
        }
        let Some((path, leaf)) = self.shape.first_dynamic() else { unreachable!("a dynamic shape has a dynamic leaf") };
        panic!("layout {self} has the dynamic extent {leaf:?} at mode {path:?} where a static shape is needed")
    }

    /// The strides as integers. Panics, naming it, if a stride is a runtime
    /// value or a basis element.
    #[track_caller]
    pub fn static_stride(&self) -> IntTuple {
        if let Some(stride) = self.stride.as_static() {
            return stride;
        }
        if let Some((path, leaf)) = self.stride.first_basis() {
            panic!("layout {self} has the basis stride {leaf} at mode {path:?} where an integer one is needed")
        }
        let Some((path, leaf)) = self.stride.first_dynamic() else {
            unreachable!("a dynamic stride has a dynamic leaf")
        };
        panic!("layout {self} has the runtime stride {leaf:?} at mode {path:?} where a static one is needed")
    }

    /// Whether every extent and stride is known when the kernel is built
    /// (CuTe's `is_static`): a static integer, or a basis element with a
    /// static scale.
    pub fn is_static(&self) -> bool {
        self.shape.is_static() && self.stride.is_static()
    }

    /// Whether a stride is a scaled basis element (CuTe DSL's
    /// `has_scaled_basis`), so that the layout maps coordinates to tuples.
    pub fn has_scaled_basis(&self) -> bool {
        self.stride.has_scaled_basis()
    }

    /// Panic, naming `what` and the first basis stride, if there is one:
    /// for the operations that need integer strides.
    #[track_caller]
    pub(crate) fn require_int_strides(&self, what: &str, why: &str) {
        if let Some((path, leaf)) = self.stride.first_basis() {
            panic!("{what}: layout {self} has the basis stride {leaf} at mode {path:?}; {why}");
        }
    }

    /// The first dynamic leaf, extents before strides, as a message names
    /// it: `extent %3 at mode [0]`.
    fn dynamic_leaf(&self) -> Option<String> {
        if let Some((path, leaf)) = self.shape.first_dynamic() {
            return Some(format!("extent {leaf:?} at mode {path:?}"));
        }
        self.stride.first_dynamic().map(|(path, leaf)| format!("stride {leaf:?} at mode {path:?}"))
    }

    /// Panic, naming the first dynamic leaf, unless the layout is static.
    #[track_caller]
    pub(crate) fn require_static(&self, what: &str, why: &str) {
        if let Some(leaf) = self.dynamic_leaf() {
            panic!("{what}: layout {self} has the dynamic {leaf}; {why}");
        }
    }

    /// Number of top-level modes.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Nesting depth of the shape.
    pub fn depth(&self) -> usize {
        self.shape.depth()
    }

    /// Size of the domain, for code that needs it when the kernel is built.
    /// Panics, naming the dynamic extent, when it is only known at run time;
    /// [`size`]`(&layout, &[])` gives an [`IndexExpr`] instead.
    #[track_caller]
    pub fn size(&self) -> usize {
        if let Some(size) = self.shape.product().as_static() {
            return size;
        }
        let Some((path, leaf)) = self.shape.first_dynamic() else {
            unreachable!("a dynamic size has a dynamic extent")
        };
        panic!(
            "Layout::size: layout {self} has the dynamic extent {leaf:?} at mode {path:?}, so its size is only known \
             at run time; use cute::size(&layout, &[]), an IndexExpr, where a static size is not needed"
        )
    }

    /// Size of the codomain: one past the largest offset. Panics, naming the
    /// leaf, when it depends on a runtime extent or stride;
    /// [`cosize`]`(&layout, &[])` gives an [`IndexExpr`] instead.
    #[track_caller]
    pub fn cosize(&self) -> usize {
        if let Some(cosize) = cosize_of(&self.shape, &self.stride).as_static() {
            return cosize;
        }
        let modes = self.flat_modes_expr();
        let dynamic = |(s, d): &&(IndexExpr, StrideLeaf)| match d {
            StrideLeaf::Int(d) => !((s.clone() - st(1)) * d).is_static(),
            StrideLeaf::Basis(b) => !s.is_static() || !b.is_static(),
        };
        let Some((s, d)) = modes.iter().find(dynamic) else { unreachable!("a dynamic cosize has a dynamic mode") };
        let leaf = if s.is_static() { format!("stride {d:?}") } else { format!("extent {s:?}") };
        panic!(
            "Layout::cosize: layout {self} has the dynamic {leaf}, so its cosize is only known at run time; use \
             cute::cosize(&layout, &[]), an IndexExpr, where a static cosize is not needed"
        )
    }

    /// The `i`-th top-level mode as a layout.
    pub fn mode(&self, i: usize) -> Layout {
        Layout { shape: self.shape.mode(i).clone(), stride: self.stride.mode(i).clone() }
    }

    /// All top-level modes.
    pub fn modes(&self) -> Vec<Layout> {
        (0..self.rank()).map(|i| self.mode(i)).collect()
    }

    /// Nested mode selection: `get(&[1, 0])` is mode 0 of mode 1.
    pub fn get(&self, path: &[usize]) -> Layout {
        path.iter().fold(self.clone(), |l, &i| l.mode(i))
    }

    /// `(shape, stride)` of every leaf mode, in order, as integers. Panics if
    /// an extent or a stride is dynamic.
    #[track_caller]
    pub fn flat_modes(&self) -> Vec<(usize, usize)> {
        self.static_shape().flat_leaves().into_iter().zip(self.static_stride().flat_leaves()).collect()
    }

    fn flat_modes_expr(&self) -> Vec<(IndexExpr, StrideLeaf)> {
        self.shape.flat_leaves().into_iter().zip(self.stride.flat_leaves()).collect()
    }

    /// [`flat_modes_expr`](Layout::flat_modes_expr) of a layout the caller
    /// knows has integer strides.
    fn flat_int_modes(&self) -> Vec<(IndexExpr, IndexExpr)> {
        self.shape.flat_leaves().into_iter().zip(self.stride.leaves().into_iter().map(|d| d.int().clone())).collect()
    }

    fn from_flat_expr<D: Clone + Into<StrideLeaf>>(modes: &[(IndexExpr, D)]) -> Layout {
        let one =
            |(s, d): &(IndexExpr, D)| Layout { shape: Tree::Leaf(s.clone()), stride: Tree::Leaf(d.clone().into()) };
        match modes {
            [] => Layout::new(1, 0),
            [m] => one(m),
            _ => Layout::from_modes(modes.iter().map(one)),
        }
    }

    /// The same layout with all nesting removed.
    pub fn flatten(&self) -> Layout {
        Layout { shape: self.shape.flatten(), stride: self.stride.flatten() }
    }

    /// Evaluate at a 1-D index. Panics if the offset depends on a runtime
    /// value; [`offset_at`](Layout::offset_at) keeps it symbolic.
    #[track_caller]
    pub fn call(&self, idx: usize) -> usize {
        let off = self.offset_at(idx);
        let Some(off) = off.as_static() else {
            panic!(
                "Layout::call: layout {self} maps index {idx} to {off:?}, which depends on runtime values; use offset_at"
            )
        };
        off
    }

    /// Evaluate at a 1-D index, keeping any runtime strides symbolic.
    pub fn offset_at(&self, idx: usize) -> IndexExpr {
        self.offset(&Tree::Leaf(st(idx)))
    }

    /// Evaluate at a static (possibly hierarchical) coordinate. Panics if the
    /// offset depends on a runtime value; [`offset`](Layout::offset) and
    /// [`crd2idx`] keep it symbolic.
    #[track_caller]
    pub fn crd2idx(&self, coord: &IntTuple) -> usize {
        let off = self.offset_checked(&coord.into(), "Layout::crd2idx");
        let Some(off) = off.as_static() else {
            panic!(
                "Layout::crd2idx: layout {self} maps {coord} to {off:?}, which depends on runtime values; use \
                 offset (or cute::crd2idx) for an IndexExpr"
            )
        };
        off
    }

    /// The coordinate of 1-D index `idx`, nested like the shape. Panics if it
    /// depends on a dynamic extent; [`idx2crd`] keeps it symbolic.
    #[track_caller]
    pub fn idx2crd(&self, idx: usize) -> IntTuple {
        let c = idx2crd(idx, &self.shape);
        let Some(c) = c.as_static() else {
            panic!(
                "Layout::idx2crd: the coordinate of index {idx} in {self} is {c:?}, which depends on dynamic \
                 extents; use cute::idx2crd for a Coord"
            )
        };
        c
    }

    /// Evaluate symbolically at `coord`. A leaf coordinate against a
    /// hierarchical mode is a 1-D index into that mode, split by the mode's
    /// extents (with a division and a remainder for each dynamic one). A
    /// tuple coordinate must have the rank of the mode it meets, except that
    /// a tuple of one coordinate stands for that coordinate; otherwise this
    /// panics, naming the mode.
    #[track_caller]
    pub fn offset(&self, coord: &Coord) -> IndexExpr {
        self.offset_checked(coord, "Layout::offset")
    }

    /// [`offset`](Layout::offset), whose misuse message names `what`.
    #[track_caller]
    pub(crate) fn offset_checked(&self, coord: &Coord, what: &str) -> IndexExpr {
        check_coord(what, coord, self, Coord::to_string);
        self.require_offsets(what, "use Layout::eval for the tuple it maps a coordinate to");
        offset_impl(coord, &self.shape, &self.stride)
    }

    /// Panic, naming `what`, if the layout maps coordinates to tuples rather
    /// than offsets; `instead` says what to use.
    #[track_caller]
    fn require_offsets(&self, what: &str, instead: &str) {
        if let Some((path, leaf)) = self.stride.first_basis() {
            panic!(
                "{what}: layout {self} has the basis stride {leaf} at mode {path:?}, so it maps a coordinate to a \
                 tuple, not an offset; {instead}"
            );
        }
    }

    /// Evaluate at `coord`, as [`offset`](Layout::offset) does, for any
    /// layout: the arithmetic tuple that the strides sum to (CuTe DSL's
    /// `crd2idx` of a layout with basis strides, cutegen's `layout_eval`).
    /// Each basis stride `v@m` adds `c * v` at the mode path `m` of the
    /// result, which has one entry for each mode the basis strides name: at
    /// `(2,3)`, `(4,8):(2@0,1@1)` gives `(4,3)`, and at `(1,2)`,
    /// `(4,8):(1@1,1@0)` gives `(2,1)`. Basis strides with a coefficient of 0 still give
    /// their mode an entry, so the result has the same nesting for every
    /// coordinate. A layout with integer strides gives its offset as a leaf.
    /// A basis layout may also have strides of a static 0, but no other
    /// integer strides (they would have no mode to add to): those panic, as
    /// do two basis strides whose mode paths conflict (`1@2` and `1@0@2`). A
    /// stride whose scale is a [`Ratio`](super::Ratio) needs a static
    /// coordinate that makes the product a whole number.
    #[track_caller]
    pub fn eval(&self, coord: &Coord) -> Coord {
        self.eval_checked(coord, "Layout::eval")
    }

    #[track_caller]
    pub(crate) fn eval_checked(&self, coord: &Coord, what: &str) -> Coord {
        check_coord(what, coord, self, Coord::to_string);
        if !self.has_scaled_basis() {
            return Tree::Leaf(offset_impl(coord, &self.shape, &self.stride));
        }
        let mut acc = Tree::Leaf(st(0));
        if let Err(why) = eval_impl(coord, &self.shape, &self.stride, &mut acc) {
            panic!("{what}: layout {self} at {coord} is not defined: {why}");
        }
        acc
    }

    /// Slice with `coord`: fixed leaves contribute to the returned offset,
    /// `None` leaves keep their mode. The kept modes form the new layout, in
    /// order (CuTe's `slice_and_offset`). The coordinate must fit the shape
    /// as in [`offset`](Layout::offset).
    #[track_caller]
    pub fn slice(&self, coord: &SliceCoord) -> (Layout, IndexExpr) {
        self.slice_checked(coord, "Layout::slice")
    }

    /// [`slice`](Layout::slice), whose misuse message names `what`.
    #[track_caller]
    pub(crate) fn slice_checked(&self, coord: &SliceCoord, what: &str) -> (Layout, IndexExpr) {
        check_coord(what, coord, self, slice_view);
        self.require_offsets(what, "use slice_and_offset for the tuple its offset is");
        if let Tree::Leaf(None) = coord {
            return (self.clone(), st(0));
        }
        let fixed = coord.map(&mut |c| c.clone().unwrap_or(st(0)));
        let offset = offset_impl(&fixed, &self.shape, &self.stride);
        let mut kept = Vec::new();
        collect_kept(coord, &self.shape, &self.stride, &mut kept);
        (Layout::from_modes(kept), offset)
    }

    /// Slice with `coord`, as [`slice`](Layout::slice) does, for any layout
    /// (CuTe's `slice_and_offset`): the kept modes, and the offset of the
    /// fixed leaves as [`eval`](Layout::eval) gives it, a tuple for a layout
    /// with basis strides (the kept modes count with coordinate 0, so the
    /// tuple has the same nesting as every value of the layout) and a leaf
    /// for one with integer strides.
    #[track_caller]
    pub fn slice_and_offset(&self, coord: &SliceCoord) -> (Layout, Coord) {
        self.slice_and_offset_in(coord, "Layout::slice_and_offset")
    }

    /// [`slice_and_offset`](Layout::slice_and_offset), whose misuse message
    /// names `what`.
    #[track_caller]
    pub(crate) fn slice_and_offset_in(&self, coord: &SliceCoord, what: &str) -> (Layout, Coord) {
        if !self.has_scaled_basis() {
            let (layout, offset) = self.slice_checked(coord, what);
            return (layout, Tree::Leaf(offset));
        }
        let fixed = coord.map(&mut |c| c.clone().unwrap_or(st(0)));
        let offset = self.eval_checked(&fixed, what);
        if let Tree::Leaf(None) = coord {
            return (self.clone(), offset);
        }
        let mut kept = Vec::new();
        collect_kept(coord, &self.shape, &self.stride, &mut kept);
        (Layout::from_modes(kept), offset)
    }

    /// For each top-level mode, the 1-D coordinate within that mode of the
    /// element this layout maps to `idx`. The layout must be static and a
    /// bijection onto `[0, size)` (thread layouts usually are).
    #[track_caller]
    pub fn coord_of(&self, idx: impl Into<IndexExpr>) -> Coord {
        self.require_static("coord_of", "a thread layout must be a static bijection");
        let inverse = self.right_inverse();
        assert_eq!(inverse.size(), self.size(), "coord_of: layout {self} is not a bijection onto [0, {})", self.size());
        let linear = inverse.offset(&Tree::Leaf(idx.into()));
        let per_mode = self.shape.product_each();
        match &per_mode {
            Tree::Leaf(_) => Tree::Leaf(linear),
            Tree::Branch(_) => idx2crd(linear, &per_mode),
        }
    }

    /// Merge adjacent modes that are contiguous with each other and drop
    /// size-1 modes. The result is flat and computes the same function. Two
    /// modes `s0:d0` and `s1:d1` merge into `(s0*s1):d0` when `s0 * d0 ==
    /// d1` is provable: always for static leaves, and for runtime leaves when
    /// the two sides are the same expression up to the order of factors
    /// ([`IndexExpr::same_value`]). A merge whose condition involves an
    /// unknown value is left undone, as in CuTe: `(4,5,?,3,2):(1,4,?,1,3)`
    /// coalesces to `(20,?,6):(1,?,1)`. Where CuTe also leaves a provable
    /// merge undone because the merged extent is dynamic, cute-rs merges:
    /// `(4,?):(1,4)` becomes `?:1`, the same function. Basis strides merge
    /// only with basis strides of the same mode whose scales chain the same
    /// way: `(64,2):(1@0,64@0)` becomes `128:1@0`.
    pub fn coalesce(&self) -> Layout {
        let mut out: Vec<(IndexExpr, StrideLeaf)> = Vec::new();
        for (s, d) in self.flat_modes_expr() {
            if is_one(&s) {
                continue;
            }
            if let Some(last) = out.last_mut()
                && last.1.try_scale(&last.0).is_some_and(|next| next.same_value(&d))
            {
                last.0 = last.0.clone() * s;
                continue;
            }
            out.push((s, d));
        }
        Layout::from_flat_expr(&out)
    }

    /// Coalesce while preserving the structure given by `profile`: a leaf in
    /// the profile coalesces the corresponding sub-layout completely (CuTe's
    /// `coalesce(layout, trg_profile)`).
    pub fn coalesce_profile(&self, profile: &IntTuple) -> Layout {
        match profile {
            Tree::Leaf(_) => self.coalesce(),
            Tree::Branch(ps) => Layout::from_modes((0..self.rank()).map(|i| match ps.get(i) {
                Some(p) => self.mode(i).coalesce_profile(p),
                None => self.mode(i),
            })),
        }
    }

    /// Replace modes with a static stride of 0 by size-1 modes.
    pub fn filter_zeros(&self) -> Layout {
        let mut stride = self.stride.leaves().into_iter();
        let shape = self.shape.map(&mut |s| if stride.next().unwrap().is_zero() { st(1) } else { s.clone() });
        Layout { shape, stride: self.stride.clone() }
    }

    /// Remove stride-0 and size-1 modes, then coalesce.
    pub fn filter(&self) -> Layout {
        self.filter_zeros().coalesce()
    }

    /// The kernel of the layout (CuTe's `nullspace`): a layout `N` from
    /// `[0, size(N))` to 1-D indices into this layout's domain, one for each
    /// coordinate that only moves along modes whose stride is a static 0, so
    /// `self(N(i)) == 0` for every `i`. Its modes are those modes, flat and
    /// in order, with their strides in the compact column-major layout of
    /// this layout's flattened shape: each is the product of all the extents
    /// before it, size-1 modes included, as in CuTe DSL (and
    /// [`Layout::col_major`]). So `nullspace((2,1,3):(0,0,1))` is
    /// `(2,1):(1,2)`, and `nullspace((1):(0))` is `1:1`. CuTe DSL raises when
    /// the shape is a single integer and the stride 0; there cute-rs gives
    /// C++ CuTe's answer, so `nullspace(4:0)` is `4:1` and `nullspace(1:0)`
    /// is `1:0`. Without zero-stride modes it is `1:0`. A runtime stride never
    /// counts as zero; a dynamic extent makes the strides after it dynamic.
    /// `size(N) == size(self) / size(filter_zeros(self))`, `N` is injective,
    /// and `nullspace(N) == 1:0`.
    pub fn nullspace(&self) -> Layout {
        // CuTe DSL's strides, the plain prefix products of the flat shape.
        // C++ CuTe takes `compact_major<LayoutLeft>`, which gives a size-1
        // mode stride 0, so for `(2,1,3):(0,0,1)` it returns `(2,1):(1,0)`
        // where CuTe DSL and cute-rs return `(2,1):(1,2)`. The two compute
        // the same function, but `Layout` equality compares the strides.
        // CuTe DSL has no answer for a shape that is a single integer with
        // stride 0: it calls `len()` on the flattened shape, which is then
        // that integer.
        // C++ gives the integer 1 stride 0, so `nullspace(1:0)` is `1:0`,
        // which keeps `nullspace(nullspace(L)) == 1:0` exact. Any other
        // integer `n` gets stride 1 from either rule: `nullspace(n:0)` is
        // `n:1`.
        let shapes = self.shape.flat_leaves();
        let compact = match &self.shape {
            Tree::Leaf(s) if is_one(s) => vec![st(0)],
            _ => self.shape.compact_col_major().flat_leaves(),
        };
        let modes: Vec<(IndexExpr, IndexExpr)> = self
            .stride
            .leaves()
            .into_iter()
            .enumerate()
            .filter(|(_, d)| d.is_zero())
            .map(|(i, _)| (shapes[i].clone(), compact[i].clone()))
            .collect();
        Layout::from_flat_expr(&modes)
    }

    /// The position of the leading dimension (CuTe DSL's `leading_dim(shape,
    /// stride)`): the first mode, depth first, whose stride is a static 1
    /// and whose extent is not 1 (a dynamic extent counts as not 1). `[i]`
    /// is top-level mode `i`, `[i, j]` mode `j` of mode `i`, and `[0]` the
    /// whole layout when its shape is a single integer, as in `8:1`. `None`
    /// if there is no such mode: no stride is a static 1, or every mode with
    /// a static unit stride has extent 1, as in `(1,1):(1,1)`.
    pub fn leading_dim(&self) -> Option<Vec<usize>> {
        leading_dim(&self.shape, &self.stride)
    }

    /// The largest `n` such that the layout maps every aligned block of `n`
    /// consecutive indices to `n` consecutive offsets starting at a multiple
    /// of `n` (C++ CuTe's `max_alignment(layout)`, CuTe DSL's
    /// `Layout.max_alignment`): the layout is coalesced, and with `f` that
    /// layout, `permuted = logical_divide(f, right_inverse(f))` gives the gcd
    /// of `size(permuted[0])` and the strides of `permuted[1]`. So
    /// `(8,16):(1,4)` gives 4 and `(8,16):(1,6)` gives 2.
    ///
    /// Runtime leaves count with what is known to divide them, as CuTe DSL's
    /// dynamic leaves count with their divisibility, where C++ CuTe takes 1
    /// for a runtime extent and 0 for a runtime stride: a runtime extent
    /// enters `f` as its known divisor (a [`param_extent`](crate::param_extent)'s
    /// `multiple_of`), so `m:1` with `m` a multiple of 16 gives 16; a runtime
    /// stride enters `f` as 0 and the result is then bounded by its known
    /// divisor, so `(4,2):(1,k)` gives 4 when `k` is a multiple of 4 and 1
    /// when nothing is known of it. A runtime extent `m` with static stride
    /// `s` also bounds the result by `divisor(m) * s`, the part of its run
    /// known to be contiguous: `(m,4):(1,16)` with `m` a multiple of 16
    /// gives 16, not the 64 that `(16,4):(1,16)` would. A layout with
    /// scaled-basis strides gives 1.
    pub fn max_alignment(&self) -> usize {
        if self.has_scaled_basis() {
            return 1;
        }
        let mut bound = 0;
        let flat: Vec<(IndexExpr, IndexExpr)> = self
            .coalesce()
            .flat_modes_expr()
            .into_iter()
            .map(|(s, d)| {
                let s = if s.is_static() {
                    s
                } else {
                    // The runtime extent `m` enters as its divisor `k`, but
                    // only `k` of its offsets are known to be contiguous:
                    // whatever follows `k * stride` in `f` does not follow
                    // `m * stride` in the layout, so the run stops there.
                    let k = s.known_divisor().max(1);
                    if let Some(stride) = d.as_int().and_then(|d| d.as_static())
                        && stride != 0
                    {
                        bound = gcd(bound, k * stride);
                    }
                    st(k)
                };
                let d = match d.as_int() {
                    Some(d) if d.is_static() => d.clone(),
                    Some(d) => {
                        bound = gcd(bound, d.known_divisor());
                        st(0)
                    }
                    None => st(0),
                };
                (s, d)
            })
            .collect();
        let f = Layout::from_flat_expr(&flat);
        let permuted = f.logical_divide(&Tree::Leaf(Some(f.right_inverse())));
        let (m0, m1) = (permuted.mode(0), permuted.mode(1));
        let a = m1.stride.leaves().into_iter().fold(m0.size(), |g, d| gcd(g, d.int().as_static().unwrap_or(0)));
        gcd(a, bound)
    }

    /// Whether top-level mode `mode` is major, its first stride being a
    /// static 1 (CuTe DSL's `is_major(mode, stride)`). A runtime stride is
    /// never major, even if it is 1 at launch. Panics if the layout has no
    /// mode `mode`, or that mode has no first stride (it is, or starts with,
    /// an empty tuple).
    #[track_caller]
    pub fn is_major(&self, mode: usize) -> bool {
        first_stride_is_one(mode, &self.stride, || format!("layout {self}"))
    }

    /// Functional composition `self ∘ rhs`: `(self ∘ rhs)(c) = self(rhs(c))`.
    /// The result has the shape of `rhs`. The dynamic extents of `self` are
    /// assumed large and divisible; where `rhs`'s runtime values cannot be
    /// placed when the kernel is built, the extents are computed at run time
    /// (see the module documentation).
    #[track_caller]
    pub fn compose(&self, rhs: &Layout) -> Layout {
        self.compose_in(rhs, &ModePath::whole(self))
    }

    /// [`compose`](Layout::compose), where `self` is the mode `at` names.
    #[track_caller]
    fn compose_in(&self, rhs: &Layout, at: &ModePath) -> Layout {
        let lhs = self.coalesce().flat_modes_expr();
        compose_impl(&lhs, &rhs.shape, &rhs.stride, self, rhs, at)
    }

    /// Composition with a tiler: by mode for tuples, whole for a layout.
    #[track_caller]
    pub fn compose_tiler(&self, tiler: &Tiler) -> Layout {
        self.compose_tiler_in(tiler, &ModePath::whole(self))
    }

    #[track_caller]
    fn compose_tiler_in(&self, tiler: &Tiler, at: &ModePath) -> Layout {
        match tiler {
            Tree::Leaf(None) => self.clone(),
            Tree::Leaf(Some(b)) => self.compose_in(b, at),
            Tree::Branch(ts) => self.by_mode(ts, ByMode::Compose, at),
        }
    }

    /// Apply `op` with each mode of the tiler `ts` to the matching mode of
    /// `self`. The operation is called here rather than through a closure,
    /// so that its messages report the caller's location.
    #[track_caller]
    fn by_mode(&self, ts: &[Tiler], op: ByMode, at: &ModePath) -> Layout {
        let what = match op {
            ByMode::Compose => "composition",
            ByMode::Divide => "logical_divide",
        };
        assert!(
            ts.len() <= self.rank(),
            "{what}: the tiler has {} modes but {} has rank {}",
            ts.len(),
            at.describe(self),
            self.rank()
        );
        let mut modes = Vec::with_capacity(self.rank());
        for i in 0..self.rank() {
            let mode = self.mode(i);
            modes.push(match (ts.get(i), op) {
                (None, _) => mode,
                (Some(t), ByMode::Compose) => mode.compose_tiler_in(t, &at.child(i)),
                (Some(t), ByMode::Divide) => mode.logical_divide_in(t, &at.child(i)),
            });
        }
        Layout::from_modes(modes)
    }

    /// The layout that, concatenated with `self`, covers `[0, cotarget)`
    /// without overlap and in increasing stride order. The cotarget may be a
    /// runtime value: the last mode's extent is then `ceil_div(cotarget,
    /// cosize)`. A runtime stride is accepted only when the layout has a
    /// single mode after filtering, as in CuTe.
    ///
    /// For a layout with basis strides, the complement is taken for each
    /// leaf of the layout's coprofile (the nesting of the tuples it maps to,
    /// so every mode a basis stride names, and the modes before it), in
    /// depth-first order, as cutegen does with an integer cotarget: the modes
    /// with strides `v@m` form an integer layout, whose complement in
    /// `[0, cotarget)` gets the strides `@m` back. So
    /// `complement((4,3):(1@0,1@1), 120)` is `(30,40):(4@0,3@1)`, which is
    /// what the products of identity layouts replicate along, and a
    /// coprofile mode that no stride walks gets all of it:
    /// `complement((4,1):(1@0,1@1), 8)` is `(2,8):(4@0,1@1)`. The basis
    /// strides need integer scales; the layout may also have strides of a
    /// static 0, but no other integer strides.
    #[track_caller]
    pub fn complement(&self, cotarget: impl Into<IndexExpr>) -> Layout {
        let cotarget = cotarget.into();
        if self.has_scaled_basis() {
            return self.complement_basis(cotarget);
        }
        let Some((mut result, current)) = self.complement_prefix() else {
            let extent = match cotarget.as_static() {
                Some(c) => st(c.max(1)),
                None => cotarget,
            };
            return Layout { shape: Tree::Leaf(extent), stride: Tree::Leaf(StrideLeaf::Int(st(1))) };
        };
        result.push((IndexExpr::ceil_div(cotarget, current.clone()), current));
        Layout::from_flat_expr(&result).coalesce()
    }

    /// [`complement`](Layout::complement) of a layout with basis strides.
    #[track_caller]
    fn complement_basis(&self, cotarget: IndexExpr) -> Layout {
        // One bucket for each leaf of the coprofile, the value at 0; a
        // `BTreeMap` of paths keeps them in depth-first order.
        let mut groups: std::collections::BTreeMap<Vec<usize>, Vec<(IndexExpr, IndexExpr)>> = Default::default();
        let coprofile = self.eval_checked(&Tree::Leaf(st(0)), "complement");
        coprofile.find_if(|_, path| {
            groups.insert(path.to_vec(), Vec::new());
            false
        });
        for (s, d) in self.flat_modes_expr() {
            if is_trivial(&s) || d.is_zero() {
                continue;
            }
            match &d {
                StrideLeaf::Basis(e) => {
                    let Some(v) = e.value().as_int() else {
                        panic!("complement: layout {self} has the basis stride {e}, whose scale is a ratio")
                    };
                    groups.entry(e.mode().to_vec()).or_default().push((s, v.clone()));
                }
                StrideLeaf::Int(v) => panic!(
                    "complement: layout {self} mixes basis strides with the integer stride {v}; the complement of a \
                     basis layout takes only basis strides and strides of 0"
                ),
            }
        }
        if let Some((mode, _)) =
            groups.iter().find(|(_, modes)| modes.len() > 1 && modes.iter().any(|(_, d)| !d.is_static()))
        {
            panic!(
                "complement: layout {self} has a runtime basis scale among the strides of mode {mode:?}; CuTe takes \
                 the complement of a runtime stride only when it is the only one"
            )
        }
        let modes = groups.into_iter().map(|(mode, modes)| {
            let c = Layout::from_flat_expr(&modes).complement(cotarget.clone());
            let stride = c.stride.map(&mut |d| {
                if d.is_zero() {
                    d.clone()
                } else {
                    StrideLeaf::Basis(super::basis::ScaledBasis::new(d.int().clone(), mode.clone()))
                }
            });
            Layout { shape: c.shape, stride }
        });
        Layout::from_modes(modes)
    }

    /// The complement of `self` with no cotarget (cutegen's one-argument
    /// `complement`, as pycute defines it): the modes that fill the gaps
    /// between `self`'s strides, in increasing stride order, followed by a
    /// mode of extent 1 whose stride is where `self` ends, so that the
    /// result can be extended past it. So `complement_no_cotarget(5:3)` is
    /// `(3,1):(1,15)` and `complement_no_cotarget((2,4):(1,2))` is `1:8`.
    /// Strides that are runtime values are taken only in a layout with a
    /// single mode.
    ///
    /// For a layout with basis strides, the complement is taken for each
    /// leaf of the coprofile, and the result nested like the coprofile, with
    /// the strides `@m` of each leaf's modes: `complement_no_cotarget(3:4@2)`
    /// is `(1,1,(4,1)):(1@0,1@1,(1@2,12@2))`. The basis scales must be
    /// integers, and the other strides static 0s.
    #[track_caller]
    pub fn complement_no_cotarget(&self) -> Layout {
        let what = "complement_no_cotarget";
        let modes = self.flat_modes_expr();
        let runtime = modes.iter().filter(|(_, d)| !d.is_static()).count();
        if runtime > 1 || (runtime == 1 && modes.len() != 1) {
            panic!(
                "{what}: layout {self} has a runtime stride among {} modes; the complement takes a runtime stride only \
                 in a layout with a single mode",
                modes.len()
            );
        }
        let coprofile = self.coprofile(what);
        let mut buckets = coprofile_buckets(&coprofile);
        for (s, d) in modes {
            if let Some((path, v)) = self.bucket_of(&d, what) {
                buckets.entry(path).or_default().push((v, s, st(0)));
            }
        }
        by_coprofile(&coprofile, &mut |path| {
            let mut list = buckets.remove(path).unwrap_or_default();
            list.sort_by_key(|(d, _, _)| d.as_static());
            let (mut shapes, mut strides) = (Vec::new(), vec![st(1)]);
            for (d, s, _) in list {
                if is_zero(&d) || is_one(&s) {
                    continue;
                }
                let last = strides.last().expect("starts with 1").clone();
                if let (Some(dv), Some(lv)) = (d.as_static(), last.as_static()) {
                    assert!(dv >= lv, "{what}: layout {self} is not injective");
                }
                shapes.push(d.clone() / last);
                strides.push(d * s);
            }
            shapes.push(st(1));
            with_basis(coalesce_z(&shapes, &strides), path)
        })
    }

    /// The coprofile (cutegen's `coprofile`): the nesting of the tuples the
    /// layout maps coordinates to, a leaf for integer strides.
    #[track_caller]
    fn coprofile(&self, what: &str) -> Coord {
        if self.has_scaled_basis() { self.eval_checked(&Tree::Leaf(st(0)), what) } else { Tree::Leaf(st(0)) }
    }

    /// The coprofile leaf a stride walks and its integer value, or `None`
    /// for a stride of a static 0. A layout with basis strides takes no
    /// other integer strides, and a basis scale must be an integer.
    #[track_caller]
    fn bucket_of(&self, d: &StrideLeaf, what: &str) -> Option<(Vec<usize>, IndexExpr)> {
        match d {
            StrideLeaf::Int(v) if is_zero(v) => None,
            StrideLeaf::Int(v) if self.has_scaled_basis() => panic!(
                "{what}: layout {self} mixes basis strides with the integer stride {v}; it takes only basis strides \
                 and strides of 0"
            ),
            StrideLeaf::Int(v) => Some((Vec::new(), v.clone())),
            StrideLeaf::Basis(e) => match e.value().as_int() {
                Some(v) => Some((e.mode().to_vec(), v.clone())),
                None => panic!("{what}: layout {self} has the basis stride {e}, whose scale is a ratio"),
            },
        }
    }

    /// The right (`left == false`) or left inverse of a layout with basis
    /// strides, as cutegen computes them.
    #[track_caller]
    fn basis_inverse(&self, left: bool) -> Layout {
        let what = if left { "left_inverse" } else { "right_inverse" };
        if let Some((path, leaf)) = self.shape.first_dynamic() {
            panic!(
                "{what}: layout {self} has the dynamic extent {leaf:?} at mode {path:?}; the inverse of a basis layout needs static extents"
            );
        }
        // Runtime basis scales count as strides of 0, which only shorten a
        // right inverse; a left inverse needs them static.
        let flat: Vec<(IndexExpr, StrideLeaf)> = self
            .flat_modes_expr()
            .into_iter()
            .map(|(s, d)| {
                if d.is_static() {
                    return (s, d);
                }
                assert!(
                    !left,
                    "{what}: layout {self} has the runtime basis stride {d}; a left inverse needs static strides"
                );
                (s, StrideLeaf::Int(st(0)))
            })
            .collect();
        let coalesced = Layout::from_flat_expr(&flat).coalesce().flat_modes_expr();
        let coprofile = self.coprofile(what);
        let mut buckets = coprofile_buckets(&coprofile);
        let mut prefix = st(1);
        for (s, d) in coalesced {
            if let Some((path, v)) = self.bucket_of(&d, what) {
                buckets.entry(path).or_default().push((v, s.clone(), prefix.clone()));
            }
            prefix = prefix * s;
        }
        by_coprofile(&coprofile, &mut |path| {
            let mut list = buckets.remove(path).unwrap_or_default();
            list.sort_by_key(|(d, _, _)| d.as_static());
            if left {
                let (mut shapes, mut strides, mut current) = (vec![st(1)], vec![st(0)], st(1));
                for (d, s, p) in list {
                    if is_zero(&d) || is_one(&s) {
                        continue;
                    }
                    assert!(
                        (d.clone() % current.clone()).as_static() == Some(0),
                        "{what}: layout {self} is not injective, or its strides do not divide each other"
                    );
                    let gap = d / current.clone();
                    current = current * gap.clone();
                    *shapes.last_mut().expect("starts with 1") = gap;
                    shapes.push(s);
                    strides.push(p);
                }
                coalesce_z(&shapes, &strides)
            } else {
                let (mut modes, mut current) = (Vec::new(), st(1));
                for (d, s, p) in list {
                    if is_zero(&d) || is_one(&s) || !d.same_value(&current) {
                        continue;
                    }
                    current = s.clone() * d;
                    modes.push((s, p));
                }
                if modes.is_empty() { Layout::new(1, 0) } else { Layout::from_flat_expr(&modes).coalesce() }
            }
        })
    }

    /// cutegen's `complement_extend(self, extend)`, the rest a tiler of
    /// basis strides leaves in a layout of shape `extend`: the complement
    /// with no cotarget, its last mode at each coprofile leaf extended over
    /// the extents of the matching mode of `extend`, and the modes of
    /// `extend` that the tiler does not walk kept whole.
    #[track_caller]
    fn complement_extend(&self, extend: &Shape) -> Layout {
        let complement = self.complement_no_cotarget();
        let coprofile = self.coprofile("logical_divide");
        extend_rest(Some((&coprofile, &complement)), extend, &mut Vec::new(), self)
    }

    /// The modes of the complement that fill the gaps of `self`, and the
    /// stride where the rest of the cotarget starts (C++ CuTe's
    /// `complement_prefix`), or `None` if `self` has no modes after
    /// filtering.
    #[track_caller]
    fn complement_prefix(&self) -> Option<(Vec<(IndexExpr, IndexExpr)>, IndexExpr)> {
        let mut modes = self.complement_modes();
        if modes.is_empty() {
            return None;
        }
        if let Some(d) = unordered_runtime_stride(&modes) {
            panic!(
                "complement: layout {self} has the runtime stride {d:?}; CuTe takes the complement of a layout with a \
                 runtime stride only when it has a single mode, and this one has {} after filtering",
                modes.len()
            );
        }
        modes.sort_by_key(|(_, d)| d.as_static());
        let mut result = Vec::new();
        let mut current = st(1);
        for (s, d) in modes {
            if let (Some(dv), Some(cv)) = (d.as_static(), current.as_static()) {
                assert!(
                    dv >= cv && dv % cv == 0,
                    "complement: layout {self} is not injective or its strides do not divide each other"
                );
            }
            result.push((d.clone() / current.clone(), current));
            current = s * d;
        }
        Some((result, current))
    }

    /// The complement of `self` in the cotarget shape `cotarget`, the form
    /// C++ CuTe's `logical_divide` takes (`complement(tiler,
    /// shape(coalesce(layout)))`): the rest of the cotarget is split over its
    /// modes, `ceil_div` folded across them as in C++ CuTe's `ceil_div` of a
    /// tuple by an integer, so that the rest modes line up with the modes of
    /// the layout being divided. A dynamic extent of the cotarget is taken to
    /// be large and divisible, so it takes whatever is left of the tile's
    /// span, and the strides after it are written as the products of the
    /// cotarget's extents that they equal under that assumption: dividing
    /// all of `(m,n)` by `32:1` leaves `(ceil_div(m,32),n):(32,m)`, where C++
    /// CuTe's strides are `(32,32*ceil_div(m,32))`. A static cotarget, or a
    /// tile whose span is a runtime value, takes the plain complement of the
    /// cotarget's size, as [`complement`](Layout::complement) does.
    #[track_caller]
    fn complement_to_shape(&self, cotarget: &Shape) -> Layout {
        let extents = cotarget.flat_leaves();
        if cotarget.is_static() || extents.iter().any(is_zero) {
            return self.complement(cotarget.product());
        }
        let (mut result, current) = self.complement_prefix().unwrap_or_else(|| (Vec::new(), st(1)));
        let Some(mut span) = current.as_static() else {
            result.push((IndexExpr::ceil_div(cotarget.product(), current.clone()), current));
            return Layout::from_flat_expr(&result).coalesce();
        };
        // `span` is what is left of the tile's span to place, `stride` the
        // stride of the next rest mode (static until a dynamic extent takes
        // what is left of the span).
        let mut stride = current;
        for c in extents {
            if span == 1 {
                result.push((c.clone(), stride.clone()));
                stride = stride * c;
                continue;
            }
            match (c.as_static(), stride.as_static()) {
                (Some(cv), _) => {
                    let rest = cv.div_ceil(span);
                    result.push((st(rest), stride.clone()));
                    stride = stride * rest;
                    span = span.div_ceil(cv);
                }
                (None, Some(sv)) => {
                    result.push((IndexExpr::ceil_div(c.clone(), st(span)), stride.clone()));
                    // `stride * ceil_div(c, span)`, which is `(stride / span) * c` when `span` divides `c`.
                    stride = if sv.is_multiple_of(span) {
                        st(sv / span) * c
                    } else {
                        stride * IndexExpr::ceil_div(c, st(span))
                    };
                    span = 1;
                }
                (None, None) => unreachable!("the stride is static until a dynamic extent takes the rest of the span"),
            }
        }
        Layout::from_flat_expr(&result).coalesce()
    }

    /// The layout `R` with `self(R(i)) == i` on the largest contiguous range
    /// `[0, size(R))` that `self` covers. Modes with runtime strides are
    /// ignored, as in C++ CuTe, so they can only shorten `R`; a dynamic
    /// extent ends it.
    ///
    /// For a layout with basis strides, which maps coordinates to tuples,
    /// the inverse maps tuples back to indices (cutegen's `right_inverse`):
    /// it has one mode for each leaf of the layout's coprofile (the nesting
    /// of the tuples it maps to), nested the same way, each inverting the
    /// strides `v@m` of that leaf from 1 as an integer layout does, with the
    /// prefix products of the layout's extents as its strides. So
    /// `(4,5):(1@0,1@1)` has the inverse `(4,5):(1,4)`. The extents must be
    /// static, the basis scales integers, and the other strides static 0s;
    /// runtime basis scales are ignored, as runtime strides are.
    #[track_caller]
    pub fn right_inverse(&self) -> Layout {
        if self.has_scaled_basis() {
            return self.basis_inverse(false);
        }
        let flat = self.coalesce().flat_int_modes();
        let mut rstride = Vec::with_capacity(flat.len());
        let mut running = st(1);
        for (s, _) in &flat {
            rstride.push(running.clone());
            running = running * s;
        }
        let mut result = Vec::new();
        let mut next = st(1);
        while let Some(i) = flat.iter().position(|(s, d)| d.is_static() && d.same_value(&next) && !is_trivial(s)) {
            result.push((flat[i].0.clone(), rstride[i].clone()));
            next = flat[i].0.clone() * &flat[i].1;
        }
        Layout::from_flat_expr(&result).coalesce()
    }

    /// The layout `L` with `L(self(i)) == i` for every `i` in the domain.
    /// Needs static strides (extents may be dynamic).
    ///
    /// For a layout with basis strides, `L` maps the tuples it gives back to
    /// indices (cutegen's `left_inverse`): one mode for each leaf of the
    /// coprofile, nested the same way, covering the gaps between the strides
    /// `v@m` of that leaf with modes of stride 0. So `(4,5):(2@0,2@1)` has
    /// the left inverse `((2,4),(2,5)):((0,1),(0,4))`. The extents must be
    /// static, the basis scales static integers that divide each other in
    /// order, and the other strides static 0s.
    #[track_caller]
    pub fn left_inverse(&self) -> Layout {
        if self.has_scaled_basis() {
            return self.basis_inverse(true);
        }
        if let Some((path, d)) = self.stride.first_dynamic() {
            panic!(
                "left_inverse: layout {self} has the runtime stride {d:?} at mode {path:?}; a left inverse needs static strides"
            );
        }
        Layout::from_modes([self.clone(), self.complement(cosize_of(&self.shape, &self.stride))]).right_inverse()
    }

    /// The same function on the domain `shape`: `self ∘ make_layout(shape)`
    /// (CuTe's `with_shape`).
    #[track_caller]
    pub fn with_shape(&self, shape: impl Into<Shape>) -> Layout {
        self.compose(&Layout::col_major(shape))
    }

    /// The layout of the same offsets counted in units `n` times larger
    /// (CuTe's `upcast<N>`, as CUTLASS 3.0 to 3.8 define it): each non-zero
    /// stride is divided by `n` and modes whose steps fall within one unit
    /// shrink, both rounding away from zero as CuTe's `shape_div` does.
    /// (Later releases round up with `ceil_div`, so they also take a shape
    /// that the step does not divide.) Static layouts only.
    #[track_caller]
    pub fn upcast(&self, n: usize) -> Layout {
        self.require_static("upcast", "upcast needs a static layout");
        self.require_int_strides("upcast", "recasting counts offsets, which a basis layout does not have");
        let stride = self.static_stride();
        let mut strides = stride.flat_leaves().into_iter();
        let mut new_strides = Vec::new();
        let shape = self.static_shape().map(&mut |&s| {
            let d = strides.next().expect("congruent");
            if d == 0 {
                new_strides.push(0);
                s
            } else {
                new_strides.push(super::tree::shape_div(d, n));
                super::tree::shape_div(s, super::tree::shape_div(n, d))
            }
        });
        Layout::new(shape.clone(), Tree::unflatten(&new_strides, &shape))
    }

    /// Append `mode` as a new last mode (a leaf layout becomes a pair).
    pub fn append(&self, mode: Layout) -> Layout {
        let mut modes = self.as_tuple_modes();
        modes.push(mode);
        Layout::from_modes(modes)
    }

    /// Prepend `mode` as a new first mode.
    pub fn prepend(&self, mode: Layout) -> Layout {
        let mut modes = vec![mode];
        modes.extend(self.as_tuple_modes());
        Layout::from_modes(modes)
    }

    fn as_tuple_modes(&self) -> Vec<Layout> {
        if self.shape.is_leaf() { vec![self.clone()] } else { self.modes() }
    }

    /// Pad with `1:0` modes up to rank `r`.
    pub fn pad_to_rank(&self, r: usize) -> Layout {
        let mut modes = self.as_tuple_modes();
        while modes.len() < r {
            modes.push(Layout::new(1, 0));
        }
        Layout::from_modes(modes)
    }

    /// Group modes `[begin, end)` into a single mode.
    #[track_caller]
    pub fn group(&self, begin: usize, end: usize) -> Layout {
        let modes = self.as_tuple_modes();
        assert!(begin <= end && end <= modes.len(), "group({begin}, {end}) on a rank-{} layout", modes.len());
        let mut out: Vec<Layout> = modes[..begin].to_vec();
        out.push(Layout::from_modes(modes[begin..end].to_vec()));
        out.extend(modes[end..].iter().cloned());
        Layout::from_modes(out)
    }

    /// Select (and reorder) top-level modes.
    pub fn select(&self, modes: &[usize]) -> Layout {
        Layout::from_modes(modes.iter().map(|&i| self.mode(i)))
    }

    /// The modes [`complement`](Layout::complement) orders: flat, filtered,
    /// and without static size-1 or stride-0 modes.
    fn complement_modes(&self) -> Vec<(IndexExpr, IndexExpr)> {
        self.filter().flat_int_modes().into_iter().filter(|(s, d)| !is_trivial(s) && !is_zero(d)).collect()
    }

    /// Divide by `tiler`: each tiled part becomes `(tile, rest)`. A dynamic
    /// extent gives a rest mode of extent `ceil_div(extent, tile)`; a tiler
    /// of the whole of a layout with dynamic extents splits the rest over the
    /// layout's modes, as C++ CuTe does. A tiler may have runtime extents,
    /// and runtime strides if it has a single mode (after filtering), as in
    /// CuTe.
    #[track_caller]
    pub fn logical_divide(&self, tiler: &Tiler) -> Layout {
        self.logical_divide_in(tiler, &ModePath::whole(self))
    }

    #[track_caller]
    fn logical_divide_in(&self, tiler: &Tiler, at: &ModePath) -> Layout {
        match tiler {
            Tree::Leaf(None) => self.clone(),
            Tree::Leaf(Some(b)) if b.has_scaled_basis() => {
                // A tiler of basis strides picks the modes of `self` it tiles
                // (cutegen's divide: composition with the tiler and its
                // complement extended over `self`'s shape).
                let rest = b.complement_extend(&self.shape);
                self.compose_in(&Layout::from_modes([b.clone(), rest]), at)
            }
            Tree::Leaf(Some(b)) => {
                let modes = b.complement_modes();
                if let Some(d) = unordered_runtime_stride(&modes) {
                    panic!(
                        "logical_divide: the tiler {b} of {} has the runtime stride {d:?} and {} modes after \
                         filtering; CuTe divides by a tiler with a runtime stride only when it has a single mode",
                        at.describe(self),
                        modes.len()
                    );
                }
                // C++ CuTe takes the complement in `shape(coalesce(layout))`;
                // for a static shape that is the same as in its size.
                let rest = if self.shape.is_static() {
                    b.complement(self.shape.product())
                } else {
                    b.complement_to_shape(self.coalesce().shape())
                };
                self.compose_in(&Layout::from_modes([b.clone(), rest]), at)
            }
            Tree::Branch(ts) => self.by_mode(ts, ByMode::Divide, at),
        }
    }

    /// `((tiles...), (rests...))`.
    #[track_caller]
    pub fn zipped_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        Layout::from_modes([tile, rest])
    }

    /// `((tiles...), rest0, rest1, ...)`.
    #[track_caller]
    pub fn tiled_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        let mut modes = vec![tile];
        modes.extend(rest.as_tuple_modes());
        Layout::from_modes(modes)
    }

    /// `(tile0, tile1, ..., rest0, rest1, ...)`.
    #[track_caller]
    pub fn flat_divide(&self, tiler: &Tiler) -> Layout {
        let (tile, rest) = zip2_by(&self.logical_divide(tiler), tiler);
        let mut modes = tile.as_tuple_modes();
        modes.extend(rest.as_tuple_modes());
        Layout::from_modes(modes)
    }
}

/// One empty bucket for each leaf of `coprofile`, keyed by its path.
type Buckets = std::collections::BTreeMap<Vec<usize>, Vec<(IndexExpr, IndexExpr, IndexExpr)>>;

fn coprofile_buckets(coprofile: &Coord) -> Buckets {
    let mut buckets = Buckets::new();
    coprofile.find_if(|_, path| {
        // A whole coprofile that is a leaf has the empty path.
        let path = if coprofile.is_leaf() { Vec::new() } else { path.to_vec() };
        buckets.insert(path, Vec::new());
        false
    });
    buckets
}

/// A layout nested like `coprofile`, with `f(path)` at each leaf (the
/// whole layout when the coprofile is a leaf).
fn by_coprofile(coprofile: &Coord, f: &mut dyn FnMut(&[usize]) -> Layout) -> Layout {
    fn walk(t: &Coord, path: &mut Vec<usize>, f: &mut dyn FnMut(&[usize]) -> Layout) -> Layout {
        match t {
            Tree::Leaf(_) => f(path),
            Tree::Branch(children) => Layout::from_modes(children.iter().enumerate().map(|(i, c)| {
                path.push(i);
                let l = walk(c, path, f);
                path.pop();
                l
            })),
        }
    }
    walk(coprofile, &mut Vec::new(), f)
}

/// `layout`'s non-zero integer strides as basis strides of the mode at
/// `path` (unchanged for the empty path).
fn with_basis(layout: Layout, path: &[usize]) -> Layout {
    if path.is_empty() {
        return layout;
    }
    let stride = layout.stride.map(&mut |d| match d {
        StrideLeaf::Int(v) if !is_zero(v) => {
            StrideLeaf::Basis(super::basis::ScaledBasis::new(v.clone(), path.to_vec()))
        }
        d => d.clone(),
    });
    Layout { shape: layout.shape, stride }
}

/// cutegen's `coalesce_z` of flat modes: a mode of extent 1 is dropped when
/// another follows it, two static modes that chain merge, and the last mode
/// stays even with extent 1, so `(3,1):(1,15)` keeps its 1.
fn coalesce_z(shapes: &[IndexExpr], strides: &[IndexExpr]) -> Layout {
    let mut out: Vec<(IndexExpr, IndexExpr)> = Vec::new();
    for (s, d) in shapes.iter().zip(strides) {
        while out.last().is_some_and(|(ls, _)| is_one(ls)) {
            out.pop();
        }
        if let Some((ls, ld)) = out.last_mut()
            && let (Some(a), Some(b), Some(c), Some(e)) = (ls.as_static(), ld.as_static(), s.as_static(), d.as_static())
            && e == a * b
        {
            *ls = st(a * c);
            continue;
        }
        out.push((s.clone(), d.clone()));
    }
    Layout::from_flat_expr(&out)
}

/// The rest of [`Layout::complement_extend`] at the mode of `extend` the
/// path names: `at` holds the coprofile there and the complement's part of
/// it, or is `None` where the tiler walks no mode.
#[track_caller]
fn extend_rest(at: Option<(&Coord, &Layout)>, extend: &Shape, path: &mut Vec<usize>, tiler: &Layout) -> Layout {
    let basis = |path: &[usize]| StrideLeaf::Basis(super::basis::ScaledBasis::new(st(1), path.to_vec()));
    match at {
        None => {
            // The whole mode, with unit basis strides at its paths.
            let mut leaves = Vec::new();
            extend.find_if(|_, sub| {
                let mut p = path.clone();
                if !extend.is_leaf() {
                    p.extend_from_slice(sub);
                }
                leaves.push(basis(&p));
                false
            });
            Layout::new(extend.clone(), Tree::unflatten(&leaves, extend))
        }
        Some((Tree::Leaf(_), part)) => {
            let mut modes = part.flat_modes_expr();
            let (_, last) = modes.pop().expect("a complement has a last mode");
            let mut size = match &last {
                StrideLeaf::Int(v) => v.clone(),
                StrideLeaf::Basis(e) => e.value().as_int().expect("the complement's scales are integers").clone(),
            };
            let mut run = st(1);
            for s in extend.flat_leaves() {
                let rest = IndexExpr::ceil_div(s.clone(), size.clone());
                size = IndexExpr::ceil_div(size, s);
                modes.push((rest.clone(), last.scale(&run, "logical_divide")));
                run = run * rest;
            }
            Layout::from_flat_expr(&modes).coalesce()
        }
        Some((Tree::Branch(cs), part)) => {
            let Tree::Branch(es) = extend else {
                panic!(
                    "logical_divide: the tiler {tiler} walks modes {path:?} of the tuples it maps to, but the layout it \
                     divides has the extent {extend} there"
                )
            };
            Layout::from_modes(es.iter().enumerate().map(|(i, e)| {
                path.push(i);
                let sub = cs.get(i).map(|c| (c, part.mode(i)));
                let l = extend_rest(sub.as_ref().map(|(c, p)| (*c, p)), e, path, tiler);
                path.pop();
                l
            }))
        }
    }
}

/// The operation a by-mode tiler applies to each mode.
#[derive(Clone, Copy)]
enum ByMode {
    Compose,
    Divide,
}

/// Where an operation applies, for its messages: the layout the user passed
/// and the path of the mode a by-mode tiler reached (empty for the whole).
#[derive(Clone)]
struct ModePath<'a> {
    root: &'a Layout,
    path: Vec<usize>,
}

impl<'a> ModePath<'a> {
    fn whole(root: &'a Layout) -> Self {
        ModePath { root, path: Vec::new() }
    }

    fn child(&self, i: usize) -> Self {
        let mut path = self.path.clone();
        path.push(i);
        ModePath { root: self.root, path }
    }

    /// `layout`, the layout at this path, as a message names it.
    fn describe(&self, layout: &Layout) -> String {
        if self.path.is_empty() {
            format!("layout {layout}")
        } else {
            format!("mode {:?} ({layout}) of layout {}", self.path, self.root)
        }
    }

    /// Where a composition of the mode at this path is, for its message:
    /// nothing for the whole layout.
    fn suffix(&self) -> String {
        if self.path.is_empty() { String::new() } else { format!(" (mode {:?} of layout {})", self.path, self.root) }
    }
}

/// A runtime stride among several modes, which the complement cannot place
/// among the others: CuTe takes the complement of a layout with a runtime
/// stride only when it has a single mode.
fn unordered_runtime_stride(modes: &[(IndexExpr, IndexExpr)]) -> Option<&IndexExpr> {
    if modes.len() < 2 {
        return None;
    }
    modes.iter().map(|(_, d)| d).find(|d| !d.is_static())
}

/// Split each tiled mode of a divided layout into its tile and rest parts,
/// gathering tiles into the first result and rests into the second
/// (CuTe's `zip2_by`).
fn zip2_by(layout: &Layout, guide: &Tiler) -> (Layout, Layout) {
    match guide {
        Tree::Leaf(_) => {
            assert_eq!(layout.rank(), 2, "zip2_by: expected a (tile, rest) pair, got {layout}");
            (layout.mode(0), layout.mode(1))
        }
        Tree::Branch(gs) => {
            let mut tiles = Vec::new();
            let mut rests = Vec::new();
            for (i, g) in gs.iter().enumerate() {
                let (t, r) = zip2_by(&layout.mode(i), g);
                tiles.push(t);
                rests.push(r);
            }
            for i in gs.len()..layout.rank() {
                rests.push(layout.mode(i));
            }
            (Layout::from_modes(tiles), Layout::from_modes(rests))
        }
    }
}

/// The first mode of `coord` that does not fit `shape` as a coordinate:
/// its path in `coord`, the sub-coordinate there and the sub-shape it meets,
/// or `None` if the whole coordinate fits. A leaf is a 1-D index into a mode
/// of any shape, and a tuple of one coordinate stands for that coordinate;
/// otherwise a tuple must meet a tuple of the same rank (C++ CuTe's
/// `crd2idx` and `slice_and_offset`).
fn coord_mismatch<'a, T>(coord: &'a Tree<T>, shape: &'a Shape) -> Option<(Vec<usize>, &'a Tree<T>, &'a Shape)> {
    let within = |i: usize, found: Option<(Vec<usize>, &'a Tree<T>, &'a Shape)>| {
        found.map(|(mut path, c, s)| {
            path.insert(0, i);
            (path, c, s)
        })
    };
    match (coord, shape) {
        (Tree::Leaf(_), _) => None,
        (Tree::Branch(cs), Tree::Leaf(_)) if cs.len() == 1 => within(0, coord_mismatch(&cs[0], shape)),
        (Tree::Branch(cs), Tree::Branch(ss)) if cs.len() == ss.len() => {
            cs.iter().zip(ss).enumerate().find_map(|(i, (c, s))| within(i, coord_mismatch(c, s)))
        }
        _ => Some((Vec::new(), coord, shape)),
    }
}

/// Panic, naming `what` and the mode, unless `coord` is a coordinate of
/// `layout` (see [`coord_mismatch`]). `view` prints a coordinate.
#[track_caller]
fn check_coord<T>(what: &str, coord: &Tree<T>, layout: &Layout, view: impl Fn(&Tree<T>) -> String) {
    let Some((path, sub, sub_shape)) = coord_mismatch(coord, &layout.shape) else { return };
    if path.is_empty() {
        panic!(
            "{what}: coordinate {} has rank {} but layout {layout} has rank {}",
            view(coord),
            coord.rank(),
            layout.rank()
        );
    }
    let meets = match sub_shape {
        Tree::Leaf(e) => format!("the extent {e}"),
        _ => format!("{sub_shape}, of rank {}", sub_shape.rank()),
    };
    panic!(
        "{what}: coordinate {} does not fit layout {layout}: its mode {path:?}, {}, has rank {} where the shape \
         has {meets}",
        view(coord),
        view(sub),
        sub.rank()
    );
}

/// A slice coordinate as CuTe prints it: `_` for a kept mode, `?` for a
/// runtime leaf.
fn slice_view(coord: &SliceCoord) -> String {
    coord.map(&mut |c| c.as_ref().map_or_else(|| "_".to_string(), |e| e.to_string())).to_string()
}

/// The modes `coord` keeps, which [`check_coord`] has checked against
/// `shape` (CuTe's `slice`, with `detail::lift_slice`).
fn collect_kept(coord: &SliceCoord, shape: &Shape, stride: &Stride, kept: &mut Vec<Layout>) {
    match coord {
        Tree::Leaf(None) => kept.push(Layout { shape: shape.clone(), stride: stride.clone() }),
        Tree::Leaf(Some(_)) => {}
        Tree::Branch(cs) if cs.len() == 1 && shape.is_leaf() => collect_kept(&cs[0], shape, stride, kept),
        Tree::Branch(cs) => {
            debug_assert!(!shape.is_leaf() && cs.len() == shape.rank(), "the slice coordinate was checked");
            for (i, c) in cs.iter().enumerate() {
                collect_kept(c, shape.mode(i), stride.mode(i), kept);
            }
        }
    }
}

/// The offset of `coord`, which [`check_coord`] has checked against `shape`.
fn offset_impl(coord: &Coord, shape: &Shape, stride: &Stride) -> IndexExpr {
    match (coord, shape, stride) {
        (Tree::Leaf(c), Tree::Leaf(_), Tree::Leaf(d)) => IndexExpr::product(c.clone(), d.int().clone()),
        (Tree::Leaf(c), Tree::Branch(ss), Tree::Branch(ds)) => {
            if is_zero(c) {
                return st(0);
            }
            let mut rest = c.clone();
            let mut acc = st(0);
            for (i, (s, d)) in ss.iter().zip(ds).enumerate() {
                let part = if i + 1 == ss.len() {
                    rest.clone()
                } else {
                    let size = s.product();
                    let part = rest.clone() % size.clone();
                    rest = rest / size;
                    part
                };
                acc = acc + offset_impl(&Tree::Leaf(part), s, d);
            }
            acc
        }
        (Tree::Branch(cs), Tree::Leaf(_), Tree::Leaf(_)) if cs.len() == 1 => offset_impl(&cs[0], shape, stride),
        (Tree::Branch(cs), Tree::Branch(ss), Tree::Branch(ds)) => {
            debug_assert_eq!(cs.len(), ss.len(), "the coordinate was checked");
            cs.iter().zip(ss).zip(ds).fold(st(0), |acc, ((c, s), d)| acc + offset_impl(c, s, d))
        }
        _ => unreachable!("coordinate {coord:?} was checked against shape {shape}"),
    }
}

/// Add the arithmetic tuple of `coord`, which [`check_coord`] has checked
/// against `shape`, to `acc`: [`offset_impl`] for basis strides. Unlike the
/// offset, a coordinate of 0 still walks its modes, so that each basis
/// stride gives its mode an entry.
fn eval_impl(coord: &Coord, shape: &Shape, stride: &Stride, acc: &mut Coord) -> Result<(), String> {
    match (coord, shape, stride) {
        (Tree::Leaf(c), Tree::Leaf(_), Tree::Leaf(d)) => match d {
            StrideLeaf::Int(v) => {
                let term = IndexExpr::product(c.clone(), v.clone());
                if is_zero(&term) {
                    return Ok(());
                }
                if !acc.is_leaf() {
                    return Err(format!("the integer stride {v:?} has no mode of the tuple to add to"));
                }
                tuple_add(acc, &[], term).map_err(|why| format!("stride {d:?}: {why}"))
            }
            StrideLeaf::Basis(e) => {
                let term = scale_coord(c, e.value()).map_err(|why| format!("stride {e:?}: {why}"))?;
                tuple_add(acc, e.mode(), term).map_err(|why| format!("stride {e:?}: {why}"))
            }
        },
        (Tree::Leaf(c), Tree::Branch(ss), Tree::Branch(ds)) => {
            let mut rest = c.clone();
            for (i, (s, d)) in ss.iter().zip(ds).enumerate() {
                let part = if i + 1 == ss.len() {
                    rest.clone()
                } else {
                    let size = s.product();
                    let part = rest.clone() % size.clone();
                    rest = rest / size;
                    part
                };
                eval_impl(&Tree::Leaf(part), s, d, acc)?;
            }
            Ok(())
        }
        (Tree::Branch(cs), Tree::Leaf(_), Tree::Leaf(_)) if cs.len() == 1 => eval_impl(&cs[0], shape, stride, acc),
        (Tree::Branch(cs), Tree::Branch(ss), Tree::Branch(ds)) => {
            debug_assert_eq!(cs.len(), ss.len(), "the coordinate was checked");
            cs.iter().zip(ss).zip(ds).try_for_each(|((c, s), d)| eval_impl(c, s, d, acc))
        }
        _ => unreachable!("coordinate {coord:?} was checked against shape {shape}"),
    }
}

/// Split a 1-D index into a coordinate nested like `shape` (colexicographic,
/// first mode fastest; CuTe's `idx2crd`). The index must be smaller than the
/// shape's size, except that the last mode takes whatever is left. The
/// index and the extents may be runtime values: `idx2crd(9, (m, 8))` is
/// `(9 % m, 9 / m)`. `shape` may be an [`IntTuple`] or a [`Shape`].
pub fn idx2crd<L: Clone + Into<IndexExpr>>(idx: impl Into<IndexExpr>, shape: &Tree<L>) -> Coord {
    let shape: Shape = shape.map(&mut |l| l.clone().into());
    idx2crd_impl(&idx.into(), &shape)
}

fn idx2crd_impl(idx: &IndexExpr, shape: &Shape) -> Coord {
    match shape {
        Tree::Leaf(_) => Tree::Leaf(idx.clone()),
        Tree::Branch(children) => {
            let mut rest = idx.clone();
            let mut out = Vec::with_capacity(children.len());
            for (i, child) in children.iter().enumerate() {
                if i + 1 == children.len() {
                    out.push(idx2crd_impl(&rest, child));
                } else {
                    let size = child.product();
                    out.push(idx2crd_impl(&(rest.clone() % size.clone()), child));
                    rest = rest / size;
                }
            }
            Tree::Branch(out)
        }
    }
}

/// The offset `layout` maps `coord` to (CuTe DSL's `crd2idx(coord,
/// layout)`), as an [`IndexExpr`]: static when the coordinate and the
/// layout's leaves it touches are. A leaf coordinate is a 1-D index.
/// `crd2idx((1, 2), &Layout::col_major((m, 8)))` is `1 + 2 * m`. Panics,
/// naming the mode, if the coordinate does not fit the layout's shape (see
/// [`Layout::offset`]). A composed layout gives `inner(offset +
/// outer(coord))`.
#[track_caller]
pub fn crd2idx(coord: impl Into<Coord>, layout: &(impl Crd2Idx + ?Sized)) -> IndexExpr {
    layout.crd2idx_of(&coord.into())
}

/// Layouts [`crd2idx`] evaluates: [`Layout`] and
/// [`ComposedLayout`](super::ComposedLayout).
pub trait Crd2Idx {
    #[track_caller]
    fn crd2idx_of(&self, coord: &Coord) -> IndexExpr;
}

impl Crd2Idx for Layout {
    #[track_caller]
    fn crd2idx_of(&self, coord: &Coord) -> IndexExpr {
        self.offset_checked(coord, "crd2idx")
    }
}

impl Crd2Idx for super::ComposedLayout {
    #[track_caller]
    fn crd2idx_of(&self, coord: &Coord) -> IndexExpr {
        self.offset_of(coord)
    }
}

#[track_caller]
fn compose_impl(
    lhs: &[(IndexExpr, StrideLeaf)],
    shape: &Shape,
    stride: &Stride,
    a: &Layout,
    b: &Layout,
    at: &ModePath,
) -> Layout {
    match (shape, stride) {
        (Tree::Branch(ss), Tree::Branch(ds)) => {
            // A loop rather than a closure, so that a refusal reports the
            // caller's location.
            let mut modes = Vec::with_capacity(ss.len());
            for (s, d) in ss.iter().zip(ds) {
                modes.push(compose_impl(lhs, s, d, a, b, at));
            }
            Layout::from_modes(modes)
        }
        (Tree::Leaf(s), Tree::Leaf(StrideLeaf::Int(d))) => compose_mode(lhs, s, d, a, b, at),
        (Tree::Leaf(s), Tree::Leaf(StrideLeaf::Basis(e))) => {
            // A basis stride `v@m` on the right-hand side picks mode `m` of
            // the left-hand side and composes it with `s:v` (C++ CuTe's and
            // cutegen's `composition_impl`). Mode 0 of a leaf is the leaf, as
            // `get` has it.
            let mut sub = a.clone();
            for (depth, &i) in e.mode().iter().enumerate() {
                if i >= sub.rank() {
                    let what =
                        if sub.shape.is_leaf() { "a leaf".to_string() } else { format!("of rank {}", sub.rank()) };
                    panic!(
                        "composition {a} ∘ {b}{} is not defined: the basis stride {e} names mode {:?}, but mode {:?} \
                         of the left-hand side is {what}",
                        at.suffix(),
                        e.mode(),
                        &e.mode()[..depth]
                    );
                }
                sub = sub.mode(i);
            }
            let Some(v) = e.value().as_int() else {
                panic!("composition {a} ∘ {b}{} is not defined: the basis stride {e} has a ratio scale", at.suffix())
            };
            sub.compose_in(&Layout { shape: Tree::Leaf(s.clone()), stride: Tree::Leaf(StrideLeaf::Int(v.clone())) }, at)
        }
        _ => unreachable!("layout shape and stride are congruent"),
    }
}

/// A mode of the flattened left-hand side as composition walks it.
struct LhsMode {
    /// The extent, or `None` for the last mode, which is unbounded.
    extent: Option<IndexExpr>,
    stride: StrideLeaf,
    /// Whether the extent is a dynamic extent of the left-hand side, or one
    /// divided by a static stride, and so is taken to be large and divisible.
    large: bool,
    /// For a mode that the stride lands inside, the left-hand extent and the
    /// stride that split it, which a refusal names.
    split: Option<(IndexExpr, IndexExpr)>,
}

/// How a step of composition divides a stride by a mode's extent.
enum StrideStep {
    /// The stride steps over the whole mode: go on with this stride.
    Skip(IndexExpr),
    /// The stride lands inside the mode: this many of its steps remain
    /// (`large` as in [`LhsMode`]).
    Split { extent: IndexExpr, large: bool },
    /// Which of the two it is is only known at run time.
    AtRunTime,
}

/// How a step of composition takes extent from a mode.
enum ShapeStep {
    /// The mode holds the rest of the extent.
    Fits,
    /// The mode is used whole: go on with this extent.
    Spans(IndexExpr),
    /// Which of the two it is is only known at run time.
    AtRunTime,
}

/// `rest_stride` against the extent `si` of a left-hand mode: `rest_stride
/// % si == 0` skips the mode, `si % rest_stride == 0` splits it. A dynamic
/// extent of the left-hand side is taken to be a large multiple of a static
/// stride. A runtime stride, which comes from the right-hand side, skips a
/// static extent only if it is provably a multiple of it; it and two
/// runtime values not provably multiples of one another are compared at run
/// time. `None` when both are static and neither divides the other. C++
/// CuTe's weak divisibility condition also lets a smaller stride split a
/// mode it does not divide, but the layout it builds from that split is
/// often not the composition, so cute-rs refuses it.
fn stride_step(rest_stride: &IndexExpr, si: &IndexExpr) -> Option<StrideStep> {
    match (rest_stride.as_static(), si.as_static()) {
        (Some(r), Some(s)) if r.is_multiple_of(s) => Some(StrideStep::Skip(st(r / s))),
        (Some(r), Some(s)) if s.is_multiple_of(r) => Some(StrideStep::Split { extent: st(s / r), large: false }),
        (Some(_), Some(_)) => None,
        (Some(_), None) => {
            Some(StrideStep::Split { extent: IndexExpr::ceil_div(si.clone(), rest_stride.clone()), large: true })
        }
        (None, Some(s)) if rest_stride.is_multiple_of(s) => Some(StrideStep::Skip(rest_stride.clone() / s)),
        (None, Some(_)) => Some(StrideStep::AtRunTime),
        (None, None) => Some(if let Some(q) = rest_stride.exact_quotient(si) {
            StrideStep::Skip(q)
        } else if let Some(q) = si.exact_quotient(rest_stride) {
            StrideStep::Split { extent: q, large: false }
        } else {
            StrideStep::AtRunTime
        }),
    }
}

/// The extent `t` of a mode against the extent `rest` still to place: a
/// mode whose extent is a multiple of the rest, or at least as large, holds
/// it (C++ CuTe keeps `min(t, rest)`, which is then `rest`), and `rest % t
/// == 0` uses the mode whole. A large `t` (see [`LhsMode`]) holds any static
/// `rest`. Otherwise a runtime value fits or spans only where that is
/// provable, and the rest is compared at run time. `None` when both are
/// static and `t` is smaller than `rest` without dividing it, which C++ CuTe
/// refuses too.
fn shape_step(t: &IndexExpr, large: bool, rest: &IndexExpr) -> Option<ShapeStep> {
    match (t.as_static(), rest.as_static()) {
        (Some(tv), Some(r)) if tv.is_multiple_of(r) => Some(ShapeStep::Fits),
        (Some(tv), Some(r)) if r.is_multiple_of(tv) => Some(ShapeStep::Spans(st(r / tv))),
        (Some(tv), Some(r)) if tv >= r => Some(ShapeStep::Fits),
        (Some(_), Some(_)) => None,
        (None, Some(r)) if large || t.is_multiple_of(r) => Some(ShapeStep::Fits),
        (Some(tv), None) if rest.is_multiple_of(tv) => Some(ShapeStep::Spans(rest.clone() / tv)),
        (None, None) if t.exact_quotient(rest).is_some() => Some(ShapeStep::Fits),
        (None, None) => Some(rest.exact_quotient(t).map_or(ShapeStep::AtRunTime, ShapeStep::Spans)),
        (None, Some(_)) | (Some(_), None) => Some(ShapeStep::AtRunTime),
    }
}

/// Compose a flat, coalesced layout with the single mode `s:d` (C++ CuTe's
/// `composition_impl`). Where the steps cannot be decided when the kernel
/// is built, the mode gets C++ CuTe's runtime extents: a stride step splits
/// off `ceil_div(si, rest_stride)` and goes on with `ceil_div(rest_stride,
/// si)`, and a shape step keeps `min(t, rest)` and goes on with
/// `ceil_div(rest, t)` (C++ CuTe divides by the minimum, which is the same
/// for values that satisfy the divisibility conditions).
#[track_caller]
fn compose_mode(
    lhs: &[(IndexExpr, StrideLeaf)],
    s: &IndexExpr,
    d: &IndexExpr,
    a: &Layout,
    b: &Layout,
    at: &ModePath,
) -> Layout {
    if is_zero(d) || is_one(s) {
        return Layout { shape: Tree::Leaf(s.clone()), stride: Tree::Leaf(StrideLeaf::Int(st(0))) };
    }
    let scale = |di: &StrideLeaf, r: &IndexExpr| di.scale(r, "composition");
    if let [(_, d0)] = lhs {
        return Layout::from_flat_expr(&[(s.clone(), scale(d0, d))]);
    }

    // Divide out the stride: skip the first `d` elements' worth of modes.
    let mut rest_stride = d.clone();
    let mut modes: Vec<LhsMode> = Vec::new();
    let last = lhs.len() - 1;
    for (i, (si, di)) in lhs.iter().enumerate() {
        if i == last {
            modes.push(LhsMode { extent: None, stride: scale(di, &rest_stride), large: false, split: None });
            break;
        }
        if is_one(&rest_stride) {
            let large = !si.is_static();
            modes.push(LhsMode { extent: Some(si.clone()), stride: di.clone(), large, split: None });
            continue;
        }
        let Some(step) = stride_step(&rest_stride, si) else {
            let stride = with_rest("stride", d, &rest_stride);
            panic!(
                "composition {a} ∘ {b}{} is not defined: {stride} is not compatible with mode size {si}",
                at.suffix()
            )
        };
        match step {
            StrideStep::Skip(next) => rest_stride = next,
            StrideStep::Split { extent, large } => {
                let split = Some((si.clone(), rest_stride.clone()));
                modes.push(LhsMode { extent: Some(extent), stride: scale(di, &rest_stride), large, split });
                rest_stride = st(1);
            }
            StrideStep::AtRunTime => {
                let extent = IndexExpr::ceil_div(si.clone(), rest_stride.clone());
                let split = Some((si.clone(), rest_stride.clone()));
                modes.push(LhsMode { extent: Some(extent), stride: scale(di, &rest_stride), large: false, split });
                rest_stride = IndexExpr::ceil_div(rest_stride, si.clone());
            }
        }
    }

    // Mod out the shape: keep modes until `s` elements are covered.
    let mut rest_shape = s.clone();
    let mut result = Vec::new();
    for mode in modes {
        if is_one(&rest_shape) {
            break;
        }
        let Some(t) = mode.extent else {
            result.push((rest_shape, mode.stride));
            break;
        };
        let Some(step) = shape_step(&t, mode.large, &rest_shape) else {
            let shape = with_rest("shape", s, &rest_shape);
            let held = match &mode.split {
                Some((si, r)) => format!("mode size {si} at stride {r}, which has {t} elements"),
                None => format!("mode size {t}"),
            };
            panic!("composition {a} ∘ {b}{} is not defined: {shape} is not compatible with {held}", at.suffix())
        };
        match step {
            ShapeStep::Fits => {
                result.push((rest_shape, mode.stride));
                rest_shape = st(1);
            }
            ShapeStep::Spans(next) => {
                result.push((t, mode.stride));
                rest_shape = next;
            }
            ShapeStep::AtRunTime => {
                result.push((IndexExpr::min(t.clone(), rest_shape.clone()), mode.stride));
                rest_shape = IndexExpr::ceil_div(rest_shape, t);
            }
        }
    }
    let result: Vec<_> = result.into_iter().filter(|(s, _)| !is_one(s)).collect();
    Layout::from_flat_expr(&result)
}

/// The right-hand side's `value` as a refusal names it (`stride 4`), with
/// what is left of it after the modes before (`stride 4 (2 after the modes
/// before it)`) when that is different.
fn with_rest(what: &str, value: &IndexExpr, rest: &IndexExpr) -> String {
    if rest == value { format!("{what} {value}") } else { format!("{what} {value} ({rest} after the modes before it)") }
}

/// One past the largest offset of `shape:stride`: `1 + sum((s - 1) * d)`,
/// or 0 for a statically empty shape. Dynamic extents are assumed to be at
/// least 1.
#[track_caller]
fn cosize_of(shape: &Shape, stride: &Stride) -> IndexExpr {
    let shapes = shape.flat_leaves();
    if shapes.iter().any(is_zero) {
        return st(0);
    }
    if stride.has_scaled_basis() {
        return coshape_of(shape, stride).product();
    }
    shapes.into_iter().zip(stride.leaves()).fold(st(1), |acc, (s, d)| acc + (s - st(1)) * d.int().clone())
}

/// The coshape of a layout with basis strides: one past its largest value in
/// each mode of the tuples it maps to (`l(size(l) - 1) + 1`, leaf by leaf).
/// Dynamic extents are assumed to be at least 1.
#[track_caller]
fn coshape_of(shape: &Shape, stride: &Stride) -> Coord {
    let mut acc = Tree::Leaf(st(0));
    for (s, d) in shape.flat_leaves().into_iter().zip(stride.flat_leaves()) {
        let last = s - st(1);
        let result = match &d {
            StrideLeaf::Int(v) if is_zero(v) => Ok(()),
            StrideLeaf::Int(v) => Err(format!("the integer stride {v:?} has no mode of the tuple to add to")),
            StrideLeaf::Basis(e) => scale_coord(&last, e.value()).and_then(|term| tuple_add(&mut acc, e.mode(), term)),
        };
        if let Err(why) = result {
            panic!("cosize: layout {shape}:{stride} has no coshape: stride {d:?}: {why}");
        }
    }
    acc.map(&mut |e| e.clone() + st(1))
}

/// The sub-tree at `mode`, as CuTe DSL's `size` and `cosize` take a mode
/// path: each index selects a mode of the tuple reached so far, the whole
/// of a leaf shape is its mode `[0]`, and a path cannot go on past a leaf.
#[track_caller]
fn select_mode<'a, T: StaticLeaf>(t: &'a Tree<T>, mode: &[usize], what: &str) -> &'a Tree<T> {
    let mut current = t;
    for (depth, &i) in mode.iter().enumerate() {
        current = match current {
            Tree::Branch(children) if i < children.len() => &children[i],
            Tree::Leaf(_) if depth == 0 && i == 0 => current,
            _ => {
                let at = if depth == 0 {
                    format!("{}", static_view(t))
                } else {
                    format!("mode {:?} of {}", &mode[..depth], static_view(t))
                };
                let why = match current {
                    Tree::Leaf(_) if depth > 0 => "is a leaf, so the path cannot go on".to_string(),
                    _ => format!("has rank {}, so it has no mode {i}", current.rank()),
                };
                panic!("{what}: mode path {mode:?} does not fit: {at} {why}")
            }
        };
    }
    current
}

/// The domain size of `x` at the mode path `mode` (CuTe DSL's `cute.size(x,
/// mode=...)`): the product of the extents there, an [`IndexExpr`] that is
/// static when they are. `x` is a tuple ([`IntTuple`] or [`Shape`]), a
/// layout, a swizzled layout or a tensor. An empty path is the whole of
/// `x`, `[1]` its mode 1 and `[1, 0]` mode 0 of that; a path that leaves the
/// tuple, or goes on past a leaf, panics. A runtime extent makes it a
/// runtime value, which a [`for_range`](crate::for_range) takes as its trip
/// count: `size(&tiles, &[2])` is the number of tiles of a rest mode. For an
/// unrolled Rust loop, use the `usize` [`Layout::size`] or
/// [`Tensor::size`](crate::Tensor::size) instead.
#[track_caller]
pub fn size<X>(x: &X, mode: &[usize]) -> IndexExpr
where
    X: HasShape + ?Sized,
    X::Leaf: StaticLeaf + Clone + Into<IndexExpr>,
{
    select_mode(x.shape(), mode, "size").leaves().into_iter().fold(st(1), |acc, e| acc * e.clone().into())
}

/// The shape of `x` at the mode path `mode` (CuTe DSL's `cute.shape(x,
/// mode=...)`): the sub-tuple of extents there, with the leaves of `x`'s
/// shape, so a [`Shape`] for a layout, a swizzled layout or a tensor (static
/// integers and runtime values) and an [`IntTuple`] for an `IntTuple`. The
/// path works as in [`size`]; the empty path gives the whole shape.
/// `shape(&layout, &[1])` is `layout.shape().mode(1)`, checked.
#[track_caller]
pub fn shape<X>(x: &X, mode: &[usize]) -> Tree<X::Leaf>
where
    X: HasShape + ?Sized,
    X::Leaf: StaticLeaf + Clone,
{
    select_mode(x.shape(), mode, "shape").clone()
}

/// The rank of `x` at the mode path `mode` (CuTe DSL's `cute.rank(x,
/// mode=...)`): the number of modes of the sub-tuple there, 1 for a leaf.
/// `x` and the path are as in [`size`]. The rank is a structural property,
/// so it is a `usize` even where the extents are runtime values.
#[track_caller]
pub fn rank<X>(x: &X, mode: &[usize]) -> usize
where
    X: HasShape + ?Sized,
    X::Leaf: StaticLeaf,
{
    select_mode(x.shape(), mode, "rank").rank()
}

/// The nesting depth of `x`'s shape (CuTe DSL's `cute.depth(x)`): 0 for an
/// integer, 1 for a flat tuple (the empty tuple included), and one more for
/// each level of nesting, so `((2,3),4)` has depth 2. `x` is as in
/// [`size`].
pub fn depth<X>(x: &X) -> usize
where
    X: HasShape + ?Sized,
{
    x.shape().depth()
}

/// Things whose codomain size [`cosize`] measures: layouts, swizzled
/// layouts and tensors, and references to them.
pub trait HasCosize {
    /// The codomain size of the sub-layout at the mode path `mode` (empty
    /// for the whole); see [`cosize`].
    #[track_caller]
    fn cosize_at(&self, mode: &[usize]) -> IndexExpr;
}

impl HasCosize for Layout {
    #[track_caller]
    fn cosize_at(&self, mode: &[usize]) -> IndexExpr {
        let shape = select_mode(&self.shape, mode, "cosize").clone();
        let stride = self.stride.get(mode).clone();
        cosize_of(&shape, &stride)
    }
}

impl<P> HasCosize for P
where
    P: Deref + ?Sized,
    P::Target: HasCosize,
{
    #[track_caller]
    fn cosize_at(&self, mode: &[usize]) -> IndexExpr {
        (**self).cosize_at(mode)
    }
}

/// The codomain size of `x` at the mode path `mode` (CuTe DSL's
/// `cute.cosize(x, mode=...)`): one past the largest offset of the
/// sub-layout there, `l(size(l) - 1) + 1`, as an [`IndexExpr`] that is
/// static when the extents and strides it depends on are. The path works as
/// in [`size`]. For a swizzled composed layout it is its outer layout's,
/// as in CuTe ([`ComposedLayout::cosize`](super::ComposedLayout::cosize)). A tensor's cosize is its layout's: neither the base
/// offset of a slice or a partition nor a swizzle counts, as in CuTe, and an
/// identity tensor's is the size of its coshape (the product over its modes
/// of one past the largest coordinate). For a static answer, use the
/// `usize` [`Layout::cosize`] instead.
#[track_caller]
pub fn cosize(x: &(impl HasCosize + ?Sized), mode: &[usize]) -> IndexExpr {
    x.cosize_at(mode)
}

/// `shape:stride` as CuTe prints it, with `?` for a runtime leaf:
/// `(?,8):(8,1)`.
impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// `shape:stride` with the expressions of runtime leaves: `(%3,8):(8,1)`.
impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.shape, self.stride)
    }
}

/// `make_layout(shape, stride)`: leaves may be static integers or runtime
/// values, so `make_layout((m, 8), (8, 1))` works with `m` an
/// [`Int32`](crate::Int32).
#[track_caller]
pub fn make_layout(shape: impl Into<Shape>, stride: impl Into<Stride>) -> Layout {
    Layout::new(shape, stride)
}

/// Merge contiguous modes and drop size-1 modes (same function, flat result).
pub fn coalesce<X: Tileable>(layout: &X) -> X {
    layout.with_layout(layout.layout_ref().coalesce())
}

/// Functional composition `lhs ∘ rhs`; the result has the shape of `rhs`.
#[track_caller]
pub fn composition<X: Tileable>(lhs: &X, rhs: &Layout) -> X {
    let rhs = rhs.clone();
    let layout = lhs.layout_ref().compose(&rhs);
    lhs.map_layout(layout, move |l| l.compose(&rhs))
}

/// The layout of the offsets in `[0, cotarget)` that `layout` does not reach,
/// ordered so that `(layout, complement)` is injective. The cotarget may be
/// a runtime value.
#[track_caller]
pub fn complement(layout: &Layout, cotarget: impl Into<IndexExpr>) -> Layout {
    layout.complement(cotarget)
}

/// `R` with `layout(R(i)) == i` on the largest contiguous range `layout` covers.
pub fn right_inverse(layout: &Layout) -> Layout {
    layout.right_inverse()
}

/// `L` with `L(layout(i)) == i` on the domain of an injective `layout`.
#[track_caller]
pub fn left_inverse(layout: &Layout) -> Layout {
    layout.left_inverse()
}

/// Panic unless `block`, the layout a product replicates, is static.
#[track_caller]
fn product_block(block: &Layout, what: &str) {
    block.require_static(what, "a product replicates a static layout (only the tiler may be dynamic)");
}

#[track_caller]
fn logical_product_impl(a: &Layout, b: &Layout) -> Layout {
    let rest = a.complement(st(a.size()) * cosize_of(&b.shape, &b.stride)).compose(b);
    Layout::from_modes([a.clone(), rest])
}

/// `(a, complement(a) ∘ b)`: `b` repeated over the codomain of `a`. `a` must
/// be static; `b`, the tiler, may have runtime extents and strides.
///
/// The tiler may also be a tile, applied mode by mode as in CuTe DSL: a
/// layout is the whole tiler, an integer `n` is the layout `n:1`, `..`
/// (CuTe's `None`) keeps the mode, and a tuple of these applies each to the
/// matching mode of `a` (and keeps the modes past it), so it may not have
/// more modes than `a`. Only the modes of `a` that are multiplied must be
/// static. A tiler extent may be a runtime [`Int32`](crate::Int32):
/// `logical_product(&block, (n, ..))` repeats mode 0 of `block` `n` times.
#[track_caller]
pub fn logical_product<X: Tileable>(a: &X, tiler: impl Into<Tiler>) -> X {
    let l = a.layout_ref();
    a.with_layout(logical_product_tile(l, &tiler.into(), &ModePath::whole(l)))
}

#[track_caller]
fn logical_product_tile(a: &Layout, tiler: &Tiler, at: &ModePath) -> Layout {
    match tiler {
        Tree::Leaf(None) => a.clone(),
        Tree::Leaf(Some(b)) => {
            if at.path.is_empty() {
                product_block(a, "logical_product");
            } else {
                a.require_static(
                    &format!("logical_product: {}", at.describe(a)),
                    "a product replicates a static layout (only the tiler may be dynamic)",
                );
            }
            logical_product_impl(a, b)
        }
        Tree::Branch(ts) => {
            assert!(
                ts.len() <= a.rank(),
                "logical_product: the tiler has {} modes but {} has rank {}",
                ts.len(),
                at.describe(a),
                a.rank()
            );
            let mut modes = Vec::with_capacity(a.rank());
            for i in 0..a.rank() {
                let mode = a.mode(i);
                modes.push(match ts.get(i) {
                    Some(t) => logical_product_tile(&mode, t, &at.child(i)),
                    None => mode,
                });
            }
            Layout::from_modes(modes)
        }
    }
}

fn zip_modes(first: &Layout, second: &Layout) -> Layout {
    assert_eq!(first.rank(), second.rank());
    Layout::from_modes((0..first.rank()).map(|i| Layout::from_modes([first.mode(i), second.mode(i)])))
}

/// Repeat `block` according to `tiler`, keeping each block contiguous:
/// mode `i` of the result is `(block_i, tiler_i)`. `block` must be static.
#[track_caller]
pub fn blocked_product<X: Tileable>(block: &X, tiler: &Layout) -> X {
    block.with_layout(blocked_product_impl(block.layout_ref(), tiler))
}

#[track_caller]
fn blocked_product_impl(block: &Layout, tiler: &Layout) -> Layout {
    product_block(block, "blocked_product");
    let r = block.rank().max(tiler.rank());
    let p = logical_product_impl(&block.pad_to_rank(r), &tiler.pad_to_rank(r));
    zip_modes(&p.mode(0), &p.mode(1))
}

/// Interleave `block` across `tiler`: mode `i` of the result is
/// `(tiler_i, block_i)`, so neighbouring elements come from different blocks.
/// `block` must be static.
#[track_caller]
pub fn raked_product<X: Tileable>(block: &X, tiler: &Layout) -> X {
    block.with_layout(raked_product_impl(block.layout_ref(), tiler))
}

#[track_caller]
fn raked_product_impl(block: &Layout, tiler: &Layout) -> Layout {
    product_block(block, "raked_product");
    let r = block.rank().max(tiler.rank());
    let p = logical_product_impl(&block.pad_to_rank(r), &tiler.pad_to_rank(r));
    zip_modes(&p.mode(1), &p.mode(0))
}

/// Build a thread-value layout from a thread layout and a per-thread value
/// layout (CuTe DSL's `make_layout_tv`). Returns `(tiler_mn, layout_tv)`
/// where `layout_tv` maps `(thread, value)` to a colexicographic index into a
/// tile of shape `tiler_mn`. Thread `t` owns a `val_layout`-shaped block at
/// the position `thr_layout` gives it. Both layouts must be static.
#[track_caller]
pub fn make_layout_tv(thr_layout: &Layout, val_layout: &Layout) -> (IntTuple, Layout) {
    let why = "thread-value layouts are static";
    thr_layout.require_static("make_layout_tv: the thread layout", why);
    val_layout.require_static("make_layout_tv: the value layout", why);
    let layout_mn = raked_product(thr_layout, val_layout);
    let tv = Layout::col_major((thr_layout.size(), val_layout.size()));
    let layout_tv = layout_mn.right_inverse().compose(&tv);
    (layout_mn.static_shape().product_each(), layout_tv)
}

/// Repeat `block` until it covers `shape`, placing the copies column-major
/// (along mode 0 first): [`tile_to_shape_ordered`] with the order
/// `0, 1, ..., rank(shape) - 1`.
#[track_caller]
pub fn tile_to_shape<X: Tileable>(block: &X, shape: impl Into<Shape>) -> X {
    let shape = shape.into();
    let order: Vec<usize> = (0..shape.rank()).collect();
    tile_to_shape_ordered(block, shape, &order)
}

/// Repeat `block` until it covers `shape`, placing the copies in the mode
/// order `order` (CuTe DSL's `tile_to_shape(atom, trg_shape, order)`). The
/// block is padded with `1:0` modes to the rank of `shape`, and mode `i` of
/// the result is `(block_i, ceil(shape_i / block_i))`: the block and how
/// often it repeats along that mode. A mode the block does not divide is
/// rounded up, as the CuTe dialect does (C++ CuTe rejects it), so the result
/// can be larger than `shape`. The target's extents may be runtime values;
/// the block must be static.
///
/// `order` must be a permutation of `0..rank(shape)`: the copies are laid
/// out along the mode with order 0 first, then the mode with order 1, and so
/// on. `[0, 1, 2]` is column-major; `[1, 0, 2]` fills mode 1 first, as the
/// staged shared-memory layouts of MN-major operands do. This is
/// `blocked_product(block, make_ordered_layout(repeats, order))`.
#[track_caller]
pub fn tile_to_shape_ordered<X: Tileable>(block: &X, shape: impl Into<Shape>, order: &[usize]) -> X {
    block.with_layout(tile_to_shape_ordered_impl(block.layout_ref(), shape.into(), order))
}

#[track_caller]
fn tile_to_shape_ordered_impl(block: &Layout, shape: Shape, order: &[usize]) -> Layout {
    let r = shape.rank();
    assert!(
        block.rank() <= r,
        "tile_to_shape: block {block} has rank {}, more than the rank-{r} target shape {shape}",
        block.rank()
    );
    let mut seen = vec![false; r];
    let is_permutation = order.len() == r && order.iter().all(|&o| o < r && !std::mem::replace(&mut seen[o], true));
    assert!(
        is_permutation,
        "tile_to_shape: order {order:?} is not a permutation of 0..{r} (one position for each mode of {shape})"
    );
    product_block(block, "tile_to_shape: the block");
    let block = block.pad_to_rank(r);
    let block_shape = block.shape().product_each();
    let target = shape.product_each();
    let repeats: Vec<IndexExpr> =
        (0..r).map(|i| IndexExpr::ceil_div(target.mode(i).product(), block_shape.mode(i).product())).collect();
    blocked_product(&block, &make_ordered_layout(repeats, order))
}

/// `((block...), (tiler...))`: `tiler` repeating `block`, with the two
/// gathered. `block` must be static.
#[track_caller]
pub fn zipped_product<X: Tileable>(block: &X, tiler: &Layout) -> X {
    block.with_layout(zipped_product_impl(block.layout_ref(), tiler))
}

#[track_caller]
fn zipped_product_impl(block: &Layout, tiler: &Layout) -> Layout {
    product_block(block, "zipped_product");
    let r = block.rank().max(tiler.rank());
    logical_product_impl(&block.pad_to_rank(r), &tiler.pad_to_rank(r))
}

/// `((block...), tiler0, tiler1, ...)`.
#[track_caller]
pub fn tiled_product<X: Tileable>(block: &X, tiler: &Layout) -> X {
    block.with_layout(tiled_product_impl(block.layout_ref(), tiler))
}

#[track_caller]
fn tiled_product_impl(block: &Layout, tiler: &Layout) -> Layout {
    let z = zipped_product(block, tiler);
    let mut modes = vec![z.mode(0)];
    modes.extend(z.mode(1).modes());
    Layout::from_modes(modes)
}

/// `(block0, ..., tiler0, ...)`.
#[track_caller]
pub fn flat_product<X: Tileable>(block: &X, tiler: &Layout) -> X {
    block.with_layout(flat_product_impl(block.layout_ref(), tiler))
}

#[track_caller]
fn flat_product_impl(block: &Layout, tiler: &Layout) -> Layout {
    let z = zipped_product(block, tiler);
    let mut modes = z.mode(0).modes();
    modes.extend(z.mode(1).modes());
    Layout::from_modes(modes)
}

/// The layout of the longest run of elements that correspond in `a` and `b`
/// and are contiguous in both (CuTe's `max_common_layout`): with `inv_b =
/// right_inverse(b)` and `common = coalesce(composition(a, inv_b))`, it is
/// `composition(inv_b, common[0])` if mode 0 of `common` has a static
/// extent and a static stride of 1, and `1:0` otherwise. It maps `[0, n)`
/// to 1-D indices of the domain with `b(L(i)) == i` and `a(L(i))` contiguous.
/// `a` and `b` are [`LayoutSource`]s: layouts, or tensors (their layouts
/// before any swizzle). Runtime strides are skipped by the inverse and read
/// symbolically, so a tile with a runtime row stride still finds its rows:
/// `(8,128):(k,1)` against `(8,128):(128,1)` gives `128:8`.
#[track_caller]
pub fn max_common_layout(a: &(impl LayoutSource + ?Sized), b: &(impl LayoutSource + ?Sized)) -> Layout {
    let (a, b) = (a.source_layout(), b.source_layout());
    let inverse = b.right_inverse();
    let common = a.compose(&inverse).coalesce();
    match static_unit_run(&common) {
        Some(n) => inverse.compose(&Layout::new(n, 1)),
        None => Layout::new(1, 0),
    }
}

/// How many leading elements correspond in `a` and `b` and are contiguous
/// (CuTe's `max_common_vector`): the widest vector copy between them, a
/// static number. A run whose length is a runtime value gives the largest
/// static number known to divide it. Coalescing merges `(8,m):(1,8)` into
/// `(8*m):1`, which gives 8; C++ CuTe, which never merges a static extent
/// into a dynamic one, keeps `(8,m):(1,8)` and also finds 8. A runtime
/// extent with no static factor gives 1. Takes layouts and tensors, as
/// [`max_common_layout`] does.
#[track_caller]
pub fn max_common_vector(a: &(impl LayoutSource + ?Sized), b: &(impl LayoutSource + ?Sized)) -> usize {
    let (a, b) = (a.source_layout(), b.source_layout());
    static_unit_run(&a.compose(&b.right_inverse()).coalesce()).unwrap_or(1)
}

/// The static length of the run of offsets `0, 1, 2, ...` that a coalesced
/// layout starts with: the extent of its first mode if that mode's stride
/// is a static 1, or, for a runtime extent, the largest static number known
/// to divide it (an extent is at least 1, so it is at least that long).
/// `None` if the first stride is not a static 1.
fn static_unit_run(common: &Layout) -> Option<usize> {
    let (s0, d0) = common.flat_modes_expr().swap_remove(0);
    (d0.static_value() == Some(1)).then(|| s0.known_divisor())
}

/// Keep the modes where `mask` is true (CuTe's `dice`).
#[track_caller]
pub fn dice(layout: &Layout, mask: &[bool]) -> Layout {
    assert_eq!(layout.rank(), mask.len(), "dice: layout {layout} and mask differ in rank");
    let keep: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
    layout.select(&keep)
}

/// A compact layout of `shape` whose modes are filled in the order given by
/// `order`: smaller values get smaller strides (CuTe's `make_ordered_layout`).
/// A dynamic extent makes the strides of the modes after it dynamic:
/// `make_ordered_layout((4,3,?,2), [2,1,3,4])` is `(4,3,?,2):(3,1,12,?)`.
#[track_caller]
pub fn make_ordered_layout(shape: impl Into<Shape>, order: &[usize]) -> Layout {
    let shape = shape.into();
    let sizes = shape.product_each().flat_leaves();
    assert_eq!(sizes.len(), order.len(), "make_ordered_layout: shape {shape} and order differ in rank");
    let mut by_order: Vec<usize> = (0..order.len()).collect();
    by_order.sort_by_key(|&i| order[i]);
    let mut stride = vec![st(0); order.len()];
    let mut running = st(1);
    for i in by_order {
        stride[i] = running.clone();
        running = running * &sizes[i];
    }
    Layout::from_modes(sizes.iter().zip(&stride).map(|(s, d)| Layout::new(s.clone(), d.clone())))
}

/// The layouts that the layout-like functions ([`make_layout_like`],
/// [`make_fragment_like_layout`], [`max_common_layout`],
/// [`max_common_vector`]) take, as CuTe DSL's take a `Layout`, a
/// `ComposedLayout` or a `Tensor`: a [`Layout`] itself, the layout of a
/// [`ComposedLayout`](crate::ComposedLayout) (before its swizzle), and the layout of a
/// [`Tensor`](crate::Tensor) (before its swizzle, for swizzled shared
/// memory).
pub trait LayoutSource {
    /// The plain layout the layout-like functions read.
    fn source_layout(&self) -> &Layout;
}

impl LayoutSource for Layout {
    fn source_layout(&self) -> &Layout {
        self
    }
}

/// CuTe's `compact_order(shape, order)` (include/cute/stride.hpp) for an
/// order congruent with the shape, flattened: the stride of each leaf is the
/// product of the extents of the leaves of smaller order, or 0 for a static
/// extent of 1 (`compact_major`'s rule). Ties share a stride. A `None` order
/// is a runtime one: larger than every static order, and ordered left to
/// right among the runtime ones.
fn compact_order(shapes: &[IndexExpr], orders: &[Option<usize>]) -> Vec<IndexExpr> {
    let max_static = orders.iter().flatten().copied().max().unwrap_or(0);
    let order: Vec<usize> = orders.iter().enumerate().map(|(i, o)| o.unwrap_or(max_static + 1 + i)).collect();
    (0..shapes.len())
        .map(|i| {
            if is_one(&shapes[i]) {
                return st(0);
            }
            (0..shapes.len()).filter(|&j| order[j] < order[i]).fold(st(1), |acc, j| acc * &shapes[j])
        })
        .collect()
}

/// CuTe's compact column-major strides (`compact_major<LayoutLeft>`): the
/// exclusive prefix products of the extents, with 0 for a static extent
/// of one. [`Layout::col_major`] gives such a mode the running product
/// instead; the two compute the same function.
fn compact_left(shapes: &[IndexExpr]) -> Vec<IndexExpr> {
    let mut running = st(1);
    shapes
        .iter()
        .map(|s| {
            let d = if is_one(s) { st(0) } else { running.clone() };
            running = running.clone() * s;
            d
        })
        .collect()
}

/// The static order of an integer stride, `None` for a runtime one.
fn stride_order(d: &StrideLeaf) -> Option<usize> {
    d.int().as_static()
}

/// A compact layout with the shape of `src` (a [`LayoutSource`]: a layout,
/// a composed layout's layout, or a tensor's layout) whose strides follow
/// the order of its strides (CuTe's `make_layout_like`, C++
/// `make_layout(shape, compact_order(filter_zeros(stride, shape), stride))`).
/// The hierarchy is kept. Modes with a static stride of 0 keep stride 0;
/// the others get compact strides in increasing order of their strides, and
/// a static extent of 1 gets stride 0, as in CuTe. A runtime stride counts
/// as larger than every static one, and runtime strides are ordered left to
/// right among themselves, so `(2,3,4,5):(0,k,1,0)` becomes
/// `(2,3,4,5):(0,4,1,0)`. Extents may be runtime values (the strides after
/// one are then runtime products). Panics on a basis stride.
#[track_caller]
pub fn make_layout_like(src: &(impl LayoutSource + ?Sized)) -> Layout {
    let layout = src.source_layout();
    layout.require_int_strides("make_layout_like", "the modes are ordered by integer strides");
    let strides = layout.stride.flat_leaves();
    let shapes: Vec<IndexExpr> = layout
        .shape
        .flat_leaves()
        .into_iter()
        .zip(&strides)
        .map(|(s, d)| if d.is_zero() { st(1) } else { s })
        .collect();
    let orders: Vec<Option<usize>> = strides.iter().map(stride_order).collect();
    let compact: Vec<StrideLeaf> = compact_order(&shapes, &orders).into_iter().map(StrideLeaf::Int).collect();
    Layout { shape: layout.shape.clone(), stride: Tree::unflatten(&compact, &layout.stride) }
}

/// The layout of a register fragment for `src` as C++ CuTe's
/// `make_fragment_like(layout)` builds it (include/cute/layout.hpp; CuTe
/// DSL's `make_fragment_like` of a layout or a composed layout, whose
/// layout before the swizzle is used). For a static shape of rank above 1,
/// mode 0 (the values an atom moves) is compact column-major, keeping its
/// stride-0 parts, and the other modes follow it ordered by their strides
/// (`tiled_product(make_layout(shape0, compact_major<LayoutLeft>(filter_zeros(stride0,
/// shape0))), make_ordered_layout(shape[1..R], stride[1..R]))`). Otherwise
/// it is the compact column-major layout of the shape. Size-1 modes get
/// stride 0, as in CuTe. Panics on a basis stride.
#[track_caller]
pub fn make_fragment_like_layout(src: &(impl LayoutSource + ?Sized)) -> Layout {
    let layout = src.source_layout();
    layout.require_int_strides("make_fragment_like_layout", "the modes are ordered by integer strides");
    let shapes = layout.shape.flat_leaves();
    if layout.rank() <= 1 || !layout.shape.is_static() {
        let compact: Vec<StrideLeaf> = compact_left(&shapes).into_iter().map(StrideLeaf::Int).collect();
        return Layout { shape: layout.shape.clone(), stride: Tree::unflatten(&compact, &layout.stride) };
    }
    let m0 = layout.mode(0);
    let s0: Vec<IndexExpr> = m0
        .shape
        .flat_leaves()
        .into_iter()
        .zip(m0.stride.flat_leaves())
        .map(|(s, d)| if d.is_zero() { st(1) } else { s })
        .collect();
    let k = s0.iter().fold(st(1), |acc, s| acc * s);
    let mut strides = compact_left(&s0);
    let rest: Vec<(IndexExpr, StrideLeaf)> =
        (1..layout.rank()).flat_map(|i| layout.mode(i).flat_modes_expr()).collect();
    let rest_shapes: Vec<IndexExpr> = rest.iter().map(|(s, _)| s.clone()).collect();
    let rest_orders: Vec<Option<usize>> = rest.iter().map(|(_, d)| stride_order(d)).collect();
    strides.extend(compact_order(&rest_shapes, &rest_orders).into_iter().map(|d| d * &k));
    let strides: Vec<StrideLeaf> = strides.into_iter().map(StrideLeaf::Int).collect();
    Layout { shape: layout.shape.clone(), stride: Tree::unflatten(&strides, &layout.stride) }
}

impl HasShape for Layout {
    type Leaf = IndexExpr;

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Whether `a` and `b` have the same nesting: both leaves, or tuples of the
/// same rank whose modes are congruent (CuTe's `is_congruent`). Tuples,
/// layouts, swizzled layouts and tensors are compared by their shapes, and
/// the leaf types may differ (a coordinate against a shape, say).
pub fn is_congruent(a: &(impl HasShape + ?Sized), b: &(impl HasShape + ?Sized)) -> bool {
    a.shape().congruent(b.shape())
}

/// Whether `a`'s nesting is a coarsening of `b`'s (CuTe's
/// `is_weakly_congruent`): `a` is a leaf, or both are tuples of the same
/// rank whose modes are weakly congruent. Takes the same arguments as
/// [`is_congruent`]; `4` is weakly congruent to `(2,2)`, but not the
/// reverse.
pub fn is_weakly_congruent(a: &(impl HasShape + ?Sized), b: &(impl HasShape + ?Sized)) -> bool {
    a.shape().weakly_congruent(b.shape())
}

/// `t` as CuTe prints it, with `?` for a leaf that is not a static integer.
fn static_view<T: StaticLeaf>(t: &Tree<T>) -> Tree<String> {
    t.map(&mut |v| v.static_value().map_or_else(|| "?".to_string(), |v| v.to_string()))
}

/// The position of the leading dimension of a shape and stride (CuTe DSL's
/// `leading_dim`): the first stride, depth first, that is a static 1 where
/// the shape is not 1 (a runtime extent counts as not 1). The position is a
/// path as [`Tree::find_if`] returns it: `[i]` for top-level mode `i`, `[i,
/// j]` for mode `j` of mode `i`. `None` if there is no such stride: none is
/// a static 1, or each static unit stride is on an extent of 1, as in shape
/// `(1,1)` and stride `(1,1)`. Panics if the shape and stride are not
/// congruent.
#[track_caller]
pub fn leading_dim<S: StaticLeaf, D: StaticLeaf>(shape: &Tree<S>, stride: &Tree<D>) -> Option<Vec<usize>> {
    assert!(
        shape.congruent(stride),
        "leading_dim: shape {} and stride {} are not congruent",
        static_view(shape),
        static_view(stride)
    );
    stride.find_if(|d, path| {
        d.static_value() == Some(1) && matches!(shape.get(path), Tree::Leaf(s) if s.static_value() != Some(1))
    })
}

/// Whether the first stride of top-level mode `mode` is a static 1 (CuTe
/// DSL's `is_major(mode, stride)`): `is_major(0, stride)` asks whether
/// mode 0 is the contiguous one. A runtime stride is never major, so the
/// answer is known when the kernel is built. Panics if `stride` has no mode
/// `mode`, or that mode has no first stride (it is, or starts with, an empty
/// tuple).
#[track_caller]
pub fn is_major<D: StaticLeaf>(mode: usize, stride: &Tree<D>) -> bool {
    first_stride_is_one(mode, stride, || format!("stride {}", static_view(stride)))
}

/// [`is_major`] for `stride`, which `what` describes in a misuse message.
#[track_caller]
fn first_stride_is_one<D: StaticLeaf>(mode: usize, stride: &Tree<D>, what: impl Fn() -> String) -> bool {
    let rank = stride.rank();
    assert!(mode < rank, "is_major: mode {mode} is out of range for the rank-{rank} {}", what());
    match stride.mode(mode).first_leaf() {
        Ok(first) => first.static_value() == Some(1),
        Err(0) => panic!("is_major: mode {mode} of the {} is the empty tuple (), which has no first stride", what()),
        Err(_) => panic!("is_major: mode {mode} of the {} has no first stride: it starts with an empty tuple", what()),
    }
}

/// The kernel of `layout` (CuTe's `nullspace`); see [`Layout::nullspace`].
pub fn nullspace(layout: &Layout) -> Layout {
    layout.nullspace()
}

/// Append size-1 modes up to rank `r`.
pub fn append_ones(layout: &Layout, r: usize) -> Layout {
    layout.pad_to_rank(r)
}

/// Prepend size-1 modes up to rank `r`.
pub fn prepend_ones(layout: &Layout, r: usize) -> Layout {
    let mut out = layout.clone();
    while out.rank() < r {
        out = out.prepend(Layout::new(1, 0));
    }
    out
}

/// Things with a layout that can be re-tiled: layouts and tensors.
pub trait Tileable: Sized {
    fn layout_ref(&self) -> &Layout;
    #[track_caller]
    fn with_layout(&self, layout: Layout) -> Self;
    /// The object rebuilt with `layout`, which the caller computed as
    /// `f(self.layout_ref())` (outside `f`, so that its panics report the
    /// caller's line). A tensor also records `f` on its predication origin
    /// (roadmap part m11-cute-ext-core), so the view functions of the
    /// algebra go through this rather than [`with_layout`](Self::with_layout).
    #[track_caller]
    fn map_layout<F: Fn(&Layout) -> Layout + 'static>(&self, layout: Layout, f: F) -> Self {
        let _ = f;
        self.with_layout(layout)
    }
}

/// The layout-like objects: [`Layout`], [`ComposedLayout`](super::ComposedLayout)
/// and [`Tensor`](crate::Tensor). They share the algebra through
/// [`Tileable`] (the layout the algebra acts on, and the object rebuilt
/// with a new one: composition, the divides and the products,
/// [`tile_to_shape`], [`coalesce`]), and the queries through [`HasShape`]
/// ([`size`], [`shape`], [`rank`], [`depth`]) and [`HasCosize`] ([`cosize`]).
pub trait LayoutLike: Tileable + HasShape<Leaf = IndexExpr> + HasCosize {}

impl<X: Tileable + HasShape<Leaf = IndexExpr> + HasCosize> LayoutLike for X {}

impl Tileable for Layout {
    fn layout_ref(&self) -> &Layout {
        self
    }
    fn with_layout(&self, layout: Layout) -> Self {
        layout
    }
}

/// Split `x` into tiles: each tiled mode becomes `(tile, rest)`.
#[track_caller]
pub fn logical_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    let tiler = tiler.into();
    let layout = x.layout_ref().logical_divide(&tiler);
    x.map_layout(layout, move |l| l.logical_divide(&tiler))
}

/// [`logical_divide`] with the tile modes and the rest modes gathered:
/// `((tiles...), (rests...))`.
#[track_caller]
pub fn zipped_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    let tiler = tiler.into();
    let layout = x.layout_ref().zipped_divide(&tiler);
    x.map_layout(layout, move |l| l.zipped_divide(&tiler))
}

/// [`zipped_divide`] with the rest modes flattened: `((tiles...), rest0, rest1, ...)`.
#[track_caller]
pub fn tiled_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    let tiler = tiler.into();
    let layout = x.layout_ref().tiled_divide(&tiler);
    x.map_layout(layout, move |l| l.tiled_divide(&tiler))
}

/// [`zipped_divide`] with everything flattened: `(tile0, ..., rest0, ...)`.
#[track_caller]
pub fn flat_divide<X: Tileable>(x: &X, tiler: impl Into<Tiler>) -> X {
    let tiler = tiler.into();
    let layout = x.layout_ref().flat_divide(&tiler);
    x.map_layout(layout, move |l| l.flat_divide(&tiler))
}

macro_rules! coord_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Coord {
            fn from(v: $t) -> Self { Tree::Leaf(IndexExpr::from(v)) }
        }
        impl From<$t> for SliceCoord {
            fn from(v: $t) -> Self { Tree::Leaf(Some(IndexExpr::from(v))) }
        }
        impl From<$t> for Tiler {
            fn from(v: $t) -> Self { Tree::Leaf(Some(Layout::new(IntTuple::from(v), 1))) }
        }
    )*};
}
coord_from_int!(usize, u8, u16, u32, u64, i32, i64);

impl From<IndexExpr> for Coord {
    fn from(e: IndexExpr) -> Self {
        Tree::Leaf(e)
    }
}

/// A runtime extent `n` (static when it folds) as a tiler: `n:1`.
impl From<IndexExpr> for Tiler {
    fn from(n: IndexExpr) -> Self {
        Tree::Leaf(Some(Layout::new(n, 1)))
    }
}

/// A runtime extent `n` as a tiler: `n:1`, as a tile of a runtime size.
impl From<crate::value::Int32> for Tiler {
    fn from(n: crate::value::Int32) -> Self {
        Tiler::from(IndexExpr::from(n))
    }
}

impl From<IndexExpr> for SliceCoord {
    fn from(e: IndexExpr) -> Self {
        Tree::Leaf(Some(e))
    }
}

impl From<RangeFull> for SliceCoord {
    fn from(_: RangeFull) -> Self {
        Tree::Leaf(None)
    }
}

impl From<RangeFull> for Tiler {
    fn from(_: RangeFull) -> Self {
        Tree::Leaf(None)
    }
}

impl From<&Coord> for SliceCoord {
    fn from(c: &Coord) -> Self {
        c.map(&mut |e| Some(e.clone()))
    }
}

impl From<Coord> for SliceCoord {
    fn from(c: Coord) -> Self {
        SliceCoord::from(&c)
    }
}

impl From<&IntTuple> for Coord {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&v| st(v))
    }
}

/// A static tuple as a [`Shape`] (or [`Stride`], or [`Coord`]).
impl From<IntTuple> for Coord {
    fn from(t: IntTuple) -> Self {
        Coord::from(&t)
    }
}

impl From<Layout> for Tiler {
    fn from(l: Layout) -> Self {
        Tree::Leaf(Some(l))
    }
}

impl From<&Layout> for Tiler {
    fn from(l: &Layout) -> Self {
        Tree::Leaf(Some(l.clone()))
    }
}

impl From<&IntTuple> for Tiler {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&n| Some(Layout::new(n, 1)))
    }
}

/// A shape as a tiler: each extent `n`, static or dynamic, tiles with `n:1`.
impl From<&Shape> for Tiler {
    fn from(t: &Shape) -> Self {
        t.map(&mut |n| Some(Layout::new(n.clone(), 1)))
    }
}

impl From<&Tiler> for Tiler {
    fn from(t: &Tiler) -> Self {
        t.clone()
    }
}

/// A borrowed [`Shape`] (or [`Stride`], or [`Coord`]), cloned, so that a
/// tensor's shape goes straight into what takes one, as in CuTe DSL:
/// `make_identity_tensor(t.shape())`, `Layout::col_major(t.shape())`.
impl From<&Shape> for Shape {
    fn from(t: &Shape) -> Self {
        t.clone()
    }
}

macro_rules! tree_from_seq {
    ($($target:ty),*) => {$(
        impl<T: Into<$target>> From<Vec<T>> for $target {
            fn from(v: Vec<T>) -> Self {
                Tree::Branch(v.into_iter().map(Into::into).collect())
            }
        }
        impl<T: Into<$target>, const N: usize> From<[T; N]> for $target {
            fn from(v: [T; N]) -> Self {
                Tree::Branch(v.into_iter().map(Into::into).collect())
            }
        }
    )*};
}
tree_from_seq!(Coord, SliceCoord, Tiler);

tree_from_tuple!(Coord;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));
tree_from_tuple!(SliceCoord;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));
tree_from_tuple!(Tiler;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F));

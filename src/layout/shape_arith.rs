// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/int_tuple.hpp (ceil_div, round_up and shape_div),
//   include/cute/stride.hpp (idx2crd with a stride, and crd2crd) and
//   include/cute/layout.hpp (get_hier_coord and get_flat_coord),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Shape and index arithmetic over hierarchical tuples whose leaves may be
//! runtime values: [`ceil_div`], [`shape_div`], [`round_up`], [`equal`], and
//! the coordinate of an index ([`Layout::get_hier_coord`],
//! [`Layout::get_flat_coord`]).
//!
//! Static leaves fold when the kernel is built; a runtime leaf gives a
//! runtime result (an [`IndexExpr`]), as in CuTe DSL, where a dynamic
//! integer stays dynamic.

use super::basis::{BasisScale, StrideLeaf};
use super::index::IndexExpr;
use super::layout::{Coord, Layout, Shape, Stride, Tiler};
use super::swizzle::ComposedLayout;
use super::tree::{IntTuple, Tree};
use crate::value::{Boolean, Int32, Int64, IntoValue};

/// The divisor of [`ceil_div`]: a shape (an integer or a tuple of them, of
/// static or runtime extents), or a tile (a [`Layout`], or a [`Tiler`]
/// holding layouts), as CuTe DSL's `Tiler` is either.
#[derive(Clone, Debug)]
pub enum ShapeTiler {
    /// An integer or a tuple: the result has the input's profile.
    Shape(Shape),
    /// A layout or a tiler of layouts: the result is the rest mode of
    /// [`zipped_divide`](super::zipped_divide), whose profile may differ.
    Tile(Tiler),
}

macro_rules! shape_tiler_from_shape {
    ($($t:ty),*) => {$(
        impl From<$t> for ShapeTiler {
            fn from(v: $t) -> Self { ShapeTiler::Shape(Shape::from(v)) }
        }
    )*};
}
shape_tiler_from_shape!(usize, u8, u16, u32, u64, i32, i64, IndexExpr, Int32, IntTuple, &IntTuple, Shape, &Shape);

impl From<Layout> for ShapeTiler {
    fn from(l: Layout) -> Self {
        ShapeTiler::Tile(Tiler::from(l))
    }
}

impl From<&Layout> for ShapeTiler {
    fn from(l: &Layout) -> Self {
        ShapeTiler::Tile(Tiler::from(l))
    }
}

impl From<Tiler> for ShapeTiler {
    fn from(t: Tiler) -> Self {
        ShapeTiler::Tile(t)
    }
}

impl From<&Tiler> for ShapeTiler {
    fn from(t: &Tiler) -> Self {
        ShapeTiler::Tile(t.clone())
    }
}

macro_rules! shape_tiler_from_tuple {
    ($(($($name:ident),+)),+) => {$(
        // The tuple's fields are bound by their generic parameter names (`A`, `B`, ...).
        #[allow(non_snake_case)]
        impl<$($name: Into<Shape>),+> From<($($name,)+)> for ShapeTiler {
            fn from(($($name,)+): ($($name,)+)) -> Self {
                ShapeTiler::Shape(Tree::Branch(vec![$($name.into()),+]))
            }
        }
    )+};
}
shape_tiler_from_tuple!((A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F));

/// The number of tiles of `tiler` it takes to cover `input` (CuTe DSL's
/// `cute.ceil_div`). With a shape as the divisor, the result has the
/// profile of `input`:
///
/// * tuple / tuple divides mode by mode, and needs `rank(input) >=
///   rank(tiler)`; the modes of `input` past the tiler's are kept:
///   `ceil_div((8,10,6), (3,4))` is `(3,3,6)`;
/// * tuple / integer consumes the divisor across the modes:
///   `ceil_div((4,5,6), 40)` is `(1,1,3)`;
/// * integer / tuple divides by the tuple's product;
/// * integer / integer is `(a + b - 1) / b`.
///
/// With a tile (a [`Layout`], or a [`Tiler`] of them), the result is the
/// rest mode of `zipped_divide(make_layout(input), tiler)`, whose profile
/// may differ: `ceil_div((8,16), (4,8):(1,4))` is `4`.
///
/// Extents may be runtime values, which give runtime results, as a grid is
/// sized from a problem shape: `ceil_div((m, n), (128, 128))`. A tiler
/// extent of a static 0, and a tuple tiler of a higher rank than the input,
/// panic.
#[track_caller]
pub fn ceil_div(input: impl Into<Shape>, tiler: impl Into<ShapeTiler>) -> Shape {
    let input = input.into();
    match tiler.into() {
        ShapeTiler::Shape(t) => {
            if t.leaves().iter().any(|e| e.as_static() == Some(0)) {
                panic!("ceil_div: cannot divide {input} by {t}: the tiler has an extent of 0");
            }
            ceil_div_shape(&input, &t, &input, &t)
        }
        ShapeTiler::Tile(t) => Layout::col_major(input).zipped_divide(&t).shape().mode(1).clone(),
    }
}

#[track_caller]
fn ceil_div_shape(a: &Shape, b: &Shape, input: &Shape, tiler: &Shape) -> Shape {
    match (a, b) {
        (Tree::Branch(xs), Tree::Branch(ys)) => {
            if ys.len() > xs.len() {
                panic!(
                    "ceil_div: cannot divide {input} by {tiler}: the tiler {b} has rank {}, more than the rank {} of \
                     the input {a} it divides",
                    ys.len(),
                    xs.len()
                );
            }
            let mut out = Vec::with_capacity(xs.len());
            for (i, x) in xs.iter().enumerate() {
                out.push(match ys.get(i) {
                    Some(y) => ceil_div_shape(x, y, input, tiler),
                    None => x.clone(),
                });
            }
            Tree::Branch(out)
        }
        (Tree::Branch(xs), Tree::Leaf(d)) => {
            let mut rest = d.clone();
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(ceil_div_shape(x, &Tree::Leaf(rest.clone()), input, tiler));
                rest = IndexExpr::ceil_div(rest, x.product());
            }
            Tree::Branch(out)
        }
        (Tree::Leaf(x), Tree::Branch(_)) => Tree::Leaf(IndexExpr::ceil_div(x.clone(), b.product())),
        (Tree::Leaf(x), Tree::Leaf(y)) => Tree::Leaf(IndexExpr::ceil_div(x.clone(), y.clone())),
    }
}

/// CuTe's shape division (CuTe DSL's `cute.shape_div`):
///
/// * tuple / tuple divides mode by mode, and needs equal ranks;
/// * tuple / integer consumes the divisor across the modes:
///   `shape_div((4,5,6), 40)` is `(1,1,3)`;
/// * integer / tuple divides by the tuple's product;
/// * integer / integer needs one of the two to divide the other, and gives
///   `a / b` rounded away from zero, so 1 when `a < b`: `shape_div(12, 4)`
///   is 3 and `shape_div(4, 12)` is 1.
///
/// The divisibility of two static leaves is checked; a runtime leaf is
/// divided rounding up, unchecked, as C++ CuTe does. Unequal ranks, a
/// divisor of a static 0, and static leaves of which neither divides the
/// other panic.
#[track_caller]
pub fn shape_div(a: impl Into<Shape>, b: impl Into<Shape>) -> Shape {
    let (a, b) = (a.into(), b.into());
    shape_div_impl(&a, &b, &a, &b)
}

#[track_caller]
fn shape_div_impl(a: &Shape, b: &Shape, lhs: &Shape, rhs: &Shape) -> Shape {
    match (a, b) {
        (Tree::Branch(xs), Tree::Branch(ys)) => {
            if xs.len() != ys.len() {
                panic!(
                    "shape_div: cannot divide {lhs} by {rhs}: {a} has rank {} and {b} rank {}, and a tuple divides a \
                     tuple mode by mode",
                    xs.len(),
                    ys.len()
                );
            }
            let mut out = Vec::with_capacity(xs.len());
            for (x, y) in xs.iter().zip(ys) {
                out.push(shape_div_impl(x, y, lhs, rhs));
            }
            Tree::Branch(out)
        }
        (Tree::Branch(xs), Tree::Leaf(_)) => {
            let mut rest = b.clone();
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(shape_div_impl(x, &rest, lhs, rhs));
                rest = shape_div_impl(&rest, &Tree::Leaf(x.product()), lhs, rhs);
            }
            Tree::Branch(out)
        }
        (Tree::Leaf(_), Tree::Branch(_)) => shape_div_impl(a, &Tree::Leaf(b.product()), lhs, rhs),
        (Tree::Leaf(x), Tree::Leaf(y)) => Tree::Leaf(match (x.as_static(), y.as_static()) {
            (_, Some(0)) => panic!("shape_div: cannot divide {lhs} by {rhs}: the divisor has an extent of 0"),
            (Some(xv), Some(yv)) if xv.is_multiple_of(yv) => IndexExpr::Static(xv / yv),
            (Some(xv), Some(yv)) if yv.is_multiple_of(xv) => IndexExpr::Static(1),
            (Some(xv), Some(yv)) => panic!(
                "shape_div: cannot divide {lhs} by {rhs}: neither {xv} nor {yv} divides the other ({xv} % {yv} and \
                 {yv} % {xv} are not 0)"
            ),
            _ => IndexExpr::ceil_div(x.clone(), y.clone()),
        }),
    }
}

/// `a` rounded up to a multiple of `b`, leaf by leaf (CuTe DSL's
/// `cute.round_up`): `((a + b - 1) / b) * b`. A tuple `a` needs a tuple `b`
/// of at most its rank, which is padded with 1s: `round_up((9,10,11), (4,8))`
/// is `(12,16,11)`. An integer `a` needs an integer `b`. Leaves may be
/// runtime values. An empty `a`, a rank mismatch and an integer `a` with a
/// tuple `b` panic.
#[track_caller]
pub fn round_up(a: impl Into<Shape>, b: impl Into<Shape>) -> Shape {
    let (a, b) = (a.into(), b.into());
    round_up_impl(&a, &b, &a, &b)
}

#[track_caller]
fn round_up_impl(a: &Shape, b: &Shape, lhs: &Shape, rhs: &Shape) -> Shape {
    match (a, b) {
        (Tree::Branch(xs), _) if xs.is_empty() => {
            panic!("round_up: cannot round {lhs} up to a multiple of {rhs}: {a} is the empty tuple")
        }
        (Tree::Branch(xs), Tree::Branch(ys)) => {
            if ys.len() > xs.len() {
                panic!(
                    "round_up: cannot round {lhs} up to a multiple of {rhs}: {b} has rank {}, more than the rank {} \
                     of {a}",
                    ys.len(),
                    xs.len()
                );
            }
            let one = Tree::Leaf(IndexExpr::Static(1));
            let mut out = Vec::with_capacity(xs.len());
            for (i, x) in xs.iter().enumerate() {
                out.push(round_up_impl(x, ys.get(i).unwrap_or(&one), lhs, rhs));
            }
            Tree::Branch(out)
        }
        (Tree::Branch(_), Tree::Leaf(_)) => panic!(
            "round_up: cannot round {lhs} up to a multiple of {rhs}: the tuple {a} needs a tuple to round to, not the \
             integer {b}"
        ),
        (Tree::Leaf(_), Tree::Branch(_)) => panic!(
            "round_up: cannot round {lhs} up to a multiple of {rhs}: the integer {a} needs an integer to round to, \
             not the tuple {b}"
        ),
        (Tree::Leaf(x), Tree::Leaf(y)) => {
            if y.as_static() == Some(0) {
                panic!("round_up: cannot round {lhs} up to a multiple of {rhs}: a multiple of 0");
            }
            Tree::Leaf(IndexExpr::ceil_div(x.clone(), y.clone()) * y.clone())
        }
    }
}

/// What [`equal`] compares: layouts, swizzled layouts, shapes (and
/// coordinates) and strides.
pub trait EqualOperand {
    /// The pairs of leaves that must be equal for `self` and `other` to be,
    /// or `None` when they are known to differ when the kernel is built
    /// (their profiles differ, say).
    #[doc(hidden)]
    #[track_caller]
    fn leaf_pairs(&self, other: &Self) -> Option<Vec<(IndexExpr, IndexExpr)>>;
}

fn shape_pairs(a: &Shape, b: &Shape, out: &mut Vec<(IndexExpr, IndexExpr)>) -> bool {
    match (a, b) {
        (Tree::Leaf(x), Tree::Leaf(y)) => {
            out.push((x.clone(), y.clone()));
            true
        }
        (Tree::Branch(xs), Tree::Branch(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| shape_pairs(x, y, out))
        }
        _ => false,
    }
}

fn stride_pairs(a: &Stride, b: &Stride, out: &mut Vec<(IndexExpr, IndexExpr)>) -> bool {
    match (a, b) {
        (Tree::Leaf(StrideLeaf::Int(x)), Tree::Leaf(StrideLeaf::Int(y))) => {
            out.push((x.clone(), y.clone()));
            true
        }
        (Tree::Leaf(StrideLeaf::Basis(x)), Tree::Leaf(StrideLeaf::Basis(y))) => {
            x.mode() == y.mode()
                && match (x.value(), y.value()) {
                    (BasisScale::Ratio(p), BasisScale::Ratio(q)) => p.reduced() == q.reduced(),
                    (p, q) => match (p.as_int(), q.as_int()) {
                        (Some(p), Some(q)) => {
                            out.push((p.clone(), q.clone()));
                            true
                        }
                        _ => false,
                    },
                }
        }
        (Tree::Branch(xs), Tree::Branch(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| stride_pairs(x, y, out))
        }
        _ => false,
    }
}

impl EqualOperand for Shape {
    fn leaf_pairs(&self, other: &Self) -> Option<Vec<(IndexExpr, IndexExpr)>> {
        let mut out = Vec::new();
        shape_pairs(self, other, &mut out).then_some(out)
    }
}

impl EqualOperand for Stride {
    fn leaf_pairs(&self, other: &Self) -> Option<Vec<(IndexExpr, IndexExpr)>> {
        let mut out = Vec::new();
        stride_pairs(self, other, &mut out).then_some(out)
    }
}

impl EqualOperand for Layout {
    fn leaf_pairs(&self, other: &Self) -> Option<Vec<(IndexExpr, IndexExpr)>> {
        let mut out = Vec::new();
        (shape_pairs(self.shape(), other.shape(), &mut out) && stride_pairs(self.stride(), other.stride(), &mut out))
            .then_some(out)
    }
}

/// Swizzled layouts are compared when the kernel is built, so both must be
/// static, as in CuTe DSL.
impl EqualOperand for ComposedLayout {
    #[track_caller]
    fn leaf_pairs(&self, other: &Self) -> Option<Vec<(IndexExpr, IndexExpr)>> {
        for l in [self, other] {
            l.outer.require_static("equal", "swizzled layouts are only compared when they are static");
        }
        (self == other).then(Vec::new)
    }
}

/// Whether `a` and `b` are equal, as a runtime [`Boolean`] (CuTe DSL's
/// `Layout.__eq__` and `cute.equal`): they have the same profile, and every
/// pair of leaves (extents and strides) has the same value. It folds to a
/// constant when that is known when the kernel is built: different
/// profiles, a pair of different static leaves, or every pair static or the
/// same expression. Otherwise the remaining pairs are compared at run time,
/// so `m * 2` and `m + m` compare equal there.
///
/// `==` on layouts stays structural equality, decided when the kernel is
/// built (two runtime leaves are equal only if they are the same
/// expression). Swizzled layouts are compared only when they are static.
#[track_caller]
pub fn equal<X: EqualOperand + ?Sized>(a: &X, b: &X) -> Boolean {
    let Some(pairs) = a.leaf_pairs(b) else { return false.into_value() };
    let mut acc: Option<Boolean> = None;
    for (x, y) in pairs {
        if x == y {
            continue;
        }
        if let (Some(xv), Some(yv)) = (x.as_static_signed(), y.as_static_signed()) {
            if xv != yv {
                return false.into_value();
            }
            continue;
        }
        let p = if x.width() == 64 || y.width() == 64 {
            Int64::from(x).eq(Int64::from(y))
        } else {
            Int32::from(x).eq(Int32::from(y))
        };
        acc = Some(match acc {
            Some(a) => a & p,
            None => p,
        });
    }
    acc.unwrap_or_else(|| true.into_value())
}

impl Layout {
    /// The hierarchical coordinate that this layout maps to the offset `idx`
    /// (CuTe DSL's `Layout.get_hier_coord`, C++ CuTe's `idx2crd(idx, shape,
    /// stride)`): `(idx / d) % s` for each leaf `s:d`, and 0 for a leaf of
    /// extent 1. The result is congruent with the shape, and `idx` and the
    /// leaves may be runtime values. As in CuTe, this inverts the layout
    /// only when it is compact (a bijection onto `[0, size)`, with strides in
    /// any order): `(4,8):(8,1)` maps 9 back to `(1,1)`. The strides must be
    /// integers.
    #[track_caller]
    pub fn get_hier_coord(&self, idx: impl Into<IndexExpr>) -> Coord {
        self.require_int_strides("get_hier_coord", "the coordinate of an offset needs integer strides");
        hier_coord(&idx.into(), self.shape(), self.stride())
    }

    /// The coordinate that this layout maps to the offset `idx`, one 1-D
    /// coordinate for each top-level mode (CuTe DSL's
    /// `Layout.get_flat_coord`): [`get_hier_coord`](Layout::get_hier_coord),
    /// with each mode's hierarchical coordinate turned into its
    /// colexicographic index within the mode (C++ CuTe's `crd2crd`). A
    /// layout whose shape is a leaf gives a leaf. Meaningful only for
    /// compact layouts, as `get_hier_coord` is.
    #[track_caller]
    pub fn get_flat_coord(&self, idx: impl Into<IndexExpr>) -> Coord {
        let hier = self.get_hier_coord(idx);
        match self.shape() {
            Tree::Leaf(_) => hier,
            Tree::Branch(modes) => Tree::Branch(
                modes
                    .iter()
                    .zip(hier.modes())
                    .map(|(s, c)| Tree::Leaf(Layout::col_major(s.clone()).offset(c)))
                    .collect(),
            ),
        }
    }
}

fn hier_coord(idx: &IndexExpr, shape: &Shape, stride: &Stride) -> Coord {
    match (shape, stride) {
        (Tree::Branch(ss), Tree::Branch(ds)) => {
            Tree::Branch(ss.iter().zip(ds).map(|(s, d)| hier_coord(idx, s, d)).collect())
        }
        (Tree::Leaf(s), Tree::Leaf(d)) => Tree::Leaf(if s.as_static() == Some(1) {
            IndexExpr::Static(0)
        } else {
            (idx.clone() / d.int().clone()) % s.clone()
        }),
        _ => unreachable!("a layout's shape and stride are congruent"),
    }
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/numeric/arithmetic_tuple.hpp and
//   include/cute/numeric/integral_ratio.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   cutlass_compiler/cutegen/include/cutegen/scaled_basis.hpp and
//   cutlass_compiler/cutegen/include/cutegen/ratio.hpp,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Scaled-basis strides and arithmetic tuples.
//!
//! A stride leaf is either an integer ([`StrideLeaf::Int`]) or a scaled
//! basis element ([`StrideLeaf::Basis`]): a scale times the unit vector of a
//! mode of a tuple, printed `scale@mode`. A layout whose strides are basis
//! elements maps a coordinate to a tuple of integers instead of an offset
//! (an arithmetic tuple): at `(2,3)`, `(4,8):(2@0,1@1)` is `(4,3)`. The
//! identity layout of a shape ([`make_identity_layout`]) maps every
//! coordinate to itself this way.
//!
//! A mode is a path, outermost index first, as [`Tree::get`] takes one:
//! `E([1, 0])` is the unit vector of mode 0 of mode 1, `((0,0,...),(1,0,...),
//! ...)`. It prints innermost first, as CuTe does, so `E([1, 0])` prints
//! `1@0@1`.
//!
//! A scale is an [`IndexExpr`] (static or runtime) or a [`Ratio`].

use std::fmt;
use std::ops::Mul;

use super::index::IndexExpr;
use super::layout::{Coord, Layout, Shape, Stride};
use super::tree::{IntTuple, StaticLeaf, Tree, tree_from_tuple};

/// An exact fraction of two integers (CuTe DSL's `cute.Ratio`, C++ CuTe's
/// `R<N, D>`). It is kept as written: `Ratio::new(2, 4)` is not
/// `Ratio::new(1, 2)` until [`reduced`](Ratio::reduced). Ratios arise as the
/// scales of basis strides when the element width of a tensor differs from
/// the unit a coordinate counts.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    numerator: i64,
    denominator: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

impl Ratio {
    /// `numerator / denominator`. Panics if the denominator is 0.
    #[track_caller]
    pub fn new(numerator: i64, denominator: i64) -> Ratio {
        assert!(denominator != 0, "Ratio::new({numerator}, 0): the denominator must not be 0");
        Ratio { numerator, denominator }
    }

    pub fn numerator(&self) -> i64 {
        self.numerator
    }

    pub fn denominator(&self) -> i64 {
        self.denominator
    }

    /// Whether the denominator divides the numerator.
    pub fn is_integral(&self) -> bool {
        self.numerator % self.denominator == 0
    }

    /// The same value in lowest terms, with a positive denominator:
    /// `Ratio::new(4, 8).reduced()` is `1/2`, `Ratio::new(99, -11).reduced()`
    /// is `-9/1`.
    pub fn reduced(&self) -> Ratio {
        let g = gcd(self.numerator, self.denominator).max(1);
        let sign = if self.denominator < 0 { -1 } else { 1 };
        Ratio { numerator: sign * self.numerator / g, denominator: sign * self.denominator / g }
    }

    /// The value as a float.
    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// The value rounded down to an integer (towards negative infinity).
    pub fn to_int(&self) -> i64 {
        let (n, d) = (self.numerator, self.denominator);
        let q = n / d;
        if n % d != 0 && ((n < 0) != (d < 0)) { q - 1 } else { q }
    }

    /// The value as a non-negative integer, if it is one.
    fn as_usize(&self) -> Option<usize> {
        self.is_integral().then(|| self.numerator / self.denominator).and_then(|v| usize::try_from(v).ok())
    }
}

impl Mul for Ratio {
    type Output = Ratio;
    fn mul(self, rhs: Ratio) -> Ratio {
        Ratio::new(self.numerator * rhs.numerator, self.denominator * rhs.denominator)
    }
}

impl Mul<i64> for Ratio {
    type Output = Ratio;
    fn mul(self, rhs: i64) -> Ratio {
        Ratio::new(self.numerator * rhs, self.denominator)
    }
}

impl Mul<Ratio> for i64 {
    type Output = Ratio;
    fn mul(self, rhs: Ratio) -> Ratio {
        rhs * self
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl fmt::Debug for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The scale of a basis element: an integer, static or runtime, or a
/// [`Ratio`].
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum BasisScale {
    Int(IndexExpr),
    Ratio(Ratio),
}

impl BasisScale {
    /// Whether the scale is known when the kernel is built.
    pub fn is_static(&self) -> bool {
        match self {
            BasisScale::Int(e) => e.is_static(),
            BasisScale::Ratio(_) => true,
        }
    }

    /// The integer scale, or `None` for a ratio.
    pub fn as_int(&self) -> Option<&IndexExpr> {
        match self {
            BasisScale::Int(e) => Some(e),
            BasisScale::Ratio(_) => None,
        }
    }

    /// `self * e`; `None` for a ratio times a runtime value, whose value
    /// the build cannot know. A ratio product that is a whole number becomes
    /// an integer scale.
    fn times(&self, e: &IndexExpr) -> Option<BasisScale> {
        match self {
            BasisScale::Int(s) => Some(BasisScale::Int(s.clone() * e)),
            BasisScale::Ratio(r) => {
                let v = i64::try_from(e.as_static()?).ok()?;
                let p = (*r * v).reduced();
                Some(match p.as_usize() {
                    Some(n) => BasisScale::Int(IndexExpr::Static(n)),
                    None => BasisScale::Ratio(p),
                })
            }
        }
    }

    /// Whether the two scales are provably the same value.
    fn same_value(&self, other: &BasisScale) -> bool {
        match (self, other) {
            (BasisScale::Int(a), BasisScale::Int(b)) => a.same_value(b),
            (BasisScale::Ratio(a), BasisScale::Ratio(b)) => a.reduced() == b.reduced(),
            _ => false,
        }
    }
}

impl fmt::Display for BasisScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisScale::Int(e) => write!(f, "{e}"),
            BasisScale::Ratio(r) => write!(f, "{r}"),
        }
    }
}

impl fmt::Debug for BasisScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisScale::Int(e) => write!(f, "{e:?}"),
            BasisScale::Ratio(r) => write!(f, "{r}"),
        }
    }
}

macro_rules! scale_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for BasisScale {
            fn from(v: $t) -> Self { BasisScale::Int(IndexExpr::from(v)) }
        }
        impl From<$t> for StrideLeaf {
            fn from(v: $t) -> Self { StrideLeaf::Int(IndexExpr::from(v)) }
        }
        impl From<$t> for Stride {
            fn from(v: $t) -> Self { Tree::Leaf(StrideLeaf::Int(IndexExpr::from(v))) }
        }
    )*};
}
scale_from_int!(usize, u8, u16, u32, u64, i32, i64);

/// `scale * E(mode)`: a basis element scaled by an integer, so `2 * E(0)` is
/// `2@0`. (Only `usize` among the integer types, so that a literal scale
/// needs no suffix; `E(mode) * n` takes any integer.)
impl Mul<StrideLeaf> for usize {
    type Output = StrideLeaf;
    #[track_caller]
    fn mul(self, rhs: StrideLeaf) -> StrideLeaf {
        rhs * IndexExpr::from(self)
    }
}

impl From<IndexExpr> for BasisScale {
    fn from(e: IndexExpr) -> Self {
        BasisScale::Int(e)
    }
}

impl From<Ratio> for BasisScale {
    fn from(r: Ratio) -> Self {
        BasisScale::Ratio(r)
    }
}

/// A mode path for a basis element: a single top-level mode (`0`) or a path
/// of indices, outermost first (`[1, 0]`).
pub trait BasisMode {
    fn into_mode(self) -> Vec<usize>;
}

impl BasisMode for usize {
    fn into_mode(self) -> Vec<usize> {
        vec![self]
    }
}

impl<const N: usize> BasisMode for [usize; N] {
    fn into_mode(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl BasisMode for &[usize] {
    fn into_mode(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl BasisMode for Vec<usize> {
    fn into_mode(self) -> Vec<usize> {
        self
    }
}

/// A scaled basis element (CuTe DSL's `cute.ScaledBasis(value, mode)`, C++
/// CuTe's `ScaledBasis<T, Ns...>`): `value` times the unit vector of the
/// mode at the path `mode`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ScaledBasis {
    value: BasisScale,
    mode: Vec<usize>,
}

impl ScaledBasis {
    /// `value` times the unit vector of `mode`, a top-level mode or a path,
    /// outermost first. Panics if the path is empty: the unit of the empty
    /// path is the integer 1, which [`E`] gives.
    #[track_caller]
    pub fn new(value: impl Into<BasisScale>, mode: impl BasisMode) -> ScaledBasis {
        let mode = mode.into_mode();
        assert!(!mode.is_empty(), "ScaledBasis::new: the mode path is empty; E([]) is the integer 1");
        ScaledBasis { value: value.into(), mode }
    }

    /// The scale.
    pub fn value(&self) -> &BasisScale {
        &self.value
    }

    /// The mode path, outermost first.
    pub fn mode(&self) -> &[usize] {
        &self.mode
    }

    /// Whether the scale is known when the kernel is built (the mode always
    /// is).
    pub fn is_static(&self) -> bool {
        self.value.is_static()
    }
}

/// `value@mode`, innermost mode first: `ScaledBasis::new(1, [1, 0])` prints
/// `1@0@1`.
impl fmt::Display for ScaledBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)?;
        self.mode.iter().rev().try_for_each(|m| write!(f, "@{m}"))
    }
}

impl fmt::Debug for ScaledBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.value)?;
        self.mode.iter().rev().try_for_each(|m| write!(f, "@{m}"))
    }
}

/// A leaf of a [`Stride`]: an integer, static or runtime, or a scaled basis
/// element (CuTe DSL's `Stride` leaf, `Int | ScaledBasis`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum StrideLeaf {
    Int(IndexExpr),
    Basis(ScaledBasis),
}

impl StrideLeaf {
    /// Whether the leaf is known when the kernel is built: a static integer
    /// or a basis element with a static scale.
    pub fn is_static(&self) -> bool {
        match self {
            StrideLeaf::Int(e) => e.is_static(),
            StrideLeaf::Basis(b) => b.is_static(),
        }
    }

    /// Whether the leaf is a scaled basis element.
    pub fn is_basis(&self) -> bool {
        matches!(self, StrideLeaf::Basis(_))
    }

    /// The integer stride, or `None` for a basis element.
    pub fn as_int(&self) -> Option<&IndexExpr> {
        match self {
            StrideLeaf::Int(e) => Some(e),
            StrideLeaf::Basis(_) => None,
        }
    }

    /// The integer stride of a leaf the caller knows is one.
    pub(crate) fn int(&self) -> &IndexExpr {
        match self {
            StrideLeaf::Int(e) => e,
            StrideLeaf::Basis(b) => unreachable!("the stride {b} was checked to be an integer"),
        }
    }

    /// Whether this is a static integer 0.
    pub(crate) fn is_zero(&self) -> bool {
        matches!(self, StrideLeaf::Int(e) if e.as_static() == Some(0))
    }

    /// `self * e` for the layout algebra: an integer stride is multiplied, a
    /// basis element scaled. `None` for a ratio-valued basis element and a
    /// runtime `e`.
    pub(crate) fn try_scale(&self, e: &IndexExpr) -> Option<StrideLeaf> {
        match self {
            StrideLeaf::Int(d) => Some(StrideLeaf::Int(d.clone() * e)),
            StrideLeaf::Basis(b) => {
                Some(StrideLeaf::Basis(ScaledBasis { value: b.value.times(e)?, mode: b.mode.clone() }))
            }
        }
    }

    /// [`try_scale`](StrideLeaf::try_scale), which panics, naming `what`, where it
    /// has no answer.
    #[track_caller]
    pub(crate) fn scale(&self, e: &IndexExpr, what: &str) -> StrideLeaf {
        match self.try_scale(e) {
            Some(d) => d,
            None => panic!(
                "{what}: the ratio-valued basis stride {self:?} would be scaled by the runtime value {e:?}, which is \
                 not supported"
            ),
        }
    }

    /// Whether the two leaves are provably the same stride.
    pub(crate) fn same_value(&self, other: &StrideLeaf) -> bool {
        match (self, other) {
            (StrideLeaf::Int(a), StrideLeaf::Int(b)) => a.same_value(b),
            (StrideLeaf::Basis(a), StrideLeaf::Basis(b)) => a.mode == b.mode && a.value.same_value(&b.value),
            _ => false,
        }
    }
}

/// `E(mode) * scale`: an integer is multiplied, a basis element scaled. As in
/// CuTe DSL, scaling a basis element whose scale is a [`Ratio`] is not
/// supported, and panics.
impl<R: Into<IndexExpr>> Mul<R> for StrideLeaf {
    type Output = StrideLeaf;
    #[track_caller]
    fn mul(self, rhs: R) -> StrideLeaf {
        let rhs = rhs.into();
        match &self {
            StrideLeaf::Int(d) => StrideLeaf::Int(d.clone() * rhs),
            StrideLeaf::Basis(b) => match &b.value {
                BasisScale::Int(s) => {
                    StrideLeaf::Basis(ScaledBasis { value: BasisScale::Int(rhs * s), mode: b.mode.clone() })
                }
                BasisScale::Ratio(_) => panic!(
                    "cannot scale the basis element {b} by {rhs}: scaling a basis element whose scale is a ratio is \
                     not supported"
                ),
            },
        }
    }
}

impl Mul<StrideLeaf> for IndexExpr {
    type Output = StrideLeaf;
    #[track_caller]
    fn mul(self, rhs: StrideLeaf) -> StrideLeaf {
        rhs * self
    }
}

impl fmt::Display for StrideLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrideLeaf::Int(e) => write!(f, "{e}"),
            StrideLeaf::Basis(b) => write!(f, "{b}"),
        }
    }
}

impl fmt::Debug for StrideLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrideLeaf::Int(e) => write!(f, "{e:?}"),
            StrideLeaf::Basis(b) => write!(f, "{b:?}"),
        }
    }
}

/// A basis element is never a static integer, so it is never a stride of 1
/// to the majorness queries.
impl StaticLeaf for StrideLeaf {
    fn static_value(&self) -> Option<usize> {
        self.as_int().and_then(IndexExpr::as_static)
    }
}

impl From<IndexExpr> for StrideLeaf {
    fn from(e: IndexExpr) -> Self {
        StrideLeaf::Int(e)
    }
}

impl From<&IndexExpr> for StrideLeaf {
    fn from(e: &IndexExpr) -> Self {
        StrideLeaf::Int(e.clone())
    }
}

impl From<ScaledBasis> for StrideLeaf {
    fn from(b: ScaledBasis) -> Self {
        StrideLeaf::Basis(b)
    }
}

/// The unit basis element of `mode` (CuTe DSL's `cute.E(mode)`, C++ CuTe's
/// `E<Ns...>`): `E(0)` is `1@0`, `E([1, 0])` is `1@0@1`, and the empty path
/// `E([])` is the integer 1. Scale it with `*`: `2 * E(0)` is `2@0`, and a
/// runtime [`Int32`](crate::Int32) works too.
// Named `E` as in CuTe.
#[allow(non_snake_case)]
pub fn E(mode: impl BasisMode) -> StrideLeaf {
    let mode = mode.into_mode();
    if mode.is_empty() {
        StrideLeaf::Int(IndexExpr::Static(1))
    } else {
        StrideLeaf::Basis(ScaledBasis { value: BasisScale::Int(IndexExpr::Static(1)), mode })
    }
}

/// The scale of a basis element, or an integer stride itself (CuTe DSL's
/// `cute.basis_value`).
pub fn basis_value(e: &StrideLeaf) -> BasisScale {
    match e {
        StrideLeaf::Int(d) => BasisScale::Int(d.clone()),
        StrideLeaf::Basis(b) => b.value.clone(),
    }
}

/// What [`basis_get`] indexes: tuples and layouts.
pub trait BasisGet: Clone {
    /// The mode at `path`, outermost index first.
    #[track_caller]
    fn basis_mode(&self, path: &[usize]) -> Self;
}

impl<T: Clone> BasisGet for Tree<T> {
    #[track_caller]
    fn basis_mode(&self, path: &[usize]) -> Self {
        self.get(path).clone()
    }
}

impl BasisGet for Layout {
    #[track_caller]
    fn basis_mode(&self, path: &[usize]) -> Self {
        Layout::new(self.shape().get(path).clone(), self.stride().get(path).clone())
    }
}

/// A composed layout's mode is its outer layout's, with the inner and the
/// offset kept.
impl BasisGet for super::ComposedLayout {
    #[track_caller]
    fn basis_mode(&self, path: &[usize]) -> Self {
        self.get(path)
    }
}

/// The mode of `t` that the basis element `e` points at (CuTe DSL's
/// `cute.basis_get`): `basis_get(&E([1, 0]), &t)` is mode 0 of mode 1 of
/// `t`, a tuple or a layout. An integer `e` gives `t` itself.
#[track_caller]
pub fn basis_get<X: BasisGet>(e: &StrideLeaf, t: &X) -> X {
    match e {
        StrideLeaf::Int(_) => t.clone(),
        StrideLeaf::Basis(b) => t.basis_mode(&b.mode),
    }
}

/// Unit basis strides like `shape` (C++ CuTe's `make_basis_like`): the leaf
/// at path `p` gets `E(p)`, so `(2,(3,4))` gets `(1@0,(1@0@1,1@1@1))`. A
/// shape that is a single integer gets the integer stride 1.
pub fn make_basis_like<T>(shape: &Tree<T>) -> Stride {
    fn go<T>(t: &Tree<T>, path: &mut Vec<usize>) -> Stride {
        match t {
            Tree::Leaf(_) => Tree::Leaf(E(path.clone())),
            Tree::Branch(children) => Tree::Branch(
                children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        path.push(i);
                        let s = go(c, path);
                        path.pop();
                        s
                    })
                    .collect(),
            ),
        }
    }
    go(shape, &mut Vec::new())
}

/// The identity layout of `shape` (CuTe DSL's `cute.make_identity_layout`):
/// `shape:make_basis_like(shape)`, which maps every coordinate to itself.
/// `make_identity_layout((2,(3,4),5))` is `(2,(3,4),5):(1@0,(1@0@1,1@1@1),1@2)`,
/// and a shape that is a single integer gets `4:1`. Extents may be runtime
/// values: `(?,3):(1@0,1@1)`.
pub fn make_identity_layout(shape: impl Into<Shape>) -> Layout {
    let shape = shape.into();
    let stride = make_basis_like(&shape);
    Layout::new(shape, stride)
}

/// Add `v` at the mode path `mode` of the arithmetic tuple `acc` (C++ CuTe's
/// sum of an `ArithmeticTuple` and a `ScaledBasis`). A static 0 leaf stands
/// for an empty tuple, so it grows into a tuple where a path goes on through
/// it; a tuple grows with zeros up to the mode the path names. `Err` with a
/// reason where a path ends at a tuple, or goes on through a leaf that is
/// not a static 0.
pub(crate) fn tuple_add(acc: &mut Coord, mode: &[usize], v: IndexExpr) -> Result<(), String> {
    let Some((&i, rest)) = mode.split_first() else {
        return match acc {
            Tree::Leaf(x) => {
                *x = x.clone() + v;
                Ok(())
            }
            Tree::Branch(_) => Err(format!("an integer meets the tuple {acc}")),
        };
    };
    if let Tree::Leaf(x) = acc {
        if x.as_static() != Some(0) {
            return Err(format!("a mode path goes on through the integer {x}"));
        }
        *acc = Tree::Branch(Vec::new());
    }
    let Tree::Branch(children) = acc else { unreachable!("made a tuple above") };
    while children.len() <= i {
        children.push(Tree::Leaf(IndexExpr::Static(0)));
    }
    tuple_add(&mut children[i], rest, v)
}

/// The element-wise sum of two arithmetic tuples (the CuTe dialect's
/// `tuple_add`), recursing into nested tuples: where one tuple is longer, its
/// extra modes pass through, and a static 0 leaf is the empty tuple, so `0 +
/// t` is `t`. A leaf other than 0 meeting a tuple panics.
#[track_caller]
pub(crate) fn tuple_sum(a: &Coord, b: &Coord) -> Coord {
    match (a, b) {
        (Tree::Leaf(x), _) if x.as_static() == Some(0) => b.clone(),
        (_, Tree::Leaf(y)) if y.as_static() == Some(0) => a.clone(),
        (Tree::Leaf(x), Tree::Leaf(y)) => Tree::Leaf(x.clone() + y.clone()),
        (Tree::Branch(xs), Tree::Branch(ys)) => Tree::Branch(
            (0..xs.len().max(ys.len()))
                .map(|i| match (xs.get(i), ys.get(i)) {
                    (Some(x), Some(y)) => tuple_sum(x, y),
                    (Some(t), None) | (None, Some(t)) => t.clone(),
                    (None, None) => unreachable!("i is below one of the lengths"),
                })
                .collect(),
        ),
        _ => panic!("tuple_add: cannot add {a} and {b}: an integer other than 0 meets a tuple"),
    }
}

/// `c * scale`, the contribution of a basis element at coordinate `c`. A
/// ratio scale needs a static product that is a whole number.
pub(crate) fn scale_coord(c: &IndexExpr, scale: &BasisScale) -> Result<IndexExpr, String> {
    match scale {
        BasisScale::Int(s) => Ok(IndexExpr::product(c.clone(), s.clone())),
        BasisScale::Ratio(r) => match BasisScale::Ratio(*r).times(c) {
            Some(BasisScale::Int(v)) => Ok(v),
            Some(BasisScale::Ratio(p)) => Err(format!("{c} * {r} is {p}, not a whole number")),
            None => Err(format!("the ratio-valued scale {r} needs a static coordinate, not {c:?}")),
        },
    }
}

impl Tree<StrideLeaf> {
    /// The strides as static integers, or `None` if a leaf is a runtime
    /// value or a basis element.
    pub fn as_static(&self) -> Option<IntTuple> {
        match self {
            Tree::Leaf(d) => d.static_value().map(Tree::Leaf),
            Tree::Branch(children) => {
                children.iter().map(Tree::<StrideLeaf>::as_static).collect::<Option<Vec<_>>>().map(Tree::Branch)
            }
        }
    }

    /// Whether every leaf is known when the kernel is built: a static
    /// integer or a basis element with a static scale (CuTe DSL's
    /// `is_static`).
    pub fn is_static(&self) -> bool {
        !self.any_leaf(&mut |d| !d.is_static())
    }

    /// Whether a leaf is a scaled basis element (CuTe DSL's
    /// `has_scaled_basis`).
    pub fn has_scaled_basis(&self) -> bool {
        self.any_leaf(&mut StrideLeaf::is_basis)
    }

    /// The strides as integers, or `None` if a leaf is a basis element.
    pub fn as_int(&self) -> Option<Shape> {
        match self {
            Tree::Leaf(d) => d.as_int().cloned().map(Tree::Leaf),
            Tree::Branch(children) => {
                children.iter().map(Tree::<StrideLeaf>::as_int).collect::<Option<Vec<_>>>().map(Tree::Branch)
            }
        }
    }

    /// The first leaf that is not static, depth first, and its path.
    pub(crate) fn first_dynamic(&self) -> Option<(Vec<usize>, &StrideLeaf)> {
        let path = self.find_if(|d, _| !d.is_static())?;
        let leaf = self.get(&path).as_leaf().expect("find_if returns the path of a leaf");
        Some((path, leaf))
    }

    /// The first basis element, depth first, and its path.
    pub(crate) fn first_basis(&self) -> Option<(Vec<usize>, &StrideLeaf)> {
        // Checked first without building a path: the offset of every tensor
        // access asks this of an integer layout.
        if !self.has_scaled_basis() {
            return None;
        }
        let path = self.find_if(|d, _| d.is_basis())?;
        let leaf = self.get(&path).as_leaf().expect("find_if returns the path of a leaf");
        Some((path, leaf))
    }
}

impl From<StrideLeaf> for Stride {
    fn from(d: StrideLeaf) -> Self {
        Tree::Leaf(d)
    }
}

impl From<ScaledBasis> for Stride {
    fn from(b: ScaledBasis) -> Self {
        Tree::Leaf(StrideLeaf::Basis(b))
    }
}

impl From<IndexExpr> for Stride {
    fn from(e: IndexExpr) -> Self {
        Tree::Leaf(StrideLeaf::Int(e))
    }
}

/// Integer strides, static or runtime, from a [`Shape`] or [`Coord`]: what
/// [`Shape::compact_col_major`] returns, say.
impl From<&Shape> for Stride {
    fn from(t: &Shape) -> Self {
        t.map(&mut |e| StrideLeaf::Int(e.clone()))
    }
}

impl From<Shape> for Stride {
    fn from(t: Shape) -> Self {
        Stride::from(&t)
    }
}

impl From<&IntTuple> for Stride {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&v| StrideLeaf::Int(IndexExpr::Static(v)))
    }
}

impl From<IntTuple> for Stride {
    fn from(t: IntTuple) -> Self {
        Stride::from(&t)
    }
}

impl From<&Stride> for Stride {
    fn from(t: &Stride) -> Self {
        t.clone()
    }
}

impl<T: Into<Stride>> From<Vec<T>> for Stride {
    fn from(v: Vec<T>) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Stride>, const N: usize> From<[T; N]> for Stride {
    fn from(v: [T; N]) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

tree_from_tuple!(Stride;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));

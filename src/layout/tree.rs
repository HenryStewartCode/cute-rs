// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/int_tuple.hpp, include/cute/stride.hpp and
//   include/cute/algorithm/tuple_algorithms.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Hierarchical tuples.
//!
//! CuTe describes shapes, strides and coordinates as nested tuples of
//! integers (`IntTuple`). [`Tree`] is that nesting, generic over the leaf so
//! the same structure can hold static integers ([`IntTuple`]), symbolic index
//! expressions ([`Coord`](super::Coord)) or slice specifiers
//! ([`SliceCoord`](super::SliceCoord)).

use std::fmt;

/// A leaf value or a tuple of sub-trees.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Tree<T> {
    Leaf(T),
    Branch(Vec<Tree<T>>),
}

/// A (possibly nested) tuple of non-negative integers: CuTe's `IntTuple`.
pub type IntTuple = Tree<usize>;

impl<T> Tree<T> {
    /// A tuple of `children`.
    pub fn tuple(children: impl IntoIterator<Item = Tree<T>>) -> Self {
        Tree::Branch(children.into_iter().collect())
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Tree::Leaf(_))
    }

    pub fn as_leaf(&self) -> Option<&T> {
        match self {
            Tree::Leaf(v) => Some(v),
            Tree::Branch(_) => None,
        }
    }

    /// Number of top-level modes. A leaf has rank 1.
    pub fn rank(&self) -> usize {
        match self {
            Tree::Leaf(_) => 1,
            Tree::Branch(children) => children.len(),
        }
    }

    /// Nesting depth. A leaf has depth 0, a flat tuple depth 1.
    pub fn depth(&self) -> usize {
        match self {
            Tree::Leaf(_) => 0,
            Tree::Branch(children) => 1 + children.iter().map(Tree::depth).max().unwrap_or(0),
        }
    }

    /// The `i`-th top-level mode. Mode 0 of a leaf is the leaf itself.
    pub fn mode(&self, i: usize) -> &Tree<T> {
        match self {
            Tree::Leaf(_) => {
                assert_eq!(i, 0, "mode {i} of a rank-1 leaf");
                self
            }
            Tree::Branch(children) => {
                children.get(i).unwrap_or_else(|| panic!("mode {i} out of range for a rank-{} tuple", children.len()))
            }
        }
    }

    /// All top-level modes (a leaf is its own single mode).
    pub fn modes(&self) -> &[Tree<T>] {
        match self {
            Tree::Leaf(_) => std::slice::from_ref(self),
            Tree::Branch(children) => children,
        }
    }

    /// The leaves in left-to-right order.
    pub fn leaves(&self) -> Vec<&T> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a T>) {
        match self {
            Tree::Leaf(v) => out.push(v),
            Tree::Branch(children) => children.iter().for_each(|c| c.collect_leaves(out)),
        }
    }

    /// Whether `pred` holds for some leaf, without collecting the leaves.
    pub fn any_leaf(&self, pred: &mut impl FnMut(&T) -> bool) -> bool {
        match self {
            Tree::Leaf(v) => pred(v),
            Tree::Branch(children) => children.iter().any(|c| c.any_leaf(pred)),
        }
    }

    pub fn map<U>(&self, f: &mut impl FnMut(&T) -> U) -> Tree<U> {
        match self {
            Tree::Leaf(v) => Tree::Leaf(f(v)),
            Tree::Branch(children) => Tree::Branch(children.iter().map(|c| c.map(f)).collect()),
        }
    }

    /// Same nesting structure (leaf positions) as `other`.
    pub fn congruent<U>(&self, other: &Tree<U>) -> bool {
        match (self, other) {
            (Tree::Leaf(_), Tree::Leaf(_)) => true,
            (Tree::Branch(a), Tree::Branch(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.congruent(y)),
            _ => false,
        }
    }

    /// Whether this nesting is a coarsening of `other`'s (CuTe's
    /// `weakly_congruent`, a partial order): `self` is a leaf, or both are
    /// tuples of the same rank whose modes are weakly congruent in turn. A
    /// leaf is weakly congruent to anything, `(2,2)` to `((2,2),4)`, but
    /// `(2,2)` not to a leaf.
    pub fn weakly_congruent<U>(&self, other: &Tree<U>) -> bool {
        match (self, other) {
            (Tree::Leaf(_), _) => true,
            (Tree::Branch(a), Tree::Branch(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.weakly_congruent(y))
            }
            (Tree::Branch(_), Tree::Leaf(_)) => false,
        }
    }

    /// The sub-tree at `path`, one mode index per level (CuTe's
    /// `get(t, mode=path)`): `get(&[1, 0])` is mode 0 of mode 1, and the
    /// empty path is the whole tree. As with [`mode`](Tree::mode), index 0
    /// of a leaf is the leaf itself. Panics if the path leaves the tree.
    #[track_caller]
    pub fn get(&self, path: &[usize]) -> &Tree<T> {
        let mut t = self;
        for (depth, &i) in path.iter().enumerate() {
            t = match t {
                Tree::Leaf(_) if i == 0 => t,
                Tree::Branch(children) if i < children.len() => &children[i],
                _ => {
                    let at =
                        if depth == 0 { "the tuple".to_string() } else { format!("its mode {:?}", &path[..depth]) };
                    let what = if t.is_leaf() { "is a leaf".to_string() } else { format!("has rank {}", t.rank()) };
                    panic!("get: path {path:?} leaves the tree: {at} {what}, so it has no mode {i}")
                }
            };
        }
        t
    }

    /// The first leaf (CuTe's `front`): mode 0, then its mode 0, and so on
    /// until a leaf is left. Panics if that reaches an empty tuple.
    #[track_caller]
    pub fn front(&self) -> &T {
        match self.first_leaf() {
            Ok(v) => v,
            Err(0) => panic!("front: the empty tuple () has no first leaf"),
            Err(depth) => {
                panic!("front: the tuple has no first leaf: its mode {:?} is the empty tuple ()", vec![0; depth])
            }
        }
    }

    /// [`front`](Tree::front), or the depth of the empty tuple that
    /// following mode 0 down reaches instead of a leaf.
    pub(crate) fn first_leaf(&self) -> Result<&T, usize> {
        match self {
            Tree::Leaf(v) => Ok(v),
            Tree::Branch(children) => match children.first() {
                Some(first) => first.first_leaf().map_err(|depth| depth + 1),
                None => Err(0),
            },
        }
    }

    /// CuTe's `wrap`: a tuple stays as it is, a leaf becomes a 1-tuple.
    pub fn wrap(self) -> Tree<T> {
        match self {
            Tree::Branch(_) => self,
            leaf => Tree::Branch(vec![leaf]),
        }
    }

    /// CuTe's `unwrap`: strip 1-tuples until something else is left, so
    /// `((4,6,8))` becomes `(4,6,8)` and `((1))` becomes `1`.
    pub fn unwrap(self) -> Tree<T> {
        let mut tree = self;
        loop {
            match tree {
                Tree::Branch(mut children) if children.len() == 1 => tree = children.pop().expect("one child"),
                other => return other,
            }
        }
    }

    /// The position of the first leaf, depth first, for which
    /// `pred(leaf, path)` holds (CuTe DSL's `find_if`). The path holds one
    /// mode index per level, as [`get`](Tree::get) takes it: `[i]` for a leaf
    /// at the top level (CuTe DSL's plain `i`), `[i, j]` for mode `j` of mode
    /// `i`, and `[0]` when the whole tree is a leaf. `None` if no leaf matches.
    pub fn find_if(&self, mut pred: impl FnMut(&T, &[usize]) -> bool) -> Option<Vec<usize>> {
        fn search<T>(t: &Tree<T>, path: &mut Vec<usize>, pred: &mut impl FnMut(&T, &[usize]) -> bool) -> bool {
            match t {
                Tree::Leaf(v) => pred(v, path),
                Tree::Branch(children) => children.iter().enumerate().any(|(i, c)| {
                    path.push(i);
                    let found = search(c, path, pred);
                    if !found {
                        path.pop();
                    }
                    found
                }),
            }
        }
        let mut path = if self.is_leaf() { vec![0] } else { Vec::new() };
        search(self, &mut path, &mut pred).then_some(path)
    }

    /// The top-level mode holding the first leaf for which `pred` holds:
    /// [`find_if`](Tree::find_if) with `hierarchical=False`.
    pub fn find_mode_if(&self, pred: impl FnMut(&T, &[usize]) -> bool) -> Option<usize> {
        self.find_if(pred).map(|path| path[0])
    }
}

/// A leaf that may be a static integer: shape and static stride leaves are,
/// runtime values and the `..` of a slice coordinate are not. This is what
/// [`Tree::find`] and the majorness queries ([`leading_dim`](super::leading_dim),
/// [`is_major`](super::is_major)) compare.
pub trait StaticLeaf {
    /// The leaf's value, if it is a static integer.
    fn static_value(&self) -> Option<usize>;
}

impl StaticLeaf for usize {
    fn static_value(&self) -> Option<usize> {
        Some(*self)
    }
}

impl<T: StaticLeaf> StaticLeaf for Option<T> {
    fn static_value(&self) -> Option<usize> {
        self.as_ref().and_then(T::static_value)
    }
}

impl<T: StaticLeaf> Tree<T> {
    /// The position of the first leaf that is the static integer `x`
    /// (CuTe DSL's `find`). Runtime leaves are skipped. See
    /// [`find_if`](Tree::find_if) for the form of the position.
    pub fn find(&self, x: usize) -> Option<Vec<usize>> {
        self.find_if(|leaf, _| leaf.static_value() == Some(x))
    }

    /// The top-level mode holding the first leaf that is the static integer
    /// `x`: [`find`](Tree::find) with `hierarchical=False`.
    pub fn find_mode(&self, x: usize) -> Option<usize> {
        self.find(x).map(|path| path[0])
    }
}

/// Anything with a hierarchical shape, as CuTe's `shape(x)` sees it: a tuple
/// is its own shape, and layouts, swizzled layouts and tensors have one, as
/// do references and smart pointers to them.
/// [`is_congruent`](super::is_congruent) and
/// [`is_weakly_congruent`](super::is_weakly_congruent) take any of them.
pub trait HasShape {
    /// The leaf type of the shape.
    type Leaf;

    /// The shape.
    fn shape(&self) -> &Tree<Self::Leaf>;
}

impl<T> HasShape for Tree<T> {
    type Leaf = T;

    fn shape(&self) -> &Tree<T> {
        self
    }
}

/// A reference or smart pointer has the shape of what it points to: anything
/// that dereferences to a type with a shape (`&T`, `&mut T`, `Box`, `Rc`,
/// `Ref`, `Cow`, `dyn Deref<Target = Layout>`, ...) has that shape. So
/// `is_congruent(&a, &b)` takes `a: &Layout` or `&tensor.layout()`, as it
/// did through deref coercion when it took `&Layout`.
impl<P> HasShape for P
where
    P: std::ops::Deref + ?Sized,
    P::Target: HasShape,
{
    type Leaf = <P::Target as HasShape>::Leaf;

    fn shape(&self) -> &Tree<Self::Leaf> {
        (**self).shape()
    }
}

impl<T: Clone> Tree<T> {
    /// Leaves as owned values.
    pub fn flat_leaves(&self) -> Vec<T> {
        self.leaves().into_iter().cloned().collect()
    }

    /// Remove all nesting: a leaf stays a leaf, a tuple becomes a flat tuple.
    pub fn flatten(&self) -> Tree<T> {
        match self {
            Tree::Leaf(_) => self.clone(),
            Tree::Branch(_) => Tree::Branch(self.flat_leaves().into_iter().map(Tree::Leaf).collect()),
        }
    }

    /// Rebuild `values` (taken in order) into the nesting of `profile`.
    pub fn unflatten<U>(values: &[T], profile: &Tree<U>) -> Tree<T> {
        let mut it = values.iter().cloned();
        let tree = Self::unflatten_from(&mut it, profile);
        assert!(it.next().is_none(), "unflatten: too many values for the profile");
        tree
    }

    fn unflatten_from<U>(it: &mut impl Iterator<Item = T>, profile: &Tree<U>) -> Tree<T> {
        match profile {
            Tree::Leaf(_) => Tree::Leaf(it.next().expect("unflatten: too few values for the profile")),
            Tree::Branch(children) => Tree::Branch(children.iter().map(|c| Self::unflatten_from(it, c)).collect()),
        }
    }
}

impl IntTuple {
    /// Product of all leaves.
    pub fn size(&self) -> usize {
        self.leaves().into_iter().product()
    }

    /// The size of each top-level mode, as a flat tuple (a leaf stays a leaf).
    pub fn product_each(&self) -> IntTuple {
        match self {
            Tree::Leaf(_) => self.clone(),
            Tree::Branch(children) => Tree::Branch(children.iter().map(|c| Tree::Leaf(c.size())).collect()),
        }
    }

    /// Compact column-major (CuTe `LayoutLeft`) strides with the same nesting:
    /// the exclusive prefix product of the leaves.
    pub fn compact_col_major(&self) -> IntTuple {
        let mut running = 1;
        self.map(&mut |&s| {
            let stride = running;
            running *= s;
            stride
        })
    }

    /// Compact row-major (CuTe `LayoutRight`) strides with the same nesting:
    /// the exclusive suffix product of the leaves.
    pub fn compact_row_major(&self) -> IntTuple {
        let shapes = self.flat_leaves();
        let mut strides = vec![0; shapes.len()];
        let mut running = 1;
        for i in (0..shapes.len()).rev() {
            strides[i] = running;
            running *= shapes[i];
        }
        Tree::unflatten(&strides, self)
    }
}

/// CuTe's `shape_div` of two static integers: `a / b` when `b` divides
/// `a`, otherwise 1 when `a` divides `b`. Panics when neither divides the
/// other. [`shape_div`](super::shape_div) takes tuples and runtime leaves.
pub(crate) fn shape_div(a: usize, b: usize) -> usize {
    assert!(b != 0, "shape_div by zero");
    if a.is_multiple_of(b) {
        a / b
    } else {
        assert!(b.is_multiple_of(a), "shape_div: neither {a} nor {b} divides the other");
        1
    }
}

impl<T: fmt::Display> fmt::Display for Tree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tree::Leaf(v) => write!(f, "{v}"),
            Tree::Branch(children) => {
                write!(f, "(")?;
                for (i, c) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{c}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Tree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tree::Leaf(v) => write!(f, "{v:?}"),
            Tree::Branch(children) => {
                write!(f, "(")?;
                for (i, c) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{c:?}")?;
                }
                write!(f, ")")
            }
        }
    }
}

macro_rules! int_tuple_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for IntTuple {
            fn from(v: $t) -> Self {
                Tree::Leaf(usize::try_from(v).unwrap_or_else(|_| panic!("negative or oversized extent {v}")))
            }
        }
    )*};
}
int_tuple_from_int!(usize, u8, u16, u32, u64, i32, i64);

macro_rules! tree_from_tuple {
    ($target:ty; $(($($name:ident),+)),+) => {$(
        #[allow(non_snake_case)]
        impl<$($name: Into<$target>),+> From<($($name,)+)> for $target {
            fn from(($($name,)+): ($($name,)+)) -> Self {
                Tree::Branch(vec![$($name.into()),+])
            }
        }
    )+};
}
pub(crate) use tree_from_tuple;

tree_from_tuple!(IntTuple;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));

impl From<&IntTuple> for IntTuple {
    fn from(t: &IntTuple) -> Self {
        t.clone()
    }
}

impl<T: Into<IntTuple>> From<Vec<T>> for IntTuple {
    fn from(v: Vec<T>) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<IntTuple>, const N: usize> From<[T; N]> for IntTuple {
    fn from(v: [T; N]) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let t: IntTuple = ((2, 3), 4).into();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.depth(), 2);
        assert_eq!(t.size(), 24);
        assert_eq!(t.to_string(), "((2,3),4)");
        assert_eq!(t.flatten().to_string(), "(2,3,4)");
        assert_eq!(t.product_each().to_string(), "(6,4)");
        assert_eq!(t.compact_col_major().to_string(), "((1,2),6)");
        assert_eq!(t.compact_row_major().to_string(), "((12,4),1)");
    }

    #[test]
    fn shape_division() {
        assert_eq!(shape_div(12, 4), 3);
        assert_eq!(shape_div(4, 12), 1);
    }
}

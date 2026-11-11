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

/// CuTe's `shape_div`: `a / b` when `b` divides `a`, otherwise 1 when `a`
/// divides `b`. Panics when neither divides the other.
pub fn shape_div(a: usize, b: usize) -> usize {
    assert!(b != 0, "shape_div by zero");
    if a.is_multiple_of(b) {
        a / b
    } else {
        assert!(b.is_multiple_of(a), "shape_div: neither {a} nor {b} divides the other");
        1
    }
}

/// `a / b` rounded up.
pub fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
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
        assert_eq!(ceil_div(10, 4), 3);
    }
}

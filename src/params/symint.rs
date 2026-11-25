//! Symbolic integers ([`SymInt`]) and the leaves of parameter specs
//! ([`SpecLeaf`], [`SpecTuple`]).

use std::collections::HashSet;
use std::fmt;
use std::ops::{Div, Mul, Rem};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ir::DType;
use crate::layout::{IntTuple, Tree};

static NEXT_SYMINT: AtomicU64 = AtomicU64::new(1);
static SYMBOLS: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);

/// A symbol kept for the life of the program (each distinct one once).
fn intern(s: &str) -> &'static str {
    let mut set = SYMBOLS.lock().unwrap_or_else(|e| e.into_inner());
    let set = set.get_or_insert_with(HashSet::new);
    if let Some(&s) = set.get(s) {
        return s;
    }
    let s: &'static str = Box::leak(s.to_string().into_boxed_str());
    set.insert(s);
    s
}

/// A symbolic integer: an extent, stride or scalar that is only known at
/// launch (CuTe DSL's `cute.SymInt`), with its width (32 or 64 bits), the
/// number its value is promised to be a multiple of (its divisibility) and
/// an optional symbol.
///
/// A `SymInt` has an identity: every place one value is used (several
/// tensors' extents, a stride, a scalar argument) stands for one runtime
/// value, declared as one kernel parameter, and launches check that every
/// occurrence was given the same value and that it meets the divisibility.
/// Two `SymInt`s made separately are different values even when their
/// width, divisibility and symbol agree (`==` compares those three, as
/// CuTe DSL's `__eq__` does; [`SymInt::same`] compares identities).
#[derive(Clone, Copy, Debug)]
pub struct SymInt {
    id: u64,
    width: u32,
    divisibility: usize,
    symbol: Option<&'static str>,
}

/// A fresh [`SymInt`] of `width` bits (32 or 64) whose value is a multiple
/// of `divisibility` (CuTe DSL's `cute.sym_int`).
pub fn sym_int(width: u32, divisibility: usize, symbol: Option<&str>) -> SymInt {
    SymInt::new(width, divisibility, symbol)
}

/// A fresh 32-bit [`SymInt`] (CuTe DSL's `cute.sym_int32`).
pub fn sym_int32(divisibility: usize, symbol: Option<&str>) -> SymInt {
    SymInt::new(32, divisibility, symbol)
}

/// A fresh 64-bit [`SymInt`] (CuTe DSL's `cute.sym_int64`).
pub fn sym_int64(divisibility: usize, symbol: Option<&str>) -> SymInt {
    SymInt::new(64, divisibility, symbol)
}

impl SymInt {
    /// A fresh `SymInt`. Panics unless `width` is 32 or 64 and
    /// `divisibility` is positive.
    pub fn new(width: u32, divisibility: usize, symbol: Option<&str>) -> SymInt {
        assert!(width == 32 || width == 64, "cute: SymInt: unsupported width {width} (32 or 64)");
        assert!(divisibility > 0, "cute: SymInt: the divisibility must be positive");
        SymInt { id: NEXT_SYMINT.fetch_add(1, Ordering::Relaxed), width, divisibility, symbol: symbol.map(intern) }
    }

    /// The identity shared by the copies of this `SymInt`.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether `other` is this `SymInt` (a copy of it), not merely one with
    /// the same width, divisibility and symbol.
    pub fn same(&self, other: &SymInt) -> bool {
        self.id == other.id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn divisibility(&self) -> usize {
        self.divisibility
    }

    pub fn symbol(&self) -> Option<&'static str> {
        self.symbol
    }

    /// The parameter type: `Int32` or `Int64`.
    pub fn dtype(&self) -> DType {
        if self.width == 64 { DType::I64 } else { DType::I32 }
    }

    /// The largest value the width holds.
    pub(crate) fn max_value(&self) -> u64 {
        if self.width == 64 { i64::MAX as u64 } else { i32::MAX as u64 }
    }

    /// How the leaf prints inside a layout: `?`, `?{div=4}`, `?{i64}`,
    /// `?{i64 div=4}`.
    pub(crate) fn leaf_text(&self) -> String {
        match (self.width == 64, self.divisibility) {
            (false, 1) => "?".into(),
            (false, d) => format!("?{{div={d}}}"),
            (true, 1) => "?{i64}".into(),
            (true, d) => format!("?{{i64 div={d}}}"),
        }
    }
}

/// The identity of a [`SymInt`] ([`SymInt::cache_key`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymIntKey(u64);

impl SymInt {
    /// The key to cache compiled kernels on: the identity, not the value (two `SymInt`s of
    /// equal width and divisibility key differently).
    pub fn cache_key(&self) -> SymIntKey {
        SymIntKey(self.id)
    }
}

/// Hashes what `==` compares (width, divisibility and symbol); [`SymInt::cache_key`] is the
/// identity.
impl std::hash::Hash for SymInt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.width, self.divisibility, self.symbol).hash(state);
    }
}

/// Width, divisibility and symbol, as CuTe DSL compares them.
impl PartialEq for SymInt {
    fn eq(&self, other: &SymInt) -> bool {
        (self.width, self.divisibility, self.symbol) == (other.width, other.divisibility, other.symbol)
    }
}

impl Eq for SymInt {}

/// `?{div=N}` for 32 bits, `?{i64 div=N}` for 64, after the symbol if it
/// has one (`M ?{i64 div=16}`).
impl fmt::Display for SymInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(s) = self.symbol {
            write!(f, "{s} ")?;
        }
        if self.width == 32 {
            write!(f, "?{{div={}}}", self.divisibility)
        } else {
            write!(f, "?{{i{} div={}}}", self.width, self.divisibility)
        }
    }
}

/// `k * n`: a fresh `SymInt` whose divisibility is multiplied by `n`.
impl Mul<usize> for SymInt {
    type Output = SymInt;
    fn mul(self, n: usize) -> SymInt {
        SymInt::new(self.width, self.divisibility * n.max(1), None)
    }
}

impl Mul<SymInt> for usize {
    type Output = SymInt;
    fn mul(self, s: SymInt) -> SymInt {
        s * self
    }
}

/// The product of two symbolic values: the wider width, the product of the
/// divisibilities.
impl Mul for SymInt {
    type Output = SymInt;
    fn mul(self, other: SymInt) -> SymInt {
        SymInt::new(self.width.max(other.width), self.divisibility * other.divisibility, None)
    }
}

/// What `SymInt % n` gives: 0 when the divisibility is a multiple of `n`,
/// otherwise a symbolic value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymRem {
    Zero,
    Sym(SymInt),
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl SymInt {
    fn rem_by(self, d: usize, width: u32) -> SymRem {
        assert!(d != 0, "cute: SymInt: integer modulo by zero");
        if self.divisibility.is_multiple_of(d) {
            SymRem::Zero
        } else {
            SymRem::Sym(SymInt::new(width, gcd(self.divisibility, d), None))
        }
    }
}

/// `s % n`: 0 when `n` divides the divisibility, else a `SymInt` whose
/// divisibility is their gcd.
impl Rem<usize> for SymInt {
    type Output = SymRem;
    fn rem(self, n: usize) -> SymRem {
        self.rem_by(n, self.width)
    }
}

/// `s % t`, by `t`'s divisibility, with the wider width.
impl Rem for SymInt {
    type Output = SymRem;
    fn rem(self, t: SymInt) -> SymRem {
        self.rem_by(t.divisibility, self.width.max(t.width))
    }
}

/// `n % s`: whether `n` conforms to `s`'s divisibility (`n % divisibility`).
impl Rem<SymInt> for usize {
    type Output = usize;
    fn rem(self, s: SymInt) -> usize {
        self % s.divisibility
    }
}

/// `s / n` (floor division): the divisibility divided by `n` when `n`
/// divides it, else 1.
impl Div<usize> for SymInt {
    type Output = SymInt;
    fn div(self, n: usize) -> SymInt {
        assert!(n != 0, "cute: SymInt: integer division by zero");
        let d = if self.divisibility.is_multiple_of(n) { self.divisibility / n } else { 1 };
        SymInt::new(self.width, d, None)
    }
}

/// `s / t`: nothing is known of the quotient (divisibility 1).
impl Div for SymInt {
    type Output = SymInt;
    fn div(self, t: SymInt) -> SymInt {
        SymInt::new(self.width.max(t.width), 1, None)
    }
}

/// A leaf of a parameter's shape, stride or tuple: a static integer, or a
/// [`SymInt`] known at launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecLeaf {
    Static(usize),
    Dynamic(SymInt),
}

impl SpecLeaf {
    pub fn as_static(&self) -> Option<usize> {
        match self {
            SpecLeaf::Static(v) => Some(*v),
            SpecLeaf::Dynamic(_) => None,
        }
    }

    pub fn as_sym(&self) -> Option<SymInt> {
        match self {
            SpecLeaf::Static(_) => None,
            SpecLeaf::Dynamic(s) => Some(*s),
        }
    }

    /// The largest number known to divide the leaf.
    pub fn divisibility(&self) -> usize {
        match self {
            SpecLeaf::Static(v) => *v,
            SpecLeaf::Dynamic(s) => s.divisibility,
        }
    }
}

impl fmt::Display for SpecLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecLeaf::Static(v) => write!(f, "{v}"),
            SpecLeaf::Dynamic(s) => f.write_str(&s.leaf_text()),
        }
    }
}

/// A (possibly nested) tuple of [`SpecLeaf`]s: the shape or stride of a
/// [`TensorSpec`](crate::TensorSpec) or a layout parameter, with static and
/// dynamic leaves. Converts from integers, `SymInt`s and tuples of them:
/// `(sym_int32(8, None), 4).into()`.
pub type SpecTuple = Tree<SpecLeaf>;

macro_rules! spec_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for SpecLeaf {
            fn from(v: $t) -> Self {
                SpecLeaf::Static(usize::try_from(v).unwrap_or_else(|_| panic!("cute: negative or oversized extent {v}")))
            }
        }
        impl From<$t> for SpecTuple {
            fn from(v: $t) -> Self {
                Tree::Leaf(v.into())
            }
        }
    )*};
}
spec_from_int!(usize, u32, u64, i32, i64);

impl From<SymInt> for SpecLeaf {
    fn from(s: SymInt) -> Self {
        SpecLeaf::Dynamic(s)
    }
}

impl From<SymInt> for SpecTuple {
    fn from(s: SymInt) -> Self {
        Tree::Leaf(SpecLeaf::Dynamic(s))
    }
}

impl From<SpecLeaf> for SpecTuple {
    fn from(l: SpecLeaf) -> Self {
        Tree::Leaf(l)
    }
}

impl From<&IntTuple> for SpecTuple {
    fn from(t: &IntTuple) -> Self {
        t.map(&mut |&v| SpecLeaf::Static(v))
    }
}

impl From<&SpecTuple> for SpecTuple {
    fn from(t: &SpecTuple) -> Self {
        t.clone()
    }
}

impl<T: Into<SpecTuple>> From<Vec<T>> for SpecTuple {
    fn from(v: Vec<T>) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<SpecTuple>, const N: usize> From<[T; N]> for SpecTuple {
    fn from(v: [T; N]) -> Self {
        Tree::Branch(v.into_iter().map(Into::into).collect())
    }
}

crate::layout::tree_from_tuple!(SpecTuple;
    (A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F),
    (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H));

/// A tuple as Python prints it: a one-element tuple is `(16,)`.
pub(crate) fn python_tuple(t: &SpecTuple) -> String {
    match t {
        Tree::Leaf(l) => l.to_string(),
        Tree::Branch(c) if c.len() == 1 => format!("({},)", python_tuple(&c[0])),
        Tree::Branch(c) => format!("({})", c.iter().map(python_tuple).collect::<Vec<_>>().join(",")),
    }
}

/// Every leaf of `t` with its path (outermost mode first).
pub(crate) fn leaves_with_paths<T>(t: &Tree<T>) -> Vec<(Vec<usize>, &T)> {
    fn walk<'a, T>(t: &'a Tree<T>, path: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, &'a T)>) {
        match t {
            Tree::Leaf(l) => out.push((path.clone(), l)),
            Tree::Branch(c) => {
                for (i, c) in c.iter().enumerate() {
                    path.push(i);
                    walk(c, path, out);
                    path.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(t, &mut Vec::new(), &mut out);
    out
}

/// The name of the leaf at `path` under `prefix`: `a_shape_0`, `a_stride_1_0`
/// (`prefix` itself for a leaf at the root).
pub(crate) fn leaf_param_name(prefix: &str, path: &[usize]) -> String {
    let mut s = prefix.to_string();
    for i in path {
        s += &format!("_{i}");
    }
    s
}

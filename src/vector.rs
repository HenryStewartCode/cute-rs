//! [`Vector`]: CuTe DSL's `cutlass.Vector[T, N]`, an immutable register
//! vector, and the functions of `cutlass.vector` (`cute::vector::full` and
//! friends, kept in this module so that they do not collide with
//! [`TensorSsa`]'s [`full`](crate::full)).
//!
//! A `Vector<T>` is a row-major n-D shape, fixed when the kernel is built,
//! and one runtime value of type `T` per element, in row-major order (the
//! last index fastest), as an MLIR `vector<2x4xf32>` holds them. It is the
//! same scalarised machinery as [`TensorSsa`] (one SSA value per element,
//! no vector type in the IR) under another face:
//!
//! * `+ - * / %`, unary `-`, `& | ^ ! << >>` with another vector, a
//!   [`Value`] or a host number, with [`Value`]'s promotion;
//!   [`lt`](Vector::lt) and the other comparisons; [`where_`], [`any_`],
//!   [`all_`]; [`reduce`](Vector::reduce); [`to`](Vector::to) and
//!   [`bitcast`](Vector::bitcast);
//! * [`get`](Vector::get) reads an element (at a runtime index too) and
//!   [`slice`](Vector::slice) takes a sub-vector with Rust ranges
//!   (`v.slice((.., 1..3))`, NumPy's `v[:, 1:3]`). There is no element
//!   assignment: the vector is immutable, as in CuTe DSL, where `vec[i] = x`
//!   raises a `TypeError`;
//! * [`TensorSsa::from_vector`] and [`TensorSsa::to_vector`] convert to and
//!   from a column-major [`TensorSsa`], permuting the elements.
//!
//! # Broadcasting
//!
//! Two operands of different shapes broadcast as NumPy's do (and MLIR's
//! `vector.broadcast`): the shapes are aligned **on the right**, the shorter
//! one padded on the left with 1s, and each pair of extents must be equal or
//! include a 1. That is the opposite of [`TensorSsa`]'s column-major rule:
//! a `(4,)` vector meets a `(3, 4)` one as `(1, 4)`, a row.
//!
//! ```
//! use cute::prelude::*;
//! Kernel::build("vector_rows", || {
//!     let out = param_tensor::<f32>("out", Layout::col_major(8));
//!     let row = Vector::<f32>::from_elements([1.0, 2.0, 3.0, 4.0]);
//!     let m = vector::broadcast_to(&row, (2, 4)) * 2.0;
//!     let sub = m.slice((.., 1..3)); // [[4, 6], [4, 6]]
//!     out.store(0, sub.get((0, 0)));
//!     out.store(1, m.reduce(ReductionOp::Add));
//! });
//! ```

use std::fmt;
use std::ops::{
    Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive, Rem, Shl, Shr, Sub,
};
use std::rc::Rc;

use crate::builder::LocationScope;
use crate::control::Carry;
use crate::elementwise;
use crate::ir::{CmpPred, DType, ValueId};
use crate::promote::CommonType;
use crate::tensor_ssa::{FillValue, ReductionOp, TensorSsa};
use crate::value::{Arith, BF16, Bitwise, Boolean, F16, Float, Integer, IntoValue, Numeric, Value, compare_values};

/// CuTe DSL's `cutlass.Vector`: a row-major n-D shape and one runtime value
/// per element, in row-major order. See the [module documentation](self).
pub struct Vector<T> {
    shape: Rc<[usize]>,
    values: Rc<[Value<T>]>,
}

impl<T> Clone for Vector<T> {
    fn clone(&self) -> Self {
        Vector { shape: self.shape.clone(), values: self.values.clone() }
    }
}

/// A vector shape: an extent or a tuple, array or slice of extents (the
/// first the slowest).
pub trait VectorShape {
    #[doc(hidden)]
    fn extents(self) -> Vec<usize>;
}

impl VectorShape for usize {
    fn extents(self) -> Vec<usize> {
        vec![self]
    }
}

impl VectorShape for (usize,) {
    fn extents(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl VectorShape for (usize, usize) {
    fn extents(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl VectorShape for (usize, usize, usize) {
    fn extents(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl VectorShape for (usize, usize, usize, usize) {
    fn extents(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl<const N: usize> VectorShape for [usize; N] {
    fn extents(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl VectorShape for &[usize] {
    fn extents(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl VectorShape for Vec<usize> {
    fn extents(self) -> Vec<usize> {
        self
    }
}

#[track_caller]
fn checked_shape(shape: impl VectorShape, what: &str) -> Vec<usize> {
    let s = shape.extents();
    assert!(!s.is_empty(), "{what}: a vector has at least one dimension");
    assert!(s.iter().all(|&e| e > 0), "{what}: the shape {s:?} has an empty dimension");
    s
}

/// The row-major strides of `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// The shape `shapes` broadcast to (NumPy's rule), or a build panic naming
/// them.
#[track_caller]
fn broadcast_shapes(shapes: &[&[usize]], what: &str) -> Vec<usize> {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(1);
    let mut out = vec![1; rank];
    for s in shapes {
        let pad = rank - s.len();
        for (k, &e) in s.iter().enumerate() {
            let o = &mut out[pad + k];
            if *o == 1 {
                *o = e;
            } else if e != 1 && e != *o {
                panic!(
                    "{what}: the vector shapes {} do not broadcast: aligned on the right (NumPy's rule, the shorter \
                     padded with 1s on the left), each pair of extents must be equal or include a 1",
                    shapes.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(" and ")
                );
            }
        }
    }
    out
}

impl<T: Numeric> Vector<T> {
    /// A vector of `shape` holding `values` in row-major order. The number
    /// of values must be the shape's size.
    #[track_caller]
    pub fn new(shape: impl VectorShape, values: Vec<Value<T>>) -> Vector<T> {
        let shape = checked_shape(shape, "Vector::new");
        let n: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            n,
            "Vector::new: the shape {shape:?} has {n} elements, but {} values were given",
            values.len()
        );
        Vector { shape: shape.into(), values: values.into() }
    }

    pub(crate) fn from_parts(shape: Vec<usize>, values: Vec<Value<T>>) -> Vector<T> {
        debug_assert_eq!(shape.iter().product::<usize>(), values.len());
        Vector { shape: shape.into(), values: values.into() }
    }

    /// A 1-D vector of the given elements (CuTe DSL's
    /// `cutlass.Vector.from_elements(elems, dtype)`): host numbers or
    /// runtime values, each converted to `T` as [`full`] converts.
    #[track_caller]
    pub fn from_elements<X: FillValue<T>>(elems: impl IntoIterator<Item = X>) -> Vector<T> {
        let _at = LocationScope::new();
        let values: Vec<Value<T>> = elems.into_iter().map(|x| x.fill_value()).collect();
        assert!(!values.is_empty(), "Vector::from_elements: a vector has at least one element");
        Vector::from_parts(vec![values.len()], values)
    }

    /// The shape, the slowest dimension first (CuTe DSL's `Vector.shape`).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of elements.
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The element type (CuTe DSL's `Vector.dtype`).
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// The elements, in row-major order (CuTe DSL's `Vector.to_elements()`
    /// of a flattened vector).
    pub fn values(&self) -> &[Value<T>] {
        &self.values
    }

    /// The elements as a `Vec`, in row-major order.
    pub fn to_elements(&self) -> Vec<Value<T>> {
        self.values.to_vec()
    }

    /// A vector of the same shape holding `f` of every element.
    #[track_caller]
    pub fn map<U: Numeric>(&self, mut f: impl FnMut(Value<T>) -> Value<U>) -> Vector<U> {
        let _at = LocationScope::new();
        Vector::from_parts(self.shape.to_vec(), self.values.iter().map(|&v| f(v)).collect())
    }

    /// The same elements with another shape of the same size (MLIR's
    /// `vector.shape_cast`): the row-major order is kept.
    #[track_caller]
    pub fn reshape(&self, shape: impl VectorShape) -> Vector<T> {
        let shape = checked_shape(shape, "Vector::reshape");
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            self.size(),
            "Vector::reshape: {self:?} has {} elements, but the shape {shape:?} has {n}",
            self.size()
        );
        Vector { shape: shape.into(), values: self.values.clone() }
    }

    /// The elements as a 1-D vector, in row-major order.
    pub fn flatten(&self) -> Vector<T> {
        Vector { shape: vec![self.size()].into(), values: self.values.clone() }
    }

    /// This vector broadcast to `shape` (CuTe DSL's
    /// `cutlass.vector.broadcast_to`, MLIR's `vector.broadcast`): aligned on
    /// the right, every extent must equal the target's or be 1, and the
    /// target may have more dimensions. No instruction is emitted.
    #[track_caller]
    pub fn broadcast_to(&self, shape: impl VectorShape) -> Vector<T> {
        let target = checked_shape(shape, "Vector::broadcast_to");
        self.broadcast_to_extents(&target, "Vector::broadcast_to")
    }

    #[track_caller]
    fn broadcast_to_extents(&self, target: &[usize], what: &str) -> Vector<T> {
        if *self.shape == *target {
            return self.clone();
        }
        let ok = self.rank() <= target.len()
            && self.shape.iter().rev().zip(target.iter().rev()).all(|(&s, &d)| s == d || s == 1);
        if !ok {
            panic!(
                "{what}: a vector of shape {:?} cannot be broadcast to {target:?}: aligned on the right, each extent \
                 must equal the target's or be 1 (NumPy's rule)",
                self.shape
            );
        }
        // Row-major over the shape is column-major over the reversed shape,
        // which elementwise's maps work on.
        let dst: Vec<usize> = target.iter().rev().copied().collect();
        let mut src: Vec<usize> = self.shape.iter().rev().copied().collect();
        src.resize(dst.len(), 1);
        let map = elementwise::broadcast_map(&src, &dst);
        Vector::from_parts(target.to_vec(), elementwise::gather(&self.values, &map))
    }

    /// One element (CuTe DSL's `vec[i]` and `vec[i, j]`): a static
    /// coordinate (`usize` for a 1-D vector, a tuple or array of `usize`
    /// otherwise) picks a value; a runtime [`Value`] index of a 1-D vector
    /// picks one with a tree of selects, or, above
    /// [`SELECT_TREE_MAX`](crate::SELECT_TREE_MAX) elements, through a
    /// per-thread stack array (as [`TensorSsa::get`]). A runtime index is
    /// not checked.
    #[track_caller]
    pub fn get(&self, index: impl VectorIndex) -> Value<T> {
        let _at = LocationScope::new();
        match index.vector_index() {
            Index::Static(coord) => {
                assert_eq!(
                    coord.len(),
                    self.rank(),
                    "Vector::get: the coordinate {coord:?} has {} indices, but {self:?} has {} dimensions (slice \
                     takes a sub-vector)",
                    coord.len(),
                    self.rank()
                );
                for (k, (&c, &e)) in coord.iter().zip(self.shape.iter()).enumerate() {
                    assert!(c < e, "Vector::get: index {c} of dimension {k} is out of range for {self:?}");
                }
                let strides = row_major_strides(&self.shape);
                let i: usize = coord.iter().zip(&strides).map(|(c, s)| c * s).sum();
                self.values[i]
            }
            Index::Runtime(i) => {
                assert_eq!(
                    self.rank(),
                    1,
                    "Vector::get: a runtime index reads a 1-D vector; flatten {self:?} first or index it statically"
                );
                TensorSsa::from_parts(crate::IntTuple::from(self.size()), self.values.to_vec()).get(i)
            }
        }
    }

    /// A sub-vector (CuTe DSL's `vec[:, 1:3]`): one item per leading
    /// dimension, the rest kept. A range (`a..b`, `a..`, `..b`, `a..=b`,
    /// `..`) keeps a dimension's elements in it; a `usize` fixes the
    /// dimension and drops it. At least one dimension must remain (an
    /// element is [`get`](Vector::get)). Picks values; emits nothing.
    #[track_caller]
    pub fn slice(&self, s: impl VectorSlice) -> Vector<T> {
        let items = s.slice_items();
        assert!(
            items.len() <= self.rank(),
            "Vector::slice: {} items for {self:?}, which has {} dimensions",
            items.len(),
            self.rank()
        );
        let mut ranges: Vec<(usize, usize, bool)> = Vec::with_capacity(self.rank());
        for (k, &e) in self.shape.iter().enumerate() {
            let (lo, hi, keep) = match items.get(k) {
                None => (0, e, true),
                Some(SliceItem::Index(i)) => (*i, i + 1, false),
                Some(SliceItem::Range(lo, hi)) => (*lo, hi.unwrap_or(e), true),
            };
            assert!(
                lo < hi && hi <= e,
                "Vector::slice: {} is out of range (or empty) for dimension {k} of {self:?}",
                if keep { format!("the range {lo}..{hi}") } else { format!("the index {lo}") }
            );
            ranges.push((lo, hi, keep));
        }
        let shape: Vec<usize> = ranges.iter().filter(|r| r.2).map(|r| r.1 - r.0).collect();
        assert!(!shape.is_empty(), "Vector::slice: every dimension is fixed; read one element with get");
        let strides = row_major_strides(&self.shape);
        let mut values = Vec::new();
        let mut coord: Vec<usize> = ranges.iter().map(|r| r.0).collect();
        loop {
            values.push(self.values[coord.iter().zip(&strides).map(|(c, s)| c * s).sum::<usize>()]);
            // Advance the last dimension fastest.
            let mut k = coord.len();
            loop {
                if k == 0 {
                    return Vector::from_parts(shape, values);
                }
                k -= 1;
                coord[k] += 1;
                if coord[k] < ranges[k].1 {
                    break;
                }
                coord[k] = ranges[k].0;
            }
        }
    }

    /// Every element converted to `U` (CuTe DSL's `Vector.to(dtype)`), as
    /// [`TensorSsa::to`] converts.
    #[track_caller]
    pub fn to<U: Numeric>(&self) -> Vector<U> {
        let values = crate::tensor_ssa::convert_values::<T, U>(&self.values, "Vector::to");
        Vector::from_parts(self.shape.to_vec(), values)
    }

    /// The bits reinterpreted as `U` (CuTe DSL's `Vector.bitcast(dtype)`,
    /// MLIR's `vector.bitcast`): the innermost dimension is scaled by the
    /// width ratio, element 0 in the low bits, so a `(2, 4)` vector of
    /// Float16 is a `(2, 2)` vector of Uint32. One width must divide the
    /// other, and the innermost extent hold a whole number of groups.
    #[track_caller]
    pub fn bitcast<U: Numeric>(&self) -> Vector<U> {
        let (tb, ub) = (T::DTYPE.bits(), U::DTYPE.bits());
        let last = *self.shape.last().expect("a vector has a dimension");
        let bits = last * tb;
        assert!(
            T::DTYPE != DType::Bool && U::DTYPE != DType::Bool && bits.is_multiple_of(ub),
            "Vector::bitcast: the innermost dimension of {self:?} is {bits} bits, which {} values of {ub} bits cannot \
             hold exactly (Boolean values have no bit width)",
            U::DTYPE.cute_name()
        );
        let flat = TensorSsa::from_parts(crate::IntTuple::from(self.size()), self.values.to_vec()).bitcast::<U>();
        let mut shape = self.shape.to_vec();
        *shape.last_mut().expect("a dimension") = bits / ub;
        Vector::from_parts(shape, flat.values().to_vec())
    }

    /// Both operands converted to their common type and broadcast to one
    /// shape, `other` a scalar broadcast to this vector's shape.
    #[track_caller]
    fn promote_with<X>(&self, other: X, what: &str) -> Promoted<T, X>
    where
        X: VectorOperand<T>,
        T: CommonType<X::Elem>,
    {
        let a = self.map(|v| v.to::<Common<T, X>>());
        let b = other.into_vector_operand().vector_like(&self.shape).map(|v| v.to::<Common<T, X>>());
        let shape = broadcast_shapes(&[&a.shape, &b.shape], what);
        (a.broadcast_to_extents(&shape, what), b.broadcast_to_extents(&shape, what))
    }

    /// `f(self, other)` element by element after promotion and broadcasting
    /// (CuTe DSL's `Vector.apply_op`).
    #[track_caller]
    pub fn apply_op<X, O: Numeric>(
        &self,
        other: X,
        f: impl FnMut(Value<Common<T, X>>, Value<Common<T, X>>) -> Value<O>,
    ) -> Vector<O>
    where
        X: VectorOperand<T>,
        T: CommonType<X::Elem>,
    {
        let _at = LocationScope::new();
        let (a, b) = self.promote_with(other, "Vector::apply_op");
        Vector::from_parts(a.shape.to_vec(), elementwise::zip_with(&a.values, &b.values, f))
    }
}

/// A row-major index of a [`Vector`]: static coordinates or a runtime
/// [`Value`] index of a 1-D vector.
pub trait VectorIndex {
    #[doc(hidden)]
    fn vector_index(self) -> Index;
}

#[doc(hidden)]
pub enum Index {
    Static(Vec<usize>),
    Runtime(Value<i32>),
}

impl VectorIndex for usize {
    fn vector_index(self) -> Index {
        Index::Static(vec![self])
    }
}

impl VectorIndex for (usize, usize) {
    fn vector_index(self) -> Index {
        Index::Static(vec![self.0, self.1])
    }
}

impl VectorIndex for (usize, usize, usize) {
    fn vector_index(self) -> Index {
        Index::Static(vec![self.0, self.1, self.2])
    }
}

impl<const N: usize> VectorIndex for [usize; N] {
    fn vector_index(self) -> Index {
        Index::Static(self.to_vec())
    }
}

impl VectorIndex for Value<i32> {
    fn vector_index(self) -> Index {
        match self.as_constant() {
            Some(c) if c >= 0 => Index::Static(vec![c as usize]),
            _ => Index::Runtime(self),
        }
    }
}

/// One item of a [`Vector::slice`]: an index (dropping the dimension) or a
/// range of it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum SliceItem {
    Index(usize),
    Range(usize, Option<usize>),
}

/// A `usize` or a Rust range, one dimension of a [`Vector::slice`].
pub trait VectorSliceItem {
    #[doc(hidden)]
    fn slice_item(self) -> SliceItem;
}

impl VectorSliceItem for usize {
    fn slice_item(self) -> SliceItem {
        SliceItem::Index(self)
    }
}

impl VectorSliceItem for Range<usize> {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(self.start, Some(self.end))
    }
}

impl VectorSliceItem for RangeInclusive<usize> {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(*self.start(), Some(self.end() + 1))
    }
}

impl VectorSliceItem for RangeFrom<usize> {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(self.start, None)
    }
}

impl VectorSliceItem for RangeTo<usize> {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(0, Some(self.end))
    }
}

impl VectorSliceItem for RangeToInclusive<usize> {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(0, Some(self.end + 1))
    }
}

impl VectorSliceItem for RangeFull {
    fn slice_item(self) -> SliceItem {
        SliceItem::Range(0, None)
    }
}

/// The items of a [`Vector::slice`]: one item, or a tuple of up to four.
pub trait VectorSlice {
    #[doc(hidden)]
    fn slice_items(self) -> Vec<SliceItem>;
}

impl<A: VectorSliceItem> VectorSlice for A {
    fn slice_items(self) -> Vec<SliceItem> {
        vec![self.slice_item()]
    }
}

macro_rules! slice_tuples {
    ($(($($n:ident),+)),+) => {$(
        #[allow(non_snake_case)]
        impl<$($n: VectorSliceItem),+> VectorSlice for ($($n,)+) {
            fn slice_items(self) -> Vec<SliceItem> {
                let ($($n,)+) = self;
                vec![$($n.slice_item()),+]
            }
        }
    )+};
}
slice_tuples!((A), (A, B), (A, B, C), (A, B, C, D));

/// A right-hand side of a vector comparison, [`Vector::apply_op`] or
/// [`where_`]: another vector (by value or reference), a [`Value`], or a
/// host number of the vector's own type.
pub trait VectorOperand<T: Numeric> {
    /// The operand's element type.
    type Elem: Numeric;
    #[doc(hidden)]
    fn into_vector_operand(self) -> VectorOperandKind<Self::Elem>;
}

#[doc(hidden)]
pub enum VectorOperandKind<U> {
    Vector(Vector<U>),
    Scalar(Value<U>),
}

impl<U: Numeric> VectorOperandKind<U> {
    fn vector_like(self, shape: &[usize]) -> Vector<U> {
        match self {
            VectorOperandKind::Vector(v) => v,
            VectorOperandKind::Scalar(s) => Vector::from_parts(shape.to_vec(), vec![s; shape.iter().product()]),
        }
    }
}

impl<T: Numeric> VectorOperand<T> for T {
    type Elem = T;
    #[track_caller]
    fn into_vector_operand(self) -> VectorOperandKind<T> {
        VectorOperandKind::Scalar(Value::constant(self))
    }
}

impl<T: Numeric, U: Numeric> VectorOperand<T> for Value<U> {
    type Elem = U;
    fn into_vector_operand(self) -> VectorOperandKind<U> {
        VectorOperandKind::Scalar(self)
    }
}

impl<T: Numeric, U: Numeric> VectorOperand<T> for Vector<U> {
    type Elem = U;
    fn into_vector_operand(self) -> VectorOperandKind<U> {
        VectorOperandKind::Vector(self)
    }
}

impl<T: Numeric, U: Numeric> VectorOperand<T> for &Vector<U> {
    type Elem = U;
    fn into_vector_operand(self) -> VectorOperandKind<U> {
        VectorOperandKind::Vector(self.clone())
    }
}

type Common<T, X> = <T as CommonType<<X as VectorOperand<T>>::Elem>>::Output;
type Promoted<T, X> = (Vector<Common<T, X>>, Vector<Common<T, X>>);

/// `f(a, b)` element by element over `a` and `b` broadcast to one shape.
#[track_caller]
fn zip_broadcast<A: Numeric, B: Numeric, C: Numeric>(
    a: &Vector<A>,
    b: &Vector<B>,
    what: &str,
    f: impl FnMut(Value<A>, Value<B>) -> Value<C>,
) -> Vector<C> {
    let _at = LocationScope::new();
    let shape = broadcast_shapes(&[&a.shape, &b.shape], what);
    let (a, b) = (a.broadcast_to_extents(&shape, what), b.broadcast_to_extents(&shape, what));
    Vector::from_parts(shape, elementwise::zip_with(&a.values, &b.values, f))
}

macro_rules! vector_ops {
    ($bound:ident; $(($trait:ident, $method:ident)),*) => {$(
        impl<T: Numeric, U: Numeric, R: $bound> $trait<Vector<U>> for Vector<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = Vector<R>;
            #[track_caller]
            fn $method(self, rhs: Vector<U>) -> Vector<R> {
                zip_broadcast(&self, &rhs, concat!("Vector ", stringify!($method)), $trait::$method)
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<&Vector<U>> for &Vector<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = Vector<R>;
            #[track_caller]
            fn $method(self, rhs: &Vector<U>) -> Vector<R> {
                zip_broadcast(self, rhs, concat!("Vector ", stringify!($method)), $trait::$method)
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<Value<U>> for Vector<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = Vector<R>;
            #[track_caller]
            fn $method(self, rhs: Value<U>) -> Vector<R> {
                self.map(|a| $trait::$method(a, rhs))
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<Vector<U>> for Value<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = Vector<R>;
            #[track_caller]
            fn $method(self, rhs: Vector<U>) -> Vector<R> {
                rhs.map(|b| $trait::$method(self, b))
            }
        }
        impl<T: $bound> $trait<T> for Vector<T>
        where
            Value<T>: $trait<Value<T>, Output = Value<T>>,
        {
            type Output = Vector<T>;
            #[track_caller]
            fn $method(self, rhs: T) -> Vector<T> {
                let c = Value::constant(rhs);
                self.map(|a| $trait::$method(a, c))
            }
        }
    )*};
}

vector_ops!(Arith; (Add, add), (Sub, sub), (Mul, mul), (Div, div));
vector_ops!(Integer; (Rem, rem), (Shl, shl), (Shr, shr));
vector_ops!(Bitwise; (BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor));

/// A host number on the left, as `T op Value<T>`.
macro_rules! host_lhs {
    ($($host:ty),*) => {$(
        impl Add<Vector<$host>> for $host {
            type Output = Vector<$host>;
            #[track_caller]
            fn add(self, rhs: Vector<$host>) -> Vector<$host> {
                let c = Value::constant(self);
                rhs.map(|b| c + b)
            }
        }
        impl Sub<Vector<$host>> for $host {
            type Output = Vector<$host>;
            #[track_caller]
            fn sub(self, rhs: Vector<$host>) -> Vector<$host> {
                let c = Value::constant(self);
                rhs.map(|b| c - b)
            }
        }
        impl Mul<Vector<$host>> for $host {
            type Output = Vector<$host>;
            #[track_caller]
            fn mul(self, rhs: Vector<$host>) -> Vector<$host> {
                let c = Value::constant(self);
                rhs.map(|b| c * b)
            }
        }
        impl Div<Vector<$host>> for $host {
            type Output = Vector<$host>;
            #[track_caller]
            fn div(self, rhs: Vector<$host>) -> Vector<$host> {
                let c = Value::constant(self);
                rhs.map(|b| c / b)
            }
        }
    )*};
}
host_lhs!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, F16, BF16);

impl<T: Arith> Neg for Vector<T> {
    type Output = Vector<T>;
    #[track_caller]
    fn neg(self) -> Vector<T> {
        self.map(|a| -a)
    }
}

impl<T: Arith> Neg for &Vector<T> {
    type Output = Vector<T>;
    #[track_caller]
    fn neg(self) -> Vector<T> {
        self.map(|a| -a)
    }
}

impl<T: Bitwise> Not for Vector<T> {
    type Output = Vector<T>;
    #[track_caller]
    fn not(self) -> Vector<T> {
        self.map(|a| !a)
    }
}

macro_rules! comparisons {
    ($($(#[$doc:meta])* $name:ident => $pred:expr;)*) => {
        impl<T: Numeric> Vector<T> {$(
            $(#[$doc])*
            #[track_caller]
            pub fn $name<X>(&self, other: X) -> Vector<bool>
            where
                X: VectorOperand<T>,
                T: CommonType<X::Elem>,
            {
                let _at = LocationScope::new();
                let (a, b) = self.promote_with(other, concat!("Vector::", stringify!($name)));
                Vector::from_parts(
                    a.shape.to_vec(),
                    elementwise::zip_with(&a.values, &b.values, |x, y| compare_values($pred, x, y)),
                )
            }
        )*}
    };
}

comparisons! {
    /// `self == other` element by element, after promotion and broadcasting.
    eq => CmpPred::Eq;
    /// `self != other` (true where either is NaN).
    ne => CmpPred::Ne;
    /// `self < other` (false where either is NaN).
    lt => CmpPred::Lt;
    /// `self <= other`.
    le => CmpPred::Le;
    /// `self > other`.
    gt => CmpPred::Gt;
    /// `self >= other`.
    ge => CmpPred::Ge;
}

impl<T: Arith> Vector<T> {
    /// The absolute value of every element.
    #[track_caller]
    pub fn abs(&self) -> Vector<T> {
        if T::DTYPE.is_int() && !T::DTYPE.is_signed() {
            return self.clone();
        }
        self.map(|a| a.abs())
    }

    /// Every element folded into one value (CuTe DSL's `Vector.reduce(op)`,
    /// MLIR's `vector.reduction`): `acc = v[0]`, then `acc = acc op v[i]`
    /// in row-major order, exactly that sequence, as LLVM's ordered
    /// `vector.reduce.fadd` does, so the simulator and the GPU agree bit
    /// for bit. Float `Max` and `Min` propagate NaN ([`maximum`](crate::maximum),
    /// `arith.maximumf`), as [`TensorSsa::reduce`] does: the wheel's
    /// `cutlass._mlir_helpers.vector` chooses its combining kind out of
    /// sight, so this is cute-rs's documented choice.
    #[track_caller]
    pub fn reduce(&self, op: ReductionOp) -> Value<T> {
        let _at = LocationScope::new();
        elementwise::seq_fold(self.values[0], &self.values[1..], &mut |a, x| op.combine(a, x))
    }

    /// [`reduce`](Vector::reduce) from an accumulator (CuTe DSL's
    /// `Vector.reduce(op, acc)`): `acc op v[0] op v[1] ...`.
    #[track_caller]
    pub fn reduce_with(&self, op: ReductionOp, acc: impl FillValue<T>) -> Value<T> {
        let _at = LocationScope::new();
        elementwise::seq_fold(acc.fill_value(), &self.values, &mut |a, x| op.combine(a, x))
    }
}

impl<T: Float> Vector<T> {
    /// `self * b + c` element by element, fused ([`fma`](crate::fma)), with
    /// broadcasting.
    #[track_caller]
    pub fn fma(&self, b: &Vector<T>, c: &Vector<T>) -> Vector<T> {
        let _at = LocationScope::new();
        let shape = broadcast_shapes(&[&self.shape, &b.shape, &c.shape], "Vector::fma");
        let (a, b, c) = (
            self.broadcast_to_extents(&shape, "Vector::fma"),
            b.broadcast_to_extents(&shape, "Vector::fma"),
            c.broadcast_to_extents(&shape, "Vector::fma"),
        );
        let values = (0..a.size()).map(|i| crate::fma(a.values[i], b.values[i], c.values[i])).collect();
        Vector::from_parts(shape, values)
    }
}

impl<T: Numeric> Carry for Vector<T> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        out.extend(self.values.iter().map(|v| v.checked()));
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let vs = (0..self.size()).map(|_| Value::from_id(values.next().expect("carry: too few values"))).collect();
        Vector::from_parts(self.shape.to_vec(), vs)
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.shape != other.shape)
            .then(|| format!("a vector of shape {:?} became one of shape {:?}", self.shape, other.shape))
    }
}

/// `vector<2x4xf32>`, as MLIR prints the type.
impl<T: Numeric> fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(|e| e.to_string()).collect();
        write!(f, "vector<{}x{}>", dims.join("x"), T::DTYPE.name())
    }
}

/// A vector of `shape` with every element `fill` converted to `T` (CuTe
/// DSL's `cutlass.vector.full(shape, fill_value, dtype)`).
#[track_caller]
pub fn full<T: Numeric>(shape: impl VectorShape, fill: impl FillValue<T>) -> Vector<T> {
    let shape = checked_shape(shape, "vector::full");
    let v = fill.fill_value();
    let n = shape.iter().product();
    Vector::from_parts(shape, vec![v; n])
}

/// A vector shaped like `v` with every element `fill` (CuTe DSL's
/// `cutlass.vector.full_like`).
#[track_caller]
pub fn full_like<T: Numeric, U: Numeric>(v: &Vector<U>, fill: impl FillValue<T>) -> Vector<T> {
    full(v.shape.to_vec(), fill)
}

/// [`full_like`] with 0 (`cutlass.vector.zeros_like`).
#[track_caller]
pub fn zeros_like<T: Numeric, U: Numeric>(v: &Vector<U>) -> Vector<T> {
    full_like(v, 0i32)
}

/// [`full_like`] with 1 (`cutlass.vector.ones_like`).
#[track_caller]
pub fn ones_like<T: Numeric, U: Numeric>(v: &Vector<U>) -> Vector<T> {
    full_like(v, 1i32)
}

/// `cutlass.vector.empty_like`: [`full_like`] with 0 here (an SSA value
/// has no uninitialised state; MLIR's would be `ub.poison`).
#[track_caller]
pub fn empty_like<T: Numeric, U: Numeric>(v: &Vector<U>) -> Vector<T> {
    full_like(v, 0i32)
}

/// `v` broadcast to `shape` (`cutlass.vector.broadcast_to`): see
/// [`Vector::broadcast_to`].
#[track_caller]
pub fn broadcast_to<T: Numeric>(v: &Vector<T>, shape: impl VectorShape) -> Vector<T> {
    v.broadcast_to(shape)
}

/// `cond ? x : y` element by element (`cutlass.vector.where`): `x` and `y`
/// are vectors, [`Value`]s or host numbers of one type, at least one a
/// vector; the three broadcast to one shape (NumPy's rule).
#[track_caller]
pub fn where_<T: Numeric, X, Y>(cond: &Vector<bool>, x: X, y: Y) -> Vector<T>
where
    X: VectorOperand<T, Elem = T>,
    Y: VectorOperand<T, Elem = T>,
{
    let _at = LocationScope::new();
    let (x, y) = (x.into_vector_operand(), y.into_vector_operand());
    let shape = match (&x, &y) {
        (VectorOperandKind::Vector(v), _) | (_, VectorOperandKind::Vector(v)) => v.shape.to_vec(),
        _ => panic!("vector::where_: neither x nor y is a vector; one of them must be (or use select for scalars)"),
    };
    let (x, y) = (x.vector_like(&shape), y.vector_like(&shape));
    let shape = broadcast_shapes(&[&cond.shape, &x.shape, &y.shape], "vector::where_");
    let c = cond.broadcast_to_extents(&shape, "vector::where_");
    let (x, y) = (x.broadcast_to_extents(&shape, "vector::where_"), y.broadcast_to_extents(&shape, "vector::where_"));
    let values = (0..c.size()).map(|i| crate::select(c.values[i], x.values[i], y.values[i])).collect();
    Vector::from_parts(shape, values)
}

#[track_caller]
fn nonzero<T: Numeric>(x: &Vector<T>) -> Vec<Boolean> {
    let zero = Value::constant(T::from_bits(0));
    x.values.iter().map(|&v| compare_values(CmpPred::Ne, v, zero)).collect()
}

/// Whether any element is nonzero (`cutlass.vector.any_`), OR-ed in a
/// balanced tree.
#[track_caller]
pub fn any_<T: Numeric>(x: &Vector<T>) -> Boolean {
    let _at = LocationScope::new();
    elementwise::tree_fold(&nonzero(x), &mut |a, b| a | b)
}

/// Whether every element is nonzero (`cutlass.vector.all_`).
#[track_caller]
pub fn all_<T: Numeric>(x: &Vector<T>) -> Boolean {
    let _at = LocationScope::new();
    elementwise::tree_fold(&nonzero(x), &mut |a, b| a & b)
}

/// A 1-D Boolean vector of `len` lanes, lane `i` true iff `i < bound`
/// (`cutlass.vector.create_mask`, MLIR's `vector.create_mask`). The bound
/// may be a runtime value; below 0 every lane is false.
#[track_caller]
pub fn create_mask(len: usize, bound: impl IntoValue<i32>) -> Vector<bool> {
    let _at = LocationScope::new();
    assert!(len > 0, "vector::create_mask: a vector has at least one lane");
    let bound = bound.into_value();
    let values = (0..len).map(|i| Value::constant(i as i32).lt(bound)).collect();
    Vector::from_parts(vec![len], values)
}

/// The outer product of two 1-D vectors (`cutlass.vector.outerproduct`,
/// MLIR's `vector.outerproduct`): an `(m, n)` vector of `a[i] * b[j]`.
#[track_caller]
pub fn outerproduct<T: Arith>(a: &Vector<T>, b: &Vector<T>) -> Vector<T> {
    let _at = LocationScope::new();
    let (m, n) = outer_extents(a, b);
    let values = (0..m * n).map(|k| a.values[k / n] * b.values[k % n]).collect();
    Vector::from_parts(vec![m, n], values)
}

/// [`outerproduct`] with an accumulator (`vector.outerproduct` with `acc`):
/// `a[i] * b[j] + acc[i, j]`, fused for floats (`vector.outerproduct`
/// lowers to `vector.fma`), a multiply and an add for integers.
#[track_caller]
pub fn outerproduct_acc<T: Arith>(a: &Vector<T>, b: &Vector<T>, acc: &Vector<T>) -> Vector<T> {
    let _at = LocationScope::new();
    let loc = std::panic::Location::caller();
    let (m, n) = outer_extents(a, b);
    assert_eq!(
        *acc.shape,
        [m, n],
        "vector::outerproduct_acc: the accumulator {acc:?} is not ({m}, {n}), the shape of the outer product of {a:?} \
         and {b:?}"
    );
    let values = (0..m * n)
        .map(|k| {
            let (x, y, z) = (a.values[k / n], b.values[k % n], acc.values[k]);
            if T::DTYPE.is_float() {
                let (x, y, z) = (x.checked(), y.checked(), z.checked());
                Value::from_id(crate::builder::with(|bs| bs.fma(x, y, z, loc)))
            } else {
                x * y + z
            }
        })
        .collect();
    Vector::from_parts(vec![m, n], values)
}

#[track_caller]
fn outer_extents<T: Numeric>(a: &Vector<T>, b: &Vector<T>) -> (usize, usize) {
    assert!(a.rank() == 1 && b.rank() == 1, "vector::outerproduct: the operands {a:?} and {b:?} must be 1-D vectors");
    (a.size(), b.size())
}

/// Print a vector from every thread that runs it
/// (`cutlass.print_nd_vector`), as `printf` calls built here: the type,
/// then the elements in nested brackets, row-major, formatted as
/// [`print_tensor`](crate::print_tensor) formats them.
///
/// ```text
/// vector<2x3xf32> = [[ 1.0000,  2.0000,  3.0000], [ 4.0000,  5.0000,  6.0000]]
/// ```
#[track_caller]
pub fn print_nd_vector<T: Numeric>(v: &Vector<T>) {
    let _at = LocationScope::new();
    crate::tensor_ssa::print_nd(&format!("{v:?} = "), &v.shape, &v.values);
}

impl<T: Numeric> TensorSsa<T> {
    /// A tensor of `shape` over the elements of `v` in their row-major
    /// order, unpermuted (CuTe DSL's `TensorSSA(value, shape, dtype)`, which
    /// wraps a flat vector): element `k` of the flattened vector is the
    /// tensor's `k`-th element in colexicographic order.
    #[track_caller]
    pub fn from_flat_vector(v: &Vector<T>, shape: impl Into<crate::IntTuple>) -> TensorSsa<T> {
        TensorSsa::new(shape, v.values.to_vec())
    }

    /// A tensor from a row-major vector (CuTe DSL's
    /// `TensorSSA.from_vector(v, shape=)`), its elements permuted from
    /// row-major to column-major order: with the leaves of `shape` (by
    /// default the vector's own shape) as the extents, the element at
    /// row-major coordinate `(i0, ..., ik)` lands at the colexicographic
    /// index of the same coordinate, so `from_vector(v, (2, 3))` puts
    /// `v[i, j]` at index `i + 2j`, element `(i, j)` of the tensor.
    #[track_caller]
    pub fn from_vector(v: &Vector<T>, shape: Option<crate::IntTuple>) -> TensorSsa<T> {
        let _at = LocationScope::new();
        let shape = shape.unwrap_or_else(|| crate::IntTuple::from(v.shape.to_vec()));
        assert_eq!(
            shape.size(),
            v.size(),
            "TensorSsa::from_vector: the shape {shape:?} has {} elements, but {v:?} has {}",
            shape.size(),
            v.size()
        );
        let map = row_to_col(&shape.flat_leaves());
        let mut values = vec![v.values[0]; v.size()];
        for (r, &c) in map.iter().enumerate() {
            values[c] = v.values[r];
        }
        TensorSsa::from_parts(shape, values)
    }

    /// This tensor as a row-major vector (CuTe DSL's
    /// `TensorSSA.to_vector(force_flatten=)`), its elements permuted from
    /// column-major to row-major order: the vector's shape is the tensor's
    /// leaves, and element `(i0, ..., ik)` of each is the same element. A
    /// nested shape needs `force_flatten` (CuTe DSL refuses it otherwise).
    #[track_caller]
    pub fn to_vector(&self, force_flatten: bool) -> Vector<T> {
        let _at = LocationScope::new();
        assert!(
            self.shape().depth() <= 1 || force_flatten,
            "TensorSsa::to_vector: {self:?} has the nested shape {:?}; pass force_flatten = true to flatten it to its \
             leaves",
            self.shape()
        );
        let leaves = self.shape().flat_leaves();
        let map = row_to_col(&leaves);
        let values = map.iter().map(|&c| self.values()[c]).collect();
        Vector::from_parts(leaves, values)
    }

    /// The elements in colexicographic order (CuTe DSL's
    /// `TensorSSA.to_elements()`).
    pub fn to_elements(&self) -> Vec<Value<T>> {
        self.values().to_vec()
    }
}

/// For each row-major index over `extents`, the colexicographic index of
/// the same coordinate (CuTe DSL's `_get_row_and_col_map`).
fn row_to_col(extents: &[usize]) -> Vec<usize> {
    let n: usize = extents.iter().product();
    let row = row_major_strides(extents);
    let mut col = vec![1; extents.len()];
    for k in 1..extents.len() {
        col[k] = col[k - 1] * extents[k - 1];
    }
    (0..n).map(|r| extents.iter().enumerate().map(|(k, &e)| (r / row[k] % e) * col[k]).sum()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_to_col_maps() {
        // v[i, j] of a (2, 3) vector lands at i + 2j.
        assert_eq!(row_to_col(&[2, 3]), vec![0, 2, 4, 1, 3, 5]);
        assert_eq!(row_to_col(&[4]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn numpy_broadcasting() {
        assert_eq!(broadcast_shapes(&[&[4], &[3, 4]], "t"), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[&[3, 1], &[1, 4]], "t"), vec![3, 4]);
        assert!(std::panic::catch_unwind(|| broadcast_shapes(&[&[3], &[3, 4]], "t")).is_err());
    }
}

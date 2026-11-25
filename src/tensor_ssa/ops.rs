//! The operators of [`TensorSsa`]: arithmetic, bitwise and shifts, element
//! by element after promotion and broadcasting, for the same right-hand
//! sides as [`Value`]'s operators (and a host number on the left where
//! [`Value`] takes one there).

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use super::{TensorSsa, broadcast_shape, full};
use crate::builder::LocationScope;
use crate::elementwise;
use crate::ir::{BinaryOp, CmpPred, DType, UnaryOp};
use crate::promote::CommonType;
use crate::value::{Arith, BF16, Bitwise, F16, Float, Integer, Numeric, Value, binary_op, compare_values, select};

/// A right-hand side of [`TensorSsa::apply_op`], the comparisons,
/// [`where_`](super::where_) and the two-operand math functions: another
/// tensor (by value or reference), a [`Value`], or a host number of the
/// tensor's own type (so an unsuffixed literal adopts it).
pub trait SsaOperand<T: Numeric> {
    /// The operand's element type.
    type Elem: Numeric;
    #[doc(hidden)]
    fn into_operand(self) -> SsaOperandKind<Self::Elem>;
}

/// An operand as a tensor or a scalar (broadcast to the other operand).
#[doc(hidden)]
pub enum SsaOperandKind<U> {
    Tensor(TensorSsa<U>),
    Scalar(Value<U>),
}

impl<T: Numeric> SsaOperand<T> for T {
    type Elem = T;
    #[track_caller]
    fn into_operand(self) -> SsaOperandKind<T> {
        SsaOperandKind::Scalar(Value::constant(self))
    }
}

impl<T: Numeric, U: Numeric> SsaOperand<T> for Value<U> {
    type Elem = U;
    fn into_operand(self) -> SsaOperandKind<U> {
        SsaOperandKind::Scalar(self)
    }
}

impl<T: Numeric, U: Numeric> SsaOperand<T> for TensorSsa<U> {
    type Elem = U;
    fn into_operand(self) -> SsaOperandKind<U> {
        SsaOperandKind::Tensor(self)
    }
}

impl<T: Numeric, U: Numeric> SsaOperand<T> for &TensorSsa<U> {
    type Elem = U;
    fn into_operand(self) -> SsaOperandKind<U> {
        SsaOperandKind::Tensor(self.clone())
    }
}

impl<U: Numeric> SsaOperandKind<U> {
    /// The operand as a tensor: a scalar is broadcast to `shape` (CuTe DSL's
    /// `full(shape, v)`).
    pub(crate) fn tensor_like(self, shape: &crate::IntTuple) -> TensorSsa<U> {
        match self {
            SsaOperandKind::Tensor(t) => t,
            SsaOperandKind::Scalar(v) => full(shape.clone(), v),
        }
    }
}

/// `f(a, b)` element by element over `a` and `b` broadcast to a common shape.
#[track_caller]
pub(crate) fn zip_broadcast<A: Numeric, B: Numeric, C: Numeric>(
    a: &TensorSsa<A>,
    b: &TensorSsa<B>,
    what: &str,
    f: impl FnMut(Value<A>, Value<B>) -> Value<C>,
) -> TensorSsa<C> {
    let _at = LocationScope::new();
    let shape = broadcast_shape(&[&a.shape, &b.shape], what);
    let a = a.broadcast_to_shape(&shape, what);
    let b = b.broadcast_to_shape(&shape, what);
    TensorSsa::from_parts(shape, elementwise::zip_with(&a.values, &b.values, f))
}

/// Tensor and tensor, tensor and value, value and tensor: what `Value<T> op
/// Value<U>` gives, element by element. `$bound` bounds the result type.
macro_rules! tensor_ops {
    ($bound:ident; $(($trait:ident, $method:ident)),*) => {$(
        impl<T: Numeric, U: Numeric, R: $bound> $trait<TensorSsa<U>> for TensorSsa<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = TensorSsa<R>;
            #[track_caller]
            fn $method(self, rhs: TensorSsa<U>) -> TensorSsa<R> {
                zip_broadcast(&self, &rhs, concat!("TensorSsa ", stringify!($method)), $trait::$method)
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<&TensorSsa<U>> for &TensorSsa<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = TensorSsa<R>;
            #[track_caller]
            fn $method(self, rhs: &TensorSsa<U>) -> TensorSsa<R> {
                zip_broadcast(self, rhs, concat!("TensorSsa ", stringify!($method)), $trait::$method)
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<Value<U>> for TensorSsa<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = TensorSsa<R>;
            #[track_caller]
            fn $method(self, rhs: Value<U>) -> TensorSsa<R> {
                self.map(|a| $trait::$method(a, rhs))
            }
        }
        impl<T: Numeric, U: Numeric, R: $bound> $trait<TensorSsa<U>> for Value<T>
        where
            Value<T>: $trait<Value<U>, Output = Value<R>>,
        {
            type Output = TensorSsa<R>;
            #[track_caller]
            fn $method(self, rhs: TensorSsa<U>) -> TensorSsa<R> {
                rhs.map(|b| $trait::$method(self, b))
            }
        }
    )*};
}

/// A tensor and a host number of its own type, as `Value<T> op T`.
macro_rules! tensor_host_rhs {
    ($bound:ident; $(($trait:ident, $method:ident)),*) => {$(
        impl<T: $bound> $trait<T> for TensorSsa<T>
        where
            Value<T>: $trait<Value<T>, Output = Value<T>>,
        {
            type Output = TensorSsa<T>;
            #[track_caller]
            fn $method(self, rhs: T) -> TensorSsa<T> {
                let c = Value::constant(rhs);
                self.map(|a| $trait::$method(a, c))
            }
        }
    )*};
}

tensor_ops!(Arith; (Add, add), (Sub, sub), (Mul, mul), (Div, div));
tensor_ops!(Integer; (Rem, rem), (Shl, shl), (Shr, shr));
tensor_ops!(Bitwise; (BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor));
tensor_host_rhs!(Arith; (Add, add), (Sub, sub), (Mul, mul), (Div, div));
tensor_host_rhs!(Integer; (Rem, rem), (Shl, shl), (Shr, shr));
tensor_host_rhs!(Bitwise; (BitAnd, bitand), (BitOr, bitor), (BitXor, bitxor));

/// A host number on the left, as `T op Value<T>`.
macro_rules! host_lhs {
    ($host:ty => [$(($trait:ident, $method:ident)),*]) => {$(
        impl $trait<TensorSsa<$host>> for $host {
            type Output = TensorSsa<$host>;
            #[track_caller]
            fn $method(self, rhs: TensorSsa<$host>) -> TensorSsa<$host> {
                let c = Value::constant(self);
                rhs.map(|b| $trait::$method(c, b))
            }
        }
    )*};
}

macro_rules! host_lhs_ints {
    ($($t:ty),*) => {$(
        host_lhs!($t => [(Add, add), (Sub, sub), (Mul, mul), (Div, div), (Rem, rem)]);
    )*};
}
host_lhs_ints!(i8, i16, i32, i64, u8, u16, u32, u64, i128, u128);

macro_rules! host_lhs_floats {
    ($($t:ty),*) => {$(
        host_lhs!($t => [(Add, add), (Sub, sub), (Mul, mul), (Div, div)]);
    )*};
}
host_lhs_floats!(f32, f64, F16, BF16);

/// `%` on floats: [`math::rem`](crate::math::rem) (C's `fmod`: the quotient
/// truncated, so the result has the dividend's sign), which CuTe DSL's
/// `arith.remf` computes; [`Value`] has no float `%`.
macro_rules! float_rem {
    ($($t:ty),*) => {$(
        impl Rem<TensorSsa<$t>> for TensorSsa<$t> {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: TensorSsa<$t>) -> TensorSsa<$t> {
                zip_broadcast(&self, &rhs, "TensorSsa rem", |a, b| crate::math::rem(a, b, false))
            }
        }
        impl Rem<&TensorSsa<$t>> for &TensorSsa<$t> {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: &TensorSsa<$t>) -> TensorSsa<$t> {
                zip_broadcast(self, rhs, "TensorSsa rem", |a, b| crate::math::rem(a, b, false))
            }
        }
        impl Rem<Value<$t>> for TensorSsa<$t> {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: Value<$t>) -> TensorSsa<$t> {
                self.map(|a| crate::math::rem(a, rhs, false))
            }
        }
        impl Rem<TensorSsa<$t>> for Value<$t> {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: TensorSsa<$t>) -> TensorSsa<$t> {
                rhs.map(|b| crate::math::rem(self, b, false))
            }
        }
        impl Rem<$t> for TensorSsa<$t> {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: $t) -> TensorSsa<$t> {
                let c = Value::constant(rhs);
                self.map(|a| crate::math::rem(a, c, false))
            }
        }
        impl Rem<TensorSsa<$t>> for $t {
            type Output = TensorSsa<$t>;
            #[track_caller]
            fn rem(self, rhs: TensorSsa<$t>) -> TensorSsa<$t> {
                let c = Value::constant(self);
                rhs.map(|b| crate::math::rem(c, b, false))
            }
        }
    )*};
}
float_rem!(f32, f64, F16, BF16);

/// `-x` element by element: for floats the sign bit flipped (an exact
/// negation, `-0.0` from `+0.0`), as [`Value`]'s `-` is.
impl<T: Arith> Neg for TensorSsa<T> {
    type Output = TensorSsa<T>;
    #[track_caller]
    fn neg(self) -> TensorSsa<T> {
        self.map(|a| -a)
    }
}

impl<T: Arith> Neg for &TensorSsa<T> {
    type Output = TensorSsa<T>;
    #[track_caller]
    fn neg(self) -> TensorSsa<T> {
        self.map(|a| -a)
    }
}

/// `!x` element by element (bitwise for integers, logical for Booleans).
impl<T: Bitwise> Not for TensorSsa<T> {
    type Output = TensorSsa<T>;
    #[track_caller]
    fn not(self) -> TensorSsa<T> {
        self.map(|a| !a)
    }
}

impl<T: Bitwise> Not for &TensorSsa<T> {
    type Output = TensorSsa<T>;
    #[track_caller]
    fn not(self) -> TensorSsa<T> {
        self.map(|a| !a)
    }
}

/// Both operands of an elementwise operation, promoted.
pub(crate) type Promoted<T, X> = (TensorSsa<Common<T, X>>, TensorSsa<Common<T, X>>);

/// The promoted type of `T` and an operand `X`.
pub(crate) type Common<T, X> = <T as CommonType<<X as SsaOperand<T>>::Elem>>::Output;

impl<T: Numeric> TensorSsa<T> {
    /// Both operands converted to their common type (CuTe DSL's promotion),
    /// a scalar broadcast to this tensor's shape, swapped if `flip`, and
    /// broadcast to one shape.
    #[track_caller]
    pub(crate) fn promote_with<X>(&self, other: X, flip: bool) -> Promoted<T, X>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
    {
        let a = self.map(|v| v.to::<Common<T, X>>());
        let b = other.into_operand().tensor_like(&self.shape).map(|v| v.to::<Common<T, X>>());
        if flip { (b, a) } else { (a, b) }
    }

    /// `op(self, other)` element by element (CuTe DSL's
    /// `TensorSSA.apply_op(op, other, flip)`): `other` (a tensor, a
    /// [`Value`] or a host number) and this tensor are converted to their
    /// common type, a scalar is broadcast to this tensor's shape, `flip`
    /// swaps the operands (`op(other, self)`), and the two broadcast to a
    /// common shape.
    ///
    /// ```
    /// use cute::prelude::*;
    /// Kernel::build("apply_op", || {
    ///     let x = param_tensor::<f32>("x", Layout::col_major(8));
    ///     let out = param_tensor::<f32>("out", Layout::col_major(8));
    ///     let t = x.load_ssa();
    ///     // 1 - t, with CuTe DSL's NaN-propagating maximum against 0.
    ///     let r = t.apply_op(1.0, true, |a, b| a - b).apply_op(0.0, false, maximum);
    ///     out.store_ssa(&r);
    /// });
    /// ```
    #[track_caller]
    pub fn apply_op<X, O: Numeric>(
        &self,
        other: X,
        flip: bool,
        f: impl FnMut(Value<Common<T, X>>, Value<Common<T, X>>) -> Value<O>,
    ) -> TensorSsa<O>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
    {
        let _at = LocationScope::new();
        let (a, b) = self.promote_with(other, flip);
        zip_broadcast(&a, &b, "TensorSsa::apply_op", f)
    }

    /// CuTe DSL's true division (`/` on integer tensors): both operands
    /// converted to Float32 and divided, a Float32 result. `other` may be a
    /// tensor, a [`Value`] or a host number of this tensor's type.
    #[track_caller]
    pub fn true_div<X>(&self, other: X) -> TensorSsa<f32>
    where
        X: SsaOperand<T>,
    {
        let _at = LocationScope::new();
        let a = self.map(|v| v.to::<f32>());
        let b = other.into_operand().tensor_like(&self.shape).map(|v| v.to::<f32>());
        zip_broadcast(&a, &b, "TensorSsa::true_div", |x, y| x / y)
    }

    /// CuTe DSL's floor division (`//`), after promotion and broadcasting:
    /// signed integers round the quotient toward negative infinity
    /// (`arith.floordivsi`), unsigned integers divide plainly, and floats
    /// give `floor(a / b)`.
    #[track_caller]
    pub fn floor_div<X>(&self, other: X) -> TensorSsa<Common<T, X>>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
        Common<T, X>: Arith,
    {
        self.apply_op(other, false, floor_div_values)
    }

    /// CuTe DSL's `**`, after promotion and broadcasting: floats raise with
    /// [`math::pow`](crate::math::pow) (`fastmath` as there), integers with
    /// [`math::ipowi`](crate::math::ipowi). [`fpowi`](TensorSsa::fpowi)
    /// takes an integer exponent for a float base.
    #[track_caller]
    pub fn pow<X>(&self, exponent: X, fastmath: bool) -> TensorSsa<Common<T, X>>
    where
        X: SsaOperand<T>,
        T: CommonType<X::Elem>,
        Common<T, X>: Arith,
    {
        self.apply_op(exponent, false, |a, b| pow_values(a, b, fastmath))
    }
}

impl<T: Float> TensorSsa<T> {
    /// A float base raised to an Int32 exponent (tensor, value or host
    /// number), [`math::fpowi`](crate::math::fpowi) element by element,
    /// broadcasting.
    #[track_caller]
    pub fn fpowi<X>(&self, exponent: X) -> TensorSsa<T>
    where
        X: SsaOperand<i32, Elem = i32>,
    {
        let e = exponent.into_operand().tensor_like(&self.shape);
        zip_broadcast(self, &e, "TensorSsa::fpowi", |a, b| crate::math::fpowi(a, b, false))
    }
}

impl<T: Arith> TensorSsa<T> {
    /// The absolute value of every element ([`Value::abs`]; an unsigned
    /// tensor is its own).
    #[track_caller]
    pub fn abs(&self) -> TensorSsa<T> {
        if T::DTYPE.is_int() && !T::DTYPE.is_signed() {
            return self.clone();
        }
        self.map(|a| a.abs())
    }
}

/// A value handle under another name for the same type.
pub(crate) fn retype<A: Numeric, B: Numeric>(v: Value<A>) -> Value<B> {
    debug_assert_eq!(A::DTYPE, B::DTYPE);
    Value::from_id(v.id())
}

/// `floor(a / b)` for floats, the floor quotient for signed integers, the
/// quotient for unsigned ones.
#[track_caller]
fn floor_div_values<R: Arith>(a: Value<R>, b: Value<R>) -> Value<R> {
    let q = a / b;
    if R::DTYPE.is_float() {
        return Value::from_id(crate::value::unary_op(UnaryOp::Floor, q).id());
    }
    if !R::DTYPE.is_signed() {
        return q;
    }
    let zero = Value::constant(R::from_bits(0));
    let r = binary_op(BinaryOp::Rem, a, b);
    let inexact = compare_values(CmpPred::Ne, r, zero);
    let signs_differ = compare_values(CmpPred::Lt, r, zero) ^ compare_values(CmpPred::Lt, b, zero);
    select(inexact & signs_differ, q - Value::constant(R::from_bits(1)), q)
}

#[track_caller]
fn pow_values<R: Arith>(a: Value<R>, b: Value<R>, fastmath: bool) -> Value<R> {
    macro_rules! float_pow {
        ($t:ty) => {
            retype(crate::math::pow::<$t>(retype(a), retype(b), fastmath))
        };
    }
    macro_rules! int_pow {
        ($t:ty) => {
            retype(crate::math::ipowi::<$t>(retype(a), retype(b)))
        };
    }
    match R::DTYPE {
        DType::F32 => float_pow!(f32),
        DType::F64 => float_pow!(f64),
        DType::F16 => float_pow!(F16),
        DType::BF16 => float_pow!(BF16),
        DType::I8 => int_pow!(i8),
        DType::I16 => int_pow!(i16),
        DType::I32 => int_pow!(i32),
        DType::I64 => int_pow!(i64),
        DType::U8 => int_pow!(u8),
        DType::U16 => int_pow!(u16),
        DType::U32 => int_pow!(u32),
        DType::U64 => int_pow!(u64),
        t => panic!("TensorSsa::pow: no power of {} values", t.cute_name()),
    }
}

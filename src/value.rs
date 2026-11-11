//! Typed runtime values and numeric types.
//!
//! [`Value<T>`] is a handle to an SSA value of element type `T` in the kernel
//! being built. Arithmetic on values records IR, so kernel code reads like
//! ordinary Rust: `let c = a * b + 1.0;`. The aliases ([`Int32`], [`Uint8`],
//! [`Float64`], [`BFloat16`], [`Boolean`] and so on) mirror CuTe DSL's
//! `cutlass.Int32` and friends.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use std::panic::Location;

use crate::builder::{self, ValueFacts};
use crate::ir::{BinaryOp, CmpPred, DType, Literal, UnaryOp, ValueId};
use crate::layout::{Coord, IndexExpr, SliceCoord, Tree};

/// IEEE 754 half precision, as stored on the host.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// `0.0`.
    pub const ZERO: F16 = F16(0);
    /// `1.0`.
    pub const ONE: F16 = F16(0x3c00);

    /// Round to nearest even.
    pub fn from_f32(x: f32) -> F16 {
        F16(f32_to_f16_bits(x))
    }

    /// Exact conversion to f32.
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }
}

impl From<f32> for F16 {
    fn from(x: f32) -> Self {
        F16::from_f32(x)
    }
}

impl From<F16> for f32 {
    fn from(x: F16) -> Self {
        x.to_f32()
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// Brain floating point (BFloat16), as stored on the host: the top 16 bits of
/// an f32.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct BF16(pub u16);

impl BF16 {
    /// Round to nearest even.
    pub fn from_f32(x: f32) -> BF16 {
        BF16(f32_to_bf16_bits(x))
    }

    /// Exact conversion to f32.
    pub fn to_f32(self) -> f32 {
        bf16_bits_to_f32(self.0)
    }
}

impl fmt::Debug for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

pub(crate) fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub(crate) fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let (high, low) = ((bits >> 16) as u16, bits & 0xffff);
    if x.is_nan() {
        return high | 0x40;
    }
    high + (low > 0x8000 || (low == 0x8000 && high & 1 == 1)) as u16
}

pub(crate) fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        let nan_payload = if mant != 0 { 0x200 | (mant >> 13) as u16 } else { 0 };
        return sign | 0x7c00 | nan_payload;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Element types usable in kernels.
pub trait Numeric: Copy + fmt::Debug + 'static {
    const DTYPE: DType;
    fn literal(self) -> Literal;
    fn from_bits(bits: u64) -> Self;

    /// The raw bit pattern, zero-extended.
    fn raw_bits(self) -> u64 {
        self.literal().bits()
    }
}

/// Types with `+ - * /`.
pub trait Arith: Numeric {}

/// Floating-point types.
pub trait Float: Arith {}

/// Integer-only operators.
pub trait Integer: Arith {}

/// Types supporting `& | ^`.
pub trait Bitwise: Numeric {}

macro_rules! numeric_ints {
    ($($t:ty => $d:ident),*) => {$(
        impl Numeric for $t {
            const DTYPE: DType = DType::$d;
            fn literal(self) -> Literal {
                Literal::$d(self)
            }
            fn from_bits(bits: u64) -> Self {
                bits as $t
            }
        }
        impl Arith for $t {}
        impl Integer for $t {}
        impl Bitwise for $t {}
    )*};
}
numeric_ints!(i8 => I8, i16 => I16, i32 => I32, i64 => I64, u8 => U8, u16 => U16, u32 => U32, u64 => U64);

impl Numeric for f32 {
    const DTYPE: DType = DType::F32;
    fn literal(self) -> Literal {
        Literal::F32(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
}

impl Numeric for f64 {
    const DTYPE: DType = DType::F64;
    fn literal(self) -> Literal {
        Literal::F64(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
}

impl Numeric for F16 {
    const DTYPE: DType = DType::F16;
    fn literal(self) -> Literal {
        Literal::F16(self.0)
    }
    fn from_bits(bits: u64) -> Self {
        F16(bits as u16)
    }
}

impl Numeric for BF16 {
    const DTYPE: DType = DType::BF16;
    fn literal(self) -> Literal {
        Literal::BF16(self.0)
    }
    fn from_bits(bits: u64) -> Self {
        BF16(bits as u16)
    }
}

impl Numeric for bool {
    const DTYPE: DType = DType::Bool;
    fn literal(self) -> Literal {
        Literal::Bool(self)
    }
    fn from_bits(bits: u64) -> Self {
        bits != 0
    }
}

impl Arith for f32 {}
impl Arith for f64 {}
impl Arith for F16 {}
impl Arith for BF16 {}
impl Float for f32 {}
impl Float for f64 {}
impl Float for F16 {}
impl Float for BF16 {}
impl Bitwise for bool {}

/// A runtime value of element type `T` in the kernel being built.
pub struct Value<T> {
    pub(crate) id: ValueId,
    pub(crate) kernel: u32,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Value<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Value<T> {}

impl<T> fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value({})", self.id)
    }
}

pub type Int8 = Value<i8>;
pub type Int16 = Value<i16>;
pub type Int32 = Value<i32>;
pub type Int64 = Value<i64>;
pub type Uint8 = Value<u8>;
pub type Uint16 = Value<u16>;
pub type Uint32 = Value<u32>;
pub type Uint64 = Value<u64>;
pub type Float16 = Value<F16>;
pub type BFloat16 = Value<BF16>;
pub type Float32 = Value<f32>;
pub type Float64 = Value<f64>;
pub type Boolean = Value<bool>;

impl<T: Numeric> Value<T> {
    pub(crate) fn from_id(id: ValueId) -> Self {
        Value { id, kernel: builder::with(|b| b.kernel), _t: PhantomData }
    }

    /// The underlying SSA value, after checking it belongs to the kernel
    /// being built.
    #[track_caller]
    pub(crate) fn checked(self) -> ValueId {
        let current = builder::with(|b| b.kernel);
        assert_eq!(self.kernel, current, "cute: a value from another kernel was used");
        self.id
    }

    /// The SSA value in the IR.
    pub fn id(&self) -> ValueId {
        self.id
    }

    /// Materialise a constant.
    #[track_caller]
    pub fn constant(v: T) -> Self {
        let loc = Location::caller();
        let id = builder::with(|b| b.constant(v.literal(), loc));
        Value::from_id(id)
    }

    /// The constant this value was created from, if any.
    pub fn as_constant(&self) -> Option<T> {
        builder::with(|b| b.constant_of(self.id)).map(|l| T::from_bits(l.bits()))
    }

    /// Numeric conversion (round to nearest for float narrowing, truncation
    /// toward zero for float to int).
    #[track_caller]
    pub fn to<U: Numeric>(self) -> Value<U> {
        let loc = Location::caller();
        let x = self.checked();
        let id = builder::with(|b| b.cast(x, U::DTYPE, loc));
        Value::from_id(id)
    }

    /// Reinterpret the bits as `U`, which must have the same width.
    #[track_caller]
    pub fn bitcast<U: Numeric>(self) -> Value<U> {
        let loc = Location::caller();
        let x = self.checked();
        Value::from_id(builder::with(|b| b.bitcast(x, U::DTYPE, loc)))
    }

    #[track_caller]
    fn cmp(self, pred: CmpPred, rhs: impl IntoValue<T>) -> Boolean {
        let loc = Location::caller();
        let (a, b) = (self.checked(), rhs.into_value().checked());
        Value::from_id(builder::with(|bs| bs.cmp(pred, a, b, loc)))
    }

    #[track_caller]
    /// `self == rhs`.
    pub fn eq(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Eq, rhs)
    }

    #[track_caller]
    /// `self != rhs` (true for NaN).
    pub fn ne(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Ne, rhs)
    }

    #[track_caller]
    /// `self < rhs`.
    pub fn lt(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Lt, rhs)
    }

    #[track_caller]
    /// `self <= rhs`.
    pub fn le(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Le, rhs)
    }

    #[track_caller]
    /// `self > rhs`.
    pub fn gt(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Gt, rhs)
    }

    #[track_caller]
    /// `self >= rhs`.
    pub fn ge(self, rhs: impl IntoValue<T>) -> Boolean {
        self.cmp(CmpPred::Ge, rhs)
    }
}

#[track_caller]
fn unary<T: Numeric>(op: UnaryOp, x: Value<T>) -> Value<T> {
    let loc = Location::caller();
    let x = x.checked();
    Value::from_id(builder::with(|b| b.unary(op, x, loc)))
}

#[track_caller]
fn binary<T: Numeric>(op: BinaryOp, a: Value<T>, b: Value<T>) -> Value<T> {
    let loc = Location::caller();
    let (a, b) = (a.checked(), b.checked());
    Value::from_id(builder::with(|bs| bs.binary(op, a, b, loc)))
}

impl<T: Arith> Value<T> {
    #[track_caller]
    /// The smaller value (for floats, a NaN operand is ignored and -0 is
    /// smaller than +0): [`math::min`](crate::math::min).
    pub fn min(self, rhs: impl IntoValue<T>) -> Value<T> {
        binary(BinaryOp::Min, self, rhs.into_value())
    }

    #[track_caller]
    /// The larger value (for floats, a NaN operand is ignored and +0 is
    /// larger than -0): [`math::max`](crate::math::max).
    pub fn max(self, rhs: impl IntoValue<T>) -> Value<T> {
        binary(BinaryOp::Max, self, rhs.into_value())
    }

    #[track_caller]
    /// Absolute value ([`math::abs`](crate::math::abs)). For floats, the
    /// sign bit cleared (IEEE 754 `abs`), as an `and` in inline PTX, so a
    /// NaN keeps its payload on the GPU as in the simulator (LLVM would
    /// otherwise emit `abs.f32` in some places, which returns the canonical
    /// NaN).
    pub fn abs(self) -> Value<T> {
        unary(UnaryOp::Abs, self)
    }
}

impl<T: Float> Value<T> {
    #[track_caller]
    /// Square root, correctly rounded (`sqrt.rn`): the same as
    /// [`math::sqrt(x, false)`](crate::math::sqrt).
    pub fn sqrt(self) -> Value<T> {
        unary(UnaryOp::Sqrt, self)
    }

    #[track_caller]
    /// `1 / sqrt(x)`, computed with the hardware approximation `rsqrt.approx`
    /// (2^-22.9 relative in Float32; Float16 and BFloat16 are computed in
    /// Float32). On Float32 and Float64 it is
    /// [`math::rsqrt_with`](crate::math::rsqrt_with) with
    /// [`MathOptions::APPROX`](crate::MathOptions::APPROX). The correctly
    /// rounded form is `math::rsqrt(x, false)`.
    pub fn rsqrt(self) -> Value<T> {
        unary(UnaryOp::Rsqrt, self)
    }

    /// `2^x`, computed with the hardware approximation `ex2.approx.f32`
    /// (Float32, and Float16 and BFloat16 computed in Float32; 2 ulp of the
    /// correctly rounded result). On Float32 it is
    /// [`math::exp2_with`](crate::math::exp2_with) with
    /// [`MathOptions::APPROX`](crate::MathOptions::APPROX).
    /// `math::exp2(x, true)` adds `.ftz`; `math::exp2(x, false)` is accurate
    /// to 1 ulp and works on Float64.
    #[track_caller]
    pub fn exp2(self) -> Value<T> {
        unary(UnaryOp::Exp2, self)
    }

    /// `log2(x)`, computed with the hardware approximation `lg2.approx.f32`
    /// (Float32, and narrower types computed in Float32): on Float32,
    /// `math::log2_with(x, MathOptions::APPROX)`. See
    /// [`math::log2`](crate::math::log2) for the precise form.
    #[track_caller]
    pub fn log2(self) -> Value<T> {
        unary(UnaryOp::Log2, self)
    }

    /// Sine, computed with the hardware approximation `sin.approx.f32`
    /// (Float32, and narrower types computed in Float32; accurate only for
    /// moderate arguments): on Float32, `math::sin_with(x,
    /// MathOptions::APPROX)`. See [`math::sin`](crate::math::sin) for the
    /// precise form.
    #[track_caller]
    pub fn sin(self) -> Value<T> {
        unary(UnaryOp::Sin, self)
    }

    /// Cosine, computed with the hardware approximation `cos.approx.f32`
    /// (Float32, and narrower types computed in Float32): on Float32,
    /// `math::cos_with(x, MathOptions::APPROX)`. See
    /// [`math::cos`](crate::math::cos) for the precise form.
    #[track_caller]
    pub fn cos(self) -> Value<T> {
        unary(UnaryOp::Cos, self)
    }

    /// `1 / x`, computed with the hardware approximation `rcp.approx.f32`
    /// (1 ulp) in Float32 (where it is `math::rcp_with(x,
    /// MathOptions::APPROX)`) and in Float16 and BFloat16 (computed in
    /// Float32), and correctly rounded (`rcp.rn.f64`, the same as
    /// [`math::rcp(x, false)`](crate::math::rcp)) in Float64.
    #[track_caller]
    pub fn rcp(self) -> Value<T> {
        unary(UnaryOp::Rcp, self)
    }

    /// Round toward negative infinity ([`math::floor`](crate::math::floor)).
    #[track_caller]
    pub fn floor(self) -> Value<T> {
        unary(UnaryOp::Floor, self)
    }

    /// Round toward positive infinity ([`math::ceil`](crate::math::ceil)).
    #[track_caller]
    pub fn ceil(self) -> Value<T> {
        unary(UnaryOp::Ceil, self)
    }

    /// Round toward zero ([`math::trunc`](crate::math::trunc)).
    #[track_caller]
    pub fn trunc(self) -> Value<T> {
        unary(UnaryOp::Trunc, self)
    }

    /// Round to the nearest integer, ties to even
    /// ([`math::roundeven`](crate::math::roundeven); `math::round` rounds
    /// ties away from zero).
    #[track_caller]
    pub fn rint(self) -> Value<T> {
        unary(UnaryOp::Rint, self)
    }

    /// `e^x` as `2^(x * log2(e))` with the hardware approximation
    /// ([`Value::exp2`]): `math::exp(x, true)` without `.ftz`. See
    /// [`math::exp`](crate::math::exp) for the precise form.
    #[track_caller]
    pub fn exp(self) -> Value<T> {
        let scale: Value<T> = Value::<f32>::constant(std::f32::consts::LOG2_E).to();
        (self * scale).exp2()
    }

    /// Natural logarithm as `log2(x) * ln(2)` with the hardware
    /// approximation ([`Value::log2`]): `math::log(x, true)` without `.ftz`.
    /// See [`math::log`](crate::math::log) for the precise form.
    #[track_caller]
    pub fn ln(self) -> Value<T> {
        let scale: Value<T> = Value::<f32>::constant(std::f32::consts::LN_2).to();
        self.log2() * scale
    }
}

impl<T: Integer> Value<T> {
    /// Number of set bits.
    #[track_caller]
    pub fn popc(self) -> Value<T> {
        unary(UnaryOp::Popc, self)
    }

    /// Number of leading zero bits (the type's width for zero).
    #[track_caller]
    pub fn clz(self) -> Value<T> {
        unary(UnaryOp::Clz, self)
    }

    /// Number of trailing zero bits (the type's width for zero).
    #[track_caller]
    pub fn ctz(self) -> Value<T> {
        unary(UnaryOp::Ctz, self)
    }

    /// Reverse the bits.
    #[track_caller]
    pub fn brev(self) -> Value<T> {
        unary(UnaryOp::Brev, self)
    }

    /// The high half of the full-width product. At most 32-bit types.
    #[track_caller]
    pub fn mul_hi(self, rhs: impl IntoValue<T>) -> Value<T> {
        binary(BinaryOp::MulHi, self, rhs.into_value())
    }
}

impl Int32 {
    /// This value as an index expression (assumed non-negative).
    pub fn as_index(self) -> IndexExpr {
        IndexExpr::from(self)
    }
}

/// `a * b + c` with a single rounding.
#[track_caller]
pub fn fma<T: Float>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>) -> Value<T> {
    let loc = Location::caller();
    let (a, b, c) = (a.into_value().checked(), b.into_value().checked(), c.into_value().checked());
    Value::from_id(builder::with(|bs| bs.fma(a, b, c, loc)))
}

/// `if cond { on_true } else { on_false }` without branching.
#[track_caller]
pub fn select<T: Numeric>(cond: Boolean, on_true: impl IntoValue<T>, on_false: impl IntoValue<T>) -> Value<T> {
    let loc = Location::caller();
    let (c, t, f) = (cond.checked(), on_true.into_value().checked(), on_false.into_value().checked());
    Value::from_id(builder::with(|bs| bs.select(c, t, f, loc)))
}

/// Conversion into a runtime value: values themselves and host constants.
pub trait IntoValue<T: Numeric> {
    fn into_value(self) -> Value<T>;
}

impl<T: Numeric> IntoValue<T> for Value<T> {
    fn into_value(self) -> Value<T> {
        self
    }
}

impl<T: Numeric> IntoValue<T> for T {
    #[track_caller]
    fn into_value(self) -> Value<T> {
        Value::constant(self)
    }
}

impl IntoValue<i32> for usize {
    #[track_caller]
    fn into_value(self) -> Value<i32> {
        Value::constant(i32::try_from(self).unwrap_or_else(|_| panic!("cute: {self} does not fit in i32")))
    }
}

impl IntoValue<i32> for IndexExpr {
    #[track_caller]
    fn into_value(self) -> Value<i32> {
        let loc = Location::caller();
        let facts = ValueFacts {
            bound: self.bound(),
            divisor: (0..31).rev().map(|k| 1usize << k).find(|&d| self.is_multiple_of(d)).unwrap_or(1),
        };
        let id = builder::with(|b| {
            let id = b.materialize(&self, loc);
            if b.constant_of(id).is_none() {
                b.set_facts(id, facts);
            }
            id
        });
        Value::from_id(id)
    }
}

impl IntoValue<i32> for &IndexExpr {
    #[track_caller]
    fn into_value(self) -> Value<i32> {
        self.clone().into_value()
    }
}

macro_rules! from_host {
    ($($t:ty => $v:ty),*) => {$(
        impl From<$t> for $v {
            #[track_caller]
            fn from(x: $t) -> Self { x.into_value() }
        }
    )*};
}
from_host!(
    i8 => Int8, i16 => Int16, i32 => Int32, i64 => Int64, u8 => Uint8, u16 => Uint16, u32 => Uint32, u64 => Uint64,
    usize => Int32, f32 => Float32, f64 => Float64, F16 => Float16, BF16 => BFloat16, bool => Boolean,
    IndexExpr => Int32
);

/// A value of any type, for operations that take a mix ([`printf`](crate::printf)
/// and [`inline_ptx`](crate::inline_ptx)).
#[derive(Clone, Copy, Debug)]
pub struct AnyValue(pub(crate) ValueId);

impl<T: Numeric> From<Value<T>> for AnyValue {
    #[track_caller]
    fn from(v: Value<T>) -> Self {
        AnyValue(v.checked())
    }
}

impl From<Int32> for IndexExpr {
    fn from(v: Int32) -> Self {
        if builder::value_facts(v.id).bound == Some(1) {
            return IndexExpr::Static(0);
        }
        if let Some(Literal::I32(c)) = builder::try_constant(v.id)
            && c >= 0
        {
            return IndexExpr::Static(c as usize);
        }
        IndexExpr::Dynamic(v.id)
    }
}

impl From<Int32> for Coord {
    fn from(v: Int32) -> Self {
        Tree::Leaf(v.into())
    }
}

impl From<Int32> for SliceCoord {
    fn from(v: Int32) -> Self {
        Tree::Leaf(Some(v.into()))
    }
}

macro_rules! arith_ops {
    ($($trait:ident, $method:ident, $op:expr, $bound:ident);* $(;)?) => {$(
        impl<T: $bound> $trait<Value<T>> for Value<T> {
            type Output = Value<T>;
            #[track_caller]
            fn $method(self, rhs: Value<T>) -> Value<T> { binary($op, self, rhs) }
        }
        impl<T: $bound> $trait<T> for Value<T> {
            type Output = Value<T>;
            #[track_caller]
            fn $method(self, rhs: T) -> Value<T> { binary($op, self, Value::constant(rhs)) }
        }
    )*};
}

#[track_caller]
fn shift_right<T: Numeric>() -> BinaryOp {
    if T::DTYPE.is_signed() { BinaryOp::ShrS } else { BinaryOp::ShrU }
}

arith_ops! {
    Add, add, BinaryOp::Add, Arith;
    Sub, sub, BinaryOp::Sub, Arith;
    Mul, mul, BinaryOp::Mul, Arith;
    Div, div, BinaryOp::Div, Arith;
    Rem, rem, BinaryOp::Rem, Integer;
    BitAnd, bitand, BinaryOp::And, Bitwise;
    BitOr, bitor, BinaryOp::Or, Bitwise;
    BitXor, bitxor, BinaryOp::Xor, Bitwise;
    Shl, shl, BinaryOp::Shl, Integer;
    Shr, shr, shift_right::<T>(), Integer;
}

macro_rules! host_lhs_ops {
    ($host:ty => [$(($trait:ident, $method:ident, $op:expr)),*]) => {$(
        impl $trait<Value<$host>> for $host {
            type Output = Value<$host>;
            #[track_caller]
            fn $method(self, rhs: Value<$host>) -> Value<$host> { binary($op, Value::constant(self), rhs) }
        }
    )*};
}

macro_rules! host_lhs_ints {
    ($($t:ty),*) => {$(
        host_lhs_ops!($t => [
            (Add, add, BinaryOp::Add), (Sub, sub, BinaryOp::Sub), (Mul, mul, BinaryOp::Mul),
            (Div, div, BinaryOp::Div), (Rem, rem, BinaryOp::Rem)
        ]);
    )*};
}
host_lhs_ints!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! host_lhs_floats {
    ($($t:ty),*) => {$(
        host_lhs_ops!($t => [
            (Add, add, BinaryOp::Add), (Sub, sub, BinaryOp::Sub), (Mul, mul, BinaryOp::Mul), (Div, div, BinaryOp::Div)
        ]);
    )*};
}
host_lhs_floats!(f32, f64, F16, BF16);

/// `-x`. For floats, the sign bit flipped (IEEE 754 `negate`), as an `xor`
/// in inline PTX, so a NaN keeps its payload on the GPU as in the
/// simulator ([`math::neg`](crate::math::neg)).
impl<T: Arith> Neg for Value<T> {
    type Output = Value<T>;
    #[track_caller]
    fn neg(self) -> Value<T> {
        unary(UnaryOp::Neg, self)
    }
}

impl<T: Bitwise> Not for Value<T> {
    type Output = Value<T>;
    #[track_caller]
    fn not(self) -> Value<T> {
        unary(UnaryOp::Not, self)
    }
}

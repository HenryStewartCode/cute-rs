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
use crate::promote::CmpOperand;

/// IEEE 754 half precision, as stored on the host.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// `0.0`.
    pub const ZERO: F16 = F16(0);
    /// `1.0`.
    pub const ONE: F16 = F16(0x3c00);
    /// The most negative finite value, -65504.
    pub const MIN: F16 = F16(0xfbff);
    /// The largest finite value, 65504.
    pub const MAX: F16 = F16(0x7bff);
    /// Positive infinity.
    pub const INFINITY: F16 = F16(0x7c00);
    /// The canonical NaN PTX produces.
    pub const NAN: F16 = F16(0x7fff);

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
    /// `0.0`.
    pub const ZERO: BF16 = BF16(0);
    /// `1.0`.
    pub const ONE: BF16 = BF16(0x3f80);
    /// The most negative finite value.
    pub const MIN: BF16 = BF16(0xff7f);
    /// The largest finite value, about 3.39e38.
    pub const MAX: BF16 = BF16(0x7f7f);
    /// Positive infinity.
    pub const INFINITY: BF16 = BF16(0x7f80);
    /// The canonical NaN PTX produces.
    pub const NAN: BF16 = BF16(0x7fff);

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

impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for BF16 {
    fn from(x: f32) -> Self {
        BF16::from_f32(x)
    }
}

impl From<BF16> for f32 {
    fn from(x: BF16) -> Self {
        x.to_f32()
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
///
/// The associated constants are CuTe DSL's type metadata (`T.width`,
/// `T.signed`, `T.is_float`, `T.is_integer`), usable in const contexts:
/// `128 / T::WIDTH` values of `T` fill a 128-bit vector. [`DType`] has the
/// rest of the table (`DType::bytes`, `min_value`, `inf`, `exponent_width`
/// and so on).
pub trait Numeric: Copy + fmt::Debug + 'static {
    const DTYPE: DType;
    /// Bits of one value (CuTe DSL's `T.width`): 1 for Boolean, 4 for Int4.
    const WIDTH: usize = Self::DTYPE.bits();
    /// Bytes of one value in memory (CuTe DSL's `T.bytes`).
    const BYTES: usize = Self::DTYPE.bytes();
    /// Whether the type has a sign (CuTe DSL's `T.signed`; true for
    /// Boolean, as there, and for the signed float formats).
    const SIGNED: bool = Self::DTYPE.info().signed;
    /// CuTe DSL's `T.is_float`.
    const IS_FLOAT: bool = Self::DTYPE.is_float();
    /// CuTe DSL's `T.is_integer` (true for Boolean, as there).
    const IS_INTEGER: bool = !Self::DTYPE.is_float();

    fn literal(self) -> Literal;
    fn from_bits(bits: u64) -> Self;

    /// The raw bit pattern, zero-extended.
    fn raw_bits(self) -> u64 {
        self.literal().bits()
    }

    /// The value with raw bits `bits`, for types up to 128 bits wide.
    fn from_bits128(bits: u128) -> Self {
        Self::from_bits(bits as u64)
    }

    /// The raw bit pattern, zero-extended to 128 bits.
    fn raw_bits128(self) -> u128 {
        self.literal().bits128()
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

macro_rules! numeric_wide_ints {
    ($($t:ty => $d:ident),*) => {$(
        impl Numeric for $t {
            const DTYPE: DType = DType::$d;
            fn literal(self) -> Literal {
                Literal::$d(self)
            }
            fn from_bits(bits: u64) -> Self {
                bits as $t
            }
            fn from_bits128(bits: u128) -> Self {
                bits as $t
            }
        }
        impl Arith for $t {}
        impl Integer for $t {}
        impl Bitwise for $t {}
    )*};
}
numeric_wide_ints!(i128 => I128, u128 => U128);

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
/// CuTe DSL's `cutlass.Int128`.
pub type Int128 = Value<i128>;
/// CuTe DSL's `cutlass.Uint128`.
pub type Uint128 = Value<u128>;

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
        builder::with(|b| b.constant_of(self.id)).map(|l| T::from_bits128(l.bits128()))
    }

    /// Numeric conversion (round to nearest for float narrowing, truncation
    /// toward zero for float to int).
    ///
    /// To [`Boolean`] it is CuTe DSL's `Boolean(x)`: `x != 0` for integers
    /// and an unordered `x != 0.0` for floats, so NaN converts to `true`.
    /// To [`TFloat32`](crate::TFloat32) it rounds as `cvt.rna.tf32.f32` does
    /// (to nearest, ties away from zero), and from it it is free.
    #[track_caller]
    pub fn to<U: Numeric>(self) -> Value<U> {
        let loc = Location::caller();
        let x = self.checked();
        let id = builder::with(|b| crate::convert::convert(b, x, U::DTYPE, loc));
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
    fn cmp(self, pred: CmpPred, rhs: impl CmpOperand<T>) -> Boolean {
        rhs.compare_with(self, pred)
    }

    #[track_caller]
    /// `self == rhs`.
    pub fn eq(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Eq, rhs)
    }

    #[track_caller]
    /// `self != rhs` (true for NaN).
    pub fn ne(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Ne, rhs)
    }

    #[track_caller]
    /// `self < rhs`.
    pub fn lt(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Lt, rhs)
    }

    #[track_caller]
    /// `self <= rhs`.
    pub fn le(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Le, rhs)
    }

    #[track_caller]
    /// `self > rhs`.
    pub fn gt(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Gt, rhs)
    }

    #[track_caller]
    /// `self >= rhs`.
    pub fn ge(self, rhs: impl CmpOperand<T>) -> Boolean {
        self.cmp(CmpPred::Ge, rhs)
    }
}

/// `a pred b`. TFloat32 and the 8-bit float formats compare as their
/// Float32 values (exactly).
#[track_caller]
pub(crate) fn compare_values<T: Numeric>(pred: CmpPred, a: Value<T>, b: Value<T>) -> Boolean {
    let loc = Location::caller();
    let (a, b) = (a.checked(), b.checked());
    Value::from_id(builder::with(|bs| {
        let (a, b) = if T::DTYPE == DType::TF32 || T::DTYPE.is_narrow_float() {
            (crate::convert::convert(bs, a, DType::F32, loc), crate::convert::convert(bs, b, DType::F32, loc))
        } else {
            (a, b)
        };
        bs.cmp(pred, a, b, loc)
    }))
}

#[track_caller]
fn unary<T: Numeric>(op: UnaryOp, x: Value<T>) -> Value<T> {
    let loc = Location::caller();
    let x = x.checked();
    Value::from_id(builder::with(|b| b.unary(op, x, loc)))
}

#[track_caller]
pub(crate) fn unary_op<T: Numeric>(op: UnaryOp, x: Value<T>) -> Value<T> {
    unary(op, x)
}

#[track_caller]
pub(crate) fn binary_op<T: Numeric>(op: BinaryOp, a: Value<T>, b: Value<T>) -> Value<T> {
    binary(op, a, b)
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

    /// The high half of the full-width product (`mul.hi.{s,u}{32,64}`), on
    /// Int32, Uint32, Int64 or Uint64 operands.
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

/// Check that `n` parts of type `part` make one value of type `whole`.
#[track_caller]
fn check_packing(what: &str, part: DType, n: usize, whole: DType) {
    assert!(
        part != DType::Bool && whole != DType::Bool,
        "cute: {what}: Boolean values have no bit width in registers; use integers"
    );
    // Word-packed values (Int4, FP4, FP6, FP6x4) may fill only the low bits
    // of an integer, the rest zero.
    let partial = part.is_word_packed() && whole.is_int() && n >= 1 && n * part.bits() < whole.bits();
    assert!(n >= 2 || partial, "cute: {what} needs at least two parts, not {n}");
    assert!(
        partial || n * part.bits() == whole.bits(),
        "cute: {what}: {n} x {part} is {} bits, but {whole} is {}",
        n * part.bits(),
        whole.bits()
    );
}

/// The bits of `parts` concatenated into one value of type `R`, part 0 in
/// the low bits: a bitcast that changes the number of values. Two Float16
/// values packed into an Int32 are a `.f16x2` register (what CuTe DSL's
/// packed half-precision operations take as an Int32), and four Int8 values
/// in a Uint32 the bytes of a `.b32` register. The parts' widths must add
/// up to `R`'s. [`unpack`] is the inverse.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("pairs", || {
///     let x = param_tensor::<F16>("x", Layout::col_major(64));
///     let out = param_tensor::<i32>("out", Layout::col_major(32));
///     let t = thread_idx().x;
///     let pair: Int32 = pack([x.load(t * 2), x.load(t * 2 + 1)]);
///     out.store(t, pair);
/// });
/// ```
#[track_caller]
pub fn pack<P: Numeric, R: Numeric, const N: usize>(parts: [Value<P>; N]) -> Value<R> {
    let loc = Location::caller();
    check_packing("pack", P::DTYPE, N, R::DTYPE);
    let ids: Vec<ValueId> = parts.iter().map(|p| p.checked()).collect();
    Value::from_id(builder::with(|b| b.pack(ids, R::DTYPE, loc)))
}

/// The bits of `x` split into `N` values of type `P`, the first from the
/// low bits: the inverse of [`pack`]. `let [lo, hi]: [Float16; 2] =
/// unpack(bits);` reads the two halves of a `.f16x2` register.
#[track_caller]
pub fn unpack<P: Numeric, R: Numeric, const N: usize>(x: Value<R>) -> [Value<P>; N] {
    let loc = Location::caller();
    check_packing("unpack", P::DTYPE, N, R::DTYPE);
    let x = x.checked();
    let ids = builder::with(|b| b.unpack(x, P::DTYPE, N, loc));
    std::array::from_fn(|k| Value::from_id(ids[k]))
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

/// Materialise `e` as a value of `width` bits, with the facts known about
/// it: its bound and its largest power-of-two divisor.
#[track_caller]
fn materialize_index(e: &IndexExpr, width: u32) -> ValueId {
    let loc = Location::caller();
    let max_shift = if width == 64 { 62 } else { 30 };
    let divisor = (0..=max_shift).rev().map(|k| 1usize << k).find(|&d| e.is_multiple_of(d)).unwrap_or(1);
    // A bound is only kept for a value known to be non-negative: the facts
    // of a value are read as a range from 0.
    let bound = if e.is_non_negative() { e.bound() } else { None };
    let facts = ValueFacts { bound, divisor, pow2: false };
    builder::with(|b| {
        let id = b.materialize_at(e, width, loc);
        if b.constant_of(id).is_none() {
            let known = b.facts_of(id);
            // Keep a power-of-two fact the value already carries (from an
            // assumption, say).
            b.set_facts(id, ValueFacts { divisor: divisor.max(known.divisor), pow2: known.pow2, ..facts });
        }
        id
    })
}

/// An `i32` index expression as a value. One that is 64-bit
/// ([`IndexExpr::width`]) panics: it converts into an [`Int64`].
impl IntoValue<i32> for IndexExpr {
    #[track_caller]
    fn into_value(self) -> Value<i32> {
        assert!(
            self.width() == 32,
            "cute: the index expression {self:?} is 64-bit (an Int64 or a static value beyond i32 enters it), so \
             it converts into an Int64, not an Int32"
        );
        Value::from_id(materialize_index(&self, 32))
    }
}

/// An index expression as an `i64` value: computed in 32 bits as far as its
/// own values allow, and widened where it meets a 64-bit part.
impl IntoValue<i64> for IndexExpr {
    #[track_caller]
    fn into_value(self) -> Value<i64> {
        Value::from_id(materialize_index(&self, 64))
    }
}

impl IntoValue<i64> for &IndexExpr {
    #[track_caller]
    fn into_value(self) -> Value<i64> {
        self.clone().into_value()
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
    IndexExpr => Int32, i128 => Int128, u128 => Uint128, crate::numeric::I4 => crate::numeric::Int4
);

impl From<IndexExpr> for Int64 {
    #[track_caller]
    fn from(x: IndexExpr) -> Self {
        x.into_value()
    }
}

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

impl AnyValue {
    /// The value's type.
    #[track_caller]
    pub fn dtype(&self) -> DType {
        builder::with(|b| b.dtype(self.0))
    }

    /// The value as a `Value<T>`; panics unless its type is `T`.
    #[track_caller]
    pub fn cast<T: Numeric>(self) -> Value<T> {
        let t = self.dtype();
        assert!(t == T::DTYPE, "AnyValue::cast: the value is {}, not {}", t.cute_name(), T::DTYPE.cute_name());
        Value::from_id(self.0)
    }
}

/// A runtime value is assumed to be non-negative, as layout arithmetic
/// needs ([`IndexExpr::Dynamic`]); a constant is static, and a negative one
/// a negative constant. [`IndexExpr::signed`] takes a runtime value that may
/// be negative.
impl From<Int32> for IndexExpr {
    fn from(v: Int32) -> Self {
        if builder::value_facts(v.id).bound == Some(1) {
            return IndexExpr::Static(0);
        }
        if let Some(Literal::I32(c)) = builder::try_constant(v.id) {
            return IndexExpr::from_signed(c.into());
        }
        IndexExpr::Dynamic(v.id)
    }
}

impl IndexExpr {
    /// A runtime `Int32` or `Int64` that may be negative, as a signed term
    /// ([`IndexExpr::Signed`]): the K residue `K - bK * k_tiles` of a GEMM,
    /// say, which [`domain_offset`](crate::domain_offset) shifts a tensor
    /// by. A constant becomes a static integer of its sign. (Converting with
    /// `IndexExpr::from` instead assumes a runtime value is non-negative.)
    #[track_caller]
    pub fn signed<T: Integer>(v: Value<T>) -> IndexExpr {
        let id = v.checked();
        match builder::try_constant(id) {
            Some(Literal::I32(c)) => return IndexExpr::from_signed(c.into()),
            Some(Literal::I64(c)) => return IndexExpr::from_signed(c),
            _ => {}
        }
        assert!(
            matches!(T::DTYPE, DType::I32 | DType::I64),
            "IndexExpr::signed: a signed index term is an Int32 or an Int64, not a {}",
            T::DTYPE
        );
        IndexExpr::Signed(id)
    }
}

/// A 64-bit leaf: index expressions it enters are computed in 64 bits.
impl From<Int64> for IndexExpr {
    fn from(v: Int64) -> Self {
        if builder::value_facts(v.id).bound == Some(1) {
            return IndexExpr::Static(0);
        }
        if let Some(Literal::I64(c)) = builder::try_constant(v.id) {
            return IndexExpr::from_signed(c);
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

macro_rules! stride_from_value {
    ($($t:ty),*) => {$(
        impl From<$t> for crate::layout::Stride {
            fn from(v: $t) -> Self {
                Tree::Leaf(crate::layout::StrideLeaf::Int(v.into()))
            }
        }

        impl From<$t> for crate::layout::StrideLeaf {
            fn from(v: $t) -> Self {
                crate::layout::StrideLeaf::Int(v.into())
            }
        }

        impl From<$t> for crate::layout::BasisScale {
            fn from(v: $t) -> Self {
                crate::layout::BasisScale::Int(v.into())
            }
        }

        /// `scale * E(mode)` with a runtime scale.
        impl std::ops::Mul<crate::layout::StrideLeaf> for $t {
            type Output = crate::layout::StrideLeaf;
            #[track_caller]
            fn mul(self, rhs: crate::layout::StrideLeaf) -> crate::layout::StrideLeaf {
                rhs * IndexExpr::from(self)
            }
        }
    )*};
}
stride_from_value!(Int32, Int64);

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
host_lhs_ints!(i8, i16, i32, i64, u8, u16, u32, u64, i128, u128);

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

/// Integers [`assume`] and [`get_divisibility`] take: integer values
/// ([`Int32`], [`Int64`], ...), index expressions and static integers.
pub trait Assumable: Sized {
    /// `self` with the promise attached; `divby` and `pow2` are checked.
    #[doc(hidden)]
    fn assume_checked(self, divby: usize, pow2: bool, loc: crate::ir::Loc) -> Self;

    /// The largest number known to divide `self`.
    #[doc(hidden)]
    fn divisibility(&self) -> usize;
}

/// Check a promise about a static value while the kernel is built.
#[track_caller]
fn check_static_assumption(value: i128, divby: usize, pow2: bool) {
    assert!(
        value % divby as i128 == 0,
        "cute: assume: the static value {value} is not a multiple of {divby} (remainder {})",
        value % divby as i128
    );
    if pow2 {
        assert!(value > 0 && value & (value - 1) == 0, "cute: assume: the static value {value} is not a power of two");
    }
}

impl<T: Integer> Assumable for Value<T> {
    #[track_caller]
    fn assume_checked(self, divby: usize, pow2: bool, loc: crate::ir::Loc) -> Self {
        let x = self.checked();
        if let Some(c) = builder::try_constant(x) {
            check_static_assumption(int_literal_value(c), divby, pow2);
            return self;
        }
        Value::from_id(builder::with(|b| b.assume(x, divby as u64, pow2, loc)))
    }

    fn divisibility(&self) -> usize {
        match builder::try_constant(self.id) {
            Some(c) => usize::try_from(int_literal_value(c).unsigned_abs()).unwrap_or(usize::MAX),
            None => builder::value_facts(self.id).divisor,
        }
    }
}

/// The value of an integer literal, sign-extended for a signed type.
fn int_literal_value(c: Literal) -> i128 {
    let ty = c.dtype();
    let shift = 128 - ty.bits() as u32;
    let bits = c.bits128() as i128;
    if ty.is_signed() { (bits << shift) >> shift } else { bits }
}

impl Assumable for IndexExpr {
    /// A static value is checked; a runtime one becomes the assumption's
    /// result, an `i32` or an `i64` as the expression's
    /// [`width`](IndexExpr::width) says.
    #[track_caller]
    fn assume_checked(self, divby: usize, pow2: bool, loc: crate::ir::Loc) -> Self {
        if let IndexExpr::Static(v) = self {
            check_static_assumption(v as i128, divby, pow2);
            return self;
        }
        let x = match self {
            IndexExpr::Dynamic(v) => v,
            e => crate::tensor::index_operand(e),
        };
        IndexExpr::Dynamic(builder::with(|b| b.assume(x, divby as u64, pow2, loc)))
    }

    fn divisibility(&self) -> usize {
        self.known_divisor()
    }
}

macro_rules! assumable_host_ints {
    ($($t:ty),*) => {$(
        impl Assumable for $t {
            #[track_caller]
            fn assume_checked(self, divby: usize, pow2: bool, _loc: crate::ir::Loc) -> Self {
                check_static_assumption(self as i128, divby, pow2);
                self
            }

            fn divisibility(&self) -> usize {
                usize::try_from((*self as i128).unsigned_abs()).unwrap_or(usize::MAX)
            }
        }
    )*};
}
assumable_host_ints!(i32, i64, u32, u64, usize);

/// `src`, which the kernel promises is a multiple of `divby` and, with
/// `pow2`, a positive power of two: CuTe DSL's `cute.assume(src, divby,
/// pow2)`. The result carries the promise, which lets index arithmetic built
/// from it prove what it could not before, such as that a runtime offset is
/// aligned for a vector copy ([`get_divisibility`] reads it back). With no
/// `divby` and no `pow2` it returns `src` unchanged.
///
/// `divby` must be positive, and a power of two with `pow2`. A static `src`
/// (a host integer, a constant value or a static index expression) is
/// checked while the kernel is built. A runtime one becomes an
/// [`Op::Assume`](crate::ir::Op::Assume), which adds no instruction (a
/// power-of-two `divby` is passed on to LLVM as an `llvm.intr.assume`), and
/// the simulator reports a value that breaks the promise, which is undefined
/// behaviour on the GPU. An assumption whose result is never used is
/// removed with it.
#[track_caller]
pub fn assume<X: Assumable>(src: X, divby: Option<usize>, pow2: bool) -> X {
    if divby.is_none() && !pow2 {
        return src;
    }
    let divby = divby.unwrap_or(1);
    assert!(divby > 0, "cute: assume: divby must be a positive integer, not 0");
    assert!(!pow2 || divby.is_power_of_two(), "cute: assume: with pow2, divby must be a power of two, not {divby}");
    src.assume_checked(divby, pow2, Location::caller())
}

/// The largest number known to divide `x` (CuTe DSL's
/// `cute.get_divisibility`): a static integer's own value, and for a runtime
/// value what [`assume`], a [`param_extent`](crate::param_extent)'s
/// `multiple_of` and the arithmetic that built it prove, 1 when nothing is
/// known.
pub fn get_divisibility<X: Assumable>(x: &X) -> usize {
    x.divisibility()
}

//! The two float types the precise expansions are written for, and helpers
//! that mirror msun's word-access macros on runtime values.
//!
//! msun (FreeBSD's libm, from Sun's fdlibm) tests and edits floats through
//! their bit patterns: the high 32-bit word of a double (`GET_HIGH_WORD`,
//! `SET_LOW_WORD`, ...) or the whole word of a float (`GET_FLOAT_WORD`).
//! The float versions of its functions are the double versions with the
//! high word replaced by the float's word and the constants changed, so the
//! expansions here are written once, generically: [`hi`] is the high word of
//! an `f64` or the word of an `f32`, and each function keeps both types'
//! constants side by side.
//!
//! The expansions are branch-free where msun branches on the argument: every
//! path is computed and the right result selected, so no lane of a warp
//! diverges. Paths are evaluated on a safe stand-in for arguments they do
//! not handle, so that no integer shift or conversion goes out of range.

use crate::value::{Bitwise, Boolean, Float, Int32, Integer, IntoValue, Uint32, Value, select};

/// `f32` or `f64`.
pub(crate) trait Real: Float + IntoValue<Self> {
    /// The unsigned integer of the same width.
    type U: Integer + Bitwise + IntoValue<Self::U>;
    /// Whether this is `f64`.
    const DOUBLE: bool;
    /// Stored significand bits.
    const MANT: u32;
    /// Exponent bias.
    const BIAS: i32;
    /// A host constant (exact for `f64`, rounded for `f32`).
    fn of(x: f64) -> Self;
    /// A host constant of the bit type.
    fn u(bits: u64) -> Self::U;
    /// Exponent bits.
    fn exp_bits() -> u32 {
        if Self::DOUBLE { 11 } else { 8 }
    }
}

impl Real for f32 {
    type U = u32;
    const DOUBLE: bool = false;
    const MANT: u32 = 23;
    const BIAS: i32 = 127;
    fn of(x: f64) -> f32 {
        x as f32
    }
    fn u(bits: u64) -> u32 {
        bits as u32
    }
}

impl Real for f64 {
    type U = u64;
    const DOUBLE: bool = true;
    const MANT: u32 = 52;
    const BIAS: i32 = 1023;
    fn of(x: f64) -> f64 {
        x
    }
    fn u(bits: u64) -> u64 {
        bits
    }
}

/// The f32 with these bits, as an f64 (exact), for constant tables.
pub(crate) const fn fb(bits: u32) -> f64 {
    f32::from_bits(bits) as f64
}

/// The f64 with these bits.
pub(crate) const fn db(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// `m * 2^e`, for msun's hexadecimal float constants (`0xaaaaaa.0p-24`).
pub(crate) const fn xp(m: u64, e: i32) -> f64 {
    m as f64 * f64::from_bits(((e + 1023) as u64) << 52)
}

/// Pick the constant for `T` from `[f32 value, f64 value]`.
pub(crate) fn pick<T: Real, C: Copy>(per_type: [C; 2]) -> C {
    per_type[T::DOUBLE as usize]
}

/// A constant.
#[track_caller]
pub(crate) fn c<T: Real>(x: f64) -> Value<T> {
    Value::constant(T::of(x))
}

/// NaN, the result of every invalid operation and NaN argument.
#[track_caller]
pub(crate) fn nan<T: Real>() -> Value<T> {
    c(f64::NAN)
}

/// Infinity with the sign of `neg`.
#[track_caller]
pub(crate) fn inf<T: Real>(neg: bool) -> Value<T> {
    c(if neg { f64::NEG_INFINITY } else { f64::INFINITY })
}

/// The raw bits.
#[track_caller]
pub(crate) fn bits<T: Real>(x: Value<T>) -> Value<T::U> {
    x.bitcast()
}

/// The float with these bits.
#[track_caller]
pub(crate) fn from_bits<T: Real>(b: Value<T::U>) -> Value<T> {
    b.bitcast()
}

/// msun's `GET_HIGH_WORD` (f64) or `GET_FLOAT_WORD` (f32), signed.
#[track_caller]
pub(crate) fn hi<T: Real>(x: Value<T>) -> Int32 {
    if T::DOUBLE { (x.bitcast::<u64>() >> 32u64).to::<u32>().bitcast() } else { x.bitcast() }
}

/// msun's `GET_LOW_WORD` (f64; zero for f32).
#[track_caller]
pub(crate) fn lo<T: Real>(x: Value<T>) -> Uint32 {
    if T::DOUBLE { x.bitcast::<u64>().to::<u32>() } else { Value::constant(0u32) }
}

/// msun's `SET_HIGH_WORD` (f64: keeps the low word) or `SET_FLOAT_WORD`.
#[track_caller]
pub(crate) fn with_hi<T: Real>(x: Value<T>, h: Int32) -> Value<T> {
    if T::DOUBLE {
        let low = x.bitcast::<u64>() & 0xffff_ffffu64;
        ((h.bitcast::<u32>().to::<u64>() << 32u64) | low).bitcast()
    } else {
        h.bitcast()
    }
}

/// msun's `INSERT_WORDS(x, h, 0)` (f64) or `SET_FLOAT_WORD(x, h)`.
#[track_caller]
pub(crate) fn from_hi<T: Real>(h: Int32) -> Value<T> {
    if T::DOUBLE { (h.bitcast::<u32>().to::<u64>() << 32u64).bitcast() } else { h.bitcast() }
}

/// msun's `SET_LOW_WORD(x, 0)` for f64, or `SET_FLOAT_WORD(x, ix & f32_mask)`
/// for f32: the leading bits, for splitting a value into exact parts.
#[track_caller]
pub(crate) fn trunc_bits<T: Real>(x: Value<T>, f32_mask: u32) -> Value<T> {
    if T::DOUBLE {
        (x.bitcast::<u64>() & 0xffff_ffff_0000_0000u64).bitcast()
    } else {
        (x.bitcast::<u32>() & f32_mask).bitcast()
    }
}

/// `2^k` for `k` in the normal exponent range.
#[track_caller]
pub(crate) fn pow2<T: Real>(k: Int32) -> Value<T> {
    if T::DOUBLE {
        ((k + T::BIAS).to::<u64>() << 52u64).bitcast()
    } else {
        ((k + T::BIAS).bitcast::<u32>() << 23u32).bitcast()
    }
}

/// `y * 2^k` with a single rounding, for `k` between about `-2.1 * BIAS`
/// and `BIAS + 1`: two exact scalings by halves of `k`, as msun does for
/// results that would overflow `2^k` or underflow to a subnormal.
#[track_caller]
pub(crate) fn scale<T: Real>(y: Value<T>, k: Int32) -> Value<T> {
    let k1 = k >> 1;
    let k2 = k - k1;
    y * pow2::<T>(k1) * pow2::<T>(k2)
}

/// `c[0] + x * (c[1] + x * (c[2] + ...))`, with a separate multiply and add
/// per step, as msun's C evaluates it.
#[track_caller]
pub(crate) fn horner<T: Real>(x: Value<T>, coeffs: &[f64]) -> Value<T> {
    let (last, rest) = coeffs.split_last().expect("a polynomial has coefficients");
    rest.iter().rev().fold(c::<T>(*last), |acc, &k| acc * x + T::of(k))
}

/// The NaN test `x != x`.
#[track_caller]
pub(crate) fn is_nan<T: Real>(x: Value<T>) -> Boolean {
    x.ne(x)
}

/// The final result: NaN results are the one quiet NaN, so the GPU (whose
/// single-precision NaNs are unspecified) and the simulator agree bit for
/// bit.
#[track_caller]
pub(crate) fn canonical<T: Real>(x: Value<T>) -> Value<T> {
    select(is_nan(x), nan::<T>(), x)
}

/// `if cond { a } else { b }` on values.
#[track_caller]
pub(crate) fn sel<T: crate::value::Numeric>(cond: Boolean, a: Value<T>, b: Value<T>) -> Value<T> {
    select(cond, a, b)
}

/// `x` where `ok` holds, else `fallback`: keeps a path's argument in the
/// range it handles.
#[track_caller]
pub(crate) fn guard<T: Real>(ok: Boolean, x: Value<T>, fallback: f64) -> Value<T> {
    select(ok, x, c(fallback))
}

/// Is the sign bit set (true for -0)?
#[track_caller]
pub(crate) fn sign_bit<T: Real>(x: Value<T>) -> Boolean {
    hi(x).lt(0)
}

/// `|x|`, clearing the sign bit with an integer operation that LLVM sees
/// through (unlike [`Value::abs`], which is opaque inline PTX): it may
/// emit `and.b32` or turn it into `abs.f32`, which gives a NaN the
/// canonical NaN on the GPU. Either serves here, because the expansions
/// return [`canonical`] NaNs.
#[track_caller]
pub(crate) fn fabs<T: Real>(x: Value<T>) -> Value<T> {
    let magnitude = Value::constant(T::u((1u64 << (T::MANT + T::exp_bits())) - 1));
    from_bits(bits(x) & magnitude)
}

/// `-x`, as the subtraction `-0 - x` (equal to `-x` for every number),
/// which LLVM turns into `fneg` and folds into its neighbours: `fma(-a, b,
/// c)` stays one instruction. ([`Value`]'s `-` is a sign-bit flip in inline
/// PTX, which LLVM cannot fold.) A NaN `x` gives the subtraction's NaN, or
/// on the GPU whatever LLVM makes of `fneg`; the expansions return
/// [`canonical`] NaNs, so either serves.
#[track_caller]
pub(crate) fn fneg<T: Real>(x: Value<T>) -> Value<T> {
    c::<T>(-0.0) - x
}

/// `x` with the sign of `neg` applied: `-x` where `neg` holds.
#[track_caller]
pub(crate) fn negate_if<T: Real>(neg: Boolean, x: Value<T>) -> Value<T> {
    select(neg, fneg(x), x)
}

/// `(s, e)` with `s + e = a + b` exactly (Knuth's TwoSum).
#[track_caller]
pub(crate) fn two_sum<T: Real>(a: Value<T>, b: Value<T>) -> (Value<T>, Value<T>) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `(s, e)` with `s + e = a + b` exactly, for `|a| >= |b|` (Dekker's
/// FastTwoSum).
#[track_caller]
pub(crate) fn fast_two_sum<T: Real>(a: Value<T>, b: Value<T>) -> (Value<T>, Value<T>) {
    let s = a + b;
    (s, b - (s - a))
}

/// `(ah + al) / (bh + bl)` as an unevaluated sum, to about twice the
/// working precision: the quotient's residual `ah - q bh` is exact with
/// fma.
#[track_caller]
pub(crate) fn dd_div<T: Real>(ah: Value<T>, al: Value<T>, bh: Value<T>, bl: Value<T>) -> (Value<T>, Value<T>) {
    let q = ah / bh;
    let r = (crate::value::fma(fneg(q), bh, ah) + al) - q * bl;
    fast_two_sum(q, r / bh)
}

/// Integer power-of-two test helpers use plain `i32` constants; msun writes
/// some high words above `0x7fffffff`, which this converts.
pub(crate) const fn w(word: u32) -> i32 {
    word as i32
}

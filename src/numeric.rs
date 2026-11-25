//! The narrow numeric types and their exact codecs.
//!
//! Host representations of CuTe DSL's `TFloat32`, the FP8 formats
//! (`Float8E4M3FN`, `Float8E5M2`, `Float8E4M3`, `Float8E4M3B11FNUZ`), the
//! scale-factor formats (`Float8E8M0FNU`, `FloatNV8E5M3FNU`) and `Int4`,
//! with the codec the simulator shares: every narrow float decodes exactly
//! to f64, and encodes from f64 with PTX `cvt`'s rounding modes and its
//! `.satfinite` and `.relu` modifiers, from the format's row of the type
//! table ([`DType::float_format`]). The codecs follow the OCP 8-bit
//! floating point specification and the PTX ISA's descriptions of the
//! formats and of `cvt`.

use std::fmt;

use crate::ir::{DType, FloatFormat, Literal, NanEncoding};
use crate::value::{Numeric, Value};

/// A rounding direction of a narrowing conversion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NarrowRounding {
    /// To nearest, ties to even (`.rn`).
    NearestEven,
    /// Toward zero (`.rz`).
    Zero,
    /// Toward positive infinity (`.rp`).
    Up,
    /// To nearest, ties away from zero (`.rna`).
    NearestAway,
}

/// `2^k` for `k` in f64's normal range.
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

const fn exp_mask(f: FloatFormat) -> u32 {
    (1 << f.exp_bits) - 1
}

/// The encoding of the largest finite value, without the sign.
pub(crate) const fn max_finite_bits(f: FloatFormat) -> u32 {
    let mant = (1u32 << f.mant_bits) - 1;
    match f.nan {
        // The largest exponent is infinity and NaN.
        NanEncoding::Ieee => (exp_mask(f) - 1) << f.mant_bits | mant,
        // All ones is NaN; with no mantissa that is the whole exponent.
        NanEncoding::AllOnes if f.mant_bits == 0 => exp_mask(f) - 1,
        NanEncoding::AllOnes => exp_mask(f) << f.mant_bits | (mant - 1),
        NanEncoding::NegativeZero | NanEncoding::None => exp_mask(f) << f.mant_bits | mant,
    }
}

/// The NaN a conversion into the format produces; for a format without NaN
/// (FP4, FP6), `+MAX_NORM`, as `cvt`'s `.satfinite` gives for a NaN.
pub(crate) const fn canonical_nan(f: FloatFormat) -> u32 {
    let mag = (1u32 << (f.exp_bits + f.mant_bits)) - 1;
    match f.nan {
        NanEncoding::Ieee | NanEncoding::AllOnes => mag,
        NanEncoding::NegativeZero => 1 << (f.exp_bits + f.mant_bits),
        NanEncoding::None => max_finite_bits(f),
    }
}

fn sign_bit(f: FloatFormat) -> u32 {
    if f.signed { 1 << (f.exp_bits + f.mant_bits) } else { 0 }
}

/// Whether `bits` is a NaN of the format.
pub(crate) fn is_nan(f: FloatFormat, bits: u32) -> bool {
    let mag = bits & ((1 << (f.exp_bits + f.mant_bits)) - 1);
    let (exp, mant) = (mag >> f.mant_bits, mag & ((1 << f.mant_bits) - 1));
    match f.nan {
        NanEncoding::Ieee => exp == exp_mask(f) && mant != 0,
        NanEncoding::AllOnes => mag == (1 << (f.exp_bits + f.mant_bits)) - 1,
        NanEncoding::NegativeZero => bits == sign_bit(f),
        NanEncoding::None => false,
    }
}

/// The exact value of an encoding of a format of at most 32 bits (NaN for
/// every NaN encoding).
pub(crate) fn decode(f: FloatFormat, bits: u32) -> f64 {
    if is_nan(f, bits) {
        return f64::NAN;
    }
    let neg = bits & sign_bit(f) != 0;
    let mag = bits & ((1 << (f.exp_bits + f.mant_bits)) - 1);
    let (exp, mant) = (mag >> f.mant_bits, mag & ((1 << f.mant_bits) - 1));
    let m = f.mant_bits as i32;
    let v = if f.has_inf && exp == exp_mask(f) {
        f64::INFINITY
    } else if exp == 0 && f.has_zero {
        mant as f64 * pow2(1 - f.bias - m)
    } else {
        ((1u64 << m) + mant as u64) as f64 * pow2(exp as i32 - f.bias - m)
    };
    if neg { -v } else { v }
}

/// `x` converted to the format, as PTX `cvt` does: rounded in direction
/// `rounding` as if the exponent range were unbounded; a magnitude past the
/// largest finite value becomes that value with `satfinite`, else infinity
/// (to-nearest, or a directed rounding away from zero) or the largest finite
/// value (directed rounding toward zero), and NaN in a format without
/// infinities. NaN gives the format's NaN. An unsigned format ignores the
/// sign, and with `relu` a negative result becomes `+0` (a NaN stays NaN).
pub(crate) fn encode(f: FloatFormat, x: f64, rounding: NarrowRounding, satfinite: bool, relu: bool) -> u32 {
    if x.is_nan() {
        return canonical_nan(f);
    }
    let neg = f.signed && x.is_sign_negative();
    if relu && neg {
        return 0;
    }
    let a = x.abs();
    let sign = if neg { sign_bit(f) } else { 0 };
    let max_bits = max_finite_bits(f);
    let max = decode(f, max_bits);
    let inf_or_nan = if f.has_inf { sign | exp_mask(f) << f.mant_bits } else { canonical_nan(f) };
    if a.is_infinite() {
        return if satfinite { sign | max_bits } else { inf_or_nan };
    }
    // The magnitude rounds toward zero for `.rz`, and for `.rp` on a
    // negative value.
    let toward_zero = match rounding {
        NarrowRounding::Zero => true,
        NarrowRounding::Up => neg,
        _ => false,
    };
    if a == 0.0 {
        // FNUZ formats have no negative zero; E8M0 has no zero at all, and
        // gives its smallest value.
        return if f.nan == NanEncoding::NegativeZero || !f.has_zero { 0 } else { sign };
    }
    let m = f.mant_bits as i32;
    let emin = if f.has_zero { 1 - f.bias } else { -f.bias };
    // The exponent of `a`'s leading bit (`a` is at least 2^-1074).
    let e = if a >= f64::MIN_POSITIVE {
        ((a.to_bits() >> 52) as i32) - 1023
    } else {
        -1023 - (a.to_bits().leading_zeros() as i32 - 12)
    };
    let q = e.max(emin) - m;
    // `a / 2^q` is exact: a power-of-two scaling within range.
    let r = if q >= -1022 { a / pow2(q) } else { a * pow2(-q) };
    let floor = r.floor();
    let frac = r - floor;
    // Whether the encoding below is odd: its mantissa's last bit, or with
    // no mantissa (E8M0) its exponent's.
    let odd = if m == 0 { (q + f.bias) & 1 == 1 } else { floor % 2.0 == 1.0 };
    let up = match rounding {
        NarrowRounding::NearestEven => frac > 0.5 || (frac == 0.5 && odd),
        NarrowRounding::NearestAway => frac >= 0.5,
        NarrowRounding::Zero => false,
        NarrowRounding::Up => !neg && frac > 0.0,
    };
    let mut n = floor as u64 + up as u64;
    let mut q = q;
    if n == 0 {
        if !f.has_zero {
            return 0;
        }
        return if f.nan == NanEncoding::NegativeZero { 0 } else { sign };
    }
    if n == 1u64 << (m + 1) {
        n >>= 1;
        q += 1;
    }
    let value = n as f64 * if q >= -1022 { pow2(q) } else { 1.0 / pow2(-q) };
    if value > max {
        // A directed rounding toward zero stops at the largest finite value.
        return if satfinite || toward_zero { sign | max_bits } else { inf_or_nan };
    }
    let bits = if n < 1u64 << m {
        // Subnormal.
        n as u32
    } else {
        let biased = (q + m + f.bias) as u32;
        biased << f.mant_bits | (n as u32 & ((1 << f.mant_bits) - 1))
    };
    sign | bits
}

/// `cvt.rna.tf32.f32`: the f32 bits rounded to 10 stored mantissa bits,
/// ties away from zero, the low 13 bits cleared (as the simulator runs it,
/// checked on SM89 by `tests/gpu.rs`). An infinity or NaN keeps its bits but
/// the low 13, and a value that rounds past the largest finite value
/// becomes infinity.
pub(crate) fn tf32_rna_bits(x: u32) -> u32 {
    let rounded = if x >> 23 & 0xff == 0xff { x } else { x + 0x1000 };
    rounded & !0x1fff
}

macro_rules! narrow_float {
    ($(#[$doc:meta])* $name:ident, $alias:ident, $dtype:ident, $rounding:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(pub u8);

        impl $name {
            const FORMAT: FloatFormat = match DType::$dtype.float_format() {
                Some(f) => f,
                None => unreachable!(),
            };

            /// The largest finite value.
            pub const MAX: $name = $name(max_finite_bits(Self::FORMAT) as u8 | sign_bit_const(Self::FORMAT, false));
            /// The most negative finite value (the smallest value of an
            /// unsigned format).
            pub const MIN: $name = $name(if Self::FORMAT.signed {
                max_finite_bits(Self::FORMAT) as u8 | sign_bit_const(Self::FORMAT, true)
            } else {
                0
            });
            /// The format's NaN.
            pub const NAN: $name = $name(canonical_nan(Self::FORMAT) as u8);

            /// `x` rounded to this format on the host: to nearest, ties to
            /// even (toward positive infinity for E8M0, which has no
            /// round-to-nearest conversion), saturating to the largest
            /// finite value as `.satfinite` does; NaN gives NaN.
            pub fn from_f32(x: f32) -> $name {
                $name(encode(Self::FORMAT, x as f64, $rounding, true, false) as u8)
            }

            /// `x` converted with an explicit rounding direction and PTX
            /// `cvt`'s `.satfinite` and `.relu` modifiers.
            pub fn from_f64_with(x: f64, rounding: NarrowRounding, satfinite: bool, relu: bool) -> $name {
                $name(encode(Self::FORMAT, x, rounding, satfinite, relu) as u8)
            }

            /// The exact value (NaN for every NaN encoding).
            pub fn to_f32(self) -> f32 {
                decode(Self::FORMAT, self.0 as u32) as f32
            }

            /// The exact value.
            pub fn to_f64(self) -> f64 {
                decode(Self::FORMAT, self.0 as u32)
            }

            pub fn is_nan(self) -> bool {
                is_nan(Self::FORMAT, self.0 as u32)
            }
        }

        impl From<f32> for $name {
            fn from(x: f32) -> Self {
                $name::from_f32(x)
            }
        }

        impl From<$name> for f32 {
            fn from(x: $name) -> f32 {
                x.to_f32()
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", self.to_f64())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_f64())
            }
        }

        impl Numeric for $name {
            const DTYPE: DType = DType::$dtype;
            fn literal(self) -> Literal {
                Literal::F8(DType::$dtype, self.0)
            }
            fn from_bits(bits: u64) -> Self {
                $name((bits & ((1 << DType::$dtype.bits()) - 1)) as u8)
            }
        }

        #[doc = concat!("CuTe DSL's `cutlass.", stringify!($alias), "`: a runtime value of [`", stringify!($name), "`].")]
        pub type $alias = Value<$name>;
    };
}

const fn sign_bit_const(f: FloatFormat, neg: bool) -> u8 {
    if f.signed && neg { 1 << (f.exp_bits + f.mant_bits) } else { 0 }
}

narrow_float!(
    /// OCP E4M3 (CuTe DSL's `Float8E4M3FN`), as stored on the host: 4
    /// exponent bits (bias 7) and 3 mantissa bits, no infinities, NaN
    /// `S.1111.111`, largest finite value 448.
    F8E4M3FN, Float8E4M3FN, F8E4M3FN, NarrowRounding::NearestEven
);
narrow_float!(
    /// OCP E5M2 (CuTe DSL's `Float8E5M2`), as stored on the host: IEEE-style
    /// with 5 exponent bits (bias 15) and 2 mantissa bits, largest finite
    /// value 57344.
    F8E5M2, Float8E5M2, F8E5M2, NarrowRounding::NearestEven
);
narrow_float!(
    /// IEEE-style E4M3 with infinities (CuTe DSL's `Float8E4M3`), as stored
    /// on the host: largest finite value 240. Kernels can read it but not
    /// convert into it.
    F8E4M3, Float8E4M3, F8E4M3, NarrowRounding::NearestEven
);
narrow_float!(
    /// E4M3 with exponent bias 11 (CuTe DSL's `Float8E4M3B11FNUZ`), as stored
    /// on the host: no infinities, no negative zero, NaN `0x80`, largest
    /// finite value 30. Kernels can read it but not convert into it.
    F8E4M3B11FNUZ, Float8E4M3B11FNUZ, F8E4M3B11FNUZ, NarrowRounding::NearestEven
);
narrow_float!(
    /// The MX scale factor UE8M0 (CuTe DSL's `Float8E8M0FNU`), as stored on
    /// the host: `2^(e - 127)` for `e` below 255, NaN `0xFF`; no sign, zero
    /// or mantissa.
    F8E8M0FNU, Float8E8M0FNU, F8E8M0FNU, NarrowRounding::Up
);
narrow_float!(
    /// NVIDIA's unsigned scale factor UE5M3 (CuTe DSL's `FloatNV8E5M3FNU`),
    /// as stored on the host: 5 exponent bits (bias 15), 3 mantissa bits,
    /// no sign, no infinities, NaN `0xFF`, largest finite value 114688.
    F8E5M3FNU, FloatNV8E5M3FNU, F8E5M3FNU, NarrowRounding::NearestEven
);

narrow_float!(
    /// OCP MX FP4 E2M1 (CuTe DSL's `Float4E2M1FN`), as stored on the host in
    /// the low 4 bits of a byte: 2 exponent bits (bias 1) and 1 mantissa bit,
    /// magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6, no infinities or NaN (so
    /// `NAN` is `+6`, what `.satfinite` makes of a NaN). Tensors store it
    /// packed, two per byte.
    F4E2M1FN, Float4E2M1FN, F4E2M1FN, NarrowRounding::NearestEven
);
narrow_float!(
    /// OCP MX FP6 E2M3 (CuTe DSL's `Float6E2M3FN`), as stored on the host in
    /// the low 6 bits of a byte: 2 exponent bits (bias 1), 3 mantissa bits,
    /// largest finite value 7.5, no infinities or NaN (`NAN` is `+7.5`).
    /// Tensors store it packed, four per three bytes.
    F6E2M3FN, Float6E2M3FN, F6E2M3FN, NarrowRounding::NearestEven
);
narrow_float!(
    /// OCP MX FP6 E3M2 (CuTe DSL's `Float6E3M2FN`), as stored on the host in
    /// the low 6 bits of a byte: 3 exponent bits (bias 3), 2 mantissa bits,
    /// largest finite value 28, no infinities or NaN (`NAN` is `+28`).
    F6E3M2FN, Float6E3M2FN, F6E3M2FN, NarrowRounding::NearestEven
);

macro_rules! packed_view {
    ($(#[$doc:meta])* $name:ident, $alias:ident, $lane:ident, $n:expr, $repr:ty, $literal:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
        #[repr(transparent)]
        pub struct $name(pub $repr);

        impl $name {
            /// The lanes packed, lane 0 in the low bits.
            pub fn new(lanes: [$lane; $n]) -> $name {
                let w = DType::$lane.bits();
                $name(lanes.iter().enumerate().fold(0, |acc, (k, l)| acc | (l.0 as $repr) << (k * w)))
            }

            /// The lanes, lane 0 from the low bits.
            pub fn lanes(self) -> [$lane; $n] {
                let w = DType::$lane.bits();
                std::array::from_fn(|k| <$lane as Numeric>::from_bits((self.0 >> (k * w)) as u64))
            }
        }

        impl Numeric for $name {
            const DTYPE: DType = DType::$name;
            fn literal(self) -> Literal {
                Literal::$literal(DType::$name, self.0)
            }
            fn from_bits(bits: u64) -> Self {
                $name((bits & ((1 << DType::$name.bits()) - 1)) as $repr)
            }
        }

        #[doc = concat!("CuTe DSL's `cutlass.", stringify!($alias), "`: a runtime value of [`", stringify!($name), "`].")]
        pub type $alias = Value<$name>;
    };
}

packed_view!(
    /// CuTe DSL's `Float4E2M1FNx2` on the host (`torch.float4_e2m1fn_x2`):
    /// two FP4 E2M1 values in a byte, lane 0 in the low nibble. A tensor of
    /// it counts packed bytes; `recast_tensor` views it as twice as many
    /// `F4E2M1FN` values.
    F4E2M1FNx2, Float4E2M1FNx2, F4E2M1FN, 2, u8, F8
);
packed_view!(
    /// CuTe DSL's `Float6E2M3FNx4` on the host: four FP6 E2M3 values in three
    /// bytes (the low 24 bits), lane 0 in the low bits. A tensor of it counts
    /// packed units; `recast_tensor` views it as four times as many
    /// `F6E2M3FN` values.
    F6E2M3FNx4, Float6E2M3FNx4, F6E2M3FN, 4, u32, F6x4
);
packed_view!(
    /// CuTe DSL's `Float6E3M2FNx4` on the host: four FP6 E3M2 values in three
    /// bytes, lane 0 in the low bits.
    F6E3M2FNx4, Float6E3M2FNx4, F6E3M2FN, 4, u32, F6x4
);

/// TFloat32 (CuTe DSL's `TFloat32`), as stored on the host: an f32 whose low
/// 13 mantissa bits are meant to be zero.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct TF32(pub f32);

impl TF32 {
    /// `x` rounded to TF32 as `cvt.rna.tf32.f32` does: to nearest, ties away
    /// from zero.
    pub fn from_f32(x: f32) -> TF32 {
        TF32(f32::from_bits(tf32_rna_bits(x.to_bits())))
    }

    pub fn to_f32(self) -> f32 {
        self.0
    }
}

impl From<f32> for TF32 {
    fn from(x: f32) -> Self {
        TF32::from_f32(x)
    }
}

impl From<TF32> for f32 {
    fn from(x: TF32) -> f32 {
        x.0
    }
}

impl fmt::Debug for TF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl fmt::Display for TF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Numeric for TF32 {
    const DTYPE: DType = DType::TF32;
    fn literal(self) -> Literal {
        Literal::TF32(self.0.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        TF32(f32::from_bits(bits as u32))
    }
}

/// CuTe DSL's `cutlass.TFloat32`: a runtime value of [`TF32`]. Arithmetic on
/// it is computed in Float32 (with a Float32 result).
pub type TFloat32 = Value<TF32>;

/// A 4-bit signed integer (CuTe DSL's `Int4`), as stored on the host: -8 to
/// 7, held in an `i8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct I4(i8);

impl I4 {
    pub const MIN: I4 = I4(-8);
    pub const MAX: I4 = I4(7);
    pub const ZERO: I4 = I4(0);

    /// `v`, which must be in -8..=7.
    #[track_caller]
    pub fn new(v: i8) -> I4 {
        assert!((-8..=7).contains(&v), "cute: {v} does not fit in an Int4 (-8 to 7)");
        I4(v)
    }

    /// `v` wrapped to 4 bits (its low 4 bits, sign-extended).
    pub fn wrapping(v: i64) -> I4 {
        I4(((v as u8) << 4) as i8 >> 4)
    }

    pub fn get(self) -> i8 {
        self.0
    }
}

impl fmt::Debug for I4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for I4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Numeric for I4 {
    const DTYPE: DType = DType::I4;
    fn literal(self) -> Literal {
        Literal::I4(self.0)
    }
    fn from_bits(bits: u64) -> Self {
        I4::wrapping(bits as i64)
    }
}

impl crate::value::Arith for I4 {}
impl crate::value::Integer for I4 {}
impl crate::value::Bitwise for I4 {}

/// CuTe DSL's `cutlass.Int4`: a runtime value of [`I4`]. It converts to and
/// from every type, and its arithmetic wraps at 4 bits; until sub-byte
/// tensors it cannot be loaded or stored (convert it to Int8).
pub type Int4 = Value<I4>;

impl From<I4> for i32 {
    fn from(x: I4) -> i32 {
        x.0 as i32
    }
}

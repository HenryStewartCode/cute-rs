//! The kernel intermediate representation.
//!
//! A kernel is a tree of structured regions (like MLIR's `scf` dialect) over
//! SSA scalar values. Memory is addressed through buffers: global buffers are
//! kernel parameters, shared buffers are per-block allocations and local
//! buffers are per-thread arrays. Register tensors never appear here: the
//! builder keeps them as arrays of SSA values, which is exactly what they
//! become after register allocation.

use std::fmt;
use std::panic::Location;

/// Source location of the user code that created an instruction.
pub type Loc = &'static Location<'static>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BufferId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Scalar element types.
///
/// Each type's facts (width, signedness, float format, MLIR storage type and
/// names) live in one table row, [`DType::info`], which the emitter, the
/// simulator's codecs and the promotion rules all read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    /// CuTe DSL's `Int4`: a 4-bit signed integer (-8 to 7), held in a
    /// sign-extended `i8` in MLIR and as a 4-bit pattern in the simulator.
    I4,
    /// CuTe DSL's `Int128`.
    I128,
    /// CuTe DSL's `Uint128`.
    U128,
    /// CuTe DSL's `TFloat32`: a Float32 with 10 mantissa bits, stored and
    /// emitted as `f32` bits (LLVM has no `tf32` type).
    TF32,
    /// CuTe DSL's `Float8E4M3FN` (OCP E4M3: no infinities, NaN `S.1111.111`,
    /// largest finite value 448).
    F8E4M3FN,
    /// CuTe DSL's `Float8E5M2` (OCP E5M2: IEEE-style, largest finite value
    /// 57344).
    F8E5M2,
    /// CuTe DSL's `Float8E4M3`: IEEE-style E4M3 with infinities (largest
    /// finite value 240). A storage type: conversions into it are refused.
    F8E4M3,
    /// CuTe DSL's `Float8E4M3B11FNUZ`: exponent bias 11, NaN `0x80`, no
    /// infinities and no negative zero (largest finite value 30). A storage
    /// type: conversions into it are refused.
    F8E4M3B11FNUZ,
    /// CuTe DSL's `Float8E8M0FNU`: the MX scale factor, `2^(e - 127)`, with
    /// NaN `0xFF`, no sign, no zero and no mantissa.
    F8E8M0FNU,
    /// CuTe DSL's `FloatNV8E5M3FNU`: NVIDIA's unsigned scale factor with 5
    /// exponent bits (bias 15) and 3 mantissa bits, NaN `0xFF` and no
    /// infinities (largest finite value 114688).
    F8E5M3FNU,
    /// CuTe DSL's `Float4E2M1FN` (OCP MX FP4 E2M1): magnitudes 0, 0.5, 1,
    /// 1.5, 2, 3, 4 and 6, no infinities or NaN. A sub-byte storage type.
    F4E2M1FN,
    /// CuTe DSL's `Float6E2M3FN` (OCP MX FP6 E2M3): largest finite value
    /// 7.5, no infinities or NaN. A sub-byte storage type.
    F6E2M3FN,
    /// CuTe DSL's `Float6E3M2FN` (OCP MX FP6 E3M2): largest finite value 28,
    /// no infinities or NaN. A sub-byte storage type.
    F6E3M2FN,
    /// CuTe DSL's `Float4E2M1FNx2`: two FP4 E2M1 values packed in a byte
    /// (`torch.float4_e2m1fn_x2`), a tensor element counting packed units.
    F4E2M1FNx2,
    /// CuTe DSL's `Float6E2M3FNx4`: four FP6 E2M3 values packed in three
    /// bytes, a tensor element counting packed units.
    F6E2M3FNx4,
    /// CuTe DSL's `Float6E3M2FNx4`: four FP6 E3M2 values packed in three
    /// bytes.
    F6E3M2FNx4,
}

/// How a float format encodes NaN.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NanEncoding {
    /// IEEE 754: the largest exponent with a nonzero mantissa.
    Ieee,
    /// Only the all-ones magnitude (`S.1111.111` for E4M3FN, `0xFF` for
    /// the unsigned formats): the "FN" formats, whose largest exponent
    /// otherwise holds finite values.
    AllOnes,
    /// The negative-zero pattern (`0x80`): the "FNUZ" formats.
    NegativeZero,
    /// No NaN at all: the OCP MX FP4 and FP6 formats, whose largest
    /// exponent holds finite values.
    None,
}

/// The bit layout and special values of a float type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FloatFormat {
    pub exp_bits: u32,
    /// Stored mantissa (fraction) bits.
    pub mant_bits: u32,
    pub bias: i32,
    /// Whether there is a sign bit ("U" formats have none).
    pub signed: bool,
    /// Whether the largest exponent with a zero mantissa is an infinity.
    pub has_inf: bool,
    pub nan: NanEncoding,
    /// Whether there are subnormals and a zero (E8M0 has neither).
    pub has_zero: bool,
}

impl FloatFormat {
    const fn ieee(exp_bits: u32, mant_bits: u32) -> FloatFormat {
        FloatFormat {
            exp_bits,
            mant_bits,
            bias: (1 << (exp_bits - 1)) - 1,
            signed: true,
            has_inf: true,
            nan: NanEncoding::Ieee,
            has_zero: true,
        }
    }

    /// The width of an encoding, sign included.
    pub const fn bits(self) -> u32 {
        self.exp_bits + self.mant_bits + self.signed as u32
    }

    /// Finite only: no infinities ("FN").
    pub const fn finite_only(self) -> bool {
        !self.has_inf
    }
}

/// One row of the type table: everything cute-rs needs to know about a
/// scalar type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DTypeInfo {
    /// Bits of one value (CuTe DSL's `T.width`).
    pub width: u32,
    pub signed: bool,
    /// The float format, for float types.
    pub float: Option<FloatFormat>,
    /// The MLIR type values are stored and computed as (integers are
    /// signless; FP8 formats are bytes).
    pub mlir: &'static str,
    /// The name in diagnostics and IR listings.
    pub name: &'static str,
    /// CuTe DSL's class name (`cutlass.Float16`).
    pub cute_name: &'static str,
}

const fn int_info(width: u32, signed: bool, mlir: &'static str, name: &'static str, cute: &'static str) -> DTypeInfo {
    DTypeInfo { width, signed, float: None, mlir, name, cute_name: cute }
}

const fn float_info(
    width: u32,
    f: FloatFormat,
    mlir: &'static str,
    name: &'static str,
    cute: &'static str,
) -> DTypeInfo {
    DTypeInfo { width, signed: f.signed, float: Some(f), mlir, name, cute_name: cute }
}

const E4M3FN: FloatFormat = FloatFormat {
    exp_bits: 4,
    mant_bits: 3,
    bias: 7,
    signed: true,
    has_inf: false,
    nan: NanEncoding::AllOnes,
    has_zero: true,
};
const E4M3B11FNUZ: FloatFormat = FloatFormat {
    exp_bits: 4,
    mant_bits: 3,
    bias: 11,
    signed: true,
    has_inf: false,
    nan: NanEncoding::NegativeZero,
    has_zero: true,
};
const E8M0FNU: FloatFormat = FloatFormat {
    exp_bits: 8,
    mant_bits: 0,
    bias: 127,
    signed: false,
    has_inf: false,
    nan: NanEncoding::AllOnes,
    has_zero: false,
};
/// The OCP MX FP4 and FP6 formats: signed, finite only, with zero and
/// subnormals, and no NaN.
const fn mx_float(exp_bits: u32, mant_bits: u32) -> FloatFormat {
    FloatFormat {
        exp_bits,
        mant_bits,
        bias: (1 << (exp_bits - 1)) - 1,
        signed: true,
        has_inf: false,
        nan: NanEncoding::None,
        has_zero: true,
    }
}
const E2M1: FloatFormat = mx_float(2, 1);
const E2M3: FloatFormat = mx_float(2, 3);
const E3M2: FloatFormat = mx_float(3, 2);

const E5M3FNU: FloatFormat = FloatFormat {
    exp_bits: 5,
    mant_bits: 3,
    bias: 15,
    signed: false,
    has_inf: false,
    nan: NanEncoding::AllOnes,
    has_zero: true,
};

impl DType {
    /// Every type, in the order of CuTe DSL's `ALL_DTYPES` (with Boolean).
    pub const ALL: [DType; 29] = [
        DType::Bool,
        DType::I4,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::I128,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::U128,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::TF32,
        DType::F64,
        DType::F8E5M2,
        DType::F8E4M3,
        DType::F8E4M3FN,
        DType::F8E8M0FNU,
        DType::F8E4M3B11FNUZ,
        DType::F4E2M1FN,
        DType::F6E2M3FN,
        DType::F6E3M2FN,
        DType::F4E2M1FNx2,
        DType::F6E2M3FNx4,
        DType::F6E3M2FNx4,
        DType::F8E5M3FNU,
    ];

    /// This type's row of the type table.
    pub const fn info(self) -> DTypeInfo {
        match self {
            DType::Bool => int_info(1, true, "i1", "i1", "Boolean"),
            DType::I4 => int_info(4, true, "i8", "i4", "Int4"),
            DType::I8 => int_info(8, true, "i8", "i8", "Int8"),
            DType::I16 => int_info(16, true, "i16", "i16", "Int16"),
            DType::I32 => int_info(32, true, "i32", "i32", "Int32"),
            DType::I64 => int_info(64, true, "i64", "i64", "Int64"),
            DType::I128 => int_info(128, true, "i128", "i128", "Int128"),
            DType::U8 => int_info(8, false, "i8", "u8", "Uint8"),
            DType::U16 => int_info(16, false, "i16", "u16", "Uint16"),
            DType::U32 => int_info(32, false, "i32", "u32", "Uint32"),
            DType::U64 => int_info(64, false, "i64", "u64", "Uint64"),
            DType::U128 => int_info(128, false, "i128", "u128", "Uint128"),
            DType::F16 => float_info(16, FloatFormat::ieee(5, 10), "f16", "f16", "Float16"),
            DType::BF16 => float_info(16, FloatFormat::ieee(8, 7), "bf16", "bf16", "BFloat16"),
            DType::F32 => float_info(32, FloatFormat::ieee(8, 23), "f32", "f32", "Float32"),
            // Stored as f32 bits: the format is f32's, and the conversion
            // into TF32 keeps 10 mantissa bits.
            DType::TF32 => float_info(32, FloatFormat::ieee(8, 23), "f32", "tf32", "TFloat32"),
            DType::F64 => float_info(64, FloatFormat::ieee(11, 52), "f64", "f64", "Float64"),
            DType::F8E5M2 => float_info(8, FloatFormat::ieee(5, 2), "i8", "f8E5M2", "Float8E5M2"),
            DType::F8E4M3 => float_info(8, FloatFormat::ieee(4, 3), "i8", "f8E4M3", "Float8E4M3"),
            DType::F8E4M3FN => float_info(8, E4M3FN, "i8", "f8E4M3FN", "Float8E4M3FN"),
            DType::F8E8M0FNU => float_info(8, E8M0FNU, "i8", "f8E8M0FNU", "Float8E8M0FNU"),
            DType::F8E4M3B11FNUZ => float_info(8, E4M3B11FNUZ, "i8", "f8E4M3B11FNUZ", "Float8E4M3B11FNUZ"),
            DType::F8E5M3FNU => float_info(8, E5M3FNU, "i8", "fnv8E5M3FNU", "FloatNV8E5M3FNU"),
            // Sub-byte floats live in the low bits of an `i8` in MLIR.
            DType::F4E2M1FN => float_info(4, E2M1, "i8", "f4E2M1FN", "Float4E2M1FN"),
            DType::F6E2M3FN => float_info(6, E2M3, "i8", "f6E2M3FN", "Float6E2M3FN"),
            DType::F6E3M2FN => float_info(6, E3M2, "i8", "f6E3M2FN", "Float6E3M2FN"),
            // Packed views: a byte, and three bytes in the low bits of an
            // `i32`; the format is the lanes'.
            DType::F4E2M1FNx2 => float_info(8, E2M1, "i8", "f4E2M1FNx2", "Float4E2M1FNx2"),
            DType::F6E2M3FNx4 => float_info(24, E2M3, "i32", "f6E2M3FNx4", "Float6E2M3FNx4"),
            DType::F6E3M2FNx4 => float_info(24, E3M2, "i32", "f6E3M2FNx4", "Float6E3M2FNx4"),
        }
    }

    pub const fn bits(self) -> usize {
        self.info().width as usize
    }

    /// Bytes occupied in memory (booleans and Int4 values one byte each:
    /// CuTe DSL's `T.bytes`).
    pub const fn bytes(self) -> usize {
        self.bits().div_ceil(8)
    }

    /// Bytes of `n` values (CuTe DSL's `T.n_bytes(n)`).
    pub const fn n_bytes(self, n: usize) -> usize {
        n * self.bytes()
    }

    pub const fn is_float(self) -> bool {
        self.info().float.is_some()
    }

    /// Integer types (booleans are not integers here; CuTe DSL's
    /// `is_integer` also holds for Boolean).
    pub const fn is_int(self) -> bool {
        !self.is_float() && !matches!(self, DType::Bool)
    }

    pub const fn is_signed(self) -> bool {
        // Boolean is signed in CuTe DSL's table, but converts as unsigned.
        self.info().signed && !matches!(self, DType::Bool)
    }

    /// The float format of a float type.
    pub const fn float_format(self) -> Option<FloatFormat> {
        self.info().float
    }

    /// The name used in diagnostics and IR listings.
    pub const fn name(self) -> &'static str {
        self.info().name
    }

    /// The MLIR type: integers are signless there, FP8 formats are bytes,
    /// Int4 lives in an `i8` and TFloat32 in an `f32`.
    pub const fn mlir(self) -> &'static str {
        self.info().mlir
    }

    /// CuTe DSL's class name: `Float16`, `Int4`, `Float8E4M3FN`, `Boolean`.
    pub const fn cute_name(self) -> &'static str {
        self.info().cute_name
    }

    /// The type with CuTe DSL's class name `name` (`cutlass.dtype(name)`).
    pub fn from_cute_name(name: &str) -> Option<DType> {
        DType::ALL.into_iter().find(|t| t.cute_name() == name)
    }

    /// CuTe DSL's `T.exponent_width`: only the narrow formats named
    /// `Float<bits>E<e>M<m>...` have one (not Float16, BFloat16, Float32,
    /// TFloat32, Float64 or FloatNV8E5M3FNU).
    pub fn exponent_width(self) -> Option<u32> {
        self.narrow_name_format().map(|f| f.exp_bits)
    }

    /// CuTe DSL's `T.mantissa_width`, for the same types as
    /// [`DType::exponent_width`].
    pub fn mantissa_width(self) -> Option<u32> {
        self.narrow_name_format().map(|f| f.mant_bits)
    }

    fn narrow_name_format(self) -> Option<FloatFormat> {
        let name = self.cute_name();
        let named = name.strip_prefix("Float").is_some_and(|rest| {
            let digits = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            digits > 0 && rest[digits..].starts_with('E')
        });
        if named { self.float_format() } else { None }
    }

    /// The FP8 formats CuTe DSL's `is_fp8_dtype` accepts: Float8E5M2 and
    /// Float8E4M3FN.
    pub fn is_fp8(self) -> bool {
        matches!(self, DType::F8E5M2 | DType::F8E4M3FN)
    }

    /// Whether values are narrower than a byte in memory (Int4, and the FP4
    /// and FP6 formats): sub-byte tensors store them packed, element 0 in the
    /// low bits, and access them in whole 32-bit words. Booleans are bytes.
    pub const fn is_sub_byte(self) -> bool {
        self.bits() < 8 && !matches!(self, DType::Bool)
    }

    /// Bits a value occupies in memory: its width for a sub-byte type, and
    /// whole bytes otherwise (8 for Boolean).
    pub const fn storage_bits(self) -> usize {
        if self.is_sub_byte() { self.bits() } else { 8 * self.bytes() }
    }

    /// Bytes of memory `n` values occupy: packed for a sub-byte type,
    /// rounded up to a whole byte.
    pub const fn storage_bytes(self, n: usize) -> usize {
        (n * self.storage_bits()).div_ceil(8)
    }

    /// The 8-bit float formats (FP8 and the scale factors), stored as bytes
    /// and converted with `cvt` or bit manipulation, with no arithmetic.
    pub fn is_narrow_float(self) -> bool {
        self.is_float() && self.bits() <= 8 && !self.is_packed_view()
    }

    /// The packed narrow-float views (CuTe DSL's `Float4E2M1FNx2`,
    /// `Float6E2M3FNx4`, `Float6E3M2FNx4`): one element is several values,
    /// and shapes and strides count those packed units.
    pub const fn is_packed_view(self) -> bool {
        matches!(self, DType::F4E2M1FNx2 | DType::F6E2M3FNx4 | DType::F6E3M2FNx4)
    }

    /// CuTe DSL's `_element_precision_width`: the width of one value of a
    /// packed view (4 for FP4x2, 6 for FP6x4), else the width.
    pub const fn element_precision_width(self) -> usize {
        match self {
            DType::F4E2M1FNx2 => 4,
            DType::F6E2M3FNx4 | DType::F6E3M2FNx4 => 6,
            t => t.bits(),
        }
    }

    /// The lane type of a packed view (`Float4E2M1FN` for FP4x2), and the
    /// number of lanes.
    pub const fn packed_lanes(self) -> Option<(DType, usize)> {
        match self {
            DType::F4E2M1FNx2 => Some((DType::F4E2M1FN, 2)),
            DType::F6E2M3FNx4 => Some((DType::F6E2M3FN, 4)),
            DType::F6E3M2FNx4 => Some((DType::F6E3M2FN, 4)),
            _ => None,
        }
    }

    /// Whether memory holds these values packed in 32-bit words, read and
    /// written only in whole words: the sub-byte types and the 24-bit FP6x4
    /// view.
    pub const fn is_word_packed(self) -> bool {
        self.is_sub_byte() || self.bits() == 24
    }

    /// The smallest value of an integer type (CuTe DSL's `T.min`), as a
    /// literal of the type (`false` for Boolean, whose CuTe DSL value, -1,
    /// comes from treating it as a signed 1-bit integer).
    pub fn min_value(self) -> Option<Literal> {
        if self == DType::Bool {
            return Some(Literal::Bool(false));
        }
        if !self.is_int() {
            return None;
        }
        let w = self.bits() as u32;
        Some(if self.is_signed() {
            Literal::from_i128(self, i128::MIN >> (128 - w))
        } else {
            Literal::from_i128(self, 0)
        })
    }

    /// The largest value of an integer type (CuTe DSL's `T.max`; Boolean's
    /// is `true`).
    pub fn max_value(self) -> Option<Literal> {
        if self == DType::Bool {
            return Some(Literal::Bool(true));
        }
        if !self.is_int() {
            return None;
        }
        let w = self.bits() as u32;
        Some(if self.is_signed() {
            Literal::from_i128(self, i128::MAX >> (128 - w))
        } else {
            Literal::from_bits128(self, u128::MAX >> (128 - w))
        })
    }

    /// Positive infinity, for a float type that has one (CuTe DSL's `T.inf`
    /// is a Python float for every float type).
    pub fn inf(self) -> Option<Literal> {
        let f = self.float_format()?;
        if !f.has_inf {
            return None;
        }
        let bits = if self == DType::TF32 { 0x7f80_0000 } else { ((1u128 << f.exp_bits) - 1) << f.mant_bits };
        Some(Literal::from_bits128(self, bits))
    }

    /// A quiet NaN of a float type: the canonical one PTX produces for the
    /// IEEE types, the only NaN pattern for the others.
    pub fn nan(self) -> Option<Literal> {
        let f = self.float_format()?;
        let mag = (1u128 << (f.exp_bits + f.mant_bits)) - 1;
        let bits = match f.nan {
            NanEncoding::Ieee | NanEncoding::AllOnes => mag,
            NanEncoding::NegativeZero => 1 << (f.exp_bits + f.mant_bits),
            NanEncoding::None => return None,
        };
        Some(Literal::from_bits128(self, bits))
    }

    /// Zero (`false` for Boolean); E8M0 has none.
    pub fn zero(self) -> Option<Literal> {
        if self.float_format().is_some_and(|f| !f.has_zero) {
            return None;
        }
        Some(Literal::zero(self))
    }

    /// The type CuTe DSL promotes two operands of types `self` and `other`
    /// to (its `_binary_op_type_promote`), or why it refuses. Arithmetic in
    /// cute-rs follows it ([`Promote`](crate::Promote)), except that a
    /// TFloat32 result is computed in Float32, and two Booleans compute in
    /// Int32.
    pub fn promote(self, other: DType) -> Result<DType, String> {
        let (a, b) = (self, other);
        if a == b {
            return Ok(a);
        }
        let refuse = || {
            Err(format!(
                "{} and {} have no implicit promotion (CuTe DSL refuses the pair): convert one with Value::to",
                a.cute_name(),
                b.cute_name()
            ))
        };
        let (aw, bw) = (a.bits(), b.bits());
        if a.is_float() || b.is_float() {
            let (mut at, mut bt) = (a, b);
            if !b.is_float() {
                match a.recast_width(aw.max(bw)) {
                    Some(t) => bt = t,
                    None => return refuse(),
                }
            } else if !a.is_float() {
                match b.recast_width(aw.max(bw)) {
                    Some(t) => at = t,
                    None => return refuse(),
                }
            }
            return if aw > bw && aw >= 16 {
                Ok(at)
            } else if bw > aw && bw >= 16 {
                Ok(bt)
            } else if aw == bw {
                [DType::F64, DType::F32, DType::F16].into_iter().find(|&t| at == t || bt == t).map_or_else(refuse, Ok)
            } else {
                refuse()
            };
        }
        // Integers, Boolean among them as a 1-bit signed integer.
        let signed = |t: DType| t.info().signed;
        if signed(a) != signed(b) {
            let (u, s) = if signed(a) { (b, a) } else { (a, b) };
            return Ok(if u.bits() >= s.bits() { u } else { s });
        }
        Ok(if aw >= bw { a } else { b })
    }

    /// The type of the same kind at `width` bits (CuTe DSL's
    /// `T.recast_width`): Int8 to Int128 for integers, Float16, Float32 or
    /// Float64 for floats.
    pub fn recast_width(self, width: usize) -> Option<DType> {
        if self.is_float() {
            return match width {
                16 => Some(DType::F16),
                32 => Some(DType::F32),
                64 => Some(DType::F64),
                _ => None,
            };
        }
        match width {
            8 => Some(DType::I8),
            16 => Some(DType::I16),
            32 => Some(DType::I32),
            64 => Some(DType::I64),
            128 => Some(DType::I128),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemSpace {
    Global,
    Shared,
    /// Per-thread memory (the stack), for arrays indexed at runtime.
    Local,
    /// Memory reached through a raw address ([`make_ptr`](crate::make_ptr)),
    /// in the given address space: a buffer of this space is no allocation
    /// but a view of whatever the address points at, and the `index` of a
    /// [`Op::Load`], [`Op::Store`], [`Op::Atomic`] or [`Op::Red`] on it is the
    /// 64-bit byte address of the access, not an element index.
    Raw(crate::tensor::AddressSpace),
    /// A module's `__constant__` global ([`module_global`](crate::module_global)): read-only
    /// on the device, at most 64 KB per module.
    Constant,
}

impl MemSpace {
    /// The address space of a raw buffer, or `None`.
    pub fn raw(self) -> Option<crate::tensor::AddressSpace> {
        match self {
            MemSpace::Raw(a) => Some(a),
            _ => None,
        }
    }
}

/// A constant. Floats are stored as raw bits so literals can be hashed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Literal {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
    /// An Int4 value (-8 to 7).
    I4(i8),
    I128(i128),
    U128(u128),
    /// TFloat32, as f32 bits.
    TF32(u32),
    /// A float of 8 bits or fewer of the given type
    /// ([`DType::is_narrow_float`], or the FP4x2 view), as its encoding in
    /// the low bits.
    F8(DType, u8),
    /// A 24-bit FP6x4 view value, as its three bytes in the low bits.
    F6x4(DType, u32),
}

impl Literal {
    pub fn dtype(self) -> DType {
        match self {
            Literal::Bool(_) => DType::Bool,
            Literal::I8(_) => DType::I8,
            Literal::I16(_) => DType::I16,
            Literal::I32(_) => DType::I32,
            Literal::I64(_) => DType::I64,
            Literal::U8(_) => DType::U8,
            Literal::U16(_) => DType::U16,
            Literal::U32(_) => DType::U32,
            Literal::U64(_) => DType::U64,
            Literal::F16(_) => DType::F16,
            Literal::BF16(_) => DType::BF16,
            Literal::F32(_) => DType::F32,
            Literal::F64(_) => DType::F64,
            Literal::I4(_) => DType::I4,
            Literal::I128(_) => DType::I128,
            Literal::U128(_) => DType::U128,
            Literal::TF32(_) => DType::TF32,
            Literal::F8(t, _) | Literal::F6x4(t, _) => t,
        }
    }

    /// Raw bit pattern, zero-extended to 64 bits, as the simulator stores it
    /// (the low 64 bits of a 128-bit value: [`Literal::bits128`]).
    pub fn bits(self) -> u64 {
        self.bits128() as u64
    }

    /// Raw bit pattern, zero-extended to 128 bits.
    pub fn bits128(self) -> u128 {
        match self {
            Literal::Bool(b) => b as u128,
            Literal::I8(v) => v as u8 as u128,
            Literal::I16(v) => v as u16 as u128,
            Literal::I32(v) => v as u32 as u128,
            Literal::I64(v) => v as u64 as u128,
            Literal::U8(v) => v as u128,
            Literal::U16(v) => v as u128,
            Literal::U32(v) => v as u128,
            Literal::U64(v) => v as u128,
            Literal::F16(b) | Literal::BF16(b) => b as u128,
            Literal::F32(b) | Literal::TF32(b) => b as u128,
            Literal::F64(b) => b as u128,
            Literal::I4(v) => (v as u8 & 0xf) as u128,
            Literal::I128(v) => v as u128,
            Literal::U128(v) => v,
            Literal::F8(t, b) => (b as u128) & ((1 << t.bits()) - 1),
            Literal::F6x4(_, b) => (b & 0xff_ffff) as u128,
        }
    }

    /// The literal of type `ty` with the raw bits `bits` (truncated to the
    /// type's width).
    pub fn from_bits128(ty: DType, bits: u128) -> Literal {
        match ty {
            DType::Bool => Literal::Bool(bits & 1 != 0),
            DType::I8 => Literal::I8(bits as i8),
            DType::I16 => Literal::I16(bits as i16),
            DType::I32 => Literal::I32(bits as i32),
            DType::I64 => Literal::I64(bits as i64),
            DType::U8 => Literal::U8(bits as u8),
            DType::U16 => Literal::U16(bits as u16),
            DType::U32 => Literal::U32(bits as u32),
            DType::U64 => Literal::U64(bits as u64),
            DType::F16 => Literal::F16(bits as u16),
            DType::BF16 => Literal::BF16(bits as u16),
            DType::F32 => Literal::F32(bits as u32),
            DType::F64 => Literal::F64(bits as u64),
            DType::I4 => Literal::I4(((bits as u8) << 4) as i8 >> 4),
            DType::I128 => Literal::I128(bits as i128),
            DType::U128 => Literal::U128(bits),
            DType::TF32 => Literal::TF32(bits as u32),
            DType::F6E2M3FNx4 | DType::F6E3M2FNx4 => Literal::F6x4(ty, bits as u32 & 0xff_ffff),
            t => Literal::F8(t, (bits & ((1 << t.bits()) - 1)) as u8),
        }
    }

    /// The literal of an integer type with the value `v`, wrapped to the
    /// type's width.
    pub fn from_i128(ty: DType, v: i128) -> Literal {
        Literal::from_bits128(ty, v as u128)
    }

    /// The zero of `ty` (for E8M0, which has none, the zero encoding,
    /// `2^-127`).
    pub fn zero(ty: DType) -> Literal {
        Literal::from_bits128(ty, 0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    /// Logical not for booleans, bitwise not for integers.
    Not,
    Abs,
    Sqrt,
    Rsqrt,
    /// `2^x` (approximate on hardware: `ex2.approx`).
    Exp2,
    /// `log2(x)` (approximate on hardware: `lg2.approx`).
    Log2,
    /// Sine (approximate on hardware: `sin.approx`).
    Sin,
    /// Cosine (approximate on hardware: `cos.approx`).
    Cos,
    /// `1 / x` (approximate on hardware: `rcp.approx`).
    Rcp,
    Floor,
    Ceil,
    Trunc,
    /// Round to the nearest integer, ties to even.
    Rint,
    /// Number of set bits.
    Popc,
    /// Leading zero bits.
    Clz,
    /// Trailing zero bits.
    Ctz,
    /// Reverse the bits.
    Brev,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// Float division, or integer division following the type's signedness.
    Div,
    /// Unsigned integer division (index arithmetic on non-negative values).
    DivU,
    /// Integer remainder following the type's signedness.
    Rem,
    /// Unsigned integer remainder.
    RemU,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    /// Logical (zero-filling) shift right.
    ShrU,
    /// Arithmetic (sign-filling) shift right.
    ShrS,
    /// The high half of the full-width product.
    MulHi,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CmpPred {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    X,
    Y,
    Z,
}

impl Dim {
    pub fn index(self) -> usize {
        match self {
            Dim::X => 0,
            Dim::Y => 1,
            Dim::Z => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dim::X => "x",
            Dim::Y => "y",
            Dim::Z => "z",
        }
    }
}

/// Lane masks relative to the executing lane (`%lanemask_*`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneMask {
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SpecialReg {
    ThreadIdx(Dim),
    BlockIdx(Dim),
    BlockDim(Dim),
    GridDim(Dim),
    LaneId,
    /// Threads per warp (32): PTX's predefined constant `WARP_SZ`, read
    /// through `llvm.nvvm.read.ptx.sreg.warpsize`. (Not a `%` register.)
    WarpSize,
    WarpId,
    NWarpId,
    SmId,
    NSmId,
    GridId,
    LaneMask(LaneMask),
    /// Cycle counter (32 bits). Changes between reads.
    Clock,
    /// Cycle counter (64 bits). Changes between reads.
    Clock64,
    /// Nanosecond timer (64 bits). Changes between reads.
    GlobalTimer,
    /// The low 32 bits of the nanosecond timer. Changes between reads.
    GlobalTimerLo,
    ClusterIdx(Dim),
    ClusterDim(Dim),
    BlockInClusterIdx(Dim),
    BlockInClusterDim(Dim),
    /// The bytes of dynamic shared memory the launch requested
    /// ([`LaunchConfig::dynamic_smem_bytes`](crate::LaunchConfig::dynamic_smem_bytes)),
    /// which the dynamic region ([`get_dyn_smem`](crate::get_dyn_smem)) gets:
    /// CuTe DSL's `dynamic_smem_size` and `get_dyn_smem_size`. It is
    /// `%dynamic_smem_size` less the shared memory sized at build time that
    /// the dynamic window holds in front of the region
    /// ([`SharedMemoryLayout::dynamic_start`]).
    DynamicSmemSize,
    /// `%total_smem_size`: the block's static and dynamic shared memory in
    /// bytes, rounded up to the 128-byte allocation unit (as an SM89 GPU
    /// reports it).
    TotalSmemSize,
    /// `%aggr_smem_size`: the block's shared memory plus the part reserved
    /// for the system (sm_90 and newer).
    AggrSmemSize,
    /// `%cluster_ctarank`: the block's linear rank in its cluster, x
    /// fastest (CuTe DSL's `block_idx_in_cluster`).
    ClusterCtaRank,
    /// `%cluster_nctarank`: the number of blocks in the cluster (CuTe DSL's
    /// `cluster_size`).
    ClusterNCtaRank,
}

impl SpecialReg {
    pub fn dtype(self) -> DType {
        match self {
            SpecialReg::Clock64 | SpecialReg::GlobalTimer => DType::I64,
            SpecialReg::LaneMask(_) | SpecialReg::Clock => DType::U32,
            _ => DType::I32,
        }
    }

    /// Whether two reads in one thread always agree (timers do not).
    pub fn is_constant(self) -> bool {
        !matches!(self, SpecialReg::Clock | SpecialReg::Clock64 | SpecialReg::GlobalTimer | SpecialReg::GlobalTimerLo)
    }
}

/// Which lane a `shfl.sync` reads from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShuffleKind {
    /// The lane given by the offset operand.
    Idx,
    /// `lane - offset`.
    Up,
    /// `lane + offset`.
    Down,
    /// `lane ^ offset`.
    Bfly,
}

/// What a `vote.sync` computes across the warp.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VoteKind {
    All,
    Any,
    /// All participating lanes agree.
    Uni,
    /// A mask of the lanes whose predicate holds.
    Ballot,
}

/// What a `match.sync` computes (CuTe DSL's `kind="any"` and `kind="all"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MatchKind {
    /// `match.any.sync`: the member lanes holding the same value as this one.
    #[default]
    Any,
    /// `match.all.sync`: the member lanes if they all hold the same value,
    /// otherwise 0.
    All,
}

/// The reduction a `redux.sync` computes across the member lanes of a warp
/// (CuTe DSL's `warp_redux_sync` kinds).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduxKind {
    /// The sum, wrapping at 32 bits (Int32, Uint32).
    Add,
    /// Bitwise and (Int32, Uint32).
    And,
    /// Bitwise or (Int32, Uint32).
    Or,
    /// Bitwise exclusive or (Int32, Uint32).
    Xor,
    /// The largest value: signed on Int32; `warp_redux_sync` takes it as
    /// [`ReduxKind::Umax`] on Uint32 and [`ReduxKind::Fmax`] on Float32, as
    /// CuTe DSL does.
    Max,
    /// The smallest value: signed on Int32; [`ReduxKind::Umin`] on Uint32
    /// and [`ReduxKind::Fmin`] on Float32.
    Min,
    /// The largest value as unsigned 32-bit integers.
    Umax,
    /// The smallest value as unsigned 32-bit integers.
    Umin,
    /// The largest Float32 (`+0 > -0`), with optional `.abs` and `.NaN`.
    Fmax,
    /// The smallest Float32 (`-0 < +0`), with optional `.abs` and `.NaN`.
    Fmin,
}

impl ReduxKind {
    /// CuTe DSL's name for the kind (and the NVVM dialect's).
    pub fn name(self) -> &'static str {
        match self {
            ReduxKind::Add => "add",
            ReduxKind::And => "and",
            ReduxKind::Or => "or",
            ReduxKind::Xor => "xor",
            ReduxKind::Max => "max",
            ReduxKind::Min => "min",
            ReduxKind::Umax => "umax",
            ReduxKind::Umin => "umin",
            ReduxKind::Fmax => "fmax",
            ReduxKind::Fmin => "fmin",
        }
    }

    /// Whether the kind reduces Float32 values.
    pub fn is_float(self) -> bool {
        matches!(self, ReduxKind::Fmax | ReduxKind::Fmin)
    }
}

/// The PTX instruction an [`Op::Redux`] issues on a value of type `ty`, e.g.
/// `redux.sync.add.s32`, `redux.sync.min.u32`, `redux.sync.xor.b32` or
/// `redux.sync.max.abs.NaN.f32`.
pub fn redux_inst_name(kind: ReduxKind, abs: bool, nan: bool, ty: DType) -> String {
    let (op, suffix) = match kind {
        ReduxKind::Add => ("add", if ty.is_signed() { "s32" } else { "u32" }),
        ReduxKind::And => ("and", "b32"),
        ReduxKind::Or => ("or", "b32"),
        ReduxKind::Xor => ("xor", "b32"),
        ReduxKind::Max => ("max", "s32"),
        ReduxKind::Min => ("min", "s32"),
        ReduxKind::Umax => ("max", "u32"),
        ReduxKind::Umin => ("min", "u32"),
        ReduxKind::Fmax => ("max", "f32"),
        ReduxKind::Fmin => ("min", "f32"),
    };
    format!("redux.sync.{op}{}{}.{suffix}", if abs { ".abs" } else { "" }, if nan { ".NaN" } else { "" })
}

/// The operation of an atomic ([`Op::Atomic`], PTX's `atom`) or a reduction
/// ([`Op::Red`], `red`). Whether `Min` and `Max` compare signed or unsigned
/// integers is the [`AtomicType`]'s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AtomicKind {
    Add,
    And,
    Or,
    Xor,
    Min,
    Max,
    Exchange,
    CompareExchange,
}

impl AtomicKind {
    /// The PTX operation, without the dot (`add`, `exch`, `cas`).
    pub fn ptx(self) -> &'static str {
        match self {
            AtomicKind::Add => "add",
            AtomicKind::And => "and",
            AtomicKind::Or => "or",
            AtomicKind::Xor => "xor",
            AtomicKind::Min => "min",
            AtomicKind::Max => "max",
            AtomicKind::Exchange => "exch",
            AtomicKind::CompareExchange => "cas",
        }
    }
}

/// The PTX type of an atomic or a reduction: the `dtype=` of CuTe DSL's
/// `cute.arch.red` (`"b32"`, `"u32"`, `"s32"`, `"f32"`, `"f16x2"`, ...),
/// which the atomics derive from their operand. The bit types are for the
/// bit operations, exchange and compare-and-swap; the packed types
/// (`F16x2`, `Bf16x2`) add two Float16 or BFloat16 values held in an
/// [`Int32`](crate::Int32), each half atomically on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AtomicType {
    B32,
    B64,
    U32,
    U64,
    S32,
    S64,
    F32,
    F64,
    F16,
    F16x2,
    Bf16,
    Bf16x2,
}

impl AtomicType {
    /// The PTX type, without the dot (`u32`, `bf16x2`).
    pub fn ptx(self) -> &'static str {
        match self {
            AtomicType::B32 => "b32",
            AtomicType::B64 => "b64",
            AtomicType::U32 => "u32",
            AtomicType::U64 => "u64",
            AtomicType::S32 => "s32",
            AtomicType::S64 => "s64",
            AtomicType::F32 => "f32",
            AtomicType::F64 => "f64",
            AtomicType::F16 => "f16",
            AtomicType::F16x2 => "f16x2",
            AtomicType::Bf16 => "bf16",
            AtomicType::Bf16x2 => "bf16x2",
        }
    }

    /// The bits of memory the operation updates.
    pub fn bits(self) -> usize {
        match self {
            AtomicType::F16 | AtomicType::Bf16 => 16,
            AtomicType::B64 | AtomicType::U64 | AtomicType::S64 | AtomicType::F64 => 64,
            _ => 32,
        }
    }

    /// Whether the values are floating point (packed ones too).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            AtomicType::F32
                | AtomicType::F64
                | AtomicType::F16
                | AtomicType::F16x2
                | AtomicType::Bf16
                | AtomicType::Bf16x2
        )
    }

    /// Whether the type packs two half-precision values into 32 bits.
    pub fn is_packed(self) -> bool {
        matches!(self, AtomicType::F16x2 | AtomicType::Bf16x2)
    }

    /// Whether it compares signed integers (`s32`, `s64`).
    pub fn is_signed(self) -> bool {
        matches!(self, AtomicType::S32 | AtomicType::S64)
    }

    /// The Float16 or BFloat16 element of a half-precision type.
    pub fn half(self) -> Option<DType> {
        match self {
            AtomicType::F16 | AtomicType::F16x2 => Some(DType::F16),
            AtomicType::Bf16 | AtomicType::Bf16x2 => Some(DType::BF16),
            _ => None,
        }
    }

    /// The element type an operation of this type on a `value` addresses
    /// memory in: the half-precision element of a packed type (its index
    /// counts those), the operand's type otherwise.
    pub fn access_elem(self, value: DType) -> DType {
        if self.is_packed() { self.half().expect("packed types are half precision") } else { value }
    }

    /// The type CuTe DSL's atomics give an operation `kind` on `value`: the
    /// bit type for the bit operations, exchange and compare-and-swap, the
    /// signed or unsigned type for min and max, and for add the value's own
    /// type, except that a 64-bit add is `u64` (PTX has no `add.s64`; the
    /// sum's bits are the same).
    pub fn for_value(kind: AtomicKind, value: DType) -> AtomicType {
        let wide = value.bits() == 64;
        match (kind, value) {
            (_, DType::F32) => AtomicType::F32,
            (_, DType::F64) => AtomicType::F64,
            (_, DType::F16) => AtomicType::F16,
            (_, DType::BF16) => AtomicType::Bf16,
            (AtomicKind::Min | AtomicKind::Max, t) if t.is_signed() => {
                if wide {
                    AtomicType::S64
                } else {
                    AtomicType::S32
                }
            }
            (AtomicKind::Min | AtomicKind::Max, _) | (AtomicKind::Add, DType::U32 | DType::I64 | DType::U64) => {
                if wide { AtomicType::U64 } else { AtomicType::U32 }
            }
            (AtomicKind::Add, _) => AtomicType::S32,
            _ => {
                if wide {
                    AtomicType::B64
                } else {
                    AtomicType::B32
                }
            }
        }
    }

    /// Whether PTX has `atom` (or, with `red`, `red`) with operation `kind`
    /// on this type, from the PTX ISA's tables of scalar atomic and
    /// reduction operations: integer add on `u32`, `s32` and `u64`, min and
    /// max on the four integer types, the bit operations on `b32` and `b64`,
    /// float add on every float type (`f16` and the packed types with
    /// `.noftz`), exchange and compare-and-swap on `b32` and `b64` (`atom`
    /// only). There is no scalar float min or max.
    pub fn supports(self, kind: AtomicKind, red: bool) -> bool {
        use AtomicType::*;
        match kind {
            AtomicKind::Add => matches!(self, U32 | S32 | U64) || self.is_float(),
            AtomicKind::Min | AtomicKind::Max => matches!(self, U32 | S32 | U64 | S64),
            AtomicKind::And | AtomicKind::Or | AtomicKind::Xor => matches!(self, B32 | B64),
            AtomicKind::Exchange | AtomicKind::CompareExchange => !red && matches!(self, B32 | B64),
        }
    }

    /// `kind` and this type as PTX: `add.u32`, `add.noftz.f16x2` (the
    /// half-precision adds require `.noftz`), `cas.b64`.
    pub fn op_suffix(self, kind: AtomicKind) -> String {
        let noftz = if self.half().is_some() { ".noftz" } else { "" };
        format!("{}{noftz}.{}", kind.ptx(), self.ptx())
    }
}

/// The `atom` (`red` with `red`) instruction of an atomic with these
/// attributes on memory in `space`, as cute-rs emits it: the semantics and
/// scope always explicit, then the state space, the operation and the type,
/// e.g. `atom.relaxed.gpu.global.add.u32` or
/// `red.release.cta.shared.add.noftz.f16x2`. Shared memory is `.shared`,
/// which is `.shared::cta`, valid from PTX 2.0 (the `::cta` spelling needs
/// PTX 7.8).
pub fn atomic_inst_name(
    red: bool,
    kind: AtomicKind,
    ty: AtomicType,
    sem: MemoryOrder,
    scope: MemoryScope,
    space: MemSpace,
) -> String {
    use crate::tensor::AddressSpace;
    let space = match space {
        MemSpace::Shared | MemSpace::Raw(AddressSpace::Smem) => ".shared",
        // Another CTA's shared memory (m8-dsmem).
        MemSpace::Raw(AddressSpace::Dsmem) => ".shared::cluster",
        // A generic address: no state space.
        MemSpace::Raw(AddressSpace::Generic) => "",
        _ => ".global",
    };
    format!("{}.{}.{}{space}.{}", if red { "red" } else { "atom" }, sem.name(), scope.ptx(), ty.op_suffix(kind))
}

/// Which way a bulk copy (`cp.async.bulk`) moves data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BulkDirection {
    /// Global to shared memory (`cp.async.bulk.shared::cluster.global`).
    G2S,
    /// Shared to global memory (`cp.async.bulk.global.shared::cta`).
    S2G,
    /// From this CTA's shared memory to a peer CTA's of its cluster
    /// (`cp.async.bulk.shared::cluster.shared::cta`, roadmap part
    /// m7-cluster-copies).
    S2S,
}

// ---- m7-tma-im2col and m7-tma-sm100-modes ----

/// The tensor access mode of a TMA instruction (the PTX ISA's `.load_mode`).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum TmaMode {
    /// `.tile`: the box at the coordinates.
    #[default]
    Tile,
    /// `.im2col` (loads and prefetches): the coordinates are the filter base
    /// `c, w, (h, (d,)) n`, and `offsets` (i16, one per spatial dimension, W
    /// first) are added to each pixel.
    Im2col { offsets: Vec<ValueId> },
    /// `.im2col_no_offs` (stores): im2col coordinates without offsets.
    Im2colNoOffs,
    /// `.tile::gather4` (loads and prefetches): five coordinates, the column
    /// and four rows of a 2-D tensor map.
    Gather4,
    /// `.tile::scatter4` (stores): as [`TmaMode::Gather4`].
    Scatter4,
}

impl TmaMode {
    /// The PTX qualifier.
    pub fn ptx(&self) -> &'static str {
        match self {
            TmaMode::Tile => ".tile",
            TmaMode::Im2col { .. } => ".im2col",
            TmaMode::Im2colNoOffs => ".im2col_no_offs",
            TmaMode::Gather4 => ".tile::gather4",
            TmaMode::Scatter4 => ".tile::scatter4",
        }
    }

    /// The four-row modes.
    pub fn is_rows4(&self) -> bool {
        matches!(self, TmaMode::Gather4 | TmaMode::Scatter4)
    }

    /// The im2col offsets (empty for the other modes).
    pub fn offsets(&self) -> &[ValueId] {
        match self {
            TmaMode::Im2col { offsets } => offsets,
            _ => &[],
        }
    }
}

/// The `.override::global_address` and `.override::global_dim{_stride}`
/// operands of a TMA instruction (sm_107f, PTX 9.4): the tensor's address
/// (an i64), and when `dims` is not empty its extents (one i16 per
/// dimension, the value in the low 8 bits) and strides (`lower_strides`,
/// one i32 per dimension after the first, and the i16 `upper_strides`
/// holding a 4-bit value per dimension after the first, the stride in bytes
/// being `(lower + (upper << 32)) << 4`). The coordinates are then zero.
#[derive(Clone, Debug, PartialEq)]
pub struct TmaOverride {
    pub global_address: ValueId,
    pub dims: Vec<ValueId>,
    pub lower_strides: Vec<ValueId>,
    pub upper_strides: Option<ValueId>,
}

impl TmaOverride {
    /// Its operands.
    pub fn operands(&self) -> Vec<ValueId> {
        let mut v = vec![self.global_address];
        v.extend(&self.dims);
        v.extend(&self.lower_strides);
        v.extend(self.upper_strides);
        v
    }
}

/// How a bulk copy reports its completion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BulkCompletion {
    /// `.mbarrier::complete_tx::bytes`: its bytes count down the
    /// transaction count of the mbarrier at element `index` of `buffer`.
    Mbarrier { buffer: BufferId, index: ValueId },
    /// `.bulk_group`: the thread's bulk async-groups
    /// (`cp.async.bulk.commit_group`, `cp.async.bulk.wait_group`).
    BulkGroup,
}

/// The proxies a `fence.proxy` orders (CuTe DSL's `fence_proxy` kinds the
/// IR has; `tensormap` and `generic` belong to the tensor-map fences).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ProxyKind {
    /// `fence.proxy.alias`: accesses to the same memory through different
    /// virtual addresses.
    Alias,
    /// `fence.proxy.async`: the generic and async proxies, in every state
    /// space.
    Async,
    /// `fence.proxy.async.global`.
    AsyncGlobal,
    /// `fence.proxy.async.shared::cta`.
    AsyncSharedCta,
    /// `fence.proxy.async.shared::cluster`.
    AsyncSharedCluster,
}

impl ProxyKind {
    /// The PTX instruction, without the semicolon.
    pub fn ptx(self) -> &'static str {
        match self {
            ProxyKind::Alias => "fence.proxy.alias",
            ProxyKind::Async => "fence.proxy.async",
            ProxyKind::AsyncGlobal => "fence.proxy.async.global",
            ProxyKind::AsyncSharedCta => "fence.proxy.async.shared::cta",
            ProxyKind::AsyncSharedCluster => "fence.proxy.async.shared::cluster",
        }
    }

    /// Whether it orders the generic proxy's shared-memory accesses before
    /// the async proxy's (every async kind but `async.global`).
    pub fn orders_async_shared(self) -> bool {
        matches!(self, ProxyKind::Async | ProxyKind::AsyncSharedCta | ProxyKind::AsyncSharedCluster)
    }
}

/// The threads a fence orders against.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceScope {
    /// The block.
    Cta,
    /// The block cluster.
    Cluster,
    /// The whole GPU.
    Gpu,
    /// The GPU and the host.
    Sys,
}

impl FenceScope {
    /// The PTX scope qualifier, without the dot (`cta`, `cluster`, `gpu`,
    /// `sys`).
    pub fn ptx(self) -> &'static str {
        match self {
            FenceScope::Cta => "cta",
            FenceScope::Cluster => "cluster",
            FenceScope::Gpu => "gpu",
            FenceScope::Sys => "sys",
        }
    }
}

/// The threads an ordered load or store synchronises with: CuTe DSL's
/// `cute.nvgpu.MemoryScope` (`CTA`, `CLUSTER`, `GPU`, `SYS`) and the
/// `scope=` of `cute.arch.load` and `cute.arch.store` (`"cta"`, `"cluster"`,
/// `"gpu"`, `"sys"`). The scopes of a fence.
pub type MemoryScope = FenceScope;

/// The memory-ordering semantics of a load or store (PTX's `.weak`,
/// `.relaxed`, `.acquire`, `.release`, `.volatile` and `.mmio`): CuTe DSL's
/// `cute.nvgpu.MemoryOrder` and the `sem=` of `cute.arch.load` and
/// `cute.arch.store`.
///
/// Every order but `Weak` and `Constant` makes the access *strong* in the
/// PTX memory model: two strong accesses of the same bytes from threads in
/// each other's scope never race (strong accesses of different sizes that
/// overlap do: a mixed-size data race). A release store (or a strong store
/// after a fence) publishes what its thread has seen, and an acquire load
/// that reads exactly what it wrote (or such a strong load followed by a
/// fence) sees it too. Each element of a vector access is one access.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemoryOrder {
    /// `.weak`, a plain access with no synchronisation (the default).
    #[default]
    Weak,
    /// `.relaxed.scope`: strong, but orders nothing else.
    Relaxed,
    /// `.acquire.scope` (loads): later accesses of the thread see what the
    /// release it reads from published.
    Acquire,
    /// `.release.scope` (stores): publishes the thread's earlier accesses.
    Release,
    /// Acquire-release: an order for atomics, which loads and stores do not
    /// take.
    AcqRel,
    /// Sequentially consistent. PTX has no `.sc` load or store; this is
    /// LLVM's mapping of a `seq_cst` access: `fence.sc.scope` followed by an
    /// acquire load or a release store.
    Sc,
    /// `.mmio.relaxed.sys`: a scalar access to memory-mapped I/O in global
    /// memory, performed exactly once, with relaxed system-scope semantics.
    Mmio,
    /// A load of memory that no thread writes while the kernel runs, as a
    /// non-coherent load (`ld.global.nc`, like CuTe DSL's `invariant`).
    Constant,
    /// `.volatile`: relaxed with system scope, never cached, and taking no
    /// cache hints.
    Volatile,
}

impl MemoryOrder {
    /// Whether accesses with this order are strong in PTX's memory model.
    pub fn is_strong(self) -> bool {
        !matches!(self, MemoryOrder::Weak | MemoryOrder::Constant)
    }

    /// Whether this order takes a scope (`.relaxed.scope` and so on).
    pub fn is_scoped(self) -> bool {
        matches!(self, MemoryOrder::Relaxed | MemoryOrder::Acquire | MemoryOrder::Release | MemoryOrder::Sc)
    }

    /// CuTe DSL's name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            MemoryOrder::Weak => "weak",
            MemoryOrder::Relaxed => "relaxed",
            MemoryOrder::Acquire => "acquire",
            MemoryOrder::Release => "release",
            MemoryOrder::AcqRel => "acq_rel",
            MemoryOrder::Sc => "sc",
            MemoryOrder::Mmio => "mmio",
            MemoryOrder::Constant => "constant",
            MemoryOrder::Volatile => "volatile",
        }
    }
}

/// An L1 eviction priority (`.L1::evict_normal`, `.L1::evict_first`,
/// `.L1::evict_last`, `.L1::evict_unchanged`, `.L1::no_allocate`): CuTe
/// DSL's `cute.nvgpu.CacheEvictionPriority` and the
/// `level1_eviction_priority=` of `cute.arch.load` and `cute.arch.store`.
/// A prefetch takes `EvictLast` and `EvictNormal` for the L2 cache.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CacheEvictionPriority {
    /// `evict_normal`.
    #[default]
    EvictNormal,
    /// `evict_first`: likely to be evicted first.
    EvictFirst,
    /// `evict_last`: likely to be evicted last.
    EvictLast,
    /// `evict_unchanged`: keep the line's current priority.
    EvictUnchanged,
    /// `no_allocate` (CuTe DSL's `evict_no_allocate` and `NO_ALLOCATE`):
    /// do not allocate the line in L1.
    NoAllocate,
}

impl CacheEvictionPriority {
    /// The PTX name of the priority (`evict_last`, `no_allocate`).
    pub fn name(self) -> &'static str {
        match self {
            CacheEvictionPriority::EvictNormal => "evict_normal",
            CacheEvictionPriority::EvictFirst => "evict_first",
            CacheEvictionPriority::EvictLast => "evict_last",
            CacheEvictionPriority::EvictUnchanged => "evict_unchanged",
            CacheEvictionPriority::NoAllocate => "no_allocate",
        }
    }

    /// The L1 qualifier, without the dot (`L1::evict_last`).
    pub fn l1_ptx(self) -> String {
        format!("L1::{}", self.name())
    }
}

/// A PTX cache operator: `.ca`, `.cg`, `.cs`, `.lu` and `.cv` for loads,
/// `.wb`, `.cg`, `.cs` and `.wt` for stores (the `cop=` of CuTe DSL's
/// `cute.arch.load` and `cute.arch.store`). A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CacheOperator {
    /// `.ca`: cache at all levels (loads; the default).
    Ca,
    /// `.cg`: cache in L2, not L1.
    Cg,
    /// `.cs`: streaming, likely to be accessed once.
    Cs,
    /// `.lu`: last use (loads).
    Lu,
    /// `.cv`: do not cache, fetch again (loads).
    Cv,
    /// `.wb`: write back (stores; the default).
    Wb,
    /// `.wt`: write through to system memory (stores).
    Wt,
}

impl CacheOperator {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            CacheOperator::Ca => "ca",
            CacheOperator::Cg => "cg",
            CacheOperator::Cs => "cs",
            CacheOperator::Lu => "lu",
            CacheOperator::Cv => "cv",
            CacheOperator::Wb => "wb",
            CacheOperator::Wt => "wt",
        }
    }

    /// Whether loads take it.
    pub fn for_loads(self) -> bool {
        !matches!(self, CacheOperator::Wb | CacheOperator::Wt)
    }

    /// Whether stores take it.
    pub fn for_stores(self) -> bool {
        matches!(self, CacheOperator::Wb | CacheOperator::Cg | CacheOperator::Cs | CacheOperator::Wt)
    }
}

/// How much more data a load asks L2 to fetch (`.L2::64B`, `.L2::128B`,
/// `.L2::256B`): CuTe DSL's `cute.nvgpu.L2PrefetchSize` and the
/// `level_prefetch_size=` of `cute.arch.load`. A performance hint only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum L2PrefetchSize {
    /// `.L2::64B` (`size_64b`, `SIZE_64B`).
    Size64B,
    /// `.L2::128B` (`size_128b`, `SIZE_128B`).
    Size128B,
    /// `.L2::256B` (`size_256b`, `SIZE_256B`), sm_80 and newer.
    Size256B,
}

impl L2PrefetchSize {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            L2PrefetchSize::Size64B => "L2::64B",
            L2PrefetchSize::Size128B => "L2::128B",
            L2PrefetchSize::Size256B => "L2::256B",
        }
    }
}

/// The memory attributes of a load or store: its ordering and scope, its
/// cache hints, whether it reads through the non-coherent cache, and an
/// optional L2 cache-policy operand (`.L2::cache_hint`, a 64-bit value from
/// `createpolicy`). The default is a plain weak access, which is what
/// tensor loads and stores are.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MemAttrs {
    pub order: MemoryOrder,
    /// The threads a relaxed, acquire, release or seq_cst access
    /// synchronises with. Weak, volatile and constant accesses have none,
    /// and MMIO is always `Sys`.
    pub scope: MemoryScope,
    /// `.L1::evict_*` (loads and stores).
    pub l1_eviction: Option<CacheEvictionPriority>,
    /// `.ca`, `.cg`, ... (loads and stores).
    pub cop: Option<CacheOperator>,
    /// `.L2::64B` and so on (loads).
    pub l2_prefetch: Option<L2PrefetchSize>,
    /// `.nc`: read through the non-coherent (read-only) cache (loads).
    pub non_coherent: bool,
    /// The 64-bit operand of `.L2::cache_hint`.
    pub cache_policy: Option<ValueId>,
    /// Distributed shared memory (roadmap part m8-dsmem): the rank (an
    /// Int32) of the CTA of the cluster whose shared memory the access
    /// reaches, at the offset it has in the buffer of this CTA (`mapa`, then
    /// `.shared::cluster`). Shared buffers only.
    pub peer: Option<ValueId>,
}

impl Default for MemAttrs {
    fn default() -> Self {
        MemAttrs {
            order: MemoryOrder::Weak,
            scope: FenceScope::Sys,
            l1_eviction: None,
            cop: None,
            l2_prefetch: None,
            non_coherent: false,
            cache_policy: None,
            peer: None,
        }
    }
}

impl MemAttrs {
    /// A weak access with no hints, as [`Tensor::load`](crate::Tensor::load)
    /// and `store` make (the scope does not matter).
    pub fn is_plain(&self) -> bool {
        self.order == MemoryOrder::Weak && !self.has_hints()
    }

    /// Whether it carries a cache hint, a cache policy or `.nc`.
    pub fn has_hints(&self) -> bool {
        self.l1_eviction.is_some()
            || self.cop.is_some()
            || self.l2_prefetch.is_some()
            || self.non_coherent
            || self.cache_policy.is_some()
    }

    /// Whether the access is strong ([`MemoryOrder::is_strong`]).
    pub fn is_strong(&self) -> bool {
        self.order.is_strong()
    }

    /// Whether it reads through the non-coherent cache (`.nc`, or
    /// [`MemoryOrder::Constant`]).
    pub fn is_non_coherent(&self) -> bool {
        self.non_coherent || self.order == MemoryOrder::Constant
    }

    /// The PTX qualifiers of the access, as in `.acquire.gpu.L1::evict_last`
    /// (the state space, vector width and type are not included), for
    /// listings and messages.
    pub fn qualifiers(&self) -> String {
        let mut q = String::new();
        match self.order {
            MemoryOrder::Weak | MemoryOrder::Constant => {}
            MemoryOrder::Volatile => q.push_str(".volatile"),
            MemoryOrder::Mmio => q.push_str(".mmio.relaxed.sys"),
            o => q.push_str(&format!(".{}.{}", o.name(), self.scope.ptx())),
        }
        if self.is_non_coherent() {
            q.push_str(".nc");
        }
        if let Some(c) = self.cop {
            q.push_str(&format!(".{}", c.ptx()));
        }
        if let Some(e) = self.l1_eviction {
            q.push_str(&format!(".{}", e.l1_ptx()));
        }
        if self.cache_policy.is_some() {
            q.push_str(".L2::cache_hint");
        }
        if let Some(p) = self.l2_prefetch {
            q.push_str(&format!(".{}", p.ptx()));
        }
        q
    }
}

/// The cache a prefetch fills: CuTe DSL's `cache_level=` of
/// `cute.arch.prefetch` (`"L1"`, the default, or `"L2"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PrefetchCacheLevel {
    #[default]
    L1,
    L2,
}

impl PrefetchCacheLevel {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            PrefetchCacheLevel::L1 => "L1",
            PrefetchCacheLevel::L2 => "L2",
        }
    }
}

/// The threads an mbarrier operation synchronises with (CuTe DSL's
/// `nvvm.MemScopeKind` as the mbarrier operations take it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemScopeKind {
    /// The CTA (the default).
    #[default]
    Cta,
    /// The CTA's cluster (SM90 and newer).
    Cluster,
}

/// How an mbarrier phase is waited on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MbarrierWaitKind {
    /// `mbarrier.try_wait`: may suspend the thread for a while (SM90).
    Try,
    /// `mbarrier.test_wait`: returns at once (SM80).
    Test,
}

/// The named-barrier instructions (`bar`, `barrier`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierKind {
    /// `bar.sync id{, count}`: arrive, and wait until the barrier completes.
    Sync,
    /// `bar.arrive id, count`: arrive without waiting.
    Arrive,
    /// `barrier.arrive id, count`: an arrive without `.aligned`, which the
    /// threads of the CTA need not all execute.
    ArriveUnaligned,
}

/// The reduction of a `barrier.cta.red` (`bar.red`; roadmap part
/// m11-prims-nvvm-layer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierRedKind {
    /// `.popc.u32`: how many participating threads' predicates are true.
    Popc,
    /// `.and.pred`: whether every participating thread's predicate is true.
    And,
    /// `.or.pred`: whether any participating thread's predicate is true.
    Or,
}

impl BarrierRedKind {
    /// The PTX reduction and type (`popc.u32`, `and.pred`, `or.pred`).
    pub fn ptx(self) -> &'static str {
        match self {
            BarrierRedKind::Popc => "popc.u32",
            BarrierRedKind::And => "and.pred",
            BarrierRedKind::Or => "or.pred",
        }
    }

    /// The reduction's name (`popc`, `and`, `or`).
    pub fn name(self) -> &'static str {
        match self {
            BarrierRedKind::Popc => "popc",
            BarrierRedKind::And => "and",
            BarrierRedKind::Or => "or",
        }
    }
}

/// The mbarrier operations of roadmap part m11-prims-nvvm-layer
/// ([`Op::MbarrierExt`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MbarrierExtKind {
    /// `mbarrier.arrive.shared.b64 state, [addr]`: one arrival that returns
    /// the phase's state token.
    ArriveState,
    /// `mbarrier.arrive_drop.shared.b64 state, [addr]`: one arrival that
    /// also lowers the arrival count of this phase and all later ones.
    ArriveDrop,
    /// `mbarrier.arrive_drop.expect_tx.shared::cta.b64 state, [addr], tx`
    /// (sm_90): raise the tx-count by `value` bytes, then arrive-drop.
    ArriveDropExpectTx,
    /// `mbarrier.arrive_drop.noComplete.shared.b64 state, [addr], count`:
    /// `value` arrive-drops that must not complete the phase.
    ArriveDropNoComplete,
    /// `mbarrier.arrive.noComplete.shared.b64 state, [addr], count`: `value`
    /// arrivals that must not complete the phase.
    ArriveNoComplete,
    /// `mbarrier.complete_tx.shared::cta.b64 [addr], tx` (sm_90): lower the
    /// tx-count by `value` bytes, as if bulk copies had landed.
    CompleteTx,
    /// `mbarrier.inval.shared.b64 [addr]`: the mbarrier's memory becomes
    /// plain shared memory again; any later use but `mbarrier.init` is an error.
    Inval,
}

impl MbarrierExtKind {
    /// The PTX operation, without its qualifiers.
    pub fn ptx(self) -> &'static str {
        match self {
            MbarrierExtKind::ArriveState => "mbarrier.arrive",
            MbarrierExtKind::ArriveDrop => "mbarrier.arrive_drop",
            MbarrierExtKind::ArriveDropExpectTx => "mbarrier.arrive_drop.expect_tx",
            MbarrierExtKind::ArriveDropNoComplete => "mbarrier.arrive_drop.noComplete",
            MbarrierExtKind::ArriveNoComplete => "mbarrier.arrive.noComplete",
            MbarrierExtKind::CompleteTx => "mbarrier.complete_tx",
            MbarrierExtKind::Inval => "mbarrier.inval",
        }
    }

    /// Whether it returns a state token (an i64).
    pub fn has_state(self) -> bool {
        !matches!(self, MbarrierExtKind::CompleteTx | MbarrierExtKind::Inval)
    }

    /// Whether it takes a value: an arrival count or transaction bytes.
    pub fn has_value(self) -> bool {
        matches!(
            self,
            MbarrierExtKind::ArriveDropExpectTx
                | MbarrierExtKind::ArriveDropNoComplete
                | MbarrierExtKind::ArriveNoComplete
                | MbarrierExtKind::CompleteTx
        )
    }
}

/// The per-thread markers and hints of roadmap part m11-prims-nvvm-layer
/// ([`Op::PrimMarker`]), which the simulator counts and otherwise skips.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PrimMarker {
    /// `nanosleep.u32 t`: suspend the thread for about `t` nanoseconds (a
    /// hint; the simulator yields).
    Nanosleep(ValueId),
    /// `pmevent.mask m`: trigger the performance-monitor events of the
    /// 16-bit mask.
    Pmevent(u16),
    /// `brkpt`: suspend the thread for a debugger.
    Breakpoint,
    /// A `// text` comment line in the PTX.
    Comment(String),
}

/// Warp-level matrix multiply-accumulate instructions (`mma.sync`, and the
/// sparse `mma.sp::ordered_metadata.sync`). The name gives the shape, then
/// the accumulator type and the operand type; `Sp` kinds are sparse.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaKind {
    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    M16N8K16F32F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
    M16N8K8F32F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
    M16N8K16F16F16,
    /// `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
    M16N8K8F16F16,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    M16N8K16F32BF16,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32`
    M16N8K8F32BF16,
    /// `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K4Tf32,
    /// `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: f32 operands
    /// truncated to 19 bits.
    M16N8K8Tf32,
    /// `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`
    M8N8K4F64,
    /// `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (sm_89, PTX 8.4)
    M16N8K32F32E4M3,
    /// `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` (sm_89, PTX 8.7)
    M16N8K32F16E4M3,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32` (sm_89, PTX 8.7)
    M16N8K16F32E4M3,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16` (sm_89, PTX 8.7)
    M16N8K16F16E4M3,
    /// `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` (sm_89, PTX 8.4)
    M16N8K32F32E5M2,
    /// `mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16` (sm_89, PTX 8.7)
    M16N8K32F16E5M2,
    /// `mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32` (sm_89, PTX 8.7)
    M16N8K16F32E5M2,
    /// `mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16` (sm_89, PTX 8.7)
    M16N8K16F16E5M2,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    SpM16N8K16F32F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
    SpM16N8K16F16F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    SpM16N8K16F32BF16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32`
    SpM16N8K32F32F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16`
    SpM16N8K32F16F16,
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32`
    SpM16N8K32F32BF16,
}

impl MmaKind {
    pub fn shape_mnk(self) -> (usize, usize, usize) {
        use MmaKind::*;
        match self {
            M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 => (16, 8, 16),
            M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 | M16N8K8Tf32 => (16, 8, 8),
            M16N8K4Tf32 => (16, 8, 4),
            M8N8K4F64 => (8, 8, 4),
            M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => (16, 8, 32),
            M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (16, 8, 16),
            SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16 => (16, 8, 16),
            SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => (16, 8, 32),
        }
    }

    /// Values held per thread for (A, B, C/D). A sparse kind's A fragment
    /// holds the kept half of A (two values of every four along K).
    pub fn fragment_sizes(self) -> (usize, usize, usize) {
        use MmaKind::*;
        match self {
            M16N8K16F32F16 | M16N8K16F16F16 | M16N8K16F32BF16 => (8, 4, 4),
            M16N8K8F32F16 | M16N8K8F16F16 | M16N8K8F32BF16 | M16N8K8Tf32 => (4, 2, 4),
            M16N8K4Tf32 => (2, 1, 4),
            M8N8K4F64 => (1, 1, 2),
            M16N8K32F32E4M3 | M16N8K32F16E4M3 | M16N8K32F32E5M2 | M16N8K32F16E5M2 => (16, 8, 4),
            M16N8K16F32E4M3 | M16N8K16F16E4M3 | M16N8K16F32E5M2 | M16N8K16F16E5M2 => (8, 4, 4),
            SpM16N8K16F32F16 | SpM16N8K16F16F16 | SpM16N8K16F32BF16 => (4, 4, 4),
            SpM16N8K32F32F16 | SpM16N8K32F16F16 | SpM16N8K32F32BF16 => (8, 8, 4),
        }
    }

    /// Element types of (A, B, C/D). TF32 operands are passed as `f32`.
    pub fn dtypes(self) -> (DType, DType, DType) {
        use MmaKind::*;
        let (ab, c) = match self {
            M16N8K16F32F16 | M16N8K8F32F16 | SpM16N8K16F32F16 | SpM16N8K32F32F16 => (DType::F16, DType::F32),
            M16N8K16F16F16 | M16N8K8F16F16 | SpM16N8K16F16F16 | SpM16N8K32F16F16 => (DType::F16, DType::F16),
            M16N8K16F32BF16 | M16N8K8F32BF16 | SpM16N8K16F32BF16 | SpM16N8K32F32BF16 => (DType::BF16, DType::F32),
            M16N8K4Tf32 | M16N8K8Tf32 => (DType::F32, DType::F32),
            M8N8K4F64 => (DType::F64, DType::F64),
            M16N8K32F32E4M3 | M16N8K16F32E4M3 => (DType::F8E4M3FN, DType::F32),
            M16N8K32F16E4M3 | M16N8K16F16E4M3 => (DType::F8E4M3FN, DType::F16),
            M16N8K32F32E5M2 | M16N8K16F32E5M2 => (DType::F8E5M2, DType::F32),
            M16N8K32F16E5M2 | M16N8K16F16E5M2 => (DType::F8E5M2, DType::F16),
        };
        (ab, ab, c)
    }

    /// Whether the operands are TF32 (f32 registers whose low 13 mantissa
    /// bits the tensor core ignores).
    pub fn is_tf32(self) -> bool {
        matches!(self, MmaKind::M16N8K4Tf32 | MmaKind::M16N8K8Tf32)
    }

    /// Whether the operands are FP8 (`.e4m3` or `.e5m2`, sm_89).
    pub fn is_fp8(self) -> bool {
        self.dtypes().0.is_fp8()
    }

    /// Whether this is a sparse `mma.sp` (A structured 2:4 sparse, with a
    /// metadata operand).
    pub fn is_sparse(self) -> bool {
        use MmaKind::*;
        matches!(
            self,
            SpM16N8K16F32F16
                | SpM16N8K16F16F16
                | SpM16N8K16F32BF16
                | SpM16N8K32F32F16
                | SpM16N8K32F16F16
                | SpM16N8K32F32BF16
        )
    }

    /// The PTX name of the operand type: `f16`, `tf32`, `e4m3`.
    fn ptx_operand(self) -> &'static str {
        match self.dtypes().0 {
            _ if self.is_tf32() => "tf32",
            DType::F8E4M3FN => "e4m3",
            DType::F8E5M2 => "e5m2",
            a => a.name(),
        }
    }

    /// The PTX instruction, e.g. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        let c = self.dtypes().2.name();
        let ab = self.ptx_operand();
        let op = if self.is_sparse() { "mma.sp::ordered_metadata.sync" } else { "mma.sync" };
        format!("{op}.aligned.m{m}n{n}k{k}.row.col.{c}.{ab}.{ab}.{c}")
    }
}

// m10-sm120-blockscaled: SM120 warp-level block-scaled mma.sync kinds.

/// A warp-level block-scaled `mma.sync` (sm_120a, PTX ISA 8.7): its A and B
/// formats and its scale-factor vector size. With FP4 (`F4E2M1FN`) A and B
/// it is `m16n8k64` (`kind::mxf4` with UE8M0 scales per 32 elements, or
/// `kind::mxf4nvf4` with UE4M3 scales per 16), with A and B packed eight
/// values per register; otherwise `m16n8k32 kind::mxf8f6f4` with UE8M0
/// scales per 32 elements, four 8-bit containers per register (an FP4 value
/// in bits 5:2 of its container). Built by the `cute::warp` operations,
/// which check the combination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockScaledMmaKind {
    pub a: DType,
    pub b: DType,
    pub sf_vec_size: u8,
}

impl BlockScaledMmaKind {
    /// Whether A and B are both FP4, packed eight per register (`m16n8k64`).
    pub fn is_fp4_packed(self) -> bool {
        self.a == DType::F4E2M1FN && self.b == DType::F4E2M1FN
    }

    pub fn shape_mnk(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (16, 8, 64) } else { (16, 8, 32) }
    }

    /// The scale-factor type: `F8E4M3FN` (read as UE4M3) for 16-element
    /// vectors, `F8E8M0FNU` (UE8M0) for 32.
    pub fn sf_dtype(self) -> DType {
        if self.sf_vec_size == 16 { DType::F8E4M3FN } else { DType::F8E8M0FNU }
    }

    /// Scale factors per row of A (and column of B) per instruction: 1, 2 or 4.
    pub fn scales_per_mma(self) -> usize {
        self.shape_mnk().2 / self.sf_vec_size as usize
    }

    /// `mxf4`, `mxf4nvf4` or `mxf8f6f4`.
    pub fn kind_name(self) -> &'static str {
        match (self.is_fp4_packed(), self.sf_vec_size) {
            (true, 16) => "mxf4nvf4",
            (true, _) => "mxf4",
            _ => "mxf8f6f4",
        }
    }

    /// The PTX name of an operand type: `e2m1`, `e4m3` or `e5m2`.
    pub fn ptx_type(t: DType) -> &'static str {
        match t {
            DType::F4E2M1FN => "e2m1",
            DType::F8E4M3FN => "e4m3",
            _ => "e5m2",
        }
    }

    /// `ue8m0` or `ue4m3`.
    pub fn ptx_sf_type(self) -> &'static str {
        if self.sf_vec_size == 16 { "ue4m3" } else { "ue8m0" }
    }

    /// Values per thread of (A, B, C/D).
    pub fn fragment_sizes(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (32, 16, 4) } else { (16, 8, 4) }
    }

    /// 32-bit registers per thread of (A, B): 4 and 2.
    pub fn operand_registers(self) -> (usize, usize) {
        (4, 2)
    }

    /// Values per register of an operand: 8 packed FP4 values, or 4 8-bit
    /// containers.
    pub fn values_per_register(self) -> usize {
        if self.is_fp4_packed() { 8 } else { 4 }
    }

    /// The PTX instruction, e.g.
    /// `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        format!(
            "mma.sync.aligned.m{m}n{n}k{k}.row.col.kind::{}.block_scale.scale_vec::{}X.f32.{}.{}.f32.{}",
            self.kind_name(),
            self.scales_per_mma(),
            Self::ptx_type(self.a),
            Self::ptx_type(self.b),
            self.ptx_sf_type()
        )
    }
}

// m11-prims-mma-forms.
/// A warp-level sparse block-scaled `mma.sp::ordered_metadata` (sm_120a,
/// PTX ISA 8.7; CuTe C++'s `SM120_SPARSE_16x8x64_TN_VS` and
/// `SM120_SPARSE_16x8x128_TN_VS`): A 2:4 sparse (FP4: pairs, 4:8), stored
/// compressed. With FP4 A and B (`kind::mxf4` or `mxf4nvf4`) it is
/// `m16n8k128`, otherwise `m16n8k64 kind::mxf8f6f4` (8-bit containers).
/// `scale_vec` (1, 2 or 4) scale factors per row of A and column of B, of
/// type `sf` (`F8E8M0FNU`, or `F8E4M3FN` read as UE4M3), each from byte
/// `bid + j` of the register of the lane `tid` selects.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpBlockScaledMmaKind {
    pub a: DType,
    pub b: DType,
    pub scale_vec: u8,
    pub sf: DType,
    pub bid_a: u8,
    pub tid_a: u8,
    pub bid_b: u8,
    pub tid_b: u8,
}

impl SpBlockScaledMmaKind {
    /// Whether A and B are both FP4, packed eight per register.
    pub fn is_fp4_packed(self) -> bool {
        self.a == DType::F4E2M1FN && self.b == DType::F4E2M1FN
    }

    /// `(16, 8, 128)` for packed FP4, `(16, 8, 64)` otherwise (logical K).
    pub fn shape_mnk(self) -> (usize, usize, usize) {
        if self.is_fp4_packed() { (16, 8, 128) } else { (16, 8, 64) }
    }

    /// Logical K elements per scale factor.
    pub fn vec(self) -> usize {
        self.shape_mnk().2 / self.scale_vec as usize
    }

    /// `mxf4`, `mxf4nvf4` or `mxf8f6f4`.
    pub fn kind_name(self) -> &'static str {
        match (self.is_fp4_packed(), self.scale_vec) {
            (true, 4) => "mxf4nvf4",
            (true, _) if self.sf == DType::F8E4M3FN => "mxf4nvf4",
            (true, _) => "mxf4",
            _ => "mxf8f6f4",
        }
    }

    /// `ue8m0` or `ue4m3`.
    pub fn ptx_sf_type(self) -> &'static str {
        if self.sf == DType::F8E4M3FN { "ue4m3" } else { "ue8m0" }
    }

    /// The PTX instruction.
    pub fn ptx(self) -> String {
        let (m, n, k) = self.shape_mnk();
        format!(
            "mma.sp::ordered_metadata.sync.aligned.m{m}n{n}k{k}.row.col.kind::{}.block_scale.scale_vec::{}X.f32.{}.{}.f32.{}",
            self.kind_name(),
            self.scale_vec,
            BlockScaledMmaKind::ptx_type(self.a),
            BlockScaledMmaKind::ptx_type(self.b),
            self.ptx_sf_type()
        )
    }

    /// Values per register: 8 packed FP4 values or 4 8-bit containers.
    pub fn values_per_register(self) -> usize {
        if self.is_fp4_packed() { 8 } else { 4 }
    }
}

/// IEEE 754 rounding directions (CuTe DSL's `cute.RoundingMode`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RoundingMode {
    /// To nearest, ties to even (`.rn`).
    NearestEven,
    /// Toward zero (`.rz`).
    Zero,
    /// Toward +infinity (`.rp`).
    Up,
    /// Toward -infinity (`.rm`).
    Down,
}

impl RoundingMode {
    /// The PTX rounding modifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            RoundingMode::NearestEven => "rn",
            RoundingMode::Zero => "rz",
            RoundingMode::Up => "rp",
            RoundingMode::Down => "rm",
        }
    }
}

/// The PTX floating-point instructions that [`Op::FloatInst`] issues with
/// explicit modifiers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatFn {
    /// `add`
    Add,
    /// `mul`
    Mul,
    /// `fma`
    Fma,
    /// `div`
    Div,
    /// `rcp`: `1 / x`.
    Rcp,
    /// `sqrt`
    Sqrt,
    /// `rsqrt`: `1 / sqrt(x)`.
    Rsqrt,
    /// `ex2`: `2^x`.
    Ex2,
    /// `lg2`: `log2(x)`.
    Lg2,
    /// `sin`
    Sin,
    /// `cos`
    Cos,
    /// `tanh`
    Tanh,
    /// `abs`
    Abs,
    /// `min`
    Min,
    /// `max`
    Max,
}

impl FloatFn {
    /// The PTX instruction name.
    pub fn ptx(self) -> &'static str {
        match self {
            FloatFn::Add => "add",
            FloatFn::Mul => "mul",
            FloatFn::Fma => "fma",
            FloatFn::Div => "div",
            FloatFn::Rcp => "rcp",
            FloatFn::Sqrt => "sqrt",
            FloatFn::Rsqrt => "rsqrt",
            FloatFn::Ex2 => "ex2",
            FloatFn::Lg2 => "lg2",
            FloatFn::Sin => "sin",
            FloatFn::Cos => "cos",
            FloatFn::Tanh => "tanh",
            FloatFn::Abs => "abs",
            FloatFn::Min => "min",
            FloatFn::Max => "max",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self {
            FloatFn::Fma => 3,
            FloatFn::Add | FloatFn::Mul | FloatFn::Div | FloatFn::Min | FloatFn::Max => 2,
            _ => 1,
        }
    }
}

/// How an [`Op::FloatInst`] computes its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatMode {
    /// IEEE 754, rounded in the given direction (`.rn`, `.rz`, `.rp`, `.rm`).
    Round(RoundingMode),
    /// The hardware approximation (`.approx`).
    Approx,
    /// `div.full`: a full-range approximate division.
    Full,
    /// Exact results that need no rounding (`abs`, `min`, `max`).
    Exact,
}

impl FloatMode {
    /// The PTX modifier, with its dot (empty for [`FloatMode::Exact`]).
    pub fn ptx(self) -> String {
        match self {
            FloatMode::Round(r) => format!(".{}", r.ptx()),
            FloatMode::Approx => ".approx".into(),
            FloatMode::Full => ".full".into(),
            FloatMode::Exact => String::new(),
        }
    }
}

/// The PTX instruction an [`Op::FloatInst`] issues, e.g. `ex2.approx.ftz.f32`
/// or `max.NaN.xorsign.abs.f32`.
pub fn float_inst_name(func: FloatFn, mode: FloatMode, ftz: bool, nan: bool, xorsign_abs: bool, ty: DType) -> String {
    format!(
        "{}{}{}{}{}.{}",
        func.ptx(),
        mode.ptx(),
        if ftz { ".ftz" } else { "" },
        if nan { ".NaN" } else { "" },
        if xorsign_abs { ".xorsign.abs" } else { "" },
        ty.name()
    )
}

/// How `prmt` picks the four bytes of its result from the eight bytes of its
/// two sources (`{b, a}`, byte 0 the low byte of `a`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PrmtMode {
    /// The generic form, CuTe DSL's `prmt`: the low 16 bits of the control
    /// operand are four 4-bit selectors, one per result byte. A selector's
    /// low 3 bits name a source byte, and its top bit replicates that
    /// byte's sign bit instead of copying it.
    #[default]
    Generic,
    /// `.f4e` (forward 4 extract): result byte `i` is source byte `s + i`,
    /// where `s` is the low 2 bits of the control operand.
    F4e,
    /// `.b4e` (backward 4 extract): result byte `i` is source byte
    /// `(s - i) mod 8`.
    B4e,
    /// `.rc8` (replicate 8): every result byte is source byte `s`.
    Rc8,
    /// `.ecl` (edge clamp left): result byte `i` is source byte `max(i, s)`.
    Ecl,
    /// `.ecr` (edge clamp right): result byte `i` is source byte `min(i, s)`.
    Ecr,
    /// `.rc16` (replicate 16): both halves are source half-word `s & 1`.
    Rc16,
}

impl PrmtMode {
    /// The PTX modifier, with its dot (empty for the generic form).
    pub fn ptx(self) -> &'static str {
        match self {
            PrmtMode::Generic => "",
            PrmtMode::F4e => ".f4e",
            PrmtMode::B4e => ".b4e",
            PrmtMode::Rc8 => ".rc8",
            PrmtMode::Ecl => ".ecl",
            PrmtMode::Ecr => ".ecr",
            PrmtMode::Rc16 => ".rc16",
        }
    }
}

/// The PTX integer and bit instructions that [`Op::IntInst`] issues: CuTe
/// DSL's `cute.arch` integer helpers. Immediates (`lop3`'s lookup table)
/// and modifiers belong to the function; the operands are values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntFn {
    /// `bfe.{s,u}{32,64}`: extract a bit field. Operands: value, start,
    /// length.
    Bfe,
    /// `bfi.b{32,64}`: insert a bit field. Operands: replacement (the
    /// field), value, start, length.
    Bfi,
    /// `lop3.b32` with the lookup table `lut` as its immediate: any bitwise
    /// function of three operands.
    Lop3 { lut: u8 },
    /// `shf.{l,r}.{clamp,wrap}.b32`, a funnel shift: operand a holds bits
    /// 31:0 of a 64-bit value and operand b bits 63:32; the third operand is
    /// the shift amount, clamped to 32 or taken mod 32.
    Shf { left: bool, clamp: bool },
    /// `prmt.b32{.mode}`: pick four of the eight bytes of operands a (bytes
    /// 0-3) and b (bytes 4-7), as the third operand selects.
    Prmt { mode: PrmtMode },
    /// `mul.wide.{s,u}{16,32}`: the whole product of two 16- or 32-bit
    /// integers, as an integer twice as wide.
    MulWide,
    /// `mul24.{lo,hi}.{s,u}32`: the 48-bit product of the low 24 bits of two
    /// 32-bit integers, its bits 31:0 (`lo`) or 47:16 (`hi`).
    Mul24 { hi: bool },
    /// `mad24.{lo,hi}.{s,u}32`: [`IntFn::Mul24`] plus a third operand,
    /// wrapping.
    Mad24 { hi: bool },
    /// `add.sat.s32`: a sum clamped to the Int32 range.
    AddSat,
    /// `sub.sat.s32`: a difference clamped to the Int32 range.
    SubSat,
    /// `dp4a.s32.s32`: operand c plus the dot product of the four signed
    /// bytes of a with the four signed bytes of b. CuTe DSL's int8
    /// conversions (`cvt_i8x4_to_f32x4`) sign-extend each byte with it.
    Dp4a,
}

impl IntFn {
    /// CuTe DSL's name for the operation (`cute.arch.<name>`), for messages.
    pub fn cute_name(self) -> &'static str {
        match self {
            IntFn::Bfe => "bfe",
            IntFn::Bfi => "bfi",
            IntFn::Lop3 { .. } => "lop3",
            IntFn::Shf { .. } => "shf",
            IntFn::Prmt { .. } => "prmt",
            IntFn::MulWide => "mul_wide",
            IntFn::Mul24 { .. } => "mul24",
            IntFn::Mad24 { .. } => "mad24",
            IntFn::AddSat => "add_sat_int",
            IntFn::SubSat => "sub_sat_int",
            IntFn::Dp4a => "dp4a",
        }
    }

    /// The operands' names, as CuTe DSL's parameters.
    pub fn operand_names(self) -> &'static [&'static str] {
        match self {
            IntFn::Bfe => &["value", "start", "length"],
            IntFn::Bfi => &["replacement", "value", "start", "length"],
            IntFn::Lop3 { .. } | IntFn::Mad24 { .. } | IntFn::Dp4a => &["a", "b", "c"],
            IntFn::Shf { .. } => &["a", "b", "shift"],
            IntFn::Prmt { .. } => &["src", "src_reg_shifted", "prmt_indices"],
            IntFn::MulWide | IntFn::Mul24 { .. } | IntFn::AddSat | IntFn::SubSat => &["a", "b"],
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        self.operand_names().len()
    }

    /// How many leading operands are data, all of one type; the rest are
    /// Int32 bit positions, lengths or shift amounts.
    pub fn data_operands(self) -> usize {
        match self {
            IntFn::Bfe => 1,
            IntFn::Bfi | IntFn::Shf { .. } => 2,
            _ => self.arity(),
        }
    }

    /// The types the data operands may have.
    pub fn operand_types(self) -> &'static [DType] {
        const WORDS: &[DType] = &[DType::I32, DType::U32, DType::I64, DType::U64];
        const WORD32: &[DType] = &[DType::I32, DType::U32];
        match self {
            IntFn::Bfe | IntFn::Bfi => WORDS,
            IntFn::Lop3 { .. } | IntFn::Shf { .. } | IntFn::Prmt { .. } | IntFn::Mul24 { .. } | IntFn::Mad24 { .. } => {
                WORD32
            }
            IntFn::MulWide => &[DType::I16, DType::U16, DType::I32, DType::U32],
            IntFn::AddSat | IntFn::SubSat | IntFn::Dp4a => &[DType::I32],
        }
    }

    /// The result type for data operands of type `ty`: twice as wide for
    /// [`IntFn::MulWide`], `ty` itself for the others.
    pub fn result_type(self, ty: DType) -> DType {
        match (self, ty) {
            (IntFn::MulWide, DType::I16) => DType::I32,
            (IntFn::MulWide, DType::U16) => DType::U32,
            (IntFn::MulWide, DType::I32) => DType::I64,
            (IntFn::MulWide, DType::U32) => DType::U64,
            _ => ty,
        }
    }

    /// Whether the first two operands can be swapped.
    pub fn is_commutative(self) -> bool {
        matches!(self, IntFn::MulWide | IntFn::Mul24 { .. } | IntFn::Mad24 { .. } | IntFn::AddSat | IntFn::Dp4a)
    }
}

/// The PTX instruction an [`Op::IntInst`] issues on data operands of type
/// `ty`, e.g. `bfe.s64` or `shf.l.clamp.b32` (immediates not included).
pub fn int_inst_name(func: IntFn, ty: DType) -> String {
    let sign = if ty.is_signed() { 's' } else { 'u' };
    let half = |hi: bool| if hi { "hi" } else { "lo" };
    let w = ty.bits();
    match func {
        IntFn::Bfe => format!("bfe.{sign}{w}"),
        IntFn::Bfi => format!("bfi.b{w}"),
        IntFn::Lop3 { .. } => "lop3.b32".into(),
        IntFn::Shf { left, clamp } => {
            format!("shf.{}.{}.b32", if left { "l" } else { "r" }, if clamp { "clamp" } else { "wrap" })
        }
        IntFn::Prmt { mode } => format!("prmt.b32{}", mode.ptx()),
        IntFn::MulWide => format!("mul.wide.{sign}{w}"),
        IntFn::Mul24 { hi } => format!("mul24.{}.{sign}32", half(hi)),
        IntFn::Mad24 { hi } => format!("mad24.{}.{sign}32", half(hi)),
        IntFn::AddSat => "add.sat.s32".into(),
        IntFn::SubSat => "sub.sat.s32".into(),
        IntFn::Dp4a => "dp4a.s32.s32".into(),
    }
}

/// The PTX half-precision floating-point instructions that [`Op::HalfInst`]
/// issues: CuTe DSL's packed and scalar half helpers (`add_packed_f16x2`,
/// `fma_bf16`, `exp2_packed_bf16x2`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HalfFn {
    /// `add`: `a + b`.
    Add,
    /// `sub`: `a - b`.
    Sub,
    /// `fma`: `a * b + c`, rounded once.
    Fma,
    /// `ex2.approx`: an approximation of `2^a`.
    Ex2,
}

impl HalfFn {
    /// The PTX instruction name.
    pub fn ptx(self) -> &'static str {
        match self {
            HalfFn::Add => "add",
            HalfFn::Sub => "sub",
            HalfFn::Fma => "fma",
            HalfFn::Ex2 => "ex2",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self {
            HalfFn::Add | HalfFn::Sub => 2,
            HalfFn::Fma => 3,
            HalfFn::Ex2 => 1,
        }
    }
}

/// The operand types of PTX's half-precision instructions: one 16-bit
/// float, or two of them packed into a 32-bit register with element 0 in
/// bits 15:0 (`.f16x2`, `.bf16x2`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HalfType {
    F16,
    Bf16,
    F16x2,
    Bf16x2,
}

impl HalfType {
    /// The PTX type suffix, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            HalfType::F16 => "f16",
            HalfType::Bf16 => "bf16",
            HalfType::F16x2 => "f16x2",
            HalfType::Bf16x2 => "bf16x2",
        }
    }

    /// The element type, Float16 or BFloat16.
    pub fn elem(self) -> DType {
        match self {
            HalfType::F16 | HalfType::F16x2 => DType::F16,
            HalfType::Bf16 | HalfType::Bf16x2 => DType::BF16,
        }
    }

    /// Whether two elements are packed into one Int32.
    pub fn is_packed(self) -> bool {
        matches!(self, HalfType::F16x2 | HalfType::Bf16x2)
    }

    /// The type of the operands and the result in the IR: Int32 for the
    /// packed forms, the element type otherwise.
    pub fn value_type(self) -> DType {
        if self.is_packed() { DType::I32 } else { self.elem() }
    }
}

/// The modifiers of an [`Op::HalfInst`] (CuTe DSL's `rnd='rn'`, `oob=`,
/// `ftz=`, `sat=` and `relu=`), each a PTX modifier written in this order:
/// `fma.rn.oob.relu.bf16x2`, `add.rn.ftz.sat.f16x2`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct HalfModifiers {
    /// `.rn`: round to nearest even, explicitly (`add` and `sub` round so
    /// without it too, but may then be fused with a multiplication).
    pub rn: bool,
    /// `.oob`: a zero result where an operand is the out-of-bounds NaN.
    pub oob: bool,
    /// `.ftz`: flush subnormal inputs and results to zero (Float16 only).
    pub ftz: bool,
    /// `.sat`: clamp the result to `[0, 1]`, a NaN to `+0` (Float16 only).
    pub sat: bool,
    /// `.relu`: a negative result becomes `+0`, a NaN the canonical NaN.
    pub relu: bool,
}

/// The PTX instruction an [`Op::HalfInst`] issues, e.g.
/// `fma.rn.oob.relu.bf16x2`, `add.rn.ftz.sat.f16x2` or
/// `ex2.approx.ftz.bf16x2` (the BFloat16 `ex2` always flushes subnormals).
pub fn half_inst_name(func: HalfFn, ty: HalfType, mods: HalfModifiers) -> String {
    if func == HalfFn::Ex2 {
        let ftz = if ty.elem() == DType::BF16 { ".ftz" } else { "" };
        return format!("ex2.approx{ftz}.{}", ty.ptx());
    }
    let flag = |on: bool, text: &'static str| if on { text } else { "" };
    format!(
        "{}{}{}{}{}{}.{}",
        func.ptx(),
        flag(mods.rn, ".rn"),
        flag(mods.oob, ".oob"),
        flag(mods.ftz, ".ftz"),
        flag(mods.sat, ".sat"),
        flag(mods.relu, ".relu"),
        ty.ptx()
    )
}

/// The two FP8 formats PTX converts to and from on sm_89 and newer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fp8Kind {
    /// OCP E4M3 (`Float8E4M3FN`, PTX `.e4m3`).
    E4M3,
    /// OCP E5M2 (`Float8E5M2`, PTX `.e5m2`).
    E5M2,
}

impl Fp8Kind {
    /// The format's type.
    pub fn dtype(self) -> DType {
        match self {
            Fp8Kind::E4M3 => DType::F8E4M3FN,
            Fp8Kind::E5M2 => DType::F8E5M2,
        }
    }

    /// The PTX pair type, `e4m3x2` or `e5m2x2`.
    pub fn ptx(self) -> &'static str {
        match self {
            Fp8Kind::E4M3 => "e4m3x2",
            Fp8Kind::E5M2 => "e5m2x2",
        }
    }
}

/// The shape and element type of an [`Op::StMatrix`] (the PTX ISA's
/// `stmatrix`, "Collectively store one or more matrices to shared memory").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StMatrixForm {
    /// `.m8n8` with `.b16` (sm_90, PTX 7.8): 8x8 matrices of 16-bit
    /// elements, optionally transposed.
    #[default]
    M8n8B16,
    /// `.m16n8` with `.b8`, `.trans` mandatory: 16x8 matrices of 8-bit
    /// elements stored transposed. Architecture-specific (sm_100a, sm_101a or
    /// sm_110a and sm_120a from PTX 8.6; the f families from 8.8).
    M16n8B8,
}

impl StMatrixForm {
    /// The PTX shape qualifier.
    pub fn shape(self) -> &'static str {
        match self {
            StMatrixForm::M8n8B16 => "m8n8",
            StMatrixForm::M16n8B8 => "m16n8",
        }
    }

    /// The PTX element type.
    pub fn format(self) -> &'static str {
        match self {
            StMatrixForm::M8n8B16 => "b16",
            StMatrixForm::M16n8B8 => "b8",
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self, num: u32, trans: bool) -> String {
        format!(
            "stmatrix.sync.aligned.{}.x{num}{}.shared.{}",
            self.shape(),
            if trans { ".trans" } else { "" },
            self.format()
        )
    }
}

/// The shape and element format of an [`Op::LdMatrix`] (the PTX ISA's
/// `ldmatrix`, "Warp-level matrix load instruction").
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum LdMatrixForm {
    /// `.m8n8` with `.b16`: 8x8 matrices of 16-bit elements (sm_75, PTX 6.5).
    #[default]
    M8n8B16,
    /// `.m16n16` (`.trans` mandatory, `.x1` or `.x2`): 16x16 matrices of
    /// 8-bit elements, `.b8`, or with `unpack` 4 or 6 the padded sub-byte
    /// sources `.b8x16.b4x16_p64` and `.b8x16.b6x16_p32` expanded to 8-bit
    /// containers. Architecture-specific (sm_100a, sm_101a/sm_110a, sm_120a
    /// from PTX 8.6; the f families from 8.8).
    M16n16B8 { unpack: Option<u8> },
    /// `.m8n16` (`.x1`, `.x2` or `.x4`): 8x16 matrices of padded 4-bit
    /// (`unpack` 4, `.b8x16.b4x16_p64`) or 6-bit (`unpack` 6,
    /// `.b8x16.b6x16_p32`) elements expanded to 8-bit containers. The
    /// targets of [`LdMatrixForm::M16n16B8`].
    M8n16B8 { unpack: u8 },
}

impl LdMatrixForm {
    /// The PTX shape qualifier.
    pub fn shape(self) -> &'static str {
        match self {
            LdMatrixForm::M8n8B16 => "m8n8",
            LdMatrixForm::M16n16B8 { .. } => "m16n16",
            LdMatrixForm::M8n16B8 { .. } => "m8n16",
        }
    }

    /// The PTX element type or `.dst_fmt.src_fmt` pair.
    pub fn format(self) -> &'static str {
        match self.unpack() {
            None if self == LdMatrixForm::M8n8B16 => "b16",
            None => "b8",
            Some(4) => "b8x16.b4x16_p64",
            Some(_) => "b8x16.b6x16_p32",
        }
    }

    /// The source element width that is expanded to 8 bits (4 or 6), if any.
    pub fn unpack(self) -> Option<u8> {
        match self {
            LdMatrixForm::M8n8B16 => None,
            LdMatrixForm::M16n16B8 { unpack } => unpack,
            LdMatrixForm::M8n16B8 { unpack } => Some(unpack),
        }
    }

    /// The rows (one per supplying lane) of each matrix.
    pub fn rows(self) -> u32 {
        match self {
            LdMatrixForm::M16n16B8 { .. } => 16,
            _ => 8,
        }
    }

    /// The results per matrix: two 16-bit values for `.m8n8`, two 32-bit
    /// registers for `.m16n16` and one for `.m8n16`.
    pub fn results_per_matrix(self) -> u32 {
        match self {
            LdMatrixForm::M8n8B16 | LdMatrixForm::M16n16B8 { .. } => 2,
            LdMatrixForm::M8n16B8 { .. } => 1,
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self, num: u32, trans: bool) -> String {
        format!(
            "ldmatrix.sync.aligned.{}.x{num}{}.shared.{}",
            self.shape(),
            if trans { ".trans" } else { "" },
            self.format()
        )
    }
}

/// The OCP MX FP4 and FP6 formats PTX converts to and from in pairs
/// (sm_100a, sm_101a and sm_120a, PTX 8.6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MxKind {
    /// FP4 E2M1 (`Float4E2M1FN`, PTX `.e2m1x2`: a pair in one byte).
    E2M1,
    /// FP6 E2M3 (`Float6E2M3FN`, PTX `.e2m3x2`: a pair in two bytes, each
    /// value in a byte's low 6 bits).
    E2M3,
    /// FP6 E3M2 (`Float6E3M2FN`, PTX `.e3m2x2`).
    E3M2,
}

impl MxKind {
    /// The format's type.
    pub fn dtype(self) -> DType {
        match self {
            MxKind::E2M1 => DType::F4E2M1FN,
            MxKind::E2M3 => DType::F6E2M3FN,
            MxKind::E3M2 => DType::F6E3M2FN,
        }
    }

    /// The PTX pair type, `e2m1x2`, `e2m3x2` or `e3m2x2`.
    pub fn ptx(self) -> &'static str {
        match self {
            MxKind::E2M1 => "e2m1x2",
            MxKind::E2M3 => "e2m3x2",
            MxKind::E3M2 => "e3m2x2",
        }
    }
}

/// The arithmetic of a [`CvtFn::Packed`] instruction (roadmap part
/// m10-ldred-packed-math): PTX's packed `.f32x2` operations (sm_100, PTX
/// 8.6) and the Rubin-only mixed-precision ones (sm_107f, PTX 9.4). A pair of
/// Float32 values is a `u64` (element 0 in the low half) and a pair of
/// Float16 or BFloat16 values an `i32`; each element is computed on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PackedOp {
    /// `add{.rnd}{.ftz}.f32x2`: `a + b` on two Float32 pairs.
    Add,
    /// `sub{.rnd}{.ftz}.f32x2`.
    Sub,
    /// `mul{.rnd}{.ftz}.f32x2`.
    Mul,
    /// `fma.rnd{.ftz}.f32x2`: `a * b + c`, rounded once.
    Fma,
    /// `add.rn.{f16x2,bf16x2}.f32x2`: two Float32 sums, each rounded once
    /// to a Float16 or BFloat16 (an `i32` pair).
    AddToHalf { bf16: bool },
    /// `sub.rn.{f16x2,bf16x2}.f32x2`.
    SubToHalf { bf16: bool },
    /// `add{.rnd}.f32x2.{f16x2,bf16x2}`: a half pair (widened exactly) plus a
    /// Float32 pair, rounded once.
    AddHalf { bf16: bool },
    /// `sub{.rnd}.f32x2.{f16x2,bf16x2}`: the half pair minus the Float32 pair.
    SubHalf { bf16: bool },
    /// `fma.rnd.f32x2.{f16x2,bf16x2}`: half pair times a Float32 pair plus a
    /// Float32 pair, rounded once.
    FmaHalf { bf16: bool },
}

/// The PTX conversions that [`Op::Cvt`] issues beyond the plain numeric
/// conversions of [`Op::Cast`]: CuTe DSL's `cvt_*` helpers, and the FP8 and
/// UE8M0 conversions behind [`Value::to`](crate::Value::to). Pairs are
/// packed with element 0 in the low half (PTX takes element 1 first and
/// puts it in the high half).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CvtFn {
    /// `cvt.rn.satfinite.bf16x2.f32`: two Float32 values (element 0, then
    /// element 1) to BFloat16, rounded to nearest even and packed into an
    /// Int32 with element 0 in the low half. A NaN gives the canonical NaN
    /// and a magnitude above BFloat16's largest finite value (infinities
    /// included) that value with its sign. PTX takes element 1 first.
    F32x2ToBf16x2,
    /// `cvt.rna.tf32.f32`: a Float32 rounded to TF32 (10 stored mantissa
    /// bits, to nearest with ties away from zero), as an Int32 holding the
    /// bits with the low 13 cleared.
    F32ToTf32,
    /// `cvt.rn.bf16.s8`: an Int8 to BFloat16, exactly.
    S8ToBf16,
    /// `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2` with the scale
    /// factor `scale` for both elements (the 16-bit operand `0x{scale}{scale}`):
    /// the two bytes of an Int16 (byte 0 low), each read as an s2f6
    /// fixed-point value (its int8 value times 2^-6) and multiplied by the
    /// UE8M0 scale `2^(scale - 127)`, as BFloat16 packed into an Int32 (byte
    /// 0's in the low half). `0x85` (2^6) gives the int8 values
    /// (`cvt_i8x2_to_bf16x2`), `0x81` (2^-4 overall) the high nibbles of the
    /// bytes as signed int4 values (`cvt_i4x2_to_bf16x2(with_shuffle=True)`).
    S2f6x2ToBf16x2 { scale: u8 },
    /// `cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f32`: two Float32 values to
    /// FP8, rounded to nearest even, a magnitude past the largest finite
    /// value saturating to it (sign kept), packed into an Int16.
    F32x2ToF8x2 { to: Fp8Kind, relu: bool },
    /// `cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.f16x2`: a packed pair of
    /// Float16 values (an Int32) to FP8, as above, packed into an Int16.
    F16x2ToF8x2 { to: Fp8Kind, relu: bool },
    /// `cvt.rn{.relu}.f16x2.{e4m3x2,e5m2x2}`: a pair of FP8 values (an Int16)
    /// to Float16, exactly, packed into an Int32.
    F8x2ToF16x2 { from: Fp8Kind, relu: bool },
    /// `cvt.{rz,rp}.satfinite.ue8m0x2.f32`: two Float32 values to UE8M0
    /// (`round_up` picks `.rp`), packed into an Int16.
    F32x2ToUe8m0x2 { round_up: bool },
    /// `cvt.rp.satfinite.ue5m3x2.f32` (roadmap part m10-ldred-packed-math):
    /// two Float32 values to UE5M3 (`FloatNV8E5M3FNU`), rounded toward +inf
    /// and saturating to the largest finite value (PTX ISA 9.4), packed into
    /// an Int16 with element 0 in the low byte (PTX takes element 1 first).
    F32x2ToUe5m3x2,
    /// `cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2`: a packed pair of BFloat16
    /// values (an Int32) to UE8M0, packed into an Int16. (The PTX ISA lists
    /// a `.relu` form, which ptxas 12.8 refuses.)
    Bf16x2ToUe8m0x2 { round_up: bool, satfinite: bool },
    /// `cvt.rn.bf16x2.ue8m0x2`: a pair of UE8M0 values (an Int16) to
    /// BFloat16, exactly, packed into an Int32.
    Ue8m0x2ToBf16x2,
    /// `cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2}.f32`: two Float32
    /// values to FP4 or FP6, rounded to nearest even, a magnitude past the
    /// largest finite value (infinities included) saturating to it with its
    /// sign and a NaN giving `+MAX_NORM`, in an Int16: an FP4 pair in the low
    /// byte (element 0 in the low nibble), an FP6 pair in the two bytes'
    /// low 6 bits (element 0 in the low byte).
    F32x2ToMx { to: MxKind, relu: bool },
    /// `cvt.rn{.relu}.f16x2.{e2m1x2,e2m3x2,e3m2x2}`: an FP4 or FP6 pair, in
    /// [`CvtFn::F32x2ToMx`]'s Int16 form, to Float16, exactly, packed into an
    /// Int32 (element 0 in the low half).
    MxToF16x2 { from: MxKind, relu: bool },
    /// A packed arithmetic instruction ([`PackedOp`]) with a rounding mode
    /// (`None`: no `.rnd`, which PTX lets ptxas fuse; computed as `.rn`) and
    /// `ftz` (m10-ldred-packed-math). Pure.
    Packed { op: PackedOp, rnd: Option<RoundingMode>, ftz: bool },
}

impl CvtFn {
    /// CuTe DSL's name for the helper that issues it (cute-rs's for the
    /// FP8 and UE8M0 pairs), for messages.
    pub fn cute_name(self) -> &'static str {
        match self {
            CvtFn::F32x2ToBf16x2 => "cvt_f32x2_bf16x2",
            CvtFn::F32ToTf32 => "cvt_f32_tf32",
            CvtFn::S8ToBf16 => "cvt_i8_bf16",
            CvtFn::S2f6x2ToBf16x2 { scale: 0x85 } => "cvt_i8x2_to_bf16x2",
            CvtFn::S2f6x2ToBf16x2 { .. } => "cvt_i4_bf16_intrinsic(with_shuffle=True)",
            CvtFn::F32x2ToF8x2 { .. } => "cvt_f32x2_to_f8x2",
            CvtFn::F16x2ToF8x2 { .. } => "cvt_f16x2_to_f8x2",
            CvtFn::F8x2ToF16x2 { .. } => "cvt_f8x2_to_f16x2",
            CvtFn::F32x2ToUe8m0x2 { .. } => "cvt_f32x2_to_ue8m0x2",
            CvtFn::F32x2ToUe5m3x2 => "cvt_f32x4_to_fnv8e5m3x4",
            CvtFn::Bf16x2ToUe8m0x2 { .. } => "cvt_bf16x2_to_ue8m0x2",
            CvtFn::Ue8m0x2ToBf16x2 => "cvt_ue8m0x2_to_bf16x2",
            CvtFn::F32x2ToMx { to: MxKind::E2M1, .. } => "cvt_f32x2_to_f4e2m1x2",
            CvtFn::F32x2ToMx { .. } => "cvt_f32x2_to_f6x2",
            CvtFn::MxToF16x2 { from: MxKind::E2M1, .. } => "cvt_f4e2m1x2_to_f16x2",
            CvtFn::MxToF16x2 { .. } => "cvt_f6x2_to_f16x2",
            CvtFn::Packed { op, .. } => match op {
                PackedOp::Add => "add_packed_f32x2",
                PackedOp::Sub => "sub_packed_f32x2",
                PackedOp::Mul => "mul_packed_f32x2",
                PackedOp::Fma => "fma_packed_f32x2",
                PackedOp::AddToHalf { bf16: false } => "add_packed_f16x2_f32x2_f32x2",
                PackedOp::AddToHalf { bf16: true } => "add_packed_bf16x2_f32x2_f32x2",
                PackedOp::SubToHalf { bf16: false } => "sub_packed_f16x2_f32x2_f32x2",
                PackedOp::SubToHalf { bf16: true } => "sub_packed_bf16x2_f32x2_f32x2",
                PackedOp::AddHalf { bf16: false } => "add_packed_f32x2_f16x2_f32x2",
                PackedOp::AddHalf { bf16: true } => "add_packed_f32x2_bf16x2_f32x2",
                PackedOp::SubHalf { bf16: false } => "sub_packed_f32x2_f16x2_f32x2",
                PackedOp::SubHalf { bf16: true } => "sub_packed_f32x2_bf16x2_f32x2",
                PackedOp::FmaHalf { bf16: false } => "fma_packed_f32x2_f16x2_f32x2_f32x2",
                PackedOp::FmaHalf { bf16: true } => "fma_packed_f32x2_bf16x2_f32x2_f32x2",
            },
        }
    }

    /// The PTX instruction (operands not included).
    pub fn ptx(self) -> String {
        let flag = |on: bool, name: &str| if on { name.to_string() } else { String::new() };
        let round = |up: bool| if up { "rp" } else { "rz" };
        match self {
            CvtFn::F32x2ToBf16x2 => "cvt.rn.satfinite.bf16x2.f32".into(),
            CvtFn::F32ToTf32 => "cvt.rna.tf32.f32".into(),
            CvtFn::S8ToBf16 => "cvt.rn.bf16.s8".into(),
            CvtFn::S2f6x2ToBf16x2 { .. } => "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2".into(),
            CvtFn::F32x2ToF8x2 { to, relu } => format!("cvt.rn.satfinite{}.{}.f32", flag(relu, ".relu"), to.ptx()),
            CvtFn::F16x2ToF8x2 { to, relu } => format!("cvt.rn.satfinite{}.{}.f16x2", flag(relu, ".relu"), to.ptx()),
            CvtFn::F8x2ToF16x2 { from, relu } => format!("cvt.rn{}.f16x2.{}", flag(relu, ".relu"), from.ptx()),
            CvtFn::F32x2ToUe8m0x2 { round_up } => format!("cvt.{}.satfinite.ue8m0x2.f32", round(round_up)),
            CvtFn::F32x2ToUe5m3x2 => "cvt.rp.satfinite.ue5m3x2.f32".into(),
            CvtFn::Bf16x2ToUe8m0x2 { round_up, satfinite } => {
                format!("cvt.{}{}.ue8m0x2.bf16x2", round(round_up), flag(satfinite, ".satfinite"))
            }
            CvtFn::Ue8m0x2ToBf16x2 => "cvt.rn.bf16x2.ue8m0x2".into(),
            CvtFn::F32x2ToMx { to, relu } => format!("cvt.rn.satfinite{}.{}.f32", flag(relu, ".relu"), to.ptx()),
            CvtFn::MxToF16x2 { from, relu } => format!("cvt.rn{}.f16x2.{}", flag(relu, ".relu"), from.ptx()),
            CvtFn::Packed { op, rnd, ftz } => {
                let r = rnd.map(|r| format!(".{}", r.ptx())).unwrap_or_default();
                let half = |bf16: bool| if bf16 { "bf16x2" } else { "f16x2" };
                match op {
                    PackedOp::Add => format!("add{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Sub => format!("sub{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Mul => format!("mul{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::Fma => format!("fma{r}{}.f32x2", flag(ftz, ".ftz")),
                    PackedOp::AddToHalf { bf16 } => format!("add.rn.{}.f32x2", half(bf16)),
                    PackedOp::SubToHalf { bf16 } => format!("sub.rn.{}.f32x2", half(bf16)),
                    PackedOp::AddHalf { bf16 } => format!("add{r}.f32x2.{}", half(bf16)),
                    PackedOp::SubHalf { bf16 } => format!("sub{r}.f32x2.{}", half(bf16)),
                    PackedOp::FmaHalf { bf16 } => format!("fma{r}.f32x2.{}", half(bf16)),
                }
            }
        }
    }

    /// The operands' types.
    pub fn operand_types(self) -> &'static [DType] {
        match self {
            CvtFn::F32x2ToBf16x2
            | CvtFn::F32x2ToF8x2 { .. }
            | CvtFn::F32x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToUe5m3x2
            | CvtFn::F32x2ToMx { .. } => &[DType::F32, DType::F32],
            CvtFn::F32ToTf32 => &[DType::F32],
            CvtFn::S8ToBf16 => &[DType::I8],
            CvtFn::S2f6x2ToBf16x2 { .. }
            | CvtFn::F8x2ToF16x2 { .. }
            | CvtFn::Ue8m0x2ToBf16x2
            | CvtFn::MxToF16x2 { .. } => &[DType::I16],
            CvtFn::F16x2ToF8x2 { .. } | CvtFn::Bf16x2ToUe8m0x2 { .. } => &[DType::I32],
            CvtFn::Packed { op, .. } => match op {
                PackedOp::Add | PackedOp::Sub | PackedOp::Mul => &[DType::U64, DType::U64],
                PackedOp::Fma => &[DType::U64, DType::U64, DType::U64],
                PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. } => &[DType::U64, DType::U64],
                PackedOp::AddHalf { .. } | PackedOp::SubHalf { .. } => &[DType::I32, DType::U64],
                PackedOp::FmaHalf { .. } => &[DType::I32, DType::U64, DType::U64],
            },
        }
    }

    /// The result's type.
    pub fn result_type(self) -> DType {
        match self {
            CvtFn::S8ToBf16 => DType::BF16,
            CvtFn::F32x2ToF8x2 { .. }
            | CvtFn::F16x2ToF8x2 { .. }
            | CvtFn::F32x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToUe5m3x2
            | CvtFn::Bf16x2ToUe8m0x2 { .. }
            | CvtFn::F32x2ToMx { .. } => DType::I16,
            CvtFn::F32x2ToBf16x2
            | CvtFn::F32ToTf32
            | CvtFn::S2f6x2ToBf16x2 { .. }
            | CvtFn::F8x2ToF16x2 { .. }
            | CvtFn::Ue8m0x2ToBf16x2
            | CvtFn::MxToF16x2 { .. } => DType::I32,
            CvtFn::Packed { op, .. } => match op {
                PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. } => DType::I32,
                _ => DType::U64,
            },
        }
    }
}

/// Simulates an inline PTX fragment: raw input bits to raw output bits.
pub type InlineSim = fn(&[u64]) -> Vec<u64>;

/// The arithmetic of an [`Op::CarryInst`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CarryOp {
    /// `a + b` (`add.cc`, `addc`).
    Add,
    /// `a - b` (`sub.cc`, `subc`).
    Sub,
    /// The low half of `a * b`, plus `c` (`mad.lo.cc`, `madc.lo`).
    MadLo,
}

/// One of PTX's extended-precision instructions, which [`Op::CarryInst`]
/// issues: an add, a subtract or a multiply-add that reads the carry flag
/// (`CC.CF`), writes it, or both. A carry-in adds the flag (a borrow-in
/// subtracts it); a carry-out sets it to the carry (borrow) out of the
/// operation. One of the two is always there: without either, the
/// instruction would be plain arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CarryFn {
    pub op: CarryOp,
    /// Reads the flag: `addc`, `subc`, `madc`.
    pub carry_in: bool,
    /// Writes the flag: `.cc`.
    pub carry_out: bool,
}

impl CarryFn {
    /// The name of the operation: CuTe DSL's (`cute.arch.add_cc`, `addc`,
    /// `sub_cc`, `subc`, `mad_cc`, `madc`), or cute-rs's for the forms CuTe
    /// DSL lacks (`addc_cc`, `subc_cc`, `madc_cc`).
    pub fn cute_name(self) -> &'static str {
        match (self.op, self.carry_in, self.carry_out) {
            (CarryOp::Add, false, _) => "add_cc",
            (CarryOp::Add, true, false) => "addc",
            (CarryOp::Add, true, true) => "addc_cc",
            (CarryOp::Sub, false, _) => "sub_cc",
            (CarryOp::Sub, true, false) => "subc",
            (CarryOp::Sub, true, true) => "subc_cc",
            (CarryOp::MadLo, false, _) => "mad_cc",
            (CarryOp::MadLo, true, false) => "madc",
            (CarryOp::MadLo, true, true) => "madc_cc",
        }
    }

    /// Number of operands.
    pub fn arity(self) -> usize {
        match self.op {
            CarryOp::Add | CarryOp::Sub => 2,
            CarryOp::MadLo => 3,
        }
    }
}

/// The PTX instruction an [`Op::CarryInst`] issues on operands of type `ty`,
/// as CuTe DSL spells it: `add.cc.u32`, `addc.cc.u64`, `subc.u32`,
/// `mad.lo.cc.s32`, `madc.lo.u64`. The adds and subtracts are `.u` for
/// either signedness (the PTX ISA: "behavior is the same for unsigned and
/// signed integers"); the multiply-adds follow the type, which only names
/// the instruction, since the low half of a product and the carry out of
/// the addition do not depend on it.
pub fn carry_inst_name(func: CarryFn, ty: DType) -> String {
    let w = ty.bits();
    let c = if func.carry_in { "c" } else { "" };
    let cc = if func.carry_out { ".cc" } else { "" };
    match func.op {
        CarryOp::Add => format!("add{c}{cc}.u{w}"),
        CarryOp::Sub => format!("sub{c}{cc}.u{w}"),
        CarryOp::MadLo => format!("mad{c}.lo{cc}.{}{w}", if ty.is_signed() { 's' } else { 'u' }),
    }
}

/// Why `op` loses the carry flag of a chain around it, if it does. PTX
/// keeps the flag only through straight-line code ("not preserved across
/// calls ... mainly intended for use in straight-line code sequences"), so
/// what ends that code in the PTX loses it: a runtime `if` or loop, a call
/// (`printf`), a hand-written MLIR operation (which may branch or call),
/// inline PTX that branches, calls or exits, and a 64-bit integer division
/// or remainder by a value that is not a constant, which LLVM (from O1 on)
/// lowers with a branch to a 32-bit division for when both operands fit
/// (its `BypassSlowDivision`; a constant divisor becomes a multiplication).
/// Inline PTX that uses the flag itself changes it behind the chain's back.
/// The validator rejects a carry-in after any of these, and the simulator
/// forgets the flag there. The warp collectives keep it: each is one
/// convergent instruction (`shfl.sync`, also with a runtime segment mask
/// and clamp, `vote.sync`, `match.{any,all}.sync`, `redux.sync`,
/// `bar.warp.sync` with any member mask) that neither branches nor touches
/// the flag. So do the special-register reads, at most one `mov` each: the
/// thread, block and lane indices and the block and grid sizes through the
/// GPU dialect, `%globaltimer_lo` as inline PTX with side effects, the
/// shared-memory size registers as pure inline PTX (after which
/// `dynamic_smem_size` subtracts the region's start, when it is not 0, with
/// ordinary arithmetic), and the rest through NVVM. So does
/// `cp.async.mbarrier.arrive` (one instruction, from an LLVM intrinsic).
/// `dtype` gives a value's type and `is_constant` whether a constant defines
/// it.
pub fn carry_flag_breaker(
    op: &Op,
    dtype: impl Fn(ValueId) -> DType,
    is_constant: impl Fn(ValueId) -> bool,
) -> Option<&'static str> {
    match op {
        Op::If { .. } => Some("runtime if"),
        Op::For { .. } | Op::While { .. } => Some("runtime loop"),
        Op::Printf { .. } => Some("printf (a call)"),
        Op::Call { .. } => Some("a call of an external device function"),
        Op::Assert { .. } => Some("device assertion (a call)"),
        Op::MlirOp { .. } => Some("hand-written MLIR operation (which may branch or call)"),
        Op::InlinePtx { template, .. } => inline_ptx_carry_breaker(template),
        // NVPTX lowers 128-bit arithmetic with the extended-precision
        // instructions (and its division with branches).
        Op::Binary { lhs: x, .. } | Op::Unary { x, .. } | Op::Cmp { lhs: x, .. } if dtype(*x).bits() == 128 => {
            Some("128-bit integer arithmetic (which LLVM lowers with add.cc and addc)")
        }
        Op::Binary { op: BinaryOp::Div | BinaryOp::DivU | BinaryOp::Rem | BinaryOp::RemU, rhs, .. }
            if dtype(*rhs).is_int() && dtype(*rhs).bits() == 64 && !is_constant(*rhs) =>
        {
            Some("64-bit integer division or remainder by a runtime value (which LLVM lowers with a branch)")
        }
        Op::Shuffle { .. } | Op::Vote { .. } | Op::Match { .. } | Op::Redux { .. } | Op::WarpBarrier { .. } => None,
        Op::Special { .. } | Op::CpAsyncMbarrierArrive { .. } => None,
        // Memory accesses, ordered or hinted, prefetches, packing and the
        // half-precision and conversion instructions: not covered by the
        // documented guarantee above, so they are listed here, not claimed.
        Op::Load { .. } | Op::Store { .. } | Op::Prefetch { .. } => None,
        Op::Pack { .. } | Op::Unpack { .. } | Op::HalfInst { .. } | Op::Cvt { .. } => None,
        Op::Assume { .. } => None,
        Op::Red { .. } => None,
        _ => None,
    }
}

/// Why hand-written PTX loses the carry flag of a chain around it, if it
/// does: it uses one of the extended-precision instructions (`add.cc`,
/// `addc`, `sub.cc`, `subc`, `mad{.lo,.hi}.cc`, `madc`), the only
/// instructions that touch the flag, or it branches, calls, returns or exits
/// (`bra`, `brx.idx`, `call`, `ret`, `exit`), which ends the straight-line
/// code that PTX keeps the flag through.
fn inline_ptx_carry_breaker(template: &str) -> Option<&'static str> {
    let tokens = || template.split(|c: char| c.is_whitespace() || matches!(c, ';' | '{' | '}' | '"'));
    let opcode = |token: &str| token.split('.').next().unwrap_or_default().to_string();
    let uses_flag = tokens().any(|token| {
        matches!(opcode(token).as_str(), "addc" | "subc" | "madc")
            || (matches!(opcode(token).as_str(), "add" | "sub" | "mad") && token.split('.').any(|part| part == "cc"))
    });
    if uses_flag {
        return Some("inline PTX that uses the carry flag");
    }
    let jumps = tokens().any(|token| matches!(opcode(token).as_str(), "bra" | "brx" | "call" | "ret" | "exit"));
    jumps.then_some("inline PTX that branches, calls or exits")
}

/// An input of an [`Op::InlinePtx`] statement or an operand of an
/// [`Op::MlirOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsmInput {
    /// A value, in a register.
    Value(ValueId),
    /// The address of element `index` (counting `elem`s) of a buffer, as a
    /// 64-bit integer: an address in the global state space for global
    /// memory (where it is also the generic address), in the shared state
    /// space for shared memory, and the generic address of a local
    /// (per-thread) buffer.
    Address { buffer: BufferId, index: ValueId, elem: DType },
}

impl AsmInput {
    /// The SSA value the input reads: the value itself, or an address's
    /// index.
    pub fn value(&self) -> ValueId {
        match *self {
            AsmInput::Value(v) | AsmInput::Address { index: v, .. } => v,
        }
    }
}

/// Simulates an inline assembly statement or a hand-written MLIR operation
/// in one thread: given the raw bits of its read-only operands (0 for an
/// address, which the simulator cannot follow) and, mutably, of its
/// read-write operands, it updates the read-write operands and returns the
/// write-only results, as raw bits (zero-extended to 64; extra high bits are
/// ignored).
#[derive(Clone)]
pub struct AsmSim(pub std::sync::Arc<AsmSimFn>);

/// The function an [`AsmSim`] holds.
pub type AsmSimFn = dyn Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync;

impl AsmSim {
    pub fn new(f: impl Fn(&[u64], &mut [u64]) -> Vec<u64> + Send + Sync + 'static) -> AsmSim {
        AsmSim(std::sync::Arc::new(f))
    }

    /// The simulation of the positional inline PTX functions: inputs to
    /// outputs.
    pub fn from_inline(f: InlineSim) -> AsmSim {
        AsmSim::new(move |reads, _| f(reads))
    }

    /// Run it.
    pub fn call(&self, reads: &[u64], read_writes: &mut [u64]) -> Vec<u64> {
        (self.0)(reads, read_writes)
    }
}

impl fmt::Debug for AsmSim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AsmSim(..)")
    }
}

/// One entry of an LLVM inline-assembly constraint string.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AsmConstraint {
    /// An output, with the code after its `=` (`r`, `&l`, ...).
    Output(String),
    /// An input in a register or an immediate (`r`, `l`, `n`, ...).
    Input(String),
    /// An input tied to (in the same register as) the output with this
    /// number.
    Tied(usize),
    /// A clobber, with the name in its braces (`memory`).
    Clobber(String),
    /// A read-write operand (`+r`, the code after `+`): GCC's spelling,
    /// which LLVM IR lacks, as `InlineAsm` takes it.
    ReadWrite(String),
}

/// The entries of an LLVM constraint string, split at its commas and
/// trimmed of spaces (none for an empty string).
pub fn asm_constraints(constraints: &str) -> Vec<AsmConstraint> {
    if constraints.trim().is_empty() {
        return Vec::new();
    }
    constraints
        .split(',')
        .map(|e| {
            let e = e.trim();
            if let Some(code) = e.strip_prefix('=') {
                AsmConstraint::Output(code.to_string())
            } else if let Some(code) = e.strip_prefix('+') {
                AsmConstraint::ReadWrite(code.to_string())
            } else if let Some(name) = e.strip_prefix('~') {
                AsmConstraint::Clobber(name.trim_start_matches('{').trim_end_matches('}').to_string())
            } else if !e.is_empty() && e.bytes().all(|b| b.is_ascii_digit()) {
                AsmConstraint::Tied(e.parse().unwrap_or(usize::MAX))
            } else {
                AsmConstraint::Input(e.to_string())
            }
        })
        .collect()
}

/// The NVPTX inline-assembly register class of a type: its constraint code
/// (`b` predicate, `c` 8-bit, `h` 16-bit, `r` 32-bit and `l` 64-bit
/// integers, `f` and `d` floats; Float16 and BFloat16 in `h`).
pub fn asm_register(ty: DType) -> &'static str {
    match ty {
        DType::Bool => "b",
        DType::I8 | DType::U8 | DType::I4 => "c",
        t if t.is_narrow_float() || t == DType::F4E2M1FNx2 => "c",
        DType::I16 | DType::U16 | DType::F16 | DType::BF16 => "h",
        DType::I32 | DType::U32 | DType::F6E2M3FNx4 | DType::F6E3M2FNx4 => "r",
        // 128-bit integers have no register class: the validator refuses
        // them as inline-assembly operands.
        DType::I64 | DType::U64 | DType::I128 | DType::U128 => "l",
        DType::F32 | DType::TF32 => "f",
        _ => "d",
    }
}

/// Whether a constraint string clobbers memory (`~{memory}`).
pub fn asm_clobbers_memory(constraints: &str) -> bool {
    asm_constraints(constraints).contains(&AsmConstraint::Clobber("memory".into()))
}

/// The position of an empty entry of a constraint string (`"=r,,r"`, a
/// trailing comma, or `=`, `+` or `~{}` alone), which LLVM rejects.
pub fn asm_empty_constraint(constraints: &str) -> Option<usize> {
    asm_constraints(constraints).iter().position(|e| match e {
        AsmConstraint::Output(c)
        | AsmConstraint::Input(c)
        | AsmConstraint::Clobber(c)
        | AsmConstraint::ReadWrite(c) => c.is_empty(),
        AsmConstraint::Tied(_) => false,
    })
}

/// The operand numbers an inline-assembly template refers to (`$3`, `${3}`,
/// `${3:modifier}`), in order, as LLVM reads them in the AT&T dialect: `$$`
/// is a literal `$`, and `$(`, `$|`, `$)` and `${:...}` are not operands.
/// Any other `$` is an error, which LLVM would only report by aborting.
pub fn asm_operand_refs(template: &str) -> Result<Vec<usize>, String> {
    let bytes = template.as_bytes();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            i += 1;
            continue;
        }
        i += 1;
        match bytes.get(i) {
            Some(b'$' | b'(' | b'|' | b')') => i += 1,
            Some(b'{') if bytes.get(i + 1) == Some(&b':') => {
                i = template[i..].find('}').map_or(bytes.len(), |end| i + end + 1);
            }
            Some(&c) if c == b'{' || c.is_ascii_digit() => {
                let braced = c == b'{';
                let start = i + braced as usize;
                let end = start + bytes[start..].iter().take_while(|b| b.is_ascii_digit()).count();
                let number = template[start..end].parse().map_err(|_| {
                    format!("`$` must be followed by an operand number (or written `$$`) in {template:?}")
                })?;
                refs.push(number);
                i = end;
                if braced {
                    match template[end..].find('}') {
                        Some(close) => i = end + close + 1,
                        None => return Err(format!("unclosed `${{` in {template:?}")),
                    }
                }
            }
            _ => return Err(format!("`$` must be followed by an operand number (or written `$$`) in {template:?}")),
        }
    }
    Ok(refs)
}

/// Hints for a runtime loop: the keyword arguments of CuTe DSL's
/// `cutlass.range(start, stop, step, unroll=, unroll_full=, vectorize=,
/// at_least_once=)`. The default attaches nothing, as in CuTe DSL.
///
/// The unroll and vectorize hints become LLVM loop metadata
/// (`#llvm.loop_annotation`), which NVPTX honours; they never change what the
/// loop computes, and the simulator ignores them. Fields not given keep
/// their defaults: `LoopOptions { vectorize: true, ..LoopOptions::default()
/// }`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LoopOptions {
    /// `unroll=K`: unroll the loop `K` times (`count = K`). 1 keeps it rolled
    /// (`disable = true`, PTX `.pragma "nounroll"`); 0 leaves the factor to
    /// LLVM. A runtime trip count gets a remainder loop.
    pub unroll: Option<u32>,
    /// `unroll_full=True`: unroll the loop completely (`full = true`). LLVM
    /// only does so when it can compute the trip count. Cannot be combined
    /// with `unroll` (CuTe DSL lets `unroll_full` win; this is an error).
    pub unroll_full: bool,
    /// `vectorize=True`: ask LLVM to vectorise the loop (`vectorize =
    /// <disable = false>`). Needs sm_100 or newer, as in CuTe DSL.
    pub vectorize: bool,
    /// `at_least_once=True`: the program promises that the loop runs at
    /// least once in every thread, so the compiler can drop the check
    /// whether to skip it (the MLIR assumes `lower < upper` with
    /// `llvm.intr.assume`). The simulator reports a thread for which it
    /// runs zero times, and the validator constant bounds that do.
    pub at_least_once: bool,
}

impl LoopOptions {
    /// `unroll=count`.
    pub fn unroll(count: u32) -> LoopOptions {
        LoopOptions { unroll: Some(count), ..LoopOptions::default() }
    }

    /// `unroll_full=True`.
    pub fn unroll_full() -> LoopOptions {
        LoopOptions { unroll_full: true, ..LoopOptions::default() }
    }

    /// Whether any hint becomes loop metadata.
    pub fn has_annotation(&self) -> bool {
        self.unroll.is_some() || self.unroll_full || self.vectorize
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Const {
        result: ValueId,
        value: Literal,
    },
    Unary {
        result: ValueId,
        op: UnaryOp,
        x: ValueId,
    },
    Binary {
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// `a * b + c` with a single rounding.
    Fma {
        result: ValueId,
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },
    /// A floating-point instruction with explicit PTX modifiers: a rounding
    /// direction, the hardware approximation, flushing subnormal inputs and
    /// results to zero (`ftz`), NaN propagation (`nan`, for `min` and
    /// `max`) or `.xorsign.abs` (`xorsign_abs`, for Float32 `min` and
    /// `max`: the magnitude of the operands' absolute values, with the
    /// exclusive or of their signs). These are `cute.math`'s `approx=`,
    /// `ftz=`, `rounding=`, `full=` and `propagate_nan=` options and its
    /// `fastmath=True` approximations, and `cute.arch.fmax`'s and `fmin`'s
    /// `abs=`, `nan=` and `ftz=`.
    FloatInst {
        result: ValueId,
        func: FloatFn,
        args: Vec<ValueId>,
        mode: FloatMode,
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
    },
    /// A PTX integer or bit instruction ([`IntFn`]), with its immediates and
    /// modifiers: CuTe DSL's `cute.arch` integer helpers (`bfe`, `lop3`,
    /// `shf`, `prmt`, `mul_wide`, `mul24`, `add_sat_int`, ...). Pure.
    IntInst {
        result: ValueId,
        func: IntFn,
        args: Vec<ValueId>,
    },
    Cmp {
        result: ValueId,
        pred: CmpPred,
        lhs: ValueId,
        rhs: ValueId,
    },
    Select {
        result: ValueId,
        cond: ValueId,
        on_true: ValueId,
        on_false: ValueId,
    },
    /// Numeric conversion to the result's type.
    Cast {
        result: ValueId,
        x: ValueId,
    },
    /// Reinterpret the bits as the result's type (same width).
    Bitcast {
        result: ValueId,
        x: ValueId,
    },
    Special {
        result: ValueId,
        reg: SpecialReg,
    },
    /// Load `results.len()` contiguous elements of the results' type. `index`
    /// counts elements of that type from the start of the buffer. `attrs`
    /// give the ordering and cache hints (plain for tensor loads). With a
    /// `pred` (Boolean), a thread whose flag is false reads nothing: its
    /// results are `pass_thru` (one value per result), or undefined when
    /// `pass_thru` is empty.
    Load {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        attrs: MemAttrs,
        pred: Option<ValueId>,
        pass_thru: Vec<ValueId>,
    },
    /// Store `values.len()` contiguous elements of the values' type. `index`
    /// counts elements of that type from the start of the buffer. `attrs`
    /// give the ordering and cache hints (plain for tensor stores). With a
    /// `pred` (Boolean), a thread whose flag is false writes nothing.
    Store {
        values: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        attrs: MemAttrs,
        pred: Option<ValueId>,
    },
    /// `cp.async` of `elems` contiguous `elem`s from global to shared memory.
    /// Indices count `elem`s. When `pred` is false the destination is
    /// zero-filled instead.
    CpAsync {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        elems: u32,
        bypass_l1: bool,
        pred: Option<ValueId>,
    },
    CpAsyncCommitGroup,
    /// Wait until at most `pending` committed groups are still in flight.
    CpAsyncWaitGroup {
        pending: u32,
    },
    /// `ldmatrix.sync.aligned.{shape}.x{num}[.trans].shared.{format}`, the
    /// shape and element format given by `form`. Each lane passes the
    /// address of one 16-byte row. For [`LdMatrixForm::M8n8B16`] `index`
    /// counts 16-bit elements and `results` holds two 16-bit values per
    /// matrix, register-major; for the 8-bit forms `index` counts bytes and
    /// `results` holds the 32-bit destination registers (each four 8-bit
    /// values, the first in the low byte).
    LdMatrix {
        results: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        num: u32,
        trans: bool,
        form: LdMatrixForm,
    },
    /// `stmatrix.sync.aligned.{shape}.x{num}[.trans].shared.{type}`: each
    /// lane passes the address of one 16-byte row of shared memory (lanes
    /// `8m` to `8m + 7` those of matrix `m`) and one 32-bit register per
    /// matrix, `values[m]`: for [`StMatrixForm::M8n8B16`] two 16-bit elements
    /// (the first in the low half) and `index` counts 16-bit elements; for
    /// [`StMatrixForm::M16n8B8`] four bytes and `index` counts bytes.
    StMatrix {
        values: Vec<ValueId>,
        buffer: BufferId,
        index: ValueId,
        num: u32,
        trans: bool,
        form: StMatrixForm,
    },
    /// `setmaxnreg.{inc,dec}.sync.aligned.u32 count`: the warpgroup's
    /// per-thread register budget becomes `count` (24 to 256, a multiple of
    /// 8). All four warps of the warpgroup execute it together.
    SetMaxNReg {
        increase: bool,
        count: u32,
    },
    /// `mma.sync`, or with a sparse kind `mma.sp::ordered_metadata.sync`,
    /// whose `e` is the 32-bit metadata register (sparsity selector 0).
    Mma {
        kind: MmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        e: Option<ValueId>,
    },
    /// `bar.sync 0` (`__syncthreads`).
    Barrier,
    /// A named barrier: barrier `id` (0 to 15) completes when `threads`
    /// threads (a multiple of 32; all of the block when `None`) have arrived.
    NamedBarrier {
        kind: BarrierKind,
        id: ValueId,
        threads: Option<ValueId>,
    },
    /// `bar.warp.sync` (`__syncwarp`) over the lanes in `mask`.
    WarpBarrier {
        mask: ValueId,
    },
    /// `shfl.sync.{kind}.b32 result, value, offset, clamp, mask`: read the
    /// 32-bit `value` of another lane of the warp. `offset` is the source
    /// lane or lane offset (its bits 4:0), and `clamp` packs the segment mask
    /// (bits 12:8) and the clamp value (bits 4:0), CuTe DSL's
    /// `mask_and_clamp`.
    Shuffle {
        result: ValueId,
        kind: ShuffleKind,
        value: ValueId,
        offset: ValueId,
        mask: ValueId,
        clamp: ValueId,
    },
    /// `vote.sync` over the lanes in `mask`.
    Vote {
        result: ValueId,
        kind: VoteKind,
        pred: ValueId,
        mask: ValueId,
    },
    /// `match.{any,all}.sync.b{32,64}`: compare a 32- or 64-bit `value`
    /// across the lanes in `mask`, giving a mask of lanes.
    Match {
        result: ValueId,
        kind: MatchKind,
        value: ValueId,
        mask: ValueId,
    },
    /// `elect.sync`: true in exactly one of the lanes in `mask`.
    ElectOne {
        result: ValueId,
        mask: ValueId,
    },
    /// `activemask.b32`.
    ActiveMask {
        result: ValueId,
    },
    /// `value`, which the program promises is the same in every lane of the
    /// warp (CuTe DSL's `make_warp_uniform`): a compiler hint, not an
    /// instruction.
    WarpUniform {
        result: ValueId,
        value: ValueId,
    },
    /// Atomic read-modify-write (`atom{.sem}{.scope}{.space}.op.type`);
    /// `result` is the old value. `index` counts elements of the operand's
    /// type, or of the half-precision element of a packed type (whose
    /// operand is an Int32 holding two). `sem` is relaxed, acquire,
    /// release or acq_rel.
    Atomic {
        result: ValueId,
        kind: AtomicKind,
        ty: AtomicType,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        compare: Option<ValueId>,
        sem: MemoryOrder,
        scope: MemoryScope,
        /// Distributed shared memory: the CTA rank, as for
        /// [`MemAttrs::peer`] (roadmap part m8-dsmem).
        peer: Option<ValueId>,
    },
    /// A reduction (`red{.sem}{.scope}{.space}.op.type`): an atomic
    /// read-modify-write without a result, addressed as [`Op::Atomic`] is.
    /// `sem` is relaxed or release.
    Red {
        kind: AtomicKind,
        ty: AtomicType,
        buffer: BufferId,
        index: ValueId,
        value: ValueId,
        sem: MemoryOrder,
        scope: MemoryScope,
        /// As for [`Op::Atomic`].
        peer: Option<ValueId>,
    },
    /// `mbarrier.init`: an mbarrier waiting for `count` arrivals per phase.
    MbarrierInit {
        buffer: BufferId,
        index: ValueId,
        count: ValueId,
    },
    /// `mbarrier.arrive`: `count` arrivals (1 when `None`), after raising the
    /// bytes of bulk copies the phase waits for by `expect_tx` when given
    /// (`.expect_tx`). With `peer`, on the mbarrier at the same offset in
    /// that CTA of the cluster (`mapa` + `.shared::cluster`). `relaxed` drops
    /// the release ordering.
    MbarrierArrive {
        buffer: BufferId,
        index: ValueId,
        count: Option<ValueId>,
        expect_tx: Option<ValueId>,
        peer: Option<ValueId>,
        relaxed: bool,
        scope: MemScopeKind,
    },
    /// `mbarrier.expect_tx`: make the current phase wait for `bytes` more
    /// bytes of bulk copies, without arriving.
    MbarrierExpectTx {
        buffer: BufferId,
        index: ValueId,
        bytes: ValueId,
        peer: Option<ValueId>,
        scope: MemScopeKind,
    },
    /// `mbarrier.try_wait.parity` / `mbarrier.test_wait.parity`: whether the
    /// phase with this parity is over.
    MbarrierWait {
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        parity: ValueId,
        kind: MbarrierWaitKind,
        /// m11-prims-nvvm-layer: `parity` is an i64 state token from an
        /// arrival (`mbarrier.{try,test}_wait`, no `.parity`).
        state: bool,
        /// m11-prims-nvvm-layer: the `suspendTimeHint` (i32, nanoseconds) of
        /// a `try_wait`.
        time_limit: Option<ValueId>,
    },
    /// `fence.mbarrier_init.release.cluster`: make earlier `mbarrier.init`s
    /// visible to the cluster.
    FenceMbarrierInit,
    /// `griddepcontrol.wait` (`launch_dependents` false): block until every prerequisite
    /// grid of a programmatic dependent launch has completed and its memory is visible; or
    /// `griddepcontrol.launch_dependents`: let the dependent grid be scheduled once every CTA
    /// has issued it or exited. Both are no-ops without programmatic stream serialization.
    GridDepControl {
        launch_dependents: bool,
    },
    /// `barrier.cluster.arrive{.relaxed}{.aligned}`: the thread arrives at
    /// its cluster's barrier, with release semantics at cluster scope unless
    /// `relaxed`. `aligned`: every thread of the warp executes it together.
    ClusterArrive {
        relaxed: bool,
        aligned: bool,
    },
    /// `barrier.cluster.wait{.aligned}`: wait until every thread of the
    /// cluster that has not exited has arrived in the thread's phase
    /// (acquire semantics at cluster scope).
    ClusterWait {
        aligned: bool,
    },
    /// `cp.async.bulk` of `bytes` bytes (a multiple of 16, both addresses
    /// 16-byte aligned) in `direction`, reporting completion as
    /// `completion` says. Indices count `elem`s. `cache_policy` is the
    /// 64-bit operand of `.L2::cache_hint` (G2S and S2G); `mcast_mask` (an
    /// i16) makes a G2S copy `.multicast::cluster`, landing at the same
    /// offset in each CTA of the mask and signalling the mbarrier at the
    /// same offset in each; `peer` (an i32 CTA rank, S2S only) is the CTA
    /// whose shared memory and mbarrier the copy addresses (`mapa`).
    CpAsyncBulk {
        src: BufferId,
        src_index: ValueId,
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        bytes: ValueId,
        direction: BulkDirection,
        completion: BulkCompletion,
        cache_policy: Option<ValueId>,
        mcast_mask: Option<ValueId>,
        peer: Option<ValueId>,
        /// `.cp_mask` with this i16 mask (S2G only; roadmap part
        /// m7-tma-sm100-modes, sm_100): bit `i` enables byte `i` of each
        /// 16-byte chunk.
        byte_mask: Option<ValueId>,
    },
    /// `cp.async.bulk.commit_group`: the thread's bulk copies issued since
    /// its last commit become a bulk async-group.
    CpAsyncBulkCommitGroup,
    /// `cp.async.bulk.wait_group{.read} pending`: wait until at most
    /// `pending` of the thread's most recent bulk async-groups are pending;
    /// with `read`, only until the others have read their sources.
    CpAsyncBulkWaitGroup {
        pending: u32,
        read: bool,
    },
    /// A TMA tile load (roadmap part m7-tma-load):
    /// `cp.async.bulk.tensor.{N}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes[.L2::cache_hint]`
    /// of the box at `coords` (one i32 per dimension of the descriptor, in its
    /// order) of the tensor map at the generic address `desc` (an i64) into
    /// shared memory at element `dst_index` of `dst` (its unswizzled start;
    /// the descriptor's swizzle places the bytes), its bytes counting down
    /// the transaction count of the mbarrier at element `mbar_index` of
    /// `mbar`. `elem` is the element type (the index counts it) and
    /// `box_bytes` the box's bytes the atom was built for.
    TmaLoad {
        dst: BufferId,
        dst_index: ValueId,
        elem: DType,
        desc: ValueId,
        coords: Vec<ValueId>,
        mbar: BufferId,
        mbar_index: ValueId,
        cache_policy: Option<ValueId>,
        box_bytes: u32,
        /// `.multicast::cluster` with this i16 CTA mask (roadmap part
        /// m7-cluster-copies): the box lands at the same offset in every CTA
        /// of the mask, and counts down the mbarrier at the same offset in
        /// each.
        mcast_mask: Option<ValueId>,
        /// The access mode (roadmap parts m7-tma-im2col and
        /// m7-tma-sm100-modes): `.tile` unless an im2col or gather4 atom.
        mode: TmaMode,
        /// `.override::*` operands (roadmap part m7-tma-sm100-modes).
        overrides: Option<TmaOverride>,
        /// `.cta_group::2` (roadmap part m10-cta-pair, a load for the CTA
        /// pair of a 2-CTA tcgen05 MMA): the box lands in the issuing CTA
        /// (or the CTAs of the mask), and its bytes count down the mbarrier
        /// at the same offset in the even CTA of each destination's pair
        /// (the address with its peer bit cleared, CUTLASS's
        /// `Sm100MmaPeerBitMask`).
        two_cta: bool,
    },
    // ---- m7-tma-store and m7-cluster-copies ----
    /// A TMA tile store (roadmap part m7-tma-store):
    /// `cp.async.bulk.tensor.{N}d.global.shared::cta.tile.bulk_group[.L2::cache_hint]`
    /// of the box at element `src_index` of `src` (its unswizzled start;
    /// the descriptor's swizzle says where each byte is read) to the box at
    /// `coords` of the tensor map at `desc` (out-of-bounds elements are not
    /// written), tracked by the thread's bulk async-groups. With `reduce`,
    /// `cp.reduce.async.bulk.tensor` combining the box into global memory
    /// with that operation.
    TmaStore {
        src: BufferId,
        src_index: ValueId,
        elem: DType,
        desc: ValueId,
        coords: Vec<ValueId>,
        reduce: Option<crate::cpasync::ReductionKind>,
        cache_policy: Option<ValueId>,
        box_bytes: u32,
        /// The access mode: `.tile`, `.im2col_no_offs` or `.tile::scatter4`.
        mode: TmaMode,
        /// `.override::*` operands (roadmap part m7-tma-sm100-modes).
        overrides: Option<TmaOverride>,
    },
    /// `cp.async.bulk.prefetch.tensor.{N}d.L2.global{mode} [desc, coords]`:
    /// bring the box into L2. A hint. `mode` is `.tile`, `.im2col` or
    /// `.tile::gather4`.
    TmaPrefetch {
        desc: ValueId,
        coords: Vec<ValueId>,
        mode: TmaMode,
    },
    /// `prefetch.tensormap [desc]`: bring the tensor map at the generic
    /// address `desc` into the cache. A hint.
    PrefetchTensormap {
        desc: ValueId,
    },
    // In-kernel tensor-map updates (roadmap part m7-tensormap-update).
    /// `tensormap.replace.tile.{field}{.ss}.b1024.{b32|b64} [addr], {ord,}
    /// value`: replace one field of the 128-byte tensor map at `addr` (an
    /// `i64` global or generic address, or an `i32` shared one, as `space`
    /// says). A generic-proxy write of the whole object.
    TensormapReplace {
        addr: ValueId,
        space: crate::tensor::AddressSpace,
        field: crate::nvgpu::tensormap::TensormapField,
        value: ValueId,
    },
    /// `fence.proxy.tensormap::generic.release.gpu`: the thread's earlier
    /// generic writes of tensor maps become visible to the tensormap proxy
    /// of threads that acquire them.
    FenceTensormapRelease,
    /// `fence.proxy.tensormap::generic.acquire.gpu [addr], 128`: the
    /// thread's later tensormap-proxy reads of the tensor map at the
    /// generic address `addr` see the released writes.
    FenceTensormapAcquire {
        addr: ValueId,
    },
    /// `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
    /// [dst], [src], 128`: copy the tensor map at the shared address `src`
    /// (`i32`) to the global address `dst` (`i64`) and release it to the
    /// tensormap proxy. Warp-collective.
    TensormapCpFenceproxy {
        dst: ValueId,
        src: ValueId,
    },
    /// `cp.async.bulk.prefetch.L2.global [a], bytes`: bring `bytes` bytes
    /// (a multiple of 16) from element `index` of a global buffer into L2.
    BulkPrefetch {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        bytes: ValueId,
    },
    /// `st.async.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.b32`:
    /// store `values` (one, two or four i32s) at element `index` of `buffer`
    /// in CTA `peer` of the cluster, counting their bytes down the mbarrier
    /// at element `mbar_index` of `mbar` in that CTA (both addresses mapped
    /// with `mapa`).
    StAsync {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: Vec<ValueId>,
        mbar: BufferId,
        mbar_index: ValueId,
        peer: ValueId,
    },
    /// `fence.proxy.{kind}`: orders this thread's accesses through one proxy
    /// against its later accesses through another.
    FenceProxy {
        kind: ProxyKind,
    },
    /// `fence.acq_rel.{scope}`, or with `sc` `fence.sc.{scope}` (`membar`;
    /// m11-prims-nvvm-layer).
    Fence {
        scope: FenceScope,
        sc: bool,
    },
    /// `vprintf`.
    Printf {
        format: String,
        args: Vec<ValueId>,
    },
    /// A device assertion (`cute::testing::assert_`, CuTe DSL's
    /// `cute.testing.assert_`): `cf.assert` when assertions are enabled
    /// (`__assertfail` with the message and the op's file and line, then
    /// the thread exits), nothing otherwise.
    Assert {
        cond: ValueId,
        message: String,
    },
    /// Hand-written PTX, as LLVM inline assembly. `template` names the
    /// operands `$0, $1, ...`: the outputs, then the inputs (`$$` is a
    /// literal `$`). `constraints` is LLVM's constraint string: an entry per
    /// output (`=r`), then per input (`r`, or an output's number for an input
    /// tied to that output, which makes the output a read-write operand),
    /// then clobbers (`~{memory}`). With `predicated`, the last input is a
    /// Boolean guarding the template, which is one instruction (`@$n ...`),
    /// and where it is false the outputs tied to an input keep that input's
    /// value and the others are 0. The simulator can only run it if `sim` is
    /// given.
    InlinePtx {
        template: String,
        inputs: Vec<AsmInput>,
        outputs: Vec<ValueId>,
        constraints: String,
        predicated: bool,
        side_effects: bool,
        sim: Option<AsmSim>,
    },
    If {
        cond: ValueId,
        then_region: Region,
        else_region: Region,
        results: Vec<ValueId>,
    },
    /// `for iv in (lower..upper).step_by(step)` with loop-carried values:
    /// `args` are the body's views of the carried values, initialised from
    /// `init` and updated from the body's yields; `results` are the final
    /// values. `opts` holds the unroll and vectorize hints.
    For {
        iv: ValueId,
        lower: ValueId,
        upper: ValueId,
        step: ValueId,
        init: Vec<ValueId>,
        args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
        opts: LoopOptions,
    },
    /// `while cond(args) { args = body(args) }`. `cond` sees `cond_args` and
    /// yields one boolean; `body` sees `body_args` and yields the next state.
    /// `results` are the state when the condition first fails.
    While {
        init: Vec<ValueId>,
        cond_args: Vec<ValueId>,
        cond: Region,
        body_args: Vec<ValueId>,
        body: Region,
        results: Vec<ValueId>,
    },
    /// `redux.sync`: reduce `value` (Int32, Uint32 or Float32) across the
    /// lanes in `mask`; every member lane gets the result. `abs` and `nan`
    /// are the Float32 modifiers `.abs` and `.NaN`.
    Redux {
        result: ValueId,
        kind: ReduxKind,
        value: ValueId,
        mask: ValueId,
        abs: bool,
        nan: bool,
    },
    /// An extended-precision instruction ([`CarryFn`]): CuTe DSL's
    /// `add_cc`, `addc`, `sub_cc`, `subc`, `mad_cc` and `madc`, and the
    /// carry-in-and-out forms. It reads or writes the carry flag, state that
    /// no SSA value shows, so it is never removed, merged with another or
    /// moved past one: the builder neither CSEs nor deletes it, and the
    /// backend emits it as inline PTX with side effects. A carry-in needs a
    /// carry-out before it in the same region with no control flow between
    /// them (the validator checks).
    CarryInst {
        result: ValueId,
        func: CarryFn,
        args: Vec<ValueId>,
    },
    /// An MLIR operation written by hand (CuTe DSL's raw `cutlass._mlir`
    /// builders inside a `dsl_user_op`), for operations cute-rs lacks:
    /// `text` is the operation without its results, with `%0, %1, ...`
    /// standing for `operands`, and `results` get its results. An address
    /// operand is an LLVM pointer: `!llvm.ptr<1>` into global memory,
    /// `!llvm.ptr<3>` into shared memory and a generic `!llvm.ptr` into a
    /// local buffer. The simulator can only run it if `sim` is given.
    /// Without side effects, it is removed when its results are unused.
    MlirOp {
        text: String,
        operands: Vec<AsmInput>,
        results: Vec<ValueId>,
        side_effects: bool,
        sim: Option<AsmSim>,
    },
    /// `cp.async.mbarrier.arrive{.noinc}.shared.b64`: an arrive-on the
    /// mbarrier once every `cp.async` this thread issued before it has
    /// completed. Without `noinc` the phase's pending count is raised by one
    /// first, so the asynchronous arrival changes nothing on balance; with
    /// it, the mbarrier's count must include the arrival.
    CpAsyncMbarrierArrive {
        buffer: BufferId,
        index: ValueId,
        noinc: bool,
    },
    /// `prefetch{.global}.level [a]`, `prefetch.global.L2::evict_*` (with
    /// `evict`) or `prefetchu.L1` (`uniform`): bring the cache line holding
    /// element `index` (counting `elem`s) of a global or local buffer into a
    /// cache, when `pred` (if given) holds. A hint: nothing is read.
    Prefetch {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        level: PrefetchCacheLevel,
        evict: Option<CacheEvictionPriority>,
        uniform: bool,
        pred: Option<ValueId>,
    },
    /// The bits of `parts` (two or more values of one type) concatenated
    /// into one value of the result's type, part 0 in the low bits: a
    /// bitcast that changes the number of values. Pure.
    Pack {
        result: ValueId,
        parts: Vec<ValueId>,
    },
    /// The bits of `x` split into `results` (two or more values of one
    /// type), result 0 from the low bits: the inverse of [`Op::Pack`]. Pure.
    Unpack {
        results: Vec<ValueId>,
        x: ValueId,
    },
    /// A PTX half-precision instruction ([`HalfFn`]) on 16-bit floats or on
    /// pairs of them packed into Int32s ([`HalfType`]), with CuTe DSL's
    /// modifiers. With a `predicate` the instruction runs only where it is
    /// true (`@p`); elsewhere the result is the first operand (CuTe DSL
    /// leaves it undefined). Pure.
    HalfInst {
        result: ValueId,
        func: HalfFn,
        ty: HalfType,
        args: Vec<ValueId>,
        mods: HalfModifiers,
        predicate: Option<ValueId>,
    },
    /// A PTX conversion with modifiers ([`CvtFn`]): CuTe DSL's narrow
    /// conversion helpers. Pure.
    Cvt {
        result: ValueId,
        func: CvtFn,
        args: Vec<ValueId>,
    },
    /// `x` itself, which the program promises is a multiple of `divby` and,
    /// with `pow2`, a positive power of two (CuTe DSL's `cute.assume`). The
    /// result carries those facts for the builder (which uses them to prove
    /// accesses aligned); breaking the promise is undefined behaviour, which
    /// the simulator reports. The MLIR backend passes a power-of-two
    /// `divby` on to LLVM as `llvm.intr.assume`, and nothing else. It is
    /// pure, so an unused one is removed: with no use, nothing relies on the
    /// promise.
    Assume {
        result: ValueId,
        x: ValueId,
        divby: u64,
        pow2: bool,
    },
    /// The address of element `index` (counting `elem`s) of `buffer`, a
    /// global, shared or per-thread buffer, as an integer in the buffer's
    /// own address space (CuTe DSL's `cute.ptrtoint`): an `i64` global
    /// address, or the `i32` offset in the block's shared-memory window or
    /// the thread's local window.
    PtrToInt {
        result: ValueId,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
    },
    /// The integer address `x` of space `from` as one of space `to`, one of
    /// them generic (`cvta` and `cvta.to`, LLVM's `addrspacecast`): `i64`
    /// for generic and global addresses, `i32` for shared and local ones.
    /// A generic address outside the target space's window is undefined.
    CvtAddr {
        result: ValueId,
        x: ValueId,
        from: crate::tensor::AddressSpace,
        to: crate::tensor::AddressSpace,
    },
    // WGMMA (roadmap parts m8-wgmma-core and m8-wgmma-variants).
    /// `wgmma.mma_async.sync.aligned.m64n{N}k{K}.{acc}.{a}.{b}` (see
    /// [`WgmmaKind`]): `d = a * b + (scale_d ? c : 0)` over a warpgroup, B
    /// read from shared memory through the 64-bit matrix descriptor
    /// `desc_b` (U64), `scale_d` a Boolean (the atom's `ACCUMULATE` field).
    /// `a` is A's 64-bit descriptor (one U64) for a shared-memory A,
    /// or, for the register-sourced form ([`WgmmaKind::a_regs`]), this
    /// thread's A fragment: 8 16-bit or 16 8-bit elements in the PTX ISA's
    /// A-fragment order, which the instruction takes as four 32-bit
    /// registers. `c` and `d` are each thread's `N / 2` accumulators in the
    /// PTX ISA's D-fragment order. The operation is asynchronous: `d` holds
    /// its results only once a [`Op::WgmmaWait`] covers it, and may
    /// meanwhile flow only into other wgmmas' `c`, waits, operand fences and
    /// loop carries.
    Wgmma {
        kind: WgmmaKind,
        d: Vec<ValueId>,
        c: Vec<ValueId>,
        a: Vec<ValueId>,
        desc_b: ValueId,
        scale_d: ValueId,
    },
    /// `wgmma.fence.sync.aligned`, after an operand fence on `inputs` (the
    /// accumulators of the warpgroup's MMAs): `results[i]` is `inputs[i]`,
    /// pinned in registers before the fence, so register writes cannot
    /// sink past it.
    WgmmaFence {
        results: Vec<ValueId>,
        inputs: Vec<ValueId>,
    },
    /// `wgmma.commit_group.sync.aligned`.
    WgmmaCommitGroup,
    /// `wgmma.wait_group.sync.aligned pending`, then an operand fence on
    /// `inputs`: `results[i]` is `inputs[i]` read after the wait.
    /// `complete[i]` says whether the builder knows `inputs[i]` to be the
    /// result of a wgmma in a group the wait completes (always so when
    /// `pending` is 0).
    WgmmaWait {
        pending: u32,
        results: Vec<ValueId>,
        inputs: Vec<ValueId>,
        complete: Vec<bool>,
    },
    // m11-extern-distributed: external device-function calls and multimem.
    /// A call of external device function `callee` ([`KernelIr::externs`]):
    /// CuTe DSL's `cute.extern` / `cute.ffi` call, `func.call` in MLIR.
    /// `args` follow the declaration's parameters: values, and for pointer
    /// parameters the address of a buffer element. `body`, when present, is
    /// what the simulator runs in place of the call (the callee traced in
    /// the caller, as a [`DeviceFunction`](crate::ffi::DeviceFunction) or an
    /// extern's DSL simulation), yielding `results`; the backend emits only
    /// the call.
    Call {
        callee: ExternId,
        args: Vec<AsmInput>,
        results: Vec<ValueId>,
        body: Option<Region>,
    },
    /// A `multimem` instruction on a multicast buffer (CuTe DSL's
    /// `cutlass.utils.distributed`): see [`MultimemOp`]. `values` are the
    /// registers stored (`St`), `results` the registers loaded (`LdReduce`).
    Multimem {
        op: MultimemOp,
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        values: Vec<ValueId>,
        results: Vec<ValueId>,
    },
    // m10-tmem-core: tensor memory.
    /// A `tcgen05` instruction (sm_100), one family for all of them: see
    /// [`Tcgen05Inst`] for what `buffer` and `args` hold. `two_cta` is
    /// `.cta_group::2` (a CTA pair) rather than `.cta_group::1`;
    /// `exclusive` is the sm_107 `.exclusive` form of alloc and dealloc.
    Tcgen05 {
        inst: Tcgen05Inst,
        two_cta: bool,
        exclusive: bool,
        buffer: Option<BufferId>,
        args: Vec<ValueId>,
    },
    // m10-tmem-ld-st: loads and stores between tensor memory and registers.
    /// `tcgen05.ld.sync.aligned.<shape>.x<num>{.pack::16b}.b32 {results},
    /// [taddr]{, half_split}` (sm_100a): `results` are the elements the
    /// 32-bit registers hold (`shape.registers(num)` registers, each of `32 /
    /// bits` elements of one type of 8, 16 or 32 bits, the first in the low
    /// bits), `taddr` the warp-uniform TMEM address (i32). Warp-collective
    /// and asynchronous: the registers are undefined until the warp's
    /// `tcgen05.wait::ld`. The `.16x32bx2` half split is
    /// [`TmemShape::half_split`].
    ///
    /// m10-ldred-packed-math: with `red`, the `tcgen05.ld.red` form (PTX ISA
    /// 8.8: sm_110a, the sm_103f and sm_110f families), which also reduces
    /// each thread's loaded values to one Float32 ([`TmemLdRed`]); the
    /// results are then Float32 values and `pack` is false. The reduction
    /// is as pending as the data.
    ///
    /// m10-rubin-extras: with `sp`, the `.spcompress` form (PTX ISA 9.4:
    /// sm_107a): `results` are only the compressed data ([`TmemLdSpCompress`]).
    Tcgen05Ld {
        results: Vec<ValueId>,
        taddr: ValueId,
        shape: TmemShape,
        num: u32,
        pack: bool,
        red: Option<TmemLdRed>,
        sp: Option<TmemLdSpCompress>,
    },
    /// `tcgen05.st.sync.aligned.<shape>.x<num>{.unpack::16b}.b32 [taddr]{,
    /// half_split}, {values}` (sm_100a): `values` are the registers'
    /// elements, as for [`Op::Tcgen05Ld`]. Warp-collective and asynchronous
    /// until the warp's `tcgen05.wait::st`.
    Tcgen05St {
        values: Vec<ValueId>,
        taddr: ValueId,
        shape: TmemShape,
        num: u32,
        unpack: bool,
    },
    // m10-umma-f16: tcgen05 MMAs and their commits.
    /// One warp-level issue of `tcgen05.mma.cta_group::1.kind::<kind>
    /// [taddr], desc_a, desc_b, idesc, enable_input_d` (sm_100a): executed by
    /// a converged warp, whose lane `elect` (the Bool of an `elect.sync`)
    /// issues the instruction. `taddr` is the TMEM address of D (i32),
    /// `desc_a` and `desc_b` the UMMA shared-memory descriptors (u64), and
    /// `enable_input_d` (Bool) whether D accumulates (`false`: D = A B). The
    /// instruction descriptor is `kind.idesc()`, fixed when the kernel is
    /// built.
    ///
    /// m10-umma-kinds: with `kind.a_tmem`, `desc_a` is A's TMEM address
    /// (i32, the `[a-tmem]` form). `idesc`, when given, is the instruction
    /// descriptor as a runtime u32 (the constant bits OR the runtime
    /// negate bits); `disable_lane`, when not empty, the
    /// `{disable-output-lane}` words (i32, `kind.disable_lane_words()`).
    Tcgen05Mma {
        kind: UmmaKind,
        taddr: ValueId,
        desc_a: ValueId,
        desc_b: ValueId,
        enable_input_d: ValueId,
        elect: ValueId,
        idesc: Option<ValueId>,
        disable_lane: Vec<ValueId>,
        /// m10-umma-blockscaled: the TMEM addresses (i32) of the scale
        /// factors of A and B (`[sfa-tmem]`, `[sfb-tmem]`) of a block-scaled
        /// kind (`kind.scale`); their scale-factor IDs (the byte of each
        /// 32-bit column the first factor is in) are in `idesc`.
        sf: Option<(ValueId, ValueId)>,
        /// m10-sparse-umma: the TMEM address (i32) of the sparsity metadata
        /// (`[sp-meta-tmem]`) of a sparse kind (`kind.sparse`); its
        /// selector is in `idesc`'s bits 0-1.
        meta: Option<ValueId>,
        /// m11-prims-mma-forms: the `zero-column-mask-desc` (u64) of a
        /// `.ws` MMA (`kind.prims.ws`).
        zero_col_mask: Option<ValueId>,
    },
    /// `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster{.multicast::cluster}.b64
    /// [mbar]{, mask}` (sm_100a): one arrival on the mbarrier at element
    /// `index` of `buffer` once the executing thread's earlier tcgen05
    /// MMAs have completed; with `mask` (an i16 of CTA ranks) on the
    /// mbarrier at the same offset of each CTA it names. Per thread.
    Tcgen05Commit {
        buffer: BufferId,
        index: ValueId,
        mask: Option<ValueId>,
        two_cta: bool,
    },
    // m10-s2t-copies: shared-memory-to-TMEM copies.
    /// One warp-level issue of `tcgen05.cp.cta_group::{1,2}.<shape> [taddr],
    /// s-desc` (sm_100a): executed by a converged warp, whose lane `elect`
    /// (the Bool of an `elect.sync`) issues the instruction. `taddr` is the
    /// destination TMEM address (i32), `desc` the shared-memory matrix
    /// descriptor of the source (u64, K-major core matrices). Asynchronous
    /// on the CTA's tensor-core pipe, in order with its `tcgen05.mma`s, and
    /// completed through `tcgen05.commit`. With `two_cta` the pair's even
    /// CTA issues it and each CTA copies from its own shared memory into its
    /// own TMEM.
    Tcgen05Cp {
        shape: TmemCpShape,
        two_cta: bool,
        taddr: ValueId,
        desc: ValueId,
        elect: ValueId,
    },
    // m10-sm120-blockscaled: SM120 warp-level block-scaled mma.sync.
    /// `mma.sync.aligned.<shape>.row.col.kind::<kind>.block_scale.scale_vec::<n>X.f32.<a>.<b>.f32.<sf>
    /// d, a, b, c, sfa, {0, 0}, sfb, {0, 0}` (sm_120a): executed by a converged
    /// warp. `a` and `b` are the operand registers (u32; see
    /// [`BlockScaledMmaKind::operand_registers`] for how values are packed),
    /// `c` and `d` the four f32 accumulators, `sfa` and `sfb` the scale-factor
    /// registers (u32, the scale of K-block `j` in byte `j`). Byte-id and
    /// thread-id are 0, as CUTLASS passes them.
    MmaBlockScaled {
        kind: BlockScaledMmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        sfa: ValueId,
        sfb: ValueId,
    },
    // m11-prims-mma-forms: SM120 warp-level sparse block-scaled mma.
    /// `mma.sp::ordered_metadata.sync.aligned.m16n8k{64,128}.row.col.kind::<kind>.block_scale.scale_vec::<n>X.f32.<a>.<b>.f32.<sf>
    /// d, a, b, c, e, 0, sfa, {bid-a, tid-a}, sfb, {bid-b, tid-b}`
    /// (sm_120a): executed by a converged warp. `a` (4 u32 registers, the
    /// compressed half of K), `b` (4), `c` and `d` (4 f32), the metadata
    /// register `e` (sparsity selector 0) and the scale-factor registers
    /// `sfa` and `sfb` (u32); the byte and thread ids are in `kind`.
    MmaSpBlockScaled {
        kind: SpBlockScaledMmaKind,
        d: Vec<ValueId>,
        a: Vec<ValueId>,
        b: Vec<ValueId>,
        c: Vec<ValueId>,
        e: ValueId,
        sfa: ValueId,
        sfb: ValueId,
    },
    // m10-clc-scheduling: cluster launch control.
    /// `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128
    /// [resp], [mbar]` (sm_100, PTX 8.6; the multicast form sm_100a): take a
    /// still-queued cluster of the grid off the launch queue. The 16-byte
    /// response lands asynchronously at element `index` of `buffer` (of type
    /// `elem`), completing 16 bytes of tx on the mbarrier at element
    /// `mbar_index` of `mbar`; with `multicast`, at the same offsets in every
    /// CTA of the cluster. Per thread.
    ClcTryCancel {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        mbar: BufferId,
        mbar_index: ValueId,
        multicast: bool,
    },
    /// A 16-byte shared-memory load of a try_cancel response at element
    /// `index` of `buffer`, decoded with
    /// `clusterlaunchcontrol.query_cancel`: `results` are the first CTA's x,
    /// y and z and whether a cluster was cancelled (1 or 0), all i32.
    ClcQueryCancel {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        results: [ValueId; 4],
    },
    // Roadmap part m11-prims-nvvm-layer.
    /// `barrier.cta.red.{popc.u32,and.pred,or.pred}.aligned id{, count}, pred`
    /// (`bar.red`): a `bar.sync` at named barrier `id` that also reduces the
    /// participating threads' predicates `pred` (Bool) and returns the
    /// result to each of them (`result`: U32 for popc, Bool otherwise).
    BarrierRed {
        result: ValueId,
        kind: BarrierRedKind,
        pred: ValueId,
        id: ValueId,
        threads: Option<ValueId>,
    },
    /// The mbarrier operations of [`MbarrierExtKind`] on the mbarrier at
    /// element `index` of shared buffer `buffer`: `value` is the count or the
    /// bytes of the kinds that take one, `result` (i64) the state token of
    /// those that return one.
    MbarrierExt {
        kind: MbarrierExtKind,
        result: Option<ValueId>,
        buffer: BufferId,
        index: ValueId,
        value: Option<ValueId>,
    },
    /// A per-thread marker or hint ([`PrimMarker`]).
    PrimMarker {
        marker: PrimMarker,
    },
    /// `exit`: the executing threads stop.
    Exit,
    /// `st.bulk.weak.shared::cta [addr], size, 0` (sm_100): zero `size` (i64)
    /// bytes of shared memory from element `index` (of type `elem`) of `buffer`.
    StBulk {
        buffer: BufferId,
        index: ValueId,
        elem: DType,
        size: ValueId,
    },
    /// `tcgen05.shift.cta_group::{1,2}.down [taddr]` (sm_100a): shift the rows
    /// of the 32-lane subpartitions at TMEM address `taddr` (i32) down by one
    /// lane, the last row of each staying. Issued by one thread.
    Tcgen05Shift {
        taddr: ValueId,
        two_cta: bool,
    },
}

/// The reduction of a `tcgen05.ld.red` (CuTe DSL's `tcgen05.TmemLoadRedOp`):
/// the minimum or maximum, of the values or of their absolute values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TmemLoadRedOp {
    #[default]
    Max,
    Min,
    MaxAbs,
    MinAbs,
}

impl TmemLoadRedOp {
    /// The PTX qualifiers: `.max`, `.min`, `.max.abs` or `.min.abs`.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemLoadRedOp::Max => ".max",
            TmemLoadRedOp::Min => ".min",
            TmemLoadRedOp::MaxAbs => ".max.abs",
            TmemLoadRedOp::MinAbs => ".min.abs",
        }
    }

    /// Whether this takes the minimum.
    pub fn is_min(self) -> bool {
        matches!(self, TmemLoadRedOp::Min | TmemLoadRedOp::MinAbs)
    }

    /// Whether this reduces absolute values.
    pub fn is_abs(self) -> bool {
        matches!(self, TmemLoadRedOp::MaxAbs | TmemLoadRedOp::MinAbs)
    }
}

/// The reduction of an [`Op::Tcgen05Ld`] (`tcgen05.ld.red`): the Float32
/// `result` is `op` over the values the thread loaded, as PTX's `min` and
/// `max` compute it (a NaN is ignored unless `nan`, which makes any NaN the
/// result). `half_split` is the immediate column offset of `.16x32bx2`'s
/// second access.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmemLdRed {
    pub result: ValueId,
    pub op: TmemLoadRedOp,
    pub nan: bool,
    pub half_split: Option<u32>,
}

/// The 2:4 compression of an [`Op::Tcgen05Ld`] (`tcgen05.ld{.red}.spcompress`,
/// PTX ISA 9.4, sm_107a; roadmap part m10-rubin-extras): of every four
/// consecutive Float32 values a thread loads, the two the row operation `op`
/// prefers (the larger for `max`, the smaller for `min`, by absolute value
/// for the `abs` forms; a NaN is the least preferred unless `nan`, the most;
/// the lower position wins a tie) are kept, in order. The op's `results` are
/// the kept values (half the loaded ones), `mdata` the metadata registers
/// (i32): the 4-bit field of group `g` (the position of the first kept value
/// in bits 0-1, of the second in bits 2-3, the sparse MMA's format) at bit
/// `4 (g % 8)` of register `g / 8`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TmemLdSpCompress {
    pub mdata: Vec<ValueId>,
    pub op: TmemLoadRedOp,
    pub nan: bool,
}

/// The shapes of `tcgen05.cp` (PTX ISA 9.7.18.9.2, roadmap part
/// m10-s2t-copies), with their broadcast (multicast) qualifiers: the
/// source is `rows` rows of `row_bits` bits in shared memory, written to
/// `broadcast()` TMEM lane groups.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemCpShape {
    /// `.128x256b`: 128 rows of 256 bits into 128 lanes.
    S128x256b,
    /// `.4x256b`: 4 rows of 256 bits into lanes 0, 32, 64 and 96.
    S4x256b,
    /// `.128x128b`: 128 rows of 128 bits into 128 lanes.
    S128x128b,
    /// `.64x128b.warpx2::02_13`: 64 rows of 128 bits, row `r` into lanes
    /// `r` and `64 + r`.
    S64x128bWarpx2_02_13,
    /// `.64x128b.warpx2::01_23`: 64 rows of 128 bits, rows 0 to 31 into
    /// lanes 0 to 63 (twice), rows 32 to 63 into lanes 64 to 127.
    S64x128bWarpx2_01_23,
    /// `.32x128b.warpx4`: 32 rows of 128 bits into each 32-lane quadrant.
    S32x128bWarpx4,
}

impl TmemCpShape {
    pub const ALL: [TmemCpShape; 6] = [
        TmemCpShape::S128x256b,
        TmemCpShape::S4x256b,
        TmemCpShape::S128x128b,
        TmemCpShape::S64x128bWarpx2_02_13,
        TmemCpShape::S64x128bWarpx2_01_23,
        TmemCpShape::S32x128bWarpx4,
    ];

    /// The PTX qualifiers, without the leading dot.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemCpShape::S128x256b => "128x256b",
            TmemCpShape::S4x256b => "4x256b",
            TmemCpShape::S128x128b => "128x128b",
            TmemCpShape::S64x128bWarpx2_02_13 => "64x128b.warpx2::02_13",
            TmemCpShape::S64x128bWarpx2_01_23 => "64x128b.warpx2::01_23",
            TmemCpShape::S32x128bWarpx4 => "32x128b.warpx4",
        }
    }

    /// Rows of the shared-memory source.
    pub fn rows(self) -> usize {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S128x128b => 128,
            TmemCpShape::S4x256b => 4,
            TmemCpShape::S64x128bWarpx2_02_13 | TmemCpShape::S64x128bWarpx2_01_23 => 64,
            TmemCpShape::S32x128bWarpx4 => 32,
        }
    }

    /// Bits of each row: 256 or 128 (8 or 4 TMEM columns).
    pub fn row_bits(self) -> usize {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S4x256b => 256,
            _ => 128,
        }
    }

    /// TMEM lane groups each source row is written to: 4 for `.warpx4`, 2
    /// for `.warpx2`, else 1.
    pub fn broadcast(self) -> usize {
        match self {
            TmemCpShape::S32x128bWarpx4 => 4,
            TmemCpShape::S64x128bWarpx2_02_13 | TmemCpShape::S64x128bWarpx2_01_23 => 2,
            _ => 1,
        }
    }

    /// The TMEM lanes (after the address's) that source row `row` is
    /// written to.
    pub fn lanes_of_row(self, row: usize) -> Vec<usize> {
        match self {
            TmemCpShape::S128x256b | TmemCpShape::S128x128b => vec![row],
            TmemCpShape::S4x256b => vec![32 * row],
            TmemCpShape::S64x128bWarpx2_02_13 => vec![row, 64 + row],
            TmemCpShape::S64x128bWarpx2_01_23 => {
                let base = 64 * (row / 32) + row % 32;
                vec![base, base + 32]
            }
            TmemCpShape::S32x128bWarpx4 => (0..4).map(|q| 32 * q + row).collect(),
        }
    }
}

/// The data-movement shapes of `tcgen05.ld` and `tcgen05.st` (PTX ISA
/// 9.7.18.2.3, roadmap part m10-tmem-ld-st).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemShape {
    /// `.16x64b`: 16 lanes of 64 bits per repetition.
    S16x64b,
    /// `.16x128b`: 16 lanes of 128 bits.
    S16x128b,
    /// `.16x256b`: 16 lanes of 256 bits.
    S16x256b,
    /// `.16x32bx2`: two accesses of 16 lanes of 32 bits, the second at the
    /// first's address plus an immediate column offset.
    S16x32bx2,
    /// `.32x32b`: 32 lanes of 32 bits.
    S32x32b,
}

impl TmemShape {
    /// The PTX qualifier, without the dot.
    pub fn ptx(self) -> &'static str {
        match self {
            TmemShape::S16x64b => "16x64b",
            TmemShape::S16x128b => "16x128b",
            TmemShape::S16x256b => "16x256b",
            TmemShape::S16x32bx2 => "16x32bx2",
            TmemShape::S32x32b => "32x32b",
        }
    }

    /// TMEM lanes one instruction accesses (16 or 32).
    pub fn lanes(self) -> u32 {
        if self == TmemShape::S32x32b { 32 } else { 16 }
    }

    /// 32-bit registers per thread per repetition (the PTX ISA's Table 59).
    pub fn registers_per_repetition(self) -> u32 {
        match self {
            TmemShape::S16x128b => 2,
            TmemShape::S16x256b => 4,
            _ => 1,
        }
    }

    /// 32-bit registers per thread of `.x<num>`.
    pub fn registers(self, num: u32) -> u32 {
        self.registers_per_repetition() * num
    }

    /// The largest repetition: at most 128 registers per thread.
    pub fn max_repetition(self) -> u32 {
        128 / self.registers_per_repetition()
    }

    /// The immediate column offset of the second access of `.16x32bx2`
    /// (the columns of the first, as CuTe's SM100_TMEM_LOAD_16dp32b
    /// instructions use), `None` for the other shapes.
    pub fn half_split(self, num: u32, pack: bool) -> Option<u32> {
        (self == TmemShape::S16x32bx2).then_some(num * if pack { 2 } else { 1 })
    }

    /// Where thread `lane` (0 to 31) of the warp finds register `reg` of an
    /// `.x<num>` access without packing: `(TMEM lane, TMEM column)` relative
    /// to the address (PTX ISA 9.7.18.2.3.1, matrix fragments, as CuTe's
    /// SM100 TMEM copy traits lay them out). With `.pack::16b`
    /// (`.unpack::16b`) the register's two halves are the low halves of
    /// columns `2 * column` and the next one: see [`TmemShape::cells`].
    pub fn fragment(self, lane: u32, reg: u32, num: u32) -> (u32, u32) {
        match self {
            TmemShape::S32x32b => (lane, reg),
            TmemShape::S16x64b => (lane / 4 + 8 * (lane % 2), (lane / 2) % 2 + 2 * reg),
            TmemShape::S16x128b => (lane / 4 + 8 * (reg % 2), lane % 4 + 4 * (reg / 2)),
            TmemShape::S16x256b => (lane / 4 + 8 * ((reg / 2) % 2), 2 * (lane % 4) + reg % 2 + 8 * (reg / 4)),
            TmemShape::S16x32bx2 => (lane % 16, num * (lane / 16) + reg),
        }
    }

    /// The TMEM cells (lane, column) relative to the address that hold
    /// register `reg` of thread `lane`: one 32-bit cell, or with packing
    /// the low halves of two adjacent cells (the register's low half
    /// first).
    pub fn cells(self, lane: u32, reg: u32, num: u32, packed: bool) -> Vec<(u32, u32)> {
        let (l, c) = self.fragment(lane, reg, num);
        if packed { vec![(l, 2 * c), (l, 2 * c + 1)] } else { vec![(l, c)] }
    }

    /// The TMEM columns an `.x<num>` access spans from its address.
    pub fn columns(self, num: u32, packed: bool) -> u32 {
        let base = match self {
            TmemShape::S32x32b => num,
            TmemShape::S16x64b | TmemShape::S16x32bx2 => 2 * num,
            TmemShape::S16x128b => 4 * num,
            TmemShape::S16x256b => 8 * num,
        };
        if packed { 2 * base } else { base }
    }
}

/// The instructions of [`Op::Tcgen05`] (roadmap part m10-tmem-core).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tcgen05Inst {
    /// `tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst],
    /// ncols`: `buffer` is the shared buffer of `dst`, `args` are `[index,
    /// ncols]`, `index` counting 32-bit words of the buffer. Warp-collective;
    /// it writes the TMEM address of the allocation to `dst`.
    Alloc,
    /// `tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32 taddr, ncols`:
    /// `args` are `[taddr, ncols]` (both i32). Warp-collective.
    Dealloc,
    /// `tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned`: no
    /// operands. Warp-collective.
    RelinquishAllocPermit,
    /// No instruction: the simulator checks that the TMEM address `args[0]`
    /// (i32), just loaded by `retrieve_tmem_ptr`, lies in a live allocation of
    /// the CTA. The backend emits nothing for it.
    CheckAddress,
    /// `tcgen05.wait::ld.sync.aligned` (roadmap part m10-tmem-ld-st): the
    /// warp's earlier `tcgen05.ld`s have completed. Warp-collective.
    WaitLd,
    /// `tcgen05.wait::st.sync.aligned`: the warp's earlier `tcgen05.st`s
    /// have completed. Warp-collective.
    WaitSt,
    /// `tcgen05.fence::before_thread_sync`: orders the thread's earlier
    /// tcgen05 operations before a later thread synchronisation. Per thread.
    FenceBeforeThreadSync,
    /// `tcgen05.fence::after_thread_sync`: orders a thread synchronisation
    /// before the thread's later tcgen05 operations. Per thread.
    FenceAfterThreadSync,
}

impl Tcgen05Inst {
    /// The PTX instruction's name (without qualifiers).
    pub fn name(self) -> &'static str {
        match self {
            Tcgen05Inst::Alloc => "tcgen05.alloc",
            Tcgen05Inst::Dealloc => "tcgen05.dealloc",
            Tcgen05Inst::RelinquishAllocPermit => "tcgen05.relinquish_alloc_permit",
            Tcgen05Inst::CheckAddress => "retrieve_tmem_ptr",
            Tcgen05Inst::WaitLd => "tcgen05.wait::ld",
            Tcgen05Inst::WaitSt => "tcgen05.wait::st",
            Tcgen05Inst::FenceBeforeThreadSync => "tcgen05.fence::before_thread_sync",
            Tcgen05Inst::FenceAfterThreadSync => "tcgen05.fence::after_thread_sync",
        }
    }

    /// Whether every lane of the warp executes it together
    /// (`.sync.aligned`): all but the fences and the address check.
    pub fn is_warp_collective(self) -> bool {
        !matches!(
            self,
            Tcgen05Inst::CheckAddress | Tcgen05Inst::FenceBeforeThreadSync | Tcgen05Inst::FenceAfterThreadSync
        )
    }

    /// Whether it carries a `cta_group` (alloc, dealloc, relinquish).
    pub fn has_cta_group(self) -> bool {
        matches!(self, Tcgen05Inst::Alloc | Tcgen05Inst::Dealloc | Tcgen05Inst::RelinquishAllocPermit)
    }
}

/// The data kinds of `tcgen05.mma` (PTX ISA 9.7.18.10, `.kind`): F16
/// (roadmap part m10-umma-f16), and TF32, I8 and F8F6F4 (m10-umma-dense-kinds)
/// on the same machinery.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UmmaDataKind {
    /// `.kind::f16`: Float16 or BFloat16 operands, Float16 or Float32
    /// accumulators.
    F16,
    // m10-umma-dense-kinds.
    /// `.kind::tf32`: TFloat32 operands (32-bit in shared memory, of which
    /// the tensor core uses the top 19 bits), a Float32 accumulator, K = 8.
    Tf32,
    /// `.kind::i8`: Int8 or Uint8 operands, an Int32 accumulator, K = 32.
    I8,
    /// `.kind::f8f6f4`: E4M3, E5M2, E2M3, E3M2 or E2M1 operands (each of A
    /// and B), a Float16 or Float32 accumulator, K = 32. FP6 and FP4
    /// operands are read from the padded shared-memory image (16 values in
    /// each 16-byte slot, PTX's `.b6x16_p32` and `.b4x16_p64`).
    F8F6F4,
    // m10-umma-blockscaled.
    /// `.kind::mxf8f6f4.block_scale`: the operands of `F8F6F4` (FP6 and FP4
    /// from the padded image), UE8M0 scale factors, one per 32 K elements,
    /// a Float32 accumulator, K = 32.
    Mxf8f6f4,
    /// `.kind::mxf4.block_scale`: E2M1 operands packed two per byte, UE8M0
    /// scale factors, one per 32 K elements, a Float32 accumulator, K = 64
    /// (96 on sm_103a).
    Mxf4,
    /// `.kind::mxf4nvf4.block_scale`: E2M1 operands packed two per byte,
    /// UE4M3 or UE8M0 scale factors, one per 16 K elements, a Float32
    /// accumulator, K = 64 (96 on sm_103a).
    Mxf4Nvf4,
}

impl UmmaDataKind {
    /// The PTX qualifier, without `.kind::`.
    pub fn ptx(self) -> &'static str {
        match self {
            UmmaDataKind::F16 => "f16",
            UmmaDataKind::Tf32 => "tf32",
            UmmaDataKind::I8 => "i8",
            UmmaDataKind::F8F6F4 => "f8f6f4",
            UmmaDataKind::Mxf8f6f4 => "mxf8f6f4",
            UmmaDataKind::Mxf4 => "mxf4",
            UmmaDataKind::Mxf4Nvf4 => "mxf4nvf4",
        }
    }

    /// m10-umma-blockscaled: whether the kind is block-scaled (its
    /// instruction descriptor is CuTe C++'s `InstrDescriptorBlockScaled`).
    pub fn is_block_scaled(self) -> bool {
        matches!(self, UmmaDataKind::Mxf8f6f4 | UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4)
    }

    /// The instruction descriptor's 3-bit format of an A or B operand of
    /// type `t` (PTX ISA Table 51; CuTe C++'s `UMMA::to_UMMAFormat`).
    pub fn operand_format(self, t: DType) -> Option<u32> {
        match (self, t) {
            (UmmaDataKind::F16, DType::F16) => Some(0),
            (UmmaDataKind::F16, DType::BF16) => Some(1),
            (UmmaDataKind::Tf32, DType::TF32) => Some(2),
            (UmmaDataKind::I8, DType::U8) => Some(0),
            (UmmaDataKind::I8, DType::I8) => Some(1),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F8E4M3FN) => Some(0),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F8E5M2) => Some(1),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F6E2M3FN) => Some(3),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F6E3M2FN) => Some(4),
            (UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4, DType::F4E2M1FN) => Some(5),
            // m10-umma-blockscaled: packed E2M1 (CuTe C++'s MXF4Format::E2M1).
            (UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4, DType::F4E2M1FN) => Some(1),
            _ => None,
        }
    }

    /// K of one instruction.
    pub fn k(self) -> usize {
        match self {
            UmmaDataKind::F16 => 16,
            UmmaDataKind::Tf32 => 8,
            UmmaDataKind::I8 | UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4 => 32,
            UmmaDataKind::Mxf4 | UmmaDataKind::Mxf4Nvf4 => 64,
        }
    }

    /// The bits one element of an operand of type `t` takes in shared
    /// memory: its width, except FP6 and FP4 under `.kind::f8f6f4`, which
    /// are read from the padded image (16 values per 16 bytes), one byte
    /// per value as far as addresses go.
    pub fn smem_bits(self, t: DType) -> usize {
        if matches!(self, UmmaDataKind::F8F6F4 | UmmaDataKind::Mxf8f6f4) && t.bits() < 8 { 8 } else { t.bits() }
    }
}

/// One `tcgen05.mma` instruction (roadmap part m10-umma-f16): shape `m x n
/// x k` (K: 32 bytes of the operand type), A and B of types `a` and `b`
/// from shared memory, accumulated in `acc` in TMEM; A and B are MN-major
/// when `trans_a` and `trans_b` are set. `two_cta` is `.cta_group::2`.
/// With `a_tmem` (roadmap part m10-umma-kinds) A comes from tensor memory
/// (the `[a-tmem]` form, always K-major).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UmmaKind {
    pub kind: UmmaDataKind,
    pub m: u32,
    pub n: u32,
    pub a: DType,
    pub b: DType,
    pub acc: DType,
    pub trans_a: bool,
    pub trans_b: bool,
    pub two_cta: bool,
    pub a_tmem: bool,
    /// m10-umma-blockscaled: the scale factors of a block-scaled kind.
    pub scale: Option<UmmaScale>,
    /// m10-sparse-umma: `tcgen05.mma.sp`, A 2:4 structured sparse (stored
    /// compressed, half of K), with metadata in tensor memory.
    pub sparse: bool,
    /// m10-rubin-umma: the Rubin (sm_107) forms: the doubled K and the
    /// collector-buffer qualifiers.
    pub sm107: Option<Sm107Umma>,
    /// m11-prims-mma-forms: the forms and modifiers only `cute::prims`
    /// issues (weight-stationary `.ws`, `.collector::a`, `.ashift`,
    /// `scale-input-d`).
    pub prims: UmmaPrims,
}

/// The parts of a `tcgen05.mma` only `cute::prims` issues (roadmap part
/// m11-prims-mma-forms). The default is none of them.
///
/// * `thread_issue`: issued by each thread that executes it (a prims call
///   inside `elect_one`), not by a warp that elects a lane (`gemm`).
/// * `ws`: `tcgen05.mma.ws` (weight stationary, `cta_group::1`, M of 32,
///   64 or 128 in the data-path Layouts G, E and D), with B's collector
///   buffer `collector_b` (`.collector::b<i>::<op>`, `i` 0 to 3) and the
///   instruction descriptor's `max_shift` (bits 30-31).
/// * `collector_a`: `.collector::a::<op>` on a plain (sm_100a) MMA.
/// * `ashift`: `.ashift` (A in TMEM, shifted up one row in each 32-lane
///   sub-partition after the MMA).
/// * `scale_input_d`: `D = A B + D 2^-s` (`.kind::f16` and `tf32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct UmmaPrims {
    pub thread_issue: bool,
    pub ws: bool,
    pub collector_b: Option<(u8, CollectorOp)>,
    pub max_shift: u8,
    pub collector_a: CollectorOp,
    pub ashift: bool,
    pub scale_input_d: Option<u8>,
}

/// CuTe DSL's `tcgen05.CollectorOp` (roadmap part m10-rubin-umma): what a
/// Rubin `tcgen05.mma` does with the collector buffer of its A or B operand
/// (PTX `.collector::a::<op>` and `.collector::b::<op>`). `FILL` reads the
/// operand from shared memory and keeps a copy in the collector; `USE`
/// reads it from the collector and keeps it there; `LASTUSE` reads it from
/// the collector and releases it; `DISCARD` (the default, no qualifier)
/// reads shared memory and leaves nothing behind.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CollectorOp {
    #[default]
    DISCARD,
    LASTUSE,
    FILL,
    USE,
}

impl CollectorOp {
    /// The PTX qualifier's last part.
    pub fn ptx(self) -> &'static str {
        match self {
            CollectorOp::DISCARD => "discard",
            CollectorOp::LASTUSE => "lastuse",
            CollectorOp::FILL => "fill",
            CollectorOp::USE => "use",
        }
    }

    /// Whether the MMA reads its operand from the collector buffer rather
    /// than shared memory.
    pub fn reads_collector(self) -> bool {
        matches!(self, CollectorOp::USE | CollectorOp::LASTUSE)
    }
}

/// The Rubin parts of a `tcgen05.mma` (roadmap part m10-rubin-umma):
/// `wide_k`, K doubled (`.kind::f8f6f4` and `.kind::mxf8f6f4` K = 64,
/// `.kind::mxf4nvf4` K = 128), and the collector usage of A and B.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sm107Umma {
    pub wide_k: bool,
    pub collector_a: CollectorOp,
    pub collector_b: CollectorOp,
}

/// The scale factors of a block-scaled `tcgen05.mma` (roadmap part
/// m10-umma-blockscaled): their type (`F8E8M0FNU` or `F8E4M3FN`), the K
/// elements each scales (`vec`, 16 or 32), and whether the instruction is
/// the sm_103a K = 96 form (`.block16` / `.block32`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UmmaScale {
    pub sf: DType,
    pub vec: u32,
    pub k96: bool,
}

impl UmmaKind {
    /// K: 32 bytes of the operand type (as it is held in shared memory);
    /// 96 for the sm_103a K = 96 forms.
    /// m10-sparse-umma: twice that for a sparse kind (A stores half).
    pub fn k(&self) -> usize {
        let k = if self.scale.is_some_and(|s| s.k96) { 96 } else { self.kind.k() };
        // m10-rubin-umma: the Rubin forms double it.
        let k = if self.sm107.is_some_and(|r| r.wide_k) { 2 * k } else { k };
        if self.sparse { 2 * k } else { k }
    }

    /// m10-sparse-umma: K of A as stored (compressed for a sparse kind).
    pub fn a_k(&self) -> usize {
        if self.sparse { self.k() / 2 } else { self.k() }
    }

    /// m10-sparse-umma: the sparse instruction descriptor's flag (bit 2)
    /// and the metadata selector field (bits 0-1, CuTe C++'s `sparse_id2_`).
    pub const SPARSE_BIT: u32 = 2;
    pub const SPARSE_ID2_MASK: u32 = 3;

    /// m10-umma-blockscaled: the scale factors per row of A (column of B)
    /// one instruction reads, K / the vector size.
    pub fn scale_factors(&self) -> usize {
        self.scale.map_or(0, |s| self.k() / s.vec as usize)
    }

    /// The block-scaled instruction descriptor's scale-factor-ID fields:
    /// A's at bit 29 (2 bits), B's at bit 4.
    pub const A_SF_ID_BIT: u32 = 29;
    pub const B_SF_ID_BIT: u32 = 4;

    /// `(M, N, K)`.
    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        (self.m as usize, self.n as usize, self.k())
    }

    /// The 32-bit instruction descriptor (PTX ISA 9.7.18.4.2, Table 51;
    /// CuTe C++'s `UMMA::InstrDescriptor`): dense, no saturation or
    /// negation, `c_format` in bits 4-5 (F16 0, F32 1, S32 2), the A and B
    /// formats in bits 7-9 and 10-12, the A and B majors (1: MN-major) in
    /// bits 15 and 16, `N >> 3` in bits 17-22 and `M >> 4` in bits 24-28.
    /// Panics on a type the kind does not take.
    ///
    /// m10-umma-blockscaled: a block-scaled kind has CuTe C++'s
    /// `InstrDescriptorBlockScaled` (ported from
    /// include/cute/arch/mma_sm100_desc.hpp): no accumulator format (it is
    /// Float32), the B and A scale-factor IDs in bits 4-5 and 29-30 (0
    /// here; the runtime descriptor ORs them in), the scale format in bit 23
    /// (UE4M3 0, UE8M0 1) and K = 96 in bit 31.
    #[track_caller]
    pub fn idesc(&self) -> u32 {
        // m11-prims-descriptors-swizzle: the bits are packed by the public
        // builders (`prims::Tcgen05InstrDesc`, `Tcgen05MxInstrDesc`,
        // `Tcgen05MxOmmaInstrDesc`), which check each field.
        use crate::prims::descriptors as d;
        let fmt = |t: DType| {
            d::Tcgen05InputFormat::for_dtype(self.kind, t)
                .unwrap_or_else(|| panic!("tcgen05.mma.kind::{}: no operand format for {t}", self.kind.ptx()))
        };
        let major =
            |trans: bool| if trans { crate::nvgpu::OperandMajorMode::MN } else { crate::nvgpu::OperandMajorMode::K };
        let wide = self.sm107.is_some_and(|r| r.wide_k);
        let fail = |e: String| -> u32 { panic!("tcgen05.mma.kind::{} instruction descriptor: {e}", self.kind.ptx()) };
        let bits = |b: d::Bits<u32>| b.as_static().expect("a descriptor of constants");
        if let Some(sc) = self.scale {
            // m10-rubin-umma: UE5M3 is format 2 of the 2-bit field (bits 23-24) of
            // CuTe C++'s Sm107MXF4NVF4InstrDescriptor.
            let scale_format = d::MxScaleFormat::for_dtype(sc.sf)
                .unwrap_or_else(|| panic!("tcgen05.mma: no scale-factor format for {}", sc.sf));
            if self.kind == UmmaDataKind::Mxf8f6f4 {
                // m10-rubin-umma: K = 64 of .kind::mxf8f6f4 is bit 31.
                let f = d::Tcgen05MxInstrDescFields {
                    sparse: self.sparse,
                    a_format: fmt(self.a),
                    b_format: fmt(self.b),
                    scale_format,
                    a_major: major(self.trans_a),
                    b_major: major(self.trans_b),
                    n: self.n,
                    m: self.m,
                    k_dim: wide || sc.k96,
                    ..Default::default()
                };
                return d::Tcgen05MxInstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()));
            }
            // m10-rubin-umma: K = 128 of .kind::mxf4nvf4 is the K-size value 2 (bit 3
            // set, bit 31 clear); the other K = 128 and the sm_103a K = 96 are 1.
            let k_dim = match (wide, self.kind) {
                (true, UmmaDataKind::Mxf4Nvf4) => 2,
                (true, _) => 1,
                (false, _) => sc.k96 as u8,
            };
            let f = d::Tcgen05MxOmmaInstrDescFields {
                sparse: self.sparse,
                a_format: fmt(self.a),
                b_format: fmt(self.b),
                scale_format,
                a_major: major(self.trans_a),
                b_major: major(self.trans_b),
                n: self.n,
                m: self.m,
                k_dim,
                ..Default::default()
            };
            return d::Tcgen05MxOmmaInstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()));
        }
        let c_format = d::Tcgen05OutputFormat::for_dtype(self.acc)
            .unwrap_or_else(|| panic!("tcgen05.mma: no accumulator format for {}", self.acc));
        let f = d::Tcgen05InstrDescFields {
            sparse: self.sparse,
            c_format,
            a_format: fmt(self.a),
            b_format: fmt(self.b),
            a_major: major(self.trans_a),
            b_major: major(self.trans_b),
            n: self.n,
            m: self.m,
            // m10-rubin-umma: K = 64 of .kind::f8f6f4 (CuTe C++'s make_sm107_instr_desc).
            k_dim: wide,
            // m11-prims-mma-forms: B reuse of the .ws forms.
            max_shift: self.prims.max_shift,
            ..Default::default()
        };
        d::Tcgen05InstrDesc::build(&f).map_or_else(fail, |x| bits(x.bits()))
    }

    /// m11-prims-mma-forms: why this combination of form and modifiers is
    /// no `tcgen05.mma` the PTX ISA defines (or cute-rs models), if it is
    /// not: the `.ws` shapes (`cta_group::1`, M of 32, 64 or 128, N of 64,
    /// 128 or 256, the dense kinds), `.ashift` (A in TMEM, M of 128 or 256,
    /// not with `.collector::a::fill` or `::use`, not `.ws`), `scale-input-d`
    /// (0 to 15, `.kind::f16` and `tf32` only), the collector buffers (B's
    /// 0 to 3, only on `.ws`; A's not on `.ws`), and the block-scaled sparse
    /// kinds (one, two or four scale factors per row: K / 64 for
    /// `.kind::mxf8f6f4` and `mxf4`, K / 32 for `mxf4nvf4`).
    pub fn form_error(&self) -> Option<String> {
        let p = self.prims;
        let dense4 =
            matches!(self.kind, UmmaDataKind::F16 | UmmaDataKind::Tf32 | UmmaDataKind::I8 | UmmaDataKind::F8F6F4);
        if self.sparse
            && let Some(sc) = self.scale
        {
            let vec = match self.kind {
                UmmaDataKind::Mxf4Nvf4 => 32,
                _ => 64,
            };
            if sc.vec as usize != vec || sc.k96 || self.sm107.is_some() {
                return Some(format!(
                    "a sparse .kind::{} MMA scales each {vec} K elements (K = {}), not {}",
                    self.kind.ptx(),
                    self.k(),
                    sc.vec
                ));
            }
        }
        if p.ws {
            if self.two_cta || ![32, 64, 128].contains(&self.m) || ![64, 128, 256].contains(&self.n) {
                return Some(format!(
                    "a weight-stationary MMA is cta_group::1 with M of 32, 64 or 128 and N of 64, 128 or 256, not \
                     M = {} and N = {}{}",
                    self.m,
                    self.n,
                    if self.two_cta { " with cta_group::2" } else { "" }
                ));
            }
            if !dense4 || self.scale.is_some() || self.sm107.is_some() {
                return Some(format!(".ws takes .kind::f16, tf32, f8f6f4 or i8, not .kind::{}", self.kind.ptx()));
            }
            if p.collector_a != CollectorOp::DISCARD || p.ashift {
                return Some(".ws has no .collector::a or .ashift".into());
            }
            if self.sparse && self.kind == UmmaDataKind::Tf32 {
                return Some("cute-rs does not model the 1:2 sparsity of .kind::tf32".into());
            }
            if self.a_tmem && self.m != 128 {
                return Some(format!(
                    "cute-rs models A in tensor memory for .ws with M = 128 only (Layout D), not M = {}",
                    self.m
                ));
            }
        } else {
            if p.collector_b.is_some() || p.max_shift != 0 {
                return Some("B's collector buffers and the max-shift field belong to .ws".into());
            }
        }
        if let Some((i, _)) = p.collector_b
            && i > 3
        {
            return Some(format!("B's collector buffer is b0 to b3, not b{i}"));
        }
        if p.max_shift > 3 {
            return Some(format!("the max-shift field is 2 bits, not {}", p.max_shift));
        }
        if p.ashift {
            if !self.a_tmem || ![128, 256].contains(&self.m) {
                return Some(format!(
                    ".ashift reads A from tensor memory with M of 128 or 256, not M = {}{}",
                    self.m,
                    if self.a_tmem { "" } else { " from shared memory" }
                ));
            }
            if matches!(p.collector_a, CollectorOp::FILL | CollectorOp::USE) {
                return Some(format!(".ashift with .collector::a::{}: only lastuse or discard", p.collector_a.ptx()));
            }
        }
        if let Some(s) = p.scale_input_d {
            if !matches!(self.kind, UmmaDataKind::F16 | UmmaDataKind::Tf32) || self.scale.is_some() {
                return Some(format!("scale-input-d is for .kind::f16 and tf32, not .kind::{}", self.kind.ptx()));
            }
            if s > 15 {
                return Some(format!("scale-input-d is 0 to 15, not {s}"));
            }
        }
        if p.collector_a != CollectorOp::DISCARD && self.sm107.is_some() {
            return Some("the A collector is given twice (prims and the sm_107 form)".into());
        }
        None
    }

    /// m10-rubin-umma: the dense instruction descriptor's K-size bit (29).
    pub const K64_BIT: u32 = 29;

    /// The instruction's name with its qualifiers.
    pub fn ptx(&self) -> String {
        let wide = self.sm107.is_some_and(|r| r.wide_k);
        let vec32 = self.scale.is_some_and(|s| s.vec == 32);
        let scale = match (self.kind, self.scale.is_some_and(|s| s.k96)) {
            // m10-rubin-umma: K = 128 names the vector size (CuTe C++'s SM107_MMA_MXF4NVF4_SS).
            (UmmaDataKind::Mxf4Nvf4, _) if wide && vec32 => ".block_scale.block32",
            (UmmaDataKind::Mxf4Nvf4, _) if wide => ".block_scale.block16",
            (UmmaDataKind::Mxf8f6f4, _) => ".block_scale",
            (UmmaDataKind::Mxf4, false) => ".block_scale.scale_vec::2X",
            (UmmaDataKind::Mxf4, true) => ".block_scale.block32",
            (UmmaDataKind::Mxf4Nvf4, false) => ".block_scale.scale_vec::4X",
            (UmmaDataKind::Mxf4Nvf4, true) => ".block_scale.block16",
            _ => "",
        };
        // m10-rubin-umma: the collector qualifiers (DISCARD, the default, is not written).
        let mut collect = String::new();
        // m11-prims-mma-forms: .ashift, then the collector usage (verified against LLVM 23's
        // NVPTX printer: `.kind::tf32.ashift.collector::a::lastuse`, `.collector::b2::use`).
        if self.prims.ashift {
            collect.push_str(".ashift");
        }
        if self.prims.collector_a != CollectorOp::DISCARD {
            collect.push_str(&format!(".collector::a::{}", self.prims.collector_a.ptx()));
        }
        if let Some((i, op)) = self.prims.collector_b
            && op != CollectorOp::DISCARD
        {
            collect.push_str(&format!(".collector::b{i}::{}", op.ptx()));
        }
        if let Some(r) = self.sm107 {
            for (which, op) in [("a", r.collector_a), ("b", r.collector_b)] {
                if op != CollectorOp::DISCARD {
                    collect.push_str(&format!(".collector::{which}::{}", op.ptx()));
                }
            }
        }
        format!(
            "tcgen05.mma{}{}.cta_group::{}.kind::{}{scale}{collect}",
            if self.prims.ws { ".ws" } else { "" },
            if self.sparse { ".sp" } else { "" },
            if self.two_cta { 2 } else { 1 },
            self.kind.ptx()
        )
    }

    /// The instruction descriptor's negate-A bit (13; m10-umma-kinds).
    pub const NEGATE_A_BIT: u32 = 13;
    /// The instruction descriptor's negate-B bit (14).
    pub const NEGATE_B_BIT: u32 = 14;

    /// The 32-bit words of `{disable-output-lane}`: 4 for `cta_group::1`, 8
    /// for `::2`.
    pub fn disable_lane_words(&self) -> usize {
        if self.two_cta { 8 } else { 4 }
    }
}

/// One `wgmma.mma_async` instruction (roadmap parts m8-wgmma-core and
/// m8-wgmma-variants): shape `m64nNkK`, operands of types `a` and `b`
/// accumulated in `acc`:
///
/// | A, B                          | accumulator | K  |
/// |-------------------------------|-------------|----|
/// | F16, F16 or BF16, BF16        | F16, F32 (BF16: F32) | 16 |
/// | F8E4M3FN or F8E5M2 (each)     | F16, F32    | 32 |
/// | I8 or U8 (each)               | I32         | 32 |
///
/// A and B are transposed (MN-major) when `trans_a` and `trans_b` are set
/// (the PTX `imm-trans-a` and `imm-trans-b`, which only the 16-bit types
/// have). With `a_regs` A comes from registers (the PTX ISA's register
/// form, CuTe DSL's `OperandSource.RMEM`), and is K-major.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WgmmaKind {
    pub n: u32,
    pub a: DType,
    pub b: DType,
    pub acc: DType,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a_regs: bool,
}

impl WgmmaKind {
    /// `[M, N, K]`: `[64, n, 16]` for 16-bit operands, `[64, n, 32]` for
    /// 8-bit ones.
    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        (64, self.n as usize, self.k())
    }

    /// K: 32 bytes of the operand type.
    pub fn k(&self) -> usize {
        256 / self.a.bits()
    }

    /// Accumulators per thread: `N / 2`.
    pub fn accumulators(&self) -> usize {
        self.n as usize / 2
    }

    /// Elements of A per thread in the register form: 8 (16-bit) or 16
    /// (8-bit), four 32-bit registers.
    pub fn a_fragment(&self) -> usize {
        64 * self.k() / 128
    }

    /// Whether the operands are 8-bit integers (the `.s32` accumulator
    /// form, which takes neither scale nor transpose immediates).
    pub fn is_integer(&self) -> bool {
        self.acc == DType::I32
    }

    /// Whether the operands are 16-bit floats (the forms with transpose
    /// immediates).
    pub fn is_16bit(&self) -> bool {
        self.a.bits() == 16
    }

    /// The instruction's PTX name, without operands.
    pub fn ptx(&self) -> String {
        let t = |d: DType| match d {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F8E4M3FN => "e4m3",
            DType::F8E5M2 => "e5m2",
            DType::I8 => "s8",
            DType::U8 => "u8",
            DType::I32 => "s32",
            _ => "f32",
        };
        format!("wgmma.mma_async.sync.aligned.m64n{}k{}.{}.{}.{}", self.n, self.k(), t(self.acc), t(self.a), t(self.b))
    }
}

impl Op {
    /// Values defined by this operation (block arguments of nested regions
    /// excluded).
    pub fn results(&self) -> Vec<ValueId> {
        match self {
            Op::Const { result, .. }
            | Op::Unary { result, .. }
            | Op::Binary { result, .. }
            | Op::Fma { result, .. }
            | Op::FloatInst { result, .. }
            | Op::IntInst { result, .. }
            | Op::Cmp { result, .. }
            | Op::Select { result, .. }
            | Op::Cast { result, .. }
            | Op::Bitcast { result, .. }
            | Op::Special { result, .. }
            | Op::Shuffle { result, .. }
            | Op::Vote { result, .. }
            | Op::Match { result, .. }
            | Op::ElectOne { result, .. }
            | Op::ActiveMask { result }
            | Op::WarpUniform { result, .. }
            | Op::MbarrierWait { result, .. }
            | Op::BarrierRed { result, .. }
            | Op::Atomic { result, .. }
            | Op::Redux { result, .. }
            | Op::Pack { result, .. }
            | Op::HalfInst { result, .. }
            | Op::Cvt { result, .. } => vec![*result],
            Op::Load { results, .. } | Op::LdMatrix { results, .. } | Op::Unpack { results, .. } => results.clone(),
            Op::Mma { d, .. } => d.clone(),
            Op::InlinePtx { outputs, .. } => outputs.clone(),
            Op::If { results, .. } | Op::For { results, .. } | Op::While { results, .. } => results.clone(),
            Op::Call { results, .. } | Op::Multimem { results, .. } => results.clone(),
            Op::Store { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncCommitGroup
            | Op::CpAsyncWaitGroup { .. }
            | Op::Barrier
            | Op::NamedBarrier { .. }
            | Op::WarpBarrier { .. }
            | Op::MbarrierInit { .. }
            | Op::MbarrierArrive { .. }
            | Op::MbarrierExpectTx { .. }
            | Op::FenceMbarrierInit
            | Op::GridDepControl { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::CpAsyncBulk { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::TmaLoad { .. }
            | Op::TmaStore { .. }
            | Op::TmaPrefetch { .. }
            | Op::PrefetchTensormap { .. }
            | Op::TensormapReplace { .. }
            | Op::FenceTensormapRelease
            | Op::FenceTensormapAcquire { .. }
            | Op::TensormapCpFenceproxy { .. }
            | Op::BulkPrefetch { .. }
            | Op::StAsync { .. }
            | Op::FenceProxy { .. }
            | Op::Fence { .. }
            | Op::Printf { .. }
            | Op::Assert { .. }
            | Op::CpAsyncMbarrierArrive { .. }
            | Op::Prefetch { .. }
            | Op::StMatrix { .. }
            | Op::SetMaxNReg { .. }
            | Op::Red { .. } => Vec::new(),
            Op::CarryInst { result, .. } => vec![*result],
            Op::MlirOp { results, .. } => results.clone(),
            Op::Assume { result, .. } | Op::PtrToInt { result, .. } | Op::CvtAddr { result, .. } => vec![*result],
            Op::Wgmma { d, .. } => d.clone(),
            Op::WgmmaFence { results, .. } | Op::WgmmaWait { results, .. } => results.clone(),
            Op::WgmmaCommitGroup
            | Op::Tcgen05 { .. }
            | Op::Tcgen05St { .. }
            | Op::Tcgen05Mma { .. }
            | Op::Tcgen05Commit { .. }
            | Op::Tcgen05Cp { .. }
            | Op::ClcTryCancel { .. } => Vec::new(),
            Op::ClcQueryCancel { results, .. } => results.to_vec(),
            Op::Tcgen05Ld { results, red, sp, .. } => results
                .iter()
                .copied()
                .chain(sp.iter().flat_map(|s| s.mdata.iter().copied()))
                .chain(red.map(|r| r.result))
                .collect(),
            Op::MmaBlockScaled { d, .. } => d.clone(),
            Op::MmaSpBlockScaled { d, .. } => d.clone(),
            Op::MbarrierExt { result, .. } => result.iter().copied().collect(),
            Op::PrimMarker { .. } | Op::Exit | Op::StBulk { .. } | Op::Tcgen05Shift { .. } => Vec::new(),
        }
    }

    /// Values read by this operation itself (not by nested regions).
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Op::Const { .. }
            | Op::Special { .. }
            | Op::ActiveMask { .. }
            | Op::Fence { .. }
            | Op::FenceProxy { .. }
            | Op::CpAsyncBulkCommitGroup
            | Op::CpAsyncBulkWaitGroup { .. }
            | Op::ClusterArrive { .. }
            | Op::ClusterWait { .. }
            | Op::GridDepControl { .. }
            | Op::FenceMbarrierInit => {
                vec![]
            }
            Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::WarpUniform { value: x, .. } => {
                vec![*x]
            }
            Op::Binary { lhs, rhs, .. } | Op::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Fma { a, b, c, .. } => vec![*a, *b, *c],
            Op::FloatInst { args, .. } | Op::IntInst { args, .. } => args.clone(),
            Op::Select { cond, on_true, on_false, .. } => vec![*cond, *on_true, *on_false],
            Op::Load { index, attrs, pred, pass_thru, .. } => {
                let mut v = vec![*index];
                v.extend(attrs.cache_policy);
                v.extend(attrs.peer);
                v.extend(pred);
                v.extend(pass_thru);
                v
            }
            Op::LdMatrix { index, .. } => vec![*index],
            Op::Store { values, index, attrs, pred, .. } => {
                let mut v = values.clone();
                v.push(*index);
                v.extend(attrs.cache_policy);
                v.extend(attrs.peer);
                v.extend(pred);
                v
            }
            Op::CpAsync { src_index, dst_index, pred, .. } => {
                let mut v = vec![*src_index, *dst_index];
                v.extend(pred);
                v
            }
            Op::CpAsyncCommitGroup | Op::CpAsyncWaitGroup { .. } | Op::Barrier | Op::SetMaxNReg { .. } => vec![],
            Op::StMatrix { values, index, .. } => {
                let mut v = values.clone();
                v.push(*index);
                v
            }
            Op::NamedBarrier { id, threads, .. } => {
                let mut v = vec![*id];
                v.extend(threads);
                v
            }
            Op::WarpBarrier { mask } | Op::ElectOne { mask, .. } => vec![*mask],
            Op::Shuffle { value, offset, mask, clamp, .. } => vec![*value, *offset, *mask, *clamp],
            Op::Vote { pred, mask, .. } => vec![*pred, *mask],
            Op::Match { value, mask, .. } | Op::Redux { value, mask, .. } => vec![*value, *mask],
            Op::Atomic { index, value, compare, peer, .. } => {
                let mut v = vec![*index, *value];
                v.extend(compare);
                v.extend(peer);
                v
            }
            Op::Red { index, value, peer, .. } => {
                let mut v = vec![*index, *value];
                v.extend(peer);
                v
            }
            Op::MbarrierInit { index, count, .. } => vec![*index, *count],
            Op::MbarrierArrive { index, count, expect_tx, peer, .. } => {
                let mut v = vec![*index];
                v.extend(count);
                v.extend(expect_tx);
                v.extend(peer);
                v
            }
            Op::MbarrierExpectTx { index, bytes, peer, .. } => {
                let mut v = vec![*index, *bytes];
                v.extend(peer);
                v
            }
            Op::MbarrierWait { index, parity, time_limit, .. } => {
                let mut v = vec![*index, *parity];
                v.extend(time_limit);
                v
            }
            Op::BarrierRed { pred, id, threads, .. } => {
                let mut v = vec![*pred, *id];
                v.extend(threads);
                v
            }
            Op::MbarrierExt { index, value, .. } => {
                let mut v = vec![*index];
                v.extend(value);
                v
            }
            Op::PrimMarker { marker: PrimMarker::Nanosleep(t) } => vec![*t],
            Op::PrimMarker { .. } | Op::Exit => vec![],
            Op::StBulk { index, size, .. } => vec![*index, *size],
            Op::Tcgen05Shift { taddr, .. } => vec![*taddr],
            Op::CpAsyncBulk {
                src_index,
                dst_index,
                bytes,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
                ..
            } => {
                let mut v = vec![*src_index, *dst_index, *bytes];
                if let BulkCompletion::Mbarrier { index, .. } = completion {
                    v.push(*index);
                }
                v.extend(cache_policy);
                v.extend(mcast_mask);
                v.extend(peer);
                v.extend(byte_mask);
                v
            }
            Op::TmaLoad { dst_index, desc, coords, mbar_index, cache_policy, mcast_mask, mode, overrides, .. } => {
                let mut v = vec![*dst_index, *desc];
                v.extend(coords);
                v.push(*mbar_index);
                v.extend(cache_policy);
                v.extend(mcast_mask);
                v.extend(mode.offsets());
                v.extend(overrides.iter().flat_map(TmaOverride::operands));
                v
            }
            Op::TmaStore { src_index, desc, coords, cache_policy, overrides, .. } => {
                let mut v = vec![*src_index, *desc];
                v.extend(coords);
                v.extend(cache_policy);
                v.extend(overrides.iter().flat_map(TmaOverride::operands));
                v
            }
            Op::TmaPrefetch { desc, coords, mode } => {
                let mut v = vec![*desc];
                v.extend(coords);
                v.extend(mode.offsets());
                v
            }
            Op::PrefetchTensormap { desc } => vec![*desc],
            Op::TensormapReplace { addr, value, .. } => vec![*addr, *value],
            Op::FenceTensormapRelease => vec![],
            Op::FenceTensormapAcquire { addr } => vec![*addr],
            Op::TensormapCpFenceproxy { dst, src } => vec![*dst, *src],
            Op::BulkPrefetch { index, bytes, .. } => vec![*index, *bytes],
            Op::StAsync { index, values, mbar_index, peer, .. } => {
                let mut v = vec![*index];
                v.extend(values);
                v.push(*mbar_index);
                v.push(*peer);
                v
            }
            Op::Printf { args, .. } => args.clone(),
            Op::Assert { cond, .. } => vec![*cond],
            Op::InlinePtx { inputs, .. } => inputs.iter().map(|i| i.value()).collect(),
            Op::Call { args, .. } => args.iter().map(|i| i.value()).collect(),
            Op::Multimem { index, values, .. } => std::iter::once(*index).chain(values.iter().copied()).collect(),
            Op::Mma { a, b, c, e, .. } => a.iter().chain(b).chain(c).chain(e).copied().collect(),
            Op::If { cond, .. } => vec![*cond],
            Op::For { lower, upper, step, init, .. } => {
                let mut v = vec![*lower, *upper, *step];
                v.extend(init);
                v
            }
            Op::While { init, .. } => init.clone(),
            Op::CarryInst { args, .. } => args.clone(),
            Op::MlirOp { operands, .. } => operands.iter().map(|i| i.value()).collect(),
            Op::CpAsyncMbarrierArrive { index, .. } => vec![*index],
            Op::Prefetch { index, pred, .. } => {
                let mut v = vec![*index];
                v.extend(pred);
                v
            }
            Op::Pack { parts, .. } => parts.clone(),
            Op::Unpack { x, .. } => vec![*x],
            Op::HalfInst { args, predicate, .. } => {
                let mut v = args.clone();
                v.extend(predicate);
                v
            }
            Op::Cvt { args, .. } => args.clone(),
            Op::Assume { x, .. } | Op::CvtAddr { x, .. } => vec![*x],
            Op::PtrToInt { index, .. } => vec![*index],
            Op::Wgmma { c, a, desc_b, scale_d, .. } => {
                let mut v = c.clone();
                v.extend(a);
                v.extend([*desc_b, *scale_d]);
                v
            }
            Op::WgmmaFence { inputs, .. } | Op::WgmmaWait { inputs, .. } => inputs.clone(),
            Op::WgmmaCommitGroup => vec![],
            Op::Tcgen05 { args, .. } => args.clone(),
            Op::Tcgen05Ld { taddr, .. } => vec![*taddr],
            Op::Tcgen05St { values, taddr, .. } => {
                let mut v = vec![*taddr];
                v.extend(values);
                v
            }
            Op::Tcgen05Mma {
                taddr,
                desc_a,
                desc_b,
                enable_input_d,
                elect,
                idesc,
                disable_lane,
                sf,
                meta,
                zero_col_mask,
                ..
            } => {
                let mut v = vec![*taddr, *desc_a, *desc_b, *enable_input_d, *elect];
                v.extend(idesc);
                v.extend(disable_lane);
                if let Some((a, b)) = sf {
                    v.extend([*a, *b]);
                }
                v.extend(meta);
                v.extend(zero_col_mask);
                v
            }
            Op::MmaSpBlockScaled { a, b, c, e, sfa, sfb, .. } => {
                a.iter().chain(b).chain(c).copied().chain([*e, *sfa, *sfb]).collect()
            }
            Op::MmaBlockScaled { a, b, c, sfa, sfb, .. } => {
                a.iter().chain(b).chain(c).copied().chain([*sfa, *sfb]).collect()
            }
            Op::Tcgen05Commit { index, mask, .. } => {
                let mut v = vec![*index];
                v.extend(mask);
                v
            }
            Op::Tcgen05Cp { taddr, desc, elect, .. } => vec![*taddr, *desc, *elect],
            Op::ClcTryCancel { index, mbar_index, .. } => vec![*index, *mbar_index],
            Op::ClcQueryCancel { index, .. } => vec![*index],
        }
    }

    /// Nested regions.
    pub fn regions(&self) -> Vec<&Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            Op::Call { body: Some(body), .. } => vec![body],
            _ => vec![],
        }
    }

    /// Nested regions, mutably.
    pub fn regions_mut(&mut self) -> Vec<&mut Region> {
        match self {
            Op::If { then_region, else_region, .. } => vec![then_region, else_region],
            Op::For { body, .. } => vec![body],
            Op::While { cond, body, .. } => vec![cond, body],
            Op::Call { body: Some(body), .. } => vec![body],
            _ => vec![],
        }
    }

    /// Pure operations can be removed when their results are unused.
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            Op::Const { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Fma { .. }
                | Op::FloatInst { .. }
                | Op::IntInst { .. }
                | Op::Cmp { .. }
                | Op::Select { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Special { .. }
                | Op::WarpUniform { .. }
                | Op::InlinePtx { side_effects: false, .. }
                | Op::MlirOp { side_effects: false, .. }
                | Op::Pack { .. }
                | Op::Unpack { .. }
                | Op::HalfInst { .. }
                | Op::Cvt { .. }
                | Op::Assume { .. }
                | Op::PtrToInt { .. }
                | Op::CvtAddr { .. }
        ) && !matches!(self, Op::InlinePtx { constraints, .. } if asm_clobbers_memory(constraints))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Const { .. } => "const",
            Op::Unary { .. } => "unary",
            Op::Binary { .. } => "binary",
            Op::Fma { .. } => "fma",
            Op::FloatInst { .. } => "float instruction",
            Op::IntInst { .. } => "integer instruction",
            Op::Cmp { .. } => "cmp",
            Op::Select { .. } => "select",
            Op::Cast { .. } => "cast",
            Op::Bitcast { .. } => "bitcast",
            Op::Special { .. } => "special register",
            Op::Load { .. } => "load",
            Op::Store { .. } => "store",
            Op::CpAsync { .. } => "cp.async",
            Op::CpAsyncCommitGroup => "cp.async.commit_group",
            Op::CpAsyncWaitGroup { .. } => "cp.async.wait_group",
            Op::LdMatrix { .. } => "ldmatrix",
            Op::StMatrix { .. } => "stmatrix",
            Op::SetMaxNReg { increase: true, .. } => "setmaxnreg.inc",
            Op::SetMaxNReg { increase: false, .. } => "setmaxnreg.dec",
            Op::Mma { kind, .. } if kind.is_sparse() => "mma.sp.sync",
            Op::Mma { .. } => "mma.sync",
            Op::Barrier => "barrier",
            Op::NamedBarrier { kind: BarrierKind::Sync, .. } => "bar.sync",
            Op::NamedBarrier { kind: BarrierKind::Arrive, .. } => "bar.arrive",
            Op::NamedBarrier { kind: BarrierKind::ArriveUnaligned, .. } => "barrier.arrive",
            Op::WarpBarrier { .. } => "bar.warp.sync",
            Op::Shuffle { .. } => "shfl.sync",
            Op::Vote { .. } => "vote.sync",
            Op::Match { kind: MatchKind::Any, .. } => "match.any.sync",
            Op::Match { kind: MatchKind::All, .. } => "match.all.sync",
            Op::ElectOne { .. } => "elect.sync",
            Op::ActiveMask { .. } => "activemask",
            Op::WarpUniform { .. } => "make_warp_uniform",
            Op::Atomic { .. } => "atomic",
            Op::Red { .. } => "red",
            Op::MbarrierInit { .. } => "mbarrier.init",
            Op::MbarrierArrive { expect_tx: Some(_), .. } => "mbarrier.arrive.expect_tx",
            Op::MbarrierArrive { .. } => "mbarrier.arrive",
            Op::MbarrierExpectTx { .. } => "mbarrier.expect_tx",
            Op::MbarrierWait { kind: MbarrierWaitKind::Try, .. } => "mbarrier.try_wait",
            Op::MbarrierWait { kind: MbarrierWaitKind::Test, .. } => "mbarrier.test_wait",
            Op::FenceMbarrierInit => "fence.mbarrier_init",
            Op::GridDepControl { launch_dependents: false } => "griddepcontrol.wait",
            Op::GridDepControl { launch_dependents: true } => "griddepcontrol.launch_dependents",
            Op::ClusterArrive { relaxed: false, .. } => "barrier.cluster.arrive",
            Op::ClusterArrive { relaxed: true, .. } => "barrier.cluster.arrive.relaxed",
            Op::ClusterWait { .. } => "barrier.cluster.wait",
            Op::CpAsyncBulk { .. } => "cp.async.bulk",
            Op::TmaLoad { .. } => "cp.async.bulk.tensor",
            Op::TmaStore { reduce: None, .. } => "cp.async.bulk.tensor (store)",
            Op::TmaStore { reduce: Some(_), .. } => "cp.reduce.async.bulk.tensor",
            Op::TmaPrefetch { .. } => "cp.async.bulk.prefetch.tensor",
            Op::PrefetchTensormap { .. } => "prefetch.tensormap",
            Op::TensormapReplace { .. } => "tensormap.replace",
            Op::FenceTensormapRelease => "fence.proxy.tensormap::generic.release",
            Op::FenceTensormapAcquire { .. } => "fence.proxy.tensormap::generic.acquire",
            Op::TensormapCpFenceproxy { .. } => "tensormap.cp_fenceproxy",
            Op::BulkPrefetch { .. } => "cp.async.bulk.prefetch",
            Op::StAsync { .. } => "st.async",
            Op::CpAsyncBulkCommitGroup => "cp.async.bulk.commit_group",
            Op::CpAsyncBulkWaitGroup { read: false, .. } => "cp.async.bulk.wait_group",
            Op::CpAsyncBulkWaitGroup { read: true, .. } => "cp.async.bulk.wait_group.read",
            Op::FenceProxy { kind } => kind.ptx(),
            Op::Fence { .. } => "fence",
            Op::Printf { .. } => "printf",
            Op::Assert { .. } => "assert",
            Op::InlinePtx { .. } => "inline ptx",
            Op::Call { .. } => "call",
            Op::Multimem { .. } => "multimem",
            Op::If { .. } => "if",
            Op::For { .. } => "for",
            Op::While { .. } => "while",
            Op::Redux { .. } => "redux.sync",
            Op::CarryInst { .. } => "carry instruction",
            Op::MlirOp { .. } => "mlir operation",
            Op::CpAsyncMbarrierArrive { noinc: true, .. } => "cp.async.mbarrier.arrive.noinc",
            Op::CpAsyncMbarrierArrive { noinc: false, .. } => "cp.async.mbarrier.arrive",
            Op::Prefetch { uniform: true, .. } => "prefetchu",
            Op::Prefetch { .. } => "prefetch",
            Op::Pack { .. } => "pack",
            Op::Unpack { .. } => "unpack",
            Op::HalfInst { .. } => "half-precision instruction",
            Op::Cvt { .. } => "cvt",
            Op::Assume { .. } => "assume",
            Op::PtrToInt { .. } => "ptrtoint",
            Op::CvtAddr { .. } => "cvta",
            Op::Wgmma { .. } => "wgmma.mma_async",
            Op::WgmmaFence { .. } => "wgmma.fence",
            Op::WgmmaCommitGroup => "wgmma.commit_group",
            Op::WgmmaWait { .. } => "wgmma.wait_group",
            Op::Tcgen05 { inst, .. } => inst.name(),
            Op::Tcgen05Ld { .. } => "tcgen05.ld",
            Op::Tcgen05St { .. } => "tcgen05.st",
            Op::Tcgen05Mma { .. } => "tcgen05.mma",
            Op::MmaBlockScaled { .. } => "mma.block_scale",
            Op::MmaSpBlockScaled { .. } => "mma.sp.block_scale",
            Op::Tcgen05Commit { .. } => "tcgen05.commit",
            Op::Tcgen05Cp { .. } => "tcgen05.cp",
            Op::ClcTryCancel { .. } => "clusterlaunchcontrol.try_cancel",
            Op::ClcQueryCancel { .. } => "clusterlaunchcontrol.query_cancel",
            Op::BarrierRed { .. } => "barrier.cta.red",
            Op::MbarrierExt { kind, .. } => kind.ptx(),
            Op::PrimMarker { marker: PrimMarker::Nanosleep(_) } => "nanosleep",
            Op::PrimMarker { marker: PrimMarker::Pmevent(_) } => "pmevent",
            Op::PrimMarker { marker: PrimMarker::Breakpoint } => "brkpt",
            Op::PrimMarker { marker: PrimMarker::Comment(_) } => "comment",
            Op::Exit => "exit",
            Op::StBulk { .. } => "st.bulk",
            Op::Tcgen05Shift { .. } => "tcgen05.shift",
        }
    }
}

/// An instruction: an operation plus the user source location that built it.
#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Op,
    pub loc: Loc,
    /// The profiling [`Scope`](crate::Scope) the instruction was built in.
    pub scope: Option<crate::profiling::ScopeId>,
}

/// A straight-line list of instructions, terminated by the values it yields
/// to its parent (`If` results or the next `For` iteration).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub insts: Vec<Inst>,
    pub yields: Vec<ValueId>,
}

#[derive(Clone, Debug)]
pub struct BufferDecl {
    pub name: String,
    pub elem: DType,
    /// Number of elements, or `None` for a buffer sized at launch: a global
    /// buffer, or the dynamic shared-memory region ([`get_dyn_smem`]), whose
    /// bytes [`LaunchConfig::dynamic_smem_bytes`] sets.
    ///
    /// [`get_dyn_smem`]: crate::get_dyn_smem
    /// [`LaunchConfig::dynamic_smem_bytes`]: crate::LaunchConfig::dynamic_smem_bytes
    pub len: Option<usize>,
    pub space: MemSpace,
    /// Byte alignment of the allocation (shared memory) or the alignment the
    /// kernel assumes for the pointer it is given (global memory).
    pub align: usize,
    /// For a shared-memory struct ([`cute_struct!`](crate::cute_struct)),
    /// the struct's name and its members as `name:size:offset` (CuTe DSL's
    /// field table); empty otherwise. [`Kernel::dump`](crate::Kernel::dump)
    /// shows it.
    pub fields: Vec<String>,
    /// The shared-memory partition the buffer belongs to (CuTe DSL's
    /// `smem.partition_id`): [`SmemPartition::Reserved`] buffers are laid
    /// out before every [`SmemPartition::User`] one. Other memory spaces
    /// ignore it.
    pub partition: SmemPartition,
}

/// The partition of shared memory a buffer is allocated in (CuTe DSL's
/// `cutlass.memory.SmemPartition`, an `IntEnum` whose values these keep).
/// [`KernelIr::shared_memory_layout`] places every reserved buffer first,
/// in declaration order, so reserved storage (pipeline mbarriers,
/// [`pipeline::alloc_reserved_mbarrier`](crate::pipeline::alloc_reserved_mbarrier))
/// keeps the lowest shared-memory addresses whatever is allocated around
/// it; the user buffers follow. (A kernel whose shared memory is not one
/// window emits each buffer as its own `.shared` variable, placed by
/// `ptxas`.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SmemPartition {
    /// Storage the library reserves for itself, at the lowest addresses.
    Reserved = 0,
    /// Ordinary allocations (the default).
    #[default]
    User = 1,
}

/// The symbol of the dynamic shared-memory window in the emitted MLIR and
/// PTX: an external zero-length array (`memref.global` without a value,
/// PTX's `.extern .shared .align 16 .b8 __cute_dynamic_smem[];`) that the
/// driver places at the start of the block's dynamic shared memory. It is
/// not `gpu.dynamic_shared_memory`, which MLIR lowers to an internal global
/// that LLVM's NVPTX backend demotes to one byte of static shared memory in
/// the kernel: that byte counts against the 48 KB and opt-in limits and in
/// `%total_smem_size`. No kernel may have this name.
pub const DYNAMIC_SMEM_SYMBOL: &str = "__cute_dynamic_smem";

/// Where one block's shared memory lives ([`KernelIr::shared_memory_layout`]).
///
/// Buffers sized at build time are static shared memory (`.shared`
/// variables) up to [`MAX_STATIC_SHARED_BYTES`] in all. Above that, or when
/// the kernel has a dynamic region ([`get_dyn_smem`]), every shared buffer
/// lives in the dynamic shared-memory window that the launch sizes
/// ([`DYNAMIC_SMEM_SYMBOL`], PTX's `.extern .shared`): the build-time
/// buffers first, at their starts, then the dynamic region, at
/// `dynamic_start`. A launch passes `dynamic_start` plus the region's bytes
/// as its dynamic shared memory, which is what `%dynamic_smem_size` reads;
/// such a kernel has no static shared memory.
///
/// [`MAX_STATIC_SHARED_BYTES`]: crate::validate::MAX_STATIC_SHARED_BYTES
/// [`get_dyn_smem`]: crate::get_dyn_smem
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedMemoryLayout {
    /// The start of each shared buffer in bytes, by buffer (`None` for the
    /// other memory spaces). The dynamic region starts at `dynamic_start`.
    pub starts: Vec<Option<usize>>,
    /// The bytes the buffers sized at build time span.
    pub static_bytes: usize,
    /// Whether the shared buffers live in the dynamic shared-memory window
    /// rather than in static shared memory.
    pub in_window: bool,
    /// Where the dynamic region starts in the window: after the build-time
    /// buffers when they are in the window (at a multiple of the region's
    /// alignment, if the kernel has a region), 0 otherwise.
    pub dynamic_start: usize,
    /// The alignment the window declares (`.extern .shared .align`): the
    /// largest of its buffers' alignments, and at least 16, so that each
    /// buffer's start in the window is aligned in shared memory too.
    pub window_align: usize,
    /// The bytes the [`SmemPartition::Reserved`] buffers span from offset
    /// 0: where the last of them ends (0 without any). The user buffers
    /// take the rest of `static_bytes`.
    pub reserved_bytes: usize,
}

impl SharedMemoryLayout {
    /// The bytes of `partition`, as CuTe DSL's
    /// `SmemAllocator.calculate_partition_size(partition, cumulative)`
    /// reports them: [`reserved_bytes`](Self::reserved_bytes) for the
    /// reserved partition, `static_bytes - reserved_bytes` for the user one
    /// (alignment padding in front of the first user buffer included);
    /// cumulatively, the reserved partition is the same and the user one is
    /// every build-time byte of the block (`static_bytes`, CuTe DSL's
    /// `calculate_total_usage`). The dynamic region
    /// ([`get_dyn_smem`](crate::get_dyn_smem)) is sized at launch and not
    /// counted.
    pub fn partition_bytes(&self, partition: SmemPartition, cumulative: bool) -> usize {
        match (partition, cumulative) {
            (SmemPartition::Reserved, _) => self.reserved_bytes,
            (SmemPartition::User, false) => self.static_bytes - self.reserved_bytes,
            (SmemPartition::User, true) => self.static_bytes,
        }
    }

    /// The dynamic shared memory a launch passes to the driver
    /// (`sharedMemBytes`, read back by `%dynamic_smem_size`) when the
    /// dynamic region gets `dynamic` bytes.
    pub fn launch_bytes(&self, dynamic: usize) -> usize {
        self.dynamic_start + dynamic
    }

    /// The shared memory one block uses, static and dynamic, when the
    /// dynamic region gets `dynamic` bytes.
    pub fn block_bytes(&self, dynamic: usize) -> usize {
        if self.in_window { self.launch_bytes(dynamic) } else { self.static_bytes + dynamic }
    }
}

#[derive(Clone, Debug)]
pub enum Param {
    Buffer(BufferId),
    /// `multiple_of` is what the kernel assumes the launch value divides by.
    Scalar {
        name: String,
        value: ValueId,
        multiple_of: usize,
    },
    /// A by-value parameter (a C struct or a 128-byte TMA descriptor): `buffer` holds its
    /// bytes, read-only, as the kernel's `.param` aggregate. With `grid_constant`, the
    /// kernel may take its address without a local copy (`nvvm.grid_constant`).
    ByValue {
        buffer: BufferId,
        grid_constant: bool,
    },
}

/// A module-scope global (`__device__` or `__constant__`, CuTe DSL's module-level
/// `cutlass.Array`), declared with [`module_global`](crate::module_global).
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleGlobal {
    /// The buffer that holds it: [`MemSpace::Global`] or [`MemSpace::Constant`], named by
    /// its C identifier (the symbol `cuModuleGetGlobal` finds).
    pub buffer: BufferId,
    /// Initial bytes (little-endian), or `None` for zeros.
    pub init: Option<Vec<u8>>,
    /// An LLVM `constant`: the kernel may fold its loads, and host writes are refused.
    pub readonly: bool,
}

#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub args: Vec<Param>,
    /// The parameters that hold tensor maps (by-value grid constants): hidden ones,
    /// which come last in `args` and are computed at launch, and ones the host binds.
    pub tensor_maps: Vec<crate::tma::TensorMapParam>,
    pub buffers: Vec<BufferDecl>,
    /// Type of every SSA value, indexed by `ValueId`.
    pub value_types: Vec<DType>,
    pub body: Region,
    /// Block size the kernel was written for, if declared.
    pub block_dim: Option<[u32; 3]>,
    /// The largest block the kernel may be launched with ([`KernelOptions::max_threads`],
    /// PTX's `.maxntid`).
    ///
    /// [`KernelOptions::max_threads`]: crate::KernelOptions::max_threads
    pub max_threads: Option<[u32; 3]>,
    /// Blocks per SM the kernel asks the compiler to fit (`.minnctapersm`).
    pub min_blocks_per_sm: Option<u32>,
    /// Module-scope globals.
    pub globals: Vec<ModuleGlobal>,
    /// Shared-memory overlays ([`smem_overlay`](crate::smem_overlay)): per group, the
    /// buffers of each alternative, which all start at the group's offset.
    pub overlays: Vec<Vec<Vec<BufferId>>>,
    /// Where the kernel was built (the `Kernel::build` call): the kernel
    /// function's own source location.
    pub loc: Loc,
    // m11-extern-distributed.
    /// External device functions the kernel calls ([`Op::Call`]), one per symbol.
    pub externs: Vec<ExternDecl>,
    /// LLVM bitcode files linked into the module (CuTe DSL's `BitCode` sources and
    /// `LinkLibraries`), in first-use order.
    pub link_libraries: Vec<std::path::PathBuf>,
    /// Set for a standalone device function ([`DeviceFunction`](crate::ffi::DeviceFunction)):
    /// the IR is then a `__device__` function with real parameters and a return value.
    pub function: Option<DeviceFunctionSig>,
    /// The multicast buffers ([`param_tensor_multicast`](crate::utils::distributed::param_tensor_multicast)):
    /// parameters holding multimem addresses, which only `multimem` instructions access.
    pub multicast: Vec<BufferId>,
    /// The profiling scopes ([`Scope`](crate::Scope)) instructions refer to by
    /// [`Inst::scope`].
    pub scopes: Vec<crate::profiling::NameScope>,
    /// The device workspace the kernel takes as its last argument, if any
    /// ([`crate::experimental::host_runtime`], roadmap part m11-cute-ext-tma).
    pub workspace: Option<crate::experimental::host_runtime::WorkspaceDecl>,
    /// The Float `+`, `-` and `*` results that must not be contracted into `fma` (built in
    /// an exact scope, such as the precise [`math`](crate::math) functions'), whatever
    /// [`CompileOptions::fp_contract`](crate::CompileOptions::fp_contract) says.
    pub exact_float: std::collections::BTreeSet<ValueId>,
}

impl KernelIr {
    /// The path of scope `id`, outermost name first, joined with `/`.
    pub fn scope_path(&self, id: crate::profiling::ScopeId) -> String {
        let mut names = Vec::new();
        let mut at = Some(id);
        while let Some(s) = at {
            let scope = &self.scopes[s.0 as usize];
            names.push(scope.name.as_str());
            at = scope.parent;
        }
        names.reverse();
        names.join("/")
    }

    /// The scope path of the first scoped instruction built at `loc` (what simulator
    /// diagnostics print), or `None` when the kernel has no scopes or none there.
    pub fn scope_path_at(&self, loc: Loc) -> Option<String> {
        fn find(r: &Region, loc: Loc) -> Option<crate::profiling::ScopeId> {
            r.insts.iter().find_map(|i| {
                let here = (std::ptr::eq(i.loc, loc) || i.loc == loc).then_some(i.scope).flatten();
                here.or_else(|| i.op.regions().into_iter().find_map(|sub| find(sub, loc)))
            })
        }
        if self.scopes.is_empty() {
            return None;
        }
        find(&self.body, loc).map(|id| self.scope_path(id))
    }

    /// Whether `b` is a multicast buffer.
    pub fn is_multicast(&self, b: BufferId) -> bool {
        self.multicast.contains(&b)
    }

    /// The declaration of external function `id`.
    pub fn extern_decl(&self, id: ExternId) -> &ExternDecl {
        &self.externs[id.0 as usize]
    }

    /// The number of hidden parameters (the last ones), which launches do not pass.
    pub fn hidden_param_count(&self) -> usize {
        self.tensor_maps.iter().filter(|t| t.source.is_some()).count()
    }

    pub fn dtype(&self, v: ValueId) -> DType {
        self.value_types[v.0 as usize]
    }

    pub fn buffer(&self, b: BufferId) -> &BufferDecl {
        &self.buffers[b.0 as usize]
    }

    /// Where one block's shared memory lives: each buffer's start, the bytes
    /// of the buffers sized at build time, and where the dynamic region
    /// ([`get_dyn_smem`](crate::get_dyn_smem)) starts. The MLIR emitter, the
    /// CUDA launcher and the simulator all place shared memory by it.
    pub fn shared_memory_layout(&self) -> SharedMemoryLayout {
        let mut total = 0usize;
        let mut reserved_bytes = 0usize;
        let mut starts: Vec<Option<usize>> = vec![None; self.buffers.len()];
        // The reserved partition first, in declaration order, then the user one.
        let reserved = |k: &usize| self.buffers[*k].partition == SmemPartition::Reserved;
        let order: Vec<usize> =
            (0..self.buffers.len()).filter(reserved).chain((0..self.buffers.len()).filter(|k| !reserved(k))).collect();
        for k in order {
            let b = &self.buffers[k];
            if b.space != MemSpace::Shared || b.len.is_none() || starts[k].is_some() {
                continue;
            }
            // An overlay group is placed at its first buffer: every alternative starts at the
            // group's start (aligned for all of them), and the group spans the largest.
            let group = self.overlays.iter().find(|g| g.iter().flatten().any(|m| m.0 as usize == k));
            let members: Vec<usize> = match group {
                Some(g) => g.iter().flatten().map(|m| m.0 as usize).collect(),
                None => vec![k],
            };
            let align = members.iter().map(|&m| self.buffers[m].align).max().unwrap_or(1);
            let start = total.next_multiple_of(align);
            let mut end = start;
            for &m in &members {
                let d = &self.buffers[m];
                starts[m] = Some(start);
                end = end.max(start + d.len.unwrap_or(0) * d.elem.bytes());
            }
            total = end;
            if b.partition == SmemPartition::Reserved {
                reserved_bytes = end;
            }
        }
        let region = self.dynamic_smem_region();
        // Overlaid buffers share bytes, which only the window's byte views express.
        let in_window =
            region.is_some() || total > crate::validate::MAX_STATIC_SHARED_BYTES || !self.overlays.is_empty();
        let dynamic_start = match region {
            Some(r) => total.next_multiple_of(self.buffer(r).align),
            None if in_window => total,
            None => 0,
        };
        if let Some(r) = region {
            starts[r.0 as usize] = Some(dynamic_start);
        }
        let window_align =
            self.buffers.iter().filter(|b| b.space == MemSpace::Shared).map(|b| b.align).fold(16, usize::max);
        SharedMemoryLayout { starts, static_bytes: total, in_window, dynamic_start, window_align, reserved_bytes }
    }

    /// Bytes of the shared buffers of one block sized at build time (the
    /// dynamic region's bytes are set at launch).
    pub fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_layout().static_bytes
    }

    /// The dynamic shared-memory region ([`get_dyn_smem`](crate::get_dyn_smem)),
    /// if the kernel has one: the shared buffer sized at launch.
    pub fn dynamic_smem_region(&self) -> Option<BufferId> {
        self.buffers.iter().position(|b| b.space == MemSpace::Shared && b.len.is_none()).map(|i| BufferId(i as u32))
    }

    /// Count instructions (recursively) matching `pred`.
    pub fn count_ops(&self, pred: impl Fn(&Op) -> bool + Copy) -> usize {
        fn walk(r: &Region, pred: impl Fn(&Op) -> bool + Copy) -> usize {
            r.insts
                .iter()
                .map(|i| {
                    let nested: usize = i.op.regions().into_iter().map(|r| walk(r, pred)).sum();
                    nested + pred(&i.op) as usize
                })
                .sum()
        }
        walk(&self.body, pred)
    }
}

// ---------------------------------------------------------------------------
// m11-extern-distributed: external device functions, standalone device
// functions and multimem instructions.
// ---------------------------------------------------------------------------

/// An external device function of a kernel ([`KernelIr::externs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternId(pub u32);

/// A parameter of an external device function: a value of a numeric type,
/// or a pointer (`!llvm.ptr<n>`) into an address space (generic, global or
/// shared).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternParam {
    Value(DType),
    Ptr(crate::tensor::AddressSpace),
}

impl ExternParam {
    /// The MLIR type of the parameter: `f32`, `i32`, `!llvm.ptr<1>`, ...
    /// (`!llvm.ptr` for a generic pointer).
    pub fn mlir(&self) -> String {
        match self {
            ExternParam::Value(t) => t.mlir().to_string(),
            ExternParam::Ptr(crate::tensor::AddressSpace::Generic) => "!llvm.ptr".to_string(),
            ExternParam::Ptr(space) => format!("!llvm.ptr<{}>", *space as u32),
        }
    }
}

/// The declaration of an external device function: its symbol, signature,
/// how calls to it are compiled and how the simulator runs them.
#[derive(Clone, Debug)]
pub struct ExternDecl {
    /// The symbol (the mangled name of an overloaded function).
    pub symbol: String,
    pub params: Vec<ExternParam>,
    /// At most one result.
    pub results: Vec<DType>,
    /// Calls may be inlined (CuTe DSL's `inline=True`, the default); without
    /// it every call is `func.call ... {no_inline}`.
    pub inline: bool,
    /// The prototype carries LLVM's `convergent` (the default), so that LLVM
    /// neither duplicates a call nor moves it into divergent code: the callee
    /// may hold barriers or warp collectives.
    pub convergent: bool,
    /// What the simulator runs for a call when it has no traced body: raw
    /// bits of the value arguments (0 for a pointer) in, the result's bits
    /// out.
    pub sim: Option<AsmSim>,
    /// Where it was first declared.
    pub loc: Loc,
}

impl ExternDecl {
    /// Whether `other` declares the same prototype (symbol, parameters and
    /// results); how calls are compiled and simulated may differ.
    pub fn same_prototype(&self, other: &ExternDecl) -> bool {
        self.symbol == other.symbol && self.params == other.params && self.results == other.results
    }

    /// The prototype as MLIR: `func.func private @sym(f32) -> f32`.
    pub fn prototype(&self) -> String {
        let params: Vec<String> = self.params.iter().map(ExternParam::mlir).collect();
        let results = match self.results.as_slice() {
            [] => String::new(),
            [t] => format!(" -> {}", t.mlir()),
            ts => format!(" -> ({})", ts.iter().map(|t| t.mlir()).collect::<Vec<_>>().join(", ")),
        };
        let attrs = if self.convergent { " attributes {passthrough = [\"convergent\"]}" } else { "" };
        format!("func.func private @{}({}){results}{attrs}", self.symbol, params.join(", "))
    }
}

/// The element type of a `multimem.ld_reduce` (CuTe DSL's typed variants):
/// packed pairs of Float16 or BFloat16 (accumulated in Float32), Float32
/// (accumulated in Float32), and packed quadruples of FP8 (accumulated in
/// Float16), each in one 32-bit register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultimemType {
    F16x2,
    Bf16x2,
    F32,
    E4m3x4,
    E5m2x4,
}

impl MultimemType {
    /// The PTX type: `f16x2`, `bf16x2`, `f32`, `e4m3x4`, `e5m2x4`.
    pub fn ptx(self) -> &'static str {
        match self {
            MultimemType::F16x2 => "f16x2",
            MultimemType::Bf16x2 => "bf16x2",
            MultimemType::F32 => "f32",
            MultimemType::E4m3x4 => "e4m3x4",
            MultimemType::E5m2x4 => "e5m2x4",
        }
    }

    /// The `.acc::` precision qualifier, if any: Float32 for the halves,
    /// Float16 for FP8, none for Float32.
    pub fn acc(self) -> Option<&'static str> {
        match self {
            MultimemType::F16x2 | MultimemType::Bf16x2 => Some(".acc::f32"),
            MultimemType::E4m3x4 | MultimemType::E5m2x4 => Some(".acc::f16"),
            MultimemType::F32 => None,
        }
    }

    /// The element type, and how many elements a 32-bit register holds.
    pub fn element(self) -> (DType, usize) {
        match self {
            MultimemType::F16x2 => (DType::F16, 2),
            MultimemType::Bf16x2 => (DType::BF16, 2),
            MultimemType::F32 => (DType::F32, 1),
            MultimemType::E4m3x4 => (DType::F8E4M3FN, 4),
            MultimemType::E5m2x4 => (DType::F8E5M2, 4),
        }
    }

    pub fn is_fp8(self) -> bool {
        matches!(self, MultimemType::E4m3x4 | MultimemType::E5m2x4)
    }
}

/// A `multimem` instruction ([`Op::Multimem`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultimemOp {
    /// `multimem.ld_reduce.weak.global.add{.acc}{.v4,.v2}.type`: the sum over
    /// every replica, `regs` (4, 2 or 1) 32-bit registers.
    LdReduce { ty: MultimemType, regs: u8 },
    /// `multimem.st.weak.global{.v4,.v2}.f32`: `regs` 32-bit registers to
    /// every replica.
    St { regs: u8 },
    /// `multimem.red.{order}.{scope}.global.add.s32 [a], 1`: add one to the
    /// 32-bit integer in every replica.
    RedAdd1 { order: MemoryOrder, scope: MemoryScope },
}

impl MultimemOp {
    /// The bytes it accesses at its address in each replica.
    pub fn bytes(&self) -> usize {
        match self {
            MultimemOp::LdReduce { regs, .. } | MultimemOp::St { regs } => 4 * *regs as usize,
            MultimemOp::RedAdd1 { .. } => 4,
        }
    }

    /// The PTX instruction, without operands.
    pub fn ptx(&self) -> String {
        let vec = |regs: u8| match regs {
            4 => ".v4",
            2 => ".v2",
            _ => "",
        };
        match *self {
            MultimemOp::LdReduce { ty, regs } => {
                format!("multimem.ld_reduce.weak.global.add{}{}.{}", ty.acc().unwrap_or(""), vec(regs), ty.ptx())
            }
            MultimemOp::St { regs } => format!("multimem.st.weak.global{}.f32", vec(regs)),
            MultimemOp::RedAdd1 { order, scope } => {
                let order = if order == MemoryOrder::Release { "release" } else { "relaxed" };
                let scope = if scope == MemoryScope::Sys { "sys" } else { "gpu" };
                format!("multimem.red.{order}.{scope}.global.add.s32")
            }
        }
    }
}

/// A standalone device function's result ([`KernelIr::function`]): the
/// kernel IR is then a `__device__` function, not a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFunctionSig {
    /// The value returned (the body's yield), if any.
    pub result: Option<ValueId>,
}

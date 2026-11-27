//! Data types (`DType` and its float formats), memory spaces and literals.

use super::*;

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

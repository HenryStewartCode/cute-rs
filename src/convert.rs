//! [`Value::to`](crate::Value::to): numeric conversions between every pair
//! of types, built from the IR's primitives.
//!
//! Conversions between the integer and IEEE float types (Float16, BFloat16,
//! Float32, Float64) are one [`Op::Cast`](crate::ir::Op::Cast). The others
//! are expanded here, so the emitter and the simulator see only operations
//! they already run:
//!
//! * to Boolean, CuTe DSL's `Boolean(x)`: `x != 0` for integers, and for
//!   floats an unordered `x != 0.0`, so NaN is true;
//! * TFloat32, stored as f32 bits: from Float32 through `cvt.rna.tf32.f32`
//!   ([`CvtFn::F32ToTf32`]), to Float32 as a reinterpretation;
//! * the FP8 formats through PTX `cvt` ([`CvtFn`]) and the formats PTX has no
//!   `cvt` for through integer bit manipulation.

use crate::builder::BuilderState as Builder;
use crate::ir::{BinaryOp, CmpPred, CvtFn, DType, Fp8Kind, Literal, Loc, NanEncoding, ValueId};

struct Conv<'a> {
    b: &'a mut Builder,
    loc: Loc,
}

impl Conv<'_> {
    fn constant(&mut self, ty: DType, bits: u128) -> ValueId {
        self.b.constant(Literal::from_bits128(ty, bits), self.loc)
    }

    fn cmp(&mut self, pred: CmpPred, x: ValueId, y: ValueId) -> ValueId {
        self.b.cmp(pred, x, y, self.loc)
    }

    fn bitcast(&mut self, x: ValueId, to: DType) -> ValueId {
        self.b.bitcast(x, to, self.loc)
    }

    fn cvt(&mut self, func: CvtFn, args: Vec<ValueId>) -> ValueId {
        self.b.cvt(func, args, self.loc)
    }

    fn bin(&mut self, op: BinaryOp, x: ValueId, y: ValueId) -> ValueId {
        self.b.binary(op, x, y, self.loc)
    }

    /// `x op c` with a constant of `x`'s type.
    fn bin_k(&mut self, op: BinaryOp, x: ValueId, c: u128) -> ValueId {
        let k = self.constant(self.b.dtype(x), c);
        self.bin(op, x, k)
    }

    fn cmp_k(&mut self, pred: CmpPred, x: ValueId, c: u128) -> ValueId {
        let k = self.constant(self.b.dtype(x), c);
        self.cmp(pred, x, k)
    }

    fn select(&mut self, c: ValueId, t: ValueId, f: ValueId) -> ValueId {
        self.b.select(c, t, f, self.loc)
    }

    fn cast(&mut self, x: ValueId, to: DType) -> ValueId {
        self.b.cast(x, to, self.loc)
    }

    fn convert(&mut self, x: ValueId, to: DType) -> ValueId {
        let from = self.b.dtype(x);
        if from == to {
            return x;
        }
        if to == DType::Bool {
            return self.boolean(x);
        }
        // A packed view holds several values: converting it (or into it) is
        // refused by the validator, as CuTe DSL refuses it.
        if from.is_packed_view() || to.is_packed_view() {
            return self.cast(x, to);
        }
        if from.is_narrow_float() {
            // E4M3FN and E5M2 widen to Float16 exactly with cvt; the others
            // decode to Float32 with integer operations.
            let wide = match fp8_kind(from) {
                Some(kind) => self.fp8_to_f16(x, kind),
                None => self.decode_to_f32(x),
            };
            return self.convert(wide, to);
        }
        if to.is_narrow_float() {
            return self.narrow(x, to);
        }
        if from == DType::TF32 {
            let f = self.bitcast(x, DType::F32);
            return self.convert(f, to);
        }
        if to == DType::TF32 {
            let f = self.convert(x, DType::F32);
            let bits = self.cvt(CvtFn::F32ToTf32, vec![f]);
            return self.bitcast(bits, DType::TF32);
        }
        self.b.cast(x, to, self.loc)
    }

    /// `x` (not an 8-bit float) converted to the 8-bit float `to`.
    fn narrow(&mut self, x: ValueId, to: DType) -> ValueId {
        let from = self.b.dtype(x);
        let byte = match (fp8_kind(to), from) {
            // One rounding from Float16 values, with the f16x2 form.
            (Some(kind), DType::F16) => {
                let zero = self.constant(DType::F16, 0);
                let pair = self.b.pack(vec![x, zero], DType::I32, self.loc);
                self.cvt(CvtFn::F16x2ToF8x2 { to: kind, relu: false }, vec![pair])
            }
            // From Float32, and from the other types through Float32
            // (exact from BFloat16 and TFloat32; Float64 and wide integers
            // round twice).
            (Some(kind), _) => {
                let f = self.convert(x, DType::F32);
                let zero = self.constant(DType::F32, 0);
                self.cvt(CvtFn::F32x2ToF8x2 { to: kind, relu: false }, vec![f, zero])
            }
            (None, _) if to == DType::F8E8M0FNU => {
                let f = self.convert(x, DType::F32);
                return self.f32_to_ue8m0(f);
            }
            // PTX has no conversion into the other formats (CuTe DSL refuses
            // Float8E4M3 and Float8E4M3B11FNUZ as destinations): the
            // validator reports this cast.
            (None, _) => return self.cast(x, to),
        };
        let low = self.cast(byte, DType::I8);
        self.bitcast(low, to)
    }

    /// A Float32 rounded up to UE8M0, saturating, as
    /// `cvt.rp.satfinite.ue8m0x2.f32` does, with integer operations (so on
    /// every target): UE8M0's exponent has f32's bias, so it is the f32
    /// exponent field, plus one for a nonzero mantissa.
    fn f32_to_ue8m0(&mut self, f: ValueId) -> ValueId {
        let u = self.bitcast(f, DType::U32);
        let mag = self.bin_k(BinaryOp::And, u, 0x7fff_ffff);
        let exp = self.bin_k(BinaryOp::ShrU, mag, 23);
        let mant = self.bin_k(BinaryOp::And, mag, 0x7f_ffff);
        let inexact = self.cmp_k(CmpPred::Ne, mant, 0);
        let inexact = self.cast(inexact, DType::U32);
        let normal = self.bin(BinaryOp::Add, exp, inexact);
        // An f32 subnormal rounds up to 2^-127 (0) up to half of 2^-126,
        // and to 2^-126 (1) above it.
        let above_half = self.cmp_k(CmpPred::Gt, mant, 0x40_0000);
        let sub = self.cast(above_half, DType::U32);
        let exp_zero = self.cmp_k(CmpPred::Eq, exp, 0);
        let e = self.select(exp_zero, sub, normal);
        let e = self.bin_k(BinaryOp::Min, e, 0xfe);
        let is_nan = self.cmp_k(CmpPred::Gt, mag, 0x7f80_0000);
        let nan = self.constant(DType::U32, 0xff);
        let e = self.select(is_nan, nan, e);
        let byte = self.cast(e, DType::U8);
        self.bitcast(byte, DType::F8E8M0FNU)
    }

    /// An E4M3FN or E5M2 value as Float16 (exact): `cvt.rn.f16x2.e4m3x2`.
    fn fp8_to_f16(&mut self, x: ValueId, kind: Fp8Kind) -> ValueId {
        let byte = self.bitcast(x, DType::U8);
        let pair = self.cast(byte, DType::I16);
        let halves = self.cvt(CvtFn::F8x2ToF16x2 { from: kind, relu: false }, vec![pair]);
        let low = self.cast(halves, DType::I16);
        self.bitcast(low, DType::F16)
    }

    /// An 8-bit float as Float32 (exact), from its bits with integer
    /// operations: the formats PTX has no conversion from.
    fn decode_to_f32(&mut self, x: ValueId) -> ValueId {
        let from = self.b.dtype(x);
        let f = from.float_format().expect("a float type");
        let (e_bits, m_bits) = (f.exp_bits as u128, f.mant_bits as u128);
        // A sub-byte float's bits, zero-extended into a byte.
        let byte =
            if from.is_sub_byte() { self.b.pack(vec![x], DType::U8, self.loc) } else { self.bitcast(x, DType::U8) };
        let u = self.cast(byte, DType::U32);
        let mag_mask = (1u128 << (e_bits + m_bits)) - 1;
        let mag = self.bin_k(BinaryOp::And, u, mag_mask);
        let e = self.bin_k(BinaryOp::ShrU, mag, m_bits);
        let m = self.bin_k(BinaryOp::And, mag, (1 << m_bits) - 1);
        // Normal: the exponent rebiased to f32's, the mantissa aligned.
        let rebiased = self.bin_k(BinaryOp::Add, e, (127 - f.bias) as u128);
        let exp_field = self.bin_k(BinaryOp::Shl, rebiased, 23);
        let mant_field = self.bin_k(BinaryOp::Shl, m, 23 - m_bits);
        let normal = self.bin(BinaryOp::Or, exp_field, mant_field);
        let e_zero = self.cmp_k(CmpPred::Eq, e, 0);
        let mut bits = if f.has_zero {
            // Subnormal: m * 2^(1 - bias - mantissa bits), exact in f32.
            let mf = self.cast(m, DType::F32);
            let scale = 2f32.powi(1 - f.bias - f.mant_bits as i32);
            let k = self.constant(DType::F32, scale.to_bits() as u128);
            let sub = self.bin(BinaryOp::Mul, mf, k);
            let sub_bits = self.bitcast(sub, DType::U32);
            self.select(e_zero, sub_bits, normal)
        } else {
            // E8M0's smallest value, 2^-127, is an f32 subnormal.
            let smallest = self.constant(DType::U32, 0x0040_0000);
            self.select(e_zero, smallest, normal)
        };
        let e_max = self.cmp_k(CmpPred::Eq, e, (1 << e_bits) - 1);
        if f.has_inf {
            let m_zero = self.cmp_k(CmpPred::Eq, m, 0);
            let is_inf = self.bin(BinaryOp::And, e_max, m_zero);
            let inf = self.constant(DType::U32, 0x7f80_0000);
            bits = self.select(is_inf, inf, bits);
        }
        if f.signed {
            let sign = self.bin_k(BinaryOp::ShrU, u, e_bits + m_bits);
            let sign = self.bin_k(BinaryOp::Shl, sign, 31);
            bits = self.bin(BinaryOp::Or, bits, sign);
        }
        let is_nan = match f.nan {
            NanEncoding::Ieee => {
                let m_nonzero = self.cmp_k(CmpPred::Ne, m, 0);
                self.bin(BinaryOp::And, e_max, m_nonzero)
            }
            NanEncoding::AllOnes => self.cmp_k(CmpPred::Eq, mag, mag_mask),
            NanEncoding::NegativeZero => self.cmp_k(CmpPred::Eq, u, 1 << (e_bits + m_bits)),
            NanEncoding::None => return self.bitcast(bits, DType::F32),
        };
        let nan = self.constant(DType::U32, 0x7fff_ffff);
        let bits = self.select(is_nan, nan, bits);
        self.bitcast(bits, DType::F32)
    }

    /// CuTe DSL's `Boolean(x)`.
    fn boolean(&mut self, x: ValueId) -> ValueId {
        let ty = self.b.dtype(x);
        if ty == DType::TF32 || ty.is_narrow_float() {
            let f = self.convert(x, DType::F32);
            return self.boolean(f);
        }
        let zero = self.constant(ty, 0);
        // On floats `Ne` is the unordered comparison (`cmpf une`): NaN is
        // true, and -0.0 equals 0.0.
        self.cmp(CmpPred::Ne, x, zero)
    }
}

/// The PTX conversion format of an FP8 type, if PTX converts it.
fn fp8_kind(t: DType) -> Option<Fp8Kind> {
    match t {
        DType::F8E4M3FN => Some(Fp8Kind::E4M3),
        DType::F8E5M2 => Some(Fp8Kind::E5M2),
        _ => None,
    }
}

/// `x` converted to `to`.
pub(crate) fn convert(b: &mut Builder, x: ValueId, to: DType, loc: Loc) -> ValueId {
    Conv { b, loc }.convert(x, to)
}

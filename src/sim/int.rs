//! Semantics of [`Op::IntInst`](crate::ir::Op::IntInst): the PTX integer and
//! bit instructions behind CuTe DSL's `cute.arch` integer helpers, written
//! from the PTX ISA's definitions (its "Semantics" pseudo-code), not from
//! the code the backend emits.
//!
//! Operands and results are raw bits, zero-extended to 64 like every value
//! in the simulator; `ty` is the type of the data operands.

use crate::ir::{DType, IntFn, PrmtMode};

/// The low `n` bits set (`n` at most 64).
fn low_bits(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// `bits`, a `width`-bit two's-complement number, as an `i64`.
fn signed(bits: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((bits << shift) as i64) >> shift
}

/// The result of `func` on data operands of type `ty`.
pub(super) fn int_inst(func: IntFn, ty: DType, args: &[u64]) -> u64 {
    let w = ty.bits() as u32;
    let result = match func {
        IntFn::Bfe => bfe(args[0], args[1], args[2], w, ty.is_signed()),
        IntFn::Bfi => bfi(args[0], args[1], args[2], args[3], w),
        IntFn::Lop3 { lut } => lop3(args[0], args[1], args[2], lut),
        IntFn::Shf { left, clamp } => shf(args[0] as u32, args[1] as u32, args[2] as u32, left, clamp) as u64,
        IntFn::Prmt { mode } => prmt(args[0] as u32, args[1] as u32, args[2] as u32, mode) as u64,
        IntFn::MulWide => {
            if ty.is_signed() {
                (signed(args[0], w) * signed(args[1], w)) as u64
            } else {
                args[0] * args[1]
            }
        }
        IntFn::Mul24 { hi } => mul24(args[0], args[1], hi, ty.is_signed()) as u64,
        IntFn::Mad24 { hi } => mul24(args[0], args[1], hi, ty.is_signed()).wrapping_add(args[2] as u32) as u64,
        IntFn::AddSat => (args[0] as u32 as i32).saturating_add(args[1] as u32 as i32) as u32 as u64,
        IntFn::SubSat => (args[0] as u32 as i32).saturating_sub(args[1] as u32 as i32) as u32 as u64,
        IntFn::Dp4a => dp4a(args[0], args[1], args[2]) as u64,
    };
    result & low_bits(func.result_type(ty).bits() as u32)
}

/// `dp4a.s32.s32`: `c` plus the products of the four bytes of `a` and `b`,
/// each sign-extended, wrapping at 32 bits.
fn dp4a(a: u64, b: u64, c: u64) -> u32 {
    let byte = |x: u64, i: u32| signed(x >> (8 * i) & 0xff, 8) as i32;
    (0..4).fold(c as u32, |d, i| d.wrapping_add(byte(a, i).wrapping_mul(byte(b, i)) as u32))
}

/// `bfe`: bits `pos..pos + len` of `a` (`pos` and `len` are the low bytes
/// of their operands), stopping at the msb, in the low bits of the result;
/// the rest is filled with the field's sign bit: zero for unsigned types or
/// an empty field, otherwise bit `min(pos + len - 1, msb)` of `a`.
fn bfe(a: u64, start: u64, length: u64, width: u32, is_signed: bool) -> u64 {
    let msb = width - 1;
    let (pos, len) = ((start & 0xff) as u32, (length & 0xff) as u32);
    let sign = if !is_signed || len == 0 { 0 } else { a >> (pos + len - 1).min(msb) & 1 };
    // Bits of `a` that exist: `pos + i <= msb`.
    let copied = if pos > msb { 0 } else { len.min(width - pos) };
    let field = if copied == 0 { 0 } else { a >> pos & low_bits(copied) };
    let fill = if sign == 1 { !low_bits(copied) } else { 0 };
    (field | fill) & low_bits(width)
}

/// `bfi`: `value` with bits `pos..pos + len` replaced by the low bits of
/// `field`, stopping at the msb.
fn bfi(field: u64, value: u64, start: u64, length: u64, width: u32) -> u64 {
    let (pos, len) = ((start & 0xff) as u32, (length & 0xff) as u32);
    if pos >= width {
        return value;
    }
    let mask = low_bits(len.min(width - pos)) << pos;
    (value & !mask | field << pos & mask) & low_bits(width)
}

/// `lop3.b32`: bit `i` of the result is bit `(a_i << 2) | (b_i << 1) | c_i`
/// of `lut`, because PTX defines the table as `F(0xF0, 0xCC, 0xAA)`. Here as
/// the sum of the minterms the table selects.
fn lop3(a: u64, b: u64, c: u64, lut: u8) -> u64 {
    let pick = |x: u64, set: bool| if set { x } else { !x };
    (0..8)
        .filter(|k| lut >> k & 1 == 1)
        .map(|k| pick(a, k & 4 != 0) & pick(b, k & 2 != 0) & pick(c, k & 1 != 0))
        .fold(0, |acc, minterm| acc | minterm)
        & low_bits(32)
}

/// `shf`: the 64-bit value `b:a` shifted by `n` (`min(c, 32)` with clamp,
/// `c mod 32` otherwise); its high word for a left shift, its low word for
/// a right one.
fn shf(a: u32, b: u32, c: u32, left: bool, clamp: bool) -> u32 {
    let n = if clamp { c.min(32) } else { c & 31 };
    let wide = (b as u64) << 32 | a as u64;
    if left { (wide << n >> 32) as u32 } else { (wide >> n) as u32 }
}

/// Source bytes of the result's bytes 0 to 3, by selector, for each mode
/// (the PTX ISA's table).
const PRMT_TABLES: [(PrmtMode, [[u8; 4]; 4]); 6] = [
    (PrmtMode::F4e, [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
    (PrmtMode::B4e, [[0, 7, 6, 5], [1, 0, 7, 6], [2, 1, 0, 7], [3, 2, 1, 0]]),
    (PrmtMode::Rc8, [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
    (PrmtMode::Ecl, [[0, 1, 2, 3], [1, 1, 2, 3], [2, 2, 2, 3], [3, 3, 3, 3]]),
    (PrmtMode::Ecr, [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3]]),
    (PrmtMode::Rc16, [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]]),
];

/// `prmt.b32`: four bytes picked from the eight of `b:a`.
fn prmt(a: u32, b: u32, c: u32, mode: PrmtMode) -> u32 {
    let source = (b as u64) << 32 | a as u64;
    let byte = |k: u32| (source >> (8 * (k & 7)) & 0xff) as u32;
    let bytes: [u32; 4] = match mode {
        PrmtMode::Generic => std::array::from_fn(|i| {
            let select = c >> (4 * i) & 0xf;
            let picked = byte(select);
            if select & 8 == 0 {
                picked
            } else if picked & 0x80 != 0 {
                0xff
            } else {
                0
            }
        }),
        _ => {
            let table = PRMT_TABLES.iter().find(|(m, _)| *m == mode).expect("every mode has a table").1;
            table[(c & 3) as usize].map(|k| byte(k as u32))
        }
    };
    bytes.iter().enumerate().fold(0, |acc, (i, &b)| acc | b << (8 * i))
}

/// `mul24`: the product of the low 24 bits of `a` and `b` (sign-extended
/// from bit 23 when signed), a 48-bit number `t`; `t[47:16]` with `hi`,
/// `t[31:0]` without.
fn mul24(a: u64, b: u64, hi: bool, is_signed: bool) -> u32 {
    let operand = |x: u64| if is_signed { signed(x & 0xff_ffff, 24) } else { (x & 0xff_ffff) as i64 };
    let t = operand(a) * operand(b);
    if hi { (t >> 16) as u32 } else { t as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference written from the PTX ISA's pseudo-code, bit by bit.
    fn bfe_ref(a: u64, b: u64, c: u64, width: u32, is_signed: bool) -> u64 {
        let msb = width - 1;
        let (pos, len) = ((b & 0xff) as u32, (c & 0xff) as u32);
        let sbit = if !is_signed || len == 0 { 0 } else { a >> (pos + len - 1).min(msb) & 1 };
        (0..=msb).fold(0, |d, i| d | (if i < len && pos + i <= msb { a >> (pos + i) & 1 } else { sbit }) << i)
    }

    fn bfi_ref(a: u64, b: u64, c: u64, d: u64, width: u32) -> u64 {
        let msb = width - 1;
        let (pos, len) = ((c & 0xff) as u32, (d & 0xff) as u32);
        let mut f = b;
        let mut i = 0;
        while i < len && pos + i <= msb {
            f = f & !(1 << (pos + i)) | (a >> i & 1) << (pos + i);
            i += 1;
        }
        f
    }

    /// A few hundred pseudo-random 64-bit patterns.
    fn patterns() -> Vec<u64> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut v: Vec<u64> =
            vec![0, 1, u64::MAX, 0x8000_0000, 0x7fff_ffff, 0x8000_0000_0000_0000, 0x0123_4567_89ab_cdef];
        for _ in 0..200 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(state);
        }
        v
    }

    #[test]
    fn bit_fields_follow_the_ptx_pseudo_code() {
        let positions: Vec<u64> = (0..300).step_by(7).chain([31, 32, 33, 63, 64, 65, 255, 256, 0x1_0005]).collect();
        for (ty, width) in [(DType::I32, 32), (DType::U32, 32), (DType::I64, 64), (DType::U64, 64)] {
            for &x in &patterns()[..40] {
                let x = x & low_bits(width);
                let y = x.rotate_left(17) & low_bits(width);
                for &p in &positions {
                    for &l in &positions {
                        let got = int_inst(IntFn::Bfe, ty, &[x, p, l]);
                        assert_eq!(got, bfe_ref(x, p, l, width, ty.is_signed()), "bfe {ty} {x:#x} {p} {l}");
                        let got = int_inst(IntFn::Bfi, ty, &[y, x, p, l]);
                        assert_eq!(got, bfi_ref(y, x, p, l, width), "bfi {ty} {y:#x} {x:#x} {p} {l}");
                    }
                }
            }
        }
        // A bfi with the replacement first, PTX's operand order, and bfe of a
        // 4-bit field, signed and unsigned.
        assert_eq!(int_inst(IntFn::Bfi, DType::U32, &[0xa, 0x8765_4321, 12, 4]), 0x8765_a321);
        assert_eq!(int_inst(IntFn::Bfe, DType::I32, &[0x0000_0f00, 8, 4]), 0xffff_ffff);
        assert_eq!(int_inst(IntFn::Bfe, DType::U32, &[0x0000_0f00, 8, 4]), 0xf);
        // An empty field is zero, even signed; a start past the msb fills
        // with the sign bit (the msb).
        assert_eq!(int_inst(IntFn::Bfe, DType::I32, &[0xffff_ffff, 3, 0]), 0);
        assert_eq!(int_inst(IntFn::Bfe, DType::I32, &[0x8000_0000, 40, 3]), 0xffff_ffff);
        assert_eq!(int_inst(IntFn::Bfe, DType::I64, &[0x8000_0000, 40, 3]), 0);
    }

    #[test]
    fn lop3_follows_its_table() {
        // The PTX ISA's examples: immLut = F(0xF0, 0xCC, 0xAA).
        let (a, b, c) = (0xF0, 0xCC, 0xAA);
        for (lut, f) in [
            (0x80u8, a & b & c),
            (0xFE, a | b | c),
            (0x40, a & b & !c & 0xff),
            (0x1A, (a & b | c) ^ a),
            (0x00, 0),
            (0xFF, 0xff),
        ] {
            // Bits 8 to 31 of the inputs are zero, so those of the result
            // are F(0, 0, 0), bit 0 of the table.
            let high = if lut & 1 == 1 { 0xffff_ff00 } else { 0 };
            assert_eq!(int_inst(IntFn::Lop3 { lut }, DType::U32, &[a, b, c]), high | f, "{lut:#x}");
        }
        for lut in 0..=255u8 {
            for w in patterns().windows(3).take(20) {
                let (x, y, z) = (w[0] & 0xffff_ffff, w[1] & 0xffff_ffff, w[2] & 0xffff_ffff);
                let want = (0..32).fold(0, |d, i| {
                    let index = (x >> i & 1) << 2 | (y >> i & 1) << 1 | z >> i & 1;
                    d | (lut as u64 >> index & 1) << i
                });
                assert_eq!(int_inst(IntFn::Lop3 { lut }, DType::I32, &[x, y, z]), want);
            }
        }
    }

    #[test]
    fn funnel_shifts_follow_the_ptx_formulas() {
        // u32 shifts by 32 give 0, as in the PTX ISA's formulas.
        let shl = |x: u32, n: u32| x.checked_shl(n).unwrap_or(0);
        let shr = |x: u32, n: u32| x.checked_shr(n).unwrap_or(0);
        for w in patterns().windows(2).take(50) {
            let (a, b) = (w[0] as u32, w[1] as u32);
            for c in (0..70).chain([u32::MAX, 0x8000_0000, 0x1_0000_0001u64 as u32]) {
                for (left, clamp) in [(true, false), (false, false), (true, true), (false, true)] {
                    let n = if clamp { c.min(32) } else { c & 0x1f };
                    let want = if left { shl(b, n) | shr(a, 32 - n) } else { shl(b, 32 - n) | shr(a, n) };
                    let got = int_inst(IntFn::Shf { left, clamp }, DType::U32, &[a as u64, b as u64, c as u64]);
                    assert_eq!(got, want as u64, "shf left {left} clamp {clamp} {a:#x} {b:#x} {c}");
                }
            }
            // With a == b, a funnel shift rotates.
            let r = int_inst(IntFn::Shf { left: true, clamp: false }, DType::U32, &[a as u64, a as u64, 13]);
            assert_eq!(r, a.rotate_left(13) as u64);
        }
    }

    #[test]
    fn prmt_follows_its_definition() {
        let (a, b) = (0x8382_0180u32, 0x0786_0584u32);
        // Identity, the high word, a byte swap and sign replication.
        for (c, want) in [
            (0x3210, a),
            (0x7654, b),
            (0x0123, a.swap_bytes()),
            (0x8888, 0xffff_ffff),
            (0x9999, 0),
            (0xbbbb, 0xffff_ffff),
            (0x7f80, 0x0700_ff80),
        ] {
            assert_eq!(
                int_inst(IntFn::Prmt { mode: PrmtMode::Generic }, DType::U32, &[a as u64, b as u64, c]),
                want as u64
            );
        }
        // The modes, from the formulas in the PTX ISA's table.
        let byte = |k: u32| ((b as u64) << 32 | a as u64) >> (8 * (k & 7)) & 0xff;
        for s in 0..8u32 {
            for (mode, index) in [
                (PrmtMode::F4e, (|s: u32, i: u32| s + i) as fn(u32, u32) -> u32),
                (PrmtMode::B4e, |s, i| (s + 8 - i) % 8),
                (PrmtMode::Rc8, |s, _| s),
                (PrmtMode::Ecl, |s, i| i.max(s)),
                (PrmtMode::Ecr, |s, i| i.min(s)),
                (PrmtMode::Rc16, |s, i| (i & 1) + 2 * (s & 1)),
            ] {
                let want = (0..4).fold(0, |d, i| d | byte(index(s & 3, i)) << (8 * i));
                let got = int_inst(IntFn::Prmt { mode }, DType::U32, &[a as u64, b as u64, s as u64 | 0xfff0]);
                assert_eq!(got, want, "{mode:?} selector {s}");
            }
        }
    }

    #[test]
    fn multiplies() {
        for w in patterns().windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            let wide = |ty, x: u64, y: u64| int_inst(IntFn::MulWide, ty, &[x, y]);
            assert_eq!(wide(DType::I16, a & 0xffff, b & 0xffff), (a as i16 as i32 * b as i16 as i32) as u32 as u64);
            assert_eq!(wide(DType::U16, a & 0xffff, b & 0xffff), (a as u16 as u64) * (b as u16 as u64));
            assert_eq!(wide(DType::I32, a & 0xffff_ffff, b & 0xffff_ffff), (a as i32 as i64 * b as i32 as i64) as u64);
            assert_eq!(wide(DType::U32, a & 0xffff_ffff, b & 0xffff_ffff), (a as u32 as u64) * (b as u32 as u64));
            let (x, y, z) = (a & 0xffff_ffff, b & 0xffff_ffff, c & 0xffff_ffff);
            // The 48-bit product of the low 24 bits, sign-extended from
            // bit 23 when signed.
            let (su, sv) = (((a as i32) << 8 >> 8) as i64, ((b as i32) << 8 >> 8) as i64);
            let (uu, uv) = ((a & 0xff_ffff) as i64, (b & 0xff_ffff) as i64);
            for (ty, t) in [(DType::I32, su * sv), (DType::U32, uu * uv)] {
                let lo = t as u32;
                let hi = (t >> 16) as u32;
                assert_eq!(int_inst(IntFn::Mul24 { hi: false }, ty, &[x, y]), lo as u64);
                assert_eq!(int_inst(IntFn::Mul24 { hi: true }, ty, &[x, y]), hi as u64);
                assert_eq!(int_inst(IntFn::Mad24 { hi: false }, ty, &[x, y, z]), lo.wrapping_add(z as u32) as u64);
                assert_eq!(int_inst(IntFn::Mad24 { hi: true }, ty, &[x, y, z]), hi.wrapping_add(z as u32) as u64);
            }
            let (p, q) = (a as i32, b as i32);
            let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32 as u64;
            assert_eq!(int_inst(IntFn::AddSat, DType::I32, &[x, y]), clamp(p as i64 + q as i64));
            assert_eq!(int_inst(IntFn::SubSat, DType::I32, &[x, y]), clamp(p as i64 - q as i64));
        }
        assert_eq!(int_inst(IntFn::AddSat, DType::I32, &[i32::MAX as u32 as u64, 1]), i32::MAX as u64);
        assert_eq!(int_inst(IntFn::SubSat, DType::I32, &[i32::MIN as u32 as u64, 1]), i32::MIN as u32 as u64);
        // mul24.hi is bits 47:16 of the 48-bit product, not its high word:
        // 0xffffff * 0xffffff = 0xfffffe000001.
        assert_eq!(int_inst(IntFn::Mul24 { hi: true }, DType::U32, &[0xffff_ffff, 0xffff_ffff]), 0xffff_fe00);
        assert_eq!(int_inst(IntFn::Mul24 { hi: false }, DType::U32, &[0xffff_ffff, 0xffff_ffff]), 0xfe00_0001);
        // Signed, 0xffffff is -1.
        assert_eq!(int_inst(IntFn::Mul24 { hi: false }, DType::I32, &[0x00ff_ffff, 5]), (-5i32) as u32 as u64);
        assert_eq!(int_inst(IntFn::Mul24 { hi: true }, DType::I32, &[0x00ff_ffff, 5]), u32::MAX as u64);
    }

    /// `dp4a.s32.s32` from the PTX ISA's pseudo-code: the accumulator plus
    /// the products of the sign-extended bytes; against a one-hot byte with
    /// a zero accumulator it sign-extends that byte of the other operand.
    #[test]
    fn dp4a_follows_the_ptx_pseudo_code() {
        for w in patterns().windows(3) {
            let (a, b, c) = (w[0] & 0xffff_ffff, w[1] & 0xffff_ffff, w[2] & 0xffff_ffff);
            let bytes = |x: u64| (x as u32).to_le_bytes().map(|v| v as i8 as i64);
            let want = bytes(a).iter().zip(bytes(b)).fold(c as u32 as i32 as i64, |d, (x, y)| d + x * y);
            assert_eq!(int_inst(IntFn::Dp4a, DType::I32, &[a, b, c]), want as u32 as u64, "{a:#x} {b:#x} {c:#x}");
            for k in 0..4 {
                let byte = (a as u32 >> (8 * k)) as u8 as i8;
                assert_eq!(int_inst(IntFn::Dp4a, DType::I32, &[a, 1 << (8 * k), 0]), byte as i32 as u32 as u64);
            }
        }
        // The largest sum wraps nothing: 4 * 128 * 128 fits.
        assert_eq!(int_inst(IntFn::Dp4a, DType::I32, &[0x8080_8080, 0x8080_8080, 0]), 65536);
        assert_eq!(int_inst(IntFn::Dp4a, DType::I32, &[0x8080_8080, 0x8080_8080, 0x7fff_ffff]), 0x8000_ffff);
    }
}

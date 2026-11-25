//! Semantics of [`Op::FloatInst`](crate::ir::Op::FloatInst): floating-point
//! instructions with explicit PTX modifiers.
//!
//! The IEEE 754 operations (`add`, `mul`, `fma`, `div`, `rcp` and `sqrt` in
//! every rounding direction) are computed with integer arithmetic, from the
//! standard's definition: the exact result, then one rounding. They do not
//! depend on the host's floating point, whose rounding direction Rust cannot
//! change. `.ftz` flushes subnormal inputs and results to zero of the same
//! sign, as the PTX ISA specifies.
//!
//! The hardware approximations (`.approx`, `div.full`) have no exact
//! definition: the simulator computes the function they approximate,
//! correctly rounded (in double precision, then rounded to single), and
//! honours their documented special cases. `rcp.approx.ftz.f64` and
//! `rsqrt.approx.ftz.f64` follow the ISA's description: the reciprocal (or
//! reciprocal square root) of the argument's high word, with the result's
//! low word cleared.
//!
//! `min` and `max` follow the PTX ISA's pseudo-code: a NaN operand is
//! ignored (both give the canonical NaN) unless `.NaN` makes any NaN the
//! result, `+0 > -0`, and `.xorsign.abs` compares the absolute values and
//! gives a result that is not NaN the exclusive or of the operands' signs.
//!
//! NaN results follow [`ptx_nan`], for these instructions and for the
//! simulator's plain arithmetic alike.
//!
//! The same exact arithmetic, on Float16 and BFloat16, gives the
//! half-precision instructions of [`Op::HalfInst`](crate::ir::Op::HalfInst)
//! (`add`, `sub` and `fma` on `.f16x2`, `.bf16x2`, `.f16` and `.bf16`, with
//! `.ftz`, `.sat`, `.relu` and `.oob`) and the single rounding of `fma` on
//! those types, and [`cvt_inst`] the conversions of
//! [`Op::Cvt`](crate::ir::Op::Cvt).

use std::cmp::Ordering;

use crate::ir::{CvtFn, DType, FloatFn, FloatMode, HalfFn, HalfModifiers, HalfType, PackedOp, RoundingMode};
use crate::numeric::{NarrowRounding, decode, encode};

#[derive(Clone, Copy)]
struct Format {
    /// Stored significand bits.
    mant: u32,
    exp_bits: u32,
}

impl Format {
    const F16: Format = Format { mant: 10, exp_bits: 5 };
    const BF16: Format = Format { mant: 7, exp_bits: 8 };
    const F32: Format = Format { mant: 23, exp_bits: 8 };
    const F64: Format = Format { mant: 52, exp_bits: 11 };

    /// The format of a float type.
    fn of(ty: DType) -> Format {
        match ty {
            DType::F16 => Format::F16,
            DType::BF16 => Format::BF16,
            DType::F32 => Format::F32,
            DType::F64 => Format::F64,
            _ => unreachable!("{ty} is not a float type"),
        }
    }

    fn bias(self) -> i32 {
        (1 << (self.exp_bits - 1)) - 1
    }

    fn max_biased(self) -> u64 {
        (1 << self.exp_bits) - 1
    }

    fn sign(self, neg: bool) -> u64 {
        (neg as u64) << (self.mant + self.exp_bits)
    }

    fn inf(self, neg: bool) -> u64 {
        self.sign(neg) | self.max_biased() << self.mant
    }

    fn max_finite(self, neg: bool) -> u64 {
        self.sign(neg) | (self.max_biased() - 1) << self.mant | ((1 << self.mant) - 1)
    }

    /// The NaN PTX instructions return (`0x7fffffff` for f32).
    fn nan(self) -> u64 {
        (1u64 << (self.mant + self.exp_bits)) - 1
    }

    fn is_subnormal(self, bits: u64) -> bool {
        (bits >> self.mant) & self.max_biased() == 0 && bits & ((1 << self.mant) - 1) != 0
    }

    /// Flush a subnormal to a zero of the same sign.
    fn flush(self, bits: u64) -> u64 {
        if self.is_subnormal(bits) { bits & self.sign(true) } else { bits }
    }
}

/// A decoded floating-point number. Finite values are `sig * 2^exp`.
#[derive(Clone, Copy, Debug)]
enum Num {
    Nan,
    Inf(bool),
    Zero(bool),
    Finite { neg: bool, sig: u128, exp: i32 },
}

fn unpack(bits: u64, f: Format) -> Num {
    let neg = bits & f.sign(true) != 0;
    let biased = (bits >> f.mant) & f.max_biased();
    let frac = bits & ((1 << f.mant) - 1);
    let min_exp = 1 - f.bias() - f.mant as i32;
    match (biased, frac) {
        (b, 0) if b == f.max_biased() => Num::Inf(neg),
        (b, _) if b == f.max_biased() => Num::Nan,
        (0, 0) => Num::Zero(neg),
        (0, _) => Num::Finite { neg, sig: frac as u128, exp: min_exp },
        _ => Num::Finite { neg, sig: (frac | 1 << f.mant) as u128, exp: min_exp + biased as i32 - 1 },
    }
}

fn bit_length(x: u128) -> i32 {
    128 - x.leading_zeros() as i32
}

/// Round `(-1)^neg * sig * 2^exp` (`sig` nonzero) to `f`. Bits of `sig`
/// below the result's precision decide the rounding; a caller that lost
/// bits sets the lowest one ("sticky"), far enough below to only break ties.
fn round_pack(neg: bool, sig: u128, exp: i32, mode: RoundingMode, f: Format) -> u64 {
    let p = f.mant as i32 + 1;
    let msb = exp + bit_length(sig) - 1;
    let emin = 1 - f.bias();
    let mut lsb = (msb - (p - 1)).max(emin - (p - 1));
    let shift = lsb - exp;
    let (q, above_half, at_half, inexact) = match shift {
        s if s <= 0 => (sig << (-s) as u32, false, false, false),
        s if s > 128 => (0, false, false, true),
        128 => (0, sig > 1 << 127, sig == 1 << 127, true),
        s => {
            let rem = sig & ((1u128 << s) - 1);
            let half = 1u128 << (s - 1);
            (sig >> s, rem > half, rem == half, rem != 0)
        }
    };
    let up = match mode {
        RoundingMode::NearestEven => above_half || (at_half && q & 1 == 1),
        RoundingMode::Zero => false,
        RoundingMode::Up => !neg && inexact,
        RoundingMode::Down => neg && inexact,
    };
    let mut q = q + up as u128;
    if q >> p != 0 {
        q >>= 1;
        lsb += 1;
    }
    if q == 0 {
        return f.sign(neg);
    }
    if q < 1 << f.mant {
        return f.sign(neg) | q as u64;
    }
    let biased = lsb + f.mant as i32 + f.bias();
    if biased >= f.max_biased() as i32 {
        return match mode {
            RoundingMode::NearestEven => f.inf(neg),
            RoundingMode::Zero => f.max_finite(neg),
            RoundingMode::Up => {
                if neg {
                    f.max_finite(true)
                } else {
                    f.inf(false)
                }
            }
            RoundingMode::Down => {
                if neg {
                    f.inf(true)
                } else {
                    f.max_finite(false)
                }
            }
        };
    }
    f.sign(neg) | (biased as u64) << f.mant | (q as u64 & ((1 << f.mant) - 1))
}

/// The sign of an exact zero sum of operands of opposite signs: `+0`, except
/// when rounding toward -infinity.
fn zero_sum(mode: RoundingMode, f: Format) -> u64 {
    f.sign(mode == RoundingMode::Down)
}

/// `a + b` for finite nonzero operands with significands of at most 106
/// bits.
#[allow(clippy::too_many_arguments)]
fn add_finite(na: bool, sa: u128, ea: i32, nb: bool, sb: u128, eb: i32, mode: RoundingMode, f: Format) -> u64 {
    let (ma, mb) = (ea + bit_length(sa) - 1, eb + bit_length(sb) - 1);
    let ((nx, sx, ex), (ny, sy, ey)) =
        if ma >= mb { ((na, sa, ea), (nb, sb, eb)) } else { ((nb, sb, eb), (na, sa, ea)) };
    // The larger operand's leading bit goes to bit 125; the smaller one is
    // aligned to it, keeping lost bits as a sticky bit at bit 0, at least 70
    // bits below where the sum rounds.
    let base = ex + bit_length(sx) - 1 - 125;
    let x = sx << (ex - base) as u32;
    let y = if ey >= base {
        sy << (ey - base) as u32
    } else {
        let s = (base - ey) as u32;
        if s >= 128 { 1 } else { (sy >> s) | (sy & ((1u128 << s) - 1) != 0) as u128 }
    };
    let (neg, mag) = match (nx == ny, x.cmp(&y)) {
        (true, _) => (nx, x + y),
        (false, Ordering::Equal) => return zero_sum(mode, f),
        (false, Ordering::Greater) => (nx, x - y),
        (false, Ordering::Less) => (ny, y - x),
    };
    round_pack(neg, mag, base, mode, f)
}

fn add(a: u64, b: u64, mode: RoundingMode, f: Format) -> u64 {
    match (unpack(a, f), unpack(b, f)) {
        (Num::Nan, _) | (_, Num::Nan) => f.nan(),
        (Num::Inf(x), Num::Inf(y)) if x != y => f.nan(),
        (Num::Inf(x), _) | (_, Num::Inf(x)) => f.inf(x),
        (Num::Zero(x), Num::Zero(y)) => {
            if x == y {
                f.sign(x)
            } else {
                zero_sum(mode, f)
            }
        }
        (Num::Zero(_), _) => b,
        (_, Num::Zero(_)) => a,
        (Num::Finite { neg: na, sig: sa, exp: ea }, Num::Finite { neg: nb, sig: sb, exp: eb }) => {
            add_finite(na, sa, ea, nb, sb, eb, mode, f)
        }
    }
}

fn mul(a: u64, b: u64, mode: RoundingMode, f: Format) -> u64 {
    match (unpack(a, f), unpack(b, f)) {
        (Num::Nan, _) | (_, Num::Nan) => f.nan(),
        (Num::Inf(_), Num::Zero(_)) | (Num::Zero(_), Num::Inf(_)) => f.nan(),
        (Num::Inf(x), y) | (y, Num::Inf(x)) => f.inf(x ^ neg_of(y)),
        (Num::Zero(x), y) | (y, Num::Zero(x)) => f.sign(x ^ neg_of(y)),
        (Num::Finite { neg: na, sig: sa, exp: ea }, Num::Finite { neg: nb, sig: sb, exp: eb }) => {
            round_pack(na ^ nb, sa * sb, ea + eb, mode, f)
        }
    }
}

fn neg_of(n: Num) -> bool {
    match n {
        Num::Nan => false,
        Num::Inf(s) | Num::Zero(s) => s,
        Num::Finite { neg, .. } => neg,
    }
}

fn fma(a: u64, b: u64, c: u64, mode: RoundingMode, f: Format) -> u64 {
    let (x, y, z) = (unpack(a, f), unpack(b, f), unpack(c, f));
    if matches!(x, Num::Nan) || matches!(y, Num::Nan) || matches!(z, Num::Nan) {
        return f.nan();
    }
    let product_neg = neg_of(x) ^ neg_of(y);
    match (x, y) {
        (Num::Inf(_), Num::Zero(_)) | (Num::Zero(_), Num::Inf(_)) => f.nan(),
        (Num::Inf(_), _) | (_, Num::Inf(_)) => match z {
            Num::Inf(s) if s != product_neg => f.nan(),
            _ => f.inf(product_neg),
        },
        (Num::Zero(_), _) | (_, Num::Zero(_)) => match z {
            Num::Inf(s) => f.inf(s),
            Num::Zero(s) if s == product_neg => f.sign(s),
            Num::Zero(_) => zero_sum(mode, f),
            _ => c,
        },
        (Num::Finite { sig: sa, exp: ea, .. }, Num::Finite { sig: sb, exp: eb, .. }) => match z {
            Num::Inf(s) => f.inf(s),
            Num::Zero(_) => round_pack(product_neg, sa * sb, ea + eb, mode, f),
            Num::Finite { neg: nc, sig: sc, exp: ec } => add_finite(product_neg, sa * sb, ea + eb, nc, sc, ec, mode, f),
            Num::Nan => unreachable!("handled above"),
        },
        _ => unreachable!("handled above"),
    }
}

fn div(a: u64, b: u64, mode: RoundingMode, f: Format) -> u64 {
    match (unpack(a, f), unpack(b, f)) {
        (Num::Nan, _) | (_, Num::Nan) => f.nan(),
        (Num::Inf(_), Num::Inf(_)) | (Num::Zero(_), Num::Zero(_)) => f.nan(),
        (Num::Inf(x), y) => f.inf(x ^ neg_of(y)),
        (x, Num::Inf(y)) => f.sign(neg_of(x) ^ y),
        (x, Num::Zero(y)) => f.inf(neg_of(x) ^ y),
        (Num::Zero(x), y) => f.sign(x ^ neg_of(y)),
        (Num::Finite { neg: na, sig: sa, exp: ea }, Num::Finite { neg: nb, sig: sb, exp: eb }) => {
            // Leading bits at bit 63, so the quotient has at least 62 bits.
            let (ta, tb) = (64 - bit_length(sa), 64 - bit_length(sb));
            let (sa, sb) = (sa << ta, sb << tb);
            let num = sa << 63;
            let q = num / sb;
            let sticky = (num % sb != 0) as u128;
            round_pack(na ^ nb, q | sticky, (ea - ta) - 63 - (eb - tb), mode, f)
        }
    }
}

fn sqrt(a: u64, mode: RoundingMode, f: Format) -> u64 {
    match unpack(a, f) {
        Num::Nan => f.nan(),
        Num::Zero(_) => a,
        Num::Inf(false) => a,
        Num::Inf(true) | Num::Finite { neg: true, .. } => f.nan(),
        Num::Finite { sig, exp, .. } => {
            // 125 or 126 bits with an even exponent: a 63-bit root.
            let mut t = 125 - bit_length(sig);
            if (exp - t) % 2 != 0 {
                t += 1;
            }
            let n = sig << t as u32;
            let r = n.isqrt();
            let sticky = (r * r != n) as u128;
            round_pack(false, r | sticky, (exp - t) / 2, mode, f)
        }
    }
}

/// The f32 value of an approximation computed in f64, with its
/// special-value behaviour.
fn approx_f32(func: FloatFn, full: bool, args: &[u64]) -> u64 {
    let x = f32::from_bits(args[0] as u32) as f64;
    let r = match func {
        FloatFn::Ex2 => x.exp2(),
        FloatFn::Lg2 => x.log2(),
        FloatFn::Sin => x.sin(),
        FloatFn::Cos => x.cos(),
        FloatFn::Tanh => x.tanh(),
        FloatFn::Rsqrt => 1.0 / x.sqrt(),
        FloatFn::Sqrt => return (x.sqrt() as f32).to_bits() as u64,
        FloatFn::Rcp => return (1.0 / x as f32).to_bits() as u64,
        FloatFn::Div => {
            let (a, b) = (f32::from_bits(args[0] as u32), f32::from_bits(args[1] as u32));
            // div.approx computes a * (1/b): for 2^126 < |b| < 2^128 the
            // reciprocal is flushed, so the result is zero, or NaN for an
            // infinite (or NaN) a.
            if !full && b.is_finite() && b.abs() > 2f32.powi(126) {
                if !a.is_finite() {
                    return Format::F32.nan();
                }
                return Format::F32.sign(a.is_sign_negative() != b.is_sign_negative());
            }
            return (a / b).to_bits() as u64;
        }
        _ => unreachable!("validated: {func:?} has no approximation"),
    };
    (r as f32).to_bits() as u64
}

/// Execute a floating-point instruction with explicit modifiers on raw
/// operand bits.
pub(super) fn float_inst(
    func: FloatFn,
    mode: FloatMode,
    ftz: bool,
    nan: bool,
    xorsign_abs: bool,
    ty: DType,
    args: &[u64],
) -> u64 {
    let f = if ty == DType::F64 { Format::F64 } else { Format::F32 };
    let mut ops = [0u64; 3];
    for (k, &a) in args.iter().enumerate() {
        ops[k] = if ftz { f.flush(a) } else { a };
    }
    let ops = &ops[..args.len()];
    let canon = |bits: u64| if matches!(unpack(bits, f), Num::Nan) { f.nan() } else { bits };
    if ty == DType::F64 && mode == FloatMode::Approx && ftz {
        return gross_f64(func, ops[0]);
    }
    let result = match mode {
        FloatMode::Round(rm) => match func {
            FloatFn::Add => add(ops[0], ops[1], rm, f),
            FloatFn::Mul => mul(ops[0], ops[1], rm, f),
            FloatFn::Fma => fma(ops[0], ops[1], ops[2], rm, f),
            FloatFn::Div => div(ops[0], ops[1], rm, f),
            FloatFn::Rcp => div(one(f), ops[0], rm, f),
            FloatFn::Sqrt => sqrt(ops[0], rm, f),
            _ => unreachable!("validated: {func:?} takes no rounding mode"),
        },
        FloatMode::Approx | FloatMode::Full if ty == DType::F32 => {
            canon(approx_f32(func, mode == FloatMode::Full, ops))
        }
        FloatMode::Approx => match func {
            // rsqrt.approx.f64 (emulated in software).
            FloatFn::Rsqrt => (1.0 / f64::from_bits(ops[0]).sqrt()).to_bits(),
            _ => unreachable!("validated: no f64 approximation of {func:?} without ftz"),
        },
        FloatMode::Full => unreachable!("validated: div.full is f32 only"),
        FloatMode::Exact => {
            let value = |b: u64| if ty == DType::F64 { f64::from_bits(b) } else { f32::from_bits(b as u32) as f64 };
            match func {
                FloatFn::Abs => ops[0] & !f.sign(true),
                FloatFn::Min | FloatFn::Max => {
                    let (mut a, mut b) = (ops[0], ops[1]);
                    // `.xorsign.abs`: the signs' exclusive or is taken
                    // before the absolute values are compared.
                    let sign = (a ^ b) & f.sign(true);
                    if xorsign_abs {
                        a &= !f.sign(true);
                        b &= !f.sign(true);
                    }
                    let (x, y) = (value(a), value(b));
                    let r = match (x.is_nan(), y.is_nan()) {
                        (true, true) => f.nan(),
                        (true, false) | (false, true) if nan => f.nan(),
                        (true, false) => b,
                        (false, true) => a,
                        // -0 < +0 here, as in PTX.
                        (false, false) => {
                            let a_first = match x.partial_cmp(&y) {
                                Some(Ordering::Less) => true,
                                Some(Ordering::Greater) => false,
                                _ => a & f.sign(true) != 0,
                            };
                            if a_first == (func == FloatFn::Min) { a } else { b }
                        }
                    };
                    // A NaN result ignores .xorsign and .abs.
                    if xorsign_abs && !matches!(unpack(r, f), Num::Nan) { r & !f.sign(true) | sign } else { r }
                }
                _ => unreachable!("validated: {func:?} rounds"),
            }
        }
    };
    let result = if ftz { f.flush(result) } else { result };
    ptx_nan(ty, ops, result)
}

/// Whether `bits` is a NaN of the float type `ty`.
pub(super) fn is_nan(ty: DType, bits: u64) -> bool {
    let (mant, exp_bits) = match ty {
        DType::F16 => (10, 5),
        DType::BF16 => (7, 8),
        DType::F32 => (23, 8),
        DType::F64 => (52, 11),
        _ => return false,
    };
    let exp_mask = (1u64 << exp_bits) - 1;
    (bits >> mant) & exp_mask == exp_mask && bits & ((1u64 << mant) - 1) != 0
}

/// The NaN a PTX floating-point instruction returns when its result is NaN.
/// The PTX ISA leaves single-precision NaNs unspecified and says double
/// precision supports NaN payloads; this is what GPUs return (checked on
/// SM89 by `tests/gpu.rs`): single and half precision give the canonical
/// NaN (`0x7fffffff`, `0x7fff`); double precision gives the first NaN
/// operand, quieted, and an invalid operation (`0/0`, `inf - inf`,
/// `sqrt(-1)`) gives `0xfff8000000000000`. Which of two NaN operands a
/// double-precision instruction returns depends on the order the compiler
/// gives them, so a kernel should not rely on it.
pub(super) fn ptx_nan(ty: DType, operands: &[u64], result: u64) -> u64 {
    if !is_nan(ty, result) {
        return result;
    }
    match ty {
        DType::F64 => operands
            .iter()
            .find(|&&b| is_nan(DType::F64, b))
            .map_or(0xfff8_0000_0000_0000, |&b| b | 0x0008_0000_0000_0000),
        DType::F32 => 0x7fff_ffff,
        _ => 0x7fff,
    }
}

/// The add of an atomic or reduction on floats of type `ty` (`atom.add`,
/// `red.add`), from the PTX ISA's notes on them: the exact sum rounded to
/// nearest even. `ftz` (a Float32 add on global memory, which has no
/// `.noftz` before PTX 9.4) flushes subnormal inputs and results to zeros
/// of their signs; on shared memory, and in every Float16, BFloat16
/// (`.noftz`) and Float64 add, subnormals are kept. A NaN result is the
/// canonical NaN ([`ptx_nan`]).
pub(super) fn atomic_add(ty: DType, old: u64, operand: u64, ftz: bool) -> u64 {
    let f = Format::of(ty);
    let flush = |bits: u64| if ftz { f.flush(bits) } else { bits };
    let (a, b) = (flush(old), flush(operand));
    let r = flush(add(a, b, RoundingMode::NearestEven, f));
    ptx_nan(ty, &[a, b], r)
}

/// An f64 rounded to f16 (to nearest even) in one step, as `cvt.rn.f16.f64`
/// does; a NaN becomes the quiet NaN of its sign.
pub(super) fn f64_to_f16(bits: u64) -> u64 {
    let half = Format::F16;
    match unpack(bits, Format::F64) {
        Num::Nan => half.sign(bits >> 63 == 1) | 0x7e00,
        Num::Inf(neg) => half.inf(neg),
        Num::Zero(neg) => half.sign(neg),
        Num::Finite { neg, sig, exp } => round_pack(neg, sig, exp, RoundingMode::NearestEven, half),
    }
}

/// An f64 rounded to nearest even in `f` in one step (NaN to `f`'s NaN).
fn round_f64(bits: u64, f: Format) -> u64 {
    match unpack(bits, Format::F64) {
        Num::Nan => f.nan(),
        Num::Inf(neg) => f.inf(neg),
        Num::Zero(neg) => f.sign(neg),
        Num::Finite { neg, sig, exp } => round_pack(neg, sig, exp, RoundingMode::NearestEven, f),
    }
}

/// `fma.rn` on Float16 or BFloat16 (`ty`): `a * b + c` exact, then rounded
/// once to nearest even, as `fma.rn.f16` and `fma.rn.bf16` do. (Through f32,
/// a BFloat16 result would be rounded twice.) NaN results are canonical.
pub(super) fn fma_rn(ty: DType, a: u64, b: u64, c: u64) -> u64 {
    let r = fma(a, b, c, RoundingMode::NearestEven, Format::of(ty));
    ptx_nan(ty, &[a, b, c], r)
}

/// Whether `bits` is a NaN of format `f`.
fn nan_in(f: Format, bits: u64) -> bool {
    matches!(unpack(bits, f), Num::Nan)
}

/// `2^x` for a Float16 or BFloat16 `x`, rounded to nearest even in its
/// format: the value `ex2.approx` approximates. `2^x` is computed in f64,
/// whose error is far too small to move a rounding of 8 or 11 bits for any
/// half-precision `x` (`tests` checks the margin for every one).
fn exp2_rounded(x: u64, f: Format) -> u64 {
    match unpack(x, f) {
        Num::Nan => f.nan(),
        Num::Inf(false) => f.inf(false),
        Num::Inf(true) => 0,
        Num::Zero(_) => one(f),
        Num::Finite { neg, sig, exp } => {
            let magnitude = sig as f64 * 2f64.powi(exp);
            round_f64((if neg { -magnitude } else { magnitude }).exp2().to_bits(), f)
        }
    }
}

/// The NaN TMA's `nan_request_zero_fma` fill writes into a 16-bit element, which `fma` with
/// `.oob` answers with `+0`: the positive NaN with every mantissa bit set.
pub(super) const OOB_NAN: u64 = 0x7fff;

/// One element of a half-precision instruction, in format `f` (`bf16`:
/// BFloat16), from the PTX ISA: `.ftz` flushes subnormal inputs and results
/// to zeros of their signs; the result is the exact value rounded once to
/// nearest even (`.rn`, also the default); NaN results are the canonical
/// NaN; `.sat` clamps to `[0, 1]`, a NaN and `-0` giving `+0`; `.relu` makes
/// a negative result (`-0` too) `+0`; `.oob` gives `+0` when a multiplicand
/// is the out-of-bounds NaN (see [`half_inst`]). `ex2.approx` is the exact `2^a`, rounded,
/// and in BFloat16 flushes subnormal inputs and results.
///
/// `.ftz` detects a subnormal result after rounding, as an SM89 GPU does
/// (the PTX ISA does not say), the way IEEE 754 defines it: the exact value
/// rounded to the format's precision as though the exponent range were
/// unbounded is below the smallest normal number. A result just below that
/// number which rounds up to it is kept, one that only the coarser
/// subnormal spacing rounds up to it flushes.
fn half_element(func: HalfFn, f: Format, bf16: bool, mods: HalfModifiers, ops: [u64; 3]) -> u64 {
    if func == HalfFn::Ex2 {
        let x = if bf16 { f.flush(ops[0]) } else { ops[0] };
        let r = exp2_rounded(x, f);
        return if bf16 { f.flush(r) } else { r };
    }
    if mods.oob && (ops[0] == OOB_NAN || ops[1] == OOB_NAN) {
        return 0;
    }
    let flush = |bits: u64| if mods.ftz { f.flush(bits) } else { bits };
    let (a, b, c) = (flush(ops[0]), flush(ops[1]), flush(ops[2]));
    let compute = |f: Format, a: u64, b: u64, c: u64| match func {
        HalfFn::Add => add(a, b, RoundingMode::NearestEven, f),
        HalfFn::Sub => add(a, b ^ f.sign(true), RoundingMode::NearestEven, f),
        HalfFn::Fma => fma(a, b, c, RoundingMode::NearestEven, f),
        HalfFn::Ex2 => unreachable!("handled above"),
    };
    let r = compute(f, a, b, c);
    // The same in a format with the same precision and three more exponent
    // bits: an unbounded exponent range for these operands.
    let r = if mods.ftz {
        let wide = Format { mant: f.mant, exp_bits: f.exp_bits + 3 };
        let widen = |x: u64| widen_exactly(x, f, wide);
        let unbounded = compute(wide, widen(a), widen(b), widen(c));
        match unpack(unbounded, wide) {
            Num::Finite { neg, sig, exp } if exp + bit_length(sig) - 1 < 1 - f.bias() => f.sign(neg),
            _ => r,
        }
    } else {
        r
    };
    let nan = nan_in(f, r);
    let negative = r & f.sign(true) != 0;
    if mods.sat {
        return match unpack(r, f) {
            Num::Nan => 0,
            _ if negative => 0,
            Num::Inf(false) => one(f),
            Num::Finite { .. } if r > one(f) => one(f),
            _ => r,
        };
    }
    if nan {
        return f.nan();
    }
    if mods.relu && negative { 0 } else { r }
}

/// `bits` of format `f` in format `wide`, which has the same precision and
/// more exponent bits, so holds every value of `f` exactly.
fn widen_exactly(bits: u64, f: Format, wide: Format) -> u64 {
    match unpack(bits, f) {
        Num::Nan => wide.nan(),
        Num::Inf(neg) => wide.inf(neg),
        Num::Zero(neg) => wide.sign(neg),
        Num::Finite { neg, sig, exp } => round_pack(neg, sig, exp, RoundingMode::NearestEven, wide),
    }
}

/// Execute a half-precision instruction on raw operand bits: Int32s holding
/// two elements (element 0 in bits 15:0) for the packed types, each pair
/// computed independently, or single Float16 or BFloat16 values.
///
/// `.oob` is specified as a zero result where an operand is the "OOB NaN"
/// that TMA's out-of-bounds NaN fill writes, a special NaN whose encoding
/// neither the PTX ISA nor the CUDA documentation gives. The simulator
/// takes it to be [`OOB_NAN`], the pattern that fill writes on a GPU, in a
/// multiplicand (`a` or `b`). Any other NaN there gives the canonical NaN,
/// as a B200 does (CUTLASS's host reference, `guarded_multiply_add`, zeroes
/// every NaN). Whether the device also takes the pattern with the sign bit
/// set for it is not known.
pub(super) fn half_inst(func: HalfFn, ty: HalfType, mods: HalfModifiers, args: &[u64]) -> u64 {
    let (f, bf16) = (Format::of(ty.elem()), ty.elem() == DType::BF16);
    let element = |shift: u32| {
        let mut ops = [0u64; 3];
        for (k, &a) in args.iter().enumerate() {
            ops[k] = a >> shift & 0xffff;
        }
        half_element(func, f, bf16, mods, ops)
    };
    if ty.is_packed() { element(0) | element(16) << 16 } else { element(0) }
}

/// Float32 bits to BFloat16 bits as `cvt.rn.satfinite.bf16.f32` converts
/// them: rounded to nearest even; a NaN gives the canonical NaN (`0x7fff`,
/// as an SM89 GPU returns), and a magnitude above the largest finite
/// BFloat16 (`0x7f7f`) that value with the input's sign, infinities
/// included.
fn bf16_satfinite(x: u64) -> u64 {
    let f = Format::BF16;
    match unpack(x, Format::F32) {
        Num::Nan => f.nan(),
        Num::Inf(neg) => f.max_finite(neg),
        Num::Zero(neg) => f.sign(neg),
        Num::Finite { neg, sig, exp } => match round_pack(neg, sig, exp, RoundingMode::NearestEven, f) {
            r if r == f.inf(neg) => f.max_finite(neg),
            r => r,
        },
    }
}

/// Float32 bits rounded to TF32 as `cvt.rna.tf32.f32` does, giving the
/// Float32 layout with the low 13 mantissa bits cleared. Finite values round
/// to the nearest value with 10 stored mantissa bits, ties away from zero:
/// half of the dropped part (`0x1000`) is added to the magnitude and the
/// dropped bits cleared, a carry raising the exponent (to infinity above the
/// largest TF32 value). Infinities and NaNs have their low 13 bits cleared,
/// as an SM89 GPU does (the PTX ISA does not say), so a NaN whose payload is
/// only in those bits becomes an infinity of its sign.
fn tf32_rna(x: u64) -> u64 {
    crate::numeric::tf32_rna_bits(x as u32) as u64
}

fn half_format(ty: DType) -> crate::ir::FloatFormat {
    ty.float_format().expect("a float type")
}

/// The sum of two Float32 values (as `f64`s) with every bit that a rounding to
/// a narrower format could depend on: the `f64` sum made odd when it is
/// inexact (round to odd), so the rounding to Float16 or BFloat16 that follows
/// is the correct single rounding of the exact sum.
fn sum_round_to_odd(a: f64, b: f64) -> f64 {
    let s = a + b;
    if !s.is_finite() {
        return s;
    }
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    if err == 0.0 || s.to_bits() & 1 == 1 {
        return s;
    }
    // Move one step toward the exact sum, to the odd neighbour.
    let bits = s.to_bits();
    f64::from_bits(if (err > 0.0) != s.is_sign_negative() { bits + 1 } else { bits - 1 })
}

/// The reduction of a `tcgen05.ld.red` (the Float32 `bits` a thread loaded):
/// their `min` or `max`, of absolute values with `.abs`, with PTX's NaN
/// rules (a NaN is ignored unless `.NaN`, which makes any NaN the result;
/// -0 < +0).
pub(super) fn tmem_load_red(red: crate::ir::TmemLdRed, bits: impl Iterator<Item = u64>) -> u64 {
    let func = if red.op.is_min() { FloatFn::Min } else { FloatFn::Max };
    let mut acc: Option<u64> = None;
    for b in bits {
        let b = if red.op.is_abs() { b & !Format::F32.sign(true) } else { b };
        acc = Some(match acc {
            None => b,
            Some(a) => float_inst(func, FloatMode::Exact, false, red.nan, false, DType::F32, &[a, b]),
        });
    }
    acc.expect("a load reduces at least one value")
}

/// `tcgen05.ld.spcompress` on one thread's Float32 values (`bits`): of every
/// four, the two `op` prefers (the larger for the max forms, the smaller for
/// the min forms, by absolute value for `abs`; a NaN is the least preferred,
/// or with `nan` the most; the lower position wins a tie) in order, and the
/// 4-bit metadata fields (first kept position in bits 0-1, second in bits
/// 2-3) packed eight to a 32-bit register.
pub(super) fn tmem_load_spcompress(op: crate::ir::TmemLoadRedOp, nan: bool, bits: &[u64]) -> (Vec<u64>, Vec<u64>) {
    // A monotonic integer key of the value, larger the more preferred.
    let key = |b: u64| -> i64 {
        let v = f32::from_bits(b as u32);
        if v.is_nan() {
            return if nan { i64::MAX } else { i64::MIN };
        }
        let v = if op.is_abs() { v.abs() } else { v };
        let i = v.to_bits() as i32;
        let ordered = i64::from(if i < 0 { i ^ i32::MAX } else { i });
        if op.is_min() { -ordered } else { ordered }
    };
    let mut kept = Vec::with_capacity(bits.len() / 2);
    let mut meta = vec![0u64; (bits.len() / 32).max(1)];
    for (g, group) in bits.as_chunks::<4>().0.iter().enumerate() {
        let mut order: Vec<usize> = (0..4).collect();
        // Stable: a tie keeps the lower position first.
        order.sort_by_key(|&i| std::cmp::Reverse(key(group[i])));
        let (a, b) = (order[0].min(order[1]), order[0].max(order[1]));
        kept.extend([group[a], group[b]]);
        meta[g / 8] |= ((a | b << 2) as u64) << (4 * (g % 8));
    }
    (kept, meta)
}

/// Execute a packed arithmetic instruction ([`PackedOp`]) on raw operand
/// bits: Float32 pairs are `u64`s (element 0 in the low half), half pairs
/// `u32`s, and every element is computed on its own, as the scalar
/// instruction would with `rnd` and `ftz`. The mixed forms widen the half
/// operand exactly and round once.
fn packed_inst(op: PackedOp, rnd: RoundingMode, ftz: bool, args: &[u64]) -> u64 {
    let mode = FloatMode::Round(rnd);
    let lane = |x: u64, k: u32| x >> (32 * k) & 0xffff_ffff;
    let flip = |b: u64| b ^ Format::F32.sign(true);
    let f32_op = |func: FloatFn, ops: &[u64]| float_inst(func, mode, ftz, false, false, DType::F32, ops);
    let half = |bf16: bool| half_format(if bf16 { DType::BF16 } else { DType::F16 });
    let widen = |bits: u64, bf16: bool| {
        let f = Format::of(if bf16 { DType::BF16 } else { DType::F16 });
        widen_exactly(bits, f, Format::F32)
    };
    let mut out = 0u64;
    for k in 0..2u32 {
        let r = match op {
            PackedOp::Add => f32_op(FloatFn::Add, &[lane(args[0], k), lane(args[1], k)]),
            PackedOp::Sub => f32_op(FloatFn::Add, &[lane(args[0], k), flip(lane(args[1], k))]),
            PackedOp::Mul => f32_op(FloatFn::Mul, &[lane(args[0], k), lane(args[1], k)]),
            PackedOp::Fma => f32_op(FloatFn::Fma, &[lane(args[0], k), lane(args[1], k), lane(args[2], k)]),
            PackedOp::AddToHalf { bf16 } | PackedOp::SubToHalf { bf16 } => {
                let (a, b) =
                    (f32::from_bits(lane(args[0], k) as u32) as f64, f32::from_bits(lane(args[1], k) as u32) as f64);
                let b = if matches!(op, PackedOp::SubToHalf { .. }) { -b } else { b };
                encode(half(bf16), sum_round_to_odd(a, b), NarrowRounding::NearestEven, false, false) as u64
            }
            PackedOp::AddHalf { bf16 } => {
                f32_op(FloatFn::Add, &[widen(args[0] >> (16 * k) & 0xffff, bf16), lane(args[1], k)])
            }
            PackedOp::SubHalf { bf16 } => {
                f32_op(FloatFn::Add, &[widen(args[0] >> (16 * k) & 0xffff, bf16), flip(lane(args[1], k))])
            }
            PackedOp::FmaHalf { bf16 } => {
                f32_op(FloatFn::Fma, &[widen(args[0] >> (16 * k) & 0xffff, bf16), lane(args[1], k), lane(args[2], k)])
            }
        };
        let shift = if matches!(op, PackedOp::AddToHalf { .. } | PackedOp::SubToHalf { .. }) { 16 } else { 32 };
        out |= (r & if shift == 16 { 0xffff } else { 0xffff_ffff }) << (shift * k);
    }
    out
}

/// Execute a conversion with modifiers on raw operand bits.
pub(super) fn cvt_inst(func: CvtFn, args: &[u64]) -> u64 {
    match func {
        CvtFn::F32x2ToBf16x2 => bf16_satfinite(args[0]) | bf16_satfinite(args[1]) << 16,
        CvtFn::F32ToTf32 => tf32_rna(args[0]),
        // Every int8 value is a BFloat16 (8 significant bits).
        CvtFn::S8ToBf16 => round_f64((args[0] as u8 as i8 as f64).to_bits(), Format::BF16),
        // Each byte as s2f6 (its int8 value times 2^-6), times the ue8m0
        // scale (0x85: 2^(0x85 - 127) = 2^6; 0x81: 2^2); exact for the
        // scales cute-rs issues, so .satfinite never applies.
        CvtFn::S2f6x2ToBf16x2 { scale } => {
            let scale = 2f64.powi(scale as i32 - 127);
            let element = |byte: u64| {
                let s2f6 = (byte as u8 as i8) as f64 * 2f64.powi(-6);
                round_f64((s2f6 * scale).to_bits(), Format::BF16)
            };
            element(args[0] & 0xff) | element(args[0] >> 8 & 0xff) << 16
        }
        CvtFn::F32x2ToF8x2 { to, relu } => {
            let f = to.dtype().float_format().expect("float");
            let one =
                |x: u64| encode(f, f32::from_bits(x as u32) as f64, NarrowRounding::NearestEven, true, relu) as u64;
            one(args[0]) | one(args[1]) << 8
        }
        CvtFn::F16x2ToF8x2 { to, relu } => {
            let (f, h) = (to.dtype().float_format().expect("float"), half_format(DType::F16));
            let one = |x: u64| encode(f, decode(h, x as u32 & 0xffff), NarrowRounding::NearestEven, true, relu) as u64;
            one(args[0]) | one(args[0] >> 16) << 8
        }
        CvtFn::F8x2ToF16x2 { from, relu } => {
            let (f, h) = (from.dtype().float_format().expect("float"), half_format(DType::F16));
            let one = |x: u64| encode(h, decode(f, x as u32 & 0xff), NarrowRounding::NearestEven, false, relu) as u64;
            one(args[0]) | one(args[0] >> 8) << 16
        }
        CvtFn::F32x2ToUe8m0x2 { round_up } => {
            let f = DType::F8E8M0FNU.float_format().expect("float");
            let mode = if round_up { NarrowRounding::Up } else { NarrowRounding::Zero };
            let one = |x: u64| encode(f, f32::from_bits(x as u32) as f64, mode, true, false) as u64;
            one(args[0]) | one(args[1]) << 8
        }
        CvtFn::F32x2ToUe5m3x2 => {
            let f = DType::F8E5M3FNU.float_format().expect("float");
            let one = |x: u64| encode(f, f32::from_bits(x as u32) as f64, NarrowRounding::Up, true, false) as u64;
            one(args[0]) | one(args[1]) << 8
        }
        CvtFn::Packed { op, rnd, ftz } => packed_inst(op, rnd.unwrap_or(RoundingMode::NearestEven), ftz, args),
        CvtFn::Bf16x2ToUe8m0x2 { round_up, satfinite } => {
            let (f, b) = (DType::F8E8M0FNU.float_format().expect("float"), half_format(DType::BF16));
            let mode = if round_up { NarrowRounding::Up } else { NarrowRounding::Zero };
            let one = |x: u64| encode(f, decode(b, x as u32 & 0xffff), mode, satfinite, false) as u64;
            one(args[0]) | one(args[0] >> 16) << 8
        }
        CvtFn::Ue8m0x2ToBf16x2 => {
            let (f, b) = (DType::F8E8M0FNU.float_format().expect("float"), half_format(DType::BF16));
            let one = |x: u64| encode(b, decode(f, x as u32 & 0xff), NarrowRounding::NearestEven, false, false) as u64;
            one(args[0]) | one(args[0] >> 8) << 16
        }
        // FP4 pairs share a byte (element 0 in the low nibble); FP6 pairs
        // take a byte each (in its low 6 bits).
        CvtFn::F32x2ToMx { to, relu } => {
            let f = to.dtype().float_format().expect("float");
            let one =
                |x: u64| encode(f, f32::from_bits(x as u32) as f64, NarrowRounding::NearestEven, true, relu) as u64;
            let shift = if to == crate::ir::MxKind::E2M1 { 4 } else { 8 };
            one(args[0]) | one(args[1]) << shift
        }
        CvtFn::MxToF16x2 { from, relu } => {
            let (f, h) = (from.dtype().float_format().expect("float"), half_format(DType::F16));
            let (shift, mask) = if from == crate::ir::MxKind::E2M1 { (4, 0xf) } else { (8, 0x3f) };
            let one = |x: u64| encode(h, decode(f, (x & mask) as u32), NarrowRounding::NearestEven, false, relu) as u64;
            one(args[0]) | one(args[0] >> shift) << 16
        }
    }
}

/// `rcp.approx.ftz.f64` and `rsqrt.approx.ftz.f64`: the function of the
/// argument's high word (in 1.11.20 format), its result's low word cleared;
/// subnormal arguments and results flush to zero, NaN arguments give the
/// canonical NaN `0x7fffffff00000000`.
fn gross_f64(func: FloatFn, bits: u64) -> u64 {
    let f = Format::F64;
    let canonical = 0x7fff_ffff_0000_0000;
    let x = f.flush(bits) & 0xffff_ffff_0000_0000;
    let v = f64::from_bits(x);
    let r = match func {
        FloatFn::Rcp => 1.0 / v,
        _ if v < 0.0 => return canonical,
        _ => 1.0 / v.sqrt(),
    };
    if r.is_nan() {
        return canonical;
    }
    f.flush(r.to_bits()) & 0xffff_ffff_0000_0000
}

fn one(f: Format) -> u64 {
    (f.bias() as u64) << f.mant
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{BF16, F16};

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Bit patterns that cover every exponent, the subnormals and the
    /// special values.
    fn sample(state: &mut u64, f: Format) -> u64 {
        let width = f.mant + f.exp_bits + 1;
        let r = rng(state);
        let bits = r & ((1u128 << width) - 1) as u64;
        match r >> 60 {
            0 => bits & ((1 << f.mant) - 1) | (bits & f.sign(true)),
            1 => (bits & f.sign(true)) | ((f.bias() as u64 + (r >> 40) % 8) << f.mant) | (bits & 0xff),
            _ => bits,
        }
    }

    fn same(a: u64, b: u64, f: Format) -> bool {
        a == b || (matches!(unpack(a, f), Num::Nan) && matches!(unpack(b, f), Num::Nan))
    }

    /// Round-to-nearest results match the host's IEEE arithmetic, and the
    /// directed roundings bracket them.
    #[test]
    fn soft_float_matches_ieee_arithmetic() {
        let mut state = 0x9e3779b97f4a7c15u64;
        for _ in 0..200_000 {
            for f in [Format::F32, Format::F64] {
                let (a, b, c) = (sample(&mut state, f), sample(&mut state, f), sample(&mut state, f));
                let rn = RoundingMode::NearestEven;
                let host = |op: usize| -> u64 {
                    if f.mant == 23 {
                        let (x, y, z) = (f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32));
                        let r = [x + y, x * y, x.mul_add(y, z), x / y, x.sqrt()][op];
                        r.to_bits() as u64
                    } else {
                        let (x, y, z) = (f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
                        [x + y, x * y, x.mul_add(y, z), x / y, x.sqrt()][op].to_bits()
                    }
                };
                let ours = |op: usize, m: RoundingMode| -> u64 {
                    match op {
                        0 => add(a, b, m, f),
                        1 => mul(a, b, m, f),
                        2 => fma(a, b, c, m, f),
                        3 => div(a, b, m, f),
                        _ => sqrt(a, m, f),
                    }
                };
                for op in 0..5 {
                    let r = ours(op, rn);
                    assert!(same(r, host(op), f), "op {op} on {a:#x} {b:#x} {c:#x}: {r:#x} vs {:#x}", host(op));
                    let value = |bits: u64| {
                        if f.mant == 23 { f32::from_bits(bits as u32) as f64 } else { f64::from_bits(bits) }
                    };
                    let (down, up, zero) =
                        (ours(op, RoundingMode::Down), ours(op, RoundingMode::Up), ours(op, RoundingMode::Zero));
                    if value(r).is_nan() {
                        continue;
                    }
                    assert!(value(down) <= value(r) && value(r) <= value(up), "op {op} on {a:#x} {b:#x} {c:#x}");
                    assert!(down == r || up == r, "op {op} on {a:#x} {b:#x} {c:#x}: RN is RD or RU");
                    let toward_zero = if value(r) > 0.0 || (value(r) == 0.0 && value(up) > 0.0) { down } else { up };
                    if value(r) != 0.0 {
                        assert_eq!(zero, toward_zero, "op {op} on {a:#x} {b:#x} {c:#x}");
                    }
                }
            }
        }
    }

    /// The NaNs instructions return (as an SM89 GPU does).
    #[test]
    fn nan_results_follow_the_gpu() {
        let (q5, n7) = (0x7ff8_0000_0000_0005u64, 0xfff8_0000_0000_0007u64);
        let one = 1f64.to_bits();
        assert_eq!(ptx_nan(DType::F64, &[0x7ff0_0000_0000_0001, one], f64::NAN.to_bits()), 0x7ff8_0000_0000_0001);
        assert_eq!(ptx_nan(DType::F64, &[one, n7], f64::NAN.to_bits()), n7);
        assert_eq!(ptx_nan(DType::F64, &[q5, n7], f64::NAN.to_bits()), q5);
        assert_eq!(ptx_nan(DType::F64, &[0, 0], f64::NAN.to_bits()), 0xfff8_0000_0000_0000);
        assert_eq!(ptx_nan(DType::F64, &[one, one], one), one);
        assert_eq!(ptx_nan(DType::F32, &[0xffc0_0007], 0xffc0_0007), 0x7fff_ffff);
        assert_eq!(ptx_nan(DType::F16, &[0x7e05], 0x7e05), 0x7fff);
        assert_eq!(ptx_nan(DType::BF16, &[0xffc7], 0xffc7), 0x7fff);
        let r =
            float_inst(FloatFn::Div, FloatMode::Round(RoundingMode::Zero), false, false, false, DType::F64, &[0, 0]);
        assert_eq!(r, 0xfff8_0000_0000_0000);
        let r = float_inst(FloatFn::Sqrt, FloatMode::Round(RoundingMode::Up), false, false, false, DType::F64, &[n7]);
        assert_eq!(r, n7);
    }

    /// `cvt.rn.f16.f64` rounds once (through f32 it would round twice).
    #[test]
    fn f64_to_f16_rounds_once() {
        let x = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
        assert_eq!(f64_to_f16(x.to_bits()), 0x3c01);
        assert_eq!(F16::from_f32(x as f32).0, 0x3c00, "f32 first would round to the tie, then to even");
        assert_eq!(f64_to_f16(65519.99f64.to_bits()), 0x7bff);
        assert_eq!(f64_to_f16(65520f64.to_bits()), 0x7c00);
        assert_eq!(f64_to_f16((-1e300f64).to_bits()), 0xfc00);
        assert_eq!(f64_to_f16(2f64.powi(-24).to_bits()), 0x0001);
        assert_eq!(f64_to_f16(2f64.powi(-25).to_bits()), 0x0000, "a tie rounds to even");
        assert_eq!(f64_to_f16((2f64.powi(-25) * (1.0 + 2f64.powi(-30))).to_bits()), 0x0001);
        assert_eq!(f64_to_f16((-0f64).to_bits()), 0x8000);
        assert_eq!(f64_to_f16(f64::NAN.to_bits()), 0x7e00);
        assert_eq!(f64_to_f16((-f64::NAN).to_bits()), 0xfe00);
        for k in -30..16 {
            let v = 2f64.powi(k) * 1.37;
            assert_eq!(f64_to_f16(v.to_bits()), F16::from_f32(v as f32).0 as u64, "{v}: exact in f32, so one rounding");
        }
    }

    /// `rcp.approx.ftz.f64` and `rsqrt.approx.ftz.f64` as the ISA describes
    /// them: the high word in, the low word of the result cleared.
    #[test]
    fn gross_f64_approximations() {
        let approx = |func, x: f64| float_inst(func, FloatMode::Approx, true, false, false, DType::F64, &[x.to_bits()]);
        let r = approx(FloatFn::Rcp, 3.0);
        assert_eq!(r & 0xffff_ffff, 0);
        assert!((f64::from_bits(r) - 1.0 / 3.0).abs() < 1.0 / 3.0 * 2f64.powi(-19));
        let r = approx(FloatFn::Rsqrt, 2.0);
        assert_eq!(r & 0xffff_ffff, 0);
        assert!((f64::from_bits(r) - std::f64::consts::FRAC_1_SQRT_2).abs() < 2f64.powi(-20));
        assert_eq!(approx(FloatFn::Rcp, 1e-310), f64::INFINITY.to_bits(), "a subnormal flushes to zero");
        assert_eq!(approx(FloatFn::Rcp, -0.0), f64::NEG_INFINITY.to_bits());
        assert_eq!(approx(FloatFn::Rcp, f64::NEG_INFINITY), (-0f64).to_bits());
        assert_eq!(approx(FloatFn::Rcp, f64::NAN), 0x7fff_ffff_0000_0000);
        assert_eq!(approx(FloatFn::Rsqrt, -4.0), 0x7fff_ffff_0000_0000);
        assert_eq!(approx(FloatFn::Rsqrt, -0.0), f64::NEG_INFINITY.to_bits());
        assert_eq!(approx(FloatFn::Rsqrt, f64::INFINITY), 0);
    }

    /// `max` and `min` with `.xorsign.abs`, `.NaN` and `.ftz`, by the PTX
    /// ISA's pseudo-code.
    #[test]
    fn xorsign_abs_min_max() {
        let run = |func, ftz, nan, xorsign, a: u32, b: u32| {
            float_inst(func, FloatMode::Exact, ftz, nan, xorsign, DType::F32, &[a as u64, b as u64]) as u32
        };
        let (max, min) = (FloatFn::Max, FloatFn::Min);
        let bits = |x: f32| x.to_bits();
        // The magnitudes compared, the signs' exclusive or applied.
        assert_eq!(run(max, false, false, true, bits(-2.0), bits(3.0)), bits(-3.0));
        assert_eq!(run(min, false, false, true, bits(-2.0), bits(3.0)), bits(-2.0));
        assert_eq!(run(max, false, false, true, bits(-2.0), bits(-3.0)), bits(3.0));
        assert_eq!(run(max, false, false, false, bits(-2.0), bits(-3.0)), bits(-2.0));
        // Signed zeros: |-0| = |+0| = +0, then the sign.
        assert_eq!(run(max, false, false, true, 0x8000_0000, 0x8000_0000), 0);
        assert_eq!(run(max, false, false, true, 0x8000_0000, 0), 0x8000_0000);
        assert_eq!(run(min, false, false, false, 0, 0x8000_0000), 0x8000_0000);
        assert_eq!(run(max, false, false, false, 0x8000_0000, 0), 0);
        // A NaN operand is ignored, but its sign still counts; with .NaN, or
        // two NaNs, the result is the canonical NaN without a sign.
        assert_eq!(run(max, false, false, true, 0xffc0_0001, bits(2.0)), bits(-2.0));
        assert_eq!(run(min, false, false, true, bits(2.0), 0xffc0_0001), bits(-2.0));
        assert_eq!(run(max, false, true, true, 0xffc0_0001, bits(2.0)), 0x7fff_ffff);
        assert_eq!(run(max, false, false, true, 0xffc0_0001, 0xff80_0001), 0x7fff_ffff);
        assert_eq!(run(max, false, false, false, 0x7f80_0001, bits(-1.0)), bits(-1.0), "signaling NaN ignored");
        // ftz: subnormals flush to zeros of their signs first.
        assert_eq!(run(max, true, false, true, 0x8000_0001, 0x0000_0003), 0x8000_0000);
        assert_eq!(run(max, false, false, true, 0x8000_0001, 0x0000_0003), 0x8000_0003);
        assert_eq!(run(min, true, false, false, 0x0000_0005, 0x8000_0007), 0x8000_0000);
        assert_eq!(run(min, false, false, false, 0x0000_0005, 0x8000_0007), 0x8000_0007);
        // Infinities.
        assert_eq!(run(max, false, false, true, 0xff80_0000, bits(1.0)), 0xff80_0000);
        assert_eq!(run(min, false, true, true, 0xff80_0000, 0x7f80_0000), 0xff80_0000);
    }

    /// An independent reference for half-precision results, in f64 only: a
    /// value with `p` significant bits (11 for Float16, 8 for BFloat16) and
    /// smallest normal exponent `emin`, from the exact result held as an
    /// unevaluated sum `s + e` (`s` the f64 sum, `e` its error).
    struct HalfRef {
        p: i32,
        emin: i32,
        emax: i32,
        bf16: bool,
    }

    const F16_REF: HalfRef = HalfRef { p: 11, emin: -14, emax: 15, bf16: false };
    const BF16_REF: HalfRef = HalfRef { p: 8, emin: -126, emax: 127, bf16: true };

    impl HalfRef {
        fn value(&self, bits: u16) -> f64 {
            if self.bf16 { f32::from_bits((bits as u32) << 16) as f64 } else { F16(bits).to_f32() as f64 }
        }

        /// The bits of an exactly representable value.
        fn bits(&self, v: f64) -> u16 {
            if self.bf16 { ((v as f32).to_bits() >> 16) as u16 } else { F16::from_f32(v as f32).0 }
        }

        fn is_subnormal(&self, bits: u16) -> bool {
            let v = self.value(bits).abs();
            v != 0.0 && v < 2f64.powi(self.emin)
        }

        /// `s + e` (exactly: Knuth's TwoSum of two f64 values).
        fn two_sum(a: f64, b: f64) -> (f64, f64) {
            let s = a + b;
            let bb = s - a;
            (s, (a - (s - bb)) + (b - bb))
        }

        /// The exact `s + e` rounded to nearest even; `zero_sign` is the sign
        /// of an exact zero result.
        fn round(&self, s: f64, e: f64, zero_negative: bool) -> u16 {
            let sign = if zero_negative { 0x8000 } else { 0 };
            if s == 0.0 && e == 0.0 {
                return sign;
            }
            let exponent = (s.abs().log2().floor() as i32).max(self.emin);
            let ulp = 2f64.powi(exponent - (self.p - 1));
            let q = s / ulp;
            let r = if q - q.floor() == 0.5 && e != 0.0 {
                if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
            } else {
                q.round_ties_even()
            };
            let v = r * ulp;
            if v.abs() >= 2f64.powi(self.emax + 1) {
                return if v > 0.0 { self.bits(f64::INFINITY) } else { self.bits(f64::NEG_INFINITY) };
            }
            if v == 0.0 {
                return if s < 0.0 { 0x8000 } else { 0 };
            }
            self.bits(v)
        }

        /// Whether the exact nonzero value `s + e`, rounded to nearest even
        /// with `p` significant bits and an unbounded exponent range, is below
        /// the smallest normal number (IEEE 754's tininess after rounding).
        fn tiny(&self, s: f64, e: f64) -> bool {
            let smallest = 2f64.powi(self.emin);
            if s == 0.0 || s.abs() >= smallest {
                return false;
            }
            let ulp = 2f64.powi(s.abs().log2().floor() as i32 - (self.p - 1));
            let q = s / ulp;
            let r = if q - q.floor() == 0.5 && e != 0.0 {
                if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
            } else {
                q.round_ties_even()
            };
            (r * ulp).abs() < smallest
        }

        /// The PTX ISA's semantics, written again: `func` on raw operand bits
        /// with `mods`.
        fn op(&self, func: HalfFn, mods: HalfModifiers, a: u16, b: u16, c: u16) -> u16 {
            let canonical = 0x7fff;
            if mods.oob && (a == 0x7fff || b == 0x7fff) {
                return 0;
            }
            let flush = |x: u16| if mods.ftz && self.is_subnormal(x) { x & 0x8000 } else { x };
            let (a, b, c) = (flush(a), flush(b), flush(c));
            let (x, y, z) = (self.value(a), self.value(b), self.value(c));
            let neg = |bits: u16| bits & 0x8000 != 0;
            let exact = match func {
                HalfFn::Add | HalfFn::Sub => {
                    let y = if func == HalfFn::Sub { -y } else { y };
                    let (s, e) = Self::two_sum(x, y);
                    // x + y is +0 when exactly zero, unless both are -0.
                    let both_negative = neg(a) && (neg(b) != (func == HalfFn::Sub));
                    (s, e, both_negative && x == 0.0 && y == 0.0)
                }
                _ => {
                    let product = x * y;
                    let (s, e) = Self::two_sum(product, z);
                    let product_negative = neg(a) != neg(b);
                    (s, e, product == 0.0 && z == 0.0 && product_negative && neg(c))
                }
            };
            let (s, e, zero_negative) = exact;
            // Infinite and NaN results need no rounding (and TwoSum's error
            // term is NaN for them). With .ftz a result that is tiny after
            // rounding with an unbounded exponent range flushes (as an SM89
            // GPU does).
            let r = if s.is_nan() {
                canonical
            } else if s.is_infinite() {
                self.bits(s)
            } else if mods.ftz && self.tiny(s, e) {
                if s < 0.0 { 0x8000 } else { 0 }
            } else {
                self.round(s, e, zero_negative)
            };
            let r = flush(r);
            let value = self.value(r);
            if mods.sat {
                return if value.is_nan() || value <= 0.0 {
                    0
                } else if value >= 1.0 {
                    self.bits(1.0)
                } else {
                    r
                };
            }
            if value.is_nan() {
                canonical
            } else if mods.relu && neg(r) {
                0
            } else {
                r
            }
        }
    }

    /// Partners for the sweeps: signed zeros, the subnormal extremes, the
    /// normal extremes and the largest finite values, infinities, NaNs with
    /// payloads, values near 1, and values whose sums and products land on
    /// rounding boundaries.
    fn partners(r: &HalfRef) -> Vec<u16> {
        let mut v: Vec<u16> = if r.bf16 {
            vec![
                0x0000, 0x8000, 0x0001, 0x807f, 0x0080, 0x7f7f, 0x7f80, 0xff80, 0x7fc1, 0x3f80, 0xbf81, 0x3f90, 0x4049,
                0xc2c8,
            ]
        } else {
            vec![
                0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7e01, 0x3c00, 0xbc01, 0x3c80, 0x4248,
                0xd640,
            ]
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..4 {
            v.push(rng(&mut state) as u16);
        }
        v
    }

    fn sweep_mods(f16: bool, func: HalfFn) -> Vec<HalfModifiers> {
        let m = |ftz, sat, relu, oob| HalfModifiers { rn: true, oob, ftz, sat, relu };
        let mut mods = vec![m(false, false, false, false)];
        match func {
            HalfFn::Fma => {
                mods.extend([m(false, false, true, false), m(false, false, false, true), m(false, false, true, true)]);
                if f16 {
                    mods.extend([
                        m(true, false, false, false),
                        m(false, true, false, false),
                        m(true, true, false, false),
                        m(true, false, true, false),
                    ]);
                }
            }
            _ if f16 => {
                mods.extend([m(true, false, false, false), m(false, true, false, false), m(true, true, false, false)])
            }
            _ => {}
        }
        mods
    }

    /// Every Float16 and every BFloat16 value against each partner for add,
    /// sub and fma, with every modifier combination: fma takes each partner
    /// as `b` with one other, fixed partner as the addend `c` (and the
    /// scalar fma every seventh value), and with modifiers other than the
    /// default the two elements of a pair share the partners between them.
    /// The simulator's packed and scalar half-precision instructions are the
    /// independent reference's.
    #[test]
    fn half_instructions_match_a_reference_on_every_value() {
        for (r, packed, scalar) in
            [(F16_REF, HalfType::F16x2, HalfType::F16), (BF16_REF, HalfType::Bf16x2, HalfType::Bf16)]
        {
            let partners = partners(&r);
            for func in [HalfFn::Add, HalfFn::Sub, HalfFn::Fma] {
                for mods in sweep_mods(!r.bf16, func) {
                    // Every partner with the default modifiers, every other one
                    // with the rest (the second element takes the others).
                    let step = if mods == sweep_mods(!r.bf16, func)[0] { 1 } else { 2 };
                    for (k, &b) in partners.iter().enumerate().step_by(step) {
                        let c = partners[(k * 7 + 3) % partners.len()];
                        for a in 0..=0xffffu16 {
                            let want = r.op(func, mods, a, b, c);
                            let hi = a ^ 0x8000;
                            let word = |lo: u16, hi: u16| lo as u64 | (hi as u64) << 16;
                            let got = half_inst(func, packed, mods, &[word(a, hi), word(b, c), word(c, b)]);
                            let want_hi = r.op(func, mods, hi, c, b);
                            assert_eq!(
                                (got as u16, (got >> 16) as u16),
                                (want, want_hi),
                                "{} {func:?} {mods:?} on {a:#06x}/{hi:#06x}, {b:#06x}, {c:#06x}",
                                packed.ptx()
                            );
                            if func == HalfFn::Fma && a % 7 == 0 {
                                let got = half_inst(func, scalar, mods, &[a as u64, b as u64, c as u64]);
                                assert_eq!(
                                    got as u16,
                                    want,
                                    "{} {mods:?} on {a:#06x}, {b:#06x}, {c:#06x}",
                                    scalar.ptx()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `fma` on Float16 and BFloat16 rounds once. Through f32 or f64, this
    /// BFloat16 case rounds to a tie and then to even.
    #[test]
    fn half_fma_rounds_once() {
        // 1.125 * 0.90625 = 261/256, a BFloat16 midpoint, plus 2^-126.
        let (a, b, c) = (0x3f90u64, 0x3f68u64, 0x0080u64);
        assert_eq!(fma_rn(DType::BF16, a, b, c), 0x3f83);
        let through_f64 = 1.125f64.mul_add(0.90625, 2f64.powi(-126));
        assert_eq!(round_f64(through_f64.to_bits(), Format::BF16), 0x3f82, "the double rounding this avoids");
        assert_eq!(fma_rn(DType::BF16, 0x7fc1, b, c), 0x7fff, "NaN results are canonical");
        assert_eq!(fma_rn(DType::F16, 0x3c00, 0x3c00, 0x3c00), 0x4000);
    }

    /// `.ftz` detects tininess after rounding, as an SM89 GPU does: these
    /// products are just below the smallest normal Float16 and round up to it
    /// in the subnormal spacing; rounded to 11 bits with an unbounded
    /// exponent, the first (2047.45 * 2^-25) stays below it and flushes, the
    /// second (2047.75 * 2^-25) reaches it and is kept (the GPU test found
    /// both).
    #[test]
    fn ftz_detects_tininess_after_rounding() {
        let fma = |ftz: bool, a: u64, b: u64, c: u64| {
            let mods = HalfModifiers { rn: true, ftz, ..HalfModifiers::default() };
            half_inst(HalfFn::Fma, HalfType::F16, mods, &[a, b, c])
        };
        // c is a subnormal, which .ftz flushes to a zero.
        assert_eq!(fma(false, 0x98ac, 0x26d9, 0x8000), 0x8400);
        assert_eq!(fma(true, 0x98ac, 0x26d9, 0x83ff), 0x8000);
        assert_eq!(fma(false, 0x18ac, 0x26d9, 0x0000), 0x0400);
        assert_eq!(fma(true, 0x18ac, 0x26d9, 0x0001), 0x0000);
        assert_eq!(fma(false, 0x96bb, 0x28c1, 0x8000), 0x8400);
        assert_eq!(fma(true, 0x96bb, 0x28c1, 0x8000), 0x8400);
        // A result that is exactly the smallest normal is kept.
        assert_eq!(fma(true, 0x0400, 0x3c00, 0x0000), 0x0400);
        assert_eq!(
            F16_REF.op(
                HalfFn::Fma,
                HalfModifiers { rn: true, ftz: true, ..HalfModifiers::default() },
                0x98ac,
                0x26d9,
                0x83ff
            ),
            0x8000
        );
    }

    /// `2^x` computed in f64 and rounded once is the correctly rounded
    /// value for every Float16 and BFloat16 `x`: f64's `exp2` is never
    /// within 2^-40 (relative) of a midpoint of the half format, far beyond
    /// its own error. The simulator's `ex2.approx` follows: special values
    /// as the PTX ISA's table, BFloat16 flushing subnormal inputs and
    /// results.
    #[test]
    fn exp2_is_correctly_rounded_for_every_half_value() {
        for (r, f, ty) in [(F16_REF, Format::F16, HalfType::F16x2), (BF16_REF, Format::BF16, HalfType::Bf16x2)] {
            for x in 0..=0xffffu16 {
                let v = r.value(x);
                if !v.is_finite() {
                    continue;
                }
                let y = v.exp2();
                // 2^x is exact in f64 for an integer x, so its rounding is
                // exact (2^-25 is a Float16 midpoint, and ties to 0).
                if y == 0.0 || !y.is_finite() || y >= 2f64.powi(r.emax + 1) || v.fract() == 0.0 {
                    continue;
                }
                let exponent = (y.log2().floor() as i32).max(r.emin);
                let ulp = 2f64.powi(exponent - (r.p - 1));
                let q = y / ulp;
                let distance = ((q - q.floor()) - 0.5).abs() * ulp;
                assert!(distance > y * 2f64.powi(-40), "2^{v} is too close to a midpoint: {y:e}");
                let x = if r.bf16 { f.flush(x as u64) as u16 } else { x };
                let want = r.round(r.value(x).exp2(), 0.0, false);
                let want = if r.bf16 { f.flush(want as u64) as u16 } else { want };
                let got = half_inst(HalfFn::Ex2, ty, HalfModifiers::default(), &[x as u64]) as u16;
                assert_eq!(got, want, "ex2 {} of {x:#06x}", ty.ptx());
            }
            let ex2 = |x: u64| half_inst(HalfFn::Ex2, ty, HalfModifiers::default(), &[x]);
            let (inf, ninf, one) = (f.inf(false), f.inf(true), one(f));
            let ex2 = |x: u64| ex2(x) & 0xffff;
            assert_eq!([ex2(ninf), ex2(0), ex2(f.sign(true)), ex2(inf)], [0, one, one, inf]);
            assert_eq!(ex2(f.nan() - 1), 0x7fff, "NaN gives the canonical NaN");
            // Subnormal inputs: kept for Float16 (2^-tiny rounds to 1), flushed
            // for BFloat16; the second element is computed on its own.
            assert_eq!(ex2(1 << 16 | 1) & 0xffff, one);
        }
        let low = |ty, x: u64| half_inst(HalfFn::Ex2, ty, HalfModifiers::default(), &[x]) & 0xffff;
        assert_eq!(low(HalfType::Bf16x2, 0xc300), 0, "2^-128 is subnormal in BFloat16, and flushes");
        assert_eq!(low(HalfType::F16x2, 0xcc80), 0x0040, "2^-18 is subnormal in Float16, and kept");
    }

    /// The conversions, by their definitions: `.satfinite` saturates to the
    /// largest finite BFloat16 and keeps NaN as the canonical NaN; RNA rounds
    /// ties away from zero and clears 13 bits; the integer conversions are
    /// exact. Element 0 goes in the low half.
    #[test]
    fn conversions_follow_their_definitions() {
        let sat = |a: f32, b: f32| cvt_inst(CvtFn::F32x2ToBf16x2, &[a.to_bits() as u64, b.to_bits() as u64]) as u32;
        assert_eq!(sat(1.0, -2.0), 0xc000_3f80);
        assert_eq!(sat(f32::INFINITY, f32::NEG_INFINITY), 0xff7f_7f7f);
        assert_eq!(sat(f32::MAX, -f32::MAX), 0xff7f_7f7f, "values that round past the largest BFloat16");
        assert_eq!(sat(f32::from_bits(0x7f7f_7fff), 0.0), 0x0000_7f7f, "below the halfway point: rounds down");
        assert_eq!(sat(f32::from_bits(0x7fc0_0001), f32::from_bits(0xff80_0001)), 0x7fff_7fff);
        assert_eq!(sat(f32::from_bits(0x0001_8000), f32::from_bits(0x8000_0001)), 0x8000_0002, "subnormals round");
        // Every Float32 that is a BFloat16 converts to itself.
        for bits in (0..=0xffffu32).filter(|b| b & 0x7f80 != 0x7f80) {
            let x = f32::from_bits(bits << 16);
            assert_eq!(sat(x, 0.0) & 0xffff, bits);
        }
        let tf32 = |bits: u32| cvt_inst(CvtFn::F32ToTf32, &[bits as u64]) as u32;
        assert_eq!(tf32(0x3f80_1000), 0x3f80_2000, "a tie rounds away from zero");
        assert_eq!(tf32(0xbf80_1000), 0xbf80_2000);
        assert_eq!(tf32(0x3f80_0fff), 0x3f80_0000);
        assert_eq!(tf32(0x3f80_3000), 0x3f80_4000);
        assert_eq!(tf32(0x7f7f_f000), 0x7f80_0000, "past the largest TF32: infinity");
        assert_eq!(tf32(0x007f_f000), 0x0080_0000, "a subnormal rounds up to the smallest normal");
        assert_eq!(tf32(0x7fff_ffff), 0x7fff_e000, "NaNs are truncated");
        assert_eq!(tf32(0x7f80_1000), 0x7f80_0000, "even into an infinity");
        for x in [-128i8, -1, 0, 1, 77, 127] {
            assert_eq!(cvt_inst(CvtFn::S8ToBf16, &[x as u8 as u64]) as u16, BF16::from_f32(x as f32).0);
            let pair = cvt_inst(
                CvtFn::S2f6x2ToBf16x2 { scale: 0x85 },
                &[(x as u8 as u64) | ((x.wrapping_neg() as u8 as u64) << 8)],
            );
            assert_eq!(pair as u16, BF16::from_f32(x as f32).0);
            assert_eq!((pair >> 16) as u16, BF16::from_f32(x.wrapping_neg() as f32).0);
        }
    }

    #[test]
    fn directed_rounding_examples() {
        let f = Format::F32;
        let one = 1f32.to_bits() as u64;
        let tiny = 1e-30f32.to_bits() as u64;
        assert_eq!(add(one, tiny, RoundingMode::Up, f), 1.0000001f32.to_bits() as u64);
        assert_eq!(add(one, tiny, RoundingMode::Zero, f), one);
        let third = div(one, 3f32.to_bits() as u64, RoundingMode::Down, f);
        assert!(f32::from_bits(third as u32) < 1.0 / 3.0);
        assert_eq!(mul(f32::MAX.to_bits() as u64, 2f32.to_bits() as u64, RoundingMode::Zero, f), f.max_finite(false));
        let neg_one = (-1f32).to_bits() as u64;
        assert_eq!(add(one, neg_one, RoundingMode::Down, f), f.sign(true), "x - x is -0 rounding down");
        assert_eq!(add(one, neg_one, RoundingMode::Up, f), 0);
        let sub = f32::from_bits(1).to_bits() as u64;
        assert_eq!(
            float_inst(
                FloatFn::Add,
                FloatMode::Round(RoundingMode::NearestEven),
                true,
                false,
                false,
                DType::F32,
                &[sub, sub]
            ),
            0
        );
    }
}

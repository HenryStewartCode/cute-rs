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
//! NaN results follow [`ptx_nan`], for these instructions and for the
//! simulator's plain arithmetic alike.

use std::cmp::Ordering;

use crate::ir::{DType, FloatFn, FloatMode, RoundingMode};

#[derive(Clone, Copy)]
struct Format {
    /// Stored significand bits.
    mant: u32,
    exp_bits: u32,
}

impl Format {
    const F32: Format = Format { mant: 23, exp_bits: 8 };
    const F64: Format = Format { mant: 52, exp_bits: 11 };

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
pub(super) fn float_inst(func: FloatFn, mode: FloatMode, ftz: bool, nan: bool, ty: DType, args: &[u64]) -> u64 {
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
                    let (a, b) = (ops[0], ops[1]);
                    let (x, y) = (value(a), value(b));
                    match (x.is_nan(), y.is_nan()) {
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
                    }
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

/// An f64 rounded to f16 (to nearest even) in one step, as `cvt.rn.f16.f64`
/// does; a NaN becomes the quiet NaN of its sign.
pub(super) fn f64_to_f16(bits: u64) -> u64 {
    let half = Format { mant: 10, exp_bits: 5 };
    match unpack(bits, Format::F64) {
        Num::Nan => half.sign(bits >> 63 == 1) | 0x7e00,
        Num::Inf(neg) => half.inf(neg),
        Num::Zero(neg) => half.sign(neg),
        Num::Finite { neg, sig, exp } => round_pack(neg, sig, exp, RoundingMode::NearestEven, half),
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
    use crate::value::F16;

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
        let r = float_inst(FloatFn::Div, FloatMode::Round(RoundingMode::Zero), false, false, DType::F64, &[0, 0]);
        assert_eq!(r, 0xfff8_0000_0000_0000);
        let r = float_inst(FloatFn::Sqrt, FloatMode::Round(RoundingMode::Up), false, false, DType::F64, &[n7]);
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
        let approx = |func, x: f64| float_inst(func, FloatMode::Approx, true, false, DType::F64, &[x.to_bits()]);
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
            float_inst(FloatFn::Add, FloatMode::Round(RoundingMode::NearestEven), true, false, DType::F32, &[sub, sub]),
            0
        );
    }
}

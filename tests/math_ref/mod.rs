//! A high-precision reference for `cute::math`: double-double arithmetic
//! (about 106 bits) with a separate exponent, so that results far outside
//! the f64 range (`exp(-800)`, `erfc(27)`) keep their full precision.
//!
//! The functions are textbook series, Newton iterations and continued
//! fractions, written independently of the msun-derived code under test;
//! the trigonometric functions reduce their arguments exactly, with the
//! integer bits of 2/pi.

#![allow(dead_code)]

use std::cmp::Ordering;

/// An unevaluated sum `hi + lo` with `|lo| <= ulp(hi) / 2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

impl Dd {
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    pub const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };
    pub const LN2: Dd = Dd { hi: f64::from_bits(0x3fe62e42_fefa39ef), lo: f64::from_bits(0x3c7abc9e_3b39803f) };
    pub const PI: Dd = Dd { hi: f64::from_bits(0x400921fb_54442d18), lo: f64::from_bits(0x3ca1a626_33145c07) };
    pub const PI_2: Dd = Dd { hi: f64::from_bits(0x3ff921fb_54442d18), lo: f64::from_bits(0x3c91a626_33145c07) };

    pub fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    pub fn add(self, o: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, o.hi);
        let (t, f) = two_sum(self.lo, o.lo);
        let (s, e) = fast_two_sum(s, e + t);
        let (hi, lo) = fast_two_sum(s, e + f);
        Dd { hi, lo }
    }

    pub fn neg(self) -> Dd {
        Dd { hi: -self.hi, lo: -self.lo }
    }

    pub fn sub(self, o: Dd) -> Dd {
        self.add(o.neg())
    }

    pub fn mul(self, o: Dd) -> Dd {
        let p = self.hi * o.hi;
        let e = self.hi.mul_add(o.hi, -p);
        let e = e + (self.hi * o.lo + self.lo * o.hi);
        let (hi, lo) = fast_two_sum(p, e);
        Dd { hi, lo }
    }

    pub fn mul_f(self, x: f64) -> Dd {
        self.mul(Dd::new(x))
    }

    pub fn div(self, o: Dd) -> Dd {
        let q1 = self.hi / o.hi;
        let r = self.sub(o.mul_f(q1));
        let q2 = r.hi / o.hi;
        let r = r.sub(o.mul_f(q2));
        let q3 = r.hi / o.hi;
        let (hi, lo) = fast_two_sum(q1, q2);
        Dd { hi, lo }.add(Dd::new(q3))
    }

    pub fn sqrt(self) -> Dd {
        if self.hi <= 0.0 {
            return Dd::new(self.hi.sqrt());
        }
        let y = Dd::new(self.hi.sqrt());
        // Newton: y + (x - y^2) / (2y), twice.
        let y = y.add(self.sub(y.mul(y)).div(y.mul_f(2.0)));
        y.add(self.sub(y.mul(y)).div(y.mul_f(2.0)))
    }

    /// `self * 2^k`, in steps that stay inside the f64 exponent range.
    pub fn scale(self, k: i32) -> Dd {
        let (mut d, mut k) = (self, k);
        while k != 0 {
            let step = k.clamp(-1000, 1000);
            let s = 2f64.powi(step);
            d = Dd { hi: d.hi * s, lo: d.lo * s };
            k -= step;
        }
        d
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn abs(self) -> Dd {
        if self.hi < 0.0 { self.neg() } else { self }
    }

    pub fn cmp(self, o: Dd) -> Ordering {
        let d = self.sub(o);
        d.hi.partial_cmp(&0.0).unwrap_or(Ordering::Equal)
    }
}

/// `m * 2^e`, with `|m.hi|` in `[1, 2)` (or `m` zero): a double-double with
/// an unbounded exponent.
#[derive(Clone, Copy, Debug)]
pub struct Big {
    pub m: Dd,
    pub e: i64,
}

fn exponent(x: f64) -> i64 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let biased = ((b >> 52) & 0x7ff) as i64;
    if biased == 0 { exponent(x * 2f64.powi(64)) - 64 } else { biased - 1023 }
}

impl Big {
    pub fn new(d: Dd) -> Big {
        if d.hi == 0.0 {
            return Big { m: Dd::ZERO, e: 0 };
        }
        let e = exponent(d.hi);
        Big { m: d.scale(-e as i32), e }
    }

    pub fn of(x: f64) -> Big {
        Big::new(Dd::new(x))
    }

    pub fn is_zero(&self) -> bool {
        self.m.hi == 0.0
    }

    pub fn mul(self, o: Big) -> Big {
        let m = self.m.mul(o.m);
        let b = Big::new(m);
        Big { m: b.m, e: b.e + self.e + o.e }
    }

    pub fn div(self, o: Big) -> Big {
        let b = Big::new(self.m.div(o.m));
        Big { m: b.m, e: b.e + self.e - o.e }
    }

    pub fn add(self, o: Big) -> Big {
        if self.is_zero() {
            return o;
        }
        if o.is_zero() {
            return self;
        }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = a.e - b.e;
        if d > 200 {
            return a;
        }
        let s = a.m.add(b.m.scale(-d as i32));
        let r = Big::new(s);
        if r.is_zero() { r } else { Big { m: r.m, e: r.e + a.e } }
    }

    pub fn neg(self) -> Big {
        Big { m: self.m.neg(), e: self.e }
    }

    pub fn sub(self, o: Big) -> Big {
        self.add(o.neg())
    }

    pub fn scale(self, k: i64) -> Big {
        Big { m: self.m, e: self.e + k }
    }

    /// The value as a double-double (for moderate exponents; others
    /// overflow or underflow).
    pub fn dd(self) -> Dd {
        self.m.scale(self.e.clamp(-1 << 20, 1 << 20) as i32)
    }

    pub fn to_f64(self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        if self.e > 1100 {
            return self.m.hi.signum() * f64::INFINITY;
        }
        if self.e < -1200 {
            return self.m.hi.signum() * 0.0;
        }
        // Two steps keep the intermediate in range.
        let h = (self.e / 2) as i32;
        let t = self.m.to_f64() * 2f64.powi(h);
        t * 2f64.powi(self.e as i32 - h)
    }
}

/// A reference result: a value, or an exact special value the function must
/// return.
#[derive(Clone, Copy, Debug)]
pub enum Want {
    Value(Big),
    Exact(f64),
}

impl Want {
    pub fn value(d: Dd) -> Want {
        Want::Value(Big::new(d))
    }
}

/// A floating-point format: significand bits and minimum normal exponent.
#[derive(Clone, Copy, Debug)]
pub struct Format {
    pub p: i64,
    pub emin: i64,
    pub emax: i64,
}

pub const F32: Format = Format { p: 24, emin: -126, emax: 127 };
pub const F64: Format = Format { p: 53, emin: -1022, emax: 1023 };
pub const F16: Format = Format { p: 11, emin: -14, emax: 15 };
pub const BF16: Format = Format { p: 8, emin: -126, emax: 127 };

fn same_special(got: f64, want: f64) -> bool {
    (got.is_nan() && want.is_nan()) || (got == want && got.is_sign_negative() == want.is_sign_negative())
}

/// `x` rounded to nearest (ties to even) in format `f`.
pub fn round_to_format(x: f64, f: Format) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = exponent(x).max(f.emin);
    let ulp = Dd::ONE.scale((e - (f.p - 1)) as i32).hi;
    let r = (x / ulp).round_ties_even() * ulp;
    let max = (2.0 - Dd::ONE.scale(1 - f.p as i32).hi) * Dd::ONE.scale(f.emax as i32).hi;
    if r.abs() > max { x.signum() * f64::INFINITY } else { r }
}

/// The error of `got` in units in the last place of the reference value in
/// format `f`. An exact reference must be met exactly (after rounding it to
/// `f`), with the sign of a zero; otherwise the error is infinite. An
/// overflowing reference expects infinity; a result beyond the format's
/// largest finite value is an infinite error unless the reference overflows
/// too.
pub fn ulp_error(got: f64, want: Want, f: Format) -> f64 {
    let v = match want {
        Want::Exact(w) => {
            return if same_special(got, round_to_format(w, f)) { 0.0 } else { f64::INFINITY };
        }
        Want::Value(v) => v,
    };
    if got.is_nan() {
        return f64::INFINITY;
    }
    if v.is_zero() {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    // The largest finite value plus half an ulp: the overflow threshold.
    let max = Big { m: Dd::new(2.0 - 2f64.powi(-(f.p as i32))), e: f.emax };
    let magnitude = Big { m: v.m.abs(), e: v.e };
    let overflows = magnitude.sub(max).m.hi >= 0.0;
    if got.is_infinite() {
        return if overflows && got.signum() == v.m.hi.signum() { 0.0 } else { f64::INFINITY };
    }
    // (A finite result for an overflowing value is measured in ulps of the
    // largest binade.)
    let ulp_e = v.e.clamp(f.emin, f.emax) - (f.p - 1);
    // got * 2^-ulp_e and v * 2^-ulp_e are both moderate.
    let g = Big::of(got).scale(-ulp_e);
    let w = v.scale(-ulp_e);
    let d = g.sub(w);
    if d.is_zero() { 0.0 } else { d.to_f64().abs() }
}

/// The f64 bits of 2/pi, from its hexadecimal expansion.
const TWO_OVER_PI: [u32; 54] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
];

fn two_over_pi_bit(i: i64) -> u128 {
    if i < 0 {
        return 0;
    }
    let i = i as usize;
    ((TWO_OVER_PI[i / 24] >> (23 - i % 24)) & 1) as u128
}

/// `x = n * pi/2 + r` exactly enough: `n mod 4` and `r` (`|r| <= pi/4`).
pub fn reduce_pio2(x: f64) -> (i64, Dd) {
    if x.abs() <= std::f64::consts::FRAC_PI_4 {
        return (0, Dd::new(x));
    }
    let e = exponent(x);
    let m = (x.abs() * 2f64.powi((52 - e) as i32)) as u128; // 53-bit integer
    let p = e - 52; // |x| = m * 2^p
    // x * 2/pi = m * sum b_i 2^(p - i). Bits with p - i >= 2 add multiples
    // of 4; take i from p - 1 to p + 180: 2 integer bits, 180 fraction bits.
    // Accumulate the fraction in three 64-bit limbs plus the integer bits.
    let mut acc = [0u128; 4]; // acc[0]: integer part (mod 4), then 64-bit fraction limbs
    let (lo, hi) = (p - 1, p + 192);
    for i in lo..=hi {
        if two_over_pi_bit(i - 1) == 0 {
            continue;
        }
        // m * 2^(p - i): its bit at weight 2^(p - i + k) for bit k of m.
        let shift = p - i; // from 1 down to -192
        // Place m * 2^shift into the fixed-point accumulator with 192
        // fraction bits: add m << (192 + shift).
        let pos = 192 + shift;
        add_shifted(&mut acc, m, pos);
    }
    // acc as a 256-bit number: limbs 3..1 are fraction bits 64 each (limb 1
    // the most significant fraction limb), limb 0 the integer part.
    let n = (acc[0] & 3) as i64;
    let frac_hi = acc[1] as u64;
    let frac_mid = acc[2] as u64;
    let frac_lo = acc[3] as u64;
    // To the nearest quadrant.
    let (n, neg) = if frac_hi >> 63 == 1 { (n + 1, true) } else { (n, false) };
    let (a, b, c) = if neg {
        // 1 - frac
        let lo = (!frac_lo).wrapping_add(1);
        let carry = (frac_lo == 0) as u64;
        let mid = (!frac_mid).wrapping_add(carry);
        let carry = (frac_mid == 0 && carry == 1) as u64;
        let hi = (!frac_hi).wrapping_add(carry);
        (hi, mid, lo)
    } else {
        (frac_hi, frac_mid, frac_lo)
    };
    // Exact 32-bit pieces, smallest first.
    let pieces = [a >> 32, a & 0xffff_ffff, b >> 32, b & 0xffff_ffff, c >> 32, c & 0xffff_ffff];
    let f = pieces
        .iter()
        .enumerate()
        .rev()
        .fold(Dd::ZERO, |acc, (k, &w)| acc.add(Dd::new(w as f64 * 2f64.powi(-32 * (k as i32 + 1)))));
    let r = Dd::PI_2.mul(f);
    let r = if neg { r.neg() } else { r };
    let (n, r) = if x < 0.0 { (-n, r.neg()) } else { (n, r) };
    (n.rem_euclid(4), r)
}

/// Add `m * 2^pos` (m < 2^54) into a little-endian... big-endian limb
/// array: `acc[0]` integer bits (weights 2^192..), `acc[1..4]` fraction.
fn add_shifted(acc: &mut [u128; 4], m: u128, pos: i64) {
    // The 256-bit number N = acc[0] * 2^192 + acc[1] * 2^128 + acc[2] * 2^64 + acc[3],
    // with acc[1..] < 2^64. Add m << pos (pos may be negative: shifted out).
    let mut limbs = [acc[3], acc[2], acc[1], acc[0]]; // little-endian 64-bit limbs (last unbounded)
    let add_at = |limbs: &mut [u128; 4], value: u128, limb: usize| {
        let mut carry = value;
        let mut k = limb;
        while carry != 0 && k < 4 {
            let s = limbs[k] + carry;
            if k == 3 {
                limbs[k] = s;
                carry = 0;
            } else {
                limbs[k] = s & 0xffff_ffff_ffff_ffff;
                carry = s >> 64;
            }
            k += 1;
        }
    };
    if pos < 0 {
        let s = (-pos) as u32;
        if s >= 128 {
            return;
        }
        add_at(&mut limbs, m >> s, 0);
    } else {
        let limb = (pos / 64) as usize;
        let off = (pos % 64) as u32;
        let v = m << off; // < 2^118
        if limb < 4 {
            add_at(&mut limbs, v & 0xffff_ffff_ffff_ffff, limb);
        }
        if limb + 1 < 4 {
            add_at(&mut limbs, v >> 64, limb + 1);
        }
    }
    *acc = [limbs[3], limbs[2], limbs[1], limbs[0]];
}

fn factorial_series(x: Dd, start: u32, step: u32, terms: u32, alternate: bool) -> Dd {
    // sum_{k} (+-1)^k x^(start + step k) / (start + step k)!
    let mut term = Dd::ONE;
    for i in 1..=start {
        term = term.mul(x).div(Dd::new(i as f64));
    }
    let mut sum = term;
    let mut n = start;
    let x2 = if step == 2 { x.mul(x) } else { x };
    for k in 1..terms {
        let a = (n + 1) as f64;
        let b = (n + 2) as f64;
        term = if step == 2 { term.mul(x2).div(Dd::new(a * b)) } else { term.mul(x).div(Dd::new(a)) };
        n += step;
        sum = if alternate && k % 2 == 1 { sum.sub(term) } else { sum.add(term) };
    }
    sum
}

/// `e^x` for a double-double argument. Arguments beyond `+-1e6` give
/// `2^(+-2^40)`, far outside every format.
pub fn exp(x: Dd) -> Big {
    if x.hi.is_nan() {
        return Big::of(f64::NAN);
    }
    if x.hi.abs() > 1e6 {
        return Big { m: Dd::ONE, e: if x.hi > 0.0 { 1 << 40 } else { -(1 << 40) } };
    }
    let k = (x.hi / Dd::LN2.hi).round();
    let r = x.sub(Dd::LN2.mul_f(k));
    // e^r = (e^(r/1024))^1024
    let s = r.scale(-10);
    let mut y = factorial_series(s, 0, 1, 16, false);
    for _ in 0..10 {
        y = y.mul(y);
    }
    Big::new(y).scale(k as i64)
}

/// `e^x - 1`.
pub fn expm1(x: f64) -> Big {
    if x.abs() < TINY {
        return Big::of(x);
    }
    if x.abs() < 0.5 {
        // x + x^2/2! + ...
        return Big::new(factorial_series(Dd::new(x), 1, 1, 40, false));
    }
    exp(Dd::new(x)).sub(Big::of(1.0))
}

/// `ln(1 + f)` for `|f| < 0.42`: `2 atanh(u)` with `u = f / (2 + f)`,
/// accurate relative to the result however small `f` is.
fn ln1p_series(f: Dd) -> Dd {
    let u = f.div(Dd::new(2.0).add(f));
    let u2 = u.mul(u);
    let mut term = u;
    let mut sum = u;
    for k in 1..70 {
        term = term.mul(u2);
        sum = sum.add(term.div(Dd::new((2 * k + 1) as f64)));
        if term.hi.abs() < 1e-40 * sum.hi.abs() {
            break;
        }
    }
    sum.mul_f(2.0)
}

/// Natural logarithm of a positive value: `v = m 2^e` with `m` in
/// `[sqrt(1/2), sqrt(2))`, then `ln m + e ln 2`. The series for `ln m` keeps
/// full relative precision near 1, where `ln` is small.
pub fn ln_big(v: Big) -> Dd {
    let (mut m, mut e) = (v.m, v.e);
    if m.hi > std::f64::consts::SQRT_2 {
        m = m.scale(-1);
        e += 1;
    }
    ln1p_series(m.sub(Dd::ONE)).add(Dd::LN2.mul_f(e as f64))
}

pub fn ln(x: f64) -> Dd {
    ln_big(Big::of(x))
}

/// `log(1 + x)`.
pub fn log1p(x: f64) -> Dd {
    log1p_dd(Dd::new(x))
}

/// `sin(r)` and `cos(r)` for `|r| <= pi/4`.
fn sincos_small(r: Dd) -> (Dd, Dd) {
    (factorial_series(r, 1, 2, 30, true), factorial_series(r, 0, 2, 30, true))
}

/// Below this, `x (1 + O(x^2))` is `x` to far more than double-double
/// precision; the series would lose bits to subnormal intermediates.
const TINY: f64 = 1e-150;

pub fn sin(x: f64) -> Dd {
    if x.abs() < TINY {
        return Dd::new(x);
    }
    let (n, r) = reduce_pio2(x);
    let (s, c) = sincos_small(r);
    match n {
        0 => s,
        1 => c,
        2 => s.neg(),
        _ => c.neg(),
    }
}

pub fn cos(x: f64) -> Dd {
    let (n, r) = reduce_pio2(x);
    let (s, c) = sincos_small(r);
    match n {
        0 => c,
        1 => s.neg(),
        2 => c.neg(),
        _ => s,
    }
}

pub fn tan(x: f64) -> Dd {
    if x.abs() < TINY {
        return Dd::new(x);
    }
    let (n, r) = reduce_pio2(x);
    let (s, c) = sincos_small(r);
    if n % 2 == 0 { s.div(c) } else { c.div(s).neg() }
}

/// atan for a double-double argument.
pub fn atan_dd(x: Dd) -> Dd {
    if x.hi.abs() < TINY {
        return x;
    }
    if x.hi < 0.0 {
        return atan_dd(x.neg()).neg();
    }
    if x.hi > 1.0 {
        return Dd::PI_2.sub(atan_dd(Dd::ONE.div(x)));
    }
    // Halve the angle four times: atan x = 2 atan(x / (1 + sqrt(1 + x^2))).
    let mut y = x;
    for _ in 0..4 {
        y = y.div(Dd::ONE.add(Dd::ONE.add(y.mul(y)).sqrt()));
    }
    let y2 = y.mul(y);
    let mut term = y;
    let mut sum = y;
    for k in 1..40 {
        term = term.mul(y2).neg();
        sum = sum.add(term.div(Dd::new((2 * k + 1) as f64)));
    }
    sum.scale(4)
}

pub fn atan(x: f64) -> Dd {
    atan_dd(Dd::new(x))
}

pub fn asin(x: f64) -> Dd {
    if x.abs() < TINY {
        return Dd::new(x);
    }
    if x.abs() == 1.0 {
        return Dd::PI_2.mul_f(x);
    }
    // atan(x / sqrt((1 - x)(1 + x)))
    let d = Dd::new(1.0).sub(Dd::new(x)).mul(Dd::new(1.0).add(Dd::new(x))).sqrt();
    atan_dd(Dd::new(x).div(d))
}

pub fn acos(x: f64) -> Dd {
    if x == -1.0 {
        return Dd::PI;
    }
    // 2 atan(sqrt((1 - x) / (1 + x)))
    let q = Dd::new(1.0).sub(Dd::new(x)).div(Dd::new(1.0).add(Dd::new(x))).sqrt();
    atan_dd(q).scale(1)
}

/// `atan2(y, x)` for finite nonzero `x` and `y`.
pub fn atan2(y: f64, x: f64) -> Big {
    let (ay, ax) = (y.abs(), x.abs());
    let ratio = Big::of(ay).div(Big::of(ax));
    let a = if ratio.e < -900 {
        // atan(t) = t for tiny t.
        ratio
    } else if ay <= ax {
        Big::new(atan_dd(ratio.dd()))
    } else {
        Big::new(Dd::PI_2.sub(atan_dd(Big::of(ax).div(Big::of(ay)).dd())))
    };
    let a = if x < 0.0 { Big::new(Dd::PI).sub(a) } else { a };
    if y < 0.0 { a.neg() } else { a }
}

pub fn sinh(x: f64) -> Big {
    if x.abs() < TINY {
        return Big::of(x);
    }
    if x.abs() < 0.5 {
        return Big::new(factorial_series(Dd::new(x), 1, 2, 30, false));
    }
    let e = exp(Dd::new(x.abs()));
    let r = e.sub(Big::of(1.0).div(e)).scale(-1);
    if x < 0.0 { r.neg() } else { r }
}

pub fn cosh(x: f64) -> Big {
    let e = exp(Dd::new(x.abs()));
    e.add(Big::of(1.0).div(e)).scale(-1)
}

pub fn tanh(x: f64) -> Big {
    if x.abs() < TINY {
        return Big::of(x);
    }
    let a = x.abs();
    let r = if a < 0.5 {
        // sinh / cosh by series
        let s = factorial_series(Dd::new(a), 1, 2, 30, false);
        let c = factorial_series(Dd::new(a), 0, 2, 30, false);
        Big::new(s.div(c))
    } else {
        // 1 - 2 / (e^(2a) + 1)
        let e = exp(Dd::new(2.0 * a));
        Big::of(1.0).sub(Big::of(2.0).div(e.add(Big::of(1.0))))
    };
    if x < 0.0 { r.neg() } else { r }
}

pub fn asinh(x: f64) -> Dd {
    if x.abs() < TINY {
        return Dd::new(x);
    }
    let a = x.abs();
    let r = if a > 1e150 {
        ln(a).add(Dd::LN2)
    } else if a < 0.5 {
        // log1p(a + a^2 / (1 + sqrt(1 + a^2)))
        let a2 = Dd::new(a).mul(Dd::new(a));
        let arg = Dd::new(a).add(a2.div(Dd::ONE.add(Dd::ONE.add(a2).sqrt())));
        log1p_dd(arg)
    } else {
        let s = Dd::new(a).mul(Dd::new(a)).add(Dd::ONE).sqrt();
        ln_big(Big::new(Dd::new(a).add(s)))
    };
    if x < 0.0 { r.neg() } else { r }
}

/// `log(1 + u)` for a double-double `u > -1`.
fn log1p_dd(u: Dd) -> Dd {
    if u.hi.abs() < TINY {
        return u;
    }
    if u.hi.abs() < 0.25 {
        return ln1p_series(u);
    }
    ln_big(Big::new(Dd::ONE.add(u)))
}

pub fn acosh(x: f64) -> Dd {
    if x > 1e150 {
        return ln(x).add(Dd::LN2);
    }
    // log(x + sqrt((x - 1)(x + 1))), as log1p((x - 1) + sqrt(...)) near 1.
    let xm1 = Dd::new(x).sub(Dd::ONE);
    let s = xm1.mul(Dd::new(x).add(Dd::ONE)).sqrt();
    log1p_dd(xm1.add(s))
}

pub fn atanh(x: f64) -> Dd {
    if x.abs() < TINY {
        return Dd::new(x);
    }
    log1p(x).sub(log1p(-x)).scale(-1)
}

fn two_over_sqrt_pi() -> Dd {
    Dd::new(2.0).div(Dd::PI.sqrt())
}

/// erf by its Taylor series, for |x| < 3.
fn erf_series(x: f64) -> Dd {
    // 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1))
    let xd = Dd::new(x);
    let x2 = xd.mul(xd);
    let mut power = xd; // x^(2n+1) / n!
    let mut sum = xd;
    for n in 1..200 {
        power = power.mul(x2).div(Dd::new(n as f64)).neg();
        let term = power.div(Dd::new((2 * n + 1) as f64));
        sum = sum.add(term);
        if term.hi.abs() < 1e-40 * sum.hi.abs() {
            break;
        }
    }
    sum.mul(two_over_sqrt_pi())
}

/// erfc(x) for x >= 3 by its continued fraction, evaluated backwards.
fn erfc_cf(x: f64) -> Big {
    // erfc x = e^(-x^2) / sqrt(pi) * 1 / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))
    let xd = Dd::new(x);
    let mut f = xd;
    for n in (1..600).rev() {
        f = xd.add(Dd::new(n as f64 * 0.5).div(f));
    }
    let e = exp(xd.mul(xd).neg());
    e.div(Big::new(f.mul(Dd::PI.sqrt())))
}

pub fn erf(x: f64) -> Big {
    if x.abs() < TINY {
        // 2x/sqrt(pi), scaled out of the subnormal range.
        return Big::new(Dd::new(x * 2f64.powi(600)).mul(two_over_sqrt_pi())).scale(-600);
    }
    if x.abs() < 3.0 {
        Big::new(erf_series(x))
    } else {
        let r = Big::of(1.0).sub(erfc_cf(x.abs()));
        if x < 0.0 { r.neg() } else { r }
    }
}

pub fn erfc(x: f64) -> Big {
    if x >= 3.0 {
        erfc_cf(x)
    } else if x <= -3.0 {
        Big::of(2.0).sub(erfc_cf(-x))
    } else {
        Big::new(Dd::ONE.sub(erf_series(x)))
    }
}

/// `x^y` for `x > 0` (finite) and finite `y`.
pub fn pow_pos(x: f64, y: f64) -> Big {
    let l = ln(x);
    // Far beyond every format (and beyond what a double-double can hold).
    if (l.hi * y).abs() > 1e6 {
        return exp(Dd::new(l.hi * y));
    }
    exp(l.mul_f(y))
}

pub fn cbrt(x: f64) -> Dd {
    let a = x.abs();
    let b = Big::of(a);
    // cbrt(m 2^e) with e = 3q + r.
    let q = b.e.div_euclid(3);
    let r = b.e.rem_euclid(3);
    let m = b.m.scale(r as i32);
    let mut y = Dd::new(m.hi.cbrt());
    for _ in 0..2 {
        // y - (y^3 - m) / (3 y^2)
        let y2 = y.mul(y);
        y = y.sub(y2.mul(y).sub(m).div(y2.mul_f(3.0)));
    }
    let y = y.scale(q as i32);
    if x < 0.0 { y.neg() } else { y }
}

pub fn exp2(x: f64) -> Big {
    exp(Dd::LN2.mul_f(x))
}

pub fn log2(x: f64) -> Dd {
    ln(x).div(Dd::LN2)
}

pub fn log10(x: f64) -> Dd {
    ln(x).div(ln(10.0))
}

pub fn rsqrt(x: f64) -> Big {
    let b = Big::of(x);
    // 1/sqrt(m 2^e) with e even.
    let (m, e) = if b.e % 2 == 0 { (b.m, b.e) } else { (b.m.scale(1), b.e - 1) };
    Big::new(Dd::ONE.div(m.sqrt())).scale(-e / 2)
}

/// `sqrt(x)` for positive `x`, from the significand scaled to `[1, 4)` so
/// that subnormal arguments keep their precision.
pub fn sqrt(x: f64) -> Big {
    let b = Big::of(x);
    let (m, e) = if b.e % 2 == 0 { (b.m, b.e) } else { (b.m.scale(1), b.e - 1) };
    Big::new(m.sqrt()).scale(e / 2)
}

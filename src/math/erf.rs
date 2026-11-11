//! `erf` and `erfc`, ported from FreeBSD msun (`s_erf.c`, `s_erff.c`). The
//! tail of `erfc` takes one exponential of a two-part exponent instead of
//! msun's product of two, and one intermediate sum is compensated: msun's
//! `erfc` is up to 3.2 ulp off, this within the documented 2.5 (2.36 is
//! the largest error found, in Float64 near x = 1.22).
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! The float version was converted by Ian Lance Taylor, Cygnus Support.

use super::exp::exp_split;
use super::real::*;
use crate::value::{Boolean, Value, fma};

struct ErfK {
    erx: f64,
    efx: f64,
    efx8: f64,
    pp: &'static [f64],
    qq: &'static [f64],
    pa: &'static [f64],
    qa: &'static [f64],
    ra: &'static [f64],
    sa: &'static [f64],
    /// `rb` and `sb`, padded with zeros to the length of `ra` and `sa`.
    rb: &'static [f64],
    sb: &'static [f64],
}

const ERF: [ErfK; 2] = [
    ErfK {
        erx: fb(0x3f57bb00),
        efx: fb(0x3e0375d4),
        efx8: fb(0x3f8375d4),
        pp: &[fb(0x3e0375d4), fb(0xbeac0c2d), fb(0xbaf422f4)],
        qq: &[1.0, fb(0x3e9fe8f9), fb(0x3cb10140), fb(0xbb025311)],
        pa: &[fb(0x3674f993), fb(0x3ed48935), fb(0xbe566bd5), fb(0x3db1b34b)],
        qa: &[1.0, fb(0x3efdba2b), fb(0x3ebe1449), fb(0x3d20c267)],
        ra: &[fb(0xbc21e64c), fb(0xbf0b2d32), fb(0xbfd58a4d), fb(0xbf3109b2)],
        sa: &[1.0, fb(0x408f8bcd), fb(0x408374ab), fb(0x3f0dc974)],
        rb: &[fb(0xbc21a0ae), fb(0xbf0c4cfe), fb(0xbfebab07), 0.0],
        sb: &[1.0, fb(0x409be1ea), fb(0x4043305e), fb(0xbf430bec)],
    },
    ErfK {
        erx: db(0x3feb0ac1_60000000),
        efx: db(0x3fc06eba_8214db69),
        efx8: db(0x3ff06eba_8214db69),
        pp: &[
            db(0x3fc06eba_8214db68),
            db(0xbfd4cd7d_691cb913),
            db(0xbf9d2a51_dbd7194f),
            db(0xbf77a291_236668e4),
            db(0xbef8ead6_120016ac),
        ],
        qq: &[
            1.0,
            db(0x3fd97779_cddadc09),
            db(0x3fb0a54c_5536ceba),
            db(0x3f74d022_c4d36b0f),
            db(0x3f215dc9_221c1a10),
            db(0xbed09c43_42a26120),
        ],
        pa: &[
            db(0xbf6359b8_bef77538),
            db(0x3fda8d00_ad92b34d),
            db(0xbfd7d240_fbb8c3f1),
            db(0x3fd45fca_805120e4),
            db(0xbfbc6398_3d3e28ec),
            db(0x3fa22a36_599795eb),
            db(0xbf61bf38_0a96073f),
        ],
        qa: &[
            1.0,
            db(0x3fbb3e66_18eee323),
            db(0x3fe14af0_92eb6f33),
            db(0x3fb2635c_d99fe9a7),
            db(0x3fc02660_e763351f),
            db(0x3f8bedc2_6b51dd1c),
            db(0x3f888b54_5735151d),
        ],
        ra: &[
            db(0xbf843412_600d6435),
            db(0xbfe63416_e4ba7360),
            db(0xc0251e04_41b0e726),
            db(0xc04f300a_e4cba38d),
            db(0xc0644cb1_84282266),
            db(0xc067135c_ebccabb2),
            db(0xc0545265_57e4d2f2),
            db(0xc023a0ef_c69ac25c),
        ],
        sa: &[
            1.0,
            db(0x4033a6b9_bd707687),
            db(0x4061350c_526ae721),
            db(0x407b290d_d58a1a71),
            db(0x40842b19_21ec2868),
            db(0x407ad021_57700314),
            db(0x405b28a3_ee48ae2c),
            db(0x401a47ef_8e484a93),
            db(0xbfaeeff2_ee749a62),
        ],
        rb: &[
            db(0xbf843412_39e86f4a),
            db(0xbfe993ba_70c285de),
            db(0xc031c209_555f995a),
            db(0xc064145d_43c5ed98),
            db(0xc083ec88_1375f228),
            db(0xc0900461_6a2e5992),
            db(0xc07e384e_9bdc383f),
            0.0,
        ],
        sb: &[
            1.0,
            db(0x403e568b_261d5190),
            db(0x40745cae_221b9f0a),
            db(0x409802eb_189d5118),
            db(0x40a8ffb7_688c246a),
            db(0x40a3f219_cedf3be6),
            db(0x407da874_e79fe763),
            db(0xc03670e2_42712d62),
            0.0,
        ],
    },
];

/// Horner's rule with coefficients chosen at run time.
fn horner_sel<T: Real>(x: Value<T>, first: Boolean, a: &[f64], b: &[f64]) -> Value<T> {
    let coeff = |i: usize| sel(first, c::<T>(a[i]), c::<T>(b[i]));
    (0..a.len() - 1).rev().fold(coeff(a.len() - 1), |acc, i| acc * x + coeff(i))
}

/// The pieces both functions share, for `|x|` in each of msun's ranges.
struct Parts<T> {
    /// `pp(z)/qq(z)`, `z = x^2` (|x| < 0.84375).
    y: Value<T>,
    /// `pa(s)/qa(s)`, `s = |x| - 1` (0.84375 <= |x| < 1.25).
    p_q: Value<T>,
    /// `exp(-z*z - 0.5625 + (z-x)*(z+x) + R/S) / |x|`, the tail (erfc of
    /// `|x|` for 1.25 <= |x| < 28, f32: 11).
    r: Value<T>,
}

fn parts<T: Real>(x: Value<T>, boundary: i32) -> Parts<T> {
    let k = &ERF[T::DOUBLE as usize];
    let ix = hi(x) & 0x7fffffff;
    let z = x * x;
    let y = horner(z, k.pp) / horner(z, k.qq);
    let ax = fabs(x);
    let s = ax - T::of(1.0);
    let p_q = horner(s, k.pa) / horner(s, k.qa);
    // The tail is only used for 1.25 <= |x| < 28 (f32: 11).
    let xt = guard(ax.ge(T::of(1.0)) & ax.lt(T::of(32.0)), ax, 2.0);
    let s = c::<T>(1.0) / (xt * xt);
    let first = ix.lt(boundary);
    let rr = horner_sel(s, first, k.ra, k.rb);
    let ss = horner_sel(s, first, k.sa, k.sb);
    let zt = trunc_bits(xt, 0xffffe000);
    // msun computes exp(-z*z - 0.5625) * exp((z - x)(z + x) + R/S): two
    // exponentials, each up to 0.9 ulp off, and their product. Here the
    // exponent is one sum, of an exact part (z has few bits) and a small
    // one, whose rounding error goes into the kernel's tail; the quotient
    // by x is taken before scaling, so results that underflow round once.
    let a = fneg(zt) * zt - T::of(0.5625);
    let b = (zt - xt) * (zt + xt) + rr / ss;
    let s = a + b;
    let err = b - (s - a);
    let (e, k) = exp_split(s, err);
    Parts { y, p_q, r: scale(e / xt, k) }
}

/// The error function.
pub(crate) fn erf<T: Real>(x: Value<T>) -> Value<T> {
    let k = &ERF[T::DOUBLE as usize];
    let (w_084, w_tiny, w_sub, w_125, w_big, w_035) = pick::<T, _>([
        (0x3f580000, 0x38800000, 0x04000000, 0x3fa00000, 0x40800000, 0x4036db8c),
        (0x3feb0000, 0x3e300000, 0x00800000, 0x3ff40000, 0x40180000, 0x4006db6e),
    ]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let neg = hx.lt(0);
    let p = parts(x, w_035);
    let r_small = x + x * p.y;
    let r_tiny = sel(ix.lt(w_sub), (x * T::of(8.0) + x * T::of(k.efx8)) / T::of(8.0), x + x * T::of(k.efx));
    let r_small = sel(ix.lt(w_tiny), r_tiny, r_small);
    let r_mid = negate_if(neg, c::<T>(k.erx) + p.p_q);
    let r_tail = negate_if(neg, c::<T>(1.0) - p.r);
    let r = sel(ix.lt(w_084), r_small, sel(ix.lt(w_125), r_mid, r_tail));
    let r = sel(ix.ge(w_big), negate_if(neg, c::<T>(1.0)), r);
    canonical(sel(is_nan(x), nan::<T>(), r))
}

/// The complementary error function `1 - erf(x)`, accurate where it is
/// small.
pub(crate) fn erfc<T: Real>(x: Value<T>) -> Value<T> {
    let k = &ERF[T::DOUBLE as usize];
    let (w_084, w_tiny, w_quarter, w_125, w_huge, w_035, w_neg_big) = pick::<T, _>([
        (0x3f580000, 0x33800000, 0x3e800000, 0x3fa00000, 0x41300000, 0x4036db8c, 0x40a00000),
        (0x3feb0000, 0x3c700000, 0x3fd00000, 0x3ff40000, 0x403c0000, 0x4006db6d, 0x40180000),
    ]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let neg = hx.lt(0);
    let one = c::<T>(1.0);
    let p = parts(x, w_035);
    let r_quarter = one - (x + x * p.y);
    // msun's 0.5 - ((x*y) + (x - 0.5)) rounds the inner sum, which costs up
    // to 2.5 ulp near x = 0.82; its rounding error and x*y's are carried
    // separately (x - 0.5 is exact).
    let xy = x * p.y;
    let xy_err = fma(x, p.y, fneg(xy));
    let h = x - T::of(0.5);
    let s = xy + h;
    let bb = s - xy;
    let s_err = (xy - (s - bb)) + (h - bb);
    let r_half = (c::<T>(0.5) - s) - (s_err + xy_err);
    let r_small = sel(hx.lt(w_quarter), r_quarter, r_half);
    let r_small = sel(ix.lt(w_tiny), one - x, r_small);
    let r_mid = sel(neg, one + (c::<T>(k.erx) + p.p_q), (one - T::of(k.erx)) - p.p_q);
    let r_tail = sel(hx.gt(0), p.r, c::<T>(2.0) - p.r);
    let r_tail = sel(neg & ix.ge(w_neg_big), c::<T>(2.0), r_tail);
    let r_far = sel(hx.gt(0), c::<T>(0.0), c::<T>(2.0));
    let r = sel(ix.lt(w_084), r_small, sel(ix.lt(w_125), r_mid, sel(ix.lt(w_huge), r_tail, r_far)));
    canonical(sel(is_nan(x), nan::<T>(), r))
}

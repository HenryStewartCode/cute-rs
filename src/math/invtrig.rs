//! `asin`, `acos`, `atan` and `atan2`, ported from FreeBSD msun (`e_asin.c`,
//! `e_acos.c`, `e_acosf.c`, `s_atan.c`, `s_atanf.c`, `e_atan2.c`,
//! `e_atan2f.c`) and, for `asinf`, from the fdlibm float version newlib
//! ships (`ef_asin.c`; msun's computes in double precision). `atan2` passes
//! the rounding error of `y / x` into `atan`, which takes it before its
//! final rounding (msun's is up to 1.45 ulp off, this 1.2).
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! The float versions were converted by Ian Lance Taylor, Cygnus Support.

use super::real::*;
use crate::value::{Value, fma};

/// `pS0..pS5` and `qS1..qS4` of `e_asin.c` (f32: `ef_asin.c`).
const ASIN_P: [&[f64]; 2] = [
    &[fb(0x3e2aaaab), fb(0xbea6b090), fb(0x3e4e0aa8), fb(0xbd241146), fb(0x3a4f7f04), fb(0x3811ef08)],
    &[
        db(0x3fc55555_55555555),
        db(0xbfd4d612_03eb6f7d),
        db(0x3fc9c155_0e884455),
        db(0xbfa48228_b5688f3b),
        db(0x3f49efe0_7501b288),
        db(0x3f023de1_0dfdf709),
    ],
];
const ASIN_Q: [&[f64]; 2] = [
    &[1.0, fb(0xc019d139), fb(0x4001572d), fb(0xbf303361), fb(0x3d9dc62e)],
    &[1.0, db(0xc0033a27_1c8a2d4b), db(0x40002ae5_9c598ac8), db(0xbfe6066c_1b8d0159), db(0x3fb3b8c5_b12e9282)],
];

/// The rational approximation `R(t) = p(t)/q(t)` shared by asin and acos.
fn ratio<T: Real>(t: Value<T>, p: &[f64], q: &[f64]) -> Value<T> {
    (t * horner(t, p)) / horner(t, q)
}

/// Arc sine.
pub(crate) fn asin<T: Real>(x: Value<T>) -> Value<T> {
    let (p, q) = (pick::<T, _>(ASIN_P), pick::<T, _>(ASIN_Q));
    let (pio2_hi, pio2_lo, pio4_hi) = pick::<T, _>([
        (fb(0x3fc90fdb), fb(0xb33bbd2e), fb(0x3f490fdb)),
        (db(0x3ff921fb_54442d18), db(0x3c91a626_33145c07), db(0x3fe921fb_54442d18)),
    ]);
    let (pio2_hi, pio2_lo, pio4_hi) = (T::of(pio2_hi), T::of(pio2_lo), T::of(pio4_hi));
    let (w_tiny, w_975, mask) =
        pick::<T, _>([(0x32000000, 0x3f79999a, 0xfffff000u32), (0x3e500000, 0x3fef3333, 0xfffff000)]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let one_word = pick::<T, _>([0x3f800000, 0x3ff00000]);
    let half_word = pick::<T, _>([0x3f000000, 0x3fe00000]);
    // |x| < 0.5
    let t = x * x;
    let r_small = x + x * ratio(t, p, q);
    let r_small = sel(ix.lt(w_tiny), x, r_small);
    // 0.5 <= |x| < 1
    let w = c::<T>(1.0) - fabs(x);
    let t = w * T::of(0.5);
    let rr = ratio(t, p, q);
    let s = t.sqrt();
    let r_975 = Value::constant(pio2_hi) - ((s + s * rr) * T::of(2.0) - pio2_lo);
    let w = trunc_bits(s, mask);
    let cc = (t - w * w) / (s + w);
    let p2 = s * T::of(2.0) * rr - (Value::constant(pio2_lo) - cc * T::of(2.0));
    let q2 = Value::constant(pio4_hi) - w * T::of(2.0);
    let r_mid = Value::constant(pio4_hi) - (p2 - q2);
    let r_big = negate_if(hx.le(0), sel(ix.ge(w_975), r_975, r_mid));
    let r = sel(ix.lt(half_word), r_small, r_big);
    // asin(+-1) = +-pi/2; |x| > 1 and NaN give NaN.
    let r = sel(ix.eq(one_word) & lo(x).eq(0u32), x * pio2_hi + x * pio2_lo, r);
    let r = sel(fabs(x).gt(T::of(1.0)) | is_nan(x), nan::<T>(), r);
    canonical(r)
}

/// `pS*`, `qS*` of `e_acos.c` and `e_acosf.c` (whose fit is shorter).
const ACOS_P: [&[f64]; 2] = [&[fb(0x3e2aaaab), fb(0xbdf4c1d1), fb(0x3bb33de9)], ASIN_P[1]];
const ACOS_Q: [&[f64]; 2] = [&[1.0, fb(0xbf956240), fb(0x3e9489f9)], ASIN_Q[1]];

/// Arc cosine.
pub(crate) fn acos<T: Real>(x: Value<T>) -> Value<T> {
    let (p, q) = (pick::<T, _>(ACOS_P), pick::<T, _>(ACOS_Q));
    let (pi, pio2_hi, pio2_lo) = pick::<T, _>([
        (fb(0x40490fda), fb(0x3fc90fda), fb(0x33a22168)),
        (db(0x400921fb_54442d18), db(0x3ff921fb_54442d18), db(0x3c91a626_33145c07)),
    ]);
    let (pi, pio2_hi, pio2_lo) = (T::of(pi), T::of(pio2_hi), T::of(pio2_lo));
    let (w_tiny, mask) = pick::<T, _>([(0x32800000, 0xfffff000u32), (0x3c600000, 0xfffff000)]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let one = c::<T>(1.0);
    // |x| < 0.5
    let z = x * x;
    let r_small = Value::constant(pio2_hi) - (x - (Value::constant(pio2_lo) - x * ratio(z, p, q)));
    let r_small = sel(ix.le(w_tiny), Value::constant(pio2_hi) + pio2_lo, r_small);
    // x < -0.5
    let z = (one + x) * T::of(0.5);
    let s = z.sqrt();
    let w = ratio(z, p, q) * s - pio2_lo;
    let r_neg = Value::constant(pi) - (s + w) * T::of(2.0);
    // x > 0.5
    let z = (one - x) * T::of(0.5);
    let s = z.sqrt();
    let df = trunc_bits(s, mask);
    let cc = (z - df * df) / (s + df);
    let w = ratio(z, p, q) * s + cc;
    let r_pos = (df + w) * T::of(2.0);
    let half_word = pick::<T, _>([0x3f000000, 0x3fe00000]);
    let r = sel(ix.lt(half_word), r_small, sel(hx.lt(0), r_neg, r_pos));
    // acos(1) = 0, acos(-1) = pi; |x| > 1 and NaN give NaN.
    let r = sel(x.eq(T::of(1.0)), c::<T>(0.0), r);
    let r = sel(x.eq(T::of(-1.0)), Value::constant(pi) + Value::constant(pio2_lo) * T::of(2.0), r);
    let r = sel(fabs(x).gt(T::of(1.0)) | is_nan(x), nan::<T>(), r);
    canonical(r)
}

/// `atanhi`, `atanlo` (atan of 0.5, 1, 1.5, inf) and `aT` of `s_atan.c`.
const ATAN_HI: [[f64; 4]; 2] = [
    [fb(0x3eed6338), fb(0x3f490fda), fb(0x3f7b985e), fb(0x3fc90fda)],
    [db(0x3fddac67_0561bb4f), db(0x3fe921fb_54442d18), db(0x3fef730b_d281f69b), db(0x3ff921fb_54442d18)],
];
const ATAN_LO: [[f64; 4]; 2] = [
    [fb(0x31ac3769), fb(0x33222168), fb(0x33140fb4), fb(0x33a22168)],
    [db(0x3c7a2b7f_222f65e2), db(0x3c81a626_33145c07), db(0x3c700788_7af0cbbd), db(0x3c91a626_33145c07)],
];
const ATAN_T: [&[f64]; 2] = [
    &[3.3333328366e-01, -1.9999158382e-01, 1.4253635705e-01, -1.0648017377e-01, 6.1687607318e-02],
    &[
        db(0x3fd55555_5555550d),
        db(0xbfc99999_9998ebc4),
        db(0x3fc24924_920083ff),
        db(0xbfbc71c6_fe231671),
        db(0x3fb745cd_c54c206e),
        db(0xbfb3b0f2_af749a6d),
        db(0x3fb10d66_a0d03d51),
        db(0xbfadde2d_52defd9a),
        db(0x3fa97b4b_24760deb),
        db(0xbfa2b444_2c6a6c2f),
        db(0x3f90ad3a_e322da11),
    ],
];

/// Arc tangent.
pub(crate) fn atan<T: Real>(x: Value<T>) -> Value<T> {
    // -0 changes nothing.
    atan_tail(x, c::<T>(-0.0))
}

/// `atan(x + tail)` for a tail much smaller than `ulp(x)` (`atan2`'s
/// quotient's rounding error times `1 / (1 + x^2)`): msun's `atan`, with
/// the tail added before its final rounding.
fn atan_tail<T: Real>(x: Value<T>, tail: Value<T>) -> Value<T> {
    let (hi_t, lo_t, at) = (pick::<T, _>(ATAN_HI), pick::<T, _>(ATAN_LO), pick::<T, _>(ATAN_T));
    let (w_huge, w_tiny, w_7_16, w_11_16, w_19_16, w_39_16) = pick::<T, _>([
        (0x4c800000, 0x39800000, 0x3ee00000, 0x3f300000, 0x3f980000, 0x401c0000),
        (0x44100000, 0x3e400000, 0x3fdc0000, 0x3fe60000, 0x3ff30000, 0x40038000),
    ]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let ax = fabs(x);
    let one = c::<T>(1.0);
    // id = -1 .. 3: the argument reduction, as one division.
    let id0 = ix.lt(w_11_16);
    let id1 = ix.lt(w_19_16);
    let id2 = ix.lt(w_39_16);
    let num = sel(id0, ax * T::of(2.0) - one, sel(id1, ax - one, sel(id2, ax - T::of(1.5), c::<T>(-1.0))));
    let den = sel(id0, c::<T>(2.0) + ax, sel(id1, ax + one, sel(id2, one + ax * T::of(1.5), ax)));
    let direct = ix.lt(w_7_16);
    let xr = sel(direct, x, num / den);
    let z = xr * xr;
    let w = z * z;
    let even: Vec<f64> = at.iter().step_by(2).copied().collect();
    let odd: Vec<f64> = at.iter().skip(1).step_by(2).copied().collect();
    let s1 = z * horner(w, &even);
    let s2 = w * horner(w, &odd);
    let r_direct = xr - (xr * (s1 + s2) - tail);
    let pick4 = |t: [f64; 4]| sel(id0, c::<T>(t[0]), sel(id1, c::<T>(t[1]), sel(id2, c::<T>(t[2]), c::<T>(t[3]))));
    let (ahi, alo) = (pick4(hi_t), pick4(lo_t));
    let z = ahi - ((xr * (s1 + s2) - (alo + tail)) - xr);
    let r_reduced = negate_if(hx.lt(0), z);
    let r = sel(direct, sel(ix.lt(w_tiny), x, r_direct), r_reduced);
    // |x| >= 2^66 (f32: 2^26): +-pi/2.
    let edge = negate_if(hx.lt(0), c::<T>(hi_t[3]) + T::of(lo_t[3]));
    let r = sel(ix.ge(w_huge), edge, r);
    canonical(sel(is_nan(x), nan::<T>(), r))
}

/// `atan2(y, x)`: the angle of the point `(x, y)`.
pub(crate) fn atan2<T: Real>(y: Value<T>, x: Value<T>) -> Value<T> {
    let (pi_o_4, pi_o_2, pi, pi_lo) = pick::<T, _>([
        (fb(0x3f490fdb), fb(0x3fc90fdb), fb(0x40490fdb), fb(0xb3bbbd2e)),
        (db(0x3fe921fb_54442d18), db(0x3ff921fb_54442d18), db(0x400921fb_54442d18), db(0x3ca1a626_33145c07)),
    ]);
    let (pi_o_4, pi_o_2, pi, pi_lo) = (c::<T>(pi_o_4), c::<T>(pi_o_2), c::<T>(pi), T::of(pi_lo));
    let (shift, kmax, inf_word) = pick::<T, _>([(23, 26, 0x7f800000), (20, 60, 0x7ff00000)]);
    let (hx, hy) = (hi(x), hi(y));
    let (ix, iy) = (hx & 0x7fffffff, hy & 0x7fffffff);
    let ysign = hy.lt(0);
    let xneg = hx.lt(0);
    // m = 2*sign(x) + sign(y)
    let zero = c::<T>(0.0);
    // y/x in the general case.
    let k = (iy - ix) >> shift;
    let ratio = fabs(y / x);
    // The quotient's rounding error, exact with fma away from the ends of
    // the exponent range, enters atan as its derivative times the error
    // (msun rounds it, costing up to 0.6 ulp).
    // Both magnitudes are scaled by a power of two (atan2 does not change)
    // into a range where the residual cannot underflow or overflow.
    let (ay, ax) = (fabs(y), fabs(x));
    let m = sel(ax.gt(ay), ax, ay);
    let (small, large, up, down) = pick::<T, _>([
        (2f64.powi(-60), 2f64.powi(60), 2f64.powi(64), 2f64.powi(-64)),
        (2f64.powi(-500), 2f64.powi(500), 2f64.powi(600), 2f64.powi(-600)),
    ]);
    let s = sel(m.gt(T::of(large)), c::<T>(down), sel(m.lt(T::of(small)), c::<T>(up), c::<T>(1.0)));
    let (sx, sy) = (ax * s, ay * s);
    let (lo_ok, hi_ok) = pick::<T, _>([(2f64.powi(-100), 2f64.powi(100)), (2f64.powi(-900), 2f64.powi(900))]);
    let in_range = |v: Value<T>| v.ge(T::of(lo_ok)) & v.le(T::of(hi_ok));
    let usable = in_range(ratio) & in_range(sx) & in_range(sy);
    let r = guard(usable, ratio, 1.0);
    let residual = fma(fneg(r), guard(usable, sx, 1.0), guard(usable, sy, 1.0)) / guard(usable, sx, 1.0);
    let tail = sel(usable, residual / (c::<T>(1.0) + r * r), c::<T>(-0.0));
    let z = atan_tail(guard(!is_nan(ratio) & ratio.lt(T::of(f64::INFINITY)), ratio, 0.0), tail);
    let z = sel(k.gt(kmax), pi_o_2 + Value::constant(pi_lo) * T::of(0.5), sel(xneg & k.lt(-kmax), zero, z));
    // With |y/x| > 2^60 the angle is pi/2 whichever sign x has.
    let xneg_eff = xneg & !k.gt(kmax);
    let general = sel(xneg_eff, sel(ysign, (z - pi_lo) - pi, pi - (z - pi_lo)), negate_if(ysign, z));
    let signed_pi_o_2 = negate_if(ysign, pi_o_2);
    let signed_pi = negate_if(ysign, pi);
    // x = +-inf
    let x_inf = ix.eq(inf_word) & lo(x).eq(0u32);
    let y_inf = iy.eq(inf_word) & lo(y).eq(0u32);
    let both_inf = sel(xneg, negate_if(ysign, pi_o_4 * T::of(3.0)), negate_if(ysign, pi_o_4));
    let x_inf_r = sel(y_inf, both_inf, sel(xneg, signed_pi, negate_if(ysign, zero)));
    let r = general;
    let r = sel(y_inf, signed_pi_o_2, r);
    let r = sel(x_inf, x_inf_r, r);
    // x = 0 (y nonzero): +-pi/2; y = 0: +-0 or +-pi.
    let r = sel(x.eq(T::of(0.0)), signed_pi_o_2, r);
    let r = sel(y.eq(T::of(0.0)), sel(xneg, signed_pi, y), r);
    // x = 1: atan(y).
    let r = sel(x.eq(T::of(1.0)), atan(y), r);
    canonical(sel(is_nan(x) | is_nan(y), nan::<T>(), r))
}

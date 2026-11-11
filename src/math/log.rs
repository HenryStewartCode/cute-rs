//! `log`, `log2`, `log10`, `log1p`, `asinh`, `acosh` and `atanh`, ported
//! from FreeBSD msun (`e_log.c`, `e_logf.c`, `k_log.h`, `k_logf.h`,
//! `e_log2.c`, `e_log2f.c`, `e_log10.c`, `e_log10f.c`, `s_log1p.c`,
//! `s_log1pf.c`, `s_asinh.c`, `s_asinhf.c`, `e_acosh.c`, `e_acoshf.c`,
//! `e_atanh.c`, `e_atanhf.c`). `asinh`, `acosh` and `atanh` compute the
//! arguments of their logarithms in extra precision and pass the tail into
//! the correction term `log1p` already has for the rounding of `1 + x`:
//! msun's versions are up to 1.7 to 2 ulp off, these less than 1.
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
use crate::value::{Int32, Value, fma};

/// The word layout msun's logarithms work on: the high word of an f64, or
/// the word of an f32, `[f32, f64]`.
#[derive(Clone, Copy)]
struct Words {
    /// Significand bits in the word.
    shift: i32,
    mant_mask: i32,
    implicit: i32,
    one: i32,
    min_normal: i32,
    infinity: i32,
    /// Scaling of subnormals: `2^sub`.
    sub: i32,
}

const WORDS: [Words; 2] = [
    Words {
        shift: 23,
        mant_mask: 0x007fffff,
        implicit: 0x00800000,
        one: 0x3f800000,
        min_normal: 0x00800000,
        infinity: 0x7f800000,
        sub: 25,
    },
    Words {
        shift: 20,
        mant_mask: 0x000fffff,
        implicit: 0x00100000,
        one: 0x3ff00000,
        min_normal: 0x00100000,
        infinity: 0x7ff00000,
        sub: 54,
    },
];

fn words<T: Real>() -> Words {
    pick::<T, _>(WORDS)
}

/// `ln2_hi`, `ln2_lo` of `e_log.c` and `s_log1p.c`.
const LN2: [(f64, f64); 2] = [(fb(0x3f317180), fb(0x3717f7d1)), (db(0x3fe62e42_fee00000), db(0x3dea39ef_35793c76))];

/// `Lg1..`: odd-indexed (`t2`) and even-indexed (`t1`) coefficients.
const LG_ODD: [&[f64]; 2] = [
    &[xp(0xaaaaaa, -24), xp(0x91e9ee, -25)],
    &[db(0x3fe55555_55555593), db(0x3fd24924_94229359), db(0x3fc74664_96cb03de), db(0x3fc2f112_df3e5244)],
];
const LG_EVEN: [&[f64]; 2] = [
    &[xp(0xccce13, -25), xp(0xf89e26, -26)],
    &[db(0x3fd99999_9997fa04), db(0x3fcc71c5_1d8e78af), db(0x3fc39a09_d078c69f)],
];

/// Split a positive finite `x` into `2^k * (1 + f)` with `sqrt(2)/2 <=
/// 1 + f < sqrt(2)`; also returns the normalized significand word.
fn reduce<T: Real>(x: Value<T>, magic: i32) -> (Int32, Value<T>, Int32) {
    let wd = words::<T>();
    let sub = hi(x).lt(wd.min_normal);
    let xs = sel(sub, x * T::of(2f64.powi(wd.sub)), x);
    let hx = hi(xs);
    let k = sel(sub, Int32::from(-wd.sub), Int32::from(0)) + (hx >> wd.shift) - T::BIAS;
    let hx = hx & wd.mant_mask;
    let i = (hx + magic) & wd.implicit;
    let xn = with_hi(xs, hx | (i ^ wd.one));
    let k = k + (i >> wd.shift);
    (k, xn - T::of(1.0), hx)
}

/// `k_log1p(f) = s*(hfsq+R)` (`k_log.h`, `k_logf.h`).
fn k_log1p<T: Real>(f: Value<T>) -> Value<T> {
    let s = f / (c::<T>(2.0) + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * horner(w, pick::<T, _>(LG_EVEN));
    let t2 = z * horner(w, pick::<T, _>(LG_ODD));
    let r = t2 + t1;
    let hfsq = f * T::of(0.5) * f;
    s * (hfsq + r)
}

/// The argument is positive and finite: a safe stand-in for the others,
/// whose results the caller selects.
fn positive_finite<T: Real>(x: Value<T>) -> (Value<T>, crate::value::Boolean) {
    let ok = x.gt(T::of(0.0)) & x.lt(T::of(f64::INFINITY));
    (guard(ok, x, 1.0), ok)
}

/// The special values of every logarithm: `log(+-0) = -inf`, `log(x < 0)
/// = NaN`, `log(+inf) = +inf`, `log(NaN) = NaN`.
fn log_specials<T: Real>(x: Value<T>, ok: crate::value::Boolean, r: Value<T>) -> Value<T> {
    let r = sel(ok, r, nan::<T>());
    let r = sel(x.eq(T::of(0.0)), inf::<T>(true), r);
    let r = sel(x.eq(T::of(f64::INFINITY)), x, r);
    canonical(r)
}

/// Natural logarithm (`e_log.c`).
pub(crate) fn log<T: Real>(x: Value<T>) -> Value<T> {
    let (xs, ok) = positive_finite(x);
    let magic = pick::<T, _>([0x95f64 << 3, 0x95f64]);
    let (k, f, hx) = reduce(xs, magic);
    let (ln2_hi, ln2_lo) = pick::<T, _>(LN2);
    let dk = k.to::<T>();
    // -2**-20 <= f < 2**-20 (f64), -2**-9 <= f < 2**-9 (f32).
    let (add, below) = pick::<T, _>([(0x8000, 0xc000), (2, 3)]);
    let small = ((hx + add) & words::<T>().mant_mask).lt(below);
    let r0 = f * f * (c::<T>(0.5) - f * T::of(db(0x3fd55555_55555555)));
    let r_small = dk * T::of(ln2_hi) - ((r0 - dk * T::of(ln2_lo)) - f);
    let r_small = sel(f.eq(T::of(0.0)), dk * T::of(ln2_hi) + dk * T::of(ln2_lo), r_small);
    let s = f / (c::<T>(2.0) + f);
    let z = s * s;
    let w = z * z;
    let (i_lo, j_hi) = pick::<T, _>([(0x6147a << 3, 0x6b851 << 3), (0x6147a, 0x6b851)]);
    let i = (hx - i_lo) | (Int32::from(j_hi) - hx);
    let t1 = w * horner(w, pick::<T, _>(LG_EVEN));
    let t2 = z * horner(w, pick::<T, _>(LG_ODD));
    let r = t2 + t1;
    let hfsq = f * T::of(0.5) * f;
    let r_a = dk * T::of(ln2_hi) - ((hfsq - (s * (hfsq + r) + dk * T::of(ln2_lo))) - f);
    let r_b = dk * T::of(ln2_hi) - ((s * (f - r) - dk * T::of(ln2_lo)) - f);
    let result = sel(small, r_small, sel(i.gt(0), r_a, r_b));
    log_specials(x, ok, result)
}

/// The head of `f - f^2/2` and the tail of `log(1 + f)` (`e_log2.c`).
fn head_tail<T: Real>(f: Value<T>) -> (Value<T>, Value<T>) {
    let hfsq = f * T::of(0.5) * f;
    let r = k_log1p(f);
    let hi_ = trunc_bits(f - hfsq, 0xfffff000);
    let lo_ = (f - hi_) - hfsq + r;
    (hi_, lo_)
}

/// Base-2 logarithm (`e_log2.c`, `e_log2f.c`).
pub(crate) fn log2<T: Real>(x: Value<T>) -> Value<T> {
    let (xs, ok) = positive_finite(x);
    let (k, f, _) = reduce(xs, pick::<T, _>([0x4afb0d, 0x95f64]));
    let y = k.to::<T>();
    let (hi_, lo_) = head_tail(f);
    let result = if T::DOUBLE {
        let (ivln2hi, ivln2lo) = (T::of(db(0x3ff71547_65200000)), T::of(db(0x3de705fc_2eefa200)));
        let val_hi = hi_ * ivln2hi;
        let val_lo = (lo_ + hi_) * ivln2lo + lo_ * ivln2hi;
        let w = y + val_hi;
        let val_lo = val_lo + ((y - w) + val_hi);
        val_lo + w
    } else {
        let (ivln2hi, ivln2lo) = (T::of(fb(0x3fb8b000)), T::of(fb(0xb9389ad4)));
        (lo_ + hi_) * ivln2lo + lo_ * ivln2hi + hi_ * ivln2hi + y
    };
    log_specials(x, ok, sel(x.eq(T::of(1.0)), c::<T>(0.0), result))
}

/// Base-10 logarithm (`e_log10.c`, `e_log10f.c`).
pub(crate) fn log10<T: Real>(x: Value<T>) -> Value<T> {
    let (xs, ok) = positive_finite(x);
    let (k, f, _) = reduce(xs, pick::<T, _>([0x4afb0d, 0x95f64]));
    let y = k.to::<T>();
    let (hi_, lo_) = head_tail(f);
    let result = if T::DOUBLE {
        let (ivln10hi, ivln10lo) = (T::of(db(0x3fdbcb7b_15200000)), T::of(db(0x3dbb9438_ca9aadd5)));
        let (log10_2hi, log10_2lo) = (T::of(db(0x3fd34413_509f6000)), T::of(db(0x3d59fef3_11f12b36)));
        let val_hi = hi_ * ivln10hi;
        let y2 = y * log10_2hi;
        let val_lo = y * log10_2lo + (lo_ + hi_) * ivln10lo + lo_ * ivln10hi;
        let w = y2 + val_hi;
        let val_lo = val_lo + ((y2 - w) + val_hi);
        val_lo + w
    } else {
        let (ivln10hi, ivln10lo) = (T::of(fb(0x3ede6000)), T::of(fb(0xb804ead9)));
        let (log10_2hi, log10_2lo) = (T::of(fb(0x3e9a2080)), T::of(fb(0x355427db)));
        y * log10_2lo + (lo_ + hi_) * ivln10lo + lo_ * ivln10hi + hi_ * ivln10hi + y * log10_2hi
    };
    log_specials(x, ok, sel(x.eq(T::of(1.0)), c::<T>(0.0), result))
}

/// `Lp1..Lp7` of `s_log1p.c` and `s_log1pf.c`.
const LP: [&[f64]; 2] = [
    &[fb(0x3f2aaaab), fb(0x3ecccccd), fb(0x3e924925), fb(0x3e638e29), fb(0x3e3a3325), fb(0x3e1cd04f), fb(0x3e178897)],
    &[
        db(0x3fe55555_55555593),
        db(0x3fd99999_9997fa04),
        db(0x3fd24924_94229359),
        db(0x3fcc71c5_1d8e78af),
        db(0x3fc74664_96cb03de),
        db(0x3fc39a09_d078c69f),
        db(0x3fc2f112_df3e5244),
    ],
];

/// `log(1 + x)`, accurate near 0 (`s_log1p.c`).
pub(crate) fn log1p<T: Real>(x: Value<T>) -> Value<T> {
    // -0 adds nothing to any value, -0 included.
    log1p_extra(x, c::<T>(-0.0))
}

/// `log(1 + x + extra)` for a tail `extra` much smaller than `ulp(x)`:
/// msun's `log1p`, whose correction for the rounding of `1 + x` also takes
/// the tail (`extra / (1 + x)`, to first order). Used by `acosh`, whose
/// argument is computed in extra precision.
fn log1p_extra<T: Real>(x: Value<T>, extra: Value<T>) -> Value<T> {
    let wd = words::<T>();
    // High words: sqrt(2) - 1, 2**-29 (f32: 2**-15), 2**-54 (2**-24),
    // sqrt(2)/2 - 1, 2**53, and the significand of sqrt(2).
    let (w_sqrt2m1, w_tiny, w_tinier, w_neg, w_big, w_sqrt2, half_word) = pick::<T, _>([
        (0x3ed413d0, 0x38000000, 0x33800000, w(0xbe95f619), 0x5a000000, 0x3504f4, 0x3f000000),
        (0x3fda827a, 0x3e200000, 0x3c900000, w(0xbfd2bec4), 0x43400000, 0x6a09e, 0x3fe00000),
    ]);
    let (ln2_hi, ln2_lo) = pick::<T, _>(LN2);
    let valid = x.gt(T::of(-1.0)) & x.lt(T::of(f64::INFINITY));
    let xs = guard(valid, x, 0.5);
    let hx = hi(xs);
    let ax = hx & 0x7fffffff;
    let tiny = hx.lt(w_sqrt2m1) & ax.lt(w_tiny);
    // msun's x (or x - x*x/2), with the tail added before the one rounding
    // that matters (for log1p itself, -0 changes nothing).
    let r_tiny = xs + sel(ax.lt(w_tinier), extra, extra - xs * xs * T::of(0.5));
    // sqrt(2)/2- <= 1+x < sqrt(2)+: f = x with no reduction.
    let direct = hx.lt(w_sqrt2m1) & (hx.gt(0) | hx.le(w_neg));
    // Otherwise u = 1 + x = 2^k * (1 + f), with c correcting u's rounding.
    let below_big = hx.lt(w_big);
    let u = sel(below_big, c::<T>(1.0) + xs, xs);
    let tail = extra / u;
    let hu = hi(u);
    let k = (hu >> wd.shift) - T::BIAS;
    let corr = sel(k.gt(0), c::<T>(1.0) - (u - xs), xs - (u - T::of(1.0))) / u;
    let corr = sel(below_big, corr, c::<T>(0.0));
    let hu = hu & wd.mant_mask;
    let normal = hu.lt(w_sqrt2);
    let u = sel(normal, with_hi(u, hu | wd.one), with_hi(u, hu | half_word));
    let k = sel(normal, k, k + 1);
    let hu = sel(normal, hu, (Int32::from(wd.implicit) - hu) >> 2);
    let f = sel(direct, xs, u - T::of(1.0));
    let k = sel(direct, Int32::from(0), k);
    let hu = sel(direct, Int32::from(1), hu);
    let corr = sel(direct, c::<T>(0.0), corr) + tail;
    let dk = k.to::<T>();
    let hfsq = f * T::of(0.5) * f;
    // |f| < 2**-20.
    let r0 = hfsq * (c::<T>(1.0) - f * T::of(db(0x3fe55555_55555555)));
    let r_small = dk * T::of(ln2_hi) - ((r0 - (dk * T::of(ln2_lo) + corr)) - f);
    let r_small = sel(f.eq(T::of(0.0)), dk * T::of(ln2_hi) + (corr + dk * T::of(ln2_lo)), r_small);
    let s = f / (c::<T>(2.0) + f);
    let z = s * s;
    let r = z * horner(z, pick::<T, _>(LP));
    let r_main = dk * T::of(ln2_hi) - ((hfsq - (s * (hfsq + r) + (dk * T::of(ln2_lo) + corr))) - f);
    let result = sel(hu.eq(0), r_small, r_main);
    let result = sel(tiny, r_tiny, result);
    let result = sel(valid, result, nan::<T>());
    let result = sel(x.eq(T::of(-1.0)), inf::<T>(true), result);
    let result = sel(x.eq(T::of(f64::INFINITY)), x, result);
    canonical(result)
}

/// Inverse hyperbolic sine (`s_asinh.c`).
pub(crate) fn asinh<T: Real>(x: Value<T>) -> Value<T> {
    let (w_tiny, w_huge) = pick::<T, _>([(0x31800000, 0x4d800000), (0x3e300000, 0x41b00000)]);
    let ln2 = pick::<T, _>([fb(0x3f317218), db(0x3fe62e42_fefa39ef)]);
    let ix = hi(x) & 0x7fffffff;
    let ax = fabs(x);
    let huge = ix.gt(w_huge);
    let big = ix.gt(0x40000000);
    // log(2|x| + 1/(sqrt(x^2 + 1) + |x|)) for 2 < |x| <= 2^28 (f32: 2^12):
    // the sum's rounding is carried into log1p of the argument minus one
    // (exact, as the argument exceeds 4).
    let ab = guard(big & !huge, ax, 4.0);
    let (arg, arg_lo) = fast_two_sum(ab * T::of(2.0), c::<T>(1.0) / ((ab * ab + T::of(1.0)).sqrt() + ab));
    let w_big = sel(huge, log(ax) + T::of(ln2), log1p_extra(arg - T::of(1.0), arg_lo));
    // log1p(|x| + x^2 / (1 + sqrt(1 + x^2))) as msun computes it, but with
    // the argument in extra precision: its roundings cost up to 1.7 ulp.
    let a = guard(!big, ax, 1.0);
    let th = a * a;
    let tl = fma(a, a, fneg(th));
    let (wh, wl) = two_sum(c::<T>(1.0), th);
    let qh = wh.sqrt();
    let ql = (fma(fneg(qh), qh, wh) + (wl + tl)) / (qh * T::of(2.0));
    let (dh, dl) = fast_two_sum(qh, c::<T>(1.0));
    let (fh, fl) = dd_div(th, tl, dh, dl + ql);
    let (ah, al) = fast_two_sum(a, fh);
    let w_small = log1p_extra(ah, al + fl);
    let w_ = sel(big, w_big, w_small);
    let r = negate_if(hi(x).le(0), w_);
    let r = sel(ix.lt(w_tiny), x, r);
    let r = sel(ix.ge(words::<T>().infinity), x + x, r);
    canonical(r)
}

/// Inverse hyperbolic cosine (`e_acosh.c`).
pub(crate) fn acosh<T: Real>(x: Value<T>) -> Value<T> {
    let w_huge = pick::<T, _>([0x4d800000, 0x41b00000]);
    let ln2 = pick::<T, _>([fb(0x3f317218), db(0x3fe62e42_fefa39ef)]);
    let hx = hi(x);
    let huge = hx.ge(w_huge);
    let big = hx.gt(0x40000000);
    // log(2x - 1/(x + sqrt(x^2 - 1))) for 2 < x < 2^28 (f32: 2^12), with
    // the difference's rounding carried into log1p of the argument minus
    // one (exact, as the argument exceeds 3).
    let xb = guard(big & !huge, x, 4.0);
    let (arg, arg_lo) = fast_two_sum(xb * T::of(2.0), fneg(c::<T>(1.0) / (xb + (xb * xb - T::of(1.0)).sqrt())));
    let r_big = sel(huge, log(x) + T::of(ln2), log1p_extra(arg - T::of(1.0), arg_lo));
    // log1p(t + sqrt(2t + t^2)) with t = x - 1 (exact), as msun computes
    // it, but with the argument in extra precision: msun's roundings of it
    // cost up to 2 ulp near x = 1.03.
    let t = x - T::of(1.0);
    let p = t * t;
    let p_err = fma(t, t, fneg(p));
    let t2 = t * T::of(2.0);
    // 2t >= t^2 for t <= 2, so these are exact two-sums.
    let s = t2 + p;
    let s_err = (p - (s - t2)) + p_err;
    let q = s.sqrt();
    // The square root's residual s - q^2 is exact with fma.
    let q_err = (fma(fneg(q), q, s) + s_err) / (q * T::of(2.0));
    let u = q + t;
    let u_err = (t - (u - q)) + q_err;
    let r_small = log1p_extra(u, u_err);
    let r = sel(big, r_big, r_small);
    let r = sel(x.eq(T::of(1.0)), c::<T>(0.0), r);
    let r = sel(hx.ge(words::<T>().infinity), x + x, r);
    let r = sel(hx.lt(words::<T>().one), nan::<T>(), r);
    canonical(r)
}

/// Inverse hyperbolic tangent (`e_atanh.c`).
pub(crate) fn atanh<T: Real>(x: Value<T>) -> Value<T> {
    let (w_tiny, w_half) = pick::<T, _>([(0x31800000, 0x3f000000), (0x3e300000, 0x3fe00000)]);
    let ix = hi(x) & 0x7fffffff;
    let ax = fabs(x);
    let small = ix.lt(w_half);
    // msun's 0.5 log1p(2x + 2x^2 / (1 - x)) below 0.5 and 0.5 log1p(2x /
    // (1 - x)) above, with the arguments in extra precision (their
    // roundings cost up to 1.7 ulp).
    let a = guard(ax.lt(T::of(1.0)), ax, 0.0);
    let (oh, ol) = two_sum(c::<T>(1.0), fneg(a));
    let ph = a * a;
    let pl = fma(a, a, fneg(ph));
    let (fh, fl) = dd_div(ph * T::of(2.0), pl * T::of(2.0), oh, ol);
    let (sh, sl) = fast_two_sum(a * T::of(2.0), fh);
    let (gh, gl) = dd_div(a * T::of(2.0), c::<T>(0.0), oh, ol);
    let (arg, arg_lo) = (sel(small, sh, gh), sel(small, sl + fl, gl));
    let r = log1p_extra(arg, arg_lo) * T::of(0.5);
    let r = negate_if(hi(x).lt(0), r);
    let r = sel(ix.lt(w_tiny), x, r);
    let r = sel(ax.eq(T::of(1.0)), x / T::of(0.0), r);
    let r = sel(ax.gt(T::of(1.0)) | is_nan(x), nan::<T>(), r);
    canonical(r)
}

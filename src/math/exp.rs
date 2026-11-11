//! `exp`, `exp2`, `expm1`, `sinh`, `cosh` and `tanh`, ported from FreeBSD
//! msun (`e_exp.c`, `e_expf.c`, `s_expm1.c`, `s_expm1f.c`, `e_sinh.c`,
//! `e_sinhf.c`, `e_cosh.c`, `e_coshf.c`, `s_tanh.c`, `s_tanhf.c`).
//! `exp2` reuses the `exp` kernel on an argument split into two parts (msun's
//! own `exp2` reads a 256-entry table, which a kernel has no place for).
//! `sinh`, `cosh` and `tanh` keep msun's ranges and formulas, but take
//! `e^|x|` and `e^|x| - 1` from the same kernels with the rounding errors of
//! their last steps kept, and combine them in extra precision: msun's
//! versions are up to 1.3 to 2.1 ulp off, these less than 1.
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! The float versions were converted by Ian Lance Taylor, Cygnus Support.

use super::real::*;
use crate::value::{Int32, Value, fma};

/// `e_exp.c` / `e_expf.c` constants: `[f32, f64]`.
struct ExpK {
    o_threshold: f64,
    u_threshold: f64,
    ln2_hi: f64,
    ln2_lo: f64,
    inv_ln2: f64,
    p: &'static [f64],
    /// High word of 0.5 ln2: larger arguments are reduced.
    half_ln2: i32,
    /// High word from which msun tests the thresholds.
    filter: i32,
}

const EXP: [ExpK; 2] = [
    ExpK {
        o_threshold: fb(0x42b17180),
        u_threshold: fb(0xc2cff1b5),
        ln2_hi: fb(0x3f317200),
        ln2_lo: fb(0x35bfbe8e),
        inv_ln2: fb(0x3fb8aa3b),
        p: &[xp(0xaaaa8f, -26), -xp(0xb55215, -32)],
        half_ln2: 0x3eb17218,
        filter: 0x42b17218,
    },
    ExpK {
        o_threshold: db(0x40862e42_fefa39ef),
        u_threshold: db(0xc0874910_d52d3051),
        ln2_hi: db(0x3fe62e42_fee00000),
        ln2_lo: db(0x3dea39ef_35793c76),
        inv_ln2: db(0x3ff71547_652b82fe),
        p: &[
            db(0x3fc55555_5555553e),
            db(0xbf66c16c_16bebd93),
            db(0x3f11566a_af25de2c),
            db(0xbebbbd41_c5d26bf1),
            db(0x3e663769_72bea4d0),
        ],
        half_ln2: 0x3fd62e42,
        filter: 0x40862e42,
    },
];

/// The exp kernel: `exp(hi - lo) = 1 - ((lo - (x*c)/(2 - c)) - hi)` with
/// `x = hi - lo` in the primary range `|x| <= 0.5 ln2`.
fn exp_kernel<T: Real>(hi: Value<T>, lo: Value<T>) -> Value<T> {
    let k = &EXP[T::DOUBLE as usize];
    let x = hi - lo;
    let t = x * x;
    let c_ = x - t * horner(t, k.p);
    c::<T>(1.0) - ((lo - (x * c_) / (c::<T>(2.0) - c_)) - hi)
}

/// `exp(x) * 2^scale` for `x` between the underflow threshold and
/// `o_threshold - scale * ln2` (`scale <= 0`), without intermediate
/// overflow: msun's `__ldexp_exp` for `scale = -1`.
pub(crate) fn exp_core<T: Real>(x: Value<T>, extra: i32) -> Value<T> {
    let k = &EXP[T::DOUBLE as usize];
    let hx = hi(x) & 0x7fffffff;
    let reduce = hx.gt(k.half_ln2);
    // k = (int)(invln2*x + halF[xsb]); for 0.5 ln2 < |x| < 1.5 ln2 this is
    // +-1, and hi/lo below are exactly msun's special case for it.
    let half = sel(x.lt(T::of(0.0)), c::<T>(-0.5), c::<T>(0.5));
    let kf = sel(reduce, (x * T::of(k.inv_ln2) + half).trunc(), c::<T>(0.0));
    let hi_ = x - kf * T::of(k.ln2_hi);
    let lo_ = kf * T::of(k.ln2_lo);
    // With k = 0, msun's `one-((x*c)/(c-2.0)-x)` has the same bits.
    let y = exp_kernel(hi_, lo_);
    scale(y, kf.to::<i32>() + extra)
}

/// `e^(s + err)` as `(y, k)`, the result being `y * 2^k`, for `s` between
/// the underflow and overflow thresholds (and beyond them, as long as
/// `y * 2^k` is scaled with [`scale`]) and a tail `err` much smaller than
/// `ulp(s)`: msun's argument reduction, with the tail carried into the
/// kernel's low part (an extension used by `erfc`).
pub(crate) fn exp_split<T: Real>(s: Value<T>, err: Value<T>) -> (Value<T>, Int32) {
    let k = &EXP[T::DOUBLE as usize];
    let half = sel(s.lt(T::of(0.0)), c::<T>(-0.5), c::<T>(0.5));
    let kf = (s * T::of(k.inv_ln2) + half).trunc();
    // kf * ln2_hi is exact, and so is the subtraction.
    let hi_ = s - kf * T::of(k.ln2_hi);
    let lo_ = kf * T::of(k.ln2_lo) - err;
    (exp_kernel(hi_, lo_), kf.to::<i32>())
}

/// `e^x`.
pub(crate) fn exp<T: Real>(x: Value<T>) -> Value<T> {
    let k = &EXP[T::DOUBLE as usize];
    let filtered = (hi(x) & 0x7fffffff).ge(k.filter);
    let over = filtered & x.gt(T::of(k.o_threshold));
    let under = filtered & x.lt(T::of(k.u_threshold));
    let xs = guard(!(over | under) & !is_nan(x), x, 0.0);
    let r = exp_core(xs, 0);
    let r = sel(over, inf::<T>(false), sel(under, c::<T>(0.0), r));
    canonical(sel(is_nan(x), nan::<T>(), r))
}

/// `2^x`: `x = n + r` with `|r| <= 1/2` exact, and `2^r = exp(r ln2)` from
/// the exp kernel with `r ln2` split into a head and an exact tail.
pub(crate) fn exp2<T: Real>(x: Value<T>) -> Value<T> {
    let (max, min) = pick::<T, _>([(128.0, -151.0), (1024.0, -1076.0)]);
    let over = x.ge(T::of(max));
    let under = x.le(T::of(min));
    let xs = guard(!(over | under) & !is_nan(x), x, 0.0);
    let n = xs.rint();
    let r = xs - n;
    // ln2 to twice the precision: the f64 tail is ln2 - 0x3fe62e42fefa39ef.
    let ln2_32 = std::f64::consts::LN_2 as f32;
    let (ln2, ln2_lo) = pick::<T, _>([
        (ln2_32 as f64, (std::f64::consts::LN_2 - ln2_32 as f64) as f32 as f64),
        (std::f64::consts::LN_2, db(0x3c7abc9e_3b39803f)),
    ]);
    let (ln2, ln2_lo) = (T::of(ln2), T::of(ln2_lo));
    let th = r * ln2;
    let tl = fma(r, ln2, fneg(th)) + r * ln2_lo;
    let y = exp_kernel(th, fneg(tl));
    let r = scale(y, n.to::<i32>());
    let r = sel(over, inf::<T>(false), sel(under, c::<T>(0.0), r));
    canonical(sel(is_nan(x), nan::<T>(), r))
}

/// `s_expm1.c` / `s_expm1f.c` constants: `[f32, f64]`.
struct Expm1K {
    o_threshold: f64,
    ln2_hi: f64,
    ln2_lo: f64,
    inv_ln2: f64,
    q: &'static [f64],
    /// High words: 56 ln2 (f64) or 27 ln2 (f32), 0.5 ln2, and the size
    /// below which `expm1(x) = x`.
    big: i32,
    half_ln2: i32,
    tiny: i32,
    /// `k` below which `1 - 2^-k` is exact in the high word.
    k_split: i32,
    /// High word from which msun tests the overflow threshold.
    filter: i32,
}

const EXPM1: [Expm1K; 2] = [
    Expm1K {
        o_threshold: fb(0x42b17180),
        ln2_hi: fb(0x3f317180),
        ln2_lo: fb(0x3717f7d1),
        inv_ln2: fb(0x3fb8aa3b),
        q: &[1.0, -xp(0x888868, -28), xp(0xcf3010, -33)],
        big: 0x4195b844,
        half_ln2: 0x3eb17218,
        tiny: 0x33000000,
        k_split: 23,
        filter: 0x42b17218,
    },
    Expm1K {
        o_threshold: db(0x40862e42_fefa39ef),
        ln2_hi: db(0x3fe62e42_fee00000),
        ln2_lo: db(0x3dea39ef_35793c76),
        inv_ln2: db(0x3ff71547_652b82fe),
        q: &[
            1.0,
            db(0xbfa11111_111110f4),
            db(0x3f5a01a0_19fe5585),
            db(0xbf14ce19_9eaadbb7),
            db(0x3ed0cfca_86e65239),
            db(0xbe8afdb7_6e09c32d),
        ],
        big: 0x4043687a,
        half_ln2: 0x3fd62e42,
        tiny: 0x3c900000,
        k_split: 20,
        filter: 0x40862e42,
    },
];

/// `e^x - 1`, accurate near 0.
pub(crate) fn expm1<T: Real>(x: Value<T>) -> Value<T> {
    let k = &EXPM1[T::DOUBLE as usize];
    let hx = hi(x) & 0x7fffffff;
    let neg = x.lt(T::of(0.0));
    let over = hx.ge(k.filter) & x.gt(T::of(k.o_threshold));
    let minus_one = hx.ge(k.big) & neg;
    let tiny = hx.lt(k.tiny);
    let xs = guard(!(over | minus_one) & !is_nan(x), x, 0.0);
    let hxs = hi(xs) & 0x7fffffff;
    let reduce = hxs.gt(k.half_ln2);
    let half = sel(neg, c::<T>(-0.5), c::<T>(0.5));
    let kf = sel(reduce, (xs * T::of(k.inv_ln2) + half).trunc(), c::<T>(0.0));
    let hi_ = xs - kf * T::of(k.ln2_hi);
    let lo_ = kf * T::of(k.ln2_lo);
    let r = hi_ - lo_;
    let cc = (hi_ - r) - lo_;
    let ki = kf.to::<i32>();
    // x is now in the primary range.
    let hfx = r * T::of(0.5);
    let hxs2 = r * hfx;
    let r1 = horner(hxs2, k.q);
    let t = c::<T>(3.0) - r1 * hfx;
    let e = hxs2 * ((r1 - t) / (c::<T>(6.0) - r * t));
    let k0 = xs - (xs * e - hxs2);
    let e = (r * (e - cc) - cc) - hxs2;
    let km1 = (r - e) * T::of(0.5) - T::of(0.5);
    let k1 = sel(r.lt(T::of(-0.25)), (e - (r + T::of(0.5))) * T::of(-2.0), c::<T>(1.0) + (r - e) * T::of(2.0));
    let far = ki.le(-2) | ki.gt(56);
    let y_far = scale(c::<T>(1.0) - (e - r), ki) - T::of(1.0);
    // t = 1 - 2^-k for 2 <= k < k_split, t = 2^-k above.
    let kc = ki.max(0).min(k.k_split);
    let one = Value::constant(T::u((T::BIAS as u64) << T::MANT));
    let t1: Value<T> = from_bits(one - (Value::constant(T::u(1u64 << (T::MANT + 1))) >> kc.to::<T::U>()));
    let y_mid = scale(t1 - (e - r), ki);
    let t2 = pow2::<T>(-ki.max(k.k_split).min(56));
    let y_high = scale((r - (e + t2)) + T::of(1.0), ki);
    let y = sel(ki.lt(k.k_split), y_mid, y_high);
    let y = sel(far, y_far, y);
    let y = sel(ki.eq(1), k1, y);
    let y = sel(ki.eq(-1), km1, y);
    let y = sel(ki.eq(0), k0, y);
    let y = sel(tiny, x, y);
    let y = sel(over, inf::<T>(false), sel(minus_one, c::<T>(-1.0), y));
    canonical(sel(is_nan(x), nan::<T>(), y))
}

/// `e^x = (hi + lo) * 2^k`, to about 0.15 ulp: msun's argument reduction
/// and kernel, with the rounding errors of the kernel's last two steps
/// kept in `lo` instead of rounded into the result (for `x` up to the
/// overflow threshold; `2^k` may need [`scale`]). Used by the hyperbolic
/// functions, which combine `e^x` with `e^-x`.
pub(crate) fn exp_dd<T: Real>(x: Value<T>) -> (Value<T>, Value<T>, Int32) {
    let k = &EXP[T::DOUBLE as usize];
    let half = sel(x.lt(T::of(0.0)), c::<T>(-0.5), c::<T>(0.5));
    let kf = (x * T::of(k.inv_ln2) + half).trunc();
    let hi_ = x - kf * T::of(k.ln2_hi);
    let lo_ = kf * T::of(k.ln2_lo);
    let r = hi_ - lo_;
    let t = r * r;
    let c_ = r - t * horner(t, k.p);
    let d = (r * c_) / (c::<T>(2.0) - c_);
    // e^r = 1 + hi - lo + d, which msun rounds as 1 - ((lo - d) - hi).
    let (q, q_err) = two_sum(lo_ - d, fneg(hi_));
    let y = c::<T>(1.0) - q;
    let y_err = ((c::<T>(1.0) - y) - q) - q_err;
    (y, y_err, kf.to::<i32>())
}

/// `e^x - 1 = hi + lo` for `|x| <= 0.5 ln2`, where msun's `expm1` needs no
/// argument reduction and returns `x - (x*e - hxs)`: the same terms, with
/// the rounding of `x^2/2` kept, to a small fraction of an ulp.
fn expm1_dd<T: Real>(x: Value<T>) -> (Value<T>, Value<T>) {
    let k = &EXPM1[T::DOUBLE as usize];
    let hfx = x * T::of(0.5);
    let hxs = x * hfx;
    let hxs_err = fma(x, hfx, fneg(hxs));
    let r1 = horner(hxs, k.q);
    let t = c::<T>(3.0) - r1 * hfx;
    let e = hxs * ((r1 - t) / (c::<T>(6.0) - x * t));
    fast_two_sum(x, (hxs - x * e) + hxs_err)
}

/// High words for sinh, cosh and tanh: `[f32, f64]`.
struct HypK {
    /// 22 (f64) or 9 (f32): below, `exp(-|x|)` still matters.
    mid: i32,
    /// The overflow threshold of sinh and cosh.
    overflow: i32,
    /// Below, sinh and tanh return `x`.
    tiny: i32,
    /// 0.5 ln2: below, `e^|x| - 1` is computed without reduction.
    half_ln2: i32,
}

const HYP: [HypK; 2] = [
    HypK { mid: 0x41100000, overflow: 0x42b2d4fc, tiny: 0x39800000, half_ln2: 0x3eb17218 },
    HypK { mid: 0x40360000, overflow: 0x408633ce, tiny: 0x3e300000, half_ln2: 0x3fd62e42 },
];

/// `(e^|x| + s e^-|x|) / 2` for `s = 1` (cosh) or `-1` (sinh), with
/// `e^|x|` in extra precision: rounded once, so the result is accurate to
/// about `0.5 + 0.15 coth|x|` ulp. Also returns `e^|x| / 2`, rounded, for
/// the large arguments where `e^-|x|` is negligible.
fn half_exp_pair<T: Real>(ax: Value<T>, hx: Int32, minus: bool) -> (Value<T>, Value<T>) {
    let k = &HYP[T::DOUBLE as usize];
    let (yh, yl, kk) = exp_dd(guard(hx.le(k.overflow), ax, 1.0));
    // Below `mid`, 2^(k - 1) is a normal number and the scaling is exact.
    let mid = hx.lt(k.mid);
    let p2 = pow2::<T>(sel(mid, kk - 1, Int32::from(0)));
    let (hh, hl) = (yh * p2, yl * p2);
    // b = e^-|x| / 2 = 0.25 / (hh + hl), with the division's residual.
    let quarter = c::<T>(0.25);
    let b = quarter / hh;
    let bl = (fma(fneg(b), hh, quarter) - b * hl) / hh;
    let (b, bl) = if minus { (fneg(b), fneg(bl)) } else { (b, bl) };
    let (s, e) = fast_two_sum(hh, b);
    let pair = s + ((e + hl) + bl);
    // e^|x| / 2 for the large arguments, scaled without overflow.
    (pair, scale(yh + yl, kk - 1))
}

/// Hyperbolic sine: msun's `s_sinh.c` / `e_sinhf.c` structure, with `e^|x|`
/// and `e^|x| - 1` in extra precision (msun's formulas round
/// `expm1(|x|)`, which costs up to 1.8 ulp).
pub(crate) fn sinh<T: Real>(x: Value<T>) -> Value<T> {
    let k = &HYP[T::DOUBLE as usize];
    let hx = hi(x) & 0x7fffffff;
    let ax = fabs(x);
    // |x| < 0.5 ln2: (t + t / (t + 1)) / 2 with t = e^|x| - 1.
    let (th, tl) = expm1_dd(guard(hx.lt(k.half_ln2), ax, 0.0));
    let (dh, dl) = fast_two_sum(c::<T>(1.0), th);
    let (uh, ul) = dd_div(th, tl, dh, dl + tl);
    let (s, e) = fast_two_sum(th, uh);
    let r_small = (s + ((e + tl) + ul)) * T::of(0.5);
    // Up to 22 (f32: 9): (e^|x| - e^-|x|) / 2; beyond, e^|x| / 2.
    let (r_mid, r_big) = half_exp_pair(ax, hx, true);
    let r = sel(hx.lt(k.half_ln2), r_small, sel(hx.lt(k.mid), r_mid, r_big));
    let r = negate_if(sign_bit(x), r);
    let r = sel(hx.lt(k.tiny), x, r);
    let r = sel(hx.gt(k.overflow), x * T::of(pick::<T, _>([1e37, 1e307])), r);
    canonical(r)
}

/// Hyperbolic cosine: msun's `e_cosh.c` / `e_coshf.c`, with
/// `(e^|x| + e^-|x|) / 2` from `e^|x|` in extra precision (msun's
/// `half*t + half/t` costs up to 1.3 ulp).
pub(crate) fn cosh<T: Real>(x: Value<T>) -> Value<T> {
    let k = &HYP[T::DOUBLE as usize];
    let hx = hi(x) & 0x7fffffff;
    let ax = fabs(x);
    let half_ln2 = pick::<T, _>([0x3eb17218, 0x3fd62e43]);
    let t = expm1(guard(hx.lt(half_ln2), ax, 0.0));
    let w = t + T::of(1.0);
    let tiny_word = pick::<T, _>([0x39800000, 0x3c800000]);
    // coshf returns one for tiny |x|, cosh returns w (= 1 then).
    let r_tiny = c::<T>(1.0) + (t * t) / (w + w);
    let r_tiny = sel(hx.lt(tiny_word), pick::<T, _>([c::<T>(1.0), w]), r_tiny);
    let (r_mid, r_big) = half_exp_pair(ax, hx, false);
    let r = sel(hx.lt(k.mid), r_mid, r_big);
    let r = sel(hx.lt(half_ln2), r_tiny, r);
    let r = sel(hx.gt(k.overflow), inf::<T>(false), r);
    canonical(sel(is_nan(x), nan::<T>(), r))
}

/// Hyperbolic tangent. msun's `s_tanh.c` / `s_tanhf.c` compute `-t / (t +
/// 2)` with `t = expm1(-2|x|)` below 1 and `1 - 2 / (expm1(2|x|) + 2)`
/// above, which costs up to 2.1 ulp. Here `e^2|x| - 1` (below `0.25 ln2`)
/// and `e^2|x|` (above) are in extra precision, and so is the division:
/// `t / (t + 2)` and `1 - 2 / (e^2|x| + 1)`.
pub(crate) fn tanh<T: Real>(x: Value<T>) -> Value<T> {
    let k = &HYP[T::DOUBLE as usize];
    let hx = hi(x) & 0x7fffffff;
    let ax = fabs(x);
    let two_ax = ax * T::of(2.0);
    let small = (hi(two_ax) & 0x7fffffff).lt(k.half_ln2);
    let (th, tl) = expm1_dd(guard(small, two_ax, 0.0));
    let (dh, dl) = fast_two_sum(c::<T>(2.0), th);
    let (qh, ql) = dd_div(th, tl, dh, dl + tl);
    let r_small = qh + ql;
    // e^2|x| = (yh + yl) 2^k, at most 2^64 below `mid`.
    let mid = hx.lt(k.mid);
    let (yh, yl, kk) = exp_dd(guard(mid & !small, two_ax, 1.0));
    let p2 = pow2::<T>(kk);
    let (eh, el) = fast_two_sum(yh * p2, c::<T>(1.0));
    let (vh, vl) = dd_div(c::<T>(2.0), c::<T>(0.0), eh, el + yl * p2);
    let (s, e) = two_sum(c::<T>(1.0), fneg(vh));
    let r_mid = s + (e - vl);
    let z = sel(small, r_small, sel(mid, r_mid, c::<T>(1.0)));
    let r = negate_if(sign_bit(x), z);
    let r = sel(hx.lt(k.tiny), x, r);
    canonical(sel(is_nan(x), nan::<T>(), r))
}

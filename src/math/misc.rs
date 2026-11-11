//! `cbrt` (FreeBSD msun's `s_cbrt.c` and `s_cbrtf.c`; the float version
//! does its two Halley steps in double precision, as msun's does), `rem`
//! (msun's fixed-point `e_fmod.c`, reducing several bits per step), and
//! the exact functions: a correctly rounded `rsqrt`, `round`, `copysign`
//! and the classification predicates.
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
//! The float versions were converted by Ian Lance Taylor, Cygnus Support;
//! `s_cbrt.c` was rewritten by Bruce D. Evans.

use super::real::*;
use crate::control::while_carry;
use crate::value::{Bitwise, Float64, Int32, Integer, Uint64, Value, fma};

/// Cube root.
pub(crate) fn cbrt<T: Real>(x: Value<T>) -> Value<T> {
    let hx = hi(x);
    let sign = hx & w(0x80000000);
    let ax_w = hx ^ sign;
    let (inf_w, min_w) = pick::<T, _>([(0x7f800000, 0x00800000), (0x7ff00000, 0x00100000)]);
    // Rough cbrt to 5 bits, from the exponent: hx/3 + B.
    let (b1, b2) = pick::<T, _>([(709958130u32, 642849266u32), (715094163, 696219795)]);
    let sub = ax_w.lt(min_w);
    let scaled = x * T::of(pick::<T, _>([2f64.powi(24), 2f64.powi(54)]));
    let high = (hi(scaled) & 0x7fffffff).bitcast::<u32>();
    let guess = sel(sub, high / 3u32 + b2, ax_w.bitcast::<u32>() / 3u32 + b1);
    let t: Value<T> = from_hi(sign | guess.bitcast::<i32>());
    let r = if T::DOUBLE {
        let p = [
            db(0x3ffe03e6_0f61e692),
            db(0xbffe28e0_92f02420),
            db(0x3ff9f160_4a49d6c2),
            db(0xbfe844cb_bee751d9),
            db(0x3fc2b000_d4e4edd7),
        ];
        // New cbrt to 23 bits.
        let r = (t * t) * (t / x);
        let t =
            t * ((r * (r * T::of(p[2]) + T::of(p[1])) + T::of(p[0])) + ((r * r) * r) * (r * T::of(p[4]) + T::of(p[3])));
        // Round t away from zero to 23 bits (32 bits of the double, the
        // last two cleared), so t*t is exact.
        let tb = (bits(t).to::<u64>() + 0x80000000u64) & 0xffff_ffff_c000_0000u64;
        let t: Value<T> = tb.bitcast();
        // One Halley step to 53 bits.
        let s = t * t;
        let r = x / s;
        let w2 = t + t;
        let r = (r - t) / (w2 + r);
        t + t * r
    } else {
        // Two Halley steps, in double precision.
        let xd: Float64 = x.to();
        let td: Float64 = t.to();
        let step = |td: Float64| {
            let r = td * td * td;
            td * ((xd + xd) + r) / ((xd + r) + r)
        };
        step(step(td)).to::<T>()
    };
    let r = sel(x.eq(T::of(0.0)), x, r);
    canonical(sel(ax_w.ge(inf_w), x + x, r))
}

/// `1 / sqrt(x)`, correctly rounded but for a few ties: the reciprocal of
/// the square root of `x`'s significand scaled to `[1, 4)`, refined by a
/// Newton step whose residual `1 - m r^2` is computed exactly with fma.
pub(crate) fn rsqrt<T: Real>(x: Value<T>) -> Value<T> {
    let (sub_scale, min_normal) = pick::<T, _>([(32, fb(0x00800000)), (64, db(0x00100000_00000000))]);
    let sub = x.lt(T::of(min_normal));
    let xn = sel(sub, x * T::of(2f64.powi(2 * sub_scale)), x);
    let ks = sel(sub, Int32::from(sub_scale), Int32::from(0));
    let e = (bits(xn).to::<u64>() >> T::MANT as u64).to::<i32>() - T::BIAS;
    let e = guard_i32(x.gt(T::of(0.0)) & x.lt(T::of(f64::INFINITY)), e);
    let k = e >> 1;
    let m = xn * pow2::<T>(-(k + k));
    let s = m.sqrt();
    let r = c::<T>(1.0) / s;
    let hh = r * r;
    let hl = fma(r, r, fneg(hh));
    let e1 = fma(fneg(m), hh, c::<T>(1.0)) - m * hl;
    let r = fma(r * T::of(0.5), e1, r);
    let r = scale(r, ks - k);
    let r = sel(x.eq(T::of(f64::INFINITY)), c::<T>(0.0), r);
    let r = sel(x.eq(T::of(0.0)), sel(sign_bit(x), inf::<T>(true), inf::<T>(false)), r);
    canonical(sel(x.lt(T::of(0.0)) | is_nan(x), nan::<T>(), r))
}

fn guard_i32(ok: crate::value::Boolean, v: Int32) -> Int32 {
    sel(ok, v, Int32::from(0))
}

/// The floating-point remainder `x - n y` with `n = trunc(x / y)` (C's
/// `fmod`), which is exact.
pub(crate) fn rem<T: Real>(x: Value<T>, y: Value<T>) -> Value<T> {
    let ax = fabs(x);
    let ay = fabs(y);
    let finite = |v: Value<T>| v.lt(T::of(f64::INFINITY));
    let normal_case = finite(ax) & finite(ay) & ay.gt(T::of(0.0)) & ax.gt(ay);
    let mant = (1u64 << T::MANT) - 1;
    let split = |v: Value<T>| -> (Uint64, Int32) {
        let b = bits(v).to::<u64>();
        let biased = (b >> T::MANT as u64).to::<i32>();
        let frac = b & mant;
        let normal = biased.gt(0);
        let m = sel(normal, frac | (1u64 << T::MANT), frac);
        (m, sel(normal, biased, Int32::from(1)))
    };
    let (mx, ex) = split(guard(normal_case, ax, 2.0));
    let (my, ey) = split(guard(normal_case, ay, 1.0));
    // Fixed point: (mx << (ex - ey)) mod my, `chunk` bits per step.
    let chunk = 63 - (T::MANT as i32 + 1);
    let (r, _) = while_carry(
        (mx % my, ex - ey),
        |(_, d)| d.gt(0),
        |(r, d)| {
            let s = d.min(chunk);
            ((r << s.to::<u64>()) % my, d - s)
        },
    );
    let magnitude = scale(r.to::<T>(), ey - T::BIAS - T::MANT as i32);
    let result = negate_if(sign_bit(x), magnitude);
    // |x| < |y|: x; |x| = |y|: zero of x's sign; y infinite: x.
    let r = sel(normal_case, result, x);
    let r = sel(ax.eq(ay), negate_if(sign_bit(x), c::<T>(0.0)), r);
    let r = sel(!finite(ay) & finite(ax), x, r);
    let invalid = ay.eq(T::of(0.0)) | !finite(ax) | is_nan(y);
    canonical(sel(invalid, nan::<T>(), r))
}

/// Round half away from zero.
pub(crate) fn round<T: Real>(x: Value<T>) -> Value<T> {
    let t = x.trunc();
    let d = fabs(x - t);
    let step = sel(sign_bit(x), c::<T>(-1.0), c::<T>(1.0));
    canonical(sel(d.ge(T::of(0.5)), t + step, t))
}

/// `x^n` for integers, wrapping on overflow. A negative exponent gives 0,
/// except for bases 1 and -1 (and 0, whose true result has no integer
/// value: 0 is returned).
pub(crate) fn ipowi<I: Integer + Bitwise>(base: Value<I>, exp: Value<I>) -> Value<I> {
    let zero = Value::constant(I::from_bits(0));
    let one = Value::constant(I::from_bits(1));
    let (r, _, _) = while_carry(
        (one, base, exp),
        |(_, _, e)| e.ne(zero),
        |(r, b, e)| {
            let r = sel((e & one).eq(one), r * b, r);
            // A logical shift, so a negative exponent also ends (it is
            // replaced below).
            let e: Value<I> = if I::DTYPE.is_signed() { lshr(e) } else { e >> one };
            (r, b * b, e)
        },
    );
    if !I::DTYPE.is_signed() {
        return r;
    }
    let minus_one = zero - one;
    let negative = exp.lt(zero);
    let odd = (exp & one).eq(one);
    let neg_r = sel(base.eq(one), one, sel(base.eq(minus_one), sel(odd, minus_one, one), zero));
    sel(negative, neg_r, r)
}

/// Logical shift right by one of a signed integer.
fn lshr<I: Integer + Bitwise>(e: Value<I>) -> Value<I> {
    let one = Value::constant(I::from_bits(1));
    let shifted = e >> one;
    // Clear the sign bit the arithmetic shift copied.
    let top = Value::constant(I::from_bits(!(1u64 << (I::DTYPE.bits() - 1))));
    shifted & top
}

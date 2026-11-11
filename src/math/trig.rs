//! `sin`, `cos`, `tan` and `sincos`. The double versions are FreeBSD msun's
//! (`s_sin.c`, `s_cos.c`, `s_tan.c`, `k_sin.c`, `k_cos.c`, `k_tan.c` and the
//! medium-size reduction of `e_rem_pio2.c`); the float versions are the
//! fdlibm float kernels that newlib still ships (`kf_sin.c`, `kf_cos.c`,
//! `kf_tan.c`, `ef_rem_pio2.c`), because msun's own float kernels compute in
//! double precision.
//!
//! Arguments too large for the three-part Cody-Waite reduction (about 1.6e6
//! for f64, 201 for f32) are reduced by Payne and Hanek's method in integer
//! arithmetic, with the bits of 2/pi from msun's `k_rem_pio2.c` table: the
//! integer part (mod 4) and 128 bits of the fraction of `x * 2/pi`. This
//! replaces `__kernel_rem_pio2`, whose loops index its table at run time.
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
//! The float versions were converted by Ian Lance Taylor, Cygnus Support.

use super::real::*;
use crate::control::if_else_yield;
use crate::value::{Boolean, Int32, Uint64, Value, fma};

/// The bits of 2/pi in 24-bit chunks (`ipio2` of `k_rem_pio2.c`).
const IPIO2: [u32; 54] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
];

/// Bits `64 j + 1 ..= 64 j + 64` of 2/pi.
fn two_over_pi_word(j: usize) -> u64 {
    let bit = |i: usize| (IPIO2[i / 24] >> (23 - i % 24)) & 1;
    (0..64).fold(0u64, |w, k| (w << 1) | bit(64 * j + k) as u64)
}

/// `values[index - first]`, by a chain of selects.
fn table(index: Int32, first: i32, values: &[u64]) -> Uint64 {
    let mut out = Value::constant(values[values.len() - 1]);
    for (k, &v) in values.iter().enumerate().rev().skip(1) {
        out = sel(index.eq(first + k as i32), Value::constant(v), out);
    }
    out
}

/// `a * b` as (high, low) 64-bit halves.
fn mul_wide(a: Uint64, b: Uint64) -> (Uint64, Uint64) {
    let mask = 0xffff_ffffu64;
    let (a0, a1, b0, b1) = (a & mask, a >> 32u64, b & mask, b >> 32u64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 32u64) + (p01 & mask) + (p10 & mask);
    let low = (p00 & mask) | (mid << 32u64);
    let high = p11 + (p01 >> 32u64) + (p10 >> 32u64) + (mid >> 32u64);
    (high, low)
}

/// `(x + y) mod 2^64` and the carry.
fn add_carry(x: Uint64, y: Uint64) -> (Uint64, Uint64) {
    let s = x + y;
    (s, s.lt(x).to::<u64>())
}

/// Payne-Hanek: for finite `|x| >= 2^MANT / 4` (or so), the quadrant `n`
/// and the reduced argument `y0 + y1 = |x| - n pi/2`, `|y0 + y1| <= pi/4`.
fn payne_hanek<T: Real>(ax: Value<T>) -> (Int32, Value<T>, Value<T>) {
    // The window of 2/pi words, and the highest first word.
    let (words, last) = pick::<T, _>([(3usize, 1i32), (4, 15)]);
    let table_words: Vec<u64> =
        std::iter::once(0).chain((0..=(last as usize + words - 1)).map(two_over_pi_word)).collect();
    let b = bits(ax).to::<u64>();
    let mant = (1u64 << T::MANT) - 1;
    let m = (b & mant) | (1u64 << T::MANT);
    let e = (b >> T::MANT as u64).to::<i32>() - T::BIAS;
    let p = e - T::MANT as i32;
    // j0 = floor((p - 2) / 64), so that the words before it only add
    // multiples of 4 and the window keeps 2^-100 or better.
    let j0 = ((p - 2 + 128) >> 6) - 2;
    let window: Vec<Uint64> = (0..words).map(|k| table(j0, -1 - k as i32, &table_words)).collect();
    // q = m * window, little-endian 64-bit words.
    let mut q: Vec<Uint64> = vec![Value::constant(0u64); words + 1];
    for k in 0..words {
        // window[0] is the most significant word.
        let (h, l) = mul_wide(m, window[words - 1 - k]);
        let (s, c1) = add_carry(q[k], l);
        q[k] = s;
        let (s, c2) = add_carry(q[k + 1], h + c1);
        q[k + 1] = s;
        let mut carry = c2;
        for word in q.iter_mut().skip(k + 2) {
            let (s, c) = add_carry(*word, carry);
            *word = s;
            carry = c;
        }
    }
    // The binary point of x * 2/pi is at bit t = 64 (j0 + words) - p of q;
    // shift left by u so that it is at bit 64 words - 2.
    let t = (j0 + words as i32) * 64 - p;
    let u = (words as i32 * 64 - 2) - t;
    let uu = u.to::<u64>();
    let inv = (63 - u).to::<u64>();
    let shifted = |k: usize| {
        let below = if k == 0 { Value::constant(0u64) } else { (q[k - 1] >> 1u64) >> inv };
        (q[k] << uu) | below
    };
    let (s2, s1, s0) = (shifted(words - 1), shifted(words - 2), shifted(words - 3));
    let mut n = (s2 >> 62u64).to::<i32>();
    let f_hi = (s2 << 2u64) | (s1 >> 62u64);
    let f_lo = (s1 << 2u64) | (s0 >> 62u64);
    // Round to the nearest quadrant: a fraction of 1/2 or more is negative.
    let negative = (f_hi >> 63u64).eq(1u64);
    n = sel(negative, n + 1, n);
    let lo_neg = !f_lo + 1u64;
    let hi_neg = !f_hi + f_lo.eq(0u64).to::<u64>();
    let (f_hi, f_lo) = (sel(negative, hi_neg, f_hi), sel(negative, lo_neg, f_lo));
    // Normalize the 128-bit magnitude and convert it to a float pair.
    let empty = f_hi.eq(0u64);
    let (f_hi, f_lo, extra) =
        (sel(empty, f_lo, f_hi), sel(empty, Value::constant(0u64), f_lo), sel(empty, Int32::from(64), Int32::from(0)));
    let lz = f_hi.clz().min(63u64);
    let lzi = lz.to::<i32>();
    let n_hi = (f_hi << lz) | ((f_lo >> 1u64) >> (63u64 - lz));
    let n_lo = f_lo << lz;
    let total = lzi + extra;
    let (a, b2) = if T::DOUBLE {
        let a = (n_hi >> 11u64).to::<T>();
        let b2 = ((n_hi & 0x7ffu64) << 53u64 | (n_lo >> 11u64)).to::<T>();
        (scale(a, -total - 53), scale(b2, -total - 117))
    } else {
        let a = (n_hi >> 40u64).to::<T>();
        let b2 = ((n_hi >> 16u64) & 0xff_ffffu64).to::<T>();
        (scale(a, -total - 24), scale(b2, -total - 48))
    };
    let (pio2_hi, pio2_lo) =
        pick::<T, _>([(fb(0x3fc90fdb), fb(0xb33bbd2e)), (db(0x3ff921fb_54442d18), db(0x3c91a626_33145c07))]);
    let (pio2_hi, pio2_lo) = (T::of(pio2_hi), T::of(pio2_lo));
    let y0 = a * pio2_hi;
    let y1 = fma(a, pio2_hi, fneg(y0)) + (a * pio2_lo + b2 * pio2_hi);
    let s = y0 + y1;
    let y1 = y1 - (s - y0);
    (n, negate_if(negative, s), negate_if(negative, y1))
}

/// `x = n pi/2 + y0 + y1` with `|y0 + y1| <~ pi/4` (`__ieee754_rem_pio2`).
fn rem_pio2<T: Real>(x: Value<T>) -> (Int32, Value<T>, Value<T>) {
    let ix = hi(x) & 0x7fffffff;
    let medium_limit = pick::<T, _>([0x43490f81, 0x413921fb]);
    let large = ix.ge(medium_limit);
    if_else_yield(
        large,
        || {
            // Payne-Hanek on |x| (NaN and infinity become a safe large value).
            let ax = guard(ix.lt(pick::<T, _>([0x7f800000, 0x7ff00000])), fabs(x), pick::<T, _>([1e10, 1e300]));
            let (n, y0, y1) = payne_hanek(ax);
            let neg = hi(x).lt(0);
            (sel(neg, -n, n), negate_if(neg, y0), negate_if(neg, y1))
        },
        || medium(x),
    )
}

/// The reduction for medium-size arguments: `e_rem_pio2.c`'s for f64,
/// `ef_rem_pio2.c`'s for f32. Each further step subtracts the next part of
/// pi/2 when the previous one cancelled (the exponent of `y0` fell by more
/// than 16 and 49 bits, f32: 8 and 25); every step is computed and the
/// right one selected. (The third step drops the second's rounding error,
/// which is only zero after such a cancellation, so it must not always be
/// taken.)
fn medium<T: Real>(x: Value<T>) -> (Int32, Value<T>, Value<T>) {
    let k: [f64; 7] = pick::<T, _>([
        [
            fb(0x3f22f984),
            fb(0x3fc90f80),
            fb(0x37354443),
            fb(0x37354400),
            fb(0x2e85a308),
            fb(0x2e85a300),
            fb(0x248d3132),
        ],
        [
            db(0x3fe45f30_6dc9c883),
            db(0x3ff921fb_54400000),
            db(0x3dd0b461_1a626331),
            db(0x3dd0b461_1a600000),
            db(0x3ba3198a_2e037073),
            db(0x3ba3198a_2e000000),
            db(0x397b839a_252049c1),
        ],
    ]);
    let [invpio2, pio2_1, pio2_1t, pio2_2, pio2_2t, pio2_3, pio2_3t] = k.map(T::of);
    // f64 reduces x, rounding to nearest; f32 reduces |x|, rounding halves up.
    let (t0, fnv) = if T::DOUBLE {
        (x, (x * invpio2).rint())
    } else {
        let ax = fabs(x);
        (ax, (ax * invpio2 + T::of(0.5)).trunc())
    };
    // The exponent field of x, and of each round's y0.
    let (shift, field, second_after, third_after) = pick::<T, _>([(23, 0xff, 8, 25), (20, 0x7ff, 16, 49)]);
    let j = (hi(t0) & 0x7fffffff) >> shift;
    let cancelled = |y0: Value<T>, bits: i32| (j - ((hi(y0) >> shift) & field)).gt(bits);
    // The first round is exact (fn * pio2_1 has few bits) up to its tail.
    let r1 = t0 - fnv * pio2_1;
    let w1 = fnv * pio2_1t;
    let step = |t: Value<T>, p: T, pt: T| {
        let w = fnv * p;
        let r = t - w;
        (r, fnv * pt - ((t - r) - w))
    };
    let (r2, w2) = step(r1, pio2_2, pio2_2t);
    let (r3, w3) = step(r2, pio2_3, pio2_3t);
    let second = cancelled(r1 - w1, second_after);
    let third = second & cancelled(r2 - w2, third_after);
    let r = sel(third, r3, sel(second, r2, r1));
    let w = sel(third, w3, sel(second, w2, w1));
    let y0 = r - w;
    let y1 = (r - y0) - w;
    let n = fnv.to::<i32>();
    if T::DOUBLE {
        (n, y0, y1)
    } else {
        let neg = hi(x).lt(0);
        (sel(neg, -n, n), negate_if(neg, y0), negate_if(neg, y1))
    }
}

/// `__kernel_sin(x, y, iy)`: sin(x + y) for |x| <~ pi/4; `iy` is whether
/// `y` is nonzero.
fn kernel_sin<T: Real>(x: Value<T>, y: Value<T>, iy: bool) -> Value<T> {
    let s: [f64; 6] = pick::<T, _>([
        [fb(0xbe2aaaab), fb(0x3c088889), fb(0xb9500d01), fb(0x3638ef1b), fb(0xb2d72f34), fb(0x2f2ec9d3)],
        [
            db(0xbfc55555_55555549),
            db(0x3f811111_1110f8a6),
            db(0xbf2a01a0_19c161d5),
            db(0x3ec71de3_57b1fe7d),
            db(0xbe5ae5e6_8a2b9ceb),
            db(0x3de5d93a_5acfd57c),
        ],
    ]);
    let [s1, s2, s3, s4, s5, s6] = s.map(T::of);
    let z = x * x;
    let v = z * x;
    let r = if T::DOUBLE {
        let w = z * z;
        (z * (z * s4 + s3) + s2) + z * w * (z * s6 + s5)
    } else {
        horner(z, &[s[1], s[2], s[3], s[4], s[5]])
    };
    if iy { x - ((z * (y * T::of(0.5) - v * r) - y) - v * s1) } else { x + v * (z * r + s1) }
}

/// `__kernel_cos(x, y)`: cos(x + y) for |x| <~ pi/4.
fn kernel_cos<T: Real>(x: Value<T>, y: Value<T>) -> Value<T> {
    let cc: [f64; 6] = pick::<T, _>([
        [fb(0x3d2aaaab), fb(0xbab60b61), fb(0x37d00d01), fb(0xb493f27c), fb(0x310f74f6), fb(0xad47d74e)],
        [
            db(0x3fa55555_5555554c),
            db(0xbf56c16c_16c15177),
            db(0x3efa01a0_19cb1590),
            db(0xbe927e4f_809c52ad),
            db(0x3e21ee9e_bdb4b1c4),
            db(0xbda8fae9_be8838d4),
        ],
    ]);
    let z = x * x;
    let one = c::<T>(1.0);
    if T::DOUBLE {
        let [c1, c2, c3, c4, c5, c6] = cc.map(T::of);
        let w = z * z;
        let r = z * (z * (z * c3 + c2) + c1) + w * w * (z * (z * c6 + c5) + c4);
        let hz = z * T::of(0.5);
        let w = one - hz;
        w + (((one - w) - hz) + (z * r - x * y))
    } else {
        let ix = hi(x) & 0x7fffffff;
        let r = z * horner(z, &cc);
        let near = one - (z * T::of(0.5) - (z * r - x * y));
        // qx = x/4 (or 0.28125 above 0.78125) splits off the leading term.
        let qx = sel(ix.gt(0x3f480000), c::<T>(0.28125), from_hi::<T>(ix - 0x01000000));
        let hz = z * T::of(0.5) - qx;
        let a = one - qx;
        let far = a - (hz - (z * r - x * y));
        sel(ix.lt(0x3e99999a), near, far)
    }
}

/// `__kernel_tan(x, y, iy)`: tan(x + y) (`iy = 1`) or -1/tan(x + y)
/// (`iy = -1`) for |x| <~ pi/4.
fn kernel_tan<T: Real>(x: Value<T>, y: Value<T>, iy: Int32) -> Value<T> {
    let tc: [f64; 15] = pick::<T, _>([
        [
            fb(0x3eaaaaab),
            fb(0x3e088889),
            fb(0x3d5d0dd1),
            fb(0x3cb327a4),
            fb(0x3c11371f),
            fb(0x3b6b6916),
            fb(0x3abede48),
            fb(0x3a1a26c8),
            fb(0x398137b9),
            fb(0x38a3f445),
            fb(0x3895c07a),
            fb(0xb79bae5f),
            fb(0x37d95384),
            fb(0x3f490fda),
            fb(0x33222168),
        ],
        [
            db(0x3fd55555_55555563),
            db(0x3fc11111_1110fe7a),
            db(0x3faba1ba_1bb341fe),
            db(0x3f9664f4_8406d637),
            db(0x3f8226e3_e96e8493),
            db(0x3f6d6d22_c9560328),
            db(0x3f57dbc8_fee08315),
            db(0x3f4344d8_f2f26501),
            db(0x3f3026f7_1a8d1068),
            db(0x3f147e88_a03792a6),
            db(0x3f12b80f_32f0a7e9),
            db(0xbef375cb_db605373),
            db(0x3efb2a70_74bf7ad4),
            db(0x3fe921fb_54442d18),
            db(0x3c81a626_33145c07),
        ],
    ]);
    let tt = tc.map(T::of);
    let (pio4, pio4lo) = (tt[13], tt[14]);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let big = ix.ge(pick::<T, _>([0x3f2ca140, 0x3fe59428]));
    let neg = hx.lt(0);
    let xr = negate_if(neg, x);
    let yr = negate_if(neg, y);
    let x1 = sel(big, (Value::constant(pio4) - xr) + (Value::constant(pio4lo) - yr), x);
    let y1 = sel(big, c::<T>(0.0), y);
    let (x, y) = (x1, y1);
    let z = x * x;
    let w = z * z;
    let r = horner(w, &[tc[1], tc[3], tc[5], tc[7], tc[9], tc[11]]);
    let v = z * horner(w, &[tc[2], tc[4], tc[6], tc[8], tc[10], tc[12]]);
    let s = z * x;
    let r = y + z * (s * (r + v) + y);
    let r = r + s * tt[0];
    let w = x + r;
    let vi = iy.to::<T>();
    let sign = sel(neg, c::<T>(-1.0), c::<T>(1.0));
    let r_big = sign * (vi - (x - (w * w / (w + vi) - r)) * T::of(2.0));
    // -1/(x + r), computed accurately.
    let z2 = trunc_bits(w, 0xfffff000);
    let v2 = r - (z2 - x);
    let a = c::<T>(-1.0) / w;
    let t = trunc_bits(a, 0xfffff000);
    let s2 = c::<T>(1.0) + t * z2;
    let r_inv = t + a * (s2 + t * v2);
    let r = sel(big, r_big, sel(iy.eq(1), w, r_inv));
    if T::DOUBLE {
        r
    } else {
        // kf_tan.c: |x| < 2**-28 returns x or -1/x directly.
        let tiny = ix.lt(0x31800000);
        let r_tiny = sel(iy.eq(1), x, c::<T>(-1.0) / x);
        sel(tiny, r_tiny, r)
    }
}

/// The argument reduction shared by sin, cos, tan and sincos: whether
/// `|x| <~ pi/4` (no reduction), the quadrant, and the reduced argument.
fn reduced<T: Real>(x: Value<T>) -> (Boolean, Int32, Value<T>, Value<T>) {
    let ix = hi(x) & 0x7fffffff;
    let small = ix.le(pick::<T, _>([0x3f490fd8, 0x3fe921fb]));
    let (n, y0, y1) = rem_pio2(x);
    (small, n, y0, y1)
}

fn is_finite<T: Real>(x: Value<T>) -> Boolean {
    (hi(x) & 0x7fffffff).lt(pick::<T, _>([0x7f800000, 0x7ff00000]))
}

/// The `n & 3` quadrant selection of `s_sin.c` and `s_cos.c`.
fn quadrant<T: Real>(n: Int32, a: Value<T>, b: Value<T>, c_: Value<T>, d: Value<T>) -> Value<T> {
    let q = n & 3;
    sel(q.eq(0), a, sel(q.eq(1), b, sel(q.eq(2), c_, d)))
}

/// `(sin(x), cos(x))`.
pub(crate) fn sincos<T: Real>(x: Value<T>) -> (Value<T>, Value<T>) {
    let ix = hi(x) & 0x7fffffff;
    let (small, n, y0, y1) = reduced(x);
    let (s, co) = (kernel_sin(y0, y1, true), kernel_cos(y0, y1));
    let sin_r = quadrant(n, s, co, fneg(s), fneg(co));
    let cos_r = quadrant(n, co, fneg(s), fneg(co), s);
    let zero = c::<T>(0.0);
    let (tiny_sin, tiny_cos) = pick::<T, _>([(0x32000000, 0x32000000), (0x3e500000, 0x3e46a09e)]);
    let sin_small = sel(ix.lt(tiny_sin), x, kernel_sin(x, zero, false));
    let cos_small = sel(ix.lt(tiny_cos), c::<T>(1.0), kernel_cos(x, zero));
    let sin_r = sel(small, sin_small, sin_r);
    let cos_r = sel(small, cos_small, cos_r);
    let finite = is_finite(x);
    (canonical(sel(finite, sin_r, nan::<T>())), canonical(sel(finite, cos_r, nan::<T>())))
}

/// Sine.
pub(crate) fn sin<T: Real>(x: Value<T>) -> Value<T> {
    let ix = hi(x) & 0x7fffffff;
    let (small, n, y0, y1) = reduced(x);
    let r = quadrant(
        n,
        kernel_sin(y0, y1, true),
        kernel_cos(y0, y1),
        fneg(kernel_sin(y0, y1, true)),
        fneg(kernel_cos(y0, y1)),
    );
    let tiny = pick::<T, _>([0x32000000, 0x3e500000]);
    let r_small = sel(ix.lt(tiny), x, kernel_sin(x, c::<T>(0.0), false));
    canonical(sel(is_finite(x), sel(small, r_small, r), nan::<T>()))
}

/// Cosine.
pub(crate) fn cos<T: Real>(x: Value<T>) -> Value<T> {
    let ix = hi(x) & 0x7fffffff;
    let (small, n, y0, y1) = reduced(x);
    let r = quadrant(
        n,
        kernel_cos(y0, y1),
        fneg(kernel_sin(y0, y1, true)),
        fneg(kernel_cos(y0, y1)),
        kernel_sin(y0, y1, true),
    );
    let tiny = pick::<T, _>([0x32000000, 0x3e46a09e]);
    let r_small = sel(ix.lt(tiny), c::<T>(1.0), kernel_cos(x, c::<T>(0.0)));
    canonical(sel(is_finite(x), sel(small, r_small, r), nan::<T>()))
}

/// Tangent.
pub(crate) fn tan<T: Real>(x: Value<T>) -> Value<T> {
    let ix = hi(x) & 0x7fffffff;
    let (_, n, y0, y1) = reduced(x);
    let small = ix.le(pick::<T, _>([0x3f490fda, 0x3fe921fb]));
    let r = kernel_tan(y0, y1, Int32::from(1) - ((n & 1) << 1));
    let r_small = kernel_tan(x, c::<T>(0.0), Int32::from(1));
    let r_small = if T::DOUBLE { sel(ix.lt(0x3e400000), x, r_small) } else { r_small };
    canonical(sel(is_finite(x), sel(small, r_small, r), nan::<T>()))
}

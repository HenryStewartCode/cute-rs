//! `pow`, ported from FreeBSD msun (`e_pow.c`, `e_powf.c`): `log2(|x|)` in
//! extra precision, times `y` split in two, then `2^(y log2|x|)`.
//!
//! ```text
//! ====================================================
//! Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//!
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! The float version was converted by Ian Lance Taylor, Cygnus Support.

use super::real::*;
use crate::value::{Boolean, Int32, Value};

struct PowK {
    dp_h: f64,
    dp_l: f64,
    l: &'static [f64],
    p: &'static [f64],
    lg2: f64,
    lg2_h: f64,
    lg2_l: f64,
    ovt: f64,
    cp: f64,
    cp_h: f64,
    cp_l: f64,
    ivln2: f64,
    ivln2_h: f64,
    ivln2_l: f64,
}

const POW: [PowK; 2] = [
    PowK {
        dp_h: fb(0x3f15c000),
        dp_l: fb(0x35d1cfdc),
        l: &[fb(0x3f19999a), fb(0x3edb6db7), fb(0x3eaaaaab), fb(0x3e8ba305), fb(0x3e6c3255), fb(0x3e53f142)],
        p: &[fb(0x3e2aaaab), fb(0xbb360b61), fb(0x388ab355), fb(0xb5ddea0e), fb(0x3331bb4c)],
        lg2: fb(0x3f317218),
        lg2_h: fb(0x3f317200),
        lg2_l: fb(0x35bfbe8c),
        ovt: 4.2995665694e-08,
        cp: fb(0x3f76384f),
        cp_h: fb(0x3f764000),
        cp_l: fb(0xb8f623c6),
        ivln2: fb(0x3fb8aa3b),
        ivln2_h: fb(0x3fb8aa00),
        ivln2_l: fb(0x36eca570),
    },
    PowK {
        dp_h: db(0x3fe2b803_40000000),
        dp_l: db(0x3e4cfdeb_43cfd006),
        l: &[
            db(0x3fe33333_33333303),
            db(0x3fdb6db6_db6fabff),
            db(0x3fd55555_518f264d),
            db(0x3fd17460_a91d4101),
            db(0x3fcd864a_93c9db65),
            db(0x3fca7e28_4a454eef),
        ],
        p: &[
            db(0x3fc55555_5555553e),
            db(0xbf66c16c_16bebd93),
            db(0x3f11566a_af25de2c),
            db(0xbebbbd41_c5d26bf1),
            db(0x3e663769_72bea4d0),
        ],
        lg2: db(0x3fe62e42_fefa39ef),
        lg2_h: db(0x3fe62e43_00000000),
        lg2_l: db(0xbe205c61_0ca86c39),
        ovt: db(0x3c971547_652b82fe),
        cp: db(0x3feec709_dc3a03fd),
        cp_h: db(0x3feec709_e0000000),
        cp_l: db(0xbe3e2fe0_145b01f5),
        ivln2: db(0x3ff71547_652b82fe),
        ivln2_h: db(0x3ff71547_60000000),
        ivln2_l: db(0x3e54ae0b_f85ddf44),
    },
];

/// `(yisint, is_int)`: whether `y` is an odd integer, and whether it is an
/// integer at all (msun's `yisint` is 1 for odd, 2 for even, 0 otherwise).
pub(crate) fn integer_parity<T: Real>(y: Value<T>) -> (Boolean, Boolean) {
    let ay = fabs(y);
    let huge = ay.ge(T::of(2f64.powi(T::MANT as i32 + 1)));
    let is_int = huge | y.trunc().eq(y);
    let half = y * T::of(0.5);
    let odd = !huge & is_int & half.trunc().ne(half);
    (odd, is_int)
}

/// `log2(ax) = t1 + t2` in extra precision, for positive finite `ax`.
fn log2_ext<T: Real>(ax: Value<T>) -> (Value<T>, Value<T>) {
    let k = &POW[T::DOUBLE as usize];
    let (shift, implicit, mant_mask, one, sub_bits, k1_max, k2_max) = pick::<T, _>([
        (23, 0x00800000, 0x007fffff, 0x3f800000, 24, 0x1cc471, 0x5db3d7),
        (20, 0x00100000, 0x000fffff, 0x3ff00000, 53, 0x3988e, 0xbb67a),
    ]);
    let ix = hi(ax);
    let sub = ix.lt(implicit);
    let ax = sel(sub, ax * T::of(2f64.powi(sub_bits)), ax);
    let ix = hi(ax);
    let n = sel(sub, Int32::from(-sub_bits), Int32::from(0)) + (ix >> shift) - T::BIAS;
    let j = ix & mant_mask;
    // The interval: |x| < sqrt(3/2) (k = 0), < sqrt(3) (k = 1), or else x/2.
    let kk = j.gt(k1_max) & j.lt(k2_max);
    let wrap = j.ge(k2_max);
    let n = sel(wrap, n + 1, n);
    let ix = sel(wrap, (j | one) - implicit, j | one);
    let ax = with_hi(ax, ix);
    let bp = sel(kk, c::<T>(1.5), c::<T>(1.0));
    let dp_h = sel(kk, c::<T>(k.dp_h), c::<T>(0.0));
    let dp_l = sel(kk, c::<T>(k.dp_l), c::<T>(0.0));
    // ss = s_h + s_l = (x - 1)/(x + 1) or (x - 1.5)/(x + 1.5).
    let u = ax - bp;
    let v = c::<T>(1.0) / (ax + bp);
    let ss = u * v;
    let s_h = trunc_bits(ss, 0xfffff000);
    // t_h = ax + bp, high part.
    let kbit = sel(kk, Int32::from(1), Int32::from(0));
    let t_h = if T::DOUBLE {
        from_hi::<T>(((ix >> 1) | 0x20000000) + 0x00080000 + (kbit << 18))
    } else {
        from_hi::<T>((((ix >> 1) & w(0xfffff000)) | 0x20000000) + 0x00400000 + (kbit << 21))
    };
    let t_l = ax - (t_h - bp);
    let s_l = v * ((u - s_h * t_h) - s_h * t_l);
    // log(ax)
    let s2 = ss * ss;
    let r = s2 * s2 * horner(s2, k.l);
    let r = r + s_l * (s_h + ss);
    let s2 = s_h * s_h;
    let t_h = trunc_bits(c::<T>(3.0) + s2 + r, 0xfffff000);
    let t_l = r - ((t_h - T::of(3.0)) - s2);
    // u + v = ss * (1 + ...)
    let u = s_h * t_h;
    let v = s_l * t_h + t_l * ss;
    // 2/(3 log2) * (ss + ...)
    let p_h = trunc_bits(u + v, 0xfffff000);
    let p_l = v - (p_h - u);
    let z_h = p_h * T::of(k.cp_h);
    let z_l = p_h * T::of(k.cp_l) + p_l * T::of(k.cp) + dp_l;
    // log2(ax) = (ss + ..) * 2/(3 log2) = n + dp_h + z_h + z_l
    let t = n.to::<T>();
    let t1 = trunc_bits(((z_h + z_l) + dp_h) + t, 0xfffff000);
    let t2 = z_l - (((t1 - t) - dp_h) - z_h);
    (t1, t2)
}

/// `x^y`.
pub(crate) fn pow<T: Real>(x: Value<T>, y: Value<T>) -> Value<T> {
    let k = &POW[T::DOUBLE as usize];
    let inf_w = pick::<T, _>([0x7f800000, 0x7ff00000]);
    let zero = c::<T>(0.0);
    let one = c::<T>(1.0);
    let infinity = inf::<T>(false);
    let (hx, hy) = (hi(x), hi(y));
    let (ix, iy) = (hx & 0x7fffffff, hy & 0x7fffffff);
    let ax = fabs(x);
    let ay = fabs(y);
    let x_neg = hx.lt(0);
    let y_neg = hy.lt(0);
    let (y_odd, y_int) = integer_parity(y);
    // s, the sign of the result: -1 for (negative)**(odd integer).
    let s = sel(x_neg & y_odd, c::<T>(-1.0), one);
    let finite_nonzero = ax.gt(T::of(0.0)) & ax.lt(T::of(f64::INFINITY));
    let axs = guard(finite_nonzero, ax, 1.0);
    // |y| > 2**31 (f32: 2**27): x near 1 or over/underflow. msun's powf
    // takes x below 1 - 10 * 2**-24 (word 0x3f7ffff6) to over/underflow, but
    // 11 to 13 ulps below 1 |y log x| can be as small as 88, which neither
    // overflows nor underflows; this port starts at 1 - 14 * 2**-24, where
    // |y log x| > 112 does, and computes the rest like x near 1.
    let (y_huge_w, below_w, above_w) =
        pick::<T, _>([(0x4d000000, 0x3f7ffff3, 0x3f800007), (0x41e00000, 0x3fefffff, 0x3ff00000)]);
    let y_huge = iy.gt(y_huge_w);
    let t = axs - T::of(1.0);
    let w_ = (t * t)
        * (c::<T>(0.5) - t * (c::<T>(pick::<T, _>([fb(0x3eaaaaab), db(0x3fd55555_55555555)])) - t * T::of(0.25)));
    let u = t * T::of(k.ivln2_h);
    let v = t * T::of(k.ivln2_l) - w_ * T::of(k.ivln2);
    let t1_near = trunc_bits(u + v, 0xfffff000);
    let t2_near = v - (t1_near - u);
    let (t1_log, t2_log) = log2_ext(axs);
    let t1 = sel(y_huge, t1_near, t1_log);
    let t2 = sel(y_huge, t2_near, t2_log);
    // Split y and compute (y1 + y2) * (t1 + t2).
    let ys = guard(ay.lt(T::of(f64::INFINITY)), y, 0.0);
    let y1 = trunc_bits(ys, 0xfffff000);
    let p_l = (ys - y1) * t1 + ys * t2;
    let p_h = y1 * t1;
    let z = p_l + p_h;
    let j = hi(z);
    // Overflow and underflow of 2^z.
    let (over, under) = if T::DOUBLE {
        let i = lo(z);
        let big = j.ge(0x40900000);
        let exactly = (j - 0x40900000).eq(0) & i.eq(0u32);
        let over = big & (!exactly | (p_l + T::of(k.ovt)).gt(z - p_h));
        let small = !big & (j & 0x7fffffff).ge(0x4090cc00);
        let exactly = j.eq(w(0xc090cc00)) & i.eq(0u32);
        let under = small & (!exactly | p_l.le(z - p_h));
        (over, under)
    } else {
        let over = j.gt(0x43000000) | (j.eq(0x43000000) & (p_l + T::of(k.ovt)).gt(z - p_h));
        let under = !j.gt(0x43000000)
            & !j.eq(0x43000000)
            & ((j & 0x7fffffff).gt(0x43160000) | (j.eq(w(0xc3160000)) & p_l.le(z - p_h)));
        (over, under)
    };
    // 2^(p_h + p_l): split off n = [z + 1/2] when |z| > 1/2.
    let in_range = !(over | under);
    let (p_h, p_l, j) = (guard(in_range, p_h, 0.0), guard(in_range, p_l, 0.0), sel(in_range, j, Int32::from(0)));
    let zz = p_l + p_h;
    let half_w = pick::<T, _>([0x3f000000, 0x3fe00000]);
    let round = (j & 0x7fffffff).gt(half_w);
    let tn = negate_if(j.lt(0), (fabs(zz) + T::of(0.5)).trunc());
    let tn = sel(round, tn, zero);
    let n = tn.to::<i32>();
    let p_h = p_h - tn;
    let t = trunc_bits(p_l + p_h, pick::<T, _>([0xffff8000, 0xfffff000]));
    let u = t * T::of(k.lg2_h);
    let v = (p_l - (t - p_h)) * T::of(k.lg2) + t * T::of(k.lg2_l);
    let z = u + v;
    let w2 = v - (z - u);
    let t = z * z;
    let t1 = z - t * horner(t, k.p);
    let r = (z * t1) / (t1 - T::of(2.0)) - (w2 + z * w2);
    let z = one - (r - z);
    let main = s * scale(z, n);
    let main = sel(over, s * infinity, sel(under, s * zero, main));
    // |y| huge: over/underflow unless x is close to one.
    let tiny_or_inf = |up: Boolean| sel(up, infinity, zero);
    let huge_r = sel(
        ix.lt(below_w),
        s * tiny_or_inf(y_neg),
        sel(ix.gt(above_w), s * tiny_or_inf(!y_neg & !y.eq(T::of(0.0))), main),
    );
    let huge_r = if T::DOUBLE {
        let very = iy.gt(0x43f00000);
        let v = sel(ix.le(0x3fefffff), tiny_or_inf(y_neg), tiny_or_inf(hy.gt(0)));
        sel(very, v, huge_r)
    } else {
        huge_r
    };
    let r = sel(y_huge, huge_r, main);
    // (x < 0)**(non-integer) is NaN.
    let r = sel(x_neg & !y_int, nan::<T>(), r);
    // x = +-0, +-inf, +-1.
    let xs_special = ax.eq(T::of(0.0)) | ax.eq(T::of(f64::INFINITY)) | ax.eq(T::of(1.0));
    let z8 = sel(y_neg, one / ax, ax);
    let z8 = sel(x_neg & ax.eq(T::of(1.0)) & !y_int, nan::<T>(), sel(x_neg & y_odd, fneg(z8), z8));
    let r = sel(xs_special, z8, r);
    // Special values of y.
    let r = sel(y.eq(T::of(0.5)) & hx.ge(0), x.sqrt(), r);
    let r = sel(y.eq(T::of(2.0)), x * x, r);
    let r = sel(y.eq(T::of(-1.0)), one / x, sel(y.eq(T::of(1.0)), x, r));
    let y_inf = iy.eq(inf_w) & lo(y).eq(0u32);
    let inf_r =
        sel(ax.eq(T::of(1.0)), one, sel(ax.gt(T::of(1.0)), sel(y_neg, zero, infinity), sel(y_neg, infinity, zero)));
    let r = sel(y_inf, inf_r, r);
    let r = sel(is_nan(x) | is_nan(y), nan::<T>(), r);
    let r = sel(x.eq(T::of(1.0)) | y.eq(T::of(0.0)), one, r);
    canonical(r)
}

/// `x^n` for an integer `n`: `pow(x, n)`, with the sign taken from `n`
/// itself (an f32 exponent above 2^24 has lost its parity).
pub(crate) fn fpowi<T: Real>(x: Value<T>, n: Int32) -> Value<T> {
    let nf = n.to::<T>();
    let magnitude = fabs(pow(fabs(x), nf));
    let odd = (n & 1).eq(1);
    let neg = sign_bit(x) & odd;
    let r = negate_if(neg, magnitude);
    // pown(x, 0) = 1, even for NaN.
    let r = sel(n.eq(0), c::<T>(1.0), sel(is_nan(x), nan::<T>(), r));
    canonical(r)
}

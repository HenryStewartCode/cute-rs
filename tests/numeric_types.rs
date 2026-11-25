//! The numeric type system: the type table against CuTe DSL's metadata,
//! the promotion table against CuTe DSL's rules, Boolean and TFloat32
//! conversions, and the narrow formats' codecs.

use cute::prelude::*;

/// CuTe DSL's metadata for each type: `(name, width, signed, is_float,
/// exponent_width, mantissa_width, mlir type)`. Booleans and FP8 formats are
/// stored as `i8` in cute-rs, and TFloat32 as `f32`.
type Meta = (&'static str, usize, bool, bool, Option<u32>, Option<u32>);

const CUTE: &[Meta] = &[
    ("Boolean", 1, true, false, None, None),
    ("Int4", 4, true, false, None, None),
    ("Int8", 8, true, false, None, None),
    ("Int16", 16, true, false, None, None),
    ("Int32", 32, true, false, None, None),
    ("Int64", 64, true, false, None, None),
    ("Int128", 128, true, false, None, None),
    ("Uint8", 8, false, false, None, None),
    ("Uint16", 16, false, false, None, None),
    ("Uint32", 32, false, false, None, None),
    ("Uint64", 64, false, false, None, None),
    ("Uint128", 128, false, false, None, None),
    ("BFloat16", 16, true, true, None, None),
    ("Float16", 16, true, true, None, None),
    ("Float32", 32, true, true, None, None),
    ("TFloat32", 32, true, true, None, None),
    ("Float64", 64, true, true, None, None),
    ("Float8E5M2", 8, true, true, Some(5), Some(2)),
    ("Float8E4M3", 8, true, true, Some(4), Some(3)),
    ("Float8E4M3FN", 8, true, true, Some(4), Some(3)),
    ("Float8E8M0FNU", 8, false, true, Some(8), Some(0)),
    ("Float8E4M3B11FNUZ", 8, true, true, Some(4), Some(3)),
    ("Float4E2M1FN", 4, true, true, Some(2), Some(1)),
    ("Float6E2M3FN", 6, true, true, Some(2), Some(3)),
    ("Float6E3M2FN", 6, true, true, Some(3), Some(2)),
    ("Float4E2M1FNx2", 8, true, true, Some(2), Some(1)),
    ("Float6E2M3FNx4", 24, true, true, Some(2), Some(3)),
    ("Float6E3M2FNx4", 24, true, true, Some(3), Some(2)),
    ("FloatNV8E5M3FNU", 8, false, true, None, None),
];

#[test]
fn type_table_matches_cute_metadata() {
    assert_eq!(DType::ALL.len(), CUTE.len());
    for &(name, width, signed, is_float, exp, mant) in CUTE {
        let t = DType::from_cute_name(name).unwrap_or_else(|| panic!("no type named {name}"));
        assert_eq!(t.cute_name(), name);
        assert_eq!(t.bits(), width, "{name}.width");
        assert_eq!(t.bytes(), width.div_ceil(8).max(1), "{name}.bytes");
        assert_eq!(t.n_bytes(3), 3 * t.bytes(), "{name}.n_bytes(3)");
        assert_eq!(t.info().signed, signed, "{name}.signed");
        assert_eq!(t.is_float(), is_float, "{name}.is_float");
        assert_eq!((t.exponent_width(), t.mantissa_width()), (exp, mant), "{name} exponent and mantissa widths");
        if !is_float {
            // T.min and T.max.
            let (lo, hi) = if signed {
                (i128::MIN >> (128 - width), (i128::MAX >> (128 - width)) as u128)
            } else {
                (0, u128::MAX >> (128 - width))
            };
            // CuTe DSL's Boolean.min and .max come out as -1 and 0 from its
            // signed 1-bit formula; cute-rs's are false and true.
            if name != "Boolean" {
                let min = t.min_value().unwrap();
                assert_eq!(min.bits128(), (lo as u128) & (u128::MAX >> (128 - width)), "{name}.min");
                assert_eq!(t.max_value().unwrap().bits128(), hi, "{name}.max");
            }
            assert!(t.inf().is_none() && t.nan().is_none());
        }
    }
    assert_eq!(DType::from_cute_name("Float17"), None);
    assert!(DType::F8E4M3FN.is_fp8() && DType::F8E5M2.is_fp8() && !DType::F8E4M3.is_fp8());
    assert!(!DType::F16.is_fp8() && !DType::F8E8M0FNU.is_fp8());
}

#[test]
fn float_specials_from_the_table() {
    let bits = |l: Option<cute::ir::Literal>| l.map(|l| l.bits128());
    assert_eq!(bits(DType::F32.inf()), Some(0x7f80_0000));
    assert_eq!(bits(DType::F16.inf()), Some(0x7c00));
    assert_eq!(bits(DType::BF16.nan()), Some(0x7fff));
    assert_eq!(bits(DType::F8E5M2.inf()), Some(0x7c));
    assert_eq!(bits(DType::F8E4M3.inf()), Some(0x78));
    assert_eq!(DType::F8E4M3FN.inf(), None);
    assert_eq!(bits(DType::F8E4M3FN.nan()), Some(0x7f));
    assert_eq!(bits(DType::F8E4M3B11FNUZ.nan()), Some(0x80));
    assert_eq!(bits(DType::F8E8M0FNU.nan()), Some(0xff));
    assert_eq!(DType::F8E8M0FNU.zero(), None);
    assert_eq!(bits(DType::F64.zero()), Some(0));
    assert_eq!(DType::F16.recast_width(32), Some(DType::F32));
    assert_eq!(DType::BF16.recast_width(16), Some(DType::F16));
    assert_eq!(DType::U8.recast_width(128), Some(DType::I128));
    assert_eq!(DType::F8E4M3FN.recast_width(8), None);
}

// The constants are usable in const contexts.
const HALVES_PER_VECTOR: usize = 128 / <F16 as Numeric>::WIDTH;
const _: () = assert!(HALVES_PER_VECTOR == 8);
const _: () = assert!(<bool as Numeric>::WIDTH == 1 && <bool as Numeric>::SIGNED && <bool as Numeric>::IS_INTEGER);
const _: () = assert!(!<u32 as Numeric>::SIGNED && <TF32 as Numeric>::IS_FLOAT && <F8E4M3FN as Numeric>::BYTES == 1);

#[test]
fn host_half_constants() {
    assert_eq!(F16::MAX.to_f32(), 65504.0);
    assert_eq!(F16::MIN.to_f32(), -65504.0);
    assert_eq!(F16::INFINITY.to_f32(), f32::INFINITY);
    assert!(F16::NAN.to_f32().is_nan());
    assert_eq!(BF16::MAX.to_f32(), f32::from_bits(0x7f7f_0000));
    assert_eq!(BF16::MIN.to_f32(), -f32::from_bits(0x7f7f_0000));
    assert_eq!(BF16::ONE.to_f32(), 1.0);
    assert_eq!(BF16::ZERO.to_f32(), 0.0);
    assert_eq!(BF16::INFINITY.to_f32(), f32::INFINITY);
    assert!(BF16::NAN.to_f32().is_nan());
    assert_eq!(BF16::from(1.5f32).to_string(), "1.5");
    assert_eq!(f32::from(BF16::from_f32(-2.0)), -2.0);
}

/// Every row of the generated promotion table is CuTe DSL's promotion (a
/// TFloat32 result computed in Float32), and every pair CuTe DSL promotes
/// has a row.
#[test]
fn promotion_table_follows_cute_rules() {
    let mut rows = std::collections::HashMap::new();
    for &(a, b, r) in PROMOTIONS {
        assert!(rows.insert((a, b), r).is_none(), "{a} + {b} twice");
    }
    let mut promoted = 0;
    for a in DType::ALL {
        for b in DType::ALL {
            if a == b {
                continue;
            }
            let cute = a.promote(b);
            let have = rows.get(&(a, b)).copied();
            match cute {
                Ok(r) if in_table(a) && in_table(b) => {
                    let r = if r == DType::TF32 { DType::F32 } else { r };
                    assert_eq!(have, Some(r), "{} with {}", a.cute_name(), b.cute_name());
                    promoted += 1;
                }
                Ok(_) => {}
                Err(_) => assert_eq!(have, None, "{} with {} has no promotion", a.cute_name(), b.cute_name()),
            }
        }
    }
    assert_eq!(promoted, rows.len());
}

/// Whether the table has rows for `t` (every type with a host type).
fn in_table(t: DType) -> bool {
    PROMOTIONS.iter().any(|&(x, _, _)| x == t)
}

#[test]
fn cute_promotion_examples() {
    use DType::*;
    let cases = [
        (F16, I32, F32),
        (F32, I64, F64),
        (BF16, I16, F16),
        (BF16, I8, BF16),
        (BF16, F16, F16),
        (TF32, F32, F32),
        (TF32, F16, TF32),
        (F8E4M3FN, I16, F16),
        (F8E5M2, F32, F32),
        (U32, I32, U32),
        (I64, U32, I64),
        (U8, I16, I16),
        (Bool, I32, I32),
        (Bool, U8, U8),
        (Bool, F16, F16),
    ];
    for (a, b, r) in cases {
        assert_eq!(a.promote(b), Ok(r), "{a} with {b}");
        assert_eq!(b.promote(a), Ok(r), "{b} with {a}");
    }
    for (a, b) in [(F8E4M3FN, F8E5M2), (F8E4M3FN, I8), (F8E5M2, Bool), (F8E8M0FNU, F8E4M3)] {
        assert!(a.promote(b).is_err() && b.promote(a).is_err(), "{a} with {b}");
    }
    assert!(F8E4M3FN.promote(F8E5M2).unwrap_err().contains("no implicit promotion"));
    // TFloat32 with Int32: the integer becomes Float32, which wins.
    assert_eq!(TF32.promote(I32), Ok(F32));
}

/// Mixed-type arithmetic has the promoted type (checked statically), and
/// computes there.
#[test]
fn mixed_arithmetic_types_and_values() {
    fn same<T>(_: &T, _: &T) {}
    let kernel = Kernel::build("mixed", || {
        let out = param_tensor::<f64>("out", Layout::col_major(16));
        let flags = param_tensor::<bool>("flags", Layout::col_major(4));
        let i: Int32 = Int32::constant(-7);
        let u: Uint32 = Uint32::constant(5);
        let h: Float16 = Float16::constant(F16::from_f32(1.5));
        let f: Float32 = Float32::constant(2.25);
        let l: Int64 = Int64::constant(1 << 40);
        let b: Boolean = Boolean::constant(true);
        let s: Int16 = Int16::constant(-300);
        let c: Uint8 = Uint8::constant(200);
        let t: TFloat32 = Float32::constant(0.5).to::<TF32>();
        let r0 = i + u; // Uint32
        same(&r0, &Uint32::constant(0));
        let r1 = h + i; // Float32
        same(&r1, &Float32::constant(0.0));
        let r2 = f * l; // Float64
        same(&r2, &Float64::constant(0.0));
        let r3 = b + i; // Int32
        same(&r3, &Int32::constant(0));
        let r4 = c + s; // Int16
        same(&r4, &Int16::constant(0));
        let r5 = t + t; // Float32 (TFloat32 arithmetic is computed in Float32)
        same(&r5, &Float32::constant(0.0));
        let r6 = b + b; // Boolean
        same(&r6, &Boolean::constant(false));
        let r7 = u % s; // Uint32: the unsigned type is the wider
        same(&r7, &Uint32::constant(0));
        let r8 = true_div(Int32::constant(7), 2);
        out.store(0, r0.to::<f64>());
        out.store(1, r1.to::<f64>());
        out.store(2, r2);
        out.store(3, r3.to::<f64>());
        out.store(4, r4.to::<f64>());
        out.store(5, r5.to::<f64>());
        out.store(6, r7.to::<f64>());
        out.store(7, r8.to::<f64>());
        out.store(8, (l >> u).to::<f64>());
        out.store(9, (c ^ i).to::<f64>());
        flags.store(0, r6);
        flags.store(1, i.lt(u)); // compared as Uint32: 2^32 - 7 < 5 is false
        flags.store(2, s.lt(c)); // compared as Int16: true
        flags.store(3, h.gt(f)); // compared as Float32: false
    });
    let mut args = vec![KernelArg::buffer(&[0f64; 16]), KernelArg::buffer(&[false; 4])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    let out = args[0].to_vec::<f64>();
    assert_eq!(out[0], (-7i32 as u32).wrapping_add(5) as f64);
    assert_eq!(out[1], -5.5);
    assert_eq!(out[2], 2.25 * (1u64 << 40) as f64);
    assert_eq!(out[3], -6.0);
    assert_eq!(out[4], -100.0);
    assert_eq!(out[5], 1.0);
    assert_eq!(out[6], (5u32 % (-300i16 as i32 as u32)) as f64);
    assert_eq!(out[7], 3.5);
    assert_eq!(out[8], ((1i64 << 40) >> 5) as f64);
    assert_eq!(out[9], (200 ^ -7) as f64);
    assert_eq!(args[1].to_vec::<bool>(), vec![true, false, true, false]);
}

/// CuTe DSL's `Boolean(x)`: nonzero integers and nonzero or NaN floats are
/// true.
#[test]
fn boolean_conversions_on_the_simulator() {
    let kernel = Kernel::build("to_bool", || {
        let fs = param_tensor::<f32>("fs", Layout::col_major(8));
        let is = param_tensor::<i32>("is", Layout::col_major(8));
        let out = param_tensor::<bool>("out", Layout::col_major((8, 6)));
        let t = thread_idx().x;
        let (f, i) = (fs.load(t), is.load(t));
        out.store((t, 0), f.to::<bool>());
        out.store((t, 1), f.to::<F16>().to::<bool>());
        out.store((t, 2), f.to::<f64>().to::<bool>());
        out.store((t, 3), i.to::<bool>());
        out.store((t, 4), i.to::<u8>().to::<bool>());
        out.store((t, 5), f.to::<TF32>().to::<bool>());
    });
    let fs = [0.0, -0.0, f32::NAN, 1e-30, -2.5, f32::INFINITY, 1e-45, 3.0];
    let is = [0, 1, -1, 256, 255, i32::MIN, 7, 0];
    let mut args = vec![KernelArg::buffer(&fs), KernelArg::buffer(&is), KernelArg::buffer(&[false; 48])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [8, 1, 1]), &mut args).unwrap();
    let out = args[2].to_vec::<bool>();
    for t in 0..8 {
        let (f, i) = (fs[t], is[t]);
        let h = F16::from_f32(f).to_f32();
        let tf = TF32::from_f32(f).to_f32();
        let want = [f != 0.0, h != 0.0, f != 0.0, i != 0, i as u8 != 0, tf != 0.0];
        for (j, w) in want.into_iter().enumerate() {
            assert_eq!(out[t + 8 * j], w, "value {t}, conversion {j}");
        }
    }
}

#[test]
fn tf32_host_rounding() {
    // Ties away from zero at the 13th bit.
    assert_eq!(TF32::from_f32(f32::from_bits(0x3f80_1000)).0.to_bits(), 0x3f80_2000);
    assert_eq!(TF32::from_f32(f32::from_bits(0x3f80_0fff)).0.to_bits(), 0x3f80_0000);
    assert_eq!(TF32::from_f32(f32::from_bits(0xbf80_1000)).0.to_bits(), 0xbf80_2000);
    // Past the largest TF32 value: infinity.
    assert_eq!(TF32::from_f32(f32::MAX).0, f32::INFINITY);
    assert!(TF32::from_f32(f32::NAN).0.is_nan());
}

/// One 8-bit float format through its host type.
struct Fmt {
    name: &'static str,
    ty: DType,
    decode: fn(u8) -> f64,
    encode: fn(f64, NarrowRounding, bool, bool) -> u8,
}

macro_rules! fmt {
    ($t:ident) => {
        Fmt {
            name: stringify!($t),
            ty: <$t as Numeric>::DTYPE,
            decode: |b| $t(b).to_f64(),
            encode: |x, r, sat, relu| $t::from_f64_with(x, r, sat, relu).0,
        }
    };
}

fn formats() -> [Fmt; 6] {
    [fmt!(F8E4M3FN), fmt!(F8E5M2), fmt!(F8E4M3), fmt!(F8E4M3B11FNUZ), fmt!(F8E8M0FNU), fmt!(F8E5M3FNU)]
}

/// The reference conversion, by search over every encoding rather than by
/// arithmetic on the format: the finite values, sorted, and the one `x`
/// rounds to. Ties go to the encoding with an even mantissa (the even
/// integer significand); past the largest finite value, by half a step for
/// to-nearest, a value overflows.
fn reference(f: &Fmt, x: f64, mode: NarrowRounding, sat: bool, relu: bool) -> u8 {
    let ff = f.ty.float_format().unwrap();
    let nan = DType::nan(f.ty).unwrap().bits128() as u8;
    if x.is_nan() {
        return nan;
    }
    let signed = ff.signed;
    let neg = signed && x.is_sign_negative();
    if relu && neg {
        return 0;
    }
    let a = x.abs();
    // Non-negative finite encodings with their values.
    let mut vals: Vec<(f64, u8)> = (0..=255u8)
        .map(|b| (b, (f.decode)(b)))
        .filter(|&(_, v)| v.is_finite() && v >= 0.0)
        .map(|(b, v)| (v, b))
        .collect();
    vals.retain(|&(v, b)| !(v == 0.0 && b != 0));
    vals.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
    vals.dedup_by(|p, q| p.0 == q.0);
    let (max, max_b) = *vals.last().unwrap();
    // The step past the largest value, to the next one an unbounded
    // exponent would give (twice the largest without a mantissa).
    let step = if ff.mant_bits == 0 { max } else { max - vals[vals.len() - 2].0 };
    let sign =
        |b: u8| if neg && signed && !(b == 0 && ff.nan == cute::ir::NanEncoding::NegativeZero) { b | 0x80 } else { b };
    // Overflow: an infinity of the sign, or the (unsigned) NaN.
    let inf = if ff.has_inf { sign(DType::inf(f.ty).unwrap().bits128() as u8) } else { nan };
    let toward_zero = mode == NarrowRounding::Zero || (mode == NarrowRounding::Up && neg);
    if a.is_infinite() {
        return if sat { sign(max_b) } else { inf };
    }
    // The neighbours below and above (the one above may be past the end).
    let i = vals.iter().rposition(|&(v, _)| v <= a);
    let lo = i.map(|i| vals[i]);
    let hi = match i {
        Some(i) if vals[i].0 == a => return sign(vals[i].1),
        Some(i) => vals.get(i + 1).copied(),
        None => Some(vals[0]),
    };
    let pick = |up: bool| -> u8 {
        match (up, lo, hi) {
            (false, Some((_, b)), _) => sign(b),
            // Below the smallest value of a format without zero.
            (false, None, Some((_, b))) => sign(b),
            (true, _, Some((_, b))) => sign(b),
            (true, Some(_), None) => {
                if sat || toward_zero {
                    sign(max_b)
                } else {
                    inf
                }
            }
            _ => unreachable!(),
        }
    };
    let hi_v = hi.map_or(max + step, |p| p.0);
    let lo_v = lo.map_or(0.0, |p| p.0);
    let up = match mode {
        NarrowRounding::Zero => false,
        NarrowRounding::Up => !neg,
        NarrowRounding::NearestAway => a - lo_v >= hi_v - a,
        NarrowRounding::NearestEven => {
            let (dl, dh) = (a - lo_v, hi_v - a);
            if dl != dh {
                dh < dl
            } else {
                // The even neighbour: the past-the-end value counts as the
                // next, odd or even, of the largest.
                let lo_b = lo.map_or(0, |p| p.1);
                lo_b % 2 == 1
            }
        }
    };
    if !ff.has_zero && lo.is_none() {
        // E8M0 below 2^-127: no zero, so the smallest value either way.
        return pick(true);
    }
    pick(up)
}

/// The f32 values every codec is held to the reference at: each format's
/// values, the midpoints between neighbours (ties), the f32 values next to
/// both, and values past the ends.
fn sweep(f: &Fmt) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..=255u8).map(f.decode).filter(|v| v.is_finite()).collect();
    vals.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let mut xs = vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-45, -1e-45, 1e30, -1e30];
    for w in vals.windows(2) {
        let mid = (w[0] + w[1]) / 2.0;
        for v in [w[0], mid, w[1] * 2.0] {
            let v32 = v as f32;
            for x in [v32, f32::from_bits(v32.to_bits() + 1), f32::from_bits(v32.to_bits().saturating_sub(1))] {
                xs.push(x as f64);
                xs.push(-x as f64);
            }
        }
    }
    let mut state = 7u64;
    for _ in 0..4000 {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        xs.push(f32::from_bits((state >> 32) as u32) as f64);
    }
    xs
}

#[test]
fn every_encoding_decodes_and_re_encodes() {
    for f in formats() {
        for b in 0..=255u8 {
            let v = (f.decode)(b);
            let back = (f.encode)(v, NarrowRounding::NearestEven, false, false);
            if v.is_nan() {
                assert_eq!(back, DType::nan(f.ty).unwrap().bits128() as u8, "{} NaN {b:#04x}", f.name);
            } else {
                assert_eq!(back, b, "{} {b:#04x} = {v}", f.name);
            }
        }
    }
    assert_eq!(F8E4M3FN::MAX.to_f32(), 448.0);
    assert_eq!(F8E4M3FN::MIN.to_f32(), -448.0);
    assert_eq!(F8E5M2::MAX.to_f32(), 57344.0);
    assert_eq!(F8E4M3::MAX.to_f32(), 240.0);
    assert_eq!(F8E4M3B11FNUZ::MAX.to_f32(), 30.0);
    assert_eq!(F8E8M0FNU::MAX.to_f64(), 2f64.powi(127));
    assert_eq!(F8E8M0FNU(0).to_f64(), 2f64.powi(-127));
    assert_eq!(F8E8M0FNU::MIN.to_f64(), 2f64.powi(-127));
    assert_eq!(F8E5M3FNU::MAX.to_f32(), 114688.0);
    assert!(F8E4M3FN::NAN.is_nan() && F8E4M3B11FNUZ(0x80).is_nan() && F8E8M0FNU(0xff).is_nan());
    assert_eq!(F8E4M3FN(0x01).to_f64(), 2f64.powi(-9));
    assert_eq!(F8E4M3B11FNUZ(0x01).to_f64(), 2f64.powi(-13));
}

#[test]
fn codecs_match_the_search_reference() {
    let modes = [NarrowRounding::NearestEven, NarrowRounding::Zero, NarrowRounding::Up, NarrowRounding::NearestAway];
    for f in formats() {
        for x in sweep(&f) {
            for mode in modes {
                for (sat, relu) in [(false, false), (true, false), (true, true), (false, true)] {
                    let have = (f.encode)(x, mode, sat, relu);
                    let want = reference(&f, x, mode, sat, relu);
                    assert_eq!(have, want, "{} of {x:e} ({mode:?}, satfinite {sat}, relu {relu})", f.name);
                }
            }
        }
    }
}

#[test]
fn host_fp8_conversions_saturate() {
    assert_eq!(F8E4M3FN::from_f32(464.0).to_f32(), 448.0);
    assert_eq!(F8E4M3FN::from_f32(1e9).to_f32(), 448.0);
    assert_eq!(F8E4M3FN::from_f32(f32::NEG_INFINITY).to_f32(), -448.0);
    assert_eq!(F8E5M2::from_f32(1e9).to_f32(), 57344.0);
    assert!(F8E5M2::from_f32(f32::NAN).is_nan());
    // Without .satfinite E5M2 overflows to infinity and E4M3FN to NaN.
    let rn = NarrowRounding::NearestEven;
    assert_eq!(F8E5M2::from_f64_with(1e9, rn, false, false).to_f32(), f32::INFINITY);
    assert!(F8E4M3FN::from_f64_with(1e9, rn, false, false).is_nan());
    // Ties to even: 1 + 2^-4 lies between 1 and 1.125 in E4M3.
    assert_eq!(F8E4M3FN::from_f32(1.0625).to_f32(), 1.0);
    assert_eq!(F8E4M3FN::from_f32(1.1875).to_f32(), 1.25);
    // E8M0 rounds up by default, and ignores the sign.
    assert_eq!(F8E8M0FNU::from_f32(3.0).to_f64(), 4.0);
    assert_eq!(F8E8M0FNU::from_f32(-0.25).to_f64(), 0.25);
    // FNUZ has no negative zero.
    assert_eq!(F8E4M3B11FNUZ::from_f32(-0.0).0, 0);
    let arg = KernelArg::fp8_from_f32::<F8E4M3FN>(&[1.0, -2.5, 1000.0, f32::NAN]);
    let back = arg.to_f32();
    assert_eq!(&back[..3], &[1.0, -2.5, 448.0]);
    assert!(back[3].is_nan());
    assert_eq!(KernelArg::buffer(&[F16::from_f32(0.5)]).to_f32(), vec![0.5]);
}

/// FP8 values promote with integers and 16-bit-or-wider floats.
#[test]
fn fp8_promotes_with_wider_types() {
    fn same<T>(_: &T, _: &T) {}
    let kernel = Kernel::build("fp8_promotion", || {
        let out = param_tensor::<f32>("out", Layout::col_major(4));
        let a = Value::constant(F8E4M3FN::from_f32(1.5));
        let b = Value::constant(F8E5M2::from_f32(-3.0));
        let h = Float16::constant(F16::from_f32(0.25));
        let i = Int32::constant(7);
        let s = Int16::constant(-2);
        let r0 = a + h; // Float16
        same(&r0, &h);
        let r1 = b * i; // Float32
        same(&r1, &Float32::constant(0.0));
        let r2 = a - s; // Float16
        same(&r2, &h);
        out.store(0, r0.to::<f32>());
        out.store(1, r1);
        out.store(2, r2.to::<f32>());
        out.store(3, select(a.lt(h), 1.0f32, 0.0f32)); // compared as Float16
    });
    let mut args = vec![KernelArg::buffer(&[0f32; 4])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<f32>(), vec![1.75, -21.0, 3.5, 0.0]);
}

/// Sub-byte host buffers pack element 0 into the low bits of byte 0 and read
/// back as many whole values as the bytes hold; a kernel declares them as
/// byte buffers of `ceil(n * bits / 8)` bytes.
#[test]
fn sub_byte_kernel_args_round_trip() {
    let values: Vec<I4> = (-8..8).chain([3, -1, 7]).map(I4::new).collect();
    let arg = KernelArg::buffer(&values);
    assert_eq!(arg.to_vec::<u8>().len(), 10);
    assert_eq!(arg.to_vec::<u8>()[0], 0x98);
    let back = arg.to_vec::<I4>();
    assert_eq!(&back[..values.len()], &values[..]);
    assert_eq!(back.len(), 20);
    // A byte buffer reads as Int4 too.
    assert_eq!(KernelArg::buffer(&[0x7fu8]).to_vec::<I4>(), vec![I4::new(-1), I4::new(7)]);
    let kernel = Kernel::build("int4_param", || {
        let p = param_tensor::<I4>("p", Layout::col_major(19));
        assert_eq!(p.size(), 19);
    });
    let decl = kernel.ir().buffer(cute::ir::BufferId(0));
    assert_eq!((decl.elem, decl.len), (DType::U8, Some(10)));
}

/// An OCP MX format under test: its name, exponent and mantissa bits, its
/// decode and encode, and its largest finite value.
type MxFormat = (&'static str, u32, u32, fn(u8) -> f64, fn(f64, bool) -> u8, f64);

/// The OCP MX FP4 and FP6 formats.
fn mx_formats() -> Vec<MxFormat> {
    vec![
        (
            "Float4E2M1FN",
            2,
            1,
            |b| F4E2M1FN(b).to_f64(),
            |x, relu| F4E2M1FN::from_f64_with(x, NarrowRounding::NearestEven, true, relu).0,
            6.0,
        ),
        (
            "Float6E2M3FN",
            2,
            3,
            |b| F6E2M3FN(b).to_f64(),
            |x, relu| F6E2M3FN::from_f64_with(x, NarrowRounding::NearestEven, true, relu).0,
            7.5,
        ),
        (
            "Float6E3M2FN",
            3,
            2,
            |b| F6E3M2FN(b).to_f64(),
            |x, relu| F6E3M2FN::from_f64_with(x, NarrowRounding::NearestEven, true, relu).0,
            28.0,
        ),
    ]
}

/// The value of an OCP MX encoding from the specification's definition:
/// bias `2^(e-1) - 1`, subnormals at exponent 0, no infinity or NaN.
fn mx_value(e: u32, m: u32, bits: u8) -> f64 {
    let sign = if bits >> (e + m) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = (bits >> m) as i32 & ((1 << e) - 1);
    let mant = (bits & ((1 << m) - 1)) as f64 / (1 << m) as f64;
    let bias = (1 << (e - 1)) - 1;
    sign * if exp == 0 { mant * 2f64.powi(1 - bias) } else { (1.0 + mant) * 2f64.powi(exp - bias) }
}

/// Every FP4 and FP6 encoding decodes to its value, and encodes back to
/// itself (negative zero included); MAX_NORM is the largest.
#[test]
fn mx_codecs_are_exact() {
    for (name, e, m, decode, encode, max) in mx_formats() {
        let n = 1u32 << (1 + e + m);
        let mut largest: f64 = 0.0;
        for b in 0..n as u8 {
            let want = mx_value(e, m, b);
            let got = decode(b);
            assert_eq!(got.to_bits(), want.to_bits(), "{name} {b:#04x}");
            assert_eq!(encode(got, false), b, "{name} {b:#04x} round trip");
            largest = largest.max(got);
        }
        assert_eq!(largest, max, "{name} MAX_NORM");
    }
    assert_eq!((F4E2M1FN::MAX.to_f32(), F4E2M1FN::MIN.to_f32()), (6.0, -6.0));
    assert_eq!((F6E3M2FN::MAX.to_f32(), F6E2M3FN::MAX.to_f32()), (28.0, 7.5));
}

/// `cvt.rn.satfinite{.relu}` into FP4 and FP6 against a reference that picks
/// the nearest of the format's values (ties to the even encoding),
/// saturates past MAX_NORM, makes a NaN +MAX_NORM and, with relu, a
/// negative value +0: over every f32 midpoint and value of each format,
/// their neighbours, and a sweep of f32 bit patterns.
#[test]
fn mx_rounding_matches_a_reference() {
    for (name, e, m, decode, encode, max) in mx_formats() {
        let mag_codes: Vec<u8> = (0..1u8 << (e + m)).collect();
        let sign_bit = 1u8 << (e + m);
        let reference = |x: f64, relu: bool| -> u8 {
            if x.is_nan() {
                return encode(max, false);
            }
            if relu && x.is_sign_negative() {
                return 0;
            }
            let a = x.abs();
            let mut best = 0u8;
            for &c in &mag_codes {
                let (v, bv) = (decode(c), decode(best));
                if (a - v).abs() < (a - bv).abs() || ((a - v).abs() == (a - bv).abs() && c % 2 == 0 && best % 2 == 1) {
                    best = c;
                }
            }
            if a >= max {
                best = encode(max, false);
            }
            if x.is_sign_negative() { best | sign_bit } else { best }
        };
        let mut xs: Vec<f64> = vec![f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1e30, -1e30];
        for c in 0..(1u8 << (e + m)) {
            let v = decode(c);
            let next = if c + 1 < 1 << (e + m) { decode(c + 1) } else { max * 2.0 };
            for x in [v, (v + next) / 2.0, v * (1.0 + 1e-6), (v + next) / 2.0 * (1.0 - 1e-6), next * 0.75] {
                xs.extend([x, -x]);
            }
        }
        let mut state = 0x1234_5678u32;
        for _ in 0..20000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            xs.push(f32::from_bits(state) as f64);
        }
        for &x in &xs {
            for relu in [false, true] {
                // Through f32, as the cvt reads its operands.
                let x = x as f32 as f64;
                assert_eq!(encode(x, relu), reference(x, relu), "{name} of {x:e} (relu {relu})");
            }
        }
    }
}

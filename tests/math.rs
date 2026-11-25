//! The accuracy of `cute::math` on the simulator.
//!
//! Every function runs on every floating-point type over a sweep of inputs
//! (signed zeros, infinities, NaN, subnormals, the largest values, known
//! hard cases and their neighbours, uniform samples of the interesting
//! ranges and random bit patterns) and is held to a double-double reference
//! (`math_ref`, written independently of the msun-derived code) within the
//! bound the function documents: special values must be exact, with the
//! sign of a zero, and other results within the documented number of ulps.
//! Fast mode is held to its documented bounds, which the simulator (whose
//! approximations are correctly rounded) meets easily; `tests/gpu.rs` holds
//! the GPU's approximations to them.
//!
//! `CUTE_MATH_SWEEP=n` sets the number of inputs per function and type
//! (default 2^14; at 65536 or more the 16-bit types get every bit pattern).

mod math_cases;
mod math_ref;

use cute::prelude::*;
use math_cases::*;
use math_ref::{Big, Want, ulp_error};

fn sweep() -> usize {
    std::env::var("CUTE_MATH_SWEEP").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 14)
}

/// Run a kernel on the simulator and return its last buffer's raw bits.
fn simulate(kernel: &Kernel, n: usize, mut args: Vec<KernelArg>, ty: Ty) -> Vec<u64> {
    kernel.simulate(launch(n), &mut args).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    ty.bits(args.last().unwrap())
}

fn report(what: String, stats: &Stats, bound: f64) {
    eprintln!("{}", stats.summary(&what));
    assert!(
        stats.failures == 0,
        "{what}: {} results exceed {bound} ulp, first {:?}",
        stats.failures,
        stats.first_failure
    );
}

/// The precise-mode sweep of a unary function on `ty`.
fn precise_unary(name: &'static str, ty: Ty) -> Vec<u64> {
    let case = unary_case(name);
    let xs = unary_inputs(&case, sweep(), ty);
    let n = xs.len();
    let bits = simulate(&unary_kernel(name, ty, n, false), n, unary_args(ty, &xs), ty);
    let bound = case.bound(ty);
    let mut stats = Stats::default();
    for (&x, &b) in xs.iter().zip(&bits) {
        let got = ty.value(b);
        stats.record(&[x], got, ulp_error(got, (case.reference)(x), ty.format()), bound);
    }
    report(format!("{name} {}", ty.name()), &stats, bound);
    bits
}

/// The fast-mode sweep of a unary function: within the documented bound
/// on Float32 (and on Float16 and BFloat16, plus their rounding), and the
/// same bits as precise mode on Float64 and for functions fastmath leaves
/// alone. A subnormal Float32 argument may be flushed to zero.
fn fast_unary(name: &'static str, ty: Ty, precise_bits: &[u64]) {
    let case = unary_case(name);
    let xs = unary_inputs(&case, sweep(), ty);
    let n = xs.len();
    let bits = simulate(&unary_kernel(name, ty, n, true), n, unary_args(ty, &xs), ty);
    let Some(bound) = case.fast.filter(|_| ty != Ty::F64) else {
        assert!(bits == precise_bits, "{name} {}: fastmath changed a result that has no fast form", ty.name());
        return;
    };
    let mut checked = 0;
    for (&x, &b) in xs.iter().zip(&bits) {
        let got = ty.value(b);
        let ok = fast_within(ty, x, 0.0, got, &(case.reference)(x), bound);
        let ok_flushed = flushed(x).and_then(|z| fast_within(ty, z, 0.0, got, &(case.reference)(z), bound));
        match (ok, ok_flushed) {
            (None, None) => continue,
            (a, b) => {
                let w = approx(&(case.reference)(x));
                assert!(
                    a == Some(true) || b == Some(true),
                    "{name} fast {}({x:e}) = {got:e}, true value {w:e}",
                    ty.name()
                );
            }
        }
        checked += 1;
    }
    assert!(checked > n / 8, "{name} fast {}: only {checked} of {n} results have a bound", ty.name());
}

macro_rules! unary_tests {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            for ty in TYPES {
                let precise = precise_unary($name, ty);
                fast_unary($name, ty, &precise);
            }
        }
    )*};
}

unary_tests! {
    exp => "exp",
    exp2 => "exp2",
    expm1 => "expm1",
    log => "log",
    log2 => "log2",
    log10 => "log10",
    log1p => "log1p",
    sin => "sin",
    cos => "cos",
    tan => "tan",
    asin => "asin",
    acos => "acos",
    atan => "atan",
    sinh => "sinh",
    cosh => "cosh",
    tanh => "tanh",
    asinh => "asinh",
    acosh => "acosh",
    atanh => "atanh",
    erf => "erf",
    erfc => "erfc",
    cbrt => "cbrt",
    sqrt => "sqrt",
    rsqrt => "rsqrt",
    rcp => "rcp",
    round => "round",
}

/// `sincos` returns the same bits as `sin` and `cos`, in both modes.
#[test]
fn sincos_is_sin_and_cos() {
    for ty in TYPES {
        for fastmath in [false, true] {
            let case = unary_case("sin");
            let xs = unary_inputs(&case, sweep() / 4, ty);
            let n = xs.len();
            let run = |name| simulate(&unary_kernel(name, ty, n, fastmath), n, unary_args(ty, &xs), ty);
            assert!(run("sincos.sin") == run("sin"), "sincos {} (fastmath {fastmath}): sine differs", ty.name());
            assert!(run("sincos.cos") == run("cos"), "sincos {} (fastmath {fastmath}): cosine differs", ty.name());
        }
    }
}

fn precise_binary(name: &'static str, ty: Ty) -> Vec<u64> {
    let case = binary_case(name);
    let pairs = (case.inputs)(sweep(), ty);
    let n = pairs.len();
    let bits = simulate(&binary_kernel(name, ty, n, false), n, binary_args(ty, &pairs), ty);
    let bound = case.bound(ty);
    let mut stats = Stats::default();
    for (&(x, y), &b) in pairs.iter().zip(&bits) {
        let got = ty.value(b);
        stats.record(&[x, y], got, ulp_error(got, (case.reference)(x, y), ty.format()), bound);
    }
    report(format!("{name} {}", ty.name()), &stats, bound);
    bits
}

fn fast_binary(name: &'static str, ty: Ty, precise_bits: &[u64]) {
    let case = binary_case(name);
    let pairs = (case.inputs)(sweep(), ty);
    let n = pairs.len();
    let bits = simulate(&binary_kernel(name, ty, n, true), n, binary_args(ty, &pairs), ty);
    let Some(bound) = case.fast.filter(|_| ty != Ty::F64) else {
        assert!(bits == precise_bits, "{name} {}: fastmath changed a result that has no fast form", ty.name());
        return;
    };
    let mut checked = 0;
    for (&(x, y), &b) in pairs.iter().zip(&bits) {
        let got = ty.value(b);
        let Some(ok) = fast_within(ty, x, y, got, &(case.reference)(x, y), bound) else { continue };
        let w = approx(&(case.reference)(x, y));
        assert!(ok, "{name} fast {}({x:e}, {y:e}) = {got:e}, true value {w:e}", ty.name());
        checked += 1;
    }
    assert!(checked > n / 8, "{name} fast {}: only {checked} of {n} results have a bound", ty.name());
}

macro_rules! binary_tests {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            for ty in TYPES {
                let precise = precise_binary($name, ty);
                fast_binary($name, ty, &precise);
            }
        }
    )*};
}

binary_tests! {
    atan2 => "atan2",
    pow => "pow",
    rem => "rem",
    copysign => "copysign",
    div => "div",
}

/// `fpowi` is `pown`: within 1 ulp of `base^n` (for Float32, exponents up to
/// 2^24 in magnitude, which it converts exactly), with the sign of an odd
/// power and `pown(x, 0) = 1` even for NaN.
#[test]
fn fpowi() {
    for ty in TYPES {
        let cases = fpowi_inputs(ty, sweep() / 4);
        let n = cases.len();
        let kernel = fpowi_kernel(ty, n);
        let (base, exponent): (Vec<f64>, Vec<i32>) = cases.iter().copied().unzip();
        let args = vec![ty.buffer(&base), KernelArg::buffer(&exponent), ty.buffer(&vec![0.0; n])];
        let bits = simulate(&kernel, n, args, ty);
        let bound = if matches!(ty, Ty::F32 | Ty::F64) { 1.0 } else { 0.501 };
        let mut stats = Stats::default();
        for (&(b, k), &g) in cases.iter().zip(&bits) {
            let got = ty.value(g);
            // A Float32 exponent beyond 2^24 is rounded: only check the sign.
            if ty != Ty::F64 && k.unsigned_abs() > 1 << 24 {
                let negative = b.is_sign_negative() && k % 2 != 0;
                assert!(
                    got.is_nan() || got.is_sign_negative() == negative,
                    "fpowi {}({b:e}, {k}) = {got:e}",
                    ty.name()
                );
                continue;
            }
            let w = if k == 0 {
                Want::Exact(1.0)
            } else if b.is_nan() {
                Want::Exact(f64::NAN)
            } else {
                pow_want(b, k as f64)
            };
            stats.record(&[b, k as f64], got, ulp_error(got, w, ty.format()), bound);
        }
        report(format!("fpowi {}", ty.name()), &stats, bound);
    }
}

/// `ipowi` wraps like Rust's `wrapping_pow`; a negative exponent gives 0
/// except for the bases 1 and -1.
#[test]
fn ipowi() {
    let pairs = ipowi_pairs();
    let n = pairs.len();
    let negative_power = |b: i64, e: i64| -> i64 {
        match b {
            1 => 1,
            -1 if e % 2 == 0 => 1,
            -1 => -1,
            _ => 0,
        }
    };
    let (a, e): (Vec<i32>, Vec<i32>) = pairs.iter().map(|&(b, e)| (b as i32, e as i32)).unzip();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&e), KernelArg::buffer(&vec![0i32; n])];
    ipowi_kernel::<i32>(n).simulate(launch(n), &mut args).unwrap();
    for (k, &got) in args[2].to_vec::<i32>().iter().enumerate() {
        let w =
            if e[k] >= 0 { a[k].wrapping_pow(e[k] as u32) } else { negative_power(a[k] as i64, e[k] as i64) as i32 };
        assert_eq!(got, w, "ipowi i32 ({}, {})", a[k], e[k]);
    }
    let (a, e): (Vec<i64>, Vec<i64>) = pairs.iter().copied().unzip();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&e), KernelArg::buffer(&vec![0i64; n])];
    ipowi_kernel::<i64>(n).simulate(launch(n), &mut args).unwrap();
    for (k, &got) in args[2].to_vec::<i64>().iter().enumerate() {
        let w = if e[k] >= 0 { a[k].wrapping_pow(e[k] as u32) } else { negative_power(a[k], e[k]) };
        assert_eq!(got, w, "ipowi i64 ({}, {})", a[k], e[k]);
    }
    // Unsigned: the exponent is unsigned too.
    let (a, e): (Vec<u32>, Vec<u32>) = pairs.iter().map(|&(b, e)| (b as u32, e.unsigned_abs() as u32)).unzip();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&e), KernelArg::buffer(&vec![0u32; n])];
    ipowi_kernel::<u32>(n).simulate(launch(n), &mut args).unwrap();
    for (k, &got) in args[2].to_vec::<u32>().iter().enumerate() {
        assert_eq!(got, a[k].wrapping_pow(e[k]), "ipowi u32 ({}, {})", a[k], e[k]);
    }
}

/// The exact functions on every type: floor, ceil, trunc, roundeven, abs,
/// absf and neg.
#[test]
fn exact_unary_functions() {
    for ty in TYPES {
        let xs = exact_inputs(ty);
        let n = xs.len();
        let kernel = exact_kernel(ty, n);
        let args = vec![ty.buffer(&xs), ty.buffer(&vec![0.0; 7 * n])];
        let bits = simulate(&kernel, n, args, ty);
        for (i, &x) in xs.iter().enumerate() {
            let want = [x.floor(), x.ceil(), x.trunc(), x.round_ties_even(), x.abs(), x.abs(), -x];
            for (k, w) in want.into_iter().enumerate() {
                let got = ty.value(bits[i * 7 + k]);
                let same = (got.is_nan() && w.is_nan()) || got.to_bits() == w.to_bits();
                assert!(same, "{} function {k} of {x:e}: {got:e}, not {w:e}", ty.name());
            }
        }
    }
}

/// `isnan`, `isinf`, `isfinite` and `isnormal` (false for zeros and
/// subnormals) on every type; every bit pattern of the 16-bit types.
#[test]
fn classification() {
    for ty in TYPES {
        let xs: Vec<f64> =
            if matches!(ty, Ty::F16 | Ty::BF16) { (0..65536).map(|b| ty.value(b)).collect() } else { exact_inputs(ty) };
        let n = xs.len();
        let kernel = classify_kernel(ty, n);
        let mut args = vec![ty.buffer(&xs), KernelArg::buffer(&vec![false; 4 * n])];
        kernel.simulate(launch(n), &mut args).unwrap();
        let out = args[1].to_vec::<bool>();
        let min_normal = 2f64.powi(ty.format().emin as i32);
        for (i, &x) in xs.iter().enumerate() {
            let want = [x.is_nan(), x.is_infinite(), x.is_finite(), x.is_finite() && x.abs() >= min_normal];
            assert_eq!(&out[4 * i..4 * i + 4], &want, "{} classification of {x:e}", ty.name());
        }
    }
}

/// The exact value of operation `k` of [`options_kernel`].
fn exact_op(k: usize, a: f64, b: f64) -> Big {
    let (ba, bb) = (Big::of(a), Big::of(b));
    match k {
        0 => ba.add(bb),
        1 => ba.mul(bb),
        2 => ba.mul(bb).add(ba),
        3 => ba.div(bb),
        4 => math_ref::sqrt(a.abs()),
        _ => Big::of(1.0).div(bb),
    }
}

/// `add`, `mul`, `fma`, `div`, `sqrt` and `rcp` with `rounding=` and `ftz=`:
/// round to nearest matches IEEE arithmetic, the directed roundings bracket
/// the exact result one ulp apart and round in their direction, and `ftz`
/// flushes subnormal operands and results.
#[test]
fn rounding_and_ftz_options() {
    use cute::math::RoundingMode as R;
    for ty in [Ty::F32, Ty::F64] {
        let mut state = 1234u64;
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        while pairs.len() < 4096 {
            let p = match rng(&mut state) % 3 {
                0 => (any(&mut state, ty), any(&mut state, ty)),
                1 => (uniform(&mut state, -10.0, 10.0), uniform(&mut state, -10.0, 10.0)),
                _ => (uniform(&mut state, -1.0, 1.0) * 1e-38, uniform(&mut state, 0.5, 2.0)),
            };
            pairs.push((ty.round(p.0), ty.round(p.1)));
        }
        let n = pairs.len();
        let run = |opts: MathOptions| -> Vec<f64> {
            let kernel = options_kernel(ty, n, opts);
            let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
            let args = vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; 6 * n])];
            simulate(&kernel, n, args, ty).into_iter().map(|bits| ty.value(bits)).collect()
        };
        let nearest = run(MathOptions::rounding(R::NearestEven));
        let up = run(MathOptions::rounding(R::Up));
        let down = run(MathOptions::rounding(R::Down));
        let zero = run(MathOptions::rounding(R::Zero));
        let plain = run(MathOptions::default());
        let le = |x: f64, y: &Big| x == f64::NEG_INFINITY || (x.is_finite() && Big::of(x).sub(*y).m.hi <= 0.0);
        let ge = |x: f64, y: &Big| x == f64::INFINITY || (x.is_finite() && Big::of(x).sub(*y).m.hi >= 0.0);
        for (i, &(a, b)) in pairs.iter().enumerate() {
            for k in 0..6 {
                let j = i * 6 + k;
                let (rn, ru, rd, rz, p) = (nearest[j], up[j], down[j], zero[j], plain[j]);
                assert!(rn.to_bits() == p.to_bits() || (rn.is_nan() && p.is_nan()), "op {k}({a:e}, {b:e}): rn {rn:e}");
                if rn.is_nan() || !a.is_finite() || !b.is_finite() || (k >= 3 && b == 0.0) {
                    continue;
                }
                let exact = exact_op(k, a, b);
                assert!(le(rd, &exact) && ge(ru, &exact), "op {k}({a:e}, {b:e}): {rd:e} .. {ru:e} misses the result");
                assert!(
                    rd == ru || rd.next_up_in(ty) == ru,
                    "op {k}({a:e}, {b:e}): {rd:e} and {ru:e} are not adjacent"
                );
                let toward_zero = if exact.m.hi >= 0.0 { rd } else { ru };
                assert!(rz == toward_zero, "op {k}({a:e}, {b:e}): rz {rz:e}, not {toward_zero:e}");
            }
        }
        if ty == Ty::F32 {
            // ftz: subnormal operands and results are flushed to zero.
            let ftz = run(MathOptions::rounding(R::NearestEven).with_ftz());
            let min_normal = f32::MIN_POSITIVE as f64;
            let flush = |v: f64| if v.abs() < min_normal { 0.0f64.copysign(v) } else { v };
            for (i, &(a, b)) in pairs.iter().enumerate() {
                let want = flush((flush(a) as f32 + flush(b) as f32) as f64);
                let got = ftz[i * 6];
                let ok = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
                assert!(ok, "add.rn.ftz({a:e}, {b:e}) = {got:e}, not {want:e}");
            }
        }
    }
}

/// The next value of the type after `self`.
trait NextUp {
    fn next_up_in(self, ty: Ty) -> f64;
}

impl NextUp for f64 {
    fn next_up_in(self, ty: Ty) -> f64 {
        match ty {
            Ty::F64 => self.next_up(),
            _ => (self as f32).next_up() as f64,
        }
    }
}

/// `min` and `max` ignore a NaN operand (IEEE 754 minimumNumber); with
/// `propagate_nan` they return NaN (IEEE 754 minimum); both order -0 below
/// +0; `clamp` is `max(lo, min(x, hi))`.
#[test]
fn min_max_and_clamp() {
    for ty in TYPES {
        let pairs = minmax_inputs(ty);
        let n = pairs.len();
        let kernel = minmax_kernel(ty, n);
        let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
        let args = vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; 6 * n])];
        let bits = simulate(&kernel, n, args, ty);
        let minimum = |x: f64, y: f64| -> f64 {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else if x == y {
                if x.is_sign_negative() { x } else { y }
            } else {
                x.min(y)
            }
        };
        let maximum = |x: f64, y: f64| -> f64 {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else if x == y {
                if x.is_sign_negative() { y } else { x }
            } else {
                x.max(y)
            }
        };
        // minimumNumber and maximumNumber, which also order -0 below +0 (as
        // PTX's min and max do).
        let minimum_number = |x: f64, y: f64| {
            if x.is_nan() {
                y
            } else if y.is_nan() {
                x
            } else {
                minimum(x, y)
            }
        };
        let maximum_number = |x: f64, y: f64| {
            if x.is_nan() {
                y
            } else if y.is_nan() {
                x
            } else {
                maximum(x, y)
            }
        };
        let same = |g: f64, w: f64| (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
        for (i, &(x, z)) in pairs.iter().enumerate() {
            let got: Vec<f64> = (0..6).map(|k| ty.value(bits[i * 6 + k])).collect();
            let name = ty.name();
            assert!(same(got[0], minimum_number(x, z)), "{name} min({x:e}, {z:e}) = {:e}", got[0]);
            assert!(same(got[1], maximum_number(x, z)), "{name} max({x:e}, {z:e}) = {:e}", got[1]);
            assert!(same(got[2], minimum(x, z)), "{name} minimum({x:e}, {z:e}) = {:e}", got[2]);
            assert!(same(got[3], maximum(x, z)), "{name} maximum({x:e}, {z:e}) = {:e}", got[3]);
            let clamped = maximum_number(z, minimum_number(x, 0.0));
            assert!(same(got[4], clamped), "{name} clamp({x:e}, {z:e}, 0) = {:e}", got[4]);
            let (lo, hi) = (-z.abs(), z.abs());
            assert!(same(got[5], maximum(lo, minimum(x, hi))), "{name} clamp NaN({x:e}, {z:e}) = {:e}", got[5]);
        }
    }
}

/// `abs`, `-x`/`neg` and `copysign` are bit operations on every type: a NaN
/// keeps its payload, also when the operation is alone in a kernel, applied
/// to a product, given a constant sign or followed by integer arithmetic on
/// its bits (where LLVM used to emit `and.b32`, `abs.f32`, `neg.f32` or an
/// integer add, each with different NaNs).
#[test]
fn sign_bit_operations() {
    for ty in TYPES {
        let xs = sign_bit_inputs(ty);
        let ss = sign_bit_partners(ty);
        let n = xs.len();
        for op in PLAIN_OPS {
            let mut args = vec![ty.buffer_bits(&xs), ty.buffer_bits(&ss), ty.buffer_bits(&vec![0; n])];
            plain_kernel(op, ty, n).simulate(launch(n), &mut args).unwrap();
            for (i, &got) in ty.bits(&args[2]).iter().enumerate() {
                let want = plain_want(op, ty, xs[i], ss[i]);
                assert_eq!(got, want, "{} {op}({:#x}, {:#x}): {got:#x}, not {want:#x}", ty.name(), xs[i], ss[i]);
            }
        }
        let mut args = vec![
            ty.buffer_bits(&xs),
            ty.buffer_bits(&ss),
            ty.buffer_bits(&vec![0; n * SIGN_CONTEXT_FLOATS]),
            ty.int_buffer(n * SIGN_CONTEXT_INTS),
        ];
        sign_context_kernel(ty, n).simulate(launch(n), &mut args).unwrap();
        let (floats, ints) = (ty.bits(&args[2]), ty.int_bits(&args[3]));
        for i in 0..n {
            let (x, s) = (xs[i], ss[i]);
            let p = floats[i * SIGN_CONTEXT_FLOATS + 3];
            // The product, correctly rounded (exact in f64 for the narrower
            // types before rounding).
            let product = match ty {
                Ty::F64 => ty.value(x) * ty.value(s),
                _ => ty.round(ty.value(x) * ty.value(s)),
            };
            let got = ty.value(p);
            let same = (got.is_nan() && product.is_nan()) || got.to_bits() == product.to_bits();
            assert!(same, "{} {:#x} * {:#x} = {got:e}, not {product:e}", ty.name(), x, s);
            let (want_floats, want_ints) = sign_context_want(ty, x, s, p);
            for (k, want) in want_floats.iter().enumerate() {
                let got = floats[i * SIGN_CONTEXT_FLOATS + k];
                if let Some(want) = *want {
                    assert_eq!(got, want, "{} context column {k} for ({x:#x}, {s:#x}): {got:#x}", ty.name());
                }
            }
            for (k, &want) in want_ints.iter().enumerate() {
                let got = ints[i * SIGN_CONTEXT_INTS + k];
                assert_eq!(got, want, "{} integer column {k} for ({x:#x}, {s:#x}): {got:#x}", ty.name());
            }
        }
    }
}

/// `abs`, `min`, `max` and `clamp` with `ftz=True`: subnormal operands
/// flush to zeros of their sign, `min` and `max` ignore one NaN operand
/// unless `propagate_nan`, and -0 orders below +0.
#[test]
fn ftz_abs_min_max_and_clamp() {
    let pairs = ftz_inputs();
    let n = pairs.len();
    let (a, b): (Vec<u64>, Vec<u64>) = pairs.iter().copied().unzip();
    let mut args = vec![Ty::F32.buffer_bits(&a), Ty::F32.buffer_bits(&b), Ty::F32.buffer_bits(&vec![0; n * FTZ_OPS])];
    ftz_kernel(n).simulate(launch(n), &mut args).unwrap();
    let out = Ty::F32.bits(&args[2]);
    let names = ["abs", "min", "max", "min NaN", "max NaN", "clamp", "clamp NaN"];
    for (i, &(x, z)) in pairs.iter().enumerate() {
        for (k, want) in ftz_want(x, z).into_iter().enumerate() {
            let got = out[i * FTZ_OPS + k];
            match want {
                Some(w) => assert_eq!(got, w, "{} ftz({x:#x}, {z:#x}): {got:#x}, not {w:#x}", names[k]),
                None => assert!(Ty::F32.is_nan_bits(got), "{} ftz({x:#x}, {z:#x}): {got:#x}, not NaN", names[k]),
            }
        }
    }
}

/// `absi` and `abs` on integers: the wrapping absolute value of a signed
/// integer (`absi(MIN)` is `MIN`), an unsigned one unchanged.
#[test]
fn integer_abs() {
    fn check<T: Integer + Bitwise>(want: fn(u64) -> u64) {
        let xs = absi_inputs(T::DTYPE.bits() as u32);
        let n = xs.len();
        let values: Vec<T> = xs.iter().map(|&b| T::from_bits(b)).collect();
        let mut args = vec![KernelArg::buffer(&values), KernelArg::buffer(&vec![T::from_bits(0); 2 * n])];
        absi_kernel::<T>(n).simulate(launch(n), &mut args).unwrap();
        let out = args[1].to_vec::<T>();
        for (i, &x) in xs.iter().enumerate() {
            for k in 0..2 {
                let got = out[2 * i + k].raw_bits();
                assert_eq!(got, want(x), "{} absi/abs {k} of {x:#x}: {got:#x}", T::DTYPE);
            }
        }
    }
    check::<i8>(|x| (x as i8).wrapping_abs() as u8 as u64);
    check::<i16>(|x| (x as i16).wrapping_abs() as u16 as u64);
    check::<i32>(|x| (x as i32).wrapping_abs() as u32 as u64);
    check::<i64>(|x| (x as i64).wrapping_abs() as u64);
    check::<u32>(|x| x);
    check::<u8>(|x| x);
}

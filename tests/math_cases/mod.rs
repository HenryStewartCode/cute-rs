//! The `cute::math` functions under test: their inputs, high-precision
//! references (`math_ref`) and error bounds. `tests/math.rs` holds the
//! simulator to the references; `tests/gpu.rs` holds the GPU to the
//! simulator (precise mode, bit for bit) and to the references (fast mode).

#![allow(dead_code)]

use cute::prelude::*;

use super::math_ref::{
    self as r, BF16 as BF16_FORMAT, Big, Dd, F16 as F16_FORMAT, F32, F64, Format, Want, round_to_format,
};

unsafe extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

/// C's `erf` from the host's libm (for special values only: it is not
/// accurate enough to be a reference).
pub fn libm_erf(x: f64) -> f64 {
    unsafe { erf(x) }
}

pub fn libm_erfc(x: f64) -> f64 {
    unsafe { erfc(x) }
}

/// The floating-point types `cute::math` covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F16,
    BF16,
    F32,
    F64,
}

pub const TYPES: [Ty; 4] = [Ty::F32, Ty::F64, Ty::F16, Ty::BF16];

impl Ty {
    pub fn format(self) -> Format {
        match self {
            Ty::F16 => F16_FORMAT,
            Ty::BF16 => BF16_FORMAT,
            Ty::F32 => F32,
            Ty::F64 => F64,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Ty::F16 => "f16",
            Ty::BF16 => "bf16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
        }
    }

    /// `x` rounded to this type, as an (exact) f64.
    pub fn round(self, x: f64) -> f64 {
        match self {
            Ty::F64 => x,
            _ => r::round_to_format(x, self.format()),
        }
    }

    /// The f64 value of this type's bit pattern `bits`.
    pub fn value(self, bits: u64) -> f64 {
        match self {
            Ty::F16 => F16(bits as u16).to_f32() as f64,
            Ty::BF16 => BF16(bits as u16).to_f32() as f64,
            Ty::F32 => f32::from_bits(bits as u32) as f64,
            Ty::F64 => f64::from_bits(bits),
        }
    }

    /// A buffer of these values, rounded to the type (NaN stays NaN).
    pub fn buffer(self, xs: &[f64]) -> KernelArg {
        match self {
            Ty::F16 => KernelArg::buffer(&xs.iter().map(|&x| F16::from_f32(x as f32)).collect::<Vec<_>>()),
            Ty::BF16 => KernelArg::buffer(&xs.iter().map(|&x| BF16::from_f32(x as f32)).collect::<Vec<_>>()),
            Ty::F32 => KernelArg::buffer(&xs.iter().map(|&x| x as f32).collect::<Vec<_>>()),
            Ty::F64 => KernelArg::buffer(xs),
        }
    }

    /// The raw bits of a buffer of this type.
    pub fn bits(self, arg: &KernelArg) -> Vec<u64> {
        match self {
            Ty::F16 => arg.to_vec::<F16>().iter().map(|v| v.0 as u64).collect(),
            Ty::BF16 => arg.to_vec::<BF16>().iter().map(|v| v.0 as u64).collect(),
            Ty::F32 => arg.to_vec::<f32>().iter().map(|v| v.to_bits() as u64).collect(),
            Ty::F64 => arg.to_vec::<f64>().iter().map(|v| v.to_bits()).collect(),
        }
    }

    /// A buffer's values as f64.
    pub fn values(self, arg: &KernelArg) -> Vec<f64> {
        self.bits(arg).into_iter().map(|b| self.value(b)).collect()
    }

    /// Every bit pattern of a 16-bit type, or a regular sample of `n` of
    /// them (`n` < 65536), covering every exponent.
    fn patterns(self, n: usize) -> Vec<f64> {
        let step = (65536 / n.max(1)).max(1);
        (0..65536u64).step_by(step).map(|b| self.value(b)).collect()
    }
}

/// The reference for `x`: C99's special values (from the host's libm) for
/// non-finite arguments, NaN and infinite results (domain errors, poles and
/// overflow) and exact zeros (with their sign), the high-precision value
/// otherwise (including results that underflow in f64).
pub fn want(x: f64, libm: fn(f64) -> f64, precise: impl Fn(f64) -> Want) -> Want {
    let s = libm(x);
    if !x.is_finite() || s.is_nan() || s.is_infinite() {
        return Want::Exact(s);
    }
    match precise(x) {
        Want::Value(v) if v.is_zero() => Want::Exact(s),
        w => w,
    }
}

fn big(b: Big) -> Want {
    Want::Value(b)
}

fn dd(d: Dd) -> Want {
    Want::value(d)
}

/// The value of a reference, as an f64 (for fast-mode error bounds).
pub fn approx(w: &Want) -> f64 {
    match w {
        Want::Exact(v) => *v,
        Want::Value(b) => b.to_f64(),
    }
}

/// How far a fast-mode Float32 result may be from the true value `v` of the
/// function at `x` (and `y`), or `None` where no bound is documented.
pub type FastBound = fn(x: f64, y: f64, v: f64) -> Option<f64>;

/// One unit in the last place of `v` as an f32 (subnormal ulps below 2^-126).
pub fn ulp32(v: f64) -> f64 {
    let e = (v.abs().log2().floor() as i32).max(-126);
    2f64.powi(e - 23)
}

/// Flushed-to-zero results: anything up to the smallest normal f32.
const MIN_NORMAL: f64 = f32::MIN_POSITIVE as f64;

fn ulps_or_flushed(v: f64, ulps: f64) -> f64 {
    if v.abs() < MIN_NORMAL { MIN_NORMAL } else { ulps * ulp32(v) }
}

/// How far `v` is from its correctly rounded f32 value (at most half an
/// ulp). The PTX ISA measures `ex2.approx`'s 2 ulp from that value; the
/// bounds of `rcp.approx` and `div.approx` are read the same way, which can
/// only be more lenient than measuring from `v`.
fn rn_gap(v: f64) -> f64 {
    let r = v as f32 as f64;
    if r.is_finite() { (r - v).abs() } else { 0.5 * ulp32(v) }
}

/// `lg2.approx`'s error (PTX ISA): 2^-22 absolute in (0.5, 2), relative
/// elsewhere.
fn lg2_error(x: f64) -> f64 {
    if x > 0.5 && x < 2.0 { 2f64.powi(-22) } else { 2f64.powi(-22) * x.log2().abs() }
}

/// The fast-mode bounds that `cute::math`'s documentation states.
pub mod fast {
    use super::*;

    pub fn exp2(x: f64, _: f64, v: f64) -> Option<f64> {
        Some(ulps_or_flushed(v, 2.0).max(if x.abs() < MIN_NORMAL { ulp32(1.0) } else { 0.0 }) + rn_gap(v))
    }

    pub fn exp(x: f64, _: f64, v: f64) -> Option<f64> {
        Some(ulps_or_flushed(v, 2.0 + 1.5 * x.abs()) + rn_gap(v))
    }

    fn positive_normal(x: f64) -> bool {
        x >= MIN_NORMAL && x.is_finite()
    }

    pub fn log2(x: f64, _: f64, _v: f64) -> Option<f64> {
        positive_normal(x).then(|| lg2_error(x) + 0.5 * ulp32(x.log2()))
    }

    pub fn log(x: f64, _: f64, v: f64) -> Option<f64> {
        if !positive_normal(x) {
            return None;
        }
        Some(if x > 0.5 && x < 2.0 { 2f64.powi(-22) } else { 2f64.powf(-21.6) * v.abs() })
    }

    pub fn log10(x: f64, _: f64, v: f64) -> Option<f64> {
        if !positive_normal(x) {
            return None;
        }
        Some(if x > 0.5 && x < 2.0 { 2f64.powi(-23) } else { 2f64.powf(-21.6) * v.abs() })
    }

    /// `sin.approx` and `cos.approx` (PTX ISA): absolute 2^-20.5 (sin) or
    /// 2^-20.4 (cos) for |x| <= 2 pi, 2^-14.7 for |x| <= 100 pi.
    fn sincos_error(x: f64, near: f64) -> Option<f64> {
        let a = x.abs();
        if a <= 2.0 * std::f64::consts::PI {
            Some(2f64.powf(near))
        } else if a <= 100.0 * std::f64::consts::PI {
            Some(2f64.powf(-14.7))
        } else {
            None
        }
    }

    pub fn sin(x: f64, _: f64, v: f64) -> Option<f64> {
        // .ftz flushes subnormal results (sin(x) = x for tiny x).
        sincos_error(x, -20.5).map(|e| if v.abs() < MIN_NORMAL { MIN_NORMAL } else { e })
    }

    pub fn cos(x: f64, _: f64, _v: f64) -> Option<f64> {
        sincos_error(x, -20.4)
    }

    /// `div.approx(sin.approx(x), cos.approx(x))`: the errors of sine and
    /// cosine over |cos x|, plus the division's 2 ulp.
    pub fn tan(x: f64, _: f64, v: f64) -> Option<f64> {
        if x.abs() > 2.0 * std::f64::consts::PI {
            return None;
        }
        let c = x.cos().abs();
        if c < 2f64.powi(-14) {
            return None;
        }
        Some((2f64.powf(-20.5) + 2f64.powf(-20.4) * v.abs()) / c * 1.01 + ulps_or_flushed(v, 2.0))
    }

    /// `tanh.approx.f32` (PTX ISA): relative 2^-11.
    pub fn tanh(_: f64, _: f64, v: f64) -> Option<f64> {
        Some(2f64.powi(-11) * v.abs() + 2f64.powi(-149))
    }

    pub fn sqrt(x: f64, _: f64, v: f64) -> Option<f64> {
        (x >= MIN_NORMAL).then(|| 2f64.powi(-23) * v.abs())
    }

    pub fn rsqrt(x: f64, _: f64, v: f64) -> Option<f64> {
        (x >= MIN_NORMAL && x.is_finite()).then(|| 2f64.powf(-22.9) * v.abs())
    }

    /// `rcp.approx.ftz.f32` (PTX ISA): 1 ulp.
    pub fn rcp(x: f64, _: f64, v: f64) -> Option<f64> {
        (x.abs() >= MIN_NORMAL && x.abs() <= 2f64.powi(126)).then(|| ulp32(v) + rn_gap(v))
    }

    /// `div.approx.ftz.f32` (PTX ISA): 2 ulp for |y| in [2^-126, 2^126].
    pub fn div(x: f64, y: f64, v: f64) -> Option<f64> {
        let ok = y.abs() >= MIN_NORMAL && y.abs() <= 2f64.powi(126) && x.is_finite() && x.abs() >= MIN_NORMAL;
        ok.then(|| ulps_or_flushed(v, 2.0) + rn_gap(v))
    }

    /// `ex2.approx(y * lg2.approx(x))` for x > 0: the exponent is off by
    /// lg2's error times |y| plus the rounding of the product, `E` in all,
    /// so the result is within a factor `2^E` of the true value, and ex2
    /// adds its 2 ulp of the correctly rounded result (2^-21 relative
    /// covers both). Results below twice the smallest normal may flush.
    pub fn pow(x: f64, y: f64, v: f64) -> Option<f64> {
        if !(x >= MIN_NORMAL && x.is_finite() && y.is_finite()) {
            return None;
        }
        let t = x.log2();
        let exponent_error = y.abs() * lg2_error(x) + (y * t).abs() * 2f64.powi(-24);
        let relative = 2f64.powf(exponent_error) * (1.0 + 2f64.powi(-21)) - 1.0;
        Some((relative * v.abs()).max(2.0 * MIN_NORMAL))
    }
}

/// One ulp of `v` in `ty`.
pub fn ulp_of(v: f64, ty: Ty) -> f64 {
    let f = ty.format();
    let e = (v.abs().log2().floor() as i64).max(f.emin);
    2f64.powi((e - (f.p - 1)) as i32)
}

/// The largest finite value of `ty`.
pub fn max_finite(ty: Ty) -> f64 {
    let f = ty.format();
    (2.0 - 2f64.powi(1 - f.p as i32)) * 2f64.powi(f.emax as i32)
}

/// Whether a fast-mode result is within `bound` of the reference `w` (plus
/// the final rounding of a 16-bit type), counting results that overflow
/// to infinity where the true value is within the bound of overflowing;
/// `None` outside the domain the bound documents.
pub fn fast_within(ty: Ty, x: f64, y: f64, got: f64, w: &Want, bound: FastBound) -> Option<bool> {
    let v = approx(w);
    let allowed = bound(x, y, v)?;
    if let Want::Exact(e) = *w
        && (e.is_nan() || e.is_infinite())
    {
        let e = round_to_format(e, ty.format());
        return Some(got.is_nan() == e.is_nan() && (e.is_nan() || got == e));
    }
    let allowed = allowed + if ty == Ty::F32 { 0.0 } else { 0.5 * ulp_of(v, ty) };
    if got.is_infinite() {
        return Some(got.signum() == v.signum() && v.abs() + allowed >= max_finite(ty));
    }
    Some((got - v).abs() <= allowed)
}

/// Inputs that `.ftz` flushes to zero: Float32 subnormals.
pub fn flushed(x: f64) -> Option<f64> {
    (x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64).then(|| 0.0f64.copysign(x))
}

pub struct Unary {
    pub name: &'static str,
    pub reference: fn(f64) -> Want,
    /// Uniform ranges for f32 and f64 inputs (besides random bit patterns).
    pub ranges32: &'static [(f64, f64)],
    pub ranges64: &'static [(f64, f64)],
    /// Known hard or boundary arguments.
    pub hard: &'static [f64],
    /// The largest error allowed in precise mode, in ulps (Float16 and
    /// BFloat16, computed in Float32 and rounded, allow 0.501).
    pub bound32: f64,
    pub bound64: f64,
    /// The fast-mode bound, for functions fastmath changes.
    pub fast: Option<FastBound>,
}

/// The precise-mode bound of a function computed in Float32 and rounded to
/// a 16-bit type: half an ulp, plus the Float32 error (a few 2^-13 ulp).
fn bound16(bound32: f64) -> f64 {
    if bound32 == 0.0 { 0.0 } else { 0.501 }
}

impl Unary {
    pub fn bound(&self, ty: Ty) -> f64 {
        match ty {
            Ty::F32 => self.bound32,
            Ty::F64 => self.bound64,
            _ => bound16(self.bound32),
        }
    }
}

/// `name` applied to `x`.
pub fn apply<T: Float>(name: &str, x: Value<T>, fastmath: bool) -> Value<T> {
    match name {
        "exp" => math::exp(x, fastmath),
        "exp2" => math::exp2(x, fastmath),
        "expm1" => math::expm1(x, fastmath),
        "log" => math::log(x, fastmath),
        "log2" => math::log2(x, fastmath),
        "log10" => math::log10(x, fastmath),
        "log1p" => math::log1p(x, fastmath),
        "sin" => math::sin(x, fastmath),
        "cos" => math::cos(x, fastmath),
        "tan" => math::tan(x, fastmath),
        "asin" => math::asin(x, fastmath),
        "acos" => math::acos(x, fastmath),
        "atan" => math::atan(x, fastmath),
        "sinh" => math::sinh(x, fastmath),
        "cosh" => math::cosh(x, fastmath),
        "tanh" => math::tanh(x, fastmath),
        "asinh" => math::asinh(x, fastmath),
        "acosh" => math::acosh(x, fastmath),
        "atanh" => math::atanh(x, fastmath),
        "erf" => math::erf(x, fastmath),
        "erfc" => math::erfc(x, fastmath),
        "cbrt" => math::cbrt(x, fastmath),
        "sqrt" => math::sqrt(x, fastmath),
        "rsqrt" => math::rsqrt(x, fastmath),
        "rcp" => math::rcp(x, fastmath),
        "round" => math::round(x, fastmath),
        "sincos.sin" => math::sincos(x, fastmath).0,
        "sincos.cos" => math::sincos(x, fastmath).1,
        other => panic!("no unary function {other}"),
    }
}

/// `name` applied to `(x, y)`.
pub fn apply2<T: Float>(name: &str, x: Value<T>, y: Value<T>, fastmath: bool) -> Value<T> {
    match name {
        "atan2" => math::atan2(x, y, fastmath),
        "pow" => math::pow(x, y, fastmath),
        "rem" => math::rem(x, y, fastmath),
        "div" => math::div(x, y, fastmath),
        "copysign" => math::copysign(x, y, fastmath),
        other => panic!("no binary function {other}"),
    }
}

const BIG_TRIG: &[f64] = &[
    1e22,
    1e300,
    1.7976931348623157e308,
    5.319372648326541e255, // 6381956970095103 * 2^797, the closest double to a multiple of pi/2
    std::f64::consts::FRAC_PI_2,
    std::f64::consts::PI,
    4.71238898038469,
    std::f64::consts::TAU,
    1647099.3291652855,
    1647099.3291652858,
    102.10176,
    1e8,
    3.4e38,
    std::f64::consts::FRAC_PI_4,
    std::f32::consts::FRAC_PI_4 as f64,
    2.356194490192345,
    13176794.0,
    1.0e-300,
    2.9e-8,
    // f32 arguments close to multiples of pi/2, and the boundaries of the
    // medium and Payne-Hanek reductions.
    16367173.0,
    105414350.0,
    201.06194,
    201.06195,
    7.0,
    22.0,
];

const EXP_HARD: &[f64] = &[
    88.72283,
    88.722839,
    88.7228394,
    -87.33654,
    -103.97208,
    -103.9721,
    -104.0,
    709.782712893384,
    709.7827128933841,
    -708.3964185322641,
    -745.1332191019411,
    -745.1332191019412,
    -744.44,
    1e-30,
    -1e-30,
    0.34657359,
    1.0397208,
    0.3465735902799726,
    1.0397207708399179,
];

const LOG_HARD: &[f64] = &[
    1.0000001,
    0.99999994,
    1.0000000000000002,
    0.9999999999999999,
    std::f32::consts::SQRT_2 as f64,
    std::f32::consts::FRAC_1_SQRT_2 as f64,
    std::f64::consts::SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
    1e-45,
    1.4e-45,
    5e-324,
    1e-310,
    1.17549435e-38,
    2.2250738585072014e-308,
    3.4028235e38,
    1.7976931348623157e308,
];

pub fn unary_cases() -> Vec<Unary> {
    vec![
        Unary {
            name: "exp",
            reference: |x| want(x, f64::exp, |x| big(r::exp(Dd::new(x)))),
            ranges32: &[(-104.0, 89.0), (-1.0, 1.0)],
            ranges64: &[(-746.0, 710.0), (-1.0, 1.0)],
            hard: EXP_HARD,
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::exp),
        },
        Unary {
            name: "exp2",
            reference: |x| want(x, f64::exp2, |x| big(r::exp2(x))),
            ranges32: &[(-151.0, 129.0), (-1.0, 1.0)],
            ranges64: &[(-1076.0, 1025.0), (-1.0, 1.0)],
            hard: &[
                128.0,
                127.99999,
                -126.0,
                -149.0,
                -149.5,
                -150.0,
                -150.00002,
                1023.9999999999999,
                -1074.0,
                -1075.0,
                -1074.5,
                0.5,
                -0.5,
            ],
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::exp2),
        },
        Unary {
            name: "expm1",
            reference: |x| want(x, f64::exp_m1, |x| big(r::expm1(x))),
            ranges32: &[(-20.0, 89.0), (-1.0, 1.0), (-1e-3, 1e-3)],
            ranges64: &[(-40.0, 710.0), (-1.0, 1.0), (-1e-3, 1e-3)],
            hard: EXP_HARD,
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "log",
            reference: |x| want(x, f64::ln, |x| dd(r::ln(x))),
            ranges32: &[(0.0, 10.0), (0.5, 2.0)],
            ranges64: &[(0.0, 10.0), (0.5, 2.0)],
            hard: LOG_HARD,
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::log),
        },
        Unary {
            name: "log2",
            reference: |x| want(x, f64::log2, |x| dd(r::log2(x))),
            ranges32: &[(0.0, 10.0), (0.5, 2.0)],
            ranges64: &[(0.0, 10.0), (0.5, 2.0)],
            hard: LOG_HARD,
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::log2),
        },
        Unary {
            name: "log10",
            reference: |x| want(x, f64::log10, |x| dd(r::log10(x))),
            ranges32: &[(0.0, 10.0), (0.5, 2.0)],
            ranges64: &[(0.0, 10.0), (0.5, 2.0)],
            hard: LOG_HARD,
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::log10),
        },
        Unary {
            name: "log1p",
            reference: |x| want(x, f64::ln_1p, |x| dd(r::log1p(x))),
            ranges32: &[(-1.0, 4.0), (-1e-3, 1e-3), (-0.3, 0.42)],
            ranges64: &[(-1.0, 4.0), (-1e-3, 1e-3), (-0.3, 0.42)],
            hard: &[
                -0.99999994,
                -0.9999999999999999,
                1e-30,
                -1e-30,
                0.41421354,
                -0.29289323,
                3e38,
                1.6e308,
                2e-8,
                9e15,
            ],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "sin",
            reference: |x| want(x, f64::sin, |x| dd(r::sin(x))),
            ranges32: &[(-10.0, 10.0), (-210.0, 210.0), (-1e5, 1e5)],
            ranges64: &[(-10.0, 10.0), (-2e6, 2e6), (-1e9, 1e9)],
            hard: BIG_TRIG,
            bound32: 1.5,
            bound64: 1.0,
            fast: Some(fast::sin),
        },
        Unary {
            name: "cos",
            reference: |x| want(x, f64::cos, |x| dd(r::cos(x))),
            ranges32: &[(-10.0, 10.0), (-210.0, 210.0), (-1e5, 1e5)],
            ranges64: &[(-10.0, 10.0), (-2e6, 2e6), (-1e9, 1e9)],
            hard: BIG_TRIG,
            bound32: 1.5,
            bound64: 1.0,
            fast: Some(fast::cos),
        },
        Unary {
            name: "tan",
            reference: |x| want(x, f64::tan, |x| dd(r::tan(x))),
            ranges32: &[(-10.0, 10.0), (-210.0, 210.0), (-1e5, 1e5)],
            ranges64: &[(-10.0, 10.0), (-2e6, 2e6), (-1e9, 1e9)],
            hard: BIG_TRIG,
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::tan),
        },
        Unary {
            name: "asin",
            reference: |x| want(x, f64::asin, |x| dd(r::asin(x))),
            ranges32: &[(-1.0, 1.0), (0.97, 1.0)],
            ranges64: &[(-1.0, 1.0), (0.97, 1.0)],
            hard: &[0.5, -0.5, 0.975, 0.97499996, 0.9999999, 0.9999999999999999, 1e-30, 1.0000001],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "acos",
            reference: |x| want(x, f64::acos, |x| dd(r::acos(x))),
            ranges32: &[(-1.0, 1.0), (0.97, 1.0), (-1.0, -0.97)],
            ranges64: &[(-1.0, 1.0), (0.97, 1.0), (-1.0, -0.97)],
            hard: &[0.5, -0.5, 0.9999999, -0.9999999, 0.9999999999999999, -0.9999999999999999, 1e-30, 1.0000001],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "atan",
            reference: |x| want(x, f64::atan, |x| dd(r::atan(x))),
            ranges32: &[(-10.0, 10.0), (-3.0, 3.0)],
            ranges64: &[(-10.0, 10.0), (-3.0, 3.0)],
            hard: &[0.4375, 0.6875, 1.1875, 2.4375, 1e30, 6.7e7, 7.4e19, 1e-30],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "sinh",
            reference: |x| want(x, f64::sinh, |x| big(r::sinh(x))),
            ranges32: &[(-90.0, 90.0), (-2.0, 2.0)],
            ranges64: &[(-711.0, 711.0), (-2.0, 2.0)],
            hard: &[9.0, 22.0, 88.72168, 89.415985, 89.41599, 709.78, 710.4758600739439, 710.475860073944, 1e-30],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "cosh",
            reference: |x| want(x, f64::cosh, |x| big(r::cosh(x))),
            ranges32: &[(-90.0, 90.0), (-2.0, 2.0)],
            ranges64: &[(-711.0, 711.0), (-2.0, 2.0)],
            hard: &[0.34657359, 9.0, 22.0, 89.415985, 89.41599, 710.4758600739439, 710.475860073944, 1e-30],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "tanh",
            reference: |x| want(x, f64::tanh, |x| big(r::tanh(x))),
            ranges32: &[(-10.0, 10.0), (-1.0, 1.0)],
            ranges64: &[(-25.0, 25.0), (-1.0, 1.0)],
            hard: &[0.55, 1.0, 0.99999994, 9.0, 22.0, 1e-30, 3.7252903e-9],
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::tanh),
        },
        Unary {
            name: "asinh",
            reference: |x| want(x, f64::asinh, |x| dd(r::asinh(x))),
            ranges32: &[(-10.0, 10.0), (-2.0, 2.0)],
            ranges64: &[(-10.0, 10.0), (-2.0, 2.0)],
            hard: &[2.0, 2.0000002, 268435456.0, 3e38, 1.7e308, 1e-30],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "acosh",
            reference: |x| want(x, f64::acosh, |x| dd(r::acosh(x))),
            ranges32: &[(1.0, 10.0), (1.0, 2.0)],
            ranges64: &[(1.0, 10.0), (1.0, 2.0)],
            hard: &[1.0000001, 1.0000000000000002, 2.0, 2.0000002, 268435456.0, 3e38, 1.7e308, 0.99999994],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "atanh",
            reference: |x| want(x, f64::atanh, |x| dd(r::atanh(x))),
            ranges32: &[(-1.0, 1.0), (0.99, 1.0)],
            ranges64: &[(-1.0, 1.0), (0.99, 1.0)],
            hard: &[0.5, 0.49999997, 0.99999994, 0.9999999999999999, 1e-30, -0.9991649877288362],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "erf",
            reference: |x| want(x, libm_erf, |x| big(r::erf(x))),
            ranges32: &[(-5.0, 5.0), (-1.5, 1.5)],
            ranges64: &[(-7.0, 7.0), (-1.5, 1.5)],
            hard: &[0.84375, 1.25, 2.857143, 4.0, 6.0, 1e-30, 1e-40, 1e-310, 2e-38],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "erfc",
            reference: |x| want(x, libm_erfc, |x| big(r::erfc(x))),
            ranges32: &[(-5.0, 11.0), (-1.5, 1.5)],
            ranges64: &[(-7.0, 28.0), (-1.5, 1.5)],
            hard: &[0.25, 0.84375, 1.25, 2.857143, 5.0, 6.0, 10.0, 10.05, 26.5, 27.2, 27.25, -5.0, 1e-30],
            bound32: 2.5,
            bound64: 2.5,
            fast: None,
        },
        Unary {
            name: "cbrt",
            reference: |x| want(x, f64::cbrt, |x| dd(r::cbrt(x))),
            ranges32: &[(-10.0, 10.0)],
            ranges64: &[(-10.0, 10.0)],
            hard: &[8.0, 27.0, -64.0, 1e-45, 5e-324, 1e-310, 3.4e38, 1.7e308],
            bound32: 1.0,
            bound64: 1.0,
            fast: None,
        },
        Unary {
            name: "sqrt",
            reference: |x| want(x, f64::sqrt, |x| big(r::sqrt(x))),
            ranges32: &[(0.0, 10.0)],
            ranges64: &[(0.0, 10.0)],
            hard: &[4.0, 2.0, 1e-45, 5e-324, 1e-310, 3.4e38, 1.7e308],
            bound32: 0.5,
            bound64: 0.5,
            fast: Some(fast::sqrt),
        },
        Unary {
            name: "rsqrt",
            reference: |x| want(x, |x| 1.0 / x.sqrt(), |x| big(r::rsqrt(x))),
            ranges32: &[(0.0, 10.0)],
            ranges64: &[(0.0, 10.0)],
            hard: &[4.0, 0.25, 2.0, 1e-45, 5e-324, 1e-310, 3.4e38, 1.7e308, 1.1754942e-38],
            bound32: 0.502,
            bound64: 0.502,
            fast: Some(fast::rsqrt),
        },
        Unary {
            name: "rcp",
            reference: |x| want(x, |x| 1.0 / x, |x| big(Big::of(1.0).div(Big::of(x)))),
            ranges32: &[(-10.0, 10.0)],
            ranges64: &[(-10.0, 10.0)],
            hard: &[3.0, 1e-45, 5e-324, 1e-310, 3.4e38, 1.7e308, 1.1754942e-38],
            bound32: 0.5,
            bound64: 0.5,
            fast: Some(fast::rcp),
        },
        Unary {
            name: "round",
            reference: |x| Want::Exact(x.round()),
            ranges32: &[(-10.0, 10.0)],
            ranges64: &[(-10.0, 10.0)],
            hard: &[0.5, -0.5, 1.5, 2.5, -2.5, 0.49999997, 0.49999999999999994, 8388607.5, 4503599627370495.5],
            bound32: 0.0,
            bound64: 0.0,
            fast: None,
        },
    ]
}

pub fn unary_case(name: &str) -> Unary {
    unary_cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

pub struct Binary {
    pub name: &'static str,
    pub reference: fn(f64, f64) -> Want,
    /// Argument pairs, rounded to the type.
    pub inputs: fn(usize, Ty) -> Vec<(f64, f64)>,
    pub bound32: f64,
    pub bound64: f64,
    pub fast: Option<FastBound>,
}

impl Binary {
    pub fn bound(&self, ty: Ty) -> f64 {
        match ty {
            Ty::F32 => self.bound32,
            Ty::F64 => self.bound64,
            _ => bound16(self.bound32),
        }
    }
}

pub fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

pub fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * ((rng(state) >> 11) as f64 / (1u64 << 53) as f64)
}

/// A random value of type `ty` with every bit pattern equally likely.
pub fn any(state: &mut u64, ty: Ty) -> f64 {
    let b = rng(state);
    match ty {
        Ty::F64 => f64::from_bits(b),
        Ty::F32 => f32::from_bits((b >> 32) as u32) as f64,
        _ => ty.value(b >> 48),
    }
}

const SPECIALS: [f64; 13] =
    [0.0, -0.0, 1.0, -1.0, 0.5, -2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 3.0, -0.75, 1e-30, 1e-310];

fn pairs(n: usize, ty: Ty, mut make: impl FnMut(&mut u64) -> (f64, f64)) -> Vec<(f64, f64)> {
    let mut state = 0x2545f4914f6cdd1d ^ (n as u64) ^ ((ty as u64) << 7);
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
    for &a in &SPECIALS {
        for &b in &SPECIALS {
            out.push((a, b));
        }
    }
    while out.len() < n {
        out.push(make(&mut state));
    }
    out.truncate(n.max(SPECIALS.len() * SPECIALS.len()));
    out.into_iter().map(|(a, b)| (ty.round(a), ty.round(b))).collect()
}

pub fn pow_want(x: f64, y: f64) -> Want {
    let s = x.powf(y);
    if !x.is_finite() || !y.is_finite() || s.is_nan() || x == 0.0 || x == 1.0 || y == 0.0 {
        return Want::Exact(s);
    }
    let odd = y.fract() == 0.0 && (y.abs() < 2f64.powi(53)) && (y % 2.0) != 0.0;
    if x < 0.0 {
        let m = r::pow_pos(-x, y);
        return Want::Value(if odd { m.neg() } else { m });
    }
    Want::Value(r::pow_pos(x, y))
}

pub fn binary_cases() -> Vec<Binary> {
    vec![
        Binary {
            name: "atan2",
            reference: |y, x| {
                let s = y.atan2(x);
                if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 || s.is_nan() {
                    Want::Exact(s)
                } else {
                    Want::Value(r::atan2(y, x))
                }
            },
            inputs: |n, ty| {
                pairs(n, ty, |s| match rng(s) % 4 {
                    0 => (any(s, ty), any(s, ty)),
                    1 => (uniform(s, -10.0, 10.0), uniform(s, -10.0, 10.0)),
                    2 => (uniform(s, -1.0, 1.0) * 1e-20, uniform(s, -10.0, 10.0)),
                    _ => (uniform(s, -10.0, 10.0), uniform(s, -1.0, 1.0) * 1e20),
                })
            },
            bound32: 1.5,
            bound64: 1.5,
            fast: None,
        },
        Binary {
            name: "pow",
            reference: pow_want,
            inputs: |n, ty| {
                let lim = if ty == Ty::F64 { 1500.0 } else { 200.0 };
                pairs(n, ty, |s| match rng(s) % 8 {
                    0 => (any(s, ty), any(s, ty)),
                    1 => (uniform(s, 0.0, 10.0), uniform(s, -50.0, 50.0)),
                    2 => (uniform(s, 0.5, 2.0), uniform(s, -lim, lim)),
                    3 => (-uniform(s, 0.0, 10.0), (uniform(s, -40.0, 40.0)).round()),
                    4 => (1.0 + uniform(s, -1e-6, 1e-6), uniform(s, -1e9, 1e9)),
                    5 => (uniform(s, 0.0, 3.0), uniform(s, -3.0, 3.0)),
                    6 => (any(s, ty), uniform(s, -4.0, 4.0).round()),
                    // x within a few ulps of 1 and |y| beyond 2^27 (f64:
                    // 2^31), where msun switches to a series for log x.
                    _ => {
                        let (ulp, huge) = if ty == Ty::F64 {
                            (2f64.powi(-30), 2f64.powi(31))
                        } else {
                            (2f64.powi(-24), 2f64.powi(27))
                        };
                        let k = (rng(s) % 40) as f64 - 20.0;
                        (1.0 + k * ulp, uniform(s, -1.0, 1.0).signum() * huge * uniform(s, 1.0, 512.0))
                    }
                })
            },
            bound32: 1.0,
            bound64: 1.0,
            fast: Some(fast::pow),
        },
        Binary {
            name: "rem",
            reference: |x, y| Want::Exact(x % y),
            inputs: |n, ty| {
                pairs(n, ty, |s| match rng(s) % 3 {
                    0 => (any(s, ty), any(s, ty)),
                    1 => (uniform(s, -1e6, 1e6), uniform(s, -10.0, 10.0)),
                    _ => (any(s, ty), uniform(s, -1.0, 1.0) * 1e-30),
                })
            },
            bound32: 0.0,
            bound64: 0.0,
            fast: None,
        },
        Binary {
            name: "copysign",
            reference: |x, y| Want::Exact(x.copysign(y)),
            inputs: |n, ty| pairs(n, ty, |s| (any(s, ty), any(s, ty))),
            bound32: 0.0,
            bound64: 0.0,
            fast: None,
        },
        Binary {
            name: "div",
            reference: |x, y| {
                let s = x / y;
                if !s.is_finite() || s == 0.0 || !x.is_finite() || !y.is_finite() {
                    Want::Exact(s)
                } else {
                    Want::Value(Big::of(x).div(Big::of(y)))
                }
            },
            inputs: |n, ty| {
                pairs(n, ty, |s| match rng(s) % 2 {
                    0 => (any(s, ty), any(s, ty)),
                    _ => (uniform(s, -10.0, 10.0), uniform(s, -10.0, 10.0)),
                })
            },
            bound32: 0.5,
            bound64: 0.5,
            fast: Some(fast::div),
        },
    ]
}

pub fn binary_case(name: &str) -> Binary {
    binary_cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

/// The inputs of a unary case: specials, hard cases with their neighbours,
/// uniform samples of the ranges and random bit patterns, rounded to `ty`.
/// A 16-bit type gets every bit pattern when `n` is at least 65536, and a
/// regular sample of them otherwise.
pub fn unary_inputs(case: &Unary, n: usize, ty: Ty) -> Vec<f64> {
    if matches!(ty, Ty::F16 | Ty::BF16) {
        let mut out = ty.patterns(n);
        out.extend([0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0, -1.0, 0.5, 2.0]);
        out.extend(case.hard.iter().map(|&h| ty.round(h)));
        return out;
    }
    let double = ty == Ty::F64;
    let mut state = 0x9e3779b97f4a7c15 ^ (n as u64).wrapping_mul(31) ^ case.name.len() as u64;
    let mut out: Vec<f64> =
        vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
    let (min_sub, max_sub, min_normal, max) = if double {
        (f64::from_bits(1), f64::from_bits((1 << 52) - 1), f64::MIN_POSITIVE, f64::MAX)
    } else {
        (f32::from_bits(1) as f64, f32::from_bits((1 << 23) - 1) as f64, f32::MIN_POSITIVE as f64, f32::MAX as f64)
    };
    for v in [min_sub, max_sub, min_normal, max] {
        out.push(v);
        out.push(-v);
    }
    for &h in case.hard {
        let h = ty.round(h);
        for k in -2i64..=2 {
            // h and its neighbours.
            let v = if double {
                f64::from_bits((h.to_bits() as i64 + k) as u64)
            } else {
                f32::from_bits(((h as f32).to_bits() as i64 + k) as u32) as f64
            };
            out.push(v);
            out.push(-v);
        }
    }
    let ranges = if double { case.ranges64 } else { case.ranges32 };
    while out.len() < n {
        let v = if out.len().is_multiple_of(4) || ranges.is_empty() {
            any(&mut state, ty)
        } else {
            let (lo, hi) = ranges[out.len() % ranges.len()];
            ty.round(uniform(&mut state, lo, hi))
        };
        out.push(v);
    }
    out
}

/// Threads per block in the math kernels.
pub const BLOCK: usize = 128;

fn kernel_name(name: &str, t: DType, fastmath: bool) -> String {
    format!("{}_{t}{}", name.replace('.', "_"), if fastmath { "_fast" } else { "" })
}

fn unary_generic<T: Float>(name: &'static str, n: usize, fastmath: bool) -> Kernel {
    Kernel::builder(kernel_name(name, T::DTYPE, fastmath)).block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::col_major(n));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || y.store(i, apply(name, x.load(i), fastmath)));
    })
}

/// A kernel applying the unary function `name` to `n` values of `ty`.
pub fn unary_kernel(name: &'static str, ty: Ty, n: usize, fastmath: bool) -> Kernel {
    match ty {
        Ty::F16 => unary_generic::<F16>(name, n, fastmath),
        Ty::BF16 => unary_generic::<BF16>(name, n, fastmath),
        Ty::F32 => unary_generic::<f32>(name, n, fastmath),
        Ty::F64 => unary_generic::<f64>(name, n, fastmath),
    }
}

fn binary_generic<T: Float>(name: &'static str, n: usize, fastmath: bool) -> Kernel {
    Kernel::builder(kernel_name(name, T::DTYPE, fastmath)).block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let b = param_tensor::<T>("b", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::col_major(n));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || y.store(i, apply2(name, a.load(i), b.load(i), fastmath)));
    })
}

/// A kernel applying the binary function `name` to `n` pairs of `ty`.
pub fn binary_kernel(name: &'static str, ty: Ty, n: usize, fastmath: bool) -> Kernel {
    match ty {
        Ty::F16 => binary_generic::<F16>(name, n, fastmath),
        Ty::BF16 => binary_generic::<BF16>(name, n, fastmath),
        Ty::F32 => binary_generic::<f32>(name, n, fastmath),
        Ty::F64 => binary_generic::<f64>(name, n, fastmath),
    }
}

pub fn launch(n: usize) -> LaunchConfig {
    LaunchConfig::new([n.div_ceil(BLOCK) as u32, 1, 1], [BLOCK as u32, 1, 1])
}

/// Arguments for a unary kernel on `xs`.
pub fn unary_args(ty: Ty, xs: &[f64]) -> Vec<KernelArg> {
    vec![ty.buffer(xs), ty.buffer(&vec![0.0; xs.len()])]
}

/// Arguments for a binary kernel on `pairs`.
pub fn binary_args(ty: Ty, pairs: &[(f64, f64)]) -> Vec<KernelArg> {
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
    vec![ty.buffer(&a), ty.buffer(&b), ty.buffer(&vec![0.0; pairs.len()])]
}

/// The largest error and where it is, and how many results exceed `bound`.
#[derive(Debug, Default)]
pub struct Stats {
    pub max: f64,
    pub at: Vec<f64>,
    pub got: f64,
    pub count: usize,
    pub failures: usize,
    pub first_failure: Option<String>,
}

impl Stats {
    pub fn record(&mut self, args: &[f64], got: f64, err: f64, bound: f64) {
        self.count += 1;
        if err > self.max || self.at.is_empty() {
            self.max = self.max.max(err);
            self.at = args.to_vec();
            self.got = got;
        }
        if err > bound {
            self.failures += 1;
            if self.first_failure.is_none() {
                self.first_failure = Some(format!("{args:?} -> {got:e} ({err:.3} ulp)"));
            }
        }
    }

    pub fn summary(&self, what: &str) -> String {
        format!("{what}: max {:.3} ulp at {:?} (got {:e}) over {} inputs", self.max, self.at, self.got, self.count)
    }
}

// Kernels for the functions without a reference sweep.

fn fpowi_generic<T: Float>(n: usize) -> Kernel {
    Kernel::builder("fpowi").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let e = param_tensor::<i32>("e", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::col_major(n));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || y.store(i, math::fpowi(a.load(i), e.load(i), false)));
    })
}

/// `fpowi(a[i], e[i])`.
pub fn fpowi_kernel(ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => fpowi_generic::<F16>(n),
        Ty::BF16 => fpowi_generic::<BF16>(n),
        Ty::F32 => fpowi_generic::<f32>(n),
        Ty::F64 => fpowi_generic::<f64>(n),
    }
}

/// Inputs for the exact functions: specials, subnormals, halves and random
/// patterns.
pub fn exact_inputs(ty: Ty) -> Vec<f64> {
    let mut state = 99u64;
    let f = ty.format();
    let min_normal = 2f64.powi(f.emin as i32);
    let mut xs = vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.5, -0.5, 1.5, -2.5, 2.5, 1e-30];
    xs.extend([min_normal, -min_normal, min_normal / 2.0, -min_normal / 4.0, 2f64.powi(f.p as i32 - 1) + 0.5]);
    while xs.len() < 4096 {
        let v = match rng(&mut state) % 3 {
            0 => any(&mut state, ty),
            1 => (uniform(&mut state, -20.0, 20.0) * 2.0).round() / 2.0,
            _ => uniform(&mut state, -10.0, 10.0),
        };
        xs.push(v);
    }
    xs.into_iter().map(|x| ty.round(x)).collect()
}

fn exact_unary_generic<T: Float>(n: usize) -> Kernel {
    Kernel::builder("exact_unary").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, 7)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let v = x.load(i);
            let results = [
                math::floor(v, false),
                math::ceil(v, false),
                math::trunc(v, false),
                math::roundeven(v, false),
                math::abs(v, false),
                math::absf(v, false),
                math::neg(v, false),
            ];
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// floor, ceil, trunc, roundeven, abs, absf and neg of each value.
pub fn exact_kernel(ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => exact_unary_generic::<F16>(n),
        Ty::BF16 => exact_unary_generic::<BF16>(n),
        Ty::F32 => exact_unary_generic::<f32>(n),
        Ty::F64 => exact_unary_generic::<f64>(n),
    }
}

fn classify_generic<T: Float>(n: usize) -> Kernel {
    Kernel::builder("classify").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let y = param_tensor::<bool>("y", Layout::row_major((n, 4)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let v = x.load(i);
            let results =
                [math::isnan(v, false), math::isinf(v, false), math::isfinite(v, false), math::isnormal(v, false)];
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// isnan, isinf, isfinite and isnormal of each value.
pub fn classify_kernel(ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => classify_generic::<F16>(n),
        Ty::BF16 => classify_generic::<BF16>(n),
        Ty::F32 => classify_generic::<f32>(n),
        Ty::F64 => classify_generic::<f64>(n),
    }
}

/// A kernel computing six operations with explicit instruction options.
fn options_generic<T: Float>(n: usize, opts: MathOptions) -> Kernel {
    Kernel::builder("options").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let b = param_tensor::<T>("b", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, 6)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let (x, z) = (a.load(i), b.load(i));
            let results = [
                math::add_with(x, z, opts),
                math::mul_with(x, z, opts),
                math::fma_with(x, z, x, opts),
                math::div_with(x, z, opts),
                math::sqrt_with(math::abs(x, false), opts),
                math::rcp_with(z, opts),
            ];
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// `add`, `mul`, `fma`, `div`, `sqrt` and `rcp` of each pair with
/// explicit options (Float32 or Float64).
pub fn options_kernel(ty: Ty, n: usize, opts: MathOptions) -> Kernel {
    match ty {
        Ty::F32 => options_generic::<f32>(n, opts),
        Ty::F64 => options_generic::<f64>(n, opts),
        _ => panic!("explicit instruction options need Float32 or Float64"),
    }
}

fn minmax_generic<T: Float>(n: usize) -> Kernel {
    Kernel::builder("minmax").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let b = param_tensor::<T>("b", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, 6)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let (x, z) = (a.load(i), b.load(i));
            let nan = MathOptions { propagate_nan: true, ..MathOptions::default() };
            let zero = Value::constant(T::from_bits(0));
            let results = [
                math::min(x, z),
                math::max(x, z),
                math::min_with(x, z, nan),
                math::max_with(x, z, nan),
                math::clamp(x, z, zero, false),
                math::clamp_with(x, -math::abs(z, false), math::abs(z, false), nan),
            ];
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// min, max, their NaN-propagating forms and two clamps of each pair.
pub fn minmax_kernel(ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => minmax_generic::<F16>(n),
        Ty::BF16 => minmax_generic::<BF16>(n),
        Ty::F32 => minmax_generic::<f32>(n),
        Ty::F64 => minmax_generic::<f64>(n),
    }
}

/// `fpowi` arguments: special bases with special exponents, then random
/// bases (near 1 too) with small, medium and large exponents.
pub fn fpowi_inputs(ty: Ty, n: usize) -> Vec<(f64, i32)> {
    let mut state = 77u64;
    let mut cases: Vec<(f64, i32)> = Vec::new();
    for &b in &[0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-310, -3.0] {
        for &k in &[0, 1, -1, 2, -2, 3, -3, 7, -7, 1000, -1000, 1 << 24, -(1 << 24), i32::MAX, i32::MIN] {
            cases.push((ty.round(b), k));
        }
    }
    while cases.len() < n {
        let k = match rng(&mut state) % 3 {
            0 => (rng(&mut state) % 64) as i32 - 32,
            1 => (rng(&mut state) % 2000) as i32 - 1000,
            _ => (rng(&mut state) % (1 << 25)) as i32 - (1 << 24),
        };
        let b = match rng(&mut state) % 3 {
            0 => uniform(&mut state, -3.0, 3.0),
            1 => 1.0 + uniform(&mut state, -1e-6, 1e-6),
            _ => any(&mut state, ty),
        };
        cases.push((ty.round(b), k));
    }
    cases
}

/// `min`/`max` arguments: every pair of special values, then random pairs.
pub fn minmax_inputs(ty: Ty) -> Vec<(f64, f64)> {
    let specials = [0.0, -0.0, 1.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-40, -2.5];
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for &a in &specials {
        for &b in &specials {
            pairs.push((ty.round(a), ty.round(b)));
        }
    }
    let mut state = 3u64;
    while pairs.len() < 1024 {
        pairs.push((ty.round(uniform(&mut state, -4.0, 4.0)), ty.round(uniform(&mut state, -4.0, 4.0))));
    }
    pairs
}

fn ipowi_generic<T: Integer + Bitwise>(n: usize) -> Kernel {
    Kernel::builder("ipowi").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let e = param_tensor::<T>("e", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::col_major(n));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || y.store(i, math::ipowi(a.load(i), e.load(i))));
    })
}

/// `ipowi(a[i], e[i])`.
pub fn ipowi_kernel<T: Integer + Bitwise>(n: usize) -> Kernel {
    ipowi_generic::<T>(n)
}

/// `ipowi` arguments: small, extreme and negative bases and exponents, then
/// random small ones (base, exponent).
pub fn ipowi_pairs() -> Vec<(i64, i64)> {
    let mut state = 5u64;
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for b in [-3i64, -2, -1, 0, 1, 2, 3, 7, i32::MAX as i64, i32::MIN as i64] {
        for e in [-3i64, -2, -1, 0, 1, 2, 3, 31, 32, 63, 1000, i32::MAX as i64] {
            pairs.push((b, e));
        }
    }
    while pairs.len() < 4096 {
        pairs.push(((rng(&mut state) % 41) as i64 - 20, (rng(&mut state) % 70) as i64 - 3));
    }
    pairs
}

/// [`ipowi_pairs`] as Int32 bases and exponents.
pub fn ipowi_inputs() -> (Vec<i32>, Vec<i32>) {
    ipowi_pairs().iter().map(|&(b, e)| (b as i32, e as i32)).unzip()
}

/// Argument pairs for the explicit instruction options: every pair of
/// special values, random bit patterns, moderate values, positive values
/// (for the logarithm and roots), subnormals and the edges of
/// `div.approx`'s range.
pub fn option_inputs(ty: Ty) -> Vec<(f64, f64)> {
    let mut state = 1234u64;
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for &a in &SPECIALS {
        for &b in &SPECIALS {
            pairs.push((a, b));
        }
    }
    let edge = 2f64.powi(126);
    while pairs.len() < 8192 {
        let p = match rng(&mut state) % 6 {
            0 => (any(&mut state, ty), any(&mut state, ty)),
            1 => (uniform(&mut state, -10.0, 10.0), uniform(&mut state, -10.0, 10.0)),
            2 => (uniform(&mut state, 0.0, 2.0), uniform(&mut state, 0.5, 2.0)),
            3 => (uniform(&mut state, -1.0, 1.0) * 1e-38, uniform(&mut state, 0.5, 2.0)),
            4 => (uniform(&mut state, -300.0, 300.0), uniform(&mut state, -1.0, 1.0) * edge * 3.0),
            _ => (any(&mut state, ty).abs(), uniform(&mut state, -100.0, 100.0)),
        };
        pairs.push(p);
    }
    pairs.into_iter().map(|(a, b)| (ty.round(a), ty.round(b))).collect()
}

/// An explicit hardware approximation and its PTX ISA error bound: whether
/// a result is within it, or `None` where the ISA gives no bound.
pub struct ApproxOp {
    pub name: &'static str,
    pub check: fn(x: f64, y: f64, got: f64, ftz: bool) -> Option<bool>,
}

fn flush32(v: f64, ftz: bool) -> f64 {
    if ftz && v != 0.0 && v.abs() < MIN_NORMAL { 0.0f64.copysign(v) } else { v }
}

fn flush64(v: f64, ftz: bool) -> f64 {
    if ftz && v != 0.0 && v.abs() < f64::MIN_POSITIVE { 0.0f64.copysign(v) } else { v }
}

/// Within 2^-19 of `want`, relative (results that flush to zero count).
fn rough(got: f64, want: f64) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() || want == 0.0 {
        return got == want;
    }
    if want.abs() < f64::MIN_POSITIVE {
        return got.abs() <= f64::MIN_POSITIVE;
    }
    ((got - want) / want).abs() <= 2f64.powi(-19)
}

/// One ulp of `v` as an f32, subnormal ulps included.
fn ulp32_any(v: f64) -> f64 {
    let e = (v.abs().log2().floor() as i32).clamp(-126, 127);
    2f64.powi(e - 23)
}

/// `got` against `want` (both after the special-value table): NaN for NaN,
/// infinities and zeros exactly (with flushing), finite values within
/// `allowed`.
fn close(got: f64, want: f64, allowed: f64, ftz: bool) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() {
        return got == want;
    }
    if ftz && want.abs() < MIN_NORMAL && got == 0.0 {
        return true;
    }
    (got - want).abs() <= allowed
}

/// The explicit approximations [`approx_kernel`] computes, in order.
pub fn approx_ops(ty: Ty, ftz: bool) -> Vec<ApproxOp> {
    if ty == Ty::F64 {
        // rcp.approx.ftz.f64 and rsqrt.approx.ftz.f64 use the high word of
        // their argument and return 20 significand bits; rsqrt.approx.f64 is
        // emulated in software. The PTX ISA gives no bound for them: this
        // checks 2^-19 relative, a few units in the 20th bit.
        let mut ops = vec![ApproxOp {
            name: "rsqrt.approx",
            check: |x, _, got, ftz| Some(rough(got, 1.0 / flush64(x, ftz).sqrt())),
        }];
        if ftz {
            ops.push(ApproxOp {
                name: "rcp.approx.ftz",
                check: |x, _, got, _| Some(rough(got, 1.0 / flush64(x, true))),
            });
        }
        return ops;
    }
    vec![
        // ex2.approx: 2 ulp of the correctly rounded result.
        ApproxOp {
            name: "ex2.approx",
            check: |x, _, got, ftz| {
                let want = flush32(flush32(x, ftz).exp2(), ftz);
                let rounded = want as f32 as f64;
                Some(close(got, want, (rounded - want).abs() + 2.0 * ulp32_any(want), ftz) || got == rounded)
            },
        },
        // lg2.approx: 2^-22 absolute in (0.5, 2), relative elsewhere.
        ApproxOp {
            name: "lg2.approx",
            check: |x, _, got, ftz| {
                let x = flush32(x, ftz);
                let want = x.log2();
                let allowed = if x > 0.5 && x < 2.0 { 2f64.powi(-22) } else { 2f64.powi(-22) * want.abs() };
                Some(close(got, want, allowed + 0.5 * ulp32_any(want), ftz))
            },
        },
        // sin.approx and cos.approx: absolute 2^-20.5 and 2^-20.4 within
        // 2 pi, 2^-14.7 within 100 pi, no bound beyond.
        ApproxOp {
            name: "sin.approx",
            check: |x, _, got, ftz| {
                let x = flush32(x, ftz);
                let a = x.abs();
                let allowed = if a <= 2.0 * std::f64::consts::PI {
                    2f64.powf(-20.5)
                } else if a <= 100.0 * std::f64::consts::PI {
                    2f64.powf(-14.7)
                } else if x.is_finite() {
                    return None;
                } else {
                    0.0
                };
                Some(close(got, flush32(x.sin(), ftz), allowed, ftz))
            },
        },
        ApproxOp {
            name: "cos.approx",
            check: |x, _, got, ftz| {
                let x = flush32(x, ftz);
                let a = x.abs();
                let allowed = if a <= 2.0 * std::f64::consts::PI {
                    2f64.powf(-20.4)
                } else if a <= 100.0 * std::f64::consts::PI {
                    2f64.powf(-14.7)
                } else if x.is_finite() {
                    return None;
                } else {
                    0.0
                };
                Some(close(got, x.cos(), allowed, ftz))
            },
        },
        // tanh.approx.f32 (no ftz form): 2^-11 relative.
        ApproxOp {
            name: "tanh.approx",
            check: |x, _, got, _| {
                let want = x.tanh();
                Some(close(got, want, 2f64.powi(-11) * want.abs() + 2f64.powi(-149), false))
            },
        },
        // rsqrt.approx: 2^-22.9 relative over the positive finite range.
        ApproxOp {
            name: "rsqrt.approx",
            check: |x, _, got, ftz| {
                let want = flush32(1.0 / flush32(x, ftz).sqrt(), ftz);
                Some(close(got, want, 2f64.powf(-22.9) * want.abs(), ftz))
            },
        },
        // sqrt.approx: 2^-23 relative.
        ApproxOp {
            name: "sqrt.approx",
            check: |x, _, got, ftz| {
                let want = flush32(flush32(x, ftz).sqrt(), ftz);
                Some(close(got, want, 2f64.powi(-23) * want.abs() + 2f64.powi(-149), ftz))
            },
        },
        // rcp.approx: 1 ulp over the full range.
        ApproxOp {
            name: "rcp.approx",
            check: |x, _, got, ftz| {
                let want = flush32(1.0 / flush32(x, ftz), ftz);
                let rounded = want as f32 as f64;
                Some(close(got, want, (rounded - want).abs() + ulp32_any(want), ftz) || got == rounded)
            },
        },
        // div.approx: 2 ulp for |y| in [2^-126, 2^126]; zero (NaN for an
        // infinite x) beyond.
        ApproxOp {
            name: "div.approx",
            check: |x, y, got, ftz| {
                let (x, y) = (flush32(x, ftz), flush32(y, ftz));
                let edge = 2f64.powi(126);
                if y.is_finite() && y.abs() > edge {
                    return Some(if x.is_infinite() || x.is_nan() { got.is_nan() } else { got == 0.0 });
                }
                if y.abs() < MIN_NORMAL && y != 0.0 {
                    return None;
                }
                let want = flush32(x / y, ftz);
                let rounded = want as f32 as f64;
                Some(close(got, want, (rounded - want).abs() + 2.0 * ulp32_any(want), ftz) || got == rounded)
            },
        },
        // div.full: 2 ulp over the full range.
        ApproxOp {
            name: "div.full",
            check: |x, y, got, ftz| {
                let (x, y) = (flush32(x, ftz), flush32(y, ftz));
                let want = flush32(x / y, ftz);
                let rounded = want as f32 as f64;
                Some(close(got, want, (rounded - want).abs() + 2.0 * ulp32_any(want), ftz) || got == rounded)
            },
        },
    ]
}

fn approx_generic<T: Float>(n: usize, ftz: bool, count: usize) -> Kernel {
    Kernel::builder("approx").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<T>("a", Layout::col_major(n));
        let b = param_tensor::<T>("b", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, count)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let (x, z) = (a.load(i), b.load(i));
            let opts = MathOptions { approx: true, ftz, ..MathOptions::default() };
            let results: Vec<Value<T>> = if T::DTYPE == DType::F64 {
                let mut r = vec![math::rsqrt_with(x, opts)];
                if ftz {
                    r.push(math::rcp_with(x, opts));
                }
                r
            } else {
                vec![
                    math::exp2_with(x, opts),
                    math::log2_with(x, opts),
                    math::sin_with(x, opts),
                    math::cos_with(x, opts),
                    math::tanh_with(x, MathOptions::APPROX),
                    math::rsqrt_with(x, opts),
                    math::sqrt_with(x, opts),
                    math::rcp_with(x, opts),
                    math::div_with(x, z, opts),
                    math::div_with(x, z, MathOptions { approx: false, full: true, ..opts }),
                ]
            };
            assert_eq!(results.len(), count);
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// The explicit hardware approximations of [`approx_ops`] (`approx=True`,
/// `full=True`, with or without `ftz=True`) of each pair.
pub fn approx_kernel(ty: Ty, n: usize, ftz: bool) -> Kernel {
    let count = approx_ops(ty, ftz).len();
    match ty {
        Ty::F32 => approx_generic::<f32>(n, ftz, count),
        Ty::F64 => approx_generic::<f64>(n, ftz, count),
        _ => panic!("explicit approximations need Float32 or Float64"),
    }
}

// The sign-bit operations (abs, neg, copysign), the ftz forms of abs, min,
// max and clamp, and absi, on raw bit patterns.

impl Ty {
    /// The width in bits.
    pub fn width(self) -> u32 {
        match self {
            Ty::F16 | Ty::BF16 => 16,
            Ty::F32 => 32,
            Ty::F64 => 64,
        }
    }

    /// The sign bit.
    pub fn sign(self) -> u64 {
        1u64 << (self.width() - 1)
    }

    /// A buffer holding these bit patterns exactly (signaling NaNs too).
    pub fn buffer_bits(self, bits: &[u64]) -> KernelArg {
        match self {
            Ty::F16 => KernelArg::buffer(&bits.iter().map(|&b| F16(b as u16)).collect::<Vec<_>>()),
            Ty::BF16 => KernelArg::buffer(&bits.iter().map(|&b| BF16(b as u16)).collect::<Vec<_>>()),
            Ty::F32 => KernelArg::buffer(&bits.iter().map(|&b| f32::from_bits(b as u32)).collect::<Vec<_>>()),
            Ty::F64 => KernelArg::buffer(&bits.iter().map(|&b| f64::from_bits(b)).collect::<Vec<_>>()),
        }
    }

    /// A zeroed buffer of the unsigned integer type of the same width.
    pub fn int_buffer(self, n: usize) -> KernelArg {
        match self.width() {
            16 => KernelArg::buffer(&vec![0u16; n]),
            32 => KernelArg::buffer(&vec![0u32; n]),
            _ => KernelArg::buffer(&vec![0u64; n]),
        }
    }

    /// The values of a buffer of the unsigned integer type of this width.
    pub fn int_bits(self, arg: &KernelArg) -> Vec<u64> {
        match self.width() {
            16 => arg.to_vec::<u16>().into_iter().map(u64::from).collect(),
            32 => arg.to_vec::<u32>().into_iter().map(u64::from).collect(),
            _ => arg.to_vec::<u64>(),
        }
    }

    /// Whether `bits` is a NaN of this type.
    pub fn is_nan_bits(self, bits: u64) -> bool {
        self.value(bits).is_nan()
    }

    /// A subnormal pattern flushed to the zero of its sign (`.ftz`).
    pub fn flush_bits(self, bits: u64) -> u64 {
        let v = self.value(bits);
        if v != 0.0 && v.abs() < 2f64.powi(self.format().emin as i32) { bits & self.sign() } else { bits }
    }
}

/// Bit patterns for the sign-bit operations: signed zeros, infinities,
/// subnormals, normal numbers, the largest finite values, and quiet and
/// signaling NaNs of both signs with assorted payloads; then random
/// patterns, a quarter of them NaNs.
pub fn sign_bit_inputs(ty: Ty) -> Vec<u64> {
    let width = ty.width();
    let mant = (ty.format().p - 1) as u32;
    let sign = ty.sign();
    let inf = (sign - 1) & !((1u64 << mant) - 1);
    let quiet = 1u64 << (mant - 1);
    let one = ty.round(1.0);
    let one_bits = match ty {
        Ty::F64 => one.to_bits(),
        Ty::F32 => (one as f32).to_bits() as u64,
        Ty::F16 => F16::from_f32(one as f32).0 as u64,
        Ty::BF16 => BF16::from_f32(one as f32).0 as u64,
    };
    let positive = [
        0,
        inf,
        1,
        (1u64 << mant) - 1,
        1u64 << mant,
        one_bits,
        inf - 1,
        inf | quiet,
        inf | quiet | 1,
        inf | quiet | 0x15,
        inf | (quiet - 1),
        inf | 1,
        inf | 0x2a,
        sign - 1,
    ];
    let mut bits: Vec<u64> = positive.iter().flat_map(|&b| [b, b | sign]).collect();
    let mut state = 0x5eed_u64 + width as u64;
    while bits.len() < 2048 {
        let r = rng(&mut state);
        let b = if r.is_multiple_of(4) { inf | (r >> 8) & ((1u64 << mant) - 1) | 1 } else { r >> 3 };
        let b = (b & (sign - 1)) | if (r >> 2) & 1 == 1 { sign } else { 0 };
        bits.push(b);
    }
    bits
}

/// A second operand for [`sign_bit_inputs`]: the same patterns, rotated.
pub fn sign_bit_partners(ty: Ty) -> Vec<u64> {
    let mut bits = sign_bit_inputs(ty);
    bits.rotate_left(7);
    bits
}

/// One sign-bit operation alone in a kernel: the loaded value feeds only
/// it, as in `y[i] = abs(x[i])`, where LLVM had emitted `and.b32` and in
/// other kernels `abs.f32`.
fn plain_generic<T: Float>(op: &'static str, n: usize) -> Kernel {
    Kernel::builder(format!("plain_{op}")).block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let s = param_tensor::<T>("s", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::col_major(n));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let r = match op {
                "abs" => math::abs(x.load(i), false),
                "absf" => math::absf(x.load(i), false),
                "neg" => math::neg(x.load(i), false),
                "minus" => -x.load(i),
                "copysign" => math::copysign(x.load(i), s.load(i), false),
                other => panic!("no sign-bit operation {other}"),
            };
            y.store(i, r);
        });
    })
}

/// The sign-bit operations [`plain_kernel`] covers.
pub const PLAIN_OPS: [&str; 5] = ["abs", "absf", "neg", "minus", "copysign"];

/// `y[i] = op(x[i])` (`copysign(x[i], s[i])`) and nothing else.
pub fn plain_kernel(op: &'static str, ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => plain_generic::<F16>(op, n),
        Ty::BF16 => plain_generic::<BF16>(op, n),
        Ty::F32 => plain_generic::<f32>(op, n),
        Ty::F64 => plain_generic::<f64>(op, n),
    }
}

/// The expected bits of a [`plain_kernel`] result: bit operations.
pub fn plain_want(op: &str, ty: Ty, x: u64, s: u64) -> u64 {
    let sign = ty.sign();
    match op {
        "abs" | "absf" => x & !sign,
        "neg" | "minus" => x ^ sign,
        _ => (x & !sign) | (s & sign),
    }
}

/// Columns of [`sign_context_kernel`]'s float output.
pub const SIGN_CONTEXT_FLOATS: usize = 9;
/// Columns of its integer output.
pub const SIGN_CONTEXT_INTS: usize = 3;

fn sign_context_generic<T: Float, U: Integer + Bitwise>(n: usize) -> Kernel {
    Kernel::builder("sign_context").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let s = param_tensor::<T>("s", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, SIGN_CONTEXT_FLOATS)));
        let z = param_tensor::<U>("z", Layout::row_major((n, SIGN_CONTEXT_INTS)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let (v, w) = (x.load(i), s.load(i));
            // A float operation's result, where LLVM had turned a sign
            // operation into abs.f32 and neg.f32.
            let p = v * w;
            let one = Value::constant(T::from_bits(match T::DTYPE {
                DType::F16 => 0x3c00,
                DType::BF16 => 0x3f80,
                DType::F32 => 1f32.to_bits() as u64,
                _ => 1f64.to_bits(),
            }));
            let floats = [
                math::abs(v, false),
                -v,
                math::copysign(v, w, false),
                p,
                math::abs(p, false),
                -p,
                math::copysign(p, one, false),
                math::copysign(p, -one, false),
                math::copysign(v, -one, false),
            ];
            for (k, r) in floats.into_iter().enumerate() {
                y.store((i, k), r);
            }
            // The results' bits consumed as integers, where LLVM had folded
            // a negation into an integer add.
            let ints = [
                (-p).bitcast::<U>() + U::from_bits(1),
                math::abs(p, false).bitcast::<U>() + U::from_bits(1),
                math::copysign(p, -one, false).bitcast::<U>() + U::from_bits(1),
            ];
            for (k, r) in ints.into_iter().enumerate() {
                z.store((i, k), r);
            }
        });
    })
}

/// The sign-bit operations next to other code: on the arguments, on a
/// product, with constant signs, and with their bits used as integers.
pub fn sign_context_kernel(ty: Ty, n: usize) -> Kernel {
    match ty {
        Ty::F16 => sign_context_generic::<F16, u16>(n),
        Ty::BF16 => sign_context_generic::<BF16, u16>(n),
        Ty::F32 => sign_context_generic::<f32, u32>(n),
        Ty::F64 => sign_context_generic::<f64, u64>(n),
    }
}

/// The expected bits of [`sign_context_kernel`]'s outputs for arguments
/// `x` and `s`, given its product `p` (column 3): `None` for column 3
/// itself, which the caller checks against the product.
pub fn sign_context_want(ty: Ty, x: u64, s: u64, p: u64) -> ([Option<u64>; SIGN_CONTEXT_FLOATS], [u64; 3]) {
    let sign = ty.sign();
    let mask = if ty.width() == 64 { u64::MAX } else { (1u64 << ty.width()) - 1 };
    let floats = [
        Some(x & !sign),
        Some(x ^ sign),
        Some((x & !sign) | (s & sign)),
        None,
        Some(p & !sign),
        Some(p ^ sign),
        Some(p & !sign),
        Some(p | sign),
        Some(x | sign),
    ];
    let ints =
        [(p ^ sign).wrapping_add(1) & mask, (p & !sign).wrapping_add(1) & mask, (p | sign).wrapping_add(1) & mask];
    (floats, ints)
}

/// Columns of [`ftz_kernel`]'s output.
pub const FTZ_OPS: usize = 7;

/// `abs`, `min`, `max` and `clamp` with `ftz=True` (and `propagate_nan=`) on
/// Float32: `abs.ftz.f32`, `min.ftz.f32`, `max.ftz.f32`,
/// `min.ftz.NaN.f32` and `max.ftz.NaN.f32`.
pub fn ftz_kernel(n: usize) -> Kernel {
    Kernel::builder("ftz").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major(n));
        let b = param_tensor::<f32>("b", Layout::col_major(n));
        let y = param_tensor::<f32>("y", Layout::row_major((n, FTZ_OPS)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let (x, z) = (a.load(i), b.load(i));
            let ftz = MathOptions { ftz: true, ..MathOptions::default() };
            let ftz_nan = MathOptions { propagate_nan: true, ..ftz };
            let hi = math::abs(z, false);
            let results = [
                math::abs_with(x, ftz),
                math::min_with(x, z, ftz),
                math::max_with(x, z, ftz),
                math::min_with(x, z, ftz_nan),
                math::max_with(x, z, ftz_nan),
                math::clamp_with(x, -hi, hi, ftz),
                math::clamp_with(x, -hi, hi, ftz_nan),
            ];
            for (k, r) in results.into_iter().enumerate() {
                y.store((i, k), r);
            }
        });
    })
}

/// Float32 argument pairs for [`ftz_kernel`]: every pair of zeros,
/// subnormals, normals, infinities and NaNs (payloads and signaling ones
/// too), then random pairs, many subnormal.
pub fn ftz_inputs() -> Vec<(u64, u64)> {
    let specials: [u32; 14] = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_fff0,
        0x0080_0000,
        0x3f80_0000,
        0xbfc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc0_0000,
        0x7f80_0001,
    ];
    let mut pairs: Vec<(u64, u64)> = Vec::new();
    for &a in &specials {
        for &b in &specials {
            pairs.push((a as u64, b as u64));
        }
    }
    let mut state = 17u64;
    while pairs.len() < 2048 {
        let pick = |state: &mut u64| -> u64 {
            let r = rng(state);
            let b = match r % 4 {
                0 => r >> 41,
                1 => specials[(r >> 8) as usize % specials.len()] as u64,
                _ => (r >> 16) & 0xffff_ffff,
            };
            (b & 0x7fff_ffff) | ((r >> 3) & 1) << 31
        };
        let a = pick(&mut state);
        let b = pick(&mut state);
        pairs.push((a, b));
    }
    pairs
}

/// The expected results of [`ftz_kernel`] (`None` where the result is NaN,
/// which the caller checks with `is_nan`): `.ftz` flushes the operands, `min`
/// and `max` ignore a NaN operand unless `propagate_nan`, and order -0
/// below +0.
pub fn ftz_want(a: u64, b: u64) -> [Option<u64>; FTZ_OPS] {
    let ty = Ty::F32;
    let flush = |v: u64| ty.flush_bits(v);
    let nan = |v: u64| ty.is_nan_bits(v);
    let value = |v: u64| ty.value(v);
    let pick = |x: u64, y: u64, is_max: bool, propagate: bool| -> Option<u64> {
        let (x, y) = (flush(x), flush(y));
        match (nan(x), nan(y)) {
            (true, true) => None,
            (true, false) | (false, true) if propagate => None,
            (true, false) => Some(y),
            (false, true) => Some(x),
            _ => {
                let x_first = match value(x).partial_cmp(&value(y)) {
                    Some(std::cmp::Ordering::Less) => true,
                    Some(std::cmp::Ordering::Greater) => false,
                    _ => x >> 31 == 1,
                };
                Some(if x_first != is_max { x } else { y })
            }
        }
    };
    let hi = b & 0x7fff_ffff;
    let lo = hi | 0x8000_0000;
    let clamp = |propagate: bool| match pick(a, hi, false, propagate) {
        Some(m) => pick(lo, m, true, propagate),
        // A NaN from min: max sees it as an operand.
        None => pick(lo, 0x7fff_ffff, true, propagate),
    };
    let abs = if nan(a) { None } else { Some(flush(a) & 0x7fff_ffff) };
    [
        abs,
        pick(a, b, false, false),
        pick(a, b, true, false),
        pick(a, b, false, true),
        pick(a, b, true, true),
        clamp(false),
        clamp(true),
    ]
}

fn absi_generic<T: Integer + Bitwise>(n: usize) -> Kernel {
    Kernel::builder("absi").block_dim([BLOCK as u32, 1, 1]).build(|| {
        let x = param_tensor::<T>("x", Layout::col_major(n));
        let y = param_tensor::<T>("y", Layout::row_major((n, 2)));
        let i = block_idx().x * BLOCK as i32 + thread_idx().x;
        if_(i.lt(n as i32), || {
            let v = x.load(i);
            y.store((i, 0), math::absi(v));
            y.store((i, 1), math::abs(v, false));
        });
    })
}

/// `absi(x[i])` and `abs(x[i])` on an integer type.
pub fn absi_kernel<T: Integer + Bitwise>(n: usize) -> Kernel {
    absi_generic::<T>(n)
}

/// Integer arguments as raw bits of `bits` width: the extremes, -1, 0 and 1,
/// then random values.
pub fn absi_inputs(bits: u32) -> Vec<u64> {
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let min = 1u64 << (bits - 1);
    let mut xs = vec![0, 1, mask, min, min + 1, min - 1, mask - 1, 2];
    let mut state = 23u64 + bits as u64;
    while xs.len() < 1024 {
        let r = rng(&mut state);
        xs.push(if r.is_multiple_of(2) { r & mask } else { (r >> 40).wrapping_neg() & mask });
    }
    xs
}

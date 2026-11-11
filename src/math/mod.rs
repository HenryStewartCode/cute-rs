//! CuTe DSL's `cute.math`: elementwise math on runtime values.
//!
//! Every function of CuTe DSL 4.8.1's `cute.math` is here, under the same
//! name. (CuTe DSL also exports them at the top level, as `cute.exp2` and
//! so on; here they stay in `cute::math`, because the crate root already
//! has `fma`, `min` and `max` of its own.) Where CuTe DSL takes
//! `fastmath=`, so does the Rust function, as its last parameter:
//! `math::exp(x, false)` is `cute.math.exp(x)` and `math::exp(x, true)` is
//! `cute.math.exp(x, fastmath=True)`. CuTe DSL's other keyword options
//! (`approx=`, `ftz=`, `rounding=`, `full=`, `propagate_nan=`) are fields
//! of [`MathOptions`], taken by the `_with` variant of each function that
//! has them: `math::exp2_with(x, MathOptions::APPROX.with_ftz())` is
//! `cute.math.exp2(x, approx=True, ftz=True)`. Each function accepts the
//! options its CuTe DSL counterpart accepts, and the kernel build panics,
//! naming the option, on any other or on a combination CuTe DSL rejects.
//!
//! The functions take scalar values. Apply one to a register tensor with
//! [`Tensor::map_in_place`](crate::Tensor::map_in_place), e.g.
//! `frag.map_in_place(|x| math::exp2(x, true))`.
//!
//! # Three ways to compute
//!
//! * **Precise** (the default, `fastmath = false`): CuTe DSL calls
//!   libdevice here, which needs the CUDA toolkit. cute-rs expands every
//!   function into exactly rounded operations instead (add, multiply,
//!   fma, divide and square root, rounded to nearest), comparisons,
//!   selects and integer operations, ported from FreeBSD's msun (Sun's
//!   fdlibm, whose permissive license notice each file keeps). No libdevice
//!   or CUDA toolkit file is needed, no hardware approximation is used, and
//!   because every operation is exactly rounded the GPU computes the
//!   simulator's bits (`tests/gpu.rs` checks every function and type on
//!   an SM89 GPU). The expansions return the one quiet NaN (`f32::NAN`,
//!   `f64::NAN`) for every NaN result; the functions that are a single
//!   instruction (`sqrt`, `rcp`, `div`, `fma`, `add`, `sub`, `mul`, the
//!   roundings, `min`, `max`) return the instruction's NaN (except where
//!   LLVM removes one that cannot change a number, such as `x * 1.0`,
//!   which then passes a NaN through unchanged); `abs`, `neg`
//!   and `copysign` are bit operations on the sign (inline PTX `and` and
//!   `xor`, which LLVM cannot turn into `abs.f32` or `neg.f32`), so a NaN
//!   keeps its payload, as IEEE 754 specifies.
//! * **Fast** (`fastmath = true`): Float32 (and Float16 and BFloat16,
//!   computed in Float32) use the hardware approximations, as CuTe DSL's
//!   `fastmath=True` and CUDA's `-use_fast_math` do, with `.ftz`
//!   (subnormal arguments and results of the approximated instruction
//!   flush to zero): `exp2` is `ex2.approx.ftz.f32`, `exp(x)` is
//!   `ex2.approx.ftz.f32(x * log2(e))`, `log2` is `lg2.approx.ftz.f32`,
//!   `log` and `log10` scale it, `sin` and `cos` are `sin.approx.ftz.f32`
//!   and `cos.approx.ftz.f32`, `tan` divides them with
//!   `div.approx.ftz.f32`, `tanh` is `tanh.approx.f32`, `pow(x, y)` is
//!   `ex2(y * lg2(x))`, and `sqrt`, `rsqrt`, `rcp` and `div` are
//!   `sqrt.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `rcp.approx.ftz.f32`
//!   and `div.approx.ftz.f32`. The other functions (which have no
//!   hardware approximation, nor a fast form in libdevice), and every
//!   Float64 function, compute as in precise mode, bit for bit. Each
//!   function documents its fast-mode error bound, derived from the PTX
//!   ISA's bounds for the instructions it uses; `tests/gpu.rs` holds the
//!   GPU to them. The simulator computes the approximated function
//!   (correctly rounded, with the flushing), so fast mode is not bit for
//!   bit. (CuTe DSL's `fastmath=True` on `add`, `sub`, `mul` and `fma`
//!   only lets its compiler contract and reorder; here it changes nothing.)
//! * **Exact instruction control** (the `_with` functions): `approx`, `ftz`,
//!   `rounding` and `full` choose one PTX instruction, such as
//!   `add.rz.ftz.f32` or `sqrt.approx.f32`, on Float32 or Float64 only, as
//!   in CuTe DSL. The directed roundings are exact IEEE operations, which
//!   the simulator computes with integer arithmetic, bit for bit like the
//!   GPU; the approximations behave as in fast mode.
//!
//! # Accuracy
//!
//! The precise-mode bounds, in ulps of the result, cover the largest
//! errors found against a double-double reference over 2^22 arguments of
//! each function (signed zeros, infinities, NaN, subnormals, the extremes,
//! known hard cases and their neighbours, and uniform and random samples;
//! `tests/math.rs` sweeps 2^14 by default and asserts these bounds).
//! Where msun's formulas lose more than an ulp (the hyperbolic functions,
//! their inverses, `erfc` and `atan2`), the expansions carry the rounding
//! errors of the critical steps in extra precision. libdevice's documented bounds
//! (CUDA 13) are for comparison.
//!
//! | function | Float32 | Float64 | libdevice (f32 / f64) |
//! |---|---|---|---|
//! | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p` | 1 | 1 | 1 to 2 / 1 |
//! | `sin`, `cos` | 1.5 | 1 | 2 / 2 |
//! | `tan` | 1 | 1 | 4 / 2 |
//! | `asin`, `acos`, `atan` | 1 | 1 | 2 / 2 |
//! | `atan2` | 1.5 | 1.5 | 3 / 2 |
//! | `sinh`, `cosh`, `tanh` | 1 | 1 | 3, 2, 2 / 2, 1, 1 |
//! | `asinh`, `acosh`, `atanh` | 1 | 1 | 3, 4, 3 / 3, 3, 2 |
//! | `erf` | 1 | 1 | 2 / 2 |
//! | `erfc` | 2.5 | 2.5 | 4 / 5 |
//! | `pow`, `cbrt` | 1 | 1 | 4, 1 / 2, 1 |
//! | `fpowi` | 1 | 1 | |
//! | `sqrt`, `rcp`, `div` | 0.5 | 0.5 | 0.5 |
//! | `rsqrt` | 0.5 | 0.5 | 2 / 1 |
//! | `rem`, `copysign`, `round`, `floor`, `ceil`, `trunc`, `roundeven`, `abs`, `neg` | exact | exact | |
//!
//! Float16 and BFloat16 arguments are computed in Float32 and rounded, as
//! CuTe DSL's lowering does: the unary functions are within 0.501 ulp on
//! every one of the 65536 arguments of each type, correctly rounded except
//! for a few arguments (up to about 1 in 3000) whose result lies next to a
//! halfway point, where the Float32 error decides the rounding.
//!
//! # The `Value` methods
//!
//! [`Value::exp2`], [`Value::log2`], [`Value::sin`], [`Value::cos`],
//! [`Value::rsqrt`] and [`Value::rcp`] are the plain hardware
//! approximations, without `.ftz`, on Float32 and narrower types (Float16
//! and BFloat16 are computed in Float32), and `rsqrt` and `rcp` on Float64
//! too. On Float32, `x.exp2()` is `math::exp2_with(x, MathOptions::APPROX)`
//! (the `_with` functions take only Float32 and Float64), and `rcp` on
//! Float64 is `rcp.rn.f64`, the same bits as `math::rcp(x, false)`.
//! [`Value::exp`] and [`Value::ln`] scale them, like `math::exp(x, true)`
//! and `math::log(x, true)` without `.ftz`. [`Value::sqrt`] is
//! `math::sqrt(x, false)` (`sqrt.rn`), and `floor`, `ceil`, `trunc`,
//! `rint`, `abs`, `min` and `max` are `math::floor`, `math::ceil`,
//! `math::trunc`, `math::roundeven`, `math::abs`, `math::min` and
//! `math::max`; `-x` is `math::neg(x, false)`.

mod erf;
mod exp;
mod invtrig;
mod log;
mod misc;
mod pow;
mod real;
mod trig;

use std::panic::Location;

use crate::builder::{self, LocationScope};
use crate::ir::{DType, FloatFn, FloatMode, ValueId};
use crate::value::{Arith, Bitwise, Boolean, Float, Float32, Integer, IntoValue, Numeric, Value, select};

pub use crate::ir::RoundingMode;

/// CuTe DSL's keyword options of `cute.math`, for the `_with` functions.
/// Each function accepts the options its CuTe DSL counterpart takes and
/// panics, naming the option, on any other.
///
/// * `fastmath`: let the compiler use hardware approximations (see the
///   [module documentation](self)).
/// * `approx`: the PTX `.approx` instruction (`sin`, `cos`, `exp2`, `log2`,
///   `tanh`, `rsqrt`, `sqrt`, `rcp`, `div`).
/// * `ftz`: flush subnormal inputs and results to zero (`.ftz`). On a
///   transcendental function it needs `approx`.
/// * `rounding`: an IEEE 754 rounding direction (`add`, `mul`, `fma`,
///   `div`, `rcp`, `sqrt`).
/// * `full`: `div.full`, a full-range approximate division.
/// * `propagate_nan`: `min`, `max` and `clamp` return NaN if an operand is
///   NaN (IEEE 754 `minimum`) instead of ignoring it (`minimumNumber`).
///
/// `fastmath` cannot be combined with `approx`, `ftz`, `rounding` or
/// `full`, which choose one PTX instruction, and those need Float32 or
/// Float64 operands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MathOptions {
    pub fastmath: bool,
    pub approx: bool,
    pub ftz: bool,
    pub rounding: Option<RoundingMode>,
    pub full: bool,
    pub propagate_nan: bool,
}

impl MathOptions {
    /// `fastmath=True`.
    pub const FASTMATH: MathOptions =
        MathOptions { fastmath: true, approx: false, ftz: false, rounding: None, full: false, propagate_nan: false };
    /// `approx=True`.
    pub const APPROX: MathOptions =
        MathOptions { fastmath: false, approx: true, ftz: false, rounding: None, full: false, propagate_nan: false };

    /// `rounding=mode`.
    pub const fn rounding(mode: RoundingMode) -> MathOptions {
        MathOptions {
            fastmath: false,
            approx: false,
            ftz: false,
            rounding: Some(mode),
            full: false,
            propagate_nan: false,
        }
    }

    /// These options with `ftz=True`.
    pub const fn with_ftz(self) -> MathOptions {
        MathOptions { ftz: true, ..self }
    }
}

/// The options a function takes besides `fastmath`.
#[derive(Clone, Copy)]
struct Accepts {
    approx: bool,
    ftz: bool,
    rounding: bool,
    full: bool,
    propagate_nan: bool,
    fastmath: bool,
    /// `ftz` needs `approx` (transcendental functions).
    ftz_needs_approx: bool,
}

const NONE: Accepts = Accepts {
    approx: false,
    ftz: false,
    rounding: false,
    full: false,
    propagate_nan: false,
    fastmath: true,
    ftz_needs_approx: false,
};

/// Check `opts` against what `op` accepts, with CuTe DSL's rules.
#[track_caller]
fn check_options(op: &str, opts: &MathOptions, accepts: Accepts, ty: DType) {
    let offered = [
        ("approx", opts.approx, accepts.approx),
        ("ftz", opts.ftz, accepts.ftz),
        ("rounding", opts.rounding.is_some(), accepts.rounding),
        ("full", opts.full, accepts.full),
        ("propagate_nan", opts.propagate_nan, accepts.propagate_nan),
        ("fastmath", opts.fastmath, accepts.fastmath),
    ];
    for (name, set, ok) in offered {
        assert!(!set || ok, "cute::math::{op} takes no {name} option");
    }
    let nvvm: Vec<&str> =
        [("approx", opts.approx), ("ftz", opts.ftz), ("rounding", opts.rounding.is_some()), ("full", opts.full)]
            .iter()
            .filter(|(_, set)| *set)
            .map(|(n, _)| *n)
            .collect();
    assert!(
        !(opts.fastmath && !nvvm.is_empty()),
        "cute::math::{op}: fastmath and {} are mutually exclusive: fastmath lets the compiler choose the \
         instruction, {} choose it exactly",
        nvvm.join("/"),
        nvvm.join("/")
    );
    assert!(
        !(accepts.ftz_needs_approx && opts.ftz && !opts.approx),
        "cute::math::{op}: ftz needs approx: there is no non-approximate ftz variant of a transcendental function"
    );
    let alternatives = [opts.approx, opts.full, opts.rounding.is_some()].iter().filter(|&&b| b).count();
    assert!(alternatives <= 1, "cute::math::{op}: approx, full and rounding are alternatives; choose one");
    assert!(
        nvvm.is_empty() || matches!(ty, DType::F32 | DType::F64),
        "cute::math::{op}: {} needs Float32 or Float64 operands, not {ty} (these options choose one PTX instruction)",
        nvvm.join("/")
    );
}

/// Reinterpret a value handle as the same element type under its concrete
/// name (callers check the dtype).
fn retype<A: Numeric, B: Numeric>(v: Value<A>) -> Value<B> {
    debug_assert_eq!(A::DTYPE, B::DTYPE);
    Value::from_id(v.id)
}

/// An f32 result rounded to a narrower float type, NaN as that type's quiet
/// NaN.
#[track_caller]
fn narrow<T: Float>(r: Float32) -> Value<T> {
    let nan = Value::constant(T::from_bits(nan_bits(T::DTYPE)));
    select(r.ne(r), nan, r.to::<T>())
}

/// The quiet NaN results are given as.
fn nan_bits(t: DType) -> u64 {
    match t {
        DType::F16 => 0x7e00,
        DType::BF16 => 0x7fc0,
        DType::F32 => f32::NAN.to_bits() as u64,
        _ => f64::NAN.to_bits(),
    }
}

type Fn1<R> = fn(Value<R>) -> Value<R>;
type Fn2<R> = fn(Value<R>, Value<R>) -> Value<R>;

/// A unary function: `precise` for f32 and f64, or `fast` (f32 only) under
/// fastmath. Float16 and BFloat16 compute in f32.
#[track_caller]
fn unary<T: Float>(x: Value<T>, fastmath: bool, p32: Fn1<f32>, p64: Fn1<f64>, fast: Option<Fn1<f32>>) -> Value<T> {
    let f32_fn = match (fastmath, fast) {
        (true, Some(f)) => f,
        _ => p32,
    };
    match T::DTYPE {
        DType::F32 => retype(f32_fn(retype(x))),
        DType::F64 => retype(p64(retype(x))),
        _ => narrow(f32_fn(x.to::<f32>())),
    }
}

/// A binary function, like [`unary`].
#[track_caller]
fn binary<T: Float>(
    x: Value<T>,
    y: Value<T>,
    fastmath: bool,
    p32: Fn2<f32>,
    p64: Fn2<f64>,
    fast: Option<Fn2<f32>>,
) -> Value<T> {
    let f32_fn = match (fastmath, fast) {
        (true, Some(f)) => f,
        _ => p32,
    };
    match T::DTYPE {
        DType::F32 => retype(f32_fn(retype(x), retype(y))),
        DType::F64 => retype(p64(retype(x), retype(y))),
        _ => narrow(f32_fn(x.to::<f32>(), y.to::<f32>())),
    }
}

/// A floating-point instruction with explicit modifiers.
#[track_caller]
fn inst<T: Numeric>(func: FloatFn, args: &[Value<T>], mode: FloatMode, ftz: bool, nan: bool) -> Value<T> {
    let loc = Location::caller();
    let ids: Vec<ValueId> = args.iter().map(|a| a.checked()).collect();
    Value::from_id(builder::with(|b| b.float_inst(func, ids, mode, ftz, nan, loc)))
}

/// The `.approx.ftz` approximation of `func` (`tanh.approx` has no `.ftz`).
#[track_caller]
fn approx_ftz(func: FloatFn, x: Float32) -> Float32 {
    inst(func, &[x], FloatMode::Approx, func != FloatFn::Tanh, false)
}

/// The fast-mode Float32 implementations.
mod fast {
    use super::*;

    pub(super) fn exp2(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Ex2, x)
    }
    pub(super) fn exp(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Ex2, x * std::f32::consts::LOG2_E)
    }
    pub(super) fn log2(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Lg2, x)
    }
    pub(super) fn log(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Lg2, x) * std::f32::consts::LN_2
    }
    pub(super) fn log10(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Lg2, x) * std::f32::consts::LOG10_2
    }
    pub(super) fn sin(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Sin, x)
    }
    pub(super) fn cos(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Cos, x)
    }
    pub(super) fn tan(x: Float32) -> Float32 {
        div(sin(x), cos(x))
    }
    pub(super) fn tanh(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Tanh, x)
    }
    pub(super) fn pow(x: Float32, y: Float32) -> Float32 {
        approx_ftz(FloatFn::Ex2, y * approx_ftz(FloatFn::Lg2, x))
    }
    pub(super) fn sqrt(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Sqrt, x)
    }
    pub(super) fn rsqrt(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Rsqrt, x)
    }
    pub(super) fn rcp(x: Float32) -> Float32 {
        approx_ftz(FloatFn::Rcp, x)
    }
    pub(super) fn div(x: Float32, y: Float32) -> Float32 {
        inst(FloatFn::Div, &[x, y], FloatMode::Approx, true, false)
    }
}

macro_rules! unary_fn {
    ($(#[$doc:meta])* $name:ident, $module:ident, $fast:expr) => {
        $(#[$doc])*
        #[track_caller]
        pub fn $name<T: Float>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
            let _at = LocationScope::new();
            unary(x.into_value(), fastmath, $module::$name::<f32>, $module::$name::<f64>, $fast)
        }
    };
}

unary_fn!(
    /// `e^x`. Precise: at most 1 ulp. Fast: `ex2.approx.ftz.f32(x *
    /// log2(e))` (CUDA's `__expf`), within `2.5 + 1.5 |x|` ulp (the PTX
    /// ISA's 2 ulp for `ex2`, which it measures from the correctly rounded
    /// result, plus the rounding of `x * log2(e)`, which the exponential
    /// magnifies by `|x|`); results below 2^-126 flush to zero.
    exp,
    exp,
    Some(fast::exp)
);
unary_fn!(
    /// `2^x`. Precise: at most 1 ulp. Fast: `ex2.approx.ftz.f32`, within
    /// 2 ulp of the correctly rounded result (PTX ISA), so 2.5 ulp of the
    /// true value; results below 2^-126 flush to zero.
    exp2,
    exp,
    Some(fast::exp2)
);
unary_fn!(
    /// `e^x - 1`, accurate near zero. At most 1 ulp; fastmath does not
    /// change it.
    expm1,
    exp,
    None
);
unary_fn!(
    /// Natural logarithm. Precise: at most 1 ulp. Fast:
    /// `lg2.approx.ftz.f32(x) * ln 2` (CUDA's `__logf`), within 2^-22
    /// absolute for `x` in `(0.5, 2)` and 2^-21.6 relative elsewhere (the
    /// PTX ISA's bounds for `lg2`, plus the product's rounding); a
    /// subnormal `x` flushes to zero, giving -inf.
    log,
    log,
    Some(fast::log)
);
unary_fn!(
    /// Base-2 logarithm. Precise: at most 1 ulp. Fast: `lg2.approx.ftz.f32`,
    /// within 2^-22 absolute for `x` in `(0.5, 2)` and 2^-22 relative
    /// elsewhere (PTX ISA); a subnormal `x` flushes to zero.
    log2,
    log,
    Some(fast::log2)
);
unary_fn!(
    /// Base-10 logarithm. Precise: at most 1 ulp. Fast:
    /// `lg2.approx.ftz.f32(x) * log10(2)` (CUDA's `__log10f`), within
    /// 2^-23 absolute for `x` in `(0.5, 2)` and 2^-21.6 relative
    /// elsewhere; a subnormal `x` flushes to zero.
    log10,
    log,
    Some(fast::log10)
);
unary_fn!(
    /// `log(1 + x)`, accurate near zero. At most 1 ulp; fastmath does not
    /// change it.
    log1p,
    log,
    None
);
unary_fn!(
    /// Sine. Precise: at most 1.5 ulp (Float32) or 1 ulp (Float64) for
    /// every argument, huge ones reduced exactly (Payne and Hanek). Fast:
    /// `sin.approx.ftz.f32`, within 2^-20.5 absolute for `|x| <= 2 pi` and
    /// 2^-14.7 for `|x| <= 100 pi` (PTX ISA); no bound beyond.
    sin,
    trig,
    Some(fast::sin)
);
unary_fn!(
    /// Cosine. Precise: at most 1.5 ulp (Float32) or 1 ulp (Float64).
    /// Fast: `cos.approx.ftz.f32`, within 2^-20.4 absolute for `|x| <= 2
    /// pi` and 2^-14.7 for `|x| <= 100 pi` (PTX ISA); no bound beyond.
    cos,
    trig,
    Some(fast::cos)
);
unary_fn!(
    /// Tangent. Precise: at most 1 ulp. Fast:
    /// `div.approx.ftz(sin.approx.ftz(x), cos.approx.ftz(x))` (like CUDA's
    /// `__tanf`), whose error is the sine's and cosine's over `|cos x|`:
    /// within `(2^-20.5 + 2^-20.4 |tan x|) / |cos x|` plus 2 ulp for
    /// `|x| <= 2 pi`.
    tan,
    trig,
    Some(fast::tan)
);
unary_fn!(
    /// Arc sine. At most 1 ulp; fastmath does not change it.
    asin,
    invtrig,
    None
);
unary_fn!(
    /// Arc cosine. At most 1 ulp; fastmath does not change it.
    acos,
    invtrig,
    None
);
unary_fn!(
    /// Arc tangent. At most 1 ulp; fastmath does not change it.
    atan,
    invtrig,
    None
);
unary_fn!(
    /// Hyperbolic sine. At most 1 ulp; fastmath does not change it.
    sinh,
    exp,
    None
);
unary_fn!(
    /// Hyperbolic cosine. At most 1 ulp; fastmath does not change it.
    cosh,
    exp,
    None
);
unary_fn!(
    /// Hyperbolic tangent. Precise: at most 1 ulp. Fast: `tanh.approx.f32`
    /// (CUDA's `__tanhf`; sm_75 and newer), within 2^-11 relative (PTX
    /// ISA), subnormal results included.
    tanh,
    exp,
    Some(fast::tanh)
);
unary_fn!(
    /// Inverse hyperbolic sine. At most 1 ulp; fastmath does not change it.
    asinh,
    log,
    None
);
unary_fn!(
    /// Inverse hyperbolic cosine. At most 1 ulp; fastmath does not change
    /// it.
    acosh,
    log,
    None
);
unary_fn!(
    /// Inverse hyperbolic tangent. At most 1 ulp; fastmath does not change
    /// it.
    atanh,
    log,
    None
);
unary_fn!(
    /// The error function. At most 1 ulp; fastmath does not change it.
    erf,
    erf,
    None
);
unary_fn!(
    /// The complementary error function `1 - erf(x)`, accurate in its tail.
    /// At most 2.5 ulp (msun's algorithm, with its two exponentials of the
    /// tail combined into one and an intermediate sum compensated);
    /// fastmath does not change it.
    erfc,
    erf,
    None
);
unary_fn!(
    /// Cube root. At most 1 ulp; fastmath does not change it. The Float32
    /// version takes its two Halley steps in Float64, as msun's does.
    cbrt,
    misc,
    None
);

/// Square root, correctly rounded (`sqrt.rn`). Fast: `sqrt.approx.ftz.f32`,
/// within 2^-23 relative (PTX ISA).
#[track_caller]
pub fn sqrt<T: Float>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    sqrt_with(x, MathOptions { fastmath, ..Default::default() })
}

/// [`sqrt`] with CuTe DSL's `approx=`, `ftz=` and `rounding=`:
/// `sqrt.approx[.ftz].f32` or `sqrt.{rnd}[.ftz]`.
#[track_caller]
pub fn sqrt_with<T: Float>(x: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let x = x.into_value();
    check_options("sqrt", &opts, Accepts { approx: true, ftz: true, rounding: true, ..NONE }, T::DTYPE);
    if opts.approx {
        return inst(FloatFn::Sqrt, &[x], FloatMode::Approx, opts.ftz, false);
    }
    if let Some(mode) = rounding(&opts) {
        return inst(FloatFn::Sqrt, &[x], mode, opts.ftz, false);
    }
    unary(x, opts.fastmath, |x| x.sqrt(), |x| x.sqrt(), Some(fast::sqrt))
}

/// `1 / sqrt(x)`. Precise: correctly rounded except in rare ties (at most
/// 0.5 ulp plus a hair), from a Newton step with an exact residual. Fast:
/// `rsqrt.approx.ftz.f32`, within 2^-22.9 relative (PTX ISA).
#[track_caller]
pub fn rsqrt<T: Float>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    rsqrt_with(x, MathOptions { fastmath, ..Default::default() })
}

/// [`rsqrt`] with CuTe DSL's `approx=` and `ftz=`: `rsqrt.approx[.ftz]` on
/// Float32 or Float64.
#[track_caller]
pub fn rsqrt_with<T: Float>(x: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let x = x.into_value();
    check_options("rsqrt", &opts, Accepts { approx: true, ftz: true, ftz_needs_approx: true, ..NONE }, T::DTYPE);
    if opts.approx {
        return inst(FloatFn::Rsqrt, &[x], FloatMode::Approx, opts.ftz, false);
    }
    unary(x, opts.fastmath, misc::rsqrt::<f32>, misc::rsqrt::<f64>, Some(fast::rsqrt))
}

macro_rules! transcendental_with {
    ($(#[$doc:meta])* $name:ident, $with:ident, $func:expr, $ftz:expr) => {
        $(#[$doc])*
        #[track_caller]
        pub fn $with<T: Float>(x: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
            let _at = LocationScope::new();
            let x = x.into_value();
            let accepts = Accepts { approx: true, ftz: $ftz, ftz_needs_approx: true, ..NONE };
            check_options(stringify!($name), &opts, accepts, T::DTYPE);
            if opts.approx {
                return inst($func, &[x], FloatMode::Approx, opts.ftz, false);
            }
            $name(x, opts.fastmath)
        }
    };
}

transcendental_with!(
    /// [`sin`] with CuTe DSL's `approx=` and `ftz=`: `sin.approx[.ftz].f32`.
    sin,
    sin_with,
    FloatFn::Sin,
    true
);
transcendental_with!(
    /// [`cos`] with CuTe DSL's `approx=` and `ftz=`: `cos.approx[.ftz].f32`.
    cos,
    cos_with,
    FloatFn::Cos,
    true
);
transcendental_with!(
    /// [`exp2`] with CuTe DSL's `approx=` and `ftz=`: `ex2.approx[.ftz].f32`.
    exp2,
    exp2_with,
    FloatFn::Ex2,
    true
);
transcendental_with!(
    /// [`log2`] with CuTe DSL's `approx=` and `ftz=`: `lg2.approx[.ftz].f32`.
    log2,
    log2_with,
    FloatFn::Lg2,
    true
);
transcendental_with!(
    /// [`tanh`] with CuTe DSL's `approx=`: `tanh.approx.f32`.
    tanh,
    tanh_with,
    FloatFn::Tanh,
    false
);

/// `(sin(x), cos(x))`, sharing the argument reduction. Accuracy as for
/// [`sin`] and [`cos`]; fast mode uses both approximations.
#[track_caller]
pub fn sincos<T: Float>(x: impl IntoValue<T>, fastmath: bool) -> (Value<T>, Value<T>) {
    let _at = LocationScope::new();
    let x = x.into_value();
    let f32_fn = |x: Float32| if fastmath { (fast::sin(x), fast::cos(x)) } else { trig::sincos::<f32>(x) };
    match T::DTYPE {
        DType::F32 => {
            let (s, c) = f32_fn(retype(x));
            (retype(s), retype(c))
        }
        DType::F64 => {
            let (s, c) = trig::sincos::<f64>(retype(x));
            (retype(s), retype(c))
        }
        _ => {
            let (s, c) = f32_fn(x.to::<f32>());
            (narrow(s), narrow(c))
        }
    }
}

/// `atan2(y, x)`, the angle of the point `(x, y)`. At most 1.5 ulp;
/// fastmath does not change it.
#[track_caller]
pub fn atan2<T: Float>(y: impl IntoValue<T>, x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    binary(y.into_value(), x.into_value(), fastmath, invtrig::atan2::<f32>, invtrig::atan2::<f64>, None)
}

/// `base^exponent`. Precise: at most 1 ulp, with C's special cases
/// (`pow(x, 0) = 1`, `pow(1, y) = 1`, a negative base to an odd integer is
/// negative, to a non-integer NaN). Fast: CUDA's `__powf`,
/// `ex2.approx.ftz(exponent * lg2.approx.ftz(base))`, for positive bases
/// (a negative one gives NaN). The exponent it raises 2 to is off by `E =
/// |y| e + 2^-24 |y log2 x|`, where `e` is `lg2`'s error (2^-22, relative
/// outside `(0.5, 2)`), so the result is within a factor `2^E` of the true
/// value, and `ex2` adds its 2 ulp of the correctly rounded result:
/// accurate for moderate `|y log2 x|` only.
#[track_caller]
pub fn pow<T: Float>(base: impl IntoValue<T>, exponent: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    binary(base.into_value(), exponent.into_value(), fastmath, pow::pow::<f32>, pow::pow::<f64>, Some(fast::pow))
}

/// `base^exponent` for an integer exponent (`pown`): exact special cases,
/// and the sign from the exponent's parity. At most 1 ulp, except that a
/// Float32 exponent beyond 2^24 in magnitude is rounded to Float32 (only
/// bases within about 2^-17 of 1 then give finite, slightly less accurate,
/// results). fastmath does not change it.
#[track_caller]
pub fn fpowi<T: Float>(base: impl IntoValue<T>, exponent: impl IntoValue<i32>, _fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    let (b, e) = (base.into_value(), exponent.into_value());
    match T::DTYPE {
        DType::F32 => retype(pow::fpowi::<f32>(retype(b), e)),
        DType::F64 => retype(pow::fpowi::<f64>(retype(b), e)),
        _ => narrow(pow::fpowi::<f32>(b.to(), e)),
    }
}

/// `base^exponent` for integers, wrapping on overflow (`math.ipowi`). A
/// negative exponent gives 0, except for bases 1 and -1; `0^negative` (no
/// integer result) gives 0.
#[track_caller]
pub fn ipowi<T: Integer + Bitwise>(base: impl IntoValue<T>, exponent: impl IntoValue<T>) -> Value<T> {
    let _at = LocationScope::new();
    misc::ipowi(base.into_value(), exponent.into_value())
}

/// The floating-point remainder of `x / y` with the quotient truncated
/// (C's `fmod`, LLVM's `frem`), which is exact. fastmath does not change it.
#[track_caller]
pub fn rem<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    binary(x.into_value(), y.into_value(), fastmath, misc::rem::<f32>, misc::rem::<f64>, None)
}

/// The magnitude of `mag` with the sign of `sign`, a bit operation (IEEE
/// 754 `copySign`): a NaN `mag` keeps its payload.
#[track_caller]
pub fn copysign<T: Float>(mag: impl IntoValue<T>, sign: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    let (m, s) = (mag.into_value(), sign.into_value());
    let negative = match T::DTYPE.bits() {
        16 => s.bitcast::<i16>().lt(0i16),
        32 => s.bitcast::<i32>().lt(0),
        _ => s.bitcast::<i64>().lt(0i64),
    };
    // Built from `abs` and `-`, which are bit operations in inline PTX. With
    // integer `and` and `or`, LLVM turns a constant sign into `abs.f32` or
    // `neg.f32`, whose NaNs are not the bit operation's.
    let a = m.abs();
    select(negative, -a, a)
}

/// Round half away from zero. fastmath does not change it.
#[track_caller]
pub fn round<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    let _at = LocationScope::new();
    let x = x.into_value();
    match T::DTYPE {
        DType::F32 => retype(misc::round::<f32>(retype(x))),
        DType::F64 => retype(misc::round::<f64>(retype(x))),
        _ => narrow(misc::round::<f32>(x.to())),
    }
}

/// Round half to even ([`Value::rint`], `cvt.rni`).
#[track_caller]
pub fn roundeven<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    x.into_value().rint()
}

/// Round toward negative infinity ([`Value::floor`]).
#[track_caller]
pub fn floor<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    x.into_value().floor()
}

/// Round toward positive infinity ([`Value::ceil`]).
#[track_caller]
pub fn ceil<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    x.into_value().ceil()
}

/// Round toward zero ([`Value::trunc`]).
#[track_caller]
pub fn trunc<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    x.into_value().trunc()
}

/// Absolute value of a float or a signed integer (an unsigned integer is
/// returned as it is). On a float it clears the sign bit (IEEE 754 `abs`,
/// [`Value::abs`]): a NaN keeps its payload.
#[track_caller]
pub fn abs<T: Arith>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    abs_with(x, MathOptions { fastmath, ..Default::default() })
}

/// CuTe DSL's historical name for [`abs`].
#[track_caller]
pub fn absf<T: Arith>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    abs(x, fastmath)
}

/// [`abs`] with CuTe DSL's `ftz=`: `abs.ftz.f32`, which flushes a subnormal
/// to +0 and gives a NaN the canonical NaN (an arithmetic instruction on
/// the GPU, unlike the bit operation of [`abs`]).
#[track_caller]
pub fn abs_with<T: Arith>(x: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let x = x.into_value();
    check_options("abs", &opts, Accepts { ftz: true, ..NONE }, T::DTYPE);
    if opts.ftz {
        return inst(FloatFn::Abs, &[x], FloatMode::Exact, true, false);
    }
    if T::DTYPE.is_int() && !T::DTYPE.is_signed() { x } else { x.abs() }
}

/// CuTe DSL's `absi`: absolute value of an integer (an unsigned integer is
/// returned as it is).
#[track_caller]
pub fn absi<T: Integer>(x: impl IntoValue<T>) -> Value<T> {
    let x = x.into_value();
    if T::DTYPE.is_signed() { x.abs() } else { x }
}

/// `-x`: the sign bit flipped (IEEE 754 `negate`), so a NaN keeps its
/// payload.
#[track_caller]
pub fn neg<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    -x.into_value()
}

/// The smaller value; a NaN operand is ignored (IEEE 754 `minimumNumber`),
/// and -0 is smaller than +0.
#[track_caller]
pub fn min<T: Arith>(x: impl IntoValue<T>, y: impl IntoValue<T>) -> Value<T> {
    min_with(x, y, MathOptions::default())
}

/// The larger value; a NaN operand is ignored (IEEE 754 `maximumNumber`),
/// and +0 is larger than -0.
#[track_caller]
pub fn max<T: Arith>(x: impl IntoValue<T>, y: impl IntoValue<T>) -> Value<T> {
    max_with(x, y, MathOptions::default())
}

/// [`min`] with CuTe DSL's `propagate_nan=` (IEEE 754 `minimum`: NaN if
/// either operand is NaN, and -0 < +0) and `ftz=` (`min.ftz.f32`).
#[track_caller]
pub fn min_with<T: Arith>(x: impl IntoValue<T>, y: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    min_max(x.into_value(), y.into_value(), opts, false)
}

/// [`max`] with CuTe DSL's `propagate_nan=` and `ftz=`.
#[track_caller]
pub fn max_with<T: Arith>(x: impl IntoValue<T>, y: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    min_max(x.into_value(), y.into_value(), opts, true)
}

#[track_caller]
fn min_max<T: Arith>(x: Value<T>, y: Value<T>, opts: MathOptions, is_max: bool) -> Value<T> {
    let _at = LocationScope::new();
    let name = if is_max { "max" } else { "min" };
    let accepts = Accepts { ftz: true, propagate_nan: true, fastmath: false, ..NONE };
    check_options(name, &opts, accepts, T::DTYPE);
    let func = if is_max { FloatFn::Max } else { FloatFn::Min };
    if T::DTYPE.is_float() && (opts.ftz || (opts.propagate_nan && T::DTYPE == DType::F32)) {
        return inst(func, &[x, y], FloatMode::Exact, opts.ftz, opts.propagate_nan);
    }
    if !(T::DTYPE.is_float() && opts.propagate_nan) {
        return if is_max { x.max(y) } else { x.min(y) };
    }
    // IEEE 754 minimum/maximum, which PTX has for f32 only.
    let either_nan = x.ne(x) | y.ne(y);
    let picked = select(if is_max { x.gt(y) } else { x.lt(y) }, x, y);
    // Equal operands: -0 and +0 order by their sign bits.
    let ordered = match T::DTYPE.bits() {
        16 => zeros_ordered::<T, u16>(x, y, is_max),
        32 => zeros_ordered::<T, u32>(x, y, is_max),
        _ => zeros_ordered::<T, u64>(x, y, is_max),
    };
    let r = select(x.eq(y), ordered, picked);
    select(either_nan, Value::constant(T::from_bits(nan_bits(T::DTYPE))), r)
}

/// For `x == y`: min is the OR of the bits (-0 if either is), max the AND.
#[track_caller]
fn zeros_ordered<T: Numeric, U: Integer + Bitwise>(x: Value<T>, y: Value<T>, is_max: bool) -> Value<T> {
    let (a, b) = (x.bitcast::<U>(), y.bitcast::<U>());
    (if is_max { a & b } else { a | b }).bitcast()
}

/// `x` clamped to `[lo, hi]`: `max(lo, min(x, hi))`, as CuTe DSL lowers it.
#[track_caller]
pub fn clamp<T: Arith>(x: impl IntoValue<T>, lo: impl IntoValue<T>, hi: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    clamp_with(x, lo, hi, MathOptions { fastmath, ..Default::default() })
}

/// [`clamp`] with CuTe DSL's `propagate_nan=` and `ftz=`.
#[track_caller]
pub fn clamp_with<T: Arith>(
    x: impl IntoValue<T>,
    lo: impl IntoValue<T>,
    hi: impl IntoValue<T>,
    opts: MathOptions,
) -> Value<T> {
    let _at = LocationScope::new();
    let (x, lo, hi) = (x.into_value(), lo.into_value(), hi.into_value());
    check_options("clamp", &opts, Accepts { ftz: true, propagate_nan: true, ..NONE }, T::DTYPE);
    let inner = MathOptions { fastmath: false, ..opts };
    min_max(lo, min_max(x, hi, inner, false), inner, true)
}

/// `a * b + c` with a single rounding.
#[track_caller]
pub fn fma<T: Float>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    fma_with(a, b, c, MathOptions { fastmath, ..Default::default() })
}

/// [`fma`] with CuTe DSL's `rounding=` and `ftz=`: `fma.{rnd}[.ftz]`.
#[track_caller]
pub fn fma_with<T: Float>(
    a: impl IntoValue<T>,
    b: impl IntoValue<T>,
    c: impl IntoValue<T>,
    opts: MathOptions,
) -> Value<T> {
    let _at = LocationScope::new();
    let (a, b, c) = (a.into_value(), b.into_value(), c.into_value());
    check_options("fma", &opts, Accepts { ftz: true, rounding: true, ..NONE }, T::DTYPE);
    match rounding(&opts) {
        Some(mode) => inst(FloatFn::Fma, &[a, b, c], mode, opts.ftz, false),
        None => crate::value::fma(a, b, c),
    }
}

/// The rounding of an operation given `rounding=` or `ftz=` (`ftz` alone
/// rounds to nearest, as in CuTe DSL).
fn rounding(opts: &MathOptions) -> Option<FloatMode> {
    (opts.rounding.is_some() || opts.ftz).then(|| FloatMode::Round(opts.rounding.unwrap_or(RoundingMode::NearestEven)))
}

/// `x + y`.
#[track_caller]
pub fn add<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    add_with(x, y, MathOptions { fastmath, ..Default::default() })
}

/// [`add`] with CuTe DSL's `rounding=` and `ftz=`: `add.{rnd}[.ftz]`.
#[track_caller]
pub fn add_with<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let (x, y) = (x.into_value(), y.into_value());
    check_options("add", &opts, Accepts { ftz: true, rounding: true, ..NONE }, T::DTYPE);
    match rounding(&opts) {
        Some(mode) => inst(FloatFn::Add, &[x, y], mode, opts.ftz, false),
        None => x + y,
    }
}

/// `x - y` (CuTe DSL's `sub` takes no rounding options either).
#[track_caller]
pub fn sub<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, _fastmath: bool) -> Value<T> {
    x.into_value() - y.into_value()
}

/// `x * y`.
#[track_caller]
pub fn mul<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    mul_with(x, y, MathOptions { fastmath, ..Default::default() })
}

/// [`mul`] with CuTe DSL's `rounding=` and `ftz=`: `mul.{rnd}[.ftz]`.
#[track_caller]
pub fn mul_with<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let (x, y) = (x.into_value(), y.into_value());
    check_options("mul", &opts, Accepts { ftz: true, rounding: true, ..NONE }, T::DTYPE);
    match rounding(&opts) {
        Some(mode) => inst(FloatFn::Mul, &[x, y], mode, opts.ftz, false),
        None => x * y,
    }
}

/// `x / y`, correctly rounded. Fast (Float32): `div.approx.ftz.f32`, CUDA's
/// `__fdividef`: within 2 ulp for `|y|` in `[2^-126, 2^126]`; for larger
/// `|y|` the result is zero (NaN for an infinite `x`).
#[track_caller]
pub fn div<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    div_with(x, y, MathOptions { fastmath, ..Default::default() })
}

/// [`div`] with CuTe DSL's `approx=` (`div.approx[.ftz].f32`), `full=`
/// (`div.full[.ftz].f32`, 2 ulp over the full range), `rounding=` and
/// `ftz=` (`div.{rnd}[.ftz]`).
#[track_caller]
pub fn div_with<T: Float>(x: impl IntoValue<T>, y: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let (x, y) = (x.into_value(), y.into_value());
    check_options("div", &opts, Accepts { approx: true, ftz: true, rounding: true, full: true, ..NONE }, T::DTYPE);
    if opts.approx || opts.full {
        let mode = if opts.approx { FloatMode::Approx } else { FloatMode::Full };
        return inst(FloatFn::Div, &[x, y], mode, opts.ftz, false);
    }
    if let Some(mode) = rounding(&opts) {
        return inst(FloatFn::Div, &[x, y], mode, opts.ftz, false);
    }
    binary(x, y, opts.fastmath, |x, y| x / y, |x, y| x / y, Some(fast::div))
}

/// `1 / x`, correctly rounded (the same bits as CuTe DSL's default
/// `rcp.rn`). Fast (Float32): `rcp.approx.ftz.f32`, within 1 ulp (PTX ISA).
#[track_caller]
pub fn rcp<T: Float>(x: impl IntoValue<T>, fastmath: bool) -> Value<T> {
    rcp_with(x, MathOptions { fastmath, ..Default::default() })
}

/// [`rcp`] with CuTe DSL's `approx=` (`rcp.approx[.ftz].f32`;
/// `rcp.approx.ftz.f64` needs `ftz`), `rounding=` and `ftz=`
/// (`rcp.{rnd}[.ftz]`).
#[track_caller]
pub fn rcp_with<T: Float>(x: impl IntoValue<T>, opts: MathOptions) -> Value<T> {
    let _at = LocationScope::new();
    let x = x.into_value();
    check_options("rcp", &opts, Accepts { approx: true, ftz: true, rounding: true, ..NONE }, T::DTYPE);
    if opts.approx {
        return inst(FloatFn::Rcp, &[x], FloatMode::Approx, opts.ftz, false);
    }
    if let Some(mode) = rounding(&opts) {
        return inst(FloatFn::Rcp, &[x], mode, opts.ftz, false);
    }
    unary(x, opts.fastmath, |x| 1.0f32 / x, |x| 1.0f64 / x, Some(fast::rcp))
}

/// Infinity and the smallest normal magnitude of `T`, as bits.
fn limits(t: DType) -> (u64, u64) {
    match t {
        DType::F16 => (0x7c00, 0x0400),
        DType::BF16 => (0x7f80, 0x0080),
        DType::F32 => (f32::INFINITY.to_bits() as u64, f32::MIN_POSITIVE.to_bits() as u64),
        _ => (f64::INFINITY.to_bits(), f64::MIN_POSITIVE.to_bits()),
    }
}

/// Whether `x` is NaN.
#[track_caller]
pub fn isnan<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Boolean {
    let x = x.into_value();
    x.ne(x)
}

/// Whether `x` is infinite.
#[track_caller]
pub fn isinf<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Boolean {
    x.into_value().abs().eq(T::from_bits(limits(T::DTYPE).0))
}

/// Whether `x` is finite (neither infinite nor NaN).
#[track_caller]
pub fn isfinite<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Boolean {
    x.into_value().abs().lt(T::from_bits(limits(T::DTYPE).0))
}

/// Whether `x` is normal: not zero, subnormal, infinite or NaN.
#[track_caller]
pub fn isnormal<T: Float>(x: impl IntoValue<T>, _fastmath: bool) -> Boolean {
    let (inf, min) = limits(T::DTYPE);
    let a = x.into_value().abs();
    a.lt(T::from_bits(inf)) & a.ge(T::from_bits(min))
}

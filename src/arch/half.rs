//! CuTe DSL's packed and scalar half-precision operations
//! (`cute.arch.add_packed_f16x2`, `fma_bf16`, `exp2_packed_f16x2`, ...):
//! one PTX instruction each, on Float16 or BFloat16 values or on pairs of
//! them packed into 32 bits as `.f16x2` and `.bf16x2` registers hold them.

use std::panic::Location;

use crate::builder::{self, LocationScope};
use crate::ir::{DType, HalfFn, HalfModifiers, HalfType, ValueId};
use crate::value::{BF16, BFloat16, Boolean, F16, Float, Float16, Int32, IntoValue, Value, pack, unpack};

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::value::F16 {}
    impl Sealed for crate::value::BF16 {}
}

/// Float16 and BFloat16: the element types of the packed half-precision
/// operations.
pub trait HalfFloat: Float + sealed::Sealed {
    /// The packed PTX type (`.f16x2` or `.bf16x2`).
    #[doc(hidden)]
    const PACKED: HalfType;
    /// The scalar PTX type (`.f16` or `.bf16`).
    #[doc(hidden)]
    const SCALAR: HalfType;
}

impl HalfFloat for F16 {
    const PACKED: HalfType = HalfType::F16x2;
    const SCALAR: HalfType = HalfType::F16;
}

impl HalfFloat for BF16 {
    const PACKED: HalfType = HalfType::Bf16x2;
    const SCALAR: HalfType = HalfType::Bf16;
}

/// A pair of Float16 or BFloat16 values in either of the forms CuTe DSL's
/// packed operations take: an [`Int32`] holding the two values' bits, element
/// 0 in the low half (a `.f16x2` or `.bf16x2` register), or a 2-tuple of
/// values. Each operation returns the form it is given, and every operand of
/// one call has the same form.
pub trait Half2<T: HalfFloat>: Copy {
    /// The pair as packed bits.
    fn to_packed(self) -> Int32;
    /// The pair of this form holding `bits`.
    fn from_packed(bits: Int32) -> Self;
}

impl<T: HalfFloat> Half2<T> for Int32 {
    fn to_packed(self) -> Int32 {
        self
    }

    fn from_packed(bits: Int32) -> Self {
        bits
    }
}

impl<T: HalfFloat> Half2<T> for (Value<T>, Value<T>) {
    #[track_caller]
    fn to_packed(self) -> Int32 {
        pack([self.0, self.1])
    }

    #[track_caller]
    fn from_packed(bits: Int32) -> Self {
        let [lo, hi] = unpack(bits);
        (lo, hi)
    }
}

/// The keyword options of CuTe DSL's packed and scalar half-precision
/// operations: `rnd='rn'`, `ftz=`, `sat=`, `relu=`, `oob=` and `predicate=`.
/// Each operation takes the ones its CuTe DSL counterpart has, as CuTe DSL
/// checks them (the kernel build panics otherwise):
///
/// * add and sub take `rn`, `ftz`, `sat` and `predicate`;
/// * fma takes `ftz`, `sat`, `relu` and `oob`, and always rounds to nearest
///   (`.rn`; `rn` changes nothing);
/// * `ftz` and `sat` apply to Float16 only, `sat` and `relu` exclude each
///   other, and `oob` excludes `ftz` and `sat`.
///
/// `HalfOptions::default()` asks for none of them: `HalfOptions { sat: true,
/// ..HalfOptions::default() }` is `sat=True`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfOptions {
    /// `rnd='rn'`: an explicit `.rn` on add and sub. They round to nearest
    /// even without it too, but PTX then lets `ptxas` fuse them with a
    /// multiplication into an fma.
    pub rn: bool,
    /// `ftz=True`: flush subnormal inputs and results to zeros of their
    /// signs (`.ftz`, Float16 only).
    pub ftz: bool,
    /// `sat=True`: clamp the result to `[0, 1]`; a NaN or `-0` gives `+0`
    /// (`.sat`, Float16 only).
    pub sat: bool,
    /// `relu=True`: a negative result (`-0` too) becomes `+0` and a NaN
    /// result the canonical NaN (`.relu`, fma only).
    pub relu: bool,
    /// `oob=True`: the result is `+0` where a multiplicand is TMA's
    /// out-of-bounds NaN (`.oob`, fma only; sm_90 and PTX 8.1). The PTX ISA
    /// does not give that NaN's encoding, and the simulator takes every NaN
    /// multiplicand for it, as CUTLASS's host reference does: a GPU may
    /// return NaN for other NaNs. No available GPU has run it.
    pub oob: bool,
    /// `predicate=p`: the instruction runs only where `p` is true (`@p`);
    /// elsewhere the result is the first operand, `a` (CuTe DSL leaves it
    /// undefined). Add and sub only.
    pub predicate: Option<Boolean>,
}

/// The PTX modifiers for `opts` on the CuTe DSL operation `name`, after
/// CuTe DSL's checks (in its order) and those of the keywords the operation
/// has.
#[track_caller]
fn modifiers(name: &str, func: HalfFn, elem: DType, opts: HalfOptions) -> HalfModifiers {
    let HalfOptions { rn, ftz, sat, relu, oob, predicate } = opts;
    let op = func.ptx();
    if func == HalfFn::Fma {
        assert!(predicate.is_none(), "cute: {name} takes no predicate (CuTe DSL's fma operations have none)");
        assert!(!(sat && relu), "cute: {name}: sat and relu exclude each other, but both are set");
    } else {
        assert!(!relu, "cute: {name} takes no relu option (CuTe DSL's {op} has none; fma has relu)");
        assert!(!oob, "cute: {name} takes no oob option (CuTe DSL's {op} has none; fma has oob)");
    }
    if elem == DType::BF16 {
        assert!(!ftz, "cute: {name}: ftz applies to Float16 only, but it is set on a BFloat16 {op}");
        assert!(!sat, "cute: {name}: sat applies to Float16 only, but it is set on a BFloat16 {op}");
    }
    assert!(
        !(oob && (ftz || sat)),
        "cute: {name}: oob excludes ftz and sat, but oob is set with {}",
        if ftz && sat {
            "ftz and sat"
        } else if ftz {
            "ftz"
        } else {
            "sat"
        }
    );
    HalfModifiers { rn: rn || func == HalfFn::Fma, oob, ftz, sat, relu }
}

/// One half-precision instruction on Int32-packed pairs.
#[track_caller]
fn packed_inst(func: HalfFn, ty: HalfType, args: Vec<ValueId>, mods: HalfModifiers, pred: Option<Boolean>) -> Int32 {
    let loc = Location::caller();
    let predicate = pred.map(|p| p.checked());
    Value::from_id(builder::with(|b| b.half_inst(func, ty, args, mods, predicate, loc)))
}

/// A packed operation of CuTe DSL's `name` on `operands`, in their form.
#[track_caller]
fn packed<T: HalfFloat, P: Half2<T>>(name: &str, func: HalfFn, operands: &[P], opts: HalfOptions) -> P {
    let mods = modifiers(name, func, T::DTYPE, opts);
    let _at = LocationScope::new();
    let args = operands.iter().map(|p| p.to_packed().checked()).collect();
    P::from_packed(packed_inst(func, T::PACKED, args, mods, opts.predicate))
}

/// A scalar fma of CuTe DSL's `name`.
#[track_caller]
fn scalar_fma<T: HalfFloat>(
    name: &str,
    a: impl IntoValue<T>,
    b: impl IntoValue<T>,
    c: impl IntoValue<T>,
    opts: HalfOptions,
) -> Value<T> {
    let mods = modifiers(name, HalfFn::Fma, T::DTYPE, opts);
    let loc = Location::caller();
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    Value::from_id(builder::with(|bs| bs.half_inst(HalfFn::Fma, T::SCALAR, args, mods, None, loc)))
}

/// `a + b` on two pairs of Float16 values (CuTe DSL's
/// `cute.arch.add_packed_f16x2(a, b, rnd=, ftz=, sat=, predicate=)`, PTX
/// `add{.rn}{.ftz}{.sat}.f16x2`), each element rounded to nearest even.
/// The pairs are Int32s holding their bits, giving an Int32, or 2-tuples,
/// giving a 2-tuple ([`Half2`]); [`HalfOptions`] has the options.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("half_sums", || {
///     let x = param_tensor::<i32>("x", Layout::col_major(64));
///     let t = thread_idx().x;
///     let opts = HalfOptions { sat: true, ..HalfOptions::default() };
///     x.store(t, add_packed_f16x2(x.load(t), x.load(t + 32), opts));
/// });
/// ```
#[track_caller]
pub fn add_packed_f16x2<P: Half2<F16>>(a: P, b: P, opts: HalfOptions) -> P {
    packed("add_packed_f16x2", HalfFn::Add, &[a, b], opts)
}

/// `a + b` on two pairs of BFloat16 values (CuTe DSL's
/// `cute.arch.add_packed_bf16x2(a, b, rnd=, predicate=)`, PTX
/// `add{.rn}.bf16x2`, sm_90 and PTX 7.8).
#[track_caller]
pub fn add_packed_bf16x2<P: Half2<BF16>>(a: P, b: P, opts: HalfOptions) -> P {
    packed("add_packed_bf16x2", HalfFn::Add, &[a, b], opts)
}

/// `a - b` on two pairs of Float16 values (CuTe DSL's
/// `cute.arch.sub_packed_f16x2(a, b, rnd=, ftz=, sat=, predicate=)`, PTX
/// `sub{.rn}{.ftz}{.sat}.f16x2`).
#[track_caller]
pub fn sub_packed_f16x2<P: Half2<F16>>(a: P, b: P, opts: HalfOptions) -> P {
    packed("sub_packed_f16x2", HalfFn::Sub, &[a, b], opts)
}

/// `a - b` on two pairs of BFloat16 values (CuTe DSL's
/// `cute.arch.sub_packed_bf16x2(a, b, rnd=, predicate=)`, PTX
/// `sub{.rn}.bf16x2`, sm_90 and PTX 7.8).
#[track_caller]
pub fn sub_packed_bf16x2<P: Half2<BF16>>(a: P, b: P, opts: HalfOptions) -> P {
    packed("sub_packed_bf16x2", HalfFn::Sub, &[a, b], opts)
}

/// `a * b + c` on pairs of Float16 values, each element rounded once (CuTe
/// DSL's `cute.arch.fma_packed_f16x2(a, b, c, rnd='rn', ftz=, sat=, relu=,
/// oob=)`, PTX `fma.rn{.ftz}{.sat}.f16x2`, `fma.rn{.ftz}.relu.f16x2` (sm_80)
/// or `fma.rn.oob{.relu}.f16x2` (sm_90, PTX 8.1)).
#[track_caller]
pub fn fma_packed_f16x2<P: Half2<F16>>(a: P, b: P, c: P, opts: HalfOptions) -> P {
    packed("fma_packed_f16x2", HalfFn::Fma, &[a, b, c], opts)
}

/// `a * b + c` on pairs of BFloat16 values, each element rounded once (CuTe
/// DSL's `cute.arch.fma_packed_bf16x2(a, b, c, rnd='rn', relu=, oob=)`, PTX
/// `fma.rn{.relu}.bf16x2` (sm_80) or `fma.rn.oob{.relu}.bf16x2` (sm_90, PTX
/// 8.1)).
#[track_caller]
pub fn fma_packed_bf16x2<P: Half2<BF16>>(a: P, b: P, c: P, opts: HalfOptions) -> P {
    packed("fma_packed_bf16x2", HalfFn::Fma, &[a, b, c], opts)
}

/// `a * b + c` on Float16 values, rounded once (CuTe DSL's
/// `cute.arch.fma_f16(a, b, c, rnd='rn', ftz=, sat=, relu=, oob=)`, PTX
/// `fma.rn{.ftz}{.sat}.f16` and the forms of [`fma_packed_f16x2`]). Without
/// options it computes what [`fma`](crate::fma) does.
#[track_caller]
pub fn fma_f16(a: impl IntoValue<F16>, b: impl IntoValue<F16>, c: impl IntoValue<F16>, opts: HalfOptions) -> Float16 {
    scalar_fma("fma_f16", a, b, c, opts)
}

/// `a * b + c` on BFloat16 values, rounded once (CuTe DSL's
/// `cute.arch.fma_bf16(a, b, c, rnd='rn', relu=, oob=)`, PTX
/// `fma.rn{.relu}.bf16` (sm_80) or `fma.rn.oob{.relu}.bf16` (sm_90)).
/// Without options it computes what [`fma`](crate::fma) does.
#[track_caller]
pub fn fma_bf16(
    a: impl IntoValue<BF16>,
    b: impl IntoValue<BF16>,
    c: impl IntoValue<BF16>,
    opts: HalfOptions,
) -> BFloat16 {
    scalar_fma("fma_bf16", a, b, c, opts)
}

/// An approximation of `2^a` on a pair of Float16 values (CuTe DSL's
/// `cute.arch.exp2_packed_f16x2(a, predicate=)`, PTX `ex2.approx.f16x2`,
/// sm_75): one of the two Float16 values on either side of the exact `2^a`
/// (`2^a` itself when it is a Float16), so within 1 ulp (measured on an SM89
/// GPU for every input; the PTX ISA's bound is 2^-9.9 relative). Subnormal
/// inputs and results are kept.
/// The simulator gives the correctly rounded value. With a predicate, the
/// instruction runs only where it is true, and elsewhere the result is `a`.
#[track_caller]
pub fn exp2_packed_f16x2<P: Half2<F16>>(a: P, predicate: Option<Boolean>) -> P {
    let opts = HalfOptions { predicate, ..HalfOptions::default() };
    packed("exp2_packed_f16x2", HalfFn::Ex2, &[a], opts)
}

/// An approximation of `2^a` on a pair of BFloat16 values (CuTe DSL's
/// `cute.arch.exp2_packed_bf16x2(a, predicate=)`, PTX
/// `ex2.approx.ftz.bf16x2`, sm_90 and PTX 7.8), which flushes subnormal
/// inputs and results to zero; the PTX ISA's bound is 2^-7 relative. The
/// simulator gives the correctly rounded value (flushed).
#[track_caller]
pub fn exp2_packed_bf16x2<P: Half2<BF16>>(a: P, predicate: Option<Boolean>) -> P {
    let opts = HalfOptions { predicate, ..HalfOptions::default() };
    packed("exp2_packed_bf16x2", HalfFn::Ex2, &[a], opts)
}

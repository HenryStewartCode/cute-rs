//! Packed `.f32x2` arithmetic and the mixed-precision packed forms: CuTe DSL's `cute.arch.add_packed_f32x2`,
//! `sub_packed_f32x2`, `mul_packed_f32x2`, `fma_packed_f32x2`,
//! `exp_packed_f32x2` and the ten Rubin-only mixed-precision functions
//! (`add_packed_f16x2_f32x2_f32x2`, ...). A pair is a 2-tuple; the
//! instruction takes it as one 64-bit register (a `.f32x2`) or, for a half
//! pair, one 32-bit register. Each element is an ordinary operation with the
//! given rounding and `ftz`.

use crate::builder::{self, LocationScope};
use crate::ir::{CvtFn, PackedOp, RoundingMode, ValueId};
use crate::value::{BF16, F16, Float32, Int32, Value, pack, unpack};

use super::half::{Half2, HalfFloat};

/// The keyword options of the packed `.f32x2` operations (CuTe DSL's
/// `rnd='rn'` and `ftz=`): `rnd` is a rounding direction or `None` (CuTe
/// DSL's `rnd='none'`, no `.rnd`, which lets `ptxas` fuse an add or multiply
/// with a neighbour; fma needs one), and `ftz` flushes subnormal inputs and
/// results to zeros of their signs. `PackedF32Options::default()` is
/// `rnd='rn'` without `ftz`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PackedF32Options {
    pub rnd: Option<RoundingMode>,
    pub ftz: bool,
}

impl Default for PackedF32Options {
    fn default() -> Self {
        PackedF32Options { rnd: Some(RoundingMode::NearestEven), ftz: false }
    }
}

impl PackedF32Options {
    /// The options with rounding direction `rnd`.
    pub fn rounding(rnd: RoundingMode) -> Self {
        PackedF32Options { rnd: Some(rnd), ftz: false }
    }

    /// The same options with `ftz`.
    pub fn with_ftz(self, ftz: bool) -> Self {
        PackedF32Options { ftz, ..self }
    }

    /// The options with no `.rnd` modifier.
    pub fn no_rounding() -> Self {
        PackedF32Options { rnd: None, ftz: false }
    }
}

/// A pair of Float32 values, as the packed instructions take them.
pub type F32x2 = (Float32, Float32);

fn f32x2_bits(p: F32x2) -> ValueId {
    pack::<f32, u64, 2>([p.0, p.1]).checked()
}

fn f32x2_from(id: ValueId) -> F32x2 {
    let [lo, hi]: [Float32; 2] = unpack(Value::<u64>::from_id(id));
    (lo, hi)
}

#[track_caller]
fn issue(func: CvtFn, args: Vec<ValueId>) -> ValueId {
    let loc = std::panic::Location::caller();
    builder::with(|b| b.cvt(func, args, loc))
}

#[track_caller]
fn f32x2_op(name: &str, op: PackedOp, operands: &[F32x2], opts: PackedF32Options) -> F32x2 {
    assert!(
        op != PackedOp::Fma || opts.rnd.is_some(),
        "cute: {name}: fma needs a rounding mode (rnd='none' is for add, sub and mul)"
    );
    let _at = LocationScope::new();
    let args = operands.iter().map(|&p| f32x2_bits(p)).collect();
    f32x2_from(issue(CvtFn::Packed { op, rnd: opts.rnd, ftz: opts.ftz }, args))
}

/// `a + b` on two Float32 pairs (CuTe DSL's `cute.arch.add_packed_f32x2(a,
/// b, rnd='rn', ftz=None)`, PTX `add{.rnd}{.ftz}.f32x2`; sm_100 and PTX 8.6,
/// `ptxas` assembles it). Each element is an `add.f32` of the same rounding
/// and `ftz`.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("pair_sums", || {
///     let x = param_tensor::<f32>("x", Layout::col_major(256));
///     let t = thread_idx().x * 2;
///     let (s0, s1) = add_packed_f32x2((x.load(t), x.load(t + 1)), (x.load(t + 128), x.load(t + 129)), PackedF32Options::default());
///     x.store(t, s0);
///     x.store(t + 1, s1);
/// });
/// ```
#[track_caller]
pub fn add_packed_f32x2(a: F32x2, b: F32x2, opts: PackedF32Options) -> F32x2 {
    f32x2_op("add_packed_f32x2", PackedOp::Add, &[a, b], opts)
}

/// `a - b` on two Float32 pairs (`cute.arch.sub_packed_f32x2`, PTX
/// `sub{.rnd}{.ftz}.f32x2`).
#[track_caller]
pub fn sub_packed_f32x2(a: F32x2, b: F32x2, opts: PackedF32Options) -> F32x2 {
    f32x2_op("sub_packed_f32x2", PackedOp::Sub, &[a, b], opts)
}

/// `a * b` on two Float32 pairs (`cute.arch.mul_packed_f32x2`, PTX
/// `mul{.rnd}{.ftz}.f32x2`).
#[track_caller]
pub fn mul_packed_f32x2(a: F32x2, b: F32x2, opts: PackedF32Options) -> F32x2 {
    f32x2_op("mul_packed_f32x2", PackedOp::Mul, &[a, b], opts)
}

/// `a * b + c` on Float32 pairs, each element rounded once
/// (`cute.arch.fma_packed_f32x2`, PTX `fma.rnd{.ftz}.f32x2`). `opts.rnd`
/// must be a direction.
#[track_caller]
pub fn fma_packed_f32x2(a: F32x2, b: F32x2, c: F32x2, opts: PackedF32Options) -> F32x2 {
    f32x2_op("fma_packed_f32x2", PackedOp::Fma, &[a, b, c], opts)
}

/// `e^a` on a pair, as CuTe DSL's deprecated `cute.arch.exp_packed_f32x2`
/// computes it: [`mul_packed_f32x2`] by `log2(e)` (rounded to nearest), then
/// two `ex2.approx.ftz.f32`.
#[deprecated(note = "use mul_packed_f32x2 and math::exp2 with the fast approximation instead")]
#[track_caller]
pub fn exp_packed_f32x2(a: F32x2) -> F32x2 {
    let log2_e = Float32::from(std::f32::consts::LOG2_E);
    let (x, y) = mul_packed_f32x2(a, (log2_e, log2_e), PackedF32Options::default());
    (crate::math::exp2(x, true), crate::math::exp2(y, true))
}

fn half_pair<T: HalfFloat, P: Half2<T>>(p: P) -> ValueId {
    p.to_packed().checked()
}

fn half_unpack<T: HalfFloat>(id: ValueId) -> (Value<T>, Value<T>) {
    let [lo, hi]: [Value<T>; 2] = unpack(Int32::from_id(id));
    (lo, hi)
}

#[track_caller]
fn to_half<T: HalfFloat>(op: PackedOp, a: F32x2, b: F32x2) -> (Value<T>, Value<T>) {
    let _at = LocationScope::new();
    let func = CvtFn::Packed { op, rnd: Some(RoundingMode::NearestEven), ftz: false };
    half_unpack(issue(func, vec![f32x2_bits(a), f32x2_bits(b)]))
}

#[track_caller]
fn from_half<T: HalfFloat, P: Half2<T>>(op: PackedOp, a: P, rest: &[F32x2], rnd: Option<RoundingMode>) -> F32x2 {
    let _at = LocationScope::new();
    let mut args = vec![half_pair::<T, P>(a)];
    args.extend(rest.iter().map(|&p| f32x2_bits(p)));
    f32x2_from(issue(CvtFn::Packed { op, rnd, ftz: false }, args))
}

macro_rules! mixed_to_half {
    ($($(#[$doc:meta])* $name:ident, $op:ident, $ty:ty, $bf16:expr;)*) => {$(
        $(#[$doc])*
        #[track_caller]
        pub fn $name(src_a: F32x2, src_b: F32x2) -> ($crate::value::Value<$ty>, $crate::value::Value<$ty>) {
            to_half::<$ty>(PackedOp::$op { bf16: $bf16 }, src_a, src_b)
        }
    )*};
}

mixed_to_half! {
    /// Two Float32 sums rounded to Float16 (`cute.arch.add_packed_f16x2_f32x2_f32x2`,
    /// PTX `add.rn.f16x2.f32x2`; Rubin only: the sm_107f family from PTX ISA
    /// 9.4, which no installed `ptxas` or LLVM has, so it is checked in the
    /// simulator and as MLIR text). Each sum is rounded once, to nearest even.
    add_packed_f16x2_f32x2_f32x2, AddToHalf, F16, false;
    /// As [`add_packed_f16x2_f32x2_f32x2`], to BFloat16 (`add.rn.bf16x2.f32x2`).
    add_packed_bf16x2_f32x2_f32x2, AddToHalf, BF16, true;
    /// `src_a - src_b` rounded to Float16 (`sub.rn.f16x2.f32x2`).
    sub_packed_f16x2_f32x2_f32x2, SubToHalf, F16, false;
    /// `src_a - src_b` rounded to BFloat16 (`sub.rn.bf16x2.f32x2`).
    sub_packed_bf16x2_f32x2_f32x2, SubToHalf, BF16, true;
}

macro_rules! mixed_from_half {
    ($($(#[$doc:meta])* $name:ident, $op:ident, $ty:ty, $bf16:expr;)*) => {$(
        $(#[$doc])*
        #[track_caller]
        pub fn $name<P: Half2<$ty>>(src_a: P, src_b: F32x2, rnd: Option<RoundingMode>) -> F32x2 {
            from_half::<$ty, P>(PackedOp::$op { bf16: $bf16 }, src_a, &[src_b], rnd)
        }
    )*};
}

mixed_from_half! {
    /// A Float16 pair (widened exactly) plus a Float32 pair, rounded once
    /// with `rnd` (`cute.arch.add_packed_f32x2_f16x2_f32x2`, PTX
    /// `add{.rnd}.f32x2.f16x2`; Rubin only, see
    /// [`add_packed_f16x2_f32x2_f32x2`]). CuTe DSL's default is `rnd='rn'`.
    add_packed_f32x2_f16x2_f32x2, AddHalf, F16, false;
    /// As [`add_packed_f32x2_f16x2_f32x2`] with a BFloat16 pair.
    add_packed_f32x2_bf16x2_f32x2, AddHalf, BF16, true;
    /// The Float16 pair minus the Float32 pair, rounded once with `rnd`
    /// (`sub{.rnd}.f32x2.f16x2`).
    sub_packed_f32x2_f16x2_f32x2, SubHalf, F16, false;
    /// As [`sub_packed_f32x2_f16x2_f32x2`] with a BFloat16 pair.
    sub_packed_f32x2_bf16x2_f32x2, SubHalf, BF16, true;
}

/// `src_a * src_b + src_c` with a Float16 pair `src_a` (widened exactly) and
/// Float32 pairs, rounded once with `rnd`, which must be a direction
/// (`cute.arch.fma_packed_f32x2_f16x2_f32x2_f32x2`, PTX `fma.rnd.f32x2.f16x2`;
/// Rubin only, see [`add_packed_f16x2_f32x2_f32x2`]).
#[track_caller]
pub fn fma_packed_f32x2_f16x2_f32x2_f32x2<P: Half2<F16>>(
    src_a: P,
    src_b: F32x2,
    src_c: F32x2,
    rnd: Option<RoundingMode>,
) -> F32x2 {
    assert!(rnd.is_some(), "cute: fma_packed_f32x2_f16x2_f32x2_f32x2: fma needs a rounding mode");
    from_half::<F16, P>(PackedOp::FmaHalf { bf16: false }, src_a, &[src_b, src_c], rnd)
}

/// As [`fma_packed_f32x2_f16x2_f32x2_f32x2`] with a BFloat16 pair.
#[track_caller]
pub fn fma_packed_f32x2_bf16x2_f32x2_f32x2<P: Half2<BF16>>(
    src_a: P,
    src_b: F32x2,
    src_c: F32x2,
    rnd: Option<RoundingMode>,
) -> F32x2 {
    assert!(rnd.is_some(), "cute: fma_packed_f32x2_bf16x2_f32x2_f32x2: fma needs a rounding mode");
    from_half::<BF16, P>(PackedOp::FmaHalf { bf16: true }, src_a, &[src_b, src_c], rnd)
}

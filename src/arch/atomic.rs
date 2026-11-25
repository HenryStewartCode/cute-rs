//! Atomics and reductions (CuTe DSL's `cute.arch.atomic_add` and friends,
//! `atomic_cas`, `atomic_fmax`, `atomic_fmin`, `atomic_max_float32` and
//! `red`): one PTX `atom` or `red` each, with explicit memory semantics and
//! scope.
//!
//! Every atomic takes CuTe DSL's `sem=` and `scope=` through
//! [`AtomicOptions`]; the plain forms (`atomic_add(ptr, v)`) use the
//! defaults, PTX's own: relaxed at GPU scope. The backend writes the PTX
//! instruction itself (`atom.relaxed.gpu.global.add.u32`), so the memory
//! semantics do not depend on the LLVM version.

use std::panic::Location;

use super::half::{Half2, HalfFloat};
use crate::builder::{self, LocationScope};
use crate::control::if_else_yield;
use crate::ir::{AtomicKind, AtomicType, DType, MemoryOrder, MemoryScope, Op, ValueId};
use crate::tensor::Ptr;
use crate::value::{BF16, F16, Float32, Int32, Integer, IntoValue, Numeric, Value};

/// CuTe DSL's `sem=` and `scope=` of an atomic or a reduction. `None` is
/// PTX's default: relaxed (`sem`) at GPU scope (`scope`).
///
/// * `sem`: `Relaxed`, `Acquire`, `Release` or `AcqRel` for the atomics,
///   `Relaxed` or `Release` for [`red`]. A release atomic publishes what its
///   warp has seen to an acquire (an acquire load or atomic, or a strong
///   read followed by a fence) that reads what it wrote, or what a later
///   atomic on the location wrote (its release sequence); an acquire atomic
///   is ordered before its thread's later accesses.
/// * `scope`: `Cta`, `Cluster` (sm_90 and PTX 7.8), `Gpu` or `Sys`: the
///   threads the ordering is for. Two atomics on one location are atomic
///   with respect to each other only if each one's scope includes the other
///   thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtomicOptions {
    pub sem: Option<MemoryOrder>,
    pub scope: Option<MemoryScope>,
}

impl AtomicOptions {
    /// The options with `sem` and `scope` set.
    pub fn new(sem: MemoryOrder, scope: MemoryScope) -> AtomicOptions {
        AtomicOptions { sem: Some(sem), scope: Some(scope) }
    }

    fn resolve(self) -> (MemoryOrder, MemoryScope) {
        (self.sem.unwrap_or(MemoryOrder::Relaxed), self.scope.unwrap_or(MemoryScope::Gpu))
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for crate::value::F16 {}
    impl Sealed for crate::value::BF16 {}
}

/// The types [`atomic_add`] takes, as CuTe DSL's does: 32- and 64-bit
/// integers (`atom.add.s32`, `.u32`, `.u64`), Float32 (`atom.add.f32`),
/// Float64 (`.f64`), and Float16 and BFloat16 (`atom.add.noftz.f16`,
/// `.bf16`; BFloat16 needs sm_90).
pub trait AtomicAddable: Numeric + sealed::Sealed {}

impl AtomicAddable for i32 {}
impl AtomicAddable for u32 {}
impl AtomicAddable for i64 {}
impl AtomicAddable for u64 {}
impl AtomicAddable for f32 {}
impl AtomicAddable for f64 {}
impl AtomicAddable for F16 {}
impl AtomicAddable for BF16 {}

/// Push an atomic (`result` of type `result_ty`) or, without one, a
/// reduction.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn push<T: Numeric>(
    kind: AtomicKind,
    ty: AtomicType,
    ptr: &Ptr<T>,
    value: ValueId,
    compare: Option<ValueId>,
    result_ty: Option<DType>,
    opts: AtomicOptions,
) -> Option<ValueId> {
    let loc = Location::caller();
    let (buffer, offset, peer) = ptr.parts_dsmem();
    let index = crate::tensor::index_operand(offset);
    let (sem, scope) = opts.resolve();
    builder::with(|b| match result_ty {
        Some(t) => {
            let result = b.new_value(t);
            b.push(Op::Atomic { result, kind, ty, buffer, index, value, compare, sem, scope, peer }, loc);
            Some(result)
        }
        None => {
            b.push(Op::Red { kind, ty, buffer, index, value, sem, scope, peer }, loc);
            None
        }
    })
}

#[track_caller]
fn atomic<T: Numeric>(
    kind: AtomicKind,
    ptr: &Ptr<T>,
    value: Value<T>,
    compare: Option<Value<T>>,
    opts: AtomicOptions,
) -> Value<T> {
    let ty = AtomicType::for_value(kind, T::DTYPE);
    let (value, compare) = (value.checked(), compare.map(|c| c.checked()));
    Value::from_id(push(kind, ty, ptr, value, compare, Some(T::DTYPE), opts).expect("an atomic has a result"))
}

#[track_caller]
fn integer_atomic<T: Integer>(
    kind: AtomicKind,
    ptr: &Ptr<T>,
    value: Value<T>,
    compare: Option<Value<T>>,
    opts: AtomicOptions,
) -> Value<T> {
    assert!(T::DTYPE.bits() >= 32, "cute: atomics are 32 or 64 bits wide, not {}", T::DTYPE);
    atomic(kind, ptr, value, compare, opts)
}

/// Atomically add `value`, returning the old contents: CuTe DSL's
/// `atomic_add(ptr, val)`, relaxed at GPU scope. Integers wrap; floats are
/// added rounding to nearest even. A Float32 add on global memory flushes
/// subnormal inputs and results to zeros of their signs, on shared memory
/// it keeps them (PTX's `atom.add.f32`); the Float16 and BFloat16 adds keep
/// them (`.noftz`), and Float64 is plain IEEE arithmetic.
#[track_caller]
pub fn atomic_add<T: AtomicAddable>(ptr: &Ptr<T>, value: impl IntoValue<T>) -> Value<T> {
    atomic(AtomicKind::Add, ptr, value.into_value(), None, AtomicOptions::default())
}

/// [`atomic_add`] with CuTe DSL's `sem=` and `scope=`.
#[track_caller]
pub fn atomic_add_with<T: AtomicAddable>(ptr: &Ptr<T>, value: impl IntoValue<T>, opts: AtomicOptions) -> Value<T> {
    atomic(AtomicKind::Add, ptr, value.into_value(), None, opts)
}

macro_rules! integer_atomics {
    ($($name:ident, $with:ident, $kind:ident, $doc:literal);* $(;)?) => {$(
        #[doc = $doc]
        #[doc = ""]
        #[doc = "Relaxed at GPU scope; the `_with` form takes CuTe DSL's `sem=` and `scope=`."]
        #[track_caller]
        pub fn $name<T: Integer>(ptr: &Ptr<T>, value: impl IntoValue<T>) -> Value<T> {
            integer_atomic(AtomicKind::$kind, ptr, value.into_value(), None, AtomicOptions::default())
        }

        #[doc = concat!("[`", stringify!($name), "`] with CuTe DSL's `sem=` and `scope=`.")]
        #[track_caller]
        pub fn $with<T: Integer>(ptr: &Ptr<T>, value: impl IntoValue<T>, opts: AtomicOptions) -> Value<T> {
            integer_atomic(AtomicKind::$kind, ptr, value.into_value(), None, opts)
        }
    )*};
}

integer_atomics! {
    atomic_and, atomic_and_with, And, "Atomically AND `value`, returning the old contents.";
    atomic_or, atomic_or_with, Or, "Atomically OR `value`, returning the old contents.";
    atomic_xor, atomic_xor_with, Xor, "Atomically XOR `value`, returning the old contents.";
    atomic_min, atomic_min_with, Min,
        "Atomically take the minimum with `value` (signed or unsigned as `T` is), returning the old contents.";
    atomic_max, atomic_max_with, Max,
        "Atomically take the maximum with `value` (signed or unsigned as `T` is), returning the old contents.";
    atomic_exch, atomic_exch_with, Exchange, "Atomically store `value`, returning the old contents.";
}

/// Atomically store `value` if the contents equal `compare`, returning the
/// old contents (CuTe DSL's `atomic_cas(ptr, cmp=, val=)`), relaxed at GPU
/// scope.
#[track_caller]
pub fn atomic_cas<T: Integer>(ptr: &Ptr<T>, compare: impl IntoValue<T>, value: impl IntoValue<T>) -> Value<T> {
    atomic_cas_with(ptr, compare, value, AtomicOptions::default())
}

/// [`atomic_cas`] with CuTe DSL's `sem=` and `scope=`.
#[track_caller]
pub fn atomic_cas_with<T: Integer>(
    ptr: &Ptr<T>,
    compare: impl IntoValue<T>,
    value: impl IntoValue<T>,
    opts: AtomicOptions,
) -> Value<T> {
    integer_atomic(AtomicKind::CompareExchange, ptr, value.into_value(), Some(compare.into_value()), opts)
}

/// The packed type of a half-precision element.
fn packed_type(elem: DType) -> AtomicType {
    if elem == DType::BF16 { AtomicType::Bf16x2 } else { AtomicType::F16x2 }
}

/// Atomically add a pair of Float16 or BFloat16 values to the two elements
/// at `ptr` (which must be 4-byte aligned), returning the old pair in the
/// form `value` has ([`Half2`]): CuTe DSL's `atomic_add` on a two-element
/// vector, PTX's `atom.add.noftz.f16x2` (`.bf16x2` needs sm_90). Each half
/// is updated atomically on its own, rounding to nearest even and keeping
/// subnormals. Relaxed at GPU scope.
#[track_caller]
pub fn atomic_add_packed<T: HalfFloat, P: Half2<T>>(ptr: &Ptr<T>, value: P) -> P {
    atomic_add_packed_with(ptr, value, AtomicOptions::default())
}

/// [`atomic_add_packed`] with CuTe DSL's `sem=` and `scope=`.
#[track_caller]
pub fn atomic_add_packed_with<T: HalfFloat, P: Half2<T>>(ptr: &Ptr<T>, value: P, opts: AtomicOptions) -> P {
    let _at = LocationScope::new();
    let bits = value.to_packed().checked();
    let old = push(AtomicKind::Add, packed_type(T::DTYPE), ptr, bits, None, Some(DType::I32), opts);
    P::from_packed(Value::from_id(old.expect("an atomic has a result")))
}

/// Float32 max or min through the integer atomics on the value's bits, as
/// CuTe DSL builds `atomic_fmax` and `atomic_fmin`: for a value whose sign
/// bit is clear, IEEE order is the order of the bits as Int32s; for one
/// whose sign bit is set, the reverse order of the bits as Uint32s. So fmax
/// takes the signed maximum or the unsigned minimum of the bits, fmin the
/// signed minimum or the unsigned maximum.
#[track_caller]
fn float_extremum(max: bool, ptr: &Ptr<f32>, value: Float32, sign_bit: Option<bool>, opts: AtomicOptions) -> Float32 {
    let _at = LocationScope::new();
    let bits: Int32 = value.bitcast();
    let negative = || {
        let p = ptr.cast::<u32>();
        let v = bits.bitcast::<u32>();
        let old = if max { atomic_min_with(&p, v, opts) } else { atomic_max_with(&p, v, opts) };
        old.bitcast::<i32>()
    };
    let positive = || {
        let p = ptr.cast::<i32>();
        if max { atomic_max_with(&p, bits, opts) } else { atomic_min_with(&p, bits, opts) }
    };
    let old = match sign_bit {
        None => if_else_yield(bits.lt(0), negative, positive),
        Some(true) => negative(),
        Some(false) => positive(),
    };
    old.bitcast()
}

/// Atomically take the IEEE maximum of the Float32 at `ptr` and `value`,
/// returning the old contents (CuTe DSL's `atomic_fmax`), relaxed at GPU
/// scope. Built from integer atomics on the bits ([`atomic_max`] on Int32s
/// for a value with its sign bit clear, [`atomic_min`] on Uint32s for one
/// with it set), chosen at run time. Exact for every value but NaNs with the
/// sign bit set, including `+-0` (`-0 < +0`), the infinities, subnormals,
/// and NaNs with the sign bit clear such as the canonical NaN, which win
/// over every number as the largest bit patterns.
#[track_caller]
pub fn atomic_fmax(ptr: &Ptr<f32>, value: impl IntoValue<f32>) -> Float32 {
    float_extremum(true, ptr, value.into_value(), None, AtomicOptions::default())
}

/// [`atomic_fmax`] with CuTe DSL's `sign_bit=`, `sem=` and `scope=`.
/// `sign_bit: Some(s)` promises that `value`'s sign bit is `s` and picks the
/// integer atomic at build time; a wrong promise gives wrong results, with
/// no diagnostic.
#[track_caller]
pub fn atomic_fmax_with(
    ptr: &Ptr<f32>,
    value: impl IntoValue<f32>,
    sign_bit: Option<bool>,
    opts: AtomicOptions,
) -> Float32 {
    float_extremum(true, ptr, value.into_value(), sign_bit, opts)
}

/// Atomically take the IEEE minimum of the Float32 at `ptr` and `value`,
/// returning the old contents (CuTe DSL's `atomic_fmin`), relaxed at GPU
/// scope: [`atomic_min`] on Int32s for a value with its sign bit clear,
/// [`atomic_max`] on Uint32s for one with it set. Exact as [`atomic_fmax`]
/// is; a NaN with the sign bit clear loses to every number.
#[track_caller]
pub fn atomic_fmin(ptr: &Ptr<f32>, value: impl IntoValue<f32>) -> Float32 {
    float_extremum(false, ptr, value.into_value(), None, AtomicOptions::default())
}

/// [`atomic_fmin`] with CuTe DSL's `sign_bit=`, `sem=` and `scope=`, as
/// [`atomic_fmax_with`] takes them.
#[track_caller]
pub fn atomic_fmin_with(
    ptr: &Ptr<f32>,
    value: impl IntoValue<f32>,
    sign_bit: Option<bool>,
    opts: AtomicOptions,
) -> Float32 {
    float_extremum(false, ptr, value.into_value(), sign_bit, opts)
}

/// CuTe DSL's deprecated `atomic_max_float32(ptr, value, positive_only=True)`:
/// [`atomic_fmax`], promising a clear sign bit when `positive_only` is set.
#[deprecated(note = "use atomic_fmax (or atomic_fmax_with and sign_bit: Some(false))")]
#[track_caller]
pub fn atomic_max_float32(ptr: &Ptr<f32>, value: impl IntoValue<f32>, positive_only: bool) -> Float32 {
    float_extremum(true, ptr, value.into_value(), positive_only.then_some(false), AtomicOptions::default())
}

/// The `op=` of [`red`] (CuTe DSL's `cute.arch.red`). `Umin` and `Umax`
/// are `Min` and `Max` on unsigned integers: they turn a signed `dtype`
/// (`S32`, `S64`) into its unsigned one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RedOp {
    Add,
    Min,
    Max,
    Umin,
    Umax,
    And,
    Or,
    Xor,
}

impl From<DType> for AtomicType {
    /// The type of CuTe DSL's `dtype=` given as a numeric type: `Int32` is
    /// `s32`, `Uint32` `u32`, `Float16` `f16`, and so on.
    fn from(t: DType) -> AtomicType {
        match t {
            DType::I32 => AtomicType::S32,
            DType::U32 => AtomicType::U32,
            DType::I64 => AtomicType::S64,
            DType::U64 => AtomicType::U64,
            DType::F32 => AtomicType::F32,
            DType::F64 => AtomicType::F64,
            DType::F16 => AtomicType::F16,
            DType::BF16 => AtomicType::Bf16,
            other => panic!("cute: red has no {other} type (it takes 32- and 64-bit integers and floats, and halves)"),
        }
    }
}

/// The operation and type of `red(op=, dtype=)`: `Umin` and `Umax` make a
/// signed type unsigned, as CuTe DSL does, and an add on `s64`, which PTX
/// lacks, is the same add on `u64`.
fn red_kind(op: RedOp, dtype: AtomicType) -> (AtomicKind, AtomicType) {
    let unsigned = |t| match t {
        AtomicType::S32 => AtomicType::U32,
        AtomicType::S64 => AtomicType::U64,
        t => t,
    };
    match op {
        RedOp::Add if dtype == AtomicType::S64 => (AtomicKind::Add, AtomicType::U64),
        RedOp::Add => (AtomicKind::Add, dtype),
        RedOp::Min => (AtomicKind::Min, dtype),
        RedOp::Max => (AtomicKind::Max, dtype),
        RedOp::Umin => (AtomicKind::Min, unsigned(dtype)),
        RedOp::Umax => (AtomicKind::Max, unsigned(dtype)),
        RedOp::And => (AtomicKind::And, dtype),
        RedOp::Or => (AtomicKind::Or, dtype),
        RedOp::Xor => (AtomicKind::Xor, dtype),
    }
}

/// Atomically combine `value` with the element at `ptr`, without returning
/// anything: CuTe DSL's `cute.arch.red(ptr, val, op=, dtype=, sem=, scope=)`,
/// PTX's `red`. `dtype` is the PTX type (`AtomicType::U32`, or a numeric type
/// such as `DType::I32`), as wide as `T`: a bit, signed or unsigned type for
/// an integer `T`, `T`'s own type for a float. PTX has add on `u32`, `s32`,
/// `u64` and the floats, min and max on the integer types, and AND, OR and
/// XOR on `b32` and `b64`; the validator rejects other combinations (there
/// is no `red.max.f32`). `sem` is relaxed or release. Floats round as
/// [`atomic_add`]'s do. The packed half-precision types go through
/// [`red_packed`].
#[track_caller]
pub fn red<T: Numeric>(
    ptr: &Ptr<T>,
    value: impl IntoValue<T>,
    op: RedOp,
    dtype: impl Into<AtomicType>,
    opts: AtomicOptions,
) {
    let dtype = dtype.into();
    assert!(!dtype.is_packed(), "cute: red with dtype {} takes a packed pair: use red_packed", dtype.ptx());
    assert!(
        dtype.bits() == T::DTYPE.bits(),
        "cute: red with dtype {} on {} values: the dtype must be as wide as the values",
        dtype.ptx(),
        T::DTYPE
    );
    let float_matches = AtomicType::from(T::DTYPE) == dtype;
    assert!(
        dtype.is_float() == T::DTYPE.is_float() && (!dtype.is_float() || float_matches),
        "cute: red with dtype {} on {} values: a float dtype must be the values' own type, an integer one needs \
         integer values",
        dtype.ptx(),
        T::DTYPE
    );
    let (kind, ty) = red_kind(op, dtype);
    push(kind, ty, ptr, value.into_value().checked(), None, None, opts);
}

/// [`red`] on a pair of Float16 or BFloat16 values in the form [`Half2`]
/// takes (`dtype` `f16x2` or `bf16x2`, which needs sm_90), at a 4-byte
/// aligned `ptr`. Each half is updated atomically on its own. PTX has only
/// add on these types.
#[track_caller]
pub fn red_packed<T: HalfFloat, P: Half2<T>>(ptr: &Ptr<T>, value: P, op: RedOp, opts: AtomicOptions) {
    let _at = LocationScope::new();
    let (kind, ty) = red_kind(op, packed_type(T::DTYPE));
    let bits = value.to_packed().checked();
    push(kind, ty, ptr, bits, None, None, opts);
}

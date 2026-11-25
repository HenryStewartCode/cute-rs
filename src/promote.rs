//! CuTe DSL's implicit type promotion for operations on two runtime values
//! of different types (`_binary_op_type_promote` in its `typing.py`).
//!
//! `a + b` with `a: Value<A>` and `b: Value<B>` converts both operands to
//! the promoted type `<A as Promote<B>>::Output` and computes there:
//!
//! * an integer meeting a float becomes that float type at the wider of the
//!   two widths (Float16, Float32 or Float64: `Float16 + Int32` is Float32,
//!   `Float32 + Int64` is Float64, `BFloat16 + Int16` is Float16);
//! * floats widen; at equal width Float64, then Float32 (over TFloat32),
//!   then Float16 (over BFloat16) wins; other pairs, such as two different
//!   FP8 formats or an FP8 format with an 8-bit integer, have no promotion
//!   and need an explicit [`Value::to`];
//! * integers of mixed signedness become the unsigned type when it is at
//!   least as wide as the signed one, else the signed type; otherwise the
//!   wider type wins (Boolean is a 1-bit signed integer there);
//! * comparisons ([`Value::lt`] and friends) promote, then compare.
//!
//! Differences from CuTe DSL, all deliberate:
//!
//! * Rust literals keep adopting the other operand's type, as before:
//!   `x_f16 + 1.0` is Float16 (CuTe DSL makes the literal a Float32 and
//!   the result Float32; write `x.to::<f32>() + 1.0` for that).
//! * Arithmetic whose CuTe DSL type would be TFloat32 is computed in
//!   Float32 and gives a Float32 (cute-rs has no TFloat32 arithmetic).
//! * Integer `/` and `%` truncate, as Rust's do (CuTe DSL's integer `/` is a
//!   true division giving Float32: [`true_div`](crate::true_div)).
//!
//! ```
//! use cute::prelude::*;
//! Kernel::build("promote", || {
//!     let out = param_tensor::<f32>("out", Layout::col_major(32));
//!     let t = thread_idx().x;
//!     let h: Float16 = t.to::<F16>();
//!     let x: Float32 = h * t + 0.5; // Float16 * Int32 is Float32
//!     out.store(t, x);
//! });
//! ```
//!
//! Pairs CuTe DSL refuses do not compile: two FP8 formats,
//!
//! ```compile_fail
//! use cute::prelude::*;
//! Kernel::build("fp8_pair", || {
//!     let a = Value::constant(F8E4M3FN::from_f32(1.0));
//!     let b = Value::constant(F8E5M2::from_f32(2.0));
//!     let _ = a + b;
//! });
//! ```
//!
//! an FP8 format with an 8-bit integer,
//!
//! ```compile_fail
//! use cute::prelude::*;
//! Kernel::build("fp8_int8", || {
//!     let a = Value::constant(F8E4M3FN::from_f32(1.0));
//!     let _ = a * Int8::constant(3);
//! });
//! ```
//!
//! and FP8 arithmetic without a conversion (the FP8 formats are storage
//! types, with no arithmetic of their own):
//!
//! ```compile_fail
//! use cute::prelude::*;
//! Kernel::build("fp8_add", || {
//!     let a = Value::constant(F8E5M2::from_f32(1.0));
//!     let _ = a + a;
//! });
//! ```
//!
//! Boolean has no negation, as in CuTe DSL:
//!
//! ```compile_fail
//! use cute::prelude::*;
//! Kernel::build("neg_bool", || {
//!     let _ = -Boolean::constant(true);
//! });
//! ```
//!
//! The table below was generated from those rules; `tests/ops.rs` checks it
//! against an independent implementation ([`DType::promote`](crate::DType::promote)).

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};

use crate::ir::{BinaryOp, CmpPred, DType};
use crate::numeric::{F8E4M3, F8E4M3B11FNUZ, F8E4M3FN, F8E5M2, F8E5M3FNU, F8E8M0FNU, I4, TF32};
use crate::value::{BF16, Boolean, F16, IntoValue, Numeric, Value, binary_op, compare_values};

/// The type CuTe DSL promotes `Self` and `Rhs` to for arithmetic (with
/// TFloat32 results computed in Float32). Implemented for every ordered
/// pair of distinct types that has a promotion.
pub trait Promote<Rhs: Numeric>: Numeric {
    type Output: Numeric;
}

/// The type both operands of an elementwise operation on tensor values
/// ([`TensorSsa::apply_op`](crate::TensorSsa::apply_op), the comparisons,
/// the two-operand math functions) are converted to: `Self` for two
/// operands of one type, else [`Promote`]'s output.
pub trait CommonType<Rhs: Numeric>: Numeric {
    type Output: Numeric;
}

impl<T: Numeric> CommonType<T> for T {
    type Output = T;
}

/// A right-hand side of a comparison with a `Value<T>`: a value or host
/// constant of type `T`, or a value of a type `T` promotes with.
pub trait CmpOperand<T: Numeric> {
    #[doc(hidden)]
    fn compare_with(self, lhs: Value<T>, pred: CmpPred) -> Boolean;
}

impl<T: Numeric, X: IntoValue<T>> CmpOperand<T> for X {
    #[track_caller]
    fn compare_with(self, lhs: Value<T>, pred: CmpPred) -> Boolean {
        compare_values(pred, lhs, self.into_value())
    }
}

#[track_caller]
fn promoted<A: Numeric, B: Numeric, R: Numeric>(op: BinaryOp, a: Value<A>, b: Value<B>) -> Value<R> {
    let op = if op == BinaryOp::ShrS && !R::DTYPE.is_signed() { BinaryOp::ShrU } else { op };
    binary_op(op, a.to::<R>(), b.to::<R>())
}

macro_rules! mixed_ops {
    ($a:ty, $b:ty => $r:ty; $(($trait:ident, $method:ident, $op:expr)),*) => {$(
        impl $trait<Value<$b>> for Value<$a> {
            type Output = Value<$r>;
            #[track_caller]
            fn $method(self, rhs: Value<$b>) -> Value<$r> { promoted($op, self, rhs) }
        }
    )*};
}

macro_rules! promote_table {
    ($($a:ty, $b:ty => $r:ty, $kind:ident);* $(;)?) => {$(
        impl Promote<$b> for $a {
            type Output = $r;
        }
        impl CmpOperand<$a> for Value<$b> {
            #[track_caller]
            fn compare_with(self, lhs: Value<$a>, pred: CmpPred) -> Boolean {
                compare_values(pred, lhs.to::<$r>(), self.to::<$r>())
            }
        }
        mixed_ops!($a, $b => $r; (Add, add, BinaryOp::Add), (Sub, sub, BinaryOp::Sub), (Mul, mul, BinaryOp::Mul),
            (Div, div, BinaryOp::Div));
        promote_table!(@$kind $a, $b => $r);
    )*};
    (@int $a:ty, $b:ty => $r:ty) => {
        mixed_ops!($a, $b => $r; (Rem, rem, BinaryOp::Rem), (BitAnd, bitand, BinaryOp::And),
            (BitOr, bitor, BinaryOp::Or), (BitXor, bitxor, BinaryOp::Xor), (Shl, shl, BinaryOp::Shl),
            (Shr, shr, BinaryOp::ShrS));
    };
    (@float $a:ty, $b:ty => $r:ty) => {};
}

/// Arithmetic on two TFloat32 values is computed in Float32.
macro_rules! tf32_same {
    ($(($trait:ident, $method:ident, $op:expr)),*) => {$(
        impl $trait<Value<TF32>> for Value<TF32> {
            type Output = Value<f32>;
            #[track_caller]
            fn $method(self, rhs: Value<TF32>) -> Value<f32> { promoted($op, self, rhs) }
        }
    )*};
}
tf32_same!((Add, add, BinaryOp::Add), (Sub, sub, BinaryOp::Sub), (Mul, mul, BinaryOp::Mul), (Div, div, BinaryOp::Div));

/// Arithmetic on two Booleans, as CuTe DSL computes it: in Int32, then
/// converted back to Boolean (`x != 0`), so `true + true` is `true`.
macro_rules! bool_same {
    ($(($trait:ident, $method:ident, $op:expr)),*) => {$(
        impl $trait<Value<bool>> for Value<bool> {
            type Output = Boolean;
            #[track_caller]
            fn $method(self, rhs: Value<bool>) -> Boolean { promoted::<bool, bool, i32>($op, self, rhs).to() }
        }
    )*};
}
bool_same!((Add, add, BinaryOp::Add), (Sub, sub, BinaryOp::Sub), (Mul, mul, BinaryOp::Mul));

/// CuTe DSL's true division of integers (its `/` on two integer values,
/// and `Int32 / Int32` in particular): both operands converted to Float32
/// and divided, a Float32 result.
#[track_caller]
pub fn true_div<T: Numeric>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<f32> {
    let (a, b) = (a.into_value().to::<f32>(), b.into_value().to::<f32>());
    a / b
}

/// The promotion table: `A, B => Output, kind` for every ordered pair of
/// distinct types with a promotion, handed to `$callback`.
macro_rules! with_promotion_table {
    ($callback:ident) => {
        $callback! {
            bool, i8 => i8, int;
            bool, i16 => i16, int;
            bool, i32 => i32, int;
            bool, i64 => i64, int;
            bool, u8 => u8, int;
            bool, u16 => u16, int;
            bool, u32 => u32, int;
            bool, u64 => u64, int;
            bool, F16 => F16, float;
            bool, BF16 => BF16, float;
            bool, f32 => f32, float;
            bool, f64 => f64, float;
            bool, TF32 => f32, float;
            bool, I4 => I4, int;
            bool, i128 => i128, int;
            bool, u128 => u128, int;
            i8, bool => i8, int;
            i8, i16 => i16, int;
            i8, i32 => i32, int;
            i8, i64 => i64, int;
            i8, u8 => u8, int;
            i8, u16 => u16, int;
            i8, u32 => u32, int;
            i8, u64 => u64, int;
            i8, F16 => F16, float;
            i8, BF16 => BF16, float;
            i8, f32 => f32, float;
            i8, f64 => f64, float;
            i8, TF32 => f32, float;
            i8, I4 => i8, int;
            i8, i128 => i128, int;
            i8, u128 => u128, int;
            i16, bool => i16, int;
            i16, i8 => i16, int;
            i16, i32 => i32, int;
            i16, i64 => i64, int;
            i16, u8 => i16, int;
            i16, u16 => u16, int;
            i16, u32 => u32, int;
            i16, u64 => u64, int;
            i16, F16 => F16, float;
            i16, BF16 => F16, float;
            i16, f32 => f32, float;
            i16, f64 => f64, float;
            i16, TF32 => f32, float;
            i16, F8E4M3FN => F16, float;
            i16, F8E5M2 => F16, float;
            i16, F8E4M3 => F16, float;
            i16, F8E4M3B11FNUZ => F16, float;
            i16, F8E8M0FNU => F16, float;
            i16, F8E5M3FNU => F16, float;
            i16, I4 => i16, int;
            i16, i128 => i128, int;
            i16, u128 => u128, int;
            i32, bool => i32, int;
            i32, i8 => i32, int;
            i32, i16 => i32, int;
            i32, i64 => i64, int;
            i32, u8 => i32, int;
            i32, u16 => i32, int;
            i32, u32 => u32, int;
            i32, u64 => u64, int;
            i32, F16 => f32, float;
            i32, BF16 => f32, float;
            i32, f32 => f32, float;
            i32, f64 => f64, float;
            i32, TF32 => f32, float;
            i32, F8E4M3FN => f32, float;
            i32, F8E5M2 => f32, float;
            i32, F8E4M3 => f32, float;
            i32, F8E4M3B11FNUZ => f32, float;
            i32, F8E8M0FNU => f32, float;
            i32, F8E5M3FNU => f32, float;
            i32, I4 => i32, int;
            i32, i128 => i128, int;
            i32, u128 => u128, int;
            i64, bool => i64, int;
            i64, i8 => i64, int;
            i64, i16 => i64, int;
            i64, i32 => i64, int;
            i64, u8 => i64, int;
            i64, u16 => i64, int;
            i64, u32 => i64, int;
            i64, u64 => u64, int;
            i64, F16 => f64, float;
            i64, BF16 => f64, float;
            i64, f32 => f64, float;
            i64, f64 => f64, float;
            i64, TF32 => f64, float;
            i64, F8E4M3FN => f64, float;
            i64, F8E5M2 => f64, float;
            i64, F8E4M3 => f64, float;
            i64, F8E4M3B11FNUZ => f64, float;
            i64, F8E8M0FNU => f64, float;
            i64, F8E5M3FNU => f64, float;
            i64, I4 => i64, int;
            i64, i128 => i128, int;
            i64, u128 => u128, int;
            u8, bool => u8, int;
            u8, i8 => u8, int;
            u8, i16 => i16, int;
            u8, i32 => i32, int;
            u8, i64 => i64, int;
            u8, u16 => u16, int;
            u8, u32 => u32, int;
            u8, u64 => u64, int;
            u8, F16 => F16, float;
            u8, BF16 => BF16, float;
            u8, f32 => f32, float;
            u8, f64 => f64, float;
            u8, TF32 => f32, float;
            u8, I4 => u8, int;
            u8, i128 => i128, int;
            u8, u128 => u128, int;
            u16, bool => u16, int;
            u16, i8 => u16, int;
            u16, i16 => u16, int;
            u16, i32 => i32, int;
            u16, i64 => i64, int;
            u16, u8 => u16, int;
            u16, u32 => u32, int;
            u16, u64 => u64, int;
            u16, F16 => F16, float;
            u16, BF16 => F16, float;
            u16, f32 => f32, float;
            u16, f64 => f64, float;
            u16, TF32 => f32, float;
            u16, F8E4M3FN => F16, float;
            u16, F8E5M2 => F16, float;
            u16, F8E4M3 => F16, float;
            u16, F8E4M3B11FNUZ => F16, float;
            u16, F8E8M0FNU => F16, float;
            u16, F8E5M3FNU => F16, float;
            u16, I4 => u16, int;
            u16, i128 => i128, int;
            u16, u128 => u128, int;
            u32, bool => u32, int;
            u32, i8 => u32, int;
            u32, i16 => u32, int;
            u32, i32 => u32, int;
            u32, i64 => i64, int;
            u32, u8 => u32, int;
            u32, u16 => u32, int;
            u32, u64 => u64, int;
            u32, F16 => f32, float;
            u32, BF16 => f32, float;
            u32, f32 => f32, float;
            u32, f64 => f64, float;
            u32, TF32 => f32, float;
            u32, F8E4M3FN => f32, float;
            u32, F8E5M2 => f32, float;
            u32, F8E4M3 => f32, float;
            u32, F8E4M3B11FNUZ => f32, float;
            u32, F8E8M0FNU => f32, float;
            u32, F8E5M3FNU => f32, float;
            u32, I4 => u32, int;
            u32, i128 => i128, int;
            u32, u128 => u128, int;
            u64, bool => u64, int;
            u64, i8 => u64, int;
            u64, i16 => u64, int;
            u64, i32 => u64, int;
            u64, i64 => u64, int;
            u64, u8 => u64, int;
            u64, u16 => u64, int;
            u64, u32 => u64, int;
            u64, F16 => f64, float;
            u64, BF16 => f64, float;
            u64, f32 => f64, float;
            u64, f64 => f64, float;
            u64, TF32 => f64, float;
            u64, F8E4M3FN => f64, float;
            u64, F8E5M2 => f64, float;
            u64, F8E4M3 => f64, float;
            u64, F8E4M3B11FNUZ => f64, float;
            u64, F8E8M0FNU => f64, float;
            u64, F8E5M3FNU => f64, float;
            u64, I4 => u64, int;
            u64, i128 => i128, int;
            u64, u128 => u128, int;
            F16, bool => F16, float;
            F16, i8 => F16, float;
            F16, i16 => F16, float;
            F16, i32 => f32, float;
            F16, i64 => f64, float;
            F16, u8 => F16, float;
            F16, u16 => F16, float;
            F16, u32 => f32, float;
            F16, u64 => f64, float;
            F16, BF16 => F16, float;
            F16, f32 => f32, float;
            F16, f64 => f64, float;
            F16, TF32 => f32, float;
            F16, F8E4M3FN => F16, float;
            F16, F8E5M2 => F16, float;
            F16, F8E4M3 => F16, float;
            F16, F8E4M3B11FNUZ => F16, float;
            F16, F8E8M0FNU => F16, float;
            F16, F8E5M3FNU => F16, float;
            F16, I4 => F16, float;
            BF16, bool => BF16, float;
            BF16, i8 => BF16, float;
            BF16, i16 => F16, float;
            BF16, i32 => f32, float;
            BF16, i64 => f64, float;
            BF16, u8 => BF16, float;
            BF16, u16 => F16, float;
            BF16, u32 => f32, float;
            BF16, u64 => f64, float;
            BF16, F16 => F16, float;
            BF16, f32 => f32, float;
            BF16, f64 => f64, float;
            BF16, TF32 => f32, float;
            BF16, F8E4M3FN => BF16, float;
            BF16, F8E5M2 => BF16, float;
            BF16, F8E4M3 => BF16, float;
            BF16, F8E4M3B11FNUZ => BF16, float;
            BF16, F8E8M0FNU => BF16, float;
            BF16, F8E5M3FNU => BF16, float;
            BF16, I4 => BF16, float;
            f32, bool => f32, float;
            f32, i8 => f32, float;
            f32, i16 => f32, float;
            f32, i32 => f32, float;
            f32, i64 => f64, float;
            f32, u8 => f32, float;
            f32, u16 => f32, float;
            f32, u32 => f32, float;
            f32, u64 => f64, float;
            f32, F16 => f32, float;
            f32, BF16 => f32, float;
            f32, f64 => f64, float;
            f32, TF32 => f32, float;
            f32, F8E4M3FN => f32, float;
            f32, F8E5M2 => f32, float;
            f32, F8E4M3 => f32, float;
            f32, F8E4M3B11FNUZ => f32, float;
            f32, F8E8M0FNU => f32, float;
            f32, F8E5M3FNU => f32, float;
            f32, I4 => f32, float;
            f64, bool => f64, float;
            f64, i8 => f64, float;
            f64, i16 => f64, float;
            f64, i32 => f64, float;
            f64, i64 => f64, float;
            f64, u8 => f64, float;
            f64, u16 => f64, float;
            f64, u32 => f64, float;
            f64, u64 => f64, float;
            f64, F16 => f64, float;
            f64, BF16 => f64, float;
            f64, f32 => f64, float;
            f64, TF32 => f64, float;
            f64, F8E4M3FN => f64, float;
            f64, F8E5M2 => f64, float;
            f64, F8E4M3 => f64, float;
            f64, F8E4M3B11FNUZ => f64, float;
            f64, F8E8M0FNU => f64, float;
            f64, F8E5M3FNU => f64, float;
            f64, I4 => f64, float;
            TF32, bool => f32, float;
            TF32, i8 => f32, float;
            TF32, i16 => f32, float;
            TF32, i32 => f32, float;
            TF32, i64 => f64, float;
            TF32, u8 => f32, float;
            TF32, u16 => f32, float;
            TF32, u32 => f32, float;
            TF32, u64 => f64, float;
            TF32, F16 => f32, float;
            TF32, BF16 => f32, float;
            TF32, f32 => f32, float;
            TF32, f64 => f64, float;
            TF32, F8E4M3FN => f32, float;
            TF32, F8E5M2 => f32, float;
            TF32, F8E4M3 => f32, float;
            TF32, F8E4M3B11FNUZ => f32, float;
            TF32, F8E8M0FNU => f32, float;
            TF32, F8E5M3FNU => f32, float;
            TF32, I4 => f32, float;
            F8E4M3FN, i16 => F16, float;
            F8E4M3FN, i32 => f32, float;
            F8E4M3FN, i64 => f64, float;
            F8E4M3FN, u16 => F16, float;
            F8E4M3FN, u32 => f32, float;
            F8E4M3FN, u64 => f64, float;
            F8E4M3FN, F16 => F16, float;
            F8E4M3FN, BF16 => BF16, float;
            F8E4M3FN, f32 => f32, float;
            F8E4M3FN, f64 => f64, float;
            F8E4M3FN, TF32 => f32, float;
            F8E5M2, i16 => F16, float;
            F8E5M2, i32 => f32, float;
            F8E5M2, i64 => f64, float;
            F8E5M2, u16 => F16, float;
            F8E5M2, u32 => f32, float;
            F8E5M2, u64 => f64, float;
            F8E5M2, F16 => F16, float;
            F8E5M2, BF16 => BF16, float;
            F8E5M2, f32 => f32, float;
            F8E5M2, f64 => f64, float;
            F8E5M2, TF32 => f32, float;
            F8E4M3, i16 => F16, float;
            F8E4M3, i32 => f32, float;
            F8E4M3, i64 => f64, float;
            F8E4M3, u16 => F16, float;
            F8E4M3, u32 => f32, float;
            F8E4M3, u64 => f64, float;
            F8E4M3, F16 => F16, float;
            F8E4M3, BF16 => BF16, float;
            F8E4M3, f32 => f32, float;
            F8E4M3, f64 => f64, float;
            F8E4M3, TF32 => f32, float;
            F8E4M3B11FNUZ, i16 => F16, float;
            F8E4M3B11FNUZ, i32 => f32, float;
            F8E4M3B11FNUZ, i64 => f64, float;
            F8E4M3B11FNUZ, u16 => F16, float;
            F8E4M3B11FNUZ, u32 => f32, float;
            F8E4M3B11FNUZ, u64 => f64, float;
            F8E4M3B11FNUZ, F16 => F16, float;
            F8E4M3B11FNUZ, BF16 => BF16, float;
            F8E4M3B11FNUZ, f32 => f32, float;
            F8E4M3B11FNUZ, f64 => f64, float;
            F8E4M3B11FNUZ, TF32 => f32, float;
            F8E8M0FNU, i16 => F16, float;
            F8E8M0FNU, i32 => f32, float;
            F8E8M0FNU, i64 => f64, float;
            F8E8M0FNU, u16 => F16, float;
            F8E8M0FNU, u32 => f32, float;
            F8E8M0FNU, u64 => f64, float;
            F8E8M0FNU, F16 => F16, float;
            F8E8M0FNU, BF16 => BF16, float;
            F8E8M0FNU, f32 => f32, float;
            F8E8M0FNU, f64 => f64, float;
            F8E8M0FNU, TF32 => f32, float;
            F8E5M3FNU, i16 => F16, float;
            F8E5M3FNU, i32 => f32, float;
            F8E5M3FNU, i64 => f64, float;
            F8E5M3FNU, u16 => F16, float;
            F8E5M3FNU, u32 => f32, float;
            F8E5M3FNU, u64 => f64, float;
            F8E5M3FNU, F16 => F16, float;
            F8E5M3FNU, BF16 => BF16, float;
            F8E5M3FNU, f32 => f32, float;
            F8E5M3FNU, f64 => f64, float;
            F8E5M3FNU, TF32 => f32, float;
            I4, bool => I4, int;
            I4, i8 => i8, int;
            I4, i16 => i16, int;
            I4, i32 => i32, int;
            I4, i64 => i64, int;
            I4, u8 => u8, int;
            I4, u16 => u16, int;
            I4, u32 => u32, int;
            I4, u64 => u64, int;
            I4, F16 => F16, float;
            I4, BF16 => BF16, float;
            I4, f32 => f32, float;
            I4, f64 => f64, float;
            I4, TF32 => f32, float;
            I4, i128 => i128, int;
            I4, u128 => u128, int;
            i128, bool => i128, int;
            i128, i8 => i128, int;
            i128, i16 => i128, int;
            i128, i32 => i128, int;
            i128, i64 => i128, int;
            i128, u8 => i128, int;
            i128, u16 => i128, int;
            i128, u32 => i128, int;
            i128, u64 => i128, int;
            i128, I4 => i128, int;
            i128, u128 => u128, int;
            u128, bool => u128, int;
            u128, i8 => u128, int;
            u128, i16 => u128, int;
            u128, i32 => u128, int;
            u128, i64 => u128, int;
            u128, u8 => u128, int;
            u128, u16 => u128, int;
            u128, u32 => u128, int;
            u128, u64 => u128, int;
            u128, I4 => u128, int;
            u128, i128 => u128, int
        }
    };
}

macro_rules! promotion_list {
    ($($a:ty, $b:ty => $r:ty, $kind:ident);* $(;)?) => {
        /// Every row of the promotion table, `(A, B, Output)`, for tests.
        #[doc(hidden)]
        pub const PROMOTIONS: &[(DType, DType, DType)] =
            &[$((<$a as Numeric>::DTYPE, <$b as Numeric>::DTYPE, <$r as Numeric>::DTYPE)),*];
    };
}

macro_rules! common_table {
    ($($a:ty, $b:ty => $r:ty, $kind:ident);* $(;)?) => {$(
        impl CommonType<$b> for $a {
            type Output = $r;
        }
    )*};
}

with_promotion_table!(promote_table);
with_promotion_table!(common_table);
with_promotion_table!(promotion_list);

//! Multi-GPU helpers over multicast memory (CuTe DSL's `cutlass.utils.distributed`): `multimem` load-reduce, store and reduction, and flag spin locks.
//!
//! # Multicast memory
//!
//! A multimem (multicast) address is a virtual address that maps to one buffer on each of
//! several NVLink-connected GPUs; the host makes one with `cuMulticastCreate`,
//! `cuMulticastAddDevice`, `cuMulticastBindMem` and a mapping
//! ([`MulticastGroup`](crate::cuda::MulticastGroup)), or gets it from NVSHMEM. The PTX ISA
//! allows only the `multimem.*` instructions on such an address.
//!
//! In cute-rs the buffer is flagged, not the pointer type: a kernel parameter declared with
//! [`param_tensor_multicast`] or [`param_ptr_multicast`] is an ordinary [`Tensor`] or [`Ptr`]
//! (so layouts, tiling and partitioning work unchanged) whose buffer is marked multicast. The
//! validator rejects ordinary loads, stores, atomics and copies on a multicast buffer, and the
//! multimem instructions on any other. The simulator gives a multicast parameter a group of
//! replicas, one per simulated GPU ([`MulticastReplicas`](crate::sim::MulticastReplicas),
//! [`Kernel::simulate_multicast`](crate::Kernel::simulate_multicast)).
//!
//! # Instructions
//!
//! * [`multimem_ld_reduce`] and its fifteen typed variants: `multimem.ld_reduce.weak.global.add`
//!   of 128, 64 or 32 bits, the sum over every replica, returned as 32-bit registers (Int32, as
//!   CuTe DSL returns them). Float16 and BFloat16 accumulate in Float32 (`.acc::f32`, PTX ISA
//!   8.2), FP8 in Float16 (`.acc::f16`; PTX ISA 8.6 on sm_100a, sm_101a/sm_110a, sm_120a and
//!   sm_121a, or their families from PTX ISA 8.8, not sm_90). The hardware's order of the
//!   additions is unspecified; the simulator adds in replica order, so tests use values whose
//!   sums are exact.
//! * [`multimem_st`] and `multimem_st_{4,2,1}xb32`: `multimem.st.weak.global` to every
//!   replica.
//! * [`multimem_red_add1`] and its four named variants: `multimem.red.{release,relaxed}.{gpu,sys}
//!   .global.add.s32 [a], 1`.
//!
//! Every multimem instruction needs sm_90 and PTX ISA 8.1, and an address aligned to its size
//! (16 bytes for 128 bits, 8 for 64, 4 for 32).
//!
//! # Flag locks
//!
//! [`red_add1`], [`spin_lock_atom_cas_relaxed_wait`], [`spin_lock_atom_cas_acquire_wait`] and
//! [`spin_lock_ld_lt_relaxed_wait`] are thin wrappers over [`red`],
//! [`atomic_cas_with`] and an ordered [`load`] in a
//! [`while_carry`] loop. Waiting for another block needs the blocks to be
//! co-resident: on a GPU a small grid or a cooperative launch
//! ([`LaunchConfig::with_cooperative`](crate::sim::LaunchConfig::with_cooperative)); in the
//! simulator a cooperative launch or [`ResidentClusters::Grid`](crate::sim::ResidentClusters),
//! which reports a wait that can never succeed as a hang naming the spinning loop.

use crate::arch::{AtomicOptions, LoadOptions, RedOp, atomic_cas_with, load, red};
use crate::builder;
use crate::control::while_carry;
use crate::ir::{AtomicType, DType, MemoryOrder, MemoryScope, MultimemOp, MultimemType, Op};
use crate::tensor::{Ptr, Tensor};
use crate::value::{Boolean, Int32, IntoValue, Numeric, Value};

/// Mark the buffer of `ptr` (a kernel parameter) multicast.
fn mark_multicast(buffer: crate::ir::BufferId) {
    builder::with(|b| {
        if !b.multicast.contains(&buffer) {
            b.multicast.push(buffer);
        }
    });
}

/// Declare a multicast tensor parameter: [`param_tensor`](crate::param_tensor) whose buffer
/// holds a multimem address, which only the multimem instructions access.
#[track_caller]
pub fn param_tensor_multicast<T: Numeric>(name: &str, layout: impl crate::params::TensorParamSource) -> Tensor<T> {
    let t = crate::param_tensor::<T>(name, layout);
    mark_multicast(t.ptr().buffer);
    t
}

/// Declare a multicast pointer parameter: [`param_ptr`](crate::param_ptr) to a multimem
/// address.
#[track_caller]
pub fn param_ptr_multicast<T: Numeric>(name: &str) -> Ptr<T> {
    let p = crate::param_ptr::<T>(name);
    mark_multicast(p.buffer);
    p
}

/// Push a multimem instruction at `ptr`.
#[track_caller]
fn push<T: Numeric>(ptr: &Ptr<T>, op: MultimemOp, values: Vec<crate::ir::ValueId>, n_results: usize) -> Vec<Int32> {
    let loc = std::panic::Location::caller();
    let (buffer, offset) = ptr.parts();
    let index = crate::tensor::index_operand(offset);
    let ids = builder::with(|b| {
        let results: Vec<_> = (0..n_results).map(|_| b.new_value(DType::I32)).collect();
        b.push(Op::Multimem { op, buffer, index, elem: T::DTYPE, values, results: results.clone() }, loc);
        results
    });
    ids.into_iter().map(Value::from_id).collect()
}

/// `multimem.ld_reduce` of type `ty` into `regs` 32-bit registers.
#[track_caller]
fn ld_reduce<T: Numeric>(mc_ptr: &Ptr<T>, ty: MultimemType, regs: u8) -> Vec<Int32> {
    push(mc_ptr, MultimemOp::LdReduce { ty, regs }, Vec::new(), regs as usize)
}

macro_rules! ld_reduce_variants {
    ($($(#[$doc:meta])* $name:ident: $ty:ident, $regs:literal -> $ret:ty, $wrap:expr;)*) => {
        $(
            $(#[$doc])*
            #[track_caller]
            pub fn $name<T: Numeric>(mc_ptr: &Ptr<T>) -> $ret {
                let regs = ld_reduce(mc_ptr, MultimemType::$ty, $regs);
                #[allow(clippy::redundant_closure_call)]
                ($wrap)(regs)
            }
        )*
    };
}

fn four(r: Vec<Int32>) -> [Int32; 4] {
    r.try_into().unwrap_or_else(|_| unreachable!("four registers"))
}

fn two(r: Vec<Int32>) -> [Int32; 2] {
    r.try_into().unwrap_or_else(|_| unreachable!("two registers"))
}

fn one(r: Vec<Int32>) -> Int32 {
    r.into_iter().next().expect("one register")
}

ld_reduce_variants! {
    /// 8 Float16 values (128 bits), accumulated in Float32: four registers of packed pairs.
    multimem_ld_reduce_8xf16: F16x2, 4 -> [Int32; 4], four;
    /// 4 Float32 values (128 bits): four registers.
    multimem_ld_reduce_4xf32: F32, 4 -> [Int32; 4], four;
    /// 8 BFloat16 values (128 bits), accumulated in Float32.
    multimem_ld_reduce_8xbf16: Bf16x2, 4 -> [Int32; 4], four;
    /// 16 Float8E4M3FN values (128 bits), accumulated in Float16 (sm_100a and newer).
    multimem_ld_reduce_16xe4m3: E4m3x4, 4 -> [Int32; 4], four;
    /// 16 Float8E5M2 values (128 bits), accumulated in Float16 (sm_100a and newer).
    multimem_ld_reduce_16xe5m2: E5m2x4, 4 -> [Int32; 4], four;
    /// 4 Float16 values (64 bits).
    multimem_ld_reduce_4xf16: F16x2, 2 -> [Int32; 2], two;
    /// 2 Float32 values (64 bits).
    multimem_ld_reduce_2xf32: F32, 2 -> [Int32; 2], two;
    /// 4 BFloat16 values (64 bits).
    multimem_ld_reduce_4xbf16: Bf16x2, 2 -> [Int32; 2], two;
    /// 8 Float8E4M3FN values (64 bits).
    multimem_ld_reduce_8xe4m3: E4m3x4, 2 -> [Int32; 2], two;
    /// 8 Float8E5M2 values (64 bits).
    multimem_ld_reduce_8xe5m2: E5m2x4, 2 -> [Int32; 2], two;
    /// 2 Float16 values (32 bits).
    multimem_ld_reduce_2xf16: F16x2, 1 -> Int32, one;
    /// 1 Float32 value (32 bits).
    multimem_ld_reduce_1xf32: F32, 1 -> Int32, one;
    /// 2 BFloat16 values (32 bits).
    multimem_ld_reduce_2xbf16: Bf16x2, 1 -> Int32, one;
    /// 4 Float8E4M3FN values (32 bits).
    multimem_ld_reduce_4xe4m3: E4m3x4, 1 -> Int32, one;
    /// 4 Float8E5M2 values (32 bits).
    multimem_ld_reduce_4xe5m2: E5m2x4, 1 -> Int32, one;
}

/// The 128-bit base (CuTe DSL's `multimem_ld_reduce_128bit_base`): `ty` into four registers.
#[track_caller]
pub fn multimem_ld_reduce_128bit<T: Numeric>(mc_ptr: &Ptr<T>, ty: MultimemType) -> [Int32; 4] {
    four(ld_reduce(mc_ptr, ty, 4))
}

/// The 64-bit base: `ty` into two registers.
#[track_caller]
pub fn multimem_ld_reduce_64bit<T: Numeric>(mc_ptr: &Ptr<T>, ty: MultimemType) -> [Int32; 2] {
    two(ld_reduce(mc_ptr, ty, 2))
}

/// The 32-bit base: `ty` into one register.
#[track_caller]
pub fn multimem_ld_reduce_32bit<T: Numeric>(mc_ptr: &Ptr<T>, ty: MultimemType) -> Int32 {
    one(ld_reduce(mc_ptr, ty, 1))
}

/// CuTe DSL's `multimem_ld_reduce(mc_ptr, dtype=, num_elements=)`: the typed variant for
/// `dtype` (Float16, BFloat16, Float32, Float8E4M3FN or Float8E5M2) and `num_elements`
/// (8, 4 or 2 halves; 4, 2 or 1 Float32; 16, 8 or 4 FP8), returning its registers. Panics on
/// any other combination.
#[track_caller]
pub fn multimem_ld_reduce<T: Numeric>(mc_ptr: &Ptr<T>, dtype: DType, num_elements: usize) -> Vec<Int32> {
    let ty = match dtype {
        DType::F16 => MultimemType::F16x2,
        DType::BF16 => MultimemType::Bf16x2,
        DType::F32 => MultimemType::F32,
        DType::F8E4M3FN => MultimemType::E4m3x4,
        DType::F8E5M2 => MultimemType::E5m2x4,
        other => builder::user_panic(
            std::panic::Location::caller(),
            format!(
                "cute: multimem_ld_reduce has no {other} variant: it takes Float16, BFloat16, Float32, \
                 Float8E4M3FN and Float8E5M2"
            ),
        ),
    };
    let per_reg = ty.element().1;
    let regs = num_elements / per_reg;
    if !num_elements.is_multiple_of(per_reg) || !matches!(regs, 1 | 2 | 4) {
        builder::user_panic(
            std::panic::Location::caller(),
            format!(
                "cute: multimem_ld_reduce of {num_elements} {dtype} elements: it loads {}, {} or {} of them \
                 (128, 64 or 32 bits)",
                4 * per_reg,
                2 * per_reg,
                per_reg
            ),
        );
    }
    ld_reduce(mc_ptr, ty, regs as u8)
}

/// Check that `values` are 32-bit and return their ids.
#[track_caller]
fn words<V: Numeric>(values: &[Value<V>]) -> Vec<crate::ir::ValueId> {
    assert!(
        V::DTYPE.bits() == 32 && V::DTYPE != DType::Bool,
        "cute: multimem_st stores 32-bit registers, not {}",
        V::DTYPE
    );
    values.iter().map(|v| v.checked()).collect()
}

/// `multimem.st.weak.global.v4.f32`: four 32-bit registers (128 bits) to every replica.
#[track_caller]
pub fn multimem_st_4xb32<T: Numeric, V: Numeric>(mc_ptr: &Ptr<T>, regs: [Value<V>; 4]) {
    push(mc_ptr, MultimemOp::St { regs: 4 }, words(&regs), 0);
}

/// `multimem.st.weak.global.v2.f32`: two registers (64 bits).
#[track_caller]
pub fn multimem_st_2xb32<T: Numeric, V: Numeric>(mc_ptr: &Ptr<T>, regs: [Value<V>; 2]) {
    push(mc_ptr, MultimemOp::St { regs: 2 }, words(&regs), 0);
}

/// `multimem.st.weak.global.f32`: one register (32 bits).
#[track_caller]
pub fn multimem_st_1xb32<T: Numeric, V: Numeric>(mc_ptr: &Ptr<T>, reg: Value<V>) {
    push(mc_ptr, MultimemOp::St { regs: 1 }, words(&[reg]), 0);
}

/// CuTe DSL's `multimem_st(mc_ptr, *regs)`: four, two or one 32-bit registers.
#[track_caller]
pub fn multimem_st<T: Numeric, V: Numeric>(mc_ptr: &Ptr<T>, regs: &[Value<V>]) {
    match regs.len() {
        1 | 2 | 4 => {
            push(mc_ptr, MultimemOp::St { regs: regs.len() as u8 }, words(regs), 0);
        }
        n => builder::user_panic(
            std::panic::Location::caller(),
            format!("cute: multimem_st stores 4, 2 or 1 registers, not {n}"),
        ),
    }
}

/// Check a multimem or flag `order=` (release or relaxed) and `scope=` (gpu or sys).
#[track_caller]
fn order_scope(what: &str, order: MemoryOrder, scope: MemoryScope) {
    let loc = std::panic::Location::caller();
    if !matches!(order, MemoryOrder::Release | MemoryOrder::Relaxed) {
        builder::user_panic(loc, format!("cute: {what} takes order Release or Relaxed, not {order:?}"));
    }
    if !matches!(scope, MemoryScope::Gpu | MemoryScope::Sys) {
        builder::user_panic(loc, format!("cute: {what} takes scope Gpu or Sys, not {scope:?}"));
    }
}

/// `multimem.red.{order}.{scope}.global.add.s32 [lock_ptr], 1`: add one to the flag in every
/// replica (`order` Release or Relaxed, `scope` Gpu or Sys).
#[track_caller]
pub fn multimem_red_add1<T: Numeric>(lock_ptr: &Ptr<T>, order: MemoryOrder, scope: MemoryScope) {
    order_scope("multimem_red_add1", order, scope);
    push(lock_ptr, MultimemOp::RedAdd1 { order, scope }, Vec::new(), 0);
}

/// [`multimem_red_add1`], release at system scope.
#[track_caller]
pub fn multimem_red_release_sys_add1<T: Numeric>(lock_ptr: &Ptr<T>) {
    multimem_red_add1(lock_ptr, MemoryOrder::Release, MemoryScope::Sys)
}

/// [`multimem_red_add1`], release at GPU scope.
#[track_caller]
pub fn multimem_red_release_gpu_add1<T: Numeric>(lock_ptr: &Ptr<T>) {
    multimem_red_add1(lock_ptr, MemoryOrder::Release, MemoryScope::Gpu)
}

/// [`multimem_red_add1`], relaxed at system scope.
#[track_caller]
pub fn multimem_red_relaxed_sys_add1<T: Numeric>(lock_ptr: &Ptr<T>) {
    multimem_red_add1(lock_ptr, MemoryOrder::Relaxed, MemoryScope::Sys)
}

/// [`multimem_red_add1`], relaxed at GPU scope.
#[track_caller]
pub fn multimem_red_relaxed_gpu_add1<T: Numeric>(lock_ptr: &Ptr<T>) {
    multimem_red_add1(lock_ptr, MemoryOrder::Relaxed, MemoryScope::Gpu)
}

/// `red.{order}.{scope}.global.add.s32 [lock_ptr], 1` on a unicast flag (CuTe DSL's
/// `red_add1`).
#[track_caller]
pub fn red_add1(lock_ptr: &Ptr<i32>, order: MemoryOrder, scope: MemoryScope) {
    order_scope("red_add1", order, scope);
    red(lock_ptr, 1i32, RedOp::Add, AtomicType::S32, AtomicOptions::new(order, scope));
}

/// Spin until an `atom.cas.{sem}.{scope}` of `lock_ptr` from `expected_val` to `reset_val`
/// returns `expected_val`: wait for the flag to reach the count, then reset it.
#[track_caller]
fn spin_cas(lock_ptr: &Ptr<i32>, expected_val: Int32, reset_val: Int32, sem: MemoryOrder, scope: MemoryScope) {
    let opts = AtomicOptions::new(sem, scope);
    // As CuTe DSL's loop: the result starts at 0, so an expected value of 0 never spins.
    let ptr = lock_ptr.clone();
    let _ = while_carry(
        Int32::from(0),
        |r: Int32| -> Boolean { r.ne(expected_val) },
        |_r: Int32| atomic_cas_with(&ptr, expected_val, reset_val, opts),
    );
}

/// CuTe DSL's `spin_lock_atom_cas_relaxed_wait`: spin on a relaxed `atom.cas` (scope Gpu or
/// Sys) until the flag holds `expected_val`, which it then replaces with `reset_val`.
#[track_caller]
pub fn spin_lock_atom_cas_relaxed_wait(
    lock_ptr: &Ptr<i32>,
    expected_val: impl IntoValue<i32>,
    reset_val: impl IntoValue<i32>,
    scope: MemoryScope,
) {
    order_scope("spin_lock_atom_cas_relaxed_wait", MemoryOrder::Relaxed, scope);
    spin_cas(lock_ptr, expected_val.into_value(), reset_val.into_value(), MemoryOrder::Relaxed, scope);
}

/// CuTe DSL's `spin_lock_atom_cas_acquire_wait`: [`spin_lock_atom_cas_relaxed_wait`] with an
/// acquire `atom.cas`, so that what the flag's writers released is visible after it.
#[track_caller]
pub fn spin_lock_atom_cas_acquire_wait(
    lock_ptr: &Ptr<i32>,
    expected_val: impl IntoValue<i32>,
    reset_val: impl IntoValue<i32>,
    scope: MemoryScope,
) {
    order_scope("spin_lock_atom_cas_acquire_wait", MemoryOrder::Relaxed, scope);
    spin_cas(lock_ptr, expected_val.into_value(), reset_val.into_value(), MemoryOrder::Acquire, scope);
}

/// CuTe DSL's `spin_lock_ld_lt_relaxed_wait`: spin on `ld.relaxed.{scope}` of the flag while it
/// is less than `expected_val`.
#[track_caller]
pub fn spin_lock_ld_lt_relaxed_wait(lock_ptr: &Ptr<i32>, expected_val: impl IntoValue<i32>, scope: MemoryScope) {
    order_scope("spin_lock_ld_lt_relaxed_wait", MemoryOrder::Relaxed, scope);
    let expected: Int32 = expected_val.into_value();
    let opts = LoadOptions { sem: MemoryOrder::Relaxed, scope: Some(scope), ..LoadOptions::default() };
    // As CuTe DSL's loop: the value starts at 0, so an expected value of 0 or less never spins.
    let ptr = lock_ptr.clone();
    let _ = while_carry(Int32::from(0), |v: Int32| -> Boolean { v.lt(expected) }, |_v: Int32| load(&ptr, opts));
}

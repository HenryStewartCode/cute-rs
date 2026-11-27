//! Hardware intrinsics (CuTe DSL's `cute.arch`): thread and block indices,
//! barriers, the `cp.async` group operations, warp-collective operations
//! (shuffles, butterfly reductions, `redux.sync`, votes, `match.sync`),
//! atomics, fences, timers, `printf`, the integer and bit instructions
//! (`bfe`, `bfi`, `lop3`, `shf`, `prmt`, `mul_hi`, `mul_wide`, `mul24`,
//! `mad24`, `add_sat_int`, `sub_sat_int`), the bit counts (`clz`, `popc`,
//! `brev`, `bfind`, `log2_of_pow2_int`), `fmax` and `fmin`, the
//! extended-precision arithmetic (`add_cc`, `addc`, `sub_cc`, `subc`,
//! `mad_cc`, `madc` and the carry-in-and-out forms), the shared-memory size
//! registers, a block's rank in its cluster, `cp.async` copies that an
//! mbarrier tracks, loads and stores with memory ordering and cache hints
//! (`load`, `store`) and `prefetch`, hand-written PTX and MLIR
//! ([`InlinePtx`], [`InlineAsm`], [`MlirOp`], and the positional
//! [`inline_ptx`]), the packed and scalar half-precision operations
//! ([`add_packed_f16x2`], [`fma_bf16`], [`exp2_packed_f16x2`], ...) and the
//! narrow conversions ([`cvt_f32x2_bf16x2`], [`cvt_f32_tf32`],
//! [`cvt_i8x4_to_f32x4`], [`cvt_i8_bf16_intrinsic`],
//! [`sext_unpacked_i4x4_to_i8x4`], ...).
//!
//! A custom operation (CuTe DSL's `@cutlass.dsl_user_op`) is a Rust function
//! marked `#[track_caller]` that builds operations from these: each records
//! the line that called the function. The [`inline`] module has the recipe.

mod atomic;
mod cvt;
mod half;
// Packed arithmetic.
mod packed;
// Tensor memory.
mod tmem;
// Cluster launch control.
mod clc;

pub use atomic::*;
pub use clc::{clc_response, issue_clc_query};
pub use cvt::*;
pub use half::*;
pub use packed::*;
pub use tmem::*;

use std::panic::Location;

use crate::builder::{self, LocationScope};
use crate::ir::{
    AsmInput, AsmSim, BarrierKind, BulkCompletion, BulkDirection, CacheEvictionPriority, CacheOperator, CarryFn,
    CarryOp, DType, Dim, FenceScope, FloatFn, FloatMode, InlineSim, IntFn, L2PrefetchSize, LaneMask, MatchKind,
    MbarrierWaitKind, MemAttrs, MemScopeKind, MemoryOrder, MemoryScope, Op, PrefetchCacheLevel, PrmtMode, ProxyKind,
    ReduxKind, ShuffleKind, SpecialReg, ValueId, VoteKind, asm_register,
};
use crate::layout::IndexExpr;
use crate::tensor::{AddressSpace, Ptr, load_vector_with, store_vector_with};
use crate::value::{
    AnyValue, Arith, Bitwise, Boolean, Float32, Int32, Int64, Integer, IntoValue, Numeric, Uint32, Value, select,
};

/// Threads per warp (CuTe DSL's `cute.arch.WARP_SIZE`).
pub const WARP_SIZE: usize = 32;

/// Warps per warpgroup (`cute.arch.WARPS_PER_WARPGROUP`).
pub const WARPS_PER_WARPGROUP: usize = 4;

/// Threads per warpgroup (`cute.arch.THREADS_PER_WARPGROUP`).
pub const THREADS_PER_WARPGROUP: usize = WARP_SIZE * WARPS_PER_WARPGROUP;

/// The lane mask of a whole warp (`FULL_MASK`).
pub const FULL_MASK: u32 = 0xffff_ffff;

/// A 3-D index or extent.
#[derive(Clone, Copy, Debug)]
pub struct Dim3 {
    pub x: Int32,
    pub y: Int32,
    pub z: Int32,
}

#[track_caller]
fn special(reg: SpecialReg) -> Int32 {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| b.special(reg, loc)))
}

#[track_caller]
fn dim3(f: impl Fn(Dim) -> SpecialReg) -> Dim3 {
    Dim3 { x: special(f(Dim::X)), y: special(f(Dim::Y)), z: special(f(Dim::Z)) }
}

/// `threadIdx`.
#[track_caller]
pub fn thread_idx() -> Dim3 {
    dim3(SpecialReg::ThreadIdx)
}

/// `blockIdx`.
#[track_caller]
pub fn block_idx() -> Dim3 {
    dim3(SpecialReg::BlockIdx)
}

/// `blockDim`. Constant when the kernel declares its block size.
#[track_caller]
pub fn block_dim() -> Dim3 {
    match builder::with(|b| b.block_dim) {
        Some([x, y, z]) => Dim3 { x: (x as i32).into_value(), y: (y as i32).into_value(), z: (z as i32).into_value() },
        None => dim3(SpecialReg::BlockDim),
    }
}

/// `gridDim`.
#[track_caller]
pub fn grid_dim() -> Dim3 {
    dim3(SpecialReg::GridDim)
}

/// Linear thread index within the block: `x + y * bdx + z * bdx * bdy`.
#[track_caller]
pub fn linear_thread_idx() -> Int32 {
    let t = thread_idx();
    let d = block_dim();
    let (x, y, z) = (IndexExpr::from(t.x), IndexExpr::from(t.y), IndexExpr::from(t.z));
    match (d.x.as_constant(), d.y.as_constant()) {
        (Some(dx), Some(dy)) => (x + y * dx as usize + z * (dx * dy) as usize).into_value(),
        _ => t.x + t.y * d.x + t.z * d.x * d.y,
    }
}

/// Lane index within the warp (`%laneid`).
#[track_caller]
pub fn lane_idx() -> Int32 {
    special(SpecialReg::LaneId)
}

/// Warp index within the block, marked warp-uniform as in CuTe DSL.
#[track_caller]
pub fn warp_idx() -> Int32 {
    make_warp_uniform((IndexExpr::from(linear_thread_idx()) / WARP_SIZE).into_value())
}

/// Threads per warp (CuTe DSL's `warpsize`): PTX's predefined constant
/// `WARP_SZ`, read through `llvm.nvvm.read.ptx.sreg.warpsize`.
/// [`WARP_SIZE`] is the same number at build time.
#[track_caller]
pub fn warpsize() -> Int32 {
    special(SpecialReg::WarpSize)
}

/// `value`, promised to be the same in every lane of the warp (CuTe DSL's
/// `make_warp_uniform`), such as a warp index used to pick a warp's role.
/// CuTe DSL lowers it to a compiler hint; upstream LLVM has no such hint, so
/// the generated code is just `value`. The simulator checks the promise.
#[track_caller]
pub fn make_warp_uniform(value: impl IntoValue<i32>) -> Int32 {
    let loc = Location::caller();
    let value = value.into_value();
    if value.as_constant().is_some() {
        return value;
    }
    let value = value.checked();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::I32);
        b.push(Op::WarpUniform { result, value }, loc);
        let facts = b.facts_of(value);
        b.set_facts(result, facts);
        result
    }))
}

/// Block-wide barrier (`__syncthreads()`). Must be reached by every thread
/// of the block.
#[track_caller]
pub fn sync_threads() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Barrier, loc));
}

#[track_caller]
pub(crate) fn named_barrier(kind: BarrierKind, barrier_id: Int32, number_of_threads: Option<Int32>) {
    let loc = Location::caller();
    let (id, threads) = (barrier_id.checked(), number_of_threads.map(|n| n.checked()));
    builder::with(|b| b.push(Op::NamedBarrier { kind, id, threads }, loc));
}

/// Wait at named barrier `barrier_id` (0 to 15) until `number_of_threads`
/// threads (a multiple of 32, counted by whole warps) have arrived at it
/// (`bar.sync id, count`, CuTe DSL's `cute.arch.barrier(barrier_id=,
/// number_of_threads=)`). Lets a subset of warps synchronise, e.g. 4 of 8
/// warps with `barrier(1, 128)` inside `if warp_idx < 4`. Every thread of
/// each participating warp must execute it; barrier 0 is the one
/// [`sync_threads`] uses. Without a thread count (every thread of the CTA)
/// it is [`pipeline::sync`](crate::pipeline::sync)`(barrier_id)`.
#[track_caller]
pub fn barrier(barrier_id: impl IntoValue<i32>, number_of_threads: impl IntoValue<i32>) {
    named_barrier(BarrierKind::Sync, barrier_id.into_value(), Some(number_of_threads.into_value()));
}

/// Arrive at named barrier `barrier_id` without waiting for it to complete
/// (`bar.arrive id, count`, CuTe DSL's `cute.arch.barrier_arrive` with its
/// default `aligned=True`): the producer side of a producer/consumer
/// hand-off whose consumers call [`barrier`] with the same id and thread
/// count. Every thread of the CTA must reach the same `bar.arrive`
/// instruction, so it may not sit in divergent code; see
/// [`barrier_arrive_unaligned`].
#[track_caller]
pub fn barrier_arrive(barrier_id: impl IntoValue<i32>, number_of_threads: impl IntoValue<i32>) {
    named_barrier(BarrierKind::Arrive, barrier_id.into_value(), Some(number_of_threads.into_value()));
}

/// [`barrier_arrive`] without `.aligned` (`barrier.arrive id, count`, CuTe
/// DSL's `barrier_arrive(..., aligned=False)`), for arrivals by a subset of
/// the CTA's warps that do not all execute the same instruction.
#[track_caller]
pub fn barrier_arrive_unaligned(barrier_id: impl IntoValue<i32>, number_of_threads: impl IntoValue<i32>) {
    named_barrier(BarrierKind::ArriveUnaligned, barrier_id.into_value(), Some(number_of_threads.into_value()));
}

/// Close the current group of `cp.async` copies issued by this thread.
#[track_caller]
pub fn cp_async_commit_group() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::CpAsyncCommitGroup, loc));
}

/// Wait until at most `pending` committed `cp.async` groups of this thread
/// are still in flight. Other threads' copies are only visible after a
/// following [`sync_threads`].
#[track_caller]
pub fn cp_async_wait_group(pending: u32) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::CpAsyncWaitGroup { pending }, loc));
}

/// Wait for all of this thread's `cp.async` copies.
#[track_caller]
pub fn cp_async_wait_all() {
    cp_async_wait_group(0)
}

#[track_caller]
fn special_as<T: Numeric>(reg: SpecialReg) -> Value<T> {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| b.special(reg, loc)))
}

/// The warp slot on its SM (`%warpid`). May change if the warp is preempted.
#[track_caller]
pub fn physical_warp_id() -> Int32 {
    special(SpecialReg::WarpId)
}

/// Warps per SM (`%nwarpid`).
#[track_caller]
pub fn nwarpid() -> Int32 {
    special(SpecialReg::NWarpId)
}

/// The SM this thread runs on (`%smid`).
#[track_caller]
pub fn smid() -> Int32 {
    special(SpecialReg::SmId)
}

/// SMs on the GPU (`%nsmid`).
#[track_caller]
pub fn nsmid() -> Int32 {
    special(SpecialReg::NSmId)
}

/// The launch's grid identifier (`%gridid`).
#[track_caller]
pub fn gridid() -> Int32 {
    special(SpecialReg::GridId)
}

/// The bit of this lane (`%lanemask_eq`).
#[track_caller]
pub fn lanemask_eq() -> Uint32 {
    special_as(SpecialReg::LaneMask(LaneMask::Eq))
}

/// Lanes up to and including this one (`%lanemask_le`).
#[track_caller]
pub fn lanemask_le() -> Uint32 {
    special_as(SpecialReg::LaneMask(LaneMask::Le))
}

/// Lanes below this one (`%lanemask_lt`).
#[track_caller]
pub fn lanemask_lt() -> Uint32 {
    special_as(SpecialReg::LaneMask(LaneMask::Lt))
}

/// This lane and those above it (`%lanemask_ge`).
#[track_caller]
pub fn lanemask_ge() -> Uint32 {
    special_as(SpecialReg::LaneMask(LaneMask::Ge))
}

/// Lanes above this one (`%lanemask_gt`).
#[track_caller]
pub fn lanemask_gt() -> Uint32 {
    special_as(SpecialReg::LaneMask(LaneMask::Gt))
}

/// A 32-bit cycle counter (`%clock`). The simulator counts instructions.
#[track_caller]
pub fn clock() -> Uint32 {
    special_as(SpecialReg::Clock)
}

/// A 64-bit cycle counter (`%clock64`). The simulator counts instructions.
#[track_caller]
pub fn clock64() -> Int64 {
    special_as(SpecialReg::Clock64)
}

/// A nanosecond timer (`%globaltimer`). The simulator counts instructions.
#[track_caller]
pub fn globaltimer() -> Int64 {
    special_as(SpecialReg::GlobalTimer)
}

/// The low 32 bits of the nanosecond timer (`%globaltimer_lo`, CuTe DSL's
/// `cute.arch.globaltimer_lo`). The simulator counts instructions.
#[track_caller]
pub fn globaltimer_lo() -> Int32 {
    special_as(SpecialReg::GlobalTimerLo)
}

/// The block cluster's index in the grid.
#[track_caller]
pub fn cluster_idx() -> Dim3 {
    dim3(SpecialReg::ClusterIdx)
}

/// The grid size in clusters.
#[track_caller]
pub fn cluster_dim() -> Dim3 {
    dim3(SpecialReg::ClusterDim)
}

/// This block's index in its cluster.
#[track_caller]
pub fn block_in_cluster_idx() -> Dim3 {
    dim3(SpecialReg::BlockInClusterIdx)
}

/// The size of a cluster in blocks.
#[track_caller]
pub fn block_in_cluster_dim() -> Dim3 {
    dim3(SpecialReg::BlockInClusterDim)
}

fn all_lanes() -> Uint32 {
    FULL_MASK.into_value()
}

/// Barrier among the lanes of this warp (`__syncwarp`, CuTe DSL's
/// `sync_warp()`): [`sync_warp_with`] for the whole warp.
#[track_caller]
pub fn sync_warp() {
    sync_warp_with(FULL_MASK);
}

/// Barrier among the lanes of this warp in `mask` (`bar.warp.sync mask`,
/// CuTe DSL's `sync_warp(mask)`): each lane waits until every lane of
/// `mask` has reached a `sync_warp` with the same mask, and the lanes'
/// memory accesses before it are ordered before those after it. The
/// executing lane must be in `mask`, and every lane of `mask` must execute
/// it, which the simulator checks (it may sit in divergent code: half a warp
/// can synchronise with `0x0000ffff` while the other half takes another
/// branch).
#[track_caller]
pub fn sync_warp_with(mask: impl IntoValue<u32>) {
    let loc = Location::caller();
    let mask = mask.into_value().checked();
    builder::with(|b| b.push(Op::WarpBarrier { mask }, loc));
}

/// One `shfl.sync` of a 32-bit value.
#[track_caller]
fn shuffle_word<T: Numeric>(kind: ShuffleKind, value: Value<T>, offset: Int32, mask: Uint32, clamp: Int32) -> Value<T> {
    let loc = Location::caller();
    let (value, offset, mask, clamp) = (value.checked(), offset.checked(), mask.checked(), clamp.checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::Shuffle { result, kind, value, offset, mask, clamp }, loc);
        result
    }))
}

/// A value narrower than 32 bits (of the same width as `U`) shuffled as
/// its bits, zero-extended.
#[track_caller]
fn shuffle_narrow<T: Numeric, U: Integer>(value: Value<T>, word: impl Fn(Uint32) -> Uint32) -> Value<T> {
    word(value.bitcast::<U>().to::<u32>()).to::<U>().bitcast()
}

/// Read `value` from another lane of the warp: CuTe DSL's
/// `cute.arch.shuffle_sync_op(value, offset, mask, mask_and_clamp, kind)`
/// (`shfl.sync.{kind}.b32`), of which [`shuffle_sync`], [`shuffle_sync_up`],
/// [`shuffle_sync_down`] and [`shuffle_sync_bfly`] (and their `_with`
/// forms) are the partial applications.
///
/// Each lane computes the lane it reads from `offset` (its bits 4:0, the
/// source lane or a lane offset, as `kind` says) and `mask_and_clamp`,
/// which packs a segment mask into bits 12:8 and a clamp value into bits
/// 4:0. With `lane` the reading lane, `segmask` and `clamp` those fields,
/// `maxLane = (lane & segmask) | (clamp & !segmask)` and `minLane = lane &
/// segmask`, the source is
///
/// * [`ShuffleKind::Idx`]: `minLane | (offset & !segmask)`, if it is at
///   most `maxLane`;
/// * [`ShuffleKind::Up`]: `lane - offset`, if it is at least `maxLane`;
/// * [`ShuffleKind::Down`]: `lane + offset`, if it is at most `maxLane`;
/// * [`ShuffleKind::Bfly`]: `lane ^ offset`, if it is at most `maxLane`;
///
/// and a lane whose source is out of range keeps its own value. So the
/// whole warp is `mask_and_clamp = 31` (and 0 for `Up`), and segments of
/// `width` lanes (a power of two) are `((32 - width) << 8) | (width - 1)`,
/// or `(32 - width) << 8` for `Up`. `mask` names the participating lanes:
/// the executing lane and the lane it reads must be in it, and every lane
/// of it must execute the shuffle (the simulator checks all three).
///
/// Any numeric type moves, CuTe DSL's 8- to 64-bit Numerics: 32-bit values
/// in one `shfl.sync`, narrower ones as their bits zero-extended to 32 bits
/// (and Boolean as 0 or 1), and 64-bit values as two shuffles, of their low
/// and of their high word, as in CuTe DSL. (CuTe DSL converts Float16 and
/// BFloat16 to Float32 and back, which turns a NaN into the canonical NaN;
/// here a NaN keeps its payload.) A 32-bit [`TensorSsa`](crate::TensorSsa)
/// or [`Vector`](crate::Vector) (two Float16, four Int8) moves as one word,
/// bitcast to a Uint32 and back, as CuTe DSL shuffles TensorSSA vectors;
/// any other width panics (see [`Shuffle`]).
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("segments").block_dim([32, 1, 1]).build(|| {
///     let x = param_tensor::<f64>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     // Each lane reads the next lane of its group of 8, wrapping around.
///     let rotated = shuffle_sync_op(x.load(t), (t + 1) % 8, FULL_MASK, (24 << 8) | 7, ShuffleKind::Idx);
///     x.store(t, rotated);
/// });
/// ```
///
/// CuTe DSL refuses values wider than 64 bits (its Int128 and Uint128), and
/// so does cute-rs, when the kernel is built (Int4 values too):
///
/// ```should_panic
/// use cute::prelude::*;
/// Kernel::build("broadcast", || {
///     let x = Uint128::constant(1 << 70);
///     let _ = shuffle_sync(x, 0); // panics: a 128-bit value
/// });
/// ```
#[track_caller]
pub fn shuffle_sync_op<S: Shuffle>(
    value: S,
    offset: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
    mask_and_clamp: impl IntoValue<i32>,
    kind: ShuffleKind,
) -> S {
    let _at = LocationScope::new();
    let (offset, mask, clamp) = (offset.into_value(), mask.into_value(), mask_and_clamp.into_value());
    value.shuffle_with(kind, offset, mask, clamp)
}

/// What the warp shuffles move: a [`Value`] of any numeric type up to 64
/// bits, or a 32-bit [`TensorSsa`](crate::TensorSsa) or
/// [`Vector`](crate::Vector) (two Float16, four Int8, ...), which CuTe DSL
/// shuffles as one word: bitcast to a Uint32, shuffled, and bitcast back to
/// the same shape.
pub trait Shuffle: Sized {
    #[doc(hidden)]
    fn shuffle_with(self, kind: ShuffleKind, offset: Int32, mask: Uint32, clamp: Int32) -> Self;
}

impl<T: Numeric> Shuffle for Value<T> {
    #[track_caller]
    fn shuffle_with(self, kind: ShuffleKind, offset: Int32, mask: Uint32, clamp: Int32) -> Self {
        let value = self;
        let word = |w: Uint32| shuffle_word(kind, w, offset, mask, clamp);
        match T::DTYPE.bits() {
            32 => shuffle_word(kind, value, offset, mask, clamp),
            64 => {
                let bits = value.bitcast::<u64>();
                let (low, high) = (word(bits.to::<u32>()), word((bits >> 32u64).to::<u32>()));
                ((high.to::<u64>() << 32u64) | low.to::<u64>()).bitcast()
            }
            16 => shuffle_narrow::<T, u16>(value, word),
            8 => shuffle_narrow::<T, u8>(value, word),
            1 => Value::from_id(word(value.to::<u32>()).ne(0u32).checked()),
            // Int128 and Uint128 (which CuTe DSL refuses too) and Int4 come
            // here; a 4- or 6-bit float, once there is one, would need its own
            // path.
            bits => panic!(
                "cute: shuffle_sync_op moves Boolean and 8-, 16-, 32- and 64-bit values, not {bits}-bit {}",
                T::DTYPE
            ),
        }
    }
}

/// The bits of `n` elements of `t`, checked to be one 32-bit word.
#[track_caller]
fn check_word(n: usize, t: DType, what: &str) {
    let bits = n * t.bits();
    assert!(
        t != DType::Bool && bits == 32,
        "cute: shuffle_sync_op moves a {what} of 32 bits as one word (CuTe DSL's TensorSSA shuffle), but {n} {} \
         elements are {bits} bits; shuffle the elements one by one, or in 32-bit pieces",
        t.cute_name()
    );
}

impl<T: Numeric> Shuffle for crate::TensorSsa<T> {
    #[track_caller]
    fn shuffle_with(self, kind: ShuffleKind, offset: Int32, mask: Uint32, clamp: Int32) -> Self {
        check_word(self.size(), T::DTYPE, "tensor value");
        let w = shuffle_word(kind, self.bitcast::<u32>().values()[0], offset, mask, clamp);
        crate::TensorSsa::new(1, vec![w]).bitcast::<T>().reshape(self.shape().clone())
    }
}

impl<T: Numeric> Shuffle for crate::Vector<T> {
    #[track_caller]
    fn shuffle_with(self, kind: ShuffleKind, offset: Int32, mask: Uint32, clamp: Int32) -> Self {
        check_word(self.size(), T::DTYPE, "vector");
        let w = shuffle_word(kind, self.flatten().bitcast::<u32>().values()[0], offset, mask, clamp);
        crate::Vector::new(1, vec![w]).bitcast::<T>().reshape(self.shape())
    }
}

/// `value` as held by lane `lane` of this warp (`__shfl_sync`, CuTe DSL's
/// `shuffle_sync`): [`shuffle_sync_op`] of kind [`ShuffleKind::Idx`] over
/// the whole warp.
#[track_caller]
pub fn shuffle_sync<S: Shuffle>(value: S, lane: impl IntoValue<i32>) -> S {
    shuffle_sync_op(value, lane, FULL_MASK, WARP_SIZE as i32 - 1, ShuffleKind::Idx)
}

/// `value` from `offset` lanes below (`__shfl_up_sync`, CuTe DSL's
/// `shuffle_sync_up`); lanes below `offset` keep their own.
#[track_caller]
pub fn shuffle_sync_up<S: Shuffle>(value: S, offset: impl IntoValue<i32>) -> S {
    shuffle_sync_op(value, offset, FULL_MASK, 0, ShuffleKind::Up)
}

/// `value` from `offset` lanes above (`__shfl_down_sync`, CuTe DSL's
/// `shuffle_sync_down`); lanes without one keep their own.
#[track_caller]
pub fn shuffle_sync_down<S: Shuffle>(value: S, offset: impl IntoValue<i32>) -> S {
    shuffle_sync_op(value, offset, FULL_MASK, WARP_SIZE as i32 - 1, ShuffleKind::Down)
}

/// `value` from lane `lane ^ offset` (`__shfl_xor_sync`, CuTe DSL's
/// `shuffle_sync_bfly`).
#[track_caller]
pub fn shuffle_sync_bfly<S: Shuffle>(value: S, offset: impl IntoValue<i32>) -> S {
    shuffle_sync_op(value, offset, FULL_MASK, WARP_SIZE as i32 - 1, ShuffleKind::Bfly)
}

/// [`shuffle_sync`] with CuTe DSL's `mask=` and `mask_and_clamp=` (see
/// [`shuffle_sync_op`]).
#[track_caller]
pub fn shuffle_sync_with<S: Shuffle>(
    value: S,
    lane: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
    mask_and_clamp: impl IntoValue<i32>,
) -> S {
    shuffle_sync_op(value, lane, mask, mask_and_clamp, ShuffleKind::Idx)
}

/// [`shuffle_sync_up`] with CuTe DSL's `mask=` and `mask_and_clamp=` (see
/// [`shuffle_sync_op`]).
#[track_caller]
pub fn shuffle_sync_up_with<S: Shuffle>(
    value: S,
    offset: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
    mask_and_clamp: impl IntoValue<i32>,
) -> S {
    shuffle_sync_op(value, offset, mask, mask_and_clamp, ShuffleKind::Up)
}

/// [`shuffle_sync_down`] with CuTe DSL's `mask=` and `mask_and_clamp=` (see
/// [`shuffle_sync_op`]).
#[track_caller]
pub fn shuffle_sync_down_with<S: Shuffle>(
    value: S,
    offset: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
    mask_and_clamp: impl IntoValue<i32>,
) -> S {
    shuffle_sync_op(value, offset, mask, mask_and_clamp, ShuffleKind::Down)
}

/// [`shuffle_sync_bfly`] with CuTe DSL's `mask=` and `mask_and_clamp=` (see
/// [`shuffle_sync_op`]).
#[track_caller]
pub fn shuffle_sync_bfly_with<S: Shuffle>(
    value: S,
    offset: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
    mask_and_clamp: impl IntoValue<i32>,
) -> S {
    shuffle_sync_op(value, offset, mask, mask_and_clamp, ShuffleKind::Bfly)
}

/// A reduction of `val` across each group of `threads_in_group` lanes of
/// the warp with the binary operation `op` (CuTe DSL's
/// `cute.arch.warp_reduction(val, op, threads_in_group=)`): a butterfly of
/// [`shuffle_sync_bfly`]s unrolled at build time, with offsets from
/// `threads_in_group / 2` halving down to 1, so that every lane of a group
/// gets the group's result. `threads_in_group` is a power of two from 1 to
/// [`WARP_SIZE`] (the whole warp); groups are aligned (lanes 0 to 7, 8 to
/// 15, ... for 8), since the offsets below the group size stay inside it.
/// Each lane combines its value with its partner's as `op(own, partner)`,
/// so a commutative `op` gives every lane the same result.
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("row_sums").block_dim([32, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     // Four sums of 8 lanes each, as in CuTe DSL's `threads_in_group=8`.
///     x.store(t, warp_reduction(x.load(t), |a, b| a + b, 8));
/// });
/// ```
#[track_caller]
pub fn warp_reduction<T: Numeric>(
    val: Value<T>,
    op: impl Fn(Value<T>, Value<T>) -> Value<T>,
    threads_in_group: usize,
) -> Value<T> {
    assert!(
        threads_in_group.is_power_of_two() && threads_in_group <= WARP_SIZE,
        "cute: warp_reduction: threads_in_group is {threads_in_group}, but it must be a power of two from 1 to 32"
    );
    let _at = LocationScope::new();
    let mut val = val;
    let mut offset = threads_in_group / 2;
    while offset > 0 {
        val = op(val, shuffle_sync_op(val, offset, FULL_MASK, WARP_SIZE as i32 - 1, ShuffleKind::Bfly));
        offset /= 2;
    }
    val
}

/// The largest `val` of each group of `threads_in_group` lanes (CuTe DSL's
/// `warp_reduction_max`): [`warp_reduction`] with CuTe DSL's operator, which
/// depends on the type, as there:
///
/// * Float32: [`fmax`] with its default modes (`max.f32`): a NaN is
///   ignored, and `+0 > -0`;
/// * Float16, BFloat16 and Float64: CuTe DSL's `cutlass.max`, which on them
///   is `arith.maximumf`, IEEE 754's `maximum`: a NaN in any lane of the
///   group makes the result NaN, and `-0 < +0`. Here it is
///   [`math::max_with`](crate::math::max_with) with `propagate_nan`, so the
///   NaN is the quiet NaN (`0x7e00`, `0x7fc0`, `0x7ff8000000000000`; MLIR
///   leaves its bits open);
/// * integers: the signed or unsigned maximum.
///
/// A group of one lane returns its value unchanged.
#[track_caller]
pub fn warp_reduction_max<T: Arith>(val: Value<T>, threads_in_group: usize) -> Value<T> {
    if T::DTYPE.is_float() && T::DTYPE != DType::F32 {
        let maximum = crate::math::MathOptions { propagate_nan: true, ..Default::default() };
        return warp_reduction(val, |a, b| crate::math::max_with(a, b, maximum), threads_in_group);
    }
    warp_reduction(val, |a, b| a.max(b), threads_in_group)
}

/// The sum of `val` over each group of `threads_in_group` lanes (CuTe DSL's
/// `warp_reduction_sum`): [`warp_reduction`] with `+`. Floats are added in
/// the butterfly's order, so the rounding differs from a sequential sum.
#[track_caller]
pub fn warp_reduction_sum<T: Arith>(val: Value<T>, threads_in_group: usize) -> Value<T> {
    warp_reduction(val, |a, b| a + b, threads_in_group)
}

#[track_caller]
fn vote(kind: VoteKind, pred: Boolean, ty: DType) -> crate::ir::ValueId {
    let loc = Location::caller();
    let (pred, mask) = (pred.checked(), all_lanes().checked());
    builder::with(|b| {
        let result = b.new_value(ty);
        b.push(Op::Vote { result, kind, pred, mask }, loc);
        result
    })
}

/// Whether `pred` holds in every active lane (`__all_sync`).
#[track_caller]
pub fn vote_all_sync(pred: Boolean) -> Boolean {
    Value::from_id(vote(VoteKind::All, pred, DType::Bool))
}

/// Whether `pred` holds in any active lane (`__any_sync`).
#[track_caller]
pub fn vote_any_sync(pred: Boolean) -> Boolean {
    Value::from_id(vote(VoteKind::Any, pred, DType::Bool))
}

/// Whether `pred` has the same value in every active lane (`__uni_sync`).
#[track_caller]
pub fn vote_uni_sync(pred: Boolean) -> Boolean {
    Value::from_id(vote(VoteKind::Uni, pred, DType::Bool))
}

/// A mask of the lanes where `pred` holds (`__ballot_sync`).
#[track_caller]
pub fn vote_ballot_sync(pred: Boolean) -> Uint32 {
    Value::from_id(vote(VoteKind::Ballot, pred, DType::U32))
}

/// Compare `value` across the lanes of the warp in `mask` (CuTe DSL's
/// `cute.arch.match_sync(mask, value, kind)`, `match.{any,all}.sync`): with
/// [`MatchKind::Any`], the mask of the lanes in `mask` holding the same
/// value as this one (`__match_any_sync`); with [`MatchKind::All`], `mask`
/// (less any lanes the warp lacks) if all of them hold the same value, and
/// 0 otherwise (`__match_all_sync`; CuTe DSL drops the instruction's
/// predicate, which is that result being nonzero). `value` is an Int32,
/// Uint32, Int64 or Uint64, compared bit for bit. The executing lane must be
/// in `mask`, and every lane of `mask` must execute it, which the
/// simulator checks.
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("same_keys").block_dim([32, 1, 1]).build(|| {
///     let keys = param_tensor::<u64>("keys", Layout::col_major(32));
///     let peers = param_tensor::<u32>("peers", Layout::col_major(32));
///     let t = thread_idx().x;
///     peers.store(t, match_sync(FULL_MASK, keys.load(t), MatchKind::Any));
/// });
/// ```
///
/// Other types do not compile (CuTe DSL converts them to Int32):
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::builder("same_floats").block_dim([32, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     let _ = match_sync(FULL_MASK, x.load(t), MatchKind::Any);
/// });
/// ```
#[track_caller]
pub fn match_sync<T: Word>(mask: impl IntoValue<u32>, value: impl IntoValue<T>, kind: MatchKind) -> Uint32 {
    let loc = Location::caller();
    let (mask, value) = (mask.into_value().checked(), value.into_value().checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::U32);
        b.push(Op::Match { result, kind, value, mask }, loc);
        result
    }))
}

/// True in exactly one active lane of the warp.
#[track_caller]
pub fn elect_one() -> Boolean {
    let loc = Location::caller();
    let mask = all_lanes().checked();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::Bool);
        b.push(Op::ElectOne { result, mask }, loc);
        result
    }))
}

/// Wait until every prerequisite grid of a programmatic dependent launch has completed and
/// its memory is visible (`griddepcontrol.wait`; CuTe DSL's `griddepcontrol_wait`; sm_90 and
/// PTX 7.8). A no-op unless the kernel was launched with
/// [`LaunchConfig::use_pdl`](crate::LaunchConfig::use_pdl); [`SimStream`](crate::SimStream)
/// checks that nothing before it touches what the previous kernel writes.
#[track_caller]
pub fn griddepcontrol_wait() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::GridDepControl { launch_dependents: false }, loc));
}

/// `setmaxnreg` with a count checked when the kernel is built.
#[track_caller]
fn setmaxnreg(increase: bool, reg_count: u32, func: &str) {
    let loc = Location::caller();
    if !(24..=256).contains(&reg_count) || !reg_count.is_multiple_of(8) {
        crate::builder::user_panic(
            loc,
            format!("cute: {func}({reg_count}): the register count is a multiple of 8 from 24 to 256"),
        );
    }
    builder::with(|b| b.push(Op::SetMaxNReg { increase, count: reg_count }, loc));
}

/// Raise the per-thread register budget of the calling warpgroup to `reg_count` (CuTe DSL's
/// `setmaxregister_increase`; `setmaxnreg.inc.sync.aligned.u32`, sm_90a, sm_100a and
/// sm_120a, and the f families from PTX 8.8). `reg_count` is a multiple of 8 from 24 to
/// 256, at least the current budget. All four warps of the warpgroup execute it together,
/// and it waits until the CTA's pool has the registers, which other warpgroups release
/// with [`setmaxregister_decrease`]; registers it gains hold undefined values. Separate
/// consecutive ones of a warpgroup with a barrier. It takes effect only in a kernel with a
/// launch bound (a declared block or `max_threads`).
///
/// The simulator starts every warpgroup at
/// [`setmaxnreg_initial_budget`](crate::sim::setmaxnreg_initial_budget) registers per
/// thread with an empty pool, and reports an increase that can never be served as a
/// deadlock naming its warps.
#[track_caller]
pub fn setmaxregister_increase(reg_count: u32) {
    setmaxnreg(true, reg_count, "setmaxregister_increase");
}

/// Lower the per-thread register budget of the calling warpgroup to `reg_count`, returning
/// the difference to the CTA's pool (CuTe DSL's `setmaxregister_decrease`;
/// `setmaxnreg.dec.sync.aligned.u32`). `reg_count` is at most the current budget; the rest
/// is as [`setmaxregister_increase`].
#[track_caller]
pub fn setmaxregister_decrease(reg_count: u32) {
    setmaxnreg(false, reg_count, "setmaxregister_decrease");
}

/// The deprecated name of [`setmaxregister_increase`] (CuTe DSL's `warpgroup_reg_alloc`).
#[deprecated(note = "use setmaxregister_increase")]
#[track_caller]
pub fn warpgroup_reg_alloc(reg_count: u32) {
    setmaxnreg(true, reg_count, "warpgroup_reg_alloc");
}

/// The deprecated name of [`setmaxregister_decrease`] (CuTe DSL's `warpgroup_reg_dealloc`).
#[deprecated(note = "use setmaxregister_decrease")]
#[track_caller]
pub fn warpgroup_reg_dealloc(reg_count: u32) {
    setmaxnreg(false, reg_count, "warpgroup_reg_dealloc");
}

/// Let the dependent grid of a programmatic dependent launch be scheduled once every CTA of
/// this grid has issued this or exited (`griddepcontrol.launch_dependents`; CuTe DSL's
/// `griddepcontrol_launch_dependents`; sm_90 and PTX 7.8). A scheduling hint with no effect
/// on correctness, and a no-op without [`LaunchConfig::use_pdl`](crate::LaunchConfig::use_pdl).
#[track_caller]
pub fn griddepcontrol_launch_dependents() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::GridDepControl { launch_dependents: true }, loc));
}

/// A mask of the active lanes of this warp (`__activemask`).
#[track_caller]
pub fn activemask() -> Uint32 {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::U32);
        b.push(Op::ActiveMask { result }, loc);
        result
    }))
}

/// Initialise an mbarrier (a 64-bit word of shared memory, SM80 and newer)
/// to wait for `count` arrivals per phase. One thread initialises it,
/// before a barrier that publishes it to the others.
#[track_caller]
pub fn mbarrier_init(mbarrier: &Ptr<u64>, count: impl IntoValue<i32>) {
    let loc = Location::caller();
    let (buffer, offset) = mbarrier.parts();
    let index = crate::tensor::index_operand(offset);
    let count = count.into_value().checked();
    builder::with(|b| b.push(Op::MbarrierInit { buffer, index, count }, loc));
}

/// Make this thread's earlier [`mbarrier_init`]s visible to the rest of
/// the cluster and to bulk copies (`fence.mbarrier_init.release.cluster`,
/// SM90 and newer).
#[track_caller]
pub fn mbarrier_init_fence() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::FenceMbarrierInit, loc));
}

/// Arrive at the cluster barrier (`barrier.cluster.arrive`, release
/// semantics at cluster scope; CuTe DSL's `cluster_arrive(aligned=None)`;
/// sm_90 and PTX 7.8). Every thread of the cluster that has not exited
/// arrives once per phase, then waits with [`cluster_wait`]; memory
/// accesses before the arrival (except asynchronous ones) are visible to
/// every thread of the cluster after its wait. The threads of a warp may
/// arrive one by one; [`cluster_arrive_aligned`] is the faster form for a
/// whole warp at once.
#[track_caller]
pub fn cluster_arrive() {
    cluster_arrive_op(false, false);
}

/// [`cluster_arrive`] executed by every thread of the warp together
/// (`barrier.cluster.arrive.aligned`; CuTe DSL's `cluster_arrive(aligned=True)`).
#[track_caller]
pub fn cluster_arrive_aligned() {
    cluster_arrive_op(false, true);
}

/// Arrive at the cluster barrier without ordering memory
/// (`barrier.cluster.arrive.relaxed`; CuTe DSL's
/// `cluster_arrive_relaxed(aligned=None)`; sm_90 and PTX 8.0). A
/// [`mbarrier_init_fence`] before it still publishes mbarrier
/// initialisations to the cluster, as in the PTX ISA's pattern.
#[track_caller]
pub fn cluster_arrive_relaxed() {
    cluster_arrive_op(true, false);
}

/// [`cluster_arrive_relaxed`] executed by every thread of the warp together
/// (`barrier.cluster.arrive.relaxed.aligned`).
#[track_caller]
pub fn cluster_arrive_relaxed_aligned() {
    cluster_arrive_op(true, true);
}

#[track_caller]
fn cluster_arrive_op(relaxed: bool, aligned: bool) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::ClusterArrive { relaxed, aligned }, loc));
}

/// Wait until every thread of the cluster that has not exited has arrived
/// at the cluster barrier in this thread's phase (`barrier.cluster.wait`,
/// acquire semantics at cluster scope; CuTe DSL's `cluster_wait()`; sm_90).
/// The thread must have arrived first; afterwards it may arrive again.
#[track_caller]
pub fn cluster_wait() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::ClusterWait { aligned: false }, loc));
}

/// [`cluster_wait`] executed by every thread of the warp together
/// (`barrier.cluster.wait.aligned`; a cute-rs extension).
#[track_caller]
pub fn cluster_wait_aligned() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::ClusterWait { aligned: true }, loc));
}

/// Synchronise the cluster: [`cluster_arrive_aligned`] then
/// [`cluster_wait_aligned`] (CuTe C++'s `cute::cluster_sync`).
#[track_caller]
pub fn cluster_sync() {
    cluster_arrive_aligned();
    cluster_wait_aligned();
}

/// The keyword arguments of CuTe DSL's mbarrier arrive and expect-tx
/// operations. The default is a release on this CTA's own mbarrier.
#[derive(Clone, Copy, Debug, Default)]
pub struct MbarrierOptions {
    /// Operate on the mbarrier at the same shared-memory offset in this CTA
    /// of the cluster (by `%cluster_ctarank`), through `mapa` (SM90).
    pub peer_cta_rank_in_cluster: Option<Int32>,
    /// Arrive with no ordering or visibility guarantees (`.relaxed`, SM90).
    /// `mbarrier.expect_tx` never orders memory, so it ignores this.
    pub relaxed: bool,
    /// The threads the operation synchronises with: the CTA (`None` and the
    /// default for arrivals, even on a peer) or the cluster (SM90).
    /// `mbarrier_expect_tx` defaults to the cluster for a peer.
    pub scope: Option<MemScopeKind>,
}

/// The peer CTA of an mbarrier operation: `peer_cta_rank_in_cluster`, or
/// the CTA a [`map_dsmem_ptr`] pointer to the mbarrier points into (as
/// CuTe DSL's examples arrive on `mapa`-ed mbarriers).
#[track_caller]
fn mbarrier_peer(mapped: Option<ValueId>, opts: MbarrierOptions, what: &str) -> Option<ValueId> {
    match (mapped, opts.peer_cta_rank_in_cluster) {
        (Some(_), Some(_)) => panic!(
            "cute: {what}: the mbarrier pointer is already mapped to another CTA (map_dsmem_ptr), so \
             peer_cta_rank_in_cluster must not be given too"
        ),
        (Some(p), None) => Some(p),
        (None, p) => p.map(|p| p.checked()),
    }
}

#[track_caller]
fn mbarrier_arrive_impl(mbarrier: &Ptr<u64>, count: Option<Int32>, expect_tx: Option<Int32>, opts: MbarrierOptions) {
    let loc = Location::caller();
    let (buffer, offset, mapped) = mbarrier.parts_dsmem();
    let index = crate::tensor::index_operand(offset);
    let (count, expect_tx) = (count.map(|c| c.checked()), expect_tx.map(|t| t.checked()));
    let peer = mbarrier_peer(mapped, opts, "mbarrier_arrive");
    let (relaxed, scope) = (opts.relaxed, opts.scope.unwrap_or_default());
    builder::with(|b| b.push(Op::MbarrierArrive { buffer, index, count, expect_tx, peer, relaxed, scope }, loc));
}

/// Arrive at an mbarrier (one arrival, release semantics; SM80 and newer).
#[track_caller]
pub fn mbarrier_arrive(mbarrier: &Ptr<u64>) {
    mbarrier_arrive_impl(mbarrier, None, None, MbarrierOptions::default());
}

/// Arrive `arrive_count` times at an mbarrier, with CuTe DSL's options
/// (`mbarrier_arrive(ptr, peer_cta_rank_in_cluster, arrive_count, relaxed=,
/// scope=)`). A count other than a constant 1, a peer CTA, relaxed
/// semantics and cluster scope need SM90.
#[track_caller]
pub fn mbarrier_arrive_with(mbarrier: &Ptr<u64>, arrive_count: impl IntoValue<i32>, opts: MbarrierOptions) {
    let count = arrive_count.into_value();
    let count = if count.as_constant() == Some(1) { None } else { Some(count) };
    mbarrier_arrive_impl(mbarrier, count, None, opts);
}

/// Arrive at an mbarrier, and make this phase wait for `bytes` more bytes of
/// bulk copies (`mbarrier.arrive.expect_tx`, SM90 and newer).
#[track_caller]
pub fn mbarrier_arrive_and_expect_tx(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>) {
    mbarrier_arrive_impl(mbarrier, None, Some(bytes.into_value()), MbarrierOptions::default());
}

/// [`mbarrier_arrive_and_expect_tx`] with CuTe DSL's options: a peer CTA,
/// relaxed semantics or cluster scope.
#[track_caller]
pub fn mbarrier_arrive_and_expect_tx_with(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>, opts: MbarrierOptions) {
    mbarrier_arrive_impl(mbarrier, None, Some(bytes.into_value()), opts);
}

/// Make the current phase of an mbarrier wait for `bytes` more bytes of bulk
/// copies, without arriving (`mbarrier.expect_tx`, SM90 and newer).
#[track_caller]
pub fn mbarrier_expect_tx(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>) {
    mbarrier_expect_tx_with(mbarrier, bytes, MbarrierOptions::default());
}

/// [`mbarrier_expect_tx`] on a peer CTA's mbarrier or with an explicit scope.
#[track_caller]
pub fn mbarrier_expect_tx_with(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>, opts: MbarrierOptions) {
    let loc = Location::caller();
    let (buffer, offset, mapped) = mbarrier.parts_dsmem();
    let index = crate::tensor::index_operand(offset);
    let bytes = bytes.into_value().checked();
    let peer = mbarrier_peer(mapped, opts, "mbarrier_expect_tx");
    let scope = opts.scope.unwrap_or(if peer.is_some() { MemScopeKind::Cluster } else { MemScopeKind::Cta });
    builder::with(|b| b.push(Op::MbarrierExpectTx { buffer, index, bytes, peer, scope }, loc));
}

/// An mbarrier wait recorded at `loc` (the user's call, also from inside the
/// closures of the waiting loops).
fn mbarrier_wait_op(mbarrier: &Ptr<u64>, parity: Int32, kind: MbarrierWaitKind, loc: crate::ir::Loc) -> Boolean {
    let (buffer, offset) = mbarrier.parts();
    let index = crate::tensor::index_operand(offset);
    let parity = parity.checked();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::Bool);
        b.push(Op::MbarrierWait { result, buffer, index, parity, kind, state: false, time_limit: None }, loc);
        result
    }))
}

/// Whether the phase with `parity` (0 or 1) of an mbarrier is over; may
/// suspend the thread for a while first (`mbarrier.try_wait.parity`, SM90
/// and newer).
#[track_caller]
pub fn mbarrier_try_wait(mbarrier: &Ptr<u64>, parity: impl IntoValue<i32>) -> Boolean {
    mbarrier_wait_op(mbarrier, parity.into_value(), MbarrierWaitKind::Try, Location::caller())
}

/// Whether the phase with `parity` (0 or 1) of an mbarrier is over, without
/// waiting (`mbarrier.test_wait.parity`, SM80 and newer). Spin on it to wait
/// on SM80.
#[track_caller]
pub fn mbarrier_test_wait(mbarrier: &Ptr<u64>, parity: impl IntoValue<i32>) -> Boolean {
    mbarrier_wait_op(mbarrier, parity.into_value(), MbarrierWaitKind::Test, Location::caller())
}

/// [`mbarrier_try_wait`] when `cond` holds, otherwise true without looking
/// at the mbarrier (CuTe DSL's `mbarrier_conditional_try_wait`).
#[track_caller]
pub fn mbarrier_conditional_try_wait(cond: Boolean, mbarrier: &Ptr<u64>, parity: impl IntoValue<i32>) -> Boolean {
    let (loc, parity) = (Location::caller(), parity.into_value());
    crate::control::if_else_yield(
        cond,
        || mbarrier_wait_op(mbarrier, parity, MbarrierWaitKind::Try, loc),
        || true.into_value(),
    )
}

/// Spin until the phase with `parity` of an mbarrier is over, with
/// `mbarrier.try_wait` (SM90 and newer, as in CuTe DSL; on SM80 loop on
/// [`mbarrier_test_wait`]).
#[track_caller]
pub fn mbarrier_wait(mbarrier: &Ptr<u64>, parity: impl IntoValue<i32>) {
    let (loc, parity) = (Location::caller(), parity.into_value());
    crate::control::while_carry((), |()| !mbarrier_wait_op(mbarrier, parity, MbarrierWaitKind::Try, loc), |()| ());
}

/// Copy `bytes` bytes (a multiple of 16) from global to shared memory in
/// bulk, without tying up a thread per 16 bytes. Completion is reported to
/// `mbarrier` (`cp.async.bulk`, SM90 and newer).
#[track_caller]
pub fn cp_async_bulk<T: Numeric>(src: &Ptr<T>, dst: &Ptr<T>, bytes: impl IntoValue<i32>, mbarrier: &Ptr<u64>) {
    let loc = Location::caller();
    let completion = || {
        let (m, m_off) = mbarrier.parts();
        BulkCompletion::Mbarrier { buffer: m, index: crate::tensor::index_operand(m_off) }
    };
    bulk_copy(src, dst, bytes, BulkDirection::G2S, completion, BulkExtras::default(), loc);
}

/// Copy `bytes` bytes from shared memory at `src` to global memory at `dst`
/// in bulk (`cp.async.bulk.global.shared::cta.bulk_group`, SM90 and newer;
/// the CuTe DSL primitive `cp_async_bulk_global_shared_cta(dst, src,
/// size)`). The size is a multiple of 16 and both addresses are 16-byte
/// aligned: a constant size and static offsets are checked when the kernel
/// is built, runtime ones by the simulator. It is not elected: every thread
/// that executes it issues a copy, so one thread usually does. Completion is
/// tracked by the thread's bulk async-groups: commit it with
/// [`cp_async_bulk_commit_group`], then [`cp_async_bulk_wait_group`]`(n,
/// true)` before overwriting the source, and `(n, false)` before reading
/// the destination. The source must have been written by the async proxy
/// (a bulk copy, a TMA load) or fenced with [`fence_view_async_shared`]
/// after generic writes.
#[track_caller]
pub fn cp_async_bulk_global_shared_cta<T: Numeric>(dst: &Ptr<T>, src: &Ptr<T>, bytes: impl IntoValue<i32>) {
    let loc = Location::caller();
    bulk_copy(src, dst, bytes, BulkDirection::S2G, || BulkCompletion::BulkGroup, BulkExtras::default(), loc);
}

/// The optional operands of a bulk copy: an L2 cache policy
/// (`.L2::cache_hint`), a multicast CTA mask (`.multicast::cluster`) and a
/// peer CTA rank (shared-to-shared copies).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BulkExtras {
    pub cache_policy: Option<ValueId>,
    pub mcast_mask: Option<ValueId>,
    pub peer: Option<ValueId>,
    /// `.cp_mask`.
    pub byte_mask: Option<ValueId>,
}

/// A bulk copy of `bytes` from `src` to `dst`: checks the constant size and
/// the static offsets for 16 bytes (the simulator checks runtime ones). The
/// operands are made in the order the instruction takes them: the
/// addresses, the completion's, then the size.
#[track_caller]
pub(crate) fn bulk_copy<T: Numeric>(
    src: &Ptr<T>,
    dst: &Ptr<T>,
    bytes: impl IntoValue<i32>,
    direction: BulkDirection,
    completion: impl FnOnce() -> BulkCompletion,
    extras: BulkExtras,
    loc: crate::ir::Loc,
) {
    let (s, s_off) = src.parts();
    let (d, d_off) = dst.parts();
    let what = match direction {
        BulkDirection::G2S => "cp.async.bulk",
        BulkDirection::S2G => "cp.async.bulk.global.shared::cta",
        BulkDirection::S2S => "cp.async.bulk.shared::cluster.shared::cta",
    };
    // Indices count elements, and the simulator's byte offsets assume whole
    // bytes per element.
    assert!(
        T::DTYPE.bits() >= 8,
        "cute: {what} of {} elements: sub-byte element types cannot be addressed in memory yet",
        T::DTYPE
    );
    let (src_index, dst_index) =
        (crate::tensor::index_operand(s_off.clone()), crate::tensor::index_operand(d_off.clone()));
    let completion = completion();
    let bytes = bytes.into_value().checked();
    if let Some(n) = builder::try_constant(bytes).map(|c| i64::from(c.bits() as u32 as i32)) {
        assert!(n > 0 && n % 16 == 0, "cute: {what} of {n} bytes: bulk copies move one or more whole 16-byte units");
    }
    for (off, role) in [(&s_off, "source"), (&d_off, "destination")] {
        let at = off.clone() * T::DTYPE.bytes();
        if let Some(b) = at.as_static() {
            assert!(
                b % 16 == 0,
                "cute: {what}: the {role} is at byte {b} of its buffer, which is not 16-byte aligned (bulk copies \
                 need both addresses 16-byte aligned)"
            );
        }
    }
    let BulkExtras { cache_policy, mcast_mask, peer, byte_mask } = extras;
    builder::with(|b| {
        b.push(
            Op::CpAsyncBulk {
                src: s,
                src_index,
                dst: d,
                dst_index,
                elem: T::DTYPE,
                bytes,
                direction,
                completion,
                cache_policy,
                mcast_mask,
                peer,
                byte_mask,
            },
            loc,
        )
    });
}

/// Commit this thread's bulk copies issued since its last commit as a bulk
/// async-group (CuTe DSL's `cute.arch.cp_async_bulk_commit_group`,
/// `cp.async.bulk.commit_group`, SM90 and newer).
#[track_caller]
pub fn cp_async_bulk_commit_group() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::CpAsyncBulkCommitGroup, loc));
}

/// Wait until at most `group` of this thread's most recent bulk
/// async-groups are pending (CuTe DSL's `cute.arch.cp_async_bulk_wait_group(group,
/// read=)`, `cp.async.bulk.wait_group{.read}`, SM90 and newer). With `read`,
/// only until the other groups have read their sources, after which their
/// shared memory may be reused; without it, until their writes are complete
/// and visible to this thread. `group` is an immediate in PTX: it must be
/// known when the kernel is built.
#[track_caller]
pub fn cp_async_bulk_wait_group(group: impl IntoValue<i32>, read: bool) {
    let loc = Location::caller();
    let group = group.into_value();
    let id = group.checked();
    let Some(n) = builder::try_constant(id).map(|c| i64::from(c.bits() as u32 as i32)) else {
        panic!(
            "cp_async_bulk_wait_group: the group count is a runtime value, but cp.async.bulk.wait_group takes an \
             immediate: give a count known when the kernel is built"
        )
    };
    assert!(n >= 0, "cp_async_bulk_wait_group: the group count {n} is negative");
    builder::with(|b| b.push(Op::CpAsyncBulkWaitGroup { pending: n as u32, read }, loc));
}

/// The proxies a [`fence_proxy`] orders: CuTe DSL's `kind` strings
/// `"alias"`, `"async"`, `"async.global"`, `"async.shared"`, `"tensormap"`
/// and `"generic"`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FenceProxyKind {
    /// `"alias"`: `fence.proxy.alias` (sm_70, PTX ISA 7.5).
    Alias,
    /// `"async"`: `fence.proxy.async` (sm_90, PTX ISA 8.0).
    Async,
    /// `"async.global"`: `fence.proxy.async.global` (sm_90).
    AsyncGlobal,
    /// `"async.shared"`: `fence.proxy.async.shared::{cta,cluster}` (sm_90);
    /// needs a space.
    AsyncShared,
    /// `"tensormap"`: refused; see [`fence_proxy`].
    Tensormap,
    /// `"generic"`: refused; see [`fence_proxy`].
    Generic,
}

/// The keyword arguments of CuTe DSL's `fence_proxy(kind, *, space=,
/// use_intrinsic=)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FenceProxyOptions {
    /// `space=`: `cta` or `cluster`, for `async.shared` only.
    pub space: Option<SharedSpace>,
    /// `use_intrinsic=`: accepted and ignored. cute-rs always emits the PTX
    /// instruction (NVVM's operation lowers differently from one LLVM
    /// release to the next).
    pub use_intrinsic: Option<bool>,
}

/// A proxy fence (CuTe DSL's `cute.arch.fence_proxy(kind)`): orders this
/// thread's memory accesses through one proxy against its later accesses
/// through another. [`fence_proxy_with`] takes the space that `AsyncShared`
/// needs.
#[track_caller]
pub fn fence_proxy(kind: FenceProxyKind) {
    fence_proxy_at(kind, FenceProxyOptions::default(), Location::caller());
}

/// [`fence_proxy`] with CuTe DSL's keyword arguments:
/// `fence_proxy_with(FenceProxyKind::AsyncShared, FenceProxyOptions { space:
/// Some(SharedSpace::Cta), ..Default::default() })` is
/// `fence.proxy.async.shared::cta`, which makes this thread's earlier
/// generic writes to the CTA's shared memory visible to the async proxy (a
/// bulk copy or TMA store reading them, a TMA load overwriting them). The
/// `tensormap` and `generic` kinds belong to the tensor-map fences
/// (`fence.proxy.tensormap::generic.acquire/release`), which NVVM's
/// `fence.proxy` refuses: they are refused here, naming
/// `fence_tma_desc_acquire` and `fence_tma_desc_release`.
#[track_caller]
pub fn fence_proxy_with(kind: FenceProxyKind, opts: FenceProxyOptions) {
    fence_proxy_at(kind, opts, Location::caller());
}

#[track_caller]
fn fence_proxy_at(kind: FenceProxyKind, opts: FenceProxyOptions, loc: crate::ir::Loc) {
    let ir_kind = match (kind, opts.space) {
        (FenceProxyKind::Tensormap | FenceProxyKind::Generic, _) => panic!(
            "fence_proxy: the {kind:?} kind is a tensor-map fence (fence.proxy.tensormap::generic.acquire or \
             .release), which fence.proxy does not take: use fence_tma_desc_acquire or fence_tma_desc_release"
        ),
        (FenceProxyKind::AsyncShared, None) => {
            panic!("fence_proxy: the AsyncShared kind needs a space (SharedSpace::Cta or SharedSpace::Cluster)")
        }
        (FenceProxyKind::AsyncShared, Some(SharedSpace::Cta)) => ProxyKind::AsyncSharedCta,
        (FenceProxyKind::AsyncShared, Some(SharedSpace::Cluster)) => ProxyKind::AsyncSharedCluster,
        (_, Some(space)) => panic!("fence_proxy: a space ({space:?}) is for the AsyncShared kind, not {kind:?}"),
        (FenceProxyKind::Alias, None) => ProxyKind::Alias,
        (FenceProxyKind::Async, None) => ProxyKind::Async,
        (FenceProxyKind::AsyncGlobal, None) => ProxyKind::AsyncGlobal,
    };
    builder::with(|b| b.push(Op::FenceProxy { kind: ir_kind }, loc));
}

/// `fence.proxy.async.shared::cta` (CuTe DSL's
/// `cute.arch.fence_view_async_shared()`, SM90 and newer): it goes after
/// generic writes to shared memory and before a bulk copy or TMA store
/// reads them (or UMMA does), in the writing threads, ordered before the
/// issue by a barrier.
#[track_caller]
pub fn fence_view_async_shared() {
    let opts = FenceProxyOptions { space: Some(SharedSpace::Cta), use_intrinsic: None };
    fence_proxy_at(FenceProxyKind::AsyncShared, opts, Location::caller());
}

#[track_caller]
fn fence(scope: FenceScope) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Fence { scope, sc: false }, loc));
}

/// Order this thread's accesses against the rest of its block.
#[track_caller]
pub fn fence_acq_rel_cta() {
    fence(FenceScope::Cta);
}

/// Order this thread's accesses against the rest of its cluster.
#[track_caller]
pub fn fence_acq_rel_cluster() {
    fence(FenceScope::Cluster);
}

/// Order this thread's accesses against the rest of the GPU.
#[track_caller]
pub fn fence_acq_rel_gpu() {
    fence(FenceScope::Gpu);
}

/// Order this thread's accesses against the rest of the system.
#[track_caller]
pub fn fence_acq_rel_sys() {
    fence(FenceScope::Sys);
}

/// Which CTA's shared memory an access addresses: CuTe DSL's
/// `cute.nvgpu.SharedSpace` and the `ss=` of `cute.arch.load` and
/// `cute.arch.store` (`"cta"`, `"cluster"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SharedSpace {
    /// This CTA's (`.shared::cta`).
    #[default]
    Cta,
    /// Any CTA's of the cluster (`.shared::cluster`, distributed shared
    /// memory): a [`map_dsmem_ptr`] pointer.
    Cluster,
}

/// The keyword arguments of CuTe DSL's `cute.arch.load(ptr, dtype, sem=,
/// scope=, level1_eviction_priority=, cop=, ss=, level_prefetch_size=)`.
/// The default is a plain (weak) load. Fields not given keep their
/// defaults: `LoadOptions { sem: MemoryOrder::Acquire, scope:
/// Some(MemoryScope::Gpu), ..Default::default() }`.
///
/// PTX's rules decide what combines (the kernel build reports the rest):
/// a relaxed or acquire load takes an L1 eviction priority, a prefetch size
/// and a cache policy but no cache operator; a weak load takes a cache
/// operator or an eviction priority, not both; a volatile load takes only a
/// prefetch size; an MMIO load is a scalar with system scope and no hints;
/// and the hints other than the cache operator need global memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// `sem=`: weak (the default), `Relaxed` or `Acquire`. cute-rs also takes
    /// `Sc` (`fence.sc` and an acquire load), `Volatile`, `Mmio` and
    /// `Constant` (a non-coherent load, `ld.global.nc`).
    pub sem: MemoryOrder,
    /// `scope=`: whom a relaxed, acquire or seq_cst load synchronises with.
    /// `None` is the system (`.sys`), the scope LLVM gives an atomic access
    /// without one.
    pub scope: Option<MemoryScope>,
    /// `level1_eviction_priority=` (`.L1::evict_last` and so on; sm_70 and
    /// PTX 7.4).
    pub level1_eviction_priority: Option<CacheEvictionPriority>,
    /// `cop=`: `Ca`, `Cg`, `Cs`, `Lu` or `Cv`.
    pub cop: Option<CacheOperator>,
    /// `ss=`: `None` follows the pointer, whose memory space cute-rs knows;
    /// `Some(SharedSpace::Cta)` requires this CTA's shared memory, and
    /// `Cluster` the shared memory of a CTA of the cluster (a
    /// [`map_dsmem_ptr`] pointer, or this CTA's own).
    pub ss: Option<SharedSpace>,
    /// `level_prefetch_size=` (`.L2::64B`, `.L2::128B` from sm_75,
    /// `.L2::256B` from sm_80).
    pub level_prefetch_size: Option<L2PrefetchSize>,
    /// An L2 cache policy for `.L2::cache_hint` (a 64-bit value from
    /// `createpolicy`; sm_80), which CuTe DSL's copy atoms take at copy time.
    /// A cute-rs extension of `load`.
    pub cache_policy: Option<Int64>,
}

/// The keyword arguments of CuTe DSL's `cute.arch.store(ptr, val,
/// level1_eviction_priority=, cop=, ss=, sem=, scope=)`. The default is a
/// plain (weak) store. A relaxed or release store takes an L1 eviction
/// priority and a cache policy but no cache operator, a weak one a cache
/// operator or an eviction priority, a volatile one no hints, and an MMIO
/// one is a scalar with system scope.
#[derive(Clone, Copy, Debug, Default)]
pub struct StoreOptions {
    /// `level1_eviction_priority=` (sm_70 and PTX 7.4).
    pub level1_eviction_priority: Option<CacheEvictionPriority>,
    /// `cop=`: `Wb`, `Cg`, `Cs` or `Wt`.
    pub cop: Option<CacheOperator>,
    /// `ss=`, as for [`LoadOptions::ss`].
    pub ss: Option<SharedSpace>,
    /// `sem=`: weak (the default), `Relaxed` or `Release`. cute-rs also takes
    /// `Sc` (`fence.sc` and a release store), `Volatile` and `Mmio`.
    pub sem: MemoryOrder,
    /// `scope=`: whom a relaxed, release or seq_cst store synchronises with.
    /// `None` is the system (`.sys`).
    pub scope: Option<MemoryScope>,
    /// An L2 cache policy for `.L2::cache_hint` (sm_80), as for
    /// [`LoadOptions::cache_policy`]. A cute-rs extension of `store`.
    pub cache_policy: Option<Int64>,
}

/// Check the `ss=` of a load or store against the pointer: `Cta` needs
/// this CTA's shared memory, `Cluster` shared memory of the cluster (a
/// [`map_dsmem_ptr`] pointer, or this CTA's own, whose `.shared::cta`
/// addresses are `.shared::cluster` ones too).
#[track_caller]
fn check_shared_space<T: Numeric>(ptr: &Ptr<T>, ss: Option<SharedSpace>, what: &str) {
    match ss {
        None => {}
        Some(SharedSpace::Cta) => assert!(
            ptr.memspace() == AddressSpace::Smem,
            "cute: a {what} with ss=SharedSpace::Cta addresses this CTA's shared memory, but the pointer is to {} \
             memory{}",
            ptr.memspace(),
            if ptr.memspace() == AddressSpace::Dsmem { " (another CTA's: use ss=SharedSpace::Cluster)" } else { "" }
        ),
        Some(SharedSpace::Cluster) => assert!(
            matches!(ptr.memspace(), AddressSpace::Dsmem | AddressSpace::Smem),
            "cute: a {what} with ss=SharedSpace::Cluster addresses the shared memory of a CTA of the cluster (a \
             map_dsmem_ptr pointer), but the pointer is to {} memory",
            ptr.memspace()
        ),
    }
}

/// The element offset of `ptr`, as a value, after checking that a vector of
/// `n` elements there is provably aligned.
#[track_caller]
fn vector_index<T: Numeric>(ptr: &Ptr<T>, n: usize, what: &str) -> ValueId {
    assert!(n > 0, "cute: {what} of 0 elements: a load or store moves at least one element");
    let (_, offset, _) = ptr.parts_dsmem();
    assert!(
        offset.is_multiple_of(n),
        "cute: {what} of {n} elements: cannot prove that the offset {offset} is aligned to {n} elements{}",
        builder::var_value_note(&offset)
    );
    crate::tensor::access_index::<T>(offset, n, what)
}

#[track_caller]
fn load_impl<T: Numeric>(ptr: &Ptr<T>, n: usize, opts: LoadOptions) -> Vec<ValueId> {
    check_shared_space(ptr, opts.ss, "load");
    let index = vector_index(ptr, n, "load_vector");
    let attrs = MemAttrs {
        order: opts.sem,
        scope: opts.scope.unwrap_or(MemoryScope::Sys),
        l1_eviction: opts.level1_eviction_priority,
        cop: opts.cop,
        l2_prefetch: opts.level_prefetch_size,
        non_coherent: false,
        cache_policy: opts.cache_policy.map(|p| p.checked()),
        peer: ptr.parts_dsmem().2,
    };
    load_vector_with::<T>(ptr.parts_dsmem().0, index, n, attrs)
}

#[track_caller]
fn store_impl<T: Numeric>(ptr: &Ptr<T>, values: &[ValueId], opts: StoreOptions) {
    check_shared_space(ptr, opts.ss, "store");
    let index = vector_index(ptr, values.len(), "store_vector");
    let attrs = MemAttrs {
        order: opts.sem,
        scope: opts.scope.unwrap_or(MemoryScope::Sys),
        l1_eviction: opts.level1_eviction_priority,
        cop: opts.cop,
        l2_prefetch: None,
        non_coherent: false,
        cache_policy: opts.cache_policy.map(|p| p.checked()),
        peer: ptr.parts_dsmem().2,
    };
    store_vector_with(ptr.parts_dsmem().0, index, values, attrs);
}

/// Load the element at `ptr` with CuTe DSL's memory semantics and cache
/// hints (`cute.arch.load`): an ordered load for flag-based synchronisation
/// between threads (`ld.acquire.gpu`, `ld.relaxed.cta`, ...), or a hinted
/// one (`ld.global.L1::evict_last`, `ld.global.cs`, ...). A spin on a flag
/// must use an ordered load: a plain one may be read once and kept.
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("handoff").block_dim([64, 1, 1]).build(|| {
///     let out = param_tensor::<u32>("out", Layout::col_major(64));
///     let flag = alloc_smem::<u32>(1, 4);
///     let data = make_smem_tensor::<u32>(Layout::col_major(32));
///     let t = thread_idx().x;
///     if_(t.eq(0), || flag.store(0u32));
///     sync_threads();
///     let cta = Some(MemoryScope::Cta);
///     if_(warp_idx().eq(0), || {
///         data.store(t, t.to::<u32>());
///         store(&flag, 1u32, StoreOptions { sem: MemoryOrder::Release, scope: cta, ..Default::default() });
///     });
///     if_(warp_idx().eq(1), || {
///         let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: cta, ..Default::default() };
///         while_carry((), |()| load(&flag, acquire).eq(0u32), |()| ());
///         out.store(t, data.load(t - 32));
///     });
/// });
/// ```
#[track_caller]
pub fn load<T: Numeric>(ptr: &Ptr<T>, opts: LoadOptions) -> Value<T> {
    Value::from_id(load_impl::<T>(ptr, 1, opts)[0])
}

/// [`load`] of `N` contiguous elements as one vector access (`.v2`,
/// `.v4`; at most 16 bytes), whose offset must be provably aligned to `N`
/// elements, in a buffer declared at least that aligned (CuTe DSL's
/// `cute.arch.load` with a vector type). As in PTX's memory model, each
/// element is an access of its own: an ordered or volatile vector of 8- or
/// 16-bit elements keeps their type (`ld.acquire.gpu.global.v4.b8`), and
/// takes one instruction per four elements beyond four.
#[track_caller]
pub fn load_vector<T: Numeric, const N: usize>(ptr: &Ptr<T>, opts: LoadOptions) -> [Value<T>; N] {
    let ids = load_impl::<T>(ptr, N, opts);
    std::array::from_fn(|i| Value::from_id(ids[i]))
}

/// Store `value` at `ptr` with CuTe DSL's memory semantics and cache hints
/// (`cute.arch.store`): an ordered store for flag-based synchronisation
/// (`st.release.gpu`, `st.relaxed.cta`, ...; see [`load`]) or a hinted one
/// (`st.global.L1::no_allocate`, `st.global.wt`, ...).
#[track_caller]
pub fn store<T: Numeric>(ptr: &Ptr<T>, value: impl IntoValue<T>, opts: StoreOptions) {
    store_impl(ptr, &[value.into_value().checked()], opts);
}

/// [`store`] of `N` contiguous elements as one vector access (at most 16
/// bytes), whose offset must be provably aligned to `N` elements, in a
/// buffer declared at least that aligned. Each element is an access of its
/// own, as for [`load_vector`].
#[track_caller]
pub fn store_vector<T: Numeric, const N: usize>(ptr: &Ptr<T>, values: [Value<T>; N], opts: StoreOptions) {
    let ids: Vec<ValueId> = values.iter().map(|v| v.checked()).collect();
    store_impl(ptr, &ids, opts);
}

/// The keyword arguments of CuTe DSL's `cute.arch.prefetch(addr,
/// cache_level=, evict_priority=, predicate=, uniform=)` for data addresses.
/// The default prefetches into L1. With `tensormap`, the address is a
/// TMA descriptor's (`prefetch.tensormap`, sm_90); `in_param_space=` is not needed, as the address is
/// generic.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefetchOptions {
    /// `cache_level=`: `L1` (`None`, the default when neither it nor
    /// `tensormap` is given) or `L2`.
    pub cache_level: Option<PrefetchCacheLevel>,
    /// `evict_priority=`: `EvictLast` or `EvictNormal`, for an L2 prefetch of
    /// global memory (`prefetch.global.L2::evict_last`; sm_80 and PTX 7.4).
    pub evict_priority: Option<CacheEvictionPriority>,
    /// `predicate=`: prefetch only where it holds (`@p prefetch`).
    pub predicate: Option<Boolean>,
    /// `uniform=True`: into the uniform cache (`prefetchu.L1`).
    pub uniform: bool,
    /// `tensormap=True`: the address is a TMA descriptor's
    /// (`prefetch.tensormap [a]`, sm_90), in global or parameter memory;
    /// the other fields must be unset.
    pub tensormap: bool,
}

/// Bring the cache line holding the element at `addr` into L1 or L2
/// (CuTe DSL's `cute.arch.prefetch`, `prefetch.global.L2` and so on): a
/// hint that reads nothing, so an address past the end of the buffer is
/// harmless, and that needs a global or local pointer (PTX does not cache
/// shared memory).
#[track_caller]
pub fn prefetch<T: Numeric>(addr: &Ptr<T>, opts: PrefetchOptions) {
    let loc = Location::caller();
    if opts.tensormap {
        assert!(
            opts.cache_level.is_none() && opts.evict_priority.is_none() && opts.predicate.is_none() && !opts.uniform,
            "prefetch: tensormap=True prefetches a TMA descriptor, which takes no cache level, eviction priority, \
             predicate or uniform flag"
        );
        assert!(
            matches!(addr.memspace(), AddressSpace::Gmem | AddressSpace::Generic),
            "prefetch: tensormap=True needs the descriptor's global or generic address, not a {} pointer",
            addr.memspace()
        );
        let desc = addr.toint_i64().checked();
        builder::with(|b| b.push(Op::PrefetchTensormap { desc }, loc));
        return;
    }
    let (buffer, offset) = addr.parts();
    let index = crate::tensor::index_operand(offset);
    let pred = opts.predicate.map(|p| p.checked());
    let (level, evict, uniform) = (opts.cache_level.unwrap_or_default(), opts.evict_priority, opts.uniform);
    builder::with(|b| {
        b.push(Op::Prefetch { buffer, index, elem: T::DTYPE, level, evict, uniform, pred }, loc);
    });
}

/// Prefetch the TMA descriptor at the generic address `desc`
/// (`prefetch.tensormap [desc]`, sm_90), such as
/// [`TmaDescriptor::get_ptr`](crate::TmaDescriptor::get_ptr): the address
/// form of [`prefetch`] with `tensormap`.
#[track_caller]
pub fn prefetch_tensormap(desc: impl IntoValue<i64>) {
    let loc = Location::caller();
    let desc = desc.into_value().checked();
    builder::with(|b| b.push(Op::PrefetchTensormap { desc }, loc));
}

/// Store `values` (one, two or four 32-bit values: Int32, Uint32 or
/// Float32, bit for bit) asynchronously into the shared memory of CTA
/// `peer_cta_rank` of the cluster, at the offset `smem_ptr` has in this
/// CTA, and count their bytes (4 per value) down the transaction count of
/// the mbarrier at `mbar_ptr`'s offset in that CTA (CuTe DSL's
/// `cute.arch.store_async_dsmem(smem_ptr, value, mbar_ptr, peer_cta_rank)`,
/// `mapa` on both pointers, then
/// `st.async.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.b32`;
/// sm_90 and PTX 8.1). The peer's phase must expect those bytes
/// (`mbarrier_expect_tx`); the data is visible to the peer's threads that
/// wait on that phase. `smem_ptr` must be aligned to the store's 4, 8 or 16
/// bytes.
#[track_caller]
pub fn store_async_dsmem<T: Numeric, const N: usize>(
    smem_ptr: &Ptr<T>,
    values: [Value<T>; N],
    mbar_ptr: &Ptr<u64>,
    peer_cta_rank: impl IntoValue<i32>,
) {
    let loc = Location::caller();
    assert!(
        matches!(T::DTYPE, DType::I32 | DType::U32 | DType::F32),
        "store_async_dsmem: st.async stores 32-bit values (Int32, Uint32 or Float32), not {}",
        T::DTYPE.cute_name()
    );
    assert!(matches!(N, 1 | 2 | 4), "store_async_dsmem: st.async stores one, two or four values, not {N}");
    for (p, role) in [(smem_ptr.memspace(), "smem_ptr"), (mbar_ptr.memspace(), "mbar_ptr")] {
        assert!(
            p == AddressSpace::Smem,
            "store_async_dsmem: {role} is a {p} pointer; both pointers are this CTA's shared memory, which the store \
             maps to the peer's"
        );
    }
    let (buffer, offset) = smem_ptr.parts();
    if let Some(o) = (offset.clone() * T::DTYPE.bytes()).as_static() {
        assert!(
            o % (4 * N) == 0,
            "store_async_dsmem: the store of {} bytes is at byte {o} of its buffer, which is not {}-byte aligned",
            4 * N,
            4 * N
        );
    }
    let index = crate::tensor::index_operand(offset);
    let values: Vec<ValueId> = values.iter().map(|v| v.bitcast::<i32>().checked()).collect();
    let (mbar, mbar_offset) = mbar_ptr.parts();
    let mbar_index = crate::tensor::index_operand(mbar_offset);
    let peer = peer_cta_rank.into_value().checked();
    builder::with(|b| b.push(Op::StAsync { buffer, index, elem: T::DTYPE, values, mbar, mbar_index, peer }, loc));
}

/// A pointer into the shared memory of CTA `cta_rank_in_cluster` of the
/// cluster, at the offset `smem_ptr` has in this CTA's (CuTe DSL's
/// `cute.arch.map_dsmem_ptr(smem_ptr, cta_rank_in_cluster)`, PTX's `mapa`;
/// sm_90). It keeps the element type, alignment and
/// swizzle (every CTA has the same shared-memory layout), and its
/// [`memspace`](Ptr::memspace) is [`AddressSpace::Dsmem`].
///
/// [`make_tensor`](crate::make_tensor) on it gives a tensor in the peer's
/// shared memory, which `load`, `store`, [`load`]/[`store`] (with `ss`
/// `None` or `SharedSpace::Cluster`), the atomics and reductions,
/// [`CopyOp::Universal`](crate::CopyOp::Universal),
/// [`CopyOp::S2R`](crate::CopyOp::S2R) and [`CopyOp::R2S`](crate::CopyOp::R2S)
/// copies, and [`CopyOp::DsmemStore`](crate::CopyOp::DsmemStore) reach
/// (`ld`, `st`, `atom` and `red` on `.shared::cluster`); other instructions
/// refuse it. As CUDA requires, the peer must be known to have started (a
/// cluster barrier after the kernel starts, `cluster_arrive` and
/// `cluster_wait`, before the first access) and must not exit while the
/// access may be outstanding (a cluster barrier before exiting); the
/// simulator checks both, and races between the CTAs.
///
/// Panics unless `smem_ptr` points to the shared memory of the CTA itself.
#[track_caller]
pub fn map_dsmem_ptr<T: Numeric>(smem_ptr: &Ptr<T>, cta_rank_in_cluster: impl IntoValue<i32>) -> Ptr<T> {
    assert!(
        smem_ptr.memspace() == AddressSpace::Smem,
        "map_dsmem_ptr: smem_ptr is a {} pointer; it maps a pointer to the CTA's own shared memory (smem) to another \
         CTA's",
        smem_ptr.memspace()
    );
    let peer = cta_rank_in_cluster.into_value().checked();
    Ptr { peer: Some(peer), ..smem_ptr.clone() }
}

/// `printf` moved to [`crate::print`] (`cute::printf`); this path is kept
/// for code written against the earlier `cute::arch::printf`, whose
/// `&[x.into(), ...]` operands still convert.
pub use crate::print::printf;

/// One PTX integer instruction on `args`, with a result of type `R`.
#[track_caller]
fn int_inst<R: Numeric>(func: IntFn, args: Vec<ValueId>) -> Value<R> {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| b.int_inst(func, args, R::DTYPE, loc)))
}

/// Extract a bit field (CuTe DSL's `cute.arch.bfe`, `bfe.{s,u}{32,64}`):
/// the `length` bits of `value` from bit `start` up, in the low bits of the
/// result, sign-extended for Int32 and Int64 and zero-extended for Uint32
/// and Uint64. As in PTX, only the low 8 bits of `start` and `length` count
/// (0 to 255), and the field stops at the most significant bit: a field
/// reaching past it is cut short and sign-extended from that bit, a
/// `start` past it gives copies of that bit (Int32, Int64) or zero (Uint32,
/// Uint64), and a zero `length` gives zero. These are the PTX ISA's results,
/// which the GPU gives too: the emitted code masks or clamps the start and
/// length where an SM89 GPU or `ptxas` would depart from them (see the
/// README).
///
/// `value` is an Int32, Uint32, Int64 or Uint64; `start` and `length` are
/// Int32 values or constants (CuTe DSL also takes Uint32 values, which need
/// `.bitcast()` here).
#[track_caller]
pub fn bfe<T: Integer>(value: impl IntoValue<T>, start: impl IntoValue<i32>, length: impl IntoValue<i32>) -> Value<T> {
    let args = vec![value.into_value().checked(), start.into_value().checked(), length.into_value().checked()];
    int_inst(IntFn::Bfe, args)
}

/// Insert a bit field (CuTe DSL's `cute.arch.bfi`, `bfi.b{32,64}`): `value`
/// with its `length` bits from bit `start` up replaced by the low bits of
/// `replacement`. Only the low 8 bits of `start` and `length` count, the
/// field stops at the most significant bit, and a zero `length` or a
/// `start` past the msb leaves `value` as it is.
///
/// The parameters are in CuTe DSL's order, which is PTX's: the replacement
/// first (CuTe DSL's docstring example passes them the other way round).
/// `bfi(0xa_u32, x, 12, 4)` sets bits 15:12 of `x` to `0xa`. `replacement`
/// and `value` are Int32, Uint32, Int64 or Uint64, of one type; `start` and
/// `length` are Int32 values or constants. As with [`bfe`], the GPU gives
/// the PTX ISA's results for every start and length.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("insert", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let y = param_tensor::<u32>("y", Layout::col_major(32));
///     let t = thread_idx().x;
///     x.store(t, bfi(y.load(t), x.load(t), 4, 8));
/// });
/// ```
///
/// As in CuTe DSL, the replacement and the value must have one type, so
/// this does not compile:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("mixed", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let y = param_tensor::<i32>("y", Layout::col_major(32));
///     let t = thread_idx().x;
///     x.store(t, bfi(y.load(t), x.load(t), 4, 8));
/// });
/// ```
#[track_caller]
pub fn bfi<T: Integer>(
    replacement: impl IntoValue<T>,
    value: impl IntoValue<T>,
    start: impl IntoValue<i32>,
    length: impl IntoValue<i32>,
) -> Value<T> {
    let args = vec![
        replacement.into_value().checked(),
        value.into_value().checked(),
        start.into_value().checked(),
        length.into_value().checked(),
    ];
    int_inst(IntFn::Bfi, args)
}

/// Any bitwise function of three values as one instruction (CuTe DSL's
/// `cute.arch.lop3`, `lop3.b32`). `lut` is the function's truth table: its
/// bit `(a << 2) | (b << 1) | c` is the result for input bits `a`, `b` and
/// `c`, which makes it the function applied to `0xF0`, `0xCC` and `0xAA`
/// ([`lop3_lut`] computes it from a closure). The operands are Int32 or
/// Uint32, of one type.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("majority", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     let (a, b, c) = (x.load(t), x.load((t + 1) % 32), x.load((t + 2) % 32));
///     x.store(t, lop3(a, b, c, lop3_lut(|a, b, c| (a & b) | (b & c) | (a & c))));
/// });
/// ```
///
/// CuTe DSL takes the table as an `int` constant, masked to 8 bits. Here it
/// is a `u8`, so it must be known when the kernel is built; a runtime value
/// does not compile:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("runtime_lut", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     let (a, b, c) = (x.load(t), x.load((t + 1) % 32), x.load((t + 2) % 32));
///     let lut = x.load((t + 3) % 32);
///     x.store(t, lop3(a, b, c, lut));
/// });
/// ```
#[track_caller]
pub fn lop3<T: Integer>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>, lut: u8) -> Value<T> {
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    int_inst(IntFn::Lop3 { lut }, args)
}

/// The [`lop3`] lookup table of a bitwise function of three operands: the
/// function applied to the bytes `0xF0`, `0xCC` and `0xAA`, as the PTX ISA
/// defines it.
///
/// ```
/// use cute::lop3_lut;
/// assert_eq!(lop3_lut(|a, b, c| a & b & !c), 0x40);
/// assert_eq!(lop3_lut(|a, b, c| a ^ b ^ c), 0x96);
/// assert_eq!(lop3_lut(|a, b, c| (a & b) | (b & c) | (a & c)), 0xe8);
/// assert_eq!(lop3_lut(|a, b, c| (a & b) | (!a & c)), 0xca);
/// assert_eq!(lop3_lut(|_, _, c| c), 0xaa);
/// ```
pub fn lop3_lut(f: impl Fn(u8, u8, u8) -> u8) -> u8 {
    f(0xF0, 0xCC, 0xAA)
}

/// The direction of a funnel shift ([`shf`]) and how it treats shift
/// amounts above 31 (CuTe DSL's `kind`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ShfKind {
    /// `"l"`: left, the amount taken mod 32 (`shf.l.wrap`).
    #[default]
    Left,
    /// `"r"`: right, the amount taken mod 32 (`shf.r.wrap`).
    Right,
    /// `"clamp_left"`: left, amounts above 32 clamped to 32
    /// (`shf.l.clamp`).
    ClampLeft,
    /// `"clamp_right"`: right, amounts above 32 clamped to 32
    /// (`shf.r.clamp`).
    ClampRight,
}

/// A funnel shift (CuTe DSL's `cute.arch.shf`, `shf.{l,r}.{wrap,clamp}.b32`):
/// shift the 64-bit value whose low word is `a` and whose high word is `b`
/// by `shift` bits, and return its high word (left shifts) or its low word
/// (right shifts). The amount is unsigned: the wrapping kinds take it mod
/// 32, the clamping kinds clamp it to 32, where a left shift returns `a`
/// and a right one `b`. With `a == b` it rotates: `shf(x, x, n,
/// ShfKind::Left)` is `x.rotate_left(n)`.
///
/// `a` is the low word, as in PTX, whose operand order CuTe DSL follows
/// (its docstring calls `a` the high part). `a` and `b` are Int32 or Uint32,
/// of one type; `shift` is an Int32 value or constant (CuTe DSL also takes a
/// Uint32 value, which needs `.bitcast()` here).
#[track_caller]
pub fn shf<T: Integer>(
    a: impl IntoValue<T>,
    b: impl IntoValue<T>,
    shift: impl IntoValue<i32>,
    kind: ShfKind,
) -> Value<T> {
    let (left, clamp) = match kind {
        ShfKind::Left => (true, false),
        ShfKind::Right => (false, false),
        ShfKind::ClampLeft => (true, true),
        ShfKind::ClampRight => (false, true),
    };
    let args = vec![a.into_value().checked(), b.into_value().checked(), shift.into_value().checked()];
    int_inst(IntFn::Shf { left, clamp }, args)
}

/// Pick four bytes of two 32-bit values (CuTe DSL's `cute.arch.prmt`,
/// `prmt.b32`). The bytes of `src` are numbered 0 to 3 and those of
/// `src_reg_shifted` 4 to 7; result byte `i` is chosen by bits `4i..4i + 4`
/// of `prmt_indices`, whose low 3 bits name a byte and whose top bit, when
/// set, fills the result byte with that byte's sign bit instead of copying
/// it. `prmt(x, y, 0x3210)` is `x`, `prmt(x, y, 0x0123)` reverses its
/// bytes, and `prmt(x, y, 0x5140)` interleaves the low two bytes of `x` and
/// `y`. The operands are Int32 or Uint32, of one type (CuTe DSL converts
/// each to Int32).
#[track_caller]
pub fn prmt<T: Integer>(
    src: impl IntoValue<T>,
    src_reg_shifted: impl IntoValue<T>,
    prmt_indices: impl IntoValue<T>,
) -> Value<T> {
    prmt_with(src, src_reg_shifted, prmt_indices, PrmtMode::Generic)
}

/// [`prmt`] in one of PTX's other modes (`prmt.b32.f4e` and so on), which
/// pick bytes by the low 2 bits of `selector` from a fixed table (see
/// [`PrmtMode`]); with [`PrmtMode::Generic`] it is [`prmt`]. A cute-rs
/// extension: CuTe DSL has only the generic form.
#[track_caller]
pub fn prmt_with<T: Integer>(
    src: impl IntoValue<T>,
    src_reg_shifted: impl IntoValue<T>,
    selector: impl IntoValue<T>,
    mode: PrmtMode,
) -> Value<T> {
    let args =
        vec![src.into_value().checked(), src_reg_shifted.into_value().checked(), selector.into_value().checked()];
    int_inst(IntFn::Prmt { mode }, args)
}

/// The high half of the whole product of `a` and `b` (CuTe DSL's
/// `cute.arch.mul_hi`, `mul.hi.{s,u}{32,64}`): bits 63:32 of the product of
/// two Int32 or Uint32 values, bits 127:64 of the product of two Int64 or
/// Uint64 values. The same as [`Value::mul_hi`].
#[track_caller]
pub fn mul_hi<T: Integer>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    a.into_value().mul_hi(b)
}

/// Integer types with an integer type twice as wide and of the same
/// signedness, which [`mul_wide`] returns: Int16 to Int32, Uint16 to
/// Uint32, Int32 to Int64 and Uint32 to Uint64.
pub trait Widen: Integer {
    /// The type twice as wide.
    type Wide: Integer;
}

impl Widen for i16 {
    type Wide = i32;
}

impl Widen for u16 {
    type Wide = u32;
}

impl Widen for i32 {
    type Wide = i64;
}

impl Widen for u32 {
    type Wide = u64;
}

/// The whole product of `a` and `b`, twice as wide as they are (CuTe DSL's
/// `cute.arch.mul_wide`, `mul.wide.{s,u}{16,32}`): Int16 or Uint16 operands
/// give an Int32 or Uint32, Int32 or Uint32 operands an Int64 or Uint64.
#[track_caller]
pub fn mul_wide<T: Widen>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T::Wide> {
    int_inst(IntFn::MulWide, vec![a.into_value().checked(), b.into_value().checked()])
}

/// The product of the low 24 bits of `a` and `b` (CuTe DSL's
/// `cute.arch.mul24`, `mul24.{lo,hi}.{s,u}32`). Bits 31:24 of the operands
/// are ignored, and Int32 operands are sign-extended from bit 23. The
/// product has 48 bits: `hi = false` returns its bits 31:0 and `hi = true`
/// its bits 47:16 (not the high word of a 64-bit product). The operands are
/// Int32 or Uint32, of one type.
#[track_caller]
pub fn mul24<T: Integer>(a: impl IntoValue<T>, b: impl IntoValue<T>, hi: bool) -> Value<T> {
    int_inst(IntFn::Mul24 { hi }, vec![a.into_value().checked(), b.into_value().checked()])
}

/// [`mul24`] plus `c` (CuTe DSL's `cute.arch.mad24`,
/// `mad24.{lo,hi}.{s,u}32`): all 32 bits of `c` are added, wrapping, to
/// bits 31:0 of the 48-bit product, or to its bits 47:16 with `hi`. The
/// operands are Int32 or Uint32, of one type.
#[track_caller]
pub fn mad24<T: Integer>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>, hi: bool) -> Value<T> {
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    int_inst(IntFn::Mad24 { hi }, args)
}

/// `a + b`, clamped to the Int32 range instead of wrapping (CuTe DSL's
/// `cute.arch.add_sat_int`, `add.sat.s32`).
#[track_caller]
pub fn add_sat_int(a: impl IntoValue<i32>, b: impl IntoValue<i32>) -> Int32 {
    int_inst(IntFn::AddSat, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a - b`, clamped to the Int32 range instead of wrapping (CuTe DSL's
/// `cute.arch.sub_sat_int`, `sub.sat.s32`).
#[track_caller]
pub fn sub_sat_int(a: impl IntoValue<i32>, b: impl IntoValue<i32>) -> Int32 {
    int_inst(IntFn::SubSat, vec![a.into_value().checked(), b.into_value().checked()])
}

/// The 32- and 64-bit integer types, Int32, Uint32, Int64 and Uint64, which
/// CuTe DSL's [`clz`], [`brev`] and [`bfind`] take.
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("highest_bit", || {
///     let x = param_tensor::<u32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     x.store(t, bfind(x.load(t)));
/// });
/// ```
///
/// Other types do not compile, as CuTe DSL rejects them:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("highest_bit", || {
///     let x = param_tensor::<u16>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     x.store(t, bfind(x.load(t)).to());
/// });
/// ```
pub trait Word: Integer + Bitwise {}

impl Word for i32 {}

impl Word for u32 {}

impl Word for i64 {}

impl Word for u64 {}

/// The number of leading zero bits of `value` (CuTe DSL's `cute.arch.clz`,
/// `clz.b{32,64}`): 32 or 64 for zero. A Uint32, as in CuTe DSL; the method
/// [`Value::clz`] returns the operand's type.
#[track_caller]
pub fn clz<T: Word>(value: impl IntoValue<T>) -> Uint32 {
    value.into_value().clz().to()
}

/// `value` with its bits in reverse order (CuTe DSL's `cute.arch.brev`,
/// `brev.b{32,64}`): the same as [`Value::brev`].
#[track_caller]
pub fn brev<T: Word>(value: impl IntoValue<T>) -> Value<T> {
    value.into_value().brev()
}

/// The number of set bits of `value`, in its own type (CuTe DSL's
/// `cute.arch.popc`, `popc.b{32,64}`): the same as [`Value::popc`].
#[track_caller]
pub fn popc<T: Integer>(value: impl IntoValue<T>) -> Value<T> {
    value.into_value().popc()
}

/// The position of the most significant bit of `value` that differs from
/// its sign bit (CuTe DSL's `cute.arch.bfind`): the highest 1 of a Uint32,
/// Uint64 or non-negative Int32 or Int64, the highest 0 of a negative one,
/// and 0xFFFFFFFF if there is none (0, or -1 when signed). A Uint32 (CuTe
/// DSL returns the same bits as an Int32 for a Uint32 operand, and as a
/// Uint32 for the others). CuTe DSL emits `bfind.{s,u}{32,64}`; cute-rs
/// computes the same result from [`clz`].
#[track_caller]
pub fn bfind<T: Word>(value: impl IntoValue<T>) -> Uint32 {
    let x = value.into_value();
    // A negative value's highest 0 is the highest 1 of its complement.
    let x = if T::DTYPE.is_signed() { select(x.lt(T::from_bits(0)), !x, x) } else { x };
    // W - 1 - clz wraps to 0xFFFFFFFF for zero, whose clz is W.
    Value::constant(T::DTYPE.bits() as u32 - 1) - x.clz().to::<u32>()
}

/// The base-2 logarithm of a power of two `a` (CuTe DSL's
/// `cute.arch.log2_of_pow2_int`): the number of trailing zero bits of `a`,
/// so the position of its lowest set bit for any other nonzero value, and
/// -1 for zero. CuTe DSL emits `brev.b32` then `bfind.shiftamt.u32`; cute-rs
/// computes the same result from [`Value::ctz`] (and a select for zero).
#[track_caller]
pub fn log2_of_pow2_int(a: impl IntoValue<i32>) -> Int32 {
    let a = a.into_value();
    select(a.eq(0), -1, a.ctz())
}

/// Hand-written PTX with one result: `$0` is the output and `$1...` the
/// inputs, as in `inline_ptx::<u32>("popc.b32 $0, $1;", &[x.into()])`. The
/// kernel can be compiled but not simulated; see [`inline_ptx_with_sim`].
/// [`InlinePtx`] is CuTe DSL's form, with named operands, several outputs,
/// read-write and pointer operands and a predicate.
#[track_caller]
pub fn inline_ptx<R: Numeric>(template: &str, inputs: &[AnyValue]) -> Value<R> {
    inline_ptx_impl(template, inputs, None)
}

/// [`inline_ptx`] with a function the simulator uses in its place. It maps the
/// inputs' raw bits to the output's raw bits (as a one-element vector).
#[track_caller]
pub fn inline_ptx_with_sim<R: Numeric>(template: &str, inputs: &[AnyValue], sim: InlineSim) -> Value<R> {
    inline_ptx_impl(template, inputs, Some(sim))
}

#[track_caller]
fn inline_ptx_impl<R: Numeric>(template: &str, inputs: &[AnyValue], sim: Option<InlineSim>) -> Value<R> {
    let loc = Location::caller();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(R::DTYPE);
        b.push(positional_inline_ptx(b, template, inputs, vec![result], false, sim), loc);
        result
    }))
}

/// An [`Op::InlinePtx`] of the positional functions: `$0..` the outputs,
/// then the inputs, each in its type's register class.
fn positional_inline_ptx(
    b: &builder::BuilderState,
    template: &str,
    inputs: &[AnyValue],
    outputs: Vec<ValueId>,
    side_effects: bool,
    sim: Option<InlineSim>,
) -> Op {
    let constraints: Vec<String> = outputs
        .iter()
        .map(|&o| format!("={}", asm_register(b.dtype(o))))
        .chain(inputs.iter().map(|a| asm_register(b.dtype(a.0)).to_string()))
        .collect();
    Op::InlinePtx {
        template: template.to_string(),
        inputs: inputs.iter().map(|a| AsmInput::Value(a.0)).collect(),
        outputs,
        constraints: constraints.join(","),
        predicated: false,
        side_effects,
        sim: sim.map(AsmSim::from_inline),
    }
}

/// Hand-written PTX with no result, run for its side effects (`$0...` are the
/// inputs). The kernel can be compiled but not simulated; see
/// [`inline_ptx_effect_with_sim`].
#[track_caller]
pub fn inline_ptx_effect(template: &str, inputs: &[AnyValue]) {
    inline_ptx_effect_impl(template, inputs, None);
}

/// [`inline_ptx_effect`] with a function the simulator calls in its place
/// (it returns no values), for instructions the simulator can treat as
/// doing nothing. A template holding `barrier.cluster.arrive` or
/// `barrier.cluster.wait` (and no predicate) also synchronises the warps of
/// the cluster in the simulator, which runs the function for each warp
/// once it gets past the barrier.
#[track_caller]
pub fn inline_ptx_effect_with_sim(template: &str, inputs: &[AnyValue], sim: InlineSim) {
    inline_ptx_effect_impl(template, inputs, Some(sim));
}

#[track_caller]
fn inline_ptx_effect_impl(template: &str, inputs: &[AnyValue], sim: Option<InlineSim>) {
    let loc = Location::caller();
    builder::with(|b| {
        let op = positional_inline_ptx(b, template, inputs, vec![], true, sim);
        b.push(op, loc)
    });
}

/// The keyword options of CuTe DSL's `cute.arch.warp_redux_sync` (for
/// [`warp_redux_sync_with`]), which apply to Float32 only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReduxOptions {
    /// `abs=True`: reduce the absolute values (`.abs`).
    pub abs: bool,
    /// `nan=True`: any NaN input makes the result the canonical NaN
    /// (`.NaN`); otherwise NaN inputs are ignored, and only an all-NaN input
    /// gives the canonical NaN.
    pub nan: bool,
}

/// A reduction of `value` across the whole warp in one instruction (CuTe
/// DSL's `cute.arch.warp_redux_sync(value, kind)`, `redux.sync`): every lane
/// gets the result. [`warp_redux_sync_with`] takes a member mask and the
/// Float32 options.
///
/// On Int32 and Uint32 (sm_80 and newer) the kinds are [`ReduxKind::Add`]
/// (wrapping), `And`, `Or`, `Xor`, `Max`, `Min` (signed on Int32), `Umax`
/// and `Umin` (unsigned); `Max` and `Min` on a Uint32 are `Umax` and
/// `Umin`, as in CuTe DSL. On Float32 they are [`ReduxKind::Fmax`] and
/// `Fmin` (`Max` and `Min` mean them), which treat `+0` as larger than `-0`
/// and ignore NaN inputs; Float32 needs sm_100a, or from PTX ISA 8.8 the
/// sm_100f family (checked when compiling).
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("warp_sums").block_dim([64, 1, 1]).build(|| {
///     let x = param_tensor::<i32>("x", Layout::col_major(64));
///     let t = thread_idx().x;
///     x.store(t, warp_redux_sync(x.load(t), ReduxKind::Add));
/// });
/// ```
#[track_caller]
pub fn warp_redux_sync<T: Numeric>(value: impl IntoValue<T>, kind: ReduxKind) -> Value<T> {
    warp_redux_sync_with(value, kind, FULL_MASK, ReduxOptions::default())
}

/// [`warp_redux_sync`] over the lanes in `mask` (CuTe DSL's positional
/// `mask_and_clamp`, which for `redux.sync` is the member mask), with CuTe
/// DSL's `abs=` and `nan=` for Float32. Every lane of `mask` gets the
/// reduction over the lanes of `mask`; the executing lane must be in
/// `mask`, and every lane of `mask` must execute it (the PTX ISA leaves
/// both undefined otherwise; the simulator reports them).
#[track_caller]
pub fn warp_redux_sync_with<T: Numeric>(
    value: impl IntoValue<T>,
    kind: ReduxKind,
    mask: impl IntoValue<u32>,
    opts: ReduxOptions,
) -> Value<T> {
    let loc = Location::caller();
    let kind = match (kind, T::DTYPE) {
        (ReduxKind::Max, DType::U32) => ReduxKind::Umax,
        (ReduxKind::Min, DType::U32) => ReduxKind::Umin,
        (ReduxKind::Max, DType::F32) => ReduxKind::Fmax,
        (ReduxKind::Min, DType::F32) => ReduxKind::Fmin,
        (kind, _) => kind,
    };
    let (value, mask) = (value.into_value().checked(), mask.into_value().checked());
    let (abs, nan) = (opts.abs, opts.nan);
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::Redux { result, kind, value, mask, abs, nan }, loc);
        result
    }))
}

/// The keyword options of CuTe DSL's `cute.arch.fmax` and `fmin` (for
/// [`fmax_with`] and [`fmin_with`]). The default is IEEE 754's
/// `maximumNumber` and `minimumNumber`: a NaN operand is ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MinMaxOptions {
    /// `abs=True`: the xorsign-abs form (`max.xorsign.abs.f32`): the
    /// larger (or smaller) of the operands' absolute values, with the
    /// exclusive or of their sign bits as its sign. A NaN result ignores
    /// both. Needs sm_86 and PTX ISA 7.2.
    pub abs: bool,
    /// `nan=True`: a NaN operand gives the canonical NaN (IEEE 754's
    /// `maximum`, `max.NaN.f32`). Needs sm_80.
    pub nan: bool,
    /// `ftz=True`: flush subnormal operands and results to zero, keeping
    /// their signs (`.ftz`).
    pub ftz: bool,
}

#[track_caller]
fn min_max(func: FloatFn, a: impl IntoValue<f32>, b: impl IntoValue<f32>, opts: MinMaxOptions) -> Float32 {
    let (a, b) = (a.into_value(), b.into_value());
    if opts == MinMaxOptions::default() {
        return if func == FloatFn::Max { a.max(b) } else { a.min(b) };
    }
    let loc = Location::caller();
    let args = vec![a.checked(), b.checked()];
    let MinMaxOptions { abs, nan, ftz } = opts;
    Value::from_id(builder::with(|bs| bs.float_inst(func, args, FloatMode::Exact, ftz, nan, abs, loc)))
}

/// The larger of two Float32 values (CuTe DSL's `cute.arch.fmax(a, b)`,
/// `max.f32`): a NaN operand is ignored (two give the canonical NaN), and
/// `+0` is larger than `-0`. The same as [`Value::max`]; [`fmax_with`]
/// takes CuTe DSL's `abs=`, `nan=` and `ftz=`.
#[track_caller]
pub fn fmax(a: impl IntoValue<f32>, b: impl IntoValue<f32>) -> Float32 {
    min_max(FloatFn::Max, a, b, MinMaxOptions::default())
}

/// The smaller of two Float32 values (CuTe DSL's `cute.arch.fmin(a, b)`,
/// `min.f32`): a NaN operand is ignored (two give the canonical NaN), and
/// `-0` is smaller than `+0`. The same as [`Value::min`].
#[track_caller]
pub fn fmin(a: impl IntoValue<f32>, b: impl IntoValue<f32>) -> Float32 {
    min_max(FloatFn::Min, a, b, MinMaxOptions::default())
}

/// [`fmax`] with CuTe DSL's options (`max{.ftz}{.NaN}{.xorsign.abs}.f32`):
/// `fmax_with(a, b, MinMaxOptions { abs: true, ..Default::default() })` is
/// `cute.arch.fmax(a, b, abs=True)`.
///
/// ```
/// use cute::prelude::*;
/// Kernel::builder("magnitudes").block_dim([32, 1, 1]).build(|| {
///     let x = param_tensor::<f32>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     let opts = MinMaxOptions { abs: true, nan: true, ..MinMaxOptions::default() };
///     x.store(t, fmax_with(x.load(t), x.load((t + 1) % 32), opts));
/// });
/// ```
#[track_caller]
pub fn fmax_with(a: impl IntoValue<f32>, b: impl IntoValue<f32>, opts: MinMaxOptions) -> Float32 {
    min_max(FloatFn::Max, a, b, opts)
}

/// [`fmin`] with CuTe DSL's options (`min{.ftz}{.NaN}{.xorsign.abs}.f32`).
#[track_caller]
pub fn fmin_with(a: impl IntoValue<f32>, b: impl IntoValue<f32>, opts: MinMaxOptions) -> Float32 {
    min_max(FloatFn::Min, a, b, opts)
}

// Extended-precision arithmetic: CuTe DSL's `add_cc`, `addc`, `sub_cc`,
// `subc`, `mad_cc` and `madc`, and the carry-in-and-out forms PTX also has.
// They pass the carry through PTX's carry flag (`CC.CF`), state that SSA
// values do not show, so each is an instruction the builder never merges or
// deletes and the backend emits with side effects, which LLVM keeps in
// order. A chain must stay in one block of straight-line code: the
// validator rejects a carry-in with no carry-out before it in its block, or
// with a runtime `if`, a loop, a `printf` or another operation that ends
// that code in the PTX between them (see `ir::carry_flag_breaker`), through
// which PTX does not keep the flag. The simulator keeps a carry flag per
// thread.

/// One extended-precision instruction.
#[track_caller]
fn carry_inst<T: Word>(op: CarryOp, carry_in: bool, carry_out: bool, args: Vec<ValueId>) -> Value<T> {
    let loc = Location::caller();
    let func = CarryFn { op, carry_in, carry_out };
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::CarryInst { result, func, args }, loc);
        result
    }))
}

/// `a + b`, setting the carry flag to the carry out of the sum (CuTe DSL's
/// `cute.arch.add_cc`, `add.cc.u32` or `add.cc.u64`): the low word of a
/// multi-word sum. The operands are Int32, Uint32, Int64 or Uint64, of one
/// type; the sum wraps, and the carry is that of the unsigned sum whatever
/// the signedness.
///
/// A chain of words continues with [`addc_cc`] and ends with [`addc`] (or
/// [`addc_cc`] followed by `addc(0, 0)`, which reads the carry out of the
/// top word):
///
/// ```
/// use cute::prelude::*;
/// // A 128-bit sum of four 32-bit words per thread, low word first.
/// Kernel::build("add128", || {
///     let a = param_tensor::<u32>("a", Layout::row_major((32, 4)));
///     let b = param_tensor::<u32>("b", Layout::row_major((32, 4)));
///     let s = param_tensor::<u32>("s", Layout::row_major((32, 4)));
///     let t = thread_idx().x;
///     let (a, b) = ((0..4).map(|i| a.load((t, i))).collect::<Vec<_>>(), (0..4).map(|i| b.load((t, i))).collect::<Vec<_>>());
///     let words = [add_cc(a[0], b[0]), addc_cc(a[1], b[1]), addc_cc(a[2], b[2]), addc(a[3], b[3])];
///     for (i, w) in words.into_iter().enumerate() {
///         s.store((t, i), w);
///     }
/// });
/// ```
///
/// The operands must have one type, as in CuTe DSL, and other types do not
/// compile:
///
/// ```compile_fail
/// use cute::prelude::*;
/// Kernel::build("narrow", || {
///     let x = param_tensor::<u16>("x", Layout::col_major(32));
///     let t = thread_idx().x;
///     x.store(t, add_cc(x.load(t), x.load(t)));
/// });
/// ```
#[track_caller]
pub fn add_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Add, false, true, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a + b` plus the carry flag (CuTe DSL's `cute.arch.addc`, `addc.u32` or
/// `addc.u64`). Like CuTe DSL's, it reads the flag but leaves it as it is, so
/// after [`add_cc`] it gives the high word of a two-word sum (two `addc`s
/// after one `add_cc` both add its carry); longer chains continue with
/// [`addc_cc`]. It must follow an addition that sets the flag ([`add_cc`],
/// [`addc_cc`], [`mad_cc`] or [`madc_cc`]; not a subtraction, see [`subc`])
/// in the same block of straight-line code, with no runtime `if`, loop,
/// `printf` or 64-bit division by a runtime value between them (see
/// [`carry_flag_breaker`](crate::ir::carry_flag_breaker)): the validator
/// rejects the kernel otherwise.
#[track_caller]
pub fn addc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Add, true, false, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a + b` plus the carry flag, setting the flag to the carry out of that
/// sum (`addc.cc.u32` or `addc.cc.u64`): the middle words of a multi-word
/// sum. A cute-rs extension: CuTe DSL's [`addc`] does not write the flag, so
/// its chains stop at two words.
#[track_caller]
pub fn addc_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Add, true, true, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a - b`, setting the carry flag to the borrow out of the difference, 1
/// when `b` is greater than `a` as unsigned numbers (CuTe DSL's
/// `cute.arch.sub_cc`, `sub.cc.u32` or `sub.cc.u64`): the low word of a
/// multi-word difference. The operands are Int32, Uint32, Int64 or Uint64,
/// of one type.
#[track_caller]
pub fn sub_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Sub, false, true, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a - (b + flag)`, with the borrow in the carry flag (CuTe DSL's
/// `cute.arch.subc`, `subc.u32` or `subc.u64`). Like [`addc`], it reads the
/// flag without writing it; longer chains continue with [`subc_cc`].
///
/// It must read a borrow, set by [`sub_cc`] or [`subc_cc`]: the PTX ISA
/// describes one flag for carries and borrows, but on the GPU (an SM89 with
/// `ptxas` 12.8) a borrow is the inverted carry, so `subc` after an
/// [`add_cc`] subtracts one minus its carry, and an [`addc`] after a
/// subtraction adds one minus its borrow. The validator rejects a chain
/// that switches between adding (adds and multiply-adds) and subtracting.
#[track_caller]
pub fn subc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Sub, true, false, vec![a.into_value().checked(), b.into_value().checked()])
}

/// `a - (b + flag)`, setting the flag to the borrow out (`subc.cc.u32` or
/// `subc.cc.u64`): the middle words of a multi-word difference. A cute-rs
/// extension, as [`addc_cc`] is.
#[track_caller]
pub fn subc_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>) -> Value<T> {
    carry_inst(CarryOp::Sub, true, true, vec![a.into_value().checked(), b.into_value().checked()])
}

/// The low half of `a * b`, plus `c`, setting the carry flag to the carry
/// out of the addition (CuTe DSL's `cute.arch.mad_cc`, `mad.lo.cc.s32`,
/// `mad.lo.cc.u32`, `mad.lo.cc.s64` or `mad.lo.cc.u64`). With [`madc`],
/// [`madc_cc`], [`addc`] and [`mul_hi`] it builds multi-word products. The
/// operands are Int32, Uint32, Int64 or Uint64, of one type; the low half
/// of the product does not depend on the signedness.
///
/// ```
/// use cute::prelude::*;
/// // The 128-bit product of two 64-bit numbers held as 32-bit words.
/// Kernel::build("mul64x64", || {
///     let a = param_tensor::<u32>("a", Layout::row_major((32, 2)));
///     let b = param_tensor::<u32>("b", Layout::row_major((32, 2)));
///     let p = param_tensor::<u32>("p", Layout::row_major((32, 4)));
///     let t = thread_idx().x;
///     let (a0, a1, b0, b1) = (a.load((t, 0)), a.load((t, 1)), b.load((t, 0)), b.load((t, 1)));
///     // a0 b0 + (a1 b0 + a0 b1) 2^32 + a1 b1 2^64, a partial product at a time.
///     let (h00, h01, h10, h11) = (mul_hi(a0, b0), mul_hi(a0, b1), mul_hi(a1, b0), mul_hi(a1, b1));
///     let p0 = a0 * b0;
///     let p1 = mad_cc(a0, b1, h00);
///     let p2 = madc_cc(a1, b1, h01);
///     let p3 = addc(h11, 0u32);
///     let p1 = mad_cc(a1, b0, p1);
///     let p2 = addc_cc(p2, h10);
///     let p3 = addc(p3, 0u32);
///     for (i, w) in [p0, p1, p2, p3].into_iter().enumerate() {
///         p.store((t, i), w);
///     }
/// });
/// ```
#[track_caller]
pub fn mad_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>) -> Value<T> {
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    carry_inst(CarryOp::MadLo, false, true, args)
}

/// The low half of `a * b`, plus `c` and the carry flag (CuTe DSL's
/// `cute.arch.madc`, `madc.lo.{s,u}{32,64}`). Like [`addc`], it reads the
/// flag without writing it; [`madc_cc`] also writes it.
#[track_caller]
pub fn madc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>) -> Value<T> {
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    carry_inst(CarryOp::MadLo, true, false, args)
}

/// The low half of `a * b`, plus `c` and the carry flag, setting the flag to
/// the carry out of that addition (`madc.lo.cc.{s,u}{32,64}`). A cute-rs
/// extension, as [`addc_cc`] is.
#[track_caller]
pub fn madc_cc<T: Word>(a: impl IntoValue<T>, b: impl IntoValue<T>, c: impl IntoValue<T>) -> Value<T> {
    let args = vec![a.into_value().checked(), b.into_value().checked(), c.into_value().checked()];
    carry_inst(CarryOp::MadLo, true, true, args)
}

/// `a + b + carry_in` and its carry out, with both carries as values, 0 or
/// 1, rather than in the carry flag (a cute-rs form, which CuTe DSL lacks).
/// A carry held as a value can cross control flow (keep it in a
/// [`Var`](crate::Var)) and start a chain from a runtime carry. Any nonzero
/// `carry_in` counts as 1.
///
/// It is three instructions: [`add_cc`] of `carry_in` and all ones, which
/// carries exactly when `carry_in` is not 0, puts the carry in the flag;
/// [`addc_cc`] adds; and [`addc`] of zeros reads the carry out.
#[track_caller]
pub fn add_with_carry<T: Word>(
    a: impl IntoValue<T>,
    b: impl IntoValue<T>,
    carry_in: impl IntoValue<T>,
) -> (Value<T>, Value<T>) {
    let (a, b, carry_in) = (a.into_value(), b.into_value(), carry_in.into_value());
    let _ = add_cc(carry_in, T::from_bits(u64::MAX));
    let sum = addc_cc(a, b);
    (sum, addc(T::from_bits(0), T::from_bits(0)))
}

/// `a - b - borrow_in` and its borrow out, with both borrows as values, 0
/// or 1, as [`add_with_carry`] does for sums (a cute-rs form). Any nonzero
/// `borrow_in` counts as 1.
///
/// It is three instructions and an `and`, all subtractions, since the GPU
/// keeps a borrow in the carry flag as the inverted carry (see [`subc`]):
/// [`sub_cc`] of 0 and `borrow_in`, which borrows exactly when `borrow_in`
/// is not 0, puts the borrow in the flag; [`subc_cc`] subtracts; and
/// [`subc`] of zeros, `0 - borrow`, gives the borrow out in every bit.
#[track_caller]
pub fn sub_with_borrow<T: Word>(
    a: impl IntoValue<T>,
    b: impl IntoValue<T>,
    borrow_in: impl IntoValue<T>,
) -> (Value<T>, Value<T>) {
    let (a, b, borrow_in) = (a.into_value(), b.into_value(), borrow_in.into_value());
    let _ = sub_cc(T::from_bits(0), borrow_in);
    let difference = subc_cc(a, b);
    (difference, subc(T::from_bits(0), T::from_bits(0)) & T::from_bits(1))
}

/// The bytes of dynamic shared memory the launch requested (CuTe DSL's
/// `cute.arch.dynamic_smem_size`): the bytes of the dynamic region
/// ([`get_dyn_smem`](crate::get_dyn_smem)),
/// [`LaunchConfig::dynamic_smem_bytes`](crate::LaunchConfig::dynamic_smem_bytes),
/// the same as [`get_dyn_smem_size`]. cute-rs keeps the shared buffers sized
/// at build time in the dynamic shared-memory window in front of the region
/// when they need more than 48 KB or the kernel has a region, and passes
/// them to the driver as dynamic shared memory too, so this is
/// `%dynamic_smem_size` less their bytes
/// ([`SharedMemoryLayout::dynamic_start`](crate::ir::SharedMemoryLayout::dynamic_start)).
#[track_caller]
pub fn dynamic_smem_size() -> Int32 {
    special(SpecialReg::DynamicSmemSize)
}

/// The bytes of the dynamic shared-memory region
/// ([`get_dyn_smem`](crate::get_dyn_smem)) the launch gave this block
/// ([`LaunchConfig::dynamic_smem_bytes`](crate::LaunchConfig::dynamic_smem_bytes),
/// CuTe DSL's `cute.arch.get_dyn_smem_size`), to check allocations from it
/// against. Like CuTe DSL's, it reads the same as [`dynamic_smem_size`].
#[track_caller]
pub fn get_dyn_smem_size() -> Int32 {
    special(SpecialReg::DynamicSmemSize)
}

/// The block's static and dynamic shared memory as the GPU counts it (CuTe
/// DSL's `cute.arch.total_smem_size`, `%total_smem_size`), without the part
/// reserved for the system. The PTX ISA reports it "in multiples of the
/// shared memory allocation unit" (128 bytes from sm_80 on), which on an
/// SM89 GPU means bytes rounded up to a multiple of 128, not a count of
/// units: 129 bytes read as 256. The simulator returns the same.
#[track_caller]
pub fn total_smem_size() -> Int32 {
    special(SpecialReg::TotalSmemSize)
}

/// The block's shared memory including the part reserved for the system
/// (CuTe DSL's `cute.arch.aggr_smem_size`, `%aggr_smem_size`; sm_90 and
/// newer). The simulator returns [`total_smem_size`]'s bytes plus a 1 KB
/// reservation, from the PTX ISA and the CUDA programming guide; no SM90 GPU
/// has checked it yet.
#[track_caller]
pub fn aggr_smem_size() -> Int32 {
    special(SpecialReg::AggrSmemSize)
}

/// This block's rank in its cluster (CuTe DSL's
/// `cute.arch.block_idx_in_cluster`, `%cluster_ctarank`; sm_90 and newer):
/// [`block_in_cluster_idx`] linearized x fastest, `x + y * dx + z * dx * dy`
/// with `dx` and `dy` from [`block_in_cluster_dim`]. It is the rank peer-CTA
/// operations take ([`MbarrierOptions::peer_cta_rank_in_cluster`]). Not to
/// be confused with [`block_in_cluster_idx`], the 3-D index (the same name
/// in both libraries).
#[track_caller]
pub fn block_idx_in_cluster() -> Int32 {
    special(SpecialReg::ClusterCtaRank)
}

/// The number of blocks in this block's cluster (CuTe DSL's
/// `cute.arch.cluster_size`, `%cluster_nctarank`; sm_90 and newer): the
/// product of [`block_in_cluster_dim`].
#[track_caller]
pub fn cluster_size() -> Int32 {
    special(SpecialReg::ClusterNCtaRank)
}

#[track_caller]
fn cp_async_mbarrier_arrive_impl(mbar_ptr: &Ptr<u64>, noinc: bool) {
    let loc = Location::caller();
    let (buffer, offset) = mbar_ptr.parts();
    let index = crate::tensor::index_operand(offset);
    builder::with(|b| b.push(Op::CpAsyncMbarrierArrive { buffer, index, noinc }, loc));
}

/// Make an mbarrier track this thread's `cp.async` copies (CuTe DSL's
/// `cute.arch.cp_async_mbarrier_arrive_noinc`,
/// `cp.async.mbarrier.arrive.noinc.shared.b64`): once every copy the thread
/// issued before it has landed, the mbarrier receives one arrival. That
/// replaces [`cp_async_commit_group`] and [`cp_async_wait_group`] for a
/// consumer in another warp, which waits on the mbarrier instead
/// ([`mbarrier_test_wait`], [`mbarrier_wait`]); the phase's completion makes
/// the copied data visible to it.
///
/// `.noinc` does not raise the phase's pending count, so the mbarrier's
/// count ([`mbarrier_init`]) must include one arrival per call (in CuTe DSL's
/// usual pattern, one per producer thread and nothing else);
/// [`cp_async_mbarrier_arrive`] is the form that raises it. PTX has it from
/// sm_80 (CuTe DSL asks for sm_90). The simulator lands the copies when a
/// thread waits on the mbarrier and only such arrivals are missing, or at a
/// covering [`cp_async_wait_group`] if that comes first, and reports a phase
/// that would complete with tracked copies in flight.
///
/// ```
/// use cute::prelude::*;
///
/// // Warp 0 copies 128 words into shared memory; warp 1 waits on the
/// // mbarrier and reads them.
/// let kernel = Kernel::builder("handoff").block_dim([64, 1, 1]).build(|| {
///     let src = param_tensor::<u32>("src", Layout::col_major(128));
///     let dst = param_tensor::<u32>("dst", Layout::col_major(128));
///     let t = thread_idx().x;
///     let bar = alloc_smem::<u64>(1, 8);
///     let data = make_smem_tensor::<u32>(Layout::col_major(128));
///     if_(t.eq(0), || mbarrier_init(&bar, 32));
///     sync_threads();
///     let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
///     let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
///     if_else(
///         warp_idx().eq(0),
///         || {
///             let thr = tiled.get_slice(t);
///             copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&data));
///             cp_async_mbarrier_arrive_noinc(&bar);
///         },
///         || {
///             while_carry((), |()| !mbarrier_test_wait(&bar, 0), |()| ());
///             for k in 0..4 {
///                 dst.store((t - 32) * 4 + k, data.load((t - 32) * 4 + k));
///             }
///         },
///     );
/// });
/// let src: Vec<u32> = (0..128).collect();
/// let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 128])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[1].to_vec::<u32>(), src);
/// ```
#[track_caller]
pub fn cp_async_mbarrier_arrive_noinc(mbar_ptr: &Ptr<u64>) {
    cp_async_mbarrier_arrive_impl(mbar_ptr, true);
}

/// [`cp_async_mbarrier_arrive_noinc`] without `.noinc` (CuTe DSL's
/// `cutlass.experimental.primitives.cp_async_mbarrier_arrive(addr,
/// noinc=False)`, `cp.async.mbarrier.arrive.shared.b64`; `noinc=True` is
/// [`cp_async_mbarrier_arrive_noinc`]). The phase's pending count is raised
/// by one first, so the asynchronous arrival changes it by nothing on
/// balance: the mbarrier's count covers only the ordinary arrivals
/// ([`mbarrier_arrive`]), and the phase still cannot complete before the
/// tracked copies land.
#[track_caller]
pub fn cp_async_mbarrier_arrive(mbar_ptr: &Ptr<u64>) {
    cp_async_mbarrier_arrive_impl(mbar_ptr, false);
}

pub mod inline;

pub use inline::{AsmOutputs, AsmPush, InlineAsm, InlinePtx, MlirOp};

//! Hardware intrinsics (CuTe DSL's `cute.arch`): thread and block indices,
//! barriers, the `cp.async` group operations, warp-collective operations
//! (shuffles, votes), atomics, fences and `printf`.

use std::panic::Location;

use crate::builder;
use crate::ir::{
    AtomicKind, BarrierKind, DType, Dim, FenceScope, InlineSim, LaneMask, MbarrierWaitKind, MemScopeKind, Op,
    ShuffleKind, SpecialReg, VoteKind,
};
use crate::layout::IndexExpr;
use crate::tensor::Ptr;
use crate::value::{AnyValue, Boolean, Int32, Int64, Integer, IntoValue, Numeric, Uint32, Value};

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

/// Barrier among the lanes of this warp (`__syncwarp`).
#[track_caller]
pub fn sync_warp() {
    let loc = Location::caller();
    let mask = all_lanes().checked();
    builder::with(|b| b.push(Op::WarpBarrier { mask }, loc));
}

#[track_caller]
fn shuffle<T: Numeric>(
    kind: ShuffleKind,
    value: Value<T>,
    offset: impl IntoValue<i32>,
    mask: impl IntoValue<u32>,
) -> Value<T> {
    let loc = Location::caller();
    assert_eq!(T::DTYPE.bits(), 32, "cute: shuffles move 32-bit values, not {}", T::DTYPE);
    let (value, offset, mask) = (value.checked(), offset.into_value().checked(), mask.into_value().checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::Shuffle { result, kind, value, offset, mask }, loc);
        result
    }))
}

/// `value` as held by lane `lane` of this warp (`__shfl_sync`).
#[track_caller]
pub fn shuffle_sync<T: Numeric>(value: Value<T>, lane: impl IntoValue<i32>) -> Value<T> {
    shuffle(ShuffleKind::Idx, value, lane, u32::MAX)
}

/// `value` from `offset` lanes below (`__shfl_up_sync`); lanes below `offset`
/// keep their own.
#[track_caller]
pub fn shuffle_sync_up<T: Numeric>(value: Value<T>, offset: impl IntoValue<i32>) -> Value<T> {
    shuffle(ShuffleKind::Up, value, offset, u32::MAX)
}

/// `value` from `offset` lanes above (`__shfl_down_sync`); lanes without one
/// keep their own.
#[track_caller]
pub fn shuffle_sync_down<T: Numeric>(value: Value<T>, offset: impl IntoValue<i32>) -> Value<T> {
    shuffle(ShuffleKind::Down, value, offset, u32::MAX)
}

/// `value` from lane `lane ^ offset` (`__shfl_xor_sync`).
#[track_caller]
pub fn shuffle_sync_bfly<T: Numeric>(value: Value<T>, offset: impl IntoValue<i32>) -> Value<T> {
    shuffle(ShuffleKind::Bfly, value, offset, u32::MAX)
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

/// A mask of the lanes holding the same `value` as this one
/// (`__match_any_sync`).
#[track_caller]
pub fn match_sync<T: Numeric>(value: Value<T>) -> Uint32 {
    let loc = Location::caller();
    assert_eq!(T::DTYPE.bits(), 32, "cute: match_sync compares 32-bit values, not {}", T::DTYPE);
    let (value, mask) = (value.checked(), all_lanes().checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::U32);
        b.push(Op::MatchAny { result, value, mask }, loc);
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

#[track_caller]
fn atomic<T: Integer>(kind: AtomicKind, ptr: &Ptr<T>, value: Value<T>, compare: Option<Value<T>>) -> Value<T> {
    let loc = Location::caller();
    assert!(T::DTYPE.bits() >= 32, "cute: atomics are 32 or 64 bits wide, not {}", T::DTYPE);
    let (buffer, offset) = ptr.parts();
    let index = offset.into_value().checked();
    let (value, compare) = (value.checked(), compare.map(|c| c.checked()));
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::Atomic { result, kind, buffer, index, value, compare }, loc);
        result
    }))
}

macro_rules! atomics {
    ($($name:ident, $kind:ident, $doc:literal);* $(;)?) => {$(
        #[doc = $doc]
        #[track_caller]
        pub fn $name<T: Integer>(ptr: &Ptr<T>, value: impl IntoValue<T>) -> Value<T> {
            atomic(AtomicKind::$kind, ptr, value.into_value(), None)
        }
    )*};
}

atomics! {
    atomic_add, Add, "Atomically add `value`, returning the old contents.";
    atomic_and, And, "Atomically AND `value`, returning the old contents.";
    atomic_or, Or, "Atomically OR `value`, returning the old contents.";
    atomic_xor, Xor, "Atomically XOR `value`, returning the old contents.";
    atomic_min, Min, "Atomically take the minimum with `value`, returning the old contents.";
    atomic_max, Max, "Atomically take the maximum with `value`, returning the old contents.";
    atomic_exch, Exchange, "Atomically store `value`, returning the old contents.";
}

/// Atomically store `value` if the contents equal `compare`, returning the
/// old contents.
#[track_caller]
pub fn atomic_cas<T: Integer>(ptr: &Ptr<T>, compare: impl IntoValue<T>, value: impl IntoValue<T>) -> Value<T> {
    atomic(AtomicKind::CompareExchange, ptr, value.into_value(), Some(compare.into_value()))
}

/// Initialise an mbarrier (a 64-bit word of shared memory, SM80 and newer)
/// to wait for `count` arrivals per phase. One thread initialises it,
/// before a barrier that publishes it to the others.
#[track_caller]
pub fn mbarrier_init(mbarrier: &Ptr<u64>, count: impl IntoValue<i32>) {
    let loc = Location::caller();
    let (buffer, offset) = mbarrier.parts();
    let index = offset.into_value().checked();
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

#[track_caller]
fn mbarrier_arrive_impl(mbarrier: &Ptr<u64>, count: Option<Int32>, expect_tx: Option<Int32>, opts: MbarrierOptions) {
    let loc = Location::caller();
    let (buffer, offset) = mbarrier.parts();
    let index = offset.into_value().checked();
    let (count, expect_tx) = (count.map(|c| c.checked()), expect_tx.map(|t| t.checked()));
    let peer = opts.peer_cta_rank_in_cluster.map(|p| p.checked());
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
    let (buffer, offset) = mbarrier.parts();
    let index = offset.into_value().checked();
    let bytes = bytes.into_value().checked();
    let peer = opts.peer_cta_rank_in_cluster.map(|p| p.checked());
    let scope = opts.scope.unwrap_or(if peer.is_some() { MemScopeKind::Cluster } else { MemScopeKind::Cta });
    builder::with(|b| b.push(Op::MbarrierExpectTx { buffer, index, bytes, peer, scope }, loc));
}

/// An mbarrier wait recorded at `loc` (the user's call, also from inside the
/// closures of the waiting loops).
fn mbarrier_wait_op(mbarrier: &Ptr<u64>, parity: Int32, kind: MbarrierWaitKind, loc: crate::ir::Loc) -> Boolean {
    let (buffer, offset) = mbarrier.parts();
    let index = offset.into_value().checked();
    let parity = parity.checked();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::Bool);
        b.push(Op::MbarrierWait { result, buffer, index, parity, kind }, loc);
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
    let (s, s_off) = src.parts();
    let (d, d_off) = dst.parts();
    let (m, m_off) = mbarrier.parts();
    let (src_index, dst_index, mbarrier_index) =
        (s_off.into_value().checked(), d_off.into_value().checked(), m_off.into_value().checked());
    let bytes = bytes.into_value().checked();
    builder::with(|b| {
        b.push(
            Op::CpAsyncBulk {
                src: s,
                src_index,
                dst: d,
                dst_index,
                elem: T::DTYPE,
                bytes,
                mbarrier: m,
                mbarrier_index,
            },
            loc,
        )
    });
}

#[track_caller]
fn fence(scope: FenceScope) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Fence { scope }, loc));
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

/// Print from the GPU (`vprintf`). `%d %i %u %x %f %e %g %%` are understood;
/// the simulator collects the lines in [`SimReport::printed`](crate::SimReport).
#[track_caller]
pub fn printf(format: &str, args: &[AnyValue]) {
    let loc = Location::caller();
    let args = args.iter().map(|a| a.0).collect();
    builder::with(|b| b.push(Op::Printf { format: format.to_string(), args }, loc));
}

/// Hand-written PTX with one result: `$0` is the output and `$1...` the
/// inputs, as in `inline_ptx::<u32>("popc.b32 $0, $1;", &[x.into()])`. The
/// kernel can be compiled but not simulated; see [`inline_ptx_with_sim`].
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
    let inputs = inputs.iter().map(|a| a.0).collect();
    Value::from_id(builder::with(|b| {
        let result = b.new_value(R::DTYPE);
        b.push(
            Op::InlinePtx { template: template.to_string(), inputs, outputs: vec![result], side_effects: false, sim },
            loc,
        );
        result
    }))
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
/// doing nothing, such as a cluster barrier in its lockstep execution.
#[track_caller]
pub fn inline_ptx_effect_with_sim(template: &str, inputs: &[AnyValue], sim: InlineSim) {
    inline_ptx_effect_impl(template, inputs, Some(sim));
}

#[track_caller]
fn inline_ptx_effect_impl(template: &str, inputs: &[AnyValue], sim: Option<InlineSim>) {
    let loc = Location::caller();
    let inputs = inputs.iter().map(|a| a.0).collect();
    builder::with(|b| {
        b.push(Op::InlinePtx { template: template.to_string(), inputs, outputs: vec![], side_effects: true, sim }, loc)
    });
}

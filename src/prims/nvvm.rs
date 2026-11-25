//! The NVVM layer of CuTe DSL's `cutlass.experimental.primitives`: the operations that have no `cute::arch`
//! counterpart, and the map from every primitives name to the cute-rs
//! function that does its job.
//!
//! # The operations added here
//!
//! * [`barrier_cta_red`] (and the typed [`barrier_cta_red_popc`],
//!   [`barrier_cta_red_and`], [`barrier_cta_red_or`]): `bar.red`, a named
//!   barrier that also reduces a predicate over its threads.
//! * The mbarrier operations: [`mbarrier_arrive`] returning the phase's state
//!   token, [`mbarrier_arrive_drop`], [`mbarrier_arrive_drop_expect_tx`],
//!   [`mbarrier_arrive_drop_nocomplete`], [`mbarrier_arrive_nocomplete`],
//!   [`mbarrier_complete_tx`], [`mbarrier_inval`], and the waits on a state
//!   token or with a suspend-time hint ([`mbarrier_test_wait`],
//!   [`mbarrier_try_wait`], [`mbarrier_try_wait_parity`],
//!   [`mbarrier_try_wait_timelimit`], [`mbarrier_wait`]).
//! * [`nanosleep`], [`pmevent`], [`pmevent_mask`], [`breakpoint`],
//!   [`ptx_add_comment`] and [`exit`].
//! * [`memory_barrier`] (`membar`, as `fence.sc`) and [`fence_sc_cluster`].
//! * [`dot_accumulate_4way`] and [`dot_accumulate_2way`] (`dp4a`, `dp2a`).
//! * [`st_bulk`] (sm_100) and [`tcgen05_shift`] (sm_100a).
//! * [`elect_sync`] with a member mask, [`cvta_to`], [`mov_b32`].
//! * [`nvvm_op`], the raw escape: an `nvvm.*` (or `llvm.call_intrinsic`)
//!   operation written as MLIR text, simulated only with a closure.
//!
//! In the simulator: `bar.red` is a named barrier (counted in whole warps)
//! that returns the reduction to every participant; an arrive-drop also
//! lowers the arrival count of the phase and of every later one; a
//! `.noComplete` arrival that would complete the phase is an error;
//! `complete_tx` lowers the transaction count; after `mbarrier.inval` any
//! use of the mbarrier but `mbarrier_init` is an error; a time-limited
//! `try_wait` is a `try_wait` (the limit is a hint); `nanosleep` yields the
//! warp; `pmevent`, `brkpt` and comments are counted in
//! [`SimReport::markers`](crate::SimReport::markers); `exit` ends a whole
//! warp (an exit by part of a warp is not simulated), and a wait that then
//! never completes is reported as a hang naming the exit. A state token is
//! opaque on the GPU; the simulator's is the number of the phase the arrival
//! was in. `tcgen05.shift` is not simulated.
//!
//! # From primitives names to cute-rs
//!
//! Names marked *here* are in this module and re-exported as `cute::prims::*`,
//! except `mbarrier_arrive`, `mbarrier_test_wait`, `mbarrier_try_wait` and
//! `mbarrier_wait`, whose names the crate root has for its parity-based
//! forms: those are `cute::prims::nvvm::*` only. The others are
//! crate-root functions (`cute::*`, from `cute::arch` and the atoms) unless
//! a path is given.
//!
//! | primitives | cute-rs |
//! |---|---|
//! | `barrier_cta_sync`, `barrier_cta_sync_aligned` | `barrier(id, count)`, or `pipeline::sync(id)` without a count; `sync_threads()` for barrier 0 |
//! | `barrier_cta_arrive`, `barrier_cta_arrive_aligned` | `barrier_arrive_unaligned`, `barrier_arrive` |
//! | `barrier_cta_red`, `barrier_cta_red_aligned` | [`barrier_cta_red`] *here* (always the `.aligned` form) |
//! | `bar_warp_sync` | `sync_warp_with(mask)` |
//! | `barrier_cluster_arrive{,_aligned,_relaxed,_relaxed_aligned}`, `barrier_cluster_wait{,_aligned}` | `cluster_arrive`, `cluster_arrive_aligned`, `cluster_arrive_relaxed`, `cluster_arrive_relaxed_aligned`, `cluster_wait`, `cluster_wait_aligned` |
//! | `elect_sync` | [`elect_sync`] *here*; `elect_one()` for the whole warp |
//! | `shfl_sync` | `shuffle_sync_op` (width and clamp in `mask_and_clamp`); the `is_valid` output is not provided |
//! | `vote_sync`, `match_sync`, `redux_sync` | `vote_{all,any,uni,ballot}_sync`, `match_sync`, `warp_redux_sync` |
//! | `mbarrier_init`, `fence_mbarrier_init` | `mbarrier_init`, `mbarrier_init_fence` |
//! | `mbarrier_arrive` | [`mbarrier_arrive`] *here* (returns the state token); `cute::mbarrier_arrive_with` for counts, peers and scopes |
//! | `mbarrier_arrive_expect_tx`, `mbarrier_expect_tx` | `mbarrier_arrive_and_expect_tx`, `mbarrier_expect_tx` (and their `_with` forms) |
//! | `mbarrier_arrive_drop`, `mbarrier_arrive_drop_expect_tx`, `mbarrier_arrive_drop_nocomplete`, `mbarrier_arrive_nocomplete`, `mbarrier_complete_tx`, `mbarrier_inval` | the same names *here* |
//! | `mbarrier_test_wait`, `mbarrier_try_wait`, `mbarrier_try_wait_parity`, `mbarrier_try_wait_timelimit`, `mbarrier_wait` | the same names *here* (on a parity or a state token) |
//! | `mbarrier_wait_parity` | `mbarrier_wait` (spins until the phase is over) |
//! | `cp_async_mbarrier_arrive` | `cp_async_mbarrier_arrive`, `cp_async_mbarrier_arrive_noinc` |
//! | `cp_async_shared_global`, `cp_async_commit_group`, `cp_async_wait_group` | the `CopyG2SOp` copy atom; `cp_async_commit_group`, `cp_async_wait_group` |
//! | `cp_async_bulk_shared_cluster_global`, `cp_async_bulk_shared_cluster_shared_cta`, `cp_async_bulk_global_shared_cta`, `cp_async_bulk_prefetch` | `cp_async_bulk`, the `CopyBulkS2SOp` atom, `cp_async_bulk_global_shared_cta`, the bulk prefetch of `prefetch_atom` |
//! | `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group` | `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group` |
//! | `cp_async_bulk_tensor_{shared_cluster_global,shared_cta_global,global_shared_cta,prefetch,reduce}` | the TMA atoms: `make_tiled_tma_atom` with the `CopyBulkTensorTile*` ops, `prefetch_atom`, `CopyReduceBulkTensorTileS2GOp` |
//! | `cp_reduce_async_bulk_global_shared_cta` | the `CopyBulkS2GOp` atom with a `ReductionKind` |
//! | `tensormap_replace`, `tensormap_cp_fenceproxy`, `prefetch_tensormap` | `tensormap_replace`, `copy_tensormap` / `cp_fence_tma_desc_release`, `prefetch_tensormap` |
//! | `fence_proxy`, `fence_proxy_acquire`, `fence_proxy_release`, `fence_proxy_sync_restrict`, `fence_proxy_async_{acquire,release}_sync_restrict`, `fence_sync_restrict` | `fence_proxy`, `fence_proxy_with`, `fence_tma_desc_acquire`, `fence_tma_desc_release`; the `sync_restrict` forms are not supported (no cute-rs operation yet) |
//! | `fence_acq_rel` | `fence_acq_rel_{cta,cluster,gpu,sys}` |
//! | `fence_sc_cluster`, `memory_barrier` | [`fence_sc_cluster`], [`memory_barrier`] *here* |
//! | `griddepcontrol`, `setmaxregister` | `griddepcontrol_wait`, `griddepcontrol_launch_dependents`; `setmaxregister_increase`, `setmaxregister_decrease` |
//! | `read_ptx_sreg_{total,dynamic,aggr}_smem_size`, `cluster_ctarank` | `total_smem_size`, `dynamic_smem_size`, `aggr_smem_size`, `block_idx_in_cluster` |
//! | `read_sreg_hw` | not supported: NVIDIA's own NVVM operation, which upstream LLVM does not have |
//! | `mapa` | `map_dsmem_ptr` |
//! | `ldmatrix`, `stmatrix` | the `LdMatrix*` and `StMatrix*` copy atoms |
//! | `load_ext`, `store_ext`, `prefetch_l1`, `prefetch_l2`, `prefetchu` | `load`, `store` (with `LoadOptions`, `StoreOptions`), `prefetch` (with `PrefetchOptions`) |
//! | `red`, `atomicrmw` | `red`, `red_packed`; `atomic_add`, `atomic_cas`, `atomic_fmax`, `atomic_fmin` and their `_with` forms |
//! | `mma_sync`, `mma_sp_sync`, `mma_block_scale` | the warp-level MMA atoms (`MmaF16BF16Op`, `MmaTF32Op`, the sparse and `MmaMXF8F6F4Op` atoms) |
//! | `mma_sp_block_scale` | `prims::mma_sp_block_scale` |
//! | `wgmma_mma_async`, `wgmma_fence_aligned`, `wgmma_commit_group_sync_aligned`, `wgmma_wait_group_sync_aligned`, `mma_smem_desc` | the warpgroup atoms of `nvgpu::warpgroup` with `gemm`, their fence, commit and wait; `prims::WgmmaSmemDesc` |
//! | `tcgen05_mma`, `tcgen05_mma_ws`, `tcgen05_mma_sp`, `tcgen05_mma_ws_sp`, `tcgen05_mma_block_scale`, `tcgen05_mma_sp_block_scale` | `prims::tcgen05_mma` and the other forms of `prims::mma`; the tcgen05 MMA atoms with `gemm` |
//! | `tcgen05_alloc`, `tcgen05_dealloc`, `tcgen05_relinquish_alloc_permit`, `tcgen05_fence`, `tcgen05_wait`, `tcgen05_commit`, `tcgen05_cp`, `tcgen05_ld`, `tcgen05_ld_red`, `tcgen05_st` | `alloc_tmem`, `dealloc_tmem`, `relinquish_tmem_alloc_permit`, `tcgen05_fence_{before,after}_thread_sync`, `fence_view_async_tmem_{load,store}`, the `commit` of the UMMA pipelines, `make_s2t_copy`, `make_tmem_copy` |
//! | `tcgen05_shift` | [`tcgen05_shift`] *here* |
//! | `make_tmem_ptr`, `make_tmem_ptr_from_warp_row_col` | `retrieve_tmem_ptr`; `prims::TmemAddr` |
//! | `clusterlaunchcontrol_query_cancel` | `issue_clc_query`, `clc_response` |
//! | the packed arithmetic (`add_packed_*`, `sub_packed_*`, `mul_packed_*`, `fma_packed_*`, `mul_bf16x2`, `mul`) | `add_packed_f32x2`, `add_packed_f16x2`, `add_packed_bf16x2`, `sub_packed_*`, `mul_packed_f32x2`, `fma_packed_*`, `fma_f16`, `fma_bf16`; the mixed-type forms are not supported |
//! | the conversions (`convert*`, `cvt_*`, `cvt_packfloat*`, `convert_and_pack_integer`) | the `cvt_*` functions of `cute::arch` (`cvt_f32x2_to_f8x2`, `cvt_f32x2_to_f4e2m1x2`, `cvt_f32_tf32`, ...) and `Value::to`; the scaled `cvt_packfloat_*_sf` forms are not supported |
//! | `prmt` | `prmt`, `prmt_with` |
//! | `dot_accumulate_2way`, `dot_accumulate_4way` | the same names *here* |
//! | `nanosleep`, `pmevent`, `breakpoint`, `ptx_add_comment`, `exit` | the same names *here* |
//! | `trace_mark` | not supported: LLVM 21 to 23 have no `trace_mark` operation or intrinsic, and the PTX ISA no instruction for it |
//! | `wmma_load`, `wmma_store` | not supported yet: the fragment layouts are unspecified by PTX (a probed Ampere/Ada mapping is future work); use the `mma.sync` atoms |
//! | `st_bulk` | [`st_bulk`] *here* |
//! | `cvta_to`, `mov_b32` | [`cvta_to`] (`Ptr::tospace`), [`mov_b32`] (`Value::bitcast`) *here* |
//! | `inline_ptx`, `dialect` | `InlinePtx`, `inline_ptx`, `inline_ptx_effect`; [`nvvm_op`] *here* (`MlirOp`) |
//! | `print_runtime` | `printf` |
//! | `cutlass.Array(dtype, n, space=smem / rmem / tmem)` | `alloc_smem`, `make_rmem_tensor`, `alloc_tmem`; module globals for gmem and cmem |
//! | the descriptor builders | `prims::descriptors` |
//!
//! The enums map to: `CTAGroup` to `CtaGroup`, `BarrierRedux` to
//! [`BarrierRedux`], `MemOrder` to `MemoryOrder`, `MemScope` and
//! `MBarrierScope` to `MemScopeKind` (and `FenceScope` for fences),
//! `MBarrierWait` to [`MBarrierWait`], `Proxy` to `FenceProxyKind`,
//! `ReductionOp` to `ReduxKind`, `AtomicOp` to `AtomicType` with the atomic
//! functions, `CpReduceOp` to `ReductionKind`, `DotAccumulateType` to
//! [`DotAccumulateType`] and `CvtaSpace` to `AddressSpace`.

use std::panic::Location;

use crate::builder;
use crate::ir::{DType, FenceScope, MbarrierExtKind, MbarrierWaitKind, Op, PrimMarker};
use crate::tensor::{AddressSpace, Ptr};
use crate::value::{Boolean, Int32, Int64, IntoValue, Numeric, Uint32, Value};

pub use crate::arch::MlirOp;
pub use crate::ir::BarrierRedKind as BarrierRedux;
pub use crate::ir::MbarrierWaitKind as MBarrierWait;
pub use crate::nvgpu::cpasync::CtaGroup as CTAGroup;
pub use crate::print::printf as print_runtime;

// ---------------------------------------------------------------------------
// bar.red
// ---------------------------------------------------------------------------

#[track_caller]
fn barrier_red<T: Numeric>(
    pred: impl IntoValue<bool>,
    barrier_id: impl IntoValue<i32>,
    kind: BarrierRedux,
    thread_count: Option<Int32>,
) -> Value<T> {
    let loc = Location::caller();
    let pred = pred.into_value().checked();
    let id = barrier_id.into_value().checked();
    let threads = thread_count.map(|n| n.checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(T::DTYPE);
        b.push(Op::BarrierRed { result, kind, pred, id, threads }, loc);
        result
    }))
}

/// Wait at named barrier `barrier_id` (0 to 15) until `thread_count`
/// threads (a multiple of 32, counted in whole warps; `None` for every
/// thread of the CTA) have arrived, and return to each the reduction of
/// their predicates (`barrier.cta.red.{popc.u32,and.pred,or.pred}.aligned`,
/// `bar.red`; primitives' `barrier_cta_red` and `barrier_cta_red_aligned`):
/// for [`BarrierRedux::Popc`] how many are true, for `And` whether all are
/// and for `Or` whether any is (1 or 0). Every thread of each participating
/// warp executes it, in converged code.
#[track_caller]
pub fn barrier_cta_red(
    pred: impl IntoValue<bool>,
    barrier_id: impl IntoValue<i32>,
    kind: BarrierRedux,
    thread_count: Option<Int32>,
) -> Uint32 {
    match kind {
        BarrierRedux::Popc => barrier_red::<u32>(pred, barrier_id, kind, thread_count),
        _ => barrier_red::<bool>(pred, barrier_id, kind, thread_count).to::<u32>(),
    }
}

/// [`barrier_cta_red`] with [`BarrierRedux::Popc`]: how many participating
/// threads' predicates are true.
#[track_caller]
pub fn barrier_cta_red_popc(
    pred: impl IntoValue<bool>,
    barrier_id: impl IntoValue<i32>,
    thread_count: Option<Int32>,
) -> Uint32 {
    barrier_red(pred, barrier_id, BarrierRedux::Popc, thread_count)
}

/// [`barrier_cta_red`] with [`BarrierRedux::And`]: whether every
/// participating thread's predicate is true.
#[track_caller]
pub fn barrier_cta_red_and(
    pred: impl IntoValue<bool>,
    barrier_id: impl IntoValue<i32>,
    thread_count: Option<Int32>,
) -> Boolean {
    barrier_red(pred, barrier_id, BarrierRedux::And, thread_count)
}

/// [`barrier_cta_red`] with [`BarrierRedux::Or`]: whether any participating
/// thread's predicate is true.
#[track_caller]
pub fn barrier_cta_red_or(
    pred: impl IntoValue<bool>,
    barrier_id: impl IntoValue<i32>,
    thread_count: Option<Int32>,
) -> Boolean {
    barrier_red(pred, barrier_id, BarrierRedux::Or, thread_count)
}

// ---------------------------------------------------------------------------
// mbarriers
// ---------------------------------------------------------------------------

#[track_caller]
fn mbarrier_ext(mbarrier: &Ptr<u64>, kind: MbarrierExtKind, value: Option<Int32>) -> Option<Int64> {
    let loc = Location::caller();
    let (buffer, offset) = mbarrier.parts();
    let index = crate::tensor::index_operand(offset);
    let value = value.map(|v| v.checked());
    builder::with(|b| {
        let result = kind.has_state().then(|| b.new_value(DType::I64));
        b.push(Op::MbarrierExt { kind, result, buffer, index, value }, loc);
        result
    })
    .map(Value::from_id)
}

/// Arrive once at an mbarrier and return the state token of the phase it
/// arrived in (`mbarrier.arrive.shared.b64 state, [addr]`, sm_80), which the
/// waits of this module take in place of a parity. The arrival count,
/// peer-CTA and scope forms are `cute::mbarrier_arrive_with`.
#[track_caller]
pub fn mbarrier_arrive(mbarrier: &Ptr<u64>) -> Int64 {
    mbarrier_ext(mbarrier, MbarrierExtKind::ArriveState, None).expect("a state")
}

/// Arrive once and lower the arrival count of this phase and of every
/// later one by one (`mbarrier.arrive_drop`, sm_80): a thread that leaves
/// a pipeline for good. Returns the state token.
#[track_caller]
pub fn mbarrier_arrive_drop(mbarrier: &Ptr<u64>) -> Int64 {
    mbarrier_ext(mbarrier, MbarrierExtKind::ArriveDrop, None).expect("a state")
}

/// Make the phase wait for `bytes` more bytes of bulk copies, then
/// [`mbarrier_arrive_drop`] (`mbarrier.arrive_drop.expect_tx`, sm_90).
#[track_caller]
pub fn mbarrier_arrive_drop_expect_tx(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>) -> Int64 {
    mbarrier_ext(mbarrier, MbarrierExtKind::ArriveDropExpectTx, Some(bytes.into_value())).expect("a state")
}

/// `count` arrive-drops that must not complete the phase
/// (`mbarrier.arrive_drop.noComplete`, sm_80; the simulator reports one that
/// would).
#[track_caller]
pub fn mbarrier_arrive_drop_nocomplete(mbarrier: &Ptr<u64>, count: impl IntoValue<i32>) -> Int64 {
    mbarrier_ext(mbarrier, MbarrierExtKind::ArriveDropNoComplete, Some(count.into_value())).expect("a state")
}

/// `count` arrivals that must not complete the phase
/// (`mbarrier.arrive.noComplete`, sm_80; the simulator reports one that
/// would).
#[track_caller]
pub fn mbarrier_arrive_nocomplete(mbarrier: &Ptr<u64>, count: impl IntoValue<i32>) -> Int64 {
    mbarrier_ext(mbarrier, MbarrierExtKind::ArriveNoComplete, Some(count.into_value())).expect("a state")
}

/// Lower the phase's transaction count by `bytes`, as bulk copies landing
/// would (`mbarrier.complete_tx`, sm_90).
#[track_caller]
pub fn mbarrier_complete_tx(mbarrier: &Ptr<u64>, bytes: impl IntoValue<i32>) {
    mbarrier_ext(mbarrier, MbarrierExtKind::CompleteTx, Some(bytes.into_value()));
}

/// Invalidate an mbarrier, so that its memory may be used for something
/// else or initialised again (`mbarrier.inval`, sm_80). Any later operation
/// on it but [`mbarrier_init`](crate::mbarrier_init) is an error.
#[track_caller]
pub fn mbarrier_inval(mbarrier: &Ptr<u64>) {
    mbarrier_ext(mbarrier, MbarrierExtKind::Inval, None);
}

/// What an mbarrier wait waits on: a phase parity (an Int32, or an `i32`
/// constant: the `.parity` forms) or a state token from an arrival (an
/// Int64).
pub trait StateOrPhase {
    /// The value and whether it is a state token.
    #[doc(hidden)]
    fn wait_operand(self) -> (crate::ir::ValueId, bool);
}

impl StateOrPhase for Int32 {
    #[track_caller]
    fn wait_operand(self) -> (crate::ir::ValueId, bool) {
        (self.checked(), false)
    }
}

impl StateOrPhase for i32 {
    #[track_caller]
    fn wait_operand(self) -> (crate::ir::ValueId, bool) {
        (self.into_value().checked(), false)
    }
}

impl StateOrPhase for Int64 {
    #[track_caller]
    fn wait_operand(self) -> (crate::ir::ValueId, bool) {
        (self.checked(), true)
    }
}

#[track_caller]
fn wait(mbarrier: &Ptr<u64>, on: impl StateOrPhase, kind: MbarrierWaitKind, time_limit: Option<Int32>) -> Boolean {
    let loc = Location::caller();
    let (buffer, offset) = mbarrier.parts();
    let index = crate::tensor::index_operand(offset);
    let (parity, state) = on.wait_operand();
    let time_limit = time_limit.map(|t| t.checked());
    Value::from_id(builder::with(|b| {
        let result = b.new_value(DType::Bool);
        b.push(Op::MbarrierWait { result, buffer, index, parity, kind, state, time_limit }, loc);
        result
    }))
}

/// Whether the phase named by `on` (a parity or a state token) is over,
/// without waiting (`mbarrier.test_wait{.parity}`, sm_80).
#[track_caller]
pub fn mbarrier_test_wait(mbarrier: &Ptr<u64>, on: impl StateOrPhase) -> Boolean {
    wait(mbarrier, on, MbarrierWaitKind::Test, None)
}

/// Whether the phase named by `on` is over, after suspending the thread for
/// up to `ticks` nanoseconds if given, or a system-dependent time
/// (`mbarrier.try_wait{.parity}`, sm_90).
#[track_caller]
pub fn mbarrier_try_wait(mbarrier: &Ptr<u64>, on: impl StateOrPhase, ticks: Option<Int32>) -> Boolean {
    wait(mbarrier, on, MbarrierWaitKind::Try, ticks)
}

/// [`mbarrier_try_wait`] on a parity with a suspend-time hint of
/// `time_limit` nanoseconds (`mbarrier.try_wait.parity`, sm_90; primitives'
/// default is 10,000,000).
#[track_caller]
pub fn mbarrier_try_wait_parity(
    mbarrier: &Ptr<u64>,
    phase: impl IntoValue<i32>,
    time_limit: impl IntoValue<i32>,
) -> Boolean {
    wait(mbarrier, phase.into_value(), MbarrierWaitKind::Try, Some(time_limit.into_value()))
}

/// [`mbarrier_try_wait`] on a state token with a suspend-time hint of
/// `time_limit` nanoseconds (`mbarrier.try_wait`, sm_90).
#[track_caller]
pub fn mbarrier_try_wait_timelimit(mbarrier: &Ptr<u64>, state: Int64, time_limit: impl IntoValue<i32>) -> Boolean {
    wait(mbarrier, state, MbarrierWaitKind::Try, Some(time_limit.into_value()))
}

/// One wait of `kind` on a state token: [`mbarrier_test_wait`] or
/// [`mbarrier_try_wait`].
#[track_caller]
pub fn mbarrier_wait(mbarrier: &Ptr<u64>, state: Int64, kind: MBarrierWait) -> Boolean {
    wait(mbarrier, state, kind, None)
}

// ---------------------------------------------------------------------------
// Markers, hints, exit and fences
// ---------------------------------------------------------------------------

#[track_caller]
fn marker(marker: PrimMarker) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::PrimMarker { marker }, loc));
}

/// Suspend the thread for about `duration` nanoseconds, at most a
/// millisecond (`nanosleep.u32`, sm_70): a back-off in spin loops. The
/// simulator yields the warp to the others.
#[track_caller]
pub fn nanosleep(duration: impl IntoValue<u32>) {
    let t = duration.into_value().bitcast::<i32>().checked();
    marker(PrimMarker::Nanosleep(t));
}

/// Trigger performance-monitor event `event_id` (0 to 15; `pmevent`).
#[track_caller]
pub fn pmevent(event_id: u32) {
    assert!(event_id < 16, "cute::prims::pmevent: event {event_id} is out of range (0 to 15)");
    marker(PrimMarker::Pmevent(1 << event_id));
}

/// Trigger the performance-monitor events of the bits of `mask`
/// (`pmevent.mask`).
#[track_caller]
pub fn pmevent_mask(mask: u16) {
    marker(PrimMarker::Pmevent(mask));
}

/// Suspend the thread for a debugger (`brkpt`).
#[track_caller]
pub fn breakpoint() {
    marker(PrimMarker::Breakpoint);
}

/// A `// comment` line in the PTX, where this call is (one line of text).
#[track_caller]
pub fn ptx_add_comment(comment: &str) {
    assert!(!comment.contains(['\n', '\r']), "cute::prims::ptx_add_comment: a comment is one line");
    marker(PrimMarker::Comment(comment.to_string()));
}

/// The executing threads stop (`exit`). A CTA ends when all its threads
/// have; an mbarrier arrival a thread owed is never made, so a wait for it
/// hangs (the simulator reports the hang and names the exit). The
/// simulator runs an exit by a whole warp only.
#[track_caller]
pub fn exit() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Exit, loc));
}

/// A sequentially consistent fence at `scope` (`membar.{cta,gl,sys}`, which
/// is `fence.sc.{cta,gpu,sys}`; `fence.sc.cluster` for the cluster, sm_90).
#[track_caller]
pub fn memory_barrier(scope: FenceScope) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Fence { scope, sc: true }, loc));
}

/// `fence.sc.cluster` (sm_90).
#[track_caller]
pub fn fence_sc_cluster() {
    memory_barrier(FenceScope::Cluster);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Whether the bytes or halves of a dot-product operand are signed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DotAccumulateType {
    Signed,
    Unsigned,
}

impl DotAccumulateType {
    fn letter(self) -> char {
        match self {
            DotAccumulateType::Signed => 's',
            DotAccumulateType::Unsigned => 'u',
        }
    }

    /// Element `i` of `width` bits of `word`, extended to an i64.
    fn element(self, word: u64, i: u32, width: u32) -> i64 {
        let raw = (word >> (i * width)) & ((1 << width) - 1);
        match self {
            DotAccumulateType::Signed => ((raw << (64 - width)) as i64) >> (64 - width),
            DotAccumulateType::Unsigned => raw as i64,
        }
    }
}

/// The four-way byte dot product `c + a.b` (`dp4a.{a}32.{b}32`, sm_61): `a`
/// and `b` each pack four 8-bit integers, signed or not as `a_type` and
/// `b_type` say; the sum wraps to 32 bits.
#[track_caller]
pub fn dot_accumulate_4way(
    a: impl IntoValue<u32>,
    a_type: DotAccumulateType,
    b: impl IntoValue<u32>,
    b_type: DotAccumulateType,
    c: impl IntoValue<i32>,
) -> Int32 {
    let name = format!("llvm.nvvm.idp4a.{}.{}", a_type.letter(), b_type.letter());
    MlirOp::new(&format!("llvm.call_intrinsic \"{name}\"(%0, %1, %2) : (i32, i32, i32) -> i32"))
        .operand(a.into_value())
        .operand(b.into_value())
        .operand(c.into_value())
        .result::<i32>()
        .pure()
        .sim(move |r, _| {
            let sum = (0..4).map(|i| a_type.element(r[0], i, 8) * b_type.element(r[1], i, 8)).sum::<i64>();
            vec![(r[2] as i64).wrapping_add(sum) as u32 as u64]
        })
        .emit()
}

/// The two-way dot product `c + a.b` of two 16-bit integers of `a` with two
/// bytes of `b` (`dp2a.{lo,hi}`, sm_61): the low two bytes of `b`, or with
/// `b_hi` the high two. The sum wraps to 32 bits.
#[track_caller]
pub fn dot_accumulate_2way(
    a: impl IntoValue<u32>,
    a_type: DotAccumulateType,
    b: impl IntoValue<u32>,
    b_type: DotAccumulateType,
    c: impl IntoValue<i32>,
    b_hi: bool,
) -> Int32 {
    let name = format!("llvm.nvvm.idp2a.{}.{}", a_type.letter(), b_type.letter());
    MlirOp::new(&format!("llvm.call_intrinsic \"{name}\"(%0, %1, %2, %3) : (i32, i32, i1, i32) -> i32"))
        .operand(a.into_value())
        .operand(b.into_value())
        .operand(b_hi)
        .operand(c.into_value())
        .result::<i32>()
        .pure()
        .sim(move |r, _| {
            let first = if b_hi { 2 } else { 0 };
            let sum = (0..2).map(|i| a_type.element(r[0], i, 16) * b_type.element(r[1], first + i, 8)).sum::<i64>();
            vec![(r[3] as i64).wrapping_add(sum) as u32 as u64]
        })
        .emit()
}

/// The bits of `a` as another 32-bit type (`mov.b32`; `Value::bitcast`).
#[track_caller]
pub fn mov_b32<T: Numeric, U: Numeric>(a: impl IntoValue<T>) -> Value<U> {
    assert!(
        T::DTYPE.bits() == 32 && U::DTYPE.bits() == 32,
        "cute::prims::mov_b32 moves 32-bit values, not {} to {}",
        T::DTYPE,
        U::DTYPE
    );
    a.into_value().bitcast::<U>()
}

// ---------------------------------------------------------------------------
// Warp, memory and tensor memory
// ---------------------------------------------------------------------------

/// True in one lane of `membermask`, the lowest of the lanes executing it
/// (`elect.sync`): every lane of the mask must execute it. With the whole
/// warp it is [`elect_one`](crate::elect_one).
#[track_caller]
pub fn elect_sync(membermask: impl IntoValue<u32>) -> Boolean {
    let mask = membermask.into_value();
    if mask.as_constant() == Some(crate::arch::FULL_MASK) {
        return crate::arch::elect_one();
    }
    let active = crate::arch::activemask() & mask;
    let lowest = active & (Uint32::constant(0) - active);
    lowest.eq(crate::arch::lanemask_eq())
}

/// `ptr` as a pointer of address space `space` (`cvta` / `cvta.to`;
/// `Ptr::tospace`): one of the two spaces is the generic one.
#[track_caller]
pub fn cvta_to<T: Numeric>(ptr: &Ptr<T>, space: AddressSpace) -> Ptr<T> {
    ptr.tospace(space)
}

/// Zero `size` bytes of shared memory from `ptr` (`st.bulk.weak.shared::cta
/// [addr], size, 0`, sm_100 and PTX 8.6): the address 16-byte aligned, the
/// size a positive multiple of 16. The only initial value PTX has is 0.
#[track_caller]
pub fn st_bulk<T: Numeric>(ptr: &Ptr<T>, size: impl IntoValue<i64>) {
    let loc = Location::caller();
    let (buffer, offset) = ptr.parts();
    let index = crate::tensor::index_operand(offset);
    let size = size.into_value().checked();
    builder::with(|b| b.push(Op::StBulk { buffer, index, elem: T::DTYPE, size }, loc));
}

/// Shift the rows of each 32-lane subpartition of the tensor-memory matrix
/// at `taddr` down by one lane, the last row of each staying
/// (`tcgen05.shift.cta_group::{1,2}.down`, sm_100a family). Issued by one
/// thread, asynchronously; `tcgen05.commit` tracks it like an MMA. Not
/// simulated.
#[track_caller]
pub fn tcgen05_shift(taddr: impl Into<super::TmemOperand>, group: CTAGroup) {
    let loc = Location::caller();
    let taddr = taddr.into().value().checked();
    let two_cta = group == CTAGroup::TWO;
    builder::with(|b| b.push(Op::Tcgen05Shift { taddr, two_cta }, loc));
}

/// The raw NVVM escape (primitives' `dialect`): an MLIR operation written as
/// text, such as `nvvm.*` or `llvm.call_intrinsic "llvm.nvvm.*"(...)`, with
/// operands `%0, %1, ...` ([`MlirOp`]). It is simulated only when given a
/// closure ([`MlirOp::sim`]).
#[track_caller]
pub fn nvvm_op(text: &str) -> MlirOp {
    MlirOp::new(text)
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pipeline whose producers fill stages with `cp.async` copies
//! ([`PipelineCpAsync`]).

use crate::tensor::Ptr;
use crate::value::{Boolean, Int32};

use super::{
    CooperativeGroup, MbarrierArray, Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineBarriers, PipelineOp,
    PipelineState, StageState,
};

/// The optional arguments of [`PipelineCpAsync::create`], the same as
/// [`PipelineAsync`]'s: barrier storage (`None` allocates it with
/// [`alloc_reserved_mbarrier`](super::alloc_reserved_mbarrier)), the
/// producer and consumer masks, `defer_sync` and a profiling name.
pub type PipelineCpAsyncOptions = PipelineAsyncOptions;

/// A pipeline whose producers load each stage with `cp.async` (CuTe DSL's
/// `cutlass.pipeline.PipelineCpAsync`): [`PipelineAsync`] with full barriers
/// arrived on by [`PipelineOp::AsyncLoad`], so a producer's commit is
/// `cp.async.mbarrier.arrive.noinc`, an arrival that fires once every
/// `cp.async` the thread issued before it has landed. The consumers arrive
/// on the empty barriers as ordinary threads ([`PipelineOp::AsyncThread`]).
///
/// # The `.noinc` contract
///
/// `.noinc` does not raise the phase's pending count, so each producer
/// thread's arrival is one of the full barrier's count: the count is the
/// producer group's thread count ([`CooperativeGroup::thread_arrive_count`],
/// 32 per warp), and every one of those threads must, for each stage,
///
/// 1. [`producer_acquire`](Pipeline::producer_acquire) it,
/// 2. issue its `cp.async` copies into it, then
/// 3. [`producer_commit`](Pipeline::producer_commit) exactly once.
///
/// A commit issued before the stage's copies does not cover them (the
/// consumer may read the stage before they land, which the simulator
/// reports), and a group counted in warps where each thread arrives
/// completes the phase early. No `cp.async` group commit or wait is needed:
/// the arrival tracks the copies.
///
/// Waits follow [`PipelineAsync`]'s: a kernel built for a target below
/// sm_90 spins on `mbarrier.test_wait`, so the pipeline runs on SM80 to
/// SM89 (`cp.async.mbarrier.arrive` needs sm_80).
///
/// ```
/// use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions};
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("cp_async_ring").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
///     let src = param_tensor::<f32>("src", Layout::col_major(6 * 128));
///     let out = param_tensor::<f32>("out", Layout::col_major(6 * 128));
///     let warp = CooperativeGroup::new(Agent::Warp, 1);
///     let pipeline = PipelineCpAsync::create(2, warp, warp, PipelineCpAsyncOptions::default());
///     let ring = make_smem_tensor::<f32>(Layout::col_major(2 * 128));
///     let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
///     let (producer, consumer) = pipeline.make_participants();
///     let lane = lane_idx();
///     let chunk = Layout::col_major(4);
///     if_else(
///         warp_idx().eq(0),
///         || {
///             let mut p = for_range_carry(0, 6, 1, producer, |k, mut p| {
///                 let handle = p.acquire_and_advance(None);
///                 let from = src.ptr().offset(k * 128 + lane * 4);
///                 let to = ring.ptr().offset(handle.index() * 128 + lane * 4);
///                 copy(&atom, &make_tensor(&from, chunk.clone()), &make_tensor(&to, chunk.clone()));
///                 handle.commit();
///                 p
///             });
///             p.tail();
///         },
///         || {
///             for_range_carry(0, 6, 1, consumer, |k, mut c| {
///                 let handle = c.wait_and_advance(None);
///                 for e in 0..4 {
///                     out.store(k * 128 + lane * 4 + e, ring.load(handle.index() * 128 + lane * 4 + e));
///                 }
///                 handle.release();
///                 c
///             });
///         },
///     );
/// });
/// let src: Vec<f32> = (0..6 * 128).map(|i| i as f32).collect();
/// let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0f32; 6 * 128])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[1].to_vec::<f32>(), src);
/// ```
#[derive(Clone, Debug)]
pub struct PipelineCpAsync {
    inner: PipelineAsync,
}

impl PipelineCpAsync {
    /// A pipeline of `num_stages` stages between `producer_group` (the
    /// threads that issue the copies) and `consumer_group` (CuTe DSL's
    /// `PipelineCpAsync.create`): full barriers counting the producer
    /// group's threads at the storage, empty ones counting the consumer
    /// group's `num_stages` further on, initialised from warp 0 and, unless
    /// `defer_sync`, published to the block as [`PipelineAsync::create`]
    /// does (every thread must reach it).
    ///
    /// Panics for 0 stages.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: PipelineCpAsyncOptions,
    ) -> PipelineCpAsync {
        let inner = PipelineAsync::create_with_ops(
            "PipelineCpAsync",
            num_stages,
            (PipelineOp::AsyncLoad, producer_group),
            (PipelineOp::AsyncThread, consumer_group),
            options,
        );
        PipelineCpAsync { inner }
    }

    /// The underlying full/empty pipeline.
    pub fn as_pipeline_async(&self) -> &PipelineAsync {
        &self.inner
    }

    /// The full barriers, which the producers' copies arrive on.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        self.inner.sync_object_full()
    }

    /// The empty barriers, which the consumers arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        self.inner.sync_object_empty()
    }

    /// The cluster rank the producers' arrivals go to, if not this CTA.
    pub fn producer_mask(&self) -> Option<Int32> {
        self.inner.producer_mask()
    }

    /// The cluster rank the consumers' arrivals go to, if not this CTA.
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.inner.consumer_mask()
    }
}

impl Pipeline for PipelineCpAsync {
    type AcquireOptions = ();

    fn num_stages(&self) -> usize {
        self.inner.num_stages()
    }

    fn producer_group(&self) -> Option<CooperativeGroup> {
        self.inner.producer_group()
    }

    fn consumer_group(&self) -> Option<CooperativeGroup> {
        self.inner.consumer_group()
    }

    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        self.inner.producer_acquire_with(state, token, ());
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.inner.producer_try_acquire(state)
    }

    /// `cp.async.mbarrier.arrive.noinc` on the stage's full barrier: the
    /// arrival lands once this thread's earlier `cp.async` copies have.
    #[track_caller]
    fn producer_commit<S: StageState>(&self, state: &S) {
        self.inner.producer_commit(state);
    }

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        self.inner.consumer_wait(state, token);
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.inner.consumer_try_wait(state)
    }

    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.inner.consumer_release(state);
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        self.inner.producer_tail(state);
    }
}

impl PipelineBarriers for PipelineCpAsync {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.inner.producer_get_barrier(state)
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.inner.consumer_get_barrier(state)
    }
}

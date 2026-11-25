// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The pipeline's protocol follows PipelineCLCFetchAsync in CUTLASS's
// include/cutlass/pipeline/sm100_pipeline.hpp (Copyright (c) 2023 - 2026
// NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause; see THIRD-PARTY-NOTICES.md).

//! The pipeline that hands cluster launch control responses from a cluster's
//! scheduler warp to its consumers (roadmap part m10-clc-scheduling):
//! [`PipelineClcFetchAsync`].

use crate::arch::{MbarrierOptions, WARP_SIZE, mbarrier_arrive_with, mbarrier_init_fence, thread_idx};
use crate::control::if_;
use crate::ir::MemScopeKind;
use crate::layout::Layout;
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32, IntoValue};

use super::async_pipeline::wait_unless;
use super::mbarrier::portable_waits;
use super::{
    Agent, BarrierParam, CooperativeGroup, MbarrierArray, MbarrierArrayOptions, Pipeline, PipelineBarriers, PipelineOp,
    PipelineState, StageState, agent_sync, alloc_reserved_mbarrier,
};

/// The optional arguments of [`PipelineClcFetchAsync::create`]. CuTe DSL's
/// `producer_mask` and `consumer_mask` are left out: `create` replaces both
/// (the producer's lanes each arm one CTA, the consumers release to CTA 0).
#[derive(Clone, Debug, Default)]
pub struct PipelineClcFetchAsyncOptions {
    /// Shared memory for the `2 * num_stages` mbarriers: full barriers
    /// first, then empty ones. `None` allocates them with
    /// [`alloc_reserved_mbarrier`].
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster's CTAs as a static `(V, M, N, L)` layout; only its size
    /// matters. `None` is a single CTA.
    pub cta_layout_vmnk: Option<Layout>,
    /// Leave out the initialisation sync, for a kernel that publishes
    /// several pipelines' barriers at once (in a cluster, with a cluster
    /// barrier).
    pub defer_sync: bool,
    /// A name for profiling; the barrier arrays are named `{name}.full` and
    /// `{name}.empty`.
    pub name: String,
}

/// The cluster launch control pipeline (CuTe DSL's
/// `cutlass.pipeline.PipelineClcFetchAsync`; sm_100): a ring of
/// `num_stages` 16-byte try_cancel responses, filled by the scheduler warp
/// of CTA 0 of the cluster and read by consumers in every CTA.
///
/// * **Producer** (the scheduler warp, in CTA 0 only, all 32 lanes):
///   [`producer_acquire`](Pipeline::producer_acquire) waits on CTA 0's
///   empty barrier, then lane `i` (for `i` below the cluster size) arms
///   CTA `i`'s full barrier with an arrival that expects `tx_count` bytes
///   (the 16-byte response; [`PipelineOp::ClcLoad`]). Then one lane issues
///   the multicast query ([`issue_clc_query`](crate::arch::issue_clc_query),
///   usually through
///   [`ClcDynamicPersistentTileScheduler::advance_to_next_work`](crate::utils::ClcDynamicPersistentTileScheduler::advance_to_next_work))
///   with [`producer_get_barrier`](PipelineBarriers::producer_get_barrier),
///   whose response lands in every CTA and completes each full barrier's
///   phase. [`producer_commit`](Pipeline::producer_commit) does nothing.
/// * **Consumers** wait on their own CTA's full barrier, read the response
///   and release to CTA 0's empty barrier (at cluster scope in a cluster,
///   so that the release orders their reads before the next response's
///   write).
///
/// So the full barriers count one arrival (`producer_group` is one thread)
/// and the empty barrier's count (`consumer_group`) must cover every
/// consumer thread of the whole cluster.
#[derive(Clone, Debug)]
pub struct PipelineClcFetchAsync {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    cluster_size: usize,
    producer_mask: Int32,
    is_signaling_thread: Boolean,
}

impl PipelineClcFetchAsync {
    /// A pipeline of `num_stages` stages, each full phase expecting
    /// `tx_count` bytes (CuTe DSL's `PipelineClcFetchAsync.create`): full
    /// barriers at the storage, empty ones `num_stages` further on,
    /// initialised from warp 0, then published unless
    /// [`defer_sync`](PipelineClcFetchAsyncOptions::defer_sync):
    /// `fence.mbarrier_init` (not below sm_90) and a CTA barrier for one CTA,
    /// a relaxed cluster arrive and a cluster wait for a cluster (which
    /// every thread of every CTA must reach).
    ///
    /// Panics for 0 stages and a dynamic cluster layout.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: impl Into<BarrierParam>,
        options: PipelineClcFetchAsyncOptions,
    ) -> PipelineClcFetchAsync {
        assert!(num_stages > 0, "cute: PipelineClcFetchAsync::create: a pipeline needs at least one stage, not 0");
        let _at = crate::builder::LocationScope::new();
        let cluster_size = match &options.cta_layout_vmnk {
            None => 1,
            Some(layout) => {
                assert!(
                    layout.is_static(),
                    "cute: PipelineClcFetchAsync::create: the cluster layout (cta_layout_vmnk) must be static"
                );
                layout.size()
            }
        };
        crate::profiling::register_barrier(&options.name, num_stages, "PipelineClcFetchAsync");
        let storage = options.barrier_storage.clone().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
        let named = |side: &str| MbarrierArrayOptions {
            name: if options.name.is_empty() { String::new() } else { format!("{}.{side}", options.name) },
            ..Default::default()
        };
        let sync_object_full = MbarrierArray::with_options(
            &storage,
            num_stages,
            (PipelineOp::ClcLoad, producer_group),
            tx_count,
            named("full"),
        );
        let sync_object_empty = MbarrierArray::with_options(
            &storage.offset(num_stages),
            num_stages,
            (PipelineOp::AsyncThread, consumer_group),
            0,
            named("empty"),
        );
        // Lane i of the producer warp arms CTA i's full barrier.
        let lane = thread_idx().x % WARP_SIZE as i32;
        let is_signaling_thread =
            if cluster_size >= WARP_SIZE { Boolean::from(true) } else { lane.lt(cluster_size as i32) };
        if !options.defer_sync {
            if !portable_waits() {
                mbarrier_init_fence();
            }
            if cluster_size == 1 {
                agent_sync(Agent::ThreadBlock, false);
            } else {
                agent_sync(Agent::ThreadBlockCluster, true);
            }
        }
        PipelineClcFetchAsync {
            sync_object_full,
            sync_object_empty,
            num_stages,
            cluster_size,
            producer_mask: lane,
            is_signaling_thread,
        }
    }

    /// The full barriers, which the responses complete.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers (CTA 0's are the ones used).
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The cluster rank whose full barrier this thread arms in
    /// [`producer_acquire`](Pipeline::producer_acquire): its lane.
    pub fn producer_mask(&self) -> Int32 {
        self.producer_mask
    }

    /// The cluster rank the consumers release to: 0.
    pub fn consumer_mask(&self) -> Int32 {
        Int32::from(0)
    }

    /// Whether this thread arms a full barrier (its lane is below the
    /// cluster size).
    pub fn is_signaling_thread(&self) -> Boolean {
        self.is_signaling_thread
    }

    /// The cluster's CTA count (the layout's size).
    pub fn cluster_size(&self) -> usize {
        self.cluster_size
    }
}

impl Pipeline for PipelineClcFetchAsync {
    type AcquireOptions = ();

    fn num_stages(&self) -> usize {
        self.num_stages
    }

    fn producer_group(&self) -> Option<CooperativeGroup> {
        Some(self.sync_object_full.group())
    }

    fn consumer_group(&self) -> Option<CooperativeGroup> {
        Some(self.sync_object_empty.group())
    }

    /// Wait until the stage is empty (unless the token says so), then each
    /// signalling lane arms its CTA's full barrier with the bytes to expect.
    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
        let dst = (self.cluster_size > 1).then_some(self.producer_mask);
        let arm = || self.sync_object_full.arrive(state.index(), dst, None);
        match self.is_signaling_thread.as_constant() {
            Some(true) => arm(),
            Some(false) => {}
            None => if_(self.is_signaling_thread, arm),
        }
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_empty.try_wait(state.index(), state.phase())
    }

    /// Nothing: the response's `complete_tx` completes the full barriers'
    /// phase.
    fn producer_commit<S: StageState>(&self, _: &S) {}

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.sync_object_full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_full.try_wait(state.index(), state.phase())
    }

    /// Arrive on CTA 0's empty barrier: this CTA's for one CTA, through
    /// `mapa` with a cluster-scope release in a cluster.
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        if self.cluster_size == 1 {
            self.sync_object_empty.arrive(state.index(), None, None);
        } else {
            mbarrier_arrive_with(
                &self.sync_object_empty.get_barrier(state.index()),
                1,
                MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(0.into_value()),
                    scope: Some(MemScopeKind::Cluster),
                    ..Default::default()
                },
            );
        }
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        for _ in 0..self.num_stages {
            self.sync_object_empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for PipelineClcFetchAsync {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/pipeline/sm100_pipeline.hpp (PipelineTmaUmmaAsync,
//   PipelineUmmaConsumerAsync and PipelineAsyncUmma: which barrier each side
//   arrives on and how, the leader-CTA arming of the full barrier, the
//   multicast arrival masks and the accumulator pipeline's producer tail),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text.

//! The Blackwell pipelines, whose one side
//! is the `tcgen05` MMA warp and arrives with `tcgen05.commit`:
//! [`PipelineTmaUmma`] (TMA loads feeding the MMAs), [`PipelineAsyncUmma`]
//! (ordinary threads feeding the MMAs) and [`PipelineUmmaAsync`] (the MMAs'
//! accumulator in TMEM handed to epilogue threads).
//!
//! Each takes the cluster as a `(V, M, N, L)` layout; a `V` mode of size 2
//! (a CTA pair running `cta_group::2` MMAs) makes the commits
//! `cta_group::2`. In a CTA pair (simulated by `tests/cta_pair_pipelines.rs` and the `blackwell_fp16_gemm`
//! example's `fp16_gemm_1` configuration):
//!
//! * [`PipelineTmaUmma`]: only the even CTA (`block_idx.x % V == 0`) arms the
//!   full barrier, with both CTAs' bytes (the pair's TMA loads report to its
//!   barrier); the release commit is multicast to the CTAs sharing this
//!   CTA's tiles and its peer's.
//! * [`PipelineAsyncUmma`]: producers arrive on the leading CTA of the pair
//!   (cluster rank `rank / V * V`); the release commit goes to the pair.
//! * [`PipelineUmmaAsync`]: the commit carries the TMEM-sync mask (the pair,
//!   mode 0 of the cluster layout), so it arrives on both CTAs' full
//!   barriers; the epilogue threads of both CTAs release to the even CTA
//!   (`rank / 2 * 2`), whose producer alone runs the tail.
//!
//! The leader of [`PipelineTmaUmma`] comes from `block_idx.x`, the others
//! from the CTA's rank in the cluster (`%cluster_ctarank`), as in CuTe DSL;
//! both agree when the pair is consecutive along x, as a cluster layout
//! from `tiled_divide(cluster, (thr_id,))` makes it.

use crate::arch::{block_idx, block_idx_in_cluster, make_warp_uniform, mbarrier_init_fence};
use crate::control::if_;
use crate::layout::image_mask::create_tma_multicast_mask;
use crate::layout::{Coord, IndexExpr, Layout, Tree, make_layout_image_mask};
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32, IntoValue};

use super::async_pipeline::wait_unless;
use super::mbarrier::portable_waits;
use super::{
    Agent, BarrierParam, CooperativeGroup, CtaGroup, MbarrierArray, MbarrierArrayOptions, Pipeline, PipelineBarriers,
    PipelineOp, PipelineState, StageState, agent_sync, alloc_reserved_mbarrier,
};
use crate::arch::WARP_SIZE;

/// The `(1, 1, 1, 1)` cluster of a single CTA.
fn single_cta() -> Layout {
    Layout::col_major((1usize, 1usize, 1usize, 1usize))
}

/// The cluster layout, checked: static and rank 4.
#[track_caller]
pub(super) fn cluster_layout(kind: &str, layout: Option<&Layout>) -> Layout {
    let layout = layout.cloned().unwrap_or_else(single_cta);
    assert!(layout.is_static(), "cute: {kind}::create: the cluster layout (cta_layout_vmnk) must be static");
    assert!(layout.rank() == 4, "cute: {kind}::create: the cluster layout {layout} must be (V, M, N, L), rank 4");
    layout
}

/// The size of the cluster layout's `V` mode.
pub(super) fn v_size(layout: &Layout) -> usize {
    layout.shape().modes()[0].size()
}

/// The CTA group of the commits: two CTAs when `V` has size 2.
pub(super) fn cta_group_of(layout: &Layout) -> CtaGroup {
    if v_size(layout) == 1 { CtaGroup::One } else { CtaGroup::Two }
}

/// This CTA's rank in the cluster, warp-uniform.
#[track_caller]
fn cta_rank() -> Int32 {
    make_warp_uniform(block_idx_in_cluster())
}

/// This CTA's flat `(V, M, N, L)` coordinate in the cluster.
#[track_caller]
fn cta_coord(layout: &Layout) -> Vec<IndexExpr> {
    match layout.get_flat_coord(cta_rank()) {
        Tree::Branch(modes) => modes
            .into_iter()
            .map(|m| match m {
                Tree::Leaf(e) => e,
                other => panic!("cute: the cluster coordinate has a nested mode {other}"),
            })
            .collect(),
        Tree::Leaf(e) => vec![e],
    }
}

/// A coordinate from its entries.
fn coord(entries: &[IndexExpr]) -> Coord {
    Tree::Branch(entries.iter().cloned().map(Tree::Leaf).collect())
}

/// The coordinate of the other CTA of this CTA's pair: `V` flipped
/// (CuTe DSL's `coord[0] ^ 1`). Without pairs (`V` of size 1) the CTA is its
/// own peer: CuTe DSL's flipped coordinate then lands on a stride-0 `V` mode
/// (the 1-CTA MMA's `thr_id` is `1:0`), but a cluster layout written by hand
/// may give `V` a nonzero stride, which would name another CTA.
#[track_caller]
fn peer_coord(layout: &Layout, c: &[IndexExpr]) -> Vec<IndexExpr> {
    if v_size(layout) == 1 {
        return c.to_vec();
    }
    let v: Int32 = IntoValue::<i32>::into_value(c[0].clone()) ^ 1;
    let mut out = c.to_vec();
    out[0] = IndexExpr::from(v);
    out
}

/// The multicast mask of mode `mode` from coordinate `c`, as an `Int32`.
#[track_caller]
fn mcast_mask(layout: &Layout, c: &[IndexExpr], mode: usize) -> Int32 {
    create_tma_multicast_mask(layout, coord(c), mode).to::<i32>()
}

/// Whether this CTA leads its pair (CuTe DSL's `_compute_is_leader_cta`):
/// `block_idx.x % size(V) == 0`.
#[track_caller]
pub(super) fn is_leader_cta(layout: &Layout) -> Boolean {
    let v = v_size(layout) as i32;
    (block_idx().x % v).eq(0)
}

/// Publish the barriers unless deferred: the init fence (not below sm_90)
/// and a CTA barrier for one CTA, a relaxed cluster arrive and a wait for a
/// cluster.
#[track_caller]
pub(super) fn init_sync(defer_sync: bool, layout: &Layout) {
    if defer_sync {
        return;
    }
    if !portable_waits() {
        mbarrier_init_fence();
    }
    if layout.size() == 1 {
        agent_sync(Agent::ThreadBlock, false);
    } else {
        agent_sync(Agent::ThreadBlockCluster, true);
    }
}

/// The full and empty barrier arrays of a pipeline: full barriers at the
/// storage, empty ones `num_stages` further on.
#[track_caller]
pub(super) fn barrier_arrays(
    kind: &str,
    num_stages: usize,
    storage: Option<&Ptr<u64>>,
    producer: (PipelineOp, CooperativeGroup),
    consumer: (PipelineOp, CooperativeGroup),
    tx_count: BarrierParam,
    name: &str,
) -> (MbarrierArray, MbarrierArray) {
    assert!(num_stages > 0, "cute: {kind}::create: a pipeline needs at least one stage, not 0");
    crate::profiling::register_barrier(name, num_stages, kind);
    let storage = storage.cloned().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
    let named = |side: &str| MbarrierArrayOptions {
        name: if name.is_empty() { String::new() } else { format!("{name}.{side}") },
        ..Default::default()
    };
    let full = MbarrierArray::with_options(&storage, num_stages, producer, tx_count, named("full"));
    let empty = MbarrierArray::with_options(&storage.offset(num_stages), num_stages, consumer, 0, named("empty"));
    (full, empty)
}

/// The optional arguments of [`PipelineTmaUmma::create`].
#[derive(Clone, Debug)]
pub struct PipelineTmaUmmaOptions {
    /// Shared memory for the `2 * num_stages` mbarriers (full, then empty);
    /// `None` allocates them with [`alloc_reserved_mbarrier`].
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster as a static `(V, M, N, L)` layout; `None` is one CTA.
    pub cta_layout_vmnk: Option<Layout>,
    /// Whether the stages are multicast along M and along N (each 0 or 1,
    /// not both 0). Default `(1, 1)`.
    pub mcast_mode_mn: (u32, u32),
    /// Count the empty barriers' arrivals from the consumer warps: the
    /// warps times the CTAs multicasting to this one (see [`PipelineTmaAsyncOptions::enable_multicast_signaling`](super::PipelineTmaAsyncOptions::enable_multicast_signaling)).
    pub enable_multicast_signaling: bool,
    /// Leave out the initialisation sync.
    pub defer_sync: bool,
    /// A name for profiling (`{name}.full`, `{name}.empty`).
    pub name: String,
}

impl Default for PipelineTmaUmmaOptions {
    fn default() -> Self {
        PipelineTmaUmmaOptions {
            barrier_storage: None,
            cta_layout_vmnk: None,
            mcast_mode_mn: (1, 1),
            enable_multicast_signaling: false,
            defer_sync: false,
            name: String::new(),
        }
    }
}

/// The pipeline of a Blackwell GEMM mainloop (CuTe DSL's
/// `cutlass.pipeline.PipelineTmaUmma`): TMA loads fill each stage, the
/// `tcgen05` MMA warp drains it.
///
/// * The full barriers ([`PipelineOp::TmaLoad`]) count the producer group
///   and expect `tx_count` bytes. [`producer_acquire`](Pipeline::producer_acquire)
///   waits on the stage's empty barrier, then, on the leader CTA of a pair
///   only (every CTA when `V = 1`), the warp's elected lane arrives on the
///   full barrier and sets the bytes to expect (the acquire option
///   overrides the count for one acquire). The TMA loads' `complete_tx`
///   finish the phase, so [`producer_commit`](Pipeline::producer_commit)
///   does nothing.
/// * The empty barriers ([`PipelineOp::TCGen05Mma`]) are arrived on by
///   [`consumer_release`](Pipeline::consumer_release): the MMA warp's
///   elected lane runs `tcgen05.commit` (multicast to the CTAs that share
///   the stage, in a cluster), so the stage is free once the MMAs issued
///   before have read it.
///
/// The consumer group is the threads that commit, usually
/// `CooperativeGroup::new(Agent::Thread, 1)` per multicast partner.
#[derive(Clone, Debug)]
pub struct PipelineTmaUmma {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    consumer_mask: Option<Int32>,
    is_leader_cta: Boolean,
    cta_group: CtaGroup,
}

/// The multicast arrival mask of [`PipelineTmaUmma`] (CuTe DSL's
/// `_compute_mcast_arrival_mask`): the CTAs that receive this CTA's A tile
/// (mode 2) or B tile (mode 1), and those of its pair's peer.
#[track_caller]
pub(super) fn mcast_arrival_mask(layout: &Layout, mcast_mode_mn: (u32, u32)) -> Int32 {
    assert!(
        mcast_mode_mn.0 <= 1 && mcast_mode_mn.1 <= 1 && mcast_mode_mn != (0, 0),
        "cute: PipelineTmaUmma::create: mcast_mode_mn is (1, 0), (0, 1) or (1, 1), not {mcast_mode_mn:?}"
    );
    let c = cta_coord(layout);
    let peer = peer_coord(layout, &c);
    let a = mcast_mask(layout, &c, 2) | mcast_mask(layout, &peer, 2);
    let b = mcast_mask(layout, &c, 1) | mcast_mask(layout, &peer, 1);
    match mcast_mode_mn {
        (1, 1) => a | b,
        (0, 1) => b,
        _ => a,
    }
}

impl PipelineTmaUmma {
    /// A pipeline of `num_stages` stages between a TMA `producer_group`
    /// and a `tcgen05` MMA `consumer_group`, each full phase expecting
    /// `tx_count` bytes (CuTe DSL's `PipelineTmaUmma.create`).
    ///
    /// Panics for 0 stages, a dynamic or non-rank-4 cluster layout, and
    /// with `enable_multicast_signaling`, a consumer count that is not a
    /// whole number of warps or a cluster without multicast partners.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: impl Into<BarrierParam>,
        options: PipelineTmaUmmaOptions,
    ) -> PipelineTmaUmma {
        const KIND: &str = "PipelineTmaUmma";
        let _at = crate::builder::LocationScope::new();
        let layout = cluster_layout(KIND, options.cta_layout_vmnk.as_ref());
        let consumer_group = if options.enable_multicast_signaling {
            let threads = consumer_group.thread_arrive_count();
            if let Some(n) = threads.as_static() {
                assert!(
                    n % WARP_SIZE as i64 == 0,
                    "cute: {KIND}::create: the consumer group's thread count is not a whole number of warps ({n})"
                );
            }
            let extent = |mode: usize| layout.shape().modes()[mode].size() as i64;
            let (mm, mn) = options.mcast_mode_mn;
            let mcast_m = if mm != 0 { extent(2) } else { 0 };
            let mcast_n = if mn != 0 { extent(1) } else { 0 };
            let overlap = i64::from(mm != 0 && mn != 0);
            let mcast_size = mcast_m + mcast_n - overlap;
            assert!(mcast_size > 0, "cute: {KIND}::create: the cluster has no multicast partners (size 0)");
            let count = match threads.as_static() {
                Some(n) => BarrierParam::Static(mcast_size * (n / WARP_SIZE as i64)),
                None => BarrierParam::Dynamic(threads.value() / WARP_SIZE as i32 * mcast_size as i32),
            };
            CooperativeGroup::new(Agent::Thread, count)
        } else {
            consumer_group
        };
        let (full, empty) = barrier_arrays(
            KIND,
            num_stages,
            options.barrier_storage.as_ref(),
            (PipelineOp::TmaLoad, producer_group),
            (PipelineOp::TCGen05Mma, consumer_group),
            tx_count.into(),
            &options.name,
        );
        let (consumer_mask, is_leader) = if layout.size() == 1 {
            (None, Boolean::from(true))
        } else {
            (Some(mcast_arrival_mask(&layout, options.mcast_mode_mn)), is_leader_cta(&layout))
        };
        let cta_group = cta_group_of(&layout);
        init_sync(options.defer_sync, &layout);
        PipelineTmaUmma {
            sync_object_full: full,
            sync_object_empty: empty,
            num_stages,
            consumer_mask,
            is_leader_cta: is_leader,
            cta_group,
        }
    }

    /// The full barriers, which the TMA loads complete.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers, which the MMA warp's commits arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The multicast mask of the release commits (`None` for one CTA).
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.consumer_mask
    }

    /// Whether this CTA arms the full barriers (the leader of its pair).
    pub fn is_leader_cta(&self) -> Boolean {
        self.is_leader_cta
    }

    /// The CTA group of the commits.
    pub fn cta_group(&self) -> CtaGroup {
        self.cta_group
    }
}

impl Pipeline for PipelineTmaUmma {
    /// The bytes to expect for this acquire (CuTe DSL's `expected_tx`):
    /// `None` uses the `tx_count` the pipeline was made with.
    type AcquireOptions = Option<Int32>;

    fn num_stages(&self) -> usize {
        self.num_stages
    }

    fn producer_group(&self) -> Option<CooperativeGroup> {
        Some(self.sync_object_full.group())
    }

    fn consumer_group(&self) -> Option<CooperativeGroup> {
        Some(self.sync_object_empty.group())
    }

    /// Wait until the stage is empty (unless the token says so), then, on
    /// the leader CTA, arm its full barrier with the elected lane's
    /// `mbarrier.arrive.expect_tx` of `expected_tx` (or `tx_count`) bytes.
    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, expected_tx: Option<Int32>) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
        let tx = expected_tx.unwrap_or_else(|| self.sync_object_full.tx_count().value());
        let index = state.index();
        let arm = || self.sync_object_full.arrive_and_expect_tx(index, tx);
        match self.is_leader_cta.as_constant() {
            Some(true) => arm(),
            Some(false) => {}
            None => if_(self.is_leader_cta, arm),
        }
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_empty.try_wait(state.index(), state.phase())
    }

    /// Nothing: the TMA loads complete the full barrier's phase.
    fn producer_commit<S: StageState>(&self, _: &S) {}

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.sync_object_full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_full.try_wait(state.index(), state.phase())
    }

    /// The MMA warp's elected lane commits to the stage's empty barrier
    /// (`tcgen05.commit`, multicast with the consumer mask in a cluster).
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.sync_object_empty.arrive(state.index(), self.consumer_mask, Some(self.cta_group));
    }

    /// Wait on every stage's empty barrier, as [`PipelineAsync`](super::PipelineAsync) does.
    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        for _ in 0..self.num_stages {
            self.sync_object_empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for PipelineTmaUmma {
    /// The full barrier the stage's TMA loads report to.
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

/// The optional arguments of [`PipelineAsyncUmma::create`] and
/// [`PipelineUmmaAsync::create`].
#[derive(Clone, Debug, Default)]
pub struct PipelineUmmaOptions {
    /// Shared memory for the `2 * num_stages` mbarriers (full, then empty);
    /// `None` allocates them with [`alloc_reserved_mbarrier`].
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster as a static `(V, M, N, L)` layout; `None` is one CTA.
    pub cta_layout_vmnk: Option<Layout>,
    /// Leave out the initialisation sync.
    pub defer_sync: bool,
    /// A name for profiling (`{name}.full`, `{name}.empty`).
    pub name: String,
}

/// The pipeline between ordinary producer threads and the `tcgen05` MMA
/// warp (CuTe DSL's `cutlass.pipeline.PipelineAsyncUmma`), such as the
/// transform warps of a mixed-input GEMM or `cp.async` loaders.
///
/// * The full barriers ([`PipelineOp::AsyncThread`]) count the producer
///   group: each producer thread arrives in
///   [`producer_commit`](Pipeline::producer_commit) (on the leader CTA of
///   its pair in a 2-CTA cluster).
/// * The empty barriers ([`PipelineOp::TCGen05Mma`]): the MMA warp's
///   elected lane commits in [`consumer_release`](Pipeline::consumer_release).
///
/// The MMAs read shared memory through the async proxy: producers that
/// write a stage with ordinary stores or `cp.async` run
/// [`fence_view_async_shared`](crate::fence_view_async_shared)
/// (`fence.proxy.async.shared::cta`) after their writes have landed and
/// before committing; the simulator reports an MMA that reads a write not
/// fenced so.
#[derive(Clone, Debug)]
pub struct PipelineAsyncUmma {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    producer_mask: Option<Int32>,
    consumer_mask: Option<Int32>,
    cta_group: CtaGroup,
}

impl PipelineAsyncUmma {
    /// A pipeline of `num_stages` stages between ordinary-thread
    /// `producer_group` and the MMA `consumer_group` (CuTe DSL's
    /// `PipelineAsyncUmma.create`). Panics for 0 stages and a dynamic or
    /// non-rank-4 cluster layout.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: PipelineUmmaOptions,
    ) -> PipelineAsyncUmma {
        const KIND: &str = "PipelineAsyncUmma";
        let _at = crate::builder::LocationScope::new();
        let layout = cluster_layout(KIND, options.cta_layout_vmnk.as_ref());
        let (full, empty) = barrier_arrays(
            KIND,
            num_stages,
            options.barrier_storage.as_ref(),
            (PipelineOp::AsyncThread, producer_group),
            (PipelineOp::TCGen05Mma, consumer_group),
            BarrierParam::Static(0),
            &options.name,
        );
        let v = v_size(&layout);
        let (producer_mask, consumer_mask) = if v == 1 {
            (None, None)
        } else {
            // The leading CTA's rank, and the pair's mask over mode 0.
            let leading = cta_rank() / v as i32 * v as i32;
            let c = cta_coord(&layout);
            let mask = mcast_mask(&layout, &c, 0) | mcast_mask(&layout, &peer_coord(&layout, &c), 0);
            (Some(leading), Some(mask))
        };
        let cta_group = cta_group_of(&layout);
        init_sync(options.defer_sync, &layout);
        PipelineAsyncUmma {
            sync_object_full: full,
            sync_object_empty: empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        }
    }

    /// The full barriers, which the producer threads arrive on.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers, which the MMA warp's commits arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The cluster rank the producers arrive on (`None` when `V = 1`).
    pub fn producer_mask(&self) -> Option<Int32> {
        self.producer_mask
    }

    /// The multicast mask of the release commits (`None` when `V = 1`).
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.consumer_mask
    }

    /// The CTA group of the commits.
    pub fn cta_group(&self) -> CtaGroup {
        self.cta_group
    }
}

impl Pipeline for PipelineAsyncUmma {
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

    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_empty.try_wait(state.index(), state.phase())
    }

    /// Every producer thread arrives on the stage's full barrier. In a 2-CTA cluster the
    /// arrival goes to the leading CTA of the pair as a release at cluster scope
    /// (`mbarrier.arrive.release.cluster` through `mapa`), so that the peer's writes of the
    /// stage are ordered before the leader's MMA reads them (CuTe DSL
    /// arrives at CTA scope, as `PipelineTmaAsync`'s docs discuss).
    #[track_caller]
    fn producer_commit<S: StageState>(&self, state: &S) {
        match self.producer_mask {
            None => self.sync_object_full.arrive(state.index(), None, None),
            Some(rank) => crate::arch::mbarrier_arrive_with(
                &self.sync_object_full.get_barrier(state.index()),
                1,
                crate::arch::MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(rank),
                    scope: Some(crate::ir::MemScopeKind::Cluster),
                    ..Default::default()
                },
            ),
        }
    }

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.sync_object_full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_full.try_wait(state.index(), state.phase())
    }

    /// The MMA warp's elected lane commits to the stage's empty barrier.
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.sync_object_empty.arrive(state.index(), self.consumer_mask, Some(self.cta_group));
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        for _ in 0..self.num_stages {
            self.sync_object_empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for PipelineAsyncUmma {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

/// The accumulator pipeline of a Blackwell GEMM (CuTe DSL's
/// `cutlass.pipeline.PipelineUmmaAsync`): the MMA warp produces each stage
/// (an accumulator in TMEM), epilogue threads consume it.
///
/// * The full barriers ([`PipelineOp::TCGen05Mma`]):
///   [`producer_commit`](Pipeline::producer_commit) is the MMA warp's
///   elected lane's `tcgen05.commit`, so the phase completes when the MMAs
///   writing the stage have finished.
/// * The empty barriers ([`PipelineOp::AsyncThread`]) count the consumer
///   group: after [`consumer_wait`](Pipeline::consumer_wait) each epilogue
///   thread reads the stage (`tcgen05.ld`), waits for its loads
///   ([`fence_view_async_tmem_load`](crate::fence_view_async_tmem_load))
///   and then arrives in [`consumer_release`](Pipeline::consumer_release).
///   The simulator reports a read before the wait and a release with loads
///   still in flight.
/// * [`producer_tail`](Pipeline::producer_tail), on the even CTA of each
///   pair, advances `num_stages - 1` stages and acquires that one, so that
///   no commit is pending when the block exits.
#[derive(Clone, Debug)]
pub struct PipelineUmmaAsync {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    producer_mask: Option<Int32>,
    consumer_mask: Option<Int32>,
    cta_group: CtaGroup,
}

impl PipelineUmmaAsync {
    /// A pipeline of `num_stages` accumulator stages between the MMA
    /// `producer_group` and ordinary-thread `consumer_group` (CuTe DSL's
    /// `PipelineUmmaAsync.create`). Panics for 0 stages and a dynamic or
    /// non-rank-4 cluster layout.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: PipelineUmmaOptions,
    ) -> PipelineUmmaAsync {
        const KIND: &str = "PipelineUmmaAsync";
        let _at = crate::builder::LocationScope::new();
        let layout = cluster_layout(KIND, options.cta_layout_vmnk.as_ref());
        let (full, empty) = barrier_arrays(
            KIND,
            num_stages,
            options.barrier_storage.as_ref(),
            (PipelineOp::TCGen05Mma, producer_group),
            (PipelineOp::AsyncThread, consumer_group),
            BarrierParam::Static(0),
            &options.name,
        );
        // The commit's mask: this CTA's pair (mode 0 of the cluster layout).
        let producer_mask = (layout.size() != 1).then(|| {
            let c = cta_coord(&layout);
            make_layout_image_mask(&layout, coord(&c), 0).to::<i32>()
        });
        // The consumers' arrivals go to the even CTA of the pair.
        let consumer_mask = (v_size(&layout) != 1).then(|| cta_rank() / 2 * 2);
        let cta_group = cta_group_of(&layout);
        init_sync(options.defer_sync, &layout);
        PipelineUmmaAsync {
            sync_object_full: full,
            sync_object_empty: empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        }
    }

    /// The full barriers, which the MMA warp's commits arrive on.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers, which the epilogue threads arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The multicast mask of the commits (`None` for one CTA).
    pub fn producer_mask(&self) -> Option<Int32> {
        self.producer_mask
    }

    /// The cluster rank the consumers arrive on (`None` when `V = 1`).
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.consumer_mask
    }

    /// The CTA group of the commits.
    pub fn cta_group(&self) -> CtaGroup {
        self.cta_group
    }
}

impl Pipeline for PipelineUmmaAsync {
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

    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_empty.try_wait(state.index(), state.phase())
    }

    /// The MMA warp's elected lane commits to the stage's full barrier.
    #[track_caller]
    fn producer_commit<S: StageState>(&self, state: &S) {
        self.sync_object_full.arrive(state.index(), self.producer_mask, Some(self.cta_group));
    }

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.sync_object_full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_full.try_wait(state.index(), state.phase())
    }

    /// Every consumer thread arrives on the stage's empty barrier.
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.sync_object_empty.arrive(state.index(), self.consumer_mask, None);
    }

    /// On the even CTA of each pair (`block_idx_in_cluster % 2 == 0`):
    /// advance `num_stages - 1` stages, then acquire, so that the last
    /// stage's release has been seen. The state advances on every CTA (the
    /// branch is decided at run time; CuTe DSL advances it only inside).
    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        let is_leader = (cta_rank() % 2).eq(0);
        for _ in 1..self.num_stages {
            state.advance();
        }
        let s = *state;
        if_(is_leader, || self.producer_acquire(&s, None));
    }
}

impl PipelineBarriers for PipelineUmmaAsync {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

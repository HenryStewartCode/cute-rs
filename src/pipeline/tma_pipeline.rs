// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TMA pipelines of the Hopper mainloop and epilogue: [`PipelineTmaAsync`], whose producers fill stages with
//! TMA loads (multicast across a cluster included), and the mbarrier-free
//! [`PipelineTmaStore`] with its [`TmaStoreFence`], which paces TMA stores
//! out of a ring of shared-memory staging buffers by bulk async-groups.

use crate::arch::{
    MbarrierOptions, WARP_SIZE, block_idx_in_cluster, cp_async_bulk_commit_group, cp_async_bulk_wait_group,
    make_warp_uniform, mbarrier_arrive_with, mbarrier_init_fence, thread_idx,
};
use crate::control::if_;
use crate::ir::MemScopeKind;
use crate::layout::{Coord, IndexExpr, Layout, Tree};
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32, IntoValue};

use super::async_pipeline::wait_unless;
use super::mbarrier::portable_waits;
use super::{
    Agent, BarrierParam, CooperativeGroup, MbarrierArray, MbarrierArrayOptions, Pipeline, PipelineBarriers, PipelineOp,
    PipelineState, StageState, SyncObject, agent_sync, alloc_reserved_mbarrier,
};

/// The optional arguments of [`PipelineTmaAsync::create`].
#[derive(Clone, Debug)]
pub struct PipelineTmaAsyncOptions {
    /// Shared memory for the `2 * num_stages` mbarriers: full barriers
    /// first, then empty ones. `None` allocates them with
    /// [`alloc_reserved_mbarrier`].
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster's CTAs as a static `(V, M, N, L)` layout from a CTA's
    /// coordinate to its rank in the cluster. `None` is a single CTA,
    /// `(1, 1, 1, 1)`. A layout with runtime extents or strides is refused.
    pub cta_layout_vmnk: Option<Layout>,
    /// The thread index the consumers' signalling lanes are picked from
    /// (only its lane, `tidx % 32`, matters). `None` reads
    /// `thread_idx().x`.
    pub tidx: Option<Int32>,
    /// Whether the stages are multicast along M (`.0`) and along N (`.1`),
    /// each 0 or 1: the empty barriers of the CTAs that share this CTA's
    /// tiles are signalled on release. `(0, 0)` is refused. Default
    /// `(1, 1)`.
    pub mcast_mode_mn: (u32, u32),
    /// Take the consumer group as the threads of one CTA that wait and
    /// release, and count the empty barriers' arrivals from it: one per
    /// consumer warp per multicast partner (see [`PipelineTmaAsync`]).
    /// `false` uses the consumer group's count as given.
    pub enable_multicast_signaling: bool,
    /// Leave out the initialisation sync, for a kernel that publishes
    /// several pipelines' barriers with one
    /// [`pipeline_init_arrive`](super::pipeline_init_arrive) and
    /// [`pipeline_init_wait`](super::pipeline_init_wait).
    pub defer_sync: bool,
    /// A name for profiling; the barrier arrays are named `{name}.full`
    /// and `{name}.empty`.
    pub name: String,
}

impl Default for PipelineTmaAsyncOptions {
    fn default() -> Self {
        PipelineTmaAsyncOptions {
            barrier_storage: None,
            cta_layout_vmnk: None,
            tidx: None,
            mcast_mode_mn: (1, 1),
            enable_multicast_signaling: false,
            defer_sync: false,
            name: String::new(),
        }
    }
}

/// The pipeline of a Hopper GEMM mainloop (CuTe DSL's
/// `cutlass.pipeline.PipelineTmaAsync`): producers fill each stage with TMA
/// loads, consumers (ordinary threads, typically warpgroups running WGMMA)
/// drain it.
///
/// * The full barriers ([`PipelineOp::TmaLoad`]) count the producer group's
///   threads and expect `tx_count` bytes per phase.
///   [`producer_acquire`](Pipeline::producer_acquire) waits on the stage's
///   empty barrier, then the producer warp's elected lane arrives on the full
///   barrier and sets the bytes to expect (`mbarrier.arrive.expect_tx`). The
///   TMA copies are then issued with
///   [`producer_get_barrier`](PipelineBarriers::producer_get_barrier) as
///   their `tma_bar_ptr`, and their completion (`complete_tx`) finishes the
///   phase, so [`producer_commit`](Pipeline::producer_commit) does nothing.
/// * The empty barriers ([`PipelineOp::AsyncThread`]) are arrived on by the
///   consumers' *signalling lanes* only: in each consumer warp, lane `i`
///   (for `i` below the cluster size) signals the CTA of rank `i` if that
///   CTA shares this CTA's tiles (see
///   [`init_empty_barrier_arrive_signal`]), through `mapa` for a peer.
///
/// # The consumer arrival count
///
/// Even in a single CTA, only lane 0 of each consumer warp arrives on an
/// empty barrier. A consumer group given in threads (`Agent::Thread, 128`)
/// therefore never completes a phase: the producer waits forever. Without
/// [`enable_multicast_signaling`](PipelineTmaAsyncOptions::enable_multicast_signaling)
/// count the consumers in warps times the number of CTAs that multicast to
/// this one, as CuTe's Hopper `dense_gemm.py` does
/// (`CooperativeGroup::new(Agent::Thread, mcast_size * num_warps)`; not
/// `Agent::Warp`, which would count 32 arrivals per warp). With it,
/// give the consumer group as the threads that call `consumer_release` in
/// one CTA (a multiple of 32): the count becomes
/// `(mcast_m + mcast_n - overlap) * warps`, with `mcast_m` the cluster's N
/// extent when multicasting along M (else 0), `mcast_n` its M extent when
/// multicasting along N (else 0), and `overlap` 1 when both are on.
///
/// # Cluster scope
///
/// A consumer's release of a stage in a peer CTA orders its reads of its own
/// shared memory before the peer's next TMA write into it, a hand-off across
/// CTAs. CuTe DSL arrives on the peer's empty barrier at CTA scope (the PTX
/// default), which its sources note is formally not enough across CTAs;
/// cute-rs arrives at cluster scope (`.release.cluster`), so that the
/// simulator, which holds kernels to the PTX memory model, sees the
/// hand-off. A single CTA arrives as CuTe DSL does.
///
/// # Initialisation
///
/// Unless [`defer_sync`](PipelineTmaAsyncOptions::defer_sync), `create`
/// publishes the barriers: `fence.mbarrier_init` and a CTA barrier for one
/// CTA, a relaxed cluster arrive and a cluster wait for a cluster (every
/// thread of every CTA must reach it).
#[derive(Clone, Debug)]
pub struct PipelineTmaAsync {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    dst_rank: Option<Int32>,
    is_signaling_thread: Boolean,
    cta_layout_vmnk: Layout,
}

/// The `(1, 1, 1, 1)` cluster of a single CTA.
fn single_cta() -> Layout {
    Layout::col_major((1usize, 1usize, 1usize, 1usize))
}

/// The static size of a cluster layout.
fn cluster_ctas(layout: &Layout) -> usize {
    layout.size()
}

/// The four entries of a `(V, M, N, L)` coordinate as `i32` values.
#[track_caller]
pub(super) fn coord_entries(coord: &Coord) -> Vec<IndexExpr> {
    match coord {
        Tree::Branch(modes) => modes
            .iter()
            .map(|m| match m {
                Tree::Leaf(e) => e.clone(),
                // A nested mode: its offset within the mode identifies it.
                other => panic!("cute: PipelineTmaAsync: the cluster layout's modes must be flat, not {other}"),
            })
            .collect(),
        Tree::Leaf(e) => vec![e.clone()],
    }
}

/// Whether two coordinate entries are equal: decided when the kernel is
/// built if both are static.
#[track_caller]
pub(super) fn same(a: &IndexExpr, b: &IndexExpr) -> Boolean {
    match (a.as_static(), b.as_static()) {
        (Some(x), Some(y)) => Boolean::from(x == y),
        _ => IntoValue::<i32>::into_value(a.clone()).eq(IntoValue::<i32>::into_value(b.clone())),
    }
}

/// `a && b`, folded when either is a constant.
#[track_caller]
pub(super) fn and(a: Boolean, b: Boolean) -> Boolean {
    match (a.as_constant(), b.as_constant()) {
        (Some(false), _) | (_, Some(false)) => Boolean::from(false),
        (Some(true), _) => b,
        (_, Some(true)) => a,
        _ => a & b,
    }
}

/// `a || b`, folded when either is a constant.
#[track_caller]
pub(super) fn or(a: Boolean, b: Boolean) -> Boolean {
    match (a.as_constant(), b.as_constant()) {
        (Some(true), _) | (_, Some(true)) => Boolean::from(true),
        (Some(false), _) => b,
        (_, Some(false)) => a,
        _ => a | b,
    }
}

/// Check a multicast mode pair: each entry 0 or 1, not both 0.
#[track_caller]
pub(super) fn check_mcast_mode(mcast_mode_mn: (u32, u32)) {
    assert!(
        mcast_mode_mn.0 <= 1 && mcast_mode_mn.1 <= 1,
        "cute: PipelineTmaAsync: mcast_mode_mn entries are 0 or 1, not {mcast_mode_mn:?}"
    );
    assert!(
        mcast_mode_mn != (0, 0),
        "cute: PipelineTmaAsync: mcast_mode_mn (0, 0) multicasts along neither M nor N; use (1, 0), (0, 1) or (1, 1)"
    );
}

/// Check that a cluster layout is a static rank-4 `(V, M, N, L)` layout.
#[track_caller]
pub(super) fn check_cluster_layout(layout: &Layout) {
    assert!(layout.is_static(), "cute: PipelineTmaAsync: cta_layout_vmnk must have a static shape");
    assert!(
        layout.rank() == 4,
        "cute: PipelineTmaAsync: the cluster layout (cta_layout_vmnk) {layout} must be (V, M, N, L), rank 4"
    );
}

/// Which consumer threads arrive on which CTA's empty barrier (CuTe DSL's
/// `PipelineTmaAsync.init_empty_barrier_arrive_signal`): returns the cluster
/// rank this thread signals and whether it signals at all.
///
/// The thread's lane (`tidx % 32`) names the destination: lane `i` signals
/// CTA `i % size` and only lanes below the cluster size signal. A lane then
/// keeps its destination only if that CTA shares this CTA's tiles: with
/// multicast along M (`mcast_mode_mn.0`) the CTAs whose `(V, M, L)`
/// coordinates match this one's, along N those whose `(V, N, L)` match, and
/// with both, either. This CTA's coordinate comes from its rank
/// ([`block_idx_in_cluster`], made warp-uniform) through
/// [`Layout::get_hier_coord`] of `cta_layout_vmnk`.
///
/// For one CTA this is "lane 0 signals CTA 0". Panics for `mcast_mode_mn`
/// `(0, 0)` and for a dynamic or non-rank-4 layout.
#[track_caller]
pub fn init_empty_barrier_arrive_signal(
    cta_layout_vmnk: &Layout,
    tidx: impl IntoValue<i32>,
    mcast_mode_mn: (u32, u32),
) -> (Int32, Boolean) {
    check_cluster_layout(cta_layout_vmnk);
    check_mcast_mode(mcast_mode_mn);
    let size = cluster_ctas(cta_layout_vmnk);
    let lane = tidx.into_value() % WARP_SIZE as i32;
    let is_signaling = if size >= WARP_SIZE { Boolean::from(true) } else { lane.lt(size as i32) };
    let dst_rank = if size == 1 { Int32::from(0) } else { lane % size as i32 };
    let (dst, cur): (Vec<IndexExpr>, Vec<IndexExpr>) = if size == 1 {
        (vec![IndexExpr::Static(0); 4], vec![IndexExpr::Static(0); 4])
    } else {
        let rank = make_warp_uniform(block_idx_in_cluster());
        (coord_entries(&cta_layout_vmnk.get_hier_coord(dst_rank)), coord_entries(&cta_layout_vmnk.get_hier_coord(rank)))
    };
    let matches =
        |modes: [usize; 3]| modes.into_iter().fold(Boolean::from(true), |acc, i| and(acc, same(&dst[i], &cur[i])));
    let shares = match mcast_mode_mn {
        (1, 0) => matches([0, 1, 3]),
        (0, 1) => matches([0, 2, 3]),
        _ => or(matches([0, 1, 3]), matches([0, 2, 3])),
    };
    (dst_rank, and(is_signaling, shares))
}

impl PipelineTmaAsync {
    /// A pipeline of `num_stages` stages between a TMA `producer_group`
    /// and an ordinary-thread `consumer_group`, each full phase expecting
    /// `tx_count` bytes (CuTe DSL's `PipelineTmaAsync.create`): full
    /// barriers at the storage, empty ones `num_stages` further on,
    /// initialised from warp 0, then published unless
    /// [`defer_sync`](PipelineTmaAsyncOptions::defer_sync) (see the type's
    /// docs for the consumer count).
    ///
    /// Panics for 0 stages, a dynamic cluster layout ("cta_layout_vmnk must have a static shape"), `mcast_mode_mn` `(0, 0)`, and
    /// with `enable_multicast_signaling`, a consumer group whose static
    /// thread count is not a multiple of 32.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: impl Into<BarrierParam>,
        options: PipelineTmaAsyncOptions,
    ) -> PipelineTmaAsync {
        assert!(num_stages > 0, "cute: PipelineTmaAsync::create: a pipeline needs at least one stage, not 0");
        let _at = crate::builder::LocationScope::new();
        let cta_layout_vmnk = options.cta_layout_vmnk.clone().unwrap_or_else(single_cta);
        check_cluster_layout(&cta_layout_vmnk);
        check_mcast_mode(options.mcast_mode_mn);
        let size = cluster_ctas(&cta_layout_vmnk);
        let consumer_group = if options.enable_multicast_signaling {
            multicast_consumer_group(consumer_group, &cta_layout_vmnk, options.mcast_mode_mn)
        } else {
            consumer_group
        };
        crate::profiling::register_barrier(&options.name, num_stages, "PipelineTmaAsync");
        let storage = options.barrier_storage.clone().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
        let named = |side: &str| MbarrierArrayOptions {
            name: if options.name.is_empty() { String::new() } else { format!("{}.{side}", options.name) },
            ..Default::default()
        };
        let sync_object_full = MbarrierArray::with_options(
            &storage,
            num_stages,
            (PipelineOp::TmaLoad, producer_group),
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
        let tidx = options.tidx.unwrap_or_else(|| thread_idx().x);
        let (dst_rank, is_signaling_thread) =
            init_empty_barrier_arrive_signal(&cta_layout_vmnk, tidx, options.mcast_mode_mn);
        let dst_rank = (size > 1).then_some(dst_rank);
        if !options.defer_sync {
            if !portable_waits() {
                mbarrier_init_fence();
            }
            if size == 1 {
                agent_sync(Agent::ThreadBlock, false);
            } else {
                agent_sync(Agent::ThreadBlockCluster, true);
            }
        }
        PipelineTmaAsync {
            sync_object_full,
            sync_object_empty,
            num_stages,
            dst_rank,
            is_signaling_thread,
            cta_layout_vmnk,
        }
    }

    /// The full barriers, which the TMA loads complete.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers, which the consumers' signalling lanes arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The cluster rank whose empty barrier this thread arrives on (`None`
    /// for a single CTA, which arrives on its own).
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.dst_rank
    }

    /// Whether this thread arrives on an empty barrier in
    /// [`consumer_release`](Pipeline::consumer_release).
    pub fn is_signaling_thread(&self) -> Boolean {
        self.is_signaling_thread
    }

    /// The cluster layout, `(1, 1, 1, 1)` for one CTA.
    pub fn cta_layout_vmnk(&self) -> &Layout {
        &self.cta_layout_vmnk
    }
}

/// The consumer group of `enable_multicast_signaling`: one arrival per
/// consumer warp per multicast partner.
#[track_caller]
fn multicast_consumer_group(group: CooperativeGroup, layout: &Layout, mcast_mode_mn: (u32, u32)) -> CooperativeGroup {
    let threads = group.thread_arrive_count();
    if let Some(n) = threads.as_static() {
        assert!(
            n % WARP_SIZE as i64 == 0,
            "cute: PipelineTmaAsync::create: the consumer group's thread count is not a whole number of warps ({n} threads)"
        );
    }
    let extent = |mode: usize| layout.shape().modes()[mode].size() as i64;
    let mcast_m = if mcast_mode_mn.0 == 1 { extent(2) } else { 0 };
    let mcast_n = if mcast_mode_mn.1 == 1 { extent(1) } else { 0 };
    let overlap = i64::from(mcast_mode_mn == (1, 1));
    let mcast_size = mcast_m + mcast_n - overlap;
    assert!(mcast_size > 0, "cute: PipelineTmaAsync::create: the cluster has no multicast partners (size 0)");
    let count = match threads.as_static() {
        Some(n) => BarrierParam::Static(mcast_size * (n / WARP_SIZE as i64)),
        None => BarrierParam::Dynamic(threads.value() / WARP_SIZE as i32 * mcast_size as i32),
    };
    CooperativeGroup::new(Agent::Thread, count)
}

impl Pipeline for PipelineTmaAsync {
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

    /// Wait until the stage is empty (unless the token says so), then arm
    /// its full barrier: the warp's elected lane arrives and sets the
    /// `tx_count` bytes the TMA loads will complete.
    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
        self.sync_object_full.arrive(state.index(), None, None);
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_empty.try_wait(state.index(), state.phase())
    }

    /// Nothing: the TMA loads' `complete_tx` completes the full barrier's
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

    /// The signalling lanes arrive on the empty barrier of their
    /// destination CTA (this CTA's for one CTA). In a cluster the arrival
    /// is a release at cluster scope (`mbarrier.arrive.release.cluster`
    /// through `mapa`); see the type's docs.
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        let release = || match self.dst_rank {
            None => self.sync_object_empty.arrive(state.index(), None, None),
            Some(rank) => mbarrier_arrive_with(
                &self.sync_object_empty.get_barrier(state.index()),
                1,
                MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(rank),
                    scope: Some(MemScopeKind::Cluster),
                    ..Default::default()
                },
            ),
        };
        match self.is_signaling_thread.as_constant() {
            Some(true) => release(),
            Some(false) => {}
            None => if_(self.is_signaling_thread, release),
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

impl PipelineBarriers for PipelineTmaAsync {
    /// The full barrier the stage's TMA loads report to (their
    /// `tma_bar_ptr`).
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

/// The synchronisation of a multi-stage TMA-store epilogue (CuTe DSL's
/// `cutlass.pipeline.TmaStoreFence`): no mbarriers, only bulk async-groups.
/// [`arrive`](Self::arrive) closes the group of TMA stores issued since the
/// last one (`cp.async.bulk.commit_group`), [`wait`](Self::wait) waits until
/// at most `num_stages - 1` groups still read their shared-memory source
/// (`cp.async.bulk.wait_group.read`), so the oldest staging buffer may be
/// rewritten, and [`tail`](Self::tail) waits until none does.
///
/// Like the stores, these are per thread: the thread that issues the stores
/// commits and waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TmaStoreFence {
    num_stages: usize,
}

impl TmaStoreFence {
    /// A fence over `num_stages` staging buffers. Panics for 0.
    #[track_caller]
    pub fn new(num_stages: usize) -> TmaStoreFence {
        assert!(num_stages > 0, "cute: TmaStoreFence: it needs at least one stage");
        TmaStoreFence { num_stages }
    }

    /// The number of staging buffers.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Close the current group of TMA stores (`cp.async.bulk.commit_group`).
    #[track_caller]
    pub fn arrive(&self) {
        cp_async_bulk_commit_group();
    }

    /// Wait until at most `num_stages - 1` groups still read shared memory
    /// (`cp.async.bulk.wait_group.read`).
    #[track_caller]
    pub fn wait(&self) {
        cp_async_bulk_wait_group(self.num_stages as i32 - 1, true);
    }

    /// [`arrive`](Self::arrive), then [`wait`](Self::wait).
    #[track_caller]
    pub fn arrive_and_wait(&self) {
        self.arrive();
        self.wait();
    }

    /// Wait until no group reads shared memory any more
    /// (`cp.async.bulk.wait_group.read 0`), before the block exits or the
    /// staging buffers are reused for something else.
    #[track_caller]
    pub fn tail(&self) {
        cp_async_bulk_wait_group(0, true);
    }

    /// Not supported (CuTe DSL raises): always panics.
    #[track_caller]
    pub fn arrive_and_drop(&self) {
        panic!("cute: TmaStoreFence::arrive_and_drop is not supported");
    }

    /// There is no barrier: always panics.
    #[track_caller]
    pub fn get_barrier(&self) {
        panic!("cute: TmaStoreFence has no mbarrier to return");
    }

    /// Not supported (there is no arrival count): always panics.
    #[track_caller]
    pub fn max(&self) -> u32 {
        panic!("cute: TmaStoreFence::max is not supported (it has no arrival count)");
    }
}

impl SyncObject for TmaStoreFence {
    type ArriveArgs = ();
    type WaitArgs = ();
    type BarrierArgs = ();
    /// There is none: [`get_barrier`](SyncObject::get_barrier) panics.
    type Barrier = ();

    #[track_caller]
    fn arrive(&self, (): ()) {
        TmaStoreFence::arrive(self);
    }

    #[track_caller]
    fn wait(&self, (): ()) {
        TmaStoreFence::wait(self);
    }

    #[track_caller]
    fn arrive_and_wait(&self, (): (), (): ()) {
        TmaStoreFence::arrive_and_wait(self);
    }

    #[track_caller]
    fn arrive_and_drop(&self) {
        TmaStoreFence::arrive_and_drop(self);
    }

    #[track_caller]
    fn get_barrier(&self, (): ()) {
        TmaStoreFence::get_barrier(self);
    }

    /// `None`: the fence has no arrival count.
    fn max(&self) -> Option<u32> {
        None
    }
}

/// The pipeline of a TMA-store epilogue (CuTe DSL's
/// `cutlass.pipeline.PipelineTmaStore`): a ring of `num_stages`
/// shared-memory staging buffers paced by a [`TmaStoreFence`], with no
/// mbarriers and no consumer side (the TMA unit consumes).
///
/// Its methods take no [`PipelineState`] (the bulk groups are counted, not
/// indexed), so it has inherent methods instead of implementing
/// [`Pipeline`]; the caller tracks which buffer to write, usually
/// `tile % num_stages`.
///
/// The epilogue pattern (CuTe's Hopper `dense_gemm.py`), for each tile:
///
/// 1. every thread writes its part of the tile into the staging buffer;
/// 2. [`fence_view_async_shared`](crate::fence_view_async_shared)
///    (`fence.proxy.async.shared::cta`) makes the writes visible to the TMA
///    unit, and a named barrier ([`sync`](super::sync)`(1)`) waits for all
///    of them;
/// 3. one warp issues the TMA store, then
///    [`producer_commit`](Self::producer_commit) and
///    [`producer_acquire`](Self::producer_acquire), which waits until the
///    buffer to be written next is no longer read;
/// 4. the named barrier again, so no thread writes that buffer early.
///
/// The issuing warp calls [`producer_tail`](Self::producer_tail) before the
/// block exits.
#[derive(Clone, Copy, Debug)]
pub struct PipelineTmaStore {
    sync_object_full: TmaStoreFence,
    producer_group: CooperativeGroup,
}

impl PipelineTmaStore {
    /// A store pipeline of `num_stages` staging buffers issued by
    /// `producer_group` (CuTe DSL's `PipelineTmaStore.create`). Emits
    /// nothing. Panics for 0 stages.
    #[track_caller]
    pub fn create(num_stages: usize, producer_group: CooperativeGroup) -> PipelineTmaStore {
        PipelineTmaStore { sync_object_full: TmaStoreFence::new(num_stages), producer_group }
    }

    /// The number of staging buffers.
    pub fn num_stages(&self) -> usize {
        self.sync_object_full.num_stages()
    }

    /// The threads that issue the stores.
    pub fn producer_group(&self) -> CooperativeGroup {
        self.producer_group
    }

    /// The fence.
    pub fn sync_object_full(&self) -> &TmaStoreFence {
        &self.sync_object_full
    }

    /// Wait until the next staging buffer is no longer read by an earlier
    /// store (`cp.async.bulk.wait_group.read num_stages - 1`).
    #[track_caller]
    pub fn producer_acquire(&self) {
        self.sync_object_full.wait();
    }

    /// Close the group of stores issued since the last commit
    /// (`cp.async.bulk.commit_group`).
    #[track_caller]
    pub fn producer_commit(&self) {
        self.sync_object_full.arrive();
    }

    /// Wait until no store reads shared memory
    /// (`cp.async.bulk.wait_group.read 0`), before the block exits.
    #[track_caller]
    pub fn producer_tail(&self) {
        self.sync_object_full.tail();
    }

    /// There is no consumer: always panics.
    #[track_caller]
    pub fn consumer_wait(&self) {
        panic!("cute: PipelineTmaStore has no consumer side");
    }

    /// There is no consumer: always panics.
    #[track_caller]
    pub fn consumer_release(&self) {
        panic!("cute: PipelineTmaStore has no consumer side");
    }
}

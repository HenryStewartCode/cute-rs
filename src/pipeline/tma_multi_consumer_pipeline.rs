// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TMA pipeline with two kinds of consumer (roadmap part
//! m10-cta-pair-pipelines): [`PipelineTmaMultiConsumersAsync`], whose stages
//! are filled by TMA loads and drained both by the `tcgen05` MMA warp and by
//! ordinary threads reading the same shared memory, as in CuTe DSL's Mamba2
//! SSD kernel.

use crate::arch::{
    MbarrierOptions, WARP_SIZE, block_idx_in_cluster, make_warp_uniform, mbarrier_arrive_with, thread_idx,
};
use crate::control::if_;
use crate::ir::MemScopeKind;
use crate::layout::{IndexExpr, Layout};
use crate::log::Level;
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32, IntoValue};

use super::async_pipeline::wait_unless;
use super::tma_pipeline::{and, check_cluster_layout, check_mcast_mode, coord_entries, or, same};
use super::umma_pipeline::{
    barrier_arrays, cluster_layout, cta_group_of, init_sync, is_leader_cta, mcast_arrival_mask,
};
use super::{
    Agent, BarrierParam, ConsumerHandle, CooperativeGroup, CtaGroup, MbarrierArray, Pipeline, PipelineBarriers,
    PipelineOp, PipelineState, StageState,
};

/// The optional arguments of [`PipelineTmaMultiConsumersAsync::create`].
#[derive(Clone, Debug)]
pub struct PipelineTmaMultiConsumersAsyncOptions {
    /// Shared memory for the `2 * num_stages` mbarriers (full, then empty);
    /// `None` allocates them with [`alloc_reserved_mbarrier`](super::alloc_reserved_mbarrier).
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster as a static `(V, M, N, L)` layout; `None` is one CTA.
    pub cta_layout_vmnk: Option<Layout>,
    /// Whether the stages are multicast along M and along N (each 0 or 1,
    /// not both 0). Default `(1, 1)`.
    pub mcast_mode_mn: (u32, u32),
    /// The thread index the async consumers' signalling lanes are picked
    /// from (only `tidx % 32` matters); `None` reads `thread_idx().x`.
    pub tidx: Option<Int32>,
    /// Take each consumer group as the threads of one CTA that release (a
    /// whole number of warps) and count one arrival per warp per multicast
    /// partner, as [`PipelineTmaAsyncOptions::enable_multicast_signaling`](super::PipelineTmaAsyncOptions::enable_multicast_signaling).
    pub enable_multicast_signaling: bool,
    /// Leave out the initialisation sync.
    pub defer_sync: bool,
    /// The legacy signalling, in which every async consumer thread arrives
    /// on its own CTA's empty barrier (CuTe DSL's deprecated
    /// `force_deprecated_per_lane_signaling`). `None` picks it as CuTe DSL
    /// does: when multicast signalling is off and a consumer group is a
    /// non-zero multiple of 32 threads, which suggests a count of every lane
    /// (a warning is logged). `Some(false)` keeps the one-lane-per-warp
    /// signalling. Refused for a cluster and with
    /// `enable_multicast_signaling`.
    pub per_lane_signaling: Option<bool>,
    /// A name for profiling (`{name}.full`, `{name}.empty`).
    pub name: String,
}

impl Default for PipelineTmaMultiConsumersAsyncOptions {
    fn default() -> Self {
        PipelineTmaMultiConsumersAsyncOptions {
            barrier_storage: None,
            cta_layout_vmnk: None,
            mcast_mode_mn: (1, 1),
            tidx: None,
            enable_multicast_signaling: false,
            defer_sync: false,
            per_lane_signaling: None,
            name: String::new(),
        }
    }
}

/// A pipeline whose stages are filled by TMA loads and read by two consumer
/// groups at once (CuTe DSL's `cutlass.pipeline.PipelineTmaMultiConsumersAsync`):
/// the `tcgen05` MMA warp and ordinary "async" threads.
///
/// * The full barriers ([`PipelineOp::TmaLoad`]) are armed as in
///   [`PipelineTmaUmma`](super::PipelineTmaUmma): on the leader CTA of a
///   pair, the producer's elected lane arrives with `tx_count` bytes, and the
///   TMA loads complete the phase.
/// * One array of empty barriers counts both groups: the UMMA group's
///   arrivals plus the async group's. It is seen as two arrays over the same
///   mbarriers ([`MbarrierArray::recast_to_new_op_type`]): the MMA warp
///   releases with a `tcgen05.commit` ([`PipelineOp::TCGen05Mma`], multicast
///   with the same mask as `PipelineTmaUmma` in a cluster), the async
///   threads with an ordinary arrival ([`PipelineOp::AsyncThread`]).
///   [`consumer_release_op`](Self::consumer_release_op) takes which;
///   [`Pipeline::consumer_release`] is the MMA's.
///
/// # The async consumers' arrivals
///
/// Only *signalling lanes* of the async threads arrive: lane `i` of each
/// warp, for `i` below the cluster size, on the empty barrier of cluster
/// rank `i` if that CTA shares this CTA's tiles along a multicast mode
/// ([`init_empty_barrier_arrive_signal_2sm`]). In one CTA that is lane 0 of
/// each warp, so `consumer_group_async` counts warps (as
/// `CooperativeGroup::new(Agent::Thread, warps)`), not threads. The other
/// lanes' reads of the stage must be ordered before the signalling lane's
/// arrival, by a [`sync_warp`](crate::sync_warp) before the release.
///
/// The legacy form, in which every async thread arrives on its own CTA's
/// barrier, is kept as [`per_lane_signaling`](PipelineTmaMultiConsumersAsyncOptions::per_lane_signaling)
/// (one CTA only); it is chosen by default for a group of a multiple of 32
/// threads, as in CuTe DSL. The simulator reports a group counted one way
/// and arriving the other: more arrivals than the barrier was initialised
/// with, or a deadlock.
///
/// CuTe DSL's `make_participants` returns `(producer, umma_consumer,
/// None)`; here [`Pipeline::make_participants`] returns the first two, and
/// async consumers release through
/// [`ConsumerHandle::release_op`](ConsumerHandle::release_op).
#[derive(Clone, Debug)]
pub struct PipelineTmaMultiConsumersAsync {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    sync_object_empty_umma: MbarrierArray,
    sync_object_empty_async: MbarrierArray,
    num_stages: usize,
    consumer_mask: Option<Int32>,
    is_leader_cta: Boolean,
    cta_group: CtaGroup,
    consumer_dst_rank_async: Option<Int32>,
    is_signaling_thread: Boolean,
    per_lane_signaling: bool,
}

/// Which async consumer threads arrive on which CTA's empty barrier in a
/// cluster with CTA pairs (CuTe DSL's
/// `PipelineTmaMultiConsumersAsync._init_empty_barrier_arrive_signal_2sm`):
/// the cluster rank this thread signals and whether it signals at all.
///
/// As [`init_empty_barrier_arrive_signal`](super::init_empty_barrier_arrive_signal),
/// lane `i` (of `tidx % 32`) signals CTA `i % size` and only lanes below the
/// cluster size signal, but the CTAs that share this CTA's tiles are matched
/// without the `V` mode, so both CTAs of a pair are signalled: along M those
/// whose `(M, L)` coordinates equal this CTA's, along N those whose `(N, L)`
/// do, with both modes either. Panics for `mcast_mode_mn` `(0, 0)` and a
/// dynamic or non-rank-4 layout.
#[track_caller]
pub fn init_empty_barrier_arrive_signal_2sm(
    cta_layout_vmnk: &Layout,
    tidx: impl IntoValue<i32>,
    mcast_mode_mn: (u32, u32),
) -> (Int32, Boolean) {
    check_cluster_layout(cta_layout_vmnk);
    check_mcast_mode(mcast_mode_mn);
    let size = cta_layout_vmnk.size();
    let lane = tidx.into_value() % WARP_SIZE as i32;
    let is_signaling = if size >= WARP_SIZE { Boolean::from(true) } else { lane.lt(size as i32) };
    let dst_rank = if size == 1 { Int32::from(0) } else { lane % size as i32 };
    let (dst, cur): (Vec<IndexExpr>, Vec<IndexExpr>) = if size == 1 {
        (vec![IndexExpr::Static(0); 4], vec![IndexExpr::Static(0); 4])
    } else {
        let rank = make_warp_uniform(block_idx_in_cluster());
        // Flat coordinates: a cluster layout from `tiled_divide` nests its `V` mode.
        (coord_entries(&cta_layout_vmnk.get_flat_coord(dst_rank)), coord_entries(&cta_layout_vmnk.get_flat_coord(rank)))
    };
    let matches =
        |modes: [usize; 2]| modes.into_iter().fold(Boolean::from(true), |acc, i| and(acc, same(&dst[i], &cur[i])));
    let shares = match mcast_mode_mn {
        (1, 0) => matches([1, 3]),
        (0, 1) => matches([2, 3]),
        _ => or(matches([1, 3]), matches([2, 3])),
    };
    (dst_rank, and(is_signaling, shares))
}

/// `a + b`, static when both are.
#[track_caller]
fn add(a: BarrierParam, b: BarrierParam) -> BarrierParam {
    match (a.as_static(), b.as_static()) {
        (Some(x), Some(y)) => BarrierParam::Static(x + y),
        _ => BarrierParam::Dynamic(a.value() + b.value()),
    }
}

/// The group of one arrival per warp per multicast partner.
#[track_caller]
fn signaling_group(kind: &str, group: CooperativeGroup, mcast_size: i64) -> CooperativeGroup {
    let threads = group.thread_arrive_count();
    if let Some(n) = threads.as_static() {
        assert!(
            n % WARP_SIZE as i64 == 0,
            "cute: PipelineTmaMultiConsumersAsync::create: the {kind} consumer group's {n} threads are not a whole \
             number of warps"
        );
    }
    let count = match threads.as_static() {
        Some(n) => BarrierParam::Static(mcast_size * (n / WARP_SIZE as i64)),
        None => BarrierParam::Dynamic(threads.value() / WARP_SIZE as i32 * mcast_size as i32),
    };
    CooperativeGroup::new(Agent::Thread, count)
}

/// Whether a group looks counted in lanes: threads, a non-zero multiple of 32.
fn looks_per_lane(group: &CooperativeGroup) -> bool {
    group.agent() == Agent::Thread
        && group.size().as_static().is_some_and(|n| n >= WARP_SIZE as i64 && n % WARP_SIZE as i64 == 0)
}

impl PipelineTmaMultiConsumersAsync {
    /// A pipeline of `num_stages` stages between a TMA `producer_group` and
    /// two consumer groups, the MMA's and the async threads', each full phase
    /// expecting `tx_count` bytes (CuTe DSL's
    /// `PipelineTmaMultiConsumersAsync.create`).
    ///
    /// Panics for 0 stages, a dynamic or non-rank-4 cluster layout, consumer
    /// groups of different agents, `mcast_mode_mn` `(0, 0)` in a cluster,
    /// with `enable_multicast_signaling` a group that is not a whole number
    /// of warps, and per-lane signalling with multicast signalling or in a
    /// cluster.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group_umma: CooperativeGroup,
        consumer_group_async: CooperativeGroup,
        tx_count: impl Into<BarrierParam>,
        options: PipelineTmaMultiConsumersAsyncOptions,
    ) -> PipelineTmaMultiConsumersAsync {
        const KIND: &str = "PipelineTmaMultiConsumersAsync";
        let _at = crate::builder::LocationScope::new();
        let layout = cluster_layout(KIND, options.cta_layout_vmnk.as_ref());
        assert!(
            consumer_group_umma.agent() == consumer_group_async.agent(),
            "cute: {KIND}::create: the UMMA consumer group ({:?}) and the async one ({:?}) must count the same kind \
             of agent",
            consumer_group_umma.agent(),
            consumer_group_async.agent()
        );
        assert!(
            !(options.enable_multicast_signaling && options.per_lane_signaling == Some(true)),
            "cute: {KIND}::create: per-lane signalling and enable_multicast_signaling exclude each other"
        );
        let (umma, async_) = if options.enable_multicast_signaling {
            check_mcast_mode(options.mcast_mode_mn);
            let extent = |mode: usize| layout.shape().modes()[mode].size() as i64;
            let (mm, mn) = options.mcast_mode_mn;
            let mcast_size = if mm != 0 { extent(2) } else { 0 } + if mn != 0 { extent(1) } else { 0 }
                - i64::from(mm != 0 && mn != 0);
            assert!(mcast_size > 0, "cute: {KIND}::create: the cluster has no multicast partners");
            (
                signaling_group("UMMA", consumer_group_umma, mcast_size),
                signaling_group("async", consumer_group_async, mcast_size),
            )
        } else {
            (consumer_group_umma, consumer_group_async)
        };
        let per_lane = match options.per_lane_signaling {
            Some(p) => p,
            None => {
                let legacy = !options.enable_multicast_signaling && (looks_per_lane(&umma) || looks_per_lane(&async_));
                if legacy {
                    crate::cute_log!(
                        Level::Warn,
                        "{KIND}::create: a consumer group of a multiple of 32 threads selects the legacy per-lane \
                         signalling (every async thread arrives); count one arrival per warp, or set \
                         per_lane_signaling to Some(false)"
                    );
                }
                legacy
            }
        };
        assert!(
            !(per_lane && layout.size() > 1),
            "cute: {KIND}::create: per-lane signalling needs a single CTA, not the cluster {layout}"
        );
        let consumer = CooperativeGroup::new(umma.agent(), add(umma.size(), async_.size()));
        let (full, empty) = barrier_arrays(
            KIND,
            num_stages,
            options.barrier_storage.as_ref(),
            (PipelineOp::TmaLoad, producer_group),
            (PipelineOp::Composite, consumer),
            tx_count.into(),
            &options.name,
        );
        let empty_umma = empty.recast_to_new_op_type(PipelineOp::TCGen05Mma);
        let empty_async = empty.recast_to_new_op_type(PipelineOp::AsyncThread);
        let (dst_rank, is_signaling) = if per_lane {
            (None, Boolean::from(true))
        } else {
            let tidx = options.tidx.unwrap_or_else(|| thread_idx().x);
            let (dst, signals) = init_empty_barrier_arrive_signal_2sm(&layout, tidx, options.mcast_mode_mn);
            (Some(dst), signals)
        };
        let (consumer_mask, consumer_dst_rank_async, is_leader) = if layout.size() == 1 {
            (None, None, Boolean::from(true))
        } else {
            (Some(mcast_arrival_mask(&layout, options.mcast_mode_mn)), dst_rank, is_leader_cta(&layout))
        };
        let cta_group = cta_group_of(&layout);
        init_sync(options.defer_sync, &layout);
        PipelineTmaMultiConsumersAsync {
            sync_object_full: full,
            sync_object_empty: empty,
            sync_object_empty_umma: empty_umma,
            sync_object_empty_async: empty_async,
            num_stages,
            consumer_mask,
            is_leader_cta: is_leader,
            cta_group,
            consumer_dst_rank_async,
            is_signaling_thread: is_signaling,
            per_lane_signaling: per_lane,
        }
    }

    /// The full barriers, which the TMA loads complete.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers ([`PipelineOp::Composite`]: both groups arrive).
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The empty barriers as the MMA warp's commits arrive on them.
    pub fn sync_object_empty_umma(&self) -> &MbarrierArray {
        &self.sync_object_empty_umma
    }

    /// The empty barriers as the async threads arrive on them.
    pub fn sync_object_empty_async(&self) -> &MbarrierArray {
        &self.sync_object_empty_async
    }

    /// The multicast mask of the MMA's release commits (`None` for one CTA).
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.consumer_mask
    }

    /// The cluster rank this async thread arrives on (`None` for one CTA).
    pub fn consumer_dst_rank_async(&self) -> Option<Int32> {
        self.consumer_dst_rank_async
    }

    /// Whether this async thread arrives in its release.
    pub fn is_signaling_thread(&self) -> Boolean {
        self.is_signaling_thread
    }

    /// Whether this CTA arms the full barriers (the leader of its pair).
    pub fn is_leader_cta(&self) -> Boolean {
        self.is_leader_cta
    }

    /// The CTA group of the MMA's commits.
    pub fn cta_group(&self) -> CtaGroup {
        self.cta_group
    }

    /// Whether every async thread arrives (the legacy signalling).
    pub fn per_lane_signaling(&self) -> bool {
        self.per_lane_signaling
    }

    /// Release stage `state` for one consumer group (CuTe DSL's
    /// `consumer_release(state, op_type)`): [`PipelineOp::TCGen05Mma`] is
    /// the MMA warp's commit, [`PipelineOp::AsyncThread`] the signalling
    /// lanes' arrival (every thread's with per-lane signalling), at cluster
    /// scope on a peer CTA (see [`PipelineTmaAsync`](super::PipelineTmaAsync)
    /// for why). Panics for any other operation.
    #[track_caller]
    pub fn consumer_release_op<S: StageState>(&self, state: &S, op: PipelineOp) {
        let _at = crate::builder::LocationScope::new();
        match op {
            PipelineOp::TCGen05Mma => {
                self.sync_object_empty_umma.arrive(state.index(), self.consumer_mask, Some(self.cta_group))
            }
            PipelineOp::AsyncThread => {
                let release = || match self.consumer_dst_rank_async {
                    None => self.sync_object_empty_async.arrive(state.index(), None, None),
                    Some(rank) => mbarrier_arrive_with(
                        &self.sync_object_empty_async.get_barrier(state.index()),
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
            other => panic!(
                "cute: PipelineTmaMultiConsumersAsync::consumer_release_op: a consumer releases as TCGen05Mma or \
                 AsyncThread, not {}",
                other.name()
            ),
        }
    }
}

impl Pipeline for PipelineTmaMultiConsumersAsync {
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

    /// Wait until the stage is empty (unless the token says so), then, on
    /// the leader CTA, arm its full barrier with `tx_count` bytes.
    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.sync_object_empty.wait(state.index(), state.phase()));
        let index = state.index();
        let arm = || self.sync_object_full.arrive(index, None, None);
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

    /// The MMA warp's release ([`consumer_release_op`](Self::consumer_release_op)
    /// with [`PipelineOp::TCGen05Mma`], CuTe DSL's default).
    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.consumer_release_op(state, PipelineOp::TCGen05Mma);
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        for _ in 0..self.num_stages {
            self.sync_object_empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for PipelineTmaMultiConsumersAsync {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

impl ConsumerHandle<'_, PipelineTmaMultiConsumersAsync> {
    /// Release the stage for one consumer group
    /// ([`PipelineTmaMultiConsumersAsync::consumer_release_op`]).
    #[track_caller]
    pub fn release_op(&self, op: PipelineOp) {
        self.get_origin().consumer_release_op(&self.state(), op);
    }
}

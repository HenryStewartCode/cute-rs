// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.pipeline (TSPipelineTmaAsync,
// TSPipelineAsyncUmma, TSPipelineUmmaUmma; NVIDIA EULA); the code and the
// documentation are written independently.

//! Task Scheduling's own pipelines: a full and an empty
//! [`MbarrierArray`] whose arrival conventions follow the kinds of agent on
//! each side, with the hooks the TaskManager and pipeline groups need
//! (a task-warp leader that alone arms a TMA stage, every consumer thread
//! releasing a TMA stage, members that share one side's barriers).

use crate::arch::{lane_idx, make_warp_uniform, mbarrier_init_fence, warp_idx};
use crate::control::if_;
use crate::pipeline::{
    Agent, CooperativeGroup, CtaGroup, MbarrierArray, MbarrierArrayOptions, Pipeline, PipelineBarriers, PipelineOp,
    PipelineState, StageState, agent_sync, alloc_reserved_mbarrier,
};
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32};

use super::enums::PipelineType;

/// Which kind of agent fills (or drains) a stage of a [`TsPipeline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgentKind {
    /// Ordinary threads (`mbarrier.arrive`, or `cp.async`'s arrival).
    Threads,
    /// TMA loads, armed with the stage's transaction bytes.
    Tma,
    /// `tcgen05` MMAs, which arrive through `tcgen05.commit`.
    Umma,
}

impl AgentKind {
    /// The producer side of `t` (`None` for the cluster launch control
    /// pipeline, which Task Scheduling's pipelines do not cover).
    pub fn producer_of(t: PipelineType) -> Option<AgentKind> {
        use PipelineType as P;
        match t {
            P::AsyncAsync | P::AsyncUmma => Some(AgentKind::Threads),
            P::TmaAsync | P::TmaUmma => Some(AgentKind::Tma),
            P::UmmaAsync | P::UmmaUmma => Some(AgentKind::Umma),
            P::ClcFetchAsync => None,
        }
    }

    /// The consumer side of `t`.
    pub fn consumer_of(t: PipelineType) -> Option<AgentKind> {
        use PipelineType as P;
        match t {
            P::AsyncAsync | P::TmaAsync | P::UmmaAsync => Some(AgentKind::Threads),
            P::TmaUmma | P::AsyncUmma | P::UmmaUmma => Some(AgentKind::Umma),
            P::ClcFetchAsync => None,
        }
    }
}

/// The optional arguments of the [`TsPipeline`] constructors.
#[derive(Clone, Debug, Default)]
pub struct TsPipelineOptions {
    /// Shared memory for the `2 * num_stages` mbarriers (full ones first);
    /// `None` takes them from the reserved partition.
    pub barrier_storage: Option<Ptr<u64>>,
    /// For a TMA producer: only the threads for which this holds arm a
    /// stage's transaction bytes when they acquire it (CuTe DSL's
    /// task-warp leader, so that several producer warps of one task make
    /// one arrival carrying the whole stage). `None`: every producer warp's
    /// elected lane arms it, as the plain TMA pipeline does.
    pub is_task_warp_leader: Option<Boolean>,
    /// For a thread consumer of a TMA pipeline: every consumer thread
    /// releases (the consumer group counts threads). Otherwise lane 0 of
    /// each consumer warp releases and the group counts warps, as
    /// [`PipelineTmaAsync`](crate::pipeline::PipelineTmaAsync) does.
    pub all_thread_release: bool,
    /// The operation thread producers commit with: `AsyncThread` (the
    /// default) or `AsyncLoad` for `cp.async` producers.
    pub producer_op: Option<PipelineOp>,
    /// Leave the barriers' initialisation unpublished (the caller fences
    /// and synchronises once for every pipeline).
    pub defer_sync: bool,
    /// A name for profiling.
    pub name: String,
}

/// A pipeline of Task Scheduling (CuTe DSL's `TSPipelineTmaAsync`,
/// `TSPipelineAsyncUmma` and `TSPipelineUmmaUmma`, and the member
/// pipelines of a [`PipelineGroup`](super::PipelineGroup)): `num_stages`
/// full mbarriers that the producers complete and as many empty ones that
/// the consumers complete, each side arriving the way its agents do:
///
/// * thread producers arrive on the full barrier when they commit; TMA
///   producers arm it with the stage's bytes when they acquire (the task's
///   leader warp only, with a task-warp-leader predicate) and their loads
///   complete it; UMMA producers commit (`tcgen05.commit`), so the arrival
///   comes once their MMAs are done;
/// * thread consumers arrive on the empty barrier when they release (for a
///   TMA pipeline, lane 0 of each warp unless every thread releases); UMMA
///   consumers commit to it.
///
/// One CTA only (no cluster layouts yet). Constructors:
/// [`create_tma_async`](Self::create_tma_async),
/// [`create_async_umma`](Self::create_async_umma),
/// [`create_umma_umma`](Self::create_umma_umma) and the general
/// [`create`](Self::create).
///
/// ```
/// use cute::pipeline::{Agent, CooperativeGroup, Pipeline};
/// use cute::prelude::*;
/// use cute::task_scheduling::{TsPipeline, TsPipelineOptions, PipelineType};
///
/// let k = Kernel::builder("ts_pipe").block_dim([64, 1, 1]).target(Arch::Sm100a).build(|| {
///     let one = CooperativeGroup::new(Agent::Thread, 1);
///     let p = TsPipeline::create_umma_umma(2, one, one, TsPipelineOptions::default());
///     assert_eq!(p.pipeline_type(), PipelineType::UmmaUmma);
///     assert_eq!(p.num_stages(), 2);
/// });
/// assert!(k.to_mlir().contains("mbarrier"));
/// ```
#[derive(Clone, Debug)]
pub struct TsPipeline {
    pipeline_type: PipelineType,
    full: MbarrierArray,
    empty: MbarrierArray,
    num_stages: usize,
    tx_bytes: i32,
    arm_predicate: Option<Boolean>,
    arms_full: bool,
    release_predicate: Option<Boolean>,
}

/// Whether the kernel being built waits portably (below sm_90).
#[track_caller]
fn portable() -> bool {
    crate::target_arch().is_some_and(|arch| arch.sm() < 90)
}

/// Publish barrier initialisation to the block.
#[track_caller]
pub(crate) fn publish_init() {
    if !portable() {
        mbarrier_init_fence();
    }
    agent_sync(Agent::ThreadBlock, false);
}

/// Wait unless the probe's token says the phase is done.
#[track_caller]
fn wait_unless(token: Option<Boolean>, wait: impl FnOnce()) {
    match token {
        None => wait(),
        Some(t) => match t.as_constant() {
            Some(true) => {}
            Some(false) => wait(),
            None => if_(!t, wait),
        },
    }
}

/// The barrier operation each side's agents arrive with.
pub(crate) fn full_op(t: PipelineType, producer_op: Option<PipelineOp>) -> PipelineOp {
    match AgentKind::producer_of(t) {
        Some(AgentKind::Tma) => PipelineOp::TmaLoad,
        Some(AgentKind::Umma) => PipelineOp::TCGen05Mma,
        _ => producer_op.unwrap_or(PipelineOp::AsyncThread),
    }
}

pub(crate) fn empty_op(t: PipelineType) -> PipelineOp {
    match AgentKind::consumer_of(t) {
        Some(AgentKind::Umma) => PipelineOp::TCGen05Mma,
        _ => PipelineOp::AsyncThread,
    }
}

/// Initialise `num_stages` mbarriers at `at` for `group`.
#[track_caller]
pub(crate) fn make_barriers(
    at: &Ptr<u64>,
    num_stages: usize,
    op: PipelineOp,
    group: CooperativeGroup,
    tx: i32,
    name: String,
) -> MbarrierArray {
    MbarrierArray::with_options(at, num_stages, (op, group), tx, MbarrierArrayOptions { name, ..Default::default() })
}

impl TsPipeline {
    /// A pipeline of type `pipeline_type` (any but `ClcFetchAsync`) with
    /// `num_stages` stages between `producer_group` and `consumer_group`,
    /// a TMA producer's stage delivering `tx_bytes`. Initialises the
    /// barriers from warp 0 and, unless `defer_sync`, publishes them (every
    /// thread of the block must reach this).
    #[track_caller]
    pub fn create(
        pipeline_type: PipelineType,
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_bytes: u32,
        options: TsPipelineOptions,
    ) -> TsPipeline {
        assert!(
            pipeline_type != PipelineType::ClcFetchAsync,
            "cute: TsPipeline: cluster launch control pipelines are not Task Scheduling pipelines"
        );
        assert!(num_stages > 0, "cute: TsPipeline: a pipeline needs at least one stage");
        let _at = crate::builder::LocationScope::new();
        crate::profiling::register_barrier(&options.name, num_stages, "TsPipeline");
        let storage = options.barrier_storage.clone().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
        let named =
            |side: &str| if options.name.is_empty() { String::new() } else { format!("{}.{side}", options.name) };
        let full = make_barriers(
            &storage,
            num_stages,
            full_op(pipeline_type, options.producer_op),
            producer_group,
            tx_bytes as i32,
            named("full"),
        );
        let empty = make_barriers(
            &storage.offset(num_stages),
            num_stages,
            empty_op(pipeline_type),
            consumer_group,
            0,
            named("empty"),
        );
        let release_predicate = (AgentKind::producer_of(pipeline_type) == Some(AgentKind::Tma)
            && AgentKind::consumer_of(pipeline_type) == Some(AgentKind::Threads)
            && !options.all_thread_release)
            .then(|| lane_idx().eq(0));
        if !options.defer_sync {
            publish_init();
        }
        TsPipeline {
            pipeline_type,
            full,
            empty,
            num_stages,
            tx_bytes: tx_bytes as i32,
            arm_predicate: options.is_task_warp_leader,
            arms_full: true,
            release_predicate,
        }
    }

    /// TMA loads fill the stages and threads drain them (CuTe DSL's
    /// `TSPipelineTmaAsync`): set
    /// [`is_task_warp_leader`](TsPipelineOptions::is_task_warp_leader) so
    /// that only the producer task's leader warp arms each stage.
    #[track_caller]
    pub fn create_tma_async(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_bytes: u32,
        options: TsPipelineOptions,
    ) -> TsPipeline {
        TsPipeline::create(PipelineType::TmaAsync, num_stages, producer_group, consumer_group, tx_bytes, options)
    }

    /// Threads (or `cp.async` copies, with
    /// [`producer_op`](TsPipelineOptions::producer_op) `AsyncLoad`) fill
    /// the stages and `tcgen05` MMAs drain them, releasing with
    /// `tcgen05.commit` (CuTe DSL's `TSPipelineAsyncUmma`).
    #[track_caller]
    pub fn create_async_umma(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: TsPipelineOptions,
    ) -> TsPipeline {
        TsPipeline::create(PipelineType::AsyncUmma, num_stages, producer_group, consumer_group, 0, options)
    }

    /// `tcgen05` MMAs on both sides (CuTe DSL's `TSPipelineUmmaUmma`): the
    /// producing MMAs commit to the full barrier, the consuming ones to the
    /// empty barrier, so a stage (an accumulator in TMEM, say) passes from
    /// one MMA warp to another without a thread arrival.
    #[track_caller]
    pub fn create_umma_umma(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: TsPipelineOptions,
    ) -> TsPipeline {
        TsPipeline::create(PipelineType::UmmaUmma, num_stages, producer_group, consumer_group, 0, options)
    }

    /// A pipeline over barriers made elsewhere (a group's shared ones).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        pipeline_type: PipelineType,
        full: MbarrierArray,
        empty: MbarrierArray,
        tx_bytes: i32,
        arm_predicate: Option<Boolean>,
        arms_full: bool,
        release_predicate: Option<Boolean>,
    ) -> TsPipeline {
        let num_stages = full.num_stages();
        TsPipeline { pipeline_type, full, empty, num_stages, tx_bytes, arm_predicate, arms_full, release_predicate }
    }

    /// The pipeline's type.
    pub fn pipeline_type(&self) -> PipelineType {
        self.pipeline_type
    }

    /// The full barriers.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.full
    }

    /// The empty barriers.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.empty
    }

    /// The bytes a TMA producer arms a stage with.
    pub fn tx_bytes(&self) -> i32 {
        self.tx_bytes
    }

    /// Whether acquiring a stage arms the full barrier (false for the
    /// secondary members of a fork group, whose shared full barrier the
    /// first member arms).
    pub fn arms_full(&self) -> bool {
        self.arms_full
    }

    /// The leader predicate of a TMA producer, if any.
    pub fn is_task_warp_leader(&self) -> Option<Boolean> {
        self.arm_predicate
    }

    fn producer_kind(&self) -> AgentKind {
        AgentKind::producer_of(self.pipeline_type).expect("not a CLC pipeline")
    }

    fn consumer_kind(&self) -> AgentKind {
        AgentKind::consumer_of(self.pipeline_type).expect("not a CLC pipeline")
    }

    /// Arrive on the empty barrier of `index` as the consumers do.
    #[track_caller]
    pub(crate) fn release_index(&self, index: Int32) {
        match self.consumer_kind() {
            AgentKind::Umma => self.empty.arrive(index, None, Some(CtaGroup::One)),
            _ => {
                let arrive = || self.empty.arrive(index, None, None);
                match self.release_predicate {
                    Some(p) => if_(p, arrive),
                    None => arrive(),
                }
            }
        }
    }

    /// Arrive on the full barrier of `index` as the producers commit.
    #[track_caller]
    pub(crate) fn commit_index(&self, index: Int32) {
        match self.producer_kind() {
            AgentKind::Tma => {}
            AgentKind::Umma => self.full.arrive(index, None, Some(CtaGroup::One)),
            AgentKind::Threads => self.full.arrive(index, None, None),
        }
    }
}

/// The task-warp-leader predicate for a task starting at warp `warp`.
#[track_caller]
pub(crate) fn task_warp_leader(warp: u32) -> Boolean {
    make_warp_uniform(warp_idx()).eq(warp as i32)
}

impl Pipeline for TsPipeline {
    type AcquireOptions = ();

    fn num_stages(&self) -> usize {
        self.num_stages
    }

    fn producer_group(&self) -> Option<CooperativeGroup> {
        Some(self.full.group())
    }

    fn consumer_group(&self) -> Option<CooperativeGroup> {
        Some(self.empty.group())
    }

    #[track_caller]
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, (): ()) {
        wait_unless(token, || self.empty.wait(state.index(), state.phase()));
        if self.producer_kind() == AgentKind::Tma && self.arms_full {
            let (idx, tx) = (state.index(), self.tx_bytes);
            let arm = || self.full.arrive_and_expect_tx(idx, tx);
            match self.arm_predicate {
                Some(p) => if_(p, arm),
                None => arm(),
            }
        }
    }

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean {
        self.empty.try_wait(state.index(), state.phase())
    }

    #[track_caller]
    fn producer_commit<S: StageState>(&self, state: &S) {
        self.commit_index(state.index());
    }

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.full.try_wait(state.index(), state.phase())
    }

    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.release_index(state.index());
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        for _ in 0..self.num_stages {
            self.empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for TsPipeline {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.empty.get_barrier(state.index())
    }
}

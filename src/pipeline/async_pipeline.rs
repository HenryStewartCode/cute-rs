//! The interface every pipeline shares ([`Pipeline`], [`PipelineBarriers`]),
//! the generic full/empty mbarrier pipeline ([`PipelineAsync`]) and the
//! pipeline that does nothing ([`PipelineEmpty`]).

use crate::control::if_;
use crate::tensor::Ptr;
use crate::value::Boolean;

use super::mbarrier::portable_waits;
use super::{
    Agent, CooperativeGroup, MbarrierArray, MbarrierArrayOptions, PipelineConsumer, PipelineOp, PipelineProducer,
    PipelineState, PipelineUserType, StageState, agent_sync, alloc_reserved_mbarrier, make_pipeline_state,
};
use crate::arch::mbarrier_init_fence;
use crate::value::Int32;

/// A pipeline: a ring of stages that producers fill and consumers drain,
/// each side waiting for the other through one barrier per stage (CuTe
/// DSL's `PipelineAsync` interface, which its specialised pipelines
/// inherit). [`PipelineAsync`] implements it here, the `cp.async` and TMA
/// pipelines and the tcgen05 pipelines do too, and [`PipelineEmpty`]
/// implements it with no barriers at all.
///
/// The methods take the position in the ring as any [`StageState`]
/// ([`PipelineState`] usually). A *token* is the result of an earlier
/// [`producer_try_acquire`](Self::producer_try_acquire) or
/// [`consumer_try_wait`](Self::consumer_try_wait) at the same position:
/// `None` waits unconditionally, `Some(t)` waits only when `t` is false (a
/// constant token decides at build time).
///
/// [`make_producer`](Self::make_producer),
/// [`make_consumer`](Self::make_consumer) and
/// [`make_participants`](Self::make_participants) wrap the pipeline in the
/// stateful [`PipelineProducer`] and [`PipelineConsumer`].
pub trait Pipeline {
    /// What a producer's acquire takes besides the state and the token:
    /// `()` for [`PipelineAsync`]; a TMA pipeline whose transaction byte
    /// count varies per acquire passes it here.
    type AcquireOptions: Default;

    /// The number of stages in the ring.
    fn num_stages(&self) -> usize;

    /// The group of producer threads, which the participants keep (`None`
    /// for a pipeline without barriers).
    fn producer_group(&self) -> Option<CooperativeGroup>;

    /// The group of consumer threads.
    fn consumer_group(&self) -> Option<CooperativeGroup>;

    /// Wait until the consumers have released stage `state.index()` (unless
    /// `token` says the stage is already free), with pipeline-specific
    /// options.
    fn producer_acquire_with<S: StageState>(&self, state: &S, token: Option<Boolean>, options: Self::AcquireOptions);

    /// [`producer_acquire_with`](Self::producer_acquire_with) with the
    /// default options (CuTe DSL's `producer_acquire`).
    #[track_caller]
    fn producer_acquire<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        self.producer_acquire_with(state, token, Self::AcquireOptions::default());
    }

    /// Whether stage `state.index()` is free already, probing once (CuTe
    /// DSL's `producer_try_acquire`): a token for
    /// [`producer_acquire`](Self::producer_acquire).
    fn producer_try_acquire<S: StageState>(&self, state: &S) -> Boolean;

    /// Signal that the producers have filled stage `state.index()`.
    fn producer_commit<S: StageState>(&self, state: &S);

    /// Wait until the producers have filled stage `state.index()` (unless
    /// `token` says it is full already).
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>);

    /// Whether stage `state.index()` is full already, probing once: a token
    /// for [`consumer_wait`](Self::consumer_wait).
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean;

    /// Signal that the consumers are done with stage `state.index()`.
    fn consumer_release<S: StageState>(&self, state: &S);

    /// Before the producers exit: wait until the consumers have released
    /// every stage, so that no arrival is left in flight when the block
    /// exits. Advances `state` once per stage, as CuTe DSL does.
    fn producer_tail(&self, state: &mut PipelineState);

    /// A producer starting at the beginning of the ring
    /// ([`make_pipeline_state`] of [`PipelineUserType::Producer`]).
    #[track_caller]
    fn make_producer(&self) -> PipelineProducer<'_, Self>
    where
        Self: Sized,
    {
        let state = make_pipeline_state(PipelineUserType::Producer, self.num_stages());
        PipelineProducer::new(self, state, self.producer_group())
    }

    /// A consumer starting at the beginning of the ring.
    #[track_caller]
    fn make_consumer(&self) -> PipelineConsumer<'_, Self>
    where
        Self: Sized,
    {
        let state = make_pipeline_state(PipelineUserType::Consumer, self.num_stages());
        PipelineConsumer::new(self, state, self.consumer_group())
    }

    /// [`make_producer`](Self::make_producer) and
    /// [`make_consumer`](Self::make_consumer).
    #[track_caller]
    fn make_participants(&self) -> (PipelineProducer<'_, Self>, PipelineConsumer<'_, Self>)
    where
        Self: Sized,
    {
        (self.make_producer(), self.make_consumer())
    }
}

/// A pipeline whose stages have barriers to hand out (CuTe DSL's
/// `producer_get_barrier` and `consumer_get_barrier`), for operations that
/// arrive on a barrier themselves, such as a TMA load or `cp.async`.
/// [`PipelineEmpty`] has no barriers and does not implement it, so code
/// that needs one does not compile with it (CuTe DSL leaves the methods off
/// it so that the mistake fails loudly).
pub trait PipelineBarriers: Pipeline {
    /// The barrier the producers arrive on for stage `state.index()` (the
    /// full barrier).
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64>;

    /// The barrier the consumers arrive on for stage `state.index()` (the
    /// empty barrier).
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64>;
}

/// Wait unless the token says the probe succeeded: always for `None`,
/// never for a constant true token, and under `!t` otherwise.
#[track_caller]
pub(crate) fn wait_unless(token: Option<Boolean>, wait: impl FnOnce()) {
    match token {
        None => wait(),
        Some(t) => match t.as_constant() {
            Some(true) => {}
            Some(false) => wait(),
            None => if_(!t, wait),
        },
    }
}

/// The optional arguments of [`PipelineAsync::create`].
#[derive(Clone, Debug, Default)]
pub struct PipelineAsyncOptions {
    /// Shared memory for the `2 * num_stages` mbarriers: full barriers
    /// first, then empty ones. `None` allocates them with
    /// [`alloc_reserved_mbarrier`].
    pub barrier_storage: Option<Ptr<u64>>,
    /// The cluster rank whose full barriers the producers arrive on (CuTe
    /// DSL's `producer_mask`; sm_90). `None` arrives on this CTA's.
    pub producer_mask: Option<Int32>,
    /// The cluster rank whose empty barriers the consumers arrive on.
    pub consumer_mask: Option<Int32>,
    /// Leave out the initialisation sync, for a kernel that publishes
    /// several pipelines' barriers with one
    /// [`pipeline_init_arrive`](super::pipeline_init_arrive) and
    /// [`pipeline_init_wait`](super::pipeline_init_wait) (or a cluster
    /// sync, which a pipeline with peer masks needs).
    pub defer_sync: bool,
    /// A name for profiling; the barrier arrays are named `{name}.full`
    /// and `{name}.empty`.
    pub name: String,
}

/// The generic pipeline (CuTe DSL's `cutlass.pipeline.PipelineAsync`): a
/// full and an empty [`MbarrierArray`] of `num_stages` stages each, both
/// arrived on by ordinary threads ([`PipelineOp::AsyncThread`]). The full
/// barriers count the producer group's threads, the empty ones the
/// consumer group's.
///
/// A stage moves through four states: a producer acquires it (waits on its
/// empty barrier), fills it and commits (arrives on its full barrier); a
/// consumer waits on the full barrier, reads the stage and releases it
/// (arrives on the empty barrier). The producer's [`PipelineState`] starts
/// at phase 1, so its first pass over the ring does not wait.
///
/// The barriers' waits follow [`MbarrierArray::wait`]: for a kernel built
/// for a target below sm_90 they spin on `mbarrier.test_wait`, the probes
/// ([`producer_try_acquire`](Pipeline::producer_try_acquire),
/// [`consumer_try_wait`](Pipeline::consumer_try_wait)) are a single
/// `mbarrier.test_wait`, and the init sync has no `fence.mbarrier_init`, so
/// the pipeline runs on SM80 to SM89.
///
/// # Warp roles
///
/// CuTe DSL kernels give each warp a role with `if warp_idx == 0:` and
/// mutate a participant inside the branch, and the DSL carries it out. In
/// cute-rs a branch returns what it changed: keep each participant inside
/// its own branch and loop, or return it from the branch with
/// [`if_else_yield`](crate::if_else_yield):
///
/// ```
/// use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("handoff").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
///     let out = param_tensor::<u32>("out", Layout::col_major(32));
///     let warp = CooperativeGroup::new(Agent::Warp, 1);
///     let pipeline = PipelineAsync::create(2, warp, warp, PipelineAsyncOptions::default());
///     let data = make_smem_tensor::<u32>(Layout::col_major(2 * 32));
///     let (producer, consumer) = pipeline.make_participants();
///     let lane = lane_idx();
///     if_else(
///         warp_idx().eq(0),
///         || {
///             let mut p = for_range_carry(0, 5, 1, producer, |k, mut p| {
///                 let handle = p.acquire_and_advance(None);
///                 data.store(handle.index() * 32 + lane, k.to::<u32>());
///                 handle.commit();
///                 p
///             });
///             p.tail();
///         },
///         || {
///             let (sum, _) = for_range_carry(0, 5, 1, (Uint32::from(0u32), consumer), |_, (sum, mut c)| {
///                 let handle = c.wait_and_advance(None);
///                 let x = data.load(handle.index() * 32 + lane);
///                 handle.release();
///                 (sum + x, c)
///             });
///             out.store(lane, sum);
///         },
///     );
/// });
/// let mut args = vec![KernelArg::buffer(&[0u32; 32])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].to_vec::<u32>(), [10u32; 32]);
/// ```
#[derive(Clone, Debug)]
pub struct PipelineAsync {
    sync_object_full: MbarrierArray,
    sync_object_empty: MbarrierArray,
    num_stages: usize,
    producer_mask: Option<Int32>,
    consumer_mask: Option<Int32>,
}

impl PipelineAsync {
    /// A pipeline of `num_stages` stages between `producer_group` and
    /// `consumer_group` (CuTe DSL's `PipelineAsync.create`): full barriers
    /// at the storage, empty ones `num_stages` further on, initialised from
    /// warp 0. Unless [`PipelineAsyncOptions::defer_sync`] is set, it then
    /// publishes them to the block: `fence.mbarrier_init` (not below
    /// sm_90) and a CTA barrier, which every thread must reach.
    ///
    /// Panics for 0 stages.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        options: PipelineAsyncOptions,
    ) -> PipelineAsync {
        PipelineAsync::create_with_ops(
            "PipelineAsync",
            num_stages,
            (PipelineOp::AsyncThread, producer_group),
            (PipelineOp::AsyncThread, consumer_group),
            options,
        )
    }

    /// [`create`](Self::create) with the operations that arrive on the
    /// full and the empty barriers: the shared body of the full/empty
    /// pipelines (`PipelineCpAsync` arrives on its full barriers with
    /// [`PipelineOp::AsyncLoad`]). `kind` names the pipeline in panics.
    #[track_caller]
    pub(crate) fn create_with_ops(
        kind: &str,
        num_stages: usize,
        producer: (PipelineOp, CooperativeGroup),
        consumer: (PipelineOp, CooperativeGroup),
        options: PipelineAsyncOptions,
    ) -> PipelineAsync {
        assert!(num_stages > 0, "cute: {kind}::create: a pipeline needs at least one stage, not 0");
        let _at = crate::builder::LocationScope::new();
        crate::profiling::register_barrier(&options.name, num_stages, kind);
        let storage = options.barrier_storage.clone().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
        let named = |side: &str| MbarrierArrayOptions {
            name: if options.name.is_empty() { String::new() } else { format!("{}.{side}", options.name) },
            ..Default::default()
        };
        let sync_object_full = MbarrierArray::with_options(&storage, num_stages, producer, 0, named("full"));
        let sync_object_empty =
            MbarrierArray::with_options(&storage.offset(num_stages), num_stages, consumer, 0, named("empty"));
        if !options.defer_sync {
            if !portable_waits() {
                mbarrier_init_fence();
            }
            agent_sync(Agent::ThreadBlock, false);
        }
        PipelineAsync {
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask: options.producer_mask,
            consumer_mask: options.consumer_mask,
        }
    }

    /// The full barriers, which the producers arrive on.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The empty barriers, which the consumers arrive on.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        &self.sync_object_empty
    }

    /// The cluster rank the producers' arrivals go to, if not this CTA.
    pub fn producer_mask(&self) -> Option<Int32> {
        self.producer_mask
    }

    /// The cluster rank the consumers' arrivals go to, if not this CTA.
    pub fn consumer_mask(&self) -> Option<Int32> {
        self.consumer_mask
    }
}

impl Pipeline for PipelineAsync {
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

    #[track_caller]
    fn producer_commit<S: StageState>(&self, state: &S) {
        self.sync_object_full.arrive(state.index(), self.producer_mask, None);
    }

    #[track_caller]
    fn consumer_wait<S: StageState>(&self, state: &S, token: Option<Boolean>) {
        wait_unless(token, || self.sync_object_full.wait(state.index(), state.phase()));
    }

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, state: &S) -> Boolean {
        self.sync_object_full.try_wait(state.index(), state.phase())
    }

    #[track_caller]
    fn consumer_release<S: StageState>(&self, state: &S) {
        self.sync_object_empty.arrive(state.index(), self.consumer_mask, None);
    }

    #[track_caller]
    fn producer_tail(&self, state: &mut PipelineState) {
        // The consumers may release the stages in any order, so wait on
        // every one of them.
        for _ in 0..self.num_stages {
            self.sync_object_empty.wait(state.index(), state.phase());
            state.advance();
        }
    }
}

impl PipelineBarriers for PipelineAsync {
    #[track_caller]
    fn producer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_full.get_barrier(state.index())
    }

    #[track_caller]
    fn consumer_get_barrier<S: StageState>(&self, state: &S) -> Ptr<u64> {
        self.sync_object_empty.get_barrier(state.index())
    }
}

/// A pipeline with no barriers, whose operations do nothing (CuTe DSL's
/// `cutlass.pipeline.PipelineEmpty`): it stands in for a real pipeline in a
/// configuration that needs none, so that one kernel body, generic over
/// [`Pipeline`], serves both (the static tile scheduler needs no pipeline
/// where the cluster launch control scheduler needs one). The probes return
/// a constant true, so an acquire or wait given their token folds away. It
/// has one notional stage, so that participants can be made from it.
///
/// It does not implement [`PipelineBarriers`]: code that asks it for a
/// barrier does not compile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineEmpty;

impl PipelineEmpty {
    /// The empty pipeline (CuTe DSL's `PipelineEmpty.create`, which ignores
    /// its arguments).
    pub fn create() -> PipelineEmpty {
        PipelineEmpty
    }
}

impl Pipeline for PipelineEmpty {
    type AcquireOptions = ();

    fn num_stages(&self) -> usize {
        1
    }

    fn producer_group(&self) -> Option<CooperativeGroup> {
        None
    }

    fn consumer_group(&self) -> Option<CooperativeGroup> {
        None
    }

    fn producer_acquire_with<S: StageState>(&self, _: &S, _: Option<Boolean>, (): ()) {}

    #[track_caller]
    fn producer_try_acquire<S: StageState>(&self, _: &S) -> Boolean {
        Boolean::from(true)
    }

    fn producer_commit<S: StageState>(&self, _: &S) {}

    fn consumer_wait<S: StageState>(&self, _: &S, _: Option<Boolean>) {}

    #[track_caller]
    fn consumer_try_wait<S: StageState>(&self, _: &S) -> Boolean {
        Boolean::from(true)
    }

    fn consumer_release<S: StageState>(&self, _: &S) {}

    fn producer_tail(&self, _: &mut PipelineState) {}
}

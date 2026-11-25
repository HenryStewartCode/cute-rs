//! cute_ext's pipeline classes: [`TMAStorePipeline`] (roadmap part
//! m11-cute-ext-tma), the staging of a TMA-store epilogue, and the
//! convenience classes over the low-level operations of
//! [`pipeline_ops`](super::pipeline_ops) (roadmap part m11-cute-ext-pipelines):
//! [`GenericPipelineBase`] and [`GenericPipeline`], [`TMAToUMMAPipeline`],
//! [`TMAToUMMACircularPipeline`], [`TMAToAsyncPipeline`],
//! [`AsyncToUMMAPipeline`] and [`UMMAtoAsyncPipeline`]; and the scheduler
//! pipelines (roadmap part m11-cute-ext-sched-tmem), which broadcast
//! scheduling results from shared memory to every warp:
//! [`GroupedGemmSchedulerPipeline`] and [`CLCPipeline`].

use crate::arch::{fence_view_async_shared, warp_idx};
use crate::builder;
use crate::control::{Carry, if_};
use crate::ir::ValueId;
use crate::pipeline::{Agent, CooperativeGroup, NamedBarrier, PipelineTmaStore};
use crate::value::{Int32, IntoValue};

/// A TMA-store epilogue's staging (cute_ext's `TMAStorePipeline(stages,
/// arv_count, barrier_id, tma_warp_id, index=0)`): several warps write a
/// tile into stage [`get_index`](Self::get_index) of a shared buffer of
/// `stages` stages, and warp `tma_warp_id` (one of them) TMA-stores it. A
/// named barrier `barrier_id` of `arv_count` threads (every thread that
/// calls the pipeline's operations, the TMA warp's included) orders the
/// two sides. Each tile:
///
/// 1. [`acquire_sync`](Self::acquire_sync): the TMA warp waits until the
///    stage about to be written is no longer read by an earlier store
///    (`cp.async.bulk.wait_group.read`, `stages - 1` groups outstanding,
///    0 with one stage), then every thread hits the barrier;
/// 2. the threads write the stage;
/// 3. [`commit_sync`](Self::commit_sync): every thread fences its writes for
///    the async proxy (`fence.proxy.async.shared::cta`) and hits the
///    barrier;
/// 4. the TMA warp stores the stage ([`tma_store`](super::tma_store));
/// 5. [`release_advance`](Self::release_advance): the TMA warp commits its
///    bulk async-group, and every thread moves to the next stage.
///
/// [`tail`](Self::tail) waits for the last stores before the CTA exits.
/// The stage index is a runtime value carried through loops (the type
/// implements [`Carry`]: carry the pipeline through a `for_range_carry`).
/// It is a thin layer over [`PipelineTmaStore`] and [`NamedBarrier`].
#[derive(Clone, Copy, Debug)]
pub struct TMAStorePipeline {
    stages: usize,
    store: PipelineTmaStore,
    barrier: NamedBarrier,
    tma_warp_id: i32,
    index: Int32,
}

impl TMAStorePipeline {
    /// A pipeline over `stages` stages, synchronised by named barrier
    /// `barrier_id` of `arv_count` threads (a multiple of 32), whose stores
    /// warp `tma_warp_id` issues; it starts at stage 0.
    #[track_caller]
    pub fn new(stages: usize, arv_count: usize, barrier_id: u32, tma_warp_id: u32) -> TMAStorePipeline {
        TMAStorePipeline::with_index(stages, arv_count, barrier_id, tma_warp_id, 0)
    }

    /// [`new`](Self::new) starting at stage `index`.
    #[track_caller]
    pub fn with_index(
        stages: usize,
        arv_count: usize,
        barrier_id: u32,
        tma_warp_id: u32,
        index: impl IntoValue<i32>,
    ) -> TMAStorePipeline {
        let _at = builder::LocationScope::new();
        assert!(stages > 0, "experimental::TMAStorePipeline: it needs at least one stage");
        assert!(
            arv_count >= 32 && arv_count.is_multiple_of(32),
            "experimental::TMAStorePipeline: arv_count {arv_count} counts the threads of whole warps (the writers and \
             the TMA warp): a positive multiple of 32"
        );
        if let Some([x, y, z]) = builder::with(|b| b.block_dim) {
            let threads = (x * y * z) as usize;
            assert!(
                arv_count <= threads,
                "experimental::TMAStorePipeline: arv_count {arv_count} is more threads than the block's {threads}"
            );
            assert!(
                (tma_warp_id as usize + 1) * 32 <= threads,
                "experimental::TMAStorePipeline: the TMA warp {tma_warp_id} is not a warp of the block's {threads} \
                 threads"
            );
        }
        let store = PipelineTmaStore::create(stages, CooperativeGroup::new(Agent::Thread, 32));
        let barrier = NamedBarrier::new(barrier_id, arv_count);
        TMAStorePipeline { stages, store, barrier, tma_warp_id: tma_warp_id as i32, index: index.into_value() }
    }

    /// The number of stages (`get_num_stages`).
    pub fn get_num_stages(&self) -> usize {
        self.stages
    }

    /// The stage the next tile goes to (`get_index`).
    pub fn get_index(&self) -> Int32 {
        self.index
    }

    /// The named barrier.
    pub fn barrier(&self) -> NamedBarrier {
        self.barrier
    }

    /// The warp that issues the TMA stores (`tma_warp_id`).
    pub fn tma_warp_id(&self) -> i32 {
        self.tma_warp_id
    }

    #[track_caller]
    fn on_tma_warp(&self, f: impl FnOnce()) {
        if_(warp_idx().eq(self.tma_warp_id), f);
    }

    /// Wait until the current stage may be written (see the type).
    #[track_caller]
    pub fn acquire_sync(&self) {
        let _at = builder::LocationScope::new();
        self.on_tma_warp(|| self.store.producer_acquire());
        self.barrier.arrive_and_wait();
    }

    /// Publish the stage's writes to the TMA warp (see the type).
    #[track_caller]
    pub fn commit_sync(&self) {
        let _at = builder::LocationScope::new();
        fence_view_async_shared();
        self.barrier.arrive_and_wait();
    }

    /// Commit the TMA warp's stores and move to the next stage (see the
    /// type).
    #[track_caller]
    pub fn release_advance(&mut self) {
        let _at = builder::LocationScope::new();
        self.on_tma_warp(|| self.store.producer_commit());
        self.index = (self.index + 1) % self.stages as i32;
    }

    /// Wait for every store to have read its stage, then the barrier, before
    /// the buffer is released or the CTA exits.
    #[track_caller]
    pub fn tail(&self) {
        let _at = builder::LocationScope::new();
        self.on_tma_warp(|| self.store.producer_tail());
        self.barrier.arrive_and_wait();
    }
}

impl Carry for TMAStorePipeline {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.index.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        TMAStorePipeline { index: self.index.rebuild(values), ..*self }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.stages != other.stages || self.tma_warp_id != other.tma_warp_id).then(|| {
            format!(
                "a TMAStorePipeline of {} stages (TMA warp {}) became one of {} stages (TMA warp {})",
                self.stages, self.tma_warp_id, other.stages, other.tma_warp_id
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience classes (m11-cute-ext-pipelines)
// ---------------------------------------------------------------------------

use super::core::OperationType;
use super::pipeline_ops::{
    self as ops, ArrivalCountOptions, ArrivalMask, CircularBufferPipelineState, OperationTypes, PipelineLocale,
    RawPipeline, Side, StageToken,
};
use crate::arch::block_idx_in_cluster;
use crate::layout::{Coord, IndexExpr, Layout, Tree, make_layout_image_mask};
use crate::pipeline::PipelineState;
use crate::value::{Boolean, Int16};

/// The commit and release flags a class fixes, whatever its caller passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Overrides {
    producer_commit: Option<(bool, bool)>,
    consumer_release: Option<(bool, bool)>,
}

/// The state every convenience pipeline keeps (CuTe DSL's
/// `GenericPipelineBase`): the [`RawPipeline`] and the producer's and the
/// consumer's [`PipelineState`], which its methods advance. The method
/// names are CuTe DSL's; those that return the pipeline in CuTe DSL (for
/// chaining) return `&mut Self`. The classes made from it fix some commit
/// and release flags (a TMA warp's commit elects one thread, say), so the
/// flags their callers pass may be ignored, as in CuTe DSL.
///
/// It implements [`Carry`]: carry the pipeline through a
/// [`for_range_carry`](crate::for_range_carry) or
/// [`while_carry`](crate::while_carry) to keep its states across
/// iterations (only the two states are loop values).
#[derive(Clone, Debug)]
pub struct GenericPipelineBase {
    raw: RawPipeline,
    producer_state: PipelineState,
    consumer_state: PipelineState,
    overrides: Overrides,
}

impl GenericPipelineBase {
    fn new(
        (raw, producer_state, consumer_state): (RawPipeline, PipelineState, PipelineState),
        overrides: Overrides,
    ) -> GenericPipelineBase {
        GenericPipelineBase { raw, producer_state, consumer_state, overrides }
    }

    /// The pipeline (CuTe DSL's `raw_pipeline`).
    pub fn raw_pipeline(&self) -> &RawPipeline {
        &self.raw
    }

    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        self.raw.num_stages()
    }

    /// The producer's state.
    pub fn producer_state(&self) -> PipelineState {
        self.producer_state
    }

    /// The consumer's state.
    pub fn consumer_state(&self) -> PipelineState {
        self.consumer_state
    }

    /// Replace the producer's state (CuTe DSL's assignment to
    /// `producer_state`, as after a manual
    /// [`pipeline_advance_iterator`](super::pipeline_advance_iterator)).
    /// Panics for a state of another ring size.
    #[track_caller]
    pub fn set_producer_state(&mut self, state: PipelineState) {
        self.check_ring(&state, "set_producer_state");
        self.producer_state = state;
    }

    /// Replace the consumer's state (CuTe DSL's assignment to
    /// `consumer_state`). Panics for a state of another ring size.
    #[track_caller]
    pub fn set_consumer_state(&mut self, state: PipelineState) {
        self.check_ring(&state, "set_consumer_state");
        self.consumer_state = state;
    }

    #[track_caller]
    fn check_ring(&self, state: &PipelineState, what: &str) {
        assert!(
            state.stages() == self.raw.num_stages(),
            "cute: experimental::GenericPipelineBase::{what}: the state is for a ring of {} stages, the pipeline has {}",
            state.stages(),
            self.raw.num_stages()
        );
    }

    /// Wait until the producer's stage is empty.
    #[track_caller]
    pub fn producer_acquire(&mut self) -> &mut Self {
        self.producer_state = ops::producer_acquire(&self.raw, &self.producer_state);
        self
    }

    /// Probe the producer's stage once ([`ops::producer_try_acquire`]).
    #[track_caller]
    pub fn producer_try_acquire(&self, token: Option<Boolean>) -> Boolean {
        ops::producer_try_acquire(&self.raw, &self.producer_state, token)
    }

    /// The producer's stage token and index.
    #[track_caller]
    pub fn get_producer_stage(&self) -> (StageToken, Int32) {
        ops::get_pipeline_produce_stage(&self.raw, &self.producer_state)
    }

    /// The consumer's stage token and index.
    #[track_caller]
    pub fn get_consumer_stage(&self) -> (StageToken, Int32) {
        ops::get_pipeline_consume_stage(&self.raw, &self.consumer_state)
    }

    /// The producer's phase (the round of the ring).
    pub fn get_producer_phase(&self) -> Int32 {
        ops::get_pipeline_produce_phase(&self.producer_state)
    }

    /// The consumer's phase.
    pub fn get_consumer_phase(&self) -> Int32 {
        ops::get_pipeline_consume_phase(&self.consumer_state)
    }

    /// Acquire the producer's stage (skipping the wait when `token`, an
    /// earlier [`producer_try_acquire`](Self::producer_try_acquire), is
    /// true) and return its token and index; the state stays at that stage
    /// for the commit.
    #[track_caller]
    pub fn producer_acquire_and_get_stage(&mut self, token: Option<Boolean>) -> (StageToken, Int32) {
        self.producer_state = ops::producer_acquire_with(&self.raw, &self.producer_state, token);
        self.get_producer_stage()
    }

    /// Acquire the producer's stage and return its phase.
    #[track_caller]
    pub fn producer_acquire_and_get_phase(&mut self, token: Option<Boolean>) -> Int32 {
        self.producer_state = ops::producer_acquire_with(&self.raw, &self.producer_state, token);
        self.get_producer_phase()
    }

    /// Commit the producer's stage ([`ops::producer_commit`]).
    #[track_caller]
    pub fn producer_commit(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        let (e, l) = self.overrides.producer_commit.unwrap_or((elect_one_sync, elect_leader_cta));
        ops::producer_commit(&self.raw, &self.producer_state, e, l);
        self
    }

    /// Probe the consumer's stage once.
    #[track_caller]
    pub fn consumer_try_wait(&self, token: Option<Boolean>) -> Boolean {
        ops::consumer_try_wait(&self.raw, &self.consumer_state, token)
    }

    /// Release the consumer's stage ([`ops::consumer_release`]).
    #[track_caller]
    pub fn consumer_release(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        let (e, l) = self.overrides.consumer_release.unwrap_or((elect_one_sync, elect_leader_cta));
        ops::consumer_release(&self.raw, &self.consumer_state, e, l);
        self
    }

    /// Commit the producer's stage and move to the next.
    #[track_caller]
    pub fn producer_commit_and_advance(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        self.producer_commit(elect_one_sync, elect_leader_cta);
        self.producer_state = ops::pipeline_advance_iterator(&self.raw, &self.producer_state);
        self
    }

    /// Wait for the consumer's stage (skipping the wait when `token` is
    /// true) and return its token and index.
    #[track_caller]
    pub fn consumer_wait_and_get_stage(&mut self, token: Option<Boolean>) -> (StageToken, Int32) {
        self.consumer_state = ops::consumer_wait_with(&self.raw, &self.consumer_state, token);
        self.get_consumer_stage()
    }

    /// Wait for the consumer's stage and return its phase.
    #[track_caller]
    pub fn consumer_wait_and_get_phase(&mut self, token: Option<Boolean>) -> Int32 {
        self.consumer_state = ops::consumer_wait_with(&self.raw, &self.consumer_state, token);
        self.get_consumer_phase()
    }

    /// Wait for the consumer's stage.
    #[track_caller]
    pub fn consumer_wait(&mut self) -> &mut Self {
        self.consumer_state = ops::consumer_wait(&self.raw, &self.consumer_state);
        self
    }

    /// Wait for the stage at `state` instead (CuTe DSL's
    /// `consumer_wait(state)`), leaving the consumer's state alone.
    #[track_caller]
    pub fn consumer_wait_at(&mut self, state: &PipelineState) -> &mut Self {
        ops::consumer_wait(&self.raw, state);
        self
    }

    /// Release the consumer's stage (with the class's flags) and move to the
    /// next.
    #[track_caller]
    pub fn consumer_release_and_advance(&mut self) -> &mut Self {
        self.consumer_release(false, false);
        self.consumer_state = ops::pipeline_advance_iterator(&self.raw, &self.consumer_state);
        self
    }

    /// Drain the pipeline from the producer's state ([`ops::pipeline_tail`]).
    #[track_caller]
    pub fn producer_tail(&mut self) -> &mut Self {
        self.producer_state = ops::pipeline_tail(&self.raw, &self.producer_state);
        self
    }

    /// Drain the pipeline from the consumer's state.
    #[track_caller]
    pub fn consumer_tail(&mut self) -> &mut Self {
        self.consumer_state = ops::pipeline_tail(&self.raw, &self.consumer_state);
        self
    }

    /// `state` one stage on, leaving the pipeline alone.
    #[track_caller]
    pub fn increment_state(&self, state: &PipelineState) -> PipelineState {
        ops::pipeline_advance_iterator(&self.raw, state)
    }

    /// A state at `stage_index` and `phase` ([`ops::create_pipeline_state_at`]).
    #[track_caller]
    pub fn create_state_at(&self, stage_index: impl IntoValue<i32>, phase: impl IntoValue<i32>) -> PipelineState {
        ops::create_pipeline_state_at(&self.raw, self.raw.num_stages(), stage_index, phase)
    }
}

impl Carry for GenericPipelineBase {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.producer_state.flatten(out);
        self.consumer_state.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        GenericPipelineBase {
            raw: self.raw.clone(),
            producer_state: self.producer_state.rebuild(values),
            consumer_state: self.consumer_state.rebuild(values),
            overrides: self.overrides,
        }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        self.raw.structure_mismatch(&other.raw)
    }
}

/// A newtype over [`GenericPipelineBase`] that derefs to it and carries it.
macro_rules! pipeline_class {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $name(GenericPipelineBase);

        impl std::ops::Deref for $name {
            type Target = GenericPipelineBase;
            fn deref(&self) -> &GenericPipelineBase {
                &self.0
            }
        }

        impl std::ops::DerefMut for $name {
            fn deref_mut(&mut self) -> &mut GenericPipelineBase {
                &mut self.0
            }
        }

        impl Carry for $name {
            #[track_caller]
            fn flatten(&self, out: &mut Vec<ValueId>) {
                self.0.flatten(out);
            }
            fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
                $name(self.0.rebuild(values))
            }
            fn structure_mismatch(&self, other: &Self) -> Option<String> {
                self.0.structure_mismatch(&other.0)
            }
        }
    };
}

pipeline_class!(
    /// A pipeline between any supported pair of kinds, with the arrival
    /// counts given (CuTe DSL's `GenericPipeline`). Its methods are
    /// [`GenericPipelineBase`]'s.
    GenericPipeline
);

impl GenericPipeline {
    /// A pipeline of `num_stages` stages from `producer` to `consumer` with
    /// the given arrival counts (CuTe DSL's `GenericPipeline.create`; see
    /// [`create_pipeline`](super::create_pipeline)).
    #[track_caller]
    pub fn create(
        producer: impl Into<OperationTypes>,
        consumer: impl Into<OperationTypes>,
        producer_arv_count: impl IntoValue<i32>,
        consumer_arv_count: impl IntoValue<i32>,
        num_stages: usize,
        locale: Option<PipelineLocale>,
    ) -> GenericPipeline {
        let raw = ops::create_pipeline(
            num_stages,
            producer,
            consumer,
            producer_arv_count,
            consumer_arv_count,
            ArrivalMask::None,
            locale,
        );
        GenericPipeline(GenericPipelineBase::new(raw, Overrides::default()))
    }
}

/// Panics unless `op` is a UMMA kind.
#[track_caller]
fn check_umma(op: OperationType) {
    assert!(
        ops::is_umma(op),
        "cute: experimental: {op} is not a UMMA operation (SM100_MMA[_SCALED]_{{1SM,2SM}}_{{SS,TS}})"
    );
}

/// The cluster layout, checked: rank 4 `(V, M, N, L)`.
#[track_caller]
fn vmnk(layout: &Layout, what: &str) -> Layout {
    assert!(layout.rank() == 4, "cute: experimental::{what}: cluster_layout_vmnk {layout} is (V, M, N, L): rank 4");
    layout.clone()
}

/// This CTA's flat `(V, M, N, L)` coordinate in the cluster.
#[track_caller]
fn cta_coord(layout: &Layout) -> Coord {
    layout.get_flat_coord(make_warp_uniform(block_idx_in_cluster()))
}

/// The coordinate of this CTA's pair peer: `V ^ 1` when `V` has size 2.
#[track_caller]
fn peer_coord(layout: &Layout, c: &Coord) -> Coord {
    let v_size = layout.shape().modes()[0].size();
    match c {
        Tree::Branch(modes) if v_size > 1 => {
            let mut modes = modes.clone();
            if let Tree::Leaf(e) = &modes[0] {
                let v: Int32 = IntoValue::<i32>::into_value(e.clone()) ^ 1;
                modes[0] = Tree::Leaf(IndexExpr::from(v));
            }
            Tree::Branch(modes)
        }
        other => other.clone(),
    }
}

use crate::arch::make_warp_uniform;

/// The pair mask: the CTAs sharing this CTA's coordinate outside mode 0
/// (CuTe DSL's `make_layout_image_mask(layout, coord, mode=0)`).
#[track_caller]
fn pair_mask(layout: &Layout) -> Int16 {
    make_layout_image_mask(layout, cta_coord(layout), 0)
}

pipeline_class!(
    /// TMA loads feeding the MMA warp (CuTe DSL's `TMAToUMMAPipeline`; the
    /// [`PipelineTmaUmma`](crate::pipeline::PipelineTmaUmma) class). The
    /// producer's commit always elects one thread, and for a 2SM TMA load
    /// only the pair's leader CTA commits (its MMAs consume the stage); the
    /// release always elects one thread.
    ///
    /// The TMA warp acquires a stage, issues its loads to the stage's
    /// [`get_mbarrier`](super::get_mbarrier) (which add their bytes) and
    /// commits; the MMA warp waits, issues its MMAs and releases with a
    /// commit.
    TMAToUMMAPipeline
);

impl TMAToUMMAPipeline {
    /// A pipeline of `num_stages` stages from `tma_operation_type` (default
    /// `SM90_TMA_LOAD`) to `mma_operation_type` (CuTe DSL's
    /// `TMAToUMMAPipeline.create`). A multicast load takes the
    /// [multicast form](Self::create_with_mask); `SM100_TMA_LOAD_2SM`
    /// arrives on the pair's leader, its release commit going to the pair
    /// (the arrival mask over mode 0 of `cluster_layout_vmnk` at this CTA).
    /// `cluster_layout_vmnk` is required for both, and a 2SM MMA needs the
    /// 2SM load.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        mma_operation_type: OperationType,
        tma_operation_type: Option<OperationType>,
        cluster_layout_vmnk: Option<&Layout>,
    ) -> TMAToUMMAPipeline {
        check_umma(mma_operation_type);
        let tma = tma_operation_type.unwrap_or(OperationType::Sm90TmaLoad);
        let opts = || ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true);
        let count =
            |side, o: ArrivalCountOptions| ops::get_arrival_count(num_stages, tma, mma_operation_type, side, &o);
        if ops::is_multicast(tma) {
            let Some(layout) = cluster_layout_vmnk else {
                panic!(
                    "cute: experimental::TMAToUMMAPipeline::create: a multicast TMA load ({tma}) needs cluster_layout_vmnk"
                )
            };
            return TMAToUMMAPipeline::create_with_mask(num_stages, tma, mma_operation_type, layout);
        }
        let overrides =
            Overrides { producer_commit: Some((true, ops::is_2sm(tma))), consumer_release: Some((true, false)) };
        let raw = match tma {
            OperationType::Sm100TmaLoad2Sm => {
                let Some(layout) = cluster_layout_vmnk else {
                    panic!(
                        "cute: experimental::TMAToUMMAPipeline::create: a 2SM pipeline ({tma}) needs \
                         cluster_layout_vmnk, which says which CTAs pair up"
                    )
                };
                let layout = vmnk(layout, "TMAToUMMAPipeline::create");
                let mask = pair_mask(&layout);
                ops::create_pipeline(
                    num_stages,
                    tma,
                    mma_operation_type,
                    count(Side::Producer, opts().elect_leader_cta(true)),
                    count(Side::Consumer, opts().fan_in(1)),
                    ArrivalMask::Mask(mask),
                    None,
                )
            }
            OperationType::Sm90TmaLoad => {
                assert!(
                    !ops::is_2sm(mma_operation_type),
                    "cute: experimental::TMAToUMMAPipeline::create: a 2SM MMA ({mma_operation_type}) needs the 2SM \
                     load SM100_TMA_LOAD_2SM (and cluster_layout_vmnk)"
                );
                ops::create_pipeline(
                    num_stages,
                    tma,
                    mma_operation_type,
                    count(Side::Producer, opts()),
                    count(Side::Consumer, opts()),
                    ArrivalMask::None,
                    None,
                )
            }
            other => panic!(
                "cute: experimental::TMAToUMMAPipeline::create: {other} is not a TMA load this pipeline takes \
                 (SM90_TMA_LOAD, SM100_TMA_LOAD_2SM or a multicast load)"
            ),
        };
        TMAToUMMAPipeline(GenericPipelineBase::new(raw, overrides))
    }

    /// The multicast form (CuTe DSL's `create_with_mask`): the release
    /// commit goes to every CTA receiving this CTA's A tile (mode 2) or B
    /// tile (mode 1), and those of its pair's peer; the empty barriers
    /// count one commit per multicast participant (`size(M) + size(N) -
    /// 1`).
    #[track_caller]
    pub fn create_with_mask(
        num_stages: usize,
        tma_operation_type: OperationType,
        mma_operation_type: OperationType,
        cluster_layout_vmnk: &Layout,
    ) -> TMAToUMMAPipeline {
        check_umma(mma_operation_type);
        let layout = vmnk(cluster_layout_vmnk, "TMAToUMMAPipeline::create_with_mask");
        let c = cta_coord(&layout);
        let peer = peer_coord(&layout, &c);
        let mask = |coord: &Coord, mode: usize| {
            crate::layout::image_mask::create_tma_multicast_mask(&layout, coord.clone(), mode).to::<i32>()
        };
        let all = mask(&c, 2) | mask(&peer, 2) | mask(&c, 1) | mask(&peer, 1);
        let modes = layout.shape().modes();
        let participants = (modes[2].size() + modes[1].size() - 1) as i32;
        let is_2sm = ops::is_2sm(tma_operation_type);
        let opts = || ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true);
        let count = |side, o: ArrivalCountOptions| {
            ops::get_arrival_count(num_stages, tma_operation_type, mma_operation_type, side, &o)
        };
        let raw = ops::create_pipeline(
            num_stages,
            tma_operation_type,
            mma_operation_type,
            count(Side::Producer, opts().elect_leader_cta(is_2sm)),
            count(Side::Consumer, opts().fan_in(participants)),
            ArrivalMask::Mask(all.to::<i16>()),
            None,
        );
        let overrides = Overrides { producer_commit: Some((true, is_2sm)), consumer_release: Some((true, false)) };
        TMAToUMMAPipeline(GenericPipelineBase::new(raw, overrides))
    }
}

pipeline_class!(
    /// TMA loads feeding thread kinds (`LD_SHARED`, `SM100_COPY_T2R`, ...;
    /// CuTe DSL's `TMAToAsyncPipeline`; the
    /// [`PipelineTmaAsync`](crate::pipeline::PipelineTmaAsync) class). The
    /// producer's commit always elects one thread.
    TMAToAsyncPipeline
);

impl TMAToAsyncPipeline {
    /// A pipeline of `num_stages` stages from `SM90_TMA_LOAD` to `consumer`,
    /// whose releases count `consumer_arv_count` arrivals (CuTe DSL's
    /// `TMAToAsyncPipeline.create`).
    #[track_caller]
    pub fn create(
        num_stages: usize,
        consumer: impl Into<OperationTypes>,
        consumer_arv_count: impl IntoValue<i32>,
    ) -> TMAToAsyncPipeline {
        let consumer = consumer.into();
        let tma = OperationType::Sm90TmaLoad;
        let producer_count = ops::get_arrival_count(
            num_stages,
            tma,
            consumer.clone(),
            Side::Producer,
            &ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true),
        );
        let raw = ops::create_pipeline(
            num_stages,
            tma,
            consumer,
            producer_count,
            consumer_arv_count,
            ArrivalMask::None,
            None,
        );
        let overrides = Overrides { producer_commit: Some((true, false)), consumer_release: None };
        TMAToAsyncPipeline(GenericPipelineBase::new(raw, overrides))
    }
}

pipeline_class!(
    /// Thread kinds feeding the MMA warp (CuTe DSL's `AsyncToUMMAPipeline`;
    /// the [`PipelineAsyncUmma`](crate::pipeline::PipelineAsyncUmma)
    /// class), such as transform warps writing shared memory (`ST_SHARED`,
    /// whose commit fences for the async proxy) or tensor memory
    /// (`SM100_COPY_R2T`). The release always elects one thread.
    AsyncToUMMAPipeline
);

impl AsyncToUMMAPipeline {
    /// A pipeline of `num_stages` stages from `producer` (one kind or a set)
    /// to `mma_operation_type`, whose commits count `producer_arv_count`
    /// arrivals (CuTe DSL's `AsyncToUMMAPipeline.create`).
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer: impl Into<OperationTypes>,
        producer_arv_count: impl IntoValue<i32>,
        mma_operation_type: OperationType,
    ) -> AsyncToUMMAPipeline {
        check_umma(mma_operation_type);
        let producer = producer.into();
        assert!(
            !producer.contains(OperationType::Sm90TmaLoad),
            "cute: experimental::AsyncToUMMAPipeline::create: a TMA producer takes TMAToUMMAPipeline"
        );
        let raw = ops::create_pipeline(
            num_stages,
            producer,
            mma_operation_type,
            producer_arv_count,
            1,
            ArrivalMask::None,
            None,
        );
        let overrides = Overrides { producer_commit: None, consumer_release: Some((true, false)) };
        AsyncToUMMAPipeline(GenericPipelineBase::new(raw, overrides))
    }
}

pipeline_class!(
    /// The MMA warp's accumulators handed to thread kinds (CuTe DSL's
    /// `UMMAtoAsyncPipeline`; the
    /// [`PipelineUmmaAsync`](crate::pipeline::PipelineUmmaAsync) class),
    /// such as epilogue warps reading TMEM (`SM100_COPY_T2R`, whose release
    /// waits for its loads first). The producer's commit always elects one
    /// thread. With a 2SM MMA the commit goes to both CTAs of the pair and
    /// the consumers of both release to the leader, so `consumer_arv_count`
    /// counts the pair's consumer threads.
    UMMAtoAsyncPipeline
);

impl UMMAtoAsyncPipeline {
    /// A pipeline of `num_stages` stages from `mma_operation_type` to
    /// `consumer` (CuTe DSL's `UMMAtoAsyncPipeline.create`); a 2SM MMA
    /// takes the [masked form](Self::create_with_mask) and needs
    /// `cluster_layout_vmnk`.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        consumer: impl Into<OperationTypes>,
        consumer_arv_count: impl IntoValue<i32>,
        mma_operation_type: OperationType,
        cluster_layout_vmnk: Option<&Layout>,
    ) -> UMMAtoAsyncPipeline {
        check_umma(mma_operation_type);
        let consumer = consumer.into();
        assert!(
            !consumer.contains(OperationType::Sm90TmaLoad),
            "cute: experimental::UMMAtoAsyncPipeline::create: a UMMA producer cannot feed a TMA load"
        );
        if ops::is_2sm(mma_operation_type) {
            let Some(layout) = cluster_layout_vmnk else {
                panic!(
                    "cute: experimental::UMMAtoAsyncPipeline::create: a 2SM MMA ({mma_operation_type}) needs \
                     cluster_layout_vmnk, which says which CTAs pair up"
                )
            };
            return UMMAtoAsyncPipeline::create_with_mask(
                num_stages,
                consumer,
                consumer_arv_count,
                mma_operation_type,
                layout,
            );
        }
        let producer_count = ops::get_arrival_count(
            num_stages,
            mma_operation_type,
            consumer.clone(),
            Side::Producer,
            &ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true),
        );
        let raw = ops::create_pipeline(
            num_stages,
            mma_operation_type,
            consumer,
            producer_count,
            consumer_arv_count,
            ArrivalMask::None,
            None,
        );
        let overrides = Overrides { producer_commit: Some((true, false)), consumer_release: None };
        UMMAtoAsyncPipeline(GenericPipelineBase::new(raw, overrides))
    }

    /// The masked form for a 2SM MMA (CuTe DSL's `create_with_mask`): the
    /// commit's mask is this CTA's pair (mode 0 of `cluster_layout_vmnk`).
    #[track_caller]
    pub fn create_with_mask(
        num_stages: usize,
        consumer: impl Into<OperationTypes>,
        consumer_arv_count: impl IntoValue<i32>,
        mma_operation_type: OperationType,
        cluster_layout_vmnk: &Layout,
    ) -> UMMAtoAsyncPipeline {
        check_umma(mma_operation_type);
        let consumer = consumer.into();
        let layout = vmnk(cluster_layout_vmnk, "UMMAtoAsyncPipeline::create_with_mask");
        let mask = pair_mask(&layout);
        let producer_count = ops::get_arrival_count(
            num_stages,
            mma_operation_type,
            consumer.clone(),
            Side::Producer,
            &ArrivalCountOptions::new().num_threads_per_cta(32).elect_one_sync(true).fan_in(1),
        );
        let raw = ops::create_pipeline(
            num_stages,
            mma_operation_type,
            consumer,
            producer_count,
            consumer_arv_count,
            ArrivalMask::Mask(mask),
            None,
        );
        let overrides = Overrides { producer_commit: Some((true, false)), consumer_release: None };
        UMMAtoAsyncPipeline(GenericPipelineBase::new(raw, overrides))
    }
}

/// [`TMAToUMMAPipeline`] whose MMA warp consumes each stage in units
/// (CuTe DSL's `TMAToUMMACircularPipeline`): each stage holds
/// `count_per_stage` units, consumed `count_per_iteration` at a time
/// through a [`CircularBufferPipelineState`] (see it for the exact
/// accounting). Its producer side is [`TMAToUMMAPipeline`]'s; on the
/// consumer side [`consumer_wait`](Self::consumer_wait),
/// [`consumer_release`](Self::consumer_release) and
/// [`consumer_release_and_advance`](Self::consumer_release_and_advance)
/// use the circular operations, and the stage-handing methods
/// ([`get_consumer_stage`](Self::get_consumer_stage),
/// [`consumer_wait_and_get_stage`](Self::consumer_wait_and_get_stage),
/// [`consumer_tail`](Self::consumer_tail)) are not supported: the kernel
/// tracks which stage and unit it reads ([`circular_state`](Self::circular_state)).
#[derive(Clone, Debug)]
pub struct TMAToUMMACircularPipeline {
    base: TMAToUMMAPipeline,
    circular: CircularBufferPipelineState,
}

impl TMAToUMMACircularPipeline {
    /// A circular pipeline of `num_stages` stages of `count_per_stage`
    /// units, consumed `count_per_iteration` at a time (CuTe DSL's
    /// `TMAToUMMACircularPipeline.create`; the other arguments are
    /// [`TMAToUMMAPipeline::create`]'s).
    #[track_caller]
    pub fn create(
        num_stages: usize,
        mma_operation_type: OperationType,
        count_per_stage: usize,
        count_per_iteration: usize,
        tma_operation_type: Option<OperationType>,
        cluster_layout_vmnk: Option<&Layout>,
    ) -> TMAToUMMACircularPipeline {
        let base = TMAToUMMAPipeline::create(num_stages, mma_operation_type, tma_operation_type, cluster_layout_vmnk);
        let circular = ops::create_circular_buffer_pipeline(
            base.raw_pipeline(),
            &base.consumer_state(),
            num_stages,
            count_per_stage,
            count_per_iteration,
        );
        TMAToUMMACircularPipeline { base, circular }
    }

    /// The circular position of the consumer.
    pub fn circular_state(&self) -> CircularBufferPipelineState {
        self.circular
    }

    /// The pipeline.
    pub fn raw_pipeline(&self) -> &RawPipeline {
        self.base.raw_pipeline()
    }

    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        self.base.num_stages()
    }

    /// The units per stage.
    pub fn count_per_stage(&self) -> usize {
        self.circular.count_per_stage()
    }

    /// The units per iteration.
    pub fn count_per_iteration(&self) -> usize {
        self.circular.count_per_iteration()
    }

    /// The producer's state.
    pub fn producer_state(&self) -> PipelineState {
        self.base.producer_state()
    }

    /// [`GenericPipelineBase::producer_acquire`].
    #[track_caller]
    pub fn producer_acquire(&mut self) -> &mut Self {
        self.base.producer_acquire();
        self
    }

    /// [`GenericPipelineBase::producer_try_acquire`].
    #[track_caller]
    pub fn producer_try_acquire(&self, token: Option<Boolean>) -> Boolean {
        self.base.producer_try_acquire(token)
    }

    /// [`GenericPipelineBase::get_producer_stage`].
    #[track_caller]
    pub fn get_producer_stage(&self) -> (StageToken, Int32) {
        self.base.get_producer_stage()
    }

    /// [`GenericPipelineBase::get_producer_phase`].
    pub fn get_producer_phase(&self) -> Int32 {
        self.base.get_producer_phase()
    }

    /// [`GenericPipelineBase::producer_acquire_and_get_stage`].
    #[track_caller]
    pub fn producer_acquire_and_get_stage(&mut self, token: Option<Boolean>) -> (StageToken, Int32) {
        self.base.producer_acquire_and_get_stage(token)
    }

    /// [`GenericPipelineBase::producer_acquire_and_get_phase`].
    #[track_caller]
    pub fn producer_acquire_and_get_phase(&mut self, token: Option<Boolean>) -> Int32 {
        self.base.producer_acquire_and_get_phase(token)
    }

    /// [`GenericPipelineBase::producer_commit`] (one elected thread; the
    /// leader CTA only for a 2SM load).
    #[track_caller]
    pub fn producer_commit(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        self.base.producer_commit(elect_one_sync, elect_leader_cta);
        self
    }

    /// [`GenericPipelineBase::producer_commit_and_advance`].
    #[track_caller]
    pub fn producer_commit_and_advance(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        self.base.producer_commit_and_advance(elect_one_sync, elect_leader_cta);
        self
    }

    /// [`GenericPipelineBase::producer_tail`].
    #[track_caller]
    pub fn producer_tail(&mut self) -> &mut Self {
        self.base.producer_tail();
        self
    }

    /// [`GenericPipelineBase::increment_state`].
    #[track_caller]
    pub fn increment_state(&self, state: &PipelineState) -> PipelineState {
        self.base.increment_state(state)
    }

    /// [`GenericPipelineBase::create_state_at`].
    #[track_caller]
    pub fn create_state_at(&self, stage_index: impl IntoValue<i32>, phase: impl IntoValue<i32>) -> PipelineState {
        self.base.create_state_at(stage_index, phase)
    }

    /// Wait for the stages the next iteration needs
    /// ([`circular_buffer_pipeline_consume`](super::circular_buffer_pipeline_consume)).
    #[track_caller]
    pub fn consumer_wait(&mut self) -> &mut Self {
        ops::circular_buffer_pipeline_consume(self.base.raw_pipeline(), &self.circular, false);
        self
    }

    /// Release the stages the iteration finishes, with one elected thread
    /// ([`circular_buffer_pipeline_consumer_release`](super::circular_buffer_pipeline_consumer_release));
    /// the flags passed are ignored.
    #[track_caller]
    pub fn consumer_release(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        let _ = (elect_one_sync, elect_leader_cta);
        ops::circular_buffer_pipeline_consumer_release(self.base.raw_pipeline(), &self.circular, true, false);
        self
    }

    /// Release, then move the circular position on by one iteration.
    #[track_caller]
    pub fn consumer_release_and_advance(&mut self) -> &mut Self {
        self.consumer_release(true, false);
        self.circular = ops::circular_buffer_pipeline_advance_iterator(self.base.raw_pipeline(), &self.circular);
        self
    }

    /// Not supported: always panics.
    #[track_caller]
    pub fn get_consumer_stage(&self) -> (StageToken, Int32) {
        panic!(
            "cute: experimental::TMAToUMMACircularPipeline: get_consumer_stage is unsupported; the consumer reads units, not stages (see circular_state)"
        )
    }

    /// Not supported: always panics.
    #[track_caller]
    pub fn consumer_wait_and_get_stage(&mut self, token: Option<Boolean>) -> (StageToken, Int32) {
        let _ = token;
        panic!(
            "cute: experimental::TMAToUMMACircularPipeline: consumer_wait_and_get_stage is unsupported; use consumer_wait and circular_state"
        )
    }

    /// Not supported: always panics.
    #[track_caller]
    pub fn consumer_tail(&mut self) -> &mut Self {
        panic!("cute: experimental::TMAToUMMACircularPipeline: consumer_tail is unsupported")
    }
}

impl Carry for TMAToUMMACircularPipeline {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.base.flatten(out);
        self.circular.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        TMAToUMMACircularPipeline { base: self.base.rebuild(values), circular: self.circular.rebuild(values) }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        self.base.structure_mismatch(&other.base).or_else(|| self.circular.structure_mismatch(&other.circular))
    }
}

// ---------------------------------------------------------------------------
// Scheduler pipelines (m11-cute-ext-sched-tmem)
// ---------------------------------------------------------------------------

use crate::arch::{
    MbarrierOptions, cluster_size, elect_one, issue_clc_query, mbarrier_arrive_with, mbarrier_expect_tx_with,
};
use crate::control::for_range;
use crate::ir::MemScopeKind;
use crate::tensor::{AddressSpace, Ptr, Tensor};
use crate::value::Numeric;

/// Access to the [`GenericPipelineBase`] underneath a convenience pipeline,
/// for helpers that take any of them (the accumulator buffering plan's
/// [`producer_acquire_and_get_view`](super::accumulator_buffering::AccumulatorBufferingPlan::producer_acquire_and_get_view),
/// say).
pub trait AsGenericPipeline: Carry + Clone {
    /// The base.
    fn base(&self) -> &GenericPipelineBase;
    /// The base, mutably.
    fn base_mut(&mut self) -> &mut GenericPipelineBase;
}

impl AsGenericPipeline for GenericPipelineBase {
    fn base(&self) -> &GenericPipelineBase {
        self
    }
    fn base_mut(&mut self) -> &mut GenericPipelineBase {
        self
    }
}

macro_rules! as_generic {
    ($($t:ty),*) => {$(
        impl AsGenericPipeline for $t {
            fn base(&self) -> &GenericPipelineBase {
                &self.0
            }
            fn base_mut(&mut self) -> &mut GenericPipelineBase {
                &mut self.0
            }
        }
    )*};
}
as_generic!(
    GenericPipeline,
    TMAToUMMAPipeline,
    TMAToAsyncPipeline,
    AsyncToUMMAPipeline,
    UMMAtoAsyncPipeline,
    GroupedGemmSchedulerPipeline
);

pipeline_class!(
    /// A dedicated scheduler warp handing grouped-GEMM tiles to every other
    /// warp of the CTA (CuTe DSL's `GroupedGemmSchedulerPipeline`): a
    /// `SREG_CTAID -> LD_SHARED` pipeline ([`PipelineClass::Async`](super::PipelineClass::Async)).
    /// The scheduler warp runs the grouped tile scheduler
    /// ([`StaticPersistentGroupTileScheduler`](crate::utils::StaticPersistentGroupTileScheduler)),
    /// acquires a stage, writes the tile's [`GroupedWorkTileInfo`](crate::utils::GroupedWorkTileInfo)
    /// fields into the stage's shared-memory slot and
    /// [`producer_commit_and_advance`](GenericPipelineBase::producer_commit_and_advance)s;
    /// the consumers [`consumer_wait`](GenericPipelineBase::consumer_wait),
    /// read the slot and [`consumer_release`](GenericPipelineBase::consumer_release)
    /// (`elect_one_sync` makes one thread per warp arrive). The arrival
    /// counts are the caller's: the producer's per commit, the consumers'
    /// per release (one per warp with `elect_one_sync`).
    GroupedGemmSchedulerPipeline
);

impl GroupedGemmSchedulerPipeline {
    /// A scheduler pipeline of `num_stages` stages whose full barriers count
    /// `producer_arv_count` arrivals and empty barriers
    /// `consumer_arv_count` (CuTe DSL's `GroupedGemmSchedulerPipeline.create`).
    #[track_caller]
    pub fn create(
        num_stages: usize,
        producer_arv_count: impl IntoValue<i32>,
        consumer_arv_count: impl IntoValue<i32>,
    ) -> GroupedGemmSchedulerPipeline {
        let raw = ops::create_pipeline(
            num_stages,
            OperationType::SregCtaid,
            OperationType::LdShared,
            producer_arv_count,
            consumer_arv_count,
            ArrivalMask::None,
            None,
        );
        GroupedGemmSchedulerPipeline(GenericPipelineBase::new(raw, Overrides::default()))
    }
}

/// The bytes of one cluster launch control response.
const CLC_RESPONSE_BYTES: usize = 16;

/// Cluster launch control responses handed from the scheduler warp to
/// every warp of the cluster (CuTe DSL's `CLCPipeline`): a
/// `SM100_LAUNCH_CONTROL -> LD_SHARED` pipeline
/// ([`PipelineClass::ClcFetchAsync`](super::PipelineClass::ClcFetchAsync))
/// over a shared response buffer of one 16-byte slot per stage.
///
/// The scheduler warp, in the cluster's CTA 0 only, runs per tile:
///
/// 1. [`producer_acquire`](Self::producer_acquire) (CTA 0's empty barrier);
/// 2. [`issue_next`](Self::issue_next): one elected thread adds the
///    response's 16 bytes to the stage's full barrier of every CTA of the
///    cluster (`mbarrier.expect_tx` through `mapa`), then issues the
///    multicast `clusterlaunchcontrol.try_cancel` into the stage's slot
///    ([`get_response_ptr`](Self::get_response_ptr)), whose response lands
///    in every CTA and completes those bytes;
/// 3. [`producer_commit`](Self::producer_commit): one elected thread
///    arrives on the stage's full barrier of every CTA (the full barriers
///    count one arrival: CuTe DSL's producer count
///    `get_arrival_count(.., side=producer, num_threads_per_cta=1)`).
///
/// Consumers in every CTA [`consumer_wait`](GenericPipelineBase::consumer_wait)
/// on their own CTA's full barrier, decode the slot
/// ([`clc_response`](crate::arch::clc_response)) and release to CTA 0's
/// empty barrier (the pipeline's locale is `cluster(all, 0)`, so
/// `consumer_arv_count` counts the consumer threads of the whole cluster;
/// the release fences the async proxy first). Creating it publishes the
/// barriers with a cluster barrier, which every thread of every CTA must
/// reach. [`issue_next`](Self::issue_next) outside an acquire/commit pair
/// is refused when the kernel is built.
///
/// It derefs to [`GenericPipelineBase`] for the consumer side and the
/// states, and implements [`Carry`] (the two states).
#[derive(Clone, Debug)]
pub struct CLCPipeline<T: Numeric> {
    base: GenericPipelineBase,
    response: Tensor<T>,
    acquired: bool,
}

impl<T: Numeric> std::ops::Deref for CLCPipeline<T> {
    type Target = GenericPipelineBase;
    fn deref(&self) -> &GenericPipelineBase {
        &self.base
    }
}

impl<T: Numeric> std::ops::DerefMut for CLCPipeline<T> {
    fn deref_mut(&mut self) -> &mut GenericPipelineBase {
        &mut self.base
    }
}

impl<T: Numeric> CLCPipeline<T> {
    /// A pipeline of `num_stages` stages whose empty barrier (CTA 0's)
    /// counts `consumer_arv_count` releases, over `response_buffer`: a
    /// compact shared-memory buffer of `num_stages` 16-byte slots, the
    /// first 16-byte aligned (CuTe DSL's `CLCPipeline.create`). Panics for
    /// a buffer of another size, in another memory or of an element type
    /// that does not divide 16 bytes.
    #[track_caller]
    pub fn create(
        num_stages: usize,
        consumer_arv_count: impl IntoValue<i32>,
        response_buffer: &Tensor<T>,
    ) -> CLCPipeline<T> {
        let what = "cute: experimental::CLCPipeline::create";
        assert!(
            response_buffer.memspace() == AddressSpace::Smem,
            "{what}: the response buffer is in {} memory; cluster launch control writes its responses to shared memory",
            response_buffer.memspace()
        );
        let bytes = T::DTYPE.bytes();
        assert!(
            !T::DTYPE.is_sub_byte() && CLC_RESPONSE_BYTES.is_multiple_of(bytes),
            "{what}: a {} element does not divide a 16-byte response slot",
            T::DTYPE.cute_name()
        );
        let layout = response_buffer.layout();
        assert!(
            layout.is_static()
                && layout.size() == layout.cosize()
                && layout.size() * bytes == num_stages * CLC_RESPONSE_BYTES,
            "{what}: the response buffer {layout} of {} is not {num_stages} compact 16-byte slots (one per stage)",
            T::DTYPE.cute_name()
        );
        let producer_count = ops::get_arrival_count(
            num_stages,
            OperationType::Sm100LaunchControl,
            OperationType::LdShared,
            Side::Producer,
            &ArrivalCountOptions::new().num_threads_per_cta(1),
        );
        let raw = ops::create_pipeline(
            num_stages,
            OperationType::Sm100LaunchControl,
            OperationType::LdShared,
            producer_count,
            consumer_arv_count,
            ArrivalMask::None,
            Some(PipelineLocale::cluster(super::MbarrierOwner::All, super::MbarrierOwner::Rank(0))),
        );
        CLCPipeline {
            base: GenericPipelineBase::new(raw, Overrides::default()),
            response: response_buffer.clone(),
            acquired: false,
        }
    }

    /// The response buffer.
    pub fn response_buffer(&self) -> &Tensor<T> {
        &self.response
    }

    /// The 16-byte response slot of stage `stage_idx` (CuTe DSL's
    /// `get_response_ptr`).
    #[track_caller]
    pub fn get_response_ptr(&self, stage_idx: impl IntoValue<i32>) -> Ptr<T> {
        let per_slot = (CLC_RESPONSE_BYTES / T::DTYPE.bytes()) as i32;
        self.response.ptr().offset(stage_idx.into_value() * per_slot)
    }

    /// Wait until the producer's stage is empty; [`issue_next`](Self::issue_next)
    /// may follow.
    #[track_caller]
    pub fn producer_acquire(&mut self) -> &mut Self {
        self.base.producer_acquire();
        self.acquired = true;
        self
    }

    /// [`GenericPipelineBase::producer_acquire_and_get_stage`]; [`issue_next`](Self::issue_next)
    /// may follow.
    #[track_caller]
    pub fn producer_acquire_and_get_stage(&mut self, token: Option<Boolean>) -> (StageToken, Int32) {
        let out = self.base.producer_acquire_and_get_stage(token);
        self.acquired = true;
        out
    }

    /// [`GenericPipelineBase::producer_acquire_and_get_phase`]; [`issue_next`](Self::issue_next)
    /// may follow.
    #[track_caller]
    pub fn producer_acquire_and_get_phase(&mut self, token: Option<Boolean>) -> Int32 {
        let out = self.base.producer_acquire_and_get_phase(token);
        self.acquired = true;
        out
    }

    /// Ask for the next work tile (CuTe DSL's `issue_next`): see the type.
    /// Must come between [`producer_acquire`](Self::producer_acquire) and
    /// [`producer_commit`](Self::producer_commit) (panics otherwise).
    #[track_caller]
    pub fn issue_next(&self) {
        let _at = builder::LocationScope::new();
        assert!(
            self.acquired,
            "cute: experimental::CLCPipeline::issue_next: no stage is acquired; call it after producer_acquire and \
             before producer_commit"
        );
        let (token, idx) = self.base.get_producer_stage();
        let mbar = ops::get_mbarrier(&token);
        let response = self.get_response_ptr(idx);
        if_(elect_one(), || {
            for_range(0, cluster_size(), 1, |rank| {
                let opts = MbarrierOptions { peer_cta_rank_in_cluster: Some(rank), ..Default::default() };
                mbarrier_expect_tx_with(&mbar, CLC_RESPONSE_BYTES as i32, opts);
            });
            issue_clc_query(&mbar, &response, true);
        });
    }

    /// Signal that the stage's query is issued (CuTe DSL's `producer_commit`):
    /// one elected thread arrives on the stage's full barrier of every CTA
    /// of the cluster (the flags are accepted for CuTe DSL's signature; one
    /// thread always arrives).
    #[track_caller]
    pub fn producer_commit(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        let _ = (elect_one_sync, elect_leader_cta);
        let _at = builder::LocationScope::new();
        let bar = self.base.raw_pipeline().full_barriers().get_barrier(self.base.producer_state().index());
        if_(elect_one(), || {
            for_range(0, cluster_size(), 1, |rank| {
                let opts = MbarrierOptions {
                    peer_cta_rank_in_cluster: Some(rank),
                    scope: Some(MemScopeKind::Cluster),
                    ..Default::default()
                };
                mbarrier_arrive_with(&bar, 1, opts);
            });
        });
        self.acquired = false;
        self
    }

    /// [`producer_commit`](Self::producer_commit), then the next stage.
    #[track_caller]
    pub fn producer_commit_and_advance(&mut self, elect_one_sync: bool, elect_leader_cta: bool) -> &mut Self {
        self.producer_commit(elect_one_sync, elect_leader_cta);
        let next = self.base.increment_state(&self.base.producer_state());
        self.base.producer_state = next;
        self
    }
}

impl<T: Numeric> Carry for CLCPipeline<T> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.base.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        CLCPipeline { base: self.base.rebuild(values), response: self.response.clone(), acquired: self.acquired }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        self.base.structure_mismatch(&other.base)
    }
}

impl<T: Numeric> AsGenericPipeline for CLCPipeline<T> {
    fn base(&self) -> &GenericPipelineBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut GenericPipelineBase {
        &mut self.base
    }
}

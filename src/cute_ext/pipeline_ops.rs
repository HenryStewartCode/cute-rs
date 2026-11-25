//! cute_ext's low-level pipeline operations (CuTe DSL's `cutlass.cute.experimental.core`
//! pipeline functions): [`create_pipeline`] and the operations on the
//! [`RawPipeline`] it returns, [`get_arrival_count`], and the
//! circular-buffer layer ([`create_circular_buffer_pipeline`] and the
//! `circular_buffer_pipeline_*` operations).
//!
//! In CuTe DSL these lower to closed `lir` operations. cute-rs implements
//! them over the m9/m10 pipeline classes, which share one mbarrier
//! machinery: [`create_pipeline`] maps the pair of operation kinds to a
//! class ([`PipelineClass`]), builds it for its full and empty
//! [`MbarrierArray`]s, and the operations here wait and arrive on those.
//!
//! # Operation kinds
//!
//! Each side of a pipeline is one [`OperationType`] or a set of them (a
//! stage written by several kinds of operation, for instance `ST_SHARED`
//! and `SM100_COPY_R2T`). How a side arrives depends on its kinds, which
//! must all arrive the same way ([`Arrival`]):
//!
//! * TMA loads (`SM90_TMA_LOAD`, the `SM100_TMA_LOAD_2SM` family): one
//!   elected thread per CTA arrives on the full barrier in
//!   [`producer_commit`]; the loads themselves add their bytes
//!   ([`tma_load`](super::tma_load) with its default `update_expect_tx`).
//! * Cluster launch control (`SM100_LAUNCH_CONTROL`): as TMA; the producer
//!   adds the response's 16 bytes with [`mbarrier_expect_tx`] before it
//!   issues the query on the stage's [`get_mbarrier`].
//! * UMMA (`SM100_MMA[_SCALED]_{1SM,2SM}_{SS,TS}`): one elected thread
//!   commits (`tcgen05.commit`, `cta_group::2` for the 2SM kinds,
//!   multicast to the pipeline's arrival mask when it has one), so the
//!   phase completes when the MMAs issued before have finished.
//! * Threads (`ST_SHARED`, `LD_SHARED`, `SM100_COPY_T2R`,
//!   `SM100_COPY_R2T`, `SREG_CTAID`): each calling thread arrives
//!   (`mbarrier.arrive`), or one elected thread per warp with
//!   `elect_one_sync` (after a `__syncwarp`). Each kind adds the fence its
//!   data needs first: `ST_SHARED` feeding UMMA a
//!   `fence.proxy.async.shared::cta`, `SM100_COPY_R2T` a `tcgen05.wait::st`,
//!   `SM100_COPY_T2R` a `tcgen05.wait::ld`, and `LD_SHARED` reading cluster
//!   launch control responses a `fence.proxy.async.shared::cta` (the next
//!   response is written by the async proxy).
//!
//! # States and phases
//!
//! A pipeline state is a [`PipelineState`]: the stage index, the number of
//! stages passed and a phase bit that flips each time the index wraps,
//! starting at 0 for both sides. The phase is the *round* of the ring, so
//! a producer and a consumer at the same position read the same phase
//! ([`get_pipeline_produce_phase`] and [`get_pipeline_consume_phase`]
//! agree, so phase-indexed storage picked by either side matches). A
//! consumer waits for the full barrier's phase of that parity, a producer
//! for the empty barrier's phase of the other parity (the empty phase the
//! previous round's releases completed).
//!
//! # Multi-CTA pipelines
//!
//! An arrival mask (an `Int16` of cluster ranks) is the multicast mask of
//! the pipeline's UMMA commits; only a pipeline with a UMMA side takes
//! one. A pipeline with a 2SM operation (a CTA pair) or a multicast load
//! spans CTAs, and its UMMA commits need a mask. Thread and TMA arrivals go
//! to the CTA that owns the barrier ([`PipelineLocale`]): by default every
//! CTA owns its own barriers ([`MbarrierOwner::All`]), except in a CTA pair,
//! where the side arriving towards a UMMA side goes to the pair's leader
//! (the even cluster rank), whose MMA warp waits or acquires there. A
//! pipeline with a mask or a cluster locale publishes its barriers with a
//! cluster barrier, which every thread of every CTA must reach.

use std::fmt;

use super::core::OperationType;
use crate::arch::{
    MbarrierOptions, block_idx_in_cluster, cluster_size, elect_one, fence_view_async_shared,
    fence_view_async_tmem_load, fence_view_async_tmem_store, make_warp_uniform, mbarrier_arrive, mbarrier_arrive_with,
    mbarrier_expect_tx_with, mbarrier_init_fence, sync_warp,
};
use crate::builder;
use crate::control::{Carry, if_, if_else_yield};
use crate::ir::{MemScopeKind, ValueId};
use crate::pipeline::{
    Agent, BarrierParam, CooperativeGroup, MbarrierArray, PipelineAsync, PipelineAsyncOptions, PipelineAsyncUmma,
    PipelineClcFetchAsync, PipelineClcFetchAsyncOptions, PipelineState, PipelineTmaAsync, PipelineTmaAsyncOptions,
    PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions, agent_sync,
};
use crate::tensor::Ptr;
use crate::value::{Boolean, Int16, Int32, IntoValue, select};

// ---------------------------------------------------------------------------
// Locales
// ---------------------------------------------------------------------------

/// The CTA whose mbarrier a side of a cluster pipeline arrives on (CuTe
/// DSL's `MbarrierOwner`, or a cluster id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MbarrierOwner {
    /// Every CTA arrives on its own barrier.
    All,
    /// The leader (even cluster rank) of each CTA pair.
    CtaPairLeader,
    /// The other CTA of each pair (cluster rank `^ 1`).
    CtaPairPeer,
    /// One CTA of the cluster, by rank.
    Rank(u32),
}

impl MbarrierOwner {
    /// The owner with cluster id `id`; panics for a negative id.
    #[track_caller]
    pub fn rank(id: i64) -> MbarrierOwner {
        assert!(id >= 0, "cute: MbarrierOwner::rank: a cluster id is never negative ({id} given)");
        MbarrierOwner::Rank(u32::try_from(id).expect("cute: MbarrierOwner::rank: the cluster id fits in u32"))
    }

    /// The owner's cluster rank for this CTA, or `None` for its own.
    #[track_caller]
    fn target(self) -> Option<Int32> {
        match self {
            MbarrierOwner::All => None,
            MbarrierOwner::CtaPairLeader => Some(cta_rank() / 2 * 2),
            MbarrierOwner::CtaPairPeer => Some(cta_rank() ^ 1),
            MbarrierOwner::Rank(r) => Some(Int32::from(r as i32)),
        }
    }

    /// Whether this CTA owns the barrier, `None` for every CTA.
    #[track_caller]
    fn is_owner(self) -> Option<Boolean> {
        match self {
            MbarrierOwner::All | MbarrierOwner::CtaPairPeer => None,
            MbarrierOwner::CtaPairLeader => Some((cta_rank() % 2).eq(0)),
            MbarrierOwner::Rank(r) => Some(cta_rank().eq(r as i32)),
        }
    }
}

impl From<u32> for MbarrierOwner {
    fn from(id: u32) -> Self {
        MbarrierOwner::Rank(id)
    }
}

impl fmt::Display for MbarrierOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MbarrierOwner::All => f.write_str("all"),
            MbarrierOwner::CtaPairLeader => f.write_str("cta_pair_leader"),
            MbarrierOwner::CtaPairPeer => f.write_str("cta_pair_peer"),
            MbarrierOwner::Rank(r) => write!(f, "{r}"),
        }
    }
}

/// Where a pipeline's barriers live and are arrived on (CuTe DSL's
/// `PipelineLocale`): [`cta`](Self::cta), each CTA on its own, or
/// [`cluster`](Self::cluster) with the owners of the full and the empty
/// barriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineLocale {
    Cta,
    Cluster { full: MbarrierOwner, empty: MbarrierOwner },
}

impl PipelineLocale {
    /// The CTA locale.
    pub fn cta() -> PipelineLocale {
        PipelineLocale::Cta
    }

    /// A cluster locale whose full barriers `full` owns and empty barriers
    /// `empty` owns.
    pub fn cluster(full: impl Into<MbarrierOwner>, empty: impl Into<MbarrierOwner>) -> PipelineLocale {
        PipelineLocale::Cluster { full: full.into(), empty: empty.into() }
    }
}

impl fmt::Display for PipelineLocale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineLocale::Cta => f.write_str("cta"),
            PipelineLocale::Cluster { full, empty } => write!(f, "cluster<{full}, {empty}>"),
        }
    }
}

// ---------------------------------------------------------------------------
// Operation kinds and classes
// ---------------------------------------------------------------------------

/// One side's operation kinds: one [`OperationType`] or a non-empty set of
/// them without duplicates (the order does not matter; they are kept
/// sorted).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OperationTypes(Vec<OperationType>);

impl OperationTypes {
    /// The kinds, in [`OperationType::ALL`] order.
    pub fn ops(&self) -> &[OperationType] {
        &self.0
    }

    /// Whether `op` is one of the kinds.
    pub fn contains(&self, op: OperationType) -> bool {
        self.0.contains(&op)
    }

    #[track_caller]
    fn checked(ops: Vec<OperationType>, what: &str) -> OperationTypes {
        assert!(!ops.is_empty(), "cute: experimental: the {what} operation set is empty; name at least one kind");
        let mut sorted = ops.clone();
        sorted.sort_by_key(|op| OperationType::ALL.iter().position(|o| o == op));
        for w in sorted.windows(2) {
            assert!(w[0] != w[1], "cute: experimental: {} appears twice in the {what} operation set", w[0]);
        }
        OperationTypes(sorted)
    }
}

impl From<OperationType> for OperationTypes {
    fn from(op: OperationType) -> Self {
        OperationTypes(vec![op])
    }
}

impl From<Vec<OperationType>> for OperationTypes {
    fn from(ops: Vec<OperationType>) -> Self {
        OperationTypes(ops)
    }
}

impl From<&[OperationType]> for OperationTypes {
    fn from(ops: &[OperationType]) -> Self {
        OperationTypes(ops.to_vec())
    }
}

impl<const N: usize> From<[OperationType; N]> for OperationTypes {
    fn from(ops: [OperationType; N]) -> Self {
        OperationTypes(ops.to_vec())
    }
}

impl fmt::Display for OperationTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let [op] = self.0.as_slice() {
            return write!(f, "{op}");
        }
        let names: Vec<&str> = self.0.iter().map(|op| op.name()).collect();
        write!(f, "[{}]", names.join(", "))
    }
}

/// How a side's operations arrive on their barrier (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arrival {
    /// TMA loads: one elected thread per CTA; the loads add their bytes.
    Tma,
    /// A cluster launch control query: as TMA.
    Clc,
    /// `tcgen05.commit` from one elected thread.
    Umma,
    /// Ordinary threads.
    Threads,
}

/// How `op` arrives, or `None` for a kind that is not a pipeline side.
fn arrival_of(op: OperationType) -> Option<Arrival> {
    use OperationType::*;
    Some(match op {
        Sm90TmaLoad
        | Sm90TmaLoadMulticast
        | Sm100TmaLoad2Sm
        | Sm100TmaLoad2SmMulticast
        | Sm100TmaLoad2SmIm2col
        | Sm100TmaLoad2SmIm2colMulticast => Arrival::Tma,
        Sm100Mma1SmSs | Sm100Mma1SmTs | Sm100Mma2SmSs | Sm100Mma2SmTs | Sm100MmaScaled1SmSs | Sm100MmaScaled1SmTs
        | Sm100MmaScaled2SmSs | Sm100MmaScaled2SmTs => Arrival::Umma,
        Sm100CopyT2r | Sm100CopyR2t | StShared | LdShared | SregCtaid => Arrival::Threads,
        Sm100LaunchControl => Arrival::Clc,
        ReduceDownShuffle | ReduceSwapShuffle => return None,
    })
}

/// Whether `op` is a UMMA kind.
pub(super) fn is_umma(op: OperationType) -> bool {
    arrival_of(op) == Some(Arrival::Umma)
}

/// Whether `op` runs on a CTA pair (a 2SM MMA or TMA load).
pub(super) fn is_2sm(op: OperationType) -> bool {
    use OperationType::*;
    matches!(
        op,
        Sm100Mma2SmSs
            | Sm100Mma2SmTs
            | Sm100MmaScaled2SmSs
            | Sm100MmaScaled2SmTs
            | Sm100TmaLoad2Sm
            | Sm100TmaLoad2SmMulticast
            | Sm100TmaLoad2SmIm2col
            | Sm100TmaLoad2SmIm2colMulticast
    )
}

/// Whether `op` is a multicast TMA load.
pub(super) fn is_multicast(op: OperationType) -> bool {
    use OperationType::*;
    matches!(op, Sm90TmaLoadMulticast | Sm100TmaLoad2SmMulticast | Sm100TmaLoad2SmIm2colMulticast)
}

/// How every kind of `ops` arrives; panics when they differ or one is not a
/// pipeline kind.
#[track_caller]
fn side_arrival(ops: &OperationTypes, what: &str) -> Arrival {
    let mut found: Option<Arrival> = None;
    for &op in ops.ops() {
        let Some(a) = arrival_of(op) else {
            panic!("cute: experimental: {op} is a warp reduction, not an operation that can be a pipeline's {what}")
        };
        if let Some(prev) = found {
            assert!(
                prev == a,
                "cute: experimental: the {what} kinds {ops} arrive in different ways ({prev:?} and {a:?}); a pipeline \
                 side's kinds must share one (several thread kinds, say)"
            );
        }
        found = Some(a);
    }
    found.expect("a non-empty operation set")
}

/// The m9/m10 pipeline class a pair of sides maps to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineClass {
    /// TMA -> UMMA: [`PipelineTmaUmma`].
    TmaUmma,
    /// TMA -> threads: [`PipelineTmaAsync`].
    TmaAsync,
    /// Threads -> UMMA: [`PipelineAsyncUmma`].
    AsyncUmma,
    /// UMMA -> threads: [`PipelineUmmaAsync`].
    UmmaAsync,
    /// Threads -> threads (`SREG_CTAID -> LD_SHARED` included):
    /// [`PipelineAsync`].
    Async,
    /// `SM100_LAUNCH_CONTROL -> LD_SHARED`: [`PipelineClcFetchAsync`].
    ClcFetchAsync,
}

/// The class of a pair, with each side's arrival; panics for a pair no
/// class supports.
#[track_caller]
fn classify(producer: &OperationTypes, consumer: &OperationTypes) -> (PipelineClass, Arrival, Arrival) {
    let (p, c) = (side_arrival(producer, "producer"), side_arrival(consumer, "consumer"));
    let class = match (p, c) {
        (Arrival::Tma, Arrival::Umma) => PipelineClass::TmaUmma,
        (Arrival::Tma, Arrival::Threads) => PipelineClass::TmaAsync,
        (Arrival::Threads, Arrival::Umma) => PipelineClass::AsyncUmma,
        (Arrival::Umma, Arrival::Threads) => PipelineClass::UmmaAsync,
        (Arrival::Threads, Arrival::Threads) => PipelineClass::Async,
        (Arrival::Clc, Arrival::Threads) if consumer.ops() == [OperationType::LdShared] => PipelineClass::ClcFetchAsync,
        _ => panic!(
            "cute: experimental: no pipeline runs from {producer} to {consumer}: the supported pairs are TMA -> UMMA, \
             TMA -> thread kinds, thread kinds -> UMMA, UMMA -> thread kinds, thread kinds -> thread kinds and \
             SM100_LAUNCH_CONTROL -> LD_SHARED"
        ),
    };
    (class, p, c)
}

/// The class instance a [`RawPipeline`] is built on.
#[derive(Clone, Debug)]
pub enum ClassPipeline {
    TmaUmma(PipelineTmaUmma),
    TmaAsync(PipelineTmaAsync),
    AsyncUmma(PipelineAsyncUmma),
    UmmaAsync(PipelineUmmaAsync),
    Async(PipelineAsync),
    ClcFetchAsync(PipelineClcFetchAsync),
}

impl ClassPipeline {
    /// The full barriers.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        match self {
            ClassPipeline::TmaUmma(p) => p.sync_object_full(),
            ClassPipeline::TmaAsync(p) => p.sync_object_full(),
            ClassPipeline::AsyncUmma(p) => p.sync_object_full(),
            ClassPipeline::UmmaAsync(p) => p.sync_object_full(),
            ClassPipeline::Async(p) => p.sync_object_full(),
            ClassPipeline::ClcFetchAsync(p) => p.sync_object_full(),
        }
    }

    /// The empty barriers.
    pub fn sync_object_empty(&self) -> &MbarrierArray {
        match self {
            ClassPipeline::TmaUmma(p) => p.sync_object_empty(),
            ClassPipeline::TmaAsync(p) => p.sync_object_empty(),
            ClassPipeline::AsyncUmma(p) => p.sync_object_empty(),
            ClassPipeline::UmmaAsync(p) => p.sync_object_empty(),
            ClassPipeline::Async(p) => p.sync_object_empty(),
            ClassPipeline::ClcFetchAsync(p) => p.sync_object_empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// The pipeline value
// ---------------------------------------------------------------------------

/// A pipeline made by [`create_pipeline`] (CuTe DSL's `!lir.pipeline`
/// value): the class instance and its barriers, the two sides' kinds and
/// how multi-CTA arrivals are routed. Its values are made once, before any
/// loop, so carrying it through a loop ([`Carry`]) carries nothing.
#[derive(Clone, Debug)]
pub struct RawPipeline {
    stages: usize,
    producer: OperationTypes,
    consumer: OperationTypes,
    class: PipelineClass,
    producer_arrival: Arrival,
    consumer_arrival: Arrival,
    inner: ClassPipeline,
    arrival_mask: Option<Int16>,
    locale: Option<PipelineLocale>,
    full_owner: MbarrierOwner,
    empty_owner: MbarrierOwner,
}

impl RawPipeline {
    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        self.stages
    }

    /// The producer's kinds.
    pub fn producer(&self) -> &OperationTypes {
        &self.producer
    }

    /// The consumer's kinds.
    pub fn consumer(&self) -> &OperationTypes {
        &self.consumer
    }

    /// The class the pair maps to.
    pub fn class(&self) -> PipelineClass {
        self.class
    }

    /// The class instance whose barriers the pipeline uses.
    pub fn class_pipeline(&self) -> &ClassPipeline {
        &self.inner
    }

    /// The full barriers (the producer arrives, the consumer waits).
    pub fn full_barriers(&self) -> &MbarrierArray {
        self.inner.sync_object_full()
    }

    /// The empty barriers (the consumer arrives, the producer waits).
    pub fn empty_barriers(&self) -> &MbarrierArray {
        self.inner.sync_object_empty()
    }

    /// The arrival mask of the UMMA commits, if any.
    pub fn arrival_mask(&self) -> Option<Int16> {
        self.arrival_mask
    }

    /// The locale given at creation.
    pub fn locale(&self) -> Option<PipelineLocale> {
        self.locale
    }

    /// The owners of the full and the empty barriers.
    pub fn owners(&self) -> (MbarrierOwner, MbarrierOwner) {
        (self.full_owner, self.empty_owner)
    }

    /// The `!lir.pipeline<...>` type CuTe DSL gives the pipeline (for
    /// messages and comparisons).
    pub fn type_string(&self) -> String {
        type_string(self.stages, &self.producer, &self.consumer, self.locale.as_ref())
    }

    #[track_caller]
    fn check_state(&self, state: &PipelineState, what: &str) {
        assert!(
            state.stages() == self.stages,
            "cute: experimental::{what}: the state is for a ring of {} stages, the pipeline has {}",
            state.stages(),
            self.stages
        );
    }

    /// The CTA group of the UMMA side's commits.
    fn cta_group(ops: &OperationTypes) -> crate::nvgpu::tcgen05::CtaGroup {
        if ops.ops().iter().any(|&op| is_2sm(op)) {
            crate::nvgpu::tcgen05::CtaGroup::TWO
        } else {
            crate::nvgpu::tcgen05::CtaGroup::ONE
        }
    }

    /// One arrival of `arrival` kind by the calling threads on `bar` (owned
    /// by `owner`), the threads' fences first.
    #[track_caller]
    fn arrive(&self, side: Side, bar: Ptr<u64>, elect_one_sync: bool) {
        let (arrival, ops, owner) = match side {
            Side::Producer => (self.producer_arrival, &self.producer, self.full_owner),
            Side::Consumer => (self.consumer_arrival, &self.consumer, self.empty_owner),
        };
        match arrival {
            Arrival::Tma | Arrival::Clc => if_(elect_one(), || plain_arrive(&bar, owner)),
            Arrival::Umma => {
                let group = RawPipeline::cta_group(ops);
                let mask = self.arrival_mask;
                if_(elect_one(), || crate::nvgpu::tcgen05::commit(&bar, mask, group));
            }
            Arrival::Threads => {
                for &op in ops.ops() {
                    match (side, op) {
                        (Side::Producer, OperationType::StShared) if self.consumer_arrival == Arrival::Umma => {
                            fence_view_async_shared()
                        }
                        (_, OperationType::Sm100CopyR2t) => fence_view_async_tmem_store(),
                        (_, OperationType::Sm100CopyT2r) => fence_view_async_tmem_load(),
                        (Side::Consumer, OperationType::LdShared) if self.producer_arrival == Arrival::Clc => {
                            fence_view_async_shared()
                        }
                        _ => {}
                    }
                }
                if elect_one_sync {
                    sync_warp();
                    if_(elect_one(), || plain_arrive(&bar, owner));
                } else {
                    plain_arrive(&bar, owner);
                }
            }
        }
    }
}

/// A side of a pipeline (CuTe DSL's `side="producer"` / `"consumer"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Producer,
    Consumer,
}

impl Carry for RawPipeline {
    fn flatten(&self, _: &mut Vec<ValueId>) {}

    fn rebuild(&self, _: &mut dyn Iterator<Item = ValueId>) -> Self {
        self.clone()
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let (a, b) = (self.type_string(), other.type_string());
        (a != b).then(|| format!("a pipeline {a} became {b}"))
    }
}

fn type_string(
    stages: usize,
    producer: &OperationTypes,
    consumer: &OperationTypes,
    locale: Option<&PipelineLocale>,
) -> String {
    match locale {
        None => format!("!lir.pipeline<{stages}, {producer} -> {consumer}>"),
        Some(l) => format!("!lir.pipeline<{stages}, {producer} -> {consumer}, {l}>"),
    }
}

/// This CTA's rank in the cluster, warp-uniform.
#[track_caller]
fn cta_rank() -> Int32 {
    make_warp_uniform(block_idx_in_cluster())
}

/// Whether this CTA leads its pair (even cluster rank).
#[track_caller]
fn is_leader_cta() -> Boolean {
    (cta_rank() % 2).eq(0)
}

/// `f` on the leader CTA of each pair when `leader_only`, else everywhere.
#[track_caller]
fn on_leader(leader_only: bool, f: impl FnOnce()) {
    if leader_only { if_(is_leader_cta(), f) } else { f() }
}

/// An ordinary arrival on `bar`, at the owner's CTA (a cluster-scope
/// release through `mapa` when that is not this CTA).
#[track_caller]
fn plain_arrive(bar: &Ptr<u64>, owner: MbarrierOwner) {
    match owner.target() {
        None => mbarrier_arrive(bar),
        Some(rank) => mbarrier_arrive_with(
            bar,
            1,
            MbarrierOptions {
                peer_cta_rank_in_cluster: Some(rank),
                scope: Some(MemScopeKind::Cluster),
                ..Default::default()
            },
        ),
    }
}

/// Wait unless the skip-wait token is true.
#[track_caller]
fn wait_unless(token: Option<Boolean>, wait: impl FnOnce()) {
    match token.map(|t| (t, t.as_constant())) {
        None | Some((_, Some(false))) => wait(),
        Some((_, Some(true))) => {}
        Some((t, None)) => if_(!t, wait),
    }
}

/// Probe unless the skip-wait token is true (then true without probing).
#[track_caller]
fn probe_unless(token: Option<Boolean>, probe: impl FnOnce() -> Boolean) -> Boolean {
    match token.map(|t| (t, t.as_constant())) {
        None | Some((_, Some(false))) => probe(),
        Some((_, Some(true))) => Boolean::from(true),
        Some((t, None)) => if_else_yield(t, || Boolean::from(true), probe),
    }
}

/// Whether mbarrier waits spin on `mbarrier.test_wait` (a target below
/// sm_90), so the init sync has no `fence.mbarrier_init`.
fn portable_waits() -> bool {
    crate::target_arch().is_some_and(|arch| arch.sm() < 90)
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

/// [`create_pipeline`]'s `arrival_mask` argument: none, a mask, or CuTe
/// DSL's legacy `multicast` flag (`false` is no mask; `true` is refused,
/// as in CuTe DSL).
#[derive(Clone, Copy, Debug)]
pub enum ArrivalMask {
    None,
    Mask(Int16),
    LegacyMulticast(bool),
}

impl From<Int16> for ArrivalMask {
    fn from(m: Int16) -> Self {
        ArrivalMask::Mask(m)
    }
}

impl From<Option<Int16>> for ArrivalMask {
    fn from(m: Option<Int16>) -> Self {
        m.map_or(ArrivalMask::None, ArrivalMask::Mask)
    }
}

impl From<bool> for ArrivalMask {
    fn from(b: bool) -> Self {
        ArrivalMask::LegacyMulticast(b)
    }
}

/// The default owners of the full and the empty barriers (see the module
/// docs): towards a UMMA side of a CTA pair, the pair's leader.
fn default_owners(pair: bool, p: Arrival, c: Arrival) -> (MbarrierOwner, MbarrierOwner) {
    let full = if pair && c == Arrival::Umma { MbarrierOwner::CtaPairLeader } else { MbarrierOwner::All };
    let empty = if pair && p == Arrival::Umma { MbarrierOwner::CtaPairLeader } else { MbarrierOwner::All };
    (full, empty)
}

/// Whether the pipeline's operations span a CTA pair, and whether they
/// span CTAs at all (a pair or a multicast load).
fn spans(producer: &OperationTypes, consumer: &OperationTypes) -> (bool, bool) {
    let all = || producer.ops().iter().chain(consumer.ops());
    let pair = all().any(|&op| is_2sm(op));
    (pair, pair || all().any(|&op| is_multicast(op)))
}

#[track_caller]
fn check_count(count: Int32, what: &str) {
    if let Some(n) = count.as_constant() {
        assert!(n >= 1, "cute: experimental::create_pipeline: the {what} arrival count must be at least 1, not {n}");
    }
}

/// A pipeline of `stage` stages from `producer` to `consumer` (CuTe DSL's
/// `create_pipeline`), whose full barriers count `producer_arv_count`
/// arrivals and empty barriers `consumer_arv_count` (use
/// [`get_arrival_count`] for them), and the producer's and the consumer's
/// starting states (stage 0, phase 0).
///
/// `arrival_mask` is the multicast mask of the UMMA side's commits
/// ([`ArrivalMask`]; [`create_pipeline_with_mask`] is the older spelling).
/// `locale` names the owners of the barriers. The barriers are
/// initialised from warp 0 and published to the CTA (to the cluster, with
/// a mask or a cluster locale), so every thread must reach the call.
///
/// Panics for 0 stages, an unsupported pair, a static count below 1, the
/// legacy `multicast = true`, a mask on a pipeline without a UMMA side, and
/// a pipeline spanning CTAs whose UMMA commits have no mask.
#[track_caller]
pub fn create_pipeline(
    stage: usize,
    producer: impl Into<OperationTypes>,
    consumer: impl Into<OperationTypes>,
    producer_arv_count: impl IntoValue<i32>,
    consumer_arv_count: impl IntoValue<i32>,
    arrival_mask: impl Into<ArrivalMask>,
    locale: Option<PipelineLocale>,
) -> (RawPipeline, PipelineState, PipelineState) {
    let _at = builder::LocationScope::new();
    assert!(stage > 0, "cute: experimental::create_pipeline: a pipeline needs at least one stage");
    let producer = OperationTypes::checked(producer.into().0, "producer");
    let consumer = OperationTypes::checked(consumer.into().0, "consumer");
    let (class, p, c) = classify(&producer, &consumer);
    let arrival_mask = match arrival_mask.into() {
        ArrivalMask::None | ArrivalMask::LegacyMulticast(false) => None,
        ArrivalMask::Mask(m) => Some(m),
        ArrivalMask::LegacyMulticast(true) => panic!(
            "cute: experimental::create_pipeline: the multicast = true form is gone (it built wrong barriers); pass \
             the arrival mask itself, or use create_pipeline_with_mask"
        ),
    };
    let has_umma = p == Arrival::Umma || c == Arrival::Umma;
    assert!(
        arrival_mask.is_none() || has_umma,
        "cute: experimental::create_pipeline: an arrival mask routes tcgen05.commit arrivals, and {producer} -> \
         {consumer} has no UMMA side"
    );
    let (pair, multi) = spans(&producer, &consumer);
    assert!(
        !(multi && has_umma && arrival_mask.is_none()),
        "cute: experimental::create_pipeline: {producer} -> {consumer} spans several CTAs, so its UMMA commits need \
         an arrival mask (the CTAs whose barriers they arrive on)"
    );
    let (full_owner, empty_owner) = match locale {
        Some(PipelineLocale::Cluster { full, empty }) => (full, empty),
        _ => default_owners(pair, p, c),
    };
    let (pc, cc) = (producer_arv_count.into_value(), consumer_arv_count.into_value());
    check_count(pc, "producer");
    check_count(cc, "consumer");
    let pg = CooperativeGroup::new(Agent::Thread, BarrierParam::Dynamic(pc));
    let cg = CooperativeGroup::new(Agent::Thread, BarrierParam::Dynamic(cc));
    let inner = match class {
        PipelineClass::TmaUmma => ClassPipeline::TmaUmma(PipelineTmaUmma::create(
            stage,
            pg,
            cg,
            0,
            PipelineTmaUmmaOptions { defer_sync: true, ..Default::default() },
        )),
        PipelineClass::TmaAsync => ClassPipeline::TmaAsync(PipelineTmaAsync::create(
            stage,
            pg,
            cg,
            0,
            PipelineTmaAsyncOptions { defer_sync: true, ..Default::default() },
        )),
        PipelineClass::AsyncUmma => ClassPipeline::AsyncUmma(PipelineAsyncUmma::create(
            stage,
            pg,
            cg,
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        )),
        PipelineClass::UmmaAsync => ClassPipeline::UmmaAsync(PipelineUmmaAsync::create(
            stage,
            pg,
            cg,
            PipelineUmmaOptions { defer_sync: true, ..Default::default() },
        )),
        PipelineClass::Async => ClassPipeline::Async(PipelineAsync::create(
            stage,
            pg,
            cg,
            PipelineAsyncOptions { defer_sync: true, ..Default::default() },
        )),
        PipelineClass::ClcFetchAsync => ClassPipeline::ClcFetchAsync(PipelineClcFetchAsync::create(
            stage,
            pg,
            cg,
            16,
            PipelineClcFetchAsyncOptions { defer_sync: true, ..Default::default() },
        )),
    };
    // Publish the barriers: to the cluster when arrivals cross CTAs.
    let cluster = arrival_mask.is_some() || matches!(locale, Some(PipelineLocale::Cluster { .. }));
    if !portable_waits() {
        mbarrier_init_fence();
    }
    if cluster {
        agent_sync(Agent::ThreadBlockCluster, true);
    } else {
        agent_sync(Agent::ThreadBlock, false);
    }
    let pipe = RawPipeline {
        stages: stage,
        producer,
        consumer,
        class,
        producer_arrival: p,
        consumer_arrival: c,
        inner,
        arrival_mask,
        locale,
        full_owner,
        empty_owner,
    };
    let start = || PipelineState::new(stage, Int32::from(0), Int32::from(0), Int32::from(0));
    (pipe, start(), start())
}

/// [`create_pipeline`] with an arrival mask (CuTe DSL's
/// `create_pipeline_with_mask`, the older spelling).
#[track_caller]
pub fn create_pipeline_with_mask(
    stage: usize,
    producer: impl Into<OperationTypes>,
    consumer: impl Into<OperationTypes>,
    producer_arv_count: impl IntoValue<i32>,
    consumer_arv_count: impl IntoValue<i32>,
    arrival_mask: impl IntoValue<i16>,
    locale: Option<PipelineLocale>,
) -> (RawPipeline, PipelineState, PipelineState) {
    create_pipeline(
        stage,
        producer,
        consumer,
        producer_arv_count,
        consumer_arv_count,
        ArrivalMask::Mask(arrival_mask.into_value()),
        locale,
    )
}

// ---------------------------------------------------------------------------
// Arrival counts
// ---------------------------------------------------------------------------

/// The optional arguments of [`get_arrival_count`] (CuTe DSL's keyword
/// arguments): the threads per CTA that issue the release (default 1), a
/// subset of the side's kinds one release covers, whether one thread per
/// warp (`elect_one_sync`) or only the pair's leader CTA
/// (`elect_leader_cta`) arrives, the CTAs converging on each barrier
/// (`fan_in`), and the pipeline's locale.
#[derive(Clone, Debug, Default)]
pub struct ArrivalCountOptions {
    pub num_threads_per_cta: Option<Int32>,
    pub op_subset: Option<Vec<OperationType>>,
    pub elect_one_sync: bool,
    pub elect_leader_cta: bool,
    pub fan_in: Option<Int32>,
    pub locale: Option<PipelineLocale>,
}

impl ArrivalCountOptions {
    pub fn new() -> ArrivalCountOptions {
        ArrivalCountOptions::default()
    }
    pub fn num_threads_per_cta(mut self, n: impl IntoValue<i32>) -> Self {
        self.num_threads_per_cta = Some(n.into_value());
        self
    }
    pub fn op_subset(mut self, ops: &[OperationType]) -> Self {
        self.op_subset = Some(ops.to_vec());
        self
    }
    pub fn elect_one_sync(mut self, yes: bool) -> Self {
        self.elect_one_sync = yes;
        self
    }
    pub fn elect_leader_cta(mut self, yes: bool) -> Self {
        self.elect_leader_cta = yes;
        self
    }
    pub fn fan_in(mut self, n: impl IntoValue<i32>) -> Self {
        self.fan_in = Some(n.into_value());
        self
    }
    pub fn locale(mut self, locale: PipelineLocale) -> Self {
        self.locale = Some(locale);
        self
    }
}

/// A count: static when it folds.
fn count(v: Int32) -> BarrierParam {
    match v.as_constant() {
        Some(n) => BarrierParam::Static(n.into()),
        None => BarrierParam::Dynamic(v),
    }
}

#[track_caller]
fn mul(a: BarrierParam, b: BarrierParam) -> BarrierParam {
    match (a.as_static(), b.as_static()) {
        (Some(x), Some(y)) => BarrierParam::Static(x * y),
        _ => BarrierParam::Dynamic(a.value() * b.value()),
    }
}

/// The arrivals one logical release of `side` adds to its barrier's phase
/// (CuTe DSL's `get_arrival_count`), for the pipeline `create_pipeline`
/// makes from the same `(stage, producer, consumer)` and locale:
/// `dedup_arrives * arrives_per_cta * fan_in`, where
///
/// * `dedup_arrives` is 1: a side's kinds share one way of arriving (or
///   the pair is refused), so the kinds of one release arrive once;
/// * `arrives_per_cta`: for thread kinds `num_threads_per_cta`, or one per
///   warp with `elect_one_sync` (`num_threads_per_cta` must then be a
///   multiple of 32); for TMA, cluster launch control and UMMA one elected
///   thread per CTA;
/// * `fan_in`: the CTAs whose releases land on one barrier: `fan_in` when
///   given, which a UMMA side of a pipeline spanning CTAs (its commits are
///   multicast to the mask) needs; otherwise from the barrier's owner: 1
///   for every CTA on its own, 2 for a pair's leader (1 with
///   `elect_leader_cta`, only the leader arriving), the cluster's size for
///   one CTA (half of it with `elect_leader_cta`).
///
/// Call it once per logical release landing in one phase and add the
/// results. The count is static when its inputs are.
#[track_caller]
pub fn get_arrival_count(
    stage: usize,
    producer: impl Into<OperationTypes>,
    consumer: impl Into<OperationTypes>,
    side: Side,
    options: &ArrivalCountOptions,
) -> Int32 {
    let _at = builder::LocationScope::new();
    assert!(stage > 0, "cute: experimental::get_arrival_count: a pipeline needs at least one stage");
    let producer = OperationTypes::checked(producer.into().0, "producer");
    let consumer = OperationTypes::checked(consumer.into().0, "consumer");
    let (_, p, c) = classify(&producer, &consumer);
    let (ops, arrival) = match side {
        Side::Producer => (&producer, p),
        Side::Consumer => (&consumer, c),
    };
    if let Some(subset) = &options.op_subset {
        let subset = OperationTypes::checked(subset.clone(), "op_subset");
        for &op in subset.ops() {
            assert!(
                ops.contains(op),
                "cute: experimental::get_arrival_count: op_subset names {op}, which is not one of the {side:?} kinds \
                 {ops}"
            );
        }
    }
    let threads = count(options.num_threads_per_cta.unwrap_or_else(|| Int32::from(1)));
    if let Some(n) = threads.as_static() {
        assert!(n >= 1, "cute: experimental::get_arrival_count: num_threads_per_cta must be at least 1, not {n}");
        assert!(
            !options.elect_one_sync || n % 32 == 0,
            "cute: experimental::get_arrival_count: elect_one_sync elects one thread per warp, so num_threads_per_cta \
             must be a multiple of 32, not {n}"
        );
    }
    let per_cta = match arrival {
        Arrival::Threads if options.elect_one_sync => match threads.as_static() {
            Some(n) => BarrierParam::Static(n / 32),
            None => BarrierParam::Dynamic(threads.value() / 32),
        },
        Arrival::Threads => threads,
        Arrival::Tma | Arrival::Clc | Arrival::Umma => BarrierParam::Static(1),
    };
    let (pair, multi) = spans(&producer, &consumer);
    let fan_in = match options.fan_in {
        Some(f) => {
            let f = count(f);
            if let Some(n) = f.as_static() {
                assert!(n >= 1, "cute: experimental::get_arrival_count: fan_in must be at least 1, not {n}");
            }
            f
        }
        None => {
            assert!(
                !(multi && arrival == Arrival::Umma),
                "cute: experimental::get_arrival_count: the {side:?} side's UMMA commits of {producer} -> {consumer} \
                 are multicast across CTAs, so how many land on one barrier is not known here: pass fan_in"
            );
            let owners = match options.locale {
                Some(PipelineLocale::Cluster { full, empty }) => (full, empty),
                _ => default_owners(pair, p, c),
            };
            let owner = if side == Side::Producer { owners.0 } else { owners.1 };
            let leader = options.elect_leader_cta;
            match owner {
                MbarrierOwner::All | MbarrierOwner::CtaPairPeer => BarrierParam::Static(1),
                MbarrierOwner::CtaPairLeader => BarrierParam::Static(if leader { 1 } else { 2 }),
                MbarrierOwner::Rank(_) => {
                    let n = cluster_size();
                    BarrierParam::Dynamic(if leader { n / 2 } else { n })
                }
            }
        }
    };
    mul(per_cta, fan_in).value()
}

// ---------------------------------------------------------------------------
// Stage tokens and states
// ---------------------------------------------------------------------------

/// A stage handed out by [`get_pipeline_produce_stage`] or
/// [`get_pipeline_consume_stage`] (CuTe DSL's `!lir.stage_token`): the
/// stage's index and the barrier its asynchronous operations report to
/// ([`get_mbarrier`]): the full barrier for the producer (a TMA load's or
/// a cluster launch control query's), the empty barrier for the consumer.
/// Only the index is carried through loops. The token also remembers the
/// pipeline's type (CuTe DSL's token type names it), so that operations that
/// need a pipeline of a given locale can check it.
#[derive(Clone, Debug)]
pub struct StageToken {
    base: Ptr<u64>,
    index: Int32,
    pipeline_type: String,
    locale: Option<PipelineLocale>,
}

impl StageToken {
    /// The stage index.
    pub fn index(&self) -> Int32 {
        self.index
    }

    /// The locale of the pipeline the token came from (`None` for a
    /// pipeline created without one).
    pub fn locale(&self) -> Option<PipelineLocale> {
        self.locale
    }

    /// The `!lir.pipeline<...>` type of the pipeline the token came from.
    pub fn pipeline_type(&self) -> &str {
        &self.pipeline_type
    }
}

impl Carry for StageToken {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.index.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        StageToken {
            base: self.base.clone(),
            index: self.index.rebuild(values),
            pipeline_type: self.pipeline_type.clone(),
            locale: self.locale,
        }
    }
}

/// The barrier of a stage token (CuTe DSL's `get_mbarrier`).
#[track_caller]
pub fn get_mbarrier(stage_token: &StageToken) -> Ptr<u64> {
    stage_token.base.offset(stage_token.index)
}

/// The producer's stage at `state`: its token and index (CuTe DSL's
/// `get_pipeline_produce_stage`).
#[track_caller]
pub fn get_pipeline_produce_stage(pipeline: &RawPipeline, state: &PipelineState) -> (StageToken, Int32) {
    pipeline.check_state(state, "get_pipeline_produce_stage");
    let token = StageToken {
        base: pipeline.full_barriers().get_barrier(0),
        index: state.index(),
        pipeline_type: pipeline.type_string(),
        locale: pipeline.locale,
    };
    (token, state.index())
}

/// The consumer's stage at `state` (CuTe DSL's `get_pipeline_consume_stage`).
#[track_caller]
pub fn get_pipeline_consume_stage(pipeline: &RawPipeline, state: &PipelineState) -> (StageToken, Int32) {
    pipeline.check_state(state, "get_pipeline_consume_stage");
    let token = StageToken {
        base: pipeline.empty_barriers().get_barrier(0),
        index: state.index(),
        pipeline_type: pipeline.type_string(),
        locale: pipeline.locale,
    };
    (token, state.index())
}

/// The phase (the round of the ring) a state carries for the producer
/// (CuTe DSL's `get_pipeline_produce_phase`); see the module docs.
pub fn get_pipeline_produce_phase(state: &PipelineState) -> Int32 {
    state.phase()
}

/// The phase a state carries for the consumer (CuTe DSL's
/// `get_pipeline_consume_phase`), the same as the producer's.
pub fn get_pipeline_consume_phase(state: &PipelineState) -> Int32 {
    state.phase()
}

/// The state one stage on (CuTe DSL's `pipeline_advance_iterator`): the
/// index wraps and the phase flips at the end of the ring.
#[track_caller]
pub fn pipeline_advance_iterator(pipeline: &RawPipeline, state: &PipelineState) -> PipelineState {
    pipeline.check_state(state, "pipeline_advance_iterator");
    let mut s = *state;
    s.advance();
    s
}

/// A state at stage `stage_index` and phase `phase` of a pipeline of
/// `stage` stages (CuTe DSL's `create_pipeline_state_at`), for an agent
/// that visits stages out of order, such as a retry sweep. It only
/// packages the values; nothing waits. Its count is `phase * stage +
/// stage_index`.
#[track_caller]
pub fn create_pipeline_state_at(
    pipeline: &RawPipeline,
    stage: usize,
    stage_index: impl IntoValue<i32>,
    phase: impl IntoValue<i32>,
) -> PipelineState {
    assert!(
        stage == pipeline.stages,
        "cute: experimental::create_pipeline_state_at: the pipeline has {} stages, not {stage}",
        pipeline.stages
    );
    let (index, phase) = (stage_index.into_value(), phase.into_value());
    let count = phase * stage as i32 + index;
    PipelineState::new(stage, count, index, phase)
}

// ---------------------------------------------------------------------------
// Producer and consumer operations
// ---------------------------------------------------------------------------

/// Wait until the stage at `state` is empty (CuTe DSL's
/// `producer_acquire`): the empty barrier's phase of the previous round.
/// Returns the state (unchanged).
#[track_caller]
pub fn producer_acquire(pipeline: &RawPipeline, state: &PipelineState) -> PipelineState {
    producer_acquire_with(pipeline, state, None)
}

/// [`producer_acquire`] that skips the wait when `token` (an earlier
/// [`producer_try_acquire`] at the same state) is true.
#[track_caller]
pub fn producer_acquire_with(pipeline: &RawPipeline, state: &PipelineState, token: Option<Boolean>) -> PipelineState {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "producer_acquire");
    let empty = pipeline.empty_barriers();
    wait_unless(token, || empty.wait(state.index(), state.phase() ^ 1));
    *state
}

/// Whether the stage at `state` is empty already, probing once (CuTe
/// DSL's `producer_try_acquire`); a true `token` (a skip-wait flag) returns
/// true without probing.
#[track_caller]
pub fn producer_try_acquire(pipeline: &RawPipeline, state: &PipelineState, token: Option<Boolean>) -> Boolean {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "producer_try_acquire");
    let empty = pipeline.empty_barriers();
    probe_unless(token, || empty.try_wait(state.index(), state.phase() ^ 1))
}

/// Signal that the stage at `state` is filled (CuTe DSL's
/// `producer_commit`): the producer side arrives on its full barrier as its
/// kinds do (see the module docs). `elect_one_sync` makes one thread per
/// warp of thread kinds arrive; `elect_leader_cta` lets only the leader CTA
/// of each pair arrive.
#[track_caller]
pub fn producer_commit(pipeline: &RawPipeline, state: &PipelineState, elect_one_sync: bool, elect_leader_cta: bool) {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "producer_commit");
    let bar = pipeline.full_barriers().get_barrier(state.index());
    on_leader(elect_leader_cta, || pipeline.arrive(Side::Producer, bar, elect_one_sync));
}

/// Wait until the stage at `state` is full (CuTe DSL's `consumer_wait`).
/// Returns the state (unchanged).
#[track_caller]
pub fn consumer_wait(pipeline: &RawPipeline, state: &PipelineState) -> PipelineState {
    consumer_wait_with(pipeline, state, None)
}

/// [`consumer_wait`] that skips the wait when `token` is true.
#[track_caller]
pub fn consumer_wait_with(pipeline: &RawPipeline, state: &PipelineState, token: Option<Boolean>) -> PipelineState {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "consumer_wait");
    let full = pipeline.full_barriers();
    wait_unless(token, || full.wait(state.index(), state.phase()));
    *state
}

/// Whether the stage at `state` is full already, probing once (CuTe DSL's
/// `consumer_try_wait`); a true `token` returns true without probing.
#[track_caller]
pub fn consumer_try_wait(pipeline: &RawPipeline, state: &PipelineState, token: Option<Boolean>) -> Boolean {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "consumer_try_wait");
    let full = pipeline.full_barriers();
    probe_unless(token, || full.try_wait(state.index(), state.phase()))
}

/// Signal that the consumer is done with the stage at `state` (CuTe DSL's
/// `consumer_release`): the consumer side arrives on its empty barrier.
#[track_caller]
pub fn consumer_release(pipeline: &RawPipeline, state: &PipelineState, elect_one_sync: bool, elect_leader_cta: bool) {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "consumer_release");
    let bar = pipeline.empty_barriers().get_barrier(state.index());
    on_leader(elect_leader_cta, || pipeline.arrive(Side::Consumer, bar, elect_one_sync));
}

/// [`consumer_release`] with `elect_one_sync` (CuTe DSL's
/// `consumer_release_elect_one_sync`).
#[track_caller]
pub fn consumer_release_elect_one_sync(pipeline: &RawPipeline, state: &PipelineState) {
    consumer_release(pipeline, state, true, false);
}

/// Drain the pipeline before exit (CuTe DSL's `pipeline_tail`): for each
/// of the ring's stages from `state` on, wait until the releases of the
/// round before have landed (the empty barrier's phase, as a producer's
/// acquire does), so that no arrival or commit is in flight when the CTA
/// exits. With the producer's state this waits for every release of the
/// stages it filled; with the consumer's, for its own releases (UMMA
/// commits included) to complete. Only the owner of the empty barriers
/// waits; the state advances everywhere. Returns the advanced state.
#[track_caller]
pub fn pipeline_tail(pipeline: &RawPipeline, state: &PipelineState) -> PipelineState {
    let _at = builder::LocationScope::new();
    pipeline.check_state(state, "pipeline_tail");
    let empty = pipeline.empty_barriers();
    let start = *state;
    let drain = || {
        let mut s = start;
        for _ in 0..pipeline.stages {
            empty.wait(s.index(), s.phase() ^ 1);
            s.advance();
        }
    };
    match pipeline.empty_owner.is_owner() {
        None => drain(),
        Some(owner) => if_(owner, drain),
    }
    let mut s = start;
    s.advance_by(pipeline.stages);
    s
}

/// The legacy name of [`pipeline_tail`] (CuTe DSL's `consumer_tail`).
#[track_caller]
pub fn consumer_tail(pipeline: &RawPipeline, state: &PipelineState) -> PipelineState {
    pipeline_tail(pipeline, state)
}

/// Add `tx_bytes` to the transaction count of the mbarrier at `mbar` (CuTe
/// DSL's `mbarrier_expect_tx`): of CTA `cta_id` of the cluster when given,
/// from one elected thread per warp with `elect_one_sync`. No arrival.
#[track_caller]
pub fn mbarrier_expect_tx(mbar: &Ptr<u64>, tx_bytes: impl IntoValue<i32>, cta_id: Option<Int32>, elect_one_sync: bool) {
    let tx = tx_bytes.into_value();
    let opts = MbarrierOptions { peer_cta_rank_in_cluster: cta_id, ..Default::default() };
    if elect_one_sync {
        if_(elect_one(), || mbarrier_expect_tx_with(mbar, tx, opts));
    } else {
        mbarrier_expect_tx_with(mbar, tx, opts);
    }
}

// ---------------------------------------------------------------------------
// Circular buffer
// ---------------------------------------------------------------------------

/// A position in a circular-buffer view of a pipeline (CuTe DSL's
/// `!lir.circular_buffer_pipeline_state<stages, count_per_stage,
/// count_per_iteration>`), made by [`create_circular_buffer_pipeline`].
///
/// Each stage holds `count_per_stage` units (`C`), consumed in order
/// `count_per_iteration` (`I`) at a time; one of `C` and `I` divides the
/// other. The position `p` counts units from the start; it is kept as the
/// stage holding unit `p` (a [`PipelineState`]: index and round phase) and
/// `p`'s offset in it. An iteration covers the units `[p, p + I)`:
///
/// * [`circular_buffer_pipeline_consume`] waits on the full barrier of
///   each stage that range overlaps and no earlier iteration did: the
///   current stage when `p` starts it (`I <= C`: only when the offset is
///   0), or the `I / C` stages from the current one (`I > C`);
/// * [`circular_buffer_pipeline_consumer_release`] releases each stage
///   whose last unit lies in the range: the current stage when the offset
///   plus `I` reaches `C` (`I <= C`), or all `I / C` stages (`I > C`), so a
///   stage is released once, after its last unit;
/// * [`circular_buffer_pipeline_advance_iterator`] moves `p` on by `I`; the
///   stage index wraps and the phase flips at the end of the ring.
///
/// The state is carried through loops ([`Carry`]: the stage state and the
/// offset).
#[derive(Clone, Copy, Debug)]
pub struct CircularBufferPipelineState {
    count_per_stage: usize,
    count_per_iteration: usize,
    state: PipelineState,
    offset: Int32,
}

impl CircularBufferPipelineState {
    /// The units per stage.
    pub fn count_per_stage(&self) -> usize {
        self.count_per_stage
    }

    /// The units per iteration.
    pub fn count_per_iteration(&self) -> usize {
        self.count_per_iteration
    }

    /// The stage holding the current unit (cute-rs's addition, for kernels
    /// that index the stage's buffer themselves).
    pub fn stage_index(&self) -> Int32 {
        self.state.index()
    }

    /// The current unit's offset in its stage (cute-rs's addition).
    pub fn unit_offset(&self) -> Int32 {
        self.offset
    }

    /// The stage state underneath.
    pub fn stage_state(&self) -> PipelineState {
        self.state
    }
}

impl Carry for CircularBufferPipelineState {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.state.flatten(out);
        self.offset.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        CircularBufferPipelineState { state: self.state.rebuild(values), offset: self.offset.rebuild(values), ..*self }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let shape = |s: &Self| (s.state.stages(), s.count_per_stage, s.count_per_iteration);
        (shape(self) != shape(other)).then(|| {
            format!(
                "a circular buffer state (stages, count_per_stage, count_per_iteration) {:?} became {:?}",
                shape(self),
                shape(other)
            )
        })
    }
}

/// A circular-buffer view of `pipeline` from `pipeline_state` (a stage
/// boundary), each of its `stages` stages holding `count_per_stage` units
/// consumed `count_per_iteration` at a time (CuTe DSL's
/// `create_circular_buffer_pipeline`; see
/// [`CircularBufferPipelineState`]). Panics when `stages` is not the
/// pipeline's, a count is 0, neither count divides the other, or an
/// iteration spans more than the ring.
#[track_caller]
pub fn create_circular_buffer_pipeline(
    pipeline: &RawPipeline,
    pipeline_state: &PipelineState,
    stages: usize,
    count_per_stage: usize,
    count_per_iteration: usize,
) -> CircularBufferPipelineState {
    const WHAT: &str = "cute: experimental::create_circular_buffer_pipeline";
    pipeline.check_state(pipeline_state, "create_circular_buffer_pipeline");
    assert!(stages == pipeline.stages, "{WHAT}: the pipeline has {} stages, not {stages}", pipeline.stages);
    assert!(
        count_per_stage > 0 && count_per_iteration > 0,
        "{WHAT}: count_per_stage ({count_per_stage}) and count_per_iteration ({count_per_iteration}) must be positive"
    );
    assert!(
        count_per_stage.is_multiple_of(count_per_iteration) || count_per_iteration.is_multiple_of(count_per_stage),
        "{WHAT}: one of count_per_stage ({count_per_stage}) and count_per_iteration ({count_per_iteration}) must \
         divide the other, so that every iteration starts and every stage ends on a unit boundary"
    );
    assert!(
        count_per_iteration <= count_per_stage * stages,
        "{WHAT}: an iteration of {count_per_iteration} units spans more than the ring's {stages} stages of \
         {count_per_stage}"
    );
    CircularBufferPipelineState { count_per_stage, count_per_iteration, state: *pipeline_state, offset: Int32::from(0) }
}

/// Wait for the stages the next iteration's units need (CuTe DSL's
/// `circular_buffer_pipeline_consume`; see [`CircularBufferPipelineState`]).
/// With `elect_leader_cta` only the leader CTA of each pair waits.
#[track_caller]
pub fn circular_buffer_pipeline_consume(
    pipeline: &RawPipeline,
    state: &CircularBufferPipelineState,
    elect_leader_cta: bool,
) {
    let _at = builder::LocationScope::new();
    let (c, i) = (state.count_per_stage, state.count_per_iteration);
    on_leader(elect_leader_cta, || {
        if i <= c {
            let wait = || {
                consumer_wait(pipeline, &state.state);
            };
            if i == c { wait() } else { if_(state.offset.eq(0), wait) }
        } else {
            let mut s = state.state;
            for _ in 0..i / c {
                consumer_wait(pipeline, &s);
                s.advance();
            }
        }
    });
}

/// Release the stages the current iteration finishes (CuTe DSL's
/// `circular_buffer_pipeline_consumer_release`; see
/// [`CircularBufferPipelineState`]), each as [`consumer_release`] does.
#[track_caller]
pub fn circular_buffer_pipeline_consumer_release(
    pipeline: &RawPipeline,
    state: &CircularBufferPipelineState,
    elect_one_sync: bool,
    elect_leader_cta: bool,
) {
    let _at = builder::LocationScope::new();
    let (c, i) = (state.count_per_stage, state.count_per_iteration);
    if i <= c {
        let release = || consumer_release(pipeline, &state.state, elect_one_sync, elect_leader_cta);
        if i == c { release() } else { if_((state.offset + i as i32).eq(c as i32), release) }
    } else {
        let mut s = state.state;
        for _ in 0..i / c {
            consumer_release(pipeline, &s, elect_one_sync, elect_leader_cta);
            s.advance();
        }
    }
}

/// The position `count_per_iteration` units on (CuTe DSL's
/// `circular_buffer_pipeline_advance_iterator`).
#[track_caller]
pub fn circular_buffer_pipeline_advance_iterator(
    pipeline: &RawPipeline,
    state: &CircularBufferPipelineState,
) -> CircularBufferPipelineState {
    let _at = builder::LocationScope::new();
    pipeline.check_state(&state.state, "circular_buffer_pipeline_advance_iterator");
    let (c, i) = (state.count_per_stage, state.count_per_iteration);
    let mut out = *state;
    if i < c {
        let next_offset = state.offset + i as i32;
        let wrap = next_offset.eq(c as i32);
        let mut next = state.state;
        next.advance();
        out.offset = select(wrap, 0, next_offset);
        out.state = PipelineState::new(
            state.state.stages(),
            select(wrap, next.count(), state.state.count()),
            select(wrap, next.index(), state.state.index()),
            select(wrap, next.phase(), state.state.phase()),
        );
    } else {
        out.state.advance_by(i / c);
    }
    out
}

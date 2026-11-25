// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.enums (NVIDIA EULA); the code is
// written independently.

//! The small vocabularies of Task Scheduling: which pipeline a resource
//! uses ([`PipelineType`]), which tile scheduler a work queue wraps
//! ([`TileSchedulerType`]), which threads signal a barrier
//! ([`SignalingThreads`]), how a work method is classified ([`WorkAttr`],
//! [`ScheduleStage`]), where in a schedule a step sits ([`Phase`]) and the
//! guards of conditional blocks ([`IterationPredicate`], [`Guard`]).

use std::fmt;

/// The pipeline a [`PipelineConfig`](super::PipelineConfig) materialises:
/// what kind of agent fills a stage and what kind drains it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    /// Threads produce, threads consume ([`PipelineAsync`](crate::pipeline::PipelineAsync),
    /// or [`PipelineCpAsync`](crate::pipeline::PipelineCpAsync) for `cp.async` producers).
    AsyncAsync,
    /// TMA loads produce, threads consume ([`PipelineTmaAsync`](crate::pipeline::PipelineTmaAsync),
    /// or a [`TsPipeline`](super::TsPipeline) with a task-warp leader or
    /// all-thread release).
    TmaAsync,
    /// TMA loads produce, `tcgen05` MMAs consume ([`PipelineTmaUmma`](crate::pipeline::PipelineTmaUmma)).
    TmaUmma,
    /// `tcgen05` MMAs produce, threads consume ([`PipelineUmmaAsync`](crate::pipeline::PipelineUmmaAsync)).
    UmmaAsync,
    /// Threads (or `cp.async` copies) produce, `tcgen05` MMAs consume (a
    /// [`TsPipeline`](super::TsPipeline), CuTe DSL's `TSPipelineAsyncUmma`).
    AsyncUmma,
    /// `tcgen05` MMAs on both sides (Task Scheduling's own
    /// [`TsPipeline`](super::TsPipeline)).
    UmmaUmma,
    /// Cluster launch control responses produce, threads consume
    /// ([`PipelineClcFetchAsync`](crate::pipeline::PipelineClcFetchAsync)).
    ClcFetchAsync,
}

/// How a [`PipelineGroup`](super::PipelineGroup) shares its members'
/// barriers (CuTe DSL's `PipelineGroupMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineGroupMode {
    /// Several producers, one consumer: each member keeps its full
    /// barriers, the empty barriers are shared, and the consumer releases
    /// every member at once through the group.
    Merge,
    /// One producer, several consumers: the full barriers are shared (the
    /// producer commits every member at once through the group), each
    /// member keeps its empty barriers.
    Fork,
    /// Several producers, one consumer, sharing both: one full barrier per
    /// stage counts every member's producer arrivals (and bytes), and the
    /// consumer releases through the group.
    FusedMerge,
}

/// The tile scheduler a [`WorkQueue`](super::WorkQueue) wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileSchedulerType {
    /// [`StaticPersistentTileScheduler`](crate::utils::StaticPersistentTileScheduler):
    /// every CTA strides over the tiles by itself, so advancing needs no
    /// producer.
    StaticPersistent,
    /// [`ClcDynamicPersistentTileScheduler`](crate::utils::ClcDynamicPersistentTileScheduler):
    /// a scheduler warp fetches tiles with cluster launch control and hands
    /// them over through a pipeline (sm_100).
    ClcDynamicPersistent,
}

/// Which threads carry out one side's barrier operations, as a set of
/// flags: every thread of the group (`ALL`), only CTA 0 of the cluster
/// (`CTA_LEADER`), or only the task's first warp (`TASK_WARP_LEADER`).
/// `ALL` combines with nothing; the two leader flags combine with each
/// other.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignalingThreads(u8);

impl SignalingThreads {
    /// Every thread of the cooperative group.
    pub const ALL: SignalingThreads = SignalingThreads(1);
    /// Only the threads of CTA 0 in the cluster.
    pub const CTA_LEADER: SignalingThreads = SignalingThreads(2);
    /// Only the first warp of the task.
    pub const TASK_WARP_LEADER: SignalingThreads = SignalingThreads(4);

    /// Whether this is a combination Task Scheduling accepts: exactly
    /// `ALL`, or a non-empty set of the leader flags.
    pub fn is_valid(self) -> bool {
        self == Self::ALL || (self.0 != 0 && self.0 & !6 == 0)
    }

    /// Whether `CTA_LEADER` is set.
    pub fn has_cta_leader(self) -> bool {
        self.0 & 2 != 0
    }

    /// Whether `TASK_WARP_LEADER` is set.
    pub fn has_task_warp_leader(self) -> bool {
        self.0 & 4 != 0
    }
}

impl std::ops::BitOr for SignalingThreads {
    type Output = SignalingThreads;
    fn bitor(self, rhs: SignalingThreads) -> SignalingThreads {
        SignalingThreads(self.0 | rhs.0)
    }
}

impl Default for SignalingThreads {
    fn default() -> Self {
        Self::ALL
    }
}

impl fmt::Debug for SignalingThreads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        for (bit, name) in [(1, "ALL"), (2, "CTA_LEADER"), (4, "TASK_WARP_LEADER")] {
            if self.0 & bit != 0 {
                names.push(name);
            }
        }
        if names.is_empty() {
            names.push("(none)");
        }
        f.write_str(&names.join(" | "))
    }
}

/// Attributes of a work method. `AUXILIARY` marks set-up work (building a
/// view of shared memory, seeding loop state) that is not part of the
/// stage hand-off and may sit outside the work-tile loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkAttr(u8);

impl WorkAttr {
    /// No attributes: ordinary producer or consumer work.
    pub const NONE: WorkAttr = WorkAttr(0);
    /// Auxiliary (set-up) work.
    pub const AUXILIARY: WorkAttr = WorkAttr(1);

    /// Whether `AUXILIARY` is set.
    pub fn is_auxiliary(self) -> bool {
        self.0 & 1 != 0
    }
}

/// What one step of a schedule does: one of the six pipeline operations,
/// or a work call of one of four kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleStage {
    /// Probe whether the producer's stage is free.
    ProducerTryAcquire,
    /// Wait until the producer's stage is free.
    ProducerAcquire,
    /// Producer work on the acquired stage.
    ProducerWork,
    /// Auxiliary producer work.
    ProducerAuxWork,
    /// Signal that the stage is filled.
    ProducerCommit,
    /// Probe whether the consumer's stage is full.
    ConsumerTryWait,
    /// Wait until the consumer's stage is full.
    ConsumerWait,
    /// Consumer work on the waited stage.
    ConsumerWork,
    /// Auxiliary consumer work.
    ConsumerAuxWork,
    /// Signal that the stage is drained.
    ConsumerRelease,
}

impl ScheduleStage {
    /// Whether this is a work call rather than a pipeline operation.
    pub fn is_work(self) -> bool {
        matches!(self, Self::ProducerWork | Self::ProducerAuxWork | Self::ConsumerWork | Self::ConsumerAuxWork)
    }

    /// Whether this is an auxiliary work call.
    pub fn is_auxiliary(self) -> bool {
        matches!(self, Self::ProducerAuxWork | Self::ConsumerAuxWork)
    }

    /// The name a schedule uses for a pipeline operation (`acquire`,
    /// `try_wait`, ...), or `None` for work.
    pub fn op_name(self) -> Option<&'static str> {
        Some(match self {
            Self::ProducerTryAcquire => "try_acquire",
            Self::ProducerAcquire => "acquire",
            Self::ProducerCommit => "commit",
            Self::ConsumerTryWait => "try_wait",
            Self::ConsumerWait => "wait",
            Self::ConsumerRelease => "release",
            _ => return None,
        })
    }

    /// The pipeline operation called `name` in a schedule.
    pub fn from_op_name(name: &str) -> Option<ScheduleStage> {
        Some(match name {
            "try_acquire" => Self::ProducerTryAcquire,
            "acquire" => Self::ProducerAcquire,
            "commit" => Self::ProducerCommit,
            "try_wait" => Self::ConsumerTryWait,
            "wait" => Self::ConsumerWait,
            "release" => Self::ConsumerRelease,
            _ => return None,
        })
    }

    /// The names of the pipeline operations, which work methods may not
    /// take.
    pub const OP_NAMES: [&'static str; 6] = ["try_acquire", "acquire", "commit", "try_wait", "wait", "release"];
}

/// Where a step sits relative to the loops of its schedule (CuTe DSL's
/// `ScheduleStageType`): before the domain loop of its body (`Head`),
/// inside it (`Loop`, or `LoopFirstIter` / `LoopLastIter` under a first- or
/// last-iteration guard), or after it (`Tail`). A work-tile loop body with
/// no domain loop is all `Loop`; a schedule with no loop at all is all
/// `Head`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Head,
    Loop,
    LoopFirstIter,
    LoopLastIter,
    Tail,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Phase::Head => "head",
            Phase::Loop => "loop",
            Phase::LoopFirstIter => "loop-first",
            Phase::LoopLastIter => "loop-last",
            Phase::Tail => "tail",
        })
    }
}

/// A guard decided from the zero-based iteration count of the enclosing
/// domain loop, so that it can be checked when the schedule is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IterationPredicate {
    /// The first iteration.
    FirstIter,
    /// The last iteration (none of an empty loop).
    LastIter,
    /// Iterations `start`, `start + period`, `start + 2 * period`, ...
    Every { period: u32, start: u32 },
}

impl IterationPredicate {
    /// Whether the guarded block runs on iteration `iter` of `num_iters`.
    pub fn fires(self, iter: u64, num_iters: u64) -> bool {
        match self {
            Self::FirstIter => iter == 0,
            Self::LastIter => num_iters > 0 && iter + 1 == num_iters,
            Self::Every { period, start } => {
                period > 0 && iter >= start as u64 && (iter - start as u64).is_multiple_of(period as u64)
            }
        }
    }
}

impl fmt::Display for IterationPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstIter => f.write_str("first_iter"),
            Self::LastIter => f.write_str("last_iter"),
            Self::Every { period, start: 0 } => write!(f, "every({period})"),
            Self::Every { period, start } => write!(f, "every({period}, start={start})"),
        }
    }
}

/// The guard of a conditional block in a captured schedule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// Runs on the iterations the predicate selects.
    Iteration(IterationPredicate),
    /// Runs when a runtime value is non-zero (or zero, when `negated`): a
    /// work output named `resource.work -> variable`, or a free-standing
    /// condition named by `key`.
    Opaque { key: String, negated: bool },
    /// The work of a [`WorkTileLoopScope::skippable`](super::WorkTileLoopScope::skippable)
    /// region: left out for tiles the work queue skips.
    Skippable,
}

impl fmt::Display for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Guard::Iteration(p) => write!(f, "{p}"),
            Guard::Opaque { key, negated: false } => write!(f, "when_true({key})"),
            Guard::Opaque { key, negated: true } => write!(f, "when_false({key})"),
            Guard::Skippable => f.write_str("skippable"),
        }
    }
}

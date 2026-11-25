//! An array of mbarriers, one per pipeline stage ([`MbarrierArray`]), the
//! operations that arrive on it ([`PipelineOp`]) and reserved storage for
//! it ([`alloc_reserved_mbarrier`]).

use crate::arch::{
    MbarrierOptions, cp_async_mbarrier_arrive_noinc, elect_one, make_warp_uniform, mbarrier_arrive,
    mbarrier_arrive_and_expect_tx, mbarrier_arrive_and_expect_tx_with, mbarrier_arrive_with, mbarrier_init,
    mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait, warp_idx,
};
use crate::control::{if_, while_carry};
use crate::layout::IndexExpr;
use crate::memory::{SmemAllocator, SmemPartition};
use crate::tensor::Ptr;
use crate::value::{Boolean, Int32, IntoValue};

use super::{BarrierParam, CooperativeGroup};

/// The hardware operation that arrives on a pipeline's barriers (CuTe DSL's
/// `cutlass.pipeline.PipelineOp`); [`MbarrierArray::arrive`] dispatches on
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineOp {
    /// Ordinary threads arrive (`mbarrier.arrive`).
    AsyncThread,
    /// A `tcgen05` MMA commits to the barrier (sm_100; the arrival comes
    /// with `tcgen05.commit`, roadmap part m10-umma-f16).
    TCGen05Mma,
    /// A TMA load: one elected thread arrives and sets the bytes to expect
    /// (`mbarrier.arrive.expect_tx`).
    TmaLoad,
    /// A cluster launch control query: every calling thread arrives with
    /// the expected bytes, on the CTA it names.
    ClcLoad,
    /// A TMA store, synchronised by bulk groups rather than an mbarrier.
    TmaStore,
    /// Several operations at once, as composite pipelines use.
    Composite,
    /// `cp.async` copies: each thread's arrival comes when its copies land
    /// (`cp.async.mbarrier.arrive.noinc`).
    AsyncLoad,
}

impl PipelineOp {
    /// CuTe DSL's name of the operation.
    pub fn name(self) -> &'static str {
        match self {
            PipelineOp::AsyncThread => "AsyncThread",
            PipelineOp::TCGen05Mma => "TCGen05Mma",
            PipelineOp::TmaLoad => "TmaLoad",
            PipelineOp::ClcLoad => "ClcLoad",
            PipelineOp::TmaStore => "TmaStore",
            PipelineOp::Composite => "Composite",
            PipelineOp::AsyncLoad => "AsyncLoad",
        }
    }
}

/// How an [`MbarrierArray`]'s barriers sit in memory (CuTe DSL's
/// `MbarrierLayout`): `V0`, one 8-byte mbarrier per stage, consecutive, is
/// the only layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MbarrierLayout {
    #[default]
    V0,
}

/// The number of CTAs a `tcgen05` operation spans (CuTe DSL's
/// `tcgen05.CtaGroup`), which [`MbarrierArray::arrive`] takes for
/// [`PipelineOp::TCGen05Mma`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CtaGroup {
    One,
    Two,
}

/// The optional arguments of [`MbarrierArray::with_options`].
#[derive(Clone, Debug, Default)]
pub struct MbarrierArrayOptions {
    /// The barriers' layout ([`MbarrierLayout::V0`], the only one).
    pub layout: MbarrierLayout,
    /// A name for profiling (CuTe DSL's `name`, which its profiler groups
    /// operations by); kept, and not used yet.
    pub name: String,
}

/// One mbarrier per pipeline stage in shared memory (CuTe DSL's
/// `cutlass.pipeline.MbarrierArray`), counting the threads of one
/// [`CooperativeGroup`], and arrived on by one [`PipelineOp`].
///
/// [`new`](Self::new) initialises every stage from warp 0's elected lane;
/// make the initialisation visible before any thread uses the barriers
/// ([`pipeline_init_arrive`](super::pipeline_init_arrive) and
/// [`pipeline_init_wait`](super::pipeline_init_wait), or a
/// [`sync_threads`](crate::sync_threads)).
///
/// # Waiting on SM80 to SM89
///
/// `mbarrier.try_wait` and `fence.mbarrier_init` need sm_90, and CuTe DSL's
/// pipelines are Hopper-only. When the kernel is built for a target below
/// sm_90 ([`KernelOptions::target`](crate::KernelOptions::target) or
/// `CUTE_RS_ARCH`, as [`target_arch`](crate::target_arch) reports it),
/// [`wait`](Self::wait) spins on `mbarrier.test_wait` (sm_80) instead,
/// [`try_wait`](Self::try_wait) probes with it, and the init helpers leave
/// the fence out (a CTA barrier publishes CTA-local mbarriers there). With
/// sm_90 or newer, or no declared target, they emit what CuTe DSL emits, so
/// such a kernel compiled for sm_89 fails, naming `mbarrier.try_wait` and
/// the line that waits.
#[derive(Clone, Debug)]
pub struct MbarrierArray {
    base: Ptr<u64>,
    num_stages: usize,
    op: PipelineOp,
    group: CooperativeGroup,
    arrive_count: BarrierParam,
    tx_count: BarrierParam,
    options: MbarrierArrayOptions,
}

impl MbarrierArray {
    /// `num_stages` mbarriers from `barrier_storage` on, arrived on by
    /// `agent.0` for the threads of `agent.1`, each phase also waiting for
    /// `tx_count` bytes when the operation is [`PipelineOp::TmaLoad`] or
    /// [`PipelineOp::ClcLoad`] (0 otherwise). Initialises them: inside warp 0
    /// ([`make_warp_uniform`] of [`warp_idx`]), its elected lane runs
    /// `mbarrier.init` on every stage with the group's thread count.
    ///
    /// Panics for 0 stages, a static thread count below 1 (a runtime count
    /// that is a constant included), and a negative static byte count for
    /// a TMA load.
    #[track_caller]
    pub fn new(
        barrier_storage: &Ptr<u64>,
        num_stages: usize,
        agent: (PipelineOp, CooperativeGroup),
        tx_count: impl Into<BarrierParam>,
    ) -> MbarrierArray {
        MbarrierArray::with_options(barrier_storage, num_stages, agent, tx_count, MbarrierArrayOptions::default())
    }

    /// [`new`](Self::new) with a layout and a name.
    #[track_caller]
    pub fn with_options(
        barrier_storage: &Ptr<u64>,
        num_stages: usize,
        agent: (PipelineOp, CooperativeGroup),
        tx_count: impl Into<BarrierParam>,
        options: MbarrierArrayOptions,
    ) -> MbarrierArray {
        let (op, group) = agent;
        let tx_count = tx_count.into();
        assert!(num_stages > 0, "cute: MbarrierArray: an mbarrier array needs at least one stage, not 0");
        let arrive_count = group.thread_arrive_count();
        if let Some(n) = arrive_count.as_static() {
            assert!(n > 0, "cute: MbarrierArray: each mbarrier must count at least one arrival, not {n}");
        }
        if op == PipelineOp::TmaLoad
            && let Some(tx) = tx_count.as_static()
        {
            assert!(tx >= 0, "cute: MbarrierArray: a TMA load's transaction byte count cannot be negative ({tx})");
        }
        let array =
            MbarrierArray { base: barrier_storage.clone(), num_stages, op, group, arrive_count, tx_count, options };
        array.init();
        array
    }

    /// Initialise every stage, from warp 0's elected lane.
    #[track_caller]
    fn init(&self) {
        let _at = crate::builder::LocationScope::new();
        let count = self.arrive_count.value();
        if_(make_warp_uniform(warp_idx()).eq(0), || {
            if_(elect_one(), || {
                for s in 0..self.num_stages {
                    mbarrier_init(&self.get_barrier(s), count);
                }
            });
        });
    }

    /// A copy that arrives with `op` instead, over the same barriers,
    /// without initialising them again (CuTe DSL's
    /// `recast_to_new_op_type`).
    pub fn recast_to_new_op_type(&self, op: PipelineOp) -> MbarrierArray {
        MbarrierArray { op, ..self.clone() }
    }

    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// The operation that arrives.
    pub fn op(&self) -> PipelineOp {
        self.op
    }

    /// The group whose threads each phase counts.
    pub fn group(&self) -> CooperativeGroup {
        self.group
    }

    /// The arrivals each phase waits for.
    pub fn arrive_count(&self) -> BarrierParam {
        self.arrive_count
    }

    /// The bytes a [`PipelineOp::TmaLoad`] or [`PipelineOp::ClcLoad`]
    /// arrival expects.
    pub fn tx_count(&self) -> BarrierParam {
        self.tx_count
    }

    /// The barriers' layout.
    pub fn layout(&self) -> MbarrierLayout {
        self.options.layout
    }

    /// The profiling name.
    pub fn name(&self) -> &str {
        &self.options.name
    }

    /// The mbarrier of stage `index`, a static or runtime index: the base
    /// pointer plus `index` (CuTe DSL's `get_barrier`).
    #[track_caller]
    pub fn get_barrier(&self, index: impl Into<IndexExpr>) -> Ptr<u64> {
        self.base.offset(index)
    }

    /// The largest arrival count an mbarrier can take, 2^20 - 1 (CuTe DSL's
    /// `max`).
    pub fn max(&self) -> u32 {
        (1 << 20) - 1
    }

    /// Arrive on stage `index` as the array's operation does (CuTe DSL's
    /// `arrive(index, dst, cta_group)`):
    ///
    /// * [`PipelineOp::AsyncThread`]: `mbarrier.arrive`, on this CTA's
    ///   barrier, or with `dst` on the barrier of that CTA rank of the
    ///   cluster ([`MbarrierOptions::peer_cta_rank_in_cluster`], sm_90);
    /// * [`PipelineOp::TmaLoad`]: the warp's elected lane arrives on this
    ///   CTA's barrier and sets the [`tx_count`](Self::tx_count) bytes to
    ///   expect; `dst` is ignored;
    /// * [`PipelineOp::ClcLoad`]: every calling thread arrives with the
    ///   bytes to expect, on the barrier of CTA `dst` (this CTA's without
    ///   one);
    /// * [`PipelineOp::AsyncLoad`]: `cp.async.mbarrier.arrive.noinc`, an
    ///   arrival when the thread's `cp.async` copies have landed;
    /// * [`PipelineOp::TCGen05Mma`] (`dst` a multicast mask): the warp's
    ///   elected lane commits (`tcgen05.commit`), so the arrival comes once
    ///   the MMAs it issued have completed;
    /// * [`PipelineOp::TmaStore`] and [`PipelineOp::Composite`], which no
    ///   mbarrier array supports, panic.
    #[track_caller]
    pub fn arrive(&self, index: impl Into<IndexExpr>, dst: Option<Int32>, cta_group: Option<CtaGroup>) {
        let _at = crate::builder::LocationScope::new();
        let bar = self.get_barrier(index);
        match self.op {
            PipelineOp::AsyncThread => match dst {
                None => mbarrier_arrive(&bar),
                Some(rank) => mbarrier_arrive_with(
                    &bar,
                    1,
                    MbarrierOptions { peer_cta_rank_in_cluster: Some(rank), ..Default::default() },
                ),
            },
            PipelineOp::TmaLoad => {
                let tx = self.tx_count.value();
                if_(elect_one(), || mbarrier_arrive_and_expect_tx(&bar, tx));
            }
            PipelineOp::ClcLoad => {
                let opts = MbarrierOptions { peer_cta_rank_in_cluster: dst, ..Default::default() };
                mbarrier_arrive_and_expect_tx_with(&bar, self.tx_count.value(), opts);
            }
            PipelineOp::AsyncLoad => cp_async_mbarrier_arrive_noinc(&bar),
            PipelineOp::TCGen05Mma => {
                // m10-umma-f16: a tcgen05.commit from the warp's elected lane, with `dst` as the
                // multicast mask.
                let group = cta_group.unwrap_or_else(|| {
                    panic!("cute: MbarrierArray::arrive: a TCGen05Mma arrival needs a CtaGroup (cta_group is None)")
                });
                let group = match group {
                    CtaGroup::One => crate::nvgpu::tcgen05::CtaGroup::ONE,
                    CtaGroup::Two => crate::nvgpu::tcgen05::CtaGroup::TWO,
                };
                let mask = dst.map(|d| d.to::<i16>());
                if_(elect_one(), || crate::nvgpu::tcgen05::commit(&bar, mask, group));
            }
            PipelineOp::TmaStore | PipelineOp::Composite => panic!(
                "cute: MbarrierArray::arrive: an mbarrier array cannot arrive for PipelineOp::{}",
                self.op.name()
            ),
        }
    }

    /// The warp's elected lane arrives on stage `index` and sets `tx_count`
    /// bytes to expect (CuTe DSL's `arrive_and_expect_tx`), whatever the
    /// array's operation; the UMMA pipelines arm a stage with a per-acquire
    /// byte count this way (m10-umma-pipelines).
    #[track_caller]
    pub fn arrive_and_expect_tx(&self, index: impl Into<IndexExpr>, tx_count: impl IntoValue<i32>) {
        let _at = crate::builder::LocationScope::new();
        let (bar, tx) = (self.get_barrier(index), tx_count.into_value());
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&bar, tx));
    }

    /// Wait until the phase with parity `phase` of stage `index` completes
    /// (CuTe DSL's `wait`): [`mbarrier_wait`], or a spin on
    /// `mbarrier.test_wait` below sm_90 (see the type's docs).
    #[track_caller]
    pub fn wait(&self, index: impl Into<IndexExpr>, phase: impl IntoValue<i32>) {
        let _at = crate::builder::LocationScope::new();
        let (bar, phase) = (self.get_barrier(index), phase.into_value());
        if portable_waits() {
            while_carry((), |()| !mbarrier_test_wait(&bar, phase), |()| ());
        } else {
            mbarrier_wait(&bar, phase);
        }
    }

    /// Whether the phase with parity `phase` of stage `index` has completed,
    /// after waiting a while if not (CuTe DSL's `try_wait`,
    /// `mbarrier.try_wait`); below sm_90 a single `mbarrier.test_wait`.
    #[track_caller]
    pub fn try_wait(&self, index: impl Into<IndexExpr>, phase: impl IntoValue<i32>) -> Boolean {
        let bar = self.get_barrier(index);
        if portable_waits() { mbarrier_test_wait(&bar, phase) } else { mbarrier_try_wait(&bar, phase) }
    }

    /// Whether the phase with parity `phase` of stage `index` has completed,
    /// without waiting (CuTe DSL's `test_wait`, `mbarrier.test_wait`).
    #[track_caller]
    pub fn test_wait(&self, index: impl Into<IndexExpr>, phase: impl IntoValue<i32>) -> Boolean {
        mbarrier_test_wait(&self.get_barrier(index), phase)
    }

    /// [`arrive`](Self::arrive) then [`wait`](Self::wait) on stage `index`.
    /// CuTe DSL 4.8.1's `MbarrierArray.arrive_and_wait` calls the named
    /// barrier functions of the module by mistake; this does what its name
    /// says.
    #[track_caller]
    pub fn arrive_and_wait(
        &self,
        index: impl Into<IndexExpr>,
        phase: impl IntoValue<i32>,
        dst: Option<Int32>,
        cta_group: Option<CtaGroup>,
    ) {
        let index = index.into();
        self.arrive(index.clone(), dst, cta_group);
        self.wait(index, phase);
    }

    /// Not supported (CuTe DSL raises `NotImplementedError`): always panics.
    #[track_caller]
    pub fn arrive_and_drop(&self) {
        panic!("cute: MbarrierArray::arrive_and_drop is not supported yet");
    }
}

/// Whether mbarrier waits spin on `mbarrier.test_wait` and the init helpers
/// leave out `fence.mbarrier_init`: when the kernel being built declares a
/// target below sm_90.
#[track_caller]
pub(crate) fn portable_waits() -> bool {
    crate::target_arch().is_some_and(|arch| arch.sm() < 90)
}

/// Shared memory for `num_stages * multiplier` mbarriers (8-byte words,
/// 8-byte aligned) in the reserved partition, so at the lowest addresses of
/// the block's shared memory (CuTe DSL's `alloc_reserved_mbarrier`;
/// `multiplier` is 2 for a pipeline's full and empty barriers). The
/// pipelines allocate their barriers with it when given no storage.
#[track_caller]
pub fn alloc_reserved_mbarrier(num_stages: usize, multiplier: usize) -> Ptr<u64> {
    SmemAllocator::new().allocate_array::<u64>(num_stages * multiplier, 8, SmemPartition::Reserved)
}

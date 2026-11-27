//! Synchronisation objects from CuTe DSL's `cutlass.pipeline`: the hardware
//! named barriers ([`NamedBarrier`] and the free functions [`arrive`],
//! [`arrive_unaligned`], [`wait`], [`wait_unaligned`], [`arrive_and_wait`]
//! and [`sync`]).
//!
//! A named barrier counts *all* the threads taking part, producers and
//! consumers alike: to hand data from one warp to another, both arrive at a
//! barrier of 64 threads (where an mbarrier would count only the 32
//! producer threads).
//!
//! The building blocks of CuTe DSL's pipelines are here too:
//!
//! * positions in a ring of stages: [`PipelineState`] (made by
//!   [`make_pipeline_state`] for a [`PipelineUserType`]),
//!   [`PerStagePipelineState`] and [`compute_k_tile`], read by the
//!   pipelines through [`StageState`];
//! * the threads on one side of a barrier: [`Agent`] and
//!   [`CooperativeGroup`];
//! * an mbarrier per stage, [`MbarrierArray`], arrived on by a
//!   [`PipelineOp`], with storage from [`alloc_reserved_mbarrier`];
//! * the [`SyncObject`] interface of [`MbarrierArray`] and
//!   [`NamedBarrier`];
//! * publishing initialised barriers, [`pipeline_init_arrive`] and
//!   [`pipeline_init_wait`], and [`agent_sync`].
//!
//! The pipelines built on them share the [`Pipeline`] interface (and
//! [`PipelineBarriers`] for those with barriers to hand out): the generic
//! [`PipelineAsync`], [`PipelineCpAsync`] (whose producers fill stages with
//! `cp.async`), and [`PipelineEmpty`], which does nothing. The TMA pipelines are
//! [`PipelineTmaAsync`] (TMA loads, multicast across a cluster) and
//! [`PipelineTmaStore`] (TMA stores paced by a [`TmaStoreFence`]). Their
//! stateful participants are [`PipelineProducer`] and [`PipelineConsumer`],
//! which return handles to the stages they acquire ([`ProducerHandle`],
//! [`ConsumerHandle`]). [`PipelineOrder`] makes groups of threads take
//! turns.
//!
//! CuTe DSL's pipelines need sm_90 (`mbarrier.try_wait`,
//! `fence.mbarrier_init`). A kernel built for a target below sm_90
//! ([`KernelOptions::target`](crate::KernelOptions::target)) gets waits that
//! spin on `mbarrier.test_wait` and no init fence instead, so the
//! mbarrier-only parts run on SM80 to SM89 (see [`MbarrierArray`]).

mod async_pipeline;
mod cp_async_pipeline;
mod group;
mod init;
mod mbarrier;
mod order;
mod participants;
mod state;
mod sync;
// The TMA load and store pipelines.
mod tma_pipeline;
// The Blackwell (tcgen05) pipelines.
mod umma_pipeline;

// The cluster launch control pipeline.
mod clc_pipeline;
pub use clc_pipeline::{PipelineClcFetchAsync, PipelineClcFetchAsyncOptions};

// The TMA pipeline with UMMA and async consumers.
mod tma_multi_consumer_pipeline;
pub use tma_multi_consumer_pipeline::{
    PipelineTmaMultiConsumersAsync, PipelineTmaMultiConsumersAsyncOptions, init_empty_barrier_arrive_signal_2sm,
};

pub use async_pipeline::{Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineBarriers, PipelineEmpty};
pub use cp_async_pipeline::{PipelineCpAsync, PipelineCpAsyncOptions};
pub use order::{PipelineOrder, PipelineOrderOptions};
pub use participants::{ConsumerHandle, PipelineConsumer, PipelineProducer, ProducerHandle};

pub use group::{Agent, CooperativeGroup};
pub use init::{agent_sync, pipeline_init_arrive, pipeline_init_wait};
pub use mbarrier::{
    CtaGroup, MbarrierArray, MbarrierArrayOptions, MbarrierLayout, PipelineOp, alloc_reserved_mbarrier,
};
pub use state::{
    PerStagePipelineState, PipelineState, PipelineUserType, StageState, compute_k_tile, make_pipeline_state,
};
pub use sync::SyncObject;
pub use tma_pipeline::{
    PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaStore, TmaStoreFence, init_empty_barrier_arrive_signal,
};
// The UMMA pipelines.
pub use umma_pipeline::{
    PipelineAsyncUmma, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync, PipelineUmmaOptions,
};

use crate::arch::named_barrier;
use crate::ir::BarrierKind;
use crate::value::{Int32, IntoValue};

/// A named barrier's id or thread count, as CuTe DSL's `Union[int, Int32]`:
/// a number fixed when the kernel is built (checked when the
/// [`NamedBarrier`] is made), or an [`Int32`] computed at run time (checked
/// by the simulator).
#[derive(Clone, Copy, Debug)]
pub enum BarrierParam {
    Static(i64),
    Dynamic(Int32),
}

impl From<i32> for BarrierParam {
    fn from(v: i32) -> Self {
        BarrierParam::Static(v.into())
    }
}

impl From<u32> for BarrierParam {
    fn from(v: u32) -> Self {
        BarrierParam::Static(v.into())
    }
}

impl From<usize> for BarrierParam {
    fn from(v: usize) -> Self {
        BarrierParam::Static(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<Int32> for BarrierParam {
    fn from(v: Int32) -> Self {
        BarrierParam::Dynamic(v)
    }
}

impl BarrierParam {
    /// The number, when it is known at build time (a constant [`Int32`]
    /// counts).
    pub fn as_static(&self) -> Option<i64> {
        match self {
            BarrierParam::Static(v) => Some(*v),
            BarrierParam::Dynamic(v) => v.as_constant().map(i64::from),
        }
    }

    /// The operand for the barrier instruction.
    #[track_caller]
    pub fn value(&self) -> Int32 {
        match self {
            BarrierParam::Static(v) => {
                i32::try_from(*v).unwrap_or_else(|_| panic!("cute: {v} does not fit in i32")).into_value()
            }
            BarrierParam::Dynamic(v) => *v,
        }
    }
}

/// One of the 16 hardware named barriers of a CTA, used by `num_threads`
/// threads (CuTe DSL's `cutlass.pipeline.NamedBarrier`). Barrier 0 is the
/// one [`sync_threads`](crate::sync_threads) uses, so kernels usually pick
/// ids from 1 up.
#[derive(Clone, Copy, Debug)]
pub struct NamedBarrier {
    barrier_id: BarrierParam,
    num_threads: BarrierParam,
}

impl NamedBarrier {
    /// Barrier `barrier_id` (0 to 15) shared by `num_threads` threads, a
    /// positive multiple of 32. Either can be a constant or an [`Int32`]
    /// computed at run time, such as an id picked by warp role; constants
    /// are checked here, and runtime values by the simulator.
    #[track_caller]
    pub fn new(barrier_id: impl Into<BarrierParam>, num_threads: impl Into<BarrierParam>) -> NamedBarrier {
        let (barrier_id, num_threads) = (barrier_id.into(), num_threads.into());
        if let Some(id) = barrier_id.as_static() {
            assert!(
                (0..16).contains(&id),
                "cute: NamedBarrier id {id} is out of range: a CTA has 16 named barriers, 0 to 15"
            );
        }
        if let Some(n) = num_threads.as_static() {
            assert!(
                n > 0 && n % 32 == 0 && n <= 1024,
                "cute: a NamedBarrier's thread count must be a multiple of 32 between 32 and 1024, not {n}"
            );
        }
        NamedBarrier { barrier_id, num_threads }
    }

    /// The barrier's id (CuTe DSL's `get_barrier`).
    pub fn get_barrier(&self) -> BarrierParam {
        self.barrier_id
    }

    /// The number of threads that complete the barrier.
    pub fn num_threads(&self) -> BarrierParam {
        self.num_threads
    }

    /// The largest arrival count a transaction barrier can take (CuTe DSL's
    /// `max`).
    pub fn max(&self) -> u32 {
        4095
    }

    /// Arrive without waiting (`bar.arrive`); every thread of the CTA must
    /// reach this same instruction.
    #[track_caller]
    pub fn arrive(&self) {
        arrive(self.barrier_id.value(), self.num_threads.value());
    }

    /// Arrive without waiting, with the unaligned `barrier.arrive`.
    #[track_caller]
    pub fn arrive_unaligned(&self) {
        arrive_unaligned(self.barrier_id.value(), self.num_threads.value());
    }

    /// Named barriers have no wait of their own: this arrives and waits,
    /// like [`arrive_and_wait`](Self::arrive_and_wait) (CuTe DSL warns and
    /// does the same).
    #[track_caller]
    pub fn wait(&self) {
        self.arrive_and_wait();
    }

    /// Same as [`wait`](Self::wait) (CuTe DSL emits the same `bar.sync`).
    #[track_caller]
    pub fn wait_unaligned(&self) {
        self.arrive_and_wait();
    }

    /// Arrive and wait until `num_threads` threads have arrived (`bar.sync`).
    #[track_caller]
    pub fn arrive_and_wait(&self) {
        arrive_and_wait(self.barrier_id.value(), self.num_threads.value());
    }

    /// Same as [`arrive_and_wait`](Self::arrive_and_wait).
    #[track_caller]
    pub fn sync(&self) {
        self.arrive_and_wait();
    }

    /// Not supported by named barriers (CuTe DSL raises
    /// `NotImplementedError`): always panics.
    #[track_caller]
    pub fn arrive_and_drop(&self) {
        panic!("cute: NamedBarrier::arrive_and_drop is not supported (a hardware named barrier cannot drop a thread)");
    }
}

/// Arrive at named barrier `barrier_id` of `num_threads` threads without
/// waiting (`bar.arrive`; CuTe DSL's `cutlass.pipeline.arrive`).
#[track_caller]
pub fn arrive(barrier_id: impl IntoValue<i32>, num_threads: impl IntoValue<i32>) {
    crate::arch::barrier_arrive(barrier_id, num_threads);
}

/// [`arrive`] with the unaligned `barrier.arrive`, for code where not every
/// thread of the CTA reaches the same instruction.
#[track_caller]
pub fn arrive_unaligned(barrier_id: impl IntoValue<i32>, num_threads: impl IntoValue<i32>) {
    crate::arch::barrier_arrive_unaligned(barrier_id, num_threads);
}

/// Named barriers have no separate wait: arrives and waits, like
/// [`arrive_and_wait`].
#[track_caller]
pub fn wait(barrier_id: impl IntoValue<i32>, num_threads: impl IntoValue<i32>) {
    arrive_and_wait(barrier_id, num_threads);
}

/// Same as [`wait`].
#[track_caller]
pub fn wait_unaligned(barrier_id: impl IntoValue<i32>, num_threads: impl IntoValue<i32>) {
    arrive_and_wait(barrier_id, num_threads);
}

/// Arrive at named barrier `barrier_id` and wait until `num_threads`
/// threads have arrived (`bar.sync`; CuTe DSL's
/// `cutlass.pipeline.arrive_and_wait`).
#[track_caller]
pub fn arrive_and_wait(barrier_id: impl IntoValue<i32>, num_threads: impl IntoValue<i32>) {
    crate::arch::barrier(barrier_id, num_threads);
}

/// Wait at named barrier `barrier_id` for every thread of the CTA
/// (`bar.sync id`; CuTe DSL's `cutlass.pipeline.sync`). `sync(0)` is
/// [`sync_threads`](crate::sync_threads).
#[track_caller]
pub fn sync(barrier_id: impl IntoValue<i32>) {
    named_barrier(BarrierKind::Sync, barrier_id.into_value(), None);
}

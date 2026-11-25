// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's PipelineConfig and
// TileSchedulerConfig (cutlass.experimental.task_scheduling.resources, NVIDIA
// EULA); the code is written independently.

//! Descriptions of the pipeline a resource hands its stages over with
//! ([`PipelineConfig`]) and of the tile scheduler a work queue wraps
//! ([`TileSchedulerConfig`]). Both are plain data, made outside a kernel if
//! need be, and materialised by [`Resource::create`](super::Resource::create).

use std::fmt;

use crate::layout::Layout;
use crate::pipeline::{CooperativeGroup, PipelineOp};
use crate::tensor::{Ptr, recast_ptr};
use crate::utils::{ClcDynamicPersistentTileSchedulerParams, PersistentTileSchedulerParams};
use crate::value::Numeric;

use super::enums::{PipelineType, SignalingThreads, TileSchedulerType};

/// The strides by which the four pipeline roles advance their stage, in
/// the order (producer acquire, producer commit, consumer wait, consumer
/// release). A stride `s > 1` lets `s` lanes (warps of a task, or the
/// domains of several tasks) share one ring: lane `i` starts at stage `i`
/// and moves on by `s`, so `s` must divide the stage count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InterleaveStride(pub [usize; 4]);

impl InterleaveStride {
    /// The same stride for every role.
    pub fn uniform(stride: usize) -> InterleaveStride {
        InterleaveStride([stride; 4])
    }

    /// The producer-acquire stride.
    pub fn producer_acquire(&self) -> usize {
        self.0[0]
    }

    /// The producer-commit stride.
    pub fn producer_commit(&self) -> usize {
        self.0[1]
    }

    /// The consumer-wait stride.
    pub fn consumer_wait(&self) -> usize {
        self.0[2]
    }

    /// The consumer-release stride.
    pub fn consumer_release(&self) -> usize {
        self.0[3]
    }

    /// The largest of the four.
    pub fn max(&self) -> usize {
        self.0.into_iter().max().unwrap_or(1)
    }

    /// Whether any role strides by more than one stage.
    pub fn is_interleaved(&self) -> bool {
        self.max() > 1
    }
}

impl Default for InterleaveStride {
    fn default() -> Self {
        InterleaveStride::uniform(1)
    }
}

impl From<usize> for InterleaveStride {
    fn from(s: usize) -> Self {
        InterleaveStride::uniform(s)
    }
}

impl From<[usize; 4]> for InterleaveStride {
    fn from(s: [usize; 4]) -> Self {
        InterleaveStride(s)
    }
}

/// A [`PipelineConfig`] that cannot be materialised, from
/// [`PipelineConfig::validate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Everything a resource needs to build its pipeline (CuTe DSL's
/// `PipelineConfig`): the kind of pipeline, the stage count, the bytes a
/// TMA or CLC stage expects, the two cooperative groups, and optional
/// barrier storage, cluster layout, signalling threads, advance policy and
/// interleaving.
///
/// Make one with a `create_*_pipeline_cfg` constructor, then adjust the
/// optional fields with the `with_*` methods. [`validate`](Self::validate)
/// checks the combination; materialising
/// ([`Resource::create`](super::Resource::create)) panics on an invalid one.
///
/// ```
/// use cute::pipeline::{Agent, CooperativeGroup};
/// use cute::task_scheduling::{PipelineConfig, PipelineType};
///
/// // 02_copy_tma.py's SMEM pipeline: one TMA thread, four epilogue warps.
/// let cfg = PipelineConfig::create_tma_async_pipeline_cfg(
///     1,
///     128 * 2,
///     CooperativeGroup::new(Agent::Thread, 1),
///     CooperativeGroup::new(Agent::Thread, 4 * 32),
/// );
/// assert_eq!(cfg.pipeline_type, PipelineType::TmaAsync);
/// assert!(cfg.validate().is_ok());
/// assert!(cfg.clone().with_interleave_stride(2).validate().is_err()); // 2 does not divide 1 stage
/// ```
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// The pipeline depth.
    pub num_stages: usize,
    /// The bytes one stage's transaction delivers (TMA and CLC pipelines;
    /// 0 otherwise).
    pub num_bytes: u32,
    /// The threads that fill a stage.
    pub producer_group: CooperativeGroup,
    /// The threads that drain a stage.
    pub consumer_group: CooperativeGroup,
    /// Which pipeline this describes.
    pub pipeline_type: PipelineType,
    /// Shared memory for the `2 * num_stages` mbarriers; `None` lets the
    /// TaskManager's [`SmemAllocator`](super::SmemAllocator) place them (or,
    /// without one, the pipeline itself).
    pub barrier_ptr: Option<Ptr<u64>>,
    /// The cluster layout (V, M, N, K) for the UMMA and CLC pipelines.
    pub cta_layout_vmnk: Option<Layout>,
    /// Which threads acquire and commit.
    pub producer_signaling_threads: SignalingThreads,
    /// Which threads release (and wait, unless
    /// [`consumer_wait_signaling_threads`](Self::consumer_wait_signaling_threads) is set).
    pub consumer_signaling_threads: SignalingThreads,
    /// Which threads wait, when they differ from those that release.
    pub consumer_wait_signaling_threads: Option<SignalingThreads>,
    /// The operation the producers of a thread-produced pipeline
    /// (`AsyncAsync`, `AsyncUmma`) commit with: `AsyncThread`, or
    /// `AsyncLoad` for `cp.async` producers.
    pub producer_op: PipelineOp,
    /// Advance the consumer's stage when it waits rather than when it
    /// releases (the release then follows a lagging state).
    pub advance_on_wait: bool,
    /// Advance the producer's stage when it acquires rather than when it
    /// commits.
    pub advance_on_acquire: bool,
    /// Bytes each producer warp of each CTA delivers, when every CTA routes
    /// its TMA completion to the leader's full barrier.
    pub num_bytes_per_warp_per_cta: Option<u32>,
    /// The multicast mode of a TMA pipeline in a cluster.
    pub mcast_mode_mn: (u32, u32),
    /// The per-role stage strides.
    pub interleave_stride: InterleaveStride,
}

impl PipelineConfig {
    fn base(
        pipeline_type: PipelineType,
        num_stages: usize,
        num_bytes: u32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
    ) -> PipelineConfig {
        PipelineConfig {
            num_stages,
            num_bytes,
            producer_group,
            consumer_group,
            pipeline_type,
            barrier_ptr: None,
            cta_layout_vmnk: None,
            producer_signaling_threads: SignalingThreads::ALL,
            consumer_signaling_threads: SignalingThreads::ALL,
            consumer_wait_signaling_threads: None,
            producer_op: PipelineOp::AsyncThread,
            advance_on_wait: false,
            advance_on_acquire: false,
            num_bytes_per_warp_per_cta: None,
            mcast_mode_mn: (1, 1),
            interleave_stride: InterleaveStride::default(),
        }
    }

    /// Threads fill and drain the stages (`AsyncAsync`).
    pub fn create_async_async_pipeline_cfg(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Option<Layout>,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk,
            ..Self::base(PipelineType::AsyncAsync, num_stages, 0, producer_group, consumer_group)
        }
    }

    /// TMA loads of `num_bytes` per stage fill the stages, threads drain
    /// them (`TmaAsync`).
    pub fn create_tma_async_pipeline_cfg(
        num_stages: usize,
        num_bytes: u32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
    ) -> PipelineConfig {
        Self::base(PipelineType::TmaAsync, num_stages, num_bytes, producer_group, consumer_group)
    }

    /// TMA loads fill the stages, `tcgen05` MMAs drain them (`TmaUmma`).
    pub fn create_tma_umma_pipeline_cfg(
        num_stages: usize,
        num_bytes: u32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Layout,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk: Some(cta_layout_vmnk),
            ..Self::base(PipelineType::TmaUmma, num_stages, num_bytes, producer_group, consumer_group)
        }
    }

    /// `tcgen05` MMAs fill the stages (accumulators), threads drain them
    /// (`UmmaAsync`).
    pub fn create_umma_async_pipeline_cfg(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Layout,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk: Some(cta_layout_vmnk),
            ..Self::base(PipelineType::UmmaAsync, num_stages, 0, producer_group, consumer_group)
        }
    }

    /// Threads fill the stages, `tcgen05` MMAs drain them (`AsyncUmma`).
    pub fn create_async_umma_pipeline_cfg(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Layout,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk: Some(cta_layout_vmnk),
            ..Self::base(PipelineType::AsyncUmma, num_stages, 0, producer_group, consumer_group)
        }
    }

    /// `tcgen05` MMAs on both sides (`UmmaUmma`).
    pub fn create_umma_umma_pipeline_cfg(
        num_stages: usize,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Layout,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk: Some(cta_layout_vmnk),
            ..Self::base(PipelineType::UmmaUmma, num_stages, 0, producer_group, consumer_group)
        }
    }

    /// Cluster launch control responses of `num_bytes` fill the stages,
    /// threads drain them (`ClcFetchAsync`).
    pub fn create_clc_fetch_async_pipeline_cfg(
        num_stages: usize,
        num_bytes: u32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        cta_layout_vmnk: Option<Layout>,
    ) -> PipelineConfig {
        PipelineConfig {
            cta_layout_vmnk,
            ..Self::base(PipelineType::ClcFetchAsync, num_stages, num_bytes, producer_group, consumer_group)
        }
    }

    /// With barrier storage of `2 * num_stages` mbarriers.
    pub fn with_barrier_ptr(mut self, barrier_ptr: Ptr<u64>) -> Self {
        self.barrier_ptr = Some(barrier_ptr);
        self
    }

    /// With a cluster layout.
    pub fn with_cta_layout_vmnk(mut self, layout: Layout) -> Self {
        self.cta_layout_vmnk = Some(layout);
        self
    }

    /// With the threads that signal each side.
    pub fn with_signaling_threads(mut self, producer: SignalingThreads, consumer: SignalingThreads) -> Self {
        self.producer_signaling_threads = producer;
        self.consumer_signaling_threads = consumer;
        self
    }

    /// With the threads that wait, when they differ from those that release.
    pub fn with_consumer_wait_signaling_threads(mut self, threads: SignalingThreads) -> Self {
        self.consumer_wait_signaling_threads = Some(threads);
        self
    }

    /// With the producers' commit operation (`AsyncAsync` and `AsyncUmma`
    /// only).
    pub fn with_producer_op(mut self, op: PipelineOp) -> Self {
        self.producer_op = op;
        self
    }

    /// Advance the consumer on wait.
    pub fn with_advance_on_wait(mut self, on: bool) -> Self {
        self.advance_on_wait = on;
        self
    }

    /// Advance the producer on acquire.
    pub fn with_advance_on_acquire(mut self, on: bool) -> Self {
        self.advance_on_acquire = on;
        self
    }

    /// With per-role stage strides (one number for all four roles, or four).
    pub fn with_interleave_stride(mut self, stride: impl Into<InterleaveStride>) -> Self {
        self.interleave_stride = stride.into();
        self
    }

    /// With the multicast mode of a TMA pipeline.
    pub fn with_mcast_mode_mn(mut self, mode: (u32, u32)) -> Self {
        self.mcast_mode_mn = mode;
        self
    }

    /// With leader-routed TMA completion of `bytes` per producer warp per CTA.
    pub fn with_num_bytes_per_warp_per_cta(mut self, bytes: u32) -> Self {
        self.num_bytes_per_warp_per_cta = Some(bytes);
        self
    }

    /// The four interleave strides (producer acquire, producer commit,
    /// consumer wait, consumer release).
    pub fn interleave_strides(&self) -> [usize; 4] {
        self.interleave_stride.0
    }

    /// The threads that wait on the consumer side.
    pub fn consumer_wait_threads(&self) -> SignalingThreads {
        self.consumer_wait_signaling_threads.unwrap_or(self.consumer_signaling_threads)
    }

    /// Check the combination of fields: at least one stage; valid
    /// signalling sets; a producer operation of `AsyncThread` or
    /// `AsyncLoad`, changed only for `AsyncAsync` and `AsyncUmma`; positive
    /// strides dividing the stage count; split acquire/commit strides only
    /// with `advance_on_acquire` and split wait/release strides only with
    /// `advance_on_wait`; a cluster layout for the UMMA pipelines.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let err = |m: String| Err(ConfigError(format!("PipelineConfig ({:?}): {m}", self.pipeline_type)));
        if self.num_stages == 0 {
            return err("the pipeline has no stages".into());
        }
        for (what, s) in [
            ("producer signalling threads", Some(self.producer_signaling_threads)),
            ("consumer signalling threads", Some(self.consumer_signaling_threads)),
            ("consumer wait signalling threads", self.consumer_wait_signaling_threads),
        ] {
            if let Some(s) = s
                && !s.is_valid()
            {
                return err(format!("{s:?} is not a valid set of {what}"));
            }
        }
        match self.producer_op {
            PipelineOp::AsyncThread => {}
            PipelineOp::AsyncLoad
                if matches!(self.pipeline_type, PipelineType::AsyncAsync | PipelineType::AsyncUmma) => {}
            PipelineOp::AsyncLoad => {
                return err("only AsyncAsync and AsyncUmma pipelines take another producer operation".into());
            }
            op => return err(format!("producers commit with AsyncThread or AsyncLoad, not {op:?}")),
        }
        let roles = ["producer acquire", "producer commit", "consumer wait", "consumer release"];
        for (role, s) in roles.iter().zip(self.interleave_stride.0) {
            if s == 0 {
                return err(format!("the {role} stride is 0"));
            }
            if !self.num_stages.is_multiple_of(s) {
                return err(format!("the {role} stride {s} does not divide the {} stages", self.num_stages));
            }
        }
        let [pa, pc, cw, cr] = self.interleave_stride.0;
        if pa != pc && !self.advance_on_acquire {
            return err(format!(
                "acquire and commit stride differently ({pa} and {pc}), which needs advance_on_acquire"
            ));
        }
        if cw != cr && !self.advance_on_wait {
            return err(format!("wait and release stride differently ({cw} and {cr}), which needs advance_on_wait"));
        }
        let needs_layout = matches!(
            self.pipeline_type,
            PipelineType::TmaUmma | PipelineType::UmmaAsync | PipelineType::AsyncUmma | PipelineType::UmmaUmma
        );
        if needs_layout && self.cta_layout_vmnk.is_none() {
            return err("the pipeline needs a cluster layout (cta_layout_vmnk)".into());
        }
        Ok(())
    }
}

/// The tile scheduler a [`WorkQueue`](super::WorkQueue) creates (CuTe
/// DSL's `TileSchedulerConfig`): its kind and its parameters, plus the
/// shared-memory response buffer (16 bytes per pipeline stage) of the
/// cluster launch control scheduler.
#[derive(Clone, Debug)]
pub enum TileSchedulerConfig {
    /// A [`StaticPersistentTileScheduler`](crate::utils::StaticPersistentTileScheduler).
    StaticPersistent(PersistentTileSchedulerParams),
    /// A [`ClcDynamicPersistentTileScheduler`](crate::utils::ClcDynamicPersistentTileScheduler).
    ClcDynamicPersistent { params: ClcDynamicPersistentTileSchedulerParams, response_ptr: Ptr<i32> },
}

impl TileSchedulerConfig {
    /// The static persistent scheduler of `params`.
    pub fn create_static_persistent_tile_scheduler_params(params: PersistentTileSchedulerParams) -> Self {
        TileSchedulerConfig::StaticPersistent(params)
    }

    /// The cluster launch control scheduler of `params`, whose responses
    /// land at `response_ptr` (16-byte aligned shared memory, 16 bytes per
    /// stage of the work queue's pipeline).
    pub fn create_clc_dynamic_persistent_tile_scheduler_params<T: Numeric>(
        params: ClcDynamicPersistentTileSchedulerParams,
        response_ptr: &Ptr<T>,
    ) -> Self {
        TileSchedulerConfig::ClcDynamicPersistent { params, response_ptr: recast_ptr::<i32, T>(response_ptr, None) }
    }

    /// The kind of scheduler.
    pub fn tile_scheduler_type(&self) -> TileSchedulerType {
        match self {
            TileSchedulerConfig::StaticPersistent(_) => TileSchedulerType::StaticPersistent,
            TileSchedulerConfig::ClcDynamicPersistent { .. } => TileSchedulerType::ClcDynamicPersistent,
        }
    }
}

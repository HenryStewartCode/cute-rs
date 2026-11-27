//! CuTe DSL's experimental fragment-free layer, `cutlass.cute.experimental`
//! ("cute_ext").
//!
//! In cute_ext a kernel allocates whole buffers and calls copy, partition,
//! dot and pipeline operations on whole tensors; a compiler pipeline behind
//! CuTe DSL's `@cute.experimental.kernel` adds the predicates, the TMA
//! descriptors and the TMEM allocation. cute-rs has no separate entry
//! point: the functions here are build-time library functions used inside
//! an ordinary [`Kernel::build`](crate::Kernel::build) closure, over the
//! machinery of the rest of the crate, and builder hooks stand in for the
//! compiler passes:
//!
//! - Automatic predication. Every global-memory tensor (a kernel argument,
//!   or one made on a global pointer) is a *predication origin*: the views
//!   made from it (slices, divides, compositions, [`partition`],
//!   `partition_s`/`partition_d` of a tiled copy, `partition_a/b/c` of a
//!   tiled MMA, `local_tile`, `domain_offset`) remember the operations that
//!   made them, and the copies of this module replay them on an identity
//!   tensor over the origin's shape to predicate every element with
//!   [`elem_less`](crate::elem_less). A view whose coordinates are all
//!   statically in bounds gets no predicate. [`predicated_tensor_origin`]
//!   starts a new origin. A view made by an operation that does not record
//!   itself (a coalesce, a product, `Tileable::with_layout`) has lost its
//!   origin, and the copies here refuse it until it is given a new one.
//! - TMEM. Each [`allocate`] in tensor memory takes a column range of one
//!   pooled allocation, which a prologue inserted at the start of the kernel
//!   body allocates (warp 0, `tcgen05.alloc`) and an epilogue inserted at
//!   its end frees.
//!
//! The submodules mirror CuTe DSL's: [`core`] (the operation types),
//! [`memory`] (allocate, copy, predicated_tensor_origin), [`algorithm`]
//! (partition, partition_and_copy, simt_auto_vec_copy, warp_reduce,
//! elementwise), [`math`] (dot and its block-scaled and sparse forms), and
//! [`tma`] (tma_load, tma_store, tma_reduce_store), [`pipeline_ops`] (the
//! low-level pipeline operations: create_pipeline, producer_acquire, ...,
//! and the circular-buffer view), [`pipeline`] (TMAStorePipeline and the
//! convenience pipeline classes: GenericPipeline, TMAToUMMAPipeline, ...),
//! [`host_runtime`] (the device workspace), [`cluster_mailbox`] (typed
//! DSMEM messages to the pair's peer CTA), [`accumulator_buffering`] (the
//! TMEM accumulator planner), [`utils`] (the GEMM building blocks:
//! CTA-V maps, TMEM layouts, `mainloop_mma` and `epilogue_tma_store`), and
//! the hooks of cute_ext's IKET tracing: [`iket`].
//!
//! The semantics follow CuTe DSL 4.8.1's documented behaviour and its BSD-3
//! examples; the implementation is cute-rs's own.

pub mod accumulator_buffering;
pub mod algorithm;
pub mod cluster_mailbox;
pub mod core;
pub mod host_runtime;
pub mod iket;
pub mod math;
pub mod memory;
pub mod pipeline;
pub mod pipeline_ops;
pub mod tma;
pub mod utils;

/// The pipeline state of cute_ext's pipelines (CuTe DSL's
/// `cute.experimental.PipelineState`), cute-rs's [`crate::pipeline::PipelineState`].
pub use crate::pipeline::PipelineState;
pub use accumulator_buffering::{
    AccumulatorBufferingKind, AccumulatorBufferingOptions, AccumulatorBufferingPlan, AccumulatorDrain,
    AccumulatorOverlapPolicy, AccumulatorPlanSelection, TmemAllocationRequest, plan_accumulator_buffering,
    plan_accumulator_buffering_columns,
};
pub use algorithm::{ElementwiseTensors, elementwise, partition, partition_and_copy, simt_auto_vec_copy, warp_reduce};
pub use cluster_mailbox::{
    B32Payload, B32PayloadCodec, CTA_PAIR_PEER_LOCALE, ClusterMailbox, DecodedPayload, PayloadField, b32_payload,
    dsmem_payload,
};
pub use core::OperationType;
pub use host_runtime::{
    AllocationRequirement, QueryDeviceWorkspaceFunc, WorkspaceSize, device_workspace_cta_slot, request_device_workspace,
};
pub use math::{AsTiledMma, dot, dot_block_scaled, dot_sparse};
pub use memory::{allocate, allocate_with, copy, predicated_tensor_origin};
pub use pipeline::{
    AsGenericPipeline, AsyncToUMMAPipeline, CLCPipeline, GenericPipeline, GenericPipelineBase,
    GroupedGemmSchedulerPipeline, TMAStorePipeline, TMAToAsyncPipeline, TMAToUMMACircularPipeline, TMAToUMMAPipeline,
    UMMAtoAsyncPipeline,
};
pub use pipeline_ops::{
    Arrival, ArrivalCountOptions, ArrivalMask, CircularBufferPipelineState, ClassPipeline, MbarrierOwner,
    OperationTypes, PipelineClass, PipelineLocale, RawPipeline, Side, StageToken,
    circular_buffer_pipeline_advance_iterator, circular_buffer_pipeline_consume,
    circular_buffer_pipeline_consumer_release, consumer_release, consumer_release_elect_one_sync, consumer_tail,
    consumer_try_wait, consumer_wait, consumer_wait_with, create_circular_buffer_pipeline, create_pipeline,
    create_pipeline_state_at, create_pipeline_with_mask, get_arrival_count, get_mbarrier, get_pipeline_consume_phase,
    get_pipeline_consume_stage, get_pipeline_produce_phase, get_pipeline_produce_stage, mbarrier_expect_tx,
    pipeline_advance_iterator, pipeline_tail, producer_acquire, producer_acquire_with, producer_commit,
    producer_try_acquire,
};
// Re-exports a deprecated name, which CuTe's `cute_ext` also exposes.
#[allow(deprecated)]
pub use tma::tma_load_multicast;
pub use tma::{
    TmaLoadOptions, TmaStoreOptions, tma_load, tma_load_with, tma_reduce_store, tma_reduce_store_with, tma_store,
    tma_store_with,
};
pub use utils::{
    EpilogueTmaStoreOptions, MmaOperand, epilogue_tma_store, epilogue_tma_store_drain, epilogue_tma_store_with,
    get_cta_v_map_ab, get_cta_v_map_c, is_preferred_cluster, mainloop_mma, mainloop_mma_sm120, make_t2r_rmem_layout,
    make_tmem_layout_a, make_tmem_layout_acc,
};

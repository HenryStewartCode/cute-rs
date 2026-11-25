//! Low-level primitives (CuTe DSL's `cutlass.experimental.primitives` and
//! `cutlass.base_dsl.swizzle`).
//!
//! * [`descriptors`]: typed bit-level encoders for the operand and
//!   instruction descriptors of `wgmma.mma_async` and `tcgen05.mma`
//!   ([`WgmmaSmemDesc`], [`Tcgen05SmemDesc`], [`Tcgen05InstrDesc`],
//!   [`Tcgen05MxInstrDesc`], [`Tcgen05MxOmmaInstrDesc`]) and the
//!   tensor-memory address ([`TmemAddr`]). The WGMMA and tcgen05 MMA atoms
//!   build their descriptors with these, so there is one encoder for each.
//! * [`swizzle`]: the base DSL's swizzle on shared-memory byte addresses
//!   ([`ByteSwizzle`], CuTe DSL's `cutlass.Swizzle`) with
//!   [`apply_swizzle`], [`load_swizzled`] and [`store_swizzled`].
//!
//! * [`mma`]: the MMA forms no CuTe
//!   atom has: weight-stationary [`tcgen05_mma_ws`] and
//!   [`tcgen05_mma_ws_sp`], [`tcgen05_mma`] with `.collector::a`,
//!   `.ashift` and `scale-input-d`, [`tcgen05_mma_sp_block_scale`] and the
//!   SM120 warp-level [`mma_sp_block_scale`].
//!
//! * [`nvvm`]: the operations no
//!   `cute::arch` function covers (`bar.red`, the rest of the mbarrier
//!   family, `nanosleep`, `pmevent`, `exit`, `dp4a`, `st.bulk`,
//!   `tcgen05.shift`, ...) and the table from each primitives name to the
//!   cute-rs function that does its job.

pub mod descriptors;
pub mod mma;
pub mod nvvm;
pub mod swizzle;

pub use descriptors::{
    Bits, MxScaleFormat, Tcgen05InputFormat, Tcgen05InstrDesc, Tcgen05InstrDescFields, Tcgen05MxInstrDesc,
    Tcgen05MxInstrDescFields, Tcgen05MxOmmaInstrDesc, Tcgen05MxOmmaInstrDescFields, Tcgen05OutputFormat,
    Tcgen05SmemDesc, Tcgen05SmemDescFields, Tcgen05SmemSwizzle, TmemAddr, WgmmaSmemDesc, WgmmaSwizzle,
};
pub use mma::{
    InstrDescOperand, MmaBlockScaleKind, MmaOperandA, ScaleOperand, SmemDescOperand, Tcgen05MmaCollectorBBuffer,
    Tcgen05MmaCollectorOp, Tcgen05MmaKind, Tcgen05MmaOptions, Tcgen05MmaScaleVec, Tcgen05MmaSpBlockScaleOptions,
    Tcgen05MmaWsOptions, TmemOperand, mma_sp_block_scale, tcgen05_mma, tcgen05_mma_sp_block_scale, tcgen05_mma_ws,
    tcgen05_mma_ws_sp, tmem_frg_ws_layout,
};
pub use swizzle::{ByteSwizzle, ScalarOrVector, apply_swizzle, load_swizzled, store_swizzled};
// The four names that cute's crate root also has (with other signatures) stay in
// `prims::nvvm`, so that `use cute::prelude::*; use cute::prims::*;` has no ambiguity.
pub use crate::ir::FenceScope;
pub use nvvm::{
    BarrierRedux, CTAGroup, DotAccumulateType, MBarrierWait, MlirOp, StateOrPhase, barrier_cta_red,
    barrier_cta_red_and, barrier_cta_red_or, barrier_cta_red_popc, breakpoint, cvta_to, dot_accumulate_2way,
    dot_accumulate_4way, elect_sync, exit, fence_sc_cluster, mbarrier_arrive_drop, mbarrier_arrive_drop_expect_tx,
    mbarrier_arrive_drop_nocomplete, mbarrier_arrive_nocomplete, mbarrier_complete_tx, mbarrier_inval,
    mbarrier_try_wait_parity, mbarrier_try_wait_timelimit, memory_barrier, mov_b32, nanosleep, nvvm_op, pmevent,
    pmevent_mask, print_runtime, ptx_add_comment, st_bulk, tcgen05_shift,
};

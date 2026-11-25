//! # cute-rs
//!
//! A CuTe-style GPU kernel DSL for Rust.
//!
//! Kernels are written as ordinary Rust inside [`Kernel::build`]. They are
//! built from the same pieces as NVIDIA's CuTe and CuTe DSL:
//!
//! * [`Layout`]s and the layout algebra (composition, complement, divides,
//!   products, inverses, swizzles), evaluated at build time on extents and
//!   strides that are static or runtime values (CuTe DSL's `(?,?):(?,1)`),
//!   with [`size`], [`shape`] and [`cosize`] queries;
//! * [`Tensor`]s over global, shared and register memory, with tiling and
//!   partitioning ([`local_tile`], [`local_partition`]), whose extents in
//!   global memory may be runtime values ([`param_tensor`] of
//!   `Layout::row_major((m, n))`), with the same [`size`], [`shape`],
//!   [`cosize`], [`rank`] and [`depth`] queries;
//! * copy atoms and [`TiledCopy`] (vectorised loads/stores, `cp.async`,
//!   `ldmatrix`) and MMA atoms and [`TiledMma`] (scalar FMA, SM80
//!   `mma.sync` on f16, bf16, TF32 and f64), with [`copy`] and [`gemm`];
//! * runtime values with operator overloading ([`Int32`], [`Float64`],
//!   [`BFloat16`], ...), runtime control flow ([`if_`], [`for_range`],
//!   [`while_carry`], with [`LoopOptions`] for unroll hints) and mutable
//!   variables that it threads through automatically ([`Var`]);
//! * warp-collective operations (shuffles of every type with segments and
//!   member masks, [`shuffle_sync_op`]; butterfly reductions,
//!   [`warp_reduction`]; `redux.sync`, [`warp_redux_sync`]; votes,
//!   [`vote_any_sync`]; [`match_sync`]; [`sync_warp_with`]), named barriers
//!   ([`barrier`], [`pipeline::NamedBarrier`]), mbarriers
//!   ([`mbarrier_arrive`]), atomics, fences, [`printf`] and [`inline_ptx`];
//! * loads and stores with CuTe DSL's memory orderings and cache hints
//!   ([`load`], [`store`], [`prefetch`], and the copy atoms
//!   [`CopyOp::G2R`] and [`CopyOp::R2G`]), whose flag hand-offs the
//!   simulator's race checker follows, and [`make_cotiled_copy`];
//! * CuTe DSL's integer and bit instructions, each one PTX instruction that
//!   the simulator runs bit for bit: bit fields ([`bfe`], [`bfi`]),
//!   three-input logic ([`lop3`]), funnel shifts ([`shf`]), byte permutes
//!   ([`prmt`]), high and widening multiplies ([`mul_hi`], [`mul_wide`]),
//!   24-bit multiplies ([`mul24`], [`mad24`]) and saturating arithmetic
//!   ([`add_sat_int`], [`sub_sat_int`]), and its bit counts ([`clz`],
//!   [`popc`], [`brev`], [`bfind`], [`log2_of_pow2_int`]);
//! * CuTe DSL's extended-precision arithmetic, chained through PTX's carry
//!   flag ([`add_cc`], [`addc`], [`sub_cc`], [`subc`], [`mad_cc`], [`madc`],
//!   and the carry-in-and-out forms [`addc_cc`], [`subc_cc`] and
//!   [`madc_cc`]), which the simulator runs with a carry flag per thread;
//! * hand-written PTX with CuTe DSL's named operands ([`InlinePtx`]), raw
//!   inline assembly with a constraint string ([`InlineAsm`]) and MLIR
//!   operations written as text ([`MlirOp`]), for what the DSL lacks, and
//!   custom operations built from them (`#[track_caller]` functions, CuTe
//!   DSL's `dsl_user_op`);
//! * raw memory ([`Ptr`], [`make_tensor`], [`alloc_smem`], pointers made
//!   from integer addresses and back ([`make_ptr`], [`Ptr::toint`],
//!   [`Ptr::tospace`], [`Ptr::align`], masking), the dynamic
//!   shared memory sized at launch, [`get_dyn_smem`]) and problem sizes
//!   known only at launch ([`param_extent`], as runtime extents or as the
//!   runtime strides of static upper bounds, [`Layout::with_strides`]);
//! * typed kernel entry points ([`Kernel::jit`], CuTe DSL's `@cute.jit` with
//!   runtime arguments by default) and the kernel argument protocol
//!   ([`params`], [`KernelParam`]): tensor specs ([`TensorSpec`]), host
//!   tensors with dynamic modes ([`HostTensor::mark_layout_dynamic`]),
//!   symbolic integers ([`SymInt`]), layouts, `Option`s, tuples, structs
//!   ([`kernel_params!`]) and native structs as arguments, checked at launch;
//! * CuTe DSL's `cute.math` library ([`math`]): exponentials, logarithms,
//!   trigonometric, hyperbolic and error functions, powers and the
//!   arithmetic forms with explicit rounding, `ftz` and `approx` control.
//!   Precise mode is built from exactly rounded operations (no libdevice),
//!   so the GPU computes the simulator's bits; `fastmath` uses the hardware
//!   approximations.
//!
//! A built [`Kernel`] can be lowered to MLIR ([`Kernel::to_mlir`]), compiled
//! to PTX with an installed `mlir-opt` ([`Kernel::compile_with`] and
//! [`CompileOptions`], with an optimization level and line information that
//! maps the PTX back to the Rust source), and run on the CPU by a checking
//! simulator ([`Kernel::simulate`]) that reports data races, missing
//! `cp.async` waits, out-of-bounds and misaligned accesses with the Rust
//! source location of the offending operation.
//!
//! ```
//! use cute::prelude::*;
//!
//! const N: usize = 1024;
//! let kernel = Kernel::builder("scale").block_dim([256, 1, 1]).build(|| {
//!     let x = param_tensor::<f32>("x", Layout::col_major(N));
//!     let y = param_tensor::<f32>("y", Layout::col_major(N));
//!     let i = block_idx().x * 256 + thread_idx().x;
//!     y.store(i, x.load(i) * 2.0);
//! });
//!
//! let mut args = vec![KernelArg::f32((0..N).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; N])];
//! kernel.simulate(LaunchConfig::new([4, 1, 1], [256, 1, 1]), &mut args).unwrap();
//! assert_eq!(args[1].as_f32()[10], 20.0);
//! ```

mod abi;
pub mod aot;
pub mod arch;
pub mod atom;
pub mod backend;
// m10-s2t-copies: CuTe DSL's cutlass.block.
pub mod block;
mod builder;
pub mod control;
mod convert;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod diagnostics;
mod elementwise;
#[path = "cute_ext/mod.rs"]
pub mod experimental;
pub mod external;
mod fast_divmod;
pub mod host;
pub mod ir;
pub mod layout;
pub mod log;
mod masked;
pub mod math;
pub mod memory;
mod numeric;
pub mod nvgpu;
pub mod params;
pub mod pipeline;
pub mod prims;
pub mod print;
pub mod profiling;
mod promote;
mod raw_ptr;
pub mod signature;
pub mod sim;
mod smem_struct;
pub mod tensor;
pub mod tensor_ssa;
// m11-ts-resources: Task Scheduling (resources, pipeline configs, work queues, schedule capture).
pub mod task_scheduling;
pub mod testing;
pub mod tvm_ffi;
pub mod utils;
pub mod validate;
mod value;
pub mod vector;
pub mod viz;
pub mod xla_ffi;

pub use abi::{MAX_CONSTANT_BYTES, StructParam, extern_global, module_global, param_bytes, param_struct, smem_overlay};
pub use aot::{AotError, ExportOptions, ExportReport, ExportedKernel};
#[allow(deprecated)]
pub use arch::atomic_max_float32;
pub use arch::{
    AsmOutputs, AsmPush, InlineAsm, InlinePtx, MlirOp, add_cc, add_with_carry, addc, addc_cc, mad_cc, madc, madc_cc,
    sub_cc, sub_with_borrow, subc, subc_cc,
};
pub use arch::{
    AtomicAddable, AtomicOptions, RedOp, atomic_add, atomic_add_packed, atomic_add_packed_with, atomic_add_with,
    atomic_and, atomic_and_with, atomic_cas, atomic_cas_with, atomic_exch, atomic_exch_with, atomic_fmax,
    atomic_fmax_with, atomic_fmin, atomic_fmin_with, atomic_max, atomic_max_with, atomic_min, atomic_min_with,
    atomic_or, atomic_or_with, atomic_xor, atomic_xor_with, red, red_packed,
};
pub use arch::{
    Dim3, FULL_MASK, MbarrierOptions, MinMaxOptions, ReduxOptions, ShfKind, Shuffle, THREADS_PER_WARPGROUP, WARP_SIZE,
    WARPS_PER_WARPGROUP, Widen, Word, activemask, add_sat_int, barrier, barrier_arrive, barrier_arrive_unaligned, bfe,
    bfi, bfind, block_dim, block_idx, block_in_cluster_dim, block_in_cluster_idx, brev, clock, clock64, cluster_arrive,
    cluster_arrive_aligned, cluster_arrive_relaxed, cluster_arrive_relaxed_aligned, cluster_dim, cluster_idx,
    cluster_sync, cluster_wait, cluster_wait_aligned, clz, cp_async_bulk, cp_async_commit_group, cp_async_wait_all,
    cp_async_wait_group, elect_one, fence_acq_rel_cluster, fence_acq_rel_cta, fence_acq_rel_gpu, fence_acq_rel_sys,
    fmax, fmax_with, fmin, fmin_with, globaltimer, globaltimer_lo, grid_dim, gridid, inline_ptx, inline_ptx_effect,
    inline_ptx_effect_with_sim, inline_ptx_with_sim, lane_idx, lanemask_eq, lanemask_ge, lanemask_gt, lanemask_le,
    lanemask_lt, linear_thread_idx, log2_of_pow2_int, lop3, lop3_lut, mad24, make_warp_uniform, match_sync,
    mbarrier_arrive, mbarrier_arrive_and_expect_tx, mbarrier_arrive_and_expect_tx_with, mbarrier_arrive_with,
    mbarrier_conditional_try_wait, mbarrier_expect_tx, mbarrier_expect_tx_with, mbarrier_init, mbarrier_init_fence,
    mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait, mul_hi, mul_wide, mul24, nsmid, nwarpid, physical_warp_id,
    popc, prmt, prmt_with, shf, shuffle_sync, shuffle_sync_bfly, shuffle_sync_bfly_with, shuffle_sync_down,
    shuffle_sync_down_with, shuffle_sync_op, shuffle_sync_up, shuffle_sync_up_with, shuffle_sync_with, smid,
    sub_sat_int, sync_threads, sync_warp, sync_warp_with, thread_idx, vote_all_sync, vote_any_sync, vote_ballot_sync,
    vote_uni_sync, warp_idx, warp_reduction, warp_reduction_max, warp_reduction_sum, warp_redux_sync,
    warp_redux_sync_with, warpsize,
};
pub use arch::{
    FenceProxyKind, FenceProxyOptions, Fp6Format, cp_async_bulk_commit_group, cp_async_bulk_global_shared_cta,
    cp_async_bulk_wait_group, cvt_f4e2m1_f16, cvt_f4e2m1x2_to_f16x2, cvt_f4e2m1x2_to_f16x2_with, cvt_f4e2m1x4_to_f16x4,
    cvt_f4e2m1x8_to_f16x8, cvt_f6x2_to_f16x2, cvt_f32x2_to_f4e2m1x2, cvt_f32x2_to_f6x2, fence_proxy, fence_proxy_with,
    fence_view_async_shared,
};
pub use arch::{
    Fp8Format, Half2, HalfFloat, HalfOptions, I8ToBf16Path, Int8x2, Int8x4, add_packed_bf16x2, add_packed_f16x2,
    cvt_bf16x2_to_ue8m0x2, cvt_f8x2_to_f16x2, cvt_f16x2_to_f8x2, cvt_f32_bf16, cvt_f32_tf32, cvt_f32x2_bf16x2,
    cvt_f32x2_to_f8x2, cvt_f32x2_to_ue8m0x2, cvt_f32x4_to_fnv8e5m3x4, cvt_i4_bf16, cvt_i4_bf16_intrinsic,
    cvt_i4_mxf8_intrinsic, cvt_i4x2_to_bf16x2, cvt_i4x4_to_bf16x4, cvt_i4x8_to_bf16x8, cvt_i4x8_to_mxf8x8, cvt_i8_bf16,
    cvt_i8_bf16_intrinsic, cvt_i8_bf16_intrinsic_with, cvt_i8x2_to_bf16x2, cvt_i8x2_to_f32x2, cvt_i8x4_to_bf16x4,
    cvt_i8x4_to_f32x4, cvt_ue8m0x2_to_bf16x2, exp2_packed_bf16x2, exp2_packed_f16x2, fma_bf16, fma_f16,
    fma_packed_bf16x2, fma_packed_f16x2, sext_unpacked_i4_i8_intrinsic, sext_unpacked_i4x4_to_i8x4, sub_packed_bf16x2,
    sub_packed_f16x2,
};
// m10-ldred-packed-math: packed f32x2 arithmetic and the Rubin mixed-precision forms.
#[allow(deprecated)]
pub use arch::exp_packed_f32x2;
pub use arch::{
    F32x2, PackedF32Options, add_packed_bf16x2_f32x2_f32x2, add_packed_f16x2_f32x2_f32x2, add_packed_f32x2,
    add_packed_f32x2_bf16x2_f32x2, add_packed_f32x2_f16x2_f32x2, fma_packed_f32x2, fma_packed_f32x2_bf16x2_f32x2_f32x2,
    fma_packed_f32x2_f16x2_f32x2_f32x2, mul_packed_f32x2, sub_packed_bf16x2_f32x2_f32x2, sub_packed_f16x2_f32x2_f32x2,
    sub_packed_f32x2, sub_packed_f32x2_bf16x2_f32x2, sub_packed_f32x2_f16x2_f32x2,
};
pub use arch::{
    LoadOptions, PrefetchOptions, SharedSpace, StoreOptions, load, load_vector, prefetch, store, store_vector,
};
pub use arch::{
    aggr_smem_size, block_idx_in_cluster, cluster_size, cp_async_mbarrier_arrive, cp_async_mbarrier_arrive_noinc,
    dynamic_smem_size, get_dyn_smem_size, total_smem_size,
};
pub use arch::{griddepcontrol_launch_dependents, griddepcontrol_wait};
pub use atom::copy::{
    AsCopyAtom, CopyAtom, CopyOp, ThrCopy, TiledCopy, autovec_copy, autovec_copy_with, basic_copy, basic_copy_if, copy,
    copy_if, make_copy_atom, make_predicate, make_tiled_copy, make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_c,
    make_tiled_copy_c_atom, make_tiled_copy_d, make_tiled_copy_s, make_tiled_copy_tv,
};
pub use atom::copy::{
    AutovecOptions, CopyOptions, G2RAttrs, LoadCacheMode, R2GAttrs, StoreCacheMode, copy_atom_call, copy_atom_call_sp,
    copy_with, make_cotiled_copy,
};
pub use atom::mma::{MmaAtom, MmaF16BF16Op, MmaOp, MmaTF32Op, ThrMma, TiledMma, gemm, make_mma_atom, make_tiled_mma};
pub use atom::mma::{
    MmaF16BF16SparseOp, MmaFP8Op, SparseMetadataFormat, gemm_sparse, sparse_compress, sparse_metadata_lane,
};
pub use backend::{
    Arch, Artifact, ArtifactNote, CUTE_DSL_VERSION, CompileError, CompileOptions, CompiledKernel, CudaTools, ENV_PTXAS,
    JitOption, LoweredModule, LoweringPipeline, MlirOptions, OptionError, TargetSupport, Tool, Toolchain, VERSION,
    best_target, driver_ptx_ceiling, jit_options_for_ptxas_flags, ptxas_max_ptx,
};
pub use backend::{CacheStats, CachedKernel, KernelCache, cache_stats};
pub use builder::{Kernel, KernelOptions, target_arch};
pub use control::{
    Carry, Var, for_range, for_range_carry, for_range_carry_with, for_range_with, if_, if_else, if_else_yield,
    while_carry,
};
pub use diagnostics::{AssertionFailure, Diagnostic, Error, ResourceUsage, Severity, SourceLoc};
pub use external::{
    DlTensor, DlpackError, DlpackSource, DlpackTensor, ExternalTensor, FromDlpack, PointerSpec, TensorSpecKey,
    from_dlpack,
};
pub use fast_divmod::{FastDivmod, HostFastDivmod, param_fast_divmod};
pub use ir::{
    AtomicType, BulkCompletion, BulkDirection, DType, LoopOptions, MatchKind, MemScopeKind, MmaKind, PrmtMode,
    ReduxKind, ShuffleKind,
};
pub use ir::{CacheEvictionPriority, CacheOperator, L2PrefetchSize, MemoryOrder, MemoryScope, PrefetchCacheLevel};
pub use layout::{
    BasisGet, BasisMode, BasisScale, E, Ratio, ScaledBasis, StrideLeaf, basis_get, basis_value, make_basis_like,
    make_identity_layout,
};
pub use layout::{
    ComposedLayout, Coord, Crd2Idx, EqualOperand, FnComposedLayout, HasCosize, HasShape, IndexExpr, Inner, IntTuple,
    Layout, LayoutLike, LayoutSource, Recast, Shape, ShapeTiler, SizeInBytes, SliceCoord, StaticLeaf, Stride, Swizzle,
    SwizzlePortion, Tileable, Tiler, Tree, append_ones, blocked_product, ceil_div, coalesce, complement, composition,
    cosize, crd2idx, depth, dice, equal, flat_divide, flat_product, get_nonswizzle_portion, get_swizzle_portion,
    idx2crd, is_congruent, is_major, is_weakly_congruent, leading_dim, left_inverse, logical_divide, logical_product,
    make_composed_layout, make_composed_layout_fn, make_fragment_like_layout, make_layout, make_layout_like,
    make_layout_tv, make_ordered_layout, make_swizzle, max_common_layout, max_common_vector, nullspace, prepend_ones,
    raked_product, rank, recast_layout, right_inverse, round_up, shape, shape_div, size, size_in_bytes,
    size_in_bytes_dtype, tile_to_shape, tile_to_shape_ordered, tile_to_shape_swizzled, tile_to_shape_swizzled_ordered,
    tiled_divide, tiled_product, zipped_divide, zipped_product,
};
pub use math::{MathOptions, RoundingMode};
pub use numeric::{
    F4E2M1FN, F4E2M1FNx2, F6E2M3FN, F6E2M3FNx4, F6E3M2FN, F6E3M2FNx4, Float4E2M1FN, Float4E2M1FNx2, Float6E2M3FN,
    Float6E2M3FNx4, Float6E3M2FN, Float6E3M2FNx4,
};
pub use numeric::{
    F8E4M3, F8E4M3B11FNUZ, F8E4M3FN, F8E5M2, F8E5M3FNU, F8E8M0FNU, Float8E4M3, Float8E4M3B11FNUZ, Float8E4M3FN,
    Float8E5M2, Float8E8M0FNU, FloatNV8E5M3FNU, I4, Int4, NarrowRounding, TF32, TFloat32,
};
pub use nvgpu::{OperandMajorMode, OutputMajorMode, cpasync, warpgroup};
pub use params::{
    ArgError, ArgPack, ComposedSpec, Constexpr, HostTensor, JitBody, KernelParam, LayoutSpec, NativeField,
    NativeLayout, NativeMember, PtrSpec, ScalarSpec, SpecError, SpecLeaf, SpecTuple, SymInt, SymIntKey, SymRem,
    TensorSpec, TypedKernel, make_fake_compact_tensor, make_fake_tensor, pack_args, pack_args_full,
    param_composed_layout, param_layout, param_shape, sym_int, sym_int32, sym_int64, write_back_args,
};
pub use print::{MAX_PRINTF_ARGS, PrintArg, printf, printf_end, printf_values};
pub use profiling::{
    DagHandle, IKET_GLOBAL, IketClock, IketOptions, IketTrace, KernelSymbols, MetaValue, Payload, RangeToken, Scope,
    SymbolRegistry, WarpScope, dag, dump_kernel_symbols, get_symbol_registry, mark, range_end, range_pop, range_push,
    range_start, register_barrier, register_symbol, reset_symbol_registry, scope, sentinel_token,
};
pub use promote::{CmpOperand, CommonType, PROMOTIONS, Promote, true_div};
pub use raw_ptr::{MaskValue, check_pointer_alignment, inttoptr, make_ptr};
pub use signature::{KernelSignature, LaunchValue, ParamKind, ParamSignature, SignatureParseError};
pub use sim::{
    AsyncCompletion, ClusterShapes, KernelArg, LaunchConfig, ResidentClusters, Schedule, SimBuffer, SimError,
    SimModule, SimOptions, SimReport, SimStream, StreamArg,
};
pub use smem_struct::{
    MAX_STRUCT_ALIGN, MemRange, MemberKind, NestedField, RangeField, ScalarField, ScalarMember, StructBuilder,
    StructLayout, StructMember,
};
pub use tensor::get_dyn_smem;
pub use tensor::{
    AddressSpace, FragmentLike, Ident, LayoutOrShape, Ptr, SmemLayout, Tensor, TensorEngine, TensorIterator,
    alloc_smem, domain_offset, elem_less, increment_coord, local_partition, local_partition_proj, local_tile,
    local_tile_proj, make_coord_tensor, make_fragment, make_fragment_like, make_fragment_like_dtype,
    make_fragment_with_layout, make_identity_tensor, make_rmem_tensor, make_smem_tensor, make_smem_tensor_aligned,
    make_tensor, param_extent, param_extent_i64, param_ptr, param_ptr_aligned, param_scalar, param_tensor,
    param_tensor_aligned, recast_ptr, recast_tensor,
};
pub use tensor_ssa::{
    FillValue, PrintTensor, ReductionOp, SELECT_TREE_MAX, SsaOperand, SsaShape, TensorSsa, all_, any_, empty_like,
    full, full_like, maximum, minimum, ones_like, print_tensor, where_, zeros_like,
};
pub use tvm_ffi::TvmFfiFunction;
pub use utils::LayoutEnum;
pub use value::{
    AnyValue, Arith, BF16, BFloat16, Bitwise, Boolean, F16, Float, Float16, Float32, Float64, Int8, Int16, Int32,
    Int64, Int128, Integer, IntoValue, Numeric, Uint8, Uint16, Uint32, Uint64, Uint128, Value, fma, select,
};
pub use value::{Assumable, assume, get_divisibility};
pub use value::{pack, unpack};
pub use vector::{Vector, VectorIndex, VectorOperand, VectorShape, VectorSlice, VectorSliceItem};

// TMA descriptors and host tensor maps (roadmap part m7-tma-descriptor).
pub mod tensor_map;
pub mod tma;
pub use tensor_map::{TensorMap, TensorMapDataFormat, create_tensor_map_tiled, create_tensor_map_tiled_from_view};
pub use tma::{
    TensorMapDataType, TensorMapFloatOOBFill, TensorMapInterleave, TensorMapL2Promotion, TensorMapOptions,
    TensorMapSwizzle, TmaDesc, TmaDescSpec, TmaDescriptor, param_tensormap, tma_descriptor,
};

// stmatrix atoms and warpgroup register reallocation (roadmap part m8-stmatrix-setmaxnreg).
#[allow(deprecated)]
pub use arch::{setmaxregister_decrease, setmaxregister_increase, warpgroup_reg_alloc, warpgroup_reg_dealloc};
pub use ir::StMatrixForm;

// TMA tile loads, cute.copy's keyword arguments and runtime atom fields (roadmap part
// m7-tma-load).
pub use atom::copy::{AuxOperand, CopyArgs, CopyElem, copy_aux};
pub use atom::fields::MmaField;

// TMA stores, reductions, L2 cache hints and prefetches (roadmap part m7-tma-store); multicast
// masks, multicast and peer-CTA copies, st.async (roadmap part m7-cluster-copies).
pub use arch::{prefetch_tensormap, store_async_dsmem};
pub use layout::make_layout_image_mask;
pub use nvgpu::tma_copy::prefetch_atom;
// Warpgroup MMA (roadmap part m8-wgmma-core).
pub use atom::mma::MmaFragmentInput;
pub use ir::WgmmaKind;
// WGMMA variants and distributed shared memory (roadmap parts m8-wgmma-variants, m8-dsmem).
pub use arch::map_dsmem_ptr;
pub use atom::copy::{R2SAttrs, S2RAttrs};

/// Everything needed to write kernels: `use cute::prelude::*;`.
pub mod prelude {
    pub use crate::*;
}

// Several kernels in one module (m6b-multi-kernel).
pub use aot::ExportedModule;
pub use backend::{CompiledModule, KernelModule, ModuleError};
pub use sim::SimKernelModule;

// m11-extern-distributed: external device functions, standalone device functions and
// cutlass.utils.distributed.
pub mod ffi;
pub use ffi::{BitCode, CallArg, DeviceFunction, ExternFn, ExternOverloads};

// m9-range-prefetch: software-pipelined loops (cutlass.range prefetch_stages).
pub use control::{Prefetch, for_range_pipelined};
pub use sim::MulticastReplicas;

// Blackwell layout atoms, UMMA descriptors and sizing helpers (roadmap part m10-umma-layouts).
#[allow(deprecated)]
pub use arch::{SM100_TMEM_CAPACITY_COLUMNS, SM100_TMEM_MIN_ALLOC_COLUMNS};
pub use arch::{get_max_tmem_alloc_cols, get_min_tmem_alloc_cols, is_tmem_allocation_exclusive};
pub use nvgpu::tcgen05;

// Tensor memory (roadmap part m10-tmem-core).
pub use arch::{
    TmemAddressSlot, alloc_tmem, dealloc_tmem, find_tmem_tensor_col_offset, relinquish_tmem_alloc_permit,
    retrieve_tmem_ptr,
};
// TMEM waits and tcgen05 fences (roadmap part m10-tmem-ld-st).
pub use arch::{
    TmemOpKind, fence_view_async_tmem_load, fence_view_async_tmem_op, fence_view_async_tmem_store,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
};

// tcgen05 UMMA (F16/BF16) into TMEM and tcgen05.commit (roadmap part m10-umma-f16):
// `cute::tcgen05::{MmaF16BF16Op, CtaGroup, OperandSource, Field, commit}`.
pub use ir::{UmmaDataKind, UmmaKind, UmmaScale};
// Rubin (SM107) UMMA forms and collector buffers (roadmap part m10-rubin-umma).
pub use ir::{CollectorOp, Sm107Umma};

// SM120 warp-level block-scaled mma.sync (roadmap part m10-sm120-blockscaled):
// `cute::warp::{MmaMXF4Op, MmaMXF4NVF4Op, MmaMXF8Op, MmaMXF8F6F4Op}` and `cute::utils::sm120`.
pub use atom::mma::{GemmOperand, gemm_block_scaled};
pub use ir::BlockScaledMmaKind;
pub use nvgpu::warp;

// tcgen05 TMEM-sourced A and runtime MMA modifiers (roadmap part m10-umma-kinds).
pub use atom::mma::gemm_disable_output_lane;

// Cluster launch control (roadmap part m10-clc-scheduling): the CLC fetch pipeline is
// `cute::pipeline::PipelineClcFetchAsync`, the scheduler `cute::utils::ClcDynamicPersistentTileScheduler`.
pub use arch::{clc_response, issue_clc_query};

// 2:4 structured-sparse tcgen05 UMMA (roadmap part m10-sparse-umma): sparse element types;
// the MMAs are `cute::tcgen05::{MmaF16BF16SparseOp, MmaFP8SparseOp}`, the layouts in
// `cute::utils::sm100`.
mod sparse_elem;
pub use sparse_elem::{
    SparseElem, SparseElemType, SparseTensor, get_sparse_elem_type, make_sparse_elem, recast_layout_from_sparse,
    recast_layout_to_sparse, recast_ptr_sparse, recast_tensor_sparse,
};

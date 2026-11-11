//! # cute-rs
//!
//! A CuTe-style GPU kernel DSL for Rust.
//!
//! Kernels are written as ordinary Rust inside [`Kernel::build`]. They are
//! built from the same pieces as NVIDIA's CuTe and CuTe DSL:
//!
//! * [`Layout`]s and the layout algebra (composition, complement, divides,
//!   products, inverses, swizzles), evaluated at build time;
//! * [`Tensor`]s over global, shared and register memory, with tiling and
//!   partitioning ([`local_tile`], [`local_partition`]);
//! * copy atoms and [`TiledCopy`] (vectorised loads/stores, `cp.async`,
//!   `ldmatrix`) and MMA atoms and [`TiledMma`] (scalar FMA, SM80
//!   `mma.sync` on f16, bf16, TF32 and f64), with [`copy`] and [`gemm`];
//! * runtime values with operator overloading ([`Int32`], [`Float64`],
//!   [`BFloat16`], ...), runtime control flow ([`if_`], [`for_range`],
//!   [`while_carry`], with [`LoopOptions`] for unroll hints) and mutable
//!   variables that it threads through automatically ([`Var`]);
//! * warp-collective operations ([`shuffle_sync`], [`vote_any_sync`]),
//!   named barriers ([`barrier`], [`pipeline::NamedBarrier`]), mbarriers
//!   ([`mbarrier_arrive`]), atomics, fences, [`printf`] and [`inline_ptx`];
//! * raw memory ([`Ptr`], [`make_tensor`], [`alloc_smem`]) and problem sizes
//!   known only at launch ([`Layout::with_strides`], [`param_extent`]);
//! * CuTe DSL's `cute.math` library ([`math`]): exponentials, logarithms,
//!   trigonometric, hyperbolic and error functions, powers and the
//!   arithmetic forms with explicit rounding, `ftz` and `approx` control.
//!   Precise mode is built from exactly rounded operations (no libdevice),
//!   so the GPU computes the simulator's bits; `fastmath` uses the hardware
//!   approximations.
//!
//! A built [`Kernel`] can be lowered to MLIR ([`Kernel::to_mlir`]), compiled
//! to PTX with an installed `mlir-opt` ([`Kernel::compile_ptx`]), and run on
//! the CPU by a checking simulator ([`Kernel::simulate`]) that reports data
//! races, missing `cp.async` waits, out-of-bounds and misaligned accesses
//! with the Rust source location of the offending operation.
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

pub mod arch;
pub mod atom;
pub mod backend;
mod builder;
pub mod control;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod ir;
pub mod layout;
pub mod math;
pub mod pipeline;
pub mod sim;
pub mod tensor;
pub mod validate;
mod value;

pub use arch::{
    Dim3, FULL_MASK, MbarrierOptions, THREADS_PER_WARPGROUP, WARP_SIZE, WARPS_PER_WARPGROUP, activemask, atomic_add,
    atomic_and, atomic_cas, atomic_exch, atomic_max, atomic_min, atomic_or, atomic_xor, barrier, barrier_arrive,
    barrier_arrive_unaligned, block_dim, block_idx, block_in_cluster_dim, block_in_cluster_idx, clock, clock64,
    cluster_dim, cluster_idx, cp_async_bulk, cp_async_commit_group, cp_async_wait_all, cp_async_wait_group, elect_one,
    fence_acq_rel_cluster, fence_acq_rel_cta, fence_acq_rel_gpu, fence_acq_rel_sys, globaltimer, grid_dim, gridid,
    inline_ptx, inline_ptx_effect, inline_ptx_effect_with_sim, inline_ptx_with_sim, lane_idx, lanemask_eq, lanemask_ge,
    lanemask_gt, lanemask_le, lanemask_lt, linear_thread_idx, make_warp_uniform, match_sync, mbarrier_arrive,
    mbarrier_arrive_and_expect_tx, mbarrier_arrive_and_expect_tx_with, mbarrier_arrive_with,
    mbarrier_conditional_try_wait, mbarrier_expect_tx, mbarrier_expect_tx_with, mbarrier_init, mbarrier_init_fence,
    mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait, nsmid, nwarpid, physical_warp_id, printf, shuffle_sync,
    shuffle_sync_bfly, shuffle_sync_down, shuffle_sync_up, smid, sync_threads, sync_warp, thread_idx, vote_all_sync,
    vote_any_sync, vote_ballot_sync, vote_uni_sync, warp_idx, warpsize,
};
pub use atom::copy::{
    AsCopyAtom, CopyAtom, CopyOp, ThrCopy, TiledCopy, autovec_copy, basic_copy, copy, copy_if, make_copy_atom,
    make_predicate, make_tiled_copy, make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_c, make_tiled_copy_c_atom,
    make_tiled_copy_d, make_tiled_copy_s, make_tiled_copy_tv,
};
pub use atom::mma::{MmaAtom, MmaF16BF16Op, MmaOp, MmaTF32Op, ThrMma, TiledMma, gemm, make_mma_atom, make_tiled_mma};
pub use backend::{Arch, CompileError, MlirOptions, Toolchain};
pub use builder::{Kernel, KernelOptions};
pub use control::{
    Carry, Var, for_range, for_range_carry, for_range_carry_with, for_range_with, if_, if_else, if_else_yield,
    while_carry,
};
pub use ir::{DType, LoopOptions, MemScopeKind, MmaKind};
pub use layout::{
    ComposedLayout, Coord, IndexExpr, IntTuple, Layout, SliceCoord, Swizzle, Tiler, Tree, append_ones, blocked_product,
    coalesce, complement, composition, dice, flat_divide, flat_product, is_congruent, left_inverse, logical_divide,
    logical_product, make_composed_layout, make_layout, make_layout_like, make_layout_tv, make_ordered_layout,
    make_swizzle, max_common_layout, max_common_vector, prepend_ones, raked_product, right_inverse, round_up,
    tile_to_shape, tile_to_shape_swizzled, tiled_divide, tiled_product, zipped_divide, zipped_product,
};
pub use math::{MathOptions, RoundingMode};
pub use sim::{KernelArg, LaunchConfig, SimError, SimOptions, SimReport};
pub use tensor::{
    Ident, Ptr, SmemLayout, Tensor, alloc_smem, elem_less, local_partition, local_partition_proj, local_tile,
    local_tile_proj, make_fragment, make_fragment_like, make_fragment_with_layout, make_identity_tensor,
    make_rmem_tensor, make_smem_tensor, make_tensor, param_extent, param_ptr, param_scalar, param_tensor,
};
pub use value::{
    AnyValue, Arith, BF16, BFloat16, Bitwise, Boolean, F16, Float, Float16, Float32, Float64, Int8, Int16, Int32,
    Int64, Integer, IntoValue, Numeric, Uint8, Uint16, Uint32, Uint64, Value, fma, select,
};

/// Everything needed to write kernels: `use cute::prelude::*;`.
pub mod prelude {
    pub use crate::*;
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's `cutlass.utils`: helpers for building kernels on the host
//! side. So far [`LayoutEnum`] and the SM90 shared-memory layout helpers
//! ([`sm90`], CuTe DSL's `cutlass.utils.hopper_helpers`).
//!
//! The static persistent tile schedulers ([`StaticPersistentTileScheduler`],
//! [`StaticPersistentRuntimeTileScheduler`] and their
//! [`PersistentTileSchedulerParams`]) are here too; the device queries they
//! size grids with are `cuda::HardwareInfo` (feature
//! `cuda`).
//!
//! For grouped GEMMs, [`StaticPersistentGroupTileScheduler`] finds each
//! tile's group with a warp-wide search over the groups' shapes, and
//! [`contiguous_group_search`] finds the group of a tile of one
//! concatenated tensor (MoE).

mod contiguous_group_search;
mod grouped_tile_scheduler;
pub mod sm90;
/// CuTe DSL's deprecated `cutlass.utils.block` spelling of
/// [`crate::block`] (roadmap part m10-s2t-copies).
pub mod block {
    /// The deprecated spelling of [`crate::block::block_copy`].
    #[deprecated(note = "use cute::block::block_copy")]
    #[track_caller]
    pub fn block_copy<
        'a,
        T: crate::value::Numeric,
        S: crate::atom::copy::CopyElem<T>,
        D: crate::atom::copy::CopyElem<T>,
    >(
        tiled_copy: &impl crate::block::BlockCopy<T>,
        src: &crate::tensor::Tensor<S>,
        dst: &crate::tensor::Tensor<D>,
        args: impl Into<crate::atom::copy::CopyArgs<'a>>,
    ) {
        crate::block::block_copy(tiled_copy, src, dst, args)
    }
}
// m9-gemm-host-helpers.
pub mod gemm;
pub mod sm100;
// m10-rubin-extras: the Rubin (SM107) helpers, CuTe DSL's rubin_helpers.
pub mod sm107;
// m10-umma-kinds: the SM100 MMA and shared-memory layout helpers (in utils::sm100).
mod sm100_mma;
// m10-sparse-umma: the sparse GEMM layouts.
mod sm100_sparse;
// m10-cta-pair-pipelines: the SM100 TMA load op selection (in utils::sm100).
mod sm100_tma;
// m10-epilogue-helpers: the SM100 epilogue helpers (in utils::sm100 and utils::gemm::sm100).
mod sm100_epilogue;
// m10-mixed-input: cutlass.utils.mixed_input_helpers.
pub mod mixed_input_helpers;
mod tensormap;
// m11-extern-distributed: cutlass.utils.distributed.
pub mod distributed;
mod tile_scheduler;
// m10-clc-scheduling.
mod clc_tile_scheduler;
pub use clc_tile_scheduler::{
    ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerOptions,
    ClcDynamicPersistentTileSchedulerParams,
};
// m10-port-clc-clusters: preferred and fallback cluster shapes.
mod mixed_cluster;
pub use mixed_cluster::{
    ClusterArm, check_cluster_shape_in_arm, check_mixed_cluster_shapes, current_cluster_arm, is_preferred_cluster,
    mixed_cluster_arms,
};
// m10-umma-layouts.
mod blockscaled_layout;
mod epilogue_tile;

// In-memory TMA descriptors (m7-tensormap-update).
pub use tensormap::{TensorMapManager, TensorMapUpdateMode};
// m10-umma-layouts.
pub use blockscaled_layout::{BlockScaledBasicChunk, Sm103BlockScaledBasicChunk, tile_atom_to_shape_sf};
// m10-sm120-blockscaled.
pub mod sm120;
pub use blockscaled_layout::{sm120_make_smem_layout_sfa, sm120_make_smem_layout_sfb};
// m10-port-nvfp4-gemm.
pub use blockscaled_layout::{
    make_smem_layout_sf, make_smem_layout_sfa, make_smem_layout_sfb, make_tmem_layout_sfa, make_tmem_layout_sfb,
    sm103_make_smem_layout_sfa, sm103_make_smem_layout_sfb,
};
pub use epilogue_tile::compute_epilogue_tile_shape;
pub use tile_scheduler::{
    PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentRuntimeTileScheduler,
    StaticPersistentTileScheduler, WorkTileInfo, swizzle_and_rasterize,
};

// The grouped GEMM schedulers (m9-grouped-gemm-schedulers).
pub use contiguous_group_search::{
    ContiguousGGSearchState, ContiguousGroupWorkTileInfo, contiguous_group_search,
    create_initial_contiguous_group_search_state, make_contiguous_group_work_tile_info,
};
#[allow(deprecated)]
pub use grouped_tile_scheduler::GroupedGemmTileSchedulerHelper;
pub use grouped_tile_scheduler::{
    GroupSearchResult, GroupedGemmGroupSearchState, GroupedWorkTileInfo, StaticPersistentGroupTileScheduler,
    create_initial_search_state,
};

// m10-umma-pipelines: the deprecated `cutlass.utils` names of the TMEM allocator.
/// Deprecated name of [`memory::TmemAllocator`](crate::memory::TmemAllocator).
#[deprecated(note = "use cute::memory::TmemAllocator")]
pub type TmemAllocator = crate::memory::TmemAllocator;

/// Deprecated: [`memory::get_num_tmem_alloc_cols`](crate::memory::get_num_tmem_alloc_cols)
/// for sm_100.
#[deprecated(note = "use cute::memory::get_num_tmem_alloc_cols")]
#[track_caller]
pub fn get_num_tmem_alloc_cols(tmem_tensors: &[&dyn crate::memory::TmemColumns], rounding: bool) -> usize {
    crate::memory::get_num_tmem_alloc_cols(tmem_tensors, rounding, crate::Arch::Sm100)
}

use std::fmt;

use crate::layout::Layout;
use crate::nvgpu::OperandMajorMode;
use crate::tensor::Tensor;

/// The majorness of a GEMM operand or output as a matrix (CuTe DSL's
/// `cutlass.utils.LayoutEnum`, from `cutlass/tensor_utils/layout.py`):
/// `ROW_MAJOR` has the second mode contiguous, `COL_MAJOR` the first.
/// For A `(M, K)` row-major is K-major, for B `(N, K)` row-major is K-major
/// too, and for C `(M, N)` row-major is N-major.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LayoutEnum {
    ROW_MAJOR,
    COL_MAJOR,
}

impl LayoutEnum {
    /// The enum's value: `"row_major"` or `"col_major"`.
    pub fn value(self) -> &'static str {
        match self {
            LayoutEnum::ROW_MAJOR => "row_major",
            LayoutEnum::COL_MAJOR => "col_major",
        }
    }

    /// The MMA operand major mode: `K` for row-major, `MN` for column-major.
    pub fn mma_major_mode(self) -> OperandMajorMode {
        match self {
            LayoutEnum::ROW_MAJOR => OperandMajorMode::K,
            LayoutEnum::COL_MAJOR => OperandMajorMode::MN,
        }
    }

    /// The SM90 (WGMMA) operand major mode, the same as
    /// [`mma_major_mode`](LayoutEnum::mma_major_mode).
    pub fn sm90_mma_major_mode(self) -> OperandMajorMode {
        self.mma_major_mode()
    }

    pub fn is_k_major_a(self) -> bool {
        self == LayoutEnum::ROW_MAJOR
    }

    pub fn is_m_major_a(self) -> bool {
        self == LayoutEnum::COL_MAJOR
    }

    pub fn is_n_major_b(self) -> bool {
        self == LayoutEnum::COL_MAJOR
    }

    pub fn is_k_major_b(self) -> bool {
        self == LayoutEnum::ROW_MAJOR
    }

    pub fn is_n_major_c(self) -> bool {
        self == LayoutEnum::ROW_MAJOR
    }

    pub fn is_m_major_c(self) -> bool {
        self == LayoutEnum::COL_MAJOR
    }

    /// The enum for a leading dimension as [`Layout::leading_dim`] gives
    /// it: its first entry (the top-level mode) 0 is `COL_MAJOR`, 1 is
    /// `ROW_MAJOR`. Panics for any other mode or for no leading dimension
    /// (CuTe DSL's `ValueError`).
    #[track_caller]
    pub fn from_leading_dim(leading_dim: Option<&[usize]>) -> LayoutEnum {
        match leading_dim.and_then(|d| d.first()) {
            Some(0) => LayoutEnum::COL_MAJOR,
            Some(1) => LayoutEnum::ROW_MAJOR,
            _ => panic!("LayoutEnum: invalid leading dimension: {leading_dim:?}"),
        }
    }

    /// The enum for a layout's leading dimension
    /// ([`from_leading_dim`](LayoutEnum::from_leading_dim)).
    #[track_caller]
    pub fn from_layout(layout: &Layout) -> LayoutEnum {
        LayoutEnum::from_leading_dim(layout.leading_dim().as_deref())
    }

    /// The enum for a tensor's leading dimension (CuTe DSL's
    /// `LayoutEnum.from_tensor(tensor)`), as
    /// [`from_leading_dim`](LayoutEnum::from_leading_dim) reads it.
    #[track_caller]
    pub fn from_tensor<T>(tensor: &Tensor<T>) -> LayoutEnum {
        LayoutEnum::from_leading_dim(tensor.leading_dim().as_deref())
    }
}

impl fmt::Display for LayoutEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutEnum::ROW_MAJOR => f.write_str("LayoutEnum.ROW_MAJOR"),
            LayoutEnum::COL_MAJOR => f.write_str("LayoutEnum.COL_MAJOR"),
        }
    }
}

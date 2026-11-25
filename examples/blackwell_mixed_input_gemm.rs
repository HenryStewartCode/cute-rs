// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py
//   (`MixedInputGemmKernel`: its warp roles, `_setup_attributes`, `__call__` and `kernel`,
//   with their comments and names),
//   examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_host_utils.py
//   (the reference: A converted, scaled per row and K granule, times B),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A reduced Blackwell mixed-input GEMM, ported from CuTe DSL's `mixed_input_gemm.py`: `C = (convert(A) [* scale]) B^T` with Int4 A, BFloat16 B
//! and a Float32 accumulator and C, on the helpers of [`cute::utils::mixed_input_helpers`].
//!
//! The warps play the example's roles: 0 to 3 the epilogue, 4 the MMA, 5 the TMA loads of A
//! and B, 6 the TMA loads of the scales, 7 idles, and 8 to 11 transform A.
//!
//! - A `(M, K)` is K-major or M-major. [`get_transform_a_source`] puts the transformed A in
//!   tensor memory for K-major A (the MMA takes A from TMEM: a TS-UMMA) and in shared memory
//!   for M-major A.
//! - The transform warps wait for a stage of A (a `PipelineTmaAsync` from the TMA warp),
//!   load their part of it ([`transform_partition`]), convert it with [`cvt_tensor_a`],
//!   multiply it by the scales in [`TransformMode::ConvertScale`] ([`scale_partition`]), and
//!   store it ([`store_transformed_a`]: `tcgen05.st.32x32b.x8` into TMEM, or 32-bit stores
//!   into shared memory), then hand it to the MMA warp through a `PipelineAsyncUmma`.
//! - The MMA warp waits for the transformed A and for B (a `PipelineTmaUmma`) and issues the
//!   MMAs; the epilogue drains the accumulator ([`epilogue_tma_store`]).
//! - With `use_2cta_instrs` a CTA pair runs each MMA: B is loaded with the 2-CTA TMA load
//!   ([`get_tma_atom_kind`]), A by each CTA for its own transform warps, and the transform
//!   warps of both CTAs arrive on the leader's `PipelineAsyncUmma` barrier.
//! - [`TransformMode::ConvertScale`] loads the scales (one per row and K granule, BFloat16,
//!   M-major) with their own TMA ([`get_gmem_layout_scale`], [`get_smem_layout_scale`],
//!   [`scale_tma_partition`]) through a second `PipelineTmaAsync`.
//!
//! Differences from the Python:
//! - one output tile per CTA: the static persistent scheduler, the grouped variants, the
//!   Int8 A and the non-TMA-store epilogue are not ported, nor host argument parsing, torch
//!   interop and benchmarking. `M`, `N` and `K` are tile multiples and fixed when the kernel
//!   is built.
//! - the stage counts are fixed ([`MixedInputGemm::stages`]) instead of filling shared memory.
//! - no batch mode (the TMA copies are 2-D, the Python's 3-D), and the warps keep the register
//!   budget they start with (the Python moves registers between its roles with `setmaxnreg`).
//! - the transform warps convert their whole part of a stage at once (the Python works in
//!   sub-tiles of at most 64 values to bound register use) and wait without `try_wait` peeks.
//! - the packaged epilogue ([`epilogue_tma_store`]) drains the accumulator, as in
//!   `blackwell_dense_gemm.rs`.
//! - a CTA pair with K-major A (a 2-CTA TS-UMMA) is not supported: cute-rs's TMEM A fragment
//!   and simulator know TS-UMMAs of one CTA only, so the CTA-pair configurations take M-major A.
//! - the shuffled Int4 conversion is off by default (`shuffle_a`): see
//!   [`cute::utils::mixed_input_helpers`].
//!
//! No Blackwell GPU is available to cute-rs's tests, so the kernel is verified by the simulator
//! (bit for bit: the data are small integers, the scales powers of two) and by
//! `ptxas -arch=sm_100a`.
//!
//! Run with `cargo run --release --example blackwell_mixed_input_gemm`.

use cute::cpasync;
use cute::memory::{SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    self, Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineAsyncUmma, PipelineBarriers, PipelineState,
    PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaStore, PipelineTmaUmma, PipelineTmaUmmaOptions,
    PipelineUmmaAsync, PipelineUmmaOptions, PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Field, OperandMajorMode, OperandSource};
use cute::utils::gemm::sm100::{EpilogueConfig, epilogue_tma_store};
use cute::utils::mixed_input_helpers::{
    TransformCopyAtom, TransformMode, compute_smem_layout, cvt_tensor_a, get_copy_atom_a_transform,
    get_gmem_layout_scale, get_smem_layout_scale, get_tma_atom_kind, get_transform_a_source, is_shuffle_a,
    is_valid_mma_tiler_and_cluster_shape, is_valid_scale_granularity, is_valid_tensor_alignment, scale_partition,
    scale_tma_partition, shuffle_a_host, store_transformed_a, transform_partition,
};
use cute::utils::sm100::{make_smem_layout_epi, make_trivial_tiled_mma};
use cute::utils::{LayoutEnum, compute_epilogue_tile_shape};

/// Int4 A, BFloat16 B (the MMA's type) and scales, Float32 accumulator and C.
pub type ADtype = I4;
pub type MmaDtype = BF16;
pub type CDtype = f32;
pub const ACC_DTYPE: DType = DType::F32;

/// The warp roles.
pub const EPILOG_WARP_ID: [usize; 4] = [0, 1, 2, 3];
pub const MMA_WARP_ID: i32 = 4;
pub const TMA_WARP_ID: i32 = 5;
pub const SCALE_TMA_WARP_ID: i32 = 6;
pub const TRANSFORM_WARP_ID: [i32; 4] = [8, 9, 10, 11];
pub const THREADS_PER_CTA: usize = 32 * 12;

/// `MixedInputGemmKernel`'s constructor arguments, with A's majorness and the problem shape
/// (fixed when the kernel is built).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedInputGemm {
    pub name: &'static str,
    /// 0 and 0 for [`TransformMode::ConvertOnly`]; otherwise 1 and a multiple of the tile's K.
    pub scale_granularity_m: usize,
    pub scale_granularity_k: usize,
    pub use_2cta_instrs: bool,
    pub mma_tiler_mnk: (usize, usize, usize),
    pub cluster_shape_mn: (usize, usize),
    /// A's majorness in global memory: K or MN (M-major).
    pub a_major: OperandMajorMode,
    /// Convert A from the shuffled order (the host data shuffled with [`shuffle_a_host`]);
    /// needs PTX ISA 9.1 (LLVM 23) to compile. Off in every configuration here.
    pub shuffle_a: bool,
}

/// K-major A in TMEM, convert only, one CTA, `(128, 64, 64)` tiles.
pub const MIXED_K_MAJOR_CONVERT: MixedInputGemm = MixedInputGemm {
    name: "blackwell_mixed_input_gemm_k_convert",
    scale_granularity_m: 0,
    scale_granularity_k: 0,
    use_2cta_instrs: false,
    mma_tiler_mnk: (128, 64, 64),
    cluster_shape_mn: (1, 1),
    a_major: OperandMajorMode::K,
    shuffle_a: false,
};

/// K-major A in TMEM, scaled per row and 128 K values, one CTA.
pub const MIXED_K_MAJOR_SCALE: MixedInputGemm = MixedInputGemm {
    name: "blackwell_mixed_input_gemm_k_scale",
    scale_granularity_m: 1,
    scale_granularity_k: 128,
    ..MIXED_K_MAJOR_CONVERT
};

/// M-major A in shared memory, convert only, one CTA.
pub const MIXED_M_MAJOR_CONVERT: MixedInputGemm = MixedInputGemm {
    name: "blackwell_mixed_input_gemm_m_convert",
    a_major: OperandMajorMode::MN,
    ..MIXED_K_MAJOR_CONVERT
};

/// M-major A in shared memory, scaled, one CTA.
pub const MIXED_M_MAJOR_SCALE: MixedInputGemm =
    MixedInputGemm { name: "blackwell_mixed_input_gemm_m_scale", a_major: OperandMajorMode::MN, ..MIXED_K_MAJOR_SCALE };

/// M-major A, a CTA pair on `(256, 64, 64)` tiles in a `(2, 1)` cluster: B with the 2-CTA load.
pub const MIXED_M_MAJOR_2CTA: MixedInputGemm = MixedInputGemm {
    name: "blackwell_mixed_input_gemm_m_2cta",
    use_2cta_instrs: true,
    mma_tiler_mnk: (256, 64, 64),
    cluster_shape_mn: (2, 1),
    ..MIXED_M_MAJOR_CONVERT
};

/// The CTA pair, scaled.
pub const MIXED_M_MAJOR_2CTA_SCALE: MixedInputGemm = MixedInputGemm {
    name: "blackwell_mixed_input_gemm_m_2cta_scale",
    scale_granularity_m: 1,
    scale_granularity_k: 128,
    ..MIXED_M_MAJOR_2CTA
};

cute_struct! {
    pub struct SharedStorage(l2t: usize, s2t: usize, t2m: usize, acc: usize) {
        a_load2trans_mbar_ptr: MemRange<i64>[l2t * 2],
        scale_load2trans_mbar_ptr: MemRange<i64>[s2t * 2],
        a_trans2mma_mbar_ptr: MemRange<i64>[t2m * 2],
        b_load2mma_mbar_ptr: MemRange<i64>[l2t * 2],
        acc_mbar_ptr: MemRange<i64>[acc * 2],
        tmem_dealloc_mbar: i64,
        tmem_holding_buf: i32,
    }
}

/// The stage counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages {
    pub load2trans: usize,
    pub scale_load2trans: usize,
    pub trans2mma: usize,
    pub acc: usize,
    pub c: usize,
}

/// What `_setup_attributes` computes.
pub struct Setup {
    pub tiled_mma: TiledMma,
    pub transform_a_source: OperandSource,
    pub scale_mode: TransformMode,
    pub cta_tile_shape_mnk: (usize, usize, usize),
    pub cluster_layout_vmnk: Layout,
    pub epi_tile: Tiler,
    pub stages: Stages,
    pub smem_layout_a: ComposedLayout,
    pub smem_layout_a_transform: ComposedLayout,
    pub smem_layout_b: ComposedLayout,
    pub c_smem_layout_staged: ComposedLayout,
    /// The scales' tile, one stage of their layout and the staged layout (ConvertScale).
    pub scale: Option<((usize, usize), ComposedLayout, ComposedLayout)>,
    pub num_acc_tmem_cols: usize,
    pub num_tmem_alloc_cols: usize,
}

impl MixedInputGemm {
    pub fn cta_group(&self) -> CtaGroup {
        if self.use_2cta_instrs { CtaGroup::TWO } else { CtaGroup::ONE }
    }

    pub fn scale_mode(&self) -> TransformMode {
        TransformMode::from_granularity(self.scale_granularity_m, self.scale_granularity_k)
    }

    /// The fixed stage counts of the reduced example.
    pub fn stages(&self) -> Stages {
        Stages { load2trans: 3, scale_load2trans: 2, trans2mma: 2, acc: 1, c: 2 }
    }

    /// `can_implement`'s checks for an `m x n x k` problem.
    pub fn can_implement(&self, (m, n, k): (usize, usize, usize)) -> bool {
        let (tm, tn, tk) = self.mma_tiler_mnk;
        let cta_m = tm / if self.use_2cta_instrs { 2 } else { 1 };
        is_valid_mma_tiler_and_cluster_shape(self.mma_tiler_mnk, self.cluster_shape_mn, self.use_2cta_instrs)
            // The Python ties Int4 A to ConvertScale; the port converts Int4 A without scales too.
            && (self.scale_mode() == TransformMode::ConvertOnly
                || is_valid_scale_granularity(self.scale_granularity_m, self.scale_granularity_k, DType::I4, k, tk))
            && is_valid_tensor_alignment(
                (m, n, k),
                DType::I4,
                DType::BF16,
                DType::F32,
                DType::BF16,
                self.a_major,
                OperandMajorMode::K,
                LayoutEnum::ROW_MAJOR,
                self.mma_tiler_mnk,
                self.use_2cta_instrs,
                self.cluster_shape_mn,
                self.scale_granularity_m,
                self.scale_granularity_k,
            )
            && m.is_multiple_of(cta_m * self.cluster_shape_mn.0)
            && n.is_multiple_of(tn * self.cluster_shape_mn.1)
            && k.is_multiple_of(tk)
            && (!self.shuffle_a || is_shuffle_a(self.a_major, k, DType::I4, DType::BF16, self.scale_granularity_k, true))
    }

    /// `_setup_attributes`.
    pub fn setup(&self) -> Setup {
        // Deduce where the transformed A tensor is stored, shared memory(SMEM) or tensor memory(TMEM)
        let transform_a_source = get_transform_a_source(self.a_major);
        let tiled_mma = make_trivial_tiled_mma(
            MmaDtype::DTYPE,
            MmaDtype::DTYPE,
            self.a_major,
            OperandMajorMode::K,
            ACC_DTYPE,
            self.cta_group(),
            (self.mma_tiler_mnk.0, self.mma_tiler_mnk.1),
            transform_a_source,
        );
        let mma_tiler = self.mma_tiler_mnk;
        let cta_tile_shape_mnk = (mma_tiler.0 / tiled_mma.thr_id().size(), mma_tiler.1, mma_tiler.2);
        let (cm, cn) = self.cluster_shape_mn;
        let cluster_layout_vmnk = Layout::col_major((cm, cn, 1))
            .tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));
        let epi_tile = compute_epilogue_tile_shape(
            &[cta_tile_shape_mnk.0, cta_tile_shape_mnk.1],
            self.use_2cta_instrs,
            LayoutEnum::ROW_MAJOR,
            CDtype::DTYPE,
            None,
            None,
            None,
        );
        let stages = self.stages();
        // TMEM columns: the accumulator, then the transformed A's stages.
        let num_acc_tmem_cols = mma_tiler.1 * stages.acc;
        let num_a_tmem_cols = if transform_a_source == OperandSource::TMEM {
            mma_tiler.2 * MmaDtype::DTYPE.bits() / 32 * stages.trans2mma
        } else {
            0
        };
        let num_tmem_alloc_cols = (num_acc_tmem_cols + num_a_tmem_cols).max(32).next_power_of_two();
        let c_smem_layout_staged = make_smem_layout_epi(CDtype::DTYPE, LayoutEnum::ROW_MAJOR, &epi_tile, stages.c);
        // Get smem layout for A, transformed A, and B
        let (smem_layout_a, smem_layout_a_transform, smem_layout_b) = compute_smem_layout(
            &tiled_mma,
            mma_tiler,
            ADtype::DTYPE,
            MmaDtype::DTYPE,
            stages.load2trans,
            stages.trans2mma,
        );
        let scale_mode = self.scale_mode();
        let scale = (scale_mode == TransformMode::ConvertScale).then(|| {
            get_smem_layout_scale(
                mma_tiler,
                self.use_2cta_instrs,
                self.scale_granularity_m,
                self.scale_granularity_k,
                OperandMajorMode::MN,
                MmaDtype::DTYPE,
                stages.scale_load2trans,
            )
        });
        Setup {
            tiled_mma,
            transform_a_source,
            scale_mode,
            cta_tile_shape_mnk,
            cluster_layout_vmnk,
            epi_tile,
            stages,
            smem_layout_a,
            smem_layout_a_transform,
            smem_layout_b,
            c_smem_layout_staged,
            scale,
            num_acc_tmem_cols,
            num_tmem_alloc_cols,
        }
    }

    /// One CTA per CTA tile of C.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let cta_m = self.mma_tiler_mnk.0 / if self.use_2cta_instrs { 2 } else { 1 };
        let grid = [m / cta_m, n / self.mma_tiler_mnk.1, 1];
        let launch = LaunchConfig::new(grid.map(|g| g as u32), [THREADS_PER_CTA as u32, 1, 1]);
        let (cx, cy) = self.cluster_shape_mn;
        if cx * cy > 1 { launch.with_cluster([cx as u32, cy as u32, 1]) } else { launch }
    }

    /// The kernel for an `m x n x k` problem: parameters `a`, then `scale` (ConvertScale
    /// only: `(M / gm, K / gk)` M-major), `b` and `c`.
    pub fn build(&self, (m, n, k): (usize, usize, usize)) -> Kernel {
        let cfg = *self;
        assert!(cfg.can_implement((m, n, k)), "{}: cannot implement {m}x{n}x{k}", cfg.name);
        Kernel::builder(cfg.name).block_dim([THREADS_PER_CTA as u32, 1, 1]).target(Arch::Sm100a).build(move || {
            let st = cfg.setup();
            let mma_tiler = cfg.mma_tiler_mnk;
            let a_layout = if cfg.a_major == OperandMajorMode::K {
                Layout::new((m, k), (k, 1))
            } else {
                Layout::new((m, k), (1, m))
            };
            let a = param_tensor_aligned::<ADtype>("a", a_layout, 32);
            let scale = (st.scale_mode == TransformMode::ConvertScale).then(|| {
                let gm = get_gmem_layout_scale(
                    (m, k, 1),
                    cfg.scale_granularity_m,
                    cfg.scale_granularity_k,
                    OperandMajorMode::MN,
                );
                let raw = param_tensor_aligned::<MmaDtype>(
                    "scale",
                    Layout::col_major(m / cfg.scale_granularity_m * (k / cfg.scale_granularity_k)),
                    32,
                );
                // (M, K) of A to its scale: drop the batch mode.
                make_tensor(raw.ptr().clone(), Layout::from_modes([gm.mode(0), gm.mode(1)]))
            });
            let b = param_tensor_aligned::<MmaDtype>("b", Layout::new((n, k), (k, 1)), 32);
            let c = param_tensor_aligned::<CDtype>("c", Layout::new((m, n), (n, 1)), 32);

            let tiled_mma = st.tiled_mma.clone();
            let cluster =
                (cfg.cluster_shape_mn.0 * cfg.cluster_shape_mn.1 > 1).then(|| st.cluster_layout_vmnk.shape().clone());
            let num_mcast_ctas_a = st.cluster_layout_vmnk.mode(2).size();
            let num_mcast_ctas_b = st.cluster_layout_vmnk.mode(1).size();

            // Set up gmem copy atoms for A, scale, and B
            let a_op = get_tma_atom_kind(num_mcast_ctas_a > 1, cfg.use_2cta_instrs, false);
            let b_op = get_tma_atom_kind(num_mcast_ctas_b > 1, cfg.use_2cta_instrs, true);
            let smem_layout_a_per_stage = st.smem_layout_a.select(&[0, 1, 2]);
            let (tma_atom_a, tma_tensor_a) = make_tiled_tma_atom_a(
                a_op,
                &a,
                smem_layout_a_per_stage.clone(),
                mma_tiler,
                &tiled_mma,
                cluster.clone(),
            )
            .into();
            let scale_tma = scale.as_ref().map(|s| {
                let (tile, per_stage, _) = st.scale.as_ref().unwrap();
                // Partition smem layout for scale tensor to make it compatible with TMA atom:
                // ((MMA_M, MMA_K), REST_M, REST_K)
                let thr0 = tiled_mma.get_slice(0);
                let for_tma = thr0.partition_a(&make_coord_tensor(0, per_stage.outer().clone())).layout().clone();
                let info = make_tiled_tma_atom_a(a_op, s, for_tma, (tile.0, 1, tile.1), &tiled_mma, cluster.clone());
                let (atom, tensor): (CopyAtom<MmaDtype>, Tensor<Ident>) = info.into();
                (atom, tensor)
            });
            let smem_layout_b_per_stage = st.smem_layout_b.select(&[0, 1, 2]);
            let (tma_atom_b, tma_tensor_b) =
                make_tiled_tma_atom_b(b_op, &b, smem_layout_b_per_stage.clone(), mma_tiler, &tiled_mma, cluster).into();

            // Calculate copy size for tensor A, B, and scale
            let bytes = |t: DType, l: &ComposedLayout| size_in_bytes_dtype(t, l).as_static().expect("static layouts");
            let num_tma_load_bytes_a = bytes(ADtype::DTYPE, &smem_layout_a_per_stage);
            let num_tma_load_bytes_b = bytes(MmaDtype::DTYPE, &smem_layout_b_per_stage) * tiled_mma.thr_id().size();
            let num_tma_load_bytes_scale = st.scale.as_ref().map_or(0, |(_, ps, _)| bytes(MmaDtype::DTYPE, ps));

            let (tma_atom_c, tma_tensor_c) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &c,
                st.c_smem_layout_staged.slice((.., .., 0)),
                st.epi_tile.clone(),
                1,
            )
            .into();
            let mut ecfg = EpilogueConfig::new(
                st.cta_tile_shape_mnk,
                LayoutEnum::ROW_MAJOR,
                CDtype::DTYPE,
                ACC_DTYPE,
                cfg.use_2cta_instrs,
            );
            ecfg.num_c_stage = st.stages.c;

            let args = KernelArgs {
                tma_atom_a: &tma_atom_a,
                m_a_mkl: &tma_tensor_a,
                scale_tma: scale_tma.as_ref(),
                tma_atom_b: &tma_atom_b,
                m_b_nkl: &tma_tensor_b,
                tma_atom_c: &tma_atom_c,
                m_c_mnl: &tma_tensor_c,
                num_tma_load_bytes_a,
                num_tma_load_bytes_b,
                num_tma_load_bytes_scale,
                ecfg: &ecfg,
            };
            kernel(&cfg, &st, tiled_mma, &args);
        })
    }
}

/// The kernel's TMA atoms, tensors and transaction sizes.
pub struct KernelArgs<'a> {
    pub tma_atom_a: &'a CopyAtom<ADtype>,
    pub m_a_mkl: &'a Tensor<Ident>,
    pub scale_tma: Option<&'a (CopyAtom<MmaDtype>, Tensor<Ident>)>,
    pub tma_atom_b: &'a CopyAtom<MmaDtype>,
    pub m_b_nkl: &'a Tensor<Ident>,
    pub tma_atom_c: &'a CopyAtom<CDtype>,
    pub m_c_mnl: &'a Tensor<Ident>,
    pub num_tma_load_bytes_a: usize,
    pub num_tma_load_bytes_b: usize,
    pub num_tma_load_bytes_scale: usize,
    pub ecfg: &'a EpilogueConfig,
}

/// `t` sliced at `idx` in its last mode.
fn stage<T>(t: &Tensor<T>, idx: Int32) -> Tensor<T> {
    let r = t.layout().rank();
    let coord: SliceCoord =
        Tree::Branch((0..r).map(|i| Tree::Leaf(if i + 1 == r { Some(idx.into()) } else { None })).collect());
    t.slice(coord)
}

/// The kernel of `MixedInputGemmKernel` (one tile per CTA).
pub fn kernel(cfg: &MixedInputGemm, st: &Setup, tiled_mma: TiledMma, ka: &KernelArgs) {
    let mma_tiler = cfg.mma_tiler_mnk;
    let cluster_layout_vmnk = &st.cluster_layout_vmnk;
    let stages = st.stages;
    let scale_mode = st.scale_mode;
    let convert_scale = scale_mode == TransformMode::ConvertScale;
    let warp_idx = make_warp_uniform(warp_idx());
    let tidx = thread_idx().x;

    // Prefetch TMA descriptors
    if_(warp_idx.eq(EPILOG_WARP_ID[0] as i32), || {
        cpasync::prefetch_descriptor(ka.tma_atom_a);
        cpasync::prefetch_descriptor(ka.tma_atom_b);
        if let Some((atom, _)) = ka.scale_tma {
            cpasync::prefetch_descriptor(atom);
        }
        cpasync::prefetch_descriptor(ka.tma_atom_c);
    });

    let use_2cta_instrs = tiled_mma.thr_id().size() == 2;
    // Compute how many k_tiles share the same scale
    let num_k_tiles_per_scale = if convert_scale { cfg.scale_granularity_k / st.cta_tile_shape_mnk.2 } else { 1 };
    let (bidx, bidy) = (block_idx().x, block_idx().y);
    let v = cluster_layout_vmnk.mode(0).size();
    let mma_tile_coord_v = bidx % v as i32;
    let is_leader_cta = mma_tile_coord_v.eq(0);
    let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
    let block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster);
    let mma_tile_coord_mnl = [bidx / v as i32, bidy];
    let mma_coord_mnk = (mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], ..);

    let smem = SmemAllocator::new();
    let storage_layout =
        SharedStorage::layout(stages.load2trans, stages.scale_load2trans, stages.trans2mma, stages.acc);
    let storage =
        SharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 8, SmemPartition::User));

    // Initialize load2transform pipeline, which tracks the dependencies between TMA's loading
    // of A and B, and the transformation of A and MMA's consumption
    let transform_thread_idx = select(tidx.ge(32 * TRANSFORM_WARP_ID[0]), tidx - 32 * TRANSFORM_WARP_ID[0], tidx);
    let a_load2trans_pipeline = PipelineTmaAsync::create(
        stages.load2trans,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (num_mcast_ctas(cluster_layout_vmnk, 2) * TRANSFORM_WARP_ID.len()) as i32),
        ka.num_tma_load_bytes_a,
        PipelineTmaAsyncOptions {
            barrier_storage: Some(storage.a_load2trans_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.flatten()),
            tidx: Some(transform_thread_idx),
            // multicast for A will only happen on the M-mode
            mcast_mode_mn: (1, 0),
            defer_sync: true,
            ..Default::default()
        },
    );
    // Initialize scale_load2trans pipeline, which tracks the dependencies between TMA's loading
    // of scale, and the transformation of A
    let scale_load2trans_pipeline = convert_scale.then(|| {
        PipelineTmaAsync::create(
            stages.scale_load2trans,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(
                Agent::Thread,
                (num_mcast_ctas(cluster_layout_vmnk, 2) * TRANSFORM_WARP_ID.len() * num_k_tiles_per_scale) as i32,
            ),
            ka.num_tma_load_bytes_scale,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.scale_load2trans_mbar_ptr.data_ptr().cast::<u64>()),
                cta_layout_vmnk: Some(cluster_layout_vmnk.flatten()),
                tidx: Some(transform_thread_idx),
                mcast_mode_mn: (1, 0),
                defer_sync: true,
                ..Default::default()
            },
        )
    });
    // Initialize transform2mma pipeline, which tracks the dependencies between the transformation
    // of A and MMA's consumption of transformed A
    let trans2mma_pipeline = PipelineAsyncUmma::create(
        stages.trans2mma,
        CooperativeGroup::new(Agent::Thread, (32 * TRANSFORM_WARP_ID.len() * v) as i32),
        CooperativeGroup::new(Agent::Thread, 1),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.a_trans2mma_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    // Initialize pipeline for tensor B load to MMA
    let b_load2mma_pipeline = PipelineTmaUmma::create(
        stages.load2trans,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, num_mcast_ctas(cluster_layout_vmnk, 1) as i32),
        ka.num_tma_load_bytes_b,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.b_load2mma_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            mcast_mode_mn: (0, 1),
            defer_sync: true,
            ..Default::default()
        },
    );
    // Initialize accumulator pipeline: the packaged epilogue's consumers are its warps.
    let acc_pipeline = PipelineUmmaAsync::create(
        stages.acc,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (v * EPILOG_WARP_ID.len()) as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cluster_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );

    // Tensor memory dealloc barrier init
    let mut tmem = TmemAllocator::new(
        NamedBarrier::new(4, THREADS_PER_CTA),
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buf.ptr()),
            allocator_warp_id: EPILOG_WARP_ID[0] as i32,
            is_two_cta: use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr: use_2cta_instrs.then(|| storage.tmem_dealloc_mbar.ptr().cast::<u64>()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );

    // Cluster arrive after barrier init
    pipeline::pipeline_init_arrive(Some(cluster_layout_vmnk.shape().clone()), true);

    // Setup smem tensor A/scale/B/C
    let s_c = smem.allocate_tensor::<CDtype>(st.c_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    let s_a_input = smem.allocate_tensor::<ADtype>(st.smem_layout_a.clone(), 1024, None, SmemPartition::User);
    let s_s_input = st
        .scale
        .as_ref()
        .map(|(_, _, staged)| smem.allocate_tensor::<MmaDtype>(staged.clone(), 1024, None, SmemPartition::User));
    let s_b_input = smem.allocate_tensor::<MmaDtype>(st.smem_layout_b.clone(), 1024, None, SmemPartition::User);
    // Get smem tensor for transformed A when transform_a_source is SMEM
    let s_a_transform = (st.transform_a_source == OperandSource::SMEM)
        .then(|| smem.allocate_tensor::<MmaDtype>(st.smem_layout_a_transform.clone(), 1024, None, SmemPartition::User));

    // Compute multicast mask for A/B buffer full
    let clustered = cluster_layout_vmnk.size() > 1;
    let a_full_mcast_mask = (clustered && num_mcast_ctas(cluster_layout_vmnk, 2) > 1)
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 2));
    let b_full_mcast_mask = (clustered && num_mcast_ctas(cluster_layout_vmnk, 1) > 1)
        .then(|| cpasync::create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk.clone(), 1));

    // local_tile partition global tensors: this CTA's row of tiles
    // (bM, bK, RestK)
    let g_a = local_tile_proj(ka.m_a_mkl, mma_tiler, mma_coord_mnk, &[true, false, true]);
    // (bN, bK, RestK)
    let g_b = local_tile_proj(ka.m_b_nkl, mma_tiler, mma_coord_mnk, &[false, true, true]);
    let k_tile_cnt = size(&g_a, &[2]);

    // Partition global tensor for TiledMMA_A/B/C
    let thr_mma = tiled_mma.get_slice(mma_tile_coord_v);
    // (MMA, MMA_M, MMA_K, RestK)
    let t_cg_a = thr_mma.partition_a(&g_a);
    let t_cg_b = thr_mma.partition_b(&g_b);

    // Setup copy atom to load A from shared memory for further transformation
    let copy_atom_a_input = convert_scale.then(|| make_copy_atom::<ADtype>(CopyOp::Universal { bits: 32 }));
    let a_smem_shape = tiled_mma.partition_shape_a((mma_tiler.0, mma_tiler.2));
    // Setup copy atom to store transformed A into tensor memory or shared memory
    let copy_atom_a_transform: TransformCopyAtom<MmaDtype, ADtype> =
        get_copy_atom_a_transform(use_2cta_instrs, st.transform_a_source, &a_smem_shape);

    let coord = |i: usize| match &block_in_cluster_coord_vmnk {
        Tree::Branch(modes) => modes[i].clone(),
        leaf => leaf.clone(),
    };
    // TMA load A partition_S/D
    let a_cta_layout = make_layout(num_mcast_ctas(cluster_layout_vmnk, 2), 1);
    // ((atom_v, rest_v), STAGE) and ((atom_v, rest_v), RestK)
    let (t_as_a, t_ag_a) = cpasync::tma_partition(
        ka.tma_atom_a,
        coord(2),
        &a_cta_layout,
        &s_a_input.group_modes(0, 3),
        &t_cg_a.group_modes(0, 3),
    );
    // The scales: (MMA, MMA_M, MMA_K, STAGE) in shared memory and their TMA partitions.
    let scale_parts = ka.scale_tma.map(|(atom, m_s_mkl)| {
        let s_s = s_s_input.as_ref().unwrap();
        let g_s = local_tile_proj(m_s_mkl, mma_tiler, mma_coord_mnk, &[true, false, true]);
        let t_cg_s = thr_mma.partition_a(&g_s);
        let thr_mma_leader_cta = tiled_mma.get_slice(0);
        let t_cs_s = thr_mma_leader_cta.partition_a(s_s);
        let (t_ss_s, t_sg_s) = scale_tma_partition(&t_cs_s, &t_cg_s, atom, &block_in_cluster_coord_vmnk, &a_cta_layout);
        (atom, t_cs_s, t_ss_s, t_sg_s)
    });
    // TMA load B partition_S/D
    let b_cta_layout = make_layout(num_mcast_ctas(cluster_layout_vmnk, 1), 1);
    let (t_bs_b, t_bg_b) = cpasync::tma_partition(
        ka.tma_atom_b,
        coord(1),
        &b_cta_layout,
        &s_b_input.group_modes(0, 3),
        &t_cg_b.group_modes(0, 3),
    );

    // (MMA, MMA_N, MMA_K, STAGE)
    let t_cr_b = tiled_mma.make_fragment_b(&s_b_input);
    // (MMA, MMA_M, MMA_N)
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
    let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&acc_shape);

    // Cluster wait before TMEM alloc and ensure pipelines are ready
    pipeline::pipeline_init_wait(Some(cluster_layout_vmnk.shape().clone()));

    // TMEM allocation
    tmem.allocate(st.num_tmem_alloc_cols);
    tmem.wait_for_alloc();
    let tmem_ptr = tmem.retrieve_ptr::<f32>();
    let t_ct_acc = make_tensor(tmem_ptr.clone(), t_ct_acc_fake.layout().clone());
    // The transformed A: in TMEM after the accumulator's columns, or in shared memory.
    // (MMA, MMA_M, MMA_K, STAGE)
    let t_cr_a = match &s_a_transform {
        None => {
            let a_shape = tiled_mma.partition_shape_a((mma_tiler.0, mma_tiler.2));
            let Tree::Branch(mut modes) = a_shape else { panic!("a tuple shape") };
            modes.push(Shape::from(stages.trans2mma));
            let frag = tiled_mma.make_tmem_fragment_a::<MmaDtype>(&Shape::from(Tree::Branch(modes)));
            let per_word = (32 / MmaDtype::DTYPE.bits()) as i32;
            let a_ptr = tmem.retrieve_ptr::<MmaDtype>() + st.num_acc_tmem_cols as i32 * per_word;
            make_tensor(a_ptr, frag.layout().clone())
        }
        Some(s) => tiled_mma.make_fragment_a(s),
    };

    //
    // Specialized TMA load warp for A/B tensor
    //
    if_(warp_idx.eq(TMA_WARP_ID), || {
        let a_state = make_pipeline_state(PipelineUserType::Producer, stages.load2trans);
        let b_state = make_pipeline_state(PipelineUserType::Producer, stages.load2trans);
        let (a_state, b_state) =
            for_range_carry(0, k_tile_cnt.clone(), 1, (a_state, b_state), |k_tile, (mut a_state, mut b_state)| {
                // Conditionally wait for A buffer empty, then load A
                a_load2trans_pipeline.producer_acquire(&a_state, None);
                let bar = CopyArgs::new().tma_bar_ptr(&a_load2trans_pipeline.producer_get_barrier(&a_state));
                let bar = match a_full_mcast_mask {
                    Some(mask) => bar.mcast_mask(mask),
                    None => bar,
                };
                copy_with(ka.tma_atom_a, &t_ag_a.slice((.., k_tile)), &stage(&t_as_a, a_state.index()), bar);
                a_state.advance();
                // Conditionally wait for B buffer empty, then load B
                b_load2mma_pipeline.producer_acquire(&b_state, None);
                let bar = CopyArgs::new().tma_bar_ptr(&b_load2mma_pipeline.producer_get_barrier(&b_state));
                let bar = match b_full_mcast_mask {
                    Some(mask) => bar.mcast_mask(mask),
                    None => bar,
                };
                copy_with(ka.tma_atom_b, &t_bg_b.slice((.., k_tile)), &stage(&t_bs_b, b_state.index()), bar);
                b_state.advance();
                (a_state, b_state)
            });
        // Wait A/B buffer empty
        let (mut a_state, mut b_state) = (a_state, b_state);
        a_load2trans_pipeline.producer_tail(&mut a_state);
        b_load2mma_pipeline.producer_tail(&mut b_state);
    });

    //
    // Specialized TMA load warp for the scale tensor: one load per scale granule of K.
    //
    if let (Some((atom, _, t_ss_s, t_sg_s)), Some(scale_pipe)) = (&scale_parts, &scale_load2trans_pipeline) {
        if_(warp_idx.eq(SCALE_TMA_WARP_ID), || {
            let s_state = make_pipeline_state(PipelineUserType::Producer, stages.scale_load2trans);
            let num_scale_tiles = k_tile_cnt.clone() / num_k_tiles_per_scale as i32;
            let mut s_state = for_range_carry(0, num_scale_tiles, 1, s_state, |s_tile, mut s_state| {
                scale_pipe.producer_acquire(&s_state, None);
                let bar = CopyArgs::new().tma_bar_ptr(&scale_pipe.producer_get_barrier(&s_state));
                let bar = match a_full_mcast_mask {
                    Some(mask) => bar.mcast_mask(mask),
                    None => bar,
                };
                let k_tile = s_tile * num_k_tiles_per_scale as i32;
                copy_with(*atom, &t_sg_s.slice((.., k_tile)), &stage(t_ss_s, s_state.index()), bar);
                s_state.advance();
                s_state
            });
            scale_pipe.producer_tail(&mut s_state);
        });
    }

    //
    // Specialized transform warps
    //
    if_(warp_idx.ge(TRANSFORM_WARP_ID[0]), || {
        let transform_local_tidx = tidx - 32 * TRANSFORM_WARP_ID[0];
        // Partition tensors for transform input and output and set up the copy atom
        // used for loading and storing transformed A tensor
        let a_transform_dst = match &s_a_transform {
            Some(s) => s.clone(),
            None => t_cr_a.clone(),
        };
        let parts = transform_partition(
            st.transform_a_source,
            scale_mode,
            copy_atom_a_input.as_ref(),
            &copy_atom_a_transform,
            &s_a_input,
            &a_transform_dst,
            transform_local_tidx,
        );
        // make rmem tensor for input A and transformed A
        let one_stage_in = stage(&parts.t_as_a_input, Int32::from(0));
        let t_ar_a = make_fragment_like::<ADtype, _>(&one_stage_in);
        let t_ar_a_transform = make_fragment_like::<MmaDtype, _>(&one_stage_in);
        // Partition scale tensor
        let sp = scale_parts.as_ref().map(|(_, t_cs_s, _, _)| {
            let src_copy_a = parts.src_copy_a.as_ref().expect("ConvertScale has a source copy");
            let sp = scale_partition(src_copy_a, t_cs_s, transform_local_tidx);
            assert_eq!(sp.t_sr_s.layout().size(), t_ar_a.layout().size(), "tSrS and tArA have different shape");
            sp
        });

        let a_state = make_pipeline_state(PipelineUserType::Consumer, stages.load2trans);
        let s_state = make_pipeline_state(PipelineUserType::Consumer, stages.scale_load2trans);
        let t_state = make_pipeline_state(PipelineUserType::Producer, stages.trans2mma);
        let transform_sync_barrier = NamedBarrier::new(3, 32 * TRANSFORM_WARP_ID.len());
        let (_, _, mut t_state) = for_range_carry(
            0,
            k_tile_cnt.clone(),
            1,
            (a_state, s_state, t_state),
            |k_tile, (mut a_state, mut s_state, mut t_state)| {
                a_load2trans_pipeline.consumer_wait(&a_state, None);
                if let Some(p) = &scale_load2trans_pipeline {
                    p.consumer_wait(&s_state, None);
                }
                trans2mma_pipeline.producer_acquire(&t_state, None);
                // load scale tensor when needed
                let cur_s_state = s_state.clone_state();
                if let Some(sp) = &sp {
                    if_((k_tile % num_k_tiles_per_scale as i32).eq(0), || {
                        let slice = stage(&sp.t_ss_s_trans, s_state.index());
                        let filtered = make_tensor(slice.ptr().clone(), slice.layout().filter_zeros());
                        autovec_copy(&filtered, &sp.t_sr_s_copy);
                    });
                    let last = ((k_tile + 1) % num_k_tiles_per_scale as i32).eq(0);
                    let mut next = s_state.clone_state();
                    next.advance();
                    s_state = select_state(last, &next, &s_state);
                }
                // Load A from shared memory
                autovec_copy(&stage(&parts.t_as_a_input, a_state.index()), &t_ar_a);
                let cur_a_state = a_state.clone_state();
                a_state.advance();
                // Convert it to mma dtype
                let mut transformed = cvt_tensor_a::<MmaDtype>(&t_ar_a, cfg.shuffle_a);
                if let Some(sp) = &sp {
                    // Apply scale (the scales in the transformed tile's shape)
                    let scale = sp.t_sr_s.load_ssa().reshape(transformed.shape().clone());
                    transformed = transformed * scale;
                }
                t_ar_a_transform.store_ssa(&transformed);
                // Store transformed A to tensor memory or shared memory
                store_transformed_a(
                    &t_ar_a_transform,
                    &stage(&parts.t_a_transform, t_state.index()),
                    parts.dst_copy_a.as_ref(),
                );
                // Ensure all transform threads have finished the copy and reached the fence
                transform_sync_barrier.arrive_and_wait();
                if st.transform_a_source == OperandSource::TMEM {
                    fence_view_async_tmem_store();
                } else {
                    fence_view_async_shared();
                }
                // Signal the completion of transformation
                if let Some(p) = &scale_load2trans_pipeline {
                    p.consumer_release(&cur_s_state);
                }
                a_load2trans_pipeline.consumer_release(&cur_a_state);
                trans2mma_pipeline.producer_commit(&t_state);
                t_state.advance();
                (a_state, s_state, t_state)
            },
        );
        // Wait a_transform buffer empty
        trans2mma_pipeline.producer_tail(&mut t_state);
    });

    //
    // Specialized MMA warp
    //
    if_(warp_idx.eq(MMA_WARP_ID), || {
        let issue = || {
            let mut acc_state = make_pipeline_state(PipelineUserType::Producer, stages.acc);
            acc_pipeline.producer_acquire(&acc_state, None);
            let mut tiled_mma = tiled_mma.clone();
            tiled_mma.set(Field::ACCUMULATE, false);
            let t_state = make_pipeline_state(PipelineUserType::Consumer, stages.trans2mma);
            let b_state = make_pipeline_state(PipelineUserType::Consumer, stages.load2trans);
            // Mma mainloop
            let _ = for_range_carry(
                0,
                k_tile_cnt.clone(),
                1,
                (t_state, b_state, tiled_mma.clone()),
                |_, (mut t_state, mut b_state, mut tiled_mma)| {
                    trans2mma_pipeline.consumer_wait(&t_state, None);
                    b_load2mma_pipeline.consumer_wait(&b_state, None);
                    let num_kblocks = t_cr_a.layout().mode(2).size();
                    for kblock_idx in 0..num_kblocks {
                        let ka_crd = (.., .., kblock_idx, t_state.index());
                        let kb_crd = (.., .., kblock_idx, b_state.index());
                        gemm(&tiled_mma, &t_ct_acc, &t_cr_a.slice(ka_crd), &t_cr_b.slice(kb_crd), &t_ct_acc);
                        // Enable accumulate on tCtAcc after first kblock
                        tiled_mma.set(Field::ACCUMULATE, true);
                    }
                    trans2mma_pipeline.consumer_release(&t_state);
                    b_load2mma_pipeline.consumer_release(&b_state);
                    t_state.advance();
                    b_state.advance();
                    (t_state, b_state, tiled_mma)
                },
            );
            // Async arrive accumulator buffer full
            acc_pipeline.producer_commit(&acc_state);
            acc_state.advance();
            // Wait for accumulator buffer empty
            acc_pipeline.producer_tail(&mut acc_state);
        };
        if use_2cta_instrs { if_(is_leader_cta, issue) } else { issue() }
    });

    //
    // Epilogue warps
    //
    tmem.relinquish_alloc_permit();
    if_(warp_idx.lt(MMA_WARP_ID), || {
        epilogue_warps(
            ka.ecfg,
            tidx,
            warp_idx,
            &mma_tile_coord_mnl,
            &thr_mma,
            ka.tma_atom_c,
            &t_ct_acc,
            &acc_shape,
            ka.m_c_mnl,
            &s_c,
            mma_tiler,
            &tiled_mma,
            &st.epi_tile,
            &acc_pipeline,
            stages.acc,
        );
    });

    // Dealloc the tensor memory buffer
    pipeline::sync(1);
    tmem.free(&tmem_ptr, 0);
}

/// The number of CTAs along mode `mode` of the cluster layout.
fn num_mcast_ctas(cluster_layout_vmnk: &Layout, mode: usize) -> usize {
    cluster_layout_vmnk.mode(mode).size()
}

/// `a` if `cond`, else `b` (the pipeline state of a conditional advance).
fn select_state(cond: Boolean, a: &PipelineState, b: &PipelineState) -> PipelineState {
    PipelineState::new(
        a.stages(),
        select(cond, a.count(), b.count()),
        select(cond, a.index(), b.index()),
        select(cond, a.phase(), b.phase()),
    )
}

/// The epilogue (warps 0 to 3): TMEM -> registers -> shared memory -> global memory through
/// [`epilogue_tma_store`], waiting for the accumulator and releasing it.
#[allow(clippy::too_many_arguments)]
pub fn epilogue_warps(
    ecfg: &EpilogueConfig,
    tidx: Int32,
    warp_idx: Int32,
    mma_tile_coord_mnl: &[Int32],
    thr_mma: &ThrMma,
    tma_atom_c: &CopyAtom<CDtype>,
    t_ct_acc: &Tensor<f32>,
    acc_shape: &Shape,
    tma_tensor_c: &Tensor<Ident>,
    s_c: &Tensor<CDtype>,
    mma_tiler: (usize, usize, usize),
    tiled_mma: &TiledMma,
    epi_tile: &Tiler,
    acc_pipeline: &PipelineUmmaAsync,
    num_acc_stage: usize,
) {
    // (bM, bN, RestM, RestN)
    let g_c_mnl = local_tile(tma_tensor_c, (mma_tiler.0, mma_tiler.1), (.., ..));
    // (MMA, MMA_M, MMA_N, RestM, RestN)
    let t_cg_c = thr_mma.partition_c(&g_c_mnl);
    // The accumulator with a stage mode: ((M, N), MMA_M, MMA_N, STAGE)
    let Tree::Branch(mut modes) = acc_shape.clone() else { panic!("a tuple shape") };
    modes.push(Shape::from(num_acc_stage));
    let staged = tiled_mma.make_fragment_c::<f32>(&Shape::from(Tree::Branch(modes)));
    let t_ct_acc = make_tensor(t_ct_acc.ptr().clone(), staged.layout().clone());
    let c_pipeline = PipelineTmaStore::create(
        ecfg.num_c_stage,
        CooperativeGroup::new(Agent::Thread, 32 * ecfg.epilogue_warp_id.len() as i32),
    );
    let acc_consumer_state = make_pipeline_state(PipelineUserType::Consumer, num_acc_stage);
    epilogue_tma_store::<f32, CDtype, _, _>(
        ecfg,
        tidx,
        warp_idx,
        tma_atom_c,
        &t_ct_acc,
        s_c,
        &t_cg_c,
        epi_tile,
        Int32::from(0),
        &|v| v,
        mma_tile_coord_mnl,
        &acc_consumer_state,
        acc_pipeline,
        &c_pipeline,
    );
    // Wait for C store complete
    if_(warp_idx.eq(ecfg.epilogue_warp_id[0] as i32), || c_pipeline.producer_tail());
}

/// A deterministic stream of numbers.
fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// The problem's data: A's Int4 values (row-major `(M, K)`), the scales (`(M, K / gk)`
/// row-major, powers of two from 1/4 to 4 with either sign; empty for ConvertOnly) and B's
/// values (row-major `(N, K)`, integers in `[-2, 2]`).
pub struct Problem {
    pub mnk: (usize, usize, usize),
    pub a: Vec<i8>,
    pub scale: Vec<f32>,
    pub b: Vec<f32>,
}

impl Problem {
    pub fn new(cfg: &MixedInputGemm, (m, n, k): (usize, usize, usize), seed: u64) -> Problem {
        let mut r = lcg(seed);
        let a: Vec<i8> = (0..m * k).map(|_| (r() % 16) as i8 - 8).collect();
        let b: Vec<f32> = (0..n * k).map(|_| (r() % 5) as f32 - 2.0).collect();
        let scale = if cfg.scale_mode() == TransformMode::ConvertScale {
            let kg = k / cfg.scale_granularity_k;
            (0..m * kg)
                .map(|_| {
                    let mag = [0.25f32, 0.5, 1.0, 2.0, 4.0][(r() % 5) as usize];
                    if r().is_multiple_of(2) { mag } else { -mag }
                })
                .collect()
        } else {
            Vec::new()
        };
        Problem { mnk: (m, n, k), a, scale, b }
    }

    /// A as the kernel reads it: K-major or M-major, shuffled within groups of 8 K values
    /// when `shuffle_a`.
    pub fn a_device(&self, cfg: &MixedInputGemm) -> Vec<I4> {
        let (m, _, k) = self.mnk;
        let a: Vec<I4> = self.a.iter().map(|&x| I4::new(x)).collect();
        if cfg.a_major == OperandMajorMode::K {
            if cfg.shuffle_a { shuffle_a_host(&a) } else { a }
        } else {
            (0..m * k).map(|i| a[(i % m) * k + i / m]).collect()
        }
    }

    /// The scales M-major, as the kernel reads them.
    pub fn scale_device(&self, cfg: &MixedInputGemm) -> Vec<BF16> {
        let (m, _, k) = self.mnk;
        if self.scale.is_empty() {
            return Vec::new();
        }
        let kg = k / cfg.scale_granularity_k;
        (0..m * kg).map(|i| BF16::from_f32(self.scale[(i % m) * kg + i / m])).collect()
    }

    /// `C = (A * scale) B^T` in Float32 (exact on these data), row-major.
    pub fn reference(&self, cfg: &MixedInputGemm) -> Vec<f32> {
        let (m, n, k) = self.mnk;
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                out[r * n + col] = (0..k)
                    .map(|i| {
                        let s = if self.scale.is_empty() {
                            1.0
                        } else {
                            self.scale[r * (k / cfg.scale_granularity_k) + i / cfg.scale_granularity_k]
                        };
                        self.a[r * k + i] as f32 * s * self.b[col * k + i]
                    })
                    .sum();
            }
        }
        out
    }
}

/// Simulate `cfg`'s kernel on an `m x n x k` problem under `opts` (and the extra schedule
/// `seeds`) and check C bit for bit.
pub fn run_with(cfg: &MixedInputGemm, mnk: (usize, usize, usize), opts: &SimOptions, seeds: &[u64]) -> SimReport {
    let (m, n, _) = mnk;
    let kernel = cfg.build(mnk);
    let p = Problem::new(cfg, mnk, 0x5eed + m as u64);
    let a = p.a_device(cfg);
    let scale = p.scale_device(cfg);
    let b: Vec<BF16> = p.b.iter().map(|&x| BF16::from_f32(x)).collect();
    let c = vec![-7.0f32; m * n];
    let mut args = vec![KernelArg::buffer(&a)];
    if !scale.is_empty() {
        args.push(KernelArg::buffer(&scale));
    }
    args.push(KernelArg::buffer(&b));
    args.push(KernelArg::buffer(&c));
    let report = kernel
        .simulate_schedules(cfg.launch(m, n), &mut args, opts, seeds)
        .unwrap_or_else(|e| panic!("{}: {e}", cfg.name));
    let got = args.last().unwrap().to_vec::<f32>();
    let want = p.reference(cfg);
    if let Some(i) = (0..m * n).find(|&i| got[i] != want[i]) {
        panic!("{}: C({}, {}) is {}, the reference {}", cfg.name, i / n, i % n, got[i], want[i]);
    }
    report
}

/// [`run_with`] under the default options.
pub fn run(cfg: &MixedInputGemm, mnk: (usize, usize, usize)) -> SimReport {
    run_with(cfg, mnk, &SimOptions::default(), &[])
}

#[allow(dead_code)]
fn main() {
    for (cfg, mnk) in [
        (MIXED_K_MAJOR_CONVERT, (128, 64, 256)),
        (MIXED_K_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_CONVERT, (128, 64, 256)),
        (MIXED_M_MAJOR_SCALE, (128, 64, 256)),
        (MIXED_M_MAJOR_2CTA, (256, 64, 256)),
    ] {
        println!("Running Blackwell mixed-input GEMM {} on {mnk:?} (simulated)", cfg.name);
        let report = run(&cfg, mnk);
        println!("{report}");
        println!("PASS");
    }
}

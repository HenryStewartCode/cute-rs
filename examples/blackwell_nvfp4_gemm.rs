// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py
//   (the kernel `Sm100BlockScaledDenseGemmKernel.kernel`, its host function `__call__`,
//   the module-level configuration, comments and names),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The first Blackwell tutorial NVFP4 GEMM, ported from CuTe DSL's `nvfp4_gemm_0.py`: a simple 1-CTA block-scaled tcgen05 GEMM,
//! `C = (A * SFA) (B * SFB)^T`.
//!
//! - A `(M, K)` and B `(N, K)` are K-major Float4E2M1FN (packed two per byte), each run of
//!   16 K elements scaled by one 8-bit scale factor (`sf_vec_size` 16): Float8E4M3FN (UE4M3,
//!   NVFP4, [`NVFP4_GEMM_0`]) or Float8E8M0FNU (UE8M0, [`NVFP4_GEMM_0_UE8M0`]). The scale
//!   factors are laid out in global memory by `tile_atom_to_shape_SF` (128-row by
//!   4-factor chunks). The accumulator is Float32 in tensor memory, C `(M, N)` row-major
//!   Float16.
//! - Each CTA computes one `(128, 256)` tile of C in k-tiles of 256 (`MmaMXF4NVF4Op`,
//!   `128x256x64`, four MMAs per k-tile), with a 4-stage `PipelineTmaUmma` (TMA loads of A,
//!   B, SFA and SFB into shared memory) and a 1-stage `PipelineUmmaAsync`, 128 threads.
//! - TMEM holds the accumulator (256 columns), then SFA and SFB
//!   (`find_tmem_tensor_col_offset`), laid out by `make_tmem_layout_sfa/sfb`. Each k-tile,
//!   warp 0 copies the stage's scale factors from shared memory into TMEM
//!   (`Cp4x32x128bOp`, built with `make_s2t_copy` and `get_s2t_smem_desc_tensor`), then
//!   issues the MMAs with the k-block's scale factors in `Field::SFA` and `Field::SFB`.
//! - All four warps run the epilogue: `tcgen05.ld.32x32b.x128`, Float16 conversion and a
//!   plain store to global memory.
//!
//! As in `blackwell_fp16_gemm`, what the Python host function builds (the tiled MMA, the
//! SMEM layouts and the TMA atoms) is built at the start of [`Nvfp4GemmConfig::build`] and
//! the kernel follows in [`kernel`]. Other differences:
//! - cute-rs's S2T copies of a broadcast atom (`Cp4x32x128bOp` writes each 32-lane block
//!   to all four sub-partitions) need the shared-memory side's broadcast mode spelt out:
//!   `tcgen05::append_s2t_broadcast_mode` before `partition_S`;
//! - the scale-factor TMA atoms move 8-bit elements (the tutorial's `internal_type=Int16`
//!   moves pairs of them as 16-bit ones; the boxes are the same bytes);
//! - the problem has no L (batch) mode, and `M`, `N` and `K` are launch values: `M` a
//!   multiple of 128, `N` of 256 and `K` of 256;
//! - the epilogue waits for its TMEM load (`fence_view_async_tmem_load`) before reading
//!   the registers.
//!
//! No Blackwell GPU is available to cute-rs's tests, so the kernel is verified by the
//! simulator, exactly: the data are E2M1 values in `[-2, 2]` and scale factors that are
//! powers of two, so every product and sum is exact in Float32 and C matches the CPU
//! reference (which decodes the FP4 and scale-factor codes itself) bit for bit; and by
//! `ptxas -arch=sm_100a`. `main` simulates a 256x512x512 problem with each variant.
//!
//! Run with `cargo run --release --example blackwell_nvfp4_gemm`.

use cute::cpasync;
use cute::cpasync::TmaAtomOptions;
use cute::memory::{SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions};
use cute::nvgpu::{
    make_tiled_tma_atom_a, make_tiled_tma_atom_a_with, make_tiled_tma_atom_b, make_tiled_tma_atom_b_with,
};
use cute::pipeline::{
    self, Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineTmaUmma, PipelineTmaUmmaOptions, PipelineUmmaAsync,
    PipelineUmmaOptions,
};
use cute::prelude::*;
use cute::tcgen05::{
    self, Cp4x32x128bOp, CtaGroup, Field, Ld32x32bOp, MmaMXF4NVF4Op, OperandSource, Repetition, make_tmem_copy,
};
use cute::utils::sm100::{make_smem_layout_a, make_smem_layout_b};
use cute::utils::{
    make_smem_layout_sfa, make_smem_layout_sfb, make_tmem_layout_sfa, make_tmem_layout_sfb, tile_atom_to_shape_sf,
};

/// The tutorial's configuration (the module-level constants of `nvfp4_gemm_0.py` and
/// `Sm100BlockScaledDenseGemmKernel.__init__`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4GemmConfig {
    /// The kernel's name.
    pub name: &'static str,
    /// The scale factors' type: Float8E4M3FN (NVFP4) or Float8E8M0FNU.
    pub sf_dtype: DType,
    pub mma_tiler_mn: (usize, usize),
    pub mma_inst_shape_k: usize,
    /// MMA instructions per k-tile along K.
    pub mma_inst_tile_k: usize,
    pub sf_vec_size: usize,
    pub threads_per_cta: usize,
    pub num_tmem_alloc_cols: usize,
    /// Pipeline stage configuration.
    pub num_ab_stage: usize,
    pub num_acc_stage: usize,
}

/// `nvfp4_gemm_0.py`'s configuration: UE4M3 scale factors.
pub const NVFP4_GEMM_0: Nvfp4GemmConfig = Nvfp4GemmConfig {
    name: "blackwell_nvfp4_gemm_0",
    sf_dtype: DType::F8E4M3FN,
    mma_tiler_mn: (128, 256),
    mma_inst_shape_k: 64,
    mma_inst_tile_k: 4,
    sf_vec_size: 16,
    threads_per_cta: 128,
    num_tmem_alloc_cols: 512,
    num_ab_stage: 4,
    num_acc_stage: 1,
};

/// The same kernel with UE8M0 scale factors (`kind::mxf4nvf4` with `scale_vec::4X` takes
/// either).
pub const NVFP4_GEMM_0_UE8M0: Nvfp4GemmConfig =
    Nvfp4GemmConfig { name: "blackwell_nvfp4_gemm_0_ue8m0", sf_dtype: DType::F8E8M0FNU, ..NVFP4_GEMM_0 };

/// Float4E2M1FN operands, Float16 output.
pub type AbDtype = F4E2M1FN;
pub type CDtype = F16;

cute_struct! {
    pub struct SharedStorage(num_ab_stage: usize, num_acc_stage: usize) {
        ab_mbar_ptr: MemRange<i64>[num_ab_stage * 2],
        acc_mbar_ptr: MemRange<i64>[num_acc_stage * 2],
        tmem_holding_buf: i32,
    }
}

impl Nvfp4GemmConfig {
    /// `(M, N, K)` of a CTA's tile: `(128, 256, 256)`.
    pub fn mma_tiler(&self) -> (usize, usize, usize) {
        (self.mma_tiler_mn.0, self.mma_tiler_mn.1, self.mma_inst_shape_k * self.mma_inst_tile_k)
    }

    /// The kernel: parameters `m`, `n`, `k` (launch values), then `a`, `b`, `sfa`, `sfb`
    /// and `c`. The first part is the tutorial's `__call__`.
    pub fn build(&self) -> Kernel {
        match self.sf_dtype {
            DType::F8E4M3FN => self.build_for::<F8E4M3FN>(),
            DType::F8E8M0FNU => self.build_for::<F8E8M0FNU>(),
            t => panic!("{}: the scale factors are Float8E4M3FN or Float8E8M0FNU, not {t}", self.name),
        }
    }

    fn build_for<S: Numeric>(&self) -> Kernel {
        let cfg = *self;
        Kernel::builder(cfg.name).block_dim([cfg.threads_per_cta as u32, 1, 1]).target(Arch::Sm100a).build(move || {
            let mma_tiler = cfg.mma_tiler();
            let (tm, tn, tk) = mma_tiler;
            let m = param_extent("m", tm);
            let n = param_extent("n", tn);
            let k = param_extent("k", tk);
            let a_tensor = param_tensor_aligned::<AbDtype>("a", Layout::new((m, k), (k, 1)), 32);
            let b_tensor = param_tensor_aligned::<AbDtype>("b", Layout::new((n, k), (k, 1)), 32);
            // Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
            // ((Atom_M, Rest_M),(Atom_K, Rest_K))
            let sfa_layout = tile_atom_to_shape_sf(a_tensor.layout().shape().clone(), cfg.sf_vec_size);
            let sfa_tensor = param_tensor_aligned::<S>("sfa", sfa_layout, 128);
            // ((Atom_N, Rest_N),(Atom_K, Rest_K))
            let sfb_layout = tile_atom_to_shape_sf(b_tensor.layout().shape().clone(), cfg.sf_vec_size);
            let sfb_tensor = param_tensor_aligned::<S>("sfb", sfb_layout, 128);
            let c_tensor = param_tensor_aligned::<CDtype>("c", Layout::new((m, n), (n, 1)), 32);

            let mma_op = MmaMXF4NVF4Op::new(
                S::DTYPE,
                (cfg.mma_tiler_mn.0, cfg.mma_tiler_mn.1, cfg.mma_inst_shape_k),
                CtaGroup::ONE,
                OperandSource::SMEM,
            );
            let tiled_mma = make_tiled_mma(mma_op, Layout::col_major((1, 1, 1)));

            // Compute A/B/SFA/SFB shared memory layout
            let a_smem_layout_staged =
                make_smem_layout_a(&tiled_mma, mma_tiler, AbDtype::DTYPE, cfg.num_ab_stage, None);
            let b_smem_layout_staged =
                make_smem_layout_b(&tiled_mma, mma_tiler, AbDtype::DTYPE, cfg.num_ab_stage, None);
            let sfa_smem_layout_staged = make_smem_layout_sfa(&tiled_mma, mma_tiler, cfg.sf_vec_size, cfg.num_ab_stage);
            let sfb_smem_layout_staged = make_smem_layout_sfb(&tiled_mma, mma_tiler, cfg.sf_vec_size, cfg.num_ab_stage);

            // TMA load for A
            let a_smem_layout = a_smem_layout_staged.select(&[0, 1, 2]);
            let tma_op = cpasync::CopyBulkTensorTileG2SOp { cta_group: CtaGroup::ONE };
            let (tma_atom_a, tma_tensor_a) =
                make_tiled_tma_atom_a(tma_op, &a_tensor, a_smem_layout.clone(), mma_tiler, &tiled_mma, None).into();
            // TMA load for B
            let b_smem_layout = b_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_b, tma_tensor_b) =
                make_tiled_tma_atom_b(tma_op, &b_tensor, b_smem_layout.clone(), mma_tiler, &tiled_mma, None).into();
            // TMA load for SFA (8-bit elements; the tutorial's internal_type=Int16 moves the
            // same bytes in pairs)
            let opts = TmaAtomOptions::new();
            let sfa_smem_layout = sfa_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_sfa, tma_tensor_sfa) = make_tiled_tma_atom_a_with(
                tma_op,
                &sfa_tensor,
                sfa_smem_layout.clone(),
                mma_tiler,
                &tiled_mma,
                None,
                &opts,
            )
            .into();
            // TMA load for SFB
            let sfb_smem_layout = sfb_smem_layout_staged.select(&[0, 1, 2]);
            let (tma_atom_sfb, tma_tensor_sfb) = make_tiled_tma_atom_b_with(
                tma_op,
                &sfb_tensor,
                sfb_smem_layout.clone(),
                mma_tiler,
                &tiled_mma,
                None,
                &opts,
            )
            .into();

            // Compute TMA load bytes
            let bytes = |n: IndexExpr| n.as_static().expect("the SMEM layouts are static");
            let a_copy_size = bytes(size_in_bytes::<AbDtype>(&a_smem_layout));
            let b_copy_size = bytes(size_in_bytes::<AbDtype>(&b_smem_layout));
            let sfa_copy_size = bytes(size_in_bytes::<S>(&sfa_smem_layout.filter_zeros()));
            let sfb_copy_size = bytes(size_in_bytes::<S>(&sfb_smem_layout.filter_zeros()));
            let num_tma_load_bytes = a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size;

            kernel::<S>(
                &cfg,
                tiled_mma,
                (&tma_atom_a, &tma_tensor_a),
                (&tma_atom_b, &tma_tensor_b),
                (&tma_atom_sfa, &tma_tensor_sfa),
                (&tma_atom_sfb, &tma_tensor_sfb),
                &c_tensor,
                (&a_smem_layout_staged, &b_smem_layout_staged),
                (&sfa_smem_layout_staged, &sfb_smem_layout_staged),
                num_tma_load_bytes,
            );
        })
    }

    /// The launch of an `(m, n)` output: one CTA per `(128, 256)` tile.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let (tm, tn) = self.mma_tiler_mn;
        LaunchConfig::new([m.div_ceil(tm) as u32, n.div_ceil(tn) as u32, 1], [self.threads_per_cta as u32, 1, 1])
    }
}

/// The tutorial's `kernel`.
#[allow(clippy::too_many_arguments)]
pub fn kernel<S: Numeric>(
    cfg: &Nvfp4GemmConfig,
    mut tiled_mma: TiledMma,
    (tma_atom_a, m_a_mkl): (&CopyAtom<AbDtype>, &Tensor<Ident>),
    (tma_atom_b, m_b_nkl): (&CopyAtom<AbDtype>, &Tensor<Ident>),
    (tma_atom_sfa, m_sfa_mkl): (&CopyAtom<S>, &Tensor<Ident>),
    (tma_atom_sfb, m_sfb_nkl): (&CopyAtom<S>, &Tensor<Ident>),
    m_c_mnl: &Tensor<CDtype>,
    (a_smem_layout_staged, b_smem_layout_staged): (&ComposedLayout, &ComposedLayout),
    (sfa_smem_layout_staged, sfb_smem_layout_staged): (&Layout, &Layout),
    num_tma_load_bytes: usize,
) {
    let mma_tiler = cfg.mma_tiler();
    let (num_ab_stage, num_acc_stage) = (cfg.num_ab_stage, cfg.num_acc_stage);
    let warp_idx = make_warp_uniform(warp_idx());
    let tidx = thread_idx().x;

    //
    // Setup cta/thread coordinates
    //
    let (bidx, bidy) = (block_idx().x, block_idx().y);
    let mma_tile_coord_mnl = (bidx / tiled_mma.thr_id().size() as i32, bidy, ..);

    //
    // Define shared storage for kernel
    //
    let smem = SmemAllocator::new();
    let storage_layout = SharedStorage::layout(num_ab_stage, num_acc_stage);
    let storage =
        SharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 8, SmemPartition::User));
    // (MMA, MMA_M, MMA_K, STAGE)
    let s_a = smem.allocate_tensor::<AbDtype>(a_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    // (MMA, MMA_N, MMA_K, STAGE)
    let s_b = smem.allocate_tensor::<AbDtype>(b_smem_layout_staged.clone(), 1024, None, SmemPartition::User);
    // (MMA, MMA_M, MMA_K, STAGE)
    let s_sfa = smem.allocate_tensor::<S>(sfa_smem_layout_staged.clone(), 128, None, SmemPartition::User);
    // (MMA, MMA_N, MMA_K, STAGE)
    let s_sfb = smem.allocate_tensor::<S>(sfb_smem_layout_staged.clone(), 128, None, SmemPartition::User);

    //
    // Initialize mainloop ab_pipeline, acc_pipeline and their states
    //
    let ab_pipeline = PipelineTmaUmma::create(
        num_ab_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, 1),
        num_tma_load_bytes,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.ab_mbar_ptr.data_ptr().cast::<u64>()),
            ..Default::default()
        },
    );
    let (ab_producer, ab_consumer) = ab_pipeline.make_participants();
    let acc_pipeline = PipelineUmmaAsync::create(
        num_acc_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, cfg.threads_per_cta as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_mbar_ptr.data_ptr().cast::<u64>()),
            ..Default::default()
        },
    );
    let (acc_producer, acc_consumer) = acc_pipeline.make_participants();

    //
    // Local_tile partition global tensors
    //
    // (bM, bK, RestK)
    let g_a = local_tile_proj(m_a_mkl, mma_tiler, mma_tile_coord_mnl, &[true, false, true]);
    // (bN, bK, RestK)
    let g_b = local_tile_proj(m_b_nkl, mma_tiler, mma_tile_coord_mnl, &[false, true, true]);
    let g_sfa = local_tile_proj(m_sfa_mkl, mma_tiler, mma_tile_coord_mnl, &[true, false, true]);
    let g_sfb = local_tile_proj(m_sfb_nkl, mma_tiler, mma_tile_coord_mnl, &[false, true, true]);
    // (bM, bN)
    let g_c = local_tile_proj(m_c_mnl, mma_tiler, mma_tile_coord_mnl, &[true, true, false]);
    let k_tile_cnt = size(&g_a, &[2]);

    //
    // Partition global tensor for TiledMMA_A/B/SFA/SFB/C
    //
    let thr_mma = tiled_mma.get_slice(0);
    // (MMA, MMA_M, MMA_K, RestK)
    let t_cg_a = thr_mma.partition_a(&g_a);
    // (MMA, MMA_N, MMA_K, RestK)
    let t_cg_b = thr_mma.partition_b(&g_b);
    // (MMA, MMA_M, MMA_K, RestK)
    let t_cg_sfa = thr_mma.partition_a(&g_sfa);
    // (MMA, MMA_N, MMA_K, RestK)
    let t_cg_sfb = thr_mma.partition_b(&g_sfb);
    // (MMA, MMA_M, MMA_N)
    let t_cg_c = thr_mma.partition_c(&g_c);

    //
    // Partition global/shared tensor for TMA load A/B/SFA/SFB
    //
    let one = make_layout(1, 0);
    // ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    let (t_as_a, t_ag_a) =
        cpasync::tma_partition(tma_atom_a, 0, &one, &s_a.group_modes(0, 3), &t_cg_a.group_modes(0, 3));
    let (t_bs_b, t_bg_b) =
        cpasync::tma_partition(tma_atom_b, 0, &one, &s_b.group_modes(0, 3), &t_cg_b.group_modes(0, 3));
    let filter_zeros = |t: &Tensor<_>| t.with_layout(t.layout().filter_zeros());
    let (t_as_sfa, t_ag_sfa) =
        cpasync::tma_partition(tma_atom_sfa, 0, &one, &s_sfa.group_modes(0, 3), &t_cg_sfa.group_modes(0, 3));
    let t_as_sfa = t_as_sfa.with_layout(t_as_sfa.layout().filter_zeros());
    let t_ag_sfa = t_ag_sfa.with_layout(t_ag_sfa.layout().filter_zeros());
    let (t_bs_sfb, t_bg_sfb) =
        cpasync::tma_partition(tma_atom_sfb, 0, &one, &s_sfb.group_modes(0, 3), &t_cg_sfb.group_modes(0, 3));
    let t_bs_sfb = t_bs_sfb.with_layout(t_bs_sfb.layout().filter_zeros());
    let t_bg_sfb = t_bg_sfb.with_layout(t_bg_sfb.layout().filter_zeros());

    //
    // Partition shared/tensor memory tensor for TiledMMA_A/B/C
    //
    // (MMA, MMA_M, MMA_K, STAGE)
    let t_cr_a = tiled_mma.make_fragment_a(&s_a);
    // (MMA, MMA_N, MMA_K, STAGE)
    let t_cr_b = tiled_mma.make_fragment_b(&s_b);
    // (MMA, MMA_M, MMA_N)
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler.0, mma_tiler.1));
    // (MMA, MMA_M, MMA_N)
    let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&acc_shape);

    //
    // Alloc tensor memory buffer
    //
    let tmem_alloc_barrier = NamedBarrier::new(1, cfg.threads_per_cta);
    let mut tmem = TmemAllocator::new(
        tmem_alloc_barrier,
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buf.ptr()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );
    tmem.allocate(cfg.num_tmem_alloc_cols);
    tmem.wait_for_alloc();
    let acc_tmem_ptr = tmem.retrieve_ptr::<f32>();
    let t_ct_acc = make_tensor(acc_tmem_ptr.clone(), t_ct_acc_fake.layout().clone());

    //
    // Make SFA/SFB tmem tensor
    //
    // Get SFA tmem ptr
    let acc_cols = tcgen05::find_tmem_tensor_col_offset(&t_ct_acc);
    let sfa_tmem_ptr = recast_ptr::<S, f32>(&(acc_tmem_ptr.clone() + acc_cols as i32), None);
    // (MMA, MMA_M, MMA_K)
    let t_ct_sfa_layout =
        make_tmem_layout_sfa(&tiled_mma, mma_tiler, cfg.sf_vec_size, &sfa_smem_layout_staged.select(&[0, 1, 2]));
    let t_ct_sfa = make_tensor(sfa_tmem_ptr, t_ct_sfa_layout);
    // Get SFB tmem ptr
    let sfa_cols = tcgen05::find_tmem_tensor_col_offset(&t_ct_sfa);
    let sfb_tmem_ptr = recast_ptr::<S, f32>(&(acc_tmem_ptr.clone() + (acc_cols + sfa_cols) as i32), None);
    // (MMA, MMA_N, MMA_K)
    let t_ct_sfb_layout =
        make_tmem_layout_sfb(&tiled_mma, mma_tiler, cfg.sf_vec_size, &sfb_smem_layout_staged.select(&[0, 1, 2]));
    let t_ct_sfb = make_tensor(sfb_tmem_ptr, t_ct_sfb_layout);

    //
    // Partition for S2T copy of SFA/SFB
    //
    // Make S2T CopyAtom
    let copy_atom_s2t = make_copy_atom::<S>(Cp4x32x128bOp::new(CtaGroup::ONE));
    let s2t = |s_sf: &Tensor<S>, t_sf: &Tensor<S>| {
        // (MMA, MMA_MN, MMA_K, STAGE)
        let t_cs_sf_compact = filter_zeros(s_sf);
        // (MMA, MMA_MN, MMA_K)
        let t_ct_sf_compact = filter_zeros(t_sf);
        let tiled_copy_s2t = tcgen05::make_s2t_copy(&copy_atom_s2t, &t_ct_sf_compact);
        let thr_copy_s2t = tiled_copy_s2t.get_slice(0);
        // ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        // (cute-rs: the broadcast mode of Cp4x32x128bOp spelt out on the SMEM side)
        let t_cs_sf_compact_s2t_ =
            thr_copy_s2t.partition_s(&tcgen05::append_s2t_broadcast_mode(copy_atom_s2t.op(), &t_cs_sf_compact));
        // ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        let t_cs_sf_compact_s2t = tcgen05::get_s2t_smem_desc_tensor(&tiled_copy_s2t, &t_cs_sf_compact_s2t_);
        // ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        let t_ct_sf_compact_s2t = thr_copy_s2t.partition_d(&t_ct_sf_compact);
        (tiled_copy_s2t, t_cs_sf_compact_s2t, t_ct_sf_compact_s2t)
    };
    let (tiled_copy_s2t_sfa, t_cs_sfa_compact_s2t, t_ct_sfa_compact_s2t) = s2t(&s_sfa, &t_ct_sfa);
    let (tiled_copy_s2t_sfb, t_cs_sfb_compact_s2t, t_ct_sfb_compact_s2t) = s2t(&s_sfb, &t_ct_sfb);

    //
    // Execute Data copy and Math computation in the k_tile loop
    //
    if_(warp_idx.eq(0), || {
        // Wait for accumulator buffer empty
        let mut acc_producer = acc_producer.clone_participant();
        let acc_empty = acc_producer.acquire_and_advance(None);
        // Set ACCUMULATE field to False for the first k_tile iteration
        tiled_mma.set(Field::ACCUMULATE, false);
        // Execute k_tile loop
        let (_, (_, _mma)) = for_range_pipelined(
            0,
            k_tile_cnt.clone(),
            1,
            Prefetch::stages(num_ab_stage as i64 - 2).bounded_by(num_ab_stage),
            (ab_producer.clone_participant(), (ab_consumer.clone_participant(), tiled_mma.clone())),
            |_, mut ab_producer| {
                // Wait for AB buffer empty
                let ab_empty = ab_producer.acquire_and_advance(None);
                // TMA load for A/B/SFA/SFB
                let bar = CopyArgs::new().tma_bar_ptr(&ab_empty.barrier());
                let (count, index) = (ab_empty.count(), ab_empty.index());
                copy_with(tma_atom_a, &t_ag_a.slice((.., count)), &t_as_a.slice((.., index)), bar.clone());
                copy_with(tma_atom_b, &t_bg_b.slice((.., count)), &t_bs_b.slice((.., index)), bar.clone());
                copy_with(tma_atom_sfa, &t_ag_sfa.slice((.., count)), &t_as_sfa.slice((.., index)), bar.clone());
                copy_with(tma_atom_sfb, &t_bg_sfb.slice((.., count)), &t_bs_sfb.slice((.., index)), bar);
                ab_producer
            },
            |_, (mut ab_consumer, mut tiled_mma)| {
                // Wait for AB buffer full
                let ab_full = ab_consumer.wait_and_advance(None);

                // Copy SFA/SFB to tmem
                let s2t_stage_coord = (.., .., .., .., ab_full.index());
                copy(&tiled_copy_s2t_sfa, &t_cs_sfa_compact_s2t.slice(s2t_stage_coord), &t_ct_sfa_compact_s2t);
                copy(&tiled_copy_s2t_sfb, &t_cs_sfb_compact_s2t.slice(s2t_stage_coord), &t_ct_sfb_compact_s2t);

                // tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                let num_kblocks = t_cr_a.layout().mode(2).size();
                for kblock_idx in 0..num_kblocks {
                    let kblock_coord = (.., .., kblock_idx, ab_full.index());

                    // Set SFA/SFB tensor to tiled_mma
                    let sf_kblock_coord = (.., .., kblock_idx);
                    tiled_mma.set_tensor(Field::SFA, &t_ct_sfa.slice(sf_kblock_coord));
                    tiled_mma.set_tensor(Field::SFB, &t_ct_sfb.slice(sf_kblock_coord));

                    gemm(&tiled_mma, &t_ct_acc, &t_cr_a.slice(kblock_coord), &t_cr_b.slice(kblock_coord), &t_ct_acc);
                    // Enable accumulate on tCtAcc after first kblock
                    tiled_mma.set(Field::ACCUMULATE, true);
                }

                // Async arrive AB buffer empty
                ab_full.release();
                (ab_consumer, tiled_mma)
            },
        );
        acc_empty.commit();
    });

    //
    // Epilogue
    // Partition for epilogue
    //
    let copy_atom_t2r = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X128));
    let tiled_copy_t2r = make_tmem_copy(&copy_atom_t2r, &t_ct_acc);
    let thr_copy_t2r = tiled_copy_t2r.get_slice(tidx);
    // (T2R_M, T2R_N, EPI_M, EPI_M)
    let t_tr_t_acc = thr_copy_t2r.partition_s(&t_ct_acc);
    // (T2R_M, T2R_N, EPI_M, EPI_N)
    let t_tr_g_c = thr_copy_t2r.partition_d(&t_cg_c);
    // (T2R_M, T2R_N, EPI_M, EPI_N)
    let t_tr_r_acc = make_fragment::<f32>(t_tr_g_c.shape().clone());
    // (T2R_M, T2R_N, EPI_M, EPI_N)
    let t_tr_r_c = make_fragment::<CDtype>(t_tr_g_c.shape().clone());

    // Release TMEM allocation lock
    tmem.relinquish_alloc_permit();

    // Wait for accumulator buffer full
    let mut acc_consumer = acc_consumer.clone_participant();
    let acc_full = acc_consumer.wait_and_advance(None);

    // Copy accumulator to register
    copy(&tiled_copy_t2r, &t_tr_t_acc, &t_tr_r_acc);
    // cute-rs makes the wait for the TMEM load explicit (tcgen05.wait::ld).
    fence_view_async_tmem_load();
    t_tr_r_c.store_ssa(&t_tr_r_acc.load_ssa().to::<CDtype>());
    // Store C to global memory
    autovec_copy(&t_tr_r_c, &t_tr_g_c);

    acc_full.release();

    // Deallocate TMEM
    pipeline::sync(1);
    tmem.free(&acc_tmem_ptr, 0);
}

// The host side: data, scale factors and the CPU reference.

/// The offset of scale factor `s` (of K elements `16 s ..`) of row `r` in a global
/// scale-factor tensor of `k` elements per row, as CUTLASS lays the chunks out: 512-byte
/// chunks of 128 rows by 4 factors (row `r` at `16 (r % 32) + 4 (r / 32 % 4)`), K-chunks
/// first. Written out independently of `tile_atom_to_shape_sf`.
pub fn sf_offset(r: usize, s: usize, k: usize, sf_vec_size: usize) -> usize {
    let k_chunks = k / sf_vec_size / 4;
    16 * (r % 32) + 4 * (r / 32 % 4) + s % 4 + 512 * (s / 4 + k_chunks * (r / 128))
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// `rows x k` E2M1 codes of values in `[-2, 2]` (steps of 0.5), row-major.
pub fn fp4_codes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = lcg(seed);
    (0..rows * k).map(|_| F4E2M1FN::from_f32((r() % 9) as f32 * 0.5 - 2.0).0).collect()
}

/// The E2M1 value of a 4-bit code, decoded here (sign, two exponent bits, one mantissa
/// bit; exponent bias 1, no infinities or NaN).
pub fn decode_e2m1(code: u8) -> f32 {
    let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((code >> 1) & 3, code & 1);
    let mag = if e == 0 { 0.5 * m as f32 } else { (1.0 + 0.5 * m as f32) * 2f32.powi(e as i32 - 1) };
    sign * mag
}

/// A buffer of codes as elements of type `T` (FP4 ones packed two per byte, the even
/// element in the low nibble).
pub fn typed<T: Numeric>(codes: &[u8]) -> KernelArg {
    KernelArg::buffer(&codes.iter().map(|&c| T::from_bits(c as u64)).collect::<Vec<T>>())
}

/// Scale-factor codes of the values 0.5, 1, 2 and 4: UE4M3 (`0x38` is 1.0) or UE8M0
/// (`127` is 1.0).
pub fn sf_codes(n: usize, seed: u64, sf_dtype: DType) -> Vec<u8> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let e = (r() % 4) as i32 - 1;
            if sf_dtype == DType::F8E4M3FN { (0x38 + 8 * e) as u8 } else { (127 + e) as u8 }
        })
        .collect()
}

/// The value of a scale-factor code, decoded here: UE4M3 (exponent bias 7, three mantissa
/// bits) or UE8M0 (a power of two, bias 127).
pub fn decode_sf(code: u8, sf_dtype: DType) -> f32 {
    if sf_dtype == DType::F8E4M3FN {
        let (e, m) = ((code >> 3) & 15, code & 7);
        if e == 0 { m as f32 / 8.0 * 2f32.powi(-6) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) }
    } else {
        2f32.powi(code as i32 - 127)
    }
}

/// One problem's inputs: the E2M1 codes of A and B, and the scale-factor codes of each
/// row's `k / 16` vectors.
pub struct Problem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub sfa: Vec<u8>,
    pub sfb: Vec<u8>,
}

impl Problem {
    pub fn new(cfg: &Nvfp4GemmConfig, m: usize, n: usize, k: usize) -> Problem {
        let spr = k / cfg.sf_vec_size;
        Problem {
            m,
            n,
            k,
            a: fp4_codes(m, k, 1111),
            b: fp4_codes(n, k, 2222),
            sfa: sf_codes(m * spr, 3333, cfg.sf_dtype),
            sfb: sf_codes(n * spr, 4444, cfg.sf_dtype),
        }
    }

    /// A scale-factor tensor in the global layout of [`sf_offset`].
    pub fn sf_image(&self, codes: &[u8], rows: usize, sf_vec_size: usize) -> Vec<u8> {
        let spr = self.k / sf_vec_size;
        let mut out = vec![0u8; rows * spr];
        for r in 0..rows {
            for s in 0..spr {
                out[sf_offset(r, s, self.k, sf_vec_size)] = codes[r * spr + s];
            }
        }
        out
    }

    /// `C = (A SFA) (B SFB)^T` in Float32 (exact for these data), converted to Float16.
    pub fn reference(&self, cfg: &Nvfp4GemmConfig) -> Vec<CDtype> {
        let (m, n, k, vs) = (self.m, self.n, self.k, cfg.sf_vec_size);
        let spr = k / vs;
        let dec = |codes: &[u8], sf: &[u8], rows: usize| -> Vec<f32> {
            (0..rows * k)
                .map(|i| decode_e2m1(codes[i]) * decode_sf(sf[(i / k) * spr + (i % k) / vs], cfg.sf_dtype))
                .collect()
        };
        let (a, b) = (dec(&self.a, &self.sfa, m), dec(&self.b, &self.sfb, n));
        let mut out = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                let acc: f32 = (0..k).map(|i| a[r * k + i] * b[c * k + i]).sum();
                out.push(CDtype::from_f32(acc));
            }
        }
        out
    }
}

/// Simulate `cfg`'s kernel on an `m x n x k` problem and check C bit for bit.
pub fn run(cfg: &Nvfp4GemmConfig, m: usize, n: usize, k: usize) -> SimReport {
    let kernel = cfg.build();
    let p = Problem::new(cfg, m, n, k);
    let (sfa, sfb) = (p.sf_image(&p.sfa, m, cfg.sf_vec_size), p.sf_image(&p.sfb, n, cfg.sf_vec_size));
    let sf = if cfg.sf_dtype == DType::F8E4M3FN { typed::<F8E4M3FN> } else { typed::<F8E8M0FNU> };
    let c = vec![CDtype::from_f32(0.0); m * n];
    let mut args = vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        typed::<AbDtype>(&p.a),
        typed::<AbDtype>(&p.b),
        sf(&sfa),
        sf(&sfb),
        KernelArg::buffer(&c),
    ];
    let report = kernel.simulate(cfg.launch(m, n), &mut args).unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
    let got = args[7].to_vec::<CDtype>();
    let want = p.reference(cfg);
    if let Some(i) = (0..m * n).find(|&i| got[i] != want[i]) {
        panic!("{}: {m}x{n}x{k}: C({}, {}) is {}, not {}", cfg.name, i / n, i % n, got[i].to_f32(), want[i].to_f32());
    }
    report
}

#[allow(dead_code)]
fn main() {
    for cfg in [NVFP4_GEMM_0, NVFP4_GEMM_0_UE8M0] {
        let (m, n, k) = (256, 512, 512);
        println!("===================================================================");
        println!("Running Blackwell NVFP4 GEMM example 0 ({}) with:", cfg.name);
        println!("  mnk:       {m},{n},{k} (simulated)");
        println!("  sf_dtype:  {}", cfg.sf_dtype);
        println!("===================================================================");
        let report = run(&cfg, m, n, k);
        println!("{report}");
        println!("PASS");
    }
}

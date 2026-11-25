// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py
//   (the kernel `kernel`, the host function `host_function`, their configuration,
//   comments and names, and the data of `run_dense_gemm`) and
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_1.py
//   (its 2-CTA, multicast configuration: the cluster layout, the leader CTA's share of
//   the mainloop, the multicast masks, the 2-CTA TMEM allocator, the pipelines' cluster
//   arguments, the producer tails and the grid), and
//   fp16_gemm_2.py, fp16_gemm_3.py, fp16_gemm_3_1.py and fp16_gemm_4.py in the same folder
//   (their configurations, `SharedStorage`, `kernel`, `cluster_specific_kernel`,
//   `compute_grid` and `host_function`: the warp roles, the persistent tile scheduling
//   loops, the CLC pipeline and scheduler warp, and the preferred/fallback cluster arms,
//   with their comments and names),
//     Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The first Blackwell tutorial GEMM, ported from CuTe DSL's `fp16_gemm_0.py`:
//! a simple 1-CTA tcgen05 dense GEMM, `C = A B^T`.
//!
//! - A `(M, K)` and B `(N, K)` are K-major Float16, the accumulator Float32 in
//!   tensor memory, C `(M, N)` row-major Float16.
//! - Each CTA computes one `(128, 256)` tile of C, in k-tiles of 64 with a
//!   4-stage `PipelineTmaUmma` (TMA loads into shared memory) and a 1-stage
//!   `PipelineUmmaAsync` (the accumulator), 128 threads.
//! - Warp 0 issues the TMA loads and the `128x256x16` MMAs (`cute.gemm` with
//!   `Field::ACCUMULATE`); all four warps run the epilogue, which loads the
//!   accumulator from TMEM (`tcgen05.ld.32x32b.x64`) in four subtiles,
//!   converts it to Float16 and stores it to global memory.
//! - TMEM is managed by a `TmemAllocator` (512 columns), with named barrier 1.
//!
//! The port keeps the kernel's structure and names. As cute-rs builds a kernel
//! in one Rust closure, what the Python `host_function` builds (the tiled MMA,
//! the SMEM layouts and the TMA atoms) is built at the start of
//! [`Fp16GemmConfig::build`], and the kernel body follows in [`kernel`]. Other
//! differences:
//! - the SMEM tiles are allocated with their swizzled layouts at 1024-byte
//!   alignment (the 128-byte swizzle's atom), not with the swizzle on the
//!   pointer;
//! - `cutlass.range(.., prefetch_stages=2)` is [`for_range_pipelined`] with
//!   the loop split into its TMA part and its MMA part;
//! - the epilogue waits for each TMEM load (`fence_view_async_tmem_load`,
//!   `tcgen05.wait::ld`) before reading its registers, which cute-rs's
//!   simulator requires to be explicit;
//! - `M`, `N` and `K` are launch values (the tutorial marks the layouts
//!   dynamic): `M` a multiple of 128, `N` of 256 and `K` of 64.
//!
//! No Blackwell GPU is available to cute-rs's tests, so the kernel is verified
//! by the simulator (bit for bit: the data are integers in `[-2, 2)`) and by
//! `ptxas -arch=sm_100a`; `main` simulates a 256x512x256 problem with
//! `fp16_gemm_0` and a 256x256x512 one with `fp16_gemm_1` (the tutorials'
//! default, 8192x8192x8192, is too large to simulate). `fp16_gemm_1` needs `M`
//! a multiple of 256.
//!
//! The tutorial's second step, `fp16_gemm_1.py`, is the configuration
//! [`FP16_GEMM_1`] of the same kernel (roadmap part m10-cta-pair-pipelines):
//! a `(2, 1, 1)` cluster whose CTA pair runs `256x256x16` MMAs
//! (`CtaGroup::TWO`) on `(256, 256, 64)` tiles with 7 A/B stages. Each CTA
//! loads its half of A and of B (multicast TMA loads, masks from
//! `create_tma_multicast_mask`) to the even CTA's full barrier, which expects
//! both CTAs' bytes; the even CTA alone issues the MMAs and commits (to both
//! CTAs); TMEM is allocated for the pair (`TmemAllocator` with `is_two_cta`);
//! both pipelines take the cluster layout `cta_layout_vmnk`, and the
//! producers run their tails before the block exits. Larger clusters (such
//! as `(2, 2, 1)`, which multicasts A across N) only change
//! `cluster_shape_mnk`.
//!
//! The tutorial's next steps (roadmap part m10-port-clc-clusters) are the configurations of
//! one warp-specialized kernel, [`Fp16GemmWsConfig::build`] (192 threads: epilogue warps 0
//! to 3, MMA warp 4, TMA warp 5, and scheduler warp 6 with CLC):
//! - [`FP16_GEMM_2`] (`fp16_gemm_2.py`): TMA, MMA and epilogue warp specialization on the
//!   CTA pair of `fp16_gemm_1` (6 A/B stages), with a TMA-store epilogue in subtiles;
//! - [`FP16_GEMM_3`] (`fp16_gemm_3.py`): persistent clusters with the
//!   `StaticPersistentTileScheduler` and 2 accumulator stages;
//! - [`FP16_GEMM_3_1`] (`fp16_gemm_3_1.py`): the `ClcDynamicPersistentTileScheduler`: CTA 0's
//!   scheduler warp cancels queued clusters (`clusterlaunchcontrol.try_cancel`) through a
//!   1-stage `PipelineClcFetchAsync` whose 16-byte response sits in the shared storage
//!   (`clc_mbar_ptr`, an aligned `clc_response`), and every role reads the next tile from it;
//!   [`Scheduling`] is the tutorial's `use_clc_dynamic_scheduler` switch;
//! - [`FP16_GEMM_4`] (`fp16_gemm_4.py`): preferred `(2, 4, 1)` and fallback `(2, 1, 1)`
//!   clusters, with CLC ([`FP16_GEMM_4_STATIC`] with the static scheduler). The kernel body
//!   is built once per cluster shape by `cute::utils::mixed_cluster_arms`, which branches on
//!   `is_preferred_cluster` (the cluster's `%cluster_nctaid`); each arm has its static
//!   cluster layout, multicast TMA atoms, pipelines and scheduler params, and the shared
//!   memory is allocated once for both (`smem_merge_branch_allocs`).
//!
//! Differences from the Python of these steps: the epilogue is the packaged
//! `epilogue_tma_store` (m10-epilogue-helpers), as in `blackwell_dense_gemm`, so the
//! accumulator's consumers are the epilogue warps (one arrival each); the leader CTA's MMA
//! warp issues a k-block at a time; every configuration publishes its barriers with one
//! cluster arrive and wait (`fp16_gemm_3.py`'s order) and drains the accumulator pipeline
//! (`acc_producer.tail()`) before exiting; the scheduler params are made in the kernel from
//! the runtime `M` and `N`; the CLC fields of the shared storage are always reserved; and
//! the static scheduler's fallback arm keeps the preferred work decomposition (see
//! `PersistentTileSchedulerParams::with_fallback_cluster_shape_mnk`), where the tutorial
//! passes the preferred params to both arms. `M`, `N` and `K` need not be tile multiples
//! (`N` and `K` multiples of 8): the TMA loads zero-fill and the stores clip, which also
//! covers the padding tiles of a grid rounded up to whole (preferred) clusters. Preferred
//! clusters and CLC have run only in the simulator, whose CLC queue hands out clusters of
//! the canceller's own shape.
//!
//! The epilogue is a hook (roadmap part m10-epilogue-helpers): [`Fp16GemmConfig::build_with`]
//! takes the declaration of C and a closure that drains the accumulator
//! ([`EpilogueCtx`] holds what the mainloop leaves it), so that the packaged epilogues of
//! `cute::utils::gemm::sm100` run on the same mainloop; [`Fp16GemmConfig::build`] is the
//! tutorial's own epilogue.
//!
//! Run with `cargo run --release --example blackwell_fp16_gemm`.

use cute::cpasync;
use cute::memory::{SmemAllocator, SmemPartition, TmemAllocator, TmemAllocatorOptions};
use cute::nvgpu::{make_tiled_tma_atom_a, make_tiled_tma_atom_b};
use cute::pipeline::{
    self, Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineBarriers, PipelineClcFetchAsync,
    PipelineClcFetchAsyncOptions, PipelineState, PipelineTmaStore, PipelineTmaUmma, PipelineTmaUmmaOptions,
    PipelineUmmaAsync, PipelineUmmaOptions, PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, Field, Ld32x32bOp, MmaF16BF16Op, OperandSource, Repetition, make_tmem_copy};
use cute::utils::gemm::sm100::{EpilogueConfig, epilogue_tma_store};
use cute::utils::sm100::{
    Sm100TmaLoadOp, cluster_shape_to_tma_atom_a, cluster_shape_to_tma_atom_b, make_smem_layout_a, make_smem_layout_b,
    make_smem_layout_epi,
};
use cute::utils::{
    ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerOptions,
    ClcDynamicPersistentTileSchedulerParams, LayoutEnum, PersistentTileSchedulerParams, StaticPersistentTileScheduler,
    WorkTileInfo, check_cluster_shape_in_arm, check_mixed_cluster_shapes, compute_epilogue_tile_shape,
    mixed_cluster_arms,
};

/// The tutorial's configuration (the module-level constants of `fp16_gemm_0.py`
/// and `fp16_gemm_1.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp16GemmConfig {
    /// The kernel's name.
    pub name: &'static str,
    /// The cluster; with [`CtaGroup::TWO`] its M extent is a multiple of 2.
    pub cluster_shape_mnk: (usize, usize, usize),
    /// [`CtaGroup::TWO`] for a CTA pair's MMA (`fp16_gemm_1`).
    pub cta_group: CtaGroup,
    pub mma_inst_shape_mnk: (usize, usize, usize),
    pub mma_tiler_mnk: (usize, usize, usize),
    pub threads_per_cta: usize,
    /// Pipeline stage configuration.
    pub ab_stages: usize,
    pub acc_stage: usize,
}

/// `fp16_gemm_0.py`'s configuration.
pub const FP16_GEMM_0: Fp16GemmConfig = Fp16GemmConfig {
    name: "blackwell_fp16_gemm_0",
    cluster_shape_mnk: (1, 1, 1),
    cta_group: CtaGroup::ONE,
    mma_inst_shape_mnk: (128, 256, 16),
    mma_tiler_mnk: (128, 256, 64),
    threads_per_cta: 128,
    ab_stages: 4,
    acc_stage: 1,
};

/// `fp16_gemm_1.py`'s configuration: a CTA pair, 7 stages.
pub const FP16_GEMM_1: Fp16GemmConfig = Fp16GemmConfig {
    name: "blackwell_fp16_gemm_1",
    cluster_shape_mnk: (2, 1, 1),
    cta_group: CtaGroup::TWO,
    mma_inst_shape_mnk: (256, 256, 16),
    mma_tiler_mnk: (256, 256, 64),
    threads_per_cta: 128,
    ab_stages: 7,
    acc_stage: 1,
};

/// Float16 operands and output, Float32 accumulation.
pub type IoDtype = F16;
pub const ACC_DTYPE: DType = DType::F32;

cute_struct! {
    pub struct SharedStorage(ab_stages: usize, acc_stage: usize) {
        ab_mbar_ptr: MemRange<i64>[ab_stages * 2],
        acc_mbar_ptr: MemRange<i64>[acc_stage * 2],
        tmem_dealloc_mbar: i64,
        tmem_holding_buf: i32,
    }
}

impl Fp16GemmConfig {
    /// The CTAs of an MMA: 2 for [`CtaGroup::TWO`].
    pub fn atom_ctas(&self) -> usize {
        if self.cta_group == CtaGroup::TWO { 2 } else { 1 }
    }

    /// The CTAs of a cluster.
    pub fn cluster_size(&self) -> usize {
        let (x, y, z) = self.cluster_shape_mnk;
        x * y * z
    }

    /// The kernel: parameters `m`, `n`, `k` (launch values), then `a`, `b`
    /// and `c`. The first part is the tutorial's `host_function`.
    pub fn build(&self) -> Kernel {
        self.build_with(self.threads_per_cta, tutorial_c, tutorial_epilogue)
    }

    /// [`build`](Self::build) with another C and epilogue: `make_c` (given `m` and `n`) declares the
    /// parameters after `a` and `b` (and anything built from C, such as a TMA store atom, which
    /// belongs in the host function) and returns what `epilogue` needs of them; `epilogue`
    /// runs on every thread once the mainloop is issued, with the accumulator stage full
    /// or filling (it waits itself). The accumulator pipeline's consumer group counts
    /// `acc_consumer_arrivals` arrivals per CTA (every thread of the CTA for the tutorial's
    /// epilogue, one per warp for the packaged ones).
    pub fn build_with<C: 'static>(
        &self,
        acc_consumer_arrivals: usize,
        make_c: impl Fn(&Fp16GemmConfig, &TiledMma, Int32, Int32) -> C + 'static,
        epilogue: impl Fn(&EpilogueCtx, &C) + 'static,
    ) -> Kernel {
        let cfg = *self;
        let name = cfg.name;
        Kernel::builder(name).block_dim([cfg.threads_per_cta as u32, 1, 1]).target(Arch::Sm100a).build(move || {
            let (tm, tn, tk) = cfg.mma_tiler_mnk;
            let m = param_extent("m", tm);
            let n = param_extent("n", tn);
            let k = param_extent("k", tk);
            let a = param_tensor_aligned::<IoDtype>("a", Layout::new((m, k), (k, 1)), 32);
            let b = param_tensor_aligned::<IoDtype>("b", Layout::new((n, k), (k, 1)), 32);

            // Construct tiled MMA
            let op = MmaF16BF16Op::new(
                IoDtype::DTYPE,
                ACC_DTYPE,
                cfg.mma_inst_shape_mnk,
                cfg.cta_group,
                OperandSource::SMEM,
                OperandMajorMode::K,
                OperandMajorMode::K,
            );
            let tiled_mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
            let c = make_c(&cfg, &tiled_mma, m, n);

            // Construct SMEM layouts for A and B
            let a_smem_layout = make_smem_layout_a(&tiled_mma, cfg.mma_tiler_mnk, IoDtype::DTYPE, cfg.ab_stages, None);
            let b_smem_layout = make_smem_layout_b(&tiled_mma, cfg.mma_tiler_mnk, IoDtype::DTYPE, cfg.ab_stages, None);
            let a_smem_layout_one_stage = a_smem_layout.select(&[0, 1, 2]);
            let b_smem_layout_one_stage = b_smem_layout.select(&[0, 1, 2]);

            // Construct the VMNK layout
            let cta_layout_mnk = Layout::col_major(cfg.cluster_shape_mnk);
            let cta_layout_vmnk =
                cta_layout_mnk.tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));

            // Construct TMA load atoms (fp16_gemm_1 multicasts within the cluster)
            let (op, cluster) = if cfg.cluster_size() > 1 {
                let op = cpasync::CopyBulkTensorTileG2SMulticastOp { cta_group: cfg.cta_group };
                (Sm100TmaLoadOp::Multicast(op), Some(cta_layout_vmnk.shape().clone()))
            } else {
                (Sm100TmaLoadOp::Tile(cpasync::CopyBulkTensorTileG2SOp { cta_group: cfg.cta_group }), None)
            };
            let (a_tma_atom, a_tma_tensor) =
                make_tiled_tma_atom_a(op, &a, a_smem_layout_one_stage, cfg.mma_tiler_mnk, &tiled_mma, cluster.clone())
                    .into();
            let (b_tma_atom, b_tma_tensor) =
                make_tiled_tma_atom_b(op, &b, b_smem_layout_one_stage, cfg.mma_tiler_mnk, &tiled_mma, cluster).into();

            kernel(
                &cfg,
                tiled_mma,
                &a_tma_atom,
                &a_tma_tensor,
                &b_tma_atom,
                &b_tma_tensor,
                &c,
                &a_smem_layout,
                &b_smem_layout,
                &cta_layout_vmnk,
                acc_consumer_arrivals,
                &epilogue,
            );
        })
    }

    /// The launch of an `(m, n)` output: one CTA per `(128, 256)` tile, or per
    /// half of a `(256, 256)` tile for a CTA pair, rounded up to whole clusters.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let (tm, tn, _) = self.mma_tiler_mnk;
        let (cx, cy, cz) = self.cluster_shape_mnk;
        let round = |x: usize, c: usize| x.div_ceil(c) * c;
        let grid = [round(m.div_ceil(tm / self.atom_ctas()), cx), round(n.div_ceil(tn), cy), cz];
        let launch = LaunchConfig::new(grid.map(|g| g as u32), [self.threads_per_cta as u32, 1, 1]);
        if self.cluster_size() > 1 { launch.with_cluster([cx as u32, cy as u32, cz as u32]) } else { launch }
    }
}

/// The tutorial's `kernel`.
#[allow(clippy::too_many_arguments)]
pub fn kernel<C>(
    cfg: &Fp16GemmConfig,
    mut tiled_mma: TiledMma,
    tma_atom_a: &CopyAtom<IoDtype>,
    m_a_mkl: &Tensor<Ident>,
    tma_atom_b: &CopyAtom<IoDtype>,
    m_b_nkl: &Tensor<Ident>,
    c: &C,
    a_smem_layout: &ComposedLayout,
    b_smem_layout: &ComposedLayout,
    cta_layout_vmnk: &Layout,
    acc_consumer_arrivals: usize,
    epilogue: &dyn Fn(&EpilogueCtx, &C),
) {
    let (ab_stages, acc_stage, threads_per_cta) = (cfg.ab_stages, cfg.acc_stage, cfg.threads_per_cta);
    let mma_tiler_mnk = cfg.mma_tiler_mnk;

    // Current thread/warp/block coordinates
    let tidx = thread_idx().x;
    let warp_idx = make_warp_uniform(warp_idx());
    let (bidx, bidy) = (block_idx().x, block_idx().y);
    let v = cta_layout_vmnk.mode(0).size();
    let clustered = cta_layout_vmnk.size() > 1;
    let cta_rank_in_cluster = block_idx_in_cluster();
    let cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster);
    let mma_coord_v = bidx % v as i32;
    let mma_coord_mnk_m = bidx / v as i32;
    let mma_coord_mnk = (mma_coord_mnk_m, bidy, ..);

    //
    // 1. Prepare args
    //

    // Allocate SMEM
    let smem = SmemAllocator::new();
    let storage_layout = SharedStorage::layout(ab_stages, acc_stage);
    let storage =
        SharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 8, SmemPartition::User));
    let s_a = smem.allocate_tensor::<IoDtype>(a_smem_layout.clone(), 1024, None, SmemPartition::User);
    let s_b = smem.allocate_tensor::<IoDtype>(b_smem_layout.clone(), 1024, None, SmemPartition::User);

    // Allocate all TMEM columns
    let tmem_alloc_barrier = NamedBarrier::new(1, threads_per_cta);
    let mut tmem = TmemAllocator::new(
        tmem_alloc_barrier,
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buf.ptr()),
            is_two_cta: v > 1,
            two_cta_tmem_dealloc_mbar_ptr: (v > 1).then(|| storage.tmem_dealloc_mbar.ptr().cast::<u64>()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );
    let num_tmem_cols = 512;
    tmem.allocate(num_tmem_cols);

    // Prefetch tma descriptor
    if_(warp_idx.eq(0), || {
        cpasync::prefetch_descriptor(tma_atom_a);
        cpasync::prefetch_descriptor(tma_atom_b);
    });

    // Pipeline configuration
    let num_tma_copy_bytes = size_in_bytes::<IoDtype>(&a_smem_layout.select(&[0, 1, 2]))
        + size_in_bytes::<IoDtype>(&b_smem_layout.select(&[0, 1, 2]));
    let num_tma_copy_bytes = num_tma_copy_bytes.as_static().expect("the SMEM layouts are static") * v;
    let num_mcast_ctas_a = cta_layout_vmnk.mode(2).size();
    let num_mcast_ctas_b = cta_layout_vmnk.mode(1).size();
    let num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1;
    let ab_pipeline = PipelineTmaUmma::create(
        ab_stages,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, num_tma_producer),
        num_tma_copy_bytes,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.ab_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cta_layout_vmnk.clone()),
            ..Default::default()
        },
    );
    let (ab_producer, ab_consumer) = ab_pipeline.make_participants();
    let acc_pipeline = PipelineUmmaAsync::create(
        acc_stage,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (v * acc_consumer_arrivals) as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cta_layout_vmnk.clone()),
            ..Default::default()
        },
    );
    let (acc_producer, acc_consumer) = acc_pipeline.make_participants();

    // Partition tensors for MMA and make fragments
    // (bM, bK, RestK)
    let g_a = local_tile_proj(m_a_mkl, mma_tiler_mnk, mma_coord_mnk, &[true, false, true]);
    // (bN, bK, RestK)
    let g_b = local_tile_proj(m_b_nkl, mma_tiler_mnk, mma_coord_mnk, &[false, true, true]);
    let thr_mma = tiled_mma.get_slice(mma_coord_v);
    // (MMA, MMA_M, MMA_K, RestK)
    let t_cg_a = thr_mma.partition_a(&g_a);
    // (MMA, MMA_N, MMA_K, RestK)
    let t_cg_b = thr_mma.partition_b(&g_b);
    // (MMA, MMA_M, MMA_K, STAGE)
    let t_cr_a = tiled_mma.make_fragment_a(&s_a);
    // (MMA, MMA_N, MMA_K, STAGE)
    let t_cr_b = tiled_mma.make_fragment_b(&s_b);
    // (MMA, MMA_M, MMA_N)
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler_mnk.0, mma_tiler_mnk.1));
    // (MMA, MMA_M, MMA_N)
    let t_ct_acc = tiled_mma.make_fragment_c::<f32>(&acc_shape);
    // Partition tensors for TMA; This requires the tensors partitioned for MMA
    let coord = |i: usize| match &cta_in_cluster_coord_vmnk {
        Tree::Branch(modes) => modes[i].clone(),
        leaf => leaf.clone(),
    };
    let (t_as_a, t_ag_a) = cpasync::tma_partition(
        tma_atom_a,
        coord(2),
        &make_layout(cta_layout_vmnk.mode(2).size(), 1),
        &s_a.group_modes(0, 3),
        &t_cg_a.group_modes(0, 3),
    );
    let (t_bs_b, t_bg_b) = cpasync::tma_partition(
        tma_atom_b,
        coord(1),
        &make_layout(cta_layout_vmnk.mode(1).size(), 1),
        &s_b.group_modes(0, 3),
        &t_cg_b.group_modes(0, 3),
    );
    // The CTAs that receive this CTA's share of A (along N) and of B (along M).
    let mcast_masks = clustered.then(|| {
        (
            cpasync::create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk.clone(), 2),
            cpasync::create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk.clone(), 1),
        )
    });

    // CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    // Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc();
    let tmem_ptr = tmem.retrieve_ptr::<f32>();
    // Swap the pointer in tCtAcc
    let t_ct_acc = make_tensor(tmem_ptr.clone(), t_ct_acc.layout().clone());

    //
    // 2. Main loop
    //
    // The even CTA of a pair issues the MMAs (every CTA without pairs).
    let is_leader_cta = mma_coord_v.eq(0);
    let as_leader = |f: &dyn Fn()| if v == 1 { f() } else { if_(is_leader_cta, f) };
    let num_k_tiles = size(&g_a, &[2]);
    if_(warp_idx.eq(0), || {
        // Wait for a empty accumulator buffer
        let mut acc_producer = acc_producer.clone_participant();
        let acc_empty = acc_producer.current_handle();
        acc_producer.advance(1);
        as_leader(&|| acc_empty.get_origin().producer_acquire(&acc_empty.state(), None));
        tiled_mma.set(Field::ACCUMULATE, false);
        let (_, (_, _mma)) = for_range_pipelined(
            0,
            num_k_tiles.clone(),
            1,
            Prefetch::stages(ab_stages as i64 - 2).bounded_by(ab_stages),
            (ab_producer.clone_participant(), (ab_consumer.clone_participant(), tiled_mma.clone())),
            // Issue TMA loads
            |_, mut ab_producer| {
                let ab_empty = ab_producer.acquire_and_advance(None);
                let bar = CopyArgs::new().tma_bar_ptr(&ab_empty.barrier());
                let (bar_a, bar_b) = match mcast_masks {
                    Some((a, b)) => (bar.clone().mcast_mask(a), bar.mcast_mask(b)),
                    None => (bar.clone(), bar),
                };
                let (count, index) = (ab_empty.count(), ab_empty.index());
                copy_with(tma_atom_a, &t_ag_a.slice((.., count)), &t_as_a.slice((.., index)), bar_a);
                copy_with(tma_atom_b, &t_bg_b.slice((.., count)), &t_bs_b.slice((.., index)), bar_b);
                ab_producer
            },
            // Execute one K-block worth of MMA instructions
            // (on the leader CTA of a pair only)
            |_, (mut ab_consumer, mut tiled_mma)| {
                let ab_full = ab_consumer.current_handle();
                ab_consumer.advance(1);
                as_leader(&|| {
                    let mut tiled_mma = tiled_mma.clone();
                    ab_full.get_origin().consumer_wait(&ab_full.state(), None);
                    let num_k_blocks = t_cr_a.layout().mode(2).size();
                    for k_block_idx in 0..num_k_blocks {
                        let k_block_coord = (.., .., k_block_idx, ab_full.index());
                        gemm(
                            &tiled_mma,
                            &t_ct_acc,
                            &t_cr_a.slice(k_block_coord),
                            &t_cr_b.slice(k_block_coord),
                            &t_ct_acc,
                        );
                        tiled_mma.set(Field::ACCUMULATE, true);
                    }
                    // Signal that the A/B buffers have been consumed and are ready for the next load
                    ab_full.release();
                });
                tiled_mma.set(Field::ACCUMULATE, true);
                (ab_consumer, tiled_mma)
            },
        );
        // Signal that the accumulator is fully computed
        as_leader(&|| acc_empty.commit());
    });

    //
    // 3. Epilogue
    //

    // Release TMEM allocation lock
    tmem.relinquish_alloc_permit();

    // The epilogue: TMEM -> RMEM -> GMEM (it waits for the accumulator and releases it).
    epilogue(
        &EpilogueCtx {
            cfg,
            tiled_mma: &tiled_mma,
            thr_mma: &thr_mma,
            t_ct_acc: &t_ct_acc,
            acc_shape: &acc_shape,
            tmem_ptr: &tmem_ptr,
            tile_m: mma_coord_mnk_m,
            tile_n: bidy,
            tidx,
            warp_idx,
            smem: &smem,
            acc_pipeline: &acc_pipeline,
            acc_consumer_state: acc_consumer.state(),
        },
        c,
    );

    // Ensure used buffers are properly synchronized before producer exit.
    // This could avoid the invalid dsmem access due to early leading CTA exit.
    // (fp16_gemm_1; one CTA's epilogue waits make it unnecessary in fp16_gemm_0.) The
    // producers' states after the mainloop are rebuilt here: `num_k_tiles` A/B stages
    // and one accumulator stage on.
    if clustered {
        if_(warp_idx.eq(0), || {
            let mut ab_producer = ab_producer.clone_participant();
            ab_producer.advance(IntoValue::<i32>::into_value(num_k_tiles.clone()));
            ab_producer.tail();
            as_leader(&|| {
                let mut acc_producer = acc_producer.clone_participant();
                acc_producer.advance(1);
                acc_producer.tail();
            });
        });
    }

    // Deallocate TMEM
    pipeline::sync(1);
    tmem.free(&tmem_ptr, 0);
}

/// What the mainloop leaves the epilogue (roadmap part m10-epilogue-helpers).
pub struct EpilogueCtx<'a> {
    pub cfg: &'a Fp16GemmConfig,
    pub tiled_mma: &'a TiledMma,
    pub thr_mma: &'a ThrMma,
    /// The accumulator `(MMA, MMA_M, MMA_N)` in TMEM (one stage).
    pub t_ct_acc: &'a Tensor<f32>,
    /// Its shape, `partition_shape_c` of the MMA tile.
    pub acc_shape: &'a Shape,
    /// The TMEM allocation's start.
    pub tmem_ptr: &'a Ptr<f32>,
    /// This CTA's tile of C, in tiles of `mma_tiler_mnk` (the CTA pair's M coordinate).
    pub tile_m: Int32,
    pub tile_n: Int32,
    pub tidx: Int32,
    pub warp_idx: Int32,
    pub smem: &'a SmemAllocator,
    pub acc_pipeline: &'a PipelineUmmaAsync,
    /// The consumer state of the accumulator stage, before its wait.
    pub acc_consumer_state: PipelineState,
}

/// `fp16_gemm_0.py`'s C: `(M, N)` row-major Float16.
pub fn tutorial_c(_: &Fp16GemmConfig, _: &TiledMma, m: Int32, n: Int32) -> Tensor<IoDtype> {
    param_tensor_aligned::<IoDtype>("c", Layout::new((m, n), (n, 1)), 32)
}

/// The tutorial's epilogue: `tcgen05.ld.32x32b.x64` in four subtiles, Float16 conversion and
/// `autovec_copy`.
pub fn tutorial_epilogue(ctx: &EpilogueCtx, m_c_mnl: &Tensor<IoDtype>) {
    let mma_coord_mnk = (ctx.tile_m, ctx.tile_n, ..);
    // (bM, bN)
    let g_c = local_tile_proj(m_c_mnl, ctx.cfg.mma_tiler_mnk, mma_coord_mnk, &[true, true, false]);
    // (MMA, MMA_M, MMA_N)
    let t_cg_c = ctx.thr_mma.partition_c(&g_c);
    let t_ct_acc = ctx.t_ct_acc;
    let tidx = ctx.tidx;

    let subtile_cnt = 4;
    // (EpiTile)
    let acc_mma = t_ct_acc.layout().mode(0);
    let epi_tiler = ((acc_mma.mode(0).size(), acc_mma.mode(1).size() / subtile_cnt),);
    // (EpiTile, NumTiles)
    let t_ct_acc_epi = zipped_divide(t_ct_acc, epi_tiler);
    // (EpiTile, NumTiles)
    let g_c_epi = zipped_divide(&t_cg_c, epi_tiler);

    // Every thread loads 64 x fp32
    let tmem_atom = make_copy_atom::<f32>(Ld32x32bOp::new(Repetition::X64));
    let tmem_tiled_copy = make_tmem_copy(&tmem_atom, &t_ct_acc_epi.slice((.., 0)));
    let tmem_thr_copy = tmem_tiled_copy.get_slice(tidx);

    // (TmemCpy,NumTmemCpy,NumTiles)
    let t_dt_c = tmem_thr_copy.partition_s(&t_ct_acc_epi);
    // (TmemCpy,NumTmemCpy,NumTiles)
    let t_dg_c = tmem_thr_copy.partition_d(&g_c_epi);

    // (TmemCpy,NumTmemCpy)
    let t_dg_c0 = t_dg_c.slice((.., .., 0));
    let t_cr_acc = make_fragment::<f32>(t_dg_c0.shape().clone());
    // (TmemCpy,NumTmemCpy)
    let t_cr_c = make_fragment::<IoDtype>(t_dg_c0.shape().clone());

    // Wait for the accumulator buffer to be full
    ctx.acc_pipeline.consumer_wait(&ctx.acc_consumer_state, None);

    // TMEM -> RMEM -> GEMM
    // Sub-tiling for better instruction-level parallelism
    let num_tiles = t_dt_c.layout().mode(2).size() as i32;
    for_range(0, num_tiles, 1, |i| {
        copy(&tmem_tiled_copy, &t_dt_c.slice((.., .., i)), &t_cr_acc);
        // cute-rs makes the wait for the TMEM load explicit (tcgen05.wait::ld).
        fence_view_async_tmem_load();
        t_cr_c.store_ssa(&t_cr_acc.load_ssa().to::<IoDtype>());
        autovec_copy(&t_cr_c, &t_dg_c.slice((.., .., i)));
    });
    ctx.acc_pipeline.consumer_release(&ctx.acc_consumer_state);
}

// ---------------------------------------------------------------------------------------
// fp16_gemm_2, 3, 3_1 and 4: warp specialization, persistent tile schedulers, cluster
// launch control and preferred clusters (roadmap part m10-port-clc-clusters).
// ---------------------------------------------------------------------------------------

/// How the warp-specialized kernel's CTAs find their tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduling {
    /// One tile per CTA, from its block index (`fp16_gemm_2.py`).
    OneTile,
    /// `StaticPersistentTileScheduler` (`fp16_gemm_3.py`, and `fp16_gemm_3_1.py` and
    /// `fp16_gemm_4.py` with `use_clc_dynamic_scheduler = False`).
    StaticPersistent,
    /// `ClcDynamicPersistentTileScheduler` with a scheduler warp (`fp16_gemm_3_1.py`,
    /// `fp16_gemm_4.py`).
    ClcDynamic,
}

/// The configuration of the warp-specialized tutorial GEMMs (the module-level constants
/// of `fp16_gemm_2.py` to `fp16_gemm_4.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp16GemmWsConfig {
    pub name: &'static str,
    pub use_2cta_instrs: bool,
    /// The cluster (the preferred cluster with a fallback).
    pub cluster_shape_mnk: (usize, usize, usize),
    /// `fp16_gemm_4.py`'s fallback cluster: the launch's clusters run at either shape.
    pub fallback_cluster_shape_mnk: Option<(usize, usize, usize)>,
    pub mma_inst_shape_mnk: (usize, usize, usize),
    pub mma_tiler_mnk: (usize, usize, usize),
    /// Epilogue threads per CTA.
    pub threads_in_epilogue: usize,
    /// Pipeline stage configuration.
    pub ab_stages: usize,
    pub epi_stages: usize,
    pub acc_stages: usize,
    pub scheduling: Scheduling,
    pub num_clc_stage: usize,
}

/// `fp16_gemm_2.py`: TMA, MMA and epilogue warps, a TMA-store epilogue, one tile per CTA
/// pair.
pub const FP16_GEMM_2: Fp16GemmWsConfig = Fp16GemmWsConfig {
    name: "blackwell_fp16_gemm_2",
    use_2cta_instrs: true,
    cluster_shape_mnk: (2, 1, 1),
    fallback_cluster_shape_mnk: None,
    mma_inst_shape_mnk: (256, 256, 16),
    mma_tiler_mnk: (256, 256, 64),
    threads_in_epilogue: 128,
    ab_stages: 6,
    epi_stages: 2,
    acc_stages: 1,
    scheduling: Scheduling::OneTile,
    num_clc_stage: 1,
};

/// `fp16_gemm_3.py`: the static persistent tile scheduler, 2 accumulator stages.
pub const FP16_GEMM_3: Fp16GemmWsConfig = Fp16GemmWsConfig {
    name: "blackwell_fp16_gemm_3",
    acc_stages: 2,
    scheduling: Scheduling::StaticPersistent,
    ..FP16_GEMM_2
};

/// `fp16_gemm_3_1.py`: the cluster launch control scheduler (`use_clc_dynamic_scheduler`),
/// one response stage.
pub const FP16_GEMM_3_1: Fp16GemmWsConfig =
    Fp16GemmWsConfig { name: "blackwell_fp16_gemm_3_1", scheduling: Scheduling::ClcDynamic, ..FP16_GEMM_3 };

/// `fp16_gemm_4.py`: preferred `(2, 4, 1)` and fallback `(2, 1, 1)` clusters, with CLC.
pub const FP16_GEMM_4: Fp16GemmWsConfig = Fp16GemmWsConfig {
    name: "blackwell_fp16_gemm_4",
    cluster_shape_mnk: (2, 4, 1),
    fallback_cluster_shape_mnk: Some((2, 1, 1)),
    ..FP16_GEMM_3_1
};

/// `fp16_gemm_4.py` with `use_clc_dynamic_scheduler = False`: preferred clusters with the
/// static persistent scheduler.
pub const FP16_GEMM_4_STATIC: Fp16GemmWsConfig =
    Fp16GemmWsConfig { name: "blackwell_fp16_gemm_4_static", scheduling: Scheduling::StaticPersistent, ..FP16_GEMM_4 };

cute_struct! {
    pub struct WsSharedStorage(ab_stages: usize, acc_stages: usize, num_clc_stage: usize) {
        ab_mbar_ptr: MemRange<i64>[ab_stages * 2],
        acc_mbar_ptr: MemRange<i64>[acc_stages * 2],
        tmem_dealloc_mbar: i64,
        tmem_holding_buffer: i32,
        // Only for CLC Dynamic Scheduler
        clc_mbar_ptr: MemRange<i64>[num_clc_stage * 2],
        #[align(16)]
        clc_response: MemRange<i32>[4],
    }
}

// The warp roles.
const EPILOGUE_WARP_IDS: [i32; 4] = [0, 1, 2, 3];
const MMA_WARP_ID: i32 = 4;
const TMA_WARP_ID: i32 = 5;
// sched_warp_id only for dynamic scheduler
const SCHED_WARP_ID: i32 = 6;
// Response size is 4B * 4 elements
const NUM_CLC_RESPONSE_BYTES: usize = 16;

/// What depends on the cluster shape: the cluster layout and the TMA load atoms (multicast
/// across it). The host function builds one per cluster shape.
pub struct ClusterSetup {
    pub cluster_shape_mnk: (usize, usize, usize),
    pub cta_layout_vmnk: Layout,
    pub tma_atom_a: CopyAtom<IoDtype>,
    pub tma_tensor_a: Tensor<Ident>,
    pub tma_atom_b: CopyAtom<IoDtype>,
    pub tma_tensor_b: Tensor<Ident>,
}

impl Fp16GemmWsConfig {
    /// The CTAs of an MMA.
    pub fn atom_ctas(&self) -> usize {
        if self.use_2cta_instrs { 2 } else { 1 }
    }

    pub fn cta_group(&self) -> CtaGroup {
        if self.use_2cta_instrs { CtaGroup::TWO } else { CtaGroup::ONE }
    }

    /// 192 threads (four epilogue warps, an MMA and a TMA warp), 224 with the scheduler warp.
    pub fn threads_per_cta(&self) -> usize {
        if self.scheduling == Scheduling::ClcDynamic { 224 } else { 192 }
    }

    /// The tile of one CTA.
    pub fn cta_tile_shape_mnk(&self) -> (usize, usize, usize) {
        let (m, n, k) = self.mma_tiler_mnk;
        (m / self.atom_ctas(), n, k)
    }

    /// The CTA tiles of an `(m, n)` C (`compute_grid`'s `num_ctas_mn`).
    pub fn num_ctas_mn(&self, m: usize, n: usize) -> (usize, usize) {
        let (tm, tn, _) = self.cta_tile_shape_mnk();
        (m.div_ceil(tm), n.div_ceil(tn))
    }

    /// The static persistent schedule of an `(m, n)` C (with the fallback shape, if any).
    pub fn static_params(&self, ntiles: impl Into<Shape>) -> PersistentTileSchedulerParams {
        let p = PersistentTileSchedulerParams::new(ntiles, shape3(self.cluster_shape_mnk), Default::default());
        match self.fallback_cluster_shape_mnk {
            Some(f) => p.with_fallback_cluster_shape_mnk(shape3(f)),
            None => p,
        }
    }

    /// The cluster launch control schedule (with the fallback shape's, if any).
    pub fn clc_params(&self, ntiles: impl Into<Shape>) -> ClcDynamicPersistentTileSchedulerParams {
        let options = ClcDynamicPersistentTileSchedulerOptions {
            fallback_cluster_shape_mnk: self.fallback_cluster_shape_mnk.map(shape3),
            ..Default::default()
        };
        ClcDynamicPersistentTileSchedulerParams::new(ntiles, shape3(self.cluster_shape_mnk), options)
    }

    /// The grid of an `(m, n)` C (`compute_grid`): one CTA per tile rounded up to whole
    /// clusters for [`Scheduling::OneTile`], at most `max_active_clusters` clusters for
    /// the static scheduler, the whole problem in clusters for CLC.
    pub fn grid(&self, m: usize, n: usize, max_active_clusters: u32) -> [u32; 3] {
        let (tm, tn) = self.num_ctas_mn(m, n);
        match self.scheduling {
            Scheduling::OneTile => {
                let (cx, cy, cz) = self.cluster_shape_mnk;
                let round = |x: usize, c: usize| (x.div_ceil(c) * c) as u32;
                [round(tm, cx), round(tn, cy), cz as u32]
            }
            Scheduling::StaticPersistent => self.static_params((tm, tn, 1)).get_grid_shape(max_active_clusters),
            Scheduling::ClcDynamic => self.clc_params((tm, tn, 1)).get_grid_shape(),
        }
    }

    /// The launch of `grid`: the cluster, and the fallback cluster of `fp16_gemm_4`. Panics
    /// when the cluster shapes break the tutorial's constraints.
    pub fn launch_grid(&self, grid: [u32; 3]) -> LaunchConfig {
        let launch = LaunchConfig::new(grid, [self.threads_per_cta() as u32, 1, 1]);
        let (cx, cy, cz) = self.cluster_shape_mnk;
        let cluster = [cx as u32, cy as u32, cz as u32];
        match self.fallback_cluster_shape_mnk {
            Some(f) => {
                let (preferred, fallback) = (shape3(self.cluster_shape_mnk), shape3(f));
                if let Err(e) = check_mixed_cluster_shapes(preferred, fallback, Some(grid)) {
                    panic!("{}: {e}", self.name);
                }
                launch.with_fallback_cluster(cluster, [f.0 as u32, f.1 as u32, f.2 as u32])
            }
            None if cx * cy * cz > 1 => launch.with_cluster(cluster),
            None => launch,
        }
    }

    /// [`launch_grid`](Self::launch_grid) of [`grid`](Self::grid).
    pub fn launch(&self, m: usize, n: usize, max_active_clusters: u32) -> LaunchConfig {
        self.launch_grid(self.grid(m, n, max_active_clusters))
    }

    /// The kernel: parameters `m`, `n`, `k` (launch values), then `a`, `b` and `c`. The
    /// first part is the tutorial's `host_function`.
    pub fn build(&self) -> Kernel {
        let cfg = *self;
        let threads = cfg.threads_per_cta() as u32;
        if let Some(f) = cfg.fallback_cluster_shape_mnk
            && let Err(e) = check_mixed_cluster_shapes(shape3(cfg.cluster_shape_mnk), shape3(f), None)
        {
            panic!("{}: {e}", cfg.name);
        }
        Kernel::builder(cfg.name).block_dim([threads, 1, 1]).target(Arch::Sm100a).build(move || {
            let mma_tiler_mnk = cfg.mma_tiler_mnk;
            let m = param_extent("m", 1);
            let n = param_extent("n", 8);
            let k = param_extent("k", 8);
            let a = param_tensor_aligned::<IoDtype>("a", Layout::new((m, k), (k, 1)), 32);
            let b = param_tensor_aligned::<IoDtype>("b", Layout::new((n, k), (k, 1)), 32);
            let c = param_tensor_aligned::<IoDtype>("c", Layout::new((m, n), (n, 1)), 32);

            //
            // Construct tiled MMA
            //
            let op = MmaF16BF16Op::new(
                IoDtype::DTYPE,
                ACC_DTYPE,
                cfg.mma_inst_shape_mnk,
                cfg.cta_group(),
                OperandSource::SMEM,
                OperandMajorMode::K,
                OperandMajorMode::K,
            );
            let tiled_mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));

            //
            // Construct SMEM layouts for A and B
            //
            let a_smem_layout = make_smem_layout_a(&tiled_mma, mma_tiler_mnk, IoDtype::DTYPE, cfg.ab_stages, None);
            let b_smem_layout = make_smem_layout_b(&tiled_mma, mma_tiler_mnk, IoDtype::DTYPE, cfg.ab_stages, None);

            // c_smem_layout_kind is an enum for row/column major, not a CuTe layout
            let c_smem_layout_kind = LayoutEnum::from_tensor(&c);

            //
            // Construct the VMNK layout and the TMA load atoms, for each cluster shape
            //
            let setup = |cluster_shape_mnk: (usize, usize, usize)| {
                let cta_layout_mnk = Layout::col_major(cluster_shape_mnk);
                let cta_layout_vmnk =
                    cta_layout_mnk.tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));
                let (cx, cy, cz) = cluster_shape_mnk;
                let cluster = (cx * cy * cz > 1).then(|| cta_layout_vmnk.shape().clone());
                let a_op = cluster_shape_to_tma_atom_a(cluster_shape_mnk, tiled_mma.thr_id())
                    .unwrap_or_else(|e| panic!("{e}"));
                let b_op = cluster_shape_to_tma_atom_b(cluster_shape_mnk, tiled_mma.thr_id())
                    .unwrap_or_else(|e| panic!("{e}"));
                let (tma_atom_a, tma_tensor_a) = make_tiled_tma_atom_a(
                    a_op,
                    &a,
                    a_smem_layout.select(&[0, 1, 2]),
                    mma_tiler_mnk,
                    &tiled_mma,
                    cluster.clone(),
                )
                .into();
                let (tma_atom_b, tma_tensor_b) = make_tiled_tma_atom_b(
                    b_op,
                    &b,
                    b_smem_layout.select(&[0, 1, 2]),
                    mma_tiler_mnk,
                    &tiled_mma,
                    cluster,
                )
                .into();
                ClusterSetup { cluster_shape_mnk, cta_layout_vmnk, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b }
            };
            let preferred = setup(cfg.cluster_shape_mnk);
            let fallback = cfg.fallback_cluster_shape_mnk.map(setup);

            let cta_tile_shape_mnk = cfg.cta_tile_shape_mnk();
            let epi_tile = compute_epilogue_tile_shape(
                &[cta_tile_shape_mnk.0, cta_tile_shape_mnk.1],
                cfg.use_2cta_instrs,
                c_smem_layout_kind,
                IoDtype::DTYPE,
                None,
                None,
                None,
            );
            let epi_smem_layout_staged =
                make_smem_layout_epi(IoDtype::DTYPE, c_smem_layout_kind, &epi_tile, cfg.epi_stages);
            let (tma_atom_c, tma_tensor_c) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &c,
                epi_smem_layout_staged.slice((.., .., 0)),
                epi_tile.clone(),
                1,
            )
            .into();

            // The tile counts of C (`compute_grid`), for the schedulers.
            let ntile = |e: Int32, t: usize| IndexExpr::from((e + (t as i32 - 1)) / t as i32);
            let num_ctas_mn = Shape::from(Tree::Branch(vec![
                Tree::Leaf(ntile(m, cta_tile_shape_mnk.0)),
                Tree::Leaf(ntile(n, cta_tile_shape_mnk.1)),
                Tree::Leaf(IndexExpr::from(1)),
            ]));

            ws_kernel(&WsKernelArgs {
                cfg: &cfg,
                tiled_mma: &tiled_mma,
                preferred: &preferred,
                fallback: fallback.as_ref(),
                tma_atom_c: &tma_atom_c,
                m_c_mnl: &tma_tensor_c,
                a_smem_layout: &a_smem_layout,
                b_smem_layout: &b_smem_layout,
                c_smem_layout_kind,
                epi_smem_layout_staged: &epi_smem_layout_staged,
                epi_tile: &epi_tile,
                num_ctas_mn: &num_ctas_mn,
            });
        })
    }
}

/// The warp-specialized kernel's arguments.
pub struct WsKernelArgs<'a> {
    pub cfg: &'a Fp16GemmWsConfig,
    pub tiled_mma: &'a TiledMma,
    pub preferred: &'a ClusterSetup,
    pub fallback: Option<&'a ClusterSetup>,
    pub tma_atom_c: &'a CopyAtom<IoDtype>,
    pub m_c_mnl: &'a Tensor<Ident>,
    pub a_smem_layout: &'a ComposedLayout,
    pub b_smem_layout: &'a ComposedLayout,
    pub c_smem_layout_kind: LayoutEnum,
    pub epi_smem_layout_staged: &'a ComposedLayout,
    pub epi_tile: &'a Tiler,
    pub num_ctas_mn: &'a Shape,
}

/// A role's tile scheduler.
enum Sched<'p> {
    OneTile,
    Static(StaticPersistentTileScheduler),
    Clc { sched: ClcDynamicPersistentTileScheduler, pipeline: &'p PipelineClcFetchAsync, num_clc_stage: usize },
}

impl Sched<'_> {
    /// The persistent tile scheduling loop of a role: `body` runs for each of the CTA's
    /// tiles, threading `state`; then the next tile comes from the static scheduler, or
    /// from the scheduler warp's CLC response (`consumer_wait`, `get_current_work`,
    /// `consumer_release`).
    fn for_each_tile<S: Carry>(&self, work: WorkTileInfo, state: S, body: impl FnOnce(WorkTileInfo, S) -> S) -> S {
        match self {
            Sched::OneTile => body(work, state),
            Sched::Static(sched) => {
                let (_, _, state) = while_carry(
                    (work, sched.clone(), state),
                    |(w, _, _)| w.is_valid_tile(),
                    |(w, mut sched, state)| {
                        let state = body(w, state);
                        // Advance to next tile
                        sched.advance_to_next_work();
                        (sched.get_current_work(), sched, state)
                    },
                );
                state
            }
            Sched::Clc { sched, pipeline, num_clc_stage } => {
                let clc_consumer_state = make_pipeline_state(PipelineUserType::Consumer, *num_clc_stage);
                let (_, _, state) = while_carry(
                    (work, clc_consumer_state, state),
                    |(w, _, _)| w.is_valid_tile(),
                    |(w, mut clc_consumer_state, state)| {
                        let state = body(w, state);
                        pipeline.consumer_wait(&clc_consumer_state, None);
                        let next = sched.get_current_work();
                        pipeline.consumer_release(&clc_consumer_state);
                        clc_consumer_state.advance();
                        (next, clc_consumer_state, state)
                    },
                );
                state
            }
        }
    }
}

/// The tutorial's `kernel` (`fp16_gemm_2.py` to `fp16_gemm_3_1.py`) and, for a fallback
/// cluster, `fp16_gemm_4.py`'s `kernel`, whose two arms call `cluster_specific_kernel`
/// with each cluster shape's layout, TMA atoms and scheduler params.
pub fn ws_kernel(args: &WsKernelArgs) {
    let cfg = args.cfg;

    //
    // Allocate SMEM: shared by both arms of a mixed-cluster kernel (CuTe DSL's
    // `smem_merge_branch_allocs`)
    //
    let smem = SmemAllocator::new();
    let storage_layout = WsSharedStorage::layout(cfg.ab_stages, cfg.acc_stages, cfg.num_clc_stage);
    let storage =
        WsSharedStorage::from_layout(&storage_layout, &smem.allocate_struct(&storage_layout, 16, SmemPartition::User));
    let s_a = smem.allocate_tensor::<IoDtype>(args.a_smem_layout.clone(), 1024, None, SmemPartition::User);
    let s_b = smem.allocate_tensor::<IoDtype>(args.b_smem_layout.clone(), 1024, None, SmemPartition::User);
    let s_c = smem.allocate_tensor::<IoDtype>(args.epi_smem_layout_staged.clone(), 1024, None, SmemPartition::User);

    let fallback = args.fallback.map(|f| shape3(f.cluster_shape_mnk));
    mixed_cluster_arms(shape3(cfg.cluster_shape_mnk), fallback, |arm| {
        let setup = match args.fallback {
            Some(f) if !arm.is_preferred => f,
            _ => args.preferred,
        };
        cluster_specific_kernel(args, setup, &storage, &s_a, &s_b, &s_c);
    });
}

/// A cluster shape as an array.
fn shape3((x, y, z): (usize, usize, usize)) -> [usize; 3] {
    [x, y, z]
}

/// `fp16_gemm_4.py`'s `cluster_specific_kernel` (the whole kernel of `fp16_gemm_2.py` to
/// `fp16_gemm_3_1.py`) for one cluster shape.
fn cluster_specific_kernel(
    args: &WsKernelArgs,
    setup: &ClusterSetup,
    storage: &WsSharedStorage,
    s_a: &Tensor<IoDtype>,
    s_b: &Tensor<IoDtype>,
    s_c: &Tensor<IoDtype>,
) {
    let cfg = args.cfg;
    let tiled_mma = args.tiled_mma;
    let mma_tiler_mnk = cfg.mma_tiler_mnk;
    let cta_layout_vmnk = &setup.cta_layout_vmnk;
    let (tma_atom_a, tma_atom_b, tma_atom_c) = (&setup.tma_atom_a, &setup.tma_atom_b, args.tma_atom_c);
    check_cluster_shape_in_arm("the cluster layout (cta_layout_vmnk)", shape3(setup.cluster_shape_mnk));

    let warp_idx = make_warp_uniform(warp_idx());
    let tidx = thread_idx().x;
    let (bidx, bidy) = (block_idx().x, block_idx().y);

    let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
    let cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster);

    let v = cta_layout_vmnk.mode(0).size();
    let mma_tile_coord_v = bidx % v as i32;
    let is_leader_cta = mma_tile_coord_v.eq(0);
    let as_leader = |f: &dyn Fn()| if v == 1 { f() } else { if_(is_leader_cta, f) };

    let epilog_sync_bar_id = 1;
    let tmem_alloc_sync_bar_id = 2;

    // Prefetch tma descriptor
    if_(warp_idx.eq(TMA_WARP_ID), || {
        cpasync::prefetch_descriptor(tma_atom_a);
        cpasync::prefetch_descriptor(tma_atom_b);
        cpasync::prefetch_descriptor(tma_atom_c);
    });

    // As many participants as the number of threads issuing the MMA in the same row and column
    // Substract one to not count twice the same thread
    let num_mcast_participants = cta_layout_vmnk.mode(1).size() + cta_layout_vmnk.mode(2).size() - 1;

    // Mcast mask initialization
    let clustered = cta_layout_vmnk.size() > 1;
    let tma_mcast_mask_a =
        clustered.then(|| cpasync::create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk.clone(), 2));
    let tma_mcast_mask_b =
        clustered.then(|| cpasync::create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk.clone(), 1));

    // Only MMA warp and epilogue warps participate in TMEM allocation synchronization
    // TMA warp does NOT participate
    let tmem_alloc_barrier = NamedBarrier::new(tmem_alloc_sync_bar_id, 32 * (1 + EPILOGUE_WARP_IDS.len()));
    let mut tmem = TmemAllocator::new(
        tmem_alloc_barrier,
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: Some(storage.tmem_holding_buffer.ptr()),
            allocator_warp_id: EPILOGUE_WARP_IDS[0],
            is_two_cta: v > 1,
            two_cta_tmem_dealloc_mbar_ptr: (v > 1).then(|| storage.tmem_dealloc_mbar.ptr().cast::<u64>()),
            arch: Arch::Sm100a,
            ..Default::default()
        },
    );

    let num_tma_copy_bytes = size_in_bytes::<IoDtype>(&args.a_smem_layout.select(&[0, 1, 2]))
        + size_in_bytes::<IoDtype>(&args.b_smem_layout.select(&[0, 1, 2]));
    let num_tma_copy_bytes = num_tma_copy_bytes.as_static().expect("the SMEM layouts are static") * v;

    // Threads/warps participating in the mainloop pipeline
    let ab_pipeline = PipelineTmaUmma::create(
        cfg.ab_stages,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, num_mcast_participants),
        num_tma_copy_bytes,
        PipelineTmaUmmaOptions {
            barrier_storage: Some(storage.ab_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cta_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (ab_producer, ab_consumer) = ab_pipeline.make_participants();

    // Threads/warps participating in the accumulator pipeline: one arrival per epilogue
    // warp (the packaged epilogue releases from one thread per warp).
    let acc_pipeline = PipelineUmmaAsync::create(
        cfg.acc_stages,
        CooperativeGroup::new(Agent::Thread, 1),
        CooperativeGroup::new(Agent::Thread, (v * EPILOGUE_WARP_IDS.len()) as i32),
        PipelineUmmaOptions {
            barrier_storage: Some(storage.acc_mbar_ptr.data_ptr().cast::<u64>()),
            cta_layout_vmnk: Some(cta_layout_vmnk.clone()),
            defer_sync: true,
            ..Default::default()
        },
    );
    let (acc_producer, acc_consumer) = acc_pipeline.make_participants();

    // Initialize clc_pipeline (barrier) and states
    // ONLY for CLC Dynamic Scheduler
    let use_clc_dynamic_scheduler = cfg.scheduling == Scheduling::ClcDynamic;
    let clc_pipeline = use_clc_dynamic_scheduler.then(|| {
        let cluster_size = cta_layout_vmnk.size();
        // The scheduler warp, and the MMA, TMA and epilogue warps of every CTA.
        let num_clc_consumer_threads = 32 * (1 + cluster_size * (2 + EPILOGUE_WARP_IDS.len()));
        PipelineClcFetchAsync::create(
            cfg.num_clc_stage,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, num_clc_consumer_threads),
            NUM_CLC_RESPONSE_BYTES,
            PipelineClcFetchAsyncOptions {
                barrier_storage: Some(storage.clc_mbar_ptr.data_ptr().cast::<u64>()),
                cta_layout_vmnk: Some(cta_layout_vmnk.clone()),
                defer_sync: true,
                ..Default::default()
            },
        )
    });
    // Initial clc response pointer
    let clc_response_ptr = storage.clc_response.data_ptr();

    // Cluster arrive after barrier init
    pipeline::pipeline_init_arrive(Some(cta_layout_vmnk.shape().clone()), true);

    // Partition tensors for MMA and make fragments
    // (bM, bK, RestM, RestK)
    let g_a = local_tile(&setup.tma_tensor_a, (mma_tiler_mnk.0, mma_tiler_mnk.2), (.., ..));
    // (bN, bK, RestN, RestK)
    let g_b = local_tile(&setup.tma_tensor_b, (mma_tiler_mnk.1, mma_tiler_mnk.2), (.., ..));
    // (bM, bN, RestM, RestN)
    let g_c = local_tile(args.m_c_mnl, (mma_tiler_mnk.0, mma_tiler_mnk.1), (.., ..));

    let thr_mma = tiled_mma.get_slice(mma_tile_coord_v);
    // (MMA, MMA_M, MMA_K, RestM, RestK)
    let t_cg_a = thr_mma.partition_a(&g_a);
    // (MMA, MMA_N, MMA_K, RestN, RestK)
    let t_cg_b = thr_mma.partition_b(&g_b);
    // (MMA, MMA_M, MMA_N, RestM, RestN)
    let t_cg_c = thr_mma.partition_c(&g_c);

    // (MMA, MMA_M, MMA_K, STAGE)
    let t_cr_a = tiled_mma.make_fragment_a(s_a);
    // (MMA, MMA_N, MMA_K, STAGE)
    let t_cr_b = tiled_mma.make_fragment_b(s_b);

    // (MMA, MMA_M, MMA_N)
    let acc_shape = tiled_mma.partition_shape_c((mma_tiler_mnk.0, mma_tiler_mnk.1));
    // (MMA, MMA_M, MMA_N, STAGE)
    let Tree::Branch(mut modes) = acc_shape.clone() else { panic!("a tuple shape") };
    modes.push(Shape::from(cfg.acc_stages));
    let t_ct_acc_fake = tiled_mma.make_fragment_c::<f32>(&Shape::from(Tree::Branch(modes)));

    // Partition tensors for TMA; This requires the tensors partitioned for MMA
    let coord = |i: usize| match &cta_in_cluster_coord_vmnk {
        Tree::Branch(modes) => modes[i].clone(),
        leaf => leaf.clone(),
    };
    // ((atom_v, rest_v), STAGE)
    // ((atom_v, rest_v), RestM, RestK)
    let (t_as_a, t_ag_a) = cpasync::tma_partition(
        tma_atom_a,
        coord(2),
        &make_layout(cta_layout_vmnk.mode(2).size(), 1),
        &s_a.group_modes(0, 3),
        &t_cg_a.group_modes(0, 3),
    );
    // ((atom_v, rest_v), STAGE)
    // ((atom_v, rest_v), RestN, RestK)
    let (t_bs_b, t_bg_b) = cpasync::tma_partition(
        tma_atom_b,
        coord(1),
        &make_layout(cta_layout_vmnk.mode(1).size(), 1),
        &s_b.group_modes(0, 3),
        &t_cg_b.group_modes(0, 3),
    );

    // Cluster wait before starting work
    pipeline::pipeline_init_wait(Some(cta_layout_vmnk.shape().clone()));

    // Construct the scheduler
    let (sched, work_tile) = match cfg.scheduling {
        Scheduling::OneTile => (Sched::OneTile, WorkTileInfo::new([bidx, bidy, Int32::from(0)], Boolean::from(true))),
        Scheduling::StaticPersistent => {
            let tile_sched_params = cfg.static_params(args.num_ctas_mn.clone());
            let tile_sched = StaticPersistentTileScheduler::create(&tile_sched_params, block_idx(), grid_dim());
            let work_tile = tile_sched.initial_work_tile_info();
            (Sched::Static(tile_sched), work_tile)
        }
        Scheduling::ClcDynamic => {
            let tile_sched_params = cfg.clc_params(args.num_ctas_mn.clone());
            let tile_sched = ClcDynamicPersistentTileScheduler::create(
                &tile_sched_params,
                block_idx(),
                grid_dim(),
                &clc_response_ptr,
                true,
            );
            let work_tile = tile_sched.initial_work_tile_info();
            let pipeline = clc_pipeline.as_ref().expect("a CLC pipeline");
            (Sched::Clc { sched: tile_sched, pipeline, num_clc_stage: cfg.num_clc_stage }, work_tile)
        }
    };
    let mma_tile_coord_mn = |w: &WorkTileInfo| {
        let t = w.tile_idx();
        (t[0] / v as i32, t[1])
    };

    let num_k_tiles = size(&g_a, &[3]);

    // TMA warp
    if_(warp_idx.eq(TMA_WARP_ID), || {
        //
        // Persistent tile scheduling loop
        //
        let mut ab_producer = sched.for_each_tile(work_tile, ab_producer.clone_participant(), |w, ab_producer| {
            // Get tile coord from tile scheduler
            let (mma_m, mma_n) = mma_tile_coord_mn(&w);
            // Slice to per mma tile index
            // ((atom_v, rest_v), RestK)
            let t_ag_a_slice = t_ag_a.slice((.., mma_m, ..));
            // ((atom_v, rest_v), RestK)
            let t_bg_b_slice = t_bg_b.slice((.., mma_n, ..));
            for_range_carry(0, num_k_tiles.clone(), 1, ab_producer, |k_tile_idx, mut ab_producer| {
                // Wait for A/B buffers to be empty before loading into them
                let handle = ab_producer.acquire_and_advance(None);
                let bar = CopyArgs::new().tma_bar_ptr(&handle.barrier());
                let (bar_a, bar_b) = match (tma_mcast_mask_a, tma_mcast_mask_b) {
                    (Some(a), Some(b)) => (bar.clone().mcast_mask(a), bar.mcast_mask(b)),
                    _ => (bar.clone(), bar),
                };
                // Issue TMA loads
                copy_with(
                    tma_atom_a,
                    &t_ag_a_slice.slice((.., k_tile_idx)),
                    &t_as_a.slice((.., handle.index())),
                    bar_a,
                );
                copy_with(
                    tma_atom_b,
                    &t_bg_b_slice.slice((.., k_tile_idx)),
                    &t_bs_b.slice((.., handle.index())),
                    bar_b,
                );
                ab_producer
            })
        });
        // This mbarrier_wait is preventing threadblocks within a set of dependent threadblocks within the cluster
        // (dependent in the context of the TMA/MMA synchronization pattern) to exit early making
        // a late tcgen05 commit_arrive illegal
        ab_producer.tail();
    });

    // Sched warp (only for dynamic scheduler)
    if let Sched::Clc { sched: tile_sched, pipeline: clc_pipeline, num_clc_stage } = &sched {
        let is_first_cta_in_cluster = cta_rank_in_cluster.eq(0);
        if_(warp_idx.eq(SCHED_WARP_ID) & is_first_cta_in_cluster, || {
            // Persistent tile scheduling loop
            let clc_producer_state = make_pipeline_state(PipelineUserType::Producer, *num_clc_stage);
            let clc_consumer_state = make_pipeline_state(PipelineUserType::Consumer, *num_clc_stage);
            let (_, mut clc_producer_state, _, _) = while_carry(
                (work_tile, clc_producer_state, clc_consumer_state, tile_sched.clone()),
                |(w, _, _, _)| w.is_valid_tile(),
                |(_, mut clc_producer_state, mut clc_consumer_state, mut tile_sched)| {
                    // Advance to next tile
                    clc_pipeline.producer_acquire(&clc_producer_state, None);
                    let mbarrier_addr = clc_pipeline.producer_get_barrier(&clc_producer_state);
                    tile_sched.advance_to_next_work(&mbarrier_addr);
                    clc_producer_state.advance();

                    clc_pipeline.consumer_wait(&clc_consumer_state, None);
                    let work_tile = tile_sched.get_current_work();
                    clc_pipeline.consumer_release(&clc_consumer_state);
                    clc_consumer_state.advance();
                    (work_tile, clc_producer_state, clc_consumer_state, tile_sched)
                },
            );
            clc_pipeline.producer_tail(&mut clc_producer_state);
        });
    }

    // MMA warp
    if_(warp_idx.eq(MMA_WARP_ID), || {
        // Wait for TMEM allocation and retrieve pointer
        tmem.wait_for_alloc();
        let tmem_ptr = tmem.retrieve_ptr::<f32>();
        // (MMA, MMA_M, MMA_N, STAGE)
        let t_ct_acc_base = make_tensor(tmem_ptr, t_ct_acc_fake.layout().clone());

        let (_, acc_producer) = sched.for_each_tile(
            work_tile,
            (ab_consumer.clone_participant(), acc_producer.clone_participant()),
            |_, (mut ab_consumer, mut acc_producer)| {
                // The states advance on every CTA; the leader CTA of a pair waits and issues.
                let acc_empty = acc_producer.current_handle();
                acc_producer.advance(1);
                as_leader(&|| {
                    // Wait for accumulator buffer empty
                    acc_empty.get_origin().producer_acquire(&acc_empty.state(), None);
                    // Set tensor memory buffer for current tile
                    // (MMA, MMA_M, MMA_N)
                    let t_ct_acc = t_ct_acc_base.slice((.., .., .., acc_empty.index()));
                    let mut tiled_mma = tiled_mma.clone();
                    tiled_mma.set(Field::ACCUMULATE, false);
                    for_range_carry(
                        0,
                        num_k_tiles.clone(),
                        1,
                        (ab_consumer.clone_participant(), tiled_mma),
                        |_, (mut ab_consumer, mut tiled_mma)| {
                            // Wait for TMA copies to complete
                            let handle = ab_consumer.wait_and_advance(None);
                            // Execute one K-block worth of MMA instructions
                            let num_k_blocks = t_cr_a.layout().mode(2).size();
                            for k_block_idx in 0..num_k_blocks {
                                let k_block_coord = (.., .., k_block_idx, handle.index());
                                gemm(
                                    &tiled_mma,
                                    &t_ct_acc,
                                    &t_cr_a.slice(k_block_coord),
                                    &t_cr_b.slice(k_block_coord),
                                    &t_ct_acc,
                                );
                                tiled_mma.set(Field::ACCUMULATE, true);
                            }
                            // Signal that the A/B buffers have been consumed and are ready for the next load
                            handle.release();
                            (ab_consumer, tiled_mma)
                        },
                    );
                    // Signal that the accumulator is fully computed
                    acc_empty.commit();
                });
                ab_consumer.advance(IntoValue::<i32>::into_value(num_k_tiles.clone()));
                (ab_consumer, acc_producer)
            },
        );

        // Wait for accumulator buffer empty
        as_leader(&|| acc_producer.clone_participant().tail());
    });

    // Epilogue warps
    if_(warp_idx.lt(MMA_WARP_ID), || {
        // Allocate TMEM (only epilogue warp 0 actually allocates)
        let num_tmem_cols = 512;
        tmem.allocate(num_tmem_cols);

        // Wait for TMEM allocation and retrieve pointer
        tmem.wait_for_alloc();
        let tmem_ptr = tmem.retrieve_ptr::<f32>();
        // (MMA, MMA_M, MMA_N, STAGE)
        let t_ct_acc_base = make_tensor(tmem_ptr.clone(), t_ct_acc_fake.layout().clone());

        // Initialize TMA store pipeline for epilogue
        let epilogue_pipeline = PipelineTmaStore::create(
            cfg.epi_stages,
            CooperativeGroup::new(Agent::Thread, cfg.threads_in_epilogue as i32),
        );

        let mut ecfg = EpilogueConfig::new(
            cfg.cta_tile_shape_mnk(),
            args.c_smem_layout_kind,
            IoDtype::DTYPE,
            ACC_DTYPE,
            cfg.use_2cta_instrs,
        );
        ecfg.num_c_stage = cfg.epi_stages;
        ecfg.epilog_sync_bar_id = epilog_sync_bar_id;

        sched.for_each_tile(work_tile, (acc_consumer.state(), Int32::from(0)), |w, (acc_state, tiles_done)| {
            // Get tile coord from tile scheduler
            let (mma_m, mma_n) = mma_tile_coord_mn(&w);
            // Wait for the accumulator, drain it subtile by subtile through SMEM with TMA
            // stores, and release it (the packaged epilogue)
            let next = epilogue_tma_store::<f32, IoDtype, _, _>(
                &ecfg,
                tidx,
                warp_idx,
                tma_atom_c,
                &t_ct_acc_base,
                s_c,
                &t_cg_c,
                args.epi_tile,
                tiles_done,
                &|x| x,
                &[mma_m, mma_n],
                &acc_state,
                &acc_pipeline,
                &epilogue_pipeline,
            );
            (next, tiles_done + 1)
        });

        // Wait for C store complete
        if_(warp_idx.eq(EPILOGUE_WARP_IDS[0]), || epilogue_pipeline.producer_tail());

        // Dealloc the tensor memory buffer
        tmem.relinquish_alloc_permit();
        tmem.free(&tmem_ptr, 0);
    });
}

/// Simulate `cfg`'s kernel on an `m x n x k` problem launched as `launch` with the options
/// `opts`, and check C bit for bit.
pub fn run_ws_with(
    cfg: &Fp16GemmWsConfig,
    (m, n, k): (usize, usize, usize),
    launch: LaunchConfig,
    opts: &SimOptions,
) -> SimReport {
    let kernel = cfg.build();
    let a = small_ints(m * k, 1111);
    let b = small_ints(n * k, 2222);
    let c = vec![IoDtype::from_f32(-7.0); m * n];
    let mut args = vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::buffer(&c),
    ];
    let report =
        kernel.simulate_with(launch, &mut args, opts).unwrap_or_else(|e| panic!("{} {m}x{n}x{k}: {e}", cfg.name));
    let got = args[5].to_vec::<IoDtype>();
    let want = reference(&a, &b, m, n, k);
    assert!(got == want, "{} {m}x{n}x{k}: C differs from the reference", cfg.name);
    report
}

/// [`run_ws_with`] on the configuration's own launch (at most `max_active_clusters`
/// clusters for the static scheduler) under the default options.
pub fn run_ws(cfg: &Fp16GemmWsConfig, mnk: (usize, usize, usize), max_active_clusters: u32) -> SimReport {
    run_ws_with(cfg, mnk, cfg.launch(mnk.0, mnk.1, max_active_clusters), &SimOptions::default())
}

/// Integers in `[-2, 2)`, as `torch.empty(..., dtype=torch.int32).random_(-2, 2)` gives
/// them (from a fixed LCG here).
pub fn small_ints(n: usize, seed: u64) -> Vec<IoDtype> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            IoDtype::from_f32(((s >> 40) % 4) as f32 - 2.0)
        })
        .collect()
}

/// `einsum("mk,nk->mn")` in Float32, converted to Float16 (exact for these data).
pub fn reference(a: &[IoDtype], b: &[IoDtype], m: usize, n: usize, k: usize) -> Vec<IoDtype> {
    let mut out = Vec::with_capacity(m * n);
    for r in 0..m {
        for col in 0..n {
            let acc: f32 = (0..k).map(|i| a[r * k + i].to_f32() * b[col * k + i].to_f32()).sum();
            out.push(IoDtype::from_f32(acc));
        }
    }
    out
}

/// Simulate `cfg`'s kernel on an `m x n x k` problem and check C bit for bit.
pub fn run(cfg: &Fp16GemmConfig, m: usize, n: usize, k: usize) -> SimReport {
    let kernel = cfg.build();
    let a = small_ints(m * k, 1111);
    let b = small_ints(n * k, 2222);
    let c = small_ints(m * n, 3333);
    let mut args = vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::buffer(&c),
    ];
    let report = kernel.simulate(cfg.launch(m, n), &mut args).unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
    let got = args[5].to_vec::<IoDtype>();
    let want = reference(&a, &b, m, n, k);
    assert!(got == want, "{m}x{n}x{k}: C differs from the reference");
    report
}

#[allow(dead_code)]
fn main() {
    // fp16_gemm_0 on 2x2 CTAs, then fp16_gemm_1 on one CTA pair with 8 k-tiles, then the
    // warp-specialized steps.
    for (cfg, (m, n, k)) in [(FP16_GEMM_0, (256, 512, 256)), (FP16_GEMM_1, (256, 256, 512))] {
        println!("===================================================================");
        println!("Running Blackwell fp16 GEMM example {} with:", &cfg.name[cfg.name.len() - 1..]);
        println!("  mnk:       {m},{n},{k} (simulated)");
        println!("  cluster:   {:?}", cfg.cluster_shape_mnk);
        println!("===================================================================");
        let report = run(&cfg, m, n, k);
        println!("{report}");
        println!("PASS");
    }
    main_ws();
}

/// `fp16_gemm_2` to `fp16_gemm_4` on a 512x512x128 problem (four CTA pairs' tiles): the
/// static scheduler with two persistent clusters, and `fp16_gemm_4` with one preferred
/// cluster split into fallback clusters.
#[allow(dead_code)]
fn main_ws() {
    let (m, n, k) = (512, 512, 128);
    for cfg in [FP16_GEMM_2, FP16_GEMM_3, FP16_GEMM_3_1, FP16_GEMM_4] {
        println!("===================================================================");
        println!("Running Blackwell fp16 GEMM example {} with:", &cfg.name["blackwell_fp16_gemm_".len()..]);
        println!("  mnk:       {m},{n},{k} (simulated)");
        println!("  cluster:   {:?} (fallback {:?})", cfg.cluster_shape_mnk, cfg.fallback_cluster_shape_mnk);
        println!("  scheduler: {:?}", cfg.scheduling);
        println!("===================================================================");
        let opts = SimOptions { cluster_shapes: ClusterShapes::Mixed { seed: 0 }, ..Default::default() };
        let report = run_ws_with(&cfg, (m, n, k), cfg.launch(m, n, 2), &opts);
        println!("{report}");
        println!("PASS");
    }
}

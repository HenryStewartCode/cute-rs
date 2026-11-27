// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py
//   (`HopperWgmmaGemmPersistentKernel`: its configuration, `__call__`,
//   `kernel`, `_compute_stages`, `_compute_grid` and the comments),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A batched dense GEMM (`C = A * B`) for NVIDIA Hopper (SM90) with warp
//! specialisation and a persistent grid, ported from the CuTe DSL example
//! `hopper/kernel/dense_gemm/dense_gemm_persistent.py`.
//!
//! It builds on [`hopper_gemm`](super) (the dense port): the configuration,
//! the shared-memory layouts, the TMA atoms, the host data and the reference
//! are that file's. What is new:
//!
//! * **Roles.** One DMA warpgroup (its warp 0 issues the TMA loads, after
//!   `setmaxregister_decrease(40)`) and `prod(atom_layout_mnk)` MMA
//!   warpgroups (`setmaxregister_increase(232)`) that run the WGMMAs and the
//!   epilogue; the first MMA warp (warp 4) issues the TMA stores.
//! * **Persistent grid.** The grid has at most `max_active_clusters`
//!   clusters ([`PersistentTileSchedulerParams::get_grid_shape`]; on a GPU,
//!   `cuda::HardwareInfo::get_max_active_clusters`), and each CTA visits the
//!   tiles of [`StaticPersistentTileScheduler`] in turn, the DMA warp and the
//!   MMA warpgroups each with a scheduler of their own. The swizzle size and
//!   the raster order shape the visiting order.
//! * **State carried across tiles.** The mainloop pipeline's read, release
//!   and producer states keep their stage and phase from tile to tile (only
//!   their counts, the k-tile index, restart), the epilogue's staging
//!   buffer is `(tiles executed * epilogue tiles + epilogue tile) % stages`,
//!   and `PipelineTmaStore`'s bulk groups keep going until `producer_tail`.
//!
//! Differences from the example:
//!
//! * The accumulators start each tile with the first MMA not accumulating
//!   (as `hopper_gemm` does), where the example fills them with zero first.
//! * One MMA warpgroup (256 threads) increases its registers to 248, the
//!   budget its launch bound already gives, not to 232: the simulator
//!   refuses an `setmaxnreg.inc` below the current budget. Two MMA
//!   warpgroups (384 threads) use the example's 40 and 232.
//! * The mainloop pipeline's try-wait tokens (peeking) are not used.
//! * For C that is not 16 bits wide, the register-to-shared copy is as in
//!   `hopper_gemm` (a 128-bit universal atom stands in for `stmatrix`).
//! * `max_active_clusters` is a launch argument, so that the simulator can
//!   run a grid with fewer clusters than tiles.
//! * [`reset_states_per_tile`](HopperWgmmaGemmPersistentKernel::reset_states_per_tile)
//!   makes the MMA warpgroups restart their pipeline states on every tile,
//!   the mistake the diagnostics test shows.
//! * K must be at least one k-tile.
//!
//! Run with `cargo run --release --example hopper_persistent_gemm` (on the
//! simulator).

// `hopper_gemm` is included as a module: its `main` and the helpers this example does not call are unused here.
#![allow(dead_code)]

#[path = "hopper_gemm.rs"]
mod hopper_gemm;

use cute::cpasync;
use cute::pipeline::{
    Agent, CooperativeGroup, NamedBarrier, Pipeline, PipelineBarriers, PipelineState, PipelineTmaAsync,
    PipelineTmaAsyncOptions, PipelineTmaStore, PipelineUserType, make_pipeline_state, pipeline_init_arrive,
    pipeline_init_wait,
};
use cute::prelude::*;
use cute::utils::sm90;
use cute::utils::{PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentTileScheduler};
use cute::warpgroup::{self};

use hopper_gemm::SharedStorage;
pub use hopper_gemm::{HopperWgmmaGemmKernel, Majors, Problem};

/// The persistent, warp-specialised Hopper GEMM (`HopperWgmmaGemmPersistentKernel`).
/// The configuration the dense port derives is in [`base`](Self::base) (its
/// `ab_stage` is replaced by this kernel's own count: C's staging tiles
/// never alias A's here).
#[derive(Clone, Debug)]
pub struct HopperWgmmaGemmPersistentKernel {
    pub base: HopperWgmmaGemmKernel,
    pub swizzle_size: usize,
    pub raster_along_m: bool,
    pub num_dma_warp_groups: usize,
    pub num_mma_warp_groups: usize,
    pub num_warps_per_warp_group: usize,
    pub num_threads_per_warp_group: usize,
    pub threads_per_cta: usize,
    pub load_warp_id: usize,
    pub epi_store_warp_id: usize,
    pub load_register_requirement: u32,
    pub mma_register_requirement: u32,
    pub num_mma_threads: usize,
    /// Restart the MMA warpgroups' pipeline states (stage and phase, not
    /// only the count) on every tile: a mistake, for the diagnostics test.
    pub reset_states_per_tile: bool,
}

impl HopperWgmmaGemmPersistentKernel {
    /// The configuration (`__init__`).
    pub fn new(
        acc_dtype: DType,
        tile_shape_mn: (usize, usize),
        cluster_shape_mn: (usize, usize),
        swizzle_size: usize,
        raster_along_m: bool,
    ) -> Self {
        let base = HopperWgmmaGemmKernel::new(acc_dtype, tile_shape_mn, cluster_shape_mn);
        let num_dma_warp_groups = 1;
        let num_mma_warp_groups = base.mma_warp_groups;
        let num_warps_per_warp_group = 4;
        let num_threads_per_warp_group = num_warps_per_warp_group * 32;
        let threads_per_cta = (num_dma_warp_groups + num_mma_warp_groups) * num_threads_per_warp_group;
        HopperWgmmaGemmPersistentKernel {
            base,
            swizzle_size,
            raster_along_m,
            num_dma_warp_groups,
            num_mma_warp_groups,
            num_warps_per_warp_group,
            num_threads_per_warp_group,
            threads_per_cta,
            load_warp_id: 0,
            epi_store_warp_id: num_dma_warp_groups * num_warps_per_warp_group,
            load_register_requirement: 40,
            // 232, but never fewer than the budget the launch bound gives
            // (248 for the 256 threads of one MMA warpgroup, where 232 would
            // be a decrease that `setmaxnreg.inc` does not take).
            mma_register_requirement: 232.max(cute::sim::setmaxnreg_initial_budget(threads_per_cta)),
            num_mma_threads: num_mma_warp_groups * num_threads_per_warp_group,
            reset_states_per_tile: false,
        }
    }

    /// The configuration for operands of these types and majors (the type
    /// checks of `__call__`, then `_setup_attributes`).
    pub fn setup(mut self, a_dtype: DType, b_dtype: DType, c_dtype: DType, majors: Majors) -> Self {
        self.base = self.base.setup(a_dtype, b_dtype, c_dtype, majors);
        let (ab, epi) = Self::compute_stages(
            self.base.tile_shape_mnk,
            a_dtype,
            b_dtype,
            self.base.epi_tile,
            c_dtype,
            self.base.smem_capacity,
            self.base.occupancy,
        );
        (self.base.ab_stage, self.base.epi_stage) = (ab, epi);
        self
    }

    /// The A/B and epilogue stage counts (`_compute_stages`): the epilogue
    /// has its own shared memory, `epi_stage` tiles of C.
    pub fn compute_stages(
        tile_shape_mnk: (usize, usize, usize),
        a_dtype: DType,
        b_dtype: DType,
        epi_tile: (usize, usize),
        c_dtype: DType,
        smem_capacity: usize,
        occupancy: usize,
    ) -> (usize, usize) {
        let (m, n, k) = tile_shape_mnk;
        let ab_bytes_per_stage = m * k * a_dtype.bits() / 8 + n * k * b_dtype.bits() / 8;
        let c_bytes_per_stage = epi_tile.0 * epi_tile.1 * c_dtype.bits() / 8;
        let epi_stage = 4;
        let epi_bytes = c_bytes_per_stage * epi_stage;
        let mbar_helpers_bytes = 1024;
        let ab_stage = (smem_capacity / occupancy - (mbar_helpers_bytes + epi_bytes)) / ab_bytes_per_stage;
        (ab_stage, epi_stage)
    }

    /// The kernel's name for its configuration.
    pub fn name(&self) -> String {
        let b = &self.base;
        let (tm, tn, _) = b.tile_shape_mnk;
        let (cm, cn) = b.cluster_shape_mn;
        let majors = hopper_gemm::gemm_majors(b);
        format!(
            "hopper_persistent_gemm_{}_{}_{}_{}_{}{}{}_t{tm}x{tn}_c{cm}x{cn}_s{}{}",
            b.a_dtype,
            b.b_dtype,
            b.c_dtype,
            b.acc_dtype,
            majors.a,
            majors.b,
            majors.c,
            self.swizzle_size,
            if self.raster_along_m { "m" } else { "n" },
        )
    }

    fn sched_options(&self) -> PersistentTileSchedulerOptions {
        PersistentTileSchedulerOptions { swizzle_size: self.swizzle_size, raster_along_m: self.raster_along_m }
    }

    fn cluster_shape_mnk(&self) -> [usize; 3] {
        [self.base.cluster_shape_mn.0, self.base.cluster_shape_mn.1, 1]
    }

    /// The schedule of an `m x n x l` problem (`_compute_grid`'s
    /// `PersistentTileSchedulerParams`).
    pub fn tile_sched_params(&self, m: usize, n: usize, l: usize) -> PersistentTileSchedulerParams {
        let (bm, bn, _) = self.base.tile_shape_mnk;
        PersistentTileSchedulerParams::new(
            (m.div_ceil(bm), n.div_ceil(bn), l),
            self.cluster_shape_mnk(),
            self.sched_options(),
        )
    }

    /// The launch for an `m x n x l` problem with at most
    /// `max_active_clusters` clusters resident (`_compute_grid`'s grid and
    /// the `launch` of `__call__`).
    pub fn launch(&self, m: usize, n: usize, l: usize, max_active_clusters: u32) -> LaunchConfig {
        let grid = self.tile_sched_params(m, n, l).get_grid_shape(max_active_clusters);
        let (cm, cn) = self.base.cluster_shape_mn;
        LaunchConfig::new(grid, [self.threads_per_cta as u32, 1, 1]).with_cluster([cm as u32, cn as u32, 1])
    }

    /// `HardwareInfo().get_max_active_clusters(cluster size)` on device 0.
    #[cfg(feature = "cuda")]
    pub fn max_active_clusters_on_device(&self) -> Result<u32, cute::cuda::CudaError> {
        let (cm, cn) = self.base.cluster_shape_mn;
        cute::cuda::HardwareInfo::new(0)?.get_max_active_clusters((cm * cn) as u32, None)
    }

    /// The kernel: its parameters are `m`, `n`, `k`, `l`, the tensors `a`,
    /// `b` and `c` (as in the dense port), then the schedule's
    /// ([`PersistentTileSchedulerParams::param`] `sched`).
    pub fn build(&self) -> Kernel {
        use DType as D;
        type E4 = F8E4M3FN;
        type E5 = F8E5M2;
        let b = &self.base;
        match (b.a_dtype, b.b_dtype, b.c_dtype, b.acc_dtype) {
            (D::F16, D::F16, D::F16, D::F32) => self.build_typed::<F16, F16, F16, f32>(),
            (D::F16, D::F16, D::F16, D::F16) => self.build_typed::<F16, F16, F16, F16>(),
            (D::F16, D::F16, D::F32, D::F32) => self.build_typed::<F16, F16, f32, f32>(),
            (D::BF16, D::BF16, D::BF16, D::F32) => self.build_typed::<BF16, BF16, BF16, f32>(),
            (D::F8E4M3FN, D::F8E4M3FN, D::F16, D::F32) => self.build_typed::<E4, E4, F16, f32>(),
            (D::F8E4M3FN, D::F8E5M2, D::F32, D::F32) => self.build_typed::<E4, E5, f32, f32>(),
            (D::F8E5M2, D::F8E4M3FN, D::F16, D::F16) => self.build_typed::<E5, E4, F16, F16>(),
            (D::F8E5M2, D::F8E5M2, D::F8E5M2, D::F32) => self.build_typed::<E5, E5, E5, f32>(),
            (D::I8, D::I8, D::I32, D::I32) => self.build_typed::<i8, i8, i32, i32>(),
            (D::U8, D::I8, D::F16, D::I32) => self.build_typed::<u8, i8, F16, i32>(),
            (a, b, c, acc) => panic!(
                "hopper_persistent_gemm: A {a}, B {b}, C {c}, Acc {acc} is not among the combinations this example \
                 instantiates"
            ),
        }
    }

    fn build_typed<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(&self) -> Kernel {
        let gemm = self.clone();
        Kernel::builder(self.name()).block_dim([self.threads_per_cta as u32, 1, 1]).target(Arch::Sm90a).build(
            move || {
                let g = &gemm.base;
                let (a_major_k, b_major_k, c_major_n) =
                    (g.a_layout.is_k_major_a(), g.b_layout.is_k_major_b(), g.c_layout.is_n_major_c());
                // The contiguous extents are whole 16-byte vectors, which TMA needs.
                let vec = |d: DType| 128 / d.bits();
                let mut align = (1, 1, 1);
                if a_major_k {
                    align.2 = align.2.max(vec(A::DTYPE))
                } else {
                    align.0 = align.0.max(vec(A::DTYPE))
                }
                if b_major_k {
                    align.2 = align.2.max(vec(B::DTYPE))
                } else {
                    align.1 = align.1.max(vec(B::DTYPE))
                }
                if c_major_n {
                    align.1 = align.1.max(vec(C::DTYPE))
                } else {
                    align.0 = align.0.max(vec(C::DTYPE))
                }
                let m = param_extent("m", align.0);
                let n = param_extent("n", align.1);
                let k = param_extent("k", align.2);
                let l = param_extent("l", 1);
                let (mi, ni, ki) = (IndexExpr::from(m), IndexExpr::from(n), IndexExpr::from(k));
                let tensor_layout = |s0: &IndexExpr, s1: &IndexExpr, mode0_major: bool| {
                    let batch = s0.clone() * s1.clone();
                    let shape = (s0.clone(), s1.clone(), IndexExpr::from(l));
                    if mode0_major {
                        Layout::new(shape, (IndexExpr::from(1), s0.clone(), batch))
                    } else {
                        Layout::new(shape, (s1.clone(), IndexExpr::from(1), batch))
                    }
                };
                let a = param_tensor_aligned::<A>("a", tensor_layout(&mi, &ki, !a_major_k), 16);
                let b = param_tensor_aligned::<B>("b", tensor_layout(&ni, &ki, !b_major_k), 16);
                let c = param_tensor_aligned::<C>("c", tensor_layout(&mi, &ni, !c_major_n), 16);
                let sched =
                    PersistentTileSchedulerParams::param("sched", gemm.cluster_shape_mnk(), gemm.sched_options());
                gemm.call::<A, B, C, Acc>(&a, &b, &c, &sched);
            },
        )
    }

    /// `__call__`: the TMA atoms and tensors, the shared-memory layouts,
    /// then the kernel.
    fn call<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(
        &self,
        a: &Tensor<A>,
        b: &Tensor<B>,
        c: &Tensor<C>,
        sched: &PersistentTileSchedulerParams,
    ) {
        let g = &self.base;
        let (a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged) =
            HopperWgmmaGemmKernel::make_smem_layouts(
                g.tile_shape_mnk,
                g.epi_tile,
                g.a_dtype,
                g.a_layout,
                g.b_dtype,
                g.b_layout,
                g.ab_stage,
                g.c_dtype,
                g.c_layout,
                g.epi_stage,
            );
        let (tma_atom_a, tma_tensor_a) = HopperWgmmaGemmKernel::make_tma_atoms_and_tensors(
            a,
            &a_smem_layout_staged,
            (g.tile_shape_mnk.0, g.tile_shape_mnk.2),
            g.cluster_shape_mn.1,
        );
        let (tma_atom_b, tma_tensor_b) = HopperWgmmaGemmKernel::make_tma_atoms_and_tensors(
            b,
            &b_smem_layout_staged,
            (g.tile_shape_mnk.1, g.tile_shape_mnk.2),
            g.cluster_shape_mn.0,
        );
        let (tma_atom_c, tma_tensor_c) =
            HopperWgmmaGemmKernel::make_tma_store_atoms_and_tensors(c, &epi_smem_layout_staged, g.epi_tile);
        self.kernel::<A, B, C, Acc>(
            &tma_atom_a,
            &tma_tensor_a,
            &tma_atom_b,
            &tma_tensor_b,
            &tma_atom_c,
            &tma_tensor_c,
            g.tiled_mma(),
            &g.cta_layout_mnk(),
            &a_smem_layout_staged,
            &b_smem_layout_staged,
            &epi_smem_layout_staged,
            sched,
        );
    }

    /// GPU device kernel performing the batched GEMM computation (`kernel`).
    fn kernel<A: Numeric, B: Numeric, C: Numeric, Acc: Numeric>(
        &self,
        tma_atom_a: &CopyAtom<A>,
        m_a_mkl: &Tensor<Ident>,
        tma_atom_b: &CopyAtom<B>,
        m_b_nkl: &Tensor<Ident>,
        tma_atom_c: &CopyAtom<C>,
        m_c_mnl: &Tensor<Ident>,
        tiled_mma: TiledMma,
        cta_layout_mnk: &Layout,
        a_smem_layout_staged: &ComposedLayout,
        b_smem_layout_staged: &ComposedLayout,
        epi_smem_layout_staged: &ComposedLayout,
        tile_sched_params: &PersistentTileSchedulerParams,
    ) {
        let g = &self.base;
        let tidx = thread_idx().x;
        let warp_idx = make_warp_uniform(warp_idx());

        // Prefetch Tma desc
        if_(warp_idx.eq(0), || {
            cpasync::prefetch_descriptor(tma_atom_a);
            cpasync::prefetch_descriptor(tma_atom_b);
            cpasync::prefetch_descriptor(tma_atom_c);
        });

        let cta_rank_in_cluster = make_warp_uniform(block_idx_in_cluster());
        let cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster);

        // Get mcast mask (a multicast-less atom takes none)
        let a_mcast_mask = make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk.clone(), 1);
        let b_mcast_mask = make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk.clone(), 0);
        let a_mcast_mask = g.is_a_mcast.then_some(a_mcast_mask);
        let b_mcast_mask = g.is_b_mcast.then_some(b_mcast_mask);
        let a_smem_layout = a_smem_layout_staged.slice((.., .., 0));
        let b_smem_layout = b_smem_layout_staged.slice((.., .., 0));
        let tma_copy_bytes = static_usize(size_in_bytes::<A>(&a_smem_layout) + size_in_bytes::<B>(&b_smem_layout));

        // Alloc and init AB full/empty mbarriers (pipeline)
        let storage = SharedStorage::<A, B, C>::allocate_aligned(
            g.buffer_align_bytes,
            g.ab_stage,
            smem_extent(a_smem_layout_staged),
            smem_extent(b_smem_layout_staged),
            smem_extent(epi_smem_layout_staged),
        );

        // Each warp contributes to the arrive count with the number of mcast size
        let mcast_size = g.num_mcast_ctas_a + g.num_mcast_ctas_b - 1;
        let consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group;
        let mainloop_pipeline = PipelineTmaAsync::create(
            g.ab_stage,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, consumer_arrive_cnt),
            tma_copy_bytes,
            PipelineTmaAsyncOptions {
                barrier_storage: Some(storage.mainloop_pipeline_array_ptr.data_ptr().cast::<u64>()),
                cta_layout_vmnk: Some(Layout::col_major((1, g.cluster_shape_mn.0, g.cluster_shape_mn.1, 1))),
                tidx: Some(tidx),
                defer_sync: true,
                ..Default::default()
            },
        );

        // Cluster arrive after barrier init
        let cluster_shape = Layout::col_major((g.cluster_shape_mn.0, g.cluster_shape_mn.1)).shape().clone();
        pipeline_init_arrive(Some(cluster_shape.clone()), true);

        // Generate smem tensor A/B/C
        let s_a = storage.s_a.get_tensor_composed(a_smem_layout_staged.clone());
        let s_b = storage.s_b.get_tensor_composed(b_smem_layout_staged.clone());
        let s_c = storage.s_c.get_tensor_composed(epi_smem_layout_staged.clone());

        // Local_tile partition global tensors: (bM, bK, RestM, RestK, RestL) and so on
        let (bm, bn, bk) = g.tile_shape_mnk;
        let g_a_mkl = local_tile(m_a_mkl, (bm, bk), (.., .., ..));
        let g_b_nkl = local_tile(m_b_nkl, (bn, bk), (.., .., ..));
        let g_c_mnl = local_tile(m_c_mnl, (bm, bn), (.., .., ..));

        // TMA load A partition_S/D
        let a_cta_layout = make_layout(g.cluster_shape_mn.1, 1);
        let a_cta_crd = cluster_coord_mnk.mode(1).clone();
        let (t_as_a, t_ag_a) = cpasync::tma_partition(
            tma_atom_a,
            a_cta_crd,
            &a_cta_layout,
            &s_a.group_modes(0, 2),
            &g_a_mkl.group_modes(0, 2),
        );
        // TMA load B partition_S/D
        let b_cta_layout = make_layout(g.cluster_shape_mn.0, 1);
        let b_cta_crd = cluster_coord_mnk.mode(0).clone();
        let (t_bs_b, t_bg_b) = cpasync::tma_partition(
            tma_atom_b,
            b_cta_crd,
            &b_cta_layout,
            &s_b.group_modes(0, 2),
            &g_b_nkl.group_modes(0, 2),
        );

        let k_tile_cnt = Int32::from(size(&g_a_mkl, &[3]));

        // Cluster wait for barrier init
        pipeline_init_wait(Some(cluster_shape));

        let warp_group_idx = make_warp_uniform(tidx / self.num_threads_per_warp_group as i32);
        let is_dma_warp_group = warp_group_idx.lt(self.num_dma_warp_groups as i32);
        if_(is_dma_warp_group, || setmaxregister_decrease(self.load_register_requirement));

        // /////////////////////////////////////////////////////////////////////
        //  DMA warp: the TMA loads, a tile at a time
        // /////////////////////////////////////////////////////////////////////
        if_(warp_idx.eq(self.load_warp_id as i32), || {
            let tile_sched = StaticPersistentTileScheduler::create(tile_sched_params, block_idx(), grid_dim());
            let work_tile = tile_sched.initial_work_tile_info();
            let producer_state = make_pipeline_state(PipelineUserType::Producer, g.ab_stage);
            let (_, _, mut producer_state) = while_carry(
                (work_tile, tile_sched, producer_state),
                |(work_tile, _, _)| work_tile.is_valid_tile(),
                |(work_tile, mut tile_sched, mut producer_state)| {
                    let [tile_m, tile_n, tile_l] = work_tile.tile_idx();
                    let t_ag_a_mkl = t_ag_a.slice((.., tile_m, .., tile_l));
                    let t_bg_b_nkl = t_bg_b.slice((.., tile_n, .., tile_l));
                    producer_state.reset_count();
                    let producer_state = for_range_carry(0, k_tile_cnt, 1, producer_state, |_k_tile, mut state| {
                        // Wait for the stage to be empty, and arm its full barrier
                        mainloop_pipeline.producer_acquire(&state, None);
                        // Slice to global/shared memref to current k_tile
                        let t_ag_a_k = t_ag_a_mkl.slice((.., state.count()));
                        let t_as_a_pipe = t_as_a.slice((.., state.index()));
                        let t_bg_b_k = t_bg_b_nkl.slice((.., state.count()));
                        let t_bs_b_pipe = t_bs_b.slice((.., state.index()));
                        // TMA load A/B
                        let bar = mainloop_pipeline.producer_get_barrier(&state);
                        let args = |mask: Option<Int16>| {
                            let args = CopyArgs::new().tma_bar_ptr(&bar);
                            match mask {
                                Some(mask) => args.mcast_mask(mask),
                                None => args,
                            }
                        };
                        copy_with(tma_atom_a, &t_ag_a_k, &t_as_a_pipe, args(a_mcast_mask));
                        copy_with(tma_atom_b, &t_bg_b_k, &t_bs_b_pipe, args(b_mcast_mask));
                        // Mainloop pipeline's producer commit is a NOP
                        mainloop_pipeline.producer_commit(&state);
                        state.advance();
                        state
                    });
                    tile_sched.advance_to_next_work();
                    (tile_sched.get_current_work(), tile_sched, producer_state)
                },
            );
            mainloop_pipeline.producer_tail(&mut producer_state);
        });

        // /////////////////////////////////////////////////////////////////////
        //  MMA warpgroups: the WGMMAs and the epilogue, a tile at a time
        // /////////////////////////////////////////////////////////////////////
        if_(!is_dma_warp_group, || {
            setmaxregister_increase(self.mma_register_requirement);
            let tile_sched = StaticPersistentTileScheduler::create(tile_sched_params, block_idx(), grid_dim());
            let work_tile = tile_sched.initial_work_tile_info();
            let read_state = make_pipeline_state(PipelineUserType::Consumer, g.ab_stage);
            let release_state = make_pipeline_state(PipelineUserType::Consumer, g.ab_stage);

            // Partition shared tensor for the TiledMMA's A/B/C
            let mma_warp_group_thread_layout = make_layout(self.num_mma_warp_groups, self.num_threads_per_warp_group);
            let mma_warp_group = warp_group_idx - self.num_dma_warp_groups as i32;
            let thr_mma = tiled_mma.get_slice(crd2idx(mma_warp_group, &mma_warp_group_thread_layout));
            let t_cs_a = thr_mma.partition_a(&s_a);
            let t_cs_b = thr_mma.partition_b(&s_b);
            let t_cr_a = tiled_mma.make_fragment_a(&t_cs_a);
            let t_cr_b = tiled_mma.make_fragment_b(&t_cs_b);
            let num_k_blocks = static_usize(size(&t_cr_a, &[2]));
            // One CTA's C tile, for the accumulators' shape
            let g_c_tile0 = local_tile(m_c_mnl, (bm, bn), (Int32::from(0), Int32::from(0), Int32::from(0)));
            let acc_shape = thr_mma.partition_c(&g_c_tile0).shape().clone();
            let accumulators = tiled_mma.make_fragment_c::<Acc>(&acc_shape);

            // Partition for epilogue (`hopper_gemm`'s, once for all tiles)
            let copy_atom_r2s = sm90::get_smem_store_op::<C>(g.c_layout, g.acc_dtype);
            let c16 = C::DTYPE.bits() == 16;
            let stmatrix = CopyOp::StMatrix { num: 4, trans: g.c_layout.is_m_major_c() };
            let copy_atom_c = make_copy_atom::<C>(if c16 { stmatrix } else { CopyOp::Universal { bits: 128 } });
            let tiled_copy_c_atom = make_tiled_copy_c_atom(&copy_atom_c, &tiled_mma);
            let tiled_copy_r2s = make_tiled_copy_s(&copy_atom_r2s, &tiled_copy_c_atom);
            // The destination, with the staging buffer as its last mode
            let thr_copy_r2s =
                tiled_copy_r2s.get_slice(tidx - (self.num_dma_warp_groups * self.num_threads_per_warp_group) as i32);
            let t_rs_sd = thr_copy_r2s.partition_d(&s_c);
            let rd_shape = thr_copy_r2s.partition_s(&s_c).shape().clone();
            let t_rs_rd_shape: Shape = Tree::Branch((0..3).map(|i| rd_shape.mode(i).clone()).collect());
            let size_t_rs_rd = make_fragment::<Acc>(t_rs_rd_shape.clone()).size();
            let epi_buffers = static_usize(size(&t_rs_sd, &[3]));
            let sepi_for_tma_partition = s_c.group_modes(0, 2);

            let k_pipe_mmas: usize = 1;
            let epilog_sync_barrier = NamedBarrier::new(1, self.num_mma_threads);
            // Initialize tma store pipeline
            let c_pipeline =
                PipelineTmaStore::create(g.epi_stage, CooperativeGroup::new(Agent::Thread, self.num_mma_threads));

            // The k-blocks of one pipeline stage.
            let mmas = |tiled_mma: &mut TiledMma, accumulators: &Tensor<Acc>, state: &PipelineState, prologue: bool| {
                for k_block_idx in 0..num_k_blocks {
                    let k_block_coord = (.., .., k_block_idx, state.index());
                    let t_cr_a_1phase = t_cr_a.slice(k_block_coord);
                    let t_cr_b_1phase = t_cr_b.slice(k_block_coord);
                    gemm(tiled_mma, accumulators, &t_cr_a_1phase, &t_cr_b_1phase, accumulators);
                    if prologue {
                        tiled_mma.set(MmaField::ACCUMULATE, true);
                    }
                }
            };

            while_carry(
                (work_tile, tile_sched, read_state, release_state, accumulators),
                |(work_tile, _, _, _, _)| work_tile.is_valid_tile(),
                |(work_tile, mut tile_sched, read_state, release_state, accumulators)| {
                    let [tile_m, tile_n, tile_l] = work_tile.tile_idx();
                    let g_c_tile = g_c_mnl.slice((.., .., tile_m, tile_n, tile_l));
                    // The states keep their stage and phase from tile to tile;
                    // only the count (the k-tile index) restarts.
                    let (mut read_state, mut release_state) = if self.reset_states_per_tile {
                        (
                            make_pipeline_state(PipelineUserType::Consumer, g.ab_stage),
                            make_pipeline_state(PipelineUserType::Consumer, g.ab_stage),
                        )
                    } else {
                        (read_state, release_state)
                    };
                    read_state.reset_count();
                    release_state.reset_count();

                    // MAINLOOP: the prologue's MMAs
                    let mut tiled_mma = tiled_mma.clone();
                    tiled_mma.set(MmaField::ACCUMULATE, false);
                    for _k_tile in 0..k_pipe_mmas {
                        // Wait for TMA copies to complete
                        mainloop_pipeline.consumer_wait(&read_state, None);
                        warpgroup::fence();
                        mmas(&mut tiled_mma, &accumulators, &read_state, true);
                        warpgroup::commit_group();
                        read_state.advance();
                    }
                    let (accumulators, _, read_state, mut release_state) = for_range_carry_with(
                        k_pipe_mmas as i32,
                        k_tile_cnt,
                        1,
                        LoopOptions::unroll(1),
                        (accumulators, tiled_mma, read_state, release_state),
                        |_k_tile, (accumulators, mut tiled_mma, mut read_state, mut release_state)| {
                            // Wait for TMA copies to complete
                            mainloop_pipeline.consumer_wait(&read_state, None);
                            // WGMMA
                            warpgroup::fence();
                            mmas(&mut tiled_mma, &accumulators, &read_state, false);
                            warpgroup::commit_group();
                            // Wait on the wgmma barrier for the earlier k_pipe_mmas MMAs to complete
                            warpgroup::wait_group(k_pipe_mmas as u32);
                            mainloop_pipeline.consumer_release(&release_state);
                            read_state.advance();
                            release_state.advance();
                            (accumulators, tiled_mma, read_state, release_state)
                        },
                    );
                    warpgroup::wait_group(0);
                    for _k_tile in 0..k_pipe_mmas {
                        mainloop_pipeline.consumer_release(&release_state);
                        release_state.advance();
                    }

                    // EPILOGUE: registers -> shared memory -> global memory
                    let t_rs_racc = tiled_copy_r2s.retile(&accumulators);
                    let t_cg_c_for_tma_partition = zipped_divide(&g_c_tile, g.epi_tile);
                    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                    let (bsg_sd, bsg_gd) = cpasync::tma_partition(
                        tma_atom_c,
                        0,
                        &make_layout(1, 1),
                        &sepi_for_tma_partition,
                        &t_cg_c_for_tma_partition,
                    );
                    let epi_tile_shape = t_cg_c_for_tma_partition.shape().mode(1).clone();
                    let epi_tile_num = static_usize(size(&t_cg_c_for_tma_partition, &[1]));
                    let epi_tile_n = epi_tile_shape.mode(1).size();
                    let epi_tile_layout = make_layout(epi_tile_shape, (epi_tile_n, 1));
                    let num_prev_epi_tiles = tile_sched.num_tiles_executed() * epi_tile_num as i32;
                    for epi_idx in 0..epi_tile_num {
                        // Copy from accumulators to D registers
                        let t_rs_rd = make_fragment::<Acc>(t_rs_rd_shape.clone());
                        for epi_v in 0..size_t_rs_rd {
                            t_rs_rd.store(epi_v, t_rs_racc.load(epi_idx * size_t_rs_rd + epi_v));
                        }
                        // Type conversion
                        let t_rs_rd_out = make_fragment::<C>(t_rs_rd_shape.clone());
                        let acc_vec = t_rs_rd.load_ssa();
                        t_rs_rd_out.store_ssa(&acc_vec.to::<C>());

                        // Copy from D registers to shared memory
                        let epi_buffer = (num_prev_epi_tiles + epi_idx as i32) % epi_buffers as i32;
                        copy(&tiled_copy_r2s, &t_rs_rd_out, &t_rs_sd.slice((.., .., .., epi_buffer)));
                        fence_proxy_with(
                            FenceProxyKind::AsyncShared,
                            FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
                        );
                        epilog_sync_barrier.arrive_and_wait();

                        let gmem_coord = epi_tile_layout.get_hier_coord(epi_idx);
                        // The store warp issues the TMA store of this tile
                        if_(warp_idx.eq(self.epi_store_warp_id as i32), || {
                            copy(tma_atom_c, &bsg_sd.slice((.., epi_buffer)), &bsg_gd.slice((.., gmem_coord.clone())));
                            c_pipeline.producer_commit();
                            c_pipeline.producer_acquire();
                        });
                        epilog_sync_barrier.arrive_and_wait();
                    }
                    tile_sched.advance_to_next_work();
                    (tile_sched.get_current_work(), tile_sched, read_state, release_state, accumulators)
                },
            );
            c_pipeline.producer_tail();
        });
    }

    /// The tensors' kernel arguments for `problem`: `m`, `n`, `k`, `l`, A, B
    /// and C (the dense port's), then the schedule's.
    pub fn inputs(&self, problem: &Problem) -> Vec<KernelArg> {
        let (m, n, _, l) = problem.mnkl;
        let mut args = hopper_gemm::inputs(&self.base, problem);
        args.extend(self.tile_sched_params(m, n, l).kernel_args());
        args
    }

    /// Simulate `kernel` (built from `self`) on `problem` with at most
    /// `max_active_clusters` clusters, under `opts`, and check C against the
    /// plain Rust reference.
    pub fn run_with(
        &self,
        kernel: &Kernel,
        problem: &Problem,
        max_active_clusters: u32,
        opts: &SimOptions,
    ) -> SimReport {
        let (m, n, k, l) = problem.mnkl;
        let majors = hopper_gemm::gemm_majors(&self.base);
        let b = &self.base;
        assert!(
            HopperWgmmaGemmKernel::is_valid_dtypes(b.a_dtype, b.b_dtype, b.acc_dtype, b.c_dtype, majors.a, majors.b),
            "unsupported combination of types and majors"
        );
        assert!(
            HopperWgmmaGemmKernel::is_valid_tensor_alignment(m, n, k, l, b.a_dtype, b.c_dtype, majors),
            "the contiguous dimension of A/B/C tensors is not 16 bytes aligned"
        );
        let mut args = self.inputs(problem);
        let report = kernel
            .simulate_with(self.launch(m, n, l, max_active_clusters), &mut args, opts)
            .unwrap_or_else(|e| panic!("{} {:?}: {e}", kernel.name(), problem.mnkl));
        hopper_gemm::check(b, problem, &args);
        report
    }

    /// [`run_with`](Self::run_with) with the default simulator options.
    pub fn run(&self, kernel: &Kernel, problem: &Problem, max_active_clusters: u32) -> SimReport {
        self.run_with(kernel, problem, max_active_clusters, &SimOptions::default())
    }
}

/// The elements of shared memory a staged layout spans: its cosize in whole
/// swizzle periods (what `make_smem_tensor` allocates).
fn smem_extent(layout: &ComposedLayout) -> usize {
    layout.cosize().next_multiple_of(layout.swizzle().map_or(1, |sw| sw.period()))
}

/// A size known when the kernel is built.
fn static_usize(e: IndexExpr) -> usize {
    e.as_static().expect("a static size")
}

fn main() {
    println!("Running Hopper Persistent Dense GEMM (in the simulator) with:");
    let (m, n, k, l) = (384, 256, 128, 1);
    let tile_shape_mn = (128, 128);
    let cluster_shape_mn = (2, 1);
    println!("mnkl: ({m}, {n}, {k}, {l})");
    println!("A dtype: Float16, B dtype: Float16, C dtype: Float16, Acc dtype: Float32");
    println!("Tile Shape: {tile_shape_mn:?}, Cluster Shape: {cluster_shape_mn:?}, swizzle 2, raster along m");
    let gemm = HopperWgmmaGemmPersistentKernel::new(DType::F32, tile_shape_mn, cluster_shape_mn, 2, true).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        Majors::KKN,
    );
    println!(
        "tile {:?}, {} A/B stages, epilogue tile {:?} x {} stages, {} threads",
        gemm.base.tile_shape_mnk, gemm.base.ab_stage, gemm.base.epi_tile, gemm.base.epi_stage, gemm.threads_per_cta
    );
    let kernel = gemm.build();
    // A GPU asks the driver (`max_active_clusters_on_device`); two clusters
    // here, so that each runs several of the 6 tiles.
    let report = gemm.run(&kernel, &Problem::new(m, n, k, l), 2);
    println!("{}: correct\n{report}\n", kernel.name());

    match Toolchain::discover() {
        Some(tc) => match kernel.compile_ptx(&tc, Arch::Sm90a) {
            Ok(ptx) => {
                let count = |pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
                println!(
                    "PTX (sm_90a): {} lines; {} setmaxnreg, {} wgmma.mma_async, {} multicast TMA loads, {} TMA stores",
                    ptx.lines().count(),
                    count("setmaxnreg"),
                    count("wgmma.mma_async"),
                    count(".multicast::cluster"),
                    count("cp.async.bulk.tensor.3d.global.shared"),
                );
            }
            Err(e) => eprintln!("PTX compilation failed: {e}"),
        },
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

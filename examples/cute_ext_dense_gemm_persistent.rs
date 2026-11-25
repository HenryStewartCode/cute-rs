// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/dense_gemm_persistent.py
//   (the class `PersistentDenseGemmKernel`: its configuration and checks, `__call__`'s
//   scheduler params and grid, and `kernel`'s warp roles, buffers, pipelines and loops,
//   with their names; and the reference check of `compare`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A persistent SM100 dense GEMM in the fragment-free style, ported from the
//! CuTe DSL example `cute_ext/blackwell/dense_gemm/dense_gemm_persistent.py`
//! (roadmap part m11-cute-ext-gemm).
//!
//! `D = A B^T`: A `(M, K)` and B `(N, K)` K-major Float16, the accumulator
//! Float32 in tensor memory, D `(M, N)` row-major Float32. 192 threads per
//! CTA, warp-specialised:
//!
//! - warp 5 (TMA): per tile of the static persistent tile scheduler, each
//!   k-tile's A and B tiles with `experimental::tma_load` into a stage of
//!   the `TMAToUMMAPipeline` (multicast across the cluster when the cluster
//!   is larger than the MMA's CTAs, 2SM loads for a 2SM MMA);
//! - warp 4 (MMA, on the leader CTA of a pair): per tile, an accumulator
//!   stage of the `UMMAtoAsyncPipeline` and `experimental::mainloop_mma`;
//! - warps 0-3 (epilogue): per tile, `experimental::epilogue_tma_store` of
//!   the accumulator stage through a `TMAStorePipeline` (warp 0 issues the
//!   stores), then the stage's release (`SM100_COPY_T2R`).
//!
//! A and B are `experimental::allocate`d in shared memory with
//! `make_smem_layout_a/b`, the accumulator in tensor memory with
//! `experimental::make_tmem_layout_acc` (pooled, 2-CTA for a 2SM MMA).
//!
//! Differences from the Python example:
//! - CuTe DSL's compiler clones the kernel body for a fallback cluster
//!   shape; here the body is built per shape with
//!   `cute::utils::mixed_cluster_arms` (the buffers allocated before it), and
//!   the cluster layout comes from the arm's static shape, not from
//!   `block_in_cluster_dim` at run time;
//! - the TMA warp and the MMA warp drain their pipelines before exiting
//!   (`producer_tail`), which the Python example leaves to its compiler;
//! - `vmnk_layout` and `multicast_mode` are given only to multicast loads
//!   (cute-rs refuses them on a plain load);
//! - [`PersistentDenseGemmConfig::explicit_cta_v_map`] (a cute-rs option)
//!   passes `get_cta_v_map_ab`'s maps to the loads;
//! - M, N and K are launch values: N a multiple of 4 and K of 8 (TMA's
//!   16-byte strides); ragged M, N and K are fine (TMA fills and drops);
//! - the reference check compares bit for bit on small-integer data and
//!   within a bound on random data, against a plain Rust GEMM.
//!
//! No Blackwell GPU is available to cute-rs's tests, so the kernel is
//! verified by the simulator and by `ptxas -arch=sm_100a`.
//!
//! Run with `cargo run --example cute_ext_dense_gemm_persistent`.

use cute::experimental::{
    self, OperationType, TMAStorePipeline, TMAToUMMAPipeline, TmaLoadOptions, UMMAtoAsyncPipeline,
};
use cute::prelude::*;
use cute::tcgen05::{CtaGroup, OperandSource};
use cute::utils::sm100::{make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma};
use cute::utils::{
    PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentTileScheduler, WorkTileInfo,
    check_mixed_cluster_shapes, mixed_cluster_arms,
};

/// The configuration of `PersistentDenseGemmKernel` (its constructor's arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentDenseGemmConfig {
    pub name: &'static str,
    /// The MMA's `(M, N)` tile (both CTAs' for a 2SM MMA).
    pub mn_tiler: (usize, usize),
    pub use_2cta_instrs: bool,
    pub cluster_shape_mn: (usize, usize),
    pub fallback_cluster_shape_mn: Option<(usize, usize)>,
    pub mainloop_stage: usize,
    pub acc_stage: usize,
    pub tma_store_stage: usize,
    pub swizzle_size: usize,
    pub raster_along_m: bool,
    /// cute-rs: give the loads `get_cta_v_map_ab`'s maps instead of the views' own.
    pub explicit_cta_v_map: bool,
}

/// The example's defaults on a small tile: 1SM, a `(1, 1)` cluster.
pub const ONE_SM: PersistentDenseGemmConfig = PersistentDenseGemmConfig {
    name: "cute_ext_dense_gemm_1sm",
    mn_tiler: (128, 64),
    use_2cta_instrs: false,
    cluster_shape_mn: (1, 1),
    fallback_cluster_shape_mn: None,
    mainloop_stage: 4,
    acc_stage: 2,
    tma_store_stage: 2,
    swizzle_size: 1,
    raster_along_m: true,
    explicit_cta_v_map: false,
};

/// A 2SM MMA on a `(2, 1)` cluster (no multicast).
pub const TWO_SM: PersistentDenseGemmConfig = PersistentDenseGemmConfig {
    name: "cute_ext_dense_gemm_2sm",
    mn_tiler: (256, 64),
    use_2cta_instrs: true,
    cluster_shape_mn: (2, 1),
    ..ONE_SM
};

/// A 2SM MMA on a `(2, 2)` cluster: A multicast along N, B along M.
pub const TWO_SM_MULTICAST: PersistentDenseGemmConfig =
    PersistentDenseGemmConfig { name: "cute_ext_dense_gemm_2sm_mcast", cluster_shape_mn: (2, 2), ..TWO_SM };

/// A 1SM MMA on `(1, 2)` preferred clusters (multicast of A) with `(1, 1)` fallback
/// clusters, a swizzled schedule rastered along N.
pub const ONE_SM_FALLBACK: PersistentDenseGemmConfig = PersistentDenseGemmConfig {
    name: "cute_ext_dense_gemm_1sm_fallback",
    cluster_shape_mn: (1, 2),
    fallback_cluster_shape_mn: Some((1, 1)),
    swizzle_size: 2,
    raster_along_m: false,
    ..ONE_SM
};

/// 32 values per thread (one `mma_inst_shape_k` of 16 times 4).
const MMA_INST_TILE_K: usize = 4;

impl PersistentDenseGemmConfig {
    /// The CTAs of an MMA.
    pub fn atom_ctas(&self) -> usize {
        if self.use_2cta_instrs { 2 } else { 1 }
    }

    pub fn cluster_shape(&self) -> [usize; 3] {
        [self.cluster_shape_mn.0, self.cluster_shape_mn.1, 1]
    }

    pub fn fallback_cluster_shape(&self) -> Option<[usize; 3]> {
        self.fallback_cluster_shape_mn.map(|(x, y)| [x, y, 1])
    }

    /// The constructor's checks.
    pub fn check(&self) {
        for (name, v) in [
            ("mainloop_stage", self.mainloop_stage),
            ("acc_stage", self.acc_stage),
            ("tma_store_stage", self.tma_store_stage),
            ("swizzle_size", self.swizzle_size),
        ] {
            assert!(v > 0, "{}: {name} must be a positive integer, got {v}", self.name);
        }
        if let Some((fm, fn_)) = self.fallback_cluster_shape_mn {
            assert!(fm >= 1 && fn_ >= 1, "{}: fallback_cluster_shape_mn dims must be >= 1", self.name);
            assert!(
                !self.use_2cta_instrs || (fm >= 2 && fm % 2 == 0),
                "{}: 2CTA mode needs fallback_cluster_shape_mn[0] a multiple of 2",
                self.name
            );
            if let Err(e) = check_mixed_cluster_shapes(self.cluster_shape(), [fm, fn_, 1], None) {
                panic!("{}: {e}", self.name);
            }
        }
    }

    /// The CTA tile `(M, N)` (`__call__`'s `cta_tile_mnk`).
    pub fn cta_tile_mn(&self) -> (usize, usize) {
        (self.mn_tiler.0 / self.atom_ctas(), self.mn_tiler.1)
    }

    /// The static persistent schedule of an `(m, n)` D (`__call__`'s `sched_params`).
    pub fn sched_params(&self, num_ctas_mnl: impl Into<Shape>) -> PersistentTileSchedulerParams {
        let options =
            PersistentTileSchedulerOptions { swizzle_size: self.swizzle_size, raster_along_m: self.raster_along_m };
        let p = PersistentTileSchedulerParams::new(num_ctas_mnl, self.cluster_shape(), options);
        match self.fallback_cluster_shape() {
            Some(f) => p.with_fallback_cluster_shape_mnk(f),
            None => p,
        }
    }

    /// The launch of an `(m, n)` problem with at most `max_active_clusters` clusters
    /// (`__call__`'s grid, cluster and fallback cluster).
    pub fn launch(&self, m: usize, n: usize, max_active_clusters: u32) -> LaunchConfig {
        let (tm, tn) = self.cta_tile_mn();
        let grid = self.sched_params((m.div_ceil(tm), n.div_ceil(tn), 1)).get_grid_shape(max_active_clusters);
        let launch = LaunchConfig::new(grid, [192, 1, 1]);
        let c = self.cluster_shape().map(|x| x as u32);
        match self.fallback_cluster_shape() {
            Some(f) => launch.with_fallback_cluster(c, f.map(|x| x as u32)),
            None if c != [1, 1, 1] => launch.with_cluster(c),
            None => launch,
        }
    }

    /// The kernel: parameters `m`, `n`, `k` (launch values), then `a`, `b` and `d`.
    pub fn build(&self) -> Kernel {
        self.check();
        let cfg = *self;
        Kernel::builder(cfg.name).block_dim([192, 1, 1]).target(Arch::Sm100a).build(move || kernel(&cfg))
    }
}

/// `PersistentDenseGemmKernel.kernel`.
fn kernel(cfg: &PersistentDenseGemmConfig) {
    let m = param_extent("m", 1);
    let n = param_extent("n", 4);
    let k = param_extent("k", 8);
    let m_a = param_tensor_aligned::<F16>("a", Layout::new((m, k), (k, 1)), 16);
    let m_b = param_tensor_aligned::<F16>("b", Layout::new((n, k), (k, 1)), 16);
    let m_d = param_tensor_aligned::<f32>("d", Layout::new((m, n), (n, 1)), 16);

    let cta_group = if cfg.use_2cta_instrs { CtaGroup::TWO } else { CtaGroup::ONE };
    let tiled_mma = make_trivial_tiled_mma(
        DType::F16,
        DType::F16,
        OperandMajorMode::K,
        OperandMajorMode::K,
        DType::F32,
        cta_group,
        cfg.mn_tiler,
        OperandSource::SMEM,
    );
    let mma_inst_shape_k = tiled_mma.atom().shape_mnk()[2];
    let mnk_tiler = (cfg.mn_tiler.0, cfg.mn_tiler.1, mma_inst_shape_k * MMA_INST_TILE_K);
    let num_mma_ctas = tiled_mma.thr_id().size();
    let cta_tile_shape_mnk = (mnk_tiler.0 / num_mma_ctas, mnk_tiler.1, mnk_tiler.2);

    let a_smem_layout_staged = make_smem_layout_a(&tiled_mma, mnk_tiler, DType::F16, cfg.mainloop_stage, None);
    let b_smem_layout_staged = make_smem_layout_b(&tiled_mma, mnk_tiler, DType::F16, cfg.mainloop_stage, None);
    let tmem_layout = experimental::make_tmem_layout_acc(&tiled_mma, mnk_tiler, cfg.acc_stage);
    let buffer_a = experimental::allocate::<F16>(AddressSpace::Smem, a_smem_layout_staged, 1024);
    let buffer_b = experimental::allocate::<F16>(AddressSpace::Smem, b_smem_layout_staged, 1024);
    let buffer_acc = experimental::allocate_with::<f32>(AddressSpace::Tmem, tmem_layout, 16, cfg.use_2cta_instrs);

    // The CTA tiles of D, for the scheduler (`__call__`'s `num_ctas_mnl`).
    let ntile = |e: Int32, t: usize| IndexExpr::from((e + (t as i32 - 1)) / t as i32);
    let num_ctas_mnl = Shape::from(Tree::Branch(vec![
        Tree::Leaf(ntile(m, cta_tile_shape_mnk.0)),
        Tree::Leaf(ntile(n, cta_tile_shape_mnk.1)),
        Tree::Leaf(IndexExpr::from(1)),
    ]));

    // CuTe DSL's compiler clones the body per cluster shape; here each arm is built with
    // its static shape.
    mixed_cluster_arms(cfg.cluster_shape(), cfg.fallback_cluster_shape(), |arm| {
        let shape = arm.cluster_shape_mnk;
        let cluster_layout_vmnk = Layout::col_major((shape[0], shape[1], shape[2]))
            .tiled_divide(&Tree::Branch(vec![Tree::Leaf(Some(tiled_mma.thr_id().clone()))]));
        let cluster_layout_v_size = cluster_layout_vmnk.mode(0).size() as i32;
        let cta_m = block_idx().x;
        let mma_coord_v = cta_m % cluster_layout_v_size;
        let is_leader_cta = mma_coord_v.eq(0);
        let thr_mma = tiled_mma.get_slice(mma_coord_v);

        let selfcast = if cfg.use_2cta_instrs { (2, 1) } else { (1, 1) };
        let use_tma_multicast = (shape[0], shape[1]) != selfcast;
        let (mma_operation_type, tma_operation_type) = match (cfg.use_2cta_instrs, use_tma_multicast) {
            (true, true) => (OperationType::Sm100Mma2SmSs, OperationType::Sm100TmaLoad2SmMulticast),
            (true, false) => (OperationType::Sm100Mma2SmSs, OperationType::Sm100TmaLoad2Sm),
            (false, true) => (OperationType::Sm100Mma1SmSs, OperationType::Sm90TmaLoadMulticast),
            (false, false) => (OperationType::Sm100Mma1SmSs, OperationType::Sm90TmaLoad),
        };
        let mainloop_pipe = TMAToUMMAPipeline::create(
            cfg.mainloop_stage,
            mma_operation_type,
            Some(tma_operation_type),
            Some(&cluster_layout_vmnk),
        );
        let acc_pipe_consumer_arv_count = if cfg.use_2cta_instrs { 256 } else { 128 };
        let acc_pipe = UMMAtoAsyncPipeline::create(
            cfg.acc_stage,
            OperationType::Sm100CopyT2r,
            acc_pipe_consumer_arv_count,
            mma_operation_type,
            Some(&cluster_layout_vmnk),
        );

        let warp_idx = make_warp_uniform(warp_idx());
        let tma_store_warp_id = 0;
        let mma_warp_id = 4;
        let tma_load_warp_id = 5;
        let is_tma_thr = warp_idx.eq(tma_load_warp_id);
        let is_mma_thr = warp_idx.eq(mma_warp_id);
        let is_epi_thr = warp_idx.lt(4);

        let tma_store_pipe = TMAStorePipeline::new(cfg.tma_store_stage, 128, 1, tma_store_warp_id as u32);

        let k_tile_size = (k + (mnk_tiler.2 as i32 - 1)) / mnk_tiler.2 as i32;
        let tile_sched =
            StaticPersistentTileScheduler::create(&cfg.sched_params(num_ctas_mnl.clone()), block_idx(), grid_dim());
        let work_tile = tile_sched.initial_work_tile_info();

        // (bM, bK, RestM, RestK) and (bN, bK, RestN, RestK)
        let g_a_mkl = local_tile(&m_a, (mnk_tiler.0, mnk_tiler.2), (.., ..));
        let g_b_nkl = local_tile(&m_b, (mnk_tiler.1, mnk_tiler.2), (.., ..));
        // (MMA, MMA_M, MMA_K, RestM, RestK) and (MMA, MMA_N, MMA_K, RestN, RestK)
        let t_cg_a = thr_mma.partition_a(&g_a_mkl);
        let t_cg_b = thr_mma.partition_b(&g_b_nkl);

        // The loads' options: the operation, and the multicast modes (A along N, B along M).
        let load_options = |mode: usize, map: Option<Layout>| {
            let mut o = TmaLoadOptions::new().tma_operation_type(tma_operation_type);
            if use_tma_multicast {
                o = o.vmnk_layout(&cluster_layout_vmnk).multicast_mode(mode);
            }
            if let Some(map) = map {
                o = o.cta_v_map(&map);
            }
            o
        };
        let map = |op| {
            let t = if op == experimental::MmaOperand::A { &m_a } else { &m_b };
            cfg.explicit_cta_v_map.then(|| experimental::get_cta_v_map_ab(t, mnk_tiler, &tiled_mma, op))
        };
        let tma_mcast_proj_a = 2;
        let tma_mcast_proj_b = 1;
        let opts_a = load_options(tma_mcast_proj_a, map(experimental::MmaOperand::A));
        let opts_b = load_options(tma_mcast_proj_b, map(experimental::MmaOperand::B));

        // The persistent loop of a role: `body` per tile, threading `state`.
        let for_each_tile = |state: (TMAToUMMAPipeline, UMMAtoAsyncPipeline, TMAStorePipeline),
                             body: &dyn Fn(
            WorkTileInfo,
            (TMAToUMMAPipeline, UMMAtoAsyncPipeline, TMAStorePipeline),
        ) -> (TMAToUMMAPipeline, UMMAtoAsyncPipeline, TMAStorePipeline)| {
            let (_, _, state) = while_carry(
                (work_tile, tile_sched.clone(), state),
                |(w, _, _)| w.is_valid_tile(),
                |(w, mut sched, state)| {
                    let state = body(w, state);
                    sched.advance_to_next_work();
                    (sched.get_current_work(), sched, state)
                },
            );
            state
        };
        let pipes = (mainloop_pipe.clone(), acc_pipe.clone(), tma_store_pipe);

        if_(is_tma_thr, || {
            let (mut mainloop_pipe, _, _) = for_each_tile(pipes.clone(), &|w, (mainloop_pipe, acc, store)| {
                let cur_tile_coord = w.tile_idx();
                let mma_tile_coord_m = cur_tile_coord[0] / cluster_layout_v_size;
                let t_ag_a_slice = t_cg_a.slice((.., .., .., mma_tile_coord_m, ..));
                let t_bg_b_slice = t_cg_b.slice((.., .., .., cur_tile_coord[1], ..));
                let mainloop_pipe = for_range_carry(0, k_tile_size, 1, mainloop_pipe, |k, mut mainloop_pipe| {
                    let g_a_k = t_ag_a_slice.slice((.., .., .., k));
                    let g_b_k = t_bg_b_slice.slice((.., .., .., k));
                    let (producer_stage_token, idx) = mainloop_pipe.producer_acquire_and_get_stage(None);
                    let mbar = experimental::get_mbarrier(&producer_stage_token);
                    let buffer_a_sliced = buffer_a.slice((.., .., .., idx));
                    let buffer_b_sliced = buffer_b.slice((.., .., .., idx));
                    experimental::tma_load_with(&g_a_k, &buffer_a_sliced, &mbar, &opts_a);
                    experimental::tma_load_with(&g_b_k, &buffer_b_sliced, &mbar, &opts_b);
                    mainloop_pipe.producer_commit_and_advance(false, false);
                    mainloop_pipe
                });
                (mainloop_pipe, acc, store)
            });
            // cute-rs: no CTA exits while a commit to its barriers may be in flight.
            mainloop_pipe.producer_tail();
        });

        if_(is_mma_thr & is_leader_cta, || {
            let (_, mut acc_pipe, _) = for_each_tile(pipes.clone(), &|_, (mainloop_pipe, mut acc_pipe, store)| {
                let (_, idx) = acc_pipe.producer_acquire_and_get_stage(None);
                let accumulators_sliced = buffer_acc.slice((.., .., .., idx));
                let (updated_a_pipe, _updated_b_pipe) = experimental::mainloop_mma(
                    &tiled_mma,
                    &buffer_a,
                    &buffer_b,
                    &accumulators_sliced,
                    0,
                    k_tile_size,
                    MMA_INST_TILE_K,
                    mainloop_pipe.clone(),
                    mainloop_pipe,
                    true,
                    false,
                );
                acc_pipe.producer_commit_and_advance(false, false);
                (updated_a_pipe, acc_pipe, store)
            });
            // cute-rs: wait for the epilogue's last releases before exiting.
            acc_pipe.producer_tail();
        });

        if_(is_epi_thr, || {
            for_each_tile(pipes.clone(), &|w, (mainloop_pipe, mut acc_pipe, tma_store_pipe)| {
                let (_, idx) = acc_pipe.consumer_wait_and_get_stage(None);
                let accumulators_sliced = buffer_acc.slice((.., .., .., idx));
                let t = w.tile_idx();
                let tma_store_pipe = experimental::epilogue_tma_store(
                    cta_tile_shape_mnk,
                    cfg.use_2cta_instrs,
                    &accumulators_sliced,
                    &m_d,
                    &[t[0], t[1]],
                    tma_store_pipe,
                    tma_store_warp_id,
                    |x| x,
                );
                acc_pipe.consumer_release_and_advance();
                (mainloop_pipe, acc_pipe, tma_store_pipe)
            });
        });
    });
}

/// Integers in `[-2, 2)` (exact in Float16, and their sums in Float32).
pub fn small_ints(n: usize, seed: u64) -> Vec<F16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            F16::from_f32(((s >> 40) % 4) as f32 - 2.0)
        })
        .collect()
}

/// Random Float16 values in `[-1, 1)`.
pub fn random_f16(n: usize, seed: u64) -> Vec<F16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            F16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        })
        .collect()
}

/// `einsum("mk,nk->mn")` in Float64, for the comparison (`compare`).
pub fn reference(a: &[F16], b: &[F16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(m * n);
    for r in 0..m {
        for c in 0..n {
            out.push((0..k).map(|i| a[r * k + i].to_f32() as f64 * b[c * k + i].to_f32() as f64).sum());
        }
    }
    out
}

/// Simulate `cfg` on an `m x n x k` problem, with at most `max_active_clusters` clusters,
/// and check D: bit for bit on small integers (`exact`), otherwise within Float32's bound
/// for a sum of `k` products of magnitude at most 1 (`k * 2^-23` per element, relative to
/// the sum of the products' magnitudes). Returns the report.
pub fn run(
    cfg: &PersistentDenseGemmConfig,
    (m, n, k): (usize, usize, usize),
    max_active_clusters: u32,
    exact: bool,
) -> SimReport {
    run_with(cfg, (m, n, k), max_active_clusters, exact, &SimOptions::default())
}

/// [`run`] with simulator options.
pub fn run_with(
    cfg: &PersistentDenseGemmConfig,
    (m, n, k): (usize, usize, usize),
    max_active_clusters: u32,
    exact: bool,
    opts: &SimOptions,
) -> SimReport {
    let kernel = cfg.build();
    let (a, b) = if exact {
        (small_ints(m * k, 1111), small_ints(n * k, 2222))
    } else {
        (random_f16(m * k, 1111), random_f16(n * k, 2222))
    };
    let mut args = vec![
        KernelArg::scalar(m as i32),
        KernelArg::scalar(n as i32),
        KernelArg::scalar(k as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&b),
        KernelArg::f32(vec![-7.0; m * n]),
    ];
    let report = kernel
        .simulate_with(cfg.launch(m, n, max_active_clusters), &mut args, opts)
        .unwrap_or_else(|e| panic!("{} {m}x{n}x{k}: {e}", cfg.name));
    let got = args[5].as_f32();
    let want = reference(&a, &b, m, n, k);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let (r, c) = (i / n, i % n);
        if exact {
            assert_eq!(g as f64, w, "{} {m}x{n}x{k}: D({r}, {c})", cfg.name);
        } else {
            let mag: f64 = (0..k).map(|j| (a[r * k + j].to_f32() as f64 * b[c * k + j].to_f32() as f64).abs()).sum();
            let bound = k as f64 * f32::EPSILON as f64 * mag.max(f64::MIN_POSITIVE);
            assert!(
                (g as f64 - w).abs() <= bound,
                "{} {m}x{n}x{k}: D({r}, {c}) is {g}, not {w} (bound {bound})",
                cfg.name
            );
        }
    }
    report
}

#[allow(dead_code)]
fn main() {
    for (cfg, mnk, clusters) in [
        (ONE_SM, (256, 128, 128), 2),
        (TWO_SM, (256, 128, 128), 2),
        (TWO_SM_MULTICAST, (256, 128, 128), 1),
        (ONE_SM_FALLBACK, (200, 100, 72), 4),
    ] {
        let report = run(&cfg, mnk, clusters, true);
        println!(
            "{}: {}x{}x{} simulated, D matches the reference ({} tcgen05.mma)",
            cfg.name, mnk.0, mnk.1, mnk.2, report.tcgen05_mma_instructions
        );
    }
}

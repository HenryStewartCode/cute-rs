//! Cluster launch control on the simulator: a
//! persistent kernel whose clusters claim the tiles of clusters that have not started
//! (`PipelineClcFetchAsync` and `ClcDynamicPersistentTileScheduler`) processes every
//! tile once, in the static scheduler's order, under several grids, cluster shapes,
//! resident-cluster counts and launch seeds; the misuse reports; and the sm_100a PTX
//! (assembled by `ptxas` when it is on the PATH).

use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineBarriers, PipelineClcFetchAsync, PipelineClcFetchAsyncOptions,
    PipelineUserType, make_pipeline_state,
};
use cute::prelude::*;
use cute::utils::{
    ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerOptions,
    ClcDynamicPersistentTileSchedulerParams, PersistentTileSchedulerOptions, PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
};

/// A problem and its schedule.
#[derive(Clone, Copy, Debug)]
struct Case {
    tiles: (usize, usize, usize),
    cluster: (usize, usize),
    swizzle: usize,
    raster_along_m: bool,
}

impl Case {
    fn ntiles(&self) -> usize {
        self.tiles.0 * self.tiles.1 * self.tiles.2
    }

    fn clc_params(&self) -> ClcDynamicPersistentTileSchedulerParams {
        let options = ClcDynamicPersistentTileSchedulerOptions {
            swizzle_size: self.swizzle,
            raster_along_m: self.raster_along_m,
            ..Default::default()
        };
        ClcDynamicPersistentTileSchedulerParams::new(self.tiles, [self.cluster.0, self.cluster.1, 1], options)
    }

    fn static_params(&self) -> PersistentTileSchedulerParams {
        let options =
            PersistentTileSchedulerOptions { swizzle_size: self.swizzle, raster_along_m: self.raster_along_m };
        PersistentTileSchedulerParams::new(self.tiles, [self.cluster.0, self.cluster.1, 1], options)
    }

    fn cluster_size(&self) -> usize {
        self.cluster.0 * self.cluster.1
    }
}

/// What a faulty kernel leaves out.
#[derive(Clone, Copy, Default)]
struct Faults {
    /// The worker warps never release the response stage.
    no_release: bool,
    /// The empty barrier counts one CTA's worker warp only.
    short_empty_count: bool,
}

/// Store `owner + 1` for an in-bounds tile and count its visits.
fn record(case: Case, tile: [Int32; 3], owner: Int32, owners: &Tensor<i32>, hits: &Tensor<i32>) {
    let (m, n, _) = case.tiles;
    let inside = tile[0].lt(m as i32) & tile[1].lt(n as i32);
    if_(inside & lane_idx().eq(0), || {
        let idx = tile[0] + tile[1] * m as i32 + tile[2] * (m * n) as i32;
        owners.store(idx, owner + 1);
        atomic_add(&hits.ptr().offset(idx), 1);
    });
}

/// The linear index, in the visiting order, of the cluster whose first CTA is at
/// `(x, y, z)` in the CLC grid.
fn cluster_id(case: Case, x: Int32, y: Int32, z: Int32) -> Int32 {
    let (cm, cn) = case.cluster;
    if case.swizzle == 1 && case.raster_along_m {
        let ncm = case.tiles.0.div_ceil(cm) as i32;
        let ncn = case.tiles.1.div_ceil(cn) as i32;
        x / cm as i32 + (y / cn as i32) * ncm + z * (ncm * ncn)
    } else {
        z
    }
}

/// The persistent CLC kernel: warp 0 of each CTA processes tiles; warp 1 of CTA 0 is the
/// cluster's scheduler (producer, and a consumer to learn when to stop). Each tile's
/// owner is the visiting-order index of the cluster whose work it was.
fn clc_kernel(case: Case, faults: Faults) -> Kernel {
    let params = case.clc_params();
    let (cm, cn) = case.cluster;
    let cs = case.cluster_size();
    Kernel::builder(format!("clc_persistent_{}x{}", cm, cn)).block_dim([64, 1, 1]).target(Arch::Sm100a).build(
        move || {
            let owners = param_tensor::<i32>("owners", Layout::col_major(case.ntiles()));
            let hits = param_tensor::<i32>("hits", Layout::col_major(case.ntiles()));
            let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
            let consumers = if faults.short_empty_count { 32 } else { 32 * (cs + 1) };
            let pipeline = PipelineClcFetchAsync::create(
                1,
                CooperativeGroup::new(Agent::Thread, 1),
                CooperativeGroup::new(Agent::Thread, consumers),
                16,
                PipelineClcFetchAsyncOptions {
                    cta_layout_vmnk: Some(Layout::col_major((1usize, cm, cn, 1usize))),
                    ..Default::default()
                },
            );
            let sched =
                ClcDynamicPersistentTileScheduler::create(&params, block_idx(), grid_dim(), &response.ptr(), true);
            let work = sched.initial_work_tile_info();
            let b = block_idx();
            let rank = sched.cta_id_in_cluster();
            let own = cluster_id(case, b.x - rank[0], b.y - rank[1], b.z);
            let warp = make_warp_uniform(warp_idx());
            if_else(
                warp.eq(0),
                || {
                    let consumer = make_pipeline_state(PipelineUserType::Consumer, 1);
                    while_carry(
                        (work, own, consumer),
                        |(w, _, _)| w.is_valid_tile(),
                        |(w, owner, mut c)| {
                            record(case, w.tile_idx(), owner, &owners, &hits);
                            pipeline.consumer_wait(&c, None);
                            let (x, y, z, _) = clc_response(&response.ptr());
                            let next = sched.get_current_work();
                            if !faults.no_release {
                                pipeline.consumer_release(&c);
                            }
                            c.advance();
                            (next, cluster_id(case, x, y, z), c)
                        },
                    );
                },
                || {
                    if_(make_warp_uniform(block_idx_in_cluster()).eq(0), || {
                        let producer = make_pipeline_state(PipelineUserType::Producer, 1);
                        let consumer = make_pipeline_state(PipelineUserType::Consumer, 1);
                        let (_, mut producer, _, _) = while_carry(
                            (work, producer, consumer, sched.clone()),
                            |(w, _, _, _)| w.is_valid_tile(),
                            |(_, mut p, mut c, mut s)| {
                                pipeline.producer_acquire(&p, None);
                                s.advance_to_next_work(&pipeline.producer_get_barrier(&p));
                                p.advance();
                                pipeline.consumer_wait(&c, None);
                                let next = s.get_current_work();
                                pipeline.consumer_release(&c);
                                c.advance();
                                (next, p, c, s)
                            },
                        );
                        pipeline.producer_tail(&mut producer);
                    });
                },
            );
            cluster_arrive_relaxed();
            cluster_wait();
        },
    )
}

/// The static persistent scheduler with one cluster per work index: each tile's owner
/// is its work index.
fn static_owners(case: Case) -> Vec<i32> {
    let params = case.static_params();
    let grid = params.get_grid_shape(u32::MAX / 64);
    let kernel = Kernel::builder("static_order").block_dim([32, 1, 1]).build(move || {
        let owners = param_tensor::<i32>("owners", Layout::col_major(case.ntiles()));
        let hits = param_tensor::<i32>("hits", Layout::col_major(case.ntiles()));
        let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
        let work = sched.initial_work_tile_info();
        if_(work.is_valid_tile(), || record(case, work.tile_idx(), block_idx().z, &owners, &hits));
    });
    let mut args = vec![KernelArg::buffer(&vec![0i32; case.ntiles()]), KernelArg::buffer(&vec![0i32; case.ntiles()])];
    let cfg = LaunchConfig::new(grid, [32, 1, 1]).with_cluster([case.cluster.0 as u32, case.cluster.1 as u32, 1]);
    let opts = SimOptions { resident_clusters: ResidentClusters::Grid, ..Default::default() };
    kernel.simulate_with(cfg, &mut args, &opts).unwrap();
    assert!(args[1].to_vec::<i32>().iter().all(|&h| h == 1), "static: {:?}", args[1].to_vec::<i32>());
    args[0].to_vec::<i32>()
}

fn launch(case: Case) -> LaunchConfig {
    let grid = case.clc_params().get_grid_shape();
    LaunchConfig::new(grid, [64, 1, 1]).with_cluster([case.cluster.0 as u32, case.cluster.1 as u32, 1])
}

fn run(case: Case, faults: Faults, opts: &SimOptions) -> Result<(Vec<i32>, Vec<i32>, SimReport), SimError> {
    let kernel = clc_kernel(case, faults);
    let mut args = vec![KernelArg::buffer(&vec![0i32; case.ntiles()]), KernelArg::buffer(&vec![0i32; case.ntiles()])];
    let report = kernel.simulate_with(launch(case), &mut args, opts)?;
    Ok((args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), report))
}

const CASES: [Case; 6] = [
    Case { tiles: (4, 3, 1), cluster: (1, 1), swizzle: 1, raster_along_m: true },
    Case { tiles: (5, 4, 2), cluster: (2, 1), swizzle: 1, raster_along_m: true },
    Case { tiles: (4, 6, 1), cluster: (2, 2), swizzle: 1, raster_along_m: false },
    Case { tiles: (6, 4, 1), cluster: (1, 2), swizzle: 2, raster_along_m: true },
    Case { tiles: (6, 3, 2), cluster: (2, 1), swizzle: 2, raster_along_m: false },
    Case { tiles: (3, 5, 1), cluster: (1, 1), swizzle: 3, raster_along_m: true },
];

/// For several grids, cluster shapes, resident-cluster counts and launch seeds: every
/// tile is processed exactly once, by the cluster the static scheduler's order gives it;
/// the cancelled clusters never started (the blocks run plus the cancelled clusters'
/// blocks are the grid); and with one resident cluster, every other cluster is claimed.
#[test]
fn every_tile_is_processed_once_in_the_static_order() {
    for case in CASES {
        let expected = static_owners(case);
        let grid = case.clc_params().get_grid_shape();
        let grid_blocks = grid.iter().map(|&g| g as u64).product::<u64>();
        let cs = case.cluster_size() as u64;
        let configs = [
            (ResidentClusters::Count(1), None, Schedule::RoundRobin),
            (ResidentClusters::Count(2), None, Schedule::RunToBlock),
            (ResidentClusters::Count(3), Some(7), Schedule::Random { seed: 3 }),
            (ResidentClusters::Count(2), Some(11), Schedule::RoundRobin),
            (ResidentClusters::Grid, None, Schedule::RoundRobin),
        ];
        for (resident, seed, schedule) in configs {
            let opts =
                SimOptions { resident_clusters: resident, clc_launch_seed: seed, schedule, ..Default::default() };
            let (owners, hits, report) =
                run(case, Faults::default(), &opts).unwrap_or_else(|e| panic!("{case:?} {resident:?} {seed:?}: {e}"));
            assert!(hits.iter().all(|&h| h == 1), "{case:?} {resident:?} {seed:?}: hits {hits:?}");
            assert_eq!(owners, expected, "{case:?} {resident:?} {seed:?}: owners");
            assert_eq!(report.blocks + report.clc_cancelled * cs, grid_blocks, "{case:?} {resident:?}: {report}");
            match resident {
                ResidentClusters::Grid => assert_eq!(report.clc_cancelled, 0),
                ResidentClusters::Count(1) if seed.is_none() => assert_eq!(report.blocks, cs, "{case:?}"),
                _ => {}
            }
        }
    }
}

/// With random async completion the responses still arrive through the pipeline.
#[test]
fn responses_arrive_under_random_completion() {
    let case = CASES[2];
    let expected = static_owners(case);
    for seed in 0..4 {
        let opts = SimOptions {
            resident_clusters: ResidentClusters::Count(2),
            async_completion: AsyncCompletion::Random(seed),
            schedule: Schedule::Random { seed },
            clc_launch_seed: Some(seed),
            ..Default::default()
        };
        let (owners, hits, _) = run(case, Faults::default(), &opts).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(hits.iter().all(|&h| h == 1), "seed {seed}: {hits:?}");
        assert_eq!(owners, expected);
    }
}

#[test]
fn a_consumer_that_never_releases_deadlocks() {
    let faults = Faults { no_release: true, ..Default::default() };
    let e = run(CASES[1], faults, &SimOptions::default()).unwrap_err().to_string();
    assert!(e.contains("deadlock"), "{e}");
}

#[test]
fn an_empty_barrier_count_short_of_the_clusters_consumers_is_reported() {
    let faults = Faults { short_empty_count: true, ..Default::default() };
    let e = run(CASES[2], faults, &SimOptions::default()).unwrap_err().to_string();
    // The empty barrier completes its phases on CTA 0's arrivals alone and runs ahead of
    // the scheduler's waits.
    assert!(e.contains("phase aliasing"), "{e}");
}

/// A multicast query after a CTA of the cluster has exited: CTA 1 exits at once and CTA 0
/// issues the query once it sees CTA 1's flag.
#[test]
fn a_multicast_query_after_a_cta_exited_is_reported() {
    let kernel = Kernel::builder("clc_after_exit").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        let flag = param_tensor::<i32>("flag", Layout::col_major(1));
        let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
        let bar = make_smem_tensor_aligned::<u64>(Layout::col_major(1), 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&bar.ptr(), 1));
        mbarrier_init_fence();
        cluster_arrive();
        cluster_wait();
        if_else(
            block_idx_in_cluster().eq(1),
            || {
                if_(thread_idx().x.eq(0), || {
                    atomic_add(&flag.ptr(), 1);
                });
            },
            || {
                while_carry((), |()| flag.load(0).eq(0), |()| ());
                for _ in 0..8 {
                    sync_warp();
                }
                if_(thread_idx().x.eq(0), || issue_clc_query(&bar.ptr(), &response.ptr(), true));
            },
        );
    });
    let mut args = vec![KernelArg::buffer(&[0i32])];
    let cfg = LaunchConfig::new([2, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    let e = kernel.simulate(cfg, &mut args).unwrap_err().to_string();
    assert!(e.contains("try_cancel") && e.contains("has exited"), "{e}");
}

/// The queue: with one resident cluster a non-multicast query by the only running
/// cluster cancels the next one in launch order, and an empty queue says so.
#[test]
fn queries_cancel_clusters_in_launch_order() {
    let kernel = Kernel::builder("clc_order").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(4 * 4));
        let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
        let bar = make_smem_tensor_aligned::<u64>(Layout::col_major(1), 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&bar.ptr(), 1));
        mbarrier_init_fence();
        sync_threads();
        if_(block_idx().x.eq(0), || {
            for k in 0..4i32 {
                if_(thread_idx().x.eq(0), || {
                    mbarrier_arrive_and_expect_tx(&bar.ptr(), 16);
                    issue_clc_query(&bar.ptr(), &response.ptr(), false);
                });
                mbarrier_wait(&bar.ptr(), k % 2);
                let (x, y, z, valid) = clc_response(&response.ptr());
                fence_view_async_shared();
                if_(thread_idx().x.eq(0), || {
                    out.store(4 * k, x);
                    out.store(4 * k + 1, y);
                    out.store(4 * k + 2, z);
                    out.store(4 * k + 3, valid);
                });
                sync_warp();
            }
        });
    });
    let mut args = vec![KernelArg::buffer(&[-1i32; 16])];
    let report = kernel.simulate(LaunchConfig::new([2, 2, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]);
    assert_eq!((report.blocks, report.clc_cancelled), (1, 3));
}

/// The fallback cluster shape is kept as a second schedule; the grid shapes follow the
/// raster and swizzle.
#[test]
fn params_grid_shapes_and_fallback() {
    let p = ClcDynamicPersistentTileSchedulerParams::new((5, 3, 2), [2, 1, 1], Default::default());
    assert_eq!(p.get_grid_shape(), [6, 3, 2]);
    let options = ClcDynamicPersistentTileSchedulerOptions { raster_along_m: false, ..Default::default() };
    assert_eq!(
        ClcDynamicPersistentTileSchedulerParams::new((5, 3, 2), [2, 1, 1], options).get_grid_shape(),
        [2, 1, 18]
    );
    let options = ClcDynamicPersistentTileSchedulerOptions { swizzle_size: 2, ..Default::default() };
    // 3 x 3 clusters, N padded to 4.
    assert_eq!(
        ClcDynamicPersistentTileSchedulerParams::new((6, 3, 1), [2, 1, 1], options).get_grid_shape(),
        [2, 1, 12]
    );
    let options =
        ClcDynamicPersistentTileSchedulerOptions { fallback_cluster_shape_mnk: Some([1, 1, 1]), ..Default::default() };
    let p = ClcDynamicPersistentTileSchedulerParams::new((4, 4, 1), [2, 2, 1], options);
    assert_eq!(p.cluster_shape_mnk(), [2, 2, 1]);
    assert_eq!(p.fallback().unwrap().cluster_shape_mnk(), [1, 1, 1]);
    assert_eq!(p.fallback().unwrap().get_grid_shape(), [4, 4, 1]);
}

/// The query needs sm_100 (the multicast form an `a` target); the MLIR keeps the
/// response in one inline-PTX statement.
#[test]
fn targets_and_mlir() {
    let kernel = clc_kernel(CASES[2], Faults::default());
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("clusterlaunchcontrol"), "{e}");
    let text = kernel.to_mlir();
    for w in [
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128",
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128",
        "get_first_ctaid::z.b32.b128",
    ] {
        assert!(text.contains(w), "no `{w}` in the MLIR");
    }
}

#[test]
fn clc_ptx_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let kernel = clc_kernel(CASES[2], Faults::default());
    let compiled = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap();
    let ptx = compiled.ptx();
    for w in [
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128",
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128",
    ] {
        assert!(ptx.contains(w), "no `{w}` in the PTX");
    }
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    let Some(ptxas) = ptxas else {
        eprintln!("skipping ptxas: not on the PATH");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cute-clc-{}", std::process::id()));
    let (src, cubin) = (dir.with_extension("ptx"), dir.with_extension("cubin"));
    std::fs::write(&src, ptx).unwrap();
    let out = std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&src).arg("-o").arg(&cubin).output().unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&cubin);
    assert!(
        out.status.success(),
        "ptxas: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------------------
// Preferred and fallback clusters.
// ---------------------------------------------------------------------------------------

use cute::utils::{ClusterArm, current_cluster_arm, mixed_cluster_arms};

const MIXED_TILES: (usize, usize, usize) = (4, 8, 1);
const PREFERRED: [usize; 3] = [2, 4, 1];
const FALLBACK: [usize; 3] = [2, 1, 1];

/// A persistent kernel built twice by `mixed_cluster_arms`, for preferred `(2, 4, 1)` and
/// fallback `(2, 1, 1)` clusters, with the CLC scheduler (warp 0 works, warp 1 of CTA 0
/// schedules) or the static one: each tile counts its visits and records the arm
/// (1 preferred, 2 fallback) that visited it.
fn mixed_kernel(clc: bool) -> Kernel {
    let ntiles = MIXED_TILES.0 * MIXED_TILES.1;
    Kernel::builder(if clc { "mixed_clc" } else { "mixed_static" }).block_dim([64, 1, 1]).target(Arch::Sm100a).build(
        move || {
            let hits = param_tensor::<i32>("hits", Layout::col_major(ntiles));
            let arms = param_tensor::<i32>("arms", Layout::col_major(ntiles));
            let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
            let barriers = make_smem_tensor_aligned::<u64>(Layout::col_major(2), 8);
            let record = |arm: ClusterArm, tile: [Int32; 3]| {
                let (m, n, _) = MIXED_TILES;
                let inside = tile[0].lt(m as i32) & tile[1].lt(n as i32);
                if_(inside & lane_idx().eq(0), || {
                    let idx = tile[0] + tile[1] * m as i32;
                    atomic_add(&hits.ptr().offset(idx), 1);
                    arms.store(idx, if arm.is_preferred { 1 } else { 2 });
                });
            };
            mixed_cluster_arms(PREFERRED, Some(FALLBACK), |arm| {
                assert_eq!(current_cluster_arm(), Some(arm));
                let [cm, cn, _] = arm.cluster_shape_mnk;
                let warp = make_warp_uniform(warp_idx());
                if !clc {
                    let params = PersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, Default::default())
                        .with_fallback_cluster_shape_mnk(FALLBACK);
                    let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
                    if_(warp.eq(0), || {
                        while_carry(
                            (sched.initial_work_tile_info(), sched.clone()),
                            |(w, _)| w.is_valid_tile(),
                            |(w, mut s)| {
                                record(arm, w.tile_idx());
                                s.advance_to_next_work();
                                (s.get_current_work(), s)
                            },
                        );
                    });
                    return;
                }
                let options = ClcDynamicPersistentTileSchedulerOptions {
                    fallback_cluster_shape_mnk: Some(FALLBACK),
                    ..Default::default()
                };
                let params = ClcDynamicPersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, options);
                let pipeline = PipelineClcFetchAsync::create(
                    1,
                    CooperativeGroup::new(Agent::Thread, 1),
                    CooperativeGroup::new(Agent::Thread, 32 * (cm * cn + 1)),
                    16,
                    PipelineClcFetchAsyncOptions {
                        barrier_storage: Some(barriers.ptr()),
                        cta_layout_vmnk: Some(Layout::col_major((1usize, cm, cn, 1usize))),
                        ..Default::default()
                    },
                );
                let sched =
                    ClcDynamicPersistentTileScheduler::create(&params, block_idx(), grid_dim(), &response.ptr(), true);
                assert_eq!(sched.params().cluster_shape_mnk(), arm.cluster_shape_mnk);
                let work = sched.initial_work_tile_info();
                if_else(
                    warp.eq(0),
                    || {
                        let consumer = make_pipeline_state(PipelineUserType::Consumer, 1);
                        while_carry(
                            (work, consumer),
                            |(w, _)| w.is_valid_tile(),
                            |(w, mut c)| {
                                record(arm, w.tile_idx());
                                pipeline.consumer_wait(&c, None);
                                let next = sched.get_current_work();
                                pipeline.consumer_release(&c);
                                c.advance();
                                (next, c)
                            },
                        );
                    },
                    || {
                        if_(make_warp_uniform(block_idx_in_cluster()).eq(0), || {
                            let producer = make_pipeline_state(PipelineUserType::Producer, 1);
                            let consumer = make_pipeline_state(PipelineUserType::Consumer, 1);
                            let (_, mut producer, _, _) = while_carry(
                                (work, producer, consumer, sched.clone()),
                                |(w, _, _, _)| w.is_valid_tile(),
                                |(_, mut p, mut c, mut s)| {
                                    pipeline.producer_acquire(&p, None);
                                    s.advance_to_next_work(&pipeline.producer_get_barrier(&p));
                                    p.advance();
                                    pipeline.consumer_wait(&c, None);
                                    let next = s.get_current_work();
                                    pipeline.consumer_release(&c);
                                    c.advance();
                                    (next, p, c, s)
                                },
                            );
                            pipeline.producer_tail(&mut producer);
                        });
                    },
                );
                cluster_arrive_relaxed();
                cluster_wait();
            });
        },
    )
}

/// Both schedulers in a mixed-cluster kernel cover every tile exactly once under all
/// preferred, all fallback and mixed clusters (seeds 0 and 6 split one of two preferred
/// clusters), with several resident counts and CLC launch seeds; each arm runs where
/// its clusters are.
#[test]
fn mixed_cluster_kernels_cover_every_tile_once() {
    let ntiles = MIXED_TILES.0 * MIXED_TILES.1;
    for clc in [true, false] {
        let kernel = mixed_kernel(clc);
        let grid = if clc {
            ClcDynamicPersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, Default::default()).get_grid_shape()
        } else {
            PersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, Default::default()).get_grid_shape(2)
        };
        assert_eq!(grid, if clc { [4, 8, 1] } else { [2, 4, 2] });
        for (shapes, resident, seed) in [
            (ClusterShapes::Preferred, ResidentClusters::Count(1), None),
            (ClusterShapes::Fallback, ResidentClusters::Count(3), Some(3)),
            (ClusterShapes::Mixed { seed: 0 }, ResidentClusters::Count(2), Some(1)),
            (ClusterShapes::Mixed { seed: 6 }, ResidentClusters::Grid, None),
        ] {
            let opts = SimOptions {
                cluster_shapes: shapes,
                resident_clusters: resident,
                clc_launch_seed: seed,
                ..Default::default()
            };
            let mut args = vec![KernelArg::buffer(&vec![0i32; ntiles]), KernelArg::buffer(&vec![0i32; ntiles])];
            let cfg = LaunchConfig::new(grid, [64, 1, 1]).with_fallback_cluster([2, 4, 1], [2, 1, 1]);
            kernel.simulate_with(cfg, &mut args, &opts).unwrap_or_else(|e| panic!("clc {clc} {shapes:?}: {e}"));
            let hits = args[0].to_vec::<i32>();
            assert!(hits.iter().all(|&h| h == 1), "clc {clc} {shapes:?}: {hits:?}");
            let arms = args[1].to_vec::<i32>();
            let want: &[i32] = match shapes {
                ClusterShapes::Preferred => &[1],
                ClusterShapes::Fallback => &[2],
                ClusterShapes::Mixed { .. } => &[1, 2],
            };
            for w in want {
                assert!(arms.contains(w), "clc {clc} {shapes:?}: arms {arms:?}");
            }
            assert!(arms.iter().all(|a| want.contains(a)), "clc {clc} {shapes:?}: arms {arms:?}");
        }
    }
}

/// A schedule made for another cluster shape than the arm's is reported when the
/// scheduler is created (cluster-shape math that bypasses the arm's static shape).
#[test]
fn a_schedule_for_another_cluster_shape_is_reported_in_a_mixed_kernel() {
    let build = |fallback_in_params: bool, clc: bool| {
        std::panic::catch_unwind(move || {
            Kernel::builder("mixed_mismatch").block_dim([32, 1, 1]).target(Arch::Sm100a).build(move || {
                let response = make_smem_tensor_aligned::<i32>(Layout::col_major(4), 16);
                mixed_cluster_arms(PREFERRED, Some(FALLBACK), |_| {
                    let fallback = fallback_in_params.then_some(FALLBACK);
                    if clc {
                        let options = ClcDynamicPersistentTileSchedulerOptions {
                            fallback_cluster_shape_mnk: fallback,
                            ..Default::default()
                        };
                        let params = ClcDynamicPersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, options);
                        ClcDynamicPersistentTileScheduler::create(
                            &params,
                            block_idx(),
                            grid_dim(),
                            &response.ptr(),
                            true,
                        );
                    } else {
                        let mut params = PersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, Default::default());
                        if let Some(f) = fallback {
                            params = params.with_fallback_cluster_shape_mnk(f);
                        }
                        StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
                    }
                });
            })
        })
    };
    for clc in [true, false] {
        assert!(build(true, clc).is_ok(), "clc {clc}");
        let e = build(false, clc).unwrap_err();
        let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(e.contains("[2, 4, 1]") && e.contains("[2, 1, 1]") && e.contains("arm"), "clc {clc}: {e}");
    }
    let e = std::panic::catch_unwind(|| {
        PersistentTileSchedulerParams::new(MIXED_TILES, PREFERRED, Default::default())
            .with_fallback_cluster_shape_mnk([2, 3, 1])
    })
    .unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("does not divide"), "{e}");
}

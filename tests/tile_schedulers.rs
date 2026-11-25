//! The static persistent tile schedulers (roadmap part
//! m9-persistent-schedulers) on the simulator: every CTA of a persistent
//! grid records the tiles it visits, for many schedules, and the record is
//! held to a plain-Rust model; the runtime scheduler stops at a runtime
//! bound; the grid sizing and the swizzled layouts match CuTe DSL's rules.

#[allow(dead_code)]
#[path = "../examples/persistent_gemm.rs"]
mod persistent_gemm;
#[allow(dead_code)]
mod pipeline_kernels;

use cute::prelude::*;
use cute::utils::{PersistentTileSchedulerOptions, PersistentTileSchedulerParams, swizzle_and_rasterize};
use pipeline_kernels::schedulers::*;

/// Every combination of tile counts, cluster shapes, swizzles (1, 2, 4),
/// raster orders and persistent grid sizes, with the schedule built from
/// static tile counts, from host-passed parameters and from runtime tile
/// counts: each real tile is visited once, by the CTA and in the order the
/// model says, and the swizzle's padding tiles count as valid.
#[test]
fn schedulers_visit_every_tile_once() {
    let mut padding = 0;
    let mut runs = 0;
    for tiles in [[5, 3, 1], [7, 4, 2], [3, 9, 1], [1, 1, 1], [8, 6, 1]] {
        for cluster in [[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1]] {
            for swizzle in [1, 2, 4] {
                for raster_along_m in [true, false] {
                    for max_active_clusters in [1, 3, 7, 1000] {
                        let s = TileSchedule { tiles, cluster, swizzle, raster_along_m, max_active_clusters };
                        for from in [ParamsFrom::Static, ParamsFrom::Host, ParamsFrom::Runtime] {
                            let kernel = enumeration_kernel(&s, from);
                            let (cfg, mut args) = enumeration_launch(&s, from);
                            kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{s:?} {from:?}: {e}"));
                            padding += check_enumeration(&s, &args);
                            runs += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(runs > 1000 && padding > 0, "{runs} runs, {padding} padding tiles");
}

/// The unswizzled schedule decodes with FastDivmod (no division in the
/// kernel for host-passed or static divisors); the swizzled one with
/// `get_flat_coord`.
#[test]
fn unswizzled_decode_uses_fast_divmod() {
    let divisions = |s: &TileSchedule, from: ParamsFrom| {
        let kernel = enumeration_kernel(s, from);
        kernel.to_mlir().matches("arith.divui").count() + kernel.to_mlir().matches("arith.divsi").count()
    };
    let plain =
        TileSchedule { tiles: [5, 3, 1], cluster: [1, 1, 1], swizzle: 1, raster_along_m: true, max_active_clusters: 3 };
    assert_eq!(divisions(&plain, ParamsFrom::Host), 0, "{}", enumeration_kernel(&plain, ParamsFrom::Host).to_mlir());
    assert_eq!(divisions(&plain, ParamsFrom::Static), 0);
    let mlir = enumeration_kernel(&plain, ParamsFrom::Host).to_mlir();
    assert!(
        mlir.contains("mulhi") || mlir.contains("mul.hi") || mlir.contains("umulh") || mlir.contains("mului_extended"),
        "{mlir}"
    );
}

/// Schedules decoded through the problem layout as CuTe DSL builds it: the
/// unswizzled layout is column-major over the cluster counts, the swizzled
/// ones are `(M, (swz, N/swz), L):(swz, (1, swz*M), M*N)` and
/// `((swz, M/swz), N, L):((1, swz*N), swz, M*N)` over the padded counts.
#[test]
fn swizzled_layouts_follow_cutlass() {
    let l = swizzle_and_rasterize(&Shape::from((3usize, 5usize, 2usize)), 1, false);
    assert_eq!(format!("{l}"), "(3,5,2):(1,3,15)");
    let l = swizzle_and_rasterize(&Shape::from((3usize, 5usize, 2usize)), 2, true);
    assert_eq!(format!("{l}"), "(3,(2,3),2):(2,(1,6),18)");
    let l = swizzle_and_rasterize(&Shape::from((3usize, 5usize, 2usize)), 4, false);
    assert_eq!(format!("{l}"), "((4,1),5,2):((1,20),4,20)");
    let p = PersistentTileSchedulerParams::new(
        (7usize, 9usize, 1usize),
        [2, 1, 1],
        PersistentTileSchedulerOptions { swizzle_size: 2, raster_along_m: true },
    );
    assert_eq!(format!("{}", p.problem_layout_ncluster_mnl()), "(4,(2,5),1):(2,(1,8),40)");
}

/// `get_grid_shape`: the cluster shape and the fewer of the problem's
/// clusters (padding included) and the clusters that fit.
#[test]
fn grid_shape_is_capped_by_the_problem() {
    let p = |tiles: (usize, usize, usize), cluster, swizzle| {
        PersistentTileSchedulerParams::new(
            tiles,
            cluster,
            PersistentTileSchedulerOptions { swizzle_size: swizzle, raster_along_m: true },
        )
    };
    assert_eq!(p((5, 3, 1), [1, 1, 1], 1).get_grid_shape(4), [1, 1, 4]);
    assert_eq!(p((5, 3, 1), [1, 1, 1], 1).get_grid_shape(100), [1, 1, 15]);
    assert_eq!(p((5, 3, 2), [2, 1, 1], 1).get_grid_shape(100), [2, 1, 18]);
    assert_eq!(p((5, 3, 1), [1, 1, 1], 4).get_grid_shape(100), [1, 1, 20]);
    assert_eq!(p((4, 4, 1), [2, 2, 1], 1).get_grid_shape(3), [2, 2, 3]);
    assert_eq!(p((4, 4, 1), [2, 2, 1], 1).get_grid_shape(0), [2, 2, 0]);
}

/// The runtime scheduler visits work indices `c, c + ctas, ...` decoded
/// with N or M fastest, and the kernel stops at a runtime tile bound.
#[test]
fn runtime_scheduler_stops_at_a_runtime_bound() {
    for inner_mode in [0, 1] {
        for (ctas, tiles, limit) in [(3, [4, 5], 3), (4, [6, 2], 5), (1, [2, 3], 2), (5, [3, 3], 0)] {
            let kernel = runtime_scheduler_kernel(ctas, inner_mode);
            let (cfg, mut args) = runtime_scheduler_launch(ctas, tiles, limit);
            kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
            check_runtime_scheduler(ctas, tiles, limit, inner_mode, &args);
        }
    }
}

/// `examples/persistent_gemm.rs`: one kernel per schedule (unswizzled,
/// swizzled along M and along N) for every problem size, with fewer CTAs
/// than tiles, bit-exact against the CPU GEMM on small integers; one run
/// under seeded schedules too.
#[test]
fn persistent_gemm_matches_the_cpu() {
    for options in persistent_gemm::variants() {
        let kernel = persistent_gemm::build(options);
        for (m, n, k, ctas) in [(256, 384, 64, 4), (384, 256, 96, 5), (128, 128, 32, 3), (512, 128, 32, 3)] {
            let (launch, mut args, a, b) = persistent_gemm::problem(m, n, k, ctas, options);
            assert!(launch.grid[2] < ((m / 128) * (n / 128)) as u32 || ctas as usize >= (m / 128) * (n / 128));
            kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{options:?} {m}x{n}x{k}: {e}"));
            persistent_gemm::check(m, n, k, &a, &b, args[2].as_f32());
        }
    }
    let options = persistent_gemm::variants()[1];
    let kernel = persistent_gemm::build(options);
    let (launch, mut args, a, b) = persistent_gemm::problem(256, 256, 64, 3, options);
    kernel.simulate_schedules(launch, &mut args, &SimOptions::default(), &[7, 8]).unwrap_or_else(|e| panic!("{e}"));
    persistent_gemm::check(256, 256, 64, &a, &b, args[2].as_f32());
}

/// A problem with no tiles along a mode has no work: every CTA's first tile
/// is invalid, whether the divisors are constants, host-passed or made in
/// the kernel (where a cluster count of 0 must not become a division by
/// zero).
#[test]
fn empty_problems_have_no_work() {
    use cute::utils::StaticPersistentTileScheduler;
    for tiles in [[0usize, 3, 1], [4, 0, 1], [2, 2, 0]] {
        for raster_along_m in [true, false] {
            let options = PersistentTileSchedulerOptions { swizzle_size: 1, raster_along_m };
            for from in 0..3 {
                let kernel = Kernel::builder("empty_problem").block_dim([32, 1, 1]).build(|| {
                    let out = param_tensor::<i32>("out", Layout::col_major(2));
                    let params = match from {
                        0 => PersistentTileSchedulerParams::new((tiles[0], tiles[1], tiles[2]), [1, 1, 1], options),
                        1 => PersistentTileSchedulerParams::param("sched", [1, 1, 1], options),
                        _ => {
                            let e = |n: &str| IndexExpr::from(param_scalar::<i32>(n));
                            let shape = Shape::from((e("tiles_m"), e("tiles_n"), e("tiles_l")));
                            PersistentTileSchedulerParams::new(shape, [1, 1, 1], options)
                        }
                    };
                    let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
                    let valid = sched.initial_work_tile_info().is_valid_tile();
                    if_(thread_idx().x.eq(0), || out.store(block_idx().z, valid.to::<i32>()));
                });
                let mut args = vec![KernelArg::buffer(&[-1i32; 2])];
                match from {
                    1 => args.extend(
                        PersistentTileSchedulerParams::new((tiles[0], tiles[1], tiles[2]), [1, 1, 1], options)
                            .kernel_args(),
                    ),
                    2 => args.extend(tiles.map(|t| KernelArg::scalar(t as i32))),
                    _ => {}
                }
                kernel
                    .simulate(LaunchConfig::new([1, 1, 2], [32, 1, 1]), &mut args)
                    .unwrap_or_else(|e| panic!("{tiles:?} from {from}: {e}"));
                assert_eq!(args[0].to_vec::<i32>(), [0, 0], "{tiles:?} from {from}");
            }
        }
    }
}

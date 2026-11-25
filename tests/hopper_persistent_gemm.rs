// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hopper persistent GEMM port:
//! the simulator against a plain Rust GEMM, with grids that have fewer
//! clusters than tiles, swizzled and rastered schedules, every cluster shape,
//! partial tiles, several scheduler seeds (the DMA warp runs ahead of the MMA
//! warpgroups) with the race detector on, and the mistake the diagnostics
//! show.

#[allow(dead_code)]
#[path = "../examples/hopper_persistent_gemm.rs"]
mod hopper_persistent_gemm;

use cute::prelude::*;
use hopper_persistent_gemm::{HopperWgmmaGemmPersistentKernel, Majors, Problem};

fn f16(
    majors: Majors,
    tile_mn: (usize, usize),
    cluster_mn: (usize, usize),
    swizzle: usize,
    raster_along_m: bool,
) -> HopperWgmmaGemmPersistentKernel {
    HopperWgmmaGemmPersistentKernel::new(DType::F32, tile_mn, cluster_mn, swizzle, raster_along_m).setup(
        DType::F16,
        DType::F16,
        DType::F16,
        majors,
    )
}

/// A problem `(m, n, k, l)` and the most clusters resident.
type Case = ((usize, usize, usize, usize), u32);

/// Build once, then simulate and check each (problem, clusters) pair.
fn check_all(gemm: &HopperWgmmaGemmPersistentKernel, problems: &[Case]) {
    let kernel = gemm.build();
    for &(mnkl, clusters) in problems {
        let (m, n, k, l) = mnkl;
        gemm.run(&kernel, &Problem::new(m, n, k, l), clusters);
    }
}

/// The configuration: a DMA warpgroup besides the MMA ones, the store warp
/// the first MMA warp, and the epilogue's own shared memory.
#[test]
fn configuration() {
    let g = f16(Majors::KKN, (128, 128), (1, 1), 1, true);
    assert_eq!((g.threads_per_cta, g.load_warp_id, g.epi_store_warp_id, g.num_mma_threads), (256, 0, 4, 128));
    assert_eq!((g.load_register_requirement, g.mma_register_requirement), (40, 248));
    let two = f16(Majors::KKN, (128, 256), (1, 1), 1, true);
    assert_eq!((two.num_mma_warp_groups, two.threads_per_cta, two.base.epi_tile), (2, 384, (128, 32)));
    assert_eq!(two.mma_register_requirement, 232);
    // 64x128x64 A/B stages of 32 KB, minus the mbarriers' 1 KB and the 4 tiles of C.
    let dense = HopperWgmmaGemmPersistentKernel::compute_stages(
        (128, 128, 64),
        DType::F16,
        DType::F16,
        (64, 32),
        DType::F16,
        232448,
        1,
    );
    assert_eq!(dense, (6, 4));
    assert_eq!(g.base.ab_stage, 6);
    // 8 tiles (the 2x4 problem), 3 clusters of 2 at most: the grid has the clusters of the problem or fewer.
    let g = f16(Majors::KKN, (64, 64), (2, 1), 1, true);
    assert_eq!(g.launch(128, 256, 1, 3).grid, [2, 1, 3]);
    assert_eq!(g.launch(128, 256, 1, 100).grid, [2, 1, 4]);
}

/// Grids with fewer clusters than tiles, in each swizzle and raster order,
/// on whole and partial tiles.
#[test]
fn fewer_clusters_than_tiles() {
    for (swizzle, along_m) in [(1, true), (1, false), (2, true), (2, false)] {
        let g = f16(Majors::KKN, (64, 64), (1, 1), swizzle, along_m);
        check_all(&g, &[((192, 128, 128, 1), 2), ((136, 200, 72, 1), 3), ((128, 64, 192, 2), 1)]);
    }
}

/// Every cluster shape, with multicast along M, N or both, several tiles per
/// cluster and tiles that cross the edge of the problem.
#[test]
fn clusters() {
    check_all(&f16(Majors::KKN, (64, 64), (2, 1), 1, true), &[((256, 128, 128, 1), 2), ((200, 72, 64, 1), 1)]);
    check_all(&f16(Majors::KKN, (64, 64), (1, 2), 2, false), &[((128, 256, 128, 1), 2), ((72, 136, 136, 1), 2)]);
    check_all(&f16(Majors::new('m', 'n', 'm'), (64, 64), (2, 2), 2, true), &[((192, 192, 64, 1), 2)]);
    check_all(&f16(Majors::KKN, (128, 128), (2, 1), 1, false), &[((384, 200, 128, 1), 2)]);
}

/// The scheduler's seeds: the DMA warp loads ahead while the MMA warpgroups
/// work, and the race detector stays quiet.
#[test]
fn random_schedules() {
    let g = f16(Majors::KKN, (64, 64), (2, 1), 1, true);
    let kernel = g.build();
    for seed in [1, 2, 3, 7, 11] {
        let opts = SimOptions { schedule: Schedule::Random { seed }, detect_races: true, ..SimOptions::default() };
        g.run_with(&kernel, &Problem::new(256, 128, 192, 1), 2, &opts);
    }
    let g = f16(Majors::KKN, (64, 64), (1, 1), 2, false);
    let kernel = g.build();
    for seed in [5, 13] {
        let opts = SimOptions { schedule: Schedule::Random { seed }, detect_races: true, ..SimOptions::default() };
        g.run_with(&kernel, &Problem::new(128, 192, 128, 1), 2, &opts);
    }
}

/// Two MMA warpgroups (tile N above 128) and BFloat16.
#[test]
fn two_warpgroups_and_bfloat16() {
    check_all(&f16(Majors::KKN, (128, 256), (1, 1), 1, true), &[((128, 512, 64, 1), 1), ((200, 264, 72, 1), 2)]);
    let bf16 = HopperWgmmaGemmPersistentKernel::new(DType::F32, (64, 128), (1, 2), 1, true).setup(
        DType::BF16,
        DType::BF16,
        DType::BF16,
        Majors::KKN,
    );
    check_all(&bf16, &[((128, 256, 64, 1), 1)]);
}

/// Restarting the MMA warpgroups' pipeline states on every tile (the
/// mistake) puts them out of phase with the loads: the simulator does not
/// let it through.
#[test]
fn resetting_the_consumer_states_per_tile_is_reported() {
    let mut g = f16(Majors::KKN, (64, 64), (1, 1), 1, true);
    g.base.ab_stage = 2;
    g.reset_states_per_tile = true;
    let kernel = g.build();
    // Three k-tiles per tile on two stages: the second tile starts mid-ring.
    let problem = Problem::new(128, 64, 64 * 3, 1);
    let mut args = g.inputs(&problem);
    let e = kernel.simulate(g.launch(128, 64, 1, 1), &mut args).expect_err("an error").to_string();
    assert!(e.contains("mbarrier.try_wait with parity") && e.contains("has not seen phase"), "{e}");
}

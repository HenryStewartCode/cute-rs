//! The grouped GEMM tile scheduler and the contiguous group search
//! (roadmap part m9-grouped-gemm-schedulers) on the simulator, against
//! plain-Rust models.

#[allow(dead_code)]
mod grouped_kernels;

use cute::prelude::*;
use grouped_kernels::*;

/// Every problem (1 to 70 groups, empty groups, clusters of 1 and 2 CTAs,
/// 1 to 7 persistent clusters) in each search mode (direct loads, cached
/// shapes, the deprecated helper): every CTA tile of every group visited
/// once with the right group, coordinates, shape and K tiles.
#[test]
fn grouped_scheduler_visits_every_tile_once() {
    for p in grouped_problems() {
        for mode in SearchMode::ALL {
            let kernel = grouped_enumeration_kernel(&p, mode);
            let (cfg, mut args) = grouped_enumeration_launch(&p);
            kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{mode:?} {p:?}: {e}"));
            check_grouped_enumeration(&p, &args);
        }
    }
}

/// The cached and direct searches under seeded schedules give the same
/// records.
#[test]
fn grouped_scheduler_under_seeded_schedules() {
    let p = GroupedProblem::random(21, 45, true, [1, 1], 3);
    for mode in [SearchMode::Direct, SearchMode::Cached] {
        let kernel = grouped_enumeration_kernel(&p, mode);
        let (cfg, mut args) = grouped_enumeration_launch(&p);
        kernel
            .simulate_schedules(cfg, &mut args, &SimOptions::default(), &[1, 2, 3, 4])
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        check_grouped_enumeration(&p, &args);
    }
}

/// The contiguous search over random boundaries (empty groups included),
/// each thread carrying its state over an increasing sequence of tiles:
/// the transcription's states, and the tiles of a direct reading.
#[test]
fn contiguous_search_matches_the_transcription() {
    for p in contiguous_problems() {
        let kernel = contiguous_search_kernel(&p);
        let (cfg, mut args) = contiguous_search_launch(&p);
        kernel.simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        check_contiguous_search(&p, &args);
    }
}

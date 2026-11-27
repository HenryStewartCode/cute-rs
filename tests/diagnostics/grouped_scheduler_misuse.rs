//! The grouped-GEMM tile scheduler: misuse diagnostics.

use super::{build_panic, simulate_err};
use cute::prelude::*;
use cute::utils::{PersistentTileSchedulerParams, StaticPersistentGroupTileScheduler, create_initial_search_state};

/// The grouped search is warp-collective: called from half a warp, its
/// shuffles read lanes that did not take part, which the simulator
/// reports. Every lane of the warp must call it.
#[test]
fn grouped_search_from_part_of_a_warp_is_reported() {
    let kernel = Kernel::builder("half_warp_search").block_dim([32, 1, 1]).build(|| {
        let shapes = param_tensor::<i32>("problem_shape_mnkl", Layout::row_major((3usize, 4usize)));
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let params = PersistentTileSchedulerParams::new((1usize, 1usize, 4usize), [1, 1, 1], Default::default());
        if_(lane_idx().lt(16), || {
            let mut sched = StaticPersistentGroupTileScheduler::create(
                &params,
                block_idx(),
                grid_dim(),
                [16, 16, 16],
                create_initial_search_state(),
                3,
                &shapes,
                false,
            );
            let work = sched.get_current_work();
            out.store(lane_idx(), work.group_search_result().group_idx);
        });
    });
    let shapes = [16, 16, 16, 1, 16, 32, 16, 1, 32, 16, 16, 1];
    let mut args = vec![KernelArg::buffer(&shapes), KernelArg::buffer(&[0i32; 32])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(
        e.to_string().contains("tests/diagnostics/grouped_scheduler_misuse.rs")
            && e.message.contains("shuffle with member mask 0xffffffff is executed by lane 0 of warp 0, but lane 16")
            && e.message.contains("every lane of the mask must execute it"),
        "{e}"
    );
}

/// No groups, a cluster tile with a 0 extent and a problem-shape tensor
/// that is not `(group_count, 4)` are refused while the kernel is
/// built.
#[test]
fn grouped_scheduler_refuses_bad_arguments() {
    let make = |groups: usize, tile: [usize; 3], rows: usize, cols: usize| {
        build_panic(move || {
            Kernel::builder("bad_grouped").block_dim([32, 1, 1]).build(|| {
                let shapes = param_tensor::<i32>("shapes", Layout::row_major((rows, cols)));
                let params =
                    PersistentTileSchedulerParams::new((1usize, 1usize, 4usize), [1, 1, 1], Default::default());
                StaticPersistentGroupTileScheduler::create(
                    &params,
                    block_idx(),
                    grid_dim(),
                    tile,
                    create_initial_search_state(),
                    groups,
                    &shapes,
                    true,
                );
            });
        })
    };
    let msg = make(0, [16, 16, 16], 1, 4);
    assert!(msg.contains("a grouped GEMM needs at least one group, not 0"), "{msg}");
    let msg = make(2, [16, 0, 16], 2, 4);
    assert!(msg.contains("the cluster tile [16, 0, 16] has a 0 extent"), "{msg}");
    let msg = make(2, [16, 16, 16], 2, 3);
    assert!(msg.contains("the problem shapes must be a (group_count, 4) = (2, 4) tensor"), "{msg}");
    let msg = make(3, [16, 16, 16], 2, 4);
    assert!(msg.contains("the problem shapes must be a (group_count, 4) = (3, 4) tensor"), "{msg}");
}

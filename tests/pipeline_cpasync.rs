//! `PipelineCpAsync` and `PipelineOrder` (roadmap part
//! m9-pipeline-cp-async-order) on the simulator: the `cp.async` rings and
//! the turn logs under seeded schedules with random completion of the
//! copies, the barriers each makes, and what the SM89 and SM90 builds emit.

mod feature_kernels;

use cute::ir::Op;
use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions, PipelineOrder};
use cute::prelude::*;

/// Every ring and log under the built-in schedules and 16 seeded ones
/// (random schedules and random `cp.async` completion), on small and
/// large inputs: each matches plain Rust.
#[test]
fn rings_and_logs_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..16).collect();
    for case in feature_kernels::cpasync_order_cases_sm89() {
        for small in [true, false] {
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

/// A `PipelineCpAsync` of 3 stages between two producer warps and one
/// consumer warp: the full barriers count the 64 producer threads and the
/// empty ones the 32 consumer threads; a commit is a `.noinc`
/// `cp.async.mbarrier.arrive`, a release an `mbarrier.arrive`.
#[test]
fn pipeline_cp_async_barriers() {
    let kernel = Kernel::builder("cp_async_barriers").block_dim([96, 1, 1]).target(Arch::Sm89).build(|| {
        let p = CooperativeGroup::new(Agent::Warp, 2);
        let c = CooperativeGroup::new(Agent::Thread, 32);
        let pipeline = PipelineCpAsync::create(3, p, c, PipelineCpAsyncOptions::default());
        assert_eq!(pipeline.sync_object_full().arrive_count().as_static(), Some(64));
        assert_eq!(pipeline.sync_object_empty().arrive_count().as_static(), Some(32));
        assert_eq!(pipeline.sync_object_full().op(), cute::pipeline::PipelineOp::AsyncLoad);
        assert_eq!(pipeline.sync_object_empty().op(), cute::pipeline::PipelineOp::AsyncThread);
        let (producer, consumer) = pipeline.make_participants();
        producer.acquire(None).commit();
        consumer.wait(None).release();
    });
    let noinc = kernel.ir().count_ops(|op| matches!(op, Op::CpAsyncMbarrierArrive { noinc: true, .. }));
    assert_eq!(noinc, 1, "one .noinc arrival per commit");
    let mlir = kernel.to_mlir();
    assert!(!mlir.contains("try_wait"), "an sm_89 build spins on test_wait");
}

/// `PipelineOrder` of depth 2 and length 3 for group 1: the barrier of
/// group `g` at stage `i` is `i * 3 + g`, the group waits on its own and
/// arrives on the next group's (2), group 1's state starts at phase 0,
/// and an explicit state comes back advanced.
#[test]
fn pipeline_order_barrier_indices() {
    let kernel = Kernel::builder("order_indices").block_dim([32, 1, 1]).target(Arch::Sm89).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(6));
        let group = CooperativeGroup::new(Agent::Warp, 1);
        let order = PipelineOrder::create(2, 3, 1, group, Default::default());
        assert_eq!((order.depth(), order.length()), (2, 3));
        assert_eq!(order.sync_object_full().num_stages(), 6);
        assert_eq!(order.state().phase().as_constant(), Some(0));
        let mut s = order.state();
        s.advance();
        let next = order.arrive_with(s);
        let first = PipelineOrder::create(2, 3, 0, group, Default::default());
        assert_eq!(first.state().phase().as_constant(), Some(1));
        if_(lane_idx().eq(0), || {
            out.store(0, order.get_barrier_for_current_stage_idx(2, None));
            out.store(1, order.get_barrier_for_current_stage_idx(1, Some(&s)));
            out.store(2, next.index());
            out.store(3, next.phase());
            out.store(4, next.count());
            out.store(5, order.state().count());
        });
    });
    let mut args = vec![KernelArg::buffer(&[-1i32; 6])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<i32>(), [2, 4, 0, 1, 2, 0]);
}

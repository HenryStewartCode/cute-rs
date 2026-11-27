//! Software-pipelined loops (range prefetch): misuse diagnostics.

use super::{build_panic, simulate_err};
use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions};
use cute::prelude::*;

/// One warp both produces into and consumes from a 2-stage
/// `PipelineCpAsync` over 6 tiles, `ahead` tiles apart (bounded by the
/// stage count when `bounded`).
fn single_warp(ahead: i64, bounded: bool) -> Kernel {
    Kernel::builder("prefetch_misuse").block_dim([32, 1, 1]).target(Arch::Sm89).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(6 * 128));
        let out = param_tensor::<u32>("out", Layout::col_major(6 * 128));
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let pipeline = PipelineCpAsync::create(2, warp, warp, PipelineCpAsyncOptions::default());
        let data = make_smem_tensor::<u32>(Layout::col_major(2 * 128));
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let t = thread_idx().x;
        let prefetch = Prefetch::stages(ahead);
        let prefetch = if bounded { prefetch.bounded_by(pipeline.num_stages()) } else { prefetch };
        for_range_pipelined(
            0,
            6,
            1,
            prefetch,
            pipeline.make_participants(),
            |k, mut p| {
                let h = p.acquire_and_advance(None);
                let from = make_tensor(&src.ptr().offset((k * 32 + t) * 4), Layout::col_major(4));
                let to = make_tensor(&data.ptr().offset((h.index() * 32 + t) * 4), Layout::col_major(4));
                copy(&atom, &from, &to);
                h.commit();
                p
            },
            |k, mut c| {
                let h = c.wait_and_advance(None);
                for j in 0..4 {
                    out.store(k * 128 + t + j * 32, data.load(h.index() * 128 + t + j * 32));
                }
                h.release();
                c
            },
        );
    })
}

#[test]
fn negative_prefetch_stages() {
    let e = build_panic(|| {
        single_warp(-1, false);
    });
    assert!(e.contains("prefetch stages must be a static integer >= 0, got -1"), "{e}");
}

#[test]
fn prefetch_stages_beyond_the_pipeline() {
    let e = build_panic(|| {
        single_warp(2, true);
    });
    assert!(e.contains("prefetch stages 2 with a 2-stage pipeline") && e.contains("at most 1 stages"), "{e}");
}

/// Without the bound, a distance of the stage count makes the warp wait
/// for a release it makes later: the simulator reports the hang (the
/// sm_89 wait spins on `mbarrier.test_wait`), in the steady-state loop.
#[test]
fn prefetch_stages_beyond_the_pipeline_hang_unbounded() {
    let kernel = single_warp(2, false);
    let mut args = [KernelArg::buffer(&[1u32; 6 * 128]), KernelArg::buffer(&[0u32; 6 * 128])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("did not terminate") && e.message.contains("a hang"), "{e}");
    assert!(e.message.contains("for_range at tests/diagnostics/range_prefetch_misuse.rs"), "{e}");
}

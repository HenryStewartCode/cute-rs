//! `PipelineAsync`, its participants, try-wait tokens and `PipelineEmpty`
//! (roadmap part m9-pipeline-async) on the simulator: the warp-specialised
//! hand-offs under seeded schedules, both paths of the token-taking waits,
//! and one generic kernel body over a real and an empty pipeline.

mod feature_kernels;
#[allow(dead_code)]
mod pipeline_kernels;

use cute::ir::Op;
use cute::prelude::*;
use pipeline_kernels::*;

/// Every `PipelineAsync` feature kernel (sm_89 hand-offs in three styles,
/// the sm_90 cluster masks) under the built-in schedules and 12 seeded
/// ones, with the race detector on: each matches plain Rust.
#[test]
fn handoffs_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..12).collect();
    let cases = feature_kernels::async_sched_cases_sm89().into_iter().chain(feature_kernels::async_sched_cases_sm90());
    for case in cases {
        for small in [true, false] {
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

/// The token kernel under the built-in schedules and 24 seeded ones: the
/// data is always right, and over all the runs each side's probe both
/// succeeded (the wait skipped) and failed (the fallback wait ran).
#[test]
fn try_wait_tokens_take_both_paths() {
    let kernel = token_kernel();
    let mut totals = [0u32; 4];
    let mut runs = Vec::new();
    for schedule in Schedule::ALL {
        runs.push(SimOptions { schedule, ..SimOptions::default() });
    }
    for seed in 0..24 {
        runs.push(SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        });
    }
    for opts in runs {
        let mut args = token_args();
        kernel.simulate_with(token_cfg(), &mut args, &opts).unwrap_or_else(|e| panic!("{:?}: {e}", opts.schedule));
        let counts = check_token_run(&args);
        for (t, c) in totals.iter_mut().zip(counts) {
            *t += c;
        }
    }
    let [p_true, p_false, c_true, c_false] = totals;
    assert!(p_true > 0 && p_false > 0, "producer tokens: {p_true} true, {p_false} false");
    assert!(c_true > 0 && c_false > 0, "consumer tokens: {c_true} true, {c_false} false");
}

/// A constant token decides at build time: true skips the wait, false
/// waits unconditionally (no branch either way).
#[test]
fn constant_tokens_fold() {
    use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
    let waits = |token: Option<bool>| {
        let kernel = Kernel::builder("constant_token").block_dim([32, 1, 1]).target(Arch::Sm89).build(|| {
            let warp = CooperativeGroup::new(Agent::Warp, 1);
            let pipeline = PipelineAsync::create(2, warp, warp, PipelineAsyncOptions::default());
            let p = pipeline.make_producer();
            p.acquire(token.map(Boolean::from));
        });
        let tests = kernel.ir().count_ops(|op| matches!(op, Op::MbarrierWait { .. }));
        let branches = kernel.ir().count_ops(|op| matches!(op, Op::If { .. }));
        (tests, branches)
    };
    let (unconditional, branches) = waits(None);
    assert!(unconditional > 0);
    assert_eq!(waits(Some(false)), (unconditional, branches));
    assert_eq!(waits(Some(true)).0, 0);
}

/// One generic body over `PipelineAsync` and `PipelineEmpty`: the same
/// results, and no mbarrier operation at all in the empty one.
#[test]
fn pipeline_empty_serves_the_same_body() {
    let mbarrier_ops = |k: &Kernel| {
        k.ir().count_ops(|op| {
            matches!(
                op,
                Op::MbarrierInit { .. }
                    | Op::MbarrierArrive { .. }
                    | Op::MbarrierWait { .. }
                    | Op::MbarrierExpectTx { .. }
            )
        })
    };
    let mut results = Vec::new();
    for empty in [false, true] {
        let kernel = generic_kernel(empty);
        let ops = mbarrier_ops(&kernel);
        assert_eq!(ops == 0, empty, "{} has {ops} mbarrier operations", kernel.name());
        let mut args = generic_args();
        kernel
            .simulate_schedules(generic_cfg(), &mut args, &SimOptions::default(), &[1, 2, 3])
            .unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
        check_generic(&args);
        results.push(args[1].to_vec::<u32>());
    }
    assert_eq!(results[0], results[1]);
}

//! `PipelineAsync` and its participants,
//! held to plain Rust. The hand-offs are built for sm_89, so that their
//! waits spin on `mbarrier.test_wait` and the SM89 GPU runs them; the
//! cluster one, whose arrivals go to a peer CTA, is sm_90.

use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineState, agent_sync,
    pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;

use super::Case;

/// Stages of the [`pipeline_async_handoff`] ring.
pub const PA_STAGES: usize = 4;
/// Items handed over by [`pipeline_async_handoff`]: the ring wraps nine
/// times, so the phases flip often.
pub const PA_ITEMS: usize = 37;

/// How [`pipeline_async_handoff`] drives the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffStyle {
    /// `make_participants`, `acquire_and_advance` with `commit` on the
    /// handle, `wait_and_advance` with `release` on the handle, and the
    /// producer's `tail`.
    Participants,
    /// Explicit `PipelineState`s and the pipeline's own methods
    /// (`producer_acquire(state)` and so on), and `producer_tail`.
    States,
    /// `acquire` and `wait` for a handle, `advance`, then the participant's
    /// `commit(Some(&handle))` and `release(Some(&handle))`.
    Handles,
}

/// Warp 0 produces and warp 1 consumes [`PA_ITEMS`] items through a
/// [`PA_STAGES`]-stage `PipelineAsync` (a warp on each side) in reserved
/// storage, in the given style. The producer writes `src[k][lane] + k` into
/// its stage; the consumer sums the word `(lane + k) % 32` of the stage.
/// Lane 0 of each warp records the index, phase and count of every stage it
/// handled.
pub fn pipeline_async_handoff(style: HandoffStyle) -> Case {
    pipeline_async_handoff_for(style, Some(Arch::Sm89))
}

/// [`pipeline_async_handoff`] built for `target`; with `None`, the kernel
/// waits as CuTe DSL's pipelines do (`mbarrier.try_wait`, sm_90).
pub fn pipeline_async_handoff_for(style: HandoffStyle, target: Option<Arch>) -> Case {
    let name = match style {
        HandoffStyle::Participants => "pipeline_async_participants",
        HandoffStyle::States => "pipeline_async_states",
        HandoffStyle::Handles => "pipeline_async_handles",
    };
    let builder = Kernel::builder(name).block_dim([64, 1, 1]);
    let builder = match target {
        Some(arch) => builder.target(arch),
        None => builder,
    };
    let kernel = builder.build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(PA_ITEMS * 32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let trace = param_tensor::<i32>("trace", Layout::row_major((2 * PA_ITEMS, 3)));
        let lane = lane_idx();
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let pipeline = PipelineAsync::create(PA_STAGES, warp, warp, PipelineAsyncOptions::default());
        let data = make_smem_tensor::<u32>(Layout::col_major(PA_STAGES * 32));
        let record = |row: Int32, s: &PipelineState| {
            if_(lane.eq(0), || {
                trace.store((row, 0), s.index());
                trace.store((row, 1), s.phase());
                trace.store((row, 2), s.count());
            });
        };
        let fill = |k: Int32, stage: Int32| data.store(stage * 32 + lane, src.load(k * 32 + lane) + k.to::<u32>());
        let drain = |k: Int32, stage: Int32| data.load(stage * 32 + (lane + k) % 32);
        let items = PA_ITEMS as i32;
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || match style {
                HandoffStyle::Participants => {
                    let mut p = for_range_carry(0, items, 1, producer, |k, mut p| {
                        record(k, &p.state());
                        let handle = p.acquire_and_advance(None);
                        fill(k, handle.index());
                        handle.commit();
                        p
                    });
                    p.tail();
                }
                HandoffStyle::States => {
                    let start = pipeline.make_producer().state();
                    let mut s = for_range_carry(0, items, 1, start, |k, mut s| {
                        record(k, &s);
                        pipeline.producer_acquire(&s, None);
                        fill(k, s.index());
                        pipeline.producer_commit(&s);
                        s.advance();
                        s
                    });
                    pipeline.producer_tail(&mut s);
                }
                HandoffStyle::Handles => {
                    let mut p = for_range_carry(0, items, 1, producer, |k, mut p| {
                        record(k, &p.state());
                        let handle = p.acquire(None);
                        p.advance(1);
                        fill(k, handle.index());
                        p.commit(Some(&handle));
                        p
                    });
                    p.tail();
                }
            },
            || {
                let sum = match style {
                    HandoffStyle::Participants => {
                        let start = (Uint32::from(0u32), consumer);
                        for_range_carry(0, items, 1, start, |k, (sum, mut c)| {
                            record(k + items, &c.state());
                            let handle = c.wait_and_advance(None);
                            let word = drain(k, handle.index());
                            handle.release();
                            (sum + word, c)
                        })
                        .0
                    }
                    HandoffStyle::States => {
                        let start = (Uint32::from(0u32), consumer.state());
                        for_range_carry(0, items, 1, start, |k, (sum, mut s)| {
                            record(k + items, &s);
                            pipeline.consumer_wait(&s, None);
                            let word = drain(k, s.index());
                            pipeline.consumer_release(&s);
                            s.advance();
                            (sum + word, s)
                        })
                        .0
                    }
                    HandoffStyle::Handles => {
                        let start = (Uint32::from(0u32), consumer);
                        for_range_carry(0, items, 1, start, |k, (sum, mut c)| {
                            record(k + items, &c.state());
                            let handle = c.wait(None);
                            c.advance(1);
                            let word = drain(k, handle.index());
                            c.release(Some(&handle));
                            (sum + word, c)
                        })
                        .0
                    }
                };
                out.store(lane, sum);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, out, trace) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<i32>());
        for (l, &got) in out.iter().enumerate() {
            let want =
                (0..PA_ITEMS).fold(0u32, |acc, k| acc.wrapping_add(src[k * 32 + (l + k) % 32].wrapping_add(k as u32)));
            assert_eq!(got, want, "lane {l}");
        }
        for k in 0..PA_ITEMS {
            let (index, wraps) = ((k % PA_STAGES) as i32, (k / PA_STAGES) as i32);
            assert_eq!(trace[k * 3..k * 3 + 3], [index, 1 ^ (wraps & 1), k as i32], "producer item {k}");
            let c = (PA_ITEMS + k) * 3;
            assert_eq!(trace[c..c + 3], [index, wraps & 1, k as i32], "consumer item {k}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

/// The `PipelineAsync` kernels built for sm_89, which the SM89 GPU runs.
pub fn async_sched_cases_sm89() -> Vec<Case> {
    vec![
        pipeline_async_handoff(HandoffStyle::Participants),
        pipeline_async_handoff(HandoffStyle::States),
        pipeline_async_handoff(HandoffStyle::Handles),
    ]
}

/// Rounds of [`pipeline_async_cluster_masks`].
pub const PM_ROUNDS: usize = 9;
/// Stages of [`pipeline_async_cluster_masks`].
pub const PM_STAGES: usize = 2;

/// Two clusters of two CTAs, each CTA making the same 2-stage
/// `PipelineAsync` with `producer_mask` 1 and `consumer_mask` 0 and the
/// init sync deferred to a cluster-wide `pipeline_init_arrive` /
/// `pipeline_init_wait`. Only CTA 0's warp 0 produces: its commits arrive
/// on CTA 1's full barriers. Only CTA 1's warp 0 consumes: its releases
/// arrive on CTA 0's empty barriers, which CTA 0's producer (and its tail)
/// waits on. Were the masks ignored, both would wait forever. Each side's
/// lane 0 records the index, phase and count of every round.
pub fn pipeline_async_cluster_masks() -> Case {
    let kernel = Kernel::builder("pipeline_async_cluster_masks").block_dim([64, 1, 1]).build(|| {
        let trace = param_tensor::<i32>("trace", Layout::row_major((4 * PM_ROUNDS, 3)));
        let rank = block_idx_in_cluster();
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let options = PipelineAsyncOptions {
            producer_mask: Some(Int32::from(1)),
            consumer_mask: Some(Int32::from(0)),
            defer_sync: true,
            ..Default::default()
        };
        let pipeline = PipelineAsync::create(PM_STAGES, warp, warp, options);
        let cluster = || Some(Shape::from((2, 1)));
        pipeline_init_arrive(cluster(), false);
        pipeline_init_wait(cluster());
        let rounds = PM_ROUNDS as i32;
        let base = block_idx().x * rounds;
        let record = |k: Int32, s: &PipelineState| {
            if_(lane_idx().eq(0), || {
                trace.store((base + k, 0), s.index());
                trace.store((base + k, 1), s.phase());
                trace.store((base + k, 2), s.count());
            });
        };
        if_(warp_idx().eq(0), || {
            if_else(
                rank.eq(0),
                || {
                    let mut p = for_range_carry(0, rounds, 1, pipeline.make_producer(), |k, mut p| {
                        let handle = p.acquire_and_advance(None);
                        record(k, &handle.state());
                        handle.commit();
                        p
                    });
                    p.tail();
                },
                || {
                    for_range_carry(0, rounds, 1, pipeline.make_consumer(), |k, mut c| {
                        let handle = c.wait_and_advance(None);
                        record(k, &handle.state());
                        handle.release();
                        c
                    });
                },
            );
        });
        // Warp 1 records nothing; its rows are left to warp 0.
        agent_sync(Agent::ThreadBlockCluster, false);
    });
    fn check(args: &[KernelArg]) {
        let trace = args[0].to_vec::<i32>();
        for b in 0..4 {
            let producer = b % 2 == 0;
            for k in 0..PM_ROUNDS {
                let (index, wraps) = ((k % PM_STAGES) as i32, (k / PM_STAGES) as i32);
                let phase = if producer { 1 ^ (wraps & 1) } else { wraps & 1 };
                let row = (b * PM_ROUNDS + k) * 3;
                assert_eq!(trace[row..row + 3], [index, phase, k as i32], "block {b}, round {k}");
            }
        }
    }
    let cfg = LaunchConfig::new([4, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    Case { name: "pipeline_async_cluster_masks", kernel, cfg, small_only: false, check }
}

/// The `PipelineAsync` kernels that need sm_90.
pub fn async_sched_cases_sm90() -> Vec<Case> {
    vec![pipeline_async_cluster_masks()]
}

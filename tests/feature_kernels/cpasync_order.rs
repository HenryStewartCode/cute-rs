//! `PipelineCpAsync` and `PipelineOrder` (roadmap part
//! m9-pipeline-cp-async-order), held to plain Rust. Built for sm_89, so
//! their waits spin on `mbarrier.test_wait` and the SM89 GPU runs them.

use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions, PipelineOrder, PipelineOrderOptions,
};
use cute::prelude::*;

use super::Case;

/// Items each [`pipeline_cp_async_ring`] hands over: more than three
/// passes of every ring, so the phases flip often.
pub const CPA_ITEMS: usize = 13;

/// A [`pipeline_cp_async_ring`] configuration: producer warps, consumer
/// warps and stages.
#[derive(Clone, Copy, Debug)]
pub struct CpAsyncRing {
    pub producers: usize,
    pub consumers: usize,
    pub stages: usize,
}

impl CpAsyncRing {
    /// Words per stage: 16 bytes per producer thread.
    pub const fn tile(&self) -> usize {
        self.producers * 32 * 4
    }

    pub const fn threads(&self) -> usize {
        (self.producers + self.consumers) * 32
    }
}

/// The configurations of [`pipeline_cp_async_ring`].
pub const CPA_RINGS: [CpAsyncRing; 4] = [
    CpAsyncRing { producers: 1, consumers: 1, stages: 2 },
    CpAsyncRing { producers: 1, consumers: 2, stages: 3 },
    CpAsyncRing { producers: 2, consumers: 1, stages: 4 },
    CpAsyncRing { producers: 2, consumers: 4, stages: 2 },
];

/// The plain-Rust result of [`pipeline_cp_async_ring`]: `x * 3 + 1` of
/// every word.
fn cp_async_ring_check(args: &[KernelArg]) {
    let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
    assert_eq!(src.len(), out.len());
    for (i, (&x, &y)) in src.iter().zip(&out).enumerate() {
        assert_eq!(y, x.wrapping_mul(3).wrapping_add(1), "word {i}");
    }
}

/// `ring.producers` producer warps load [`CPA_ITEMS`] tiles of `src` into a
/// `ring.stages`-stage ring with `cp.async` (16 bytes per thread, `.cg`)
/// through a `PipelineCpAsync` whose full barriers count every producer
/// thread, each committing once per stage after its copy. `ring.consumers`
/// consumer warps wait, write `x * 3 + 1` of each word to `out` and
/// release.
pub fn pipeline_cp_async_ring(ring: CpAsyncRing) -> Case {
    let names = [
        "pipeline_cp_async_p1c1s2",
        "pipeline_cp_async_p1c2s3",
        "pipeline_cp_async_p2c1s4",
        "pipeline_cp_async_p2c4s2",
    ];
    let index = CPA_RINGS
        .iter()
        .position(|r| (r.producers, r.consumers, r.stages) == (ring.producers, ring.consumers, ring.stages))
        .expect("a configuration of CPA_RINGS");
    let name = names[index];
    let tile = ring.tile();
    let threads = ring.threads();
    let kernel = Kernel::builder(name).block_dim([threads as u32, 1, 1]).target(Arch::Sm89).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(CPA_ITEMS * tile));
        let out = param_tensor::<u32>("out", Layout::col_major(CPA_ITEMS * tile));
        let producer_group = CooperativeGroup::new(Agent::Warp, ring.producers);
        let consumer_group = CooperativeGroup::new(Agent::Warp, ring.consumers);
        let pipeline =
            PipelineCpAsync::create(ring.stages, producer_group, consumer_group, PipelineCpAsyncOptions::default());
        let data = make_smem_tensor::<u32>(Layout::col_major(ring.stages * tile));
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let chunk = Layout::col_major(4);
        let t = thread_idx().x;
        let (producer, consumer) = pipeline.make_participants();
        let tile_i = tile as i32;
        if_else(
            make_warp_uniform(warp_idx()).lt(ring.producers as i32),
            || {
                let mut p = for_range_carry(0, CPA_ITEMS as i32, 1, producer, |k, mut p| {
                    let handle = p.acquire_and_advance(None);
                    let from = src.ptr().offset(k * tile_i + t * 4);
                    let to = data.ptr().offset(handle.index() * tile_i + t * 4);
                    copy(&atom, &make_tensor(&from, chunk.clone()), &make_tensor(&to, chunk.clone()));
                    handle.commit();
                    p
                });
                p.tail();
            },
            || {
                let c = t - (ring.producers * 32) as i32;
                let per = tile / (ring.consumers * 32);
                for_range_carry(0, CPA_ITEMS as i32, 1, consumer, |k, mut cons| {
                    let handle = cons.wait_and_advance(None);
                    for j in 0..per {
                        let w = c + (j * ring.consumers * 32) as i32;
                        let x = data.load(handle.index() * tile_i + w);
                        out.store(k * tile_i + w, x * 3u32 + 1u32);
                    }
                    handle.release();
                    cons
                });
            },
        );
    });
    let cfg = LaunchConfig::new([1, 1, 1], [threads as u32, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check: cp_async_ring_check }
}

/// Rounds each group takes in [`pipeline_order_log`].
pub const PO_ROUNDS: usize = 5;
/// The depth of [`pipeline_order_log`]'s pipeline.
pub const PO_DEPTH: usize = 2;

/// A [`pipeline_order_log`] configuration: groups, warps per group, and
/// whether the groups pass explicit states (`wait_with`, `arrive_with`).
#[derive(Clone, Copy, Debug)]
pub struct OrderLog {
    pub groups: usize,
    pub group_warps: usize,
    pub explicit_state: bool,
}

/// The configurations of [`pipeline_order_log`].
pub const PO_LOGS: [OrderLog; 4] = [
    OrderLog { groups: 2, group_warps: 1, explicit_state: false },
    OrderLog { groups: 3, group_warps: 1, explicit_state: false },
    OrderLog { groups: 3, group_warps: 1, explicit_state: true },
    OrderLog { groups: 2, group_warps: 2, explicit_state: false },
];

/// `log.groups` groups of `log.group_warps` warps take [`PO_ROUNDS`]
/// rounds through a `PipelineOrder` of depth [`PO_DEPTH`] (the group index
/// a runtime value), carried through the loop. A round is [`PO_DEPTH`]
/// turns, one per stage of the pipeline, as a ping-pong kernel's math and
/// epilogue sections are. In its turn at stage `d`, a group's first lane
/// appends `group * 100 + round` to log `d` at the position a shared
/// counter of that log gives, and bumps the counter; every thread of the
/// group then arrives. The groups take each stage's turns in strict order,
/// so every log comes out `0, 100, 200, ..., 1, 101, ...`.
pub fn pipeline_order_log(log: OrderLog) -> Case {
    let names = ["pipeline_order_g2", "pipeline_order_g3", "pipeline_order_g3_states", "pipeline_order_g2_warpgroups"];
    let index = PO_LOGS
        .iter()
        .position(|l| (l.groups, l.group_warps, l.explicit_state) == (log.groups, log.group_warps, log.explicit_state))
        .expect("a configuration of PO_LOGS");
    let name = names[index];
    let threads = log.groups * log.group_warps * 32;
    let entries = log.groups * PO_ROUNDS;
    let kernel = Kernel::builder(name).block_dim([threads as u32, 1, 1]).target(Arch::Sm89).build(move || {
        let out = param_tensor::<i32>("log", Layout::col_major(PO_DEPTH * entries + 1));
        let counters = make_smem_tensor::<i32>(Layout::col_major(PO_DEPTH));
        let t = thread_idx().x;
        // Written before the pipeline's init sync.
        if_(t.eq(0), || {
            for d in 0..PO_DEPTH {
                counters.store(d, 0);
            }
        });
        let group = CooperativeGroup::new(Agent::Warp, log.group_warps);
        let group_id = make_warp_uniform(warp_idx()) / log.group_warps as i32;
        let order = PipelineOrder::create(PO_DEPTH, log.groups, group_id, group, PipelineOrderOptions::default());
        let leader = (t % (log.group_warps * 32) as i32).eq(0);
        let turn = |d: usize, round: Int32| {
            if_(leader, || {
                let pos = counters.load(d);
                out.store(pos + (d * entries) as i32, group_id * 100 + round);
                counters.store(d, pos + 1);
            });
        };
        let count = if log.explicit_state {
            let start = order.state();
            let end = for_range_carry(0, PO_ROUNDS as i32, 1, start, |round, mut state| {
                for d in 0..PO_DEPTH {
                    order.wait_with(&state);
                    turn(d, round);
                    state = order.arrive_with(state);
                }
                state
            });
            end.count()
        } else {
            let end = for_range_carry(0, PO_ROUNDS as i32, 1, order, |round, mut order| {
                for d in 0..PO_DEPTH {
                    order.wait();
                    turn(d, round);
                    order.arrive();
                }
                order
            });
            end.state().count()
        };
        if_(t.eq(0), || out.store(PO_DEPTH * entries, count));
    });
    fn check(args: &[KernelArg]) {
        let got = args[0].to_vec::<i32>();
        let entries = (got.len() - 1) / PO_DEPTH;
        let groups = entries / PO_ROUNDS;
        let want: Vec<i32> = (0..PO_ROUNDS).flat_map(|r| (0..groups).map(move |g| (g * 100 + r) as i32)).collect();
        for d in 0..PO_DEPTH {
            assert_eq!(got[d * entries..(d + 1) * entries], want[..], "the turn log of stage {d}");
        }
        assert_eq!(got[PO_DEPTH * entries], (PO_ROUNDS * PO_DEPTH) as i32, "the state's count after the rounds");
    }
    let cfg = LaunchConfig::new([1, 1, 1], [threads as u32, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

/// The `PipelineCpAsync` rings and `PipelineOrder` logs, all built for
/// sm_89.
pub fn cpasync_order_cases_sm89() -> Vec<Case> {
    CPA_RINGS.into_iter().map(pipeline_cp_async_ring).chain(PO_LOGS.into_iter().map(pipeline_order_log)).collect()
}

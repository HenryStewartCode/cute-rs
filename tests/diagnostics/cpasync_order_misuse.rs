//! cp.async ordering misuse diagnostics.

use super::simulate_err;
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions, PipelineOrder, PipelineOrderOptions,
};
use cute::prelude::*;

const ITEMS: i32 = 6;

/// How [`ring`] gets the `.noinc` contract wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Misuse {
    /// Commit (the `.noinc` arrival) before the stage's `cp.async`.
    CommitBeforeCopy,
    /// Full barriers counting the producer warp as one arrival (a
    /// group of one thread), where its 32 threads each arrive.
    CountInWarps,
}

/// Warp 0 loads [`ITEMS`] tiles of 128 words with `cp.async` into a
/// 2-stage `PipelineCpAsync`; warp 1 copies them out.
fn ring(misuse: Misuse) -> Kernel {
    Kernel::builder("cp_async_misuse").block_dim([64, 1, 1]).target(Arch::Sm89).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(ITEMS as usize * 128));
        let out = param_tensor::<u32>("out", Layout::col_major(ITEMS as usize * 128));
        let producers = match misuse {
            Misuse::CountInWarps => CooperativeGroup::new(Agent::Thread, 1),
            Misuse::CommitBeforeCopy => CooperativeGroup::new(Agent::Warp, 1),
        };
        let consumers = CooperativeGroup::new(Agent::Warp, 1);
        let pipeline = PipelineCpAsync::create(2, producers, consumers, PipelineCpAsyncOptions::default());
        let data = make_smem_tensor::<u32>(Layout::col_major(2 * 128));
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let chunk = Layout::col_major(4);
        let lane = lane_idx();
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, ITEMS, 1, producer, |k, mut p| {
                    let handle = p.acquire_and_advance(None);
                    if misuse == Misuse::CommitBeforeCopy {
                        handle.commit();
                    }
                    let from = src.ptr().offset(k * 128 + lane * 4);
                    let to = data.ptr().offset(handle.index() * 128 + lane * 4);
                    copy(&atom, &make_tensor(&from, chunk.clone()), &make_tensor(&to, chunk.clone()));
                    if misuse != Misuse::CommitBeforeCopy {
                        handle.commit();
                    }
                    p
                });
                p.tail();
            },
            || {
                for_range_carry(0, ITEMS, 1, consumer, |k, mut c| {
                    let handle = c.wait_and_advance(None);
                    for e in 0..4 {
                        let w = lane * 4 + e;
                        out.store(k * 128 + w, data.load(handle.index() * 128 + w));
                    }
                    handle.release();
                    c
                });
            },
        );
    })
}

fn ring_args() -> Vec<KernelArg> {
    vec![
        KernelArg::buffer(&(0..ITEMS as u32 * 128).collect::<Vec<_>>()),
        KernelArg::buffer(&vec![0u32; ITEMS as usize * 128]),
    ]
}

/// A commit issued before the stage's `cp.async`: the `.noinc` arrival
/// covers no copy, so the consumer's read of the stage, which the copy
/// may still be writing, is reported.
#[test]
fn commit_before_the_copies_is_reported() {
    let e = simulate_err(&ring(Misuse::CommitBeforeCopy), 64, &mut ring_args());
    assert!(e.message.contains("while a cp.async is still writing it"), "{e}");
}

/// Full barriers counting one arrival where 32 producer threads arrive
/// with `.noinc`: the phase has no room for the other lanes' arrivals
/// (it would complete at the first), and the simulator reports the
/// count.
#[test]
fn full_count_in_warps_is_reported() {
    let e = simulate_err(&ring(Misuse::CountInWarps), 64, &mut ring_args());
    assert!(
        e.message.contains("cp_async_mbarrier_arrive_noinc on an mbarrier whose phase is waiting for 1 arrival(s)")
            && e.message.contains("the mbarrier's count must include one arrival per call"),
        "{e}"
    );
}

/// Three warps take turns through a `PipelineOrder` of depth 2 with
/// explicit states; warp 1 skips its arrive in round 2, so warp 2 waits
/// forever on its barrier of stage 0 (mbarrier 2): the simulator
/// reports the hang, naming the mbarrier and the phase.
#[test]
fn pipeline_order_skipped_arrive_deadlocks() {
    let kernel = Kernel::builder("order_skips").block_dim([96, 1, 1]).target(Arch::Sm89).build(|| {
        let group = CooperativeGroup::new(Agent::Warp, 1);
        let warp = make_warp_uniform(warp_idx());
        let order = PipelineOrder::create(2, 3, warp, group, PipelineOrderOptions::default());
        for_range_carry(0, 4, 1, order.state(), |round, state| {
            order.wait_with(&state);
            let skip = warp.eq(1) & round.eq(2);
            if_else_yield(
                skip,
                || {
                    let mut s = state;
                    s.advance();
                    s
                },
                || order.arrive_with(state),
            )
        });
    });
    let e = simulate_err(&kernel, 96, &mut []);
    assert!(
        e.message.contains("the kernel did not terminate")
            && e.message.contains("warp 2 of block (0,0,0) spins")
            && e.message.contains("the mbarrier smem0[2] for phase 1 (parity 1), which still waits for 32 of its 32"),
        "{e}"
    );
}

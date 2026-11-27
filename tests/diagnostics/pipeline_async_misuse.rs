//! PipelineAsync misuse diagnostics.

use super::{build_panic, simulate_err};
use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions};
use cute::prelude::*;

const ITEMS: i32 = 6;

/// One producer warp (warp 0) and `consumers` consumer warps hand
/// [`ITEMS`] words through a 1-stage `PipelineAsync` whose empty
/// barriers count `declared` consumer warps. The consumers wait on the
/// full barrier unless `skip_wait`.
fn handoff(consumers: u32, declared: i32, skip_wait: bool) -> (Kernel, u32) {
    let threads = 32 * (1 + consumers);
    let kernel = Kernel::builder("pipeline_async_misuse").block_dim([threads, 1, 1]).target(Arch::Sm89).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(threads as usize));
        let producers = CooperativeGroup::new(Agent::Warp, 1);
        let consumer_group = CooperativeGroup::new(Agent::Warp, declared);
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        let lane = lane_idx();
        // Written before the pipeline's init sync, so that every read is
        // of initialised memory.
        if_(warp_idx().eq(0), || data.store(lane, 0u32));
        let pipeline = PipelineAsync::create(1, producers, consumer_group, PipelineAsyncOptions::default());
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, ITEMS, 1, producer, |k, mut p| {
                    let handle = p.acquire_and_advance(None);
                    data.store(lane, k.to::<u32>() + 1u32);
                    handle.commit();
                    p
                });
                p.tail();
            },
            || {
                let start = (Uint32::from(0u32), consumer);
                let (sum, _) = for_range_carry(0, ITEMS, 1, start, |_, (sum, mut c)| {
                    let handle = if skip_wait {
                        let h = c.current_handle();
                        c.advance(1);
                        h
                    } else {
                        c.wait_and_advance(None)
                    };
                    let x = data.load(lane);
                    handle.release();
                    (sum + x, c)
                });
                out.store(thread_idx().x, sum);
            },
        );
    });
    (kernel, threads)
}

/// Empty barriers counting two consumer warps where one consumes: the
/// producer's second acquire waits for a phase that never completes,
/// and the simulator reports the hang.
#[test]
fn consumer_count_too_high_hangs() {
    let (kernel, threads) = handoff(1, 2, false);
    let mut args = vec![KernelArg::buffer(&vec![0u32; threads as usize])];
    let e = simulate_err(&kernel, threads, &mut args);
    assert!(e.message.contains("the kernel did not terminate"), "{e}");
}

/// Empty barriers counting one consumer warp where two consume: each
/// round's releases complete two phases, the first as soon as one
/// consumer has released the stage. The producer then overwrites the
/// stage while the other may still read it, and its next wait finds
/// the barrier two phases ahead. Under the round-robin schedule the
/// simulator reports the phase aliasing first; under seeded schedules,
/// whichever comes first of it and the race on the stage.
#[test]
fn consumer_count_too_low_is_reported() {
    let (kernel, threads) = handoff(2, 1, false);
    let mut args = vec![KernelArg::buffer(&vec![0u32; threads as usize])];
    let e = simulate_err(&kernel, threads, &mut args);
    assert!(e.message.contains("phase aliasing"), "{e}");
    let mut races = 0;
    for seed in 0..16 {
        let opts = SimOptions { schedule: Schedule::Random { seed }, ..SimOptions::default() };
        let mut args = vec![KernelArg::buffer(&vec![0u32; threads as usize])];
        let cfg = LaunchConfig::new([1, 1, 1], [threads, 1, 1]);
        let e = kernel.simulate_with(cfg, &mut args, &opts).expect_err("a too-low consumer count is caught");
        assert!(e.message.contains("phase aliasing") || e.message.contains("data race"), "{e}");
        races += usize::from(e.message.contains("data race"));
    }
    assert!(races > 0, "no seeded schedule reported the race on the stage");
}

/// A consumer that reads the stage without waiting on its full barrier
/// races with the producer's write.
#[test]
fn consumer_reading_without_waiting_races() {
    let (kernel, threads) = handoff(1, 1, true);
    let mut args = vec![KernelArg::buffer(&vec![0u32; threads as usize])];
    let e = simulate_err(&kernel, threads, &mut args);
    assert!(e.message.contains("data race"), "{e}");
}

/// Committing a handle of one pipeline through another's producer, or
/// releasing one through another's consumer, and a pipeline of 0
/// stages, are refused while the kernel is built.
#[test]
fn pipeline_async_refuses_bad_arguments() {
    let two = || {
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let options = || PipelineAsyncOptions { defer_sync: true, ..Default::default() };
        (PipelineAsync::create(2, warp, warp, options()), PipelineAsync::create(2, warp, warp, options()))
    };
    let msg = build_panic(|| {
        Kernel::builder("foreign_handle").block_dim([32, 1, 1]).target(Arch::Sm89).build(|| {
            let (a, b) = two();
            let handle = b.make_producer().current_handle();
            a.make_producer().commit(Some(&handle));
        });
    });
    assert!(
        msg.contains("PipelineProducer::commit: the handle (ResourceHandle) was acquired from another pipeline"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::builder("foreign_handle").block_dim([32, 1, 1]).target(Arch::Sm89).build(|| {
            let (a, b) = two();
            let handle = a.make_consumer().current_handle();
            b.make_consumer().release(Some(&handle));
        });
    });
    assert!(
        msg.contains("PipelineConsumer::release: the handle (ResourceHandle) was acquired from another pipeline"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::builder("no_stages").block_dim([32, 1, 1]).build(|| {
            let warp = CooperativeGroup::new(Agent::Warp, 1);
            PipelineAsync::create(0, warp, warp, PipelineAsyncOptions::default());
        });
    });
    assert!(msg.contains("PipelineAsync::create: a pipeline needs at least one stage, not 0"), "{msg}");
}

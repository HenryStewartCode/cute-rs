//! Kernels of the async pipelines and the persistent schedulers
//! whose results depend on timing or that take arguments of their own, so
//! that they are not feature kernels: shared by `tests/pipeline_async.rs`,
//! `tests/tile_schedulers.rs` and `tests/gpu_async_sched.rs`.

pub mod schedulers;

use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineAsync, PipelineAsyncOptions, PipelineEmpty};
use cute::prelude::*;

/// Stages of [`token_kernel`]'s ring: few, so that each side often finds
/// the other behind.
pub const TOKEN_STAGES: usize = 2;
/// Items handed over by [`token_kernel`].
pub const TOKEN_ITEMS: usize = 24;

/// Warp 0 produces and warp 1 consumes [`TOKEN_ITEMS`] items through a
/// [`TOKEN_STAGES`]-stage `PipelineAsync` (sm_89). Each side probes first
/// (`try_acquire`, `try_wait`), loads its operand from global memory
/// meanwhile, then calls `acquire_and_advance(token)` /
/// `wait_and_advance(token)`, which waits only if the probe failed. The
/// consumer sums `3 * x + k` over the items per lane into `sums`; lane 0
/// of each side counts its true and false tokens into `counts`
/// (producer true, producer false, consumer true, consumer false).
pub fn token_kernel() -> Kernel {
    Kernel::builder("pipeline_tokens").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(TOKEN_ITEMS * 32));
        let sums = param_tensor::<u32>("sums", Layout::col_major(32));
        let counts = param_tensor::<u32>("counts", Layout::col_major(4));
        let lane = lane_idx();
        let warp = CooperativeGroup::new(Agent::Warp, 1);
        let pipeline = PipelineAsync::create(TOKEN_STAGES, warp, warp, PipelineAsyncOptions::default());
        let data = make_smem_tensor::<u32>(Layout::col_major(TOKEN_STAGES * 32));
        let (producer, consumer) = pipeline.make_participants();
        let items = TOKEN_ITEMS as i32;
        let zero = || Uint32::from(0u32);
        let write_counts = |base: i32, yes: Uint32, no: Uint32| {
            if_(lane.eq(0), || {
                counts.store(base, yes);
                counts.store(base + 1, no);
            });
        };
        if_else(
            warp_idx().eq(0),
            || {
                let start = (zero(), zero(), producer);
                let (yes, no, mut p) = for_range_carry(0, items, 1, start, |k, (yes, no, mut p)| {
                    let token = p.try_acquire();
                    let x = src.load(k * 32 + lane) * 3u32 + k.to::<u32>();
                    let handle = p.acquire_and_advance(Some(token));
                    data.store(handle.index() * 32 + lane, x);
                    handle.commit();
                    let t = token.to::<u32>();
                    (yes + t, no + (1u32 - t), p)
                });
                p.tail();
                write_counts(0, yes, no);
            },
            || {
                let start = (zero(), zero(), zero(), consumer);
                let (sum, yes, no, _) = for_range_carry(0, items, 1, start, |_, (sum, yes, no, mut c)| {
                    let token = c.try_wait();
                    let handle = c.wait_and_advance(Some(token));
                    let x = data.load(handle.index() * 32 + lane);
                    handle.release();
                    let t = token.to::<u32>();
                    (sum + x, yes + t, no + (1u32 - t), c)
                });
                sums.store(lane, sum);
                write_counts(2, yes, no);
            },
        );
    })
}

/// The launch of [`token_kernel`].
pub fn token_cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [64, 1, 1])
}

/// The arguments of [`token_kernel`]: small numbers, and zeroed outputs.
pub fn token_args() -> Vec<KernelArg> {
    let src: Vec<u32> = (0..TOKEN_ITEMS * 32).map(|i| (i as u32 * 7919) % 1000).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 32]), KernelArg::buffer(&[0u32; 4])]
}

/// Check the sums of a [`token_kernel`] run, and that each side counted
/// one token per item; returns the four counts.
pub fn check_token_run(args: &[KernelArg]) -> [u32; 4] {
    let (src, sums, counts) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
    for (l, &got) in sums.iter().enumerate() {
        let want: u32 = (0..TOKEN_ITEMS).map(|k| src[k * 32 + l] * 3 + k as u32).sum();
        assert_eq!(got, want, "lane {l}");
    }
    let counts: [u32; 4] = counts.try_into().unwrap();
    assert_eq!(counts[0] + counts[1], TOKEN_ITEMS as u32, "producer tokens {counts:?}");
    assert_eq!(counts[2] + counts[3], TOKEN_ITEMS as u32, "consumer tokens {counts:?}");
    counts
}

/// Items of [`generic_kernel`].
pub const GENERIC_ITEMS: usize = 9;

/// The body of [`generic_kernel`], generic over the pipeline: one warp
/// both produces and consumes each item in turn (probing first, with the
/// probe's token), each lane writing its own word of the stage and reading
/// it back doubled into `out`. Correct with any pipeline, since a lane
/// only reads what it wrote itself: a configuration that needs no
/// hand-off.
pub fn generic_body<P: Pipeline>(pipeline: &P, src: &Tensor<u32>, out: &Tensor<u32>) {
    let lane = lane_idx();
    let data = make_smem_tensor::<u32>(Layout::col_major(pipeline.num_stages() * 32));
    let (producer, consumer) = pipeline.make_participants();
    let (mut p, _) = for_range_carry(0, GENERIC_ITEMS as i32, 1, (producer, consumer), |k, (mut p, mut c)| {
        let token = p.try_acquire();
        let put = p.acquire_and_advance(Some(token));
        data.store(put.index() * 32 + lane, src.load(k * 32 + lane));
        put.commit();
        let token = c.try_wait();
        let got = c.wait_and_advance(Some(token));
        out.store(k * 32 + lane, data.load(got.index() * 32 + lane) * 2u32);
        got.release();
        (p, c)
    });
    p.tail();
}

/// [`generic_body`] over a 2-stage `PipelineAsync` of one warp a side
/// (sm_89), or over `PipelineEmpty`.
pub fn generic_kernel(empty: bool) -> Kernel {
    let name = if empty { "generic_pipeline_empty" } else { "generic_pipeline_async" };
    Kernel::builder(name).block_dim([32, 1, 1]).target(Arch::Sm89).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(GENERIC_ITEMS * 32));
        let out = param_tensor::<u32>("out", Layout::col_major(GENERIC_ITEMS * 32));
        if empty {
            generic_body(&PipelineEmpty::create(), &src, &out);
        } else {
            let warp = CooperativeGroup::new(Agent::Warp, 1);
            let pipeline = PipelineAsync::create(2, warp, warp, PipelineAsyncOptions::default());
            generic_body(&pipeline, &src, &out);
        }
    })
}

/// The launch of [`generic_kernel`].
pub fn generic_cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [32, 1, 1])
}

/// The arguments of [`generic_kernel`].
pub fn generic_args() -> Vec<KernelArg> {
    let src: Vec<u32> = (0..GENERIC_ITEMS * 32).map(|i| i as u32 * 5 + 1).collect();
    vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0u32; GENERIC_ITEMS * 32])]
}

/// Check [`generic_kernel`]'s output.
pub fn check_generic(args: &[KernelArg]) {
    let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
    for (i, (&s, &o)) in src.iter().zip(&out).enumerate() {
        assert_eq!(o, s * 2, "word {i}");
    }
}

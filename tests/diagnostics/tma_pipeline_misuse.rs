//! PipelineTmaAsync and PipelineTmaStore: misuse diagnostics.

use super::build_panic;
use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaStore, TmaStoreFence,
    pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;

/// K tiles of the hand-offs.
const TILES: i32 = 4;

/// A producer warp loads [`TILES`] 16x16 f16 tiles by TMA through a 2-stage
/// `PipelineTmaAsync` whose empty barriers count `consumers`; one consumer warp
/// sums them.
fn handoff(consumers: CooperativeGroup) -> Kernel {
    Kernel::builder("tma_pipeline_count").block_dim([64, 1, 1]).build(move || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((16, 64), (64, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let smem = Layout::new((16, 16, 2), (16, 1, 256));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma) = cpasync::make_tiled_tma_atom(g2s, &a, Layout::new((16, 16), (16, 1)), (16, 16), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 128);
        let pipeline = PipelineTmaAsync::create(
            2,
            CooperativeGroup::new(Agent::Thread, 1),
            consumers,
            512,
            PipelineTmaAsyncOptions::default(),
        );
        let g = local_tile(&tma, (16, 16), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, TILES, 1, producer, |k, mut p| {
                    let h = p.acquire_and_advance(None);
                    copy_with(
                        &atom,
                        &tg.slice((.., k)),
                        &ts.slice((.., h.index())),
                        CopyArgs::new().tma_bar_ptr(&h.barrier()),
                    );
                    h.commit();
                    p
                });
                p.tail();
            },
            || {
                let lane = lane_idx();
                let (sum, _) = for_range_carry(0, TILES, 1, (Float32::from(0.0f32), consumer), |_, (sum, mut c)| {
                    let h = c.wait_and_advance(None);
                    let x = s.load((lane % 16, lane / 16, h.index())).to::<f32>();
                    h.release();
                    (sum + x, c)
                });
                out.store(lane, sum);
            },
        );
    })
}

fn handoff_args() -> Vec<KernelArg> {
    let a: Vec<F16> = (0..16 * 64).map(|i| F16::from_f32((i % 7) as f32)).collect();
    vec![KernelArg::buffer(&a), KernelArg::buffer(&[0f32; 32])]
}

/// The pitfall of the TMA pipeline: only lane 0 of each consumer warp arrives on
/// an empty barrier, so a consumer group counted in threads never completes a
/// phase. The simulator's hang report names the mbarrier and the waiting warp.
#[test]
fn consumer_group_in_threads_hangs() {
    let kernel = handoff(CooperativeGroup::new(Agent::Thread, 32));
    let mut args = handoff_args();
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).expect_err("a hang");
    let text = e.to_string();
    assert!(text.contains("the kernel did not terminate"), "{e}");
    // The producer (warp 0) waits on the first empty barrier (after the two full
    // ones), which has 1 of its 32 arrivals.
    assert!(text.contains("warp 0 of block (0,0,0) waits at mbarrier.try_wait"), "{e}");
    assert!(text.contains("on the mbarrier smem1[2] for phase 0"), "{e}");
    assert!(text.contains("still waits for 31 of its 32 arrival(s)"), "{e}");
    // The count in warps of signalling lanes works.
    let mut args = handoff_args();
    handoff(CooperativeGroup::new(Agent::Thread, 1))
        .simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// A 2x1 cluster's multicast pipeline whose consumer count leaves out the
/// multicast partner (4 warps, not 2 x 4): a CTA's empty barrier completes a
/// phase on its own consumers' releases, and its producer multicasts into the
/// peer's stage while the peer may still read it. The simulator reports it.
#[test]
fn multicast_count_without_the_partners_is_reported() {
    let kernel = Kernel::builder("tma_pipeline_mcast_count").block_dim([160, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((32, 128), (128, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::col_major(2 * 128));
        let smem = Layout::new((16, 16, 2), (16, 1, 256));
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
        let (atom, tma) = cpasync::make_tiled_tma_atom(mc, &a, Layout::new((16, 16), (16, 1)), (16, 16), 2).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 128);
        // Both CTAs need the same tiles: multicast along M of a (1, 2, 1, 1) cluster.
        let pipeline = PipelineTmaAsync::create(
            2,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, 4),
            512,
            PipelineTmaAsyncOptions {
                cta_layout_vmnk: Some(Layout::col_major((1usize, 2usize, 1usize, 1usize))),
                mcast_mode_mn: (0, 1),
                defer_sync: true,
                ..Default::default()
            },
        );
        let cluster = Layout::col_major((2usize, 1usize, 1usize)).shape().clone();
        pipeline_init_arrive(Some(cluster.clone()), true);
        pipeline_init_wait(Some(cluster));
        let rank = block_idx_in_cluster();
        let mask = make_layout_image_mask(&Layout::col_major((2usize, 1usize)), (rank, 0), 0);
        let g = local_tile(&tma, (16, 16), (0, ..));
        let (ts, tg) =
            cpasync::tma_partition(&atom, rank, &make_layout(2, 1), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, 8, 1, producer, |k, mut p| {
                    let h = p.acquire_and_advance(None);
                    let args = CopyArgs::new().tma_bar_ptr(&h.barrier()).mcast_mask(mask);
                    copy_with(&atom, &tg.slice((.., k)), &ts.slice((.., h.index())), args);
                    h.commit();
                    p
                });
                p.tail();
            },
            || {
                let ct = thread_idx().x - 32;
                let (sum, _) = for_range_carry(0, 8, 1, (Float32::from(0.0f32), consumer), |_, (sum, mut c)| {
                    let h = c.wait_and_advance(None);
                    let x = s.load((ct % 16, ct / 16 % 16, h.index())).to::<f32>();
                    h.release();
                    (sum + x, c)
                });
                out.store(rank * 128 + ct, sum);
            },
        );
    });
    let cfg = LaunchConfig::new([2, 1, 1], [160, 1, 1]).with_cluster([2, 1, 1]);
    let mut caught = 0;
    for seed in 0..8 {
        let a: Vec<F16> = (0..32 * 128).map(|i| F16::from_f32((i % 5) as f32)).collect();
        let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&[0f32; 256])];
        let opts = SimOptions {
            schedule: Schedule::Random { seed },
            async_completion: AsyncCompletion::Random(seed),
            ..SimOptions::default()
        };
        if let Err(e) = kernel.simulate_with(cfg, &mut args, &opts) {
            let m = &e.message;
            assert!(m.contains("data race") || m.contains("phase aliasing") || m.contains("while a bulk copy"), "{e}");
            caught += 1;
        }
    }
    assert!(caught > 0, "no schedule caught the missing partner arrivals");
}

/// A TMA-store epilogue through `PipelineTmaStore` over 4 tiles and 2 staging
/// buffers. Without `acquire`, the threads rewrite a buffer an earlier TMA store
/// may still be reading; without `tail`, the block exits with a store still
/// reading shared memory.
fn store_epilogue(acquire: bool, tail: bool) -> Kernel {
    Kernel::builder("tma_store_pipeline_misuse").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor_aligned::<f32>("out", Layout::new((16, 64), (64, 1)), 16);
        let smem = Layout::new((16, 16, 2), (16, 1, 256));
        let s2g = cpasync::CopyBulkTensorTileS2GOp;
        let (atom, tma) = cpasync::make_tiled_tma_atom(s2g, &out, Layout::new((16, 16), (16, 1)), (16, 16), 1).into();
        let s = make_smem_tensor_aligned::<f32>(smem, 128);
        let g = local_tile(&tma, (16, 16), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let pipeline = PipelineTmaStore::create(2, CooperativeGroup::new(Agent::Thread, 64));
        let t = thread_idx().x;
        for_range(0, 4, 1, |j| {
            let st = j % 2;
            for_range(0, 4, 1, |k| {
                let idx = k * 64 + t;
                s.store((idx / 16, idx % 16, st), (j * 256 + idx).to::<f32>());
            });
            fence_view_async_shared();
            cute::pipeline::sync(1);
            if_(warp_idx().eq(0), || {
                copy(&atom, &ts.slice((.., st)), &tg.slice((.., j)));
                pipeline.producer_commit();
                if acquire {
                    pipeline.producer_acquire();
                }
            });
            cute::pipeline::sync(1);
        });
        if tail {
            if_(warp_idx().eq(0), || pipeline.producer_tail());
        }
    })
}

/// The epilogue with every step works; without `producer_acquire` a write to a
/// staging buffer races with the TMA store still reading it.
#[test]
fn staging_buffer_rewritten_before_acquire_is_reported() {
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    let mut args = vec![KernelArg::buffer(&[0f32; 16 * 64])];
    store_epilogue(true, true).simulate(cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<f32> = (0..16 * 64).map(|i| ((i % 64) / 16 * 256 + (i / 64) * 16 + i % 16) as f32).collect();
    assert_eq!(args[0].to_vec::<f32>(), want);
    let mut args = vec![KernelArg::buffer(&[0f32; 16 * 64])];
    let e = store_epilogue(false, true).simulate(cfg, &mut args).expect_err("the rewrite races");
    assert!(e.message.contains("data race: warp 0 overwrites smem0[0], which an asynchronous operation"), "{e}");
}

/// Exiting without `producer_tail` leaves a TMA store reading shared memory as
/// the block exits: reported.
#[test]
fn exit_without_producer_tail_is_reported() {
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    let mut args = vec![KernelArg::buffer(&[0f32; 16 * 64])];
    let e = store_epilogue(true, false).simulate(cfg, &mut args).expect_err("the exit is reported");
    assert!(e.message.contains("exits while an asynchronous operation still reads its shared memory"), "{e}");
}

/// Build-time misuse: multicast along neither mode, a dynamic cluster layout, 0
/// stages, a multicast consumer count that is not whole warps, and the fence's
/// and store pipeline's unsupported calls.
#[test]
fn tma_pipelines_refuse_bad_arguments() {
    let one = || CooperativeGroup::new(Agent::Thread, 1);
    type Misuse = (&'static str, Box<dyn FnOnce()>);
    let cases: Vec<Misuse> = vec![
        (
            "mcast_mode_mn (0, 0) multicasts along neither M nor N",
            Box::new(move || {
                let o = PipelineTmaAsyncOptions { mcast_mode_mn: (0, 0), ..Default::default() };
                PipelineTmaAsync::create(2, one(), one(), 64, o);
            }),
        ),
        (
            "cta_layout_vmnk must have a static shape",
            Box::new(move || {
                let n = param_extent("n", 1);
                let o = PipelineTmaAsyncOptions {
                    cta_layout_vmnk: Some(Layout::col_major((1usize, n, 1usize, 1usize))),
                    ..Default::default()
                };
                PipelineTmaAsync::create(2, one(), one(), 64, o);
            }),
        ),
        (
            "PipelineTmaAsync::create: a pipeline needs at least one stage, not 0",
            Box::new(move || {
                PipelineTmaAsync::create(0, one(), one(), 64, PipelineTmaAsyncOptions::default());
            }),
        ),
        (
            "the consumer group's thread count is not a whole number of warps (48 threads)",
            Box::new(move || {
                let o = PipelineTmaAsyncOptions { enable_multicast_signaling: true, ..Default::default() };
                PipelineTmaAsync::create(2, one(), CooperativeGroup::new(Agent::Thread, 48), 64, o);
            }),
        ),
        (
            "it needs at least one stage",
            Box::new(|| {
                let _ = TmaStoreFence::new(0);
            }),
        ),
        ("TmaStoreFence has no mbarrier to return", Box::new(|| TmaStoreFence::new(2).get_barrier())),
        ("TmaStoreFence::arrive_and_drop is not supported", Box::new(|| TmaStoreFence::new(2).arrive_and_drop())),
        (
            "TmaStoreFence::max is not supported",
            Box::new(|| {
                let _ = TmaStoreFence::new(2).max();
            }),
        ),
        ("PipelineTmaStore has no consumer side", Box::new(move || PipelineTmaStore::create(2, one()).consumer_wait())),
        (
            "PipelineTmaStore has no consumer side",
            Box::new(move || PipelineTmaStore::create(2, one()).consumer_release()),
        ),
    ];
    for (want, body) in cases {
        let msg = build_panic(|| {
            Kernel::builder("tma_pipeline_misuse").block_dim([64, 1, 1]).build(body);
        });
        assert!(msg.contains(want), "expected `{want}` in: {msg}");
    }
}

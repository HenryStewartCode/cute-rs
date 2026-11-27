// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Barrier kernels: named barriers, mbarriers, cluster barriers, cp.async and bulk-copy hand-offs, and the consumer-first variants.

use super::*;
/// Named barriers on 8 warps: two groups of 4 warps each exchange data
/// behind their own barrier inside a branch, warps 0-3 hand data to warps
/// 4-7 with `bar.arrive` / `bar.sync` and get an unaligned arrive back, and
/// a whole-block named barrier orders the final cross-group reads. Last, a
/// `NamedBarrier` whose id is computed at run time (7 for warps 0-3, 8 for
/// warps 4-7) orders another exchange within each group. The simulator's
/// race checker must accept every access.
pub fn named_barriers() -> Case {
    const N: usize = 256;
    let kernel = Kernel::builder("named_barriers").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let out = param_tensor::<u32>("out", Layout::row_major((N, 5)));
        let t = thread_idx().x;
        let warp = warp_idx();
        let smem = make_smem_tensor::<u32>(Layout::col_major(3 * N));
        let x = xs.load(t);
        if_else(
            warp.lt(4),
            || {
                smem.store(t, x);
                barrier(1, 128);
                out.store((t, 0), smem.load((t + 32) % 128));
            },
            || {
                smem.store(t, x * 3u32);
                barrier(2, 128);
                out.store((t, 0), smem.load((t + 32) % 128 + 128));
            },
        );
        let full = pipeline::NamedBarrier::new(3, N as u32);
        if_else(
            warp.lt(4),
            || {
                smem.store(t + 256, x ^ 0xffu32);
                full.arrive();
                out.store((t, 1), 0u32);
            },
            || {
                full.arrive_and_wait();
                out.store((t, 1), smem.load(t + 128));
                pipeline::arrive_unaligned(6, N as i32);
            },
        );
        // The producers reuse the buffer once the consumers have read it.
        if_(warp.lt(4), || {
            pipeline::wait(6, N as i32);
            smem.store(t + 256, x + 1u32);
        });
        pipeline::sync(5);
        out.store((t, 2), smem.load((t + 128) % 256));
        out.store((t, 3), smem.load((t % 128) + 256));
        let group = pipeline::NamedBarrier::new(warp / 4 + 7, 128);
        smem.store(t + 512, x * 7u32 + 1u32);
        group.arrive_and_wait();
        out.store((t, 4), smem.load(t / 128 * 128 + (t + 64) % 128 + 512));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        let first = |i: usize| if i < 128 { xs[i] } else { xs[i].wrapping_mul(3) };
        for t in 0..N {
            let row = &out[t * 5..t * 5 + 5];
            let group = t / 128 * 128;
            let want = [
                first(group + (t + 32) % 128),
                if t < 128 { 0 } else { xs[t - 128] ^ 0xff },
                first((t + 128) % 256),
                xs[t % 128].wrapping_add(1),
                xs[group + (t + 64) % 128].wrapping_mul(7).wrapping_add(1),
            ];
            assert_eq!(row, want, "thread {t}");
        }
    }
    Case {
        name: "named_barriers",
        kernel,
        cfg: LaunchConfig::new([2, 1, 1], [N as u32, 1, 1]),
        small_only: false,
        check,
    }
}

/// mbarriers on SM80: warps 0-1 hand data to warps 2-3 through an mbarrier
/// (`mbarrier.arrive` releases, a spin on `mbarrier.test_wait` acquires),
/// and the next phase goes the other way. Each consumer also tests the
/// phase still running (false) and the one just completed (true).
///
/// Warps 0-1 wait for phase 0 themselves before waiting on parity 1: while
/// phase 0 is running, parity 1 names the phase before it, which counts as
/// complete, so without that wait a warp that arrives early would pass its
/// parity-1 wait at once and read data warps 2-3 have not written.
///
/// The spins give up after [`SPIN_LIMIT`] polls, and each thread stores how
/// many did in `gave_up`.
pub fn mbarriers_sm80() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("mbarriers_sm80").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let out = param_tensor::<u32>("out", Layout::row_major((N, 3)));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(N));
        let t = thread_idx().x;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(N));
        let misses = Var::new(Uint32::from(0u32));
        let spin = |parity: i32| {
            let missed = bounded_spin(&bar, Int32::from(parity));
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        if_(t.eq(0), || mbarrier_init(&bar, 64));
        sync_threads();
        let x = xs.load(t);
        let consume = |offset: i32, parity: i32| {
            spin(parity);
            out.store((t, 0), data.load(t + offset));
            out.store((t, 1), mbarrier_test_wait(&bar, 1 - parity).to::<u32>());
            out.store((t, 2), mbarrier_test_wait(&bar, parity).to::<u32>());
        };
        if_else(
            warp_idx().lt(2),
            || {
                data.store(t, x * 5u32);
                mbarrier_arrive(&bar);
                spin(0);
            },
            || consume(-64, 0),
        );
        if_else(
            warp_idx().ge(2),
            || {
                data.store(t, x + 7u32);
                mbarrier_arrive_with(&bar, 1, MbarrierOptions::default());
            },
            || consume(64, 1),
        );
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for t in 0..N {
            let moved = if t < 64 { xs[t + 64].wrapping_add(7) } else { xs[t - 64].wrapping_mul(5) };
            assert_eq!(out[t * 3..t * 3 + 3], [moved, 0, 1], "thread {t}");
        }
    }
    Case { name: "mbarriers_sm80", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// The cluster barrier in all its forms (`cluster_arrive` by threads one by
/// one in divergent code, `cluster_wait`, the relaxed and aligned forms, and
/// `cluster_sync`), over three phases: each thread writes a value to global
/// memory before arriving and reads its peer CTA's after waiting, so every
/// phase must hold back each thread until the whole cluster has arrived.
/// Each row holds the block's linear index, its rank, the cluster size and
/// the values read in the three phases.
pub fn cluster_barriers_sm90(grid: [u32; 3], cluster: [u32; 3]) -> Case {
    const N: usize = 64;
    let blocks = grid.iter().product::<u32>() as usize;
    let name = match cluster {
        [2, 1, 1] => "cluster_barriers_2",
        [4, 1, 1] => "cluster_barriers_4",
        _ => "cluster_barriers_1x2x2",
    };
    let kernel = Kernel::builder(name).block_dim([N as u32, 1, 1]).build(move || {
        let staged = param_tensor::<u32>("staged", Layout::col_major(3 * blocks * N));
        let out = param_tensor::<u32>("out", Layout::row_major((blocks * N, 6)));
        let t = thread_idx().x;
        let (b, g) = (block_idx(), grid_dim());
        let linear = b.x + g.x * (b.y + g.y * b.z);
        let (rank, size) = (block_idx_in_cluster(), cluster_size());
        // Blocks of a cluster are consecutive here, ranked like the grid.
        let peer = linear - rank + (rank + 1) % size;
        let row = linear * N as i32 + t;
        let value = |phase: i32| (linear * 1000 + t + phase * 100_000).to::<u32>();
        let slot = |phase: i32, block: Int32| (phase * blocks as i32 + block) * N as i32 + t;
        out.store((row, 0), linear.to::<u32>());
        out.store((row, 1), rank.to::<u32>());
        out.store((row, 2), size.to::<u32>());
        // Phase 0: threads arrive one by one, from both sides of a branch.
        staged.store(slot(0, linear), value(0));
        if_else((t % 2).eq(0), cluster_arrive, cluster_arrive);
        cluster_wait();
        out.store((row, 3), staged.load(slot(0, peer)));
        // Phase 1: a relaxed aligned arrival orders nothing, so a fence
        // releases the write first.
        staged.store(slot(1, linear), value(1));
        fence_acq_rel_cluster();
        cluster_arrive_relaxed_aligned();
        cluster_wait_aligned();
        fence_acq_rel_cluster();
        out.store((row, 4), staged.load(slot(1, peer)));
        // Phase 2: cute::cluster_sync.
        staged.store(slot(2, linear), value(2));
        cluster_sync();
        out.store((row, 5), staged.load(slot(2, peer)));
    });
    fn check(args: &[KernelArg]) {
        let out = args[1].to_vec::<u32>();
        for (r, row) in out.chunks(6).enumerate() {
            let (t, linear, rank, size) = (r % N, row[0], row[1], row[2]);
            let peer = linear - rank + (rank + 1) % size;
            for phase in 0..3 {
                assert_eq!(row[3 + phase as usize], peer * 1000 + t as u32 + phase * 100_000, "row {r}");
            }
        }
    }
    let cfg = LaunchConfig::new(grid, [N as u32, 1, 1]).with_cluster(cluster);
    Case { name, kernel, cfg, small_only: false, check }
}

pub(super) fn launch_n(blocks: u32, threads: usize) -> LaunchConfig {
    LaunchConfig::new([blocks, 1, 1], [threads as u32, 1, 1])
}

/// `warpsize`, the warp constants, `warp_idx` and `make_warp_uniform` on a
/// value every lane of a warp loads from the same address.
pub fn warp_helpers() -> Case {
    let kernel = Kernel::builder("warp_helpers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, 4)));
        let t = thread_idx().x;
        let w = warp_idx();
        out.store((t, 0), warpsize());
        out.store((t, 1), w * WARPS_PER_WARPGROUP as i32 + THREADS_PER_WARPGROUP as i32);
        let shared = make_warp_uniform(xs.load(w * WARP_SIZE as i32));
        out.store((t, 2), shared);
        let flag =
            if_else_yield(make_warp_uniform(w * 3 + 1).eq(4), || (FULL_MASK as i32).into_value(), || 7.into_value());
        out.store((t, 3), flag);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let w = t / 32;
            let want = [32, (w * 4 + 128) as i32, xs[w * 32], if w == 1 { -1 } else { 7 }];
            assert_eq!(out[t * 4..t * 4 + 4], want, "thread {t}");
        }
    }
    Case { name: "warp_helpers", kernel, cfg: launch(1), small_only: false, check }
}

/// Words [`cp_async_mbarrier`] hands over each time: 16 bytes per producer lane.
const HANDOFF_WORDS: usize = 128;

/// Polls of an mbarrier phase after which [`bounded_spin`] gives up: far more than a phase that
/// completes needs (a spin that never succeeds gives up after 0.09 s at O3 and 1 s at O0 on an
/// SM89 GPU), so that only a lost or miscounted arrival gives up, which then fails the test
/// instead of hanging the GPU.
pub const SPIN_LIMIT: i32 = 1 << 22;

/// Spin on `mbarrier_test_wait(bar, parity)` until it succeeds or [`SPIN_LIMIT`] polls have
/// failed; returns whether it gave up. (In the simulator the spinning warp polls once per
/// scheduling round until the arrivals are in.)
pub(super) fn bounded_spin(bar: &Ptr<u64>, parity: Int32) -> Boolean {
    let (_, done) = while_carry(
        (Int32::from(0), Boolean::from(false)),
        |(polls, done)| !done & polls.lt(SPIN_LIMIT),
        |(polls, _)| (polls + 1, mbarrier_test_wait(bar, parity)),
    );
    !done
}

/// Check the waits of each thread that [`bounded_spin`] counted as given up.
pub(super) fn check_no_spin_gave_up(gave_up: &[u32]) {
    for (t, &n) in gave_up.iter().enumerate() {
        assert_eq!(n, 0, "thread {t} gave up waiting on {n} mbarrier phase(s) after {SPIN_LIMIT} polls each");
    }
}

/// Hand-offs from a producer warp to a consumer warp, each through an mbarrier of its own that
/// counts 32 arrivals, with the producer's `cp.async` copies tracked by `cp.async.mbarrier.arrive`
/// (CuTe DSL's `experimental/primitives/cp_async_mbarrier_arrive_cta.py`; the producer comes
/// first in program order). Each time, every producer lane copies 16 bytes into a fresh
/// buffer. The first hand-off tracks them with
/// `cp_async_mbarrier_arrive_noinc`, whose 32 arrivals are the count; the second with
/// `cp_async_mbarrier_arrive`, which raises the count by one first, followed by an ordinary
/// `mbarrier_arrive` per lane; the third like the first, but the producer then also waits for
/// its copies (`cp_async_wait_all`) and reads them back, so they land before the mbarrier sees
/// them. The consumer warp spins on `mbarrier_test_wait` for each and reads the copies.
///
/// Each mbarrier completes one phase. The producer never waits for the consumer, so on a GPU it
/// can finish every hand-off before the consumer first polls: with one mbarrier for all three,
/// its phase would be two ahead of the consumer's parity, which then names a phase that never
/// completes (the consumer would spin forever). [`cp_async_pipeline`] reuses its mbarriers,
/// behind an `empty` mbarrier that keeps the producer at most one phase ahead. The spins give up
/// after [`SPIN_LIMIT`] polls, and each thread stores how many did in `gave_up`.
pub fn cp_async_mbarrier() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_mbarrier").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(3 * W));
        let out = param_tensor::<u32>("out", Layout::col_major(4 * W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bars = alloc_smem::<u64>(3, 8);
        let stages: Vec<Tensor<u32>> = (0..3).map(|_| make_smem_tensor::<u32>(Layout::col_major(W))).collect();
        if_(t.eq(0), || {
            for k in 0..3usize {
                mbarrier_init(&bars.offset(k), 32);
            }
        });
        sync_threads();
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        // Each lane's own 4 words of a stage, into `out` from `at`.
        let read_back = |stage: &Tensor<u32>, at: usize| {
            for k in 0..4 {
                out.store(lane * 4 + k + at as i32, stage.load(lane * 4 + k));
            }
        };
        let misses = Var::new(Uint32::from(0u32));
        for (handoff, stage) in stages.iter().enumerate() {
            let bar = bars.offset(handoff);
            let source = make_tensor(&src.ptr().offset(handoff * W), Layout::col_major(W));
            if_else(
                warp_idx().eq(0),
                || {
                    let thr = tiled.get_slice(lane);
                    copy(&tiled, &thr.partition_s(&source), &thr.partition_d(stage));
                    match handoff {
                        0 => cp_async_mbarrier_arrive_noinc(&bar),
                        1 => {
                            cp_async_mbarrier_arrive(&bar);
                            mbarrier_arrive(&bar);
                        }
                        _ => {
                            cp_async_commit_group();
                            cp_async_mbarrier_arrive_noinc(&bar);
                            cp_async_wait_all();
                            read_back(stage, 3 * W);
                        }
                    }
                },
                || {
                    let missed = bounded_spin(&bar, Int32::from(0));
                    misses.update(|m| m + select(missed, 1u32, 0u32));
                    read_back(stage, handoff * W);
                },
            );
        }
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for i in 0..3 * W {
            assert_eq!(out[i], src[i], "hand-off {} word {} at the consumer", i / W, i % W);
        }
        for i in 0..W {
            assert_eq!(out[3 * W + i], src[2 * W + i], "hand-off 2 word {i} at the producer");
        }
    }
    Case { name: "cp_async_mbarrier", kernel, cfg: launch(1), small_only: false, check }
}

/// A hand-off whose consumer warp comes first in program order (and so
/// polls first in the simulator's schedules): warp 0 spins on
/// `mbarrier_test_wait` until warp 1's `cp.async` copies, tracked by
/// `cp_async_mbarrier_arrive_noinc`, have landed, then reads them. The
/// mbarrier counts the 32 deferred arrivals, which fire when each lane's
/// copies complete, after the consumer started waiting.
pub fn cp_async_consumer_first() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_consumer_first").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(W));
        let out = param_tensor::<u32>("out", Layout::col_major(W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let stage = make_smem_tensor::<u32>(Layout::col_major(W));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_spin(&bar, Int32::from(0));
                for k in 0..4 {
                    out.store(lane * 4 + k, stage.load(lane * 4 + k));
                }
                select(missed, 1u32, 0u32)
            },
            || {
                let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
                let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
                let thr = tiled.get_slice(lane);
                let source = make_tensor(&src.ptr(), Layout::col_major(W));
                copy(&tiled, &thr.partition_s(&source), &thr.partition_d(&stage));
                cp_async_mbarrier_arrive_noinc(&bar);
                // Done with the copies before exiting, as CUDA requires.
                cp_async_wait_all();
                Uint32::from(0u32)
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        assert_eq!(out, src, "the consumer reads what the producer copied");
    }
    Case { name: "cp_async_consumer_first", kernel, cfg: launch(1), small_only: false, check }
}

/// A warp-specialised bulk-copy pipeline: warp 0 produces, filling two
/// stages with `cp_async_bulk` behind `full` mbarriers (with
/// `mbarrier_arrive_and_expect_tx`), and warps 1 to 3 consume, each summing
/// its part of every stage and arriving on the stage's `empty` mbarrier,
/// which the producer waits on before refilling it.
pub fn bulk_copy_producer_warp() -> Case {
    const WORDS: usize = 96;
    const ROUNDS: usize = 6;
    let kernel = Kernel::builder("bulk_copy_producer_warp").block_dim([128, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(ROUNDS * WORDS));
        let out = param_tensor::<u32>("out", Layout::col_major(96));
        let t = thread_idx().x;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = alloc_smem::<u32>(2 * WORDS, 128);
        if_(t.eq(0), || {
            for k in 0..2usize {
                mbarrier_init(&full.offset(k), 1);
                mbarrier_init(&empty.offset(k), 96);
            }
            mbarrier_init_fence();
        });
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                for_range(0, ROUNDS as i32, 1, |r| {
                    let (stage, parity) = (r % 2, (r / 2) % 2);
                    // The consumers must have emptied the stage first.
                    if_(r.ge(2), || mbarrier_wait(&empty.offset(stage), 1 - parity));
                    if_(t.eq(0), || {
                        let bar = full.offset(stage);
                        mbarrier_arrive_and_expect_tx(&bar, (WORDS * 4) as i32);
                        cp_async_bulk(
                            &src.ptr().offset(r * WORDS as i32),
                            &stages.offset(stage * WORDS as i32),
                            (WORDS * 4) as i32,
                            &bar,
                        );
                    });
                });
            },
            || {
                let me = t - 32;
                let sum = for_range_carry(0, ROUNDS as i32, 1, Uint32::from(0u32), |r, sum| {
                    let (stage, parity) = (r % 2, (r / 2) % 2);
                    mbarrier_wait(&full.offset(stage), parity);
                    let word = make_tensor(&stages.offset(stage * WORDS as i32), Layout::col_major(WORDS)).load(me);
                    mbarrier_arrive(&empty.offset(stage));
                    sum + word
                });
                out.store(me, sum);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (i, &got) in out.iter().enumerate() {
            let want = (0..ROUNDS).fold(0u32, |acc, r| acc.wrapping_add(src[r * WORDS + i]));
            assert_eq!(got, want, "word {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    Case { name: "bulk_copy_producer_warp", kernel, cfg, small_only: false, check }
}

/// Iterations of [`cp_async_pipeline`]'s loop: each of its two stages is filled three times.
const PIPELINE_ITERS: usize = 6;

/// A two-stage producer/consumer pipeline in a runtime loop (the pattern of CuTe DSL's
/// `PipelineCpAsync`): each iteration, the producer warp waits until the consumer has emptied the
/// stage (its `empty` mbarrier), refills it with `cp.async` and makes the stage's `full` mbarrier
/// track the copies (`cp_async_mbarrier_arrive_noinc`); the consumer warp waits on `full`, adds
/// the stage into running sums kept in a `Var`, and arrives on `empty`. Phase parities flip every
/// second iteration, and each stage's buffer is reused only after the consumer has read it. The
/// spins give up after [`SPIN_LIMIT`] polls, and each thread stores how many did in `gave_up`.
pub fn cp_async_pipeline() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_pipeline").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(PIPELINE_ITERS * W));
        let out = param_tensor::<u32>("out", Layout::col_major(W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = alloc_smem::<u32>(2 * W, 128);
        if_(t.eq(0), || {
            for s in 0..2usize {
                mbarrier_init(&full.offset(s), 32);
                mbarrier_init(&empty.offset(s), 32);
            }
        });
        sync_threads();
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let sums = Var::new([Uint32::from(0u32); 4]);
        let misses = Var::new(Uint32::from(0u32));
        let spin = |bar: Ptr<u64>, parity: Int32| {
            let missed = bounded_spin(&bar, parity);
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        for_range(0, PIPELINE_ITERS as i32, 1, |i| {
            let stage = i % 2;
            let parity = (i / 2) % 2;
            // Offsets as index expressions, which know they are multiples of W.
            let buffer = make_tensor(&stages.offset(IndexExpr::from(stage) * W), Layout::col_major(W));
            if_else(
                warp_idx().eq(0),
                || {
                    // A fresh mbarrier counts the phase before its first as complete.
                    spin(empty.offset(stage), 1 - parity);
                    let source = make_tensor(&src.ptr().offset(IndexExpr::from(i) * W), Layout::col_major(W));
                    let thr = tiled.get_slice(lane);
                    copy(&tiled, &thr.partition_s(&source), &thr.partition_d(&buffer));
                    cp_async_mbarrier_arrive_noinc(&full.offset(stage));
                },
                || {
                    spin(full.offset(stage), parity);
                    sums.update(|s| std::array::from_fn(|k| s[k] + buffer.load(lane * 4 + k as i32)));
                    mbarrier_arrive(&empty.offset(stage));
                },
            );
        });
        if_(warp_idx().eq(1), || {
            let s = sums.get();
            for (k, v) in s.into_iter().enumerate() {
                out.store(lane * 4 + k as i32, v);
            }
        });
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for i in 0..W {
            let want = (0..PIPELINE_ITERS).fold(0u32, |acc, it| acc.wrapping_add(src[it * W + i]));
            assert_eq!(out[i], want, "word {i}");
        }
    }
    Case { name: "cp_async_pipeline", kernel, cfg: launch(1), small_only: false, check }
}

// Warp-specialised hand-offs whose waits come before the matching arrivals in program order.

/// Threads per block of [`consumer_first_named_barrier`]: four producer and four consumer warps.
const HANDOFF_BLOCK: usize = 256;

/// The consumer warps (4-7, the `then` branch, so first in program order) wait at named
/// barrier 1 of all 256 threads before the producer warps (0-3) have stored their words into
/// shared memory and arrived with `bar.arrive`; then each consumer reads a producer's word.
pub fn consumer_first_named_barrier() -> Case {
    const N: usize = HANDOFF_BLOCK;
    let kernel = Kernel::builder("consumer_first_named_barrier").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N / 2));
        let out = param_tensor::<u32>("out", Layout::col_major(N / 2));
        let smem = make_smem_tensor::<u32>(Layout::col_major(N / 2));
        let t = thread_idx().x;
        if_else(
            warp_idx().ge(4),
            || {
                barrier(1, N as i32);
                out.store(t - N as i32 / 2, smem.load((t + 32) % (N as i32 / 2)));
            },
            || {
                smem.store(t, xs.load(t) * 3u32 + 1u32);
                barrier_arrive(1, N as i32);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (c, &got) in out.iter().enumerate() {
            let p = (c + 32) % (N / 2);
            assert_eq!(got, xs[p].wrapping_mul(3).wrapping_add(1), "consumer thread {}", c + N / 2);
        }
    }
    Case { name: "consumer_first_named_barrier", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// Warp 0 spins on `mbarrier_test_wait` (the `then` branch, so before the producer in program
/// order) until warp 1 has stored its words into shared memory and arrived; then it reads them.
/// The spin gives up after [`SPIN_LIMIT`] polls, and each thread stores whether it did.
pub fn consumer_first_mbarrier() -> Case {
    let kernel = Kernel::builder("consumer_first_mbarrier").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_spin(&bar, 0.into_value());
                out.store(lane, data.load((lane + 1) % 32));
                select(missed, 1u32, 0u32)
            },
            || {
                data.store(lane, xs.load(lane) ^ 0x5a5a_5a5au32);
                mbarrier_arrive(&bar);
                0u32.into_value()
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            assert_eq!(got, xs[(l + 1) % 32] ^ 0x5a5a_5a5a, "lane {l}");
        }
    }
    Case { name: "consumer_first_mbarrier", kernel, cfg: launch(1), small_only: false, check }
}

/// Iterations of [`mbarrier_ring`]'s loop: each of its two stages is filled four times.
const RING_ITERS: usize = 8;

/// A two-stage ring of shared-memory buffers between a consumer warp and a producer warp, each
/// stage behind a `full` and an `empty` mbarrier of 32 arrivals, over a runtime loop. The
/// consumer (warp 0) comes first in program order: each iteration it waits until the stage is
/// full, adds it into a running sum and arrives on `empty`; the producer (warp 1) waits until
/// the stage is empty, stores the iteration's words and arrives on `full`. Phase parities flip
/// every second iteration. The spins give up after [`SPIN_LIMIT`] polls, and each thread stores
/// how many did in `gave_up`.
pub fn mbarrier_ring() -> Case {
    let kernel = Kernel::builder("mbarrier_ring").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(RING_ITERS * 32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = make_smem_tensor::<u32>(Layout::col_major(64));
        if_(t.eq(0), || {
            for s in 0..2usize {
                mbarrier_init(&full.offset(s), 32);
                mbarrier_init(&empty.offset(s), 32);
            }
        });
        sync_threads();
        let sum = Var::new(Uint32::from(0u32));
        let misses = Var::new(Uint32::from(0u32));
        let spin = |bar: Ptr<u64>, parity: Int32| {
            let missed = bounded_spin(&bar, parity);
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        for_range(0, RING_ITERS as i32, 1, |i| {
            let stage = i % 2;
            let parity = (i / 2) % 2;
            if_else(
                warp_idx().eq(0),
                || {
                    spin(full.offset(stage), parity);
                    sum.update(|s| s + stages.load(stage * 32 + (lane + i) % 32));
                    mbarrier_arrive(&empty.offset(stage));
                },
                || {
                    // A fresh mbarrier counts the phase before its first as complete.
                    spin(empty.offset(stage), 1 - parity);
                    stages.store(stage * 32 + lane, src.load(i * 32 + lane) + i.to::<u32>());
                    mbarrier_arrive(&full.offset(stage));
                },
            );
        });
        if_(warp_idx().eq(0), || out.store(lane, sum.get()));
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            let want = (0..RING_ITERS)
                .fold(0u32, |acc, i| acc.wrapping_add(src[i * 32 + (l + i) % 32].wrapping_add(i as u32)));
            assert_eq!(got, want, "lane {l}");
        }
    }
    Case { name: "mbarrier_ring", kernel, cfg: launch(1), small_only: false, check }
}

/// [`bounded_spin`] polling with `mbarrier_try_wait` (SM90 and newer), whose thread can be held
/// back for a time-limit before the poll gives up and returns false.
fn bounded_try_spin(bar: &Ptr<u64>, parity: Int32) -> Boolean {
    let (_, done) = while_carry(
        (Int32::from(0), Boolean::from(false)),
        |(polls, done)| !done & polls.lt(SPIN_LIMIT),
        |(polls, _)| (polls + 1, mbarrier_try_wait(bar, parity)),
    );
    !done
}

/// [`consumer_first_mbarrier`] with `mbarrier_try_wait`: warp 0 waits on the mbarrier before
/// warp 1 has stored its words and arrived, and then reads them.
pub fn consumer_first_try_wait() -> Case {
    let kernel = Kernel::builder("consumer_first_try_wait").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_try_spin(&bar, 0.into_value());
                out.store(lane, data.load((lane + 3) % 32));
                select(missed, 1u32, 0u32)
            },
            || {
                data.store(lane, xs.load(lane) + 7u32);
                mbarrier_arrive(&bar);
                0u32.into_value()
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            assert_eq!(got, xs[(l + 3) % 32].wrapping_add(7), "lane {l}");
        }
    }
    Case { name: "consumer_first_try_wait", kernel, cfg: launch(1), small_only: false, check }
}

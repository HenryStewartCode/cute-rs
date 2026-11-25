//! Seeded schedules, the schedule-dependence and phase-aliasing checks, and
//! independent thread scheduling (sub-warp convergence groups): planted bugs
//! that only some interleavings show, and divergent warps whose lanes wait
//! for each other.

use cute::prelude::*;

/// `CUTE_SIM_SEEDS` seeds (20 by default).
fn seeds() -> Vec<u64> {
    let n = std::env::var("CUTE_SIM_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    (0..n).collect()
}

fn one_block(threads: u32) -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [threads, 1, 1])
}

/// The same seed replays the same run, and the report records it.
#[test]
fn a_seed_replays_its_run() {
    let k = Kernel::builder("replay").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(4));
        let t = thread_idx().x;
        if_(lane_idx().eq(0), || {
            let old = atomic_add(&out.ptr(), 1u32);
            out.store(warp_idx() + 1, old);
        });
        let _ = t;
    });
    let run = |seed: u64| {
        let opts = SimOptions { schedule: Schedule::Random { seed }, ..SimOptions::default() };
        let mut args = vec![KernelArg::buffer(&[0u32; 4])];
        let report = k.simulate_with(one_block(64), &mut args, &opts).unwrap_or_else(|e| panic!("{e}"));
        (args[0].to_vec::<u32>(), report.seed, report.steps)
    };
    for seed in seeds() {
        let (out, recorded, steps) = run(seed);
        assert_eq!(recorded, Some(seed));
        assert_eq!(run(seed), (out, recorded, steps), "seed {seed} does not replay");
    }
    // Some seed lets warp 1 take the counter first.
    assert!(seeds().into_iter().any(|seed| run(seed).0[2] == 0), "no seed reorders the atomics");
}

/// An `atomic_cas` race whose winner writes its id: the output depends on
/// the schedule, which `simulate_schedules` reports with the element and the
/// two schedules.
#[test]
fn atomic_cas_race_is_schedule_dependent() {
    let k = Kernel::builder("cas_race").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(2));
        let me = warp_idx().to::<u32>() + 1u32;
        if_(lane_idx().eq(0), || {
            let old = atomic_cas(&out.ptr(), 0u32, me);
            if_(old.eq(0u32), || out.store(1, me));
        });
    });
    let mut args = vec![KernelArg::buffer(&[0u32; 2])];
    // The built-in schedules agree (warp 0 wins under both).
    k.simulate_all_orders(one_block(64), &mut args.clone(), &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    let e = k.simulate_schedules(one_block(64), &mut args, &SimOptions::default(), &seeds()).unwrap_err();
    assert!(
        e.message.contains(
            "`out`[0] (argument 0) is 1 under Schedule::RoundRobin with AsyncCompletion::Late and 2 under \
             Schedule::Random { seed: "
        ),
        "{e}"
    );
    assert!(e.message.contains("depend on how its warps interleave"), "{e}");
}

/// A producer that hands three phases through one mbarrier without waiting
/// for an empty barrier: a schedule that runs it ahead shows the parity
/// aliasing (the README's three hand-offs that deadlocked on an SM89 GPU).
fn no_back_pressure() -> Kernel {
    Kernel::builder("no_back_pressure").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(3 * 32));
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(3 * 32));
        let lane = lane_idx();
        if_(thread_idx().x.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                for k in 0..3 {
                    data.store(lane + k * 32, lane.to::<u32>() + k as u32 * 100);
                    mbarrier_arrive(&bar);
                }
            },
            || {
                for k in 0..3 {
                    mbarrier_wait(&bar, k & 1);
                    out.store(lane + k * 32, data.load(lane + k * 32));
                }
            },
        );
    })
}

#[test]
fn phase_aliasing_is_reported() {
    let k = no_back_pressure();
    let opts = SimOptions { schedule: Schedule::RunToBlock, ..SimOptions::default() };
    let e = k.simulate_with(one_block(64), &mut [KernelArg::buffer(&[0u32; 96])], &opts).unwrap_err();
    assert!(e.message.contains("phase aliasing"), "{e}");
    assert!(e.message.contains("the mbarrier has completed 3 phases, but the warp has seen only 0 complete"), "{e}");
    assert!(e.to_string().contains("tests/schedules.rs"), "{e}");
    // Seeds find it too.
    let e = k
        .simulate_schedules(one_block(64), &mut [KernelArg::buffer(&[0u32; 96])], &SimOptions::default(), &seeds())
        .unwrap_err();
    assert!(e.message.contains("phase aliasing") || e.message.contains("has not seen phase"), "{e}");
}

/// Producer warp 0 copies with `cp.async` and makes the mbarrier track the
/// copies; after a `sync_threads`, consumer warp 1 polls once with
/// `mbarrier_test_wait` and reads whatever the result. Late completion
/// lands the copies at the poll, early completion at issue: only random
/// completion shows the copies still in flight.
fn polls_once(check_result: bool) -> Kernel {
    Kernel::builder("polls_once").block_dim([64, 1, 1]).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(128));
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        let bar = alloc_smem::<u64>(1, 8);
        let stage = make_smem_tensor::<u32>(Layout::col_major(128));
        let t = thread_idx().x;
        let lane = lane_idx();
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        if_(warp_idx().eq(0), || {
            let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
            let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
            let thr = tiled.get_slice(lane);
            copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&stage));
            cp_async_mbarrier_arrive_noinc(&bar);
        });
        sync_threads();
        if_(warp_idx().eq(1), || {
            let ready = mbarrier_test_wait(&bar, 0);
            if check_result {
                // Spin until the phase completes, then read.
                while_carry(ready, |r| !r, |_| mbarrier_test_wait(&bar, 0));
            }
            for k in 0..4 {
                out.store(lane * 4 + k, stage.load(lane * 4 + k));
            }
        });
        // Nobody exits with the copies in flight.
        if_(warp_idx().eq(0), || mbarrier_wait_spin(&bar));
    })
}

/// Warp 0 waits for the phase before it exits (SM80: spin on test_wait).
fn mbarrier_wait_spin(bar: &Ptr<u64>) {
    while_carry(Boolean::from(false), |done| !done, |_| mbarrier_test_wait(bar, 0));
}

#[test]
fn premature_read_is_caught_only_by_random_completion() {
    let src: Vec<u32> = (0..128).collect();
    let args = || vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 128])];
    polls_once(false)
        .simulate_all_orders(one_block(64), &mut args(), &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let e =
        polls_once(false).simulate_schedules(one_block(64), &mut args(), &SimOptions::default(), &seeds()).unwrap_err();
    assert!(e.message.contains("cp.async"), "{e}");
    assert!(e.to_string().contains("Schedule::Random { seed: "), "{e}");
    // The failure names its seed, and simulating again with it replays it.
    let seed = e.seed.expect("a seeded failure");
    let opts = SimOptions {
        schedule: Schedule::Random { seed },
        async_completion: AsyncCompletion::Random(seed),
        ..SimOptions::default()
    };
    let again = polls_once(false).simulate_with(one_block(64), &mut args(), &opts).unwrap_err();
    assert_eq!((again.seed, again.steps, again.loc), (e.seed, e.steps, e.loc));
    assert!(e.message.ends_with(&again.message), "{e}\n{again}");
    // Waiting for the phase is correct under every seed.
    let mut a = args();
    polls_once(true)
        .simulate_schedules(one_block(64), &mut a, &SimOptions::default(), &seeds())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a[1].to_vec::<u32>(), src);
}

/// Independent thread scheduling: lane 0 waits on an mbarrier that lanes
/// 1 to 31 of its own warp arrive at in the else-branch (a deadlock on a
/// SIMT stack that runs the then-branch first), by blocking `try_wait` and
/// by a `test_wait` spin.
#[test]
fn a_lane_waits_for_lanes_of_its_own_warp() {
    for spin in [false, true] {
        let k = Kernel::builder("its_mbarrier").block_dim([32, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(32));
            let bar = alloc_smem::<u64>(1, 8);
            let data = make_smem_tensor::<u32>(Layout::col_major(32));
            let lane = lane_idx();
            if_(lane.eq(0), || mbarrier_init(&bar, 31));
            sync_warp();
            if_else(
                lane.eq(0),
                || {
                    if spin {
                        mbarrier_wait_spin(&bar);
                    } else {
                        mbarrier_wait(&bar, 0);
                    }
                    let sum = (1..32).fold(Uint32::from(0u32), |acc, l| acc + data.load(l));
                    out.store(0, sum);
                },
                || {
                    data.store(lane, lane.to::<u32>() * 2u32);
                    mbarrier_arrive(&bar);
                },
            );
        });
        let mut args = vec![KernelArg::buffer(&[0u32; 32])];
        k.simulate_schedules(one_block(32), &mut args, &SimOptions::default(), &seeds())
            .unwrap_or_else(|e| panic!("spin {spin}: {e}"));
        assert_eq!(args[0].to_vec::<u32>()[0], (1..32).map(|l| 2 * l).sum::<u32>());
    }
}

/// `barrier.arrive` without `.aligned` from divergent lanes of warp 0, at two
/// instructions: the warp's arrival counts once all its lanes have arrived,
/// and warp 1's `bar.sync` then sees warp 0's writes.
#[test]
fn divergent_unaligned_arrivals_complete() {
    let build = |all: bool| {
        Kernel::builder("unaligned").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(32));
            let data = make_smem_tensor::<u32>(Layout::col_major(32));
            let lane = lane_idx();
            if_else(
                warp_idx().eq(0),
                || {
                    data.store(lane, lane.to::<u32>() + 5u32);
                    if all {
                        if_else(lane.lt(16), || barrier_arrive_unaligned(1, 64), || barrier_arrive_unaligned(1, 64));
                    } else {
                        if_(lane.lt(16), || barrier_arrive_unaligned(1, 64));
                    }
                },
                || {
                    barrier(1, 64);
                    out.store(lane, data.load(lane));
                },
            );
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 32])];
    build(true)
        .simulate_schedules(one_block(64), &mut args, &SimOptions::default(), &seeds())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>(), (5..37).collect::<Vec<u32>>());
    // Half the warp arriving is no arrival: warp 1 waits forever.
    let e = build(false).simulate(one_block(64), &mut [KernelArg::buffer(&[0u32; 32])]).unwrap_err();
    assert!(e.message.contains("deadlock") && e.message.contains("warp 1 of block (0,0,0) waits at barrier 1"), "{e}");
}

/// A `shfl.sync` whose member mask spans both branches of a divergent `if`:
/// the two groups meet (at different instructions, as sm_70 allows) and it
/// runs once for the whole mask.
#[test]
fn shuffle_rendezvous_across_groups() {
    let k = Kernel::builder("rendezvous").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let lane = lane_idx();
        let v = x.load(lane);
        let r = if_else_yield(lane.lt(16), || shuffle_sync_bfly(v, 16) + 100u32, || shuffle_sync_bfly(v, 16));
        out.store(lane, r);
    });
    let x: Vec<u32> = (0..32).map(|l| l * 3).collect();
    let mut args = vec![KernelArg::buffer(&x), KernelArg::buffer(&[0u32; 32])];
    k.simulate_schedules(one_block(32), &mut args, &SimOptions::default(), &seeds()).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<u32> = (0..32u32).map(|l| (l ^ 16) * 3 + if l < 16 { 100 } else { 0 }).collect();
    assert_eq!(args[1].to_vec::<u32>(), want);
}

/// Shuffles of different values defined before the `if` meet across its
/// branches: each lane sends its own branch's operand, and the values keep
/// what they held after the branches rejoin.
#[test]
fn rendezvous_keeps_each_branchs_operands() {
    let k = Kernel::builder("rendezvous_operands").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let lane = lane_idx();
        let a = x.load(lane);
        let b = a + 1000u32;
        let r = if_else_yield(lane.lt(16), || shuffle_sync_bfly(a, 16), || shuffle_sync_bfly(b, 16));
        out.store(lane, r + a * 10000u32 + b * 100000u32);
    });
    let x: Vec<u32> = (0..32).map(|l| l * 3).collect();
    let mut args = vec![KernelArg::buffer(&x), KernelArg::buffer(&[0u32; 32])];
    k.simulate_schedules(one_block(32), &mut args, &SimOptions::default(), &seeds()).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<u32> = (0..32u32)
        .map(|l| {
            let (a, b) = (3 * l, 3 * l + 1000);
            let src = l ^ 16;
            let r = if l < 16 { 3 * src + 1000 } else { 3 * src };
            r.wrapping_add(a.wrapping_mul(10000)).wrapping_add(b.wrapping_mul(100000))
        })
        .collect();
    assert_eq!(args[1].to_vec::<u32>(), want);
}

/// Lanes that left a loop wait at its end: a full-mask vote in a later
/// iteration is executed without them, which is reported.
#[test]
fn collective_after_lanes_leave_a_loop_is_reported() {
    let k = Kernel::builder("loop_vote").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let lane = lane_idx();
        for_range(0, lane + 1, 1, |i| out.store(lane, vote_ballot_sync(lane.eq(i))));
    });
    let e = k.simulate(one_block(32), &mut [KernelArg::buffer(&[0u32; 32])]).unwrap_err();
    assert!(e.message.contains("lane 0 of the mask is not active here"), "{e}");
    assert!(e.message.contains("waits at the end of a branch or loop"), "{e}");
}

/// `mma.sync` is `.aligned`: a convergence group with part of the warp
/// executing it is reported.
#[test]
fn divergent_mma_sync_is_reported() {
    let k = Kernel::builder("divergent_mma").block_dim([32, 1, 1]).build(|| {
        let d = param_tensor::<f32>("d", Layout::row_major((16, 8)));
        let mma = make_tiled_mma(make_mma_atom(MmaOp::Sm80(MmaKind::M16N8K16F32F16)), Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let a = make_smem_tensor::<F16>(Layout::row_major((16, 16)));
        let b = make_smem_tensor::<F16>(Layout::col_major((8, 16)));
        let fa = thr.partition_fragment_a::<F16, _>(&a);
        let fb = thr.partition_fragment_b::<F16, _>(&b);
        let acc = thr.partition_fragment_c::<f32, _>(&d);
        fa.fill(F16::from_f32(1.0));
        fb.fill(F16::from_f32(1.0));
        acc.fill(0.0);
        if_(lane_idx().lt(16), || gemm(&mma, &acc, &fa, &fb, &acc));
    });
    let e = k.simulate(one_block(32), &mut [KernelArg::f32(vec![0.0; 128])]).unwrap_err();
    assert!(e.message.contains("16 of the 32 lanes of warp 0"), "{e}");
}

/// A deadlock inside a divergent warp names each group's lanes.
#[test]
fn deadlocks_name_the_lanes_of_each_group() {
    let k = Kernel::builder("its_deadlock").block_dim([32, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        let lane = lane_idx();
        if_(lane.eq(0), || mbarrier_init(&bar, 32));
        sync_warp();
        // Lane 0 waits for 32 arrivals, but only lanes 1..=31 arrive.
        if_else(lane.eq(0), || mbarrier_wait(&bar, 0), || mbarrier_arrive(&bar));
    });
    let e = k.simulate(one_block(32), &mut []).unwrap_err();
    assert!(e.message.contains("warp 0 (lane 0) of block (0,0,0) waits at mbarrier.try_wait"), "{e}");
    assert!(e.message.contains("still waits for 1 of its 32 arrival(s)"), "{e}");
}

/// The pipeline mistakes CuTe's pipeline documentation warns about, in a
/// producer/consumer ring of two stages like `pipeline_software_fill`'s
/// (full and empty mbarriers of one arrival, the empty ones pre-arrived).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mistake {
    None,
    /// The consumer never arrives on `empty`.
    MissingRelease,
    /// The consumer waits on parity 1 of a fresh `full` mbarrier, which
    /// passes at once (the producer's start state applied to the consumer).
    WrongInitialPhase,
    /// `full` counts one arrival, and the producer arrives twice per phase.
    ArriveTwice,
    /// The consumer first spins on a flag nobody sets.
    FlagNobodySets,
}

fn pipeline(mistake: Mistake) -> Kernel {
    const STAGES: i32 = 2;
    Kernel::builder("pipeline").block_dim([64, 1, 1]).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(4 * 32));
        let dst = param_tensor::<u32>("dst", Layout::col_major(4 * 32));
        let buf = make_smem_tensor::<u32>(Layout::col_major(2 * 32));
        let mbar = alloc_smem::<u64>(4, 8);
        let flag = alloc_smem::<u32>(1, 4);
        let (full, empty) = (|s: Int32| mbar.offset(s), |s: Int32| mbar.offset(s + STAGES));
        let wait = |bar: &Ptr<u64>, parity: Int32| while_carry((), |()| !mbarrier_try_wait(bar, parity), |()| ());
        let lane = lane_idx();
        if_(thread_idx().x.eq(0), || {
            for s in 0..STAGES {
                mbarrier_init(&full(s.into_value()), 1);
                mbarrier_init(&empty(s.into_value()), 1);
            }
            flag.store(0u32);
        });
        sync_threads();
        if_(thread_idx().x.eq(0), || (0..STAGES).for_each(|s| mbarrier_arrive(&empty(s.into_value()))));
        sync_threads();
        if_(warp_idx().eq(0), || {
            for_range(0, 4, 1, |k| {
                let (s, phase) = (k % STAGES, (k / STAGES) & 1);
                wait(&empty(s), phase);
                buf.store(s * 32 + lane, src.load(k * 32 + lane));
                sync_warp();
                if_(elect_one(), || {
                    mbarrier_arrive(&full(s));
                    if mistake == Mistake::ArriveTwice {
                        mbarrier_arrive(&full(s));
                    }
                });
            });
        });
        if_(warp_idx().eq(1), || {
            if mistake == Mistake::FlagNobodySets {
                let acquire =
                    LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Cta), ..Default::default() };
                while_carry((), |()| load(&flag, acquire).eq(0u32), |()| ());
            }
            for_range(0, 4, 1, |k| {
                let (s, phase) = (k % STAGES, (k / STAGES) & 1);
                let parity = if mistake == Mistake::WrongInitialPhase { phase ^ 1 } else { phase };
                wait(&full(s), parity);
                dst.store(k * 32 + lane, buf.load(s * 32 + lane));
                sync_warp();
                if mistake != Mistake::MissingRelease {
                    if_(elect_one(), || mbarrier_arrive(&empty(s)));
                }
            });
        });
    })
}

#[test]
fn pipeline_mistakes_are_reported() {
    let src: Vec<u32> = (0..128).collect();
    let args = || vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 128])];
    let opts = SimOptions { max_idle_steps: 1 << 14, ..SimOptions::default() };
    let run = |m: Mistake| pipeline(m).simulate_schedules(one_block(64), &mut args(), &opts, &seeds());
    let mut a = args();
    pipeline(Mistake::None)
        .simulate_schedules(one_block(64), &mut a, &opts, &seeds())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a[1].to_vec::<u32>(), src);
    // A missing release: the producer waits for a stage forever.
    let e = run(Mistake::MissingRelease).unwrap_err();
    assert!(e.message.contains("(a deadlock)"), "{e}");
    assert!(e.message.contains("warp 0 of block (0,0,0) waits at mbarrier.try_wait"), "{e}");
    assert!(
        e.message.contains("in for_range at tests/schedules.rs") && e.message.contains("on the mbarrier smem1[2]"),
        "{e}"
    );
    // A wrong initial phase: the consumer reads a stage nobody has filled.
    let e = run(Mistake::WrongInitialPhase).unwrap_err();
    assert!(e.message.contains("data race: warp 1 reads smem0[0], written by warp 0"), "{e}");
    // Two arrivals per phase of a one-arrival mbarrier: it runs ahead.
    let e = run(Mistake::ArriveTwice).unwrap_err();
    assert!(e.message.contains("the mbarrier has completed 2 phases") && e.message.contains("phase aliasing"), "{e}");
    // A spin on a flag nobody sets: a hang that names the poll.
    let e = run(Mistake::FlagNobodySets).unwrap_err();
    assert!(e.message.contains("(a hang)") && e.message.contains("polling smem2[0] with an ordered load"), "{e}");
}

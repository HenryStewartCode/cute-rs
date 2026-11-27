//! Named barriers, mbarriers, bulk-copy transaction counts, cluster barriers, deadlock and hang reports, and the review regressions on them.

use super::*;
/// Warps 0-3 exchange values in shared memory, ordered by `order`.
fn warp_group_exchange(order: fn()) -> Kernel {
    Kernel::builder("warp_group_exchange").block_dim([256, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(256));
        let smem = make_smem_tensor::<u32>(Layout::col_major(128));
        let t = thread_idx().x;
        if_(warp_idx().lt(4), || {
            smem.store(t, t.to::<u32>());
            order();
            out.store(t, smem.load((t + 32) % 128));
        });
    })
}

#[test]
fn named_barrier_orders_a_subset_of_warps() {
    let mut args = vec![KernelArg::buffer(&[0u32; 256])];
    let report = warp_group_exchange(|| barrier(1, 128))
        .simulate(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.barriers, 1);
    assert_eq!(args[0].to_vec::<u32>()[0], 32);
    // Without it the other warp's write is a race.
    let e = simulate_err(&warp_group_exchange(|| {}), 256, &mut [KernelArg::buffer(&[0u32; 256])]);
    assert!(e.message.contains("data race"), "{e}");
}

#[test]
fn named_barrier_does_not_order_warps_outside_it() {
    // Warps 0-3 write, sync among themselves only, and warps 4-7 read.
    let kernel = Kernel::builder("outside").block_dim([256, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(256));
        let smem = make_smem_tensor::<u32>(Layout::col_major(128));
        let t = thread_idx().x;
        if_(warp_idx().lt(4), || {
            smem.store(t, t.to::<u32>());
            barrier(1, 128);
        });
        if_(warp_idx().ge(4), || out.store(t, smem.load(t - 128)));
    });
    let e = simulate_err(&kernel, 256, &mut [KernelArg::buffer(&[0u32; 256])]);
    assert!(e.message.contains("data race: warp 4 reads"), "{e}");
}

#[test]
fn named_barrier_whose_consumers_wait_first() {
    // The consumers wait first; the producers store and arrive later in program order.
    let kernel = |producers_arrive: bool| {
        Kernel::builder("late_arrival").block_dim([256, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(256));
            let smem = make_smem_tensor::<u32>(Layout::col_major(128));
            let t = thread_idx().x;
            if_else(
                warp_idx().ge(4),
                || {
                    barrier(1, 256);
                    out.store(t, smem.load(t - 128));
                },
                || {
                    smem.store(t, t.to::<u32>() + 1u32);
                    if producers_arrive {
                        barrier_arrive(1, 256);
                    }
                },
            );
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 256])];
    let report =
        kernel(true).simulate(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.barriers, 1);
    assert_eq!(args[0].to_vec::<u32>()[128..131], [1, 2, 3]);
    // Without the producers' arrivals the consumers wait forever.
    let e = simulate_err(&kernel(false), 256, &mut [KernelArg::buffer(&[0u32; 256])]);
    assert!(e.message.contains("did not terminate") && e.message.contains("deadlock"), "{e}");
    assert!(e.message.contains("warps 4..=7 of block (0,0,0) wait at barrier 1"), "{e}");
    assert!(e.message.contains("needs 8 warps") && e.message.contains("warps 0..=3 have exited"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/barriers.rs"), "{e}");
    // The same hand-off in the other order completes.
    let kernel = Kernel::builder("early_arrival").block_dim([256, 1, 1]).build(|| {
        let w = warp_idx();
        if_else(w.lt(4), || barrier_arrive(1, 256), || barrier(1, 256));
    });
    kernel.simulate(LaunchConfig::new([1, 1, 1], [256, 1, 1]), &mut []).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn named_barrier_misuse_in_the_simulator() {
    let run = |f: fn()| simulate_err(&Kernel::builder("misuse").block_dim([256, 1, 1]).build(f), 256, &mut []);
    let e = run(|| barrier(1, 128));
    assert!(e.message.contains("8 warps arrive at barrier 1, which completes after 4"), "{e}");
    let e = run(|| if_(lane_idx().lt(16), || barrier(1, 64)));
    assert!(e.message.contains("16 of the 32 lanes of warp 0"), "{e}");
    let e = run(|| barrier(warp_idx() + 12, 256));
    assert!(e.message.contains("barrier id 16 is out of range"), "{e}");
    let e = run(|| barrier(1, warp_idx() / 4 * 32 + 96));
    assert!(e.message.contains("disagree on its thread count"), "{e}");
    let e = run(|| {
        if_(warp_idx().lt(2), || {
            barrier_arrive(1, 256);
            barrier_arrive(1, 256);
        })
    });
    assert!(e.message.contains("warp 0 arrives at barrier 1 again"), "{e}");
    let e = run(|| {
        if_(warp_idx().lt(2), || barrier_arrive(0, 256));
        sync_threads();
    });
    assert!(e.message.contains("sync_threads() uses barrier 0 while warps 0..=1"), "{e}");
}

#[test]
fn named_barrier_operands_are_validated() {
    let msg = build_panic(|| {
        Kernel::build("bad_id", || barrier(16, 64));
    });
    assert!(msg.contains("barrier id 16 is out of range"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bad_count", || barrier_arrive(1, 48));
    });
    assert!(msg.contains("48 is not a positive multiple of 32"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("too_many").block_dim([128, 1, 1]).build(|| barrier(1, 256));
    });
    assert!(msg.contains("256 threads is more than the block has (128)"), "{msg}");
    let msg = build_panic(|| {
        pipeline::NamedBarrier::new(16, 64);
    });
    assert!(msg.contains("NamedBarrier id 16 is out of range"), "{msg}");
    let msg = build_panic(|| {
        pipeline::NamedBarrier::new(1, 100);
    });
    assert!(msg.contains("multiple of 32"), "{msg}");
}

#[derive(Clone, Copy, PartialEq)]
enum Handoff {
    Correct,
    /// Warp 0 writes after its arrival.
    WriteAfterArrival,
    /// Warp 0 arrives with relaxed semantics, which release nothing.
    RelaxedArrival,
}

/// Warp 0 writes shared memory and arrives at an mbarrier; warp 1 waits on
/// it and reads.
fn mbarrier_handoff(mode: Handoff) -> Kernel {
    Kernel::builder("mbarrier_handoff").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let bar = alloc_smem::<u64>(1, 8);
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                if mode != Handoff::WriteAfterArrival {
                    smem.store(t, t.to::<u32>());
                }
                if mode == Handoff::RelaxedArrival {
                    mbarrier_arrive_with(&bar, 1, MbarrierOptions { relaxed: true, ..Default::default() });
                } else {
                    mbarrier_arrive(&bar);
                }
                if mode == Handoff::WriteAfterArrival {
                    smem.store(t, t.to::<u32>());
                }
            },
            || {
                while_carry((), |()| !mbarrier_test_wait(&bar, 0), |()| ());
                out.store(t, smem.load(t - 32));
            },
        );
    })
}

#[test]
fn mbarrier_orders_a_handoff() {
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    mbarrier_handoff(Handoff::Correct)
        .simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[40], 8);
    // A write after the arrival is not covered by the wait, and a relaxed
    // arrival orders nothing.
    for mode in [Handoff::WriteAfterArrival, Handoff::RelaxedArrival] {
        let e = simulate_err(&mbarrier_handoff(mode), 64, &mut [KernelArg::buffer(&[0u32; 64])]);
        assert!(e.message.contains("data race: warp 1 reads"), "{e}");
    }
}

/// `mbarriers_sm80`'s two-way hand-off, optionally without its fix: warps
/// 0-1 arrive in phase 0 and then wait on parity 1 for warps 2-3's data.
/// While phase 0 is still running on hardware, parity 1 names the phase
/// before it, which counts as complete, so the wait passes at once; the
/// simulator's schedule need not show that, so it must reject a wait whose
/// warp has not seen the previous phase complete.
fn two_way_handoff(producers_wait: bool) -> Kernel {
    Kernel::builder("two_way_handoff").block_dim([128, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        let t = thread_idx().x;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(128));
        let spin = |parity: i32| while_carry((), |()| !mbarrier_test_wait(&bar, parity), |()| ());
        if_(t.eq(0), || mbarrier_init(&bar, 64));
        sync_threads();
        if_else(
            warp_idx().lt(2),
            || {
                data.store(t, t.to::<u32>());
                mbarrier_arrive(&bar);
                if producers_wait {
                    spin(0);
                }
            },
            || {
                spin(0);
                out.store(t, data.load(t - 64));
            },
        );
        if_else(
            warp_idx().ge(2),
            || {
                data.store(t, t.to::<u32>() * 2u32);
                mbarrier_arrive(&bar);
            },
            || {
                spin(1);
                out.store(t, data.load(t + 64));
            },
        );
    })
}

#[test]
fn mbarrier_parity_needs_the_previous_phase() {
    let mut args = vec![KernelArg::buffer(&[0u32; 128])];
    two_way_handoff(true)
        .simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[..2], [128, 130]);
    let e = simulate_err(&two_way_handoff(false), 128, &mut [KernelArg::buffer(&[0u32; 128])]);
    assert!(
        e.message.contains("mbarrier.test_wait with parity 1 asks about phase 1 of the mbarrier, but warp 0 has not seen phase 0 complete"),
        "{e}"
    );
    assert!(e.to_string().contains("tests/diagnostics/barriers.rs"), "{e}");
    // What one warp has seen, warps ordered after it know too: warp 0 alone
    // waits for phase 0, and a barrier then tells warp 1.
    let shared = |tell: bool| {
        Kernel::builder("shared_knowledge").block_dim([64, 1, 1]).build(move || {
            let bar = alloc_smem::<u64>(1, 8);
            let spin = |parity: i32| while_carry((), |()| !mbarrier_test_wait(&bar, parity), |()| ());
            if_(thread_idx().x.eq(0), || mbarrier_init(&bar, 32));
            sync_threads();
            if_(warp_idx().eq(0), || {
                mbarrier_arrive(&bar);
                spin(0);
            });
            if tell {
                sync_threads();
            }
            if_(warp_idx().eq(0), || mbarrier_arrive(&bar));
            spin(1);
        })
    };
    shared(true).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut []).unwrap_or_else(|e| panic!("{e}"));
    let e = simulate_err(&shared(false), 64, &mut []);
    assert!(e.message.contains("warp 1 has not seen phase 0 complete"), "{e}");
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tx {
    Correct,
    /// Thread 1 issues its copy without `mbarrier_expect_tx`.
    MissingExpect,
    /// Thread 1 expects only half its copy.
    TooSmall,
    /// Neither copy is expected: the last arrival completes the phase.
    NoExpect,
    /// Both arrivals come first, so the phase is over before the copy.
    CopyAfterPhase,
    /// Only warp 0 waits on the mbarrier, but both warps read the data.
    OneWarpWaits,
}

/// Two bulk copies of 64 bytes complete an mbarrier phase that two threads
/// arrive at (an SM90 kernel, checked in the simulator).
fn bulk_copies(mode: Tx) -> Kernel {
    Kernel::builder("bulk_copies").block_dim([64, 1, 1]).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let bar = alloc_smem::<u64>(1, 8);
        let data = alloc_smem::<u32>(32, 128);
        if_(t.eq(0), || mbarrier_init(&bar, 2));
        sync_threads();
        if_(t.eq(1), || {
            match mode {
                Tx::MissingExpect | Tx::NoExpect | Tx::CopyAfterPhase => {}
                Tx::TooSmall => mbarrier_expect_tx(&bar, 32),
                Tx::Correct | Tx::OneWarpWaits => mbarrier_expect_tx(&bar, 64),
            }
            if mode == Tx::CopyAfterPhase {
                mbarrier_arrive(&bar);
            } else {
                cp_async_bulk(&src.ptr(), &data, 64, &bar);
                mbarrier_arrive(&bar);
            }
        });
        if_(t.eq(32), || {
            if matches!(mode, Tx::NoExpect | Tx::CopyAfterPhase) {
                mbarrier_arrive(&bar);
            } else {
                mbarrier_arrive_and_expect_tx(&bar, 64);
            }
            cp_async_bulk(&src.ptr().offset(16usize), &data.offset(16usize), 64, &bar);
        });
        if mode == Tx::CopyAfterPhase {
            if_(t.eq(1), || cp_async_bulk(&src.ptr(), &data, 64, &bar));
        }
        if mode == Tx::OneWarpWaits {
            if_(warp_idx().eq(0), || mbarrier_wait(&bar, 0));
        } else {
            mbarrier_wait(&bar, 0);
        }
        out.store(t, make_tensor(&data, Layout::col_major(32)).load(t % 32));
    })
}

/// The simulator holds bulk copies to their mbarrier's transaction count:
/// a phase must not complete while copies reporting to it are in flight.
#[test]
fn bulk_copies_need_their_transaction_count() {
    let src: Vec<u32> = (500..532).collect();
    let run = |mode: Tx| {
        let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 64])];
        bulk_copies(mode).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).map(|_| args)
    };
    let args = run(Tx::Correct).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<u32>()[33], 501);
    for (mode, want) in [
        (
            Tx::MissingExpect,
            "2 bulk copies of 128 bytes in all report to an mbarrier phase whose transaction count is 64: \
             the phase completes once 64 bytes have landed",
        ),
        (Tx::TooSmall, "2 bulk copies of 128 bytes in all report to an mbarrier phase whose transaction count is 96"),
        (
            Tx::NoExpect,
            "an mbarrier phase completed with 1 bulk copy reporting to it still in flight: its transaction count \
             did not include their bytes (missing mbarrier_expect_tx?)",
        ),
        (Tx::CopyAfterPhase, "reads smem1[0] while a bulk copy is still writing it"),
        (Tx::OneWarpWaits, "data race: warp 1 reads smem1[0], which a bulk copy wrote, without having waited"),
    ] {
        let e = run(mode).err().unwrap_or_else(|| panic!("{mode:?} should fail"));
        assert!(e.message.contains(want), "{mode:?}: {e}");
    }
}

#[test]
fn mbarrier_misuse() {
    let e = simulate_err(
        &Kernel::builder("too_many_arrivals").block_dim([32, 1, 1]).build(|| {
            let bar = alloc_smem::<u64>(1, 8);
            if_(thread_idx().x.eq(0), || mbarrier_init(&bar, 3));
            sync_threads();
            if_(thread_idx().x.eq(0), || {
                mbarrier_arrive_with(&bar, 2, MbarrierOptions::default());
                mbarrier_arrive_with(&bar, 2, MbarrierOptions::default());
            });
        }),
        32,
        &mut [],
    );
    assert!(e.message.contains("2 arrival(s) at an mbarrier whose phase is waiting for 1"), "{e}");
    let kernel = Kernel::builder("bad_peer").block_dim([32, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || {
            mbarrier_init(&bar, 32);
            mbarrier_init_fence();
        });
        cluster_sync();
        let peer =
            MbarrierOptions { peer_cta_rank_in_cluster: Some(block_in_cluster_idx().x + 1), ..Default::default() };
        mbarrier_arrive_with(&bar, 1, peer);
    });
    let cfg = LaunchConfig::new([2, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    let e = kernel.simulate(cfg, &mut []).unwrap_err();
    assert!(e.message.contains("mbarrier.arrive on CTA 2 of a cluster of 2"), "{e}");
    let msg = build_panic(|| {
        Kernel::build("zero_arrivals", || {
            let bar = alloc_smem::<u64>(1, 8);
            mbarrier_arrive_with(&bar, 0, MbarrierOptions::default());
        });
    });
    assert!(msg.contains("arrival count of 0"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("zero_count", || mbarrier_init(&alloc_smem::<u64>(1, 8), 0));
    });
    assert!(msg.contains("mbarrier count of 0"), "{msg}");
}

/// What each target supports is checked before compiling, with the
/// instruction's source location.
#[test]
fn mbarrier_target_requirements() {
    use cute::validate::{check_target, target_requirement};
    let kernel = Kernel::build("needs_sm90", || {
        let bar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&bar, 1);
        let _ = mbarrier_test_wait(&bar, 0);
        mbarrier_arrive_with(&bar, 1, MbarrierOptions { relaxed: true, ..Default::default() });
    });
    let e = check_target(kernel.ir(), Arch::Sm86).unwrap_err();
    assert!(e.message.contains("a relaxed mbarrier.arrive needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/barriers.rs"), "{e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 86, "relaxed needs PTX 8.6");
    let sm80 = Kernel::build("sm80_ok", || {
        let bar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&bar, 1);
        mbarrier_arrive(&bar);
        let _ = mbarrier_test_wait(&bar, 0);
    });
    assert_eq!(check_target(sm80.ir(), Arch::Sm80).unwrap(), 71, "test_wait.parity needs PTX 7.1");
    assert_eq!(check_target(sm80.ir(), Arch::Sm89).unwrap(), 78);
    let fence = cute::ir::Op::FenceMbarrierInit;
    assert_eq!(target_requirement(&fence).map(|r| (r.sm, r.ptx)), Some((90, 80)));
}

#[test]
fn make_warp_uniform_checks_its_promise() {
    let kernel = Kernel::builder("not_uniform").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        out.store(t, make_warp_uniform(t / 4));
    });
    let e = simulate_err(&kernel, 64, &mut [KernelArg::i32(vec![0; 64])]);
    assert!(
        e.message.contains("lane 4 of warp 0 holds 1, but lane 0 holds 0") && e.message.contains("not warp-uniform"),
        "{e}"
    );
}

/// `sync_threads` is for every thread of the block: one that some warps
/// skip (they exit instead), and two that different warps reach, are
/// reported where the waiting warps wait.
#[test]
fn sync_threads_that_not_every_warp_reaches() {
    let run = |f: fn()| simulate_err(&Kernel::builder("partial_sync").block_dim([128, 1, 1]).build(f), 128, &mut []);
    let e = run(|| if_(warp_idx().lt(2), || sync_threads()));
    assert!(
        e.message.contains("sync_threads() reached by 64 of 128 threads: warps 2..=3 exited without reaching it"),
        "{e}"
    );
    assert!(
        e.message.contains("divergent control flow") && e.to_string().contains("tests/diagnostics/barriers.rs"),
        "{e}"
    );
    let e = run(|| if_else(warp_idx().lt(1), || sync_threads(), || sync_threads()));
    assert!(e.message.contains("sync_threads() reached by") && e.message.contains("wait at another one"), "{e}");
}

/// Inline PTX holding the cluster barrier synchronises the cluster's warps
/// in the simulator: CTA 1 arrives on CTA 0's mbarrier only after CTA 0's
/// late init, a split arrive and wait orders shared memory within a block
/// (a relaxed arrival does not), and a warp that waits at it while the
/// rest of the cluster waits at `sync_threads` is a deadlock.
#[test]
fn cluster_barriers_order_the_cluster() {
    let cluster = LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    let spin = |n: Int32| {
        let _ = while_carry(0i32.into_value(), |i| i.lt(n), |i| i + 1);
    };
    let late_init = Kernel::builder("late_init").block_dim([64, 1, 1]).build(|| {
        let (t, rank) = (thread_idx().x, block_in_cluster_idx().x);
        let bar = alloc_smem::<u64>(1, 8);
        if_(t.eq(0) & rank.eq(0), || {
            spin(20.into_value());
            mbarrier_init(&bar, 64);
            mbarrier_init_fence();
        });
        inline_ptx_effect_with_sim("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;", &[], |_| vec![]);
        if_(rank.eq(1), || {
            let peer = MbarrierOptions { peer_cta_rank_in_cluster: Some(0.into_value()), ..Default::default() };
            mbarrier_arrive_with(&bar, 1, peer);
        });
        if_(rank.eq(0), || mbarrier_wait(&bar, 0));
    });
    late_init.simulate(cluster, &mut []).unwrap_or_else(|e| panic!("{e}"));
    let split = |arrive: &'static str| {
        Kernel::builder("split").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(128));
            let smem = make_smem_tensor::<u32>(Layout::col_major(32));
            let t = thread_idx().x;
            let lane = t % 32;
            if_(warp_idx().eq(0), || {
                spin(9.into_value());
                smem.store(lane, lane.to::<u32>() + 5u32);
            });
            inline_ptx_effect_with_sim(arrive, &[], |_| vec![]);
            spin(t % 3);
            inline_ptx_effect_with_sim("barrier.cluster.wait.aligned;", &[], |_| vec![]);
            out.store(block_idx().x * 64 + t, smem.load(lane));
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 128])];
    split("barrier.cluster.arrive.aligned;").simulate(cluster, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[125..], [34, 35, 36]);
    let e = split("barrier.cluster.arrive.relaxed.aligned;")
        .simulate(cluster, &mut [KernelArg::buffer(&[0u32; 128])])
        .unwrap_err();
    assert!(e.message.contains("data race: warp 1 reads"), "{e}");
    let stuck = Kernel::builder("stuck").block_dim([64, 1, 1]).build(|| {
        if_else(
            warp_idx().eq(0),
            || {
                inline_ptx_effect_with_sim(
                    "barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;",
                    &[],
                    |_| vec![],
                )
            },
            sync_threads,
        );
    });
    let e = stuck.simulate(cluster, &mut []).unwrap_err();
    assert!(e.message.contains("deadlock") && e.message.contains("wait at the cluster barrier"), "{e}");
    assert!(
        e.message.contains("warp 1 of block (0,0,0), warp 1 of block (1,0,0) of the cluster have not arrived"),
        "{e}"
    );
}

/// The errors of `kernel` under each built-in schedule, which must all fail.
fn errors_under_every_schedule(
    kernel: &Kernel,
    cfg: LaunchConfig,
    args: &[KernelArg],
    opts: SimOptions,
) -> Vec<SimError> {
    Schedule::ALL
        .into_iter()
        .map(|schedule| {
            let opts = SimOptions { schedule, ..opts.clone() };
            match kernel.simulate_with(cfg, &mut args.to_vec(), &opts) {
                Ok(_) => panic!("expected `{}` to fail under {schedule:?}", kernel.name()),
                Err(e) => e,
            }
        })
        .collect()
}

/// The line of the caller, for messages that name source lines.
#[track_caller]
fn here() -> u32 {
    std::panic::Location::caller().line()
}

/// A named barrier that the other warps never reach: the report names the
/// waiting warps, the loop iteration and branch they wait in, and where the
/// missing warps are.
#[test]
fn deadlock_report_names_warps_lines_and_constructs() {
    let lines = std::cell::Cell::new((0, 0, 0, 0));
    let kernel = Kernel::builder("never_reached").block_dim([128, 1, 1]).build(|| {
        let loop_line = here() + 1;
        for_range(0, 4, 1, |i| {
            let (if_line, bar_line) = (here() + 1, here() + 2);
            if_(warp_idx().lt(2) & i.eq(2), || {
                barrier(1, 128);
            });
            let other = here() + 1;
            if_(warp_idx().ge(2) & i.eq(0), || barrier(2, 128));
            lines.set((loop_line, if_line, bar_line, other));
        });
    });
    let (loop_line, if_line, bar_line, other) = lines.get();
    for e in errors_under_every_schedule(&kernel, LaunchConfig::new([1, 1, 1], [128, 1, 1]), &[], SimOptions::default())
    {
        let m = &e.message;
        assert!(m.contains("did not terminate") && m.contains("(a deadlock)"), "{e}");
        let file = "tests/diagnostics/barriers.rs";
        assert!(
            m.contains(&format!(
                "warps 0..=1 of block (0,0,0) wait at barrier 1 ({file}:{bar_line}), in the then-branch of if_ at \
                 {file}:{if_line}; in for_range at {file}:{loop_line}, iteration 2, which needs 4 warps"
            )),
            "{e}"
        );
        assert!(
            m.contains(&format!("warps 2..=3 wait at barrier 2 ({file}:{other})")) && m.contains("iteration 0"),
            "{e}"
        );
        assert_eq!(e.loc.map(|l| l.line()), Some(bar_line), "{e}");
    }
}

/// An mbarrier that nobody arrives on, and a consumer that waits for a
/// phase the producer never completes: once the waits have given up with
/// nothing else happening, the report names the mbarrier, the phase and
/// its pending arrivals.
#[test]
fn mbarrier_deadlocks_name_the_phase() {
    let never = Kernel::builder("never_arrived").block_dim([32, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&bar, 1));
        sync_threads();
        mbarrier_wait(&bar, 0);
    });
    for e in errors_under_every_schedule(&never, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &[], SimOptions::default()) {
        assert!(
            e.message.contains("(a deadlock)")
                && e.message.contains("warp 0 of block (0,0,0) waits at mbarrier.try_wait")
                && e.message.contains(
                    "on the mbarrier smem0[0] for phase 0 (parity 0), which still waits for 1 of its 1 arrival(s), \
                     with a tx-count of 0 byte(s) and 0 bulk copies in flight"
                ),
            "{e}"
        );
    }
    let wrong_parity = Kernel::builder("wrong_parity").block_dim([64, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&bar, 1));
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || if_(t.eq(0), || mbarrier_arrive(&bar)),
            || {
                mbarrier_wait(&bar, 0);
                // The producer arrives once: phase 1 never completes.
                mbarrier_wait(&bar, 1);
            },
        );
    });
    for e in
        errors_under_every_schedule(&wrong_parity, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &[], SimOptions::default())
    {
        assert!(
            e.message.contains("warp 1 of block (0,0,0) waits at mbarrier.try_wait")
                && e.message.contains("in the else-branch of if_")
                && e.message.contains("for phase 1 (parity 1), which still waits for 1 of its 1 arrival(s)"),
            "{e}"
        );
    }
}

/// A counted named barrier whose missing warps have all exited can only
/// complete through PTX's exit rule: the report says so and names them.
#[test]
fn named_barrier_left_to_exited_warps() {
    let kernel = Kernel::builder("left_to_exited").block_dim([128, 1, 1]).build(|| {
        if_(warp_idx().ge(2), || barrier(1, 128));
    });
    for e in errors_under_every_schedule(&kernel, LaunchConfig::new([1, 1, 1], [128, 1, 1]), &[], SimOptions::default())
    {
        assert!(
            e.message.contains("warps 2..=3 of block (0,0,0) wait at barrier 1")
                && e.message.contains("which needs 4 warps; warps 0..=1 have exited")
                && e.message.contains("It can only complete through warps that have exited (0..=1)"),
            "{e}"
        );
    }
}

/// Spinning on a flag nobody writes, and on an mbarrier whose producer is
/// blocked, is reported as a hang (not as a loop that did not terminate),
/// with what each warp polls and where the others wait.
#[test]
fn spins_that_cannot_succeed_are_hangs() {
    let poll_line = std::cell::Cell::new(0);
    let flag = Kernel::builder("unwritten_flag").block_dim([32, 1, 1]).build(|| {
        let flag = param_tensor::<i32>("flag", Layout::col_major(1));
        let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Gpu), ..Default::default() };
        poll_line.set(here() + 1);
        let _ = while_carry(0i32.into_value(), |v| v.eq(0), |_| load(&flag.ptr(), acquire));
    });
    let opts = SimOptions { max_idle_steps: 5000, ..SimOptions::default() };
    let args = [KernelArg::i32(vec![0])];
    for e in errors_under_every_schedule(&flag, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &args, opts.clone()) {
        assert!(e.message.contains("did not terminate") && e.message.contains("(a hang)"), "{e}");
        assert!(
            e.message.contains(&format!(
                "warp 0 of block (0,0,0) spins, in while_carry at tests/diagnostics/barriers.rs:{}",
                poll_line.get()
            )) && e.message.contains("polling flag[0] with an ordered load"),
            "{e}"
        );
    }
    let blocked_producer = Kernel::builder("blocked_producer").block_dim([64, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&bar, 1));
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                // Waits for warp 1, which waits for this warp's arrival.
                barrier(1, 64);
                if_(t.eq(0), || mbarrier_arrive(&bar));
            },
            || {
                let _ = while_carry(false.into_value(), |done| !done, |_| mbarrier_test_wait(&bar, 0));
                barrier(1, 64);
            },
        );
    });
    for e in errors_under_every_schedule(&blocked_producer, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &[], opts) {
        assert!(e.message.contains("(a hang)"), "{e}");
        assert!(
            e.message.contains("warp 1 of block (0,0,0) spins")
                && e.message.contains("polling with mbarrier.test_wait")
                && e.message.contains("the mbarrier smem0[0] for phase 0 (parity 0)")
                && e.message.contains("warp 0 of block (0,0,0) waits at barrier 1")
                && e.message.contains("needs 2 warps; warp 1 spins"),
            "{e}"
        );
    }
}

fn cluster_of_two() -> LaunchConfig {
    LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1])
}

/// Misused cluster barriers: a thread arriving twice in a phase, a wait
/// without an arrival, an aligned arrival by part of a warp, and a deadlock
/// naming the CTAs, warps and lanes that have not arrived.
#[test]
fn cluster_barrier_misuse() {
    let build = |f: fn()| Kernel::builder("cluster_misuse").block_dim([64, 1, 1]).build(f);
    let run = |f: fn()| errors_under_every_schedule(&build(f), cluster_of_two(), &[], SimOptions::default());
    for e in run(|| {
        cluster_arrive();
        cluster_arrive();
        cluster_wait();
    }) {
        assert!(
            e.message.contains("cluster_arrive: the thread arrives at the cluster barrier again without having waited"),
            "{e}"
        );
    }
    for e in run(cluster_wait) {
        assert!(e.message.contains("cluster_wait by a thread that has not arrived at the cluster barrier"), "{e}");
    }
    for e in run(|| {
        if_(thread_idx().x.lt(16), cluster_arrive_aligned);
        cluster_wait();
    }) {
        assert!(
            e.message.contains("cluster_arrive executed by 16 of the 32 lanes of warp 0: its .aligned form"),
            "{e}"
        );
    }
    for e in run(|| {
        // Lanes 16..=31 of CTA 1's warp 0 never arrive.
        let t = thread_idx().x;
        if_(block_idx_in_cluster().eq(0) | t.lt(16) | t.ge(32), || {
            cluster_arrive();
            cluster_wait();
        });
    }) {
        assert!(
            e.message.contains("(a deadlock)")
                && e.message.contains("wait at the cluster barrier")
                && e.message.contains("which warp 0 (lanes 16..=31) of block (1,0,0) of the cluster have not arrived"),
            "{e}"
        );
    }
    // The threads of a warp may arrive one by one with the non-aligned form.
    build(|| {
        if_else(thread_idx().x.lt(7), cluster_arrive, cluster_arrive);
        cluster_wait();
        cluster_sync();
    })
    .simulate_all_orders(cluster_of_two(), &mut [], &SimOptions::default())
    .unwrap_or_else(|e| panic!("{e}"));
}

/// An mbarrier a peer CTA arrives on must have had its initialisation
/// published to it: `mbarrier_init_fence` and then a cluster barrier (a
/// relaxed arrival will do, as in the PTX ISA's pattern) or a
/// release/acquire chain. A peer that has exited has no shared memory to
/// arrive on.
#[test]
fn remote_mbarriers_need_published_inits_and_live_peers() {
    let kernel = |fence: bool, relaxed: bool| {
        Kernel::builder("remote_init").block_dim([64, 1, 1]).build(move || {
            let bar = alloc_smem::<u64>(1, 8);
            let (t, rank) = (thread_idx().x, block_in_cluster_idx().x);
            if_(t.eq(0), || {
                mbarrier_init(&bar, 1);
                if fence {
                    mbarrier_init_fence();
                }
            });
            if relaxed {
                cluster_arrive_relaxed();
            } else {
                cluster_arrive();
            }
            cluster_wait();
            let peer = MbarrierOptions {
                peer_cta_rank_in_cluster: Some(1.into_value()),
                scope: Some(MemScopeKind::Cluster),
                ..Default::default()
            };
            if_(rank.eq(0) & t.eq(0), || mbarrier_arrive_with(&bar, 1, peer));
            if_(rank.eq(1), || mbarrier_wait(&bar, 0));
        })
    };
    for relaxed in [false, true] {
        kernel(true, relaxed)
            .simulate_all_orders(cluster_of_two(), &mut [], &SimOptions::default())
            .unwrap_or_else(|e| panic!("relaxed {relaxed}: {e}"));
    }
    for e in errors_under_every_schedule(&kernel(false, false), cluster_of_two(), &[], SimOptions::default()) {
        assert!(
            e.message.contains(
                "mbarrier.arrive by CTA 0 on the mbarrier smem0[0] of CTA 1 before its initialisation was published \
                 to CTA 0: warp 0 of CTA 1 initialised it"
            ) && e.message.contains("has not executed mbarrier_init_fence()"),
            "{e}"
        );
    }
    let exited = Kernel::builder("exited_peer").block_dim([32, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        let (t, rank) = (thread_idx().x, block_in_cluster_idx().x);
        if_(t.eq(0), || {
            mbarrier_init(&bar, 1);
            mbarrier_init_fence();
        });
        cluster_sync();
        if_(rank.eq(0), || {
            let _ = while_carry(0i32.into_value(), |i| i.lt(50), |i| i + 1);
            let peer = MbarrierOptions { peer_cta_rank_in_cluster: Some(1.into_value()), ..Default::default() };
            if_(t.eq(0), || mbarrier_arrive_with(&bar, 1, peer));
        });
    });
    let cfg = LaunchConfig::new([2, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    for e in errors_under_every_schedule(&exited, cfg, &[], SimOptions::default()) {
        assert!(
            e.message.contains("mbarrier.arrive by CTA 0 on the mbarrier smem0[0] of CTA 1, which has exited"),
            "{e}"
        );
    }
}

/// Vector clocks span the cluster: a release/acquire chain through CTA 1
/// tells a warp of CTA 0 which mbarrier phases another warp of CTA 0 saw,
/// so its parity wait for the next phase is meaningful; with a relaxed link
/// in the chain it is not.
#[test]
fn happens_before_spans_the_cluster() {
    let kernel = |release: bool| {
        Kernel::builder("cluster_chain").block_dim([64, 1, 1]).build(move || {
            let flags = param_tensor::<i32>("flags", Layout::col_major(2)).ptr();
            let out = param_tensor::<u32>("out", Layout::col_major(1));
            let bar = alloc_smem::<u64>(1, 8);
            let (t, rank, warp) = (thread_idx().x, block_in_cluster_idx().x, warp_idx());
            let gpu = Some(MemoryScope::Gpu);
            let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..Default::default() };
            let spin = |i: usize| {
                let _ = while_carry(0i32.into_value(), |v| v.eq(0), |_| load(&flags.offset(i), acquire));
            };
            let set = |i: usize, sem: MemoryOrder| {
                store(&flags.offset(i), 1, StoreOptions { sem, scope: gpu, ..Default::default() })
            };
            if_(t.eq(0), || mbarrier_init(&bar, 1));
            sync_threads();
            if_(rank.eq(0) & warp.eq(0), || {
                if_(t.eq(0), || mbarrier_arrive(&bar));
                mbarrier_wait(&bar, 0);
                set(0, MemoryOrder::Release);
            });
            if_(rank.eq(1) & warp.eq(0), || {
                spin(0);
                set(1, if release { MemoryOrder::Release } else { MemoryOrder::Relaxed });
            });
            if_(rank.eq(0) & warp.eq(1), || {
                spin(1);
                // Asks about phase 1, which only means something to a warp
                // that knows phase 0 is over.
                let done = mbarrier_test_wait(&bar, 1);
                if_(t.eq(32), || out.store(0, done.to::<u32>() + 7u32));
            });
        })
    };
    let args = || vec![KernelArg::i32(vec![0, 0]), KernelArg::buffer(&[0u32])];
    let mut ok = args();
    kernel(true)
        .simulate_all_orders(cluster_of_two(), &mut ok, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(ok[1].to_vec::<u32>(), [7]);
    for e in errors_under_every_schedule(&kernel(false), cluster_of_two(), &args(), SimOptions::default()) {
        assert!(e.message.contains("but warp 1 has not seen phase 0 complete"), "{e}");
    }
}

/// A grid-wide barrier built from atomics needs every block of the grid to
/// run together: with one cluster at a time it hangs (and the report
/// suggests resident_clusters), with the whole grid resident it completes.
#[test]
fn grid_barrier_needs_resident_clusters() {
    let kernel = Kernel::builder("grid_barrier").block_dim([32, 1, 1]).build(|| {
        let counter = param_tensor::<i32>("counter", Layout::col_major(1));
        let out = param_tensor::<i32>("out", Layout::col_major(4 * 32));
        let t = thread_idx().x;
        let gpu = AtomicOptions::new(MemoryOrder::Release, MemoryScope::Gpu);
        if_(t.eq(0), || {
            let _ = atomic_add_with(&counter.ptr(), 1, gpu);
        });
        let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Gpu), ..Default::default() };
        let arrived = while_carry(0i32.into_value(), |n| n.lt(4), |_| load(&counter.ptr(), acquire));
        out.store(block_idx().x * 32 + t, arrived);
    });
    let cfg = LaunchConfig::new([4, 1, 1], [32, 1, 1]);
    let args = || vec![KernelArg::i32(vec![0]), KernelArg::i32(vec![0; 128])];
    let opts = SimOptions { max_idle_steps: 5000, ..SimOptions::default() };
    for e in errors_under_every_schedule(&kernel, cfg, &args(), opts.clone()) {
        assert!(
            e.message.contains("(a hang)")
                && e.message.contains("polling counter[0] with an ordered load")
                && e.message.contains("3 cluster(s) of the grid have not started")
                && e.message.contains("ResidentClusters::Grid"),
            "{e}"
        );
    }
    for resident in [ResidentClusters::Grid, ResidentClusters::Count(4)] {
        let mut a = args();
        let opts = SimOptions { resident_clusters: resident, ..opts.clone() };
        let report = kernel.simulate_all_orders(cfg, &mut a, &opts).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(report.blocks, 4);
        assert_eq!(a[1].to_vec::<i32>(), vec![4; 128]);
    }
    // Two at a time: the first two blocks wait for the others forever.
    let opts = SimOptions { resident_clusters: ResidentClusters::Count(2), ..opts };
    let e = kernel.simulate_with(cfg, &mut args(), &opts).unwrap_err();
    assert!(e.message.contains("2 cluster(s) of the grid have not started"), "{e}");
}

/// A read of `cp.async` data without waiting for it: caught when copies
/// complete as late as possible (the default), harmless when they complete
/// at issue, where the data is already there.
#[test]
fn premature_reads_depend_on_async_completion() {
    let src: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let mut args = vec![KernelArg::f32(src.clone()), KernelArg::f32(vec![0.0; 128])];
    let late = staged_copy(false).simulate(cfg, &mut args).unwrap_err();
    assert!(late.message.contains("while a cp.async is still writing it (missing cp_async_wait_group?)"), "{late}");
    let early = SimOptions { async_completion: AsyncCompletion::Early, ..SimOptions::default() };
    let mut args = vec![KernelArg::f32(src.clone()), KernelArg::f32(vec![0.0; 128])];
    staged_copy(false).simulate_with(cfg, &mut args, &early).unwrap_or_else(|e| panic!("{e}"));
    let out = args[1].as_f32();
    assert_eq!([out[4], out[8], out[124]], [src[4], src[8], src[124]]);
    // Random completion: the correct kernel works under every seed, the
    // premature read is caught under some.
    let mut caught = 0;
    for seed in 0..8 {
        let random = SimOptions { async_completion: AsyncCompletion::Random(seed), ..SimOptions::default() };
        let mut args = vec![KernelArg::f32(src.clone()), KernelArg::f32(vec![0.0; 128])];
        staged_copy(true).simulate_with(cfg, &mut args, &random).unwrap_or_else(|e| panic!("{e}"));
        assert!((0..32).all(|i| args[1].as_f32()[4 * i] == src[4 * i]));
        caught += staged_copy(false).simulate_with(cfg, &mut args, &random).is_err() as u32;
    }
    assert!(caught > 0);
}

/// A block that exits with a copy into its shared memory in flight, and a
/// bulk copy whose bytes arrive before (or without) the `expect_tx` that
/// covers them when copies complete at issue.
#[test]
fn async_copies_outliving_their_block_or_their_phase() {
    let kernel = Kernel::builder("exit_in_flight").block_dim([32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(128));
        let smem = make_smem_tensor::<u32>(Layout::col_major(128));
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let thr = tiled.get_slice(thread_idx().x);
        copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&smem));
        cp_async_commit_group();
    });
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let e = kernel.simulate(cfg, &mut [KernelArg::buffer(&[0u32; 128])]).unwrap_err();
    assert!(
        e.message.contains("CTA 0 (block (0,0,0)) exits while an asynchronous copy is still writing its shared memory"),
        "{e}"
    );
    let short_tx = Kernel::builder("short_tx").block_dim([32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(16));
        let bar = alloc_smem::<u64>(1, 8);
        let data = alloc_smem::<u32>(16, 128);
        if_(thread_idx().x.eq(0), || {
            mbarrier_init(&bar, 1);
            cp_async_bulk(&src.ptr(), &data, 64, &bar);
            mbarrier_arrive_and_expect_tx(&bar, 32);
        });
        mbarrier_wait(&bar, 0);
    });
    let early = SimOptions { async_completion: AsyncCompletion::Early, ..SimOptions::default() };
    let e = short_tx.simulate_with(cfg, &mut [KernelArg::buffer(&[0u32; 16])], &early).unwrap_err();
    assert!(e.message.contains("landed 32 byte(s) more than its transaction count"), "{e}");
    let e = short_tx.simulate(cfg, &mut [KernelArg::buffer(&[0u32; 16])]).unwrap_err();
    assert!(e.message.contains("mbarrier_expect_tx too small?"), "{e}");
}

// Review regressions: read epochs, cluster barrier patterns, try_wait give-ups and hangs.

fn run_all_orders(k: &Kernel, threads: u32, args: &mut [KernelArg]) -> Result<SimReport, SimError> {
    k.simulate_all_orders(LaunchConfig::new([1, 1, 1], [threads, 1, 1]), args, &SimOptions::default())
}

/// Warps 1 and 2 read concurrently; warp 3 is ordered after warp 1 only, then writes.
#[test]
fn read_epochs_keep_concurrent_readers() {
    let k = Kernel::builder("p1").block_dim([128, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        let w = warp_idx();
        if_(w.eq(0), || smem.store(t % 32, 7u32));
        sync_threads();
        if_(w.eq(1) | w.eq(2), || out.store(t, smem.load(t % 32)));
        if_(w.eq(1), || barrier_arrive(1, 64));
        if_(w.eq(3), || {
            barrier(1, 64);
            smem.store(t % 32, 9u32);
        });
    });
    let e = run_all_orders(&k, 128, &mut [KernelArg::buffer(&[0u32; 128])]).unwrap_err();
    assert!(e.message.contains("overwrites") && e.message.contains("warp 2 read"), "{e}");
}

/// Same, but warp 3 is ordered after both readers: no race.
#[test]
fn read_epochs_forget_ordered_readers() {
    let k = Kernel::builder("p1b").block_dim([128, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        let w = warp_idx();
        if_(w.eq(0), || smem.store(t % 32, 7u32));
        sync_threads();
        if_(w.eq(1) | w.eq(2), || {
            out.store(t, smem.load(t % 32));
            barrier_arrive(1, 96);
        });
        if_(w.eq(3), || {
            barrier(1, 96);
            smem.store(t % 32, 9u32);
        });
    });
    run_all_orders(&k, 128, &mut [KernelArg::buffer(&[0u32; 128])]).unwrap_or_else(|e| panic!("{e}"));
}

/// Warp 1 reads; warp 2, ordered after warp 1, reads (forgets warp 1's epoch);
/// warp 1 then writes without being ordered after warp 2's read: a race.
#[test]
fn forgotten_reader_still_races_with_later_readers() {
    let k = Kernel::builder("p2").block_dim([96, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(96));
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        let w = warp_idx();
        if_(w.eq(0), || smem.store(t % 32, 7u32));
        sync_threads();
        if_(w.eq(1), || {
            out.store(t, smem.load(t % 32));
            barrier_arrive(1, 64);
            smem.store(t % 32, 3u32);
        });
        if_(w.eq(2), || {
            barrier(1, 64);
            out.store(t, smem.load(t % 32));
        });
    });
    let e = run_all_orders(&k, 96, &mut [KernelArg::buffer(&[0u32; 96])]).unwrap_err();
    assert!(e.message.contains("data race"), "{e}");
}

/// The PTX ISA's pattern: fence.acq_rel then a relaxed cluster arrive orders
/// the warp's earlier writes for the waiters.
#[test]
fn fence_then_relaxed_cluster_arrive_releases() {
    let k = Kernel::builder("p3").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || smem.store(t, t.to::<u32>()));
        fence_acq_rel_cluster();
        cluster_arrive_relaxed_aligned();
        cluster_wait_aligned();
        out.store(t, smem.load(t % 32));
    });
    let cfg = LaunchConfig::new([2, 1, 1], [64, 1, 1]).with_cluster([2, 1, 1]);
    k.simulate_all_orders(cfg, &mut [KernelArg::buffer(&[0u32; 64])], &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// A consumer that gives up on try_wait a few times (counting, without
/// anything visible) before it does the arrival the producer needs.
#[test]
fn try_wait_may_give_up_several_times() {
    let k = Kernel::builder("p4").block_dim([64, 1, 1]).build(|| {
        let bars = alloc_smem::<u64>(2, 8);
        let a = bars.clone();
        let b = bars.offset(1);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            mbarrier_init(&a, 1);
            mbarrier_init(&b, 1);
        });
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                // Producer: waits on b, then arrives on a.
                mbarrier_wait(&b, 0);
                if_(t.eq(0), || mbarrier_arrive(&a));
            },
            || {
                // Consumer: try a three times, then arrive on b and wait for a.
                let tries = while_carry(0i32.into_value(), |n| n.lt(3) & !mbarrier_try_wait(&a, 0), |n| n + 1);
                let _ = tries;
                if_(t.eq(32), || mbarrier_arrive(&b));
                mbarrier_wait(&a, 0);
            },
        );
    });
    run_all_orders(&k, 64, &mut []).unwrap_or_else(|e| panic!("{e}"));
}

/// Lanes of a warp arrive in two branches (non-aligned), then all wait.
#[test]
fn non_aligned_cluster_arrivals_in_branches() {
    let k = Kernel::builder("p5").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        let smem = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || smem.store(t, t.to::<u32>()));
        if_else((t % 32).lt(16), cluster_arrive, cluster_arrive);
        cluster_wait();
        out.store(block_idx().x * 64 + t, smem.load(t % 32));
    });
    k.simulate_all_orders(cluster_of_two(), &mut [KernelArg::buffer(&[0u32; 128])], &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// CTA 1 exits at once; CTA 0 synchronises the cluster alone.
#[test]
fn cluster_barrier_ignores_exited_ctas() {
    let k = Kernel::builder("p6").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(128));
        if_(block_in_cluster_idx().x.eq(0), || {
            cluster_sync();
            out.store(thread_idx().x, 1u32);
        });
    });
    k.simulate_all_orders(cluster_of_two(), &mut [KernelArg::buffer(&[0u32; 128])], &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// Remote relaxed arrival hand-off: CTA 1 writes global data, arrives
/// (release, cluster scope) on CTA 0's mbarrier; CTA 0 waits.
#[test]
fn remote_arrival_after_relaxed_init_publication() {
    let k = Kernel::builder("p7").block_dim([64, 1, 1]).build(|| {
        let bar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_in_cluster_idx().x;
        if_(t.eq(0), || {
            mbarrier_init(&bar, 1);
            mbarrier_init_fence();
        });
        cluster_arrive_relaxed_aligned();
        cluster_wait_aligned();
        if_(rank.eq(1) & t.eq(0), || {
            let o = MbarrierOptions {
                peer_cta_rank_in_cluster: Some(0.into_value()),
                scope: Some(MemScopeKind::Cluster),
                ..Default::default()
            };
            mbarrier_arrive_with(&bar, 1, o);
        });
        if_(rank.eq(0), || mbarrier_wait(&bar, 0));
        cluster_sync();
    });
    k.simulate_all_orders(cluster_of_two(), &mut [], &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
}

/// A long loop that only computes in registers while the other warp waits
/// at a barrier is not a hang.
#[test]
fn register_only_loop_is_not_a_hang() {
    let k = Kernel::builder("p8").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let n = while_carry(0i32.into_value(), |i| i.lt(2000), |i| i + 1);
        sync_threads();
        out.store(t, n);
    });
    let opts = SimOptions { max_idle_steps: 500, ..SimOptions::default() };
    k.simulate_all_orders(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut [KernelArg::i32(vec![0; 64])], &opts)
        .unwrap_or_else(|e| panic!("{e}"));
}

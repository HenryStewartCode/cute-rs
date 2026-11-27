//! Dynamic shared memory, tracked hand-offs, memory orders and attributes, spin locks and atomics, and the races they allow or forbid.

use super::*;
/// The dynamic shared-memory region holds the bytes the launch gives it: an access past them is
/// out of bounds, naming the region; a launch whose static and dynamic shared memory exceed what
/// any GPU gives a block is refused; `get_dyn_smem`'s alignment must be a power of two and at
/// least the element's size; and `aggr_smem_size` needs sm_90.
#[test]
fn dynamic_shared_memory_misuse() {
    let kernel = feature_kernels::dynamic_region_kernel();
    let launch = |bytes: u32| LaunchConfig::new([1, 1, 1], [64, 1, 1]).with_dynamic_smem(bytes);
    let mut args = feature_kernels::inputs(&kernel, true);
    // 50 words: thread 50's store at the front is past the end.
    let e = kernel.simulate(launch(200), &mut args).unwrap_err();
    assert!(
        e.message.contains(
            "store of 1 element(s) at index 50 is out of bounds for `dynamic_smem` (50 elements): the dynamic \
             shared memory region (get_dyn_smem) has the 200 bytes the launch gives it"
        ),
        "{e}"
    );
    assert_eq!(e.thread, Some([50, 0, 0]));
    assert!(e.to_string().contains("tests/diagnostics/../feature_kernels/mod.rs"), "{e}");
    let e = kernel.simulate(launch(327 * 1024), &mut args).unwrap_err();
    assert!(
        e.message.contains(
            "kernel `dynamic_region` needs 335088 bytes of shared memory per block (240 sized when it was built and \
             334848 of dynamic shared memory at launch); no GPU gives a block more than 334848"
        ),
        "{e}"
    );
    kernel.simulate(launch(327 * 1024 - 240), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let msg = build_panic(|| {
        Kernel::build("odd_alignment", || {
            get_dyn_smem::<u32>(Some(12)).store(1u32);
        });
    });
    assert!(msg.contains("get_dyn_smem: an alignment of 12 bytes is not a power of two"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("small_alignment", || {
            get_dyn_smem::<u64>(Some(4)).store(1u64);
        });
    });
    assert!(msg.contains("get_dyn_smem: an alignment of 4 bytes is less than the size of u64 (8 bytes)"), "{msg}");
    // The dynamic shared memory's symbol is not a kernel name.
    let msg = build_panic(|| {
        Kernel::build("__cute_dynamic_smem", || {});
    });
    assert!(msg.contains("kernel name \"__cute_dynamic_smem\" is reserved"), "{msg}");
    let aggr = Kernel::build("aggr", || {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        out.store(thread_idx().x, aggr_smem_size());
    });
    let e = cute::validate::check_target(aggr.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("%aggr_smem_size (aggr_smem_size) needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
    assert_eq!(cute::validate::check_target(aggr.ir(), Arch::Sm90).unwrap(), 81, "%aggr_smem_size needs PTX 8.1");
    let sizes = Kernel::build("sizes", || {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        out.store(thread_idx().x, dynamic_smem_size() + get_dyn_smem_size() + total_smem_size());
    });
    assert_eq!(cute::validate::check_target(sizes.ir(), Arch::Sm80).unwrap(), 70);
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Track {
    Correct,
    /// The mbarrier counts 16 arrivals, but 32 lanes arrive with `.noinc`.
    CountTooSmall,
    /// Each producer lane also arrives with `mbarrier_arrive`, which the count leaves no room for.
    ExtraArrivals,
    /// The consumer reads before it waits.
    ReadBeforeWait,
    /// A third warp reads without waiting on the mbarrier.
    UnorderedReader,
    /// The producer arrives with `mbarrier_arrive` instead: nothing tracks the copies.
    Untracked,
    /// The producer initialises the mbarrier again while the copies' arrivals are owed.
    ReinitWhileOwed,
    /// The producer stores a word before it copies, and the consumer reads it after waiting: the
    /// asynchronous arrival orders only the thread's `cp.async` copies, not its other stores.
    OrdinaryStoreNotReleased,
    /// A third warp makes another mbarrier track copies of its own, waits for them with
    /// `cp_async_wait_group` and reads them, then reads the producer's copies, which nothing has
    /// ordered before it.
    WaitGroupReader,
    /// The mbarrier counts 1 arrival, which `mbarrier_arrive_and_expect_tx` makes before the
    /// producer's `.noinc` arrivals: the phase only waits for transaction bytes.
    OnlyTxLeft,
    /// Correct: the producer commits its copies as a `cp.async` group too; the consumer's wait
    /// lands them, and the producer then waits for the group and reads them back.
    WaitGroupAfterPhase,
    /// Correct: no warp waits on the mbarrier. The producer commits its copies as a group and
    /// waits for it, and after a barrier warp 2 reads them.
    WaitGroupThenBarrier,
}

/// Warp 0 copies 128 words into shared memory with `cp.async` and makes an mbarrier counting 32
/// arrivals track them (`cp_async_mbarrier_arrive_noinc`); warp 1 waits on it and reads them.
fn tracked_handoff(mode: Track) -> Kernel {
    Kernel::builder("tracked_handoff").block_dim([96, 1, 1]).build(move || {
        let src = param_tensor::<u32>("src", Layout::col_major(128));
        let out = param_tensor::<u32>("out", Layout::col_major(96 * 4));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(128));
        let count = match mode {
            Track::CountTooSmall => 16,
            Track::OnlyTxLeft => 1,
            _ => 32,
        };
        // Warp 2's own copies, and the mbarrier that tracks them.
        let own = (mode == Track::WaitGroupReader)
            .then(|| (make_smem_tensor::<u32>(Layout::col_major(128)), alloc_smem::<u64>(1, 8)));
        let flag = (mode == Track::OrdinaryStoreNotReleased).then(|| alloc_smem::<u32>(1, 4));
        if_(t.eq(0), || {
            mbarrier_init(&bar, count);
            if let Some((_, own_bar)) = &own {
                mbarrier_init(own_bar, 32);
            }
        });
        sync_threads();
        if mode == Track::OnlyTxLeft {
            if_(t.eq(0), || mbarrier_arrive_and_expect_tx(&bar, 512));
        }
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let read_from = |buffer: &Tensor<u32>| {
            for k in 0..4 {
                out.store(t * 4 + k, buffer.load(lane * 4 + k));
            }
        };
        let read = || read_from(&data);
        if_else(
            warp_idx().eq(0),
            || {
                if let Some(flag) = &flag {
                    flag.store(7u32);
                }
                let thr = tiled.get_slice(lane);
                copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&data));
                if matches!(mode, Track::WaitGroupAfterPhase | Track::WaitGroupThenBarrier) {
                    cp_async_commit_group();
                }
                if mode == Track::Untracked {
                    mbarrier_arrive(&bar);
                } else {
                    cp_async_mbarrier_arrive_noinc(&bar);
                }
                if mode == Track::ExtraArrivals {
                    mbarrier_arrive(&bar);
                }
                if mode == Track::ReinitWhileOwed {
                    if_(lane.eq(0), || mbarrier_init(&bar, 32));
                }
                if mode == Track::WaitGroupThenBarrier {
                    cp_async_wait_group(0);
                }
            },
            || {
                // With ReinitWhileOwed nothing waits, so the owed arrivals stay owed.
                if matches!(mode, Track::WaitGroupThenBarrier | Track::ReinitWhileOwed) {
                    return;
                }
                if_(warp_idx().eq(1), || {
                    if mode == Track::ReadBeforeWait {
                        read();
                    }
                    while_carry((), |()| !mbarrier_test_wait(&bar, 0), |()| ());
                    if let Some(flag) = &flag {
                        out.store(t * 4, flag.load());
                    }
                    read();
                });
                if mode == Track::UnorderedReader {
                    if_(warp_idx().eq(2), read);
                }
                if let Some((own_data, own_bar)) = &own {
                    if_(warp_idx().eq(2), || {
                        let thr = tiled.get_slice(lane);
                        copy(&tiled, &thr.partition_s(&src), &thr.partition_d(own_data));
                        cp_async_commit_group();
                        cp_async_mbarrier_arrive_noinc(own_bar);
                        cp_async_wait_group(0);
                        // The wait orders warp 2's own copies before it, and nothing else.
                        read_from(own_data);
                        read();
                    });
                }
            },
        );
        match mode {
            Track::WaitGroupAfterPhase => if_(warp_idx().eq(0), || {
                cp_async_wait_group(0);
                read();
            }),
            Track::WaitGroupThenBarrier => {
                sync_threads();
                if_(warp_idx().eq(2), read);
            }
            _ => {}
        }
    })
}

/// The simulator holds `cp.async` copies tracked by an mbarrier to its phase: they land once
/// the phase has every other arrival and a thread waits on it, for the warps that waited. An
/// mbarrier count with no room for the tracked arrivals, a read before the phase completes, a
/// reader that did not wait, copies nothing tracks and an `mbarrier.init` while arrivals are
/// owed are reported, and so are a producer's ordinary stores read after the wait (the
/// asynchronous arrival orders only its `cp.async` copies), a reader whose `cp_async_wait_group`
/// covered only copies of its own, and a `.noinc` arrival at a phase that has all its arrivals.
/// A producer's own group wait orders the copies before it, whether the phase landed them first
/// or not, and before a warp that meets it at a barrier afterwards. The instruction needs sm_80
/// and PTX 7.0.
#[test]
fn cp_async_tracked_by_an_mbarrier() {
    let src: Vec<u32> = (1000..1128).collect();
    let run = |mode: Track| {
        let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&[0u32; 96 * 4])];
        tracked_handoff(mode).simulate(LaunchConfig::new([1, 1, 1], [96, 1, 1]), &mut args).map(|_| args)
    };
    // Each correct variant, and the warps whose reads it checks.
    for (mode, readers) in
        [(Track::Correct, 1..2), (Track::WaitGroupAfterPhase, 0..2), (Track::WaitGroupThenBarrier, 2..3)]
    {
        let args = run(mode).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        let out = args[1].to_vec::<u32>();
        for warp in readers {
            assert_eq!(out[warp * 128..(warp + 1) * 128], src[..], "{mode:?}: warp {warp}");
        }
    }
    for (mode, want, thread) in [
        (
            Track::CountTooSmall,
            "cp_async_mbarrier_arrive_noinc on an mbarrier whose phase is waiting for 16 arrival(s), all owed by \
             earlier cp.async.mbarrier.arrive: .noinc does not raise the pending count, so the mbarrier's count must \
             include one arrival per call",
            16,
        ),
        (
            Track::ExtraArrivals,
            "1 arrival(s) at an mbarrier whose phase is waiting for 32, 32 of which the cp.async copies tracked by \
             cp_async_mbarrier_arrive owe: the phase would complete with those copies still in flight",
            0,
        ),
        (
            Track::ReadBeforeWait,
            "reads smem1[0] while a cp.async is still writing it: the mbarrier that cp_async_mbarrier_arrive made \
             track it has not completed the phase",
            32,
        ),
        (
            Track::UnorderedReader,
            "data race: warp 2 reads smem1[0], which a cp.async tracked by cp_async_mbarrier_arrive wrote, without \
             having waited for the mbarrier phase that tracks it",
            64,
        ),
        (Track::Untracked, "reads smem1[0] while a cp.async is still writing it (missing cp_async_wait_group?)", 32),
        (
            Track::ReinitWhileOwed,
            "mbarrier.init on an mbarrier that still waits for 32 arrival(s) of cp.async.mbarrier.arrive",
            0,
        ),
        (
            Track::OrdinaryStoreNotReleased,
            "data race: warp 1 reads smem2[0], written by warp 0 since the last barrier (missing sync_threads?)",
            32,
        ),
        (
            Track::WaitGroupReader,
            "data race: warp 2 reads smem1[0], which a cp.async tracked by cp_async_mbarrier_arrive wrote, without \
             having waited for the mbarrier phase that tracks it",
            64,
        ),
        (
            Track::OnlyTxLeft,
            "cp_async_mbarrier_arrive_noinc on an mbarrier whose phase already has all its arrivals and only waits \
             for transaction bytes: .noinc does not raise the pending count, so the arrival has no place in the \
             phase (count it in mbarrier_init, or use cp_async_mbarrier_arrive)",
            0,
        ),
    ] {
        let e = run(mode).err().unwrap_or_else(|| panic!("{mode:?} should fail"));
        assert!(e.message.contains(want), "{mode:?}: {e}");
        assert_eq!(e.thread, Some([thread, 0, 0]), "{mode:?}: {e}");
        assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{mode:?}: {e}");
    }
    let kernel = tracked_handoff(Track::Correct);
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap(), 71, "test_wait.parity needs 7.1");
    let arrive =
        cute::ir::Op::CpAsyncMbarrierArrive { buffer: cute::ir::BufferId(0), index: cute::ir::ValueId(0), noinc: true };
    let need = cute::validate::target_requirement(&arrive).map(|r| (r.sm, r.ptx));
    assert_eq!(need, Some((80, 70)));
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum RegionBug {
    None,
    /// No barrier between the stores and the loads.
    ReadAfterWrite,
    /// No barrier between the loads and the stores that follow them.
    WriteAfterRead,
    /// Only warp 0 stores.
    Uninitialised,
}

/// Two warps exchange words through the dynamic shared-memory region: each thread stores its
/// word, and after a barrier reads the other warp's and stores again.
fn region_exchange(bug: RegionBug) -> Kernel {
    Kernel::builder("region_exchange").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let region = make_tensor(&get_dyn_smem::<u32>(None), Layout::col_major(64));
        if bug == RegionBug::Uninitialised {
            if_(t.lt(32), || region.store(t, t.to::<u32>() * 5u32));
        } else {
            region.store(t, t.to::<u32>() * 5u32);
        }
        if bug != RegionBug::ReadAfterWrite {
            sync_threads();
        }
        let v = region.load((t + 32) % 64);
        if bug != RegionBug::WriteAfterRead {
            sync_threads();
        }
        region.store(t, v + 1u32);
        out.store(t, v);
    })
}

/// The dynamic region is checked like the shared memory sized at build time: a read of what
/// another warp stored with no barrier between, a store over what another warp read and a read
/// of bytes nothing stored are reported, naming the region and the byte (the region has no
/// element type of its own: here the word at element 32 is byte 128); with the barriers the
/// exchange passes.
#[test]
fn dynamic_region_races_and_uninitialised_reads() {
    let run = |bug: RegionBug| {
        let mut args = vec![KernelArg::buffer(&[0u32; 64])];
        let launch = LaunchConfig::new([1, 1, 1], [64, 1, 1]).with_dynamic_smem(256);
        region_exchange(bug).simulate(launch, &mut args).map(|_| args[0].to_vec::<u32>())
    };
    let out = run(RegionBug::None).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<u32> = (0..64).map(|t| (t + 32) % 64 * 5).collect();
    assert_eq!(out, want);
    for (bug, message) in [
        (
            RegionBug::ReadAfterWrite,
            "data race: warp 0 reads byte 128 of the dynamic shared memory region (get_dyn_smem), written by warp 1 \
             since the last barrier (missing sync_threads?)",
        ),
        (
            RegionBug::WriteAfterRead,
            "data race: warp 0 overwrites byte 0 of the dynamic shared memory region (get_dyn_smem), which warp 1 \
             read since the last barrier (missing sync_threads?)",
        ),
        (
            RegionBug::Uninitialised,
            "reads byte 128 of the dynamic shared memory region (get_dyn_smem), which was never written",
        ),
    ] {
        let e = run(bug).err().unwrap_or_else(|| panic!("{bug:?} should fail"));
        assert!(e.message.contains(message), "{bug:?}: {e}");
        assert_eq!(e.thread, Some([0, 0, 0]), "{bug:?}: {e}");
        assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{bug:?}: {e}");
    }
}

// ---------------------------------------------------------------------------
// Memory-ordered and cache-hinted loads and stores, prefetch.
// ---------------------------------------------------------------------------

/// How [`flag_handoff`]'s warps hand data over through a flag.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Flag {
    /// A release store; an acquire load.
    ReleaseAcquire,
    /// A release store; a relaxed load, which acquires nothing.
    RelaxedLoad,
    /// A relaxed store, which releases nothing; an acquire load.
    RelaxedStore,
    /// `fence; st.relaxed`; `ld.relaxed; fence`, and then a plain re-read
    /// of the flag, which the relaxed loads that read it order.
    Fences,
    /// `fence; st.relaxed`; a relaxed load with no fence after it.
    NoConsumerFence,
    /// A relaxed store with no fence before it; `ld.relaxed; fence`.
    NoProducerFence,
    /// A plain store of the flag; an acquire load.
    PlainStore,
    /// A release store, and the data written after it.
    WriteAfterRelease,
    /// A release store and then a relaxed store of the same warp, which
    /// still releases (`st.release [M]; st.relaxed [M]`).
    ReleaseThenRelaxed,
    /// A release store, and then another warp's relaxed store, whose value
    /// the consumer reads: that store releases nothing.
    OtherWarpRelaxed,
    /// A release store, and then another warp's atomic add, which continues
    /// the release sequence.
    AtomicContinues,
    /// `fence; atom.exch`; a spin on an atomic or, then a fence.
    AtomicFences,
}

/// Warp 0 writes shared data and sets a flag (in shared or global memory);
/// warp 1 spins on the flag and reads the data; warp 2 may touch the flag
/// in between.
fn flag_handoff(mode: Flag, global_flag: bool) -> Kernel {
    Kernel::builder("flag_handoff").block_dim([96, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(96));
        let gflag = param_tensor::<u32>("gflag", Layout::col_major(1));
        let sflag = alloc_smem::<u32>(1, 4);
        let flag = if global_flag { gflag.ptr() } else { sflag };
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        let lane = t % 32;
        let cta = Some(MemoryScope::Cta);
        let ld = |sem| LoadOptions { sem, scope: cta, ..Default::default() };
        let st = |sem| StoreOptions { sem, scope: cta, ..Default::default() };
        if_(t.eq(0), || flag.store(0u32));
        sync_threads();
        if_(warp_idx().eq(0), || {
            if mode != Flag::WriteAfterRelease {
                data.store(lane, lane.to::<u32>() + 1u32);
            }
            match mode {
                Flag::RelaxedStore | Flag::NoProducerFence => store(&flag, 1u32, st(MemoryOrder::Relaxed)),
                Flag::Fences | Flag::NoConsumerFence => {
                    fence_acq_rel_cta();
                    store(&flag, 1u32, st(MemoryOrder::Relaxed));
                }
                Flag::PlainStore => flag.store(1u32),
                Flag::AtomicFences => {
                    fence_acq_rel_cta();
                    let _ = atomic_exch(&flag, 1u32);
                }
                _ => store(&flag, 1u32, st(MemoryOrder::Release)),
            }
            match mode {
                Flag::ReleaseThenRelaxed => store(&flag, 2u32, st(MemoryOrder::Relaxed)),
                Flag::WriteAfterRelease => data.store(lane, lane.to::<u32>() + 1u32),
                _ => {}
            }
        });
        if_(warp_idx().eq(2), || match mode {
            Flag::OtherWarpRelaxed => store(&flag, 2u32, st(MemoryOrder::Relaxed)),
            Flag::AtomicContinues => {
                let _ = atomic_add(&flag, 1u32);
            }
            _ => {}
        });
        if_(warp_idx().eq(1), || {
            match mode {
                Flag::RelaxedLoad | Flag::NoConsumerFence => {
                    while_carry((), |()| load(&flag, ld(MemoryOrder::Relaxed)).eq(0u32), |()| ());
                }
                Flag::Fences | Flag::NoProducerFence => {
                    while_carry((), |()| load(&flag, ld(MemoryOrder::Relaxed)).eq(0u32), |()| ());
                    fence_acq_rel_cta();
                }
                Flag::AtomicFences => {
                    while_carry((), |()| atomic_or(&flag, 0u32).eq(0u32), |()| ());
                    fence_acq_rel_cta();
                }
                _ => while_carry((), |()| load(&flag, ld(MemoryOrder::Acquire)).eq(0u32), |()| ()),
            }
            let seen = if mode == Flag::Fences { flag.load() } else { 1u32.into_value() };
            out.store(t, data.load(lane) * seen);
        });
    })
}

/// The simulator orders accesses through release and acquire patterns as
/// the PTX memory model defines them, and finds the races where a pattern
/// is incomplete or broken.
#[test]
fn flags_order_data_through_release_and_acquire() {
    let run = |mode: Flag, global: bool| {
        let mut args = vec![KernelArg::buffer(&[0u32; 96]), KernelArg::buffer(&[7u32])];
        flag_handoff(mode, global).simulate(LaunchConfig::new([1, 1, 1], [96, 1, 1]), &mut args).map(|_| args)
    };
    for mode in
        [Flag::ReleaseAcquire, Flag::Fences, Flag::ReleaseThenRelaxed, Flag::AtomicContinues, Flag::AtomicFences]
    {
        for global in [false, true] {
            let args = run(mode, global).unwrap_or_else(|e| panic!("{mode:?}, global flag {global}: {e}"));
            assert_eq!(args[0].to_vec::<u32>()[32..36], [1, 2, 3, 4], "{mode:?}");
        }
    }
    for mode in [
        Flag::RelaxedLoad,
        Flag::RelaxedStore,
        Flag::NoConsumerFence,
        Flag::NoProducerFence,
        Flag::WriteAfterRelease,
        Flag::OtherWarpRelaxed,
    ] {
        for global in [false, true] {
            let e = run(mode, global).err().unwrap_or_else(|| panic!("{mode:?}, global flag {global} should race"));
            assert!(e.message.contains("data race: warp 1 reads smem1[0], written by warp 0"), "{mode:?}: {e}");
            assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
        }
    }
    // A plain store of a shared flag races with the ordered loads of it; a
    // global flag is not checked, and orders nothing.
    let e = run(Flag::PlainStore, false).unwrap_err();
    assert!(e.message.contains("data race: warp 1 reads smem0[0], written by warp 0"), "{e}");
    let e = run(Flag::PlainStore, true).unwrap_err();
    assert!(e.message.contains("data race: warp 1 reads smem1[0], written by warp 0"), "{e}");
}

/// Strong accesses (ordered, volatile, atomic) to one location never race
/// with each other, but do with plain ones that nothing orders against them.
#[test]
fn strong_accesses_race_only_with_plain_ones() {
    let kernel = |plain_second: bool, atomic: bool| {
        Kernel::builder("strong").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(64));
            let cell = alloc_smem::<u32>(1, 4);
            let t = thread_idx().x;
            let relaxed =
                StoreOptions { sem: MemoryOrder::Relaxed, scope: Some(MemoryScope::Cta), ..Default::default() };
            if_(t.eq(0), || cell.store(0u32));
            sync_threads();
            if_(warp_idx().eq(0), || store(&cell, 1u32, relaxed));
            if_(warp_idx().eq(1), || {
                if plain_second {
                    cell.store(2u32);
                } else if atomic {
                    let _ = atomic_add(&cell, 2u32);
                } else {
                    store(&cell, 2u32, StoreOptions { sem: MemoryOrder::Volatile, ..relaxed });
                }
            });
            let v = load(
                &cell,
                LoadOptions { sem: MemoryOrder::Relaxed, scope: Some(MemoryScope::Cta), ..Default::default() },
            );
            out.store(t, v);
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    for atomic in [false, true] {
        kernel(false, atomic)
            .simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let e = simulate_err(&kernel(true, false), 64, &mut args);
    assert!(e.message.contains("data race: warps 0 and 1 both write smem0[0]"), "{e}");
    // A plain write and an atomic of another warp race too.
    let mixed = Kernel::builder("plain_and_atomic").block_dim([64, 1, 1]).build(|| {
        let cell = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        if_(t.eq(0), || cell.store(0u32));
        sync_threads();
        if_(t.eq(32), || {
            let _ = atomic_add(&cell, 1u32);
        });
        if_(t.eq(0), || cell.store(5u32));
    });
    let e = simulate_err(&mixed, 64, &mut []);
    assert!(e.message.contains("data race: warps 1 and 0 both write smem0[0]"), "{e}");
}

/// Ignore a loaded value (the loads under test fail before it is used).
fn discard<T>(_: T) {}

/// The message of a kernel build that applies `f` to a global and a shared
/// pointer, which must fail, and the line `f` is on.
#[track_caller]
fn access_misuse(f: impl Fn(&Ptr<u32>, &Ptr<u32>)) -> (String, u32) {
    let line = std::panic::Location::caller().line();
    let msg = build_panic(move || {
        Kernel::builder("access_misuse").block_dim([32, 1, 1]).build(|| {
            let g = param_tensor::<u32>("g", Layout::col_major(64));
            let s = alloc_smem::<u32>(64, 16);
            f(&g.ptr(), &s);
        });
    });
    (msg, line)
}

/// Loads, stores and prefetches whose memory attributes PTX does not allow
/// are rejected when the kernel is built, at the user's line.
#[test]
fn memory_attributes_follow_ptx_rules() {
    use CacheEvictionPriority::*;
    use MemoryOrder::*;
    let lo = |sem| LoadOptions { sem, ..Default::default() };
    let so = |sem| StoreOptions { sem, ..Default::default() };
    let gpu = Some(MemoryScope::Gpu);
    let cases: Vec<((String, u32), &str)> = vec![
        (access_misuse(|g, _| discard(load(g, lo(Release)))), "a load cannot be a release (MemoryOrder::Release)"),
        (access_misuse(|g, _| store(g, 1u32, so(Acquire))), "a store cannot be an acquire (MemoryOrder::Acquire)"),
        (access_misuse(|g, _| discard(load(g, lo(AcqRel)))), "acq_rel (MemoryOrder::AcqRel) is an order for atomics"),
        (access_misuse(|g, _| store(g, 1u32, so(Constant))), "a store cannot be constant"),
        (
            access_misuse(|g, _| discard(load(g, LoadOptions { cop: Some(CacheOperator::Cg), ..lo(Relaxed) }))),
            "a relaxed load takes no cache operator (.cg): PTX allows cache operators only on weak loads",
        ),
        (
            access_misuse(|g, _| store(g, 1u32, StoreOptions { cop: Some(CacheOperator::Wt), ..so(Release) })),
            "a release store takes no cache operator (.wt)",
        ),
        (
            access_misuse(|g, _| {
                discard(load(
                    g,
                    LoadOptions { cop: Some(CacheOperator::Cs), level1_eviction_priority: Some(EvictLast), ..lo(Weak) },
                ))
            }),
            "a cache operator (.cs) and an L1 eviction priority (.L1::evict_last) are separate forms of PTX's ld",
        ),
        (
            access_misuse(|g, _| {
                discard(load(g, LoadOptions { level1_eviction_priority: Some(EvictFirst), ..lo(Volatile) }))
            }),
            "a volatile load cannot take an L1 eviction priority (.L1::evict_first)",
        ),
        (
            access_misuse(|g, _| store(g, 1u32, StoreOptions { cop: Some(CacheOperator::Cg), ..so(Volatile) })),
            "a volatile store cannot take a cache operator (.cg): PTX's st.volatile takes no hints",
        ),
        (
            access_misuse(|g, _| {
                discard(load_vector::<u32, 2>(g, LoadOptions { scope: Some(MemoryScope::Sys), ..lo(Mmio) }))
            }),
            "an MMIO load (.mmio) moves one element; this one moves 2",
        ),
        (
            access_misuse(|g, _| discard(load(g, LoadOptions { scope: gpu, ..lo(Mmio) }))),
            "an MMIO load has system scope",
        ),
        (access_misuse(|_, s| discard(load(s, lo(Mmio)))), "an MMIO load (.mmio) needs global memory"),
        (
            access_misuse(|g, _| store(g, 1u32, StoreOptions { cop: Some(CacheOperator::Lu), ..so(Weak) })),
            ".lu is a load cache operator; stores take .wb, .cg, .cs or .wt",
        ),
        (
            access_misuse(|g, _| discard(load(g, LoadOptions { cop: Some(CacheOperator::Wb), ..lo(Weak) }))),
            ".wb is a store cache operator; loads take .ca, .cg, .cs, .lu or .cv",
        ),
        (
            access_misuse(|_, s| {
                discard(load(s, LoadOptions { level1_eviction_priority: Some(EvictLast), ..lo(Acquire) }))
            }),
            "an L1 eviction priority (.L1::evict_last) needs global memory, but this load is to shared memory",
        ),
        (
            access_misuse(|_, s| {
                discard(load(s, LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size64B), ..lo(Weak) }))
            }),
            "a prefetch size (.L2::64B) needs global memory",
        ),
        (
            access_misuse(|_, s| prefetch(s, PrefetchOptions::default())),
            "prefetch takes a global or local address (PTX); shared memory is not cached",
        ),
        (
            access_misuse(|g, _| {
                prefetch(g, PrefetchOptions { evict_priority: Some(EvictLast), ..Default::default() })
            }),
            "an eviction priority applies to an L2 prefetch",
        ),
        (
            access_misuse(|g, _| {
                let opts = PrefetchOptions {
                    cache_level: Some(PrefetchCacheLevel::L2),
                    evict_priority: Some(EvictFirst),
                    ..Default::default()
                };
                prefetch(g, opts)
            }),
            "a prefetch's eviction priority is evict_last or evict_normal",
        ),
        (
            access_misuse(|g, _| {
                prefetch(
                    g,
                    PrefetchOptions { cache_level: Some(PrefetchCacheLevel::L2), uniform: true, ..Default::default() },
                )
            }),
            "a uniform prefetch (prefetchu) fills the L1 cache only",
        ),
        (
            access_misuse(|g, _| {
                discard(load(g, LoadOptions { level1_eviction_priority: Some(EvictLast), ..lo(Mmio) }))
            }),
            "an MMIO load (.mmio) takes no cache operator, eviction priority, prefetch size or cache policy",
        ),
        (
            access_misuse(|g, _| {
                store(g, 1u32, StoreOptions { cache_policy: Some(Int64::from(0i64)), ..so(Mmio) });
            }),
            "an MMIO store (.mmio) takes no cache operator, eviction priority, prefetch size or cache policy",
        ),
        (
            access_misuse(|g, _| {
                discard(load(g, LoadOptions { cache_policy: Some(Int64::from(0i64)), ..lo(Volatile) }))
            }),
            "a volatile load cannot take an L2 cache policy (.L2::cache_hint): PTX's ld.volatile takes only a prefetch \
             size",
        ),
        (
            access_misuse(|g, _| discard(load(g, LoadOptions { cop: Some(CacheOperator::Cg), ..lo(Volatile) }))),
            "a volatile load cannot take a cache operator (.cg): PTX's ld.volatile takes only a prefetch size",
        ),
        (
            access_misuse(|_, s| discard(load(s, lo(Constant)))),
            "a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant) needs global memory, but this load \
             is to shared memory",
        ),
        (
            access_misuse(|_, s| discard(load(s, LoadOptions { cache_policy: Some(Int64::from(0i64)), ..lo(Weak) }))),
            "an L2 cache policy (.L2::cache_hint) needs global memory, but this load is to shared memory",
        ),
        (
            access_misuse(|_, s| {
                store(s, 1u32, StoreOptions { cache_policy: Some(Int64::from(0i64)), ..so(Release) });
            }),
            "an L2 cache policy (.L2::cache_hint) needs global memory, but this store is to shared memory",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        // The access is on the line of the `access_misuse` call or a few after it.
        let at = msg
            .split("tests/diagnostics/memory_model.rs:")
            .nth(1)
            .and_then(|r| r.split(':').next()?.parse::<u32>().ok());
        assert!(at.is_some_and(|at| (line..line + 8).contains(&at)), "the error should point at line {line}: {msg}");
    }
    // A Boolean or per-thread memory takes plain accesses only.
    let msg = build_panic(|| {
        Kernel::build("bool_flag", || {
            let b = param_tensor::<bool>("b", Layout::col_major(4));
            discard(load(&b.ptr(), lo(Relaxed)));
        });
    });
    assert!(msg.contains("not Booleans"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("local_flag", || {
            let stack = make_rmem_tensor::<u32>(Layout::col_major(4));
            store(&stack.ptr(), 1u32, so(Relaxed));
        });
    });
    assert!(msg.contains("per-thread (local) memory takes only plain stores"), "{msg}");
    // PTX prefetches with an eviction priority only from global memory.
    let msg = build_panic(|| {
        Kernel::build("local_prefetch", || {
            let stack = make_rmem_tensor::<u32>(Layout::col_major(4));
            let opts = PrefetchOptions {
                cache_level: Some(PrefetchCacheLevel::L2),
                evict_priority: Some(EvictLast),
                ..Default::default()
            };
            prefetch(&stack.ptr(), opts);
        });
    });
    assert!(
        msg.contains("a prefetch with an eviction priority (prefetch.global.L2::evict_*) needs global memory"),
        "{msg}"
    );
    // `ss=` must match the pointer (another CTA's shared memory: tests/dsmem.rs).
    let (msg, _) = access_misuse(|g, _| discard(load(g, LoadOptions { ss: Some(SharedSpace::Cta), ..lo(Weak) })));
    assert!(
        msg.contains("ss=SharedSpace::Cta addresses this CTA's shared memory, but the pointer is to gmem"),
        "{msg}"
    );
    let (msg, _) = access_misuse(|g, _| store(g, 1u32, StoreOptions { ss: Some(SharedSpace::Cluster), ..so(Weak) }));
    assert!(msg.contains("ss=SharedSpace::Cluster addresses the shared memory of a CTA of the cluster"), "{msg}");
    // A vector access must be provably aligned.
    let (msg, _) = access_misuse(|g, _| discard(load_vector::<u32, 4>(&g.offset(2usize), lo(Weak))));
    assert!(
        msg.contains("load_vector of 4 elements: cannot prove that the offset 2 is aligned to 4 elements"),
        "{msg}"
    );
}

/// What each memory attribute needs from the target, from the PTX ISA.
#[test]
fn memory_attribute_target_requirements() {
    use cute::validate::check_target;
    let kernel = |opts: LoadOptions| {
        Kernel::build("attribute_target", move || {
            let g = param_tensor::<u32>("g", Layout::col_major(64));
            let v = load(&g.ptr(), opts);
            g.store(1, v);
        })
    };
    let ordered = |scope| LoadOptions { sem: MemoryOrder::Acquire, scope: Some(scope), ..Default::default() };
    // Ordered loads need nothing new, except cluster scope (sm_90, PTX 7.8).
    assert_eq!(check_target(kernel(ordered(MemoryScope::Gpu)).ir(), Arch::Sm80).unwrap(), 70);
    let cluster = kernel(ordered(MemoryScope::Cluster));
    let e = check_target(cluster.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("an ordered load or store with cluster scope needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
    assert_eq!(check_target(cluster.ir(), Arch::Sm90).unwrap(), 78);
    // The hints need PTX 7.4, and MMIO 8.2.
    let evict = LoadOptions { level1_eviction_priority: Some(CacheEvictionPriority::EvictLast), ..Default::default() };
    assert_eq!(check_target(kernel(evict).ir(), Arch::Sm80).unwrap(), 74);
    let wide = LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size256B), ..Default::default() };
    assert_eq!(check_target(kernel(wide).ir(), Arch::Sm80).unwrap(), 74);
    let mmio = LoadOptions { sem: MemoryOrder::Mmio, ..Default::default() };
    assert_eq!(check_target(kernel(mmio).ir(), Arch::Sm80).unwrap(), 82);
    assert_eq!(check_target(kernel(mmio).ir(), Arch::Sm89).unwrap(), 82);
    let cop = LoadOptions { cop: Some(CacheOperator::Cs), ..Default::default() };
    assert_eq!(check_target(kernel(cop).ir(), Arch::Sm80).unwrap(), 70);
}

/// The message of a panic in `f`, run outside a kernel (copy atoms are made
/// before one is built).
fn atom_panic(f: impl FnOnce()) -> String {
    build_panic(f)
}

/// G2R and R2G atoms check their memory attributes by PTX's rules when they
/// are made, and copies check the memory each side is in.
#[test]
fn global_copy_atom_misuse() {
    use CacheEvictionPriority::*;
    let g2r = |attrs: G2RAttrs| atom_panic(move || drop(make_copy_atom::<f32>(CopyOp::G2R { bits: 128, attrs })));
    let r2g = |attrs: R2GAttrs| atom_panic(move || drop(make_copy_atom::<f32>(CopyOp::R2G { bits: 128, attrs })));
    let order = |memory_order| G2RAttrs { memory_order, ..G2RAttrs::default() };
    for (msg, want) in [
        (g2r(order(MemoryOrder::Release)), "make_copy_atom(CopyG2ROp, f32): a load cannot be a release"),
        (
            r2g(R2GAttrs { memory_order: MemoryOrder::Acquire, ..R2GAttrs::default() }),
            "make_copy_atom(CopyR2GOp, f32): a store cannot be an acquire",
        ),
        (
            g2r(G2RAttrs { memory_scope: MemoryScope::Sys, ..order(MemoryOrder::Mmio) }),
            "an MMIO load (.mmio) moves one element; this one moves 4",
        ),
        (
            atom_panic(|| {
                let attrs = R2GAttrs { memory_order: MemoryOrder::Mmio, ..R2GAttrs::default() };
                drop(make_copy_atom::<f32>(CopyOp::R2G { bits: 32, attrs }))
            }),
            "an MMIO store has system scope (PTX allows only .mmio.relaxed.sys): set the scope to MemoryScope::Sys, \
             not Cta",
        ),
        (
            g2r(G2RAttrs { invariant: true, ..order(MemoryOrder::Acquire) }),
            "a non-coherent load (.nc, invariant) is weak; it cannot also be acquire",
        ),
        (
            g2r(G2RAttrs {
                load_cache_mode: LoadCacheMode::Streaming,
                l1c_evict_priority: EvictLast,
                ..G2RAttrs::default()
            }),
            "a cache operator (.cs) and an L1 eviction priority (.L1::evict_last) are separate forms of PTX's ld",
        ),
        (
            g2r(G2RAttrs { invariant: true, load_cache_mode: LoadCacheMode::LastUse, ..G2RAttrs::default() }),
            "a non-coherent load (ld.global.nc) takes .ca, .cg or .cs, not .lu",
        ),
        (
            r2g(R2GAttrs {
                store_cache_mode: StoreCacheMode::WriteThrough,
                ..R2GAttrs { memory_order: MemoryOrder::Release, ..R2GAttrs::default() }
            }),
            "a release store takes no cache operator (.wt)",
        ),
        (
            g2r(G2RAttrs { shared_space: SharedSpace::Cluster, ..G2RAttrs::default() }),
            "shared_space=SharedSpace::Cluster addresses another CTA's shared memory (DSMEM), which this \
             global-memory copy does not reach",
        ),
        (
            atom_panic(|| drop(make_copy_atom::<f32>(CopyOp::G2R { bits: 48, attrs: G2RAttrs::default() }))),
            "make_copy_atom(CopyG2ROp): num_bits_per_copy 48 is not a whole number of 32-bit elements",
        ),
    ] {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
    }
    // The memory on each side of the copy.
    let copy_misuse = |which: u32| {
        build_panic(move || {
            Kernel::builder("copy_misuse").block_dim([32, 1, 1]).build(|| {
                let g = param_tensor::<f32>("g", Layout::col_major(32 * 4));
                let s = make_smem_tensor::<f32>(Layout::col_major(32 * 4));
                let t = thread_idx().x;
                let part = |x: &Tensor<f32>| local_tile(x, 4, t);
                let frag = make_fragment::<f32>(4);
                frag.fill(0.0);
                let g2r = make_copy_atom::<f32>(CopyOp::G2R { bits: 128, attrs: G2RAttrs::default() });
                let r2g = make_copy_atom::<f32>(CopyOp::R2G { bits: 128, attrs: R2GAttrs::default() });
                let universal = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
                match which {
                    0 => copy(&g2r, &part(&s), &frag),
                    1 => copy(&g2r, &part(&g), &part(&s)),
                    2 => copy(&r2g, &part(&s), &part(&g)),
                    3 => copy(&r2g, &frag, &part(&s)),
                    _ => {
                        let policy = Int64::from(0i64);
                        copy_with(
                            &universal,
                            &part(&g),
                            &frag,
                            CopyOptions { cache_policy: Some(policy), ..Default::default() },
                        )
                    }
                }
            });
        })
    };
    for (which, want) in [
        (0, "copy: CopyG2ROp reads global memory"),
        (1, "copy: CopyG2ROp writes registers (a fragment or make_rmem_tensor), not Shared memory"),
        (2, "copy: CopyR2GOp reads registers (a fragment or make_rmem_tensor), not Shared memory"),
        (3, "copy: CopyR2GOp writes global memory"),
        (4, "copy: a cache policy is for CopyG2ROp and CopyR2GOp atoms"),
    ] {
        let msg = copy_misuse(which);
        assert!(msg.contains(want), "expected {want:?} in {msg}");
    }
    // Cluster scope needs sm_90.
    let kernel = Kernel::builder("cluster_atom").block_dim([32, 1, 1]).build(|| {
        let g = param_tensor::<f32>("g", Layout::col_major(32 * 4));
        let t = thread_idx().x;
        let attrs =
            G2RAttrs { memory_order: MemoryOrder::Relaxed, memory_scope: MemoryScope::Cluster, ..G2RAttrs::default() };
        let frag = make_fragment::<f32>(4);
        copy(&make_copy_atom::<f32>(CopyOp::G2R { bits: 128, attrs }), &local_tile(&g, 4, t), &frag);
        g.store(t * 4, frag.load(0) + frag.load(3));
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert!(e.message.contains("an ordered load or store with cluster scope needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "the error should point at the copy: {e}");
}

/// A non-coherent load is only for memory that no thread writes while the
/// kernel runs: the simulator reports a write of what one read, and a read
/// of what the kernel wrote, also between blocks.
#[test]
fn non_coherent_loads_of_written_memory() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Order {
        WriteThenRead,
        ReadThenWrite,
        OtherBlockWrites,
        Separate,
    }
    let kernel = |order: Order| {
        Kernel::builder("nc_written").block_dim([32, 1, 1]).build(move || {
            let x = param_tensor::<u32>("x", Layout::col_major(256));
            let out = param_tensor::<u32>("out", Layout::col_major(64));
            let t = thread_idx().x;
            let b = block_idx().x;
            let constant = LoadOptions { sem: MemoryOrder::Constant, ..Default::default() };
            let row = b * 32 + t;
            match order {
                Order::WriteThenRead => {
                    x.store(t, 1u32);
                    out.store(row, load(&x.ptr().offset(t), constant));
                }
                Order::ReadThenWrite => {
                    out.store(row, load(&x.ptr().offset(t), constant));
                    x.store((t + 1) % 32, 1u32);
                }
                Order::OtherBlockWrites => {
                    out.store(row, load(&x.ptr().offset(t), constant));
                    if_(b.eq(1), || x.store(t + 16, 2u32));
                }
                Order::Separate => {
                    out.store(row, load(&x.ptr().offset(t), constant));
                    x.store(t + 128, 2u32);
                }
            }
        })
    };
    let run = |order: Order| {
        let mut args = vec![KernelArg::buffer(&[0u32; 256]), KernelArg::buffer(&[0u32; 64])];
        kernel(order).simulate(LaunchConfig::new([2, 1, 1], [32, 1, 1]), &mut args)
    };
    run(Order::Separate).unwrap_or_else(|e| panic!("{e}"));
    let e = run(Order::WriteThenRead).unwrap_err();
    assert!(
        e.message.contains(
            "a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant) reads x[0], which the kernel wrote"
        ),
        "{e}"
    );
    let e = run(Order::ReadThenWrite).unwrap_err();
    assert!(
        e.message.contains(
            "writes x[1], which a non-coherent load (ld.global.nc: invariant, or MemoryOrder::Constant) reads"
        ),
        "{e}"
    );
    let e = run(Order::OtherBlockWrites).unwrap_err();
    assert!(e.message.contains("writes x[16], which a non-coherent load"), "{e}");
    assert_eq!(e.block, [1, 0, 0]);
}

/// The memory orders of G2R and R2G atoms take part in the simulator's
/// happens-before: a release copy and an acquire copy of a flag order the
/// data around them, and weaker orders do not.
#[test]
fn copy_atom_orders_order_data() {
    let handoff = |store: MemoryOrder, load: MemoryOrder| {
        Kernel::builder("atom_handoff").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(64));
            let flag = param_tensor::<u32>("flag", Layout::col_major(1));
            let data = make_smem_tensor::<u32>(Layout::col_major(32));
            let t = thread_idx().x;
            let lane = t % 32;
            let gpu = MemoryScope::Gpu;
            if_(t.eq(0), || flag.store(0, 0u32));
            sync_threads();
            if_(warp_idx().eq(0), || {
                data.store(lane, lane.to::<u32>() * 3u32);
                let one = make_fragment::<u32>(1);
                one.fill(1u32);
                let r2g = make_copy_atom::<u32>(CopyOp::R2G {
                    bits: 32,
                    attrs: R2GAttrs { memory_order: store, memory_scope: gpu, ..Default::default() },
                });
                copy(&r2g, &one, &flag);
            });
            if_(warp_idx().eq(1), || {
                let g2r = make_copy_atom::<u32>(CopyOp::G2R {
                    bits: 32,
                    attrs: G2RAttrs { memory_order: load, memory_scope: gpu, ..Default::default() },
                });
                let unset = |()| {
                    let seen = make_fragment::<u32>(1);
                    copy(&g2r, &flag, &seen);
                    seen.load(0).eq(0u32)
                };
                while_carry((), unset, |()| ());
                out.store(t, data.load(lane));
            });
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 64]), KernelArg::buffer(&[5u32])];
    handoff(MemoryOrder::Release, MemoryOrder::Acquire)
        .simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[33], 3);
    for (store, load) in [(MemoryOrder::Relaxed, MemoryOrder::Acquire), (MemoryOrder::Release, MemoryOrder::Volatile)] {
        let e = simulate_err(&handoff(store, load), 64, &mut args);
        assert!(e.message.contains("data race: warp 1 reads smem0[0], written by warp 0"), "{store:?}/{load:?}: {e}");
    }
}

/// How [`mixed_sizes`] accesses one shared cell from two warps.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mixed {
    /// A 4-byte relaxed store, then another warp's 1-byte relaxed load of it.
    WideStoreNarrowLoad,
    /// A 2-byte relaxed store into a 4-byte word, then another warp's 4-byte
    /// relaxed load of the word.
    NarrowStoreWideLoad,
    /// A 1-byte relaxed load, then another warp's 4-byte relaxed store.
    NarrowLoadWideStore,
    /// A 4-byte relaxed store, then another warp's 2-byte volatile store.
    TwoStores,
    /// A 32-bit atomic, then another warp's 64-bit atomic on the same bytes.
    Atomics,
    /// A relaxed vector store of four bytes, then another warp's relaxed load
    /// of one of them: an element of a vector is an access of its own.
    VectorElement,
    /// The same vector store, then another warp's 2-byte relaxed load of two
    /// of its elements.
    VectorPair,
}

/// Two warps access an 8-byte shared cell, with no barrier between them.
fn mixed_sizes(mode: Mixed) -> Kernel {
    Kernel::builder("mixed_sizes").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let cell = alloc_smem::<u64>(1, 8);
        let word = cell.cast::<u32>();
        let t = thread_idx().x;
        let cta = Some(MemoryScope::Cta);
        let ld = LoadOptions { sem: MemoryOrder::Relaxed, scope: cta, ..Default::default() };
        let st = StoreOptions { sem: MemoryOrder::Relaxed, scope: cta, ..Default::default() };
        if_(t.eq(0), || cell.store(0u64));
        sync_threads();
        let (w0, w1) = (warp_idx().eq(0), warp_idx().eq(1));
        match mode {
            Mixed::WideStoreNarrowLoad => {
                if_(w0, || store(&word, 7u32, st));
                if_(w1, || out.store(t, load(&word.cast::<u8>(), ld).to::<u32>()));
            }
            Mixed::NarrowStoreWideLoad => {
                if_(w0, || store(&word.cast::<u16>().offset(1usize), 7u16, st));
                if_(w1, || out.store(t, load(&word, ld)));
            }
            Mixed::NarrowLoadWideStore => {
                if_(w1, || out.store(t, load(&word.cast::<u8>().offset(3usize), ld).to::<u32>()));
                if_(w0, || store(&word, 7u32, st));
            }
            Mixed::TwoStores => {
                if_(w0, || store(&word, 7u32, st));
                if_(w1, || store(&word.cast::<u16>(), 9u16, StoreOptions { sem: MemoryOrder::Volatile, ..st }));
            }
            Mixed::Atomics => {
                if_(w0, || {
                    let _ = atomic_add(&word, 1u32);
                });
                if_(w1, || {
                    let _ = atomic_add(&cell, 1u64);
                });
            }
            Mixed::VectorElement | Mixed::VectorPair => {
                let one = Uint8::from(1u8);
                if_(w0, || store_vector(&word.cast::<u8>(), [one; 4], st));
                if_(w1, || {
                    let v = if mode == Mixed::VectorElement {
                        load(&word.cast::<u8>().offset(2usize), ld).to::<u32>()
                    } else {
                        load(&word.cast::<u16>().offset(1usize), ld).to::<u32>()
                    };
                    out.store(t, v);
                });
            }
        }
    })
}

/// PTX's memory model makes two strong accesses morally strong (so they
/// neither race nor fail to synchronise) only if they cover the same bytes:
/// strong accesses of different sizes to overlapping bytes, unordered, are a
/// mixed-size data race. The elements of a vector are accesses of their own.
#[test]
fn strong_accesses_of_different_sizes_race() {
    let mixed =
        "that do not cover the same bytes are not morally strong in PTX's memory model (a mixed-size data race)";
    for (mode, want) in [
        (
            Mixed::WideStoreNarrowLoad,
            "data race: warp 1 reads smem0[0], written by warp 0: both accesses are strong, but a 1-byte read and a \
             4-byte write",
        ),
        (
            Mixed::NarrowStoreWideLoad,
            "data race: warp 1 reads smem0[0], written by warp 0: both accesses are strong, but a 4-byte read and a \
             2-byte write",
        ),
        (
            Mixed::NarrowLoadWideStore,
            "data race: warp 0 overwrites smem0[0], which warp 1 read since the last barrier: both accesses are \
             strong, but a 4-byte write and a 1-byte read",
        ),
        (
            Mixed::TwoStores,
            "data race: warps 0 and 1 both write smem0[0] between barriers: both writes are strong, but a 2-byte \
             write and a 4-byte write",
        ),
        (
            Mixed::Atomics,
            "data race: warp 1 reads smem0[0], written by warp 0: both accesses are strong, but an 8-byte read and \
             a 4-byte write",
        ),
        (
            Mixed::VectorPair,
            "data race: warp 1 reads smem0[0], written by warp 0: both accesses are strong, but a 2-byte read and a \
             1-byte write",
        ),
    ] {
        let e = simulate_err(&mixed_sizes(mode), 64, &mut [KernelArg::buffer(&[0u32; 64])]);
        assert!(e.message.contains(want) && e.message.contains(mixed), "{mode:?}: {e}");
        assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
    }
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    mixed_sizes(Mixed::VectorElement)
        .simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[32], 1);
}

/// Warp 0 writes shared data and releases a 32-bit flag (in shared or global
/// memory), as one 32-bit store or as a vector of four bytes; warp 1 spins on
/// byte 2 of the flag with a 1-byte acquire and then reads the data.
fn byte_flag_handoff(vector_release: bool, global_flag: bool) -> Kernel {
    Kernel::builder("byte_flag").block_dim([64, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let gflag = param_tensor::<u32>("gflag", Layout::col_major(1));
        let sflag = alloc_smem::<u32>(1, 4);
        let flag = if global_flag { gflag.ptr() } else { sflag };
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        let t = thread_idx().x;
        let lane = t % 32;
        let cta = Some(MemoryScope::Cta);
        if_(t.eq(0), || flag.store(0u32));
        sync_threads();
        if_(warp_idx().eq(0), || {
            data.store(lane, lane.to::<u32>() + 1u32);
            let release = StoreOptions { sem: MemoryOrder::Release, scope: cta, ..Default::default() };
            if vector_release {
                store_vector(&flag.cast::<u8>(), [Uint8::from(1u8); 4], release);
            } else {
                store(&flag, 0x0101_0101u32, release);
            }
        });
        if_(warp_idx().eq(1), || {
            let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: cta, ..Default::default() };
            while_carry((), |()| load(&flag.cast::<u8>().offset(2usize), acquire).eq(0u8), |()| ());
            out.store(t, data.load(lane));
        });
    })
}

/// An acquire synchronises with a release only if it reads the bytes of one
/// of the release's elements exactly: a 1-byte acquire of a 4-byte release
/// orders nothing (and on a shared flag races with it), while a 1-byte
/// acquire of an element of a vector of bytes synchronises with it.
#[test]
fn acquires_synchronise_only_with_releases_of_the_same_bytes() {
    let run = |vector: bool, global: bool| {
        let mut args = vec![KernelArg::buffer(&[0u32; 64]), KernelArg::buffer(&[7u32])];
        byte_flag_handoff(vector, global).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).map(|_| args)
    };
    for global in [false, true] {
        let args = run(true, global).unwrap_or_else(|e| panic!("global flag {global}: {e}"));
        assert_eq!(args[0].to_vec::<u32>()[32..36], [1, 2, 3, 4]);
    }
    // A global flag is not race-checked, but the acquire orders nothing, so
    // the data races.
    let e = run(false, true).unwrap_err();
    assert!(e.message.contains("data race: warp 1 reads smem1[0], written by warp 0 since the last barrier"), "{e}");
    // A shared one races with the release.
    let e = run(false, false).unwrap_err();
    assert!(
        e.message.contains("data race: warp 1 reads smem0[0], written by warp 0: both accesses are strong, but a 1-byte read and a 4-byte write"),
        "{e}"
    );
}

/// PTX requires every access to be aligned to its size. The builder proves
/// an access's offset from the start of its buffer a multiple of its size,
/// so the buffer must be declared at least that aligned.
#[test]
fn accesses_need_a_buffer_aligned_to_their_size() {
    let build = |which: u32| {
        build_panic(move || {
            Kernel::builder("aligned").block_dim([32, 1, 1]).build(|| {
                let g = param_tensor::<u32>("g", Layout::col_major(128));
                let first = alloc_smem::<u32>(1, 4);
                let words = alloc_smem::<u32>(4, 4);
                let t = thread_idx().x;
                first.store(t.to::<u32>());
                let tile = make_tensor(&words, Layout::col_major(4));
                match which {
                    0 => discard(load_vector::<u32, 4>(
                        &words,
                        LoadOptions { sem: MemoryOrder::Volatile, ..Default::default() },
                    )),
                    1 => store_vector(&words, [Uint32::from(1u32); 2], StoreOptions::default()),
                    2 => discard(words.cast::<u64>().load()),
                    3 => copy(&make_copy_atom::<u32>(CopyOp::Universal { bits: 128 }), &local_tile(&g, 4, t), &tile),
                    4 => copy(
                        &make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: false }),
                        &local_tile(&g, 4, t),
                        &tile,
                    ),
                    5 => discard(atomic_add(&words.cast::<u64>(), 1u64)),
                    _ => {
                        // ldmatrix reads a 16-byte row at each thread's address.
                        let halves = make_tensor(&alloc_smem::<F16>(32 * 8, 8), Layout::row_major((32, 8)));
                        let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
                        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::row_major((4, 2)));
                        let frag = make_fragment::<F16>((2, 4));
                        copy(&tiled, &tiled.get_slice(t).partition_s(&halves), &tiled.retile(&frag));
                    }
                }
            });
        })
    };
    for (which, want) in [
        (0, "a load of 16 bytes in `smem1`, which is only declared 4-byte aligned"),
        (1, "a store of 8 bytes in `smem1`, which is only declared 4-byte aligned"),
        (2, "a load of 8 bytes in `smem1`, which is only declared 4-byte aligned"),
        (3, "a store of 16 bytes in `smem1`, which is only declared 4-byte aligned"),
        (4, "cp.async of 16 bytes in `smem1`, which is only declared 4-byte aligned"),
        (5, "an atomic of 8 bytes in `smem1`, which is only declared 4-byte aligned"),
        (6, "ldmatrix of 16 bytes in `smem2`, which is only declared 8-byte aligned"),
    ] {
        let msg = build(which);
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(
            msg.contains("PTX requires each access to be aligned to its size (allocate it with alloc_smem(count, "),
            "{msg}"
        );
        assert!(msg.contains("tests/diagnostics/memory_model.rs:"), "the error should point at the access: {msg}");
    }
    // Declared aligned to the vector, the load is fine.
    let kernel = Kernel::builder("aligned_ok").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let first = alloc_smem::<u32>(1, 4);
        let words = alloc_smem::<u32>(4, 16);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            first.store(1u32);
            store_vector(&words, [Uint32::from(2u32); 4], StoreOptions::default());
        });
        sync_threads();
        let v = load_vector::<u32, 4>(&words, LoadOptions { sem: MemoryOrder::Volatile, ..Default::default() });
        out.store(t, v[3] + first.load());
    });
    let mut args = vec![KernelArg::buffer(&[0u32; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].to_vec::<u32>()[5], 3);
}

/// A vector load or store moves at least one element.
#[test]
fn vectors_of_no_elements_are_rejected() {
    let msg = build_panic(|| {
        Kernel::build("empty_load", || {
            let g = param_tensor::<u32>("g", Layout::col_major(4));
            let [] = load_vector::<u32, 0>(&g.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("cute: load_vector of 0 elements: a load or store moves at least one element"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("empty_store", || {
            let g = param_tensor::<u32>("g", Layout::col_major(4));
            store_vector::<u32, 0>(&g.ptr(), [], StoreOptions::default());
        });
    });
    assert!(msg.contains("cute: store_vector of 0 elements: a load or store moves at least one element"), "{msg}");
}

/// A spin lock in shared memory around a plain counter, taken by lane 0 of
/// each of four warps with `atomic_cas` and given back with a release
/// `atomic_exch`, the critical section inside the loop that tries for it.
fn spin_lock(take: MemoryOrder) -> Kernel {
    Kernel::builder("spin_lock").block_dim([128, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(1));
        let lock = alloc_smem::<u32>(1, 4);
        let count = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            lock.store(0u32);
            count.store(0u32);
        });
        sync_threads();
        let take = AtomicOptions::new(take, MemoryScope::Cta);
        let give = AtomicOptions::new(MemoryOrder::Release, MemoryScope::Cta);
        if_((t % 32).eq(0), || {
            while_carry(
                Boolean::from(false),
                |done| !done,
                |done| {
                    let got = atomic_cas_with(&lock, 0u32, 1u32, take).eq(0u32);
                    if_(got, || {
                        count.store(count.load() + 1u32);
                        let _ = atomic_exch_with(&lock, 0u32, give);
                    });
                    done | got
                },
            );
        });
        sync_threads();
        if_(t.eq(0), || out.store(0, count.load()));
    })
}

/// The lock orders the counter's accesses only if taking it acquires (an
/// acquire or acq_rel `atomic_cas` reading the release `atomic_exch`).
#[test]
fn spin_lock_needs_an_acquire() {
    for take in [MemoryOrder::Acquire, MemoryOrder::AcqRel] {
        let mut args = vec![KernelArg::buffer(&[0u32])];
        spin_lock(take).simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap();
        assert_eq!(args[0].to_vec::<u32>(), [4], "{take:?}");
    }
    for take in [MemoryOrder::Relaxed, MemoryOrder::Release] {
        let e = simulate_err(&spin_lock(take), 128, &mut [KernelArg::buffer(&[0u32])]);
        assert!(e.message.contains("data race: warp 1 reads smem1[0], written by warp 0"), "{take:?}: {e}");
        assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
    }
}

/// A release atomic heads a release sequence that later atomics on the
/// location continue, whatever their own semantics: warp 0 writes data and
/// adds to a flag with release semantics, warp 2 adds to it relaxed, and
/// warp 1, spinning with an acquire atomic until it sees both, reads the
/// data. Without the release (or the acquire) the read races.
#[test]
fn release_atomics_head_release_sequences() {
    let kernel = |publish: MemoryOrder, observe: MemoryOrder| {
        Kernel::builder("release_sequence").block_dim([96, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(96));
            let flag = alloc_smem::<u32>(1, 4);
            let data = make_smem_tensor::<u32>(Layout::col_major(32));
            let t = thread_idx().x;
            let lane = t % 32;
            if_(t.eq(0), || flag.store(0u32));
            sync_threads();
            if_(warp_idx().eq(0), || {
                data.store(lane, lane.to::<u32>() + 1u32);
                if_(lane.eq(0), || {
                    let _ = atomic_add_with(&flag, 1u32, AtomicOptions::new(publish, MemoryScope::Cta));
                });
            });
            if_(warp_idx().eq(2) & lane.eq(0), || {
                red(&flag, 1u32, RedOp::Add, AtomicType::U32, AtomicOptions::default());
            });
            if_(warp_idx().eq(1), || {
                let observe = AtomicOptions::new(observe, MemoryScope::Cta);
                while_carry((), |()| atomic_or_with(&flag, 0u32, observe).ne(2u32), |()| ());
                out.store(t, data.load(lane));
            });
        })
    };
    let launch = LaunchConfig::new([1, 1, 1], [96, 1, 1]);
    for (publish, observe) in [(MemoryOrder::Release, MemoryOrder::Acquire), (MemoryOrder::AcqRel, MemoryOrder::AcqRel)]
    {
        let mut args = vec![KernelArg::buffer(&[0u32; 96])];
        kernel(publish, observe).simulate(launch, &mut args).unwrap_or_else(|e| panic!("{publish:?}: {e}"));
        assert_eq!(args[0].to_vec::<u32>()[32..35], [1, 2, 3]);
    }
    for (publish, observe) in
        [(MemoryOrder::Relaxed, MemoryOrder::Acquire), (MemoryOrder::Release, MemoryOrder::Relaxed)]
    {
        let e = simulate_err(&kernel(publish, observe), 96, &mut [KernelArg::buffer(&[0u32; 96])]);
        assert!(e.message.contains("data race: warp 1 reads"), "{publish:?}, {observe:?}: {e}");
    }
}

/// A reduction is an atomic: reductions and atomics of one location never
/// race with each other, a plain store does.
#[test]
fn reductions_race_only_with_plain_accesses() {
    let kernel = |plain: bool| {
        Kernel::builder("red_race").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(1));
            let cell = alloc_smem::<u32>(1, 4);
            let t = thread_idx().x;
            if_(t.eq(0), || cell.store(0u32));
            sync_threads();
            if_(warp_idx().eq(0), || red(&cell, 1u32, RedOp::Add, AtomicType::U32, AtomicOptions::default()));
            if_(warp_idx().eq(1) & (t % 32).eq(0), || {
                if plain {
                    cell.store(5u32);
                } else {
                    let _ = atomic_add(&cell, 100u32);
                }
            });
            sync_threads();
            if_(t.eq(0), || out.store(0, cell.load()));
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32])];
    kernel(false).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<u32>(), [132]);
    let e = simulate_err(&kernel(true), 64, &mut [KernelArg::buffer(&[0u32])]);
    assert!(e.message.contains("data race: warps 0 and 1 both write smem0[0]"), "{e}");
}

/// Misused atomics and reductions: the semantics `red` lacks, operations
/// PTX lacks on a type, cluster scope and BFloat16 below sm_90, and dtypes
/// that do not fit the values. `Umin` on a signed dtype is CuTe DSL's
/// unsigned min.
#[test]
fn atomic_misuse() {
    use cute::validate::check_target;
    let red_kernel = |f: fn(&Ptr<u32>, &Ptr<f32>)| {
        Kernel::builder("red_misuse").block_dim([32, 1, 1]).build(move || {
            let g = param_tensor::<u32>("g", Layout::col_major(4));
            let h = param_tensor::<f32>("f", Layout::col_major(4));
            f(&g.ptr(), &h.ptr());
        })
    };
    for (f, want) in [
        (
            (|g: &Ptr<u32>, _: &Ptr<f32>| {
                red(g, 1u32, RedOp::Add, AtomicType::U32, AtomicOptions::new(MemoryOrder::Acquire, MemoryScope::Gpu))
            }) as fn(&Ptr<u32>, &Ptr<f32>),
            "a reduction (red) cannot be acquire: its sem is relaxed, release",
        ),
        (
            |_, f| red(f, 1.0f32, RedOp::Max, AtomicType::F32, AtomicOptions::default()),
            "PTX has no red.max.f32 on a scalar float type",
        ),
        (|_, f| red(f, 1.0f32, RedOp::Umin, AtomicType::F32, AtomicOptions::default()), "PTX has no red.min.f32"),
        (|g, _| red(g, 1u32, RedOp::Min, AtomicType::B32, AtomicOptions::default()), "PTX has no red.min.b32"),
        (|g, _| red(g, 1u32, RedOp::And, AtomicType::U32, AtomicOptions::default()), "PTX has no red.and.u32"),
        (
            |g, _| {
                let _ = atomic_add_with(g, 1u32, AtomicOptions { sem: Some(MemoryOrder::Sc), scope: None });
            },
            "an atomic cannot be sc: its sem is relaxed, acquire, release, acq_rel",
        ),
    ] {
        let msg = build_panic(|| {
            red_kernel(f);
        });
        assert!(msg.contains(want), "{want}: {msg}");
    }
    // A dtype must fit the values.
    let msg = build_panic(|| {
        red_kernel(|g, _| red(g, 1u32, RedOp::Add, AtomicType::F32, AtomicOptions::default()));
    });
    assert!(msg.contains("a float dtype must be the values' own type"), "{msg}");
    let msg = build_panic(|| {
        red_kernel(|g, _| red(g, 1u32, RedOp::Add, AtomicType::U64, AtomicOptions::default()));
    });
    assert!(msg.contains("the dtype must be as wide as the values"), "{msg}");
    // Cluster scope and BFloat16 need sm_90.
    let cluster = red_kernel(|g, _| {
        let _ = atomic_cas_with(g, 0u32, 1u32, AtomicOptions::new(MemoryOrder::AcqRel, MemoryScope::Cluster));
    });
    let e = check_target(cluster.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("an atomic or reduction with cluster scope needs sm_90 or newer"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics/memory_model.rs"), "{e}");
    assert_eq!(check_target(cluster.ir(), Arch::Sm90).unwrap(), 78);
    let bf16 = Kernel::builder("bf16_atomic").block_dim([32, 1, 1]).build(|| {
        let h = param_tensor::<BF16>("h", Layout::col_major(2));
        let _ = atomic_add(&h.ptr(), BF16::from_f32(1.0));
    });
    let e = check_target(bf16.ir(), Arch::Sm80).unwrap_err();
    assert!(e.message.contains("a BFloat16 atomic add"), "{e}");
    // umin on s32 compares unsigned: -1 is the largest value.
    let umin = Kernel::builder("umin").block_dim([32, 1, 1]).build(|| {
        let g = param_tensor::<i32>("g", Layout::col_major(1));
        let t = thread_idx().x;
        red(&g.ptr(), select(t.eq(0), -1, 7), RedOp::Umin, AtomicType::S32, AtomicOptions::default());
    });
    let mut args = vec![KernelArg::buffer(&[-2i32])];
    umin.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), [7]);
}

/// An atomic followed by a fence acquires what the atomic read (PTX's
/// `atom.relaxed [M]; fence.acquire` pattern); a reduction followed by a
/// fence does not (`red [M]; fence.acquire` is no acquire pattern), so the
/// read of the data that warp 0 released races.
#[test]
fn reductions_do_not_acquire() {
    let kernel = |use_red: bool| {
        Kernel::builder("red_acquire").block_dim([64, 1, 1]).build(move || {
            let out = param_tensor::<u32>("out", Layout::col_major(64));
            let flag = alloc_smem::<u32>(1, 4);
            let data = make_smem_tensor::<u32>(Layout::col_major(32));
            let t = thread_idx().x;
            let lane = t % 32;
            let cta = |sem| AtomicOptions::new(sem, MemoryScope::Cta);
            let bar = alloc_smem::<u64>(1, 8);
            if_(t.eq(0), || {
                flag.store(0u32);
                mbarrier_init(&bar, 32);
            });
            sync_threads();
            if_(warp_idx().eq(0), || {
                data.store(lane, lane.to::<u32>() + 1u32);
                if_(lane.eq(0), || {
                    let _ = atomic_exch_with(&flag, 1u32, cta(MemoryOrder::Release));
                });
                mbarrier_arrive_with(&bar, 1, MbarrierOptions { relaxed: true, ..Default::default() });
            });
            if_(warp_idx().eq(1), || {
                // Wait for warp 0's release through an mbarrier whose relaxed
                // arrival orders nothing.
                while_carry((), |()| !mbarrier_test_wait(&bar, 0), |()| ());
                if use_red {
                    red(&flag, 0u32, RedOp::Add, AtomicType::U32, cta(MemoryOrder::Relaxed));
                } else {
                    let _ = atomic_add_with(&flag, 0u32, cta(MemoryOrder::Relaxed));
                }
                fence_acq_rel_cta();
                out.store(t, data.load(lane));
            });
        })
    };
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    kernel(false).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<u32>()[32..35], [1, 2, 3]);
    let e = simulate_err(&kernel(true), 64, &mut [KernelArg::buffer(&[0u32; 64])]);
    assert!(e.message.contains("data race: warp 1 reads"), "{e}");
}

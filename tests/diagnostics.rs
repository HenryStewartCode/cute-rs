//! Kernels with deliberate bugs: the simulator and the validator must catch
//! each one and point at the offending line.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

fn simulate_err(kernel: &Kernel, block: u32, args: &mut [KernelArg]) -> SimError {
    match kernel.simulate(LaunchConfig::new([1, 1, 1], [block, 1, 1]), args) {
        Ok(_) => panic!("expected `{}` to fail in the simulator", kernel.name()),
        Err(e) => e,
    }
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Stage 64 floats through shared memory with cp.async, then read them back.
fn staged_copy(wait: bool) -> Kernel {
    Kernel::builder("staged_copy").block_dim([32, 1, 1]).build(|| {
        let src = param_tensor::<f32>("src", Layout::col_major(128));
        let dst = param_tensor::<f32>("dst", Layout::col_major(128));
        let smem = make_smem_tensor::<f32>(Layout::col_major(128));
        let t = thread_idx().x;
        let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let thr = tiled.get_slice(t);
        copy(&tiled, &thr.partition_s(&src), &thr.partition_d(&smem));
        cp_async_commit_group();
        if wait {
            cp_async_wait_group(0);
        }
        sync_threads();
        let i = (t + 1) % 32 * 4;
        dst.store(i, smem.load(i));
    })
}

#[test]
fn correct_cp_async_staging_passes() {
    let mut args = vec![KernelArg::f32((0..128).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; 128])];
    staged_copy(true).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].as_f32()[4], 4.0);
}

#[test]
fn missing_cp_async_wait() {
    let mut args = vec![KernelArg::f32(vec![1.0; 128]), KernelArg::f32(vec![0.0; 128])];
    let e = simulate_err(&staged_copy(false), 32, &mut args);
    assert!(e.message.contains("cp.async is still writing"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics.rs"), "error should point at the kernel source: {e}");
}

/// Warp 0 writes shared memory and warp 1 reads it.
fn producer_consumer(barrier: bool) -> Kernel {
    Kernel::builder("producer_consumer").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let smem = make_smem_tensor::<f32>(Layout::col_major(32));
        let t = thread_idx().x;
        if_(t.lt(32), || smem.store(t, t.to::<f32>()));
        if barrier {
            sync_threads();
        }
        if_(t.ge(32), || out.store(t, smem.load(t - 32)));
    })
}

#[test]
fn barrier_orders_warps() {
    let mut args = vec![KernelArg::f32(vec![0.0; 64])];
    producer_consumer(true).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].as_f32()[40], 8.0);
}

#[test]
fn read_after_write_race() {
    let mut args = vec![KernelArg::f32(vec![0.0; 64])];
    let e = simulate_err(&producer_consumer(false), 64, &mut args);
    assert!(e.message.contains("data race") && e.message.contains("missing sync_threads"), "{e}");
    assert_eq!(e.thread, Some([32, 0, 0]));
}

#[test]
fn write_after_read_race() {
    let kernel = Kernel::builder("war").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let smem = make_smem_tensor::<f32>(Layout::col_major(64));
        let t = thread_idx().x;
        smem.store(t, 1.0f32);
        sync_threads();
        let v = smem.load((t + 32) % 64);
        // Missing barrier: warp 1 overwrites what warp 0 just read.
        smem.store(t, v + 1.0);
        out.store(t, v);
    });
    let e = simulate_err(&kernel, 64, &mut [KernelArg::f32(vec![0.0; 64])]);
    assert!(e.message.contains("overwrites"), "{e}");
}

#[test]
fn uninitialised_shared_memory() {
    let kernel = Kernel::builder("uninit").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let smem = make_smem_tensor::<f32>(Layout::col_major(32));
        out.store(thread_idx().x, smem.load(thread_idx().x));
    });
    let e = simulate_err(&kernel, 32, &mut [KernelArg::f32(vec![0.0; 32])]);
    assert!(e.message.contains("never written"), "{e}");
}

#[test]
fn out_of_bounds_global_access() {
    let kernel = Kernel::build("oob", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t + 1, 0.0f32);
    });
    let e = simulate_err(&kernel, 32, &mut [KernelArg::f32(vec![0.0; 32])]);
    assert!(e.message.contains("out of bounds"), "{e}");
    assert_eq!(e.thread, Some([31, 0, 0]));
}

#[test]
fn barrier_in_divergent_code() {
    let kernel = Kernel::build("divergent_barrier", || {
        let t = thread_idx().x;
        if_(t.lt(16), sync_threads);
    });
    let e = simulate_err(&kernel, 32, &mut []);
    assert!(e.message.contains("divergent"), "{e}");
}

#[test]
fn integer_division_by_zero() {
    let kernel = Kernel::build("div0", || {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, 100 / (t - 3));
    });
    let e = simulate_err(&kernel, 32, &mut [KernelArg::i32(vec![0; 32])]);
    assert!(e.message.contains("division by zero"), "{e}");
    assert_eq!(e.thread, Some([3, 0, 0]));
}

#[test]
fn wrong_launch_configuration() {
    let kernel = Kernel::builder("fixed_block").block_dim([128, 1, 1]).build(|| {});
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut []).unwrap_err();
    assert!(e.message.contains("built for blocks of"), "{e}");
}

#[test]
fn wrong_arguments() {
    let kernel = Kernel::build("args", || {
        param_tensor::<f32>("x", Layout::col_major(64));
    });
    let e =
        kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut [KernelArg::f32(vec![0.0; 10])]).unwrap_err();
    assert!(e.message.contains("needs at least 64 elements"), "{e}");
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut [KernelArg::i32(vec![0; 64])]).unwrap_err();
    assert!(e.message.contains("must be a f32 buffer"), "{e}");
}

#[test]
fn value_escaping_its_region_is_rejected() {
    let msg = build_panic(|| {
        Kernel::build("escape", || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            let mut leaked = None;
            if_(t.lt(4), || leaked = Some(t * 2));
            out.store(t, leaked.unwrap());
        });
    });
    assert!(msg.contains("outside the region that defines it"), "{msg}");
}

#[test]
fn unprovable_vector_alignment_is_rejected() {
    let msg = build_panic(|| {
        Kernel::build("misaligned", || {
            let x = param_tensor::<f32>("x", Layout::col_major(128));
            let frag = make_fragment::<f32>(4);
            let t = thread_idx().x;
            // Overlapping windows: thread t reads elements t..t+4, which is
            // only 16-byte aligned for every fourth thread.
            let view = x.compose(Layout::new((4, 32), (1, 1))).slice((.., t));
            copy(&make_copy_atom::<f32>(CopyOp::Universal { bits: 128 }), &view, &frag);
        });
    });
    assert!(msg.contains("cannot prove") && msg.contains("aligned"), "{msg}");
}

#[test]
fn non_contiguous_vector_copy_is_rejected() {
    let msg = build_panic(|| {
        Kernel::build("strided", || {
            let x = param_tensor::<f32>("x", Layout::new(4, 2));
            let frag = make_fragment::<f32>(4);
            copy(&make_copy_atom::<f32>(CopyOp::Universal { bits: 128 }), &x, &frag);
        });
    });
    assert!(msg.contains("contiguous"), "{msg}");
}

#[test]
fn register_tensors_need_static_indices() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_register_index", || {
            let frag = make_fragment::<f32>(8);
            frag.fill(0.0);
            let _ = frag.load(thread_idx().x % 8);
        });
    });
    assert!(msg.contains("static coordinates"), "{msg}");
}

#[test]
fn too_much_shared_memory() {
    let msg = build_panic(|| {
        Kernel::build("big_smem", || {
            let s = make_smem_tensor::<f32>(Layout::col_major(128 * 1024));
            s.store(0, 0.0f32);
        });
    });
    assert!(msg.contains("bytes of shared memory"), "{msg}");
}

#[test]
fn races_can_be_ignored() {
    let opts = SimOptions { detect_races: false, ..SimOptions::default() };
    let mut args = vec![KernelArg::f32(vec![0.0; 64])];
    producer_consumer(false).simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts).unwrap();
}

/// CuTe DSL's warp MMA constructors pick the `mma.sync` kind and reject the
/// combinations PTX does not have.
#[test]
fn warp_mma_constructors() {
    assert_eq!(MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 16)).kind(), MmaKind::M16N8K16F32BF16);
    assert_eq!(MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 8)).kind(), MmaKind::M16N8K8F32BF16);
    assert_eq!(MmaF16BF16Op::new(DType::F16, DType::F16, (16, 8, 8)).kind(), MmaKind::M16N8K8F16F16);
    assert_eq!(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)).kind(), MmaKind::M16N8K16F32F16);
    assert_eq!(MmaTF32Op::new((16, 8, 8)).kind(), MmaKind::M16N8K8Tf32);
    assert_eq!(MmaTF32Op::new((16, 8, 4)).kind(), MmaKind::M16N8K4Tf32);
    assert_eq!(MmaOp::from(MmaTF32Op::new((16, 8, 4))), MmaOp::Sm80(MmaKind::M16N8K4Tf32));

    let msg = build_panic(|| {
        MmaF16BF16Op::new(DType::BF16, DType::F16, (16, 8, 16));
    });
    assert!(msg.contains("'acc_dtype'") && msg.contains("Float32 when 'ab_dtype' is BFloat16"), "{msg}");
    let msg = build_panic(|| {
        MmaF16BF16Op::new(DType::F32, DType::F32, (16, 8, 16));
    });
    assert!(msg.contains("'ab_dtype'"), "{msg}");
    let msg = build_panic(|| {
        MmaF16BF16Op::new(DType::F16, DType::F64, (16, 8, 16));
    });
    assert!(msg.contains("'acc_dtype'"), "{msg}");
    let msg = build_panic(|| {
        MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 32));
    });
    assert!(msg.contains("'shape_mnk'"), "{msg}");
    let msg = build_panic(|| {
        MmaTF32Op::new((16, 8, 16));
    });
    assert!(msg.contains("(16,8,4) or (16,8,8)"), "{msg}");
}

/// Fragments must have the atom's element types.
#[test]
fn bf16_fragments_hold_bf16() {
    let msg = build_panic(|| {
        Kernel::builder("wrong_fragment").block_dim([32, 1, 1]).build(|| {
            let a = param_tensor::<F16>("a", Layout::col_major((16, 16)));
            let mma = make_tiled_mma(MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 16)), Layout::col_major(1));
            let _ = mma.get_slice(lane_idx()).partition_fragment_a::<F16, _>(&a);
        });
    });
    assert!(msg.contains("hold bf16, not f16"), "{msg}");
}

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
fn named_barrier_that_cannot_complete_in_lockstep() {
    // The consumers wait first; the producers' arrivals come later in program order.
    let kernel = Kernel::builder("late_arrival").block_dim([256, 1, 1]).build(|| {
        let w = warp_idx();
        if_else(w.ge(4), || barrier(1, 256), || barrier_arrive(1, 256));
    });
    let e = simulate_err(&kernel, 256, &mut []);
    assert!(e.message.contains("barrier 1 needs 8 warps") && e.message.contains("lockstep"), "{e}");
    assert!(e.message.contains("warps 4..=7"), "{e}");
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
    assert!(msg.contains("NamedBarrier ID must be in [0,15]"), "{msg}");
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
/// simulator's lockstep execution never shows that, so it must reject a
/// wait whose warp has not seen the previous phase complete.
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
        if_(thread_idx().x.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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

/// A kernel applying `f` to a Float32 (and a second Float32) parameter.
fn math_kernel(f: impl Fn(Float32, Float32) -> Float32) -> Kernel {
    Kernel::builder("math_misuse").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, f(x.load(t), x.load(t)));
    })
}

/// CuTe DSL's rules for `cute.math`'s keyword options: fastmath excludes the
/// options that choose an instruction, ftz on a transcendental needs approx,
/// each function takes only its own options, and approx/full/rounding are
/// alternatives.
#[test]
fn math_options_follow_cute_dsl_rules() {
    let fast_approx = MathOptions { fastmath: true, approx: true, ..MathOptions::default() };
    let msg = build_panic(|| {
        math_kernel(|x, _| math::exp2_with(x, fast_approx));
    });
    assert!(msg.contains("fastmath and approx are mutually exclusive"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| {
            math::add_with(x, y, MathOptions { fastmath: true, ..MathOptions::rounding(RoundingMode::Zero) })
        });
    });
    assert!(msg.contains("fastmath and rounding are mutually exclusive"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, _| math::sin_with(x, MathOptions { ftz: true, ..MathOptions::default() }));
    });
    assert!(msg.contains("ftz needs approx"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, _| math::tanh_with(x, MathOptions::APPROX.with_ftz()));
    });
    assert!(msg.contains("cute::math::tanh takes no ftz option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| math::add_with(x, y, MathOptions::APPROX));
    });
    assert!(msg.contains("cute::math::add takes no approx option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| math::min_with(x, y, MathOptions::FASTMATH));
    });
    assert!(msg.contains("cute::math::min takes no fastmath option"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| {
            math::div_with(x, y, MathOptions { approx: true, ..MathOptions::rounding(RoundingMode::Up) })
        });
    });
    assert!(msg.contains("approx, full and rounding are alternatives"), "{msg}");
}

/// The options that choose one PTX instruction need Float32 or Float64, and
/// the instruction must exist: the validator names it and the user's line.
#[test]
fn math_instruction_options_must_exist_in_ptx() {
    let msg = build_panic(|| {
        Kernel::builder("half_approx").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<F16>("x", Layout::col_major(32));
            let t = thread_idx().x;
            x.store(t, math::sqrt_with(x.load(t), MathOptions::APPROX));
        });
    });
    assert!(msg.contains("needs Float32 or Float64 operands, not f16"), "{msg}");
    let double = |f: fn(Float64) -> Float64| {
        build_panic(move || {
            Kernel::builder("double_approx").block_dim([32, 1, 1]).build(|| {
                let x = param_tensor::<f64>("x", Layout::col_major(32));
                let t = thread_idx().x;
                x.store(t, f(x.load(t)));
            });
        })
    };
    let msg = double(|x| math::sin_with(x, MathOptions::APPROX));
    assert!(msg.contains("sin.approx.f64 is not a PTX instruction") && msg.contains("Float32 only"), "{msg}");
    assert!(msg.contains("tests/diagnostics.rs"), "the error should point at the call: {msg}");
    let msg = double(|x| math::rcp_with(x, MathOptions::APPROX));
    assert!(msg.contains("rcp.approx.f64 is not a PTX instruction") && msg.contains("rcp.approx.ftz.f64"), "{msg}");
    // ftz on Float64 exists only in the two approximations: not with a
    // rounding mode, explicit or implied.
    let msg = double(|x| math::rcp_with(x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("rcp.rn.ftz.f64 is not a PTX instruction: ftz applies to Float32 only"), "{msg}");
    let msg = double(|x| math::rcp_with(x, MathOptions::rounding(RoundingMode::Zero).with_ftz()));
    assert!(msg.contains("rcp.rz.ftz.f64 is not a PTX instruction"), "{msg}");
    let msg = double(|x| math::sqrt_with(x, MathOptions::rounding(RoundingMode::Up).with_ftz()));
    assert!(msg.contains("sqrt.rp.ftz.f64 is not a PTX instruction"), "{msg}");
    let msg = double(|x| math::min_with(x, x, MathOptions::rounding(RoundingMode::NearestEven).with_ftz()));
    assert!(msg.contains("cute::math::min takes no rounding option"), "{msg}");
    let msg = double(|x| math::min_with(x, x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("min.ftz.f64 is not a PTX instruction: ftz applies to Float32 only"), "{msg}");
    let msg = double(|x| math::add_with(x, x, MathOptions { ftz: true, ..MathOptions::default() }));
    assert!(msg.contains("add.rn.ftz.f64 is not a PTX instruction"), "{msg}");
    // The plain approximations of `Value` are Float32 only too.
    let msg = double(|x| x.exp2());
    assert!(msg.contains("ex2.approx") && msg.contains("cute::math::exp2 computes Float64"), "{msg}");
}

/// Errors inside a math expansion point at the user's call, not at
/// `src/math`.
#[test]
fn math_expansion_errors_point_at_the_caller() {
    let kernel = Kernel::builder("math_oob").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t + 1, math::exp(x.load(t), false));
    });
    let mut args = vec![KernelArg::f32(vec![0.5; 32])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
    let dump = kernel.dump();
    assert!(!dump.is_empty());
    // Every instruction of the expansion is attributed to this file.
    let ir = kernel.ir();
    let mut files = std::collections::BTreeSet::new();
    fn walk(r: &cute::ir::Region, files: &mut std::collections::BTreeSet<String>) {
        for inst in &r.insts {
            files.insert(inst.loc.file().to_string());
            for region in inst.op.regions() {
                walk(region, files);
            }
        }
    }
    walk(&ir.body, &mut files);
    assert!(files.iter().all(|f| !f.contains("src/math")), "{files:?}");
}

/// Setting a Var to a value of other types names both the line that created
/// it and the line that set it.
#[test]
fn var_type_change_names_both_lines() {
    let (mut created, mut set) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("var_type_change", || {
            let t = thread_idx().x;
            let lanes = Var::new(vec![t, t + 1]);
            created = line!() - 1;
            for_range(0, t, 1, |i| {
                set = line!() + 1;
                lanes.set(vec![i, i, i]);
            });
        });
    });
    assert!(msg.contains("holds (i32, i32), but is set to (i32, i32, i32)"), "{msg}");
    assert!(msg.contains("a Var's type cannot change"), "{msg}");
    for line in [created, set] {
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "line {line}: {msg}");
    }
    // A register tensor of another size is another type too.
    let msg = build_panic(|| {
        Kernel::build("var_tensor_change", || {
            let frag = make_fragment::<f32>(4);
            frag.fill(0.0);
            let acc = Var::new(frag);
            let bigger = make_fragment::<f32>(8);
            bigger.fill(1.0);
            acc.set(bigger);
        });
    });
    assert!(
        msg.contains("holds (f32, f32, f32, f32), but is set to (f32, f32, f32, f32, f32, f32, f32, f32)"),
        "{msg}"
    );
}

/// What a loop reads from a Var belongs to the loop body, like any value
/// computed there: using it after the loop is the escape error. The Var
/// itself holds the loop's result.
#[test]
fn value_read_from_a_var_in_a_loop_cannot_escape() {
    let msg = build_panic(|| {
        Kernel::builder("var_escape").block_dim([32, 1, 1]).build(|| {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            let sum = Var::new(Int32::from(0));
            let mut seen = None;
            for_range(0, t, 1, |i| {
                seen = Some(sum.get());
                sum.set(sum.get() + i);
            });
            out.store(t, seen.unwrap() + sum.get());
        });
    });
    assert!(
        msg.contains("outside the region that defines it") && msg.contains("including values read from a Var"),
        "{msg}"
    );
    assert!(msg.contains("tests/diagnostics.rs"), "the error should point at the kernel source: {msg}");
}

/// A Var is local to the region that created it.
#[test]
fn var_used_outside_its_region() {
    let msg = build_panic(|| {
        Kernel::build("var_after_branch", || {
            let t = thread_idx().x;
            let mut inner = None;
            if_(t.lt(4), || inner = Some(Var::new(t)));
            let _ = inner.unwrap().get();
        });
    });
    assert!(msg.contains("outside the region it was created in"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("var_in_other_branch", || {
            let t = thread_idx().x;
            let inner = std::cell::RefCell::new(None);
            if_else(t.lt(4), || *inner.borrow_mut() = Some(Var::new(t)), || inner.borrow().as_ref().unwrap().set(t));
        });
    });
    assert!(msg.contains("outside the region it was created in"), "{msg}");
    let mut stash = None;
    Kernel::build("first", || stash = Some(Var::new(thread_idx().x)));
    let msg = build_panic(|| {
        Kernel::build("second", || {
            let _ = stash.as_ref().unwrap().get();
        });
    });
    assert!(msg.contains("a Var from another kernel"), "{msg}");
}

/// A while condition reads Vars; it cannot set the ones created outside it.
#[test]
fn var_set_in_a_while_condition() {
    let msg = build_panic(|| {
        Kernel::build("var_in_condition", || {
            let n = Var::new(thread_idx().x);
            while_carry(
                (),
                |()| {
                    n.set(n.get() - 1);
                    n.get().gt(0)
                },
                |()| (),
            );
        });
    });
    assert!(msg.contains("inside a while_carry condition") && msg.contains("set it in the loop body"), "{msg}");
}

/// A loop marked at_least_once that runs zero times breaks the promise the
/// compiler may rely on: the simulator names the thread, the validator
/// catches constant bounds.
#[test]
fn at_least_once_loop_that_runs_zero_times() {
    let once = LoopOptions { at_least_once: true, ..LoopOptions::default() };
    let kernel = Kernel::builder("at_least_once").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        let acc = Var::new(Int32::from(0));
        for_range_with(0, t % 8, 1, once, |i| acc.set(acc.get() + i));
        out.store(t, acc.get());
    });
    let e = simulate_err(&kernel, 32, &mut [KernelArg::i32(vec![0; 32])]);
    assert!(e.message.contains("marked at_least_once, but runs zero times") && e.message.contains("(0..0)"), "{e}");
    assert_eq!(e.thread, Some([0, 0, 0]));
    assert!(e.to_string().contains("tests/diagnostics.rs"), "the error should point at the loop: {e}");
    let msg = build_panic(|| {
        Kernel::build("never", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(4, 4, 1, once, |i| x.store(i, i));
        });
    });
    assert!(msg.contains("marked at_least_once, but 4..4 runs zero times"), "{msg}");
}

/// Loop hints that contradict each other or do not fit are rejected.
#[test]
fn loop_hints_are_validated() {
    let both = LoopOptions { unroll: Some(2), unroll_full: true, ..LoopOptions::default() };
    let msg = build_panic(|| {
        Kernel::build("both_unrolls", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(0, 8, 1, both, |i| x.store(i, i));
        });
    });
    assert!(msg.contains("unroll (2) and unroll_full contradict each other"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("huge_unroll", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(0, 8, 1, LoopOptions::unroll(u32::MAX), |i| x.store(i, i));
        });
    });
    assert!(msg.contains("an unroll count of 4294967295 does not fit in i32"), "{msg}");
}

/// The vectorize hint needs sm_100 (CuTe DSL raises UNSUP_ARCH below it).
#[test]
fn vectorize_needs_sm100() {
    use cute::validate::check_target;
    let kernel = Kernel::build("vectorized", || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let opts = LoopOptions { vectorize: true, ..LoopOptions::default() };
        for_range_with(0, 64, 1, opts, |i| x.store(i, x.load(i) * 2.0));
    });
    for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90a] {
        let e = check_target(kernel.ir(), arch).unwrap_err();
        let want = format!("needs sm_100 or newer; the kernel is being compiled for {}", arch.chip());
        assert!(e.message.contains("a vectorized loop") && e.message.contains(&want), "{e}");
        assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    assert_eq!(check_target(kernel.ir(), Arch::Sm120).unwrap(), 87);
}

/// A register fragment with `values` in its registers, laid out by `layout`.
fn filled_fragment(layout: Layout, values: impl Fn(usize) -> i32) -> Tensor<i32> {
    let frag = make_fragment_with_layout::<i32>(layout);
    for k in 0..frag.size() {
        frag.store(k, values(k));
    }
    frag
}

/// A register tensor's layout is part of a Var's type: setting the Var to a
/// tensor with another layout names both lines, at the top level and in a
/// loop (where `get` would otherwise read the registers in the old order).
#[test]
fn var_structure_change_names_both_lines() {
    let (mut created, mut set) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("var_layout_change", || {
            let v = Var::new(filled_fragment(Layout::col_major((2, 2)), |k| k as i32));
            created = line!() - 1;
            set = line!() + 1;
            v.set(filled_fragment(Layout::row_major((2, 2)), |k| 10 + k as i32));
        });
    });
    assert!(
        msg.contains("is set to a value of another structure")
            && msg.contains("a register tensor with layout (2,2):(1,2) became one with layout (2,2):(2,1)")
            && msg.contains("a Var's type cannot change"),
        "{msg}"
    );
    for line in [created, set] {
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "line {line}: {msg}");
    }
    // Inside a loop, and nested in a struct field and a tuple element.
    let msg = build_panic(|| {
        Kernel::build("var_field_layout_change", || {
            let v = Var::new((Int32::from(0), vec![filled_fragment(Layout::col_major(4), |k| k as i32)]));
            for_range(0, thread_idx().x, 1, |i| {
                let nested = make_fragment_with_layout::<i32>(Layout::col_major((2, 2)));
                nested.fill(i);
                v.set((i, vec![nested]));
            });
        });
    });
    assert!(
        msg.contains("element 1: element 0: a register tensor with layout 4:1 became one with layout (2,2):(1,2)"),
        "{msg}"
    );
    // Vectors of vectors with the same flattened types but another nesting.
    let msg = build_panic(|| {
        Kernel::build("var_nesting_change", || {
            let t = thread_idx().x;
            let v = Var::new(vec![vec![t, t], vec![t]]);
            v.set(vec![vec![t], vec![t, t]]);
        });
    });
    assert!(msg.contains("element 0: a vector of 2 elements became one of 1"), "{msg}");
}

/// What a loop body or a branch produces must keep the structure of what
/// came in, as CuTe DSL requires of the values its regions carry.
#[test]
fn region_structure_change_is_reported() {
    let msg = build_panic(|| {
        Kernel::build("loop_layout_change", || {
            let init = filled_fragment(Layout::col_major((2, 2)), |k| k as i32);
            for_range_carry(0, thread_idx().x, 1, init, |_, _| {
                filled_fragment(Layout::row_major((2, 2)), |k| k as i32)
            });
        });
    });
    assert!(
        msg.contains("for_range_carry at tests/diagnostics.rs")
            && msg.contains("the body returns state of another structure than `init`")
            && msg.contains("layout (2,2):(1,2) became one with layout (2,2):(2,1)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("branch_layout_change", || {
            let t = thread_idx().x;
            if_else_yield(
                t.lt(4),
                || filled_fragment(Layout::col_major((2, 2)), |k| k as i32),
                || filled_fragment(Layout::row_major((2, 2)), |k| k as i32),
            );
        });
    });
    assert!(msg.contains("if_else_yield at tests/diagnostics.rs") && msg.contains("different structures"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("while_layout_change", || {
            let init = filled_fragment(Layout::col_major((2, 2)), |k| k as i32);
            while_carry(init, |f| f.load(0usize).lt(10), |_| filled_fragment(Layout::row_major((2, 2)), |_| 0));
        });
    });
    assert!(msg.contains("while_carry at tests/diagnostics.rs"), "{msg}");
}

/// Writing a register of a fragment created outside a runtime loop after
/// the loop read it would not reach the next iteration (register fragments
/// are build-time arrays of SSA values, not memory as in CuTe DSL), so the
/// build fails, naming both lines.
#[test]
fn fragment_written_after_read_in_a_loop() {
    let (mut read, mut written) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("running_sum_in_registers", || {
            let out = param_tensor::<i32>("out", Layout::col_major(8));
            let f = make_fragment::<i32>(1);
            f.store(0usize, 0);
            for_range(0, 8, 1, |i| {
                read = line!() + 1;
                let sum = f.load(0usize) + i;
                written = line!() + 1;
                f.store(0usize, sum);
                out.store(i, f.load(0usize));
            });
        });
    });
    assert!(
        msg.contains("register 0 of a fragment created outside a runtime loop")
            && msg.contains("the loop's next iteration would not see the write")
            && msg.contains("Var::update"),
        "{msg}"
    );
    assert!(msg.contains(&format!("read at tests/diagnostics.rs:{read}:")), "{msg}");
    assert!(msg.contains(&format!("written at tests/diagnostics.rs:{written}:")), "{msg}");

    // Read in an outer loop and written in an inner one.
    let msg = build_panic(|| {
        Kernel::build("outer_read_inner_write", || {
            let f = make_fragment::<i32>(2);
            f.fill(1);
            for_range(0, 4, 1, |i| {
                let x = f.load(1usize);
                for_range(0, i, 1, |j| f.store(1usize, x + j));
            });
        });
    });
    assert!(msg.contains("register 1 of a fragment"), "{msg}");
    // Read by a while condition and written by the body.
    let msg = build_panic(|| {
        Kernel::build("condition_read_body_write", || {
            let f = make_fragment::<i32>(1);
            f.store(0usize, thread_idx().x);
            while_carry((), |()| f.load(0usize).gt(0), |()| f.store(0usize, thread_idx().x - 1));
        });
    });
    assert!(msg.contains("register 0 of a fragment"), "{msg}");
    // Read as an inner loop's initial state, then written: the next outer
    // iteration would start the inner loop from the old registers.
    let msg = build_panic(|| {
        Kernel::build("state_read_then_written", || {
            let f = make_fragment::<i32>(2);
            f.fill(0);
            for_range(0, 4, 1, |i| {
                let g = for_range_carry(0, i, 1, f.clone(), |j, g| {
                    g.store(0usize, g.load(0usize) + j);
                    g
                });
                f.store(1usize, g.load(0usize));
            });
        });
    });
    assert!(msg.contains("register 1 of a fragment"), "{msg}");
}

/// What a branch or loop result is known to be divisible by holds for every
/// side: with one side odd, a 128-bit copy at the result is refused.
#[test]
fn merged_facts_are_not_overstated() {
    let copy_at = |x: &Tensor<f32>, offset: Int32| {
        let frag = make_fragment::<f32>(4);
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy(&atom, &x.compose(Layout::new((4, 250), (1, 1))).slice((.., offset)), &frag);
    };
    let index = |t: Int32, k: usize, c: usize| (IndexExpr::from(t) * k + c).into_value();
    let msg = build_panic(|| {
        Kernel::builder("odd_branch").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let t = thread_idx().x;
            copy_at(&x, if_else_yield(t.lt(16), || index(t, 8, 0), || index(t, 4, 1)));
        });
    });
    assert!(msg.contains("cannot prove that the source offset"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("odd_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let t = thread_idx().x;
            copy_at(&x, for_range_carry(0, 2, 1, index(t, 8, 0), |_, _| index(t, 2, 0)));
        });
    });
    assert!(msg.contains("cannot prove that the source offset"), "{msg}");
}

/// Inside a runtime loop a Var is a loop-carried value, about which nothing
/// is known even when the loop only reads it: the errors this causes name
/// the Var and say to read it before the loop, which works.
#[test]
fn var_read_in_a_loop_loses_what_is_known() {
    let aligned_copy = |x: &Tensor<f32>, offset: Int32| {
        let frag = make_fragment::<f32>(4);
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy(&atom, &x.compose(Layout::new((4, 253), (1, 1))).slice((.., offset)), &frag);
    };
    let mut created = 0;
    let msg = build_panic(|| {
        Kernel::builder("var_offset_in_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let offset = Var::new((IndexExpr::from(thread_idx().x) * 4usize).into_value());
            created = line!() - 1;
            aligned_copy(&x, offset.get());
            for_range(0, 2, 1, |_| aligned_copy(&x, offset.get()));
        });
    });
    assert!(
        msg.contains("cannot prove that the source offset")
            && msg.contains(&format!("is the value of the Var created at tests/diagnostics.rs:{created}:"))
            && msg.contains("read the Var before the loop"),
        "{msg}"
    );
    Kernel::builder("var_offset_read_before_loop").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(256));
        let offset = Var::new((IndexExpr::from(thread_idx().x) * 4usize).into_value());
        let before = offset.get();
        for_range(0, 2, 1, |_| aligned_copy(&x, before));
    });
    // A Var holding a constant indexes registers, but not inside a loop.
    let msg = build_panic(|| {
        Kernel::build("var_register_index_in_loop", || {
            let out = param_tensor::<i32>("out", Layout::col_major(4));
            let frag = make_fragment::<i32>(4);
            frag.fill(7);
            let slot = Var::new(Int32::from(2));
            out.store(0usize, frag.load(slot.get()));
            for_range(0, 2, 1, |i| out.store(i, frag.load(slot.get())));
        });
    });
    assert!(
        msg.contains("register tensors must be indexed with static coordinates")
            && msg.contains("is the value of the Var created at tests/diagnostics.rs:"),
        "{msg}"
    );
    // The same error about a value that is not a Var's has no note.
    let msg = build_panic(|| {
        Kernel::builder("plain_offset_in_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            for_range(0, 2, 1, |i| aligned_copy(&x, i + 1));
        });
    });
    assert!(msg.contains("cannot prove that the source offset") && !msg.contains("Var"), "{msg}");
}

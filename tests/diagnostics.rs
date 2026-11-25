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
    assert!(msg.contains("'acc_dtype'") && msg.contains("BFloat16 operands") && msg.contains("Float32 only"), "{msg}");
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
    assert!(msg.contains("(16,8,4) and (16,8,8)"), "{msg}");
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
    assert!(msg.contains("fastmath cannot be combined with approx"), "{msg}");
    let msg = build_panic(|| {
        math_kernel(|x, y| {
            math::add_with(x, y, MathOptions { fastmath: true, ..MathOptions::rounding(RoundingMode::Zero) })
        });
    });
    assert!(msg.contains("fastmath cannot be combined with rounding"), "{msg}");
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

/// Misused compile options are reported, naming the option, before anything is compiled:
/// a typed level out of range panics at the caller (as CuTe DSL raises for `OptLevel(4)`);
/// strings and the environment return errors; options whose feature has not landed name the
/// part that adds them; and compiling needs a target.
#[test]
fn compile_option_misuse() {
    let message = build_panic(|| {
        let _ = CompileOptions::new().opt_level(4);
    });
    assert!(message.contains("optimization level must be 0 to 3") && message.contains("not 4"), "{message}");

    for (text, want) in [
        ("--opt-level 5", "`--opt-level 5`: the optimization level must be 0 to 3"),
        ("--gpu-arch sm_70", "unknown GPU architecture `sm_70`"),
        ("--gpu-arch sm_88", "sm_88 is in the PTX ISA but not in CuTe DSL 4.8.1's Arch"),
        ("dump-ptx-path=maybe", "takes 1, true, yes or on"),
        ("remarks{ptxas}", "`ptxas` is not a diagnostic category"),
        ("--link-libraries libdevice.bc", "linking device libraries is not available yet"),
        ("iket", "IKET in-kernel event tracing is not available yet"),
        ("--frobnicate", "unknown option `--frobnicate`"),
        ("--keep", "ambiguous option `--keep`"),
        ("--h x", "ambiguous option `--h`: it could be --help, --host-target"),
        ("--opt-level 1 -", "unexpected `-`"),
        ("--opt-level 1 ''", "unexpected empty word, which holds no option"),
        ("--ptxas -v", "--ptxas-options needs a value"),
        ("--opt-level '1", "unterminated quote"),
    ] {
        let e = CompileOptions::parse(text).unwrap_err().to_string();
        assert!(e.contains(want), "{text}: {e}");
    }
    for (name, value, want) in [
        ("CUTE_RS_LINEINFO", "maybe", "CUTE_RS_LINEINFO=\"maybe\" is not a boolean"),
        ("CUTE_RS_KEEP", "ptx,asm", "CUTE_RS_KEEP: `asm` is not something to keep"),
    ] {
        let env = |n: &str| (n == name).then(|| value.to_string());
        let e = CompileOptions::new().with_env_from(env).unwrap_err().to_string();
        assert!(e.contains(want), "{name}={value}: {e}");
    }

    // A kernel records where it was built, for kernel-level diagnostics.
    let kernel = Kernel::build("no_target", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        x.store(thread_idx().x, Float32::from(1.0));
    });
    let built_at = line!() - 4;
    assert_eq!((kernel.ir().loc.file(), kernel.ir().loc.line()), (file!(), built_at));
    if let Some(tc) = Toolchain::discover() {
        let e = kernel.compile_with(&tc, &CompileOptions::new()).unwrap_err().to_string();
        assert!(e.contains("no target architecture to compile kernel `no_target` for"), "{e}");
        let hook =
            CompileOptions::new().gpu_arch(Arch::Sm80).stage_hook(|m| m.set_argument_attribute("x", 0, "a", "unit"));
        let e = kernel.compile_with(&tc, &hook).unwrap_err().to_string();
        assert!(e.contains("stage hook failed") && e.contains("no llvm.func @x"), "{e}");
    }
}

/// The message of a kernel build applying `f` to an element of a `T`
/// parameter, which must fail, and the line `f` is on.
#[track_caller]
fn int_misuse<T: Numeric, U: Numeric>(f: impl Fn(Value<T>) -> Value<U>) -> (String, u32) {
    let line = std::panic::Location::caller().line();
    let msg = build_panic(move || {
        Kernel::builder("int_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<T>("x", Layout::col_major(32));
            let y = param_tensor::<U>("y", Layout::col_major(32));
            let t = thread_idx().x;
            y.store(t, f(x.load(t)));
        });
    });
    (msg, line)
}

/// The integer and bit instructions take the types CuTe DSL takes (the
/// rest is up to Rust's types: one type for the data operands, a `u8`
/// lookup table, Int32 saturating arithmetic), and the validator names the
/// instruction, the type and the user's line.
#[test]
fn int_instructions_take_cute_dsl_types() {
    let cases: Vec<((String, u32), &str)> = vec![
        (int_misuse(|x: Int8| bfe(x, 0, 4)), "bfe takes Int32, Uint32, Int64 or Uint64 operands, not i8"),
        (int_misuse(|x: Uint16| bfe(x, 4, 8)), "bfe takes Int32, Uint32, Int64 or Uint64 operands, not u16"),
        (int_misuse(|x: Int16| bfi(x, x, 0, 4)), "bfi takes Int32, Uint32, Int64 or Uint64 operands, not i16"),
        (int_misuse(|x: Int64| lop3(x, x, x, 0x96)), "lop3 takes Int32 or Uint32 operands, not i64"),
        (int_misuse(|x: Uint16| shf(x, x, 3, ShfKind::Left)), "shf takes Int32 or Uint32 operands, not u16"),
        (int_misuse(|x: Uint64| prmt(x, x, 0x3210u64)), "prmt takes Int32 or Uint32 operands, not u64"),
        (int_misuse(|x: Int8| prmt(x, x, 0x10i8)), "prmt takes Int32 or Uint32 operands, not i8"),
        (int_misuse(|x: Int64| mul24(x, x, true)), "mul24 takes Int32 or Uint32 operands, not i64"),
        (int_misuse(|x: Uint8| mad24(x, x, x, false)), "mad24 takes Int32 or Uint32 operands, not u8"),
        (int_misuse(|x: Int16| mul_hi(x, x)), "mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not i16"),
        (int_misuse(|x: Uint8| x.mul_hi(x)), "mul_hi takes Int32, Uint32, Int64 or Uint64 operands, not u8"),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "the error should point at line {line}: {msg}");
    }
}

/// One warp of 32 threads running `f` on its lane index and an element of a
/// Uint32 parameter; the simulator's error, which must be reported.
fn warp_misuse(name: &str, f: impl Fn(Int32, Uint32) -> Uint32) -> SimError {
    let kernel = Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, f(lane_idx(), x.load(t)));
    });
    simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[7u32; 32])])
}

/// What the PTX ISA leaves undefined for the warp collectives, reported by
/// the simulator at the user's line: a lane executing one outside its member
/// mask, a lane of the mask that never executes it (lanes of another branch
/// may, with independent thread scheduling, but these take the branch that
/// does not: a deadlock naming them), lanes whose masks disagree, and a
/// shuffle reading a lane outside the mask.
#[test]
fn warp_collectives_need_their_member_masks() {
    let plain = ReduxOptions::default();
    let cases: Vec<(SimError, &str)> = vec![
        (
            warp_misuse("shuffle_outside", |_, x| shuffle_sync_with(x, 0, 0x0000_ffffu32, 31)),
            "shuffle executed by lane 16 of warp 0, which is not in its member mask 0x0000ffff",
        ),
        (
            warp_misuse("redux_outside", |_, x| warp_redux_sync_with(x, ReduxKind::Add, 0xffff_fffeu32, plain)),
            "warp_redux_sync executed by lane 0 of warp 0, which is not in its member mask 0xfffffffe",
        ),
        (
            warp_misuse("match_outside", |_, x| match_sync(0x7fff_ffffu32, x, MatchKind::All)),
            "match_sync executed by lane 31 of warp 0, which is not in its member mask 0x7fffffff",
        ),
        (
            warp_misuse("sync_warp_outside", |_, x| {
                sync_warp_with(0x0000_ffffu32);
                x
            }),
            "sync_warp executed by lane 16 of warp 0, which is not in its member mask 0x0000ffff",
        ),
        (
            warp_misuse("redux_inactive", |lane, x| {
                if_else_yield(lane.lt(16), || warp_redux_sync(x, ReduxKind::Max), || x)
            }),
            "with member mask 0xffffffff, for lanes 16..=31 of the mask to reach it",
        ),
        (
            warp_misuse("sync_warp_inactive", |lane, x| {
                if_(lane.ge(8), || sync_warp());
                x
            }),
            "for lanes 0..=7 of the mask to reach it",
        ),
        (
            warp_misuse("match_inactive", |lane, x| {
                if_else_yield(lane.ne(5), || match_sync(FULL_MASK, x, MatchKind::Any), || x)
            }),
            "lane 5 took another path and waits at the end of a branch or loop",
        ),
        (
            warp_misuse("masks_disagree", |lane, x| {
                sync_warp_with(select(lane.eq(0), 3u32, 0xffff_fffeu32));
                x
            }),
            "lane 0 of warp 0 executes sync_warp with member mask 0x00000003, which includes lane 1, but lane 1 \
             executes it with 0xfffffffe",
        ),
        (
            warp_misuse("shuffle_non_member", |lane, x| {
                if_else_yield(lane.lt(16), || shuffle_sync_with(x, 20, 0x0000_ffffu32, 31), || x)
            }),
            "shuffle reads lane 20, which is not in the member mask 0x0000ffff",
        ),
        (
            warp_misuse("vote_inactive", |lane, x| if_else_yield(lane.lt(16), || vote_ballot_sync(x.eq(7u32)), || x)),
            "warp 0 (lanes 0..=15) of block (0,0,0) waits at vote_ballot_sync",
        ),
        (
            warp_misuse("vote_all_inactive", |lane, x| {
                if_else_yield(lane.ne(9), || vote_all_sync(x.eq(7u32)).to(), || x)
            }),
            "waits at vote_all_sync",
        ),
    ];
    for (e, want) in cases {
        assert!(e.message.contains(want), "expected {want:?} in {e}");
        assert!(e.to_string().contains("tests/diagnostics.rs"), "the error should point at the kernel source: {e}");
    }
}

/// The kinds and options of `warp_redux_sync` are checked against the value's
/// type, at the user's line: integer kinds on Int32 and Uint32, `fmax` and
/// `fmin` on Float32, `abs` and `nan` on Float32 only, nothing wider or
/// narrower than 32 bits.
#[test]
fn redux_kinds_follow_the_value_type() {
    let abs = ReduxOptions { abs: true, ..ReduxOptions::default() };
    let cases: Vec<((String, u32), &str)> = vec![
        (
            int_misuse(|x: Float32| warp_redux_sync(x, ReduxKind::Add)),
            "warp_redux_sync: add reduces Int32 or Uint32 values, not f32 (Float32 takes fmax, fmin, max and min)",
        ),
        (
            int_misuse(|x: Int32| warp_redux_sync(x, ReduxKind::Fmax)),
            "warp_redux_sync: fmax reduces Float32 values, not i32",
        ),
        (
            int_misuse(|x: Int32| warp_redux_sync_with(x, ReduxKind::Max, FULL_MASK, abs)),
            "redux.sync.max.abs.s32 is not a PTX instruction: abs and nan apply to the Float32 kinds (fmax, fmin) only",
        ),
        (
            int_misuse(|x: Uint64| warp_redux_sync(x, ReduxKind::Max)),
            "warp_redux_sync: max reduces Int32 or Uint32 values, not u64",
        ),
        (
            int_misuse(|x: Int16| warp_redux_sync(x, ReduxKind::Xor)),
            "warp_redux_sync: xor reduces Int32 or Uint32 values, not i16",
        ),
        (
            int_misuse(|x: Float64| warp_redux_sync(x, ReduxKind::Min)),
            "warp_redux_sync: min reduces Int32 or Uint32 values, not f64",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "the error should point at line {line}: {msg}");
    }
}

/// Float32 `redux.sync` exists on sm_100a, and from PTX ISA 8.8 on the
/// sm_100f family, only: compiling for any other target is refused with the
/// accepted targets and the user's line, naming the instruction with its
/// modifiers.
#[test]
fn float_redux_is_architecture_specific() {
    use cute::validate::{check_target, target_requirement};
    let line = line!() + 5;
    let kernel = Kernel::builder("float_redux").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = ReduxOptions { abs: true, nan: true };
        x.store(t, warp_redux_sync_with(x.load(t), ReduxKind::Min, FULL_MASK, opts));
    });
    for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90, Arch::Sm90a, Arch::Sm100, Arch::Sm120, Arch::Sm120a] {
        let e = check_target(kernel.ir(), arch).unwrap_err();
        let want = format!(
            "redux.sync.min.abs.NaN.f32 needs sm_100a (or the sm_100f family with PTX 8.8); the kernel is being \
             compiled for {}",
            arch.chip()
        );
        assert_eq!(e.message, want);
        assert!(e.to_string().contains(&format!("tests/diagnostics.rs:{line}:")), "{e}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    let redux = kernel.ir().body.insts.iter().find(|i| matches!(i.op, cute::ir::Op::Redux { .. })).unwrap();
    let need = target_requirement(&redux.op).unwrap();
    assert_eq!((need.sm, need.ptx), (100, 86));
    assert!(need.arch_specific.is_some());
    // The integer forms need sm_80 and PTX 7.0, like every supported target.
    let int = Kernel::builder("int_redux").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, warp_redux_sync(x.load(t), ReduxKind::Add));
    });
    for arch in Arch::ALL {
        assert_eq!(check_target(int.ir(), arch).unwrap(), arch.min_ptx(), "{arch:?}");
    }
}

/// `fmax` and `fmin` with `abs` are `.xorsign.abs`, which needs sm_86 (and
/// PTX 7.2): an sm_80 kernel is refused at the user's line. `nan` needs
/// sm_80, and `ftz` nothing.
#[test]
fn fmax_abs_needs_sm86() {
    use cute::validate::check_target;
    let line = line!() + 5;
    let kernel = Kernel::builder("fmax_abs").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = MinMaxOptions { abs: true, ftz: true, ..MinMaxOptions::default() };
        x.store(t, fmax_with(x.load(t), x.load((t + 1) % 32), opts));
    });
    let e = check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert_eq!(
        e.message,
        "min and max with .xorsign.abs (fmax and fmin with abs) needs sm_86 or newer; the kernel is being compiled \
         for sm_80"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics.rs:{line}:")), "{e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm86).unwrap(), 72);
    assert_eq!(check_target(kernel.ir(), Arch::Sm89).unwrap(), 78);
    let nan = Kernel::builder("fmin_nan").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let opts = MinMaxOptions { nan: true, ftz: true, ..MinMaxOptions::default() };
        x.store(t, fmin_with(x.load(t), 0.5f32, opts));
    });
    assert_eq!(check_target(nan.ir(), Arch::Sm80).unwrap(), 70);
}

/// `warp_reduction`'s groups are a power of two up to the warp.
#[test]
fn warp_reduction_groups_are_powers_of_two() {
    for group in [12, 0, 64, 33] {
        let msg = build_panic(|| {
            Kernel::builder("bad_group").block_dim([32, 1, 1]).build(|| {
                let x = param_tensor::<f32>("x", Layout::col_major(32));
                let t = thread_idx().x;
                x.store(t, warp_reduction_sum(x.load(t), group));
            });
        });
        let want = format!("warp_reduction: threads_in_group is {group}, but it must be a power of two from 1 to 32");
        assert!(msg.contains(&want), "{msg}");
    }
}

/// The second warp of a block of 48 threads has 16 lanes: the full mask's
/// other lanes count as exited, so warp collectives with it run there, but
/// a shuffle reading one of them (undefined in the PTX ISA) is reported.
#[test]
fn short_warps_lack_the_lanes_past_the_block() {
    let kernel = Kernel::builder("short_read").block_dim([48, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(48));
        let t = thread_idx().x;
        let all = match_sync(FULL_MASK, 1u32, MatchKind::All);
        x.store(t, shuffle_sync(x.load(t) + all, 20));
    });
    let e = simulate_err(&kernel, 48, &mut [KernelArg::buffer(&[7u32; 48])]);
    let want = "shuffle reads lane 20, which warp 1 does not have: it is the last warp of a block of 48 threads, \
                with 16 lanes";
    assert!(e.message.contains(want), "{e}");
    assert!(e.to_string().contains("tests/diagnostics.rs"), "the error should point at the kernel source: {e}");
}

/// The message of a kernel build over a `u32` parameter `x`, which `body`
/// gets with the thread index and must fail to build.
fn carry_misuse(body: impl FnOnce(Tensor<u32>, Int32)) -> String {
    build_panic(move || {
        Kernel::builder("carry_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<u32>("x", Layout::col_major(32));
            let t = thread_idx().x;
            body(x, t);
        });
    })
}

/// A carry-in instruction needs a carry-out before it in the same block of
/// straight-line code: the validator names it, the line, and what came
/// between, if anything.
#[test]
fn carry_chains_stay_in_straight_line_code() {
    let here = |line: u32| format!("tests/diagnostics.rs:{line}:");
    // No carry-out at all.
    let mut line = 0;
    let msg = carry_misuse(|x, t| {
        line = line!() + 1;
        x.store(t, addc(x.load(t), 1u32));
    });
    assert!(
        msg.contains("addc reads the carry flag, but nothing before it in this block of straight-line code sets it")
    );
    assert!(msg.contains(&here(line)), "{msg}");
    // A chain broken by an if_.
    let (mut set, mut broken) = (0, 0);
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        set = line!() + 1;
        let low = add_cc(a, a);
        broken = line!() + 1;
        if_(t.eq(0), || x.store(t, low));
        line = line!() + 1;
        x.store(t, addc(a, a));
    });
    assert!(msg.contains(&format!("addc reads the carry flag set by the add_cc at {}", here(set))), "{msg}");
    assert!(msg.contains(&format!("but the runtime if at {}", here(broken))), "{msg}");
    assert!(msg.contains(&here(line)) && msg.contains("keep each chain in straight-line code"), "{msg}");
    // The flag does not reach into a branch.
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        let low = add_cc(a, a);
        line = line!() + 1;
        if_(t.eq(0), || x.store(t, subc(low, a)));
    });
    assert!(msg.contains("subc reads the carry flag, but nothing before it in this block"), "{msg}");
    assert!(msg.contains(&here(line)), "{msg}");
    // Loops, calls, inline PTX that uses the flag or branches, and a 64-bit
    // division or remainder by a runtime value (which LLVM lowers with a
    // branch) lose it too.
    let division = "the 64-bit integer division or remainder by a runtime value (which LLVM lowers with a branch) at";
    for (what, breaker) in [
        ("the runtime loop at", 0),
        ("the printf (a call) at", 1),
        ("the inline PTX that uses the carry flag at", 2),
        ("the inline PTX that branches, calls or exits at", 3),
        (division, 4),
        (division, 5),
        (division, 6),
    ] {
        let msg = carry_misuse(|x, t| {
            let a = x.load(t);
            let low = mad_cc(a, a, a);
            let wide = a.to::<u64>() + 1u64;
            match breaker {
                0 => for_range(0, 2, 1, |i| x.store(i, low)),
                1 => printf("%u\n", &[low.into()]),
                2 => inline_ptx_effect("{ .reg .b32 r; add.cc.u32 r, $0, 1; }", &[low.into()]),
                3 => inline_ptx_effect("{ .reg .pred p; setp.ne.u32 p, $0, 0; @p bra.uni skip; skip: }", &[low.into()]),
                4 => x.store(t, (wide * 5u64 / wide).to::<u32>()),
                5 => x.store(t, (wide.bitcast::<i64>() % wide.bitcast::<i64>()).to::<u32>()),
                _ => x.store(t, (7u64 / wide).to::<u32>()),
            }
            x.store(t, madc_cc(a, a, low));
        });
        assert!(msg.contains("madc_cc reads the carry flag set by the mad_cc at") && msg.contains(what), "{msg}");
    }
    // A chain cannot switch between adding and subtracting: the GPU keeps a
    // borrow as the inverted carry.
    let (mut set, mut read) = (0, 0);
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        set = line!() + 1;
        let low = add_cc(a, 3u32);
        read = line!() + 1;
        x.store(t, subc(low, a));
    });
    assert!(msg.contains(&format!("subc reads the carry flag set by the add_cc at {}", here(set))), "{msg}");
    assert!(msg.contains("but a subtraction reads it as a borrow, which an addition or multiply-add does not leave"));
    assert!(msg.contains("start a new chain (sub_cc)") && msg.contains(&here(read)), "{msg}");
    let msg = carry_misuse(|x, t| {
        let a = x.load(t);
        let low = sub_cc(a, 3u32);
        x.store(t, madc_cc(low, a, a));
    });
    assert!(msg.contains("madc_cc reads the carry flag set by the sub_cc at"), "{msg}");
    assert!(msg.contains("but an addition reads it as a carry, which a subtraction does not leave"), "{msg}");
    // Inline PTX that leaves the flag alone, loads, stores and arithmetic do not.
    Kernel::builder("carry_ok").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let low = add_cc(x.load(t), 5u32);
        x.store(t, low * 3u32);
        inline_ptx_effect("{ .reg .b32 r; add.u32 r, $0, 1; }", &[low.into()]);
        x.store((t + 1) % 32, addc(x.load(t), 7u32));
        // Adds and multiply-adds are one kind.
        let low = mad_cc(low, 3u32, 1u32);
        x.store(t, addc_cc(low, 2u32) ^ madc(low, low, low));
        // A 64-bit division by a constant (a multiplication in the PTX) and
        // a 32-bit division by a runtime value (one div.u32) keep the flag.
        let wide = x.load(t).to::<u64>();
        let low = add_cc(x.load(t), 9u32);
        let quotient = (wide / 10u64 + (wide.bitcast::<i64>() % -7i64).bitcast::<u64>()).to::<u32>();
        let narrow = x.load(t) / (x.load(t) | 1u32);
        x.store(t, addc(low, quotient ^ narrow));
    });
}

/// The validator also checks carry instructions built as IR by hand.
#[test]
fn carry_instructions_are_validated() {
    let kernel = Kernel::builder("carry_ir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        x.store(t, add_cc(x.load(t), x.load(t)));
    });
    let mutate = |f: &dyn Fn(&mut cute::ir::Op)| {
        let mut ir = kernel.ir().clone();
        for inst in &mut ir.body.insts {
            if matches!(inst.op, cute::ir::Op::CarryInst { .. }) {
                f(&mut inst.op);
            }
        }
        Kernel::from_ir(ir).map(|_| ()).unwrap_err().to_string()
    };
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { func, .. } = op {
            func.carry_out = false;
        }
    });
    assert!(
        e.contains(
            "add_cc: an extended-precision instruction reads or writes the carry flag, and this one does neither"
        ),
        "{e}"
    );
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { args, .. } = op {
            args.pop();
        }
    });
    assert!(e.contains("add_cc takes 2 operands, not 1"), "{e}");
    let e = mutate(&|op| {
        if let cute::ir::Op::CarryInst { func, .. } = op {
            func.carry_in = true;
        }
    });
    assert!(e.contains("addc_cc reads the carry flag, but nothing before it"), "{e}");
}

/// The message of a kernel build that must fail, over a `u32` parameter
/// `x`, and the line of the statement it fails at: the first `::new(` call
/// (`InlinePtx::new`, `InlineAsm::new`, `MlirOp::new`) after the line
/// `body` passes to `mark`, which is where the statement is located however
/// the formatter wraps it.
fn asm_misuse(body: impl FnOnce(Tensor<u32>, Int32, &mut dyn FnMut(u32))) -> (String, u32) {
    let mut line = 0;
    let msg = build_panic(|| {
        Kernel::builder("asm_misuse").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<u32>("x", Layout::col_major(32));
            let t = thread_idx().x;
            body(x, t, &mut |l| line = l);
        });
    });
    let source = include_str!("diagnostics.rs");
    let at = source.lines().enumerate().skip(line as usize).find(|(_, l)| l.contains("::new(")).expect("a statement");
    (msg, at.0 as u32 + 1)
}

/// Named inline PTX: every name must have an operand and every operand a
/// name, a named template takes no positional operands, and a predicate
/// guards one instruction. The build fails at the template's line.
#[test]
fn inline_ptx_names_its_operands() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlinePtx::new("add.u32 {$w0}, {$r0}, {$r3};").write::<u32>().read(x.load(t)).read(1u32).emit();
                x.store(t, y);
            }),
            "inline_ptx: `{$r3}` names no operand: there are 2 read-only operands (.read, .read_ptr), {$r0} to {$r1}",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlinePtx::new("mov.u32 {$w0}, {$r0};").write::<u32>().read(x.load(t)).read(5u32).emit();
                x.store(t, y);
            }),
            "operand {$r1} (one of the read-only operands (.read, .read_ptr)) does not appear in the template",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 {$w0}, {$rw0};").write::<u32>().emit();
                x.store(t, y);
            }),
            "`{$rw0}` names no operand: there are no read-write operands (.read_write)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 $0, {$r0};").write::<u32>().read(x.load(t)).emit();
                x.store(t, y);
            }),
            "`$0` is a positional operand, but this template names its operands",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("mov.u32 {$out}, {$r0};").write::<u32>().read(x.load(t)).emit();
                x.store(t, y);
            }),
            "`{$out}` is not an operand name: inline PTX names its operands {$wN}",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("add.u32 {$rw0}, {$rw0}, 1; add.u32 {$rw0}, {$rw0}, 2;")
                    .read_write(x.load(t))
                    .predicate(t.lt(4))
                    .emit();
                x.store(t, y);
            }),
            "a predicate guards one instruction (PTX's @p), but the template",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlinePtx::new("@{$r0} add.u32 {$rw0}, {$rw0}, 1;")
                    .read_write(x.load(t))
                    .read(t.lt(8))
                    .predicate(t.lt(4))
                    .emit();
                x.store(t, y);
            }),
            "has its own guard, and a predicate adds another (PTX's @p takes one)",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "the error should point at line {line}: {msg}");
    }
}

/// Raw inline assembly: the constraint string must match the outputs and
/// operands given, each code must be one NVPTX takes, each register class
/// must hold its operand's type, an immediate must be an integer constant,
/// an address must be 64-bit (or 32-bit into shared memory), a tied input
/// must have its output's type, and every `$n` must name an operand.
#[test]
fn inline_asm_constraints_match_their_operands() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "reads 2 operand(s) (0 + entries and 2 input(s)), but 1 were given with .arg",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let (a, b) =
                    InlineAsm::new("mov.u32 $0, $1;", "=r,r").output::<u32>().output::<u32>().arg(x.load(t)).emit();
                x.store(t, a + b);
            }),
            "has 1 output(s) (= and + entries), but 2 type(s) were given with .output::<T>()",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("add.u32 $0, $0, 1;", "+r").output::<u32>().arg(t).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 (i32) is tied to output $0 (u32), so they must have one type",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint64 =
                    InlineAsm::new("mov.b64 $0, $1;", "=l,r").output::<u64>().arg(x.load(t).to::<u64>()).emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: input $1 is u64, which a 32-bit integer register (constraint \"r\") does not hold; use \"l\"",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("shl.b32 $0, $1, $2;", "=r,r,n")
                    .output::<u32>()
                    .arg(x.load(t))
                    .arg(x.load(t))
                    .emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the immediate constraint \"n\", so it must be a constant",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $3;", "=r,r,r").output::<u32>().arg(x.load(t)).arg(1u32).emit();
                x.store(t, y);
            }),
            "inline assembly: the template refers to $3, but there are 3 operand(s) ($0 to $2)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $L;", "=r").output::<u32>().emit();
                x.store(t, y);
            }),
            "inline assembly: `$` must be followed by an operand number (or written `$$`)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                InlineAsm::new("st.global.u32 [$0], $1;", "l,r").arg_ptr(&x.ptr()).arg(t).emit();
            }),
            "inline assembly with no outputs, no side effects and no memory clobber does nothing",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "r,=r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "must list the outputs, then the inputs, then the clobbers",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=r,r,").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline_asm: entry 2 of the constraint string \"=r,r,\" is empty",
        ),
        // Codes NVPTX does not take, which LLVM 23 would drop with the
        // statement (LLVM 21 rejects them).
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,x").output::<u32>().arg(x.load(t)).arg(t).emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the constraint \"x\", which is not an NVPTX register class: b (Boolean), c \
             (8-bit), h (16-bit), r (32-bit), l (64-bit), f (Float32) or d (Float64), or the immediate n (an integer \
             constant); use \"r\" for i32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=x,r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: output $0 has the constraint \"x\", which is not an NVPTX register class: b (Boolean), \
             c (8-bit), h (16-bit), r (32-bit), l (64-bit), f (Float32) or d (Float64); use \"r\" for u32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, 1;", "=n").output::<u32>().emit();
                x.store(t, y);
            }),
            "inline assembly: output $0 has the immediate constraint \"n\", but an output is a register; use \"r\" for u32",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint64 =
                    InlineAsm::new("mov.b64 $0, $1;", "=q,l").output::<u64>().arg(x.load(t).to::<u64>()).emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: output $0 has the constraint \"q\", a 128-bit register, and cute-rs has no 128-bit type; \
             use \"l\" for u64",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("ld.u32 $0, $1;", "=r,m").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 has the constraint \"m\", which is not an NVPTX register class",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("mov.u32 $0, $1;", "=r,&r").output::<u32>().arg(x.load(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 has the constraint \"&r\", which is not an NVPTX register class",
        ),
        // An immediate is an integer constant: LLVM rejects a float (LLVM 23
        // only prints its error) and prints a Boolean `true` as -1.
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Float32 = InlineAsm::new("add.f32 $0, $1, $2;", "=f,f,n")
                    .output::<f32>()
                    .arg(x.load(t).to::<f32>())
                    .arg(1.5f32)
                    .emit();
                x.store(t, y.to::<u32>());
            }),
            "inline assembly: input $2 has the immediate constraint \"n\", which takes an integer constant, and this \
             one is f32; pass it in a register (\"f\")",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,i").output::<u32>().arg(x.load(t)).arg(true).emit();
                x.store(t, y);
            }),
            "inline assembly: input $2 has the immediate constraint \"i\", which takes an integer constant, and this \
             one is i1; pass it in a register (\"b\")",
        ),
        // Addresses are 64-bit, and 32-bit only into shared memory.
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = InlineAsm::new("ld.global.u32 $0, [$1];", "=r,r")
                    .output::<u32>()
                    .arg_ptr(&x.ptr().offset(t))
                    .side_effects()
                    .emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 is the address of global memory, which needs a 64-bit register (constraint \
             \"l\"); only a shared-memory address fits in 32 bits (\"r\")",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let smem = alloc_smem::<u32>(32, 16);
                let y: Uint32 =
                    InlineAsm::new("ld.shared.u32 $0, [$1];", "=r,h").output::<u32>().arg_ptr(&smem.offset(t)).emit();
                x.store(t, y);
            }),
            "inline assembly: input $1 is an address, which is a 64-bit register (constraint \"l\"), or a 32-bit one \
             (\"r\") for shared memory, not \"h\"",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "the error should point at line {line}: {msg}");
    }
    // What NVPTX takes builds: an early-clobber output, the immediate `i`, a
    // shared address in 32 or 64 bits, clobbers LLVM ignores, and a memory
    // clobber written in the constraint string, which keeps a statement
    // without outputs (with side effects) as .memory_clobber() does.
    let kernel = Kernel::builder("constraints_ok").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let smem = alloc_smem::<u32>(32, 16);
        let y: Uint32 =
            InlineAsm::new("add.u32 $0, $1, $2;", "=&r,r,i").output::<u32>().arg(x.load(t)).arg(-5i32).emit();
        InlineAsm::new("st.shared.u32 [$0], $1;", "r,r,~{memory}").arg_ptr(&smem.offset(t)).arg(y).emit();
        InlineAsm::new("st.shared.u32 [$0], $1;", "l,r, ~{cc}, ~{memory}").arg_ptr(&smem.offset(t)).arg(y).emit();
        x.store(t, make_tensor(&smem, Layout::col_major(32)).load(t));
    });
    let asm: Vec<String> =
        kernel.to_mlir().lines().filter(|l| l.contains("llvm.inline_asm")).map(String::from).collect();
    assert_eq!(asm.len(), 3, "{asm:#?}");
    assert!(asm[1..].iter().all(|l| l.contains("llvm.inline_asm has_side_effects")), "{asm:#?}");
}

/// A hand-written MLIR operation: every `%n` must name an operand and every
/// operand appear, and the text is one operation without its results.
#[test]
fn mlir_op_text_is_checked() {
    let cases: Vec<((String, u32), &str)> = vec![
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 =
                    MlirOp::new("llvm.intr.bswap(%1) : (i32) -> i32").operand(x.load(t)).result::<u32>().emit();
                x.store(t, y);
            }),
            "mlir_op: `%1` names no operand: there are 1 (%0 to %0)",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Uint32 = MlirOp::new("llvm.intr.bswap(%0) : (i32) -> i32")
                    .operand(x.load(t))
                    .operand(t)
                    .result::<u32>()
                    .emit();
                x.store(t, y);
            }),
            "mlir_op: operand %1 does not appear in",
        ),
        (
            asm_misuse(|x, t, mark| {
                mark(line!());
                let y: Int32 = MlirOp::new("%r = nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit();
                x.store(t, y.bitcast::<u32>());
            }),
            "`%r` is not an operand: the operands are %0, %1, ... in order (write the operation without its results",
        ),
    ];
    for ((msg, line), want) in cases {
        assert!(msg.contains(want), "expected {want:?} in {msg}");
        assert!(msg.contains(&format!("tests/diagnostics.rs:{line}:")), "the error should point at line {line}: {msg}");
    }
}

/// Without a simulation closure, hand-written PTX and MLIR can be compiled
/// but not simulated, and a closure must return one value per write-only
/// output; the errors name the line.
#[test]
fn inline_statements_need_a_simulation() {
    let run = |kernel: Kernel| simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[3u32; 32])]);
    let line = line!() + 4;
    let kernel = Kernel::builder("unsimulated").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = InlinePtx::new("popc.b32 {$w0}, {$r0};").write::<u32>().read(x.load(t)).pure().emit();
        x.store(t, y);
    });
    let e = run(kernel);
    assert!(e.message.contains("cannot be simulated; describe it with a Rust closure (InlinePtx::sim"), "{e}");
    assert!(e.to_string().contains(&format!("tests/diagnostics.rs:{line}:")), "{e}");
    let kernel = Kernel::builder("unsimulated_mlir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = MlirOp::new("llvm.intr.ctpop(%0) : (i32) -> i32").operand(x.load(t)).result::<u32>().emit();
        x.store(t, y);
    });
    assert!(
        run(kernel).message.contains("the MLIR operation `llvm.intr.ctpop(%0) : (i32) -> i32` cannot be simulated")
    );
    let kernel = Kernel::builder("short_simulation").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let (a, b) = InlinePtx::new("mov.b64 {{$w0}, {$w1}}, {$r0};")
            .write::<u32>()
            .write::<u32>()
            .read(x.load(t).to::<u64>())
            .sim(|r, _| vec![r[0]])
            .emit();
        x.store(t, a ^ b);
    });
    assert!(run(kernel).message.contains("inline PTX simulation returned 1 values, not 2"));
}

/// The validator checks inline assembly built as IR by hand too: an empty
/// entry of the constraint string, which LLVM rejects, is named.
#[test]
fn inline_asm_ir_is_validated() {
    let kernel = Kernel::builder("asm_ir").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<u32>("x", Layout::col_major(32));
        let t = thread_idx().x;
        let y: Uint32 = InlineAsm::new("add.u32 $0, $1, $2;", "=r,r,r").output::<u32>().arg(x.load(t)).arg(t).emit();
        x.store(t, y);
    });
    let with_constraints = |constraints: &str| {
        let mut ir = kernel.ir().clone();
        for inst in &mut ir.body.insts {
            if let cute::ir::Op::InlinePtx { constraints: c, .. } = &mut inst.op {
                *c = constraints.to_string();
            }
        }
        Kernel::from_ir(ir).map(|_| ()).err().map(|e| e.to_string())
    };
    let e = with_constraints("=r,,r").expect("an empty entry is rejected");
    assert!(e.contains("inline assembly: entry 1 of the constraint string \"=r,,r\" is empty"), "{e}");
    let e = with_constraints("=r,r,r,").expect("a trailing comma is rejected");
    assert!(e.contains("entry 3 of the constraint string"), "{e}");
    let e = with_constraints("=r,r,l").expect("a 64-bit register for an Int32 is rejected");
    assert!(e.contains("input $2 is i32, which a 64-bit integer register (constraint \"l\") does not hold"), "{e}");
    assert_eq!(with_constraints("=r,r,r"), None);
}

/// `increment_coord` needs a value in every leaf and a congruent shape.
#[test]
fn increment_coord_misuse() {
    let msg = build_panic(|| {
        Kernel::build("increment_underscore", || {
            let t = thread_idx().x;
            let _ = increment_coord((t, ..), (4, 8));
        });
    });
    assert!(msg.contains("increment_coord: coordinate (") && msg.contains(",_) has a `..`"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("increment_incongruent", || {
            let t = thread_idx().x;
            let _ = increment_coord((t, 1), (4, (2, 4)));
        });
    });
    assert!(msg.contains("is not congruent with shape (4,(2,4))"), "{msg}");
    // A leaf is not congruent with a 1-tuple, nor a pair with a leaf.
    let msg = build_panic(|| {
        let _ = increment_coord(1, (4,));
    });
    assert!(msg.contains("coordinate 1 is not congruent with shape (4)"), "{msg}");
    let msg = build_panic(|| {
        let _ = increment_coord((1, 2), 8);
    });
    assert!(msg.contains("coordinate (1,2) is not congruent with shape 8"), "{msg}");
}

/// `tile_to_shape` with an order takes a permutation of the target's modes,
/// and the target needs at least the block's rank.
#[test]
fn tile_to_shape_misuse() {
    let block = Layout::row_major((8, 64));
    for order in [&[0, 0][..], &[1, 2], &[0], &[0, 1, 2]] {
        let msg = build_panic(|| {
            tile_to_shape_ordered(&block, (64, 128), order);
        });
        assert!(msg.contains(&format!("order {order:?} is not a permutation of 0..2")), "{order:?}: {msg}");
    }
    let atom = make_composed_layout(make_swizzle(3, 3, 3), 0, block.clone());
    let msg = build_panic(|| {
        tile_to_shape_swizzled_ordered(&atom, (64, 128, 2), &[2, 0, 2]);
    });
    assert!(msg.contains("order [2, 0, 2] is not a permutation of 0..3"), "{msg}");
    let msg = build_panic(|| {
        tile_to_shape(&block, (64,));
    });
    assert!(msg.contains("block (8,64):(64,1) has rank 2, more than the rank-1 target shape (64)"), "{msg}");
}

/// `is_major` with a mode the stride does not have names itself, the mode and
/// the layout or stride, rather than failing inside the tuple code.
#[test]
fn is_major_mode_out_of_range() {
    let msg = build_panic(|| {
        Layout::row_major((4, 8)).is_major(5);
    });
    assert!(msg.contains("is_major: mode 5 is out of range for the rank-2 layout (4,8):(8,1)"), "{msg}");
    let msg = build_panic(|| {
        Layout::col_major(8).is_major(1);
    });
    assert!(msg.contains("is_major: mode 1 is out of range for the rank-1 layout 8:1"), "{msg}");
    let msg = build_panic(|| {
        is_major(2, &IntTuple::from((8, (1, 2))));
    });
    assert!(msg.contains("is_major: mode 2 is out of range for the rank-2 stride (8,(1,2))"), "{msg}");
    // A runtime stride shows as `?`, in a bare stride and in a layout.
    let msg = build_panic(|| {
        Kernel::build("is_major_runtime_stride", || {
            let layout = Layout::with_strides((4, 8), (thread_idx().x.as_index(), 1));
            is_major(3, layout.stride());
        });
    });
    assert!(msg.contains("is_major: mode 3 is out of range for the rank-2 stride (?,1)"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("is_major_runtime_layout_stride", || {
            Layout::with_strides((4, 8), (thread_idx().x.as_index(), 1)).is_major(5);
        });
    });
    assert!(msg.contains("is_major: mode 5 is out of range for the rank-2 layout (4,8):(?,1)"), "{msg}");
    // A mode that is an empty tuple, or starts with one, has no first stride.
    let msg = build_panic(|| {
        "((),8):((),1)".parse::<Layout>().unwrap().is_major(0);
    });
    assert!(msg.contains("is_major: mode 0 of the layout ((),8):((),1) is the empty tuple ()"), "{msg}");
    let msg = build_panic(|| {
        "(((),2),1):(((),1),2)".parse::<Layout>().unwrap().is_major(0);
    });
    assert!(
        msg.contains(
            "is_major: mode 0 of the layout (((),2),1):(((),1),2) has no first stride: it starts with an empty tuple"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        is_major(1, &"(1,((),2))".parse::<IntTuple>().unwrap());
    });
    assert!(msg.contains("is_major: mode 1 of the stride (1,((),2)) has no first stride: it starts with"), "{msg}");
}

/// `front` of a tuple that has no first leaf says where the empty tuple is.
#[test]
fn front_of_an_empty_tuple() {
    let msg = build_panic(|| {
        IntTuple::tuple([]).front();
    });
    assert!(msg.contains("front: the empty tuple () has no first leaf"), "{msg}");
    let msg = build_panic(|| {
        "(((),2),1)".parse::<IntTuple>().unwrap().front();
    });
    assert!(msg.contains("front: the tuple has no first leaf: its mode [0, 0] is the empty tuple ()"), "{msg}");
}

/// `leading_dim` needs a shape and a stride with the same nesting.
#[test]
fn leading_dim_incongruent() {
    let msg = build_panic(|| {
        leading_dim(&IntTuple::from((4, 8)), &IntTuple::from((1,)));
    });
    assert!(msg.contains("leading_dim: shape (4,8) and stride (1) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        leading_dim(&IntTuple::from((4, (2, 2))), &IntTuple::from((1, 4)));
    });
    assert!(msg.contains("leading_dim: shape (4,(2,2)) and stride (1,4) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("leading_dim_incongruent", || {
            let stride = Coord::from((1usize, thread_idx().x.as_index(), 4usize));
            leading_dim(&IntTuple::from((4, 8)), &stride);
        });
    });
    assert!(msg.contains("leading_dim: shape (4,8) and stride (1,?,4) are not congruent"), "{msg}");
}

/// The `usize` sizes that Rust loops unroll over refuse a layout whose size
/// is only known at run time, naming the dynamic leaf and the `IndexExpr`
/// query to use instead.
#[test]
fn static_sizes_of_dynamic_layouts() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_size", || {
            let m = thread_idx().x;
            let _ = Layout::col_major((m, 8)).size();
        });
    });
    assert!(
        msg.contains("Layout::size: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0], so its size is only known at run time; use cute::size(&layout, &[])"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_cosize", || {
            let k = thread_idx().x;
            let _ = Layout::with_strides((4, 8), (k, 1)).cosize();
        });
    });
    assert!(
        msg.contains("Layout::cosize: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("so its cosize is only known at run time; use cute::cosize(&layout, &[])"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_shape_size", || {
            let m = thread_idx().x;
            let _ = Shape::from((4, (m, 2))).size();
        });
    });
    assert!(
        msg.contains("size: shape (4,(?,2)) has the dynamic extent %")
            && msg.contains("at mode [1, 0], so its size is only known at run time; use cute::size(&shape, &[])"),
        "{msg}"
    );
    // A runtime stride on a size-1 mode does not count.
    Kernel::build("static_cosize", || {
        let k = thread_idx().x;
        assert_eq!(Layout::with_strides((1, 8), (k, 1)).cosize(), 8);
    });
}

/// Thread-value layouts, and so the tiled copies built from them, are static.
#[test]
fn thread_value_layouts_must_be_static() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_thread_layout", || {
            let m = thread_idx().x;
            let _ = make_layout_tv(&Layout::col_major((m, 4)), &Layout::col_major(4));
        });
    });
    assert!(
        msg.contains("make_layout_tv: the thread layout: layout (?,4):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0]; thread-value layouts are static"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_value_layout", || {
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let k = thread_idx().x;
            let _ = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::with_strides((2, 4), (1, k)));
        });
    });
    assert!(
        msg.contains("make_layout_tv: the value layout: layout (2,4):(1,?) has the dynamic stride %")
            && msg.contains("at mode [1]; thread-value layouts are static"),
        "{msg}"
    );
}

/// `upcast` counts offsets in larger units, which needs static strides.
#[test]
fn upcast_needs_a_static_layout() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_upcast", || {
            let k = thread_idx().x;
            let _ = Layout::with_strides((4, 8), (k, 1)).upcast(2);
        });
    });
    assert!(
        msg.contains("upcast: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("at mode [0]; upcast needs a static layout"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_upcast_extent", || {
            let m = thread_idx().x;
            let _ = Layout::col_major((8, m)).upcast(4);
        });
    });
    assert!(msg.contains("upcast: layout (8,?):(1,8) has the dynamic extent %"), "{msg}");
}

/// What the algebra cannot do with runtime leaves is refused with a message
/// naming the operation and the leaf, as the CuTe dialect refuses it.
#[test]
fn layout_algebra_limits_with_runtime_leaves() {
    // A product replicates a static block; only the tiler may be dynamic.
    let msg = build_panic(|| {
        Kernel::build("dynamic_block", || {
            let m = thread_idx().x;
            let _ = logical_product(&Layout::col_major((m, 8)), Layout::col_major((2, 4)));
        });
    });
    assert!(
        msg.contains("logical_product: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("a product replicates a static layout (only the tiler may be dynamic)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_raked_block", || {
            let k = thread_idx().x;
            let _ = raked_product(&Layout::with_strides(4, k), &Layout::col_major(2));
        });
    });
    assert!(msg.contains("raked_product: layout 4:? has the dynamic stride %"), "{msg}");
    // A tiler with a runtime stride must have a single mode.
    let msg = build_panic(|| {
        Kernel::build("dynamic_stride_tiler", || {
            let (m, k) = (thread_idx().x, block_idx().x);
            let _ = logical_divide(&Layout::row_major((64, 64)), Layout::new((m, 4), (1, k)));
        });
    });
    assert!(
        msg.contains("logical_divide: the tiler (?,4):(1,?) of layout (64,64):(64,1) has the runtime stride %")
            && msg.contains("and 2 modes after filtering; CuTe divides by a tiler with a runtime stride only when it"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_stride_complement", || {
            let k = thread_idx().x;
            let _ = complement(&Layout::with_strides((4, 8), (1, k)), 64);
        });
    });
    assert!(
        msg.contains("complement: layout (4,8):(1,?) has the runtime stride %")
            && msg.contains("only when it has a single mode, and this one has 2 after filtering"),
        "{msg}"
    );
    // Two runtime extents that cannot be ordered when the kernel is built are
    // compared at run time, but a static stride that does not divide the
    // static extent it lands in is refused, as by the static algebra, naming
    // the mode of a by-mode tiler.
    Kernel::build("unordered_extents", || {
        let (m, k, s) = (thread_idx().x, block_idx().x, block_idx().y);
        let c = composition(&Layout::new((m, 4), (1, s)), &Layout::new(k, 1));
        assert!(!c.shape().is_static());
    });
    let msg = build_panic(|| {
        Kernel::build("incompatible_mode", || {
            let k = thread_idx().x;
            let _ = logical_divide(&Layout::with_strides((8, (6, 8)), (k, (8, 1))), (.., Layout::new(4, 1)));
        });
    });
    assert!(
        msg.contains(
            "composition (6,8):(8,1) ∘ (4,12):(1,4) (mode [1] of layout (8,(6,8)):(?,(8,1))) is not defined: "
        ) && msg.contains("stride 4 is not compatible with mode size 6"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_left_inverse", || {
            let k = thread_idx().x;
            let _ = left_inverse(&Layout::with_strides((4, 8), (k, 1)));
        });
    });
    assert!(
        msg.contains("left_inverse: layout (4,8):(?,1) has the runtime stride %")
            && msg.contains("at mode [0]; a left inverse needs static strides"),
        "{msg}"
    );
}

/// Buffers sized when the kernel is built (fragments, shared memory,
/// per-thread arrays) are sized from their layouts, so those must be static.
/// (A tensor parameter with runtime extents is a buffer sized at launch.)
#[test]
fn buffers_need_static_layouts() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_stack_array", || {
            let m = thread_idx().x;
            let _ = make_rmem_tensor::<f32>(Layout::col_major((m, 8)));
        });
    });
    assert!(
        msg.contains("make_rmem_tensor: layout (?,8):(1,?) has the dynamic extent %")
            && msg.contains("a per-thread array is sized from its layout"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_fragment", || {
            let m = thread_idx().x;
            let _ = make_fragment::<f32>((4, m));
        });
    });
    assert!(
        msg.contains("make_fragment: layout (4,?):(1,4) has the dynamic extent %")
            && msg.contains("register fragments have static shapes"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_smem", || {
            let k = thread_idx().x;
            let _ = make_smem_tensor::<f32>(Layout::with_strides((4, 8), (k, 1)));
        });
    });
    assert!(
        msg.contains("make_smem_tensor: layout (4,8):(?,1) has the dynamic stride %")
            && msg.contains("a shared-memory allocation is sized from its layout"),
        "{msg}"
    );
}

/// `size`, `shape` and `cosize` take mode paths that stay inside the tuple
/// and stop at a leaf, as in the CuTe dialect.
#[test]
fn mode_paths_of_size_queries() {
    let l = Layout::new((2, (3, 4)), (12, (1, 3)));
    let msg = build_panic(|| {
        size(&l, &[2]);
    });
    assert!(msg.contains("size: mode path [2] does not fit: (2,(3,4)) has rank 2, so it has no mode 2"), "{msg}");
    let msg = build_panic(|| {
        cosize(&l, &[0, 0]);
    });
    assert!(
        msg.contains(
            "cosize: mode path [0, 0] does not fit: mode [0] of (2,(3,4)) is a leaf, so the path cannot go on"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        shape(&l, &[1, 5]);
    });
    assert!(
        msg.contains("shape: mode path [1, 5] does not fit: mode [1] of (2,(3,4)) has rank 2, so it has no mode 5"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_mode_path", || {
            let m = thread_idx().x;
            size(&Layout::col_major((m, 8)), &[0, 1]);
        });
    });
    assert!(msg.contains("size: mode path [0, 1] does not fit: mode [0] of (?,8) is a leaf"), "{msg}");
}

/// A `?` in a parsed layout string is a placeholder for exercising the
/// algebra; an offset that depends on it cannot be computed in a kernel.
#[test]
fn parsed_placeholders_cannot_be_evaluated() {
    let msg = build_panic(|| {
        Kernel::build("placeholder", || {
            let out = param_tensor::<i32>("out", Layout::col_major(4));
            let l: Layout = "(?,4):(1,?)".parse().unwrap();
            out.store(0usize, Int32::from(l.offset_at(5)));
        });
    });
    assert!(
        msg.contains("which does not belong to kernel `placeholder`")
            && msg.contains("a `?` in a parsed layout string is a placeholder and cannot be evaluated in a kernel"),
        "{msg}"
    );
}

thread_local! {
    /// Where the last panic on this thread happened, as `file:line`.
    static PANIC_AT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// [`build_panic`], also returning where the panic was raised. A panic hook,
/// installed once for the whole test binary, records the location of each
/// panic on the panicking thread and then prints it as usual.
fn panic_location(f: impl FnOnce()) -> (String, String) {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let at = info.location().map(|l| format!("{}:{}", l.file(), l.line()));
            PANIC_AT.with(|p| *p.borrow_mut() = at);
            default(info);
        }));
    });
    let msg = build_panic(f);
    let at = PANIC_AT.with(|p| p.borrow_mut().take()).expect("the panic hook recorded a location");
    (msg, at)
}

/// What a panic's message names, and code that panics so.
type PanicCase = (&'static str, Box<dyn FnOnce()>);

/// The layout algebra's misuse messages point at the caller's line, not
/// into the library, including those raised in the middle of a composition
/// or of a by-mode divide, and name the function: a coordinate that does
/// not fit a layout's shape names the mode where it does not.
#[test]
fn layout_panics_report_the_callers_line() {
    let cases: Vec<PanicCase> = vec![
        (
            "Layout::size",
            Box::new(|| {
                Kernel::build("size_location", || {
                    let _ = Layout::col_major((thread_idx().x, 8)).size();
                });
            }),
        ),
        (
            "Layout::cosize",
            Box::new(|| {
                Kernel::build("cosize_location", || {
                    let _ = Layout::with_strides((4, 8), (thread_idx().x, 1)).cosize();
                });
            }),
        ),
        (
            "size: shape",
            Box::new(|| {
                Kernel::build("shape_size_location", || {
                    let _ = Shape::from((4, thread_idx().x)).size();
                });
            }),
        ),
        (
            "composition (6,8):(8,1) ∘ 4:2 is not defined",
            Box::new(|| {
                let _ = composition(&Layout::row_major((6, 8)), &Layout::new(4, 2));
            }),
        ),
        (
            "(mode [1] of layout (8,(6,8)):(1,(8,1))) is not defined",
            Box::new(|| {
                let _ = logical_divide(&Layout::new((8, (6, 8)), (1, (8, 1))), (.., 4));
            }),
        ),
        (
            "crd2idx: coordinate (1,2,3) has rank 3 but layout (4,8):(1,4) has rank 2",
            Box::new(|| {
                let _ = crd2idx((1usize, 2usize, 3usize), &make_layout((4, 8), (1, 4)));
            }),
        ),
        (
            "crd2idx: coordinate ((1,2),3) does not fit layout (4,8):(1,4): its mode [0], (1,2), has rank 2 where \
             the shape has the extent 4",
            Box::new(|| {
                let _ = crd2idx(((1usize, 2usize), 3usize), &make_layout((4, 8), (1, 4)));
            }),
        ),
        (
            "Layout::offset: coordinate (1,(2,3,4)) does not fit layout (4,(2,4)):(1,(4,8)): its mode [1], (2,3,4), \
             has rank 3 where the shape has (2,4), of rank 2",
            Box::new(|| {
                let _ = Layout::col_major((4, (2, 4))).offset(&Coord::from((1usize, (2usize, 3usize, 4usize))));
            }),
        ),
        (
            "Layout::crd2idx: coordinate (1,2,3) has rank 3",
            Box::new(|| {
                let _ = Layout::col_major((4, 8)).crd2idx(&IntTuple::from((1, 2, 3)));
            }),
        ),
        (
            "Layout::slice: coordinate (1,_,2) has rank 3 but layout (4,8):(1,4) has rank 2",
            Box::new(|| {
                let _ = Layout::col_major((4, 8)).slice(&(1usize, .., 2usize).into());
            }),
        ),
        (
            "Tensor::load: coordinate (1,?) has rank 2 but layout 32:1 has rank 1",
            Box::new(|| {
                Kernel::build("load_location", || {
                    let x = param_tensor::<f32>("x", Layout::col_major(32));
                    let _ = x.load((1usize, thread_idx().x));
                });
            }),
        ),
        (
            "logical_divide: the tiler (?,4):(1,?) of mode [1] (64:1) of layout (64,64):(64,1) has the runtime stride",
            Box::new(|| {
                Kernel::build("divide_location", || {
                    let (m, k) = (thread_idx().x, block_idx().x);
                    let _ = zipped_divide(&Layout::row_major((64, 64)), (.., Layout::new((m, 4), (1, k))));
                });
            }),
        ),
        (
            "Tensor::size: tensor (?,8):(1,?)",
            Box::new(|| {
                Kernel::build("tensor_location", || {
                    let _ = make_tensor(&param_ptr::<f32>("x"), Layout::col_major((thread_idx().x, 8))).size();
                });
            }),
        ),
        (
            "copy: the source tensor",
            Box::new(|| {
                Kernel::build("copy_location", || {
                    let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((thread_idx().x, 8)));
                    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
                    copy(&atom, &x, &make_fragment::<f32>((4, 8)));
                });
            }),
        ),
        (
            "partition_fragment_c: tensor",
            Box::new(|| {
                Kernel::build("fragment_location", || {
                    let c = make_tensor(&param_ptr::<f32>("c"), Layout::col_major((thread_idx().x, 16)));
                    let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
                    let _ = mma.get_slice(0usize).partition_fragment_c::<f32, _>(&c);
                });
            }),
        ),
        (
            "elem_less: (?,1) and (3,4,5) are not congruent",
            Box::new(|| {
                Kernel::build("elem_less_location", || {
                    let _ = elem_less((thread_idx().x, 1), (3, 4, 5));
                });
            }),
        ),
        (
            "ceil_div: cannot divide (4,8) by (2,4,2)",
            Box::new(|| {
                let _ = ceil_div((4, 8), (2, 4, 2));
            }),
        ),
    ];
    for (what, f) in cases {
        let (msg, at) = panic_location(f);
        assert!(msg.contains(what), "expected {what:?} in: {msg}");
        assert!(at.starts_with("tests/diagnostics.rs:"), "the panic for {what:?} is reported at {at}: {msg}");
    }
}

/// Tensors take runtime extents: `make_tensor` with a layout of runtime
/// extents, and dividing or composing a tensor by a tiler with runtime
/// extents, or with runtime strides that are not provably multiples of the
/// tensor's extents (which gives it runtime extents from composition,
/// whether a divide or a composition gives it), build. A runtime tiler of a
/// tiled copy is still refused by make_tiled_copy itself, and the static
/// upper bound idiom of examples/runtime_gemm.rs still works.
#[test]
fn tensors_take_runtime_extents() {
    Kernel::build("dynamic_tensor", || {
        let x = param_ptr::<f32>("x");
        let m = thread_idx().x;
        let t = make_tensor(&x, Layout::col_major((m, 8)));
        assert_eq!(t.shape().to_string(), "(?,8)");
        assert!(!size(&t, &[]).is_static());
        assert_eq!(size(&t, &[1]), IndexExpr::Static(8));
    });
    Kernel::build("dynamic_tile", || {
        let x = param_tensor::<f32>("x", Layout::col_major((64, 64)));
        let m = thread_idx().x;
        let tile = local_tile(&x, (Layout::new(m, 1), Layout::new(8, 1)), (0, 0));
        assert_eq!(tile.layout().to_string(), "(?,8):(1,64)");
    });
    for compose in [false, true] {
        Kernel::build("runtime_stride_tile", || {
            let x = param_tensor::<f32>("x", Layout::row_major((6, 8)));
            let s = param_scalar::<i32>("s");
            let t = if compose { x.compose(Layout::new(2, s * 8)) } else { logical_divide(&x, Layout::new(2, s)) };
            assert!(!t.shape().is_static(), "{t:?}");
        });
    }
    // A runtime stride that is provably a multiple of the rows steps over
    // them when the kernel is built, so the extent stays static.
    Kernel::build("multiple_stride_tile", || {
        let x = param_tensor::<f32>("x", Layout::row_major((6, 8)));
        let k = param_extent("k", 6);
        assert_eq!(x.compose(Layout::new(2, k)).shape(), &Shape::from(2));
    });
    // A runtime tiler of a tiled copy is refused by make_tiled_copy itself.
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_tiler", || {
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
            let m = thread_idx().x;
            let _ = make_tiled_copy(&atom, &Layout::col_major((32, 4)), (Layout::new(m, 1), Layout::new(4, 1)));
        });
    });
    assert!(
        msg.contains("make_tiled_copy: the tiler: layout ?:1 has the dynamic extent %")
            && msg.contains("a tiled copy covers a static tile"),
        "{msg}"
    );
    // Static extents with runtime strides, as in examples/runtime_gemm.rs.
    Kernel::build("runtime_strides", || {
        let x = param_ptr::<f32>("x");
        let k = thread_idx().x;
        let t = make_tensor(&x, Layout::with_strides((64, 64), (k, 1)));
        let tile = local_tile(&t, (8, 8), (1, 2));
        assert_eq!(tile.size(), 64);
    });
}

/// A refused composition names the right-hand side's stride or extent as
/// written, with what is left of it after the modes before when that is
/// different, and the elements a mode holds at the stride that split it.
#[test]
fn composition_refusals_name_the_right_hand_values() {
    let a = Layout::new((2, 3, 5), (1, 100, 1000));
    let msg = build_panic(|| {
        let _ = composition(&a, &Layout::new(2, 4));
    });
    assert!(
        msg.contains(
            "composition (2,3,5):(1,100,1000) ∘ 2:4 is not defined: stride 4 (2 after the modes before it) is not \
             compatible with mode size 3"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = composition(&a, &Layout::new(8, 1));
    });
    assert!(msg.contains("shape 8 (4 after the modes before it) is not compatible with mode size 3"), "{msg}");
    let msg = build_panic(|| {
        let _ = composition(&Layout::row_major((6, 8)), &Layout::new(4, 2));
    });
    assert!(
        msg.contains(
            "composition (6,8):(8,1) ∘ 4:2 is not defined: shape 4 is not compatible with mode size 6 at stride 2, \
             which has 3 elements"
        ),
        "{msg}"
    );
    // Inside a divide, the tiler's complement is the right-hand side.
    let msg = build_panic(|| {
        let _ = logical_divide(&Layout::row_major((6, 8)), Layout::new(2, 4));
    });
    assert!(
        msg.contains("composition (6,8):(8,1) ∘ (2,(4,6)):(4,(1,8)) is not defined: stride 4 is not compatible"),
        "{msg}"
    );
}

/// A runtime stride on the right-hand side of a composition, or in the
/// tiler of a divide, must not be 0 at launch, as in C++ CuTe: the extents
/// composition computes at run time divide by it, and the simulator reports
/// the division by zero. A static stride of 0 is fine.
#[test]
fn runtime_tiler_strides_must_not_be_zero() {
    let a = Layout::row_major((6, 8));
    assert_eq!(composition(&a, &Layout::new(2, 0)), Layout::new(2, 0));
    // Each thread stores the offset of index t % n, a runtime index into the
    // runtime extents, of (6,8):(8,1) composed with or divided by 2:s; at s =
    // 3 that is the static result with 2:3.
    type Op = fn(&Layout, &Layout) -> Layout;
    let ops: [(&str, Op, i32, &str); 2] = [
        ("compose_runtime_stride", |a, b| composition(a, b), 2, "2:24"),
        ("divide_runtime_stride", |a, b| logical_divide(a, b), 48, "(2,(3,8)):(24,(8,1))"),
    ];
    for (name, op, n, static_result) in ops {
        let kernel = Kernel::builder(name).block_dim([64, 1, 1]).build(|| {
            let s = param_scalar::<i32>("s");
            let out = param_tensor::<i32>("out", Layout::col_major(64));
            let t = thread_idx().x;
            out.store(t, Int32::from(crd2idx(t % n, &op(&Layout::row_major((6, 8)), &Layout::new(2, s)))));
        });
        let run = |s: i32| {
            let mut args = vec![KernelArg::ScalarI32(s), KernelArg::i32(vec![0; 64])];
            kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).map(|_| args[1].to_vec::<i32>())
        };
        let want = op(&a, &Layout::new(2, 3));
        assert_eq!(want.to_string(), static_result);
        let want: Vec<i32> = (0..64).map(|t| want.call(t % n as usize) as i32).collect();
        assert_eq!(run(3).unwrap(), want, "{name}");
        let e = run(0).unwrap_err();
        assert!(e.message.contains("division by zero"), "{name}: {e}");
    }
}

mod feature_kernels;

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
    assert!(e.to_string().contains("tests/feature_kernels/mod.rs"), "{e}");
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
        assert!(e.to_string().contains("tests/diagnostics.rs"), "{mode:?}: {e}");
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
        assert!(e.to_string().contains("tests/diagnostics.rs"), "{bug:?}: {e}");
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
            assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
        let at = msg.split("tests/diagnostics.rs:").nth(1).and_then(|r| r.split(':').next()?.parse::<u32>().ok());
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "the error should point at the copy: {e}");
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
        assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
        assert!(msg.contains("tests/diagnostics.rs:"), "the error should point at the access: {msg}");
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

/// An operation on two Int32 words, for [`packed_kernel`].
type PackedOp = Box<dyn Fn(Int32, Int32) -> Int32>;

/// A kernel applying `f` to two Int32 words.
fn packed_kernel(f: impl Fn(Int32, Int32) -> Int32) -> Kernel {
    Kernel::builder("packed").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major(64));
        let t = thread_idx().x;
        x.store(t, f(x.load(t), x.load(t + 32)));
    })
}

/// The half-precision options are checked as CuTe DSL checks them, in its
/// order, with cute-rs's own messages: `ftz` and `sat` are Float16 only,
/// `sat` and `relu` exclude each other, `oob` excludes `ftz` and `sat`, and
/// each operation takes only the keywords its CuTe DSL counterpart has.
#[test]
fn half_precision_options_follow_cute_dsl() {
    let opts = |ftz, sat, relu, oob| HalfOptions { ftz, sat, relu, oob, ..HalfOptions::default() };
    let cases: [(PackedOp, &str); 10] = [
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, a, opts(false, true, true, false))),
            "cute: fma_packed_f16x2: sat and relu exclude each other, but both are set",
        ),
        (
            Box::new(move |a, b| fma_packed_bf16x2(a, b, a, opts(true, false, false, false))),
            "cute: fma_packed_bf16x2: ftz applies to Float16 only, but it is set on a BFloat16 fma",
        ),
        (
            Box::new(move |a, b| add_packed_bf16x2(a, b, opts(false, true, false, false))),
            "cute: add_packed_bf16x2: sat applies to Float16 only, but it is set on a BFloat16 add",
        ),
        (
            Box::new(move |a, b| sub_packed_bf16x2(a, b, opts(true, false, false, false))),
            "cute: sub_packed_bf16x2: ftz applies to Float16 only, but it is set on a BFloat16 sub",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(true, false, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with ftz",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(false, true, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with sat",
        ),
        (
            Box::new(move |a, b| fma_packed_f16x2(a, b, b, opts(true, true, false, true))),
            "cute: fma_packed_f16x2: oob excludes ftz and sat, but oob is set with ftz and sat",
        ),
        (
            Box::new(move |a, b| add_packed_f16x2(a, b, opts(false, false, true, false))),
            "cute: add_packed_f16x2 takes no relu option",
        ),
        (
            Box::new(move |a, b| sub_packed_f16x2(a, b, opts(false, false, false, true))),
            "cute: sub_packed_f16x2 takes no oob option",
        ),
        (
            Box::new(|a, b| {
                let predicate = Some(a.lt(0));
                fma_packed_f16x2(a, b, b, HalfOptions { predicate, ..HalfOptions::default() })
            }),
            "cute: fma_packed_f16x2 takes no predicate",
        ),
    ];
    for (f, want) in cases {
        let msg = build_panic(|| {
            packed_kernel(f);
        });
        assert!(msg.contains(want), "expected {want:?} in {msg}");
    }
    let msg = build_panic(|| {
        Kernel::build("scalar", || {
            let x = param_tensor::<BF16>("x", Layout::col_major(32));
            let t = thread_idx().x;
            let opts = HalfOptions { sat: true, ..HalfOptions::default() };
            x.store(t, fma_bf16(x.load(t), x.load(t), x.load(t), opts));
        });
    });
    assert!(msg.contains("cute: fma_bf16: sat applies to Float16 only, but it is set on a BFloat16 fma"), "{msg}");
}

/// bf16x2 `add` and `sub`, `ex2.approx.ftz.bf16x2` and fma with `.oob` need
/// sm_90: compiling for sm_89 is refused at the user's line; `.oob` also
/// raises sm_90's PTX version to 8.1.
#[test]
fn bf16x2_add_needs_sm90() {
    use cute::validate::check_target;
    let line = line!() + 1;
    let kernel = packed_kernel(|a, b| add_packed_bf16x2(a, b, HalfOptions::default()));
    let e = check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert_eq!(
        e.message,
        "add.bf16x2 (add_packed_bf16x2) needs sm_90 or newer; the kernel is being compiled for sm_89"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics.rs:{line}:")), "{e}");
    // sm_90's own PTX ISA version, 7.8.
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 78);
    let kernel = packed_kernel(|a, _| exp2_packed_bf16x2(a, None));
    let e = check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert!(e.message.contains("ex2.approx.ftz.bf16x2 (exp2_packed_bf16x2) needs sm_90 or newer"), "{e}");
    let oob = HalfOptions { oob: true, ..HalfOptions::default() };
    let kernel = packed_kernel(move |a, b| fma_packed_f16x2(a, b, b, oob));
    let e = check_target(kernel.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("fma with .oob (oob=True) needs sm_90 or newer"), "{e}");
    assert_eq!(check_target(kernel.ir(), Arch::Sm90).unwrap(), 81);
    // The Float16 forms and BFloat16 fma need nothing beyond sm_80 and PTX 7.0.
    let kernel = packed_kernel(|a, b| {
        let relu = HalfOptions { relu: true, ..HalfOptions::default() };
        fma_packed_bf16x2(add_packed_f16x2(a, b, HalfOptions::default()), b, a, relu)
    });
    assert_eq!(check_target(kernel.ir(), Arch::Sm80).unwrap(), 70);
}

/// The s2f6x2 path of the int8 to BFloat16 conversion is
/// architecture-specific: sm_100 (no `a`) is refused with the targets that
/// have it and the user's line.
#[test]
fn s2f6x2_path_needs_an_arch_specific_target() {
    use cute::validate::check_target;
    let line = line!() + 6;
    let kernel = Kernel::builder("s2f6x2").block_dim([32, 1, 1]).build(|| {
        let q = param_tensor::<i8>("q", Layout::col_major(128));
        let out = param_tensor::<BF16>("out", Layout::col_major(128));
        let t = thread_idx().x;
        let values: Vec<Int8> = (0..4).map(|k| q.load(t * 4 + k)).collect();
        let converted = cvt_i8_bf16_intrinsic_with(&values, 4, I8ToBf16Path::S2f6x2);
        for (k, v) in converted.into_iter().enumerate() {
            out.store(t * 4 + k as i32, v);
        }
    });
    let e = check_target(kernel.ir(), Arch::Sm100).unwrap_err();
    assert_eq!(
        e.message,
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 (the s2f6x2 int8 conversion) needs sm_100a, sm_103a, \
         sm_107a, sm_110a, sm_120a or sm_121a; the kernel is being compiled for sm_100"
    );
    assert!(e.to_string().contains(&format!("tests/diagnostics.rs:{line}:")), "{e}");
    for arch in [Arch::Sm80, Arch::Sm90, Arch::Sm90a, Arch::Sm120] {
        assert!(check_target(kernel.ir(), arch).is_err(), "{arch:?}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 91);
    assert_eq!(check_target(kernel.ir(), Arch::Sm120a).unwrap(), 91);
}

/// `cvt_i8_bf16_intrinsic` converts one to `vec_i8.len()` values and
/// `sext_unpacked_i4_i8_intrinsic` a positive multiple of four; other
/// lengths are refused.
#[test]
fn conversion_intrinsic_lengths_are_checked() {
    let cvt = |n: usize, length: usize| {
        build_panic(move || {
            Kernel::build("bad_length", || {
                let q = param_tensor::<i8>("q", Layout::col_major(64));
                let out = param_tensor::<BF16>("out", Layout::col_major(64));
                let t = thread_idx().x;
                let values: Vec<Int8> = (0..n as i32).map(|k| q.load(t + k)).collect();
                for v in cvt_i8_bf16_intrinsic(&values, length) {
                    out.store(t, v);
                }
            });
        })
    };
    let msg = cvt(4, 0);
    assert!(
        msg.contains("cute: cvt_i8_bf16_intrinsic: length is 0, but it must be from 1 to vec_i8.len() (4)"),
        "{msg}"
    );
    let msg = cvt(3, 4);
    assert!(
        msg.contains("cute: cvt_i8_bf16_intrinsic: length is 4, but it must be from 1 to vec_i8.len() (3)"),
        "{msg}"
    );
    for (n, length) in [(8usize, 6usize), (8, 0), (4, 8)] {
        let msg = build_panic(move || {
            Kernel::build("bad_sext", || {
                let q = param_tensor::<i8>("q", Layout::col_major(64));
                let t = thread_idx().x;
                let values: Vec<Int8> = (0..n as i32).map(|k| q.load(t + k)).collect();
                for v in sext_unpacked_i4_i8_intrinsic(&values, length) {
                    q.store(t, v);
                }
            });
        });
        let want = format!(
            "cute: sext_unpacked_i4_i8_intrinsic: length is {length}, but it must be a multiple of 4 from 4 to \
             vec_unpacked_i4.len() ({n})"
        );
        assert!(msg.contains(&want), "{msg}");
    }
}

/// `pack` and `unpack` need two or more parts whose widths add up to the
/// whole's, and no Booleans.
#[test]
fn pack_widths_must_match() {
    let msg = build_panic(|| {
        Kernel::build("bad_pack", || {
            let x = param_tensor::<i8>("x", Layout::col_major(32));
            let y = param_tensor::<i32>("y", Layout::col_major(32));
            let t = thread_idx().x;
            y.store(t, pack::<i8, i32, 3>([x.load(t), x.load(t), x.load(t)]));
        });
    });
    assert!(msg.contains("cute: pack: 3 x i8 is 24 bits, but i32 is 32"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bad_unpack", || {
            let x = param_tensor::<f64>("x", Layout::col_major(32));
            let t = thread_idx().x;
            let [a, _, _]: [Float32; 3] = unpack(x.load(t));
            x.store(t, a.to::<f64>());
        });
    });
    assert!(msg.contains("cute: unpack: 3 x f32 is 96 bits, but f64 is 64"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bool_pack", || {
            let y = param_tensor::<u8>("y", Layout::col_major(32));
            let t = thread_idx().x;
            let b = t.lt(3);
            y.store(t, pack::<bool, u8, 8>([b; 8]));
        });
    });
    assert!(msg.contains("Boolean values have no bit width"), "{msg}");
}

/// The validator checks half-precision instructions and Pack/Unpack in IR
/// built by other means: modifiers PTX lacks, an undefined predicate, a
/// predicate on fma, a predicate that is not a Boolean, operand types, and
/// widths.
#[test]
fn validator_checks_half_instructions_and_packing() {
    use cute::ir::{HalfFn, Op, Region};
    fn edit(r: &mut Region, f: &mut dyn FnMut(&mut Op)) {
        for inst in &mut r.insts {
            f(&mut inst.op);
            for region in inst.op.regions_mut() {
                edit(region, f);
            }
        }
    }
    let kernel = packed_kernel(|a, b| fma_packed_bf16x2(a, b, a, HalfOptions::default()));
    let mut ir = kernel.ir().clone();
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { mods, .. } = op {
            mods.ftz = true;
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("fma.rn.ftz.bf16x2 is not a PTX instruction: ftz applies to Float16 only"), "{e}");
    let mut ir = kernel.ir().clone();
    let p = ir.value_types.len();
    ir.value_types.push(DType::Bool);
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { predicate, .. } = op {
            *predicate = Some(cute::ir::ValueId(p as u32));
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains(&format!("%{p} is used before it is defined")), "{e}");
    // The predicate of an add (a Boolean defined before the fma) moved onto
    // the fma, and an Int32 as the add's predicate.
    let kernel = packed_kernel(|a, b| {
        let sum = add_packed_f16x2(a, b, HalfOptions { predicate: Some(a.eq(b)), ..HalfOptions::default() });
        fma_packed_f16x2(sum, b, a, HalfOptions::default())
    });
    let mut ir = kernel.ir().clone();
    let mut add_predicate = None;
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { func, predicate, .. } = op {
            match func {
                HalfFn::Add => add_predicate = *predicate,
                _ => *predicate = add_predicate,
            }
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("fma.rn.f16x2: CuTe DSL's fma helpers take no predicate"), "{e}");
    let mut ir = kernel.ir().clone();
    edit(&mut ir.body, &mut |op| {
        if let Op::HalfInst { func: HalfFn::Add, args, predicate, .. } = op {
            *predicate = Some(args[0]);
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("the predicate of add.f16x2 must be i1, but"), "{e}");
    let kernel = Kernel::build("pairs", || {
        let x = param_tensor::<F16>("x", Layout::col_major(64));
        let y = param_tensor::<i32>("y", Layout::col_major(32));
        let t = thread_idx().x;
        y.store(t, pack([x.load(t), x.load(t + 32)]));
    });
    let mut ir = kernel.ir().clone();
    let mut parts = None;
    edit(&mut ir.body, &mut |op| {
        if let Op::Pack { parts: p, .. } = op {
            parts = Some(p.clone());
            p.pop();
        }
    });
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("pack needs at least two parts, not 1"), "{e}");
    let mut ir = kernel.ir().clone();
    let first = parts.unwrap()[0];
    ir.value_types[first.0 as usize] = DType::I8;
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("pack: operand types differ"), "{e}");
}

/// A tensor of runtime extents `(m, 8)`, in global memory.
fn dynamic_tensor(name: &str) -> Tensor<f32> {
    let m = param_extent(&format!("{name}_m"), 1);
    make_tensor(&param_ptr::<f32>(name), Layout::col_major((m, 8)))
}

/// The `usize` `Tensor::size` and the loops over every element refuse a
/// tensor whose size is only known at run time, naming the dynamic leaf and
/// the `IndexExpr` query or the for_range to use instead.
#[test]
fn static_sizes_of_dynamic_tensors() {
    let msg = build_panic(|| {
        Kernel::build("dynamic_tensor_size", || {
            let _ = dynamic_tensor("x").size();
        });
    });
    assert!(
        msg.contains("Tensor::size: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("at mode [0], so its size is only known at run time; use cute::size(&tensor, &[])"),
        "{msg}"
    );
    type Visit = fn(&Tensor<f32>);
    let visits: [(&str, Visit); 4] = [
        ("Tensor::fill", |t| t.fill(1.0)),
        ("Tensor::values", |t| drop(t.values())),
        ("Tensor::map_in_place", |t| t.map_in_place(|v| v + 1.0)),
        ("Tensor::store_values", |t| t.store_values(&[])),
    ];
    for (what, visit) in visits {
        let msg = build_panic(|| {
            Kernel::build("dynamic_tensor_visit", || visit(&dynamic_tensor("x")));
        });
        assert!(
            msg.contains(&format!("{what}: tensor (?,8):(1,?) has the dynamic extent %"))
                && msg.contains(&format!("but {what} visits every element when the kernel is built"))
                && msg.contains("write a for_range over the rest mode"),
            "{what}: {msg}"
        );
    }
}

/// Register fragments have static shapes: a fragment shaped like a tensor
/// of runtime extents, an MMA fragment of a whole tensor of runtime extents
/// rather than of a tile, and a fragment divided by a tiler with runtime
/// extents are refused, naming the dynamic extent.
#[test]
fn fragments_of_dynamic_shapes() {
    let msg = build_panic(|| {
        Kernel::build("fragment_like_dynamic", || {
            let _ = make_fragment_like::<f32, _>(&dynamic_tensor("x"));
        });
    });
    assert!(
        msg.contains("make_fragment_like: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("register fragments have static shapes")
            && msg.contains("make the fragment from one tile")
            && msg.contains("for_range over the rest mode"),
        "{msg}"
    );
    // A conversion makes a fragment too, and says so under its own name.
    let msg = build_panic(|| {
        Kernel::build("converted_dynamic", || {
            let _ = dynamic_tensor("x").to::<F16>();
        });
    });
    assert!(
        msg.starts_with("Tensor::to: tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("register fragments have static shapes"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("mma_fragment_dynamic", || {
            let m = param_extent("m", 1);
            let c = make_tensor(&param_ptr::<f32>("c"), Layout::col_major((m, 16)));
            let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
            let _ = mma.get_slice(thread_idx().x).partition_fragment_c::<f32, _>(&c);
        });
    });
    assert!(msg.contains("partition_fragment_c: tensor (1,?,2):(0,4,?{div=8}) has the dynamic extent ((%"), "{msg}");
    assert!(msg.contains("at mode [1]") && msg.contains("register fragments have static shapes"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("fragment_dynamic_tiler", || {
            let frag = make_fragment::<f32>((8, 4));
            let _ = local_tile(&frag, (Layout::new(param_extent("m", 1), 1), 4), (0, 0));
        });
    });
    assert!(
        msg.contains(
            "cute: a register fragment cannot take the layout ((?,4),(?,1)):((1,8),(?,0)), which has the dynamic"
        ) && msg.contains("register fragments have static shapes (a fragment is an array of registers)"),
        "{msg}"
    );
}

/// Copies and gemm unroll the per-thread chunks and the rest modes when the
/// kernel is built: a copy whose per-thread chunk is dynamic, or whose rest
/// mode is, an element-wise copy of runtime extents, gemm on an operand of
/// runtime extents and retile of one are refused, naming the leaf and what
/// must be static.
#[test]
fn copies_unroll_static_extents() {
    let atom = || make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
    // The per-thread chunk (mode 0) of the source has a runtime extent.
    let msg = build_panic(|| {
        Kernel::build("dynamic_chunk", || {
            let m = param_extent("m", 4);
            let src = make_tensor(&param_ptr::<f32>("x"), Layout::col_major((m, 2)));
            copy(&atom(), &src, &make_fragment::<f32>((8, 2)));
        });
    });
    assert!(
        msg.contains("copy: the source tensor (?{div=4},2):(1,?{div=4}) has the dynamic extent %")
            && msg.contains("at mode [0], in its mode 0, the chunk of values each thread copies")
            && msg.contains("splits mode 0 into the Universal { bits: 128 } atom's chunks when the kernel is built, so it must be static"),
        "{msg}"
    );
    // A rest mode of the partition of a whole tensor of runtime extents: one
    // tile at a time is copied, its rest mode sliced away by a for_range.
    let msg = build_panic(|| {
        Kernel::build("dynamic_rest", || {
            let m = param_extent("m", 1);
            let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((m, 16)));
            let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
            let thr = tiled.get_slice(thread_idx().x);
            let src = thr.partition_s(&x);
            let dst = make_fragment::<f32>((4, 4, 1));
            copy(&tiled, &src, &dst);
        });
    });
    assert!(
        msg.contains("copy: the source tensor ((4,1),?,1):((1,0),128,0) has the dynamic extent ((%")
            && msg.contains("at mode [1], a rest mode that copy unrolls when the kernel is built")
            && msg.contains("write a for_range over that mode and copy one slice of it per iteration"),
        "{msg}"
    );
    // The same copy of one tile per iteration of a for_range builds.
    Kernel::build("dynamic_rest_by_tile", || {
        let m = param_extent("m", 8);
        let x = make_tensor(&param_ptr::<f32>("x"), Layout::row_major((m, 16)));
        let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
        let thr = tiled.get_slice(thread_idx().x);
        let src = thr.partition_s(&x);
        for_range(0, size(&src, &[1]), 1, |i| {
            let dst = make_fragment::<f32>((4, 1));
            copy(&tiled, &src.slice((.., i, ..)), &dst);
        });
    });
    // copy_if names itself and the predicate.
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_if", || {
            let src = dynamic_tensor("x");
            let pred = make_fragment::<bool>(1);
            copy_if(&make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }), &pred, &src, &make_fragment::<f32>(8));
        });
    });
    assert!(msg.contains("copy_if: the source tensor (?,8):(1,?) has the dynamic extent %"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("dynamic_copy_if_destination", || {
            let pred = make_fragment::<bool>(8);
            pred.fill(true);
            let src = make_fragment::<f32>(8);
            src.fill(1.0);
            copy_if(&make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }), &pred, &src, &dynamic_tensor("y"));
        });
    });
    assert!(
        msg.contains("copy_if: the destination tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("in its mode 0, the chunk of values each thread copies"),
        "{msg}"
    );
    // autovec_copy copies every element with unrolled instructions.
    let msg = build_panic(|| {
        Kernel::build("dynamic_elementwise", || autovec_copy(&make_fragment::<f32>((4, 8)), &dynamic_tensor("y")));
    });
    assert!(
        msg.contains("autovec_copy: the destination tensor (?,8):(1,?) has the dynamic extent %")
            && msg.contains("but autovec_copy copies every element with instructions unrolled when the kernel")
            && msg.contains("write a for_range over the rest mode"),
        "{msg}"
    );
    // basic_copy and basic_copy_if of runtime size loop at run time, which
    // a register fragment cannot take part in.
    let msg = build_panic(|| {
        Kernel::build("dynamic_basic_copy", || basic_copy(&make_fragment::<f32>((4, 8)), &dynamic_tensor("y")));
    });
    assert!(
        msg.contains(
            "basic_copy: the source tensor is a register fragment, but the source (4,8):(1,4) and \
                      destination (?,8):(1,?) do not both have static shapes"
        ) && msg.contains("a register fragment is indexed with static coordinates only"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("dynamic_basic_copy_if", || {
            let y = dynamic_tensor("y");
            let pred = make_fragment::<bool>((4, 8));
            pred.fill(true);
            basic_copy_if(&pred, &dynamic_tensor("x"), &y);
        });
    });
    assert!(msg.contains("basic_copy_if: the predicate tensor is a register fragment"), "{msg}");
    // gemm unrolls its modes.
    let msg = build_panic(|| {
        Kernel::build("dynamic_gemm", || {
            let mma = make_tiled_mma(MmaOp::UniversalFma(DType::F32), Layout::col_major((4, 8, 1)));
            let thr = mma.get_slice(thread_idx().x);
            let k = param_extent("k", 1);
            let a = make_tensor(&param_ptr::<f32>("a"), Layout::col_major((4, k)));
            let b = make_tensor(&param_ptr::<f32>("b"), Layout::col_major((8, k)));
            let acc =
                thr.partition_fragment_c::<f32, _>(&make_tensor(&param_ptr::<f32>("c"), Layout::col_major((4, 8))));
            gemm(&mma, &acc, &thr.partition_a(&a), &thr.partition_b(&b), &acc);
        });
    });
    assert!(
        msg.contains("gemm: operand A ((1,1,?):(0,0,4)) has the dynamic extent %")
            && msg.contains(
                "gemm unrolls the per-thread values and the MMA_M, MMA_N and MMA_K modes when the kernel is built"
            ),
        "{msg}"
    );
    // retile regroups a fragment's values.
    let msg = build_panic(|| {
        Kernel::build("dynamic_retile", || {
            let tiled = make_tiled_copy_tv(&atom(), &Layout::row_major((8, 4)), &Layout::row_major((1, 4)));
            let _ = tiled.retile(&dynamic_tensor("x"));
        });
    });
    assert!(msg.contains("retile: tensor (?,8):(1,?) has the dynamic extent %"), "{msg}");
}

/// Coordinate tensors: elem_less needs congruent tuples, a swizzled tensor
/// takes only a shift that is provably non-negative, and an access before
/// the start of a tensor shifted by a negative domain_offset is reported
/// when it is not predicated off (a predicated-off cp.async there is not).
#[test]
fn coordinate_tensor_misuse() {
    // The CuTe dialect's incongruent elem_less operands.
    let msg = build_panic(|| {
        Kernel::build("elem_less_ranks", || {
            let _ = elem_less((3, 4), (3, 4, 5));
        });
    });
    assert!(msg.contains("elem_less: (3,4) and (3,4,5) are not congruent"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("elem_less_nesting", || {
            let _ = elem_less(((1, 2), 3), (4, 5));
        });
    });
    assert!(msg.contains("elem_less: ((1,2),3) and (4,5) are not congruent"), "{msg}");
    // A negative shift of a swizzled tensor.
    let msg = build_panic(|| {
        Kernel::build("swizzled_shift", || {
            let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
            let r = IndexExpr::signed(param_scalar::<i32>("r"));
            let _ = domain_offset((0, r), &s);
        });
    });
    assert!(
        msg.contains("domain_offset: the shift %") && msg.contains("swizzled by S<2,2,3> may be negative"),
        "{msg}"
    );
    // A non-negative shift of a swizzled tensor keeps the swizzle.
    Kernel::build("swizzled_shift_ok", || {
        let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
        let shifted = domain_offset((1, 4), &s);
        assert_eq!((shifted.base_offset(), shifted.swizzle()), (&IndexExpr::Static(36), s.swizzle()));
    });
    // A load before the shifted origin is out of bounds, at its line.
    let line = std::cell::Cell::new(0);
    let kernel = Kernel::builder("before_origin").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let y = param_tensor::<f32>("y", Layout::col_major(32));
        let r = IndexExpr::signed(param_scalar::<i32>("r"));
        let shifted = domain_offset(r, &x);
        let t = thread_idx().x;
        line.set(line!() + 1);
        y.store(t, shifted.load(t));
    });
    let mut args = vec![KernelArg::f32(vec![1.0; 64]), KernelArg::f32(vec![0.0; 32]), KernelArg::ScalarI32(-3)];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("out of bounds for `x` (64 elements)"), "{e}");
    let loc = e.loc.expect("the load's location");
    assert_eq!((loc.file(), loc.line()), ("tests/diagnostics.rs", line.get()), "{e}");
    let mut args = vec![KernelArg::f32(vec![1.0; 64]), KernelArg::f32(vec![0.0; 32]), KernelArg::ScalarI32(3)];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    // cp.async before the origin: zero-filled when predicated off, reported
    // when not.
    let kernel = Kernel::builder("cp_async_before_origin").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let y = param_tensor::<f32>("y", Layout::col_major(32));
        let r = IndexExpr::signed(param_scalar::<i32>("r"));
        let all = param_scalar::<i32>("all");
        let shifted = domain_offset(r.clone(), &x);
        let coords = domain_offset(r, &make_identity_tensor(64));
        let smem = make_smem_tensor::<f32>(Layout::col_major(32));
        let atom = make_copy_atom::<f32>(CopyOp::CpAsync { bytes: 4, bypass_l1: false });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(1));
        let t = thread_idx().x;
        let thr = tiled.get_slice(t);
        let src = thr.partition_s(&local_tile(&shifted, 32, 0));
        let c = thr.partition_s(&local_tile(&coords, 32, 0));
        let pred = make_predicate(&c, |c| elem_less(-1, c) | all.ne(0));
        copy_if(&tiled, &pred, &src, &thr.partition_d(&smem));
        cp_async_commit_group();
        cp_async_wait_group(0);
        sync_threads();
        y.store(t, smem.load(t));
    });
    let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let mut args = vec![KernelArg::f32(input.clone()), KernelArg::f32(vec![7.0; 32]), KernelArg::ScalarI32(-3)];
    args.push(KernelArg::ScalarI32(0));
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<f32> = (0..32).map(|t| if t < 3 { 0.0 } else { (t - 3) as f32 }).collect();
    assert_eq!(args[1].as_f32(), want);
    let mut args = vec![KernelArg::f32(input), KernelArg::f32(vec![7.0; 32]), KernelArg::ScalarI32(-3)];
    args.push(KernelArg::ScalarI32(1));
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("cp.async source read") && e.message.contains("out of bounds for `x`"), "{e}");
}

/// The shape arithmetic's misuse: ceil_div and round_up rank errors,
/// shape_div of static leaves of which neither divides the other, and the
/// widths of basic_copy_if.
#[test]
fn shape_arithmetic_misuse() {
    let msg = build_panic(|| {
        let _ = ceil_div((4, 8), (2, 4, 2));
    });
    assert!(
        msg.contains("ceil_div: cannot divide (4,8) by (2,4,2): the tiler (2,4,2) has rank 3, more than the rank 2"),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = ceil_div(7, 0);
    });
    assert!(msg.contains("ceil_div: cannot divide 7 by 0: the tiler has an extent of 0"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up((9, 10), (4, 8, 2));
    });
    assert!(msg.contains("round_up: cannot round (9,10) up to a multiple of (4,8,2): (4,8,2) has rank 3"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up(9, (4, 8));
    });
    assert!(msg.contains("the integer 9 needs an integer to round to, not the tuple (4,8)"), "{msg}");
    let msg = build_panic(|| {
        let _ = round_up(Shape::Branch(vec![]), 4);
    });
    assert!(msg.contains("round_up: cannot round () up to a multiple of 4: () is the empty tuple"), "{msg}");
    let msg = build_panic(|| {
        let _ = shape_div(7, 3);
    });
    assert!(msg.contains("shape_div: cannot divide 7 by 3: neither 7 nor 3 divides the other"), "{msg}");
    let msg = build_panic(|| {
        let _ = shape_div((4, 6), (2, 4, 1));
    });
    assert!(msg.contains("shape_div: cannot divide (4,6) by (2,4,1): (4,6) has rank 2 and (2,4,1) rank 3"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("copy_if_widths", || {
            let pred = make_fragment::<bool>(8);
            pred.fill(true);
            basic_copy_if(&pred, &make_fragment::<f32>(8), &make_fragment::<f64>(8));
        });
    });
    assert!(
        msg.contains(
            "basic_copy_if: the source elements (f32, 32 bits) and the destination elements (f64, 64 bits) \
                      differ in width"
        ),
        "{msg}"
    );
    // Dividing or masking an index that may be negative.
    let msg = build_panic(|| {
        Kernel::build("negative_division", || {
            let r = IndexExpr::signed(param_scalar::<i32>("r"));
            let _ = (r + 8usize) / 4usize;
        });
    });
    assert!(
        msg.contains("cute: index expression (%")
            && msg.contains("may be negative (it holds a negative or signed term)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = IndexExpr::from(-3) % 2usize;
    });
    assert!(msg.contains("cute: index expression -3 % 2: the operand -3 may be negative"), "{msg}");
}

/// Accesses past a runtime extent index the buffer like any other: the
/// simulator reports the one that falls outside the buffer, at its line.
#[test]
fn overruns_of_runtime_extents_are_reported() {
    let line = std::cell::Cell::new(0);
    let kernel = Kernel::builder("runtime_overrun").block_dim([32, 1, 1]).build(|| {
        let n = param_extent("n", 1);
        let x = param_tensor::<f32>("x", Layout::col_major(n));
        let t = thread_idx().x;
        line.set(line!() + 1);
        x.store(t, 1.0);
    });
    let mut args = vec![KernelArg::ScalarI32(20), KernelArg::f32(vec![0.0; 20])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("out of bounds for `x` (20 elements)"), "{e}");
    let loc = e.loc.expect("the store's location");
    assert_eq!((loc.file(), loc.line()), ("tests/diagnostics.rs", line.get()), "{e}");
    // In bounds, it runs.
    let mut args = vec![KernelArg::ScalarI32(32), KernelArg::f32(vec![0.0; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), vec![1.0; 32]);
}

/// `rank` takes a mode path as `size`, `shape` and `cosize` do, and refuses
/// one that leaves the tuple or goes on past a leaf, naming the function.
#[test]
fn mode_paths_of_rank() {
    let l = Layout::new((4, (3, 2)), (2, (8, 1)));
    let msg = build_panic(|| {
        rank(&l, &[2]);
    });
    assert!(msg.contains("rank: mode path [2] does not fit: (4,(3,2)) has rank 2, so it has no mode 2"), "{msg}");
    let msg = build_panic(|| {
        rank(&make_composed_layout(make_swizzle(2, 3, 3), 0, l.clone()), &[0, 0]);
    });
    assert!(
        msg.contains("rank: mode path [0, 0] does not fit: mode [0] of (4,(3,2)) is a leaf, so the path cannot go on"),
        "{msg}"
    );
    let msg = build_panic(|| {
        rank(&IntTuple::from(8), &[1]);
    });
    assert!(msg.contains("rank: mode path [1] does not fit: 8 has rank 1, so it has no mode 1"), "{msg}");
}

/// Breaking an assumption is undefined behaviour on the GPU: the simulator
/// reports the value, the promised divisor and the `assume` line.
#[test]
fn violated_assumption_names_its_line() {
    let mut line = 0;
    let kernel = Kernel::builder("broken_assume").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        line = line!() + 1;
        let even = assume(t * 2 + 2, Some(4), false);
        out.store(t, even);
    });
    let mut args = vec![KernelArg::buffer(&[0i32; 32])];
    let err = simulate_err(&kernel, 32, &mut args);
    assert_eq!(err.loc.map(|l| l.line()), Some(line), "{err}");
    assert_eq!(err.thread, Some([0, 0, 0]), "{err}");
    assert!(err.message.contains("the value 2 is not a multiple of 4"), "{err}");
    let kernel = Kernel::builder("broken_pow2").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, assume(t + 1, None, true));
    });
    let err = simulate_err(&kernel, 32, &mut [KernelArg::buffer(&[0i32; 32])]);
    assert_eq!(err.thread, Some([2, 0, 0]), "{err}");
    assert!(err.message.contains("the value 3 is not a power of two"), "{err}");
}

/// `assume` checks its arguments while the kernel is built: a divisor of 0,
/// `pow2` with a divisor that is not a power of two, and a static value
/// that breaks its own promise.
#[test]
fn bad_assumptions_are_refused_at_build_time() {
    let msg = build_panic(|| {
        Kernel::build("divby_zero", || {
            let _ = assume(param_scalar::<i32>("n"), Some(0), false);
        });
    });
    assert!(msg.contains("divby must be a positive integer"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("pow2_divby", || {
            let _ = assume(param_scalar::<i32>("n"), Some(12), true);
        });
    });
    assert!(msg.contains("with pow2, divby must be a power of two, not 12"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("static_breaks", || {
            let _ = assume(Int32::from(18), Some(4), false);
        });
    });
    assert!(msg.contains("the static value 18 is not a multiple of 4"), "{msg}");
    let msg = build_panic(|| {
        let _ = assume(IndexExpr::Static(24), Some(8), true);
    });
    assert!(msg.contains("the static value 24 is not a power of two"), "{msg}");
    let msg = build_panic(|| {
        let _ = assume(-8i32, None, true);
    });
    assert!(msg.contains("the static value -8 is not a power of two"), "{msg}");
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
        assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
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

/// Layouts with scaled-basis strides map coordinates to tuples, so what
/// needs offsets refuses them, naming the basis stride.
#[test]
fn basis_strides_are_refused_where_offsets_are_needed() {
    let id = make_identity_layout((4, 8));
    let msg = build_panic(|| {
        id.offset(&(1, 2).into());
    });
    assert!(
        msg.contains(
            "Layout::offset: layout (4,8):(1@0,1@1) has the basis stride 1@0 at mode [0], so it maps a \
                      coordinate to a tuple, not an offset; use Layout::eval"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        crd2idx((1, 2), &id);
    });
    assert!(msg.contains("crd2idx: layout (4,8):(1@0,1@1) has the basis stride 1@0"), "{msg}");
    let msg = build_panic(|| {
        id.slice(&(.., 1).into());
    });
    assert!(
        msg.contains("Layout::slice: layout (4,8):(1@0,1@1) has the basis stride 1@0")
            && msg.contains("slice_and_offset"),
        "{msg}"
    );
    let msg = build_panic(|| {
        make_composed_layout(make_swizzle(2, 3, 3), 0, id.clone());
    });
    assert!(
        msg.contains("make_composed_layout: invalid composition S<2,3,3> o 0 o (4,8):(1@0,1@1)")
            && msg.contains("basis stride 1@0 at mode [0]"),
        "{msg}"
    );
    let msg = build_panic(|| {
        id.upcast(2);
    });
    assert!(msg.contains("upcast: layout (4,8):(1@0,1@1) has the basis stride 1@0 at mode [0]"), "{msg}");
    let msg = build_panic(|| {
        make_layout_like(&id);
    });
    assert!(msg.contains("make_layout_like: layout (4,8):(1@0,1@1) has the basis stride"), "{msg}");
    // The inverses and complements of basis layouts take integer scales
    // and no integer strides besides 0.
    let msg = build_panic(|| {
        "(4,8):(1@0,4)".parse::<Layout>().unwrap().right_inverse();
    });
    assert!(msg.contains("right_inverse: layout (4,8):(1@0,4) mixes basis strides with the integer stride 4"), "{msg}");
    let msg = build_panic(|| {
        "(2,3):(2@0,3@0)".parse::<Layout>().unwrap().left_inverse();
    });
    assert!(msg.contains("left_inverse: layout (2,3):(2@0,3@0) is not injective"), "{msg}");
    // Arithmetic tuples that are not defined.
    let msg = build_panic(|| {
        "(2,3):(1@0,2)".parse::<Layout>().unwrap().eval(&(1, 1).into());
    });
    assert!(msg.contains("Layout::eval: layout (2,3):(1@0,2) at (1,1) is not defined: the integer stride 2"), "{msg}");
    let msg = build_panic(|| {
        "(2,3):(1@2,1@0@2)".parse::<Layout>().unwrap().eval(&(1, 1).into());
    });
    assert!(msg.contains("stride 1@0@2: a mode path goes on through the integer 1"), "{msg}");
    let msg = build_panic(|| {
        "4:1/4@0".parse::<Layout>().unwrap().eval(&2.into());
    });
    assert!(msg.contains("2 * 1/4 is 1/2, not a whole number"), "{msg}");
    let msg = build_panic(|| {
        complement(&"(2,3):(1@0,2)".parse::<Layout>().unwrap(), 12);
    });
    assert!(msg.contains("complement: layout (2,3):(1@0,2) mixes basis strides with the integer stride 2"), "{msg}");
}

/// As in CuTe DSL, a basis element whose scale is a ratio cannot be scaled.
#[test]
fn scaling_a_ratio_basis_is_refused() {
    let msg = build_panic(|| {
        let _ = StrideLeaf::from(ScaledBasis::new(Ratio::new(1, 2), 0)) * 4usize;
    });
    assert!(
        msg.contains(
            "cannot scale the basis element 1/2@0 by 4: scaling a basis element whose scale is a ratio is \
                      not supported"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        let _ = Ratio::new(1, 0);
    });
    assert!(msg.contains("Ratio::new(1, 0): the denominator must not be 0"), "{msg}");
    let msg = build_panic(|| {
        composition(&Layout::col_major((8, 8)), &"2:1/2@0".parse::<Layout>().unwrap());
    });
    assert!(
        msg.contains("composition (8,8):(1,8) ∘ 2:1/2@0 is not defined: the basis stride 1/2@0 has a ratio"),
        "{msg}"
    );
    let msg = build_panic(|| {
        composition(&Layout::col_major((8, 8)), &"2:1@1@1".parse::<Layout>().unwrap());
    });
    assert!(
        msg.contains(
            "composition (8,8):(1,8) ∘ 2:1@1@1 is not defined: the basis stride 1@1@1 names mode [1, 1], but \
                      mode [1] of the left-hand side is a leaf"
        ),
        "{msg}"
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
    assert!(e.message.contains("divergent control flow") && e.to_string().contains("tests/diagnostics.rs"), "{e}");
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
        let file = "tests/diagnostics.rs";
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
                "warp 0 of block (0,0,0) spins, in while_carry at tests/diagnostics.rs:{}",
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

#[test]
fn tf32_fragments_need_a_tf32_mma() {
    let msg = build_panic(|| {
        Kernel::build("tf32_on_f16_mma", || {
            let a = param_tensor::<F16>("a", Layout::col_major((16, 8)));
            let mma = make_tiled_mma(MmaKind::M16N8K8F32F16, Layout::col_major((1, 1, 1)));
            let thr = mma.get_slice(lane_idx());
            let _ = thr.partition_fragment_a::<TF32, _>(&a);
        });
    });
    assert!(msg.contains("fragments of") && msg.contains("hold f16, not tf32"), "{msg}");
}

#[test]
fn tf32_values_have_no_tf32_arithmetic_type() {
    // TFloat32 arithmetic is computed in Float32: the result is a Float32.
    let kernel = Kernel::build("tf32_sum", || {
        let out = param_tensor::<f32>("out", Layout::col_major(1));
        let x: TFloat32 = Float32::constant(1.0).to();
        let y: Float32 = x + x;
        out.store(0, y);
    });
    let mut args = vec![KernelArg::buffer(&[0f32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<f32>(), vec![2.0]);
}

#[test]
fn conversions_into_storage_only_fp8_formats_are_refused() {
    for (name, msg) in [
        ("Float8E4M3", "cannot convert Float32 to Float8E4M3: CuTe DSL refuses it as a conversion destination"),
        ("Float8E4M3B11FNUZ", "cannot convert Float32 to Float8E4M3B11FNUZ"),
        ("FloatNV8E5M3FNU", "cannot convert Float32 to FloatNV8E5M3FNU: its cvt (.ue5m3x2) needs PTX ISA 9.4"),
    ] {
        let got = build_panic(|| {
            Kernel::build("into_storage_fp8", || {
                let x = param_tensor::<f32>("x", Layout::col_major(32));
                let out = param_tensor::<u8>("out", Layout::col_major(32));
                let t = thread_idx().x;
                let v = x.load(t);
                let byte: Uint8 = match name {
                    "Float8E4M3" => v.to::<F8E4M3>().bitcast(),
                    "Float8E4M3B11FNUZ" => v.to::<F8E4M3B11FNUZ>().bitcast(),
                    _ => v.to::<F8E5M3FNU>().bitcast(),
                };
                out.store(t, byte);
            });
        });
        assert!(got.contains(msg) && got.contains("tests/diagnostics.rs"), "{name}: {got}");
    }
    // Reading them is fine.
    Kernel::build("from_storage_fp8", || {
        let x = param_tensor::<F8E4M3B11FNUZ>("x", Layout::col_major(32));
        let out = param_tensor::<f32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, x.load(t).to::<f32>());
    });
}

#[test]
fn ue8m0_conversions_need_sm100a() {
    let kernel = Kernel::build("ue8m0_on_sm90", || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let out = param_tensor::<F8E8M0FNU>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let [a, b] = cvt_f32x2_to_ue8m0x2([x.load(2 * t), x.load(2 * t + 1)], true);
        out.store(2 * t, a);
        out.store(2 * t + 1, b);
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap_err();
    assert!(e.message.contains("cvt with .ue8m0x2") && e.message.contains("sm_100a"), "{e}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    // FP8 conversions need sm_89.
    let fp8 = Kernel::build("fp8_on_sm80", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let out = param_tensor::<F8E4M3FN>("out", Layout::col_major(32));
        let t = thread_idx().x;
        out.store(t, x.load(t).to::<F8E4M3FN>());
    });
    let e = cute::validate::check_target(fp8.ir(), Arch::Sm86).unwrap_err();
    assert!(e.message.contains("needs sm_89 or newer"), "{e}");
}

#[test]
fn int128_values_are_refused_where_there_is_no_lowering() {
    let printf_msg = build_panic(|| {
        Kernel::build("printf_i128", || {
            let x = Int128::constant(1 << 100);
            printf("%lld\n", &[x.into()]);
        });
    });
    assert!(
        printf_msg.contains("cannot take a 128-bit value") && printf_msg.contains("tests/diagnostics.rs"),
        "{printf_msg}"
    );
    let shuffle_msg = build_panic(|| {
        Kernel::build("shuffle_i128", || {
            let out = param_tensor::<i128>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, shuffle_sync(t.to::<i128>(), 0));
        });
    });
    assert!(shuffle_msg.contains("128-bit"), "{shuffle_msg}");
    let ptx_msg = build_panic(|| {
        Kernel::build("inline_ptx_i128", || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            let y: Int32 = InlinePtx::new("mov.b32 {$w0}, {$r0};").write::<i32>().read(t.to::<i128>()).emit();
            out.store(t, y);
        });
    });
    assert!(ptx_msg.contains("cannot take a 128-bit value"), "{ptx_msg}");
    let float_msg = build_panic(|| {
        Kernel::build("i128_to_f32", || {
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<i128>().to::<f32>());
        });
    });
    assert!(float_msg.contains("convert through Int64"), "{float_msg}");
}

/// A scalar access to a sub-byte tensor is refused when the kernel is
/// built, as CuTe DSL's `_check_can_dereference` refuses it.
#[test]
fn int4_scalar_access_is_refused() {
    let msg = build_panic(|| {
        Kernel::build("store_i4", || {
            let out = param_tensor::<I4>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<I4>());
        });
    });
    assert!(msg.contains("a store of one Int4 element") && msg.contains("element by element"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("load_i4_ptr", || {
            let src = param_tensor::<I4>("src", Layout::col_major(32));
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            out.store(0, src.ptr().load().to::<i32>());
        });
    });
    assert!(msg.contains("a load of one Int4 element"), "{msg}");
    let popc = build_panic(|| {
        Kernel::build("popc_i4", || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            out.store(t, t.to::<I4>().popc().to::<i32>());
        });
    });
    assert!(popc.contains("cannot take an Int4 value"), "{popc}");
}

#[test]
fn int128_arithmetic_breaks_a_carry_chain() {
    let msg = build_panic(|| {
        Kernel::build("carry_around_i128", || {
            let out = param_tensor::<u32>("out", Layout::col_major(32));
            let wide = param_tensor::<i128>("wide", Layout::col_major(32));
            let t = thread_idx().x.to::<u32>();
            let lo = add_cc(t, 1u32);
            wide.store(0, wide.load(1) + wide.load(2));
            let hi = addc(t, 0u32);
            out.store(0, lo ^ hi);
        });
    });
    assert!(msg.contains("128-bit integer arithmetic"), "{msg}");
}

/// CuTe DSL's `MmaFP8Op` and `MmaF16BF16SparseOp` constructors pick the
/// instruction and reject what PTX lacks, with `MmaF16BF16Op`'s wording.
#[test]
fn fp8_and_sparse_mma_constructors() {
    let fp8 = |ab, acc, shape| MmaFP8Op::new(ab, acc, shape).kind();
    assert_eq!(fp8(DType::F8E4M3FN, DType::F32, (16, 8, 32)), MmaKind::M16N8K32F32E4M3);
    assert_eq!(fp8(DType::F8E5M2, DType::F16, (16, 8, 16)), MmaKind::M16N8K16F16E5M2);
    assert_eq!(MmaKind::M16N8K16F16E5M2.ptx(), "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16");
    let sp = |ab, acc, shape| MmaF16BF16SparseOp::new(ab, acc, shape, SparseMetadataFormat::Tid).kind();
    assert_eq!(sp(DType::BF16, DType::F32, (16, 8, 32)), MmaKind::SpM16N8K32F32BF16);
    assert_eq!(sp(DType::F16, DType::F16, (16, 8, 16)), MmaKind::SpM16N8K16F16F16);
    assert_eq!(
        MmaKind::SpM16N8K32F32BF16.ptx(),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
    );
    for (f, want) in [
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F16, DType::F32, (16, 8, 32));
            }) as Box<dyn FnOnce()>,
            "'ab_dtype'",
        ),
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F8E4M3FN, DType::BF16, (16, 8, 32));
            }),
            "'acc_dtype'",
        ),
        (
            Box::new(|| {
                let _ = MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 8));
            }),
            "(16,8,16) and (16,8,32)",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::F8E4M3FN, DType::F32, (16, 8, 16));
            }),
            "'ab_dtype'",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::BF16, DType::F16, (16, 8, 16));
            }),
            "Float32 only",
        ),
        (
            Box::new(|| {
                let _ = sp(DType::F16, DType::F32, (16, 8, 8));
            }),
            "'shape_mnk'",
        ),
    ] {
        let msg = build_panic(f);
        assert!(msg.contains(want), "{msg}");
    }
    // A sparse MMA goes through gemm_sparse, a dense one through gemm.
    let msg = build_panic(|| {
        Kernel::build("dense_gemm_on_sparse", || {
            let mma = make_tiled_mma(
                MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
                Layout::col_major((1, 1, 1)),
            );
            let (a, b, c) =
                (make_fragment::<F16>((4, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
            gemm(&mma, &c, &a, &b, &c);
        });
    });
    assert!(msg.contains("call gemm_sparse"), "{msg}");
    // FP8 MMAs need sm_89.
    let kernel = Kernel::build("fp8_mma_on_sm80", || {
        let mma = make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((1, 1, 1)));
        let (a, b, c) = (
            make_fragment::<F8E4M3FN>((16, 1, 1)),
            make_fragment::<F8E4M3FN>((8, 1, 1)),
            make_fragment::<f32>((4, 1, 1)),
        );
        a.fill(F8E4M3FN::from_f32(1.0));
        b.fill(F8E4M3FN::from_f32(1.0));
        c.fill(0.0);
        gemm(&mma, &c, &a, &b, &c);
    });
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm80).unwrap_err();
    assert!(e.to_string().contains("sm_89") && e.to_string().contains("FP8 mma.sync"), "{e}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm89).unwrap(), 84);
}

/// `LdMatrix8x16x8bOp` takes 8-bit elements, unpacks only 4 or 6 bits, and
/// needs K-major (row-contiguous) operands: an M-major FP8 A in shared
/// memory cannot feed it.
#[test]
fn ldmatrix_8x16x8b_misuse() {
    let msg = build_panic(|| drop(make_copy_atom::<F16>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None })));
    assert!(msg.contains("8-bit elements"), "{msg}");
    let msg = build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: Some(5) })));
    assert!(
        msg.contains("LdMatrix8x16x8bOp unpacks 4- or 6-bit elements") && msg.contains("not unpack_bits=5"),
        "{msg}"
    );
    let msg = build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix8x16x8b { num: 3, unpack_bits: Some(4) })));
    assert!(msg.contains("1, 2 or 4 matrices, not 3"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("m_major_fp8_a").block_dim([32, 1, 1]).build(|| {
            let sa = make_smem_tensor::<F8E4M3FN>(Layout::col_major((16, 32)));
            let mma =
                make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((1, 1, 1)));
            let ldsm = make_tiled_copy_a(
                &make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None }),
                &mma,
            );
            let t = thread_idx().x;
            let fa = mma.get_slice(t).partition_fragment_a::<F8E4M3FN, _>(&sa);
            copy(&ldsm, &ldsm.get_slice(t).partition_s(&sa), &ldsm.retile(&fa));
        });
    });
    assert!(msg.contains("contiguous source elements"), "{msg}");
}

/// A sparse MMA's metadata must name two increasing positions per chunk of
/// four; anything else (here `0b0000`) is undefined on the GPU and reported
/// by the simulator.
#[test]
fn invalid_sparse_metadata() {
    let kernel = Kernel::builder("bad_sparse_metadata").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(128));
        let op = MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid);
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let (a, b, c) =
            (make_fragment::<F16>((4, 1, 1)), make_fragment::<F16>((4, 1, 1)), make_fragment::<f32>((4, 1, 1)));
        let e = make_fragment::<u32>((1, 1, 1));
        a.fill(F16::ONE);
        b.fill(F16::ONE);
        c.fill(0.0);
        let lane = lane_idx();
        e.store(0, select(lane.eq(4), Uint32::from(0x4444_4440u32), Uint32::from(0x4444_4444u32)));
        gemm_sparse(&mma, &c, &a, &e, &b, &c);
        for i in 0..4usize {
            out.store(lane * 4 + i as i32, c.load(i));
        }
    });
    let mut args = vec![KernelArg::f32(vec![0.0; 128])];
    let e = simulate_err(&kernel, 32, &mut args);
    assert!(e.message.contains("nibble 0 of lane 4") && e.message.contains("0b0000"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
}

/// Sub-byte tensors: shared memory in whole bytes, vectors of whole 32-bit
/// words, recasts whose unit-stride mode divides evenly, and two warps
/// writing different nibbles of one byte (whole words on hardware) race.
#[test]
fn sub_byte_tensor_misuse() {
    let msg = build_panic(|| {
        Kernel::build("odd_int4_smem", || {
            let _ = make_smem_tensor::<I4>(Layout::col_major(15));
        });
    });
    assert!(msg.contains("15 Int4 values is 60 bits, not a whole number of bytes"), "{msg}");
    let msg = build_panic(|| drop(make_copy_atom::<I4>(CopyOp::Universal { bits: 16 })));
    assert!(msg.contains("multiple of 32 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("int4_vector_of_16_bits", || {
            let src = param_tensor::<I4>("src", Layout::col_major(64));
            let _: [Int4; 4] = load_vector(&src.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("4 Int4 values is 16 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("int4_autovec_strided", || {
            let src = param_tensor::<I4>("src", Layout::new(8, 2));
            let frag = make_fragment::<I4>(8);
            autovec_copy(&src, &frag);
        });
    });
    assert!(msg.contains("whole 32-bit words"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("recast_uneven", || {
            let src = param_tensor::<I4>("src", Layout::row_major((4, 3)));
            let _ = recast_tensor::<u8, _>(&src);
        });
    });
    assert!(msg.contains("unit-stride mode") && msg.contains("extent 3"), "{msg}");

    // Warp 0 and warp 1 each store a word of eight Int4 values to the same
    // shared word, meaning to change one nibble each.
    let kernel = Kernel::builder("nibble_race").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<I4>("out", Layout::col_major(8));
        let s = make_smem_tensor::<I4>(Layout::col_major(8));
        let t = thread_idx().x;
        for (thread, nibble, value) in [(0, 0, 3), (32, 1, 5)] {
            if_(t.eq(thread), || {
                let frag = make_fragment::<I4>(8);
                frag.fill(I4::ZERO);
                frag.store(nibble, I4::new(value));
                autovec_copy(&frag, &s);
            });
        }
        sync_threads();
        if_(t.eq(0), || {
            let back = make_fragment::<I4>(8);
            autovec_copy(&s, &back);
            autovec_copy(&back, &out);
        });
    });
    let mut args = vec![KernelArg::buffer(&[I4::ZERO; 8])];
    let e = simulate_err(&kernel, 64, &mut args);
    assert!(e.message.contains("race"), "{e}");
}

/// FP4 and FP6: no scalar `to()` into them (CuTe DSL refuses them as
/// destinations), no packed view as a conversion operand, their `cvt`s need
/// sm_100a, sm_101a or sm_120a, and an FP6 vector is a whole number of
/// 16-value (96-bit) groups.
#[test]
fn fp4_fp6_misuse() {
    let msg = build_panic(|| {
        Kernel::build("to_fp4", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let out = param_tensor::<F4E2M1FN>("out", Layout::col_major(8));
            let frag = make_fragment::<F4E2M1FN>(8);
            frag.fill(x.load(0).to::<F4E2M1FN>());
            autovec_copy(&frag, &out);
        });
    });
    assert!(msg.contains("to() destination") && msg.contains("cvt_f32x2_to_f4e2m1x2"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("to_fp6", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let out = param_tensor::<F6E3M2FN>("out", Layout::col_major(16));
            let frag = make_fragment::<F6E3M2FN>(16);
            frag.fill(x.load(0).to::<F6E3M2FN>());
            autovec_copy(&frag, &out);
        });
    });
    assert!(msg.contains("Float6E3M2FN") && msg.contains("cvt_f32x2_to_f6x2"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("view_to_f32", || {
            let x = param_tensor::<F4E2M1FNx2>("x", Layout::col_major(32));
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            out.store(0, x.load(0).to::<f32>());
        });
    });
    assert!(msg.contains("packed view") && msg.contains("Float4E2M1FN"), "{msg}");
    let kernel = Kernel::build("fp4_cvt", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        let out = param_tensor::<F16>("out", Layout::col_major(32));
        let [a, b] = cvt_f32x2_to_f4e2m1x2([x.load(0), x.load(1)], false);
        let (lo, _): (Float16, Float16) = cvt_f4e2m1x2_to_f16x2([a, b]);
        out.store(0, lo);
    });
    for arch in [Arch::Sm90, Arch::Sm90a, Arch::Sm100] {
        let e = cute::validate::check_target(kernel.ir(), arch).unwrap_err().to_string();
        assert!(e.contains(".e2m1x2") && e.contains("sm_100a"), "{e}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm120a).unwrap(), 87);
    let msg = build_panic(|| {
        Kernel::build("fp6_by_eights", || {
            let src = param_tensor::<F6E2M3FN>("src", Layout::col_major(64));
            let frag = make_fragment::<F6E2M3FN>(8);
            copy(&make_copy_atom::<F6E2M3FN>(CopyOp::Universal { bits: 48 }), &local_tile(&src, 8, 0), &frag);
        });
    });
    assert!(msg.contains("multiple of 32 bits"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("fp6_misaligned", || {
            let src = param_tensor::<F6E2M3FN>("src", Layout::col_major(64));
            let _: [Float6E2M3FN; 16] = load_vector(&src.ptr().offset(8usize), LoadOptions::default());
        });
    });
    assert!(msg.contains("aligned to 16 elements") || msg.contains("whole number of 32-bit words"), "{msg}");
}

/// The tensor metadata and `autovec_copy` refuse what CuTe DSL refuses:
/// the stride of a swizzled tensor, the address space of a coordinate
/// tensor, a copy between tensors of different sizes, and an eviction
/// priority outside a copy between global memory and registers.
#[test]
fn tensor_metadata_and_autovec_misuse() {
    let msg = build_panic(|| {
        Kernel::build("swizzled_stride", || {
            let s = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(3, 2, 3), 0, Layout::row_major((8, 32))));
            let _ = s.stride();
        });
    });
    assert!(msg.contains("Tensor::stride: tensor (8,32):(32,1) is swizzled by S<3,2,3>"), "{msg}");
    assert!(msg.contains("use tensor.layout().stride()"), "{msg}");
    let msg = build_panic(|| {
        let _ = make_identity_tensor((4, 8)).memspace();
    });
    assert!(msg.contains("Tensor::memspace: tensor (4,8):(1@0,1@1) is a coordinate tensor"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("autovec_sizes", || {
            let x = param_tensor::<f32>("x", Layout::row_major((4, 8)));
            autovec_copy(&x, &make_fragment::<f32>((4, 4)));
        });
    });
    assert!(
        msg.contains(
            "autovec_copy: tensors have different sizes: the source (4,8):(8,1) has 32 elements and the \
                      destination (4,4):(1,4) has 16"
        ),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("autovec_evict_smem", || {
            let x = param_tensor::<f32>("x", Layout::col_major(32));
            let s = make_smem_tensor::<f32>(Layout::col_major(32));
            let opts = AutovecOptions { l1c_evict_priority: CacheEvictionPriority::EvictFirst };
            autovec_copy_with(&x, &s, opts);
        });
    });
    assert!(
        msg.contains("autovec_copy_with: an L1 eviction priority (evict_first) is for a copy between global"),
        "{msg}"
    );
    assert!(msg.contains("this copy is from gmem to smem"), "{msg}");
}

/// A 32-byte vector needs a 32-byte-aligned global buffer of 32- or 64-bit
/// elements: the validator refuses one to shared memory, of 16-bit
/// elements, or from a buffer declared 16-byte aligned.
#[test]
fn wide_vector_misuse() {
    let msg = build_panic(|| {
        Kernel::build("wide_smem", || {
            let s = make_smem_tensor::<f32>(Layout::col_major(64));
            let _: [Float32; 8] = load_vector(&s.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("vector access of 8 x f32 is not a power-of-two width of at most 16 bytes (32-byte"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_f16", || {
            let x = param_tensor_aligned::<F16>("x", Layout::col_major(64), 32);
            let _: [Float16; 16] = load_vector(&x.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("vector access of 16 x f16 is not a power-of-two width of at most 16 bytes"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_16", || {
            let x = param_tensor::<f32>("x", Layout::col_major(64));
            let _: [Float32; 8] = load_vector(&x.ptr(), LoadOptions::default());
        });
    });
    assert!(msg.contains("a load of 32 bytes in `x`, which is only declared 16-byte aligned"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide_param", || {
            let _ = param_tensor_aligned::<f32>("x", Layout::col_major(64), 24);
        });
    });
    assert!(msg.contains("param_tensor_aligned: an alignment of 24 bytes is not a power of two of at least 16"));
}

/// What each bulk-store diagnostic kernel gets wrong (see [`bulk_store_bug`]).
#[derive(Clone, Copy, PartialEq)]
enum BulkBug {
    /// Generic writes, a barrier and the copy, without a proxy fence.
    NoProxyFence,
    /// Warp 1 fences after the barrier, so nothing orders its fence before
    /// warp 0's copy.
    UnorderedFence,
    /// The source is overwritten without `cp_async_bulk_wait_group(0, true)`.
    NoReadWait,
    /// The destination is read after the `.read` wait, before the plain one.
    EarlyGlobalRead,
    /// Thread 0 waits for the source to be read, but warp 1 overwrites it
    /// with no barrier ordering it after that wait.
    ReadWaitNoBarrier,
    /// A runtime size of 24 bytes.
    RuntimeSize,
    /// A runtime source offset of 4 elements: 16 bytes past an aligned start
    /// of an f32 buffer is fine, so the offset is 2 elements (8 bytes).
    RuntimeMisaligned,
}

/// 64 threads write a 64-word shared tile; thread 0 stores it to `out` with
/// a bulk copy, as [`BulkBug`] says, wrongly.
fn bulk_store_bug(bug: BulkBug) -> Kernel {
    Kernel::builder("bulk_store_bug").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<f32>("out", Layout::col_major(64));
        let n = param_scalar::<i32>("n");
        let smem = alloc_smem::<f32>(64, 128);
        let t = thread_idx().x;
        let tile = make_tensor(&smem, Layout::col_major(64));
        if bug != BulkBug::UnorderedFence {
            tile.store(t, t.to::<f32>());
        } else {
            if_(warp_idx().eq(1), || tile.store(t, t.to::<f32>()));
        }
        if !matches!(bug, BulkBug::NoProxyFence | BulkBug::UnorderedFence) {
            fence_view_async_shared();
        }
        sync_threads();
        if bug == BulkBug::UnorderedFence {
            if_(warp_idx().eq(1), fence_view_async_shared);
        }
        if_(t.eq(0), || {
            let (src, bytes) = match bug {
                BulkBug::RuntimeSize => (smem.clone(), Int32::from(n)),
                BulkBug::RuntimeMisaligned => (smem.offset(n), Int32::from(64)),
                _ => (smem.clone(), Int32::from(256)),
            };
            cp_async_bulk_global_shared_cta(&out.ptr(), &src, bytes);
            cp_async_bulk_commit_group();
            if bug != BulkBug::NoReadWait {
                cp_async_bulk_wait_group(0, true);
            }
            if bug == BulkBug::EarlyGlobalRead {
                tile.store(0, out.load(0));
            }
        });
        if bug != BulkBug::ReadWaitNoBarrier {
            sync_threads();
        }
        tile.store(t, 0.0f32);
        if_(t.eq(0), || cp_async_bulk_wait_group(0, false));
    })
}

#[test]
fn bulk_store_misuse_in_the_simulator() {
    let run = |bug, n: i32| {
        let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::ScalarI32(n)];
        bulk_store_bug(bug).simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).err()
    };
    // The kernel as written for the runtime cases is correct with the right values.
    assert!(run(BulkBug::RuntimeSize, 256).is_none());
    let e = run(BulkBug::NoProxyFence, 0).expect("a missing proxy fence");
    assert!(e.message.contains("cp.async.bulk.global.shared::cta reads shared memory through the async proxy"), "{e}");
    assert!(e.message.contains("through the generic proxy, without a fence.proxy.async since"), "{e}");
    assert!(e.to_string().contains("tests/diagnostics.rs"), "{e}");
    let e = run(BulkBug::UnorderedFence, 0).expect("an unordered fence");
    assert!(e.message.contains("nothing orders the fence before this operation's issue"), "{e}");
    let e = run(BulkBug::NoReadWait, 0).expect("a source overwritten too early");
    assert!(e.message.contains("data race: warp"), "{e}");
    assert!(e.message.contains("which an asynchronous operation issued by"), "{e}");
    let e = run(BulkBug::EarlyGlobalRead, 0).expect("a destination read too early");
    assert!(e.message.contains("reads element 0 of `out`, which a cp.async.bulk issued by thread 0"), "{e}");
    // Warp 1's overwrite must be ordered after thread 0's `.read` wait,
    // whichever order the warps run in and whenever the copy completes.
    for schedule in Schedule::ALL {
        for async_completion in [AsyncCompletion::Late, AsyncCompletion::Early] {
            let opts = SimOptions { schedule, async_completion, ..SimOptions::default() };
            let mut args = vec![KernelArg::f32(vec![0.0; 64]), KernelArg::ScalarI32(0)];
            let e = bulk_store_bug(BulkBug::ReadWaitNoBarrier)
                .simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts)
                .expect_err("an overwrite not ordered after the .read wait");
            assert!(e.message.contains("data race"), "{schedule:?} {async_completion:?}: {e}");
        }
    }
    let e = run(BulkBug::RuntimeSize, 24).expect("a size of 24 bytes");
    assert!(e.message.contains("cp.async.bulk of 24 bytes is not one or more whole 16-byte units"), "{e}");
    let e = run(BulkBug::RuntimeMisaligned, 2).expect("an 8-byte offset");
    assert!(
        e.message.contains("misaligned cp.async.bulk source read: element 2 of `smem0` is not 16-byte aligned"),
        "{e}"
    );
}

#[test]
fn bulk_store_misuse_when_built() {
    let msg = build_panic(|| {
        Kernel::build("bulk_size", || {
            let out = param_tensor::<f32>("out", Layout::col_major(64));
            let smem = alloc_smem::<f32>(64, 128);
            cp_async_bulk_global_shared_cta(&out.ptr(), &smem, 24);
        });
    });
    assert!(
        msg.contains("cp.async.bulk.global.shared::cta of 24 bytes: bulk copies move one or more whole 16-byte units"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("bulk_offset", || {
            let out = param_tensor::<f32>("out", Layout::col_major(64));
            let smem = alloc_smem::<f32>(64, 128);
            cp_async_bulk_global_shared_cta(&out.ptr().offset(2), &smem, 32);
        });
    });
    assert!(msg.contains("the destination is at byte 8 of its buffer, which is not 16-byte aligned"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_atom_size", || {
            let out = param_tensor::<f32>("out", Layout::col_major(6));
            let smem = make_smem_tensor::<f32>(Layout::col_major(6));
            copy(&make_copy_atom::<f32>(cpasync::CopyBulkS2GOp), &smem, &out);
        });
    });
    assert!(msg.contains("copy: a CopyBulkS2GOp copy of 6 f32 elements is 24 bytes, not a multiple of 16"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_tensormap", || fence_proxy(FenceProxyKind::Tensormap));
    });
    assert!(msg.contains("use fence_tma_desc_acquire or fence_tma_desc_release"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_space", || fence_proxy(FenceProxyKind::AsyncShared));
    });
    assert!(msg.contains("the AsyncShared kind needs a space"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("bulk_wait", || cp_async_bulk_wait_group(param_scalar::<i32>("n"), false));
    });
    assert!(msg.contains("cp_async_bulk_wait_group: the group count is a runtime value"), "{msg}");
}

/// Threads initialise a shared stage with ordinary stores that a bulk copy
/// then overwrites through the async proxy: without a proxy fence between
/// them the simulator reports it; with `fence_view_async_shared` the kernel
/// runs.
#[test]
fn bulk_load_over_generic_writes_needs_a_proxy_fence() {
    let kernel = |fenced: bool| {
        Kernel::builder("bulk_load_overwrite").block_dim([32, 1, 1]).build(move || {
            let src = param_tensor::<f32>("src", Layout::col_major(32));
            let out = param_tensor::<f32>("out", Layout::col_major(32));
            let (stage, bar) = (alloc_smem::<f32>(32, 128), alloc_smem::<u64>(1, 8));
            let t = thread_idx().x;
            make_tensor(&stage, Layout::col_major(32)).store(t, 0.0f32);
            if fenced {
                fence_view_async_shared();
            }
            if_(t.eq(0), || mbarrier_init(&bar, 1));
            mbarrier_init_fence();
            sync_threads();
            if_(t.eq(0), || {
                mbarrier_arrive_and_expect_tx(&bar, 128);
                cp_async_bulk(&src.ptr(), &stage, 128, &bar);
            });
            mbarrier_wait(&bar, 0);
            out.store(t, make_tensor(&stage, Layout::col_major(32)).load(t));
        })
    };
    let mut args = vec![KernelArg::f32((0..32).map(|i| i as f32).collect()), KernelArg::f32(vec![0.0; 32])];
    let e = kernel(false).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("cp.async.bulk overwrites shared memory through the async proxy"), "{e}");
    assert!(e.message.contains("without a fence.proxy.async since"), "{e}");
    kernel(true).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].as_f32()[31], 31.0);
}

/// The 8-bit `ldmatrix` atoms of CuTe DSL's `LdMatrix16x16x8bOp` and
/// `LdMatrix16x8x8bOp` must be transposed and take 1 or 2 (2 or 4) matrices
/// and 4, 6 or no unpacked bits, as CuTe DSL's constructors check; the
/// `.m16n16` and `.m8n16` shapes need an sm_100a-class target (not sm_90a
/// or plain sm_100), compiled at the PTX ISA version of that target.
#[test]
fn ldmatrix_b8_misuse() {
    let msg = build_panic(|| {
        drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 1, trans: false, unpack_bits: None }))
    });
    assert!(msg.contains("LdMatrix16x16x8bOp must be transposed"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 4, trans: true, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x16x8bOp loads 1 or 2 matrices, not 4"), "{msg}");
    let msg = build_panic(|| {
        drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 2, trans: true, unpack_bits: Some(8) }))
    });
    assert!(msg.contains("LdMatrix16x16x8bOp unpacks") && msg.contains("not unpack_bits=8"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x8x8b { num: 2, trans: false, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x8x8bOp must be transposed"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<u8>(CopyOp::LdMatrix16x8x8b { num: 1, trans: true, unpack_bits: None })));
    assert!(msg.contains("LdMatrix16x8x8bOp loads 2 or 4 matrices, not 1"), "{msg}");
    let msg =
        build_panic(|| drop(make_copy_atom::<F16>(CopyOp::LdMatrix16x8x8b { num: 2, trans: true, unpack_bits: None })));
    assert!(msg.contains("8-bit elements"), "{msg}");
    let kernel = Kernel::builder("ldmatrix_m16n16_on_sm90").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<u8>("out", Layout::row_major((32, 8)));
        let smem = make_smem_tensor::<u8>(Layout::col_major((16, 16)));
        let atom = make_copy_atom::<u8>(CopyOp::LdMatrix16x16x8b { num: 1, trans: true, unpack_bits: None });
        let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (16, 16));
        let t = thread_idx().x;
        let frag = make_fragment::<u8>(8);
        copy(&tiled, &tiled.get_slice(t).partition_s(&smem), &tiled.retile(&frag));
        for v in 0..8 {
            out.store((t, v), frag.load(v));
        }
    });
    for arch in [Arch::Sm90a, Arch::Sm100] {
        let e = cute::validate::check_target(kernel.ir(), arch).unwrap_err().to_string();
        assert!(e.contains("ldmatrix.m16n16 with .b8") && e.contains("needs sm_100a"), "{e}");
        assert!(e.contains(&format!("compiled for {}", arch.chip())), "{e}");
    }
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm120a).unwrap(), 87);
    assert_eq!(kernel.min_llvm_version(), Some(22));
}

/// Pointer swizzles (`recast_ptr`): one with a swizzled composed layout, a
/// buffer not aligned to the swizzle's period, and a byte swizzle finer than
/// the element are refused at the user's line.
#[test]
fn pointer_swizzle_misuse() {
    let msg = build_panic(|| {
        Kernel::build("two_swizzles", || {
            let p = recast_ptr::<F16, _>(&alloc_smem::<F16>(512, 1024), Some(make_swizzle(3, 4, 3)));
            let _ = make_tensor(&p, make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64))));
        });
    });
    assert!(
        msg.contains("make_tensor: the pointer is swizzled by S<3,4,3> and the layout S<3,3,3> o 0 o (8,64):(64,1)")
            && msg.contains("get_nonswizzle_portion"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("unaligned_swizzle_base", || {
            let _ = recast_ptr::<f32, _>(&param_ptr::<f32>("x"), Some(make_swizzle(3, 4, 3)));
        });
    });
    assert!(
        msg.contains("the pointer swizzle S<3,4,3> acts on absolute byte addresses")
            && msg.contains("period of 1024 bytes")
            && msg.contains("only known to be 16-byte aligned"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("fine_swizzle", || {
            let _ = recast_ptr::<f32, _>(&alloc_smem::<f32>(256, 1024), Some(make_swizzle(3, 1, 3)));
        });
    });
    assert!(
        msg.contains("a pointer to Float32 swizzled by S<3,1,3> (on bytes)")
            && msg.contains("finer than the 4 bytes of an element"),
        "{msg}"
    );
    // A swizzled composed layout on a pointer that is not a whole number of
    // periods into its buffer.
    let msg = build_panic(|| {
        Kernel::build("swizzle_off_period", || {
            let p = alloc_smem::<f32>(1024, 128).offset(4);
            let _ = make_tensor(&p, make_composed_layout(make_swizzle(2, 2, 3), 0, Layout::row_major((8, 32))));
        });
    });
    assert!(msg.contains("which cute-rs needs to be a multiple of 128 elements"), "{msg}");
}

/// A kernel built for a target (`KernelOptions::target`) is checked when the
/// build ends, naming the line of the instruction the target lacks; it
/// compiles for that target by default and refuses another; the body reads
/// the target with `target_arch`.
#[test]
fn build_time_targets() {
    use cute::validate::check_target;
    let fp4 = |arch: Arch| {
        Kernel::builder("fp4_pair").block_dim([32, 1, 1]).target(arch).build(|| {
            let out = param_tensor::<F4E2M1FN>("out", Layout::col_major(64));
            assert_eq!(target_arch(), Some(arch));
            let t = thread_idx().x;
            let x = t.to::<f32>();
            let line = line!() + 1;
            let [p, q] = cvt_f32x2_to_f4e2m1x2([x, -x], false);
            let pair = make_fragment::<F4E2M1FN>(8);
            for i in 0..4 {
                pair.store(2 * i, p);
                pair.store(2 * i + 1, q);
            }
            autovec_copy(&pair, &local_tile(&out, 8, t));
            let _ = line;
        })
    };
    // The FP4 conversion: sm_100a or sm_101a (PTX 8.6), sm_110a (9.0),
    // sm_120a (8.7), and the sm_100f, sm_110f and sm_120f families (8.8).
    let msg = build_panic(|| {
        fp4(Arch::Sm90a);
    });
    assert!(
        msg.contains("kernel `fp4_pair` is built for sm_90a (KernelOptions::target), which it cannot run on")
            && msg.contains("tests/diagnostics.rs:")
            && msg.contains("needs sm_100a, sm_101a, sm_110a or sm_120a (or the sm_100f family with PTX 8.8"),
        "{msg}"
    );
    for (arch, ptx) in
        [(Arch::Sm100a, 86), (Arch::Sm103f, 88), (Arch::Sm110a, 90), (Arch::Sm121a, 88), (Arch::Sm120a, 87)]
    {
        let k = fp4(arch);
        assert_eq!((k.target(), check_target(k.ir(), arch).unwrap()), (Some(arch), ptx), "{arch}");
    }
    for arch in [Arch::Sm100, Arch::Sm103, Arch::Sm107, Arch::Sm120] {
        assert!(build_panic(|| drop(fp4(arch))).contains("which it cannot run on"), "{arch}");
    }
    // Compiling: the kernel's target by default, another one refused.
    if let Some(tc) = Toolchain::discover() {
        let k = fp4(Arch::Sm100a);
        let e = k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm120a)).unwrap_err().to_string();
        assert!(
            e.contains("was built for sm_100a (KernelOptions::target), so it cannot be compiled for sm_120a"),
            "{e}"
        );
        let compiled = k.compile_with(&tc, &CompileOptions::new()).unwrap();
        assert_eq!((compiled.arch(), compiled.ptx_version()), (Arch::Sm100a, 86));
        // Without a target, nothing names one.
        let e = Kernel::build("untargeted", || {}).compile_with(&tc, &CompileOptions::new()).unwrap_err();
        assert!(e.to_string().contains("KernelOptions::target"), "{e}");
        // Rubin: no LLVM emits sm_107a yet, so compiling fails clearly, though
        // the kernel builds and checks for it.
        let k = Kernel::builder("rubin").target(Arch::Sm107a).build(|| {});
        assert_eq!(check_target(k.ir(), Arch::Sm107a).unwrap(), 94);
        if tc.targets().is_some_and(|s| !s.has("sm_107a")) {
            let e = k.compile_with(&tc, &CompileOptions::new()).unwrap_err().to_string();
            assert!(e.contains("cannot emit sm_107a: its NVPTX backend") && e.contains("no such target"), "{e}");
        }
    }
    // Outside a target, target_arch reads CUTE_RS_ARCH (unset here).
    if std::env::var_os("CUTE_RS_ARCH").is_none() {
        Kernel::build("no_target", || assert_eq!(target_arch(), None));
    }
}

/// A kernel that needs more PTX than the toolchain emits fails to compile,
/// naming the version, the toolchain's newest and the instruction that
/// needs it, at its line.
#[test]
fn ptx_ceiling_of_the_toolchain() {
    let Some(tc) = Toolchain::discover() else { return };
    let Some(max) = tc.targets().map(|s| s.max_ptx) else { return };
    let line = line!() + 4;
    let kernel = Kernel::build("s2f6", || {
        let out = param_tensor::<BF16>("out", Layout::col_major(64));
        let t = thread_idx().x;
        let v = cvt_i8_bf16_intrinsic_with(&[t.to::<i8>(), (t + 1).to::<i8>()], 2, I8ToBf16Path::S2f6x2);
        out.store(2 * t, v[0]);
        out.store(2 * t + 1, v[1]);
    });
    let result = kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a));
    if max >= 91 {
        assert!(result.is_ok(), "{:?}", result.err());
        return;
    }
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("needs PTX ISA 9.1 on sm_100a")
            && e.contains(&format!("emits PTX ISA {}.{} at most", max / 10, max % 10))
            && e.contains(&format!("tests/diagnostics.rs:{line}:"))
            && e.contains("s2f6x2"),
        "{e}"
    );
}

cute_struct! {
    struct EmptyStruct {}
}

cute_union! {
    struct EmptyUnion {}
}

cute_struct! {
    struct OddAlign { #[align(48)] x: f32 }
}

cute_struct! {
    struct HugeAlign { #[align(2048)] x: MemRange<f32>[4] }
}

cute_struct! {
    struct Ranges { head: u32, r: MemRange<f32>[4], marker: MemRange<u64>[0], #[align(16)] h: MemRange<F16>[512] }
}

/// The shared-storage struct mistakes CuTe DSL refuses, each refused when
/// the kernel is built with the member and the rule.
#[test]
fn struct_misuse() {
    let e = build_panic(|| {
        EmptyStruct::layout();
    });
    assert!(e.contains("struct EmptyStruct has no members"), "{e}");
    let e = build_panic(|| {
        EmptyUnion::layout();
    });
    assert!(e.contains("union EmptyUnion has no members"), "{e}");
    let e = build_panic(|| {
        OddAlign::layout();
    });
    assert!(e.contains("member `x` of OddAlign: an alignment of 48 bytes is not a power of two"), "{e}");
    let e = build_panic(|| {
        HugeAlign::layout();
    });
    assert!(e.contains("an alignment of 2048 bytes is more than shared memory's 1024"), "{e}");
    let e = build_panic(|| {
        StructLayout::builder("Nibbles").mem_range::<I4>("odd", 3, None);
    });
    assert!(e.contains("MemRange[Int4, 3] is 12 bits, not a whole number of bytes"), "{e}");
    let e = build_panic(|| {
        let mut b = StructLayout::builder("Twice");
        b.scalar::<f32>("x", None);
        b.scalar::<i32>("x", None);
    });
    assert!(e.contains("Twice has two members named `x`"), "{e}");

    let in_kernel = |f: fn(&Ranges)| {
        build_panic(move || {
            let _ = Kernel::builder("struct_misuse").block_dim([32, 1, 1]).build(|| f(&Ranges::allocate()));
        })
    };
    let e = in_kernel(|s| {
        s.marker.get_tensor(Layout::col_major(1));
    });
    assert!(e.contains("get_tensor on MemRange `marker` of length 0"), "{e}");
    let e = in_kernel(|s| {
        s.r.store(4, 1.0f32);
    });
    assert!(e.contains("a store of element 4 of MemRange `r`, which has 4 elements"), "{e}");
    let e = in_kernel(|s| {
        s.r.get_tensor(Layout::col_major((2, 3)));
    });
    assert!(e.contains("get_tensor on MemRange `r`: the layout needs 6 elements, but the range has 4"), "{e}");
    let e = in_kernel(|s| {
        s.h.get_tensor_composed(make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((16, 64))));
    });
    assert!(
        e.contains("get_tensor_composed on MemRange `h`: the layout needs 1024 elements, but the range has 512"),
        "{e}"
    );
    // A swizzled composed layout needs its range at a multiple of the swizzle's period.
    let e = in_kernel(|s| {
        s.h.get_tensor_composed(make_composed_layout(make_swizzle(3, 3, 3), 0, Layout::row_major((8, 64))));
    });
    assert!(e.contains("needs to be a multiple of 512 elements (the swizzle's period) into its buffer"), "{e}");
    let e = in_kernel(|s| {
        s.h.get_tensor_swizzled(Layout::row_major((8, 64)), make_swizzle(3, 4, 3));
    });
    assert!(e.contains("the range must start at a known multiple of its period of 1024 bytes"), "{e}");
    let e = build_panic(|| {
        let _ = Kernel::builder("struct_misuse").block_dim([32, 1, 1]).build(|| {
            let _ = Ranges::allocate_aligned(4096);
        });
    });
    assert!(e.contains("allocating Ranges: an alignment of 4096 bytes is more than shared memory's 1024"), "{e}");
}

// ---- m6-diagnostics: located errors, code frames, compiler diagnostics,
// ---- device assertions.

/// `{:#}` (and `render`) add a code frame of the failing line; plain
/// `Display` is unchanged.
#[test]
fn errors_carry_code_frames() {
    let k = Kernel::builder("framed").block_dim([1, 1, 1]).build(|| {
        let x = param_scalar::<i32>("x");
        cute::testing::assert_(x.lt(3), "x must be below 3");
    });
    let e = k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut [KernelArg::scalar(7i32)]).unwrap_err();
    let plain = e.to_string();
    let e = cute::Error::from(e);
    assert_eq!(e.to_string(), plain);
    let loc = e.loc().expect("a location");
    assert_eq!(loc.file, file!());
    let framed = format!("{e:#}");
    assert!(framed.starts_with(&plain), "{framed}");
    assert!(
        framed.contains("cute::testing::assert_(x.lt(3), \"x must be below 3\");") && framed.contains("^"),
        "{framed}"
    );
    assert_eq!(framed, e.render());
    // A file that cannot be read has no frame.
    let missing = cute::SourceLoc { file: "no/such/file.rs".into(), line: 1, column: 1 };
    assert!(missing.code_frame().is_none());
}

/// A misuse found inside the builder names the user's line, not cute-rs's.
#[test]
fn builder_misuse_names_the_users_line() {
    let line = line!() + 4;
    let msg = build_panic(|| {
        Kernel::build("dup_param", || {
            let _a = param_scalar::<i32>("a");
            let _b = param_scalar::<f32>("a");
        });
    });
    assert!(msg.starts_with(&format!("{}:{line}:", file!())), "{msg}");
    assert!(msg.contains("duplicate parameter name \"a\""), "{msg}");
}

/// An operation mlir-opt rejects: the error names the line that built it,
/// although the kernel is compiled without line information.
#[test]
fn mlir_opt_errors_name_the_line() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let line = line!() + 4;
    let k = Kernel::builder("bad_mlir").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<u32>("x");
        let smem = alloc_smem::<u32>(32, 16);
        MlirOp::new("nvvm.cp.async.shared.global %0, %1, 3, cache = ca : !llvm.ptr<3>, !llvm.ptr<1>")
            .operand_ptr(&smem)
            .operand_ptr(&x)
            .emit();
    });
    let e = k.compile_ptx(&tc, Arch::Sm80).unwrap_err();
    let loc = e.loc().unwrap_or_else(|| panic!("no location in {e}"));
    assert_eq!((loc.file.as_str(), loc.line), (file!(), line), "{e}");
    let rendered = cute::Error::from(e).render();
    assert!(rendered.contains("MlirOp::new(\"nvvm.cp.async"), "{rendered}");
}

#[test]
fn diagnostic_option_spellings() {
    let o = CompileOptions::parse("warnings{ptx} remarks").unwrap();
    assert_eq!(o.get_warnings(), Some(&["ptx".to_string()][..]));
    assert_eq!(o.get_remarks(), Some(&[][..]));
    let o = CompileOptions::parse("--warnings{nvvm,ptx} remarks{ptx} remarks{loop}").unwrap();
    assert_eq!(o.get_warnings().unwrap(), ["nvvm", "ptx"]);
    assert_eq!(o.get_remarks().unwrap(), ["ptx", "loop"]);
    let o = CompileOptions::parse("remarks{ptx} remarks").unwrap();
    assert_eq!(o.get_remarks(), Some(&[][..]));
    let o = CompileOptions::parse("--enable-assertions --remark-filter 'Memory|ptx' --remark-output r.yaml").unwrap();
    assert_eq!(o.get_enable_assertions(), Some(true));
    let e = CompileOptions::parse("remarks{bogus}").unwrap_err();
    assert!(e.to_string().contains("`bogus` is not a diagnostic category"), "{e}");
    let e = CompileOptions::parse("warnings=1").unwrap_err();
    assert!(e.to_string().contains("takes a braced category list"), "{e}");
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        CompileOptions::from_env_with(move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()))
    };
    let o = env(&[("CUTE_RS_ENABLE_ASSERTIONS", "1"), ("CUTE_RS_WARNINGS_AS_ERRORS", "yes")]).unwrap();
    assert_eq!(o.get_enable_assertions(), Some(true));
    assert!(format!("{o:?}").contains("warnings_as_errors: Some(true)"));
}

/// ptxas -v output and the JIT's verbose info log, as captured on an SM89
/// with CUDA 12.
#[test]
fn resource_usage_from_logs() {
    let ptxas = "\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'spill' for 'sm_89'
ptxas info    : Function properties for spill
    264 bytes stack frame, 260 bytes spill stores, 260 bytes spill loads
ptxas info    : Used 16 registers, used 1 barriers, 264 bytes cumulative stack size, 368 bytes cmem[0]
ptxas info    : Compiling entry function 'other' for 'sm_89'
ptxas info    : Function properties for other
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 2048 bytes smem, 360 bytes cmem[0], 8 bytes cmem[2]
";
    let u = ResourceUsage::from_ptxas_log(ptxas, "spill").unwrap();
    assert_eq!(
        (u.registers, u.stack_bytes, u.spill_store_bytes, u.spill_load_bytes, u.constant_bytes, u.shared_bytes),
        (Some(16), Some(264), Some(260), Some(260), Some(368), Some(0))
    );
    assert!(u.spills());
    let o = ResourceUsage::from_ptxas_log(ptxas, "other").unwrap();
    assert_eq!((o.registers, o.shared_bytes, o.constant_bytes, o.spills()), (Some(8), Some(2048), Some(368), false));
    assert!(ResourceUsage::from_ptxas_log(ptxas, "missing").is_none());
    let jit = "\
ptxas warning : For profile sm_89 adjusting per thread register count of 16 to lower bound of 24
ptxas info    : Overriding maximum register limit 256 for 'spill' with  24 of maxrregcount option
ptxas info    : 0 bytes gmem
ptxas info    : Function properties for spill
ptxas         .     264 bytes stack frame, 260 bytes spill stores, 260 bytes spill loads
ptxas info    : Used 16 registers, used 1 barriers, 264 bytes cumulative stack size, 368 bytes cmem[0]
ptxas info    : Compile time = 5.113 ms
";
    assert_eq!(ResourceUsage::from_jit_log(jit, "spill"), Some(u));
}

/// Simulated assertions: a passing one is silent, a failing one stops the
/// run naming its line, block, thread and message; off, it is a warning.
#[test]
fn simulated_assertions() {
    let line = line!() + 4;
    let k = Kernel::builder("asserts").block_dim([4, 1, 1]).build(|| {
        let t = thread_idx().x;
        cute::testing::assert_(t.lt(4), "every thread is below 4");
        cute::testing::assert_(t.ne(2), "thread 2 is not welcome");
    });
    let cfg = LaunchConfig::new([1, 1, 1], [4, 1, 1]);
    let e = k.simulate(cfg, &mut []).unwrap_err();
    assert_eq!(e.loc.map(|l| (l.file(), l.line())), Some((file!(), line)), "{e}");
    assert_eq!(e.thread, Some([2, 0, 0]));
    assert!(e.message.contains("device assertion failed: thread 2 is not welcome"), "{e}");
    let opts = SimOptions { assertions: false, ..SimOptions::default() };
    let report = k.simulate_with(cfg, &mut [], &opts).unwrap();
    assert_eq!(report.diagnostics.len(), 1, "{:?}", report.diagnostics);
    assert!(report.diagnostics[0].message.contains("thread 2 is not welcome"));
    // Validation: the condition is a Boolean.
    let ok = Kernel::build("assert_ok", || cute::testing::assert_(true, "fine"));
    ok.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut []).unwrap();
}

/// cf.assert is in the MLIR only with assertions enabled, with a plain
/// location even with line information, and the PTX then calls
/// __assertfail.
#[test]
fn assertions_compile_only_when_enabled() {
    let k = Kernel::builder("asserting").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let t = thread_idx().x;
        cute::testing::assert_(x.load_at(t).ge(0), "negative input");
    });
    assert!(!k.to_mlir().contains("cf.assert"));
    let on = MlirOptions { assertions: true, line_info: true, locations: true, ..Default::default() };
    let mlir = k.to_mlir_with(&on);
    let line = mlir.lines().find(|l| l.contains("cf.assert")).expect("cf.assert");
    assert!(line.contains("\"negative input\" loc(\"tests/diagnostics.rs\":"), "{line}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let off = k.compile_ptx(&tc, Arch::Sm80).unwrap();
    assert!(!off.contains("__assertfail"));
    for line_info in [false, true] {
        let opts = CompileOptions::new()
            .gpu_arch(Arch::Sm80)
            .use_ptxas(false)
            .enable_assertions(true)
            .generate_line_info(line_info);
        let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).into_ptx();
        assert!(ptx.contains("__assertfail"), "{ptx}");
        // The file name is a byte array in the PTX.
        let bytes: Vec<String> = file!().bytes().map(|b| b.to_string()).collect();
        assert!(ptx.contains(&bytes.join(", ")), "the assertion's file is not in the PTX:\n{ptx}");
    }
}

/// An assertion under divergent control flow (a runtime `if_` and a
/// runtime loop) stays inside that region in the MLIR, lowers, and is
/// checked by the simulator only for the threads that reach it.
#[test]
fn assertions_under_divergence() {
    let k = Kernel::builder("divergent_assert").block_dim([32, 1, 1]).build(|| {
        let x = param_ptr::<i32>("x");
        let n = param_scalar::<i32>("n");
        let t = thread_idx().x;
        if_(t.lt(4), || cute::testing::assert_(x.load_at(t).ge(0), "low threads read non-negative values"));
        for_range(0, n, 1, |i| cute::testing::assert_(x.load_at(i).ne(99), "no 99 in the first n"));
    });
    let mlir = k.to_mlir_with(&MlirOptions { assertions: true, ..Default::default() });
    // Each cf.assert is nested deeper than the function body: inside the
    // scf.if and the scf.for.
    let asserts: Vec<&str> = mlir.lines().filter(|l| l.contains("cf.assert")).collect();
    assert_eq!(asserts.len(), 2, "{mlir}");
    let body_indent = mlir
        .lines()
        .find(|l| l.trim_start().starts_with("gpu.return"))
        .map(|l| l.len() - l.trim_start().len())
        .expect("the function's return");
    for a in &asserts {
        assert!(a.len() - a.trim_start().len() > body_indent, "not nested: {a}\n{mlir}");
    }
    // The simulator: a negative value at thread 5 is not looked at; one at
    // thread 2 is; 99 past n is not.
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let mut input = vec![1; 32];
    input[5] = -1;
    input[20] = 99;
    k.simulate(cfg, &mut [KernelArg::i32(input.clone()), KernelArg::scalar(8i32)]).unwrap();
    input[2] = -1;
    let e = k.simulate(cfg, &mut [KernelArg::i32(input), KernelArg::scalar(8i32)]).unwrap_err();
    assert_eq!(e.thread, Some([2, 0, 0]), "{e}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm80).use_ptxas(false).enable_assertions(true);
    let ptx = k.compile_with(&tc, &opts).unwrap_or_else(|e| panic!("{e}")).into_ptx();
    assert!(ptx.matches("__assertfail").count() >= 3, "{ptx}");
}

/// RuntimeAssertion on the simulator: the first stored false flag fails
/// verify, with its message and line; true ones pass; disabled ones
/// declare nothing.
#[test]
fn runtime_assertions_on_the_simulator() {
    use cute::testing::{DeviceAssertions, RuntimeAssertion};
    let ra = RuntimeAssertion::new(3);
    let spec = ra.spec();
    let line = line!() + 4;
    let k = Kernel::builder("runtime_asserts").block_dim([1, 1, 1]).build(|| {
        let x = param_scalar::<i32>("x");
        let checks = DeviceAssertions::param("checks", spec);
        checks.store(0, x.gt(0), "x is positive");
        checks.store(2, x.lt(10), "x is below 10");
    });
    assert_eq!(k.param_names(), ["x", "checks"]);
    for (x, want) in [(5, None), (50, Some((2, "x is below 10"))), (-1, Some((0, "x is positive")))] {
        let mut ra = ra.clone();
        let mut args = vec![KernelArg::scalar(x), ra.kernel_arg().unwrap()];
        k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
        ra.read_back(&args[1]);
        match (ra.verify(&k), want) {
            (Ok(()), None) => {}
            (Err(f), Some((index, msg))) => {
                assert_eq!((f.index, f.message.as_str()), (index, msg));
                assert_eq!(f.loc.file, file!());
                assert_eq!(f.loc.line, if index == 2 { line + 1 } else { line });
            }
            (got, want) => panic!("x = {x}: {got:?}, want {want:?}"),
        }
    }
    // Disabled: no parameter, and verify passes.
    let off = RuntimeAssertion::new(2).disable(true);
    let spec = off.spec();
    let k = Kernel::build("disabled_asserts", || {
        DeviceAssertions::param("checks", spec).store(0, false, "never checked");
    });
    assert!(k.param_names().is_empty());
    assert!(off.kernel_arg().is_none() && off.verify(&k).is_ok());
}

// ---- m9: pipeline primitives and shared-memory partitions ----

mod pipeline_misuse {
    use super::{build_panic, simulate_err};
    use cute::memory::{SmemAllocator, SmemPartition, get_smem_capacity_in_bytes};
    use cute::pipeline::{
        Agent, CooperativeGroup, MbarrierArray, PipelineOp, PipelineUserType, agent_sync, alloc_reserved_mbarrier,
        make_pipeline_state,
    };
    use cute::prelude::*;

    type Misuse = (&'static str, Box<dyn FnOnce()>);

    fn warp() -> CooperativeGroup {
        CooperativeGroup::new(Agent::Warp, 1)
    }

    fn array(op: PipelineOp) -> MbarrierArray {
        MbarrierArray::new(&alloc_reserved_mbarrier(2, 1), 2, (op, warp()), 0)
    }

    /// Static misuse of the pipeline states, groups and mbarrier arrays is
    /// refused while the kernel is built.
    #[test]
    fn pipeline_primitives_refuse_bad_arguments() {
        let cases: Vec<Misuse> = vec![
            (
                "PipelineState::advance_by: count must be >= 1, not 0",
                Box::new(|| make_pipeline_state(PipelineUserType::Consumer, 3).advance_by(0)),
            ),
            (
                "CooperativeGroup: the group must have at least one agent, not 0",
                Box::new(|| {
                    CooperativeGroup::new(Agent::Thread, 0);
                }),
            ),
            (
                "an mbarrier array needs at least one stage, not 0",
                Box::new(|| {
                    MbarrierArray::new(&alloc_smem::<u64>(1, 8), 0, (PipelineOp::AsyncThread, warp()), 0);
                }),
            ),
            (
                "each mbarrier must count at least one arrival, not 0",
                Box::new(|| {
                    let none = CooperativeGroup::new(Agent::Thread, Int32::from(0));
                    MbarrierArray::new(&alloc_smem::<u64>(1, 8), 1, (PipelineOp::AsyncThread, none), 0);
                }),
            ),
            (
                "a TMA load's transaction byte count cannot be negative (-16)",
                Box::new(|| {
                    MbarrierArray::new(&alloc_smem::<u64>(1, 8), 1, (PipelineOp::TmaLoad, warp()), -16);
                }),
            ),
            ("a TCGen05Mma arrival needs a CtaGroup", Box::new(|| array(PipelineOp::TCGen05Mma).arrive(0, None, None))),
            (
                "an mbarrier array cannot arrive for PipelineOp::TmaStore",
                Box::new(|| array(PipelineOp::TmaStore).arrive(0, None, None)),
            ),
            (
                "MbarrierArray::arrive_and_drop is not supported yet",
                Box::new(|| array(PipelineOp::AsyncThread).arrive_and_drop()),
            ),
            ("agent_sync(Agent::Thread) is not supported", Box::new(|| agent_sync(Agent::Thread, false))),
            (
                "agent_sync(Agent::Warp): there is no warp barrier instruction",
                Box::new(|| agent_sync(Agent::Warp, false)),
            ),
        ];
        for (want, body) in cases {
            let msg = build_panic(|| {
                Kernel::builder("pipeline_misuse").block_dim([64, 1, 1]).build(body);
            });
            assert!(msg.contains(want), "expected `{want}` in: {msg}");
        }
    }

    /// A pipeline built without a target waits as CuTe DSL does, with
    /// `mbarrier.try_wait`: compiling it for sm_89 fails at the line that
    /// waits.
    #[test]
    fn pipeline_without_a_target_needs_sm90() {
        let line = std::cell::Cell::new(0);
        let kernel = Kernel::builder("untargeted_pipeline").block_dim([64, 1, 1]).build(|| {
            let full = array(PipelineOp::AsyncThread);
            sync_threads();
            if_(warp_idx().eq(1), || full.arrive(0, None, None));
            full.wait(0, 0);
            line.set(line!() - 1);
        });
        let e = cute::validate::check_target(kernel.ir(), Arch::Sm89).unwrap_err();
        assert!(e.message.contains("mbarrier.try_wait needs sm_90 or newer"), "{e}");
        let loc = e.loc.expect("a source location");
        assert!(loc.file().ends_with("diagnostics.rs") && loc.line() == line.get(), "{e} (want line {})", line.get());
        cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Without the init sync, nothing stops warp 1 from waiting on an
    /// mbarrier before warp 0 has initialised it: here warp 0 is held at a
    /// named barrier until warp 1 has polled. The simulator reports the
    /// poll.
    #[test]
    fn waiting_before_the_pipeline_init_is_reported() {
        let kernel = Kernel::builder("no_init_sync").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
            let out = param_tensor::<i32>("out", Layout::col_major(64));
            let hold = cute::pipeline::NamedBarrier::new(1, 64);
            if_(warp_idx().eq(0), || hold.arrive_and_wait());
            let full = array(PipelineOp::AsyncThread);
            // Missing: pipeline_init_arrive(None, false); pipeline_init_wait(None);
            let done = if_else_yield(
                warp_idx().eq(1),
                || {
                    let done = full.try_wait(0, 0);
                    hold.arrive();
                    done
                },
                || Boolean::from(true),
            );
            out.store(thread_idx().x, done.to::<i32>());
        });
        let mut args = vec![KernelArg::buffer(&[0i32; 64])];
        let e = simulate_err(&kernel, 64, &mut args);
        assert!(e.message.contains("on an mbarrier that was never initialised"), "{e}");
        assert!(e.to_string().contains("diagnostics.rs"), "{e}");
    }

    /// The allocator's limits, a capacity query with no target to go by,
    /// and a kernel too big for its target.
    #[test]
    fn smem_allocator_and_capacity_misuse() {
        let msg = build_panic(|| {
            Kernel::build("align_2048", || {
                SmemAllocator::new().allocate_array::<u32>(4, 2048, SmemPartition::User);
            });
        });
        assert!(msg.contains("an alignment of 2048 bytes is more than the 1024 bytes shared memory allows"), "{msg}");
        let msg = build_panic(|| {
            Kernel::build("no_elements", || {
                SmemAllocator::new().allocate_array::<u32>(0, 4, SmemPartition::Reserved);
            });
        });
        assert!(msg.contains("allocate_array: num_elems is 0"), "{msg}");
        if std::env::var("CUTE_RS_ARCH").is_err() {
            let msg = build_panic(|| {
                get_smem_capacity_in_bytes(None);
            });
            assert!(msg.contains("unsupported compute capability: no architecture was given"), "{msg}");
        }
        let msg = build_panic(|| {
            Kernel::builder("too_big_for_sm89").target(Arch::Sm89).build(|| {
                alloc_smem::<u8>(120 * 1024, 16).store(1u8);
            });
        });
        assert!(
            msg.contains(
                "the kernel uses 122880 bytes of shared memory, more than the 101376 bytes (99 KiB) a block can have \
                 on sm_89"
            ),
            "{msg}"
        );
    }
}

// ---- m9: PipelineAsync and the static persistent tile schedulers ----

mod pipeline_async_misuse {
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
        let kernel =
            Kernel::builder("pipeline_async_misuse").block_dim([threads, 1, 1]).target(Arch::Sm89).build(|| {
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
}

mod tile_scheduler_misuse {
    use super::build_panic;
    use cute::prelude::*;
    use cute::utils::{
        PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentRuntimeTileScheduler,
    };

    /// A cluster deeper than one CTA along K, a swizzle size of 0 and an
    /// inner mode of 2 are refused.
    #[test]
    fn schedulers_refuse_bad_arguments() {
        let msg = build_panic(|| {
            PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 2], Default::default());
        });
        assert!(msg.contains("unsupported cluster_shape_k 2"), "{msg}");
        let msg = build_panic(|| {
            let options = PersistentTileSchedulerOptions { swizzle_size: 0, raster_along_m: true };
            PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 1], options);
        });
        assert!(msg.contains("swizzle_size must be >= 1, not 0"), "{msg}");
        let msg = build_panic(|| {
            Kernel::builder("inner_mode_2").block_dim([32, 1, 1]).build(|| {
                let params =
                    PersistentTileSchedulerParams::new((4usize, 4usize, 1usize), [1, 1, 1], Default::default());
                StaticPersistentRuntimeTileScheduler::create(&params, block_idx(), grid_dim(), 2);
            });
        });
        assert!(msg.contains("inner_mode is 0 (M first) or 1 (N first), not 2"), "{msg}");
        let msg = build_panic(|| {
            Kernel::builder("runtime_grid").block_dim([32, 1, 1]).build(|| {
                let m = IndexExpr::from(param_scalar::<i32>("m"));
                let p =
                    PersistentTileSchedulerParams::new(Shape::from((m, 4usize, 1usize)), [1, 1, 1], Default::default());
                p.get_grid_shape(4);
            });
        });
        assert!(msg.contains("get_grid_shape needs a static problem shape"), "{msg}");
    }

    /// A cluster of 33 CTAs (or 0) is refused by `HardwareInfo` (needs a
    /// GPU; skipped without one).
    #[cfg(feature = "cuda")]
    #[test]
    fn hardware_info_refuses_bad_cluster_sizes() {
        let Ok(info) = cute::cuda::HardwareInfo::new(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        for bad in [33, 0] {
            let e = info.get_max_active_clusters(bad, None).unwrap_err();
            assert!(e.message.contains(&format!("a cluster has 1 to 32 CTAs, not {bad}")), "{e}");
        }
    }
}

mod cpasync_order_misuse {
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
                && e.message
                    .contains("the mbarrier smem0[2] for phase 1 (parity 1), which still waits for 32 of its 32"),
            "{e}"
        );
    }
}

mod grouped_scheduler_misuse {
    use super::{build_panic, simulate_err};
    use cute::prelude::*;
    use cute::utils::{PersistentTileSchedulerParams, StaticPersistentGroupTileScheduler, create_initial_search_state};

    /// The grouped search is warp-collective: called from half a warp, its
    /// shuffles read lanes that did not take part, which the simulator
    /// reports. Every lane of the warp must call it.
    #[test]
    fn grouped_search_from_part_of_a_warp_is_reported() {
        let kernel = Kernel::builder("half_warp_search").block_dim([32, 1, 1]).build(|| {
            let shapes = param_tensor::<i32>("problem_shape_mnkl", Layout::row_major((3usize, 4usize)));
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let params = PersistentTileSchedulerParams::new((1usize, 1usize, 4usize), [1, 1, 1], Default::default());
            if_(lane_idx().lt(16), || {
                let mut sched = StaticPersistentGroupTileScheduler::create(
                    &params,
                    block_idx(),
                    grid_dim(),
                    [16, 16, 16],
                    create_initial_search_state(),
                    3,
                    &shapes,
                    false,
                );
                let work = sched.get_current_work();
                out.store(lane_idx(), work.group_search_result().group_idx);
            });
        });
        let shapes = [16, 16, 16, 1, 16, 32, 16, 1, 32, 16, 16, 1];
        let mut args = vec![KernelArg::buffer(&shapes), KernelArg::buffer(&[0i32; 32])];
        let e = simulate_err(&kernel, 32, &mut args);
        assert!(
            e.to_string().contains("tests/diagnostics.rs")
                && e.message
                    .contains("shuffle with member mask 0xffffffff is executed by lane 0 of warp 0, but lane 16")
                && e.message.contains("every lane of the mask must execute it"),
            "{e}"
        );
    }

    /// No groups, a cluster tile with a 0 extent and a problem-shape tensor
    /// that is not `(group_count, 4)` are refused while the kernel is
    /// built.
    #[test]
    fn grouped_scheduler_refuses_bad_arguments() {
        let make = |groups: usize, tile: [usize; 3], rows: usize, cols: usize| {
            build_panic(move || {
                Kernel::builder("bad_grouped").block_dim([32, 1, 1]).build(|| {
                    let shapes = param_tensor::<i32>("shapes", Layout::row_major((rows, cols)));
                    let params =
                        PersistentTileSchedulerParams::new((1usize, 1usize, 4usize), [1, 1, 1], Default::default());
                    StaticPersistentGroupTileScheduler::create(
                        &params,
                        block_idx(),
                        grid_dim(),
                        tile,
                        create_initial_search_state(),
                        groups,
                        &shapes,
                        true,
                    );
                });
            })
        };
        let msg = make(0, [16, 16, 16], 1, 4);
        assert!(msg.contains("a grouped GEMM needs at least one group, not 0"), "{msg}");
        let msg = make(2, [16, 0, 16], 2, 4);
        assert!(msg.contains("the cluster tile [16, 0, 16] has a 0 extent"), "{msg}");
        let msg = make(2, [16, 16, 16], 2, 3);
        assert!(msg.contains("the problem shapes must be a (group_count, 4) = (2, 4) tensor"), "{msg}");
        let msg = make(3, [16, 16, 16], 2, 4);
        assert!(msg.contains("the problem shapes must be a (group_count, 4) = (3, 4) tensor"), "{msg}");
    }
}

/// Tensor maps edited in the kernel: a TMA copy
/// through a descriptor written without the tensormap-proxy release or acquire, or with
/// them in the wrong order, through bytes that hold no descriptor, a divergent
/// `tensormap.cp_fenceproxy`, and `tensormap.replace` for plain sm_90.
mod tensormap_misuse {
    use super::*;
    use cute::cpasync::{self, TensormapField};

    #[derive(Clone, Copy, PartialEq)]
    enum Bug {
        NoAcquire,
        NoRelease,
        AcquireFirst,
        NotCopied,
        ReplaceUncopied,
        DivergentCpFence,
        CpFenceNoAcquire,
    }

    /// One warp copies the TMA atom's descriptor to `desc` (in global memory, or through
    /// shared memory with `tensormap.cp_fenceproxy`), points it at `b`, releases,
    /// acquires and loads a 16x32 box through it, except for `bug`.
    fn kernel(bug: Bug) -> Kernel {
        Kernel::builder("tensormap_misuse").block_dim([32, 1, 1]).build(move || {
            let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64), (64, 1)), 16);
            let b = param_tensor_aligned::<F16>("b", Layout::new((64, 64), (64, 1)), 16);
            let desc = param_tensor_aligned::<u64>("desc", Layout::col_major(16), 128);
            let smem_layout = Layout::new((16, 32), (32, 1));
            let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
            let (atom, tma) = cpasync::make_tiled_tma_atom(g2s, &a, smem_layout.clone(), (16, 32), 1).into();
            let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
            let mbar = alloc_smem::<u64>(1, 8);
            if_(elect_one(), || mbarrier_init(&mbar, 1));
            mbarrier_init_fence();
            sync_threads();
            let d = desc.ptr();
            if matches!(bug, Bug::DivergentCpFence | Bug::CpFenceNoAcquire) {
                let sd = alloc_smem::<u64>(16, 128);
                if_(elect_one(), || {
                    cpasync::copy_tensormap(&atom, &sd);
                    cpasync::tensormap_replace(&sd, TensormapField::GlobalAddress, b.ptr().toint_i64());
                });
                sync_warp();
                if bug == Bug::DivergentCpFence {
                    if_(elect_one(), || cpasync::cp_fence_tma_desc_release(&d, &sd));
                } else {
                    cpasync::cp_fence_tma_desc_release(&d, &sd);
                }
            } else if bug != Bug::NotCopied {
                if_(elect_one(), || {
                    if bug != Bug::ReplaceUncopied {
                        cpasync::copy_tensormap(&atom, &d);
                    }
                    cpasync::tensormap_replace(&d, TensormapField::GlobalAddress, b.ptr().toint_i64());
                    if bug == Bug::AcquireFirst {
                        cpasync::fence_tma_desc_acquire(&d);
                    }
                    if bug != Bug::NoRelease {
                        cpasync::fence_tma_desc_release();
                    }
                });
            }
            if !matches!(bug, Bug::NoAcquire | Bug::AcquireFirst | Bug::CpFenceNoAcquire) {
                cpasync::fence_tma_desc_acquire(&d);
            }
            let g = local_tile(&tma, (16, 32), (0, 0));
            let (ts, tg) =
                cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 16 * 32 * 2));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(d.toint_i64()));
            mbarrier_wait(&mbar, 0);
        })
    }

    fn run(bug: Bug) -> SimError {
        let mut args = vec![
            KernelArg::buffer(&vec![F16::from_f32(1.0); 64 * 64]),
            KernelArg::buffer(&vec![F16::from_f32(2.0); 64 * 64]),
            KernelArg::buffer(&[0u64; 16]),
        ];
        simulate_err(&kernel(bug), 32, &mut args)
    }

    #[test]
    fn tma_through_an_updated_descriptor_needs_the_acquire() {
        let e = run(Bug::NoAcquire);
        assert!(
            e.message.contains("cp.async.bulk.tensor")
                && e.message.contains("this thread has not acquired it since warp 0 wrote and released it")
                && e.message.contains("fence_tma_desc_acquire(ptr)"),
            "{e}"
        );
    }

    #[test]
    fn tma_through_an_updated_descriptor_needs_the_release() {
        let e = run(Bug::NoRelease);
        assert!(
            e.message.contains("warp 0 wrote it through the generic proxy without releasing it since")
                && e.message.contains("fence_tma_desc_release()"),
            "{e}"
        );
    }

    #[test]
    fn tma_through_a_descriptor_published_from_shared_memory_needs_the_acquire() {
        let e = run(Bug::CpFenceNoAcquire);
        assert!(
            e.message.contains("cp.async.bulk.tensor")
                && e.message.contains("this thread has not acquired it since warp 0 wrote and released it"),
            "{e}"
        );
    }

    /// A TMA store through a descriptor in global memory, then a rewrite of that
    /// descriptor by the issuing lane, with or without first waiting for the store's
    /// bulk group to read its source.
    fn rewrite_after_store(wait: bool) -> Kernel {
        Kernel::builder("tensormap_rewrite_after_store").block_dim([32, 1, 1]).build(move || {
            let inp = param_tensor_aligned::<F16>("inp", Layout::new((16, 32), (32, 1)), 16);
            let o = param_tensor_aligned::<F16>("o", Layout::new((64, 64), (64, 1)), 16);
            let o2 = param_tensor_aligned::<F16>("o2", Layout::new((64, 64), (64, 1)), 16);
            let desc = param_tensor_aligned::<u64>("desc", Layout::col_major(16), 128);
            let smem_layout = Layout::new((16, 32), (32, 1));
            let (atom, tma) =
                cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &o, smem_layout.clone(), (16, 32), 1)
                    .into();
            let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
            let t = thread_idx().x;
            for_range(0, 16, 1, |j| {
                let idx = j * 32 + t;
                let (m, k) = (idx / 32, idx % 32);
                s.store((m, k), inp.load((m, k)));
            });
            fence_view_async_shared();
            sync_threads();
            let d = desc.ptr();
            if_(elect_one(), || {
                cpasync::copy_tensormap(&atom, &d);
                cpasync::fence_tma_desc_release();
            });
            cpasync::fence_tma_desc_acquire(&d);
            let g = local_tile(&tma, (16, 32), (0, 0));
            let (ts, tg) =
                cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
            copy_with(&atom, &ts, &tg, CopyArgs::new().tma_desc_ptr(d.toint_i64()));
            if_(elect_one(), || {
                cp_async_bulk_commit_group();
                if wait {
                    cp_async_bulk_wait_group(0, true);
                }
                cpasync::tensormap_replace(&d, TensormapField::GlobalAddress, o2.ptr().toint_i64());
                cpasync::fence_tma_desc_release();
                cp_async_bulk_wait_group(0, false);
            });
        })
    }

    #[test]
    fn rewriting_a_descriptor_a_tma_store_still_reads() {
        let args = || {
            vec![
                KernelArg::buffer(&vec![F16::from_f32(1.0); 16 * 32]),
                KernelArg::buffer(&vec![F16::from_f32(0.0); 64 * 64]),
                KernelArg::buffer(&vec![F16::from_f32(0.0); 64 * 64]),
                KernelArg::buffer(&[0u64; 16]),
            ]
        };
        let e = simulate_err(&rewrite_after_store(false), 32, &mut args());
        assert!(
            e.message.contains("a generic write to the tensor map at byte 0 of `desc`, which a TMA store")
                && e.message.contains("cp_async_bulk_wait_group(0, true)"),
            "{e}"
        );
        let mut ok = args();
        rewrite_after_store(true)
            .simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut ok)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(ok[1].to_vec::<F16>()[..32].iter().all(|x| x.to_f32() == 1.0));
    }

    #[test]
    fn the_acquire_must_follow_the_release() {
        let e = run(Bug::AcquireFirst);
        assert!(e.message.contains("is not ordered after that release"), "{e}");
    }

    #[test]
    fn tma_desc_ptr_at_memory_without_a_descriptor() {
        let e = run(Bug::NotCopied);
        assert!(
            e.message.contains("do not hold a tensor map (uninitialised memory")
                && e.message.contains("copy_tensormap"),
            "{e}"
        );
        let e = run(Bug::ReplaceUncopied);
        assert!(e.message.starts_with("tensormap.replace") && e.message.contains("do not hold a tensor map"), "{e}");
    }

    #[test]
    fn cp_fenceproxy_is_warp_collective() {
        let e = run(Bug::DivergentCpFence);
        assert!(e.message.contains("tensormap.cp_fenceproxy") && e.message.contains("1 of the 32 lanes"), "{e}");
    }

    #[test]
    fn tensormap_replace_is_refused_for_plain_sm90() {
        let k = kernel(Bug::NoAcquire);
        let e = cute::validate::check_target(k.ir(), Arch::Sm90).unwrap_err();
        assert!(
            e.message.contains("tensormap.replace (update_tma_descriptor, tensormap_replace) needs sm_90a")
                && e.message.contains("being compiled for sm_90"),
            "{e:?}"
        );
        assert!(e.loc.is_some_and(|l| l.file().ends_with("diagnostics.rs")), "{e:?}");
        assert!(cute::validate::check_target(k.ir(), Arch::Sm90a).is_ok());
    }
}

// ---- m9-pipeline-tma: PipelineTmaAsync and PipelineTmaStore ----

mod tma_pipeline_misuse {
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
            let (ts, tg) =
                cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
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
                    let (sum, _) =
                        for_range_carry(0, TILES, 1, (Float32::from(0.0f32), consumer), |_, (sum, mut c)| {
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
                assert!(
                    m.contains("data race") || m.contains("phase aliasing") || m.contains("while a bulk copy"),
                    "{e}"
                );
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
            let (atom, tma) =
                cpasync::make_tiled_tma_atom(s2g, &out, Layout::new((16, 16), (16, 1)), (16, 16), 1).into();
            let s = make_smem_tensor_aligned::<f32>(smem, 128);
            let g = local_tile(&tma, (16, 16), (0, ..));
            let (ts, tg) =
                cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
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
            (
                "PipelineTmaStore has no consumer side",
                Box::new(move || PipelineTmaStore::create(2, one()).consumer_wait()),
            ),
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
}

// Misuse of tensor-memory allocation.
mod tmem_misuse {
    use super::*;

    /// Simulate `body` in one CTA of 64 threads (or a cluster of `cluster` CTAs) and return
    /// the simulator's report.
    fn sim_in(cluster: u32, body: impl FnOnce()) -> SimError {
        let kernel = Kernel::builder("tmem_misuse").block_dim([64, 1, 1]).build(body);
        let cfg = LaunchConfig::new([cluster, 1, 1], [64, 1, 1]).with_cluster([cluster, 1, 1]);
        match kernel.simulate(cfg, &mut Vec::new()) {
            Ok(_) => panic!("expected the simulator to report the misuse"),
            Err(e) => e,
        }
    }

    fn sim(body: impl FnOnce()) -> String {
        sim_in(1, body).message
    }

    fn warp0(f: impl FnOnce()) {
        if_(warp_idx().eq(0), f);
    }

    const A: Arch = Arch::Sm100a;

    #[test]
    fn leak_at_exit() {
        let e = sim(|| warp0(|| alloc_tmem(64, &alloc_smem::<i32>(1, 4), None, A)));
        assert!(e.contains("exits with tensor memory still allocated") && e.contains("columns 0 to 63"), "{e}");
    }

    #[test]
    fn double_free() {
        let e = sim(|| {
            warp0(|| {
                let slot = alloc_smem::<i32>(1, 4);
                alloc_tmem(32, &slot, None, A);
                let p = retrieve_tmem_ptr::<f32>(4, &slot);
                dealloc_tmem(&p, 32, None, A);
                dealloc_tmem(&p, 32, None, A);
            })
        });
        assert!(e.contains("not the start of a live allocation") && e.contains("freed twice"), "{e}");
    }

    #[test]
    fn mismatched_free() {
        let e = sim(|| {
            warp0(|| {
                let slot = alloc_smem::<i32>(1, 4);
                alloc_tmem(64, &slot, None, A);
                dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), 32, None, A);
            })
        });
        assert!(e.contains("32 columns, but the allocation has 64"), "{e}");
    }

    #[test]
    fn non_collective_issue() {
        let e = sim(|| {
            let slot = alloc_smem::<i32>(1, 4);
            if_(thread_idx().x.lt(16), || alloc_tmem(32, &slot, None, A));
        });
        assert!(e.contains("tcgen05.alloc executed by 16 of the 32 lanes"), "{e}");
    }

    #[test]
    fn non_uniform_operands() {
        let e = sim(|| {
            warp0(|| {
                let slot = alloc_smem::<i32>(1, 4);
                let n = select(thread_idx().x.lt(4), Int32::from(64), Int32::from(32));
                alloc_tmem(n, &slot, None, A);
            })
        });
        assert!(e.contains("every lane of the warp gives the same operands"), "{e}");
    }

    #[test]
    fn mixed_cta_group() {
        let e = sim_in(2, || {
            warp0(|| {
                relinquish_tmem_alloc_permit(Some(true));
                alloc_tmem(32, &alloc_smem::<i32>(1, 4), None, A);
            })
        })
        .message;
        assert!(e.contains("all tcgen05 instructions of a kernel use the same cta_group"), "{e}");
    }

    #[test]
    fn allocation_beyond_capacity_is_a_deadlock() {
        let e = sim(|| {
            warp0(|| {
                let (a, b) = (alloc_smem::<i32>(1, 4), alloc_smem::<i32>(1, 4));
                alloc_tmem(512, &a, None, A);
                alloc_tmem(32, &b, None, A);
            })
        });
        assert!(e.contains("deadlock") && e.contains("columns 0 to 511") && e.contains("hang"), "{e}");
    }

    #[test]
    fn alloc_after_relinquish() {
        let e = sim(|| {
            warp0(|| {
                relinquish_tmem_alloc_permit(None);
                alloc_tmem(32, &alloc_smem::<i32>(1, 4), None, A);
            })
        });
        assert!(e.contains("relinquished its allocation permit"), "{e}");
    }

    #[test]
    fn retrieve_of_a_freed_allocation() {
        let e = sim(|| {
            warp0(|| {
                let slot = alloc_smem::<i32>(1, 4);
                alloc_tmem(32, &slot, None, A);
                dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), 32, None, A);
                let _ = retrieve_tmem_ptr::<f32>(4, &slot);
            })
        });
        assert!(e.contains("not in a live TMEM allocation"), "{e}");
    }

    /// Warp 1 reads the address without a barrier after warp 0's allocation wrote it.
    #[test]
    fn retrieve_without_a_barrier_races() {
        let e = sim(|| {
            let slot = alloc_smem::<i32>(1, 4);
            warp0(|| alloc_tmem(32, &slot, None, A));
            if_(warp_idx().eq(1), || {
                let _ = retrieve_tmem_ptr::<f32>(4, &slot);
            });
        });
        assert!(e.contains("race"), "{e}");
    }

    #[test]
    fn pair_allocation_needs_an_even_cluster() {
        let e = sim(|| warp0(|| alloc_tmem(32, &alloc_smem::<i32>(1, 4), Some(true), A)));
        assert!(e.contains("even number of CTAs"), "{e}");
    }

    #[test]
    fn exclusive_allocation_freed_by_a_plain_dealloc() {
        let e = sim(|| {
            warp0(|| {
                let slot = alloc_smem::<i32>(1, 4);
                alloc_tmem(576, &slot, None, Arch::Sm107f);
                // A runtime count (64 * 9), so that only the simulator sees it.
                dealloc_tmem(&retrieve_tmem_ptr::<f32>(4, &slot), block_dim().x * 9, None, Arch::Sm100a);
            })
        });
        assert!(e.contains("freed by tcgen05.dealloc.exclusive of the same size"), "{e}");
    }
}

// Software-pipelined loops.
mod range_prefetch_misuse {
    use super::{build_panic, simulate_err};
    use cute::pipeline::{Agent, CooperativeGroup, Pipeline, PipelineCpAsync, PipelineCpAsyncOptions};
    use cute::prelude::*;

    /// One warp both produces into and consumes from a 2-stage
    /// `PipelineCpAsync` over 6 tiles, `ahead` tiles apart (bounded by the
    /// stage count when `bounded`).
    fn single_warp(ahead: i64, bounded: bool) -> Kernel {
        Kernel::builder("prefetch_misuse").block_dim([32, 1, 1]).target(Arch::Sm89).build(move || {
            let src = param_tensor::<u32>("src", Layout::col_major(6 * 128));
            let out = param_tensor::<u32>("out", Layout::col_major(6 * 128));
            let warp = CooperativeGroup::new(Agent::Warp, 1);
            let pipeline = PipelineCpAsync::create(2, warp, warp, PipelineCpAsyncOptions::default());
            let data = make_smem_tensor::<u32>(Layout::col_major(2 * 128));
            let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
            let t = thread_idx().x;
            let prefetch = Prefetch::stages(ahead);
            let prefetch = if bounded { prefetch.bounded_by(pipeline.num_stages()) } else { prefetch };
            for_range_pipelined(
                0,
                6,
                1,
                prefetch,
                pipeline.make_participants(),
                |k, mut p| {
                    let h = p.acquire_and_advance(None);
                    let from = make_tensor(&src.ptr().offset((k * 32 + t) * 4), Layout::col_major(4));
                    let to = make_tensor(&data.ptr().offset((h.index() * 32 + t) * 4), Layout::col_major(4));
                    copy(&atom, &from, &to);
                    h.commit();
                    p
                },
                |k, mut c| {
                    let h = c.wait_and_advance(None);
                    for j in 0..4 {
                        out.store(k * 128 + t + j * 32, data.load(h.index() * 128 + t + j * 32));
                    }
                    h.release();
                    c
                },
            );
        })
    }

    #[test]
    fn negative_prefetch_stages() {
        let e = build_panic(|| {
            single_warp(-1, false);
        });
        assert!(e.contains("prefetch stages must be a static integer >= 0, got -1"), "{e}");
    }

    #[test]
    fn prefetch_stages_beyond_the_pipeline() {
        let e = build_panic(|| {
            single_warp(2, true);
        });
        assert!(e.contains("prefetch stages 2 with a 2-stage pipeline") && e.contains("at most 1 stages"), "{e}");
    }

    /// Without the bound, a distance of the stage count makes the warp wait
    /// for a release it makes later: the simulator reports the hang (the
    /// sm_89 wait spins on `mbarrier.test_wait`), in the steady-state loop.
    #[test]
    fn prefetch_stages_beyond_the_pipeline_hang_unbounded() {
        let kernel = single_warp(2, false);
        let mut args = [KernelArg::buffer(&[1u32; 6 * 128]), KernelArg::buffer(&[0u32; 6 * 128])];
        let e = simulate_err(&kernel, 32, &mut args);
        assert!(e.message.contains("did not terminate") && e.message.contains("a hang"), "{e}");
        assert!(e.message.contains("for_range at tests/diagnostics.rs"), "{e}");
    }
}

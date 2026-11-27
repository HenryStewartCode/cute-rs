//! The first diagnostics: cp.async staging, shared-memory races, out-of-bounds and divergence errors, launch and argument checks, warp mma constructors.

use super::*;
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
    assert!(e.to_string().contains("tests/diagnostics/main.rs"), "error should point at the kernel source: {e}");
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

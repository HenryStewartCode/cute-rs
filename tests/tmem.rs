//! Tensor memory: the TMEM allocation kernels in the simulator
//! and with `mlir-opt`, sm_107's exclusive allocation, the target requirements and the
//! build-time checks. (`tests/diagnostics.rs` has the simulator's misuse reports, and
//! `tests/ptxas.rs` assembles the kernels for sm_100a.)

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use cute::validate::check_target;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn tmem_kernels_match_the_rust_reference() {
    for case in feature_kernels::sm100() {
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

#[test]
fn tmem_kernels_compile_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for case in feature_kernels::sm100() {
        let options = feature_kernels::options_for(&case.kernel, &tc).gpu_arch(Arch::Sm100a);
        case.kernel.compile_with(&tc, &options).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    }
}

#[test]
fn tmem_mlir_uses_the_nvvm_operations() {
    let one = feature_kernels::tmem_alloc_32().kernel.to_mlir();
    assert!(one.contains("nvvm.tcgen05.alloc"), "{one}");
    assert!(one.contains(" : !llvm.ptr<3>, i32") && !one.contains("group ="), "{one}");
    assert!(one.contains("nvvm.tcgen05.dealloc"), "{one}");
    assert!(one.contains("nvvm.tcgen05.relinquish_alloc_permit"), "{one}");
    let pair = feature_kernels::tmem_alloc_pair().kernel.to_mlir();
    assert!(pair.contains("{group = #nvvm.tcgen05_group<cta_2>}"), "{pair}");
}

/// sm_107's exclusive allocation of all 576 columns: simulated, its MLIR text checked (no
/// installed LLVM or ptxas knows PTX 9.4), and its target requirement.
fn exclusive_kernel() -> Kernel {
    Kernel::builder("tmem_exclusive").block_dim([64, 1, 1]).target(Arch::Sm107f).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64));
        let slot = alloc_smem::<u32>(1, 4);
        if_(warp_idx().eq(0), || alloc_tmem(576, &slot, None, Arch::Sm107f));
        sync_threads();
        let p = retrieve_tmem_ptr::<f32>(4, &slot);
        out.store(thread_idx().x, p.toint_i32());
        sync_threads();
        if_(warp_idx().eq(0), || dealloc_tmem(&p, 576, None, Arch::Sm107f));
    })
}

#[test]
fn exclusive_allocation_on_sm107() {
    let kernel = exclusive_kernel();
    let mut args = vec![KernelArg::i32(vec![-1; 64])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![0; 64], "the exclusive allocation starts at column 0");
    let text = kernel.to_mlir();
    assert!(text.contains("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;"), "{text}");
    assert!(text.contains("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 $0, $1;"), "{text}");
    check_target(kernel.ir(), Arch::Sm107f).unwrap();
    let e = check_target(kernel.ir(), Arch::Sm100a).unwrap_err().message;
    assert!(e.contains("tcgen05.alloc.exclusive"), "{e}");
}

#[test]
fn tmem_target_requirements() {
    let ir = feature_kernels::tmem_alloc_64().kernel;
    for a in [Arch::Sm100a, Arch::Sm100f, Arch::Sm110a, Arch::Sm103a] {
        check_target(ir.ir(), a).unwrap_or_else(|e| panic!("{a:?}: {e}"));
    }
    for a in [Arch::Sm90a, Arch::Sm100, Arch::Sm120a] {
        let e = check_target(ir.ir(), a).unwrap_err().message;
        assert!(e.contains("tcgen05.alloc"), "{a:?}: {e}");
    }
}

#[test]
fn tmem_build_time_checks() {
    let kernel = |f: Box<dyn FnOnce()>| {
        Kernel::builder("tmem_bad").block_dim([32, 1, 1]).build(f);
    };
    let e = build_panic(|| {
        kernel(Box::new(|| alloc_tmem(48, &alloc_smem::<i32>(1, 4), None, Arch::Sm100a)));
    });
    assert!(e.contains("power of two from 32 to 512"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| alloc_tmem(1024, &alloc_smem::<i32>(1, 4), None, Arch::Sm100a)));
    });
    assert!(e.contains("power of two from 32 to 512"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| alloc_tmem(530, &alloc_smem::<i32>(1, 4), None, Arch::Sm107f)));
    });
    assert!(e.contains("multiple of 32"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| alloc_tmem(32, &alloc_smem::<u64>(1, 8), None, Arch::Sm100a)));
    });
    assert!(e.contains("32-bit word"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| {
            let x = param_ptr::<i32>("x");
            alloc_tmem(32, &x, None, Arch::Sm100a)
        }));
    });
    assert!(e.contains("shared memory"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| dealloc_tmem(&alloc_smem::<f32>(1, 4), 32, None, Arch::Sm100a)));
    });
    assert!(e.contains("not a tensor-memory one"), "{e}");
    let e = build_panic(|| {
        kernel(Box::new(|| {
            let p = retrieve_tmem_ptr::<f32>(4, &alloc_smem::<i32>(1, 4));
            let t = make_tensor(p, Layout::new((128, 32), (1 << 16, 1)));
            let _ = t.load((0, 0));
        }));
    });
    assert!(e.contains("TMEM tensors are accessed only by tcgen05 copies and MMAs"), "{e}");
}

/// A cute.struct member holds the address as well as a pointer does.
#[test]
fn struct_member_slot() {
    let kernel = Kernel::builder("tmem_struct_slot").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let mut b = StructLayout::builder("Storage");
        let _bar = b.scalar::<u64>("bar", None);
        let holder = b.scalar::<u32>("tmem_holding_buf", None);
        let layout = b.build();
        let base = layout.allocate();
        let slot = holder.at(&base);
        alloc_tmem(128, &slot, None, Arch::Sm100a);
        sync_warp();
        let p = retrieve_tmem_ptr::<f32>(16, &slot);
        out.store(thread_idx().x, p.toint_i32());
        dealloc_tmem(&p, 128, None, Arch::Sm100a);
    });
    let mut args = vec![KernelArg::i32(vec![-1; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![0; 32]);
}

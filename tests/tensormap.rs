//! In-kernel tensor-map copies and updates (roadmap part m7-tensormap-update): the
//! feature kernels under seeded schedules and random completion, the build-time checks
//! of the descriptor functions and `TensorMapManager`, and the PTX they lower to.
//! Misuse the simulator reports is in `tests/diagnostics.rs`.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::cpasync::{self, TensormapField};
use cute::prelude::*;
use cute::utils::{TensorMapManager, TensorMapUpdateMode};

/// Every tensor-map kernel under the built-in schedules and 8 seeded ones with random
/// completion: each matches plain Rust.
#[test]
fn tensormap_kernels_match_rust_under_seeded_schedules() {
    let seeds: Vec<u64> = (0..8).collect();
    for case in feature_kernels::tensormap_cases_sm90a() {
        for small in [true, false] {
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            let report = case
                .kernel
                .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
            assert!(report.tma_instructions > 0, "{}: {report}", case.name);
            let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
            case.kernel
                .simulate_schedules(case.cfg, &mut args, &SimOptions::default(), &seeds)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            (case.check)(&args);
        }
    }
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A TMA load atom over a row-major `(64, 128)` f16 parameter, 16x32 tiles.
fn load_atom() -> (CopyAtom<F16>, Tensor<F16>) {
    let a = param_tensor_aligned::<F16>("a", Layout::new((64, 128), (128, 1)), 16);
    let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
    let (atom, _) = cpasync::make_tiled_tma_atom(g2s, &a, Layout::new((16, 32), (32, 1)), (16, 32), 1).into();
    (atom, a)
}

/// What the descriptor functions refuse when the kernel is built.
#[test]
fn descriptor_functions_check_their_operands() {
    let e = build_panic(|| {
        Kernel::build("bad_replace_ord", || {
            let d = param_ptr::<u64>("d");
            cpasync::tensormap_replace(&d, TensormapField::GlobalStride(4), Int64::from(16));
        });
    });
    assert!(e.contains("ordinals 0 to 3"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_replace_float", || {
            let d = param_ptr::<u64>("d");
            cpasync::tensormap_replace(&d, TensormapField::GlobalDim(0), Float32::from(1.0));
        });
    });
    assert!(e.contains("is an integer"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_copy_atom", || {
            let d = param_ptr::<u64>("d");
            let atom = make_copy_atom::<F16>(CopyOp::Universal { bits: 16 });
            cpasync::copy_tensormap(&atom, &d);
        });
    });
    assert!(e.contains("is not a TMA atom"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_update_rank", || {
            let (atom, _) = load_atom();
            let d = param_ptr::<u64>("d");
            let other = param_tensor::<F16>("other", Layout::new((4, 64, 128), (8192, 128, 1)));
            cpasync::update_tma_descriptor(&atom, &other, &d);
        });
    });
    assert!(e.contains("has 3 flat modes, but the atom was made for a tensor of 2"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_update_stride", || {
            let (atom, _) = load_atom();
            let d = param_ptr::<u64>("d");
            let other = param_tensor::<F16>("other", Layout::new((64, 128), (1, 64)));
            cpasync::update_tma_descriptor(&atom, &other, &d);
        });
    });
    assert!(e.contains("TMA's dimension 0, which is contiguous, but its stride is 64"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_acquire_smem", || {
            let s = alloc_smem::<u64>(16, 128);
            cpasync::fence_tma_desc_acquire(&s);
        });
    });
    assert!(e.contains("lives in global memory"), "{e}");
    let e = build_panic(|| {
        Kernel::build("bad_cp_fence", || {
            let g = param_ptr::<u64>("g");
            let h = param_ptr::<u64>("h");
            cpasync::cp_fence_tma_desc_release(&g, &h);
        });
    });
    assert!(e.contains("it is the descriptor in shared memory"), "{e}");
    let e = build_panic(|| {
        let _ = TensorMapManager::new(TensorMapUpdateMode::GMEM, 64);
    });
    assert!(e.contains("at least 128"), "{e}");
}

/// `tensormap.replace` is architecture-specific: plain sm_90 refuses it, sm_90a,
/// sm_100a and sm_120a take it (PTX ISA 8.3 on sm_90a), and the fences need sm_90.
#[test]
fn tensormap_replace_needs_an_arch_specific_target() {
    let kernel = feature_kernels::tensormap_replace_load().kernel;
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90).unwrap_err();
    assert!(e.message.contains("tensormap.replace") && e.message.contains("sm_90a"), "{e:?}");
    assert_eq!(cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap(), 83);
    assert!(cute::validate::check_target(kernel.ir(), Arch::Sm100a).is_ok());
    assert!(cute::validate::check_target(kernel.ir(), Arch::Sm120a).is_ok());
    let fences = Kernel::build("fences_only", || {
        let d = param_ptr::<u64>("d");
        cpasync::fence_tma_desc_release();
        cpasync::fence_tma_desc_acquire(&d);
    });
    let e = cute::validate::check_target(fences.ir(), Arch::Sm89).unwrap_err();
    assert!(e.message.contains("needs sm_90"), "{e:?}");
    assert_eq!(cute::validate::check_target(fences.ir(), Arch::Sm90).unwrap(), 83);
}

/// The PTX of each operation, as inline assembly in the MLIR.
#[test]
fn tensormap_operations_lower_to_inline_ptx() {
    let mlir = feature_kernels::tensormap_replace_load().kernel.to_mlir();
    for want in [
        "tensormap.replace.tile.global_address.global.b1024.b64 [$0], $1;",
        "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], 1, $1;",
        "fence.proxy.tensormap::generic.release.gpu;",
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "cp.async.bulk.tensor.2d",
    ] {
        assert!(mlir.contains(want), "missing {want:?} in\n{mlir}");
    }
    let mlir = feature_kernels::tensormap_grouped_smem().kernel.to_mlir();
    for want in [
        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [$0], $1;",
        "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], 0, $1;",
        "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], 0, $1;",
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [$0], [$1], 128;",
        "nvvm.cp.async.bulk.wait_group 0 {read}",
    ] {
        assert!(mlir.contains(want), "missing {want:?} in\n{mlir}");
    }
    let mlir = feature_kernels::tensormap_grouped_gmem().kernel.to_mlir();
    assert!(mlir.contains("tensormap.replace.tile.global_stride.global.b1024.b64 [$0], 0, $1;"), "{mlir}");
    assert!(mlir.contains("fence.acq_rel.cta") || mlir.contains("nvvm.fence"), "{mlir}");
    assert!(!mlir.contains("cp_fenceproxy"), "{mlir}");
}

/// Each kernel compiles to PTX for sm_90a (when `mlir-opt` is available), with the
/// instructions in it; `ptxas` assembles them on the pod (`tests/ptxas.rs`).
#[test]
fn tensormap_kernels_compile_for_sm90a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for case in feature_kernels::tensormap_cases_sm90a() {
        let ptx = case.kernel.compile_ptx(&tc, Arch::Sm90a).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(ptx.contains(".target sm_90a"), "{}", case.name);
        assert!(ptx.contains("tensormap.replace.tile."), "{}", case.name);
        assert!(ptx.contains("fence.proxy.tensormap::generic.acquire.gpu"), "{}", case.name);
        assert!(ptx.contains("cp.async.bulk.tensor.2d"), "{}", case.name);
    }
}

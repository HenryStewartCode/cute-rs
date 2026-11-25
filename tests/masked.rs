//! Masked accesses, gather and scatter, and the `cutlass.Pointer` methods:
//! what they lower to, and the simulator's and the build's reports for
//! their misuse. The kernels themselves are feature kernels
//! (`tests/feature_kernels/masked.rs`).

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use cute::tensor::{gather, scatter};

mod feature_kernels;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn toolchain() -> Option<Toolchain> {
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    tc
}

fn count(text: &str, pattern: &str) -> usize {
    text.lines().filter(|l| l.contains(pattern)).count()
}

/// A masked load is an `scf.if` around the access yielding the pass-through
/// value or a frozen `ub.poison`, and a masked store an `scf.if` around the store;
/// NVPTX guards each with a branch on its flag (as it scalarises
/// `llvm.masked.load`). An all-true mask is a plain vector access, and a
/// constant-false store lane is dropped.
#[test]
fn masked_accesses_lower_to_guarded_accesses() {
    let kernel = feature_kernels::masked_tensor_ssa().kernel;
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("= scf.if %v") && mlir.contains("ub.poison : f32"), "{mlir}");
    assert!(mlir.contains("= llvm.freeze %poison"), "{mlir}");
    assert!(count(&mlir, "-> (f32) {") >= 8 * 4, "{mlir}");
    let Some(tc) = toolchain() else { return };
    let ptx = kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    // Lane 7's flag (7 < n & 7) folds to false: seven guarded lanes per
    // global access, each a branch over one scalar load or store (LLVM
    // forwards the shared-memory loads from the thread's own stores).
    assert!(count(&ptx, "bra") >= 3 * 7, "{ptx}");
    // (LLVM 22 types the moves .b32, older LLVMs .f32.)
    let scalar = |p: &str, op: &str| count(p, &format!("{op}.global.b32")) + count(p, &format!("{op}.global.f32"));
    assert!(scalar(&ptx, "ld") >= 2 * 7 && scalar(&ptx, "st") >= 7, "{ptx}");
    let ptx = feature_kernels::masked_load_store().kernel.compile_ptx(&tc, Arch::Sm80).unwrap();
    assert_eq!(count(&ptx, "ld.global.v"), 1, "{ptx}");
    assert_eq!(scalar(&ptx, "st"), 2, "{ptx}");
    assert!(!ptx.contains("bra"), "{ptx}");
}

/// The ordered, volatile, non-coherent and cache-hinted `Ptr` accesses
/// reach the PTX with their qualifiers; cluster scope needs sm_90.
#[test]
fn pointer_qualifiers() {
    let Some(tc) = toolchain() else { return };
    let ptx = feature_kernels::pointer_api().kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    for pattern in [
        "ld.acquire.gpu.global.b32",
        "ld.global.cs.b32",
        "ld.global.nc.b32",
        "ld.global.L1::evict_last.L2::128B.v4.b32",
        "st.release.gpu.global.b32",
        "st.global.wt.b32",
        "st.global.L1::evict_last.v4.b32",
        "ld.volatile.global.b32",
        "ld.global.v4.",
    ] {
        assert!(ptx.contains(pattern), "no {pattern} in\n{ptx}");
    }
    let cluster = feature_kernels::pointer_cluster_scope().kernel;
    let e = cluster.compile_ptx(&tc, Arch::Sm80).unwrap_err().to_string();
    assert!(e.contains("cluster scope needs sm_90"), "{e}");
    let ptx = cluster.compile_ptx(&tc, Arch::Sm90).unwrap_or_else(|e| panic!("{e}"));
    for pattern in ["ld.relaxed.cluster.global.v4.b32", "st.relaxed.cluster.global.v4.b32"] {
        assert!(ptx.contains(pattern), "no {pattern} in\n{ptx}");
    }
}

/// A use of a masked-off lane of a masked load without `pass_thru` is
/// reported, naming the load's line; a select that picks another value
/// is not a use.
#[test]
fn reading_an_undefined_lane_is_reported() {
    let line = line!() + 7;
    let kernel = |select_it: bool| {
        Kernel::builder("undefined_lane").block_dim([32, 1, 1]).build(move || {
            let x = param_tensor::<f32>("x", Layout::col_major(8));
            let out = param_tensor::<f32>("out", Layout::col_major(8));
            let lanes = TensorSsa::new(8, (0..8).map(Int32::constant).collect());
            let mask = lanes.lt(thread_idx().x);
            let v = x.load_ssa_masked(&mask, None);
            let v = if select_it { where_(&mask, &(v * 2.0), 0.0) } else { v * 2.0 };
            if_(thread_idx().x.eq(3), || out.store_ssa(&v));
        })
    };
    let mut args = vec![KernelArg::f32(vec![1.0; 8]), KernelArg::f32(vec![0.0; 8])];
    kernel(true).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].as_f32(), [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let e = kernel(false).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(
        e.message.contains("store reads a value that a masked-off lane of the masked load at")
            && e.message.contains(&format!("tests/masked.rs:{line}:")),
        "{e}"
    );
}

/// Indices past the buffer are reported by the simulator's bounds checks
/// (CuTe DSL leaves them undefined); the build checks the shapes.
#[test]
fn gather_and_scatter_misuse() {
    let kernel = Kernel::builder("gather_oob").block_dim([1, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((8, 2)));
        let idx = param_tensor::<i32>("idx", Layout::col_major((2, 2)));
        let out = param_tensor::<f32>("out", Layout::col_major((2, 2)));
        out.store_ssa(&gather(&x, 0, &idx.load_ssa()));
    });
    let mut args = vec![KernelArg::f32(vec![1.0; 16]), KernelArg::i32(vec![0, 1, 2, 40]), KernelArg::f32(vec![0.0; 4])];
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("out of bounds"), "{e}");
    let e = build_panic(|| {
        Kernel::build("gather_rank", || {
            let x = param_tensor::<f32>("x", Layout::col_major((8, 2)));
            let _ = gather(&x, 0, &full::<i32>(4, 0));
        });
    });
    assert!(e.contains("have different ranks"), "{e}");
    let e = build_panic(|| {
        Kernel::build("gather_mode", || {
            let x = param_tensor::<f32>("x", Layout::col_major((8, 2)));
            let _ = gather(&x, 2, &full::<i32>((1, 1), 0));
        });
    });
    assert!(e.contains("mode 2 is out of range"), "{e}");
    let e = build_panic(|| {
        Kernel::build("gather_extent", || {
            let x = param_tensor::<f32>("x", Layout::col_major((8, 2)));
            let _ = gather(&x, 0, &full::<i32>((2, 3), 0));
        });
    });
    assert!(e.contains("mode 1 of the index shape (2,3) (3) is larger than the tensor's (2)"), "{e}");
    let e = build_panic(|| {
        Kernel::build("scatter_register", || {
            let frag = make_fragment::<f32>((64, 2));
            frag.fill(0.0);
            scatter(&frag, 0, &full::<i32>((1, 2), 0), &full::<f32>((1, 2), 1.0));
        });
    });
    assert!(e.contains("more than the 32 a select tree") && e.contains("make_rmem_tensor"), "{e}");
}

#[test]
fn masked_access_sizes_and_pointer_steps_are_checked() {
    let e = build_panic(|| {
        Kernel::build("mask_size", || {
            let x = param_tensor::<f32>("x", Layout::col_major(8));
            let _ = x.load_ssa_masked(&full::<bool>(4, true), None);
        });
    });
    assert!(e.contains("the tensor has 8 elements, but the mask has 4"), "{e}");
    let e = build_panic(|| {
        Kernel::build("ptr_back", || {
            let x = param_tensor::<f32>("x", Layout::col_major(8));
            let p = x.ptr() + 2usize;
            let _ = p - 3usize;
        });
    });
    assert!(e.contains("has a static part of only 2 elements"), "{e}");
    let e = build_panic(|| {
        Kernel::build("ptr_misaligned", || {
            let x = param_tensor::<f32>("x", Layout::col_major(16));
            let _ = (x.ptr() + 1usize).load_vec(4, 16);
        });
    });
    assert!(e.contains("the alignment 16 is wrong") && e.contains("4 bytes past a 16-byte boundary"), "{e}");
    // A runtime offset cannot be checked: the alignment is trusted.
    Kernel::build("ptr_trusted", || {
        let x = param_tensor::<f32>("x", Layout::col_major(16));
        let _ = (x.ptr() + thread_idx().x).load_vec(4, 16);
    });
}

/// A masked load's undefined lanes may be another masked load's
/// pass-through (that is not a use), but an integer division by one is
/// undefined behaviour, reported.
#[test]
fn undefined_pass_through_and_division() {
    let kernel = |divide: bool| {
        Kernel::builder("undefined_chain").block_dim([32, 1, 1]).build(move || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            let out = param_tensor::<i32>("out", Layout::col_major(8));
            let lanes = TensorSsa::new(8, (0..8).map(Int32::constant).collect());
            let mask = lanes.lt(thread_idx().x);
            let a = x.load_ssa_masked(&mask, None);
            let b = x.load_ssa_masked(&mask, Some(&a));
            let b = if divide { full::<i32>(8, 100) / b } else { b };
            if_(thread_idx().x.eq(3), || out.store_ssa(&where_(&mask, &b, 0)));
        })
    };
    let mut args = vec![KernelArg::i32(vec![5; 8]), KernelArg::i32(vec![0; 8])];
    kernel(false).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[1].to_vec::<i32>(), [5, 5, 5, 0, 0, 0, 0, 0]);
    let e = kernel(true).simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("reads a value that a masked-off lane of the masked load at"), "{e}");
}

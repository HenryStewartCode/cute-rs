//! `stmatrix` atoms and `setmaxnreg`: the emitted
//! NVVM per LLVM version, the PTX of each installed toolchain, the simulator's register
//! budgets, and the diagnostics. The feature kernels (`tests/feature_kernels/stmatrix.rs`)
//! check the results against the PTX ISA's fragment tables.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::ir::Op;
use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn block(threads: u32) -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [threads, 1, 1])
}

fn sim_err(kernel: &Kernel, threads: u32, args: &mut [KernelArg]) -> SimError {
    kernel.simulate(block(threads), args).expect_err("expected the simulator to report an error")
}

#[test]
fn copies_issue_one_stmatrix_per_atom() {
    let stores = |k: &Kernel| k.ir().count_ops(|op| matches!(op, Op::StMatrix { .. }));
    assert_eq!(stores(&feature_kernels::stmatrix_forms().kernel), 6);
    // Each thread holds 8 f16 accumulators of its warp's 16x16: one x4 store.
    assert_eq!(stores(&feature_kernels::stmatrix_epilogue().kernel), 1);
    assert_eq!(stores(&feature_kernels::stmatrix_b8_forms().kernel), 3);
    let k = feature_kernels::setmaxnreg_warp_specialized().kernel;
    assert_eq!(k.ir().count_ops(|op| matches!(op, Op::SetMaxNReg { .. })), 3);
}

#[test]
fn stmatrix_nvvm_follows_the_llvm_version() {
    let k = feature_kernels::stmatrix_forms().kernel;
    // The newest syntax by default: shape and element type, lowered to the convergent intrinsic.
    let mlir = k.to_mlir();
    assert!(
        mlir.contains(
            "{layout = #nvvm.mma_layout<col>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = \
             #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32, i32, i32, i32"
        ),
        "{mlir}"
    );
    // LLVM 21 has only the layout (and lowers it to inline assembly).
    let old = k.to_mlir_with(&MlirOptions { llvm_major: Some(21), ..Default::default() });
    assert!(old.contains("{layout = #nvvm.mma_layout<row>} : !llvm.ptr<3>, i32\n"), "{old}");
    assert!(!old.contains("ld_st_matrix_shape"), "{old}");
    let b8 = feature_kernels::stmatrix_b8_forms().kernel;
    assert_eq!(b8.min_llvm_version(), Some(22));
    let mlir = b8.to_mlir();
    assert!(
        mlir.contains("shape = #nvvm.ld_st_matrix_shape<m = 16, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b8>")
    );
    let inline = b8.to_mlir_with(&MlirOptions { mma_inline_ptx: true, ..Default::default() });
    assert!(inline.contains("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [$0], {$1, $2, $3, $4};"), "{inline}");
    let k = feature_kernels::setmaxnreg_warp_specialized().kernel;
    let mlir = k.to_mlir();
    assert!(mlir.contains("nvvm.setmaxregister decrease 40"), "{mlir}");
    assert!(mlir.contains("nvvm.setmaxregister increase 232"), "{mlir}");
}

/// The PTX of the toolchain `CUTE_MLIR_OPT` names (LLVM 21 and 23 on the Mac, 22 on the pod).
#[test]
fn stmatrix_and_setmaxnreg_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let compile = |k: &Kernel, arch: Arch| {
        let options = feature_kernels::options_for(k, &tc).gpu_arch(arch);
        k.compile_with(&tc, &options)
            .unwrap_or_else(|e| panic!("{}: {e}", k.name()))
            .ptx()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let ptx = compile(&feature_kernels::stmatrix_forms().kernel, Arch::Sm90);
    // LLVM 21 writes the count before the shape (`.x4.m8n8`), which ptxas accepts too.
    for (num, trans) in [(1, ""), (2, ""), (4, ""), (1, ".trans"), (2, ".trans"), (4, ".trans")] {
        let a = format!("stmatrix.sync.aligned.m8n8.x{num}{trans}.shared.b16");
        let b = format!("stmatrix.sync.aligned.x{num}{trans}.m8n8.shared.b16");
        assert!(ptx.contains(&a) || ptx.contains(&b), "no {a}:\n{ptx}");
    }
    let ptx = compile(&feature_kernels::stmatrix_epilogue().kernel, Arch::Sm90);
    assert!(ptx.contains("stmatrix.sync.aligned"), "{ptx}");
    let ptx = compile(&feature_kernels::setmaxnreg_warp_specialized().kernel, Arch::Sm90a);
    assert!(ptx.contains("setmaxnreg.dec.sync.aligned.u32 40;"), "{ptx}");
    assert!(ptx.contains("setmaxnreg.inc.sync.aligned.u32 232;"), "{ptx}");
    assert!(ptx.contains("setmaxnreg.dec.sync.aligned.u32 160;"), "{ptx}");
    // The register bound setmaxnreg needs reaches the PTX.
    assert!(ptx.contains(".reqntid 384, 1, 1"), "{ptx}");
    let ptx = compile(&feature_kernels::stmatrix_b8_forms().kernel, Arch::Sm100a);
    for num in [1, 2, 4] {
        assert!(ptx.contains(&format!("stmatrix.sync.aligned.m16n8.x{num}.trans.shared.b8")), "x{num}:\n{ptx}");
    }
}

#[test]
fn setmaxnreg_needs_an_architecture_specific_target() {
    let k = feature_kernels::setmaxnreg_warp_specialized().kernel;
    let e = cute::validate::check_target(k.ir(), Arch::Sm90).unwrap_err().to_string();
    assert!(e.contains("setmaxnreg") && e.contains("sm_90a"), "{e}");
    assert!(cute::validate::check_target(k.ir(), Arch::Sm90a).is_ok());
    assert!(cute::validate::check_target(k.ir(), Arch::Sm100a).is_ok());
    let b8 = feature_kernels::stmatrix_b8_forms().kernel;
    let e = cute::validate::check_target(b8.ir(), Arch::Sm90a).unwrap_err().to_string();
    assert!(e.contains("stmatrix.m16n8"), "{e}");
    let e = cute::validate::check_target(feature_kernels::stmatrix_forms().kernel.ir(), Arch::Sm89).unwrap_err();
    assert!(e.to_string().contains("stmatrix"), "{e}");
}

#[test]
fn initial_register_budget() {
    assert_eq!(cute::sim::setmaxnreg_initial_budget(384), 168);
    assert_eq!(cute::sim::setmaxnreg_initial_budget(128), 248);
    assert_eq!(cute::sim::setmaxnreg_initial_budget(256), 248);
    assert_eq!(cute::sim::setmaxnreg_initial_budget(1024), 64);
}

/// stmatrix by half a warp.
#[test]
fn stmatrix_by_a_partial_warp() {
    let k = Kernel::builder("partial_stmatrix").block_dim([32, 1, 1]).build(|| {
        let smem = make_smem_tensor::<F16>(Layout::col_major((8, 8)));
        let lane = thread_idx().x;
        let atom = make_copy_atom::<F16>(CopyOp::StMatrix { num: 1, trans: false });
        let tiled = make_tiled_copy(&atom, atom.layout_ref_tv(), (8, 8));
        let frag = make_fragment::<F16>(2);
        frag.fill(F16::from_f32(1.0));
        if_(lane.lt(16), || copy(&tiled, &tiled.retile(&frag), &tiled.get_slice(lane).partition_d(&smem)));
    });
    let e = sim_err(&k, 32, &mut []);
    assert!(e.message.contains("stmatrix executed by 16 of the 32 lanes of warp 0"), "{e}");
}

#[test]
fn stmatrix_m16n8_must_be_transposed() {
    let msg = build_panic(|| {
        Kernel::build("untransposed", || {
            let _ = make_copy_atom::<u8>(CopyOp::StMatrix16x8x8b { num: 2, trans: false });
        });
    });
    assert!(msg.contains("StMatrix16x8x8bOp must be transposed"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("wide", || {
            let _ = make_copy_atom::<f64>(CopyOp::StMatrix { num: 4, trans: false });
        });
    });
    assert!(msg.contains("stores 16-bit elements"), "{msg}");
    // 32-bit elements go one to a register, untransposed only.
    let msg = build_panic(|| {
        Kernel::build("wide_transposed", || {
            let _ = make_copy_atom::<f32>(CopyOp::StMatrix { num: 4, trans: true });
        });
    });
    assert!(msg.contains("only untransposed"), "{msg}");
}

#[test]
fn setmaxnreg_counts_are_checked_when_built() {
    for (n, inc) in [(100u32, true), (16, false), (264, true)] {
        let msg = build_panic(|| {
            Kernel::builder("bad_count")
                .block_dim([128, 1, 1])
                .build(|| if inc { setmaxregister_increase(n) } else { setmaxregister_decrease(n) });
        });
        assert!(msg.contains(&format!("({n}): the register count is a multiple of 8 from 24 to 256")), "{msg}");
        assert!(msg.contains("tests/stmatrix.rs"), "{msg}");
    }
    // Without a launch bound setmaxnreg has no effect: refused.
    let msg = build_panic(|| {
        Kernel::build("unbounded", || setmaxregister_decrease(40));
    });
    assert!(msg.contains("launch bound"), "{msg}");
    // The deprecated names issue the same instruction.
    #[allow(deprecated)]
    let k = Kernel::builder("deprecated").block_dim([128, 1, 1]).build(|| {
        warpgroup_reg_dealloc(40);
        barrier(1, 128);
        warpgroup_reg_alloc(64);
    });
    assert_eq!(k.ir().count_ops(|op| matches!(op, Op::SetMaxNReg { .. })), 2);
    k.simulate(block(128), &mut []).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn setmaxnreg_by_part_of_a_warpgroup() {
    // Half a warp.
    let k = Kernel::builder("half_warp").block_dim([128, 1, 1]).build(|| {
        if_(thread_idx().x.lt(16), || setmaxregister_decrease(40));
    });
    let e = sim_err(&k, 128, &mut []);
    assert!(e.message.contains("setmaxnreg.dec 40 executed by 16 of the 32 lanes of warp 0"), "{e}");
    // Two of the four warps: the others never come.
    let k = Kernel::builder("two_warps").block_dim([128, 1, 1]).build(|| {
        if_(warp_idx().lt(2), || setmaxregister_decrease(40));
    });
    let e = sim_err(&k, 128, &mut []);
    assert!(e.message.contains("deadlock"), "{e}");
    assert!(e.message.contains("which warps 2..=3 of warpgroup 0 have not executed"), "{e}");
    // Warps of one warpgroup disagree.
    let k = Kernel::builder("disagree").block_dim([128, 1, 1]).build(|| {
        if_else(warp_idx().lt(2), || setmaxregister_decrease(40), || setmaxregister_decrease(48));
    });
    let e = sim_err(&k, 128, &mut []);
    assert!(e.message.contains("the warps of a warpgroup execute the same setmaxnreg"), "{e}");
    // Not a whole number of warpgroups.
    let k = Kernel::builder("ragged").block_dim([96, 1, 1]).build(|| setmaxregister_decrease(40));
    let e = sim_err(&k, 96, &mut []);
    assert!(e.message.contains("needs whole warpgroups of 128 threads"), "{e}");
}

#[test]
fn setmaxnreg_without_a_barrier_between() {
    let k = Kernel::builder("back_to_back").block_dim([128, 1, 1]).build(|| {
        setmaxregister_decrease(200);
        setmaxregister_decrease(120);
    });
    let e = sim_err(&k, 128, &mut []);
    assert!(e.message.contains("before it has synchronized with warp"), "{e}");
    assert!(e.to_string().contains("tests/stmatrix.rs"), "{e}");
    // A barrier over the warpgroup between them is enough.
    let k = Kernel::builder("separated").block_dim([128, 1, 1]).build(|| {
        setmaxregister_decrease(200);
        barrier(1, 128);
        setmaxregister_decrease(120);
    });
    k.simulate(block(128), &mut []).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn setmaxnreg_budgets() {
    // An increase that no decrease ever pays for: a deadlock naming the warps.
    let k = Kernel::builder("unserved").block_dim([256, 1, 1]).build(|| {
        if_(warp_idx().lt(4), || setmaxregister_increase(256));
    });
    let e = sim_err(&k, 256, &mut []);
    assert!(e.message.contains("deadlock"), "{e}");
    assert!(e.message.contains("can never be served"), "{e}");
    assert!(e.message.contains("needs 1024 registers from the block's pool, which has 0"), "{e}");
    // An increase below the current budget.
    let k = Kernel::builder("shrinking_inc").block_dim([128, 1, 1]).build(|| setmaxregister_increase(232));
    let e = sim_err(&k, 128, &mut []);
    assert!(e.message.contains("already 248 registers per thread"), "{e}");
    // A decrease above it.
    let k = Kernel::builder("growing_dec").block_dim([384, 1, 1]).build(|| setmaxregister_decrease(200));
    let e = sim_err(&k, 384, &mut []);
    assert!(e.message.contains("only 168 registers per thread"), "{e}");
    // The increase waits for the decrease, whichever runs first.
    let k = Kernel::builder("waits").block_dim([256, 1, 1]).build(|| {
        if_else(warp_idx().lt(4), || setmaxregister_increase(256), || setmaxregister_decrease(240));
    });
    k.simulate_all_orders(block(256), &mut [], &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
}

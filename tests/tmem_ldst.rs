//! TMEM loads and stores (roadmap part m10-tmem-ld-st): the `tcgen05.ld`/`tcgen05.st` kernels
//! in the simulator (every schedule), with `mlir-opt` and in the MLIR text; the simulator's
//! misuse reports; and (where `ptxas` is on PATH) their sm_100a PTX.

mod feature_kernels;

use cute::prelude::*;
use cute::tcgen05::{Ld16x64bOp, Ld32x32bOp, Repetition, St32x32bOp};

#[test]
fn tmem_ldst_kernels_match_the_rust_reference() {
    for case in feature_kernels::tmem_ldst_cases() {
        let mut args = feature_kernels::inputs_with_len(&case.kernel, true, 256);
        case.kernel
            .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        (case.check)(&args);
    }
}

#[test]
fn tmem_ldst_kernels_compile_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for case in feature_kernels::tmem_ldst_cases() {
        let options = feature_kernels::options_for(&case.kernel, &tc).gpu_arch(Arch::Sm100a);
        let compiled = case.kernel.compile_with(&tc, &options).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let ptx = compiled.ptx();
        let want: &[&str] = match case.name {
            "tmem_copy_f32" => &[
                "tcgen05.ld.sync.aligned.32x32b.x64.b32",
                "tcgen05.st.sync.aligned.16x256b.x2.b32",
                "tcgen05.wait::ld.sync.aligned",
                "tcgen05.wait::st.sync.aligned",
                "tcgen05.fence::before_thread_sync",
                "tcgen05.fence::after_thread_sync",
            ],
            "tmem_copy_f16" => &["tcgen05.st.sync.aligned.32x32b.x64.b32", "tcgen05.ld.sync.aligned.16x64b.x8.b32"],
            _ => &[
                "tcgen05.ld.sync.aligned.16x64b.x16.b32",
                "tcgen05.ld.sync.aligned.16x128b.x8.b32",
                "tcgen05.ld.sync.aligned.16x256b.x4.b32",
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32",
                "tcgen05.ld.sync.aligned.32x32b.x128.b32",
                "tcgen05.ld.sync.aligned.32x32b.x1.b32",
                "tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32",
                "tcgen05.st.sync.aligned.16x128b.x1.b32",
                "tcgen05.st.sync.aligned.16x32bx2.x2.b32",
                "tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32",
                "tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32",
            ],
        };
        for w in want {
            assert!(ptx.contains(w), "{}: no `{w}` in the PTX", case.name);
        }
        // The half split of .16x32bx2: as many columns as the first access spans.
        if case.name == "tmem_ldst_shapes" {
            assert!(ptx.contains("16x32bx2.x16.b32") && ptx.contains("], 16;"), "{}", case.name);
        }
    }
}

#[test]
fn tmem_ldst_mlir_and_fences() {
    let text = feature_kernels::tmem_copy_f32().kernel.to_mlir();
    for w in [
        "nvvm.tcgen05.ld",
        "nvvm.tcgen05.st",
        "{shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64xi32>",
        "nvvm.tcgen05.wait <load>",
        "nvvm.tcgen05.wait <store>",
    ] {
        assert!(text.contains(w), "no `{w}` in\n{text}");
    }
    // Every named-barrier sync has the thread-sync fences around it.
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let syncs: Vec<usize> =
        (0..lines.len()).filter(|&i| lines[i].contains("llvm.nvvm.barrier.cta.sync.aligned.count")).collect();
    assert!(!syncs.is_empty());
    for i in syncs {
        assert_eq!(lines[i - 1], "nvvm.tcgen05.fence <before>", "{text}");
        assert_eq!(lines[i + 1], "nvvm.tcgen05.fence <after>", "{text}");
    }
    // A kernel that only allocates is emitted as before: no fences.
    let alloc = feature_kernels::tmem_alloc_32().kernel.to_mlir();
    assert!(!alloc.contains("tcgen05.fence"), "{alloc}");
    // One register: an i32 before LLVM 23, a vector<1xi32> from LLVM 23.
    let shapes = feature_kernels::tmem_ldst_shapes().kernel;
    let old = shapes.to_mlir_with(&MlirOptions { llvm_major: Some(21), ..Default::default() });
    assert!(old.contains("{shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : i32"), "{old}");
    let new = shapes.to_mlir_with(&MlirOptions { llvm_major: Some(23), ..Default::default() });
    assert!(new.contains("{shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<1xi32>"), "{new}");
}

#[test]
fn tmem_ldst_target_requirements() {
    let kernel = feature_kernels::tmem_copy_f32().kernel;
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("tcgen05"), "{e}");
}

/// A 32-bit view of 32 TMEM lanes at one column, the layout of a `.32x32b.x1` access.
fn lanes_view(p: &Ptr<u32>, lane: Int32, col: i32) -> Tensor<u32> {
    make_tensor(p.clone() + (lane * (1 << 16) + col), Layout::from_modes([Layout::new(32, 1 << 16)]))
}

/// The simulator's report for `body` in a kernel of `threads` threads that allocates 32
/// columns (warp 0) and frees them at the end.
fn sim_report(threads: u32, body: impl FnOnce(&Ptr<u32>, &Tensor<u32>) + 'static) -> String {
    let kernel = Kernel::builder("tmem_ldst_misuse").block_dim([threads, 1, 1]).target(Arch::Sm100a).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(threads as usize));
        let slot = alloc_smem::<u32>(1, 4);
        if_(warp_idx().eq(0), || alloc_tmem(32, &slot, None, Arch::Sm100a));
        sync_threads();
        let p = retrieve_tmem_ptr::<u32>(16, &slot);
        body(&p, &out);
        sync_threads();
        if_(warp_idx().eq(0), || dealloc_tmem(&p, 32, None, Arch::Sm100a));
    });
    let mut args = vec![KernelArg::buffer(&vec![0u32; threads as usize])];
    match kernel.simulate(LaunchConfig::new([1, 1, 1], [threads, 1, 1]), &mut args) {
        Ok(_) => panic!("expected a simulator report"),
        Err(e) => e.to_string(),
    }
}

fn ld1() -> CopyAtom<u32> {
    make_copy_atom::<u32>(Ld32x32bOp::new(Repetition::X1))
}

fn st1() -> CopyAtom<u32> {
    make_copy_atom::<u32>(St32x32bOp::new(Repetition::X1))
}

/// Warp `w` stores its thread index to its quadrant's lanes, column 0, and waits.
fn store_quadrant(p: &Ptr<u32>) {
    let frag = make_fragment::<u32>(1);
    frag.store(0, thread_idx().x.to::<u32>());
    copy(&st1(), &frag, &lanes_view(p, warp_idx() * 32, 0));
    fence_view_async_tmem_store();
}

#[test]
fn reading_a_pending_load_register_is_reported() {
    let e = sim_report(32, |p, out| {
        store_quadrant(p);
        let frag = make_fragment::<u32>(1);
        copy(&ld1(), &lanes_view(p, Int32::from(0), 0), &frag);
        out.store(thread_idx().x, frag.load(0));
        fence_view_async_tmem_load();
    });
    assert!(e.contains("before a tcgen05.wait::ld"), "{e}");
}

#[test]
fn another_warps_quadrant_is_reported() {
    let e = sim_report(64, |p, _| {
        let frag = make_fragment::<u32>(1);
        frag.store(0, 1u32);
        // Every warp at lanes 0 to 31: warp 1 may reach only 32 to 63.
        copy(&st1(), &frag, &lanes_view(p, Int32::from(0), 0));
        fence_view_async_tmem_store();
    });
    assert!(e.contains("may access only lanes 32 to 63"), "{e}");
}

#[test]
fn tmem_outside_the_allocation_is_reported() {
    let e = sim_report(32, |p, _| {
        let frag = make_fragment::<u32>(1);
        frag.store(0, 1u32);
        copy(&st1(), &frag, &lanes_view(p, Int32::from(0), 32));
        fence_view_async_tmem_store();
    });
    assert!(e.contains("not in a live TMEM allocation"), "{e}");
}

#[test]
fn loading_unwritten_tmem_is_reported() {
    let e = sim_report(32, |p, out| {
        let frag = make_fragment::<u32>(1);
        copy(&ld1(), &lanes_view(p, Int32::from(0), 0), &frag);
        fence_view_async_tmem_load();
        out.store(thread_idx().x, frag.load(0));
    });
    assert!(e.contains("which nothing wrote"), "{e}");
}

#[test]
fn a_divergent_load_is_reported() {
    let e = sim_report(32, |p, out| {
        store_quadrant(p);
        if_(lane_idx().lt(16), || {
            let frag = make_fragment::<u32>(1);
            copy(&ld1(), &lanes_view(p, Int32::from(0), 0), &frag);
            fence_view_async_tmem_load();
            out.store(thread_idx().x, frag.load(0));
        });
    });
    assert!(e.contains("tcgen05.ld executed by 16 of the 32 lanes"), "{e}");
}

#[test]
fn dealloc_before_the_store_wait_is_reported() {
    let e = sim_report(32, |p, _| {
        let frag = make_fragment::<u32>(1);
        frag.store(0, 1u32);
        copy(&st1(), &frag, &lanes_view(p, Int32::from(0), 0));
    });
    assert!(e.contains("tcgen05.dealloc") && e.contains("tcgen05.wait::st"), "{e}");
}

#[test]
fn tmem_atom_build_time_checks() {
    let e = std::panic::catch_unwind(|| make_copy_atom::<f32>(cute::tcgen05::Ld16x256bOp::new(Repetition::X64)))
        .map(|_| ())
        .unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("at most 128"), "{e}");
    let e = std::panic::catch_unwind(|| {
        make_copy_atom::<f32>(Ld16x64bOp::new(Repetition::X1).with_pack(cute::tcgen05::Pack::Pack16bIn32b))
    })
    .map(|_| ())
    .unwrap_err();
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(e.contains("at most 16 bits"), "{e}");
}

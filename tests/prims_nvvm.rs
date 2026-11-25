// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//! The primitives operations without a `cute::arch`
//! counterpart (`bar.red`, the rest of the mbarrier family, the markers, `exit`,
//! `dp4a`/`dp2a`, `st.bulk`, `tcgen05.shift`) on the simulator against plain Rust, in the
//! PTX for sm_80 to sm_100a (and through `ptxas` when one is on PATH), and (with
//! `--features cuda`) on a GPU. The first two tests port CuTe DSL's BSD-3 examples
//! `experimental/primitives/barrier_cta_red.py` and `mbarrier_lifecycle.py`.

use cute::prelude::*;
use cute::prims;
use cute::prims::{BarrierRedux, DotAccumulateType as Dot};

fn run(kernel: &Kernel, threads: u32, args: &mut [KernelArg]) -> Result<SimReport, String> {
    kernel.simulate(LaunchConfig::new([1, 1, 1], [threads, 1, 1]), args).map_err(|e| e.to_string())
}

/// The PTX of `kernel` for `arch`, assembled with `ptxas` when one is on PATH; `None`
/// without an `mlir-opt`.
fn ptx_for(kernel: &Kernel, arch: Arch) -> Option<String> {
    let tc = Toolchain::discover()?;
    let name = kernel.ir().name.clone();
    let ptx = kernel
        .compile_with(&tc, &CompileOptions::new().gpu_arch(arch))
        .unwrap_or_else(|e| panic!("{name} for {arch:?}: {e}"))
        .ptx()
        .to_string();
    assert!(!ptx.contains(".extern .func llvm.nvvm"), "{name}: an unknown intrinsic became a call:\n{ptx}");
    let ptxas =
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("ptxas")).find(|f| f.is_file()));
    if let Some(ptxas) = ptxas {
        let base = std::env::temp_dir().join(format!("cute-prims-nvvm-{}-{name}-{arch:?}", std::process::id()));
        let (src, cubin) = (base.with_extension("ptx"), base.with_extension("cubin"));
        std::fs::write(&src, &ptx).unwrap();
        let sm = format!("-arch={}", arch.chip());
        let out = std::process::Command::new(ptxas).arg(&sm).arg(&src).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            out.status.success(),
            "{name}: ptxas {sm}: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{name}: ptxas {sm} ok");
    }
    Some(ptx)
}

// ---------------------------------------------------------------------------------------
// barrier_cta_red.py: three thread-block predicate reductions on named barriers 0, 1, 2.
// ---------------------------------------------------------------------------------------

fn barrier_cta_red_kernel(threads: u32) -> Kernel {
    Kernel::builder("prims_barrier_cta_red").block_dim([threads, 1, 1]).build(move || {
        let out = param_tensor::<i32>("out", Layout::col_major(4));
        let t = thread_idx().x;
        let n = Some(Int32::constant(threads as i32));
        let any_lane0 = prims::barrier_cta_red_or(t.eq(0), 0, n);
        let all_in_range = prims::barrier_cta_red_and(t.lt(threads as i32), 1, n);
        let all_nonzero = prims::barrier_cta_red_and(t.ne(0), 2, n);
        // A cute-rs addition: a count over a barrier with no thread count.
        let thirds = prims::barrier_cta_red((t % 3i32).ne(0), 3, BarrierRedux::Popc, None);
        if_(t.eq(0), || {
            out.store(0, any_lane0.to::<i32>());
            out.store(1, all_in_range.to::<i32>());
            out.store(2, all_nonzero.to::<i32>());
            out.store(3, thirds.to::<i32>());
        });
    })
}

#[test]
fn barrier_cta_red_example() {
    for threads in [32u32, 64, 128] {
        let k = barrier_cta_red_kernel(threads);
        let mut args = vec![KernelArg::buffer(&[0i32; 4])];
        let report = run(&k, threads, &mut args).unwrap();
        // Threads t with t % 3 != 0 count as true.
        let thirds = (0..threads).filter(|t| t % 3 != 0).count() as i32;
        assert_eq!(args[0].to_vec::<i32>(), vec![1, 1, 0, thirds], "{threads} threads");
        assert_eq!(report.barriers, 4);
    }
    let k = barrier_cta_red_kernel(64);
    for arch in [Arch::Sm80, Arch::Sm89] {
        if let Some(ptx) = ptx_for(&k, arch) {
            assert!(ptx.contains("bar.red.popc.u32") || ptx.contains("barrier.red.popc"), "{ptx}");
            assert!(ptx.contains(".and.pred") && ptx.contains(".or.pred"), "{ptx}");
        }
    }
}

#[test]
fn barrier_cta_red_mixing_and_divergence_are_reported() {
    // Warps 0 and 1 reduce on barrier 1 with different operations.
    let k = Kernel::builder("prims_bar_red_mixed").block_dim([64, 1, 1]).build(|| {
        let w = warp_idx();
        if_(w.eq(0), || {
            prims::barrier_cta_red_and(true, 1, Some(Int32::constant(64)));
        });
        if_(w.eq(1), || {
            prims::barrier_cta_red_or(true, 1, Some(Int32::constant(64)));
        });
    });
    let e = run(&k, 64, &mut []).unwrap_err();
    assert!(e.contains("barrier_cta_red (and)") && e.contains("barrier_cta_red (or)"), "{e}");
}

// ---------------------------------------------------------------------------------------
// mbarrier_lifecycle.py: init, arrive (state token), time-limited wait, inval; then the
// memory is initialised again and reused.
// ---------------------------------------------------------------------------------------

fn lifecycle_kernel() -> Kernel {
    Kernel::builder("prims_mbarrier_lifecycle").block_dim([1, 1, 1]).target(Arch::Sm90a).build(|| {
        let dst = param_tensor::<i32>("dst", Layout::col_major(2));
        let mbar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&mbar, 1);
        mbarrier_init_fence();
        sync_threads();
        let state = prims::nvvm::mbarrier_arrive(&mbar);
        let done = prims::mbarrier_try_wait_timelimit(&mbar, state, 10_000_000);
        dst.store(0, 0i32);
        if_(done, || dst.store(0, 1i32));
        prims::mbarrier_inval(&mbar);
        // Re-initialise the same memory and use it again.
        mbarrier_init(&mbar, 1);
        let state = prims::nvvm::mbarrier_arrive(&mbar);
        dst.store(1, prims::nvvm::mbarrier_test_wait(&mbar, state).to::<i32>());
        prims::mbarrier_inval(&mbar);
    })
}

#[test]
fn mbarrier_lifecycle_example() {
    let k = lifecycle_kernel();
    let mut args = vec![KernelArg::buffer(&[7i32; 2])];
    run(&k, 1, &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![1, 1]);
    cute::validate::check_target(k.ir(), Arch::Sm90a).unwrap();
    let e = cute::validate::check_target(k.ir(), Arch::Sm80).unwrap_err().message;
    assert!(e.contains("sm_90"), "{e}");
    if let Some(ptx) = ptx_for(&k, Arch::Sm90a) {
        for want in ["mbarrier.inval.shared.b64", "mbarrier.try_wait.shared::cta.b64", "mbarrier.arrive.shared.b64"] {
            assert!(ptx.contains(want), "{want}:\n{ptx}");
        }
    }
}

// ---------------------------------------------------------------------------------------
// The other mbarrier operations, by one thread, against the PTX ISA's phase arithmetic.
// ---------------------------------------------------------------------------------------

fn mbarrier_ops_kernel() -> Kernel {
    Kernel::builder("prims_mbarrier_ops").block_dim([1, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(8));
        let mbar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&mbar, 3);
        // Phase 0 waits for 3: two non-completing arrivals, then a drop completes it.
        let s0 = prims::mbarrier_arrive_nocomplete(&mbar, 2);
        out.store(0, prims::nvvm::mbarrier_test_wait(&mbar, s0).to::<i32>());
        let s0b = prims::mbarrier_arrive_drop(&mbar);
        out.store(1, prims::nvvm::mbarrier_test_wait(&mbar, s0b).to::<i32>());
        // Phase 1 waits for 2 (one dropped): a drop-nocomplete leaves 1, for every later phase.
        let s1 = prims::mbarrier_arrive_drop_nocomplete(&mbar, 1);
        out.store(2, prims::nvvm::mbarrier_test_wait(&mbar, s1).to::<i32>());
        let s1b = prims::nvvm::mbarrier_arrive(&mbar);
        out.store(3, prims::nvvm::mbarrier_test_wait(&mbar, s1b).to::<i32>());
        // Phase 2 completes with one arrival.
        let s2 = prims::nvvm::mbarrier_arrive(&mbar);
        out.store(4, prims::nvvm::mbarrier_test_wait(&mbar, s2).to::<i32>());
        // Phase parity of the current phase (3) is not over; of phase 2 it is.
        out.store(5, prims::nvvm::mbarrier_test_wait(&mbar, 1).to::<i32>());
        out.store(6, prims::nvvm::mbarrier_test_wait(&mbar, 0).to::<i32>());
        prims::mbarrier_inval(&mbar);
    })
}

#[test]
fn arrive_drop_nocomplete_and_complete_tx() {
    let k = mbarrier_ops_kernel();
    let mut args = vec![KernelArg::buffer(&[9i32; 8])];
    run(&k, 1, &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![0, 1, 0, 1, 1, 0, 1, 9]);

    // expect_tx and complete_tx (sm_90): the phase completes once its bytes are in.
    let k = Kernel::builder("prims_mbarrier_tx").block_dim([1, 1, 1]).target(Arch::Sm90a).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(3));
        let mbar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&mbar, 1);
        let s = prims::mbarrier_arrive_drop_expect_tx(&mbar, 64);
        out.store(0, prims::nvvm::mbarrier_test_wait(&mbar, s).to::<i32>());
        prims::mbarrier_complete_tx(&mbar, 48);
        out.store(1, prims::nvvm::mbarrier_test_wait(&mbar, s).to::<i32>());
        prims::mbarrier_complete_tx(&mbar, 16);
        out.store(2, prims::nvvm::mbarrier_try_wait(&mbar, s, None).to::<i32>());
    });
    let mut args = vec![KernelArg::buffer(&[9i32; 3])];
    run(&k, 1, &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>(), vec![0, 0, 1]);
    let e = cute::validate::check_target(k.ir(), Arch::Sm89).unwrap_err().message;
    assert!(e.contains("sm_90"), "{e}");
    if let Some(ptx) = ptx_for(&k, Arch::Sm90a) {
        for want in ["mbarrier.arrive_drop.expect_tx.shared::cta.b64", "mbarrier.complete_tx.shared::cta.b64"] {
            assert!(ptx.contains(want), "{want}:\n{ptx}");
        }
    }
}

#[test]
fn mbarrier_misuse_is_reported() {
    // An operation after inval names the inval.
    let k = Kernel::builder("prims_after_inval").block_dim([1, 1, 1]).build(|| {
        let mbar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&mbar, 1);
        prims::mbarrier_inval(&mbar);
        prims::nvvm::mbarrier_arrive(&mbar);
    });
    let e = run(&k, 1, &mut []).unwrap_err();
    assert!(e.contains("mbarrier_inval invalidated at") && e.contains("prims_nvvm.rs"), "{e}");

    // A .noComplete arrival that completes the phase.
    let k = Kernel::builder("prims_nocomplete_completes").block_dim([1, 1, 1]).build(|| {
        let mbar = alloc_smem::<u64>(1, 8);
        mbarrier_init(&mbar, 2);
        prims::mbarrier_arrive_nocomplete(&mbar, 2);
    });
    let e = run(&k, 1, &mut []).unwrap_err();
    assert!(e.contains(".noComplete"), "{e}");

    // A warp exits owing its arrival: the waiting warp hangs, and the report names the exit.
    let k = Kernel::builder("prims_exit_owes").block_dim([64, 1, 1]).build(|| {
        let mbar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&mbar, 1));
        sync_threads();
        if_(warp_idx().eq(1), || {
            prims::exit();
            if_(lane_idx().eq(0), || {
                prims::nvvm::mbarrier_arrive(&mbar);
            });
        });
        if_(warp_idx().eq(0), || {
            cute::control::while_carry(
                (),
                |()| !prims::nvvm::mbarrier_test_wait(&mbar, 0),
                |()| prims::nanosleep(100u32),
            );
        });
    });
    let e = run(&k, 64, &mut []).unwrap_err();
    assert!(e.contains("exited at the exit at") && e.contains("prims_nvvm.rs"), "{e}");
}

fn exit_kernel() -> Kernel {
    Kernel::builder("prims_exit").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        if_(warp_idx().eq(1), prims::exit);
        out.store(t, 1i32);
    })
}

#[test]
fn exit_ends_whole_warps() {
    let k = exit_kernel();
    let mut args = vec![KernelArg::buffer(&[0i32; 64])];
    let report = run(&k, 64, &mut args).unwrap();
    let want: Vec<i32> = (0..64).map(|t| (t < 32) as i32).collect();
    assert_eq!(args[0].to_vec::<i32>(), want);
    assert_eq!(report.exits, 1);
    // By part of a warp it is not simulated.
    let k = Kernel::builder("prims_exit_lanes").block_dim([32, 1, 1]).build(|| {
        if_(lane_idx().lt(4), prims::exit);
    });
    let e = run(&k, 32, &mut []).unwrap_err();
    assert!(e.contains("not simulated"), "{e}");
    if let Some(ptx) = ptx_for(&k, Arch::Sm80) {
        assert!(ptx.contains("exit;"), "{ptx}");
    }
}

// ---------------------------------------------------------------------------------------
// Per-thread operations.
// ---------------------------------------------------------------------------------------

fn dp4a_ref(a: u32, at: Dot, b: u32, bt: Dot, c: i32) -> i32 {
    let el = |w: u32, i: u32, t: Dot| {
        let byte = (w >> (8 * i)) as u8;
        if t == Dot::Signed { byte as i8 as i64 } else { byte as i64 }
    };
    (c as i64 + (0..4).map(|i| el(a, i, at) * el(b, i, bt)).sum::<i64>()) as i32
}

fn dp2a_ref(a: u32, at: Dot, b: u32, bt: Dot, c: i32, hi: bool) -> i32 {
    let half = |i: u32| {
        let h = (a >> (16 * i)) as u16;
        if at == Dot::Signed { h as i16 as i64 } else { h as i64 }
    };
    let byte = |i: u32| {
        let x = (b >> (8 * (i + 2 * hi as u32))) as u8;
        if bt == Dot::Signed { x as i8 as i64 } else { x as i64 }
    };
    (c as i64 + half(0) * byte(0) + half(1) * byte(1)) as i32
}

const KINDS: [(Dot, Dot); 4] = [
    (Dot::Signed, Dot::Signed),
    (Dot::Signed, Dot::Unsigned),
    (Dot::Unsigned, Dot::Signed),
    (Dot::Unsigned, Dot::Unsigned),
];

fn dot_kernel() -> Kernel {
    Kernel::builder("prims_dot").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<u32>("a", Layout::col_major(32));
        let b = param_tensor::<u32>("b", Layout::col_major(32));
        let c = param_tensor::<i32>("c", Layout::col_major(32));
        let out = param_tensor::<i32>("out", Layout::row_major((32, 12)));
        let t = thread_idx().x;
        let (x, y, z) = (a.load(t), b.load(t), c.load(t));
        for (i, (at, bt)) in KINDS.into_iter().enumerate() {
            out.store((t, 3 * i as i32), prims::dot_accumulate_4way(x, at, y, bt, z));
            out.store((t, 3 * i as i32 + 1), prims::dot_accumulate_2way(x, at, y, bt, z, false));
            out.store((t, 3 * i as i32 + 2), prims::dot_accumulate_2way(x, at, y, bt, z, true));
        }
    })
}

fn dot_inputs() -> (Vec<u32>, Vec<u32>, Vec<i32>) {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 32) as u32
    };
    let a: Vec<u32> = (0..32).map(|_| next()).collect();
    let b: Vec<u32> = (0..32).map(|_| next()).collect();
    let c: Vec<i32> = (0..32).map(|_| next() as i32).collect();
    (a, b, c)
}

fn dot_expected(a: &[u32], b: &[u32], c: &[i32]) -> Vec<i32> {
    let mut want = Vec::new();
    for t in 0..32 {
        for (at, bt) in KINDS {
            want.push(dp4a_ref(a[t], at, b[t], bt, c[t]));
            want.push(dp2a_ref(a[t], at, b[t], bt, c[t], false));
            want.push(dp2a_ref(a[t], at, b[t], bt, c[t], true));
        }
    }
    want
}

#[test]
fn dot_products_match_plain_rust() {
    let (a, b, c) = dot_inputs();
    let k = dot_kernel();
    let mut args =
        vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&c), KernelArg::buffer(&[0i32; 32 * 12])];
    run(&k, 32, &mut args).unwrap();
    assert_eq!(args[3].to_vec::<i32>(), dot_expected(&a, &b, &c));
    if let Some(ptx) = ptx_for(&k, Arch::Sm89) {
        assert!(ptx.contains("dp4a.s32.s32") && ptx.contains("dp4a.u32.u32"), "{ptx}");
        assert!(ptx.contains("dp2a.lo.s32.u32") && ptx.contains("dp2a.hi.u32.s32"), "{ptx}");
    }
}

fn markers_kernel() -> Kernel {
    Kernel::builder("prims_markers").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let t = thread_idx().x;
        prims::ptx_add_comment("cute-rs marker: $tile 0");
        prims::pmevent(3);
        prims::pmevent_mask(0x00f0);
        prims::nanosleep(32u32);
        prims::memory_barrier(prims::FenceScope::Cta);
        prims::memory_barrier(prims::FenceScope::Gpu);
        prims::memory_barrier(prims::FenceScope::Sys);
        let elected = prims::elect_sync(0x0000_ffffu32);
        let low = prims::elect_sync(FULL_MASK);
        out.store(t, elected.to::<u32>() + low.to::<u32>() * 2u32 + prims::mov_b32::<f32, u32>(1.0f32));
    })
}

#[test]
fn markers_fences_and_elect() {
    let k = markers_kernel();
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    let report = run(&k, 64, &mut args).unwrap();
    // Lane 0 of each warp is elected by both.
    let one = 1.0f32.to_bits();
    let want: Vec<u32> = (0..64).map(|t| one + if t % 32 == 0 { 3 } else { 0 }).collect();
    assert_eq!(args[0].to_vec::<u32>(), want);
    assert_eq!((report.markers, report.nanosleeps), (6, 2));
    if let Some(ptx) = ptx_for(&k, Arch::Sm80) {
        for want in [
            "// cute-rs marker: $tile 0",
            "pmevent.mask 0x8;",
            "pmevent.mask 0xf0;",
            "nanosleep.u32",
            "fence.sc.cta;",
            "fence.sc.gpu;",
            "fence.sc.sys;",
        ] {
            assert!(ptx.contains(want), "{want}:\n{ptx}");
        }
    }
}

// ---------------------------------------------------------------------------------------
// sm_100: st.bulk and tcgen05.shift.
// ---------------------------------------------------------------------------------------

fn st_bulk_kernel(offset: usize, size: i64) -> Kernel {
    Kernel::builder("prims_st_bulk").block_dim([32, 1, 1]).target(Arch::Sm100a).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(64));
        let smem = alloc_smem::<u32>(64, 16);
        let t = thread_idx().x;
        smem.offset(t).store(0xdead_0000u32 + t.to::<u32>());
        smem.offset(t + 32).store(0xbeef_0000u32 + t.to::<u32>());
        sync_threads();
        if_(t.eq(0), || prims::st_bulk(&smem.offset(offset), size));
        sync_threads();
        out.store(t, smem.offset(t).load());
        out.store(t + 32, smem.offset(t + 32).load());
    })
}

#[test]
fn st_bulk_zero_fills_and_is_checked() {
    let k = st_bulk_kernel(4, 96);
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    run(&k, 32, &mut args).unwrap();
    let want: Vec<u32> = (0..64u32)
        .map(|i| {
            if (4..28).contains(&i) {
                0
            } else if i < 32 {
                0xdead_0000 + i
            } else {
                0xbeef_0000 + i - 32
            }
        })
        .collect();
    assert_eq!(args[0].to_vec::<u32>(), want);
    let e = cute::validate::check_target(k.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("st.bulk"), "{e}");
    if let Some(ptx) = ptx_for(&k, Arch::Sm100a) {
        assert!(ptx.contains("st.bulk.weak.shared::cta"), "{ptx}");
    }
    // A size that is not a multiple of 16 bytes, and a start that is not 16-byte aligned.
    let e = std::panic::catch_unwind(|| st_bulk_kernel(0, 40)).map(|k| cute::validate::validate(k.ir()));
    let text = match e {
        Ok(Err(e)) => e.message,
        Err(p) => p.downcast_ref::<String>().cloned().unwrap_or_default(),
        Ok(Ok(())) => String::new(),
    };
    assert!(text.contains("multiple of 16"), "{text}");
    let e = run(&st_bulk_kernel(2, 32), 32, &mut [KernelArg::buffer(&[0u32; 64])]).unwrap_err();
    assert!(e.contains("st.bulk"), "{e}");
}

#[test]
fn tcgen05_shift_compiles_for_sm100a_only() {
    let k = Kernel::builder("prims_tcgen05_shift").block_dim([32, 1, 1]).target(Arch::Sm100a).build(|| {
        if_(thread_idx().x.eq(0), || prims::tcgen05_shift(Int32::constant(0), prims::CTAGroup::ONE));
    });
    let e = cute::validate::check_target(k.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("tcgen05.shift"), "{e}");
    let e = run(&k, 32, &mut []).unwrap_err();
    assert!(e.contains("not simulated"), "{e}");
    if let Some(ptx) = ptx_for(&k, Arch::Sm100a) {
        assert!(ptx.contains("tcgen05.shift.cta_group::1.down"), "{ptx}");
    }
}

// ---------------------------------------------------------------------------------------
// The GPU (SM89): bit-exact with the simulator.
// ---------------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn prims_on_the_gpu() {
    use cute::cuda::CudaContext;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let Some(tc) = Toolchain::discover() else { return eprintln!("skipping: no mlir-opt found") };
    let arch = ctx.arch().expect("SM80 or newer");
    let launch = |k: &Kernel, threads: u32, args: &mut [KernelArg]| {
        let compiled = k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
        let function = ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"));
        ctx.launch(k, &function, LaunchConfig::new([1, 1, 1], [threads, 1, 1]), args).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("{}: GPU ok", k.ir().name);
    };
    for threads in [32u32, 64, 128] {
        let k = barrier_cta_red_kernel(threads);
        let mut args = vec![KernelArg::buffer(&[0i32; 4])];
        launch(&k, threads, &mut args);
        let thirds = (0..threads).filter(|t| t % 3 != 0).count() as i32;
        assert_eq!(args[0].to_vec::<i32>(), vec![1, 1, 0, thirds], "{threads} threads");
    }
    let (a, b, c) = dot_inputs();
    let mut args =
        vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&c), KernelArg::buffer(&[0i32; 32 * 12])];
    launch(&dot_kernel(), 32, &mut args);
    assert_eq!(args[3].to_vec::<i32>(), dot_expected(&a, &b, &c));
    let mut args = vec![KernelArg::buffer(&[9i32; 8])];
    launch(&mbarrier_ops_kernel(), 1, &mut args);
    assert_eq!(args[0].to_vec::<i32>(), vec![0, 1, 0, 1, 1, 0, 1, 9]);
    let mut args = vec![KernelArg::buffer(&[0i32; 64])];
    launch(&exit_kernel(), 64, &mut args);
    assert_eq!(args[0].to_vec::<i32>(), (0..64).map(|t| (t < 32) as i32).collect::<Vec<_>>());
    let mut args = vec![KernelArg::buffer(&[0u32; 64])];
    launch(&markers_kernel(), 64, &mut args);
    let one = 1.0f32.to_bits();
    let want: Vec<u32> = (0..64).map(|t| one + if t % 32 == 0 { 3 } else { 0 }).collect();
    assert_eq!(args[0].to_vec::<u32>(), want);
}

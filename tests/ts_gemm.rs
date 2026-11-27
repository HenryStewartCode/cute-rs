// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The GEMM under test is the port, in examples/ts_gemm_simple.rs, of NVIDIA
// CUTLASS 4.8.1's
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   02_gemm_simple_ts/01_fp16_bf16_gemm_3.py,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and the IKET test expects its schedules' steps. That portion is under the
// BSD-3-Clause licence, and the rest of the file is cute-rs's own, under MIT
// OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.

//! The Task Scheduling tcgen05 GEMM
//! (`02_gemm_simple_ts/01_fp16_bf16_gemm_3.py`) end to end. On the
//! simulator against plain Rust: bit for bit on small integers (FP16 and
//! BF16, a tile-aligned size and a ragged one), within the FP32 summation
//! bound on random data; the TaskManager's static verification (in the
//! build) and its seeded interleavings; an IKET trace with one range per
//! task step; the sm_100a MLIR and PTX (`tcgen05.mma`, `tcgen05.ld`,
//! `cp.async.bulk.tensor`, `setmaxnreg`), written to `CUTE_TS_PTX_DIR` when
//! set for `ptxas` elsewhere. Also the two TS-layer gaps the port exposed:
//! compile-time work arguments and register budgets shared by tasks smaller
//! than a warpgroup.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/ts_gemm_simple.rs"]
mod ts_gemm_simple;

use cute::prelude::*;
use cute::task_scheduling::*;
use ts_gemm_simple as g;

#[test]
fn exact_on_small_integers_fp16_and_bf16() {
    g::run_exact::<F16>(128, 256, 128);
    g::run_exact::<BF16>(128, 256, 64);
}

#[test]
fn exact_on_a_ragged_problem() {
    // Two by two CTAs, partial in M and N; K = 136 is two full stages and a
    // partial one (TMA zero-fills); N = 300 is not a multiple of 8, so every
    // store is scalar, and N = 264 takes the vector path with a ragged tail.
    g::run_exact::<F16>(200, 300, 136);
    g::run_exact::<BF16>(130, 264, 72);
}

#[test]
fn random_data_within_the_fp32_summation_bound() {
    g::run_random::<F16>(128, 256, 192);
    g::run_random::<BF16>(150, 256, 128);
}

#[test]
fn seeded_interleavings() {
    let (m, n, k) = (128, 256, 128);
    let kernel = g::build::<F16>(m, n, k, None);
    let (av, bv) = (g::small_ints(m * k, 3), g::small_ints(n * k, 7));
    let to_t = |v: &[f32]| v.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>();
    let mut args =
        vec![KernelArg::buffer(&to_t(&av)), KernelArg::buffer(&to_t(&bv)), KernelArg::buffer(&vec![F16::ZERO; m * n])];
    let report =
        TaskManager::verify_interleavings(&kernel, g::launch(m, n), &mut args, &[5]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.tcgen05_mma_instructions, (k / 16) as u64);
    let (want, _) = g::reference(&av, &bv, m, n, k);
    let got: Vec<f32> = args[2].to_vec::<F16>().into_iter().map(F16::to_f32).collect();
    assert!(got.iter().zip(&want).all(|(g, w)| *g == *w as f32));
}

#[test]
fn iket_trace_has_one_range_per_task_step() {
    let (m, n, k) = (128, 256, 128);
    let kernel = g::build::<F16>(m, n, k, Some(IketOptions { capacity: 256, clock: IketClock::Clock64 }));
    let (av, bv) = (g::small_ints(m * k, 3), g::small_ints(n * k, 7));
    let to_t = |v: &[f32]| v.iter().map(|&x| F16::from_f32(x)).collect::<Vec<_>>();
    let mut args =
        vec![KernelArg::buffer(&to_t(&av)), KernelArg::buffer(&to_t(&bv)), KernelArg::buffer(&vec![F16::ZERO; m * n])];
    let mut module = kernel.sim_module();
    module.simulate(g::launch(m, n), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    let trace = kernel.decode_iket(&words).unwrap_or_else(|e| panic!("{e}"));
    assert!(trace.unmatched.is_empty() && !trace.overflowed, "{:?}", trace.unmatched);
    let events = |warp: u32| -> Vec<String> {
        trace.warp_records(0, warp).iter().map(|r| format!("{:?}:{}", r.kind, r.event)).collect()
    };
    let ranges = |task: &str, head: &[&str], body: &[&str], tail: &[&str]| {
        let mut want = vec![format!("Start:{task}")];
        let mut range = |s: &&str| want.extend([format!("Start:{s}"), format!("End:{s}")]);
        head.iter().for_each(&mut range);
        for _ in 0..k / 64 {
            body.iter().for_each(&mut range);
        }
        tail.iter().for_each(&mut range);
        want.push(format!("End:{task}"));
        want
    };
    let load = [
        "#1 GmemAb.compute_coords",
        "#2 SmemAb.try_acquire",
        "#3 SmemAb.acquire",
        "#4 SmemAb.tma_load",
        "#5 SmemAb.commit",
    ];
    assert_eq!(events(4), ranges("LoadTask", &["#0 SmemAb.init_load_state"], &load, &[]));
    let mma =
        ["#5 SmemAb.try_wait", "#6 SmemAb.wait", "#7 SmemAb.build_descriptors", "#8 TmemC.mma", "#9 SmemAb.release"];
    let head = [
        "#0 SmemAb.init_descriptors",
        "#1 TmemC.init_accumulator_state",
        "#2 TmemC.init_work_tile_state",
        "#3 TmemC.try_acquire",
        "#4 TmemC.acquire",
    ];
    assert_eq!(events(5), ranges("MmaTask", &head, &mma, &["#10 TmemC.commit"]));
    let mut tail = vec!["#1 TmemC.try_wait".to_string(), "#2 TmemC.wait".to_string()];
    for i in 0..8 {
        tail.push(format!("#{} TmemC.load_subtile", 3 + 2 * i));
        tail.push(format!("#{} GmemD.store", 4 + 2 * i));
    }
    tail.push("#19 TmemC.release".into());
    let tail: Vec<&str> = tail.iter().map(String::as_str).collect();
    for warp in 0..4 {
        assert_eq!(events(warp), ranges("StoreTask", &["#0 TmemC.init_store_state"], &[], &tail));
    }
    for warp in 6..8 {
        assert_eq!(events(warp), ranges("PaddingTask", &[], &[], &[]));
    }
}

fn toolchain() -> Option<Toolchain> {
    let tc = Toolchain::discover();
    if tc.is_none() {
        eprintln!("skipping the PTX: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    tc
}

#[test]
fn sm100a_mlir_and_ptx() {
    let kernel = g::build::<BF16>(256, 512, 256, None);
    let mlir = kernel.to_mlir();
    for w in ["tcgen05.mma", "tcgen05.ld", "nvvm.setmaxregister", "cp.async.bulk.tensor"] {
        assert!(mlir.contains(w), "no `{w}` in the MLIR");
    }
    let Some(tc) = toolchain() else { return };
    let compiled =
        kernel.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
    let ptx = compiled.ptx();
    if let Some(dir) = std::env::var_os("CUTE_TS_PTX_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", kernel.name())), ptx).unwrap();
    }
    let flat = ptx.split_whitespace().collect::<Vec<_>>().join(" ");
    // One setmaxnreg per warpgroup: the second warpgroup's budget is set
    // once for its three tasks.
    assert_eq!(flat.matches("setmaxnreg.dec").count(), 2);
    for w in [
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.ld.sync.aligned.32x32b.x32",
        "cp.async.bulk.tensor.2d",
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one",
        "tcgen05.alloc.cta_group::1",
        "tcgen05.dealloc.cta_group::1",
        "setmaxnreg.dec.sync.aligned.u32 40;",
        "setmaxnreg.dec.sync.aligned.u32 160;",
        "prefetch.tensormap",
    ] {
        assert!(flat.contains(w), "no `{w}` in the PTX");
    }
}

// ---------------------------------------------------------------------------
// The TS-layer gaps the port exposed
// ---------------------------------------------------------------------------

fn nothing(_: &StageInfo, _: &WorkArgs) -> WorkResults {
    WorkResults::none()
}

#[test]
fn constexpr_work_arguments_in_listings_and_diagnostics() {
    let v = TaskLocalVariable::<Int32>::scalar("v", 0, "a value");
    let src =
        MemoryResource::new("src").work(Work::consumer("load").constexpr_input("idx").returns(&v).run(nothing_v(&v)));
    let dst = MemoryResource::new("dst")
        .work(Work::producer("store").input::<Int32>("v").constexpr_input("idx").run(nothing));
    let s = schedule("subtiles", &[&src, &dst], |_, r| {
        for i in 0..2 {
            let x = r[0].call_with("load", [("idx", ArgSource::constexpr(i))]).one();
            r[1].call_with("store", [("v", ArgSource::from(x)), ("idx", ArgSource::constexpr(i))]);
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let listing = s.to_string();
    assert!(listing.contains("#0 head src.load(idx = 0) -> v"), "{listing}");
    assert!(listing.contains("#3 head dst.store(v <- #2.v, idx = 1)"), "{listing}");
    assert_eq!(s.steps()[1].inputs[1].source, Source::Constexpr(0));

    // A constant for a variable input, and a variable for a constant one.
    let err = schedule("bad", &[&src, &dst], |_, r| {
        let x = r[0].call_with("load", [("idx", ArgSource::constexpr(0))]).one();
        r[1].call_with("store", [("v", ArgSource::constexpr(1)), ("idx", ArgSource::from(x))]);
    })
    .unwrap_err();
    let kinds: Vec<IssueKind> = err.issues.iter().map(|i| i.kind).collect();
    assert_eq!(kinds, [IssueKind::BadArgument, IssueKind::BadArgument], "{err}");
}

fn nothing_v(v: &TaskLocalVariable<Int32>) -> impl Fn(&StageInfo, &WorkArgs) -> WorkResults + 'static {
    let v = v.clone();
    move |_, args| WorkResults::of(&v, Int32::constant(args.constexpr("idx") as i32))
}

/// Tasks on warps `ranges` with budgets `budgets` (empty schedules).
fn budget_manager(tasks: &[(&str, u32, u32, Option<u32>)]) -> TaskManager {
    let tasks = tasks
        .iter()
        .map(|&(name, w, n, regs)| {
            let s = schedule(name, &[], |s, _| s.domain_loop(0, 1, 1, 1, |_| {})).unwrap();
            let t = Task::new(name, &[], &[], w, n, s);
            match regs {
                Some(r) => t.setmaxnreg_decrease(r),
                None => t,
            }
        })
        .collect();
    TaskManager::new(tasks, ResourceDependencyGraph::new())
}

#[test]
fn register_budgets_shared_within_a_warpgroup() {
    // The tutorial's layout: a 4-warp store task, and a load, an MMA and a
    // padding task sharing the second warpgroup's budget.
    let ok = budget_manager(&[
        ("Store", 0, 4, Some(160)),
        ("Load", 4, 1, Some(40)),
        ("Mma", 5, 1, Some(40)),
        ("Pad", 6, 2, Some(40)),
    ]);
    ok.verify().unwrap_or_else(|e| panic!("{e}"));
    // Different budgets, or a warp without one, in a warpgroup.
    for bad in [
        budget_manager(&[("Load", 4, 1, Some(40)), ("Mma", 5, 1, Some(48)), ("Pad", 6, 2, Some(40))]),
        budget_manager(&[("Load", 4, 1, Some(40)), ("Mma", 5, 1, Some(40)), ("Pad", 6, 2, None)]),
    ] {
        let err = bad.verify().unwrap_err();
        assert!(err.issues.iter().any(|i| i.kind == VerifyIssueKind::BadTask), "{err}");
    }
}

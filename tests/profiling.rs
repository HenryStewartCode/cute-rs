//! Profiling scopes, the kernel symbol registry and IKET event tracing (roadmap part
//! m11-profiling-iket) on the simulator, and on the MLIR the scopes lower to.

mod iket_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;
use iket_kernels::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn iket_on() -> Option<IketOptions> {
    Some(IketOptions { capacity: 256, clock: IketClock::Clock64 })
}

fn run_trace(kernel: &Kernel) -> IketTrace {
    let mut module = kernel.sim_module();
    let mut args = handoff_args();
    module.simulate(handoff_cfg(), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<u32>(), handoff_sums());
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    kernel.decode_iket(&words).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn kernel_symbols_json() {
    let kernel = traced_handoff(None);
    let s = kernel.symbols();
    assert_eq!(
        s.warps.iter().map(|w| (w.name.as_str(), w.warp_start, w.warp_end)).collect::<Vec<_>>(),
        [("producer", 0, 0), ("consumer", 1, 1)]
    );
    // PipelineAsync registers its barrier under the pipeline's name.
    assert_eq!(s.barriers.len(), 1);
    assert_eq!((s.barriers[0].name.as_str(), s.barriers[0].num_stages), ("handoff", 2));
    assert_eq!(s.bar_alloc, "handoff:2");
    let json = kernel.kernel_symbols_json();
    assert!(json.contains("\"producer\": {\"warp_start\": 0, \"warp_end\": 0}"), "{json}");
    assert!(json.contains("\"handoff\": {\"num_stages\": 2, \"pipeline_type\": \"PipelineAsync\"}"), "{json}");
    assert!(json.contains("\"bar_alloc\": \"handoff:2\""), "{json}");

    let dir = std::env::temp_dir().join(format!("cute_symbols_{}", std::process::id()));
    let path = kernel
        .dump_profiling_metadata(&dir, &[("kernel", "traced_handoff".into()), ("version", 3.into())])
        .unwrap()
        .expect("written");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"kernel\": \"traced_handoff\"") && text.contains("\"version\": 3"), "{text}");
    // Neither barriers nor warps: nothing written.
    let empty = Kernel::build("nothing", || {});
    let quiet = dir.join("quiet");
    assert_eq!(empty.dump_profiling_metadata(&quiet, &[]).unwrap(), None);
    assert!(!quiet.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn registry_api_merges_and_clears() {
    let kernel = Kernel::build("registry", || {
        register_symbol("w", Some("warp"), &[("warp_start", 2.into())]);
        register_symbol("w", None, &[("warp_end", 5.into()), ("note", "x".into())]);
        let reg = get_symbol_registry();
        assert_eq!(reg.entries.len(), 1);
        assert_eq!(reg.entries[0].metadata.len(), 3);
        let dumped = dump_kernel_symbols();
        assert_eq!(dumped.warps.len(), 1);
        assert_eq!((dumped.warps[0].warp_start, dumped.warps[0].warp_end), (2, 5));
        assert!(get_symbol_registry().entries.is_empty(), "dump clears the registry");
        register_barrier("b", 3, "PipelineTmaAsync");
        register_barrier("", 3, "ignored");
        reset_symbol_registry();
        assert!(get_symbol_registry().entries.is_empty());
        register_barrier("kept", 4, "PipelineAsync");
    });
    assert_eq!(kernel.symbols().bar_alloc, "kept:4");
}

fn scoped_kernel() -> Kernel {
    Kernel::builder("scoped").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let i = thread_idx().x;
        let outer = Scope::path(&["mainloop", "stage"]);
        out.store(i, i + 1);
        let mut s = Scope::new("epilogue");
        out.store(i, i + 2);
        s.switch("store", &[("bytes", 128.into())]);
        out.store(i, i + 3);
        s.close();
        outer.close();
        out.store(i, i + 4);
    })
}

#[test]
fn name_locs_nest_outermost_first() {
    let kernel = scoped_kernel();
    let mlir = kernel.to_mlir_with(&MlirOptions { locations: true, ..Default::default() });
    assert!(mlir.contains("loc(\"mainloop\"(\"stage\"(\""), "{mlir}");
    assert!(mlir.contains("loc(\"mainloop\"(\"stage\"(\"epilogue\"(\""), "{mlir}");
    assert!(mlir.contains("loc(\"mainloop\"(\"stage\"(\"store\"(\""), "{mlir}");
    // The scope closed: the last store is outside every scope.
    let last = mlir.lines().rfind(|l| l.contains("store")).unwrap();
    assert!(!last.contains("loc(\"mainloop\""), "{last}");
    assert!(kernel.dump().contains("// scope mainloop/stage"), "{}", kernel.dump());
    // switch registered metadata under the new name.
    assert!(get_meta_note(&kernel));
    // A stock mlir-opt accepts the NameLocs.
    if let Some(tc) = Toolchain::discover() {
        tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}\n{mlir}"));
    }
}

fn get_meta_note(kernel: &Kernel) -> bool {
    // Metadata without a kind is kept out of the grouped symbols.
    kernel.symbols().is_empty()
}

/// A race between two warps, reported with the scope the failing read was built in.
#[test]
fn scope_path_in_race_report() {
    let kernel = Kernel::builder("scoped_race").block_dim([64, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let tile = make_smem_tensor::<u32>(Layout::col_major(32));
        let lane = lane_idx();
        // Whichever warp the scheduler runs second reports the race, inside the scope.
        let _s = Scope::path(&["consumer", "read_tile"]);
        if_else(warp_idx().eq(0), || tile.store(lane, lane.to::<u32>()), || out.store(lane, tile.load(lane)));
    });
    let mut args = vec![KernelArg::buffer(&[0u32; 32])];
    let opts = SimOptions { detect_uninitialized: false, ..SimOptions::default() };
    let e = kernel.simulate_with(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args, &opts).unwrap_err();
    assert!(e.message.contains("data race"), "{e}");
    assert!(e.message.contains("in profiling scope `consumer/read_tile`"), "{e}");
}

#[test]
fn scopes_across_runtime_control_flow() {
    let kernel = Kernel::builder("scoped_flow").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let i = thread_idx().x;
        let _all = Scope::new("all");
        for_range(0, 4, 1, |k| {
            let _per = Scope::new("iteration");
            if_(i.eq(0), || {
                let _inner = Scope::new("lead");
                out.store(i, k);
            });
        });
    });
    let ir = kernel.ir();
    // The scope table dedups: all, all/iteration, all/iteration/lead.
    assert_eq!(ir.scopes.len(), 3);
    let mut args = vec![KernelArg::i32(vec![9; 32])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(args[0].to_vec::<i32>()[0], 3);
    assert!(kernel.dump().contains("// scope all/iteration/lead"));

    // A scope opened in a body must close in it.
    let msg = build_panic(|| {
        Kernel::build("leaky", || {
            let leaked = std::cell::RefCell::new(None);
            for_range(0, 2, 1, |_| {
                *leaked.borrow_mut() = Some(Scope::new("escapes"));
            });
        });
    });
    assert!(msg.contains("must be closed in the if_/for_range/while body that opened it"), "{msg}");
    // Scopes close in reverse order.
    let msg = build_panic(|| {
        Kernel::build("crossed", || {
            let a = Scope::new("a");
            let _b = Scope::new("b");
            a.close();
        });
    });
    assert!(msg.contains("reverse order"), "{msg}");
}

#[test]
fn iket_trace_orders_and_pairs_events_per_warp() {
    let kernel = traced_handoff(iket_on());
    let trace = run_trace(&kernel);
    assert!(!trace.overflowed && trace.dropped == 0);
    assert!(trace.unmatched.is_empty(), "{:?}", trace.unmatched);

    let mut producer = vec!["Start:producer".to_string()];
    for k in 0..ITEMS {
        producer.push("Mark:produced".into());
        producer.push("Start:fill".into());
        if k + 1 < ITEMS {
            producer.push("End:fill".into());
        }
    }
    // The sentinel's first end emitted nothing; the last end follows the loop.
    let mut want = vec!["Start:producer".to_string()];
    for k in 0..ITEMS {
        if k > 0 {
            want.push("End:fill".into());
        }
        want.push("Mark:produced".into());
        want.push("Start:fill".into());
    }
    want.extend(["End:fill".into(), "End:producer".into()]);
    assert_eq!(warp_events(&trace, 0), want);

    let mut want = vec!["Start:consumer".to_string()];
    for _ in 0..ITEMS {
        want.extend(["Push:wait", "Pop:pop", "Push:consume", "Pop:pop", "Mark:consumed"].map(String::from));
    }
    want.push("End:consumer".into());
    assert_eq!(warp_events(&trace, 1), want);

    // Pairing: fill spans carry the start payload; pushes and pops pair last-in first-out.
    let fills: Vec<_> = trace.spans.iter().filter(|s| s.name == "fill" && s.warp == 0).collect();
    assert_eq!(fills.len(), ITEMS);
    assert_eq!(fills[1].start_payload.unwrap().as_f64(), 1.0);
    assert_eq!(fills[0].start_payload.unwrap().ty, "i32");
    assert_eq!(trace.spans.iter().filter(|s| s.name == "wait").count(), ITEMS);
    assert_eq!(trace.spans.iter().filter(|s| s.name == "consume").count(), ITEMS);
    assert!(trace.spans.iter().all(|s| s.end >= s.start));
    // Deterministic: a second run gives the same trace.
    let again = run_trace(&kernel);
    assert_eq!(trace.records, again.records);
    // Timestamps never decrease within a warp.
    for warp in 0..2 {
        let ts: Vec<u64> = trace.warp_records(0, warp).iter().map(|r| r.ts).collect();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "{ts:?}");
    }

    // The Chrome trace is JSON-shaped, with the symbols as metadata.
    let json = trace.to_chrome_json();
    assert!(json.contains("\"ph\": \"X\"") && json.contains("\"name\": \"fill\""), "{json}");
    assert!(json.contains("\"symbols\": {\"warps\"") && json.contains("handoff:2"), "{json}");
}

/// With IKET off the event calls emit nothing: same MLIR (and PTX) as without them.
#[test]
fn iket_off_is_free() {
    let plain = |calls: bool| {
        Kernel::builder("free").block_dim([32, 1, 1]).build(move || {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let i = thread_idx().x;
            let scope = Scope::with(&["s"], true, &[]);
            if calls {
                mark("m", Some(i.into()));
                let t = range_start("r", None);
                range_push("p", None);
                range_pop();
                range_end(t, None);
                dag("d").edge("a", "b", Some("l"), None);
            }
            out.store(i, i * 2);
            scope.close();
        })
    };
    let (with, without) = (plain(true), plain(false));
    assert_eq!(with.to_mlir(), without.to_mlir());
    assert!(with.iket().is_none());
    assert!(with.decode_iket(&[0; 8]).is_err());
    if let Some(tc) = Toolchain::discover() {
        let a = with.compile_ptx(&tc, Arch::Sm80).unwrap();
        let b = without.compile_ptx(&tc, Arch::Sm80).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn iket_overflow_sets_a_flag_instead_of_writing_out_of_bounds() {
    let kernel = Kernel::builder("overflow")
        .block_dim([32, 1, 1])
        .iket(IketOptions { capacity: 4, ..Default::default() })
        .build(|| {
            for_range(0, 10, 1, |k| mark("tick", Some(k.into())));
        });
    let mut module = kernel.sim_module();
    module.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut []).unwrap();
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    assert_eq!(words.len(), 4 + 8 * 4);
    let trace = kernel.decode_iket(&words).unwrap();
    assert!(trace.overflowed);
    assert_eq!((trace.records.len(), trace.dropped), (4, 6));
    let ks: Vec<f64> = trace.records.iter().map(|r| r.payload.unwrap().as_f64()).collect();
    assert_eq!(ks, [0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn iket_misuse_is_diagnosed() {
    let on = || Some(IketOptions::default());
    // A token from another kernel.
    let mut foreign = None;
    let mut a = Kernel::builder("a").iket(on().unwrap()).block_dim([32, 1, 1]);
    a = a.max_threads([32, 1, 1]).block_dim([32, 1, 1]);
    a.build(|| foreign = Some(range_start("r", None)));
    let token = foreign.unwrap();
    let msg = build_panic(|| {
        Kernel::builder("b").iket(IketOptions::default()).build(|| range_end(token, None));
    });
    assert!(msg.contains("another kernel"), "{msg}");
    // Start and end payloads of different types.
    let msg = build_panic(|| {
        Kernel::builder("c").iket(IketOptions::default()).build(|| {
            let t = range_start("r", Some(Int32::constant(1).into()));
            range_end(t, Some(Value::<f32>::constant(1.0).into()));
        });
    });
    assert!(msg.contains("started with a i32 payload but ends with a f32 payload"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("c2").iket(IketOptions::default()).build(|| {
            let t = range_start("r", Some(Int32::constant(1).into()));
            range_end(t, None);
        });
    });
    assert!(msg.contains("i32 payload but ends with a none payload"), "{msg}");
    // Push/pop imbalance.
    let msg = build_panic(|| {
        Kernel::builder("d").iket(IketOptions::default()).build(|| {
            range_push("open", None);
            range_push("inner", None);
            range_pop();
        });
    });
    assert!(msg.contains("1 range_push call(s) not popped") && msg.contains("`open`"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("e").iket(IketOptions::default()).build(range_pop);
    });
    assert!(msg.contains("range_pop without a matching range_push"), "{msg}");
}

/// Unmatched events are reported by the decoder: a range never ended, a pop that has no push.
#[test]
fn trace_reports_unmatched_events() {
    let kernel = Kernel::builder("lopsided").block_dim([32, 1, 1]).iket(IketOptions::default()).build(|| {
        let _never_ended = range_start("open", None);
        let t = sentinel_token("late");
        // A sentinel ends nothing: no record.
        range_end(t, None);
        mark("done", None);
    });
    let mut module = kernel.sim_module();
    module.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut []).unwrap();
    let words = module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>();
    let trace = kernel.decode_iket(&words).unwrap();
    assert_eq!(trace.records.len(), 2);
    assert_eq!(trace.unmatched.len(), 1);
    assert!(trace.unmatched[0].contains("range start never ended") && trace.unmatched[0].contains("`open`"));
}

/// The instrumented `tensorop_gemm` on the simulator: every warp pushes and pops the prologue,
/// each K tile's wait and issue, and the epilogue.
#[test]
fn instrumented_gemm_records_each_phase_per_warp() {
    #[allow(dead_code)]
    #[path = "../examples/tensorop_gemm.rs"]
    mod tensorop_gemm;
    let (m, n, k) = (128, 128, 96);
    let cfg = tensorop_gemm::GemmConfig {
        iket: Some(IketOptions { capacity: 512, clock: IketClock::Clock64 }),
        ..tensorop_gemm::GemmConfig::new(m, n, k)
    };
    let kernel = tensorop_gemm::build(cfg);
    let (a, b) = (tensorop_gemm::small_integers(m * k, 1), tensorop_gemm::small_integers(n * k, 2));
    let mut args = vec![
        tensorop_gemm::input(cfg.ab, &a),
        tensorop_gemm::input(cfg.ab, &b),
        tensorop_gemm::input(cfg.c, &vec![0.0; m * n]),
    ];
    let mut module = kernel.sim_module();
    module.simulate(LaunchConfig::new([1, 1, 1], [128, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let want: f32 = (0..k).map(|l| a[l] * b[l]).sum();
    assert_eq!(args[2].to_f32()[0], want);
    let trace = kernel.decode_iket(&module.read_global(IKET_GLOBAL).unwrap().to_vec::<u32>()).unwrap();
    assert!(trace.unmatched.is_empty() && !trace.overflowed);
    let k_tiles = k / cfg.bk;
    for warp in 0..4 {
        let events = warp_events(&trace, warp);
        let mut want = vec!["Push:prologue".to_string(), "Pop:pop".to_string()];
        for _ in 0..k_tiles {
            want.extend(["Push:wait", "Pop:pop", "Push:issue", "Pop:pop"].map(String::from));
        }
        want.extend(["Push:epilogue".to_string(), "Pop:pop".to_string()]);
        assert_eq!(events, want, "warp {warp}");
    }
}

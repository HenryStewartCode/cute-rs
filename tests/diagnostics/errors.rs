//! Located errors, code frames, compiler diagnostics and device assertions.

use super::*;
// ---- Located errors, code frames, compiler diagnostics,
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
    assert!(line.contains("\"negative input\" loc(\"tests/diagnostics/errors.rs\":"), "{line}");
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

//! `printf` with `{}` placeholders (CuTe DSL's `cute.printf`): every kind of
//! operand on the simulator, the refusals, and the MLIR and PTX.
//!
//! `hello_world` ports the device printing of the notebook
//! `examples/python/CuTeDSL/notebooks/1_dsl_features/01_hello_world.ipynb`
//! of CUTLASS 4.8.1, Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
//! All rights reserved. BSD-3-Clause; see THIRD-PARTY-NOTICES.md.

mod support;

use cute::prelude::*;

pub use support::printf_cases::*;

fn sim_lines(case: &PrintCase) -> Vec<String> {
    let mut args = (case.args)();
    case.kernel.simulate(case.cfg, &mut args).unwrap_or_else(|e| panic!("{}: {e}", case.kernel.name())).printed
}

#[test]
fn every_operand_kind_prints() {
    let case = everything();
    assert_eq!(sim_lines(&case).concat(), case.expected);
}

#[test]
fn hello_world() {
    let case = hello();
    let lines = sim_lines(&case);
    assert_eq!(lines, vec!["hello from the GPU -- block 0, thread 0\n\n".to_string()]);
    let mut args = (case.args)();
    case.kernel.simulate(case.cfg, &mut args).unwrap();
    assert_eq!(args[2].to_f32(), (0..case.n).map(|i| 3.0 * i as f32).collect::<Vec<_>>());
}

/// The text a tensor operand prints is `print_tensor`'s, after the
/// format's own text.
#[test]
fn tensor_operand() {
    let k = Kernel::builder("print_tensor_operand").block_dim([1, 1, 1]).build(|| {
        let f = make_rmem_tensor::<f32>(Layout::col_major(2));
        f.store(0, 1.0);
        f.store(1, -2.5);
        printf("t: {} end\n", &[(&f).into()]);
    });
    let report = k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut []).unwrap();
    assert_eq!(
        report.printed.concat(),
        "t: tensor(raw_ptr<Float32, rmem, align(16)> o 2:1, data=\n       [ 1.0000, -2.5000]) end\n"
    );
}

/// More than 32 operands are split over several `vprintf` calls, between
/// operands.
#[test]
fn long_prints_split() {
    let k = Kernel::builder("print_many").block_dim([1, 1, 1]).build(|| {
        let args: Vec<PrintArg> = (0..40).map(PrintArg::from).collect();
        printf_values(&args);
    });
    let report = k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut []).unwrap();
    assert_eq!(report.printed.len(), 2);
    let want = (0..40).map(|i| i.to_string()).collect::<Vec<_>>().join(", ") + "\n";
    assert_eq!(report.printed.concat(), want);
}

/// A structure with more runtime leaves than one call takes is cut inside
/// its text, between conversions, and still prints whole and in order.
#[test]
fn oversized_structures_split() {
    let k = Kernel::builder("print_wide_tuple").block_dim([1, 1, 1]).build(|| {
        let m = param_scalar::<i32>("m");
        let wide: Tree<IndexExpr> = Tree::Branch((0..40).map(|i| Tree::Leaf(IndexExpr::from(m + i))).collect());
        printf("a %d wide {} 100%% %d\n", &[7.into(), wide.into(), 8.into()]);
    });
    let mut args = vec![KernelArg::scalar(1i32)];
    let report = k.simulate(LaunchConfig::new([1, 1, 1], [1, 1, 1]), &mut args).unwrap();
    assert!(report.printed.len() >= 2, "{:?}", report.printed);
    let tuple = (1..=40).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    assert_eq!(report.printed.concat(), format!("a 7 wide ({tuple}) 100% 8\n"));
}

fn build_panic(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("the build should fail");
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

#[test]
fn refusals_name_the_line() {
    type Case = (Box<dyn Fn()>, &'static str);
    let cases: Vec<Case> = vec![
        (Box::new(|| printf("{} {}\n", &[1.into()])), "has 2 placeholders and conversions, but 1 operand is given"),
        (Box::new(|| printf("%d\n", &[1.into(), 2.into()])), "has 1 placeholder and conversion, but 2 operands"),
        (Box::new(|| printf("%d\n", &[1.5f32.into()])), "the conversion `%d` reads a 32-bit integer"),
        (Box::new(|| printf("%f\n", &[1.into()])), "the conversion `%f` reads a float, but its operand is Int32"),
        (Box::new(|| printf("%lld\n", &[1.into()])), "`%lld` reads a signed 64-bit integer (Int64)"),
        (Box::new(|| printf("%d\n", &[7i64.into()])), "(use %lld or %llu)"),
        (Box::new(|| printf("%s\n", &[1.into()])), "`%s` is not a conversion printf takes"),
        (Box::new(|| printf("100%\n", &[])), "`%\n` is not a conversion"),
        (Box::new(|| printf("%d\n", &[(&Layout::col_major(4)).into()])), "a layout prints through a {} placeholder"),
        (
            Box::new(|| {
                let p = alloc_smem::<f32>(512, 1024);
                let t = make_tensor(&p, make_composed_layout(make_swizzle(3, 2, 3), 0, Layout::row_major((8, 64))));
                printf("{}\n", &[(&t).into()]);
            }),
            "a tensor whose layout is swizzled (a swizzled composed layout, PISL) cannot be printed",
        ),
    ];
    for (i, (body, want)) in cases.into_iter().enumerate() {
        let msg = build_panic(|| {
            Kernel::build(format!("refused_{i}"), body);
        });
        assert!(msg.contains(want), "case {i}: {msg}");
        assert!(msg.contains("tests/printf.rs"), "case {i} names no line of the test: {msg}");
    }
    // A swizzled pointer's tensor prints.
    Kernel::build("pdsl", || {
        let p = alloc_smem::<f32>(512, 1024);
        let t = make_tensor(&recast_ptr::<f32, _>(&p, Some(make_swizzle(3, 4, 3))), Layout::row_major((2, 2)));
        printf("{}\n", &[(&t).into()]);
    });
}

/// IR built by hand is checked too: the conversions against the operands.
#[test]
fn the_validator_checks_printf_ops() {
    let k = Kernel::build("hand_built", || printf("%d %d\n", &[1.into(), 2.into()]));
    let mut ir = k.ir().clone();
    for inst in &mut ir.body.insts {
        if let cute::ir::Op::Printf { format, .. } = &mut inst.op {
            *format = "%d\n".into();
        }
    }
    let e = Kernel::from_ir(ir.clone()).unwrap_err();
    assert!(e.message.contains("has 1 conversion(s) but 2 operand(s)"), "{e}");
    for inst in &mut ir.body.insts {
        if let cute::ir::Op::Printf { format, .. } = &mut inst.op {
            *format = "%f %d\n".into();
        }
    }
    let e = Kernel::from_ir(ir).unwrap_err();
    assert!(e.message.contains("`%f` reads a float"), "{e}");
}

/// Narrow operands are widened in the IR (C's variadic promotion); the
/// MLIR verifies, and the PTX calls `vprintf`.
#[test]
fn mlir_and_ptx() {
    let case = everything();
    let mlir = case.kernel.to_mlir();
    let printfs: Vec<&str> = mlir.lines().filter(|l| l.contains("gpu.printf")).collect();
    assert!(!printfs.is_empty(), "{mlir}");
    for line in printfs {
        let types = line.rsplit_once(" : ").map_or("", |(_, t)| t);
        let narrow = types
            .split(", ")
            .any(|t| matches!(t.split_whitespace().next(), Some("i1" | "i8" | "i16" | "f16" | "bf16")));
        assert!(!narrow, "an operand is not widened: {line}");
    }
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    tc.verify(&mlir).unwrap_or_else(|e| panic!("{e}"));
    let ptx = case.kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains("call.uni") && ptx.contains("vprintf"), "{ptx}");
}

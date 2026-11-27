//! Vars, region structure, loop hints and fragments written in loops, and compile-option misuse.

use super::*;
/// Setting a Var to a value of other types names both the line that created
/// it and the line that set it.
#[test]
fn var_type_change_names_both_lines() {
    let (mut created, mut set) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("var_type_change", || {
            let t = thread_idx().x;
            let lanes = Var::new(vec![t, t + 1]);
            created = line!() - 1;
            for_range(0, t, 1, |i| {
                set = line!() + 1;
                lanes.set(vec![i, i, i]);
            });
        });
    });
    assert!(msg.contains("holds (i32, i32), but is set to (i32, i32, i32)"), "{msg}");
    assert!(msg.contains("a Var's type cannot change"), "{msg}");
    for line in [created, set] {
        assert!(msg.contains(&format!("tests/diagnostics/vars_and_loops.rs:{line}:")), "line {line}: {msg}");
    }
    // A register tensor of another size is another type too.
    let msg = build_panic(|| {
        Kernel::build("var_tensor_change", || {
            let frag = make_fragment::<f32>(4);
            frag.fill(0.0);
            let acc = Var::new(frag);
            let bigger = make_fragment::<f32>(8);
            bigger.fill(1.0);
            acc.set(bigger);
        });
    });
    assert!(
        msg.contains("holds (f32, f32, f32, f32), but is set to (f32, f32, f32, f32, f32, f32, f32, f32)"),
        "{msg}"
    );
}

/// What a loop reads from a Var belongs to the loop body, like any value
/// computed there: using it after the loop is the escape error. The Var
/// itself holds the loop's result.
#[test]
fn value_read_from_a_var_in_a_loop_cannot_escape() {
    let msg = build_panic(|| {
        Kernel::builder("var_escape").block_dim([32, 1, 1]).build(|| {
            let out = param_tensor::<i32>("out", Layout::col_major(32));
            let t = thread_idx().x;
            let sum = Var::new(Int32::from(0));
            let mut seen = None;
            for_range(0, t, 1, |i| {
                seen = Some(sum.get());
                sum.set(sum.get() + i);
            });
            out.store(t, seen.unwrap() + sum.get());
        });
    });
    assert!(
        msg.contains("outside the region that defines it") && msg.contains("including values read from a Var"),
        "{msg}"
    );
    assert!(msg.contains("tests/diagnostics/vars_and_loops.rs"), "the error should point at the kernel source: {msg}");
}

/// A Var is local to the region that created it.
#[test]
fn var_used_outside_its_region() {
    let msg = build_panic(|| {
        Kernel::build("var_after_branch", || {
            let t = thread_idx().x;
            let mut inner = None;
            if_(t.lt(4), || inner = Some(Var::new(t)));
            let _ = inner.unwrap().get();
        });
    });
    assert!(msg.contains("outside the region it was created in"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("var_in_other_branch", || {
            let t = thread_idx().x;
            let inner = std::cell::RefCell::new(None);
            if_else(t.lt(4), || *inner.borrow_mut() = Some(Var::new(t)), || inner.borrow().as_ref().unwrap().set(t));
        });
    });
    assert!(msg.contains("outside the region it was created in"), "{msg}");
    let mut stash = None;
    Kernel::build("first", || stash = Some(Var::new(thread_idx().x)));
    let msg = build_panic(|| {
        Kernel::build("second", || {
            let _ = stash.as_ref().unwrap().get();
        });
    });
    assert!(msg.contains("a Var from another kernel"), "{msg}");
}

/// A while condition reads Vars; it cannot set the ones created outside it.
#[test]
fn var_set_in_a_while_condition() {
    let msg = build_panic(|| {
        Kernel::build("var_in_condition", || {
            let n = Var::new(thread_idx().x);
            while_carry(
                (),
                |()| {
                    n.set(n.get() - 1);
                    n.get().gt(0)
                },
                |()| (),
            );
        });
    });
    assert!(msg.contains("inside a while_carry condition") && msg.contains("set it in the loop body"), "{msg}");
}

/// A loop marked at_least_once that runs zero times breaks the promise the
/// compiler may rely on: the simulator names the thread, the validator
/// catches constant bounds.
#[test]
fn at_least_once_loop_that_runs_zero_times() {
    let once = LoopOptions { at_least_once: true, ..LoopOptions::default() };
    let kernel = Kernel::builder("at_least_once").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(32));
        let t = thread_idx().x;
        let acc = Var::new(Int32::from(0));
        for_range_with(0, t % 8, 1, once, |i| acc.set(acc.get() + i));
        out.store(t, acc.get());
    });
    let e = simulate_err(&kernel, 32, &mut [KernelArg::i32(vec![0; 32])]);
    assert!(e.message.contains("marked at_least_once, but runs zero times") && e.message.contains("(0..0)"), "{e}");
    assert_eq!(e.thread, Some([0, 0, 0]));
    assert!(e.to_string().contains("tests/diagnostics/vars_and_loops.rs"), "the error should point at the loop: {e}");
    let msg = build_panic(|| {
        Kernel::build("never", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(4, 4, 1, once, |i| x.store(i, i));
        });
    });
    assert!(msg.contains("marked at_least_once, but 4..4 runs zero times"), "{msg}");
}

/// Loop hints that contradict each other or do not fit are rejected.
#[test]
fn loop_hints_are_validated() {
    let both = LoopOptions { unroll: Some(2), unroll_full: true, ..LoopOptions::default() };
    let msg = build_panic(|| {
        Kernel::build("both_unrolls", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(0, 8, 1, both, |i| x.store(i, i));
        });
    });
    assert!(msg.contains("unroll (2) and unroll_full contradict each other"), "{msg}");
    let msg = build_panic(|| {
        Kernel::build("huge_unroll", || {
            let x = param_tensor::<i32>("x", Layout::col_major(8));
            for_range_with(0, 8, 1, LoopOptions::unroll(u32::MAX), |i| x.store(i, i));
        });
    });
    assert!(msg.contains("an unroll count of 4294967295 does not fit in i32"), "{msg}");
}

/// The vectorize hint needs sm_100 (CuTe DSL raises UNSUP_ARCH below it).
#[test]
fn vectorize_needs_sm100() {
    use cute::validate::check_target;
    let kernel = Kernel::build("vectorized", || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let opts = LoopOptions { vectorize: true, ..LoopOptions::default() };
        for_range_with(0, 64, 1, opts, |i| x.store(i, x.load(i) * 2.0));
    });
    for arch in [Arch::Sm80, Arch::Sm89, Arch::Sm90a] {
        let e = check_target(kernel.ir(), arch).unwrap_err();
        let want = format!("needs sm_100 or newer; the kernel is being compiled for {}", arch.chip());
        assert!(e.message.contains("a vectorized loop") && e.message.contains(&want), "{e}");
        assert!(e.to_string().contains("tests/diagnostics/vars_and_loops.rs"), "{e}");
    }
    assert_eq!(check_target(kernel.ir(), Arch::Sm100a).unwrap(), 86);
    assert_eq!(check_target(kernel.ir(), Arch::Sm120).unwrap(), 87);
}

/// A register fragment with `values` in its registers, laid out by `layout`.
fn filled_fragment(layout: Layout, values: impl Fn(usize) -> i32) -> Tensor<i32> {
    let frag = make_fragment_with_layout::<i32>(layout);
    for k in 0..frag.size() {
        frag.store(k, values(k));
    }
    frag
}

/// A register tensor's layout is part of a Var's type: setting the Var to a
/// tensor with another layout names both lines, at the top level and in a
/// loop (where `get` would otherwise read the registers in the old order).
#[test]
fn var_structure_change_names_both_lines() {
    let (mut created, mut set) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("var_layout_change", || {
            let v = Var::new(filled_fragment(Layout::col_major((2, 2)), |k| k as i32));
            created = line!() - 1;
            set = line!() + 1;
            v.set(filled_fragment(Layout::row_major((2, 2)), |k| 10 + k as i32));
        });
    });
    assert!(
        msg.contains("is set to a value of another structure")
            && msg.contains("a register tensor with layout (2,2):(1,2) became one with layout (2,2):(2,1)")
            && msg.contains("a Var's type cannot change"),
        "{msg}"
    );
    for line in [created, set] {
        assert!(msg.contains(&format!("tests/diagnostics/vars_and_loops.rs:{line}:")), "line {line}: {msg}");
    }
    // Inside a loop, and nested in a struct field and a tuple element.
    let msg = build_panic(|| {
        Kernel::build("var_field_layout_change", || {
            let v = Var::new((Int32::from(0), vec![filled_fragment(Layout::col_major(4), |k| k as i32)]));
            for_range(0, thread_idx().x, 1, |i| {
                let nested = make_fragment_with_layout::<i32>(Layout::col_major((2, 2)));
                nested.fill(i);
                v.set((i, vec![nested]));
            });
        });
    });
    assert!(
        msg.contains("element 1: element 0: a register tensor with layout 4:1 became one with layout (2,2):(1,2)"),
        "{msg}"
    );
    // Vectors of vectors with the same flattened types but another nesting.
    let msg = build_panic(|| {
        Kernel::build("var_nesting_change", || {
            let t = thread_idx().x;
            let v = Var::new(vec![vec![t, t], vec![t]]);
            v.set(vec![vec![t], vec![t, t]]);
        });
    });
    assert!(msg.contains("element 0: a vector of 2 elements became one of 1"), "{msg}");
}

/// What a loop body or a branch produces must keep the structure of what
/// came in, as CuTe DSL requires of the values its regions carry.
#[test]
fn region_structure_change_is_reported() {
    let msg = build_panic(|| {
        Kernel::build("loop_layout_change", || {
            let init = filled_fragment(Layout::col_major((2, 2)), |k| k as i32);
            for_range_carry(0, thread_idx().x, 1, init, |_, _| {
                filled_fragment(Layout::row_major((2, 2)), |k| k as i32)
            });
        });
    });
    assert!(
        msg.contains("for_range_carry at tests/diagnostics/vars_and_loops.rs")
            && msg.contains("the body returns state of another structure than `init`")
            && msg.contains("layout (2,2):(1,2) became one with layout (2,2):(2,1)"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("branch_layout_change", || {
            let t = thread_idx().x;
            if_else_yield(
                t.lt(4),
                || filled_fragment(Layout::col_major((2, 2)), |k| k as i32),
                || filled_fragment(Layout::row_major((2, 2)), |k| k as i32),
            );
        });
    });
    assert!(
        msg.contains("if_else_yield at tests/diagnostics/vars_and_loops.rs") && msg.contains("different structures"),
        "{msg}"
    );
    let msg = build_panic(|| {
        Kernel::build("while_layout_change", || {
            let init = filled_fragment(Layout::col_major((2, 2)), |k| k as i32);
            while_carry(init, |f| f.load(0usize).lt(10), |_| filled_fragment(Layout::row_major((2, 2)), |_| 0));
        });
    });
    assert!(msg.contains("while_carry at tests/diagnostics/vars_and_loops.rs"), "{msg}");
}

/// Writing a register of a fragment created outside a runtime loop after
/// the loop read it would not reach the next iteration (register fragments
/// are build-time arrays of SSA values, not memory as in CuTe DSL), so the
/// build fails, naming both lines.
#[test]
fn fragment_written_after_read_in_a_loop() {
    let (mut read, mut written) = (0, 0);
    let msg = build_panic(|| {
        Kernel::build("running_sum_in_registers", || {
            let out = param_tensor::<i32>("out", Layout::col_major(8));
            let f = make_fragment::<i32>(1);
            f.store(0usize, 0);
            for_range(0, 8, 1, |i| {
                read = line!() + 1;
                let sum = f.load(0usize) + i;
                written = line!() + 1;
                f.store(0usize, sum);
                out.store(i, f.load(0usize));
            });
        });
    });
    assert!(
        msg.contains("register 0 of a fragment created outside a runtime loop")
            && msg.contains("the loop's next iteration would not see the write")
            && msg.contains("Var::update"),
        "{msg}"
    );
    assert!(msg.contains(&format!("read at tests/diagnostics/vars_and_loops.rs:{read}:")), "{msg}");
    assert!(msg.contains(&format!("written at tests/diagnostics/vars_and_loops.rs:{written}:")), "{msg}");

    // Read in an outer loop and written in an inner one.
    let msg = build_panic(|| {
        Kernel::build("outer_read_inner_write", || {
            let f = make_fragment::<i32>(2);
            f.fill(1);
            for_range(0, 4, 1, |i| {
                let x = f.load(1usize);
                for_range(0, i, 1, |j| f.store(1usize, x + j));
            });
        });
    });
    assert!(msg.contains("register 1 of a fragment"), "{msg}");
    // Read by a while condition and written by the body.
    let msg = build_panic(|| {
        Kernel::build("condition_read_body_write", || {
            let f = make_fragment::<i32>(1);
            f.store(0usize, thread_idx().x);
            while_carry((), |()| f.load(0usize).gt(0), |()| f.store(0usize, thread_idx().x - 1));
        });
    });
    assert!(msg.contains("register 0 of a fragment"), "{msg}");
    // Read as an inner loop's initial state, then written: the next outer
    // iteration would start the inner loop from the old registers.
    let msg = build_panic(|| {
        Kernel::build("state_read_then_written", || {
            let f = make_fragment::<i32>(2);
            f.fill(0);
            for_range(0, 4, 1, |i| {
                let g = for_range_carry(0, i, 1, f.clone(), |j, g| {
                    g.store(0usize, g.load(0usize) + j);
                    g
                });
                f.store(1usize, g.load(0usize));
            });
        });
    });
    assert!(msg.contains("register 1 of a fragment"), "{msg}");
}

/// What a branch or loop result is known to be divisible by holds for every
/// side: with one side odd, a 128-bit copy at the result is refused.
#[test]
fn merged_facts_are_not_overstated() {
    let copy_at = |x: &Tensor<f32>, offset: Int32| {
        let frag = make_fragment::<f32>(4);
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy(&atom, &x.compose(Layout::new((4, 250), (1, 1))).slice((.., offset)), &frag);
    };
    let index = |t: Int32, k: usize, c: usize| (IndexExpr::from(t) * k + c).into_value();
    let msg = build_panic(|| {
        Kernel::builder("odd_branch").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let t = thread_idx().x;
            copy_at(&x, if_else_yield(t.lt(16), || index(t, 8, 0), || index(t, 4, 1)));
        });
    });
    assert!(msg.contains("cannot prove that the source offset"), "{msg}");
    let msg = build_panic(|| {
        Kernel::builder("odd_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let t = thread_idx().x;
            copy_at(&x, for_range_carry(0, 2, 1, index(t, 8, 0), |_, _| index(t, 2, 0)));
        });
    });
    assert!(msg.contains("cannot prove that the source offset"), "{msg}");
}

/// Inside a runtime loop a Var is a loop-carried value, about which nothing
/// is known even when the loop only reads it: the errors this causes name
/// the Var and say to read it before the loop, which works.
#[test]
fn var_read_in_a_loop_loses_what_is_known() {
    let aligned_copy = |x: &Tensor<f32>, offset: Int32| {
        let frag = make_fragment::<f32>(4);
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        copy(&atom, &x.compose(Layout::new((4, 253), (1, 1))).slice((.., offset)), &frag);
    };
    let mut created = 0;
    let msg = build_panic(|| {
        Kernel::builder("var_offset_in_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            let offset = Var::new((IndexExpr::from(thread_idx().x) * 4usize).into_value());
            created = line!() - 1;
            aligned_copy(&x, offset.get());
            for_range(0, 2, 1, |_| aligned_copy(&x, offset.get()));
        });
    });
    assert!(
        msg.contains("cannot prove that the source offset")
            && msg.contains(&format!(
                "is the value of the Var created at tests/diagnostics/vars_and_loops.rs:{created}:"
            ))
            && msg.contains("read the Var before the loop"),
        "{msg}"
    );
    Kernel::builder("var_offset_read_before_loop").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(256));
        let offset = Var::new((IndexExpr::from(thread_idx().x) * 4usize).into_value());
        let before = offset.get();
        for_range(0, 2, 1, |_| aligned_copy(&x, before));
    });
    // A Var holding a constant indexes registers, but not inside a loop.
    let msg = build_panic(|| {
        Kernel::build("var_register_index_in_loop", || {
            let out = param_tensor::<i32>("out", Layout::col_major(4));
            let frag = make_fragment::<i32>(4);
            frag.fill(7);
            let slot = Var::new(Int32::from(2));
            out.store(0usize, frag.load(slot.get()));
            for_range(0, 2, 1, |i| out.store(i, frag.load(slot.get())));
        });
    });
    assert!(
        msg.contains("register tensors must be indexed with static coordinates")
            && msg.contains("is the value of the Var created at tests/diagnostics/vars_and_loops.rs:"),
        "{msg}"
    );
    // The same error about a value that is not a Var's has no note.
    let msg = build_panic(|| {
        Kernel::builder("plain_offset_in_loop").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor::<f32>("x", Layout::col_major(256));
            for_range(0, 2, 1, |i| aligned_copy(&x, i + 1));
        });
    });
    assert!(msg.contains("cannot prove that the source offset") && !msg.contains("Var"), "{msg}");
}

/// Misused compile options are reported, naming the option, before anything is compiled:
/// a typed level out of range panics at the caller (as CuTe DSL raises for `OptLevel(4)`);
/// strings and the environment return errors; options whose feature has not landed name the
/// part that adds them; and compiling needs a target.
#[test]
fn compile_option_misuse() {
    let message = build_panic(|| {
        let _ = CompileOptions::new().opt_level(4);
    });
    assert!(message.contains("optimization level must be 0 to 3") && message.contains("not 4"), "{message}");

    for (text, want) in [
        ("--opt-level 5", "`--opt-level 5`: the optimization level must be 0 to 3"),
        ("--gpu-arch sm_70", "unknown GPU architecture `sm_70`"),
        ("--gpu-arch sm_88", "sm_88 is in the PTX ISA but not in CuTe DSL 4.8.1's Arch"),
        ("dump-ptx-path=maybe", "takes 1, true, yes or on"),
        ("remarks{ptxas}", "`ptxas` is not a diagnostic category"),
        ("--link-libraries libdevice.bc", "linking device libraries is not supported as a compile option string"),
        ("iket", "IKET in-kernel event tracing is not supported as a compile option string"),
        ("--frobnicate", "unknown option `--frobnicate`"),
        ("--keep", "ambiguous option `--keep`"),
        ("--h x", "ambiguous option `--h`: it could be --help, --host-target"),
        ("--opt-level 1 -", "unexpected `-`"),
        ("--opt-level 1 ''", "unexpected empty word, which holds no option"),
        ("--ptxas -v", "--ptxas-options needs a value"),
        ("--opt-level '1", "unterminated quote"),
    ] {
        let e = CompileOptions::parse(text).unwrap_err().to_string();
        assert!(e.contains(want), "{text}: {e}");
    }
    for (name, value, want) in [
        ("CUTE_RS_LINEINFO", "maybe", "CUTE_RS_LINEINFO=\"maybe\" is not a boolean"),
        ("CUTE_RS_KEEP", "ptx,asm", "CUTE_RS_KEEP: `asm` is not something to keep"),
    ] {
        let env = |n: &str| (n == name).then(|| value.to_string());
        let e = CompileOptions::new().with_env_from(env).unwrap_err().to_string();
        assert!(e.contains(want), "{name}={value}: {e}");
    }

    // A kernel records where it was built, for kernel-level diagnostics.
    let kernel = Kernel::build("no_target", || {
        let x = param_tensor::<f32>("x", Layout::col_major(32));
        x.store(thread_idx().x, Float32::from(1.0));
    });
    let built_at = line!() - 4;
    assert_eq!((kernel.ir().loc.file(), kernel.ir().loc.line()), (file!(), built_at));
    if let Some(tc) = Toolchain::discover() {
        let e = kernel.compile_with(&tc, &CompileOptions::new()).unwrap_err().to_string();
        assert!(e.contains("no target architecture to compile kernel `no_target` for"), "{e}");
        let hook =
            CompileOptions::new().gpu_arch(Arch::Sm80).stage_hook(|m| m.set_argument_attribute("x", 0, "a", "unit"));
        let e = kernel.compile_with(&tc, &hook).unwrap_err().to_string();
        assert!(e.contains("stage hook failed") && e.contains("no llvm.func @x"), "{e}");
    }
}

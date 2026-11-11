//! Mutable variables (`Var`): the IR they turn into (loop-carried values and
//! `if` results), and patterns beyond `feature_kernels::vars`, simulated and
//! checked against the same computation in Rust. Misuse is covered in
//! `tests/diagnostics.rs`.

use cute::ir::{KernelIr, Op, Region};
use cute::prelude::*;

const T: usize = 32;

fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [T as u32, 1, 1])
}

/// Run a kernel taking `xs` and writing one i32 per thread.
fn run(kernel: &Kernel, xs: &[i32]) -> Vec<i32> {
    let mut args = vec![KernelArg::i32(xs.to_vec()), KernelArg::i32(vec![0; T])];
    kernel.simulate(launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    args[1].as_i32().to_vec()
}

fn inputs() -> Vec<i32> {
    (0..T as i32).map(|i| (i * 37) % 23 - 11).collect()
}

/// Every operation in the kernel, outermost first.
fn ops(ir: &KernelIr) -> Vec<&Op> {
    fn walk<'a>(r: &'a Region, out: &mut Vec<&'a Op>) {
        for inst in &r.insts {
            out.push(&inst.op);
            inst.op.regions().into_iter().for_each(|r| walk(r, out));
        }
    }
    let mut out = Vec::new();
    walk(&ir.body, &mut out);
    out
}

fn mlir_compiles(kernel: &Kernel) {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    tc.verify(&kernel.to_mlir()).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
}

/// A Var set in a loop becomes an `scf.for` iter_arg, and one set in a
/// branch becomes an `scf.if` result.
#[test]
fn vars_become_iter_args_and_if_results() {
    let kernel = Kernel::builder("carried").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let sum = Var::new(xs.load(t));
        let flag = Var::new(Int32::from(0));
        for_range(0, t % 5, 1, |i| sum.set(sum.get() + i));
        if_(t.gt(3), || flag.set(t * 2));
        out.store(t, sum.get() * 100 + flag.get());
    });
    let ir = kernel.ir();
    let fors: Vec<_> = ops(ir).into_iter().filter(|op| matches!(op, Op::For { .. })).collect();
    let ifs: Vec<_> = ops(ir).into_iter().filter(|op| matches!(op, Op::If { .. })).collect();
    let [Op::For { init, args, results, body, .. }] = fors[..] else { panic!("one loop expected:\n{}", kernel.dump()) };
    assert_eq!((init.len(), args.len(), results.len(), body.yields.len()), (1, 1, 1, 1), "{}", kernel.dump());
    let [Op::If { results, then_region, else_region, .. }] = ifs[..] else { panic!("one if expected") };
    assert_eq!((results.len(), then_region.yields.len(), else_region.yields.len()), (1, 1, 1), "{}", kernel.dump());

    let mlir = kernel.to_mlir();
    assert!(mlir.contains("iter_args(") && mlir.contains("-> (i32) : i32 {"), "{mlir}");
    assert!(mlir.lines().any(|l| l.contains("= scf.if") && l.ends_with("-> (i32) {")), "{mlir}");
    mlir_compiles(&kernel);

    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T as i32 {
        let sum = xs[t as usize] + (0..t % 5).sum::<i32>();
        let flag = if t > 3 { t * 2 } else { 0 };
        assert_eq!(got[t as usize], sum * 100 + flag, "thread {t}");
    }
}

/// A loop that only reads a Var still carries it (it cannot know, when the
/// Var is first read, that it will not be set later), but the Var leaves the
/// loop as it came in: the `if` around the loop gets no result for it.
#[test]
fn vars_only_read_in_a_loop_are_not_results() {
    let kernel = Kernel::builder("read_only").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let scale = Var::new(xs.load(t));
        let unchanged = Var::new(Int32::from(5));
        let total = make_rmem_tensor::<i32>(Layout::col_major(1));
        total.store(0usize, 0);
        if_(t.gt(1), || {
            for_range(0, t, 1, |i| total.store(0usize, total.load(0usize) + i * scale.get()));
            // Setting a Var to the value it has is no change either.
            unchanged.set(unchanged.get());
        });
        out.store(t, total.load(0usize) + scale.get() + unchanged.get());
    });
    let ifs: Vec<_> = ops(kernel.ir()).into_iter().filter(|op| matches!(op, Op::If { .. })).collect();
    let [Op::If { results, .. }] = ifs[..] else { panic!("one if expected") };
    assert!(results.is_empty(), "{}", kernel.dump());
    mlir_compiles(&kernel);
    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T as i32 {
        let x = xs[t as usize];
        let total: i32 = if t > 1 { (0..t).map(|i| i * x).sum() } else { 0 };
        assert_eq!(got[t as usize], total + x + 5, "thread {t}");
    }
}

/// `if_else_yield` returns its explicit results; the Vars come after them.
#[test]
fn vars_and_explicit_results_together() {
    let kernel = Kernel::builder("mixed_if").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let x = xs.load(t);
        let side = Var::new(Int32::from(-1));
        let (p, q) = if_else_yield(
            x.lt(0),
            || {
                side.set(x * 2);
                (x + 1, Int32::from(10))
            },
            || (x - 1, Int32::from(20)),
        );
        // Explicit loop state and Vars in a while loop: halve until below 4,
        // counting the steps in a Var the condition never reads.
        let halvings = Var::new(Int32::from(0));
        let small = while_carry(
            x.abs() + 7,
            |v| v.ge(4),
            |v| {
                halvings.set(halvings.get() + 1);
                v / 2
            },
        );
        out.store(t, ((p * 31 + q) * 31 + side.get()) * 31 + small * 7 + halvings.get());
    });
    mlir_compiles(&kernel);
    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T {
        let x = xs[t];
        let (p, q, side) = if x < 0 { (x + 1, 10, x * 2) } else { (x - 1, 20, -1) };
        let (mut small, mut halvings) = (x.abs() + 7, 0);
        while small >= 4 {
            small /= 2;
            halvings += 1;
        }
        assert_eq!(got[t], ((p * 31 + q) * 31 + side) * 31 + small * 7 + halvings, "thread {t}");
    }
}

/// A Var of a vector, carried through a loop inside a while body, with a Var
/// created in a branch used only there.
#[test]
fn vars_of_vectors_in_nested_loops() {
    let kernel = Kernel::builder("vector_var").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let x = xs.load(t);
        let lanes = Var::new(vec![x, x + 1, x + 2]);
        let rounds = Var::new(Int32::from(0));
        while_carry(
            (),
            |()| rounds.get().lt(t % 4),
            |()| {
                for_range(0, 2, 1, |i| {
                    let v = lanes.get();
                    lanes.set(vec![v[1] + i, v[2], v[0] * 2]);
                });
                if_(rounds.get().eq(1), || {
                    let bonus = Var::new(Int32::from(100));
                    bonus.set(bonus.get() + x);
                    let v = lanes.get();
                    lanes.set(vec![v[0] + bonus.get(), v[1], v[2]]);
                });
                rounds.set(rounds.get() + 1);
            },
        );
        let v = lanes.get();
        out.store(t, (v[0] * 1000 + v[1]) * 1000 + v[2]);
    });
    mlir_compiles(&kernel);
    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T {
        let x = xs[t];
        let mut v = [x, x + 1, x + 2];
        for round in 0..(t as i32 % 4) {
            for i in 0..2 {
                v = [v[1] + i, v[2], v[0] * 2];
            }
            if round == 1 {
                v[0] += 100 + x;
            }
        }
        assert_eq!(got[t], (v[0] * 1000 + v[1]) * 1000 + v[2], "thread {t}");
    }
}

/// A Var only set in a loop, never read, still flows out of it.
#[test]
fn vars_only_set_in_a_loop() {
    let kernel = Kernel::builder("set_only").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let x = xs.load(t);
        let first_negative = Var::new(Int32::from(-1));
        for_range(0, t % 6, 1, |i| if_((x + i * 3).lt(0), || first_negative.set(i)));
        out.store(t, first_negative.get());
    });
    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T {
        let x = xs[t];
        let last = (0..t as i32 % 6).rfind(|i| x + i * 3 < 0).unwrap_or(-1);
        assert_eq!(got[t], last, "thread {t}");
    }
}

/// The hints show in the IR listing and become one LLVM loop annotation.
#[test]
fn loop_hints_in_the_listing_and_mlir() {
    let kernel = Kernel::builder("hints").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let acc = Var::new(t);
        for_range_with(0, t, 1, LoopOptions::unroll(4), |i| acc.set(acc.get() + i));
        for_range_with(0, t, 1, LoopOptions::unroll(1), |i| acc.set(acc.get() ^ i));
        let once = LoopOptions { at_least_once: true, ..LoopOptions::default() };
        for_range_with(0, t + 1, 1, once, |i| acc.set(acc.get() * 3 + i));
        for_range_with(0, 16, 1, LoopOptions::unroll_full(), |i| acc.set(acc.get() - i));
        out.store(t, acc.get());
    });
    let dump = kernel.dump();
    for hint in [") unroll(4) {", ") unroll(1) {", ") at_least_once {", ") unroll_full {"] {
        assert_eq!(dump.matches(hint).count(), 1, "{hint}\n{dump}");
    }
    let mlir = kernel.to_mlir();
    for annotation in [
        "} {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}",
        "} {loop_annotation = #llvm.loop_annotation<unroll = <disable = true, count = 1 : i32>>}",
        "} {loop_annotation = #llvm.loop_annotation<unroll = <full = true>>}",
    ] {
        assert_eq!(mlir.matches(annotation).count(), 1, "{annotation}\n{mlir}");
    }
    // at_least_once is not loop metadata but an assumption about the
    // bounds in front of its loop: four loops, three annotations.
    assert_eq!(mlir.matches("scf.for").count(), 4, "{mlir}");
    assert_eq!(mlir.matches("{loop_annotation = ").count(), 3, "{mlir}");
    let lines: Vec<&str> = mlir.lines().map(str::trim).collect();
    let assume = lines.iter().position(|l| l.starts_with("llvm.intr.assume")).expect("an assumption");
    assert_eq!(lines.iter().filter(|l| l.starts_with("llvm.intr.assume")).count(), 1, "{mlir}");
    let (runs, rest) = lines[assume - 1].split_once(" = arith.cmpi slt, ").expect("the comparison before it");
    assert_eq!(lines[assume], format!("llvm.intr.assume {runs} : i1"));
    let upper = rest.trim_end_matches(" : i32").split(", ").nth(1).unwrap();
    assert!(lines[assume + 1].contains(&format!(" to {upper} step ")), "the loop over those bounds follows:\n{mlir}");
    mlir_compiles(&kernel);
    let mut args = vec![KernelArg::i32(vec![0; T])];
    kernel.simulate(launch(), &mut args).unwrap();
    let got = args[0].as_i32();
    for t in 0..T as i32 {
        let mut acc = (0..t).fold(t, |a, i| a + i);
        acc = (0..t).fold(acc, |a, i| a ^ i);
        acc = (0..t + 1).fold(acc, |a, i| a.wrapping_mul(3).wrapping_add(i));
        acc = (0..16).fold(acc, |a, i| a.wrapping_sub(i));
        assert_eq!(got[t as usize], acc, "thread {t}");
    }
}

/// The vectorize hint is accepted from sm_100 on, sharing one annotation
/// with an unroll hint; the simulator ignores it.
#[test]
fn vectorize_hint() {
    let kernel = Kernel::builder("vectorized").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T * 8));
        let out = param_tensor::<f32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let opts = LoopOptions { vectorize: true, unroll: Some(2), ..LoopOptions::default() };
        let sum = for_range_carry_with(0, 8, 1, opts, Float32::from(0.0), |i, acc| acc + xs.load(t * 8 + i));
        out.store(t, sum);
    });
    let annotation =
        "{loop_annotation = #llvm.loop_annotation<vectorize = <disable = false>, unroll = <count = 2 : i32>>}";
    assert!(kernel.to_mlir().contains(annotation), "{}", kernel.to_mlir());
    let xs: Vec<f32> = (0..T * 8).map(|i| i as f32 * 0.25).collect();
    let mut args = vec![KernelArg::f32(xs.clone()), KernelArg::f32(vec![0.0; T])];
    kernel.simulate(launch(), &mut args).unwrap();
    for t in 0..T {
        let want = xs[t * 8..t * 8 + 8].iter().fold(0.0f32, |a, &x| a + x);
        assert_eq!(args[1].as_f32()[t], want, "thread {t}");
    }
}

/// A while condition can set a Var it created itself (only the Vars created
/// outside it must be set in the body).
#[test]
fn var_local_to_a_while_condition() {
    let kernel = Kernel::builder("condition_local").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let n = Var::new(xs.load(t).abs());
        let tries = while_carry(
            Int32::from(0),
            |_| {
                let doubled = Var::new(n.get());
                doubled.set(doubled.get() * 2);
                doubled.get().lt(100)
            },
            |k| {
                n.set(n.get() + 7);
                k + 1
            },
        );
        out.store(t, tries * 1000 + n.get());
    });
    let xs = inputs();
    let got = run(&kernel, &xs);
    for t in 0..T {
        let (mut n, mut tries) = (xs[t].abs(), 0);
        while n * 2 < 100 {
            n += 7;
            tries += 1;
        }
        assert_eq!(got[t], tries * 1000 + n, "thread {t}");
    }
}

/// The register-fragment patterns a runtime loop may use: registers written
/// before they are read in the body, a fragment created in the body, one
/// only read, one carried in a Var, and a predicated copy into a fragment
/// from outside (whose skipped registers keep the value from before).
#[test]
fn register_fragments_in_loops() {
    let kernel = Kernel::builder("fragments_in_loops").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let n = t % 6;

        let scratch = make_fragment::<i32>(2);
        let constant = make_fragment::<i32>(1);
        constant.store(0usize, x * 3);
        let sums = Var::new(Int32::from(0));
        let carried = make_fragment::<i32>(1);
        carried.store(0usize, x);
        let running = Var::new(carried);
        for_range(0, n, 1, |i| {
            scratch.store(0usize, x + i);
            scratch.store(1usize, scratch.load(0usize) * 2);
            let local = make_fragment::<i32>(1);
            local.store(0usize, i);
            local.store(0usize, local.load(0usize) + constant.load(0usize));
            sums.set(sums.get() + scratch.load(1usize) + local.load(0usize));
            running.update(|f| {
                f.store(0usize, f.load(0usize) + i);
                f
            });
        });
        out.store((t, 0), sums.get());
        out.store((t, 1), running.get().load(0usize));

        // Copy 4 values of a row into registers in each iteration, under a
        // predicate, and store them back under the same predicate.
        let row = param_tensor::<i32>("row", Layout::row_major((T, 4)));
        let dst = param_tensor::<i32>("dst", Layout::row_major((T, 4)));
        let regs = make_fragment::<i32>(4);
        regs.fill(-1);
        let atom = make_copy_atom::<i32>(CopyOp::Universal { bits: 32 });
        for_range(0, 2, 1, |i| {
            let pred = make_fragment::<bool>(4);
            for k in 0..4 {
                pred.store(k, (t + k + i).lt(T as i32 + 1));
            }
            copy_if(&atom, &pred, &row.slice((t, ..)), &regs);
            copy_if(&atom, &pred, &regs, &dst.slice((t, ..)));
        });
    });
    let xs = inputs();
    let row: Vec<i32> = (0..T as i32 * 4).map(|i| i * 7 - 3).collect();
    let mut args = vec![
        KernelArg::i32(xs.clone()),
        KernelArg::i32(vec![0; T * 2]),
        KernelArg::i32(row.clone()),
        KernelArg::i32(vec![9; T * 4]),
    ];
    kernel.simulate(launch(), &mut args).unwrap();
    let (out, dst) = (args[1].as_i32(), args[3].as_i32());
    for t in 0..T {
        let (x, n) = (xs[t], (t % 6) as i32);
        let sums: i32 = (0..n).map(|i| (x + i) * 2 + i + x * 3).sum();
        let running = x + (0..n).sum::<i32>();
        assert_eq!(out[t * 2..t * 2 + 2], [sums, running], "thread {t}");
        for k in 0..4 {
            let copied = (0..2).any(|i| t + k + i < T + 1);
            assert_eq!(dst[t * 4 + k], if copied { row[t * 4 + k] } else { 9 }, "thread {t}, element {k}");
        }
    }
}

/// An `if` result keeps what both branches' values have in common, a part
/// of a Var both branches set alike is no result at all, and a part a loop
/// leaves as it came in keeps its value after the loop.
#[test]
fn results_keep_what_is_known() {
    let kernel = Kernel::builder("known_results").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, 4)));
        let big = param_tensor::<i32>("big", Layout::col_major(512));
        let t = thread_idx().x;
        let x = xs.load(t);
        let index = |k: usize, c: usize| (IndexExpr::from(t) * k + c).into_value();

        let offset = Var::new(Int32::from(0));
        if_else(x.lt(0), || offset.set(index(8, 0)), || offset.set(index(4, 12)));
        assert!(IndexExpr::from(offset.get()).is_multiple_of(4));
        assert!(!IndexExpr::from(offset.get()).is_multiple_of(8));
        assert_eq!(IndexExpr::from(offset.get()).bound(), Some((T - 1) * 8 + 1));

        // Part 0 is set to 7 in both branches (two constants, one value).
        let pair = Var::new((Int32::from(0), x));
        if_else(x.lt(0), || pair.set((Int32::from(7), x * 2)), || pair.set((Int32::from(7), x - 1)));
        assert_eq!(pair.get().0.as_constant(), Some(7));

        // The loop counts in part 0 and sets part 1 back to what it was.
        let state = Var::new((Int32::from(0), Int32::from(5)));
        for_range(0, t % 4, 1, |i| state.set((state.get().0 + i, Int32::from(5))));
        assert_eq!(state.get().1.as_constant(), Some(5));

        out.store((t, 0), offset.get());
        out.store((t, 1), pair.get().0 * 100 + pair.get().1);
        out.store((t, 2), state.get().0 * 10 + state.get().1);
        // Read before a loop, an aligned offset stays aligned inside it.
        let aligned = offset.get();
        for_range(0, 1, 1, |_| {
            let frag = make_fragment::<i32>(4);
            let atom = make_copy_atom::<i32>(CopyOp::Universal { bits: 128 });
            copy(&atom, &big.compose(Layout::new((4, 509), (1, 1))).slice((.., aligned)), &frag);
            out.store((t, 3), frag.load(0usize) + frag.load(3usize));
        });
    });
    let ifs: Vec<_> = ops(kernel.ir()).into_iter().filter(|op| matches!(op, Op::If { .. })).collect();
    let [Op::If { results: offset_results, .. }, Op::If { results: pair_results, .. }] = ifs[..] else {
        panic!("two ifs expected:\n{}", kernel.dump())
    };
    assert_eq!((offset_results.len(), pair_results.len()), (1, 1), "{}", kernel.dump());
    mlir_compiles(&kernel);
    let xs = inputs();
    let big: Vec<i32> = (0..512).map(|i| i * 3 + 1).collect();
    let mut args = vec![KernelArg::i32(xs.clone()), KernelArg::i32(vec![0; T * 4]), KernelArg::i32(big.clone())];
    kernel.simulate(launch(), &mut args).unwrap();
    let out = args[1].as_i32();
    for t in 0..T {
        let x = xs[t];
        let offset = if x < 0 { t * 8 } else { t * 4 + 12 } as i32;
        let pair = 700 + if x < 0 { x * 2 } else { x - 1 };
        let state = (0..t as i32 % 4).sum::<i32>() * 10 + 5;
        let at = offset as usize;
        assert_eq!(out[t * 4..t * 4 + 4], [offset, pair, state, big[at] + big[at + 3]], "thread {t}");
    }
}

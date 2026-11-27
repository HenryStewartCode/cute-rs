//! Divisibility facts (`assume`, `get_divisibility` and what arithmetic
//! keeps of them), 64-bit index leaves and `FastDivmod` on the host.

mod feature_kernels;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::prelude::*;

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>().cloned().or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string())).unwrap()
}

/// Facts flow through value arithmetic: a sum or difference keeps the gcd,
/// a product the product, a shift left by k multiplies by 2^k, a constant
/// mask keeps its power of two, a remainder by a constant the gcd, and a
/// floor division by a constant keeps the quotient of the divisor only when
/// the constant divides it.
#[test]
fn facts_flow_through_value_arithmetic() {
    Kernel::build("facts", || {
        let a = assume(param_scalar::<i32>("a"), Some(12), false);
        let b = assume(param_scalar::<i32>("b"), Some(8), false);
        let u = param_scalar::<i32>("u");
        assert_eq!(get_divisibility(&a), 12);
        assert_eq!(get_divisibility(&u), 1);
        assert_eq!(get_divisibility(&(a + b)), 4);
        assert_eq!(get_divisibility(&(a - b)), 4);
        assert_eq!(get_divisibility(&(a * b)), 96);
        assert_eq!(get_divisibility(&(a * 3)), 36);
        assert_eq!(get_divisibility(&(u * 6)), 6);
        assert_eq!(get_divisibility(&(a << 2)), 48);
        assert_eq!(get_divisibility(&(u & !7)), 8);
        assert_eq!(get_divisibility(&(a & u)), 4);
        assert_eq!(get_divisibility(&(a % 8)), 4);
        assert_eq!(get_divisibility(&(a / 4)), 3);
        assert_eq!(get_divisibility(&(a / 8)), 1);
        assert_eq!(get_divisibility(&a.max(b)), 4);
        assert_eq!(get_divisibility(&(a + u)), 1);
        // Widening keeps the facts; an assumption adds to what is known.
        assert_eq!(get_divisibility(&a.to::<i64>()), 12);
        assert_eq!(get_divisibility(&assume(a, Some(8), false)), 24);
        let wide = assume(param_scalar::<i64>("w"), Some(16), true);
        assert_eq!(get_divisibility(&(wide * 2)), 32);
        // Static values are their own divisibility.
        assert_eq!(get_divisibility(&48usize), 48);
        assert_eq!(get_divisibility(&Int32::from(20)), 20);
        // No divby and no pow2: unchanged, and no instruction.
        assert_eq!(assume(u, None, false).id(), u.id());
    });
}

/// The same rules on index expressions, including floor division by a
/// constant, and the printing of a dynamic integer's facts.
#[test]
fn facts_in_index_expressions() {
    Kernel::build("index_facts", || {
        let m = IndexExpr::from(param_extent("m", 8));
        let n = IndexExpr::from(param_scalar::<i32>("n"));
        assert_eq!(get_divisibility(&m), 8);
        assert_eq!(get_divisibility(&(m.clone() * 6usize)), 48);
        assert_eq!(get_divisibility(&IndexExpr::Div(Box::new(m.clone() * 6usize), Box::new(IndexExpr::Static(4)))), 12);
        assert_eq!(get_divisibility(&IndexExpr::Div(Box::new(m.clone()), Box::new(IndexExpr::Static(3)))), 1);
        assert_eq!(get_divisibility(&(m.clone() + n.clone())), 1);
        let k = assume(n.clone() * 3usize, Some(4), false);
        assert_eq!(get_divisibility(&k), 12);
        assert_eq!(get_divisibility(&assume(IndexExpr::Static(64), Some(16), true)), 64);
        // CuTe DSL's notation, which leaves out a divisor of 1.
        assert_eq!(m.to_string(), "?{div=8}");
        assert_eq!(n.to_string(), "?");
        let s = IndexExpr::from(param_extent_i64("s", 4));
        assert_eq!(s.to_string(), "?{i64 div=4}");
        assert_eq!((n.clone() * s.clone()).to_string(), "?{i64 div=4}");
        assert_eq!(IndexExpr::from(param_extent_i64("t", 1)).to_string(), "?{i64}");
        assert_eq!(Layout::new((m.clone(), 4), (1, s.clone())).to_string(), "(?{div=8},4):(1,?{i64 div=4})");
    });
}

/// An `i64` leaf, or a static value beyond `i32`, makes an index expression
/// 64-bit; its 32-bit parts are computed in 32 bits and widened.
#[test]
fn index_widths() {
    Kernel::build("widths", || {
        let t = IndexExpr::from(thread_idx().x);
        let s = IndexExpr::from(param_extent_i64("s", 1));
        assert_eq!(t.width(), 32);
        assert_eq!((t.clone() * 4usize).width(), 32);
        assert_eq!((t.clone() * s.clone()).width(), 64);
        assert_eq!((t.clone() + (1usize << 33)).width(), 64);
        let wide: Int64 = (t.clone() % 16usize * s + t.clone() / 16usize).into_value();
        let _ = wide;
        let msg = catch_unwind(AssertUnwindSafe(|| {
            let _: Int32 = (t.clone() + (1usize << 33)).into_value();
        }))
        .expect_err("a 64-bit index expression is not an Int32");
        let msg = msg.downcast_ref::<String>().unwrap();
        assert!(msg.contains("is 64-bit") && msg.contains("converts into an Int64"), "{msg}");
    });
}

/// The vector copy of `assume_vector_copy` is refused without the
/// assumption; with it the load and store are 128-bit, and LLVM is told
/// the divisor too.
#[test]
fn assume_enables_a_vector_copy() {
    let msg = build_panic(|| {
        Kernel::build("unproven", || {
            let n = param_scalar::<i32>("n");
            let layout = Layout::new((128, 2), (1, n - n % 4));
            let x = make_tensor(&param_ptr::<f32>("x"), layout).slice((.., 1));
            let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
            let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
            let thr = tiled.get_slice(thread_idx().x);
            let t_x = thr.partition_s(&x);
            copy(&tiled, &t_x, &make_fragment_like::<f32, _>(&t_x));
        });
    });
    assert!(msg.contains("cannot prove that the source offset"), "{msg}");
    let case = feature_kernels::assume_vector_copy();
    assert_eq!(case.kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Assume { .. })), 1);
    let mlir = case.kernel.to_mlir();
    assert!(mlir.contains("vector<4xf32>"), "{mlir}");
    // A power-of-two divisor is passed on to LLVM.
    assert!(mlir.contains("arith.andi %v3, %amask") && mlir.contains("llvm.intr.assume %atest"), "{mlir}");
    // It costs no instruction: the PTX is no longer than without it.
    let Some(tc) = Toolchain::discover() else { return };
    let without: String = mlir
        .lines()
        .filter(|l| !["%amask", "%alow", "%azero", "%atest"].iter().any(|n| l.contains(n)))
        .map(|l| format!("{l}\n"))
        .collect();
    let count = |mlir: &str| {
        let ptx = tc.compile_ptx(mlir, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
        ptx.lines().filter(|l| l.starts_with('\t') && l[1..].starts_with(|c: char| c.is_ascii_lowercase())).count()
    };
    assert!(count(&mlir) <= count(&without));
}

/// An unused assumption is removed with its result.
#[test]
fn unused_assumption_is_removed() {
    let kernel = Kernel::build("unused_assume", || {
        let out = param_tensor::<i32>("out", Layout::col_major(1));
        let n = param_scalar::<i32>("n");
        let _ = assume(n, Some(4), false);
        out.store(0, n);
    });
    assert_eq!(kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Assume { .. })), 0);
}

/// `int64_strides` computes its offsets in 64 bits and the thread's
/// coordinates in 32: in the MLIR, and in the PTX when a toolchain is
/// available.
#[test]
fn int64_offsets_in_mlir_and_ptx() {
    let case = feature_kernels::int64_strides();
    let mlir = case.kernel.to_mlir();
    assert!(mlir.contains("arith.remsi %v1, %v4 : i32") && mlir.contains("arith.divsi %v1, %v4 : i32"), "{mlir}");
    assert!(mlir.contains("arith.extsi %v5 : i32 to i64") && mlir.contains(": i64 to index"), "{mlir}");
    assert!(mlir.contains("arith.muli %v7, %v0 : i64") && mlir.contains("arith.addi %v8, %v9 : i64"), "{mlir}");
    let Some(tc) = Toolchain::discover() else { return };
    let ptx = case.kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
    assert!(ptx.contains("and.b32") && ptx.contains("shr.u32"), "{ptx}");
    assert!(ptx.contains("mad.lo.s64") || ptx.contains("mul.lo.s64"), "{ptx}");
    assert!(!ptx.contains("rem.s64") && !ptx.contains("div.s64"), "{ptx}");
}

/// A 64-bit stride beyond `i32` with only small offsets used: the simulator
/// computes the offset in 64 bits and bounds-checks it.
#[test]
fn int64_offsets_in_the_simulator() {
    let kernel = Kernel::builder("far_stride").block_dim([4, 1, 1]).build(|| {
        let s = param_extent_i64("s", 1);
        let x = make_tensor(&param_ptr::<i32>("x"), Layout::new((4, 2), (1, IndexExpr::from(s))));
        let t = thread_idx().x;
        x.store((t, 1), t + 1);
    });
    let cfg = LaunchConfig::new([1, 1, 1], [4, 1, 1]);
    let mut args = vec![KernelArg::scalar(10i64), KernelArg::buffer(&[0i32; 16])];
    kernel.simulate(cfg, &mut args).unwrap();
    assert_eq!(&args[1].to_vec::<i32>()[10..14], &[1, 2, 3, 4]);
    let mut args = vec![KernelArg::scalar(1i64 << 32), KernelArg::buffer(&[0i32; 16])];
    let err = kernel.simulate(cfg, &mut args).unwrap_err().to_string();
    assert!(err.contains(&format!("at index {}", 1u64 << 32)) && err.contains("out of bounds"), "{err}");
}

/// Atomics and reductions take a 64-bit index like every other buffer access.
#[test]
fn atomics_with_64_bit_indices() {
    let kernel = Kernel::builder("far_atomics").block_dim([4, 1, 1]).build(|| {
        let s = param_extent_i64("s", 1);
        let x = param_ptr::<i32>("x");
        let p = x.offset(IndexExpr::from(s));
        let _ = atomic_add(&p, 1i32);
        red(&p.offset(1usize), 2i32, RedOp::Add, DType::I32, AtomicOptions::default());
    });
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("i64"), "{mlir}");
    let mut args = vec![KernelArg::scalar(10i64), KernelArg::buffer(&[0i32; 16])];
    kernel.simulate(LaunchConfig::new([1, 1, 1], [4, 1, 1]), &mut args).unwrap();
    assert_eq!(&args[1].to_vec::<i32>()[10..12], &[4, 8]);
    let mut args = vec![KernelArg::scalar(1i64 << 32), KernelArg::buffer(&[0i32; 16])];
    let err = kernel.simulate(LaunchConfig::new([1, 1, 1], [4, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(err.contains("out of bounds"), "{err}");
}

/// A divisor prepared on the host and passed as three scalars.
#[test]
fn host_fast_divmod() {
    let kernel = Kernel::builder("host_fast_divmod").block_dim([64, 1, 1]).build(|| {
        let d = param_fast_divmod("d");
        let out = param_tensor::<i32>("out", Layout::row_major((64, 3)));
        let t = thread_idx().x;
        let n = t * 33_554_431 + 7;
        let (q, r) = d.divmod(n);
        out.store((t, 0), q);
        out.store((t, 1), r);
        out.store((t, 2), d.divisor);
    });
    for divisor in [1, 2, 3, 7, 1000, 65_537, i32::MAX] {
        let mut args: Vec<KernelArg> = KernelArg::fast_divmod(divisor).into();
        args.push(KernelArg::buffer(&[0i32; 64 * 3]));
        kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
        let out = args[3].to_vec::<i32>();
        for t in 0..64 {
            let n = t as i32 * 33_554_431 + 7;
            assert_eq!(&out[t * 3..t * 3 + 3], &[n / divisor, n % divisor, divisor], "{n} divmod {divisor}");
        }
    }
    assert_eq!(FastDivmod::host(7), HostFastDivmod { divisor: 7, multiplier: 2_454_267_027, shift: 2 });
    assert_eq!(FastDivmod::host(7).divmod(100), (14, 2));
}

/// Every divisor the fact rules claim divides the values the simulator
/// computes: random multiples (negative ones too) of the assumed divisors,
/// through each rule and a chain of them.
#[test]
fn claimed_divisors_hold_at_run_time() {
    type Rule = fn(Int32, Int32) -> Int32;
    let rules: [(&str, Rule); 18] = [
        ("a + b", |a, b| a + b),
        ("a - b", |a, b| a - b),
        ("a * b", |a, b| a * b),
        ("a * 10", |a, _| a * 10),
        ("a << 3", |a, _| a << 3),
        ("a & b", |a, b| a & b),
        ("a & !11", |a, _| a & !11),
        ("a & -8", |a, _| a & -8),
        ("a % 6", |a, _| a % 6),
        ("a % -6", |a, _| a % -6),
        ("a % b", |a, b| a % b),
        ("a / 4", |a, _| a / 4),
        ("a / -4", |a, _| a / -4),
        ("a / 6", |a, _| a / 6),
        ("a / b", |a, b| a / b),
        ("max", |a, b| a.max(b)),
        ("min", |a, b| a.min(b)),
        ("chain", |a, b| ((a * 3 - b) / 2 % 12 + (a & Int32::from(0))) << 1),
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for (name, rule) in rules {
        for (da, db) in [(1, 1), (12, 8), (8, 12), (6, 4), (16, 3), (24, 24)] {
            let mut claim = 0;
            let kernel = Kernel::builder("rule").block_dim([32, 1, 1]).build(|| {
                let out = param_tensor::<i32>("out", Layout::col_major(32));
                let xs = param_tensor::<i32>("xs", Layout::col_major(64));
                let t = thread_idx().x;
                let a = assume(xs.load(t), Some(da), false);
                let b = assume(xs.load(t + 32), Some(db), false);
                let c = rule(a, b);
                claim = get_divisibility(&c) as i64;
                out.store(t, c);
            });
            for _ in 0..4 {
                let xs: Vec<i32> = (0..64)
                    .map(|i| {
                        let d = if i < 32 { da } else { db } as i64;
                        let m = (next() % 2001) as i64 - 1000;
                        // A non-zero divisor operand.
                        (if i >= 32 && m == 0 { d } else { m * d }) as i32
                    })
                    .collect();
                let mut args = vec![KernelArg::buffer(&[0i32; 32]), KernelArg::buffer(&xs)];
                kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
                for (t, v) in args[0].to_vec::<i32>().into_iter().enumerate() {
                    let (a, b) = (xs[t], xs[t + 32]);
                    assert_eq!(v as i64 % claim, 0, "{name} with a = {a} (div {da}), b = {b} (div {db}) is {v}");
                }
            }
        }
    }
}

/// Divisors beyond what the arithmetic can hold: a product of two large
/// divisors keeps a power of two that divides it (not a saturated, odd
/// `usize::MAX`), and a power-of-two `divby` beyond the type's range still
/// lowers to a valid `llvm.intr.assume`.
#[test]
fn oversized_divisors() {
    let kernel = Kernel::build("big_divisors", || {
        let out = param_tensor::<i32>("out", Layout::col_major(2));
        let a = IndexExpr::from(assume(param_scalar::<i64>("a"), Some(1 << 40), false));
        let b = IndexExpr::from(assume(param_scalar::<i64>("b"), Some(1 << 40), false));
        let p = a * b;
        assert_eq!(get_divisibility(&p), 1 << 62);
        assert!(p.is_multiple_of(1 << 62) && !p.is_multiple_of(3));
        let n = assume(param_scalar::<i32>("n"), Some(1 << 40), false);
        let m = assume(param_scalar::<i8>("m"), Some(256), false);
        out.store(0, n);
        out.store(1, m.to::<i32>());
    });
    let mlir = kernel.to_mlir();
    assert!(mlir.contains("arith.constant 2147483647 : i32") && mlir.contains("arith.constant 127 : i8"), "{mlir}");
    let Some(tc) = Toolchain::discover() else { return };
    kernel.compile_ptx(&tc, Arch::Sm80).unwrap_or_else(|e| panic!("{e}"));
}

/// A product with a constant 0 is always 0, which every number divides: it used to count as
/// divisible only by 1, which failed alignment checks on offsets built from it.
#[test]
fn a_product_with_zero_divides_by_anything() {
    Kernel::build("zero_product", || {
        let u = param_scalar::<i32>("u");
        let a = assume(param_scalar::<i32>("a"), Some(16), false);
        // The product with a literal zero is what is being tested.
        #[allow(clippy::erasing_op)]
        let z = u * 0;
        assert_eq!(get_divisibility(&(z + a)), 16);
        assert_eq!(get_divisibility(&(z * 128 + a)), 16);
        assert_eq!(get_divisibility(&((z << 3) + a)), 16);
        assert_eq!(get_divisibility(&(u + a)), 1);
    });
}

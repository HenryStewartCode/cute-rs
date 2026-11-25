//! TMEM loads with a row reduction, packed f32x2 arithmetic, the Rubin mixed-precision packed
//! forms and the UE5M3 conversion (roadmap part m10-ldred-packed-math): kernels in the
//! simulator against plain Rust, and the generated MLIR and PTX where a toolchain has them.
//! The `.f32x2` forms assemble for sm_100a (see `tests/ptxas.rs`); `tcgen05.ld.red` is PTX 8.8
//! and the mixed forms and `.ue5m3x2` PTX 9.4, so those are checked as MLIR text only.

use cute::prelude::*;
use cute::tcgen05::{LdRed16x32bx2Op, LdRed32x32bOp, Repetition, St32x32bOp, TmemLoadRedOp};

/// A deterministic stream of Float32 test values: random bit patterns of every kind, with the
/// special values in the first few.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0003),
        f32::MAX,
        -f32::MAX,
        1.5,
        3.0e-39,
        65504.0,
        65520.0,
        1.0e-8,
    ];
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|i| {
            if i < specials.len() {
                specials[i]
            } else if next() % 4 == 0 {
                // Any bit pattern (NaNs, infinities and subnormals included).
                f32::from_bits(next() as u32)
            } else {
                // A moderate value, scaled by a power of two sometimes.
                let v = (next() % 2001) as f32 / 100.0 - 10.0;
                if next() % 2 == 0 { v } else { v * 2f32.powi((next() % 40) as i32 - 20) }
            }
        })
        .collect()
}

/// The reference of PTX's `min`/`max` over Float32 values.
fn ptx_reduce(vals: &[f32], op: TmemLoadRedOp, nan: bool) -> f32 {
    let abs = matches!(op, TmemLoadRedOp::MaxAbs | TmemLoadRedOp::MinAbs);
    let min = matches!(op, TmemLoadRedOp::Min | TmemLoadRedOp::MinAbs);
    let mut acc: Option<f32> = None;
    for &v in vals {
        let v = if abs { v.abs() } else { v };
        acc = Some(match acc {
            None => v,
            Some(a) => {
                if a.is_nan() || v.is_nan() {
                    if nan || (a.is_nan() && v.is_nan()) {
                        f32::NAN
                    } else if a.is_nan() {
                        v
                    } else {
                        a
                    }
                } else if a == v {
                    // -0 < +0.
                    if min == a.is_sign_negative() { a } else { v }
                } else if (a < v) == min {
                    a
                } else {
                    v
                }
            }
        });
    }
    acc.unwrap()
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// One warp stores `input` (lane `i`'s `cols` values at `input[i * cols..]`) to TMEM with
/// `St32x32bOp`, loads it back with the load-reduce atom `ld` (a `LdRed*Op`, `regs` registers
/// per thread, the TMEM view `layout`) and writes each thread's data and reduction. With
/// `wait` false it reads the loaded registers before the wait.
fn ld_red_kernel(
    name: &'static str,
    cols: usize,
    ld: impl Into<CopyOp> + Copy + 'static,
    regs: usize,
    layout: Layout,
    wait: bool,
) -> Kernel {
    let ld = ld.into();
    Kernel::builder(name).block_dim([32, 1, 1]).target(Arch::Sm110a).build(move || {
        let input = param_tensor::<f32>("input", Layout::col_major(32 * cols));
        let data = param_tensor::<f32>("data", Layout::col_major(32 * regs));
        let red = param_tensor::<f32>("red", Layout::col_major(32));
        let slot = alloc_smem::<u32>(1, 4);
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || alloc_tmem(64, &slot, None, Arch::Sm100a));
        sync_threads();
        let p = retrieve_tmem_ptr::<f32>(16, &slot);
        // Column `k` of the 32 lanes, one `.32x32b.x1` store each.
        let st = make_copy_atom::<f32>(St32x32bOp::new(Repetition::X1));
        let frag = make_fragment::<f32>(1);
        for k in 0..cols {
            frag.store(0, input.load(t * cols as i32 + k as i32));
            copy(&st, &frag, &make_tensor(p.clone() + k as i32, Layout::from_modes([Layout::new(32, 1 << 16)])));
        }
        fence_view_async_tmem_store();
        sync_threads();
        let atom = make_copy_atom::<f32>(ld);
        let d = make_fragment::<f32>(regs);
        let r = make_fragment::<f32>(1);
        copy_atom_call(&atom, &make_tensor(p.clone(), layout.clone()), &[&d, &r]);
        if wait {
            fence_view_async_tmem_load();
        }
        for k in 0..regs {
            data.store(t * regs as i32 + k as i32, d.load(k));
        }
        red.store(t, r.load(0));
        fence_view_async_tmem_load();
        sync_threads();
        if_(warp_idx().eq(0), || dealloc_tmem(&p, 64, None, Arch::Sm100a));
    })
}

fn args_for(input: &[f32], regs: usize) -> Vec<KernelArg> {
    vec![KernelArg::buffer(input), KernelArg::buffer(&vec![0f32; 32 * regs]), KernelArg::buffer(&[0f32; 32])]
}

fn run(kernel: &Kernel, input: &[f32], regs: usize) -> (Vec<f32>, Vec<f32>) {
    let mut args = args_for(input, regs);
    kernel
        .simulate_all_orders(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    (args[1].to_vec::<f32>(), args[2].to_vec::<f32>())
}

const OPS: [TmemLoadRedOp; 4] = [TmemLoadRedOp::Max, TmemLoadRedOp::Min, TmemLoadRedOp::MaxAbs, TmemLoadRedOp::MinAbs];

fn x32(num: usize) -> Layout {
    Layout::new((num, 32), (1, 1 << 16))
}

#[test]
fn ld_red_32x32b_matches_the_reference() {
    let input = values(32 * 64, 7);
    for op in OPS {
        for nan in [false, true] {
            let ld = LdRed32x32bOp::new(Repetition::X64, op).with_nan(nan);
            let kernel = ld_red_kernel("ld_red_32x32b", 64, ld, 64, x32(64), true);
            let (data, red) = run(&kernel, &input, 64);
            for lane in 0..32 {
                let row = &input[lane * 64..lane * 64 + 64];
                assert!(
                    row.iter().zip(&data[lane * 64..]).all(|(a, b)| same_f32(*a, *b)),
                    "{op:?} nan={nan}: lane {lane}'s data"
                );
                let want = ptx_reduce(row, op, nan);
                assert!(same_f32(red[lane], want), "{op:?} nan={nan}: lane {lane}: {} vs {want}", red[lane]);
            }
        }
    }
}

/// Rows without a NaN, with one NaN among finite values, and all NaN.
#[test]
fn ld_red_nan_options() {
    let mut input = values(32 * 64, 11);
    for (lane, row) in input.chunks_mut(64).enumerate() {
        match lane % 4 {
            0 => row.iter_mut().filter(|v| v.is_nan()).for_each(|v| *v = 0.25),
            1 => row[5] = f32::NAN,
            2 => row.fill(f32::NAN),
            _ => {}
        }
    }
    for op in [TmemLoadRedOp::Max, TmemLoadRedOp::MinAbs] {
        for nan in [false, true] {
            let ld = LdRed32x32bOp::new(Repetition::X64, op).with_nan(nan);
            let kernel = ld_red_kernel("ld_red_nan", 64, ld, 64, x32(64), true);
            let (_, red) = run(&kernel, &input, 64);
            for lane in 0..32 {
                let want = ptx_reduce(&input[lane * 64..lane * 64 + 64], op, nan);
                assert!(same_f32(red[lane], want), "{op:?} nan={nan}: lane {lane}: {} vs {want}", red[lane]);
            }
        }
    }
}

/// `.16x32bx2.x8`: thread `i` loads columns `8 * (i / 16)..` of lane `i % 16` and reduces
/// those eight values.
#[test]
fn ld_red_16x32bx2_matches_the_reference() {
    let input = values(32 * 16, 3);
    for op in OPS {
        let ld = LdRed16x32bx2Op::new(Repetition::X8, op);
        let kernel = ld_red_kernel("ld_red_16x32bx2", 16, ld, 8, Layout::new((16, 16), (1, 1 << 16)), true);
        let (data, red) = run(&kernel, &input, 8);
        for i in 0..32 {
            let start = (i % 16) * 16 + 8 * (i / 16);
            let mine = &input[start..start + 8];
            assert!(mine.iter().zip(&data[i * 8..]).all(|(a, b)| same_f32(*a, *b)), "{op:?}: thread {i}'s data");
            let want = ptx_reduce(mine, op, false);
            assert!(same_f32(red[i], want), "{op:?}: thread {i}: {} vs {want}", red[i]);
        }
    }
}

#[test]
fn ld_red_reduction_is_pending_until_the_wait() {
    let ld = LdRed32x32bOp::new(Repetition::X4, TmemLoadRedOp::Max);
    let kernel = ld_red_kernel("ld_red_early", 64, ld, 4, x32(4), false);
    let mut args = args_for(&values(32 * 64, 1), 4);
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err().to_string();
    assert!(e.contains("before a tcgen05.wait::ld"), "{e}");
}

#[test]
fn ld_red_target_rule_and_mlir() {
    let ld = LdRed32x32bOp::new(Repetition::X4, TmemLoadRedOp::MaxAbs).with_nan(true);
    let kernel = ld_red_kernel("ld_red_target", 64, ld, 4, x32(4), true);
    let ir = kernel.ir();
    for arch in [Arch::Sm110a, Arch::Sm103a, Arch::Sm103f, Arch::Sm110f] {
        cute::validate::check_target(ir, arch).unwrap_or_else(|e| panic!("{arch}: {}", e.message));
    }
    let e = cute::validate::check_target(ir, Arch::Sm100a).unwrap_err().message;
    assert!(e.contains("tcgen05.ld.red"), "{e}");
    let text = kernel.to_mlir_with(&MlirOptions { llvm_major: Some(23), ..Default::default() });
    assert!(text.contains("nvvm.tcgen05.ld.red max"), "{text}");
    assert!(text.contains("abs, nan") && text.contains(": vector<4xf32>, f32"), "{text}");
    // The 16x32bx2 form passes its half split as an immediate.
    let ld = LdRed16x32bx2Op::new(Repetition::X8, TmemLoadRedOp::Min);
    let kernel = ld_red_kernel("ld_red_split", 16, ld, 8, Layout::new((16, 16), (1, 1 << 16)), true);
    let split = kernel.to_mlir_with(&MlirOptions { llvm_major: Some(23), ..Default::default() });
    assert!(split.contains("llvm.mlir.constant(8 : i64)") && split.contains("nvvm.tcgen05.ld.red min"), "{split}");
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping the mlir-opt check: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    if tc.llvm_version().is_some_and(|v| v >= 23) {
        for t in [&text, &split] {
            tc.verify(t).unwrap_or_else(|e| panic!("{e}\n{t}"));
        }
        // LLVM 23 lowers it to the PTX 8.8 instruction.
        let ptx = tc.compile_ptx(&text, Arch::Sm110a).unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains("tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.NaN.f32"), "{ptx}");
    } else {
        eprintln!("skipping the mlir-opt check: tcgen05.ld.red needs LLVM 23");
    }
}

// ---------------------------------------------------------------------------------------------
// Packed f32x2 arithmetic.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dir {
    Rn,
    Rz,
    Rm,
    Rp,
}

fn flush(x: f32, ftz: bool) -> f32 {
    if ftz && x != 0.0 && x.abs() < f32::MIN_POSITIVE { 0.0f32.copysign(x) } else { x }
}

fn next_up(x: f32) -> f32 {
    if x == f32::INFINITY {
        x
    } else if x == 0.0 {
        f32::from_bits(1)
    } else if x > 0.0 {
        f32::from_bits(x.to_bits() + 1)
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

/// The exact `a + b` as an unevaluated sum of two `f64`s (TwoSum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// The sum `hi + lo` (exact, finite) rounded to Float32 in direction `dir`; `zero` gives the
/// sign of an exact zero.
fn round_f32(hi: f64, lo: f64, dir: Dir, zero_sign_negative: bool) -> f32 {
    if hi == 0.0 && lo == 0.0 {
        return if zero_sign_negative { -0.0 } else { 0.0 };
    }
    let r = hi as f32;
    // The sign of (exact - r).
    let d = if r == f32::INFINITY {
        -1.0
    } else if r == f32::NEG_INFINITY {
        1.0
    } else {
        (hi - r as f64) + lo
    };
    let positive = hi > 0.0 || (hi == 0.0 && lo > 0.0);
    match dir {
        Dir::Rn => r,
        Dir::Rp => {
            if d > 0.0 {
                next_up(r)
            } else {
                r
            }
        }
        Dir::Rm => {
            if d < 0.0 {
                next_down(r)
            } else {
                r
            }
        }
        Dir::Rz => {
            if positive {
                round_f32(hi, lo, Dir::Rm, false)
            } else {
                round_f32(hi, lo, Dir::Rp, false)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arith {
    Add,
    Sub,
    Mul,
    Fma,
}

/// One element of `add`, `sub`, `mul` or `fma` with direction `dir` and `ftz`, as an f32
/// operation of PTX.
fn ref_f32(op: Arith, a: f32, b: f32, c: f32, dir: Dir, ftz: bool) -> f32 {
    let (a, b, c) = (flush(a, ftz), flush(b, ftz), flush(c, ftz));
    let b = if op == Arith::Sub { -b } else { b };
    let finite = a.is_finite() && b.is_finite() && (op != Arith::Fma || c.is_finite());
    let native = match op {
        Arith::Add | Arith::Sub => a + b,
        Arith::Mul => a * b,
        Arith::Fma => a.mul_add(b, c),
    };
    if !finite || native.is_nan() {
        return native;
    }
    let (hi, lo, zero_neg) = match op {
        Arith::Add | Arith::Sub => {
            let (hi, lo) = two_sum(a as f64, b as f64);
            let (an, bn) = (a.is_sign_negative(), b.is_sign_negative());
            (hi, lo, if a == 0.0 && b == 0.0 && an == bn { an } else { dir == Dir::Rm })
        }
        Arith::Mul => {
            let p = a as f64 * b as f64;
            (p, 0.0, a.is_sign_negative() != b.is_sign_negative())
        }
        Arith::Fma => {
            let p = a as f64 * b as f64;
            let (hi, lo) = two_sum(p, c as f64);
            let p_zero = p == 0.0;
            let neg_p = a.is_sign_negative() != b.is_sign_negative();
            let zero_neg = if p_zero && c == 0.0 && neg_p == c.is_sign_negative() { neg_p } else { dir == Dir::Rm };
            (hi, lo, zero_neg)
        }
    };
    flush(round_f32(hi, lo, dir, zero_neg), ftz)
}

const DIRS: [(Option<Dir>, &str); 5] =
    [(Some(Dir::Rn), "rn"), (Some(Dir::Rz), "rz"), (Some(Dir::Rm), "rm"), (Some(Dir::Rp), "rp"), (None, "none")];

fn rounding(d: Option<Dir>) -> Option<RoundingMode> {
    d.map(|d| match d {
        Dir::Rn => RoundingMode::NearestEven,
        Dir::Rz => RoundingMode::Zero,
        Dir::Rm => RoundingMode::Down,
        Dir::Rp => RoundingMode::Up,
    })
}

const T: usize = 176;

/// Three operand arrays of `2 * T` elements: the first 324 hold every pair of the special
/// values (with a third one), the rest random values.
fn operands() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 2 * T;
    let special = values(18, 0);
    let (mut a, mut b, mut c) = (values(n, 21), values(n, 22), values(n, 23));
    for i in 0..18 * 18 {
        a[i] = special[i % 18];
        b[i] = special[i / 18];
        c[i] = special[(i * 7 + i / 18) % 18];
    }
    (a, b, c)
}

/// A kernel of `T` threads, each doing `op` on two elements with each rounding mode and `ftz`
/// setting (10 variants; fma has no `none`) into `out[variant][thread][element]`.
fn f32x2_kernel(op: Arith) -> Kernel {
    Kernel::builder("packed_f32x2").block_dim([T as u32, 1, 1]).target(Arch::Sm100a).build(move || {
        let a = param_tensor::<f32>("a", Layout::col_major(2 * T));
        let b = param_tensor::<f32>("b", Layout::col_major(2 * T));
        let c = param_tensor::<f32>("c", Layout::col_major(2 * T));
        let out = param_tensor::<f32>("out", Layout::col_major(10 * 2 * T));
        let t = thread_idx().x * 2;
        let pair = |x: &Tensor<f32>| (x.load(t), x.load(t + 1));
        let mut variant = 0;
        for ftz in [false, true] {
            for (dir, _) in DIRS {
                if op == Arith::Fma && dir.is_none() {
                    continue;
                }
                let opts = PackedF32Options { rnd: rounding(dir), ftz };
                let (x, y) = (pair(&a), pair(&b));
                let r = match op {
                    Arith::Add => add_packed_f32x2(x, y, opts),
                    Arith::Sub => sub_packed_f32x2(x, y, opts),
                    Arith::Mul => mul_packed_f32x2(x, y, opts),
                    Arith::Fma => fma_packed_f32x2(x, y, pair(&c), opts),
                };
                out.store(variant * 2 * T as i32 + t, r.0);
                out.store(variant * 2 * T as i32 + t + 1, r.1);
                variant += 1;
            }
        }
    })
}

#[test]
fn packed_f32x2_matches_the_reference_in_every_mode() {
    let (a, b, c) = operands();
    for op in [Arith::Add, Arith::Sub, Arith::Mul, Arith::Fma] {
        let kernel = f32x2_kernel(op);
        let mut args = vec![
            KernelArg::buffer(&a),
            KernelArg::buffer(&b),
            KernelArg::buffer(&c),
            KernelArg::buffer(&vec![0f32; 10 * 2 * T]),
        ];
        kernel
            .simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args)
            .unwrap_or_else(|e| panic!("{op:?}: {e}"));
        let out = args[3].to_vec::<f32>();
        let mut variant = 0;
        for ftz in [false, true] {
            for (dir, name) in DIRS {
                if op == Arith::Fma && dir.is_none() {
                    continue;
                }
                // No `.rnd` rounds to nearest.
                let d = dir.unwrap_or(Dir::Rn);
                for i in 0..2 * T {
                    let want = ref_f32(op, a[i], b[i], c[i], d, ftz);
                    let got = out[variant * 2 * T + i];
                    assert!(
                        same_f32(got, want),
                        "{op:?}.{name} ftz={ftz}: ({:e}, {:e}, {:e}) gave {got:e} ({:#010x}), not {want:e} ({:#010x})",
                        a[i],
                        b[i],
                        c[i],
                        got.to_bits(),
                        want.to_bits()
                    );
                }
                variant += 1;
            }
        }
    }
}

#[test]
fn packed_f32x2_mlir_and_deprecated_exp() {
    let kernel = f32x2_kernel(Arith::Fma);
    let text = kernel.to_mlir();
    for w in ["fma.rn.f32x2", "fma.rz.f32x2", "fma.rm.ftz.f32x2", "fma.rp.ftz.f32x2"] {
        assert!(text.contains(w), "no `{w}` in\n{text}");
    }
    let add = f32x2_kernel(Arith::Add).to_mlir();
    assert!(add.contains("add.f32x2") && add.contains("add.ftz.f32x2") && add.contains("add.rn.f32x2"), "{add}");
    // exp_packed_f32x2: a multiplication by log2(e) and two fast exp2s.
    #[allow(deprecated)]
    let exp = Kernel::build("exp_pair", || {
        let x = param_tensor::<f32>("x", Layout::col_major(64));
        let t = thread_idx().x * 2;
        let (a, b) = exp_packed_f32x2((x.load(t), x.load(t + 1)));
        x.store(t, a);
        x.store(t + 1, b);
    });
    let mut args = vec![KernelArg::buffer(&(0..64).map(|i| i as f32 / 16.0 - 2.0).collect::<Vec<_>>())];
    exp.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    for (i, got) in args[0].to_vec::<f32>().into_iter().enumerate() {
        let want = (i as f32 / 16.0 - 2.0).exp();
        assert!((got - want).abs() <= 1e-5 * want, "e^{} = {got}, not {want}", i as f32 / 16.0 - 2.0);
    }
    let text = exp.to_mlir();
    assert!(text.contains("mul.rn.f32x2") && text.contains("ex2.approx.ftz.f32"), "{text}");
}

#[test]
fn packed_f32x2_needs_sm100_and_fma_needs_a_rounding_mode() {
    let kernel = f32x2_kernel(Arith::Mul);
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm90a).unwrap_err().message;
    assert!(e.contains("f32x2"), "{e}");
    let panic = std::panic::catch_unwind(|| {
        Kernel::build("fma_no_rounding", || {
            let x = param_tensor::<f32>("x", Layout::col_major(64));
            let p = (x.load(0), x.load(1));
            let _ = fma_packed_f32x2(p, p, p, PackedF32Options::no_rounding());
        })
    });
    assert!(panic.is_err());
}

// ---------------------------------------------------------------------------------------------
// The Rubin mixed-precision packed forms (sm_107f, PTX 9.4): the simulator and MLIR text only.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Half {
    F16,
    Bf16,
}

fn pow2(k: i32) -> f64 {
    assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// A Float16 or BFloat16 as an `f32` (exactly).
fn half_to_f32(bits: u16, h: Half) -> f32 {
    match h {
        Half::Bf16 => f32::from_bits((bits as u32) << 16),
        Half::F16 => {
            let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let (e, m) = ((bits >> 10 & 0x1f) as i32, (bits & 0x3ff) as f32);
            match e {
                0 => sign * m * 2f32.powi(-24),
                31 if m == 0.0 => sign * f32::INFINITY,
                31 => f32::NAN,
                _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
            }
        }
    }
}

fn is_half_nan(bits: u16, h: Half) -> bool {
    match h {
        Half::F16 => bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0,
        Half::Bf16 => bits & 0x7f80 == 0x7f80 && bits & 0x7f != 0,
    }
}

/// The exact sum `hi + lo` rounded to nearest even in a half format: its bits.
fn round_half(hi: f64, lo: f64, h: Half) -> u16 {
    let (p, emin, max, inf, sign_bit) = match h {
        Half::F16 => (11, -14, 65504.0f64, 0x7c00u16, 0x8000u16),
        Half::Bf16 => (8, -126, f32::from_bits(0x7f7f_0000) as f64, 0x7f80, 0x8000),
    };
    let negative = hi < 0.0 || (hi == 0.0 && lo < 0.0);
    let sign = if negative { sign_bit } else { 0 };
    let (hi, lo) = if negative { (-hi, -lo) } else { (hi, lo) };
    if hi == 0.0 {
        return sign;
    }
    let e = ((hi.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let q = e.max(emin) - (p - 1);
    let scaled = hi * pow2(-q);
    let fl = scaled.floor();
    let diff = (scaled - fl - 0.5) + lo * pow2(-q);
    let n = fl + if diff > 0.0 || (diff == 0.0 && fl % 2.0 == 1.0) { 1.0 } else { 0.0 };
    let v = n * pow2(q);
    if v > max {
        return sign | inf;
    }
    if v == 0.0 {
        return sign;
    }
    match h {
        Half::Bf16 => sign | ((v as f32).to_bits() >> 16) as u16,
        Half::F16 => {
            if v < pow2(-14) {
                sign | (v * pow2(24)) as u16
            } else {
                let e = ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023;
                sign | (((e + 15) as u16) << 10) | ((v / pow2(e) - 1.0) * 1024.0) as u16
            }
        }
    }
}

/// `a ± b` (Float32 values) rounded once to a half format.
fn ref_to_half(a: f32, b: f32, sub: bool, h: Half) -> u16 {
    let b = if sub { -b } else { b };
    let native = a + b;
    if !(a.is_finite() && b.is_finite()) || native.is_nan() {
        return match (native.is_nan(), native.is_sign_negative(), h) {
            (true, _, Half::F16) => 0x7fff,
            (true, _, Half::Bf16) => 0x7fff,
            (false, neg, Half::F16) => {
                if neg {
                    0xfc00
                } else {
                    0x7c00
                }
            }
            (false, neg, Half::Bf16) => {
                if neg {
                    0xff80
                } else {
                    0x7f80
                }
            }
        };
    }
    if native == 0.0 {
        return if native.is_sign_negative() { 0x8000 } else { 0 };
    }
    let (hi, lo) = two_sum(a as f64, b as f64);
    round_half(hi, lo, h)
}

fn to_half_kernel() -> Kernel {
    Kernel::builder("mixed_to_half").block_dim([T as u32, 1, 1]).target(Arch::Sm107f).build(|| {
        let a = param_tensor::<f32>("a", Layout::col_major(2 * T));
        let b = param_tensor::<f32>("b", Layout::col_major(2 * T));
        let out = param_tensor::<i32>("out", Layout::col_major(4 * T));
        let t = thread_idx().x;
        let (i, j) = (t * 2, t * 2 + 1);
        let (x, y) = ((a.load(i), a.load(j)), (b.load(i), b.load(j)));
        let f16 = |(lo, hi): (Float16, Float16)| -> Int32 { pack([lo, hi]) };
        let bf16 = |(lo, hi): (BFloat16, BFloat16)| -> Int32 { pack([lo, hi]) };
        out.store(t, f16(add_packed_f16x2_f32x2_f32x2(x, y)));
        out.store(t + T as i32, f16(sub_packed_f16x2_f32x2_f32x2(x, y)));
        out.store(t + 2 * T as i32, bf16(add_packed_bf16x2_f32x2_f32x2(x, y)));
        out.store(t + 3 * T as i32, bf16(sub_packed_bf16x2_f32x2_f32x2(x, y)));
    })
}

#[test]
fn mixed_packed_to_half_rounds_the_float32_sum_once() {
    let (a, b, _) = operands();
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&b), KernelArg::buffer(&vec![0i32; 4 * T])];
    to_half_kernel()
        .simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = args[2].to_vec::<i32>();
    for (k, (sub, h)) in
        [(false, Half::F16), (true, Half::F16), (false, Half::Bf16), (true, Half::Bf16)].into_iter().enumerate()
    {
        for t in 0..T {
            let word = out[k * T + t] as u32;
            for lane in 0..2 {
                let got = (word >> (16 * lane)) as u16;
                let want = ref_to_half(a[2 * t + lane], b[2 * t + lane], sub, h);
                assert!(
                    got == want || (is_half_nan(got, h) && is_half_nan(want, h)),
                    "{h:?} sub={sub}: {:e} {} {:e} gave {got:#06x}, not {want:#06x}",
                    a[2 * t + lane],
                    if sub { "-" } else { "+" },
                    b[2 * t + lane]
                );
            }
        }
    }
}

/// A sum that a rounding to Float32 first would round to the wrong half value (the
/// "double rounding" case): 1 + 2^-11 + 2^-40 is just above the half-way point between two
/// Float16 values, but its Float32 sum with the tiny part dropped is exactly half-way.
#[test]
fn mixed_packed_to_half_does_not_round_twice() {
    let (a, b) = (1.0f32 + 2f32.powi(-11), 2f32.powi(-40));
    assert_eq!((a + b).to_bits(), a.to_bits(), "the Float32 sum drops the tiny part");
    let want = ref_to_half(a, b, false, Half::F16);
    assert_eq!(want, 0x3c01, "above the tie: rounds up");
    let mut inputs_a = vec![0.0f32; 2 * T];
    let mut inputs_b = vec![0.0f32; 2 * T];
    inputs_a[0] = a;
    inputs_b[0] = b;
    let mut args =
        vec![KernelArg::buffer(&inputs_a), KernelArg::buffer(&inputs_b), KernelArg::buffer(&vec![0i32; 4 * T])];
    to_half_kernel()
        .simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[2].to_vec::<i32>()[0] as u32 & 0xffff, want as u32);
}

/// Each form of the kind that takes a half pair and Float32 pairs: add, sub and fma, with
/// Float16 and BFloat16, in the four directions and with none.
fn from_half_kernel() -> Kernel {
    Kernel::builder("mixed_from_half").block_dim([T as u32, 1, 1]).target(Arch::Sm107f).build(|| {
        let ha = param_tensor::<i32>("ha", Layout::col_major(T));
        let b = param_tensor::<f32>("b", Layout::col_major(2 * T));
        let c = param_tensor::<f32>("c", Layout::col_major(2 * T));
        let out = param_tensor::<f32>("out", Layout::col_major(28 * 2 * T));
        let t = thread_idx().x;
        let (i, j) = (t * 2, t * 2 + 1);
        let (y, z) = ((b.load(i), b.load(j)), (c.load(i), c.load(j)));
        let mut variant = 0i32;
        let mut put = |r: (Float32, Float32)| {
            out.store(variant * 2 * T as i32 + i, r.0);
            out.store(variant * 2 * T as i32 + j, r.1);
            variant += 1;
        };
        for (dir, _) in DIRS {
            let rnd = rounding(dir);
            put(add_packed_f32x2_f16x2_f32x2(ha.load(t), y, rnd));
            put(sub_packed_f32x2_f16x2_f32x2(ha.load(t), y, rnd));
            put(add_packed_f32x2_bf16x2_f32x2(ha.load(t), y, rnd));
            put(sub_packed_f32x2_bf16x2_f32x2(ha.load(t), y, rnd));
            if rnd.is_some() {
                put(fma_packed_f32x2_f16x2_f32x2_f32x2(ha.load(t), y, z, rnd));
                put(fma_packed_f32x2_bf16x2_f32x2_f32x2(ha.load(t), y, z, rnd));
            }
        }
    })
}

#[test]
fn mixed_packed_from_half_widens_then_rounds_once() {
    let (_, b, c) = operands();
    // Half pairs: the bits of Float32 values (any patterns, NaNs and infinities included).
    let ha: Vec<u32> = values(T, 5).iter().map(|f| f.to_bits()).collect();
    let mut args = vec![
        KernelArg::buffer(&ha.iter().map(|&x| x as i32).collect::<Vec<_>>()),
        KernelArg::buffer(&b),
        KernelArg::buffer(&c),
        KernelArg::buffer(&vec![0f32; 28 * 2 * T]),
    ];
    from_half_kernel()
        .simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = args[3].to_vec::<f32>();
    let mut variant = 0;
    for (dir, name) in DIRS {
        let d = dir.unwrap_or(Dir::Rn);
        let forms: [(Arith, Half); 6] = [
            (Arith::Add, Half::F16),
            (Arith::Sub, Half::F16),
            (Arith::Add, Half::Bf16),
            (Arith::Sub, Half::Bf16),
            (Arith::Fma, Half::F16),
            (Arith::Fma, Half::Bf16),
        ];
        for (op, h) in forms {
            if op == Arith::Fma && dir.is_none() {
                continue;
            }
            for i in 0..2 * T {
                let half = half_to_f32((ha[i / 2] >> (16 * (i % 2))) as u16, h);
                let want = ref_f32(op, half, b[i], c[i], d, false);
                let got = out[variant * 2 * T + i];
                assert!(
                    same_f32(got, want),
                    "{op:?} {h:?}.{name}: ({half:e}, {:e}, {:e}) gave {got:e}, not {want:e}",
                    b[i],
                    c[i]
                );
            }
            variant += 1;
        }
    }
}

#[test]
fn mixed_packed_mlir_and_target() {
    for kernel in [to_half_kernel(), from_half_kernel()] {
        cute::validate::check_target(kernel.ir(), Arch::Sm107f).unwrap();
        let e = cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap_err().message;
        assert!(e.contains("mixed-precision packed"), "{e}");
    }
    let text = to_half_kernel().to_mlir();
    for w in ["add.rn.f16x2.f32x2", "sub.rn.f16x2.f32x2", "add.rn.bf16x2.f32x2", "sub.rn.bf16x2.f32x2"] {
        assert!(text.contains(w), "no `{w}` in\n{text}");
    }
    let text = from_half_kernel().to_mlir();
    for w in [
        "add.rn.f32x2.f16x2",
        "sub.rz.f32x2.bf16x2",
        "add.rm.f32x2.bf16x2",
        "add.f32x2.f16x2",
        "fma.rp.f32x2.f16x2",
        "fma.rn.f32x2.bf16x2",
    ] {
        assert!(text.contains(w), "no `{w}` in\n{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// UE5M3 conversion (sm_107f, PTX 9.4).

/// The 255 finite UE5M3 values, by code (code 0xff is the NaN).
fn ue5m3_value(code: u8) -> f64 {
    let (e, m) = ((code >> 3) as i32, (code & 7) as f64);
    if e == 0 { m / 8.0 * pow2(-14) } else { (1.0 + m / 8.0) * pow2(e - 15) }
}

/// `cvt.rp.satfinite.ue5m3x2.f32` of one value: the smallest UE5M3 value that is at least the
/// magnitude, saturating at the largest (`0xfe`); the sign is ignored and a NaN gives `0xff`.
fn ref_ue5m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0xff;
    }
    let a = (x as f64).abs();
    (0..=0xfeu8).find(|&c| ue5m3_value(c) >= a).unwrap_or(0xfe)
}

fn ue5m3_kernel() -> Kernel {
    Kernel::builder("cvt_ue5m3").block_dim([T as u32, 1, 1]).target(Arch::Sm107f).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(4 * T));
        let out = param_tensor::<u8>("out", Layout::col_major(4 * T));
        let t = thread_idx().x * 4;
        let src = [0, 1, 2, 3].map(|k| x.load(t + k));
        let q = cvt_f32x4_to_fnv8e5m3x4(src);
        for (k, v) in q.into_iter().enumerate() {
            out.store(t + k as i32, v.bitcast::<u8>());
        }
    })
}

#[test]
fn ue5m3_conversion_matches_the_reference() {
    let mut x = values(4 * T, 31);
    // Edge cases: exact grid values, values just above and below them, the clamp, zeros.
    let edges = [
        0.0,
        -0.0,
        1.0,
        1.0625,
        1.07,
        1.125,
        0.001,
        1e-6,
        2f32.powi(-17),
        2f32.powi(-17) * 1.01,
        2f32.powi(-14),
        114688.0,
        114688.5,
        114689.0,
        120000.0,
        1e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -0.3,
        -1e-9,
        -114688.0,
        -200000.0,
        0.24,
        0.25,
        0.2501,
        7.5,
        7.75,
        8.0,
        8.01,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ];
    x[..edges.len()].copy_from_slice(&edges);
    // Every code, its upper neighbour's midpoint, and values on either side of each midpoint.
    for c in 0..=0xfeu8 {
        let (lo, hi) = (ue5m3_value(c), ue5m3_value((c + 1).min(0xfe)));
        let mid = ((lo + hi) / 2.0) as f32;
        x.extend([lo as f32, mid, f32::from_bits(mid.to_bits() + 1), f32::from_bits(mid.to_bits() - 1)]);
    }
    x.resize(x.len().div_ceil(4 * T) * 4 * T, 0.0);
    let kernel = ue5m3_kernel();
    for chunk in x.chunks(4 * T) {
        let mut args = vec![KernelArg::buffer(chunk), KernelArg::buffer(&vec![0u8; 4 * T])];
        kernel.simulate(LaunchConfig::new([1, 1, 1], [T as u32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
        for (v, got) in chunk.iter().zip(args[1].to_vec::<u8>()) {
            assert_eq!(got, ref_ue5m3(*v), "{v:e} ({:#010x})", v.to_bits());
        }
    }
}

#[test]
fn ue5m3_conversion_mlir_and_target() {
    let kernel = ue5m3_kernel();
    let text = kernel.to_mlir();
    assert!(text.matches("cvt.rp.satfinite.ue5m3x2.f32").count() == 2, "{text}");
    cute::validate::check_target(kernel.ir(), Arch::Sm107f).unwrap();
    let e = cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap_err().message;
    assert!(e.contains("ue5m3x2"), "{e}");
}

// ---------------------------------------------------------------------------------------------
// The generated PTX: the `.f32x2` kernels for sm_100a, assembled by `ptxas` when it is on PATH
// (`PATH=/usr/local/cuda/bin:$PATH cargo test --test ldred_packed`).

#[test]
fn packed_f32x2_kernels_assemble_for_sm100a() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let ptxas = std::env::var_os("PATH")
        .and_then(|path| std::env::split_paths(&path).map(|dir| dir.join("ptxas")).find(|p| p.is_file()));
    for op in [Arith::Add, Arith::Sub, Arith::Mul, Arith::Fma] {
        let kernel = f32x2_kernel(op);
        let compiled = kernel
            .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm100a))
            .unwrap_or_else(|e| panic!("{op:?}: {e}"));
        let ptx = compiled.ptx().to_string();
        let name = match op {
            Arith::Add => "add",
            Arith::Sub => "sub",
            Arith::Mul => "mul",
            Arith::Fma => "fma",
        };
        for w in [format!("{name}.rz.f32x2"), format!("{name}.rm.ftz.f32x2"), format!("{name}.rp.f32x2")] {
            assert!(ptx.contains(&w), "{op:?}: no `{w}` in the PTX");
        }
        let Some(ptxas) = &ptxas else {
            continue;
        };
        let stem = std::env::temp_dir().join(format!("cute-f32x2-{}-{name}", std::process::id()));
        let (source, cubin) = (stem.with_extension("ptx"), stem.with_extension("cubin"));
        std::fs::write(&source, &ptx).unwrap();
        let output =
            std::process::Command::new(ptxas).arg("-arch=sm_100a").arg(&source).arg("-o").arg(&cubin).output().unwrap();
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cubin);
        assert!(
            output.status.success(),
            "{op:?}: ptxas: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

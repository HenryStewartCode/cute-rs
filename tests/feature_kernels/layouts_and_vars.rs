// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! Layout kernels (dynamic and basis layouts), math cases, Vars and loop hints, carried structs, and dynamic tiles and divmod.

use super::*;
/// Layouts whose extents and strides are runtime values, as CuTe DSL builds
/// them for framework tensors (`(?,?):(?,1)`), evaluated at runtime
/// coordinates. Each thread has its own extents `m` and `n` and leading
/// dimension `ld`, so the divisions by runtime extents in `idx2crd`, in
/// `crd2idx` with a 1-D index into a hierarchical mode, and in the rest
/// modes of a `zipped_divide` (of extent `ceil_div(m, 4)` and
/// `ceil_div(n, 8)`) see many values. Also `size` and `cosize`, a divide of
/// the whole of a layout with runtime extents (whose rest is split over the
/// layout's modes), a composition whose right-hand side has a runtime extent
/// that is compared with a static one at run time (both smaller and larger
/// than it), and a complement of a runtime extent.
pub fn dynamic_layouts() -> Case {
    const COLS: usize = 19;
    let kernel = Kernel::builder("dynamic_layouts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = (xs.load(t) % 1024u32 + t.to::<u32>()).to::<i32>();
        let (m, n) = (x % 13 + 1, x / 13 % 11 + 1);
        let ld = n + x % 3;
        let mut values: Vec<IndexExpr> = Vec::new();

        // A row-major m x n matrix with leading dimension ld: (?,?):(?,1).
        let a = make_layout((m, n), (ld, 1));
        values.push(size(&a, &[]));
        values.push(cosize(&a, &[]));
        // idx2crd and crd2idx at a runtime index below m * n.
        let idx = x * 7 % Int32::from(size(&a, &[]));
        let c = idx2crd(idx, a.shape());
        values.extend(c.flat_leaves());
        values.push(crd2idx(c, &a));

        // 4x8 tiles: static tiles, rest modes of runtime extent.
        let z = zipped_divide(&a, (4, 8));
        let (rm, rn) = (size(&z, &[1, 0]), size(&z, &[1, 1]));
        values.extend([rm.clone(), rn.clone()]);
        let (ti, tj) = (t % 4, t / 4 % 8);
        let (bi, bj) = (x % Int32::from(rm), x / 5 % Int32::from(rn));
        values.push(crd2idx(((ti, tj), (bi, bj)), &z));
        // A 1-D index into the rest mode divides by its runtime extents.
        let rest = z.mode(1);
        let r = x * 3 % Int32::from(size(&rest, &[]));
        values.push(crd2idx(r, &rest));

        // A compact hierarchical layout of runtime extents is the identity
        // on 1-D indices, split by runtime divisions.
        let h = Layout::col_major(((m, 3), n));
        let q = x * 11 % Int32::from(size(&h, &[]));
        values.push(crd2idx(q, &h));
        values.extend(idx2crd(q, h.shape()).flat_leaves());

        // A runtime extent on the right of a composition, 1, 2, 4, 8 or 16,
        // against the 4 rows of (4,8):(8,1): (min(4,e),ceil_div(e,4)):(8,1).
        let e = Int32::from(1) << (x % 5);
        let composed = composition(&Layout::new((4, 8), (8, 1)), &Layout::new(e, 1));
        values.push(size(&composed, &[]));
        values.push(crd2idx(x % e, &composed));
        // A complement of a runtime extent.
        let gaps = complement(&Layout::new(m, 1), 64);
        values.push(size(&gaps, &[]));
        let u = x % Int32::from(size(&gaps, &[]));
        values.push(crd2idx(u, &gaps));

        // The whole of an (8w, n):(ld, 1) matrix divided into runs of 8
        // along its first mode: (8,(w,n)):(ld,(8*ld,1)).
        let w = x % 3 + 1;
        let runs = logical_divide(&make_layout((w * 8, n), (ld, 1)), 8);
        values.push(size(&runs, &[1, 0]));
        values.push(crd2idx((t % 8, (x % w, x / 3 % n)), &runs));

        assert_eq!(values.len(), COLS);
        for (j, v) in values.into_iter().enumerate() {
            out.store((t, j), Int32::from(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = (xs[t] % 1024) as usize + t;
            let (m, n) = (x % 13 + 1, x / 13 % 11 + 1);
            let ld = n + x % 3;
            let idx = x * 7 % (m * n);
            let (rm, rn) = (m.div_ceil(4), n.div_ceil(8));
            let (ti, tj, bi, bj) = (t % 4, t / 4 % 8, x % rm, x / 5 % rn);
            let r = x * 3 % (rm * rn);
            let q = x * 11 % (3 * m * n);
            let e = 1 << (x % 5);
            let p = x % e;
            let reps = 64usize.div_ceil(m);
            let u = x % reps;
            let w = x % 3 + 1;
            let want: [usize; COLS] = [
                m * n,
                (m - 1) * ld + n,
                idx % m,
                idx / m,
                idx % m * ld + idx / m,
                rm,
                rn,
                (bi * 4 + ti) * ld + bj * 8 + tj,
                r % rm * 4 * ld + r / rm * 8,
                q,
                q % m,
                q / m % 3,
                q / (3 * m),
                e,
                p % 4 * 8 + p / 4,
                reps,
                u * m,
                w,
                ((x % w) * 8 + t % 8) * ld + x / 3 % n,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want.map(|v| v as i32), "thread {t}: m={m} n={n} ld={ld}");
        }
    }
    Case { name: "dynamic_layouts", kernel, cfg: launch(1), small_only: false, check }
}

/// Layouts with scaled-basis strides evaluated at runtime coordinates: an
/// identity layout with a runtime extent at a 1-D index, the tiles of a
/// TMA-like coordinate layout `(512,64,2):(1@1,1@0,64@0)` (whose divide
/// scales its basis strides), a hierarchical identity layout at a 1-D index
/// into a nested mode, and a slice whose arithmetic-tuple offset has a
/// runtime entry. Each entry of each tuple is checked against plain Rust.
pub fn basis_layouts() -> Case {
    const COLS: usize = 9;
    let kernel = Kernel::builder("basis_layouts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = (xs.load(t) % 1024u32 + t.to::<u32>()).to::<i32>();
        let mut values: Vec<IndexExpr> = Vec::new();

        // The identity layout of an (m, 8) shape maps a 1-D index i to its
        // coordinate (i % m, i / m).
        let m = x % 7 + 1;
        let id = make_identity_layout((m, 8));
        let i = x * 5 % (m * 8);
        values.extend(id.eval(&i.into()).flat_leaves());

        // The (32, 4) tiles of a TMA-like coordinate layout:
        // ((32,4),(16,16,2)):((1@1,1@0),(32@1,4@0,64@0)).
        let tma = "(512,64,2):(1@1,1@0,64@0)".parse::<Layout>().unwrap();
        let tiles = zipped_divide(&tma, (32, 4));
        assert_eq!(tiles.to_string(), "((32,4),(16,16,2)):((1@1,1@0),(32@1,4@0,64@0))");
        let crd = ((t % 32, x % 4), (x / 4 % 16, x / 64 % 16, t % 2));
        values.extend(tiles.eval(&crd.into()).flat_leaves());

        // A hierarchical identity layout, with a 1-D index into its first mode.
        let h = make_identity_layout(((4, 3), 5));
        values.extend(h.eval(&(x % 12, x % 5).into()).flat_leaves());

        // A slice of a basis layout: the fixed runtime leaf goes to mode 1.
        let s = "(16,8):(1@0,4@1)".parse::<Layout>().unwrap();
        let (kept, offset) = s.slice_and_offset(&(.., x % 8).into());
        assert_eq!(kept.to_string(), "(16):(1@0)");
        values.extend(offset.flat_leaves());

        assert_eq!(values.len(), COLS);
        for (j, v) in values.into_iter().enumerate() {
            out.store((t, j), Int32::from(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = (xs[t] % 1024) as usize + t;
            let m = x % 7 + 1;
            let i = x * 5 % (m * 8);
            // Tile coordinate (a, b) of rest (r0, r1, r2) is gmem coordinate
            // (b + 4 r1 + 64 r2, a + 32 r0).
            let (a, b, r0, r1, r2) = (t % 32, x % 4, x / 4 % 16, x / 64 % 16, t % 2);
            let j = x % 12;
            let want: [usize; COLS] =
                [i % m, i / m, b + 4 * r1 + 64 * r2, a + 32 * r0, j % 4, j / 4, x % 5, 0, 4 * (x % 8)];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want.map(|v| v as i32), "thread {t}: x={x}");
        }
    }
    Case { name: "basis_layouts", kernel, cfg: launch(1), small_only: false, check }
}

unsafe extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

/// Whether `got` is within `ulps` units in the last place (of a format with
/// `p` significand bits and minimum normal exponent `emin`) of `want`;
/// NaN matches NaN and infinities and zeros must be equal.
fn within_ulps(got: f64, want: f64, ulps: f64, p: i32, emin: i32) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    if got == want {
        return true;
    }
    if want.is_infinite() || got.is_infinite() {
        return false;
    }
    let e = (want.abs().log2().floor() as i32).max(emin);
    (got - want).abs() <= ulps * 2f64.powi(e - (p - 1))
}

/// The `cute::math` functions on one float type (precise mode, so the GPU
/// must give the simulator's bits), each held to Rust's `std` (the host's
/// libm) within the function's documented bound plus the host's error.
macro_rules! math_case {
    ($name:ident, $t:ty, $p:expr, $emin:expr) => {
        pub fn $name() -> Case {
            const OPS: usize = 26;
            let kernel = Kernel::builder(concat!("math_", stringify!($t))).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                let ys = param_tensor::<$t>("ys", Layout::col_major(T));
                let out = param_tensor::<$t>("out", Layout::row_major((T, OPS)));
                let t = thread_idx().x;
                let (x, y) = (xs.load(t), ys.load(t));
                let eighth = x * 0.125;
                let one = Value::constant(1.0 as $t);
                let flag = |b: Boolean| select(b, one, Value::constant(0.0 as $t));
                let results: [Value<$t>; OPS] = [
                    math::exp(x, false),
                    math::exp2(x * 10.0, false),
                    math::log(x, false),
                    math::log2(y, false),
                    math::log10(x, false),
                    math::expm1(eighth, false),
                    math::log1p(eighth, false),
                    math::sin(x * 100.0, false),
                    math::cos(x, false),
                    math::tan(x, false),
                    math::asin(eighth, false),
                    math::acos(eighth, false),
                    math::atan(x, false),
                    math::atan2(y, x, false),
                    math::sinh(x, false),
                    math::tanh(x, false),
                    math::asinh(x, false),
                    math::erf(x, false),
                    math::erfc(x, false),
                    math::cbrt(x, false),
                    math::pow(x, y, false),
                    math::fpowi(x, 3, false),
                    math::rem(x, y, false),
                    math::copysign(x, y, false),
                    math::round(x * 2.5, false),
                    flag(math::isnan(x / y, false))
                        + flag(math::isinf(x / y, false)) * 2.0
                        + flag(math::isnormal(x * 1e-38, false)) * 4.0,
                ];
                for (j, v) in results.into_iter().enumerate() {
                    out.store((t, j), v);
                }
            });
            fn check(args: &[KernelArg]) {
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                for t in 0..T {
                    let (x, y) = (xs[t], ys[t]);
                    let eighth = x * 0.125;
                    let normal = (x * 1e-38).is_normal();
                    let classes =
                        ((x / y).is_nan() as i32 + 2 * (x / y).is_infinite() as i32 + 4 * normal as i32) as $t;
                    // (reference, allowed ulps)
                    let want: [($t, f64); OPS] = [
                        (x.exp(), 2.0),
                        ((x * 10.0).exp2(), 2.0),
                        (x.ln(), 2.0),
                        (y.log2(), 2.0),
                        (x.log10(), 2.0),
                        (eighth.exp_m1(), 2.0),
                        (eighth.ln_1p(), 2.0),
                        ((x * 100.0).sin(), 2.5),
                        (x.cos(), 2.5),
                        (x.tan(), 2.5),
                        (eighth.asin(), 2.0),
                        (eighth.acos(), 2.0),
                        (x.atan(), 2.0),
                        (y.atan2(x), 2.5),
                        (x.sinh(), 3.0),
                        (x.tanh(), 3.0),
                        (x.asinh(), 3.0),
                        (unsafe { erf(x as f64) } as $t, 2.5),
                        (unsafe { erfc(x as f64) } as $t, 3.5),
                        (x.cbrt(), 2.0),
                        (x.powf(y), 2.0),
                        (x.powi(3), 3.0),
                        (x % y, 0.0),
                        (x.copysign(y), 0.0),
                        ((x * 2.5).round(), 0.0),
                        (classes, 0.0),
                    ];
                    for (j, (w, ulps)) in want.into_iter().enumerate() {
                        let got = out[t * OPS + j];
                        assert!(
                            within_ulps(got as f64, w as f64, ulps, $p, $emin),
                            "{} function {j} at x = {x:e}, y = {y:e}: {got:e}, not {w:e}",
                            stringify!($t)
                        );
                    }
                }
            }
            Case { name: concat!("math_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

math_case!(math_f32, f32, 24, -126);
math_case!(math_f64, f64, 53, -1022);

/// `cute::math` on Float16 and BFloat16 (computed in Float32, then
/// rounded): within one unit of the 16-bit result of Rust's f32 function.
pub fn math_half() -> Case {
    const OPS: usize = 8;
    let kernel = Kernel::builder("math_half").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<F16>("xs", Layout::col_major(T));
        let ys = param_tensor::<BF16>("ys", Layout::col_major(T));
        let out = param_tensor::<F16>("out", Layout::row_major((T, OPS / 2)));
        let outb = param_tensor::<BF16>("outb", Layout::row_major((T, OPS / 2)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t), ys.load(t));
        // Precise mode only: the GPU test compares these bit for bit (fast
        // tanh, 2^-11 relative, can round differently to Float16 on a
        // conforming GPU; tests/gpu.rs checks it against its bound).
        for (j, v) in [math::exp(x, false), math::sin(x, false), math::log(x, false), math::tanh(x, false)]
            .into_iter()
            .enumerate()
        {
            out.store((t, j), v);
        }
        for (j, v) in [math::exp(y, false), math::erf(y, false), math::atan2(y, y, false), math::pow(y, y, false)]
            .into_iter()
            .enumerate()
        {
            outb.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<F16>(), args[1].to_vec::<BF16>());
        let (out, outb) = (args[2].to_vec::<F16>(), args[3].to_vec::<BF16>());
        for t in 0..T {
            let (x, y) = (xs[t].to_f32(), ys[t].to_f32());
            let want = [x.exp(), x.sin(), x.ln(), x.tanh()];
            for (j, w) in want.into_iter().enumerate() {
                let got = out[t * 4 + j].to_f32() as f64;
                let w = F16::from_f32(w).to_f32() as f64;
                assert!(within_ulps(got, w, 1.0, 11, -14), "f16 function {j} at {x:e}: {got:e}, not {w:e}");
            }
            let want = [y.exp(), unsafe { erf(y as f64) } as f32, y.atan2(y), y.powf(y)];
            for (j, w) in want.into_iter().enumerate() {
                let got = outb[t * 4 + j].to_f32() as f64;
                let w = BF16::from_f32(w).to_f32() as f64;
                assert!(within_ulps(got, w, 1.0, 8, -126), "bf16 function {j} at {y:e}: {got:e}, not {w:e}");
            }
        }
    }
    Case { name: "math_half", kernel, cfg: launch(1), small_only: false, check }
}

carry_struct! {
    /// Statistics carried through a loop in one Var.
    #[derive(Clone, Copy)]
    struct Stats {
        sum: Int32,
        max: Int32,
        count: Int32,
    }
}

/// Mutable variables (`Var`) that runtime loops and branches thread through
/// automatically. Each output column is one pattern; every seventh thread's
/// loops run zero times.
pub fn vars() -> Case {
    const COLS: usize = 23;
    let kernel = Kernel::builder("vars").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let n = t % 7;
        let col = |j: usize, v: Int32| out.store((t, j), v);

        // A running sum.
        let sum = Var::new(x);
        for_range(0, n, 1, |i| sum.set(sum.get() + i * 3 + 1));
        col(0, sum.get());

        // A Var the loop only sets: the last value, or the initial one.
        let last = Var::new(Int32::from(-1));
        for_range(0, n, 1, |i| last.set(i * x));
        col(1, last.get());

        // Nested loops; the inner one runs zero times when i is 0.
        let nested = Var::new(Int32::from(0));
        for_range(0, n, 1, |i| for_range(0, i, 1, |j| nested.set(nested.get() + (i ^ j) + x)));
        col(2, nested.get());

        // Conditional updates inside a loop.
        let count = Var::new(Int32::from(0));
        let best = Var::new(Int32::from(-1));
        for_range(0, n + 2, 1, |i| {
            if_(((x + i) % 3).eq(0), || count.set(count.get() + 1));
            let v = (x ^ (i * 5)) & 15;
            if_(v.gt(best.get()), || best.set(v));
        });
        col(3, count.get());
        col(4, best.get());

        // A Var set in one branch only.
        let odd = Var::new(x);
        if_((x & 1).eq(1), || odd.set(odd.get() * 2 + 1));
        col(5, odd.get());

        // Two Vars, each set in a different branch.
        let (a, b) = (Var::new(Int32::from(1)), Var::new(Int32::from(2)));
        if_else(t.lt(32), || a.set(a.get() + x), || b.set(b.get() - x));
        col(6, a.get());
        col(7, b.get());

        // A while loop whose condition reads one Var and whose body sets
        // another first: the steps of the Collatz iteration.
        let current = Var::new((x & 63) + 1);
        let steps = Var::new(Int32::from(0));
        while_carry(
            (),
            |()| current.get().ne(1),
            |()| {
                steps.set(steps.get() + 1);
                let c = current.get();
                current.set(select((c & 1).eq(0), c / 2, c * 3 + 1));
            },
        );
        col(8, steps.get());

        // A register tensor.
        let frag = make_fragment::<i32>(4);
        for k in 0..4 {
            frag.store(k, x + k);
        }
        let acc = Var::new(frag);
        for_range(0, n, 1, |i| {
            acc.update(|f| {
                for k in 0..4 {
                    f.store(k, f.load(k) * 2 + i * (k + 1));
                }
                f
            })
        });
        let f = acc.get();
        for k in 0..4 {
            col(9 + k, f.load(k));
        }

        // A carry_struct! struct.
        let stats = Var::new(Stats { sum: Int32::from(0), max: Int32::from(-1), count: Int32::from(0) });
        for_range(0, (x & 3) + n, 1, |i| {
            let s = stats.get();
            let v = (x ^ i) & 15;
            stats.set(Stats { sum: s.sum + v, max: s.max.max(v), count: s.count + 1 });
        });
        let s = stats.get();
        col(13, s.sum);
        col(14, s.max);
        col(15, s.count);

        // Explicit loop state and a Var in the same loop.
        let total = Var::new(Int32::from(0));
        let product = for_range_carry(0, n, 1, Int32::from(1), |i, p| {
            total.set(total.get() + p);
            p * 2 + i
        });
        col(16, product);
        col(17, total.get());

        // A Var created inside a loop body is local to it.
        let outer = Var::new(Int32::from(0));
        for_range(0, n, 1, |i| {
            let inner = Var::new(i + x);
            for_range(0, 3, 1, |j| inner.set(inner.get() * 2 + j));
            outer.set(outer.get() + inner.get());
        });
        col(18, outer.get());

        // A tuple holding a float (scaling by 0.5 is exact, so contracting
        // the multiply and add into an FMA cannot change the result).
        let half = x.to::<f32>() * 0.5;
        let pair = Var::new((Int32::from(0), Float32::from(1.0)));
        for_range(0, n, 1, |i| {
            let (p, q) = pair.get();
            pair.set((p + i, q * 0.5 + half));
        });
        let (p, q) = pair.get();
        col(19, p);
        col(20, q.bitcast::<i32>());

        // A Var set in an if inside a loop inside an if.
        let deep = Var::new(Int32::from(0));
        if_(t.ge(8), || for_range(0, n, 1, |i| if_((i & 1).eq(0), || deep.set(deep.get() + i + x))));
        col(21, deep.get());

        // A Var the loop only reads.
        let triple = Var::new(x * 3);
        let reads = Var::new(Int32::from(0));
        for_range(0, n, 1, |_| reads.set(reads.get() + triple.get()));
        col(22, reads.get() - triple.get());
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = xs[t];
            let n = (t % 7) as i32;
            let mut want = [0i32; COLS];

            let mut sum = x;
            for i in 0..n {
                sum = sum.wrapping_add(i * 3 + 1);
            }
            want[0] = sum;

            want[1] = if n == 0 { -1 } else { (n - 1).wrapping_mul(x) };

            let mut nested = 0i32;
            for i in 0..n {
                for j in 0..i {
                    nested = nested.wrapping_add(i ^ j).wrapping_add(x);
                }
            }
            want[2] = nested;

            let (mut count, mut best) = (0, -1);
            for i in 0..n + 2 {
                if x.wrapping_add(i) % 3 == 0 {
                    count += 1;
                }
                best = best.max((x ^ (i * 5)) & 15);
            }
            (want[3], want[4]) = (count, best);

            want[5] = if x & 1 == 1 { x.wrapping_mul(2).wrapping_add(1) } else { x };
            (want[6], want[7]) = if t < 32 { (1i32.wrapping_add(x), 2) } else { (1, 2i32.wrapping_sub(x)) };

            let (mut current, mut steps) = ((x & 63) + 1, 0);
            while current != 1 {
                current = if current & 1 == 0 { current / 2 } else { current * 3 + 1 };
                steps += 1;
            }
            want[8] = steps;

            let mut frag: [i32; 4] = std::array::from_fn(|k| x.wrapping_add(k as i32));
            for i in 0..n {
                for (k, f) in frag.iter_mut().enumerate() {
                    *f = f.wrapping_mul(2).wrapping_add(i * (k as i32 + 1));
                }
            }
            want[9..13].copy_from_slice(&frag);

            let (mut s, mut m, mut c) = (0, -1, 0);
            for i in 0..(x & 3) + n {
                let v = (x ^ i) & 15;
                (s, m, c) = (s + v, m.max(v), c + 1);
            }
            (want[13], want[14], want[15]) = (s, m, c);

            let (mut product, mut total) = (1i32, 0i32);
            for i in 0..n {
                total = total.wrapping_add(product);
                product = product.wrapping_mul(2).wrapping_add(i);
            }
            (want[16], want[17]) = (product, total);

            let mut outer = 0i32;
            for i in 0..n {
                let mut inner = i.wrapping_add(x);
                for j in 0..3 {
                    inner = inner.wrapping_mul(2).wrapping_add(j);
                }
                outer = outer.wrapping_add(inner);
            }
            want[18] = outer;

            let half = x as f32 * 0.5;
            let (mut p, mut q) = (0, 1.0f32);
            for i in 0..n {
                (p, q) = (p + i, q * 0.5 + half);
            }
            (want[19], want[20]) = (p, q.to_bits() as i32);

            let mut deep = 0i32;
            if t >= 8 {
                for i in (0..n).filter(|i| i & 1 == 0) {
                    deep = deep.wrapping_add(i).wrapping_add(x);
                }
            }
            want[21] = deep;

            let triple = x.wrapping_mul(3);
            want[22] = triple.wrapping_mul(n).wrapping_sub(triple);

            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t} (x = {x}, n = {n})");
        }
    }
    Case { name: "vars", kernel, cfg: launch(1), small_only: false, check }
}

/// Loops with CuTe DSL's unroll hints (`cutlass.range(..., unroll=K,
/// unroll_full=True, at_least_once=True)`) compute what they compute without
/// them, including the remainder iterations of a runtime trip count.
pub fn loop_hints() -> Case {
    const COLS: usize = 6;
    let kernel = Kernel::builder("loop_hints").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = xs.load(t);
        // Trip counts 0 to 12, so a loop unrolled 4 times needs its remainder.
        let n = t % 13;
        let run = |opts: LoopOptions, lower: i32, upper: Int32, step: i32| {
            let acc = Var::new(x);
            for_range_with(lower, upper, step, opts, |i| acc.set(acc.get() * 3 + (i ^ x)));
            acc.get()
        };
        out.store((t, 0), run(LoopOptions::unroll(4), 0, n, 1));
        out.store((t, 1), run(LoopOptions::unroll(1), 0, n, 1));
        out.store((t, 2), run(LoopOptions::unroll(0), 0, n, 1));
        let once = LoopOptions { unroll: Some(2), at_least_once: true, ..LoopOptions::default() };
        out.store((t, 3), run(once, 0, n + 1, 1));
        out.store((t, 4), run(LoopOptions::unroll(4), 1, n * 2, 3));
        // A static trip count, unrolled completely.
        let full = for_range_carry_with(0, 64, 1, LoopOptions::unroll_full(), x, |i, acc| acc * 3 + (i ^ x));
        out.store((t, 5), full);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = xs[t];
            let n = (t % 13) as i32;
            let run =
                |range: &mut dyn Iterator<Item = i32>| range.fold(x, |acc, i| acc.wrapping_mul(3).wrapping_add(i ^ x));
            let want = [
                run(&mut (0..n)),
                run(&mut (0..n)),
                run(&mut (0..n)),
                run(&mut (0..n + 1)),
                run(&mut (1..n * 2).step_by(3)),
                run(&mut (0..64)),
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t} (x = {x})");
        }
    }
    Case { name: "loop_hints", kernel, cfg: launch(1), small_only: false, check }
}

/// A generic struct given `Carry` by `carry_struct!`'s second form, with the
/// carried fields listed out of declaration order.
#[derive(Clone)]
pub struct Window<T: Numeric> {
    acc: Tensor<T>,
    steps: Int32,
    /// A build-time constant: rebuilding keeps it.
    scale: i32,
}
carry_struct!([T: Numeric] Window<T> { steps, acc });

/// A tuple struct with a build-time constant field.
#[derive(Clone)]
pub struct Span(Int32, Int32, usize);
carry_struct!(Span { 1, 0 });

/// Structs of `carry_struct!`'s second form held in a `Var`, carried as
/// explicit loop state, yielded from branches and carried by `while_carry`.
pub fn carry_structs() -> Case {
    const COLS: usize = 14;
    let kernel = Kernel::builder("carry_structs").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = xs.load(t) & 255;
        let n = t % 5;
        let col = |j: usize, v: Int32| out.store((t, j), v);

        // In a Var, through a loop and a branch.
        let acc = make_fragment::<i32>(3);
        for k in 0..3 {
            acc.store(k, x + k);
        }
        let window = Var::new(Window { acc, steps: Int32::from(0), scale: 3 });
        for_range(0, n, 1, |i| {
            window.update(|w| {
                for k in 0..3 {
                    w.acc.store(k, w.acc.load(k) * w.scale + i * (k + 1));
                }
                Window { steps: w.steps + 1, ..w }
            })
        });
        if_((x & 1).eq(1), || window.update(|w| Window { steps: w.steps * 10 + w.scale, ..w }));
        let w = window.get();
        for k in 0..3 {
            col(k, w.acc.load(k));
        }
        col(3, w.steps);
        col(4, Int32::from(w.scale));

        // As explicit loop state, with f32 registers (halving is exact).
        let acc = make_fragment::<f32>(2);
        acc.store(0usize, x.to::<f32>());
        acc.store(1usize, x.to::<f32>() * 2.0);
        let init = Window { acc, steps: Int32::from(1), scale: 2 };
        let w = for_range_carry(0, n, 1, init, |i, w| {
            for k in 0..2 {
                w.acc.store(k, w.acc.load(k) * 0.5 + i.to::<f32>());
            }
            Window { steps: w.steps * w.scale + i, ..w }
        });
        col(5, w.acc.load(0usize).bitcast::<i32>());
        col(6, w.acc.load(1usize).bitcast::<i32>());
        col(7, w.steps);

        // A tuple struct in a Var: the constant field feeds the arithmetic.
        let span = Var::new(Span(x, x * 2, 7));
        for_range(0, n, 1, |i| span.update(|s| Span(s.0 + s.1, s.1 - i * s.2 as i32, s.2)));
        let s = span.get();
        col(8, s.0);
        col(9, s.1);

        // Yielded from a branch, and carried by a while loop.
        let s = if_else_yield(t.lt(T / 2), || Span(x, x + 1, 5), || Span(x * 3, x - 1, 5));
        col(10, s.0 * s.2 as i32);
        col(11, s.1);
        let s = while_carry(Span(Int32::from(0), x & 15, 3), |s| s.1.gt(0), |s| Span(s.0 + s.2 as i32, s.1 - 1, s.2));
        col(12, s.0);
        col(13, s.1);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = xs[t] & 255;
            let n = (t % 5) as i32;
            let mut want = [0i32; COLS];

            let (mut acc, mut steps) = ([x, x + 1, x + 2], 0);
            for i in 0..n {
                for (k, a) in acc.iter_mut().enumerate() {
                    *a = a.wrapping_mul(3).wrapping_add(i * (k as i32 + 1));
                }
                steps += 1;
            }
            if x & 1 == 1 {
                steps = steps * 10 + 3;
            }
            want[..3].copy_from_slice(&acc);
            (want[3], want[4]) = (steps, 3);

            let (mut acc, mut steps) = ([x as f32, x as f32 * 2.0], 1);
            for i in 0..n {
                for a in &mut acc {
                    *a = *a * 0.5 + i as f32;
                }
                steps = steps * 2 + i;
            }
            (want[5], want[6], want[7]) = (acc[0].to_bits() as i32, acc[1].to_bits() as i32, steps);

            let (mut a, mut b) = (x, x * 2);
            for i in 0..n {
                (a, b) = (a + b, b - i * 7);
            }
            (want[8], want[9]) = (a, b);

            let (a, b) = if t < T / 2 { (x, x + 1) } else { (x * 3, x - 1) };
            (want[10], want[11]) = (a * 5, b);
            (want[12], want[13]) = ((x & 15) * 3, 0);

            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t} (x = {x})");
        }
    }
    Case { name: "carry_structs", kernel, cfg: launch(1), small_only: false, check }
}

/// Offsets from branches and loops keep what is known about them: each is a
/// multiple of 4 elements, so a 128-bit copy may use it (the simulator
/// checks the alignment), and a Var both branches set to the same constant
/// stays a constant that indexes registers.
pub fn merged_facts() -> Case {
    const COLS: usize = 13;
    let kernel = Kernel::builder("merged_facts").block_dim([T as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(1024));
        let out = param_tensor::<f32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        let columns = x.compose(Layout::new((4, 1020), (1, 1)));
        // `t * k + c`, built as an index expression so that its divisor is
        // known (plain `Int32` arithmetic records no facts).
        let index = |k: usize, c: usize| (IndexExpr::from(t) * k + c).into_value();
        let copy_at = |first: usize, offset: Int32| {
            let frag = make_fragment::<f32>(4);
            copy(&atom, &columns.slice((.., offset)), &frag);
            for k in 0..4 {
                out.store((t, first + k), frag.load(k));
            }
        };

        // A Var set to a different multiple of 4 in each branch.
        let offset = Var::new(Int32::from(0));
        if_else(t.lt(T / 2), || offset.set(index(8, 0)), || offset.set(index(4, 4)));
        copy_at(0, offset.get());

        // An explicit result, one side the constant 0.
        let offset = if_else_yield((t % 3).eq(0), || index(12, 0), || Int32::from(0));
        copy_at(4, offset);

        // A loop result: the initial value or what the body yields.
        let offset = for_range_carry(0, 8, 4, index(8, 0), |i, _| {
            (IndexExpr::from(t) * 4usize + IndexExpr::from(i) * 4usize).into_value()
        });
        copy_at(8, offset);

        // Both branches set the same constant, so the Var is still that
        // constant and can index registers.
        let slot = Var::new(Int32::from(1));
        if_else(t.lt(5), || slot.set(3), || slot.set(3));
        let frag = make_fragment::<f32>(4);
        copy(&atom, &columns.slice((.., index(4, 0))), &frag);
        out.store((t, 12), frag.load(slot.get()));
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..T {
            let offsets = [if t < T / 2 { t * 8 } else { t * 4 + 4 }, if t % 3 == 0 { t * 12 } else { 0 }, t * 4 + 16];
            let mut want: Vec<f32> = offsets.iter().flat_map(|&o| x[o..o + 4].to_vec()).collect();
            want.push(x[t * 4 + 3]);
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "merged_facts", kernel, cfg: launch(1), small_only: false, check }
}

/// The tile of [`dynamic_tiles`]: `TILE_M` rows by `TILE_N` columns, copied
/// by `TILE_THREADS` threads, four 32-bit values each.
pub const TILE_M: usize = 8;
pub const TILE_N: usize = 8;
pub const TILE_THREADS: usize = 16;

/// Global tensors whose extents are launch-time values (`param_extent`):
/// an `m x n` row-major `x`, a tensor parameter, and `y`, a tensor made on a
/// pointer, with `n` a multiple of 4 so that rows stay 16-byte aligned for
/// the 128-bit copies. Each block takes a band of `TILE_M` rows with
/// `local_tile`, keeping the rest mode of column tiles, `ceil_div(n,
/// TILE_N)`, which a `for_range` walks (its trip count is the rest mode's
/// size). A tiled copy partitions the band of `x` once, rest mode and all,
/// and each tile of `y` inside the loop. A bounds predicate written by hand
/// (the tile coordinates of an identity tensor, offset by the tile's
/// position, against the runtime `m` and `n`) guards the loads and the
/// stores of the partial tiles at the bottom and right edges. Each element
/// becomes `3 * x + 1000 * row + column`, so that an element read from or
/// written to the wrong place shows.
pub fn dynamic_tiles() -> Case {
    let kernel = Kernel::builder("dynamic_tiles").block_dim([TILE_THREADS as u32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let n = param_extent("n", 4);
        let x = param_tensor::<i32>("x", Layout::row_major((m, n)));
        let y = make_tensor(&param_ptr::<i32>("y"), Layout::row_major((m, n)));
        let (bid, tid) = (block_idx().x, thread_idx().x);
        // This block's rows: (TILE_M, TILE_N, ceil_div(n, TILE_N)).
        let band_x = local_tile(&x, (TILE_M, TILE_N), (bid, ..));
        let band_y = local_tile(&y, (TILE_M, TILE_N), (bid, ..));
        let atom = make_copy_atom::<i32>(CopyOp::Universal { bits: 128 });
        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 2)), &Layout::row_major((1, 4)));
        let thr = tiled.get_slice(tid);
        // The band's partition keeps the rest mode: (CPY, 1, 1, tiles).
        let t_x = thr.partition_s(&band_x);
        // Each of this thread's values' coordinates in a tile.
        let coords = thr.partition_s(&make_identity_tensor((TILE_M, TILE_N)));
        let leaf = |c: &Coord, i: usize| Int32::from(c.mode(i).as_leaf().expect("a coordinate per mode").clone());
        for_range(0, size(&band_x, &[2]), 1, |j| {
            let t_y = thr.partition_d(&band_y.slice((.., .., j)));
            let (row0, col0) = (bid * TILE_M as i32, j * TILE_N as i32);
            let pred = make_predicate(&coords, |c| (leaf(c, 0) + row0).lt(m) & (leaf(c, 1) + col0).lt(n));
            let frag = make_fragment_like::<i32, _>(&t_y);
            copy_if(&tiled, &pred, &t_x.slice((.., .., .., j)), &frag);
            for i in 0..frag.size() {
                let c = coords.coord(i);
                frag.store(i, frag.load(i) * 3 + (leaf(&c, 0) + row0) * 1000 + leaf(&c, 1) + col0);
            }
            copy_if(&tiled, &pred, &frag, &t_y);
        });
    });
    fn check(args: &[KernelArg]) {
        let (KernelArg::ScalarI32(m), KernelArg::ScalarI32(n)) = (&args[0], &args[1]) else {
            panic!("dynamic_tiles takes the extents m and n first")
        };
        let (m, n) = (*m as usize, *n as usize);
        let (x, y) = (args[2].to_vec::<i32>(), args[3].to_vec::<i32>());
        for r in 0..m {
            for c in 0..n {
                let i = r * n + c;
                assert_eq!(y[i], dynamic_tiles_value(x[i], r, c), "y[{r}, {c}] of the {m} x {n} tensors");
            }
        }
    }
    // The extents `inputs_with_len` gives: 13 rows (two bands, the second
    // partial) and 16 columns (two tiles).
    let cfg = LaunchConfig::new([13usize.div_ceil(TILE_M) as u32, 1, 1], [TILE_THREADS as u32, 1, 1]);
    Case { name: "dynamic_tiles", kernel, cfg, small_only: false, check }
}

/// What [`dynamic_tiles`] makes of `x[row, col]`.
pub fn dynamic_tiles_value(x: i32, row: usize, col: usize) -> i32 {
    x.wrapping_mul(3).wrapping_add((row * 1000 + col) as i32)
}

/// The launch and arguments of [`dynamic_tiles`] for `m x n` tensors in
/// buffers `pad` elements longer than `m * n`: `x` from `seed`, `y` filled
/// with -7 (which the kernel must leave alone outside the tensor).
pub fn dynamic_tiles_problem(m: usize, n: usize, pad: usize, seed: u64) -> (LaunchConfig, Vec<KernelArg>) {
    let mut state = seed;
    let x: Vec<i32> = (0..m * n + pad)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as i32
        })
        .collect();
    let args = vec![
        KernelArg::ScalarI32(m as i32),
        KernelArg::ScalarI32(n as i32),
        KernelArg::buffer(&x),
        KernelArg::buffer(&vec![-7i32; m * n + pad]),
    ];
    (LaunchConfig::new([m.div_ceil(TILE_M) as u32, 1, 1], [TILE_THREADS as u32, 1, 1]), args)
}

/// The whole `y` buffer [`dynamic_tiles`] must leave for the arguments of
/// [`dynamic_tiles_problem`]: each element of the `m x n` tensor computed,
/// the padding untouched.
pub fn dynamic_tiles_expected(m: usize, n: usize, args: &[KernelArg]) -> Vec<i32> {
    let x = args[2].to_vec::<i32>();
    let mut y = vec![-7i32; x.len()];
    for r in 0..m {
        for c in 0..n {
            y[r * n + c] = dynamic_tiles_value(x[r * n + c], r, c);
        }
    }
    y
}

/// Threads of [`assume_vector_copy`], each copying 4 elements.
const ASSUME_THREADS: usize = 32;

/// A 128-bit vector copy at a runtime offset that only an assumption proves
/// aligned: `ld = n - n % 4` is a multiple of 4, but the arithmetic does not
/// show it (a difference of an unknown value and a remainder), so without
/// `assume` the copy is refused.
pub fn assume_vector_copy() -> Case {
    let kernel = Kernel::builder("assume_vector_copy").block_dim([ASSUME_THREADS as u32, 1, 1]).build(|| {
        let n = param_scalar::<i32>("n");
        let ld = assume(n - n % 4, Some(4), false);
        // Column 1 of (128,2):(1,ld) starts at element ld.
        let layout = Layout::new((ASSUME_THREADS * 4, 2), (1, ld));
        let x = make_tensor(&param_ptr::<f32>("x"), layout.clone()).slice((.., 1));
        let y = make_tensor(&param_ptr::<f32>("y"), layout).slice((.., 1));
        let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(ASSUME_THREADS), &Layout::col_major(4));
        let thr = tiled.get_slice(thread_idx().x);
        let t_x = thr.partition_s(&x);
        let frag = make_fragment_like::<f32, _>(&t_x);
        copy(&tiled, &t_x, &frag);
        for i in 0..frag.size() {
            frag.store(i, frag.load(i) * 2.0);
        }
        copy(&tiled, &frag, &thr.partition_d(&y));
    });
    fn check(args: &[KernelArg]) {
        let KernelArg::ScalarI32(n) = args[0] else { panic!("assume_vector_copy takes n first") };
        let ld = (n - n % 4) as usize;
        let (x, y) = (args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        for i in ld..ld + ASSUME_THREADS * 4 {
            assert_eq!(y[i], x[i] * 2.0, "y[{i}]");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [ASSUME_THREADS as u32, 1, 1]);
    Case { name: "assume_vector_copy", kernel, cfg, small_only: false, check }
}

/// Tensors with a 64-bit runtime stride on small buffers: a (16,4):(s,1)
/// view of `x` and `z` with `s` an `i64` parameter, so their offsets are
/// computed in 64 bits, while the thread's coordinates stay 32-bit.
pub fn int64_strides() -> Case {
    let kernel = Kernel::builder("int64_strides").block_dim([T as u32, 1, 1]).build(|| {
        let s = param_extent_i64("s", 1);
        let layout = Layout::new((16, 4), (IndexExpr::from(s), 1));
        let x = make_tensor(&param_ptr::<i32>("x"), layout.clone());
        let z = make_tensor(&param_ptr::<i32>("z"), layout);
        let y = param_tensor::<i32>("y", Layout::col_major(T));
        let t = thread_idx().x;
        let (r, c) = (t % 16, t / 16);
        let v = x.load((r, c));
        y.store(t, v);
        z.store((r, c), v + 1);
    });
    fn check(args: &[KernelArg]) {
        let KernelArg::Scalar { bits, .. } = args[0] else { panic!("int64_strides takes the stride first") };
        let (x, z, y) = (args[1].to_vec::<i32>(), args[2].to_vec::<i32>(), args[3].to_vec::<i32>());
        for (t, &yt) in y.iter().enumerate() {
            let i = (t % 16) * bits as usize + t / 16;
            assert_eq!(yt, x[i], "y[{t}]");
            assert_eq!(z[i], x[i].wrapping_add(1), "z[{i}]");
        }
    }
    Case { name: "int64_strides", kernel, cfg: launch(1), small_only: false, check }
}

/// The columns of [`fast_divmod`].
const FAST_DIVMOD_COLUMNS: usize = 5;

/// The divisor and dividend [`fast_divmod`]'s thread `t` uses for inputs
/// `x` and `d`: a dividend below 2^31 (2^31 - 1 for thread 5), and a
/// positive divisor, below 1001 for every fourth thread.
pub fn fast_divmod_operands(t: usize, x: i32, d: i32) -> (i32, i32) {
    let n = if t == 5 { i32::MAX } else { x & 0x7fff_ffff };
    let raw = d & 0x7fff_ffff;
    (n, if t.is_multiple_of(4) { raw % 1000 + 1 } else { raw.max(1) })
}

/// `FastDivmod` built in the kernel (one 64-bit unsigned division) from a
/// runtime divisor per thread, against `/` and `%`: its `divmod`, the `/`
/// and `%` operators and the divisor read back.
pub fn fast_divmod() -> Case {
    let kernel = Kernel::builder("fast_divmod").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ds = param_tensor::<i32>("ds", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, FAST_DIVMOD_COLUMNS)));
        let t = thread_idx().x;
        let n = select(t.eq(5), i32::MAX, xs.load(t) & 0x7fff_ffff);
        let raw = ds.load(t) & 0x7fff_ffff;
        let d = select((t % 4).eq(0), raw % 1000 + 1, raw.max(1));
        let fd = FastDivmod::new(d);
        let (q, r) = fd.divmod(n);
        for (j, v) in [q, r, n / fd, n % fd, fd.divisor].into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ds, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        for t in 0..T {
            let (n, d) = fast_divmod_operands(t, xs[t], ds[t]);
            let row = &out[t * FAST_DIVMOD_COLUMNS..(t + 1) * FAST_DIVMOD_COLUMNS];
            assert_eq!(row, [n / d, n % d, n / d, n % d, d], "thread {t}: {n} divmod {d}");
        }
    }
    Case { name: "fast_divmod", kernel, cfg: launch(1), small_only: false, check }
}

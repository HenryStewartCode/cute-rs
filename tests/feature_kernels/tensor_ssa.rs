// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TensorSsa`: whole-tensor loads and stores, arithmetic with CuTe DSL's
//! promotion and broadcasting, indexing, reshaping and carrying, held to
//! plain Rust.

use cute::prelude::*;

use super::Case;

/// Threads of these kernels: one warp.
const THREADS: usize = 32;

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1])
}

/// Whether two floats are the same value: equal bits, or both NaN.
fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Each thread's row-major 4 x 4 tile loaded whole, `2x + 1` stored whole
/// to a column-major tile; a fragment round trip; and rows through
/// swizzled shared memory (written by one thread, read by the next).
pub fn ssa_load_store() -> Case {
    let kernel = Kernel::builder("ssa_load_store").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((THREADS * 4, 4)));
        let y = param_tensor::<f32>("y", Layout::col_major((THREADS * 4, 4)));
        let w = param_tensor::<f32>("w", Layout::row_major((THREADS, 32)));
        let z = param_tensor::<f32>("z", Layout::row_major((THREADS, 32)));
        let t = thread_idx().x;
        let v = local_tile(&x, (4, 4), (t, 0)).load_ssa();
        // Through a fragment shaped like the value, and back.
        let frag = make_fragment_like::<f32, f32>(&v);
        frag.store_ssa(&(v * 2.0 + 1.0));
        local_tile(&y, (4, 4), (t, 0)).store_ssa(&frag.load_ssa());
        let shape = (THREADS, 32);
        let smem = make_smem_tensor::<f32>(make_composed_layout(make_swizzle(3, 2, 3), 0, Layout::row_major(shape)));
        smem.slice((t, ..)).store_ssa(&(w.slice((t, ..)).load_ssa() * 3.0));
        sync_threads();
        let next = (t + 1) % THREADS as i32;
        z.slice((t, ..)).store_ssa(&smem.slice((next, ..)).load_ssa());
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        let rows = THREADS * 4;
        for r in 0..rows {
            for c in 0..4 {
                assert_eq!(y[c * rows + r], x[r * 4 + c] * 2.0 + 1.0, "ssa_load_store: y({r}, {c})");
            }
        }
        let (w, z) = (args[2].to_vec::<f32>(), args[3].to_vec::<f32>());
        for t in 0..THREADS {
            for c in 0..32 {
                assert_eq!(z[t * 32 + c], w[(t + 1) % THREADS * 32 + c] * 3.0, "ssa_load_store: z({t}, {c})");
            }
        }
    }
    Case { name: "ssa_load_store", kernel, cfg: cfg(), small_only: false, check }
}

/// Columns of the integer arithmetic kernels.
const INT_OPS: usize = 9;

macro_rules! ssa_int_case {
    ($name:ident, $t:ty) => {
        /// Tensor/tensor, tensor/value, tensor/literal and literal/tensor
        /// arithmetic on four elements per thread.
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([THREADS as u32, 1, 1]).build(|| {
                let n = THREADS * 4;
                let xs = param_tensor::<$t>("xs", Layout::col_major(n));
                let ys = param_tensor::<$t>("ys", Layout::col_major(n));
                let out = param_tensor::<$t>("out", Layout::col_major((n, INT_OPS)));
                let t = thread_idx().x;
                let x = local_tile(&xs, 4, t).load_ssa();
                let y = local_tile(&ys, 4, t).load_ssa();
                let v = ys.load(t * 4);
                let d = (y.clone() & (7 as $t)) + (1 as $t);
                let results: [TensorSsa<$t>; INT_OPS] = [
                    x.clone() + y.clone(),
                    x.clone() - v,
                    x.clone() * (3 as $t),
                    (5 as $t) - x.clone(),
                    x.clone() / d.clone(),
                    x.clone() % d.clone(),
                    v * x.clone(),
                    -x.clone(),
                    (&x * &y).abs(),
                ];
                for (j, r) in results.iter().enumerate() {
                    local_tile(&out.slice((.., j)), 4, t).store_ssa(r);
                }
            });
            fn check(args: &[KernelArg]) {
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                let n = xs.len();
                for i in 0..n {
                    let (x, y, v) = (xs[i], ys[i], ys[i / 4 * 4]);
                    let d = (y & 7) + 1;
                    let p = x.wrapping_mul(y);
                    let abs = if p < (0 as $t) { (0 as $t).wrapping_sub(p) } else { p };
                    let want: [$t; INT_OPS] = [
                        x.wrapping_add(y),
                        x.wrapping_sub(v),
                        x.wrapping_mul(3),
                        (5 as $t).wrapping_sub(x),
                        x.wrapping_div(d),
                        x.wrapping_rem(d),
                        v.wrapping_mul(x),
                        (0 as $t).wrapping_sub(x),
                        abs,
                    ];
                    for (j, w) in want.into_iter().enumerate() {
                        assert_eq!(out[j * n + i], w, "{} op {j}, x = {x}, y = {y}", stringify!($name));
                    }
                }
            }
            Case { name: stringify!($name), kernel, cfg: cfg(), small_only: false, check }
        }
    };
}

ssa_int_case!(ssa_arith_i8, i8);
ssa_int_case!(ssa_arith_i16, i16);
ssa_int_case!(ssa_arith_i32, i32);
ssa_int_case!(ssa_arith_i64, i64);
ssa_int_case!(ssa_arith_u8, u8);
ssa_int_case!(ssa_arith_u16, u16);
ssa_int_case!(ssa_arith_u32, u32);
ssa_int_case!(ssa_arith_u64, u64);

/// Columns of the float arithmetic kernels.
const FLOAT_OPS: usize = 9;

macro_rules! ssa_float_case {
    ($name:ident, $t:ty, $host:ty, $widen:expr, $narrow:expr, $same:expr) => {
        /// The float counterpart of the integer kernels, `%` included
        /// (`math::rem`).
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([THREADS as u32, 1, 1]).build(|| {
                let narrow: fn($host) -> $t = $narrow;
                let n = THREADS * 4;
                let xs = param_tensor::<$t>("xs", Layout::col_major(n));
                let ys = param_tensor::<$t>("ys", Layout::col_major(n));
                let out = param_tensor::<$t>("out", Layout::col_major((n, FLOAT_OPS)));
                let t = thread_idx().x;
                let x = local_tile(&xs, 4, t).load_ssa();
                let y = local_tile(&ys, 4, t).load_ssa();
                let v = ys.load(t * 4);
                let results: [TensorSsa<$t>; FLOAT_OPS] = [
                    x.clone() + y.clone(),
                    x.clone() - v,
                    x.clone() * narrow(3.0),
                    narrow(5.0) - x.clone(),
                    x.clone() / y.clone(),
                    x.clone() % y.clone(),
                    v * x.clone(),
                    -x.clone(),
                    (&x * &y).abs(),
                ];
                for (j, r) in results.iter().enumerate() {
                    local_tile(&out.slice((.., j)), 4, t).store_ssa(r);
                }
            });
            fn check(args: &[KernelArg]) {
                let widen: fn($t) -> $host = $widen;
                let narrow: fn($host) -> $t = $narrow;
                let same: fn($host, $host) -> bool = $same;
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                let n = xs.len();
                for i in 0..n {
                    let (x, y, v) = (widen(xs[i]), widen(ys[i]), widen(ys[i / 4 * 4]));
                    let want: [$host; FLOAT_OPS] =
                        [x + y, x - v, x * 3.0, 5.0 - x, x / y, x % y, v * x, -x, (x * y).abs()];
                    for (j, w) in want.into_iter().enumerate() {
                        let got = widen(out[j * n + i]);
                        // Through the narrow type and back: rounded once.
                        assert!(same(got, widen(narrow(w))), "{} op {j}, x = {x}, y = {y}: {got}", stringify!($name));
                    }
                    // -x flips the sign bit (of +0.0 too).
                    assert_eq!(out[7 * n + i].raw_bits() ^ xs[i].raw_bits(), 1 << (<$t as Numeric>::WIDTH - 1));
                }
            }
            Case { name: stringify!($name), kernel, cfg: cfg(), small_only: false, check }
        }
    };
}

ssa_float_case!(ssa_arith_f32, f32, f32, |x| x, |x| x, same_f32);
ssa_float_case!(ssa_arith_f64, f64, f64, |x| x, |x| x, same_f64);
ssa_float_case!(ssa_arith_f16, F16, f32, |x: F16| x.to_f32(), F16::from_f32, same_f32);
ssa_float_case!(ssa_arith_bf16, BF16, f32, |x: BF16| x.to_f32(), BF16::from_f32, same_f32);

/// CuTe DSL's promotion: a Float16 tensor times a Float32 value is Float32,
/// Int8 plus Int32 is Int32, Int32 with a Uint32 value is Uint32, Uint8
/// with Int16 is Int16 and Int64 with Uint32 is Int64.
pub fn ssa_promotion() -> Case {
    let kernel = Kernel::builder("ssa_promotion").block_dim([THREADS as u32, 1, 1]).build(|| {
        let h = param_tensor::<F16>("h", Layout::col_major(THREADS * 4));
        let i8s = param_tensor::<i8>("i8s", Layout::col_major(THREADS * 4));
        let i32s = param_tensor::<i32>("i32s", Layout::col_major(THREADS * 4));
        let f = param_tensor::<f32>("f", Layout::col_major(THREADS * 4));
        let a = param_tensor::<i32>("a", Layout::col_major(THREADS * 4));
        let u = param_tensor::<u32>("u", Layout::col_major(THREADS * 4));
        let s = param_tensor::<i16>("s", Layout::col_major(THREADS * 4));
        let l = param_tensor::<i64>("l", Layout::col_major(THREADS * 4));
        let t = thread_idx().x;
        fn tile<E>(g: &Tensor<E>, t: Int32) -> Tensor<E> {
            local_tile(g, 4, t)
        }
        let hv = tile(&h, t).load_ssa();
        let scaled: TensorSsa<f32> = hv * Value::constant(1.5f32);
        tile(&f, t).store_ssa(&scaled);
        let b = tile(&i8s, t).load_ssa();
        let c = tile(&i32s, t).load_ssa();
        let sum: TensorSsa<i32> = b.clone() + c.clone();
        tile(&a, t).store_ssa(&sum);
        let mixed: TensorSsa<u32> = c.clone() * Value::constant(3u32);
        tile(&u, t).store_ssa(&mixed);
        let bytes: TensorSsa<u8> = b.to::<u8>();
        let wide: TensorSsa<i16> = bytes - c.to::<i16>();
        tile(&s, t).store_ssa(&wide);
        let long: TensorSsa<i64> = c.to::<i64>() + Value::constant(7u32);
        tile(&l, t).store_ssa(&long);
    });
    fn check(args: &[KernelArg]) {
        let h = args[0].to_vec::<F16>();
        let (b, c) = (args[1].to_vec::<i8>(), args[2].to_vec::<i32>());
        let (f, a, u, s, l) = (
            args[3].to_vec::<f32>(),
            args[4].to_vec::<i32>(),
            args[5].to_vec::<u32>(),
            args[6].to_vec::<i16>(),
            args[7].to_vec::<i64>(),
        );
        for i in 0..h.len() {
            assert_eq!(f[i], h[i].to_f32() * 1.5, "ssa_promotion: f16 * f32 at {i}");
            assert_eq!(a[i], (b[i] as i32).wrapping_add(c[i]), "ssa_promotion: i8 + i32 at {i}");
            assert_eq!(u[i], (c[i] as u32).wrapping_mul(3), "ssa_promotion: i32 * u32 at {i}");
            assert_eq!(s[i], (b[i] as u8 as i16).wrapping_sub(c[i] as i16), "ssa_promotion: u8 - i16 at {i}");
            assert_eq!(l[i], c[i] as i64 + 7, "ssa_promotion: i64 + u32 at {i}");
        }
    }
    Case { name: "ssa_promotion", kernel, cfg: cfg(), small_only: false, check }
}

/// Columns of [`ssa_division`].
const DIV_OPS: usize = 6;

/// CuTe DSL's `//`, `%`, `/` and `**` on negative operands: Int32 floor
/// and truncating division, true division to Float32, integer powers,
/// Float32 floor division, `%` and `fpowi`.
pub fn ssa_division() -> Case {
    let kernel = Kernel::builder("ssa_division").block_dim([THREADS as u32, 1, 1]).build(|| {
        let n = THREADS * 4;
        let xs = param_tensor::<i32>("xs", Layout::col_major(n));
        let ys = param_tensor::<i32>("ys", Layout::col_major(n));
        let ints = param_tensor::<i32>("ints", Layout::col_major((n, DIV_OPS)));
        let floats = param_tensor::<f32>("floats", Layout::col_major((n, DIV_OPS)));
        let t = thread_idx().x;
        let x = local_tile(&xs, 4, t).load_ssa() % 1000;
        let y = local_tile(&ys, 4, t).load_ssa();
        // A divisor in [-8, 8] without 0.
        let mag = (y.clone() & 7) + 1;
        let d = where_(&(y.clone() & 8).ne(0), -mag.clone(), &mag);
        let e = y.clone() & 3;
        let fx = x.to::<f32>();
        let fd = d.to::<f32>();
        let int_results = [x.floor_div(&d), x.clone() / d.clone(), x.clone() % d.clone(), x.pow(&e, false)];
        for (j, r) in int_results.iter().enumerate() {
            local_tile(&ints.slice((.., j)), 4, t).store_ssa(r);
        }
        let float_results = [
            x.true_div(&d),
            fx.floor_div(&fd),
            fx.clone() % fd.clone(),
            fx.fpowi(&e),
            -(fx.clone() * 0.0),
            fx.pow(0.5, false),
        ];
        for (j, r) in float_results.iter().enumerate() {
            local_tile(&floats.slice((.., j)), 4, t).store_ssa(r);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, ints, floats) =
            (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>(), args[3].to_vec::<f32>());
        let n = xs.len();
        for i in 0..n {
            let (x, y) = (xs[i] % 1000, ys[i]);
            let mag = (y & 7) + 1;
            let d = if y & 8 != 0 { -mag } else { mag };
            let e = y & 3;
            let floor_q = (x as f64 / d as f64).floor() as i32;
            let int_want = [floor_q, x / d, x % d, x.wrapping_pow(e as u32)];
            for (j, w) in int_want.into_iter().enumerate() {
                assert_eq!(ints[j * n + i], w, "ssa_division: int op {j}, x = {x}, d = {d}");
            }
            let (fx, fd) = (x as f32, d as f32);
            let float_want = [x as f32 / d as f32, (fx / fd).floor(), fx % fd, (fx as f64).powi(e) as f32, -(fx * 0.0)];
            for (j, w) in float_want.into_iter().enumerate() {
                let got = floats[j * n + i];
                // fpowi is within an ulp.
                let close = j == 3 && (got.to_bits() as i64 - w.to_bits() as i64).abs() <= 1;
                assert!(same_f32(got, w) || close, "ssa_division: float op {j}, x = {x}, d = {d}: {got} != {w}");
            }
            // -(x * 0.0) is -0.0 for x >= 0 (an exact sign flip of +0.0).
            if x >= 0 {
                assert_eq!(floats[4 * n + i].to_bits(), 0x8000_0000);
            }
            let p = floats[5 * n + i];
            assert!(same_f32(p, fx.powf(0.5)) || (p - fx.powf(0.5)).abs() <= 1e-6 * p.abs(), "pow({fx}, 0.5) = {p}");
        }
    }
    Case { name: "ssa_division", kernel, cfg: cfg(), small_only: false, check }
}

/// Broadcasting: `(4,)` against `(4, 5)` (padded on the right to `(4, 1)`),
/// `(1, 4)` to `(3, 4)`, and `(2, 1)` against `(2, (2, 2))`; a reshape; a
/// static element and a static slice.
pub fn ssa_broadcast() -> Case {
    let kernel = Kernel::builder("ssa_broadcast").block_dim([THREADS as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major((4, THREADS)));
        let ys = param_tensor::<i32>("ys", Layout::col_major((20, THREADS)));
        let a = param_tensor::<i32>("a", Layout::col_major((20, THREADS)));
        let b = param_tensor::<i32>("b", Layout::col_major((12, THREADS)));
        let c = param_tensor::<i32>("c", Layout::col_major((8, THREADS)));
        let d = param_tensor::<i32>("d", Layout::col_major((4, THREADS)));
        let t = thread_idx().x;
        let x = xs.slice((.., t)).load_ssa();
        let y = ys.slice((.., t)).load_ssa().reshape((4, 5));
        let col = x.reshape((4,));
        a.slice((.., t)).store_ssa(&(col * 10 + y.clone()));
        let row = x.reshape((1, 4));
        b.slice((.., t)).store_ssa(&row.broadcast_to((3, 4)));
        let nested = TensorSsa::new((2, (2, 2)), y.values()[..8].to_vec());
        let two = TensorSsa::new((2, 1), vec![y.get(0), y.get((1, 0))]);
        c.slice((.., t)).store_ssa(&(nested - two));
        // A static slice of (4, (5, 4)) by (.., (.., 1)) is (4, 5).
        let big = y.reshape((4, (5, 1))).broadcast_to((4, (5, 4)));
        let kept = big.slice((.., (.., 1)));
        assert_eq!(kept.shape(), &IntTuple::from((4, 5)));
        d.slice((.., t)).store_ssa(&kept.slice((.., 2)));
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        let (a, b, c, d) =
            (args[2].to_vec::<i32>(), args[3].to_vec::<i32>(), args[4].to_vec::<i32>(), args[5].to_vec::<i32>());
        for t in 0..THREADS {
            let x = &xs[t * 4..t * 4 + 4];
            let y = &ys[t * 20..t * 20 + 20];
            for j in 0..5 {
                for i in 0..4 {
                    assert_eq!(a[t * 20 + j * 4 + i], x[i].wrapping_mul(10).wrapping_add(y[j * 4 + i]), "a({i}, {j})");
                }
            }
            for j in 0..4 {
                for i in 0..3 {
                    assert_eq!(b[t * 12 + j * 3 + i], x[j], "b({i}, {j})");
                }
            }
            for k in 0..8 {
                assert_eq!(c[t * 8 + k], y[k].wrapping_sub(y[k % 2]), "c[{k}]");
            }
            for i in 0..4 {
                assert_eq!(d[t * 4 + i], y[i + 8], "d[{i}]");
            }
        }
    }
    Case { name: "ssa_broadcast", kernel, cfg: cfg(), small_only: false, check }
}

/// `get` at a runtime coordinate: a select tree for 8 and for 32 elements
/// (the threshold), a stack array for 64; and a tensor value carried
/// through a loop.
pub fn ssa_indexing() -> Case {
    let kernel = Kernel::builder("ssa_indexing").block_dim([THREADS as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major((64, THREADS)));
        let idx = param_tensor::<i32>("idx", Layout::col_major(THREADS));
        let out = param_tensor::<i32>("out", Layout::col_major((4, THREADS)));
        let acc_out = param_tensor::<i32>("acc_out", Layout::col_major((8, THREADS)));
        let t = thread_idx().x;
        let x = xs.slice((.., t)).load_ssa();
        let i = idx.load(t) & 63;
        let eight = x.reshape((8, 8)).slice((.., 0));
        let row32 = x.reshape((32, 2)).slice((.., 1));
        let sixty_four = x.reshape((8, 8));
        out.store((0, t), eight.get(i & 7));
        out.store((1, t), row32.get(i & 31));
        out.store((2, t), sixty_four.get(i));
        out.store((3, t), sixty_four.get((i & 7, i >> 3)));
        let start = eight.clone();
        let acc = for_range_carry(0, idx.load(t) & 3, 1, start, |k, acc| acc + eight.clone() * k);
        acc_out.slice((.., t)).store_ssa(&acc);
    });
    fn check(args: &[KernelArg]) {
        let (xs, idx, out, acc) =
            (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>(), args[3].to_vec::<i32>());
        for t in 0..THREADS {
            let x = &xs[t * 64..t * 64 + 64];
            let i = (idx[t] & 63) as usize;
            assert_eq!(out[t * 4], x[i & 7], "get of 8");
            assert_eq!(out[t * 4 + 1], x[32 + (i & 31)], "get of 32");
            assert_eq!(out[t * 4 + 2], x[i], "get of 64");
            assert_eq!(out[t * 4 + 3], x[(i & 7) + 8 * (i >> 3)], "get (i, j) of 64");
            let n = idx[t] & 3;
            let factor = 1 + (0..n).sum::<i32>();
            for k in 0..8 {
                assert_eq!(acc[t * 8 + k], x[k].wrapping_mul(factor), "carried element {k}");
            }
        }
    }
    Case { name: "ssa_indexing", kernel, cfg: cfg(), small_only: false, check }
}

/// The TensorSsa kernels of every target.
pub fn tensor_ssa_cases() -> Vec<Case> {
    vec![
        ssa_load_store(),
        ssa_arith_i8(),
        ssa_arith_i16(),
        ssa_arith_i32(),
        ssa_arith_i64(),
        ssa_arith_u8(),
        ssa_arith_u16(),
        ssa_arith_u32(),
        ssa_arith_u64(),
        ssa_arith_f32(),
        ssa_arith_f64(),
        ssa_arith_f16(),
        ssa_arith_bf16(),
        ssa_promotion(),
        ssa_division(),
        ssa_broadcast(),
        ssa_indexing(),
        ssa_cmp_i8(),
        ssa_cmp_i16(),
        ssa_cmp_i32(),
        ssa_cmp_i64(),
        ssa_cmp_u8(),
        ssa_cmp_u16(),
        ssa_cmp_u32(),
        ssa_cmp_u64(),
        ssa_cmp_f32(),
        ssa_cmp_f64(),
        ssa_cmp_f16(),
        ssa_cmp_bf16(),
        ssa_logic(),
        ssa_softmax_rows(),
        ssa_reductions(),
        ssa_math(),
        double_rounding(),
        int_to_bf16_fast_paths(),
        ssa_bitcasts(),
        ssa_print(),
        bool_copies(),
    ]
    .into_iter()
    .chain(conversion_matrix())
    .collect()
}

/// Columns of the comparison kernels: lt, le, gt, ge, eq, ne.
const CMP_OPS: usize = 6;

macro_rules! ssa_cmp_case {
    ($name:ident, $t:ty, $widen:expr, $specials:expr) => {
        /// The six comparisons of random tensors, then of special values
        /// (NaN, signed zeros and infinities for floats) against each other.
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([THREADS as u32, 1, 1]).build(|| {
                let n = THREADS * 4;
                let xs = param_tensor::<$t>("xs", Layout::col_major(n));
                let ys = param_tensor::<$t>("ys", Layout::col_major(n));
                let out = param_tensor::<bool>("out", Layout::col_major((n, CMP_OPS)));
                let special = param_tensor::<bool>("special", Layout::col_major((64, CMP_OPS)));
                let t = thread_idx().x;
                let x = local_tile(&xs, 4, t).load_ssa();
                let y = local_tile(&ys, 4, t).load_ssa();
                let results = [x.lt(&y), x.le(&y), x.gt(&y), x.ge(&y), x.eq(&y), x.ne(&y)];
                for (j, r) in results.iter().enumerate() {
                    local_tile(&out.slice((.., j)), 4, t).store_ssa(r);
                }
                // Every pair of the eight special values, as (8, 1) against (1, 8).
                let specials: [$t; 8] = $specials;
                let s = TensorSsa::new(8, specials.iter().map(|&v| Value::constant(v)).collect());
                let (a, b) = (s.reshape((8, 1)), s.reshape((1, 8)));
                let results = [a.lt(&b), a.le(&b), a.gt(&b), a.ge(&b), a.eq(&b), a.ne(&b)];
                if_(t.eq(0), || {
                    for (j, r) in results.iter().enumerate() {
                        special.slice((.., j)).store_ssa(r);
                    }
                });
            });
            fn check(args: &[KernelArg]) {
                let widen = $widen;
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<bool>());
                let n = xs.len();
                let cmp = |x: $t, y: $t| {
                    let (x, y) = (widen(x), widen(y));
                    [x < y, x <= y, x > y, x >= y, x == y, x != y]
                };
                for i in 0..n {
                    for (j, w) in cmp(xs[i], ys[i]).into_iter().enumerate() {
                        assert_eq!(out[j * n + i], w, "{} op {j} at {i}", stringify!($name));
                    }
                }
                let specials: [$t; 8] = $specials;
                let special = args[3].to_vec::<bool>();
                for a in 0..8 {
                    for b in 0..8 {
                        for (j, w) in cmp(specials[a], specials[b]).into_iter().enumerate() {
                            assert_eq!(
                                special[j * 64 + b * 8 + a],
                                w,
                                "{} op {j} of specials {a}, {b}",
                                stringify!($name)
                            );
                        }
                    }
                }
            }
            Case { name: stringify!($name), kernel, cfg: cfg(), small_only: false, check }
        }
    };
}

macro_rules! int_specials {
    ($t:ty) => {
        [<$t>::MIN, <$t>::MAX, 0, 1, <$t>::MIN + 1, <$t>::MAX - 1, 7, 0]
    };
}

ssa_cmp_case!(ssa_cmp_i8, i8, |x: i8| x, int_specials!(i8));
ssa_cmp_case!(ssa_cmp_i16, i16, |x: i16| x, int_specials!(i16));
ssa_cmp_case!(ssa_cmp_i32, i32, |x: i32| x, int_specials!(i32));
ssa_cmp_case!(ssa_cmp_i64, i64, |x: i64| x, int_specials!(i64));
ssa_cmp_case!(ssa_cmp_u8, u8, |x: u8| x, int_specials!(u8));
ssa_cmp_case!(ssa_cmp_u16, u16, |x: u16| x, int_specials!(u16));
ssa_cmp_case!(ssa_cmp_u32, u32, |x: u32| x, int_specials!(u32));
ssa_cmp_case!(ssa_cmp_u64, u64, |x: u64| x, int_specials!(u64));
ssa_cmp_case!(
    ssa_cmp_f32,
    f32,
    |x: f32| x,
    [f32::NAN, -0.0, 0.0, 1.0, f32::NEG_INFINITY, f32::INFINITY, -1.5, f32::MIN_POSITIVE / 4.0]
);
ssa_cmp_case!(
    ssa_cmp_f64,
    f64,
    |x: f64| x,
    [f64::NAN, -0.0, 0.0, 1.0, f64::NEG_INFINITY, f64::INFINITY, -1.5, f64::MIN_POSITIVE / 4.0]
);
ssa_cmp_case!(
    ssa_cmp_f16,
    F16,
    |x: F16| x.to_f32(),
    [F16::NAN, F16(0x8000), F16::ZERO, F16::ONE, F16(0xfc00), F16::INFINITY, F16(0xbe00), F16(0x0001)]
);
ssa_cmp_case!(
    ssa_cmp_bf16,
    BF16,
    |x: BF16| x.to_f32(),
    [BF16(0x7fc0), BF16(0x8000), BF16(0), BF16(0x3f80), BF16(0xff80), BF16(0x7f80), BF16(0xbfc0), BF16(0x0001)]
);

/// Boolean masks combined with `& | ^ !` and compared with `eq`/`ne`;
/// shifts of Int32 and Uint32 tensors by tensors and by literals; `where_`
/// with tensor and scalar operands and broadcasting; `any_` and `all_`.
pub fn ssa_logic() -> Case {
    const OPS: usize = 8;
    let kernel = Kernel::builder("ssa_logic").block_dim([THREADS as u32, 1, 1]).build(|| {
        let n = THREADS * 4;
        let xs = param_tensor::<i32>("xs", Layout::col_major(n));
        let ys = param_tensor::<i32>("ys", Layout::col_major(n));
        let masks = param_tensor::<bool>("masks", Layout::col_major((n, 6)));
        let shifts = param_tensor::<i32>("shifts", Layout::col_major((n, OPS)));
        let picks = param_tensor::<i32>("picks", Layout::col_major((4, 8, THREADS)));
        let flags = param_tensor::<bool>("flags", Layout::col_major((8, THREADS)));
        let t = thread_idx().x;
        let x = local_tile(&xs, 4, t).load_ssa();
        let y = local_tile(&ys, 4, t).load_ssa();
        let (m1, m2) = (x.gt(&y), x.lt(0));
        let mask_results = [&m1 & &m2, &m1 | &m2, &m1 ^ &m2, !&m1, m1.eq(&m2), m1.ne(true)];
        for (j, r) in mask_results.iter().enumerate() {
            local_tile(&masks.slice((.., j)), 4, t).store_ssa(r);
        }
        let s = y.clone() & 31;
        let u = x.to::<u32>();
        let shift_results = [
            x.clone() << s.clone(),
            x.clone() >> s.clone(),
            (u.clone() >> s.to::<u32>()).to::<i32>(),
            x.clone() << 3,
            x.clone() >> 3,
            (u >> 3u32).to::<i32>(),
            x.clone() & y.clone() | !x.clone() ^ 0x55,
            Value::constant(1) - (x.clone() ^ y.clone()),
        ];
        for (j, r) in shift_results.iter().enumerate() {
            local_tile(&shifts.slice((.., j)), 4, t).store_ssa(r);
        }
        // where_: a (4, 8) condition, a (4,) tensor, a scalar.
        let y8 = TensorSsa::new((1, 8), (0..8).map(|j| y.values()[j % 4]).collect());
        let cond = x.reshape((4, 1)).lt(&y8);
        let picked = where_(&cond, &x, 0);
        picks.slice((.., .., t)).store_ssa(&picked);
        // any_ and all_ of floats: NaN counts as nonzero, -0.0 as zero.
        let k = |v: f32| Value::constant(v);
        let zeros = TensorSsa::new(2, vec![k(-0.0), k(0.0)]);
        let nan = TensorSsa::new(2, vec![k(f32::NAN), k(0.0)]);
        let ones = TensorSsa::new(2, vec![k(f32::NAN), k(-1.0)]);
        let fx = x.to::<f32>();
        let fl = [any_(&zeros), all_(&zeros), any_(&nan), all_(&nan), any_(&ones), all_(&ones), any_(&fx), all_(&m1)];
        for (j, f) in fl.into_iter().enumerate() {
            flags.store((j, t), f);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        let (masks, shifts, picks, flags) =
            (args[2].to_vec::<bool>(), args[3].to_vec::<i32>(), args[4].to_vec::<i32>(), args[5].to_vec::<bool>());
        let n = xs.len();
        for i in 0..n {
            let (x, y) = (xs[i], ys[i]);
            let (m1, m2) = (x > y, x < 0);
            let want = [m1 & m2, m1 | m2, m1 ^ m2, !m1, m1 == m2, !m1];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(masks[j * n + i], w, "ssa_logic: mask op {j}, x = {x}, y = {y}");
            }
            let s = (y & 31) as u32;
            let want = [
                x.wrapping_shl(s),
                x.wrapping_shr(s),
                (x as u32).wrapping_shr(s) as i32,
                x.wrapping_shl(3),
                x >> 3,
                ((x as u32) >> 3) as i32,
                x & y | !x ^ 0x55,
                1i32.wrapping_sub(x ^ y),
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(shifts[j * n + i], w, "ssa_logic: shift op {j}, x = {x}, y = {y}");
            }
        }
        for t in 0..THREADS {
            let x = &xs[t * 4..t * 4 + 4];
            let y = &ys[t * 4..t * 4 + 4];
            for j in 0..8 {
                for i in 0..4 {
                    let want = if x[i] < y[j % 4] { x[i] } else { 0 };
                    assert_eq!(picks[t * 32 + j * 4 + i], want, "ssa_logic: where_ ({i}, {j}) of thread {t}");
                }
            }
            let any_x = x.iter().any(|&v| v != 0);
            let all_m1 = (0..4).all(|i| x[i] > y[i]);
            let want = [false, false, true, false, true, true, any_x, all_m1];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(flags[t * 8 + j], w, "ssa_logic: flag {j} of thread {t}");
            }
        }
    }
    Case { name: "ssa_logic", kernel, cfg: cfg(), small_only: false, check }
}

/// A softmax's row statistics on each thread's (2, 8) Float32 tile, as
/// flash attention computes them: the row maximum (NaN-propagating), `2^(x
/// - max)` (precise `exp2`), and the row sum, each a sequential fold.
pub fn ssa_softmax_rows() -> Case {
    let kernel = Kernel::builder("ssa_softmax_rows").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((THREADS * 2, 8)));
        let e = param_tensor::<f32>("e", Layout::row_major((THREADS * 2, 8)));
        let stats = param_tensor::<f32>("stats", Layout::col_major((2, 2, THREADS)));
        let t = thread_idx().x;
        let tile = local_tile(&x, (2, 8), (t, 0)).load_ssa();
        let row_max = tile.reduce(ReductionOp::Max, f32::NEG_INFINITY, (.., 0));
        let p = (tile - row_max.clone()).exp2(false);
        let row_sum = p.reduce(ReductionOp::Add, 0.0, (.., 0));
        local_tile(&e, (2, 8), (t, 0)).store_ssa(&p);
        stats.slice((.., 0, t)).store_ssa(&row_max);
        stats.slice((.., 1, t)).store_ssa(&row_sum);
    });
    fn check(args: &[KernelArg]) {
        let (x, e, stats) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        for r in 0..THREADS * 2 {
            let row = &x[r * 8..r * 8 + 8];
            let max =
                row.iter().fold(f32::NEG_INFINITY, |a, &b| if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) });
            let (t, i) = (r / 2, r % 2);
            assert!(same_f32(stats[t * 4 + i], max), "ssa_softmax_rows: max of row {r}");
            let mut sum = 0.0f32;
            for c in 0..8 {
                let want = (row[c] - max).exp2();
                let got = e[r * 8 + c];
                assert!((got - want).abs() <= want * 2e-7, "ssa_softmax_rows: exp2 at ({r}, {c}): {got} vs {want}");
                sum += got;
            }
            assert_eq!(stats[t * 4 + 2 + i].to_bits(), sum.to_bits(), "ssa_softmax_rows: sum of row {r}");
        }
    }
    Case { name: "ssa_softmax_rows", kernel, cfg: cfg(), small_only: false, check }
}

/// Reductions: per-mode sums of (4, (2, 2)) Int32 tiles with `(.., 1)` and
/// `(.., (.., 1))`, a product, integer maximum and minimum, Float16 and
/// BFloat16 sums and extrema with a NaN propagated, and `apply_op` with
/// `flip`.
pub fn ssa_reductions() -> Case {
    let kernel = Kernel::builder("ssa_reductions").block_dim([THREADS as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major((16, THREADS)));
        let hs = param_tensor::<F16>("hs", Layout::col_major((8, THREADS)));
        let bs = param_tensor::<BF16>("bs", Layout::col_major((8, THREADS)));
        let ints = param_tensor::<i32>("ints", Layout::col_major((20, THREADS)));
        let halves = param_tensor::<F16>("halves", Layout::col_major((3, THREADS)));
        let brains = param_tensor::<BF16>("brains", Layout::col_major((11, THREADS)));
        let t = thread_idx().x;
        let x = xs.slice((.., t)).load_ssa().reshape((4, (2, 2)));
        let by_row = x.reduce(ReductionOp::Add, 0, (.., 1));
        let by_pair = x.reduce(ReductionOp::Add, 100, (.., (.., 1)));
        assert_eq!(by_row.shape(), &IntTuple::from((4,)));
        assert_eq!(by_pair.shape(), &IntTuple::from((4, 2)));
        let small = x.clone() % 5;
        let scalars = [
            small.reduce_all(ReductionOp::Mul, 1),
            x.reduce_all(ReductionOp::Max, i32::MIN),
            x.reduce_all(ReductionOp::Min, 0),
            x.to::<u32>().reduce_all(ReductionOp::Max, 0u32).to::<i32>(),
        ];
        let flipped = x.apply_op(7, true, |a, b| a - b).reshape(16).slice((0,)).reshape(1);
        let mut out = by_row.values().to_vec();
        out.extend(by_pair.values().iter().copied());
        out.extend(scalars);
        out.extend(flipped.values().iter().copied());
        out.extend(x.reshape(16).values()[..3].iter().map(|&v| Value::constant(7) - v));
        ints.slice((.., t)).store_ssa(&TensorSsa::new(20, out));
        let h = hs.slice((.., t)).load_ssa();
        let hv = [
            h.reduce_all(ReductionOp::Add, 0.0f32),
            h.reduce_all(ReductionOp::Max, F16::MIN),
            h.reduce_all(ReductionOp::Min, F16::MAX),
        ];
        halves.slice((.., t)).store_ssa(&TensorSsa::new(3, hv.to_vec()));
        // A NaN in the middle of a BFloat16 tensor: the extrema are NaN.
        let b = bs.slice((.., t)).load_ssa();
        let nan = Value::constant(BF16(0x7fc0));
        let mut with_nan = b.values().to_vec();
        with_nan[3] = nan;
        let bn = TensorSsa::new(8, with_nan);
        let maxima = b.max(&bn.reshape((8,)));
        let mut bv = vec![
            b.reduce_all(ReductionOp::Max, BF16(0xff80)),
            bn.reduce_all(ReductionOp::Max, BF16(0xff80)),
            bn.reduce_all(ReductionOp::Min, BF16(0x7f80)),
        ];
        bv.extend(maxima.values().iter().copied());
        brains.slice((.., t)).store_ssa(&TensorSsa::new(11, bv));
    });
    fn check(args: &[KernelArg]) {
        let (xs, hs, bs) = (args[0].to_vec::<i32>(), args[1].to_vec::<F16>(), args[2].to_vec::<BF16>());
        let (ints, halves, brains) = (args[3].to_vec::<i32>(), args[4].to_vec::<F16>(), args[5].to_vec::<BF16>());
        for t in 0..THREADS {
            let x = &xs[t * 16..t * 16 + 16];
            let got = &ints[t * 20..t * 20 + 20];
            // x(i, (j, k)) is x[i + 4 j + 8 k].
            for i in 0..4 {
                let want = (0..4).fold(0i32, |a, jk| a.wrapping_add(x[i + 4 * jk]));
                assert_eq!(got[i], want, "ssa_reductions: row {i} of thread {t}");
                for j in 0..2 {
                    let want = 100i32.wrapping_add(x[i + 4 * j]).wrapping_add(x[i + 4 * j + 8]);
                    assert_eq!(got[4 + j * 4 + i], want, "ssa_reductions: pair ({i}, {j}) of thread {t}");
                }
            }
            let prod = x.iter().fold(1i32, |a, &v| a.wrapping_mul(v % 5));
            let want = [
                prod,
                *x.iter().max().unwrap(),
                (*x.iter().min().unwrap()).min(0),
                x.iter().map(|&v| v as u32).max().unwrap() as i32,
                7 - x[0],
            ];
            assert_eq!(&got[12..17], &want, "ssa_reductions: scalars of thread {t}");
            assert_eq!(&got[17..20], &[7 - x[0], 7 - x[1], 7 - x[2]], "ssa_reductions: flipped of thread {t}");
            let h = &hs[t * 8..t * 8 + 8];
            let sum = h.iter().fold(F16::ZERO, |a, &v| F16::from_f32(a.to_f32() + v.to_f32()));
            let max = h.iter().fold(F16::MIN, |a, &v| if v.to_f32() > a.to_f32() { v } else { a });
            let min = h.iter().fold(F16::MAX, |a, &v| if v.to_f32() < a.to_f32() { v } else { a });
            let got = &halves[t * 3..t * 3 + 3];
            assert_eq!(got[0].0, sum.0, "ssa_reductions: f16 sum of thread {t}");
            assert_eq!((got[1].to_f32(), got[2].to_f32()), (max.to_f32(), min.to_f32()), "ssa_reductions: f16 extrema");
            let b = &bs[t * 8..t * 8 + 8];
            let got = &brains[t * 11..t * 11 + 11];
            let bmax = b.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v.to_f32()));
            assert_eq!(got[0].to_f32(), bmax, "ssa_reductions: bf16 max of thread {t}");
            assert!(got[1].to_f32().is_nan() && got[2].to_f32().is_nan(), "ssa_reductions: NaN extrema of thread {t}");
            for i in 0..8 {
                let m = got[3 + i].to_f32();
                if i == 3 {
                    assert!(m.is_nan());
                } else {
                    assert_eq!(m, b[i].to_f32());
                }
            }
        }
    }
    Case { name: "ssa_reductions", kernel, cfg: cfg(), small_only: false, check }
}

/// `cute.math` over tensors: precise `exp2`, `sqrt`, `atan2` with a
/// broadcast scalar, `isnan` and `fma`.
pub fn ssa_math() -> Case {
    let kernel = Kernel::builder("ssa_math").block_dim([THREADS as u32, 1, 1]).build(|| {
        let n = THREADS * 4;
        let xs = param_tensor::<f32>("xs", Layout::col_major(n));
        let out = param_tensor::<f32>("out", Layout::col_major((n, 4)));
        let nans = param_tensor::<bool>("nans", Layout::col_major(n));
        let t = thread_idx().x;
        let x = local_tile(&xs, 4, t).load_ssa();
        let r = x.sqrt(false);
        local_tile(&nans, 4, t).store_ssa(&r.isnan(false));
        let results = [x.exp2(false), r, x.atan2(1.0, false), x.fma(&x, 2.0, false)];
        for (j, v) in results.iter().enumerate() {
            local_tile(&out.slice((.., j)), 4, t).store_ssa(v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out, nans) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<bool>());
        let n = xs.len();
        for i in 0..n {
            let x = xs[i];
            assert_eq!(nans[i], x < 0.0, "ssa_math: isnan(sqrt({x}))");
            let close = |a: f32, b: f32| same_f32(a, b) || (a.to_bits() as i64 - b.to_bits() as i64).abs() <= 2;
            assert!(close(out[i], x.exp2()), "ssa_math: exp2({x})");
            assert!(same_f32(out[n + i], x.sqrt()), "ssa_math: sqrt({x})");
            assert!(close(out[2 * n + i], x.atan2(1.0)), "ssa_math: atan2({x}, 1)");
            assert!(same_f32(out[3 * n + i], x.mul_add(x, 2.0)), "ssa_math: fma({x})");
        }
    }
    Case { name: "ssa_math", kernel, cfg: cfg(), small_only: false, check }
}

/// A host value of any standard type, exactly.
#[derive(Clone, Copy, Debug)]
pub enum Exact {
    Int(i128),
    Float(f64),
    /// A Float64, which converts to Float16 with one rounding.
    Double(f64),
}

/// `x` as a Float16 rounded once (to nearest even), as `cvt.rn.f16.f64`.
pub fn f64_to_f16(x: f64) -> F16 {
    let h = F16::from_f32(x as f32);
    if x.is_nan() || h.to_f32().is_infinite() && (x as f32).is_infinite() {
        return h;
    }
    // The nearest of h and its neighbours, ties to the even encoding.
    let candidates = [h.0.wrapping_sub(1), h.0, h.0.wrapping_add(1)].map(F16).into_iter().filter(|c| {
        !c.to_f32().is_nan() && (c.0 & 0x7fff) <= 0x7c00 && (c.to_f32() >= 0.0) == (h.to_f32() >= 0.0) || c.0 == h.0
    });
    candidates
        .min_by(|a, b| {
            let (da, db) = ((a.to_f32() as f64 - x).abs(), (b.to_f32() as f64 - x).abs());
            da.partial_cmp(&db).unwrap().then((a.0 & 1).cmp(&(b.0 & 1)))
        })
        .unwrap()
}

/// A host type of the conversion matrix.
pub trait MatrixType: Numeric {
    fn exact(self) -> Exact;
    fn from_exact(x: Exact) -> Self;
}

macro_rules! matrix_ints {
    ($($t:ty),*) => {$(
        impl MatrixType for $t {
            fn exact(self) -> Exact {
                Exact::Int(self as i128)
            }
            fn from_exact(x: Exact) -> Self {
                match x {
                    Exact::Int(v) => v as $t,
                    Exact::Float(f) | Exact::Double(f) => f as $t,
                }
            }
        }
    )*};
}
matrix_ints!(i8, i16, i32, i64, u8, u16, u32, u64);

impl MatrixType for bool {
    fn exact(self) -> Exact {
        Exact::Int(self as i128)
    }
    fn from_exact(x: Exact) -> Self {
        match x {
            Exact::Int(v) => v != 0,
            Exact::Float(f) | Exact::Double(f) => f != 0.0,
        }
    }
}

impl MatrixType for f32 {
    fn exact(self) -> Exact {
        Exact::Float(self as f64)
    }
    fn from_exact(x: Exact) -> Self {
        match x {
            Exact::Int(v) => v as f32,
            Exact::Float(f) | Exact::Double(f) => f as f32,
        }
    }
}

impl MatrixType for f64 {
    fn exact(self) -> Exact {
        Exact::Double(self)
    }
    fn from_exact(x: Exact) -> Self {
        match x {
            Exact::Int(v) => v as f64,
            Exact::Float(f) | Exact::Double(f) => f,
        }
    }
}

impl MatrixType for F16 {
    fn exact(self) -> Exact {
        Exact::Float(self.to_f32() as f64)
    }
    fn from_exact(x: Exact) -> Self {
        match x {
            Exact::Int(v) => F16::from_f32(v as f32),
            Exact::Float(f) => F16::from_f32(f as f32),
            Exact::Double(f) => f64_to_f16(f),
        }
    }
}

/// BFloat16 converts through Float32 from every type (from Float64 and
/// the 64-bit integers that is two roundings, as emitted).
impl MatrixType for BF16 {
    fn exact(self) -> Exact {
        Exact::Float(self.to_f32() as f64)
    }
    fn from_exact(x: Exact) -> Self {
        match x {
            Exact::Int(v) => BF16::from_f32(v as f32),
            Exact::Float(f) | Exact::Double(f) => BF16::from_f32(f as f32),
        }
    }
}

macro_rules! conversion_row {
    ($name:ident, $src:ty; $($dst:ty),*) => {
        /// One row of the conversion matrix: four values per thread
        /// converted with `TensorSsa::to` to every standard type.
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([THREADS as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$src>("xs", Layout::col_major(THREADS * 4));
                let t = thread_idx().x;
                let x = local_tile(&xs, 4, t).load_ssa();
                $(
                    let out = param_tensor::<$dst>(stringify!($dst), Layout::col_major(THREADS * 4));
                    local_tile(&out, 4, t).store_ssa(&x.to::<$dst>());
                )*
            });
            fn check(args: &[KernelArg]) {
                let xs = args[0].to_vec::<$src>();
                let mut k = 1;
                $(
                    let got = args[k].to_vec::<$dst>();
                    for (i, &x) in xs.iter().enumerate() {
                        let want = <$dst as MatrixType>::from_exact(x.exact());
                        assert_eq!(
                            got[i].raw_bits(),
                            want.raw_bits(),
                            "{}: {x:?} to {} gave {:?}, not {want:?}",
                            stringify!($name),
                            stringify!($dst),
                            got[i]
                        );
                    }
                    k += 1;
                )*
                let _ = k;
            }
            Case { name: stringify!($name), kernel, cfg: cfg(), small_only: false, check }
        }
    };
}

macro_rules! conversion_matrix {
    ($($name:ident: $src:ty),*) => {
        $(conversion_row!($name, $src; bool, i8, i16, i32, i64, u8, u16, u32, u64, F16, BF16, f32, f64);)*
        /// The conversion matrix: every standard type to every other.
        pub fn conversion_matrix() -> Vec<Case> {
            vec![$($name()),*]
        }
    };
}

conversion_matrix!(
    convert_from_bool: bool, convert_from_i8: i8, convert_from_i16: i16, convert_from_i32: i32,
    convert_from_i64: i64, convert_from_u8: u8, convert_from_u16: u16, convert_from_u32: u32,
    convert_from_u64: u64, convert_from_f16: F16, convert_from_bf16: BF16, convert_from_f32: f32,
    convert_from_f64: f64
);

/// Float64 values a conversion through Float32 rounds differently: `cvt.rn.f16.f64` rounds once (and the
/// simulator too), while BFloat16 converts through Float32 as emitted (two roundings, documented).
pub const DOUBLE_ROUNDING: [f64; 8] = [
    1.0 + 1.0 / 2048.0 + 1.0 / (1u64 << 40) as f64,
    -(1.0 + 1.0 / 2048.0 + 1.0 / (1u64 << 40) as f64),
    1.0 + 3.0 / 2048.0 - 1.0 / (1u64 << 40) as f64,
    65504.0 + 16.0 - 1.0 / 1024.0,
    6.103515625e-05 * (1.0 + 1.0 / 2048.0 + 1.0 / (1u64 << 40) as f64),
    5.960464477539063e-08 * (0.5 + 1.0 / (1u64 << 30) as f64),
    1.0 + 1.0 / 256.0 + 1.0 / (1u64 << 40) as f64,
    -(1.0 + 3.0 / 256.0 - 1.0 / (1u64 << 40) as f64),
];

/// The Float64 to Float16 double-rounding regression, and BFloat16's
/// documented route through Float32, on values where the two differ.
pub fn double_rounding() -> Case {
    let kernel = Kernel::builder("double_rounding").block_dim([1, 1, 1]).build(|| {
        let h = param_tensor::<F16>("h", Layout::col_major(8));
        let b = param_tensor::<BF16>("b", Layout::col_major(8));
        let x = TensorSsa::new(8, DOUBLE_ROUNDING.iter().map(|&v| Value::constant(v)).collect());
        // Through a runtime value, so that nothing folds at build time.
        let x = x * thread_idx().x.to::<f64>().max(1.0).min(1.0);
        h.store_ssa(&x.to::<F16>());
        b.store_ssa(&x.to::<BF16>());
    });
    fn check(args: &[KernelArg]) {
        let (h, b) = (args[0].to_vec::<F16>(), args[1].to_vec::<BF16>());
        for (i, &x) in DOUBLE_ROUNDING.iter().enumerate() {
            let once = f64_to_f16(x);
            assert_eq!(h[i].0, once.0, "double_rounding: f16 of {x:e}");
            if i < 6 {
                assert_ne!(once.0, F16::from_f32(x as f32).0, "double_rounding: {x:e} must round differently twice");
            }
            assert_eq!(b[i].0, BF16::from_f32(x as f32).0, "double_rounding: bf16 of {x:e}");
        }
    }
    Case { name: "double_rounding", kernel, cfg: LaunchConfig::new([1, 1, 1], [1, 1, 1]), small_only: false, check }
}

/// The Int8 and Int4 to BFloat16 fast paths of `TensorSsa::to` over every
/// input, against the plain conversion and Rust.
pub fn int_to_bf16_fast_paths() -> Case {
    let kernel = Kernel::builder("int_to_bf16_fast_paths").block_dim([THREADS as u32, 1, 1]).build(|| {
        let fast = param_tensor::<BF16>("fast", Layout::col_major((8, THREADS)));
        let plain = param_tensor::<BF16>("plain", Layout::col_major((8, THREADS)));
        let fast4 = param_tensor::<BF16>("fast4", Layout::col_major((8, THREADS)));
        let plain4 = param_tensor::<BF16>("plain4", Layout::col_major((8, THREADS)));
        let t = thread_idx().x;
        let v = TensorSsa::new(8, (0..8).map(|k| (t * 8 + k - 128).to::<i8>()).collect());
        fast.slice((.., t)).store_ssa(&v.to::<BF16>());
        plain.slice((.., t)).store_ssa(&v.map(|x| x.to::<BF16>()));
        let n = TensorSsa::new(8, (0..8).map(|k| ((t * 8 + k) & 15).to::<I4>()).collect());
        fast4.slice((.., t)).store_ssa(&n.to::<BF16>());
        plain4.slice((.., t)).store_ssa(&n.map(|x| x.to::<BF16>()));
    });
    fn check(args: &[KernelArg]) {
        let (fast, plain) = (args[0].to_vec::<BF16>(), args[1].to_vec::<BF16>());
        let (fast4, plain4) = (args[2].to_vec::<BF16>(), args[3].to_vec::<BF16>());
        for i in 0..256 {
            let want = BF16::from_f32((i as i32 - 128) as f32);
            assert_eq!((fast[i].0, plain[i].0), (want.0, want.0), "int_to_bf16_fast_paths: Int8 {}", i as i32 - 128);
            let nib = (i & 15) as i32;
            let want = BF16::from_f32((if nib >= 8 { nib - 16 } else { nib }) as f32);
            assert_eq!((fast4[i].0, plain4[i].0), (want.0, want.0), "int_to_bf16_fast_paths: Int4 {nib}");
        }
    }
    Case { name: "int_to_bf16_fast_paths", kernel, cfg: cfg(), small_only: false, check }
}

/// Bitcasts and their inverses: four Float32 as eight Float16, two
/// BFloat16 as a Uint32, eight Uint8 as a Uint64, eight Int4 as a Uint32,
/// and equal widths.
pub fn ssa_bitcasts() -> Case {
    let kernel = Kernel::builder("ssa_bitcasts").block_dim([THREADS as u32, 1, 1]).build(|| {
        let f = param_tensor::<f32>("f", Layout::col_major((4, THREADS)));
        let b = param_tensor::<BF16>("b", Layout::col_major((2, THREADS)));
        let u = param_tensor::<u8>("u", Layout::col_major((8, THREADS)));
        let w = param_tensor::<u32>("w", Layout::col_major(THREADS));
        let halves = param_tensor::<F16>("halves", Layout::col_major((8, THREADS)));
        let words = param_tensor::<u32>("words", Layout::col_major((2, THREADS)));
        let longs = param_tensor::<u64>("longs", Layout::col_major(THREADS));
        let nibbles = param_tensor::<i8>("nibbles", Layout::col_major((8, THREADS)));
        let back = param_tensor::<f32>("back", Layout::col_major((4, THREADS)));
        let round = param_tensor::<bool>("round", Layout::col_major(THREADS));
        let t = thread_idx().x;
        let fv = f.slice((.., t)).load_ssa();
        let h = fv.bitcast::<F16>();
        assert_eq!(h.shape(), &IntTuple::from(8));
        halves.slice((.., t)).store_ssa(&h);
        back.slice((.., t)).store_ssa(&h.bitcast::<f32>().bitcast::<i32>().bitcast::<f32>());
        let pair = b.slice((.., t)).load_ssa().bitcast::<u32>();
        let bytes = u.slice((.., t)).load_ssa().bitcast::<u64>();
        words.store((0, t), pair.get(0));
        words.store((1, t), bytes.bitcast::<u32>().get(1));
        longs.store(t, bytes.get(0));
        let nib = TensorSsa::new(1, vec![w.load(t)]).bitcast::<I4>();
        assert_eq!(nib.size(), 8);
        nibbles.slice((.., t)).store_ssa(&nib.map(|x| x.to::<i8>()));
        let again = nib.bitcast::<u32>();
        round.store(t, again.get(0).eq(w.load(t)));
    });
    fn check(args: &[KernelArg]) {
        let (f, b, u, w) =
            (args[0].to_vec::<f32>(), args[1].to_vec::<BF16>(), args[2].to_vec::<u8>(), args[3].to_vec::<u32>());
        let (halves, words, longs) = (args[4].to_vec::<F16>(), args[5].to_vec::<u32>(), args[6].to_vec::<u64>());
        let (nibbles, back) = (args[7].to_vec::<i8>(), args[8].to_vec::<f32>());
        assert!(args[9].to_vec::<bool>().iter().all(|&b| b), "ssa_bitcasts: Int4 round trip");
        for t in 0..THREADS {
            let bytes: Vec<u8> = f[t * 4..t * 4 + 4].iter().flat_map(|x| x.to_le_bytes()).collect();
            for k in 0..8 {
                assert_eq!(halves[t * 8 + k].0, u16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]]), "f32 as f16 {k}");
            }
            for k in 0..4 {
                assert_eq!(back[t * 4 + k].to_bits(), f[t * 4 + k].to_bits(), "f32 round trip {k}");
            }
            let pair = u32::from(b[t * 2].0) | u32::from(b[t * 2 + 1].0) << 16;
            assert_eq!(words[t * 2], pair, "2 x bf16 as u32");
            let long = u64::from_le_bytes(u[t * 8..t * 8 + 8].try_into().unwrap());
            assert_eq!(longs[t], long, "8 x u8 as u64");
            assert_eq!(words[t * 2 + 1], (long >> 32) as u32, "u64 as 2 x u32");
            for k in 0..8 {
                let nib = (w[t] >> (4 * k) & 15) as i8;
                assert_eq!(nibbles[t * 8 + k], if nib >= 8 { nib - 16 } else { nib }, "u32 as Int4 {k}");
            }
        }
    }
    Case { name: "ssa_bitcasts", kernel, cfg: cfg(), small_only: false, check }
}

/// Float32 and Float16 tensors to FP8 two at a time (`TensorSsa::to`),
/// against the scalar conversion, and back (sm_89).
pub fn ssa_fp8() -> Case {
    let kernel = Kernel::builder("ssa_fp8").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((8, THREADS)));
        let e4 = param_tensor::<F8E4M3FN>("e4", Layout::col_major((8, THREADS)));
        let e5 = param_tensor::<F8E5M2>("e5", Layout::col_major((8, THREADS)));
        let h4 = param_tensor::<F8E4M3FN>("h4", Layout::col_major((8, THREADS)));
        let back = param_tensor::<f32>("back", Layout::col_major((8, THREADS)));
        let same = param_tensor::<bool>("same", Layout::col_major(THREADS));
        let t = thread_idx().x;
        let v = x.slice((.., t)).load_ssa() * 100.0;
        let a = v.to::<F8E4M3FN>();
        e4.slice((.., t)).store_ssa(&a);
        e5.slice((.., t)).store_ssa(&v.to::<F8E5M2>());
        h4.slice((.., t)).store_ssa(&v.to::<F16>().to::<F8E4M3FN>());
        back.slice((.., t)).store_ssa(&a.to::<f32>());
        // The scalar conversion gives the same bits.
        let scalar = v.map(|s| s.to::<F8E4M3FN>().bitcast::<u8>().to::<i32>());
        same.store(t, all_(&scalar.eq(a.map(|s| s.bitcast::<u8>().to::<i32>()))));
    });
    fn check(args: &[KernelArg]) {
        let x = args[0].to_vec::<f32>();
        let (e4, e5, h4, back) = (
            args[1].to_vec::<F8E4M3FN>(),
            args[2].to_vec::<F8E5M2>(),
            args[3].to_vec::<F8E4M3FN>(),
            args[4].to_vec::<f32>(),
        );
        for i in 0..x.len() {
            let v = x[i] * 100.0;
            assert_eq!(e4[i].0, F8E4M3FN::from_f32(v).0, "ssa_fp8: e4m3 of {v}");
            assert_eq!(e5[i].0, F8E5M2::from_f32(v).0, "ssa_fp8: e5m2 of {v}");
            assert_eq!(h4[i].0, F8E4M3FN::from_f32(F16::from_f32(v).to_f32()).0, "ssa_fp8: e4m3 of f16 {v}");
            assert_eq!(back[i], e4[i].to_f32(), "ssa_fp8: e4m3 back");
        }
        assert!(args[5].to_vec::<bool>().iter().all(|&b| b), "ssa_fp8: pairs and scalars differ");
    }
    Case { name: "ssa_fp8", kernel, cfg: cfg(), small_only: false, check }
}

/// Float32 tensors to FP4 and FP6 in pairs (sm_100a, sm_120a), checked on
/// the simulator.
pub fn ssa_fp4_fp6() -> Case {
    let kernel = Kernel::builder("ssa_fp4_fp6").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((8, THREADS)));
        let out = param_tensor::<f32>("out", Layout::col_major((8, 3, THREADS)));
        let t = thread_idx().x;
        let v = x.slice((.., t)).load_ssa() * 4.0;
        out.slice((.., 0, t)).store_ssa(&v.to::<F4E2M1FN>().to::<f32>());
        out.slice((.., 1, t)).store_ssa(&v.to::<F6E2M3FN>().to::<f32>());
        out.slice((.., 2, t)).store_ssa(&v.to::<F6E3M2FN>().to::<f32>());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..THREADS {
            for k in 0..8 {
                let v = x[t * 8 + k] * 4.0;
                let want =
                    [F4E2M1FN::from_f32(v).to_f32(), F6E2M3FN::from_f32(v).to_f32(), F6E3M2FN::from_f32(v).to_f32()];
                for (j, w) in want.into_iter().enumerate() {
                    assert_eq!(out[t * 24 + j * 8 + k], w, "ssa_fp4_fp6: format {j} of {v}");
                }
            }
        }
    }
    Case { name: "ssa_fp4_fp6", kernel, cfg: cfg(), small_only: false, check }
}

/// `print_tensor` of global (row- and column-major), shared, fragment,
/// tensor-value and coordinate tensors, verbose too, and of every type,
/// from thread 0. `tests/tensor_ssa.rs` checks the simulator's text; the
/// GPU runs it.
pub fn ssa_print() -> Case {
    let kernel = Kernel::builder("ssa_print").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::row_major((2, 3)));
        let y = param_tensor::<i32>("y", Layout::col_major((2, 3)));
        let smem = make_smem_tensor::<F16>(Layout::row_major((2, 2)));
        let t = thread_idx().x;
        if_(t.eq(0), || {
            for i in 0..2 {
                for j in 0..2 {
                    smem.store((i, j), x.load((i, j)).to::<F16>());
                }
            }
        });
        sync_threads();
        if_(t.eq(0), || {
            print_tensor(&x, false);
            print_tensor(&y, false);
            print_tensor(&smem, false);
            let frag = make_fragment::<u8>(3);
            frag.store_ssa(&TensorSsa::new(3, vec![Value::constant(7u8), Value::constant(200u8), t.to::<u8>()]));
            print_tensor(&frag, false);
            let cube = TensorSsa::new((2, 2, 2), (0..8).map(|k| Value::constant(k as i64 * 1000 - 3000)).collect());
            print_tensor(&cube, false);
            print_tensor(&domain_offset((2, 3), &make_identity_tensor((2, 2))), false);
            let mask = x.load_ssa().gt(1.0).reshape((2, 3));
            print_tensor(&mask, true);
            let k = |v: f32| Value::constant(v);
            print_tensor(&TensorSsa::new(2, vec![k(1.5).to::<BF16>(), k(-0.25).to::<BF16>()]), false);
            print_tensor(&TensorSsa::new(2, vec![Value::constant(1e10f64), Value::constant(-2.0f64)]), false);
            print_tensor(&TensorSsa::new(2, vec![Value::constant(u64::MAX), Value::constant(3u64)]), false);
            print_tensor(&TensorSsa::new(2, vec![Value::constant(-128i8), Value::constant(127i8)]), false);
            print_tensor(&TensorSsa::new(2, vec![Value::constant(65535u16), Value::constant(0u16)]), false);
            print_tensor(&TensorSsa::new(1, vec![Value::constant(-1i128)]), false);
            let fp8 = [F8E4M3::from_f32(240.0), F8E4M3::from_f32(-0.5)];
            print_tensor(&TensorSsa::new(2, fp8.iter().map(|&v| Value::constant(v)).collect()), false);
            print_tensor(&TensorSsa::new(2, vec![k(-3.0).to::<I4>(), k(5.0).to::<I4>()]), false);
        });
    });
    fn check(_: &[KernelArg]) {}
    Case { name: "ssa_print", kernel, cfg: cfg(), small_only: false, check }
}

/// Boolean tensors copied with multi-element atoms: a 32-bit universal
/// copy, G2R/R2G copies that pick their own width, and `load_ssa` /
/// `store_ssa`. A Boolean is a byte in memory and an `i1` in MLIR, whose
/// vectors LLVM bit-packs, so each element must move as its own byte.
pub fn bool_copies() -> Case {
    let kernel = Kernel::builder("bool_copies").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<bool>("x", Layout::col_major((8, THREADS)));
        let y = param_tensor::<bool>("y", Layout::col_major((8, THREADS)));
        let z = param_tensor::<bool>("z", Layout::col_major((8, THREADS)));
        let w = param_tensor::<bool>("w", Layout::col_major((8, THREADS)));
        let t = thread_idx().x;
        let (xs, ys, zs, ws) = (x.slice((.., t)), y.slice((.., t)), z.slice((.., t)), w.slice((.., t)));
        let wide = make_copy_atom::<bool>(CopyOp::Universal { bits: 32 });
        let frag = make_fragment_like::<bool, bool>(&xs);
        copy(&wide, &xs, &frag);
        copy(&wide, &frag, &ys);
        let frag2 = make_fragment_like::<bool, bool>(&xs);
        copy(&make_copy_atom::<bool>(CopyOp::G2R { bits: 0, attrs: G2RAttrs::default() }), &xs, &frag2);
        copy(&make_copy_atom::<bool>(CopyOp::R2G { bits: 0, attrs: R2GAttrs::default() }), &frag2, &zs);
        ws.store_ssa(&!xs.load_ssa());
    });
    fn check(args: &[KernelArg]) {
        let x = args[0].to_vec::<bool>();
        let (y, z, w) = (args[1].to_vec::<bool>(), args[2].to_vec::<bool>(), args[3].to_vec::<bool>());
        assert_eq!(y, x, "bool_copies: universal copy");
        assert_eq!(z, x, "bool_copies: G2R and R2G copies");
        assert!(w.iter().zip(&x).all(|(&a, &b)| a == !b), "bool_copies: load_ssa and store_ssa");
    }
    Case { name: "bool_copies", kernel, cfg: cfg(), small_only: false, check }
}

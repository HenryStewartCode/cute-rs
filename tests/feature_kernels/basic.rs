// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause

//! The basic cases: integer and float arithmetic, casts, memory kinds, warp operations, atomics, printing and a small inline-PTX fragment.

use super::*;
macro_rules! int_case {
    ($name:ident, $t:ty) => {
        pub fn $name() -> Case {
            const OPS: usize = 18;
            let kernel = Kernel::builder(concat!("int_", stringify!($t))).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                let ys = param_tensor::<$t>("ys", Layout::col_major(T));
                let out = param_tensor::<$t>("out", Layout::row_major((T, OPS)));
                let t = thread_idx().x;
                let (x, y) = (xs.load(t), ys.load(t));
                let d = (y & (7 as $t)) + (1 as $t);
                let s = y & (<$t>::BITS as $t - 1);
                let results: [Value<$t>; OPS] = [
                    x + y,
                    x - y,
                    x * y,
                    x / d,
                    x % d,
                    x.min(y),
                    x.max(y),
                    x & y,
                    x | y,
                    x ^ y,
                    x << s,
                    x >> s,
                    Value::constant(0 as $t) - x,
                    !x,
                    x.popc(),
                    x.clz(),
                    x.ctz(),
                    x.brev(),
                ];
                for (j, v) in results.into_iter().enumerate() {
                    out.store((t, j), v);
                }
            });
            fn check(args: &[KernelArg]) {
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                for t in 0..T {
                    let (x, y) = (xs[t], ys[t]);
                    let d = (y & 7) + 1;
                    let s = y & (<$t>::BITS as $t - 1);
                    let want: [$t; OPS] = [
                        x.wrapping_add(y),
                        x.wrapping_sub(y),
                        x.wrapping_mul(y),
                        x.wrapping_div(d),
                        x.wrapping_rem(d),
                        x.min(y),
                        x.max(y),
                        x & y,
                        x | y,
                        x ^ y,
                        x.wrapping_shl(s as u32),
                        x.wrapping_shr(s as u32),
                        (0 as $t).wrapping_sub(x),
                        !x,
                        x.count_ones() as $t,
                        x.leading_zeros() as $t,
                        x.trailing_zeros() as $t,
                        x.reverse_bits(),
                    ];
                    for (j, w) in want.into_iter().enumerate() {
                        assert_eq!(out[t * OPS + j], w, "{} op {j}, x = {x}, y = {y}", stringify!($t));
                    }
                }
            }
            Case { name: concat!("int_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

int_case!(int_i8, i8);
int_case!(int_i16, i16);
int_case!(int_i32, i32);
int_case!(int_i64, i64);
int_case!(int_u8, u8);
int_case!(int_u16, u16);
int_case!(int_u32, u32);
int_case!(int_u64, u64);

macro_rules! float_case {
    ($name:ident, $t:ty, $host:ty, $widen:expr, $narrow:expr, $fma:expr) => {
        pub fn $name() -> Case {
            const OPS: usize = 14;
            let kernel = Kernel::builder(concat!("float_", stringify!($t))).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                let ys = param_tensor::<$t>("ys", Layout::col_major(T));
                let out = param_tensor::<$t>("out", Layout::row_major((T, OPS)));
                let t = thread_idx().x;
                let (x, y) = (xs.load(t), ys.load(t));
                let results: [Value<$t>; OPS] = [
                    x + y,
                    x - y,
                    x * y,
                    x / y,
                    x.min(y),
                    x.max(y),
                    -x,
                    x.abs(),
                    x.abs().sqrt(),
                    (x * y).floor(),
                    (x * y).ceil(),
                    (x * y).trunc(),
                    (x * y).rint(),
                    fma(x, y, x),
                ];
                for (j, v) in results.into_iter().enumerate() {
                    out.store((t, j), v);
                }
            });
            fn check(args: &[KernelArg]) {
                let widen: fn($t) -> $host = $widen;
                let narrow: fn($host) -> $t = $narrow;
                // `x * y + z` rounded once to the type itself.
                let fma_once: fn($t, $t, $t) -> $t = $fma;
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                for t in 0..T {
                    let (x, y) = (widen(xs[t]), widen(ys[t]));
                    let p = widen(narrow(x * y));
                    let want: [$host; OPS - 1] = [
                        x + y,
                        x - y,
                        x * y,
                        x / y,
                        x.min(y),
                        x.max(y),
                        -x,
                        x.abs(),
                        x.abs().sqrt(),
                        p.floor(),
                        p.ceil(),
                        p.trunc(),
                        p.round_ties_even(),
                    ];
                    for (j, w) in want.into_iter().enumerate() {
                        assert_eq!(
                            out[t * OPS + j].raw_bits(),
                            narrow(w).raw_bits(),
                            "{} op {j}, x = {x}, y = {y}",
                            stringify!($t)
                        );
                    }
                    // fma rounds once: an f32 fma narrowed to Float16 or BFloat16
                    // would round twice, and differ where the first rounding ties.
                    assert_eq!(
                        out[t * OPS + OPS - 1].raw_bits(),
                        fma_once(xs[t], ys[t], xs[t]).raw_bits(),
                        "{} fma, x = {x}, y = {y}",
                        stringify!($t)
                    );
                }
            }
            Case { name: concat!("float_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

float_case!(float_f32, f32, f32, |x| x, |x| x, f32::mul_add);
float_case!(float_f64, f64, f64, |x| x, |x| x, f64::mul_add);
float_case!(float_f16, F16, f32, F16::to_f32, F16::from_f32, |x, y, z| F16(F16_FORMAT.fma(x.0, y.0, z.0)));
float_case!(float_bf16, BF16, f32, BF16::to_f32, BF16::from_f32, |x, y, z| BF16(BF16_FORMAT.fma(x.0, y.0, z.0)));

pub fn casts() -> Case {
    const N: usize = 16;
    let kernel = Kernel::builder("casts").block_dim([T as u32, 1, 1]).build(|| {
        let ints = param_tensor::<i32>("ints", Layout::col_major(T));
        let floats = param_tensor::<f32>("floats", Layout::col_major(T));
        let out = param_tensor::<i64>("out", Layout::row_major((T, N)));
        let t = thread_idx().x;
        let (y, x) = (ints.load(t), floats.load(t) * 1000.0);
        let results: [Int64; N] = [
            y.to::<i8>().to(),
            y.to::<u8>().to(),
            y.to::<i16>().to(),
            y.to::<u16>().to(),
            y.to::<i64>(),
            y.to::<u32>().to(),
            y.to::<u32>().to::<u64>().to::<i32>().to(),
            y.to::<f64>().bitcast(),
            y.to::<f32>().to::<f64>().bitcast(),
            x.to::<i32>().to(),
            x.abs().to::<u32>().to(),
            x.to::<i64>(),
            x.to::<F16>().to::<f64>().bitcast(),
            x.to::<BF16>().to::<f64>().bitcast(),
            x.to::<f64>().to::<f32>().to::<f64>().bitcast(),
            x.to::<F16>().to::<BF16>().to::<f64>().bitcast(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (ys, xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<f32>(), args[2].to_vec::<i64>());
        for t in 0..T {
            let (y, x) = (ys[t], xs[t] * 1000.0);
            let f = |v: f64| v.to_bits() as i64;
            let want: [i64; N] = [
                y as i8 as i64,
                y as u8 as i64,
                y as i16 as i64,
                y as u16 as i64,
                y as i64,
                y as u32 as i64,
                y as i64,
                f(y as f64),
                f(y as f32 as f64),
                x as i32 as i64,
                x.abs() as u32 as i64,
                x as i64,
                f(F16::from_f32(x).to_f32() as f64),
                f(BF16::from_f32(x).to_f32() as f64),
                f(x as f64),
                f(BF16::from_f32(F16::from_f32(x).to_f32()).to_f32() as f64),
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * N + j], w, "cast {j}, y = {y}, x = {x}");
            }
        }
    }
    Case { name: "casts", kernel, cfg: launch(1), small_only: false, check }
}

pub fn wide_multiply() -> Case {
    let kernel = Kernel::builder("wide_multiply").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t), ys.load(t));
        out.store((t, 0), x.mul_hi(x));
        out.store((t, 1), y.mul_hi(y).bitcast::<u32>());
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            assert_eq!(out[t * 2], ((xs[t] as u64 * xs[t] as u64) >> 32) as u32);
            assert_eq!(out[t * 2 + 1], ((ys[t] as i64 * ys[t] as i64) >> 32) as u32);
        }
    }
    Case { name: "wide_multiply", kernel, cfg: launch(1), small_only: false, check }
}

/// Steps of the Collatz iteration from a different start in every thread, so
/// the warp diverges and reconverges in a `while` loop.
pub fn while_loop() -> Case {
    let kernel = Kernel::builder("while_loop").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let start = (xs.load(t) & 255u32) + 1u32;
        let (_, steps) = while_carry(
            (start, Int32::from(0)),
            |(n, _)| n.ne(1u32),
            |(n, steps)| (select((n & 1u32).eq(0u32), n / 2u32, n * 3u32 + 1u32), steps + 1),
        );
        out.store(t, steps);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let mut n = (xs[t] & 255) + 1;
            let mut steps = 0;
            while n != 1 {
                n = if n % 2 == 0 { n / 2 } else { n * 3 + 1 };
                steps += 1;
            }
            assert_eq!(out[t], steps, "thread {t}");
        }
    }
    Case { name: "while_loop", kernel, cfg: launch(1), small_only: false, check }
}

/// A per-thread array written and read at runtime indices.
pub fn stack_array() -> Case {
    let kernel = Kernel::builder("stack_array").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let x = xs.load(t);
        let table = make_rmem_tensor::<u32>(Layout::col_major(16));
        for_range(0, 16, 1, |i| table.store(i, x + i.to::<u32>() * 3u32));
        let index = (x & 15u32).to::<i32>();
        out.store(t, table.load(index));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            assert_eq!(out[t], xs[t].wrapping_add((xs[t] & 15) * 3), "thread {t}");
        }
    }
    Case { name: "stack_array", kernel, cfg: launch(1), small_only: false, check }
}

/// One shared-memory allocation written as words and read back as halves and
/// bytes by the other warp.
pub fn reinterpreted_memory() -> Case {
    let kernel = Kernel::builder("reinterpreted_memory").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 3)));
        let t = thread_idx().x;
        let words = alloc_smem::<u32>(T, 16);
        make_tensor(&words, Layout::col_major(T)).store(t, xs.load(t));
        sync_threads();
        let other = (t + 32) % 64;
        let halves = make_tensor(&words.cast::<u16>(), Layout::col_major(T * 2));
        let bytes = make_tensor(&words.cast::<u8>(), Layout::col_major(T * 4));
        out.store((t, 0), halves.load(other * 2).to::<u32>());
        out.store((t, 1), halves.load(other * 2 + 1).to::<u32>());
        out.store((t, 2), bytes.load(other * 4 + 3).to::<u32>());
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let x = xs[(t + 32) % 64];
            assert_eq!([out[t * 3], out[t * 3 + 1], out[t * 3 + 2]], [x & 0xffff, x >> 16, x >> 24], "thread {t}");
        }
    }
    Case { name: "reinterpreted_memory", kernel, cfg: launch(1), small_only: false, check }
}

/// A global buffer whose length is only known at launch, passed the same way
/// as a pointer.
pub fn runtime_sized_buffer() -> Case {
    let kernel = Kernel::builder("runtime_sized_buffer").block_dim([T as u32, 1, 1]).build(|| {
        let p = param_ptr::<u32>("data");
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let tile = make_tensor(&p.offset(t * 2), Layout::col_major(2));
        out.store(t, tile.load(0usize) ^ tile.load(1usize));
    });
    fn check(args: &[KernelArg]) {
        let (data, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            assert_eq!(out[t], data[t * 2] ^ data[t * 2 + 1], "thread {t}");
        }
    }
    Case { name: "runtime_sized_buffer", kernel, cfg: launch(1), small_only: false, check }
}

/// More shared memory than can be allocated statically.
pub fn dynamic_shared_memory() -> Case {
    const WORDS: usize = 16 * 1024;
    let kernel = Kernel::builder("dynamic_shared_memory").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let smem = make_smem_tensor::<u32>(Layout::col_major(WORDS));
        let far = |i: Int32| i * 200 + 1000;
        smem.store(far(t), xs.load(t));
        sync_threads();
        out.store(t, smem.load(far((t + 32) % 64)));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            assert_eq!(out[t], xs[(t + 32) % 64], "thread {t}");
        }
    }
    Case { name: "dynamic_shared_memory", kernel, cfg: launch(1), small_only: false, check }
}

pub fn warp_operations() -> Case {
    const N: usize = 16;
    let kernel = Kernel::builder("warp_operations").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, N)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let flag = (x & 1u32).eq(1u32);
        sync_warp();
        let halves = if_else_yield((t % 2).eq(0), activemask, || 0u32.into_value());
        let results: [Uint32; N] = [
            shuffle_sync(x, 3),
            shuffle_sync_up(x, 2),
            shuffle_sync_down(x, 5),
            shuffle_sync_bfly(x, 1),
            vote_all_sync(flag).to(),
            vote_any_sync(flag).to(),
            vote_uni_sync(flag).to(),
            vote_ballot_sync(flag),
            match_sync(FULL_MASK, x & 3u32, MatchKind::Any),
            vote_ballot_sync(elect_one()).popc(),
            activemask(),
            lanemask_eq(),
            lanemask_le(),
            lanemask_lt(),
            lanemask_ge() ^ lanemask_gt(),
            halves,
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for warp in 0..T / 32 {
            let x = |l: usize| xs[warp * 32 + l];
            let flags = |l: usize| x(l) & 1 == 1;
            let ballot = (0..32).fold(0u32, |acc, l| acc | (flags(l) as u32) << l);
            for l in 0..32 {
                let row = &out[(warp * 32 + l) * N..(warp * 32 + l + 1) * N];
                let same = (0..32).fold(0u32, |acc, o| acc | ((x(o) & 3 == x(l) & 3) as u32) << o);
                let bit = 1u32 << l;
                let want = [
                    x(3),
                    if l >= 2 { x(l - 2) } else { x(l) },
                    if l + 5 < 32 { x(l + 5) } else { x(l) },
                    x(l ^ 1),
                    (ballot == u32::MAX) as u32,
                    (ballot != 0) as u32,
                    (ballot == u32::MAX || ballot == 0) as u32,
                    ballot,
                    same,
                    1,
                    u32::MAX,
                    bit,
                    bit | (bit - 1),
                    bit - 1,
                    bit,
                    if l % 2 == 0 { 0x5555_5555 } else { 0 },
                ];
                assert_eq!(row, want, "warp {warp} lane {l}");
            }
        }
    }
    Case { name: "warp_operations", kernel, cfg: launch(1), small_only: false, check }
}

pub fn atomics() -> Case {
    let kernel = Kernel::builder("atomics").block_dim([T as u32, 1, 1]).build(|| {
        let g = param_tensor::<u32>("g", Layout::col_major(8));
        let wide = param_tensor::<u64>("wide", Layout::col_major(2));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let u = t.to::<u32>();
        let smem = alloc_smem::<u32>(1, 4);
        if_(t.eq(0), || smem.store(0u32));
        sync_threads();
        atomic_add(&smem, u);
        sync_threads();
        out.store(t, smem.load());
        // Global memory, from every thread of every block.
        let at = |i: usize| g.ptr().offset(i);
        atomic_add(&at(0), 1u32);
        atomic_and(&at(1), !(1u32 << 3));
        atomic_or(&at(2), 1u32 << 3);
        atomic_xor(&at(3), 0xffu32);
        atomic_min(&at(4), 3u32);
        atomic_max(&at(5), 1u32 << 30);
        atomic_exch(&at(6), 77u32);
        // Exactly one thread sees the old value and swaps it.
        atomic_cas(&at(7), 5u32, 6u32);
        atomic_add(&wide.ptr(), 1u64 << 40);
        atomic_max(&wide.ptr().offset(1usize), u.to::<u64>());
    });
    fn check(args: &[KernelArg]) {
        let out = args[2].to_vec::<u32>();
        let (g, wide) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>());
        assert_eq!(out, vec![(0..T as u32).sum::<u32>(); T]);
        assert_eq!(g[6], 77);
        assert!(g[5] >= 1 << 30);
        assert!(g[4] <= 3);
        assert!(wide[1] >= T as u64 - 1);
    }
    Case { name: "atomics", kernel, cfg: launch(2), small_only: false, check }
}

pub fn printing() -> Case {
    let kernel = Kernel::builder("printing").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let t = thread_idx().x;
        if_(t.eq(5), || printf("thread %d has %f\n", &[t.into(), xs.load(t).into()]));
    });
    fn check(_: &[KernelArg]) {}
    Case { name: "printing", kernel, cfg: launch(1), small_only: false, check }
}

pub fn inline_ptx_fragment() -> Case {
    let kernel = Kernel::builder("inline_ptx_fragment").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let x = xs.load(t);
        let bits: Uint32 = inline_ptx_with_sim("popc.b32 $0, $1;", &[x.into()], |a| vec![a[0].count_ones() as u64]);
        out.store(t, bits);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            assert_eq!(out[t], xs[t].count_ones());
        }
    }
    Case { name: "inline_ptx_fragment", kernel, cfg: launch(1), small_only: false, check }
}

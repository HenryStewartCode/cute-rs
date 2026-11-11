//! Kernels covering every scalar type, operation and intrinsic. Each has a
//! check written in plain Rust; `tests/features.rs` holds the simulator to
//! it, and `tests/gpu.rs` holds the GPU to the simulator.

#![allow(dead_code)]

use cute::ir::Param;
use cute::prelude::*;

/// Threads per block in most of these kernels: two warps.
const T: usize = 64;

pub struct Case {
    pub name: &'static str,
    pub kernel: Kernel,
    pub cfg: LaunchConfig,
    /// Only small integers are bit-exact between the simulator and the GPU.
    pub small_only: bool,
    /// Checks the simulator's output against a reference in Rust.
    pub check: fn(&[KernelArg]),
}

fn launch(blocks: u32) -> LaunchConfig {
    LaunchConfig::new([blocks, 1, 1], [T as u32, 1, 1])
}

/// Inputs for every parameter of `kernel`, the same on every run. Buffers
/// sized at launch get `runtime_len` elements.
pub fn inputs_with_len(kernel: &Kernel, small: bool, runtime_len: usize) -> Vec<KernelArg> {
    let ir = kernel.ir();
    let mut state = 1u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state
    };
    ir.args
        .iter()
        .map(|p| match p {
            Param::Buffer(b) => {
                let d = ir.buffer(*b);
                let n = d.len.unwrap_or(runtime_len);
                macro_rules! ints {
                    ($t:ty) => {{
                        let v: Vec<$t> = (0..n)
                            .map(|_| if small { ((next() >> 40) % 17) as $t } else { (next() >> 16) as $t })
                            .collect();
                        KernelArg::buffer(&v)
                    }};
                }
                let float = |next: &mut dyn FnMut() -> u64| -> f64 {
                    let bits = next() >> 40;
                    if small { (bits % 17) as f64 - 8.0 } else { bits as f64 / (1u64 << 23) as f64 - 1.0 }
                };
                match d.elem {
                    DType::I8 => ints!(i8),
                    DType::I16 => ints!(i16),
                    DType::I32 => ints!(i32),
                    DType::I64 => ints!(i64),
                    DType::U8 => ints!(u8),
                    DType::U16 => ints!(u16),
                    DType::U32 => ints!(u32),
                    DType::U64 => ints!(u64),
                    DType::F16 => {
                        KernelArg::buffer(&(0..n).map(|_| F16::from_f32(float(&mut next) as f32)).collect::<Vec<_>>())
                    }
                    DType::BF16 => {
                        KernelArg::buffer(&(0..n).map(|_| BF16::from_f32(float(&mut next) as f32)).collect::<Vec<_>>())
                    }
                    DType::F32 => KernelArg::buffer(&(0..n).map(|_| float(&mut next) as f32).collect::<Vec<_>>()),
                    DType::F64 => KernelArg::buffer(&(0..n).map(|_| float(&mut next)).collect::<Vec<_>>()),
                    DType::Bool => KernelArg::buffer(&(0..n).map(|_| next() >> 63 == 1).collect::<Vec<_>>()),
                }
            }
            Param::Scalar { value, .. } => match ir.dtype(*value) {
                DType::F32 => KernelArg::ScalarF32(1.5),
                other => panic!("no test input for a {other} scalar"),
            },
        })
        .collect()
}

pub fn inputs(kernel: &Kernel, small: bool) -> Vec<KernelArg> {
    inputs_with_len(kernel, small, 0)
}

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
    ($name:ident, $t:ty, $host:ty, $widen:expr, $narrow:expr) => {
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
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                for t in 0..T {
                    let (x, y) = (widen(xs[t]), widen(ys[t]));
                    let p = widen(narrow(x * y));
                    let want: [$host; OPS] = [
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
                        x.mul_add(y, x),
                    ];
                    for (j, w) in want.into_iter().enumerate() {
                        assert_eq!(
                            out[t * OPS + j].raw_bits(),
                            narrow(w).raw_bits(),
                            "{} op {j}, x = {x}, y = {y}",
                            stringify!($t)
                        );
                    }
                }
            }
            Case { name: concat!("float_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

float_case!(float_f32, f32, f32, |x| x, |x| x);
float_case!(float_f64, f64, f64, |x| x, |x| x);
float_case!(float_f16, F16, f32, F16::to_f32, F16::from_f32);
float_case!(float_bf16, BF16, f32, BF16::to_f32, BF16::from_f32);

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
            match_sync(x & 3u32),
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

/// The `mma.sync` kind of an SM80 MMA operation.
fn sm80_kind(op: impl Into<MmaOp>) -> MmaKind {
    match op.into() {
        MmaOp::Sm80(kind) => kind,
        other => panic!("{other:?} is not an mma.sync"),
    }
}

/// One warp, one `mma.sync` per tile of K, `K_TILES` tiles.
fn mma_kernel<TA: Numeric, TC: Numeric>(name: &'static str, op: MmaOp, zero: TC) -> Kernel {
    let (m, n, k_atom) = sm80_kind(op).shape_mnk();
    let k = k_atom * K_TILES;
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<TA>("a", Layout::col_major((m, k)));
        let b = param_tensor::<TA>("b", Layout::col_major((n, k)));
        let c = param_tensor::<TC>("c", Layout::col_major((m, n)));
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let thr = mma.get_slice(lane_idx());
        let (ta, tb, tc) = (thr.partition_a(&a), thr.partition_b(&b), thr.partition_c(&c));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TA, _>(&b));
        let acc = thr.partition_fragment_c::<TC, _>(&c);
        for i in 0..fa.size() {
            fa.store(i, ta.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, tb.load(i));
        }
        acc.fill(zero);
        gemm(&mma, &acc, &fa, &fb, &acc);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

const K_TILES: usize = 2;

macro_rules! mma_case {
    ($name:ident, $op:expr, $ta:ty, $tc:ty, $zero:expr, $as_f64:expr) => {
        pub fn $name() -> Case {
            let kernel = mma_kernel::<$ta, $tc>(stringify!($name), $op.into(), $zero);
            fn check(args: &[KernelArg]) {
                let as_a: fn($ta) -> f64 = $as_f64.0;
                let as_c: fn($tc) -> f64 = $as_f64.1;
                let (m, n, k_atom) = sm80_kind($op).shape_mnk();
                let k = k_atom * K_TILES;
                let (a, b, c) = (args[0].to_vec::<$ta>(), args[1].to_vec::<$ta>(), args[2].to_vec::<$tc>());
                for i in 0..m {
                    for j in 0..n {
                        let want: f64 = (0..k).map(|l| as_a(a[i + m * l]) * as_a(b[j + n * l])).sum();
                        assert_eq!(as_c(c[i + m * j]), want, "C[{i},{j}]");
                    }
                }
            }
            Case {
                name: stringify!($name),
                kernel,
                cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
                small_only: true,
                check,
            }
        }
    };
}

mma_case!(mma_k8, MmaKind::M16N8K8F32F16, F16, f32, 0.0f32, (|x: F16| x.to_f32() as f64, |x: f32| x as f64));
mma_case!(
    mma_f16_accumulate,
    MmaKind::M16N8K16F16F16,
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
mma_case!(mma_tf32, MmaKind::M16N8K4Tf32, f32, f32, 0.0f32, (|x: f32| x as f64, |x: f32| x as f64));
mma_case!(mma_f64, MmaKind::M8N8K4F64, f64, f64, 0.0f64, (|x: f64| x, |x: f64| x));
mma_case!(
    mma_bf16,
    MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 16)),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
mma_case!(
    mma_bf16_k8,
    MmaF16BF16Op::new(DType::BF16, DType::F32, (16, 8, 8)),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
mma_case!(
    mma_f16_accumulate_k8,
    MmaF16BF16Op::new(DType::F16, DType::F16, (16, 8, 8)),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
mma_case!(mma_tf32_k8, MmaTF32Op::new((16, 8, 8)), f32, f32, 0.0f32, (|x: f32| x as f64, |x: f32| x as f64));

/// `make_tiled_copy_tv` with a warp-level atom: `ldmatrix.x4` over a
/// `32 x 8` tile, 8 x 4 threads each receiving a `4 x 2` block. The
/// destination is a `(2, 4)` register tensor (columns fastest), whose values
/// only CuTe's general `retile` gathers in the copy's order. The output must
/// be the input.
pub fn tiled_copy_tv_ldmatrix() -> Case {
    let kernel = Kernel::builder("tiled_copy_tv_ldmatrix").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((32, 8)));
        let out = param_tensor::<F16>("out", Layout::row_major((32, 8)));
        let t = thread_idx().x;
        let s = make_smem_tensor::<F16>(Layout::row_major((32, 8)));
        for j in 0..8 {
            s.store((t, j), a.load((t, j)));
        }
        sync_threads();
        let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
        let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::row_major((4, 2)));
        assert_eq!((tiled.size(), tiled.tile_shape()), (32, IntTuple::from((32, 8))));
        let frag = make_fragment::<F16>((2, 4));
        copy(&tiled, &tiled.get_slice(t).partition_s(&s), &tiled.retile(&frag));
        // Thread `4 * tm + tk` owns rows `4 * tm + i` and columns `2 * tk + j`.
        let (tm, tk) = (t / 4, t % 4);
        for i in 0..4 {
            for j in 0..2 {
                out.store((tm * 4 + i, tk * 2 + j), frag.load((j, i)));
            }
        }
    });
    fn check(args: &[KernelArg]) {
        assert_eq!(args[1].to_vec::<F16>(), args[0].to_vec::<F16>());
    }
    Case {
        name: "tiled_copy_tv_ldmatrix",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
        small_only: false,
        check,
    }
}

/// The tiled-copy constructors around a `2 x 2`-warp m16n8k16 MMA: the A and
/// B fragments are loaded from global memory by `make_tiled_copy_d` copies
/// that partition like the `ldmatrix` copies feeding the MMA, C is stored
/// with `make_tiled_copy_c_atom` and `retile`, and `make_tiled_copy_s`
/// copies out the row each thread would hand `ldmatrix`.
pub fn mma_tiled_copies() -> Case {
    const M: usize = 32;
    const N: usize = 16;
    const K: usize = 16;
    let kernel = Kernel::builder("mma_tiled_copies").block_dim([128, 1, 1]).build(|| {
        let a = param_tensor::<F16>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F16>("b", Layout::row_major((N, K)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let rows = param_tensor::<F16>("rows", Layout::row_major((128, 8)));
        let t = thread_idx().x;
        let mma = make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major((2, 2, 1)));
        let ldsm_a = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
        let ldsm_b = make_tiled_copy_b(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 2, trans: false }), &mma);
        let pairs = make_copy_atom::<F16>(CopyOp::Universal { bits: 32 });
        let g2r_a = make_tiled_copy_d(&pairs, &ldsm_a);
        let g2r_b = make_tiled_copy_d(&pairs, &ldsm_b);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) = (thr_mma.partition_fragment_a::<F16, _>(&a), thr_mma.partition_fragment_b::<F16, _>(&b));
        copy(&g2r_a, &g2r_a.get_slice(t).partition_s(&a), &g2r_a.retile(&fa));
        copy(&g2r_b, &g2r_b.get_slice(t).partition_s(&b), &g2r_b.retile(&fb));
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let r2g = make_tiled_copy_c_atom(&make_copy_atom::<f32>(CopyOp::Universal { bits: 64 }), &mma);
        copy(&r2g, &r2g.retile(&acc), &r2g.get_slice(t).partition_d(&c));
        let s_copy = make_tiled_copy_s(&make_copy_atom::<F16>(CopyOp::Universal { bits: 128 }), &ldsm_a);
        let src = s_copy.get_slice(t).partition_s(&a);
        let row = make_fragment_like::<F16, F16>(&src);
        copy(&s_copy, &src, &row);
        for v in 0..8 {
            rows.store((t, v), row.load(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c, rows) =
            (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>(), args[3].to_vec::<F16>());
        for i in 0..M {
            for j in 0..N {
                let want: f32 = (0..K).map(|l| a[i * K + l].to_f32() * b[j * K + l].to_f32()).sum();
                assert_eq!(c[i * N + j], want, "C[{i},{j}]");
            }
        }
        // ldmatrix.x4 for an m16n8k16 A fragment loads matrices A[0:8, 0:8],
        // A[8:16, 0:8], A[0:8, 8:16] and A[8:16, 8:16] (registers a0a1, a2a3,
        // a4a5, a6a7 of the PTX fragment table); lane `l` supplies row `l % 8`
        // of matrix `l / 8`. Warp `w` covers rows `16 * (w % 2)` on.
        for t in 0..128 {
            let (w, l) = (t / 32, t % 32);
            let (row, col) = (16 * (w % 2) + 8 * ((l / 8) % 2) + l % 8, 8 * (l / 16));
            for v in 0..8 {
                assert_eq!(rows[t * 8 + v], a[row * K + col + v], "thread {t} value {v}");
            }
        }
    }
    Case { name: "mma_tiled_copies", kernel, cfg: LaunchConfig::new([1, 1, 1], [128, 1, 1]), small_only: true, check }
}

/// Named barriers on 8 warps: two groups of 4 warps each exchange data
/// behind their own barrier inside a branch, warps 0-3 hand data to warps
/// 4-7 with `bar.arrive` / `bar.sync` and get an unaligned arrive back, and
/// a whole-block named barrier orders the final cross-group reads. Last, a
/// `NamedBarrier` whose id is computed at run time (7 for warps 0-3, 8 for
/// warps 4-7) orders another exchange within each group. The simulator's
/// race checker must accept every access.
pub fn named_barriers() -> Case {
    const N: usize = 256;
    let kernel = Kernel::builder("named_barriers").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let out = param_tensor::<u32>("out", Layout::row_major((N, 5)));
        let t = thread_idx().x;
        let warp = warp_idx();
        let smem = make_smem_tensor::<u32>(Layout::col_major(3 * N));
        let x = xs.load(t);
        if_else(
            warp.lt(4),
            || {
                smem.store(t, x);
                barrier(1, 128);
                out.store((t, 0), smem.load((t + 32) % 128));
            },
            || {
                smem.store(t, x * 3u32);
                barrier(2, 128);
                out.store((t, 0), smem.load((t + 32) % 128 + 128));
            },
        );
        let full = pipeline::NamedBarrier::new(3, N as u32);
        if_else(
            warp.lt(4),
            || {
                smem.store(t + 256, x ^ 0xffu32);
                full.arrive();
                out.store((t, 1), 0u32);
            },
            || {
                full.arrive_and_wait();
                out.store((t, 1), smem.load(t + 128));
                pipeline::arrive_unaligned(6, N as i32);
            },
        );
        // The producers reuse the buffer once the consumers have read it.
        if_(warp.lt(4), || {
            pipeline::wait(6, N as i32);
            smem.store(t + 256, x + 1u32);
        });
        pipeline::sync(5);
        out.store((t, 2), smem.load((t + 128) % 256));
        out.store((t, 3), smem.load((t % 128) + 256));
        let group = pipeline::NamedBarrier::new(warp / 4 + 7, 128);
        smem.store(t + 512, x * 7u32 + 1u32);
        group.arrive_and_wait();
        out.store((t, 4), smem.load(t / 128 * 128 + (t + 64) % 128 + 512));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        let first = |i: usize| if i < 128 { xs[i] } else { xs[i].wrapping_mul(3) };
        for t in 0..N {
            let row = &out[t * 5..t * 5 + 5];
            let group = t / 128 * 128;
            let want = [
                first(group + (t + 32) % 128),
                if t < 128 { 0 } else { xs[t - 128] ^ 0xff },
                first((t + 128) % 256),
                xs[t % 128].wrapping_add(1),
                xs[group + (t + 64) % 128].wrapping_mul(7).wrapping_add(1),
            ];
            assert_eq!(row, want, "thread {t}");
        }
    }
    Case {
        name: "named_barriers",
        kernel,
        cfg: LaunchConfig::new([2, 1, 1], [N as u32, 1, 1]),
        small_only: false,
        check,
    }
}

/// mbarriers on SM80: warps 0-1 hand data to warps 2-3 through an mbarrier
/// (`mbarrier.arrive` releases, a spin on `mbarrier.test_wait` acquires),
/// and the next phase goes the other way. Each consumer also tests the
/// phase still running (false) and the one just completed (true).
///
/// Warps 0-1 wait for phase 0 themselves before waiting on parity 1: while
/// phase 0 is running, parity 1 names the phase before it, which counts as
/// complete, so without that wait a warp that arrives early would pass its
/// parity-1 wait at once and read data warps 2-3 have not written.
pub fn mbarriers_sm80() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("mbarriers_sm80").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let out = param_tensor::<u32>("out", Layout::row_major((N, 3)));
        let t = thread_idx().x;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(N));
        let spin = |parity: i32| while_carry((), |()| !mbarrier_test_wait(&bar, parity), |()| ());
        if_(t.eq(0), || mbarrier_init(&bar, 64));
        sync_threads();
        let x = xs.load(t);
        let consume = |offset: i32, parity: i32| {
            spin(parity);
            out.store((t, 0), data.load(t + offset));
            out.store((t, 1), mbarrier_test_wait(&bar, 1 - parity).to::<u32>());
            out.store((t, 2), mbarrier_test_wait(&bar, parity).to::<u32>());
        };
        if_else(
            warp_idx().lt(2),
            || {
                data.store(t, x * 5u32);
                mbarrier_arrive(&bar);
                spin(0);
            },
            || consume(-64, 0),
        );
        if_else(
            warp_idx().ge(2),
            || {
                data.store(t, x + 7u32);
                mbarrier_arrive_with(&bar, 1, MbarrierOptions::default());
            },
            || consume(64, 1),
        );
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..N {
            let moved = if t < 64 { xs[t + 64].wrapping_add(7) } else { xs[t - 64].wrapping_mul(5) };
            assert_eq!(out[t * 3..t * 3 + 3], [moved, 0, 1], "thread {t}");
        }
    }
    Case { name: "mbarriers_sm80", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// The SM90 mbarrier operations, in a cluster of two CTAs: a standalone
/// `mbarrier.expect_tx`, arrivals with a count, relaxed and cluster-scope
/// arrivals, two bulk copies completing a phase through its transaction
/// count, and waits with `try_wait`, a conditional wait and `test_wait`.
///
/// Only CTA 0 touches the other CTA's mbarriers (through `mapa`): each of
/// its threads arrives on CTA 1's `from_peer`, and one thread makes CTA 1's
/// `tx_peer` phase wait for a copy CTA 1 issues, half through a standalone
/// `expect_tx` and half through an `arrive.expect_tx`. CTA 1's phases must
/// complete, and CTA 0's own `from_peer` and `tx_peer` must stay incomplete.
pub fn mbarriers_sm90() -> Case {
    const N: usize = 64;
    let kernel = Kernel::builder("mbarriers_sm90").block_dim([N as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(48));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * N, 5)));
        let t = thread_idx().x;
        let rank = block_in_cluster_idx().x;
        let row = block_idx().x * N as i32 + t;
        let bars = alloc_smem::<u64>(3, 8);
        let (bar, from_peer, tx_peer) = (bars.clone(), bars.offset(1usize), bars.offset(2usize));
        let data = alloc_smem::<u32>(48, 128);
        if_(t.eq(0), || {
            // Two arrivals from thread 1 and one from thread 32.
            mbarrier_init(&bar, 3);
            // One arrival from every thread of CTA 0.
            mbarrier_init(&from_peer, N as i32);
            // One arrival, from CTA 0's thread 2.
            mbarrier_init(&tx_peer, 1);
            mbarrier_init_fence();
        });
        // A cluster barrier publishes the inits to the other CTA. (No thread
        // index is tested for 0 again after it: LLVM would merge branches on
        // the same condition and duplicate the non-convergent inline PTX into
        // both paths.)
        inline_ptx_effect_with_sim("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;", &[], |_| vec![]);
        if_(t.eq(1), || {
            mbarrier_expect_tx(&bar, 64);
            cp_async_bulk(&src.ptr(), &data, 64, &bar);
            mbarrier_arrive_with(&bar, 2, MbarrierOptions { scope: Some(MemScopeKind::Cluster), ..Default::default() });
        });
        if_(t.eq(32), || {
            mbarrier_arrive_and_expect_tx_with(&bar, 64, MbarrierOptions { relaxed: true, ..Default::default() });
            cp_async_bulk(&src.ptr().offset(16usize), &data.offset(16usize), 64, &bar);
        });
        let cta1 = MbarrierOptions { peer_cta_rank_in_cluster: Some(1.into_value()), ..Default::default() };
        if_(rank.eq(0), || {
            mbarrier_arrive_with(&from_peer, 1, cta1);
            if_(t.eq(2), || {
                mbarrier_expect_tx_with(&tx_peer, 32, cta1);
                let cluster = MbarrierOptions { scope: Some(MemScopeKind::Cluster), ..cta1 };
                mbarrier_arrive_and_expect_tx_with(&tx_peer, 32, cluster);
            });
        });
        if_(rank.eq(1) & t.eq(3), || {
            cp_async_bulk(&src.ptr().offset(32usize), &data.offset(32usize), 64, &tx_peer);
        });
        mbarrier_wait(&bar, 0);
        if_(rank.eq(1), || {
            mbarrier_wait(&from_peer, 0);
            mbarrier_wait(&tx_peer, 0);
        });
        let landed = make_tensor(&data, Layout::col_major(48));
        out.store((row, 0), landed.load(t % 32));
        out.store((row, 1), mbarrier_test_wait(&bar, 1).to::<u32>());
        // 3 in CTA 1, whose peer phases are over; 0 in CTA 0.
        let peer_phases =
            mbarrier_test_wait(&from_peer, 0).to::<u32>() + mbarrier_try_wait(&tx_peer, 0).to::<u32>() * 2u32;
        out.store((row, 2), peer_phases);
        // Warp 0 asks the mbarrier; warp 1 does not, and gets true.
        out.store((row, 3), mbarrier_conditional_try_wait(t.lt(32), &from_peer, 0).to::<u32>());
        out.store((row, 4), if_else_yield(rank.eq(1), || landed.load(32 + t % 16), || 0u32.into_value()));
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for r in 0..2 * N {
            let (cta1, t) = (r >= N, r % N);
            let want = [
                src[t % 32],
                0,
                if cta1 { 3 } else { 0 },
                (cta1 || t >= 32) as u32,
                if cta1 { src[32 + t % 16] } else { 0 },
            ];
            assert_eq!(out[r * 5..r * 5 + 5], want, "row {r}");
        }
    }
    let cfg = LaunchConfig::new([2, 1, 1], [N as u32, 1, 1]).with_cluster([2, 1, 1]);
    Case { name: "mbarriers_sm90", kernel, cfg, small_only: false, check }
}

fn launch_n(blocks: u32, threads: usize) -> LaunchConfig {
    LaunchConfig::new([blocks, 1, 1], [threads as u32, 1, 1])
}

/// `warpsize`, the warp constants, `warp_idx` and `make_warp_uniform` on a
/// value every lane of a warp loads from the same address.
pub fn warp_helpers() -> Case {
    let kernel = Kernel::builder("warp_helpers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, 4)));
        let t = thread_idx().x;
        let w = warp_idx();
        out.store((t, 0), warpsize());
        out.store((t, 1), w * WARPS_PER_WARPGROUP as i32 + THREADS_PER_WARPGROUP as i32);
        let shared = make_warp_uniform(xs.load(w * WARP_SIZE as i32));
        out.store((t, 2), shared);
        let flag =
            if_else_yield(make_warp_uniform(w * 3 + 1).eq(4), || (FULL_MASK as i32).into_value(), || 7.into_value());
        out.store((t, 3), flag);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let w = t / 32;
            let want = [32, (w * 4 + 128) as i32, xs[w * 32], if w == 1 { -1 } else { 7 }];
            assert_eq!(out[t * 4..t * 4 + 4], want, "thread {t}");
        }
    }
    Case { name: "warp_helpers", kernel, cfg: launch(1), small_only: false, check }
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

/// Kernels that need SM90 or newer: simulated, and compiled for sm_90, but
/// not run on older GPUs.
pub fn sm90() -> Vec<Case> {
    vec![mbarriers_sm90()]
}

pub fn all() -> Vec<Case> {
    vec![
        int_i8(),
        int_i16(),
        int_i32(),
        int_i64(),
        int_u8(),
        int_u16(),
        int_u32(),
        int_u64(),
        float_f32(),
        float_f64(),
        float_f16(),
        float_bf16(),
        casts(),
        wide_multiply(),
        while_loop(),
        stack_array(),
        reinterpreted_memory(),
        runtime_sized_buffer(),
        dynamic_shared_memory(),
        warp_operations(),
        atomics(),
        printing(),
        inline_ptx_fragment(),
        mma_k8(),
        mma_f16_accumulate(),
        mma_tf32(),
        mma_f64(),
        mma_bf16(),
        mma_bf16_k8(),
        mma_f16_accumulate_k8(),
        mma_tf32_k8(),
        tiled_copy_tv_ldmatrix(),
        mma_tiled_copies(),
        named_barriers(),
        mbarriers_sm80(),
        warp_helpers(),
        math_f32(),
        math_f64(),
        math_half(),
        vars(),
        loop_hints(),
        carry_structs(),
        merged_facts(),
    ]
}

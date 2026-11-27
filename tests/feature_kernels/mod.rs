// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   cutlass_compiler/cutegen/include/cutegen/layout.hpp (the `increment`
//   reference of the `increment_coords` case),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Kernels covering every scalar type, operation and intrinsic. Each has a
//! check written in plain Rust; `tests/features.rs` holds the simulator to
//! it, and `tests/gpu.rs` holds the GPU to the simulator.

// Shared by several test binaries; each uses part of it.
#![allow(dead_code)]

mod async_sched;
mod atomics;
mod autovec;
mod bulk;
mod cluster_copies;
mod composed;
mod cpasync_order;
mod cvt_ldmatrix;
mod dsmem;
mod extras;
mod gemm_helpers;
mod masked;
mod mx;
pub mod params;
mod pipelines;
mod raw_ptr;
mod shape_arith;
mod stmatrix;
mod structs;
mod subbyte;
mod tensor_ssa;
mod tensormap;
// The TMA im2col and SM100-mode kernels.
mod tma_im2col;
mod tma_load;
mod tma_pipelines;
mod tma_sm100;
mod tma_store;
// The TMEM kernels.
mod tmem;
// The TMEM load and store kernels.
mod tmem_ldst;
mod types;
mod vector;
mod wgmma;
mod wgmma_variants;

pub use async_sched::*;
pub use atomics::*;
pub use autovec::*;
pub use bulk::*;
pub use cluster_copies::*;
pub use composed::{composed_layout_inner, pointer_swizzle};
pub use cpasync_order::*;
use cute::ir::Param;
use cute::prelude::*;
pub use cvt_ldmatrix::*;
pub use dsmem::*;
pub use extras::*;
pub use gemm_helpers::*;
pub use masked::*;
pub use mx::*;
pub use pipelines::*;
pub use raw_ptr::*;
pub use shape_arith::*;
pub use stmatrix::*;
pub use structs::*;
pub use subbyte::*;
pub use tensor_ssa::*;
pub use tensormap::*;
pub use tma_im2col::*;
pub use tma_load::*;
pub use tma_pipelines::*;
pub use tma_sm100::*;
pub use tma_store::*;
pub use tmem::*;
pub use tmem_ldst::*;
pub use types::*;
pub use vector::*;
pub use wgmma::*;
pub use wgmma_variants::*;

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

/// `3x + 1`, built here for a kernel defined in another file: `tests/mlir.rs` checks that the
/// line information of the multiplication names this file and line.
pub fn triple_plus_one(x: Float32) -> Float32 {
    let tripled = x * 3.0; // line info: helper statement
    tripled + 1.0
}

/// Compile options for `kernel` with `tc`: the MMAs whose NVVM form `tc`'s LLVM predates
/// (FP8 `mma.sync` before LLVM 23, `mma.sp` before LLVM 22) as inline PTX.
pub fn options_for(kernel: &Kernel, tc: &Toolchain) -> CompileOptions {
    let old = kernel.min_llvm_version().is_some_and(|need| tc.llvm_version().is_some_and(|have| have < need));
    // Without contraction (cute-rs's default fuses `a * b + c` into `fma`), so that the GPU
    // computes the simulator's bits: the feature kernels are compared with it exactly.
    CompileOptions::new().mma_inline_ptx(old).fp_contract(false)
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
    // Hidden parameters (a TMA atom's descriptors, the last ones) are not passed.
    let passed = ir.args.len() - ir.hidden_param_count();
    ir.args[..passed]
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
                    DType::TF32 => {
                        KernelArg::buffer(&(0..n).map(|_| TF32::from_f32(float(&mut next) as f32)).collect::<Vec<_>>())
                    }
                    // Packed views: any bytes.
                    t if t.is_packed_view() => {
                        let data = (0..n * t.bytes()).map(|_| (next() >> 32) as u8).collect();
                        KernelArg::Bytes { elem: t, data }
                    }
                    // Small values, or any encoding (NaNs included).
                    t if t.is_narrow_float() => {
                        let data = (0..n)
                            .map(|_| {
                                if small {
                                    let x = float(&mut next) as f32;
                                    narrow_bits(t, x)
                                } else {
                                    (next() >> 32) as u8
                                }
                            })
                            .collect();
                        KernelArg::Bytes { elem: t, data }
                    }
                    // Small values, or any 128 bits.
                    DType::I128 | DType::U128 => {
                        let v: Vec<u128> = (0..n)
                            .map(|_| {
                                if small {
                                    ((next() >> 40) % 17) as u128
                                } else {
                                    (next() as u128) << 64 | next() as u128
                                }
                            })
                            .collect();
                        if d.elem == DType::I128 {
                            KernelArg::buffer(&v.iter().map(|&x| x as i128).collect::<Vec<_>>())
                        } else {
                            KernelArg::buffer(&v)
                        }
                    }
                    other => panic!("no test input for a {other} buffer"),
                }
            }
            Param::Scalar { value, multiple_of, .. } => match ir.dtype(*value) {
                DType::F32 => KernelArg::ScalarF32(1.5),
                // A launch-time extent (`param_extent`): 13, rounded up to
                // what the kernel promises it is a multiple of.
                DType::I32 => KernelArg::ScalarI32(13usize.next_multiple_of(*multiple_of) as i32),
                // A 64-bit extent or stride (`param_extent_i64`), the same.
                DType::I64 => KernelArg::scalar(13usize.next_multiple_of(*multiple_of) as i64),
                // A 128-bit scalar with both words set.
                DType::I128 => KernelArg::scalar(-(3i128 << 70) - 5),
                other => panic!("no test input for a {other} scalar"),
            },
            Param::ByValue { .. } => panic!("no test input for a by-value parameter"),
        })
        .collect()
}

/// `x` encoded in the 8-bit float format `t` (rounded to nearest even,
/// saturating).
pub fn narrow_bits(t: DType, x: f32) -> u8 {
    match t {
        DType::F8E4M3FN => F8E4M3FN::from_f32(x).0,
        DType::F8E5M2 => F8E5M2::from_f32(x).0,
        DType::F8E4M3 => F8E4M3::from_f32(x).0,
        DType::F8E4M3B11FNUZ => F8E4M3B11FNUZ::from_f32(x).0,
        DType::F8E8M0FNU => F8E8M0FNU::from_f32(x).0,
        DType::F8E5M3FNU => F8E5M3FNU::from_f32(x).0,
        other => panic!("{other} is not an 8-bit float"),
    }
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

mma_case!(
    mma_fp8_e4m3_f32_k32,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)),
    F8E4M3FN,
    f32,
    0.0f32,
    (|x: F8E4M3FN| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e4m3_f16_k32,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F16, (16, 8, 32)),
    F8E4M3FN,
    F16,
    F16::ZERO,
    (|x: F8E4M3FN| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e4m3_f32_k16,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 16)),
    F8E4M3FN,
    f32,
    0.0f32,
    (|x: F8E4M3FN| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e4m3_f16_k16,
    MmaFP8Op::new(DType::F8E4M3FN, DType::F16, (16, 8, 16)),
    F8E4M3FN,
    F16,
    F16::ZERO,
    (|x: F8E4M3FN| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e5m2_f32_k32,
    MmaFP8Op::new(DType::F8E5M2, DType::F32, (16, 8, 32)),
    F8E5M2,
    f32,
    0.0f32,
    (|x: F8E5M2| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e5m2_f16_k32,
    MmaFP8Op::new(DType::F8E5M2, DType::F16, (16, 8, 32)),
    F8E5M2,
    F16,
    F16::ZERO,
    (|x: F8E5M2| x.to_f64(), |x: F16| x.to_f32() as f64)
);
mma_case!(
    mma_fp8_e5m2_f32_k16,
    MmaFP8Op::new(DType::F8E5M2, DType::F32, (16, 8, 16)),
    F8E5M2,
    f32,
    0.0f32,
    (|x: F8E5M2| x.to_f64(), |x: f32| x as f64)
);
mma_case!(
    mma_fp8_e5m2_f16_k16,
    MmaFP8Op::new(DType::F8E5M2, DType::F16, (16, 8, 16)),
    F8E5M2,
    F16,
    F16::ZERO,
    (|x: F8E5M2| x.to_f64(), |x: F16| x.to_f32() as f64)
);

/// Every FP8 `mma.sync` (sm_89): both formats, both accumulators, both
/// shapes.
pub fn mma_fp8() -> Vec<Case> {
    vec![
        mma_fp8_e4m3_f32_k32(),
        mma_fp8_e4m3_f16_k32(),
        mma_fp8_e4m3_f32_k16(),
        mma_fp8_e4m3_f16_k16(),
        mma_fp8_e5m2_f32_k32(),
        mma_fp8_e5m2_f16_k32(),
        mma_fp8_e5m2_f32_k16(),
        mma_fp8_e5m2_f16_k16(),
    ]
}

/// Metadata words whose nibbles are all valid index pairs (`0b0100`,
/// `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110`); every rotation by a
/// whole number of nibbles keeps them valid.
const SPARSE_META: u32 = 0x489C_DE49;

/// One warp, `K_TILES` sparse MMAs: A is the compressed `(M, K/2)` tile,
/// the metadata of lane `l` and tile `t` is [`SPARSE_META`] rotated left by
/// `4 * ((l + t) % 7 + 1)` bits, which the kernel writes to `meta` for the
/// check.
fn sparse_mma_kernel<TA: Numeric, TC: Numeric>(name: &'static str, op: MmaF16BF16SparseOp, zero: TC) -> Kernel {
    let (m, n, k_atom) = op.shape_mnk();
    let k = k_atom * K_TILES;
    Kernel::builder(name).block_dim([32, 1, 1]).build(|| {
        let a = param_tensor::<TA>("a", Layout::col_major((m, k / 2)));
        let b = param_tensor::<TA>("b", Layout::col_major((n, k)));
        let c = param_tensor::<TC>("c", Layout::col_major((m, n)));
        let meta = param_tensor::<u32>("meta", Layout::col_major((32, K_TILES)));
        let mma = make_tiled_mma(op, Layout::col_major((1, 1, 1)));
        let lane = lane_idx();
        let thr = mma.get_slice(lane);
        let (ta, tb, tc) = (thr.partition_a(&a), thr.partition_b(&b), thr.partition_c(&c));
        let (fa, fb) = (thr.partition_fragment_a::<TA, _>(&a), thr.partition_fragment_b::<TA, _>(&b));
        let fe = thr.partition_fragment_e(&a);
        assert_eq!(fe.size(), K_TILES);
        let acc = thr.partition_fragment_c::<TC, _>(&c);
        for i in 0..fa.size() {
            fa.store(i, ta.load(i));
        }
        for i in 0..fb.size() {
            fb.store(i, tb.load(i));
        }
        let lane_u = lane.to::<u32>();
        for t in 0..K_TILES {
            let s = ((lane_u + t as u32) % 7u32 + 1u32) * 4u32;
            let word = Uint32::from(SPARSE_META);
            let e = (word << s) | (word >> (32u32 - s));
            fe.store(t, e);
            meta.store((lane, t), e);
        }
        acc.fill(zero);
        gemm_sparse(&mma, &acc, &fa, &fe, &fb, &acc);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    })
}

/// The dense `m x k` A (column-major) of a sparse MMA kernel's compressed A
/// and metadata words (`meta[lane + 32 * tile]`), from the PTX ISA's
/// metadata layout: lane `4 * (r % 8)` (plus `chunk / 4` for m16n8k32)
/// holds the nibble of chunk `chunk % 4` of row `r` at `4 * (r / 8) + chunk
/// % 4`; the nibble's low two bits place the chunk's first kept value.
pub fn sparse_dense_a(a: &[f64], meta: &[u32], m: usize, k_atom: usize, tiles: usize) -> Vec<f64> {
    let k = k_atom * tiles;
    let mut dense = vec![0.0; m * k];
    for tile in 0..tiles {
        for r in 0..m {
            for chunk in 0..k_atom / 4 {
                let lane = 4 * (r % 8) + if k_atom == 32 { chunk / 4 } else { 0 };
                let nibble = 4 * (r / 8) + chunk % 4;
                let bits = meta[lane + 32 * tile] >> (4 * nibble) & 0xf;
                let (i0, i1) = ((bits & 3) as usize, (bits >> 2) as usize);
                assert!(i0 < i1, "invalid metadata nibble {bits:#06b}");
                let (dense_col, comp_col) = (tile * k_atom + 4 * chunk, tile * k_atom / 2 + 2 * chunk);
                dense[r + m * (dense_col + i0)] = a[r + m * comp_col];
                dense[r + m * (dense_col + i1)] = a[r + m * (comp_col + 1)];
            }
        }
    }
    dense
}

macro_rules! sparse_mma_case {
    ($name:ident, $op:expr, $ta:ty, $tc:ty, $zero:expr, $as_f64:expr) => {
        pub fn $name() -> Case {
            let kernel = sparse_mma_kernel::<$ta, $tc>(stringify!($name), $op, $zero);
            fn check(args: &[KernelArg]) {
                let as_a: fn($ta) -> f64 = $as_f64.0;
                let as_c: fn($tc) -> f64 = $as_f64.1;
                let (m, n, k_atom) = $op.shape_mnk();
                let k = k_atom * K_TILES;
                let a: Vec<f64> = args[0].to_vec::<$ta>().into_iter().map(as_a).collect();
                let (b, c, meta) = (args[1].to_vec::<$ta>(), args[2].to_vec::<$tc>(), args[3].to_vec::<u32>());
                let dense = sparse_dense_a(&a, &meta, m, k_atom, K_TILES);
                for i in 0..m {
                    for j in 0..n {
                        let want: f64 = (0..k).map(|l| dense[i + m * l] * as_a(b[j + n * l])).sum();
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

sparse_mma_case!(
    mma_sparse_f16_f32_k16,
    MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
    F16,
    f32,
    0.0f32,
    (|x: F16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f16_k16,
    MmaF16BF16SparseOp::new(DType::F16, DType::F16, (16, 8, 16), SparseMetadataFormat::Tid),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
sparse_mma_case!(
    mma_sparse_bf16_f32_k16,
    MmaF16BF16SparseOp::new(DType::BF16, DType::F32, (16, 8, 16), SparseMetadataFormat::Tid),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f32_k32,
    MmaF16BF16SparseOp::new(DType::F16, DType::F32, (16, 8, 32), SparseMetadataFormat::Tid),
    F16,
    f32,
    0.0f32,
    (|x: F16| x.to_f32() as f64, |x: f32| x as f64)
);
sparse_mma_case!(
    mma_sparse_f16_f16_k32,
    MmaF16BF16SparseOp::new(DType::F16, DType::F16, (16, 8, 32), SparseMetadataFormat::Tid),
    F16,
    F16,
    F16::ZERO,
    (|x: F16| x.to_f32() as f64, |x: F16| x.to_f32() as f64)
);
sparse_mma_case!(
    mma_sparse_bf16_f32_k32,
    MmaF16BF16SparseOp::new(DType::BF16, DType::F32, (16, 8, 32), SparseMetadataFormat::Tid),
    BF16,
    f32,
    0.0f32,
    (|x: BF16| x.to_f32() as f64, |x: f32| x as f64)
);

/// Every sparse `mma.sp` (sm_80): F16 with F16 and F32 accumulators, BF16
/// with F32, both shapes.
pub fn mma_sparse() -> Vec<Case> {
    vec![
        mma_sparse_f16_f32_k16(),
        mma_sparse_f16_f16_k16(),
        mma_sparse_bf16_f32_k16(),
        mma_sparse_f16_f32_k32(),
        mma_sparse_f16_f16_k32(),
        mma_sparse_bf16_f32_k32(),
    ]
}

/// A `2 x 1`-warp FP8 m16n8k32 MMA fed from shared memory by
/// `LdMatrix8x16x8bOp` tiled copies built with `make_tiled_copy_a` (four
/// matrices) and `make_tiled_copy_b` (two), as CuTe's Ampere GEMM loads
/// K-major FP8 operands on SM89. The fragments the copies fill must be the
/// MMA's: the product must be exact.
pub fn ldmatrix_8x16x8b_fp8() -> Case {
    const M: usize = 32;
    const N: usize = 8;
    const K: usize = 64;
    let kernel = Kernel::builder("ldmatrix_8x16x8b_fp8").block_dim([64, 1, 1]).build(|| {
        let a = param_tensor::<F8E4M3FN>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F8E4M3FN>("b", Layout::row_major((N, K)));
        let c = param_tensor::<f32>("c", Layout::row_major((M, N)));
        let frag_a = param_tensor::<F8E4M3FN>("frag_a", Layout::row_major((64, 32)));
        let t = thread_idx().x;
        let (sa, sb) = (
            make_smem_tensor::<F8E4M3FN>(Layout::row_major((M, K))),
            make_smem_tensor::<F8E4M3FN>(Layout::row_major((N, K))),
        );
        let bytes16 = make_copy_atom::<F8E4M3FN>(CopyOp::Universal { bits: 128 });
        let g2s = make_tiled_copy_tv(&bytes16, &Layout::row_major((16, 4)), &Layout::row_major((1, 16)));
        let (gs, ss) = (g2s.get_slice(t).partition_s(&a), g2s.get_slice(t).partition_d(&sa));
        copy(&g2s, &gs, &ss);
        let bytes8 = make_copy_atom::<F8E4M3FN>(CopyOp::Universal { bits: 64 });
        let g2s_b = make_tiled_copy_tv(&bytes8, &Layout::row_major((8, 8)), &Layout::row_major((1, 8)));
        copy(&g2s_b, &g2s_b.get_slice(t).partition_s(&b), &g2s_b.get_slice(t).partition_d(&sb));
        sync_threads();
        let mma = make_tiled_mma(MmaFP8Op::new(DType::F8E4M3FN, DType::F32, (16, 8, 32)), Layout::col_major((2, 1, 1)));
        let ldsm_a =
            make_tiled_copy_a(&make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 4, unpack_bits: None }), &mma);
        let ldsm_b =
            make_tiled_copy_b(&make_copy_atom::<F8E4M3FN>(CopyOp::LdMatrix8x16x8b { num: 2, unpack_bits: None }), &mma);
        let thr_mma = mma.get_slice(t);
        let (fa, fb) =
            (thr_mma.partition_fragment_a::<F8E4M3FN, _>(&sa), thr_mma.partition_fragment_b::<F8E4M3FN, _>(&sb));
        copy(&ldsm_a, &ldsm_a.get_slice(t).partition_s(&sa), &ldsm_a.retile(&fa));
        copy(&ldsm_b, &ldsm_b.get_slice(t).partition_s(&sb), &ldsm_b.retile(&fb));
        for v in 0..fa.size() {
            frag_a.store((t, v), fa.load(v));
        }
        let acc = thr_mma.partition_fragment_c::<f32, _>(&c);
        acc.fill(0.0);
        gemm(&mma, &acc, &fa, &fb, &acc);
        let tc = thr_mma.partition_c(&c);
        for i in 0..acc.size() {
            tc.store(i, acc.load(i));
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, b, c, frag_a) = (
            args[0].to_vec::<F8E4M3FN>(),
            args[1].to_vec::<F8E4M3FN>(),
            args[2].to_vec::<f32>(),
            args[3].to_vec::<F8E4M3FN>(),
        );
        // The A fragment of thread `t` (warp `w`, lane `l`) holds, for each
        // k-block `kb` of 32, a_i at row 16w + g + 8 ((i >> 2) & 1) and
        // column 32kb + 4t' + (i & 3) + 16 (i >> 3) (the PTX ISA's
        // m16n8k32 FP8 fragment, g = l / 4, t' = l % 4).
        for t in 0..64 {
            let (w, l) = (t / 32, t % 32);
            let (g, tq) = (l / 4, l % 4);
            for kb in 0..K / 32 {
                for i in 0..16 {
                    let row = 16 * w + g + 8 * ((i >> 2) & 1);
                    let col = 32 * kb + 4 * tq + (i & 3) + 16 * (i >> 3);
                    assert_eq!(frag_a[t * 32 + 16 * kb + i], a[row * K + col], "thread {t} k-block {kb} a{i}");
                }
            }
        }
        for i in 0..M {
            for j in 0..N {
                let want: f64 = (0..K).map(|l| a[i * K + l].to_f64() * b[j * K + l].to_f64()).sum();
                assert_eq!(c[i * N + j] as f64, want, "C[{i},{j}]");
            }
        }
    }
    Case {
        name: "ldmatrix_8x16x8b_fp8",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [64, 1, 1]),
        small_only: true,
        check,
    }
}

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
///
/// The spins give up after [`SPIN_LIMIT`] polls, and each thread stores how
/// many did in `gave_up`.
pub fn mbarriers_sm80() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("mbarriers_sm80").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N));
        let out = param_tensor::<u32>("out", Layout::row_major((N, 3)));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(N));
        let t = thread_idx().x;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(N));
        let misses = Var::new(Uint32::from(0u32));
        let spin = |parity: i32| {
            let missed = bounded_spin(&bar, Int32::from(parity));
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
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
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
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
        // A cluster barrier publishes the inits to the other CTA.
        cluster_sync();
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

/// The cluster barrier in all its forms (`cluster_arrive` by threads one by
/// one in divergent code, `cluster_wait`, the relaxed and aligned forms, and
/// `cluster_sync`), over three phases: each thread writes a value to global
/// memory before arriving and reads its peer CTA's after waiting, so every
/// phase must hold back each thread until the whole cluster has arrived.
/// Each row holds the block's linear index, its rank, the cluster size and
/// the values read in the three phases.
pub fn cluster_barriers_sm90(grid: [u32; 3], cluster: [u32; 3]) -> Case {
    const N: usize = 64;
    let blocks = grid.iter().product::<u32>() as usize;
    let name = match cluster {
        [2, 1, 1] => "cluster_barriers_2",
        [4, 1, 1] => "cluster_barriers_4",
        _ => "cluster_barriers_1x2x2",
    };
    let kernel = Kernel::builder(name).block_dim([N as u32, 1, 1]).build(move || {
        let staged = param_tensor::<u32>("staged", Layout::col_major(3 * blocks * N));
        let out = param_tensor::<u32>("out", Layout::row_major((blocks * N, 6)));
        let t = thread_idx().x;
        let (b, g) = (block_idx(), grid_dim());
        let linear = b.x + g.x * (b.y + g.y * b.z);
        let (rank, size) = (block_idx_in_cluster(), cluster_size());
        // Blocks of a cluster are consecutive here, ranked like the grid.
        let peer = linear - rank + (rank + 1) % size;
        let row = linear * N as i32 + t;
        let value = |phase: i32| (linear * 1000 + t + phase * 100_000).to::<u32>();
        let slot = |phase: i32, block: Int32| (phase * blocks as i32 + block) * N as i32 + t;
        out.store((row, 0), linear.to::<u32>());
        out.store((row, 1), rank.to::<u32>());
        out.store((row, 2), size.to::<u32>());
        // Phase 0: threads arrive one by one, from both sides of a branch.
        staged.store(slot(0, linear), value(0));
        if_else((t % 2).eq(0), cluster_arrive, cluster_arrive);
        cluster_wait();
        out.store((row, 3), staged.load(slot(0, peer)));
        // Phase 1: a relaxed aligned arrival orders nothing, so a fence
        // releases the write first.
        staged.store(slot(1, linear), value(1));
        fence_acq_rel_cluster();
        cluster_arrive_relaxed_aligned();
        cluster_wait_aligned();
        fence_acq_rel_cluster();
        out.store((row, 4), staged.load(slot(1, peer)));
        // Phase 2: cute::cluster_sync.
        staged.store(slot(2, linear), value(2));
        cluster_sync();
        out.store((row, 5), staged.load(slot(2, peer)));
    });
    fn check(args: &[KernelArg]) {
        let out = args[1].to_vec::<u32>();
        for (r, row) in out.chunks(6).enumerate() {
            let (t, linear, rank, size) = (r % N, row[0], row[1], row[2]);
            let peer = linear - rank + (rank + 1) % size;
            for phase in 0..3 {
                assert_eq!(row[3 + phase as usize], peer * 1000 + t as u32 + phase * 100_000, "row {r}");
            }
        }
    }
    let cfg = LaunchConfig::new(grid, [N as u32, 1, 1]).with_cluster(cluster);
    Case { name, kernel, cfg, small_only: false, check }
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

/// `increment_coord` on runtime coordinates. Every thread steps a coordinate
/// of a hierarchical shape with a size-1 mode through its carries, the
/// threads near the end past the last coordinate, and increments
/// coordinates that mix static and runtime leaves, some out of range.
pub fn increment_coords() -> Case {
    const STEPS: usize = 3;
    const LEAVES: usize = 5;
    /// The shape's size plus two, so that some threads start past the end.
    const SPAN: u32 = 122;
    const COLS: usize = LEAVES * (STEPS + 1) + 6;
    let kernel = Kernel::builder("increment_coords").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<i32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let shape = IntTuple::from(((2, 3), 1, (4, 5)));
        let i = ((t * 2).to::<u32>() + (x & 1u32)) % SPAN;
        let mut c = cute::layout::idx2crd(i.to::<i32>().as_index(), &shape);
        let mut col = 0;
        for step in 0..=STEPS {
            for leaf in c.flat_leaves() {
                out.store((t, col), Int32::from(leaf));
                col += 1;
            }
            if step < STEPS {
                c = increment_coord(c, shape.clone());
            }
        }
        // Static first leaves, one that wraps and one that does not, then
        // runtime leaves; d is out of range when it is 3 or more.
        let (d, e) = ((x % 7u32).to::<i32>(), (x % 5u32).to::<i32>());
        for first in [1usize, 0] {
            for leaf in increment_coord((first, d, e), (2, 3, 5)).flat_leaves() {
                out.store((t, col), Int32::from(leaf));
                col += 1;
            }
        }
    });
    fn check(args: &[KernelArg]) {
        /// The dialect's increment, as cutlass_compiler implements it, on
        /// flat leaves: a port of cutegen's `detail::increment_coord_impl`.
        fn increment(c: &mut [usize], shape: &[usize]) {
            let mut carry = 1;
            let last = c.len() - 1;
            for k in 0..last {
                if shape[k] == 1 {
                    c[k] = 0;
                    continue;
                }
                let sum = c[k] + carry;
                (c[k], carry) = if sum < shape[k] { (sum, 0) } else { (0, 1) };
            }
            c[last] += carry;
        }
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        for t in 0..T {
            let x = xs[t];
            let row = &out[t * COLS..(t + 1) * COLS];
            let i = (2 * t + (x & 1) as usize) % SPAN as usize;
            // Colexicographic, with the last leaf unbounded past the end.
            let mut c = [i % 2, i / 2 % 3, 0, i / 6 % 4, i / 24];
            for step in 0..=STEPS {
                assert_eq!(row[step * LEAVES..(step + 1) * LEAVES], c.map(|v| v as i32), "thread {t}, {i} + {step}");
                increment(&mut c, &[2, 3, 1, 4, 5]);
            }
            let (d, e) = ((x % 7) as usize, (x % 5) as usize);
            for (j, first) in [1, 0].into_iter().enumerate() {
                let mut m = [first, d, e];
                increment(&mut m, &[2, 3, 5]);
                let at = LEAVES * (STEPS + 1) + 3 * j;
                assert_eq!(row[at..at + 3], m.map(|v| v as i32), "thread {t}: ({first},{d},{e}) in (2,3,5)");
            }
        }
    }
    Case { name: "increment_coords", kernel, cfg: launch(1), small_only: false, check }
}

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

// References for the integer and bit instructions, written from the PTX
// ISA's definitions (bit by bit where it gives pseudo-code), independently
// of the simulator.

/// An odd multiplier that spreads test data over every bit of its type (as
/// a wrapping product, a bijection). The generated inputs of the 64-bit
/// types are below 2^48 and never negative, so the integer and bit kernels
/// multiply their data by it to reach the high bits and the sign.
const SPREAD: u64 = 0x9e37_79b9_7f4a_7c15;

/// `bfe`: the PTX ISA's pseudo-code.
pub fn bfe_ref(a: u64, b: i32, c: i32, width: u32, signed: bool) -> u64 {
    let msb = width - 1;
    let (pos, len) = (b as u32 & 0xff, c as u32 & 0xff);
    let sbit = if !signed || len == 0 { 0 } else { a >> (pos + len - 1).min(msb) & 1 };
    (0..=msb).fold(0, |d, i| d | (if i < len && pos + i <= msb { a >> (pos + i) & 1 } else { sbit }) << i)
}

/// `bfi`: the PTX ISA's pseudo-code (`a` is the field, `b` the base).
pub fn bfi_ref(a: u64, b: u64, c: i32, d: i32, width: u32) -> u64 {
    let msb = width - 1;
    let (pos, len) = (c as u32 & 0xff, d as u32 & 0xff);
    let mut f = b;
    let mut i = 0;
    while i < len && pos + i <= msb {
        f = f & !(1 << (pos + i)) | (a >> i & 1) << (pos + i);
        i += 1;
    }
    f
}

/// `lop3.b32`: each result bit looked up in the table.
pub fn lop3_ref(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    (0..32).fold(0, |d, i| {
        let index = (a >> i & 1) << 2 | (b >> i & 1) << 1 | (c >> i & 1);
        d | (lut as u32 >> index & 1) << i
    })
}

/// `shf`: the PTX ISA's formulas, where a shift by 32 gives 0.
pub fn shf_ref(a: u32, b: u32, c: u32, kind: ShfKind) -> u32 {
    let (left, clamp) = match kind {
        ShfKind::Left => (true, false),
        ShfKind::Right => (false, false),
        ShfKind::ClampLeft => (true, true),
        ShfKind::ClampRight => (false, true),
    };
    let n = if clamp { c.min(32) } else { c & 0x1f };
    let shl = |x: u32, n: u32| x.checked_shl(n).unwrap_or(0);
    let shr = |x: u32, n: u32| x.checked_shr(n).unwrap_or(0);
    if left { shl(b, n) | shr(a, 32 - n) } else { shl(b, 32 - n) | shr(a, n) }
}

/// `prmt.b32{.mode}`: the generic form from the ISA's description, the
/// modes from formulas for its table.
pub fn prmt_ref(a: u32, b: u32, c: u32, mode: PrmtMode) -> u32 {
    let byte = |k: u32| ((b as u64) << 32 | a as u64) >> (8 * k) & 0xff;
    let s = c & 3;
    let source = |i: u32| match mode {
        PrmtMode::Generic => unreachable!(),
        PrmtMode::F4e => s + i,
        PrmtMode::B4e => (s + 8 - i) % 8,
        PrmtMode::Rc8 => s,
        PrmtMode::Ecl => i.max(s),
        PrmtMode::Ecr => i.min(s),
        PrmtMode::Rc16 => (i & 1) + 2 * (s & 1),
    };
    (0..4).fold(0, |d, i| {
        let value = if mode == PrmtMode::Generic {
            let select = c >> (4 * i) & 0xf;
            let picked = byte(select & 7);
            match (select >> 3, picked >> 7) {
                (0, _) => picked,
                (_, 0) => 0,
                _ => 0xff,
            }
        } else {
            byte(source(i))
        };
        d | (value as u32) << (8 * i)
    })
}

/// The bit-field starts and lengths every thread tries on a type of
/// `width` bits, as constants: empty and full fields, fields reaching past
/// the msb or starting beyond it, fields longer than the type starting inside
/// it, and operands with bits above the low byte (which only count mod 256),
/// including a zero low byte (0x100, 0x200: a start of 0 or an empty field).
/// Constant starts and lengths are folded by `ptxas`, which on its own gets
/// some of these wrong (`bfe.s32` from beyond the msb, `bfi.b32` fields longer
/// than 31 bits).
fn bit_field_edges(width: i32) -> Vec<(i32, i32)> {
    let w = width;
    vec![
        (0, 0),
        (0, 1),
        (0, w),
        (0, 255),
        (3, 0),
        (5, 7),
        (w - 8, 8),
        (w - 4, 8),
        (w - 1, 1),
        (w - 1, 2),
        (w, 0),
        (w, 1),
        (w + 3, 0),
        (w + 8, 5),
        (200, 100),
        (255, 255),
        (1, w),
        (3, w + 8),
        (17, 100),
        (w - 2, w + 1),
        (9, 254),
        (8, w),
        (w / 2 + 3, w),
        (0x103, 0x204),
        (-1, -1),
        (0x7fff_ff07, 0x109),
        (i32::MIN, 0x7fff_ffff),
        (5, 0x100),
        (w + 8, 0x100),
        (0x100, 3),
        (0x200, w),
        (0x100, 0x100),
        (0x100 + w - 4, 8),
    ]
}

/// Constants paired with a runtime start or length in [`bit_field_case`].
const HALF_CONSTANT: [i32; 5] = [5, 40, 70, 300, 0x100];

/// Iterations of the bit-field loop in [`bit_field_case`], whose starts and
/// lengths become constants only once LLVM or `ptxas` unrolls it.
const BIT_FIELD_LOOP: i32 = 6;

/// The start, `bfe` length and `bfi` length of iteration `i` of that loop:
/// starts from 9 to 74 and `bfi` lengths from 30 to 85.
fn bit_field_loop(i: i32) -> (i32, i32, i32) {
    (i * 13 + 9, 9, i * 11 + 30)
}

/// Pairs of a runtime start and length in [`bit_field_case`]: from the data,
/// from the thread index, and two with a zero low byte on some threads.
const BIT_FIELD_RUNTIME: usize = 4;

/// The output columns of [`bit_field_case`] on a type of `width` bits.
fn bit_field_columns(width: i32) -> usize {
    2 * (BIT_FIELD_RUNTIME + bit_field_edges(width).len() + 2 * HALF_CONSTANT.len() + BIT_FIELD_LOOP as usize)
}

/// `bfe` and `bfi` on one type, on data spread over every bit (so fields
/// past the msb fill with a set sign bit as well as a clear one, and `bfi`
/// writes into values with their high bits set): starts and lengths from
/// the input data (any 32-bit value, of which only the low byte counts),
/// from the thread index (fields inside, across and beyond the type; and
/// starts or lengths of 0x100 to 0x700, whose low byte is 0, on every eighth
/// thread), the constant edge cases of [`bit_field_edges`], a constant start
/// or length with the other from the data, and a runtime loop whose starts
/// and lengths are constants once unrolled.
macro_rules! bit_field_case {
    ($name:ident, $t:ty, $u:ty) => {
        pub fn $name() -> Case {
            const W: i32 = <$t>::BITS as i32;
            let cols = bit_field_columns(W);
            let kernel =
                Kernel::builder(concat!("bit_fields_", stringify!($t))).block_dim([T as u32, 1, 1]).build(|| {
                    let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                    let ys = param_tensor::<$t>("ys", Layout::col_major(T));
                    let ps = param_tensor::<i32>("ps", Layout::col_major(T));
                    let ls = param_tensor::<i32>("ls", Layout::col_major(T));
                    let out = param_tensor::<$t>("out", Layout::row_major((T, cols)));
                    let t = thread_idx().x;
                    let (x, y) = (xs.load(t) * SPREAD as $t, ys.load(t) * SPREAD as $t);
                    let (p, l) = (ps.load(t), ls.load(t));
                    let (tp, tl) = (t * 5 % (2 * W + 6), t * 11 % (2 * W + 6));
                    // As in the check: a zero low byte when t % 8 == 0 and
                    // when t < 8 respectively.
                    let (zp, zl) = (((t >> 3) << 8) | ((t & 7) * 9), ((t & 7) << 8) | ((t >> 3) * 9));
                    let mut results = vec![bfe(x, p, l), bfi(y, x, p, l), bfe(x, tp, tl), bfi(y, x, tp, tl)];
                    results.extend([bfe(x, zp, tl), bfi(y, x, zp, tl), bfe(x, tp, zl), bfi(y, x, tp, zl)]);
                    for (start, length) in bit_field_edges(W) {
                        results.push(bfe(x, start, length));
                        results.push(bfi(y, x, start, length));
                    }
                    for c in HALF_CONSTANT {
                        results.extend([bfe(x, c, l), bfi(y, x, c, l), bfe(x, p, c), bfi(y, x, p, c)]);
                    }
                    let looped = results.len();
                    for (j, v) in results.into_iter().enumerate() {
                        out.store((t, j), v);
                    }
                    for_range(0, BIT_FIELD_LOOP, 1, |i| {
                        // As in bit_field_loop.
                        let (start, length_e, length_i) = (i * 13 + 9, 9, i * 11 + 30);
                        let column = i * 2 + looped as i32;
                        out.store((t, column), bfe(x, start, length_e));
                        out.store((t, column + 1), bfi(y, x, start, length_i));
                    });
                    assert_eq!(looped + 2 * BIT_FIELD_LOOP as usize, cols);
                });
            fn check(args: &[KernelArg]) {
                let (xs, ys) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>());
                let (ps, ls, out) = (args[2].to_vec::<i32>(), args[3].to_vec::<i32>(), args[4].to_vec::<$t>());
                let signed = <$t>::MIN != 0;
                let (w, cols) = (W as u32, bit_field_columns(W));
                let spread = |v: $t| v.wrapping_mul(SPREAD as $t) as $u as u64;
                let msb_set = (0..T).filter(|&t| spread(xs[t]) >> (w - 1) == 1).count();
                assert!(msb_set > 0 && msb_set < T, "{}: {msb_set} values with the msb set", stringify!($t));
                for t in 0..T {
                    let (x, y) = (spread(xs[t]), spread(ys[t]));
                    let ti = t as i32;
                    // (start, bfe length, bfi length) for each pair of columns.
                    let mut fields = vec![(ps[t], ls[t], ls[t])];
                    let (tp, tl) = (ti * 5 % (2 * W + 6), ti * 11 % (2 * W + 6));
                    let (zp, zl) = (((ti >> 3) << 8) | ((ti & 7) * 9), ((ti & 7) << 8) | ((ti >> 3) * 9));
                    fields.extend([(tp, tl, tl), (zp, tl, tl), (tp, zl, zl)]);
                    fields.extend(bit_field_edges(W).into_iter().map(|(p, l)| (p, l, l)));
                    for c in HALF_CONSTANT {
                        fields.extend([(c, ls[t], ls[t]), (ps[t], c, c)]);
                    }
                    fields.extend((0..BIT_FIELD_LOOP).map(bit_field_loop));
                    assert_eq!(2 * fields.len(), cols);
                    for (k, (p, le, li)) in fields.into_iter().enumerate() {
                        let got_e = out[t * cols + 2 * k] as $u as u64;
                        let got_i = out[t * cols + 2 * k + 1] as $u as u64;
                        let what = format!("{}: x = {x:#x}, start {p}", stringify!($t));
                        assert_eq!(got_e, bfe_ref(x, p, le, w, signed), "bfe, {what}, length {le}");
                        assert_eq!(got_i, bfi_ref(y, x, p, li, w), "bfi, {what}, length {li}, replacement {y:#x}");
                    }
                }
            }
            Case { name: concat!("bit_fields_", stringify!($t)), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

bit_field_case!(bit_fields_i32, i32, u32);
bit_field_case!(bit_fields_u32, u32, u32);
bit_field_case!(bit_fields_i64, i64, u64);
bit_field_case!(bit_fields_u64, u64, u64);

/// The `lop3` tables every thread applies: constants, the PTX ISA's
/// examples, xor, majority and select.
const LOP3_TABLES: [u8; 9] = [0x00, 0xff, 0x80, 0xfe, 0x40, 0x1a, 0x96, 0xe8, 0xca];

/// Constant funnel-shift amounts: around 0 and 32, and amounts that wrap
/// and clamp differently (`-1` is 2^32 - 1 unsigned).
const SHIFTS: [i32; 8] = [0, 1, 13, 31, 32, 33, 64, -1];

/// Constant `prmt` selectors: identity, the high word, a byte reversal, an
/// interleave, and sign replication from each byte.
const SELECTORS: [u32; 8] = [0x3210, 0x7654, 0x0123, 0x5140, 0x8888, 0xba98, 0xfedc, 0x7f80];

const SHF_KINDS: [ShfKind; 4] = [ShfKind::Left, ShfKind::Right, ShfKind::ClampLeft, ShfKind::ClampRight];

const PRMT_MODES: [PrmtMode; 6] =
    [PrmtMode::F4e, PrmtMode::B4e, PrmtMode::Rc8, PrmtMode::Ecl, PrmtMode::Ecr, PrmtMode::Rc16];

/// `lop3`, `shf` (rotates included) and `prmt` (every mode) on Uint32 and
/// Int32, with runtime and constant tables, shift amounts and selectors.
pub fn logic_and_shifts() -> Case {
    const COLS: usize = 9 + 1 + 4 + 4 * 8 + 3 + 1 + 8 + 1 + 6 + 6;
    let kernel = Kernel::builder("logic_and_shifts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let zs = param_tensor::<u32>("zs", Layout::col_major(T));
        let ss = param_tensor::<i32>("ss", Layout::col_major(T));
        let ws = param_tensor::<i32>("ws", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y, z, s, w) = (xs.load(t), ys.load(t), zs.load(t), ss.load(t), ws.load(t));
        let mut results: Vec<Uint32> = LOP3_TABLES.iter().map(|&lut| lop3(x, y, z, lut)).collect();
        results.push(lop3(w, w ^ 0x5a5a_5a5a, s, 0xd8).bitcast());
        results.extend(SHF_KINDS.map(|kind| shf(x, y, s, kind)));
        for kind in SHF_KINDS {
            results.extend(SHIFTS.map(|n| shf(x, y, n, kind)));
        }
        results.push(shf(x, x, s, ShfKind::Left));
        results.push(shf(x, x, s, ShfKind::Right));
        results.push(shf(w, s, w, ShfKind::ClampRight).bitcast());
        results.push(prmt(x, y, z));
        results.extend(SELECTORS.map(|c| prmt(x, y, c)));
        results.push(prmt(w, s, w).bitcast());
        results.extend(PRMT_MODES.map(|mode| prmt_with(x, y, z, mode)));
        results.extend(PRMT_MODES.map(|mode| prmt_with(x, y, 3u32, mode)));
        assert_eq!(results.len(), COLS);
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, zs) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        let (ss, ws, out) = (args[3].to_vec::<i32>(), args[4].to_vec::<i32>(), args[5].to_vec::<u32>());
        for t in 0..T {
            let (x, y, z, s, w) = (xs[t], ys[t], zs[t], ss[t] as u32, ws[t] as u32);
            let mut want: Vec<u32> = LOP3_TABLES.iter().map(|&lut| lop3_ref(x, y, z, lut)).collect();
            want.push(lop3_ref(w, w ^ 0x5a5a_5a5a, s, 0xd8));
            want.extend(SHF_KINDS.map(|kind| shf_ref(x, y, s, kind)));
            for kind in SHF_KINDS {
                want.extend(SHIFTS.map(|n| shf_ref(x, y, n as u32, kind)));
            }
            want.push(x.rotate_left(s % 32));
            want.push(x.rotate_right(s % 32));
            want.push(shf_ref(w, s, w, ShfKind::ClampRight));
            want.push(prmt_ref(x, y, z, PrmtMode::Generic));
            want.extend(SELECTORS.map(|c| prmt_ref(x, y, c, PrmtMode::Generic)));
            want.push(prmt_ref(w, s, w, PrmtMode::Generic));
            want.extend(PRMT_MODES.map(|mode| prmt_ref(x, y, z, mode)));
            want.extend(PRMT_MODES.map(|mode| prmt_ref(x, y, 3, mode)));
            for (j, w) in want.into_iter().enumerate() {
                let got = out[t * COLS + j];
                assert_eq!(got, w, "column {j}, thread {t}: x = {x:#x}, y = {y:#x}, z = {z:#x}, s = {s:#x}");
            }
        }
    }
    Case { name: "logic_and_shifts", kernel, cfg: launch(1), small_only: false, check }
}

/// Constant operands of the 64-bit `mul_hi` in [`int_multiplies`], each
/// multiplied by the data: negative values (with high bits set and clear),
/// the extremes and -1; and unsigned values with bit 63 set.
const MUL_HI_I64: [i64; 5] = [i64::MIN, i64::MIN + 3, -0x1234_5678_9abc_def0, -1, i64::MAX];
const MUL_HI_U64: [u64; 3] = [u64::MAX, u64::MAX - 5, 0x8000_0000_0000_0001];

/// `mul_hi` on the four 32- and 64-bit types (the 64-bit data spread over
/// all 64 bits, so negative and with all 128 bits of the product in use, and
/// times negative and high-bit constants), `mul_wide` on the four 16- and
/// 32-bit types, `mul24` and `mad24` (both halves, both signednesses, with
/// garbage in bits 31:24 and bit 23 set at random) and the saturating
/// add and subtract, on the input data and on edge constants.
pub fn int_multiplies() -> Case {
    const N64: usize = 4 + MUL_HI_I64.len() + MUL_HI_U64.len() + 2;
    const N32: usize = 2 + 4 * 2 + 2 + 8 + 6;
    let kernel = Kernel::builder("int_multiplies").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let ls = param_tensor::<i64>("ls", Layout::col_major(T));
        let ms = param_tensor::<i64>("ms", Layout::col_major(T));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let t = thread_idx().x;
        let (x, y, z) = (xs.load(t), ys.load(t), zs.load(t));
        let (l, m) = (ls.load(t) * SPREAD as i64, ms.load(t) * SPREAD as i64);
        let (ux, uy, uz) = (x.bitcast::<u32>(), y.bitcast::<u32>(), z.bitcast::<u32>());
        let (ul, um) = (l.bitcast::<u64>(), m.bitcast::<u64>());
        let mut wide = vec![mul_hi(l, m).bitcast::<u64>(), mul_hi(ul, um), mul_wide(x, y).bitcast(), mul_wide(ux, uy)];
        wide.extend(MUL_HI_I64.map(|c| mul_hi(l, c).bitcast::<u64>()));
        wide.extend(MUL_HI_U64.map(|c| mul_hi(ul, c)));
        wide.push(mul_hi(Value::constant(i64::MIN + 3), -0x1234_5678_9abc_def0i64).bitcast());
        wide.push(mul_hi(Value::constant(u64::MAX - 5), 0x8000_0000_0000_0001u64));
        assert_eq!(wide.len(), N64);
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
        let (hx, hy) = (x.to::<i16>(), y.to::<i16>());
        let mut narrow = vec![mul_hi(x, y).bitcast::<u32>(), mul_hi(ux, uy)];
        for hi in [false, true] {
            narrow.push(mul24(x, y, hi).bitcast());
            narrow.push(mul24(ux, uy, hi));
            narrow.push(mad24(x, y, z, hi).bitcast());
            narrow.push(mad24(ux, uy, uz, hi));
        }
        narrow.push(mul_wide(hx, hy).bitcast());
        narrow.push(mul_wide(hx.bitcast::<u16>(), hy.bitcast::<u16>()));
        for c in [i32::MAX, i32::MIN, -1, 1] {
            narrow.push(add_sat_int(x, c).bitcast());
            narrow.push(sub_sat_int(x, c).bitcast());
        }
        narrow.push(add_sat_int(x, y).bitcast());
        narrow.push(sub_sat_int(x, y).bitcast());
        // The sign extension from bit 23 and the 47:16 slice, on constants.
        narrow.push(mul24(0x00ff_ffffu32, 0x00ff_ffffu32, true));
        narrow.push(mul24(0x0080_0000i32, x, true).bitcast());
        narrow.push(mul24(0x0080_0000i32, x, false).bitcast());
        narrow.push(mul24(-0x0080_0000i32, -1, false).bitcast());
        assert_eq!(narrow.len(), N32);
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let (ls, ms) = (args[3].to_vec::<i64>(), args[4].to_vec::<i64>());
        let (out64, out32) = (args[5].to_vec::<u64>(), args[6].to_vec::<u32>());
        // 24-bit operands, sign-extended from bit 23 or not, and the
        // product's slices.
        let s24 = |v: i32| ((v as u32 & 0xff_ffff) ^ 0x80_0000) as i64 - 0x80_0000;
        let u24 = |v: i32| (v as u32 & 0xff_ffff) as i64;
        let slice = |p: i64, hi: bool| if hi { (p >> 16) as u32 } else { p as u32 };
        // The high halves of 128-bit products.
        let hi_s = |a: i64, b: i64| ((a as i128 * b as i128) >> 64) as u64;
        let hi_u = |a: u64, b: u64| ((a as u128 * b as u128) >> 64) as u64;
        let spread = |v: i64| v.wrapping_mul(SPREAD as i64);
        // The 64-bit operands come in every combination of signs.
        let signs: std::collections::HashSet<_> = (0..T).map(|t| (spread(ls[t]) < 0, spread(ms[t]) < 0)).collect();
        assert_eq!(signs.len(), 4, "64-bit mul_hi operands lack a combination of signs: {signs:?}");
        for t in 0..T {
            let (x, y, z, l, m) = (xs[t], ys[t], zs[t], spread(ls[t]), spread(ms[t]));
            let mut want64 = vec![
                hi_s(l, m),
                hi_u(l as u64, m as u64),
                (x as i64 * y as i64) as u64,
                x as u32 as u64 * y as u32 as u64,
            ];
            want64.extend(MUL_HI_I64.map(|c| hi_s(l, c)));
            want64.extend(MUL_HI_U64.map(|c| hi_u(l as u64, c)));
            want64.extend([hi_s(i64::MIN + 3, -0x1234_5678_9abc_def0), hi_u(u64::MAX - 5, 0x8000_0000_0000_0001)]);
            assert_eq!(out64[t * N64..(t + 1) * N64], want64[..], "64-bit products, thread {t}: {l:#x} {m:#x} {x} {y}");
            let mut want =
                vec![((x as i64 * y as i64) >> 32) as u32, ((x as u32 as u64 * y as u32 as u64) >> 32) as u32];
            for hi in [false, true] {
                want.push(slice(s24(x) * s24(y), hi));
                want.push(slice(u24(x) * u24(y), hi));
                want.push(slice(s24(x) * s24(y), hi).wrapping_add(z as u32));
                want.push(slice(u24(x) * u24(y), hi).wrapping_add(z as u32));
            }
            let (hx, hy) = (x as i16, y as i16);
            want.push((hx as i32 * hy as i32) as u32);
            want.push(hx as u16 as u32 * hy as u16 as u32);
            let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32;
            for c in [i32::MAX, i32::MIN, -1, 1] {
                want.push(clamp(x as i64 + c as i64));
                want.push(clamp(x as i64 - c as i64));
            }
            want.push(clamp(x as i64 + y as i64));
            want.push(clamp(x as i64 - y as i64));
            want.push(0xffff_fe00);
            want.push(slice(-0x80_0000 * s24(x), true));
            want.push(slice(-0x80_0000 * s24(x), false));
            want.push(slice(-0x80_0000 * -1, false));
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "32-bit products, thread {t}: {x:#x} {y:#x} {z:#x}");
        }
    }
    Case { name: "int_multiplies", kernel, cfg: launch(1), small_only: false, check }
}

/// `bfind` from the PTX ISA's pseudo-code: the highest bit of `v` (a
/// `width`-bit value) unlike the sign bit when `signed`, else the highest 1,
/// or 0xFFFFFFFF.
pub fn bfind_ref(v: u64, width: u32, signed: bool) -> u32 {
    let msb = width - 1;
    let v = if signed && v >> msb & 1 == 1 { !v } else { v };
    (0..=msb).rev().find(|&i| v >> i & 1 == 1).unwrap_or(u32::MAX)
}

/// Constant operands of [`bit_counts`]: zero, one, all ones, and the
/// extremes of the signed types.
const BIT_COUNT_CONSTANTS: [u64; 5] = [0, 1, u64::MAX, 0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff];

/// CuTe DSL's names for the bit counts (`clz`, `brev`, `popc`, `bfind`,
/// `log2_of_pow2_int`) on the four 32- and 64-bit types, on the input data
/// (the 64-bit data spread over all 64 bits on odd threads) and on
/// constants, and `globaltimer_lo` (whose two reads must be close together
/// and agree with `globaltimer`).
pub fn bit_counts() -> Case {
    const N32: usize = 12 + 4 + 10 * BIT_COUNT_CONSTANTS.len() + 3 + 2;
    const N64: usize = 4 + 4 * BIT_COUNT_CONSTANTS.len();
    let kernel = Kernel::builder("bit_counts").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let (x, l) = (xs.load(t), ls.load(t));
        let l = select((t & 1).eq(1), l * SPREAD, l);
        let (sx, sl) = (x.bitcast::<i32>(), l.bitcast::<i64>());
        let mut narrow: Vec<Uint32> = vec![
            clz(x),
            clz(sx),
            clz(l),
            clz(sl),
            brev(x),
            brev(sx).bitcast(),
            popc(x),
            popc(sx).bitcast(),
            bfind(x),
            bfind(sx),
            bfind(l),
            bfind(sl),
        ];
        narrow.extend([popc(x.to::<u8>()).to(), popc(sx.to::<i16>()).to(), bfind(x >> 7u32), bfind(sx >> 9i32)]);
        for c in BIT_COUNT_CONSTANTS {
            narrow.extend([clz(c as u32), clz(c as i32), bfind(c as u32), bfind(c as i32), brev(c as u32)]);
            narrow.extend([clz(c), clz(c as i64), bfind(c), bfind(c as i64), popc(c as i32).bitcast()]);
        }
        let power = Value::constant(1i32) << (sx & 31);
        narrow.extend([log2_of_pow2_int(sx), log2_of_pow2_int(power), log2_of_pow2_int(0)].map(|v| v.bitcast()));
        // Two reads of the low timer word, and the whole timer between them:
        // within a second of each other, in whichever order they ran.
        let before = globaltimer_lo().bitcast::<u32>();
        let whole = globaltimer();
        let after = globaltimer_lo().bitcast::<u32>();
        let second = 1_000_000_000u32;
        let close = |a: Uint32, b: Uint32| select((b - a).lt(second) | (a - b).lt(second), 1u32, 0u32);
        narrow.extend([close(before, after), close(before, whole.to::<u32>())]);
        assert_eq!(narrow.len(), N32);
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
        let mut wide = vec![brev(l), brev(sl).bitcast(), popc(l), popc(sl).bitcast()];
        for c in BIT_COUNT_CONSTANTS {
            wide.extend([brev(c), brev(c as i64).bitcast(), popc(c), popc(c as i64).bitcast()]);
        }
        assert_eq!(wide.len(), N64);
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>());
        let (out32, out64) = (args[2].to_vec::<u32>(), args[3].to_vec::<u64>());
        for t in 0..T {
            let (x, l) = (xs[t], if t % 2 == 1 { ls[t].wrapping_mul(SPREAD) } else { ls[t] });
            let mut want = vec![
                x.leading_zeros(),
                x.leading_zeros(),
                l.leading_zeros(),
                l.leading_zeros(),
                x.reverse_bits(),
                x.reverse_bits(),
                x.count_ones(),
                x.count_ones(),
                bfind_ref(x as u64, 32, false),
                bfind_ref(x as u64, 32, true),
                bfind_ref(l, 64, false),
                bfind_ref(l, 64, true),
            ];
            want.extend([
                (x as u8).count_ones(),
                (x as u16).count_ones(),
                bfind_ref((x >> 7) as u64, 32, false),
                bfind_ref(((x as i32) >> 9) as u32 as u64, 32, true),
            ]);
            for c in BIT_COUNT_CONSTANTS {
                let c32 = c as u32 as u64;
                let (zeros32, zeros64) = ((c as u32).leading_zeros(), c.leading_zeros());
                want.extend([zeros32, zeros32, bfind_ref(c32, 32, false), bfind_ref(c32, 32, true)]);
                want.push((c as u32).reverse_bits());
                want.extend([zeros64, zeros64, bfind_ref(c, 64, false), bfind_ref(c, 64, true)]);
                want.push((c as u32).count_ones());
            }
            let log2 = |a: u32| if a == 0 { u32::MAX } else { a.trailing_zeros() };
            want.extend([log2(x), x & 31, u32::MAX]);
            want.extend([1, 1]);
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}: x = {x:#x}, l = {l:#x}");
            let mut wide = vec![l.reverse_bits(), l.reverse_bits(), l.count_ones() as u64, l.count_ones() as u64];
            for c in BIT_COUNT_CONSTANTS {
                wide.extend([c.reverse_bits(), c.reverse_bits(), c.count_ones() as u64, c.count_ones() as u64]);
            }
            assert_eq!(out64[t * N64..(t + 1) * N64], wide[..], "thread {t}: l = {l:#x}");
        }
    }
    Case { name: "bit_counts", kernel, cfg: launch(1), small_only: false, check }
}

/// Kernels that need SM90 or newer: simulated, and compiled for sm_90, but
/// not run on older GPUs.
pub fn sm90() -> Vec<Case> {
    vec![
        mbarriers_sm90(),
        cluster_barriers_sm90([4, 1, 1], [2, 1, 1]),
        cluster_barriers_sm90([4, 1, 1], [4, 1, 1]),
        cluster_barriers_sm90([1, 2, 2], [1, 2, 2]),
        bulk_copy_producer_warp(),
        half_sm90(),
        cvt_i8_bf16_sm90(),
        cluster_ranks(),
        cluster_ranks_2x2(),
        carry_around_smem_registers_sm90(),
        cluster_scope_handoff(),
        atomics_sm90(),
        consumer_first_try_wait(),
        pointer_cluster_scope(),
        stmatrix_forms(),
        stmatrix_epilogue(),
    ]
    .into_iter()
    .chain(bulk_cases_sm90())
    .chain(pipeline_cases_sm90())
    .chain(async_sched_cases_sm90())
    .chain(tma_load_cases_sm90())
    .chain(tma_store_cases_sm90())
    .chain(tma_im2col_cases_sm90())
    .chain(cluster_copy_cases_sm90())
    // Distributed shared memory.
    .chain(dsmem_cases_sm90())
    // The GEMM host helper cases.
    .chain(gemm_helper_cases_sm90())
    // The TMA pipelines.
    .chain(tma_pipeline_cases_sm90())
    .collect()
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
        increment_coords(),
        dynamic_layouts(),
        basis_layouts(),
        math_f32(),
        math_f64(),
        math_half(),
        vars(),
        loop_hints(),
        carry_structs(),
        merged_facts(),
        bit_fields_i32(),
        bit_fields_u32(),
        bit_fields_i64(),
        bit_fields_u64(),
        logic_and_shifts(),
        int_multiplies(),
        bit_counts(),
        carry_words_u32(),
        carry_words_u64(),
        carry_multiply(),
        carry_in_loop(),
        inline_ptx_named(),
        inline_asm_raw(),
        mlir_ops(),
        carry_around_division(),
        carry_around_collectives(),
        dynamic_tiles(),
        dynamic_region(),
        cp_async_mbarrier(),
        cp_async_consumer_first(),
        cp_async_pipeline(),
        carry_around_smem_registers(),
        ordered_handoffs(),
        cache_hints(),
        copy_atom_cache_hints(),
        prefetches(),
        global_copy_atoms(),
        cotiled_copies(),
        composed_layout_inner(),
        pointer_swizzle(),
        strong_vectors(),
        carry_around_memory_accesses(),
        assume_vector_copy(),
        int64_strides(),
        fast_divmod(),
        atomic_orders(),
        atomic_lock(),
        atomic_float_adds(),
        atomic_float_extrema(),
        reductions(),
        consumer_first_named_barrier(),
        consumer_first_mbarrier(),
        mbarrier_ring(),
        shape_arith(),
        coord_tensor_copy(),
        boolean_conversions(),
        tf32_conversions(),
        mma_tf32_fragments(),
        mixed_promotion(),
        int128_ops(),
        int4_conversions(),
        int4_fast_conversions(),
    ]
    .into_iter()
    .chain(warp_collectives())
    .chain(packed_half_and_conversions())
    .chain(mma_sparse())
    .chain(subbyte_tensors())
    .chain([mx_storage()])
    .chain(autovec_cases())
    .chain([fence_proxy_alias()])
    .chain(tensor_ssa_cases())
    .chain(vector_cases())
    .chain(masked_cases())
    .chain(extras_cases())
    .chain(raw_ptr_cases())
    .chain(struct_cases())
    .chain(pipeline_cases())
    .collect()
}

/// The lane a `shfl.sync` executed by lane `lane` reads, from its operand
/// `b` and its packed segment mask and clamp `c`, as the PTX ISA's
/// pseudo-code computes it; `None` when the lane keeps its own value.
pub fn shfl_ref(kind: ShuffleKind, lane: u32, b: u32, c: u32) -> Option<u32> {
    let (bval, cval, segmask) = (b & 31, c & 31, c >> 8 & 31);
    let max_lane = (lane & segmask) | (cval & !segmask);
    let min_lane = lane & segmask;
    match kind {
        ShuffleKind::Up => (lane >= bval && lane - bval >= max_lane).then(|| lane - bval),
        ShuffleKind::Down => (lane + bval <= max_lane).then(|| lane + bval),
        ShuffleKind::Bfly => (lane ^ bval <= max_lane).then_some(lane ^ bval),
        ShuffleKind::Idx => Some(min_lane | (bval & !segmask)).filter(|&j| j <= max_lane),
    }
}

/// The shuffles every thread of the shuffle kernels does over the whole
/// warp: the kind, the offset (`None`: from the data, any 32-bit value, of
/// which bits 4:0 count) and `mask_and_clamp`. The whole warp (31, and 0
/// for up), segments of 8 and 16 lanes, a clamp below the segment's top,
/// up and down past the ends (lanes keep their own value), an offset above
/// 31, up with the clamp of the other kinds (every lane keeps its own, as
/// CuTe DSL notes), and butterflies across segments, which the PTX ISA
/// allows when the lane read is at most `maxLane`.
pub const SHUFFLES: [(ShuffleKind, Option<i32>, i32); 23] = [
    (ShuffleKind::Idx, Some(3), 31),
    (ShuffleKind::Idx, None, 31),
    (ShuffleKind::Idx, Some(5), 0x1807),
    (ShuffleKind::Idx, Some(13), 0x1807),
    (ShuffleKind::Idx, Some(13), 0x100f),
    (ShuffleKind::Idx, Some(6), 0x1803),
    (ShuffleKind::Idx, None, 0x1807),
    (ShuffleKind::Up, Some(2), 0),
    (ShuffleKind::Up, Some(5), 0),
    (ShuffleKind::Up, None, 0),
    (ShuffleKind::Up, Some(3), 0x1800),
    (ShuffleKind::Up, Some(6), 0x1000),
    (ShuffleKind::Up, Some(1), 31),
    (ShuffleKind::Down, Some(5), 31),
    (ShuffleKind::Down, None, 31),
    (ShuffleKind::Down, Some(3), 0x1807),
    (ShuffleKind::Down, Some(6), 0x100f),
    (ShuffleKind::Down, Some(33), 31),
    (ShuffleKind::Bfly, Some(1), 31),
    (ShuffleKind::Bfly, Some(16), 31),
    (ShuffleKind::Bfly, None, 31),
    (ShuffleKind::Bfly, Some(4), 0x1807),
    (ShuffleKind::Bfly, Some(8), 0x1807),
];

/// The output columns of the shuffle kernels: [`SHUFFLES`], then three
/// shuffles with member masks.
const SHUFFLE_COLUMNS: usize = SHUFFLES.len() + 3;

/// Shuffles of one type: every entry of [`SHUFFLES`], then shuffles with
/// member masks (the two half-warps in divergent branches, each with its
/// own mask; per-lane half-warp masks in converged code; per-lane masks of
/// 8-lane segments with a segmented up), on the loaded values with lane 7
/// of each warp holding `special` (for the float types a NaN with a
/// payload, which must arrive unchanged; for the integers a value with the
/// sign bit set), and a Boolean butterfly.
macro_rules! shuffle_case {
    ($name:ident, $t:ty, $special:expr) => {
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("xs", Layout::col_major(T));
                let bs = param_tensor::<i32>("bs", Layout::col_major(T));
                let out = param_tensor::<$t>("out", Layout::row_major((T, SHUFFLE_COLUMNS)));
                let bools = param_tensor::<u32>("bools", Layout::col_major(T));
                let t = thread_idx().x;
                let lane = lane_idx();
                let special: $t = $special;
                let v = select(lane.eq(7), special, xs.load(t));
                let b = bs.load(t);
                let mut results = Vec::new();
                for (kind, offset, clamp) in SHUFFLES {
                    let offset = offset.map_or(b, Int32::from);
                    results.push(shuffle_sync_op(v, offset, FULL_MASK, clamp, kind));
                }
                results.push(if_else_yield(
                    lane.lt(16),
                    || shuffle_sync_bfly_with(v, 1, 0x0000_ffffu32, 31),
                    || shuffle_sync_down_with(v, 2, 0xffff_0000u32, 31),
                ));
                let halves = select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32);
                results.push(shuffle_sync_with(v, lane ^ 5, halves, 31));
                let segment = Value::constant(0xffu32) << (lane & !7).bitcast::<u32>();
                results.push(shuffle_sync_up_with(v, 1, segment, 0x1800));
                assert_eq!(results.len(), SHUFFLE_COLUMNS);
                for (j, r) in results.into_iter().enumerate() {
                    out.store((t, j), r);
                }
                let flag = shuffle_sync_bfly((t % 3).eq(0), 3);
                bools.store(t, select(flag, 1u32, 0u32));
            });
            fn check(args: &[KernelArg]) {
                let (xs, bs) = (args[0].to_vec::<$t>(), args[1].to_vec::<i32>());
                let (out, bools) = (args[2].to_vec::<$t>(), args[3].to_vec::<u32>());
                let special: $t = $special;
                let value = |t: usize| if t % 32 == 7 { special } else { xs[t] };
                for t in 0..T {
                    let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
                    let read = |source: Option<u32>| value(source.map_or(t, |s| warp + s as usize)).raw_bits();
                    let mut want: Vec<u64> = SHUFFLES
                        .iter()
                        .map(|&(kind, offset, clamp)| {
                            read(shfl_ref(kind, lane, offset.unwrap_or(bs[t]) as u32, clamp as u32))
                        })
                        .collect();
                    want.push(read(if lane < 16 {
                        shfl_ref(ShuffleKind::Bfly, lane, 1, 31)
                    } else {
                        shfl_ref(ShuffleKind::Down, lane, 2, 31)
                    }));
                    want.push(read(shfl_ref(ShuffleKind::Idx, lane, lane ^ 5, 31)));
                    want.push(read(shfl_ref(ShuffleKind::Up, lane, 1, 0x1800)));
                    let got: Vec<u64> =
                        out[t * SHUFFLE_COLUMNS..(t + 1) * SHUFFLE_COLUMNS].iter().map(|v| v.raw_bits()).collect();
                    assert_eq!(got, want, "{} thread {t}, offset {:#x}", stringify!($t), bs[t]);
                    assert_eq!(bools[t], ((warp + (lane ^ 3) as usize) % 3 == 0) as u32, "Boolean, thread {t}");
                }
            }
            Case { name: stringify!($name), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

shuffle_case!(shuffles_i8, i8, i8::MIN);
shuffle_case!(shuffles_u16, u16, u16::MAX - 1);
shuffle_case!(shuffles_f16, F16, F16(0x7e05));
shuffle_case!(shuffles_bf16, BF16, BF16(0xffc3));
shuffle_case!(shuffles_i32, i32, i32::MIN + 1);
shuffle_case!(shuffles_f32, f32, f32::from_bits(0x7fc0_0123));
shuffle_case!(shuffles_i64, i64, i64::MIN + 5);
shuffle_case!(shuffles_f64, f64, f64::from_bits(0xfff8_0000_0000_4567));

/// The group sizes of [`warp_reductions`]: every power of two up to the warp.
const GROUPS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// `op` applied as CuTe DSL's butterfly applies it: each lane combines its
/// value with that of the lane `offset` away, for offsets from half the
/// group down to 1.
pub fn butterfly_ref<V: Copy>(values: &[V], group: usize, op: impl Fn(V, V) -> V) -> Vec<V> {
    let mut v = values.to_vec();
    let mut offset = group / 2;
    while offset > 0 {
        v = (0..v.len()).map(|l| op(v[l], v[l ^ offset])).collect();
        offset /= 2;
    }
    v
}

/// PTX's `max` on float values: a NaN is ignored, and `+0 > -0`.
pub fn max_ref(a: f64, b: f64) -> f64 {
    match (a.is_nan(), b.is_nan()) {
        (true, _) => b,
        (_, true) => a,
        _ if a == b => {
            if a.is_sign_negative() {
                b
            } else {
                a
            }
        }
        _ => a.max(b),
    }
}

/// PTX's `min` on float values: a NaN is ignored, and `-0 < +0`.
pub fn min_ref(a: f64, b: f64) -> f64 {
    -max_ref(-a, -b)
}

/// IEEE 754's `maximum` (`arith.maximumf`, CuTe DSL's `cutlass.max` on
/// floats other than Float32): a NaN operand gives NaN, here `nan`, and
/// `-0 < +0`.
pub fn maximum_ref(a: f64, b: f64, nan: f64) -> f64 {
    if a.is_nan() || b.is_nan() { nan } else { max_ref(a, b) }
}

/// The lane of each warp of [`warp_reductions`] whose Float16, BFloat16 and
/// Float64 values are NaNs (negative, with payloads) for the maxima.
pub fn reduction_nan_lane(warp: usize) -> usize {
    [3, 26][warp % 2]
}

/// `warp_reduction`, `warp_reduction_sum` and `warp_reduction_max` over
/// every group size, on Int32 (spread over every bit, so the maxima see
/// negative values and the sums wrap), Float32 (whose sums round in the
/// butterfly's order), Float16, Float64, Uint8 (8-bit values shuffled as
/// 32-bit words), Int64 (two shuffles a step) and BFloat16; and with other
/// operators: xor, min, and `2a - b`, which is not commutative, so each
/// lane's result depends on the order `op(own, partner)`; and signed zeros,
/// whose maximum is `+0` on every float type and minimum `-0`. One lane of
/// each warp holds a NaN for the Float16, BFloat16 and Float64 maxima,
/// which are IEEE 754's `maximum` as in CuTe DSL (unlike Float32's `fmax`):
/// its group's result is the quiet NaN (a group of one keeps the NaN it
/// has).
pub fn warp_reductions() -> Case {
    const N: usize = GROUPS.len();
    let kernel = Kernel::builder("warp_reductions").block_dim([T as u32, 1, 1]).build(|| {
        let xi = param_tensor::<i32>("xi", Layout::col_major(T));
        let xf = param_tensor::<f32>("xf", Layout::col_major(T));
        let xh = param_tensor::<F16>("xh", Layout::col_major(T));
        let xd = param_tensor::<f64>("xd", Layout::col_major(T));
        let xb = param_tensor::<u8>("xb", Layout::col_major(T));
        let xl = param_tensor::<i64>("xl", Layout::col_major(T));
        let oi = param_tensor::<i32>("oi", Layout::row_major((T, 2 * N + 3)));
        let of = param_tensor::<f32>("of", Layout::row_major((T, 2 * N + 2)));
        let oh = param_tensor::<F16>("oh", Layout::row_major((T, 2 * N + 1)));
        let od = param_tensor::<f64>("od", Layout::row_major((T, 2 * N + 1)));
        let ob = param_tensor::<u8>("ob", Layout::row_major((T, 2 * N)));
        let ol = param_tensor::<i64>("ol", Layout::row_major((T, 2 * N)));
        let xg = param_tensor::<BF16>("xg", Layout::col_major(T));
        let og = param_tensor::<BF16>("og", Layout::row_major((T, 2 * N + 1)));
        let t = thread_idx().x;
        let i = xi.load(t) * SPREAD as i32;
        let (f, h, d, b, l) = (xf.load(t), xh.load(t), xd.load(t), xb.load(t), xl.load(t));
        let bf = xg.load(t);
        let nan_lane = lane_idx().eq(select(t.lt(32), reduction_nan_lane(0) as i32, reduction_nan_lane(1) as i32));
        let hn = select(nan_lane, F16(0xfe05), h);
        let dn = select(nan_lane, f64::from_bits(0xfff8_0000_0000_4567), d);
        let bn = select(nan_lane, BF16(0xffc3), bf);
        for (k, &g) in GROUPS.iter().enumerate() {
            oi.store((t, 2 * k), warp_reduction_sum(i, g));
            oi.store((t, 2 * k + 1), warp_reduction_max(i, g));
            of.store((t, 2 * k), warp_reduction_sum(f, g));
            of.store((t, 2 * k + 1), warp_reduction_max(f, g));
            oh.store((t, 2 * k), warp_reduction_sum(h, g));
            oh.store((t, 2 * k + 1), warp_reduction_max(hn, g));
            od.store((t, 2 * k), warp_reduction_sum(d, g));
            od.store((t, 2 * k + 1), warp_reduction_max(dn, g));
            ob.store((t, 2 * k), warp_reduction_sum(b, g));
            ob.store((t, 2 * k + 1), warp_reduction_max(b, g));
            ol.store((t, 2 * k), warp_reduction_sum(l, g));
            ol.store((t, 2 * k + 1), warp_reduction_max(l, g));
            og.store((t, 2 * k), warp_reduction_sum(bf, g));
            og.store((t, 2 * k + 1), warp_reduction_max(bn, g));
        }
        oi.store((t, 2 * N), warp_reduction(i, |a, c| a ^ c, 8));
        oi.store((t, 2 * N + 1), warp_reduction(i, |a, c| a.min(c), 16));
        oi.store((t, 2 * N + 2), warp_reduction(i, |a, c| a * 2 - c, 4));
        let zero = select((lane_idx() & 1).eq(1), -0.0f32, 0.0f32);
        of.store((t, 2 * N), warp_reduction_max(zero, 32));
        of.store((t, 2 * N + 1), warp_reduction(zero, |a, c| a.min(c), 32));
        // IEEE 754 maximum orders -0 below +0 too.
        let negative = (lane_idx() & 1).eq(1);
        oh.store((t, 2 * N), warp_reduction_max(select(negative, F16(0x8000), F16(0)), 32));
        od.store((t, 2 * N), warp_reduction_max(select(negative, -0.0f64, 0.0f64), 32));
        og.store((t, 2 * N), warp_reduction_max(select(negative, BF16(0x8000), BF16(0)), 32));
    });
    fn check(args: &[KernelArg]) {
        let xi: Vec<i32> = args[0].to_vec::<i32>().iter().map(|v| v.wrapping_mul(SPREAD as i32)).collect();
        let (xf, xh, xd) = (args[1].to_vec::<f32>(), args[2].to_vec::<F16>(), args[3].to_vec::<f64>());
        let (xb, xl) = (args[4].to_vec::<u8>(), args[5].to_vec::<i64>());
        let (oi, of, oh) = (args[6].to_vec::<i32>(), args[7].to_vec::<f32>(), args[8].to_vec::<F16>());
        let (od, ob, ol) = (args[9].to_vec::<f64>(), args[10].to_vec::<u8>(), args[11].to_vec::<i64>());
        let (xg, og) = (args[12].to_vec::<BF16>(), args[13].to_vec::<BF16>());
        const N: usize = GROUPS.len();
        let half_add = |a: F16, b: F16| F16::from_f32(a.to_f32() + b.to_f32());
        let bf_add = |a: BF16, b: BF16| BF16::from_f32(a.to_f32() + b.to_f32());
        // IEEE 754 maximum; the NaN is the quiet NaN of each type.
        let half_max = |a: F16, b: F16| {
            let (x, y) = (a.to_f32() as f64, b.to_f32() as f64);
            if x.is_nan() || y.is_nan() { F16(0x7e00) } else { F16::from_f32(max_ref(x, y) as f32) }
        };
        let bf_max = |a: BF16, b: BF16| {
            let (x, y) = (a.to_f32() as f64, b.to_f32() as f64);
            if x.is_nan() || y.is_nan() { BF16(0x7fc0) } else { BF16::from_f32(max_ref(x, y) as f32) }
        };
        let double_max = |a: f64, b: f64| maximum_ref(a, b, f64::from_bits(0x7ff8_0000_0000_0000));
        /// The warp's 32 values, with `nan` in the NaN lane.
        fn with_nan<V: Copy>(values: &[V], w: usize, nan: V) -> Vec<V> {
            (0..32).map(|l| if l == reduction_nan_lane(w) { nan } else { values[w * 32 + l] }).collect()
        }
        for w in 0..T / 32 {
            let lanes = w * 32..(w + 1) * 32;
            let nan_lane = reduction_nan_lane(w);
            let hn = with_nan(&xh, w, F16(0xfe05));
            let dn = with_nan(&xd, w, f64::from_bits(0xfff8_0000_0000_4567));
            let bn = with_nan(&xg, w, BF16(0xffc3));
            for (k, &g) in GROUPS.iter().enumerate() {
                let i_sum = butterfly_ref(&xi[lanes.clone()], g, i32::wrapping_add);
                let i_max = butterfly_ref(&xi[lanes.clone()], g, i32::max);
                let f_sum = butterfly_ref(&xf[lanes.clone()], g, |a, b| a + b);
                let f_max = butterfly_ref(&xf[lanes.clone()], g, |a, b| max_ref(a as f64, b as f64) as f32);
                let h_sum = butterfly_ref(&xh[lanes.clone()], g, half_add);
                let h_max = butterfly_ref(&hn, g, half_max);
                let d_sum = butterfly_ref(&xd[lanes.clone()], g, |a, b| a + b);
                let d_max = butterfly_ref(&dn, g, double_max);
                let b_sum = butterfly_ref(&xb[lanes.clone()], g, u8::wrapping_add);
                let b_max = butterfly_ref(&xb[lanes.clone()], g, u8::max);
                let l_sum = butterfly_ref(&xl[lanes.clone()], g, i64::wrapping_add);
                let l_max = butterfly_ref(&xl[lanes.clone()], g, i64::max);
                let g_sum = butterfly_ref(&xg[lanes.clone()], g, bf_add);
                let g_max = butterfly_ref(&bn, g, bf_max);
                // The reference's maxima are NaN across the NaN lane's group:
                // the quiet NaN, unless the group is the lane alone.
                for l in nan_lane / g * g..(nan_lane / g + 1) * g {
                    let alone = g == 1;
                    assert_eq!(h_max[l].0, if alone { 0xfe05 } else { 0x7e00 });
                    assert_eq!(d_max[l].to_bits(), if alone { 0xfff8_0000_0000_4567 } else { 0x7ff8 << 48 });
                    assert_eq!(g_max[l].0, if alone { 0xffc3 } else { 0x7fc0 });
                }
                for l in 0..32 {
                    let t = w * 32 + l;
                    let what = format!("group {g}, thread {t}");
                    assert_eq!(
                        [oi[t * (2 * N + 3) + 2 * k], oi[t * (2 * N + 3) + 2 * k + 1]],
                        [i_sum[l], i_max[l]],
                        "Int32 {what}"
                    );
                    let f = [of[t * (2 * N + 2) + 2 * k].to_bits(), of[t * (2 * N + 2) + 2 * k + 1].to_bits()];
                    assert_eq!(f, [f_sum[l].to_bits(), f_max[l].to_bits()], "Float32 {what}");
                    assert_eq!(
                        [oh[t * (2 * N + 1) + 2 * k].0, oh[t * (2 * N + 1) + 2 * k + 1].0],
                        [h_sum[l].0, h_max[l].0],
                        "Float16 {what}"
                    );
                    let d = [od[t * (2 * N + 1) + 2 * k].to_bits(), od[t * (2 * N + 1) + 2 * k + 1].to_bits()];
                    assert_eq!(d, [d_sum[l].to_bits(), d_max[l].to_bits()], "Float64 {what}");
                    assert_eq!(
                        [ob[t * 2 * N + 2 * k], ob[t * 2 * N + 2 * k + 1]],
                        [b_sum[l], b_max[l]],
                        "Uint8 {what}"
                    );
                    assert_eq!(
                        [ol[t * 2 * N + 2 * k], ol[t * 2 * N + 2 * k + 1]],
                        [l_sum[l], l_max[l]],
                        "Int64 {what}"
                    );
                    assert_eq!(
                        [og[t * (2 * N + 1) + 2 * k].0, og[t * (2 * N + 1) + 2 * k + 1].0],
                        [g_sum[l].0, g_max[l].0],
                        "BFloat16 {what}"
                    );
                }
            }
            let xor = butterfly_ref(&xi[lanes.clone()], 8, |a, b| a ^ b);
            let min = butterfly_ref(&xi[lanes.clone()], 16, i32::min);
            let skew = butterfly_ref(&xi[lanes.clone()], 4, |a: i32, b: i32| a.wrapping_mul(2).wrapping_sub(b));
            for l in 0..32 {
                let t = w * 32 + l;
                assert_eq!(oi[t * (2 * N + 3) + 2 * N..(t + 1) * (2 * N + 3)], [xor[l], min[l], skew[l]], "thread {t}");
                let zeros = [of[t * (2 * N + 2) + 2 * N].to_bits(), of[t * (2 * N + 2) + 2 * N + 1].to_bits()];
                assert_eq!(zeros, [0, 0x8000_0000], "signed zeros, thread {t}");
                let other = [
                    oh[t * (2 * N + 1) + 2 * N].0 as u64,
                    od[t * (2 * N + 1) + 2 * N].to_bits(),
                    og[t * (2 * N + 1) + 2 * N].0 as u64,
                ];
                assert_eq!(other, [0, 0, 0], "signed zeros of Float16, Float64 and BFloat16, thread {t}");
            }
        }
    }
    Case { name: "warp_reductions", kernel, cfg: launch(1), small_only: false, check }
}

/// The integer kinds of `redux.sync`.
const REDUX_INT_KINDS: [ReduxKind; 8] = [
    ReduxKind::Add,
    ReduxKind::And,
    ReduxKind::Or,
    ReduxKind::Xor,
    ReduxKind::Max,
    ReduxKind::Min,
    ReduxKind::Umax,
    ReduxKind::Umin,
];

/// `redux.sync` over 32-bit integers, from the PTX ISA's definitions: the
/// sum wraps, `max` and `min` compare signed unless the value is unsigned
/// (CuTe DSL takes them as `umax` and `umin` on a Uint32).
pub fn redux_int_ref(kind: ReduxKind, unsigned: bool, values: &[u32]) -> u32 {
    let signed = || values.iter().map(|&v| v as i32);
    match (kind, unsigned) {
        (ReduxKind::Add, _) => values.iter().fold(0u32, |a, &v| a.wrapping_add(v)),
        (ReduxKind::And, _) => values.iter().fold(u32::MAX, |a, &v| a & v),
        (ReduxKind::Or, _) => values.iter().fold(0, |a, &v| a | v),
        (ReduxKind::Xor, _) => values.iter().fold(0, |a, &v| a ^ v),
        (ReduxKind::Max, false) => signed().max().unwrap() as u32,
        (ReduxKind::Min, false) => signed().min().unwrap() as u32,
        (ReduxKind::Max | ReduxKind::Umax, _) => *values.iter().max().unwrap(),
        (ReduxKind::Min | ReduxKind::Umin, _) => *values.iter().min().unwrap(),
        (other, _) => panic!("{other:?} is not an integer kind"),
    }
}

/// Every integer `redux.sync` kind on Int32 and Uint32 data spread over
/// every bit (so signed and unsigned maxima differ and sums wrap), over the
/// whole warp; then with member masks: the two half-warps in divergent
/// branches, per-lane masks of interleaved groups of four lanes in
/// converged code, and only the lanes of a mask executing it.
pub fn redux_int() -> Case {
    const K: usize = REDUX_INT_KINDS.len();
    let thirds = (0..32).filter(|l| l % 3 != 0).fold(0u32, |m, l| m | 1 << l);
    let kernel = Kernel::builder("redux_int").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let us = param_tensor::<u32>("us", Layout::col_major(T));
        let oi = param_tensor::<i32>("oi", Layout::row_major((T, K + 2)));
        let ou = param_tensor::<u32>("ou", Layout::row_major((T, K + 2)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let (x, u) = (xs.load(t) * SPREAD as i32, us.load(t) * SPREAD as u32);
        for (k, kind) in REDUX_INT_KINDS.into_iter().enumerate() {
            oi.store((t, k), warp_redux_sync(x, kind));
            ou.store((t, k), warp_redux_sync(u, kind));
        }
        let plain = ReduxOptions::default();
        let halves = if_else_yield(
            lane.lt(16),
            || warp_redux_sync_with(x, ReduxKind::Add, 0x0000_ffffu32, plain),
            || warp_redux_sync_with(x, ReduxKind::Max, 0xffff_0000u32, plain),
        );
        oi.store((t, K), halves);
        let quads = select((lane & 4).eq(0), 0x0f0f_0f0fu32, 0xf0f0_f0f0u32);
        oi.store((t, K + 1), warp_redux_sync_with(x, ReduxKind::Min, quads, plain));
        ou.store((t, K), warp_redux_sync_with(u, ReduxKind::Max, quads, plain));
        let partial = if_else_yield((lane % 3).ne(0), || warp_redux_sync_with(u, ReduxKind::Xor, thirds, plain), || u);
        ou.store((t, K + 1), partial);
    });
    fn check(args: &[KernelArg]) {
        const K: usize = REDUX_INT_KINDS.len();
        let xs: Vec<u32> = args[0].to_vec::<i32>().iter().map(|v| v.wrapping_mul(SPREAD as i32) as u32).collect();
        let us: Vec<u32> = args[1].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let (oi, ou) = (args[2].to_vec::<i32>(), args[3].to_vec::<u32>());
        let thirds = (0..32).filter(|l| l % 3 != 0).fold(0u32, |m, l| m | 1 << l);
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            let over = |data: &[u32], mask: u32| -> Vec<u32> {
                (0..32).filter(|l| mask >> l & 1 == 1).map(|l| data[warp + l]).collect()
            };
            let mut want_i: Vec<u32> =
                REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, false, &over(&xs, !0))).collect();
            let mut want_u: Vec<u32> =
                REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, true, &over(&us, !0))).collect();
            want_i.push(if lane < 16 {
                redux_int_ref(ReduxKind::Add, false, &over(&xs, 0x0000_ffff))
            } else {
                redux_int_ref(ReduxKind::Max, false, &over(&xs, 0xffff_0000))
            });
            let quads = if lane & 4 == 0 { 0x0f0f_0f0f } else { 0xf0f0_f0f0 };
            want_i.push(redux_int_ref(ReduxKind::Min, false, &over(&xs, quads)));
            want_u.push(redux_int_ref(ReduxKind::Umax, true, &over(&us, quads)));
            want_u.push(if lane % 3 != 0 { redux_int_ref(ReduxKind::Xor, true, &over(&us, thirds)) } else { us[t] });
            let got_i: Vec<u32> = oi[t * (K + 2)..(t + 1) * (K + 2)].iter().map(|&v| v as u32).collect();
            assert_eq!(got_i, want_i, "Int32, thread {t}");
            assert_eq!(ou[t * (K + 2)..(t + 1) * (K + 2)], want_u[..], "Uint32, thread {t}");
        }
    }
    Case { name: "redux_int", kernel, cfg: launch(1), small_only: false, check }
}

/// `match_sync` on 32- and 64-bit values: `any` on small values (so lanes
/// share them), on negative Int32 values, on Uint64 values that differ in
/// the high word only, and on Int64; `all` over the whole warp (with
/// different values: 0), over each half-warp with per-lane masks (each half
/// agrees), with one lane of a half disagreeing, on a warp-uniform Uint64
/// and on one whose high word differs in one lane; and both in divergent
/// branches, each with the mask of its own lanes.
pub fn warp_match() -> Case {
    const COLS: usize = 11;
    let kernel = Kernel::builder("warp_match").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let x = xs.load(t);
        let l = ls.load(t) * SPREAD;
        let half = (lane / 16).bitcast::<u32>();
        let halves = select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32);
        let uniform = ls.load(t - lane) * SPREAD;
        let results: [Uint32; COLS] = [
            match_sync(FULL_MASK, x & 3u32, MatchKind::Any),
            match_sync(FULL_MASK, (x & 7u32).bitcast::<i32>() - 4, MatchKind::Any),
            match_sync(FULL_MASK, half, MatchKind::All),
            match_sync(halves, half, MatchKind::All),
            match_sync(halves, select(lane.eq(20), 7u32, half), MatchKind::All),
            match_sync(FULL_MASK, ((lane % 4).to::<u64>() << 40u64) | 5u64, MatchKind::Any),
            match_sync(FULL_MASK, uniform, MatchKind::All),
            match_sync(FULL_MASK, uniform ^ select(lane.eq(9), 1u64 << 50, 0u64), MatchKind::All),
            match_sync(FULL_MASK, l & 0x3_0000_0000u64, MatchKind::Any),
            match_sync(FULL_MASK, (l % 3u64).bitcast::<i64>() - 1, MatchKind::Any),
            if_else_yield(
                (lane % 2).eq(0),
                || match_sync(0x5555_5555u32, x & 3u32, MatchKind::Any),
                || match_sync(0xaaaa_aaaau32, l & 1u64, MatchKind::All),
            ),
        ];
        for (j, r) in results.into_iter().enumerate() {
            out.store((t, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[2].to_vec::<u32>());
        let ls: Vec<u64> = args[1].to_vec::<u64>().iter().map(|v| v.wrapping_mul(SPREAD)).collect();
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            // `any`: the lanes of `mask` whose value equals this lane's;
            // `all`: `mask` if they all hold one value, else 0.
            let any = |mask: u32, v: &dyn Fn(usize) -> u64| -> u32 {
                (0..32).filter(|&o| mask >> o & 1 == 1 && v(o) == v(lane)).fold(0, |m, o| m | 1 << o)
            };
            let all = |mask: u32, v: &dyn Fn(usize) -> u64| -> u32 {
                let members: Vec<usize> = (0..32).filter(|&o| mask >> o & 1 == 1).collect();
                if members.iter().all(|&o| v(o) == v(members[0])) { mask } else { 0 }
            };
            let x = |o: usize| xs[warp + o] as u64;
            let l = |o: usize| ls[warp + o];
            let half = |o: usize| (o / 16) as u64;
            let halves = if lane < 16 { 0x0000_ffff } else { 0xffff_0000 };
            let uniform = ls[warp];
            let want = [
                any(!0, &|o| x(o) & 3),
                any(!0, &|o| ((x(o) & 7) as i32 - 4) as u32 as u64),
                all(!0, &half),
                all(halves, &half),
                all(halves, &|o| if o == 20 { 7 } else { half(o) }),
                any(!0, &|o| ((o % 4) as u64) << 40 | 5),
                all(!0, &|_| uniform),
                all(!0, &|o| uniform ^ if o == 9 { 1 << 50 } else { 0 }),
                any(!0, &|o| l(o) & 0x3_0000_0000),
                any(!0, &|o| ((l(o) % 3) as i64 - 1) as u64),
                if lane % 2 == 0 { any(0x5555_5555, &|o| x(o) & 3) } else { all(0xaaaa_aaaa, &|o| l(o) & 1) },
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
        }
    }
    Case { name: "warp_match", kernel, cfg: launch(1), small_only: false, check }
}

/// `sync_warp_with` and half-warp masks: each half-warp synchronises
/// separately in a divergent branch and reads what its neighbours stored
/// to shared memory; both halves synchronise in converged code with
/// per-lane masks; and the whole warp synchronises before reading the other
/// half's values.
pub fn warp_barriers() -> Case {
    let kernel = Kernel::builder("warp_barriers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 3)));
        let smem = make_smem_tensor::<u32>(Layout::col_major(T));
        let t = thread_idx().x;
        let lane = lane_idx();
        smem.store(t, xs.load(t));
        let first = if_else_yield(
            lane.lt(16),
            || {
                sync_warp_with(0x0000_ffffu32);
                smem.load(t ^ 1)
            },
            || {
                sync_warp_with(0xffff_0000u32);
                smem.load(t - lane + 16 + (lane - 13) % 16)
            },
        );
        out.store((t, 0), first);
        sync_warp_with(select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32));
        out.store((t, 1), smem.load(t ^ 7));
        sync_warp();
        out.store((t, 2), smem.load(t ^ 16));
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            let first = if lane < 16 { xs[t ^ 1] } else { xs[warp + 16 + (lane - 13) % 16] };
            assert_eq!(out[t * 3..t * 3 + 3], [first, xs[t ^ 7], xs[t ^ 16]], "thread {t}");
        }
    }
    Case { name: "warp_barriers", kernel, cfg: launch(1), small_only: false, check }
}

/// Float32 bit patterns `fmax` and `fmin` meet in every pair: signed zeros
/// and infinities; NaNs (canonical, quiet, negative with a payload,
/// signaling); subnormals (the smallest, a negative one, the largest); the
/// smallest normal numbers; and ordinary and extreme numbers.
pub const MINMAX_VALUES: [u32; 19] = [
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fff_ffff,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x0000_0001,
    0x8000_0003,
    0x007f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3fc0_0000,
    0xc020_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
];

/// Every combination of `fmax`'s and `fmin`'s `abs`, `nan` and `ftz`.
pub fn minmax_options() -> [MinMaxOptions; 8] {
    std::array::from_fn(|k| MinMaxOptions { abs: k & 1 != 0, nan: k & 2 != 0, ftz: k & 4 != 0 })
}

/// `max{.ftz}{.NaN}{.xorsign.abs}.f32` (or `min`) on raw bits, from the PTX
/// ISA's pseudo-code: `ftz` flushes subnormal operands (and results) to
/// zeros of their signs; with `abs` the sign of the result is the
/// exclusive or of the operands' signs, taken before their absolute values
/// are compared, unless the result is NaN; NaN results are the canonical
/// NaN.
pub fn minmax_ref(is_max: bool, a: u32, b: u32, opts: MinMaxOptions) -> u32 {
    let flush = |v: u32| if opts.ftz && v & 0x7f80_0000 == 0 { v & 0x8000_0000 } else { v };
    let is_nan = |v: u32| v & 0x7fff_ffff > 0x7f80_0000;
    let (a, b) = (flush(a), flush(b));
    let sign = (a ^ b) & 0x8000_0000;
    let (a, b) = if opts.abs { (a & 0x7fff_ffff, b & 0x7fff_ffff) } else { (a, b) };
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    let d = match (is_nan(a), is_nan(b)) {
        (true, true) => 0x7fff_ffff,
        (true, false) | (false, true) if opts.nan => 0x7fff_ffff,
        (true, false) => b,
        (false, true) => a,
        // Equal values differ only as signed zeros.
        _ if x == y => {
            if (a >> 31 == 0) == is_max {
                a
            } else {
                b
            }
        }
        _ => {
            if (x > y) == is_max {
                a
            } else {
                b
            }
        }
    };
    if is_nan(d) {
        0x7fff_ffff
    } else if opts.abs {
        flush(d & 0x7fff_ffff | sign)
    } else {
        flush(d)
    }
}

/// Rounds of pairs per thread in [`fmax_fmin`]: 6 * 64 threads cover the 361
/// pairs of [`MINMAX_VALUES`].
const MINMAX_ROUNDS: usize = 6;

/// `fmax` and `fmin` with every combination of `abs`, `nan` and `ftz`, on
/// every pair of [`MINMAX_VALUES`] (read from a table at run time, so
/// nothing folds) and on the input data. Needs sm_86 (`.xorsign.abs`).
pub fn fmax_fmin() -> Case {
    const COLS: usize = (MINMAX_ROUNDS + 1) * 16;
    let kernel = Kernel::builder("fmax_fmin").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let ys = param_tensor::<f32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let n = MINMAX_VALUES.len() as i32;
        let table = make_rmem_tensor::<u32>(Layout::col_major(MINMAX_VALUES.len()));
        for (i, v) in MINMAX_VALUES.into_iter().enumerate() {
            table.store(i, v);
        }
        let mut pairs = Vec::new();
        for r in 0..MINMAX_ROUNDS as i32 {
            let p = (t + T as i32 * r) % (n * n);
            pairs.push((table.load(p / n).bitcast::<f32>(), table.load(p % n).bitcast::<f32>()));
        }
        pairs.push((xs.load(t), ys.load(t)));
        let mut column = 0;
        for (a, b) in pairs {
            for opts in minmax_options() {
                out.store((t, column), fmax_with(a, b, opts).bitcast::<u32>());
                out.store((t, column + 8), fmin_with(a, b, opts).bitcast::<u32>());
                column += 1;
            }
            column += 8;
        }
        assert_eq!(column, COLS);
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = (MINMAX_ROUNDS + 1) * 16;
        let (xs, ys, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<u32>());
        let n = MINMAX_VALUES.len();
        for t in 0..T {
            let mut pairs: Vec<(u32, u32)> = (0..MINMAX_ROUNDS)
                .map(|r| (t + T * r) % (n * n))
                .map(|p| (MINMAX_VALUES[p / n], MINMAX_VALUES[p % n]))
                .collect();
            pairs.push((xs[t].to_bits(), ys[t].to_bits()));
            for (k, &(a, b)) in pairs.iter().enumerate() {
                for (j, opts) in minmax_options().into_iter().enumerate() {
                    let (max, min) = (out[t * COLS + 16 * k + j], out[t * COLS + 16 * k + 8 + j]);
                    let what = format!("thread {t}, {a:#010x} and {b:#010x}, {opts:?}");
                    assert_eq!(max, minmax_ref(true, a, b, opts), "fmax of {what}: got {max:#010x}");
                    assert_eq!(min, minmax_ref(false, a, b, opts), "fmin of {what}: got {min:#010x}");
                }
            }
        }
    }
    Case { name: "fmax_fmin", kernel, cfg: launch(1), small_only: false, check }
}

/// Float32 values of the first four groups of four lanes of [`redux_f32`]:
/// signed zeros; NaNs among numbers; NaNs only (with payloads, negative,
/// signaling); infinities and subnormals.
pub const REDUX_F32_SPECIALS: [u32; 16] = [
    0x8000_0000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0000,
    0x7fc0_0000,
    0x3fc0_0000,
    0xc020_0000,
    0xffc0_0001,
    0x7fc0_0000,
    0xffc0_0005,
    0x7f80_0001,
    0x7fff_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0001,
    0x8000_0003,
];

/// `redux.sync.{max,min}{.abs}{.NaN}.f32` over raw bits, from the PTX
/// ISA's definition: `.abs` reduces the absolute values; `+0 > -0`; a NaN
/// input is ignored (all NaN: the canonical NaN) unless `.NaN` makes the
/// result the canonical NaN.
pub fn redux_f32_ref(is_max: bool, abs: bool, nan: bool, values: &[u32]) -> u32 {
    let is_nan = |v: u32| v & 0x7fff_ffff > 0x7f80_0000;
    let values: Vec<u32> = values.iter().map(|&v| if abs { v & 0x7fff_ffff } else { v }).collect();
    if nan && values.iter().any(|&v| is_nan(v)) {
        return 0x7fff_ffff;
    }
    let opts = MinMaxOptions::default();
    values.into_iter().filter(|&v| !is_nan(v)).reduce(|a, b| minmax_ref(is_max, a, b, opts)).unwrap_or(0x7fff_ffff)
}

/// Float32 `redux.sync` (sm_100a, or the sm_100f family): `fmax` and `fmin`
/// with every combination of `abs` and `nan`, over the whole warp and over
/// groups of four lanes (per-lane masks), on lanes holding
/// [`REDUX_F32_SPECIALS`] and the input data; and `max` and `min`, which
/// mean `fmax` and `fmin` on Float32.
pub fn redux_f32() -> Case {
    const COLS: usize = 2 * 4 * 2 + 2;
    let kernel = Kernel::builder("redux_f32").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let table = make_rmem_tensor::<u32>(Layout::col_major(REDUX_F32_SPECIALS.len()));
        for (i, v) in REDUX_F32_SPECIALS.into_iter().enumerate() {
            table.store(i, v);
        }
        let v = select(lane.lt(16), table.load(lane & 15).bitcast::<f32>(), xs.load(t));
        let quad = Value::constant(0xfu32) << (lane & !3).bitcast::<u32>();
        let mut column = 0;
        for kind in [ReduxKind::Fmax, ReduxKind::Fmin] {
            for (abs, nan) in [(false, false), (true, false), (false, true), (true, true)] {
                let opts = ReduxOptions { abs, nan };
                out.store((t, column), warp_redux_sync_with(v, kind, FULL_MASK, opts).bitcast::<u32>());
                out.store((t, column + 1), warp_redux_sync_with(v, kind, quad, opts).bitcast::<u32>());
                column += 2;
            }
        }
        out.store((t, column), warp_redux_sync(v, ReduxKind::Max).bitcast::<u32>());
        out.store((t, column + 1), warp_redux_sync(v, ReduxKind::Min).bitcast::<u32>());
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = 2 * 4 * 2 + 2;
        let (xs, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, t % 32);
            let value = |l: usize| if l < 16 { REDUX_F32_SPECIALS[l] } else { xs[warp + l].to_bits() };
            let warp_values: Vec<u32> = (0..32).map(value).collect();
            let quad: Vec<u32> = (lane & !3..(lane & !3) + 4).map(value).collect();
            let mut want = Vec::new();
            for is_max in [true, false] {
                for (abs, nan) in [(false, false), (true, false), (false, true), (true, true)] {
                    want.push(redux_f32_ref(is_max, abs, nan, &warp_values));
                    want.push(redux_f32_ref(is_max, abs, nan, &quad));
                }
            }
            want.push(redux_f32_ref(true, false, false, &warp_values));
            want.push(redux_f32_ref(false, false, false, &warp_values));
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "redux_f32", kernel, cfg: launch(1), small_only: false, check }
}

/// Shuffles of every kind whose operands `b` and `c` (`mask_and_clamp`)
/// both come from the data, different in every lane: on random data, any
/// segment mask and clamp, held to the PTX ISA's pseudo-code.
pub fn shuffle_operands() -> Case {
    const KINDS: [ShuffleKind; 4] = [ShuffleKind::Idx, ShuffleKind::Up, ShuffleKind::Down, ShuffleKind::Bfly];
    let kernel = Kernel::builder("shuffle_operands").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let bs = param_tensor::<i32>("bs", Layout::col_major(T));
        let cs = param_tensor::<i32>("cs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, KINDS.len())));
        let t = thread_idx().x;
        let (x, b, c) = (xs.load(t), bs.load(t), cs.load(t));
        for (j, kind) in KINDS.into_iter().enumerate() {
            out.store((t, j), shuffle_sync_op(x, b, FULL_MASK, c, kind));
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, bs, cs) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
            let want = KINDS
                .map(|kind| shfl_ref(kind, lane, bs[t] as u32, cs[t] as u32).map_or(xs[t], |s| xs[warp + s as usize]));
            assert_eq!(out[t * 4..t * 4 + 4], want, "thread {t}: b = {:#x}, c = {:#x}", bs[t], cs[t]);
        }
    }
    Case { name: "shuffle_operands", kernel, cfg: launch(1), small_only: false, check }
}

/// Warp collectives over a random partition of each warp: every lane's
/// member mask is the set of lanes with the same group (2 bits of the data),
/// found with `match_sync(.., MatchKind::Any)`. Each group then reduces with
/// every integer `redux.sync` kind, compares with `match.all`, and
/// broadcasts its lowest lane's value with a masked shuffle.
pub fn warp_partitions() -> Case {
    const COLS: usize = REDUX_INT_KINDS.len() + 3;
    let kernel = Kernel::builder("warp_partitions").block_dim([T as u32, 1, 1]).build(|| {
        let gs = param_tensor::<u32>("gs", Layout::col_major(T));
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let group = gs.load(t) >> 3u32 & 3u32;
        let x = xs.load(t) * SPREAD as u32;
        let mask = match_sync(FULL_MASK, group, MatchKind::Any);
        let plain = ReduxOptions::default();
        let mut results: Vec<Uint32> = REDUX_INT_KINDS
            .into_iter()
            .map(|kind| warp_redux_sync_with(x.bitcast::<i32>(), kind, mask, plain).bitcast())
            .collect();
        results.push(mask);
        results.push(match_sync(mask, x & 1u32, MatchKind::All));
        results.push(shuffle_sync_with(x, mask.ctz().bitcast::<i32>(), mask, 31));
        for (j, r) in results.into_iter().enumerate() {
            out.store((t, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = REDUX_INT_KINDS.len() + 3;
        let groups: Vec<u32> = args[0].to_vec::<u32>().iter().map(|g| g >> 3 & 3).collect();
        let xs: Vec<u32> = args[1].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let out = args[2].to_vec::<u32>();
        let sizes: std::collections::BTreeSet<u32> =
            (0..T).map(|t| (0..32).filter(|l| groups[t / 32 * 32 + l] == groups[t]).count() as u32).collect();
        assert!(sizes.len() > 1, "the partitions should have groups of several sizes: {sizes:?}");
        for t in 0..T {
            let warp = t / 32 * 32;
            let mask = (0..32).filter(|&l| groups[warp + l] == groups[t]).fold(0u32, |m, l| m | 1 << l);
            let members: Vec<u32> = (0..32).filter(|l| mask >> l & 1 == 1).map(|l| xs[warp + l]).collect();
            let mut want: Vec<u32> = REDUX_INT_KINDS.iter().map(|&k| redux_int_ref(k, false, &members)).collect();
            want.push(mask);
            want.push(if members.iter().all(|v| v & 1 == members[0] & 1) { mask } else { 0 });
            want.push(xs[warp + mask.trailing_zeros() as usize]);
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}, mask {mask:#010x}");
        }
    }
    Case { name: "warp_partitions", kernel, cfg: launch(1), small_only: false, check }
}

/// The warp-collective kernels added with full shuffles: shuffles of every
/// kind on eight types and with operands from the data, butterfly
/// reductions, integer `redux.sync`, `match_sync`, masked `sync_warp`,
/// random partitions, and the collectives in short warps. They run on every
/// target, and are in [`all`].
pub fn warp_collectives() -> Vec<Case> {
    vec![
        shuffles_i8(),
        shuffles_u16(),
        shuffles_f16(),
        shuffles_bf16(),
        shuffles_i32(),
        shuffles_f32(),
        shuffles_i64(),
        shuffles_f64(),
        shuffle_operands(),
        warp_reductions(),
        redux_int(),
        warp_match(),
        warp_barriers(),
        warp_partitions(),
        short_warps(),
    ]
}

/// Kernels that need sm_86 or newer (`max.xorsign.abs`), which the SM89 GPU
/// runs: simulated, compiled for sm_86, and run on GPUs of sm_86 or newer.
pub fn sm86() -> Vec<Case> {
    vec![fmax_fmin()]
}

/// Kernels that need sm_100a (or the sm_100f family): simulated, compiled
/// for sm_100a with `mlir-opt`, and assembled by `tests/ptxas.rs`.
pub fn sm100a() -> Vec<Case> {
    let mut cases =
        vec![redux_f32(), ue8m0_pairs(), mx_conversions(), ldmatrix_b8_forms(), ssa_fp4_fp6(), stmatrix_b8_forms()];
    // Gather4, scatter4, byte-masked bulk stores, FP4/FP6 TMA formats.
    cases.extend(tma_sm100_cases());
    cases
}

/// Blackwell kernels with tensor memory: simulated, compiled
/// for sm_100a with `mlir-opt` and assembled by `tests/ptxas.rs`; `tests/gpu.rs` skips them
/// (no GPU here runs sm_100).
pub fn sm100() -> Vec<Case> {
    let mut cases = tmem_cases();
    // TMEM loads and stores.
    cases.extend(tmem_ldst_cases());
    cases
}

/// Kernels that need sm_90a (setmaxnreg; sm_100a and sm_120a have it too): simulated,
/// compiled for sm_90a, and assembled by `tests/ptxas.rs` and `tests/gpu.rs`.
pub fn sm90a() -> Vec<Case> {
    let mut cases = vec![setmaxnreg_warp_specialized()];
    // Warpgroup MMAs: a few here (`tests/wgmma.rs` runs
    // them all).
    cases.extend(wgmma_cases_sampled());
    // The FP8, integer and register-A variants (`tests/wgmma_variants.rs`
    // runs them all).
    cases.extend(wgmma_variant_cases_sampled());
    // In-kernel tensor-map updates.
    cases.extend(tensormap_cases_sm90a());
    cases
}

/// Kernels that need sm_120a (or the sm_120f family): simulated, compiled
/// for sm_120a with `mlir-opt`, and assembled by `tests/ptxas.rs`.
pub fn sm120a() -> Vec<Case> {
    vec![mx_conversions(), ldmatrix_b8_forms(), stmatrix_b8_forms()]
}

/// Kernels that need sm_89 or newer (the FP8 conversions), which the SM89
/// GPU runs: simulated, compiled for sm_89, and run on GPUs of sm_89 or
/// newer.
pub fn sm89() -> Vec<Case> {
    let mut cases = vec![fp8_conversions(), fp8_f16_sweep(), ssa_fp8()];
    cases.extend(mma_fp8());
    cases.push(ldmatrix_8x16x8b_fp8());
    cases.extend(pipeline_cases_sm89());
    cases.extend(async_sched_cases_sm89());
    cases.extend(cpasync_order_cases_sm89());
    cases.extend(gemm_helper_cases_sm89());
    cases
}

/// Threads per block of [`short_warps`]: a whole warp, then a warp of 16
/// lanes.
pub const SHORT_BLOCK: usize = 48;

/// The columns of [`short_warps`].
const SHORT_COLUMNS: usize = 18;

/// Warp collectives with the full member mask in blocks of [`SHORT_BLOCK`]
/// threads, whose second warp has 16 lanes. The lanes such a warp lacks
/// count as exited, as the PTX ISA says of `shfl.sync`, `vote.sync`,
/// `match.sync` and `redux.sync`: `sync_warp` and shuffles that read lanes
/// the warp has work; `match.all` returns `0x0000ffff`; `redux.sync`,
/// `warp_reduction` over groups of 16 and the votes take the 16 lanes
/// (`vote_all_sync` of "lane < 16" holds); ballots, `match.any` and
/// `activemask` have 16 bits. Two blocks, so that reading past a short warp
/// would read the next block's threads.
pub fn short_warps() -> Case {
    let kernel = Kernel::builder("short_warps").block_dim([SHORT_BLOCK as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * SHORT_BLOCK));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * SHORT_BLOCK, SHORT_COLUMNS)));
        let smem = make_smem_tensor::<u32>(Layout::col_major(SHORT_BLOCK));
        let t = thread_idx().x;
        let g = block_idx().x * SHORT_BLOCK as i32 + t;
        let lane = lane_idx();
        let x = xs.load(g) * SPREAD as u32;
        smem.store(t, x);
        sync_warp();
        let low = lane.lt(16);
        let results: [Uint32; SHORT_COLUMNS] = [
            smem.load(t ^ 1),
            shuffle_sync(x, 5),
            shuffle_sync_bfly(x, 4),
            shuffle_sync_up(x, 3),
            shuffle_sync_down_with(x, 3, FULL_MASK, 0x100f),
            match_sync(FULL_MASK, x & 3u32, MatchKind::Any),
            match_sync(FULL_MASK, warp_idx(), MatchKind::All),
            match_sync(FULL_MASK, x & 1u32, MatchKind::All),
            warp_redux_sync(x, ReduxKind::Add),
            warp_redux_sync(x, ReduxKind::Max),
            warp_redux_sync(x.bitcast::<i32>(), ReduxKind::Min).bitcast(),
            warp_reduction_sum(x, 16),
            vote_ballot_sync((x & 1u32).eq(1u32)),
            vote_all_sync(low).to(),
            vote_any_sync(lane.ge(16)).to(),
            vote_uni_sync(low).to(),
            vote_ballot_sync(elect_one()),
            activemask(),
        ];
        for (j, r) in results.into_iter().enumerate() {
            out.store((g, j), r);
        }
    });
    fn check(args: &[KernelArg]) {
        let xs: Vec<u32> = args[0].to_vec::<u32>().iter().map(|v| v.wrapping_mul(SPREAD as u32)).collect();
        let out = args[1].to_vec::<u32>();
        for g in 0..2 * SHORT_BLOCK {
            let (block, t) = (g / SHORT_BLOCK, g % SHORT_BLOCK);
            let (warp, lane) = (t / 32, t % 32);
            // The warp's lanes: 32, or 16 for the second warp of a block.
            let size = (SHORT_BLOCK - warp * 32).min(32);
            let x = |l: usize| {
                assert!(l < size, "the reference reads lane {l} of a warp of {size}");
                xs[block * SHORT_BLOCK + warp * 32 + l]
            };
            let mask_of = |f: &dyn Fn(usize) -> bool| (0..size).filter(|&l| f(l)).fold(0u32, |m, l| m | 1 << l);
            let members = mask_of(&|_| true);
            let half = lane / 16 * 16;
            let want: [u32; SHORT_COLUMNS] = [
                x(lane ^ 1),
                x(5),
                x(lane ^ 4),
                if lane >= 3 { x(lane - 3) } else { x(lane) },
                if lane % 16 + 3 < 16 { x(lane + 3) } else { x(lane) },
                mask_of(&|l| x(l) & 3 == x(lane) & 3),
                members,
                if (0..size).all(|l| x(l) & 1 == x(0) & 1) { members } else { 0 },
                (0..size).fold(0u32, |a, l| a.wrapping_add(x(l))),
                (0..size).map(x).max().unwrap(),
                (0..size).map(|l| x(l) as i32).min().unwrap() as u32,
                (half..half + 16).fold(0u32, |a, l| a.wrapping_add(x(l))),
                mask_of(&|l| x(l) & 1 == 1),
                (size == 16) as u32,
                (size == 32) as u32,
                (size == 16) as u32,
                1,
                members,
            ];
            assert_eq!(
                out[g * SHORT_COLUMNS..(g + 1) * SHORT_COLUMNS],
                want,
                "block {block}, warp {warp}, lane {lane}"
            );
        }
    }
    Case { name: "short_warps", kernel, cfg: launch_n(2, SHORT_BLOCK), small_only: false, check }
}

// Extended-precision arithmetic: carry chains through PTX's carry flag,
// checked against u128 arithmetic.

/// The 256-bit `a + b + carry_in` of numbers held as (low, high) u128
/// halves, and its carry out.
fn add256(a: [u128; 2], b: [u128; 2], carry_in: bool) -> ([u128; 2], bool) {
    let (lo, c1) = a[0].overflowing_add(b[0]);
    let (lo, c2) = lo.overflowing_add(carry_in as u128);
    let (hi, c3) = a[1].overflowing_add(b[1]);
    let (hi, c4) = hi.overflowing_add((c1 | c2) as u128);
    ([lo, hi], c3 | c4)
}

/// The 256-bit `a - b - borrow_in` of (low, high) u128 halves, and its
/// borrow out.
fn sub256(a: [u128; 2], b: [u128; 2], borrow_in: bool) -> ([u128; 2], bool) {
    let (lo, b1) = a[0].overflowing_sub(b[0]);
    let (lo, b2) = lo.overflowing_sub(borrow_in as u128);
    let (hi, b3) = a[1].overflowing_sub(b[1]);
    let (hi, b4) = hi.overflowing_sub((b1 | b2) as u128);
    ([lo, hi], b3 | b4)
}

/// Four 32-bit words (low first) as a u128.
fn u128_of(w: [u32; 4]) -> u128 {
    w.iter().rev().fold(0, |acc, &x| acc << 32 | x as u128)
}

/// A u128 as four 32-bit words, low first.
fn words_of(x: u128) -> [u32; 4] {
    [x as u32, (x >> 32) as u32, (x >> 64) as u32, (x >> 96) as u32]
}

/// `a + b + carry_in` over four 32-bit words, with u128 arithmetic.
fn add_words_u32(a: [u32; 4], b: [u32; 4], carry_in: bool) -> ([u32; 4], bool) {
    let (s, c1) = u128_of(a).overflowing_add(u128_of(b));
    let (s, c2) = s.overflowing_add(carry_in as u128);
    (words_of(s), c1 | c2)
}

/// `a - b - borrow_in` over four 32-bit words, with u128 arithmetic.
fn sub_words_u32(a: [u32; 4], b: [u32; 4], borrow_in: bool) -> ([u32; 4], bool) {
    let (d, b1) = u128_of(a).overflowing_sub(u128_of(b));
    let (d, b2) = d.overflowing_sub(borrow_in as u128);
    (words_of(d), b1 | b2)
}

/// Four 64-bit words (low first) as (low, high) u128 halves.
fn halves_of(w: [u64; 4]) -> [u128; 2] {
    [w[0] as u128 | (w[1] as u128) << 64, w[2] as u128 | (w[3] as u128) << 64]
}

/// `a + b + carry_in` over four 64-bit words, with u128 arithmetic.
fn add_words_u64(a: [u64; 4], b: [u64; 4], carry_in: bool) -> ([u64; 4], bool) {
    let ([lo, hi], c) = add256(halves_of(a), halves_of(b), carry_in);
    ([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64], c)
}

/// `a - b - borrow_in` over four 64-bit words, with u128 arithmetic.
fn sub_words_u64(a: [u64; 4], b: [u64; 4], borrow_in: bool) -> ([u64; 4], bool) {
    let ([lo, hi], br) = sub256(halves_of(a), halves_of(b), borrow_in);
    ([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64], br)
}

/// Columns of the carry-chain kernels: a sum of four words with its carry
/// out, then a difference with its borrow out (5 + 5), a word computed
/// between two carry instructions (1), CuTe DSL's two-word forms, whose
/// `addc` and `subc` do not write the flag, so that two of them add the same
/// carry (3 + 3), the sum and difference on the signed type of the same width
/// (4 + 4), and `add_with_carry` and `sub_with_borrow` chains from a runtime
/// carry (5 + 5).
const CARRY_COLS: usize = 5 + 5 + 1 + 3 + 3 + 4 + 4 + 5 + 5;

/// Multi-word sums and differences of four words of `$t` (signed `$s`): a
/// 128-bit number for 32-bit words, 256 bits for 64-bit words. The words are
/// spread over every bit, and threads shape them so that carries run
/// through every word: `t % 4 == 1` gives `a` all-ones low words and an odd
/// low word of `b` (the sum carries all the way), `t % 4 == 2` zero low words
/// of `a` and an odd low word of `b` (the difference borrows all the way),
/// and `t % 4 == 3` low words of `b` that complement `a`'s (the sum carries
/// all the way only with a carry in). `$add` and `$sub` are the references,
/// in u128 arithmetic.
macro_rules! carry_words_case {
    ($name:ident, $t:ty, $s:ty, $add:expr, $sub:expr) => {
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([T as u32, 1, 1]).build(|| {
                let xs = param_tensor::<$t>("a", Layout::row_major((T, 4)));
                let ys = param_tensor::<$t>("b", Layout::row_major((T, 4)));
                let out = param_tensor::<$t>("out", Layout::row_major((T, CARRY_COLS)));
                let t = thread_idx().x;
                let m = t % 4;
                let ones = <$t>::MAX;
                let mut a: [Value<$t>; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as $t);
                let mut b: [Value<$t>; 4] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as $t);
                for i in 0..3 {
                    b[i] = select(m.eq(3), !a[i], b[i]);
                    a[i] = select(m.eq(1), Value::constant(ones), select(m.eq(2), Value::constant(0), a[i]));
                }
                b[0] = select(m.eq(1) | m.eq(2), b[0] | 1, b[0]);
                let zero = || Value::constant(0 as $t);
                let mut cols: Vec<Value<$t>> = Vec::new();
                // The sum, with a word computed between two carry instructions.
                cols.push(add_cc(a[0], b[0]));
                let between = (a[1] ^ b[2]) * 3 + (a[0] >> 3);
                cols.extend([addc_cc(a[1], b[1]), addc_cc(a[2], b[2]), addc_cc(a[3], b[3]), addc(zero(), zero())]);
                // The difference; the last subc gives 0 - borrow.
                cols.extend([sub_cc(a[0], b[0]), subc_cc(a[1], b[1]), subc_cc(a[2], b[2]), subc_cc(a[3], b[3])]);
                cols.push(subc(zero(), zero()));
                cols.push(between);
                // CuTe DSL's addc and subc read the flag and leave it.
                cols.extend([add_cc(a[0], b[0]), addc(a[1], b[1]), addc(a[2], b[2])]);
                cols.extend([sub_cc(a[0], b[0]), subc(a[1], b[1]), subc(a[2], b[2])]);
                // The signed type gives the same bits.
                let (sa, sb) = (a.map(|v| v.bitcast::<$s>()), b.map(|v| v.bitcast::<$s>()));
                let signed = [add_cc(sa[0], sb[0]), addc_cc(sa[1], sb[1]), addc_cc(sa[2], sb[2]), addc(sa[3], sb[3])];
                cols.extend(signed.map(|v| v.bitcast::<$t>()));
                let signed = [sub_cc(sa[0], sb[0]), subc_cc(sa[1], sb[1]), subc_cc(sa[2], sb[2]), subc(sa[3], sb[3])];
                cols.extend(signed.map(|v| v.bitcast::<$t>()));
                // Carries as values, from a runtime carry in.
                let mut carry = ((t >> 2) & 1).to::<$t>();
                for i in 0..4 {
                    let (s, c) = add_with_carry(a[i], b[i], carry);
                    cols.push(s);
                    carry = c;
                }
                cols.push(carry);
                let mut borrow = ((t >> 3) & 1).to::<$t>();
                for i in 0..4 {
                    let (d, c) = sub_with_borrow(a[i], b[i], borrow);
                    cols.push(d);
                    borrow = c;
                }
                cols.push(borrow);
                assert_eq!(cols.len(), CARRY_COLS);
                for (j, v) in cols.into_iter().enumerate() {
                    out.store((t, j), v);
                }
            });
            fn check(args: &[KernelArg]) {
                let add: fn([$t; 4], [$t; 4], bool) -> ([$t; 4], bool) = $add;
                let sub: fn([$t; 4], [$t; 4], bool) -> ([$t; 4], bool) = $sub;
                let (xs, ys, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>(), args[2].to_vec::<$t>());
                let mut rippled = [0; 4];
                for t in 0..T {
                    let mut a: [$t; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as $t));
                    let mut b: [$t; 4] = std::array::from_fn(|i| ys[t * 4 + i].wrapping_mul(SPREAD as $t));
                    let m = t % 4;
                    for i in 0..3 {
                        if m == 3 {
                            b[i] = !a[i];
                        }
                        a[i] = match m {
                            1 => <$t>::MAX,
                            2 => 0,
                            _ => a[i],
                        };
                    }
                    if m == 1 || m == 2 {
                        b[0] |= 1;
                    }
                    let mut want: Vec<$t> = Vec::new();
                    let (s, c) = add(a, b, false);
                    want.extend(s);
                    want.push(c as $t);
                    let (d, br) = sub(a, b, false);
                    want.extend(d);
                    want.push(if br { <$t>::MAX } else { 0 });
                    want.push((a[1] ^ b[2]).wrapping_mul(3).wrapping_add(a[0] >> 3));
                    let (x0, c0) = a[0].overflowing_add(b[0]);
                    want.extend([x0, a[1].wrapping_add(b[1]).wrapping_add(c0 as $t)]);
                    want.push(a[2].wrapping_add(b[2]).wrapping_add(c0 as $t));
                    let (y0, b0) = a[0].overflowing_sub(b[0]);
                    want.extend([y0, a[1].wrapping_sub(b[1]).wrapping_sub(b0 as $t)]);
                    want.push(a[2].wrapping_sub(b[2]).wrapping_sub(b0 as $t));
                    want.extend(s);
                    want.extend(d);
                    let (s, c) = add(a, b, (t >> 2) & 1 == 1);
                    want.extend(s);
                    want.push(c as $t);
                    let (d, br) = sub(a, b, (t >> 3) & 1 == 1);
                    want.extend(d);
                    want.push(br as $t);
                    // The shaped threads' sums carry (and differences borrow)
                    // out of each of the low three words.
                    let carries = (0..3).filter(|&i| {
                        let lower = |x: &[$t; 4]| {
                            let mut w = *x;
                            w[i + 1..].fill(0);
                            w
                        };
                        add(lower(&a), lower(&b), false).0[i + 1] == 1
                    });
                    rippled[m] += carries.count();
                    let got = &out[t * CARRY_COLS..(t + 1) * CARRY_COLS];
                    assert_eq!(got, &want[..], "thread {t}: a = {a:x?}, b = {b:x?}");
                }
                assert_eq!(rippled[1], 3 * T / 4, "every shaped sum should carry out of each low word");
            }
            Case { name: stringify!($name), kernel, cfg: launch(1), small_only: false, check }
        }
    };
}

carry_words_case!(carry_words_u32, u32, i32, add_words_u32, sub_words_u32);
carry_words_case!(carry_words_u64, u64, i64, add_words_u64, sub_words_u64);

/// Products through the multiply-adds, on data spread over every bit: the
/// 128-bit product of two 64-bit numbers held as 32-bit words, from
/// `mad_cc`, `madc_cc`, `addc` and `mul_hi` (4 columns); a `mad_cc`,
/// `madc_cc`, `madc` chain on Int32 (whose low product half is the signed
/// product's, 3); the low 128 bits of a 128-bit number times a 64-bit one
/// plus a 128-bit one, from 64-bit `mad_cc`, `madc` and `mul_hi` (2); and a
/// chain on Int64 (3).
pub fn carry_multiply() -> Case {
    const N32: usize = 4 + 3;
    const N64: usize = 2 + 3;
    let kernel = Kernel::builder("carry_multiply").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::row_major((T, 4)));
        let ss = param_tensor::<i32>("ss", Layout::row_major((T, 6)));
        let ls = param_tensor::<u64>("ls", Layout::row_major((T, 5)));
        let ms = param_tensor::<i64>("ms", Layout::row_major((T, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let x: [Uint32; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as u32);
        let s: [Int32; 6] = std::array::from_fn(|i| ss.load((t, i)) * SPREAD as i32);
        let l: [Uint64; 5] = std::array::from_fn(|i| ls.load((t, i)) * SPREAD);
        let m: [Int64; 5] = std::array::from_fn(|i| ms.load((t, i)) * SPREAD as i64);
        // (a1 a0) * (b1 b0), a partial product at a time.
        let (a0, a1, b0, b1) = (x[0], x[1], x[2], x[3]);
        let (h00, h01, h10, h11) = (mul_hi(a0, b0), mul_hi(a0, b1), mul_hi(a1, b0), mul_hi(a1, b1));
        let p0 = a0 * b0;
        let p1 = mad_cc(a0, b1, h00);
        let p2 = madc_cc(a1, b1, h01);
        let p3 = addc(h11, 0u32);
        let p1 = mad_cc(a1, b0, p1);
        let p2 = addc_cc(p2, h10);
        let p3 = addc(p3, 0u32);
        let signed = [mad_cc(s[0], s[1], s[2]), madc_cc(s[3], s[4], s[5]), madc(s[0], s[4], s[2])];
        let mut narrow = vec![p0, p1, p2, p3];
        narrow.extend(signed.map(|v| v.bitcast::<u32>()));
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
        // (x1 x0) * y + (z1 z0), modulo 2^128.
        let (x0, x1, y, z0, z1) = (l[0], l[1], l[2], l[3], l[4]);
        let r0 = mad_cc(x0, y, z0);
        let r1 = madc(x1, y, z1) + mul_hi(x0, y);
        let signed = [mad_cc(m[0], m[1], m[2]), madc_cc(m[3], m[4], m[0]), madc(m[1], m[3], m[4])];
        let mut wide = vec![r0, r1];
        wide.extend(signed.map(|v| v.bitcast::<u64>()));
        for (j, v) in wide.into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ss) = (args[0].to_vec::<u32>(), args[1].to_vec::<i32>());
        let (ls, ms) = (args[2].to_vec::<u64>(), args[3].to_vec::<i64>());
        let (out32, out64) = (args[4].to_vec::<u32>(), args[5].to_vec::<u64>());
        let mut carries = 0;
        for t in 0..T {
            let x: [u32; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as u32));
            let s: [i32; 6] = std::array::from_fn(|i| ss[t * 6 + i].wrapping_mul(SPREAD as i32));
            let l: [u64; 5] = std::array::from_fn(|i| ls[t * 5 + i].wrapping_mul(SPREAD));
            let m: [i64; 5] = std::array::from_fn(|i| ms[t * 5 + i].wrapping_mul(SPREAD as i64));
            let product = (x[1] as u128) << 32 | x[0] as u128;
            let product = product * ((x[3] as u128) << 32 | x[2] as u128);
            let mut want = words_of(product).to_vec();
            // Signed: the low half of the signed product, plus the addend and
            // the carry, carrying out of 32 bits.
            let low = |p: i32, q: i32| (p as i64 * q as i64) as u32 as u64;
            let e0 = low(s[0], s[1]) + s[2] as u32 as u64;
            let e1 = low(s[3], s[4]) + s[5] as u32 as u64 + (e0 >> 32);
            let e2 = low(s[0], s[4]) + s[2] as u32 as u64 + (e1 >> 32);
            carries += (e0 >> 32) + (e1 >> 32);
            want.extend([e0 as u32, e1 as u32, e2 as u32]);
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}: x = {x:#x?}, s = {s:#x?}");
            let wide = |lo: u64, hi: u64| (hi as u128) << 64 | lo as u128;
            let r = wide(l[0], l[1]).wrapping_mul(l[2] as u128).wrapping_add(wide(l[3], l[4]));
            let mut want = vec![r as u64, (r >> 64) as u64];
            let low = |p: i64, q: i64| (p as i128 * q as i128) as u64 as u128;
            let e0 = low(m[0], m[1]) + m[2] as u64 as u128;
            let e1 = low(m[3], m[4]) + m[0] as u64 as u128 + (e0 >> 64);
            let e2 = low(m[1], m[3]) + m[4] as u64 as u128 + (e1 >> 64);
            carries += ((e0 >> 64) + (e1 >> 64)) as u64;
            want.extend([e0 as u64, e1 as u64, e2 as u64]);
            assert_eq!(out64[t * N64..(t + 1) * N64], want[..], "thread {t}: l = {l:#x?}, m = {m:#x?}");
        }
        assert!(carries > T as u64 / 2, "the multiply-add chains should carry often, not {carries} times");
    }
    Case { name: "carry_multiply", kernel, cfg: launch(1), small_only: false, check }
}

/// Whole carry chains inside control flow, which each body holds from
/// start to end: a runtime loop with a trip count that differs between
/// threads adds a 128-bit number to one accumulator and subtracts it from
/// another in every iteration (the accumulators carried through the loop),
/// and the odd threads add a third chain in an `if_` branch.
pub fn carry_in_loop() -> Case {
    const COLS: usize = 12;
    let kernel = Kernel::builder("carry_in_loop").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 4)));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let x: [Uint32; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD as u32);
        let y: [Uint32; 4] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as u32);
        let (sum, diff) = for_range_carry(0, t % 4 + 3, 1, (x, x), |i, (s, d)| {
            let z = [y[0] ^ (i.to::<u32>() * 0x9e37_79b9u32), y[1], y[2], y[3] + i.to::<u32>()];
            let s = [add_cc(s[0], z[0]), addc_cc(s[1], z[1]), addc_cc(s[2], z[2]), addc(s[3], z[3])];
            let d = [sub_cc(d[0], z[0]), subc_cc(d[1], z[1]), subc_cc(d[2], z[2]), subc(d[3], z[3])];
            (s, d)
        });
        let extra = Var::new(sum);
        if_((t & 1).eq(1), || {
            let s = extra.get();
            extra.set([add_cc(s[0], x[0]), addc_cc(s[1], x[1]), addc_cc(s[2], x[2]), addc(s[3], x[3])]);
        });
        for (j, v) in sum.into_iter().chain(diff).chain(extra.get()).enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let x: [u32; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD as u32));
            let y: [u32; 4] = std::array::from_fn(|i| ys[t * 4 + i].wrapping_mul(SPREAD as u32));
            let (mut sum, mut diff) = (u128_of(x), u128_of(x));
            for i in 0..(t % 4 + 3) as u32 {
                let z = u128_of([y[0] ^ i.wrapping_mul(0x9e37_79b9), y[1], y[2], y[3].wrapping_add(i)]);
                sum = sum.wrapping_add(z);
                diff = diff.wrapping_sub(z);
            }
            let extra = if t % 2 == 1 { sum.wrapping_add(u128_of(x)) } else { sum };
            let want: Vec<u32> = [sum, diff, extra].into_iter().flat_map(words_of).collect();
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "carry_in_loop", kernel, cfg: launch(1), small_only: false, check }
}

/// Carry chains around divisions that leave the carry flag alone: a 64-bit
/// division and remainder by constants (which LLVM turns into
/// multiplications) and a 32-bit division by a runtime value (one
/// `div.u32`), each between a carry-out and its carry-in, in the 128-bit
/// sum and difference of numbers held as two u64 words; the odd threads'
/// low words carry and borrow. (A 64-bit division by a runtime value, which
/// LLVM lowers with a branch, cannot sit inside a chain: the validator
/// rejects it.)
pub fn carry_around_division() -> Case {
    const COLS: usize = 9;
    let kernel = Kernel::builder("carry_around_division").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 2)));
        let out = param_tensor::<u64>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let (y, z) = (ys.load((t, 0)) * SPREAD as u32, ys.load((t, 1)) | 1u32);
        let low = add_cc(a0, b0);
        let quotient = a0 / 7u64;
        let remainder = b1 % 1_000_003u64;
        let narrow = (y / z).to::<u64>();
        let high = addc_cc(a1, b1);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        let d0 = sub_cc(b0, a0);
        let signed = (a1.bitcast::<i64>() / -3i64).bitcast::<u64>();
        let d1 = subc(b1, a1);
        for (j, v) in [low, high, top, quotient, remainder, narrow, d0, d1, signed].into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>(), args[2].to_vec::<u64>());
        let (mut carries, mut borrows) = (0, 0);
        for t in 0..T {
            let w: [u64; 4] = std::array::from_fn(|i| xs[t * 4 + i].wrapping_mul(SPREAD));
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w[0] }, w[1]);
            let (b0, b1) = (if odd { w[2] | 1 } else { w[2] }, w[3]);
            let (y, z) = (ys[t * 2].wrapping_mul(SPREAD as u32), ys[t * 2 + 1] | 1);
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            let want = [
                sum as u64,
                (sum >> 64) as u64,
                carry as u64,
                a0 / 7,
                b1 % 1_000_003,
                (y / z) as u64,
                difference as u64,
                (difference >> 64) as u64,
                (a1 as i64 / -3) as u64,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}: a = {a:#x}, b = {b:#x}");
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
    }
    Case { name: "carry_around_division", kernel, cfg: launch(1), small_only: false, check }
}

/// Carry chains through the warp collectives, which keep the carry flag:
/// each is one convergent PTX instruction that neither branches nor touches
/// the flag (see `cute::ir::carry_flag_breaker`), so the validator accepts a
/// chain around them and the simulator keeps the flag. Between a carry-out
/// and its carry-in sit shuffles with a runtime segment mask and clamp
/// (groups of 8 lanes below lane 16, the whole warp above), a 64-bit
/// butterfly shuffle (two `shfl.sync`), `redux.sync`, `match.any` on 32-
/// and 64-bit values and `match.all`, a ballot, a butterfly reduction over
/// groups of 8 and `sync_warp_with` per-lane half-warp masks: in the
/// 128-bit sum and difference of numbers held as two u64 words and in a
/// 96-bit sum of u32 words, whose odd threads carry (and borrow) out of
/// every word below the top one.
pub fn carry_around_collectives() -> Case {
    const N64: usize = 6;
    const N32: usize = 3 + 10;
    let kernel = Kernel::builder("carry_around_collectives").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, 6)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let t = thread_idx().x;
        let lane = lane_idx();
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let y: [Uint32; 6] = std::array::from_fn(|i| ys.load((t, i)) * SPREAD as u32);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        // A runtime mask_and_clamp: groups of 8 lanes below lane 16, the
        // whole warp above.
        let clamp = select(lane.lt(16), 0x1807i32, 31i32);
        // The 128-bit sum, with collectives between its words.
        let low = add_cc(a0, b0);
        let shuffled = shuffle_sync_op(y[0], lane + 1, FULL_MASK, clamp, ShuffleKind::Idx);
        let swapped = shuffle_sync_bfly(w[1], 3);
        let total = warp_redux_sync(y[1], ReduxKind::Add);
        let high = addc_cc(a1, b1);
        let peers = match_sync(FULL_MASK, y[2] & 3u32, MatchKind::Any);
        let wide_peers = match_sync(FULL_MASK, w[0] & 7u64, MatchKind::Any);
        let uniform = match_sync(FULL_MASK, warp_idx(), MatchKind::All);
        let ballot = vote_ballot_sync(odd);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        // The difference, around a butterfly reduction and a warp barrier.
        let d0 = sub_cc(b0, a0);
        let group_sum = warp_reduction_sum(y[3], 8);
        sync_warp_with(select(lane.lt(16), 0x0000_ffffu32, 0xffff_0000u32));
        let d1 = subc(b1, a1);
        // A 96-bit sum of u32 words, around shuffles with the runtime clamp
        // and a redux.
        let x = [select(odd, u32::MAX, y[3]), select(odd, u32::MAX, y[4]), y[5]];
        let z = [select(odd, y[0] | 1u32, y[0]), select(odd, 0u32, y[1]), y[2]];
        let s0 = add_cc(x[0], z[0]);
        let rotated = shuffle_sync_op(y[4], lane + 3, FULL_MASK, clamp, ShuffleKind::Idx);
        let s1 = addc_cc(x[1], z[1]);
        let maximum = warp_redux_sync(y[5], ReduxKind::Max);
        let down = shuffle_sync_op(y[5], 2, FULL_MASK, clamp, ShuffleKind::Down);
        let s2 = addc(x[2], z[2]);
        for (j, v) in [low, high, top, d0, d1, swapped].into_iter().enumerate() {
            out64.store((t, j), v);
        }
        let narrow =
            [s0, s1, s2, shuffled, total, peers, wide_peers, uniform, ballot, group_sum, rotated, maximum, down];
        for (j, v) in narrow.into_iter().enumerate() {
            out32.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
        let (out64, out32) = (args[2].to_vec::<u64>(), args[3].to_vec::<u32>());
        let w = |t: usize, i: usize| xs[t * 4 + i].wrapping_mul(SPREAD);
        let y = |t: usize, i: usize| ys[t * 6 + i].wrapping_mul(SPREAD as u32);
        let (mut carries, mut borrows) = (0, 0);
        for t in 0..T {
            let (warp, lane) = (t / 32 * 32, (t % 32) as u32);
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w(t, 0) }, w(t, 1));
            let (b0, b1) = (if odd { w(t, 2) | 1 } else { w(t, 2) }, w(t, 3));
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            let swapped = w(warp + (lane ^ 3) as usize, 1);
            let want =
                [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64, swapped];
            assert_eq!(out64[t * N64..(t + 1) * N64], want, "thread {t}: a = {a:#x}, b = {b:#x}");
            let clamp = if lane < 16 { 0x1807 } else { 31 };
            let read = |i: usize, source: Option<u32>| y(source.map_or(t, |s| warp + s as usize), i);
            let mask_of = |f: &dyn Fn(usize) -> bool| (0..32).filter(|&l| f(warp + l)).fold(0u32, |m, l| m | 1 << l);
            let x = [if odd { u32::MAX } else { y(t, 3) }, if odd { u32::MAX } else { y(t, 4) }, y(t, 5)];
            let z = [if odd { y(t, 0) | 1 } else { y(t, 0) }, if odd { 0 } else { y(t, 1) }, y(t, 2)];
            let words = |v: [u32; 3]| v[0] as u128 | (v[1] as u128) << 32 | (v[2] as u128) << 64;
            let s = words(x) + words(z);
            let group = warp + (lane as usize & !7);
            let want = [
                s as u32,
                (s >> 32) as u32,
                (s >> 64) as u32,
                read(0, shfl_ref(ShuffleKind::Idx, lane, lane + 1, clamp)),
                (warp..warp + 32).fold(0u32, |acc, l| acc.wrapping_add(y(l, 1))),
                mask_of(&|l| y(l, 2) & 3 == y(t, 2) & 3),
                mask_of(&|l| w(l, 0) & 7 == w(t, 0) & 7),
                u32::MAX,
                0xaaaa_aaaa,
                (group..group + 8).fold(0u32, |acc, l| acc.wrapping_add(y(l, 3))),
                read(4, shfl_ref(ShuffleKind::Idx, lane, lane + 3, clamp)),
                (warp..warp + 32).map(|l| y(l, 5)).max().unwrap(),
                read(5, shfl_ref(ShuffleKind::Down, lane, 2, clamp)),
            ];
            assert_eq!(out32[t * N32..(t + 1) * N32], want, "thread {t}");
            if odd {
                assert_eq!(s >> 32 & 0xffff_ffff, 0, "thread {t}: the middle word should carry");
            }
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
    }
    Case { name: "carry_around_collectives", kernel, cfg: launch(1), small_only: false, check }
}

// Hand-written PTX and MLIR.

/// CuTe DSL's `inline_ptx` with named operands: two write-only outputs; a
/// read-write operand beside a write-only output; a predicate that holds in
/// the odd lanes only, guarding a read-write operand (kept elsewhere) and a
/// write-only output (0 elsewhere); constant Float32 operands (`f`
/// registers, never an immediate); 16-bit and Boolean operands; integer
/// constants as immediates (`lop3`'s table, which must be one, and
/// constants LLVM prints as negative numbers: a Uint32 and a Uint64 with
/// the top bit set); a constant Boolean in a predicate register; a `$` that
/// is PTX's own; and a pointer operand that the simulation ignores.
pub fn inline_ptx_named() -> Case {
    const COLS: usize = 14;
    let kernel = Kernel::builder("inline_ptx_named").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let fs = param_tensor::<f32>("fs", Layout::col_major(T));
        let hs = param_tensor::<u16>("hs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y, f, h) = (xs.load(t) * SPREAD, ys.load(t) * SPREAD as u32, fs.load(t), hs.load(t));
        let (lo, hi) = InlinePtx::new("mov.b64 {{$w0}, {$w1}}, {$r0};")
            .write::<u32>()
            .write::<u32>()
            .read(x)
            .pure()
            .sim(|r, _| vec![r[0] & 0xffff_ffff, r[0] >> 32])
            .emit();
        let (acc, rotated) =
            InlinePtx::new("add.u32 {$rw0}, {$rw0}, {$r0};\n\tshf.l.wrap.b32 {$w0}, {$r1}, {$r1}, {$r0};")
                .read_write(y)
                .write::<u32>()
                .read(lo)
                .read(hi)
                .sim(|r, rw| {
                    rw[0] = (rw[0] + r[0]) & 0xffff_ffff;
                    vec![(r[1] as u32).rotate_left(r[0] as u32 % 32) as u64]
                })
                .emit();
        let odd = (y & 1u32).eq(1u32);
        let kept: Uint32 = InlinePtx::new("max.u32 {$rw0}, {$rw0}, {$r0};")
            .read_write(lo)
            .read(hi)
            .predicate(odd)
            .sim(|r, rw| {
                rw[0] = rw[0].max(r[0]);
                vec![]
            })
            .emit();
        let guarded: Uint32 = InlinePtx::new("xor.b32 {$w0}, {$r0}, {$r1};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .predicate(odd)
            .sim(|r, _| vec![r[0] ^ r[1]])
            .emit();
        let fused: Float32 = InlinePtx::new("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<f32>()
            .read(f)
            .read(1.5f32)
            .read(-0.25f32)
            .pure()
            .sim(|r, _| vec![f32::from_bits(r[0] as u32).mul_add(1.5, -0.25).to_bits() as u64])
            .emit();
        let narrow: Uint16 = InlinePtx::new("{ .reg .b16 t$h; add.u16 t$h, {$r0}, {$r1}; mov.b16 {$w0}, t$h; }")
            .write::<u16>()
            .read(h)
            .read(7u16)
            .sim(|r, _| vec![(r[0] + r[1]) & 0xffff])
            .emit();
        let chosen: Uint32 = InlinePtx::new("selp.b32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(odd)
            .sim(|r, _| vec![if r[2] != 0 { r[0] } else { r[1] }])
            .emit();
        let table: Uint32 = InlinePtx::new("lop3.b32 {$w0}, {$r0}, {$r1}, {$r2}, {$r3};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(acc)
            .read(0x96u32)
            .pure()
            .sim(|r, _| vec![lop3_ref(r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u8) as u64])
            .emit();
        let lowered: Uint32 = InlinePtx::new("add.u32 {$w0}, {$r0}, {$r1};")
            .write::<u32>()
            .read(lo)
            .read(0xffff_fff0u32)
            .pure()
            .sim(|r, _| vec![(r[0] + r[1]) & 0xffff_ffff])
            .emit();
        let flipped: Uint64 = InlinePtx::new("xor.b64 {$w0}, {$r0}, {$r1};")
            .write::<u64>()
            .read(x)
            .read(0x8000_0000_0000_0001u64)
            .pure()
            .sim(|r, _| vec![r[0] ^ r[1]])
            .emit();
        let never: Uint32 = InlinePtx::new("selp.b32 {$w0}, {$r0}, {$r1}, {$r2};")
            .write::<u32>()
            .read(lo)
            .read(hi)
            .read(false)
            .pure()
            .sim(|r, _| vec![if r[2] != 0 { r[0] } else { r[1] }])
            .emit();
        InlinePtx::new("prefetch.global.L2 [{$r0}];").read_ptr(&xs.ptr().offset(t)).sim(|_, _| vec![]).emit();
        let cols = [
            lo,
            hi,
            acc,
            rotated,
            kept,
            guarded,
            fused.bitcast(),
            narrow.to(),
            chosen,
            table,
            lowered,
            flipped.to(),
            (flipped >> 32u64).to(),
            never,
        ];
        for (j, v) in cols.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
        let (fs, hs, out) = (args[2].to_vec::<f32>(), args[3].to_vec::<u16>(), args[4].to_vec::<u32>());
        let odd_lanes = (0..T).filter(|&t| ys[t] & 1 == 1).count();
        assert!(odd_lanes > 0 && odd_lanes < T, "the predicate should hold in some lanes only, not {odd_lanes}");
        for t in 0..T {
            let (x, y) = (xs[t].wrapping_mul(SPREAD), ys[t].wrapping_mul(SPREAD as u32));
            let (lo, hi) = (x as u32, (x >> 32) as u32);
            let odd = y & 1 == 1;
            let want = [
                lo,
                hi,
                y.wrapping_add(lo),
                hi.rotate_left(lo % 32),
                if odd { lo.max(hi) } else { lo },
                if odd { lo ^ hi } else { 0 },
                fs[t].mul_add(1.5, -0.25).to_bits(),
                hs[t].wrapping_add(7) as u32,
                if odd { lo } else { hi },
                lo ^ hi ^ y.wrapping_add(lo),
                lo.wrapping_add(0xffff_fff0),
                (x ^ 0x8000_0000_0000_0001) as u32,
                ((x ^ 0x8000_0000_0000_0001) >> 32) as u32,
                hi,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}: x = {x:#x}, y = {y:#x}");
        }
    }
    Case { name: "inline_ptx_named", kernel, cfg: launch(1), small_only: false, check }
}

/// Raw inline assembly with constraint strings: a `+r` read-write operand
/// beside an `=r` output, an input tied to an output by number (in a string
/// written with spaces, which are dropped), an immediate (`n`) operand,
/// Float64 (`d`) operands, and a memory clobber with side effects.
pub fn inline_asm_raw() -> Case {
    const N32: usize = 3;
    const N64: usize = 2;
    let kernel = Kernel::builder("inline_asm_raw").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let ds = param_tensor::<f64>("ds", Layout::col_major(T));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, N32)));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, N64)));
        let t = thread_idx().x;
        let (x, l, d) = (xs.load(t) * SPREAD as u32, ls.load(t) * SPREAD, ds.load(t));
        let y = t.bitcast::<u32>() * 0x0101_0101u32;
        let (sum, product) = InlineAsm::new("add.u32 $0, $0, $2;\n\tmul.lo.u32 $1, $2, $2;", "+r,=r,r")
            .output::<u32>()
            .output::<u32>()
            .arg(x)
            .arg(y)
            .sim(|r, rw| {
                rw[0] = (rw[0] + r[0]) & 0xffff_ffff;
                vec![(r[0] * r[0]) & 0xffff_ffff]
            })
            .emit();
        let difference: Uint32 = InlineAsm::new("sub.u32 $0, $0, $1;", "=r, r, 0")
            .output::<u32>()
            .arg(y)
            .arg(x)
            .sim(|r, rw| {
                rw[0] = rw[0].wrapping_sub(r[0]) & 0xffff_ffff;
                vec![]
            })
            .emit();
        let shifted: Uint64 = InlineAsm::new("shl.b64 $0, $1, $2;", "=l,l,n")
            .output::<u64>()
            .arg(l)
            .arg(7u32)
            .sim(|r, _| vec![r[0] << 7])
            .emit();
        let fused: Float64 = InlineAsm::new("fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d")
            .output::<f64>()
            .arg(d)
            .arg(d)
            .arg(2.0f64)
            .side_effects()
            .memory_clobber()
            .sim(|r, _| vec![f64::from_bits(r[0]).mul_add(f64::from_bits(r[1]), 2.0).to_bits()])
            .emit();
        for (j, v) in [sum, product, difference].into_iter().enumerate() {
            out32.store((t, j), v);
        }
        for (j, v) in [shifted, fused.bitcast()].into_iter().enumerate() {
            out64.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls, ds) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>(), args[2].to_vec::<f64>());
        let (out32, out64) = (args[3].to_vec::<u32>(), args[4].to_vec::<u64>());
        for t in 0..T {
            let (x, l, d) = (xs[t].wrapping_mul(SPREAD as u32), ls[t].wrapping_mul(SPREAD), ds[t]);
            let y = (t as u32).wrapping_mul(0x0101_0101);
            let want = [x.wrapping_add(y), y.wrapping_mul(y), x.wrapping_sub(y)];
            assert_eq!(out32[t * N32..(t + 1) * N32], want[..], "thread {t}");
            let want = [l << 7, d.mul_add(d, 2.0).to_bits()];
            assert_eq!(out64[t * N64..(t + 1) * N64], want[..], "thread {t}");
        }
    }
    Case { name: "inline_asm_raw", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-written MLIR operations: LLVM's byte swap and funnel shift and
/// `arith.mului_extended` (two results), each simulated by a closure.
pub fn mlir_ops() -> Case {
    const COLS: usize = 4;
    let kernel = Kernel::builder("mlir_ops").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t) * SPREAD as u32, ys.load(t) * SPREAD as u32);
        let swapped: Uint32 = MlirOp::new("llvm.intr.bswap(%0) : (i32) -> i32")
            .operand(x)
            .result::<u32>()
            .pure()
            .sim(|r, _| vec![(r[0] as u32).swap_bytes() as u64])
            .emit();
        let (low, high) = MlirOp::new("arith.mului_extended %0, %1 : i32")
            .operand(x)
            .operand(y)
            .result::<u32>()
            .result::<u32>()
            .pure()
            .sim(|r, _| vec![(r[0] * r[1]) & 0xffff_ffff, (r[0] * r[1]) >> 32])
            .emit();
        let funnel: Uint32 = MlirOp::new("llvm.intr.fshl(%1, %0, %2) : (i32, i32, i32) -> i32")
            .operand(x)
            .operand(y)
            .operand(t.bitcast::<u32>())
            .result::<u32>()
            .sim(|r, _| vec![(((r[1] << 32 | r[0]) << (r[2] % 32)) >> 32) & 0xffff_ffff])
            .emit();
        for (j, v) in [swapped, low, high, funnel].into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let (x, y) = (xs[t].wrapping_mul(SPREAD as u32), ys[t].wrapping_mul(SPREAD as u32));
            let product = x as u64 * y as u64;
            let funnel = (((y as u64) << 32 | x as u64) << (t % 32) >> 32) as u32;
            let want = [x.swap_bytes(), product as u32, (product >> 32) as u32, funnel];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want[..], "thread {t}");
        }
    }
    Case { name: "mlir_ops", kernel, cfg: launch(1), small_only: false, check }
}

/// Inline PTX with pointer operands, which the simulator cannot follow, so
/// the GPU is held to plain Rust instead: CuTe DSL's example global load
/// (`ld.global.nc`), a load from another thread's shared memory through a
/// shared-state-space address, a global store guarded by a predicate (odd
/// lanes only: the even lanes keep the sentinel an ordinary store put
/// there first), raw assembly with a memory clobber that writes shared
/// memory between an ordinary store and an ordinary load of it, which must
/// see its write, a generic load from a per-thread array, the distance
/// between two shared addresses computed in 32 bits (which `ptxas` accepts
/// only if they are 32-bit registers, as in CuTe DSL), a global load at an
/// immediate offset (`[{$r0}+{$r1}]` with a constant), and raw assembly
/// whose memory clobber is written in its constraint string.
pub fn inline_ptx_pointers() -> Case {
    const COLS: usize = 8;
    const SENTINEL: u64 = 0x5e17_1ee1_0dd0_c0de;
    let kernel = Kernel::builder("inline_ptx_pointers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ls = param_tensor::<u64>("ls", Layout::col_major(T));
        let wide = param_tensor::<u64>("wide", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let loaded: Uint32 =
            InlinePtx::new("ld.global.nc.u32 {$w0}, [{$r0}];").write::<u32>().read_ptr(&xs.ptr().offset(t)).emit();
        let smem = alloc_smem::<u64>(T, 16);
        make_tensor(&smem, Layout::col_major(T)).store(t, ls.load(t));
        sync_threads();
        let neighbour: Uint64 = InlinePtx::new("ld.shared.u64 {$w0}, [{$r0}];")
            .write::<u64>()
            .read_ptr(&smem.offset((t + 5) % T as i32))
            .emit();
        wide.store(t, SENTINEL);
        InlinePtx::new("st.global.u64 [{$r0}], {$r1};")
            .read_ptr(&wide.ptr().offset(t))
            .read(neighbour + 1u64)
            .predicate((t & 1).eq(1))
            .emit();
        let words = alloc_smem::<u32>(T, 16);
        let tile = make_tensor(&words, Layout::col_major(T));
        tile.store(t, loaded);
        InlineAsm::new("st.shared.u32 [$0], $1;", "l,r")
            .arg_ptr(&words.offset(t))
            .arg(loaded * 3u32)
            .memory_clobber()
            .emit();
        let reread = tile.load(t);
        let table = make_rmem_tensor::<u32>(Layout::col_major(4));
        for_range(0, 4, 1, |i| table.store(i, loaded + i.to::<u32>() * 100u32));
        let picked: Uint32 =
            InlinePtx::new("ld.u32 {$w0}, [{$r0}];").write::<u32>().read_ptr(&table.ptr().offset(t % 4)).emit();
        let step: Uint32 = InlinePtx::new("sub.u32 {$w0}, {$r1}, {$r0};")
            .write::<u32>()
            .read_ptr(&smem.offset(t))
            .read_ptr(&smem.offset(t ^ 1))
            .emit();
        let next: Uint32 = InlinePtx::new("ld.global.nc.u32 {$w0}, [{$r0}+{$r1}];")
            .write::<u32>()
            .read_ptr(&xs.ptr().offset(t - (t & 1)))
            .read(4i32)
            .emit();
        let marks = alloc_smem::<u32>(T, 16);
        let marked = make_tensor(&marks, Layout::col_major(T));
        marked.store(t, loaded);
        InlineAsm::new("st.shared.u32 [$0], $1;", "r,r,~{memory}")
            .arg_ptr(&marks.offset(t))
            .arg(loaded ^ 0x55u32)
            .emit();
        let remarked = marked.load(t);
        let cols =
            [loaded, neighbour.to::<u32>(), (neighbour >> 32u64).to::<u32>(), reread, picked, step, next, remarked];
        for (j, v) in cols.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ls, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u64>(), args[3].to_vec::<u32>());
        let wide = args[2].to_vec::<u64>();
        for t in 0..T {
            let n = ls[(t + 5) % T];
            let step = ((t ^ 1) as i32 - t as i32) * 8;
            let want = [
                xs[t],
                n as u32,
                (n >> 32) as u32,
                xs[t].wrapping_mul(3),
                xs[t].wrapping_add(t as u32 % 4 * 100),
                step as u32,
                xs[t / 2 * 2 + 1],
                xs[t] ^ 0x55,
            ];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
            let stored = if t % 2 == 1 { n.wrapping_add(1) } else { SENTINEL };
            assert_eq!(wide[t], stored, "thread {t}: the predicated store");
        }
    }
    Case { name: "inline_ptx_pointers", kernel, cfg: launch(1), small_only: false, check }
}

/// CuTe DSL's example of raw MLIR, `nvvm.read.ptx.sreg.nsmid`, beside
/// cute-rs's own `nsmid()`: the GPU must give the same positive number for
/// both. The simulator cannot run the hand-written operation.
pub fn mlir_nsmid() -> Case {
    let kernel = Kernel::builder("mlir_nsmid").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let sms: Int32 = MlirOp::new("nvvm.read.ptx.sreg.nsmid : i32").result::<i32>().emit();
        out.store((t, 0), sms);
        out.store((t, 1), nsmid());
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<i32>();
        for t in 0..T {
            assert!(
                out[2 * t] > 0 && out[2 * t] == out[2 * t + 1],
                "thread {t}: {} and {}",
                out[2 * t],
                out[2 * t + 1]
            );
        }
    }
    Case { name: "mlir_nsmid", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-written MLIR operations on pointers, which the simulator cannot
/// follow, so the GPU is held to plain Rust: a global load through an
/// `!llvm.ptr<1>`, a 4-byte `nvvm.cp.async` from global into shared memory
/// (`!llvm.ptr<3>` and `!llvm.ptr<1>`) with its commit and wait written as
/// MLIR too, read back after a barrier by an ordinary load of another
/// thread's element, a global store through a pointer, and a load from a
/// per-thread array through a generic `!llvm.ptr`.
pub fn mlir_pointers() -> Case {
    const COLS: usize = 4;
    let kernel = Kernel::builder("mlir_pointers").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let loaded: Uint32 =
            MlirOp::new("llvm.load %0 : !llvm.ptr<1> -> i32").operand_ptr(&xs.ptr().offset(t)).result::<u32>().emit();
        let smem = alloc_smem::<u32>(T, 16);
        MlirOp::new("nvvm.cp.async.shared.global %0, %1, 4, cache = ca : !llvm.ptr<3>, !llvm.ptr<1>")
            .operand_ptr(&smem.offset(t))
            .operand_ptr(&ys.ptr().offset(t))
            .emit();
        MlirOp::new("nvvm.cp.async.commit.group").emit();
        MlirOp::new("nvvm.cp.async.wait.group 0").emit();
        sync_threads();
        let neighbour = make_tensor(&smem, Layout::col_major(T)).load((t + 5) % T as i32);
        MlirOp::new("llvm.store %1, %0 : i32, !llvm.ptr<1>")
            .operand_ptr(&out.ptr().offset(t * COLS as i32 + 2))
            .operand(loaded ^ 0x5a5a_5a5au32)
            .emit();
        let table = make_rmem_tensor::<u32>(Layout::col_major(4));
        for_range(0, 4, 1, |i| table.store(i, loaded + i.to::<u32>() * 100u32));
        let picked: Uint32 = MlirOp::new("llvm.load %0 : !llvm.ptr -> i32")
            .operand_ptr(&table.ptr().offset(t % 4))
            .result::<u32>()
            .emit();
        out.store((t, 0), loaded);
        out.store((t, 1), neighbour);
        out.store((t, 3), picked);
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let want = [xs[t], ys[(t + 5) % T], xs[t] ^ 0x5a5a_5a5a, xs[t].wrapping_add(t as u32 % 4 * 100)];
            assert_eq!(out[t * COLS..(t + 1) * COLS], want, "thread {t}");
        }
    }
    Case { name: "mlir_pointers", kernel, cfg: launch(1), small_only: false, check }
}

/// What the GPU does with a carry flag that one kind of instruction sets
/// and the other reads, which the validator rejects for the carry
/// instructions and which inline PTX holding the whole chain can still
/// write. The PTX ISA describes one flag for carries and borrows, which
/// would make `subc` after `add.cc` subtract the carry and `addc` after
/// `sub.cc` add the borrow; the GPU (an SM89 with `ptxas` 12.8) keeps a
/// borrow as the inverted carry instead, so they subtract one minus the
/// carry and add one minus the borrow. This holds the GPU to that, so that a
/// GPU or `ptxas` that changes it shows here.
pub fn mixed_carry_flag() -> Case {
    let kernel = Kernel::builder("mixed_carry_flag").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 2)));
        let t = thread_idx().x;
        let (x, y, flag) = (xs.load(t) * SPREAD as u32, ys.load(t) * SPREAD as u32, (t & 1).bitcast::<u32>());
        // add.cc carries exactly when flag is 1; sub.cc borrows exactly then.
        let after_add: Uint32 =
            InlinePtx::new("{ .reg .u32 t; add.cc.u32 t, {$r0}, 0xffffffff; subc.u32 {$w0}, {$r1}, {$r2}; }")
                .write::<u32>()
                .read(flag)
                .read(x)
                .read(y)
                .emit();
        let after_sub: Uint32 =
            InlinePtx::new("{ .reg .u32 t; sub.cc.u32 t, 0, {$r0}; addc.u32 {$w0}, {$r1}, {$r2}; }")
                .write::<u32>()
                .read(flag)
                .read(x)
                .read(y)
                .emit();
        out.store((t, 0), after_add);
        out.store((t, 1), after_sub);
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for t in 0..T {
            let (x, y, flag) = (xs[t].wrapping_mul(SPREAD as u32), ys[t].wrapping_mul(SPREAD as u32), t as u32 & 1);
            let want = [x.wrapping_sub(y).wrapping_sub(1 - flag), x.wrapping_add(y).wrapping_add(1 - flag)];
            assert_eq!(out[2 * t..2 * t + 2], want, "thread {t}: the flag read across kinds (x = {x:#x}, y = {y:#x})");
        }
    }
    Case { name: "mixed_carry_flag", kernel, cfg: launch(1), small_only: false, check }
}

/// Kernels the simulator cannot run (their inline PTX or hand-written MLIR
/// follows pointers, their inline PTX reads the carry flag across kinds, or
/// they read hardware state through hand-written MLIR): compiled, and held
/// to plain Rust on the GPU.
pub fn gpu_only() -> Vec<Case> {
    vec![inline_ptx_pointers(), mlir_nsmid(), mixed_carry_flag(), mlir_pointers()]
}

/// The bytes [`dynamic_region`] gives its dynamic shared-memory region: not
/// a multiple of 128, so the shared-memory size registers round.
pub const DYNAMIC_REGION_BYTES: u32 = 4108;

/// Where [`dynamic_region_kernel`]'s dynamic region starts: after its
/// static buffers' 100 + 128 bytes, at a multiple of 16.
pub const DYNAMIC_REGION_START: u32 = 240;

/// Shared buffers sized at build time in front of the dynamic region
/// ([`get_dyn_smem`]), whose size the launch sets: each thread stores into
/// the static buffers and at both ends of the region (as far apart as the
/// region's bytes, read with `get_dyn_smem_size`, allow), and after a barrier
/// reads what the other warp stored, a byte of the region through a second
/// pointer to it, and the shared-memory size registers. The region must hold
/// at least 128 words.
pub fn dynamic_region_kernel() -> Kernel {
    Kernel::builder("dynamic_region").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, 8)));
        let t = thread_idx().x;
        let x = xs.load(t);
        // 100 bytes, then 128 bytes of halves from byte 100.
        let words = make_tensor(&alloc_smem::<u32>(25, 4), Layout::col_major(25));
        let halves = make_tensor(&alloc_smem::<u16>(T, 2), Layout::col_major(T));
        let region = get_dyn_smem::<u32>(None);
        let last = get_dyn_smem_size() / 4 - 1;
        if_(t.lt(25), || words.store(t, x ^ 0x5a5a_5a5au32));
        halves.store(t, x.to::<u16>());
        region.offset(t).store(x * 3u32);
        region.offset(last - t).store(x + 7u32);
        sync_threads();
        let o = (t + 32) % 64;
        let row = [
            region.offset(o).load(),
            region.offset(last - o).load(),
            words.load(o % 25),
            halves.load(o).to::<u32>(),
            get_dyn_smem::<u8>(None).offset(o * 4).load().to::<u32>(),
            dynamic_smem_size().bitcast::<u32>(),
            get_dyn_smem_size().bitcast::<u32>(),
            total_smem_size().bitcast::<u32>(),
        ];
        for (j, v) in row.into_iter().enumerate() {
            out.store((t, j), v);
        }
    })
}

/// The plain-Rust result of [`dynamic_region_kernel`] with a region of
/// `region_bytes`: `dynamic_smem_size` and `get_dyn_smem_size` read the
/// region's bytes, and `total_smem_size` the block's shared memory (the
/// static buffers live in the dynamic window too, in front of the region)
/// rounded up to 128 bytes, as an SM89 GPU reports it.
pub fn check_dynamic_region(args: &[KernelArg], region_bytes: u32) {
    let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
    let window = DYNAMIC_REGION_START + region_bytes;
    for t in 0..T {
        let o = (t + 32) % 64;
        let want = [
            xs[o].wrapping_mul(3),
            xs[o].wrapping_add(7),
            xs[o % 25] ^ 0x5a5a_5a5a,
            xs[o] & 0xffff,
            xs[o].wrapping_mul(3) & 0xff,
            region_bytes,
            region_bytes,
            window.next_multiple_of(128),
        ];
        assert_eq!(out[t * 8..t * 8 + 8], want, "thread {t} (a region of {region_bytes} bytes)");
    }
}

/// [`dynamic_region_kernel`] with a region of [`DYNAMIC_REGION_BYTES`].
pub fn dynamic_region() -> Case {
    let cfg = launch(1).with_dynamic_smem(DYNAMIC_REGION_BYTES);
    let check = |args: &[KernelArg]| check_dynamic_region(args, DYNAMIC_REGION_BYTES);
    Case { name: "dynamic_region", kernel: dynamic_region_kernel(), cfg, small_only: false, check }
}

/// Blocks [`cluster_ranks`] launches at most.
const CLUSTER_BLOCKS: usize = 8;

/// CuTe DSL's linear rank in the cluster and cluster size (`block_idx_in_cluster`,
/// `cluster_size`), against the 3-D index and shape they linearize (x fastest), and a block's
/// shared-memory size registers, `aggr_smem_size` (sm_90) among them. Each block of one warp
/// passes its rank around its 128 bytes of shared memory.
pub fn cluster_ranks_kernel() -> Kernel {
    Kernel::builder("cluster_ranks").block_dim([32, 1, 1]).build(|| {
        let out = param_tensor::<i32>("out", Layout::row_major((CLUSTER_BLOCKS * 32, 7)));
        let t = thread_idx().x;
        let (b, g) = (block_idx(), grid_dim());
        let row = (b.x + b.y * g.x + b.z * g.x * g.y) * 32 + t;
        let (idx, dim) = (block_in_cluster_idx(), block_in_cluster_dim());
        let rank = block_idx_in_cluster();
        let words = make_tensor(&alloc_smem::<i32>(32, 4), Layout::col_major(32));
        words.store(t, rank * 100 + t);
        sync_threads();
        let results = [
            rank,
            cluster_size(),
            idx.x + idx.y * dim.x + idx.z * dim.x * dim.y,
            dim.x * dim.y * dim.z,
            words.load((t + 1) % 32),
            total_smem_size(),
            aggr_smem_size(),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((row, j), v);
        }
    })
}

/// The plain-Rust result of [`cluster_ranks_kernel`] on a `grid` of blocks in clusters of
/// `cluster`: 128 bytes of shared memory, plus the 1 KB the system reserves for
/// `%aggr_smem_size` (from the PTX ISA and the CUDA guide; not checked on an SM90 GPU).
fn check_cluster_ranks(args: &[KernelArg], grid: [usize; 3], cluster: [usize; 3]) {
    let out = args[0].to_vec::<i32>();
    for (z, y, x) in (0..grid[2]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[0]).map(move |x| (z, y, x)))) {
        let block = x + y * grid[0] + z * grid[0] * grid[1];
        let rank = (x % cluster[0] + (y % cluster[1]) * cluster[0] + (z % cluster[2]) * cluster[0] * cluster[1]) as i32;
        let size = (cluster[0] * cluster[1] * cluster[2]) as i32;
        for t in 0..32 {
            let row = block * 32 + t;
            let want = [rank, size, rank, size, rank * 100 + (t as i32 + 1) % 32, 128, 128 + 1024];
            assert_eq!(out[row * 7..row * 7 + 7], want, "block ({x},{y},{z}) thread {t}");
        }
    }
}

/// [`cluster_ranks_kernel`] on 4 blocks in clusters of 2.
pub fn cluster_ranks() -> Case {
    let cfg = LaunchConfig::new([4, 1, 1], [32, 1, 1]).with_cluster([2, 1, 1]);
    let check = |args: &[KernelArg]| check_cluster_ranks(args, [4, 1, 1], [2, 1, 1]);
    Case { name: "cluster_ranks", kernel: cluster_ranks_kernel(), cfg, small_only: false, check }
}

/// [`cluster_ranks_kernel`] on 2 x 4 blocks in clusters of 2 x 2.
pub fn cluster_ranks_2x2() -> Case {
    let cfg = LaunchConfig::new([2, 4, 1], [32, 1, 1]).with_cluster([2, 2, 1]);
    let check = |args: &[KernelArg]| check_cluster_ranks(args, [2, 4, 1], [2, 2, 1]);
    Case { name: "cluster_ranks_2x2", kernel: cluster_ranks_kernel(), cfg, small_only: false, check }
}

/// Words [`cp_async_mbarrier`] hands over each time: 16 bytes per producer lane.
const HANDOFF_WORDS: usize = 128;

/// Polls of an mbarrier phase after which [`bounded_spin`] gives up: far more than a phase that
/// completes needs (a spin that never succeeds gives up after 0.09 s at O3 and 1 s at O0 on an
/// SM89 GPU), so that only a lost or miscounted arrival gives up, which then fails the test
/// instead of hanging the GPU.
pub const SPIN_LIMIT: i32 = 1 << 22;

/// Spin on `mbarrier_test_wait(bar, parity)` until it succeeds or [`SPIN_LIMIT`] polls have
/// failed; returns whether it gave up. (In the simulator the spinning warp polls once per
/// scheduling round until the arrivals are in.)
fn bounded_spin(bar: &Ptr<u64>, parity: Int32) -> Boolean {
    let (_, done) = while_carry(
        (Int32::from(0), Boolean::from(false)),
        |(polls, done)| !done & polls.lt(SPIN_LIMIT),
        |(polls, _)| (polls + 1, mbarrier_test_wait(bar, parity)),
    );
    !done
}

/// Check the waits of each thread that [`bounded_spin`] counted as given up.
fn check_no_spin_gave_up(gave_up: &[u32]) {
    for (t, &n) in gave_up.iter().enumerate() {
        assert_eq!(n, 0, "thread {t} gave up waiting on {n} mbarrier phase(s) after {SPIN_LIMIT} polls each");
    }
}

/// Hand-offs from a producer warp to a consumer warp, each through an mbarrier of its own that
/// counts 32 arrivals, with the producer's `cp.async` copies tracked by `cp.async.mbarrier.arrive`
/// (CuTe DSL's `experimental/primitives/cp_async_mbarrier_arrive_cta.py`; the producer comes
/// first in program order). Each time, every producer lane copies 16 bytes into a fresh
/// buffer. The first hand-off tracks them with
/// `cp_async_mbarrier_arrive_noinc`, whose 32 arrivals are the count; the second with
/// `cp_async_mbarrier_arrive`, which raises the count by one first, followed by an ordinary
/// `mbarrier_arrive` per lane; the third like the first, but the producer then also waits for
/// its copies (`cp_async_wait_all`) and reads them back, so they land before the mbarrier sees
/// them. The consumer warp spins on `mbarrier_test_wait` for each and reads the copies.
///
/// Each mbarrier completes one phase. The producer never waits for the consumer, so on a GPU it
/// can finish every hand-off before the consumer first polls: with one mbarrier for all three,
/// its phase would be two ahead of the consumer's parity, which then names a phase that never
/// completes (the consumer would spin forever). [`cp_async_pipeline`] reuses its mbarriers,
/// behind an `empty` mbarrier that keeps the producer at most one phase ahead. The spins give up
/// after [`SPIN_LIMIT`] polls, and each thread stores how many did in `gave_up`.
pub fn cp_async_mbarrier() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_mbarrier").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(3 * W));
        let out = param_tensor::<u32>("out", Layout::col_major(4 * W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bars = alloc_smem::<u64>(3, 8);
        let stages: Vec<Tensor<u32>> = (0..3).map(|_| make_smem_tensor::<u32>(Layout::col_major(W))).collect();
        if_(t.eq(0), || {
            for k in 0..3usize {
                mbarrier_init(&bars.offset(k), 32);
            }
        });
        sync_threads();
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        // Each lane's own 4 words of a stage, into `out` from `at`.
        let read_back = |stage: &Tensor<u32>, at: usize| {
            for k in 0..4 {
                out.store(lane * 4 + k + at as i32, stage.load(lane * 4 + k));
            }
        };
        let misses = Var::new(Uint32::from(0u32));
        for (handoff, stage) in stages.iter().enumerate() {
            let bar = bars.offset(handoff);
            let source = make_tensor(&src.ptr().offset(handoff * W), Layout::col_major(W));
            if_else(
                warp_idx().eq(0),
                || {
                    let thr = tiled.get_slice(lane);
                    copy(&tiled, &thr.partition_s(&source), &thr.partition_d(stage));
                    match handoff {
                        0 => cp_async_mbarrier_arrive_noinc(&bar),
                        1 => {
                            cp_async_mbarrier_arrive(&bar);
                            mbarrier_arrive(&bar);
                        }
                        _ => {
                            cp_async_commit_group();
                            cp_async_mbarrier_arrive_noinc(&bar);
                            cp_async_wait_all();
                            read_back(stage, 3 * W);
                        }
                    }
                },
                || {
                    let missed = bounded_spin(&bar, Int32::from(0));
                    misses.update(|m| m + select(missed, 1u32, 0u32));
                    read_back(stage, handoff * W);
                },
            );
        }
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for i in 0..3 * W {
            assert_eq!(out[i], src[i], "hand-off {} word {} at the consumer", i / W, i % W);
        }
        for i in 0..W {
            assert_eq!(out[3 * W + i], src[2 * W + i], "hand-off 2 word {i} at the producer");
        }
    }
    Case { name: "cp_async_mbarrier", kernel, cfg: launch(1), small_only: false, check }
}

/// A hand-off whose consumer warp comes first in program order (and so
/// polls first in the simulator's schedules): warp 0 spins on
/// `mbarrier_test_wait` until warp 1's `cp.async` copies, tracked by
/// `cp_async_mbarrier_arrive_noinc`, have landed, then reads them. The
/// mbarrier counts the 32 deferred arrivals, which fire when each lane's
/// copies complete, after the consumer started waiting.
pub fn cp_async_consumer_first() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_consumer_first").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(W));
        let out = param_tensor::<u32>("out", Layout::col_major(W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let stage = make_smem_tensor::<u32>(Layout::col_major(W));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_spin(&bar, Int32::from(0));
                for k in 0..4 {
                    out.store(lane * 4 + k, stage.load(lane * 4 + k));
                }
                select(missed, 1u32, 0u32)
            },
            || {
                let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
                let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
                let thr = tiled.get_slice(lane);
                let source = make_tensor(&src.ptr(), Layout::col_major(W));
                copy(&tiled, &thr.partition_s(&source), &thr.partition_d(&stage));
                cp_async_mbarrier_arrive_noinc(&bar);
                // Done with the copies before exiting, as CUDA requires.
                cp_async_wait_all();
                Uint32::from(0u32)
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        assert_eq!(out, src, "the consumer reads what the producer copied");
    }
    Case { name: "cp_async_consumer_first", kernel, cfg: launch(1), small_only: false, check }
}

/// A warp-specialised bulk-copy pipeline: warp 0 produces, filling two
/// stages with `cp_async_bulk` behind `full` mbarriers (with
/// `mbarrier_arrive_and_expect_tx`), and warps 1 to 3 consume, each summing
/// its part of every stage and arriving on the stage's `empty` mbarrier,
/// which the producer waits on before refilling it.
pub fn bulk_copy_producer_warp() -> Case {
    const WORDS: usize = 96;
    const ROUNDS: usize = 6;
    let kernel = Kernel::builder("bulk_copy_producer_warp").block_dim([128, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(ROUNDS * WORDS));
        let out = param_tensor::<u32>("out", Layout::col_major(96));
        let t = thread_idx().x;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = alloc_smem::<u32>(2 * WORDS, 128);
        if_(t.eq(0), || {
            for k in 0..2usize {
                mbarrier_init(&full.offset(k), 1);
                mbarrier_init(&empty.offset(k), 96);
            }
            mbarrier_init_fence();
        });
        sync_threads();
        if_else(
            warp_idx().eq(0),
            || {
                for_range(0, ROUNDS as i32, 1, |r| {
                    let (stage, parity) = (r % 2, (r / 2) % 2);
                    // The consumers must have emptied the stage first.
                    if_(r.ge(2), || mbarrier_wait(&empty.offset(stage), 1 - parity));
                    if_(t.eq(0), || {
                        let bar = full.offset(stage);
                        mbarrier_arrive_and_expect_tx(&bar, (WORDS * 4) as i32);
                        cp_async_bulk(
                            &src.ptr().offset(r * WORDS as i32),
                            &stages.offset(stage * WORDS as i32),
                            (WORDS * 4) as i32,
                            &bar,
                        );
                    });
                });
            },
            || {
                let me = t - 32;
                let sum = for_range_carry(0, ROUNDS as i32, 1, Uint32::from(0u32), |r, sum| {
                    let (stage, parity) = (r % 2, (r / 2) % 2);
                    mbarrier_wait(&full.offset(stage), parity);
                    let word = make_tensor(&stages.offset(stage * WORDS as i32), Layout::col_major(WORDS)).load(me);
                    mbarrier_arrive(&empty.offset(stage));
                    sum + word
                });
                out.store(me, sum);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (i, &got) in out.iter().enumerate() {
            let want = (0..ROUNDS).fold(0u32, |acc, r| acc.wrapping_add(src[r * WORDS + i]));
            assert_eq!(got, want, "word {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    Case { name: "bulk_copy_producer_warp", kernel, cfg, small_only: false, check }
}

/// Iterations of [`cp_async_pipeline`]'s loop: each of its two stages is filled three times.
const PIPELINE_ITERS: usize = 6;

/// A two-stage producer/consumer pipeline in a runtime loop (the pattern of CuTe DSL's
/// `PipelineCpAsync`): each iteration, the producer warp waits until the consumer has emptied the
/// stage (its `empty` mbarrier), refills it with `cp.async` and makes the stage's `full` mbarrier
/// track the copies (`cp_async_mbarrier_arrive_noinc`); the consumer warp waits on `full`, adds
/// the stage into running sums kept in a `Var`, and arrives on `empty`. Phase parities flip every
/// second iteration, and each stage's buffer is reused only after the consumer has read it. The
/// spins give up after [`SPIN_LIMIT`] polls, and each thread stores how many did in `gave_up`.
pub fn cp_async_pipeline() -> Case {
    const W: usize = HANDOFF_WORDS;
    let kernel = Kernel::builder("cp_async_pipeline").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(PIPELINE_ITERS * W));
        let out = param_tensor::<u32>("out", Layout::col_major(W));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = alloc_smem::<u32>(2 * W, 128);
        if_(t.eq(0), || {
            for s in 0..2usize {
                mbarrier_init(&full.offset(s), 32);
                mbarrier_init(&empty.offset(s), 32);
            }
        });
        sync_threads();
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(32), &Layout::col_major(4));
        let sums = Var::new([Uint32::from(0u32); 4]);
        let misses = Var::new(Uint32::from(0u32));
        let spin = |bar: Ptr<u64>, parity: Int32| {
            let missed = bounded_spin(&bar, parity);
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        for_range(0, PIPELINE_ITERS as i32, 1, |i| {
            let stage = i % 2;
            let parity = (i / 2) % 2;
            // Offsets as index expressions, which know they are multiples of W.
            let buffer = make_tensor(&stages.offset(IndexExpr::from(stage) * W), Layout::col_major(W));
            if_else(
                warp_idx().eq(0),
                || {
                    // A fresh mbarrier counts the phase before its first as complete.
                    spin(empty.offset(stage), 1 - parity);
                    let source = make_tensor(&src.ptr().offset(IndexExpr::from(i) * W), Layout::col_major(W));
                    let thr = tiled.get_slice(lane);
                    copy(&tiled, &thr.partition_s(&source), &thr.partition_d(&buffer));
                    cp_async_mbarrier_arrive_noinc(&full.offset(stage));
                },
                || {
                    spin(full.offset(stage), parity);
                    sums.update(|s| std::array::from_fn(|k| s[k] + buffer.load(lane * 4 + k as i32)));
                    mbarrier_arrive(&empty.offset(stage));
                },
            );
        });
        if_(warp_idx().eq(1), || {
            let s = sums.get();
            for (k, v) in s.into_iter().enumerate() {
                out.store(lane * 4 + k as i32, v);
            }
        });
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for i in 0..W {
            let want = (0..PIPELINE_ITERS).fold(0u32, |acc, it| acc.wrapping_add(src[it * W + i]));
            assert_eq!(out[i], want, "word {i}");
        }
    }
    Case { name: "cp_async_pipeline", kernel, cfg: launch(1), small_only: false, check }
}

// Carry chains through the special-register reads and cp.async.mbarrier.arrive.

/// Output rows of [`carry_around_smem_kernel`]: one per thread of its two blocks.
const SMEM_CARRY_ROWS: usize = 2 * T;

/// Words of each of [`carry_around_smem_kernel`]'s copies: 16 bytes per thread.
const SMEM_CARRY_WORDS: usize = 4 * T;

/// The bytes [`carry_around_smem_kernel`]'s launches give its dynamic region: one copy's.
const SMEM_CARRY_REGION_BYTES: u32 = 4 * SMEM_CARRY_WORDS as u32;

/// Where [`carry_around_smem_kernel`]'s dynamic region starts: after its two mbarriers and its
/// buffer sized at build time (16 + 1024 bytes).
pub const SMEM_CARRY_REGION_START: u32 = 16 + 4 * SMEM_CARRY_WORDS as u32;

/// Words per row of [`carry_around_smem_kernel`]'s `out32`: the three size reads, four words of
/// each copy and, with `cluster`, the block's rank, the cluster's size and `aggr_smem_size`.
fn smem_carry_columns(cluster: bool) -> usize {
    3 + 4 + 4 + if cluster { 3 } else { 0 }
}

/// Carry chains through the special-register reads and `cp.async.mbarrier.arrive`, which keep
/// the carry flag like the warp collectives (see `cute::ir::carry_flag_breaker`), so the
/// validator accepts a chain around them and the simulator keeps the flag. Two mbarriers and a
/// 1 KB buffer sized at build time come first in shared memory, and the dynamic region follows
/// them (at [`SMEM_CARRY_REGION_START`]). Each thread copies 16 bytes into the buffer with
/// `cp.async`, and then 16 bytes into the region. Between a carry-out and its carry-in sit
/// `dynamic_smem_size` (a `mov` and a subtraction of the region's start), `get_dyn_smem_size`
/// (the same read, which the builder merges with it), `total_smem_size` and
/// `cp_async_mbarrier_arrive_noinc`, which makes the first mbarrier track the first copy, in the
/// 128-bit sum of numbers held as two u64 words; and `cp_async_mbarrier_arrive`, which makes the
/// second track both copies, in their difference, with `block_idx_in_cluster`, `cluster_size` and
/// `aggr_smem_size` (sm_90) too when `cluster` is set. The odd threads carry and borrow out of the
/// low word. After the chains each thread arrives on the second mbarrier, waits on both (giving
/// up after [`SPIN_LIMIT`] polls, counted in `gave_up`) and reads the other warp's copies.
pub fn carry_around_smem_kernel(cluster: bool) -> Kernel {
    const W: usize = SMEM_CARRY_WORDS;
    let name = if cluster { "carry_around_smem_registers_sm90" } else { "carry_around_smem_registers" };
    Kernel::builder(name).block_dim([T as u32, 1, 1]).build(move || {
        let xs = param_tensor::<u64>("xs", Layout::row_major((SMEM_CARRY_ROWS, 4)));
        let src = param_tensor::<u32>("src", Layout::col_major(2 * W));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((SMEM_CARRY_ROWS, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((SMEM_CARRY_ROWS, smem_carry_columns(cluster))));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(SMEM_CARRY_ROWS));
        let t = thread_idx().x;
        let row = block_idx().x * T as i32 + t;
        let bars = alloc_smem::<u64>(2, 8);
        let (full, tracked) = (bars.clone(), bars.offset(1usize));
        let stage = make_tensor(&alloc_smem::<u32>(W, 16), Layout::col_major(W));
        let region = make_tensor(&get_dyn_smem::<u32>(None), Layout::col_major(W));
        if_(t.eq(0), || {
            mbarrier_init(&full, T as i32);
            mbarrier_init(&tracked, T as i32);
        });
        sync_threads();
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((row, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let atom = make_copy_atom::<u32>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
        let tiled = make_tiled_copy_tv(&atom, &Layout::col_major(T), &Layout::col_major(4));
        let thr = tiled.get_slice(t);
        let source = |half: usize| make_tensor(&src.ptr().offset(half * W), Layout::col_major(W));
        copy(&tiled, &thr.partition_s(&source(0)), &thr.partition_d(&stage));
        // The sum, around the size registers and the first copy's arrival.
        let low = add_cc(a0, b0);
        let dynamic = dynamic_smem_size();
        cp_async_mbarrier_arrive_noinc(&full);
        let region_bytes = get_dyn_smem_size();
        let high = addc_cc(a1, b1);
        let total = total_smem_size();
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        copy(&tiled, &thr.partition_s(&source(1)), &thr.partition_d(&region));
        // The difference, around the arrival that tracks both copies (and the cluster registers).
        let d0 = sub_cc(b0, a0);
        cp_async_mbarrier_arrive(&tracked);
        let cluster_registers = cluster.then(|| [block_idx_in_cluster(), cluster_size(), aggr_smem_size()]);
        let d1 = subc(b1, a1);
        mbarrier_arrive(&tracked);
        let misses = Var::new(Uint32::from(0u32));
        for bar in [&full, &tracked] {
            let missed = bounded_spin(bar, Int32::from(0));
            misses.update(|m| m + select(missed, 1u32, 0u32));
        }
        gave_up.store(row, misses.get());
        for (j, v) in [low, high, top, d0, d1].into_iter().enumerate() {
            out64.store((row, j), v);
        }
        // The other warp's copies.
        let o = (t + 32) % 64;
        let mut words = vec![dynamic.bitcast::<u32>(), region_bytes.bitcast::<u32>(), total.bitcast::<u32>()];
        words.extend((0..4).map(|k| stage.load(o * 4 + k)));
        words.extend((0..4).map(|k| region.load(o * 4 + k)));
        words.extend(cluster_registers.into_iter().flatten().map(|v| v.bitcast::<u32>()));
        for (j, v) in words.into_iter().enumerate() {
            out32.store((row, j), v);
        }
    })
}

/// The plain-Rust result of [`carry_around_smem_kernel`] (in clusters of 2 with `cluster`): the
/// sum and difference in u128 arithmetic, the region's bytes, the block's shared memory
/// rounded up to 128 bytes (and 1 KB more for `aggr_smem_size`, as in
/// [`check_cluster_ranks`]), the copied words and, with `cluster`, the block's rank and the
/// cluster's size.
fn check_carry_around_smem(args: &[KernelArg], cluster: bool) {
    let (xs, src) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>());
    let (out64, out32) = (args[2].to_vec::<u64>(), args[3].to_vec::<u32>());
    check_no_spin_gave_up(&args[4].to_vec::<u32>());
    let n32 = smem_carry_columns(cluster);
    let total = (SMEM_CARRY_REGION_START + SMEM_CARRY_REGION_BYTES).next_multiple_of(128);
    let (mut carries, mut borrows, mut top_carries) = (0, 0, 0);
    for r in 0..SMEM_CARRY_ROWS {
        let (block, t) = (r / T, r % T);
        let w = |i: usize| xs[r * 4 + i].wrapping_mul(SPREAD);
        let odd = t % 2 == 1;
        let (a0, a1) = (if odd { u64::MAX } else { w(0) }, w(1));
        let (b0, b1) = (if odd { w(2) | 1 } else { w(2) }, w(3));
        let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
        let (sum, carry) = a.overflowing_add(b);
        let difference = b.wrapping_sub(a);
        carries += a0.overflowing_add(b0).1 as usize;
        borrows += (b0 < a0) as usize;
        top_carries += carry as usize;
        let want = [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64];
        assert_eq!(out64[r * 5..r * 5 + 5], want, "row {r}: a = {a:#x}, b = {b:#x}");
        let o = (t + 32) % T;
        let mut want = vec![SMEM_CARRY_REGION_BYTES, SMEM_CARRY_REGION_BYTES, total];
        want.extend(&src[o * 4..o * 4 + 4]);
        want.extend(&src[SMEM_CARRY_WORDS + o * 4..SMEM_CARRY_WORDS + o * 4 + 4]);
        if cluster {
            // `%aggr_smem_size` is not rounded to the allocation unit (a B200 gives this).
            want.extend([block as u32 % 2, 2, SMEM_CARRY_REGION_START + SMEM_CARRY_REGION_BYTES + 1024]);
        }
        assert_eq!(out32[r * n32..(r + 1) * n32], want[..], "row {r}");
    }
    assert!(
        carries >= SMEM_CARRY_ROWS / 2 && borrows >= SMEM_CARRY_ROWS / 2,
        "the odd threads should carry and borrow: {carries}, {borrows}"
    );
    // The carry out of the high word is the one that crosses total_smem_size: keep it exercised.
    assert!(top_carries > 0, "no row carries out of the high word, so the chain through total_smem_size goes untested");
}

/// [`carry_around_smem_kernel`] on two blocks with a region of [`SMEM_CARRY_REGION_BYTES`].
pub fn carry_around_smem_registers() -> Case {
    let cfg = launch(2).with_dynamic_smem(SMEM_CARRY_REGION_BYTES);
    let check = |args: &[KernelArg]| check_carry_around_smem(args, false);
    Case { name: "carry_around_smem_registers", kernel: carry_around_smem_kernel(false), cfg, small_only: false, check }
}

/// [`carry_around_smem_kernel`] with the cluster registers and `aggr_smem_size` (sm_90), on two
/// blocks in a cluster of 2.
pub fn carry_around_smem_registers_sm90() -> Case {
    let cfg = launch(2).with_cluster([2, 1, 1]).with_dynamic_smem(SMEM_CARRY_REGION_BYTES);
    let check = |args: &[KernelArg]| check_carry_around_smem(args, true);
    let kernel = carry_around_smem_kernel(true);
    Case { name: "carry_around_smem_registers_sm90", kernel, cfg, small_only: false, check }
}

// ---------------------------------------------------------------------------
// Memory-ordered and cache-hinted loads and stores, prefetch, the G2R and R2G
// copy atoms with memory attributes, make_cotiled_copy.
// ---------------------------------------------------------------------------

/// What lane `l` of exchange `k` of [`ordered_handoffs`] hands over, from
/// the input `x` of its thread.
pub fn handoff_value(k: u32, x: u32) -> u32 {
    x.wrapping_mul(2 * k + 3).wrapping_add(k << 20) ^ (k * 0x1111_1111)
}

/// Seven hand-offs of shared-memory data between the four warps of a block,
/// each through a flag that the producer sets after writing its data and the
/// consumer spins on before reading it, in every ordered form: a release
/// store and an acquire load on shared memory (CTA scope) and on global
/// memory (GPU scope), `fence; st.relaxed` and `ld.relaxed; fence`, seq_cst
/// accesses, volatile ones between fences, a vector release and acquire, and
/// a release and acquire with L1 eviction priorities (inline PTX). Each
/// producer comes first in program order; the simulator's race checker
/// must accept every access.
pub fn ordered_handoffs() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("ordered_handoffs").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * N));
        let gflags = param_tensor::<u32>("gflags", Layout::col_major(2 * 8));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * N, 2)));
        let t = thread_idx().x;
        let b = block_idx().x;
        let row = b * N as i32 + t;
        let x = xs.load(row);
        let warp = warp_idx();
        let lane = t % 32;
        let data = make_smem_tensor::<u32>(Layout::col_major(7 * 32));
        let sflags = alloc_smem::<u32>(4, 16);
        let sflag = |k: usize| sflags.offset(k);
        let gflag = |k: usize| gflags.ptr().offset(IndexExpr::from(b) * 8 + k);
        let (cta, gpu) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu));
        let ld = |sem, scope| LoadOptions { sem, scope, ..Default::default() };
        let st = |sem, scope| StoreOptions { sem, scope, ..Default::default() };
        let spin = |flag: &Ptr<u32>, opts: LoadOptions| {
            while_carry((), |()| load(flag, opts).eq(0u32), |()| ());
        };
        let value = |k: u32| (x * (2 * k + 3) + (k << 20)) ^ (k * 0x1111_1111);
        // Every flag starts at zero.
        if_(t.lt(4), || sflags.offset(t).store(0u32));
        if_(t.lt(8), || gflags.ptr().offset(b * 8 + t).store(0u32));
        sync_threads();
        // A release and an acquire on shared memory, warp 0 to warp 1.
        if_(warp.eq(0), || {
            data.store(lane, value(0));
            store(&sflag(0), 1u32, st(MemoryOrder::Release, cta));
        });
        if_(warp.eq(1), || {
            spin(&sflag(0), ld(MemoryOrder::Acquire, cta));
            out.store((row, 0), data.load(lane));
        });
        // A release and an acquire on global memory, warp 2 to warp 3.
        if_(warp.eq(2), || {
            data.store(lane + 32, value(1));
            store(&gflag(0), 1u32, st(MemoryOrder::Release, gpu));
        });
        if_(warp.eq(3), || {
            spin(&gflag(0), ld(MemoryOrder::Acquire, gpu));
            out.store((row, 0), data.load(lane + 32));
        });
        // A fence and a relaxed store; a relaxed load and a fence. Warp 1 to 2.
        if_(warp.eq(1), || {
            data.store(lane + 64, value(2));
            fence_acq_rel_cta();
            store(&sflag(1), 1u32, st(MemoryOrder::Relaxed, cta));
        });
        if_(warp.eq(2), || {
            spin(&sflag(1), ld(MemoryOrder::Relaxed, cta));
            fence_acq_rel_cta();
            out.store((row, 0), data.load(lane + 64));
        });
        // seq_cst accesses on global memory, warp 3 to warp 0.
        if_(warp.eq(3), || {
            data.store(lane + 96, value(3));
            store(&gflag(1), 1u32, st(MemoryOrder::Sc, gpu));
        });
        if_(warp.eq(0), || {
            spin(&gflag(1), ld(MemoryOrder::Sc, gpu));
            out.store((row, 0), data.load(lane + 96));
        });
        // Volatile accesses between fences on shared memory, warp 0 to 2.
        if_(warp.eq(0), || {
            data.store(lane + 128, value(4));
            fence_acq_rel_cta();
            store(&sflag(2), 1u32, st(MemoryOrder::Volatile, None));
        });
        if_(warp.eq(2), || {
            spin(&sflag(2), ld(MemoryOrder::Volatile, None));
            fence_acq_rel_cta();
            out.store((row, 1), data.load(lane + 128));
        });
        // A vector release and acquire on global memory, warp 2 to warp 1.
        if_(warp.eq(2), || {
            data.store(lane + 160, value(5));
            let one = Uint32::from(1u32);
            store_vector(&gflag(2), [one, one], st(MemoryOrder::Release, gpu));
        });
        if_(warp.eq(1), || {
            let unset = |()| {
                let [p, q] = load_vector::<u32, 2>(&gflag(2), ld(MemoryOrder::Acquire, gpu));
                p.eq(0u32) | q.eq(0u32)
            };
            while_carry((), unset, |()| ());
            out.store((row, 1), data.load(lane + 160));
        });
        // A release and an acquire with eviction priorities, warp 1 to 3.
        if_(warp.eq(1), || {
            data.store(lane + 192, value(6));
            let hint = Some(CacheEvictionPriority::NoAllocate);
            store(&gflag(4), 1u32, StoreOptions { level1_eviction_priority: hint, ..st(MemoryOrder::Release, gpu) });
        });
        if_(warp.eq(3), || {
            let hint = Some(CacheEvictionPriority::EvictLast);
            spin(&gflag(4), LoadOptions { level1_eviction_priority: hint, ..ld(MemoryOrder::Acquire, gpu) });
            out.store((row, 1), data.load(lane + 192));
        });
        if_(warp.eq(0), || out.store((row, 1), 0u32));
    });
    fn check(args: &[KernelArg]) {
        const N: usize = 128;
        let (xs, gflags, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for b in 0..2 {
            for t in 0..N {
                let row = b * N + t;
                let from = |warp: usize, k: u32| handoff_value(k, xs[b * N + warp * 32 + t % 32]);
                let want = match t / 32 {
                    0 => [from(3, 3), 0],
                    1 => [from(0, 0), from(2, 5)],
                    2 => [from(1, 2), from(0, 4)],
                    _ => [from(2, 1), from(1, 6)],
                };
                assert_eq!(out[row * 2..row * 2 + 2], want, "block {b} thread {t}");
            }
            assert_eq!(gflags[b * 8..b * 8 + 8], [1, 1, 1, 1, 1, 0, 0, 0], "block {b}'s flags");
        }
    }
    Case { name: "ordered_handoffs", kernel, cfg: launch_n(2, N), small_only: false, check }
}

/// The u32 loads and stores of [`cache_hints`], paired: element `j` of a
/// thread is loaded with `.0` and stored, xor `j * 0x01010101`, with `.1`.
fn hinted_pairs() -> Vec<(LoadOptions, StoreOptions)> {
    use CacheEvictionPriority::*;
    use CacheOperator::*;
    use L2PrefetchSize::*;
    let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
    let (l, s) = (LoadOptions::default(), StoreOptions::default());
    let lc = |c| LoadOptions { cop: Some(c), ..l };
    let le = |e| LoadOptions { level1_eviction_priority: Some(e), ..l };
    let lp = |p| LoadOptions { level_prefetch_size: Some(p), ..l };
    let lo = |sem, scope| LoadOptions { sem, scope, ..l };
    let sc = |c| StoreOptions { cop: Some(c), ..s };
    let se = |e| StoreOptions { level1_eviction_priority: Some(e), ..s };
    let so = |sem, scope| StoreOptions { sem, scope, ..s };
    vec![
        (lc(Ca), sc(Wb)),
        (lc(Cg), sc(Cg)),
        (lc(Cs), sc(Cs)),
        (lc(Lu), sc(Wt)),
        (lc(Cv), se(EvictNormal)),
        (le(EvictNormal), se(EvictFirst)),
        (le(EvictFirst), se(EvictLast)),
        (le(EvictLast), se(EvictUnchanged)),
        (le(EvictUnchanged), se(NoAllocate)),
        (le(NoAllocate), so(MemoryOrder::Weak, None)),
        (lp(Size64B), so(MemoryOrder::Release, gpu)),
        (lp(Size128B), StoreOptions { level1_eviction_priority: Some(NoAllocate), ..so(MemoryOrder::Relaxed, sys) }),
        (lp(Size256B), so(MemoryOrder::Mmio, sys)),
        (
            LoadOptions { cop: Some(Cs), ..lp(Size128B) },
            StoreOptions { level1_eviction_priority: Some(EvictFirst), ..so(MemoryOrder::Sc, gpu) },
        ),
        (lo(MemoryOrder::Weak, None), so(MemoryOrder::Relaxed, cta)),
        (LoadOptions { level_prefetch_size: Some(Size64B), ..le(EvictLast) }, so(MemoryOrder::Volatile, None)),
        (
            LoadOptions {
                level1_eviction_priority: Some(EvictFirst),
                level_prefetch_size: Some(Size256B),
                ..lo(MemoryOrder::Relaxed, gpu)
            },
            so(MemoryOrder::Release, sys),
        ),
        (lo(MemoryOrder::Acquire, sys), so(MemoryOrder::Sc, cta)),
        (lo(MemoryOrder::Mmio, sys), se(EvictLast)),
        (LoadOptions { level1_eviction_priority: Some(EvictLast), ..lo(MemoryOrder::Sc, gpu) }, sc(Cg)),
        (lo(MemoryOrder::Volatile, None), so(MemoryOrder::Relaxed, gpu)),
        (lo(MemoryOrder::Relaxed, cta), so(MemoryOrder::Release, cta)),
        (lo(MemoryOrder::Acquire, gpu), sc(Wt)),
        (LoadOptions { level_prefetch_size: Some(Size128B), ..lo(MemoryOrder::Volatile, None) }, sc(Cs)),
    ]
}

/// Which pairs of [`hinted_pairs`] also take the kernel's L2 cache policy:
/// `(load, store)`.
fn hinted_pair_policy(j: usize) -> (bool, bool) {
    (matches!(j, 14 | 15 | 17), matches!(j, 9 | 10))
}

/// Every cache hint of `ld` and `st` on global memory (each cache operator,
/// L1 eviction priority and L2 prefetch size, an L2 cache policy from
/// `createpolicy`, and their combinations with each other and with relaxed,
/// acquire, release, seq_cst, volatile and MMIO accesses), as scalars and as
/// `.v2` and `.v4` vectors, on every element type (8- and 16-bit elements
/// packed into 32-bit words by the inline PTX), scalar constant
/// (non-coherent, `ld.global.nc`) loads, and cache operators and volatile
/// accesses on shared memory. Each thread moves its own data, so
/// the outputs are its inputs, which the GPU must reproduce bit for bit.
pub fn cache_hints() -> Case {
    use CacheEvictionPriority::*;
    use CacheOperator::*;
    const K: usize = 40;
    let kernel = Kernel::builder("cache_hints").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T * K));
        let hs = param_tensor::<F16>("hs", Layout::col_major(T * 16));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 32));
        let ws = param_tensor::<u16>("ws", Layout::col_major(T * 2));
        let ds = param_tensor::<f64>("ds", Layout::col_major(T * 4));
        let ls = param_tensor::<i64>("ls", Layout::col_major(T));
        let fs = param_tensor::<f32>("fs", Layout::col_major(T * 4));
        let qs = param_tensor::<BF16>("qs", Layout::col_major(T * 2));
        let is = param_tensor::<i8>("is", Layout::col_major(T * 4));
        let out = param_tensor::<u32>("out", Layout::col_major(T * K));
        let oh = param_tensor::<F16>("oh", Layout::col_major(T * 16));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 32));
        let ow = param_tensor::<u16>("ow", Layout::col_major(T * 2));
        let od = param_tensor::<f64>("od", Layout::col_major(T * 4));
        let ol = param_tensor::<i64>("ol", Layout::col_major(T));
        let of = param_tensor::<f32>("of", Layout::col_major(T * 4));
        let oq = param_tensor::<BF16>("oq", Layout::col_major(T * 2));
        let oi = param_tensor::<i8>("oi", Layout::col_major(T * 4));
        let t = thread_idx().x;
        let at = |n: usize, j: usize| IndexExpr::from(t) * n + j;
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        let (l, s) = (LoadOptions::default(), StoreOptions::default());
        let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
        for (j, (lo, so)) in hinted_pairs().into_iter().enumerate() {
            let (lp, sp) = hinted_pair_policy(j);
            let lo = LoadOptions { cache_policy: lp.then_some(policy), ..lo };
            let so = StoreOptions { cache_policy: sp.then_some(policy), ..so };
            let v = load(&xs.ptr().offset(at(K, j)), lo);
            store(&out.ptr().offset(at(K, j)), v ^ (j as u32 * 0x0101_0101), so);
        }
        // Vectors of u32.
        let volatile = MemoryOrder::Volatile;
        let v = load_vector::<u32, 4>(
            &xs.ptr().offset(at(K, 24)),
            LoadOptions { sem: volatile, level_prefetch_size: Some(L2PrefetchSize::Size64B), ..l },
        );
        store_vector(&out.ptr().offset(at(K, 24)), v.map(|x| x ^ 0x5a5a_5a5a), StoreOptions { sem: volatile, ..s });
        let v = load_vector::<u32, 2>(
            &xs.ptr().offset(at(K, 28)),
            LoadOptions { sem: MemoryOrder::Relaxed, scope: cta, level1_eviction_priority: Some(NoAllocate), ..l },
        );
        store_vector(&out.ptr().offset(at(K, 28)), v, StoreOptions { cop: Some(Cs), ..s });
        let v = load_vector::<u32, 2>(
            &xs.ptr().offset(at(K, 30)),
            LoadOptions { sem: MemoryOrder::Sc, scope: gpu, cache_policy: Some(policy), ..l },
        );
        store_vector(
            &out.ptr().offset(at(K, 30)),
            v,
            StoreOptions {
                sem: MemoryOrder::Release,
                scope: gpu,
                level1_eviction_priority: Some(EvictLast),
                cache_policy: Some(policy),
                ..s
            },
        );
        let v = load_vector::<u32, 4>(&xs.ptr().offset(at(K, 32)), LoadOptions { cop: Some(Cg), ..l });
        store_vector(&out.ptr().offset(at(K, 32)), v, StoreOptions { sem: MemoryOrder::Sc, scope: sys, ..s });
        // Cache operators and volatile accesses on shared memory.
        let smem = make_smem_tensor::<u32>(Layout::col_major(3 * T));
        store(&smem.ptr().offset(t), load(&xs.ptr().offset(at(K, 36)), l), StoreOptions { cop: Some(Wt), ..s });
        let pair = load_vector::<u32, 2>(&xs.ptr().offset(at(K, 38)), l);
        store_vector(&smem.ptr().offset(IndexExpr::from(t) * 2 + T), pair, StoreOptions { sem: volatile, ..s });
        sync_threads();
        let next = (t + 1) % T as i32;
        let a = load(&smem.ptr().offset(next), LoadOptions { cop: Some(Cs), ..l });
        out.store(at(K, 36), a);
        out.store(at(K, 37), 0u32);
        let p = load_vector::<u32, 2>(
            &smem.ptr().offset(IndexExpr::from(next) * 2 + T),
            LoadOptions { sem: volatile, ..l },
        );
        store_vector(&out.ptr().offset(at(K, 38)), p, s);
        // Other element types, packed into words where they are narrower.
        let h = load_vector::<F16, 8>(
            &hs.ptr().offset(at(16, 0)),
            LoadOptions { level1_eviction_priority: Some(EvictLast), ..l },
        );
        store_vector(&oh.ptr().offset(at(16, 0)), h, StoreOptions { cop: Some(Cs), ..s });
        let h = load(&hs.ptr().offset(at(16, 8)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oh.ptr().offset(at(16, 8)), h, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let h = load(&hs.ptr().offset(at(16, 9)), LoadOptions { sem: MemoryOrder::Sc, scope: cta, ..l });
        store(&oh.ptr().offset(at(16, 9)), h, StoreOptions { sem: volatile, ..s });
        // A scalar constant (non-coherent) load: the kernel never writes hs.
        let h = load(&hs.ptr().offset(at(16, 10)), LoadOptions { sem: MemoryOrder::Constant, ..l });
        oh.store(at(16, 10), h);
        for k in 11..16 {
            oh.store(at(16, k), F16::from_f32(0.0));
        }
        let b = load_vector::<u8, 16>(&bs.ptr().offset(at(32, 0)), LoadOptions { sem: volatile, ..l });
        store_vector(&ob.ptr().offset(at(32, 0)), b, StoreOptions { cop: Some(Wt), ..s });
        let b = load_vector::<u8, 2>(&bs.ptr().offset(at(32, 16)), LoadOptions { cop: Some(Cg), ..l });
        store_vector(&ob.ptr().offset(at(32, 16)), b, StoreOptions { level1_eviction_priority: Some(EvictFirst), ..s });
        let b = load(&bs.ptr().offset(at(32, 18)), LoadOptions { level1_eviction_priority: Some(EvictLast), ..l });
        store(&ob.ptr().offset(at(32, 18)), b, StoreOptions { cop: Some(Cs), ..s });
        let b = load_vector::<u8, 4>(
            &bs.ptr().offset(at(32, 20)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l },
        );
        store_vector(&ob.ptr().offset(at(32, 20)), b, StoreOptions { sem: MemoryOrder::Relaxed, scope: cta, ..s });
        let b = load_vector::<u8, 8>(
            &bs.ptr().offset(at(32, 24)),
            LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size128B), ..l },
        );
        store_vector(&ob.ptr().offset(at(32, 24)), b, StoreOptions { level1_eviction_priority: Some(NoAllocate), ..s });
        let nc = LoadOptions {
            sem: MemoryOrder::Constant,
            cop: Some(Cs),
            level_prefetch_size: Some(L2PrefetchSize::Size64B),
            ..l
        };
        ob.store(at(32, 19), load(&bs.ptr().offset(at(32, 19)), nc));
        let w = load_vector::<u16, 2>(
            &ws.ptr().offset(at(2, 0)),
            LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, level1_eviction_priority: Some(EvictLast), ..l },
        );
        store_vector(&ow.ptr().offset(at(2, 0)), w, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let d = load_vector::<f64, 2>(
            &ds.ptr().offset(at(4, 0)),
            LoadOptions { level1_eviction_priority: Some(NoAllocate), ..l },
        );
        store_vector(&od.ptr().offset(at(4, 0)), d, StoreOptions { cop: Some(Wt), ..s });
        let d = load(&ds.ptr().offset(at(4, 2)), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store(&od.ptr().offset(at(4, 2)), d, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let d = load(&ds.ptr().offset(at(4, 3)), LoadOptions { sem: volatile, ..l });
        store(&od.ptr().offset(at(4, 3)), d, StoreOptions { sem: MemoryOrder::Relaxed, scope: sys, ..s });
        let q = load(&ls.ptr().offset(t), LoadOptions { cache_policy: Some(policy), ..l });
        store(&ol.ptr().offset(t), q, StoreOptions { cop: Some(Cg), ..s });
        let f = load_vector::<f32, 4>(
            &fs.ptr().offset(at(4, 0)),
            LoadOptions { level_prefetch_size: Some(L2PrefetchSize::Size256B), ..l },
        );
        store_vector(
            &of.ptr().offset(at(4, 0)),
            f,
            StoreOptions { level1_eviction_priority: Some(EvictUnchanged), ..s },
        );
        let e = load(
            &qs.ptr().offset(at(2, 0)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, level1_eviction_priority: Some(EvictFirst), ..l },
        );
        store(
            &oq.ptr().offset(at(2, 0)),
            e,
            StoreOptions { sem: MemoryOrder::Relaxed, scope: gpu, level1_eviction_priority: Some(EvictLast), ..s },
        );
        let e = load(&qs.ptr().offset(at(2, 1)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oq.ptr().offset(at(2, 1)), e, StoreOptions { sem: MemoryOrder::Release, scope: cta, ..s });
        let i = load(&is.ptr().offset(at(4, 0)), LoadOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..l });
        store(&oi.ptr().offset(at(4, 0)), i, StoreOptions { sem: MemoryOrder::Relaxed, scope: cta, ..s });
        let i = load(
            &is.ptr().offset(at(4, 1)),
            LoadOptions { sem: MemoryOrder::Acquire, scope: sys, level1_eviction_priority: Some(EvictFirst), ..l },
        );
        store(
            &oi.ptr().offset(at(4, 1)),
            i,
            StoreOptions { sem: MemoryOrder::Release, scope: gpu, level1_eviction_priority: Some(NoAllocate), ..s },
        );
        let i = load_vector::<i8, 2>(&is.ptr().offset(at(4, 2)), LoadOptions { sem: volatile, ..l });
        store_vector(&oi.ptr().offset(at(4, 2)), i, StoreOptions { sem: MemoryOrder::Sc, scope: gpu, ..s });
    });
    fn check(args: &[KernelArg]) {
        const K: usize = 40;
        let xs = args[0].to_vec::<u32>();
        let out = args[9].to_vec::<u32>();
        for t in 0..T {
            let x = &xs[t * K..t * K + K];
            let o = &out[t * K..t * K + K];
            for j in 0..24 {
                assert_eq!(o[j], x[j] ^ (j as u32 * 0x0101_0101), "thread {t} pair {j}");
            }
            let flip: Vec<u32> = x[24..28].iter().map(|v| v ^ 0x5a5a_5a5a).collect();
            assert_eq!(o[24..28], flip[..], "thread {t}");
            assert_eq!(o[28..36], x[28..36], "thread {t}");
            let n = (t + 1) % T;
            assert_eq!([o[36], o[37]], [xs[n * K + 36], 0], "thread {t}");
            assert_eq!(o[38..40], xs[n * K + 38..n * K + 40], "thread {t}");
        }
        // F16, u8, u16, f64, i64, f32, bf16, i8: each output holds its input
        // where the kernel moved data, and zeros where it filled.
        let h_in = args[1].to_vec::<F16>();
        let h_out = args[10].to_vec::<F16>();
        for t in 0..T {
            assert_eq!(h_out[t * 16..t * 16 + 11], h_in[t * 16..t * 16 + 11], "halves of thread {t}");
            assert!(h_out[t * 16 + 11..t * 16 + 16].iter().all(|h| h.to_f32() == 0.0));
        }
        let (b_in, b_out) = (args[2].to_vec::<u8>(), args[11].to_vec::<u8>());
        for t in 0..T {
            assert_eq!(b_out[t * 32..t * 32 + 32], b_in[t * 32..t * 32 + 32], "bytes of thread {t}");
        }
        assert_eq!(args[12].to_vec::<u16>(), args[3].to_vec::<u16>());
        let (d_in, d_out) = (args[4].to_vec::<f64>(), args[13].to_vec::<f64>());
        assert!(d_in.iter().zip(&d_out).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(args[14].to_vec::<i64>(), args[5].to_vec::<i64>());
        let (f_in, f_out) = (args[6].to_vec::<f32>(), args[15].to_vec::<f32>());
        assert!(f_in.iter().zip(&f_out).all(|(a, b)| a.to_bits() == b.to_bits()));
        let (q_in, q_out) = (args[7].to_vec::<BF16>(), args[16].to_vec::<BF16>());
        assert!(q_in.iter().zip(&q_out).all(|(a, b)| a.0 == b.0));
        assert_eq!(args[17].to_vec::<i8>(), args[8].to_vec::<i8>());
    }
    Case { name: "cache_hints", kernel, cfg: launch(1), small_only: false, check }
}

/// `cute.arch.prefetch` in each form: into L1 and L2, L2 with each eviction
/// priority, predicated, uniform (`prefetchu.L1`), through a pointer of
/// another element type, past the end of the buffer (harmless: nothing is
/// read), and on per-thread memory (generic addressing). The prefetches
/// change nothing, so the result is the plain copy.
pub fn prefetches() -> Case {
    use PrefetchCacheLevel::L2;
    let kernel = Kernel::builder("prefetches").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let out = param_tensor::<f32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let p = xs.ptr().offset(t);
        let l2 = PrefetchOptions { cache_level: Some(L2), ..Default::default() };
        prefetch(&p, PrefetchOptions::default());
        prefetch(&p, l2);
        prefetch(&p, PrefetchOptions { evict_priority: Some(CacheEvictionPriority::EvictLast), ..l2 });
        prefetch(&p, PrefetchOptions { evict_priority: Some(CacheEvictionPriority::EvictNormal), ..l2 });
        prefetch(&p, PrefetchOptions { uniform: true, ..Default::default() });
        prefetch(&xs.ptr().offset(IndexExpr::from(t) + T), PrefetchOptions { predicate: Some(t.lt(32)), ..l2 });
        prefetch(&p.cast::<u8>().offset(2usize), PrefetchOptions::default());
        let stack = make_rmem_tensor::<f32>(Layout::col_major(4));
        stack.store(0, xs.load(t));
        prefetch(&stack.ptr(), PrefetchOptions::default());
        prefetch(&stack.ptr(), PrefetchOptions { predicate: Some(t.ge(16)), ..l2 });
        out.store(t, stack.load(0) * 2.0);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..T {
            assert_eq!(out[t].to_bits(), (xs[t] * 2.0).to_bits(), "thread {t}");
        }
    }
    Case { name: "prefetches", kernel, cfg: launch(1), small_only: false, check }
}

/// Hand-offs at cluster scope (SM90): a release and an acquire on a shared
/// flag, and a vector release and acquire with eviction priorities on a
/// global flag, in each CTA of a cluster of two, which have flags of their
/// own.
pub fn cluster_scope_handoff() -> Case {
    const N: usize = 64;
    let kernel = Kernel::builder("cluster_scope_handoff").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(2 * N));
        let gflags = param_tensor::<u32>("gflags", Layout::col_major(2 * 4));
        let out = param_tensor::<u32>("out", Layout::row_major((2 * N, 2)));
        let t = thread_idx().x;
        let b = block_idx().x;
        let row = b * N as i32 + t;
        let x = xs.load(row);
        let lane = t % 32;
        let data = make_smem_tensor::<u32>(Layout::col_major(2 * 32));
        let sflag = alloc_smem::<u32>(1, 4);
        let gflag = gflags.ptr().offset(IndexExpr::from(b) * 4);
        let cluster = Some(MemoryScope::Cluster);
        if_(t.eq(0), || sflag.store(0u32));
        if_(t.lt(4), || gflags.ptr().offset(IndexExpr::from(b) * 4 + t).store(0u32));
        sync_threads();
        if_(warp_idx().eq(0), || {
            data.store(lane, x + 7u32);
            store(&sflag, 1u32, StoreOptions { sem: MemoryOrder::Release, scope: cluster, ..Default::default() });
        });
        if_(warp_idx().eq(1), || {
            let acquire = LoadOptions { sem: MemoryOrder::Acquire, scope: cluster, ..Default::default() };
            while_carry((), |()| load(&sflag, acquire).eq(0u32), |()| ());
            out.store((row, 0), data.load(lane));
            data.store(lane + 32, x * 5u32);
            let one = Uint32::from(1u32);
            let hint = Some(CacheEvictionPriority::NoAllocate);
            let release = StoreOptions {
                sem: MemoryOrder::Release,
                scope: cluster,
                level1_eviction_priority: hint,
                ..Default::default()
            };
            store_vector(&gflag, [one, one], release);
        });
        if_(warp_idx().eq(0), || {
            let hint = Some(CacheEvictionPriority::EvictLast);
            let acquire = LoadOptions {
                sem: MemoryOrder::Acquire,
                scope: cluster,
                level1_eviction_priority: hint,
                ..Default::default()
            };
            let unset = |()| {
                let [p, q] = load_vector::<u32, 2>(&gflag, acquire);
                p.eq(0u32) | q.eq(0u32)
            };
            while_carry((), unset, |()| ());
            out.store((row, 0), data.load(lane + 32));
        });
        out.store(
            (row, 1),
            load(&sflag, LoadOptions { sem: MemoryOrder::Relaxed, scope: cluster, ..Default::default() }),
        );
    });
    fn check(args: &[KernelArg]) {
        const N: usize = 64;
        let (xs, gflags, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u32>());
        for b in 0..2 {
            for t in 0..N {
                let row = b * N + t;
                let from = |warp: usize| xs[b * N + warp * 32 + t % 32];
                let want = if t < 32 { from(1).wrapping_mul(5) } else { from(0).wrapping_add(7) };
                assert_eq!(out[row * 2..row * 2 + 2], [want, 1], "block {b} thread {t}");
            }
            assert_eq!(gflags[b * 4..b * 4 + 4], [1, 1, 0, 0], "block {b}'s flags");
        }
    }
    let cfg = LaunchConfig::new([2, 1, 1], [N as u32, 1, 1]).with_cluster([2, 1, 1]);
    Case { name: "cluster_scope_handoff", kernel, cfg, small_only: false, check }
}

/// Tiled copies of a 32 x 16 tile from global memory to registers and back
/// with the G2R and R2G atoms (CuTe DSL's `CopyG2ROp` and `CopyR2GOp`) and
/// their memory attributes: volatile, acquire, seq_cst and MMIO loads,
/// release, relaxed, seq_cst and MMIO stores, invariant (non-coherent) and
/// constant loads of a buffer the kernel never writes, an L2 prefetch size,
/// eviction priorities, the cache modes, a cache policy given at copy time,
/// `num_bits_per_copy = 0` (vectorised by the copy, or element by element
/// under a predicate and for MMIO), predicated copies, 16-bit elements, and
/// a `make_rmem_tensor` on the register side.
pub fn global_copy_atoms() -> Case {
    use CacheEvictionPriority::*;
    const M: usize = 32;
    const N: usize = 16;
    let kernel = Kernel::builder("global_copy_atoms").block_dim([T as u32, 1, 1]).build(|| {
        let tile = Layout::col_major((M, N));
        let a = param_tensor::<f32>("a", tile.clone());
        let b = param_tensor::<f32>("b", tile.clone());
        let h = param_tensor::<F16>("h", tile.clone());
        let o: Vec<Tensor<f32>> = (0..8).map(|k| param_tensor::<f32>(&format!("o{k}"), tile.clone())).collect();
        let oh = param_tensor::<F16>("oh", tile.clone());
        let t = thread_idx().x;
        let (cta, gpu, sys) = (MemoryScope::Cta, MemoryScope::Gpu, MemoryScope::Sys);
        let g2r = |bits, attrs| make_copy_atom::<f32>(CopyOp::G2R { bits, attrs });
        let r2g = |bits, attrs| make_copy_atom::<f32>(CopyOp::R2G { bits, attrs });
        let load = |memory_order, memory_scope| G2RAttrs { memory_order, memory_scope, ..G2RAttrs::default() };
        let store = |memory_order, memory_scope| R2GAttrs { memory_order, memory_scope, ..R2GAttrs::default() };
        let tiled =
            |atom: &CopyAtom<f32>| make_tiled_copy_tv(atom, &Layout::col_major((8, 8)), &Layout::col_major((4, 2)));
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_first.b64 $0, 0.5;", &[], |_| vec![0]);
        let move_tile =
            |src: &Tensor<f32>, g: CopyAtom<f32>, dst: &Tensor<f32>, r: CopyAtom<f32>, policy: Option<Int64>| {
                let (tg, tr) = (tiled(&g), tiled(&r));
                let s = tg.get_slice(t).partition_s(src);
                let frag = make_fragment_like::<f32, f32>(&s);
                copy_with(&tg, &s, &frag, CopyOptions { cache_policy: policy, ..CopyOptions::default() });
                copy_with(
                    &tr,
                    &frag,
                    &tr.get_slice(t).partition_d(dst),
                    CopyOptions { cache_policy: policy, ..CopyOptions::default() },
                );
            };
        let evict_last = R2GAttrs { l1c_evict_priority: EvictLast, ..store(MemoryOrder::Release, gpu) };
        move_tile(&a, g2r(128, load(MemoryOrder::Volatile, sys)), &o[0], r2g(128, evict_last), None);
        let no_allocate = R2GAttrs { l1c_evict_priority: NoAllocate, ..store(MemoryOrder::Relaxed, cta) };
        move_tile(&a, g2r(128, load(MemoryOrder::Acquire, gpu)), &o[1], r2g(64, no_allocate), None);
        let invariant = G2RAttrs {
            invariant: true,
            l2_prefetch_size: Some(L2PrefetchSize::Size256B),
            load_cache_mode: LoadCacheMode::Streaming,
            ..G2RAttrs::default()
        };
        let streaming = R2GAttrs { store_cache_mode: StoreCacheMode::Streaming, ..R2GAttrs::default() };
        move_tile(&b, g2r(128, invariant), &o[2], r2g(128, streaming), None);
        let through = R2GAttrs { store_cache_mode: StoreCacheMode::WriteThrough, ..R2GAttrs::default() };
        move_tile(&b, g2r(0, load(MemoryOrder::Constant, cta)), &o[3], r2g(0, through), None);
        let last_use = G2RAttrs { load_cache_mode: LoadCacheMode::LastUse, ..G2RAttrs::default() };
        let global = R2GAttrs { store_cache_mode: StoreCacheMode::Global, ..R2GAttrs::default() };
        move_tile(&a, g2r(64, last_use), &o[4], r2g(64, global), Some(policy));
        move_tile(&a, g2r(128, load(MemoryOrder::Sc, gpu)), &o[5], r2g(32, store(MemoryOrder::Sc, cta)), None);
        move_tile(&a, g2r(0, load(MemoryOrder::Mmio, sys)), &o[6], r2g(32, store(MemoryOrder::Mmio, sys)), None);
        // Rows 0 to 19 only, element by element.
        let (tg, tr) =
            (tiled(&g2r(0, load(MemoryOrder::Relaxed, gpu))), tiled(&r2g(0, store(MemoryOrder::Release, gpu))));
        let thr = tg.get_slice(t);
        let s = thr.partition_s(&a);
        let rows = IntTuple::from((20, N));
        let pred = make_predicate(&thr.partition_s(&make_identity_tensor((M, N))), |c| elem_less(c, &rows));
        let frag = make_fragment_like::<f32, f32>(&s);
        frag.fill(0.0);
        copy_with(&tg, &s, &frag, CopyOptions { pred: Some(&pred), ..CopyOptions::default() });
        copy_with(
            &tr,
            &frag,
            &tr.get_slice(t).partition_d(&o[7]),
            CopyOptions { pred: Some(&pred), ..CopyOptions::default() },
        );
        // Eight halves per thread, through the non-coherent cache.
        let hg = make_copy_atom::<F16>(CopyOp::G2R {
            bits: 128,
            attrs: G2RAttrs { invariant: true, l1c_evict_priority: EvictFirst, ..G2RAttrs::default() },
        });
        let hr = make_copy_atom::<F16>(CopyOp::R2G { bits: 128, attrs: store(MemoryOrder::Volatile, sys) });
        let (thr_h, val_h) = (Layout::col_major((4, 16)), Layout::col_major((8, 1)));
        let (tg, tr) = (make_tiled_copy_tv(&hg, &thr_h, &val_h), make_tiled_copy_tv(&hr, &thr_h, &val_h));
        let s = tg.get_slice(t).partition_s(&h);
        let frag = make_fragment_like::<F16, F16>(&s);
        copy(&tg, &s, &frag);
        copy(&tr, &frag, &tr.get_slice(t).partition_d(&oh));
        // The register side as a make_rmem_tensor (CuTe DSL's register
        // tensor, per-thread memory here) rather than a fragment: an acquire
        // load into it and a release store from it (vectorised by the copy).
        let orm = param_tensor::<f32>("orm", tile.clone());
        let (tg, tr) =
            (tiled(&g2r(128, load(MemoryOrder::Acquire, gpu))), tiled(&r2g(0, store(MemoryOrder::Release, cta))));
        let s = tg.get_slice(t).partition_s(&b);
        let stack = make_rmem_tensor::<f32>(Layout::col_major(s.shape().clone()));
        copy(&tg, &s, &stack);
        copy(&tr, &stack, &tr.get_slice(t).partition_d(&orm));
    });
    fn check(args: &[KernelArg]) {
        let (a, b) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (k, want) in [(0, &a), (1, &a), (2, &b), (3, &b), (4, &a), (5, &a), (6, &a)] {
            assert_eq!(bits(&args[3 + k].to_vec::<f32>()), bits(want), "o{k}");
        }
        let o7 = args[10].to_vec::<f32>();
        for n in 0..16 {
            for m in 0..20 {
                assert_eq!(o7[m + 32 * n].to_bits(), a[m + 32 * n].to_bits(), "o7[{m}, {n}]");
            }
        }
        let (h, oh) = (args[2].to_vec::<F16>(), args[11].to_vec::<F16>());
        assert!(h.iter().zip(&oh).all(|(x, y)| x.0 == y.0), "oh");
        assert_eq!(bits(&args[12].to_vec::<f32>()), bits(&b), "orm");
    }
    Case { name: "global_copy_atoms", kernel, cfg: launch(1), small_only: false, check }
}

/// The owner `make_cotiled_copy` gives the element at address `a` of the
/// tiles of [`cotiled_copies`]: thread `(t0, t1)` (index `t0 + 4 t1`) copies
/// addresses `32 t0 + 4 t1 .. + 4`.
pub fn cotiled_owner(a: usize) -> usize {
    a / 32 + 4 * ((a % 32) / 4)
}

/// `make_cotiled_copy` in copies: 32 threads with four values each, spread
/// over a 16 x 8 tile by the atom layout `((4, 8), 4):((32, 4), 1)` (thread
/// `(t0, t1)` takes addresses `32 t0 + 4 t1 .. + 4`), on a row-major tile
/// (whose tiler is the whole tile) and on a column-major one (whose tiler
/// selects rows out of order). Each tile goes from global memory through
/// registers to another tensor in 128-bit G2R (invariant) and R2G copies,
/// and each thread writes its index where it copied, so the outputs show
/// the copy and the thread that `make_cotiled_copy` assigned each element.
pub fn cotiled_copies() -> Case {
    const M: usize = 16;
    const N: usize = 8;
    let kernel = Kernel::builder("cotiled_copies").block_dim([32, 1, 1]).build(|| {
        let t = thread_idx().x;
        let atom_tv = Layout::new(((4, 8), 4), ((32, 4), 1));
        let invariant = G2RAttrs { invariant: true, ..G2RAttrs::default() };
        let g2r = make_copy_atom::<f32>(CopyOp::G2R { bits: 128, attrs: invariant });
        let r2g = make_copy_atom::<f32>(CopyOp::R2G { bits: 128, attrs: R2GAttrs::default() });
        for (k, data) in [Layout::row_major((M, N)), Layout::col_major((M, N))].into_iter().enumerate() {
            let a = param_tensor::<f32>(&format!("a{k}"), data.clone());
            let b = param_tensor::<f32>(&format!("b{k}"), data.clone());
            let owner = param_tensor::<i32>(&format!("owner{k}"), data.clone());
            let (load, store) = (make_cotiled_copy(&g2r, &atom_tv, &data), make_cotiled_copy(&r2g, &atom_tv, &data));
            let src = load.get_slice(t).partition_s(&a);
            let frag = make_fragment_like::<f32, f32>(&src);
            copy(&load, &src, &frag);
            copy(&store, &frag, &store.get_slice(t).partition_d(&b));
            let id = make_copy_atom::<i32>(CopyOp::R2G { bits: 128, attrs: R2GAttrs::default() });
            let mark = make_cotiled_copy(&id, &atom_tv, &data);
            let dst = mark.get_slice(t).partition_d(&owner);
            let mine = make_fragment_like::<i32, i32>(&dst);
            mine.fill(t);
            copy(&mark, &mine, &dst);
        }
    });
    fn check(args: &[KernelArg]) {
        for k in 0..2 {
            let (a, b, owner) =
                (args[3 * k].to_vec::<f32>(), args[3 * k + 1].to_vec::<f32>(), args[3 * k + 2].to_vec::<i32>());
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "tile {k}");
            for (addr, &o) in owner.iter().enumerate() {
                assert_eq!(o as usize, cotiled_owner(addr), "tile {k}, address {addr}");
            }
        }
    }
    Case { name: "cotiled_copies", kernel, cfg: launch_n(1, 32), small_only: false, check }
}

/// Strong (ordered and volatile) vectors of 8- and 16-bit elements, which
/// PTX moves with their element type (`.v4.b8`, `.v2.b16`), since its
/// memory model treats each element of a vector access as one access, and
/// one instruction per four elements beyond four: relaxed, acquire,
/// release, seq_cst and volatile vectors of 2, 4, 8 and 16 elements on
/// global and shared memory, with an eviction priority and a cache policy,
/// and a flag hand-off between warps in which the consumer acquires one byte
/// of the producer's vector release (the same bytes as one of its elements,
/// so the two synchronise). Each thread's outputs are its inputs, or its
/// neighbour's through shared memory, which the GPU must reproduce bit for
/// bit.
pub fn strong_vectors() -> Case {
    use CacheEvictionPriority::*;
    let kernel = Kernel::builder("strong_vectors").block_dim([T as u32, 1, 1]).build(|| {
        let hs = param_tensor::<F16>("hs", Layout::col_major(T * 8));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 16));
        let is = param_tensor::<i8>("is", Layout::col_major(T * 8));
        let qs = param_tensor::<BF16>("qs", Layout::col_major(T * 4));
        let ws = param_tensor::<u16>("ws", Layout::col_major(T * 2));
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let oh = param_tensor::<F16>("oh", Layout::col_major(T * 8));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 16));
        let oi = param_tensor::<i8>("oi", Layout::col_major(T * 8));
        let oq = param_tensor::<BF16>("oq", Layout::col_major(T * 4));
        let ow = param_tensor::<u16>("ow", Layout::col_major(T * 2));
        let os = param_tensor::<u8>("os", Layout::col_major(T * 8));
        let out = param_tensor::<u32>("out", Layout::col_major(T));
        let t = thread_idx().x;
        let at = |n: usize| IndexExpr::from(t) * n;
        let (cta, gpu, sys) = (Some(MemoryScope::Cta), Some(MemoryScope::Gpu), Some(MemoryScope::Sys));
        let lo = |sem, scope| LoadOptions { sem, scope, ..Default::default() };
        let so = |sem, scope| StoreOptions { sem, scope, ..Default::default() };
        // Eight halves: two instructions each way.
        let h = load_vector::<F16, 8>(&hs.ptr().offset(at(8)), lo(MemoryOrder::Relaxed, gpu));
        store_vector(&oh.ptr().offset(at(8)), h, so(MemoryOrder::Release, gpu));
        // Sixteen bytes: four instructions each way, after one fence for the
        // seq_cst store.
        let b = load_vector::<u8, 16>(&bs.ptr().offset(at(16)), lo(MemoryOrder::Acquire, cta));
        store_vector(&ob.ptr().offset(at(16)), b, so(MemoryOrder::Sc, gpu));
        let i = load_vector::<i8, 8>(&is.ptr().offset(at(8)), lo(MemoryOrder::Sc, sys));
        store_vector(&oi.ptr().offset(at(8)), i, so(MemoryOrder::Volatile, None));
        let q = load_vector::<BF16, 4>(&qs.ptr().offset(at(4)), lo(MemoryOrder::Volatile, None));
        store_vector(&oq.ptr().offset(at(4)), q, so(MemoryOrder::Relaxed, cta));
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        let w = load_vector::<u16, 2>(
            &ws.ptr().offset(at(2)),
            LoadOptions {
                level1_eviction_priority: Some(EvictFirst),
                cache_policy: Some(policy),
                ..lo(MemoryOrder::Acquire, gpu)
            },
        );
        store_vector(
            &ow.ptr().offset(at(2)),
            w,
            StoreOptions {
                level1_eviction_priority: Some(NoAllocate),
                cache_policy: Some(policy),
                ..so(MemoryOrder::Relaxed, sys)
            },
        );
        // Shared memory: eight bytes per thread with a release, read by the
        // next thread with an acquire after a barrier.
        let smem = make_smem_tensor::<u8>(Layout::col_major(T * 8));
        let mine = load_vector::<u8, 8>(&bs.ptr().offset(at(16)), LoadOptions::default());
        store_vector(&smem.ptr().offset(at(8)), mine, so(MemoryOrder::Release, cta));
        sync_threads();
        let next = (t + 1) % T as i32;
        let theirs = load_vector::<u8, 8>(&smem.ptr().offset(IndexExpr::from(next) * 8), lo(MemoryOrder::Acquire, cta));
        store_vector(&os.ptr().offset(at(8)), theirs, StoreOptions::default());
        // Warp 0 writes data and then four flag bytes with one release; warp
        // 1 spins on byte 2 with an acquire and reads the data.
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        let flags = alloc_smem::<u8>(4, 4);
        if_(t.lt(4), || flags.offset(t).store(0u8));
        sync_threads();
        let lane = t % 32;
        if_(warp_idx().eq(0), || {
            data.store(lane, xs.load(lane) * 3u32 + 1u32);
            let one = Uint8::from(1u8);
            store_vector(&flags, [one, one, one, one], so(MemoryOrder::Release, cta));
            out.store(t, 0u32);
        });
        if_(warp_idx().eq(1), || {
            while_carry((), |()| load(&flags.offset(2usize), lo(MemoryOrder::Acquire, cta)).eq(0u8), |()| ());
            out.store(t, data.load(lane));
        });
    });
    fn check(args: &[KernelArg]) {
        let half_bits = |a: &KernelArg| a.to_vec::<F16>().iter().map(|h| h.0).collect::<Vec<_>>();
        let bf16_bits = |a: &KernelArg| a.to_vec::<BF16>().iter().map(|h| h.0).collect::<Vec<_>>();
        assert_eq!(half_bits(&args[6]), half_bits(&args[0]), "oh");
        let (bs, ob) = (args[1].to_vec::<u8>(), args[7].to_vec::<u8>());
        assert_eq!(ob, bs, "ob");
        assert_eq!(args[8].to_vec::<i8>(), args[2].to_vec::<i8>(), "oi");
        assert_eq!(bf16_bits(&args[9]), bf16_bits(&args[3]), "oq");
        assert_eq!(args[10].to_vec::<u16>(), args[4].to_vec::<u16>(), "ow");
        let os = args[11].to_vec::<u8>();
        for t in 0..T {
            let n = (t + 1) % T;
            assert_eq!(os[t * 8..t * 8 + 8], bs[n * 16..n * 16 + 8], "os of thread {t}");
        }
        let (xs, out) = (args[5].to_vec::<u32>(), args[12].to_vec::<u32>());
        for t in 0..T {
            let want = if t < 32 { 0 } else { xs[t - 32].wrapping_mul(3).wrapping_add(1) };
            assert_eq!(out[t], want, "out[{t}]");
        }
    }
    Case { name: "strong_vectors", kernel, cfg: launch(1), small_only: false, check }
}

// ---------------------------------------------------------------------------
// Carry chains through loads and stores with memory attributes, and prefetches.
// ---------------------------------------------------------------------------

/// Words per thread that [`carry_around_memory_accesses`] loads (`ys`) and stores (`out32`).
const MEM_CARRY_WORDS: usize = 8;

/// Carry chains through loads and stores with memory attributes and through prefetches, which
/// keep the carry flag like the warp collectives (see `cute::ir::carry_flag_breaker`), so the
/// validator accepts a chain around them and the simulator keeps the flag. Between a carry-out
/// and its carry-in sit, in the 128-bit sum of numbers held as two u64 words: an acquire load
/// and a release store of global memory and a relaxed load and a release store of the thread's
/// own word of shared memory (LLVM atomics with a sync scope), a seq_cst load (`fence.sc` and an
/// acquire load), an L2 prefetch with an eviction priority, a load with an L1 eviction priority
/// and an L2 cache policy (inline PTX), a volatile load (an LLVM volatile load), a uniform
/// prefetch and a plain load and store; and in their difference: an acquire vector of eight
/// bytes and a relaxed one (each two instructions of four), a non-coherent load, a predicated
/// prefetch, a streaming store and a volatile one. The odd threads carry and borrow out of the
/// low word.
pub fn carry_around_memory_accesses() -> Case {
    const W: usize = MEM_CARRY_WORDS;
    let kernel = Kernel::builder("carry_around_memory_accesses").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u64>("xs", Layout::row_major((T, 4)));
        let ys = param_tensor::<u32>("ys", Layout::row_major((T, W)));
        let bs = param_tensor::<u8>("bs", Layout::col_major(T * 8));
        let out64 = param_tensor::<u64>("out64", Layout::row_major((T, 5)));
        let out32 = param_tensor::<u32>("out32", Layout::row_major((T, W)));
        let ob = param_tensor::<u8>("ob", Layout::col_major(T * 8));
        let t = thread_idx().x;
        let own = make_smem_tensor::<u32>(Layout::col_major(T)).ptr().offset(t);
        let w: [Uint64; 4] = std::array::from_fn(|i| xs.load((t, i)) * SPREAD);
        let odd = (t & 1).eq(1);
        let (a0, a1) = (select(odd, Value::constant(u64::MAX), w[0]), w[1]);
        let (b0, b1) = (select(odd, w[2] | 1u64, w[2]), w[3]);
        let policy: Int64 =
            inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| vec![0]);
        own.store(ys.load((t, 6)));
        let (l, s, p) = (LoadOptions::default(), StoreOptions::default(), PrefetchOptions::default());
        let (gpu, cta) = (Some(MemoryScope::Gpu), Some(MemoryScope::Cta));
        let y = |j: usize| ys.ptr().offset(IndexExpr::from(t) * W + j);
        let o = |j: usize| out32.ptr().offset(IndexExpr::from(t) * W + j);
        let byte_row = |b: &Tensor<u8>| b.ptr().offset(IndexExpr::from(t) * 8);
        // The sum.
        let low = add_cc(a0, b0);
        let acquired = load(&y(0), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store(&o(0), acquired ^ 0x0101_0101u32, StoreOptions { sem: MemoryOrder::Release, scope: gpu, ..s });
        let relaxed = load(&own, LoadOptions { sem: MemoryOrder::Relaxed, scope: cta, ..l });
        store(&own, relaxed + 1u32, StoreOptions { sem: MemoryOrder::Release, scope: cta, ..s });
        let sequential = load(&y(1), LoadOptions { sem: MemoryOrder::Sc, scope: gpu, ..l });
        let evict_last = Some(CacheEvictionPriority::EvictLast);
        prefetch(&y(2), PrefetchOptions { cache_level: Some(PrefetchCacheLevel::L2), evict_priority: evict_last, ..p });
        let high = addc_cc(a1, b1);
        let hinted = load(&y(2), LoadOptions { level1_eviction_priority: evict_last, cache_policy: Some(policy), ..l });
        let volatile = load(&y(3), LoadOptions { sem: MemoryOrder::Volatile, ..l });
        prefetch(&y(4), PrefetchOptions { uniform: true, ..p });
        let plain = ys.load((t, 4));
        out32.store((t, 4), plain);
        let top = addc(Value::constant(0u64), Value::constant(0u64));
        // The difference.
        let d0 = sub_cc(b0, a0);
        let bytes = load_vector::<u8, 8>(&byte_row(&bs), LoadOptions { sem: MemoryOrder::Acquire, scope: gpu, ..l });
        store_vector(&byte_row(&ob), bytes, StoreOptions { sem: MemoryOrder::Relaxed, scope: gpu, ..s });
        let non_coherent = load(&y(5), LoadOptions { sem: MemoryOrder::Constant, ..l });
        prefetch(&y(7), PrefetchOptions { predicate: Some(odd), ..p });
        store(&o(1), sequential, StoreOptions { cop: Some(CacheOperator::Cs), ..s });
        store(&o(2), hinted, StoreOptions { sem: MemoryOrder::Volatile, ..s });
        let d1 = subc(b1, a1);
        for (j, v) in [low, high, top, d0, d1].into_iter().enumerate() {
            out64.store((t, j), v);
        }
        out32.store((t, 3), volatile);
        out32.store((t, 5), non_coherent);
        out32.store((t, 6), own.load());
        out32.store((t, 7), ys.load((t, 7)));
    });
    fn check(args: &[KernelArg]) {
        const W: usize = MEM_CARRY_WORDS;
        let (xs, ys, bs) = (args[0].to_vec::<u64>(), args[1].to_vec::<u32>(), args[2].to_vec::<u8>());
        let (out64, out32, ob) = (args[3].to_vec::<u64>(), args[4].to_vec::<u32>(), args[5].to_vec::<u8>());
        let (mut carries, mut borrows, mut top_carries) = (0, 0, 0);
        for t in 0..T {
            let w = |i: usize| xs[t * 4 + i].wrapping_mul(SPREAD);
            let odd = t % 2 == 1;
            let (a0, a1) = (if odd { u64::MAX } else { w(0) }, w(1));
            let (b0, b1) = (if odd { w(2) | 1 } else { w(2) }, w(3));
            let (a, b) = ((a1 as u128) << 64 | a0 as u128, (b1 as u128) << 64 | b0 as u128);
            let (sum, carry) = a.overflowing_add(b);
            let difference = b.wrapping_sub(a);
            carries += a0.overflowing_add(b0).1 as usize;
            borrows += (b0 < a0) as usize;
            top_carries += carry as usize;
            let want = [sum as u64, (sum >> 64) as u64, carry as u64, difference as u64, (difference >> 64) as u64];
            assert_eq!(out64[t * 5..t * 5 + 5], want, "thread {t}: a = {a:#x}, b = {b:#x}");
            let y = |j: usize| ys[t * W + j];
            let want = [y(0) ^ 0x0101_0101, y(1), y(2), y(3), y(4), y(5), y(6).wrapping_add(1), y(7)];
            assert_eq!(out32[t * W..(t + 1) * W], want, "thread {t}");
            assert_eq!(ob[t * 8..t * 8 + 8], bs[t * 8..t * 8 + 8], "thread {t}");
        }
        assert!(carries >= T / 2 && borrows >= T / 2, "the odd threads should carry and borrow: {carries}, {borrows}");
        // The carry out of the high word crosses the hinted and volatile loads: keep it exercised.
        assert!(top_carries > 0, "no thread carries out of the high word");
    }
    Case { name: "carry_around_memory_accesses", kernel, cfg: launch(1), small_only: false, check }
}

// ---------------------------------------------------------------------------
// Pack and Unpack, the packed and scalar half-precision operations, packed
// exp2 and the narrow conversions.
// ---------------------------------------------------------------------------

/// A reference for the half-precision instructions in plain Rust, from the
/// PTX ISA: a 16-bit float format with `p` significant bits, normal
/// exponents `emin..=emax`.
pub struct HalfFormat {
    pub p: i32,
    pub emin: i32,
    pub emax: i32,
    pub bf16: bool,
}

/// Float16.
pub const F16_FORMAT: HalfFormat = HalfFormat { p: 11, emin: -14, emax: 15, bf16: false };

/// BFloat16.
pub const BF16_FORMAT: HalfFormat = HalfFormat { p: 8, emin: -126, emax: 127, bf16: true };

impl HalfFormat {
    pub fn value(&self, bits: u16) -> f64 {
        if self.bf16 { f32::from_bits((bits as u32) << 16) as f64 } else { F16(bits).to_f32() as f64 }
    }

    /// The bits of a value the format holds exactly.
    pub fn bits(&self, v: f64) -> u16 {
        if self.bf16 { ((v as f32).to_bits() >> 16) as u16 } else { F16::from_f32(v as f32).0 }
    }

    pub fn is_nan(&self, bits: u16) -> bool {
        self.value(bits).is_nan()
    }

    pub fn is_subnormal(&self, bits: u16) -> bool {
        let v = self.value(bits).abs();
        v != 0.0 && v < 2f64.powi(self.emin)
    }

    /// The size of one unit in the last place at `|v|` (finite, nonzero).
    pub fn ulp(&self, v: f64) -> f64 {
        2f64.powi((v.abs().log2().floor() as i32).max(self.emin) - (self.p - 1))
    }

    /// The exact value `s + e` (an f64 and its error, `|e|` at most half an
    /// f64 ulp of `s`) rounded to nearest even; `zero_negative` is the sign
    /// of an exact zero.
    pub fn round(&self, s: f64, e: f64, zero_negative: bool) -> u16 {
        if s.is_nan() {
            return 0x7fff;
        }
        if s.is_infinite() {
            return self.bits(s);
        }
        if s == 0.0 && e == 0.0 {
            return if zero_negative { 0x8000 } else { 0 };
        }
        let ulp = self.ulp(s);
        let q = s / ulp;
        // A tie in s is decided by the error term.
        let r = if q - q.floor() == 0.5 && e != 0.0 {
            if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
        } else {
            q.round_ties_even()
        };
        let v = r * ulp;
        if v.abs() >= 2f64.powi(self.emax + 1) {
            return self.bits(if v > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY });
        }
        if v == 0.0 {
            return if s < 0.0 { 0x8000 } else { 0 };
        }
        self.bits(v)
    }

    /// Whether the exact nonzero value `s + e`, rounded to nearest even with
    /// `p` significant bits as though the exponent range were unbounded, is
    /// below the smallest normal number: IEEE 754's tininess after rounding,
    /// which decides `.ftz` on an SM89 GPU.
    pub fn tiny(&self, s: f64, e: f64) -> bool {
        let smallest = 2f64.powi(self.emin);
        if s == 0.0 || s.abs() >= smallest {
            return false;
        }
        let ulp = 2f64.powi(s.abs().log2().floor() as i32 - (self.p - 1));
        let q = s / ulp;
        let r = if q - q.floor() == 0.5 && e != 0.0 {
            if e > 0.0 { q.floor() + 1.0 } else { q.floor() }
        } else {
            q.round_ties_even()
        };
        (r * ulp).abs() < smallest
    }

    /// `a + b` exactly, as an f64 and its error (Knuth's TwoSum).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }

    /// One element of `add`, `sub` or `fma` with `mods`: `.ftz` flushes
    /// subnormal inputs and results to zeros of their signs (a result when
    /// it is tiny after rounding, as an SM89 GPU detects it), the exact
    /// result rounds once to nearest even, NaN results are `0x7fff`, `.sat`
    /// clamps to `[0, 1]` (NaN and `-0` to `+0`), `.relu` makes a negative
    /// result `+0`, and `.oob` gives `+0` for a multiplicand that is the out-of-bounds NaN
    /// `0x7fff` (any other NaN gives the canonical NaN, as a B200 does).
    pub fn op(&self, func: cute::ir::HalfFn, mods: cute::ir::HalfModifiers, a: u16, b: u16, c: u16) -> u16 {
        use cute::ir::HalfFn;
        if mods.oob && (a == 0x7fff || b == 0x7fff) {
            return 0;
        }
        let flush = |x: u16| if mods.ftz && self.is_subnormal(x) { x & 0x8000 } else { x };
        let (a, b, c) = (flush(a), flush(b), flush(c));
        let (x, y, z) = (self.value(a), self.value(b), self.value(c));
        let negative = |bits: u16| bits & 0x8000 != 0;
        let (s, e, zero_negative) = match func {
            HalfFn::Add | HalfFn::Sub => {
                let sub = func == HalfFn::Sub;
                let (s, e) = Self::two_sum(x, if sub { -y } else { y });
                let both_negative = negative(a) && (negative(b) != sub);
                (s, e, both_negative && x == 0.0 && y == 0.0)
            }
            HalfFn::Fma => {
                let product = x * y;
                let (s, e) = Self::two_sum(product, z);
                (s, e, product == 0.0 && z == 0.0 && (negative(a) != negative(b)) && negative(c))
            }
            HalfFn::Ex2 => return self.exp2(a),
        };
        let r = if mods.ftz && s.is_finite() && self.tiny(s, e) {
            if s < 0.0 { 0x8000 } else { 0 }
        } else {
            self.round(s, e, zero_negative)
        };
        let r = flush(r);
        let v = self.value(r);
        if mods.sat {
            return if v.is_nan() || v <= 0.0 {
                0
            } else if v >= 1.0 {
                self.bits(1.0)
            } else {
                r
            };
        }
        if v.is_nan() {
            0x7fff
        } else if mods.relu && negative(r) {
            0
        } else {
            r
        }
    }

    /// `2^a` rounded to nearest even (what `ex2.approx` approximates); in
    /// BFloat16 subnormal inputs and results flush to zero.
    pub fn exp2(&self, a: u16) -> u16 {
        let a = if self.bf16 && self.is_subnormal(a) { a & 0x8000 } else { a };
        let x = self.value(a);
        let r = if x.is_nan() { 0x7fff } else { self.round(x.exp2(), 0.0, false) };
        if self.bf16 && self.is_subnormal(r) { r & 0x8000 } else { r }
    }

    /// [`HalfFormat::op`] on both elements of packed words.
    pub fn op2(&self, func: cute::ir::HalfFn, mods: cute::ir::HalfModifiers, a: u32, b: u32, c: u32) -> u32 {
        let half = |w: u32, k: u32| (w >> (16 * k)) as u16;
        (0..2).fold(0, |d, k| d | (self.op(func, mods, half(a, k), half(b, k), half(c, k)) as u32) << (16 * k))
    }

    /// `a * b + c` rounded once, as `fma.rn` computes it (and `fma` on
    /// Float16 and BFloat16 values).
    pub fn fma(&self, a: u16, b: u16, c: u16) -> u16 {
        let rn = cute::ir::HalfModifiers { rn: true, ..cute::ir::HalfModifiers::default() };
        self.op(cute::ir::HalfFn::Fma, rn, a, b, c)
    }
}

/// The PTX modifiers of `HalfOptions` for the reference (fma always `.rn`).
pub fn half_mods(func: cute::ir::HalfFn, opts: HalfOptions) -> cute::ir::HalfModifiers {
    let fma = func == cute::ir::HalfFn::Fma;
    cute::ir::HalfModifiers { rn: opts.rn || fma, oob: opts.oob, ftz: opts.ftz, sat: opts.sat, relu: opts.relu }
}

/// Options for the kernels: `rn`, `ftz`, `sat` and `relu`.
pub fn half_opts(rn: bool, ftz: bool, sat: bool, relu: bool) -> HalfOptions {
    HalfOptions { rn, ftz, sat, relu, ..HalfOptions::default() }
}

/// Float16 bit patterns the packed kernels combine: signed zeros, the
/// subnormal extremes, the smallest normals, 1 and its neighbours, a third,
/// sums that tie (`1 + 2^-11`), the largest finite values, infinities, NaNs
/// (quiet, signaling, negative) and values whose products overflow.
pub const F16_SPECIALS: [u16; 22] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3c01, 0x3555, 0x1000, 0x1400,
    0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7d01, 0xfe55, 0x5bff,
];

/// BFloat16 bit patterns, as [`F16_SPECIALS`], with the factors of a product
/// that is a midpoint (`1.125 * 0.90625 = 261/256`).
pub const BF16_SPECIALS: [u16; 22] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3f81, 0x3eab, 0x3b80, 0x3c00,
    0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffc5, 0x3f90,
];

/// Rounds of table operands per thread in the half-precision kernels.
const HALF_ROUNDS: usize = 4;

/// The packed operands of round `r` of thread `t` from `table`: words `a`,
/// `b` and `c`, each two table entries, and the predicate (every other
/// thread and round).
pub fn half_table_operands(table: &[u16], t: usize, r: usize) -> (u32, u32, u32, bool) {
    let n = table.len();
    let idx = (t + T * r) % (n * n);
    let (i, j) = (idx / n, idx % n);
    let w = |lo: usize, hi: usize| table[lo % n] as u32 | (table[hi % n] as u32) << 16;
    (w(i, j), w(j, i + j), w(i + 3, j + 5), (t + r).is_multiple_of(2))
}

/// The same in a kernel: `table` holds the entries (read at run time, so
/// nothing folds).
fn half_table_operands_in_kernel(table: &Tensor<u32>, n: usize, t: Int32, r: usize) -> (Int32, Int32, Int32, Boolean) {
    let n = n as i32;
    let idx = (t + (T * r) as i32) % (n * n);
    let (i, j) = (idx / n, idx % n);
    let w = |lo: Int32, hi: Int32| (table.load(lo % n) | (table.load(hi % n) << 16u32)).bitcast::<i32>();
    (w(i, j), w(j, i + j), w(i + 3, j + 5), ((t + r as i32) % 2).eq(0))
}

/// A table of 16-bit patterns in local memory.
fn half_table(values: &[u16]) -> Tensor<u32> {
    let table = make_rmem_tensor::<u32>(Layout::col_major(values.len()));
    for (i, &v) in values.iter().enumerate() {
        table.store(i, v as u32);
    }
    table
}

/// A Float16 or BFloat16 as its bits in a Uint32.
fn half_bits<T: Numeric>(v: Value<T>) -> Uint32 {
    v.bitcast::<u16>().to::<u32>()
}

/// Columns of [`half_f16`].
const HALF_F16_COLS: usize = 24;

/// Packed and scalar Float16 arithmetic: `add_packed_f16x2` and
/// `sub_packed_f16x2` with every combination of `rn`, `ftz` and `sat` and
/// with a predicate, `fma_packed_f16x2` with `ftz`, `sat` and `relu`, the
/// tuple forms, `fma_f16` with options and `fma` on Float16 values; on pairs
/// of [`F16_SPECIALS`] and on the input data (as bits, so every pattern
/// occurs).
pub fn half_f16() -> Case {
    const COLS: usize = (HALF_ROUNDS + 1) * HALF_F16_COLS;
    let kernel = Kernel::builder("half_f16").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let table = half_table(&F16_SPECIALS);
        let mut sets: Vec<(Int32, Int32, Int32, Boolean)> =
            (0..HALF_ROUNDS).map(|r| half_table_operands_in_kernel(&table, F16_SPECIALS.len(), t, r)).collect();
        let x = xs.load(t);
        sets.push((x, ys.load(t), zs.load(t), (x & 4).eq(0)));
        for (r, (a, b, c, p)) in sets.into_iter().enumerate() {
            let o = half_opts;
            let [alo, ahi]: [Float16; 2] = unpack(a);
            let [blo, bhi]: [Float16; 2] = unpack(b);
            let [clo, chi]: [Float16; 2] = unpack(c);
            // The tuple forms on other halves than the packed ones, so that
            // they stay separate instructions.
            let tuple_add = add_packed_f16x2((alo, bhi), (blo, ahi), o(true, false, false, false));
            let tuple_fma = fma_packed_f16x2((ahi, alo), (bhi, blo), (chi, clo), o(false, false, false, true));
            let results: [Uint32; HALF_F16_COLS] = [
                add_packed_f16x2(a, b, HalfOptions::default()).bitcast(),
                add_packed_f16x2(a, b, o(true, false, false, false)).bitcast(),
                add_packed_f16x2(a, b, o(false, true, false, false)).bitcast(),
                add_packed_f16x2(a, b, o(false, false, true, false)).bitcast(),
                add_packed_f16x2(a, b, o(true, true, true, false)).bitcast(),
                sub_packed_f16x2(a, b, HalfOptions::default()).bitcast(),
                sub_packed_f16x2(a, b, o(true, false, false, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, true, false, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, false, true, false)).bitcast(),
                sub_packed_f16x2(a, b, o(false, true, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, HalfOptions::default()).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, false, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, false, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, false, false, true)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, true, false)).bitcast(),
                fma_packed_f16x2(a, b, c, o(false, true, false, true)).bitcast(),
                add_packed_f16x2(a, b, HalfOptions { predicate: Some(p), ..HalfOptions::default() }).bitcast(),
                sub_packed_f16x2(a, b, HalfOptions { predicate: Some(p), ftz: true, ..HalfOptions::default() })
                    .bitcast(),
                pack([tuple_add.0, tuple_add.1]),
                half_bits(fma_f16(alo, blo, clo, HalfOptions::default())),
                half_bits(fma_f16(ahi, bhi, chi, o(false, true, false, true))),
                half_bits(fma_f16(alo, bhi, chi, o(false, false, true, false))),
                half_bits(fma(ahi, blo, clo)),
                pack([tuple_fma.0, tuple_fma.1]),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * HALF_F16_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        use cute::ir::HalfFn::{Add, Fma, Sub};
        const COLS: usize = (HALF_ROUNDS + 1) * HALF_F16_COLS;
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        let f = &F16_FORMAT;
        let o = half_opts;
        for t in 0..T {
            let mut sets: Vec<(u32, u32, u32, bool)> =
                (0..HALF_ROUNDS).map(|r| half_table_operands(&F16_SPECIALS, t, r)).collect();
            sets.push((xs[t] as u32, ys[t] as u32, zs[t] as u32, xs[t] & 4 == 0));
            for (r, &(a, b, c, p)) in sets.iter().enumerate() {
                let op2 = |func, opts: HalfOptions, a, b, c| f.op2(func, half_mods(func, opts), a, b, c);
                let op = |func, opts: HalfOptions, a: u32, b: u32, c: u32| {
                    f.op(func, half_mods(func, opts), a as u16, b as u16, c as u16) as u32
                };
                let none = HalfOptions::default();
                let want: [u32; HALF_F16_COLS] = [
                    op2(Add, none, a, b, 0),
                    op2(Add, o(true, false, false, false), a, b, 0),
                    op2(Add, o(false, true, false, false), a, b, 0),
                    op2(Add, o(false, false, true, false), a, b, 0),
                    op2(Add, o(true, true, true, false), a, b, 0),
                    op2(Sub, none, a, b, 0),
                    op2(Sub, o(true, false, false, false), a, b, 0),
                    op2(Sub, o(false, true, false, false), a, b, 0),
                    op2(Sub, o(false, false, true, false), a, b, 0),
                    op2(Sub, o(false, true, true, false), a, b, 0),
                    op2(Fma, none, a, b, c),
                    op2(Fma, o(false, true, false, false), a, b, c),
                    op2(Fma, o(false, false, true, false), a, b, c),
                    op2(Fma, o(false, false, false, true), a, b, c),
                    op2(Fma, o(false, true, true, false), a, b, c),
                    op2(Fma, o(false, true, false, true), a, b, c),
                    if p { op2(Add, none, a, b, 0) } else { a },
                    if p { op2(Sub, o(false, true, false, false), a, b, 0) } else { a },
                    op2(
                        Add,
                        o(true, false, false, false),
                        a & 0xffff | b & 0xffff_0000,
                        b & 0xffff | a & 0xffff_0000,
                        0,
                    ),
                    op(Fma, none, a, b, c),
                    op(Fma, o(false, true, false, true), a >> 16, b >> 16, c >> 16),
                    op(Fma, o(false, false, true, false), a, b >> 16, c >> 16),
                    op(Fma, none, a >> 16, b, c),
                    op2(Fma, o(false, false, false, true), a.rotate_left(16), b.rotate_left(16), c.rotate_left(16)),
                ];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * HALF_F16_COLS + j];
                    assert_eq!(got, w, "half_f16 column {j}, thread {t}, round {r}: {a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
    }
    Case { name: "half_f16", kernel, cfg: launch(1), small_only: false, check }
}

/// Columns of [`half_bf16`].
const HALF_BF16_COLS: usize = 6;

/// BFloat16 fma, which needs sm_80: `fma_packed_bf16x2` with and without
/// `relu` (and its tuple form), `fma_bf16`, and `fma` on BFloat16 values,
/// now `fma.rn.bf16` rounding once; on pairs of [`BF16_SPECIALS`], on a
/// product that is a midpoint plus the smallest normal (rounding twice would
/// tie and go to even), and on the input data.
pub fn half_bf16() -> Case {
    const SETS: usize = HALF_ROUNDS + 2;
    const COLS: usize = SETS * HALF_BF16_COLS;
    let kernel = Kernel::builder("half_bf16").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let table = half_table(&BF16_SPECIALS);
        let mut sets: Vec<(Int32, Int32, Int32)> = (0..HALF_ROUNDS)
            .map(|r| {
                let (a, b, c, _) = half_table_operands_in_kernel(&table, BF16_SPECIALS.len(), t, r);
                (a, b, c)
            })
            .collect();
        let (a, b, c) = half_bf16_midpoint();
        sets.push((xs.load(t) & 0xffff_0000u32 as i32 | a, ys.load(t) & 0xffff_0000u32 as i32 | b, Int32::from(c)));
        sets.push((xs.load(t), ys.load(t), zs.load(t)));
        for (r, (a, b, c)) in sets.into_iter().enumerate() {
            let [alo, ahi]: [BFloat16; 2] = unpack(a);
            let [blo, bhi]: [BFloat16; 2] = unpack(b);
            let [clo, chi]: [BFloat16; 2] = unpack(c);
            let relu = HalfOptions { relu: true, ..HalfOptions::default() };
            let tuple = fma_packed_bf16x2((ahi, alo), (bhi, blo), (chi, clo), HalfOptions::default());
            let results: [Uint32; HALF_BF16_COLS] = [
                fma_packed_bf16x2(a, b, c, HalfOptions::default()).bitcast(),
                fma_packed_bf16x2(a, b, c, relu).bitcast(),
                half_bits(fma_bf16(alo, blo, clo, HalfOptions::default())),
                half_bits(fma_bf16(ahi, bhi, chi, relu)),
                half_bits(fma(ahi, blo, chi)),
                pack([tuple.0, tuple.1]),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * HALF_BF16_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        use cute::ir::HalfFn::Fma;
        const SETS: usize = HALF_ROUNDS + 2;
        const COLS: usize = SETS * HALF_BF16_COLS;
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        let f = &BF16_FORMAT;
        let none = half_mods(Fma, HalfOptions::default());
        let relu = half_mods(Fma, HalfOptions { relu: true, ..HalfOptions::default() });
        for t in 0..T {
            let mut sets: Vec<(u32, u32, u32)> = (0..HALF_ROUNDS)
                .map(|r| {
                    let (a, b, c, _) = half_table_operands(&BF16_SPECIALS, t, r);
                    (a, b, c)
                })
                .collect();
            let (a, b, c) = half_bf16_midpoint();
            let high = |v: i32| v as u32 & 0xffff_0000;
            sets.push((high(xs[t]) | a as u32, high(ys[t]) | b as u32, c as u32));
            sets.push((xs[t] as u32, ys[t] as u32, zs[t] as u32));
            for (r, &(a, b, c)) in sets.iter().enumerate() {
                let want = [
                    f.op2(Fma, none, a, b, c),
                    f.op2(Fma, relu, a, b, c),
                    f.op(Fma, none, a as u16, b as u16, c as u16) as u32,
                    f.op(Fma, relu, (a >> 16) as u16, (b >> 16) as u16, (c >> 16) as u16) as u32,
                    f.op(Fma, none, (a >> 16) as u16, b as u16, (c >> 16) as u16) as u32,
                    f.op2(Fma, none, a.rotate_left(16), b.rotate_left(16), c.rotate_left(16)),
                ];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * HALF_BF16_COLS + j];
                    assert_eq!(got, w, "half_bf16 column {j}, thread {t}, round {r}: {a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
        // The midpoint round: 261/256 + 2^-126 rounds up, to 1 + 3 * 2^-7.
        assert_eq!(out[COLS - 2 * HALF_BF16_COLS + 2] & 0xffff, 0x3f83);
    }
    Case { name: "half_bf16", kernel, cfg: launch(1), small_only: false, check }
}

/// BFloat16 `a`, `b` and `c` (low halves): `a * b` is 261/256, a midpoint,
/// and `c` the smallest normal, so the exact sum rounds up; through f32 it
/// would round to the midpoint and then to even (down).
pub fn half_bf16_midpoint() -> (i32, i32, i32) {
    (0x3f90, 0x3f68, 0x0080)
}

/// Columns of [`half_sm90`].
const HALF_SM90_COLS: usize = 12;

/// The half-precision operations that need sm_90: `add_packed_bf16x2` and
/// `sub_packed_bf16x2` (with `rn` and a predicate), `exp2_packed_bf16x2`
/// (with and without a predicate), and fma with `oob` on Float16 and
/// BFloat16 pairs and scalars, alone and with `relu`; on pairs of
/// [`BF16_SPECIALS`] (read as Float16 too) and on the input data.
pub fn half_sm90() -> Case {
    const COLS: usize = (HALF_ROUNDS + 1) * HALF_SM90_COLS;
    let kernel = Kernel::builder("half_sm90").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let ys = param_tensor::<i32>("ys", Layout::col_major(T));
        let zs = param_tensor::<i32>("zs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let table = half_table(&BF16_SPECIALS);
        let mut sets: Vec<(Int32, Int32, Int32, Boolean)> =
            (0..HALF_ROUNDS).map(|r| half_table_operands_in_kernel(&table, BF16_SPECIALS.len(), t, r)).collect();
        let x = xs.load(t);
        sets.push((x, ys.load(t), zs.load(t), (x & 4).eq(0)));
        for (r, (a, b, c, p)) in sets.into_iter().enumerate() {
            let [alo, _]: [Float16; 2] = unpack(a);
            let [blo, _]: [Float16; 2] = unpack(b);
            let [clo, _]: [Float16; 2] = unpack(c);
            let [_, ahi]: [BFloat16; 2] = unpack(a);
            let [_, bhi]: [BFloat16; 2] = unpack(b);
            let [_, chi]: [BFloat16; 2] = unpack(c);
            let oob = HalfOptions { oob: true, ..HalfOptions::default() };
            let oob_relu = HalfOptions { oob: true, relu: true, ..HalfOptions::default() };
            let rn_p = HalfOptions { rn: true, predicate: Some(p), ..HalfOptions::default() };
            let results: [Uint32; HALF_SM90_COLS] = [
                add_packed_bf16x2(a, b, HalfOptions::default()).bitcast(),
                add_packed_bf16x2(a, b, rn_p).bitcast(),
                sub_packed_bf16x2(a, b, HalfOptions::default()).bitcast(),
                sub_packed_bf16x2(a, b, HalfOptions { rn: true, ..HalfOptions::default() }).bitcast(),
                exp2_packed_bf16x2(a, None).bitcast(),
                exp2_packed_bf16x2(b, Some(p)).bitcast(),
                fma_packed_f16x2(a, b, c, oob).bitcast(),
                fma_packed_f16x2(a, b, c, oob_relu).bitcast(),
                fma_packed_bf16x2(a, b, c, oob).bitcast(),
                fma_packed_bf16x2(a, b, c, oob_relu).bitcast(),
                half_bits(fma_f16(alo, blo, clo, oob)),
                half_bits(fma_bf16(ahi, bhi, chi, oob_relu)),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * HALF_SM90_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        use cute::ir::HalfFn::{Add, Fma, Sub};
        const COLS: usize = (HALF_ROUNDS + 1) * HALF_SM90_COLS;
        let (xs, ys, zs) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        let out = args[3].to_vec::<u32>();
        let (h, b16) = (&F16_FORMAT, &BF16_FORMAT);
        let m = |func, opts: HalfOptions| half_mods(func, opts);
        let none = HalfOptions::default();
        let rn = HalfOptions { rn: true, ..none };
        let oob = HalfOptions { oob: true, ..none };
        let oob_relu = HalfOptions { oob: true, relu: true, ..none };
        let exp2 = |w: u32| b16.exp2(w as u16) as u32 | (b16.exp2((w >> 16) as u16) as u32) << 16;
        for t in 0..T {
            let mut sets: Vec<(u32, u32, u32, bool)> =
                (0..HALF_ROUNDS).map(|r| half_table_operands(&BF16_SPECIALS, t, r)).collect();
            sets.push((xs[t] as u32, ys[t] as u32, zs[t] as u32, xs[t] & 4 == 0));
            for (r, &(a, b, c, p)) in sets.iter().enumerate() {
                let hi = |w: u32| (w >> 16) as u16;
                let want: [u32; HALF_SM90_COLS] = [
                    b16.op2(Add, m(Add, none), a, b, 0),
                    if p { b16.op2(Add, m(Add, rn), a, b, 0) } else { a },
                    b16.op2(Sub, m(Sub, none), a, b, 0),
                    b16.op2(Sub, m(Sub, rn), a, b, 0),
                    exp2(a),
                    if p { exp2(b) } else { b },
                    h.op2(Fma, m(Fma, oob), a, b, c),
                    h.op2(Fma, m(Fma, oob_relu), a, b, c),
                    b16.op2(Fma, m(Fma, oob), a, b, c),
                    b16.op2(Fma, m(Fma, oob_relu), a, b, c),
                    h.op(Fma, m(Fma, oob), a as u16, b as u16, c as u16) as u32,
                    b16.op(Fma, m(Fma, oob_relu), hi(a), hi(b), hi(c)) as u32,
                ];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * HALF_SM90_COLS + j];
                    assert_eq!(got, w, "half_sm90 column {j}, thread {t}, round {r}: {a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
    }
    Case { name: "half_sm90", kernel, cfg: launch(1), small_only: false, check }
}

/// Words of [`exp2_f16x2`]: each holds two Float16 values, `2g` and
/// `2g + 1` for word `g`, so every value occurs once.
pub const EXP2_WORDS: usize = 32768;

/// `exp2_packed_f16x2` on every Float16 value, alone and with a predicate
/// (true in two words of three); a word per thread, no inputs. The GPU's
/// `ex2.approx.f16x2` is an approximation: `tests/gpu.rs` holds it to its
/// bound, the simulator to the correctly rounded value.
pub fn exp2_f16x2() -> Case {
    let kernel = Kernel::builder("exp2_f16x2").block_dim([256, 1, 1]).build(|| {
        let out = param_tensor::<u32>("out", Layout::col_major(EXP2_WORDS));
        let guarded = param_tensor::<u32>("guarded", Layout::col_major(EXP2_WORDS));
        let g = block_idx().x * 256 + thread_idx().x;
        let word = (g * 2) | ((g * 2 + 1) << 16);
        out.store(g, exp2_packed_f16x2(word, None).bitcast::<u32>());
        let p = (g % 3).ne(0);
        guarded.store(g, exp2_packed_f16x2(word, Some(p)).bitcast::<u32>());
    });
    fn check(args: &[KernelArg]) {
        let (out, guarded) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        let f = &F16_FORMAT;
        for g in 0..EXP2_WORDS {
            let word = (2 * g as u32) | ((2 * g as u32 + 1) << 16);
            let want = f.exp2(word as u16) as u32 | (f.exp2((word >> 16) as u16) as u32) << 16;
            assert_eq!(out[g], want, "exp2_packed_f16x2 of {word:#010x}");
            assert_eq!(guarded[g], if g % 3 != 0 { want } else { word }, "predicated, word {g}");
        }
    }
    Case { name: "exp2_f16x2", kernel, cfg: launch_n((EXP2_WORDS / 256) as u32, 256), small_only: false, check }
}

/// Float32 bit patterns for the conversions: signed zeros, subnormals (a
/// BFloat16 tie among them), 1, BFloat16 and TF32 ties (`0x...8000`,
/// `0x...1000`), the largest finite values and the patterns just around the
/// BFloat16 and TF32 overflow thresholds, infinities, and NaNs whose payload
/// lies in the high bits, only in the low 13 bits, or both.
pub const CVT_F32_SPECIALS: [u32; 30] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x0001_8000,
    0x007f_ffff,
    0x007f_f000,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_1000,
    0xbf80_1000,
    0x3f80_3000,
    0x3f80_8000,
    0x3f81_8000,
    0x3f80_0fff,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f7f_8000,
    0x7f7f_7fff,
    0x7f7f_f000,
    0x7f7f_efff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x7f80_1000,
    0x7fff_ffff,
    0x7fbf_ffff,
];

/// `cvt.rn.satfinite.bf16.f32` on Float32 bits, from the PTX ISA: NaN gives
/// the canonical NaN, a magnitude above BFloat16's largest finite value its
/// largest finite value with the sign, others round to nearest even.
pub fn bf16_satfinite_ref(x: u32) -> u16 {
    let v = f32::from_bits(x);
    let f = &BF16_FORMAT;
    if v.is_nan() {
        return 0x7fff;
    }
    let largest = f.value(0x7f7f);
    if (v as f64).abs() > largest {
        return if v < 0.0 { 0xff7f } else { 0x7f7f };
    }
    f.round(v as f64, 0.0, v.is_sign_negative())
}

/// `cvt.rn.bf16.f32` on Float32 bits: to nearest even, overflowing to
/// infinity; NaN gives the canonical NaN (as on an SM89 GPU).
pub fn bf16_rn_ref(x: u32) -> u16 {
    let v = f32::from_bits(x);
    if v.is_nan() { 0x7fff } else { BF16_FORMAT.round(v as f64, 0.0, v.is_sign_negative()) }
}

/// `cvt.rna.tf32.f32` on Float32 bits: the nearest value with 11 significant
/// bits, ties away from zero, overflowing to infinity; infinities and NaNs
/// keep their bits but the low 13 (what an SM89 GPU does).
pub fn tf32_ref(x: u32) -> u32 {
    let v = f32::from_bits(x) as f64;
    if !v.is_finite() {
        return x & !0x1fff;
    }
    if v == 0.0 {
        return x;
    }
    let ulp = 2f64.powi((v.abs().log2().floor() as i32).max(-126) - 10);
    let magnitude = (v.abs() / ulp + 0.5).floor() * ulp;
    let r = if v < 0.0 { -magnitude } else { magnitude };
    if r.abs() >= 2f64.powi(128) { (x & 0x8000_0000) | 0x7f80_0000 } else { (r as f32).to_bits() }
}

/// Rounds of table pairs per thread in [`cvt_floats`].
const CVT_ROUNDS: usize = 15;

/// Columns of [`cvt_floats`].
const CVT_COLS: usize = 5;

/// The Float32 conversions: `cvt_f32x2_bf16x2` (packed, and as a tuple of
/// the swapped pair), `cvt_f32_bf16`, `cvt_f32_tf32` and `Value::to_tf32`,
/// on every pair of [`CVT_F32_SPECIALS`] and on the input data. Needs PTX 8.1
/// (`.satfinite`).
pub fn cvt_floats() -> Case {
    const COLS: usize = (CVT_ROUNDS + 1) * CVT_COLS;
    let kernel = Kernel::builder("cvt_floats").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<f32>("xs", Layout::col_major(T));
        let ys = param_tensor::<f32>("ys", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let n = CVT_F32_SPECIALS.len() as i32;
        let table = make_rmem_tensor::<u32>(Layout::col_major(CVT_F32_SPECIALS.len()));
        for (i, v) in CVT_F32_SPECIALS.into_iter().enumerate() {
            table.store(i, v);
        }
        let mut pairs: Vec<(Float32, Float32)> = (0..CVT_ROUNDS as i32)
            .map(|r| {
                let p = (t + T as i32 * r) % (n * n);
                (table.load(p / n).bitcast::<f32>(), table.load(p % n).bitcast::<f32>())
            })
            .collect();
        pairs.push((xs.load(t), ys.load(t)));
        for (r, (x, y)) in pairs.into_iter().enumerate() {
            let packed: Int32 = cvt_f32x2_bf16x2([x, y]);
            let (lo, hi): (BFloat16, BFloat16) = cvt_f32x2_bf16x2([y, x]);
            let results: [Uint32; CVT_COLS] = [
                packed.bitcast(),
                pack([lo, hi]),
                half_bits(cvt_f32_bf16(x)),
                cvt_f32_tf32(x).bitcast(),
                y.to_tf32().bitcast(),
            ];
            for (j, v) in results.into_iter().enumerate() {
                out.store((t, r * CVT_COLS + j), v);
            }
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = (CVT_ROUNDS + 1) * CVT_COLS;
        let (xs, ys, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<u32>());
        let n = CVT_F32_SPECIALS.len();
        for t in 0..T {
            let mut pairs: Vec<(u32, u32)> = (0..CVT_ROUNDS)
                .map(|r| (t + T * r) % (n * n))
                .map(|p| (CVT_F32_SPECIALS[p / n], CVT_F32_SPECIALS[p % n]))
                .collect();
            pairs.push((xs[t].to_bits(), ys[t].to_bits()));
            for (r, &(x, y)) in pairs.iter().enumerate() {
                let sat = |lo: u32, hi: u32| bf16_satfinite_ref(lo) as u32 | (bf16_satfinite_ref(hi) as u32) << 16;
                let want = [sat(x, y), sat(y, x), bf16_rn_ref(x) as u32, tf32_ref(x), tf32_ref(y)];
                for (j, w) in want.into_iter().enumerate() {
                    let got = out[t * COLS + r * CVT_COLS + j];
                    assert_eq!(got, w, "cvt_floats column {j}, thread {t}: {x:#010x} {y:#010x}: got {got:#010x}");
                }
            }
        }
    }
    Case { name: "cvt_floats", kernel, cfg: launch(1), small_only: false, check }
}

/// Int8 values of thread `t`'s `k`-th operand in [`cvt_int8`]: consecutive
/// over the threads, so every value occurs.
pub fn cvt_int8_value(t: usize, k: usize, per_thread: usize) -> i8 {
    (t * per_thread + k) as u8 as i8
}

/// Columns of [`cvt_int8`]: 18 Float32 results, then 45 BFloat16 ones.
const CVT_INT8_COLS: usize = 18 + 45;

/// The int8 conversions: `cvt_i8x4_to_f32x4` on every int8 value (four per
/// thread, as an array) and on the input data as an Int32 and a Uint32
/// word, `cvt_i8x2_to_f32x2` on arrays and on Int16 and Uint16 halves of the
/// data, and `cvt_i8_bf16_intrinsic` (the Ampere path, which every target
/// has) on every length from 1 to 9 (nine consecutive values per thread).
pub fn cvt_int8() -> Case {
    let kernel = Kernel::builder("cvt_int8").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<i32>("xs", Layout::col_major(T));
        let out = param_tensor::<u32>("out", Layout::row_major((T, CVT_INT8_COLS)));
        let t = thread_idx().x;
        let x = xs.load(t);
        let value = |k: i32, per: i32| ((t * per + k) & 0xff).to::<i8>();
        let quad: [Int8; 4] = std::array::from_fn(|k| value(k as i32, 4));
        let mut results: Vec<Uint32> = Vec::new();
        let bits = |v: Float32| v.bitcast::<u32>();
        results.extend(cvt_i8x4_to_f32x4(quad).map(bits));
        results.extend(cvt_i8x4_to_f32x4(x).map(bits));
        results.extend(cvt_i8x4_to_f32x4(x.bitcast::<u32>().rotate_bytes()).map(bits));
        results.extend(cvt_i8x2_to_f32x2([quad[1], quad[3]]).map(bits));
        results.extend(cvt_i8x2_to_f32x2(x.to::<i16>()).map(bits));
        results.extend(cvt_i8x2_to_f32x2((x.bitcast::<u32>() >> 16u32).to::<u16>()).map(bits));
        let nine: Vec<Int8> = (0..9).map(|k| value(k, 9)).collect();
        for len in 1..=9 {
            results.extend(cvt_i8_bf16_intrinsic(&nine, len).into_iter().map(half_bits));
        }
        assert_eq!(results.len(), CVT_INT8_COLS);
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<i32>(), args[1].to_vec::<u32>());
        for t in 0..T {
            let x = xs[t] as u32;
            let bytes = |w: u32| w.to_le_bytes().map(|b| b as i8);
            let mut want: Vec<u32> = Vec::new();
            let quad: [i8; 4] = std::array::from_fn(|k| cvt_int8_value(t, k, 4));
            let f = |v: i8| (v as f32).to_bits();
            want.extend(quad.map(f));
            want.extend(bytes(x).map(f));
            want.extend(bytes(x.rotate_right(8)).map(f));
            want.extend([quad[1], quad[3]].map(f));
            want.extend([bytes(x)[0], bytes(x)[1]].map(f));
            want.extend([bytes(x)[2], bytes(x)[3]].map(f));
            let nine: Vec<i8> = (0..9).map(|k| cvt_int8_value(t, k, 9)).collect();
            for len in 1..=9 {
                want.extend(nine[..len].iter().map(|&v| BF16::from_f32(v as f32).0 as u32));
            }
            assert_eq!(out[t * CVT_INT8_COLS..(t + 1) * CVT_INT8_COLS], want, "cvt_int8, thread {t}, data {x:#010x}");
        }
    }
    Case { name: "cvt_int8", kernel, cfg: launch(1), small_only: false, check }
}

/// A Uint32's bytes rotated down by one (byte 1 to byte 0), with a funnel
/// shift.
trait RotateBytes {
    fn rotate_bytes(self) -> Uint32;
}

impl RotateBytes for Uint32 {
    fn rotate_bytes(self) -> Uint32 {
        shf(self, self, 8, ShfKind::Right)
    }
}

/// Unpacked int4 words of [`sext_i4`]: word `i` holds the four hex digits of
/// `i`, one per byte (digit 0 in byte 0).
pub fn sext_word(i: u32) -> u32 {
    (0..4).fold(0, |w, k| w | ((i >> (4 * k)) & 0xf) << (8 * k))
}

/// Threads of [`sext_i4`], and words per thread.
const SEXT_THREADS: usize = 4096;
const SEXT_PER_THREAD: usize = 16;

/// `sext_unpacked_i4x4_to_i8x4` on every combination of four int4 values
/// (65536 words) as Uint32 words, on the first of each thread's words as an
/// array, and `sext_unpacked_i4_i8_intrinsic` on its first two words (eight
/// values).
pub fn sext_i4() -> Case {
    let words = SEXT_THREADS * SEXT_PER_THREAD;
    let kernel = Kernel::builder("sext_i4").block_dim([256, 1, 1]).build(move || {
        let out = param_tensor::<u32>("out", Layout::col_major(words));
        let out4 = param_tensor::<i8>("out4", Layout::col_major(SEXT_THREADS * 4));
        let out8 = param_tensor::<i8>("out8", Layout::col_major(SEXT_THREADS * 8));
        let g = block_idx().x * 256 + thread_idx().x;
        let word = |i: Int32| {
            let i = i.bitcast::<u32>();
            (0..4).fold(Uint32::from(0u32), |w, k| w | ((i >> (4 * k) as u32) & 0xfu32) << (8 * k) as u32)
        };
        let mut first: Vec<Uint32> = Vec::new();
        for k in 0..SEXT_PER_THREAD as i32 {
            let i = g * SEXT_PER_THREAD as i32 + k;
            let w = word(i);
            out.store(i, sext_unpacked_i4x4_to_i8x4(w));
            if k < 2 {
                first.push(w);
            }
        }
        let array: [Int8; 4] = unpack(first[0]);
        for (k, v) in sext_unpacked_i4x4_to_i8x4(array).into_iter().enumerate() {
            out4.store(g * 4 + k as i32, v);
        }
        let values: Vec<Int8> = first.iter().flat_map(|&w| unpack::<i8, u32, 4>(w)).collect();
        for (k, v) in sext_unpacked_i4_i8_intrinsic(&values, 8).into_iter().enumerate() {
            out8.store(g * 8 + k as i32, v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (out, out4, out8) = (args[0].to_vec::<u32>(), args[1].to_vec::<i8>(), args[2].to_vec::<i8>());
        let sext = |w: u32| w.to_le_bytes().map(|b| ((b as i8) << 4) >> 4);
        for (i, &got) in out.iter().enumerate() {
            let want = u32::from_le_bytes(sext(sext_word(i as u32)).map(|b| b as u8));
            assert_eq!(got, want, "sext of {:#010x}", sext_word(i as u32));
        }
        for g in 0..SEXT_THREADS {
            let first = (g * SEXT_PER_THREAD) as u32;
            assert_eq!(out4[g * 4..g * 4 + 4], sext(sext_word(first)), "array form, thread {g}");
            let both: Vec<i8> = sext(sext_word(first)).into_iter().chain(sext(sext_word(first + 1))).collect();
            assert_eq!(out8[g * 8..g * 8 + 8], both[..], "intrinsic, thread {g}");
        }
    }
    let cfg = launch_n((SEXT_THREADS / 256) as u32, 256);
    Case { name: "sext_i4", kernel, cfg, small_only: false, check }
}

/// `cvt_i8_bf16` (`cvt.rn.bf16.s8`, sm_90) on every int8 value, four per
/// thread.
pub fn cvt_i8_bf16_sm90() -> Case {
    let kernel = Kernel::builder("cvt_i8_bf16_sm90").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<BF16>("out", Layout::col_major(4 * T));
        let t = thread_idx().x;
        for k in 0..4 {
            let v = ((t * 4 + k) & 0xff).to::<i8>();
            out.store(t * 4 + k, cvt_i8_bf16(v));
        }
    });
    fn check(args: &[KernelArg]) {
        let out = args[0].to_vec::<BF16>();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.0, BF16::from_f32(i as u8 as i8 as f32).0, "cvt_i8_bf16 of {}", i as u8 as i8);
        }
    }
    Case { name: "cvt_i8_bf16_sm90", kernel, cfg: launch(1), small_only: false, check }
}

/// The int8 to BFloat16 conversions of sm_100a-class targets (PTX 9.1):
/// `cvt_i8x2_to_bf16x2`, `cvt_i8x4_to_bf16x4` and `cvt_i8_bf16_intrinsic`
/// on the s2f6x2 path for every length from 1 to 9. Neither `ptxas` 12.8 nor
/// LLVM 21 or 22 handles PTX 9.1: it is in no kernel list, but simulated and
/// checked as MLIR (and as PTX when the `mlir-opt` is LLVM 23).
pub fn cvt_i8_bf16_s2f6x2() -> Case {
    const COLS: usize = 6 + 45;
    let kernel = Kernel::builder("cvt_i8_bf16_s2f6x2").block_dim([T as u32, 1, 1]).build(|| {
        let out = param_tensor::<BF16>("out", Layout::row_major((T, COLS)));
        let t = thread_idx().x;
        let value = |k: i32, per: i32| ((t * per + k) & 0xff).to::<i8>();
        let quad: [Int8; 4] = std::array::from_fn(|k| value(k as i32, 4));
        let mut results: Vec<BFloat16> = Vec::new();
        results.extend(cvt_i8x4_to_bf16x4(quad));
        let (lo, hi): (BFloat16, BFloat16) = cvt_i8x2_to_bf16x2([quad[2], quad[0]]);
        results.extend([lo, hi]);
        let nine: Vec<Int8> = (0..9).map(|k| value(k, 9)).collect();
        for len in 1..=9 {
            results.extend(cvt_i8_bf16_intrinsic_with(&nine, len, I8ToBf16Path::S2f6x2));
        }
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        const COLS: usize = 6 + 45;
        let out = args[0].to_vec::<BF16>();
        let bf = |v: i8| BF16::from_f32(v as f32).0;
        for t in 0..T {
            let quad: [i8; 4] = std::array::from_fn(|k| cvt_int8_value(t, k, 4));
            let mut want: Vec<u16> = quad.map(bf).to_vec();
            want.extend([bf(quad[2]), bf(quad[0])]);
            let nine: Vec<i8> = (0..9).map(|k| cvt_int8_value(t, k, 9)).collect();
            for len in 1..=9 {
                want.extend(nine[..len].iter().map(|&v| bf(v)));
            }
            let got: Vec<u16> = out[t * COLS..(t + 1) * COLS].iter().map(|v| v.0).collect();
            assert_eq!(got, want, "thread {t}");
        }
    }
    Case { name: "cvt_i8_bf16_s2f6x2", kernel, cfg: launch(1), small_only: false, check }
}

/// Columns of [`pack_unpack`].
const PACK_COLS: usize = 20;

/// `pack` and `unpack` between every width: two Float16, two BFloat16, four
/// Int8 and two Uint8 values into 32 and 16 bits, two Int16 into a Float32
/// (NaN patterns kept), two Int32 or Float32 and four Uint16 into 64 bits,
/// eight Int8 into a Float64, and back, on the input data's bits.
pub fn pack_unpack() -> Case {
    let kernel = Kernel::builder("pack_unpack").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(T));
        let ys = param_tensor::<u32>("ys", Layout::col_major(T));
        let out = param_tensor::<u64>("out", Layout::row_major((T, PACK_COLS)));
        let t = thread_idx().x;
        let (x, y) = (xs.load(t), ys.load(t) * 0x9e37_79b9u32);
        let [x0, x1]: [Float16; 2] = unpack(x);
        let [y0, y1]: [Float16; 2] = unpack(y);
        let [bx0, bx1]: [BFloat16; 2] = unpack(x);
        let [by0, by1]: [BFloat16; 2] = unpack(y);
        let xb: [Int8; 4] = unpack(x);
        let yb: [Int8; 4] = unpack(y);
        let xu: [Uint8; 4] = unpack(x);
        let yu: [Uint8; 4] = unpack(y);
        let xh: [Int16; 2] = unpack(x);
        let yh: [Uint16; 2] = unpack(y);
        let wide = |v: Uint32| v.to::<u64>();
        let i32_bits = |v: Int32| wide(v.bitcast());
        let xy: Int64 = pack([x.bitcast::<i32>(), y.bitcast::<i32>()]);
        let floats: Uint64 = pack([x.bitcast::<f32>(), y.bitcast::<f32>()]);
        let quads: Int64 = pack([xh[1].bitcast::<u16>(), yh[0], xh[0].bitcast::<u16>(), yh[1]]);
        let eight: Float64 = pack([xb[0], yb[1], xb[2], yb[3], yb[0], xb[1], yb[2], xb[3]]);
        let [w0, w1]: [Float32; 2] = unpack(floats);
        let bytes: [Uint8; 8] = unpack(xy);
        let results: [Uint64; PACK_COLS] = [
            i32_bits(pack([x0, y1])),
            wide(pack([by0, bx1])),
            wide(pack([xb[0], yb[1], xb[2], yb[3]])),
            pack::<u8, i16, 2>([xu[3], yu[0]]).bitcast::<u16>().to::<u64>(),
            wide(pack::<i16, f32, 2>([xh[0], yh[1].bitcast()]).bitcast()),
            xy.bitcast(),
            floats,
            quads.bitcast(),
            eight.bitcast(),
            wide(w0.bitcast()),
            wide(w1.bitcast()),
            bytes[0].to(),
            bytes[3].to(),
            bytes[4].to(),
            bytes[7].to(),
            half_bits(x1).to(),
            half_bits(y0).to(),
            half_bits(bx0).to(),
            half_bits(by1).to(),
            wide(pack([x0, x1])),
        ];
        for (j, v) in results.into_iter().enumerate() {
            out.store((t, j), v);
        }
    });
    fn check(args: &[KernelArg]) {
        let (xs, ys, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>(), args[2].to_vec::<u64>());
        for t in 0..T {
            let (x, y) = (xs[t], ys[t].wrapping_mul(0x9e37_79b9));
            let (xb, yb) = (x.to_le_bytes(), y.to_le_bytes());
            let (lo, hi) = (|w: u32| (w & 0xffff) as u64, |w: u32| (w >> 16) as u64);
            let join8 = |b: [u8; 8]| u64::from_le_bytes(b);
            let want: [u64; PACK_COLS] = [
                lo(x) | hi(y) << 16,
                lo(y) | hi(x) << 16,
                u32::from_le_bytes([xb[0], yb[1], xb[2], yb[3]]) as u64,
                (xb[3] as u64) | (yb[0] as u64) << 8,
                lo(x) | hi(y) << 16,
                x as u64 | (y as u64) << 32,
                x as u64 | (y as u64) << 32,
                hi(x) | lo(y) << 16 | lo(x) << 32 | hi(y) << 48,
                join8([xb[0], yb[1], xb[2], yb[3], yb[0], xb[1], yb[2], xb[3]]),
                x as u64,
                y as u64,
                xb[0] as u64,
                xb[3] as u64,
                yb[0] as u64,
                yb[3] as u64,
                hi(x),
                lo(y),
                lo(x),
                hi(y),
                x as u64,
            ];
            for (j, w) in want.into_iter().enumerate() {
                assert_eq!(out[t * PACK_COLS + j], w, "pack_unpack column {j}, thread {t}: {x:#010x} {y:#010x}");
            }
        }
    }
    Case { name: "pack_unpack", kernel, cfg: launch(1), small_only: false, check }
}

/// The kernels that run on every target (in [`all`]): Pack
/// and Unpack, Float16 and BFloat16 (fma) arithmetic, the Float32 and int8
/// conversions and the int4 sign extension.
pub fn packed_half_and_conversions() -> Vec<Case> {
    vec![pack_unpack(), half_f16(), half_bf16(), cvt_floats(), cvt_int8(), sext_i4()]
}

/// The half-precision sweep for `tests/gpu.rs`: `func` on `n` words (pairs
/// of `elem` values) for each of `options`, `out[i * options.len() + m]`
/// from `a[i]`, `b[i]` and `c[i]`; the packed forms, or with `scalar` the
/// scalar fma (`fma_f16`, `fma_bf16`) on the low halves, or with `plain` the
/// `fma` of `Value`s.
pub fn half_sweep(func: cute::ir::HalfFn, elem: DType, form: SweepForm, options: &[HalfOptions], n: usize) -> Kernel {
    use cute::ir::HalfFn;
    let options = options.to_vec();
    let name = format!("sweep_{}_{}_{:?}", func.ptx(), elem.name(), form).to_lowercase();
    Kernel::builder(name).block_dim([256, 1, 1]).build(move || {
        let a = param_tensor::<i32>("a", Layout::col_major(n));
        let b = param_tensor::<i32>("b", Layout::col_major(n));
        let c = param_tensor::<i32>("c", Layout::col_major(n));
        let out = param_tensor::<i32>("out", Layout::col_major(n * options.len()));
        let i = block_idx().x * 256 + thread_idx().x;
        let (x, y, z) = (a.load(i), b.load(i), c.load(i));
        let m = options.len() as i32;
        for (k, &opts) in options.iter().enumerate() {
            let r: Int32 = match (form, elem) {
                (SweepForm::Packed, DType::F16) => match func {
                    HalfFn::Add => add_packed_f16x2(x, y, opts),
                    HalfFn::Sub => sub_packed_f16x2(x, y, opts),
                    HalfFn::Fma => fma_packed_f16x2(x, y, z, opts),
                    HalfFn::Ex2 => exp2_packed_f16x2(x, opts.predicate),
                },
                (SweepForm::Packed, _) => match func {
                    HalfFn::Add => add_packed_bf16x2(x, y, opts),
                    HalfFn::Sub => sub_packed_bf16x2(x, y, opts),
                    HalfFn::Fma => fma_packed_bf16x2(x, y, z, opts),
                    HalfFn::Ex2 => exp2_packed_bf16x2(x, opts.predicate),
                },
                (_, DType::F16) => {
                    let [xl, _]: [Float16; 2] = unpack(x);
                    let [yl, _]: [Float16; 2] = unpack(y);
                    let [zl, _]: [Float16; 2] = unpack(z);
                    let r = if form == SweepForm::Scalar { fma_f16(xl, yl, zl, opts) } else { fma(xl, yl, zl) };
                    half_bits(r).bitcast()
                }
                _ => {
                    let [xl, _]: [BFloat16; 2] = unpack(x);
                    let [yl, _]: [BFloat16; 2] = unpack(y);
                    let [zl, _]: [BFloat16; 2] = unpack(z);
                    let r = if form == SweepForm::Scalar { fma_bf16(xl, yl, zl, opts) } else { fma(xl, yl, zl) };
                    half_bits(r).bitcast()
                }
            };
            out.store(i * m + k as i32, r);
        }
    })
}

/// Which form of the operation [`half_sweep`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepForm {
    /// The packed operation on Int32 pairs.
    Packed,
    /// `fma_f16` or `fma_bf16` on the low halves.
    Scalar,
    /// `fma` on the low halves' `Value`s.
    Plain,
}

/// The Float32 conversions on `n` bit patterns for `tests/gpu.rs`:
/// `out[3i..3i + 3]` is `cvt_f32x2_bf16x2([x, -x])`, `cvt_f32_bf16(x)` and
/// `cvt_f32_tf32(x)` for `x = xs[i]`.
pub fn cvt_sweep(n: usize) -> Kernel {
    Kernel::builder("cvt_sweep").block_dim([256, 1, 1]).build(move || {
        let xs = param_tensor::<u32>("xs", Layout::col_major(n));
        let out = param_tensor::<u32>("out", Layout::col_major(3 * n));
        let i = block_idx().x * 256 + thread_idx().x;
        let bits = xs.load(i);
        let (x, negated) = (bits.bitcast::<f32>(), (bits ^ 0x8000_0000u32).bitcast::<f32>());
        out.store(i * 3, cvt_f32x2_bf16x2::<Int32>([x, negated]).bitcast::<u32>());
        out.store(i * 3 + 1, half_bits(cvt_f32_bf16(x)));
        out.store(i * 3 + 2, cvt_f32_tf32(x).bitcast::<u32>());
    })
}

/// Float32 patterns for [`cvt_sweep`]: every sign and exponent with
/// mantissas around the BFloat16 and TF32 rounding points (ties, one either
/// side, the extremes) and pseudo-random ones.
pub fn cvt_sweep_inputs() -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let edges =
        [0u32, 1, 0x7f_ffff, 0x40_0000, 0x0fff, 0x1000, 0x1001, 0x1fff, 0x3000, 0x7fff, 0x8000, 0x8001, 0x1_8000];
    let mut v = Vec::new();
    for top in 0..512u32 {
        for (k, &m) in edges.iter().enumerate() {
            v.push(top << 23 | m);
            v.push(top << 23 | (m ^ (0x7f_0000 >> k)) & 0x7f_ffff);
        }
        for _ in 0..998 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(top << 23 | (state as u32 & 0x7f_ffff));
        }
    }
    v
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

// Warp-specialised hand-offs whose waits come before the matching arrivals in program order.

/// Threads per block of [`consumer_first_named_barrier`]: four producer and four consumer warps.
const HANDOFF_BLOCK: usize = 256;

/// The consumer warps (4-7, the `then` branch, so first in program order) wait at named
/// barrier 1 of all 256 threads before the producer warps (0-3) have stored their words into
/// shared memory and arrived with `bar.arrive`; then each consumer reads a producer's word.
pub fn consumer_first_named_barrier() -> Case {
    const N: usize = HANDOFF_BLOCK;
    let kernel = Kernel::builder("consumer_first_named_barrier").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N / 2));
        let out = param_tensor::<u32>("out", Layout::col_major(N / 2));
        let smem = make_smem_tensor::<u32>(Layout::col_major(N / 2));
        let t = thread_idx().x;
        if_else(
            warp_idx().ge(4),
            || {
                barrier(1, N as i32);
                out.store(t - N as i32 / 2, smem.load((t + 32) % (N as i32 / 2)));
            },
            || {
                smem.store(t, xs.load(t) * 3u32 + 1u32);
                barrier_arrive(1, N as i32);
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        for (c, &got) in out.iter().enumerate() {
            let p = (c + 32) % (N / 2);
            assert_eq!(got, xs[p].wrapping_mul(3).wrapping_add(1), "consumer thread {}", c + N / 2);
        }
    }
    Case { name: "consumer_first_named_barrier", kernel, cfg: launch_n(1, N), small_only: false, check }
}

/// Warp 0 spins on `mbarrier_test_wait` (the `then` branch, so before the producer in program
/// order) until warp 1 has stored its words into shared memory and arrived; then it reads them.
/// The spin gives up after [`SPIN_LIMIT`] polls, and each thread stores whether it did.
pub fn consumer_first_mbarrier() -> Case {
    let kernel = Kernel::builder("consumer_first_mbarrier").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_spin(&bar, 0.into_value());
                out.store(lane, data.load((lane + 1) % 32));
                select(missed, 1u32, 0u32)
            },
            || {
                data.store(lane, xs.load(lane) ^ 0x5a5a_5a5au32);
                mbarrier_arrive(&bar);
                0u32.into_value()
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            assert_eq!(got, xs[(l + 1) % 32] ^ 0x5a5a_5a5a, "lane {l}");
        }
    }
    Case { name: "consumer_first_mbarrier", kernel, cfg: launch(1), small_only: false, check }
}

/// Iterations of [`mbarrier_ring`]'s loop: each of its two stages is filled four times.
const RING_ITERS: usize = 8;

/// A two-stage ring of shared-memory buffers between a consumer warp and a producer warp, each
/// stage behind a `full` and an `empty` mbarrier of 32 arrivals, over a runtime loop. The
/// consumer (warp 0) comes first in program order: each iteration it waits until the stage is
/// full, adds it into a running sum and arrives on `empty`; the producer (warp 1) waits until
/// the stage is empty, stores the iteration's words and arrives on `full`. Phase parities flip
/// every second iteration. The spins give up after [`SPIN_LIMIT`] polls, and each thread stores
/// how many did in `gave_up`.
pub fn mbarrier_ring() -> Case {
    let kernel = Kernel::builder("mbarrier_ring").block_dim([T as u32, 1, 1]).build(|| {
        let src = param_tensor::<u32>("src", Layout::col_major(RING_ITERS * 32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let stages = make_smem_tensor::<u32>(Layout::col_major(64));
        if_(t.eq(0), || {
            for s in 0..2usize {
                mbarrier_init(&full.offset(s), 32);
                mbarrier_init(&empty.offset(s), 32);
            }
        });
        sync_threads();
        let sum = Var::new(Uint32::from(0u32));
        let misses = Var::new(Uint32::from(0u32));
        let spin = |bar: Ptr<u64>, parity: Int32| {
            let missed = bounded_spin(&bar, parity);
            misses.update(|m| m + select(missed, 1u32, 0u32));
        };
        for_range(0, RING_ITERS as i32, 1, |i| {
            let stage = i % 2;
            let parity = (i / 2) % 2;
            if_else(
                warp_idx().eq(0),
                || {
                    spin(full.offset(stage), parity);
                    sum.update(|s| s + stages.load(stage * 32 + (lane + i) % 32));
                    mbarrier_arrive(&empty.offset(stage));
                },
                || {
                    // A fresh mbarrier counts the phase before its first as complete.
                    spin(empty.offset(stage), 1 - parity);
                    stages.store(stage * 32 + lane, src.load(i * 32 + lane) + i.to::<u32>());
                    mbarrier_arrive(&full.offset(stage));
                },
            );
        });
        if_(warp_idx().eq(0), || out.store(lane, sum.get()));
        gave_up.store(t, misses.get());
    });
    fn check(args: &[KernelArg]) {
        let (src, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            let want = (0..RING_ITERS)
                .fold(0u32, |acc, i| acc.wrapping_add(src[i * 32 + (l + i) % 32].wrapping_add(i as u32)));
            assert_eq!(got, want, "lane {l}");
        }
    }
    Case { name: "mbarrier_ring", kernel, cfg: launch(1), small_only: false, check }
}

/// [`bounded_spin`] polling with `mbarrier_try_wait` (SM90 and newer), whose thread can be held
/// back for a time-limit before the poll gives up and returns false.
fn bounded_try_spin(bar: &Ptr<u64>, parity: Int32) -> Boolean {
    let (_, done) = while_carry(
        (Int32::from(0), Boolean::from(false)),
        |(polls, done)| !done & polls.lt(SPIN_LIMIT),
        |(polls, _)| (polls + 1, mbarrier_try_wait(bar, parity)),
    );
    !done
}

/// [`consumer_first_mbarrier`] with `mbarrier_try_wait`: warp 0 waits on the mbarrier before
/// warp 1 has stored its words and arrived, and then reads them.
pub fn consumer_first_try_wait() -> Case {
    let kernel = Kernel::builder("consumer_first_try_wait").block_dim([T as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(32));
        let out = param_tensor::<u32>("out", Layout::col_major(32));
        let gave_up = param_tensor::<u32>("gave_up", Layout::col_major(T));
        let t = thread_idx().x;
        let lane = t % 32;
        let bar = alloc_smem::<u64>(1, 8);
        let data = make_smem_tensor::<u32>(Layout::col_major(32));
        if_(t.eq(0), || mbarrier_init(&bar, 32));
        sync_threads();
        let missed = if_else_yield(
            warp_idx().eq(0),
            || {
                let missed = bounded_try_spin(&bar, 0.into_value());
                out.store(lane, data.load((lane + 3) % 32));
                select(missed, 1u32, 0u32)
            },
            || {
                data.store(lane, xs.load(lane) + 7u32);
                mbarrier_arrive(&bar);
                0u32.into_value()
            },
        );
        gave_up.store(t, missed);
    });
    fn check(args: &[KernelArg]) {
        let (xs, out) = (args[0].to_vec::<u32>(), args[1].to_vec::<u32>());
        check_no_spin_gave_up(&args[2].to_vec::<u32>());
        for (l, &got) in out.iter().enumerate() {
            assert_eq!(got, xs[(l + 3) % 32].wrapping_add(7), "lane {l}");
        }
    }
    Case { name: "consumer_first_try_wait", kernel, cfg: launch(1), small_only: false, check }
}

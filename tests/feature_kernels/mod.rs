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
mod barriers;
mod basic;
mod carry_asm;
mod half_and_cvt;
mod int_bit_ops;
mod layouts_and_vars;
mod memory_model;
mod mma_cases;
mod tmem_ldst;
mod types;
mod vector;
mod warp_collectives;
mod wgmma;
mod wgmma_variants;

pub use async_sched::*;
pub use atomics::*;
pub use autovec::*;
pub use barriers::*;
pub use basic::*;
pub use bulk::*;
pub use carry_asm::*;
pub use cluster_copies::*;
pub use composed::{composed_layout_inner, pointer_swizzle};
pub use cpasync_order::*;
use cute::ir::Param;
use cute::prelude::*;
pub use cvt_ldmatrix::*;
pub use dsmem::*;
pub use extras::*;
pub use gemm_helpers::*;
pub use half_and_cvt::*;
pub use int_bit_ops::*;
pub use layouts_and_vars::*;
pub use masked::*;
pub use memory_model::*;
pub use mma_cases::*;
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
pub use warp_collectives::*;
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

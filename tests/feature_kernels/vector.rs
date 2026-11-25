// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/basic_types.py (the vector
//   kernels and their expected values) and
//   examples/python/CuTeDSL/experimental/primitives/reduction/thread_vector_reduce.py
//   (the reductions),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and examples/python/CuTeDSL/notebooks/2_primitives/02_vector_concepts.ipynb
//   (its vector operations), a notebook with no header of its own, under
//   CUTLASS's LICENSE.txt,
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! `Vector` (CuTe DSL's `cutlass.Vector`): the 4.8.1 examples' vector
//! kernels (`basic_types.py`, `02_vector_concepts`, `thread_vector_reduce`),
//! bitcasts, `TensorSsa` interop and 32-bit tensor shuffles, held to plain
//! Rust.

use cute::prelude::*;

use super::Case;
use super::tensor_ssa::f64_to_f16;

/// Threads of these kernels: one warp.
const THREADS: usize = 32;

fn cfg() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1])
}

/// Outputs per thread of `vector_basic_types`.
const BASIC: usize = 16;

/// `basic_types.py`'s vector kernels, per thread: a 2-D slice of a
/// broadcast row (`[2, 3, 2, 3]`), `base * scale + 1` with a runtime scale,
/// and an outer product (`[3, 4, 5, 6, 8, 10]`).
pub fn vector_basic_types() -> Case {
    let kernel = Kernel::builder("vector_basic_types").block_dim([THREADS as u32, 1, 1]).build(|| {
        let scales = param_tensor::<f32>("scales", Layout::col_major(THREADS));
        let out = param_tensor::<f32>("out", Layout::col_major((BASIC, THREADS)));
        let t = thread_idx().x;
        let dst = out.slice((.., t));
        let row = Vector::<f32>::from_elements([1.0, 2.0, 3.0, 4.0]);
        let matrix = vector::broadcast_to(&row, (2, 4));
        let sub = matrix.slice((.., 1..3));
        for (k, c) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            dst.store(k, sub.get(c));
        }
        let base = Vector::<f32>::from_elements([1.0, 2.0, 3.0, 4.0]);
        let scaled = base * scales.load(t) + Float32::constant(1.0);
        for k in 0..4 {
            dst.store(4 + k, scaled.get(k));
        }
        let a = Vector::<f32>::from_elements([1.0, 2.0]);
        let b = Vector::<f32>::from_elements([3.0, 4.0, 5.0]);
        let m = vector::outerproduct(&a, &b);
        for k in 0..6 {
            dst.store(8 + k, m.get((k / 3, k % 3)));
        }
        // The accumulator form, fused: a[i] * b[j] + m[i, j].
        let acc = vector::outerproduct_acc(&a, &b, &m);
        dst.store(14, acc.get((1, 2)));
        dst.store(15, acc.reduce(ReductionOp::Add));
    });
    fn check(args: &[KernelArg]) {
        let (scales, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for t in 0..THREADS {
            let o = &out[t * BASIC..(t + 1) * BASIC];
            assert_eq!(o[..4], [2.0, 3.0, 2.0, 3.0], "vector_2d_slice, thread {t}");
            for k in 0..4 {
                assert_eq!(o[4 + k], (k + 1) as f32 * scales[t] + 1.0, "vector_scalar_broadcast {k}, thread {t}");
            }
            assert_eq!(o[8..14], [3.0, 4.0, 5.0, 6.0, 8.0, 10.0], "vector_outerproduct, thread {t}");
            assert_eq!(o[14], 20.0, "outerproduct_acc, thread {t}");
            assert_eq!(o[15], 72.0, "outerproduct_acc reduce, thread {t}");
        }
    }
    Case { name: "vector_basic_types", kernel, cfg: cfg(), small_only: false, check }
}

/// Elements per thread of the vector concept kernels.
const V: usize = 8;
/// Outputs per thread of `vector_concepts`.
const CONCEPTS: usize = 4 * V + 6;

/// `02_vector_concepts`' operations on eight Float32 per thread: ReLU
/// (`where(u > zeros, u, zeros)`), square, abs and `vec * A + B`, the four
/// reductions, a runtime-bounded mask, and a vector carried through a loop.
pub fn vector_concepts() -> Case {
    let kernel = Kernel::builder("vector_concepts").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((V, THREADS)));
        let n = param_tensor::<i32>("n", Layout::col_major(THREADS));
        let out = param_tensor::<f32>("out", Layout::col_major((CONCEPTS, THREADS)));
        let t = thread_idx().x;
        let src = x.slice((.., t));
        let dst = out.slice((.., t));
        let u = Vector::<f32>::from_elements((0..V).map(|i| src.load(i)));
        let zeros = vector::full::<f32>(V, 0.0);
        let relu = vector::where_(&u.gt(&zeros), &u, &zeros);
        let square = &u * &u;
        let abs = u.abs();
        let affine = u.clone() * 2.0 + 0.5;
        for (j, r) in [relu, square, abs, affine].iter().enumerate() {
            for k in 0..V {
                dst.store(j * V + k, r.get(k));
            }
        }
        for (j, op) in [ReductionOp::Add, ReductionOp::Mul, ReductionOp::Max, ReductionOp::Min].into_iter().enumerate()
        {
            dst.store(4 * V + j, u.reduce(op));
        }
        // Lanes below a runtime bound, counted.
        let mask = vector::create_mask(V, n.load(t) & 15);
        let ones = vector::where_(&mask, vector::full::<f32>(V, 1.0), 0.0f32);
        let picked = select(vector::any_(&mask), ones.reduce(ReductionOp::Add), Float32::constant(-1.0));
        dst.store(4 * V + 4, picked);
        // A vector carried through a runtime loop: u + u + ... (k times).
        let acc = for_range_carry(0, n.load(t) & 3, 1, u.clone(), |_, acc| acc + u.clone());
        dst.store(4 * V + 5, acc.get(Int32::constant(0) + (n.load(t) & 7)));
    });
    fn check(args: &[KernelArg]) {
        let (x, n, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<i32>(), args[2].to_vec::<f32>());
        for t in 0..THREADS {
            let u = &x[t * V..(t + 1) * V];
            let o = &out[t * CONCEPTS..(t + 1) * CONCEPTS];
            for k in 0..V {
                let relu = if u[k] > 0.0 { u[k] } else { 0.0 };
                assert_eq!(o[k].to_bits(), relu.to_bits(), "relu {k}, thread {t}");
                assert_eq!(o[V + k], u[k] * u[k], "square {k}, thread {t}");
                assert_eq!(o[2 * V + k], u[k].abs(), "abs {k}, thread {t}");
                assert_eq!(o[3 * V + k], u[k] * 2.0 + 0.5, "vec * A + B {k}, thread {t}");
            }
            let fold = |f: &dyn Fn(f32, f32) -> f32| u[1..].iter().fold(u[0], |a, &b| f(a, b));
            assert_eq!(o[4 * V], fold(&|a, b| a + b), "reduce add, thread {t}");
            assert_eq!(o[4 * V + 1], fold(&|a, b| a * b), "reduce mul, thread {t}");
            assert_eq!(o[4 * V + 2], fold(&f32::max), "reduce max, thread {t}");
            assert_eq!(o[4 * V + 3], fold(&f32::min), "reduce min, thread {t}");
            let bound = (n[t] & 15).min(V as i32);
            let want = if bound > 0 { bound as f32 } else { -1.0 };
            assert_eq!(o[4 * V + 4], want, "create_mask, thread {t}");
            let k = (n[t] & 7) as usize;
            let times = 1 + (n[t] & 3);
            let carried = (1..times).fold(u[k], |a, _| a + u[k]);
            assert_eq!(o[4 * V + 5], carried, "carried vector, thread {t}");
        }
    }
    Case { name: "vector_concepts", kernel, cfg: cfg(), small_only: false, check }
}

macro_rules! thread_vector_reduce {
    ($name:ident, $t:ty, $small:expr, $to:expr, $from:expr) => {
        /// `thread_vector_reduce.py`: eight elements per thread reduced
        /// with add, max and min.
        pub fn $name() -> Case {
            let kernel = Kernel::builder(stringify!($name)).block_dim([THREADS as u32, 1, 1]).build(|| {
                let x = param_tensor::<$t>("x", Layout::col_major((V, THREADS)));
                let out = param_tensor::<$t>("out", Layout::col_major((3, THREADS)));
                let t = thread_idx().x;
                let src = x.slice((.., t));
                let values = Vector::<$t>::from_elements((0..V).map(|i| src.load(i)));
                for (j, op) in [ReductionOp::Add, ReductionOp::Max, ReductionOp::Min].into_iter().enumerate() {
                    out.store((j, t), values.reduce(op));
                }
            });
            fn check(args: &[KernelArg]) {
                let (x, out) = (args[0].to_vec::<$t>(), args[1].to_vec::<$t>());
                let to: fn($t) -> f64 = $to;
                let from: fn(f64) -> $t = $from;
                for t in 0..THREADS {
                    let u: Vec<f64> = x[t * V..(t + 1) * V].iter().map(|&v| to(v)).collect();
                    let add = u[1..].iter().fold(u[0], |a, &b| to(from(a + b)));
                    let max = u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let min = u.iter().copied().fold(f64::INFINITY, f64::min);
                    for (j, w) in [add, max, min].into_iter().enumerate() {
                        assert_eq!(to(out[t * 3 + j]), w, "{} op {j}, thread {t}", stringify!($name));
                    }
                }
            }
            Case { name: stringify!($name), kernel, cfg: cfg(), small_only: $small, check }
        }
    };
}

thread_vector_reduce!(vector_reduce_f16, F16, true, |v| v.to_f32() as f64, f64_to_f16);
thread_vector_reduce!(vector_reduce_bf16, BF16, true, |v| v.to_f32() as f64, |v| BF16::from_f32(v as f32));
thread_vector_reduce!(vector_reduce_f32, f32, false, |v| v as f64, |v| v as f32);
thread_vector_reduce!(vector_reduce_i32, i32, false, |v| v as f64, |v| v as i64 as i32);

/// Bitcasts with element 0 in the low bits: 2 x Float16 to a Uint32,
/// 4 x Uint8 to a Uint32, a (2, 4) Float16 vector to (2, 2) Uint32 and
/// back, and a conversion to Float16.
pub fn vector_bitcasts() -> Case {
    let kernel = Kernel::builder("vector_bitcasts").block_dim([THREADS as u32, 1, 1]).build(|| {
        let h = param_tensor::<F16>("h", Layout::col_major((8, THREADS)));
        let b = param_tensor::<u8>("b", Layout::col_major((4, THREADS)));
        let f = param_tensor::<f32>("f", Layout::col_major((2, THREADS)));
        let words = param_tensor::<u32>("words", Layout::col_major((4, THREADS)));
        let halves = param_tensor::<F16>("halves", Layout::col_major((10, THREADS)));
        let t = thread_idx().x;
        let hv = Vector::<F16>::from_elements((0..2).map(|i| h.load((i, t))));
        words.store((0, t), hv.bitcast::<u32>().get(0));
        let bv = Vector::<u8>::from_elements((0..4).map(|i| b.load((i, t))));
        words.store((1, t), bv.bitcast::<u32>().get(0));
        let m = Vector::<F16>::new((2, 4), (0..8).map(|i| h.load((i, t))).collect());
        let w = m.bitcast::<u32>();
        assert_eq!(w.shape(), &[2, 2]);
        words.store((2, t), w.get((0, 1)));
        words.store((3, t), w.get((1, 0)));
        let back = w.bitcast::<F16>();
        for k in 0..8 {
            halves.store((k, t), back.get((k / 4, k % 4)));
        }
        let fv = Vector::<f32>::from_elements((0..2).map(|i| f.load((i, t)))).to::<F16>();
        halves.store((8, t), fv.get(0));
        halves.store((9, t), fv.get(1));
    });
    fn check(args: &[KernelArg]) {
        let (h, b, f) = (args[0].to_vec::<F16>(), args[1].to_vec::<u8>(), args[2].to_vec::<f32>());
        let (words, halves) = (args[3].to_vec::<u32>(), args[4].to_vec::<F16>());
        let pair = |lo: F16, hi: F16| lo.0 as u32 | (hi.0 as u32) << 16;
        for t in 0..THREADS {
            let hs = &h[t * 8..t * 8 + 8];
            assert_eq!(words[t * 4], pair(hs[0], hs[1]), "2 x f16 to u32, thread {t}");
            let bs = &b[t * 4..t * 4 + 4];
            assert_eq!(words[t * 4 + 1], u32::from_le_bytes([bs[0], bs[1], bs[2], bs[3]]), "4 x u8, thread {t}");
            assert_eq!(words[t * 4 + 2], pair(hs[2], hs[3]), "(2, 4) f16 bitcast (0, 1), thread {t}");
            assert_eq!(words[t * 4 + 3], pair(hs[4], hs[5]), "(2, 4) f16 bitcast (1, 0), thread {t}");
            for k in 0..8 {
                assert_eq!(halves[t * 10 + k].0, hs[k].0, "bitcast back {k}, thread {t}");
            }
            for k in 0..2 {
                let want = f64_to_f16(f[t * 2 + k] as f64);
                assert_eq!(halves[t * 10 + 8 + k].0, want.0, "to f16 {k}, thread {t}");
            }
        }
    }
    Case { name: "vector_bitcasts", kernel, cfg: cfg(), small_only: false, check }
}

/// `TensorSsa::from_vector` and `to_vector`: a (2, 3) row-major vector to a
/// (2, 3) tensor (`v[i, j]` at `i + 2j`) and back, and a ((2, 2), 3)
/// tensor flattened to a (2, 2, 3) vector; `from_flat_vector` and
/// `to_elements` keep the order.
pub fn vector_tensor_interop() -> Case {
    let kernel = Kernel::builder("vector_tensor_interop").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major((12, THREADS)));
        let cols = param_tensor::<i32>("cols", Layout::col_major((6, THREADS)));
        let rows = param_tensor::<i32>("rows", Layout::col_major((24, THREADS)));
        let t = thread_idx().x;
        let src = x.slice((.., t));
        let v = Vector::<i32>::new((2, 3), (0..6).map(|i| src.load(i)).collect());
        let ts = TensorSsa::from_vector(&v, None);
        assert_eq!(ts.shape(), &IntTuple::from((2, 3)));
        cols.slice((.., t)).store_ssa(&ts);
        let back = ts.to_vector(false);
        for k in 0..6 {
            rows.store((k, t), back.get((k / 3, k % 3)));
        }
        let nested = make_tensor(&src.ptr(), Layout::col_major(((2, 2), 3))).load_ssa();
        let flat = nested.to_vector(true);
        assert_eq!(flat.shape(), &[2, 2, 3]);
        for (k, e) in flat.values().iter().enumerate() {
            rows.store((6 + k, t), *e);
        }
        let same = TensorSsa::from_flat_vector(&flat.flatten(), ((2, 2), 3)).to_elements();
        for (k, &e) in same.iter().take(6).enumerate() {
            rows.store((18 + k, t), e);
        }
    });
    fn check(args: &[KernelArg]) {
        let (x, cols, rows) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>(), args[2].to_vec::<i32>());
        for t in 0..THREADS {
            let s = &x[t * 12..t * 12 + 12];
            for i in 0..2 {
                for j in 0..3 {
                    assert_eq!(cols[t * 6 + i + 2 * j], s[i * 3 + j], "from_vector ({i}, {j}), thread {t}");
                    assert_eq!(rows[t * 24 + i * 3 + j], s[i * 3 + j], "to_vector ({i}, {j}), thread {t}");
                }
            }
            // flat[a, b, c] is tensor element ((a, b), c), at a + 2b + 4c.
            for a in 0..2 {
                for b in 0..2 {
                    for c in 0..3 {
                        let r = (a * 2 + b) * 3 + c;
                        assert_eq!(rows[t * 24 + 6 + r], s[a + 2 * b + 4 * c], "force_flatten {r}, thread {t}");
                    }
                }
            }
            for k in 0..6 {
                // Unpermuted: the flattened vector's order.
                assert_eq!(rows[t * 24 + 18 + k], rows[t * 24 + 6 + k], "from_flat_vector {k}, thread {t}");
            }
        }
    }
    Case { name: "vector_tensor_interop", kernel, cfg: cfg(), small_only: false, check }
}

/// A 2 x Float16 `TensorSsa` and a 4 x Int8 `Vector` shuffled across lanes
/// as one 32-bit word each (butterfly and indexed).
pub fn vector_shuffles() -> Case {
    let kernel = Kernel::builder("vector_shuffles").block_dim([THREADS as u32, 1, 1]).build(|| {
        let h = param_tensor::<F16>("h", Layout::col_major((2, THREADS)));
        let b = param_tensor::<i8>("b", Layout::col_major((4, THREADS)));
        let ho = param_tensor::<F16>("ho", Layout::col_major((2, THREADS)));
        let bo = param_tensor::<i8>("bo", Layout::col_major((4, THREADS)));
        let t = thread_idx().x;
        let pair = h.slice((.., t)).load_ssa();
        ho.slice((.., t)).store_ssa(&shuffle_sync_bfly(pair, 5));
        let bytes = Vector::<i8>::from_elements((0..4).map(|i| b.load((i, t))));
        let moved = shuffle_sync(bytes, (t + 3) % THREADS as i32);
        for k in 0..4 {
            bo.store((k, t), moved.get(k));
        }
    });
    fn check(args: &[KernelArg]) {
        let (h, b) = (args[0].to_vec::<F16>(), args[1].to_vec::<i8>());
        let (ho, bo) = (args[2].to_vec::<F16>(), args[3].to_vec::<i8>());
        for t in 0..THREADS {
            let from = t ^ 5;
            for k in 0..2 {
                assert_eq!(ho[t * 2 + k].0, h[from * 2 + k].0, "f16 pair {k}, thread {t}");
            }
            let from = (t + 3) % THREADS;
            for k in 0..4 {
                assert_eq!(bo[t * 4 + k], b[from * 4 + k], "i8 x 4 {k}, thread {t}");
            }
        }
    }
    Case { name: "vector_shuffles", kernel, cfg: cfg(), small_only: false, check }
}

/// The vector kernels.
pub fn vector_cases() -> Vec<Case> {
    vec![
        vector_basic_types(),
        vector_concepts(),
        vector_reduce_f16(),
        vector_reduce_bf16(),
        vector_reduce_f32(),
        vector_reduce_i32(),
        vector_bitcasts(),
        vector_tensor_interop(),
        vector_shuffles(),
    ]
}

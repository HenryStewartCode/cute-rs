// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/cta_norm/cta_norm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! LayerNorm and RMSNorm with one CTA per row, ported from the CuTe DSL
//! example `cta_norm.py`, on Float16, BFloat16 and Float32 data.
//!
//! Each thread moves 8 elements at a time with 128-bit copies
//! (`make_tiled_copy_tv` of a `Universal { bits: 128 }` atom over a linear
//! thread layout), chunk by chunk along the row, each chunk under a
//! Boolean predicate `(i * threads + tidx) * 8 < N`. The statistics are
//! Float32: a sum per thread, a butterfly `warp_reduce`
//! (`shuffle_sync_bfly`), and a `cta_reduce` through a small shared-memory
//! array; `factor = rsqrt(var / N + eps)` (precise). The epilogue is
//! TensorSSA arithmetic, `(x - mean) * factor * weight + bias` for
//! LayerNorm and `x * factor * weight` for RMSNorm, promoted to Float32 and
//! converted back with `.to()`.
//!
//! Differences from the example: `N` and the thread count are static (as
//! the example's are, at compile time), `M` too (runtime shapes arrive with
//! m5); `eps` is a constant rather than a kernel argument; and each chunk
//! is loaded with `load_ssa` inside an `if_else_yield` that yields zeros
//! when its predicate is off, where the example zero-fills `x`'s fragment
//! and `autovec_copy`s under `if pred[i]` (a register written inside a
//! cute-rs branch must be yielded out of it), so the weight and bias
//! fragments are zero there too (the example leaves them unwritten).
//!
//! Run with `cargo run --example cta_norm`.

use std::ops::{Add, Mul, Sub};

use cute::prelude::*;

/// Rows (one CTA each).
pub const M: usize = 8;
/// Elements per thread per chunk: one 128-bit copy of Float16.
const ELEMS_PER_THREAD: usize = 8;
pub const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Layer,
    Rms,
}

/// One configuration: the row length (a multiple of 8) and the threads per
/// CTA (a multiple of 32).
#[derive(Clone, Copy, Debug)]
pub struct NormConfig {
    pub norm: Norm,
    pub n: usize,
    pub threads: usize,
}

impl NormConfig {
    pub fn launch(&self) -> LaunchConfig {
        LaunchConfig::new([M as u32, 1, 1], [self.threads as u32, 1, 1])
    }

    fn warps(&self) -> usize {
        self.threads.div_ceil(32)
    }
}

/// `val` summed over the warp: a butterfly of `shuffle_sync_bfly`s.
fn warp_reduce(mut val: Float32) -> Float32 {
    for i in 0..5 {
        val = val + shuffle_sync_bfly(val, 1 << i);
    }
    val
}

/// `val` summed over the CTA: each warp's sum through `acc`, summed by warp
/// 0, and broadcast through `acc[warps]`.
fn cta_reduce(val: Float32, acc: &Tensor<f32>, tidx: Int32, warps: usize) -> Float32 {
    let warp_id = tidx >> 5;
    let lane_id = tidx & 31;
    if_(lane_id.eq(0), || acc.store(warp_id, val));
    sync_threads();
    if_(warp_id.eq(0), || {
        let v = if_else_yield(lane_id.lt(warps as i32), || acc.load(lane_id), || Float32::constant(0.0));
        acc.store(warps, warp_reduce(v));
    });
    sync_threads();
    acc.load(warps)
}

/// The mean (0 for RMSNorm) and `rsqrt(var / N + eps)` of the row, from
/// each thread's chunks as Float32: LayerNorm sums every element (the
/// masked-off chunks are zeros) for the mean, then the squared deviations
/// of the chunks its predicates allow; RMSNorm the squares of those.
fn statistics(
    norm: Norm,
    xs: &[Vec<Float32>],
    pred: &[Boolean],
    n: usize,
    acc: &Tensor<f32>,
    tidx: Int32,
    warps: usize,
) -> (Float32, Float32) {
    let mean = match norm {
        Norm::Layer => {
            let mut val = Float32::constant(0.0);
            for &x in xs.iter().flatten() {
                val = val + x;
            }
            cta_reduce(warp_reduce(val), acc, tidx, warps) / n as f32
        }
        Norm::Rms => Float32::constant(0.0),
    };
    let mut val = Float32::constant(0.0);
    for (chunk, &p) in xs.iter().zip(pred) {
        let mut sum = val;
        for &x in chunk {
            let d = if norm == Norm::Layer { x - mean } else { x };
            sum = sum + d * d;
        }
        val = select(p, sum, val);
    }
    let var = cta_reduce(warp_reduce(val), acc, tidx, warps);
    (mean, math::rsqrt(var / n as f32 + EPS, false))
}

/// The kernel of `cfg` on `T` elements: parameters `x` (`M x N` row-major),
/// `weight` and, for LayerNorm, `bias` (`N`), and the output `y`.
pub fn cta_norm<T: Numeric>(cfg: NormConfig) -> Kernel
where
    Value<T>: Sub<Value<f32>, Output = Value<f32>> + Mul<Value<f32>, Output = Value<f32>>,
    Value<f32>: Mul<Value<T>, Output = Value<f32>> + Add<Value<T>, Output = Value<f32>>,
{
    let NormConfig { norm, n, threads } = cfg;
    assert!(n.is_multiple_of(ELEMS_PER_THREAD), "cta_norm: N = {n} must be a multiple of 8");
    let warps = cfg.warps();
    let name =
        format!("cta_{}norm_{}_n{n}_t{threads}", if norm == Norm::Layer { "layer" } else { "rms" }, T::DTYPE.mlir());
    Kernel::builder(&name).block_dim([threads as u32, 1, 1]).build(|| {
        let mx = param_tensor::<T>("x", Layout::row_major((M, n)));
        let mw = param_tensor::<T>("weight", Layout::col_major(n));
        let mb = (norm == Norm::Layer).then(|| param_tensor::<T>("bias", Layout::col_major(n)));
        let my = param_tensor::<T>("y", Layout::row_major((M, n)));
        let tidx = thread_idx().x;
        let bidx = block_idx().x;

        let atom = make_copy_atom::<T>(CopyOp::Universal { bits: 128 });
        let tiled_copy = make_tiled_copy_tv(&atom, &Layout::col_major(threads), &Layout::col_major(ELEMS_PER_THREAD));
        let thr_copy = tiled_copy.get_slice(tidx);

        let gy = local_tile(&my, (1, n), (bidx, 0)).slice((0, ..));
        let gx = local_tile(&mx, (1, n), (bidx, 0)).slice((0, ..));
        let tygy = thr_copy.partition_s(&gy);
        let chunks = size(&tygy, &[1]).as_static().expect("a static row");
        let pred: Vec<Boolean> =
            (0..chunks).map(|i| ((tidx + (i * threads) as i32) * ELEMS_PER_THREAD as i32).lt(n as i32)).collect();
        let txgx = thr_copy.partition_s(&gx);
        let twgw = thr_copy.partition_s(&mw);
        let tbgb = mb.as_ref().map(|b| thr_copy.partition_s(b));

        // The fragments, zero where a chunk's predicate is off (CuTe DSL fills
        // x's with 0 and copies each chunk under `if pred[i]`; a register
        // written inside a branch is yielded out of it here).
        let txrx = make_fragment_like::<T, _>(&txgx);
        let twrw = make_fragment_like::<T, _>(&twgw);
        let tbrb = tbgb.as_ref().map(|t| make_fragment_like::<T, _>(t));
        for (i, &p) in pred.iter().enumerate() {
            let load = |g: &Tensor<T>, r: &Tensor<T>| {
                let chunk = g.slice((.., i));
                let zeros = zeros_like::<T>(&chunk);
                r.slice((.., i)).store_ssa(&if_else_yield(p, || chunk.load_ssa(), || zeros));
            };
            load(&txgx, &txrx);
            load(&twgw, &twrw);
            if let (Some(g), Some(r)) = (&tbgb, &tbrb) {
                load(g, r);
            }
        }

        let acc = make_smem_tensor::<f32>(Layout::col_major(warps + 1));
        let per_chunk = size(&txrx, &[0]).as_static().expect("static");
        let xs: Vec<Vec<Float32>> =
            (0..chunks).map(|i| (0..per_chunk).map(|k| txrx.slice((.., i)).load(k).to::<f32>()).collect()).collect();
        let (mean, factor) = statistics(norm, &xs, &pred, n, &acc, tidx, warps);
        let normed: TensorSsa<f32> = match norm {
            Norm::Layer => {
                let bias = tbrb.as_ref().expect("LayerNorm has a bias").load_ssa();
                (txrx.load_ssa() - mean) * factor * twrw.load_ssa() + bias
            }
            Norm::Rms => txrx.load_ssa() * factor * twrw.load_ssa(),
        };
        let tyry = make_fragment_like::<T, _>(&txrx);
        tyry.store_ssa(&normed.to::<T>());
        for (i, &p) in pred.iter().enumerate() {
            if_(p, || autovec_copy(&tyry.slice((.., i)), &tygy.slice((.., i))));
        }
    })
}

/// Deterministic data in [-2, 2).
pub fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// The reference in f64.
pub fn reference(cfg: NormConfig, x: &[f64], w: &[f64], b: Option<&[f64]>) -> Vec<f64> {
    let n = cfg.n;
    let mut y = vec![0.0; x.len()];
    for r in 0..M {
        let row = &x[r * n..(r + 1) * n];
        let (mean, scale) = match cfg.norm {
            Norm::Layer => {
                let mean = row.iter().sum::<f64>() / n as f64;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
                (mean, 1.0 / (var + EPS as f64).sqrt())
            }
            Norm::Rms => (0.0, 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / n as f64 + EPS as f64).sqrt()),
        };
        for k in 0..n {
            y[r * n + k] = (row[k] - mean) * scale * w[k] + b.map_or(0.0, |b| b[k]);
        }
    }
    y
}

/// The inputs of `kernel` (built for `cfg`), as `to` makes them.
pub fn inputs<T: Numeric>(cfg: NormConfig, to: impl Fn(f32) -> T) -> Vec<KernelArg> {
    let n = cfg.n;
    let conv = |v: Vec<f32>| v.into_iter().map(&to).collect::<Vec<T>>();
    let mut args = vec![KernelArg::buffer(&conv(data(M * n, 1))), KernelArg::buffer(&conv(data(n, 2)))];
    if cfg.norm == Norm::Layer {
        args.push(KernelArg::buffer(&conv(data(n, 3))));
    }
    args.push(KernelArg::buffer(&conv(vec![0.0; M * n])));
    args
}

/// Simulate `cfg` on `T` data and check it against the f64 reference within
/// `tol` (absolute plus relative): the output's rounding (2^-11 for
/// Float16, 2^-8 for BFloat16) dominates; Float32 is held to 1e-4.
pub fn run<T: Numeric>(cfg: NormConfig, to: impl Fn(f32) -> T, from: impl Fn(T) -> f32, tol: f64) -> SimReport
where
    Value<T>: Sub<Value<f32>, Output = Value<f32>> + Mul<Value<f32>, Output = Value<f32>>,
    Value<f32>: Mul<Value<T>, Output = Value<f32>> + Add<Value<T>, Output = Value<f32>>,
{
    let kernel = cta_norm::<T>(cfg);
    let mut args = inputs(cfg, &to);
    let report = kernel.simulate(cfg.launch(), &mut args).unwrap_or_else(|e| panic!("{}: {e}", kernel.name()));
    let f64s = |a: &KernelArg| a.to_vec::<T>().into_iter().map(|v| from(v) as f64).collect::<Vec<f64>>();
    let (x, w) = (f64s(&args[0]), f64s(&args[1]));
    let b = (cfg.norm == Norm::Layer).then(|| f64s(&args[2]));
    let want = reference(cfg, &x, &w, b.as_deref());
    let got = f64s(args.last().unwrap());
    for (i, (&g, &r)) in got.iter().zip(&want).enumerate() {
        assert!((g - r).abs() <= tol * (1.0 + r.abs()), "{}: element {i}: {g} against {r}", kernel.name());
    }
    report
}

/// The configurations the tests run: a row that is a whole number of
/// chunks and one that is not (the last chunk's predicates matter).
pub const CONFIGS: [NormConfig; 4] = [
    NormConfig { norm: Norm::Layer, n: 1024, threads: 128 },
    NormConfig { norm: Norm::Layer, n: 1528, threads: 64 },
    NormConfig { norm: Norm::Rms, n: 1024, threads: 128 },
    NormConfig { norm: Norm::Rms, n: 1528, threads: 64 },
];

pub fn run_f16(cfg: NormConfig) -> SimReport {
    run::<F16>(cfg, F16::from_f32, F16::to_f32, 4e-3)
}

pub fn run_bf16(cfg: NormConfig) -> SimReport {
    run::<BF16>(cfg, BF16::from_f32, BF16::to_f32, 2e-2)
}

pub fn run_f32(cfg: NormConfig) -> SimReport {
    run::<f32>(cfg, |x| x, |x| x, 1e-4)
}

#[allow(dead_code)]
fn main() {
    for cfg in CONFIGS {
        run_f16(cfg);
        run_bf16(cfg);
        run_f32(cfg);
        println!("{:?}: Float16, BFloat16 and Float32 within tolerance", cfg);
    }
}

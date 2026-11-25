// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// namely the host function's two launches from one compiled function (the split-KV kernel,
// then `reduction_kernel`) and `reduction_kernel` itself. Those portions are under the
// BSD-3-Clause licence, and the rest of the file is cute-rs's own, under MIT OR Apache-2.0.
// THIRD-PARTY-NOTICES.md has the licence text and lists what each part ports.

//! Split-KV attention decode as one module of two kernels, the way CuTe DSL's MLA decode
//! example launches its split-KV kernel and then its `reduction_kernel` from one `@cute.jit`
//! host function, compiled together.
//!
//! * `split_kv_decode` (cute-rs's own, a plain SIMT stand-in for the example's tcgen05 MLA
//!   kernel): block `(split, head)` attends one query per head to its slice of the keys and
//!   values with an online softmax in base 2, and writes the slice's normalised output
//!   `acc_o[head, split, :]` and log-sum-exp `acc_lse[head, split]`;
//! * `reduction_kernel` (ported): block `head`'s first warp finds the global log-sum-exp over
//!   the splits (a warp max and a warp sum, one split per lane) and each split's scale
//!   `exp2(lse_split - lse)` in shared memory; then every thread sums its output elements over
//!   the splits with those scales.
//!
//! Differences from the example: one query per head and no latent/rope split, Float32
//! throughout, `MAX_SPLITS` is one warp (so one split per lane), a full-block barrier where the
//! example uses named barrier 4, and precise `exp2`/`log2` instead of `fastmath`.
//!
//! Run with `cargo run --example split_kv_decode` (simulated);
//! tests/gpu_multi_kernel.rs runs the module on a GPU.

use cute::prelude::*;

/// Heads (blocks of the reduction; the second grid dimension of the split kernel).
pub const HEADS: usize = 4;
/// Head dimension: the threads of a split block, one output element each.
pub const D: usize = 64;
/// Keys and values per head.
pub const SEQ: usize = 200;
/// The splits of the sequence (`split_kv`).
pub const SPLITS: usize = 7;
/// The most splits the reduction handles: one per lane of a warp.
pub const MAX_SPLITS: usize = 32;
/// Threads of the reduction: two warps, each element of `D` owned by one thread per pass.
pub const REDUCE_THREADS: usize = 64;
/// `1 / sqrt(D) * log2(e)`: the scores' scale in base 2.
pub const SCALE_LOG2: f32 = 0.125 * std::f32::consts::LOG2_E;

/// Keys per split (the last split may have fewer).
pub const fn chunk() -> usize {
    SEQ.div_ceil(SPLITS)
}

/// The split-KV kernel: grid `(SPLITS, HEADS)`, `D` threads.
pub fn split_kernel() -> Kernel {
    Kernel::builder("split_kv_decode").block_dim([D as u32, 1, 1]).build(|| {
        let q = param_tensor::<f32>("q", Layout::col_major((HEADS * D,)));
        let k = param_tensor::<f32>("k", Layout::col_major((SEQ * HEADS * D,)));
        let v = param_tensor::<f32>("v", Layout::col_major((SEQ * HEADS * D,)));
        let acc_o = param_tensor::<f32>("acc_o", Layout::col_major((HEADS * MAX_SPLITS * D,)));
        let acc_lse = param_tensor::<f32>("acc_lse", Layout::col_major((HEADS * MAX_SPLITS,)));
        let (d, split, h) = (thread_idx().x, block_idx().x, block_idx().y);
        let (hd, dd) = (HEADS as i32, D as i32);
        let start = split * chunk() as i32;
        let end = (start + chunk() as i32).min(SEQ as i32);
        let m = Var::new(Float32::from(f32::NEG_INFINITY));
        let l = Var::new(Float32::from(0.0));
        let acc = Var::new(Float32::from(0.0));
        for_range(start, end, 1, |j| {
            let row = (j * hd + h) * dd;
            let dot = for_range_carry(0, dd, 1, Float32::from(0.0), |e, s| s + q.load(h * dd + e) * k.load(row + e));
            let s = dot * SCALE_LOG2;
            let m_new = fmax(m.get(), s);
            let correction = (m.get() - m_new).exp2();
            let p = (s - m_new).exp2();
            l.set(l.get() * correction + p);
            acc.set(acc.get() * correction + p * v.load(row + d));
            m.set(m_new);
        });
        let slot = h * MAX_SPLITS as i32 + split;
        acc_o.store(slot * dd + d, acc.get() / l.get());
        if_(d.eq(0), || acc_lse.store(slot, m.get() + l.get().log2()));
    })
}

/// `reduction_kernel` of the example: grid `(HEADS)`, `REDUCE_THREADS` threads.
pub fn reduction_kernel() -> Kernel {
    Kernel::builder("reduction_kernel").block_dim([REDUCE_THREADS as u32, 1, 1]).build(|| {
        let out = param_tensor::<f32>("o", Layout::col_major((HEADS * D,)));
        let lse = param_tensor::<f32>("lse", Layout::col_major((HEADS,)));
        let acc_o = param_tensor::<f32>("acc_o", Layout::col_major((HEADS * MAX_SPLITS * D,)));
        let acc_lse = param_tensor::<f32>("acc_lse", Layout::col_major((HEADS * MAX_SPLITS,)));
        let split_kv = param_scalar::<i32>("split_kv");
        let (h, tidx) = (block_idx().x, thread_idx().x);
        let smem_lse_scale = make_smem_tensor::<f32>(Layout::col_major((MAX_SPLITS,)));
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(0), || {
            // The global log-sum-exp, and exp2(local_lse - global_lse) per split.
            let local_lse =
                select(tidx.lt(split_kv), acc_lse.load(h * MAX_SPLITS as i32 + tidx), Float32::from(f32::NEG_INFINITY));
            let lse_max = warp_reduction_max(local_lse, WARP_SIZE);
            let lse_max = select(lse_max.ne(f32::NEG_INFINITY), lse_max, Float32::from(0.0));
            let sum_lse = warp_reduction_sum((local_lse - lse_max).exp2(), WARP_SIZE);
            let global_lse = select(sum_lse.ne(0.0f32), lse_max + sum_lse.log2(), Float32::from(f32::INFINITY));
            if_(tidx.eq(0), || lse.store(h, global_lse));
            if_(tidx.lt(split_kv), || smem_lse_scale.store(tidx, (local_lse - global_lse).exp2()));
        });
        sync_threads();
        let dd = D as i32;
        for j in 0..D.div_ceil(REDUCE_THREADS) {
            let element = tidx + (j * REDUCE_THREADS) as i32;
            let acc = for_range_carry(0, split_kv, 1, Float32::from(0.0), |i, a| {
                a + acc_o.load((h * MAX_SPLITS as i32 + i) * dd + element) * smem_lse_scale.load(i)
            });
            out.store(h * dd + element, acc);
        }
    })
}

/// The module: both kernels, compiled and loaded together.
pub fn module() -> KernelModule {
    KernelModule::new("split_kv_attention", [split_kernel(), reduction_kernel()]).expect("distinct kernels")
}

/// The launches of the host function, in order.
pub fn split_config() -> LaunchConfig {
    LaunchConfig::new([SPLITS as u32, HEADS as u32, 1], [D as u32, 1, 1])
}

pub fn reduction_config() -> LaunchConfig {
    LaunchConfig::new([HEADS as u32, 1, 1], [REDUCE_THREADS as u32, 1, 1])
}

/// Deterministic inputs: `q`, `k` and `v`.
pub fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let f = |i: usize, a: usize, b: usize| ((i * a + b) % 97) as f32 / 48.0 - 1.0;
    let q = (0..HEADS * D).map(|i| f(i, 7, 3)).collect();
    let k = (0..SEQ * HEADS * D).map(|i| f(i, 13, 5)).collect();
    let v = (0..SEQ * HEADS * D).map(|i| f(i, 29, 11)).collect();
    (q, k, v)
}

/// Attention on the host, in f64: the output `o[h, :]` and log-sum-exp (base 2) per head.
pub fn reference(q: &[f32], k: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut o = vec![0f32; HEADS * D];
    let mut lse = vec![0f32; HEADS];
    for h in 0..HEADS {
        let scores: Vec<f64> = (0..SEQ)
            .map(|j| {
                let dot: f64 = (0..D).map(|e| q[h * D + e] as f64 * k[(j * HEADS + h) * D + e] as f64).sum();
                dot * SCALE_LOG2 as f64
            })
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = scores.iter().map(|s| (s - m).exp2()).sum();
        lse[h] = (m + sum.log2()) as f32;
        for d in 0..D {
            let acc: f64 = (0..SEQ).map(|j| (scores[j] - m).exp2() * v[(j * HEADS + h) * D + d] as f64).sum();
            o[h * D + d] = (acc / sum) as f32;
        }
    }
    (o, lse)
}

/// The host function on the simulator: the split kernel, then the reduction, from one
/// simulated module. Returns `o` and `lse`.
pub fn simulate() -> Result<(Vec<f32>, Vec<f32>), SimError> {
    let (q, k, v) = inputs();
    let module = module();
    let mut sim = module.sim_module();
    let mut split_args = vec![
        KernelArg::f32(q),
        KernelArg::f32(k),
        KernelArg::f32(v),
        KernelArg::f32(vec![0.0; HEADS * MAX_SPLITS * D]),
        KernelArg::f32(vec![0.0; HEADS * MAX_SPLITS]),
    ];
    sim.simulate("split_kv_decode", split_config(), &mut split_args)?;
    let mut reduce_args = vec![
        KernelArg::f32(vec![0.0; HEADS * D]),
        KernelArg::f32(vec![0.0; HEADS]),
        split_args[3].clone(),
        split_args[4].clone(),
        KernelArg::ScalarI32(SPLITS as i32),
    ];
    sim.simulate("reduction_kernel", reduction_config(), &mut reduce_args)?;
    Ok((reduce_args[0].as_f32().to_vec(), reduce_args[1].as_f32().to_vec()))
}

/// The largest difference between two vectors.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[allow(dead_code)]
fn main() {
    let (q, k, v) = inputs();
    let (want_o, want_lse) = reference(&q, &k, &v);
    let (o, lse) = simulate().unwrap_or_else(|e| panic!("{e}"));
    println!("split-KV decode: {HEADS} heads, {SEQ} keys in {SPLITS} splits of {}, head dimension {D}", chunk());
    println!(
        "max |o - reference| = {:e}, max |lse - reference| = {:e}",
        max_error(&o, &want_o),
        max_error(&lse, &want_lse)
    );
    assert!(max_error(&o, &want_o) < 1e-4 && max_error(&lse, &want_lse) < 1e-4);
    let mlir = module().to_mlir();
    println!("one module: {} kernels in one gpu.module", mlir.matches("gpu.func @").count());
}

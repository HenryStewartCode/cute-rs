// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/redux_sync.py (kernels 1
//   to 4, `integer_reductions_kernel`, `float_reductions_kernel`,
//   `redux_vs_shuffle_kernel` and `absmax_quantize_kernel`, the host data of
//   their `run_*`, and the tolerances of `verify_absmax_quantize`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Warp reductions in one instruction (`redux.sync`), ported from kernels 1
//! to 4 of CuTe DSL's `experimental/primitives/redux_sync.py`, on one warp:
//!
//! 1. `integer_reductions`: the eight integer `redux.sync` operations (add,
//!    min, max, and, or, xor, umin, umax; the example's `prims.redux_sync`
//!    is `warp_redux_sync`), each also computed by a butterfly of shuffles
//!    for comparison. sm_80.
//! 2. `float_reductions`: Float32 min and max, plain and of absolute values
//!    (`redux.sync.max.abs.f32`). sm_100a only.
//! 3. `redux_vs_shuffle`: the warp's largest absolute value, by
//!    `redux.sync.max.abs.f32` (sm_100a only; left out with `redux: false`)
//!    and by a `shuffle_sync_down` tree whose lane 0 is broadcast with
//!    `shuffle_sync`.
//! 4. `absmax_quantize`: per-row abs-max then FP8 quantisation, the MXFP8
//!    pattern: each 32-element BFloat16 row's largest |x| (by
//!    `redux.sync.max.abs.f32` on sm_100a, or with `redux: false` by the
//!    shuffle tree of kernel 3, which runs on SM89), the scale `amax / 448`,
//!    and each element times `448 / amax` (`rcp.approx.ftz.f32`) converted
//!    to Float8E4M3FN with `cvt.rn.satfinite` (sm_89). One block per row.
//!
//! Run with `cargo run --release --example redux_sync`.

use cute::prelude::*;

/// The integer operations of kernel 1, in its output order.
pub const INT_KINDS: [ReduxKind; 8] = [
    ReduxKind::Add,
    ReduxKind::Min,
    ReduxKind::Max,
    ReduxKind::And,
    ReduxKind::Or,
    ReduxKind::Xor,
    ReduxKind::Umin,
    ReduxKind::Umax,
];

/// `op` on two lanes' values, as `redux.sync` combines them.
fn int_combine(kind: ReduxKind, a: Int32, b: Int32) -> Int32 {
    let unsigned = |f: fn(Uint32, Uint32) -> Uint32| f(a.bitcast(), b.bitcast()).bitcast::<i32>();
    match kind {
        ReduxKind::Add => a + b,
        ReduxKind::Min => a.min(b),
        ReduxKind::Max => a.max(b),
        ReduxKind::And => a & b,
        ReduxKind::Or => a | b,
        ReduxKind::Xor => a ^ b,
        ReduxKind::Umin => unsigned(|x, y| x.min(y)),
        ReduxKind::Umax => unsigned(|x, y| x.max(y)),
        _ => unreachable!("integer kinds only"),
    }
}

/// Kernel 1, `integer_reductions_kernel`: `vals` (one Int32 per lane) into
/// `out` (the eight `redux.sync` results) and `tree` (the same by a
/// butterfly of `shuffle_sync_bfly`).
pub fn integer_reductions() -> Kernel {
    Kernel::builder("integer_reductions").block_dim([WARP_SIZE as u32, 1, 1]).build(|| {
        let vals = param_tensor::<i32>("vals", Layout::col_major(WARP_SIZE));
        let out = param_tensor::<i32>("out", Layout::col_major(INT_KINDS.len()));
        let tree = param_tensor::<i32>("tree", Layout::col_major(INT_KINDS.len()));
        let tidx = thread_idx().x;
        let v = vals.load(tidx);
        let results: Vec<Int32> = INT_KINDS.iter().map(|&k| warp_redux_sync(v, k)).collect();
        let trees: Vec<Int32> = INT_KINDS
            .iter()
            .map(|&k| {
                let mut x = v;
                for offset in [16, 8, 4, 2, 1] {
                    x = int_combine(k, x, shuffle_sync_bfly(x, offset));
                }
                x
            })
            .collect();
        if_(tidx.eq(0), || {
            for (i, (&r, &t)) in results.iter().zip(&trees).enumerate() {
                out.store(i, r);
                tree.store(i, t);
            }
        });
    })
}

/// Kernel 1's results for `vals` in plain Rust.
pub fn integer_reference(vals: &[i32]) -> [i32; 8] {
    let u: Vec<u32> = vals.iter().map(|&v| v as u32).collect();
    [
        vals.iter().fold(0i32, |a, &b| a.wrapping_add(b)),
        *vals.iter().min().unwrap(),
        *vals.iter().max().unwrap(),
        vals.iter().fold(-1, |a, &b| a & b),
        vals.iter().fold(0, |a, &b| a | b),
        vals.iter().fold(0, |a, &b| a ^ b),
        *u.iter().min().unwrap() as i32,
        *u.iter().max().unwrap() as i32,
    ]
}

/// Kernel 2, `float_reductions_kernel`: Float32 `vals` into `out`: min,
/// max, the largest and the smallest absolute value. sm_100a.
pub fn float_reductions() -> Kernel {
    Kernel::builder("float_reductions").block_dim([WARP_SIZE as u32, 1, 1]).build(|| {
        let vals = param_tensor::<f32>("vals", Layout::col_major(WARP_SIZE));
        let out = param_tensor::<f32>("out", Layout::col_major(4));
        let tidx = thread_idx().x;
        let v = vals.load(tidx);
        let abs = ReduxOptions { abs: true, ..Default::default() };
        let fmin_r = warp_redux_sync(v, ReduxKind::Fmin);
        let fmax_r = warp_redux_sync(v, ReduxKind::Fmax);
        let absmax_r = warp_redux_sync_with(v, ReduxKind::Fmax, FULL_MASK, abs);
        let absmin_r = warp_redux_sync_with(v, ReduxKind::Fmin, FULL_MASK, abs);
        if_(tidx.eq(0), || {
            out.store(0, fmin_r);
            out.store(1, fmax_r);
            out.store(2, absmax_r);
            out.store(3, absmin_r);
        });
    })
}

/// Kernel 3, `redux_vs_shuffle_kernel`: the largest absolute value of
/// `vals`, by `redux.sync.max.abs.f32` into `redux_arr` (with `redux`;
/// sm_100a) and by a `shuffle_sync_down` tree, broadcast from lane 0, into
/// `shuffle_arr`.
pub fn redux_vs_shuffle(redux: bool) -> Kernel {
    let name = if redux { "redux_vs_shuffle" } else { "shuffle_absmax" };
    Kernel::builder(name).block_dim([WARP_SIZE as u32, 1, 1]).build(move || {
        let vals = param_tensor::<f32>("vals", Layout::col_major(WARP_SIZE));
        let redux_arr = param_tensor::<f32>("redux_arr", Layout::col_major(1));
        let shuffle_arr = param_tensor::<f32>("shuffle_arr", Layout::col_major(1));
        let tidx = thread_idx().x;
        let v = vals.load(tidx);
        let absmax_redux = redux.then(|| {
            let abs = ReduxOptions { abs: true, ..Default::default() };
            warp_redux_sync_with(v, ReduxKind::Fmax, FULL_MASK, abs)
        });
        // |v| as max(v, 0 - v).
        let mut absval = v.max(0.0f32.into_value() - v);
        for offset in [16, 8, 4, 2, 1] {
            let other = shuffle_sync_down(absval, offset);
            absval = absval.max(other);
        }
        let absmax_shuffle = shuffle_sync(absval, 0);
        if_(tidx.eq(0), || {
            if let Some(r) = absmax_redux {
                redux_arr.store(0, r);
            }
            shuffle_arr.store(0, absmax_shuffle);
        });
    })
}

/// Float8E4M3FN's largest finite value (the example's `_FP8_MAX`).
pub const FP8_MAX: f32 = 448.0;

/// Kernel 4's rows (the example's `_DEFAULT_NUM_ROWS`).
pub const DEFAULT_NUM_ROWS: usize = 8;

/// The largest |v| of the warp by kernel 3's `shuffle_sync_down` tree,
/// broadcast from lane 0.
fn shuffle_absmax(v: Float32) -> Float32 {
    let mut absval = v.max(0.0f32.into_value() - v);
    for offset in [16, 8, 4, 2, 1] {
        let other = shuffle_sync_down(absval, offset);
        absval = absval.max(other);
    }
    shuffle_sync(absval, 0)
}

/// Kernel 4, `absmax_quantize_kernel`: each row of `inp` (`rows` x 32
/// BFloat16, one block of one warp per row, one element per lane) quantised
/// to Float8E4M3FN in `fp8_arr` with the per-row dequantisation scale `amax
/// / 448` in `scales_arr`. The abs-max is `redux.sync.max.abs.f32` with
/// `redux` (sm_100a), else the shuffle tree (SM89: the FP8 `cvt` needs
/// sm_89).
pub fn absmax_quantize(rows: usize, redux: bool) -> Kernel {
    let name = if redux { "absmax_quantize" } else { "absmax_quantize_shuffle" };
    Kernel::builder(name).block_dim([WARP_SIZE as u32, 1, 1]).build(move || {
        let inp = param_tensor::<BF16>("inp", Layout::row_major((rows, WARP_SIZE)));
        let fp8_arr = param_tensor::<F8E4M3FN>("fp8_arr", Layout::row_major((rows, WARP_SIZE)));
        let scales_arr = param_tensor::<f32>("scales_arr", Layout::col_major(rows));
        let tidx = thread_idx().x;
        let row = block_idx().x;
        let inv_fp8_max = (1.0f64 / FP8_MAX as f64) as f32;
        // Step 1: load the BFloat16 element and widen it to Float32.
        let x = inp.load((row, tidx)).to::<f32>();
        // Step 2: the warp's abs-max.
        let amax = if redux {
            warp_redux_sync_with(x, ReduxKind::Fmax, FULL_MASK, ReduxOptions { abs: true, ..Default::default() })
        } else {
            shuffle_absmax(x)
        };
        // Step 3: guard against all-zero rows; the inverse scale 448 / amax.
        let safe_amax = math::max(amax, 1e-12f32);
        let inv_scale = math::rcp_with(safe_amax, MathOptions::APPROX.with_ftz()) * FP8_MAX;
        // Step 4: quantise; the satfinite cvt saturates to +-448.
        fp8_arr.store((row, tidx), (x * inv_scale).to::<F8E4M3FN>());
        // Step 5: lane 0 writes the row's dequantisation scale.
        if_(tidx.eq(0), || scales_arr.store(row, safe_amax * inv_fp8_max));
    })
}

/// Kernel 4's launch: one block of one warp per row.
pub fn quantize_launch(rows: usize) -> LaunchConfig {
    LaunchConfig::new([rows as u32, 1, 1], [WARP_SIZE as u32, 1, 1])
}

/// Kernel 4's input: `rows` x 32 BFloat16 values of a standard normal
/// distribution (the example's `torch.randn`; Box-Muller over a seeded
/// generator), with row 1 all zeros to exercise the `1e-12` guard when
/// there are at least two rows.
pub fn quantize_inputs(rows: usize, seed: u64) -> Vec<BF16> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..rows * WARP_SIZE)
        .map(|i| {
            let (u, v) = (uniform(), uniform());
            let normal = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
            if rows >= 2 && i / WARP_SIZE == 1 { BF16::from_f32(0.0) } else { BF16::from_f32(normal as f32) }
        })
        .collect()
}

/// Kernel 4 in plain Rust, as the simulator computes it (`rcp.approx` as
/// the correctly rounded reciprocal): the FP8 values and the scales.
pub fn quantize_reference(inp: &[BF16]) -> (Vec<F8E4M3FN>, Vec<f32>) {
    let inv_fp8_max = (1.0f64 / FP8_MAX as f64) as f32;
    let (mut fp8, mut scales) = (Vec::new(), Vec::new());
    for row in inp.chunks(WARP_SIZE) {
        let amax = row.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
        let safe_amax = amax.max(1e-12);
        let inv_scale = (1.0 / safe_amax) * FP8_MAX;
        fp8.extend(row.iter().map(|x| F8E4M3FN::from_f32(x.to_f32() * inv_scale)));
        scales.push(safe_amax * inv_fp8_max);
    }
    (fp8, scales)
}

/// The example's check (`verify_absmax_quantize`) of `fp8` and `scales`
/// for `inp`: the scales within `1e-4` of `amax / 448` and the
/// dequantised values within `0.1` of the reference's (which divides
/// exactly and clamps to +-448 before converting), or one FP8 step from it:
/// `rcp.approx` can move a product across a rounding boundary, and near 448
/// an E4M3 step is 32 / 448 of the row's abs-max, more than `0.1` once that
/// passes 1.4 (the example's 8 rows of `randn` stay below it by chance).
pub fn verify_quantized(inp: &[BF16], fp8: &[F8E4M3FN], scales: &[f32]) {
    for (r, row) in inp.chunks(WARP_SIZE).enumerate() {
        let amax = row.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
        let ref_scale = amax / FP8_MAX;
        // An all-zero row has the guarded scale 1e-12 / 448.
        if amax > 0.0 {
            assert!((scales[r] - ref_scale).abs() <= 1e-4, "row {r}: scale {} vs {ref_scale}", scales[r]);
        }
        for (j, x) in row.iter().enumerate() {
            let q = if amax > 0.0 {
                F8E4M3FN::from_f32((x.to_f32() * (FP8_MAX / amax)).clamp(-FP8_MAX, FP8_MAX))
            } else {
                F8E4M3FN::from_f32(0.0)
            };
            let (got, want) = (fp8[r * WARP_SIZE + j].to_f32() * scales[r], q.to_f32() * ref_scale);
            let step = (fp8[r * WARP_SIZE + j].0 as i32 - q.0 as i32).abs() <= 1;
            assert!((got - want).abs() <= 0.1 || step, "row {r} element {j}: dequantised {got} vs {want}");
        }
    }
}

/// Simulate kernel 4 (with or without `redux.sync`) on `inp` and check it
/// against [`quantize_reference`] bit for bit and the example's check.
pub fn run_quantize(inp: &[BF16], redux: bool) -> (Vec<F8E4M3FN>, Vec<f32>) {
    let rows = inp.len() / WARP_SIZE;
    let mut args = vec![
        KernelArg::buffer(inp),
        KernelArg::buffer(&vec![F8E4M3FN(0); rows * WARP_SIZE]),
        KernelArg::f32(vec![0.0; rows]),
    ];
    absmax_quantize(rows, redux)
        .simulate_all_orders(quantize_launch(rows), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let (fp8, scales) = (args[1].to_vec::<F8E4M3FN>(), args[2].as_f32().to_vec());
    let (want_fp8, want_scales) = quantize_reference(inp);
    assert_eq!(fp8.iter().map(|q| q.0).collect::<Vec<_>>(), want_fp8.iter().map(|q| q.0).collect::<Vec<_>>());
    assert_eq!(scales, want_scales);
    verify_quantized(inp, &fp8, &scales);
    (fp8, scales)
}

/// One warp.
pub fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [WARP_SIZE as u32, 1, 1])
}

/// The example's inputs: `0, 1, ..., 31` for kernel 1, and 32 evenly spaced
/// values from -15.5 to 15.5 for kernels 2 and 3.
pub fn int_inputs() -> Vec<i32> {
    (0..WARP_SIZE as i32).collect()
}

pub fn float_inputs() -> Vec<f32> {
    (0..WARP_SIZE).map(|i| i as f32 - 15.5).collect()
}

/// Kernel 2's and 3's reference: min, max, largest and smallest |v|.
pub fn float_reference(vals: &[f32]) -> [f32; 4] {
    let fold = |init: f32, f: fn(f32, f32) -> f32, g: fn(f32) -> f32| vals.iter().fold(init, |a, &b| f(a, g(b)));
    [
        fold(f32::INFINITY, f32::min, |x| x),
        fold(f32::NEG_INFINITY, f32::max, |x| x),
        fold(0.0, f32::max, f32::abs),
        fold(f32::INFINITY, f32::min, f32::abs),
    ]
}

/// Simulate kernel 1 on `vals` and check both forms against Rust.
pub fn run_integer(vals: &[i32]) {
    let mut args = vec![KernelArg::buffer(vals), KernelArg::buffer(&[0i32; 8]), KernelArg::buffer(&[0i32; 8])];
    integer_reductions()
        .simulate_all_orders(launch(), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let want = integer_reference(vals);
    assert_eq!(args[1].to_vec::<i32>(), want, "redux.sync on {vals:?}");
    assert_eq!(args[2].to_vec::<i32>(), want, "the shuffle tree on {vals:?}");
}

/// Simulate kernels 2 and 3 on `vals` and check them against Rust.
pub fn run_float(vals: &[f32]) {
    let want = float_reference(vals);
    let mut args = vec![KernelArg::f32(vals.to_vec()), KernelArg::f32(vec![0.0; 4])];
    float_reductions()
        .simulate_all_orders(launch(), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].as_f32(), want, "float_reductions on {vals:?}");
    for redux in [true, false] {
        let mut args = vec![KernelArg::f32(vals.to_vec()), KernelArg::f32(vec![0.0]), KernelArg::f32(vec![0.0])];
        redux_vs_shuffle(redux)
            .simulate_all_orders(launch(), &mut args, &SimOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(args[2].as_f32(), [want[2]], "the shuffle tree's abs-max");
        if redux {
            assert_eq!(args[1].as_f32(), [want[2]], "redux.sync.max.abs.f32");
        }
    }
}

#[allow(dead_code)]
fn main() {
    run_integer(&int_inputs());
    println!("integer_reductions: correct ({:?})", integer_reference(&int_inputs()));
    run_float(&float_inputs());
    println!("float_reductions and redux_vs_shuffle: correct ({:?})", float_reference(&float_inputs()));
    let inp = quantize_inputs(DEFAULT_NUM_ROWS, 0);
    for redux in [true, false] {
        let (_, scales) = run_quantize(&inp, redux);
        println!("absmax_quantize (redux.sync: {redux}): correct, scales {scales:?}");
    }
    match Toolchain::discover() {
        Some(tc) => {
            for (kernel, arch) in [
                (integer_reductions(), Arch::Sm80),
                (redux_vs_shuffle(true), Arch::Sm100a),
                (absmax_quantize(DEFAULT_NUM_ROWS, true), Arch::Sm100a),
            ] {
                match kernel.compile_ptx(&tc, arch) {
                    Ok(ptx) => {
                        let redux = ptx.lines().filter(|l| l.contains("redux.sync")).count();
                        println!("PTX for {} ({}): {redux} redux.sync", kernel.name(), arch.chip());
                    }
                    Err(e) => eprintln!("PTX compilation failed: {e}"),
                }
            }
        }
        None => println!("(no mlir-opt found; set CUTE_MLIR_OPT to compile to PTX)"),
    }
}

// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's sorting-free samplers ([`flashinfer::sampling`]): `sampling_from_probs`,
//! `top_k_sampling_from_probs`, `top_p_sampling_from_probs` and
//! `top_k_top_p_sampling_from_probs` (the "joint" filter). Simulated on small rows, then
//! lowered to PTX if a toolchain is found. With `--features cuda` and `GPU=1` it runs on the
//! GPU instead: checks against the f64 emulation (same Philox draws) and the top-k / top-p
//! sets, then CUDA-graph timings on shapes of FlashInfer's `benchmarks/bench_sampling.py`
//! (vocab 128512, batch 1 to 512, softmax of normal(1) and gumbel(0.1) logits), the same
//! data `sampling_flashinfer_bench.py` builds for FlashInfer.
//!
//! It also covers `min_p_sampling_from_probs`, `top_p_renorm_probs`, `top_k_renorm_probs`,
//! `top_k_mask_logits` and `softmax` with a temperature (`sampling_variants_flashinfer_bench.py`
//! is the FlashInfer side of their benchmark).
//!
//! The newer entry points are `sampling_from_logits` (Gumbel-max), `top_k_top_p_sampling_from_probs`
//! and `..._from_logits` in both `filter_apply_order`s, and per-row seed / offset arrays;
//! `SAMPLING_NEW=1` runs just their checks and benchmark (`sampling_logits_flashinfer_bench.py`
//! is the FlashInfer side).
//!
//! `cargo run -p cute-flashinfer --example sampling`; on a GPU,
//! `GPU=1 cargo run -p cute-flashinfer --features cuda --example sampling`.

#[cfg(feature = "cuda")]
use std::error::Error;

#[cfg(feature = "cuda")]
use cute::cuda::CudaContext;
use cute::prelude::*;
#[cfg(feature = "cuda")]
use flashinfer::sampling::gpu::{
    Kernels, OpKernels, TransformKernels, run, run_filtered, run_logits, run_op, run_top_k_first_compact, run_transform,
};
use flashinfer::sampling::spec;
use flashinfer::sampling::transform::{self, Kind, TConfig, TProblem};
use flashinfer::sampling::*;

/// The benchmarked samplers: mode, top_k, top_p.
pub const BENCH: [(Mode, i32, f32); 6] = [
    (Mode::Plain, 0, 1.0),
    (Mode::TopK, 10, 1.0),
    (Mode::TopK, 1000, 1.0),
    (Mode::TopP, 0, 0.5),
    (Mode::TopP, 0, 0.9),
    (Mode::TopKTopP, 50, 0.9),
];

/// The benchmarked min-p samplers (`top_p` carries min_p) and row operations: op, top_k, param.
pub const BENCH_MIN_P: [f32; 2] = [0.1, 0.5];
pub const BENCH_OPS: [(Op, i32, f32); 8] = [
    (Op::TopPRenorm, 0, 0.5),
    (Op::TopPRenorm, 0, 0.9),
    (Op::TopKRenorm, 10, 0.0),
    (Op::TopKRenorm, 1000, 0.0),
    (Op::TopKMask, 10, 0.0),
    (Op::TopKMask, 1000, 0.0),
    (Op::Softmax, 0, 0.7),
    (Op::Softmax, 0, 1.0),
];
const OPS: [Op; 4] = [Op::TopPRenorm, Op::TopKRenorm, Op::TopKMask, Op::Softmax];

/// Host reference of `top_k_renorm_probs` on rows of `d`: keep the values at or above the k-th
/// largest, scale to sum one.
#[cfg(feature = "cuda")]
fn host_top_k_renorm(probs: &[f32], d: usize, k: &[i32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(probs.len());
    for (b, r) in probs.chunks(d).enumerate() {
        let mut sorted = r.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let thr = sorted[(k[b] as usize).clamp(1, d) - 1];
        let kept: f64 = r.iter().filter(|&&x| x >= thr).map(|&x| x as f64).sum();
        out.extend(r.iter().map(|&x| if x >= thr { (x as f64 / kept) as f32 } else { 0.0 }));
    }
    out
}

/// Host reference of `top_k_mask_logits` then softmax.
#[cfg(feature = "cuda")]
fn host_mask_softmax(logits: &[f32], d: usize, k: &[i32]) -> Vec<f32> {
    let mut masked = Vec::with_capacity(logits.len());
    for (b, r) in logits.chunks(d).enumerate() {
        let mut sorted = r.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let thr = sorted[(k[b] as usize).clamp(1, d) - 1];
        masked.extend(r.iter().map(|&x| if x >= thr { x as f64 } else { f64::NEG_INFINITY }));
    }
    softmax_rows(&masked, d)
}

/// Checks and benchmark of `sampling_from_logits`, `top_k_top_p_sampling_from_{probs,logits}`
/// (both orders) and the per-seed samplers.
#[cfg(feature = "cuda")]
fn new_ops(
    ctx: &CudaContext,
    tc: &Toolchain,
    ks: &mut Kernels<'_>,
    oks: &mut OpKernels<'_>,
) -> Result<(), Box<dyn Error>> {
    for (batch, d, dist) in
        [(7usize, 1000usize, Dist::Normal(3.0)), (16, 32000, Dist::Normal(1.0)), (12, 128256, Dist::Gumbel(0.1))]
    {
        let mut lp = LogitsProblem::new(batch, d, dist, 99);
        lp.offset = 12;
        let (out, _) = run_logits(ctx, Config::new(Mode::Plain), tc, &lp, false)?;
        let near = lp.check(&out);
        lp.seeds = (0..batch as u64).map(|b| 5 + b).collect();
        lp.offsets = (0..batch as u64).map(|b| 4 * b).collect();
        let (out, _) = run_logits(ctx, Config::new(Mode::Plain).per_seed(true), tc, &lp, false)?;
        let near2 = lp.check(&out);
        println!(
            "check sampling_from_logits batch {batch} d {d} {}: ok ({near} / per-seed {near2} rows within rounding)",
            dist.name()
        );

        // Per-row seeds and offsets on a probability sampler.
        let mut p = Problem::new(batch, d, dist, 50, 0.9, 3);
        p.seeds = (0..batch as u64).map(|b| 11 * b + 1).collect();
        p.offsets = (0..batch as u64).map(|b| b % 3).collect();
        let (out, valid, _) = run(ctx, ks, &p, Config::new(Mode::TopKTopP).per_seed(true), false)?;
        let near = p.check(Mode::TopKTopP, &out, &valid);
        println!("check top_k_top_p per-seed batch {batch} d {d}: ok ({near} rows within rounding)");

        // The pipelines, against host references built from the same Philox draws.
        let logits: Vec<f32> = lp.logits.clone();
        let probs = softmax_rows(&logits.iter().map(|&x| x as f64).collect::<Vec<_>>(), d);
        let ks_row: Vec<i32> = (0..batch as i32).map(|b| 1 + b * 11).collect();
        let ps_row: Vec<f32> = (0..batch).map(|b| 0.3 + 0.65 * b as f32 / batch as f32).collect();
        for per_row in [false, true] {
            let (kk, pp) = if per_row { (ks_row.clone(), ps_row.clone()) } else { (vec![40; batch], vec![0.9; batch]) };
            for from_logits in [false, true] {
                for order in [FilterOrder::Joint, FilterOrder::TopKFirst] {
                    let input = if from_logits { logits.clone() } else { probs.clone() };
                    let mut q = Problem::from_probs(batch, d, input, 0, 0.0, 7);
                    q.top_k = kk.clone();
                    q.top_p = pp.clone();
                    q.offset = 4;
                    let (out, valid, _) = run_filtered(ctx, ks, oks, &q, from_logits, order, per_row, false)?;
                    // The reference problem: what the sampler sees, and its mode.
                    let (mode, sees) = match (from_logits, order) {
                        (false, FilterOrder::Joint) => (Mode::TopKTopP, probs.clone()),
                        (true, FilterOrder::Joint) => (Mode::TopKTopP, probs.clone()),
                        (false, FilterOrder::TopKFirst) => (Mode::TopP, host_top_k_renorm(&probs, d, &kk)),
                        (true, FilterOrder::TopKFirst) => (Mode::TopP, host_mask_softmax(&logits, d, &kk)),
                    };
                    let mut r = Problem::from_probs(batch, d, sees, 0, 0.0, 7);
                    r.top_k = kk.clone();
                    r.top_p = pp.clone();
                    r.offset = 4;
                    let near = r.check(mode, &out, &valid);
                    println!(
                        "check top_k_top_p from_{} {:?} per_row {per_row} batch {batch} d {d}: ok ({near} rows within rounding)",
                        if from_logits { "logits" } else { "probs" },
                        order
                    );
                }
            }
        }
    }

    // The compact top-k-first path: every sample is in the top-k / top-p set.
    for (batch, d) in [(7usize, 1000usize), (16, 32000), (12, 128256)] {
        let lp = LogitsProblem::new(batch, d, Dist::Gumbel(0.3), 5);
        for (k, pp) in [(4, 0.5f32), (50, 0.9), (51, 0.7), (256, 0.95)] {
            let mut q = Problem::from_probs(batch, d, lp.logits.clone(), k, pp, 9);
            q.offset = 4;
            let (out, valid, _) = run_top_k_first_compact(ctx, ks, oks, tc, &q, false)?;
            let sees = host_mask_softmax(&lp.logits, d, &q.top_k);
            let r = Problem::from_probs(batch, d, sees, k, pp, 9);
            for b in 0..batch {
                assert!(valid[b] == 1 && r.in_set(Mode::TopP, b, out[b] as usize, 1e-4), "row {b}: token {}", out[b]);
            }
            println!("check top_k_top_p from_logits compact batch {batch} d {d} k {k} p {pp}: ok");
        }
    }

    let d = 128256;
    println!("kernel                          batch  time(us)");
    let all = LogitsProblem::new(128, d, Dist::Normal(1.0), 42);
    let probs_all = softmax_rows(&all.logits.iter().map(|&x| x as f64).collect::<Vec<_>>(), d);
    for b in [1usize, 16, 128] {
        let lp = LogitsProblem { batch: b, logits: all.logits[..b * d].to_vec(), ..all.clone() };
        let (_, t) = run_logits(ctx, Config::new(Mode::Plain), tc, &lp, true)?;
        println!("{:30}  {b:5}  {:8.2}", "sampling_from_logits", t.unwrap());
    }
    for (from_logits, order) in [
        (false, FilterOrder::Joint),
        (false, FilterOrder::TopKFirst),
        (true, FilterOrder::Joint),
        (true, FilterOrder::TopKFirst),
    ] {
        for b in [1usize, 16, 128] {
            let input = if from_logits { all.logits[..b * d].to_vec() } else { probs_all[..b * d].to_vec() };
            let q = Problem::from_probs(b, d, input, 50, 0.9, 42);
            let (_, _, t) = run_filtered(ctx, ks, oks, &q, from_logits, order, false, true)?;
            let name = format!("top_k_top_p_from_{}_{:?}", if from_logits { "logits" } else { "probs" }, order);
            println!("{name:30}  {b:5}  {:8.2}", t.unwrap());
        }
    }
    for b in [1usize, 16, 128] {
        let q = Problem::from_probs(b, d, all.logits[..b * d].to_vec(), 50, 0.9, 42);
        let (_, _, t) = run_top_k_first_compact(ctx, ks, oks, tc, &q, true)?;
        println!("{:30}  {b:5}  {:8.2}", "top_k_top_p_from_logits_compact", t.unwrap());
    }
    Ok(())
}

/// Window lengths of a benchmark: `full` rows are all `seq`; else a fixed spread over
/// `[seq / 4, seq]` (the same arithmetic as `topk_flashinfer_bench.py`).
#[cfg(feature = "cuda")]
fn bench_lengths(rows: usize, seq: usize, full: bool) -> Vec<i32> {
    (0..rows as u64)
        .map(|i| if full { seq as i32 } else { (seq as u64 / 4 + (i * 2654435761) % (3 * seq as u64 / 4 + 1)) as i32 })
        .collect()
}

/// Checks and benchmark of `top_k_page_table_transform` and `top_k_ragged_transform` on the GPU.
#[cfg(feature = "cuda")]
fn transform_gpu(ctx: &CudaContext, tc: &Toolchain) -> Result<(), Box<dyn Error>> {
    let mut ks = TransformKernels::new(ctx, tc);
    // Checks: short rows, row starts, page sizes, several blocks per row, ties.
    for (kind, stride, k, shift, split, rows) in [
        (Kind::Ragged, 70000usize, 2048usize, 0u32, 1usize, 5usize),
        (Kind::Ragged, 70000, 2048, 0, 8, 5),
        (Kind::PageTable, 70000, 2048, 0, 4, 5),
        (Kind::PageTable, 70000, 512, 6, 3, 5),
        (Kind::PageTable, 9000, 2048, 2, 1, 9),
    ] {
        let lengths: Vec<i32> =
            (0..rows as i32).map(|i| [stride as i32 - 40, 100, 2048, 2049, 5000][i as usize % 5]).collect();
        let starts: Vec<i32> = (0..rows as i32).map(|i| [0, 3, 40, 7, 1][i as usize % 5] * 4 + 4 * (i / 5)).collect();
        let p = TProblem::new(kind, stride, k, Dist::Normal(1.0), lengths, starts, 3, shift);
        let cfg = TConfig::new(kind).split(split).page_shift(shift);
        let (out, _) = run_transform(ctx, &mut ks, &p, cfg, false)?;
        p.check(&out);
        println!("check top_k {kind:?} stride {stride} k {k} page_shift {shift} split {split}: ok");
    }
    let mut q = TProblem::new(Kind::Ragged, 20000, 2048, Dist::Normal(1.0), vec![20000; 4], vec![0; 4], 4, 0);
    for x in q.input.iter_mut() {
        *x = (*x * 2.0).round() / 2.0;
    }
    let (out, _) = run_transform(ctx, &mut ks, &q, TConfig::new(Kind::Ragged).split(4), false)?;
    q.check(&out);
    println!("check top_k ragged quantised scores split 4: ok");

    println!("kernel                 lengths  batch  seq     k      time(us) by threads x blocks per row");
    let splits: Vec<usize> =
        std::env::var("SPLITS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let thread_counts: Vec<usize> = std::env::var("THREADS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_default();
    for seq in [4096usize, 16384, 65536, 131072] {
        for b in [1usize, 4, 16, 64] {
            for full in [true, false] {
                for kind in [Kind::PageTable, Kind::Ragged] {
                    let k = 2048usize;
                    let shift = 0;
                    let p = TProblem::new(
                        kind,
                        seq,
                        k,
                        Dist::Normal(1.0),
                        bench_lengths(b, seq, full),
                        vec![0; b],
                        b,
                        shift,
                    );
                    let mut line = String::new();
                    let auto = TConfig::for_rows(kind, b, seq);
                    let (out, t) = run_transform(ctx, &mut ks, &p, auto, true)?;
                    p.check(&out);
                    line += &format!("  auto({}x{}):{:8.2}", auto.threads, auto.split, t.unwrap());
                    for &threads in &thread_counts {
                        for &split in &splits {
                            if seq / split < 2048 {
                                continue;
                            }
                            let cfg = TConfig::new(kind).threads(threads).split(split);
                            let (out, t) = run_transform(ctx, &mut ks, &p, cfg, true)?;
                            if b <= 4 {
                                p.check(&out);
                            }
                            line += &format!("  {threads}x{split}:{:8.2}", t.unwrap());
                        }
                    }
                    let name = format!("{kind:?}");
                    println!("{name:21} {:7}  {b:5}  {seq:6} {k:5} {line}", if full { "full" } else { "var" });
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let tc = Toolchain::discover().ok_or("no mlir-opt found; set CUTE_MLIR_OPT")?;
    let mut ks = Kernels::new(&ctx, &tc);
    println!("device: {}", ctx.name());
    if std::env::var_os("TRANSFORM_ONLY").is_some() {
        return transform_gpu(&ctx, &tc);
    }
    if std::env::var_os("SAMPLING_NEW").is_some() {
        let mut oks = OpKernels::new(&ctx, &tc);
        return new_ops(&ctx, &tc, &mut ks, &mut oks);
    }
    let modes = [Mode::Plain, Mode::TopK, Mode::TopP, Mode::TopKTopP];
    for (batch, d, dist) in
        [(7usize, 1000usize, Dist::Normal(3.0)), (16, 32000, Dist::Normal(1.0)), (12, 128256, Dist::Gumbel(0.1))]
    {
        let mut p = Problem::new(batch, d, dist, 50, 0.9, 1234);
        p.offset = 5;
        for m in modes {
            let (out, valid, _) = run(&ctx, &mut ks, &p, Config::new(m), false)?;
            let near = p.check(m, &out, &valid);
            println!("check {} batch {batch} d {d} {}: ok ({near} rows at a rounding margin)", m.name(), dist.name());
        }
        p.top_k = (0..batch as i32).map(|b| 1 + b * 7).collect();
        p.top_p = (0..batch).map(|b| 0.05 + 0.9 * b as f32 / batch as f32).collect();
        let (out, valid, _) = run(&ctx, &mut ks, &p, Config::new(Mode::TopKTopP).per_row(true), false)?;
        let near = p.check(Mode::TopKTopP, &out, &valid);
        println!("check top_k_top_p per-row batch {batch} d {d}: ok ({near} rows at a rounding margin)");
        // Min-p (the `top_p` parameter carries min_p), per-row.
        p.top_p = (0..batch).map(|b| 0.02 + 0.9 * b as f32 / batch as f32).collect();
        let (out, valid, _) = run(&ctx, &mut ks, &p, Config::new(Mode::MinP).per_row(true), false)?;
        let near = p.check(Mode::MinP, &out, &valid);
        println!("check min_p per-row batch {batch} d {d}: ok ({near} rows at a rounding margin)");
    }
    let mut oks = OpKernels::new(&ctx, &tc);
    for op in OPS {
        for (batch, d, dist) in
            [(7usize, 1000usize, Dist::Normal(3.0)), (9, 32000, Dist::Normal(1.0)), (5, 128256, Dist::Gumbel(0.1))]
        {
            let (k, param) = if op == Op::Softmax { (0, 0.7) } else { (50, 0.9) };
            let mut p = OpProblem::new(op, batch, d, dist, k, param);
            let (out, _, _) = run_op(&ctx, &mut oks, &p, OpConfig::new(op), false)?;
            let e1 = p.check(op, &out);
            p.top_k = (0..batch as i32).map(|b| 1 + b * 97).collect();
            p.param = (0..batch)
                .map(|b| if op == Op::Softmax { 0.3 + b as f32 } else { 0.1 + 0.85 * b as f32 / batch as f32 })
                .collect();
            let (out, _, _) = run_op(&ctx, &mut oks, &p, OpConfig::new(op).per_row(true), false)?;
            let e2 = p.check(op, &out);
            println!(
                "check {} batch {batch} d {d} {}: max relative error {e1:.1e} / per-row {e2:.1e}",
                op.name(),
                dist.name()
            );
        }
    }

    // Multi-block rows against the reference on the GPU.
    for op in [Op::Softmax, Op::TopKMask, Op::TopKRenorm, Op::TopPRenorm] {
        for (batch, d, split) in [(1usize, 128512usize, 16usize), (3, 32000, 8), (2, 1000, 3)] {
            let (k, param) = if op == Op::Softmax { (0, 0.7) } else { (50, 0.9) };
            let p = OpProblem::new(op, batch, d, Dist::Gumbel(0.1), k, param);
            let (out, _, _) = run_op(&ctx, &mut oks, &p, OpConfig::new(op).split(split), false)?;
            println!(
                "check {} split {split} batch {batch} d {d}: max relative error {:.1e}",
                op.name(),
                p.check(op, &out)
            );
        }
    }

    // `top_k` (values and indices) and chain speculative sampling against their references.
    for (batch, d, k) in [(7usize, 1000usize, 50i32), (9, 32000, 256), (3, 128512, 2048)] {
        let p = OpProblem::new(Op::TopK, batch, d, Dist::Normal(1.0), k, 0.0);
        let (v, i, _) = run_op(&ctx, &mut oks, &p, OpConfig::new(Op::TopK), false)?;
        p.check_top_k(&v, &i);
        println!("check top_k batch {batch} d {d} k {k}: ok");
    }
    for (batch, d, k, split) in [(1usize, 128512usize, 50i32, 16usize), (3, 32000, 2048, 8), (2, 1000, 20, 3)] {
        let p = OpProblem::new(Op::TopK, batch, d, Dist::Gumbel(0.1), k, 0.0);
        let (v, i, _) = run_op(&ctx, &mut oks, &p, OpConfig::new(Op::TopK).split(split), false)?;
        p.check_top_k(&v, &i);
        println!("check top_k split {split} batch {batch} d {d} k {k}: ok");
    }
    for (batch, n, d) in [(5usize, 3usize, 1000usize), (16, 4, 32000), (3, 40, 128512)] {
        let mut p = spec::Problem::new(batch, n, d, Dist::Normal(1.0), Dist::Normal(1.0), 8);
        p.offset = 3;
        let (out, acc, em, _) = spec::gpu::run(&ctx, &tc, &p, 1024, false)?;
        p.check(&out, &acc, &em, 3, 5);
        println!("check chain_speculative_sampling batch {batch} n {n} d {d}: ok");
    }

    let d = 128512;
    // `MAX_BATCH=16` benchmarks only the small batches.
    let max_batch: usize = std::env::var("MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let batches: Vec<usize> = [1usize, 16, 128, 512].into_iter().filter(|&b| b <= max_batch).collect();
    println!("kernel        k     p     distribution  batch  time(us)");
    // `NEW_ONLY=1` skips the benchmarks of the original samplers.
    let dists: &[Dist] =
        if std::env::var_os("NEW_ONLY").is_some() { &[] } else { &[Dist::Normal(1.0), Dist::Gumbel(0.1)] };
    for &dist in dists {
        let all = Problem::new(*batches.last().unwrap(), d, dist, 0, 1.0, 42);
        for (m, k, pp) in BENCH {
            for &b in &batches {
                let p = Problem::from_probs(b, d, all.probs[..b * d].to_vec(), k, pp, 42);
                let (_, _, t) = run(&ctx, &mut ks, &p, Config::new(m), true)?;
                println!("{:12}  {k:4}  {pp:.1}  {:12}  {b:5}  {:8.2}", m.name(), dist.name(), t.unwrap());
            }
        }
    }
    // min_p, renorm, mask and softmax, on normal(1) data (`sampling_variants_flashinfer_bench.py`).
    println!("kernel        param  batch  time(us)");
    let all = Problem::new(*batches.last().unwrap(), d, Dist::Normal(1.0), 0, 1.0, 42);
    let old = std::env::var_os("NEW_ONLY").is_none();
    for mp in BENCH_MIN_P.into_iter().filter(|_| old) {
        for &b in &batches {
            let p = Problem::from_probs(b, d, all.probs[..b * d].to_vec(), 0, mp, 42);
            let (_, _, t) = run(&ctx, &mut ks, &p, Config::new(Mode::MinP), true)?;
            println!("{:13} {mp:5}  {b:5}  {:8.2}", "min_p", t.unwrap());
        }
    }
    let logits = OpProblem::new(Op::Softmax, *batches.last().unwrap(), d, Dist::Normal(1.0), 0, 1.0);
    for (op, k, pp) in BENCH_OPS.into_iter().filter(|_| old) {
        for &b in &batches {
            let input = if op.takes_logits() { logits.input[..b * d].to_vec() } else { all.probs[..b * d].to_vec() };
            let p = OpProblem { batch: b, d, input, top_k: vec![k; b], param: vec![pp; b] };
            let (_, _, t) = run_op(&ctx, &mut oks, &p, OpConfig::new(op), true)?;
            let param = if op == Op::Softmax || op == Op::TopPRenorm { pp } else { k as f32 };
            let auto = OpConfig::for_batch(op, b);
            let split = if auto.split > 1 {
                let (_, _, ts) = run_op(&ctx, &mut oks, &p, auto, true)?;
                format!("  split {}: {:8.2}", auto.split, ts.unwrap())
            } else {
                String::new()
            };
            println!("{:13} {param:5}  {b:5}  {:8.2}{split}", op.name(), t.unwrap());
        }
    }
    // `top_k` (values and indices) on normal(1) logits, `bench_topk.py`'s randn scores.
    println!("kernel        k      batch  time(us)");
    for &b in &batches {
        for k in [50i32, 256, 1000, 2048] {
            let p = OpProblem {
                batch: b,
                d,
                input: logits.input[..b * d].to_vec(),
                top_k: vec![k; b],
                param: vec![0.0; b],
            };
            let (_, _, t) = run_op(&ctx, &mut oks, &p, OpConfig::new(Op::TopK), true)?;
            let auto = OpConfig::for_batch(Op::TopK, b);
            let split = if auto.split > 1 {
                let (_, _, ts) = run_op(&ctx, &mut oks, &p, auto, true)?;
                format!("  split {}: {:8.2}", auto.split, ts.unwrap())
            } else {
                String::new()
            };
            println!("{:13} {k:5}  {b:5}  {:8.2}{split}", "top_k", t.unwrap());
        }
    }
    // The page-table and ragged top-k transforms (`bench_topk.py`'s shapes: randn scores, k 2048).
    transform_gpu(&ctx, &tc)?;
    // Chain speculative sampling with 4 draft tokens, normal(1) draft and target rows.
    println!("kernel                  n  batch  time(us)");
    for &b in &batches {
        for n in [1usize, 4] {
            let p = spec::Problem::new(b, n, d, Dist::Normal(1.0), Dist::Normal(1.0), 42);
            let (_, _, _, t) = spec::gpu::run(&ctx, &tc, &p, 1024, true)?;
            println!("{:20} {n:3}  {b:5}  {:8.2}", "chain_speculative", t.unwrap());
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::var_os("GPU").is_some() {
        gpu_main().unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    let mut p = Problem::new(4, 1000, Dist::Normal(2.0), 20, 0.8, 7);
    p.offset = 4;
    for m in [Mode::Plain, Mode::TopK, Mode::TopP, Mode::TopKTopP] {
        let out = simulate_checked(&p, Config::new(m).threads(64));
        println!("simulated {} (k 20, p 0.8) d 1000: samples {out:?}", m.name());
    }
    p.top_p = vec![0.2; 4];
    let out = simulate_checked(&p, Config::new(Mode::MinP).threads(64));
    println!("simulated min_p (min_p 0.2) d 1000: samples {out:?}");
    for op in OPS {
        let q = OpProblem::new(op, 2, 1000, Dist::Normal(2.0), 20, if op == Op::Softmax { 0.7 } else { 0.8 });
        let e = simulate_op_checked(&q, OpConfig::new(op).threads(64));
        println!("simulated {} d 1000: max relative error {e:.1e}", op.name());
    }
    let q = OpProblem::new(Op::TopK, 2, 1000, Dist::Normal(2.0), 20, 0.0);
    let (v, i) = simulate_top_k(&q, OpConfig::new(Op::TopK).threads(64));
    println!("simulated top_k d 1000 k 20: row 0 values {:.2?} at {:?}", &v[..4], &i[..4]);
    for kind in [Kind::PageTable, Kind::Ragged] {
        let p = TProblem::new(kind, 1000, 20, Dist::Normal(2.0), vec![1000, 15, 400], vec![0, 0, 8], 3, 2);
        let out = transform::simulate_checked(&p, TConfig::new(kind).threads(64).split(2).page_shift(2));
        println!("simulated top_k {kind:?} k 20 lengths {:?}: row 0 {:?}", p.lengths, &out[..4]);
    }
    let sp = spec::Problem::new(4, 3, 1000, Dist::Normal(1.0), Dist::Normal(1.0), 7);
    let out = spec::simulate_checked(&sp, 64);
    println!("simulated chain_speculative_sampling n 3 d 1000: output ids {out:?}");
    match Toolchain::discover() {
        Some(tc) => {
            for m in [Mode::Plain, Mode::TopK, Mode::TopP, Mode::TopKTopP, Mode::MinP] {
                let k = kernel(Config::new(m));
                let ptx = k
                    .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .ptx()
                    .to_string();
                if let Some(dir) = std::env::var_os("PTX_DIR") {
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.ptx", k.name())), &ptx)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                println!(
                    "PTX {}: {} bytes, {} 16-byte loads, {} shfl, {} bar.sync",
                    k.name(),
                    ptx.len(),
                    ptx.matches("ld.global.v4").count(),
                    ptx.matches("shfl.sync").count(),
                    ptx.matches("bar.sync").count()
                );
            }
        }
        None => println!("no MLIR toolchain; skipped PTX"),
    }
}

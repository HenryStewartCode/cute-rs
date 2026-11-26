// SPDX-License-Identifier: Apache-2.0

//! `flashinfer::sampling` on the inputs of `reference/cases_sampling.py`: the samplers (with FlashInfer's seed
//! and offset), renorm / mask / softmax, top-k, speculative sampling and the top-k index transforms.

use flashinfer::sampling::gpu::{
    Kernels, OpKernels, TransformKernels, run as run_sampler, run_filtered, run_logits, run_op,
    run_top_k_first_compact, run_transform,
};
use flashinfer::sampling::transform::{Kind, TConfig, TProblem};
use flashinfer::sampling::{Config, FilterOrder, LogitsProblem, Mode, Op, OpConfig, OpProblem, Problem, spec};

use std::cell::RefCell;

use cute::cuda::CudaContext;
use cute::prelude::Toolchain;

use crate::case::{Case, Gpu, Res};

/// Compiled kernels, kept between cases: compiling dominates a case's time.
struct Cache {
    ctx: &'static CudaContext,
    tc: &'static Toolchain,
    ks: Kernels<'static>,
    oks: OpKernels<'static>,
    tks: TransformKernels<'static>,
}

thread_local! {
    static CACHE: RefCell<Option<Cache>> = const { RefCell::new(None) };
}

fn cached<R>(gpu: &'static Gpu, f: impl FnOnce(&mut Cache) -> Res<R>) -> Res<R> {
    CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let cache = slot.get_or_insert_with(|| {
            let (ctx, tc) = (&gpu.ctx, &gpu.tc);
            Cache {
                ctx,
                tc,
                ks: Kernels::new(ctx, tc),
                oks: OpKernels::new(ctx, tc),
                tks: TransformKernels::new(ctx, tc),
            }
        });
        f(cache)
    })
}

fn run_case(gpu: &'static Gpu, case: &Case) -> Res<()> {
    let f = case.str("fn");
    match f {
        "sample" | "top_k" | "top_p" | "top_k_top_p_joint" | "min_p" => sampler(gpu, case, f),
        "sample_logits" => sample_logits(gpu, case),
        "top_k_top_p_probs" | "top_k_top_p_logits" => filtered(gpu, case, f == "top_k_top_p_logits"),
        "top_p_renorm" | "top_k_renorm" | "top_k_mask" | "softmax" | "topk" => op(gpu, case, f),
        "spec" => speculative(gpu, case),
        "transform_ragged" | "transform_page" => transform(gpu, case, f == "transform_page"),
        other => Err(format!("fn {other}").into()),
    }
}

pub fn run(gpu: &'static Gpu, case: &Case) -> Res<()> {
    run_case(gpu, case)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The seed and offset of the case: scalars in the meta, or per-row arrays among the inputs.
fn seeds(case: &Case) -> (u64, u64, Vec<u64>, Vec<u64>) {
    if case.has("seed_arr") {
        let u = |name: &str| case.tensor(name).i64s().into_iter().map(|v| v as u64).collect::<Vec<_>>();
        (0, 0, u("seed_arr"), u("offset_arr"))
    } else {
        (case.i64("seed") as u64, case.i64("offset") as u64, Vec::new(), Vec::new())
    }
}

/// `top_k` and `top_p` (or `min_p`), per row from the arrays or the scalars of the meta, for a batch.
fn params(case: &Case, batch: usize) -> (Vec<i32>, Vec<f32>) {
    if case.has("top_k_arr") {
        (case.tensor("top_k_arr").i32s(), case.tensor("top_p_arr").f32s())
    } else {
        let k = case.get("top_k").map_or(0, |_| case.int("top_k") as i32);
        let p = case.get("top_p").map_or(1.0, |_| case.float("top_p") as f32);
        (vec![k; batch], vec![p; batch])
    }
}

fn problem(case: &Case, probs: Vec<f32>) -> Problem {
    let (batch, d) = (case.int("batch"), case.int("d"));
    let (k, p) = params(case, batch);
    let (seed, offset, seeds, offsets) = seeds(case);
    let mut pr = Problem::from_probs(batch, d, probs, 0, 1.0, seed);
    pr.top_k = k;
    pr.top_p = p;
    pr.offset = offset;
    pr.seeds = seeds;
    pr.offsets = offsets;
    pr
}

fn flag(case: &Case, key: &str) -> bool {
    case.get(key).is_some_and(|v| v != "0")
}

fn sampler(gpu: &'static Gpu, case: &Case, f: &str) -> Res<()> {
    let mode = match f {
        "sample" => Mode::Plain,
        "top_k" => Mode::TopK,
        "top_p" => Mode::TopP,
        "top_k_top_p_joint" => Mode::TopKTopP,
        _ => Mode::MinP,
    };
    let p = problem(case, case.tensor("probs").f32s());
    let cfg = Config::new(mode).per_row(flag(case, "per_row")).per_seed(flag(case, "per_seed"));
    let (out, valid, _) = cached(gpu, |c| run_sampler(c.ctx, &mut c.ks, &p, cfg, false))?;
    if let Some(row) = valid.iter().position(|&v| v == 0) {
        eprintln!("{}: the port marks row {row} invalid", case.name);
    }
    case.write("out", &i32_bytes(&out))
}

fn sample_logits(gpu: &'static Gpu, case: &Case) -> Res<()> {
    let (batch, d) = (case.int("batch"), case.int("d"));
    let (seed, offset, seeds, offsets) = seeds(case);
    let mut p = LogitsProblem::new(batch, d, flashinfer::sampling::Dist::Normal(1.0), seed);
    p.logits = case.tensor("logits").f32s();
    p.offset = offset;
    p.seeds = seeds;
    p.offsets = offsets;
    let cfg = Config::new(Mode::Plain).per_seed(flag(case, "per_seed"));
    let (out, _) = run_logits(&gpu.ctx, cfg, &gpu.tc, &p, false)?;
    case.write("out", &i32_bytes(&out))
}

/// `top_k_top_p_sampling_from_{probs,logits}` in either order. FlashInfer's `top_k_first` with a scalar `k` of at most
/// 256 on a vocabulary of 65536 or more takes a compact path (sorted top-k, then top-p over the k kept); the port's
/// matching path starts from logits, so probabilities are given as their logarithms (its softmax then renormalises).
fn filtered(gpu: &'static Gpu, case: &Case, from_logits: bool) -> Res<()> {
    let (batch, d) = (case.int("batch"), case.int("d"));
    let x = case.tensor("x").f32s();
    let order = if case.str("order") == "joint" { FilterOrder::Joint } else { FilterOrder::TopKFirst };
    let per_row = flag(case, "per_row");
    let k = case.int("top_k");
    let compact = order == FilterOrder::TopKFirst && !per_row && (1..=256).contains(&k) && d >= 65536;
    let x = if compact && !from_logits { x.iter().map(|v| v.ln()).collect() } else { x };
    let p = problem(case, x);
    debug_assert_eq!(p.batch, batch);
    let (out, _valid, _) = cached(gpu, |c| {
        if compact {
            run_top_k_first_compact(c.ctx, &mut c.ks, &mut c.oks, c.tc, &p, false)
        } else {
            run_filtered(c.ctx, &mut c.ks, &mut c.oks, &p, from_logits, order, per_row, false)
        }
    })?;
    case.write("out", &i32_bytes(&out))
}

fn op(gpu: &'static Gpu, case: &Case, f: &str) -> Res<()> {
    let (batch, d) = (case.int("batch"), case.int("d"));
    let per_row = flag(case, "per_row");
    let op = match f {
        "top_p_renorm" => Op::TopPRenorm,
        "top_k_renorm" => Op::TopKRenorm,
        "top_k_mask" => Op::TopKMask,
        "softmax" => Op::Softmax,
        _ => Op::TopK,
    };
    let (top_k, mut param) =
        if f == "topk" { (vec![case.int("k") as i32; batch], vec![0.0; batch]) } else { params(case, batch) };
    if f == "softmax" {
        param = if case.has("temperature") {
            case.tensor("temperature").f32s()
        } else {
            vec![case.float("temperature") as f32; batch]
        };
    }
    let p = OpProblem { batch, d, input: case.tensor("x").f32s(), top_k, param };
    let cfg = OpConfig::for_batch(op, batch).per_row(per_row);
    let (out, indices, _) = cached(gpu, |c| run_op(c.ctx, &mut c.oks, &p, cfg, false))?;
    if op != Op::TopK {
        return case.write_f32("out", &out);
    }
    // FlashInfer's rows are sorted by value, largest first; the port's are not. Values in a row are distinct.
    let k = p.top_k[0] as usize;
    let (mut values, mut idx) = (Vec::with_capacity(out.len()), Vec::with_capacity(out.len()));
    for b in 0..batch {
        let mut row: Vec<(f32, i32)> = (0..k).map(|j| (out[b * k + j], indices[b * k + j])).collect();
        row.sort_by(|x, y| y.0.total_cmp(&x.0));
        values.extend(row.iter().map(|r| r.0));
        idx.extend(row.iter().map(|r| r.1));
    }
    case.write_f32("values", &values)?;
    match case.ref_dtype("indices") {
        "int64" => case.write("indices", &idx.iter().flat_map(|&i| (i as i64).to_le_bytes()).collect::<Vec<_>>()),
        _ => case.write("indices", &i32_bytes(&idx)),
    }
}

fn speculative(gpu: &'static Gpu, case: &Case) -> Res<()> {
    let (batch, n, d) = (case.int("batch"), case.int("n"), case.int("d"));
    let p = spec::Problem {
        batch,
        n,
        d,
        draft_probs: case.tensor("draft_probs").f32s(),
        draft_ids: case.tensor("draft_ids").i32s(),
        target_probs: case.tensor("target_probs").f32s(),
        seed: case.i64("seed") as u64,
        offset: case.i64("offset") as u64,
    };
    // The port's runner starts its counters at 3 and 5, the values the case gave FlashInfer.
    assert!(case.tensor("accepted_init").i32s().iter().all(|&v| v == 3));
    assert!(case.tensor("emitted_init").i32s().iter().all(|&v| v == 5));
    let (out, acc, em, _) = spec::gpu::run(&gpu.ctx, &gpu.tc, &p, 1024, false)?;
    case.write("out", &i32_bytes(&out))?;
    case.write("accepted", &i32_bytes(&acc))?;
    case.write("emitted", &i32_bytes(&em))
}

fn transform(gpu: &'static Gpu, case: &Case, page: bool) -> Res<()> {
    let (rows, stride, k, page_size) = (case.int("rows"), case.int("stride"), case.int("k"), case.int("page_size"));
    let page_shift = page_size.trailing_zeros();
    let kind = if page { Kind::PageTable } else { Kind::Ragged };
    let table = if page { case.tensor("table") } else { case.tensor("lengths") };
    let p = TProblem {
        kind,
        rows,
        stride,
        k,
        input: case.tensor("scores").f32s(),
        lengths: case.tensor("lengths").i32s(),
        row_starts: case.tensor("row_starts").i32s(),
        pt_stride: if page { table.shape[1] } else { 0 },
        table: if page { table.i32s() } else { Vec::new() },
        row_to_batch: if page { case.tensor("row_to_batch").i32s() } else { Vec::new() },
        pt_row_starts: if page { case.tensor("pt_row_starts").i32s() } else { Vec::new() },
        page_shift,
        offsets: if page { Vec::new() } else { case.tensor("offsets").i32s() },
    };
    let max_len = p.lengths.iter().copied().max().unwrap_or(0) as usize;
    let cfg = TConfig::for_rows(kind, rows, max_len).page_shift(page_shift);
    let (mut out, _) = cached(gpu, |c| run_transform(c.ctx, &mut c.tks, &p, cfg, false))?;
    // Both libraries return an unsorted set per row; compare each row in ascending order.
    for row in out.chunks_mut(k) {
        row.sort_unstable();
    }
    case.write("out", &i32_bytes(&out))
}

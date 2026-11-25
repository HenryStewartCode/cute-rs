// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's persistent `BatchAttention` on the simulator: mixed prefill and decode
//! batches over a paged cache, checked against the f64 reference with cute-rs's race and
//! `cp.async` checks; written through and split-KV (the in-kernel persistent reduction);
//! and the planner's FlashInfer semantics.

use flashinfer::attention::*;

fn check(p: &Problem, num_sm: usize) -> Plan {
    let (plan, eo, el) = simulate(p, num_sm);
    assert!(
        eo < 2e-3 && el < 1e-3,
        "chunks {}/{}, merge rows {}: errors {eo} {el}",
        plan.tasks[0].len_kv_chunk,
        plan.tasks[1].len_kv_chunk,
        plan.num_merge()
    );
    plan
}

/// One causal prefill (128-row tiles, GQA 4) next to decodes with partial last pages,
/// page size 16, written through (a grid with room for all of it).
#[test]
fn mixed_batch_written_through() {
    let p = Problem::new(&[1, 40, 1, 1], &[37, 75, 9, 70], 16, 1, 4, true, 3);
    let plan = check(&p, 1);
    assert_eq!(plan.num_merge(), 0);
    assert!(plan.tasks[0].num_works() > 0 && plan.tasks[1].num_works() > 0);
}

/// Many CTAs for little work: every tile longer than the chunk is split, the partial
/// states merged after the grid barrier; two KV heads, causal tiles of both tasks.
#[test]
fn mixed_batch_split() {
    let p = Problem::new(&[1, 48, 1, 3], &[300, 200, 520, 260], 16, 2, 4, true, 7);
    let plan = check(&p, 8);
    assert!(plan.num_merge() > 0, "{plan:?}");
    assert!(plan.tasks[0].split.iter().any(|&s| s) && plan.tasks[1].split.iter().any(|&s| s));
}

/// Not causal, GQA 8 (a 2-query request is still a 16-row decode tile), some requests split
/// and some not in the same task.
#[test]
fn not_causal_partial_split() {
    let p = Problem::new(&[2, 1, 20, 1], &[700, 40, 150, 90], 16, 1, 8, false, 11);
    let plan = check(&p, 4);
    let t1 = &plan.tasks[1];
    assert!(t1.split.iter().any(|&s| s) && t1.split.iter().any(|&s| !s), "{plan:?}");
}

/// The alternate order (task 1 first in the second half of the grid), split and merged.
#[test]
fn alternate_order() {
    let p = Problem::new(&[1, 48, 1, 3], &[300, 200, 520, 260], 16, 2, 4, true, 5);
    let plan = p.plan(8);
    let spec = AttentionSpec { alternate: true, ..p.spec(&plan) };
    let (o, lse) = execute_spec(&p, &plan, &spec);
    let (want_o, want_lse) = p.reference();
    let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
    assert!(eo < 2e-3 && el < 1e-3, "errors {eo} {el}");
}

/// The planner's chunk lengths, as `TwoStageHolisticPlan` computes them: the total KV per
/// CTA (causal-trimmed) times the KV heads, rounded up to 128 / a multiple of 256, halved
/// for the 128-row task.
#[test]
fn planner_chunks() {
    // 128 decodes of 8192 + one 4096-query chunk over 8192, 8 KV heads, 22 SMs (44 CTAs).
    let mut qo = vec![1usize; 128];
    let mut kv = vec![8192usize; 128];
    qo.push(4096);
    kv.push(8192);
    let plan = Plan::new(&qo, &kv, 4, 8, true, 22);
    // Prefill tiles: 128 of 128 packed rows (32 queries), causal ends 4096 + 32 (t + 1).
    let prefill: i64 = (0..128).map(|t| if t == 127 { 8192 } else { 4096 + 32 * (t + 1) }).sum();
    let total = 128 * 8192 + prefill;
    let per = (total * 8 + 43) / 44;
    let f = (per + 255) / 256 * 256;
    assert_eq!(plan.tasks[1].len_kv_chunk as i64, f);
    assert_eq!(plan.tasks[0].len_kv_chunk as i64, f / 2);
    assert_eq!(plan.tasks[0].work_indptr.len(), 45);
}

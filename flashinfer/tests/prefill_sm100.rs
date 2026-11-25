// SPDX-License-Identifier: Apache-2.0

//! FlashInfer's Blackwell (SM100) ragged prefill on cute-rs's simulator (which models
//! tcgen05 MMAs, TMEM loads/stores and commits, and checks for races) against the f64
//! reference, and its host-side work plan.

use flashinfer::prefill_sm100::{Plan, Problem, TILE_Q, simulate};

fn check(p: &Problem, num_ctas: usize) {
    let (eo, el) = simulate(p, num_ctas);
    let tol = if p.bf16 { 1e-2 } else { 2e-3 };
    assert!(
        eo < tol && el < 1e-4,
        "qo {:?} kv {:?} heads {}/{} causal {} bf16 {}: max |o - ref| {eo:.3e}, |lse - ref| {el:.3e}",
        p.qo_lens,
        p.kv_lens,
        p.num_qo_heads,
        p.num_kv_heads,
        p.causal,
        p.bf16
    );
}

#[test]
fn varlen_full_f16() {
    // A ragged K/V tail (200 = 128 + 72) and a request shorter than a Q half.
    check(&Problem::new(&[300, 70], &[300, 200], 2, 1, false, false, 1), 3);
}

#[test]
fn varlen_causal_f16() {
    // qo_len < kv_len: the causal diagonal is offset by kv_len - qo_len.
    check(&Problem::new(&[300, 70], &[300, 200], 2, 1, true, false, 2), 3);
}

#[test]
fn varlen_bf16_full_and_causal() {
    check(&Problem::new(&[129, 260], &[257, 260], 2, 2, false, true, 3), 2);
    check(&Problem::new(&[129, 260], &[257, 260], 2, 2, true, true, 4), 2);
}

#[test]
fn tile_multiples_and_gqa() {
    // Whole tiles (no residual mask) and four query heads per K/V head, one CTA for all.
    check(&Problem::new(&[256], &[384], 4, 1, false, false, 5), 1);
    check(&Problem::new(&[256], &[256], 4, 1, true, false, 6), 1);
}

#[test]
fn more_ctas_than_work() {
    // Idle CTAs (an empty work range) next to busy ones.
    check(&Problem::new(&[100], &[150], 1, 1, true, false, 7), 4);
}

#[test]
fn plan_covers_every_tile_once_and_balances() {
    let (qo, kv) = ([1000usize, 77, 4096, 513], [1000usize, 300, 4096, 1700]);
    for causal in [false, true] {
        let plan = Plan::new(&qo, &kv, 32, 148, causal);
        assert_eq!(plan.num_ctas(), 148);
        let mut seen: Vec<(i32, i32, i32)> = (0..plan.work_items())
            .map(|i| (plan.qo_tile_indices[i], plan.head_indices[i], plan.batch_indices[i]))
            .collect();
        seen.sort();
        let mut want = vec![];
        for (b, &q) in qo.iter().enumerate() {
            for h in 0..32 {
                for t in 0..q.div_ceil(TILE_Q) {
                    want.push((t as i32, h, b as i32));
                }
            }
        }
        want.sort();
        assert_eq!(seen, want);
        // The greedy plan: no CTA's cost exceeds the least loaded by more than one item.
        let cost = |c: usize| -> usize {
            (plan.work_indptr[c]..plan.work_indptr[c + 1])
                .map(|w| {
                    let (t, b) = (plan.qo_tile_indices[w as usize] as usize, plan.batch_indices[w as usize] as usize);
                    let tiles = qo[b].div_ceil(TILE_Q);
                    if causal { kv[b] - (tiles - t - 1) * TILE_Q } else { kv[b] }
                })
                .sum()
        };
        let costs: Vec<usize> = (0..148).map(cost).collect();
        let (lo, hi) = (*costs.iter().min().unwrap(), *costs.iter().max().unwrap());
        assert!(hi - lo <= 4096, "unbalanced plan: {lo}..{hi}");
    }
}

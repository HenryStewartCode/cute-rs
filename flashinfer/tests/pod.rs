// SPDX-License-Identifier: Apache-2.0

//! POD attention on the simulator: a prefill and a decode batch in one launch, both halves
//! checked against the f64 reference with cute-rs's race and `cp.async` checks, whole and
//! split-KV (with the merges), and the scheduler's clean-up of its counters.

use flashinfer::pod::*;

fn check(p: &Problem, grid: usize, min_chunk: usize) -> Plans {
    let (plans, e) = simulate(p, grid, min_chunk);
    assert!(
        e[0] < 2e-3 && e[1] < 1e-3 && e[2] < 2e-3 && e[3] < 1e-3,
        "tiles {}/{} split {}/{}: errors {e:?}",
        plans.prefill.tile.cta_tile_q,
        plans.decode.tile.cta_tile_q,
        plans.prefill.split,
        plans.decode.split
    );
    plans
}

/// A causal 128-row-tile prefill with GQA 4 next to a decode batch with partial last
/// pages, unsplit (a one-wave grid).
#[test]
fn causal_prefill_with_decodes() {
    let p = Problem::new(40, 75, &[9, 37, 70, 16], 8, 1, 4, true, 3);
    let plans = check(&p, 1, 128);
    assert_eq!(plans.prefill.tile.cta_tile_q, 128);
    assert_eq!(plans.decode.tile.cta_tile_q, 16);
    assert!(!plans.prefill.split && !plans.decode.split);
}

/// Both halves split over KV chunks (a large grid), merged afterwards; non-causal
/// 16-row-tile prefill, two KV heads.
#[test]
fn split_halves() {
    let p = Problem::new(3, 600, &[300, 5, 130], 4, 2, 4, false, 7);
    let plans = check(&p, 1000, 1);
    assert!(plans.prefill.split && plans.decode.split, "{plans:?}");
}

/// More decode CTAs than prefill CTAs and the converse (both scheduler branches), and
/// the scheduler buffer cleared by the last CTA.
#[test]
fn scheduler_ratios_and_reset() {
    for (qo, decodes) in [(8usize, vec![20usize; 9]), (64, vec![33])] {
        let p = Problem::new(qo, qo + 10, &decodes, 16, 1, 4, true, 11);
        let plans = p.plans(1);
        let mut args = p.args(&plans);
        kernel(&p.spec(&plans)).simulate(p.launch(&plans), &mut args).unwrap();
        assert!(args[p::SCHED].as_i32().iter().all(|&x| x == 0), "scheduler buffer not cleared");
        let (cp, cd) = plans.ctas();
        assert!(if qo == 8 { cp < cd } else { cp > cd }, "{cp} {cd}");
        let out = execute(&p, &plans);
        let e = errors(&p, &out);
        assert!(e[0] < 2e-3 && e[2] < 2e-3, "{e:?}");
    }
}

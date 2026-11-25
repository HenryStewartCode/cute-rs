// SPDX-License-Identifier: Apache-2.0

//! MLA paged attention on the simulator against the f64 reference, with cute-rs's race and
//! `cp.async` checks (the simulator runs the cooperative grid resident, so the grid
//! barrier and the merge run too); the planner against FlashInfer's rules; and, when a
//! toolchain is found, the PTX it lowers to.

use cute::prelude::*;
use flashinfer::mla::*;

fn check(qo: &[usize], kv: &[usize], page: usize, heads: usize, causal: bool, num_sm: usize, split: bool) {
    let p = Problem::new(qo, kv, page, heads, causal, 5);
    let (plan, eo, el) = simulate(&p, num_sm);
    let what = format!("qo {qo:?} kv {kv:?} page {page} heads {heads} causal {causal} sms {num_sm}");
    assert_eq!(plan.split(), split, "{what}: {plan:?}");
    assert!(eo < 2e-3 && el < 1e-3, "{what}: |o| {eo:.2e}, |lse| {el:.2e}");
}

/// Decode (one query per request), 16 heads: single tokens, partial and whole pages.
#[test]
fn decode_16_heads() {
    check(&[1; 4], &[1, 7, 21, 40], 4, 16, false, 8, false);
}

/// Decode with 128 heads: two CTAs per cluster share each request's KV.
#[test]
fn decode_128_heads() {
    check(&[1, 1], &[5, 33], 16, 128, false, 4, false);
}

/// A short causal prefill: ragged queries, a second query tile, partial pages.
#[test]
fn prefill_causal() {
    check(&[3, 5], &[10, 37], 4, 16, true, 8, false);
}

/// Split KV: one long request cut into chunks on several clusters, its rows merged by
/// several merge CTAs after the grid barrier; a short request stays whole.
#[test]
fn decode_split_and_merge() {
    let p = Problem::new(&[1, 1], &[1024, 9], 16, 16, false, 5);
    let plan = p.plan(8);
    let merge_ctas = plan.merge.chunks(MERGE_FIELDS).filter(|m| m[1] > m[0]).count();
    assert!(merge_ctas >= 2, "{plan:?}");
    check(&[1, 1], &[1024, 9], 16, 16, false, 8, true);
}

/// Causal prefill with a split chunk.
#[test]
fn prefill_causal_split() {
    check(&[4], &[150], 4, 16, true, 8, true);
}

/// The planner: clusters of two above 64 packed rows, a whole wave of clusters, every
/// tile's KV covered exactly once, and merge rows covering the split tiles.
#[test]
fn plan_follows_flashinfer() {
    let qo_indptr = [0, 1, 2, 3];
    let kv_indptr = [0, 40, 80, 120];
    let plan = Plan::new(&qo_indptr, &kv_indptr, &[4096, 300, 5], 128, false, 22);
    assert_eq!((plan.cluster_size, plan.num_clusters), (2, 11));
    assert_eq!(plan.work_indptr.len(), 12);
    // total 4401 / 11 clusters = 401 -> 512.
    assert_eq!(plan.kv_len_limit, 512);
    let mut covered = [0usize; 3];
    for w in plan.work.chunks(WORK_FIELDS) {
        let b = w[0] as usize;
        covered[b] += (w[7] - w[6]) as usize;
        assert_eq!(w[2] >= 0, b == 0, "{w:?}");
    }
    assert_eq!(covered, [4096, 300, 5]);
    assert_eq!(plan.partial_rows, 8 * 128);
    let merged: i32 = plan.merge.chunks(MERGE_FIELDS).map(|m| m[1] - m[0]).sum();
    assert_eq!(merged, 128);
    let small = Plan::new(&[0, 1, 2], &[0, 1, 2], &[10, 20], 16, false, 22);
    assert_eq!((small.cluster_size, small.num_clusters), (1, 22));
}

/// The kernel lowers to 16-byte `cp.async`, `ldmatrix` (plain and transposed), f16 MMAs
/// with f32 accumulators and `ex2.approx.ftz`.
#[test]
fn lowers_to_mma_and_ldmatrix() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let spec = KernelSpec { num_heads: 128, page_size: 1, causal: false };
    let compiled = kernel(&spec).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap();
    let ptx = compiled.ptx();
    assert!(ptx.contains("cp.async.cg.shared.global"), "{ptx}");
    assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"), "{ptx}");
    assert!(ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count() > 72 + 32, "{ptx}");
    assert!(ptx.contains("ex2.approx.ftz.f32"), "{ptx}");
}

// SPDX-License-Identifier: Apache-2.0

//! Block-sparse attention on the simulator, against the f64 reference with cute-rs's race
//! and `cp.async` checks: BSR blocks of several `R x C` (head dim 64 and 128, GQA, a custom
//! mask, empty block rows), whole and split-KV; and FlashInfer's variable block sizes,
//! per-head sizes and patterns.

use flashinfer::sparse::*;

/// Simulate `p` whole (one CTA per KV head, never split) and split (a large grid with
/// one-page chunks, merged); returns the query tile.
fn check(p: &Problem, what: &str) -> usize {
    let mut tile = 0;
    for (grid, split) in [(p.num_kv_heads, false), (4000, true)] {
        let (plan, eo, el) = simulate(p, grid, 1);
        assert_eq!(plan.split, split, "{what}: {plan:?}");
        assert!(
            eo < 2e-3 && el < 1e-3,
            "{what}: tile {} split {}: |o| {eo:.2e}, |lse| {el:.2e}",
            plan.tile.cta_tile_q,
            plan.split
        );
        tile = plan.tile.cta_tile_q;
    }
    tile
}

/// BSR, head dim 128: 16 x 16 blocks (64-row tile with GQA 4), 128 x 8 (128-row tile, 1:1),
/// 1 x 4 rows (the 16-row tile).
#[test]
fn bsr_head_dim_128() {
    assert_eq!(check(&Problem::random_bsr((64, 96, 16, 16), 0.5, 4, 1, 128, false, 1), "16x16 g4"), 64);
    assert_eq!(check(&Problem::random_bsr((256, 64, 128, 8), 0.5, 1, 1, 128, false, 2), "128x8 g1"), 128);
    assert_eq!(check(&Problem::random_bsr((8, 64, 1, 4), 0.6, 2, 1, 128, false, 3), "1x4 g2"), 16);
}

/// BSR, head dim 64 (FlashInfer's tiles for 64: KV tiles of 64 / 128 / 128 tokens), each
/// query tile.
#[test]
fn bsr_head_dim_64() {
    assert_eq!(tile(128, 64).cta_tile_kv(), 64);
    assert_eq!(tile(64, 64).cta_tile_kv(), 128);
    assert_eq!(tile(16, 64).cta_tile_kv(), 128);
    assert_eq!(check(&Problem::random_bsr((128, 160, 64, 16), 0.5, 2, 1, 64, false, 4), "64x16 g2"), 128);
    assert_eq!(check(&Problem::random_bsr((64, 256, 16, 32), 0.4, 4, 1, 64, false, 5), "16x32 g4"), 64);
    assert_eq!(check(&Problem::random_bsr((8, 64, 2, 1), 0.5, 2, 2, 64, false, 6), "2x1 g1"), 16);
}

/// A custom mask on top of the blocks (FlashInfer's `[nnz, R, C]` mask), both head dims.
#[test]
fn bsr_custom_mask() {
    for hd in [64, 128] {
        let p = Problem::random_bsr((64, 96, 16, 16), 0.5, 4, 2, hd, true, 7);
        assert!(p.has_mask());
        check(&p, &format!("mask d{hd}"));
    }
}

/// Block rows without any block: `o = 0`, `lse = -inf` (also through the merge).
#[test]
fn bsr_empty_rows() {
    let (r, c) = (16, 16);
    let indptr = [0, 2, 2, 3, 3];
    let indices = [0, 3, 1];
    let mut p = Problem::random_bsr((64, 64, r, c), 1.0, 4, 1, 128, false, 8);
    p = Problem::from_bsr((64, 64, r, c), &indptr, &indices, 4, 1, 128, (p.q, p.k, p.v), None);
    assert_eq!(p.kv_lens, vec![32, 0, 16, 0]);
    check(&p, "empty rows");
    let (_, rl) = p.reference();
    assert!(rl[16 * 4..32 * 4].iter().all(|x| *x == f32::NEG_INFINITY));
}

/// The BSR page table is FlashInfer's: block rows are requests of `R` queries over their
/// block columns as full pages of `C` tokens.
#[test]
fn bsr_maps_to_pages() {
    let base = Problem::random_bsr((8, 16, 4, 4), 1.0, 1, 1, 64, false, 9);
    let p = Problem::from_bsr((8, 16, 4, 4), &[0, 1, 3], &[2, 0, 3], 1, 1, 64, (base.q, base.k, base.v), None);
    assert_eq!(p.qo_indptr, vec![0, 4, 8]);
    assert_eq!((p.page_size, p.kv_lens.clone()), (4, vec![4, 8]));
    assert_eq!(p.kv_indptr, vec![1, 2, 4]);
    assert_eq!(&p.kv_indices[1..4], &[2, 0, 3]);
    assert_eq!(p.kv_last_page_len, vec![4, 4]);
}

/// FlashInfer's variable block sizes: equal sizes with a remainder, then per-head sizes and
/// patterns, head dims 64 and 128, GQA.
#[test]
fn variable_blocks() {
    for (seq, blocks, hq, hkv, hd, jitter) in [
        (100usize, (3usize, 5usize), 2usize, 2usize, 128usize, false),
        (96, (4, 6), 4, 2, 64, true),
        (130, (2, 7), 4, 1, 128, true),
    ] {
        let p = Problem::random_variable(seq, blocks, 0.5, hq, hkv, hd, jitter, 10);
        assert_eq!((p.num_kv_heads, p.page_size), (1, 1));
        check(&p, &format!("variable seq {seq} blocks {blocks:?} heads {hq}/{hkv} d{hd} jitter {jitter}"));
    }
}

/// FlashInfer's docstring example: block rows `[1, 2, 3]`, columns `[3, 1, 2]`, mask
/// `[[0, 0, 1], [1, 0, 1], [0, 1, 1]]`; the output comes back in `[num_qo_heads, qo_len,
/// D]`.
#[test]
fn variable_docstring_example() {
    let map = [false, false, true, true, false, true, false, true, true];
    let base = Problem::random_variable(6, (1, 1), 1.0, 1, 1, 128, false, 11);
    let p = Problem::from_variable_blocks(&map, &[1, 2, 3], &[3, 1, 2], 1, 1, 128, (base.q, base.k, base.v));
    assert_eq!(p.kv_lens, vec![2, 5, 3]);
    assert_eq!(&p.kv_indices[1..], &[4, 5, 0, 1, 2, 4, 5, 3, 4, 5, 0]);
    check(&p, "docstring");
    let (o, _) = p.reference();
    assert_eq!(p.output_layout(&o), o);
}

/// The variable-block GQA layout round-trips: `[num_qo_heads, qo_len, D]` in,
/// `[(num_kv_heads qo_len), group, D]` for the kernel, and back.
#[test]
fn variable_layout_round_trip() {
    let p = Problem::random_variable(10, (2, 2), 1.0, 4, 2, 64, false, 12);
    let q: Vec<f32> = p.q.iter().map(|x| x.to_f32()).collect();
    let back = p.output_layout(&q);
    // Query head 3 (KV head 1, group index 1), token 7: HND row 3 * 10 + 7, kernel row
    // (1 * 10 + 7) * 2 + 1.
    assert_eq!(back.len(), q.len());
    assert_eq!(back[(3 * 10 + 7) * 64], q[((10 + 7) * 2 + 1) * 64]);
}

/// The kernel lowers to the prefill kernel's instructions for head dim 64.
#[test]
fn lowers_to_mma_ldmatrix_cp_async() {
    use cute::prelude::*;
    let Some(tc) = Toolchain::discover() else { return };
    let p = Problem::random_bsr((128, 128, 64, 16), 0.5, 4, 1, 64, false, 13);
    let ptx = kernel(&p.spec(&p.plan(44)))
        .compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89))
        .unwrap_or_else(|e| panic!("{e}"))
        .ptx()
        .to_string();
    for needle in [
        "mma.sync.aligned.m16n8k16",
        "ldmatrix.sync.aligned.m8n8.x4.trans",
        "cp.async.cg.shared.global",
        "ex2.approx.ftz",
    ] {
        assert!(ptx.contains(needle), "no {needle}");
    }
}

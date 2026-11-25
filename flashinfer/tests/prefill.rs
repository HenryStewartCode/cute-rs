// SPDX-License-Identifier: Apache-2.0

//! Batch prefill on the simulator: each query tile size, causal and not, GQA and 1:1,
//! whole requests and split-KV with the merge, against the f64 reference, with cute-rs's
//! race and `cp.async` checks; the planner against FlashInfer's rules; and, when a
//! toolchain is found, the PTX the kernel lowers to.

use cute::prelude::*;
use flashinfer::prefill::*;

/// Simulate and check one problem, whole (a one-wave grid of one CTA per KV head, which
/// never splits) and split (a large grid with 1-page chunks, merged by the decode port's
/// merge kernel); returns the tile size.
fn check(qo: &[usize], kv: &[usize], page: usize, heads: usize, group: usize, causal: bool) -> usize {
    let p = Problem::new(qo, kv, page, heads, group, causal, 5);
    let mut tile = 0;
    for (grid, split) in [(heads, false), (1000, true)] {
        let (plan, eo, el) = simulate(&p, grid, 1);
        assert_eq!(plan.split, split, "{plan:?}");
        assert!(
            eo < 2e-3 && el < 1e-3,
            "qo {qo:?} kv {kv:?} page {page} group {group} causal {causal} tile {} split {}: |o| {eo:.2e}, |lse| {el:.2e}",
            plan.tile.cta_tile_q,
            plan.split
        );
        tile = plan.tile.cta_tile_q;
    }
    tile
}

/// The 64-row tile (4 x 1 warps, 64-token KV tiles): ragged requests with partial last
/// pages, GQA 4, causal and not.
#[test]
fn tile64_gqa() {
    for causal in [false, true] {
        assert_eq!(check(&[9, 13], &[37, 13], 4, 1, 4, causal), 64);
    }
}

/// The 128-row tile (4 x 1 warps, 32-token KV tiles, two MMA rows per warp), 1:1 heads
/// and GQA, causal and not, with requests spanning several query tiles.
#[test]
fn tile128() {
    for (group, causal) in [(1, true), (4, false), (4, true)] {
        let qo = if group == 1 { vec![150usize, 70] } else { vec![40, 33] };
        assert_eq!(check(&qo, &[160, 75], 16, 1, group, causal), 128);
    }
}

/// The 16-row tile (1 x 4 warps splitting each 64-token KV tile, merged at the end), two
/// KV heads.
#[test]
fn tile16_kv_warps() {
    for causal in [false, true] {
        assert_eq!(check(&[3, 1, 4], &[70, 9, 130], 8, 2, 4, causal), 16);
    }
}

/// FlashInfer's planner rules: the query tile from the average packed query length, no
/// split when the query tiles fill the GPU, and otherwise chunks whose CTAs fit.
#[test]
fn plan_follows_flashinfer() {
    let whole = Plan::new(&[512; 8], &[32; 8], 16, 4, 8, 44);
    assert!(!whole.split);
    assert_eq!(whole.tile.cta_tile_q, 128);
    assert_eq!(whole.num_ctas(), 8 * 16);
    assert_eq!(whole.tile.cta_tile_kv(), 32);
    let short = Plan::new(&[1], &[256], 16, 4, 8, 44);
    assert_eq!(short.tile.cta_tile_q, 16);
    assert!(short.split);
    assert!(short.num_ctas() <= 44 / 8, "{short:?}");
    assert_eq!(short.merge_indptr, vec![0, short.num_ctas() as i32]);
    assert_eq!(TileConfig::sm89(64).cta_tile_kv(), 64);
    assert_eq!(TileConfig::sm89(16).cta_tile_kv(), 64);
}

/// The kernel lowers to the instructions it is built around: `mma.sync m16n8k16`,
/// `ldmatrix` (plain and transposed), 16-byte `cp.async` and `ex2.approx.ftz`.
#[test]
fn lowers_to_mma_ldmatrix_cp_async() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let spec = KernelSpec {
        tile: TileConfig::sm89(128),
        group: 4,
        page_size: 16,
        causal: true,
        partition: false,
        variant: Variant::NONE,
    };
    let ptx = kernel(&spec).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
    // Q K^T and P V: 2 x 2 x 8 x 2 MMAs each, and 4 row-sum MMAs, per iteration.
    assert_eq!(ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count(), 132, "{ptx}");
    assert_eq!(ptx.matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16").count(), 16);
    assert_eq!(ptx.matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16").count(), 32);
    assert!(ptx.contains("cp.async.cg.shared.global"));
    assert!(ptx.contains("ex2.approx.ftz.f32"));
}

/// Ragged K/V (`BatchPrefillWithRaggedKVCacheWrapper`): each query tile size, causal and
/// not, GQA and 1:1, ragged lengths, whole and split, against the f64 reference.
#[test]
fn ragged_matches_reference() {
    for (qo, kv, heads, group, tile) in [
        (vec![3usize, 1, 4], vec![70usize, 9, 130], 2usize, 4usize, 16usize),
        (vec![9, 13], vec![37, 13], 1, 4, 64),
        (vec![150, 70], vec![160, 75], 1, 1, 128),
    ] {
        for causal in [false, true] {
            let p = Problem::new_ragged(&qo, &kv, heads, group, causal, 5);
            assert!(p.is_ragged() && p.spec(&p.plan(heads)).name().contains("ragged"));
            for (grid, split) in [(heads, false), (1000, true)] {
                let (plan, eo, el) = simulate(&p, grid, 16);
                assert_eq!((plan.split, plan.tile.cta_tile_q), (split, tile), "{plan:?}");
                assert!(
                    eo < 2e-3 && el < 1e-3,
                    "qo {qo:?} kv {kv:?} causal {causal} split {split}: |o| {eo:.2e}, |lse| {el:.2e}"
                );
            }
        }
    }
}

/// `single_prefill_with_kv_cache`: one request, contiguous K/V, whole and split by
/// FlashInfer's single-prefill chunk rule, causal and not, GQA 4 and 1:1.
#[test]
fn single_prefill_matches_reference() {
    for (qo, kv, heads, group, grid) in
        [(5usize, 300usize, 2usize, 4usize, 1000usize), (40, 40, 1, 1, 4), (20, 520, 1, 4, 64)]
    {
        for causal in [false, true] {
            let p = Problem::new_ragged(&[qo], &[kv], heads, group, causal, 9);
            let plan = p.single_plan(grid);
            assert_eq!(plan.split, kv > 256, "{plan:?}");
            let (o, lse) = single_prefill_with_kv_cache(
                p.q.clone(),
                p.k_cache.clone(),
                p.v_cache.clone(),
                heads * group,
                heads,
                causal,
                grid,
            );
            let (want_o, want_lse) = p.reference();
            let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
            assert!(eo < 2e-3 && el < 1e-3, "qo {qo} kv {kv} causal {causal}: |o| {eo:.2e}, |lse| {el:.2e}");
        }
    }
}

/// FlashInfer's single-prefill chunk rule: `max(ceil(kv_len / max_chunks), 256)` tokens,
/// `max_chunks = grid / (kv_heads * query tiles)`.
#[test]
fn single_plan_follows_flashinfer() {
    // 1024 queries x 4 heads at tile 128: 32 tiles x 8 heads fill a 44-CTA grid.
    assert!(!Plan::single(1024, 1024, 4, 8, 44).split);
    let p = Plan::single(1, 8192, 4, 8, 44);
    assert_eq!((p.tile.cta_tile_q, p.kv_chunk_size, p.num_ctas()), (16, 1639, 5));
    assert_eq!(p.merge_indptr, vec![0, 5]);
    assert_eq!(Plan::single(1, 1000, 4, 8, 440).kv_chunk_size, 256);
}

/// FlashInfer's attention variants on every tile size (16, 64, 128 rows), causal and not,
/// GQA, whole and split, paged and ragged, against the f64 reference: soft-cap, sliding
/// window (with tiles left of the window skipped), custom mask, ALiBi, and combinations.
#[test]
fn variants_match_reference() {
    type Make = fn(Problem) -> Problem;
    let cases: [(&str, Make); 7] = [
        ("cap", |p| p.with_soft_cap(4.0)),
        ("win", |p| p.with_window(20)),
        ("mask", |p| p.with_custom_mask(3)),
        ("alibi", |p| p.with_alibi()),
        ("cap + win", |p| p.with_soft_cap(2.0).with_window(40)),
        ("mask + win + cap", |p| p.with_custom_mask(9).with_window(33).with_soft_cap(3.0)),
        ("alibi + win off", |p| p.with_alibi().with_window(-1)),
    ];
    // (qo, kv, page, heads, group): tiles 16, 64 and 128.
    type Shape = (&'static [usize], &'static [usize], usize, usize, usize);
    let shapes: [Shape; 3] =
        [(&[3, 1, 4], &[70, 9, 130], 8, 2, 4), (&[9, 13], &[150, 13], 4, 1, 4), (&[90, 70], &[160, 75], 16, 1, 1)];
    for (name, make) in cases {
        for &(qo, kv, page, heads, group) in &shapes {
            for causal in [false, true] {
                let p = make(Problem::new(qo, kv, page, heads, group, causal, 7));
                for grid in [heads, 1000] {
                    let (plan, eo, el) = simulate(&p, grid, 1);
                    assert!(
                        eo < 2e-3 && el < 1e-3,
                        "{name} qo {qo:?} causal {causal} tile {} split {}: |o| {eo:.2e}, |lse| {el:.2e}",
                        plan.tile.cta_tile_q,
                        plan.split
                    );
                }
            }
        }
        let p = make(Problem::new_ragged(&[9, 13], &[150, 13], 1, 4, true, 3));
        let (_, eo, el) = simulate(&p, 1000, 1);
        assert!(eo < 2e-3 && el < 1e-3, "ragged {name}: |o| {eo:.2e}, |lse| {el:.2e}");
    }
}

/// A causal window plans only `window_left + CTA_TILE_Q` tokens per request, and the
/// variant kernels lower to `tanh.approx`.
#[test]
fn window_plan_and_variant_ptx() {
    let p = Problem::new(&[2048], &[2048], 16, 8, 4, true, 1).with_window(1024);
    let plan = p.plan(44);
    let narrow = Problem::new(&[2048], &[1024 + 128], 16, 8, 4, true, 1).plan(44);
    assert_eq!(plan.num_ctas(), narrow.num_ctas());
    let Some(tc) = Toolchain::discover() else { return };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    let variant = Variant { logits_soft_cap: true, sliding_window: true, custom_mask: true, alibi: false };
    let spec =
        KernelSpec { tile: TileConfig::sm89(128), group: 4, page_size: 16, causal: true, partition: true, variant };
    let ptx = kernel(&spec).compile_with(&tc, &opts).unwrap().ptx().to_string();
    assert!(ptx.contains("tanh.approx.f32"));
}

/// FP8 K/V (E4M3 and E5M2, with `k_scale`/`v_scale`) with Float16 queries: every query tile
/// size, paged and ragged, causal and not, whole and split, against the f64 reference on the
/// dequantised cache (FlashInfer's dequantise-after-`ldmatrix` kernels).
#[test]
fn fp8_kv_matches_reference() {
    for kv in [KvDtype::E4M3, KvDtype::E5M2] {
        for (qo, kv_lens, page, heads, group, tile) in [
            (vec![3usize, 1, 4], vec![70usize, 9, 130], 8usize, 2usize, 4usize, 16usize),
            (vec![9, 13], vec![37, 13], 4, 1, 4, 64),
            (vec![150, 70], vec![160, 75], 16, 1, 1, 128),
        ] {
            for causal in [false, true] {
                for ragged in [false, true] {
                    let p = if ragged {
                        Problem::new_ragged(&qo, &kv_lens, heads, group, causal, 5)
                    } else {
                        Problem::new(&qo, &kv_lens, page, heads, group, causal, 5)
                    };
                    let p = p.quantized(kv, 0.5, 1.5);
                    assert!(p.spec(&p.plan(heads)).name().contains(if kv == KvDtype::E4M3 { "e4m3" } else { "e5m2" }));
                    for (grid, split) in [(heads, false), (1000, true)] {
                        let (plan, eo, el) = simulate(&p, grid, if ragged { 16 } else { 1 });
                        assert_eq!((plan.split, plan.tile.cta_tile_q), (split, tile), "{plan:?}");
                        assert!(
                            eo < 2e-3 && el < 1e-3,
                            "{kv:?} qo {qo:?} kv {kv_lens:?} ragged {ragged} causal {causal} split {split}: |o| {eo:.2e}, |lse| {el:.2e}"
                        );
                    }
                }
            }
        }
    }
}

/// FP8 halves the K and V tiles, so FlashInfer's shared-memory rule picks longer KV tiles.
#[test]
fn fp8_tiles_are_longer() {
    for (q, f16, fp8) in [(128, 32, 64), (64, 64, 128), (16, 64, 128)] {
        assert_eq!(TileConfig::sm89(q).cta_tile_kv(), f16);
        assert_eq!(TileConfig::sm89_kv(q, KvDtype::E4M3).cta_tile_kv(), fp8);
        assert!(TileConfig::sm89_kv(q, KvDtype::E5M2).smem_bytes() <= 101376 / 2 + 1);
    }
}

/// The FP8 kernel keeps the MMA/`ldmatrix` structure and widens with `cvt.rn.f16x2`.
#[test]
fn fp8_lowers_to_cvt() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let p = Problem::new(&[9], &[37], 16, 1, 4, true, 5).quantized(KvDtype::E4M3, 1.0, 1.0);
    let spec = p.spec(&p.plan(1));
    let ptx = kernel(&spec).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
    assert!(ptx.contains("cvt.rn.f16x2.e4m3x2"), "{ptx}");
    assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"));
    assert!(ptx.contains("cp.async.cg.shared.global"));
}

/// Head dimensions 64 and 256 and BFloat16 queries/K/V/output (FlashInfer's fa2 prefill
/// instantiations), paged and ragged, causal and not, whole and split (so the merge runs
/// too), against the f64 reference on the BFloat16-rounded inputs (the output is rounded to
/// BFloat16, so its tolerance is wider). Head dimension 128 in Float16 is the original path.
#[test]
fn head_dims_and_bf16_match_reference() {
    for hd in [64usize, 128, 256] {
        for bf16 in [false, true] {
            if hd == 128 && !bf16 {
                continue;
            }
            let (to, tl) = if bf16 { (8e-3, 4e-3) } else { (4e-3, 2e-3) };
            let make = |p: Problem| if bf16 { p.bf16() } else { p };
            // A decode-like and a prefill-like batch (the 16- and 64/128-row query tiles).
            for (qo, kv, group, causal) in
                [(vec![1usize, 2, 3], vec![9usize, 40, 21], 4usize, false), (vec![20, 33], vec![37, 50], 2, true)]
            {
                let p = make(Problem::new_hd(&qo, &kv, 4, 2, group, causal, hd, 7));
                for (max_grid, min_chunk) in [(1000, 32), (6, 1)] {
                    let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
                    assert!(
                        eo < to && el < tl,
                        "hd {hd} bf16 {bf16} qo {qo:?} split {} tile {}: |o| {eo:.2e}, |lse| {el:.2e}",
                        plan.split,
                        plan.tile.cta_tile_q
                    );
                }
            }
            let p = make(Problem::new_ragged_hd(&[17, 5], &[30, 22], 1, 4, true, hd, 3)).with_soft_cap(5.0);
            let (_, eo, el) = simulate(&p, 6, 1);
            assert!(eo < to && el < tl, "ragged hd {hd} bf16 {bf16} cap: |o| {eo:.2e}, |lse| {el:.2e}");
        }
    }
}

/// Head dimension 256 never plans the 128-row tile; the new shapes keep distinct kernel
/// names, and the original keeps its own (so its PTX is unchanged).
#[test]
fn head_dim_planning_and_names() {
    assert_eq!(fa2_cta_tile_q_hd(500, 128), 128);
    assert_eq!(fa2_cta_tile_q_hd(500, 256), 64);
    assert_eq!(fa2_cta_tile_q_hd(10, 256), 16);
    let p = Problem::new_hd(&[300], &[300], 16, 1, 1, true, 256, 1);
    assert_eq!(p.plan(44).tile.cta_tile_q, 64);
    let spec = p.spec(&p.plan(44));
    assert_eq!(kernel_shape(&spec, 256, true).name(), format!("{}_hd256_bf16", spec.name()));
    let q = Problem::new(&[300], &[300], 16, 1, 1, true, 1);
    let qs = q.spec(&q.plan(44));
    assert_eq!(kernel(&qs).name(), qs.name());
}

// SPDX-License-Identifier: Apache-2.0

//! Batch decode on the simulator: every path (whole requests and split-KV with the merge)
//! against the f64 reference, with cute-rs's race and `cp.async` checks; and, when a
//! toolchain is found, the PTX it lowers to.

use cute::prelude::*;
use flashinfer::decode::*;

/// Lengths cover a single token, partial and exact last pages, and longer requests; each
/// problem runs whole (a large grid) and split (a small grid with 1-page chunks).
#[test]
fn matches_reference_whole_and_split() {
    for (lens, page_size, heads) in [(vec![1usize, 7, 21, 40], 4usize, 2usize), (vec![16, 3, 33], 16, 1)] {
        let p = Problem::new(&lens, page_size, heads, 7);
        for (max_grid, min_chunk, split) in [(1000, 32, false), (12, 1, true)] {
            let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
            assert_eq!(plan.split, split, "{lens:?} grid {max_grid}");
            assert!(eo < 2e-3 && el < 1e-3, "{lens:?} split {split}: |o| {eo:.2e}, |lse| {el:.2e}");
        }
    }
}

/// FlashInfer's planner rules: no split when the requests fill the GPU, and the chosen
/// chunk's CTAs fit in one wave otherwise.
#[test]
fn plan_follows_flashinfer() {
    let whole = Plan::new(&[64; 32], 16, 8, 154);
    assert!(!whole.split);
    assert_eq!(whole.num_ctas(), 32);
    let split = Plan::new(&[256], 16, 8, 154);
    assert!(split.split);
    assert!(split.num_ctas() * 8 <= 154, "{split:?}");
    assert_eq!(split.o_indptr, vec![0, split.num_ctas() as i32]);
}

/// The kernels lower to the instructions they are built around: 16-byte `cp.async`
/// copies, hardware `exp2`, and fused multiply-adds (cute-rs contracts by default).
#[test]
fn lowers_to_cp_async_and_fma() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for partition in [false, true] {
        let compiled = kernel(partition).compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap();
        let ptx = compiled.ptx();
        assert_eq!(ptx.matches("cp.async.cg.shared.global").count(), 4, "{ptx}");
        assert!(ptx.contains("ex2.approx.ftz.f32"), "{ptx}");
        assert!(ptx.matches("fma.rn.f32").count() >= 64, "{ptx}");
    }
}

/// The decode kernel over contiguous (ragged) K/V, whole and split.
#[test]
fn ragged_matches_reference_whole_and_split() {
    let p = Problem::new_ragged(&[1, 7, 21, 40], 2, 7);
    for (max_grid, min_chunk, split) in [(1000, 128, false), (12, 4, true)] {
        let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
        assert_eq!(plan.split, split);
        assert!(eo < 2e-3 && el < 1e-3, "split {split}: |o| {eo:.2e}, |lse| {el:.2e}");
    }
}

/// `single_decode_with_kv_cache`: one request, contiguous K/V, whole (up to 256 tokens)
/// and split by FlashInfer's single-decode chunk rule, then merged.
#[test]
fn single_decode_matches_reference() {
    for (kv_len, heads, split) in [(1usize, 1usize, false), (200, 2, false), (600, 1, true)] {
        let p = Problem::new_ragged(&[kv_len], heads, 3);
        assert_eq!(p.single_plan(1000).split, split);
        let (o, lse) = single_decode_with_kv_cache(p.q.clone(), p.k_cache.clone(), p.v_cache.clone(), heads, 1000);
        let (want_o, want_lse) = p.reference();
        let (eo, el) = (max_error(&o, &want_o), max_error(&lse, &want_lse));
        assert!(eo < 2e-3 && el < 1e-3, "kv_len {kv_len}: |o| {eo:.2e}, |lse| {el:.2e}");
    }
}

/// FlashInfer's single-decode chunk rule.
#[test]
fn single_plan_follows_flashinfer() {
    assert!(!Plan::single(256, 8, 264).split);
    let p = Plan::single(1024, 8, 264);
    assert_eq!((p.chunk_pages, p.num_ctas()), (256, 4));
    let p = Plan::single(32768, 8, 264);
    assert_eq!((p.chunk_pages, p.num_ctas()), (993, 33));
    assert_eq!(p.o_indptr, vec![0, 33]);
}

/// An FP8 (E4M3 and E5M2) K/V cache, paged and ragged, whole and split, unscaled and with
/// FlashInfer's `k_scale`/`v_scale`, against the f64 reference on the dequantised cache.
#[test]
fn fp8_cache_matches_reference() {
    for kv in [KvDtype::E4M3, KvDtype::E5M2] {
        for (scales, ragged) in [((1.0f32, 1.0f32), false), ((0.5, 2.0), false), ((0.25, 0.75), true)] {
            let base = if ragged {
                Problem::new_ragged(&[1, 7, 21, 40], 2, 7)
            } else {
                Problem::new(&[1, 7, 21, 40], 4, 2, 7)
            };
            let p = base.quantized(kv, scales.0, scales.1);
            for (max_grid, min_chunk, split) in [(1000, 64, false), (12, 1, true)] {
                let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
                assert_eq!(plan.split, split);
                assert!(
                    eo < 2e-3 * scales.1.max(1.0) && el < 1e-3,
                    "{kv:?} {scales:?} ragged {ragged} split {split}: |o| {eo:.2e}, |lse| {el:.2e}"
                );
            }
        }
    }
}

/// The FP8 kernels copy 16 elements per `cp.async` and widen them in pairs with the
/// hardware conversion.
#[test]
fn fp8_lowers_to_paired_cvt() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    for (kv, cvt) in [(KvDtype::E4M3, "cvt.rn.f16x2.e4m3x2"), (KvDtype::E5M2, "cvt.rn.f16x2.e5m2x2")] {
        let k = kernel_variant_kv(true, STAGES, false, kv);
        let ptx = k.compile_with(&tc, &CompileOptions::new().gpu_arch(Arch::Sm89)).unwrap().ptx().to_string();
        assert_eq!(ptx.matches("cp.async.cg.shared.global").count(), 4, "{ptx}");
        // 16 K and 16 V elements for each of a thread's 4 tokens, two per conversion.
        assert_eq!(ptx.matches(cvt).count(), 64, "{ptx}");
    }
}

/// FlashInfer's attention variants, alone and combined, paged (GQA, whole and split) and
/// ragged, against the f64 reference: soft-cap, sliding window (the window shorter than,
/// equal to and longer than requests, and off with `window_left = -1`), and ALiBi.
#[test]
fn variants_match_reference() {
    let lens = [1usize, 7, 21, 40, 70];
    type Make = fn(Problem) -> Problem;
    let cases: [(&str, Make); 6] = [
        ("cap", |p| p.with_soft_cap(5.0)),
        ("win", |p| p.with_window(9)),
        ("win off", |p| p.with_window(-1)),
        ("alibi", |p| p.with_alibi()),
        ("cap + win", |p| p.with_soft_cap(2.0).with_window(20)),
        ("alibi + win", |p| p.with_alibi().with_window(0)),
    ];
    for (name, make) in cases {
        for (max_grid, min_chunk) in [(1000, 32), (12, 1)] {
            let p = make(Problem::new(&lens, 4, 2, 7));
            let (plan, eo, el) = simulate(&p, max_grid, min_chunk);
            assert!(eo < 2e-3 && el < 1e-3, "{name} split {}: |o| {eo:.2e}, |lse| {el:.2e}", plan.split);
        }
        let p = make(Problem::new_ragged(&lens, 1, 5));
        let (_, eo, el) = simulate(&p, 6, 1);
        assert!(eo < 2e-3 && el < 1e-3, "ragged {name}: |o| {eo:.2e}, |lse| {el:.2e}");
    }
    // A window over an FP8 cache.
    let p = Problem::new(&lens, 4, 2, 7).quantized(KvDtype::E4M3, 0.5, 2.0).with_window(13).with_soft_cap(3.0);
    let (_, eo, el) = simulate(&p, 12, 1);
    assert!(eo < 4e-3 && el < 1e-3, "fp8 cap + win: |o| {eo:.2e}, |lse| {el:.2e}");
}

/// A window plans only its own pages, and the variant kernels lower to `tanh.approx`.
#[test]
fn window_plan_and_variant_ptx() {
    let p = Problem::new(&[4096], 16, 8, 1).with_window(1023);
    let plan = p.plan(154);
    assert_eq!(plan.num_ctas(), Problem::new(&[1024], 16, 8, 1).plan(154).num_ctas());
    let Some(tc) = Toolchain::discover() else { return };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    let v = Variant { logits_soft_cap: true, sliding_window: true, ..Variant::NONE };
    let ptx = kernel_full(false, STAGES, false, KvDtype::F16, v).compile_with(&tc, &opts).unwrap().ptx().to_string();
    assert!(ptx.contains("tanh.approx.f32"));
}

// SPDX-License-Identifier: Apache-2.0

//! RoPE with a cos/sin cache and the paged-KV append on the simulator (with cute-rs's race
//! checks) against f64 / exact references; the launch planner; and the lowered PTX.

use cute::prelude::*;
use flashinfer::rope::*;

fn check_rope<T: Half, I: IndexType>(head_dim: usize, interleave: bool, nnz: usize, nq: usize, nk: usize) {
    let p = Problem::<T, I>::new(nnz, nq, nk, head_dim, interleave, 300, 5);
    for (variant, split) in [(Variant::HeadParallel, 1), (Variant::Unsplit, 1), (Variant::Split, 3)] {
        let (eq, ek) = simulate(&p, variant, split);
        let tol = T::TOL;
        assert!(
            eq < tol && ek < tol,
            "{} {} head_dim {head_dim} interleave {interleave} {variant:?}: |q| {eq:.2e}, |k| {ek:.2e}",
            T::NAME,
            I::NAME
        );
    }
}

#[test]
fn rope_matches_reference() {
    // NeoX and GPT-J, f16 and bf16, i32 and i64 positions, head dim 64 and 128; token counts
    // that fill and do not fill the last block.
    for interleave in [false, true] {
        check_rope::<F16, i32>(128, interleave, 19, 4, 2);
        check_rope::<BF16, i64>(128, interleave, 9, 3, 1);
        check_rope::<F16, i64>(64, interleave, 35, 2, 2);
        check_rope::<BF16, i32>(64, interleave, 1, 5, 3);
    }
}

#[test]
fn append_matches_reference() {
    // Prefill-style (several tokens per request) and decode-style (one each), partial pages,
    // an append that starts mid-page, and fewer grid blocks than tokens.
    for (seq, app, page, heads, dim, grid) in [
        (vec![5usize, 17, 1, 40], vec![5usize, 3, 1, 9], 4usize, 2usize, 128usize, 3usize),
        (vec![16, 33, 2], vec![1, 1, 1], 16, 8, 128, 2),
        (vec![9, 20], vec![9, 4], 8, 3, 64, 100),
    ] {
        let p = AppendProblem::<F16>::new(&seq, &app, page, heads, dim, 3);
        assert_eq!(simulate_append(&p, grid), 0, "{seq:?} {app:?} page {page}");
        let p = AppendProblem::<BF16>::new(&seq, &app, page, heads, dim, 4);
        assert_eq!(simulate_append(&p, grid), 0, "bf16 {seq:?}");
        // An FP8 cache (FP8 inputs, as FlashInfer's append requires), 16 per thread.
        let p = AppendProblem::<F8E4M3FN>::new(&seq, &app, page, heads, dim, 5);
        assert_eq!(simulate_append(&p, grid), 0, "e4m3 {seq:?}");
        let p = AppendProblem::<F8E5M2>::new(&seq, &app, page, heads, dim, 6);
        assert_eq!(simulate_append(&p, grid), 0, "e5m2 {seq:?}");
    }
}

#[test]
fn batch_indices_positions_follow_flashinfer() {
    let (b, pos) = batch_indices_positions(&[0, 2, 3, 6], &[5, 1, 10]);
    assert_eq!(b, vec![0, 0, 1, 2, 2, 2]);
    assert_eq!(pos, vec![3, 4, 0, 7, 8, 9]);
}

/// FlashInfer's planner: head-parallel below one wave of token blocks, then the unsplit
/// kernel when the grid fills the GPU, else a split of the head loop.
#[test]
fn plan_follows_flashinfer() {
    // 22 SMs, 12 resident 128-thread blocks each (an RTX 2000 Ada-like device).
    let (ru, rs) = (264, 240);
    let few = plan(128, 16, 32, 8, ru, rs);
    assert_eq!(few.variant, Variant::HeadParallel);
    assert_eq!(few.grid, [2, 40, 1]);
    let many = plan(128, 4096, 32, 8, ru, rs);
    assert_ne!(many.variant, Variant::HeadParallel);
}

/// The kernels lower to 16-byte vector accesses and fused multiply-adds.
#[test]
fn lowers_to_vector_accesses() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    for variant in [Variant::HeadParallel, Variant::Unsplit, Variant::Split] {
        let compiled = kernel::<F16, i32>(128, false, variant).compile_with(&tc, &opts).unwrap();
        let ptx = compiled.ptx();
        assert!(ptx.contains("ld.global.v4.b32"), "{ptx}");
        assert!(ptx.contains("st.global.v4.b32"), "{ptx}");
        assert!(ptx.contains("fma.rn.f32"), "{ptx}");
    }
    let ptx = append_kernel::<F16>(8, 128).compile_with(&tc, &opts).unwrap().ptx().to_string();
    assert!(ptx.matches("st.global.v4.b32").count() >= 2, "{ptx}");
}

fn check_freq<T: Half, I: IndexType>(hd: usize, rd: usize, interleave: bool, cfg: FreqConfig, nnz: usize) {
    let p = Problem::<T, I>::new(nnz, 3, 2, hd, interleave, 300, 7);
    for (variant, split) in [(Variant::HeadParallel, 1), (Variant::Unsplit, 1), (Variant::Split, 2)] {
        let (eq, ek) = simulate_freq(&p, &cfg, variant, split);
        assert!(
            eq < T::TOL && ek < T::TOL,
            "{} hd {hd} rd {rd} interleave {interleave} {variant:?}: |q| {eq:.2e}, |k| {ek:.2e}",
            T::NAME
        );
    }
}

#[test]
fn freq_rope_matches_reference() {
    for interleave in [false, true] {
        check_freq::<F16, i32>(128, 128, interleave, FreqConfig::new(128), 11);
        check_freq::<BF16, i64>(64, 64, interleave, FreqConfig::new(64), 5);
        // rotary_dim < head_dim: the tail is untouched.
        check_freq::<F16, i32>(128, 64, interleave, FreqConfig::new(64), 9);
        check_freq::<F16, i32>(128, 32, interleave, FreqConfig::new(32), 6);
        check_freq::<BF16, i32>(64, 16, interleave, FreqConfig::new(16), 6);
        // Scaled and Llama 3.1.
        check_freq::<F16, i32>(128, 96, interleave, FreqConfig { rope_scale: 4.0, ..FreqConfig::new(96) }, 7);
        check_freq::<F16, i32>(128, 128, interleave, FreqConfig::llama31(128), 13);
        check_freq::<BF16, i64>(128, 64, interleave, FreqConfig::llama31(64), 4);
    }
}

#[test]
fn indptr_positions() {
    assert_eq!(pos_ids_from_indptr(&[0, 3, 3, 5], &[10, 0, 7]), vec![10, 11, 12, 7, 8]);
}

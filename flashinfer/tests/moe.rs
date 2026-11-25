// SPDX-License-Identifier: Apache-2.0

//! The fused MoE pipeline on the simulator (with cute-rs's race and alignment checks) against
//! the f64 reference, and the PTX its GEMMs lower to.

use cute::prelude::*;
use flashinfer::moe::*;

/// Decode-sized: experts with zero, one and a few rows, FlashInfer's 16-row tile.
#[test]
fn decode_tile_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        let p = Problem::new(dtype, MoeShape::new(128, 128, 4, 2), 5, 1);
        assert_eq!((p.tile1, p.tile2), (DECODE_TILE, DECODE_TILE));
        let err = simulate(&p);
        assert!(err <= 1.0, "{dtype:?}: error {err}");
    }
}

/// Top-1 and top-4 routing over more experts than rows (most experts empty).
#[test]
fn sparse_experts_match_reference() {
    for (e, k, tokens) in [(16usize, 1usize, 3usize), (12, 4, 2)] {
        let p = Problem::new(Dtype::BF16, MoeShape::new(128, 128, e, k), tokens, 2);
        let err = simulate(&p);
        assert!(err <= 1.0, "e{e} k{k}: error {err}");
    }
}

/// The prefill tiles: experts spanning several M tiles, partial last tiles.
#[test]
fn prefill_tiles_match_reference() {
    let p = Problem::new(Dtype::F16, MoeShape::new(128, 128, 4, 2), 50, 3);
    assert_eq!(p.tile1.bm, 64);
    assert!(simulate(&p) <= 1.0);
    let mut p = Problem::new(Dtype::BF16, MoeShape::new(64, 128, 2, 2), 75, 4);
    assert_eq!(p.tile1.bm, 128);
    p.tile2 = Tile { bm: 32, bn: 64, bk: 32, stages: 2, wm: 1, wn: 2 };
    assert!(simulate(&p) <= 1.0);
}

/// `mma.sync` m16n8k16, `ldmatrix` and 16-byte `cp.async` in both GEMMs.
#[test]
fn gemms_lower_to_tensor_core_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    let shape = MoeShape::new(4096, 14336, 8, 2);
    let count = |ptx: &str, pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
    for k in [gemm1_kernel(Dtype::BF16, shape, DECODE_TILE), gemm2_kernel(Dtype::BF16, shape, DECODE_TILE)] {
        let ptx = k.compile_with(&tc, &opts).unwrap().ptx().to_string();
        assert!(count(&ptx, "mma.sync.aligned.m16n8k16") >= 16, "{}", k.name());
        assert!(count(&ptx, "ldmatrix.sync") > 0 && count(&ptx, "cp.async.cg.shared.global") > 0, "{}", k.name());
    }
}

/// Softmax-top-k (with and without renormalizing) and Renormalize: one expert per lane,
/// padded lanes, and several experts per lane; f32, f16 and bf16 logits.
#[test]
fn softmax_routing_matches_reference() {
    for routing in
        [Routing::SoftmaxTopK { renormalize: false }, Routing::SoftmaxTopK { renormalize: true }, Routing::Renormalize]
    {
        for (score, e, k) in [(ScoreType::F32, 8usize, 2usize), (ScoreType::BF16, 60, 4), (ScoreType::F16, 128, 8)] {
            let p = RouteProblem::new(routing, score, 6, e, k, 11);
            let err = simulate_route(&p);
            assert!(err <= 1.0, "{routing:?} {score:?} e{e} k{k}: error {err}");
        }
    }
}

/// DeepSeek-V3 grouped routing: DeepSeek-V3's 256 experts in 8 groups, a small grouped
/// layout, and Kimi K2's ungrouped 384 experts.
#[test]
fn deepseek_routing_matches_reference() {
    for (score, e, n_group, topk_group, k) in [
        (ScoreType::BF16, 256usize, 8usize, 4usize, 8usize),
        (ScoreType::F32, 256, 8, 4, 8),
        (ScoreType::F16, 64, 8, 4, 6),
        (ScoreType::F32, 384, 1, 1, 8),
    ] {
        let p = RouteProblem::new(Routing::DeepSeekV3 { n_group, topk_group }, score, 9, e, k, 12);
        let err = simulate_route(&p);
        assert!(err <= 1.0, "{score:?} e{e} groups {n_group}/{topk_group} k{k}: error {err}");
    }
}

// SPDX-License-Identifier: Apache-2.0

//! The segment GEMM and `bmm_fp8` on the simulator (with cute-rs's race and alignment checks)
//! against the f64 reference, and the PTX they lower to.

use cute::prelude::*;
use flashinfer::gemm::*;

const SMALL: Tile = Tile { bm: 32, bn: 64, bk: 32, stages: 3, wm: 1, wn: 2 };

/// Segments shorter than, equal to and longer than the M tile (and an empty one), column- and
/// row-major weights, with and without `weight_indices`, accumulating into `y`.
#[test]
fn segment_gemm_matches_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for col_major in [true, false] {
            for indices in [None, Some(vec![2, 0, 2, 1])] {
                let mut p = SegmentProblem::new(dtype, &[5, 32, 0, 41], 96, 128, col_major, 4, indices, true, 3);
                p.tile = SMALL;
                let err = simulate_segment(&p);
                assert!(err <= 1.0, "{dtype:?} col_major={col_major}: error {err}");
            }
        }
    }
}

/// FlashInfer's own tile, and the short-segment tile [`segment_tile`] picks.
#[test]
fn segment_gemm_flashinfer_tile() {
    let mut p = SegmentProblem::new(Dtype::BF16, &[3, 130], 64, 128, true, 2, None, false, 7);
    assert_eq!(p.tile, Tile::SEGMENT);
    assert!(simulate_segment(&p) <= 1.0);
    p = SegmentProblem::new(Dtype::F16, &[3, 60, 1], 128, 128, false, 3, None, true, 8);
    assert_eq!(p.tile, segment_tile(64, 3));
    assert!(simulate_segment(&p) <= 1.0);
}

/// Both FP8 types and both output types, `m` a partial tile and several batches.
#[test]
fn bmm_fp8_matches_reference() {
    for fp8 in [Fp8::E4M3, Fp8::E5M2] {
        for out in [Dtype::BF16, Dtype::F16] {
            let mut p = BmmFp8Problem::new(fp8, out, 2, 20, 64, 128, 5);
            p.tile = Tile { bm: 32, bn: 32, bk: 64, stages: 2, wm: 1, wn: 2 };
            let err = simulate_bmm_fp8(&p);
            assert!(err <= 1.0, "{fp8:?} {out:?}: error {err}");
        }
    }
    let mut p = BmmFp8Problem::new(Fp8::E4M3, Dtype::BF16, 3, 4, 128, 256, 9);
    assert_eq!(p.tile.bm, 16);
    p.tile.stages = 2;
    assert!(simulate_bmm_fp8(&p) <= 1.0);
}

/// `mma.sync` m16n8k16 / m16n8k32 E4M3, `ldmatrix` and 16-byte `cp.async`.
#[test]
fn lowers_to_tensor_core_ptx() {
    let Some(tc) = Toolchain::discover() else {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
        return;
    };
    let opts = CompileOptions::new().gpu_arch(Arch::Sm89);
    let count = |ptx: &str, pat: &str| ptx.lines().filter(|l| l.contains(pat)).count();
    for col_major in [true, false] {
        let ptx = segment_gemm_kernel(Dtype::F16, 4096, 4096, col_major, Tile::SEGMENT)
            .compile_with(&tc, &opts)
            .unwrap()
            .ptx()
            .to_string();
        assert!(count(&ptx, "mma.sync.aligned.m16n8k16") >= 64, "col_major={col_major}");
        assert!(count(&ptx, "ldmatrix.sync") > 0 && count(&ptx, "cp.async.cg.shared.global") > 0);
    }
    let k = bmm_fp8_kernel(Fp8::E4M3, Dtype::BF16, 4096, 4096, BmmFp8Problem::tile(64, 4096));
    let old = k.min_llvm_version().is_some_and(|need| tc.llvm_version().is_some_and(|have| have < need));
    let ptx = k.compile_with(&tc, &opts.mma_inline_ptx(old)).unwrap().ptx().to_string();
    assert!(count(&ptx, "m16n8k32.row.col.f32.e4m3.e4m3.f32") > 0);
}

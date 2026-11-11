//! The example kernels, simulated and checked against CPU references.
//!
//! The examples are compiled into this test as modules, so the kernels
//! tested here are exactly the ones in `examples/`.

#[allow(dead_code)]
#[path = "../examples/elementwise_add.rs"]
mod elementwise_add;
#[allow(dead_code)]
#[path = "../examples/sgemm.rs"]
mod sgemm;
#[allow(dead_code)]
#[path = "../examples/single_warp_pipeline.rs"]
mod single_warp_pipeline;
#[allow(dead_code)]
#[path = "../examples/tensorop_gemm.rs"]
mod tensorop_gemm;

use tensorop_gemm::GemmConfig;

#[test]
fn elementwise_add_naive() {
    let report = elementwise_add::run(&elementwise_add::naive(), elementwise_add::NAIVE_GRID, [256, 1, 1]);
    assert_eq!(report.global_efficiency(), 1.0);
}

#[test]
fn elementwise_add_tiled_with_partial_tiles() {
    let kernel = elementwise_add::tiled();
    let report = elementwise_add::run(&kernel, elementwise_add::TILED_GRID, [128, 1, 1]);
    // Each thread moves its 4x4 block with four 128-bit loads per input.
    let vector_loads =
        kernel.ir().count_ops(|op| matches!(op, cute::ir::Op::Load { results, .. } if results.len() == 4));
    assert_eq!(vector_loads, 8);
    assert_eq!(report.global_efficiency(), 1.0);
}

#[test]
fn sgemm() {
    let report = sgemm::run(128, 128, 32);
    assert_eq!(report.blocks, 1);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

#[test]
fn sgemm_multiple_blocks() {
    sgemm::run(256, 128, 16);
}

#[test]
fn single_warp_pipeline() {
    let report = single_warp_pipeline::run();
    assert_eq!(report.mma_instructions, 128 / 16);
}

#[test]
fn tensorop_gemm_single_tile() {
    let report = tensorop_gemm::run(GemmConfig::new(128, 128, 64));
    // 4 warps x (4 x 8 MMAs per 16-deep k-block) x 4 k-blocks.
    assert_eq!(report.mma_instructions, 4 * 32 * 4);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

#[test]
fn tensorop_gemm_grid_and_short_pipeline() {
    // Two blocks along M, and a K loop of exactly `stages` tiles.
    tensorop_gemm::run(GemmConfig::new(256, 128, 96));
}

#[test]
fn tensorop_gemm_bf16() {
    let report = tensorop_gemm::run(GemmConfig { ab: cute::DType::BF16, ..GemmConfig::new(256, 128, 96) });
    assert_eq!(report.mma_instructions, 2 * 4 * 32 * 6);
    assert_eq!(report.bank_conflict_factor(), 1.0);
}

#[test]
fn tensorop_gemm_two_stages_unswizzled() {
    let report = tensorop_gemm::run(GemmConfig { stages: 2, swizzle: false, ..GemmConfig::new(128, 128, 64) });
    assert!(report.bank_conflict_factor() > 2.0, "unswizzled K-major tiles should conflict: {report}");
}

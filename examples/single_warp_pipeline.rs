//! A single-warp, double-buffered tensor-core GEMM.
//!
//! This is the kernel the project started from (the original `main.rs`),
//! rewritten with the finished DSL. The address arithmetic that used to be
//! written by hand now comes from tiled copies and the MMA's layouts, but the
//! pipeline is the same, and so are the explanations.
//!
//! Unlike `tensorop_gemm`, the K loop here is a Rust loop: it runs while the
//! kernel is *built* and unrolls the whole pipeline into straight-line code.
//! B is stored K x N row-major (N-major), so its fragments are loaded with
//! the transposing `ldmatrix.trans`.
//!
//! Run with `cargo run --example single_warp_pipeline`.

use cute::prelude::*;

// =========================================================
// GEMM DIMENSIONS
//
// A = M x K
// B = K x N
// D = M x N
//
// M and N are fixed to one MMA tile.
//
// K can be any compile-time multiple of 16.
// =========================================================

const M: usize = 16;
const N: usize = 8;

const K: usize = 128;
const K_TILE: usize = 16;
const _: () = assert!(K > 0 && K.is_multiple_of(K_TILE));

pub fn build() -> Kernel {
    let number_of_k_tiles = K / K_TILE;

    Kernel::builder("single_warp_pipeline").block_dim([32, 1, 1]).build(|| {
        // =====================================================
        // GLOBAL MEMORY
        //
        // A is row-major M x K. B is row-major K x N; CuTe views B as
        // (N, K), so that is N-major: stride 1 along N, N along K.
        // =====================================================

        let a = param_tensor::<F16>("a", Layout::row_major((M, K)));
        let b = param_tensor::<F16>("b", Layout::new((N, K), (1, N)));
        let d = param_tensor::<f32>("d", Layout::row_major((M, N)));

        // One (M, K_TILE) tile of A and one (N, K_TILE) tile of B per K step:
        // a_tiles is (M, K_TILE, k), b_tiles is (N, K_TILE, k).
        let a_tiles = local_tile(&a, (M, K_TILE), (0, ..));
        let b_tiles = local_tile(&b, (N, K_TILE), (0, ..));

        // =====================================================
        // DOUBLE-BUFFERED SHARED MEMORY
        //
        // Even K tiles use stage 0.
        // Odd K tiles use stage 1.
        //
        // tile 0 -> stage 0
        // tile 1 -> stage 1
        // tile 2 -> stage 0
        // tile 3 -> stage 1
        // ...
        // =====================================================

        let shared_a = make_smem_tensor::<F16>(Layout::new((M, K_TILE, 2), (K_TILE, 1, M * K_TILE)));
        let shared_b = make_smem_tensor::<F16>(Layout::new((N, K_TILE, 2), (1, N, N * K_TILE)));

        // =====================================================
        // GLOBAL -> SHARED COPIES
        //
        // A tile: 16 x 16 = 256 F16 values over 32 lanes
        //         = 8 F16 per lane = 16 bytes per lane
        //
        // B tile: 16 x 8 = 128 F16 values over 32 lanes
        //         = 4 F16 per lane = 8 bytes per lane
        // =====================================================

        let lane_id = lane_idx();
        let copy_a = make_tiled_copy_tv(
            &make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: false }),
            &Layout::row_major((16, 2)),
            &Layout::row_major((1, 8)),
        );
        let copy_b = make_tiled_copy_tv(
            &make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 8, bypass_l1: false }),
            &Layout::col_major((2, 16)),
            &Layout::col_major((4, 1)),
        );
        let thr_copy_a = copy_a.get_slice(lane_id);
        let thr_copy_b = copy_b.get_slice(lane_id);
        let global_a = thr_copy_a.partition_s(&a_tiles);
        let global_b = thr_copy_b.partition_s(&b_tiles);
        let staged_a = thr_copy_a.partition_d(&shared_a);
        let staged_b = thr_copy_b.partition_d(&shared_b);

        let issue = |k_tile: usize| {
            let stage = k_tile % 2;
            copy(&copy_a, &global_a.slice((.., .., .., k_tile)), &staged_a.slice((.., .., .., stage)));
            copy(&copy_b, &global_b.slice((.., .., .., k_tile)), &staged_b.slice((.., .., .., stage)));
            // A and B for this K tile become one group.
            cp_async_commit_group();
        };

        // =====================================================
        // MMA SETUP
        // =====================================================

        let mma = make_tiled_mma(make_mma_atom(MmaOp::Sm80(MmaKind::M16N8K16F32F16)), Layout::col_major((1, 1, 1)));
        let thread_mma = mma.get_slice(lane_id);
        let a_fragment = thread_mma.partition_fragment_a::<F16, _>(&shared_a.slice((.., .., 0)));
        let b_fragment = thread_mma.partition_fragment_b::<F16, _>(&shared_b.slice((.., .., 0)));
        let accumulator = thread_mma.partition_fragment_c::<f32, _>(&d);
        accumulator.fill(0.0);

        let load_a = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
        let load_b = make_tiled_copy_b(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 2, trans: true }), &mma);
        let thr_load_a = load_a.get_slice(lane_id);
        let thr_load_b = load_b.get_slice(lane_id);
        let shared_a_rows = thr_load_a.partition_s(&shared_a);
        let shared_b_rows = thr_load_b.partition_s(&shared_b);
        let a_fragment_view = thr_load_a.retile_d(&a_fragment);
        let b_fragment_view = thr_load_b.retile_d(&b_fragment);

        // =====================================================
        // PIPELINE PROLOGUE
        //
        // Initially we have no computation to perform.
        //
        // So get the first TWO K tiles moving:
        //
        // tile 0 -> stage 0 -> group 0
        // tile 1 -> stage 1 -> group 1
        //
        // This Rust loop runs while BUILDING the IR.
        //
        // It is NOT a GPU runtime loop.
        // =====================================================

        for k_tile in 0..number_of_k_tiles.min(2) {
            issue(k_tile);
        }

        // =====================================================
        // STEADY-STATE PIPELINE
        //
        // This is also a Rust loop.
        //
        // It UNROLLS the K pipeline while constructing the IR.
        // =====================================================

        for current_k_tile in 0..number_of_k_tiles {
            // =================================================
            // WAIT UNTIL CURRENT TILE IS READY
            //
            // Usually there is one newer tile already in flight.
            //
            // So:
            //
            // wait_group(1)
            //
            // says:
            //
            // current/older group must be finished
            // newest group may still be running
            //
            // On the final tile there is no newer group, so use 0.
            // =================================================

            cp_async_wait_group(if current_k_tile + 1 < number_of_k_tiles { 1 } else { 0 });
            sync_threads();

            // =================================================
            // SHARED -> REGISTERS
            //
            // After BOTH ldmatrix operations have completed,
            // this shared-memory stage is no longer needed for the
            // current MMA.
            //
            // The MMA operates on the register fragments.
            // =================================================

            let stage = current_k_tile % 2;
            copy(&load_a, &shared_a_rows.slice((.., .., .., stage)), &a_fragment_view);
            copy(&load_b, &shared_b_rows.slice((.., .., .., stage)), &b_fragment_view);

            // =================================================
            // MAKE SURE EVERY THREAD IS DONE READING THIS SHARED
            // STAGE BEFORE WE START OVERWRITING IT.
            //
            // With a single-warp kernel this is more conservative
            // than necessary, but it makes the buffer lifetime
            // explicit and stays correct with more warps.
            // =================================================

            sync_threads();

            // =================================================
            // PREFETCH TILE current + 2
            //
            // current tile:
            //     using one shared stage
            //
            // current + 1:
            //     already occupying the other stage
            //
            // current + 2:
            //     goes back into the stage we JUST FINISHED reading.
            // =================================================

            if current_k_tile + 2 < number_of_k_tiles {
                issue(current_k_tile + 2);
            }

            // =================================================
            // MMA CURRENT TILE
            //
            // Current A and B are already in REGISTERS.
            //
            // So a future cp.async can now be writing into the
            // shared stage we just freed while this MMA runs.
            // =================================================

            gemm(&mma, &accumulator, &a_fragment, &b_fragment, &accumulator);
        }

        // =====================================================
        // STORE FINAL RESULT
        // =====================================================

        autovec_copy(&accumulator, &thread_mma.partition_c(&d));
    })
}

pub fn run() -> SimReport {
    let kernel = build();
    let a: Vec<f32> = (0..M * K).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i * 5) % 9) as f32 - 4.0).collect();
    let mut args = vec![KernelArg::f16_from_f32(&a), KernelArg::f16_from_f32(&b), KernelArg::f32(vec![0.0; M * N])];
    let report = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let d = args[2].as_f32();
    for m in 0..M {
        for n in 0..N {
            let expected: f32 = (0..K).map(|k| a[m * K + k] * b[k * N + n]).sum();
            assert_eq!(d[m * N + n], expected, "D[{m},{n}]");
        }
    }
    report
}

#[allow(dead_code)]
fn main() {
    let report = run();
    println!("single_warp_pipeline (M={M}, N={N}, K={K}): correct\n{report}\n");
    println!("{}", build().to_mlir());
}

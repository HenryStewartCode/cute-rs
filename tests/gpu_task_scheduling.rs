// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Runs the kernels of tests/ts_kernels (structured after NVIDIA CUTLASS
// 4.8.1's BSD-3-Clause Task Scheduling copy tutorials; see that file and
// THIRD-PARTY-NOTICES.md) on a real GPU.

//! Task Scheduling resources called from hand-written warp-specialised
//! kernels on a real GPU: the grid-stride
//! copy, the staged copy through an `AsyncAsync` shared-memory resource and
//! the persistent copy over a static work queue that skips over-launched
//! tiles. All SM80-level, so they run on the SM89 GPU. Needs the `cuda`
//! feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_task_scheduling
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/ts_copy_grid_stride.rs"]
mod ts_copy_grid_stride;
#[path = "../examples/ts_copy_persistent_skip_tile.rs"]
mod ts_copy_persistent_skip_tile;
mod ts_kernels;

use cute::cuda::CudaContext;
use cute::prelude::*;
use ts_kernels::*;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

#[test]
fn gpu_task_scheduling_copies() {
    let Some((ctx, tc)) = device() else { return };

    let src = ramp_i16(GS_ENTRIES);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0i16; GS_ENTRIES])];
    let cfg = LaunchConfig::new([GS_BLOCKS, 1, 1], [GS_THREADS, 1, 1]);
    grid_stride_copy().run_cuda(&ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i16>(), src);

    let a = ramp_i32(ST_ROWS * ST_COLS);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0i32; ST_ROWS * ST_COLS])];
    let cfg = LaunchConfig::new([ST_BLOCKS as u32, 1, 1], [ST_THREADS, 1, 1]);
    staged_copy().run_cuda(&ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i32>(), a);

    let a = ramp_i32(PS_ROWS * PS_COLS);
    let mut args = vec![
        KernelArg::buffer(&a),
        KernelArg::buffer(&vec![0i32; PS_ROWS * PS_COLS]),
        KernelArg::buffer(&[0i32; PS_LAUNCH_TILES]),
    ];
    let cfg = LaunchConfig::new(PS_GRID, [PS_TILE as u32, 1, 1]);
    persistent_skip_copy().run_cuda(&ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i32>(), a);
    assert_eq!(args[2].to_vec::<i32>(), [1, 1, 1, 2, 2]);
}

/// The TaskManager's kernels: the two
/// ported examples and the two-task pipelined copy, against plain Rust.
#[test]
fn gpu_task_manager_kernels() {
    let Some((ctx, tc)) = device() else { return };

    use ts_copy_grid_stride as gs;
    let n = 1 << 16;
    let src = gs::source(n);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0i16; n])];
    let kernel = gs::build(n, 8, 2, None);
    kernel.run_cuda(&ctx, &tc, gs::launch(16, 8), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i16>(), gs::reference(&src));

    use ts_copy_persistent_skip_tile as ps;
    let a = ps::matrix(8, 5 * ps::TILE);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0.0f32; a.len()])];
    ps::build(8, 5, 4).run_cuda(&ctx, &tc, ps::launch(3), &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<f32>(), ps::reference(&a));

    let a = ramp_i32(ST_ROWS * ST_COLS);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0i32; ST_ROWS * ST_COLS])];
    let cfg = LaunchConfig::new([ST_BLOCKS as u32, 1, 1], [ST_THREADS, 1, 1]);
    managed_staged_copy(false).run_cuda(&ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[1].to_vec::<i32>(), a);
}

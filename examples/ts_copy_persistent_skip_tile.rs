// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py
//   (the resources OversubscribedCopyWorkQueue and PersistentCopyResource, the
//   schedule `copy_schedule`, the task `PersistentCopyTask`, the dependency
//   graph and the TaskManager calls of `tma_copy_persistent_skip_kernel`, and
//   the over-launched tile grid of its host function),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A persistent copy over a static work queue that skips over-launched
//! tiles, written with Task Scheduling and ported from
//! `01_copy_persistent_skip_tile.py`. The matrix (`rows x cols`, row-major)
//! is copied a column tile of `TILE` at a time; the static persistent
//! scheduler hands out `logical_tiles + overlaunch_tiles` tiles, and the
//! work queue's skip predicate (`tile_idx.x >= logical_tiles`) marks the
//! extra ones. The schedule's work-tile loop puts the row loop in a
//! skippable region, so a skipped tile runs only the queue's bookkeeping
//! (`try_wait`, `wait`, `get_and_advance_work_tile`, `release`). One task on
//! every warp runs it; the TaskManager seeds the first tile and emits the
//! persistent loop.
//!
//! The tutorial copies `f16` with one CTA per tile; this port copies `f32`
//! with fewer CTAs than tiles, so each CTA also strides over several tiles.
//! SM80-level: it runs on the simulator and the SM89 GPU. Run with
//! `cargo run --release --example ts_copy_persistent_skip_tile`.

use cute::prelude::*;
use cute::task_scheduling::*;
use cute::utils::{PersistentTileSchedulerParams, WorkTileInfo};

/// Columns per tile, and threads per block (one column per thread).
pub const TILE: usize = 64;
/// The problem the test and `main` run.
pub const ROWS: usize = 4;
pub const LOGICAL_TILES: usize = 3;
pub const OVERLAUNCH_TILES: usize = 2;
/// Persistent CTAs (fewer than the launched tiles).
pub const CTAS: u32 = 2;

/// The scheduler's tiles: `launch_tiles` along x, one CTA per cluster.
pub fn params(launch_tiles: usize) -> PersistentTileSchedulerParams {
    PersistentTileSchedulerParams::new((launch_tiles, 1, 1), [1, 1, 1], Default::default())
}

/// The copy kernel of a `rows x logical_tiles * TILE` `f32` matrix over
/// `logical_tiles + overlaunch_tiles` scheduled tiles.
pub fn build(rows: usize, logical_tiles: usize, overlaunch_tiles: usize) -> Kernel {
    let cols = logical_tiles * TILE;
    let sched = params(logical_tiles + overlaunch_tiles);
    Kernel::builder("ts_copy_persistent_skip").block_dim([TILE as u32, 1, 1]).target(Arch::Sm89).build(move || {
        let (tile, ncols) = (TILE as i32, cols as i32);
        let matrix_a = param_tensor::<f32>("matrix_a", Layout::col_major(rows * cols));
        let matrix_b = param_tensor::<f32>("matrix_b", Layout::col_major(rows * cols));

        // Resource construction: the work queue drives which column tile
        // is copied, and skips the over-launched ones.
        let copy_resource =
            MemoryResource::new("CopyResource").work(Work::producer("copy_tile_row").run(move |si, _| {
                let [tile_x, _, _] = si.work_tile.expect("a work tile").tile_idx();
                let col = thread_idx().x + tile_x * tile;
                let idx = si.loop_offset * ncols + col;
                matrix_b.store(idx, matrix_a.load(idx));
                WorkResults::none()
            }));
        let logical = logical_tiles as i32;
        let work_queue = WorkQueue::new(
            "WorkQueue",
            TileSchedulerConfig::create_static_persistent_tile_scheduler_params(sched.clone()),
        )
        .with_skip_if(move |wt: &WorkTileInfo| wt.tile_idx()[0].ge(logical));

        // The schedule: only the data copy is skippable; the queue's
        // bookkeeping runs for every tile.
        let copy_schedule = schedule("copy_schedule", &[&copy_resource, &work_queue], |s, h| {
            let (copy_res, wq) = (h[0], h[1]);
            s.work_tile_loop_skipping(wq, |wtl| {
                wtl.skippable(|| {
                    s.domain_loop(0, rows as i32, 1, 1, |_| {
                        copy_res.call("copy_tile_row");
                    })
                });
                wq.try_wait();
                wq.wait();
                wq.call("get_and_advance_work_tile");
                wq.release();
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let copy_task =
            Task::new("PersistentCopyTask", &[&work_queue], &[&copy_resource], 0, (TILE / 32) as u32, copy_schedule);

        let graph = ResourceDependencyGraph::new().depends_on(&copy_resource, [&work_queue]);
        let mut task_manager = TaskManager::new(vec![copy_task], graph).resource(copy_resource).resource(work_queue);
        task_manager.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        task_manager.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

/// The launch: `ctas` persistent CTAs of `TILE` threads.
pub fn launch(ctas: u32) -> LaunchConfig {
    LaunchConfig::new([1, 1, ctas], [TILE as u32, 1, 1])
}

/// The input matrix, row-major.
pub fn matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| ((i * 37) % 101) as f32 * 0.5 - 20.0).collect()
}

/// The plain-Rust copy: the reference.
pub fn reference(a: &[f32]) -> Vec<f32> {
    a.to_vec()
}

/// Copy on the simulator and check against the reference.
pub fn run(rows: usize, logical_tiles: usize, overlaunch_tiles: usize, ctas: u32) -> Vec<f32> {
    let a = matrix(rows, logical_tiles * TILE);
    let mut args = vec![KernelArg::buffer(&a), KernelArg::buffer(&vec![0.0f32; a.len()])];
    build(rows, logical_tiles, overlaunch_tiles).simulate(launch(ctas), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let b = args[1].to_vec::<f32>();
    assert_eq!(b, reference(&a));
    b
}

fn main() {
    run(ROWS, LOGICAL_TILES, OVERLAUNCH_TILES, CTAS);
    println!(
        "ts_copy_persistent_skip_tile: {ROWS} x {} over {} tiles ({OVERLAUNCH_TILES} skipped) on {CTAS} CTAs: PASS",
        LOGICAL_TILES * TILE,
        LOGICAL_TILES + OVERLAUNCH_TILES
    );
}

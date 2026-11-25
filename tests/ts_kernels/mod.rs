// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The kernels' structure (a grid-stride copy, a load warp handing rows of a
// CTA tile through a staged shared-memory resource to four epilogue-style
// warps, and a persistent copy that skips over-launched tiles) follows the
// BSD-3-Clause Task Scheduling tutorials of NVIDIA CUTLASS 4.8.1:
// examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
// 01_copy_basics_ts/01_copy_grid_stride.py, 02_copy_tma.py and
// 03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See THIRD-PARTY-NOTICES.md.

//! Hand-written warp-specialised kernels that call Task Scheduling
//! resources directly (work methods, the pipelined acquire / commit /
//! wait / release and the work queue), shared by `tests/task_scheduling.rs`
//! (simulator) and `tests/gpu_task_scheduling.rs` (SM89 GPU). Everything is
//! SM80-level: the staged resource uses an `AsyncAsync` pipeline (threads on
//! both sides) where the tutorial uses TMA.

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::task_scheduling::*;
use cute::utils::{PersistentTileSchedulerParams, WorkTileInfo};

// ---------------------------------------------------------------------------
// 01_copy_grid_stride: every thread copies one element per iteration.
// ---------------------------------------------------------------------------

pub const GS_ENTRIES: usize = 1000;
pub const GS_BLOCKS: u32 = 3;
pub const GS_THREADS: u32 = 64;

/// `src` to `dst` (`GS_ENTRIES` i16 each) by a grid-stride domain loop over
/// an input resource's `get_item` and an output resource's `set_item`.
pub fn grid_stride_copy() -> Kernel {
    Kernel::builder("ts_grid_stride").block_dim([GS_THREADS, 1, 1]).target(Arch::Sm89).build(|| {
        let n = GS_ENTRIES as i32;
        let src = param_tensor::<i16>("src", Layout::col_major(GS_ENTRIES));
        let dst = param_tensor::<i16>("dst", Layout::col_major(GS_ENTRIES));
        let item = TaskLocalVariable::<Int16>::scalar("item", 0, "the element of this iteration");
        let mut input = MemoryResource::new("inputGmemResource").work(Work::consumer("get_item").returns(&item).run({
            let item = item.clone();
            move |si, _| {
                let gid = si.loop_offset;
                let val = if_else_yield(gid.lt(n), || src.load(gid), || Int16::constant(0));
                WorkResults::of(&item, val)
            }
        }));
        let mut output = MemoryResource::new("outputGmemResource").work(
            Work::producer("set_item").input::<Int16>("data").run(move |si, args| {
                let gid = si.loop_offset;
                let data: Int16 = args.get("data");
                if_(gid.lt(n), || dst.store(gid, data));
                WorkResults::none()
            }),
        );
        input.create();
        output.create();
        let mut locals = TaskLocals::new();
        locals.declare(&item);
        let threads = GS_THREADS as i32;
        let start = block_idx().x * threads + thread_idx().x;
        let step = grid_dim().x * threads;
        for_range(start, n, step, |gid| {
            let si = StageInfo::new(gid).with_bounds(start, n, step);
            locals.store(&input.run_work("get_item", &si, &WorkArgs::new()));
            output.run_work("set_item", &si, &WorkArgs::new().with("data", locals.get(&item)));
        });
    })
}

// ---------------------------------------------------------------------------
// 02_copy_tma: a load warp fills staged rows, four warps drain them.
// ---------------------------------------------------------------------------

pub const ST_TILE: usize = 128;
pub const ST_ROWS: usize = 5;
pub const ST_BLOCKS: usize = 2;
pub const ST_COLS: usize = ST_TILE * ST_BLOCKS;
pub const ST_STAGES: usize = 2;
/// Warps 0-3 store, warp 4 loads.
pub const ST_THREADS: u32 = 160;

/// Copy the `ST_ROWS x ST_COLS` row-major i32 matrix `a` to `b`: each CTA
/// owns a column tile of `ST_TILE`; warp 4 loads it row by row into a
/// two-stage shared-memory resource (`compute_coords` routed into `load`),
/// warps 0-3 read one element each (`read_smem`) and store it
/// (`store(smem_val)`).
pub fn staged_copy() -> Kernel {
    Kernel::builder("ts_staged_copy").block_dim([ST_THREADS, 1, 1]).target(Arch::Sm89).build(|| {
        let (tile, rows, cols) = (ST_TILE as i32, ST_ROWS as i32, ST_COLS as i32);
        let a = param_tensor::<i32>("a", Layout::col_major(ST_ROWS * ST_COLS));
        let b = param_tensor::<i32>("b", Layout::col_major(ST_ROWS * ST_COLS));
        let data = make_smem_tensor::<i32>(Layout::col_major(ST_STAGES * ST_TILE));
        let gmem_idx = TaskLocalVariable::<Int32>::scalar("gmem_idx", 0, "the first column of the CTA's tile");
        let smem_val = TaskLocalVariable::<Int32>::scalar("smem_val", 0, "the element this thread read");
        let cfg = PipelineConfig::create_async_async_pipeline_cfg(
            ST_STAGES,
            CooperativeGroup::new(Agent::Thread, 32),
            CooperativeGroup::new(Agent::Thread, 128),
            None,
        );
        let mut input =
            MemoryResource::new("inputGmemResource").work(Work::consumer("compute_coords").returns(&gmem_idx).run({
                let v = gmem_idx.clone();
                move |_, _| WorkResults::of(&v, block_idx().x * tile)
            }));
        let mut smem = MemoryResource::new("smemResource")
            .with_pipeline_config(cfg)
            .smem_requirement(SmemAllocation::new::<i32>("smem_data", ST_STAGES * ST_TILE, 128))
            .work(Work::producer("load").input::<Int32>("gmem_idx").run({
                let data = data.clone();
                move |si, args| {
                    let col0: Int32 = args.get("gmem_idx");
                    let stage = si.stage_idx.expect("a staged resource");
                    for k in 0..(ST_TILE / 32) as i32 {
                        let c = lane_idx() + k * 32;
                        data.store(stage * tile + c, a.load(si.loop_offset * cols + col0 + c));
                    }
                    WorkResults::none()
                }
            }))
            .work(Work::consumer("read_smem").returns(&smem_val).run({
                let v = smem_val.clone();
                move |si, _| {
                    let stage = si.stage_idx.expect("a staged resource");
                    WorkResults::of(&v, data.load(stage * tile + thread_idx().x))
                }
            }));
        let mut output = MemoryResource::new("outputGmemResource").work(
            Work::producer("store").input::<Int32>("smem_val").run(move |si, args| {
                let col = block_idx().x * tile + thread_idx().x;
                b.store(si.loop_offset * cols + col, args.get::<Int32>("smem_val"));
                WorkResults::none()
            }),
        );
        input.create();
        smem.create();
        output.create();
        let mut locals = TaskLocals::new();
        locals.declare(&gmem_idx);
        locals.declare(&smem_val);
        if_else(
            warp_idx().eq(4),
            || {
                let mut st = for_range_carry(0, rows, 1, smem.producer_state(), |row, mut st| {
                    let si = StageInfo::new(row).with_bounds(0, rows, 1);
                    locals.store(&input.run_work("compute_coords", &si, &WorkArgs::new()));
                    smem.try_acquire(&st);
                    smem.acquire(&st);
                    let si = smem.producer_stage(&st, si);
                    smem.run_work("load", &si, &WorkArgs::new().with("gmem_idx", locals.get(&gmem_idx)));
                    smem.commit(&mut st);
                    st
                });
                smem.producer_tail(&mut st);
            },
            || {
                for_range_carry(0, rows, 1, smem.consumer_state(), |row, mut st| {
                    let si = StageInfo::new(row).with_bounds(0, rows, 1);
                    smem.try_wait(&st);
                    smem.wait(&st);
                    locals.store(&smem.run_work("read_smem", &smem.consumer_stage(&st, si.clone()), &WorkArgs::new()));
                    smem.release(&mut st);
                    output.run_work("store", &si, &WorkArgs::new().with("smem_val", locals.get(&smem_val)));
                    st
                });
            },
        );
    })
}

// ---------------------------------------------------------------------------
// 01_copy_persistent_skip_tile: a persistent copy over a work queue.
// ---------------------------------------------------------------------------

pub const PS_TILE: usize = 64;
pub const PS_ROWS: usize = 3;
pub const PS_LOGICAL_TILES: usize = 3;
pub const PS_LAUNCH_TILES: usize = 5;
pub const PS_COLS: usize = PS_TILE * PS_LOGICAL_TILES;
/// Two persistent CTAs for five tiles.
pub const PS_GRID: [u32; 3] = [1, 1, 2];

/// The schedule of `PS_LAUNCH_TILES` tiles, one CTA per cluster.
pub fn persistent_params() -> PersistentTileSchedulerParams {
    PersistentTileSchedulerParams::new((PS_LAUNCH_TILES, 1, 1), [1, 1, 1], Default::default())
}

/// Copy `a` to `b` (`PS_ROWS x PS_COLS` row-major i32) a column tile at a
/// time, over a static persistent work queue of `PS_LAUNCH_TILES` tiles
/// whose skip predicate drops the over-launched ones. `marks[tile]` is 1
/// for a copied tile and 2 for a skipped one (written by thread 0).
pub fn persistent_skip_copy() -> Kernel {
    let params = persistent_params();
    Kernel::builder("ts_persistent_skip").block_dim([PS_TILE as u32, 1, 1]).target(Arch::Sm89).build(move || {
        let (tile, rows, cols) = (PS_TILE as i32, PS_ROWS as i32, PS_COLS as i32);
        let a = param_tensor::<i32>("a", Layout::col_major(PS_ROWS * PS_COLS));
        let b = param_tensor::<i32>("b", Layout::col_major(PS_ROWS * PS_COLS));
        let marks = param_tensor::<i32>("marks", Layout::col_major(PS_LAUNCH_TILES));
        let mut copy = MemoryResource::new("CopyResource").work(Work::producer("copy_tile_row").run(move |si, _| {
            let [tile_x, _, _] = si.work_tile.expect("a work tile").tile_idx();
            let idx = si.loop_offset * cols + thread_idx().x + tile_x * tile;
            b.store(idx, a.load(idx));
            WorkResults::none()
        }));
        let mut wq =
            WorkQueue::new("WorkQueue", TileSchedulerConfig::create_static_persistent_tile_scheduler_params(params))
                .with_skip_if(|wt: &WorkTileInfo| wt.tile_idx()[0].ge(PS_LOGICAL_TILES as i32));
        copy.create();
        wq.create();
        let (first, _) = wq.init_work_tile(&StageInfo::new(0));
        while_carry(
            (first, wq.consumer_state()),
            |(t, _)| t.is_valid_tile(),
            |(t, mut st)| {
                let skip = wq.skip_work_tile_if(&t);
                let tile_x = t.tile_idx()[0];
                if_else(
                    skip,
                    || if_(thread_idx().x.eq(0), || marks.store(tile_x, 2)),
                    || {
                        for_range(0, rows, 1, |row| {
                            let si = StageInfo::new(row).with_bounds(0, rows, 1).with_work_tile(t);
                            copy.run_work("copy_tile_row", &si, &WorkArgs::new());
                        });
                        if_(thread_idx().x.eq(0), || marks.store(tile_x, 1));
                    },
                );
                wq.try_wait(&st);
                wq.wait(&st);
                let next = wq.get_and_advance_work_tile(&wq.consumer_stage(&st, StageInfo::new(0).with_work_tile(t)));
                wq.release(&mut st);
                (next, st)
            },
        );
    })
}

/// The inputs of the copies: `0, 1, 2, ...` scrambled a little.
pub fn ramp_i32(n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| i * 7 - 3).collect()
}

/// The `i16` input of the grid-stride copy.
pub fn ramp_i16(n: usize) -> Vec<i16> {
    (0..n as i32).map(|i| (i * 37 - 500) as i16).collect()
}

// ---------------------------------------------------------------------------
// 02_copy_tma through the TaskManager (roadmap part m11-ts-task-manager)
// ---------------------------------------------------------------------------

/// [`staged_copy`] written as two tasks for the TaskManager: `LoadTask` on
/// warp 4 (`compute_coords`, then `try_acquire` / `acquire` / `load` /
/// `commit` on the staged resource) and `StoreTask` on warps 0-3
/// (`try_wait` / `wait` / `read_smem` / `release`, then `store`). With
/// `racy`, the shared-memory resource has no pipeline (and the schedules no
/// pipeline operations), so the store warps read rows the load warp may
/// not have written yet: the static checks pass and the simulator catches
/// it.
pub fn managed_staged_copy(racy: bool) -> Kernel {
    let name = if racy { "ts_managed_staged_copy_racy" } else { "ts_managed_staged_copy" };
    Kernel::builder(name).block_dim([ST_THREADS, 1, 1]).target(Arch::Sm89).build(move || {
        let (tile, rows, cols) = (ST_TILE as i32, ST_ROWS as i32, ST_COLS as i32);
        let a = param_tensor::<i32>("a", Layout::col_major(ST_ROWS * ST_COLS));
        let b = param_tensor::<i32>("b", Layout::col_major(ST_ROWS * ST_COLS));
        let data = make_smem_tensor::<i32>(Layout::col_major(ST_STAGES * ST_TILE));
        let gmem_idx = TaskLocalVariable::<Int32>::scalar("gmem_idx", 0, "the first column of the CTA's tile");
        let smem_val = TaskLocalVariable::<Int32>::scalar("smem_val", 0, "the element this thread read");
        let input =
            MemoryResource::new("inputGmemResource").work(Work::consumer("compute_coords").returns(&gmem_idx).run({
                let v = gmem_idx.clone();
                move |_, _| WorkResults::of(&v, block_idx().x * tile)
            }));
        let mut smem = MemoryResource::new("smemResource")
            .work(Work::producer("load").input::<Int32>("gmem_idx").run({
                let data = data.clone();
                move |si, args| {
                    let col0: Int32 = args.get("gmem_idx");
                    let stage = si.stage_idx.unwrap_or(Int32::constant(0));
                    for k in 0..(ST_TILE / 32) as i32 {
                        let c = lane_idx() + k * 32;
                        data.store(stage * tile + c, a.load(si.loop_offset * cols + col0 + c));
                    }
                    WorkResults::none()
                }
            }))
            .work(Work::consumer("read_smem").returns(&smem_val).run({
                let v = smem_val.clone();
                move |si, _| {
                    let stage = si.stage_idx.unwrap_or(Int32::constant(0));
                    WorkResults::of(&v, data.load(stage * tile + thread_idx().x))
                }
            }));
        if !racy {
            smem = smem.with_pipeline_config(PipelineConfig::create_async_async_pipeline_cfg(
                ST_STAGES,
                CooperativeGroup::new(Agent::Thread, 32),
                CooperativeGroup::new(Agent::Thread, 128),
                None,
            ));
        }
        let output = MemoryResource::new("outputGmemResource").work(
            Work::producer("store").input::<Int32>("smem_val").run(move |si, args| {
                let col = block_idx().x * tile + thread_idx().x;
                b.store(si.loop_offset * cols + col, args.get::<Int32>("smem_val"));
                WorkResults::none()
            }),
        );
        let num_rows = Bound::runtime("num_rows", Int32::constant(rows));
        let load = schedule("load_schedule", &[&input, &smem], |s, h| {
            let (input, smem) = (h[0], h[1]);
            s.domain_loop(0, num_rows.clone(), 1, 1, |_| {
                let gmem_idx = input.call("compute_coords").one();
                if !racy {
                    smem.try_acquire();
                    smem.acquire();
                }
                smem.call_with("load", [("gmem_idx", &gmem_idx)]);
                if !racy {
                    smem.commit();
                }
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let store = schedule("store_schedule", &[&smem, &output], |s, h| {
            let (smem, output) = (h[0], h[1]);
            s.domain_loop(0, num_rows.clone(), 1, 1, |_| {
                if !racy {
                    smem.try_wait();
                    smem.wait();
                }
                let smem_val = smem.call("read_smem").one();
                if !racy {
                    smem.release();
                }
                output.call_with("store", [("smem_val", &smem_val)]);
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let load_task = Task::new("LoadTask", &[&input], &[&smem], 4, 1, load);
        let store_task = Task::new("StoreTask", &[&smem], &[&output], 0, 4, store);
        let graph = ResourceDependencyGraph::new().depends_on(&smem, [&input]).depends_on(&output, [&smem]);
        let mut tm =
            TaskManager::new(vec![load_task, store_task], graph).resource(input).resource(smem).resource(output);
        tm.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        tm.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

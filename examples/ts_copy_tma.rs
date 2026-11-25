// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   01_copy_basics_ts/02_copy_tma.py (the resources InputGmemResource,
//   SmemResource and OutputGmemResource, the schedules load_schedule,
//   store_schedule and padding_schedule, the tasks LoadTask, StoreTask and
//   PaddingTask, the SmemAllocator calls and the TaskManager calls of
//   `tma_copy_kernel`),
//     Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A CTA-local TMA copy written with Task Scheduling, ported from
//! `02_copy_tma.py`. Each CTA copies a 128-column strip of a row-major f16
//! matrix, one row per domain-loop iteration:
//!
//! * `inputGmemResource` returns the strip's first column (`gmem_idx`);
//! * `smemResource` is a one-stage `TmaAsync` pipeline over shared memory
//!   it declares (`SmemAllocation`), laid out by the declarative
//!   `SmemAllocator` together with the pipeline's barriers (its
//!   configuration has no barrier storage, so the allocator places it);
//!   its auxiliary work builds the typed view from the unified block's base
//!   (`stage_info.context.smem_base` plus the allocation's offset);
//! * `LoadTask` (warp 4) acquires the stage, whose acquire arms the full
//!   barrier with the row's 256 bytes, and an elected lane issues the TMA
//!   load; `StoreTask` (warps 0-3) waits, reads one element per thread,
//!   releases (every consumer thread releases: the consumer group counts
//!   the task's 128 threads, so the TaskManager makes the pipeline a
//!   `TsPipeline` with all-thread release) and stores to the output.
//!
//! `build(.., leader)` makes the load's producers signal as
//! `TASK_WARP_LEADER`: then only the load task's first warp arms each stage.
//! The kernel targets sm_90a (TMA), so it runs on the simulator here; the
//! tests also check its MLIR and PTX. Run with
//! `cargo run --release --example ts_copy_tma`.

use std::cell::RefCell;
use std::rc::Rc;

use cute::pipeline::{Agent, CooperativeGroup};
use cute::prelude::*;
use cute::task_scheduling::*;

/// Stages of the shared-memory pipeline (the tutorial's `num_stages`).
pub const NUM_STAGES: usize = 1;
/// Columns per CTA (the tutorial's `tile_size`).
pub const TILE: usize = 128;
/// The problem `main` and the tests run (the tutorial's default is 256 x 512).
pub const ROWS: usize = 6;
pub const COLS: usize = 256;

/// The copy kernel of a `rows x cols` f16 matrix (`cols` a multiple of
/// [`TILE`]).
pub fn build(rows: usize, cols: usize, leader: bool) -> Kernel {
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(TILE), "cols must be a positive multiple of {TILE}");
    let name = if leader { "ts_tma_copy_leader" } else { "ts_tma_copy" };
    Kernel::builder(name).block_dim([256, 1, 1]).target(Arch::Sm90a).build(move || {
        let a = param_tensor_aligned::<F16>("matrix_a", Layout::new((rows, cols), (cols, 1)), 16);
        // The output as a flat array (row-major offsets).
        let b = param_tensor::<F16>("matrix_b", Layout::col_major(rows * cols));
        // The TMA descriptor of a (1, TILE) box.
        let row_tile = Layout::new((1usize, TILE), (TILE, 1usize));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_a) = cpasync::make_tiled_tma_atom(g2s, &a, row_tile.clone(), (1, TILE), 1).into();
        // Prefetch the TMA descriptor (the barriers belong to the SmemAllocator).
        if_(elect_one(), || cpasync::prefetch_descriptor(&atom));
        let num_warps_epilogue = 4;
        let mut pipeline_config = PipelineConfig::create_tma_async_pipeline_cfg(
            NUM_STAGES,
            (TILE * 2) as u32,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, num_warps_epilogue * 32),
        );
        if leader {
            pipeline_config =
                pipeline_config.with_signaling_threads(SignalingThreads::TASK_WARP_LEADER, SignalingThreads::ALL);
        }

        // inputGmemResource: the strip's first column.
        let gmem_idx = TaskLocalVariable::<Int32>::scalar("gmem_idx", 0, "the first column of the CTA's strip");
        let input_gmem_resource =
            MemoryResource::new("inputGmemResource").work(Work::consumer("compute_coords").returns(&gmem_idx).run({
                let v = gmem_idx.clone();
                move |_, _| WorkResults::of(&v, block_idx().x * TILE as i32)
            }));

        // smemResource: its shared memory comes from the allocator, its typed
        // view from the context.
        let smem_val = TaskLocalVariable::<Float16>::scalar("smem_val", F16::from_f32(0.0), "the element read");
        let alloc_smem = SmemAllocation::new::<F16>("smem_data", NUM_STAGES * TILE, 128);
        let view: Rc<RefCell<Option<Ptr<F16>>>> = Rc::default();
        let init = {
            let (alloc, view) = (alloc_smem.clone(), view.clone());
            move |si: &StageInfo, _: &WorkArgs| {
                *view.borrow_mut() = Some(alloc.ptr::<F16>(&si.context));
                WorkResults::none()
            }
        };
        let smem = move |view: &Rc<RefCell<Option<Ptr<F16>>>>| view.borrow().clone().expect("init_*_state ran first");
        let smem_resource = MemoryResource::new("smemResource")
            .with_pipeline_config(pipeline_config)
            .smem_requirement(alloc_smem)
            .work(Work::producer("init_load_state").auxiliary().run(init.clone()))
            .work(Work::consumer("init_read_state").auxiliary().run(init))
            .work(Work::consumer("read_smem").returns(&smem_val).run({
                let (v, view) = (smem_val.clone(), view.clone());
                move |si, _| {
                    let stage = si.stage_idx.expect("a pipelined stage");
                    let data: Float16 = smem(&view).offset(stage * TILE as i32 + thread_idx().x).load();
                    WorkResults::of(&v, data)
                }
            }))
            .work(Work::producer("tma_load").input::<Int32>("gmem_idx").run(move |si, args| {
                let col: Int32 = args.get("gmem_idx");
                let stage = si.stage_idx.expect("a pipelined stage");
                let s = make_tensor(&smem(&view).offset(stage * TILE as i32), row_tile.clone());
                let g = local_tile(&tma_a, (1, TILE), (si.loop_offset, col / TILE as i32));
                let (ts, tg) =
                    cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
                let bar = si.barrier().expect("the stage's full barrier");
                if_(elect_one(), || copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&bar)));
                WorkResults::none()
            }));

        // outputGmemResource: a pure sink.
        let output_gmem_resource = MemoryResource::new("outputGmemResource").work(
            Work::producer("store").input::<Float16>("smem_val").run(move |si, args| {
                let m_idx = thread_idx().x + block_idx().x * TILE as i32;
                b.store(si.loop_offset * cols as i32 + m_idx, args.get::<Float16>("smem_val"));
                WorkResults::none()
            }),
        );

        let mut allocator = SmemAllocator::new();
        allocator.add_resource(&smem_resource);
        allocator.compute_layout().unwrap_or_else(|e| panic!("{e}"));

        let num_rows = rows as i64;
        let load_schedule = schedule("load_schedule", &[&input_gmem_resource, &smem_resource], |s, r| {
            let (input_gmem, smem) = (r[0], r[1]);
            smem.call("init_load_state");
            s.domain_loop(0, num_rows, 1, 1, |_| {
                let gmem_idx = input_gmem.call("compute_coords").one();
                smem.try_acquire();
                smem.acquire();
                smem.call_with("tma_load", [("gmem_idx", &gmem_idx)]);
                smem.commit();
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let store_schedule = schedule("store_schedule", &[&smem_resource, &output_gmem_resource], |s, r| {
            let (smem, output_gmem) = (r[0], r[1]);
            smem.call("init_read_state");
            s.domain_loop(0, num_rows, 1, 1, |_| {
                smem.try_wait();
                smem.wait();
                let smem_val = smem.call("read_smem").one();
                smem.release();
                output_gmem.call_with("store", [("smem_val", &smem_val)]);
            });
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let padding_schedule = schedule("padding_schedule", &[], |s, _| s.domain_loop(0, num_rows, 1, 1, |_| {}))
            .unwrap_or_else(|e| panic!("{e}"));

        // The register budgets are the tutorial's `num_registers`: the load and padding tasks share
        // warpgroup 1, so they take the same count.
        let load_task = Task::new("LoadTask", &[&input_gmem_resource], &[&smem_resource], 4, 1, load_schedule)
            .setmaxnreg_decrease(40);
        let store_task = Task::new(
            "StoreTask",
            &[&smem_resource],
            &[&output_gmem_resource],
            0,
            num_warps_epilogue as u32,
            store_schedule,
        )
        .setmaxnreg_decrease(160);
        let padding_task = Task::new("PaddingTask", &[], &[], 5, 3, padding_schedule).setmaxnreg_decrease(40);
        let resource_dependency_graph = ResourceDependencyGraph::new()
            .depends_on(&smem_resource, [&input_gmem_resource])
            .depends_on(&output_gmem_resource, [&smem_resource]);
        let mut task_manager = TaskManager::new(vec![load_task, store_task, padding_task], resource_dependency_graph)
            .resource(input_gmem_resource)
            .resource(smem_resource)
            .resource(output_gmem_resource)
            .smem_allocator(allocator);
        task_manager.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        task_manager.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

/// One CTA per strip, 8 warps.
pub fn launch(cols: usize) -> LaunchConfig {
    LaunchConfig::new([(cols / TILE) as u32, 1, 1], [256, 1, 1])
}

/// The source: scrambled small values, exact in f16.
pub fn source(rows: usize, cols: usize) -> Vec<F16> {
    (0..rows * cols).map(|i| F16::from_f32(((i * 7919 + 13) % 2001) as f32 - 1000.0)).collect()
}

/// Copy on the simulator, check against plain Rust (the copy is the
/// source), and return the copy.
pub fn run(rows: usize, cols: usize, leader: bool) -> Vec<F16> {
    let src = source(rows, cols);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![F16::from_f32(0.0); rows * cols])];
    build(rows, cols, leader).simulate(launch(cols), &mut args).unwrap_or_else(|e| panic!("{e}"));
    let out = args[1].to_vec::<F16>();
    assert_eq!(out, src);
    out
}

#[allow(dead_code)]
fn main() {
    run(ROWS, COLS, false);
    run(ROWS, COLS, true);
    println!("ts_copy_tma: copied a {ROWS} x {COLS} f16 matrix through TMA and a TaskManager: PASS");
}

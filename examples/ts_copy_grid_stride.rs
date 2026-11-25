// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/
//   01_copy_basics_ts/01_copy_grid_stride.py (the resources InputGmemResource
//   and OutputGmemResource, the schedule `schedule_fn`, the task `CopyTask`,
//   the dependency graph and the TaskManager calls of `gmem_copy_kernel`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A grid-stride GMEM copy written with Task Scheduling, ported from
//! `01_copy_grid_stride.py`: one task (`CopyTask`) on every warp of the block
//! runs a schedule whose domain loop is the grid-stride range
//! `range(block_idx.x * threads + thread_idx.x, num_entries, grid_dim.x *
//! threads)`; each iteration calls the input resource's consumer work
//! `get_item` (which returns the task-local variable `item`) and passes it to
//! the output resource's producer work `set_item`. There is no pipeline: the
//! value flows through the task's own storage. The TaskManager checks the
//! one edge of the dependency graph (`outputGmemResource` depends on
//! `inputGmemResource`) and emits the kernel.
//!
//! Everything is SM80-level, so it runs on the simulator and the SM89 GPU.
//! Run with `cargo run --release --example ts_copy_grid_stride`.

use cute::prelude::*;
use cute::task_scheduling::*;

/// The problem the test and `main` run (the tutorial's default is 2^24
/// entries over 256 blocks of 8 warps).
pub const NUM_ENTRIES: usize = 1000;
pub const NUM_BLOCKS: u32 = 3;
pub const NUM_WARPS: u32 = 2;

/// The copy kernel of `num_entries` `i16`s on blocks of `num_warps` warps,
/// with the domain loop unrolled `unroll` times and, with `iket`, IKET
/// ranges around the task and each step (the kernel is then built with
/// those IKET options).
pub fn build(num_entries: usize, num_warps: u32, unroll: u32, iket: Option<IketOptions>) -> Kernel {
    let threads = num_warps * 32;
    let mut options = Kernel::builder("ts_gmem_copy").block_dim([threads, 1, 1]).target(Arch::Sm89);
    if let Some(o) = iket {
        options = options.iket(o);
    }
    options.build(move || {
        let n = num_entries as i32;
        let source = param_tensor::<i16>("source_tensor", Layout::col_major(num_entries));
        let destination = param_tensor::<i16>("destination_tensor", Layout::col_major(num_entries));

        // Resource construction: the source value flows straight to the
        // destination, so neither resource needs a pipeline.
        let item = TaskLocalVariable::<Int16>::scalar("item", 0, "the element of the current grid-stride iteration");
        let input = MemoryResource::new("inputGmemResource").work(Work::consumer("get_item").returns(&item).run({
            let item = item.clone();
            move |si, _| {
                let gid = si.loop_offset;
                let val = if_else_yield(gid.lt(n), || source.load(gid), || Int16::constant(0));
                WorkResults::of(&item, val)
            }
        }));
        let output = MemoryResource::new("outputGmemResource").work(
            Work::producer("set_item").input::<Int16>("data").run(move |si, args| {
                let gid = si.loop_offset;
                let data: Int16 = args.get("data");
                if_(gid.lt(n), || destination.store(gid, data));
                WorkResults::none()
            }),
        );

        // The schedule: a grid-stride domain loop.
        let threads = threads as i32;
        let start = block_idx().x * threads + thread_idx().x;
        let step = grid_dim().x * threads;
        let schedule_fn = schedule("schedule_fn", &[&input, &output], |s, r| {
            let (input, output) = (r[0], r[1]);
            s.domain_loop(
                Bound::runtime("start", start),
                Bound::runtime("num_entries", Int32::constant(n)),
                Bound::runtime("step", step),
                unroll,
                |_| {
                    let res = input.call("get_item").one();
                    output.call_with("set_item", [("data", &res)]);
                },
            );
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let task = Task::new("CopyTask", &[&input], &[&output], 0, num_warps, schedule_fn);

        // The TaskManager checks the graph and emits the schedule.
        let graph = ResourceDependencyGraph::new().depends_on(&output, [&input]);
        let mut task_manager =
            TaskManager::new(vec![task], graph).resource(input).resource(output).iket_enable_profiling(iket.is_some());
        task_manager.setup_resources_and_tasks().unwrap_or_else(|e| panic!("{e}"));
        task_manager.run().unwrap_or_else(|e| panic!("{e}"));
    })
}

/// The launch: `num_blocks` blocks of `num_warps` warps.
pub fn launch(num_blocks: u32, num_warps: u32) -> LaunchConfig {
    LaunchConfig::new([num_blocks, 1, 1], [num_warps * 32, 1, 1])
}

/// The source: `num_entries` scrambled `i16`s.
pub fn source(num_entries: usize) -> Vec<i16> {
    (0..num_entries as i64).map(|i| ((i * 7919 + 13) % 65536 - 32768) as i16).collect()
}

/// The plain-Rust copy: the reference.
pub fn reference(source: &[i16]) -> Vec<i16> {
    source.to_vec()
}

/// Copy on the simulator and check against the reference; returns the copy.
pub fn run(num_entries: usize, num_blocks: u32, num_warps: u32) -> Vec<i16> {
    let src = source(num_entries);
    let mut args = vec![KernelArg::buffer(&src), KernelArg::buffer(&vec![0i16; num_entries])];
    build(num_entries, num_warps, 1, None)
        .simulate(launch(num_blocks, num_warps), &mut args)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = args[1].to_vec::<i16>();
    assert_eq!(out, reference(&src));
    out
}

fn main() {
    run(NUM_ENTRIES, NUM_BLOCKS, NUM_WARPS);
    println!("ts_copy_grid_stride: copied {NUM_ENTRIES} entries with {NUM_BLOCKS} blocks of {NUM_WARPS} warps: PASS");
}

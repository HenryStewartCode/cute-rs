//! CUDA graphs: capture launches on a stream once, then replay them with one call.
//!
//! A port of CUTLASS 4.8.1's `examples/python/CuTeDSL/cute/notebooks/cuda_graphs.ipynb`,
//! Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! BSD-3-Clause; see THIRD-PARTY-NOTICES.md. The notebook's kernel prints "Hello world"; this one
//! counts its runs with an atomic instead, so that the replays can be checked, and the times
//! come from the host clock rather than from
//! PyTorch's CUDA events.
//!
//! ```text
//! cargo run --release --features cuda --example cuda_graphs
//! ```

use std::error::Error;
use std::time::Instant;

use cute::cuda::{CaptureMode, CudaContext, DeviceArg};
use cute::prelude::*;

/// One thread that adds one to a counter.
pub fn hello_world_kernel() -> Kernel {
    Kernel::builder("hello_world_kernel").block_dim([1, 1, 1]).build(|| {
        let count = param_tensor::<i32>("count", Layout::col_major((1,)));
        let _ = atomic_add(&count.ptr(), 1i32);
    })
}

/// What [`run`] measured.
pub struct GraphRun {
    /// Seconds for `launches` launches queued one by one on a stream.
    pub stream_seconds: f64,
    /// Seconds for one replay of a graph of `launches` launches.
    pub graph_seconds: f64,
    /// The counter after everything: the kernel's runs.
    pub count: i32,
    /// The runs the launches and replays queued.
    pub expected: i32,
}

/// Capture two launches and replay them; then time `launches` launches on a stream against
/// one replay of a graph that captured as many.
pub fn run(ctx: &CudaContext, toolchain: &Toolchain, launches: usize) -> Result<GraphRun, Box<dyn Error>> {
    let kernel = hello_world_kernel();
    let arch = ctx.best_target(toolchain).ok_or("no target for this device")?;
    let compiled = kernel.compile_with(toolchain, &CompileOptions::new().gpu_arch(arch))?;
    let function = ctx.load(&compiled)?;
    let stream = ctx.stream()?;
    let count = ctx.alloc::<i32>(1)?;
    count.memset(0)?;
    let args = [DeviceArg::Buffer(&count)];
    let cfg = LaunchConfig::new([1, 1, 1], [1, 1, 1]);
    let launch = || function.launch_builder(cfg)?.stream(&stream).args(&args).submit();

    // A graph of two launches, replayed once.
    let capture = stream.begin_capture(CaptureMode::Global)?;
    for _ in 0..2 {
        launch()?;
    }
    let small = capture.end()?.instantiate()?;
    small.launch(&stream)?;
    stream.synchronize()?;

    // A graph of `launches` launches.
    let capture = stream.begin_capture(CaptureMode::Global)?;
    for _ in 0..launches {
        launch()?;
    }
    let graph = capture.end()?;
    let big = graph.instantiate()?;

    // Warm up, then time the launches one by one.
    for _ in 0..launches {
        launch()?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..launches {
        launch()?;
    }
    stream.synchronize()?;
    let stream_seconds = start.elapsed().as_secs_f64();

    // Warm up, then time one replay.
    big.launch(&stream)?;
    stream.synchronize()?;
    let start = Instant::now();
    big.launch(&stream)?;
    stream.synchronize()?;
    let graph_seconds = start.elapsed().as_secs_f64();

    let count = count.to_vec::<i32>()?[0];
    Ok(GraphRun { stream_seconds, graph_seconds, count, expected: (2 + 4 * launches) as i32 })
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let toolchain = Toolchain::discover().ok_or("mlir-opt not found (set CUTE_MLIR_OPT)")?;
    let r = run(&ctx, &toolchain, 100)?;
    assert_eq!(r.count, r.expected);
    println!("{} kernel runs, as queued", r.count);
    println!("100 launches on a stream: {:.3} ms", r.stream_seconds * 1e3);
    println!("a graph of 100 launches:  {:.3} ms", r.graph_seconds * 1e3);
    let speedup = (r.stream_seconds - r.graph_seconds) / r.graph_seconds;
    println!("{:.2}% speedup when using CUDA graphs for this kernel!", speedup * 100.0);
    Ok(())
}

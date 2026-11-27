//! Programmatic dependent launch, launch events, CUDA events and cooperative launch on a real
//! GPU. Needs the `cuda` feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_pdl_events
//! ```
//!
//! What the SM89 test GPU (driver 595) does with the SM90 features is printed and asserted:
//! see each test.

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/cooperative_launch.rs"]
mod cooperative_launch;

use cute::cuda::{CudaContext, CudaEvent, CudaFunction, DeviceArg, EventFlags};
use cute::prelude::*;

const N: usize = 4096;

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn load<'c>(ctx: &'c CudaContext, tc: &Toolchain, k: &Kernel) -> CudaFunction<'c> {
    let arch = ctx.best_target(tc).expect("a target");
    let compiled = k.compile_with(tc, &CompileOptions::new().gpu_arch(arch)).unwrap_or_else(|e| panic!("{e}"));
    ctx.load(&compiled).unwrap_or_else(|e| panic!("{e}"))
}

/// `x[i] = i` after `iters` iterations of a loop the store depends on, so the kernel runs
/// long enough to time and to race with.
fn slow_fill() -> Kernel {
    Kernel::builder("slow_fill").block_dim([128, 1, 1]).build(|| {
        let iters = param_scalar::<i32>("iters");
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let i = block_idx().x * 128 + thread_idx().x;
        let acc = for_range_carry(0, iters, 1, Float32::from(0.0f32), |_, acc| acc * 0.5f32 + 1.0f32);
        x.store(i, i.to::<f32>() + select(acc.gt(100.0f32), 1.0f32, 0.0f32));
    })
}

/// `y[i] = x[i] + 1`.
fn plus_one() -> Kernel {
    Kernel::builder("plus_one").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(N));
        let y = param_tensor::<f32>("y", Layout::col_major(N));
        let i = block_idx().x * 128 + thread_idx().x;
        y.store(i, x.load(i) + 1.0f32);
    })
}

fn cfg() -> LaunchConfig {
    LaunchConfig::new([(N / 128) as u32, 1, 1], [128, 1, 1])
}

fn want(offset: f32) -> Vec<f32> {
    (0..N).map(|i| i as f32 + offset).collect()
}

/// Two kernels back to back in one stream, the second with `use_pdl`
/// (`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`). The SM89 GPU has no
/// griddepcontrol (sm_90), so the kernels have none; the driver accepts the attribute and
/// runs the pair in order.
#[test]
fn pdl_launches_on_this_device() {
    let Some((ctx, tc)) = device() else { return };
    let (fill, add) = (slow_fill(), plus_one());
    let (f_fill, f_add) = (load(&ctx, &tc, &fill), load(&ctx, &tc, &add));
    let stream = ctx.stream().unwrap();
    let (x, y) = (ctx.alloc::<f32>(N).unwrap(), ctx.alloc::<f32>(N).unwrap());
    x.memset(0).unwrap();
    let fill_args = [DeviceArg::Scalar(KernelArg::scalar(20000i32)), DeviceArg::Buffer(&x)];
    let add_args = [DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)];
    f_fill.launch_builder(cfg().with_pdl(true)).unwrap().stream(&stream).args(&fill_args).submit().unwrap();
    let r = f_add.launch_builder(cfg().with_pdl(true)).unwrap().stream(&stream).args(&add_args).submit();
    let cc = ctx.compute_capability();
    eprintln!("PDL launch on sm_{}{} (driver {:?}): {r:?}", cc.0, cc.1, ctx.driver_version());
    r.unwrap_or_else(|e| panic!("the driver refused the PDL attribute: {e}"));
    stream.synchronize().unwrap();
    // Without griddepcontrol.wait, an SM90 device could start the second kernel early; this
    // one runs them in order, so the second sees the first's writes.
    if cc.0 < 9 {
        assert_eq!(y.to_vec::<f32>().unwrap(), want(1.0));
    }
}

/// Events around a kernel time it; `cuEventQuery` reports pending and complete work.
#[test]
fn events_time_a_kernel() {
    let Some((ctx, tc)) = device() else { return };
    let fill = slow_fill();
    let f = load(&ctx, &tc, &fill);
    let stream = ctx.stream().unwrap();
    let x = ctx.alloc::<f32>(N).unwrap();
    let (start, end) = (ctx.event(EventFlags::DEFAULT).unwrap(), ctx.event(EventFlags::DEFAULT).unwrap());
    assert!(end.is_complete().unwrap(), "an event never recorded is complete");
    let args = [DeviceArg::Scalar(KernelArg::scalar(200000i32)), DeviceArg::Buffer(&x)];
    start.record(&stream).unwrap();
    f.launch_builder(cfg()).unwrap().stream(&stream).args(&args).submit().unwrap();
    end.record(&stream).unwrap();
    let pending = !end.is_complete().unwrap();
    end.synchronize().unwrap();
    assert!(end.is_complete().unwrap());
    let ms = end.elapsed_ms_since(&start).unwrap();
    eprintln!("slow_fill: {ms:.3} ms (pending right after the record: {pending})");
    assert!(ms > 0.0, "elapsed {ms} ms");
    assert_eq!(x.to_vec::<f32>().unwrap(), want(0.0));
    let blocking = ctx.event(EventFlags { blocking_sync: true, disable_timing: true }).unwrap();
    blocking.record(&stream).unwrap();
    blocking.synchronize().unwrap();
    let e = end.elapsed_ms_since(&blocking).unwrap_err();
    assert!(e.message.contains("cuEventElapsedTime"), "{e}");
}

/// `cuStreamWaitEvent` orders a launch on one stream after a slow one on another.
#[test]
fn stream_wait_event_orders_two_streams() {
    let Some((ctx, tc)) = device() else { return };
    let (fill, add) = (slow_fill(), plus_one());
    let (f_fill, f_add) = (load(&ctx, &tc, &fill), load(&ctx, &tc, &add));
    let (s1, s2) = (ctx.stream().unwrap(), ctx.stream().unwrap());
    for _ in 0..3 {
        let (x, y) = (ctx.alloc::<f32>(N).unwrap(), ctx.alloc::<f32>(N).unwrap());
        x.memset(0).unwrap();
        let done = ctx.event(EventFlags::DISABLE_TIMING).unwrap();
        let fill_args = [DeviceArg::Scalar(KernelArg::scalar(100000i32)), DeviceArg::Buffer(&x)];
        f_fill.launch_builder(cfg()).unwrap().stream(&s1).args(&fill_args).submit().unwrap();
        done.record(&s1).unwrap();
        s2.wait_event(&done).unwrap();
        let add_args = [DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)];
        f_add.launch_builder(cfg()).unwrap().stream(&s2).args(&add_args).submit().unwrap();
        s2.synchronize().unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), want(1.0));
        s1.synchronize().unwrap();
    }
}

/// `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT` and `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`
/// on this device: what the driver does is printed. On the SM89 GPU (driver 595, CUDA 13.2)
/// both are accepted and complete (with no griddepcontrol, the programmatic event fires as
/// the blocks exit); an event with timing is refused with CUDA_ERROR_NOT_SUPPORTED (801).
#[test]
fn launch_events_on_this_device() {
    let Some((ctx, tc)) = device() else { return };
    let fill = slow_fill();
    let f = load(&ctx, &tc, &fill);
    let stream = ctx.stream().unwrap();
    let x = ctx.alloc::<f32>(N).unwrap();
    let args = [DeviceArg::Scalar(KernelArg::scalar(20000i32)), DeviceArg::Buffer(&x)];
    let completion = ctx.event(EventFlags::DISABLE_TIMING).unwrap();
    let r =
        f.launch_builder(cfg()).unwrap().stream(&stream).args(&args).launch_completion_event(&completion, 0).submit();
    eprintln!("launch-completion event: {r:?}");
    r.unwrap_or_else(|e| panic!("{e}"));
    completion.synchronize().unwrap();
    stream.synchronize().unwrap();
    assert!(completion.is_complete().unwrap());
    let programmatic: CudaEvent<'_> = ctx.event(EventFlags::DISABLE_TIMING).unwrap();
    let r = f
        .launch_builder(cfg())
        .unwrap()
        .stream(&stream)
        .args(&args)
        .programmatic_event(&programmatic, 0, false)
        .submit();
    eprintln!("programmatic event (trigger at block exit): {r:?}");
    if r.is_ok() {
        stream.synchronize().unwrap();
        assert!(programmatic.is_complete().unwrap());
    }
    // A timing event is refused by the driver (the attribute needs CU_EVENT_DISABLE_TIMING).
    let timed = ctx.event(EventFlags::DEFAULT).unwrap();
    let r = f.launch_builder(cfg()).unwrap().stream(&stream).args(&args).launch_completion_event(&timed, 0).submit();
    eprintln!("launch-completion event with timing: {r:?}");
    assert_eq!(r.unwrap_err().code, Some(801), "CUDA_ERROR_NOT_SUPPORTED");
    stream.synchronize().unwrap();
}

/// `cooperative_launch.py`'s kernel with a grid far too large to be resident at once: the
/// occupancy check refuses it before the driver's CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE.
#[test]
fn a_too_large_cooperative_grid_is_refused_clearly() {
    let Some((ctx, tc)) = device() else { return };
    assert_eq!(ctx.device_attribute(cute::cuda::DeviceAttribute::CooperativeLaunch).unwrap(), 1);
    let kernel = cooperative_launch::build();
    let f = load(&ctx, &tc, &kernel);
    let per_sm = f.max_active_blocks_per_multiprocessor(128, 0).unwrap();
    assert!(per_sm >= 1, "{per_sm}");
    let bufs: Vec<_> = cooperative_launch::args().iter().map(|a| ctx.upload(a).unwrap()).collect();
    let args: Vec<DeviceArg<'_>> = bufs.iter().map(DeviceArg::Buffer).collect();
    let big = LaunchConfig::new([65536, 1, 1], cooperative_launch::BLOCK).with_cooperative(true);
    let e = ctx.launch_on_device(&kernel, &f, big, &args).unwrap_err();
    eprintln!("{e}");
    assert!(
        e.message.contains("launched cooperatively with 65536 blocks")
            && e.message.contains("cuOccupancyMaxActiveBlocksPerMultiprocessor"),
        "{e}"
    );
    assert_eq!(e.code, None, "refused before the driver");
    // The example's grid as a cooperative launch.
    ctx.launch_on_device(&kernel, &f, cooperative_launch::launch(), &args).unwrap();
    let mut out = cooperative_launch::args();
    for (b, a) in bufs.iter().zip(out.iter_mut()) {
        ctx.download(b, a).unwrap();
    }
    cooperative_launch::check(&out);
}

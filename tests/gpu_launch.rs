//! Streams, driver helpers, graphs, the parameter ABI and launch attributes on a real GPU.
//! Needs the `cuda` feature; skipped (with a note) without an NVIDIA GPU or an `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_launch
//! ```

// The examples are included as modules: their `main`s, and the helpers this file does not call, are unused here.
#![allow(dead_code)]

#[path = "../examples/cuda_graphs.rs"]
mod cuda_graphs;

mod abi_kernels;

use abi_kernels::*;
use cute::cuda::{
    CaptureMode, CudaContext, CudaFunction, CudaStream, DeviceArg, DeviceAttribute, DeviceBuffer, FuncAttribute,
};
use cute::prelude::*;

/// A capture in `CaptureMode::Global` makes every thread's unsafe calls (allocations,
/// synchronisation) fail while it lasts, so the tests of this file run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// `y[i] = x[i] * a + y[i]` over 1024 elements.
fn axpy() -> Kernel {
    Kernel::builder("axpy").block_dim([128, 1, 1]).build(|| {
        let a = param_scalar::<f32>("a");
        let x = param_tensor::<f32>("x", Layout::col_major((1024,)));
        let y = param_tensor::<f32>("y", Layout::col_major((1024,)));
        let i = block_idx().x * 128 + thread_idx().x;
        y.store(i, x.load(i) * a + y.load(i));
    })
}

fn axpy_cfg() -> LaunchConfig {
    LaunchConfig::new([8, 1, 1], [128, 1, 1])
}

fn data(seed: u32) -> Vec<f32> {
    (0..1024).map(|i| ((i * 7 + seed) % 23) as f32 - 11.0).collect()
}

#[test]
fn launches_on_two_streams_match_the_simulator() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = axpy();
    let f = load(&ctx, &tc, &k);
    let (s1, s2) = (ctx.stream().unwrap(), ctx.stream().unwrap());
    let mut sims = Vec::new();
    let mut bufs: Vec<(DeviceBuffer<'_>, DeviceBuffer<'_>)> = Vec::new();
    for seed in [1, 2] {
        let mut args =
            vec![KernelArg::ScalarF32(seed as f32), KernelArg::f32(data(seed)), KernelArg::f32(data(seed + 5))];
        bufs.push((ctx.upload_slice(&data(seed)).unwrap(), ctx.upload_slice(&data(seed + 5)).unwrap()));
        k.simulate(axpy_cfg(), &mut args).unwrap();
        k.simulate(axpy_cfg(), &mut args).unwrap();
        sims.push(args[2].as_f32().to_vec());
    }
    for (i, stream) in [&s1, &s2].into_iter().enumerate() {
        let (x, y) = &bufs[i];
        let args =
            [DeviceArg::Scalar(KernelArg::ScalarF32((i + 1) as f32)), DeviceArg::Buffer(x), DeviceArg::Buffer(y)];
        for _ in 0..2 {
            f.launch_builder(axpy_cfg()).unwrap().stream(stream).args(&args).submit().unwrap();
        }
    }
    s1.synchronize().unwrap();
    s2.synchronize().unwrap();
    for (i, (_, y)) in bufs.iter().enumerate() {
        assert_eq!(y.to_vec::<f32>().unwrap(), sims[i], "stream {i}");
    }
}

#[cfg(unix)]
mod raw_driver {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    pub type Create = unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int;
    pub type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
    /// `cuStreamCreate` and `cuStreamDestroy_v2`, called directly.
    pub fn stream_api() -> (Create, Destroy) {
        // SAFETY: the driver library and two entry points with these signatures (cuda.h).
        unsafe {
            let h = dlopen(c"libcuda.so.1".as_ptr(), 2);
            assert!(!h.is_null());
            let create = dlsym(h, c"cuStreamCreate".as_ptr());
            let destroy = dlsym(h, c"cuStreamDestroy_v2".as_ptr());
            assert!(!create.is_null() && !destroy.is_null());
            (std::mem::transmute::<*mut c_void, Create>(create), std::mem::transmute::<*mut c_void, Destroy>(destroy))
        }
    }
}

#[cfg(unix)]
#[test]
fn a_stream_from_the_raw_driver_api() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = axpy();
    let f = load(&ctx, &tc, &k);
    let (create, destroy) = raw_driver::stream_api();
    let mut raw = std::ptr::null_mut();
    // SAFETY: the context is current (CudaContext::new made it so); a valid out-pointer.
    assert_eq!(unsafe { create(&mut raw, 1) }, 0);
    {
        // SAFETY: a live stream of this context, destroyed below after this is dropped.
        let stream = unsafe { CudaStream::from_raw(&ctx, raw) };
        assert_eq!(stream.as_raw(), raw);
        let (x, y) = (ctx.upload_slice(&data(3)).unwrap(), ctx.upload_slice(&data(4)).unwrap());
        let args = [DeviceArg::Scalar(KernelArg::ScalarF32(2.0)), DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)];
        ctx.launch_on(&stream, &k, &f, axpy_cfg(), &args).unwrap();
        stream.synchronize().unwrap();
        assert!(stream.is_idle().unwrap());
        let want: Vec<f32> = data(3).iter().zip(data(4)).map(|(a, b)| a * 2.0 + b).collect();
        assert_eq!(y.to_vec::<f32>().unwrap(), want);
    }
    // Not destroyed by the CudaStream: destroying it here succeeds.
    // SAFETY: the stream is live and no longer used.
    assert_eq!(unsafe { destroy(raw) }, 0);
}

#[test]
fn pinned_async_copies_round_trip() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let src = ctx.pinned_from(&data(9)).unwrap();
    let buf = ctx.alloc::<f32>(1024).unwrap();
    let mut back = ctx.alloc_pinned::<f32>(512).unwrap();
    stream.upload_async(&src, &buf, 0).unwrap().wait().unwrap();
    stream.download_async(&buf, 2048, &mut back).unwrap().wait().unwrap();
    assert_eq!(&back[..], &data(9)[512..]);
    // Device-to-device and memset in stream order.
    let other = ctx.alloc::<f32>(1024).unwrap();
    stream.memset_async(&other, 0).unwrap();
    stream.copy_async(&other, 0, &buf, 4, 8).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(&other.to_vec::<f32>().unwrap()[..3], &[data(9)[1], data(9)[2], 0.0]);
}

#[test]
fn device_info_and_attributes() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _)) = device() else { return };
    let info = ctx.device_info().unwrap();
    println!("{info}");
    assert!(CudaContext::device_count().unwrap() >= 1);
    assert!(info.sm_count > 0 && info.l2_cache_bytes > 0 && info.total_memory > (1 << 30));
    assert!(info.shared_memory_per_sm >= info.shared_memory_per_block_optin);
    assert!(info.driver_version >= 12000);
    assert_eq!(ctx.device_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap(), 1024);
    assert_eq!(ctx.device_attribute(DeviceAttribute::WarpSize).unwrap(), 32);
    assert!(ctx.device_attribute(DeviceAttribute::TotalConstantMemory).unwrap() == 65536);
    assert!(ctx.device_attribute(DeviceAttribute::MaxRegistersPerBlock).unwrap() >= 32768);
    if info.compute_capability == (8, 9) {
        assert_eq!(info.shared_memory_per_block_optin, 101_376);
        assert_eq!(info.shared_memory_per_sm, 102_400);
        assert!(!info.cluster_launch);
        assert_eq!(ctx.max_shared_memory_per_block(), 101_376);
    }
    let text = info.to_string();
    assert!(text.contains(&info.name) && text.contains("SMs: ") && text.contains("driver: CUDA 1"), "{text}");
}

#[test]
fn allocation_memset_and_copies_at_offsets() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _)) = device() else { return };
    let buf = ctx.alloc::<i32>(64).unwrap();
    assert_eq!(buf.bytes(), 256);
    buf.memset(0xff).unwrap();
    assert!(buf.to_vec::<i32>().unwrap().iter().all(|&v| v == -1));
    buf.write(16, &[1i32, 2, 3]).unwrap();
    let mut out = [0i32; 5];
    buf.read(12, &mut out).unwrap();
    assert_eq!(out, [-1, 1, 2, 3, -1]);
    let other = ctx.alloc_bytes(256).unwrap();
    other.memset(0).unwrap();
    other.copy_from(100, &buf, 16, 12).unwrap();
    assert_eq!(&other.to_vec::<u8>().unwrap()[100..104], &1i32.to_le_bytes());
    let e = buf.write(250, &[1i32, 2]).unwrap_err().to_string();
    assert!(e.contains("a write of 8 bytes at byte 250 is outside the buffer of 256 bytes"), "{e}");
    if ctx.device_attribute(DeviceAttribute::MemoryPoolsSupported).unwrap() != 0 {
        let stream = ctx.stream().unwrap();
        let pooled = ctx.alloc_async(1024, &stream).unwrap();
        stream.memset_async(&pooled, 7).unwrap();
        stream.synchronize().unwrap();
        assert!(pooled.to_vec::<u8>().unwrap().iter().all(|&b| b == 7));
        pooled.free_async(&stream).unwrap();
        stream.synchronize().unwrap();
    }
}

/// `x = x * 2 + 1` then `y = y + x`: two dependent launches.
fn graph_kernels() -> (Kernel, Kernel) {
    let double = Kernel::builder("double_plus_one").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((1024,)));
        let i = block_idx().x * 128 + thread_idx().x;
        x.store(i, x.load(i) * 2.0f32 + 1.0f32);
    });
    let accumulate = Kernel::builder("accumulate").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((1024,)));
        let y = param_tensor::<f32>("y", Layout::col_major((1024,)));
        let i = block_idx().x * 128 + thread_idx().x;
        y.store(i, y.load(i) + x.load(i));
    });
    (double, accumulate)
}

#[test]
fn graph_replays_match_direct_launches() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let (double, accumulate) = graph_kernels();
    let (fd, fa) = (load(&ctx, &tc, &double), load(&ctx, &tc, &accumulate));
    let small = |seed: u32| data(seed).iter().map(|v| v / 64.0).collect::<Vec<f32>>();
    let run = |graph: bool| -> Vec<f32> {
        let stream = ctx.stream().unwrap();
        let (x, y) = (ctx.upload_slice(&small(1)).unwrap(), ctx.upload_slice(&small(2)).unwrap());
        let scratch = ctx.alloc::<f32>(1024).unwrap();
        let (ax, axy) = ([DeviceArg::Buffer(&x)], [DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)]);
        let step = |s: &CudaStream<'_>| {
            fd.launch_builder(axpy_cfg()).unwrap().stream(s).args(&ax).submit().unwrap();
            fa.launch_builder(axpy_cfg()).unwrap().stream(s).args(&axy).submit().unwrap();
            s.memset_async(&scratch, 0).unwrap();
            s.copy_async(&scratch, 0, &y, 0, 4096).unwrap();
        };
        if graph {
            let capture = stream.begin_capture(CaptureMode::ThreadLocal).unwrap();
            for _ in 0..3 {
                step(capture.stream());
            }
            let g = capture.end().unwrap();
            assert_eq!(g.node_count().unwrap(), 12);
            let exec = g.instantiate().unwrap();
            for _ in 0..5 {
                exec.launch(&stream).unwrap();
            }
        } else {
            for _ in 0..15 {
                step(&stream);
            }
        }
        stream.synchronize().unwrap();
        scratch.to_vec::<f32>().unwrap()
    };
    let direct = run(false);
    assert_eq!(run(true), direct);
    // And the simulator agrees.
    let mut x = vec![KernelArg::f32(small(1))];
    let mut y = small(2);
    for _ in 0..15 {
        double.simulate(axpy_cfg(), &mut x).unwrap();
        let mut args = vec![x[0].clone(), KernelArg::f32(y.clone())];
        accumulate.simulate(axpy_cfg(), &mut args).unwrap();
        y = args[1].as_f32().to_vec();
    }
    assert_eq!(direct, y);
}

#[test]
fn capture_misuse_is_explained() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = axpy();
    let f = load(&ctx, &tc, &k);
    let e = ctx.default_stream().begin_capture(CaptureMode::Global).err().unwrap().to_string();
    assert!(e.contains("the legacy NULL stream cannot be captured"), "{e}");
    let stream = ctx.stream().unwrap();
    // A host copy cannot be captured safely.
    let pinned = ctx.pinned_from(&[1.0f32; 4]).unwrap();
    let buf = ctx.alloc::<f32>(4).unwrap();
    {
        let capture = stream.begin_capture(CaptureMode::Global).unwrap();
        let e = capture.stream().upload_async(&pinned, &buf, 0).err().unwrap().to_string();
        assert!(e.contains("on a stream that is capturing a graph"), "{e}");
        capture.end().unwrap();
    }
    // CudaContext::launch uploads and synchronises: the driver refuses that during a capture.
    let capture = stream.begin_capture(CaptureMode::Global).unwrap();
    let mut args = vec![KernelArg::ScalarF32(1.0), KernelArg::f32(data(1)), KernelArg::f32(data(2))];
    let e = ctx.launch(&k, &f, axpy_cfg(), &mut args).unwrap_err();
    let text = e.to_string();
    assert!(matches!(e.code, Some(900) | Some(901)), "{text}");
    assert!(text.contains("CUDA_ERROR_STREAM_CAPTURE_"), "{text}");
    assert!(
        text.contains("graphs record stream-ordered work only") || text.contains("invalidated the stream's capture"),
        "{text}"
    );
    let end = capture.end();
    if let Err(e) = end {
        assert!(e.to_string().contains("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"), "{e}");
    }
    // The stream works again afterwards.
    let (x, y) = (ctx.upload_slice(&data(1)).unwrap(), ctx.upload_slice(&data(2)).unwrap());
    let dargs = [DeviceArg::Scalar(KernelArg::ScalarF32(1.0)), DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)];
    ctx.launch_on(&stream, &k, &f, axpy_cfg(), &dargs).unwrap();
    stream.synchronize().unwrap();
}

#[test]
fn a_launch_above_48kb_can_be_captured() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = region_kernel();
    let f = load(&ctx, &tc, &k);
    let stream = ctx.stream().unwrap();
    let out = ctx.alloc::<i32>(65).unwrap();
    out.memset(0xff).unwrap();
    let args = [DeviceArg::Buffer(&out)];
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]).with_smem(60_000);
    let capture = stream.begin_capture(CaptureMode::Global).unwrap();
    // cuFuncSetAttribute (the dynamic shared-memory limit) is allowed while capturing.
    f.launch_builder(cfg).unwrap().stream(capture.stream()).args(&args).submit().unwrap();
    let exec = capture.end().unwrap().instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let got = out.to_vec::<i32>().unwrap();
    assert_eq!(got[0], 60_000 - REGION_START as i32);
    assert_eq!(&got[1..4], &[0, 5, 10]);
}

#[test]
fn cuda_graphs_example() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let r = cuda_graphs::run(&ctx, &tc, 50).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(r.count, r.expected);
    assert!(r.stream_seconds > 0.0 && r.graph_seconds > 0.0);
}

#[test]
fn by_value_struct_on_the_gpu() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = struct_kernel();
    let mut args = struct_args();
    k.run_cuda(&ctx, &tc, LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_or_else(|e| panic!("{e}"));
    check_struct(&args);
    // Through device arguments, and through the compiled kernel's own signature.
    let f = load(&ctx, &tc, &k);
    let (ints, floats) = (ctx.alloc::<i64>(3).unwrap(), ctx.alloc::<f64>(2).unwrap());
    let dargs = [DeviceArg::ByValue(struct_bytes()), DeviceArg::Buffer(&ints), DeviceArg::Buffer(&floats)];
    f.launch_builder(LaunchConfig::new([1, 1, 1], [32, 1, 1])).unwrap().args(&dargs).submit().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(ints.to_vec::<i64>().unwrap(), [-5, 123_456, STRUCT_C]);
    assert_eq!(floats.to_vec::<f64>().unwrap(), [1.25, std::f64::consts::PI]);
    let short = [DeviceArg::ByValue(vec![0; 24]), DeviceArg::Buffer(&ints), DeviceArg::Buffer(&floats)];
    let e = f.launch_builder(LaunchConfig::new([1, 1, 1], [32, 1, 1])).unwrap().args(&short).submit().unwrap_err();
    assert!(e.to_string().contains("argument 0 (`s`) is passed by value as 32 bytes, got 24"), "{e}");
}

#[test]
fn signature_errors_on_the_gpu() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = Kernel::builder("leaves").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major((64,)));
        let n = param_extent("n", 4);
        let wide = param_extent_i64("wide", 1);
        let i = thread_idx().x;
        if_(i.lt(n), || x.store(i, wide.to::<f32>()));
    });
    let f = load(&ctx, &tc, &k);
    let buf = ctx.alloc::<f32>(128).unwrap();
    let cfg = LaunchConfig::new([1, 1, 1], [32, 1, 1]);
    let launch = |args: &[DeviceArg<'_>]| ctx.launch_async(&k, &f, cfg, args).err().map(|e| e.to_string());
    let ok = [
        DeviceArg::Buffer(&buf),
        DeviceArg::Scalar(KernelArg::ScalarI32(8)),
        DeviceArg::Scalar(KernelArg::scalar(3i64)),
    ];
    assert_eq!(launch(&ok), None);
    ctx.synchronize().unwrap();
    let e = launch(&[
        DeviceArg::Buffer(&buf),
        DeviceArg::Scalar(KernelArg::ScalarF32(8.0)),
        DeviceArg::Scalar(KernelArg::scalar(3i64)),
    ])
    .unwrap();
    assert!(e.contains("argument 1 (`n`) must be a i32 scalar"), "{e}");
    let e = launch(&[
        DeviceArg::Buffer(&buf),
        DeviceArg::Scalar(KernelArg::ScalarI32(6)),
        DeviceArg::Scalar(KernelArg::scalar(3i64)),
    ])
    .unwrap();
    assert!(e.contains("argument 1 (`n`) must be a multiple of 4, got 6"), "{e}");
    let e = launch(&[
        DeviceArg::Buffer(&buf),
        DeviceArg::Scalar(KernelArg::ScalarI32(8)),
        DeviceArg::Scalar(KernelArg::ScalarI32(3)),
    ])
    .unwrap();
    assert!(e.contains("argument 2 (`wide`) is a 64-bit leaf (i64)"), "{e}");
    let view = DeviceArg::View { buffer: &buf, byte_offset: 4 };
    let e = launch(&[view, DeviceArg::Scalar(KernelArg::ScalarI32(8)), DeviceArg::Scalar(KernelArg::scalar(3i64))])
        .unwrap();
    assert!(e.contains("argument 0: `x` is at") && e.contains("not 16-byte aligned"), "{e}");
    // An aligned view that reaches too few elements.
    let view = DeviceArg::View { buffer: &buf, byte_offset: 384 };
    let e = launch(&[view, DeviceArg::Scalar(KernelArg::ScalarI32(8)), DeviceArg::Scalar(KernelArg::scalar(3i64))])
        .unwrap();
    assert!(e.contains("needs 64 f32 elements (256 bytes), the pointer reaches 128"), "{e}");
    let view = DeviceArg::View { buffer: &buf, byte_offset: 256 };
    assert_eq!(
        launch(&[view, DeviceArg::Scalar(KernelArg::ScalarI32(8)), DeviceArg::Scalar(KernelArg::scalar(3i64))]),
        None
    );
    ctx.synchronize().unwrap();
}

#[test]
fn module_globals_on_the_gpu() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = globals_kernel();
    let f = load(&ctx, &tc, &k);
    let cfg = LaunchConfig::new([1, 1, 1], [4, 1, 1]);
    for want in 1..=2 {
        let mut args = globals_args();
        ctx.launch(&k, &f, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(args[1].as_f32(), [2.0, 6.0, 12.0, 20.0]);
        assert_eq!(args[2].as_i32(), [want], "the counter persists across launches of one module");
    }
    let count = f.global("run_count").unwrap();
    assert_eq!((count.bytes(), count.read::<i32>().unwrap()), (4, vec![2]));
    count.write(&[41i32]).unwrap();
    f.global("gamma").unwrap().write(&[2.0f32; 4]).unwrap();
    let mut args = globals_args();
    ctx.launch(&k, &f, cfg, &mut args).unwrap();
    assert_eq!((args[1].as_f32(), args[2].as_i32()), ([4.0, 8.0, 12.0, 16.0].as_slice(), [42].as_slice()));
    let e = count.write(&[1i64]).unwrap_err().to_string();
    assert!(e.contains("module global `run_count` is 4 bytes, got 8"), "{e}");

    let k = readonly_kernel();
    let f = load(&ctx, &tc, &k);
    let mut args = vec![KernelArg::i32(vec![0; 4])];
    ctx.launch(&k, &f, cfg, &mut args).unwrap();
    assert_eq!(args[0].as_i32(), [10, 21, 32, 43]);
    let e = f.global("offset").unwrap().write(&[0i32; 4]).unwrap_err().to_string();
    assert!(e.contains("module global `offset` is read-only"), "{e}");
}

#[test]
fn ptxas_assembles_grid_constants_and_clusters() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tc) = Toolchain::discover() else { return };
    if tc.cuda_tools().ptxas.is_none() {
        eprintln!("skipping: no ptxas");
        return;
    }
    let options = |arch| CompileOptions::new().gpu_arch(arch).use_ptxas(true);
    let c = grid_constant_kernel().compile_with(&tc, &options(Arch::Sm90a)).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.cubin().is_some_and(|b| !b.is_empty()));
    assert!(c.ptx().contains(".param .align 64 .b8 grid_constant_desc_param_0[128]"), "{}", c.ptx());
    let c = cluster_kernel().compile_with(&tc, &options(Arch::Sm100a)).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.cubin().is_some_and(|b| !b.is_empty()));
    assert!(!c.ptx().contains(".reqnctapercluster") && !c.ptx().contains(".explicitcluster"), "{}", c.ptx());
}

#[test]
fn launch_smem_up_to_the_opt_in_limit() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = region_kernel();
    let f = load(&ctx, &tc, &k);
    let limit = ctx.max_shared_memory_per_block() as u32;
    let out = ctx.alloc::<i32>(65).unwrap();
    let launch = |smem: u32| {
        ctx.launch_on_device(
            &k,
            &f,
            LaunchConfig::new([1, 1, 1], [64, 1, 1]).with_smem(smem),
            &[DeviceArg::Buffer(&out)],
        )
    };
    launch(limit).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.to_vec::<i32>().unwrap()[0], (limit - REGION_START) as i32);
    let e = launch(limit + 1).unwrap_err().to_string();
    assert!(e.contains(&format!("needs {} bytes of shared memory per block", limit + 1)), "{e}");
    assert!(e.contains(&format!("allows a block {limit}")), "{e}");
    // Matching the simulator at one size.
    let mut sim = vec![KernelArg::i32(vec![-1; 65])];
    k.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]).with_smem(REGION_START + 128), &mut sim).unwrap();
    out.memset(0xff).unwrap();
    launch(REGION_START + 128).unwrap();
    assert_eq!(out.to_vec::<i32>().unwrap(), sim[0].as_i32());
}

#[test]
fn carveout_and_occupancy_hints() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = bounds_kernel(None, None);
    let f = load(&ctx, &tc, &k);
    let out = ctx.alloc::<i32>(128).unwrap();
    let cfg = LaunchConfig::new([1, 1, 1], [128, 1, 1]);
    ctx.launch_on_device(&k, &f, cfg.with_smem_carveout(50), &[DeviceArg::Buffer(&out)]).unwrap();
    assert_eq!(f.get_attribute(FuncAttribute::PreferredSharedMemoryCarveout).unwrap(), 50);
    f.set_attribute(FuncAttribute::PreferredSharedMemoryCarveout, 25).unwrap();
    assert_eq!(f.get_attribute(FuncAttribute::PreferredSharedMemoryCarveout).unwrap(), 25);
    // min_blocks_per_sm compiles (.minnctapersm), runs, and sets a carveout for its blocks.
    let k = bounds_kernel(None, Some(4));
    let mut args = vec![KernelArg::i32(vec![0; 128])];
    k.run_cuda(&ctx, &tc, cfg, &mut args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(args[0].as_i32(), (0..128).map(|i| i * 2).collect::<Vec<_>>());
    let f = load(&ctx, &tc, &k);
    ctx.launch_on_device(&k, &f, cfg, &[DeviceArg::Buffer(&out)]).unwrap();
    let pct = f.get_attribute(FuncAttribute::PreferredSharedMemoryCarveout).unwrap();
    assert!((0..=100).contains(&pct), "{pct}");
    // max_threads: a smaller block runs, a larger one is refused before the launch.
    let k = bounds_kernel(Some([128, 1, 1]), None);
    let f = load(&ctx, &tc, &k);
    ctx.launch_on_device(&k, &f, LaunchConfig::new([1, 1, 1], [64, 1, 1]), &[DeviceArg::Buffer(&out)]).unwrap();
    let e = ctx.launch_on_device(&k, &f, LaunchConfig::new([1, 1, 1], [256, 1, 1]), &[DeviceArg::Buffer(&out)]);
    assert!(e.unwrap_err().to_string().contains("takes blocks of at most [128, 1, 1] (max_threads)"));
}

#[test]
fn clustered_launches_need_a_cluster_device() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    if ctx.device_attribute(DeviceAttribute::ClusterLaunch).unwrap_or(0) != 0 {
        eprintln!("skipping: this device launches clusters");
        return;
    }
    let k = bounds_kernel(None, None);
    let f = load(&ctx, &tc, &k);
    let out = ctx.alloc::<i32>(128).unwrap();
    let base = LaunchConfig::new([4, 1, 1], [128, 1, 1]);
    for cfg in [base.with_cluster([2, 1, 1]), base.with_fallback_cluster([4, 1, 1], [2, 1, 1])] {
        let e = ctx.launch_on_device(&k, &f, cfg, &[DeviceArg::Buffer(&out)]).unwrap_err().to_string();
        assert!(e.contains("cannot launch thread block clusters (CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH is 0"), "{e}");
    }
}

/// A graph replay is refused once a buffer its capture used has been dropped (replaying it
/// would be a use-after-free on the device); dropping an unrelated buffer does not matter.
#[test]
fn a_graph_replay_refuses_a_freed_buffer() {
    let Some((ctx, _)) = device() else { return };
    let stream = ctx.stream().unwrap();
    let kept = ctx.alloc::<f32>(64).unwrap();
    let freed = ctx.alloc::<f32>(64).unwrap();
    let other = ctx.alloc::<f32>(64).unwrap();
    let capture = stream.begin_capture(CaptureMode::Relaxed).unwrap();
    stream.memset_async(&kept, 1).unwrap();
    stream.memset_async(&freed, 2).unwrap();
    let graph = capture.end().unwrap();
    let exec = graph.instantiate().unwrap();
    drop(other);
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    drop(freed);
    let e = exec.launch(&stream).unwrap_err();
    assert!(format!("{e:?}").contains("freed"), "{e:?}");
}

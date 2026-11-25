//! The device dtype conversion, host tensors on the device and launches on a caller's stream,
//! on a real GPU. Needs the `cuda` feature; skipped (with a note) without an NVIDIA GPU or an
//! `mlir-opt`:
//!
//! ```text
//! cargo test --release --features cuda --test gpu_host_tensor
//! ```
//!
//! FP4 and FP6 destinations need the paired `cvt` instructions of sm_100a and sm_120a, so
//! they run only on such a device (they are compared on the simulator by `tests/host_tensor.rs`).

use cute::cuda::{CudaContext, CudaStream, DeviceArg, DeviceBuffer};
use cute::host::HostTensor;
use cute::host::*;
use cute::layout::Layout;
use cute::params::HostTensor as ParamTensor;
use cute::prelude::*;
use cute::testing::{convert, convert_sim};

const SWEEP: [DType; 10] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::F8E4M3FN,
    DType::F8E5M2,
    DType::F6E2M3FN,
    DType::F6E3M2FN,
    DType::F4E2M1FN,
    DType::I4,
    DType::I8,
];

/// Conversions and stream launches change the device's current context state; run one test
/// at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn device() -> Option<(CudaContext, Toolchain)> {
    let ctx = CudaContext::new(0).map_err(|e| eprintln!("skipping: {e}")).ok()?;
    let toolchain = Toolchain::discover();
    if toolchain.is_none() {
        eprintln!("skipping: no mlir-opt found (set CUTE_MLIR_OPT)");
    }
    Some((ctx, toolchain?))
}

fn offsets(layout: &Layout) -> Vec<usize> {
    HostTensor::<u8>::zeros(layout.clone()).offsets()
}

fn tensor_of(dtype: DType, layout: &Layout, values: &[f64]) -> ParamTensor {
    let mut bits = vec![0u64; layout.cosize()];
    for (o, &v) in offsets(layout).iter().zip(values) {
        bits[*o] = encode_f64(dtype, v);
    }
    ParamTensor::new(KernelArg::Bytes { elem: dtype, data: pack_elements(dtype, &bits) }, layout.clone())
}

fn logical_bits(t: &ParamTensor) -> Vec<u64> {
    let all = unpack_elements(t.element_type(), &arg_bytes(t.data()), t.layout().cosize());
    offsets(&t.layout()).into_iter().map(|o| all[o]).collect()
}

fn quarters(n: usize, seed: u64, limit: usize) -> Vec<f64> {
    let mut rng = cute::utils::gemm::Rng::new(seed);
    (0..n).map(|_| (rng.below(2 * limit + 1) as f64 - limit as f64) / 4.0).collect()
}

fn paired_cvt(ctx: &CudaContext) -> bool {
    ctx.compute_capability().0 >= 10
}

#[test]
fn device_convert_matches_the_simulator_and_the_host() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _tc)) = device() else { return };
    let layouts = [Layout::row_major((4, 32)), Layout::col_major((32, 4))];
    let mut run = 0;
    for &src in &SWEEP {
        for &dst in &SWEEP {
            if dst.is_float() && dst.bits() < 8 && !paired_cvt(&ctx) {
                continue;
            }
            for layout in &layouts {
                let values = quarters(layout.size(), 11 + run, if dst == DType::I4 { 13 } else { 30 });
                let host_src = tensor_of(src, layout, &values);
                let zeros = tensor_of(dst, layout, &vec![0.0; layout.size()]);
                let (d_src, d_dst) =
                    (DeviceTensor::upload(&ctx, &host_src).unwrap(), DeviceTensor::upload(&ctx, &zeros).unwrap());
                convert(&ctx, &d_src, &d_dst).unwrap_or_else(|e| panic!("{src} -> {dst} on {layout}: {e}"));
                let got = logical_bits(&d_dst.download().unwrap());
                let want: Vec<u64> =
                    logical_bits(&host_src).into_iter().map(|b| encode_f64(dst, decode_f64(src, b))).collect();
                assert_eq!(got, want, "{src} -> {dst} on {layout}: GPU against the host conversion");
                let mut sim = zeros.clone();
                convert_sim(&host_src, &mut sim).unwrap();
                assert_eq!(got, logical_bits(&sim), "{src} -> {dst} on {layout}: GPU against the simulator");
                run += 1;
            }
        }
    }
    eprintln!("{run} conversions on the GPU");
    assert!(run >= 2 * 10 * 6);
}

#[test]
fn device_convert_refuses_what_the_simulator_refuses() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _tc)) = device() else { return };
    let row = Layout::row_major((4, 30));
    let s = DeviceTensor::zeros(&ctx, DType::F32, row.clone()).unwrap();
    let d = DeviceTensor::zeros(&ctx, DType::F8E5M2, row).unwrap();
    assert!(convert(&ctx, &s, &d).unwrap_err().0.contains("multiple of 4"));
    let two = Layout::new((4, 4), (1, 1));
    let s = DeviceTensor::zeros(&ctx, DType::F32, two.clone()).unwrap();
    let d = DeviceTensor::zeros(&ctx, DType::F16, two).unwrap();
    assert!(convert(&ctx, &s, &d).unwrap_err().0.contains("leading mode must be unique"));
}

#[test]
fn host_tensors_round_trip_through_the_device() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, _tc)) = device() else { return };
    let t = matrix::<F16>(3, 8, 6, false, TensorInit::RANDOM, 4);
    let d = DeviceTensor::upload_typed(&ctx, &t).unwrap();
    assert_eq!(d.download_typed::<F16>().unwrap(), t);
    // A narrow-typed tensor made through the device kernel's rules and read back.
    let f = matrix::<f32>(2, 4, 32, false, TensorInit::Random { min: -3, max: 4 }, 8);
    let q = cute_tensor_like(&f, DType::I4, false, None).unwrap();
    let dq = DeviceTensor::upload(&ctx, &q).unwrap();
    let back = dq.download().unwrap();
    assert_eq!(logical_bits(&back), logical_bits(&q));
    // The leading-dimension query on a permuted tensor.
    let p = create_and_permute_tensor::<f32>(&[5, 7], Some(&[1, 0]), TensorInit::GAUSSIAN, 1);
    assert_eq!(
        (p.leading_dim(), DeviceTensor::upload_typed(&ctx, &p).unwrap().layout().clone()),
        (Some(0), p.layout().clone())
    );
}

/// `y[i] = 2 x[i]`.
fn doubler() -> Kernel {
    Kernel::builder("doubler").block_dim([128, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(1024));
        let y = param_tensor::<f32>("y", Layout::col_major(1024));
        let i = block_idx().x * 128 + thread_idx().x;
        y.store(i, x.load(i) * 2.0f32);
    })
}

/// A launch on a stream another library owns: here one the test creates with the driver and
/// wraps from its raw handle, as a caller would wrap torch's current stream.
#[test]
fn launches_on_a_callers_raw_stream() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, tc)) = device() else { return };
    let k = doubler();
    let arch = ctx.best_target(&tc).expect("a target");
    let f = ctx.load(&k.compile_with(&tc, &CompileOptions::new().gpu_arch(arch)).unwrap()).unwrap();
    let data: Vec<f32> = (0..1024).map(|i| i as f32 - 500.0).collect();
    let (x, y): (DeviceBuffer<'_>, DeviceBuffer<'_>) =
        (ctx.upload_slice(&data).unwrap(), ctx.alloc::<f32>(1024).unwrap());
    // The raw handle of a driver-created stream: the caller keeps ownership.
    let owner = ctx.stream().unwrap();
    // SAFETY: `owner` is a live stream of `ctx` that outlives `raw`.
    let raw = unsafe { CudaStream::from_raw(&ctx, owner.as_raw()) };
    assert!(!raw.is_default());
    f.launch_builder(LaunchConfig::new([8, 1, 1], [128, 1, 1]))
        .unwrap()
        .stream(&raw)
        .args(&[DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)])
        .submit()
        .unwrap();
    raw.synchronize().unwrap();
    let out: Vec<f32> = y.to_vec().unwrap();
    assert!(out.iter().zip(&data).all(|(o, d)| *o == 2.0 * d));
    // A null handle is the legacy default stream.
    let null = unsafe { CudaStream::from_raw(&ctx, std::ptr::null_mut()) };
    assert!(null.is_default());
    y.memset(0).unwrap();
    f.launch_builder(LaunchConfig::new([8, 1, 1], [128, 1, 1]))
        .unwrap()
        .stream(&null)
        .args(&[DeviceArg::Buffer(&x), DeviceArg::Buffer(&y)])
        .submit()
        .unwrap();
    null.synchronize().unwrap();
    assert_eq!(y.to_vec::<f32>().unwrap()[1023], 2.0 * data[1023]);
}

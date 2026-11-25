//! Kernels, DLTensors and a stub TVM-FFI runtime for the TVM-FFI tests (`tests/tvm_ffi.rs`
//! on any machine, `tests/gpu_tvm_ffi.rs` on a GPU).

#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::ffi::{CStr, c_char, c_void};

use cute::external::dlpack::{DLDataType, DLDevice, DLTensor, device_type, type_code};
use cute::prelude::*;
use cute::tvm_ffi::{TVMFFIAny, TvmFfiRuntime, set_runtime_override};

/// `c = a + b` over `M x N` tensors of row stride `S` (one `SymInt` each, shared by the three
/// tensors), a grid-stride loop of one element per thread.
pub type AddParams = (DlTensor<f32>, DlTensor<f32>, DlTensor<f32>);
pub type AddHost = (DlpackTensor, DlpackTensor, DlpackTensor);

pub fn add_kernel() -> TypedKernel<AddParams> {
    let (m, n, s) = (sym_int32(1, Some("M")), sym_int32(1, Some("N")), sym_int64(1, Some("S")));
    let t = make_fake_tensor::<f32>((m, n), (s, 1));
    Kernel::builder("ffi_add").block_dim([128, 1, 1]).jit(
        (t.clone(), t.clone(), t),
        |a: DlTensor<f32>, b: DlTensor<f32>, c: DlTensor<f32>| {
            let m: Int32 = size(a.tensor(), &[0]).into_value();
            let n: Int32 = size(a.tensor(), &[1]).into_value();
            let start = block_idx().x * 128 + thread_idx().x;
            for_range(start, m * n, grid_dim().x * 128, |i| {
                let (r, k) = (i / n, i % n);
                c.store((r, k), a.load((r, k)) + b.load((r, k)));
            });
        },
    )
}

/// The add's launch: 128 threads per block, at most 64 blocks.
pub fn add_config(h: &AddHost) -> Result<LaunchConfig, String> {
    let elems: i64 = h.0.shape().iter().product();
    let blocks = (elems as u64).div_ceil(128).clamp(1, 64) as u32;
    Ok(LaunchConfig::new([blocks, 1, 1], [128, 1, 1]))
}

/// `y = alpha * x` over `n` elements, `n` a multiple of 4 and both tensors 16-byte aligned.
pub type ScaleParams = (DlTensor<f32>, DlTensor<f32>, Float32);
pub type ScaleHost = (DlpackTensor, DlpackTensor, f32);

pub fn scale_kernel() -> TypedKernel<ScaleParams> {
    let n = sym_int32(4, Some("n"));
    let t = make_fake_tensor::<f32>((n,), (1,)).with_assumed_align(16);
    Kernel::builder("ffi_scale").block_dim([64, 1, 1]).jit(
        (t.clone(), t, ScalarSpec::Plain),
        |x: DlTensor<f32>, y: DlTensor<f32>, alpha: Float32| {
            let n: Int32 = size(x.tensor(), &[0]).into_value();
            for_range(thread_idx().x, n, 64, |i| y.store(i, alpha * x.load(i)));
        },
    )
}

pub fn scale_config(h: &ScaleHost) -> Result<LaunchConfig, String> {
    if h.2.is_nan() {
        return Err("alpha is NaN".to_string());
    }
    Ok(LaunchConfig::new([1, 1, 1], [64, 1, 1]))
}

/// The kernel of CuTe DSL's `error_reporting.py`: `b = a + 1` on tensors of the static
/// layout `(10,):(1,)`, one thread.
pub type StaticParams = (DlTensor<f32>, DlTensor<f32>);
pub type StaticHost = (DlpackTensor, DlpackTensor);

pub fn static_kernel() -> TypedKernel<StaticParams> {
    let t = make_fake_tensor::<f32>((10,), (1,));
    Kernel::builder("ffi_add_one_static").block_dim([1, 1, 1]).jit(
        (t.clone(), t),
        |a: DlTensor<f32>, b: DlTensor<f32>| {
            for_range(0, 10, 1, |i| b.store(i, a.load(i) + 1.0f32));
        },
    )
}

pub fn one_thread(_: &StaticHost) -> Result<LaunchConfig, String> {
    Ok(LaunchConfig::new([1, 1, 1], [1, 1, 1]))
}

/// An export of `kernel` with a placeholder PTX: enough to decode and check calls on a machine
/// without a GPU or an `mlir-opt`.
pub fn placeholder_export(kernel: &Kernel) -> cute::ExportedKernel {
    let mut sig = kernel.signature();
    sig.target = Some(Arch::Sm89);
    sig.ptx_version = Some(78);
    cute::ExportedKernel::new(sig, Some("// placeholder\n".to_string()), None).expect("an export")
}

pub fn f32_type() -> DLDataType {
    DLDataType { code: type_code::KDL_FLOAT, bits: 32, lanes: 1 }
}

pub fn cuda(id: i32) -> DLDevice {
    DLDevice { device_type: device_type::KDL_CUDA, device_id: id }
}

/// A `DLTensor` with its own shape and strides (boxed, so the pointers stay put).
pub struct Dl {
    pub t: Box<DLTensor>,
    _shape: Box<[i64]>,
    _strides: Option<Box<[i64]>>,
}

impl Dl {
    pub fn new(address: u64, dtype: DLDataType, shape: &[i64], strides: Option<&[i64]>, device: DLDevice) -> Dl {
        let mut shape: Box<[i64]> = shape.into();
        let mut strides: Option<Box<[i64]>> = strides.map(Into::into);
        let t = Box::new(DLTensor {
            data: address as usize as *mut c_void,
            device,
            ndim: shape.len() as i32,
            dtype,
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut().map_or(std::ptr::null_mut(), |s| s.as_mut_ptr()),
            byte_offset: 0,
        });
        Dl { t, _shape: shape, _strides: strides }
    }

    /// A row-major f32 tensor on CUDA device 0.
    pub fn f32(address: u64, shape: &[i64]) -> Dl {
        Dl::new(address, f32_type(), shape, None, cuda(0))
    }

    pub fn any(&self) -> TVMFFIAny {
        TVMFFIAny::dl_tensor(&*self.t)
    }
}

thread_local! {
    static RAISED: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
    static STREAM: Cell<usize> = const { Cell::new(0) };
}

unsafe extern "C" fn stub_raise(kind: *const c_char, parts: *const *const c_char, num_parts: i32) {
    // SAFETY: the runtime's contract: NUL-terminated strings, `num_parts` of them (each may be
    // null).
    let text = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let mut message = String::new();
    for i in 0..num_parts.max(0) as usize {
        // SAFETY: as above.
        let p = unsafe { *parts.add(i) };
        if !p.is_null() {
            message += &text(p);
        }
    }
    RAISED.with(|r| r.borrow_mut().push((text(kind), message)));
}

unsafe extern "C" fn stub_stream(device_type: i32, _device_id: i32) -> *mut c_void {
    assert_eq!(device_type, cute::external::dlpack::device_type::KDL_CUDA);
    STREAM.with(|s| s.get()) as *mut c_void
}

/// Install the stub runtime: raised errors are kept per thread ([`take_raised`]) and the
/// environment's stream is this thread's [`set_stream`] (null by default).
pub fn install_stubs() {
    set_runtime_override(Some(TvmFfiRuntime { set_raised_from_c_str_parts: stub_raise, env_get_stream: stub_stream }));
}

/// The error last raised on this thread, if any (and forget it).
pub fn take_raised() -> Option<(String, String)> {
    RAISED.with(|r| r.borrow_mut().pop())
}

/// The stream `TVMFFIEnvGetStream` returns on this thread.
pub fn set_stream(raw: *mut c_void) {
    STREAM.with(|s| s.set(raw as usize));
}

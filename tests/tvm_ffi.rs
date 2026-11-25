//! TVM-FFI callables without a GPU: the ABI's layout, the exported safe call
//! (`tvm_ffi_export!`) through its C ABI with a stub runtime, the decoding of every argument
//! kind, and each check failing with its exception kind and message. The launches themselves
//! are in `tests/gpu_tvm_ffi.rs`.

#[path = "support/tvm_ffi_kernels.rs"]
mod kernels;

use std::mem::offset_of;

use cute::external::dlpack::{DLDataType, device_type, type_code};
use cute::prelude::*;
use cute::tvm_ffi::{
    FfiError, PreparedArg, SYMBOL_PREFIX, TVM_FFI_VERSION, TVMFFIAny, TVMFFIObject, TVMFFISafeCallType, TvmFfiFunction,
    type_index,
};
use kernels::*;

fn add_fn() -> TvmFfiFunction<AddParams> {
    let k = add_kernel();
    let image = placeholder_export(k.kernel());
    TvmFfiFunction::from_exported(k, image, add_config).unwrap().with_arg_names(&["a", "b", "c"])
}

fn scale_fn() -> TvmFfiFunction<ScaleParams> {
    let k = scale_kernel();
    let image = placeholder_export(k.kernel());
    TvmFfiFunction::from_exported(k, image, scale_config).unwrap().with_arg_names(&["x", "y", "alpha"])
}

fn static_fn() -> Result<TvmFfiFunction<StaticParams>, String> {
    let k = static_kernel();
    let image = placeholder_export(k.kernel());
    Ok(TvmFfiFunction::from_exported(k, image, one_thread)?.with_name("add_one").with_arg_names(&["a", "b"]))
}

cute::tvm_ffi_export!(ffi_add, add_fn());
cute::tvm_ffi_export!(ffi_static, static_fn());
cute::tvm_ffi_export!(ffi_unavailable, Err::<TvmFfiFunction<StaticParams>, _>("no toolchain here"));

/// Constructor runs of `ffi_flaky`, which fails on its first.
static FLAKY_BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn flaky_fn() -> Result<TvmFfiFunction<StaticParams>, String> {
    if FLAKY_BUILDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
        return Err("not yet".to_string());
    }
    static_fn()
}

cute::tvm_ffi_export!(ffi_flaky, flaky_fn());
cute::tvm_ffi_register_all!(ffi_add, "cute_rs.static" => ffi_static);

/// Call an exported safe call through its C ABI; the raised error, if it failed.
fn call(f: TVMFFISafeCallType, args: &[TVMFFIAny]) -> Result<(), (String, String)> {
    install_stubs();
    let mut result = TVMFFIAny::int(7);
    // SAFETY: valid arguments whose tensors outlive the call.
    let rc = unsafe { f(std::ptr::null_mut(), args.as_ptr(), args.len() as i32, &mut result) };
    match rc {
        0 => {
            assert_eq!(result.type_index, type_index::NONE);
            Ok(())
        }
        -1 => Err(take_raised().expect("an error raised with -1")),
        rc => panic!("return code {rc}"),
    }
}

fn expect_err(r: Result<impl std::fmt::Debug, FfiError>, kind: &str, parts: &[&str]) {
    let e = r.expect_err("an error");
    assert_eq!(e.kind, kind, "{e}");
    for p in parts {
        assert!(e.message.contains(p), "missing {p:?} in: {e}");
    }
}

#[test]
fn abi_layout_matches_c_api_h() {
    assert_eq!(TVM_FFI_VERSION, (0, 1, 14));
    assert_eq!(SYMBOL_PREFIX, "__tvm_ffi_");
    assert_eq!((std::mem::size_of::<TVMFFIAny>(), std::mem::align_of::<TVMFFIAny>()), (16, 8));
    assert_eq!((offset_of!(TVMFFIAny, type_index), offset_of!(TVMFFIAny, value)), (0, 8));
    assert_eq!(std::mem::size_of::<TVMFFIObject>(), 24);
    assert_eq!((offset_of!(TVMFFIObject, combined_ref_count), offset_of!(TVMFFIObject, type_index)), (0, 8));
    assert_eq!(offset_of!(TVMFFIObject, deleter), 16);
    let indices = [
        (type_index::NONE, 0),
        (type_index::INT, 1),
        (type_index::BOOL, 2),
        (type_index::FLOAT, 3),
        (type_index::OPAQUE_PTR, 4),
        (type_index::DL_TENSOR_PTR, 7),
        (type_index::OBJECT, 64),
        (type_index::TENSOR, 70),
        (type_index::ARRAY, 71),
        (type_index::OPAQUE_PY_OBJECT, 74),
    ];
    for (got, want) in indices {
        assert_eq!(got, want);
    }
    // The constructors set the member the type index names.
    // SAFETY: reading the member just written.
    unsafe {
        assert_eq!(TVMFFIAny::int(-3).value.v_int64, -3);
        assert_eq!(TVMFFIAny::float(2.5).value.v_float64, 2.5);
        assert_eq!(TVMFFIAny::bool(true).value.v_int64, 1);
    }
    assert_eq!(TVMFFIAny::none().zero_padding, 0);
}

#[test]
fn the_exported_symbol_has_the_tvm_ffi_signature() {
    let f: TVMFFISafeCallType = ffi_add;
    let _ = f;
    let s: TVMFFISafeCallType = ffi_static;
    let _ = s;
}

/// A call that passes every check: the prepared launch has the tensors' addresses and spans,
/// the dynamic extents and stride (shared by the three tensors), and the configuration; the
/// launch itself then fails here as a RuntimeError (no GPU, or no `cuda` feature).
#[test]
fn a_valid_call_is_prepared_and_launched() {
    let f = add_fn();
    assert_eq!(f.signature_text(), "ffi_add(a: Tensor, b: Tensor, c: Tensor)");
    let (a, b, c) = (Dl::f32(0x10000, &[37, 64]), Dl::f32(0x20000, &[37, 64]), Dl::f32(0x30000, &[37, 64]));
    let p = f.prepare(&[a.any(), b.any(), c.any()]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(p.device, 0);
    assert_eq!(p.config.grid, [19, 1, 1]);
    let bytes = 37 * 64 * 4;
    assert_eq!(p.args[0], PreparedArg::Pointer { address: 0x10000, bytes, elem: DType::F32 });
    // M and N are 32-bit, the row stride 64-bit.
    let scalars = [KernelArg::scalar(37i32), KernelArg::scalar(64i32), KernelArg::scalar(64i64)];
    assert_eq!(&p.args[1..4], scalars.map(PreparedArg::Scalar).as_slice());
    assert_eq!(p.args.len(), 6, "{:?}", p.args);
    // Through the exported C symbol: decoded and checked, then the launch fails.
    let e = call(ffi_add, &[a.any(), b.any(), c.any()]).unwrap_err();
    assert_eq!(e.0, "RuntimeError", "{e:?}");
}

/// A tensor object (`kTVMFFITensor`) is a DLTensor after the object header.
#[test]
fn tensor_objects_are_decoded() {
    #[repr(C)]
    struct TensorObject {
        header: TVMFFIObject,
        tensor: cute::external::dlpack::DLTensor,
    }
    let (a, b, c) = (Dl::f32(0x10000, &[4, 8]), Dl::f32(0x20000, &[4, 8]), Dl::f32(0x30000, &[4, 8]));
    let object = |d: &Dl| {
        Box::new(TensorObject {
            header: TVMFFIObject { combined_ref_count: 1, type_index: type_index::TENSOR, padding: 0, deleter: None },
            tensor: *copy_dl(d),
        })
    };
    let (mut oa, mut ob) = (object(&a), object(&b));
    let args = [TVMFFIAny::tensor_object(&mut oa.header), TVMFFIAny::tensor_object(&mut ob.header), c.any()];
    let p = add_fn().prepare(&args).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(p.args[0], PreparedArg::Pointer { address: 0x10000, bytes: 128, elem: DType::F32 });
    assert_eq!(p.args[4], PreparedArg::Pointer { address: 0x20000, bytes: 128, elem: DType::F32 });
}

fn copy_dl(d: &Dl) -> Box<cute::external::dlpack::DLTensor> {
    let t = &*d.t;
    Box::new(cute::external::dlpack::DLTensor {
        data: t.data,
        device: t.device,
        ndim: t.ndim,
        dtype: t.dtype,
        shape: t.shape,
        strides: t.strides,
        byte_offset: t.byte_offset,
    })
}

#[test]
fn scalars_are_decoded_and_range_checked() {
    let f = scale_fn();
    assert_eq!(f.signature_text(), "ffi_scale(x: Tensor, y: Tensor, alpha: Float32)");
    let (x, y) = (Dl::f32(0x1000, &[64]), Dl::f32(0x2000, &[64]));
    for alpha in [TVMFFIAny::float(0.5), TVMFFIAny::int(3)] {
        let p = f.prepare(&[x.any(), y.any(), alpha]).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(p.args.last(), Some(PreparedArg::Scalar(_))));
    }
    expect_err(
        f.prepare(&[x.any(), y.any(), x.any()]),
        "TypeError",
        &[
            "ffi_scale(x: Tensor, y: Tensor, alpha: Float32): argument 2 (`alpha`)",
            "must be a float (Float32), got a tensor",
        ],
    );
    // The configuration function's error.
    expect_err(f.prepare(&[x.any(), y.any(), TVMFFIAny::float(f64::NAN)]), "ValueError", &["alpha is NaN"]);
}

#[test]
fn integer_arguments_are_range_checked() {
    use cute::tvm_ffi::{FfiArgs, FromTvmFfi};
    let decode =
        |a: TVMFFIAny| <(i32, u8, bool)>::decode(&mut FfiArgs::new(&[TVMFFIAny::int(1), a, TVMFFIAny::bool(true)]));
    assert_eq!(decode(TVMFFIAny::int(255)).unwrap(), (1, 255, true));
    expect_err(decode(TVMFFIAny::int(256)), "ValueError", &["argument 1 is 256, which does not fit in an Uint8"]);
    expect_err(decode(TVMFFIAny::float(1.0)), "TypeError", &["argument 1 must be an int (Uint8), got float"]);
    assert_eq!(<(i32, u8, bool)>::arity(), 3);
}

/// Each check, through the exported safe call: the exception kind and message, which name
/// the argument and the signature.
#[test]
fn each_check_fails_with_its_message() {
    let ok = || (Dl::f32(0x10000, &[37, 64]), Dl::f32(0x20000, &[37, 64]), Dl::f32(0x30000, &[37, 64]));
    let sig = "ffi_add(a: Tensor, b: Tensor, c: Tensor)";
    let check = |args: &[TVMFFIAny], kind: &str, parts: &[&str]| {
        let (k, m) = call(ffi_add, args).unwrap_err();
        assert_eq!(k, kind, "{m}");
        assert!(m.starts_with(sig), "{m}");
        for p in parts {
            assert!(m.contains(p), "missing {p:?} in: {m}");
        }
    };
    let (a, b, c) = ok();
    // The argument count.
    check(&[a.any(), b.any()], "TypeError", &["takes 3 arguments, got 2"]);
    // Not a tensor.
    check(&[a.any(), TVMFFIAny::int(4), c.any()], "TypeError", &["argument 1 (`b`): must be a tensor", "got int"]);
    // The element type.
    let half =
        Dl::new(0x20000, DLDataType { code: type_code::KDL_FLOAT, bits: 16, lanes: 1 }, &[37, 64], None, cuda(0));
    check(
        &[a.any(), half.any(), c.any()],
        "ValueError",
        &["argument 1 (`b`)", "expected a tensor of Float32, got one of Float16"],
    );
    // The rank.
    let flat = Dl::f32(0x20000, &[37 * 64]);
    check(&[a.any(), flat.any(), c.any()], "ValueError", &["argument 1 (`b`)", "layout"]);
    // A shape that disagrees with a SymInt another tensor set.
    let short = Dl::f32(0x20000, &[36, 64]);
    check(
        &[a.any(), short.any(), c.any()],
        "ValueError",
        &["argument 1 (`b`)", "`b_shape_0`: got 36, but `a_shape_0` is the same SymInt"],
    );
    // Host memory.
    let host = Dl::new(0x20000, f32_type(), &[37, 64], None, cpu());
    check(&[a.any(), host.any(), c.any()], "ValueError", &["argument 1 (`b`)", "cpu:0", "CUDA tensors"]);
    // Another device.
    let other = Dl::new(0x30000, f32_type(), &[37, 64], None, cuda(1));
    check(
        &[a.any(), b.any(), other.any()],
        "ValueError",
        &["argument 2 (`c`): is memory of CUDA device 1, but `a` is on device 0"],
    );
    // A null DLTensor*.
    check(&[a.any(), TVMFFIAny::dl_tensor(std::ptr::null()), c.any()], "ValueError", &["null tensor"]);
    // A misaligned start (an f32 tensor must be 4-byte aligned).
    let odd = Dl::f32(0x20002, &[37, 64]);
    check(&[a.any(), odd.any(), c.any()], "ValueError", &["argument 1 (`b`)", "0x20002"]);
}

/// Divisibility and the assumed alignment of the scale kernel's tensors.
#[test]
fn divisibility_and_alignment_are_checked() {
    let f = scale_fn();
    let alpha = TVMFFIAny::float(2.0);
    let (x, y) = (Dl::f32(0x1000, &[62]), Dl::f32(0x2000, &[62]));
    expect_err(f.prepare(&[x.any(), y.any(), alpha]), "ValueError", &["argument 0 (`x`)", "multiple of 4, got 62"]);
    let (x, y) = (Dl::f32(0x1000, &[64]), Dl::f32(0x2008, &[64]));
    expect_err(
        f.prepare(&[x.any(), y.any(), alpha]),
        "ValueError",
        &["argument 1 (`y`): `y` is at 0x2008, which is not 16-byte aligned"],
    );
}

/// The static layout of `error_reporting.py`: a tensor of 5 elements is refused.
#[test]
fn a_static_shape_mismatch_is_reported() {
    let (a, b) = (Dl::f32(0x1000, &[5]), Dl::f32(0x2000, &[10]));
    let (kind, m) = call(ffi_static, &[a.any(), b.any()]).unwrap_err();
    assert_eq!(kind, "ValueError");
    assert!(
        m.starts_with("add_one(a: Tensor, b: Tensor): argument 0 (`a`)") && m.contains("10") && m.contains('5'),
        "{m}"
    );
}

/// A constructor that failed: every call raises its error.
#[test]
fn an_unavailable_kernel_raises_its_constructor_error() {
    for _ in 0..2 {
        let (kind, m) = call(ffi_unavailable, &[]).unwrap_err();
        assert_eq!(kind, "RuntimeError");
        assert!(m.contains("__tvm_ffi_ffi_unavailable") && m.contains("no toolchain here"), "{m}");
    }
}

/// A failed constructor is not kept: the next call on the thread builds the callable again,
/// and a built one is kept.
#[test]
fn a_failed_constructor_runs_again() {
    std::thread::spawn(|| {
        let (kind, m) = call(ffi_flaky, &[]).unwrap_err();
        assert_eq!(kind, "RuntimeError");
        assert!(m.contains("the kernel is unavailable: not yet"), "{m}");
        let (kind, m) = call(ffi_flaky, &[]).unwrap_err();
        assert_eq!(kind, "TypeError", "{m}");
        assert!(m.contains("takes 2 arguments, got 0"), "{m}");
        let _ = call(ffi_flaky, &[]);
        assert_eq!(FLAKY_BUILDS.load(std::sync::atomic::Ordering::SeqCst), 2);
    })
    .join()
    .unwrap();
}

/// A negative DLPack device id is not a CUDA ordinal.
#[test]
fn a_negative_device_id_is_refused() {
    let (a, b) = (Dl::f32(0x10000, &[10]), Dl::new(0x20000, f32_type(), &[10], None, cuda(-1)));
    let (kind, m) = call(ffi_static, &[a.any(), b.any()]).unwrap_err();
    assert_eq!(kind, "ValueError", "{m}");
    assert!(m.contains("argument 1 (`b`)") && m.contains("not a CUDA device ordinal"), "{m}");
}

#[test]
fn an_image_of_another_kernel_is_refused() {
    let image = placeholder_export(scale_kernel().kernel());
    let e = TvmFfiFunction::from_exported(add_kernel(), image, add_config).unwrap_err();
    assert!(e.contains("not the same kernel"), "{e}");
}

fn cpu() -> cute::external::dlpack::DLDevice {
    cute::external::dlpack::DLDevice { device_type: device_type::KDL_CPU, device_id: 0 }
}

#[test]
fn enable_tvm_ffi_is_an_accepted_option() {
    let o = CompileOptions::parse("--enable-tvm-ffi").unwrap();
    assert_eq!(o.get_enable_tvm_ffi(), Some(true));
    assert_eq!(CompileOptions::parse("enable-tvm-ffi=0").unwrap().get_enable_tvm_ffi(), Some(false));
    let env = |k: &str| (k == "CUTE_RS_ENABLE_TVM_FFI").then(|| "1".to_string());
    assert_eq!(CompileOptions::from_env_with(env).unwrap().get_enable_tvm_ffi(), Some(true));
}

/// `cute_tvm_ffi_register_all` tries every function and counts the failures: here there is no
/// `libtvm_ffi` in the process, so each registration fails with a message saying so.
#[test]
fn registration_needs_a_loaded_runtime() {
    let e = cute::tvm_ffi::register_global_function("cute_rs.add", ffi_add, true).unwrap_err();
    assert!(e.contains("not loaded") && e.contains("import tvm_ffi"), "{e}");
    assert_eq!(cute_tvm_ffi_register_all(), 2);
}

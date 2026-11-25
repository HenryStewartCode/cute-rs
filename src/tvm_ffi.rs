//! TVM-FFI callables (CuTe DSL's `cute.compile(..., options="--enable-tvm-ffi")`,
//! `cute.EnableTVMFFI` and `CUTE_DSL_ENABLE_TVM_FFI`): a cute-rs kernel exported from a Rust
//! `cdylib` as an Apache TVM-FFI function, which Python frameworks call with their own tensors
//! (`tvm_ffi.load_module(path).name(a, b)` on `torch.Tensor`s).
//!
//! [`TvmFfiFunction`] wraps a [`TypedKernel`] whose host values the call's arguments decode
//! into ([`FromTvmFfi`]: DLPack tensors, integers, floats and Booleans, and tuples of them as
//! consecutive arguments), its compiled image and a function choosing the launch
//! configuration from the arguments. [`tvm_ffi_export!`](crate::tvm_ffi_export) exports it as
//! the C symbol `__tvm_ffi_<name>` with the TVM-FFI safe-call signature
//! ([`TVMFFISafeCallType`]). A call:
//!
//! 1. decodes each [`TVMFFIAny`] argument: a `DLTensor*` (`kTVMFFIDLTensorPtr`) or a
//!    `tvm_ffi.Tensor` object (`kTVMFFITensor`, whose `DLTensor` follows the object header)
//!    becomes a borrowed [`DlpackTensor`] over CUDA memory, converted as a tensor passed straight
//!    to a CuTe DSL function is (`from_dlpack(t).mark_layout_dynamic()`); integers, floats and
//!    Booleans are range-checked;
//! 2. checks the tensors against the kernel's specs through [`TypedKernel::pack_full`] (element
//!    type, static extents and strides, the divisibility and width of dynamic ones, a
//!    [`SymInt`](crate::SymInt) shared between tensors) and the packed values against the
//!    kernel's [`KernelSignature`] (pointer alignment, scalar types); every tensor must be CUDA
//!    memory of one device;
//! 3. takes that device's stream from the environment (`TVMFFIEnvGetStream`, the stream the
//!    framework is using) and launches there (feature `cuda`), through the launcher's own
//!    checks;
//! 4. reports any failure through `TVMFFIErrorSetRaisedFromCStrParts` as a `TypeError`,
//!    `ValueError` or `RuntimeError` naming the argument and the function's signature, and
//!    returns -1; Python raises it.
//!
//! The TVM-FFI runtime functions live in `libtvm_ffi`, which the host process (Python, through
//! `import tvm_ffi`, which loads it `RTLD_GLOBAL`) has already loaded: they are looked up with
//! `dlsym` at call time, so the `cdylib` links nothing. [`set_runtime_override`] installs other
//! implementations (stubs in a test or a C host without TVM-FFI).
//!
//! The ABI is that of `include/tvm/ffi/c_api.h` and `include/tvm/ffi/extra/c_env_api.h` of
//! apache-tvm-ffi 0.1.14 ([`TVM_FFI_VERSION`]). nvidia-cutlass-dsl 4.8.1 does not pin
//! apache-tvm-ffi (its PyPI metadata lists no dependency on it, and the tutorials'
//! `requirements.txt` names it without a version), so cute-rs follows the release installed
//! next to CuTe DSL on the test machine, 0.1.14.post0. The struct layouts and numbers follow
//! that header (Apache-2.0; THIRD-PARTY-NOTICES.md says so).
//!
//! What is supported of CuTe DSL's argument conventions: positional tensors and scalars, and
//! Rust tuples of parameters as runs of consecutive arguments (a Python caller passes a
//! tuple's fields one by one). TVM-FFI `Array`, `Map` and dataclass objects are not decoded,
//! and defaults and keyword arguments, which CuTe DSL's Python wrapper resolves before the
//! call, have no counterpart: a cute-rs callable takes exactly its positional arguments.
//!
//! ```ignore
//! // In a crate built as a `cdylib`:
//! fn add_one() -> Result<TvmFfiFunction<(DlTensor<f32>, DlTensor<f32>)>, Box<dyn std::error::Error>> {
//!     let kernel = Kernel::builder("add_one").jit(specs, |a: DlTensor<f32>, b: DlTensor<f32>| { ... });
//!     let toolchain = Toolchain::discover().ok_or("no mlir-opt")?;
//!     Ok(TvmFfiFunction::compile(kernel, &toolchain, &CompileOptions::new(), |_| {
//!         Ok(LaunchConfig::new([1, 1, 1], [1, 1, 1]))
//!     })?)
//! }
//! cute::tvm_ffi_export!(add_one, add_one());
//! // Python: tvm_ffi.load_module("libmy_kernels.so").add_one(a, b)
//! ```

use std::cell::OnceCell;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::sync::{Mutex, OnceLock};

use crate::aot::ExportedKernel;
use crate::backend::CompiledKernel;
use crate::external::DlpackTensor;
use crate::external::dlpack::{DLDataType, DLDevice, DLTensor, device_type};
use crate::ir::DType;
use crate::params::{ArgPack, KernelParam, TypedKernel};
use crate::signature::{KernelSignature, LaunchValue, ParamKind};
use crate::sim::{KernelArg, LaunchConfig};

/// The apache-tvm-ffi version whose C ABI this module follows (`TVM_FFI_VERSION_MAJOR`,
/// `_MINOR`, `_PATCH` of its `c_api.h`).
pub const TVM_FFI_VERSION: (u32, u32, u32) = (0, 1, 14);

/// The prefix of an exported function's symbol (`tvm_ffi_symbol_prefix`).
pub const SYMBOL_PREFIX: &str = "__tvm_ffi_";

// Several kernels in one module.
mod multi;
pub use multi::shared_module_loads;

/// `TVMFFITypeIndex` values (an `int32_t` enum in C).
pub mod type_index {
    /// `kTVMFFINone`.
    pub const NONE: i32 = 0;
    /// `kTVMFFIInt`: `v_int64`.
    pub const INT: i32 = 1;
    /// `kTVMFFIBool`: `v_int64`, 0 or 1.
    pub const BOOL: i32 = 2;
    /// `kTVMFFIFloat`: `v_float64`.
    pub const FLOAT: i32 = 3;
    /// `kTVMFFIOpaquePtr`: `v_ptr`.
    pub const OPAQUE_PTR: i32 = 4;
    /// `kTVMFFIDataType`: `v_dtype`.
    pub const DATA_TYPE: i32 = 5;
    /// `kTVMFFIDevice`: `v_device`.
    pub const DEVICE: i32 = 6;
    /// `kTVMFFIDLTensorPtr`: `v_ptr`, a `DLTensor*`.
    pub const DL_TENSOR_PTR: i32 = 7;
    /// `kTVMFFIRawStr`: `v_c_str`.
    pub const RAW_STR: i32 = 8;
    /// `kTVMFFIByteArrayPtr`.
    pub const BYTE_ARRAY_PTR: i32 = 9;
    /// `kTVMFFIObjectRValueRef`.
    pub const OBJECT_RVALUE_REF: i32 = 10;
    /// `kTVMFFISmallStr`.
    pub const SMALL_STR: i32 = 11;
    /// `kTVMFFISmallBytes`.
    pub const SMALL_BYTES: i32 = 12;
    /// `kTVMFFIObject`, the first object type index (`kTVMFFIStaticObjectBegin`).
    pub const OBJECT: i32 = 64;
    /// `kTVMFFIStr`.
    pub const STR: i32 = 65;
    /// `kTVMFFIBytes`.
    pub const BYTES: i32 = 66;
    /// `kTVMFFIError`.
    pub const ERROR: i32 = 67;
    /// `kTVMFFIFunction`.
    pub const FUNCTION: i32 = 68;
    /// `kTVMFFIShape`.
    pub const SHAPE: i32 = 69;
    /// `kTVMFFITensor`: `v_obj`, a tensor object whose `DLTensor` follows the
    /// [`TVMFFIObject`](super::TVMFFIObject) header.
    pub const TENSOR: i32 = 70;
    /// `kTVMFFIArray`.
    pub const ARRAY: i32 = 71;
    /// `kTVMFFIMap`.
    pub const MAP: i32 = 72;
    /// `kTVMFFIModule`.
    pub const MODULE: i32 = 73;
    /// `kTVMFFIOpaquePyObject`.
    pub const OPAQUE_PY_OBJECT: i32 = 74;
}

/// The 8-byte value of a [`TVMFFIAny`] (its anonymous union in C).
#[repr(C)]
#[derive(Clone, Copy)]
pub union TVMFFIAnyValue {
    pub v_int64: i64,
    pub v_float64: f64,
    pub v_ptr: *mut c_void,
    pub v_c_str: *const c_char,
    pub v_obj: *mut TVMFFIObject,
    pub v_dtype: DLDataType,
    pub v_device: DLDevice,
    pub v_bytes: [u8; 8],
    pub v_uint64: u64,
}

/// `TVMFFIAny`: a type index, 4 bytes of padding (or a small string's length) and an 8-byte
/// value; 16 bytes in all.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub zero_padding: u32,
    pub value: TVMFFIAnyValue,
}

/// `TVMFFIObject`: the header of every heap object (a reference count, its type index and the
/// deleter); a tensor object's `DLTensor` follows it.
#[repr(C)]
pub struct TVMFFIObject {
    pub combined_ref_count: u64,
    pub type_index: i32,
    pub padding: u32,
    pub deleter: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
}

const _: () = assert!(std::mem::size_of::<TVMFFIAny>() == 16 && std::mem::size_of::<TVMFFIObject>() == 24);

/// `TVMFFISafeCallType`: `int (*)(void* handle, const TVMFFIAny* args, int32_t num_args,
/// TVMFFIAny* result)`, returning 0, or -1 with the error raised.
pub type TVMFFISafeCallType = unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> c_int;

impl TVMFFIAny {
    fn with(type_index: i32, value: TVMFFIAnyValue) -> TVMFFIAny {
        TVMFFIAny { type_index, zero_padding: 0, value }
    }

    /// `None`.
    pub fn none() -> TVMFFIAny {
        TVMFFIAny::with(type_index::NONE, TVMFFIAnyValue { v_int64: 0 })
    }

    /// An integer (`kTVMFFIInt`).
    pub fn int(v: i64) -> TVMFFIAny {
        TVMFFIAny::with(type_index::INT, TVMFFIAnyValue { v_int64: v })
    }

    /// A Boolean (`kTVMFFIBool`).
    pub fn bool(v: bool) -> TVMFFIAny {
        TVMFFIAny::with(type_index::BOOL, TVMFFIAnyValue { v_int64: v as i64 })
    }

    /// A float (`kTVMFFIFloat`).
    pub fn float(v: f64) -> TVMFFIAny {
        TVMFFIAny::with(type_index::FLOAT, TVMFFIAnyValue { v_float64: v })
    }

    /// A borrowed `DLTensor*` (`kTVMFFIDLTensorPtr`).
    pub fn dl_tensor(t: *const DLTensor) -> TVMFFIAny {
        TVMFFIAny::with(type_index::DL_TENSOR_PTR, TVMFFIAnyValue { v_ptr: t as *mut c_void })
    }

    /// A tensor object (`kTVMFFITensor`): `obj` points at a [`TVMFFIObject`] header followed by
    /// a `DLTensor`.
    pub fn tensor_object(obj: *mut TVMFFIObject) -> TVMFFIAny {
        TVMFFIAny::with(type_index::TENSOR, TVMFFIAnyValue { v_obj: obj })
    }

    /// An opaque pointer (`kTVMFFIOpaquePtr`).
    pub fn opaque_ptr(p: *mut c_void) -> TVMFFIAny {
        TVMFFIAny::with(type_index::OPAQUE_PTR, TVMFFIAnyValue { v_ptr: p })
    }

    /// The Python-facing name of the value's type, for messages.
    pub fn type_name(&self) -> String {
        match self.type_index {
            type_index::NONE => "None".to_string(),
            type_index::INT => "int".to_string(),
            type_index::BOOL => "bool".to_string(),
            type_index::FLOAT => "float".to_string(),
            type_index::OPAQUE_PTR => "an opaque pointer".to_string(),
            type_index::DATA_TYPE => "a dtype".to_string(),
            type_index::DEVICE => "a device".to_string(),
            type_index::DL_TENSOR_PTR | type_index::TENSOR => "a tensor".to_string(),
            type_index::RAW_STR | type_index::SMALL_STR | type_index::STR => "str".to_string(),
            type_index::BYTE_ARRAY_PTR | type_index::SMALL_BYTES | type_index::BYTES => "bytes".to_string(),
            type_index::FUNCTION => "a function".to_string(),
            type_index::SHAPE => "a shape".to_string(),
            type_index::ARRAY => "an array (tuple or list)".to_string(),
            type_index::MAP => "a map (dict)".to_string(),
            type_index::MODULE => "a module".to_string(),
            type_index::OPAQUE_PY_OBJECT => "a Python object".to_string(),
            t => format!("an object of type index {t}"),
        }
    }
}

impl fmt::Debug for TVMFFIAny {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: every member of the union is 8 plain bytes.
        write!(f, "TVMFFIAny({}, {:#x})", self.type_index, unsafe { self.value.v_uint64 })
    }
}

// ---------------------------------------------------------------------------------------------
// The runtime functions.

/// `void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** message_parts,
/// int32_t num_parts)`.
pub type SetRaisedFromCStrParts = unsafe extern "C" fn(*const c_char, *const *const c_char, i32);

/// `TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id)`.
pub type EnvGetStream = unsafe extern "C" fn(i32, i32) -> *mut c_void;

/// The TVM-FFI runtime functions a call uses: where it reports errors and where it finds the
/// framework's stream.
#[derive(Clone, Copy, Debug)]
pub struct TvmFfiRuntime {
    pub set_raised_from_c_str_parts: SetRaisedFromCStrParts,
    pub env_get_stream: EnvGetStream,
}

static OVERRIDE: Mutex<Option<TvmFfiRuntime>> = Mutex::new(None);
static RESOLVED: OnceLock<TvmFfiRuntime> = OnceLock::new();

/// Use `runtime` instead of `libtvm_ffi`'s functions (stubs in a test, or a host without
/// TVM-FFI); `None` goes back to looking them up. Returns the previous override.
pub fn set_runtime_override(runtime: Option<TvmFfiRuntime>) -> Option<TvmFfiRuntime> {
    std::mem::replace(&mut *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()), runtime)
}

#[cfg(unix)]
mod dl {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    pub const RTLD_NOW: c_int = 2;
    #[cfg(target_os = "macos")]
    pub const RTLD_NOLOAD: c_int = 0x10;
    #[cfg(not(target_os = "macos"))]
    pub const RTLD_NOLOAD: c_int = 4;
}

/// The two functions from the process's global symbols, or from an already loaded
/// `libtvm_ffi` (never loading it).
#[cfg(unix)]
fn lookup() -> Option<TvmFfiRuntime> {
    let find = |handle: *mut c_void| -> Option<TvmFfiRuntime> {
        if handle.is_null() {
            return None;
        }
        // SAFETY: a live handle and NUL-terminated names.
        let (raise, stream) = unsafe {
            (
                dl::dlsym(handle, c"TVMFFIErrorSetRaisedFromCStrParts".as_ptr()),
                dl::dlsym(handle, c"TVMFFIEnvGetStream".as_ptr()),
            )
        };
        if raise.is_null() || stream.is_null() {
            return None;
        }
        // SAFETY: the symbols have these C signatures (c_api.h, c_env_api.h).
        unsafe {
            Some(TvmFfiRuntime {
                set_raised_from_c_str_parts: std::mem::transmute::<*mut c_void, SetRaisedFromCStrParts>(raise),
                env_get_stream: std::mem::transmute::<*mut c_void, EnvGetStream>(stream),
            })
        }
    };
    // SAFETY: dlopen(NULL) is the global scope; RTLD_NOLOAD only returns libraries already loaded.
    unsafe {
        find(dl::dlopen(std::ptr::null(), dl::RTLD_NOW))
            .or_else(|| find(dl::dlopen(c"libtvm_ffi.so".as_ptr(), dl::RTLD_NOW | dl::RTLD_NOLOAD)))
            .or_else(|| find(dl::dlopen(c"libtvm_ffi.dylib".as_ptr(), dl::RTLD_NOW | dl::RTLD_NOLOAD)))
    }
}

#[cfg(not(unix))]
fn lookup() -> Option<TvmFfiRuntime> {
    None
}

impl TvmFfiRuntime {
    /// The override ([`set_runtime_override`]) if there is one, else `libtvm_ffi`'s functions,
    /// looked up in the process (the first success is kept).
    pub fn resolve() -> Result<TvmFfiRuntime, String> {
        if let Some(rt) = *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) {
            return Ok(rt);
        }
        if let Some(rt) = RESOLVED.get() {
            return Ok(*rt);
        }
        let rt = lookup().ok_or_else(|| {
            "the TVM-FFI runtime is not loaded in this process (no TVMFFIErrorSetRaisedFromCStrParts and \
             TVMFFIEnvGetStream among its symbols): call the function through tvm_ffi, or install stubs with \
             cute::tvm_ffi::set_runtime_override"
                .to_string()
        })?;
        Ok(*RESOLVED.get_or_init(|| rt))
    }

    /// Raise `error` (`TVMFFIErrorSetRaisedFromCStrParts` with one part).
    pub fn raise(&self, error: &FfiError) {
        let kind = CString::new(error.kind).unwrap_or_default();
        let message = CString::new(error.message.replace('\0', " ")).unwrap_or_default();
        let parts = [message.as_ptr()];
        // SAFETY: NUL-terminated strings alive across the call, which copies them.
        unsafe { (self.set_raised_from_c_str_parts)(kind.as_ptr(), parts.as_ptr(), 1) };
    }

    /// The environment's stream for a device (null: the legacy default stream).
    pub fn stream(&self, device_type: i32, device_id: i32) -> *mut c_void {
        // SAFETY: a plain query.
        unsafe { (self.env_get_stream)(device_type, device_id) }
    }
}

// ---------------------------------------------------------------------------------------------
// The global function registry (`tvm_ffi.register_global_func`).

/// Register `f` as the TVM-FFI global function `name` (`TVMFFIFunctionCreate` and
/// `TVMFFIFunctionSetGlobal` of the already loaded `libtvm_ffi`), so Python finds it with
/// `tvm_ffi.get_global_func(name)`. With `override_existing` false a name already taken is an
/// error. The C signatures are those of `c_api.h` of [`TVM_FFI_VERSION`]; this is not run
/// against a real `libtvm_ffi` in cute-rs' tests (they have none).
#[cfg(unix)]
pub fn register_global_function(name: &str, f: TVMFFISafeCallType, override_existing: bool) -> Result<(), String> {
    #[repr(C)]
    struct ByteArray {
        data: *const c_char,
        size: usize,
    }
    type Create = unsafe extern "C" fn(
        *mut c_void,
        TVMFFISafeCallType,
        Option<unsafe extern "C" fn(*mut c_void)>,
        *mut *mut c_void,
    ) -> c_int;
    type SetGlobal = unsafe extern "C" fn(*const ByteArray, *mut c_void, c_int) -> c_int;
    type DecRef = unsafe extern "C" fn(*mut c_void) -> c_int;
    let find = |symbol: &std::ffi::CStr| -> *mut c_void {
        // SAFETY: dlopen(NULL) is the global scope; RTLD_NOLOAD never loads a library.
        unsafe {
            let global = dl::dlsym(dl::dlopen(std::ptr::null(), dl::RTLD_NOW), symbol.as_ptr());
            if !global.is_null() {
                return global;
            }
            let lib = dl::dlopen(c"libtvm_ffi.so".as_ptr(), dl::RTLD_NOW | dl::RTLD_NOLOAD);
            if lib.is_null() { lib } else { dl::dlsym(lib, symbol.as_ptr()) }
        }
    };
    let (create, set, dec) =
        (find(c"TVMFFIFunctionCreate"), find(c"TVMFFIFunctionSetGlobal"), find(c"TVMFFIObjectDecRef"));
    if create.is_null() || set.is_null() {
        return Err("the TVM-FFI runtime is not loaded in this process (no TVMFFIFunctionCreate and \
                    TVMFFIFunctionSetGlobal among its symbols): import tvm_ffi first"
            .to_string());
    }
    // SAFETY: the symbols have these C signatures (c_api.h); the name outlives the call, which
    // copies it, and the function object is released after the registry took its own reference.
    unsafe {
        let create: Create = std::mem::transmute(create);
        let set: SetGlobal = std::mem::transmute(set);
        let mut handle = std::ptr::null_mut();
        if create(std::ptr::null_mut(), f, None, &mut handle) != 0 {
            return Err(format!("TVMFFIFunctionCreate failed for `{name}`"));
        }
        let key = ByteArray { data: name.as_ptr() as *const c_char, size: name.len() };
        let rc = set(&key, handle, override_existing as c_int);
        if !dec.is_null() {
            let dec: DecRef = std::mem::transmute(dec);
            dec(handle);
        }
        if rc != 0 {
            return Err(format!("TVMFFIFunctionSetGlobal refused `{name}` (already registered?)"));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn register_global_function(_: &str, _: TVMFFISafeCallType, _: bool) -> Result<(), String> {
    Err("registering TVM-FFI global functions needs a Unix host".to_string())
}

// ---------------------------------------------------------------------------------------------
// Errors and argument decoding.

/// A failed call: the Python exception kind (`TypeError`, `ValueError`, `RuntimeError`) and
/// the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfiError {
    pub kind: &'static str,
    pub message: String,
}

impl FfiError {
    pub fn type_error(message: impl Into<String>) -> FfiError {
        FfiError { kind: "TypeError", message: message.into() }
    }

    pub fn value_error(message: impl Into<String>) -> FfiError {
        FfiError { kind: "ValueError", message: message.into() }
    }

    pub fn runtime_error(message: impl Into<String>) -> FfiError {
        FfiError { kind: "RuntimeError", message: message.into() }
    }
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for FfiError {}

/// The arguments of one call, read in order by [`FromTvmFfi::decode`].
pub struct FfiArgs<'a> {
    args: &'a [TVMFFIAny],
    next: usize,
}

impl<'a> FfiArgs<'a> {
    pub fn new(args: &'a [TVMFFIAny]) -> FfiArgs<'a> {
        FfiArgs { args, next: 0 }
    }

    /// The next argument and its position.
    pub fn take(&mut self) -> Result<(usize, &'a TVMFFIAny), FfiError> {
        let i = self.next;
        let a = self.args.get(i).ok_or_else(|| FfiError::type_error(format!("missing argument {i}")))?;
        self.next += 1;
        Ok((i, a))
    }

    /// How many arguments were read.
    pub fn position(&self) -> usize {
        self.next
    }
}

/// A host value a TVM-FFI call's arguments decode into: one argument for a tensor or a scalar,
/// consecutive arguments for a tuple. Messages say "argument {i}"; [`TvmFfiFunction`] adds
/// the argument's name and the function's signature.
pub trait FromTvmFfi: Sized {
    /// The number of positional arguments.
    fn arity() -> usize;
    /// The type of each argument, as the signature in messages shows it.
    fn type_names(out: &mut Vec<String>);
    fn decode(args: &mut FfiArgs<'_>) -> Result<Self, FfiError>;
}

fn device_name(d: DLDevice) -> String {
    let name = match d.device_type {
        device_type::KDL_CPU => "cpu",
        device_type::KDL_CUDA => "cuda",
        device_type::KDL_CUDA_HOST => "cuda_host (pinned host memory)",
        device_type::KDL_CUDA_MANAGED => "cuda_managed",
        device_type::KDL_ROCM => "rocm",
        device_type::KDL_METAL => "metal",
        _ => "an unknown device",
    };
    format!("{name}:{} (DLPack device type {})", d.device_id, d.device_type)
}

/// A DLPack tensor over CUDA memory, borrowed for the call: a `DLTensor*` or a tensor object.
impl FromTvmFfi for DlpackTensor {
    fn arity() -> usize {
        1
    }

    fn type_names(out: &mut Vec<String>) {
        out.push("Tensor".to_string());
    }

    fn decode(args: &mut FfiArgs<'_>) -> Result<DlpackTensor, FfiError> {
        let (i, a) = args.take()?;
        // SAFETY: the type index says which member is live.
        let t: *const DLTensor = match a.type_index {
            type_index::DL_TENSOR_PTR => unsafe { a.value.v_ptr as *const DLTensor },
            type_index::TENSOR => {
                let obj = unsafe { a.value.v_obj };
                if obj.is_null() {
                    std::ptr::null()
                } else {
                    // SAFETY: a tensor object is laid out as { TVMFFIObject, DLTensor, ... }.
                    unsafe { (obj as *const u8).add(std::mem::size_of::<TVMFFIObject>()) as *const DLTensor }
                }
            }
            _ => {
                return Err(FfiError::type_error(format!(
                    "argument {i} must be a tensor (a DLTensor* or a tvm_ffi.Tensor), got {}",
                    a.type_name()
                )));
            }
        };
        if t.is_null() {
            return Err(FfiError::value_error(format!("argument {i} is a null tensor")));
        }
        // SAFETY: the caller keeps the tensor alive for the call, which the host value does
        // not outlive.
        let t = unsafe { DlpackTensor::borrowed(t) };
        let d = t.dl_tensor().device;
        if !matches!(d.device_type, device_type::KDL_CUDA | device_type::KDL_CUDA_MANAGED) {
            return Err(FfiError::value_error(format!(
                "argument {i} is a tensor on {}; the kernel runs on CUDA tensors",
                device_name(d)
            )));
        }
        if d.device_id < 0 {
            return Err(FfiError::value_error(format!(
                "argument {i} is a tensor on {}, which is not a CUDA device ordinal",
                device_name(d)
            )));
        }
        Ok(t)
    }
}

macro_rules! ints {
    ($($t:ty => $name:literal),*) => {$(
        impl FromTvmFfi for $t {
            fn arity() -> usize {
                1
            }

            fn type_names(out: &mut Vec<String>) {
                out.push($name.to_string());
            }

            fn decode(args: &mut FfiArgs<'_>) -> Result<$t, FfiError> {
                let (i, a) = args.take()?;
                match a.type_index {
                    // SAFETY: an int or a bool holds v_int64.
                    type_index::INT | type_index::BOOL => {
                        let v = unsafe { a.value.v_int64 };
                        <$t>::try_from(v).map_err(|_| {
                            FfiError::value_error(format!(
                                "argument {i} is {v}, which does not fit in an {}",
                                $name
                            ))
                        })
                    }
                    _ => Err(FfiError::type_error(format!(
                        "argument {i} must be an int ({}), got {}",
                        $name,
                        a.type_name()
                    ))),
                }
            }
        }
    )*};
}
ints!(i8 => "Int8", i16 => "Int16", i32 => "Int32", i64 => "Int64", u8 => "Uint8", u16 => "Uint16",
      u32 => "Uint32", u64 => "Uint64");

macro_rules! floats {
    ($($t:ty => $name:literal),*) => {$(
        impl FromTvmFfi for $t {
            fn arity() -> usize {
                1
            }

            fn type_names(out: &mut Vec<String>) {
                out.push($name.to_string());
            }

            fn decode(args: &mut FfiArgs<'_>) -> Result<$t, FfiError> {
                let (i, a) = args.take()?;
                // SAFETY: the type index says which member is live.
                match a.type_index {
                    type_index::FLOAT => Ok(unsafe { a.value.v_float64 } as $t),
                    type_index::INT | type_index::BOOL => Ok(unsafe { a.value.v_int64 } as $t),
                    _ => Err(FfiError::type_error(format!(
                        "argument {i} must be a float ({}), got {}",
                        $name,
                        a.type_name()
                    ))),
                }
            }
        }
    )*};
}
floats!(f32 => "Float32", f64 => "Float64");

impl FromTvmFfi for bool {
    fn arity() -> usize {
        1
    }

    fn type_names(out: &mut Vec<String>) {
        out.push("Boolean".to_string());
    }

    fn decode(args: &mut FfiArgs<'_>) -> Result<bool, FfiError> {
        let (i, a) = args.take()?;
        match a.type_index {
            // SAFETY: an int or a bool holds v_int64.
            type_index::BOOL | type_index::INT => Ok(unsafe { a.value.v_int64 } != 0),
            _ => Err(FfiError::type_error(format!("argument {i} must be a bool, got {}", a.type_name()))),
        }
    }
}

impl FromTvmFfi for () {
    fn arity() -> usize {
        0
    }

    fn type_names(_: &mut Vec<String>) {}

    fn decode(_: &mut FfiArgs<'_>) -> Result<(), FfiError> {
        Ok(())
    }
}

macro_rules! tuples {
    ($($name:ident),+) => {
        impl<$($name: FromTvmFfi),+> FromTvmFfi for ($($name,)+) {
            fn arity() -> usize {
                0 $(+ $name::arity())+
            }

            fn type_names(out: &mut Vec<String>) {
                $($name::type_names(out);)+
            }

            fn decode(args: &mut FfiArgs<'_>) -> Result<Self, FfiError> {
                Ok(($($name::decode(args)?,)+))
            }
        }
    };
}
tuples!(A);
tuples!(A, B);
tuples!(A, B, C);
tuples!(A, B, C, D);
tuples!(A, B, C, D, E);
tuples!(A, B, C, D, E, F);
tuples!(A, B, C, D, E, F, G);
tuples!(A, B, C, D, E, F, G, H);

// ---------------------------------------------------------------------------------------------
// The callable.

/// The image a [`TvmFfiFunction`] loads.
enum Image {
    Compiled(Box<CompiledKernel>),
    Exported(Box<ExportedKernel>),
    /// A kernel of a module of several, loaded once per thread and device
    /// for every function of the module ([`TvmFfiFunction::from_module`]).
    Module(Box<multi::SharedModule>),
}

/// One launch value of a prepared call.
#[derive(Clone, Debug, PartialEq)]
pub enum PreparedArg {
    /// Device memory: the address, the bytes reachable from it and the element type.
    Pointer {
        address: u64,
        bytes: usize,
        elem: DType,
    },
    Scalar(KernelArg),
    ByValue(Vec<u8>),
}

/// A call decoded and checked, ready to launch ([`TvmFfiFunction::prepare`]).
#[derive(Clone, Debug)]
pub struct PreparedCall {
    /// The CUDA device ordinal of the tensors (0 when there are none).
    pub device: i32,
    pub config: LaunchConfig,
    pub args: Vec<PreparedArg>,
}

type ConfigFn<H> = Box<dyn Fn(&H) -> Result<LaunchConfig, String>>;

/// A kernel callable through TVM-FFI: the typed kernel, its compiled image and the function
/// that chooses the launch configuration from the decoded arguments (CuTe DSL's `@cute.jit`
/// host function, whose `.launch(grid=, block=)` this stands in for). See the
/// [module documentation](self).
///
/// It is not `Send`: [`tvm_ffi_export!`](crate::tvm_ffi_export) builds one per calling
/// thread, and the CUDA module it loads is kept per thread and device.
pub struct TvmFfiFunction<P: KernelParam>
where
    P::Host: FromTvmFfi,
{
    name: String,
    kernel: TypedKernel<P>,
    signature: KernelSignature,
    image: Image,
    config: ConfigFn<P::Host>,
    arg_names: Vec<String>,
    #[cfg(feature = "cuda")]
    loaded: std::cell::RefCell<Vec<(i32, &'static crate::cuda::CudaFunction<'static>)>>,
}

impl<P: KernelParam> fmt::Debug for TvmFfiFunction<P>
where
    P::Host: FromTvmFfi,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TvmFfiFunction({})", self.signature_text())
    }
}

/// Whether two signatures declare the same parameters.
fn same_params(a: &KernelSignature, b: &KernelSignature) -> bool {
    a.params == b.params
}

impl<P: KernelParam> TvmFfiFunction<P>
where
    P::Host: FromTvmFfi,
{
    fn with_image(
        kernel: TypedKernel<P>,
        signature: KernelSignature,
        image: Image,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, String> {
        let ir = kernel.kernel().signature();
        if ir.hidden_param_count() > 0 {
            return Err(format!(
                "kernel `{}` has hidden tensor-map parameters (tma_descriptor), which TVM-FFI callables do not \
                 encode yet: bind host-built tensor maps with param_tensormap instead",
                kernel.name()
            ));
        }
        if !same_params(&ir, &signature) {
            return Err(format!(
                "the image's kernel takes {signature}, but the typed kernel `{}` takes {ir}: they are not the \
                 same kernel",
                kernel.name()
            ));
        }
        let arity = <P::Host as FromTvmFfi>::arity();
        Ok(TvmFfiFunction {
            name: kernel.name().to_string(),
            arg_names: (0..arity).map(|i| format!("arg{i}")).collect(),
            kernel,
            signature,
            image,
            config: Box::new(config),
            #[cfg(feature = "cuda")]
            loaded: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// The callable of `kernel`, compiled as `compiled` (which must be compiled from it).
    pub fn new(
        kernel: TypedKernel<P>,
        compiled: CompiledKernel,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, String> {
        let signature = compiled.signature().clone();
        TvmFfiFunction::with_image(kernel, signature, Image::Compiled(Box::new(compiled)), config)
    }

    /// Compile `kernel` with `options` plus `CompileOptions::enable_tvm_ffi` and make its
    /// callable. The target is the options', else the kernel's, else (feature `cuda`) the best
    /// one for CUDA device 0.
    pub fn compile(
        kernel: TypedKernel<P>,
        toolchain: &crate::Toolchain,
        options: &crate::CompileOptions,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, Box<dyn std::error::Error>> {
        let options = options.clone().enable_tvm_ffi(true);
        #[cfg(feature = "cuda")]
        let options = match options.get_gpu_arch().or(kernel.target()) {
            Some(_) => options,
            None => {
                let ctx = crate::cuda::CudaContext::new(0)?;
                let arch = ctx.best_target(toolchain).ok_or("no target cute-rs knows runs on CUDA device 0")?;
                options.gpu_arch(arch)
            }
        };
        #[cfg(not(feature = "cuda"))]
        if options.get_gpu_arch().or(kernel.target()).is_none() {
            return Err("TvmFfiFunction::compile: name a target (CompileOptions::gpu_arch or CUTE_RS_ARCH)".into());
        }
        let compiled = kernel.compile_with(toolchain, &options)?;
        Ok(TvmFfiFunction::new(kernel, compiled, config)?)
    }

    /// The callable of `kernel` from its exported image ([`ExportedKernel`], read from the
    /// files [`CompiledKernel::export_c`] wrote or embedded with `include_str!`): no
    /// `mlir-opt` is needed when it is called. The export must be of the same kernel.
    pub fn from_exported(
        kernel: TypedKernel<P>,
        exported: ExportedKernel,
        config: impl Fn(&P::Host) -> Result<LaunchConfig, String> + 'static,
    ) -> Result<TvmFfiFunction<P>, String> {
        let signature = exported.signature.clone();
        TvmFfiFunction::with_image(kernel, signature, Image::Exported(Box::new(exported)), config)
    }

    /// Name the positional arguments in messages and the signature (`arg0`, `arg1`, ... by
    /// default). Panics unless there is one name per argument.
    pub fn with_arg_names(mut self, names: &[&str]) -> Self {
        assert_eq!(
            names.len(),
            self.arg_names.len(),
            "cute: TvmFfiFunction::with_arg_names: `{}` takes {} arguments",
            self.name,
            self.arg_names.len()
        );
        self.arg_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Call it something other than the kernel's name in messages.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kernel(&self) -> &TypedKernel<P> {
        &self.kernel
    }

    /// The compiled kernel it loads, when it was made from one.
    pub fn compiled(&self) -> Option<&CompiledKernel> {
        match &self.image {
            Image::Compiled(k) => Some(k),
            Image::Module(m) => m.compiled(&self.signature.name),
            Image::Exported(_) => None,
        }
    }

    /// The exported kernel it loads, when it was made from one.
    pub fn exported(&self) -> Option<&ExportedKernel> {
        match &self.image {
            Image::Exported(k) => Some(k),
            Image::Compiled(_) | Image::Module(_) => None,
        }
    }

    /// The compiled kernel's signature, which launches are checked and packed by.
    pub fn signature(&self) -> &KernelSignature {
        &self.signature
    }

    /// The signature messages show: `add_one(a: Tensor, b: Tensor)`.
    pub fn signature_text(&self) -> String {
        let mut types = Vec::new();
        <P::Host as FromTvmFfi>::type_names(&mut types);
        let args: Vec<String> = self.arg_names.iter().zip(types).map(|(n, t)| format!("{n}: {t}")).collect();
        format!("{}({})", self.name, args.join(", "))
    }

    /// `message` about argument `i` (or the call), with the argument's name and the signature.
    fn context(&self, at: Option<usize>, message: &str) -> String {
        // Kernel parameter names (`arg1`, `arg1_shape_0`) read as the argument names.
        let mut message = message.to_string();
        for (i, n) in self.arg_names.iter().enumerate().rev() {
            message = message.replace(&format!("`arg{i}"), &format!("`{n}"));
        }
        let message = message.as_str();
        match at.and_then(|i| self.arg_names.get(i).map(|n| (i, n))) {
            Some((i, n)) => format!("{}: argument {i} (`{n}`): {message}", self.signature_text()),
            None => format!("{}: {message}", self.signature_text()),
        }
    }

    /// The position a kernel parameter or decoding message is about: `argN...` leaf names
    /// and "argument N" messages name it.
    fn position_of(text: &str) -> Option<usize> {
        let rest = text.strip_prefix("argument ").or_else(|| text.strip_prefix("arg"))?;
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }

    /// Decode the arguments into the kernel's host value.
    pub fn decode(&self, args: &[TVMFFIAny]) -> Result<P::Host, FfiError> {
        let want = <P::Host as FromTvmFfi>::arity();
        if args.len() != want {
            return Err(FfiError::type_error(
                self.context(None, &format!("takes {want} arguments, got {}", args.len())),
            ));
        }
        let mut cursor = FfiArgs::new(args);
        <P::Host as FromTvmFfi>::decode(&mut cursor).map_err(|e| {
            let at = Self::position_of(&e.message);
            let message = match at {
                Some(i) => e.message.strip_prefix(&format!("argument {i} ")).unwrap_or(&e.message).to_string(),
                None => e.message.clone(),
            };
            FfiError { kind: e.kind, message: self.context(at, &message) }
        })
    }

    /// Decode and check a call without launching it: the host value against the kernel's
    /// specs, the devices, the packed values against the signature, and the launch
    /// configuration.
    pub fn prepare(&self, args: &[TVMFFIAny]) -> Result<PreparedCall, FfiError> {
        let host = self.decode(args)?;
        let pack: ArgPack = self.kernel.pack_full(&host).map_err(|e| {
            let at = Self::position_of(&e.path);
            let message = if at.is_some_and(|i| e.path == format!("arg{i}")) {
                e.message.clone()
            } else {
                format!("`{}`: {}", e.path, e.message)
            };
            FfiError::value_error(self.context(at, &message))
        })?;
        let mut device = None;
        let mut out = Vec::with_capacity(pack.args().len());
        for (i, ((arg, name), param)) in pack.args().iter().zip(pack.names()).zip(&self.signature.params).enumerate() {
            let external = pack.externals().get(i).copied().flatten();
            out.push(match (&param.kind, external.and_then(|e| e.device)) {
                (ParamKind::Pointer { elem, .. }, Some((address, bytes, id))) => {
                    match device {
                        None => device = Some((id, name.clone())),
                        Some((d, ref first)) if d != id => {
                            return Err(FfiError::value_error(self.context(
                                Self::position_of(name),
                                &format!("is memory of CUDA device {id}, but `{first}` is on device {d}"),
                            )));
                        }
                        Some(_) => {}
                    }
                    PreparedArg::Pointer { address, bytes, elem: *elem }
                }
                (ParamKind::Pointer { .. }, None) => {
                    return Err(FfiError::type_error(self.context(
                        Self::position_of(name),
                        &format!("`{name}` is not device memory: a TVM-FFI call takes CUDA tensors"),
                    )));
                }
                (ParamKind::Scalar { .. }, _) => PreparedArg::Scalar(arg.clone()),
                (ParamKind::ByValue { .. }, _) => PreparedArg::ByValue(arg.byte_data().to_vec()),
            });
        }
        let values: Vec<LaunchValue<'_>> = out
            .iter()
            .map(|a| match a {
                PreparedArg::Pointer { address, bytes, .. } => {
                    LaunchValue::Pointer { address: *address, bytes: Some(*bytes) }
                }
                PreparedArg::Scalar(s) => LaunchValue::Scalar(s),
                PreparedArg::ByValue(b) => LaunchValue::Bytes(b),
            })
            .collect();
        if let Err(e) = self.signature.pack(&values) {
            // "argument {i}: ..." / "argument {i} (`name`) ..." count kernel parameters.
            let at = e
                .strip_prefix("argument ")
                .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|k| pack.names().get(k))
                .and_then(|n| Self::position_of(n));
            // Drop the launcher's own "argument {k}: " prefix, which counts kernel parameters.
            let e = match e.split_once(": ") {
                Some((head, rest)) if head.starts_with("argument ") && !head.contains('`') => rest.to_string(),
                _ => e,
            };
            return Err(FfiError::value_error(self.context(at, &e)));
        }
        let config = (self.config)(&host).map_err(|e| FfiError::value_error(self.context(None, &e)))?;
        Ok(PreparedCall { device: device.map_or(0, |d| d.0), config, args: out })
    }

    /// Decode, check and launch a call on the environment's stream for the tensors' device.
    pub fn call(&self, args: &[TVMFFIAny]) -> Result<(), FfiError> {
        let prepared = self.prepare(args)?;
        self.launch(&prepared)
    }

    #[cfg(not(feature = "cuda"))]
    fn launch(&self, _: &PreparedCall) -> Result<(), FfiError> {
        Err(FfiError::runtime_error(
            self.context(None, "cute-rs was built without the `cuda` feature, so it cannot launch kernels"),
        ))
    }

    #[cfg(feature = "cuda")]
    fn launch(&self, p: &PreparedCall) -> Result<(), FfiError> {
        let rt = TvmFfiRuntime::resolve().map_err(|e| FfiError::runtime_error(self.context(None, &e)))?;
        self.launch_on(p, rt.stream(device_type::KDL_CUDA, p.device))
    }

    /// Decode, check and launch a call on `stream` (a raw `CUstream` of the tensors' device, or
    /// null for the legacy stream) instead of the environment's: the stream an XLA custom call
    /// is given ([`crate::xla_ffi`]).
    ///
    /// # Safety
    ///
    /// `stream` is null or a live stream of the primary context of the tensors' device.
    #[cfg(feature = "cuda")]
    pub unsafe fn call_on_stream(&self, args: &[TVMFFIAny], stream: *mut c_void) -> Result<(), FfiError> {
        let prepared = self.prepare(args)?;
        self.launch_on(&prepared, stream)
    }

    #[cfg(feature = "cuda")]
    fn launch_on(&self, p: &PreparedCall, raw_stream: *mut c_void) -> Result<(), FfiError> {
        use crate::cuda::{CudaStream, DeviceArg, DevicePtr};
        let runtime = |e: crate::cuda::CudaError| FfiError::runtime_error(self.context(None, &e.to_string()));
        let ctx = context(p.device).map_err(runtime)?;
        let function = self.function(ctx, p.device).map_err(runtime)?;
        // SAFETY: the stream is a live stream of the device's primary context (or null, the
        // legacy stream) for the duration of the call.
        let stream = unsafe { CudaStream::from_raw(ctx, raw_stream) };
        let mut args = Vec::with_capacity(p.args.len());
        for a in &p.args {
            args.push(match a {
                // SAFETY: CUDA memory of this device, alive for the call, as the caller's
                // DLPack tensor promises; the alignment was checked against the signature.
                PreparedArg::Pointer { address, bytes, elem } => DeviceArg::Ptr(
                    unsafe { DevicePtr::from_raw(*elem, *address, Some(1), Some(*bytes)) }.map_err(runtime)?,
                ),
                PreparedArg::Scalar(s) => DeviceArg::Scalar(s.clone()),
                PreparedArg::ByValue(b) => DeviceArg::ByValue(b.clone()),
            });
        }
        function.launch_builder(p.config).map_err(runtime)?.stream(&stream).args(&args).submit().map_err(runtime)
    }

    /// The function loaded on device `device`'s context (once per thread and device).
    #[cfg(feature = "cuda")]
    fn function(
        &self,
        ctx: &'static crate::cuda::CudaContext,
        device: i32,
    ) -> Result<&'static crate::cuda::CudaFunction<'static>, crate::cuda::CudaError> {
        if let Some((_, f)) = self.loaded.borrow().iter().find(|(d, _)| *d == device) {
            return Ok(f);
        }
        let f = match &self.image {
            Image::Compiled(k) => ctx.load(k)?,
            Image::Exported(k) => ctx.load_exported_kernel(k)?,
            Image::Module(m) => {
                let f = m.function(ctx, device, &self.signature.name)?;
                self.loaded.borrow_mut().push((device, f));
                return Ok(f);
            }
        };
        // Leaked with its context: TLS destructors at exit must not unload modules.
        let f: &'static crate::cuda::CudaFunction<'static> = Box::leak(Box::new(f));
        self.loaded.borrow_mut().push((device, f));
        Ok(f)
    }

    /// The safe call: decode `num_args` arguments at `args`, check and launch, set `result` to
    /// `None` and return 0; or raise the error through the TVM-FFI runtime and return -1.
    /// A panic is caught and raised as a `RuntimeError`.
    ///
    /// # Safety
    ///
    /// `args` points at `num_args` valid [`TVMFFIAny`]s (or is null with `num_args` 0), whose
    /// tensors stay alive for the call, and `result` is null or writable.
    pub unsafe fn safe_call(&self, args: *const TVMFFIAny, num_args: i32, result: *mut TVMFFIAny) -> c_int {
        // SAFETY: the caller's contract.
        unsafe { finish(raw_args(args, num_args).and_then(|a| self.call(a)), result) }
    }
}

/// The argument slice of a safe call.
///
/// # Safety
///
/// As [`TvmFfiFunction::safe_call`].
unsafe fn raw_args<'a>(args: *const TVMFFIAny, num_args: i32) -> Result<&'a [TVMFFIAny], FfiError> {
    match (args.is_null(), num_args) {
        (_, n) if n < 0 => Err(FfiError::type_error(format!("a negative argument count {n}"))),
        (_, 0) => Ok(&[]),
        (true, _) => Err(FfiError::type_error("a null argument array")),
        // SAFETY: the caller's contract.
        (false, n) => Ok(unsafe { std::slice::from_raw_parts(args, n as usize) }),
    }
}

/// Report `outcome`: `None` into `result` and 0, or the error raised and -1.
///
/// # Safety
///
/// `result` is null or writable.
unsafe fn finish(outcome: Result<(), FfiError>, result: *mut TVMFFIAny) -> c_int {
    match outcome {
        Ok(()) => {
            if !result.is_null() {
                // SAFETY: the caller's contract.
                unsafe { result.write(TVMFFIAny::none()) };
            }
            0
        }
        Err(e) => {
            match TvmFfiRuntime::resolve() {
                Ok(rt) => rt.raise(&e),
                Err(why) => {
                    // Not `eprintln!`, which panics (across the C boundary) if stderr is gone.
                    use std::io::Write;
                    let _ = writeln!(std::io::stderr(), "cute: {e} (and it cannot be raised: {why})");
                }
            }
            -1
        }
    }
}

/// The primary context of device `device`, retained once per thread and kept for the process.
#[cfg(feature = "cuda")]
fn context(device: i32) -> Result<&'static crate::cuda::CudaContext, crate::cuda::CudaError> {
    thread_local! {
        static CONTEXTS: std::cell::RefCell<Vec<(i32, &'static crate::cuda::CudaContext)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    CONTEXTS.with(|c| {
        if let Some((_, ctx)) = c.borrow().iter().find(|(d, _)| *d == device) {
            return Ok(*ctx);
        }
        // A negative ordinal (refused when decoding) becomes one cuDeviceGet refuses.
        let ordinal = usize::try_from(device).unwrap_or(i32::MAX as usize);
        let ctx: &'static crate::cuda::CudaContext = Box::leak(Box::new(crate::cuda::CudaContext::new(ordinal)?));
        c.borrow_mut().push((device, ctx));
        Ok(ctx)
    })
}

/// A [`TvmFfiFunction`] of any parameters, as [`tvm_ffi_export!`](crate::tvm_ffi_export)
/// stores it.
pub trait TvmFfiCallable {
    fn name(&self) -> &str;
    /// Decode, check and launch a call.
    fn call_any(&self, args: &[TVMFFIAny]) -> Result<(), FfiError>;
    /// Decode, check and launch a call on a given stream.
    ///
    /// # Safety
    ///
    /// `stream` is null or a live `CUstream` of the primary context of the tensors' device.
    #[cfg(feature = "cuda")]
    unsafe fn call_any_on_stream(&self, args: &[TVMFFIAny], stream: *mut c_void) -> Result<(), FfiError>;
}

impl<P: KernelParam> TvmFfiCallable for TvmFfiFunction<P>
where
    P::Host: FromTvmFfi,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn call_any(&self, args: &[TVMFFIAny]) -> Result<(), FfiError> {
        self.call(args)
    }

    #[cfg(feature = "cuda")]
    unsafe fn call_any_on_stream(&self, args: &[TVMFFIAny], stream: *mut c_void) -> Result<(), FfiError> {
        // SAFETY: the caller's contract.
        unsafe { self.call_on_stream(args, stream) }
    }
}

/// What an exported function's constructor may return: a [`TvmFfiFunction`], or a `Result`
/// of one.
pub trait IntoTvmFfiCallable {
    fn into_callable(self) -> Result<Box<dyn TvmFfiCallable>, String>;
}

impl<P: KernelParam + 'static> IntoTvmFfiCallable for TvmFfiFunction<P>
where
    P::Host: FromTvmFfi,
{
    fn into_callable(self) -> Result<Box<dyn TvmFfiCallable>, String> {
        Ok(Box::new(self))
    }
}

impl<P: KernelParam + 'static, E: fmt::Display> IntoTvmFfiCallable for Result<TvmFfiFunction<P>, E>
where
    P::Host: FromTvmFfi,
{
    fn into_callable(self) -> Result<Box<dyn TvmFfiCallable>, String> {
        self.map(|f| Box::new(f) as Box<dyn TvmFfiCallable>).map_err(|e| e.to_string())
    }
}

/// The cell an exported function keeps its callable in (one per thread). Only a built
/// callable is kept: a constructor that failed runs again on the next call.
pub type ExportCell = OnceCell<Box<dyn TvmFfiCallable>>;

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// The body of a function [`tvm_ffi_export!`](crate::tvm_ffi_export) generates: build the
/// callable on this thread's first call (a failure is raised, and the next call builds it
/// again, so a missing toolchain or environment variable fixed since takes effect), then
/// make the safe call. Panics are caught and raised as `RuntimeError`s.
///
/// # Safety
///
/// As [`TvmFfiFunction::safe_call`].
pub unsafe fn export_call<R: IntoTvmFfiCallable>(
    cell: &ExportCell,
    name: &str,
    make: impl FnOnce() -> R,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> c_int {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let callable = match cell.get() {
            Some(c) => c,
            None => {
                let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| make().into_callable()))
                    .unwrap_or_else(|p| Err(format!("building it panicked: {}", panic_message(&*p))))
                    .map_err(|e| {
                        FfiError::runtime_error(format!("{SYMBOL_PREFIX}{name}: the kernel is unavailable: {e}"))
                    })?;
                cell.get_or_init(|| built)
            }
        };
        // SAFETY: the caller's contract.
        let args = unsafe { raw_args(args, num_args) }?;
        callable.call_any(args)
    }))
    .unwrap_or_else(|p| {
        Err(FfiError::runtime_error(format!("{SYMBOL_PREFIX}{name} panicked: {}", panic_message(&*p))))
    });
    // SAFETY: the caller's contract.
    unsafe { finish(outcome, result) }
}

/// [`export_call`] for a call made while the thread is exiting, when its callable is gone:
/// raise a `RuntimeError` and return -1.
#[doc(hidden)]
pub fn export_call_on_exiting_thread(name: &str) -> c_int {
    let e = FfiError::runtime_error(format!("{SYMBOL_PREFIX}{name}: called while its thread is exiting"));
    // SAFETY: a null result is never written.
    unsafe { finish(Err(e), std::ptr::null_mut()) }
}

/// Export a [`TvmFfiFunction`] from a `cdylib` as the TVM-FFI function `__tvm_ffi_<name>`
/// (the Rust function is `name`, with the safe-call signature [`TVMFFISafeCallType`]), which
/// `tvm_ffi.load_module(path).<name>(...)` calls. `$make` is evaluated on each calling
/// thread's first call and gives a `TvmFfiFunction` or a `Result` of one (whose error the
/// call raises as a `RuntimeError`; the next call evaluates `$make` again).
///
/// ```ignore
/// cute::tvm_ffi_export!(add_one, build_add_one());
/// ```
#[macro_export]
macro_rules! tvm_ffi_export {
    ($name:ident, $make:expr) => {
        /// A TVM-FFI safe call (`__tvm_ffi_` + the name), exported by `cute::tvm_ffi_export!`.
        ///
        /// # Safety
        ///
        /// `args` points at `num_args` valid `TVMFFIAny`s whose tensors live for the call, and
        /// `result` is null or writable.
        #[unsafe(export_name = concat!("__tvm_ffi_", stringify!($name)))]
        pub unsafe extern "C" fn $name(
            _handle: *mut ::std::ffi::c_void,
            args: *const $crate::tvm_ffi::TVMFFIAny,
            num_args: i32,
            result: *mut $crate::tvm_ffi::TVMFFIAny,
        ) -> ::std::ffi::c_int {
            ::std::thread_local! {
                static CALLABLE: $crate::tvm_ffi::ExportCell = const { ::std::cell::OnceCell::new() };
            }
            CALLABLE
                .try_with(|cell| {
                    // SAFETY: this function's contract.
                    unsafe { $crate::tvm_ffi::export_call(cell, stringify!($name), || $make, args, num_args, result) }
                })
                .unwrap_or_else(|_| $crate::tvm_ffi::export_call_on_exiting_thread(stringify!($name)))
        }
    };
}

/// Generate `cute_tvm_ffi_register_all()`, which registers functions made by
/// [`tvm_ffi_export!`](crate::tvm_ffi_export) in the global registry
/// ([`register_global_function`]) under the given names: `register_all!(add_one,
/// "my.add" => add_other)` (a bare ident registers under its own name). It returns the
/// number of failures (0: all registered) after trying every name, so a loader calls it
/// once (from Python, `ctypes.CDLL(path).cute_tvm_ffi_register_all()`) after importing
/// `tvm_ffi`.
#[macro_export]
macro_rules! tvm_ffi_register_all {
    ($($($key:literal =>)? $name:ident),+ $(,)?) => {
        /// Register this library's TVM-FFI functions globally; the number of failures.
        #[unsafe(no_mangle)]
        pub extern "C" fn cute_tvm_ffi_register_all() -> ::std::ffi::c_int {
            let mut failures = 0;
            $(
                let key: &str = $crate::tvm_ffi_register_all!(@key $($key)? $name);
                if $crate::tvm_ffi::register_global_function(key, $name, true).is_err() {
                    failures += 1;
                }
            )+
            failures
        }
    };
    (@key $key:literal $name:ident) => { $key };
    (@key $name:ident) => { stringify!($name) };
}

/// Register at load time (Linux `.init_array`): call [`tvm_ffi_register_all!`]'s
/// `cute_tvm_ffi_register_all` when the library is loaded, ignoring failure (a process
/// without `libtvm_ffi` loaded, such as a C host, just skips it). Opt in per library
/// because it runs code during `dlopen`.
#[macro_export]
macro_rules! tvm_ffi_register_on_load {
    () => {
        #[cfg(target_os = "linux")]
        #[used]
        #[unsafe(link_section = ".init_array")]
        static CUTE_TVM_FFI_REGISTER_ON_LOAD: extern "C" fn() = {
            extern "C" fn on_load() {
                let _ = cute_tvm_ffi_register_all();
            }
            on_load
        };
    };
}

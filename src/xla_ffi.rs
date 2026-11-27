//! An XLA FFI handler for JAX (the Rust counterpart of CuTe DSL's `cutlass.jax.cutlass_call`):
//! cute-rs kernels as XLA custom calls inside `jax.jit`.
//!
//! JAX lowers `jax.ffi.ffi_call` to a custom call whose target is a C function with the
//! signature of `xla/ffi/api/c_api.h`: `XLA_FFI_Error* handler(XLA_FFI_CallFrame*)`. This
//! module declares the parts of that ABI a handler reads ([`XlaCallFrame`], [`XlaApi`],
//! [`XlaBuffer`], the attribute types) with hand-written `#[repr(C)]` structs, checked against
//! OpenXLA's header at XLA FFI API 0.5, so the core crate links nothing and needs no XLA
//! headers.
//!
//! # How a call runs
//!
//! [`xla_ffi_export!`](crate::xla_ffi_export) defines the handler symbols in a `cdylib`
//! (`cute_xla_cutlass_call`, a `_nograph` twin and a handler bundle); both call
//! [`handle_call_frame`]. In the execute stage the handler
//!
//! 1. reads the argument and result buffers (dtype, device pointer, dims) and the attributes;
//!    the string attribute `cute_target` names a target registered with
//!    [`register_cutlass_call`]; the integer-array attributes `arg_m2m_<i>` and `ret_m2m_<i>`
//!    give a buffer's XLA layout (minor-to-major; default row-major); the integer attribute
//!    `device` overrides the CUDA device (default: the thread's current one, which XLA makes
//!    current); scalar attributes (integers, floats, Booleans) become scalar kernel
//!    arguments after the tensors, in name order;
//! 2. looks the call up in a per-thread cache keyed by the target and the specs of every
//!    buffer and attribute ([`XlaCallSpec`]), calling the target's builder on a miss;
//! 3. checks declared aliases (a result that must share an argument's memory, the
//!    `input_output_aliases` of `cutlass_call`);
//! 4. passes the buffers as DLPack tensors (strides from the layout), arguments then results,
//!    to the [`TvmFfiCallable`] the builder made, which checks them against the kernel's
//!    specs and launches it on the stream `XLA_FFI_Stream_Get` returns (feature `cuda`).
//!
//! The instantiate, prepare and initialize stages do nothing. Errors reach XLA through
//! `XLA_FFI_Error_Create` with an absl-style code (invalid argument, not found, failed
//! precondition, internal) and a message naming the target and the buffer.
//!
//! The builder gets the call's specs and returns a [`XlaTarget`]: usually a
//! [`TvmFfiFunction`](crate::tvm_ffi::TvmFfiFunction) compiled for them, or one made
//! from an [`ExportedKernel`](crate::aot::ExportedKernel) with
//! [`TvmFfiFunction::from_exported`](crate::tvm_ffi::TvmFfiFunction::from_exported), which
//! runs without `mlir-opt` (the AOT form of the kernel: the exported PTX or cubin and its
//! signature).
//!
//! # Struct layouts
//!
//! XLA's header is not part of this repository and the structs follow the API 0.x layout from
//! memory of `c_api.h`: `struct_size` and `extension_start` open every struct, the API table
//! lists `Error_Create`, `Error_GetMessage`, `Error_Destroy`, `Handler_Register`, `Stream_Get`
//! in that order after the version and the internal API pointer, and the call frame holds
//! the API, the execution context, the stage, the arguments, the results and the attributes.
//! [`XlaApi`] stops after `Stream_Get`, which is all a handler uses. The layouts are only
//! exercised by tests that build frames with these same structs, not against a real XLA.
//!
//! Out of scope: sharding (`shard_map`, `custom_partitioning`) is the Python side's job;
//! typed handler state, futures and non-buffer arguments are not read.

// The handler plumbing is used only when the `cuda` feature builds the runtime that calls it.
#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::external::dlpack::{DLDevice, DLTensor, device_type};
use crate::host::dl_data_type;
use crate::ir::DType;
use crate::tvm_ffi::{FfiError, TVMFFIAny, TvmFfiCallable};

// ---------------------------------------------------------------------------------------------
// The XLA FFI C ABI.

/// `XLA_FFI_Error`: opaque to the handler.
#[repr(C)]
pub struct XlaError {
    _private: [u8; 0],
}

/// `XLA_FFI_ExecutionContext`: opaque.
#[repr(C)]
pub struct XlaExecutionContext {
    _private: [u8; 0],
}

/// `XLA_FFI_ExecutionStage`.
pub mod stage {
    pub const INSTANTIATE: i32 = 0;
    pub const PREPARE: i32 = 1;
    pub const INITIALIZE: i32 = 2;
    pub const EXECUTE: i32 = 3;
    /// Recording into a command buffer (XLA FFI API 0.5).
    pub const RECORD: i32 = 4;
}

/// `XLA_FFI_Error_Code` (the absl status codes).
pub mod error_code {
    pub const INVALID_ARGUMENT: i32 = 3;
    pub const NOT_FOUND: i32 = 5;
    pub const FAILED_PRECONDITION: i32 = 9;
    pub const INTERNAL: i32 = 13;
    pub const UNIMPLEMENTED: i32 = 12;
}

/// `XLA_FFI_ArgType` / `XLA_FFI_RetType` of a buffer.
pub const BUFFER_TYPE: i32 = 1;

/// `XLA_FFI_AttrType`.
pub mod attr_type {
    pub const ARRAY: i32 = 1;
    pub const DICTIONARY: i32 = 2;
    pub const SCALAR: i32 = 3;
    pub const STRING: i32 = 4;
}

/// `XLA_FFI_DataType` (the values of XLA's `PrimitiveType`).
pub mod data_type {
    pub const INVALID: i32 = 0;
    pub const PRED: i32 = 1;
    pub const S8: i32 = 2;
    pub const S16: i32 = 3;
    pub const S32: i32 = 4;
    pub const S64: i32 = 5;
    pub const U8: i32 = 6;
    pub const U16: i32 = 7;
    pub const U32: i32 = 8;
    pub const U64: i32 = 9;
    pub const F16: i32 = 10;
    pub const F32: i32 = 11;
    pub const F64: i32 = 12;
    pub const BF16: i32 = 16;
    pub const F8E5M2: i32 = 19;
    pub const F8E4M3FN: i32 = 20;
    pub const S4: i32 = 21;
    pub const F8E4M3B11FNUZ: i32 = 23;
    pub const F8E4M3: i32 = 28;
    pub const F4E2M1FN: i32 = 32;
    pub const F8E8M0FNU: i32 = 33;
}

/// `XLA_FFI_Error_Create_Args`.
#[repr(C)]
pub struct XlaErrorCreateArgs {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub message: *const c_char,
    pub errc: i32,
}

/// `XLA_FFI_Stream_Get_Args`.
#[repr(C)]
pub struct XlaStreamGetArgs {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub ctx: *mut XlaExecutionContext,
    pub stream: *mut c_void,
}

/// `XLA_FFI_Api_Version`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XlaApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub major_version: c_int,
    pub minor_version: c_int,
}

/// The head of `XLA_FFI_Api`, up to `XLA_FFI_Stream_Get`.
#[repr(C)]
pub struct XlaApi {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub api_version: XlaApiVersion,
    pub internal_api: *mut c_void,
    pub error_create: Option<unsafe extern "C" fn(*mut XlaErrorCreateArgs) -> *mut XlaError>,
    pub error_get_details: *mut c_void,
    pub error_destroy: *mut c_void,
    pub handler_register: *mut c_void,
    pub stream_get: Option<unsafe extern "C" fn(*mut XlaStreamGetArgs) -> *mut XlaError>,
}

/// `XLA_FFI_ByteSpan`.
#[repr(C)]
pub struct XlaByteSpan {
    pub ptr: *const c_char,
    pub len: usize,
}

/// `XLA_FFI_Buffer`: a dense device buffer whose `dims` are in logical order.
#[repr(C)]
pub struct XlaBuffer {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub dtype: i32,
    pub data: *mut c_void,
    pub rank: i64,
    pub dims: *mut i64,
}

/// `XLA_FFI_Args`.
#[repr(C)]
pub struct XlaArgs {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub size: i64,
    pub types: *mut i32,
    pub args: *mut *mut c_void,
}

/// `XLA_FFI_Rets` (the same shape as [`XlaArgs`]).
#[repr(C)]
pub struct XlaRets {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub size: i64,
    pub types: *mut i32,
    pub rets: *mut *mut c_void,
}

/// `XLA_FFI_Attrs`.
#[repr(C)]
pub struct XlaAttrs {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub size: i64,
    pub types: *mut i32,
    pub names: *mut *mut XlaByteSpan,
    pub attrs: *mut *mut c_void,
}

/// `XLA_FFI_Scalar`.
#[repr(C)]
pub struct XlaScalar {
    pub dtype: i32,
    pub value: *mut c_void,
}

/// `XLA_FFI_Array`.
#[repr(C)]
pub struct XlaArray {
    pub dtype: i32,
    pub size: usize,
    pub data: *mut c_void,
}

/// `XLA_FFI_CallFrame`.
#[repr(C)]
pub struct XlaCallFrame {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub api: *const XlaApi,
    pub ctx: *mut XlaExecutionContext,
    pub stage: i32,
    pub args: XlaArgs,
    pub rets: XlaRets,
    pub attrs: XlaAttrs,
    /// Out: an asynchronous handler's completion (`XLA_FFI_Future`); cute-rs's handler is
    /// synchronous and leaves it null.
    pub future: *mut c_void,
}

/// `XLA_FFI_Handler`.
pub type XlaHandler = unsafe extern "C" fn(*mut XlaCallFrame) -> *mut XlaError;

/// `XLA_FFI_Handler_Bundle`: the stage handlers (null stages are skipped by XLA).
#[repr(C)]
pub struct XlaHandlerBundle {
    pub instantiate: Option<XlaHandler>,
    pub prepare: Option<XlaHandler>,
    pub initialize: Option<XlaHandler>,
    pub execute: Option<XlaHandler>,
    /// The command-buffer recording stage (XLA FFI API 0.5).
    pub record: Option<XlaHandler>,
}

// ---------------------------------------------------------------------------------------------
// Names and the dtype and layout converters (cutlass.jax's `jax_to_cutlass_dtype`,
// `jax_to_cutlass_layout_order` and the FFI name helpers).

/// The XLA FFI target `jax.ffi.ffi_call` names for cute-rs' handler (`allow_cuda_graph`
/// picks the command-buffer-compatible one): `cute_rs_cutlass_call[_nograph]_v<version>`.
pub fn get_cutlass_call_ffi_name(allow_cuda_graph: bool, version: u32) -> String {
    let suffix = if allow_cuda_graph { "" } else { "_nograph" };
    format!("cute_rs_cutlass_call{suffix}_v{version}")
}

/// The FFI version for a JAX release `(major, minor, patch)`: 2 from 0.9.1, else 1.
pub fn get_cutlass_call_ffi_version(jax_version: (u32, u32, u32)) -> u32 {
    if jax_version >= (0, 9, 1) { 2 } else { 1 }
}

const DTYPE_TABLE: [(i32, &str, DType); 20] = [
    (data_type::PRED, "bool", DType::Bool),
    (data_type::S4, "int4", DType::I4),
    (data_type::S8, "int8", DType::I8),
    (data_type::S16, "int16", DType::I16),
    (data_type::S32, "int32", DType::I32),
    (data_type::S64, "int64", DType::I64),
    (data_type::U8, "uint8", DType::U8),
    (data_type::U16, "uint16", DType::U16),
    (data_type::U32, "uint32", DType::U32),
    (data_type::U64, "uint64", DType::U64),
    (data_type::F16, "float16", DType::F16),
    (data_type::BF16, "bfloat16", DType::BF16),
    (data_type::F32, "float32", DType::F32),
    (data_type::F64, "float64", DType::F64),
    (data_type::F8E5M2, "float8_e5m2", DType::F8E5M2),
    (data_type::F8E4M3FN, "float8_e4m3fn", DType::F8E4M3FN),
    (data_type::F8E4M3B11FNUZ, "float8_e4m3b11fnuz", DType::F8E4M3B11FNUZ),
    (data_type::F8E4M3, "float8_e4m3", DType::F8E4M3),
    (data_type::F4E2M1FN, "float4_e2m1fn", DType::F4E2M1FN),
    (data_type::F8E8M0FNU, "float8_e8m0fnu", DType::F8E8M0FNU),
];

/// The cute-rs type of an `XLA_FFI_DataType` code, if it has one.
pub fn dtype_from_xla_code(code: i32) -> Option<DType> {
    DTYPE_TABLE.iter().find(|t| t.0 == code).map(|t| t.2)
}

/// The `XLA_FFI_DataType` code of a cute-rs type, if it has one (TF32 is Float32 in XLA).
pub fn xla_code_of_dtype(dtype: DType) -> Option<i32> {
    let dtype = if dtype == DType::TF32 { DType::F32 } else { dtype };
    DTYPE_TABLE.iter().find(|t| t.2 == dtype).map(|t| t.0)
}

/// The cute-rs type of a JAX dtype name (`"bfloat16"`).
pub fn jax_to_cutlass_dtype(name: &str) -> Option<DType> {
    DTYPE_TABLE.iter().find(|t| t.1 == name).map(|t| t.2)
}

/// JAX's name for a cute-rs type (TF32 is `"float32"`).
pub fn cutlass_to_jax_dtype(dtype: DType) -> Option<&'static str> {
    let dtype = if dtype == DType::TF32 { DType::F32 } else { dtype };
    DTYPE_TABLE.iter().find(|t| t.2 == dtype).map(|t| t.1)
}

/// XLA's row-major layout of a rank-`rank` array in minor-to-major order (`[rank-1, ..., 0]`).
pub fn row_major_minor_to_major(rank: usize) -> Vec<i64> {
    (0..rank as i64).rev().collect()
}

fn permutation_inverse(p: &[i64]) -> Result<Vec<i64>, String> {
    let n = p.len();
    let mut out = vec![-1i64; n];
    for (i, &v) in p.iter().enumerate() {
        match usize::try_from(v) {
            Ok(v) if v < n && out[v] < 0 => out[v] = i as i64,
            _ => return Err(format!("{p:?} is not a permutation of 0..{n}")),
        }
    }
    Ok(out)
}

/// The cute stride order of an XLA layout: `order[d]` is the rank of dimension `d` among the
/// strides, smallest first (XLA's `minor_to_major` lists dimensions from the fastest, so the
/// order is its inverse permutation: row-major `[1, 0]` is order `[1, 0]` too, and
/// column-major `[0, 1]` is `[0, 1]`; `[1, 2, 0]` is `[2, 0, 1]`).
pub fn jax_to_cutlass_layout_order(minor_to_major: &[i64]) -> Result<Vec<i64>, String> {
    permutation_inverse(minor_to_major)
}

/// The XLA `minor_to_major` of a cute stride order (the inverse of
/// [`jax_to_cutlass_layout_order`]).
pub fn cutlass_to_jax_layout_order(order: &[i64]) -> Result<Vec<i64>, String> {
    permutation_inverse(order)
}

/// The element strides of a dense array of `dims` laid out as `minor_to_major` says.
pub fn strides_of_layout(dims: &[i64], minor_to_major: &[i64]) -> Result<Vec<i64>, String> {
    if dims.len() != minor_to_major.len() {
        return Err(format!("a layout of rank {} for an array of rank {}", minor_to_major.len(), dims.len()));
    }
    permutation_inverse(minor_to_major)?;
    let mut strides = vec![0i64; dims.len()];
    let mut acc = 1i64;
    for &d in minor_to_major {
        strides[d as usize] = acc;
        acc = acc.saturating_mul(dims[d as usize].max(1));
    }
    Ok(strides)
}

// ---------------------------------------------------------------------------------------------
// Specs, targets and the registry.

/// A tensor of a call: element type, logical dims and XLA layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub dtype: DType,
    pub dims: Vec<i64>,
    /// XLA's layout, fastest dimension first.
    pub minor_to_major: Vec<i64>,
}

impl TensorSpec {
    /// A row-major tensor.
    pub fn new(dtype: DType, dims: &[i64]) -> TensorSpec {
        TensorSpec { dtype, dims: dims.to_vec(), minor_to_major: row_major_minor_to_major(dims.len()) }
    }

    /// The cute stride order of its layout.
    pub fn stride_order(&self) -> Vec<i64> {
        jax_to_cutlass_layout_order(&self.minor_to_major).unwrap_or_default()
    }

    /// Its element strides.
    pub fn strides(&self) -> Vec<i64> {
        strides_of_layout(&self.dims, &self.minor_to_major).unwrap_or_default()
    }
}

/// An attribute of a call.
#[derive(Clone, Debug, PartialEq)]
pub enum XlaAttr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

/// What the builder is given: the buffers and attributes of a call (attributes by name).
#[derive(Clone, Debug, PartialEq)]
pub struct XlaCallSpec {
    pub target: String,
    pub args: Vec<TensorSpec>,
    pub rets: Vec<TensorSpec>,
    pub attrs: Vec<(String, XlaAttr)>,
}

impl XlaCallSpec {
    /// The cache key: the target and every spec and attribute.
    pub fn key(&self) -> String {
        format!("{}|{:?}|{:?}|{:?}", self.target, self.args, self.rets, self.attrs)
    }

    /// The attribute `name`, if present.
    pub fn attr(&self, name: &str) -> Option<&XlaAttr> {
        self.attrs.iter().find(|a| a.0 == name).map(|a| &a.1)
    }
}

/// What a builder makes: the callable for one call spec and the results that must alias
/// arguments (`input_output_aliases`).
pub struct XlaTarget {
    callable: Box<dyn TvmFfiCallable>,
    aliases: Vec<(usize, usize)>,
}

impl XlaTarget {
    /// A callable taking the arguments, then the results, then the scalar attributes as its
    /// arguments (a [`TvmFfiFunction`](crate::tvm_ffi::TvmFfiFunction) through
    /// [`IntoTvmFfiCallable`](crate::tvm_ffi::IntoTvmFfiCallable), or any
    /// [`TvmFfiCallable`]).
    pub fn new(callable: Box<dyn TvmFfiCallable>) -> XlaTarget {
        XlaTarget { callable, aliases: Vec::new() }
    }

    /// Require result `ret` to be the memory of argument `arg` (as XLA does for
    /// `input_output_aliases={arg: ret}`).
    pub fn alias(mut self, arg: usize, ret: usize) -> XlaTarget {
        self.aliases.push((arg, ret));
        self
    }
}

/// A builder: the call's specs to a target, or why it cannot be built.
pub type XlaBuilder = dyn Fn(&XlaCallSpec) -> Result<XlaTarget, String> + Send + Sync;

type Registry = HashMap<String, (Arc<XlaBuilder>, u64)>;

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
static GENERATION: AtomicU64 = AtomicU64::new(0);
static VERSIONS: AtomicU64 = AtomicU64::new(0);
static STATS: OnceLock<Mutex<HashMap<String, CallCacheStats>>> = OnceLock::new();

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

fn stats() -> std::sync::MutexGuard<'static, HashMap<String, CallCacheStats>> {
    STATS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Register (or replace) the builder of the target `name` (`cutlass_call`'s compiled kernel
/// per shape and spec). Replacing a builder drops that target's cached kernels.
pub fn register_cutlass_call(
    name: &str,
    builder: impl Fn(&XlaCallSpec) -> Result<XlaTarget, String> + Send + Sync + 'static,
) {
    // A new builder gets a new version, so only this target's cached kernels go stale.
    let version = VERSIONS.fetch_add(1, Ordering::SeqCst) + 1;
    registry().insert(name.to_string(), (Arc::new(builder), version));
}

/// Remove a target; returns whether it was registered.
pub fn unregister_cutlass_call(name: &str) -> bool {
    registry().remove(name).is_some()
}

/// Whether a target is registered.
pub fn is_cutlass_call_registered(name: &str) -> bool {
    registry().contains_key(name)
}

/// Drop every cached kernel (each thread's cache is emptied at its next call).
pub fn release_compile_cache() {
    GENERATION.fetch_add(1, Ordering::SeqCst);
}

/// Cache counters of a target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallCacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// The cache counters of target `name` (summed over threads; misses include failed builds).
pub fn call_cache_stats(name: &str) -> CallCacheStats {
    stats().get(name).copied().unwrap_or_default()
}

struct Cache {
    generation: u64,
    entries: HashMap<String, Rc<XlaTarget>>,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache { generation: 0, entries: HashMap::new() });
}

// ---------------------------------------------------------------------------------------------
// Reading a call frame.

/// A failed call: an XLA error code and the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XlaFailure {
    pub code: i32,
    pub message: String,
}

fn fail<T>(code: i32, message: impl Into<String>) -> Result<T, XlaFailure> {
    Err(XlaFailure { code, message: message.into() })
}

struct RawBuffer {
    dtype: i32,
    data: *mut c_void,
    dims: Vec<i64>,
}

/// # Safety
///
/// `p` is null or a valid `XLA_FFI_Buffer`.
unsafe fn read_buffer(p: *const c_void, what: &str) -> Result<RawBuffer, XlaFailure> {
    if p.is_null() {
        return fail(error_code::INVALID_ARGUMENT, format!("{what} is a null buffer"));
    }
    // SAFETY: the caller's contract.
    let b = unsafe { &*(p as *const XlaBuffer) };
    if b.rank < 0 || (b.rank > 0 && b.dims.is_null()) {
        return fail(error_code::INVALID_ARGUMENT, format!("{what} has rank {} and no dims", b.rank));
    }
    let dims = if b.rank == 0 {
        Vec::new()
    } else {
        // SAFETY: `rank` dims, per the buffer's contract.
        unsafe { std::slice::from_raw_parts(b.dims, b.rank as usize) }.to_vec()
    };
    Ok(RawBuffer { dtype: b.dtype, data: b.data, dims })
}

/// # Safety
///
/// `span` is null or a valid byte span.
unsafe fn span_str(span: *const XlaByteSpan) -> String {
    if span.is_null() {
        return String::new();
    }
    // SAFETY: the caller's contract.
    let s = unsafe { &*span };
    if s.ptr.is_null() {
        return String::new();
    }
    // SAFETY: `len` bytes at `ptr`.
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(s.ptr as *const u8, s.len) }).into_owned()
}

/// Read element `i` of a scalar of XLA dtype `dtype` at `p` as an attribute.
///
/// # Safety
///
/// `p` points at a value of that type.
unsafe fn read_scalar(dtype: i32, p: *const c_void, i: usize) -> Option<XlaAttr> {
    // SAFETY: the caller's contract; unaligned reads, the pointer's alignment being XLA's.
    unsafe {
        Some(match dtype {
            data_type::PRED => XlaAttr::Bool((p as *const u8).add(i).read_unaligned() != 0),
            data_type::S8 => XlaAttr::Int((p as *const i8).add(i).read_unaligned() as i64),
            data_type::S16 => XlaAttr::Int((p as *const i16).add(i).read_unaligned() as i64),
            data_type::S32 => XlaAttr::Int((p as *const i32).add(i).read_unaligned() as i64),
            data_type::S64 => XlaAttr::Int((p as *const i64).add(i).read_unaligned()),
            data_type::U8 => XlaAttr::Int((p as *const u8).add(i).read_unaligned() as i64),
            data_type::U16 => XlaAttr::Int((p as *const u16).add(i).read_unaligned() as i64),
            data_type::U32 => XlaAttr::Int((p as *const u32).add(i).read_unaligned() as i64),
            data_type::U64 => XlaAttr::Int((p as *const u64).add(i).read_unaligned() as i64),
            data_type::F32 => XlaAttr::Float((p as *const f32).add(i).read_unaligned() as f64),
            data_type::F64 => XlaAttr::Float((p as *const f64).add(i).read_unaligned()),
            _ => return None,
        })
    }
}

/// # Safety
///
/// `frame` is a valid call frame.
unsafe fn read_attrs(frame: &XlaCallFrame) -> Result<Vec<(String, XlaAttr)>, XlaFailure> {
    let n = frame.attrs.size.max(0) as usize;
    let mut out = Vec::with_capacity(n);
    if n > 0 && (frame.attrs.types.is_null() || frame.attrs.names.is_null() || frame.attrs.attrs.is_null()) {
        return fail(error_code::INVALID_ARGUMENT, "the call frame's attributes have null arrays");
    }
    for i in 0..n {
        // SAFETY: `n` entries in each array.
        let (ty, name, value) =
            unsafe { (*frame.attrs.types.add(i), span_str(*frame.attrs.names.add(i)), *frame.attrs.attrs.add(i)) };
        let attr = match ty {
            attr_type::STRING => {
                // SAFETY: a string attribute is a byte span.
                XlaAttr::Str(unsafe { span_str(value as *const XlaByteSpan) })
            }
            attr_type::SCALAR if !value.is_null() => {
                // SAFETY: a scalar attribute is an XLA_FFI_Scalar.
                let s = unsafe { &*(value as *const XlaScalar) };
                // SAFETY: `value` points at one value of `dtype`.
                match unsafe { read_scalar(s.dtype, s.value, 0) } {
                    Some(a) => a,
                    None => continue,
                }
            }
            attr_type::ARRAY if !value.is_null() => {
                // SAFETY: an array attribute is an XLA_FFI_Array.
                let a = unsafe { &*(value as *const XlaArray) };
                let items: Vec<XlaAttr> = (0..a.size)
                    // SAFETY: `size` values of `dtype` at `data`.
                    .map_while(|k| unsafe { read_scalar(a.dtype, a.data, k) })
                    .collect();
                if items.len() != a.size {
                    continue;
                }
                match items.first() {
                    Some(XlaAttr::Float(_)) => XlaAttr::FloatArray(
                        items.iter().map(|x| if let XlaAttr::Float(f) = x { *f } else { 0.0 }).collect(),
                    ),
                    _ => XlaAttr::IntArray(
                        items
                            .iter()
                            .map(|x| match x {
                                XlaAttr::Int(v) => *v,
                                XlaAttr::Bool(b) => *b as i64,
                                _ => 0,
                            })
                            .collect(),
                    ),
                }
            }
            // Dictionaries and unknown kinds are not read.
            _ => continue,
        };
        out.push((name, attr));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn is_reserved(name: &str) -> bool {
    name == "cute_target" || name == "device" || name.starts_with("arg_m2m_") || name.starts_with("ret_m2m_")
}

fn tensor_spec(what: &str, b: &RawBuffer, attrs: &[(String, XlaAttr)], key: &str) -> Result<TensorSpec, XlaFailure> {
    let Some(dtype) = dtype_from_xla_code(b.dtype) else {
        return fail(
            error_code::INVALID_ARGUMENT,
            format!("{what} has XLA element type {}, which has no cute-rs type", b.dtype),
        );
    };
    let minor_to_major = match attrs.iter().find(|a| a.0 == key) {
        Some((_, XlaAttr::IntArray(v))) => v.clone(),
        Some(_) => return fail(error_code::INVALID_ARGUMENT, format!("attribute `{key}` must be an integer array")),
        None => row_major_minor_to_major(b.dims.len()),
    };
    if let Err(e) = strides_of_layout(&b.dims, &minor_to_major) {
        return fail(error_code::INVALID_ARGUMENT, format!("{what}: layout {minor_to_major:?}: {e}"));
    }
    Ok(TensorSpec { dtype, dims: b.dims.clone(), minor_to_major })
}

/// A DLTensor over a buffer, with the shape and strides it points at.
struct OwnedDl {
    t: Box<DLTensor>,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

fn owned_dl(b: &RawBuffer, spec: &TensorSpec, device: i32) -> Result<OwnedDl, XlaFailure> {
    let Some(dt) = dl_data_type(spec.dtype) else {
        return fail(error_code::INVALID_ARGUMENT, format!("{} has no DLPack type", spec.dtype.cute_name()));
    };
    let mut shape = spec.dims.clone();
    let mut strides = spec.strides();
    let t = Box::new(DLTensor {
        data: b.data,
        device: DLDevice { device_type: device_type::KDL_CUDA, device_id: device },
        ndim: shape.len() as i32,
        dtype: dt,
        shape: if shape.is_empty() { std::ptr::null_mut() } else { shape.as_mut_ptr() },
        strides: if strides.is_empty() { std::ptr::null_mut() } else { strides.as_mut_ptr() },
        byte_offset: 0,
    });
    Ok(OwnedDl { t, _shape: shape, _strides: strides })
}

/// Run the execute stage of a call frame: build or find the kernel for it and launch it.
///
/// # Safety
///
/// `frame` is a valid XLA call frame as [`XlaCallFrame`] describes it, whose buffers are
/// device memory alive for the call.
#[cfg(feature = "cuda")]
unsafe fn execute(frame: &XlaCallFrame) -> Result<(), XlaFailure> {
    // SAFETY: the caller's contract, for every read below.
    let attrs = unsafe { read_attrs(frame) }?;
    let target = match attrs.iter().find(|a| a.0 == "cute_target") {
        Some((_, XlaAttr::Str(s))) => s.clone(),
        _ => {
            return fail(
                error_code::INVALID_ARGUMENT,
                "the call has no string attribute `cute_target` naming its kernel",
            );
        }
    };
    let (na, nr) = (frame.args.size.max(0) as usize, frame.rets.size.max(0) as usize);
    if (na > 0 && (frame.args.types.is_null() || frame.args.args.is_null()))
        || (nr > 0 && (frame.rets.types.is_null() || frame.rets.rets.is_null()))
    {
        return fail(error_code::INVALID_ARGUMENT, format!("{target}: the call frame's buffer arrays are null"));
    }
    let mut args = Vec::new();
    let mut rets = Vec::new();
    for i in 0..na {
        // SAFETY: `na` entries.
        let (ty, p) = unsafe { (*frame.args.types.add(i), *frame.args.args.add(i)) };
        if ty != BUFFER_TYPE {
            return fail(error_code::INVALID_ARGUMENT, format!("{target}: argument {i} is not a buffer (type {ty})"));
        }
        // SAFETY: a buffer argument.
        args.push(unsafe { read_buffer(p, &format!("{target}: argument {i}")) }?);
    }
    for i in 0..nr {
        // SAFETY: `nr` entries.
        let (ty, p) = unsafe { (*frame.rets.types.add(i), *frame.rets.rets.add(i)) };
        if ty != BUFFER_TYPE {
            return fail(error_code::INVALID_ARGUMENT, format!("{target}: result {i} is not a buffer (type {ty})"));
        }
        // SAFETY: a buffer result.
        rets.push(unsafe { read_buffer(p, &format!("{target}: result {i}")) }?);
    }
    let mut specs_a = Vec::new();
    let mut specs_r = Vec::new();
    for (i, b) in args.iter().enumerate() {
        specs_a.push(tensor_spec(&format!("{target}: argument {i}"), b, &attrs, &format!("arg_m2m_{i}"))?);
    }
    for (i, b) in rets.iter().enumerate() {
        specs_r.push(tensor_spec(&format!("{target}: result {i}"), b, &attrs, &format!("ret_m2m_{i}"))?);
    }
    let spec = XlaCallSpec { target: target.clone(), args: specs_a, rets: specs_r, attrs };

    // The cache, per thread, emptied when the registry changed.
    let generation = GENERATION.load(Ordering::SeqCst);
    let Some((builder, version)) = registry().get(&target).cloned() else {
        return fail(error_code::NOT_FOUND, format!("no cutlass_call target `{target}` is registered"));
    };
    let key = format!("{version}|{}", spec.key());
    let cached = CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.generation != generation {
            c.entries.clear();
            c.generation = generation;
        }
        c.entries.get(&key).cloned()
    });
    let entry = match cached {
        Some(e) => {
            stats().entry(target.clone()).or_default().hits += 1;
            e
        }
        None => {
            stats().entry(target.clone()).or_default().misses += 1;
            let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| builder(&spec)))
                .unwrap_or_else(|_| Err("the builder panicked".to_string()))
                .map_err(|e| XlaFailure {
                    code: error_code::INTERNAL,
                    message: format!("{target}: building the kernel failed: {e}"),
                })?;
            let e = Rc::new(built);
            CACHE.with(|c| c.borrow_mut().entries.insert(key, e.clone()));
            e
        }
    };

    // Declared aliases: the result is the argument's memory.
    for &(a, r) in &entry.aliases {
        let (Some(x), Some(y)) = (args.get(a), rets.get(r)) else {
            return fail(error_code::INTERNAL, format!("{target}: alias {a} -> {r} names a missing buffer"));
        };
        if spec.args[a].dtype != spec.rets[r].dtype || spec.args[a].dims != spec.rets[r].dims {
            return fail(
                error_code::INVALID_ARGUMENT,
                format!("{target}: result {r} aliases argument {a} but differs from it in dtype or shape"),
            );
        }
        if x.data != y.data {
            return fail(
                error_code::FAILED_PRECONDITION,
                format!("{target}: result {r} must alias argument {a}, but XLA gave it its own buffer"),
            );
        }
    }

    let device = match spec.attr("device") {
        Some(XlaAttr::Int(d)) => i32::try_from(*d).map_err(|_| XlaFailure {
            code: error_code::INVALID_ARGUMENT,
            message: format!("{target}: device {d} is out of range"),
        })?,
        _ => crate::cuda::current_device_ordinal()
            .map_err(|e| XlaFailure { code: error_code::INTERNAL, message: format!("{target}: {e}") })?,
    };
    let stream = {
        // SAFETY: the frame's API table.
        let api = unsafe { frame.api.as_ref() };
        let Some(get) = api.and_then(|a| a.stream_get) else {
            return fail(error_code::INTERNAL, format!("{target}: the XLA API table has no stream getter"));
        };
        let mut a = XlaStreamGetArgs {
            struct_size: std::mem::size_of::<XlaStreamGetArgs>(),
            extension_start: std::ptr::null_mut(),
            ctx: frame.ctx,
            stream: std::ptr::null_mut(),
        };
        // SAFETY: a plain query with valid arguments.
        let err = unsafe { get(&mut a) };
        if !err.is_null() {
            return fail(error_code::INTERNAL, format!("{target}: XLA_FFI_Stream_Get failed"));
        }
        a.stream
    };

    let mut tensors = Vec::new();
    for (b, s) in args.iter().zip(&spec.args).chain(rets.iter().zip(&spec.rets)) {
        tensors.push(owned_dl(b, s, device)?);
    }
    let mut anys: Vec<TVMFFIAny> = tensors.iter().map(|t| TVMFFIAny::dl_tensor(&*t.t)).collect();
    for (name, a) in &spec.attrs {
        if is_reserved(name) {
            continue;
        }
        match a {
            XlaAttr::Int(v) => anys.push(TVMFFIAny::int(*v)),
            XlaAttr::Float(v) => anys.push(TVMFFIAny::float(*v)),
            XlaAttr::Bool(v) => anys.push(TVMFFIAny::bool(*v)),
            _ => {}
        }
    }
    // SAFETY: the stream is XLA's for this device, live for the call.
    unsafe { entry.callable.call_any_on_stream(&anys, stream) }.map_err(|e: FfiError| XlaFailure {
        code: if e.kind == "RuntimeError" { error_code::INTERNAL } else { error_code::INVALID_ARGUMENT },
        message: format!("{target}: {}", e.message),
    })
}

#[cfg(not(feature = "cuda"))]
unsafe fn execute(_: &XlaCallFrame) -> Result<(), XlaFailure> {
    fail(error_code::UNIMPLEMENTED, "cute-rs was built without the `cuda` feature, so it cannot launch kernels")
}

/// Make an `XLA_FFI_Error` (null if there is no way to: the API table lacks `Error_Create`).
///
/// # Safety
///
/// `api` is null or a valid API table.
unsafe fn make_error(api: *const XlaApi, f: &XlaFailure) -> *mut XlaError {
    // SAFETY: the caller's contract.
    let Some(create) = (unsafe { api.as_ref() }).and_then(|a| a.error_create) else {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "cute: {} (and it cannot be reported to XLA)", f.message);
        return std::ptr::null_mut();
    };
    let message = CString::new(f.message.replace('\0', " ")).unwrap_or_default();
    let mut args = XlaErrorCreateArgs {
        struct_size: std::mem::size_of::<XlaErrorCreateArgs>(),
        extension_start: std::ptr::null_mut(),
        message: message.as_ptr(),
        errc: f.code,
    };
    // SAFETY: valid arguments; XLA copies the message.
    unsafe { create(&mut args) }
}

/// The handler body behind the generated symbols: in the execute stage run the call; the
/// other stages succeed without work. `allow_cuda_graph` is recorded by which symbol XLA
/// calls, not used here (the launch is capturable: nothing synchronises or allocates).
///
/// # Safety
///
/// `frame` is null or a valid [`XlaCallFrame`] whose buffers are device memory alive for the
/// call.
pub unsafe fn handle_call_frame(frame: *mut XlaCallFrame) -> *mut XlaError {
    // SAFETY: the caller's contract.
    let Some(frame) = (unsafe { frame.as_ref() }) else {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "cute: an XLA custom call got a null call frame");
        return std::ptr::null_mut();
    };
    if frame.stage != stage::EXECUTE {
        return std::ptr::null_mut();
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        unsafe { execute(frame) }
    }))
    .unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let mut m = String::from("the cutlass_call handler panicked");
        if !msg.is_empty() {
            let _ = write!(m, ": {msg}");
        }
        fail(error_code::INTERNAL, m)
    });
    match outcome {
        Ok(()) => std::ptr::null_mut(),
        // SAFETY: the frame's API table.
        Err(f) => unsafe { make_error(frame.api, &f) },
    }
}

/// A handler that does nothing, for the stages of the bundle that have no work.
///
/// # Safety
///
/// Never reads its argument.
pub unsafe extern "C" fn noop_handler(_: *mut XlaCallFrame) -> *mut XlaError {
    std::ptr::null_mut()
}

/// Export the XLA FFI handler symbols from a `cdylib`: `cute_xla_cutlass_call` (execute only,
/// the plain handler `jax.ffi.register_ffi_target` takes as a capsule),
/// `cute_xla_cutlass_call_nograph` (the same, for registering without command-buffer
/// compatibility) and the static `cute_xla_cutlass_call_bundle` (an `XLA_FFI_Handler_Bundle`
/// whose stages other than execute do nothing). Register targets with
/// [`register_cutlass_call`] from a function the library exports for Python to call
/// before the first use.
#[macro_export]
macro_rules! xla_ffi_export {
    () => {
        /// The XLA FFI handler (see `cute::xla_ffi`).
        ///
        /// # Safety
        ///
        /// Called by XLA with a valid call frame.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn cute_xla_cutlass_call(
            frame: *mut $crate::xla_ffi::XlaCallFrame,
        ) -> *mut $crate::xla_ffi::XlaError {
            // SAFETY: this function's contract.
            unsafe { $crate::xla_ffi::handle_call_frame(frame) }
        }

        /// The same handler, for registering without command-buffer compatibility.
        ///
        /// # Safety
        ///
        /// Called by XLA with a valid call frame.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn cute_xla_cutlass_call_nograph(
            frame: *mut $crate::xla_ffi::XlaCallFrame,
        ) -> *mut $crate::xla_ffi::XlaError {
            // SAFETY: this function's contract.
            unsafe { $crate::xla_ffi::handle_call_frame(frame) }
        }

        /// The stage handlers as one bundle.
        #[unsafe(no_mangle)]
        pub static cute_xla_cutlass_call_bundle: $crate::xla_ffi::XlaHandlerBundle =
            $crate::xla_ffi::XlaHandlerBundle {
                instantiate: Some($crate::xla_ffi::noop_handler),
                prepare: Some($crate::xla_ffi::noop_handler),
                initialize: Some($crate::xla_ffi::noop_handler),
                execute: Some(cute_xla_cutlass_call),
                record: None,
            };
    };
}

// `XlaHandlerBundle` holds only function pointers.
// SAFETY: immutable function pointers.
unsafe impl Sync for XlaHandlerBundle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tables_round_trip() {
        for (code, name, dtype) in DTYPE_TABLE {
            assert_eq!(dtype_from_xla_code(code), Some(dtype));
            assert_eq!(xla_code_of_dtype(dtype), Some(code));
            assert_eq!(jax_to_cutlass_dtype(name), Some(dtype));
            assert_eq!(cutlass_to_jax_dtype(dtype), Some(name));
            assert!(dl_data_type(dtype).is_some(), "{name}");
        }
        assert_eq!(xla_code_of_dtype(DType::TF32), Some(data_type::F32));
        assert_eq!(cutlass_to_jax_dtype(DType::TF32), Some("float32"));
        assert_eq!(dtype_from_xla_code(data_type::INVALID), None);
        assert_eq!(jax_to_cutlass_dtype("complex64"), None);
    }

    #[test]
    fn layout_orders() {
        assert_eq!(row_major_minor_to_major(3), vec![2, 1, 0]);
        assert_eq!(jax_to_cutlass_layout_order(&[1, 2, 0]).unwrap(), vec![2, 0, 1]);
        assert_eq!(cutlass_to_jax_layout_order(&[2, 0, 1]).unwrap(), vec![1, 2, 0]);
        assert_eq!(jax_to_cutlass_layout_order(&[1, 0]).unwrap(), vec![1, 0]);
        assert!(jax_to_cutlass_layout_order(&[0, 0]).is_err());
        assert!(jax_to_cutlass_layout_order(&[0, 2]).is_err());
        assert!(jax_to_cutlass_layout_order(&[-1, 0]).is_err());
        for p in [[0i64, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]] {
            let o = jax_to_cutlass_layout_order(&p).unwrap();
            assert_eq!(cutlass_to_jax_layout_order(&o).unwrap(), p);
        }
    }

    #[test]
    fn layout_strides() {
        assert_eq!(strides_of_layout(&[3, 4], &[1, 0]).unwrap(), vec![4, 1]);
        assert_eq!(strides_of_layout(&[3, 4], &[0, 1]).unwrap(), vec![1, 3]);
        assert_eq!(strides_of_layout(&[2, 3, 4], &[1, 2, 0]).unwrap(), vec![12, 1, 3]);
        assert!(strides_of_layout(&[2, 3], &[0]).is_err());
        assert_eq!(TensorSpec::new(DType::F32, &[5, 6]).strides(), vec![6, 1]);
    }

    #[test]
    fn ffi_names() {
        assert_eq!(get_cutlass_call_ffi_name(true, 2), "cute_rs_cutlass_call_v2");
        assert_eq!(get_cutlass_call_ffi_name(false, 1), "cute_rs_cutlass_call_nograph_v1");
        assert_eq!(get_cutlass_call_ffi_version((0, 9, 1)), 2);
        assert_eq!(get_cutlass_call_ffi_version((0, 9, 0)), 1);
        assert_eq!(get_cutlass_call_ffi_version((1, 0, 0)), 2);
        assert_eq!(get_cutlass_call_ffi_version((0, 6, 0)), 1);
    }
}

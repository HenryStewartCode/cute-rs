//! External device functions and standalone `__device__` functions (CuTe DSL's `cute.extern`,
//! `cute.ffi` and `cute.compile[cute.DeviceTarget]`).
//!
//! # Calling external device functions
//!
//! An [`ExternFn`] declares a device function by its typed signature: CuTe DSL's
//! `cute.ffi(name=, params_types=, return_type=, inline=, source=)`, or a `@cute.extern`
//! function. [`ExternFn::call`] inside a kernel adds the declaration to the kernel's module (a
//! `func.func private` prototype; one declaration per symbol, and a second one with another
//! signature is an error) and a call of it (`func.call`). The function's code comes from LLVM
//! bitcode ([`ExternFn::source`], CuTe DSL's `BitCode(path)`, or
//! [`CompileOptions::link_libraries`](crate::CompileOptions::link_libraries), its
//! `cute.LinkLibraries`), linked into the module before it is optimized, or from another PTX
//! module linked at load time ([`CudaContext::load_linked`](crate::cuda::CudaContext::load_linked)).
//!
//! Pointer parameters ([`ExternFn::param_ptr`]) take a [`Ptr`] (or a tensor's pointer): its
//! address becomes an `!llvm.ptr` in the parameter's address space, a shared-memory pointer
//! passed to a generic parameter going through `addrspacecast`, as CuTe DSL converts a
//! `!cute.ptr` to the `!llvm.ptr` a prototype wants.
//!
//! Decisions:
//!
//! * calls are inlined by default (CuTe DSL's `inline=True`); [`ExternFn::inline`]`(false)`
//!   keeps them calls (`func.call ... {no_inline}`, a PTX `call`);
//! * every prototype is `convergent` by default ([`ExternFn::convergent`] opts out): the callee
//!   may hold barriers, shuffles or other warp collectives, so LLVM must neither duplicate a
//!   call nor sink it into divergent code;
//! * there is no recursion and no callback into the calling kernel.
//!
//! # Overloads and generics
//!
//! An overloaded function's symbol is mangled from its concrete parameter types, as CuTe DSL's
//! `default_name_mangler` does ([`mangle_symbol`]): the name, then `_` and the parts joined by
//! `_`, one part per parameter, the MLIR spelling of its type (`f32`, `i32`, `!llvm.ptr<1>`)
//! with every character outside `[A-Za-z0-9_]` replaced by `_XX` (its code in two upper-case
//! hexadecimal digits, [`mangle`]). A compile-time constant parameter ([`ExternFn::param_const`],
//! CuTe DSL's `Constexpr`, integers only) is not passed at run time: it contributes its type
//! and its value. [`ExternOverloads`] picks the first variant whose parameters match the
//! arguments (CuTe DSL's `@overload`), and a generic parameter ([`ExternFn::param_generic`],
//! CuTe DSL's `TypeVar`) takes the type of its argument.
//!
//! # Simulation
//!
//! The simulator runs a call by one of:
//!
//! * a DSL body ([`ExternFn::sim_body`]): code traced in the calling kernel at the call, which
//!   reads and writes memory through pointer arguments like any kernel code (a
//!   [`DeviceFunction`]'s calls use the device function's own body);
//! * raw bits ([`ExternFn::sim`]): a Rust closure from the value arguments' bits (0 for a
//!   pointer) to the result's, like [`InlinePtx::sim`](crate::InlinePtx::sim).
//!
//! Without either, the kernel can be compiled but simulating it fails at the call ("extern
//! function `f` has no simulator implementation").
//!
//! # Standalone device functions
//!
//! A [`DeviceFunction`] is CuTe DSL's `cute.compile[cute.DeviceTarget](fn, ...)`: a function
//! with real parameters and a return value instead of a kernel, compiled as the PTX of an
//! `extern "C" __device__` function (`.visible .func (.param .b32 func_retval0) axpy(...)`).
//! It returns at most one value (what a C declaration can express; pass out-pointers for
//! more), allocates no shared memory and assumes no block size. Barriers are allowed, but then
//! every caller must call it convergently. [`DeviceFunction::compile_relocatable`] assembles it
//! as relocatable device code (`ptxas -c`, CuTe DSL's `cute.RDC`), to link into CUDA C++.

use std::path::PathBuf;
use std::sync::Arc;

use crate::backend::{Arch, CompileError, CompileOptions, Toolchain};
use crate::builder::{self, Kernel, RegionKind};
use crate::ir::{AsmInput, AsmSim, DType, DeviceFunctionSig, ExternDecl, ExternId, ExternParam, Literal, MemSpace, Op};
use crate::layout::IndexExpr;
pub use crate::tensor::AddressSpace;
use crate::tensor::Ptr;
use crate::value::{AnyValue, Numeric, Value};

/// CuTe DSL's `mangle`: `name` with every character outside `[A-Za-z0-9_]` replaced by `_XX`,
/// its code point in (at least) two upper-case hexadecimal digits.
///
/// ```
/// assert_eq!(cute::ffi::mangle("!llvm.ptr<1>"), "_21llvm_2Eptr_3C1_3E");
/// ```
pub fn mangle(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c.to_string() } else { format!("_{:02X}", c as u32) })
        .collect()
}

/// One parameter's contribution to a mangled symbol ([`mangle_symbol`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MangledParam {
    /// A run-time parameter of this MLIR type (`f32`, `!llvm.ptr<1>`).
    Type(String),
    /// A compile-time constant: its MLIR type and value.
    Const(String, i64),
}

/// CuTe DSL's `default_name_mangler`: `name`, `_`, then the parameters' parts joined by `_`:
/// each run-time parameter's [`mangle`]d type, and for a constant its mangled type and then its
/// mangled value.
///
/// ```
/// use cute::ffi::{MangledParam, mangle_symbol};
/// let parts = [MangledParam::Type("f32".into()), MangledParam::Const("i32".into(), -3)];
/// assert_eq!(mangle_symbol("scale", &parts), "scale_f32_i32__2D3");
/// ```
pub fn mangle_symbol(name: &str, params: &[MangledParam]) -> String {
    let mut parts = Vec::new();
    for p in params {
        match p {
            MangledParam::Type(t) => parts.push(mangle(t)),
            MangledParam::Const(t, v) => {
                parts.push(mangle(t));
                parts.push(mangle(&v.to_string()));
            }
        }
    }
    format!("{name}_{}", parts.join("_"))
}

/// CuTe DSL's `BitCode(path)`: an LLVM bitcode file an external function's code comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitCode(pub PathBuf);

impl BitCode {
    pub fn new(path: impl Into<PathBuf>) -> BitCode {
        BitCode(path.into())
    }
}

/// A pointer passed to an external function, without its element type in the type system.
#[derive(Clone, Debug)]
pub struct AnyPtr {
    buffer: crate::ir::BufferId,
    space: MemSpace,
    offset: IndexExpr,
    elem: DType,
}

impl AnyPtr {
    /// The element type of the pointer passed.
    pub fn elem(&self) -> DType {
        self.elem
    }

    /// As a pointer to `T`, which must be the element type it was passed as (use
    /// [`Ptr::cast`] for another).
    #[track_caller]
    pub fn to_ptr<T: Numeric>(&self) -> Ptr<T> {
        assert_eq!(self.elem, T::DTYPE, "cute: the pointer argument points to {}, not {}", self.elem, T::DTYPE);
        Ptr {
            buffer: self.buffer,
            space: self.space,
            offset: self.offset.clone(),
            swizzle: None,
            // Ptr::parts refuses DSMEM pointers, so an extern call's pointer argument is local.
            peer: None,
            _t: Default::default(),
        }
    }

    /// Its address space as a pointer parameter sees it.
    pub fn address_space(&self) -> AddressSpace {
        match self.space {
            MemSpace::Global => AddressSpace::Gmem,
            MemSpace::Shared => AddressSpace::Smem,
            MemSpace::Raw(a) => a,
            _ => AddressSpace::Generic,
        }
    }
}

/// An argument of an external function call: a value, a pointer, or a compile-time integer
/// (for a [`param_const`](ExternFn::param_const) parameter).
#[derive(Clone, Debug)]
pub enum CallArg {
    Value(AnyValue),
    Ptr(AnyPtr),
    Const(i64),
}

impl<T: Numeric> From<Value<T>> for CallArg {
    #[track_caller]
    fn from(v: Value<T>) -> CallArg {
        CallArg::Value(v.into())
    }
}

impl From<AnyValue> for CallArg {
    fn from(v: AnyValue) -> CallArg {
        CallArg::Value(v)
    }
}

impl<T: Numeric> From<&Ptr<T>> for CallArg {
    #[track_caller]
    fn from(p: &Ptr<T>) -> CallArg {
        let (buffer, offset) = p.parts();
        CallArg::Ptr(AnyPtr { buffer, space: p.space, offset, elem: T::DTYPE })
    }
}

impl<T: Numeric> From<Ptr<T>> for CallArg {
    #[track_caller]
    fn from(p: Ptr<T>) -> CallArg {
        CallArg::from(&p)
    }
}

impl CallArg {
    /// The value, as a `T` (its type must be `T`).
    #[track_caller]
    pub fn value<T: Numeric>(&self) -> Value<T> {
        match self {
            CallArg::Value(v) => {
                let t = builder::value_dtype(v.0).expect("a value of the kernel being built");
                assert_eq!(t, T::DTYPE, "cute: the argument is {t}, not {}", T::DTYPE);
                Value::from_id(v.0)
            }
            other => panic!("cute: the argument {other:?} is not a value"),
        }
    }

    /// The pointer, as a pointer to `T` (its element type must be `T`).
    #[track_caller]
    pub fn ptr<T: Numeric>(&self) -> Ptr<T> {
        match self {
            CallArg::Ptr(p) => p.to_ptr(),
            other => panic!("cute: the argument {other:?} is not a pointer"),
        }
    }

    fn dtype(&self) -> Option<DType> {
        match self {
            CallArg::Value(v) => builder::value_dtype(v.0),
            _ => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            CallArg::Value(_) => format!("a {} value", self.dtype().map_or("?".to_string(), |t| t.to_string())),
            CallArg::Ptr(p) => format!("a pointer to {} ({:?})", p.elem, p.address_space()),
            CallArg::Const(c) => format!("the constant {c}"),
        }
    }
}

/// A parameter of an [`ExternFn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamType {
    /// A run-time value of this type.
    Value(DType),
    /// A pointer into this address space ([`AddressSpace::Generic`], `Gmem` or `Smem`).
    Ptr(AddressSpace),
    /// A compile-time integer of this type (CuTe DSL's `Constexpr`): part of the mangled symbol,
    /// not passed.
    Const(DType),
    /// A generic parameter (CuTe DSL's `TypeVar`): the type of its argument.
    Generic,
}

/// The result of an [`ExternFn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnType {
    None,
    Value(DType),
    /// The type of the run-time parameter at this index (a generic function's `-> T`).
    SameAs(usize),
}

/// The DSL code the simulator runs for a call ([`ExternFn::sim_body`]).
pub type SimBody = Arc<dyn Fn(&[CallArg]) -> Option<AnyValue> + Send + Sync>;

/// A custom mangler ([`ExternFn::name_mangler`]): the name and the parameters' parts to the
/// symbol.
pub type NameMangler = Arc<dyn Fn(&str, &[MangledParam]) -> String + Send + Sync>;

/// An external device function: CuTe DSL's `cute.ffi(...)`, or a `@cute.extern` declaration.
#[derive(Clone)]
pub struct ExternFn {
    name: String,
    params: Vec<ParamType>,
    ret: ReturnType,
    inline: bool,
    convergent: bool,
    source: Option<PathBuf>,
    overloaded: Option<bool>,
    mangler: Option<NameMangler>,
    sim: Option<AsmSim>,
    sim_body: Option<SimBody>,
}

impl std::fmt::Debug for ExternFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternFn")
            .field("name", &self.name)
            .field("params", &self.params)
            .field("ret", &self.ret)
            .field("inline", &self.inline)
            .field("convergent", &self.convergent)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// A valid symbol: an identifier of letters, digits, `_`, `$` and `.` not starting with a digit.
fn valid_symbol(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'))
}

impl ExternFn {
    /// A function named `name` with no parameters and no result, inlined, convergent.
    #[track_caller]
    pub fn new(name: impl Into<String>) -> ExternFn {
        let name = name.into();
        assert!(valid_symbol(&name), "cute: external function name {name:?} is not a valid symbol");
        ExternFn {
            name,
            params: Vec::new(),
            ret: ReturnType::None,
            inline: true,
            convergent: true,
            source: None,
            overloaded: None,
            mangler: None,
            sim: None,
            sim_body: None,
        }
    }

    /// The next parameter: a `T` value.
    pub fn param<T: Numeric>(self) -> ExternFn {
        self.param_type(ParamType::Value(T::DTYPE))
    }

    /// The next parameter: a pointer into `space` (generic, global or shared).
    #[track_caller]
    pub fn param_ptr(self, space: AddressSpace) -> ExternFn {
        assert!(
            matches!(space, AddressSpace::Generic | AddressSpace::Gmem | AddressSpace::Smem),
            "cute: an external function's pointer parameter is generic, global or shared, not {space:?}"
        );
        self.param_type(ParamType::Ptr(space))
    }

    /// The next parameter: a compile-time `T` integer (CuTe DSL's `Constexpr`). The function
    /// becomes overloaded (its symbol mangled) unless [`overloaded`](Self::overloaded) says
    /// otherwise.
    #[track_caller]
    pub fn param_const<T: Numeric>(self) -> ExternFn {
        assert!(T::DTYPE.is_int(), "cute: a constant parameter is an integer, not {}", T::DTYPE);
        self.param_type(ParamType::Const(T::DTYPE))
    }

    /// The next parameter: generic (CuTe DSL's `TypeVar`), the type of its argument. The
    /// function becomes overloaded unless [`overloaded`](Self::overloaded) says otherwise.
    pub fn param_generic(self) -> ExternFn {
        self.param_type(ParamType::Generic)
    }

    /// The next parameter, of any kind.
    pub fn param_type(mut self, p: ParamType) -> ExternFn {
        self.params.push(p);
        self
    }

    /// The result: a `T`.
    pub fn returns<T: Numeric>(mut self) -> ExternFn {
        self.ret = ReturnType::Value(T::DTYPE);
        self
    }

    /// The result: the type of run-time parameter `index` (for generic functions).
    pub fn returns_same_as(mut self, index: usize) -> ExternFn {
        self.ret = ReturnType::SameAs(index);
        self
    }

    /// Whether calls may be inlined (the default; CuTe DSL's `inline=`).
    pub fn inline(mut self, inline: bool) -> ExternFn {
        self.inline = inline;
        self
    }

    /// Whether the prototype is `convergent` (the default): turn it off only for a function
    /// with no barrier, shuffle or other collective, so that LLVM may move calls freely.
    pub fn convergent(mut self, convergent: bool) -> ExternFn {
        self.convergent = convergent;
        self
    }

    /// The LLVM bitcode file the function's code comes from (CuTe DSL's
    /// `source=BitCode(path)`), linked into every module that calls it. The file must exist
    /// when the kernel is compiled.
    pub fn source(mut self, bitcode: BitCode) -> ExternFn {
        self.source = Some(bitcode.0);
        self
    }

    /// Whether the symbol is mangled from the parameter types (CuTe DSL's `overloaded=`). By
    /// default it is when the function has a constant or generic parameter.
    pub fn overloaded(mut self, overloaded: bool) -> ExternFn {
        self.overloaded = Some(overloaded);
        self
    }

    /// A custom mangler (CuTe DSL's `name_mangler=`), used when the function is overloaded.
    pub fn name_mangler(mut self, f: impl Fn(&str, &[MangledParam]) -> String + Send + Sync + 'static) -> ExternFn {
        self.mangler = Some(Arc::new(f));
        self
    }

    /// What the simulator runs for a call: the value arguments' raw bits (zero-extended to
    /// 64; 0 for a pointer) to the result's (one entry, or none without a result).
    pub fn sim(mut self, f: impl Fn(&[u64]) -> Vec<u64> + Send + Sync + 'static) -> ExternFn {
        self.sim = Some(AsmSim::new(move |r, _| f(r)));
        self
    }

    /// What the simulator runs for a call, as DSL code traced in the calling kernel: given the
    /// arguments (constants included), it returns the result (`None` without one). It can read
    /// and write memory through pointer arguments.
    pub fn sim_body(mut self, f: impl Fn(&[CallArg]) -> Option<AnyValue> + Send + Sync + 'static) -> ExternFn {
        self.sim_body = Some(Arc::new(f));
        self
    }

    /// The function's name (unmangled).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The declared parameters.
    pub fn params(&self) -> &[ParamType] {
        &self.params
    }

    fn is_overloaded(&self) -> bool {
        self.overloaded
            .unwrap_or_else(|| self.params.iter().any(|p| matches!(p, ParamType::Const(_) | ParamType::Generic)))
    }

    /// Whether `args` fit the parameters (for [`ExternOverloads`]).
    fn accepts(&self, args: &[CallArg]) -> bool {
        args.len() == self.params.len()
            && self.params.iter().zip(args).all(|(p, a)| match (p, a) {
                (ParamType::Value(t), CallArg::Value(_)) => a.dtype() == Some(*t),
                (ParamType::Ptr(_), CallArg::Ptr(_)) => true,
                (ParamType::Const(_), CallArg::Const(_)) => true,
                (ParamType::Const(_), CallArg::Value(v)) => builder::try_constant(v.0).is_some(),
                (ParamType::Generic, CallArg::Value(_) | CallArg::Ptr(_)) => true,
                _ => false,
            })
    }

    /// The symbol, prototype parameters and result for these arguments, or why they do not fit.
    fn resolve(&self, args: &[CallArg]) -> Result<(String, Vec<ExternParam>, Vec<DType>), String> {
        if args.len() != self.params.len() {
            return Err(format!(
                "external function `{}` takes {} arguments, but the call passes {}",
                self.name,
                self.params.len(),
                args.len()
            ));
        }
        let mut concrete = Vec::new();
        let mut mangled = Vec::new();
        for (k, (p, a)) in self.params.iter().zip(args).enumerate() {
            let mismatch = |want: String| {
                format!("argument {k} of `{}` must be {want}, but the call passes {}", self.name, a.describe())
            };
            match (p, a) {
                (ParamType::Value(t), CallArg::Value(_)) if a.dtype() == Some(*t) => {
                    concrete.push(ExternParam::Value(*t));
                    mangled.push(MangledParam::Type(t.mlir().to_string()));
                }
                (ParamType::Value(t), _) => return Err(mismatch(format!("a {t} value"))),
                (ParamType::Ptr(space), CallArg::Ptr(_)) => {
                    let e = ExternParam::Ptr(*space);
                    mangled.push(MangledParam::Type(e.mlir()));
                    concrete.push(e);
                }
                (ParamType::Ptr(space), _) => return Err(mismatch(format!("a pointer ({space:?})"))),
                (ParamType::Const(t), CallArg::Const(c)) => mangled.push(MangledParam::Const(t.mlir().to_string(), *c)),
                (ParamType::Const(t), CallArg::Value(v)) => match builder::try_constant(v.0) {
                    Some(lit) if lit.dtype().is_int() => {
                        mangled.push(MangledParam::Const(t.mlir().to_string(), literal_int(lit)))
                    }
                    _ => return Err(mismatch(format!("a compile-time {t} constant"))),
                },
                (ParamType::Const(t), _) => return Err(mismatch(format!("a compile-time {t} constant"))),
                (ParamType::Generic, CallArg::Value(_)) => {
                    let t = a.dtype().expect("a value of the kernel being built");
                    concrete.push(ExternParam::Value(t));
                    mangled.push(MangledParam::Type(t.mlir().to_string()));
                }
                (ParamType::Generic, CallArg::Ptr(ptr)) => {
                    let space = match ptr.address_space() {
                        AddressSpace::Gmem | AddressSpace::Smem => ptr.address_space(),
                        _ => AddressSpace::Generic,
                    };
                    let e = ExternParam::Ptr(space);
                    mangled.push(MangledParam::Type(e.mlir()));
                    concrete.push(e);
                }
                (ParamType::Generic, CallArg::Const(_)) => {
                    return Err(mismatch("a run-time value or pointer (a generic parameter)".to_string()));
                }
            }
        }
        let results = match self.ret {
            ReturnType::None => vec![],
            ReturnType::Value(t) => vec![t],
            ReturnType::SameAs(i) => match concrete.get(i) {
                Some(ExternParam::Value(t)) => vec![*t],
                _ => {
                    return Err(format!(
                        "`{}` returns the type of run-time parameter {i}, which is not a value",
                        self.name
                    ));
                }
            },
        };
        let symbol = if self.is_overloaded() {
            match &self.mangler {
                Some(m) => m(&self.name, &mangled),
                None => mangle_symbol(&self.name, &mangled),
            }
        } else {
            self.name.clone()
        };
        if !valid_symbol(&symbol) {
            return Err(format!("the symbol {symbol:?} of `{}` is not a valid symbol", self.name));
        }
        Ok((symbol, concrete, results))
    }

    /// Call the function with `args` in the kernel being built, returning its result. Panics,
    /// naming the caller's line, if the arguments do not fit the parameters or another
    /// declaration of the symbol has another signature.
    #[track_caller]
    pub fn call(&self, args: &[CallArg]) -> Option<AnyValue> {
        let loc = std::panic::Location::caller();
        let (symbol, params, results) =
            self.resolve(args).unwrap_or_else(|m| builder::user_panic(loc, format!("cute: {m}")));
        let decl = ExternDecl {
            symbol,
            params,
            results: results.clone(),
            inline: self.inline,
            convergent: self.convergent,
            sim: self.sim.clone(),
            loc,
        };
        let callee = declare(decl, self.source.clone(), loc);
        // The run-time arguments: values, and pointers' addresses.
        let mut operands = Vec::new();
        for (p, a) in self.params.iter().zip(args) {
            match (p, a) {
                (ParamType::Const(_), _) => {}
                (_, CallArg::Value(v)) => operands.push(AsmInput::Value(v.0)),
                (_, CallArg::Ptr(ptr)) => {
                    let index = crate::tensor::index_operand(ptr.offset.clone());
                    operands.push(AsmInput::Address { buffer: ptr.buffer, index, elem: ptr.elem });
                }
                (_, CallArg::Const(_)) => unreachable!("resolved: constants are for constant parameters"),
            }
        }
        let body = self.sim_body.as_ref().map(|f| trace_body(f, args, &results, &self.name, loc));
        let result = builder::with(|b| {
            let out: Vec<_> = results.iter().map(|&t| b.new_value(t)).collect();
            b.push(Op::Call { callee, args: operands, results: out.clone(), body }, loc);
            out
        });
        result.first().map(|&v| AnyValue(v))
    }

    /// [`call`](Self::call) for a function returning a `T`.
    #[track_caller]
    pub fn call_value<T: Numeric>(&self, args: &[CallArg]) -> Value<T> {
        let loc = std::panic::Location::caller();
        match self.call(args) {
            Some(v) if builder::value_dtype(v.0) == Some(T::DTYPE) => Value::from_id(v.0),
            _ => builder::user_panic(loc, format!("cute: `{}` does not return a {}", self.name, T::DTYPE)),
        }
    }
}

/// The integer value of an integer literal.
fn literal_int(lit: Literal) -> i64 {
    let (t, bits) = (lit.dtype(), lit.bits());
    let width = t.bits();
    if t.is_signed() && width < 64 { ((bits << (64 - width)) as i64) >> (64 - width) } else { bits as i64 }
}

/// Trace `f` on `args` in a region of the kernel being built: what the simulator runs for the
/// call.
fn trace_body(f: &SimBody, args: &[CallArg], results: &[DType], name: &str, loc: crate::ir::Loc) -> crate::ir::Region {
    builder::with(|b| b.push_region(RegionKind::Branch));
    let out = f(args);
    let closed = builder::with(|b| b.pop_region());
    if !closed.written.is_empty() {
        builder::user_panic(loc, format!("cute: the simulated body of `{name}` sets a Var of the caller"));
    }
    let yields: Vec<_> = out.into_iter().map(|v| v.0).collect();
    let types: Vec<DType> = yields.iter().map(|&v| builder::value_dtype(v).expect("a value")).collect();
    if types != results {
        builder::user_panic(
            loc,
            format!("cute: the simulated body of `{name}` returns {types:?}, but the function returns {results:?}"),
        );
    }
    crate::ir::Region { insts: closed.insts, yields }
}

/// Add `decl` to the kernel being built (or find the same prototype), and its bitcode source.
fn declare(decl: ExternDecl, source: Option<PathBuf>, loc: crate::ir::Loc) -> ExternId {
    builder::with(|b| {
        if let Some(lib) = source
            && !b.link_libraries.contains(&lib)
        {
            b.link_libraries.push(lib);
        }
        if let Some(i) = b.externs.iter().position(|d| d.symbol == decl.symbol) {
            let existing = &b.externs[i];
            if !existing.same_prototype(&decl) {
                builder::user_panic(
                    loc,
                    format!(
                        "cute: conflicting prototypes for external function `{}`: `{}` (declared at {}) and `{}`",
                        decl.symbol,
                        existing.prototype(),
                        existing.loc,
                        decl.prototype()
                    ),
                );
            }
            // How calls are compiled is the declaration's, so declarations of one symbol
            // must agree on inlining; convergence is kept if any asks for it, and a raw-bits
            // simulation is taken from whichever declaration has one.
            if existing.inline != decl.inline {
                builder::user_panic(
                    loc,
                    format!(
                        "cute: external function `{}` is called both inlined and not inlined (declared at {}): \
                         use one inline setting per symbol in a kernel",
                        decl.symbol, existing.loc
                    ),
                );
            }
            let existing = &mut b.externs[i];
            existing.convergent |= decl.convergent;
            if existing.sim.is_none() {
                existing.sim = decl.sim;
            }
            return ExternId(i as u32);
        }
        b.externs.push(decl);
        ExternId(b.externs.len() as u32 - 1)
    })
}

/// Overloads of one external function (CuTe DSL's `@overload` variants of a `@cute.extern`):
/// a call takes the first variant whose parameters fit the arguments. Each variant's symbol is
/// mangled.
#[derive(Clone, Debug, Default)]
pub struct ExternOverloads {
    variants: Vec<ExternFn>,
}

impl ExternOverloads {
    pub fn new() -> ExternOverloads {
        ExternOverloads::default()
    }

    /// Add a variant (overloaded unless it says otherwise).
    pub fn variant(mut self, f: ExternFn) -> ExternOverloads {
        let f = if f.overloaded.is_none() { f.overloaded(true) } else { f };
        self.variants.push(f);
        self
    }

    /// Call the first variant that fits `args`.
    #[track_caller]
    pub fn call(&self, args: &[CallArg]) -> Option<AnyValue> {
        let loc = std::panic::Location::caller();
        match self.variants.iter().find(|v| v.accepts(args)) {
            Some(v) => v.call(args),
            None => {
                let name = self.variants.first().map_or("?", |v| v.name.as_str());
                let given: Vec<String> = args.iter().map(CallArg::describe).collect();
                builder::user_panic(loc, format!("cute: no overload of `{name}` takes ({})", given.join(", ")))
            }
        }
    }
}

/// A parameter of a [`DeviceFunction`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceParam {
    Value(DType),
    /// A pointer (generic in the ABI) to elements of this type.
    Ptr(DType, usize),
}

/// Builds a [`DeviceFunction`]: [`DeviceFunction::builder`].
#[derive(Clone, Debug)]
pub struct DeviceFunctionBuilder {
    name: String,
    params: Vec<DeviceParam>,
    ret: Option<DType>,
    target: Option<Arch>,
}

impl DeviceFunctionBuilder {
    /// The next parameter: a `T` value.
    pub fn param<T: Numeric>(mut self) -> Self {
        self.params.push(DeviceParam::Value(T::DTYPE));
        self
    }

    /// The next parameter: a pointer to `T` elements (a C `T*`: generic, 64 bits), which the
    /// body may read and write as global memory. The caller promises the element's alignment;
    /// [`param_ptr_aligned`](Self::param_ptr_aligned) promises more.
    pub fn param_ptr<T: Numeric>(self) -> Self {
        self.param_ptr_aligned::<T>(T::DTYPE.bytes().max(1))
    }

    /// [`param_ptr`](Self::param_ptr) with the alignment the caller promises.
    pub fn param_ptr_aligned<T: Numeric>(mut self, align: usize) -> Self {
        self.params.push(DeviceParam::Ptr(T::DTYPE, align));
        self
    }

    /// The result: a `T`.
    pub fn returns<T: Numeric>(mut self) -> Self {
        self.ret = Some(T::DTYPE);
        self
    }

    /// The target the function is built for (checked at the end of the build).
    pub fn target(mut self, arch: Arch) -> Self {
        self.target = Some(arch);
        self
    }

    /// Build the function from `body`, which gets the parameters as [`CallArg`]s (values and
    /// pointers) and returns the result (`None` without one). The body is also what the
    /// simulator runs for every call of the function from a kernel. Panics if the body
    /// misuses the DSL or breaks a device function's restrictions.
    #[track_caller]
    pub fn build(self, body: impl Fn(&[CallArg]) -> Option<AnyValue> + Send + Sync + 'static) -> DeviceFunction {
        let body: SimBody = Arc::new(body);
        let traced = body.clone();
        let params = self.params.clone();
        let ret = self.ret;
        let name = self.name.clone();
        let mut options = Kernel::builder(self.name.clone());
        if let Some(arch) = self.target {
            options = options.target(arch);
        }
        let loc = std::panic::Location::caller();
        let kernel = options.build(move || {
            builder::with(|b| b.device_function = Some(DeviceFunctionSig { result: None }));
            let args: Vec<CallArg> = params
                .iter()
                .enumerate()
                .map(|(k, p)| match *p {
                    DeviceParam::Value(t) => {
                        CallArg::Value(AnyValue(builder::with(|b| b.declare_scalar_param(&format!("arg{k}"), t))))
                    }
                    DeviceParam::Ptr(t, align) => {
                        let buffer = builder::with(|b| {
                            b.declare_buffer(crate::ir::BufferDecl {
                                name: format!("arg{k}"),
                                elem: t,
                                len: None,
                                space: MemSpace::Global,
                                align,
                                fields: Vec::new(),
                                partition: crate::ir::SmemPartition::User,
                            })
                        });
                        CallArg::Ptr(AnyPtr { buffer, space: MemSpace::Global, offset: IndexExpr::Static(0), elem: t })
                    }
                })
                .collect();
            let out = traced(&args);
            let got = out.map(|v| (v.0, builder::value_dtype(v.0).expect("a value")));
            match (ret, got) {
                (None, None) => {}
                (Some(t), Some((_, u))) if t == u => {}
                (want, got) => builder::user_panic(
                    loc,
                    format!(
                        "cute: device function `{name}` is declared to return {want:?}, but its body returns {:?}",
                        got.map(|g| g.1)
                    ),
                ),
            }
            builder::with(|b| b.device_function = Some(DeviceFunctionSig { result: got.map(|g| g.0) }));
        });
        DeviceFunction { kernel, params: self.params, ret: self.ret, body }
    }
}

/// A standalone `__device__` function (CuTe DSL's `cute.compile[cute.DeviceTarget]`).
#[derive(Clone)]
pub struct DeviceFunction {
    kernel: Kernel,
    params: Vec<DeviceParam>,
    ret: Option<DType>,
    body: SimBody,
}

impl std::fmt::Debug for DeviceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFunction")
            .field("name", &self.name())
            .field("params", &self.params)
            .field("ret", &self.ret)
            .finish_non_exhaustive()
    }
}

impl DeviceFunction {
    /// Start building a device function named `name` (its C symbol).
    pub fn builder(name: impl Into<String>) -> DeviceFunctionBuilder {
        DeviceFunctionBuilder { name: name.into(), params: Vec::new(), ret: None, target: None }
    }

    /// The function's name (its symbol).
    pub fn name(&self) -> &str {
        self.kernel.name()
    }

    /// Its IR (a kernel IR whose [`function`](crate::ir::KernelIr::function) is set).
    pub fn ir(&self) -> &crate::ir::KernelIr {
        self.kernel.ir()
    }

    /// Human-readable IR listing.
    pub fn dump(&self) -> String {
        self.kernel.dump()
    }

    /// The function as MLIR: a non-kernel `gpu.func` with a result.
    pub fn to_mlir(&self) -> String {
        self.kernel.to_mlir()
    }

    /// Compile to PTX for `arch` (`.visible .func`).
    pub fn compile_ptx(&self, toolchain: &Toolchain, arch: Arch) -> Result<String, CompileError> {
        self.kernel.compile_ptx(toolchain, arch)
    }

    /// Compile with `options` ([`Kernel::compile_with`]).
    pub fn compile_with(
        &self,
        toolchain: &Toolchain,
        options: &CompileOptions,
    ) -> Result<crate::CompiledKernel, CompileError> {
        self.kernel.compile_with(toolchain, options)
    }

    /// Relocatable device code (CuTe DSL's `cute.RDC`): the PTX assembled with `ptxas -c`
    /// into a relocatable cubin, which `nvlink` (or `nvcc -rdc=true`) links into CUDA C++
    /// that declares `extern "C" __device__` the function. Needs the CUDA toolkit's `ptxas`.
    pub fn compile_relocatable(&self, toolchain: &Toolchain, arch: Arch) -> Result<Vec<u8>, CompileError> {
        let ptx = self.compile_ptx(toolchain, arch)?;
        let tool = toolchain.cuda_tools().ptxas.as_ref().ok_or_else(|| {
            CompileError::Tool("relocatable device code needs ptxas (the CUDA toolkit), which was not found".into())
        })?;
        let out = crate::backend::run_ptxas(tool, arch.chip(), &ptx, 3, false, &["-c".to_string()])?;
        Ok(out.cubin)
    }

    /// The function as an [`ExternFn`] for kernels to call: not inlined (the function is in
    /// another module, linked at load time with
    /// [`CudaContext::load_linked`](crate::cuda::CudaContext::load_linked)), convergent, and
    /// simulated by this function's own body.
    pub fn extern_fn(&self) -> ExternFn {
        let mut f = ExternFn::new(self.name()).inline(false);
        for p in &self.params {
            f = match *p {
                DeviceParam::Value(t) => f.param_type(ParamType::Value(t)),
                DeviceParam::Ptr(..) => f.param_ptr(AddressSpace::Generic),
            };
        }
        if let Some(t) = self.ret {
            f.ret = ReturnType::Value(t);
        }
        f.sim_body = Some(self.body.clone());
        f
    }

    /// Call the function from the kernel being built ([`extern_fn`](Self::extern_fn)).
    #[track_caller]
    pub fn call(&self, args: &[CallArg]) -> Option<AnyValue> {
        self.extern_fn().call(args)
    }
}

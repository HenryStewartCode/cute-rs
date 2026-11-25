//! The kernel argument protocol: what a kernel parameter is on both sides of
//! a launch.
//!
//! A type implementing [`KernelParam`] declares kernel parameters from a
//! build-time description (its [`Spec`](KernelParam::Spec)) and rebuilds
//! itself from them inside the kernel, and on the host checks a matching
//! value (its [`Host`](KernelParam::Host)) against the same description,
//! packs it into the flat argument list in the same order, and takes the
//! outputs back after a run. It is CuTe DSL's `JitArgument` protocol
//! (`__get_mlir_types__`, `__new_from_mlir_values__`, `__c_pointers__`)
//! with the description made explicit; its loop half is [`Carry`] (CuTe
//! DSL's `DynamicExpression`).
//!
//! Scalars ([`Value`]), tensors ([`Tensor`](crate::Tensor) from a [`TensorSpec`]), pointers
//! ([`Ptr`]), layouts, shapes and composed layouts ([`LayoutSpec`],
//! [`ComposedSpec`]), host-built [`FastDivmod`]s, tuples, arrays, `Vec`s and
//! `Option`s implement it, as do build-time constants (plain Rust integers,
//! floats and `bool`s, and [`Constexpr`]), which declare nothing and are
//! checked at launch. Structs get it from [`kernel_params!`](crate::kernel_params).
//! Third-party types implement the trait themselves, or through a newtype
//! (Rust's orphan rule takes the place of CuTe DSL's
//! `register_jit_arg_adapter`).
//!
//! Parameter names are built from the name given and each field's or
//! leaf's path: a tensor `a` declares the buffer `a` and its dynamic leaves
//! `a_shape_0`, `a_stride_1`; element `i` of a tuple named `t` is `t_i`
//! (`arg{i}` at the top level of a [`Kernel::jit`](crate::Kernel::jit)
//! signature); a struct's field `f` is `{name}_f` (`f` at the top level).
//! A [`SymInt`] used in several places is one parameter, named after its
//! first occurrence. The kernel's IR ([`KernelIr::args`](crate::ir::KernelIr))
//! stays the single list of parameters: [`ArgPack`] builds the matching
//! arguments, and [`TypedKernel`] checks the names against it.
//!
//! ```
//! use cute::prelude::*;
//!
//! // One runtime extent shared by both tensors, a multiple of 4.
//! let n = sym_int32(4, Some("n"));
//! let t = make_fake_compact_tensor::<f32>((n,), None, false);
//! let k = Kernel::builder("axpy").block_dim([64, 1, 1]).jit(
//!     (ScalarSpec::Plain, t.clone(), t),
//!     |a: Float32, x: Tensor<f32>, y: Tensor<f32>| {
//!         let i = block_idx().x * 64 + thread_idx().x;
//!         if_(i.lt(size(&x, &[0])), || y.store(i, a * x.load(i) + y.load(i)));
//!     },
//! );
//! assert_eq!(k.param_names(), ["arg0", "arg1", "arg1_shape_0", "arg2"]);
//!
//! let host = |v: Vec<f32>| HostTensor::from_slice(&v, Layout::col_major((v.len(),)));
//! let mut args = (2.0f32, host(vec![1.0; 8]), host(vec![0.5; 8]));
//! k.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
//! assert_eq!(args.2.to_vec::<f32>(), [2.5; 8]);
//! // A length that breaks the divisibility is refused before the launch.
//! let mut bad = (2.0f32, host(vec![1.0; 6]), host(vec![0.5; 6]));
//! assert!(k.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut bad).is_err());
//! ```

mod jit;
mod layout;
mod structs;
mod symint;
mod tensor;

pub use jit::{JitBody, TypedKernel};
pub use layout::{ComposedSpec, LayoutSpec, param_composed_layout, param_layout, param_shape};
pub use structs::{NativeField, NativeLayout, NativeMember};
pub use symint::{SpecLeaf, SpecTuple, SymInt, SymIntKey, SymRem, sym_int, sym_int32, sym_int64};
pub use tensor::{HostTensor, SpecError, TensorParamSource, TensorSpec, make_fake_compact_tensor, make_fake_tensor};
pub(crate) use tensor::{declare_tensor, pack_tensor, pack_tensor_with};

use std::collections::HashMap;
use std::fmt;

use crate::builder;
use crate::control::Carry;
use crate::fast_divmod::{FastDivmod, HostFastDivmod, param_fast_divmod};
use crate::ir::{DType, ValueId};
use crate::sim::KernelArg;
use crate::tensor::{Ptr, param_ptr_aligned, param_scalar, storage_decl};
use crate::value::{Numeric, Value};

/// A launch argument that does not match the kernel's parameter: `path` is
/// the parameter (or the first parameter of the value) it was given for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgError {
    pub path: String,
    pub message: String,
}

impl ArgError {
    pub(crate) fn new(path: &str, message: impl Into<String>) -> ArgError {
        ArgError { path: path.to_string(), message: message.into() }
    }
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "argument `{}`: {}", self.path, self.message)
    }
}

impl std::error::Error for ArgError {}

/// The flat argument list of a launch, built from host values parameter by
/// parameter in declaration order ([`ArgPack::push`]), with the name of each
/// argument's parameter and the value given to each [`SymInt`].
#[derive(Default, Debug)]
pub struct ArgPack {
    args: Vec<KernelArg>,
    names: Vec<String>,
    syms: HashMap<u64, (u64, String)>,
    /// What each argument is backed by when it is external memory (a DLPack tensor).
    external: Vec<Option<ExternalArg>>,
}

/// A buffer argument backed by external memory ([`DlTensor`](crate::DlTensor)): device memory
/// the launch passes by address (the argument itself is an empty placeholder), or host memory
/// copied in; and whether the producer marked it read-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalArg {
    /// The device address, the bytes reachable from it and the CUDA device ordinal, for
    /// device memory.
    pub device: Option<(u64, usize, i32)>,
    pub read_only: bool,
}

impl ArgPack {
    pub fn new() -> ArgPack {
        ArgPack::default()
    }

    /// Check `host` against `spec` and append its arguments, as `P` named
    /// `name` declared its parameters.
    pub fn push<P: KernelParam>(&mut self, name: &str, spec: &P::Spec, host: &P::Host) -> Result<(), ArgError> {
        P::pack(name, spec, host, self)
    }

    /// Append one argument for the parameter `name`.
    pub fn arg(&mut self, name: &str, arg: KernelArg) {
        self.args.push(arg);
        self.names.push(name.to_string());
        self.external.push(None);
    }

    /// Mark argument `index` as backed by external memory.
    pub fn set_external(&mut self, index: usize, external: ExternalArg) {
        self.external[index] = Some(external);
    }

    /// What backs each argument, when it is external memory.
    pub fn externals(&self) -> &[Option<ExternalArg>] {
        &self.external
    }

    /// The value of the [`SymInt`] `sym` at `name`: checked against its
    /// divisibility and width, and against the value its earlier occurrences
    /// were given; appended (as an `i32` or `i64`) at its first occurrence.
    pub fn sym(&mut self, name: &str, sym: SymInt, value: u64) -> Result<(), ArgError> {
        let what = || match sym.symbol() {
            Some(s) => format!("`{s}` ({})", sym.leaf_text()),
            None => sym.leaf_text(),
        };
        if !value.is_multiple_of(sym.divisibility() as u64) {
            return Err(ArgError::new(
                name,
                format!("{} must be a multiple of {}, got {value}", what(), sym.divisibility()),
            ));
        }
        if value > sym.max_value() {
            return Err(ArgError::new(name, format!("{value} does not fit in the {}-bit {}", sym.width(), what())));
        }
        if let Some((first, at)) = self.syms.get(&sym.id()) {
            if *first != value {
                return Err(ArgError::new(
                    name,
                    format!("got {value}, but `{at}` is the same SymInt {} and got {first}", what()),
                ));
            }
            return Ok(());
        }
        self.syms.insert(sym.id(), (value, name.to_string()));
        let arg = if sym.width() == 64 { KernelArg::scalar(value as i64) } else { KernelArg::scalar(value as i32) };
        self.arg(name, arg);
        Ok(())
    }

    /// The arguments so far.
    pub fn args(&self) -> &[KernelArg] {
        &self.args
    }

    /// The parameter name of each argument.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn finish(self) -> Vec<KernelArg> {
        self.args
    }
}

/// A type that is a kernel parameter: declared inside [`Kernel::build`]
/// from a [`Spec`](Self::Spec) and given a [`Host`](Self::Host) value at
/// launch (see the [module documentation](self)).
///
/// [`Kernel::build`]: crate::Kernel::build
pub trait KernelParam: Sized + Clone {
    /// What the kernel is built for: a [`TensorSpec`] for a tensor, a
    /// [`LayoutSpec`] for a layout, the value itself for a constant.
    type Spec: Clone;
    /// What a launch passes.
    type Host;

    /// Declare the parameters under `name` and build the value the kernel
    /// sees (inside [`Kernel::build`](crate::Kernel::build)).
    fn declare(name: &str, spec: &Self::Spec) -> Self;

    /// The spec a host value implies (CuTe DSL's call-site typing): a host
    /// tensor's own spec, a dynamic leaf for each integer of a shape, the
    /// value of a constant.
    fn spec_of(host: &Self::Host) -> Self::Spec;

    /// Check `host` against `spec` and append its arguments to `out`, in the
    /// order [`declare`](Self::declare) declared the parameters.
    fn pack(name: &str, spec: &Self::Spec, host: &Self::Host, out: &mut ArgPack) -> Result<(), ArgError>;

    /// Take the outputs of a run back into `host`: `buffers` yields the
    /// launch's buffer arguments (not its scalars) in order, and each buffer
    /// parameter takes the next one.
    fn write_back(spec: &Self::Spec, host: &mut Self::Host, buffers: &mut dyn Iterator<Item = KernelArg>) {
        let _ = (spec, host, buffers);
    }

    /// [`Carry::flatten`] for a field of a [`kernel_params!`](crate::kernel_params)
    /// struct: runtime values are carried through loops; tensors in memory,
    /// layouts and constants are not (they are the same in every iteration).
    fn carry_flatten(&self, out: &mut Vec<ValueId>) {
        let _ = out;
    }

    /// [`Carry::rebuild`] to match [`carry_flatten`](Self::carry_flatten).
    fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let _ = values;
        self.clone()
    }

    /// [`Carry::structure_mismatch`]: what is not carried must not change.
    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        let _ = other;
        None
    }

    /// A native-struct field's initial value: zero, or (with `zero` false)
    /// undefined contents. Only scalars, constants and native structs can be
    /// native-struct fields.
    fn native_init(spec: &Self::Spec, zero: bool) -> Self {
        let _ = (spec, zero);
        panic!("cute: {} cannot be a field of a native struct", std::any::type_name::<Self>())
    }

    /// What this value adds to a native struct's layout: its scalars, or
    /// (for a native struct) itself as one member.
    fn native_fields(name: &str, spec: &Self::Spec, out: &mut Vec<NativeMember>) {
        let _ = (name, spec, out);
        panic!("cute: {} cannot be a field of a native struct", std::any::type_name::<Self>())
    }
}

/// The name of a single parameter (`arg0` at the top level, where the name
/// is empty).
pub(crate) fn leaf_name(name: &str) -> String {
    if name.is_empty() { "arg0".into() } else { name.to_string() }
}

/// The name of element `i` of a tuple or array named `name`.
pub(crate) fn element_name(name: &str, i: usize) -> String {
    if name.is_empty() { format!("arg{i}") } else { format!("{name}_{i}") }
}

/// The name of field `field` of a struct named `name`.
pub fn field_name(name: &str, field: &str) -> String {
    if name.is_empty() { field.to_string() } else { format!("{name}_{field}") }
}

/// How a scalar parameter is declared: a plain scalar of its type, or the
/// value of a [`SymInt`] (shared with every other place that `SymInt`
/// appears, with its divisibility known).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalarSpec {
    #[default]
    Plain,
    Sym(SymInt),
}

impl From<SymInt> for ScalarSpec {
    fn from(s: SymInt) -> Self {
        ScalarSpec::Sym(s)
    }
}

/// The integer value of `x`, sign-extended from `t`'s width.
fn integer_value(t: DType, bits: u64) -> i128 {
    let w = t.bits() as u32;
    if t.is_signed() && w < 64 {
        let shift = 64 - w;
        (((bits << shift) as i64) >> shift) as i128
    } else if t.is_signed() {
        bits as i64 as i128
    } else {
        bits as i128
    }
}

impl<T: Numeric> KernelParam for Value<T> {
    type Spec = ScalarSpec;
    type Host = T;

    #[track_caller]
    fn declare(name: &str, spec: &ScalarSpec) -> Self {
        let name = leaf_name(name);
        match spec {
            ScalarSpec::Plain => param_scalar::<T>(&name),
            ScalarSpec::Sym(s) => {
                assert!(
                    T::DTYPE == s.dtype(),
                    "cute: parameter `{name}`: the {}-bit SymInt {s} is an {}, not a {}",
                    s.width(),
                    s.dtype().cute_name(),
                    T::DTYPE.cute_name()
                );
                let v = builder::with(|b| b.declare_symint(s.id(), &name, s.divisibility(), s.dtype()));
                Value::from_id(v)
            }
        }
    }

    fn spec_of(_: &T) -> ScalarSpec {
        ScalarSpec::Plain
    }

    fn pack(name: &str, spec: &ScalarSpec, host: &T, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        match spec {
            ScalarSpec::Plain => out.arg(&name, KernelArg::scalar(*host)),
            ScalarSpec::Sym(s) => {
                let v = integer_value(T::DTYPE, host.raw_bits());
                if v < 0 {
                    return Err(ArgError::new(&name, format!("{} must be non-negative, got {v}", s.leaf_text())));
                }
                out.sym(&name, *s, v as u64)?;
            }
        }
        Ok(())
    }

    #[track_caller]
    fn carry_flatten(&self, out: &mut Vec<ValueId>) {
        Carry::flatten(self, out);
    }

    fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        Carry::rebuild(self, values)
    }

    #[track_caller]
    fn native_init(_: &ScalarSpec, _zero: bool) -> Self {
        // cute-rs has no undefined value: an uninitialised field starts at 0.
        Value::constant(T::from_bits(0))
    }

    fn native_fields(name: &str, _: &ScalarSpec, out: &mut Vec<NativeMember>) {
        out.push(NativeMember::Scalar(leaf_name(name), T::DTYPE));
    }
}

/// A build-time constant parameter (CuTe DSL's `cutlass.Constexpr[T]`): no
/// runtime parameter; the kernel sees the value it was built with, and a
/// launch checks that the host passes the same one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Constexpr<T>(pub T);

impl<T> std::ops::Deref for Constexpr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

fn check_constant<T: PartialEq + fmt::Debug>(name: &str, spec: &T, host: &T) -> Result<(), ArgError> {
    if spec == host {
        return Ok(());
    }
    Err(ArgError::new(
        &leaf_name(name),
        format!("the kernel was built with the constant {spec:?}, launched with {host:?}"),
    ))
}

fn constant_mismatch<T: PartialEq + fmt::Debug>(a: &T, b: &T) -> Option<String> {
    (a != b).then(|| format!("the constant {a:?} became {b:?}"))
}

impl<T: Clone + PartialEq + fmt::Debug> KernelParam for Constexpr<T> {
    type Spec = T;
    type Host = T;

    fn declare(_: &str, spec: &T) -> Self {
        Constexpr(spec.clone())
    }

    fn spec_of(host: &T) -> T {
        host.clone()
    }

    fn pack(name: &str, spec: &T, host: &T, _: &mut ArgPack) -> Result<(), ArgError> {
        check_constant(name, spec, host)
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        constant_mismatch(&self.0, &other.0)
    }

    fn native_init(spec: &T, _: bool) -> Self {
        Constexpr(spec.clone())
    }

    fn native_fields(_: &str, _: &T, _: &mut Vec<NativeMember>) {}
}

macro_rules! constant_params {
    ($($t:ty),*) => {$(
        /// A plain Rust value is a build-time constant, as a Python-valued
        /// field is in CuTe DSL.
        impl KernelParam for $t {
            type Spec = $t;
            type Host = $t;

            fn declare(_: &str, spec: &$t) -> Self {
                spec.clone()
            }

            fn spec_of(host: &$t) -> $t {
                host.clone()
            }

            fn pack(name: &str, spec: &$t, host: &$t, _: &mut ArgPack) -> Result<(), ArgError> {
                check_constant(name, spec, host)
            }

            fn carry_mismatch(&self, other: &Self) -> Option<String> {
                constant_mismatch(self, other)
            }

            fn native_init(spec: &$t, _: bool) -> Self {
                spec.clone()
            }

            fn native_fields(_: &str, _: &$t, _: &mut Vec<NativeMember>) {}
        }
    )*};
}
constant_params!(usize, isize, u8, u16, u32, u64, i8, i16, i32, i64, bool, f32, f64, String, &'static str);

impl KernelParam for () {
    type Spec = ();
    type Host = ();

    fn declare(_: &str, _: &()) -> Self {}

    fn spec_of(_: &()) {}

    fn pack(_: &str, _: &(), _: &(), _: &mut ArgPack) -> Result<(), ArgError> {
        Ok(())
    }
}

macro_rules! tuple_params {
    ($(($($name:ident $idx:tt),+)),+) => {$(
        impl<$($name: KernelParam),+> KernelParam for ($($name,)+) {
            type Spec = ($($name::Spec,)+);
            type Host = ($($name::Host,)+);

            #[track_caller]
            fn declare(name: &str, spec: &Self::Spec) -> Self {
                ($($name::declare(&element_name(name, $idx), &spec.$idx),)+)
            }

            fn spec_of(host: &Self::Host) -> Self::Spec {
                ($($name::spec_of(&host.$idx),)+)
            }

            fn pack(name: &str, spec: &Self::Spec, host: &Self::Host, out: &mut ArgPack) -> Result<(), ArgError> {
                $($name::pack(&element_name(name, $idx), &spec.$idx, &host.$idx, out)?;)+
                Ok(())
            }

            fn write_back(spec: &Self::Spec, host: &mut Self::Host, buffers: &mut dyn Iterator<Item = KernelArg>) {
                $($name::write_back(&spec.$idx, &mut host.$idx, buffers);)+
            }

            #[track_caller]
            fn carry_flatten(&self, out: &mut Vec<ValueId>) {
                $(self.$idx.carry_flatten(out);)+
            }

            fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
                ($(self.$idx.carry_rebuild(values),)+)
            }

            fn carry_mismatch(&self, other: &Self) -> Option<String> {
                None$(.or_else(|| self.$idx.carry_mismatch(&other.$idx).map(|m| format!("element {}: {m}", $idx))))+
            }

            #[track_caller]
            fn native_init(spec: &Self::Spec, zero: bool) -> Self {
                ($($name::native_init(&spec.$idx, zero),)+)
            }

            fn native_fields(name: &str, spec: &Self::Spec, out: &mut Vec<NativeMember>) {
                $($name::native_fields(&element_name(name, $idx), &spec.$idx, out);)+
            }
        }
    )+};
}
tuple_params!((A 0), (A 0, B 1), (A 0, B 1, C 2), (A 0, B 1, C 2, D 3), (A 0, B 1, C 2, D 3, E 4),
    (A 0, B 1, C 2, D 3, E 4, F 5), (A 0, B 1, C 2, D 3, E 4, F 5, G 6), (A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7));

impl<P: KernelParam, const N: usize> KernelParam for [P; N] {
    type Spec = [P::Spec; N];
    type Host = [P::Host; N];

    #[track_caller]
    fn declare(name: &str, spec: &Self::Spec) -> Self {
        let mut out = Vec::with_capacity(N);
        for (i, s) in spec.iter().enumerate() {
            out.push(P::declare(&element_name(name, i), s));
        }
        out.try_into().unwrap_or_else(|_| unreachable!())
    }

    fn spec_of(host: &Self::Host) -> Self::Spec {
        std::array::from_fn(|i| P::spec_of(&host[i]))
    }

    fn pack(name: &str, spec: &Self::Spec, host: &Self::Host, out: &mut ArgPack) -> Result<(), ArgError> {
        for (i, (s, h)) in spec.iter().zip(host).enumerate() {
            P::pack(&element_name(name, i), s, h, out)?;
        }
        Ok(())
    }

    fn write_back(spec: &Self::Spec, host: &mut Self::Host, buffers: &mut dyn Iterator<Item = KernelArg>) {
        for (s, h) in spec.iter().zip(host.iter_mut()) {
            P::write_back(s, h, buffers);
        }
    }

    #[track_caller]
    fn carry_flatten(&self, out: &mut Vec<ValueId>) {
        self.iter().for_each(|x| x.carry_flatten(out));
    }

    fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        std::array::from_fn(|i| self[i].carry_rebuild(values))
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        self.iter()
            .zip(other)
            .enumerate()
            .find_map(|(i, (a, b))| a.carry_mismatch(b).map(|m| format!("element {i}: {m}")))
    }
}

impl<P: KernelParam> KernelParam for Vec<P> {
    type Spec = Vec<P::Spec>;
    type Host = Vec<P::Host>;

    #[track_caller]
    fn declare(name: &str, spec: &Self::Spec) -> Self {
        spec.iter().enumerate().map(|(i, s)| P::declare(&element_name(name, i), s)).collect()
    }

    fn spec_of(host: &Self::Host) -> Self::Spec {
        host.iter().map(P::spec_of).collect()
    }

    fn pack(name: &str, spec: &Self::Spec, host: &Self::Host, out: &mut ArgPack) -> Result<(), ArgError> {
        if spec.len() != host.len() {
            return Err(ArgError::new(
                &leaf_name(name),
                format!("the kernel takes {} elements, got {}", spec.len(), host.len()),
            ));
        }
        for (i, (s, h)) in spec.iter().zip(host).enumerate() {
            P::pack(&element_name(name, i), s, h, out)?;
        }
        Ok(())
    }

    fn write_back(spec: &Self::Spec, host: &mut Self::Host, buffers: &mut dyn Iterator<Item = KernelArg>) {
        for (s, h) in spec.iter().zip(host.iter_mut()) {
            P::write_back(s, h, buffers);
        }
    }

    #[track_caller]
    fn carry_flatten(&self, out: &mut Vec<ValueId>) {
        self.iter().for_each(|x| x.carry_flatten(out));
    }

    fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        self.iter().map(|x| x.carry_rebuild(values)).collect()
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        if self.len() != other.len() {
            return Some(format!("{} elements became {}", self.len(), other.len()));
        }
        self.iter()
            .zip(other)
            .enumerate()
            .find_map(|(i, (a, b))| a.carry_mismatch(b).map(|m| format!("element {i}: {m}")))
    }
}

/// An optional parameter: `None` declares nothing (CuTe DSL drops a `None`
/// argument), and a launch must agree with the build on which it is.
impl<P: KernelParam> KernelParam for Option<P> {
    type Spec = Option<P::Spec>;
    type Host = Option<P::Host>;

    #[track_caller]
    fn declare(name: &str, spec: &Self::Spec) -> Self {
        spec.as_ref().map(|s| P::declare(name, s))
    }

    fn spec_of(host: &Self::Host) -> Self::Spec {
        host.as_ref().map(P::spec_of)
    }

    fn pack(name: &str, spec: &Self::Spec, host: &Self::Host, out: &mut ArgPack) -> Result<(), ArgError> {
        match (spec, host) {
            (Some(s), Some(h)) => P::pack(name, s, h, out),
            (None, None) => Ok(()),
            (Some(_), None) => {
                Err(ArgError::new(&leaf_name(name), "the kernel was built with a value (Some), launched with None"))
            }
            (None, Some(_)) => {
                Err(ArgError::new(&leaf_name(name), "the kernel was built without a value (None), launched with Some"))
            }
        }
    }

    fn write_back(spec: &Self::Spec, host: &mut Self::Host, buffers: &mut dyn Iterator<Item = KernelArg>) {
        if let (Some(s), Some(h)) = (spec, host) {
            P::write_back(s, h, buffers);
        }
    }

    #[track_caller]
    fn carry_flatten(&self, out: &mut Vec<ValueId>) {
        if let Some(x) = self {
            x.carry_flatten(out);
        }
    }

    fn carry_rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        self.as_ref().map(|x| x.carry_rebuild(values))
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        match (self, other) {
            (Some(a), Some(b)) => a.carry_mismatch(b),
            (None, None) => None,
            (Some(_), None) => Some("Some became None".into()),
            (None, Some(_)) => Some("None became Some".into()),
        }
    }
}

/// How a pointer parameter is declared: the alignment the host promises for
/// the buffer's start (CuTe DSL's `assumed_align`; 16 bytes by default, as
/// [`param_ptr`](crate::param_ptr)'s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtrSpec {
    pub assumed_align: usize,
}

impl Default for PtrSpec {
    fn default() -> Self {
        PtrSpec { assumed_align: 16 }
    }
}

/// The element type of a buffer argument (the logical type of packed
/// sub-byte values) and its number of elements.
pub(crate) fn buffer_elements(arg: &KernelArg) -> Option<(DType, usize)> {
    match arg {
        KernelArg::Bytes { elem, data } if elem.is_word_packed() => Some((*elem, data.len() * 8 / elem.bits())),
        _ => arg.buffer_type(),
    }
}

/// Check that `arg` is a buffer of `t` (or of its storage type).
fn check_buffer(name: &str, t: DType, arg: &KernelArg) -> Result<(), ArgError> {
    let storage = storage_decl(t, 0).0;
    match arg.buffer_type() {
        Some((e, _)) if e == storage || buffer_elements(arg).is_some_and(|(e, _)| e == t) => Ok(()),
        _ => Err(ArgError::new(name, format!("expected a {} buffer, got {}", t.cute_name(), arg.describe()))),
    }
}

impl<T: Numeric> KernelParam for Ptr<T> {
    type Spec = PtrSpec;
    type Host = KernelArg;

    #[track_caller]
    fn declare(name: &str, spec: &PtrSpec) -> Self {
        param_ptr_aligned::<T>(&leaf_name(name), spec.assumed_align)
    }

    fn spec_of(_: &KernelArg) -> PtrSpec {
        PtrSpec::default()
    }

    fn pack(name: &str, _: &PtrSpec, host: &KernelArg, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        check_buffer(&name, T::DTYPE, host)?;
        out.arg(&name, host.clone());
        Ok(())
    }

    fn write_back(_: &PtrSpec, host: &mut KernelArg, buffers: &mut dyn Iterator<Item = KernelArg>) {
        *host = buffers.next().expect("cute: write_back: too few buffers");
    }
}

/// A [`FastDivmod`] prepared on the host: three scalar parameters,
/// `{name}_divisor`, `{name}_multiplier` and `{name}_shift`.
impl KernelParam for FastDivmod {
    type Spec = ();
    type Host = HostFastDivmod;

    #[track_caller]
    fn declare(name: &str, _: &()) -> Self {
        param_fast_divmod(&leaf_name(name))
    }

    fn spec_of(_: &HostFastDivmod) {}

    fn pack(name: &str, _: &(), host: &HostFastDivmod, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        if host.divisor <= 0 {
            return Err(ArgError::new(&name, format!("a FastDivmod needs a positive divisor, got {}", host.divisor)));
        }
        let [d, m, s] = host.kernel_args();
        out.arg(&format!("{name}_divisor"), d);
        out.arg(&format!("{name}_multiplier"), m);
        out.arg(&format!("{name}_shift"), s);
        Ok(())
    }
}

/// The names of `ir`'s parameters, in order.
pub(crate) fn param_names(ir: &crate::ir::KernelIr) -> Vec<String> {
    ir.args
        .iter()
        .map(|p| match p {
            crate::ir::Param::Buffer(b) => ir.buffer(*b).name.clone(),
            crate::ir::Param::Scalar { name, .. } => name.clone(),
            crate::ir::Param::ByValue { buffer, .. } => ir.buffer(*buffer).name.clone(),
        })
        .collect()
}

impl crate::Kernel {
    /// The names of the kernel's parameters, in order (buffers and scalars).
    pub fn param_names(&self) -> Vec<String> {
        param_names(self.ir())
    }
}

/// Pack `host` for the parameters `P` declared under `name` in `kernel`, as
/// every parameter of the kernel: the names must be the kernel's own.
pub fn pack_args<P: KernelParam>(
    kernel: &crate::Kernel,
    name: &str,
    spec: &P::Spec,
    host: &P::Host,
) -> Result<Vec<KernelArg>, ArgError> {
    Ok(pack_args_full::<P>(kernel, name, spec, host)?.finish())
}

/// [`pack_args`], keeping the [`ArgPack`] (with what backs each argument). A read-only
/// external tensor given for a buffer the kernel writes is refused.
pub fn pack_args_full<P: KernelParam>(
    kernel: &crate::Kernel,
    name: &str,
    spec: &P::Spec,
    host: &P::Host,
) -> Result<ArgPack, ArgError> {
    let pack = pack_names::<P>(kernel, name, spec, host)?;
    let ir = kernel.ir();
    for ((p, ext), param) in ir.args.iter().zip(pack.externals()).zip(pack.names()) {
        if let (crate::ir::Param::Buffer(b), Some(ExternalArg { read_only: true, .. })) = (p, ext)
            && crate::signature::writes_buffer(ir, *b)
        {
            return Err(ArgError::new(
                param,
                format!(
                    "the tensor is read-only (DLPACK_FLAG_BITMASK_READ_ONLY), but kernel `{}` writes it",
                    kernel.name()
                ),
            ));
        }
    }
    Ok(pack)
}

fn pack_names<P: KernelParam>(
    kernel: &crate::Kernel,
    name: &str,
    spec: &P::Spec,
    host: &P::Host,
) -> Result<ArgPack, ArgError> {
    let mut pack = ArgPack::new();
    pack.push::<P>(name, spec, host)?;
    let want = param_names(kernel.ir());
    if pack.names() != want.as_slice() {
        return Err(ArgError::new(
            name,
            format!(
                "the arguments are for the parameters ({}), but kernel `{}` declares ({})",
                pack.names().join(", "),
                kernel.name(),
                want.join(", ")
            ),
        ));
    }
    Ok(pack)
}

/// Take the buffers of `args` (after a run) back into `host`.
pub fn write_back_args<P: KernelParam>(spec: &P::Spec, host: &mut P::Host, args: Vec<KernelArg>) {
    let mut buffers = args.into_iter().filter(|a| a.buffer_type().is_some());
    P::write_back(spec, host, &mut buffers);
}

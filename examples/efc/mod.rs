// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/efc/common_efc.py
//   (the EFC framework: `EFC`, its `Phase`s, `Tensor.ParameterAttributes`, the parameter
//   analysis, `Configuration` with `accum`, `maximum`, `minimum`, `full_like`, `where` and
//   the activation functions of `ACTIVATION_FUNCTIONS`, `foreach_*_tensor` and
//   `evaluate_on_cpu`, with their names),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! EFC, the epilogue fusion configuration framework of CuTe DSL's Blackwell EFC examples
//! (`common_efc.py`), as example-side code shared by the three
//! `*_custom_epilogue_dense_gemm` examples. It is not a crate API.
//!
//! The user writes one epilogue function over named scalars and tensors; EFC runs it in
//! three [`Phase`]s:
//!
//! - [`Phase::ParameterAnalysis`]: which tensors it reads ([`EfcConfig::load`]) and writes
//!   ([`EfcConfig::store`]), from which the GEMM derives its extra TMA loads, TMA stores and
//!   shared memory ([`dense_gemm::DenseGemmEfc`]);
//! - [`Phase::ThreadOperation`]: traced into the GEMM's epilogue, once per epilogue subtile,
//!   on the thread's register values (`TensorSsa<f32>`);
//! - [`Phase::HostEvaluation`]: the CPU reference, on `f64` matrices (the Python's
//!   `PyTorchEvaluation`).
//!
//! Differences from the Python, whose epilogue is a Python function with named parameters:
//! - Rust closures have no parameter names, so the signature is declared: [`Efc::new`] takes
//!   the [`Param`]s (the first must be [`Param::Config`], the Python's `efc_config`) and a
//!   closure over the [`EfcConfig`]; [`Efc::builder`] adds `efc_config` itself. Tensors
//!   are named in `load`/`store` and scalars in [`EfcConfig::scalar`].
//! - A tensor parameter declares its element type and majorness, and a scalar is a `Float32`
//!   kernel parameter. The epilogue type (`epi_dtype`) and the accumulator are Float32.
//! - The values are one type, [`EfcValue`], with a variant per phase (a marker during the
//!   analysis, `TensorSsa<f32>`/`Value<f32>` on the device, `f64` vectors on the host) and
//!   the arithmetic operators, so that one closure serves the three phases; the activation
//!   functions are its methods. The device formulas are the Python's `ThreadOperation`
//!   ones (GELU is the tanh approximation); the host ones are PyTorch's (GELU with `erf`).
//! - The rules are checked when the [`Efc`] is made, as [`EfcError`]s: the `efc_config`
//!   parameter, at least one written tensor, and (by the GEMM) one majorness for every
//!   tensor.

#![allow(clippy::should_implement_trait)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use cute::prelude::*;
use cute::utils::LayoutEnum;

pub mod dense_gemm;

/// The activation functions of [`EfcValue`] (and [`EfcConfig::activation`]).
pub const ACTIVATION_FUNCTIONS: [&str; 8] =
    ["identity", "relu", "leaky_relu", "tanh", "sigmoid", "silu", "hardswish", "gelu"];

/// The phases EFC runs the epilogue in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Recording which tensors the epilogue reads and writes.
    ParameterAnalysis,
    /// Traced into the kernel, on register subtiles.
    ThreadOperation,
    /// The CPU reference.
    HostEvaluation,
}

/// A parameter of the epilogue function.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    /// The configuration parameter, `efc_config` (first, and only first).
    Config,
    /// A Float32 scalar.
    Scalar(String),
    /// An `(M, N)` tensor of `dtype` in `layout`.
    Tensor { name: String, dtype: DType, layout: LayoutEnum },
}

impl Param {
    pub fn scalar(name: &str) -> Param {
        Param::Scalar(name.to_string())
    }

    /// A row-major tensor.
    pub fn tensor(name: &str, dtype: DType) -> Param {
        Param::tensor_with_layout(name, dtype, LayoutEnum::ROW_MAJOR)
    }

    pub fn tensor_with_layout(name: &str, dtype: DType, layout: LayoutEnum) -> Param {
        Param::Tensor { name: name.to_string(), dtype, layout }
    }

    /// The parameter's name (`efc_config` for [`Param::Config`]).
    pub fn name(&self) -> &str {
        match self {
            Param::Config => "efc_config",
            Param::Scalar(n) | Param::Tensor { name: n, .. } => n,
        }
    }
}

/// What the analysis found about a parameter (the Python's `ParameterAttributes`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParameterAttributes {
    pub is_tensor: bool,
    pub is_read: bool,
    pub is_written: bool,
}

/// The rules EFC enforces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EfcError {
    /// The signature does not start with the configuration parameter.
    MissingConfigParameter,
    /// The configuration parameter appears again after the first position.
    MisplacedConfigParameter,
    /// Two parameters share a name, or a name is one of the GEMM's own (`m`, `n`, `k`, `a`,
    /// `b`).
    BadParameterName(String),
    /// The epilogue stores to no tensor.
    NoWrittenTensor,
    /// A tensor's majorness differs from that of the C/D tensors.
    LayoutMismatch { tensor: String, layout: LayoutEnum, d_layout: LayoutEnum },
}

impl fmt::Display for EfcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EfcError::MissingConfigParameter => {
                write!(f, "EFC: the epilogue's first parameter must be the configuration, `efc_config`")
            }
            EfcError::MisplacedConfigParameter => write!(f, "EFC: `efc_config` may only be the first parameter"),
            EfcError::BadParameterName(n) => {
                write!(f, "EFC: the parameter name `{n}` is repeated or reserved for the GEMM")
            }
            EfcError::NoWrittenTensor => {
                write!(f, "EFC: the epilogue stores to no tensor, so it would compute nothing")
            }
            EfcError::LayoutMismatch { tensor, layout, d_layout } => write!(
                f,
                "EFC: tensor `{tensor}` is {layout:?} but the C/D tensors are {d_layout:?}; every EFC tensor needs \
                 the C/D majorness"
            ),
        }
    }
}

impl std::error::Error for EfcError {}

/// The epilogue function.
pub type EpilogueFn = dyn Fn(&EfcConfig);

/// An epilogue and what its analysis found (the Python's `EFC`).
#[derive(Clone)]
pub struct Efc {
    params: Vec<Param>,
    attributes: Vec<ParameterAttributes>,
    epilogue: Rc<EpilogueFn>,
    /// The tensors read or written, in parameter order.
    pub used_tensor_names: Vec<String>,
    pub read_tensor_names: Vec<String>,
    pub written_tensor_names: Vec<String>,
}

/// [`Efc::builder`]'s builder.
pub struct EfcBuilder {
    params: Vec<Param>,
}

impl EfcBuilder {
    pub fn scalar(mut self, name: &str) -> Self {
        self.params.push(Param::scalar(name));
        self
    }

    pub fn tensor(mut self, name: &str, dtype: DType) -> Self {
        self.params.push(Param::tensor(name, dtype));
        self
    }

    pub fn tensor_with_layout(mut self, name: &str, dtype: DType, layout: LayoutEnum) -> Self {
        self.params.push(Param::tensor_with_layout(name, dtype, layout));
        self
    }

    pub fn build(self, epilogue: impl Fn(&EfcConfig) + 'static) -> Result<Efc, EfcError> {
        Efc::new(self.params, epilogue)
    }
}

/// The GEMM's own kernel parameters.
const RESERVED: [&str; 5] = ["m", "n", "k", "a", "b"];

impl Efc {
    /// A signature starting with `efc_config`.
    pub fn builder() -> EfcBuilder {
        EfcBuilder { params: vec![Param::Config] }
    }

    /// The epilogue of `signature` (`efc_config` first), analysed (the Python's constructor
    /// and `compile`).
    pub fn new(signature: Vec<Param>, epilogue: impl Fn(&EfcConfig) + 'static) -> Result<Efc, EfcError> {
        let mut it = signature.into_iter();
        if it.next() != Some(Param::Config) {
            return Err(EfcError::MissingConfigParameter);
        }
        let params: Vec<Param> = it.collect();
        for (i, p) in params.iter().enumerate() {
            if *p == Param::Config {
                return Err(EfcError::MisplacedConfigParameter);
            }
            if RESERVED.contains(&p.name()) || params[..i].iter().any(|q| q.name() == p.name()) {
                return Err(EfcError::BadParameterName(p.name().to_string()));
            }
        }
        let mut efc = Efc {
            attributes: params
                .iter()
                .map(|p| ParameterAttributes { is_tensor: matches!(p, Param::Tensor { .. }), ..Default::default() })
                .collect(),
            params,
            epilogue: Rc::new(epilogue),
            used_tensor_names: vec![],
            read_tensor_names: vec![],
            written_tensor_names: vec![],
        };
        efc.analyze_epilogue();
        if efc.written_tensor_names.is_empty() {
            return Err(EfcError::NoWrittenTensor);
        }
        Ok(efc)
    }

    /// The parameters after `efc_config`.
    pub fn params(&self) -> &[Param] {
        &self.params
    }

    /// The analysis of parameter `name`.
    #[track_caller]
    pub fn parameter_attributes(&self, name: &str) -> ParameterAttributes {
        self.attributes[self.index(name)]
    }

    #[track_caller]
    fn index(&self, name: &str) -> usize {
        self.params.iter().position(|p| p.name() == name).unwrap_or_else(|| panic!("EFC: no parameter `{name}`"))
    }

    /// The declaration of tensor `name`.
    #[track_caller]
    pub fn tensor_param(&self, name: &str) -> (DType, LayoutEnum) {
        match &self.params[self.index(name)] {
            Param::Tensor { dtype, layout, .. } => (*dtype, *layout),
            _ => panic!("EFC: `{name}` is not a tensor parameter"),
        }
    }

    /// Run the epilogue in the analysis phase and record the read and written tensors.
    fn analyze_epilogue(&mut self) {
        let config = EfcConfig { efc: self, state: State::Analysis(RefCell::new(self.attributes.clone())) };
        (self.epilogue)(&config);
        let State::Analysis(attributes) = config.state else { unreachable!() };
        self.attributes = attributes.into_inner();
        for (p, a) in self.params.iter().zip(&self.attributes) {
            if !a.is_tensor {
                continue;
            }
            let name = p.name().to_string();
            if a.is_read || a.is_written {
                self.used_tensor_names.push(name.clone());
            }
            if a.is_read {
                self.read_tensor_names.push(name.clone());
            }
            if a.is_written {
                self.written_tensor_names.push(name);
            }
        }
    }

    /// The epilogue on one register subtile of the kernel: `acc` the accumulator converted
    /// to Float32, `loads` the read tensors' values and `scalars` the scalar parameters.
    /// Returns the value stored to each written tensor, which the kernel converts to the
    /// tensor's type.
    pub fn evaluate_on_device(
        &self,
        acc: TensorSsa<f32>,
        loads: &HashMap<String, TensorSsa<f32>>,
        scalars: &HashMap<String, Value<f32>>,
    ) -> HashMap<String, TensorSsa<f32>> {
        let config = EfcConfig {
            efc: self,
            state: State::Device { acc: acc.clone(), loads, scalars, stores: RefCell::new(HashMap::new()) },
        };
        (self.epilogue)(&config);
        let State::Device { stores, .. } = config.state else { unreachable!() };
        stores
            .into_inner()
            .into_iter()
            .map(|(name, v)| {
                let t = match v {
                    EfcValue::Tensor(t) => t,
                    EfcValue::Scalar(s) => acc.map(|_| s),
                    EfcValue::Const(c) => acc.map(|_| Value::constant(c as f32)),
                    _ => panic!("EFC: `{name}` stores a value of another phase"),
                };
                (name, t)
            })
            .collect()
    }

    /// The epilogue on the CPU (the Python's `evaluate_on_cpu`): `acc` the product `A B^T`
    /// in the tensors' layout; the written tensors of `args` are overwritten (rounded to
    /// their element types).
    pub fn evaluate_on_host(&self, acc: &[f64], args: &mut HostArgs) {
        let config = EfcConfig { efc: self, state: State::Host { acc, args: RefCell::new(args) } };
        (self.epilogue)(&config);
    }
}

/// The arguments of the host evaluation: scalars, and tensors as `f64` values.
#[derive(Clone, Debug, Default)]
pub struct HostArgs {
    pub scalars: HashMap<String, f64>,
    pub tensors: HashMap<String, Vec<f64>>,
}

/// `x` rounded to `dtype`, as stored in a tensor of that type (through Float32, as the
/// kernel converts its Float32 values).
pub fn round_to(dtype: DType, x: f64) -> f64 {
    let x = x as f32;
    match dtype {
        DType::F16 => F16::from_f32(x).to_f32() as f64,
        DType::BF16 => BF16::from_f32(x).to_f32() as f64,
        DType::F32 => x as f64,
        _ => panic!("EFC: tensors of {dtype} are not supported"),
    }
}

enum State<'a> {
    Analysis(RefCell<Vec<ParameterAttributes>>),
    Device {
        acc: TensorSsa<f32>,
        loads: &'a HashMap<String, TensorSsa<f32>>,
        scalars: &'a HashMap<String, Value<f32>>,
        stores: RefCell<HashMap<String, EfcValue>>,
    },
    Host {
        acc: &'a [f64],
        args: RefCell<&'a mut HostArgs>,
    },
}

/// The epilogue's view of EFC, its `efc_config` argument (the Python's `Configuration`).
pub struct EfcConfig<'a> {
    efc: &'a Efc,
    state: State<'a>,
}

impl EfcConfig<'_> {
    pub fn phase(&self) -> Phase {
        match self.state {
            State::Analysis(_) => Phase::ParameterAnalysis,
            State::Device { .. } => Phase::ThreadOperation,
            State::Host { .. } => Phase::HostEvaluation,
        }
    }

    #[track_caller]
    fn check(&self, name: &str, tensor: bool) -> usize {
        let i = self.efc.index(name);
        let is_tensor = matches!(self.efc.params[i], Param::Tensor { .. });
        assert!(is_tensor == tensor, "EFC: `{name}` is not a {} parameter", if tensor { "tensor" } else { "scalar" });
        i
    }

    /// The accumulator subtile.
    pub fn accum(&self) -> EfcValue {
        match &self.state {
            State::Analysis(_) => EfcValue::Marker,
            State::Device { acc, .. } => EfcValue::Tensor(acc.clone()),
            State::Host { acc, .. } => EfcValue::Host(acc.to_vec()),
        }
    }

    /// The scalar parameter `name`.
    #[track_caller]
    pub fn scalar(&self, name: &str) -> EfcValue {
        self.check(name, false);
        match &self.state {
            State::Analysis(_) => EfcValue::Marker,
            State::Device { scalars, .. } => EfcValue::Scalar(scalars[name]),
            State::Host { args, .. } => EfcValue::HostScalar(
                *args.borrow().scalars.get(name).unwrap_or_else(|| panic!("EFC: no host value for `{name}`")),
            ),
        }
    }

    /// Read tensor `name` (its subtile on the device).
    #[track_caller]
    pub fn load(&self, name: &str) -> EfcValue {
        let i = self.check(name, true);
        match &self.state {
            State::Analysis(a) => {
                a.borrow_mut()[i].is_read = true;
                EfcValue::Marker
            }
            State::Device { loads, .. } => EfcValue::Tensor(loads[name].clone()),
            State::Host { args, .. } => EfcValue::Host(
                args.borrow().tensors.get(name).unwrap_or_else(|| panic!("EFC: no host tensor `{name}`")).clone(),
            ),
        }
    }

    /// Write `value` to tensor `name`.
    #[track_caller]
    pub fn store(&self, name: &str, value: impl Into<EfcValue>) {
        let i = self.check(name, true);
        let value = value.into();
        match &self.state {
            State::Analysis(a) => a.borrow_mut()[i].is_written = true,
            State::Device { stores, .. } => {
                stores.borrow_mut().insert(name.to_string(), value);
            }
            State::Host { acc, args } => {
                let (dtype, _) = self.efc.tensor_param(name);
                let values = match value {
                    EfcValue::Host(v) => v,
                    EfcValue::HostScalar(c) | EfcValue::Const(c) => vec![c; acc.len()],
                    _ => panic!("EFC: `{name}` stores a value of another phase"),
                };
                assert_eq!(values.len(), acc.len(), "EFC: `{name}` stores a value of another size");
                let values = values.into_iter().map(|x| round_to(dtype, x)).collect();
                args.borrow_mut().tensors.insert(name.to_string(), values);
            }
        }
    }

    /// Element-wise maximum.
    pub fn maximum(&self, x: &EfcValue, y: &EfcValue) -> EfcValue {
        x.maximum(y)
    }

    /// Element-wise minimum.
    pub fn minimum(&self, x: &EfcValue, y: &EfcValue) -> EfcValue {
        x.minimum(y)
    }

    /// `cond ? x : y` element by element (CuTe DSL's `cute.where`, `torch.where`).
    pub fn where_(&self, cond: &EfcValue, x: impl Into<EfcValue>, y: impl Into<EfcValue>) -> EfcValue {
        EfcValue::where_(cond, &x.into(), &y.into())
    }

    /// A value of `x`'s shape filled with `fill`.
    pub fn full_like(&self, x: &EfcValue, fill: impl Into<EfcValue>) -> EfcValue {
        x.full_like(&fill.into())
    }

    /// The activation function `name` of [`ACTIVATION_FUNCTIONS`] (the Python's
    /// `getattr(efc_config, name)`); `slope` is leaky ReLU's negative slope.
    #[track_caller]
    pub fn activation(&self, name: &str, x: &EfcValue, slope: impl Into<EfcValue>) -> EfcValue {
        match name {
            "identity" => x.identity(),
            "relu" => x.relu(),
            "leaky_relu" => x.leaky_relu(slope),
            "tanh" => x.tanh(),
            "sigmoid" => x.sigmoid(),
            "silu" => x.silu(),
            "hardswish" => x.hardswish(),
            "gelu" => x.gelu(),
            _ => panic!("EFC: no activation function `{name}`"),
        }
    }
}

/// A value of the epilogue, in one of the phases.
#[derive(Clone)]
pub enum EfcValue {
    /// Any value during the analysis.
    Marker,
    /// A host number, taking the phase of the other operand.
    Const(f64),
    /// A device scalar.
    Scalar(Value<f32>),
    /// A device subtile.
    Tensor(TensorSsa<f32>),
    /// A device comparison.
    Mask(TensorSsa<bool>),
    HostScalar(f64),
    Host(Vec<f64>),
    HostMask(Vec<bool>),
}

impl From<f64> for EfcValue {
    fn from(x: f64) -> EfcValue {
        EfcValue::Const(x)
    }
}

impl From<&EfcValue> for EfcValue {
    fn from(x: &EfcValue) -> EfcValue {
        x.clone()
    }
}

fn zip_ssa<T: Numeric>(
    x: &TensorSsa<f32>,
    y: &TensorSsa<f32>,
    f: impl Fn(Value<f32>, Value<f32>) -> Value<T>,
) -> TensorSsa<T> {
    assert_eq!(x.shape(), y.shape(), "EFC: subtiles of different shapes");
    TensorSsa::new(x.shape().clone(), x.values().iter().zip(y.values()).map(|(&a, &b)| f(a, b)).collect())
}

fn zip_host<T>(x: &[f64], y: &[f64], f: impl Fn(f64, f64) -> T) -> Vec<T> {
    assert_eq!(x.len(), y.len(), "EFC: host tensors of different sizes");
    x.iter().zip(y).map(|(&a, &b)| f(a, b)).collect()
}

impl EfcValue {
    /// `self` with constants made device or host values when the other side is one.
    fn binary(
        &self,
        other: &EfcValue,
        dev: impl Fn(Value<f32>, Value<f32>) -> Value<f32>,
        host: impl Fn(f64, f64) -> f64,
    ) -> EfcValue {
        use EfcValue::*;
        let c = |x: f64| Value::constant(x as f32);
        match (self, other) {
            (Marker, _) | (_, Marker) => Marker,
            (Const(x), Const(y)) => Const(host(*x, *y)),
            (Scalar(x), Scalar(y)) => Scalar(dev(*x, *y)),
            (Scalar(x), Const(y)) => Scalar(dev(*x, c(*y))),
            (Const(x), Scalar(y)) => Scalar(dev(c(*x), *y)),
            (Tensor(x), Scalar(y)) => Tensor(x.map(|v| dev(v, *y))),
            (Tensor(x), Const(y)) => Tensor(x.map(|v| dev(v, c(*y)))),
            (Scalar(x), Tensor(y)) => Tensor(y.map(|v| dev(*x, v))),
            (Const(x), Tensor(y)) => Tensor(y.map(|v| dev(c(*x), v))),
            (Tensor(x), Tensor(y)) => Tensor(zip_ssa(x, y, dev)),
            (HostScalar(x) | Const(x), HostScalar(y)) | (HostScalar(x), Const(y)) => HostScalar(host(*x, *y)),
            (Host(x), HostScalar(y) | Const(y)) => Host(x.iter().map(|&v| host(v, *y)).collect()),
            (HostScalar(x) | Const(x), Host(y)) => Host(y.iter().map(|&v| host(*x, v)).collect()),
            (Host(x), Host(y)) => Host(zip_host(x, y, host)),
            _ => panic!("EFC: an operation mixes values of different phases or a comparison"),
        }
    }

    /// An element-wise comparison: a mask.
    fn compare(
        &self,
        other: &EfcValue,
        dev: fn(Value<f32>, Value<f32>) -> Boolean,
        host: fn(f64, f64) -> bool,
    ) -> Self {
        use EfcValue::*;
        let c = |x: f64| Value::constant(x as f32);
        match (self, other) {
            (Marker, _) | (_, Marker) => Marker,
            (Tensor(x), Scalar(y)) => Mask(x.map(|v| dev(v, *y))),
            (Tensor(x), Const(y)) => Mask(x.map(|v| dev(v, c(*y)))),
            (Scalar(x), Tensor(y)) => Mask(y.map(|v| dev(*x, v))),
            (Const(x), Tensor(y)) => Mask(y.map(|v| dev(c(*x), v))),
            (Tensor(x), Tensor(y)) => Mask(zip_ssa(x, y, dev)),
            (Host(x), HostScalar(y) | Const(y)) => HostMask(x.iter().map(|&v| host(v, *y)).collect()),
            (HostScalar(x) | Const(x), Host(y)) => HostMask(y.iter().map(|&v| host(*x, v)).collect()),
            (Host(x), Host(y)) => HostMask(zip_host(x, y, host)),
            _ => panic!("EFC: a comparison needs a tensor operand"),
        }
    }

    pub fn lt(&self, other: impl Into<EfcValue>) -> EfcValue {
        self.compare(&other.into(), |a, b| a.lt(b), |a, b| a < b)
    }

    pub fn gt(&self, other: impl Into<EfcValue>) -> EfcValue {
        self.compare(&other.into(), |a, b| a.gt(b), |a, b| a > b)
    }

    /// `cond ? x : y` element by element.
    pub fn where_(cond: &EfcValue, x: &EfcValue, y: &EfcValue) -> EfcValue {
        use EfcValue::*;
        let dev = |v: &EfcValue, i: usize| match v {
            Tensor(t) => t.values()[i],
            Scalar(s) => *s,
            Const(c) => Value::constant(*c as f32),
            _ => panic!("EFC: where_ mixes phases"),
        };
        let host = |v: &EfcValue, i: usize| match v {
            Host(t) => t[i],
            HostScalar(c) | Const(c) => *c,
            _ => panic!("EFC: where_ mixes phases"),
        };
        match cond {
            Marker => Marker,
            Mask(m) => Tensor(TensorSsa::new(
                m.shape().clone(),
                m.values().iter().enumerate().map(|(i, &c)| select(c, dev(x, i), dev(y, i))).collect(),
            )),
            HostMask(m) => Host(m.iter().enumerate().map(|(i, &c)| if c { host(x, i) } else { host(y, i) }).collect()),
            _ => panic!("EFC: where_ needs a comparison as its condition"),
        }
    }

    /// A value of this one's shape filled with `fill`.
    pub fn full_like(&self, fill: &EfcValue) -> EfcValue {
        use EfcValue::*;
        match (self, fill) {
            (Marker, _) | (_, Marker) => Marker,
            (Tensor(t), Scalar(s)) => Tensor(t.map(|_| *s)),
            (Tensor(t), Const(c)) => Tensor(t.map(|_| Value::constant(*c as f32))),
            (Scalar(_), Scalar(s)) => Scalar(*s),
            (Scalar(_), Const(c)) => Scalar(Value::constant(*c as f32)),
            (Host(t), HostScalar(c) | Const(c)) => Host(vec![*c; t.len()]),
            (HostScalar(_) | Const(_), HostScalar(c) | Const(c)) => HostScalar(*c),
            _ => panic!("EFC: full_like mixes phases"),
        }
    }

    /// `f` element by element on the host, `dev` on the device.
    fn unary(&self, dev: impl Fn(&EfcValue) -> EfcValue, host: impl Fn(f64) -> f64) -> EfcValue {
        match self {
            EfcValue::Host(v) => EfcValue::Host(v.iter().map(|&x| host(x)).collect()),
            EfcValue::HostScalar(x) => EfcValue::HostScalar(host(*x)),
            EfcValue::Const(x) => EfcValue::Const(host(*x)),
            _ => dev(self),
        }
    }

    /// `where(x > y, x, y)` (the Python's `EFC.maximum`; `torch.maximum` on the host).
    pub fn maximum(&self, other: &EfcValue) -> EfcValue {
        self.binary(other, |a, b| select(a.gt(b), a, b), |a, b| if a > b || a.is_nan() { a } else { b })
    }

    /// `where(x < y, x, y)`.
    pub fn minimum(&self, other: &EfcValue) -> EfcValue {
        self.binary(other, |a, b| select(a.lt(b), a, b), |a, b| if a < b || a.is_nan() { a } else { b })
    }

    pub fn exp(&self) -> EfcValue {
        self.unary(|x| x.map_device(|v| cute::math::exp(v, false)), f64::exp)
    }

    fn map_device(&self, f: impl Fn(Value<f32>) -> Value<f32>) -> EfcValue {
        match self {
            EfcValue::Marker => EfcValue::Marker,
            EfcValue::Tensor(t) => EfcValue::Tensor(t.map(f)),
            EfcValue::Scalar(s) => EfcValue::Scalar(f(*s)),
            EfcValue::Const(c) => EfcValue::Scalar(f(Value::constant(*c as f32))),
            _ => panic!("EFC: a device function on a host value"),
        }
    }

    // The activation functions.

    pub fn identity(&self) -> EfcValue {
        self.clone()
    }

    /// `max(x, 0)`.
    pub fn relu(&self) -> EfcValue {
        self.unary(|x| x.maximum(&x.full_like(&0.0.into())), |x| x.max(0.0))
    }

    /// `max(x, 0) + slope * min(x, 0)`.
    pub fn leaky_relu(&self, slope: impl Into<EfcValue>) -> EfcValue {
        let slope = slope.into();
        let zero = self.full_like(&0.0.into());
        self.maximum(&zero) + self.minimum(&zero) * self.full_like(&slope)
    }

    pub fn tanh(&self) -> EfcValue {
        self.unary(|x| x.map_device(|v| cute::math::tanh(v, false)), f64::tanh)
    }

    /// `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self) -> EfcValue {
        self.unary(|x| x.full_like(&1.0.into()) / (x.full_like(&1.0.into()) + (-x).exp()), |x| 1.0 / (1.0 + (-x).exp()))
    }

    /// `x * sigmoid(x)`.
    pub fn silu(&self) -> EfcValue {
        self * &self.sigmoid()
    }

    /// `x * relu6(x + 3) / 6`.
    pub fn hardswish(&self) -> EfcValue {
        self.unary(
            |x| {
                let relu6 = (x + &x.full_like(&3.0.into()))
                    .maximum(&x.full_like(&0.0.into()))
                    .minimum(&x.full_like(&6.0.into()));
                x * &relu6 / x.full_like(&6.0.into())
            },
            |x| x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
        )
    }

    /// GELU: the tanh approximation on the device (the Python's), the exact
    /// `x/2 (1 + erf(x / sqrt 2))` on the host (PyTorch's default).
    pub fn gelu(&self) -> EfcValue {
        self.unary(
            |x| {
                let k = |c: f64| x.full_like(&c.into());
                let inner = k(0.7978845608028654) * (x + &(k(0.044715) * x * x * x));
                k(0.5) * x * (k(1.0) + inner.tanh())
            },
            |x| 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)),
        )
    }
}

/// `erf(x)` to about 1e-7 (Abramowitz and Stegun 7.1.26), for the host GELU.
pub fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

macro_rules! efc_binop {
    ($($trait:ident $method:ident $op:tt),*) => {$(
        impl $trait<&EfcValue> for &EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: &EfcValue) -> EfcValue {
                self.binary(rhs, |a, b| a $op b, |a, b| a $op b)
            }
        }
        impl $trait<EfcValue> for EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: EfcValue) -> EfcValue {
                (&self).$method(&rhs)
            }
        }
        impl $trait<&EfcValue> for EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: &EfcValue) -> EfcValue {
                (&self).$method(rhs)
            }
        }
        impl $trait<EfcValue> for &EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: EfcValue) -> EfcValue {
                self.$method(&rhs)
            }
        }
        impl $trait<f64> for EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: f64) -> EfcValue {
                (&self).$method(&EfcValue::Const(rhs))
            }
        }
        impl $trait<f64> for &EfcValue {
            type Output = EfcValue;
            fn $method(self, rhs: f64) -> EfcValue {
                self.$method(&EfcValue::Const(rhs))
            }
        }
        impl $trait<EfcValue> for f64 {
            type Output = EfcValue;
            fn $method(self, rhs: EfcValue) -> EfcValue {
                (&EfcValue::Const(self)).$method(&rhs)
            }
        }
        impl $trait<&EfcValue> for f64 {
            type Output = EfcValue;
            fn $method(self, rhs: &EfcValue) -> EfcValue {
                (&EfcValue::Const(self)).$method(rhs)
            }
        }
    )*};
}

efc_binop!(Add add +, Sub sub -, Mul mul *, Div div /);

impl Neg for &EfcValue {
    type Output = EfcValue;
    fn neg(self) -> EfcValue {
        self.unary(|x| x.map_device(|v| -v), |x| -x)
    }
}

impl Neg for EfcValue {
    type Output = EfcValue;
    fn neg(self) -> EfcValue {
        -&self
    }
}

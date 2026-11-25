// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   media/docs/pythonDSL/guides/framework_integration.rst (the rules and
//   error messages of mark_layout_dynamic and mark_compact_shape_dynamic),
//     Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The ported portions are under the BSD-3-Clause licence.
// THIRD-PARTY-NOTICES.md has the licence text and lists what each part ports.

//! Tensor parameters: [`TensorSpec`] (CuTe DSL's `cute.Tensor[dtype, shape,
//! stride]`, a `TypedTensor`), the host tensor a launch passes
//! ([`HostTensor`]) with CuTe DSL's `mark_layout_dynamic` and
//! `mark_compact_shape_dynamic`, and the fake-tensor constructors.

use std::fmt;

use super::symint::{SpecLeaf, SpecTuple, SymInt, leaf_param_name, leaves_with_paths, python_tuple};
use super::{ArgError, ArgPack, KernelParam, buffer_elements, leaf_name};
use crate::builder;
use crate::ir::{BufferDecl, DType, MemSpace};
use crate::layout::{IndexExpr, IntTuple, Layout, Shape, StrideLeaf, Tree};
use crate::sim::KernelArg;
use crate::tensor::{AddressSpace, Engine, Tensor, storage_decl};
use crate::value::Numeric;

/// The description of a tensor parameter (CuTe DSL's `cute.Tensor[dtype,
/// shape, stride]`, a `TypedTensor`): its element type, its shape and
/// stride with static and dynamic ([`SymInt`]) leaves, its memory space and
/// the alignment assumed for its start (by default the element's size, as
/// CuTe DSL's).
///
/// [`param_tensor`](crate::param_tensor) declares a parameter from it (the
/// buffer, then one scalar per dynamic leaf), [`TensorSpec::isinstance`]
/// checks a tensor against it, and a launch checks a [`HostTensor`] against
/// it. It prints as CuTe DSL's does: `Tensor<f32, (16,), (1,)>`. CuTe DSL's
/// `mlir_type` has no counterpart: the kernel's IR carries the parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    dtype: DType,
    shape: SpecTuple,
    stride: SpecTuple,
    memspace: AddressSpace,
    assumed_align: usize,
}

impl TensorSpec {
    /// A spec of `T` elements (CuTe DSL's `cute.Tensor[T, shape, stride]`):
    /// generic memory space, aligned to the element's size.
    #[track_caller]
    pub fn new<T: Numeric>(shape: impl Into<SpecTuple>, stride: impl Into<SpecTuple>) -> TensorSpec {
        TensorSpec::of(T::DTYPE, shape, stride)
    }

    /// [`TensorSpec::new`] with the element type as a value.
    #[track_caller]
    pub fn of(dtype: DType, shape: impl Into<SpecTuple>, stride: impl Into<SpecTuple>) -> TensorSpec {
        let (shape, stride) = (shape.into(), stride.into());
        assert!(shape.congruent(&stride), "cute: TensorSpec: shape {shape} and stride {stride} are not congruent");
        TensorSpec {
            dtype,
            shape,
            stride,
            memspace: AddressSpace::Generic,
            assumed_align: dtype.storage_bytes(1).max(1),
        }
    }

    /// The same spec with the start assumed aligned to `bytes` (a power of
    /// two of at least the element's size).
    #[track_caller]
    pub fn with_assumed_align(mut self, bytes: usize) -> TensorSpec {
        let size = self.dtype.storage_bytes(1).max(1);
        assert!(
            bytes.is_power_of_two() && bytes >= size,
            "cute: TensorSpec: an alignment of {bytes} bytes is not a power of two of at least the {size} bytes of {}",
            self.dtype
        );
        self.assumed_align = bytes;
        self
    }

    /// The same spec in `memspace` (a kernel parameter must be `Generic` or
    /// `Gmem`).
    pub fn with_memspace(mut self, memspace: AddressSpace) -> TensorSpec {
        self.memspace = memspace;
        self
    }

    pub fn element_type(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &SpecTuple {
        &self.shape
    }

    pub fn stride(&self) -> &SpecTuple {
        &self.stride
    }

    pub fn memspace(&self) -> AddressSpace {
        self.memspace
    }

    pub fn assumed_align(&self) -> usize {
        self.assumed_align
    }

    /// `shape:stride`, as CuTe prints a layout: `(?{div=2},4):(1,?{i64 div=2})`.
    pub fn layout_string(&self) -> String {
        format!("{}:{}", self.shape, self.stride)
    }

    /// The layout, if every leaf is static.
    pub fn static_layout(&self) -> Option<Layout> {
        let shape = self.shape.map(&mut |l| l.as_static());
        let stride = self.stride.map(&mut |l| l.as_static());
        if shape.any_leaf(&mut |l| l.is_none()) || stride.any_leaf(&mut |l| l.is_none()) {
            return None;
        }
        let shape: IntTuple = shape.map(&mut |l| l.unwrap());
        let stride: IntTuple = stride.map(&mut |l| l.unwrap());
        Some(Layout::new(&shape, &stride))
    }

    /// Whether `t` is a tensor of this type (CuTe DSL's
    /// `TypedTensor.isinstance`): the element type, the static leaves, and a
    /// runtime leaf of the same width with at least the divisibility of each
    /// dynamic one, in global memory (or any memory for a `Generic` spec).
    pub fn isinstance<T: Numeric>(&self, t: &Tensor<T>) -> bool {
        if T::DTYPE != self.dtype || t.swizzle().is_some() || matches!(t.engine, Engine::ArithTuple(_)) {
            return false;
        }
        if self.memspace != AddressSpace::Generic && t.memspace() != self.memspace {
            return false;
        }
        let layout = t.layout();
        let leaf_ok = |s: &SpecLeaf, e: &IndexExpr| match s {
            SpecLeaf::Static(v) => e.as_static() == Some(*v),
            SpecLeaf::Dynamic(sym) => {
                !e.is_static() && e.width() == sym.width() && e.known_divisor().is_multiple_of(sym.divisibility())
            }
        };
        if !self.shape.congruent(layout.shape()) || !self.stride.congruent(layout.stride()) {
            return false;
        }
        let shapes = leaves_with_paths(&self.shape).into_iter().zip(layout.shape().leaves());
        let strides = leaves_with_paths(&self.stride).into_iter().zip(layout.stride().leaves());
        shapes.into_iter().all(|((_, s), e)| leaf_ok(s, e))
            && strides.into_iter().all(|((_, s), e)| match e {
                StrideLeaf::Int(e) => leaf_ok(s, e),
                StrideLeaf::Basis(_) => false,
            })
    }
}

/// `Tensor<f32, (16,), (1,)>`, CuTe DSL's `TypedTensor` text.
impl fmt::Display for TensorSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}, {}, {}>", self.dtype, python_tuple(&self.shape), python_tuple(&self.stride))
    }
}

/// Declare the leaves of `tree` under `prefix`: a static leaf is a constant,
/// a dynamic one its [`SymInt`]'s parameter.
#[track_caller]
pub(crate) fn declare_tree(prefix: &str, tree: &SpecTuple) -> Shape {
    fn walk(prefix: &str, t: &SpecTuple, path: &mut Vec<usize>) -> Shape {
        match t {
            Tree::Leaf(SpecLeaf::Static(v)) => Tree::Leaf(IndexExpr::Static(*v)),
            Tree::Leaf(SpecLeaf::Dynamic(s)) => {
                let name = leaf_param_name(prefix, path);
                let v = builder::with(|b| b.declare_symint(s.id(), &name, s.divisibility(), s.dtype()));
                Tree::Leaf(IndexExpr::Dynamic(v))
            }
            Tree::Branch(c) => Tree::Branch(
                c.iter()
                    .enumerate()
                    .map(|(i, c)| {
                        path.push(i);
                        let r = walk(prefix, c, path);
                        path.pop();
                        r
                    })
                    .collect(),
            ),
        }
    }
    walk(prefix, tree, &mut Vec::new())
}

/// Check the leaves of `host` against `spec`'s (congruent) and pack the
/// dynamic ones, named under `prefix`.
pub(crate) fn pack_tree(
    prefix: &str,
    spec: &SpecTuple,
    host: &IntTuple,
    what: &str,
    out: &mut ArgPack,
) -> Result<(), ArgError> {
    for ((path, s), &h) in leaves_with_paths(spec).into_iter().zip(host.leaves()) {
        let name = leaf_param_name(prefix, &path);
        match s {
            SpecLeaf::Static(v) if *v == h => {}
            SpecLeaf::Static(v) => {
                return Err(ArgError::new(
                    &name,
                    format!("the kernel was built for the static {what} {v}, launched with {h}"),
                ));
            }
            SpecLeaf::Dynamic(sym) => out.sym(&name, *sym, h as u64)?,
        }
    }
    Ok(())
}

/// The arguments of a [`param_tensor`](crate::param_tensor): a [`Layout`]
/// (the buffer is 16-byte aligned) or a [`TensorSpec`].
pub trait TensorParamSource {
    #[track_caller]
    fn declare_tensor<T: Numeric>(self, name: &str) -> Tensor<T>;
}

impl TensorParamSource for Layout {
    #[track_caller]
    fn declare_tensor<T: Numeric>(self, name: &str) -> Tensor<T> {
        crate::tensor::param_tensor_aligned(name, self, 16)
    }
}

impl TensorParamSource for &TensorSpec {
    #[track_caller]
    fn declare_tensor<T: Numeric>(self, name: &str) -> Tensor<T> {
        declare_tensor(name, self)
    }
}

impl TensorParamSource for TensorSpec {
    #[track_caller]
    fn declare_tensor<T: Numeric>(self, name: &str) -> Tensor<T> {
        declare_tensor(name, &self)
    }
}

/// Declare a tensor parameter from a spec: the buffer `name`, then the
/// dynamic shape leaves (`{name}_shape_{path}`) and stride leaves
/// (`{name}_stride_{path}`), each [`SymInt`] once.
#[track_caller]
pub(crate) fn declare_tensor<T: Numeric>(name: &str, spec: &TensorSpec) -> Tensor<T> {
    let name = leaf_name(name);
    assert!(
        spec.dtype == T::DTYPE,
        "cute: parameter `{name}`: the spec is a tensor of {}, declared as one of {}",
        spec.dtype.cute_name(),
        T::DTYPE.cute_name()
    );
    assert!(
        matches!(spec.memspace, AddressSpace::Generic | AddressSpace::Gmem),
        "cute: parameter `{name}`: a tensor parameter lives in global memory, not {:?}",
        spec.memspace
    );
    let size = T::DTYPE.storage_bytes(1).max(1);
    let align = spec.assumed_align;
    assert!(
        align.is_power_of_two() && align >= size,
        "cute: parameter `{name}`: an alignment of {align} bytes is not a power of two of at least the {size} bytes of {}",
        T::DTYPE
    );
    let len = spec.static_layout().map(|l| storage_decl(T::DTYPE, l.cosize()).1);
    let elem = storage_decl(T::DTYPE, 0).0;
    let buffer = builder::with(|b| {
        b.declare_buffer(BufferDecl {
            name: name.clone(),
            elem,
            len,
            space: MemSpace::Global,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    let shape = declare_tree(&format!("{name}_shape"), &spec.shape);
    let stride = declare_tree(&format!("{name}_stride"), &spec.stride).map(&mut |e| StrideLeaf::Int(e.clone()));
    Tensor::new(Engine::Memory { buffer, space: MemSpace::Global, peer: None }, Layout::new(shape, stride))
}

/// What marking a [`HostTensor`] or making a spec refused, with CuTe DSL's
/// message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecError(pub String);

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpecError {}

fn err<T>(m: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError(m.into()))
}

/// A tensor on the host, as a launch passes it (what CuTe DSL's
/// `from_dlpack` makes of a framework tensor): a buffer, its concrete shape
/// and strides (in elements), and the [`TensorSpec`] the kernel is built for
/// from it, all static until [`mark_layout_dynamic`](Self::mark_layout_dynamic)
/// or [`mark_compact_shape_dynamic`](Self::mark_compact_shape_dynamic) makes
/// modes dynamic. A launch packs the buffer followed by the values of the
/// dynamic leaves.
///
/// Dynamic shapes are `Int32`s; dynamic strides are `Int64`s (as DLPack's
/// are), or `Int32`s with [`with_32bit_stride`](Self::with_32bit_stride).
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    data: KernelArg,
    dtype: DType,
    shape: IntTuple,
    stride: IntTuple,
    spec: TensorSpec,
    use_32bit_stride: bool,
    stride_order: Option<Vec<usize>>,
    assumed_align: Option<usize>,
}

impl HostTensor {
    /// `data` (a buffer argument) seen through the static `layout`.
    #[track_caller]
    pub fn new(data: KernelArg, layout: Layout) -> HostTensor {
        let (dtype, _) =
            buffer_elements(&data).unwrap_or_else(|| panic!("cute: HostTensor: {} is not a buffer", data.describe()));
        let (shape, stride) = match (layout.shape().as_static(), static_stride(&layout)) {
            (Some(s), Some(d)) => (s, d),
            _ => panic!("cute: HostTensor: the layout {layout} of a host tensor must be static"),
        };
        let spec = TensorSpec {
            dtype,
            shape: (&shape).into(),
            stride: (&stride).into(),
            memspace: AddressSpace::Gmem,
            assumed_align: dtype.storage_bytes(1).max(1),
        };
        HostTensor {
            data,
            dtype,
            shape,
            stride,
            spec,
            use_32bit_stride: false,
            stride_order: None,
            assumed_align: None,
        }
    }

    /// `values` seen through the static `layout`.
    #[track_caller]
    pub fn from_slice<T: Numeric>(values: &[T], layout: Layout) -> HostTensor {
        HostTensor::new(KernelArg::buffer(values), layout)
    }

    /// The same tensor with 32-bit dynamic strides (CuTe DSL's
    /// `use_32bit_stride=True`), whose layout must then fit in an `Int32`.
    pub fn with_32bit_stride(mut self, yes: bool) -> HostTensor {
        self.use_32bit_stride = yes;
        self
    }

    /// The same tensor promising only that its start is a multiple of
    /// `bytes` (a view into a larger allocation, say; CuTe DSL's
    /// `assumed_align`). A kernel that assumes more is refused at launch.
    /// Without it, the buffer is an allocation of its own (256-byte aligned
    /// on the device) and the spec assumes the element's size.
    pub fn with_assumed_align(mut self, bytes: usize) -> HostTensor {
        self.assumed_align = Some(bytes);
        self.spec.assumed_align = bytes;
        self
    }

    pub fn data(&self) -> &KernelArg {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut KernelArg {
        &mut self.data
    }

    pub fn into_data(self) -> KernelArg {
        self.data
    }

    /// The contents as `T` (see [`KernelArg::to_vec`]).
    pub fn to_vec<T: Numeric>(&self) -> Vec<T> {
        self.data.to_vec()
    }

    pub fn element_type(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &IntTuple {
        &self.shape
    }

    pub fn stride(&self) -> &IntTuple {
        &self.stride
    }

    pub fn layout(&self) -> Layout {
        Layout::new(&self.shape, &self.stride)
    }

    /// The spec a kernel built from this tensor takes.
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    /// The spec's layout, `shape:stride`: `(?,?,?,?):(?{i64},?{i64},?{i64},1)`.
    pub fn layout_string(&self) -> String {
        self.spec.layout_string()
    }

    /// The same tensor with its elements read as `dtype` (the shape and strides unchanged),
    /// with the data replaced by `data`.
    pub(crate) fn relabel(mut self, dtype: DType, data: KernelArg) -> HostTensor {
        self.dtype = dtype;
        self.spec.dtype = dtype;
        let size = dtype.storage_bytes(1).max(1);
        if self.assumed_align.is_none() || self.spec.assumed_align < size {
            self.spec.assumed_align = self.spec.assumed_align.max(size);
        }
        self.data = data;
        self
    }

    /// The stride order recorded by [`mark_compact_shape_dynamic`](Self::mark_compact_shape_dynamic).
    pub fn stride_order(&self) -> Option<&[usize]> {
        self.stride_order.as_deref()
    }

    fn flat_modes(&self, what: &str) -> Result<(Vec<usize>, Vec<usize>), SpecError> {
        if self.shape.depth() > 1 {
            return err(format!("{what} supports flat layouts only, not {}", self.layout()));
        }
        Ok((self.shape.leaves().into_iter().copied().collect(), self.stride.leaves().into_iter().copied().collect()))
    }

    fn stride_width(&self) -> u32 {
        if self.use_32bit_stride { 32 } else { 64 }
    }

    /// With 32-bit strides, the layout must fit in an `Int32`; shapes always
    /// must.
    fn check_widths(&self, shape: &[usize], stride: &[usize]) -> Result<(), SpecError> {
        let span: u128 = if shape.contains(&0) {
            0
        } else {
            1 + shape.iter().zip(stride).map(|(&s, &d)| (s as u128 - 1) * d as u128).sum::<u128>()
        };
        if self.use_32bit_stride && (span > i32::MAX as u128 || stride.iter().any(|&d| d > i32::MAX as usize)) {
            return err("Layout in HostTensor has int32 overflow risk. Please set use_32bit_stride to false.");
        }
        if let Some((i, s)) = shape.iter().enumerate().find(|&(_, &s)| s > i32::MAX as usize) {
            return err(format!("The shape({s}) of mode({i}) does not fit in an Int32"));
        }
        Ok(())
    }

    /// Make every mode of the layout dynamic (CuTe DSL's
    /// `mark_layout_dynamic(leading_dim)`): the shapes, and the strides
    /// except the leading dimension's (which stays 1) and zero (broadcast)
    /// strides. Without `leading_dim` it is the one mode of stride 1, or the
    /// one of stride 1 and size above 1 among several of stride 1; with no
    /// mode of stride 1 every stride is dynamic. A negative `leading_dim`
    /// counts from the end. Flat layouts only.
    pub fn mark_layout_dynamic(mut self, leading_dim: Option<i32>) -> Result<HostTensor, SpecError> {
        let (shape, stride) = self.flat_modes("mark_layout_dynamic")?;
        let rank = shape.len();
        let leading = match leading_dim {
            Some(d) => {
                let i = if d < 0 { rank as i64 + d as i64 } else { d as i64 };
                if !(0..rank as i64).contains(&i) {
                    return err(format!("Expected leading_dim to be in range [-{rank}, {rank}), but got {d}"));
                }
                let i = i as usize;
                if stride[i] != 1 {
                    return err(format!("Expected strides[leading_dim] == 1, but got {}", stride[i]));
                }
                Some(i)
            }
            None => {
                let ones: Vec<usize> = (0..rank).filter(|&i| stride[i] == 1).collect();
                match ones.len() {
                    0 => None,
                    1 => Some(ones[0]),
                    _ => {
                        let big: Vec<usize> = ones.iter().copied().filter(|&i| shape[i] > 1).collect();
                        if big.len() == 1 {
                            Some(big[0])
                        } else {
                            let which = if big.is_empty() { "none of them has" } else { "more than one of them has" };
                            return err(format!(
                                "Can't deduce the leading dimension from layout: several modes have stride 1 and \
                                 {which} size > 1; specify leading_dim"
                            ));
                        }
                    }
                }
            }
        };
        self.check_widths(&shape, &stride)?;
        let width = self.stride_width();
        let shape_spec: Vec<SpecLeaf> = (0..rank).map(|_| SpecLeaf::Dynamic(SymInt::new(32, 1, None))).collect();
        let stride_spec: Vec<SpecLeaf> = (0..rank)
            .map(|i| {
                if Some(i) == leading {
                    SpecLeaf::Static(1)
                } else if stride[i] == 0 {
                    SpecLeaf::Static(0)
                } else {
                    SpecLeaf::Dynamic(SymInt::new(width, 1, None))
                }
            })
            .collect();
        self.spec.shape = Tree::unflatten(&shape_spec, &self.shape);
        self.spec.stride = Tree::unflatten(&stride_spec, &self.stride);
        Ok(self)
    }

    /// Make mode `mode`'s extent dynamic with `divisibility`, for a compact
    /// layout (CuTe DSL's `mark_compact_shape_dynamic`): the strides of the
    /// modes outside it in the compact order become dynamic, with the
    /// divisibility of the product under them, and size-1 modes get stride
    /// 0. `stride_order` lists the modes outermost first (as PyTorch's
    /// `dim_order()`); without it the order is the last one used, or deduced
    /// from the strides (descending) when exactly one stride is 1. Flat
    /// layouts only.
    pub fn mark_compact_shape_dynamic(
        mut self,
        mode: usize,
        stride_order: Option<&[usize]>,
        divisibility: usize,
    ) -> Result<HostTensor, SpecError> {
        let (shape, stride) = self.flat_modes("mark_compact_shape_dynamic")?;
        let rank = shape.len();
        if mode >= rank {
            return err(format!("Expected mode value to be in range [0, {rank}), but got {mode}"));
        }
        assert!(divisibility > 0, "cute: mark_compact_shape_dynamic: the divisibility must be positive");
        let canonical: Vec<usize> = (0..rank).map(|i| if shape[i] == 1 { 0 } else { stride[i] }).collect();
        let compact_for = |order: &[usize]| -> Vec<usize> {
            let mut out = vec![0; rank];
            let mut running = 1;
            for &m in order.iter().rev() {
                out[m] = if shape[m] == 1 { 0 } else { running };
                running *= shape[m];
            }
            out
        };
        if let Some(given) = stride_order {
            if given.len() != rank {
                return err(format!("Expected stride_order to have {rank} elements, but got {}.", given.len()));
            }
            if let Some(d) = (0..rank).find(|d| !given.contains(d)) {
                return err(format!(
                    "Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain {d}."
                ));
            }
        }
        let order: Vec<usize> = if let Some(last) = &self.stride_order {
            if stride_order.is_some_and(|g| g != last.as_slice()) {
                return err("The stride_order is not consistent with the last stride_order");
            }
            last.clone()
        } else {
            let deduced = (stride.iter().filter(|&&d| d == 1).count() == 1).then(|| {
                let mut modes: Vec<usize> = (0..rank).collect();
                modes.sort_by(|&a, &b| stride[b].cmp(&stride[a]).then(a.cmp(&b)));
                modes
            });
            match (stride_order, deduced) {
                (Some(g), Some(d)) if g != d.as_slice() => {
                    return err("The stride_order is not consistent with the deduced stride_order");
                }
                (Some(g), _) => g.to_vec(),
                (None, Some(d)) => d,
                (None, None) => {
                    return err("The layout could not be deduced, please specify the stride_order explicitly");
                }
            }
        };
        if compact_for(&order) != canonical {
            return err("The stride_order is not consistent with the layout");
        }
        if !shape[mode].is_multiple_of(divisibility) {
            return err(format!(
                "The shape({}) of mode({mode}) is not divisible by the divisibility({divisibility})",
                shape[mode]
            ));
        }
        self.check_widths(&shape, &canonical)?;
        let width = self.stride_width();
        let mut shape_spec: Vec<SpecLeaf> = self.spec.shape.leaves().into_iter().copied().collect();
        shape_spec[mode] = SpecLeaf::Dynamic(SymInt::new(32, divisibility, None));
        let mut stride_spec = vec![SpecLeaf::Static(0); rank];
        // The running product under each mode: static, or dynamic with a
        // known divisibility.
        let (mut value, mut div, mut dynamic) = (1usize, 1usize, false);
        for &m in order.iter().rev() {
            let unit = matches!(shape_spec[m], SpecLeaf::Static(1));
            stride_spec[m] = if unit {
                SpecLeaf::Static(0)
            } else if dynamic {
                SpecLeaf::Dynamic(SymInt::new(width, div, None))
            } else {
                SpecLeaf::Static(value)
            };
            match shape_spec[m] {
                SpecLeaf::Static(v) => {
                    value *= v;
                    div *= v;
                }
                SpecLeaf::Dynamic(s) => {
                    dynamic = true;
                    div *= s.divisibility();
                }
            }
        }
        // The host's strides: the spec's static ones (a dynamic mode of
        // extent 1 keeps its compact stride, not 0, as the spec does), and
        // the canonical stride under each dynamic one.
        let host_stride: Vec<usize> = (0..rank).map(|m| stride_spec[m].as_static().unwrap_or(canonical[m])).collect();
        self.stride = Tree::unflatten(&host_stride, &self.shape);
        self.spec.shape = Tree::unflatten(&shape_spec, &self.shape);
        self.spec.stride = Tree::unflatten(&stride_spec, &self.shape);
        self.stride_order = Some(order);
        Ok(self)
    }
}

/// The spec's text: `Tensor<f32, (?,4), (1,?{i64})>`.
impl fmt::Display for HostTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.spec.fmt(f)
    }
}

/// The strides of `layout`, if they are static integers.
pub(crate) fn static_stride(layout: &Layout) -> Option<IntTuple> {
    let t = layout.stride().map(&mut |l| match l {
        StrideLeaf::Int(e) => e.as_static(),
        StrideLeaf::Basis(_) => None,
    });
    if t.any_leaf(&mut |l| l.is_none()) {
        return None;
    }
    Some(t.map(&mut |l| l.unwrap()))
}

/// The number of elements a layout reaches: one past its largest offset.
fn span(shape: &IntTuple, stride: &IntTuple) -> usize {
    let (s, d) = (shape.leaves(), stride.leaves());
    if s.iter().any(|&&x| x == 0) {
        return 0;
    }
    1 + s.iter().zip(d).map(|(&&s, &d)| (s - 1) * d).sum::<usize>()
}

/// Check `host` against `spec` and pack it for the tensor parameter `name`.
pub(crate) fn pack_tensor(name: &str, spec: &TensorSpec, host: &HostTensor, out: &mut ArgPack) -> Result<(), ArgError> {
    let have = buffer_elements(&host.data).map_or(0, |(_, n)| n);
    pack_tensor_with(name, spec, host, host.data.clone(), have, out)
}

/// [`pack_tensor`] for a tensor whose memory is elsewhere (device memory passed by address):
/// `placeholder` goes in the buffer's place, and `have` elements are reachable.
pub(crate) fn pack_tensor_with(
    name: &str,
    spec: &TensorSpec,
    host: &HostTensor,
    placeholder: KernelArg,
    have: usize,
    out: &mut ArgPack,
) -> Result<(), ArgError> {
    let name = leaf_name(name);
    if host.dtype != spec.dtype {
        return Err(ArgError::new(
            &name,
            format!("expected a tensor of {}, got one of {}", spec.dtype.cute_name(), host.dtype.cute_name()),
        ));
    }
    if !host.shape.congruent(&spec.shape) || !host.stride.congruent(&spec.stride) {
        return Err(ArgError::new(
            &name,
            format!(
                "the kernel takes a tensor of layout {}, got one of layout {}",
                spec.layout_string(),
                host.layout()
            ),
        ));
    }
    if let Some(a) = host.assumed_align
        && a < spec.assumed_align
    {
        return Err(ArgError::new(
            &name,
            format!(
                "the kernel assumes a {}-byte-aligned start, but the host tensor promises only {a} bytes",
                spec.assumed_align
            ),
        ));
    }
    out.arg(&name, placeholder);
    pack_tree(&format!("{name}_shape"), &spec.shape, &host.shape, "extent", out)?;
    // A mode of extent 1 never moves the offset, so a static stride takes
    // whatever stride the host gives it (`mark_compact_shape_dynamic` makes
    // it 0): a kernel built for a K-major `(m, k)` with stride `(?, 1)`
    // also runs with `k = 1`.
    let strides: Vec<usize> = leaves_with_paths(&spec.stride)
        .into_iter()
        .zip(host.shape.leaves().into_iter().zip(host.stride.leaves()))
        .map(|((_, s), (&e, &h))| match s {
            SpecLeaf::Static(v) if e == 1 => *v,
            _ => h,
        })
        .collect();
    let stride = Tree::unflatten(&strides, &host.stride);
    pack_tree(&format!("{name}_stride"), &spec.stride, &stride, "stride", out)?;
    let need = span(&host.shape, &host.stride);
    if have < need {
        return Err(ArgError::new(
            &name,
            format!("a tensor of layout {} needs {need} elements, the buffer has {have}", host.layout()),
        ));
    }
    Ok(())
}

impl<T: Numeric> KernelParam for Tensor<T> {
    type Spec = TensorSpec;
    type Host = HostTensor;

    #[track_caller]
    fn declare(name: &str, spec: &TensorSpec) -> Self {
        declare_tensor(name, spec)
    }

    fn spec_of(host: &HostTensor) -> TensorSpec {
        host.spec.clone()
    }

    fn pack(name: &str, spec: &TensorSpec, host: &HostTensor, out: &mut ArgPack) -> Result<(), ArgError> {
        pack_tensor(name, spec, host, out)
    }

    fn write_back(_: &TensorSpec, host: &mut HostTensor, buffers: &mut dyn Iterator<Item = KernelArg>) {
        host.data = buffers.next().expect("cute: write_back: too few buffers");
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        (self.layout() != other.layout())
            .then(|| format!("a tensor of layout {:?} became one of {:?}", self.layout(), other.layout()))
    }
}

/// A spec for a tensor with an explicit layout (CuTe DSL's
/// `runtime.make_fake_tensor`): each leaf of `shape` and `stride` a static
/// integer or a [`SymInt`]; global memory, aligned to the element's size.
/// Reuse one `SymInt` wherever the same runtime value appears.
#[track_caller]
pub fn make_fake_tensor<T: Numeric>(shape: impl Into<SpecTuple>, stride: impl Into<SpecTuple>) -> TensorSpec {
    TensorSpec::new::<T>(shape, stride).with_memspace(AddressSpace::Gmem)
}

/// A spec for a compact tensor (CuTe DSL's `runtime.make_fake_compact_tensor`):
/// the strides follow from `shape` (static integers or [`SymInt`]s) and
/// `stride_order`, which gives each mode's rank from the innermost (0, the
/// stride-1 mode); by default `(0, 1, ..., n-1)`, column-major. A stride
/// after a dynamic extent is a fresh `SymInt` with the product's
/// divisibility, 64 bits wide unless `use_32bit_stride`.
#[track_caller]
pub fn make_fake_compact_tensor<T: Numeric>(
    shape: impl Into<SpecTuple>,
    stride_order: Option<&[usize]>,
    use_32bit_stride: bool,
) -> TensorSpec {
    let shape = shape.into();
    assert!(shape.depth() <= 1, "cute: make_fake_compact_tensor: the shape {shape} must be flat");
    let modes: Vec<SpecLeaf> = shape.leaves().into_iter().copied().collect();
    let n = modes.len();
    let order: Vec<usize> = match stride_order {
        Some(o) => {
            assert!(
                o.len() == n && (0..n).all(|r| o.contains(&r)),
                "cute: make_fake_compact_tensor: stride_order {o:?} must be a permutation of 0..{n}, one per mode of {shape}"
            );
            o.to_vec()
        }
        None => (0..n).collect(),
    };
    let width = if use_32bit_stride { 32 } else { 64 };
    let mut stride = vec![SpecLeaf::Static(0); n];
    let (mut value, mut div, mut dynamic) = (1usize, 1usize, false);
    for rank in 0..n {
        let m = order.iter().position(|&r| r == rank).unwrap();
        stride[m] = if dynamic { SpecLeaf::Dynamic(SymInt::new(width, div, None)) } else { SpecLeaf::Static(value) };
        match modes[m] {
            SpecLeaf::Static(v) => {
                value *= v;
                div *= v;
            }
            SpecLeaf::Dynamic(s) => {
                dynamic = true;
                div *= s.divisibility();
            }
        }
    }
    let stride: SpecTuple = if shape.is_leaf() {
        Tree::Leaf(stride[0])
    } else {
        Tree::Branch(stride.into_iter().map(Tree::Leaf).collect())
    };
    make_fake_tensor::<T>(shape, stride)
}

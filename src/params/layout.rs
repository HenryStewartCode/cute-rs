//! Layouts, shapes and composed layouts as kernel parameters (CuTe DSL
//! passes `cute.Layout`, `ComposedLayout` and Shape/Stride/Coord tuples to
//! kernels): the static parts are part of the kernel, and each dynamic leaf
//! is one scalar parameter with its width and divisibility.

use super::symint::{SpecLeaf, SpecTuple, SymInt};
use super::tensor::{declare_tree, pack_tree, static_stride};
use super::{ArgError, ArgPack, KernelParam, leaf_name};
use crate::layout::{ComposedLayout, Inner, IntTuple, Layout, Shape, StrideLeaf};

/// The description of a layout parameter: a shape whose layout is compact
/// column-major (CuTe's `make_layout(shape)`, its strides computed in the
/// kernel from the extents), or a shape and a stride. Leaves are static
/// integers or [`SymInt`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutSpec {
    Compact(SpecTuple),
    Strided { shape: SpecTuple, stride: SpecTuple },
}

impl LayoutSpec {
    /// `shape:stride`.
    #[track_caller]
    pub fn new(shape: impl Into<SpecTuple>, stride: impl Into<SpecTuple>) -> LayoutSpec {
        let (shape, stride) = (shape.into(), stride.into());
        assert!(shape.congruent(&stride), "cute: LayoutSpec: shape {shape} and stride {stride} are not congruent");
        LayoutSpec::Strided { shape, stride }
    }

    /// The compact column-major layout of `shape`.
    pub fn col_major(shape: impl Into<SpecTuple>) -> LayoutSpec {
        LayoutSpec::Compact(shape.into())
    }

    /// A static layout.
    #[track_caller]
    pub fn of(layout: &Layout) -> LayoutSpec {
        let (Some(shape), Some(stride)) = (layout.shape().as_static(), static_stride(layout)) else {
            panic!("cute: LayoutSpec::of: the layout {layout} is not static");
        };
        LayoutSpec::new(&shape, &stride)
    }

    pub fn shape(&self) -> &SpecTuple {
        match self {
            LayoutSpec::Compact(s) | LayoutSpec::Strided { shape: s, .. } => s,
        }
    }
}

/// Declare a layout parameter `name` (CuTe DSL's `cute.Layout` argument):
/// one scalar per dynamic leaf, `{name}_shape_{path}` and
/// `{name}_stride_{path}`, each with its [`SymInt`]'s width and
/// divisibility. A compact spec's strides are computed from the extents.
#[track_caller]
pub fn param_layout(name: &str, spec: &LayoutSpec) -> Layout {
    let name = leaf_name(name);
    match spec {
        LayoutSpec::Compact(shape) => Layout::col_major(declare_tree(&format!("{name}_shape"), shape)),
        LayoutSpec::Strided { shape, stride } => {
            let shape = declare_tree(&format!("{name}_shape"), shape);
            let stride = declare_tree(&format!("{name}_stride"), stride).map(&mut |e| StrideLeaf::Int(e.clone()));
            Layout::new(shape, stride)
        }
    }
}

/// Declare a shape (or stride, or coordinate) parameter `name`: one scalar
/// per dynamic leaf, `{name}_{path}`.
#[track_caller]
pub fn param_shape(name: &str, spec: &SpecTuple) -> Shape {
    declare_tree(&leaf_name(name), spec)
}

/// The description of a composed-layout parameter: a static inner function
/// (a swizzle or a layout), and an offset and outer layout with dynamic
/// leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedSpec {
    pub inner: Inner,
    pub offset: SpecTuple,
    pub outer: LayoutSpec,
}

/// Declare a composed-layout parameter `name`: the dynamic leaves of its
/// offset (`{name}_offset_{path}`) and of its outer layout
/// (`{name}_outer_shape_{path}`, `{name}_outer_stride_{path}`).
#[track_caller]
pub fn param_composed_layout(name: &str, spec: &ComposedSpec) -> ComposedLayout {
    let name = leaf_name(name);
    let offset = declare_tree(&format!("{name}_offset"), &spec.offset);
    let outer = param_layout(&format!("{name}_outer"), &spec.outer);
    ComposedLayout { inner: spec.inner.clone(), offset, outer }
}

fn static_parts(name: &str, layout: &Layout) -> Result<(IntTuple, IntTuple), ArgError> {
    match (layout.shape().as_static(), static_stride(layout)) {
        (Some(s), Some(d)) => Ok((s, d)),
        _ => Err(ArgError::new(name, format!("a host layout must be static, got {layout}"))),
    }
}

fn pack_layout(name: &str, spec: &LayoutSpec, host: &Layout, out: &mut ArgPack) -> Result<(), ArgError> {
    let name = leaf_name(name);
    let (shape, stride) = static_parts(&name, host)?;
    if !shape.congruent(spec.shape()) {
        return Err(ArgError::new(&name, format!("the kernel takes a layout of shape {}, got {host}", spec.shape())));
    }
    pack_tree(&format!("{name}_shape"), spec.shape(), &shape, "extent", out)?;
    match spec {
        LayoutSpec::Compact(_) => {
            let compact = shape.compact_col_major();
            if stride != compact {
                return Err(ArgError::new(
                    &name,
                    format!("the kernel takes a compact column-major layout, got {host} (not {shape}:{compact})"),
                ));
            }
        }
        LayoutSpec::Strided { stride: s, .. } => pack_tree(&format!("{name}_stride"), s, &stride, "stride", out)?,
    }
    Ok(())
}

/// A layout parameter: the host passes a static [`Layout`] whose static
/// leaves match the spec's.
impl KernelParam for Layout {
    type Spec = LayoutSpec;
    type Host = Layout;

    #[track_caller]
    fn declare(name: &str, spec: &LayoutSpec) -> Self {
        param_layout(name, spec)
    }

    /// The host's layout, static.
    fn spec_of(host: &Layout) -> LayoutSpec {
        LayoutSpec::of(host)
    }

    fn pack(name: &str, spec: &LayoutSpec, host: &Layout, out: &mut ArgPack) -> Result<(), ArgError> {
        pack_layout(name, spec, host, out)
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        (self != other).then(|| format!("the layout {self:?} became {other:?}"))
    }
}

/// A shape, stride or coordinate parameter (CuTe DSL's dynamic `IntTuple`
/// arguments): the host passes an [`IntTuple`] of the spec's profile.
impl KernelParam for Shape {
    type Spec = SpecTuple;
    type Host = IntTuple;

    #[track_caller]
    fn declare(name: &str, spec: &SpecTuple) -> Self {
        param_shape(name, spec)
    }

    /// Every integer a dynamic `Int32` leaf, as CuTe DSL makes a tuple
    /// argument a dynamic `IntTuple`.
    fn spec_of(host: &IntTuple) -> SpecTuple {
        host.map(&mut |_| SpecLeaf::Dynamic(SymInt::new(32, 1, None)))
    }

    fn pack(name: &str, spec: &SpecTuple, host: &IntTuple, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        if !host.congruent(spec) {
            return Err(ArgError::new(&name, format!("the kernel takes a tuple of profile {spec}, got {host}")));
        }
        pack_tree(&name, spec, host, "value", out)
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        (self != other).then(|| format!("the tuple {self:?} became {other:?}"))
    }
}

/// A composed-layout parameter: the host passes a static
/// [`ComposedLayout`] with the spec's inner function.
impl KernelParam for ComposedLayout {
    type Spec = ComposedSpec;
    type Host = ComposedLayout;

    #[track_caller]
    fn declare(name: &str, spec: &ComposedSpec) -> Self {
        param_composed_layout(name, spec)
    }

    #[track_caller]
    fn spec_of(host: &ComposedLayout) -> ComposedSpec {
        let offset = host.offset.as_static().unwrap_or_else(|| panic!("cute: a host composed layout must be static"));
        ComposedSpec { inner: host.inner.clone(), offset: (&offset).into(), outer: LayoutSpec::of(&host.outer) }
    }

    fn pack(name: &str, spec: &ComposedSpec, host: &ComposedLayout, out: &mut ArgPack) -> Result<(), ArgError> {
        let name = leaf_name(name);
        if host.inner != spec.inner {
            return Err(ArgError::new(
                &name,
                format!("the kernel was built for the inner function {:?}, launched with {:?}", spec.inner, host.inner),
            ));
        }
        let offset = host
            .offset
            .as_static()
            .ok_or_else(|| ArgError::new(&name, format!("a host composed layout must be static, got {host}")))?;
        if !offset.congruent(&spec.offset) {
            return Err(ArgError::new(
                &name,
                format!("the kernel takes an offset of profile {}, got {offset}", spec.offset),
            ));
        }
        pack_tree(&format!("{name}_offset"), &spec.offset, &offset, "offset", out)?;
        pack_layout(&format!("{name}_outer"), &spec.outer, &host.outer, out)
    }

    fn carry_mismatch(&self, other: &Self) -> Option<String> {
        (self != other).then(|| format!("the composed layout {self} became {other}"))
    }
}

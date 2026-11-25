//! [`TensorSsa`]: CuTe DSL's `TensorSSA`, an immutable tensor-valued SSA
//! value held by one thread.
//!
//! A `TensorSsa<T>` is a static shape (an [`IntTuple`]) and one runtime
//! value of type `T` per element, in colexicographic (column-major) order.
//! It is scalarised: the IR holds one SSA value per element and no vector
//! type, so every operation on a tensor is one scalar operation per element,
//! and broadcasting, slicing and reshaping only pick values at build time.
//! Cloning is cheap (the values are shared).
//!
//! * [`Tensor::load_ssa`] reads a whole tensor and [`Tensor::store_ssa`]
//!   writes one (CuTe DSL's `Tensor.load()` and `Tensor.store(v)`); [`full`],
//!   [`full_like`], [`zeros_like`], [`ones_like`], [`empty_like`] and
//!   [`TensorSsa::new`] make one.
//! * `+ - * / %`, unary `-`, [`abs`](TensorSsa::abs), the comparisons
//!   ([`lt`](TensorSsa::lt) and friends), `& | ^ ! << >>`, [`where_`],
//!   [`any_`], [`all_`], [`reduce`](TensorSsa::reduce),
//!   [`apply_op`](TensorSsa::apply_op), [`map`](TensorSsa::map) and the
//!   `cute.math` functions work element by element, with CuTe DSL's
//!   promotion and broadcasting.
//! * [`get`](TensorSsa::get) and [`slice`](TensorSsa::slice) index it,
//!   [`broadcast_to`](TensorSsa::broadcast_to) and
//!   [`reshape`](TensorSsa::reshape) reshape it, [`to`](TensorSsa::to)
//!   converts it and [`bitcast`](TensorSsa::bitcast) reinterprets it.
//!
//! # Promotion
//!
//! `a op b` on a `TensorSsa<T>` has the element type `Value<T> op b` has
//! (see [`Promote`](crate::Promote)): the operators are implemented for the
//! same right-hand sides as [`Value`]'s, so an unsuffixed Rust literal adopts
//! the tensor's type exactly as it does with a scalar (`t_f16 * 2.0` is
//! Float16, where CuTe DSL makes the literal a Float32 and the result
//! Float32). A [`Value`] or another tensor of a different type promotes as
//! two values do: an `Int8` tensor plus an `Int32` tensor is an `Int32`
//! tensor.
//!
//! # Broadcasting
//!
//! Two operands of different shapes broadcast as CuTe DSL's
//! `_infer_broadcast_shape` does: the lower-rank shape is padded **on the
//! right** with size-1 modes (column-major order, unlike NumPy's left
//! padding: a `(4,)` tensor meets a `(4, 5)` one as `(4, 1)`), the padded
//! shapes must be congruent, and at each leaf the extents must be equal or
//! one of them 1. cute-rs also lets a 1 stand against a whole nested mode
//! (`(2, 1)` broadcasts against `(2, (2, 2))`), which CuTe DSL refuses, and
//! a one-mode tuple against its mode (a reduction's `(4,)` meets a `4`).
//!
//! # Differences from CuTe DSL
//!
//! * Integer `/` truncates, as Rust's and [`Value`]'s do: CuTe DSL's `/`
//!   on integers is a true division giving Float32, here
//!   [`true_div`](TensorSsa::true_div). [`floor_div`](TensorSsa::floor_div)
//!   is CuTe DSL's `//`, [`pow`](TensorSsa::pow) its `**`.
//! * Boolean tensors have no arithmetic (CuTe DSL computes `+` and `-` of
//!   Boolean tensors in Int32 and converts back); they have the logical
//!   operators, `eq` and `ne`.
//! * A store of another type converts implicitly only to a wider type of
//!   the same kind (float or integer) that holds every value exactly
//!   (Float16 to Float32, Uint8 to Int16); any other mismatch needs an
//!   explicit [`to`](TensorSsa::to) or [`bitcast`](TensorSsa::bitcast).
//!   CuTe DSL also converts implicitly to an equally wide type of the same
//!   kind and to wider types that round or overflow (Float16 to BFloat16,
//!   Int32 to Uint32, Int8 to Uint16), and stores the bits of a value of the
//!   other kind and the same width.
//! * Swizzled (composed-layout) tensors load and store whole too:
//!   [`autovec_copy`](crate::autovec_copy) handles swizzles, where CuTe DSL
//!   refuses a vectorized load or store of a composed layout.

mod convert;
mod logic;
mod ops;
mod print;

use std::fmt;
use std::rc::Rc;

use crate::builder::LocationScope;
use crate::control::Carry;
use crate::elementwise;
use crate::ir::{DType, ValueId};
use crate::layout::{Coord, IntTuple, Layout, SliceCoord, Tree};
use crate::tensor::{Engine, Tensor};
use crate::value::{IntoValue, Numeric, Value, select};

pub(crate) use convert::{convert_values, has_wide_conversion};
pub use logic::{ReductionOp, all_, any_, maximum, minimum, where_};
pub use ops::{SsaOperand, SsaOperandKind};
pub(crate) use print::print_nd;
pub(crate) use print::print_pieces;
pub use print::{PrintParts, PrintTensor, print_tensor};

/// CuTe DSL's `TensorSSA`: a static shape and one runtime value per
/// element, in colexicographic order. See the [module documentation](self).
pub struct TensorSsa<T> {
    shape: IntTuple,
    values: Rc<[Value<T>]>,
}

impl<T> Clone for TensorSsa<T> {
    fn clone(&self) -> Self {
        TensorSsa { shape: self.shape.clone(), values: self.values.clone() }
    }
}

/// Above this many elements, [`TensorSsa::get`] with a runtime coordinate
/// spills the tensor to a per-thread stack array and loads the element from
/// it; up to it, a balanced tree of selects (`ceil(log2 n)` deep) picks it.
pub const SELECT_TREE_MAX: usize = 32;

impl<T: Numeric> TensorSsa<T> {
    /// A tensor of `shape` holding `values` in colexicographic order
    /// (CuTe DSL's `TensorSSA(value, shape, dtype)`). The number of values
    /// must be the size of the shape.
    #[track_caller]
    pub fn new(shape: impl Into<IntTuple>, values: Vec<Value<T>>) -> TensorSsa<T> {
        let shape = shape.into();
        assert_eq!(
            values.len(),
            shape.size(),
            "TensorSsa::new: shape {shape:?} has {} elements, but {} values were given",
            shape.size(),
            values.len()
        );
        TensorSsa { shape, values: values.into() }
    }

    pub(crate) fn from_parts(shape: IntTuple, values: Vec<Value<T>>) -> TensorSsa<T> {
        debug_assert_eq!(shape.size(), values.len());
        TensorSsa { shape, values: values.into() }
    }

    /// The shape (CuTe DSL's `TensorSSA.shape`).
    pub fn shape(&self) -> &IntTuple {
        &self.shape
    }

    /// The number of elements.
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// The number of top-level modes.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// The element type (CuTe DSL's `TensorSSA.dtype`).
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// The elements, in colexicographic order.
    pub fn values(&self) -> &[Value<T>] {
        &self.values
    }

    /// A tensor of the same shape holding `f` of every element.
    #[track_caller]
    pub fn map<U: Numeric>(&self, mut f: impl FnMut(Value<T>) -> Value<U>) -> TensorSsa<U> {
        let _at = LocationScope::new();
        TensorSsa::from_parts(self.shape.clone(), self.values.iter().map(|&v| f(v)).collect())
    }

    /// The same elements with shape `shape`, whose size must be this
    /// tensor's (CuTe DSL's `TensorSSA.reshape`); the colexicographic order
    /// is kept.
    #[track_caller]
    pub fn reshape(&self, shape: impl Into<IntTuple>) -> TensorSsa<T> {
        let shape = shape.into();
        assert_eq!(
            shape.size(),
            self.size(),
            "TensorSsa::reshape: {self:?} has {} elements, but the shape {shape:?} has {}",
            self.size(),
            shape.size()
        );
        TensorSsa { shape, values: self.values.clone() }
    }

    /// This tensor broadcast to `target` (CuTe DSL's
    /// `TensorSSA.broadcast_to`): its shape is padded on the right with
    /// size-1 modes up to `target`'s rank, and every leaf must then equal
    /// `target`'s or be 1 (a 1 may also stand against a whole nested mode).
    /// No instruction is emitted: elements are repeated.
    #[track_caller]
    pub fn broadcast_to(&self, target: impl Into<IntTuple>) -> TensorSsa<T> {
        let target = target.into();
        self.broadcast_to_shape(&target, "TensorSsa::broadcast_to")
    }

    #[track_caller]
    pub(crate) fn broadcast_to_shape(&self, target: &IntTuple, what: &str) -> TensorSsa<T> {
        if &self.shape == target {
            return self.clone();
        }
        let refuse = || -> ! {
            panic!(
                "{what}: a tensor of shape {:?} cannot be broadcast to {target:?}: padded on the right with size-1 \
                 modes it must be congruent with the target, each extent equal to the target's or 1",
                self.shape
            )
        };
        if self.shape.rank() > target.rank() {
            refuse();
        }
        let padded = pad_rank(&self.shape, target.rank());
        let Some(expanded) = expand_like(&padded, target) else { refuse() };
        let (src, dst) = (expanded.flat_leaves(), target.flat_leaves());
        if src.iter().zip(&dst).any(|(&s, &d)| s != d && s != 1) {
            refuse();
        }
        let map = elementwise::broadcast_map(&src, &dst);
        TensorSsa::from_parts(target.clone(), elementwise::gather(&self.values, &map))
    }

    /// One element (CuTe DSL's `tensor_ssa[crd]`): the coordinate is mapped
    /// through the compact column-major layout of the shape (`crd2idx`).
    /// With a static coordinate this picks a value; with a runtime one, a
    /// balanced tree of selects picks it for up to [`SELECT_TREE_MAX`]
    /// elements, and a larger tensor is written to a per-thread stack array
    /// and the element loaded from it. A runtime coordinate is not checked:
    /// out of range, the select tree gives some element and the stack array
    /// is read past its end (CuTe DSL's dynamic `vector.extractelement`
    /// gives poison).
    #[track_caller]
    pub fn get(&self, coord: impl Into<Coord>) -> Value<T> {
        let _at = LocationScope::new();
        let layout = Layout::col_major(self.shape.clone());
        let idx = crate::layout::crd2idx(coord, &layout);
        if let Some(i) = idx.as_static() {
            return self.values[i];
        }
        let n = self.size();
        if n <= SELECT_TREE_MAX || T::DTYPE.is_word_packed() {
            let i: Value<i32> = idx.into_value();
            return select_tree(&self.values, i);
        }
        let stack = crate::tensor::make_rmem_tensor::<T>(Layout::col_major(n));
        for (k, &v) in self.values.iter().enumerate() {
            stack.store(k, v);
        }
        stack.load(Coord::Leaf(idx))
    }

    /// The modes `coord` keeps (CuTe DSL's `tensor_ssa[None, i]`): its
    /// `..` (`None`) leaves keep a mode, its other leaves, which must be
    /// static, fix one. The result's shape is `slice_(shape, coord)`, the
    /// kept modes in order and flattened as [`Layout::slice`] keeps them
    /// (`(4, (5, 4))` sliced by `(.., (.., 1))` is `(4, 5)`).
    #[track_caller]
    pub fn slice(&self, coord: impl Into<SliceCoord>) -> TensorSsa<T> {
        let coord = coord.into();
        if coord.find_if(|c, _| c.as_ref().is_some_and(|e| !e.is_static())).is_some() {
            panic!(
                "TensorSsa::slice: the coordinate {} holds a runtime value; slices take static coordinates (get \
                 reads one element at a runtime coordinate)",
                slice_text(&coord)
            );
        }
        let layout = Layout::col_major(self.shape.clone());
        let (kept, offset) = layout.slice_checked(&coord, "TensorSsa::slice");
        let base = offset.as_static().expect("a static slice");
        let shape = static_shape(kept.shape());
        let values = (0..kept.size()).map(|k| self.values[base + kept.call(k)]).collect();
        TensorSsa::from_parts(shape, values)
    }
}

/// A balanced tree of selects: level `k` picks between pairs by bit `k` of
/// `i` (`ceil(log2 n)` comparisons and `n - 1` selects). An index past the
/// end gives one of the elements.
#[track_caller]
pub(crate) fn select_tree<T: Numeric>(values: &[Value<T>], i: Value<i32>) -> Value<T> {
    let mut level: Vec<Value<T>> = values.to_vec();
    let mut bit = 0;
    while level.len() > 1 {
        let set = (i & (1i32 << bit)).ne(0);
        level = level.chunks(2).map(|p| if p.len() == 2 { select(set, p[1], p[0]) } else { p[0] }).collect();
        bit += 1;
    }
    level[0]
}

/// `shape` with size-1 modes appended up to rank `rank`.
fn pad_rank(shape: &IntTuple, rank: usize) -> IntTuple {
    if shape.rank() >= rank {
        return shape.clone();
    }
    let mut modes: Vec<IntTuple> = match shape {
        Tree::Leaf(_) => vec![shape.clone()],
        Tree::Branch(m) => m.clone(),
    };
    modes.resize(rank, Tree::Leaf(1));
    Tree::Branch(modes)
}

/// `src` with the structure of `dst`: a leaf 1 against a nested mode
/// becomes that mode with every extent 1, and a one-mode tuple stands for
/// its mode. `None` if they are otherwise not congruent.
fn expand_like(src: &IntTuple, dst: &IntTuple) -> Option<IntTuple> {
    match (src, dst) {
        (Tree::Leaf(_), Tree::Leaf(_)) => Some(src.clone()),
        (Tree::Leaf(1), Tree::Branch(_)) => Some(dst.map(&mut |_| 1)),
        // A one-mode tuple is its mode: `(4,)` stands for `4`.
        (Tree::Branch(a), Tree::Leaf(_)) if a.len() == 1 => expand_like(&a[0], dst),
        (Tree::Leaf(_), Tree::Branch(b)) if b.len() == 1 => expand_like(src, &b[0]),
        (Tree::Branch(a), Tree::Branch(b)) if a.len() == b.len() => {
            a.iter().zip(b).map(|(x, y)| expand_like(x, y)).collect::<Option<Vec<_>>>().map(Tree::Branch)
        }
        _ => None,
    }
}

/// The shape two shapes broadcast to (CuTe DSL's `_infer_broadcast_shape`),
/// or `None`.
fn merge_shapes(a: &IntTuple, b: &IntTuple) -> Option<IntTuple> {
    match (a, b) {
        (Tree::Leaf(x), Tree::Leaf(y)) => match (*x, *y) {
            (x, y) if x == y => Some(Tree::Leaf(x)),
            (1, y) => Some(Tree::Leaf(y)),
            (x, 1) => Some(Tree::Leaf(x)),
            _ => None,
        },
        (Tree::Leaf(1), Tree::Branch(_)) => Some(b.clone()),
        (Tree::Branch(_), Tree::Leaf(1)) => Some(a.clone()),
        // A one-mode tuple is its mode: `(4,)` meets `4`.
        (Tree::Branch(xs), Tree::Leaf(_)) if xs.len() == 1 => merge_shapes(&xs[0], b),
        (Tree::Leaf(_), Tree::Branch(ys)) if ys.len() == 1 => merge_shapes(a, &ys[0]),
        (Tree::Branch(xs), Tree::Branch(ys)) if xs.len() == ys.len() => {
            xs.iter().zip(ys).map(|(x, y)| merge_shapes(x, y)).collect::<Option<Vec<_>>>().map(Tree::Branch)
        }
        _ => None,
    }
}

/// The shape `shapes` broadcast to, or a build panic naming them.
#[track_caller]
pub(crate) fn broadcast_shape(shapes: &[&IntTuple], what: &str) -> IntTuple {
    let rank = shapes.iter().map(|s| s.rank()).max().unwrap_or(1);
    let mut out = pad_rank(shapes[0], rank);
    for s in &shapes[1..] {
        let padded = pad_rank(s, rank);
        out = merge_shapes(&out, &padded).unwrap_or_else(|| {
            panic!(
                "{what}: the shapes {} do not broadcast: padded on the right with size-1 modes to rank {rank}, they \
                 must be congruent, with equal extents or 1 at each leaf",
                shapes.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(" and ")
            )
        });
    }
    if shapes.iter().all(|s| *s == shapes[0]) { shapes[0].clone() } else { out }
}

/// A slice coordinate or profile as CuTe prints it: `_` for `None`.
pub(crate) fn slice_text(c: &SliceCoord) -> String {
    c.map(&mut |l| l.as_ref().map_or("_".to_string(), |e| e.to_string())).to_string()
}

/// A static shape as an [`IntTuple`].
#[track_caller]
pub(crate) fn static_shape(shape: &crate::layout::Shape) -> IntTuple {
    shape.as_static().unwrap_or_else(|| panic!("cute: the shape {shape:?} is not static"))
}

/// Something a tensor value can be made like ([`full_like`] and friends):
/// a [`TensorSsa`] or a [`Tensor`] with a static shape.
pub trait SsaShape {
    #[doc(hidden)]
    fn ssa_shape(&self) -> IntTuple;
}

impl<U> SsaShape for TensorSsa<U> {
    fn ssa_shape(&self) -> IntTuple {
        self.shape.clone()
    }
}

impl<U> SsaShape for Tensor<U> {
    #[track_caller]
    fn ssa_shape(&self) -> IntTuple {
        self.shape().as_static().unwrap_or_else(|| {
            panic!("cute: tensor {:?} has a dynamic shape; tensor values have static shapes", self.layout())
        })
    }
}

/// A fill value: a host number or a runtime value, converted to the
/// element type with [`Value::to`] (CuTe DSL converts with `.to(dtype)`).
pub trait FillValue<T: Numeric> {
    #[doc(hidden)]
    fn fill_value(self) -> Value<T>;
}

impl<T: Numeric, U: Numeric> FillValue<T> for Value<U> {
    #[track_caller]
    fn fill_value(self) -> Value<T> {
        self.to::<T>()
    }
}

macro_rules! fill_from_host {
    ($($t:ty),*) => {$(
        impl<T: Numeric> FillValue<T> for $t {
            #[track_caller]
            fn fill_value(self) -> Value<T> {
                if <$t as Numeric>::DTYPE == T::DTYPE {
                    return Value::from_id(Value::constant(self).id());
                }
                Value::constant(self).to::<T>()
            }
        }
    )*};
}
fill_from_host!(
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    i128,
    u128,
    f32,
    f64,
    crate::value::F16,
    crate::value::BF16,
    crate::numeric::I4,
    crate::numeric::TF32,
    crate::numeric::F8E4M3FN,
    crate::numeric::F8E5M2,
    crate::numeric::F8E4M3,
    crate::numeric::F8E4M3B11FNUZ,
    crate::numeric::F8E8M0FNU,
    crate::numeric::F8E5M3FNU
);

/// A tensor of `shape` with every element `fill` converted to `T` (CuTe
/// DSL's `cute.full(shape, fill_value, dtype)`).
///
/// ```
/// use cute::prelude::*;
/// Kernel::build("full", || {
///     let out = param_tensor::<f32>("out", Layout::col_major((4, 8)));
///     let t = full::<f32>((4, 8), 1);
///     out.store_ssa(&(t * 2.0));
/// });
/// ```
#[track_caller]
pub fn full<T: Numeric>(shape: impl Into<IntTuple>, fill: impl FillValue<T>) -> TensorSsa<T> {
    let shape = shape.into();
    let v = fill.fill_value();
    TensorSsa::from_parts(shape.clone(), vec![v; shape.size()])
}

/// A tensor shaped like `a` (a [`TensorSsa`] or a [`Tensor`] with a static
/// shape) with every element `fill` (CuTe DSL's `cute.full_like`). The
/// element type is a type parameter: name it or let it be inferred.
#[track_caller]
pub fn full_like<T: Numeric>(a: &impl SsaShape, fill: impl FillValue<T>) -> TensorSsa<T> {
    full(a.ssa_shape(), fill)
}

/// [`full_like`] with 0 (CuTe DSL's `cute.zeros_like`).
#[track_caller]
pub fn zeros_like<T: Numeric>(a: &impl SsaShape) -> TensorSsa<T> {
    full_like(a, 0i32)
}

/// [`full_like`] with 1 (CuTe DSL's `cute.ones_like`).
#[track_caller]
pub fn ones_like<T: Numeric>(a: &impl SsaShape) -> TensorSsa<T> {
    full_like(a, 1i32)
}

/// CuTe DSL's `cute.empty_like`, which is [`full_like`] with 0 there too
/// (an SSA value has no uninitialised state).
#[track_caller]
pub fn empty_like<T: Numeric>(a: &impl SsaShape) -> TensorSsa<T> {
    full_like(a, 0i32)
}

impl<T: Numeric> Tensor<T> {
    /// The whole tensor as a [`TensorSsa`] (CuTe DSL's `Tensor.load()`).
    /// The shape must be static. A register fragment is read directly; a
    /// tensor in memory is copied into a fragment like it
    /// ([`make_fragment_like`](crate::make_fragment_like), whose layout
    /// follows the order of its strides) with
    /// [`autovec_copy`](crate::autovec_copy), so contiguous runs move as
    /// vectors (a row-major 4 x 4 Float32 tile as four 16-byte loads).
    /// Sub-byte elements must fill whole 32-bit words. A coordinate tensor
    /// cannot be loaded (it holds coordinates, read with
    /// [`Tensor::coord`]):
    ///
    /// ```compile_fail
    /// use cute::prelude::*;
    /// Kernel::build("load_identity", || {
    ///     let _ = make_identity_tensor((4, 4)).load_ssa();
    /// });
    /// ```
    #[track_caller]
    pub fn load_ssa(&self) -> TensorSsa<T> {
        let _at = LocationScope::new();
        if let Engine::ArithTuple(_) = self.engine {
            panic!(
                "Tensor::load_ssa: {self:?} is a coordinate tensor, which holds coordinates, not values; read them \
                 with coord()"
            );
        }
        let shape = self.ssa_shape();
        if self.is_register() {
            let values = (0..shape.size()).map(|i| self.load(i)).collect();
            return TensorSsa::from_parts(shape, values);
        }
        check_sub_byte_words::<T>(shape.size(), "Tensor::load_ssa");
        let frag = crate::tensor::make_fragment_like::<T, T>(self);
        crate::atom::copy::autovec_copy(self, &frag);
        let values = (0..shape.size()).map(|i| frag.load(i)).collect();
        TensorSsa::from_parts(shape, values)
    }

    /// Write every element from `data` (CuTe DSL's `Tensor.store(data)`),
    /// in colexicographic order: the sizes must be equal, not the shapes.
    /// Data of a narrower type of the same kind (float or integer) whose
    /// every value `T` holds exactly is converted to `T`; any other type
    /// difference panics (convert with [`TensorSsa::to`] first). A register fragment is written directly;
    /// a tensor in memory through a fragment like it and
    /// [`autovec_copy`](crate::autovec_copy). Slice assignment, CuTe DSL's
    /// `t[None, i] = v`, is `t.slice((.., i)).store_ssa(&v)`.
    #[track_caller]
    pub fn store_ssa<U: Numeric>(&self, data: &TensorSsa<U>) {
        let _at = LocationScope::new();
        if let Engine::ArithTuple(_) = self.engine {
            panic!("Tensor::store_ssa: {self:?} is a coordinate tensor, which is read-only");
        }
        let shape = self.ssa_shape();
        assert_eq!(
            shape.size(),
            data.size(),
            "Tensor::store_ssa: the tensor {self:?} of shape {shape:?} has {} elements, but the data {data:?} has {}",
            shape.size(),
            data.size()
        );
        let values = cvt_to_dest::<U, T>(data, "Tensor::store_ssa");
        if self.is_register() {
            for (i, v) in values.into_iter().enumerate() {
                self.store(i, v);
            }
            return;
        }
        check_sub_byte_words::<T>(shape.size(), "Tensor::store_ssa");
        let frag = crate::tensor::make_fragment_like::<T, T>(self);
        for (i, v) in values.into_iter().enumerate() {
            frag.store(i, v);
        }
        crate::atom::copy::autovec_copy(&frag, self);
    }
}

#[track_caller]
fn check_sub_byte_words<T: Numeric>(n: usize, what: &str) {
    if T::DTYPE.is_word_packed() {
        let bits = n * T::DTYPE.bits();
        assert!(
            bits.is_multiple_of(32),
            "{what}: {n} {} elements are {bits} bits, but sub-byte tensors move in whole 32-bit words",
            T::DTYPE.cute_name()
        );
    }
}

/// `data`'s values as `T`, as CuTe DSL's `_cvt_to_dest` converts implicitly
/// (to a wider or equally wide type of the same kind), but only where the
/// conversion is exact: every `U` value is a `T` value. CuTe DSL also
/// converts implicitly between types of one width (Float16 and BFloat16,
/// Float32 to TFloat32, Int32 and Uint32, the FP8 formats) and widens
/// into types of a smaller range or precision (Int8 to Uint16,
/// Float8E8M0FNU to Float16), all of which round, overflow or change
/// values; here they need an explicit `.to()`.
#[track_caller]
pub(crate) fn cvt_to_dest<U: Numeric, T: Numeric>(data: &TensorSsa<U>, what: &str) -> Vec<Value<T>> {
    if U::DTYPE == T::DTYPE {
        return data.values.iter().map(|v| Value::from_id(v.id())).collect();
    }
    assert!(
        converts_exactly(U::DTYPE, T::DTYPE),
        "{what}: a {} tensor cannot be stored to {} elements implicitly: only an exact conversion to a wider type of \
         the same kind is; convert the data with .to::<{}>() (or reinterpret it with .bitcast())",
        U::DTYPE.cute_name(),
        T::DTYPE.cute_name(),
        T::DTYPE.cute_name()
    );
    data.to::<T>().values.to_vec()
}

/// Whether every value of `from` converts to `to` exactly (two integer or
/// two float types; not Booleans or the packed narrow-float views).
pub(crate) fn converts_exactly(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }
    if from.is_packed_view() || to.is_packed_view() {
        return false;
    }
    if from.is_int() && to.is_int() {
        let (fb, tb) = (from.bits(), to.bits());
        return match (from.is_signed(), to.is_signed()) {
            (false, false) | (true, true) => tb >= fb,
            (false, true) => tb > fb,
            (true, false) => false,
        };
    }
    let (Some(f), Some(t)) = (from.float_format(), to.float_format()) else { return false };
    use crate::ir::NanEncoding;
    // TFloat32 is stored as an f32 but keeps 10 mantissa bits.
    let precision = |d: DType, mant: u32| if d == DType::TF32 { 10 } else { mant };
    // The exponents of the largest finite and the smallest positive value.
    let emax = |x: crate::ir::FloatFormat| {
        let top = (1i32 << x.exp_bits) - 1 - x.bias;
        let top_reserved = x.nan == NanEncoding::Ieee || (x.nan == NanEncoding::AllOnes && x.mant_bits == 0);
        if top_reserved { top - 1 } else { top }
    };
    let emin = |x: crate::ir::FloatFormat, p: u32| if x.has_zero { 1 - x.bias - p as i32 } else { -x.bias };
    let (fp, tp) = (precision(from, f.mant_bits), precision(to, t.mant_bits));
    let negative_zero = |x: crate::ir::FloatFormat| x.signed && x.has_zero && x.nan != NanEncoding::NegativeZero;
    (t.signed || !f.signed)
        && tp >= fp
        && emax(t) >= emax(f)
        && emin(t, tp) <= emin(f, fp)
        && (t.has_inf || !f.has_inf)
        && (t.nan != NanEncoding::None || f.nan == NanEncoding::None)
        && (negative_zero(t) || !negative_zero(f))
}

impl<T: Numeric> Carry for TensorSsa<T> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        out.extend(self.values.iter().map(|v| v.checked()));
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let vs = (0..self.size()).map(|_| Value::from_id(values.next().expect("carry: too few values"))).collect();
        TensorSsa::from_parts(self.shape.clone(), vs)
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.shape != other.shape)
            .then(|| format!("a tensor value of shape {:?} became one of shape {:?}", self.shape, other.shape))
    }
}

/// `tensor_value<vector<NxT> o shape>`, as CuTe DSL prints a `TensorSSA`.
impl<T: Numeric> fmt::Debug for TensorSsa<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor_value<vector<{}x{}> o {:?}>", self.size(), T::DTYPE.name(), self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::converts_exactly;
    use crate::ir::DType::*;

    #[test]
    fn exact_conversions() {
        for (from, to) in [
            (F16, F32),
            (BF16, F32),
            (F32, F64),
            (F16, TF32),
            (TF32, F32),
            (F8E4M3FN, F16),
            (F8E5M2, F16),
            (F8E4M3FN, BF16),
            (F8E8M0FNU, BF16),
            (F6E3M2FN, F8E4M3FN),
            (F4E2M1FN, F16),
            (I8, I16),
            (U8, I16),
            (U8, U32),
            (I4, I8),
        ] {
            assert!(converts_exactly(from, to), "{from:?} to {to:?}");
        }
        for (from, to) in [
            (F16, BF16),
            (BF16, F16),
            (F32, TF32),
            (F32, F16),
            (F8E4M3FN, F8E5M2),
            (F8E8M0FNU, F16),
            (F8E5M3FNU, F16),
            (F6E2M3FN, F8E5M2),
            (I32, U32),
            (U32, I32),
            (I8, U16),
            (I32, I16),
            (I32, F32),
            (Bool, I8),
        ] {
            assert!(!converts_exactly(from, to), "{from:?} to {to:?}");
        }
    }
}

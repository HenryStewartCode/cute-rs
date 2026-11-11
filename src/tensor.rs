//! Tensors: a layout over some storage.
//!
//! As in CuTe, a [`Tensor`] is an engine (where the elements live) composed
//! with a [`Layout`] (how coordinates map to element offsets), plus a base
//! offset that may depend on runtime values (a thread's partition, a block's
//! tile). Slicing, tiling and partitioning only change the layout and the
//! offset, so they cost nothing at runtime.
//!
//! Storage comes in these kinds:
//!
//! * global memory: kernel parameters ([`param_tensor`], or
//!   [`param_ptr`] for buffers sized at launch),
//! * shared memory: per-block allocations ([`make_smem_tensor`],
//!   [`alloc_smem`]), optionally swizzled,
//! * registers: fragments ([`make_fragment`]). Register tensors must be
//!   indexed with static coordinates; they hold SSA values directly,
//! * per-thread memory: arrays that can be indexed at runtime
//!   ([`make_rmem_tensor`]).
//!
//! A [`Ptr`] is a typed position in global or shared memory. [`make_tensor`]
//! puts a layout on one, and [`Ptr::cast`] reinterprets the memory as another
//! element type, which is how one shared-memory allocation is reused for
//! different purposes.
//!
//! Identity tensors ([`make_identity_tensor`]) hold coordinates instead of
//! data and are used to compute predicates for partial tiles.

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::panic::Location;
use std::rc::Rc;

use crate::builder;
use crate::control::Carry;
use crate::ir::{BufferDecl, BufferId, MemSpace, Op, ValueId};
use crate::layout::{
    ComposedLayout, Coord, IndexExpr, IntTuple, Layout, SliceCoord, Swizzle, Tileable, Tiler, Tree, zipped_divide,
};
use crate::value::{Boolean, Int32, IntoValue, Numeric, Value};

/// Element marker for identity (coordinate) tensors.
#[derive(Clone, Copy, Debug)]
pub struct Ident;

#[derive(Clone)]
pub(crate) enum Engine {
    Memory { buffer: BufferId, space: MemSpace },
    Register(Rc<RegisterStorage>),
    Identity(IntTuple),
}

/// The registers of a fragment: the SSA value each holds, shared by every
/// view of the fragment.
pub(crate) struct RegisterStorage {
    slots: RefCell<Vec<Option<ValueId>>>,
    /// The builder's latest region serial when the fragment was created:
    /// the runtime loops opened after it see it as state from outside,
    /// which they do not carry.
    created: u64,
}

impl RegisterStorage {
    fn new(slots: Vec<Option<ValueId>>) -> Rc<RegisterStorage> {
        Rc::new(RegisterStorage { slots: RefCell::new(slots), created: builder::latest_serial() })
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.borrow().len()
    }

    /// The value in register `i`, if written, without counting as a read.
    pub(crate) fn peek(&self, i: usize) -> Option<ValueId> {
        self.slots.borrow().get(i).copied().flatten()
    }

    /// Read register `i` at the caller's location.
    #[track_caller]
    fn read(self: &Rc<Self>, i: usize) -> Option<ValueId> {
        let v = self.peek(i);
        if v.is_some() {
            let loc = Location::caller();
            builder::with(|b| b.fragment_read((Rc::as_ptr(self) as usize, i), self.created, loc));
        }
        v
    }

    /// Write register `i` at the caller's location.
    #[track_caller]
    fn write(self: &Rc<Self>, i: usize, v: ValueId) {
        let n = self.len();
        assert!(i < n, "cute: register tensor index {i} out of range ({n} slots)");
        let loc = Location::caller();
        builder::with(|b| b.fragment_write((Rc::as_ptr(self) as usize, i), self.created, loc));
        self.slots.borrow_mut()[i] = Some(v);
    }
}

pub struct Tensor<T> {
    pub(crate) engine: Engine,
    pub(crate) layout: Layout,
    pub(crate) offset: IndexExpr,
    pub(crate) swizzle: Option<Swizzle>,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            engine: self.engine.clone(),
            layout: self.layout.clone(),
            offset: self.offset.clone(),
            swizzle: self.swizzle,
            _t: PhantomData,
        }
    }
}

impl<T> Tileable for Tensor<T> {
    fn layout_ref(&self) -> &Layout {
        &self.layout
    }
    fn with_layout(&self, layout: Layout) -> Self {
        Tensor { layout, ..self.clone() }
    }
}

impl<T> Tensor<T> {
    pub(crate) fn new(engine: Engine, layout: Layout) -> Self {
        Tensor { engine, layout, offset: IndexExpr::Static(0), swizzle: None, _t: PhantomData }
    }

    /// The layout from coordinates to element offsets.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The tensor's shape.
    pub fn shape(&self) -> &IntTuple {
        self.layout.shape()
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Number of top-level modes.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// The base element offset (before the layout is applied).
    pub fn base_offset(&self) -> &IndexExpr {
        &self.offset
    }

    /// The swizzle applied to offsets (swizzled shared memory).
    pub fn swizzle(&self) -> Option<Swizzle> {
        self.swizzle
    }

    /// Whether this is a register fragment.
    pub fn is_register(&self) -> bool {
        matches!(self.engine, Engine::Register(_))
    }

    /// Where a memory tensor lives, or `None` for registers and identity
    /// tensors.
    pub fn memory_space(&self) -> Option<MemSpace> {
        match self.engine {
            Engine::Memory { space, .. } => Some(space),
            _ => None,
        }
    }

    pub(crate) fn buffer(&self) -> Option<BufferId> {
        match self.engine {
            Engine::Memory { buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    /// A pointer to the tensor's first element. Memory tensors only.
    #[track_caller]
    pub fn ptr(&self) -> Ptr<T> {
        match self.engine {
            Engine::Memory { buffer, space } => Ptr { buffer, space, offset: self.offset.clone(), _t: PhantomData },
            _ => panic!("cute: only memory tensors have an address"),
        }
    }

    /// Slice with `coord`; `..` keeps a mode (CuTe's `tensor[None, 0]`).
    pub fn slice(&self, coord: impl Into<SliceCoord>) -> Tensor<T> {
        let (layout, off) = self.layout.slice(&coord.into());
        Tensor { layout, offset: self.offset.clone() + off, ..self.clone() }
    }

    /// Compose the layout with a tiler (by mode for tuples).
    pub fn compose(&self, tiler: impl Into<Tiler>) -> Tensor<T> {
        self.with_layout(self.layout.compose_tiler(&tiler.into()))
    }

    /// Group modes `[begin, end)` into one.
    pub fn group_modes(&self, begin: usize, end: usize) -> Tensor<T> {
        self.with_layout(self.layout.group(begin, end))
    }

    /// Physical element offset of `coord` (swizzle applied).
    pub fn element_offset(&self, coord: &Coord) -> IndexExpr {
        let linear = self.offset.clone() + self.layout.offset(coord);
        match self.swizzle {
            Some(sw) => sw.apply_expr(linear),
            None => linear,
        }
    }

    pub(crate) fn register_storage(&self) -> Option<&Rc<RegisterStorage>> {
        match &self.engine {
            Engine::Register(s) => Some(s),
            _ => None,
        }
    }
}

#[track_caller]
fn static_register_index(off: &IndexExpr) -> usize {
    off.as_static().unwrap_or_else(|| {
        let note = builder::var_value_note(off);
        panic!("cute: register tensors must be indexed with static coordinates (got offset {off}){note}")
    })
}

impl<T: Numeric> Tensor<T> {
    /// Read one element (CuTe's `tensor[coord]`).
    #[track_caller]
    pub fn load(&self, coord: impl Into<Coord>) -> Value<T> {
        let off = self.element_offset(&coord.into());
        self.load_at(off)
    }

    #[track_caller]
    pub(crate) fn load_at(&self, off: IndexExpr) -> Value<T> {
        match &self.engine {
            Engine::Register(storage) => {
                let i = static_register_index(&off);
                let v = storage.read(i);
                Value::from_id(
                    v.unwrap_or_else(|| panic!("cute: register tensor element {i} read before it was written")),
                )
            }
            Engine::Memory { buffer, .. } => Value::from_id(load_vector::<T>(*buffer, off, 1)[0]),
            Engine::Identity(_) => panic!("cute: identity tensors hold coordinates; use coord() instead of load()"),
        }
    }

    /// Write one element (CuTe's `tensor[coord] = value`).
    #[track_caller]
    pub fn store(&self, coord: impl Into<Coord>, value: impl IntoValue<T>) {
        let v = value.into_value();
        let off = self.element_offset(&coord.into());
        self.store_at(off, v);
    }

    #[track_caller]
    pub(crate) fn store_at(&self, off: IndexExpr, v: Value<T>) {
        match &self.engine {
            Engine::Register(storage) => {
                let i = static_register_index(&off);
                storage.write(i, v.checked());
            }
            Engine::Memory { buffer, .. } => store_vector(*buffer, off, &[v.checked()]),
            Engine::Identity(_) => panic!("cute: identity tensors are read-only"),
        }
    }

    /// Every element, in colexicographic order.
    #[track_caller]
    pub fn values(&self) -> Vec<Value<T>> {
        (0..self.size()).map(|i| self.load(i)).collect()
    }

    /// Write every element from `values` (colexicographic order).
    #[track_caller]
    pub fn store_values(&self, values: &[Value<T>]) {
        assert_eq!(values.len(), self.size(), "cute: store_values length mismatch");
        for (i, v) in values.iter().enumerate() {
            self.store(i, *v);
        }
    }

    /// Set every element to `value`.
    #[track_caller]
    pub fn fill(&self, value: impl IntoValue<T>) {
        let v = value.into_value();
        for i in 0..self.size() {
            self.store(i, v);
        }
    }

    /// Apply `f` to every element in place (register tensors) or with a
    /// load/store per element (memory).
    #[track_caller]
    pub fn map_in_place(&self, mut f: impl FnMut(Value<T>) -> Value<T>) {
        for i in 0..self.size() {
            let v = self.load(i);
            self.store(i, f(v));
        }
    }

    /// A new register fragment holding every element converted to `U`.
    #[track_caller]
    pub fn to<U: Numeric>(&self) -> Tensor<U> {
        let out = make_fragment_like::<U, T>(self);
        for i in 0..self.size() {
            out.store(i, self.load(i).to::<U>());
        }
        out
    }
}

/// Identity tensors give each top-level mode its own field of the offset,
/// `IDENTITY_FIELD` apart, so coordinates past the end of a mode (in a
/// partial tile) stay distinct instead of wrapping into the next mode. The
/// fields are separated symbolically and never materialised together.
const IDENTITY_FIELD: usize = 1 << 20;

impl Tensor<Ident> {
    /// The coordinate held at `coord`: one index per top-level mode of the
    /// identity tensor's shape.
    pub fn coord(&self, coord: impl Into<Coord>) -> Coord {
        let Engine::Identity(shape) = &self.engine else { unreachable!() };
        let off = self.element_offset(&coord.into());
        let rank = shape.rank();
        let fields = (0..rank)
            .map(|i| {
                let field = off.clone() / IDENTITY_FIELD.pow(i as u32);
                Tree::Leaf(if i + 1 == rank { field } else { field % IDENTITY_FIELD })
            })
            .collect();
        Tree::Branch(fields)
    }
}

/// `coord < bound` in every mode, as a runtime predicate.
#[track_caller]
pub fn elem_less(coord: &Coord, bound: &IntTuple) -> Boolean {
    let cs = coord.leaves();
    let bs = bound.leaves();
    assert_eq!(cs.len(), bs.len(), "elem_less: coordinate {coord:?} and bound {bound} differ in rank");
    let mut acc: Option<Boolean> = None;
    for (c, &b) in cs.into_iter().zip(bs) {
        if let Some(cv) = c.bound()
            && cv <= b
        {
            continue;
        }
        if let Some(cv) = c.as_static() {
            if cv >= b {
                return false.into_value();
            }
            continue;
        }
        let p = c.clone().into_value().lt(b);
        acc = Some(match acc {
            Some(a) => a & p,
            None => p,
        });
    }
    acc.unwrap_or_else(|| true.into_value())
}

#[track_caller]
pub(crate) fn load_vector<T: Numeric>(buffer: BufferId, off: IndexExpr, n: usize) -> Vec<ValueId> {
    let index = off.into_value().checked();
    load_vector_at::<T>(buffer, index, n)
}

#[track_caller]
pub(crate) fn load_vector_at<T: Numeric>(buffer: BufferId, index: ValueId, n: usize) -> Vec<ValueId> {
    let loc = Location::caller();
    builder::with(|b| {
        let results: Vec<ValueId> = (0..n).map(|_| b.new_value(T::DTYPE)).collect();
        b.push(Op::Load { results: results.clone(), buffer, index }, loc);
        results
    })
}

#[track_caller]
pub(crate) fn store_vector(buffer: BufferId, off: IndexExpr, values: &[ValueId]) {
    let index = off.into_value().checked();
    store_vector_at(buffer, index, values)
}

#[track_caller]
pub(crate) fn store_vector_at(buffer: BufferId, index: ValueId, values: &[ValueId]) {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::Store { values: values.to_vec(), buffer, index }, loc));
}

/// Declare a global-memory tensor parameter. The host must pass a buffer of
/// at least `layout.cosize()` elements, 16-byte aligned.
#[track_caller]
pub fn param_tensor<T: Numeric>(name: &str, layout: Layout) -> Tensor<T> {
    let len = layout.cosize();
    let buffer = builder::with(|b| {
        b.declare_buffer(BufferDecl {
            name: name.to_string(),
            elem: T::DTYPE,
            len: Some(len),
            space: MemSpace::Global,
            align: 16,
        })
    });
    Tensor::new(Engine::Memory { buffer, space: MemSpace::Global }, layout)
}

/// Declare a global-memory parameter whose length is only known at launch.
#[track_caller]
pub fn param_ptr<T: Numeric>(name: &str) -> Ptr<T> {
    let buffer = builder::with(|b| {
        b.declare_buffer(BufferDecl {
            name: name.to_string(),
            elem: T::DTYPE,
            len: None,
            space: MemSpace::Global,
            align: 16,
        })
    });
    Ptr { buffer, space: MemSpace::Global, offset: IndexExpr::Static(0), _t: PhantomData }
}

/// A typed position in global or shared memory.
pub struct Ptr<T> {
    buffer: BufferId,
    space: MemSpace,
    offset: IndexExpr,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        Ptr { buffer: self.buffer, space: self.space, offset: self.offset.clone(), _t: PhantomData }
    }
}

impl<T> fmt::Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ptr<{:?}>({:?} + {})", self.space, self.buffer, self.offset)
    }
}

impl<T: Numeric> Ptr<T> {
    /// The pointer `n` elements further on.
    pub fn offset(&self, n: impl Into<IndexExpr>) -> Ptr<T> {
        Ptr { offset: self.offset.clone() + n.into(), ..self.clone() }
    }

    /// The same address as a pointer to `U`. The address must be provably
    /// aligned to `U`.
    #[track_caller]
    pub fn cast<U: Numeric>(&self) -> Ptr<U> {
        let (from, to) = (T::DTYPE.bytes(), U::DTYPE.bytes());
        let bytes = self.offset.clone() * from;
        assert!(
            bytes.is_multiple_of(to),
            "cute: cannot prove that {} is aligned to {}{}",
            self.offset,
            U::DTYPE,
            builder::var_value_note(&self.offset)
        );
        Ptr { buffer: self.buffer, space: self.space, offset: bytes / to, _t: PhantomData }
    }

    /// Read the element at this address.
    #[track_caller]
    pub fn load(&self) -> Value<T> {
        Value::from_id(load_vector::<T>(self.buffer, self.offset.clone(), 1)[0])
    }

    /// Write the element at this address.
    #[track_caller]
    pub fn store(&self, value: impl IntoValue<T>) {
        store_vector(self.buffer, self.offset.clone(), &[value.into_value().checked()]);
    }

    pub(crate) fn parts(&self) -> (BufferId, IndexExpr) {
        (self.buffer, self.offset.clone())
    }
}

/// A tensor of `layout` starting at `ptr` (CuTe's `make_tensor`).
pub fn make_tensor<T: Numeric>(ptr: &Ptr<T>, layout: Layout) -> Tensor<T> {
    Tensor {
        offset: ptr.offset.clone(),
        ..Tensor::new(Engine::Memory { buffer: ptr.buffer, space: ptr.space }, layout)
    }
}

/// Allocate `count` elements of shared memory aligned to `align` bytes and
/// return a pointer to them (CuTe DSL's `cute.arch.alloc_smem`).
#[track_caller]
pub fn alloc_smem<T: Numeric>(count: usize, align: usize) -> Ptr<T> {
    assert!(align.is_power_of_two(), "cute: shared memory alignment {align} is not a power of two");
    let buffer = builder::with(|b| {
        let name = format!("smem{}", b.shared_buffer_count());
        b.declare_buffer(BufferDecl { name, elem: T::DTYPE, len: Some(count), space: MemSpace::Shared, align })
    });
    Ptr { buffer, space: MemSpace::Shared, offset: IndexExpr::Static(0), _t: PhantomData }
}

/// A per-thread array that, unlike a register fragment, can be indexed with
/// runtime coordinates (CuTe DSL's `make_rmem_tensor`). It lives on the
/// thread's stack.
#[track_caller]
pub fn make_rmem_tensor<T: Numeric>(layout: Layout) -> Tensor<T> {
    let len = layout.cosize();
    let buffer = builder::with(|b| {
        let name = format!("stack{}", b.local_buffer_count());
        b.declare_buffer(BufferDecl { name, elem: T::DTYPE, len: Some(len), space: MemSpace::Local, align: 16 })
    });
    Tensor::new(Engine::Memory { buffer, space: MemSpace::Local }, layout)
}

/// Declare a scalar kernel parameter.
#[track_caller]
pub fn param_scalar<T: Numeric>(name: &str) -> Value<T> {
    Value::from_id(builder::with(|b| b.declare_scalar_param(name, T::DTYPE)))
}

/// Declare a non-negative `i32` parameter that the host promises is a
/// multiple of `multiple_of`, such as a problem size that is a whole number
/// of tiles. Layouts can use it as a stride with that divisibility known,
/// which is what proves vector and `cp.async` accesses aligned. Launches
/// check the promise.
#[track_caller]
pub fn param_extent(name: &str, multiple_of: usize) -> Int32 {
    Value::from_id(builder::with(|b| b.declare_extent_param(name, multiple_of)))
}

/// Layouts accepted for shared-memory tensors: plain or swizzled.
pub trait SmemLayout {
    fn into_parts(self) -> (Layout, Option<Swizzle>, usize);
}

impl SmemLayout for Layout {
    fn into_parts(self) -> (Layout, Option<Swizzle>, usize) {
        let cosize = self.cosize();
        (self, None, cosize)
    }
}

impl SmemLayout for ComposedLayout {
    fn into_parts(self) -> (Layout, Option<Swizzle>, usize) {
        let cosize = self.cosize();
        (self.layout, Some(self.swizzle), cosize)
    }
}

/// Allocate a shared-memory tensor (128-byte aligned).
#[track_caller]
pub fn make_smem_tensor<T: Numeric>(layout: impl SmemLayout) -> Tensor<T> {
    let (layout, swizzle, len) = layout.into_parts();
    let buffer = builder::with(|b| {
        let name = format!("smem{}", b.shared_buffer_count());
        b.declare_buffer(BufferDecl { name, elem: T::DTYPE, len: Some(len), space: MemSpace::Shared, align: 128 })
    });
    Tensor { swizzle, ..Tensor::new(Engine::Memory { buffer, space: MemSpace::Shared }, layout) }
}

/// A register fragment with a compact column-major layout of `shape`.
pub fn make_fragment<T: Numeric>(shape: impl Into<IntTuple>) -> Tensor<T> {
    make_fragment_with_layout(Layout::col_major(shape))
}

/// A register fragment with an explicit layout.
pub fn make_fragment_with_layout<T: Numeric>(layout: Layout) -> Tensor<T> {
    let slots = layout.cosize();
    Tensor::new(Engine::Register(RegisterStorage::new(vec![None; slots])), layout)
}

/// A register fragment with the same shape as `t`.
pub fn make_fragment_like<T: Numeric, U>(t: &Tensor<U>) -> Tensor<T> {
    make_fragment::<T>(t.shape().clone())
}

/// A tensor whose element at coordinate `c` is `c` itself (one index per
/// top-level mode). At most three modes, each smaller than 2^20.
pub fn make_identity_tensor(shape: impl Into<IntTuple>) -> Tensor<Ident> {
    let shape = shape.into();
    assert!(shape.rank() <= 3, "identity tensors have at most 3 modes");
    let modes: Vec<Layout> = shape
        .modes()
        .iter()
        .enumerate()
        .map(|(i, m)| {
            assert!(m.size() < IDENTITY_FIELD, "identity tensor mode {m} is too large");
            let field = IDENTITY_FIELD.pow(i as u32);
            Layout::new(m.clone(), m.compact_col_major().map(&mut |&d| d * field))
        })
        .collect();
    let layout = if shape.is_leaf() { modes[0].clone() } else { Layout::from_modes(modes) };
    Tensor::new(Engine::Identity(shape), layout)
}

/// The tile of `t` at `coord`: `zipped_divide(t, tiler)[(.., coord)]`.
/// Modes of the rest not fixed by `coord` (use `..`) are kept after the tile
/// modes (CuTe's `local_tile`).
pub fn local_tile<T>(t: &Tensor<T>, tiler: impl Into<Tiler>, coord: impl Into<SliceCoord>) -> Tensor<T> {
    let z = zipped_divide(t, tiler);
    let tile_modes = z.layout.mode(0).rank();
    let rest_modes = z.layout.mode(1).rank();
    let coord = match coord.into() {
        Tree::Branch(mut cs) => {
            cs.resize(rest_modes, Tree::Leaf(None));
            Tree::Branch(cs)
        }
        leaf => leaf,
    };
    z.slice(Tree::Branch(vec![Tree::Branch(vec![Tree::Leaf(None); tile_modes]), coord]))
}

/// [`local_tile`] keeping only the modes of a larger tiler selected by
/// `proj` (CuTe's `local_tile(t, tiler, coord, Step<_1, X, _1>{})`). For a
/// GEMM, `local_tile_proj(&mA, (BM, BN, BK), (m, n, ..), &[true, false, true])`
/// gives A's `(BM, BK, k)` tiles.
pub fn local_tile_proj<T>(
    t: &Tensor<T>,
    tiler: impl Into<IntTuple>,
    coord: impl Into<SliceCoord>,
    proj: &[bool],
) -> Tensor<T> {
    let tiler = tiler.into();
    let coord = coord.into();
    assert_eq!(tiler.rank(), proj.len(), "local_tile_proj: tiler and projection differ in rank");
    assert_eq!(coord.rank(), proj.len(), "local_tile_proj: coordinate and projection differ in rank");
    let keep = |i: usize| proj[i];
    let tiler: Vec<IntTuple> = (0..proj.len()).filter(|&i| keep(i)).map(|i| tiler.mode(i).clone()).collect();
    let coord: Vec<SliceCoord> = (0..proj.len()).filter(|&i| keep(i)).map(|i| coord.mode(i).clone()).collect();
    let tiler: Tiler = (&IntTuple::Branch(tiler)).into();
    local_tile(t, tiler, Tree::Branch(coord))
}

/// Partition `t` among the threads of `thr_layout`: thread `thr_idx` gets
/// the elements at its position in every `shape(thr_layout)`-sized tile
/// (CuTe's `local_partition`).
pub fn local_partition<T>(t: &Tensor<T>, thr_layout: &Layout, thr_idx: impl Into<IndexExpr>) -> Tensor<T> {
    let tiler: Tiler = (&thr_layout.shape().product_each()).into();
    let z = zipped_divide(t, tiler);
    let rest_modes = z.layout.mode(1).rank();
    let coord: SliceCoord = (&thr_layout.coord_of(thr_idx)).into();
    z.slice(Tree::Branch(vec![coord, Tree::Branch(vec![Tree::Leaf(None); rest_modes])]))
}

/// [`local_partition`] using only the modes of `thr_layout` selected by
/// `proj` (CuTe's `local_partition(t, tile, idx, Step<_1, X>{})`): for a
/// `16 x 16` thread layout, `proj = [true, false]` partitions the M mode of
/// an `(M, K)` tile among the 16 thread rows and keeps K whole.
pub fn local_partition_proj<T>(
    t: &Tensor<T>,
    thr_layout: &Layout,
    thr_idx: impl Into<IndexExpr>,
    proj: &[bool],
) -> Tensor<T> {
    assert_eq!(thr_layout.rank(), proj.len(), "local_partition_proj: thread layout and projection differ in rank");
    let coord = thr_layout.coord_of(thr_idx);
    let sizes = thr_layout.shape().product_each();
    let keep: Vec<usize> = (0..proj.len()).filter(|&i| proj[i]).collect();
    let tiler: Tiler =
        Tree::Branch(keep.iter().map(|&i| Tree::Leaf(Some(Layout::new(sizes.mode(i).clone(), 1)))).collect());
    let z = zipped_divide(t, tiler);
    let rest_modes = z.layout.mode(1).rank();
    let coord: SliceCoord =
        Tree::Branch(keep.iter().map(|&i| Tree::Leaf(Some(coord.mode(i).as_leaf().unwrap().clone()))).collect());
    z.slice(Tree::Branch(vec![coord, Tree::Branch(vec![Tree::Leaf(None); rest_modes])]))
}

impl<T: Numeric> Carry for Tensor<T> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        let storage = self.register_storage().expect("cute: only register tensors can be carried through loops");
        for i in 0..storage.len() {
            out.push(storage.read(i).unwrap_or_else(|| {
                panic!("cute: register tensor slot {i} is uninitialised; fill() it before carrying it through a loop")
            }));
        }
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let n = self.register_storage().unwrap().len();
        let slots: Vec<Option<ValueId>> = (0..n).map(|_| Some(values.next().expect("carry: too few values"))).collect();
        Tensor { engine: Engine::Register(RegisterStorage::new(slots)), ..self.clone() }
    }

    /// Register tensors are rebuilt with the layout, offset and swizzle of
    /// the one rebuilt from, so these must match.
    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let registers = |t: &Self| t.register_storage().map(|s| s.len());
        let same_registers = registers(self) == registers(other);
        if self.layout == other.layout && self.offset == other.offset && self.swizzle == other.swizzle && same_registers
        {
            return None;
        }
        let describe = |t: &Self| {
            let mut s = format!("layout {}", t.layout);
            if t.offset != IndexExpr::Static(0) {
                s += &format!(" at offset {}", t.offset);
            }
            if let Some(sw) = t.swizzle {
                s += &format!(" swizzled by {sw}");
            }
            if let (false, Some(n)) = (same_registers, registers(t)) {
                s += &format!(" in {n} registers");
            }
            s
        };
        Some(format!("a register tensor with {} became one with {}", describe(self), describe(other)))
    }
}

impl<T> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.engine {
            Engine::Memory { space: MemSpace::Global, .. } => "gmem",
            Engine::Memory { space: MemSpace::Shared, .. } => "smem",
            Engine::Memory { space: MemSpace::Local, .. } => "stack",
            Engine::Register(_) => "rmem",
            Engine::Identity(_) => "coord",
        };
        write!(f, "tensor<{kind}")?;
        if let Some(sw) = self.swizzle {
            write!(f, ", {sw}")?;
        }
        write!(f, "> {} + {}", self.layout, self.offset)
    }
}

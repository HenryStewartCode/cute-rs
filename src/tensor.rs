// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/tensor_impl.hpp and include/cute/int_tuple.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   and cutlass_compiler/cutegen/include/cutegen/layout.hpp,
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

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
//!   [`alloc_smem`]), optionally swizzled, and the dynamic region whose
//!   size is set at launch ([`get_dyn_smem`]),
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
//! Coordinate tensors ([`make_coord_tensor`], and the identity tensors of
//! [`make_identity_tensor`]) hold coordinates instead of data, and are used
//! to compute predicates for partial tiles. As in CuTe, a coordinate tensor
//! is an origin (an arithmetic tuple) and a layout, usually with scaled-basis
//! strides ([`make_identity_layout`](crate::make_identity_layout)): the
//! element at `c` is the origin plus the tuple the layout maps `c` to
//! ([`Tensor::coord`]). Slicing, tiling and partitioning act on the layout
//! and fold what they fix into the origin, so the extents may be runtime
//! values and the shape any hierarchy, and [`domain_offset`] shifts the
//! origin, by a negative amount too.
//!
//! # Runtime extents
//!
//! A memory tensor's extents may be runtime values, as CuTe DSL's framework
//! tensors are (`(?,?):(?,1)`): [`param_tensor`] and [`make_tensor`] take a
//! layout such as `Layout::row_major((m, n))` with `m` and `n` from
//! [`param_extent`], and slicing, [`local_tile`], [`local_partition`], the
//! divides and compositions, and the partitions of tiled copies and MMAs
//! carry them through: a tile of a static tiler has static extents, and the
//! rest modes that count the tiles have runtime ones (`ceil_div(m, 128)`).
//! [`size`](crate::size)`(&tensor, &[2])` gives such a mode's extent as an
//! [`IndexExpr`], which a [`for_range`](crate::for_range) can take as its
//! trip count. Offsets stay [`IndexExpr`]s, so accesses past a runtime
//! extent are only found by the simulator's bounds checks: a partial tile
//! needs a predicate.
//!
//! What a kernel unrolls when it is built stays static, and meeting a
//! runtime extent there panics, naming it: the per-thread chunks and rest
//! modes of [`copy`](crate::copy) and [`gemm`](crate::gemm), loops over every
//! element ([`Tensor::fill`], [`Tensor::values`], [`autovec_copy`](crate::autovec_copy)),
//! and the shapes of register fragments and the buffers that are sized when
//! the kernel is built (shared memory, per-thread arrays).

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::panic::Location;
use std::rc::Rc;

use crate::builder::{self, ValueFacts};
use crate::control::Carry;
use crate::ir::{BufferDecl, BufferId, DType, MemAttrs, MemSpace, Op, ValueId};
use crate::layout::{
    ComposedLayout, Coord, HasCosize, HasShape, IndexExpr, Inner, IntTuple, Layout, LayoutSource, Shape, SliceCoord,
    Stride, Swizzle, Tileable, Tiler, Tree, make_identity_layout, make_layout_like, tuple_sum, zipped_divide,
};
use crate::value::{Boolean, Int32, Int64, IntoValue, Numeric, Value, select};

pub use crate::masked::{gather, scatter};

/// Element marker for coordinate tensors (identity tensors among them).
#[derive(Clone, Copy, Debug)]
pub struct Ident;

/// The address spaces of CuTe DSL's `cutlass.AddressSpace` (re-exported as
/// `cute.AddressSpace`), an integer enum whose values are NVVM's address
/// space numbers: `AddressSpace::Smem as u32 == 3`. [`Tensor::memspace`]
/// and [`Ptr::memspace`] say where an object lives. cute-rs builds global
/// memory ([`param_tensor`]), shared memory ([`make_smem_tensor`],
/// [`alloc_smem`]) and register memory (fragments and
/// [`make_rmem_tensor`]'s per-thread arrays, which CuTe keeps in one space);
/// `Generic`, `Cmem`, `Tmem` and `Dsmem` exist for the roadmap's later
/// parts (generic pointers, distributed shared memory, tensor memory), and
/// nothing builds them yet.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum AddressSpace {
    /// `generic` (0): a generic address.
    Generic = 0,
    /// `gmem` (1): global memory.
    Gmem = 1,
    /// `smem` (3): the CTA's shared memory.
    Smem = 3,
    /// `cmem` (4): constant memory.
    Cmem = 4,
    /// `rmem` (5): registers and per-thread (local) memory.
    Rmem = 5,
    /// `tmem` (6): tensor memory (sm_100).
    Tmem = 6,
    /// `dsmem` (7): distributed shared memory, another CTA's in the cluster.
    Dsmem = 7,
}

impl AddressSpace {
    /// CuTe DSL's name for the space: `gmem`, `smem`, ...
    pub fn name(self) -> &'static str {
        match self {
            AddressSpace::Generic => "generic",
            AddressSpace::Gmem => "gmem",
            AddressSpace::Smem => "smem",
            AddressSpace::Cmem => "cmem",
            AddressSpace::Rmem => "rmem",
            AddressSpace::Tmem => "tmem",
            AddressSpace::Dsmem => "dsmem",
        }
    }
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The IR's memory spaces are a subset: global, shared, and per-thread
/// memory, which is CuTe's register memory.
impl From<MemSpace> for AddressSpace {
    fn from(space: MemSpace) -> AddressSpace {
        match space {
            MemSpace::Global => AddressSpace::Gmem,
            MemSpace::Shared => AddressSpace::Smem,
            MemSpace::Local => AddressSpace::Rmem,
            MemSpace::Raw(a) => a,
            MemSpace::Constant => AddressSpace::Cmem,
        }
    }
}

/// The IR memory space of an address space, for the spaces the IR has
/// (`Rmem` is per-thread memory); the space itself when it has none.
impl TryFrom<AddressSpace> for MemSpace {
    type Error = AddressSpace;

    fn try_from(space: AddressSpace) -> Result<MemSpace, AddressSpace> {
        match space {
            AddressSpace::Gmem => Ok(MemSpace::Global),
            AddressSpace::Smem => Ok(MemSpace::Shared),
            AddressSpace::Rmem => Ok(MemSpace::Local),
            other => Err(other),
        }
    }
}

/// What a tensor's elements start at (CuTe DSL's `Tensor.iterator`): a
/// pointer for a tensor in memory, the origin of a coordinate tensor, or,
/// for a register fragment (an array of SSA values, which has no address in
/// cute-rs), [`TensorIterator::Registers`].
#[derive(Clone, Debug)]
pub enum TensorIterator<T> {
    /// A tensor in global, shared or per-thread memory: its first element.
    Ptr(Ptr<T>),
    /// A coordinate tensor: the origin its layout's tuples are added to.
    Coord(Coord),
    /// A register fragment.
    Registers,
}

#[derive(Clone)]
pub(crate) enum Engine {
    Memory {
        buffer: BufferId,
        space: MemSpace,
        /// A tensor in another CTA's shared memory (made from a
        /// [`map_dsmem_ptr`](crate::map_dsmem_ptr) pointer, roadmap part
        /// m8-dsmem): that CTA's rank in the cluster.
        peer: Option<ValueId>,
    },
    Register(Rc<RegisterStorage>),
    /// A coordinate tensor: CuTe's arithmetic-tuple iterator, the origin
    /// that the tuple the layout gives is added to.
    ArithTuple(Coord),
    /// A WGMMA shared-memory operand (roadmap part m8-wgmma-core): CuTe's
    /// GMMA descriptor iterator. `desc` (a Uint64) describes the operand's
    /// first MMA tile; element `i` is `desc` plus the layout's offset of
    /// `i`, in units of 16 bytes ([`crate::nvgpu::warpgroup`]).
    SmemDesc {
        desc: ValueId,
    },
}

/// The registers of a fragment: the SSA value each holds, shared by every
/// view of the fragment.
pub(crate) struct RegisterStorage {
    slots: RefCell<Vec<Option<ValueId>>>,
    /// The builder's latest region serial when the fragment was created:
    /// the runtime loops opened after it see it as state from outside,
    /// which they do not carry.
    pub(crate) created: u64,
    /// The fragment holds the accumulators of warpgroup MMAs, which
    /// `warpgroup::fence` and `wait_group` fence ([`crate::nvgpu::warpgroup`]).
    pub(crate) wgmma_acc: std::cell::Cell<bool>,
    /// The fragment holds the register-sourced A operands of warpgroup
    /// MMAs, which `warpgroup::fence` fences (roadmap part
    /// m8-wgmma-variants).
    pub(crate) wgmma_a: std::cell::Cell<bool>,
}

impl RegisterStorage {
    fn new(slots: Vec<Option<ValueId>>) -> Rc<RegisterStorage> {
        Rc::new(RegisterStorage {
            slots: RefCell::new(slots),
            created: builder::latest_serial(),
            wgmma_acc: std::cell::Cell::new(false),
            wgmma_a: std::cell::Cell::new(false),
        })
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
    pub(crate) fn read(self: &Rc<Self>, i: usize) -> Option<ValueId> {
        let v = self.peek(i);
        if v.is_some() {
            let loc = Location::caller();
            builder::with(|b| b.fragment_read((Rc::as_ptr(self) as usize, i), self.created, loc));
        }
        v
    }

    /// Write register `i` at the caller's location.
    #[track_caller]
    pub(crate) fn write(self: &Rc<Self>, i: usize, v: ValueId) {
        let n = self.len();
        assert!(i < n, "cute: register tensor index {i} out of range ({n} slots)");
        let loc = Location::caller();
        builder::with(|b| b.fragment_write((Rc::as_ptr(self) as usize, i), self.created, loc));
        self.slots.borrow_mut()[i] = Some(v);
    }
}

/// What a memory tensor's offsets go through before they address an
/// element: nothing, a swizzle (of element offsets, relative to the start of
/// the buffer), or a layout (a tensor over the composed layout `inner o
/// offset o outer`), whose results are added to `base`, the pointer's
/// offset.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum TensorInner {
    Identity,
    /// The swizzle on element offsets, and the same swizzle on bytes (what
    /// the tensor's pointer carries), which elements of a width other than a
    /// power of two (FP6) do not have.
    Swizzle {
        elem: Swizzle,
        bytes: Option<Swizzle>,
        /// Whether the swizzle came from the layout (a swizzled composed
        /// layout, CuTe's PISL) rather than the pointer (PDSL).
        in_layout: bool,
    },
    Layout {
        inner: Rc<Layout>,
        base: IndexExpr,
    },
}

impl TensorInner {
    /// The swizzle `elem` on the element offsets of a tensor of `T`.
    #[track_caller]
    fn swizzle<T: Numeric>(elem: Swizzle) -> TensorInner {
        if elem.is_identity() {
            return TensorInner::Identity;
        }
        let bits = T::DTYPE.storage_bits();
        let bytes = (bits.is_power_of_two() && (bits >= 8 || elem.base >= (8 / bits).trailing_zeros()))
            .then(|| elem.to_bytes(bits));
        TensorInner::Swizzle { elem, bytes, in_layout: true }
    }
}

/// A layout over some storage. A tensor in memory holds the general
/// composed layout `inner o offset o outer` of CuTe: [`layout`](Tensor::layout)
/// is the outer layout, [`base_offset`](Tensor::base_offset) the offset
/// (which also holds the pointer's, except for a layout inner), and the
/// inner function is a swizzle ([`swizzle`](Tensor::swizzle)) or a layout
/// ([`inner_layout`](Tensor::inner_layout)); [`composed_layout`](Tensor::composed_layout)
/// gives the three together.
pub struct Tensor<T> {
    pub(crate) engine: Engine,
    pub(crate) layout: Layout,
    pub(crate) offset: IndexExpr,
    pub(crate) inner: TensorInner,
    /// The predication origin (roadmap part m11-cute-ext-core): for a
    /// tensor viewed from a global-memory tensor, the origin's shape and the
    /// view operations since, which [`crate::experimental`]'s copies replay
    /// on an identity tensor to bound their accesses.
    pub(crate) origin: Option<PredOrigin>,
    _t: PhantomData<fn() -> T>,
}

/// A view operation recorded on a [`PredOrigin`]: a layout function (a
/// divide, a composition, a partition), a slice, or a domain offset.
#[derive(Clone)]
pub(crate) enum ViewOp {
    Layout(Rc<dyn Fn(&Layout) -> Layout>),
    Slice(SliceCoord),
    Offset(Coord),
}

pub(crate) struct ViewNode {
    op: ViewOp,
    prev: Option<Rc<ViewNode>>,
}

/// Where a tensor's automatic out-of-bounds predicates come from
/// (cute_ext's predicated tensor origin): `Tracked` holds the shape of the
/// origin tensor and the view operations applied since, as a shared list;
/// `Lost` marks a view made by an operation that does not record itself
/// (a coalesce, a product, a raw layout swap), whose coordinates in the
/// origin are unknown.
#[derive(Clone)]
pub(crate) enum PredOrigin {
    /// `root` is the origin tensor's own layout, offset and inner (recorded
    /// at the first view operation; `None` while the tensor is the origin
    /// itself), which cute_ext's TMA operations describe.
    Tracked {
        shape: Rc<Shape>,
        root: Option<Rc<OriginRoot>>,
        last: Option<Rc<ViewNode>>,
    },
    Lost,
}

/// The layout, offset and inner function of a predication origin tensor
/// (roadmap part m11-cute-ext-tma).
pub(crate) struct OriginRoot {
    pub(crate) layout: Layout,
    pub(crate) offset: IndexExpr,
    pub(crate) inner: TensorInner,
}

impl PredOrigin {
    pub(crate) fn fresh(shape: &Shape) -> PredOrigin {
        PredOrigin::Tracked { shape: Rc::new(shape.clone()), root: None, last: None }
    }

    /// The origin after the view operation `op` applied to `viewed` (the
    /// tensor this origin belongs to).
    pub(crate) fn push<T>(&self, viewed: &Tensor<T>, op: ViewOp) -> PredOrigin {
        match self {
            PredOrigin::Tracked { shape, root, last } => {
                let root = match (root, last) {
                    (Some(r), _) => Some(r.clone()),
                    (None, None) => Some(Rc::new(OriginRoot {
                        layout: viewed.layout.clone(),
                        offset: viewed.offset.clone(),
                        inner: viewed.inner.clone(),
                    })),
                    (None, Some(_)) => None,
                };
                PredOrigin::Tracked {
                    shape: shape.clone(),
                    root,
                    last: Some(Rc::new(ViewNode { op, prev: last.clone() })),
                }
            }
            PredOrigin::Lost => PredOrigin::Lost,
        }
    }

    /// The view's coordinates in the origin (an identity tensor over the
    /// origin's shape put through the recorded operations) and the origin's
    /// shape; `None` for a lost origin.
    #[track_caller]
    pub(crate) fn replay(&self) -> Option<(Tensor<Ident>, Shape)> {
        let PredOrigin::Tracked { shape, .. } = self else { return None };
        let coords = self.replay_on(make_identity_tensor((**shape).clone()))?;
        Some((coords, (**shape).clone()))
    }

    /// The recorded view operations applied to `start`, a coordinate tensor
    /// over the origin's shape (an identity tensor, or a TMA tensor);
    /// `None` for a lost origin.
    #[track_caller]
    pub(crate) fn replay_on(&self, start: Tensor<Ident>) -> Option<Tensor<Ident>> {
        let PredOrigin::Tracked { last, .. } = self else { return None };
        let mut ops = Vec::new();
        let mut node = last.clone();
        while let Some(n) = node {
            ops.push(n.op.clone());
            node = n.prev.clone();
        }
        let mut coords = start;
        for op in ops.into_iter().rev() {
            coords = match op {
                ViewOp::Layout(f) => {
                    let l = f(&coords.layout);
                    Tensor { layout: l, ..coords }
                }
                ViewOp::Slice(c) => coords.slice(c),
                ViewOp::Offset(c) => domain_offset(c, &coords),
            };
        }
        Some(coords)
    }

    /// The origin tensor's root record, when views have been made from it;
    /// `None` for the origin itself or a lost origin.
    pub(crate) fn root(&self) -> Option<&OriginRoot> {
        match self {
            PredOrigin::Tracked { root: Some(r), .. } => Some(r),
            _ => None,
        }
    }

    /// Whether no view operation has been recorded (the tensor is the origin).
    pub(crate) fn is_fresh(&self) -> bool {
        matches!(self, PredOrigin::Tracked { last: None, .. })
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            engine: self.engine.clone(),
            layout: self.layout.clone(),
            offset: self.offset.clone(),
            inner: self.inner.clone(),
            origin: self.origin.clone(),
            _t: PhantomData,
        }
    }
}

impl<T> Tileable for Tensor<T> {
    fn layout_ref(&self) -> &Layout {
        &self.layout
    }

    /// The tensor with a new layout, as dividing, composing, tiling and
    /// partitioning give it. A memory tensor and a coordinate tensor take
    /// runtime extents and strides. A register fragment keeps static
    /// extents: a layout with a runtime extent (from a tiler with runtime
    /// extents, say) panics, naming it.
    #[track_caller]
    fn with_layout(&self, layout: Layout) -> Self {
        if let Engine::Register(_) = &self.engine {
            require_static_extents(&layout, "a register fragment", FRAGMENT_EXTENTS);
        }
        // A layout given whole cannot be replayed on the origin's
        // coordinates: the view loses its predication origin.
        Tensor { layout, origin: self.origin.as_ref().map(|_| PredOrigin::Lost), ..self.clone() }
    }

    /// The tensor with the layout `f` makes of its layout; the predication
    /// origin records `f`.
    #[track_caller]
    fn map_layout<F: Fn(&Layout) -> Layout + 'static>(&self, layout: Layout, f: F) -> Self {
        let origin = self.origin.as_ref().map(|o| o.push(self, ViewOp::Layout(Rc::new(f))));
        Tensor { origin, ..self.with_layout(layout) }
    }
}

/// Why a register fragment's extents are static, as a refusal says it.
const FRAGMENT_EXTENTS: &str = "register fragments have static shapes (a fragment is an array of registers), so they \
                                are divided, composed and partitioned by tilers of static extents";

/// Panic, naming the leaf, if `layout`, which `what` (a register fragment,
/// an identity tensor) would take, has a runtime extent; `why` says why it
/// must not.
#[track_caller]
fn require_static_extents(layout: &Layout, what: &str, why: &str) {
    if let Some((path, leaf)) = layout.shape().first_dynamic() {
        panic!(
            "cute: {what} cannot take the layout {layout}, which has the dynamic extent {leaf:?} at mode {path:?}: {why}"
        );
    }
}

/// A tensor's layout-like functions read its layout, before any swizzle.
impl<T> LayoutSource for Tensor<T> {
    fn source_layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T> HasShape for Tensor<T> {
    type Leaf = IndexExpr;

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }
}

/// The codomain size of the tensor's layout at a mode path (CuTe DSL's
/// `cute.cosize(tensor, mode=...)`), an [`IndexExpr`] that is static when
/// the extents and strides it depends on are. The base offset (of a slice
/// or a partition) does not count, and a swizzle does not either, as in
/// CuTe, where a swizzled layout's cosize is its outer layout's.
///
/// A coordinate tensor's codomain is coordinates, so its cosize is that of
/// its layout of basis strides, CuTe's `size(coshape)`: the product, over
/// the coordinate's modes, of one past the largest coordinate in that mode
/// (the origin does not count). A `(8,8)` identity tensor has cosize 64,
/// and a part of it holding the coordinates `(r, 0..4)` has cosize 4,
/// whatever `r` is.
impl<T> HasCosize for Tensor<T> {
    #[track_caller]
    fn cosize_at(&self, mode: &[usize]) -> IndexExpr {
        self.layout.cosize_at(mode)
    }
}

impl<T> Tensor<T> {
    pub(crate) fn new(engine: Engine, layout: Layout) -> Self {
        // Every global-memory tensor (a kernel argument, or one made on a
        // global pointer) starts a predication origin.
        let origin =
            matches!(engine, Engine::Memory { space: MemSpace::Global, .. }).then(|| PredOrigin::fresh(layout.shape()));
        Tensor { engine, layout, offset: IndexExpr::Static(0), inner: TensorInner::Identity, origin, _t: PhantomData }
    }

    /// This tensor as the predication origin of the views made from it
    /// (cute_ext's `predicated_tensor_origin`): their automatic predicates
    /// bound them by this tensor's shape.
    pub(crate) fn with_fresh_origin(&self) -> Tensor<T> {
        Tensor { origin: Some(PredOrigin::fresh(self.layout.shape())), ..self.clone() }
    }

    /// The predication origin this tensor was viewed from, as a tensor
    /// (roadmap part m11-cute-ext-tma): the origin's engine, layout, offset
    /// and inner, without an origin of its own. `None` without a tracked
    /// origin.
    pub(crate) fn origin_root(&self) -> Option<Tensor<T>> {
        let origin = self.origin.as_ref()?;
        let (layout, offset, inner) = if origin.is_fresh() {
            (self.layout.clone(), self.offset.clone(), self.inner.clone())
        } else {
            let r = origin.root()?;
            (r.layout.clone(), r.offset.clone(), r.inner.clone())
        };
        Some(Tensor { engine: self.engine.clone(), layout, offset, inner, origin: None, _t: PhantomData })
    }

    /// The layout from coordinates to element offsets.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The tensor's shape, whose extents may be runtime values (printed
    /// `?`). [`shape`](crate::shape)`(&tensor, &[i])` is the sub-shape at a
    /// mode path, and [`size`](crate::size)`(&tensor, &[i])` its size as an
    /// [`IndexExpr`].
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Number of elements, for code that needs it when the kernel is built
    /// (a Rust loop over the elements, say). Panics, naming the dynamic
    /// extent, when it is only known at run time; [`size`](crate::size)`(&tensor,
    /// &[])` gives an [`IndexExpr`] instead, static when it folds.
    #[track_caller]
    pub fn size(&self) -> usize {
        if let Some(size) = self.layout.shape().product().as_static() {
            return size;
        }
        let Some((path, leaf)) = self.layout.shape().first_dynamic() else {
            unreachable!("a dynamic size has a dynamic extent")
        };
        panic!(
            "Tensor::size: tensor {} has the dynamic extent {leaf:?} at mode {path:?}, so its size is only known at \
             run time; use cute::size(&tensor, &[]), an IndexExpr, where a static size is not needed",
            self.layout
        )
    }

    /// The number of elements of a tensor that `what` visits one by one
    /// when the kernel is built: panics, naming the dynamic extent, if it
    /// is only known at run time.
    #[track_caller]
    fn unrolled_size(&self, what: &str) -> usize {
        if let Some(size) = self.layout.shape().product().as_static() {
            return size;
        }
        let Some((path, leaf)) = self.layout.shape().first_dynamic() else {
            unreachable!("a dynamic size has a dynamic extent")
        };
        panic!(
            "{what}: tensor {} has the dynamic extent {leaf:?} at mode {path:?}, but {what} visits every element when \
             the kernel is built, so its size must be static: for a runtime number of tiles, write a for_range over \
             the rest mode (cute::size(&tensor, &[mode]) is its extent) and work on one tile per iteration",
            self.layout
        )
    }

    /// Number of top-level modes (CuTe DSL's `cute.rank`; [`rank`](crate::rank)
    /// takes a mode path).
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// Nesting depth of the shape: 0 for an integer, 1 for a flat tuple
    /// (CuTe DSL's `cute.depth`).
    pub fn depth(&self) -> usize {
        self.layout.depth()
    }

    /// The base element offset (before the layout is applied).
    pub fn base_offset(&self) -> &IndexExpr {
        &self.offset
    }

    /// The swizzle applied to offsets (swizzled shared memory), on element
    /// offsets from the start of the buffer.
    pub fn swizzle(&self) -> Option<Swizzle> {
        match self.inner {
            TensorInner::Swizzle { elem, .. } => Some(elem),
            _ => None,
        }
    }

    /// Whether the tensor's swizzle came from a swizzled composed layout
    /// (CuTe's PISL) rather than a swizzled pointer (PDSL).
    pub(crate) fn swizzle_in_layout(&self) -> bool {
        matches!(self.inner, TensorInner::Swizzle { in_layout: true, .. })
    }

    /// The inner layout of a tensor over a composed layout whose inner
    /// function is a layout.
    pub fn inner_layout(&self) -> Option<&Layout> {
        match &self.inner {
            TensorInner::Layout { inner, .. } => Some(inner),
            _ => None,
        }
    }

    /// The tensor's layout as CuTe's composed layout `inner o offset o
    /// outer` (CuTe DSL's `tensor.layout` of a swizzled tensor): the
    /// swizzle (the identity `S<0,4,3>` for a plain tensor) or the inner
    /// layout, the base offset, and [`layout`](Tensor::layout).
    pub fn composed_layout(&self) -> ComposedLayout {
        let inner = match &self.inner {
            TensorInner::Identity => Inner::Swizzle(Swizzle::identity()),
            TensorInner::Swizzle { elem, .. } => Inner::Swizzle(*elem),
            TensorInner::Layout { inner, .. } => Inner::Layout((**inner).clone()),
        };
        ComposedLayout { inner, offset: Tree::Leaf(self.offset.clone()), outer: self.layout.clone() }
    }

    /// The number of consecutive offsets that stay consecutive elements: a
    /// swizzle's run, 1 for a layout inner, unbounded otherwise.
    pub(crate) fn contiguous_run(&self) -> Option<usize> {
        match &self.inner {
            TensorInner::Identity => None,
            TensorInner::Swizzle { elem, .. } => Some(elem.contiguous_run()),
            TensorInner::Layout { .. } => Some(1),
        }
    }

    /// The element offset of `linear`, an offset before the inner function
    /// (the base offset plus what the layout gives).
    pub(crate) fn physical_offset(&self, linear: IndexExpr) -> IndexExpr {
        match &self.inner {
            TensorInner::Identity => linear,
            TensorInner::Swizzle { elem, .. } => elem.apply_expr(linear),
            TensorInner::Layout { inner, base } => {
                base.clone() + inner.offset_checked(&Tree::Leaf(linear), "the inner layout of a composed tensor")
            }
        }
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

    /// The address space of the tensor's elements (CuTe DSL's
    /// `Tensor.memspace`): [`AddressSpace::Gmem`], [`AddressSpace::Smem`],
    /// or [`AddressSpace::Rmem`] for a register fragment and for a
    /// [`make_rmem_tensor`] array alike. Panics for a coordinate tensor,
    /// which holds coordinates, not memory, as CuTe DSL raises.
    #[track_caller]
    pub fn memspace(&self) -> AddressSpace {
        match self.engine {
            Engine::Memory { peer: Some(_), .. } => AddressSpace::Dsmem,
            Engine::Memory { space, .. } => space.into(),
            Engine::Register(_) => AddressSpace::Rmem,
            Engine::SmemDesc { .. } => AddressSpace::Smem,
            Engine::ArithTuple(_) => panic!(
                "Tensor::memspace: tensor {} is a coordinate tensor, which holds coordinates, not elements in \
                 memory, so it has no address space",
                self.layout
            ),
        }
    }

    /// The stride of the tensor's layout (CuTe DSL's `Tensor.stride`).
    /// Panics for a swizzled tensor, whose layout is a composed one, as CuTe
    /// DSL raises; `layout().stride()` is the stride before the swizzle.
    #[track_caller]
    pub fn stride(&self) -> &Stride {
        if let TensorInner::Layout { inner, .. } = &self.inner {
            panic!(
                "Tensor::stride: tensor {} is over a composed layout with the inner layout {inner}, which has no \
                 stride; use tensor.layout().stride() for the stride of its outer layout",
                self.layout
            );
        }
        if let Some(sw) = self.swizzle() {
            panic!(
                "Tensor::stride: tensor {} is swizzled by {sw}, so its layout is a composed layout, which has no \
                 stride; use tensor.layout().stride() for the stride before the swizzle",
                self.layout
            );
        }
        self.layout.stride()
    }

    /// The position of the leading dimension of the tensor's layout (CuTe
    /// DSL's `leading_dim(tensor.shape, tensor.stride)`): the first mode,
    /// depth first, whose stride is a static 1 on an extent other than 1, as
    /// a path (`[0]` for a layout of one integer mode), or `None`. See
    /// [`Layout::leading_dim`].
    pub fn leading_dim(&self) -> Option<Vec<usize>> {
        self.layout.leading_dim()
    }

    /// What the tensor's elements start at (CuTe DSL's `Tensor.iterator`):
    /// a [`Ptr`] for a tensor in memory, the origin for a coordinate tensor,
    /// [`TensorIterator::Registers`] for a register fragment.
    pub fn iterator(&self) -> TensorIterator<T> {
        match &self.engine {
            Engine::Memory { .. } => TensorIterator::Ptr(self.ptr()),
            Engine::Register(_) | Engine::SmemDesc { .. } => TensorIterator::Registers,
            Engine::ArithTuple(origin) => TensorIterator::Coord(origin.clone()),
        }
    }

    /// The largest power of two of bytes that the address of the first
    /// element provably is a multiple of, limited by the swizzle's
    /// contiguous run (CuTe's `max_alignment` of a swizzled pointer):
    /// `None` for a register fragment or a coordinate tensor.
    pub(crate) fn iterator_alignment(&self, elem_bytes: usize) -> Option<usize> {
        let Engine::Memory { buffer, .. } = self.engine else { return None };
        Some(match &self.inner {
            TensorInner::Identity => ptr_alignment(buffer, &self.offset, elem_bytes),
            TensorInner::Swizzle { elem, .. } => {
                ptr_alignment(buffer, &self.offset, elem_bytes).min(elem.max_alignment() * elem_bytes)
            }
            TensorInner::Layout { base, .. } => ptr_alignment(buffer, base, elem_bytes).min(elem_bytes.max(1)),
        })
    }

    /// The buffer of a memory tensor. Panics for a tensor in another CTA's
    /// shared memory, which only the accesses that know the peer reach
    /// ([`Tensor::buffer_and_peer`]).
    #[track_caller]
    pub(crate) fn buffer(&self) -> Option<BufferId> {
        match self.engine {
            Engine::Memory { buffer, peer, .. } => {
                refuse_dsmem(peer, &format!("the tensor {self:?}"));
                Some(buffer)
            }
            _ => None,
        }
    }

    /// The buffer of a memory tensor and, for one in another CTA's shared
    /// memory, that CTA's rank (roadmap part m8-dsmem).
    pub(crate) fn buffer_and_peer(&self) -> Option<(BufferId, Option<ValueId>)> {
        match self.engine {
            Engine::Memory { buffer, peer, .. } => Some((buffer, peer)),
            _ => None,
        }
    }

    /// The CTA of the cluster whose shared memory this tensor is in, for a
    /// tensor made from a [`map_dsmem_ptr`](crate::map_dsmem_ptr) pointer;
    /// `None` otherwise.
    pub fn dsmem_peer(&self) -> Option<Int32> {
        match self.engine {
            Engine::Memory { peer, .. } => peer.map(Value::from_id),
            _ => None,
        }
    }

    /// A pointer to the tensor's first element (CuTe DSL's
    /// `tensor.iterator`). Memory tensors only. A swizzled tensor's pointer
    /// carries the swizzle, in bytes as CuTe DSL's pointer swizzles are
    /// ([`Ptr::swizzle`]), so [`make_tensor`] on it gives the swizzled
    /// tensor back. A tensor over a composed layout with a layout inner
    /// gives the pointer the inner layout's offsets start from.
    #[track_caller]
    pub fn ptr(&self) -> Ptr<T> {
        match self.engine {
            Engine::Memory { buffer, space, peer } => {
                let (offset, swizzle) = match &self.inner {
                    TensorInner::Identity => (self.offset.clone(), None),
                    TensorInner::Swizzle { elem, bytes, .. } => {
                        let Some(bytes) = bytes else {
                            panic!(
                                "Tensor::ptr: the swizzle {elem} of tensor {} acts within bytes, so no pointer \
                                 swizzle (on bytes) carries it",
                                self.layout
                            )
                        };
                        (self.offset.clone(), Some(*bytes))
                    }
                    TensorInner::Layout { base, .. } => (base.clone(), None),
                };
                Ptr { buffer, space, offset, swizzle, peer, _t: PhantomData }
            }
            _ => panic!("cute: only memory tensors have an address"),
        }
    }

    /// Slice with `coord`; `..` keeps a mode (CuTe's `tensor[None, 0]`).
    /// A coordinate tensor adds the tuple of the fixed modes to its origin.
    #[track_caller]
    pub fn slice(&self, coord: impl Into<SliceCoord>) -> Tensor<T> {
        let coord = coord.into();
        if let Engine::ArithTuple(origin) = &self.engine {
            let (layout, off) = self.layout.slice_and_offset_in(&coord, "Tensor::slice");
            return Tensor { engine: Engine::ArithTuple(tuple_sum(origin, &off)), layout, ..self.clone() };
        }
        let (layout, off) = self.layout.slice_checked(&coord, "Tensor::slice");
        let origin = self.origin.as_ref().map(|o| o.push(self, ViewOp::Slice(coord.clone())));
        Tensor { layout, offset: self.offset.clone() + off, origin, ..self.clone() }
    }

    /// Compose the layout with a tiler (by mode for tuples).
    #[track_caller]
    pub fn compose(&self, tiler: impl Into<Tiler>) -> Tensor<T> {
        let tiler = tiler.into();
        let layout = self.layout.compose_tiler(&tiler);
        self.map_layout(layout, move |l| l.compose_tiler(&tiler))
    }

    /// Group modes `[begin, end)` into one.
    #[track_caller]
    pub fn group_modes(&self, begin: usize, end: usize) -> Tensor<T> {
        self.map_layout(self.layout.group(begin, end), move |l| l.group(begin, end))
    }

    /// Physical element offset of `coord` (swizzle applied).
    #[track_caller]
    pub fn element_offset(&self, coord: &Coord) -> IndexExpr {
        self.element_offset_in(coord, "Tensor::element_offset")
    }

    /// [`element_offset`](Tensor::element_offset) for `what`, which a
    /// coordinate that does not fit the layout's shape is reported as.
    #[track_caller]
    fn element_offset_in(&self, coord: &Coord, what: &str) -> IndexExpr {
        if let Engine::ArithTuple(_) = self.engine {
            panic!("{what}: a coordinate tensor holds coordinates, not elements in memory; use coord()");
        }
        let linear = self.offset.clone() + self.layout.offset_checked(coord, what);
        self.physical_offset(linear)
    }

    /// The same tensor with the element type `U`, which the caller knows has
    /// the same [`DType`] as `T`.
    pub(crate) fn retyped<U>(&self) -> Tensor<U> {
        Tensor {
            engine: self.engine.clone(),
            layout: self.layout.clone(),
            offset: self.offset.clone(),
            inner: self.inner.clone(),
            origin: self.origin.clone(),
            _t: PhantomData,
        }
    }

    pub(crate) fn register_storage(&self) -> Option<&Rc<RegisterStorage>> {
        match &self.engine {
            Engine::Register(s) => Some(s),
            _ => None,
        }
    }
}

/// Panic for an access of one element of a sub-byte type in memory, which
/// CuTe DSL refuses (`_check_can_dereference`): hardware moves whole bytes,
/// so sub-byte tensors are read and written in vectors of whole 32-bit
/// words.
#[track_caller]
fn refuse_scalar_sub_byte(t: DType, what: &str) {
    assert!(
        !t.is_word_packed(),
        "cute: {what} of one {} element: a sub-byte tensor cannot be dereferenced element by element (CuTe DSL's \
         _check_can_dereference); copy whole vectors of a multiple of 32 bits (copy, autovec_copy) between it and a \
         register fragment",
        t.cute_name()
    );
}

#[track_caller]
fn static_register_index(off: &IndexExpr) -> usize {
    let Some(i) = off.as_static() else {
        let note = builder::var_value_note(off);
        panic!("cute: register tensors must be indexed with static coordinates (got offset {off:?}){note}")
    };
    i
}

impl<T: Numeric> Tensor<T> {
    /// The element type (CuTe DSL's `Tensor.element_type`).
    pub fn element_type(&self) -> DType {
        T::DTYPE
    }

    /// The element type (CuTe DSL's `Tensor.dtype`), as
    /// [`element_type`](Tensor::element_type).
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// Read one element (CuTe's `tensor[coord]`).
    #[track_caller]
    pub fn load(&self, coord: impl Into<Coord>) -> Value<T> {
        let off = self.element_offset_in(&coord.into(), "Tensor::load");
        self.load_at(off)
    }

    /// Element `i` (in the layout's index order), or zero for a register
    /// never written (roadmap part m11-cute-ext-core: the current values of
    /// `experimental::elementwise`'s outputs).
    #[track_caller]
    pub(crate) fn load_or_zero(&self, i: usize) -> Value<T> {
        let off = self.element_offset_in(&Coord::from(i), "Tensor::load");
        if let Engine::Register(storage) = &self.engine
            && storage.peek(static_register_index(&off)).is_none()
        {
            let loc = Location::caller();
            return Value::from_id(builder::with(|b| b.constant(crate::ir::Literal::zero(T::DTYPE), loc)));
        }
        self.load_at(off)
    }

    #[track_caller]
    pub(crate) fn load_at(&self, off: IndexExpr) -> Value<T> {
        match &self.engine {
            Engine::Register(storage) => {
                let i = static_register_index(&off);
                let v = storage.read(i);
                let Some(v) = v else { panic!("cute: register tensor element {i} read before it was written") };
                Value::from_id(v)
            }
            Engine::Memory { buffer, peer, .. } => {
                refuse_scalar_sub_byte(T::DTYPE, "a load");
                Value::from_id(load_vector_in::<T>(*buffer, off, 1, *peer)[0])
            }
            Engine::ArithTuple(_) => panic!("cute: coordinate tensors hold coordinates; use coord() instead of load()"),
            Engine::SmemDesc { .. } => panic!(
                "cute: a WGMMA operand fragment holds shared-memory matrix descriptors, not elements: gemm takes it \
                 as it is (Tensor::smem_desc gives a descriptor)"
            ),
        }
    }

    /// Write one element (CuTe's `tensor[coord] = value`).
    #[track_caller]
    pub fn store(&self, coord: impl Into<Coord>, value: impl IntoValue<T>) {
        let v = value.into_value();
        let off = self.element_offset_in(&coord.into(), "Tensor::store");
        self.store_at(off, v);
    }

    #[track_caller]
    pub(crate) fn store_at(&self, off: IndexExpr, v: Value<T>) {
        match &self.engine {
            Engine::Register(storage) => {
                let i = static_register_index(&off);
                storage.write(i, v.checked());
            }
            Engine::Memory { buffer, peer, .. } => {
                refuse_scalar_sub_byte(T::DTYPE, "a store");
                store_vector_in(*buffer, off, &[v.checked()], *peer)
            }
            Engine::ArithTuple(_) => panic!("cute: coordinate tensors are read-only"),
            Engine::SmemDesc { .. } => panic!("cute: a WGMMA operand fragment of matrix descriptors is read-only"),
        }
    }

    /// Every element, in colexicographic order. The tensor's size must be
    /// static.
    #[track_caller]
    pub fn values(&self) -> Vec<Value<T>> {
        (0..self.unrolled_size("Tensor::values")).map(|i| self.load(i)).collect()
    }

    /// Write every element from `values` (colexicographic order). The
    /// tensor's size must be static.
    #[track_caller]
    pub fn store_values(&self, values: &[Value<T>]) {
        assert_eq!(values.len(), self.unrolled_size("Tensor::store_values"), "cute: store_values length mismatch");
        for (i, v) in values.iter().enumerate() {
            self.store(i, *v);
        }
    }

    /// Set every element to `value`. The tensor's size must be static.
    #[track_caller]
    pub fn fill(&self, value: impl IntoValue<T>) {
        let v = value.into_value();
        for i in 0..self.unrolled_size("Tensor::fill") {
            self.store(i, v);
        }
    }

    /// Apply `f` to every element in place (register tensors) or with a
    /// load/store per element (memory). The tensor's size must be static.
    #[track_caller]
    pub fn map_in_place(&self, mut f: impl FnMut(Value<T>) -> Value<T>) {
        for i in 0..self.unrolled_size("Tensor::map_in_place") {
            let v = self.load(i);
            self.store(i, f(v));
        }
    }

    /// A new register fragment holding every element converted to `U`, as
    /// [`TensorSsa::to`](crate::TensorSsa::to) converts (with its paired
    /// FP8, FP4 and FP6 and fast Int8 and Int4 to BFloat16 forms). The
    /// tensor's shape must be static.
    #[track_caller]
    pub fn to<U: Numeric>(&self) -> Tensor<U> {
        require_static_fragment_shape(self, "Tensor::to");
        let out = make_compact_fragment_like::<U, T>(self);
        if !crate::tensor_ssa::has_wide_conversion(T::DTYPE, U::DTYPE) {
            for i in 0..self.size() {
                out.store(i, self.load(i).to::<U>());
            }
            return out;
        }
        // Converted as TensorSsa::to converts, with its wider instructions.
        let values: Vec<Value<T>> = (0..self.size()).map(|i| self.load(i)).collect();
        for (i, v) in crate::tensor_ssa::convert_values::<T, U>(&values, "Tensor::to").into_iter().enumerate() {
            out.store(i, v);
        }
        out
    }
}

impl Tensor<Ident> {
    /// The coordinate held at `coord` (CuTe DSL's `tensor[coord]` on a
    /// coordinate tensor): the origin plus the tuple the layout maps `coord`
    /// to ([`Layout::eval`]). For an identity tensor it is congruent with
    /// the shape of the tensor it was made for, whatever slices, tiles and
    /// partitions were taken of it since: `((1,0),2)` at index 7 of a
    /// `((2,1),3)` identity tensor, and `(r, c)` for a partition of an
    /// `(M, N)` one. Leaves may be runtime values, and negative where the
    /// tensor was shifted by a negative [`domain_offset`].
    #[track_caller]
    pub fn coord(&self, coord: impl Into<Coord>) -> Coord {
        let Engine::ArithTuple(origin) = &self.engine else { unreachable!("Tensor<Ident> is a coordinate tensor") };
        tuple_sum(origin, &self.layout.eval_checked(&coord.into(), "Tensor::coord"))
    }

    /// The origin of a coordinate tensor: the coordinate its layout's tuples
    /// are added to, 0 for a fresh identity tensor, and what slices and
    /// [`domain_offset`] have added since.
    pub fn origin(&self) -> &Coord {
        let Engine::ArithTuple(origin) = &self.engine else { unreachable!("Tensor<Ident> is a coordinate tensor") };
        origin
    }
}

/// Whether `lhs < rhs` in every leaf (CuTe DSL's `cute.elem_less`), as a
/// [`Boolean`]. Either side may be a coordinate, a shape or an integer
/// tuple, with static or runtime leaves: `elem_less(c, t.shape())` checks a
/// coordinate against a tensor's extents known at launch. The two must be
/// congruent (the same nesting, compared mode by mode), and leaves that
/// may be negative ([`IndexExpr::signed`], a negative constant, a negative
/// [`domain_offset`]) compare signed. A leaf whose result is known when the
/// kernel is built folds: two static leaves, a leaf whose upper bound is at
/// most a static bound, and a negative constant against a value that is
/// provably non-negative (CuTe's `elem_less((m, -1), (M, k))`); if every
/// leaf folds, so does the result.
#[track_caller]
pub fn elem_less(lhs: impl Into<Coord>, rhs: impl Into<Coord>) -> Boolean {
    let (lhs, rhs) = (lhs.into(), rhs.into());
    assert!(
        lhs.congruent(&rhs),
        "elem_less: {lhs} and {rhs} are not congruent: elem_less compares tuples of the same nesting, leaf by leaf"
    );
    let mut acc: Option<Boolean> = None;
    for (c, b) in lhs.leaves().into_iter().zip(rhs.leaves()) {
        let p = match b.as_static() {
            Some(b) => {
                if let Some(cv) = c.bound()
                    && cv <= b
                {
                    continue;
                }
                if let Some(cv) = c.as_static_signed() {
                    if cv >= b as i64 {
                        return false.into_value();
                    }
                    continue;
                }
                if c.width() == 64 {
                    Int64::from(c.clone()).lt(b as i64)
                } else if c.is_non_negative() {
                    // An unsigned comparison of a value known to be
                    // non-negative, which lets LLVM merge the predicates of
                    // neighbouring coordinates (c, c + 1, ...) of an aligned
                    // tile.
                    let b =
                        u32::try_from(b).unwrap_or_else(|_| panic!("elem_less: the bound {b} is past the u32 range"));
                    Int32::from(c.clone()).bitcast::<u32>().lt(b)
                } else {
                    c.clone().into_value().lt(b)
                }
            }
            None => {
                if let (Some(cv), Some(bv)) = (c.as_static_signed(), b.as_static_signed()) {
                    if cv >= bv {
                        return false.into_value();
                    }
                    continue;
                }
                if c.as_static_signed().is_some_and(|cv| cv < 0) && b.is_non_negative() {
                    continue;
                }
                if c == b {
                    return false.into_value();
                }
                if c.width() == 64 || b.width() == 64 {
                    Int64::from(c.clone()).lt(Int64::from(b.clone()))
                } else {
                    Int32::from(c.clone()).lt(Int32::from(b.clone()))
                }
            }
        };
        acc = Some(match acc {
            Some(a) => a & p,
            None => p,
        });
    }
    acc.unwrap_or_else(|| true.into_value())
}

/// `tensor` with its domain shifted by `coord` (CuTe DSL's
/// `cute.domain_offset`): the same layout, from a start moved to the
/// element at `coord`, so that element `c` of the result is element `coord
/// + c` of `tensor`. A memory tensor's base offset grows by
/// `crd2idx(coord, layout)`, and its swizzle, which applies to the whole
/// offset, is kept. A coordinate tensor adds the tuple `coord` maps to to
/// its origin (CuTe's `tuple_add`).
///
/// The shift may be negative, as the K residue of a GEMM is: `K - bK *
/// k_tiles`, a runtime value of either sign that [`IndexExpr::signed`]
/// takes, so the first tile reaches before the tensor's start and its
/// accesses there are predicated off (the simulator reports one that is
/// not). A coordinate tensor takes every runtime value of the shift as one
/// of either sign, even an `Int32` passed as it is (as CuTe DSL passes the
/// residue), so the coordinates it gives compare signed in [`elem_less`]. A swizzled tensor takes only a shift that is provably
/// non-negative, as its swizzle is a function of the non-negative offset.
/// A register fragment takes only a static shift.
#[track_caller]
pub fn domain_offset<T>(coord: impl Into<Coord>, tensor: &Tensor<T>) -> Tensor<T> {
    let coord = coord.into();
    match &tensor.engine {
        Engine::ArithTuple(origin) => {
            // A runtime shift may be negative whatever it was converted
            // from (an `Int32` K residue, say), so the coordinates it moves
            // compare signed and never fold as non-negative.
            let coord = coord.map(&mut IndexExpr::with_signed_runtime_values);
            let shift = tensor.layout.eval_checked(&coord, "domain_offset");
            Tensor { engine: Engine::ArithTuple(tuple_sum(origin, &shift)), ..tensor.clone() }
        }
        Engine::Memory { .. } | Engine::Register(_) | Engine::SmemDesc { .. } => {
            let shift = tensor.layout.offset_checked(&coord, "domain_offset");
            if let Some(sw) = tensor.swizzle()
                && !shift.is_non_negative()
            {
                panic!(
                    "domain_offset: the shift {shift:?} of the tensor {} swizzled by {sw} may be negative; a swizzled \
                     tensor takes only shifts that are provably non-negative",
                    tensor.layout
                );
            }
            if tensor.is_register() && shift.as_static().is_none() {
                panic!(
                    "domain_offset: the shift {shift:?} of a register fragment is not static; register fragments are \
                     indexed with static coordinates"
                );
            }
            let origin = tensor.origin.as_ref().map(|o| o.push(tensor, ViewOp::Offset(coord.clone())));
            Tensor { offset: tensor.offset.clone() + shift, origin, ..tensor.clone() }
        }
    }
}

/// [`domain_offset`] by a shift whose runtime values are known to be
/// non-negative (a CTA's multicast share in [`tma_partition`](crate::cpasync::tma_partition),
/// which refuses a share that [`IndexExpr::is_non_negative`] cannot vouch for):
/// a coordinate tensor evaluates its layout at the shift as it is, where
/// [`domain_offset`] takes its runtime values as signed, which the layout's
/// divisions refuse; the origin still takes them as signed after.
#[track_caller]
pub(crate) fn domain_offset_non_negative<T>(coord: Coord, tensor: &Tensor<T>) -> Tensor<T> {
    assert!(
        coord.leaves().iter().all(|e| e.is_non_negative()),
        "domain_offset_non_negative: the shift {coord:?} may be negative"
    );
    match &tensor.engine {
        Engine::ArithTuple(origin) => {
            let shift = tensor.layout.eval_checked(&coord, "domain_offset");
            let shift = shift.map(&mut IndexExpr::with_signed_runtime_values);
            Tensor { engine: Engine::ArithTuple(tuple_sum(origin, &shift)), ..tensor.clone() }
        }
        _ => domain_offset(coord, tensor),
    }
}

/// The colexicographic successor of `coord` within `shape` (CuTe's
/// `increment_coord`). It adds 1 to leaf 0; a leaf that this brings to its
/// extent becomes 0 and passes the 1 on to the next leaf. The last leaf
/// does not wrap, so the successor of the last coordinate of `shape` has
/// the last leaf at its extent, an end marker as in C++ CuTe, and further
/// increments keep counting from there. `(3,1,0)` in `(4,2,5)` becomes
/// `(0,0,1)`, and the last coordinate, `(3,1,4)`, becomes `(0,0,5)`.
///
/// `coord` must be congruent with `shape` and have no `..` leaf. As in the
/// CuTe dialect, a leaf other than the last that is at or past its extent
/// after adding the carry wraps to 0, and a leaf whose extent is a static 1
/// is always 0 and passes the carry on. Static coordinates and extents are
/// folded when the kernel is built (and need no kernel); a runtime leaf, or
/// a runtime extent, costs a compare and two selects.
#[track_caller]
pub fn increment_coord(coord: impl Into<SliceCoord>, shape: impl Into<Shape>) -> Coord {
    let coord = coord.into();
    let shape = shape.into();
    let shown = || coord.map(&mut |c| c.as_ref().map_or_else(|| "_".to_string(), |e| e.to_string()));
    assert!(
        coord.leaves().iter().all(|c| c.is_some()),
        "increment_coord: coordinate {} has a `..` (`_`) leaf; every mode needs a value",
        shown()
    );
    assert!(coord.congruent(&shape), "increment_coord: coordinate {} is not congruent with shape {shape}", shown());
    let leaves: Vec<IndexExpr> = coord.leaves().into_iter().flatten().cloned().collect();
    let extents = shape.flat_leaves();
    let mut out = Vec::with_capacity(leaves.len());
    let mut carry = IndexExpr::Static(1);
    for (i, (c, extent)) in leaves.into_iter().zip(&extents).enumerate() {
        if i + 1 == extents.len() {
            out.push(c + carry.clone());
        } else if extent.as_static() == Some(1) {
            out.push(IndexExpr::Static(0));
        } else {
            let sum = c + carry.clone();
            let fits = match (sum.as_static(), extent.as_static()) {
                (Some(v), Some(e)) => Some(v < e),
                (None, Some(e)) if sum.bound().is_some_and(|b| b <= e) => Some(true),
                _ => None,
            };
            match fits {
                Some(true) => {
                    out.push(sum);
                    carry = IndexExpr::Static(0);
                }
                Some(false) => {
                    out.push(IndexExpr::Static(0));
                    carry = IndexExpr::Static(1);
                }
                None => {
                    let value: Int32 = sum.into_value();
                    let fits = value.lt(extent.clone());
                    let next = select(fits, value, 0i32);
                    let carried = select(fits, 0i32, 1i32);
                    let bound = extent.as_static().map(|e| e.max(1));
                    builder::with(|b| {
                        b.set_facts(next.id, ValueFacts { bound, divisor: 1, pow2: false });
                        b.set_facts(carried.id, ValueFacts { bound: Some(2), divisor: 1, pow2: false });
                    });
                    out.push(next.into());
                    carry = carried.into();
                }
            }
        }
    }
    Tree::unflatten(&out, &coord)
}

/// Panic, naming `what`, if it is in another CTA's shared memory (`peer`):
/// the instruction asking reaches only this CTA's (roadmap part m8-dsmem).
#[track_caller]
pub(crate) fn refuse_dsmem(peer: Option<ValueId>, what: &str) {
    if peer.is_some() {
        panic!(
            "cute: {what} is in another CTA's shared memory (a map_dsmem_ptr pointer, distributed shared memory), \
             which this instruction cannot reach: loads and stores (Tensor::load and store, cute::load and store, \
             CopyUniversalOp, CopyS2ROp and CopyR2SOp copies), atomics and reductions, and CopyDsmemStoreOp reach it"
        );
    }
}

/// A load of `n` contiguous elements at `off`, from the shared memory of CTA `peer` of the cluster
/// when given (roadmap part m8-dsmem).
#[track_caller]
pub(crate) fn load_vector_in<T: Numeric>(
    buffer: BufferId,
    off: IndexExpr,
    n: usize,
    peer: Option<ValueId>,
) -> Vec<ValueId> {
    let index = index_operand(off);
    load_vector_with::<T>(buffer, index, n, MemAttrs { peer, ..MemAttrs::default() })
}

/// A store of contiguous elements at `off`, into the shared memory of CTA `peer` of the cluster
/// when given (roadmap part m8-dsmem).
#[track_caller]
pub(crate) fn store_vector_in(buffer: BufferId, off: IndexExpr, values: &[ValueId], peer: Option<ValueId>) {
    let index = index_operand(off);
    store_vector_with(buffer, index, values, MemAttrs { peer, ..MemAttrs::default() })
}

/// A load of `n` contiguous elements with memory attributes (ordering and
/// cache hints).
#[track_caller]
pub(crate) fn load_vector_with<T: Numeric>(
    buffer: BufferId,
    index: ValueId,
    n: usize,
    attrs: MemAttrs,
) -> Vec<ValueId> {
    let loc = Location::caller();
    builder::with(|b| {
        if T::DTYPE.is_word_packed() {
            // Whole 32-bit words, unpacked (`index` counts words). Three
            // words (16 FP6 values) are three loads: PTX has no 3-word
            // vector, and 12-byte groups are only 4-byte aligned.
            let count = sub_byte_words(T::DTYPE, n);
            let mut words = Vec::with_capacity(count);
            for (i, k) in word_accesses(b, index, count, loc) {
                let w: Vec<ValueId> = (0..k).map(|_| b.new_value(DType::U32)).collect();
                b.push(Op::Load { results: w.clone(), buffer, index: i, attrs, pred: None, pass_thru: vec![] }, loc);
                words.extend(w);
            }
            return unpack_words(b, &words, T::DTYPE, n, loc);
        }
        let results: Vec<ValueId> = (0..n).map(|_| b.new_value(T::DTYPE)).collect();
        b.push(Op::Load { results: results.clone(), buffer, index, attrs, pred: None, pass_thru: vec![] }, loc);
        results
    })
}

/// The 32-bit words `n` packed values of the word-packed type `t` fill.
#[track_caller]
fn sub_byte_words(t: DType, n: usize) -> usize {
    let bits = n * t.bits();
    assert!(
        bits.is_multiple_of(32) && bits <= 128,
        "cute: a vector of {n} {} values is {bits} bits: sub-byte tensors are accessed in vectors of a multiple of 32 \
         bits, at most 128",
        t.cute_name()
    );
    bits / 32
}

/// The accesses that move `count` words from word index `index`: one
/// vector, or for three words three single words, as `(index, words)`.
fn word_accesses(
    b: &mut builder::BuilderState,
    index: ValueId,
    count: usize,
    loc: crate::ir::Loc,
) -> Vec<(ValueId, usize)> {
    if count.is_power_of_two() {
        return vec![(index, count)];
    }
    let ty = b.dtype(index);
    (0..count)
        .map(|k| {
            if k == 0 {
                return (index, 1);
            }
            let c = b.constant(crate::ir::Literal::from_bits128(ty, k as u128), loc);
            (b.binary(crate::ir::BinaryOp::Add, index, c, loc), 1)
        })
        .collect()
}

/// Values of a word-packed type that fill a whole number of 32-bit words
/// (8 Int4 or FP4, 16 FP6, 4 FP6x4), and those words.
pub(crate) fn word_group(t: DType) -> (usize, usize) {
    let g = 32 / gcd(32, t.bits());
    (g, g * t.bits() / 32)
}

/// `words` (packed 32-bit words) as `n` values of the word-packed type
/// `t`, element 0 in the low bits of word 0. A value that straddles two
/// words (FP6, FP6x4) is put together from both.
pub(crate) fn unpack_words(
    b: &mut builder::BuilderState,
    words: &[ValueId],
    t: DType,
    n: usize,
    loc: crate::ir::Loc,
) -> Vec<ValueId> {
    let bits = t.bits();
    if 32 % bits == 0 {
        let per = 32 / bits;
        return words.iter().flat_map(|&w| b.unpack(w, t, per, loc)).take(n).collect();
    }
    let k = |b: &mut builder::BuilderState, v: u32| b.constant(crate::ir::Literal::U32(v), loc);
    (0..n)
        .map(|i| {
            let (w, s) = (i * bits / 32, (i * bits % 32) as u32);
            let mut x = words[w];
            if s > 0 {
                let c = k(b, s);
                x = b.binary(crate::ir::BinaryOp::ShrU, x, c, loc);
            }
            if s as usize + bits > 32 {
                let c = k(b, 32 - s);
                let hi = b.binary(crate::ir::BinaryOp::Shl, words[w + 1], c, loc);
                x = b.binary(crate::ir::BinaryOp::Or, x, hi, loc);
            }
            b.unpack(x, t, 1, loc)[0]
        })
        .collect()
}

/// `values` (of the word-packed type `t`) packed into 32-bit words, element
/// 0 in the low bits of word 0.
pub(crate) fn pack_words(
    b: &mut builder::BuilderState,
    values: &[ValueId],
    t: DType,
    loc: crate::ir::Loc,
) -> Vec<ValueId> {
    let bits = t.bits();
    if 32 % bits == 0 {
        let per = 32 / bits;
        return values.chunks(per).map(|c| b.pack(c.to_vec(), DType::U32, loc)).collect();
    }
    let k = |b: &mut builder::BuilderState, v: u32| b.constant(crate::ir::Literal::U32(v), loc);
    let words = values.len() * bits / 32;
    let mut out: Vec<Option<ValueId>> = vec![None; words];
    let mut or_into = |b: &mut builder::BuilderState, w: usize, x: ValueId| {
        out[w] = Some(match out[w] {
            None => x,
            Some(acc) => b.binary(crate::ir::BinaryOp::Or, acc, x, loc),
        });
    };
    for (i, &v) in values.iter().enumerate() {
        let (w, s) = (i * bits / 32, (i * bits % 32) as u32);
        let x = b.pack(vec![v], DType::U32, loc);
        let low = if s > 0 {
            let c = k(b, s);
            b.binary(crate::ir::BinaryOp::Shl, x, c, loc)
        } else {
            x
        };
        or_into(b, w, low);
        if s as usize + bits > 32 {
            let c = k(b, 32 - s);
            let high = b.binary(crate::ir::BinaryOp::ShrU, x, c, loc);
            or_into(b, w + 1, high);
        }
    }
    out.into_iter().map(|w| w.expect("every word holds a value")).collect()
}

/// The index operand of an access of `n` elements of `T` at element offset
/// `off`: the offset itself, or for a word-packed type the offset in 32-bit
/// words, which `off` must provably be a whole number of (with `n` a
/// multiple of 32 bits, at most 128).
#[track_caller]
pub(crate) fn access_index<T: Numeric>(off: IndexExpr, n: usize, what: &str) -> ValueId {
    let t = T::DTYPE;
    if !t.is_word_packed() {
        return index_operand(off);
    }
    sub_byte_words(t, n);
    let (group, words) = word_group(t);
    assert!(
        off.is_multiple_of(group),
        "cute: {what}: cannot prove that the offset {off:?} of a vector of {n} {} values is a whole number of 32-bit \
         words ({group} values){}",
        t.cute_name(),
        builder::var_value_note(&off)
    );
    index_operand(off / group * words)
}

/// A store of contiguous elements with memory attributes (ordering and
/// cache hints).
#[track_caller]
pub(crate) fn store_vector_with(buffer: BufferId, index: ValueId, values: &[ValueId], attrs: MemAttrs) {
    let loc = Location::caller();
    builder::with(|b| {
        let t = b.dtype(values[0]);
        if t.is_word_packed() {
            let count = sub_byte_words(t, values.len());
            let words = pack_words(b, values, t, loc);
            let mut at = 0;
            for (i, k) in word_accesses(b, index, count, loc) {
                b.push(Op::Store { values: words[at..at + k].to_vec(), buffer, index: i, attrs, pred: None }, loc);
                at += k;
            }
            return;
        }
        b.push(Op::Store { values: values.to_vec(), buffer, index, attrs, pred: None }, loc)
    });
}

/// The element type and length of the buffer holding `n` values of `t`: a
/// sub-byte type lives in bytes (`ceil(n * bits / 8)` of them).
pub(crate) fn storage_decl(t: DType, n: usize) -> (DType, usize) {
    if t.is_word_packed() { (DType::U8, t.storage_bytes(n)) } else { (t, n) }
}

/// Panic unless `n` values of `t` fill a whole number of bytes, as CuTe
/// DSL requires of shared-memory allocations.
#[track_caller]
fn require_whole_bytes(t: DType, n: usize, what: &str) {
    assert!(
        !t.is_sub_byte() || (n * t.bits()).is_multiple_of(8),
        "cute: {what} of {n} {} values is {} bits, not a whole number of bytes",
        t.cute_name(),
        n * t.bits()
    );
}

/// Declare a global-memory tensor parameter. The host must pass a buffer of
/// at least `layout.cosize()` elements, 16-byte aligned. A layout whose
/// cosize is only known at launch, with runtime extents or strides
/// (`Layout::row_major((m, n))` with `m` and `n` from [`param_extent`], CuTe
/// DSL's `(?,?):(?,1)`), declares a buffer sized at launch, as [`param_ptr`]
/// does: launches do not check its length, and the simulator reports the
/// accesses that fall outside it.
///
/// A [`TensorSpec`](crate::TensorSpec) in place of the layout (CuTe DSL's
/// `cute.Tensor[dtype, shape, stride]` annotation) declares the buffer with
/// the spec's alignment, then one scalar parameter per dynamic leaf
/// (`{name}_shape_{path}`, `{name}_stride_{path}`; see [`crate::params`]).
#[track_caller]
pub fn param_tensor<T: Numeric>(name: &str, layout: impl crate::params::TensorParamSource) -> Tensor<T> {
    let _at = builder::LocationScope::new();
    layout.declare_tensor(name)
}

/// [`param_tensor`] with a stronger alignment promise: the host passes a
/// buffer whose start is a multiple of `align` bytes (a power of two of at
/// least 16; device allocations are 256-byte aligned), and the kernel may
/// use accesses that wide, such as the 32-byte vectors that
/// [`autovec_copy`](crate::autovec_copy) issues to global memory for sm_100.
/// The kernel parameter carries it (`llvm.align`), the launcher checks the
/// device pointer, and the simulator checks every access against it.
#[track_caller]
pub fn param_tensor_aligned<T: Numeric>(name: &str, layout: Layout, align: usize) -> Tensor<T> {
    assert!(
        align.is_power_of_two() && align >= 16,
        "param_tensor_aligned: an alignment of {align} bytes is not a power of two of at least 16"
    );
    let len = layout.cosize_at(&[]).as_static();
    let elem = storage_decl(T::DTYPE, 0).0;
    let len = len.map(|n| storage_decl(T::DTYPE, n).1);
    let buffer = builder::with(|b| {
        b.declare_buffer(BufferDecl {
            name: name.to_string(),
            elem,
            len,
            space: MemSpace::Global,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    Tensor::new(Engine::Memory { buffer, space: MemSpace::Global, peer: None }, layout)
}

/// Declare a global-memory parameter whose length is only known at launch.
/// The host passes a 16-byte-aligned buffer ([`param_ptr_aligned`] takes
/// another alignment).
#[track_caller]
pub fn param_ptr<T: Numeric>(name: &str) -> Ptr<T> {
    let _at = builder::LocationScope::new();
    param_ptr_aligned(name, 16)
}

/// [`param_ptr`] with the alignment the host promises for the buffer's start
/// (CuTe DSL's `assumed_align`): a power of two of at least the element's
/// size. The kernel parameter carries it (`llvm.align`), vector accesses are
/// proved aligned from it ([`Ptr::alignment`]), the CUDA launcher checks the
/// device pointer ([`check_pointer_alignment`](crate::check_pointer_alignment))
/// and the simulator checks every access.
#[track_caller]
pub fn param_ptr_aligned<T: Numeric>(name: &str, align: usize) -> Ptr<T> {
    let size = T::DTYPE.storage_bytes(1).max(1);
    assert!(
        align.is_power_of_two() && align >= size,
        "param_ptr_aligned: an alignment of {align} bytes is not a power of two of at least the {size} bytes of {}",
        T::DTYPE
    );
    let buffer = builder::with(|b| {
        b.declare_buffer(BufferDecl {
            name: name.to_string(),
            elem: storage_decl(T::DTYPE, 0).0,
            len: None,
            space: MemSpace::Global,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    Ptr { buffer, space: MemSpace::Global, offset: IndexExpr::Static(0), swizzle: None, peer: None, _t: PhantomData }
}

/// A typed position in global or shared memory, optionally carrying a
/// swizzle (CuTe DSL's swizzled pointers, `make_ptr(..., swizzle_=...)` and
/// [`recast_ptr`]). As in CuTe DSL, the swizzle acts on byte addresses: the
/// element at `p + i` is at the byte address `swizzle(address(p) + i *
/// size)`. cute-rs evaluates it on the byte offset from the start of the
/// buffer, which is the same thing because the buffer's start is checked
/// to be aligned to the swizzle's [`period`](Swizzle::period).
pub struct Ptr<T> {
    pub(crate) buffer: BufferId,
    pub(crate) space: MemSpace,
    pub(crate) offset: IndexExpr,
    pub(crate) swizzle: Option<Swizzle>,
    /// A distributed-shared-memory pointer ([`map_dsmem_ptr`](crate::map_dsmem_ptr),
    /// roadmap part m8-dsmem): the rank (an Int32) of the CTA of the cluster
    /// whose shared memory it points into, at the offset `buffer` has here.
    pub(crate) peer: Option<ValueId>,
    pub(crate) _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        Ptr {
            buffer: self.buffer,
            space: self.space,
            offset: self.offset.clone(),
            swizzle: self.swizzle,
            peer: self.peer,
            _t: PhantomData,
        }
    }
}

impl<T> fmt::Debug for Ptr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ptr<{:?}", self.space)?;
        if let Some(p) = self.peer {
            write!(f, ", dsmem of CTA {p:?}")?;
        }
        if let Some(sw) = self.swizzle {
            write!(f, ", {sw}")?;
        }
        write!(f, ">({:?} + {:?})", self.buffer, self.offset)
    }
}

impl<T> Ptr<T> {
    /// The pointer's swizzle, on bytes (CuTe DSL's `Pointer.type.swizzle_type`),
    /// or `None`.
    pub fn swizzle(&self) -> Option<Swizzle> {
        self.swizzle
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
        if T::DTYPE.is_word_packed() || U::DTYPE.is_word_packed() {
            // Offsets in bits.
            let (from, to) = (T::DTYPE.storage_bits(), U::DTYPE.storage_bits());
            let bits = self.offset.clone() * from;
            assert!(
                bits.is_multiple_of(to),
                "cute: cannot prove that {:?} {} values are a whole number of {} values{}",
                self.offset,
                T::DTYPE.cute_name(),
                U::DTYPE.cute_name(),
                builder::var_value_note(&self.offset)
            );
            return Ptr {
                buffer: self.buffer,
                space: self.space,
                offset: bits / to,
                swizzle: self.swizzle,
                peer: self.peer,
                _t: PhantomData,
            };
        }
        let (from, to) = (T::DTYPE.bytes(), U::DTYPE.bytes());
        let bytes = self.offset.clone() * from;
        assert!(
            bytes.is_multiple_of(to),
            "cute: cannot prove that {:?} is aligned to {}{}",
            self.offset,
            U::DTYPE,
            builder::var_value_note(&self.offset)
        );
        Ptr {
            buffer: self.buffer,
            space: self.space,
            offset: bytes / to,
            swizzle: self.swizzle,
            peer: self.peer,
            _t: PhantomData,
        }
    }

    /// The swizzle of the pointer on the offsets of `T` elements: `None`
    /// without one. Panics if the swizzle splits `T` elements or `T` is not
    /// a power-of-two number of bits wide, or if the buffer's start is not
    /// aligned to the swizzle's period (the swizzle acts on absolute
    /// addresses).
    #[track_caller]
    fn element_swizzle(&self) -> Option<Swizzle> {
        let sw = self.swizzle?;
        if sw.is_identity() {
            return None;
        }
        let elem = sw.to_elements(T::DTYPE.storage_bits()).unwrap_or_else(|why| {
            panic!("cute: a pointer to {} swizzled by {sw} (on bytes): {why}", T::DTYPE.cute_name())
        });
        let align = builder::with(|b| b.buffer(self.buffer).align);
        assert!(
            align >= sw.period(),
            "cute: the pointer swizzle {sw} acts on absolute byte addresses, so it needs a base aligned to its period \
             of {} bytes, but the buffer it points into is only known to be {align}-byte aligned (allocate it with \
             that alignment: alloc_smem(count, {}))",
            sw.period(),
            sw.period()
        );
        Some(elem)
    }

    /// The element offset this pointer addresses, its swizzle applied.
    #[track_caller]
    fn physical_offset(&self) -> IndexExpr {
        match self.element_swizzle() {
            Some(sw) => sw.apply_expr(self.offset.clone()),
            None => self.offset.clone(),
        }
    }

    /// The plain pointer to the element this pointer addresses, its swizzle
    /// applied (roadmap part m11-prims-descriptors-swizzle's `apply_swizzle`).
    #[track_caller]
    pub(crate) fn physical_ptr(&self) -> Ptr<T> {
        Ptr { offset: self.physical_offset(), swizzle: None, ..self.clone() }
    }

    /// Read the element at this address.
    #[track_caller]
    pub fn load(&self) -> Value<T> {
        refuse_scalar_sub_byte(T::DTYPE, "a load");
        Value::from_id(load_vector_in::<T>(self.buffer, self.physical_offset(), 1, self.peer)[0])
    }

    /// Write the element at this address.
    #[track_caller]
    pub fn store(&self, value: impl IntoValue<T>) {
        refuse_scalar_sub_byte(T::DTYPE, "a store");
        store_vector_in(self.buffer, self.physical_offset(), &[value.into_value().checked()], self.peer);
    }

    /// The buffer and the logical element offset, swizzle or not.
    pub(crate) fn parts_unswizzled(&self) -> (BufferId, IndexExpr) {
        (self.buffer, self.offset.clone())
    }

    /// The same pointer at the logical element offset `offset`.
    pub(crate) fn with_offset(&self, offset: IndexExpr) -> Ptr<T> {
        Ptr { offset, ..self.clone() }
    }

    /// The buffer and the element offset, for an instruction that takes a
    /// plain address: panics for a swizzled pointer, whose elements are not
    /// where the offset says.
    #[track_caller]
    pub(crate) fn parts(&self) -> (BufferId, IndexExpr) {
        refuse_dsmem(self.peer, &format!("the pointer {self:?}"));
        let (buffer, offset, _) = self.parts_dsmem();
        (buffer, offset)
    }

    /// [`Ptr::parts`] and the peer CTA of a distributed-shared-memory
    /// pointer, for the instructions that reach another CTA's shared memory
    /// (loads, stores, atomics and reductions; roadmap part m8-dsmem).
    #[track_caller]
    pub(crate) fn parts_dsmem(&self) -> (BufferId, IndexExpr, Option<ValueId>) {
        if let Some(sw) = self.swizzle
            && !sw.is_identity()
        {
            panic!(
                "cute: the pointer {self:?} is swizzled by {sw}; an instruction that takes an address takes a plain \
                 pointer (recast_ptr(&ptr, None) drops the swizzle)"
            );
        }
        (self.buffer, self.offset.clone(), self.peer)
    }

    /// The CTA of the cluster whose shared memory a distributed-shared-memory
    /// pointer ([`map_dsmem_ptr`](crate::map_dsmem_ptr)) points into, or
    /// `None` for any other pointer.
    pub fn dsmem_peer(&self) -> Option<Int32> {
        self.peer.map(Value::from_id)
    }

    /// The memory the pointer addresses: global, shared or per-thread.
    pub fn memory_space(&self) -> MemSpace {
        self.space
    }

    /// The address space the pointer addresses (CuTe DSL's
    /// `Pointer.memspace`): [`AddressSpace::Gmem`], [`AddressSpace::Smem`]
    /// or [`AddressSpace::Rmem`] (a [`make_rmem_tensor`] array), and
    /// [`AddressSpace::Dsmem`] for a [`map_dsmem_ptr`](crate::map_dsmem_ptr)
    /// pointer into another CTA's shared memory.
    pub fn memspace(&self) -> AddressSpace {
        if self.peer.is_some() { AddressSpace::Dsmem } else { self.space.into() }
    }

    /// The largest power of two of bytes the address provably is a multiple
    /// of (CuTe DSL's `Pointer.max_alignment`): the alignment the buffer is
    /// declared with (16 bytes for [`param_tensor`] and [`param_ptr`], what
    /// [`alloc_smem`] or [`param_tensor_aligned`] give), reduced by what is
    /// known to divide the offset (a [`param_extent`]'s `multiple_of`
    /// counts).
    /// A swizzled pointer's is bounded by the swizzle's unchanged run,
    /// `2^M` bytes.
    pub fn max_alignment(&self) -> usize {
        let align = ptr_alignment(self.buffer, &self.offset, T::DTYPE.bytes());
        match self.swizzle {
            Some(sw) if !sw.is_identity() => align.min(sw.max_alignment()),
            _ => align,
        }
    }
}

/// The alignment in bytes of element `offset` of `buffer`, whose elements
/// are `elem_bytes` wide: the buffer's declared alignment, halved until the
/// offset in bytes is provably a multiple of it.
pub(crate) fn ptr_alignment(buffer: BufferId, offset: &IndexExpr, elem_bytes: usize) -> usize {
    let mut align = builder::with(|b| b.buffer(buffer).align);
    let bytes = offset.clone() * elem_bytes;
    while align > 1 && !bytes.is_multiple_of(align) {
        align /= 2;
    }
    align
}

/// A tensor of `layout` over an iterator (CuTe DSL's
/// `cute.make_tensor(iterator, layout_or_shape)`).
///
/// The iterator ([`TensorEngine`]) is a [`Ptr`] (by reference or by value),
/// which gives a tensor over memory, or an origin (an integer, an
/// [`IndexExpr`], a [`Coord`] or a tuple of them), which gives a coordinate
/// tensor ([`make_coord_tensor`]).
///
/// The layout ([`LayoutOrShape`]) is a [`Layout`], a [`ComposedLayout`], or
/// a shape, which becomes `make_layout(shape)` (compact column-major). Its
/// extents and strides may be runtime values: `Layout::row_major((m, k))`
/// with `m` and `k` from [`param_extent`] is an `m x k` matrix whose size is
/// only known at launch. A static upper bound as the extent with runtime
/// strides (`Layout::with_strides((BOUND, BOUND), (k, 1))`, as
/// `examples/runtime_gemm.rs`'s `build` does) also works. A composed layout
/// whose inner is a function ([`make_composed_layout_fn`](crate::make_composed_layout_fn))
/// has nothing memory can hold, and is refused, as in CuTe DSL.
///
/// Over a pointer, a [`ComposedLayout`] is a swizzle over the outer layout
/// (relative to the pointer, which must then be a whole number of swizzle
/// periods into its buffer) or a layout inner, as CuTe's
/// `make_tensor(ptr, composed)`. A pointer with a swizzle ([`recast_ptr`])
/// gives a swizzled tensor, as in CuTe DSL; it does not combine with a
/// swizzled composed layout. Over an origin, a composed layout must be
/// normal (it decays to its outer layout).
#[track_caller]
pub fn make_tensor<E: TensorEngine>(iterator: E, layout: impl Into<LayoutOrShape>) -> Tensor<E::Elem> {
    match layout.into() {
        LayoutOrShape::Layout(c) => iterator.into_tensor(c),
        LayoutOrShape::FnInner(text) => panic!(
            "make_tensor: the composed layout {text} has a function as its inner, which has no memory \
             representation (as in CuTe DSL, which cannot make a tensor of it either): call the layout to get \
             offsets, and load through a tensor over the pointer with a plain layout"
        ),
    }
}

/// The iterators [`make_tensor`] takes: a [`Ptr`], which gives a tensor of
/// its element type, or an origin, which gives a coordinate tensor
/// (`Tensor<Ident>`).
pub trait TensorEngine {
    type Elem;
    #[track_caller]
    fn into_tensor(self, layout: ComposedLayout) -> Tensor<Self::Elem>;
}

impl<T: Numeric> TensorEngine for &Ptr<T> {
    type Elem = T;
    #[track_caller]
    fn into_tensor(self, layout: ComposedLayout) -> Tensor<T> {
        ptr_tensor(self, layout)
    }
}

impl<T: Numeric> TensorEngine for Ptr<T> {
    type Elem = T;
    #[track_caller]
    fn into_tensor(self, layout: ComposedLayout) -> Tensor<T> {
        ptr_tensor(&self, layout)
    }
}

#[track_caller]
fn origin_tensor(origin: Coord, c: ComposedLayout) -> Tensor<Ident> {
    let Some(outer) = c.as_layout() else {
        panic!(
            "make_tensor: a coordinate tensor takes a plain layout, but {c} has an inner function and offset that \
             coordinates cannot go through"
        )
    };
    make_coord_tensor(origin, outer.clone())
}

impl TensorEngine for Coord {
    type Elem = Ident;
    #[track_caller]
    fn into_tensor(self, layout: ComposedLayout) -> Tensor<Ident> {
        origin_tensor(self, layout)
    }
}

impl TensorEngine for &Coord {
    type Elem = Ident;
    #[track_caller]
    fn into_tensor(self, layout: ComposedLayout) -> Tensor<Ident> {
        origin_tensor(self.clone(), layout)
    }
}

macro_rules! origin_engine {
    ($($t:ty),*) => {$(
        impl TensorEngine for $t {
            type Elem = Ident;
            #[track_caller]
            fn into_tensor(self, layout: ComposedLayout) -> Tensor<Ident> {
                origin_tensor(Coord::from(self), layout)
            }
        }
    )*};
}
origin_engine!(usize, u32, i32, i64, IndexExpr, Int32);

macro_rules! origin_engine_tuple {
    ($(($($name:ident),+)),+) => {$(
        #[allow(non_snake_case)]
        impl<$($name: Into<Coord>),+> TensorEngine for ($($name,)+) {
            type Elem = Ident;
            #[track_caller]
            fn into_tensor(self, layout: ComposedLayout) -> Tensor<Ident> {
                let ($($name,)+) = self;
                origin_tensor(Tree::Branch(vec![$($name.into()),+]), layout)
            }
        }
    )+};
}
origin_engine_tuple!((A), (A, B), (A, B, C), (A, B, C, D));

/// What [`make_tensor`] lays its iterator out with: a layout (a plain or a
/// composed one), or a shape, which becomes its compact column-major layout
/// (`make_layout(shape)`). Built from a [`Layout`], a [`ComposedLayout`]
/// (or a reference to either), a shape (an integer, an [`IndexExpr`], an
/// [`Int32`], a [`Shape`], an [`IntTuple`], or a tuple of them, so runtime
/// extents are fine), or a [`FnComposedLayout`](crate::FnComposedLayout),
/// which [`make_tensor`] refuses.
#[derive(Clone, Debug)]
pub enum LayoutOrShape {
    /// The layout, as a composed layout (a plain layout is a normal one).
    Layout(ComposedLayout),
    /// A composed layout with a function inner, as it prints.
    FnInner(String),
}

impl From<Layout> for LayoutOrShape {
    fn from(l: Layout) -> Self {
        LayoutOrShape::Layout(l.into_composed())
    }
}

impl From<&Layout> for LayoutOrShape {
    fn from(l: &Layout) -> Self {
        LayoutOrShape::from(l.clone())
    }
}

impl From<ComposedLayout> for LayoutOrShape {
    fn from(c: ComposedLayout) -> Self {
        LayoutOrShape::Layout(c)
    }
}

impl From<&ComposedLayout> for LayoutOrShape {
    fn from(c: &ComposedLayout) -> Self {
        LayoutOrShape::Layout(c.clone())
    }
}

impl<F> From<crate::layout::FnComposedLayout<F>> for LayoutOrShape {
    fn from(c: crate::layout::FnComposedLayout<F>) -> Self {
        LayoutOrShape::FnInner(c.to_string())
    }
}

impl<F> From<&crate::layout::FnComposedLayout<F>> for LayoutOrShape {
    fn from(c: &crate::layout::FnComposedLayout<F>) -> Self {
        LayoutOrShape::FnInner(c.to_string())
    }
}

macro_rules! layout_from_shape {
    ($($t:ty),*) => {$(
        impl From<$t> for LayoutOrShape {
            fn from(v: $t) -> Self { LayoutOrShape::from(Layout::col_major(Shape::from(v))) }
        }
    )*};
}
layout_from_shape!(usize, u32, i32, i64, IndexExpr, Int32, IntTuple, &IntTuple, Shape, &Shape);

macro_rules! layout_from_tuple {
    ($(($($name:ident),+)),+) => {$(
        #[allow(non_snake_case)]
        impl<$($name: Into<Shape>),+> From<($($name,)+)> for LayoutOrShape {
            fn from(($($name,)+): ($($name,)+)) -> Self {
                LayoutOrShape::from(Layout::col_major(Shape::from(Tree::Branch(vec![$($name.into()),+]))))
            }
        }
    )+};
}
layout_from_tuple!((A), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F));

#[track_caller]
fn ptr_tensor<T: Numeric>(ptr: &Ptr<T>, c: ComposedLayout) -> Tensor<T> {
    let ptr_sw = ptr.element_swizzle();
    let engine = Engine::Memory { buffer: ptr.buffer, space: ptr.space, peer: ptr.peer };
    let leaf_offset = |what: &str| -> IndexExpr {
        c.offset
            .as_leaf()
            .cloned()
            .unwrap_or_else(|| panic!("make_tensor: the offset {} of {c} must be an integer for {what}", c.offset))
    };
    match &c.inner {
        Inner::Swizzle(sw) if !sw.is_identity() => {
            if let Some(psw) = ptr.swizzle {
                panic!(
                    "make_tensor: the pointer is swizzled by {psw} and the layout {c} by {sw}; a tensor takes one \
                     swizzle (CuTe DSL refuses a swizzled pointer with a swizzled composed layout): use \
                     get_nonswizzle_portion(&layout) with the swizzled pointer, or a plain pointer"
                );
            }
            // The swizzle applies to offsets from the pointer; the tensor
            // keeps offsets from the buffer, which is the same when the
            // pointer is a whole number of periods in.
            assert!(
                ptr.offset.is_multiple_of(sw.period()),
                "make_tensor: the swizzle {sw} of {c} acts on offsets from the pointer, which cute-rs needs to be a \
                 multiple of {} elements (the swizzle's period) into its buffer, but it is {:?} in{}",
                sw.period(),
                ptr.offset,
                builder::var_value_note(&ptr.offset)
            );
            Tensor {
                offset: ptr.offset.clone() + leaf_offset("a swizzle"),
                inner: TensorInner::swizzle::<T>(*sw),
                ..Tensor::new(engine, c.outer.clone())
            }
        }
        Inner::Swizzle(_) => {
            let offset = ptr.offset.clone() + leaf_offset("a tensor");
            let inner = match ptr_sw {
                Some(sw) => TensorInner::Swizzle { elem: sw, bytes: ptr.swizzle, in_layout: false },
                None => TensorInner::Identity,
            };
            Tensor { offset, inner, ..Tensor::new(engine, c.outer.clone()) }
        }
        Inner::Layout(inner) => {
            if ptr_sw.is_some() {
                panic!(
                    "make_tensor: a swizzled pointer with the composed layout {c}, whose inner is a layout, is not \
                     supported; drop the pointer's swizzle (recast_ptr(&ptr, None))"
                );
            }
            inner.require_int_strides("make_tensor", "the inner layout of a tensor's composed layout gives offsets");
            Tensor {
                offset: leaf_offset("a layout inner"),
                inner: TensorInner::Layout { inner: Rc::new(inner.clone()), base: ptr.offset.clone() },
                ..Tensor::new(engine, c.outer.clone())
            }
        }
    }
}

/// CuTe DSL's `cute.recast_ptr(ptr, swizzle_, dtype)`: the same address as
/// a pointer to `U` ([`Ptr::cast`], which must prove it aligned to `U`; a
/// Boolean counts as a byte), with the swizzle `swizzle` (on bytes, as
/// CuTe DSL's values are), or none. A swizzled pointer gives
/// [`make_tensor`] a swizzled tensor: `make_tensor(&recast_ptr::<T, _>(&p,
/// Some(get_swizzle_portion(&l))), get_nonswizzle_portion(&l))` is how a
/// kernel puts a swizzled layout on memory through its pointer. The swizzle
/// acts on absolute byte addresses, so the buffer must be aligned to its
/// [`period`](Swizzle::period) (`alloc_smem(n, 1024)` for `S<3,4,3>`), and
/// it must not split a `U` element (`M >= log2(bytes of U)`); both are
/// checked here.
///
/// ```
/// use cute::prelude::*;
///
/// // Swizzle<3,4,3> on bytes is Swizzle<3,3,3> on f16 elements.
/// let kernel = Kernel::builder("ptr_swizzle").block_dim([64, 1, 1]).build(|| {
///     let out = param_tensor::<F16>("out", Layout::col_major(512));
///     let base = alloc_smem::<F16>(512, 1024);
///     let s = make_tensor(&recast_ptr::<F16, _>(&base, Some(make_swizzle(3, 4, 3))), Layout::row_major((8, 64)));
///     assert_eq!(s.swizzle(), Some(make_swizzle(3, 3, 3)));
///     let t = thread_idx().x;
///     for i in 0..8 {
///         s.store((i, t), out.load(i * 64 + t));
///     }
/// });
/// # let _ = kernel;
/// ```
#[track_caller]
pub fn recast_ptr<U: Numeric, T: Numeric>(ptr: &Ptr<T>, swizzle: Option<Swizzle>) -> Ptr<U> {
    let plain = Ptr::<T> { swizzle: None, ..ptr.clone() };
    let out = Ptr { swizzle: swizzle.filter(|sw| !sw.is_identity()), ..plain.cast::<U>() };
    out.element_swizzle();
    out
}

/// Allocate `count` elements of shared memory aligned to `align` bytes and
/// return a pointer to them (CuTe DSL's `cute.arch.alloc_smem`).
#[track_caller]
pub fn alloc_smem<T: Numeric>(count: usize, align: usize) -> Ptr<T> {
    assert!(align.is_power_of_two(), "cute: shared memory alignment {align} is not a power of two");
    require_whole_bytes(T::DTYPE, count, "alloc_smem");
    let (elem, len) = storage_decl(T::DTYPE, count);
    let buffer = builder::with(|b| {
        let name = format!("smem{}", b.shared_buffer_count());
        b.declare_buffer(BufferDecl {
            name,
            elem,
            len: Some(len),
            space: MemSpace::Shared,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    Ptr { buffer, space: MemSpace::Shared, offset: IndexExpr::Static(0), swizzle: None, peer: None, _t: PhantomData }
}

/// A pointer to the kernel's dynamic shared memory, a region whose size is
/// set at launch (CuTe DSL's `cute.arch.get_dyn_smem(element_type,
/// alignment=None)`): [`LaunchConfig::with_dynamic_smem`] gives it its
/// bytes, and [`get_dyn_smem_size`] reads them in the kernel.
///
/// The region follows the shared buffers sized at build time
/// ([`alloc_smem`], [`make_smem_tensor`]), at a multiple of `alignment`
/// bytes: by default the element's size, as in CuTe DSL, and at least 16, so
/// that 16-byte vector accesses at aligned offsets are aligned. A kernel has
/// one region: every call returns its start, aligned for every call's
/// alignment, so `get_dyn_smem::<f32>(None)` and `get_dyn_smem::<u8>(None)`
/// address the same bytes. Offset and cast the pointer to carve the region
/// up. Tensors over the region take static extents (shared memory is sized when the kernel is built, so a runtime extent is refused there): give a mode
/// sized by the launch a static upper bound as its extent
/// ([`Layout::with_strides`]), or index the region through [`Ptr::offset`].
/// A kernel with a region keeps all its shared memory in the dynamic
/// shared-memory window (see [`SharedMemoryLayout`](crate::ir::SharedMemoryLayout)).
///
/// ```
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("scratch").block_dim([32, 1, 1]).build(|| {
///     let out = param_tensor::<u32>("out", Layout::col_major(32));
///     let t = thread_idx().x;
///     let scratch = make_tensor(&get_dyn_smem::<u32>(None), Layout::col_major(32));
///     scratch.store(t, t.to::<u32>() * 3u32);
///     sync_threads();
///     out.store(t, scratch.load(31 - t));
/// });
/// let mut args = vec![KernelArg::buffer(&[0u32; 32])];
/// let launch = LaunchConfig::new([1, 1, 1], [32, 1, 1]).with_dynamic_smem(128);
/// kernel.simulate(launch, &mut args).unwrap();
/// assert_eq!(args[0].to_vec::<u32>()[0], 93);
/// ```
///
/// [`LaunchConfig::with_dynamic_smem`]: crate::LaunchConfig::with_dynamic_smem
/// [`get_dyn_smem_size`]: crate::get_dyn_smem_size
#[track_caller]
pub fn get_dyn_smem<T: Numeric>(alignment: Option<usize>) -> Ptr<T> {
    let size = T::DTYPE.bytes();
    let align = alignment.unwrap_or(size);
    assert!(align.is_power_of_two(), "cute: get_dyn_smem: an alignment of {align} bytes is not a power of two");
    assert!(
        align >= size,
        "cute: get_dyn_smem: an alignment of {align} bytes is less than the size of {} ({size} bytes)",
        T::DTYPE
    );
    let buffer = builder::with(|b| b.dynamic_smem_region(align));
    Ptr { buffer, space: MemSpace::Shared, offset: IndexExpr::Static(0), swizzle: None, peer: None, _t: PhantomData }
}

/// A per-thread array that, unlike a register fragment, can be indexed with
/// runtime coordinates (CuTe DSL's `make_rmem_tensor`). It lives on the
/// thread's stack.
#[track_caller]
pub fn make_rmem_tensor<T: Numeric>(layout: Layout) -> Tensor<T> {
    layout.require_static("make_rmem_tensor", "a per-thread array is sized from its layout");
    let (elem, len) = storage_decl(T::DTYPE, layout.cosize());
    let buffer = builder::with(|b| {
        let name = format!("stack{}", b.local_buffer_count());
        b.declare_buffer(BufferDecl {
            name,
            elem,
            len: Some(len),
            space: MemSpace::Local,
            align: 16,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    Tensor::new(Engine::Memory { buffer, space: MemSpace::Local, peer: None }, layout)
}

/// Declare a scalar kernel parameter.
#[track_caller]
pub fn param_scalar<T: Numeric>(name: &str) -> Value<T> {
    let _at = builder::LocationScope::new();
    Value::from_id(builder::with(|b| b.declare_scalar_param(name, T::DTYPE)))
}

/// Declare a non-negative `i32` parameter that the host promises is a
/// multiple of `multiple_of`, such as a problem size that is a whole number
/// of tiles. Layouts can use it as a stride with that divisibility known,
/// which is what proves vector and `cp.async` accesses aligned. Launches
/// check the promise.
#[track_caller]
pub fn param_extent(name: &str, multiple_of: usize) -> Int32 {
    let _at = builder::LocationScope::new();
    Value::from_id(builder::with(|b| b.declare_extent_param(name, multiple_of, DType::I32)))
}

/// [`param_extent`] as a 64-bit parameter: a non-negative `i64` that the
/// host promises is a multiple of `multiple_of`. Index expressions it enters
/// (as a stride, say, through `IndexExpr::from`) are computed in 64 bits,
/// so offsets can go beyond `i32`. Launches check the promise.
#[track_caller]
pub fn param_extent_i64(name: &str, multiple_of: usize) -> Int64 {
    Value::from_id(builder::with(|b| b.declare_extent_param(name, multiple_of, DType::I64)))
}

/// A buffer index for `off`: an `i32`, or an `i64` when the offset is 64-bit
/// ([`IndexExpr::width`]).
#[track_caller]
pub(crate) fn index_operand(off: IndexExpr) -> ValueId {
    if off.width() == 64 {
        IntoValue::<i64>::into_value(off).checked()
    } else {
        IntoValue::<i32>::into_value(off).checked()
    }
}

/// The layouts a tensor in memory takes ([`make_tensor`],
/// [`make_smem_tensor`]): a [`Layout`], or a [`ComposedLayout`] (a swizzle
/// or a layout over the outer layout).
pub trait SmemLayout {
    /// The layout as a composed layout (a plain one with the identity
    /// swizzle and offset 0).
    fn into_composed(self) -> ComposedLayout;
}

impl SmemLayout for Layout {
    fn into_composed(self) -> ComposedLayout {
        ComposedLayout {
            inner: Inner::Swizzle(Swizzle::identity()),
            offset: Tree::Leaf(IndexExpr::Static(0)),
            outer: self,
        }
    }
}

impl SmemLayout for ComposedLayout {
    fn into_composed(self) -> ComposedLayout {
        self
    }
}

/// Allocate a shared-memory tensor (128-byte aligned). A swizzled layout's
/// allocation is rounded up to whole swizzle periods (so the swizzled
/// offsets stay inside it); a composed layout with a layout inner gets the
/// inner layout's cosize. [`make_smem_tensor_aligned`] takes another
/// alignment.
#[track_caller]
pub fn make_smem_tensor<T: Numeric>(layout: impl SmemLayout) -> Tensor<T> {
    make_smem_tensor_aligned(layout, 128)
}

/// [`make_smem_tensor`] with the allocation aligned to `align` bytes, a
/// power of two (CuTe DSL's `byte_alignment`): 1024 for the operand
/// buffers that TMA and WGMMA read with a 128-byte swizzle.
#[track_caller]
pub fn make_smem_tensor_aligned<T: Numeric>(layout: impl SmemLayout, align: usize) -> Tensor<T> {
    assert!(align.is_power_of_two(), "cute: shared memory alignment {align} is not a power of two");
    let c = layout.into_composed();
    let why = "a shared-memory allocation is sized from its layout";
    c.outer.require_static("make_smem_tensor", why);
    if let Inner::Layout(inner) = &c.inner {
        inner.require_static("make_smem_tensor: the inner layout", why);
    }
    let len = if c.is_normal() { c.outer.cosize() } else { c.allocation_size() };
    require_whole_bytes(T::DTYPE, len, "a shared-memory tensor");
    let (elem, len) = storage_decl(T::DTYPE, len);
    let buffer = builder::with(|b| {
        let name = format!("smem{}", b.shared_buffer_count());
        b.declare_buffer(BufferDecl {
            name,
            elem,
            len: Some(len),
            space: MemSpace::Shared,
            align,
            fields: Vec::new(),
            partition: crate::ir::SmemPartition::User,
        })
    });
    let ptr = Ptr::<T> {
        buffer,
        space: MemSpace::Shared,
        offset: IndexExpr::Static(0),
        swizzle: None,
        peer: None,
        _t: PhantomData,
    };
    make_tensor(&ptr, c)
}

/// A register fragment with a compact column-major layout of `shape`, which
/// must be static: a fragment is an array of registers.
#[track_caller]
pub fn make_fragment<T: Numeric>(shape: impl Into<Shape>) -> Tensor<T> {
    let layout = Layout::col_major(shape);
    layout.require_static("make_fragment", "register fragments have static shapes");
    make_fragment_with_layout(layout)
}

/// A register fragment with an explicit layout, which must be static.
#[track_caller]
pub fn make_fragment_with_layout<T: Numeric>(layout: Layout) -> Tensor<T> {
    layout.require_static("make_fragment_with_layout", "register fragments have static layouts");
    let slots = layout.cosize();
    Tensor::new(Engine::Register(RegisterStorage::new(vec![None; slots])), layout)
}

/// A register fragment shaped like `t`, whose shape must be static (CuTe
/// DSL's `make_fragment_like(tensor, dtype)`, which is
/// `make_rmem_tensor_like`): its layout is
/// [`make_layout_like`]`(t.layout())`, compact with the order of `t`'s
/// strides, so a fragment of a row-major tile is row-major and
/// [`autovec_copy`](crate::autovec_copy) between the two stays wide. A
/// static-0 stride stays 0 (a broadcast mode shares its registers). A
/// coordinate tensor gives a compact column-major fragment of its shape, and
/// so does a [`TensorSsa`](crate::TensorSsa).
#[track_caller]
pub fn make_fragment_like<T: Numeric, U>(t: &(impl FragmentLike<U> + ?Sized)) -> Tensor<T> {
    make_fragment_with_layout(t.fragment_layout())
}

/// What [`make_fragment_like`] makes a fragment like: a [`Tensor`], or a
/// [`TensorSsa`](crate::TensorSsa), whose fragment has the compact
/// column-major layout of its shape (as CuTe DSL's `make_rmem_tensor_like`
/// gives a `TensorSSA`). `U` is the element type.
pub trait FragmentLike<U> {
    #[doc(hidden)]
    fn fragment_layout(&self) -> Layout;
}

impl<U> FragmentLike<U> for Tensor<U> {
    #[track_caller]
    fn fragment_layout(&self) -> Layout {
        require_static_fragment_shape(self, "make_fragment_like");
        match self.engine {
            Engine::ArithTuple(_) => Layout::col_major(self.shape().clone()),
            _ => make_layout_like(&self.layout),
        }
    }
}

impl<U: Numeric> FragmentLike<U> for crate::TensorSsa<U> {
    fn fragment_layout(&self) -> Layout {
        Layout::col_major(self.shape().clone())
    }
}

/// A register fragment with the layout [`make_fragment_like_layout`]`(src)`
/// gives for a layout or a composed layout (CuTe DSL's
/// `make_fragment_like(layout, dtype)`), whose shape must be static.
///
/// [`make_fragment_like_layout`]: crate::make_fragment_like_layout
#[track_caller]
pub fn make_fragment_like_dtype<T: Numeric>(src: &(impl LayoutSource + ?Sized)) -> Tensor<T> {
    let layout = crate::layout::make_fragment_like_layout(src);
    layout.require_static("make_fragment_like_dtype", "register fragments have static shapes");
    make_fragment_with_layout(layout)
}

/// A register fragment with the compact column-major layout of `t`'s shape,
/// which must be static: what MMA operand fragments and [`Tensor::to`] use.
#[track_caller]
pub(crate) fn make_compact_fragment_like<T: Numeric, U>(t: &Tensor<U>) -> Tensor<T> {
    require_static_fragment_shape(t, "make_fragment_like");
    make_fragment_with_layout(Layout::col_major(t.shape().clone()))
}

/// Panic, naming the leaf, if `t`, which `what` makes a register fragment
/// shaped like, has a runtime extent.
#[track_caller]
pub(crate) fn require_static_fragment_shape<U>(t: &Tensor<U>, what: &str) {
    if let Some((path, leaf)) = t.shape().first_dynamic() {
        panic!(
            "{what}: tensor {} has the dynamic extent {leaf:?} at mode {path:?}, but register fragments have static \
             shapes (a fragment is an array of registers): make the fragment from one tile, whose extents are \
             static, and for a runtime number of tiles write a for_range over the rest mode",
            t.layout
        );
    }
}

/// A coordinate tensor (CuTe DSL's `make_tensor` of an integer or tuple
/// origin and a layout, whose iterator is an arithmetic tuple): the element
/// at `c` is `origin + layout(c)`, the tuple [`Layout::eval`] gives added
/// to the origin leaf by leaf (see [`Tensor::coord`]). The layout usually
/// has scaled-basis strides, `make_layout(shape, (1*E(0), 1*E(1)))` say;
/// one with integer strides gives an integer coordinate. The origin may
/// hold runtime values, which are taken to be of either sign (see
/// [`domain_offset`]). Slicing, composing, the divides,
/// [`local_tile`], [`local_partition`], the partitions of tiled copies and
/// [`make_predicate`](crate::make_predicate) take coordinate tensors as
/// they take memory tensors, and [`domain_offset`] shifts the origin.
#[track_caller]
pub fn make_coord_tensor(origin: impl Into<Coord>, layout: Layout) -> Tensor<Ident> {
    // A runtime origin may be negative, as a domain_offset shift may.
    let origin = origin.into().map(&mut IndexExpr::with_signed_runtime_values);
    Tensor::new(Engine::ArithTuple(origin), layout)
}

/// A tensor whose element at coordinate `c` is `c` itself (CuTe DSL's
/// `cute.make_identity_tensor`): the coordinate tensor of origin 0 over
/// [`make_identity_layout`](crate::make_identity_layout)`(shape)`. The shape
/// may have any rank and nesting, and runtime extents
/// (`make_identity_tensor(m_a.shape())`, as CuTe DSL's
/// `make_identity_tensor(mA.shape)`). Its elements are hierarchical
/// coordinates in colexicographic order, congruent with the shape: for
/// `((2,1),3)` they are `((0,0),0)`, `((1,0),0)`, `((0,0),1)`, ... A shape
/// that is a single integer gives integer coordinates.
#[track_caller]
pub fn make_identity_tensor(shape: impl Into<Shape>) -> Tensor<Ident> {
    make_coord_tensor(0, make_identity_layout(shape))
}

/// The tile of `t` at `coord`: `zipped_divide(t, tiler)[(.., coord)]`.
/// Modes of the rest not fixed by `coord` (use `..`) are kept after the tile
/// modes (CuTe's `local_tile`).
#[track_caller]
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
#[track_caller]
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
#[track_caller]
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
#[track_caller]
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
        let loc = std::panic::Location::caller();
        for i in 0..storage.len() {
            // Warpgroup-MMA accumulators not yet written start as zeros (an
            // MMA that does not accumulate ignores them; roadmap part
            // m8-wgmma-variants).
            if storage.wgmma_acc.get() && storage.peek(i).is_none() {
                let zero = builder::with(|b| b.constant(crate::ir::Literal::zero(T::DTYPE), loc));
                storage.write(i, zero);
            }
            out.push(storage.read(i).unwrap_or_else(|| {
                panic!("cute: register tensor slot {i} is uninitialised; fill() it before carrying it through a loop")
            }));
        }
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let n = self.register_storage().unwrap().len();
        let slots: Vec<Option<ValueId>> = (0..n).map(|_| Some(values.next().expect("carry: too few values"))).collect();
        let storage = RegisterStorage::new(slots);
        if self.register_storage().is_some_and(|s| s.wgmma_acc.get()) {
            crate::nvgpu::warpgroup::mark_accumulators(&storage, T::DTYPE);
        }
        if self.register_storage().is_some_and(|s| s.wgmma_a.get()) {
            crate::nvgpu::warpgroup::mark_a_fragment(&storage, T::DTYPE);
        }
        Tensor { engine: Engine::Register(storage), ..self.clone() }
    }

    /// Register tensors are rebuilt with the layout, offset and swizzle of
    /// the one rebuilt from, so these must match.
    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        let registers = |t: &Self| t.register_storage().map(|s| s.len());
        let same_registers = registers(self) == registers(other);
        if self.layout == other.layout && self.offset == other.offset && self.inner == other.inner && same_registers {
            return None;
        }
        let describe = |t: &Self| {
            let mut s = format!("layout {:?}", t.layout);
            if t.offset != IndexExpr::Static(0) {
                s += &format!(" at offset {:?}", t.offset);
            }
            if let Some(sw) = t.swizzle() {
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
            Engine::Memory { space: MemSpace::Raw(_), .. } => "raw",
            Engine::Memory { space: MemSpace::Constant, .. } => "cmem",
            Engine::Register(_) => "rmem",
            Engine::ArithTuple(_) => "coord",
            Engine::SmemDesc { .. } => "smem_desc",
        };
        write!(f, "tensor<{kind}")?;
        match &self.inner {
            TensorInner::Identity => {}
            TensorInner::Swizzle { elem, .. } => write!(f, ", {elem}")?,
            TensorInner::Layout { inner, base } => write!(f, ", {:?} + {inner:?} o", base)?,
        }
        match &self.engine {
            Engine::ArithTuple(origin) => write!(f, "> {origin:?} + {:?}", self.layout),
            _ => write!(f, "> {:?} + {:?}", self.layout, self.offset),
        }
    }
}

/// CuTe DSL's `cute.recast_tensor(src, dtype)`: the same storage viewed as
/// elements of `U` (`swizzle_` is reserved there and has no counterpart).
/// The layout is recast with [`recast_layout`](crate::recast_layout) (new
/// width first; Boolean counts as 8 bits, sub-byte types as their width) and
/// a swizzle moves with it.
///
/// * A memory tensor keeps its buffer: its pointer is cast
///   ([`Ptr::cast`], which must prove the start a whole number of `U`
///   values) and its layout recast. Every static non-zero stride must be a
///   multiple of the widening ratio, except the unit stride, whose extent
///   must be (the unit-stride mode divides evenly), or the build panics.
/// * A register fragment is repacked: narrowing unpacks each register into
///   several values ([`Op::Unpack`], element 0 from the low bits), widening
///   packs consecutive registers into one ([`Op::Pack`]), and equal widths
///   reinterpret the bits. Its storage is indexed as memory is, so the ratio
///   must divide its number of registers.
///
/// ```
/// use cute::prelude::*;
///
/// // Bytes viewed as Int4: twice the elements, element 0 in the low nibble.
/// let kernel = Kernel::builder("bytes_as_int4").block_dim([32, 1, 1]).build(|| {
///     let bytes = param_tensor::<u8>("bytes", Layout::col_major(128));
///     let out = param_tensor::<i8>("out", Layout::col_major(256));
///     let nibbles = recast_tensor::<I4, _>(&bytes);
///     assert_eq!(nibbles.size(), 256);
///     let t = thread_idx().x;
///     let frag = make_fragment::<I4>(8);
///     let mine = local_tile(&nibbles, 8, t);
///     autovec_copy(&mine, &frag);
///     for i in 0..8 {
///         out.store(t * 8 + i as i32, frag.load(i).to::<i8>());
///     }
/// });
/// let mut args = vec![KernelArg::buffer(&[0x9fu8; 128]), KernelArg::buffer(&[0i8; 256])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(&args[1].to_vec::<i8>()[..2], &[-1, -7]);
/// ```
#[track_caller]
pub fn recast_tensor<U: Numeric, T: Numeric>(src: &Tensor<T>) -> Tensor<U> {
    let (tb, ub) = (T::DTYPE.storage_bits(), U::DTYPE.storage_bits());
    let recast_swizzle = |sw: Swizzle| {
        if ub > tb {
            assert!(ub.is_multiple_of(tb), "recast_tensor: a swizzled tensor needs widths in a power-of-two ratio");
            sw.upcast(ub / tb)
        } else if tb > ub {
            assert!(tb.is_multiple_of(ub), "recast_tensor: a swizzled tensor needs widths in a power-of-two ratio");
            sw.downcast(tb / ub)
        } else {
            sw
        }
    };
    match &src.engine {
        Engine::ArithTuple(_) => panic!("cute: recast_tensor: a coordinate tensor holds coordinates, not elements"),
        Engine::SmemDesc { .. } => panic!("cute: recast_tensor: a WGMMA fragment holds matrix descriptors"),
        Engine::Memory { .. } => {
            let g = gcd(tb, ub);
            let up = ub / g;
            if up > 1 {
                let shape = src.layout.shape().flat_leaves();
                let stride = src.layout.stride().flat_leaves();
                // The unit-stride mode first: it is the one that must divide.
                let mut leaves: Vec<_> = shape.iter().zip(&stride).collect();
                leaves.sort_by_key(|(_, d)| {
                    !matches!(d, crate::layout::StrideLeaf::Int(d) if matches!(d.as_static_signed(), Some(1 | -1)))
                });
                for (s, d) in leaves {
                    let crate::layout::StrideLeaf::Int(d) = d else { continue };
                    match d.as_static_signed() {
                        Some(0) | None => {}
                        Some(1 | -1) => assert!(
                            s.as_static().is_some_and(|s| s.is_multiple_of(up)),
                            "cute: recast_tensor from {} to {}: the unit-stride mode of {} has extent {s:?}, which is \
                             not a multiple of {up}, so it does not divide into whole {} values",
                            T::DTYPE.cute_name(),
                            U::DTYPE.cute_name(),
                            src.layout,
                            U::DTYPE.cute_name()
                        ),
                        Some(v) => assert!(
                            v.unsigned_abs().is_multiple_of(up as u64),
                            "cute: recast_tensor from {} to {}: the stride {v} of {} is not a multiple of {up}, so \
                             its elements do not start whole {} values",
                            T::DTYPE.cute_name(),
                            U::DTYPE.cute_name(),
                            src.layout,
                            U::DTYPE.cute_name()
                        ),
                    }
                }
            }
            let layout = crate::layout::recast_layout(ub, tb, &src.layout);
            if src.inner_layout().is_some() {
                panic!(
                    "cute: recast_tensor: tensor {} is over a composed layout with a layout inner, which \
                     recast_tensor does not recast",
                    src.layout
                );
            }
            let (buffer, space, peer, offset) = match src.engine {
                Engine::Memory { buffer, space, peer } => (buffer, space, peer, src.offset.clone()),
                _ => unreachable!("a memory tensor"),
            };
            let ptr = Ptr::<T> { buffer, space, offset, swizzle: None, peer, _t: PhantomData }.cast::<U>();
            let inner = match (src.swizzle(), src.swizzle_in_layout()) {
                (Some(sw), in_layout) => match TensorInner::swizzle::<U>(recast_swizzle(sw)) {
                    TensorInner::Swizzle { elem, bytes, .. } => TensorInner::Swizzle { elem, bytes, in_layout },
                    other => other,
                },
                (None, _) => TensorInner::Identity,
            };
            Tensor { inner, ..make_tensor(&ptr, layout) }
        }
        Engine::Register(_) => {
            assert!(
                T::DTYPE != DType::Bool && U::DTYPE != DType::Bool,
                "cute: recast_tensor: Boolean fragments have no bit layout to repack"
            );
            let layout = crate::layout::recast_layout(ub, tb, &src.layout);
            let out = make_fragment_with_layout::<U>(layout);
            let (n_src, n_dst) = (src.layout.cosize(), out.layout.cosize());
            let loc = Location::caller();
            let read = |i: usize| src.load_at(IndexExpr::Static(i)).checked();
            if ub == tb {
                for i in 0..n_src {
                    let x = read(i);
                    let v = builder::with(|b| b.bitcast(x, U::DTYPE, loc));
                    out.store_at(IndexExpr::Static(i), Value::from_id(v));
                }
            } else if tb > ub {
                assert!(
                    tb.is_multiple_of(ub),
                    "cute: recast_tensor: {tb}-bit registers do not split into {ub}-bit ones"
                );
                let k = tb / ub;
                for j in 0..n_src {
                    let x = read(j);
                    let parts = builder::with(|b| b.unpack(x, U::DTYPE, k, loc));
                    for (m, p) in parts.into_iter().enumerate() {
                        out.store_at(IndexExpr::Static(j * k + m), Value::from_id(p));
                    }
                }
            } else {
                assert!(
                    ub.is_multiple_of(tb),
                    "cute: recast_tensor: {tb}-bit registers do not pack into {ub}-bit ones"
                );
                let k = ub / tb;
                assert!(
                    n_src == n_dst * k,
                    "cute: recast_tensor: a fragment of {n_src} {} registers does not pack into whole {} values",
                    T::DTYPE.cute_name(),
                    U::DTYPE.cute_name()
                );
                for j in 0..n_dst {
                    let parts: Vec<ValueId> = (0..k).map(|m| read(j * k + m)).collect();
                    let v = builder::with(|b| b.pack(parts, U::DTYPE, loc));
                    out.store_at(IndexExpr::Static(j), Value::from_id(v));
                }
            }
            out
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl<T: Numeric> Tensor<T> {
    /// [`recast_tensor`] as a method: `t.recast::<I4>()`.
    #[track_caller]
    pub fn recast<U: Numeric>(&self) -> Tensor<U> {
        recast_tensor::<U, T>(self)
    }
}

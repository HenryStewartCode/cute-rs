// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared-storage structs: CuTe DSL's `@cute.struct`, `cute.struct.MemRange`,
//! `cute.struct.Align` and `@cute.union`, and the struct form of
//! `SmemAllocator.allocate`.
//!
//! A struct is laid out in one byte buffer of shared memory. Its members are
//! scalars, ranges of elements ([`MemRange`]) and nested structs, each placed
//! at the next offset that is a multiple of its alignment: a scalar's and a
//! range's alignment is its element's size (at least 1 byte), a nested
//! struct's is its own, and `#[align(n)]` (CuTe DSL's `Align[T, n]`) raises
//! a member's alignment to `n`. The struct's alignment is its members'
//! largest, and its size the end of its last member rounded up to it. A
//! union puts every member at offset 0.
//!
//! CuTe DSL computes member sizes when the kernel is traced
//! (`MemRange[Int64, 2 * stages]`), which for cute-rs is when the kernel is
//! built: ordinary Rust run time. So the layout is a runtime value, a
//! [`StructLayout`], built by a [`StructBuilder`] (whose field handles give
//! typed member views) or declared with [`cute_struct!`](crate::cute_struct)
//! and [`cute_union!`](crate::cute_union), which generate a view type whose
//! fields are the members and whose build-time parameters size them.
//!
//! Member pointers are the struct's byte pointer offset by the member's
//! offset and cast to the element type; the offsets are multiples of the
//! element size by construction, so [`Ptr::cast`]'s alignment proof always
//! holds. The MLIR backend accesses the byte buffer through a typed view per
//! element type, and the simulator's byte-addressed memory and per-byte race
//! tracking handle members of different types and the aliasing of a union.

use std::fmt;
use std::marker::PhantomData;

use crate::builder;
use crate::ir::DType;
use crate::layout::{ComposedLayout, IndexExpr, Layout, Swizzle};
use crate::tensor::{Ptr, SmemLayout, Tensor, alloc_smem, make_tensor, recast_ptr};
use crate::value::{IntoValue, Numeric, Value};

/// The largest alignment shared memory gives (CuTe DSL's `SmemAllocator`
/// caps alignments at 1024 bytes).
pub const MAX_STRUCT_ALIGN: usize = 1024;

/// What a struct member is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemberKind {
    /// One value (CuTe DSL's scalar member, `_ScalarData` in a view).
    Scalar(DType),
    /// `len` values (CuTe DSL's `MemRange[elem, len]`).
    MemRange { elem: DType, len: usize },
    /// A nested struct or union.
    Struct(StructLayout),
}

/// One member of a [`StructLayout`]: where it is and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructMember {
    pub name: String,
    pub kind: MemberKind,
    /// Byte offset from the start of the struct.
    pub offset: usize,
    /// Bytes the member takes.
    pub size: usize,
    /// The member's alignment: its natural one, raised by `#[align(n)]`.
    pub align: usize,
}

impl StructMember {
    /// The scalar member's view at the struct's byte pointer `base`.
    /// Panics unless the member is a scalar of `T`.
    #[track_caller]
    pub fn scalar<T: Numeric>(&self, base: &Ptr<u8>) -> ScalarMember<T> {
        match self.kind {
            MemberKind::Scalar(t) if t == T::DTYPE => ScalarMember { ptr: member_ptr(base, self.offset) },
            _ => panic!("cute struct: member `{}` is {}, not a {} scalar", self.name, self.kind, T::DTYPE.cute_name()),
        }
    }

    /// The range member's view at the struct's byte pointer `base`. Panics
    /// unless the member is a range of `T`.
    #[track_caller]
    pub fn mem_range<T: Numeric>(&self, base: &Ptr<u8>) -> MemRange<T> {
        match self.kind {
            MemberKind::MemRange { elem, len } if elem == T::DTYPE => MemRange {
                ptr: member_ptr(base, self.offset),
                len,
                byte_offset: buffer_offset(base, self.offset),
                name: self.name.clone(),
            },
            _ => panic!(
                "cute struct: member `{}` is {}, not a MemRange of {}",
                self.name,
                self.kind,
                T::DTYPE.cute_name()
            ),
        }
    }

    /// The nested struct's layout. Panics unless the member is a struct.
    #[track_caller]
    pub fn nested_layout(&self) -> &StructLayout {
        match &self.kind {
            MemberKind::Struct(s) => s,
            other => panic!("cute struct: member `{}` is {other}, not a struct", self.name),
        }
    }

    /// The nested struct's byte pointer, at the struct's byte pointer `base`.
    pub fn nested_base(&self, base: &Ptr<u8>) -> Ptr<u8> {
        base.offset(self.offset)
    }
}

impl fmt::Display for MemberKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemberKind::Scalar(t) => write!(f, "a {} scalar", t.cute_name()),
            MemberKind::MemRange { elem, len } => write!(f, "MemRange[{}, {len}]", elem.cute_name()),
            MemberKind::Struct(s) => write!(f, "{} {}", if s.is_union { "union" } else { "struct" }, s.name),
        }
    }
}

/// The member at byte offset `offset` of the struct at `base`, as a
/// pointer to `T`.
#[track_caller]
fn member_ptr<T: Numeric>(base: &Ptr<u8>, offset: usize) -> Ptr<T> {
    assert!(
        base.swizzle().is_none_or(|s| s.is_identity()),
        "cute struct: the struct's base pointer is swizzled; a struct takes a plain byte pointer"
    );
    base.offset(offset).cast::<T>()
}

/// The byte offset in its buffer of the member at `offset` in the struct at
/// `base`, when `base` is a static offset.
fn buffer_offset(base: &Ptr<u8>, offset: usize) -> Option<usize> {
    base.offset.as_static().map(|b| b + offset)
}

/// The layout of a shared-storage struct or union (CuTe DSL's
/// `@cute.struct` and `@cute.union` classes): its members with their
/// offsets, sizes and alignments, and the struct's size and alignment
/// (CuTe DSL's `__sizeof__`, `__alignof__` and `size_in_bytes`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout {
    name: String,
    is_union: bool,
    members: Vec<StructMember>,
    size: usize,
    align: usize,
}

impl StructLayout {
    /// A builder for a struct named `name`.
    pub fn builder(name: &str) -> StructBuilder {
        StructLayout::__builder(name, false)
    }

    /// A builder for a union named `name`: every member at offset 0.
    pub fn union_builder(name: &str) -> StructBuilder {
        StructLayout::__builder(name, true)
    }

    #[doc(hidden)]
    pub fn __builder(name: &str, is_union: bool) -> StructBuilder {
        StructBuilder { name: name.to_string(), is_union, members: Vec::new(), end: 0, align: 1 }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_union(&self) -> bool {
        self.is_union
    }

    /// Bytes the struct takes, a multiple of its alignment (CuTe DSL's
    /// `__sizeof__()`).
    pub fn size_of(&self) -> usize {
        self.size
    }

    /// The struct's alignment, its members' largest (CuTe DSL's
    /// `__alignof__()`).
    pub fn align_of(&self) -> usize {
        self.align
    }

    /// The same as [`size_of`](StructLayout::size_of) (CuTe DSL's
    /// `size_in_bytes()`).
    pub fn size_in_bytes(&self) -> usize {
        self.size
    }

    pub fn members(&self) -> &[StructMember] {
        &self.members
    }

    /// The member named `name`. Panics if there is none.
    #[track_caller]
    pub fn member(&self, name: &str) -> &StructMember {
        self.members
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("cute struct: {} has no member `{name}`", self.name))
    }

    /// The field table, `name:size:offset` per member, as CuTe DSL's
    /// `SmemAllocator` records it.
    pub fn field_table(&self) -> Vec<String> {
        self.members.iter().map(|m| format!("{}:{}:{}", m.name, m.size, m.offset)).collect()
    }

    /// Allocate the struct in shared memory, aligned to its alignment, and
    /// return its byte pointer (CuTe DSL's `SmemAllocator.allocate(Struct)`).
    #[track_caller]
    pub fn allocate(&self) -> Ptr<u8> {
        self.allocate_aligned(1)
    }

    /// [`allocate`](StructLayout::allocate) aligned to `byte_alignment` too:
    /// the allocation is aligned to the larger of `byte_alignment` and the
    /// struct's alignment, which must be a power of two of at most 1024
    /// bytes. The buffer records the field table, which
    /// [`Kernel::dump`](crate::Kernel::dump) shows.
    #[track_caller]
    pub fn allocate_aligned(&self, byte_alignment: usize) -> Ptr<u8> {
        check_align(byte_alignment, &format!("allocating {}", self.name));
        let align = byte_alignment.max(self.align);
        // A struct of zero-size markers still takes a byte.
        let ptr = alloc_smem::<u8>(self.size.max(1), align);
        let mut fields = vec![self.name.clone()];
        fields.extend(self.field_table());
        builder::with(|b| b.buffer_mut(ptr.buffer).fields = fields);
        ptr
    }
}

impl fmt::Display for StructLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_union { "union" } else { "struct" };
        write!(
            f,
            "{kind} {} (size {}, align {}) {{ {} }}",
            self.name,
            self.size,
            self.align,
            self.field_table().join(", ")
        )
    }
}

#[track_caller]
fn check_align(align: usize, what: &str) {
    assert!(align.is_power_of_two(), "cute struct: {what}: an alignment of {align} bytes is not a power of two");
    assert!(
        align <= MAX_STRUCT_ALIGN,
        "cute struct: {what}: an alignment of {align} bytes is more than shared memory's {MAX_STRUCT_ALIGN}"
    );
}

/// Builds a [`StructLayout`] member by member, in order (or, for a union,
/// all at offset 0). Each member takes an optional alignment (CuTe DSL's
/// `Align[member, n]`), a power of two of at most 1024 that raises the
/// member's natural alignment. The handles it returns give typed views of
/// the member at a struct's byte pointer.
pub struct StructBuilder {
    name: String,
    is_union: bool,
    members: Vec<StructMember>,
    end: usize,
    align: usize,
}

impl StructBuilder {
    #[track_caller]
    fn push(&mut self, name: &str, kind: MemberKind, size: usize, natural: usize, align: Option<usize>) -> usize {
        let what = format!("member `{name}` of {}", self.name);
        assert!(
            self.members.iter().all(|m| m.name != name),
            "cute struct: {} has two members named `{name}`",
            self.name
        );
        check_align(natural, &what);
        if let Some(a) = align {
            check_align(a, &what);
        }
        let align = natural.max(align.unwrap_or(1));
        let offset = if self.is_union { 0 } else { self.end.next_multiple_of(align) };
        self.end = if self.is_union { self.end.max(size) } else { offset + size };
        self.align = self.align.max(align);
        self.members.push(StructMember { name: name.to_string(), kind, offset, size, align });
        offset
    }

    /// A scalar member of type `T`: `max(1, width / 8)` bytes, aligned to
    /// that size.
    #[track_caller]
    pub fn scalar<T: Numeric>(&mut self, name: &str, align: Option<usize>) -> ScalarField<T> {
        let size = T::DTYPE.bytes();
        let offset = self.push(name, MemberKind::Scalar(T::DTYPE), size, size, align);
        ScalarField { offset, _t: PhantomData }
    }

    /// A range of `len` values of `T` (CuTe DSL's `MemRange[T, len]`):
    /// `len * width / 8` bytes (a Boolean counts as a byte), aligned to the
    /// element's size (at least 1). `len` may be 0, which makes the member
    /// an address marker. Sub-byte elements must fill whole bytes.
    #[track_caller]
    pub fn mem_range<T: Numeric>(&mut self, name: &str, len: usize, align: Option<usize>) -> RangeField<T> {
        let t = T::DTYPE;
        let bits = if t == DType::Bool { 8 } else { t.bits() };
        assert!(
            (len * bits).is_multiple_of(8),
            "cute struct: member `{name}` of {}: MemRange[{}, {len}] is {} bits, not a whole number of bytes",
            self.name,
            t.cute_name(),
            len * bits
        );
        let natural = (bits / 8).max(1);
        let offset = self.push(name, MemberKind::MemRange { elem: t, len }, len * bits / 8, natural, align);
        RangeField { offset, len, name: name.to_string(), _t: PhantomData }
    }

    /// A nested struct or union, with its own size and alignment.
    #[track_caller]
    pub fn nested(&mut self, name: &str, layout: &StructLayout, align: Option<usize>) -> NestedField {
        let offset = self.push(name, MemberKind::Struct(layout.clone()), layout.size, layout.align, align);
        NestedField { offset }
    }

    /// The layout. Panics for a struct without members (CuTe DSL refuses
    /// empty structs and unions).
    #[track_caller]
    pub fn build(self) -> StructLayout {
        let kind = if self.is_union { "union" } else { "struct" };
        assert!(
            !self.members.is_empty(),
            "cute struct: {kind} {} has no members; an empty {kind} is refused",
            self.name
        );
        StructLayout {
            size: self.end.next_multiple_of(self.align),
            name: self.name,
            is_union: self.is_union,
            members: self.members,
            align: self.align,
        }
    }
}

/// A scalar member's place in its struct, from [`StructBuilder::scalar`].
pub struct ScalarField<T> {
    pub offset: usize,
    _t: PhantomData<fn() -> T>,
}

impl<T: Numeric> ScalarField<T> {
    /// The member's view in the struct at `base`.
    #[track_caller]
    pub fn at(&self, base: &Ptr<u8>) -> ScalarMember<T> {
        ScalarMember { ptr: member_ptr(base, self.offset) }
    }
}

/// A range member's place in its struct, from [`StructBuilder::mem_range`].
pub struct RangeField<T> {
    pub offset: usize,
    pub len: usize,
    name: String,
    _t: PhantomData<fn() -> T>,
}

impl<T: Numeric> RangeField<T> {
    /// The member's view in the struct at `base`.
    #[track_caller]
    pub fn at(&self, base: &Ptr<u8>) -> MemRange<T> {
        MemRange {
            ptr: member_ptr(base, self.offset),
            len: self.len,
            byte_offset: buffer_offset(base, self.offset),
            name: self.name.clone(),
        }
    }
}

/// A nested struct's place in its struct, from [`StructBuilder::nested`].
pub struct NestedField {
    pub offset: usize,
}

impl NestedField {
    /// The nested struct's byte pointer in the struct at `base`.
    pub fn at(&self, base: &Ptr<u8>) -> Ptr<u8> {
        base.offset(self.offset)
    }
}

/// A scalar member of a struct view (CuTe DSL's `_ScalarData`): its
/// pointer, and loads and stores (CuTe DSL's assignment to the member,
/// which converts the value to the member's type: [`store_from`]).
///
/// [`store_from`]: ScalarMember::store_from
pub struct ScalarMember<T> {
    ptr: Ptr<T>,
}

impl<T> Clone for ScalarMember<T> {
    fn clone(&self) -> Self {
        ScalarMember { ptr: self.ptr.clone() }
    }
}

impl<T: Numeric> ScalarMember<T> {
    /// The member's address (CuTe DSL's `.ptr`).
    pub fn ptr(&self) -> Ptr<T> {
        self.ptr.clone()
    }

    #[track_caller]
    pub fn load(&self) -> Value<T> {
        self.ptr.load()
    }

    #[track_caller]
    pub fn store(&self, value: impl IntoValue<T>) {
        self.ptr.store(value)
    }

    /// Store `value` converted to the member's type (CuTe DSL's
    /// `s.x = value`).
    #[track_caller]
    pub fn store_from<U: Numeric>(&self, value: Value<U>) {
        self.ptr.store(value.to::<T>())
    }
}

/// A range member of a struct view (CuTe DSL's `MemRange[T, n]` member):
/// `len` values of `T`, as a pointer, a tensor, or element by element.
pub struct MemRange<T> {
    ptr: Ptr<T>,
    len: usize,
    byte_offset: Option<usize>,
    name: String,
}

impl<T> Clone for MemRange<T> {
    fn clone(&self) -> Self {
        MemRange { ptr: self.ptr.clone(), len: self.len, byte_offset: self.byte_offset, name: self.name.clone() }
    }
}

impl<T: Numeric> MemRange<T> {
    /// The range's first element (CuTe DSL's `data_ptr()`); for a range of
    /// length 0, the address it marks.
    pub fn data_ptr(&self) -> Ptr<T> {
        self.ptr.clone()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes the range takes.
    pub fn size_in_bytes(&self) -> usize {
        let t = T::DTYPE;
        self.len * if t == DType::Bool { 8 } else { t.bits() } / 8
    }

    /// Byte offset of the range in the shared buffer it lives in (the
    /// outermost struct's, for a nested struct's member), or `None` when the
    /// struct's base pointer is a runtime offset into it.
    pub fn byte_offset(&self) -> Option<usize> {
        self.byte_offset
    }

    #[track_caller]
    fn require_room(&self, elems: usize, what: &str) {
        assert!(self.len > 0, "cute struct: {what} on MemRange `{}` of length 0, an address marker", self.name);
        assert!(
            elems <= self.len,
            "cute struct: {what} on MemRange `{}`: the layout needs {elems} elements, but the range has {}",
            self.name,
            self.len
        );
    }

    /// A tensor over the range with `layout` (CuTe DSL's
    /// `get_tensor(layout)`). The layout must be static and its cosize fit
    /// the range.
    #[track_caller]
    pub fn get_tensor(&self, layout: Layout) -> Tensor<T> {
        layout.require_static("MemRange::get_tensor", "a struct member has a size fixed when the kernel is built");
        self.require_room(layout.cosize(), "get_tensor");
        make_tensor(&self.ptr, layout)
    }

    /// A tensor over the range with a composed layout (CuTe DSL's
    /// `get_tensor(composed_layout)`), such as a swizzled layout from
    /// [`utils::sm90`](crate::utils::sm90). Its swizzle acts on element
    /// offsets from the range's start, which must be a multiple of the
    /// swizzle's period into the buffer (`#[align(..)]` the member). A
    /// swizzled layout's extent is rounded up to whole swizzle periods, as
    /// [`make_smem_tensor`](crate::make_smem_tensor) sizes it.
    #[track_caller]
    pub fn get_tensor_composed(&self, layout: ComposedLayout) -> Tensor<T> {
        layout.outer().require_static(
            "MemRange::get_tensor_composed",
            "a struct member has a size fixed when the kernel is built",
        );
        let need = if layout.is_normal() { layout.outer().cosize() } else { layout.allocation_size() };
        self.require_room(need, "get_tensor_composed");
        make_tensor(&self.ptr, layout.into_composed())
    }

    /// A tensor over the range with `layout`, through a pointer swizzled by
    /// `swizzle` on byte addresses (CuTe DSL's `get_tensor(layout,
    /// swizzle=sw)`). The swizzle acts on absolute addresses, so the range
    /// must start at a multiple of its period (`#[align(1024)]` for
    /// `Swizzle<3,4,3>`, with the struct aligned as much) and hold the
    /// layout's cosize rounded up to whole periods. A composed layout and a
    /// swizzle together, which CuTe DSL refuses, cannot be written.
    #[track_caller]
    pub fn get_tensor_swizzled(&self, layout: Layout, swizzle: Swizzle) -> Tensor<T> {
        layout.require_static(
            "MemRange::get_tensor_swizzled",
            "a struct member has a size fixed when the kernel is built",
        );
        let elem_bits = T::DTYPE.bits();
        let period_elems = (swizzle.period() * 8).div_ceil(elem_bits).max(1);
        self.require_room(layout.cosize().next_multiple_of(period_elems), "get_tensor_swizzled");
        assert!(
            self.byte_offset.is_some_and(|b| b.is_multiple_of(swizzle.period())),
            "cute struct: get_tensor_swizzled on MemRange `{}` at byte offset {:?}: the swizzle {swizzle} acts on \
             absolute addresses, so the range must start at a known multiple of its period of {} bytes (give the \
             member #[align({})])",
            self.name,
            self.byte_offset,
            swizzle.period(),
            swizzle.period()
        );
        make_tensor(&recast_ptr::<T, T>(&self.ptr, Some(swizzle)), layout)
    }

    /// A tensor over the range's bytes as elements of `U` (CuTe DSL's
    /// `get_tensor(layout, dtype=U)`). The layout's cosize in `U` must fit
    /// the range's bytes.
    #[track_caller]
    pub fn get_tensor_as<U: Numeric>(&self, layout: Layout) -> Tensor<U> {
        layout.require_static("MemRange::get_tensor_as", "a struct member has a size fixed when the kernel is built");
        let bits = layout.cosize() * U::DTYPE.bits();
        let have = self.size_in_bytes() * 8;
        assert!(self.len > 0, "cute struct: get_tensor_as on MemRange `{}` of length 0, an address marker", self.name);
        assert!(
            bits <= have,
            "cute struct: get_tensor_as on MemRange `{}`: the layout needs {bits} bits of {}, but the range has {have}",
            self.name,
            U::DTYPE.cute_name()
        );
        make_tensor(&self.ptr.cast::<U>(), layout)
    }

    #[track_caller]
    fn element(&self, i: IndexExpr, what: &str) -> Ptr<T> {
        if let Some(i) = i.as_static() {
            assert!(
                i < self.len,
                "cute struct: {what} of element {i} of MemRange `{}`, which has {} elements",
                self.name,
                self.len
            );
        }
        self.ptr.offset(i)
    }

    /// Element `i` (CuTe DSL's `range[i]`). A static index is checked against
    /// the length when the kernel is built; the simulator checks runtime
    /// ones.
    #[track_caller]
    pub fn load(&self, i: impl Into<IndexExpr>) -> Value<T> {
        self.element(i.into(), "a load").load()
    }

    /// Write element `i` (CuTe DSL's `range[i] = v`), checked as
    /// [`load`](MemRange::load) is.
    #[track_caller]
    pub fn store(&self, i: impl Into<IndexExpr>, value: impl IntoValue<T>) {
        self.element(i.into(), "a store").store(value)
    }

    /// Write element `i` converted to the range's type.
    #[track_caller]
    pub fn store_from<U: Numeric>(&self, i: impl Into<IndexExpr>, value: Value<U>) {
        self.element(i.into(), "a store").store(value.to::<T>())
    }
}

/// Declare a shared-storage struct (CuTe DSL's `@cute.struct`): a view type
/// whose fields are the members, laid out in one byte buffer by a
/// [`StructLayout`](crate::StructLayout).
///
/// ```
/// use cute::prelude::*;
///
/// cute_struct! {
///     /// Two f32s.
///     pub struct Complex { real: f32, imag: f32 }
/// }
///
/// cute_struct! {
///     pub struct SharedStorage(stages: usize) {
///         mbar: MemRange<u64>[2 * stages],
///         #[align(1024)]
///         staging: MemRange<f32>[stages],
///         alpha: f32,
///         z: Complex(),
///     }
/// }
///
/// let layout = SharedStorage::layout(4);
/// assert_eq!(layout.field_table(), ["mbar:64:0", "staging:16:1024", "alpha:4:1040", "z:8:1044"]);
/// assert_eq!((SharedStorage::size_of(4), SharedStorage::align_of(4)), (2048, 1024));
///
/// let kernel = Kernel::builder("storage").block_dim([32, 1, 1]).build(|| {
///     let out = param_tensor::<f32>("out", Layout::col_major(1));
///     let s = SharedStorage::allocate(4);
///     if_(thread_idx().x.eq(0), || {
///         s.alpha.store(1.5f32);
///         s.z.real.store_from(Int32::from(2));
///         s.staging.store(3, s.alpha.load() + s.z.real.load());
///         out.store(0, s.staging.load(3));
///     });
/// });
/// let mut args = vec![KernelArg::f32(vec![0.0])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].as_f32()[0], 3.5);
/// ```
///
/// A member is a scalar (`alpha: f32`, any [`Numeric`](crate::Numeric)
/// type), a range of `n` values (`mbar: MemRange<u64>[n]`, `n` any `usize`
/// expression of the parameters, 0 for an address marker) or a nested
/// struct or union with its arguments (`z: Complex()`, `inner:
/// Inner(stages)`; the parentheses are needed). `#[align(n)]` before a
/// member is CuTe DSL's `Align[member, n]`. The struct may take type
/// parameters, each a `Numeric` type (`struct Storage<AB>(n: usize) { a:
/// MemRange<AB>[n] }`), and build-time parameters in parentheses, which
/// every associated function takes:
///
/// * `layout(params)`, the [`StructLayout`](crate::StructLayout);
///   `size_of(params)`, `align_of(params)` and `size_in_bytes(params)`;
/// * `allocate(params)` in shared memory (CuTe DSL's
///   `SmemAllocator.allocate(Struct)`) and `allocate_aligned(byte_alignment,
///   params)`;
/// * `at(&base, params)`, the view of the struct at a byte pointer, and
///   `from_layout(&layout, &base)`;
/// * the method `base()`, the struct's byte pointer.
///
/// Scalar members are [`ScalarMember`](crate::ScalarMember)s, ranges
/// [`MemRange`](crate::MemRange)s and nested structs their own view types.
#[macro_export]
macro_rules! cute_struct {
    ($(#[$meta:meta])* $vis:vis struct $name:ident $(<$($g:ident),+ $(,)?>)? $(($($p:ident : $pt:ty),* $(,)?))? { $($body:tt)* }) => {
        $crate::__cute_aggregate! { @parse [struct] [$(#[$meta])*] [$vis] [$name] [$($($g),+)?] [$($($p : $pt),*)?] [] $($body)* }
    };
}

/// Declare a shared-storage union (CuTe DSL's `@cute.union`): as
/// [`cute_struct!`](crate::cute_struct), with every member at offset 0. Its
/// alignment is the largest member alignment and its size the largest
/// member size rounded up to it. Used to alias the mainloop's operand
/// buffers with the epilogue's staging buffers.
///
/// ```
/// use cute::prelude::*;
///
/// cute_union! {
///     pub struct Scratch { halves: MemRange<F16>[4096], words: MemRange<f32>[1024], #[align(16)] tag: u64 }
/// }
/// let l = Scratch::layout();
/// assert_eq!((l.size_of(), l.align_of()), (8192, 16));
/// assert_eq!(l.field_table(), ["halves:8192:0", "words:4096:0", "tag:8:0"]);
/// ```
#[macro_export]
macro_rules! cute_union {
    ($(#[$meta:meta])* $vis:vis struct $name:ident $(<$($g:ident),+ $(,)?>)? $(($($p:ident : $pt:ty),* $(,)?))? { $($body:tt)* }) => {
        $crate::__cute_aggregate! { @parse [union] [$(#[$meta])*] [$vis] [$name] [$($($g),+)?] [$($($p : $pt),*)?] [] $($body)* }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __cute_aggregate {
    // A range member.
    (@parse $kind:tt $meta:tt $vis:tt $name:tt $gen:tt $params:tt [$($acc:tt)*]
        $(#[align($a:expr)])? $f:ident : MemRange<$t:ty>[$n:expr] $(, $($rest:tt)*)?) => {
        $crate::__cute_aggregate! { @parse $kind $meta $vis $name $gen $params
            [$($acc)* [$f] (range [$t] [$n] [$($a)?])] $($($rest)*)? }
    };
    // A nested struct.
    (@parse $kind:tt $meta:tt $vis:tt $name:tt $gen:tt $params:tt [$($acc:tt)*]
        $(#[align($a:expr)])? $f:ident : $s:ident ( $($arg:expr),* $(,)? ) $(, $($rest:tt)*)?) => {
        $crate::__cute_aggregate! { @parse $kind $meta $vis $name $gen $params
            [$($acc)* [$f] (nested [$s] [$($arg),*] [$($a)?])] $($($rest)*)? }
    };
    // A scalar.
    (@parse $kind:tt $meta:tt $vis:tt $name:tt $gen:tt $params:tt [$($acc:tt)*]
        $(#[align($a:expr)])? $f:ident : $t:ty $(, $($rest:tt)*)?) => {
        $crate::__cute_aggregate! { @parse $kind $meta $vis $name $gen $params
            [$($acc)* [$f] (scalar [$t] [$($a)?])] $($($rest)*)? }
    };
    // Every member parsed.
    (@parse [$kind:ident] [$($meta:tt)*] [$vis:vis] [$name:ident] [$($g:ident),*] [$($p:ident : $pt:ty),*]
        [$([$f:ident] $m:tt)*]) => {
        $($meta)*
        $vis struct $name<$($g: $crate::Numeric),*> {
            $(pub $f: $crate::__cute_member!(@ty $m),)*
            __base: $crate::Ptr<u8>,
            __marker: ::core::marker::PhantomData<fn() -> ($($g,)*)>,
        }

        // Expanded in the user's crate: `layout` may go unused, and it takes one argument per parameter.
        #[allow(dead_code, clippy::too_many_arguments)]
        impl<$($g: $crate::Numeric),*> $name<$($g),*> {
            /// The struct's layout for these parameters.
            // A struct without parameters leaves `layout` with no arguments to use.
            #[allow(unused_variables)]
            #[track_caller]
            pub fn layout($($p: $pt),*) -> $crate::StructLayout {
                let mut builder = $crate::StructLayout::__builder(
                    ::core::stringify!($name),
                    ::core::stringify!($kind) == "union",
                );
                $($crate::__cute_member!(@add builder, ::core::stringify!($f), $m);)*
                builder.build()
            }

            #[track_caller]
            pub fn size_of($($p: $pt),*) -> usize {
                Self::layout($($p),*).size_of()
            }

            #[track_caller]
            pub fn align_of($($p: $pt),*) -> usize {
                Self::layout($($p),*).align_of()
            }

            #[track_caller]
            pub fn size_in_bytes($($p: $pt),*) -> usize {
                Self::layout($($p),*).size_in_bytes()
            }

            /// Allocate the struct in shared memory and view it.
            #[track_caller]
            pub fn allocate($($p: $pt),*) -> Self {
                let layout = Self::layout($($p),*);
                Self::from_layout(&layout, &layout.allocate())
            }

            /// Allocate the struct aligned to `byte_alignment` too.
            #[track_caller]
            pub fn allocate_aligned(byte_alignment: usize, $($p: $pt),*) -> Self {
                let layout = Self::layout($($p),*);
                Self::from_layout(&layout, &layout.allocate_aligned(byte_alignment))
            }

            /// The struct at the byte pointer `base`.
            #[track_caller]
            pub fn at(base: &$crate::Ptr<u8>, $($p: $pt),*) -> Self {
                Self::from_layout(&Self::layout($($p),*), base)
            }

            /// The struct of layout `layout` at the byte pointer `base`.
            #[track_caller]
            pub fn from_layout(layout: &$crate::StructLayout, base: &$crate::Ptr<u8>) -> Self {
                $name {
                    $($f: $crate::__cute_member!(@view layout, base, ::core::stringify!($f), $m),)*
                    __base: base.clone(),
                    __marker: ::core::marker::PhantomData,
                }
            }

            /// The struct's byte pointer.
            pub fn base(&self) -> $crate::Ptr<u8> {
                self.__base.clone()
            }
        }
    };
    // Anything else: a member the grammar does not take.
    (@parse $kind:tt $meta:tt $vis:tt $name:tt $gen:tt $params:tt [$($acc:tt)*] $($rest:tt)+) => {
        ::core::compile_error!(::core::concat!(
            "cute_struct!: cannot read the member `", ::core::stringify!($($rest)+),
            "`; a member is `name: Type`, `name: MemRange<Type>[len]` or `name: Nested(args)`, optionally after #[align(n)]"
        ));
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __cute_member {
    (@ty (range [$t:ty] $n:tt $a:tt)) => { $crate::MemRange<$t> };
    (@ty (scalar [$t:ty] $a:tt)) => { $crate::ScalarMember<$t> };
    (@ty (nested [$s:ident] $args:tt $a:tt)) => { $s };
    (@align []) => { ::core::option::Option::None };
    (@align [$a:expr]) => { ::core::option::Option::Some($a) };
    (@add $b:ident, $fname:expr, (range [$t:ty] [$n:expr] $a:tt)) => {
        let _ = $b.mem_range::<$t>($fname, $n, $crate::__cute_member!(@align $a));
    };
    (@add $b:ident, $fname:expr, (scalar [$t:ty] $a:tt)) => {
        let _ = $b.scalar::<$t>($fname, $crate::__cute_member!(@align $a));
    };
    (@add $b:ident, $fname:expr, (nested [$s:ident] [$($arg:expr),*] $a:tt)) => {
        let _ = $b.nested($fname, &$s::layout($($arg),*), $crate::__cute_member!(@align $a));
    };
    (@view $l:ident, $base:ident, $fname:expr, (range [$t:ty] $n:tt $a:tt)) => {
        $l.member($fname).mem_range::<$t>($base)
    };
    (@view $l:ident, $base:ident, $fname:expr, (scalar [$t:ty] $a:tt)) => {
        $l.member($fname).scalar::<$t>($base)
    };
    (@view $l:ident, $base:ident, $fname:expr, (nested [$s:ident] $args:tt $a:tt)) => {
        $s::from_layout($l.member($fname).nested_layout(), &$l.member($fname).nested_base($base))
    };
}

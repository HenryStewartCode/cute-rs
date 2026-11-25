// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse element types (roadmap part m10-sparse-umma): CuTe DSL's
//! `cute.get_sparse_elem_type(num_logical, num_phys, elem_type)` and
//! `cute.make_sparse_elem(num_logical, elems)`, and the recasts that take
//! them as a type.
//!
//! A sparse element type stands for `num_logical` logical elements stored
//! in `num_phys` physical elements of `elem_type`, so one logical element
//! is `num_phys * width / num_logical` bits wide, a fraction of a bit for
//! the metadata of 2:4 sparse MMAs (8 logical elements per Uint8: one bit
//! per logical element). Its users are the SM100 sparse GEMM helpers
//! ([`make_sparse_a_dtype`](crate::utils::sm100::make_sparse_a_dtype),
//! [`make_sparse_e_dtype`](crate::utils::sm100::make_sparse_e_dtype)),
//! whose layouts count logical elements.
//!
//! cute-rs keeps the type out of [`DType`] (no IR value has it): a
//! [`SparseElemType`] is a build-time description. [`recast_layout_to_sparse`]
//! and [`recast_layout_from_sparse`] rescale layouts between it and a
//! physical type with the sub-byte recast machinery, a [`SparseTensor`]
//! views a physical tensor in logical elements ([`recast_tensor_sparse`],
//! [`recast_ptr_sparse`], [`SmemAllocator::allocate_sparse_tensor`]), and
//! loading or storing one logical element panics when the kernel is built
//! (a logical element has no address of its own). In the IR and the
//! simulator the data are the physical type's.
//!
//! cute-rs's sparse MMAs (`tcgen05.MmaF16BF16SparseOp`, `MmaFP8SparseOp`)
//! take compressed A as a plain tensor of the physical type, half of K;
//! the metadata layouts of [`utils::sm100`](crate::utils::sm100) count
//! logical metadata elements and are recast to Uint8 for the copies.

use std::marker::PhantomData;

use crate::ir::DType;
use crate::layout::{Layout, recast_layout};
use crate::memory::{SmemAllocator, SmemPartition};
use crate::tensor::{Ptr, Tensor};
use crate::value::Numeric;

/// A sparse element type (CuTe DSL's `SparseElemType`): `num_logical`
/// logical elements in `num_phys` physical elements of `elem_type`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SparseElemType {
    num_logical: usize,
    num_phys: usize,
    elem: DType,
}

/// CuTe DSL's `cute.get_sparse_elem_type(num_logical, num_phys,
/// elem_type)`. Panics unless `num_logical` is a positive multiple of
/// `num_phys` (more logical elements than physical ones).
#[track_caller]
pub fn get_sparse_elem_type(num_logical: usize, num_phys: usize, elem_type: DType) -> SparseElemType {
    assert!(
        num_phys > 0 && num_logical > num_phys && num_logical.is_multiple_of(num_phys),
        "cute: get_sparse_elem_type: {num_logical} logical elements in {num_phys} physical {} elements; a sparse \
         element type packs a whole number (more than one) of logical elements into each physical one",
        elem_type.cute_name()
    );
    assert!(
        !elem_type.is_word_packed() && elem_type.bits() > 0,
        "cute: get_sparse_elem_type: {} cannot hold sparse elements",
        elem_type.cute_name()
    );
    SparseElemType { num_logical, num_phys, elem: elem_type }
}

impl SparseElemType {
    pub fn num_logical(&self) -> usize {
        self.num_logical
    }

    pub fn num_phys(&self) -> usize {
        self.num_phys
    }

    /// The physical element type.
    pub fn elem_type(&self) -> DType {
        self.elem
    }

    /// Logical elements per physical element.
    pub fn sparsity(&self) -> usize {
        self.num_logical / self.num_phys
    }

    /// The width of one logical element in bits, as a reduced fraction
    /// `(numerator, denominator)`: `(1, 1)` for 8 logical elements per
    /// Uint8, `(1, 2)` for 16.
    pub fn width(&self) -> (usize, usize) {
        let (n, d) = (self.elem.bits() * self.num_phys, self.num_logical);
        let g = gcd(n, d);
        (n / g, d / g)
    }

    /// How many physical elements `count` logical elements take (rounded up).
    pub fn physical_count(&self, count: usize) -> usize {
        count.div_ceil(self.sparsity())
    }
}

impl std::fmt::Display for SparseElemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sparse_elem<{}, {}, {}>", self.num_logical, self.num_phys, self.elem.cute_name())
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A value of a sparse element type (CuTe DSL's `cute.make_sparse_elem`):
/// the physical values that hold `num_logical` logical elements.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseElem<T> {
    ty: SparseElemType,
    phys: Vec<T>,
}

/// CuTe DSL's `cute.make_sparse_elem(num_logical, elems)`: `elems` (the
/// physical values) packed into one value of
/// `get_sparse_elem_type(num_logical, elems.len(), T)`.
#[track_caller]
pub fn make_sparse_elem<T: Numeric>(num_logical: usize, elems: &[T]) -> SparseElem<T> {
    SparseElem { ty: get_sparse_elem_type(num_logical, elems.len(), T::DTYPE), phys: elems.to_vec() }
}

impl<T: Clone> SparseElem<T> {
    pub fn ty(&self) -> SparseElemType {
        self.ty
    }

    /// The physical values.
    pub fn physical(&self) -> &[T] {
        &self.phys
    }
}

/// A layout over elements of `from` (`from_bits` wide) restated over
/// logical elements of `to` (CuTe DSL's `recast_layout` with a sparse
/// element type as the new type): every stride scaled by the ratio of the
/// widths, as [`recast_layout`] does for sub-byte types.
#[track_caller]
pub fn recast_layout_to_sparse(layout: &Layout, from_bits: usize, to: SparseElemType) -> Layout {
    let (n, d) = to.width();
    recast_layout(n, from_bits * d, layout)
}

/// A layout over logical elements of `from` restated over physical
/// elements `to_bits` wide (the inverse of [`recast_layout_to_sparse`]).
#[track_caller]
pub fn recast_layout_from_sparse(layout: &Layout, from: SparseElemType, to_bits: usize) -> Layout {
    let (n, d) = from.width();
    recast_layout(to_bits * d, n, layout)
}

/// A tensor of sparse elements: a physical tensor of `T` seen through a
/// layout over logical elements of `ty` (CuTe DSL's tensor of a
/// `SparseElemType`). Its layout and slices are what the sparse layout
/// helpers produce; its data are reached through [`physical`](Self::physical).
/// [`load`](Self::load) and [`store`](Self::store) panic: a logical element
/// is not addressable on its own.
#[derive(Clone, Debug)]
pub struct SparseTensor<T> {
    ty: SparseElemType,
    ptr: Ptr<T>,
    layout: Layout,
    _t: PhantomData<T>,
}

impl<T: Numeric> SparseTensor<T> {
    /// A sparse tensor of type `ty` over `layout` (logical elements) at the
    /// physical pointer `ptr`, whose element type must be `ty`'s.
    #[track_caller]
    pub fn new(ty: SparseElemType, ptr: &Ptr<T>, layout: Layout) -> SparseTensor<T> {
        assert!(
            ty.elem == T::DTYPE,
            "cute: a {ty} tensor needs a pointer to {} elements, not {}",
            ty.elem.cute_name(),
            T::DTYPE.cute_name()
        );
        SparseTensor { ty, ptr: ptr.clone(), layout, _t: PhantomData }
    }

    pub fn ty(&self) -> SparseElemType {
        self.ty
    }

    /// The layout over logical elements.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The physical pointer.
    pub fn ptr(&self) -> Ptr<T> {
        self.ptr.clone()
    }

    /// The same data as a tensor of the physical type (the layout recast
    /// back with [`recast_layout_from_sparse`]).
    #[track_caller]
    pub fn physical(&self) -> Tensor<T> {
        let layout = recast_layout_from_sparse(&self.layout, self.ty, T::DTYPE.bits());
        crate::make_tensor(self.ptr.clone(), layout)
    }

    /// Always panics: a logical element of a sparse type has no address of
    /// its own, so scalar loads are refused when the kernel is built.
    #[track_caller]
    pub fn load(&self, _coord: impl Into<crate::layout::Coord>) -> ! {
        self.deref_panic("load")
    }

    /// Always panics, as [`load`](Self::load) does.
    #[track_caller]
    pub fn store(&self, _coord: impl Into<crate::layout::Coord>, _value: crate::value::Value<T>) -> ! {
        self.deref_panic("store")
    }

    #[track_caller]
    fn deref_panic(&self, what: &str) -> ! {
        panic!(
            "cute: {what} of one element of a {} tensor: {} logical elements share {} {} element(s), so a single \
             logical element cannot be read or written; recast to the physical type first (SparseTensor::physical)",
            self.ty,
            self.ty.num_logical,
            self.ty.num_phys,
            self.ty.elem.cute_name()
        )
    }
}

/// CuTe DSL's `recast_tensor(src, sparse_type)`: the tensor `src` of `U`
/// seen as logical elements of `ty`, its layout rescaled by the widths
/// ([`recast_layout_to_sparse`]) on a pointer to `ty`'s physical type `T`.
#[track_caller]
pub fn recast_tensor_sparse<T: Numeric, U: Numeric>(src: &Tensor<U>, ty: SparseElemType) -> SparseTensor<T> {
    let layout = recast_layout_to_sparse(src.layout(), U::DTYPE.bits(), ty);
    SparseTensor::new(ty, &src.ptr().cast::<T>(), layout)
}

/// CuTe DSL's `recast_ptr(ptr, dtype=sparse_type)`: a pointer to `ty`'s
/// physical type `T` at `ptr`'s address, for [`SparseTensor::new`].
#[track_caller]
pub fn recast_ptr_sparse<T: Numeric, U: Numeric>(ptr: &Ptr<U>, ty: SparseElemType) -> Ptr<T> {
    assert!(
        ty.elem == T::DTYPE,
        "cute: recast_ptr to {ty}: the pointer's element type is the physical {}, not {}",
        ty.elem.cute_name(),
        T::DTYPE.cute_name()
    );
    crate::recast_ptr::<T, U>(ptr, None)
}

impl SmemAllocator {
    /// [`allocate_tensor`](SmemAllocator::allocate_tensor) with a sparse
    /// element type (CuTe DSL's `allocate_tensor(sparse_type, layout, ...)`):
    /// shared memory for `layout` (logical elements of `ty`) and the sparse
    /// tensor over it.
    #[track_caller]
    pub fn allocate_sparse_tensor<T: Numeric>(
        &self,
        ty: SparseElemType,
        layout: Layout,
        byte_alignment: usize,
        partition: SmemPartition,
    ) -> SparseTensor<T> {
        let phys = recast_layout_from_sparse(&layout, ty, T::DTYPE.bits());
        let t = self.allocate_tensor::<T>(phys, byte_alignment, None, partition);
        SparseTensor::new(ty, &t.ptr(), layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_and_recasts() {
        let e = get_sparse_elem_type(8, 1, DType::U8);
        assert_eq!((e.width(), e.sparsity()), ((1, 1), 8));
        assert_eq!(get_sparse_elem_type(16, 1, DType::U8).width(), (1, 2));
        assert_eq!(get_sparse_elem_type(2, 1, DType::F16).width(), (8, 1));
        // 128 rows of 128 one-bit logical elements are 128 rows of 16 bytes.
        let logical = Layout::new((128, 128), (128, 1));
        let bytes = recast_layout_from_sparse(&logical, e, 8);
        assert_eq!(bytes, Layout::new((128, 16), (16, 1)));
        assert_eq!(recast_layout_to_sparse(&bytes, 8, e), logical);
    }
}

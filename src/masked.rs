//! Masked and gathered accesses: [`Tensor::load_ssa_masked`] and
//! [`Tensor::store_ssa_masked`] (CuTe DSL's `Tensor.load(mask=,
//! pass_thru=)` and `Tensor.store(data, mask=)`), [`gather`] and
//! [`scatter`] along a mode (`cutlass.cute.tensor.gather`/`scatter`), and
//! the `cutlass.Pointer` methods on [`Ptr`].
//!
//! A mask flag guards one access: an `Op::Load` or `Op::Store` with a
//! predicate, which a thread whose flag is false skips (it touches no
//! memory, so no bounds, alignment or race check applies to it). A masked
//! load's masked-off results are its pass-through values, or undefined
//! without them: the simulator reports a read of an undefined value by
//! anything but a pure operation (which passes the mark on) or a select
//! that picks another operand. MLIR gets an `scf.if` around the access,
//! yielding the pass-through values or a frozen `ub.poison`, which NVPTX
//! turns into a branch over each access.

use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::panic::Location;

use crate::arch::{LoadOptions, StoreOptions};
use crate::builder::{self, LocationScope};
use crate::ir::{DType, MemAttrs, MemoryScope, Op, ValueId};
use crate::layout::{Coord, IndexExpr, Tree};
use crate::tensor::{Engine, Ptr, Tensor, index_operand};
use crate::tensor_ssa::{SELECT_TREE_MAX, TensorSsa, select_tree};
use crate::value::{Integer, IntoValue, Numeric, Value, select};
use crate::vector::Vector;

/// One masked load of `n` contiguous elements at `index` of `buffer`.
#[track_caller]
fn masked_load_ids<T: Numeric>(
    buffer: crate::ir::BufferId,
    index: ValueId,
    n: usize,
    attrs: MemAttrs,
    pred: ValueId,
    pass_thru: Vec<ValueId>,
) -> Vec<ValueId> {
    let loc = Location::caller();
    builder::with(|b| {
        let results: Vec<ValueId> = (0..n).map(|_| b.new_value(T::DTYPE)).collect();
        b.push(Op::Load { results: results.clone(), buffer, index, attrs, pred: Some(pred), pass_thru }, loc);
        results
    })
}

/// One masked store of `values` at `index` of `buffer`.
#[track_caller]
fn masked_store_ids(buffer: crate::ir::BufferId, index: ValueId, values: &[ValueId], attrs: MemAttrs, pred: ValueId) {
    let loc = Location::caller();
    builder::with(|b| {
        b.push(Op::Store { values: values.to_vec(), buffer, index, attrs, pred: Some(pred) }, loc);
    });
}

#[track_caller]
fn refuse_sub_byte<T: Numeric>(what: &str) {
    assert!(
        !T::DTYPE.is_word_packed(),
        "{what}: {} elements are packed into 32-bit words, and a mask guards whole elements; masked and gathered \
         accesses take types of at least a byte",
        T::DTYPE.cute_name()
    );
}

/// Whether a flag is the constant `true`.
fn always(flag: Value<bool>) -> bool {
    flag.as_constant() == Some(true)
}

/// Whether a flag is the constant `false` (a store under it is dropped).
fn never(flag: Value<bool>) -> bool {
    flag.as_constant() == Some(false)
}

impl<T: Numeric> Tensor<T> {
    /// The whole tensor as a [`TensorSsa`], reading only the elements whose
    /// `mask` flag is true (CuTe DSL's `Tensor.load(mask=, pass_thru=)`):
    /// a masked-off element is its `pass_thru` element, or undefined without
    /// one (the simulator reports any use of it other than a select that
    /// picks something else). The mask and `pass_thru` have the tensor's
    /// size (colexicographic order, as [`load_ssa`](Tensor::load_ssa)).
    /// In memory each element is one access guarded by its flag (CuTe DSL's
    /// masked `memref_load_vec`, which NVPTX scalarises the same way); a
    /// register fragment's elements are selected.
    ///
    /// ```
    /// use cute::prelude::*;
    /// Kernel::build("masked_tail", || {
    ///     let x = param_tensor::<f32>("x", Layout::col_major(6));
    ///     let out = param_tensor::<f32>("out", Layout::col_major(8));
    ///     let t = thread_idx().x;
    ///     // Eight lanes over a six-element buffer: the last two read nothing.
    ///     let tile = make_tensor(&x.ptr(), Layout::col_major(8));
    ///     let mask = TensorSsa::new(8, (0..8).map(|i| Int32::constant(i).lt(6)).collect());
    ///     let v = tile.load_ssa_masked(&mask, Some(&full::<f32>(8, 0.0)));
    ///     if_(t.eq(0), || out.store_ssa(&v));
    /// });
    /// ```
    #[track_caller]
    pub fn load_ssa_masked(&self, mask: &TensorSsa<bool>, pass_thru: Option<&TensorSsa<T>>) -> TensorSsa<T> {
        let _at = LocationScope::new();
        let what = "Tensor::load_ssa_masked";
        if let Engine::ArithTuple(_) = self.engine {
            panic!("{what}: {self:?} is a coordinate tensor, which holds coordinates, not values");
        }
        let shape = crate::tensor_ssa::SsaShape::ssa_shape(self);
        let n = shape.size();
        check_size(n, mask.size(), "mask", what);
        if let Some(p) = pass_thru {
            check_size(n, p.size(), "pass_thru", what);
        }
        if self.is_register() {
            let values = (0..n)
                .map(|i| {
                    let v = self.load(i);
                    match pass_thru {
                        Some(p) if !always(mask.values()[i]) => select(mask.values()[i], v, p.values()[i]),
                        _ => v,
                    }
                })
                .collect();
            return TensorSsa::new(shape, values);
        }
        refuse_sub_byte::<T>(what);
        // A tensor in another CTA's shared memory keeps its peer.
        let Engine::Memory { buffer, peer, .. } = self.engine else { unreachable!() };
        let attrs = MemAttrs { peer, ..MemAttrs::default() };
        let values = (0..n)
            .map(|i| {
                let flag = mask.values()[i];
                let off = self.element_offset(&Coord::from(i));
                if always(flag) {
                    return self.load(i);
                }
                let pt = pass_thru.map(|p| vec![p.values()[i].checked()]).unwrap_or_default();
                let ids = masked_load_ids::<T>(buffer, index_operand(off), 1, attrs, flag.checked(), pt);
                Value::from_id(ids[0])
            })
            .collect();
        TensorSsa::new(shape, values)
    }

    /// Write the elements of `data` whose `mask` flag is true (CuTe DSL's
    /// `Tensor.store(data, mask=)`); the others are left alone and their
    /// memory is not touched. `data` and the mask have the tensor's size;
    /// `data` converts as [`store_ssa`](Tensor::store_ssa)'s does. In memory
    /// each element is one store guarded by its flag; a register fragment
    /// keeps the old element where the flag is false (a select).
    #[track_caller]
    pub fn store_ssa_masked<U: Numeric>(&self, data: &TensorSsa<U>, mask: &TensorSsa<bool>) {
        let _at = LocationScope::new();
        let what = "Tensor::store_ssa_masked";
        if let Engine::ArithTuple(_) = self.engine {
            panic!("{what}: {self:?} is a coordinate tensor, which is read-only");
        }
        let shape = crate::tensor_ssa::SsaShape::ssa_shape(self);
        let n = shape.size();
        check_size(n, data.size(), "data", what);
        check_size(n, mask.size(), "mask", what);
        let values = crate::tensor_ssa::cvt_to_dest::<U, T>(data, what);
        if self.is_register() {
            for (i, v) in values.into_iter().enumerate() {
                let flag = mask.values()[i];
                let v = if always(flag) { v } else { select(flag, v, self.load(i)) };
                self.store(i, v);
            }
            return;
        }
        refuse_sub_byte::<T>(what);
        let Engine::Memory { buffer, peer, .. } = self.engine else { unreachable!() };
        let attrs = MemAttrs { peer, ..MemAttrs::default() };
        for (i, v) in values.into_iter().enumerate() {
            let flag = mask.values()[i];
            if always(flag) {
                self.store(i, v);
                continue;
            }
            if never(flag) {
                continue;
            }
            let off = self.element_offset(&Coord::from(i));
            masked_store_ids(buffer, index_operand(off), &[v.checked()], attrs, flag.checked());
        }
    }
}

#[track_caller]
fn check_size(n: usize, got: usize, name: &str, what: &str) {
    assert_eq!(got, n, "{what}: the tensor has {n} elements, but the {name} has {got}");
}

/// The checks of CuTe DSL's `_check_can_gather_scatter` on a tensor and an
/// index tensor.
#[track_caller]
fn check_gather_scatter<T: Numeric>(tensor: &Tensor<T>, mode: usize, index: &crate::IntTuple, what: &str) {
    let shape = crate::tensor_ssa::SsaShape::ssa_shape(tensor);
    let rank = tensor.rank();
    assert!(mode < rank, "{what}: mode {mode} is out of range for the rank-{rank} tensor {tensor:?}");
    assert_eq!(index.rank(), rank, "{what}: the index shape {index:?} and the tensor {tensor:?} have different ranks");
    assert!(
        tensor.depth() <= 1 && index.depth() <= 1,
        "{what}: the tensor {tensor:?} and the index shape {index:?} must be flat (depth 1)"
    );
    assert!(
        tensor.swizzle().is_none() && tensor.inner_layout().is_none(),
        "{what}: {tensor:?} has a composed layout; gather and scatter take plain layouts (CuTe DSL refuses \
         ComposedLayout)"
    );
    assert!(!T::DTYPE.is_word_packed(), "{what}: sub-byte element types ({}) are not supported", T::DTYPE.cute_name());
    for m in (0..rank).filter(|&m| m != mode) {
        let (i, t) = (mode_size(index, m), mode_size(&shape, m));
        assert!(i <= t, "{what}: mode {m} of the index shape {index:?} ({i}) is larger than the tensor's ({t})");
    }
}

fn mode_size(shape: &crate::IntTuple, m: usize) -> usize {
    match shape {
        Tree::Leaf(e) => *e,
        Tree::Branch(ms) => ms[m].size(),
    }
}

/// The index tensor's elements as Int32 positions (as IndexExprs).
#[track_caller]
fn positions<I: Integer>(index: &TensorSsa<I>) -> Vec<IndexExpr> {
    index
        .values()
        .iter()
        .map(|&v| match I::DTYPE {
            DType::I64 | DType::U64 => IndexExpr::from(v.to::<i64>()),
            _ => IndexExpr::from(v.to::<i32>()),
        })
        .collect()
}

/// The coordinate of `input` that element `k` of an index tensor of
/// `shape` addresses along `mode` at `pos`.
fn gather_coord(shape: &crate::IntTuple, k: usize, mode: usize, pos: IndexExpr) -> Coord {
    let leaves = shape.flat_leaves();
    let mut rest = k;
    let coords: Vec<IndexExpr> = leaves
        .iter()
        .enumerate()
        .map(|(m, &e)| {
            let c = rest % e;
            rest /= e;
            if m == mode { pos.clone() } else { IndexExpr::Static(c) }
        })
        .collect();
    if coords.len() == 1 {
        Tree::Leaf(coords[0].clone())
    } else {
        Tree::Branch(coords.into_iter().map(Tree::Leaf).collect())
    }
}

/// CuTe DSL's `cutlass.cute.tensor.gather(input, mode, index)`: a
/// [`TensorSsa`] shaped like `index` (an integer [`TensorSsa`] of `input`'s
/// rank) whose element at `(a, b, c)` is the element of `input` at
/// `(index(a, b, c), b, c)`
/// for mode 0, and likewise along the other modes. `input` is a flat
/// tensor with a static shape and no swizzle, of a type of at least a
/// byte; the index's other modes must be no larger than `input`'s. Each
/// element is one load at a runtime offset; a register fragment is read
/// through a tree of selects over the mode, of at most
/// [`SELECT_TREE_MAX`] elements (a larger one belongs in a
/// [`make_rmem_tensor`](crate::make_rmem_tensor) array). An index outside
/// the mode is undefined behaviour, as in CuTe DSL; the simulator reports
/// one that reaches outside the buffer.
///
/// ```
/// use cute::prelude::*;
/// use cute::tensor::gather;
/// Kernel::build("gather_rows", || {
///     let x = param_tensor::<f32>("x", Layout::col_major((8, 4)));
///     let rows = param_tensor::<i32>("rows", Layout::col_major((2, 4)));
///     let out = param_tensor::<f32>("out", Layout::col_major((2, 4)));
///     // out[i, j] = x[rows[i, j], j]
///     out.store_ssa(&gather(&x, 0, &rows.load_ssa()));
/// });
/// ```
#[track_caller]
pub fn gather<T: Numeric, I: Integer>(input: &Tensor<T>, mode: usize, index: &TensorSsa<I>) -> TensorSsa<T> {
    let _at = LocationScope::new();
    let what = "gather";
    check_gather_scatter(input, mode, index.shape(), what);
    let pos = positions(index);
    let shape = index.shape().clone();
    let values = (0..index.size())
        .map(|k| {
            let c = gather_coord(&shape, k, mode, pos[k].clone());
            if input.is_register() {
                let extent = mode_size(&crate::tensor_ssa::SsaShape::ssa_shape(input), mode);
                check_register_extent(extent, what);
                let fiber: Vec<Value<T>> =
                    (0..extent).map(|e| input.load(gather_coord(&shape, k, mode, IndexExpr::Static(e)))).collect();
                return select_tree(&fiber, IntoValue::<i32>::into_value(pos[k].clone()));
            }
            input.load(c)
        })
        .collect();
    TensorSsa::new(shape, values)
}

#[track_caller]
fn check_register_extent(extent: usize, what: &str) {
    assert!(
        extent <= SELECT_TREE_MAX,
        "{what}: the register fragment's mode has {extent} elements, more than the {SELECT_TREE_MAX} a select tree \
         indexes at run time; put the values in a make_rmem_tensor array, which takes runtime offsets"
    );
}

/// CuTe DSL's `cutlass.cute.tensor.scatter(output, mode, index, data)`:
/// writes `data(a, b, c)` to `output` at `(index(a, b, c), b, c)` for mode 0, and
/// likewise along the other modes, with the checks of [`gather`]; `data`
/// is shaped like `index`. The stores go in colexicographic order of the
/// index, so of duplicate indices the last wins (CuTe DSL leaves
/// duplicates undefined). A register fragment is updated through selects
/// over the mode (at most [`SELECT_TREE_MAX`] elements).
#[track_caller]
pub fn scatter<T: Numeric, I: Integer>(output: &Tensor<T>, mode: usize, index: &TensorSsa<I>, data: &TensorSsa<T>) {
    let _at = LocationScope::new();
    let what = "scatter";
    check_gather_scatter(output, mode, index.shape(), what);
    assert_eq!(data.shape(), index.shape(), "{what}: the data {data:?} is not shaped like the index {index:?}");
    let pos = positions(index);
    let shape = index.shape().clone();
    for (k, (pos_k, &v)) in pos.iter().zip(data.values()).enumerate() {
        if output.is_register() {
            let extent = mode_size(&crate::tensor_ssa::SsaShape::ssa_shape(output), mode);
            check_register_extent(extent, what);
            let p: Value<i32> = IntoValue::<i32>::into_value(pos_k.clone());
            for e in 0..extent {
                let c = gather_coord(&shape, k, mode, IndexExpr::Static(e));
                let old = output.load(c.clone());
                output.store(c, select(p.eq(e as i32), v, old));
            }
            continue;
        }
        output.store(gather_coord(&shape, k, mode, pos_k.clone()), v);
    }
}

impl<T: Numeric> Ptr<T> {
    /// The element `i` places on (CuTe DSL's `ptr[i]`).
    #[track_caller]
    pub fn load_at(&self, i: impl Into<IndexExpr>) -> Value<T> {
        self.offset(i).load()
    }

    /// Write the element `i` places on (CuTe DSL's `ptr[i] = v`).
    #[track_caller]
    pub fn store_at(&self, i: impl Into<IndexExpr>, v: impl IntoValue<T>) {
        self.offset(i).store(v)
    }

    /// The element type (CuTe DSL's `Pointer.dtype`).
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// The element with CuTe DSL's memory semantics and cache hints
    /// (`Pointer.load(is_volatile=, is_invariant=, ordering=, syncscope=)`
    /// and `nvvm_load_ext(cache_modifier=, evict=)`): [`load`](crate::load)
    /// with `opts` (`is_volatile` is `MemoryOrder::Volatile`, `is_invariant`
    /// `MemoryOrder::Constant`, `ld.global.nc`).
    #[track_caller]
    pub fn load_with(&self, opts: LoadOptions) -> Value<T> {
        crate::load(self, opts)
    }

    /// [`store`](crate::store) with `opts` (CuTe DSL's `Pointer.store(v,
    /// is_volatile=, ordering=, syncscope=)` and `nvvm_store_ext`).
    #[track_caller]
    pub fn store_with(&self, v: impl IntoValue<T>, opts: StoreOptions) {
        crate::store(self, v, opts)
    }

    /// `count` contiguous elements as a [`Vector`] (CuTe DSL's
    /// `Pointer.load(alignment, count=N)`). `alignment` (bytes) is trusted
    /// as CuTe DSL's `llvm.load` trusts it, and the simulator checks every
    /// access's alignment at run time: the elements move in vectors of
    /// `alignment` bytes (at most 16, and at most the buffer's declared
    /// alignment).
    #[track_caller]
    pub fn load_vec(&self, count: usize, alignment: usize) -> Vector<T> {
        self.load_vec_with(count, alignment, LoadOptions::default())
    }

    /// [`load_vec`](Ptr::load_vec) with memory semantics and cache hints
    /// (see [`load_with`](Ptr::load_with)); the combinations
    /// [`load_vector`](crate::load_vector) refuses are refused.
    #[track_caller]
    pub fn load_vec_with(&self, count: usize, alignment: usize, opts: LoadOptions) -> Vector<T> {
        let _at = LocationScope::new();
        let attrs = load_attrs(opts);
        let chunk = self.chunk(count, alignment, "Ptr::load_vec");
        let mut values = Vec::with_capacity(count);
        for start in (0..count).step_by(chunk) {
            let (buffer, off) = self.offset(start).parts();
            let ids = crate::tensor::load_vector_with::<T>(buffer, index_operand(off), chunk, attrs);
            values.extend(ids.into_iter().map(Value::from_id));
        }
        Vector::new(count, values)
    }

    /// Write a 1-D [`Vector`]'s elements contiguously (CuTe DSL's
    /// `Pointer.store(vec, alignment=)`), in vectors of `alignment` bytes
    /// as [`load_vec`](Ptr::load_vec) reads them.
    #[track_caller]
    pub fn store_vec(&self, v: &Vector<T>, alignment: usize) {
        self.store_vec_with(v, alignment, StoreOptions::default())
    }

    /// [`store_vec`](Ptr::store_vec) with memory semantics and cache hints.
    #[track_caller]
    pub fn store_vec_with(&self, v: &Vector<T>, alignment: usize, opts: StoreOptions) {
        let _at = LocationScope::new();
        let attrs = store_attrs(opts);
        let count = v.size();
        let chunk = self.chunk(count, alignment, "Ptr::store_vec");
        let ids: Vec<ValueId> = v.values().iter().map(|x| x.checked()).collect();
        for start in (0..count).step_by(chunk) {
            let (buffer, off) = self.offset(start).parts();
            crate::tensor::store_vector_with(buffer, index_operand(off), &ids[start..start + chunk], attrs);
        }
    }

    /// The elements of a vector access: the largest power of two of
    /// elements that `alignment`, 16 bytes, the buffer's alignment and
    /// `count` allow. An alignment the offset provably contradicts (a
    /// static misalignment) panics; one that cannot be checked is trusted.
    #[track_caller]
    fn chunk(&self, count: usize, alignment: usize, what: &str) -> usize {
        refuse_sub_byte::<T>(what);
        assert!(count > 0, "{what}: a vector has at least one element");
        assert!(alignment.is_power_of_two(), "{what}: the alignment {alignment} is not a power of two");
        let (buffer, off) = self.parts();
        let declared = builder::with(|b| b.buffer(buffer).align);
        let bytes = T::DTYPE.bytes();
        let mut chunk = (alignment.min(16).min(declared) / bytes).max(1);
        while chunk > 1 && !count.is_multiple_of(chunk) {
            chunk /= 2;
        }
        // The alignment is trusted, but not against a proof that it is
        // wrong: a static offset (plus a runtime part known to be a
        // multiple of the vector) that leaves the vector misaligned.
        let vbytes = chunk * bytes;
        if chunk > 1 {
            let fixed = static_part(&off);
            let rest = IndexExpr::difference(off.clone(), IndexExpr::Static(fixed));
            assert!(
                !((rest * bytes).is_multiple_of(vbytes) && !(fixed * bytes).is_multiple_of(vbytes)),
                "{what}: the alignment {alignment} is wrong for {self:?}: its offset of {off:?} elements puts the \
                 {vbytes}-byte vector accesses {} bytes past a {vbytes}-byte boundary",
                (fixed * bytes) % vbytes
            );
        }
        chunk
    }

    /// CuTe DSL's `Pointer.masked_load(mask, pass_thru, alignment)`: the
    /// elements of a 1-D Boolean vector's true lanes, read; the others
    /// `pass_thru`'s, or undefined (see [`Tensor::load_ssa_masked`]).
    /// Lanes sharing one flag value (a broadcast flag) move as one vector
    /// access of up to `alignment` bytes; constant-true lanes as plain ones;
    /// the rest one guarded access each.
    #[track_caller]
    pub fn masked_load(&self, mask: &Vector<bool>, pass_thru: Option<&Vector<T>>, alignment: usize) -> Vector<T> {
        let _at = LocationScope::new();
        let count = mask.size();
        if let Some(p) = pass_thru {
            assert_eq!(p.size(), count, "Ptr::masked_load: the mask has {count} lanes, the pass_thru {}", p.size());
        }
        let chunk = self.chunk(count, alignment, "Ptr::masked_load");
        let flags = mask.values();
        let mut values = Vec::with_capacity(count);
        for start in (0..count).step_by(chunk) {
            let group = &flags[start..start + chunk];
            let pt = |s: usize, n: usize| -> Vec<ValueId> {
                pass_thru.map(|p| p.values()[s..s + n].iter().map(|v| v.checked()).collect()).unwrap_or_default()
            };
            let (buffer, off) = self.offset(start).parts();
            if group.iter().all(|&f| always(f)) {
                let ids = crate::tensor::load_vector_with::<T>(buffer, index_operand(off), chunk, MemAttrs::default());
                values.extend(ids.into_iter().map(Value::from_id));
            } else if group.iter().all(|f| f.id() == group[0].id()) {
                let ids = masked_load_ids::<T>(
                    buffer,
                    index_operand(off),
                    chunk,
                    MemAttrs::default(),
                    group[0].checked(),
                    pt(start, chunk),
                );
                values.extend(ids.into_iter().map(Value::from_id));
            } else {
                for (k, &f) in group.iter().enumerate() {
                    let at = self.offset(start + k);
                    if always(f) {
                        values.push(at.load());
                        continue;
                    }
                    let (buffer, off) = at.parts();
                    let ids = masked_load_ids::<T>(
                        buffer,
                        index_operand(off),
                        1,
                        MemAttrs::default(),
                        f.checked(),
                        pt(start + k, 1),
                    );
                    values.push(Value::from_id(ids[0]));
                }
            }
        }
        Vector::new(count, values)
    }

    /// CuTe DSL's `Pointer.masked_store(value, mask, alignment)`: the
    /// elements of a 1-D vector whose `mask` lane is true, written; the
    /// others' memory is not touched. Grouped as
    /// [`masked_load`](Ptr::masked_load) groups.
    #[track_caller]
    pub fn masked_store(&self, value: &Vector<T>, mask: &Vector<bool>, alignment: usize) {
        let _at = LocationScope::new();
        let count = value.size();
        assert_eq!(mask.size(), count, "Ptr::masked_store: {count} values but {} mask lanes", mask.size());
        let chunk = self.chunk(count, alignment, "Ptr::masked_store");
        let flags = mask.values();
        let ids: Vec<ValueId> = value.values().iter().map(|v| v.checked()).collect();
        for start in (0..count).step_by(chunk) {
            let group = &flags[start..start + chunk];
            let (buffer, off) = self.offset(start).parts();
            let part = &ids[start..start + chunk];
            if group.iter().all(|&f| always(f)) {
                crate::tensor::store_vector_with(buffer, index_operand(off), part, MemAttrs::default());
            } else if group.iter().all(|f| f.id() == group[0].id()) {
                masked_store_ids(buffer, index_operand(off), part, MemAttrs::default(), group[0].checked());
            } else {
                for (k, &f) in group.iter().enumerate() {
                    let (buffer, off) = self.offset(start + k).parts();
                    if always(f) {
                        crate::tensor::store_vector_with(
                            buffer,
                            index_operand(off),
                            &part[k..k + 1],
                            MemAttrs::default(),
                        );
                    } else if !never(f) {
                        masked_store_ids(buffer, index_operand(off), &part[k..k + 1], MemAttrs::default(), f.checked());
                    }
                }
            }
        }
    }

    /// CuTe DSL's `Pointer.masked_gather(offsets, mask, pass_thru,
    /// alignment)`: lane `i` reads `ptr[offsets[i]]` if its `mask` lane is
    /// true, and is `pass_thru[i]` (or undefined) otherwise. Each lane is
    /// one guarded scalar load at a runtime offset (non-negative, element
    /// counts); `alignment` must be at least the element's.
    #[track_caller]
    pub fn masked_gather(
        &self,
        offsets: &Vector<i32>,
        mask: &Vector<bool>,
        pass_thru: Option<&Vector<T>>,
        alignment: usize,
    ) -> Vector<T> {
        let _at = LocationScope::new();
        let what = "Ptr::masked_gather";
        refuse_sub_byte::<T>(what);
        let count = offsets.size();
        assert_eq!(mask.size(), count, "{what}: {count} offsets but {} mask lanes", mask.size());
        if let Some(p) = pass_thru {
            assert_eq!(p.size(), count, "{what}: {count} offsets but {} pass_thru lanes", p.size());
        }
        assert!(
            alignment >= T::DTYPE.bytes(),
            "{what}: the alignment {alignment} is below the {} bytes of an element",
            T::DTYPE.bytes()
        );
        let values = (0..count)
            .map(|i| {
                let at = self.offset(IndexExpr::from(offsets.values()[i]));
                let f = mask.values()[i];
                if always(f) {
                    return at.load();
                }
                let (buffer, off) = at.parts();
                let pt = pass_thru.map(|p| vec![p.values()[i].checked()]).unwrap_or_default();
                Value::from_id(
                    masked_load_ids::<T>(buffer, index_operand(off), 1, MemAttrs::default(), f.checked(), pt)[0],
                )
            })
            .collect();
        Vector::new(count, values)
    }
}

fn load_attrs(opts: LoadOptions) -> MemAttrs {
    MemAttrs {
        order: opts.sem,
        scope: opts.scope.unwrap_or(MemoryScope::Sys),
        l1_eviction: opts.level1_eviction_priority,
        cop: opts.cop,
        l2_prefetch: opts.level_prefetch_size,
        non_coherent: false,
        cache_policy: opts.cache_policy.map(|p| p.checked()),
        peer: None,
    }
}

fn store_attrs(opts: StoreOptions) -> MemAttrs {
    MemAttrs {
        order: opts.sem,
        scope: opts.scope.unwrap_or(MemoryScope::Sys),
        l1_eviction: opts.level1_eviction_priority,
        cop: opts.cop,
        l2_prefetch: None,
        non_coherent: false,
        cache_policy: opts.cache_policy.map(|p| p.checked()),
        peer: None,
    }
}

/// The static part of an offset: its constant term.
fn static_part(e: &IndexExpr) -> usize {
    match e {
        IndexExpr::Static(v) => *v,
        IndexExpr::Add(a, b) => static_part(a) + static_part(b),
        _ => 0,
    }
}

/// `ptr + n`: `n` elements further on (CuTe DSL's `Pointer.__add__`).
impl<T: Numeric, X: Into<IndexExpr>> Add<X> for Ptr<T> {
    type Output = Ptr<T>;
    fn add(self, n: X) -> Ptr<T> {
        self.offset(n)
    }
}

impl<T: Numeric, X: Into<IndexExpr>> Add<X> for &Ptr<T> {
    type Output = Ptr<T>;
    fn add(self, n: X) -> Ptr<T> {
        self.offset(n)
    }
}

impl<T: Numeric, X: Into<IndexExpr>> AddAssign<X> for Ptr<T> {
    fn add_assign(&mut self, n: X) {
        *self = self.offset(n);
    }
}

/// `ptr - n`: `n` elements back. Offsets are non-negative until signed
/// index arithmetic (m5), so `n` must be static and at most the static
/// part of the pointer's offset, or the build panics.
impl<T: Numeric, X: Into<IndexExpr>> Sub<X> for Ptr<T> {
    type Output = Ptr<T>;
    #[track_caller]
    fn sub(self, n: X) -> Ptr<T> {
        back(&self, n.into())
    }
}

impl<T: Numeric, X: Into<IndexExpr>> Sub<X> for &Ptr<T> {
    type Output = Ptr<T>;
    #[track_caller]
    fn sub(self, n: X) -> Ptr<T> {
        back(self, n.into())
    }
}

impl<T: Numeric, X: Into<IndexExpr>> SubAssign<X> for Ptr<T> {
    #[track_caller]
    fn sub_assign(&mut self, n: X) {
        *self = back(self, n.into());
    }
}

#[track_caller]
fn back<T: Numeric>(p: &Ptr<T>, n: IndexExpr) -> Ptr<T> {
    let (_, off) = p.parts_unswizzled();
    let Some(k) = n.as_static() else {
        panic!("Ptr - n: n = {n:?} is a runtime value; a pointer moves back only by a static count (until m5)")
    };
    let have = static_part(&off);
    assert!(
        k <= have,
        "Ptr - {k}: the pointer's offset {off:?} has a static part of only {have} elements, and offsets cannot go \
         negative (until m5's signed index arithmetic)"
    );
    p.with_offset(IndexExpr::difference(off, IndexExpr::Static(k)))
}

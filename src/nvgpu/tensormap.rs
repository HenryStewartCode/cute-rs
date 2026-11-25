// SPDX-License-Identifier: MIT OR Apache-2.0

//! TMA descriptors that live in memory and that kernels modify, for grouped and pointer-array GEMMs whose
//! groups each need their own tensor map: CuTe DSL's
//! `cpasync.copy_tensormap`, `update_tma_descriptor`,
//! `fence_tma_desc_acquire`, `fence_tma_desc_release` and
//! `cp_fence_tma_desc_release`, and the single-field
//! [`tensormap_replace`] they are built on.
//!
//! A TMA atom's descriptor is a grid-constant kernel parameter. A kernel
//! copies it to a 128-byte, 128-byte-aligned workspace in global or shared
//! memory ([`copy_tensormap`]), rewrites the global address, extents and
//! strides from another tensor ([`update_tma_descriptor`],
//! `tensormap.replace`, which needs sm_90a, sm_100a, sm_110a or sm_120a),
//! and issues TMA copies through it ([`CopyArgs::tma_desc_ptr`]). TMA reads
//! descriptors through the *tensormap proxy*, so the generic writes that
//! made one must be released to that proxy (by the writer:
//! [`fence_tma_desc_release`], or [`cp_fence_tma_desc_release`], which also
//! copies a descriptor edited in shared memory to global memory) and
//! acquired by the thread that issues the copy ([`fence_tma_desc_acquire`]
//! on the descriptor's address), in that order. The simulator checks this,
//! and that the bytes a TMA instruction reads hold a descriptor.
//!
//! The PTX of every operation here is inline assembly: LLVM 21 has no NVVM
//! operation for `tensormap.replace` (LLVM 23 has `nvvm.tensormap.replace`)
//! and neither has one for `tensormap.cp_fenceproxy`. That instruction is
//! warp-collective (`.sync.aligned`), and LLVM does not treat inline
//! assembly as convergent: call [`cp_fence_tma_desc_release`] from a whole,
//! converged warp, outside code whose branches LLVM could duplicate it into
//! (the caveat the README gives for inline PTX).
//!
//! [`CopyArgs::tma_desc_ptr`]: crate::CopyArgs::tma_desc_ptr

use std::fmt;
use std::panic::Location;

use crate::atom::copy::CopyAtom;
use crate::builder;
use crate::ir::{DType, Op, ValueId};
use crate::layout::{IndexExpr, StrideLeaf};
use crate::tensor::{AddressSpace, Ptr, Tensor, TensorInner, recast_ptr};
use crate::value::{AnyValue, IntoValue, Numeric};

/// A field of a tiled tensor map that `tensormap.replace` rewrites (the PTX
/// ISA's `.field1`, `.field2` and `.field4` with a value operand): the
/// global address (`.b64`), the extent of TMA dimension `ord` (`.b32`,
/// elements) and the stride of TMA dimension `ord + 1` (`.b64`, bytes, a
/// multiple of 16: dimension 0's stride is the element, so ordinal 0 is
/// dimension 1's, as C++ CuTe's `tma_descriptor_replace_dims_strides_in_shared_mem`
/// numbers them). Ordinals are 0 to 4 for extents and 0 to 3 for strides.
/// The other fields (box, element strides, rank, element type, swizzle,
/// ...) are not modelled yet.
///
/// The stride's unit: the PTX ISA does not state it. The assembler of CUDA
/// 12.5 and later takes the stride in bytes, which is what cute-rs passes;
/// earlier ones took it in 16-byte units (C++ CuTe's
/// `tma_descriptor_replace_dims_strides_in_shared_mem` shifts it right by 4
/// for them). Compile kernels that replace strides with CUDA 12.5 or later,
/// and run them on a driver of that release or later (whose JIT assembles
/// the PTX).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensormapField {
    GlobalAddress,
    GlobalDim(u8),
    GlobalStride(u8),
}

impl TensormapField {
    /// The PTX field qualifier: `global_address`, `global_dim` or
    /// `global_stride`.
    pub fn ptx_name(self) -> &'static str {
        match self {
            TensormapField::GlobalAddress => "global_address",
            TensormapField::GlobalDim(_) => "global_dim",
            TensormapField::GlobalStride(_) => "global_stride",
        }
    }

    /// The ordinal operand, for the per-dimension fields.
    pub fn ord(self) -> Option<u8> {
        match self {
            TensormapField::GlobalAddress => None,
            TensormapField::GlobalDim(o) | TensormapField::GlobalStride(o) => Some(o),
        }
    }

    /// The type of the new value: `i64` (`.b64`) for the address and the
    /// strides, `i32` (`.b32`) for the extents.
    pub fn value_type(self) -> DType {
        match self {
            TensormapField::GlobalDim(_) => DType::I32,
            _ => DType::I64,
        }
    }
}

impl fmt::Display for TensormapField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ord() {
            Some(o) => write!(f, "{}[{o}]", self.ptx_name()),
            None => f.write_str(self.ptx_name()),
        }
    }
}

/// The address operand of an instruction on the tensor map at `ptr`: an
/// `i64` for a global or generic pointer, an `i32` shared-window offset for
/// a shared one.
#[track_caller]
fn map_address<U: Numeric>(ptr: &Ptr<U>, func: &str, shared_ok: bool) -> (ValueId, AddressSpace) {
    if let Some(sw) = ptr.swizzle()
        && !sw.is_identity()
    {
        panic!("{func}: the tensor-map pointer is swizzled by {sw}; a tensor map is 128 plain bytes");
    }
    match ptr.memspace() {
        space @ (AddressSpace::Gmem | AddressSpace::Generic) => (ptr.toint_i64().checked(), space),
        AddressSpace::Smem if shared_ok => (ptr.toint_i32().checked(), AddressSpace::Smem),
        other => panic!(
            "{func}: the tensor map is at a {other} pointer; it lives in global memory{} (or at a generic address of it)",
            if shared_ok { " or shared memory" } else { "" }
        ),
    }
}

/// Replace one field of the tiled tensor map at `ptr` with `value`
/// (`tensormap.replace.tile.{field}{.global|.shared::cta}.b1024.{b32|b64}`;
/// CuTe DSL's experimental `tensormap_replace(field, ptr, new_value=, ord=)`).
/// `ptr` is a global, shared or generic pointer to the 128-byte,
/// 128-byte-aligned descriptor; `value` is an integer, converted to the
/// field's type ([`TensormapField::value_type`]). Every thread that executes
/// it writes (the whole object, a weak generic-proxy write): run it on one
/// thread. Needs sm_90a, sm_100a, sm_110a or sm_120a (or the sm_100f, sm_110f
/// and sm_120f families from PTX ISA 8.8).
#[track_caller]
pub fn tensormap_replace<U: Numeric>(ptr: &Ptr<U>, field: TensormapField, value: impl Into<AnyValue>) {
    let loc = Location::caller();
    let func = "tensormap_replace";
    match field {
        TensormapField::GlobalDim(o) => {
            assert!(o <= 4, "{func}: the ordinal of {field} is {o}; tensor maps have at most five dimensions (0 to 4)")
        }
        TensormapField::GlobalStride(o) => assert!(
            o <= 3,
            "{func}: the ordinal of {field} is {o}; strides are given for dimensions 1 to 4 (ordinals 0 to 3)"
        ),
        TensormapField::GlobalAddress => {}
    }
    let (addr, space) = map_address(ptr, func, true);
    let value = value.into().0;
    let ty = field.value_type();
    let value = builder::with(|b| {
        let have = b.dtype(value);
        assert!(
            matches!(have, DType::I32 | DType::U32 | DType::I64 | DType::U64),
            "{func}: the new {field} is an integer, not {have}"
        );
        if have == ty {
            value
        } else if have.bytes() == ty.bytes() {
            b.bitcast(value, ty, loc)
        } else {
            b.cast(value, ty, loc)
        }
    });
    builder::with(|b| b.push(Op::TensormapReplace { addr, space, field, value }, loc));
}

/// Copy the descriptor of the TMA atom `atom` to the 128 bytes at `dst`
/// (CuTe DSL's `cpasync.copy_tensormap(atom, ptr)`): sixteen 64-bit loads
/// from the grid-constant parameter and stores to `dst`, a global, shared or
/// generic pointer of any element type, which must be 128-byte aligned for
/// [`update_tma_descriptor`] (64 for a TMA copy). Every thread that executes
/// it copies: run it on one thread (CuTe DSL's `TensorMapManager` elects
/// one). The stores are generic-proxy writes: release them before a TMA
/// copy reads the descriptor.
#[track_caller]
pub fn copy_tensormap<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, dst: &Ptr<U>) {
    let func = "copy_tensormap";
    let Some(info) = atom.tma_info() else {
        panic!("{func}: {:?} is not a TMA atom (make one with cpasync::make_tiled_tma_atom)", atom.op())
    };
    map_address(dst, func, true);
    let words: Vec<_> = (0..crate::tma::TENSOR_MAP_BYTES / 8).map(|k| info.descriptor().load_word(k)).collect();
    let p = recast_ptr::<u64, U>(dst, None);
    for (k, w) in words.into_iter().enumerate() {
        (&p + k).store(w);
    }
}

/// Rewrite the descriptor at `desc_ptr` (a copy of `atom`'s, from
/// [`copy_tensormap`]) to describe `gmem_tensor` (CuTe DSL's
/// `cpasync.update_tma_descriptor(atom, gmem_tensor, desc_ptr)`): its global
/// address, then the extent of every TMA dimension and the byte stride of
/// dimensions 1 and up, each a `tensormap.replace`; the box, swizzle and
/// the other fields stay the atom's. `gmem_tensor` has the modes of the
/// tensor the atom was made for, in the same order, so each TMA dimension
/// takes its extent and stride from the same mode; its extents, strides and
/// pointer may be runtime values (a group's tensor in a grouped GEMM). Its
/// address must be 16-byte aligned and its strides multiples of 16 bytes
/// (the innermost mode's stride 1), as TMA requires: the simulator checks the
/// descriptor when a copy uses it. `desc_ptr` is global, shared or generic;
/// run it on one thread, then release the writes ([`fence_tma_desc_release`]
/// in global memory, [`cp_fence_tma_desc_release`] from shared memory).
#[track_caller]
pub fn update_tma_descriptor<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, gmem_tensor: &Tensor<T>, desc_ptr: &Ptr<U>) {
    let func = "update_tma_descriptor";
    let Some(info) = atom.tma_info() else {
        panic!("{func}: {:?} is not a TMA atom (make one with cpasync::make_tiled_tma_atom)", atom.op())
    };
    match gmem_tensor.memspace() {
        AddressSpace::Gmem | AddressSpace::Generic => {}
        other => panic!("{func}: the tensor is in {other} memory; a TMA descriptor describes global memory"),
    }
    if gmem_tensor.is_register() || !matches!(gmem_tensor.inner, TensorInner::Identity) {
        panic!("{func}: the tensor is not a plain strided tensor in memory, which is what a descriptor describes");
    }
    let flat = gmem_tensor.layout().flatten();
    let extents = flat.shape().flat_leaves();
    let strides: Vec<IndexExpr> = flat
        .stride()
        .flat_leaves()
        .into_iter()
        .enumerate()
        .map(|(k, s)| match s {
            StrideLeaf::Int(e) => e,
            StrideLeaf::Basis(b) => panic!("{func}: mode {k} of the tensor has the basis stride {b}, not an integer"),
        })
        .collect();
    if extents.len() != info.flat_modes() {
        panic!(
            "{func}: the tensor {} has {} flat modes, but the atom was made for a tensor of {}: give the tensor of \
             the new group with the atom's tensor's modes",
            gmem_tensor.layout(),
            extents.len(),
            info.flat_modes()
        );
    }
    let dims = info.dim_modes();
    for (k, e) in extents.iter().enumerate() {
        let walked = dims.iter().any(|m| m.contains(&k));
        if !walked && e.as_static().is_some_and(|e| e != 1) && strides[k].as_static() != Some(0) {
            panic!(
                "{func}: mode {k} of the tensor has {} elements, but the atom's tensor had a single element there \
                 (or stride 0), so its descriptor has no dimension for it",
                e.as_static().unwrap()
            );
        }
    }
    if let Some(s) = strides[dims[0][0]].as_static()
        && s != 1
    {
        panic!(
            "{func}: mode {} of the tensor is TMA's dimension 0, which is contiguous, but its stride is {s}",
            dims[0][0]
        );
    }
    let elem = T::DTYPE.bytes();
    // Everything computed before the first write.
    let address = recast_ptr::<T, T>(&gmem_tensor.ptr(), None).toint_i64();
    let mut updates: Vec<(TensormapField, AnyValue)> = vec![(TensormapField::GlobalAddress, address.into())];
    for (j, m) in dims.iter().enumerate() {
        if m.len() != 1 {
            panic!(
                "{func}: TMA dimension {j} of the atom folds the tensor's modes {m:?} into one, and rewriting a folded \
                 dimension is not supported"
            );
        }
        let k = m[0];
        let extent: crate::value::Int32 = (&extents[k]).into_value();
        updates.push((TensormapField::GlobalDim(j as u8), extent.into()));
    }
    for (j, m) in dims.iter().enumerate().skip(1) {
        let bytes = IndexExpr::product(strides[m[0]].clone(), IndexExpr::Static(elem));
        let stride: crate::value::Int64 = bytes.into_value();
        updates.push((TensormapField::GlobalStride(j as u8 - 1), stride.into()));
    }
    for (field, value) in updates {
        tensormap_replace(desc_ptr, field, value);
    }
}

/// `fence.proxy.tensormap::generic.acquire.gpu [ptr], 128` (CuTe DSL's
/// `cpasync.fence_tma_desc_acquire(ptr)`): this thread's later TMA copies
/// through the descriptor at `ptr` (global or generic) see the generic
/// writes released to the tensormap proxy before it. The thread that issues
/// the copy runs it (every thread of the warp, as `TensorMapManager` does,
/// covers the elected one). sm_90, PTX ISA 8.3.
#[track_caller]
pub fn fence_tma_desc_acquire<U: Numeric>(ptr: &Ptr<U>) {
    let loc = Location::caller();
    let (addr, _) = map_address(ptr, "fence_tma_desc_acquire", false);
    builder::with(|b| b.push(Op::FenceTensormapAcquire { addr }, loc));
}

/// `fence.proxy.tensormap::generic.release.gpu` (CuTe DSL's
/// `cpasync.fence_tma_desc_release()`): this thread's earlier generic writes
/// of tensor maps (stores, `tensormap.replace`) are released to the
/// tensormap proxy, for threads that acquire them. sm_90, PTX ISA 8.3.
#[track_caller]
pub fn fence_tma_desc_release() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::FenceTensormapRelease, loc));
}

/// Copy the descriptor at `sptr` in shared memory to `gptr` in global memory
/// and release it to the tensormap proxy
/// (`tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
/// [g], [s], 128`; CuTe DSL's `cpasync.cp_fence_tma_desc_release(gptr,
/// sptr)`). Warp-collective: every lane of a converged warp executes it with
/// the same addresses, and the warp copies once. sm_90, PTX ISA 8.3. Inline
/// assembly, which LLVM does not treat as convergent (see the module
/// documentation).
#[track_caller]
pub fn cp_fence_tma_desc_release<U: Numeric, V: Numeric>(gptr: &Ptr<U>, sptr: &Ptr<V>) {
    let loc = Location::caller();
    let func = "cp_fence_tma_desc_release";
    let g = match gptr.memspace() {
        AddressSpace::Generic => gptr.tospace(AddressSpace::Gmem),
        _ => gptr.clone(),
    };
    let (dst, gspace) = map_address(&g, func, false);
    assert!(gspace == AddressSpace::Gmem, "{func}: the destination is a global pointer");
    if sptr.memspace() != AddressSpace::Smem {
        panic!("{func}: the source is a {} pointer; it is the descriptor in shared memory", sptr.memspace());
    }
    let (src, _) = map_address(sptr, func, true);
    builder::with(|b| b.push(Op::TensormapCpFenceproxy { dst, src }, loc));
}

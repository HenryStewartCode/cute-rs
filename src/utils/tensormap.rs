// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's `TensorMapManager` (`cutlass/tensor_utils/tensormap.py`, with
//! deprecated aliases in `cutlass.utils`): the life cycle of the per-group
//! TMA descriptors of grouped and pointer-array GEMMs (roadmap part
//! m7-tensormap-update), built on [`cpasync`](crate::cpasync)'s
//! [`copy_tensormap`](crate::cpasync::copy_tensormap),
//! [`update_tma_descriptor`](crate::cpasync::update_tma_descriptor) and the
//! tensormap-proxy fences.
//!
//! The descriptors live in a workspace in global memory, typically a kernel
//! buffer parameter of `number of descriptors x 128` bytes, 128-byte aligned
//! ([`param_tensor_aligned`](crate::param_tensor_aligned) with alignment
//! 128), one slot per descriptor and CTA (or SM). In
//! [`TensorMapUpdateMode::GMEM`] the slot is edited in place; in
//! [`TensorMapUpdateMode::SMEM`] a copy in shared memory is edited and then
//! published to the slot with `tensormap.cp_fenceproxy`.

use crate::arch::{
    cp_async_bulk_commit_group, cp_async_bulk_wait_group, elect_one, fence_acq_rel_cta, make_warp_uniform, sync_warp,
    warp_idx,
};
use crate::atom::copy::CopyAtom;
use crate::control::if_;
use crate::cpasync::{
    copy_tensormap, cp_fence_tma_desc_release, fence_tma_desc_acquire, fence_tma_desc_release, update_tma_descriptor,
};
use crate::raw_ptr::make_ptr;
use crate::tensor::{AddressSpace, Ptr, Tensor};
use crate::value::Numeric;

/// Where [`TensorMapManager::update_tensormap`] edits a descriptor (CuTe
/// DSL's `TensorMapUpdateMode`): in place in global memory, or in a copy in
/// shared memory that is then copied back.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorMapUpdateMode {
    GMEM,
    SMEM,
}

/// The operations on in-memory TMA descriptors of a kernel (CuTe DSL's
/// `TensorMapManager(tensormap_update_mode, bytes_per_tensormap)`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TensorMapManager {
    pub tensormap_update_mode: TensorMapUpdateMode,
    /// The bytes of one descriptor's slot, which is also its alignment: 128.
    pub bytes_per_tensormap: usize,
}

impl TensorMapManager {
    /// A manager for `mode` with slots of `bytes_per_tensormap` bytes (128 in
    /// CuTe DSL's kernels; a power of two of at least 128).
    pub fn new(tensormap_update_mode: TensorMapUpdateMode, bytes_per_tensormap: usize) -> TensorMapManager {
        assert!(
            bytes_per_tensormap.is_power_of_two() && bytes_per_tensormap >= crate::tma::TENSOR_MAP_BYTES,
            "TensorMapManager: a descriptor's slot of {bytes_per_tensormap} bytes is not a power of two of at least \
             128"
        );
        TensorMapManager { tensormap_update_mode, bytes_per_tensormap }
    }

    /// The descriptor at the address of `ptr` as a pointer into
    /// `address_space` (global or generic), promised `bytes_per_tensormap`
    /// aligned (CuTe DSL's `get_tensormap_ptr(ptr, address_space=gmem)`).
    #[track_caller]
    pub fn get_tensormap_ptr<U: Numeric>(&self, ptr: &Ptr<U>, address_space: AddressSpace) -> Ptr<u64> {
        assert!(
            matches!(address_space, AddressSpace::Gmem | AddressSpace::Generic),
            "get_tensormap_ptr: a tensor-map pointer is global or generic, not {address_space}"
        );
        assert!(
            matches!(ptr.memspace(), AddressSpace::Gmem | AddressSpace::Generic),
            "get_tensormap_ptr: the pointer is in {} memory; the descriptors' workspace is in global memory",
            ptr.memspace()
        );
        make_ptr::<u64>(ptr.toint_i64(), address_space, Some(self.bytes_per_tensormap))
    }

    /// Initialise the descriptor at `dst_ptr` (global or shared) with `atom`'s:
    /// one elected lane of warp `warp_id` copies it, then every warp
    /// synchronises its lanes (CuTe DSL's `init_tensormap_from_atom`).
    #[track_caller]
    pub fn init_tensormap_from_atom<T: Numeric, U: Numeric>(&self, atom: &CopyAtom<T>, dst_ptr: &Ptr<U>, warp_id: i32) {
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(warp_id), || if_(elect_one(), || copy_tensormap(atom, dst_ptr)));
        sync_warp();
    }

    /// Order the initialisations before what follows (CuTe DSL's
    /// `fence_tensormap_initialization`): `fence.acq_rel.cta` in GMEM mode,
    /// nothing in SMEM mode.
    #[track_caller]
    pub fn fence_tensormap_initialization(&self) {
        if self.tensormap_update_mode == TensorMapUpdateMode::GMEM {
            fence_acq_rel_cta();
        }
    }

    /// Acquire the updated descriptor at `tensormap_ptr` for this thread's
    /// TMA copies (CuTe DSL's `fence_tensormap_update`,
    /// `fence.proxy.tensormap::generic.acquire.gpu` over its 128 bytes).
    #[track_caller]
    pub fn fence_tensormap_update<U: Numeric>(&self, tensormap_ptr: &Ptr<U>) {
        fence_tma_desc_acquire(tensormap_ptr);
    }

    /// Point the descriptors at `tensormap_gmem_ptr` (one per atom) at the
    /// tensors `tensor_gmem` (CuTe DSL's `update_tensormap(tensor_gmem,
    /// tma_copy_atom, tensormap_gmem_ptr, warp_id, tensormap_smem_ptr)`),
    /// on warp `warp_id`. In SMEM mode the warp rewrites the copies at
    /// `tensormap_smem_ptr`, waits (one elected lane) until its bulk
    /// async-groups have read their sources, as the earlier TMA stores
    /// through the old descriptors must before the global slots change, and
    /// publishes each copy to its global slot with
    /// `tensormap.cp_fenceproxy` (which releases it). In GMEM mode it waits
    /// the same way, rewrites the global slots in place and releases them
    /// with `fence.proxy.tensormap::generic.release`; `tensormap_smem_ptr`
    /// is unused. The threads that then copy through a descriptor acquire it
    /// first ([`fence_tensormap_update`](Self::fence_tensormap_update)).
    #[track_caller]
    pub fn update_tensormap<T: Numeric, U: Numeric, V: Numeric>(
        &self,
        tensor_gmem: &[&Tensor<T>],
        tma_copy_atom: &[&CopyAtom<T>],
        tensormap_gmem_ptr: &[Ptr<U>],
        warp_id: i32,
        tensormap_smem_ptr: &[Ptr<V>],
    ) {
        let n = tma_copy_atom.len();
        assert!(
            tensor_gmem.len() == n && tensormap_gmem_ptr.len() == n,
            "update_tensormap: {} tensors, {n} atoms and {} global descriptor pointers: give one of each per \
             descriptor",
            tensor_gmem.len(),
            tensormap_gmem_ptr.len()
        );
        let smem_mode = self.tensormap_update_mode == TensorMapUpdateMode::SMEM;
        if smem_mode {
            assert!(
                tensormap_smem_ptr.len() == n,
                "update_tensormap: SMEM mode takes one shared descriptor pointer per atom, not {}",
                tensormap_smem_ptr.len()
            );
        }
        // The shared addresses as warp-uniform values, before the branch (CuTe DSL does this
        // to keep sm_90a's code generation unpredicated).
        let uniform: Vec<Ptr<u64>> = if smem_mode {
            tensormap_smem_ptr
                .iter()
                .map(|p| {
                    assert!(
                        p.memspace() == AddressSpace::Smem,
                        "update_tensormap: the SMEM-mode descriptor copies are in shared memory, not {}",
                        p.memspace()
                    );
                    make_ptr::<u64>(make_warp_uniform(p.toint_i32()), AddressSpace::Smem, Some(p.alignment().max(8)))
                })
                .collect()
        } else {
            Vec::new()
        };
        let warp = make_warp_uniform(warp_idx());
        if_(warp.eq(warp_id), || {
            if smem_mode {
                for ((atom, tensor), sptr) in tma_copy_atom.iter().zip(tensor_gmem).zip(&uniform) {
                    update_tma_descriptor(atom, tensor, sptr);
                }
            }
            // Wait until the shared memory (or the global descriptors) may be rewritten.
            if_(elect_one(), || {
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, true);
            });
            sync_warp();
            if smem_mode {
                for (gptr, sptr) in tensormap_gmem_ptr.iter().zip(&uniform) {
                    cp_fence_tma_desc_release(gptr, sptr);
                }
            } else {
                for ((atom, tensor), gptr) in tma_copy_atom.iter().zip(tensor_gmem).zip(tensormap_gmem_ptr) {
                    update_tma_descriptor(atom, tensor, gptr);
                }
                sync_warp();
                fence_tma_desc_release();
            }
        });
    }
}

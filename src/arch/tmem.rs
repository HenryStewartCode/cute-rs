// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor-memory capacity (CuTe DSL's `cute.arch` TMEM helpers, roadmap
//! part m10-umma-layouts): how many TMEM columns a CTA can allocate on each
//! Blackwell target, and when an allocation must take the whole of TMEM.

use crate::Arch;

/// TMEM columns on sm_100 (deprecated in CuTe DSL in favour of
/// [`get_max_tmem_alloc_cols`]).
#[deprecated(note = "use get_max_tmem_alloc_cols(Arch::Sm100)")]
pub const SM100_TMEM_CAPACITY_COLUMNS: usize = 512;

/// The smallest TMEM allocation on sm_100, in columns (deprecated in CuTe
/// DSL in favour of [`get_min_tmem_alloc_cols`]).
#[deprecated(note = "use get_min_tmem_alloc_cols(Arch::Sm100)")]
pub const SM100_TMEM_MIN_ALLOC_COLUMNS: usize = 32;

/// The TMEM columns of `compute_capability` (CuTe DSL's
/// `cute.arch.get_max_tmem_alloc_cols`): 512 for sm_100, sm_103 and
/// sm_120, 576 for sm_107; the `a` and `f` targets have their base's.
/// Panics for any other target, which has no TMEM table entry.
#[track_caller]
pub fn get_max_tmem_alloc_cols(compute_capability: Arch) -> usize {
    match compute_capability.version() {
        (10, 0) | (10, 3) | (12, 0) => 512,
        (10, 7) => 576,
        _ => panic!("get_max_tmem_alloc_cols: unsupported compute capability {compute_capability}"),
    }
}

/// The smallest TMEM allocation of `compute_capability`, in columns (CuTe
/// DSL's `cute.arch.get_min_tmem_alloc_cols`): 32 on sm_100, sm_103,
/// sm_107 and sm_120 and their `a` and `f` targets. Panics for any other
/// target.
#[track_caller]
pub fn get_min_tmem_alloc_cols(compute_capability: Arch) -> usize {
    match compute_capability.version() {
        (10, 0) | (10, 3) | (10, 7) | (12, 0) => 32,
        _ => panic!("get_min_tmem_alloc_cols: unsupported compute capability {compute_capability}"),
    }
}

/// Whether allocating `num_columns` TMEM columns on `arch` needs the whole
/// of TMEM (CuTe DSL's `cute.arch.is_tmem_allocation_exclusive`): sm_107's
/// columns past 512 can only be had by allocating all 576, so it is true
/// for more than 512 columns on sm_107 (and sm_107a and sm_107f), false
/// otherwise.
pub fn is_tmem_allocation_exclusive(num_columns: usize, arch: Arch) -> bool {
    num_columns > 512 && arch.version() == (10, 7)
}

// m10-tmem-core: allocating tensor memory (PTX ISA 9.7.18.7.1, tcgen05.alloc,
// tcgen05.dealloc and tcgen05.relinquish_alloc_permit).

use std::panic::Location;

use crate::builder;
use crate::ir::MemSpace;
use crate::ir::{BufferId, DType, Op, Tcgen05Inst, ValueId};
use crate::layout::IndexExpr;
use crate::smem_struct::ScalarMember;
use crate::tensor::{AddressSpace, Ptr, index_operand, recast_ptr};
use crate::value::{Int32, IntoValue, Numeric, Value};

/// A 32-bit shared-memory word that holds a TMEM address: the slot
/// [`alloc_tmem`] writes the allocation's address to and
/// [`retrieve_tmem_ptr`] reads it back from. A [`Ptr`] to a 32-bit type in a
/// shared buffer, or a 32-bit scalar member of a shared struct (CuTe DSL's
/// `cute.struct` field).
pub trait TmemAddressSlot {
    /// The slot as an `Int32` pointer, checked.
    #[doc(hidden)]
    fn slot_ptr(&self, what: &str) -> Ptr<i32>;
}

impl<T: Numeric> TmemAddressSlot for Ptr<T> {
    #[track_caller]
    fn slot_ptr(&self, what: &str) -> Ptr<i32> {
        assert!(
            T::DTYPE.storage_bits() == 32,
            "{what}: a TMEM address is a 32-bit word, so the slot points at a 32-bit type, not {}",
            T::DTYPE
        );
        assert!(
            self.memory_space() == MemSpace::Shared && self.dsmem_peer().is_none(),
            "{what}: the TMEM address slot is in this CTA's shared memory, not {} ({self:?})",
            self.memspace()
        );
        recast_ptr::<i32, T>(self, None)
    }
}

impl<T: Numeric> TmemAddressSlot for ScalarMember<T> {
    #[track_caller]
    fn slot_ptr(&self, what: &str) -> Ptr<i32> {
        self.ptr().slot_ptr(what)
    }
}

/// The column count of an alloc or dealloc: its value, whether it is
/// exclusive (sm_107's allocations past 512 columns), checked when constant.
#[track_caller]
fn tmem_columns(num_columns: Int32, arch: Arch, what: &str) -> (ValueId, bool) {
    let max = get_max_tmem_alloc_cols(arch);
    let min = get_min_tmem_alloc_cols(arch);
    let mut exclusive = false;
    if let Some(n) = num_columns.as_constant() {
        let cols = usize::try_from(n).unwrap_or(0);
        exclusive = is_tmem_allocation_exclusive(cols, arch);
        if exclusive {
            assert!(
                cols.is_multiple_of(32) && cols <= max,
                "{what}: {n} columns on {arch}: an exclusive allocation takes a multiple of 32 columns, at most {max}"
            );
        } else {
            assert!(
                cols.is_power_of_two() && (min..=max.min(512)).contains(&cols),
                "{what}: {n} TMEM columns on {arch}: the count must be a power of two from {min} to {}",
                max.min(512)
            );
        }
    }
    (num_columns.checked(), exclusive)
}

/// Allocate `num_columns` columns of tensor memory (all 128 lanes) and write
/// the allocation's TMEM address (lane 0 in bits 31:16, the first column in
/// bits 15:0) to the shared-memory word `smem_ptr_to_write_address` (CuTe
/// DSL's `cute.arch.alloc_tmem`; `tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32`,
/// sm_100a and sm_110a, the sm_100f and sm_110f families from PTX 8.8).
///
/// One warp of the CTA executes it, all 32 lanes together; it blocks until
/// the columns are free. With `is_two_cta` (`.cta_group::2`) one warp in each
/// CTA of a pair (even and odd `%cluster_ctarank`) executes it, and both get
/// the same columns. Other warps read the address after a barrier
/// ([`retrieve_tmem_ptr`]). A constant `num_columns` must be a power of two
/// from 32 to 512; on sm_107 more than 512 makes the allocation exclusive
/// (`tcgen05.alloc.exclusive`: a multiple of 32 up to 576, the only live
/// allocation, PTX 9.4), emitted as inline PTX since no installed LLVM knows
/// it, which LLVM does not treat as convergent (see the README's notes on
/// collective instructions in inline assembly): call it from a converged
/// warp.
///
/// Everything allocated must be freed ([`dealloc_tmem`]) before the kernel
/// exits; all `tcgen05` instructions of a kernel use the same `cta_group`.
/// The simulator checks these rules, reports an allocation that can never be
/// served as a deadlock, and places an allocation at the lowest free range
/// aligned to its size, as the hardware does (code should not rely on it).
#[track_caller]
pub fn alloc_tmem(
    num_columns: impl IntoValue<i32>,
    smem_ptr_to_write_address: &impl TmemAddressSlot,
    is_two_cta: Option<bool>,
    arch: Arch,
) {
    let loc = Location::caller();
    let what = "alloc_tmem";
    let slot = smem_ptr_to_write_address.slot_ptr(what);
    let (ncols, exclusive) = tmem_columns(num_columns.into_value(), arch, what);
    let (buffer, offset): (BufferId, IndexExpr) = slot.parts();
    let index = index_operand(offset);
    builder::with(|b| {
        b.push(
            Op::Tcgen05 {
                inst: Tcgen05Inst::Alloc,
                two_cta: is_two_cta.unwrap_or(false),
                exclusive,
                buffer: Some(buffer),
                args: vec![index, ncols],
            },
            loc,
        )
    });
}

/// Free the `num_columns` TMEM columns at `tmem_ptr` that [`alloc_tmem`]
/// allocated (CuTe DSL's `cute.arch.dealloc_tmem`;
/// `tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32`). Warp-collective;
/// the count, `is_two_cta` and exclusiveness (sm_107, more than 512 columns:
/// `tcgen05.dealloc.exclusive`) must match the allocation's.
#[track_caller]
pub fn dealloc_tmem<T: Numeric>(
    tmem_ptr: &Ptr<T>,
    num_columns: impl IntoValue<i32>,
    is_two_cta: Option<bool>,
    arch: Arch,
) {
    let loc = Location::caller();
    let what = "dealloc_tmem";
    assert!(
        tmem_ptr.memspace() == AddressSpace::Tmem,
        "{what}: the pointer is a {} pointer, not a tensor-memory one",
        tmem_ptr.memspace()
    );
    let (ncols, exclusive) = tmem_columns(num_columns.into_value(), arch, what);
    let taddr = tmem_ptr.toint_i32().checked();
    builder::with(|b| {
        b.push(
            Op::Tcgen05 {
                inst: Tcgen05Inst::Dealloc,
                two_cta: is_two_cta.unwrap_or(false),
                exclusive,
                buffer: None,
                args: vec![taddr, ncols],
            },
            loc,
        )
    });
}

/// Give up the CTA's right to allocate tensor memory (CuTe DSL's
/// `cute.arch.relinquish_tmem_alloc_permit`;
/// `tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned`), so that
/// other CTAs on the SM may: a later [`alloc_tmem`] by this CTA is illegal.
/// Warp-collective.
#[track_caller]
pub fn relinquish_tmem_alloc_permit(is_two_cta: Option<bool>) {
    let loc = Location::caller();
    builder::with(|b| {
        b.push(
            Op::Tcgen05 {
                inst: Tcgen05Inst::RelinquishAllocPermit,
                two_cta: is_two_cta.unwrap_or(false),
                exclusive: false,
                buffer: None,
                args: Vec::new(),
            },
            loc,
        )
    });
}

/// The TMEM pointer [`alloc_tmem`] wrote to `ptr_to_buffer_holding_addr`
/// (CuTe DSL's `cute.arch.retrieve_tmem_ptr(element_type, alignment, ptr)`):
/// a 32-bit shared-memory load of the address, as a pointer to `T` in
/// [`AddressSpace::Tmem`] assumed `alignment`-byte aligned. Read it after a
/// barrier that orders it after the allocating warp's write; the simulator
/// checks that the address is in a live allocation of this CTA.
#[track_caller]
pub fn retrieve_tmem_ptr<T: Numeric>(alignment: usize, ptr_to_buffer_holding_addr: &impl TmemAddressSlot) -> Ptr<T> {
    let loc = Location::caller();
    let slot = ptr_to_buffer_holding_addr.slot_ptr("retrieve_tmem_ptr");
    let addr: Int32 = slot.load();
    let id = addr.checked();
    builder::with(|b| {
        debug_assert_eq!(b.dtype(id), DType::I32);
        b.push(
            Op::Tcgen05 {
                inst: Tcgen05Inst::CheckAddress,
                two_cta: false,
                exclusive: false,
                buffer: None,
                args: vec![id],
            },
            loc,
        )
    });
    crate::make_ptr::<T>(Value::<i32>::from_id(id), AddressSpace::Tmem, Some(alignment))
}

/// The column just past a TMEM tensor (CuTe DSL's
/// `cutlass.utils.find_tmem_tensor_col_offset`): the tensor recast to 32-bit
/// cells, whose cosize's low 16 bits (the column part of a TMEM offset) are
/// the columns it spans. Used to place a second tensor after the first.
#[track_caller]
pub fn find_tmem_tensor_col_offset<T: Numeric>(tmem_tensor: &crate::Tensor<T>) -> usize {
    assert!(
        tmem_tensor.ptr().memspace() == AddressSpace::Tmem,
        "find_tmem_tensor_col_offset: the tensor is not in tensor memory"
    );
    let cells = crate::recast_tensor::<i32, T>(tmem_tensor);
    cells.layout().cosize() & 0xFFFF
}

// m10-tmem-ld-st: waiting for TMEM loads and stores, and the tcgen05 thread-sync
// fences (PTX ISA 9.7.18.8.5 tcgen05.wait, 9.7.18.11.1 tcgen05.fence).

/// The tcgen05 operations [`fence_view_async_tmem_op`] waits for (CuTe DSL's
/// `kind="load"` and `kind="store"`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemOpKind {
    /// `tcgen05.ld`.
    Load,
    /// `tcgen05.st`.
    Store,
}

fn push_tcgen05(inst: Tcgen05Inst, loc: &'static Location<'static>) {
    builder::with(|b| {
        b.push(Op::Tcgen05 { inst, two_cta: false, exclusive: false, buffer: None, args: Vec::new() }, loc)
    });
}

/// Wait until the warp's earlier TMEM loads or stores have completed (CuTe
/// DSL's `cute.arch.fence_view_async_tmem_op(kind)`;
/// `tcgen05.wait::ld.sync.aligned` or `tcgen05.wait::st.sync.aligned`,
/// sm_100a and sm_110a). Warp-collective.
///
/// The registers a `tcgen05.ld` writes are undefined until the warp's
/// `wait::ld`, and the simulator reports reading one earlier. A
/// `tcgen05.st` stays in flight until the warp's `wait::st`: other warps
/// (after a barrier), MMAs and `dealloc_tmem` may use the columns only
/// after it.
#[track_caller]
pub fn fence_view_async_tmem_op(kind: TmemOpKind) {
    let inst = if kind == TmemOpKind::Load { Tcgen05Inst::WaitLd } else { Tcgen05Inst::WaitSt };
    push_tcgen05(inst, Location::caller());
}

/// [`fence_view_async_tmem_op`] for loads (`tcgen05.wait::ld`; CuTe DSL's
/// `cute.arch.fence_view_async_tmem_load`).
#[track_caller]
pub fn fence_view_async_tmem_load() {
    push_tcgen05(Tcgen05Inst::WaitLd, Location::caller());
}

/// [`fence_view_async_tmem_op`] for stores (`tcgen05.wait::st`; CuTe DSL's
/// `cute.arch.fence_view_async_tmem_store`).
#[track_caller]
pub fn fence_view_async_tmem_store() {
    push_tcgen05(Tcgen05Inst::WaitSt, Location::caller());
}

/// `tcgen05.fence::before_thread_sync` (PTX ISA 9.7.18.11.1): orders the
/// thread's earlier tcgen05 operations before a later thread
/// synchronisation (a barrier, an mbarrier arrive). Per thread.
///
/// CuTe DSL code does not write it: in kernels that load or store tensor
/// memory, the backend puts one before every barrier, mbarrier arrive and
/// cluster arrive by itself (and `after_thread_sync` after every barrier
/// and wait), so this is for code that wants it explicit.
#[track_caller]
pub fn tcgen05_fence_before_thread_sync() {
    push_tcgen05(Tcgen05Inst::FenceBeforeThreadSync, Location::caller());
}

/// `tcgen05.fence::after_thread_sync`: orders a thread synchronisation
/// before the thread's later tcgen05 operations. Per thread. See
/// [`tcgen05_fence_before_thread_sync`].
#[track_caller]
pub fn tcgen05_fence_after_thread_sync() {
    push_tcgen05(Tcgen05Inst::FenceAfterThreadSync, Location::caller());
}

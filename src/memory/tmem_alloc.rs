// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/tmem_allocator_sm100.hpp (Allocator1Sm / Allocator2Sm:
//   allocate, release_allocation_lock and free, and the 2-SM dealloc
//   handshake on the peer CTA's mbarrier),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text.

//! Tensor-memory allocation inside a kernel (roadmap part
//! m10-umma-pipelines): CuTe DSL's `cutlass.memory.TmemAllocator`, the
//! [`TmemBufferPool`] it reserves, and the column counts of TMEM layouts and
//! tensors ([`compute_tmem_cols_from_layout`], [`get_num_tmem_alloc_cols`]).

use crate::Arch;
use crate::arch::{
    MbarrierOptions, WARP_SIZE, block_idx_in_cluster, elect_one, make_warp_uniform, mbarrier_arrive_with,
    mbarrier_init, mbarrier_init_fence, mbarrier_wait, warp_idx,
};
use crate::arch::{
    alloc_tmem, dealloc_tmem, find_tmem_tensor_col_offset, get_max_tmem_alloc_cols, get_min_tmem_alloc_cols,
    relinquish_tmem_alloc_permit, retrieve_tmem_ptr,
};
use crate::control::if_;
use crate::ir::DType;
use crate::layout::{Layout, recast_layout};
use crate::pipeline::NamedBarrier;
use crate::tensor::{AddressSpace, Ptr, Tensor, make_tensor, recast_ptr};
use crate::value::Numeric;

use super::{SmemAllocator, SmemPartition};

/// The column bits of a TMEM address.
const TMEM_COL_MASK: usize = 0xFFFF;

/// The TMEM columns a layout of `dtype` elements spans (CuTe DSL's
/// `cutlass.memory.compute_tmem_cols_from_layout`): the layout recast to
/// 32-bit cells, whose cosize's low 16 bits are the columns. A Boolean
/// counts as 8 bits. The layout must be static (panics otherwise).
#[track_caller]
pub fn compute_tmem_cols_from_layout(layout: &Layout, dtype: DType) -> usize {
    let width = if dtype == DType::Bool { 8 } else { dtype.bits() };
    assert!(
        layout.is_static(),
        "cute: compute_tmem_cols_from_layout: the layout {layout} must be static, so that its TMEM columns are known \
         when the kernel is built"
    );
    recast_layout(32, width, layout).cosize() & TMEM_COL_MASK
}

/// A TMEM tensor whose columns [`get_num_tmem_alloc_cols`] counts (any
/// element type).
pub trait TmemColumns {
    /// The columns the tensor spans
    /// ([`find_tmem_tensor_col_offset`]).
    fn tmem_cols(&self) -> usize;
}

impl<T: Numeric> TmemColumns for Tensor<T> {
    #[track_caller]
    fn tmem_cols(&self) -> usize {
        find_tmem_tensor_col_offset(self)
    }
}

/// The TMEM columns to allocate for `tmem_tensors` (CuTe DSL's
/// `cutlass.memory.get_num_tmem_alloc_cols`): the sum of their column spans,
/// rounded up (with `rounding`) to a power of two no smaller than `arch`'s
/// minimum allocation. Panics when the count exceeds `arch`'s TMEM (512
/// columns, 576 on sm_107) or is below the minimum.
#[track_caller]
pub fn get_num_tmem_alloc_cols(tmem_tensors: &[&dyn TmemColumns], rounding: bool, arch: Arch) -> usize {
    let (max, min) = (get_max_tmem_alloc_cols(arch), get_min_tmem_alloc_cols(arch));
    let mut cols: usize = tmem_tensors.iter().map(|t| t.tmem_cols()).sum();
    if rounding {
        cols = cols.next_power_of_two().max(min);
    }
    assert!(
        (min..=max).contains(&cols),
        "cute: get_num_tmem_alloc_cols: {cols} TMEM columns is more than {arch}'s {max} or fewer than {min}"
    );
    cols
}

/// What [`TmemBufferPool::allocate`] takes: a number of columns, or a TMEM
/// layout sized with [`compute_tmem_cols_from_layout`].
#[derive(Clone, Debug)]
pub enum TmemPoolSize {
    Columns(usize),
    Layout(Layout),
}

impl From<usize> for TmemPoolSize {
    fn from(n: usize) -> Self {
        TmemPoolSize::Columns(n)
    }
}

impl From<Layout> for TmemPoolSize {
    fn from(l: Layout) -> Self {
        TmemPoolSize::Layout(l)
    }
}

impl From<&Layout> for TmemPoolSize {
    fn from(l: &Layout) -> Self {
        TmemPoolSize::Layout(l.clone())
    }
}

/// Sub-allocation from a reserved block of TMEM (CuTe DSL's
/// `cutlass.memory.TmemBufferPool`), made by [`TmemAllocator::reserve`].
/// The offsets are build-time column counts; only the base pointer is a
/// runtime value.
#[derive(Clone, Debug)]
pub struct TmemBufferPool {
    base_ptr: Ptr<f32>,
    total_cols: usize,
    current_offset: usize,
}

impl TmemBufferPool {
    /// A pool of `total_cols` columns from `base_ptr` (a TMEM pointer).
    #[track_caller]
    pub fn new<T: Numeric>(base_ptr: &Ptr<T>, total_cols: usize) -> TmemBufferPool {
        assert!(
            base_ptr.memspace() == AddressSpace::Tmem,
            "cute: TmemBufferPool: the base pointer is a {} pointer, not a tensor-memory one",
            base_ptr.memspace()
        );
        TmemBufferPool { base_ptr: recast_ptr::<f32, T>(base_ptr, None), total_cols, current_offset: 0 }
    }

    /// Where the pool starts in TMEM (as Float32, one element per column).
    pub fn base_ptr(&self) -> &Ptr<f32> {
        &self.base_ptr
    }

    /// The columns of the pool.
    pub fn total_cols(&self) -> usize {
        self.total_cols
    }

    /// The columns handed out so far.
    pub fn current_offset(&self) -> usize {
        self.current_offset
    }

    /// The columns left.
    pub fn remaining_cols(&self) -> usize {
        self.total_cols - self.current_offset
    }

    /// The next `size` columns (a count, or a layout of `T` elements), as a
    /// pointer to `T` at the current column. Panics when the pool has too
    /// few columns left.
    #[track_caller]
    pub fn allocate<T: Numeric>(&mut self, size: impl Into<TmemPoolSize>) -> Ptr<T> {
        let num_cols = match size.into() {
            TmemPoolSize::Columns(n) => n,
            TmemPoolSize::Layout(l) => compute_tmem_cols_from_layout(&l, T::DTYPE),
        };
        assert!(
            self.current_offset + num_cols <= self.total_cols,
            "cute: TmemBufferPool::allocate: {num_cols} columns asked for, {} left",
            self.remaining_cols()
        );
        let ptr = if self.current_offset == 0 {
            recast_ptr::<T, f32>(&self.base_ptr, None)
        } else {
            recast_ptr::<T, f32>(&self.base_ptr.offset(self.current_offset), None)
        };
        self.current_offset += num_cols;
        ptr
    }

    /// [`allocate`](Self::allocate) of `layout`, as a tensor with it.
    #[track_caller]
    pub fn allocate_tensor<T: Numeric>(&mut self, layout: &Layout) -> Tensor<T> {
        let ptr = self.allocate::<T>(layout);
        make_tensor(ptr, layout.clone())
    }
}

/// The optional arguments of [`TmemAllocator::new`].
#[derive(Clone, Debug)]
pub struct TmemAllocatorOptions {
    /// The shared-memory word the allocation's address is written to;
    /// `None` allocates one in the reserved partition.
    pub alloc_result_dst_smem_ptr: Option<Ptr<i32>>,
    /// The warp that allocates, relinquishes and frees. Default 0.
    pub allocator_warp_id: i32,
    /// Allocate for a CTA pair (`cta_group::2`).
    pub is_two_cta: bool,
    /// Columns already allocated (by an earlier allocator of the kernel).
    pub num_allocated_columns: usize,
    /// The mbarrier of the 2-CTA dealloc handshake; `None` with
    /// `is_two_cta` allocates one in the reserved partition. Give both
    /// pointers or neither.
    pub two_cta_tmem_dealloc_mbar_ptr: Option<Ptr<u64>>,
    /// The target, for the column limits (CuTe DSL's `arch` string).
    pub arch: Arch,
    /// Initialise the dealloc mbarrier (2-CTA only). Default true.
    pub initialize_mbarrier: bool,
}

impl Default for TmemAllocatorOptions {
    fn default() -> Self {
        TmemAllocatorOptions {
            alloc_result_dst_smem_ptr: None,
            allocator_warp_id: 0,
            is_two_cta: false,
            num_allocated_columns: 0,
            two_cta_tmem_dealloc_mbar_ptr: None,
            arch: Arch::Sm100,
            initialize_mbarrier: true,
        }
    }
}

/// Tensor-memory allocation for a kernel (CuTe DSL's
/// `cutlass.memory.TmemAllocator`): one warp allocates
/// ([`allocate`](Self::allocate)), every warp that uses the allocation waits
/// on a named barrier ([`wait_for_alloc`](Self::wait_for_alloc)) and reads
/// the address back ([`retrieve_ptr`](Self::retrieve_ptr)); the allocating
/// warp gives up the permit ([`relinquish_alloc_permit`](Self::relinquish_alloc_permit))
/// and frees the columns ([`free`](Self::free)) before the block exits.
///
/// It is built inside the kernel; the column count it keeps is a build-time
/// number. In 2-CTA mode each CTA of a pair allocates with its allocator
/// warp, and [`free`](Self::free) first arrives on the peer CTA's dealloc
/// mbarrier and waits on its own, so neither CTA frees while the other still
/// uses the pair's columns. That mbarrier is initialised by the constructor
/// (counting the 32 threads of the peer's allocator warp); publish it to the
/// cluster (a cluster barrier) before the first free.
#[derive(Clone, Debug)]
pub struct TmemAllocator {
    alloc_result_dst_smem_ptr: Ptr<i32>,
    barrier_for_retrieve: NamedBarrier,
    allocator_warp_id: i32,
    is_two_cta: bool,
    num_allocated_columns: usize,
    two_cta_tmem_dealloc_mbar_ptr: Option<Ptr<u64>>,
    arch: Arch,
    max_tmem_columns: usize,
}

impl TmemAllocator {
    /// An allocator whose readers synchronise on `barrier_for_retrieve`.
    /// Panics in 2-CTA mode when exactly one of the two shared-memory
    /// pointers is given.
    #[track_caller]
    pub fn new(barrier_for_retrieve: NamedBarrier, options: TmemAllocatorOptions) -> TmemAllocator {
        let smem = SmemAllocator::new();
        assert!(
            !options.is_two_cta
                || options.two_cta_tmem_dealloc_mbar_ptr.is_none() == options.alloc_result_dst_smem_ptr.is_none(),
            "cute: TmemAllocator::new: in 2-CTA mode give both the address slot and the dealloc mbarrier, or neither"
        );
        let slot = options
            .alloc_result_dst_smem_ptr
            .clone()
            .unwrap_or_else(|| smem.allocate::<i32>(8, SmemPartition::Reserved));
        let mbar = match (&options.two_cta_tmem_dealloc_mbar_ptr, options.is_two_cta) {
            (Some(p), _) => Some(p.clone()),
            (None, true) => Some(smem.allocate::<u64>(8, SmemPartition::Reserved)),
            (None, false) => None,
        };
        let allocator = TmemAllocator {
            alloc_result_dst_smem_ptr: slot,
            barrier_for_retrieve,
            allocator_warp_id: options.allocator_warp_id,
            is_two_cta: options.is_two_cta,
            num_allocated_columns: options.num_allocated_columns,
            two_cta_tmem_dealloc_mbar_ptr: mbar,
            arch: options.arch,
            max_tmem_columns: get_max_tmem_alloc_cols(options.arch),
        };
        if allocator.is_two_cta && options.initialize_mbarrier {
            allocator.init_dealloc_mbarrier();
        }
        allocator
    }

    /// The allocator warp's elected lane initialises the dealloc mbarrier
    /// for the 32 threads of the peer's allocator warp; then the init
    /// fence.
    #[track_caller]
    fn init_dealloc_mbarrier(&self) {
        let _at = crate::builder::LocationScope::new();
        let mbar = self.dealloc_mbarrier();
        if_(self.is_allocator_warp(), || {
            if_(elect_one(), || mbarrier_init(&mbar, WARP_SIZE as i32));
        });
        mbarrier_init_fence();
    }

    #[track_caller]
    fn dealloc_mbarrier(&self) -> Ptr<u64> {
        self.two_cta_tmem_dealloc_mbar_ptr
            .clone()
            .unwrap_or_else(|| panic!("cute: TmemAllocator: 2-CTA mode needs the dealloc mbarrier"))
    }

    #[track_caller]
    fn is_allocator_warp(&self) -> crate::value::Boolean {
        warp_idx().eq(self.allocator_warp_id)
    }

    /// The shared-memory word the allocation's address goes to.
    pub fn alloc_result_dst_smem_ptr(&self) -> &Ptr<i32> {
        &self.alloc_result_dst_smem_ptr
    }

    /// The 2-CTA dealloc mbarrier, if any.
    pub fn two_cta_tmem_dealloc_mbar_ptr(&self) -> Option<&Ptr<u64>> {
        self.two_cta_tmem_dealloc_mbar_ptr.as_ref()
    }

    /// The columns allocated so far (a build-time count).
    pub fn num_allocated_columns(&self) -> usize {
        self.num_allocated_columns
    }

    /// Whether the allocator works for a CTA pair.
    pub fn is_two_cta(&self) -> bool {
        self.is_two_cta
    }

    /// The allocating warp.
    pub fn allocator_warp_id(&self) -> i32 {
        self.allocator_warp_id
    }

    /// Whether `num_columns` can be allocated on the allocator's target
    /// (CuTe DSL's `check_valid_num_columns`): a non-negative multiple of 32,
    /// at most the target's TMEM, and a power of two, except that sm_107
    /// also takes counts above 512 that are not (its exclusive allocation of
    /// up to 576 columns).
    pub fn check_valid_num_columns(&self, num_columns: usize) -> bool {
        if num_columns > self.max_tmem_columns || !num_columns.is_multiple_of(32) {
            return false;
        }
        let pow2 = num_columns & num_columns.wrapping_sub(1) == 0;
        pow2 || (num_columns > 512 && self.arch.version() == (10, 7))
    }

    /// Allocate `num_columns` columns from the allocator warp
    /// (`tcgen05.alloc`). Panics for an invalid count, or one that would take
    /// the allocator past the target's TMEM.
    #[track_caller]
    pub fn allocate(&mut self, num_columns: usize) {
        let _at = crate::builder::LocationScope::new();
        assert!(
            self.check_valid_num_columns(num_columns),
            "cute: TmemAllocator::allocate: {num_columns} columns cannot be allocated: allocations are powers of two \
             (in steps of 32 columns) up to {}",
            self.max_tmem_columns
        );
        assert!(
            self.num_allocated_columns + num_columns <= self.max_tmem_columns,
            "cute: TmemAllocator::allocate: {} columns allocated in all, more than {}",
            self.num_allocated_columns + num_columns,
            self.max_tmem_columns
        );
        let (slot, two_cta, arch) = (&self.alloc_result_dst_smem_ptr, self.is_two_cta, self.arch);
        if_(self.is_allocator_warp(), || alloc_tmem(num_columns as i32, slot, Some(two_cta), arch));
        self.num_allocated_columns += num_columns;
    }

    /// Wait until the allocation is done: the retrieve barrier's
    /// `arrive_and_wait` (every thread that reads the address, the
    /// allocator warp included, takes part).
    #[track_caller]
    pub fn wait_for_alloc(&self) {
        self.barrier_for_retrieve.arrive_and_wait();
    }

    /// The allocation's address as a pointer to `T` in TMEM, 16-byte
    /// aligned ([`retrieve_tmem_ptr`]), read after
    /// [`wait_for_alloc`](Self::wait_for_alloc).
    #[track_caller]
    pub fn retrieve_ptr<T: Numeric>(&self) -> Ptr<T> {
        retrieve_tmem_ptr::<T>(16, &self.alloc_result_dst_smem_ptr)
    }

    /// [`allocate`](Self::allocate), [`wait_for_alloc`](Self::wait_for_alloc)
    /// and [`retrieve_ptr`](Self::retrieve_ptr), as a pool of `num_columns`
    /// columns to hand out.
    #[track_caller]
    pub fn reserve(&mut self, num_columns: usize) -> TmemBufferPool {
        self.allocate(num_columns);
        self.wait_for_alloc();
        let base = self.retrieve_ptr::<f32>();
        TmemBufferPool::new(&base, num_columns)
    }

    /// The allocator warp gives up the CTA's permit to allocate
    /// (`tcgen05.relinquish_alloc_permit`); a later allocation is illegal.
    #[track_caller]
    pub fn relinquish_alloc_permit(&self) {
        let _at = crate::builder::LocationScope::new();
        let two_cta = self.is_two_cta;
        if_(self.is_allocator_warp(), || relinquish_tmem_alloc_permit(Some(two_cta)));
    }

    /// Free `num_columns` columns at `tmem_ptr` from the allocator warp
    /// (`tcgen05.dealloc`); 0 frees everything allocated. In 2-CTA mode the
    /// allocator warp first arrives on the peer CTA's dealloc mbarrier
    /// (cluster rank `rank ^ 1`) and waits for the peer's arrival on its
    /// own. Panics for more columns than allocated or an invalid count.
    #[track_caller]
    pub fn free<T: Numeric>(&mut self, tmem_ptr: &Ptr<T>, num_columns: usize) {
        let _at = crate::builder::LocationScope::new();
        assert!(
            num_columns <= self.num_allocated_columns,
            "cute: TmemAllocator::free: {num_columns} columns, but only {} are allocated",
            self.num_allocated_columns
        );
        if num_columns != 0 {
            assert!(
                self.check_valid_num_columns(num_columns),
                "cute: TmemAllocator::free: {num_columns} is not a valid column count"
            );
        }
        let n = if num_columns == 0 { self.num_allocated_columns } else { num_columns };
        self.num_allocated_columns -= n;
        let (two_cta, arch) = (self.is_two_cta, self.arch);
        let mbar = self.two_cta_tmem_dealloc_mbar_ptr.clone();
        if_(self.is_allocator_warp(), || {
            if let Some(mbar) = &mbar {
                let rank = make_warp_uniform(block_idx_in_cluster());
                mbarrier_arrive_with(
                    mbar,
                    1,
                    MbarrierOptions { peer_cta_rank_in_cluster: Some(rank ^ 1), ..Default::default() },
                );
                mbarrier_wait(mbar, 0);
            }
            dealloc_tmem(tmem_ptr, n as i32, Some(two_cta), arch);
        });
    }
}

//! Shared-memory allocation by partition and capacity queries, from CuTe
//! DSL's `cutlass.memory`: [`SmemPartition`], [`SmemAllocator`], the
//! partition-size queries [`calculate_partition_size`] and
//! [`calculate_total_usage`], and [`get_smem_capacity_in_bytes`] and
//! [`get_kernel_smem_size`].
//!
//! Every shared buffer belongs to a partition. [`alloc_smem`](crate::alloc_smem),
//! [`make_smem_tensor`](crate::make_smem_tensor) and struct allocations use
//! the user partition; the allocator's methods take the partition as an
//! argument. The block's shared memory is laid out with every reserved
//! buffer first, in the order they were declared, then the user ones
//! ([`KernelIr::shared_memory_layout`](crate::ir::KernelIr::shared_memory_layout)),
//! so reserved storage such as the pipelines' mbarriers
//! ([`pipeline::alloc_reserved_mbarrier`](crate::pipeline::alloc_reserved_mbarrier))
//! sits at the lowest addresses even when it is allocated after user
//! buffers. On the GPU that holds when the kernel's shared memory is one
//! window (above 48 KiB, with a dynamic region or with overlays); a smaller
//! kernel's buffers are separate `.shared` variables that `ptxas` places
//! (the simulator still uses the layout). The partition sizes are the same
//! either way.
//!
//! The partition sizes are defined on that layout
//! ([`SharedMemoryLayout::partition_bytes`](crate::ir::SharedMemoryLayout::partition_bytes)):
//!
//! * reserved: where the last reserved buffer ends (0 without any);
//! * user: the rest of the build-time bytes, alignment padding in front of
//!   the first user buffer included;
//! * cumulative reserved: the same as reserved;
//! * cumulative user ([`calculate_total_usage`]): every build-time byte of
//!   the block, [`get_kernel_smem_size`].
//!
//! The dynamic region ([`get_dyn_smem`](crate::get_dyn_smem)) is sized at
//! launch and counts in none of them. A query returns an [`Int32`] that is
//! filled in when the kernel's build ends, so it sees allocations made after
//! it (CuTe DSL resolves the same query after tracing); it is not a constant
//! while the kernel is being built.
//!
//! ```
//! use cute::memory::{SmemAllocator, SmemPartition, calculate_partition_size};
//! use cute::prelude::*;
//!
//! let kernel = Kernel::builder("partitions").block_dim([32, 1, 1]).build(|| {
//!     let out = param_tensor::<i32>("out", Layout::col_major(2));
//!     let smem = SmemAllocator::new();
//!     let _data = smem.allocate_array::<f32>(100, 16, SmemPartition::User);
//!     // Asked for before the reserved array exists, answered with it.
//!     let reserved = calculate_partition_size(SmemPartition::Reserved, false);
//!     let _bars = smem.allocate_array::<u64>(4, 8, SmemPartition::Reserved);
//!     if_(thread_idx().x.eq(0), || {
//!         out.store(0, reserved);
//!         out.store(1, calculate_partition_size(SmemPartition::User, true));
//!     });
//! });
//! let layout = kernel.ir().shared_memory_layout();
//! // The reserved array comes first, then the user one.
//! assert_eq!((layout.reserved_bytes, layout.static_bytes), (32, 432));
//! let mut args = vec![KernelArg::buffer(&[0i32; 2])];
//! kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
//! assert_eq!(args[0].to_vec::<i32>(), [32, 432]);
//! ```

use crate::backend::Arch;
use crate::builder::{self, Kernel};
use crate::ir::BufferId;
pub use crate::ir::SmemPartition;
use crate::layout::{Inner, Swizzle};
use crate::smem_struct::StructLayout;
use crate::tensor::{Ptr, SmemLayout, Tensor};
use crate::value::{Int32, Numeric, Value};

// TMEM allocation in a kernel.
mod tmem_alloc;
pub use tmem_alloc::{
    TmemAllocator, TmemAllocatorOptions, TmemBufferPool, TmemColumns, TmemPoolSize, compute_tmem_cols_from_layout,
    get_num_tmem_alloc_cols,
};

/// The largest alignment a shared-memory allocation can ask for, in bytes.
pub const MAX_SMEM_ALIGN: usize = 1024;

/// Allocates shared memory in a partition (CuTe DSL's
/// `cutlass.memory.SmemAllocator`). It holds no state: the kernel's shared
/// memory is laid out once the build ends, from every allocation made
/// through any allocator or through [`alloc_smem`](crate::alloc_smem) and
/// its relatives, so allocators can be made wherever convenient. Sizes and
/// layouts are static, as in CuTe DSL.
///
/// Alignments are in bytes, powers of two of at most [`MAX_SMEM_ALIGN`]
/// (CuTe DSL's limit); every allocation is aligned to at least its
/// element's size.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmemAllocator;

impl SmemAllocator {
    /// An allocator for the kernel being built.
    pub fn new() -> SmemAllocator {
        SmemAllocator
    }

    /// The shared memory a block can use on `arch`
    /// ([`get_smem_capacity_in_bytes`]).
    #[track_caller]
    pub fn capacity_in_bytes(arch: Option<Arch>) -> usize {
        get_smem_capacity_in_bytes(arch)
    }

    /// `size` bytes, aligned to `byte_alignment` (CuTe DSL's
    /// `allocate(size)`); a size of 0 still takes a byte.
    #[track_caller]
    pub fn allocate_bytes(&self, size: usize, byte_alignment: usize, partition: SmemPartition) -> Ptr<u8> {
        check_alignment(byte_alignment, "allocate_bytes");
        let ptr = crate::alloc_smem::<u8>(size.max(1), byte_alignment);
        set_partition(ptr.buffer, partition);
        ptr
    }

    /// One `T` (CuTe DSL's `allocate(Numeric)`), aligned to the larger of
    /// `byte_alignment` and its size.
    #[track_caller]
    pub fn allocate<T: Numeric>(&self, byte_alignment: usize, partition: SmemPartition) -> Ptr<T> {
        self.allocate_array::<T>(1, byte_alignment, partition)
    }

    /// A struct (CuTe DSL's `allocate(struct)`), aligned to the larger of
    /// `byte_alignment` and the struct's alignment; returns its byte
    /// pointer, as [`StructLayout::allocate_aligned`] does.
    #[track_caller]
    pub fn allocate_struct(&self, layout: &StructLayout, byte_alignment: usize, partition: SmemPartition) -> Ptr<u8> {
        check_alignment(byte_alignment, "allocate_struct");
        let ptr = layout.allocate_aligned(byte_alignment);
        set_partition(ptr.buffer, partition);
        ptr
    }

    /// `num_elems` elements of `T`, at least one (CuTe DSL's
    /// `allocate_array`), aligned to the larger of `byte_alignment` and the
    /// element's size.
    #[track_caller]
    pub fn allocate_array<T: Numeric>(
        &self,
        num_elems: usize,
        byte_alignment: usize,
        partition: SmemPartition,
    ) -> Ptr<T> {
        assert!(num_elems >= 1, "cute: SmemAllocator::allocate_array: num_elems is 0; allocate at least one element");
        check_alignment(byte_alignment, "allocate_array");
        let align = byte_alignment.max(T::DTYPE.storage_bytes(1).max(1));
        let ptr = crate::alloc_smem::<T>(num_elems, align);
        set_partition(ptr.buffer, partition);
        ptr
    }

    /// A tensor of `T` with a static `layout` (CuTe DSL's
    /// `allocate_tensor`), aligned to the larger of `byte_alignment` and the
    /// element's size. `swizzle` swizzles the pointer (CuTe DSL's iterator
    /// swizzle), which a layout that is already swizzled (a composed layout
    /// with a swizzle inside) cannot also have. Sub-byte elements must fill
    /// whole bytes. The allocation is sized as
    /// [`make_smem_tensor`](crate::make_smem_tensor) sizes it.
    #[track_caller]
    pub fn allocate_tensor<T: Numeric>(
        &self,
        layout: impl SmemLayout,
        byte_alignment: usize,
        swizzle: Option<Swizzle>,
        partition: SmemPartition,
    ) -> Tensor<T> {
        check_alignment(byte_alignment, "allocate_tensor");
        let c = layout.into_composed();
        let layout_swizzled = matches!(&c.inner, Inner::Swizzle(sw) if !sw.is_identity());
        assert!(
            !(layout_swizzled && swizzle.is_some()),
            "cute: SmemAllocator::allocate_tensor: the layout is swizzled already ({}), so the pointer cannot take \
             a swizzle too",
            c
        );
        let align = byte_alignment.max(T::DTYPE.storage_bytes(1).max(1));
        let tensor = crate::make_smem_tensor_aligned::<T>(c.clone(), align);
        let ptr = tensor.ptr();
        set_partition(ptr.buffer, partition);
        match swizzle {
            Some(sw) => crate::make_tensor(&crate::recast_ptr::<T, T>(&ptr, Some(sw)), c.outer),
            None => tensor,
        }
    }

    /// [`calculate_partition_size`].
    #[track_caller]
    pub fn calculate_partition_size(&self, partition: SmemPartition, cumulative: bool) -> Int32 {
        calculate_partition_size(partition, cumulative)
    }

    /// [`calculate_total_usage`].
    #[track_caller]
    pub fn calculate_total_usage(&self) -> Int32 {
        calculate_total_usage()
    }
}

/// The bytes of shared memory `partition` takes in the finished kernel,
/// counting the partitions before it too when `cumulative` (CuTe DSL's
/// `SmemAllocator.calculate_partition_size`; the module docs define each
/// size). Allocations made after the call count: the value is set when the
/// build ends.
#[track_caller]
pub fn calculate_partition_size(partition: SmemPartition, cumulative: bool) -> Int32 {
    let loc = std::panic::Location::caller();
    Value::from_id(builder::with(|b| b.partition_size(partition, cumulative, loc)))
}

/// Every build-time byte of the block's shared memory (CuTe DSL's
/// `SmemAllocator.calculate_total_usage`): the cumulative size of the user
/// partition.
#[track_caller]
pub fn calculate_total_usage() -> Int32 {
    calculate_partition_size(SmemPartition::User, true)
}

/// The shared memory one block can use on `arch`, in bytes (CuTe DSL's
/// `get_smem_capacity_in_bytes`, [`Arch::smem_capacity_bytes`]). `None`
/// asks for the target of the kernel being built
/// ([`KernelOptions::target`](crate::KernelOptions::target), else
/// `CUTE_RS_ARCH`, as [`target_arch`](crate::target_arch) reports it); with
/// no target to go by, it panics, as CuTe DSL refuses a compute capability
/// it does not know.
#[track_caller]
pub fn get_smem_capacity_in_bytes(arch: Option<Arch>) -> usize {
    let arch = arch.or_else(builder::declared_target).unwrap_or_else(|| {
        panic!(
            "cute: get_smem_capacity_in_bytes: unsupported compute capability: no architecture was given and no \
             target is declared (KernelOptions::target or {})",
            crate::backend::ENV_ARCH
        )
    });
    arch.smem_capacity_bytes()
}

/// The build-time shared memory of `kernel`, in bytes (CuTe DSL's
/// `get_kernel_smem_size`): [`KernelIr::shared_memory_bytes`](crate::ir::KernelIr::shared_memory_bytes).
pub fn get_kernel_smem_size(kernel: &Kernel) -> usize {
    kernel.ir().shared_memory_bytes()
}

#[track_caller]
fn check_alignment(align: usize, what: &str) {
    assert!(
        align.is_power_of_two(),
        "cute: SmemAllocator::{what}: an alignment of {align} bytes is not a power of two"
    );
    assert!(
        align <= MAX_SMEM_ALIGN,
        "cute: SmemAllocator::{what}: an alignment of {align} bytes is more than the {MAX_SMEM_ALIGN} bytes shared \
         memory allows"
    );
}

fn set_partition(buffer: BufferId, partition: SmemPartition) {
    builder::with(|b| b.buffer_mut(buffer).partition = partition);
}

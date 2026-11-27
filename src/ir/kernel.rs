//! The kernel container: `Inst`, `Region`, buffer and parameter declarations, `KernelIr`.

use super::*;

/// An instruction: an operation plus the user source location that built it.
#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Op,
    pub loc: Loc,
    /// The profiling [`Scope`](crate::Scope) the instruction was built in.
    pub scope: Option<crate::profiling::ScopeId>,
}

/// A straight-line list of instructions, terminated by the values it yields
/// to its parent (`If` results or the next `For` iteration).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub insts: Vec<Inst>,
    pub yields: Vec<ValueId>,
}

#[derive(Clone, Debug)]
pub struct BufferDecl {
    pub name: String,
    pub elem: DType,
    /// Number of elements, or `None` for a buffer sized at launch: a global
    /// buffer, or the dynamic shared-memory region ([`get_dyn_smem`]), whose
    /// bytes [`LaunchConfig::dynamic_smem_bytes`] sets.
    ///
    /// [`get_dyn_smem`]: crate::get_dyn_smem
    /// [`LaunchConfig::dynamic_smem_bytes`]: crate::LaunchConfig::dynamic_smem_bytes
    pub len: Option<usize>,
    pub space: MemSpace,
    /// Byte alignment of the allocation (shared memory) or the alignment the
    /// kernel assumes for the pointer it is given (global memory).
    pub align: usize,
    /// For a shared-memory struct ([`cute_struct!`](crate::cute_struct)),
    /// the struct's name and its members as `name:size:offset` (CuTe DSL's
    /// field table); empty otherwise. [`Kernel::dump`](crate::Kernel::dump)
    /// shows it.
    pub fields: Vec<String>,
    /// The shared-memory partition the buffer belongs to (CuTe DSL's
    /// `smem.partition_id`): [`SmemPartition::Reserved`] buffers are laid
    /// out before every [`SmemPartition::User`] one. Other memory spaces
    /// ignore it.
    pub partition: SmemPartition,
}

/// The partition of shared memory a buffer is allocated in (CuTe DSL's
/// `cutlass.memory.SmemPartition`, an `IntEnum` whose values these keep).
/// [`KernelIr::shared_memory_layout`] places every reserved buffer first,
/// in declaration order, so reserved storage (pipeline mbarriers,
/// [`pipeline::alloc_reserved_mbarrier`](crate::pipeline::alloc_reserved_mbarrier))
/// keeps the lowest shared-memory addresses whatever is allocated around
/// it; the user buffers follow. (A kernel whose shared memory is not one
/// window emits each buffer as its own `.shared` variable, placed by
/// `ptxas`.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SmemPartition {
    /// Storage the library reserves for itself, at the lowest addresses.
    Reserved = 0,
    /// Ordinary allocations (the default).
    #[default]
    User = 1,
}

/// The symbol of the dynamic shared-memory window in the emitted MLIR and
/// PTX: an external zero-length array (`memref.global` without a value,
/// PTX's `.extern .shared .align 16 .b8 __cute_dynamic_smem[];`) that the
/// driver places at the start of the block's dynamic shared memory. It is
/// not `gpu.dynamic_shared_memory`, which MLIR lowers to an internal global
/// that LLVM's NVPTX backend demotes to one byte of static shared memory in
/// the kernel: that byte counts against the 48 KB and opt-in limits and in
/// `%total_smem_size`. No kernel may have this name.
pub const DYNAMIC_SMEM_SYMBOL: &str = "__cute_dynamic_smem";

/// Where one block's shared memory lives ([`KernelIr::shared_memory_layout`]).
///
/// Buffers sized at build time are static shared memory (`.shared`
/// variables) up to [`MAX_STATIC_SHARED_BYTES`] in all. Above that, or when
/// the kernel has a dynamic region ([`get_dyn_smem`]), every shared buffer
/// lives in the dynamic shared-memory window that the launch sizes
/// ([`DYNAMIC_SMEM_SYMBOL`], PTX's `.extern .shared`): the build-time
/// buffers first, at their starts, then the dynamic region, at
/// `dynamic_start`. A launch passes `dynamic_start` plus the region's bytes
/// as its dynamic shared memory, which is what `%dynamic_smem_size` reads;
/// such a kernel has no static shared memory.
///
/// [`MAX_STATIC_SHARED_BYTES`]: crate::validate::MAX_STATIC_SHARED_BYTES
/// [`get_dyn_smem`]: crate::get_dyn_smem
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedMemoryLayout {
    /// The start of each shared buffer in bytes, by buffer (`None` for the
    /// other memory spaces). The dynamic region starts at `dynamic_start`.
    pub starts: Vec<Option<usize>>,
    /// The bytes the buffers sized at build time span.
    pub static_bytes: usize,
    /// Whether the shared buffers live in the dynamic shared-memory window
    /// rather than in static shared memory.
    pub in_window: bool,
    /// Where the dynamic region starts in the window: after the build-time
    /// buffers when they are in the window (at a multiple of the region's
    /// alignment, if the kernel has a region), 0 otherwise.
    pub dynamic_start: usize,
    /// The alignment the window declares (`.extern .shared .align`): the
    /// largest of its buffers' alignments, and at least 16, so that each
    /// buffer's start in the window is aligned in shared memory too.
    pub window_align: usize,
    /// The bytes the [`SmemPartition::Reserved`] buffers span from offset
    /// 0: where the last of them ends (0 without any). The user buffers
    /// take the rest of `static_bytes`.
    pub reserved_bytes: usize,
}

impl SharedMemoryLayout {
    /// The bytes of `partition`, as CuTe DSL's
    /// `SmemAllocator.calculate_partition_size(partition, cumulative)`
    /// reports them: [`reserved_bytes`](Self::reserved_bytes) for the
    /// reserved partition, `static_bytes - reserved_bytes` for the user one
    /// (alignment padding in front of the first user buffer included);
    /// cumulatively, the reserved partition is the same and the user one is
    /// every build-time byte of the block (`static_bytes`, CuTe DSL's
    /// `calculate_total_usage`). The dynamic region
    /// ([`get_dyn_smem`](crate::get_dyn_smem)) is sized at launch and not
    /// counted.
    pub fn partition_bytes(&self, partition: SmemPartition, cumulative: bool) -> usize {
        match (partition, cumulative) {
            (SmemPartition::Reserved, _) => self.reserved_bytes,
            (SmemPartition::User, false) => self.static_bytes - self.reserved_bytes,
            (SmemPartition::User, true) => self.static_bytes,
        }
    }

    /// The dynamic shared memory a launch passes to the driver
    /// (`sharedMemBytes`, read back by `%dynamic_smem_size`) when the
    /// dynamic region gets `dynamic` bytes.
    pub fn launch_bytes(&self, dynamic: usize) -> usize {
        self.dynamic_start + dynamic
    }

    /// The shared memory one block uses, static and dynamic, when the
    /// dynamic region gets `dynamic` bytes.
    pub fn block_bytes(&self, dynamic: usize) -> usize {
        if self.in_window { self.launch_bytes(dynamic) } else { self.static_bytes + dynamic }
    }
}

#[derive(Clone, Debug)]
pub enum Param {
    Buffer(BufferId),
    /// `multiple_of` is what the kernel assumes the launch value divides by.
    Scalar {
        name: String,
        value: ValueId,
        multiple_of: usize,
    },
    /// A by-value parameter (a C struct or a 128-byte TMA descriptor): `buffer` holds its
    /// bytes, read-only, as the kernel's `.param` aggregate. With `grid_constant`, the
    /// kernel may take its address without a local copy (`nvvm.grid_constant`).
    ByValue {
        buffer: BufferId,
        grid_constant: bool,
    },
}

/// A module-scope global (`__device__` or `__constant__`, CuTe DSL's module-level
/// `cutlass.Array`), declared with [`module_global`](crate::module_global).
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleGlobal {
    /// The buffer that holds it: [`MemSpace::Global`] or [`MemSpace::Constant`], named by
    /// its C identifier (the symbol `cuModuleGetGlobal` finds).
    pub buffer: BufferId,
    /// Initial bytes (little-endian), or `None` for zeros.
    pub init: Option<Vec<u8>>,
    /// An LLVM `constant`: the kernel may fold its loads, and host writes are refused.
    pub readonly: bool,
}

#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub args: Vec<Param>,
    /// The parameters that hold tensor maps (by-value grid constants): hidden ones,
    /// which come last in `args` and are computed at launch, and ones the host binds.
    pub tensor_maps: Vec<crate::tma::TensorMapParam>,
    pub buffers: Vec<BufferDecl>,
    /// Type of every SSA value, indexed by `ValueId`.
    pub value_types: Vec<DType>,
    pub body: Region,
    /// Block size the kernel was written for, if declared.
    pub block_dim: Option<[u32; 3]>,
    /// The largest block the kernel may be launched with ([`KernelOptions::max_threads`],
    /// PTX's `.maxntid`).
    ///
    /// [`KernelOptions::max_threads`]: crate::KernelOptions::max_threads
    pub max_threads: Option<[u32; 3]>,
    /// Blocks per SM the kernel asks the compiler to fit (`.minnctapersm`).
    pub min_blocks_per_sm: Option<u32>,
    /// Module-scope globals.
    pub globals: Vec<ModuleGlobal>,
    /// Shared-memory overlays ([`smem_overlay`](crate::smem_overlay)): per group, the
    /// buffers of each alternative, which all start at the group's offset.
    pub overlays: Vec<Vec<Vec<BufferId>>>,
    /// Where the kernel was built (the `Kernel::build` call): the kernel
    /// function's own source location.
    pub loc: Loc,
    /// External device functions the kernel calls ([`Op::Call`]), one per symbol.
    pub externs: Vec<ExternDecl>,
    /// LLVM bitcode files linked into the module (CuTe DSL's `BitCode` sources and
    /// `LinkLibraries`), in first-use order.
    pub link_libraries: Vec<std::path::PathBuf>,
    /// Set for a standalone device function ([`DeviceFunction`](crate::ffi::DeviceFunction)):
    /// the IR is then a `__device__` function with real parameters and a return value.
    pub function: Option<DeviceFunctionSig>,
    /// The multicast buffers ([`param_tensor_multicast`](crate::utils::distributed::param_tensor_multicast)):
    /// parameters holding multimem addresses, which only `multimem` instructions access.
    pub multicast: Vec<BufferId>,
    /// The profiling scopes ([`Scope`](crate::Scope)) instructions refer to by
    /// [`Inst::scope`].
    pub scopes: Vec<crate::profiling::NameScope>,
    /// The device workspace the kernel takes as its last argument, if any
    /// ([`crate::experimental::host_runtime`]).
    pub workspace: Option<crate::experimental::host_runtime::WorkspaceDecl>,
    /// The Float `+`, `-` and `*` results that must not be contracted into `fma` (built in
    /// an exact scope, such as the precise [`math`](crate::math) functions'), whatever
    /// [`CompileOptions::fp_contract`](crate::CompileOptions::fp_contract) says.
    pub exact_float: std::collections::BTreeSet<ValueId>,
}

impl KernelIr {
    /// The path of scope `id`, outermost name first, joined with `/`.
    pub fn scope_path(&self, id: crate::profiling::ScopeId) -> String {
        let mut names = Vec::new();
        let mut at = Some(id);
        while let Some(s) = at {
            let scope = &self.scopes[s.0 as usize];
            names.push(scope.name.as_str());
            at = scope.parent;
        }
        names.reverse();
        names.join("/")
    }

    /// The scope path of the first scoped instruction built at `loc` (what simulator
    /// diagnostics print), or `None` when the kernel has no scopes or none there.
    pub fn scope_path_at(&self, loc: Loc) -> Option<String> {
        fn find(r: &Region, loc: Loc) -> Option<crate::profiling::ScopeId> {
            r.insts.iter().find_map(|i| {
                let here = (std::ptr::eq(i.loc, loc) || i.loc == loc).then_some(i.scope).flatten();
                here.or_else(|| i.op.regions().into_iter().find_map(|sub| find(sub, loc)))
            })
        }
        if self.scopes.is_empty() {
            return None;
        }
        find(&self.body, loc).map(|id| self.scope_path(id))
    }

    /// Whether `b` is a multicast buffer.
    pub fn is_multicast(&self, b: BufferId) -> bool {
        self.multicast.contains(&b)
    }

    /// The declaration of external function `id`.
    pub fn extern_decl(&self, id: ExternId) -> &ExternDecl {
        &self.externs[id.0 as usize]
    }

    /// The number of hidden parameters (the last ones), which launches do not pass.
    pub fn hidden_param_count(&self) -> usize {
        self.tensor_maps.iter().filter(|t| t.source.is_some()).count()
    }

    pub fn dtype(&self, v: ValueId) -> DType {
        self.value_types[v.0 as usize]
    }

    pub fn buffer(&self, b: BufferId) -> &BufferDecl {
        &self.buffers[b.0 as usize]
    }

    /// Where one block's shared memory lives: each buffer's start, the bytes
    /// of the buffers sized at build time, and where the dynamic region
    /// ([`get_dyn_smem`](crate::get_dyn_smem)) starts. The MLIR emitter, the
    /// CUDA launcher and the simulator all place shared memory by it.
    pub fn shared_memory_layout(&self) -> SharedMemoryLayout {
        let mut total = 0usize;
        let mut reserved_bytes = 0usize;
        let mut starts: Vec<Option<usize>> = vec![None; self.buffers.len()];
        // The reserved partition first, in declaration order, then the user one.
        let reserved = |k: &usize| self.buffers[*k].partition == SmemPartition::Reserved;
        let order: Vec<usize> =
            (0..self.buffers.len()).filter(reserved).chain((0..self.buffers.len()).filter(|k| !reserved(k))).collect();
        for k in order {
            let b = &self.buffers[k];
            if b.space != MemSpace::Shared || b.len.is_none() || starts[k].is_some() {
                continue;
            }
            // An overlay group is placed at its first buffer: every alternative starts at the
            // group's start (aligned for all of them), and the group spans the largest.
            let group = self.overlays.iter().find(|g| g.iter().flatten().any(|m| m.0 as usize == k));
            let members: Vec<usize> = match group {
                Some(g) => g.iter().flatten().map(|m| m.0 as usize).collect(),
                None => vec![k],
            };
            let align = members.iter().map(|&m| self.buffers[m].align).max().unwrap_or(1);
            let start = total.next_multiple_of(align);
            let mut end = start;
            for &m in &members {
                let d = &self.buffers[m];
                starts[m] = Some(start);
                end = end.max(start + d.len.unwrap_or(0) * d.elem.bytes());
            }
            total = end;
            if b.partition == SmemPartition::Reserved {
                reserved_bytes = end;
            }
        }
        let region = self.dynamic_smem_region();
        // Overlaid buffers share bytes, which only the window's byte views express.
        let in_window =
            region.is_some() || total > crate::validate::MAX_STATIC_SHARED_BYTES || !self.overlays.is_empty();
        let dynamic_start = match region {
            Some(r) => total.next_multiple_of(self.buffer(r).align),
            None if in_window => total,
            None => 0,
        };
        if let Some(r) = region {
            starts[r.0 as usize] = Some(dynamic_start);
        }
        let window_align =
            self.buffers.iter().filter(|b| b.space == MemSpace::Shared).map(|b| b.align).fold(16, usize::max);
        SharedMemoryLayout { starts, static_bytes: total, in_window, dynamic_start, window_align, reserved_bytes }
    }

    /// Bytes of the shared buffers of one block sized at build time (the
    /// dynamic region's bytes are set at launch).
    pub fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_layout().static_bytes
    }

    /// The dynamic shared-memory region ([`get_dyn_smem`](crate::get_dyn_smem)),
    /// if the kernel has one: the shared buffer sized at launch.
    pub fn dynamic_smem_region(&self) -> Option<BufferId> {
        self.buffers.iter().position(|b| b.space == MemSpace::Shared && b.len.is_none()).map(|i| BufferId(i as u32))
    }

    /// Count instructions (recursively) matching `pred`.
    pub fn count_ops(&self, pred: impl Fn(&Op) -> bool + Copy) -> usize {
        fn walk(r: &Region, pred: impl Fn(&Op) -> bool + Copy) -> usize {
            r.insts
                .iter()
                .map(|i| {
                    let nested: usize = i.op.regions().into_iter().map(|r| walk(r, pred)).sum();
                    nested + pred(&i.op) as usize
                })
                .sum()
        }
        walk(&self.body, pred)
    }
}

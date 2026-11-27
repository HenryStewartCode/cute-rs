// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.memory (SmemAllocator,
// TmemAllocator, alias groups; NVIDIA EULA); the code, the layout report
// and the messages are written independently.

//! Declarative allocators: resources declare the shared memory
//! ([`SmemAllocation`]) and tensor memory ([`TmemAllocation`]) they need,
//! the allocators lay everything out once on the host, and the TaskManager
//! materialises the layout in the kernel's prologue (one unified shared
//! memory block holding the data and the pipelines' barriers; one TMEM
//! allocation, freed in the epilogue).

use std::fmt;

use crate::memory::MAX_SMEM_ALIGN;

use super::pipeline_group::PipelineGroup;
use super::resource::{Resource, SmemAllocation, TmemAllocation};

/// Bytes of one mbarrier.
pub const MBARRIER_BYTES: usize = 8;

/// The default shared-memory capacity the allocator checks against:
/// 227 KiB, what one block of sm_90 and sm_100 can opt in to.
pub const DEFAULT_SMEM_CAPACITY: usize = 227 * 1024;

/// TMEM columns of one CTA (sm_100).
pub const TMEM_COLUMNS: usize = 512;

/// Why a layout cannot be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// An alignment that is not a power of two, or above the hardware's.
    BadAlignment { name: String, alignment: usize },
    /// More memory than the capacity (`unit` is "bytes" or "columns").
    OverCapacity { what: &'static str, needed: usize, capacity: usize, unit: &'static str },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::BadAlignment { name, alignment } => write!(
                f,
                "`{name}` asks for {alignment}-byte alignment; alignments are powers of two up to {MAX_SMEM_ALIGN}"
            ),
            AllocError::OverCapacity { what, needed, capacity, unit } => {
                write!(f, "the {what} layout needs {needed} {unit}, more than the {capacity} available")
            }
        }
    }
}

impl std::error::Error for AllocError {}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// One placed region of a layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// The allocation's (or barrier block's) name.
    pub name: String,
    /// The resource that declared it, if known.
    pub owner: Option<String>,
    /// Where it starts (bytes, or TMEM columns).
    pub offset: usize,
    /// Its size.
    pub size: usize,
    /// Its alignment (1 for TMEM).
    pub alignment: usize,
    /// The alias group it shares memory in, if any.
    pub alias_group: Option<usize>,
    /// Its phase within that group.
    pub phase: usize,
}

impl Region {
    /// One past its last byte (or column).
    pub fn end(&self) -> usize {
        self.offset + self.size
    }
}

/// Shared declarations and layout of both allocators.
#[derive(Clone, Debug, Default)]
struct Plan<A> {
    /// (owner, allocation)
    allocations: Vec<(Option<String>, A)>,
    /// alias groups: phases of allocations sharing one block.
    alias_groups: Vec<Vec<Vec<A>>>,
}

trait Alloc: Clone {
    fn name(&self) -> &str;
    fn size(&self) -> usize;
    fn alignment(&self) -> usize;
    fn same(&self, other: &Self) -> bool;
    fn set(&self, offset: usize);
}

impl Alloc for SmemAllocation {
    fn name(&self) -> &str {
        &self.name
    }
    fn size(&self) -> usize {
        self.size_bytes()
    }
    fn alignment(&self) -> usize {
        self.alignment.max(1)
    }
    fn same(&self, other: &Self) -> bool {
        self.same_allocation(other)
    }
    fn set(&self, offset: usize) {
        self.set_offset(offset);
    }
}

impl Alloc for TmemAllocation {
    fn name(&self) -> &str {
        &self.name
    }
    fn size(&self) -> usize {
        self.num_columns
    }
    fn alignment(&self) -> usize {
        1
    }
    fn same(&self, other: &Self) -> bool {
        self.same_allocation(other)
    }
    fn set(&self, offset: usize) {
        self.set_offset(offset);
    }
}

impl<A: Alloc> Plan<A> {
    fn add(&mut self, owner: Option<&str>, a: A) {
        if !self.allocations.iter().any(|(_, b)| b.same(&a)) {
            self.allocations.push((owner.map(str::to_string), a));
        }
    }

    fn owner_of(&self, a: &A) -> Option<String> {
        self.allocations.iter().find(|(_, b)| b.same(a)).and_then(|(o, _)| o.clone())
    }

    fn aliased(&self, a: &A) -> bool {
        self.alias_groups.iter().flatten().flatten().any(|b| b.same(a))
    }

    /// Lay out: standalone allocations and alias groups become blocks
    /// (a group's block is as large as its largest phase, each phase laid
    /// out from the block's start), placed largest alignment first (then
    /// in declaration order) to cut padding. Returns the regions and the
    /// end of the data.
    fn layout(&self) -> (Vec<(A, Region)>, usize) {
        struct Block<A> {
            alignment: usize,
            size: usize,
            members: Vec<(A, usize, Option<usize>, usize)>, // (alloc, offset in block, group, phase)
        }
        let mut blocks: Vec<Block<A>> = vec![];
        for (_, a) in &self.allocations {
            if !self.aliased(a) {
                blocks.push(Block { alignment: a.alignment(), size: a.size(), members: vec![(a.clone(), 0, None, 0)] });
            }
        }
        for (gi, group) in self.alias_groups.iter().enumerate() {
            let alignment = group.iter().flatten().map(Alloc::alignment).max().unwrap_or(1);
            let mut size = 0;
            let mut members = vec![];
            for (pi, phase) in group.iter().enumerate() {
                let mut cursor = 0;
                for a in phase {
                    let off = align_up(cursor, a.alignment());
                    members.push((a.clone(), off, Some(gi), pi));
                    cursor = off + a.size();
                }
                size = size.max(cursor);
            }
            blocks.push(Block { alignment, size, members });
        }
        blocks.sort_by_key(|b| std::cmp::Reverse(b.alignment));
        let mut cursor = 0;
        let mut out = vec![];
        for b in blocks {
            let base = align_up(cursor, b.alignment);
            for (a, off, group, phase) in b.members {
                let region = Region {
                    name: a.name().to_string(),
                    owner: self.owner_of(&a),
                    offset: base + off,
                    size: a.size(),
                    alignment: a.alignment(),
                    alias_group: group,
                    phase,
                };
                out.push((a, region));
            }
            cursor = base + b.size;
        }
        (out, cursor)
    }
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

/// A block of pipeline barriers the allocator places.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierRegion {
    /// The resource (or group) whose pipeline uses them.
    pub owner: String,
    /// Whether the owner is a pipeline group.
    pub group: bool,
    /// The number of mbarriers.
    pub count: usize,
    /// Their byte offset in the unified block.
    pub offset: usize,
}

/// The unified shared-memory layout [`SmemAllocator::compute_layout`]
/// makes: data regions first, then the pipelines' barriers (8-byte
/// aligned).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmemLayout {
    /// The data regions, by offset.
    pub regions: Vec<Region>,
    /// The barrier blocks, by offset.
    pub barriers: Vec<BarrierRegion>,
    /// Bytes of data (the end of the last data region).
    pub data_bytes: usize,
    /// Bytes of the whole block.
    pub total_bytes: usize,
    /// The block's alignment (the largest region alignment, at least 8).
    pub alignment: usize,
}

impl SmemLayout {
    /// The data region called `name`.
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }

    /// The barrier block of `owner`.
    pub fn barrier(&self, owner: &str) -> Option<&BarrierRegion> {
        self.barriers.iter().find(|b| b.owner == owner)
    }
}

impl fmt::Display for SmemLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "shared memory: {} bytes ({} data, aligned to {})",
            self.total_bytes, self.data_bytes, self.alignment
        )?;
        for r in &self.regions {
            let alias = match r.alias_group {
                Some(g) => format!("  alias group {g} phase {}", r.phase),
                None => String::new(),
            };
            let owner = r.owner.as_deref().unwrap_or("-");
            writeln!(
                f,
                "  [{:>6}, {:>6}) {:<20} {:<20} align {:>4}{alias}",
                r.offset,
                r.end(),
                r.name,
                owner,
                r.alignment
            )?;
        }
        for b in &self.barriers {
            let what = if b.group { "group" } else { "pipeline" };
            writeln!(
                f,
                "  [{:>6}, {:>6}) {} mbarriers of {what} `{}`",
                b.offset,
                b.offset + b.count * MBARRIER_BYTES,
                b.count,
                b.owner
            )?;
        }
        Ok(())
    }
}

/// The shared-memory allocator of a Task Scheduling kernel (CuTe DSL's
/// task-scheduling `SmemAllocator`): collects the resources' requirements
/// ([`Resource::get_smem_requirements`]) and the barrier storage of every
/// pipeline whose configuration has none, lays them out as one block
/// ([`compute_layout`](Self::compute_layout), on the host), and is handed to
/// [`TaskManager::smem_allocator`](super::TaskManager::smem_allocator),
/// which allocates the block in the kernel, publishes its base in
/// [`ResourceContext::smem_base`](super::ResourceContext) and points each
/// pipeline at its barriers before creating it.
///
/// Allocations whose lifetimes do not overlap may share memory: an alias
/// group ([`add_alias_group`](Self::add_alias_group)) lays each phase out
/// from the same start. The TaskManager's checks then require the
/// dependency graph to order the owners of any two allocations that share
/// bytes by control edges, so that one's use ends before the other's
/// begins.
///
/// ```
/// use cute::task_scheduling::*;
///
/// let a = SmemAllocation::new::<cute::F16>("a", 100, 16); // 200 bytes
/// let b = SmemAllocation::new::<u8>("b", 64, 128);
/// let r = MemoryResource::new("r").smem_requirement(a.clone()).smem_requirement(b.clone());
/// let mut alloc = SmemAllocator::new();
/// alloc.add_resource(&r);
/// let layout = alloc.compute_layout().unwrap();
/// assert_eq!((b.offset(), a.offset()), (0, 64)); // the 128-aligned region first
/// assert_eq!(layout.data_bytes, 264);
/// ```
#[derive(Clone, Debug)]
pub struct SmemAllocator {
    plan: Plan<SmemAllocation>,
    /// (resource, mbarriers) of pipelines without barrier storage.
    resource_barriers: Vec<(String, usize)>,
    /// (group, mbarriers, members)
    group_barriers: Vec<(String, usize, Vec<String>)>,
    capacity: usize,
    layout: Option<SmemLayout>,
}

impl Default for SmemAllocator {
    fn default() -> Self {
        SmemAllocator::new()
    }
}

impl SmemAllocator {
    /// An empty allocator checking against [`DEFAULT_SMEM_CAPACITY`].
    pub fn new() -> SmemAllocator {
        SmemAllocator {
            plan: Plan { allocations: vec![], alias_groups: vec![] },
            resource_barriers: vec![],
            group_barriers: vec![],
            capacity: DEFAULT_SMEM_CAPACITY,
            layout: None,
        }
    }

    /// Checking against `bytes` of shared memory instead.
    pub fn with_capacity(mut self, bytes: usize) -> SmemAllocator {
        self.capacity = bytes;
        self
    }

    /// Register a standalone allocation without an owner.
    pub fn add(&mut self, alloc: &SmemAllocation) {
        self.plan.add(None, alloc.clone());
        self.layout = None;
    }

    /// Register `resource`'s shared memory and, when its pipeline
    /// configuration has no barrier storage, its `2 * num_stages`
    /// mbarriers (unless it belongs to a group registered with
    /// [`add_pipeline_group`](Self::add_pipeline_group), whose barriers
    /// cover it).
    pub fn add_resource(&mut self, resource: &dyn Resource) {
        for a in resource.get_smem_requirements() {
            self.plan.add(Some(resource.name()), a);
        }
        if let Some(cfg) = resource.pipeline_config()
            && cfg.barrier_ptr.is_none()
            && resource.as_group().is_none()
            && !self.resource_barriers.iter().any(|(n, _)| n == resource.name())
        {
            self.resource_barriers.push((resource.name().to_string(), 2 * cfg.num_stages));
        }
        if let Some(g) = resource.as_group() {
            self.add_pipeline_group(g);
        }
        self.layout = None;
    }

    /// Register a pipeline group's shared barriers
    /// ([`PipelineGroup::num_barriers`]).
    pub fn add_pipeline_group(&mut self, group: &PipelineGroup) {
        if group.pipeline_config().is_some_and(|c| c.barrier_ptr.is_some())
            || self.group_barriers.iter().any(|(n, _, _)| n == group.name())
        {
            return;
        }
        self.group_barriers.push((group.name().to_string(), group.num_barriers(), group.member_names().to_vec()));
        self.layout = None;
    }

    /// Let the allocations of each phase share memory with those of the
    /// other phases (CuTe DSL's `add_alias_group`): within a phase they are
    /// laid out one after another, and every phase starts at the group's
    /// start.
    pub fn add_alias_group(&mut self, phases: Vec<Vec<SmemAllocation>>) {
        self.plan.alias_groups.push(phases);
        self.layout = None;
    }

    /// The registered allocations with their owners.
    pub fn allocations(&self) -> impl Iterator<Item = (Option<&str>, &SmemAllocation)> {
        self.plan.allocations.iter().map(|(o, a)| (o.as_deref(), a))
    }

    /// Whether the allocator places `resource`'s barriers (itself, or
    /// through its group).
    pub fn places_barriers_of(&self, resource: &str) -> bool {
        self.resource_barriers.iter().any(|(n, _)| n == resource)
            || self.group_barriers.iter().any(|(n, _, m)| n == resource || m.iter().any(|x| x == resource))
    }

    /// The layout, without assigning the allocations' offsets.
    pub fn plan(&self) -> Result<SmemLayout, AllocError> {
        for (_, a) in &self.plan.allocations {
            check_alignment(a)?;
        }
        for a in self.plan.alias_groups.iter().flatten().flatten() {
            check_alignment(a)?;
        }
        let (placed, data_bytes) = self.plan.layout();
        let mut cursor = if self.resource_barriers.is_empty() && self.group_barriers.is_empty() {
            data_bytes
        } else {
            align_up(data_bytes, MBARRIER_BYTES)
        };
        let mut barriers = vec![];
        let grouped = |n: &str| self.group_barriers.iter().any(|(_, _, m)| m.iter().any(|x| x == n));
        for (owner, count) in &self.resource_barriers {
            if grouped(owner) {
                continue;
            }
            barriers.push(BarrierRegion { owner: owner.clone(), group: false, count: *count, offset: cursor });
            cursor += count * MBARRIER_BYTES;
        }
        for (owner, count, _) in &self.group_barriers {
            barriers.push(BarrierRegion { owner: owner.clone(), group: true, count: *count, offset: cursor });
            cursor += count * MBARRIER_BYTES;
        }
        let mut regions: Vec<Region> = placed.into_iter().map(|(_, r)| r).collect();
        regions.sort_by_key(|r| (r.offset, r.name.clone()));
        let alignment = regions.iter().map(|r| r.alignment).max().unwrap_or(1).max(MBARRIER_BYTES);
        if cursor > self.capacity {
            return Err(AllocError::OverCapacity {
                what: "shared memory",
                needed: cursor,
                capacity: self.capacity,
                unit: "bytes",
            });
        }
        Ok(SmemLayout { regions, barriers, data_bytes, total_bytes: cursor, alignment })
    }

    /// Lay everything out (CuTe DSL's `compute_layout`): set every
    /// allocation's offset and keep the layout for the TaskManager.
    pub fn compute_layout(&mut self) -> Result<&SmemLayout, AllocError> {
        let layout = self.plan()?;
        let (placed, _) = self.plan.layout();
        for (a, r) in placed {
            a.set(r.offset);
        }
        self.layout = Some(layout);
        Ok(self.layout.as_ref().expect("just set"))
    }

    /// The layout of the last [`compute_layout`](Self::compute_layout).
    pub fn layout(&self) -> Option<&SmemLayout> {
        self.layout.as_ref()
    }

    /// Bytes of mbarrier storage it places.
    pub fn barrier_smem_bytes(&self) -> usize {
        self.plan().map(|l| l.barriers.iter().map(|b| b.count * MBARRIER_BYTES).sum()).unwrap_or(0)
    }

    pub(crate) fn alias_groups(&self) -> &[Vec<Vec<SmemAllocation>>] {
        &self.plan.alias_groups
    }

    pub(crate) fn owner_of(&self, a: &SmemAllocation) -> Option<String> {
        self.plan.owner_of(a)
    }
}

fn check_alignment(a: &SmemAllocation) -> Result<(), AllocError> {
    let al = a.alignment;
    if al == 0 || !al.is_power_of_two() || al > MAX_SMEM_ALIGN {
        return Err(AllocError::BadAlignment { name: a.name.clone(), alignment: al });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor memory
// ---------------------------------------------------------------------------

/// The TMEM layout [`TmemAllocator::compute_layout`] makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TmemLayout {
    /// The column regions, by offset.
    pub regions: Vec<Region>,
    /// Columns in use.
    pub used_columns: usize,
    /// Columns the kernel allocates: a power of two of at least 32.
    pub alloc_columns: usize,
}

/// The tensor-memory allocator of a Task Scheduling kernel (CuTe DSL's
/// task-scheduling `TmemAllocator`): collects the resources' column
/// requirements ([`Resource::get_tmem_requirements`]), lays them out on the
/// host (one after another; alias groups share columns), and is handed to
/// [`TaskManager::tmem_allocator`](super::TaskManager::tmem_allocator), which
/// allocates the columns from warp 0 in the prologue over
/// [`TmemAllocator`](crate::memory::TmemAllocator), publishes the address
/// slot in [`ResourceContext::tmem_ptr_i32`](super::ResourceContext), and
/// relinquishes the permit and frees the columns after every task is done.
///
/// ```
/// use cute::task_scheduling::*;
///
/// let acc = TmemAllocation::new("acc", 128);
/// let aux = TmemAllocation::new("aux", 64);
/// let r = MemoryResource::new("r").tmem_requirement(acc.clone()).tmem_requirement(aux.clone());
/// let mut t = TmemAllocator::new();
/// t.add_resource(&r);
/// let l = t.compute_layout().unwrap();
/// assert_eq!((acc.offset(), aux.offset(), l.alloc_columns), (0, 128, 256));
/// ```
#[derive(Clone, Debug)]
pub struct TmemAllocator {
    plan: Plan<TmemAllocation>,
    capacity: usize,
    layout: Option<TmemLayout>,
}

impl Default for TmemAllocator {
    fn default() -> Self {
        TmemAllocator::new()
    }
}

impl TmemAllocator {
    /// An empty allocator for [`TMEM_COLUMNS`] columns.
    pub fn new() -> TmemAllocator {
        TmemAllocator { plan: Plan { allocations: vec![], alias_groups: vec![] }, capacity: TMEM_COLUMNS, layout: None }
    }

    /// With `columns` of TMEM instead (576 on sm_107).
    pub fn with_capacity(mut self, columns: usize) -> TmemAllocator {
        self.capacity = columns;
        self
    }

    /// Register a standalone allocation without an owner.
    pub fn add(&mut self, alloc: &TmemAllocation) {
        self.plan.add(None, alloc.clone());
        self.layout = None;
    }

    /// Register `resource`'s columns.
    pub fn add_resource(&mut self, resource: &dyn Resource) {
        for a in resource.get_tmem_requirements() {
            self.plan.add(Some(resource.name()), a);
        }
        self.layout = None;
    }

    /// Let each phase's allocations share columns with the other phases'.
    pub fn add_alias_group(&mut self, phases: Vec<Vec<TmemAllocation>>) {
        self.plan.alias_groups.push(phases);
        self.layout = None;
    }

    /// The layout, without assigning offsets.
    pub fn plan(&self) -> Result<TmemLayout, AllocError> {
        let (placed, used) = self.plan.layout();
        if used > self.capacity {
            return Err(AllocError::OverCapacity {
                what: "TMEM",
                needed: used,
                capacity: self.capacity,
                unit: "columns",
            });
        }
        let alloc_columns = used.max(32).next_power_of_two();
        let alloc_columns = if alloc_columns > self.capacity { used.div_ceil(32) * 32 } else { alloc_columns };
        let mut regions: Vec<Region> = placed.into_iter().map(|(_, r)| r).collect();
        regions.sort_by_key(|r| (r.offset, r.name.clone()));
        Ok(TmemLayout { regions, used_columns: used, alloc_columns })
    }

    /// Lay everything out and set every allocation's column offset.
    pub fn compute_layout(&mut self) -> Result<&TmemLayout, AllocError> {
        let layout = self.plan()?;
        let (placed, _) = self.plan.layout();
        for (a, r) in placed {
            a.set(r.offset);
        }
        self.layout = Some(layout);
        Ok(self.layout.as_ref().expect("just set"))
    }

    /// The layout of the last [`compute_layout`](Self::compute_layout).
    pub fn layout(&self) -> Option<&TmemLayout> {
        self.layout.as_ref()
    }

    /// The columns in use (after [`plan`](Self::plan)).
    pub fn total_tmem_columns(&self) -> usize {
        self.plan.layout().1
    }

    /// The capacity in columns.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

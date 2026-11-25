// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.task_manager (TaskManager,
// DependencyGraphEdge, DependencyGraphControlEdge) and task.Task's
// execution (NVIDIA EULA); the code and the messages are written
// independently.

//! The TaskManager: owns the resources, checks the tasks and the
//! dependency graph, initialises the pipelines and emits the
//! warp-specialised kernel.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::panic::Location;

use crate::arch::{block_idx, setmaxregister_decrease, setmaxregister_increase, warp_idx};
use crate::builder::{self, Kernel};
use crate::control::{Var, for_range_with, if_, while_carry};
use crate::ir::{Loc, LoopOptions};
use crate::pipeline::{Agent, PipelineState, agent_sync, pipeline_init_arrive};
use crate::profiling::{Scope, WarpScope};
use crate::sim::{KernelArg, LaunchConfig, SimError, SimOptions, SimReport};
use crate::utils::WorkTileInfo;
use crate::value::{Boolean, Int32};

use super::enums::{Guard, IterationPredicate, PipelineType, ScheduleStage};
use super::memory::{SmemAllocator, TmemAllocator};
use super::resource::{Resource, ResourceContext, Side, StageInfo, TaskLocals, VarRef, WorkArgs};
use super::schedule::{Bound, Node, Source, Step};
use super::task::{RegisterBudget, Task};

// ---------------------------------------------------------------------------
// The dependency graph
// ---------------------------------------------------------------------------

/// An upstream entry of the resource dependency graph (CuTe DSL's
/// `DependencyGraphEdge`): the downstream resource's producer work reads
/// variables that consumer work on the upstream resource returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyGraphEdge {
    upstream: String,
    control: bool,
}

impl DependencyGraphEdge {
    /// A data edge from `upstream`.
    pub fn new(upstream: &(impl Resource + ?Sized)) -> DependencyGraphEdge {
        DependencyGraphEdge { upstream: upstream.name().to_string(), control: false }
    }

    /// The upstream resource's name.
    pub fn upstream(&self) -> &str {
        &self.upstream
    }

    /// Whether the edge only orders (no variables flow along it).
    pub fn is_control(&self) -> bool {
        self.control
    }
}

/// An ordering-only upstream entry (CuTe DSL's `DependencyGraphControlEdge`):
/// the downstream resource must come after the upstream one, but no
/// variables flow. A [`PdlWaitBarrier`](super::PdlWaitBarrier) upstream is
/// ordering-only whichever edge names it.
#[derive(Clone, Copy, Debug)]
pub struct DependencyGraphControlEdge;

impl DependencyGraphControlEdge {
    /// A control edge from `upstream`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(upstream: &(impl Resource + ?Sized)) -> DependencyGraphEdge {
        DependencyGraphEdge { upstream: upstream.name().to_string(), control: true }
    }
}

impl<T: Resource + ?Sized> From<&T> for DependencyGraphEdge {
    fn from(r: &T) -> DependencyGraphEdge {
        DependencyGraphEdge::new(r)
    }
}

/// CuTe DSL's `resource_dependency_graph = {resource: [upstream, ...]}`:
/// for each downstream resource, the resources whose consumer work feeds
/// its producer work (data edges) or that it must follow (control edges).
///
/// ```
/// use cute::task_scheduling::*;
///
/// let input = MemoryResource::new("input");
/// let output = MemoryResource::new("output");
/// let pdl = PdlWaitBarrier::new("pdl_wait");
/// let graph = ResourceDependencyGraph::new()
///     .depends_on(&output, [DependencyGraphEdge::new(&input), DependencyGraphControlEdge::new(&pdl)]);
/// assert_eq!(graph.upstreams("output").count(), 2);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ResourceDependencyGraph {
    nodes: Vec<(String, Vec<DependencyGraphEdge>)>,
}

impl ResourceDependencyGraph {
    /// An empty graph.
    pub fn new() -> ResourceDependencyGraph {
        ResourceDependencyGraph::default()
    }

    /// With `downstream` depending on `upstreams` (added to any entries it
    /// already has).
    pub fn depends_on<E: Into<DependencyGraphEdge>>(
        mut self,
        downstream: &(impl Resource + ?Sized),
        upstreams: impl IntoIterator<Item = E>,
    ) -> ResourceDependencyGraph {
        let edges: Vec<DependencyGraphEdge> = upstreams.into_iter().map(Into::into).collect();
        match self.nodes.iter_mut().find(|(d, _)| d == downstream.name()) {
            Some((_, e)) => e.extend(edges),
            None => self.nodes.push((downstream.name().to_string(), edges)),
        }
        self
    }

    /// The downstream resources, in insertion order.
    pub fn downstreams(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|(d, _)| d.as_str())
    }

    /// The upstream entries of `downstream`.
    pub fn upstreams<'a>(&'a self, downstream: &str) -> impl Iterator<Item = &'a DependencyGraphEdge> + 'a {
        let d = downstream.to_string();
        self.nodes.iter().filter(move |(n, _)| *n == d).flat_map(|(_, e)| e.iter())
    }

    /// Every resource the graph mentions.
    pub fn resources(&self) -> HashSet<&str> {
        let mut out = HashSet::new();
        for (d, e) in &self.nodes {
            out.insert(d.as_str());
            out.extend(e.iter().map(|e| e.upstream.as_str()));
        }
        out
    }

    pub(crate) fn nodes(&self) -> &[(String, Vec<DependencyGraphEdge>)] {
        &self.nodes
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The kind of a problem the TaskManager's checks find.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerifyIssueKind {
    /// A task's name, warp range or register budget is invalid, or two
    /// tasks share a warp.
    BadTask,
    /// A resource a task names is not given to the manager (so its
    /// barriers would never be initialised), or `run` came before
    /// `setup_resources_and_tasks`.
    BarrierInit,
    /// A schedule step on a resource the task does not declare in the
    /// right role (producer-side steps need it in `dst_resources`,
    /// consumer-side ones in `src_resources`), or a declared resource the
    /// schedule never uses.
    RoleMismatch,
    /// The dependency graph names an unknown resource, has an edge no task
    /// backs, leaves out a task's resource, or makes a launch barrier
    /// downstream.
    BadGraph,
    /// The dependency graph has a cycle.
    CyclicGraph,
    /// A pipelined resource without exactly one producing task, or without
    /// a consuming one.
    PipelineSides,
    /// The commits of a pipelined resource and its releases sit in loop
    /// nests or guards that do not run the same number of times.
    UnbalancedPipeline,
    /// A pipeline's cooperative group does not count the threads of the
    /// tasks on that side.
    ThreadCount,
    /// A variable read by a step on another resource without a data edge
    /// between the two, or a variable without a default.
    VariableFlow,
    /// A task produces (or reads directly) a resource that depends on the
    /// previous grid before it waits for it.
    PdlOrdering,
    /// A `launch_griddep` that can run before any `wait_griddep`.
    PdlLaunchBeforeWait,
    /// A run-time loop bound or condition the schedule names has no value.
    UnboundValue,
    /// Two allocations share memory but the graph does not order their
    /// owners.
    SmemAlias,
    /// The shared-memory layout cannot be made (alignment, capacity).
    SmemLayout,
    /// More TMEM columns than the CTA has.
    TmemCapacity,
    /// A pipelined resource or group with no barrier storage.
    BarrierStorage,
    /// A pipeline group whose members do not fit together, or that is used
    /// wrongly.
    BadGroup,
    /// A group's shared barrier counts other arrivals than its members
    /// make.
    GroupArrivalCount,
}

/// One problem found by the TaskManager's checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyIssue {
    pub kind: VerifyIssueKind,
    pub message: String,
}

/// Every problem the TaskManager's checks found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyError {
    pub issues: Vec<VerifyIssue>,
}

impl VerifyError {
    /// Whether some issue is of `kind`.
    pub fn has(&self, kind: VerifyIssueKind) -> bool {
        self.issues.iter().any(|i| i.kind == kind)
    }

    pub(crate) fn single(kind: VerifyIssueKind, message: String) -> VerifyError {
        VerifyError { issues: vec![VerifyIssue { kind, message }] }
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the task schedule is invalid:")?;
        for i in &self.issues {
            write!(f, "\n  - {}", i.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for VerifyError {}

/// What went wrong in [`TaskManager::verify_interleavings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterleavingErrorKind {
    /// Every warp that has not exited is blocked.
    Deadlock,
    /// Unordered accesses to the same memory.
    Race,
    /// The runs under different interleavings gave different results.
    OrderDependent,
    /// Another simulator error.
    Other,
}

/// A failure of [`TaskManager::verify_interleavings`], attributed to the
/// task and schedule step it happened in (from the profiling scopes the
/// TaskManager opens) and the schedule line that recorded the step.
#[derive(Clone, Debug)]
pub struct InterleavingError {
    pub kind: InterleavingErrorKind,
    /// The task the failing warp ran.
    pub task: Option<String>,
    /// The schedule step, as `#id resource.operation`.
    pub step: Option<String>,
    /// The source line (the schedule step for a pipeline operation, the
    /// work body for an access in work).
    pub loc: Option<Loc>,
    /// The seed of the failing random interleaving (it replays the run).
    pub seed: Option<u64>,
    /// The simulator's report.
    pub message: String,
}

impl InterleavingError {
    fn from_sim(e: SimError) -> InterleavingError {
        let msg = e.message.clone();
        let kind = if msg.contains("deadlock") {
            InterleavingErrorKind::Deadlock
        } else if msg.contains("race") {
            InterleavingErrorKind::Race
        } else if msg.contains(" under ") {
            InterleavingErrorKind::OrderDependent
        } else {
            InterleavingErrorKind::Other
        };
        let path = msg.split("(in profiling scope `").nth(1).and_then(|r| r.split('`').next()).unwrap_or("");
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let (task, step) = match parts.iter().position(|p| p.starts_with('#')) {
            Some(i) => (i.checked_sub(1).map(|j| parts[j].to_string()), Some(parts[i].to_string())),
            None => (parts.last().map(|p| p.to_string()), None),
        };
        InterleavingError { kind, task, step, loc: e.loc, seed: e.seed, message: e.to_string() }
    }
}

impl fmt::Display for InterleavingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind)?;
        if let Some(t) = &self.task {
            write!(f, " in task {t}")?;
        }
        if let Some(s) = &self.step {
            write!(f, " at step {s}")?;
        }
        if let Some(seed) = self.seed {
            write!(f, " (seed {seed})")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for InterleavingError {}

// ---------------------------------------------------------------------------
// The manager
// ---------------------------------------------------------------------------

/// Runs tasks as one warp-specialised kernel (CuTe DSL's
/// `TaskManager(tasks, resource_dependency_graph, iket_enable_profiling)`).
///
/// The manager owns the resources ([`resource`](Self::resource)); tasks
/// and schedules refer to them by name. Inside a kernel,
/// [`setup_resources_and_tasks`](Self::setup_resources_and_tasks) checks
/// everything ([`verify`](Self::verify), which also runs without a kernel)
/// and creates every resource, publishing all pipeline barriers with one
/// fence and block (or cluster) barrier; [`run`](Self::run) then emits, for
/// each task, a branch on the warp index that
///
/// * opens a [`WarpScope`] named after the task (an IKET range with
///   `iket_enable_profiling`) and applies its register budget;
/// * makes storage ([`TaskLocals`]) for every variable its steps produce or
///   read, starting at the defaults, and a pipeline state per pipelined
///   resource and side it uses (kept in [`Var`]s, so they are carried
///   through the loops);
/// * emits the schedule's nodes in order: a pipeline operation calls the
///   resource's method (attributed to the schedule line that recorded the
///   step, so a deadlock is reported there); a work call builds the
///   [`StageInfo`] (the domain index and bounds, the pipeline stage for
///   stage work, the work tile), routes its inputs from the variables and
///   stores its results; a domain loop is a `for_range` (with its unroll
///   factor) and iteration guards test the index; a work-tile loop seeds
///   the queue's tile (`init_work_tile`) and is a `while_carry` over it,
///   with the skippable regions under the queue's skip predicate; every
///   step sits in a profiling scope `#id resource.operation` (an IKET range
///   with profiling on);
/// * ends with `producer_tail` for every resource it acquired.
///
/// Warps no task covers do nothing (give them a padding task to set their
/// register budget).
pub struct TaskManager {
    tasks: Vec<Task>,
    graph: ResourceDependencyGraph,
    resources: Vec<Box<dyn Resource>>,
    iket: bool,
    bounds: HashMap<String, Int32>,
    conditions: HashMap<String, Boolean>,
    set_up: bool,
    smem: Option<SmemAllocator>,
    tmem: Option<TmemAllocator>,
    context: ResourceContext,
    tmem_runtime: RefCell<Option<crate::memory::TmemAllocator>>,
}

impl fmt::Debug for TaskManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskManager")
            .field("tasks", &self.tasks.iter().map(|t| t.name()).collect::<Vec<_>>())
            .field("resources", &self.resources.iter().map(|r| r.name()).collect::<Vec<_>>())
            .field("iket", &self.iket)
            .finish()
    }
}

impl TaskManager {
    /// A manager for `tasks` over `resource_dependency_graph`, without
    /// resources yet.
    pub fn new(tasks: Vec<Task>, resource_dependency_graph: ResourceDependencyGraph) -> TaskManager {
        TaskManager {
            tasks,
            graph: resource_dependency_graph,
            resources: vec![],
            iket: false,
            bounds: HashMap::new(),
            conditions: HashMap::new(),
            set_up: false,
            smem: None,
            tmem: None,
            context: ResourceContext::default(),
            tmem_runtime: RefCell::new(None),
        }
    }

    /// With the declarative shared-memory allocator (CuTe DSL's
    /// `smem_allocator`): the prologue allocates its block, publishes the
    /// base in every work call's [`ResourceContext`] and places the
    /// barriers of the pipelines and groups it registered. Its layout is
    /// computed then if [`SmemAllocator::compute_layout`] was not called.
    pub fn smem_allocator(mut self, allocator: SmemAllocator) -> TaskManager {
        self.smem = Some(allocator);
        self
    }

    /// With the declarative TMEM allocator: the prologue allocates its
    /// columns from warp 0 (publishing the address slot in the
    /// [`ResourceContext`]) and [`run`](Self::run) frees them after every
    /// task.
    pub fn tmem_allocator(mut self, allocator: TmemAllocator) -> TaskManager {
        self.tmem = Some(allocator);
        self
    }

    /// The shared-memory allocator, if any.
    pub fn smem_allocator_ref(&self) -> Option<&SmemAllocator> {
        self.smem.as_ref()
    }

    /// The TMEM allocator, if any.
    pub fn tmem_allocator_ref(&self) -> Option<&TmemAllocator> {
        self.tmem.as_ref()
    }

    /// The framework context work calls see (after setup).
    pub fn context(&self) -> &ResourceContext {
        &self.context
    }

    /// Owning the resource `r`.
    pub fn resource(mut self, r: impl Resource + 'static) -> TaskManager {
        self.resources.push(Box::new(r));
        self
    }

    /// With IKET ranges around each task and step (CuTe DSL's
    /// `iket_enable_profiling`); the kernel must be built with
    /// [`KernelOptions::iket`](crate::KernelOptions::iket) for them to
    /// record anything.
    pub fn iket_enable_profiling(mut self, on: bool) -> TaskManager {
        self.iket = on;
        self
    }

    /// With the run-time value of the loop bound a schedule named with
    /// [`Bound::symbol`].
    pub fn bind(mut self, name: &str, value: Int32) -> TaskManager {
        self.bounds.insert(name.to_string(), value);
        self
    }

    /// With the run-time value of an opaque `when_true` / `when_false`
    /// condition.
    pub fn bind_condition(mut self, key: &str, value: Boolean) -> TaskManager {
        self.conditions.insert(key.to_string(), value);
        self
    }

    /// The tasks.
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// The dependency graph.
    pub fn graph(&self) -> &ResourceDependencyGraph {
        &self.graph
    }

    /// The resource called `name`.
    pub fn find_resource(&self, name: &str) -> Option<&dyn Resource> {
        self.resources.iter().find(|r| r.name() == name).map(|r| r.as_ref())
    }

    pub(crate) fn resource_list(&self) -> impl Iterator<Item = &dyn Resource> {
        self.resources.iter().map(|r| r.as_ref())
    }

    #[track_caller]
    fn res(&self, name: &str) -> &dyn Resource {
        self.find_resource(name)
            .unwrap_or_else(|| panic!("cute: task_scheduling: the TaskManager has no resource `{name}`"))
    }

    /// Check everything, then create every resource and publish the
    /// pipelines' barriers (in a kernel; every thread of the block must
    /// reach it). Call once, before [`run`](Self::run).
    #[track_caller]
    pub fn setup_resources_and_tasks(&mut self) -> Result<(), VerifyError> {
        assert!(!self.set_up, "cute: task_scheduling: setup_resources_and_tasks may only be called once");
        self.verify()?;
        let pipelined = self.resources.iter().any(|r| r.pipeline_config().is_some());
        let cluster =
            self.resources.iter().filter_map(|r| r.pipeline_config()?.cta_layout_vmnk.as_ref()).any(|l| l.size() != 1);
        self.materialise_memory()?;
        let hints = self.pipeline_hints();
        let hint = |n: &str| hints.get(n).copied().unwrap_or((None, false));
        let grouped: HashSet<String> =
            self.resources.iter().filter_map(|r| r.as_group()).flat_map(|g| g.member_names().to_vec()).collect();
        for r in &mut self.resources {
            if r.as_group().is_some() || grouped.contains(r.name()) {
                continue;
            }
            let (leader, all) = hint(r.name());
            r.memory_mut().set_pipeline_hints(leader, all);
            r.create_with(true);
        }
        for gi in 0..self.resources.len() {
            let Some(g) = self.resources[gi].as_group_mut() else { continue };
            for (member, p) in g.build(&hint) {
                let r = self.resources.iter_mut().find(|r| r.name() == member).expect("checked by verify");
                r.memory_mut().create_with_pipeline(Some(p));
            }
        }
        if pipelined {
            pipeline_init_arrive(None, false);
            if cluster {
                agent_sync(Agent::ThreadBlockCluster, true);
            } else {
                agent_sync(Agent::ThreadBlock, false);
            }
        } else if self.tmem.is_some() {
            agent_sync(Agent::ThreadBlock, false);
        }
        self.set_up = true;
        Ok(())
    }

    /// Allocate the unified shared-memory block and the TMEM columns, and
    /// point the pipelines at their barriers.
    #[track_caller]
    fn materialise_memory(&mut self) -> Result<(), VerifyError> {
        if let Some(alloc) = self.smem.as_mut() {
            if alloc.layout().is_none() {
                alloc.compute_layout().map_err(|e| VerifyError::single(VerifyIssueKind::SmemLayout, e.to_string()))?;
            }
            let layout = alloc.layout().expect("computed").clone();
            if layout.total_bytes > 0 {
                let align = layout.alignment.min(crate::memory::MAX_SMEM_ALIGN);
                let base = crate::memory::SmemAllocator::new().allocate_bytes(
                    layout.total_bytes,
                    align,
                    crate::ir::SmemPartition::User,
                );
                for b in &layout.barriers {
                    let ptr = base.offset(b.offset as i32).cast::<u64>();
                    if let Some(r) = self.resources.iter_mut().find(|r| r.name() == b.owner) {
                        r.memory_mut().set_barrier_ptr(ptr);
                    }
                }
                self.context.smem_base = Some(base);
            }
        }
        if let Some(alloc) = self.tmem.as_mut() {
            if alloc.layout().is_none() {
                alloc
                    .compute_layout()
                    .map_err(|e| VerifyError::single(VerifyIssueKind::TmemCapacity, e.to_string()))?;
            }
            let cols = alloc.layout().expect("computed").alloc_columns;
            let options = crate::memory::TmemAllocatorOptions { arch: tmem_arch(), ..Default::default() };
            let mut t = crate::memory::TmemAllocator::new(crate::pipeline::NamedBarrier::new(1, 32), options);
            t.allocate(cols);
            self.context.tmem_ptr_i32 = Some(t.alloc_result_dst_smem_ptr().clone());
            *self.tmem_runtime.borrow_mut() = Some(t);
        }
        Ok(())
    }

    /// For each pipelined resource (and group): the first warp of the task
    /// producing it, and whether every consumer thread releases a
    /// `TmaAsync` stage (its consumer group counts the consuming tasks'
    /// threads rather than their warps).
    pub(crate) fn pipeline_hints(&self) -> HashMap<String, (Option<u32>, bool)> {
        let mut out = HashMap::new();
        for r in &self.resources {
            let Some(cfg) = r.pipeline_config() else { continue };
            let name = r.name();
            let group = r.as_group();
            let on = |s: &Step| s.resource == name || group.is_some_and(|g| g.has_member(&s.resource));
            let (mut leader, mut consumer_threads, mut consumer_warps) = (None, 0u32, 0u32);
            for t in &self.tasks {
                let (mut p, mut c) = (false, false);
                walk_steps(&t.schedule().body, &mut |s, _| {
                    if s.stage.op_name().is_some() && on(s) {
                        match side_of(s.stage) {
                            Side::Producer => p = true,
                            Side::Consumer => c = true,
                        }
                    }
                });
                if p && leader.is_none() {
                    leader = Some(t.warp_idx());
                }
                if c {
                    consumer_threads += t.num_threads();
                    consumer_warps += t.num_warps();
                }
            }
            let count = match cfg.consumer_group.agent() {
                Agent::Thread | Agent::Warp => cfg.consumer_group.thread_arrive_count().as_static(),
                _ => None,
            };
            let tma = cfg.pipeline_type == PipelineType::TmaAsync
                || group.is_some_and(|g| {
                    g.member_names().iter().all(|m| {
                        self.find_resource(m)
                            .and_then(|r| r.pipeline_config())
                            .is_some_and(|c| c.pipeline_type == PipelineType::TmaAsync)
                    })
                });
            let all = tma && consumer_threads != consumer_warps && count == Some(consumer_threads as i64);
            out.insert(name.to_string(), (leader, all));
        }
        out
    }

    /// Emit every task (in a kernel, after
    /// [`setup_resources_and_tasks`](Self::setup_resources_and_tasks)).
    #[track_caller]
    pub fn run(&self) -> Result<(), VerifyError> {
        if !self.set_up {
            return Err(VerifyError::single(
                VerifyIssueKind::BarrierInit,
                "run() before setup_resources_and_tasks(): the pipelines' barriers would be waited on before they \
                 are initialised"
                    .into(),
            ));
        }
        self.check_bindings()?;
        let w = warp_idx();
        // A budget shared by tasks smaller than a warpgroup is set once per
        // warpgroup, before the dispatch, so that its four warps execute
        // the same `setmaxnreg` (the instruction is `.sync.aligned`).
        for (group, budget) in shared_budgets(&self.tasks) {
            let w0 = 4 * group as i32;
            if_(w.ge(w0) & w.lt(w0 + 4), || set_budget(budget));
        }
        for task in &self.tasks {
            let (w0, w1) = (task.warp_idx() as i32, (task.warp_idx() + task.num_warps()) as i32);
            if_(w.ge(w0) & w.lt(w1), || self.emit_task(task));
        }
        if let Some(t) = self.tmem_runtime.borrow_mut().as_mut() {
            // Read the address while the columns are live; every task is
            // done with them before warp 0 frees them.
            let base = t.retrieve_ptr::<f32>();
            agent_sync(Agent::ThreadBlock, false);
            t.relinquish_alloc_permit();
            t.free(&base, 0);
        }
        Ok(())
    }

    fn check_bindings(&self) -> Result<(), VerifyError> {
        let mut issues = vec![];
        for task in &self.tasks {
            walk_nodes(&task.schedule().body, &mut vec![], &mut |n, _| match n {
                Node::DomainLoop(l) => {
                    for b in [&l.start, &l.end, &l.step] {
                        if let Bound::Runtime { name, value: None } = b
                            && !self.bounds.contains_key(name)
                        {
                            issues.push(VerifyIssue {
                                kind: VerifyIssueKind::UnboundValue,
                                message: format!(
                                    "task {}: the loop bound `{name}` has no value; capture the schedule with \
                                     Bound::runtime or bind it",
                                    task.name()
                                ),
                            });
                        }
                    }
                }
                Node::Block(b) => {
                    if let (Guard::Opaque { key, .. }, None) = (&b.guard, &b.condition)
                        && !self.conditions.contains_key(key)
                    {
                        issues.push(VerifyIssue {
                            kind: VerifyIssueKind::UnboundValue,
                            message: format!("task {}: the condition `{key}` has no value; bind it", task.name()),
                        });
                    }
                }
                _ => {}
            });
        }
        if issues.is_empty() { Ok(()) } else { Err(VerifyError { issues }) }
    }

    /// Run `kernel` (built with a TaskManager) on the simulator under every
    /// built-in interleaving and a seeded random one per seed
    /// ([`Kernel::simulate_schedules`]), and report a deadlock, a race or
    /// results that depend on the interleaving with the task, the schedule
    /// step and the source line it happened at.
    pub fn verify_interleavings(
        kernel: &Kernel,
        cfg: LaunchConfig,
        args: &mut [KernelArg],
        seeds: &[u64],
    ) -> Result<SimReport, InterleavingError> {
        kernel.simulate_schedules(cfg, args, &SimOptions::default(), seeds).map_err(InterleavingError::from_sim)
    }

    // -----------------------------------------------------------------------
    // Emission
    // -----------------------------------------------------------------------

    #[track_caller]
    fn emit_task(&self, task: &Task) {
        let last = task.warp_idx() + task.num_warps() - 1;
        let scope = WarpScope::with(task.name(), task.warp_idx(), last, self.iket);
        if let Some(budget) = task.registers()
            && whole_warpgroups(task)
        {
            set_budget(budget);
        }
        let em = Emitter::new(self, task);
        let bi = block_idx();
        let mut ctx = Ctx {
            tile: WorkTileInfo::new([bi.x, bi.y, bi.z], Boolean::constant(true)),
            domain: None,
            bounds: None,
            skip: None,
        };
        if let Some(wq) = em.work_queue {
            let r = wq.memory().run_work("init_work_tile", &StageInfo::new(0), &WorkArgs::new());
            em.locals.store(&r);
            ctx.tile = em.locals.get(wq.work_tile());
        }
        em.nodes(&task.schedule().body, &ctx);
        em.producer_tails();
        scope.close();
    }
}

fn set_budget(budget: RegisterBudget) {
    match budget {
        RegisterBudget::Increase(n) => setmaxregister_increase(n),
        RegisterBudget::Decrease(n) => setmaxregister_decrease(n),
    }
}

/// Whether the task's warps are whole, aligned warpgroups.
fn whole_warpgroups(task: &Task) -> bool {
    task.warp_idx().is_multiple_of(4) && task.num_warps().is_multiple_of(4)
}

/// Whether every warpgroup `task` touches is covered by tasks with its
/// register budget (CuTe DSL's rule for budgets on tasks smaller than a
/// warpgroup).
pub(crate) fn shares_budget(tasks: &[Task], task: &Task) -> bool {
    let Some(budget) = task.registers() else { return true };
    let (first, last) = (task.warp_idx() / 4, (task.warp_idx() + task.num_warps() - 1) / 4);
    (first..=last).all(|g| {
        (4 * g..4 * g + 4).all(|w| {
            tasks.iter().any(|t| w >= t.warp_idx() && w < t.warp_idx() + t.num_warps() && t.registers() == Some(budget))
        })
    })
}

/// The warpgroups whose register budget is shared by tasks smaller than a
/// warpgroup, with that budget.
fn shared_budgets(tasks: &[Task]) -> Vec<(u32, RegisterBudget)> {
    let mut out: Vec<(u32, RegisterBudget)> = vec![];
    for t in tasks.iter().filter(|t| t.num_warps() > 0 && !whole_warpgroups(t)) {
        let Some(budget) = t.registers() else { continue };
        for g in t.warp_idx() / 4..=(t.warp_idx() + t.num_warps() - 1) / 4 {
            if !out.iter().any(|&(o, _)| o == g) {
                out.push((g, budget));
            }
        }
    }
    out.sort_by_key(|&(g, _)| g);
    out
}

/// The target TMEM is allocated for: the kernel's, or sm_100a.
#[track_caller]
fn tmem_arch() -> crate::Arch {
    crate::target_arch().unwrap_or(crate::Arch::Sm100a)
}

/// Visit every node with its ancestors.
pub(crate) fn walk_nodes<'a>(nodes: &'a [Node], path: &mut Vec<&'a Node>, f: &mut dyn FnMut(&'a Node, &[&'a Node])) {
    for n in nodes {
        f(n, path);
        path.push(n);
        walk_nodes(n.body(), path, f);
        path.pop();
    }
}

/// Visit every step with its ancestors.
pub(crate) fn walk_steps<'a>(nodes: &'a [Node], f: &mut dyn FnMut(&'a Step, &[&'a Node])) {
    walk_nodes(nodes, &mut vec![], &mut |n, path| {
        if let Node::Step(s) = n {
            f(s, path);
        }
    });
}

/// The side a step works on.
pub(crate) fn side_of(stage: ScheduleStage) -> Side {
    use ScheduleStage as S;
    match stage {
        S::ProducerTryAcquire | S::ProducerAcquire | S::ProducerWork | S::ProducerAuxWork | S::ProducerCommit => {
            Side::Producer
        }
        _ => Side::Consumer,
    }
}

/// Where the steps being emitted run.
#[derive(Clone)]
struct Ctx {
    tile: WorkTileInfo,
    /// Inside a domain loop: (index, start, end, step).
    domain: Option<(Int32, Int32, Int32, Int32)>,
    /// The bounds of the domain loop of the body being emitted.
    bounds: Option<(Int32, Int32, Int32)>,
    /// The skip predicate of the current work tile.
    skip: Option<Boolean>,
}

struct Emitter<'a> {
    m: &'a TaskManager,
    task: &'a Task,
    outputs: HashMap<(usize, String), VarRef>,
    locals: TaskLocals,
    states: HashMap<(String, Side), Var<PipelineState>>,
    work_queue: Option<&'a super::WorkQueue>,
}

impl<'a> Emitter<'a> {
    #[track_caller]
    fn new(m: &'a TaskManager, task: &'a Task) -> Emitter<'a> {
        let sched = task.schedule();
        let mut vars: Vec<VarRef> = vec![];
        let mut add = |v: &VarRef| {
            if !vars.contains(v) {
                vars.push(v.clone());
            }
        };
        let mut outputs = HashMap::new();
        let mut states = HashMap::new();
        let mut work_queue = None;
        walk_nodes(&sched.body, &mut vec![], &mut |n, _| match n {
            Node::Step(s) => {
                for v in &s.outputs {
                    add(v);
                    outputs.insert((s.id, v.name().to_string()), v.clone());
                }
                let r = m.res(&s.resource);
                for route in &s.inputs {
                    if let Source::Peer { resource, variable } = &route.source {
                        let owner = m.res(resource);
                        if let Some(v) = owner.memory().variables().find(|v| v.name() == variable) {
                            add(v);
                        }
                    }
                }
                let side = side_of(s.stage);
                if r.pipeline_config().is_some() {
                    states.entry((s.resource.clone(), side)).or_insert_with(|| {
                        Var::new(match side {
                            Side::Producer => r.memory().producer_state(),
                            Side::Consumer => r.memory().consumer_state(),
                        })
                    });
                }
            }
            Node::Block(b) => {
                if let Some(v) = &b.condition {
                    add(v);
                }
            }
            Node::WorkTileLoop(l) => {
                let wq = m.res(&l.work_queue).as_work_queue().expect("checked by verify");
                add(wq.work_tile().var_ref());
                add(wq.skip_work_tile().var_ref());
                work_queue = Some(wq);
            }
            Node::DomainLoop(_) => {}
        });
        let mut locals = TaskLocals::new();
        for v in &vars {
            assert!(locals.declare_ref(v), "cute: task_scheduling: `{}` has no default (checked by verify)", v.name());
        }
        Emitter { m, task, outputs, locals, states, work_queue }
    }

    fn bound(&self, b: &Bound) -> Int32 {
        match b {
            Bound::Const(c) => Int32::constant(*c as i32),
            Bound::Runtime { value: Some(v), .. } => *v,
            Bound::Runtime { name, value: None } => *self.m.bounds.get(name).expect("checked by run"),
        }
    }

    /// The bounds of the first domain loop in `nodes`.
    fn body_bounds(&self, nodes: &[Node]) -> Option<(Int32, Int32, Int32)> {
        nodes.iter().find_map(|n| match n {
            Node::DomainLoop(l) => Some((self.bound(&l.start), self.bound(&l.end), self.bound(&l.step))),
            Node::Block(b) => self.body_bounds(&b.body),
            _ => None,
        })
    }

    #[track_caller]
    fn nodes(&self, nodes: &[Node], ctx: &Ctx) {
        let ctx = match (ctx.domain.is_none(), self.body_bounds(nodes)) {
            (true, Some(b)) => Ctx { bounds: Some(b), ..ctx.clone() },
            _ => ctx.clone(),
        };
        for n in nodes {
            match n {
                Node::Step(s) => self.step(s, &ctx),
                Node::Block(b) => {
                    let cond = match &b.guard {
                        Guard::Iteration(p) => {
                            let d = ctx.domain.expect("an iteration guard is inside a domain loop");
                            fires(*p, d)
                        }
                        Guard::Opaque { key, negated } => {
                            let c = match &b.condition {
                                Some(v) => self.locals.condition_ref(v).expect("a scalar guard variable"),
                                None => *self.m.conditions.get(key).expect("checked by run"),
                            };
                            if *negated { !c } else { c }
                        }
                        Guard::Skippable => match ctx.skip {
                            Some(s) => !s,
                            None => Boolean::constant(true),
                        },
                    };
                    if_(cond, || self.nodes(&b.body, &ctx));
                }
                Node::DomainLoop(l) => {
                    let (start, end, step) = (self.bound(&l.start), self.bound(&l.end), self.bound(&l.step));
                    let opts = if l.unroll > 1 { LoopOptions::unroll(l.unroll) } else { LoopOptions::default() };
                    for_range_with(start, end, step, opts, |i| {
                        let c = Ctx {
                            domain: Some((i, start, end, step)),
                            bounds: Some((start, end, step)),
                            ..ctx.clone()
                        };
                        self.nodes(&l.body, &c);
                    });
                }
                Node::WorkTileLoop(l) => {
                    let wq = self.work_queue.expect("a work queue");
                    while_carry(
                        (),
                        |()| self.locals.get(wq.work_tile()).is_valid_tile(),
                        |()| {
                            let tile = self.locals.get(wq.work_tile());
                            let skip = l.skip_if.then(|| {
                                let s = wq.skip_work_tile_if(&tile);
                                self.locals.set(wq.skip_work_tile(), s);
                                s
                            });
                            let c = Ctx { tile, skip, ..ctx.clone() };
                            self.nodes(&l.body, &c);
                        },
                    );
                }
            }
        }
    }

    fn info(&self, ctx: &Ctx) -> StageInfo {
        let si = match (ctx.domain, ctx.bounds) {
            (Some((i, s, e, st)), _) => StageInfo::new(i).with_bounds(s, e, st),
            (None, Some((s, e, st))) => StageInfo::new(s).with_bounds(s, e, st),
            (None, None) => StageInfo::new(0),
        };
        si.with_work_tile(ctx.tile).with_context(self.m.context.clone())
    }

    #[track_caller]
    fn step(&self, s: &Step, ctx: &Ctx) {
        let r = self.m.res(&s.resource);
        let mem = r.memory();
        let what = s.label.as_deref().or(s.stage.op_name()).unwrap_or("?");
        let scope = Scope::with(&[&format!("#{} {}.{what}", s.id, s.resource)], self.m.iket, &[]);
        let side = side_of(s.stage);
        let state = self.states.get(&(s.resource.clone(), side));
        use ScheduleStage as St;
        if let Some(g) = r.as_group() {
            self.group_close(g, s, side);
        } else if s.stage.is_work() {
            let mut info = self.info(ctx).with_label(what);
            if let Some(st) = state {
                info = match s.stage {
                    St::ProducerWork => mem.producer_stage(&st.get(), info),
                    St::ConsumerWork => mem.consumer_stage(&st.get(), info),
                    _ => info,
                };
            }
            let mut args = WorkArgs::new();
            for route in &s.inputs {
                let var = match &route.source {
                    Source::Step { step, variable } => self.outputs.get(&(*step, variable.clone())).cloned(),
                    Source::Peer { resource, variable } => {
                        self.m.res(resource).memory().variables().find(|v| v.name() == variable).cloned()
                    }
                    Source::Constexpr(v) => {
                        args = args.with_constexpr(&route.argument, *v);
                        continue;
                    }
                };
                let var = var.expect("a routed variable has storage");
                args = args.with_raw(&route.argument, self.locals.load_ref(&var));
            }
            let results = mem.run_work(what, &info, &args);
            self.locals.store(&results);
        } else if let Some(st) = state {
            let set = s.loc.map(|loc| builder::with(|b| b.set_location(loc))).unwrap_or(false);
            match s.stage {
                St::ProducerTryAcquire => {
                    mem.try_acquire(&st.get());
                }
                St::ProducerAcquire => mem.acquire(&st.get()),
                St::ProducerCommit => {
                    let mut x = st.get();
                    mem.commit(&mut x);
                    st.set(x);
                }
                St::ConsumerTryWait => {
                    mem.try_wait(&st.get());
                }
                St::ConsumerWait => mem.wait(&st.get()),
                St::ConsumerRelease => {
                    let mut x = st.get();
                    mem.release(&mut x);
                    st.set(x);
                }
                _ => unreachable!(),
            }
            if set {
                builder::with(|b| b.clear_location());
            }
        }
        scope.close();
    }

    /// A pipeline group's release (commit): one arrival on the shared
    /// barrier for the stage the task's member states are at, then every
    /// member state of the task advances.
    #[track_caller]
    fn group_close(&self, g: &super::PipelineGroup, s: &Step, side: Side) {
        let members: Vec<(&str, &Var<PipelineState>)> = g
            .member_names()
            .iter()
            .filter_map(|m| self.states.get(&(m.clone(), side)).map(|st| (m.as_str(), st)))
            .collect();
        let Some((_, first)) = members.first() else { return };
        let p = g.shared_pipeline().expect("a group created in setup");
        let set = s.loc.map(|loc| builder::with(|b| b.set_location(loc))).unwrap_or(false);
        let index = first.get().index();
        match side {
            Side::Consumer => p.release_index(index),
            Side::Producer => p.commit_index(index),
        }
        for (m, st) in &members {
            let mut x = st.get();
            self.m.res(m).memory().advance_after_close(&mut x, side);
            st.set(x);
        }
        if set {
            builder::with(|b| b.clear_location());
        }
    }

    /// `producer_tail` for every resource the task acquires.
    #[track_caller]
    fn producer_tails(&self) {
        let mut done = HashSet::new();
        walk_steps(&self.task.schedule().body, &mut |s, _| {
            if s.stage == ScheduleStage::ProducerAcquire
                && done.insert(s.resource.clone())
                && let Some(st) = self.states.get(&(s.resource.clone(), Side::Producer))
            {
                let mut x = st.get();
                self.m.res(&s.resource).memory().producer_tail(&mut x);
                st.set(x);
            }
        });
    }
}

/// Whether iteration `i` of `range(start, end, step)` satisfies `p`.
#[track_caller]
fn fires(p: IterationPredicate, (i, start, end, step): (Int32, Int32, Int32, Int32)) -> Boolean {
    match p {
        IterationPredicate::FirstIter => i.eq(start),
        IterationPredicate::LastIter => (i + step).ge(end),
        IterationPredicate::Every { period, start: first } => {
            let k = (i - start) / step;
            let (first, period) = (Int32::constant(first as i32), Int32::constant(period as i32));
            k.ge(first) & ((k - first) % period).eq(Int32::constant(0))
        }
    }
}

/// The schedule line of a step, for messages.
pub(crate) fn at(s: &Step) -> String {
    match s.loc {
        Some(l) => format!(" ({}:{})", short_file(l), l.line()),
        None => String::new(),
    }
}

fn short_file(l: &Location<'_>) -> String {
    l.file().rsplit(['/', '\\']).next().unwrap_or(l.file()).to_string()
}

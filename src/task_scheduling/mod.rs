// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling (NVIDIA EULA); the code and the
// documentation are written independently.

//! Task Scheduling (CuTe DSL's `cutlass.experimental.task_scheduling`):
//! warp-specialised kernels written declaratively, as resources, schedules
//! and tasks.
//!
//! # Design
//!
//! CuTe DSL's Task Scheduling describes a warp-specialised kernel in three
//! layers, which cute-rs builds over three roadmap parts:
//!
//! 1. **Resources** (this part). A [`MemoryResource`] is named memory with
//!    *work methods*: producer work fills it, consumer work reads it and
//!    returns values into task-local variables. CuTe DSL declares work with
//!    decorators on methods of a dataclass; here a resource is a value
//!    built at kernel-construction time whose work methods are registered
//!    as named closures ([`Work::producer`], [`Work::consumer`]) taking a
//!    [`StageInfo`] and named inputs ([`WorkArgs`]) and returning named
//!    results ([`WorkResults`]). No attribute macros are needed, and a
//!    closure captures whatever the work reads (tensors, descriptors,
//!    constants) by move. A resource with a [`PipelineConfig`] is staged:
//!    [`Resource::create`] materialises m9/m10's pipeline
//!    ([`ResourcePipeline`]: `PipelineAsync`, `PipelineCpAsync`,
//!    `PipelineTmaAsync`, `PipelineTmaUmma`, `PipelineUmmaAsync`,
//!    `PipelineAsyncUmma`, `PipelineClcFetchAsync`), and the resource offers
//!    `try_acquire` / `acquire` / `commit` on the producer side and
//!    `try_wait` / `wait` / `release` on the consumer side, over explicit
//!    [`PipelineState`](crate::pipeline::PipelineState)s, which are
//!    [`Carry`](crate::Carry) and so thread through loops. A probe's result
//!    is kept in the side's status flag and taken as the next wait's token.
//!    [`WorkQueue`] is a resource over a persistent tile scheduler (m9's
//!    static one, m10's cluster launch control one) whose work methods
//!    seed, advance and fetch the current [`WorkTileInfo`](crate::utils::WorkTileInfo).
//!    [`TaskLocalVariable`] is a typed handle (its type is the value it
//!    holds); its storage is not part of it but allocated by whoever runs
//!    the work: [`TaskLocals`] in a hand-written kernel, the TaskManager of
//!    the next part. Storage is a [`Var`](crate::Var), so values cross
//!    loop iterations and branches. [`StageInfo`] is a struct of values
//!    implementing [`Carry`](crate::Carry). The resource trait
//!    [`Resource`] exposes the name, pipeline configuration and the
//!    memory-requirement hooks ([`SmemAllocation`], [`TmemAllocation`])
//!    that the allocators of the third part consume.
//!
//! 2. **Schedules** (this part). [`schedule`] runs a builder closure over
//!    [`ResourceHandle`]s and records, without emitting IR, an ordered tree
//!    ([`Schedule`]): work calls with their routes (which earlier step's
//!    variable feeds which input), pipeline operations, one
//!    [`domain_loop`](ScheduleBuilder::domain_loop), one
//!    [`work_tile_loop`](ScheduleBuilder::work_tile_loop) with
//!    [`skippable`](WorkTileLoopScope::skippable) regions, and guarded
//!    blocks ([`when_true`](ScheduleBuilder::when_true) /
//!    [`when_false`](ScheduleBuilder::when_false) over an
//!    [`IterationPredicate`], a work output or an opaque condition). Each
//!    step gets its [`Phase`] (head, loop, tail). Work calls return
//!    [`Flow`] handles, the Rust form of CuTe DSL's data-flow tokens.
//!    Capture validates: unknown work names (with a did-you-mean
//!    suggestion), inputs that nothing produced (or that no pipeline peer
//!    mirrors), wrong or stale inputs, acquire/commit and wait/release
//!    pairing within each block, guards on unsupported predicates, and the
//!    placement rules of loops and blocks. Every problem is collected into
//!    one [`ScheduleError`] of [`Issue`]s; the messages are cute-rs's own.
//!    The schedule prints as a listing, which the tests compare with
//!    expected text.
//!
//! 3. **Tasks and the manager** (part m11-ts-task-manager). A [`Task`]
//!    binds a schedule to a warp range (optionally with a `setmaxnreg`
//!    budget) and names the resources it consumes and produces. The
//!    [`TaskManager`] owns the resources and a [`ResourceDependencyGraph`]
//!    of data ([`DependencyGraphEdge`]) and ordering-only
//!    ([`DependencyGraphControlEdge`]) edges; [`TaskManager::verify`]
//!    checks, without a kernel, the warp ranges, resource roles, the graph
//!    (acyclic, backed, covering), pipeline sides, commit/release balance
//!    across loop nests, cooperative-group thread counts, variable flow
//!    along data edges and programmatic-dependent-launch ordering
//!    ([`PdlWaitBarrier`], [`PdlLaunchBarrier`]). In a kernel,
//!    `setup_resources_and_tasks` creates every resource and publishes the
//!    barriers once, and `run` emits a branch per task: [`TaskLocals`]
//!    storage for the routed variables, pipeline states in `Var`s, the
//!    domain loop as a `for_range`, the work-tile loop as a `while_carry`
//!    over the work queue, profiling scopes (IKET ranges with profiling)
//!    per task and step, and `producer_tail`s. Pipeline operations are
//!    attributed to the schedule line that recorded them, so
//!    [`TaskManager::verify_interleavings`] (the simulator under every
//!    built-in and seeded interleaving) reports deadlocks and races with
//!    the task, step and line.
//!
//! 4. **Memory, groups and Task Scheduling's pipelines** (part
//!    m11-ts-memory-groups). Resources declare memory
//!    ([`SmemAllocation`], [`TmemAllocation`]); the declarative
//!    [`SmemAllocator`] lays every requirement and the barriers of every
//!    pipeline without storage out as one block on the host (largest
//!    alignment first, alias groups sharing bytes), and the
//!    [`TmemAllocator`] lays out TMEM columns. Given to the manager
//!    ([`TaskManager::smem_allocator`], [`TaskManager::tmem_allocator`]),
//!    they are materialised in the prologue: one shared-memory block whose
//!    base every work call sees in [`ResourceContext::smem_base`] (an
//!    allocation's [`ptr`](SmemAllocation::ptr) is its typed view), the
//!    barrier pointers patched into the pipeline configurations, and a TMEM
//!    allocation from warp 0 whose address slot is
//!    [`ResourceContext::tmem_ptr_i32`] and which is freed after every task.
//!    A [`PipelineGroup`] ([`PipelineGroupMode`] `Merge`, `Fork`,
//!    `FusedMerge`) makes its members share one side's barriers: a merge's
//!    consumer waits on each member and releases them all through the
//!    group, a fork's producer commits them all through it. Capture,
//!    verification and emission understand groups. [`TsPipeline`] is Task
//!    Scheduling's own pipeline (`TSPipelineTmaAsync` with a task-warp
//!    leader arming each TMA stage or every consumer thread releasing,
//!    `TSPipelineAsyncUmma`, `TSPipelineUmmaUmma`) and backs `UmmaUmma`
//!    resources and group members. The manager's checks add alias races
//!    (shared bytes whose owners the graph does not order with control
//!    edges), group arrival counts, TMEM capacity and pipelines left
//!    without barrier storage.
//!
//! 5. **The end-to-end GEMM** (part m11-ts-gemm-example,
//!    `examples/ts_gemm_simple.rs`, the port of
//!    `02_gemm_simple_ts/01_fp16_bf16_gemm_3.py`) added what that tutorial
//!    needs: compile-time work arguments ([`Work::constexpr_input`],
//!    [`ArgSource::constexpr`], [`WorkArgs::constexpr`]; CuTe DSL's
//!    `Constexpr[int]` keyword arguments such as `subtile_idx`), register
//!    vectors as task-local values ([`LocalValue`] for
//!    [`Vector`](crate::Vector)), and register budgets on tasks smaller
//!    than a warpgroup when the tasks covering each warpgroup agree on one
//!    (set once per warpgroup, before the dispatch, so that its four warps
//!    execute the same `setmaxnreg`).
//!
//! Scope: what the copy tutorials need (`01_copy_basics_ts` and
//! `03_persistent_scheduling_dynamic_domain_ts/01_copy_persistent_skip_tile.py`)
//! and the single-CTA GEMM above. Pipelines left without barrier storage
//! stay an error when an `SmemAllocator` is given (CuTe DSL lets such a
//! pipeline allocate its own barriers): register the resource with the
//! allocator.
//! Deferred: callable (dynamic-domain) loop
//! bounds, work labels selecting among hooks, `advance_on_acquire` /
//! `advance_on_wait` and interleaved lane starts at run time (the
//! configuration accepts and validates them; the resource methods use the
//! default policy), signalling-thread restrictions at run time (other
//! than the task-warp leader of a TMA producer), exhaustive (rather than
//! seeded) interleaving exploration, loop peeling for first/last
//! iteration guards (they are `if`s in the loop), IKET ranges per phase
//! (there is one per task and one per step), and clusters for
//! [`TsPipeline`]s and pipeline groups (one CTA; 2-CTA UMMA commits and
//! multicast masks are not modelled).
//!
//! # Example
//!
//! The grid-stride copy of `01_copy_grid_stride.py`, captured:
//!
//! ```
//! use cute::prelude::*;
//! use cute::task_scheduling::*;
//!
//! let item = TaskLocalVariable::<Int16>::scalar("item", 0, "the element of this iteration");
//! let input = MemoryResource::new("inputGmemResource")
//!     .work(Work::consumer("get_item").returns(&item).run(|_, _| WorkResults::none()));
//! let output = MemoryResource::new("outputGmemResource")
//!     .work(Work::producer("set_item").input::<Int16>("data").run(|_, _| WorkResults::none()));
//! let s = schedule("schedule_fn", &[&input, &output], |s, r| {
//!     let (input, output) = (r[0], r[1]);
//!     s.domain_loop(Bound::symbol("start"), Bound::symbol("num_entries"), Bound::symbol("step"), 1, |_| {
//!         let res = input.call("get_item").one();
//!         output.call_with("set_item", [("data", &res)]);
//!     });
//! })
//! .unwrap();
//! assert_eq!(s.steps().len(), 2);
//! ```

mod config;
mod enums;
mod manager;
mod memory;
mod pdl;
mod pipeline;
mod pipeline_group;
mod resource;
mod schedule;
mod task;
mod verify;
mod work_queue;

pub use config::{ConfigError, InterleaveStride, PipelineConfig, TileSchedulerConfig};
pub use enums::{
    Guard, IterationPredicate, Phase, PipelineGroupMode, PipelineType, ScheduleStage, SignalingThreads,
    TileSchedulerType, WorkAttr,
};
pub use manager::{
    DependencyGraphControlEdge, DependencyGraphEdge, InterleavingError, InterleavingErrorKind, ResourceDependencyGraph,
    TaskManager, VerifyError, VerifyIssue, VerifyIssueKind,
};
pub use memory::{
    AllocError, BarrierRegion, DEFAULT_SMEM_CAPACITY, MBARRIER_BYTES, Region, SmemAllocator, SmemLayout, TMEM_COLUMNS,
    TmemAllocator, TmemLayout,
};
pub use pdl::{PdlLaunchBarrier, PdlRole, PdlWaitBarrier};
pub use pipeline::{AgentKind, TsPipeline, TsPipelineOptions};
pub use pipeline_group::PipelineGroup;
pub use resource::{
    Constexpr, LocalValue, MemoryResource, Resource, ResourceContext, ResourcePipeline, Side, SmemAllocation,
    StageInfo, TaskLocalVariable, TaskLocals, TmemAllocation, VarRef, Work, WorkArgs, WorkDecl, WorkFn, WorkInput,
    WorkResults,
};
pub use schedule::{
    ArgSource, Bound, Condition, ConditionalBlock, DomainLoop, DomainLoopScope, Flow, Issue, IssueKind, Node, Outputs,
    ResourceHandle, ResourceInfo, Route, Schedule, ScheduleBuilder, ScheduleError, Source, Step, WorkTileLoop,
    WorkTileLoopScope, schedule,
};
pub use task::{RegisterBudget, Task};
pub use work_queue::{TileScheduler, WorkQueue};

/// The first-iteration predicate (CuTe DSL's `LoopFirstIter`).
pub const FIRST_ITER: IterationPredicate = IterationPredicate::FirstIter;
/// The last-iteration predicate (CuTe DSL's `LoopLastIter`).
pub const LAST_ITER: IterationPredicate = IterationPredicate::LastIter;

/// The periodic predicate `Every(period, start)`.
pub fn every(period: u32, start: u32) -> IterationPredicate {
    IterationPredicate::Every { period, start }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.schedule_builder (NVIDIA EULA); the
// code, the listing format and the messages are written independently.

//! Schedule capture: a builder closure over resource handles records the
//! order of work calls and pipeline operations, loops and guards into a
//! [`Schedule`], without emitting IR, and validates it.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::value::Int32;

use super::enums::{Guard, IterationPredicate, Phase, ScheduleStage};
use super::resource::{Constexpr, LocalValue, Resource, Side, TaskLocalVariable, VarRef};

// ---------------------------------------------------------------------------
// The captured graph
// ---------------------------------------------------------------------------

/// A bound of a domain loop: a number fixed when the schedule is built, or
/// a run-time value (named for listings; the value itself is only needed
/// when the schedule is emitted into a kernel).
#[derive(Clone, Debug)]
pub enum Bound {
    Const(i64),
    Runtime { name: String, value: Option<Int32> },
}

impl Bound {
    /// The run-time value `value`, listed as `name`.
    pub fn runtime(name: &str, value: Int32) -> Bound {
        Bound::Runtime { name: name.to_string(), value: Some(value) }
    }

    /// A run-time value called `name`, not yet known (capture outside a
    /// kernel).
    pub fn symbol(name: &str) -> Bound {
        Bound::Runtime { name: name.to_string(), value: None }
    }

    /// The constant, if this is one.
    pub fn as_const(&self) -> Option<i64> {
        match self {
            Bound::Const(c) => Some(*c),
            Bound::Runtime { .. } => None,
        }
    }
}

macro_rules! bound_from {
    ($($t:ty),*) => {$(
        impl From<$t> for Bound {
            fn from(v: $t) -> Bound {
                Bound::Const(v as i64)
            }
        }
    )*};
}
bound_from!(i32, i64, u32, usize);

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bound::Const(c) => write!(f, "{c}"),
            Bound::Runtime { name, .. } => f.write_str(name),
        }
    }
}

/// Where a work input's value comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A variable an earlier step of this schedule produced.
    Step { step: usize, variable: String },
    /// A variable of a pipelined resource, produced by the task on the
    /// other side of the pipeline and mirrored into this task.
    Peer { resource: String, variable: String },
    /// A compile-time integer the call site gives
    /// ([`ArgSource::constexpr`]).
    Constexpr(i64),
}

/// One input of a work step and where it comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub argument: String,
    pub source: Source,
}

/// A leaf of a schedule: a pipeline operation or a work call on a
/// resource.
#[derive(Clone, Debug)]
pub struct Step {
    /// Numbered in capture order from 0.
    pub id: usize,
    /// The resource's name.
    pub resource: String,
    /// What the step does.
    pub stage: ScheduleStage,
    /// The work method's name (`None` for a pipeline operation).
    pub label: Option<String>,
    /// Where the step sits relative to its body's domain loop.
    pub phase: Phase,
    /// The inputs of a work call.
    pub inputs: Vec<Route>,
    /// The variables a work call produces.
    pub outputs: Vec<VarRef>,
    /// Where the schedule recorded it: the TaskManager attributes the
    /// pipeline operation it emits to this line, so that a deadlock or a
    /// race is reported at the schedule step.
    pub loc: Option<&'static std::panic::Location<'static>>,
}

/// A region run only when its guard holds.
#[derive(Clone, Debug)]
pub struct ConditionalBlock {
    pub guard: Guard,
    /// The variable a work-output guard tests.
    pub condition: Option<VarRef>,
    pub body: Vec<Node>,
}

/// A `range(start, end, step)` loop over the domain.
#[derive(Clone, Debug)]
pub struct DomainLoop {
    pub start: Bound,
    pub end: Bound,
    pub step: Bound,
    pub unroll: u32,
    pub body: Vec<Node>,
}

/// The persistent loop over the tiles of a work queue.
#[derive(Clone, Debug)]
pub struct WorkTileLoop {
    /// The work queue's name.
    pub work_queue: String,
    /// Whether the queue's skip predicate leaves out the skippable regions.
    pub skip_if: bool,
    pub body: Vec<Node>,
}

/// A node of a schedule.
#[derive(Clone, Debug)]
pub enum Node {
    Step(Step),
    Block(ConditionalBlock),
    DomainLoop(DomainLoop),
    WorkTileLoop(WorkTileLoop),
}

impl Node {
    /// The nodes inside a block or loop (none for a step).
    pub fn body(&self) -> &[Node] {
        match self {
            Node::Step(_) => &[],
            Node::Block(b) => &b.body,
            Node::DomainLoop(l) => &l.body,
            Node::WorkTileLoop(l) => &l.body,
        }
    }

    fn body_mut(&mut self) -> Option<&mut Vec<Node>> {
        match self {
            Node::Step(_) => None,
            Node::Block(b) => Some(&mut b.body),
            Node::DomainLoop(l) => Some(&mut l.body),
            Node::WorkTileLoop(l) => Some(&mut l.body),
        }
    }
}

/// What a schedule knows of a resource: its name, whether it is pipelined
/// or a work queue, its work methods' signatures and its variables.
#[derive(Clone, Debug)]
pub struct ResourceInfo {
    pub name: String,
    pub pipelined: bool,
    pub work_queue: bool,
    /// For a [`PipelineGroup`](super::PipelineGroup): the side it closes
    /// for its members (release or commit) and the members.
    pub group: Option<(Side, Vec<String>)>,
    works: Vec<WorkSig>,
    variables: Vec<VarRef>,
}

#[derive(Clone, Debug)]
struct WorkSig {
    name: String,
    stage: ScheduleStage,
    inputs: Vec<(String, String, TypeId)>,
    returns: Vec<VarRef>,
}

impl ResourceInfo {
    fn of(r: &dyn Resource) -> ResourceInfo {
        let m = r.memory();
        ResourceInfo {
            name: r.name().to_string(),
            pipelined: r.pipeline_config().is_some(),
            work_queue: r.is_work_queue(),
            group: r.as_group().map(|g| (g.shared_side(), g.member_names().to_vec())),
            works: m
                .works()
                .iter()
                .map(|w| WorkSig {
                    name: w.name().to_string(),
                    stage: w.stage(),
                    inputs: w
                        .inputs()
                        .iter()
                        .map(|i| (i.name().to_string(), i.type_label().to_string(), i.type_id()))
                        .collect(),
                    returns: w.returns().to_vec(),
                })
                .collect(),
            variables: m.variables().cloned().collect(),
        }
    }

    /// The names of its work methods.
    pub fn work_names(&self) -> impl Iterator<Item = &str> {
        self.works.iter().map(|w| w.name.as_str())
    }
}

/// A captured schedule (CuTe DSL's `Schedule`): an ordered tree of steps,
/// guarded blocks, a domain loop and a work-tile loop, over the resources
/// it was captured with. Its [`Display`](fmt::Display) is a listing, one
/// line per node:
///
/// ```text
/// schedule copy [input, output]
///   domain_loop(start, n, step)
///     #0 loop input.get_item -> item
///     #1 loop output.set_item(data <- #0.item)
/// ```
#[derive(Clone, Debug)]
pub struct Schedule {
    pub name: String,
    pub resources: Vec<ResourceInfo>,
    pub body: Vec<Node>,
    /// Advisories: outputs no later step reads.
    pub warnings: Vec<String>,
}

impl Schedule {
    /// Every step, in capture order.
    pub fn steps(&self) -> Vec<&Step> {
        fn walk<'a>(nodes: &'a [Node], out: &mut Vec<&'a Step>) {
            for n in nodes {
                if let Node::Step(s) = n {
                    out.push(s);
                }
                walk(n.body(), out);
            }
        }
        let mut out = Vec::new();
        walk(&self.body, &mut out);
        out
    }

    /// Whether the schedule has a work-tile loop.
    pub fn has_work_tile_loop(&self) -> bool {
        self.body.iter().any(|n| matches!(n, Node::WorkTileLoop(_)))
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.resources.iter().map(|r| r.name.as_str()).collect();
        write!(f, "schedule {} [{}]", self.name, names.join(", "))?;
        fn node(f: &mut fmt::Formatter<'_>, n: &Node, depth: usize) -> fmt::Result {
            let pad = "  ".repeat(depth);
            match n {
                Node::Step(s) => {
                    let what = s.label.as_deref().or(s.stage.op_name()).unwrap_or("?");
                    write!(f, "\n{pad}#{} {} {}.{}", s.id, s.phase, s.resource, what)?;
                    if s.stage.is_auxiliary() {
                        f.write_str(" [aux]")?;
                    }
                    if !s.inputs.is_empty() {
                        let ins: Vec<String> = s
                            .inputs
                            .iter()
                            .map(|r| match &r.source {
                                Source::Step { step, variable } => format!("{} <- #{step}.{variable}", r.argument),
                                Source::Peer { resource, variable } => {
                                    format!("{} <- peer {resource}.{variable}", r.argument)
                                }
                                Source::Constexpr(v) => format!("{} = {v}", r.argument),
                            })
                            .collect();
                        write!(f, "({})", ins.join(", "))?;
                    }
                    if !s.outputs.is_empty() {
                        let outs: Vec<&str> = s.outputs.iter().map(|v| v.name()).collect();
                        write!(f, " -> {}", outs.join(", "))?;
                    }
                }
                Node::Block(b) => write!(f, "\n{pad}{}", b.guard)?,
                Node::DomainLoop(l) => {
                    write!(f, "\n{pad}domain_loop({}, {}, {}", l.start, l.end, l.step)?;
                    if l.unroll != 1 {
                        write!(f, ", unroll={}", l.unroll)?;
                    }
                    f.write_str(")")?;
                }
                Node::WorkTileLoop(l) => {
                    write!(f, "\n{pad}work_tile_loop({}", l.work_queue)?;
                    if l.skip_if {
                        write!(f, ", skip_if={}.skip_work_tile_if", l.work_queue)?;
                    }
                    f.write_str(")")?;
                }
            }
            for c in n.body() {
                node(f, c, depth + 1)?;
            }
            Ok(())
        }
        for n in &self.body {
            node(f, n, 1)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The kind of a capture problem, for matching in tests and tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IssueKind {
    /// A work method the resource does not have.
    UnknownWork,
    /// A work input with no producer: missing, from a step that produces
    /// nothing, or a variable neither produced earlier nor mirrored from a
    /// pipeline peer.
    UnproducedVariable,
    /// An input that is unknown, of the wrong type, or a stale handle.
    BadArgument,
    /// An acquire without a commit, a commit without an acquire, a probe
    /// not followed by its wait, and the same on the consumer side.
    UnpairedPipelineOp,
    /// A guard on a predicate that cannot be used there.
    UnsupportedGuard,
    /// A loop or block where it may not be.
    BadStructure,
}

/// One problem found while capturing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub kind: IssueKind,
    pub message: String,
}

/// Why a schedule could not be captured: every problem found (CuTe DSL's
/// `ScheduleError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleError {
    pub schedule: String,
    pub issues: Vec<Issue>,
}

impl ScheduleError {
    /// Whether some issue is of `kind`.
    pub fn has(&self, kind: IssueKind) -> bool {
        self.issues.iter().any(|i| i.kind == kind)
    }
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schedule `{}` is invalid:", self.schedule)?;
        for i in &self.issues {
            write!(f, "\n  - {}", i.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for ScheduleError {}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// The candidate closest to `name`, if close enough to be a likely typo.
pub(crate) fn did_you_mean<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).max(2);
    candidates
        .into_iter()
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

static NEXT_SCHEDULE_ID: AtomicU64 = AtomicU64::new(1);

/// A handle to a value a work call produced (CuTe DSL's data-flow token):
/// pass it to later work as a named input, or to `when_true` /
/// `when_false` as a guard.
#[derive(Clone, Debug)]
pub struct Flow {
    schedule: u64,
    step: Option<usize>,
    resource: String,
    label: String,
    var: Option<VarRef>,
    generation: u64,
}

impl Flow {
    /// The variable the value is stored in (`None` for the handle of a
    /// call that produces nothing).
    pub fn variable(&self) -> Option<&VarRef> {
        self.var.as_ref()
    }

    /// The step that produced it.
    pub fn step(&self) -> Option<usize> {
        self.step
    }
}

/// The handles a work call returns, one per declared result.
#[derive(Clone, Debug)]
pub struct Outputs<'b> {
    b: &'b ScheduleBuilder,
    flows: Vec<Flow>,
    empty: Flow,
}

impl Outputs<'_> {
    /// The only result. For a call that produces nothing this is a handle
    /// without a variable, which cannot feed another call; for a call that
    /// produces several, use [`named`](Self::named).
    pub fn one(&self) -> Flow {
        match self.flows.len() {
            0 => self.empty.clone(),
            1 => self.flows[0].clone(),
            n => {
                self.b.issue(
                    IssueKind::BadArgument,
                    format!("{}.{} returns {n} values; pick one by name", self.empty.resource, self.empty.label),
                );
                self.flows[0].clone()
            }
        }
    }

    /// The result stored in the variable `name`.
    pub fn named(&self, name: &str) -> Flow {
        match self.flows.iter().find(|f| f.var.as_ref().is_some_and(|v| v.name() == name)) {
            Some(f) => f.clone(),
            None => {
                self.b.issue(
                    IssueKind::BadArgument,
                    format!("{}.{} does not return `{name}`", self.empty.resource, self.empty.label),
                );
                self.empty.clone()
            }
        }
    }

    /// All the results, in declaration order.
    pub fn all(&self) -> &[Flow] {
        &self.flows
    }
}

/// A work input: a handle a call returned, or a variable by name (which
/// must have been produced earlier in the schedule or be mirrored from a
/// pipeline peer).
#[derive(Clone, Debug)]
pub enum ArgSource {
    Flow(Flow),
    Var(VarRef),
    /// A compile-time integer, for an input declared with
    /// [`Work::constexpr_input`](super::Work::constexpr_input).
    Constexpr(i64),
}

impl ArgSource {
    /// The compile-time integer `value` (CuTe DSL's constexpr keyword
    /// arguments, `load_subtile(subtile_idx=i)`).
    pub fn constexpr(value: i64) -> ArgSource {
        ArgSource::Constexpr(value)
    }
}

impl From<Flow> for ArgSource {
    fn from(f: Flow) -> Self {
        ArgSource::Flow(f)
    }
}

impl From<&Flow> for ArgSource {
    fn from(f: &Flow) -> Self {
        ArgSource::Flow(f.clone())
    }
}

impl<C: LocalValue> From<&TaskLocalVariable<C>> for ArgSource {
    fn from(v: &TaskLocalVariable<C>) -> Self {
        ArgSource::Var(v.var_ref().clone())
    }
}

/// The guard of `when_true` / `when_false`.
#[derive(Clone, Debug)]
pub enum Condition {
    /// An iteration predicate (only `when_true`, inside a domain loop).
    Iteration(IterationPredicate),
    /// A boolean or integer a work call produced.
    Flow(Flow),
    /// A run-time condition known by a key.
    Opaque(String),
}

impl From<IterationPredicate> for Condition {
    fn from(p: IterationPredicate) -> Self {
        Condition::Iteration(p)
    }
}

impl From<Flow> for Condition {
    fn from(f: Flow) -> Self {
        Condition::Flow(f)
    }
}

impl From<&Flow> for Condition {
    fn from(f: &Flow) -> Self {
        Condition::Flow(f.clone())
    }
}

#[derive(Clone, Debug)]
enum FrameKind {
    Root,
    Block(Guard, Option<VarRef>),
    Domain { start: Bound, end: Bound, step: Bound, unroll: u32 },
    WorkTile { work_queue: String, skip_if: bool },
}

struct Frame {
    kind: FrameKind,
    body: Vec<Node>,
}

struct State {
    frames: Vec<Frame>,
    issues: Vec<Issue>,
    next_step: usize,
    /// var id -> (generation of the latest handle, producing step).
    latest: HashMap<u64, (u64, usize)>,
    next_generation: u64,
    /// (step, var id) pairs some later step or guard reads.
    used: HashSet<(usize, u64)>,
    has_work_tile_loop: bool,
    after_work_tile_loop: bool,
}

/// The recorder a schedule closure runs against (CuTe DSL's
/// `ScheduleBuilder`): it opens loops and blocks around closures and
/// collects the steps the [`ResourceHandle`]s record.
pub struct ScheduleBuilder {
    id: u64,
    resources: Vec<ResourceInfo>,
    state: RefCell<State>,
}

impl fmt::Debug for ScheduleBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduleBuilder").field("id", &self.id).finish()
    }
}

/// A resource as seen by a schedule closure: calls record steps.
#[derive(Clone, Copy, Debug)]
pub struct ResourceHandle<'b> {
    b: &'b ScheduleBuilder,
    idx: usize,
}

/// The domain loop being recorded, for its iteration guards.
#[derive(Debug)]
pub struct DomainLoopScope<'b> {
    b: &'b ScheduleBuilder,
}

/// The work-tile loop being recorded, for its skippable regions.
#[derive(Debug)]
pub struct WorkTileLoopScope<'b> {
    b: &'b ScheduleBuilder,
}

/// Capture the schedule `name` over `resources` (CuTe DSL's `@schedule`):
/// `body` gets the builder and a handle per resource, in order, and records
/// steps by calling them. Nothing is emitted; the result is the validated
/// [`Schedule`] or every problem found.
///
/// ```
/// use cute::prelude::*;
/// use cute::task_scheduling::*;
///
/// let item = TaskLocalVariable::<Int16>::scalar("item", 0, "");
/// let input = MemoryResource::new("input")
///     .work(Work::consumer("get_item").returns(&item).run(|_, _| WorkResults::none()));
/// let output = MemoryResource::new("output")
///     .work(Work::producer("set_item").input::<Int16>("data").run(|_, _| WorkResults::none()));
/// let s = schedule("copy", &[&input, &output], |s, r| {
///     let (input, output) = (r[0], r[1]);
///     s.domain_loop(Bound::symbol("start"), Bound::symbol("n"), Bound::symbol("step"), 1, |_| {
///         let res = input.call("get_item").one();
///         output.call_with("set_item", [("data", &res)]);
///     });
/// })
/// .unwrap();
/// assert_eq!(
///     s.to_string(),
///     "schedule copy [input, output]\n  domain_loop(start, n, step)\n    #0 loop input.get_item -> item\n    \
///      #1 loop output.set_item(data <- #0.item)"
/// );
/// ```
pub fn schedule(
    name: &str,
    resources: &[&dyn Resource],
    body: impl FnOnce(&ScheduleBuilder, &[ResourceHandle<'_>]),
) -> Result<Schedule, ScheduleError> {
    let b = ScheduleBuilder {
        id: NEXT_SCHEDULE_ID.fetch_add(1, Ordering::Relaxed),
        resources: resources.iter().map(|r| ResourceInfo::of(*r)).collect(),
        state: RefCell::new(State {
            frames: vec![Frame { kind: FrameKind::Root, body: vec![] }],
            issues: vec![],
            next_step: 0,
            latest: HashMap::new(),
            next_generation: 1,
            used: HashSet::new(),
            has_work_tile_loop: false,
            after_work_tile_loop: false,
        }),
    };
    let mut seen = HashSet::new();
    for r in &b.resources {
        if !seen.insert(r.name.clone()) {
            b.issue(IssueKind::BadStructure, format!("two resources are called `{}`", r.name));
        }
    }
    let handles: Vec<ResourceHandle<'_>> = (0..b.resources.len()).map(|idx| ResourceHandle { b: &b, idx }).collect();
    body(&b, &handles);
    b.finish(name)
}

impl ScheduleBuilder {
    fn issue(&self, kind: IssueKind, message: String) {
        self.state.borrow_mut().issues.push(Issue { kind, message });
    }

    fn frames_any(&self, f: impl Fn(&FrameKind) -> bool) -> bool {
        self.state.borrow().frames.iter().any(|fr| f(&fr.kind))
    }

    fn in_domain_loop(&self) -> bool {
        self.frames_any(|k| matches!(k, FrameKind::Domain { .. }))
    }

    fn in_skippable(&self) -> bool {
        self.frames_any(|k| matches!(k, FrameKind::Block(Guard::Skippable, _)))
    }

    fn in_condition(&self) -> bool {
        self.frames_any(|k| matches!(k, FrameKind::Block(Guard::Iteration(_) | Guard::Opaque { .. }, _)))
    }

    fn skipping_loop(&self) -> Option<bool> {
        self.state.borrow().frames.iter().rev().find_map(|fr| match &fr.kind {
            FrameKind::WorkTile { skip_if, .. } => Some(*skip_if),
            _ => None,
        })
    }

    fn with_frame(&self, kind: FrameKind, f: impl FnOnce()) {
        self.state.borrow_mut().frames.push(Frame { kind, body: vec![] });
        f();
        let mut st = self.state.borrow_mut();
        let frame = st.frames.pop().expect("schedule frame");
        let node = match frame.kind {
            FrameKind::Root => unreachable!(),
            FrameKind::Block(guard, condition) => Node::Block(ConditionalBlock { guard, condition, body: frame.body }),
            FrameKind::Domain { start, end, step, unroll } => {
                Node::DomainLoop(DomainLoop { start, end, step, unroll, body: frame.body })
            }
            FrameKind::WorkTile { work_queue, skip_if } => {
                st.after_work_tile_loop = true;
                Node::WorkTileLoop(WorkTileLoop { work_queue, skip_if, body: frame.body })
            }
        };
        st.frames.last_mut().expect("schedule root").body.push(node);
    }

    /// Record a domain loop `range(start, end, step)` (CuTe DSL's
    /// `domain_loop(start, end, step, unroll=1)`) around `body`. Domain
    /// loops do not nest; inside a `work_tile_loop` with a skip predicate
    /// the loop must be in a skippable region.
    pub fn domain_loop(
        &self,
        start: impl Into<Bound>,
        end: impl Into<Bound>,
        step: impl Into<Bound>,
        unroll: u32,
        body: impl FnOnce(&DomainLoopScope<'_>),
    ) {
        let step = step.into();
        if self.in_domain_loop() {
            self.issue(IssueKind::BadStructure, "a domain_loop is nested in another domain_loop".into());
        }
        if self.skipping_loop() == Some(true) && !self.in_skippable() {
            self.issue(
                IssueKind::BadStructure,
                "in a work_tile_loop with a skip predicate, the domain_loop must be inside skippable()".into(),
            );
        }
        if step.as_const() == Some(0) {
            self.issue(IssueKind::BadStructure, "the domain_loop's step is 0".into());
        }
        if unroll == 0 {
            self.issue(IssueKind::BadStructure, "the domain_loop's unroll factor is 0".into());
        }
        self.check_not_after_work_tile_loop("domain_loop");
        let kind = FrameKind::Domain { start: start.into(), end: end.into(), step, unroll };
        self.with_frame(kind, || body(&DomainLoopScope { b: self }));
    }

    /// Record the persistent loop over the tiles of `work_queue` (CuTe
    /// DSL's `work_tile_loop(wq)`) around `body`: top level only, at most
    /// once, with only auxiliary steps before and after it.
    pub fn work_tile_loop(&self, work_queue: ResourceHandle<'_>, body: impl FnOnce(&WorkTileLoopScope<'_>)) {
        self.work_tile_loop_impl(work_queue, false, body);
    }

    /// [`work_tile_loop`](Self::work_tile_loop) with the queue's skip
    /// predicate (`work_tile_loop(wq, skip_if=WQ.skip_work_tile_if)`): the
    /// [`skippable`](WorkTileLoopScope::skippable) regions are left out for
    /// the tiles it selects, while the queue's bookkeeping still runs.
    pub fn work_tile_loop_skipping(&self, work_queue: ResourceHandle<'_>, body: impl FnOnce(&WorkTileLoopScope<'_>)) {
        self.work_tile_loop_impl(work_queue, true, body);
    }

    fn work_tile_loop_impl(&self, wq: ResourceHandle<'_>, skip_if: bool, body: impl FnOnce(&WorkTileLoopScope<'_>)) {
        let info = &self.resources[wq.idx];
        if !info.work_queue {
            self.issue(
                IssueKind::BadStructure,
                format!("work_tile_loop needs a WorkQueue, and `{}` is not one", info.name),
            );
        }
        {
            let st = self.state.borrow();
            if st.frames.len() != 1 {
                drop(st);
                self.issue(IssueKind::BadStructure, "a work_tile_loop must be at the top level of the schedule".into());
            } else if st.has_work_tile_loop {
                drop(st);
                self.issue(IssueKind::BadStructure, "a schedule has at most one work_tile_loop".into());
            } else if st.frames[0].body.iter().any(|n| !allowed_outside_work_tile_loop(n)) {
                drop(st);
                self.issue(IssueKind::BadStructure, "only auxiliary work may come before a work_tile_loop".into());
            }
        }
        self.state.borrow_mut().has_work_tile_loop = true;
        let kind = FrameKind::WorkTile { work_queue: info.name.clone(), skip_if };
        self.with_frame(kind, || body(&WorkTileLoopScope { b: self }));
    }

    /// Record `body` as run only when `cond` holds (CuTe DSL's
    /// `when_true`): an iteration predicate (inside a domain loop), a value
    /// a work call produced, or an opaque run-time condition. Per-iteration
    /// and opaque blocks do not nest.
    pub fn when_true(&self, cond: impl Into<Condition>, body: impl FnOnce()) {
        self.when(cond.into(), false, body);
    }

    /// Record `body` as run only when `cond` is false (CuTe DSL's
    /// `when_false`). Iteration predicates cannot be negated.
    pub fn when_false(&self, cond: impl Into<Condition>, body: impl FnOnce()) {
        self.when(cond.into(), true, body);
    }

    fn when(&self, cond: Condition, negated: bool, body: impl FnOnce()) {
        let api = if negated { "when_false" } else { "when_true" };
        if self.in_condition() {
            self.issue(
                IssueKind::UnsupportedGuard,
                format!("{api}: guarded blocks do not nest; write independent conditions as sibling blocks"),
            );
        }
        let mut condition = None;
        let guard = match cond {
            Condition::Iteration(p) => {
                if negated {
                    self.issue(
                        IssueKind::UnsupportedGuard,
                        format!("{api}: the iteration predicate {p} cannot be negated"),
                    );
                }
                if !self.in_domain_loop() {
                    self.issue(
                        IssueKind::UnsupportedGuard,
                        format!("{api}: the iteration predicate {p} needs a domain_loop"),
                    );
                }
                if let IterationPredicate::Every { period: 0, .. } = p {
                    self.issue(IssueKind::UnsupportedGuard, format!("{api}: every() needs a positive period"));
                }
                Guard::Iteration(p)
            }
            Condition::Flow(flow) => {
                let key = format!("{}.{}", flow.resource, flow.label);
                match &flow.var {
                    _ if flow.schedule != self.id => {
                        self.issue(
                            IssueKind::UnsupportedGuard,
                            format!("{api}: the handle of {key} is from another schedule"),
                        );
                        Guard::Opaque { key, negated }
                    }
                    None => {
                        self.issue(
                            IssueKind::UnsupportedGuard,
                            format!("{api}: {key} produces no value to test (declare it with returns)"),
                        );
                        Guard::Opaque { key, negated }
                    }
                    Some(var) => {
                        if !var.is_scalar() {
                            self.issue(
                                IssueKind::UnsupportedGuard,
                                format!("{api}: `{}` is a {}, not a boolean or integer", var.name(), var.type_label()),
                            );
                        }
                        self.check_fresh(&flow, var, api);
                        if let Some(step) = flow.step {
                            self.state.borrow_mut().used.insert((step, var.id()));
                        }
                        condition = Some(var.clone());
                        Guard::Opaque { key: format!("{key} -> {}", var.name()), negated }
                    }
                }
            }
            Condition::Opaque(key) => Guard::Opaque { key, negated },
        };
        self.with_frame(FrameKind::Block(guard, condition), body);
    }

    fn check_fresh(&self, flow: &Flow, var: &VarRef, what: &str) {
        let latest = self.state.borrow().latest.get(&var.id()).map(|(g, _)| *g);
        if latest != Some(flow.generation) {
            self.issue(
                IssueKind::BadArgument,
                format!(
                    "{what}: the handle of `{}` from {}.{} is stale; a later call produced it again",
                    var.name(),
                    flow.resource,
                    flow.label
                ),
            );
        }
    }

    fn check_not_after_work_tile_loop(&self, what: &str) {
        if self.state.borrow().after_work_tile_loop {
            self.issue(
                IssueKind::BadStructure,
                format!("{what} after the work_tile_loop; only auxiliary work may follow it"),
            );
        }
    }

    #[track_caller]
    fn record(&self, idx: usize, stage: ScheduleStage, label: Option<&str>) -> usize {
        let loc = std::panic::Location::caller();
        let info = &self.resources[idx];
        if info.work_queue && self.in_skippable() {
            self.issue(
                IssueKind::BadStructure,
                format!(
                    "{}.{}: work queue steps cannot be skippable; skipped tiles still need the queue's bookkeeping",
                    info.name,
                    label.or(stage.op_name()).unwrap_or("?")
                ),
            );
        }
        let mut st = self.state.borrow_mut();
        if st.after_work_tile_loop && !stage.is_auxiliary() {
            st.issues.push(Issue {
                kind: IssueKind::BadStructure,
                message: format!(
                    "{}.{} after the work_tile_loop; only auxiliary work may follow it",
                    info.name,
                    label.or(stage.op_name()).unwrap_or("?")
                ),
            });
        }
        let id = st.next_step;
        st.next_step += 1;
        st.frames.last_mut().expect("schedule frame").body.push(Node::Step(Step {
            id,
            resource: info.name.clone(),
            stage,
            label: label.map(str::to_string),
            phase: Phase::Head,
            inputs: vec![],
            outputs: vec![],
            loc: Some(loc),
        }));
        id
    }

    fn step_mut<R>(&self, id: usize, f: impl FnOnce(&mut Step) -> R) -> R {
        fn find(nodes: &mut [Node], id: usize) -> Option<&mut Step> {
            for n in nodes.iter_mut().rev() {
                if let Node::Step(s) = n {
                    if s.id == id {
                        return Some(s);
                    }
                } else if let Some(s) = n.body_mut().and_then(|b| find(b, id)) {
                    return Some(s);
                }
            }
            None
        }
        let mut st = self.state.borrow_mut();
        let frame = st.frames.last_mut().expect("schedule frame");
        f(find(&mut frame.body, id).expect("the step just recorded"))
    }

    fn finish(self, name: &str) -> Result<Schedule, ScheduleError> {
        let mut st = self.state.into_inner();
        let root = st.frames.pop().expect("schedule root");
        let mut body = root.body;
        let groups: HashMap<&str, (Side, Vec<&str>)> = self
            .resources
            .iter()
            .filter_map(|r| {
                let (side, members) = r.group.as_ref()?;
                Some((r.name.as_str(), (*side, members.iter().map(String::as_str).collect())))
            })
            .collect();
        check_pairing(&body, &groups, &mut st.issues);
        assign_phases(&mut body, false);
        if !st.issues.is_empty() {
            return Err(ScheduleError { schedule: name.to_string(), issues: st.issues });
        }
        let mut sched = Schedule { name: name.to_string(), resources: self.resources, body, warnings: vec![] };
        let wq: HashSet<&str> = sched.resources.iter().filter(|r| r.work_queue).map(|r| r.name.as_str()).collect();
        let mut warnings = vec![];
        for s in sched.steps() {
            if wq.contains(s.resource.as_str()) {
                continue;
            }
            for v in &s.outputs {
                if !st.used.contains(&(s.id, v.id())) {
                    warnings.push(format!(
                        "{}.{} produces `{}`, which no later step reads",
                        s.resource,
                        s.label.as_deref().unwrap_or("?"),
                        v.name()
                    ));
                }
            }
        }
        sched.warnings = warnings;
        Ok(sched)
    }
}

fn allowed_outside_work_tile_loop(n: &Node) -> bool {
    matches!(n, Node::Step(s) if s.stage.is_auxiliary())
}

/// Check that, in every body, each resource's acquires are committed and
/// its waits released, with probes followed by their acquire or wait. A
/// pipeline group's release (commit) closes the waits (acquires) its
/// members hold in the body, and is the only operation a group takes.
fn check_pairing(body: &[Node], groups: &HashMap<&str, (Side, Vec<&str>)>, issues: &mut Vec<Issue>) {
    #[derive(Clone, Copy, PartialEq)]
    enum S {
        Idle,
        Probed,
        Held,
    }
    // (resource, side) -> state
    let mut state: HashMap<(&str, Side), S> = HashMap::new();
    let mut bad = |msg: String| issues.push(Issue { kind: IssueKind::UnpairedPipelineOp, message: msg });
    for n in body {
        let Node::Step(step) = n else { continue };
        use ScheduleStage as St;
        let (side, open, probe, close, closed) = match step.stage {
            St::ProducerTryAcquire | St::ProducerAcquire | St::ProducerCommit => {
                (Side::Producer, "acquire", "try_acquire", "commit", "committed")
            }
            St::ConsumerTryWait | St::ConsumerWait | St::ConsumerRelease => {
                (Side::Consumer, "wait", "try_wait", "release", "released")
            }
            _ => continue,
        };
        let r = step.resource.as_str();
        let op = step.stage.op_name().unwrap_or("?");
        if let Some((shared, members)) = groups.get(r) {
            if *shared != side || op != close {
                let want = if *shared == Side::Producer { "commit" } else { "release" };
                bad(format!("#{}: the pipeline group {r} takes only {want} (its members take {op})", step.id));
                continue;
            }
            let mut closed = 0;
            for m in members {
                if let Some(s) = state.get_mut(&(*m, side))
                    && *s == S::Held
                {
                    *s = S::Idle;
                    closed += 1;
                }
            }
            if closed == 0 {
                bad(format!(
                    "#{}: {r}.{op} closes no member: none of {} has a pending {open}",
                    step.id,
                    members.join(", ")
                ));
            }
            continue;
        }
        let s = state.entry((r, side)).or_insert(S::Idle);
        match (op, *s) {
            (o, S::Idle) if o == probe => *s = S::Probed,
            (o, S::Idle | S::Probed) if o == open => *s = S::Held,
            (o, S::Held) if o == close => *s = S::Idle,
            (o, S::Held) => bad(format!("#{}: {r}.{o} while the previous {open} is not yet {closed}", step.id)),
            (o, S::Probed) => bad(format!("#{}: {r}.{probe} is followed by {o}, not {open}", step.id)),
            (o, S::Idle) => bad(format!("#{}: {r}.{o} without a preceding {open}", step.id)),
        }
    }
    let mut left: Vec<_> = state.into_iter().filter(|(_, s)| *s != S::Idle).collect();
    left.sort_by_key(|((r, side), _)| (*r, *side == Side::Consumer));
    for ((r, side), s) in left {
        let (open, probe, closed) = if side == Side::Producer {
            ("acquire", "try_acquire", "committed")
        } else {
            ("wait", "try_wait", "released")
        };
        if s == S::Held {
            bad(format!("{r}.{open} is never {closed} in its block"));
        } else {
            bad(format!("{r}.{probe} is not followed by {open} in its block"));
        }
    }
    for n in body {
        if !n.body().is_empty() {
            check_pairing(n.body(), groups, issues);
        }
    }
}

fn contains_domain_loop(n: &Node) -> bool {
    matches!(n, Node::DomainLoop(_)) || n.body().iter().any(contains_domain_loop)
}

/// Give every step its phase: in the domain loop `Loop` (or the first- or
/// last-iteration variants under those guards); elsewhere `Head` before the
/// domain loop of its body and `Tail` after it, and `Loop` in a work-tile
/// loop body without a domain loop.
fn assign_phases(body: &mut [Node], in_work_tile: bool) {
    let has_loop = body.iter().any(contains_domain_loop);
    let mut seen_loop = false;
    for n in body.iter_mut() {
        let here = if !has_loop {
            if in_work_tile { Phase::Loop } else { Phase::Head }
        } else if seen_loop {
            Phase::Tail
        } else {
            Phase::Head
        };
        let had_loop = contains_domain_loop(n);
        match n {
            Node::Step(s) => s.phase = here,
            Node::DomainLoop(l) => set_loop_phases(&mut l.body, Phase::Loop),
            Node::WorkTileLoop(l) => assign_phases(&mut l.body, true),
            Node::Block(b) if had_loop => assign_phases(&mut b.body, in_work_tile),
            Node::Block(b) => set_flat(&mut b.body, here),
        }
        seen_loop |= had_loop;
    }
}

fn set_flat(body: &mut [Node], phase: Phase) {
    for n in body {
        match n {
            Node::Step(s) => s.phase = phase,
            other => {
                if let Some(b) = other.body_mut() {
                    set_flat(b, phase);
                }
            }
        }
    }
}

fn set_loop_phases(body: &mut [Node], phase: Phase) {
    for n in body {
        match n {
            Node::Step(s) => s.phase = phase,
            Node::Block(b) => {
                let p = match b.guard {
                    Guard::Iteration(IterationPredicate::FirstIter) => Phase::LoopFirstIter,
                    Guard::Iteration(IterationPredicate::LastIter) => Phase::LoopLastIter,
                    _ => phase,
                };
                set_loop_phases(&mut b.body, p);
            }
            other => {
                if let Some(b) = other.body_mut() {
                    set_loop_phases(b, phase);
                }
            }
        }
    }
}

impl DomainLoopScope<'_> {
    /// Record `body` as run on the first iteration only.
    pub fn first_iter(&self, body: impl FnOnce()) {
        self.b.when_true(IterationPredicate::FirstIter, body);
    }

    /// Record `body` as run on the last iteration only.
    pub fn last_iter(&self, body: impl FnOnce()) {
        self.b.when_true(IterationPredicate::LastIter, body);
    }

    /// Record `body` as run on iterations `start`, `start + period`, ...
    pub fn every(&self, period: u32, start: u32, body: impl FnOnce()) {
        self.b.when_true(IterationPredicate::Every { period, start }, body);
    }
}

impl WorkTileLoopScope<'_> {
    /// Record `body` as left out for the tiles the work queue skips (CuTe
    /// DSL's `wtl.skippable()`). Needs a `work_tile_loop_skipping`; does not
    /// nest; may not hold work queue steps.
    pub fn skippable(&self, body: impl FnOnce()) {
        if self.b.in_skippable() {
            self.b.issue(IssueKind::BadStructure, "skippable regions do not nest".into());
        }
        if self.b.skipping_loop() != Some(true) {
            self.b.issue(
                IssueKind::BadStructure,
                "skippable() needs a work_tile_loop with a skip predicate (work_tile_loop_skipping)".into(),
            );
        }
        self.b.with_frame(FrameKind::Block(Guard::Skippable, None), body);
    }
}

impl<'b> ResourceHandle<'b> {
    fn info(&self) -> &'b ResourceInfo {
        &self.b.resources[self.idx]
    }

    /// The resource's name.
    pub fn name(&self) -> &'b str {
        &self.info().name
    }

    #[track_caller]
    fn op(&self, stage: ScheduleStage) {
        self.b.record(self.idx, stage, None);
    }

    /// Record a producer probe.
    #[track_caller]
    pub fn try_acquire(&self) {
        self.op(ScheduleStage::ProducerTryAcquire);
    }

    /// Record a producer acquire.
    #[track_caller]
    pub fn acquire(&self) {
        self.op(ScheduleStage::ProducerAcquire);
    }

    /// Record a producer commit.
    #[track_caller]
    pub fn commit(&self) {
        self.op(ScheduleStage::ProducerCommit);
    }

    /// Record a consumer probe.
    #[track_caller]
    pub fn try_wait(&self) {
        self.op(ScheduleStage::ConsumerTryWait);
    }

    /// Record a consumer wait.
    #[track_caller]
    pub fn wait(&self) {
        self.op(ScheduleStage::ConsumerWait);
    }

    /// Record a consumer release.
    #[track_caller]
    pub fn release(&self) {
        self.op(ScheduleStage::ConsumerRelease);
    }

    /// Record a call of the work method `work` without inputs.
    #[track_caller]
    pub fn call(&self, work: &str) -> Outputs<'b> {
        self.call_with(work, std::iter::empty::<(&str, ArgSource)>())
    }

    /// Record a call of the work method `work` with named inputs.
    #[track_caller]
    pub fn call_with<'a, A: Into<ArgSource>>(
        &self,
        work: &str,
        args: impl IntoIterator<Item = (&'a str, A)>,
    ) -> Outputs<'b> {
        let info = self.info();
        let b = self.b;
        let empty = Flow {
            schedule: b.id,
            step: None,
            resource: info.name.clone(),
            label: work.to_string(),
            var: None,
            generation: 0,
        };
        let Some(sig) = info.works.iter().find(|w| w.name == work) else {
            let mut msg = format!("{} has no work method `{work}`", info.name);
            if info.variables.iter().any(|v| v.name() == work) {
                msg.push_str(&format!("; `{work}` is one of its variables, produced by the work that returns it"));
            } else if ScheduleStage::from_op_name(work).is_some() {
                msg.push_str("; call the pipeline operation's own method");
            } else if let Some(s) = did_you_mean(work, info.work_names()) {
                msg.push_str(&format!("; did you mean `{s}`?"));
            } else {
                let names: Vec<&str> = info.work_names().collect();
                msg.push_str(&format!(" (it has: {})", names.join(", ")));
            }
            b.issue(IssueKind::UnknownWork, msg);
            return Outputs { b, flows: vec![], empty };
        };
        let id = b.record(self.idx, sig.stage, Some(work));
        let site = format!("#{id} {}.{work}", info.name);
        let mut given: Vec<(String, ArgSource)> = args.into_iter().map(|(n, a)| (n.to_string(), a.into())).collect();
        let mut routes = vec![];
        for (name, type_label, type_id) in &sig.inputs {
            let Some(pos) = given.iter().position(|(n, _)| n == name) else {
                b.issue(IssueKind::UnproducedVariable, format!("{site}: no value for the input `{name}`"));
                continue;
            };
            let (_, src) = given.remove(pos);
            if let Some(route) = self.route(&site, name, type_label, *type_id, src) {
                routes.push(route);
            }
        }
        for (name, _) in &given {
            let mut msg = format!("{site} has no input `{name}`");
            if let Some(s) = did_you_mean(name, sig.inputs.iter().map(|(n, ..)| n.as_str())) {
                msg.push_str(&format!("; did you mean `{s}`?"));
            }
            b.issue(IssueKind::BadArgument, msg);
        }
        let mut flows = vec![];
        {
            let mut st = b.state.borrow_mut();
            for v in &sig.returns {
                let generation = st.next_generation;
                st.next_generation += 1;
                st.latest.insert(v.id(), (generation, id));
                flows.push(Flow { var: Some(v.clone()), step: Some(id), generation, ..empty.clone() });
            }
        }
        b.step_mut(id, |s| {
            s.inputs = routes;
            s.outputs = sig.returns.clone();
        });
        Outputs { b, flows, empty: Flow { step: Some(id), ..empty } }
    }

    fn route(&self, site: &str, name: &str, type_label: &str, type_id: TypeId, src: ArgSource) -> Option<Route> {
        let b = self.b;
        let constexpr = type_id == TypeId::of::<Constexpr>();
        if let ArgSource::Constexpr(v) = src {
            if !constexpr {
                b.issue(
                    IssueKind::BadArgument,
                    format!("{site}: the input `{name}` is a {type_label} variable, not a compile-time constant"),
                );
                return None;
            }
            return Some(Route { argument: name.to_string(), source: Source::Constexpr(v) });
        }
        if constexpr {
            b.issue(
                IssueKind::BadArgument,
                format!("{site}: the input `{name}` is a compile-time constant: give it with ArgSource::constexpr"),
            );
            return None;
        }
        let var = match &src {
            ArgSource::Flow(f) => {
                if f.schedule != b.id {
                    b.issue(
                        IssueKind::BadArgument,
                        format!("{site}: the input `{name}` is a handle from another schedule"),
                    );
                    return None;
                }
                match &f.var {
                    None => {
                        b.issue(
                            IssueKind::UnproducedVariable,
                            format!(
                                "{site}: the input `{name}` comes from {}.{}, which produces nothing",
                                f.resource, f.label
                            ),
                        );
                        return None;
                    }
                    Some(v) => v.clone(),
                }
            }
            ArgSource::Var(v) => v.clone(),
            ArgSource::Constexpr(_) => unreachable!("handled above"),
        };
        if var.type_id() != type_id {
            b.issue(
                IssueKind::BadArgument,
                format!(
                    "{site}: the input `{name}` is a {type_label}, but `{}` holds a {}",
                    var.name(),
                    var.type_label()
                ),
            );
        }
        let source = match src {
            ArgSource::Flow(f) => {
                b.check_fresh(&f, &var, site);
                let step = f.step.expect("a handle with a variable has a step");
                b.state.borrow_mut().used.insert((step, var.id()));
                Source::Step { step, variable: var.name().to_string() }
            }
            ArgSource::Var(v) => {
                let produced = b.state.borrow().latest.get(&v.id()).map(|(_, s)| *s);
                if let Some(step) = produced {
                    b.state.borrow_mut().used.insert((step, v.id()));
                    Source::Step { step, variable: v.name().to_string() }
                } else {
                    match b.resources.iter().find(|r| r.variables.contains(&v)) {
                        Some(r) if r.pipelined => {
                            Source::Peer { resource: r.name.clone(), variable: v.name().to_string() }
                        }
                        owner => {
                            let whose = owner.map_or(String::new(), |r| format!(" of {}", r.name));
                            b.issue(
                                IssueKind::UnproducedVariable,
                                format!(
                                    "{site}: the variable `{}`{whose} is read before any step produces it, and no \
                                     pipeline peer mirrors it",
                                    v.name()
                                ),
                            );
                            return None;
                        }
                    }
                }
            }
            ArgSource::Constexpr(_) => unreachable!("handled above"),
        };
        Some(Route { argument: name.to_string(), source })
    }
}

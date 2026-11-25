// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The checks follow the rules CuTe DSL 4.8.1's TaskManager enforces
// (cutlass.experimental.task_scheduling.task_manager, NVIDIA EULA); the code
// and the messages are written independently.

//! The TaskManager's static checks: tasks and warp ranges, resource roles,
//! the dependency graph, pipeline sides, balance and thread counts,
//! variable flow and programmatic-dependent-launch ordering.

use std::collections::{HashMap, HashSet};

use crate::pipeline::{Agent, BarrierParam};

use super::enums::{Guard, IterationPredicate, PipelineType, ScheduleStage, SignalingThreads};
use super::manager::{TaskManager, VerifyError, VerifyIssue, VerifyIssueKind as K, at, side_of, walk_steps};
use super::pdl::PdlRole;
use super::resource::{Resource, Side};
use super::schedule::{DomainLoop, Node, Source, Step};
use super::task::Task;

struct Checker<'a> {
    m: &'a TaskManager,
    issues: Vec<VerifyIssue>,
}

impl Checker<'_> {
    fn bad(&mut self, kind: K, message: String) {
        let issue = VerifyIssue { kind, message };
        if !self.issues.contains(&issue) {
            self.issues.push(issue);
        }
    }
}

fn is_pipeline_op(stage: ScheduleStage) -> bool {
    stage.op_name().is_some()
}

/// The constant trip count of a domain loop, if its bounds are constants.
fn trip_count(l: &DomainLoop) -> Option<i64> {
    let (s, e, st) = (l.start.as_const()?, l.end.as_const()?, l.step.as_const()?);
    (st > 0).then(|| ((e - s).max(0) + st - 1) / st)
}

/// How often a step at the end of `path` runs, as text that is equal for
/// two sites exactly when they run equally often (as far as can be told
/// at build time).
fn site(path: &[&Node]) -> String {
    let parts: Vec<String> = path
        .iter()
        .map(|n| match n {
            Node::WorkTileLoop(l) => format!("work_tile_loop({})", l.work_queue),
            Node::DomainLoop(l) => match trip_count(l) {
                Some(n) => format!("domain_loop of {n} iterations"),
                None => format!("domain_loop({}, {}, {})", l.start, l.end, l.step),
            },
            Node::Block(b) => b.guard.to_string(),
            Node::Step(_) => String::new(),
        })
        .collect();
    if parts.is_empty() { "the top level".into() } else { parts.join(" > ") }
}

impl TaskManager {
    /// Check the tasks, their schedules and the dependency graph without
    /// emitting anything (CuTe DSL's checks before emission): every problem
    /// is collected into one [`VerifyError`].
    ///
    /// * tasks have unique names and disjoint warp ranges, and a register
    ///   budget only on whole warpgroups, or on tasks that together cover
    ///   whole warpgroups with one budget (a 1-warp load task, a 1-warp MMA
    ///   task and a 2-warp padding task, say);
    /// * every resource a task names is given to the manager (so its
    ///   barriers are initialised in the prologue), producer-side steps are
    ///   on `dst_resources` and consumer-side ones on `src_resources`, and
    ///   every declared resource is used;
    /// * the graph names only known resources, is acyclic (self-edges
    ///   aside), each edge is backed by a task consuming the upstream and
    ///   producing the downstream (or, for a barrier, mentioning both),
    ///   every task resource appears in it, and no launch barrier is
    ///   downstream;
    /// * each pipelined resource has one producing task and at least one
    ///   consuming task (one, unless it is a work queue); its commits and
    ///   releases sit in loop nests and guards that run equally often
    ///   (constant trip counts compared by value, run-time ones by name);
    ///   and a side whose threads arrive on the barrier has a cooperative
    ///   group of exactly its tasks' threads;
    /// * a variable read by a step on another resource flows along a data
    ///   edge, and every variable a task stores has a default;
    /// * a task producing (or directly reading an unpipelined) resource
    ///   that depends on a [`PdlWaitBarrier`](super::PdlWaitBarrier) runs
    ///   `wait_griddep` before that work on every path, and every
    ///   `launch_griddep` follows a `wait_griddep` on every path, in its
    ///   own task or in one it waits on through a pipeline whose producer
    ///   had waited before committing.
    pub fn verify(&self) -> Result<(), VerifyError> {
        let mut c = Checker { m: self, issues: vec![] };
        c.tasks();
        c.roles();
        c.graph();
        c.pipelines();
        c.variables();
        c.pdl();
        c.groups();
        c.memory();
        if c.issues.is_empty() { Ok(()) } else { Err(VerifyError { issues: c.issues }) }
    }
}

impl Checker<'_> {
    fn groups(&mut self) {
        let m = self.m;
        let mut owner: HashMap<&str, &str> = HashMap::new();
        for r in m.resource_list() {
            let Some(g) = r.as_group() else { continue };
            let gname = r.name();
            for p in g.derivation_problems() {
                self.bad(K::BadGroup, p.clone());
            }
            for p in g.arrival_count_problems() {
                self.bad(K::GroupArrivalCount, p);
            }
            for member in g.member_names() {
                if m.find_resource(member).is_none() {
                    self.bad(
                        K::BadGroup,
                        format!("group `{gname}`: its member `{member}` is not given to the TaskManager"),
                    );
                }
                if let Some(other) = owner.insert(member, gname) {
                    self.bad(K::BadGroup, format!("`{member}` belongs to two groups, `{other}` and `{gname}`"));
                }
            }
            let placed = m.smem_allocator_ref().is_some_and(|a| a.places_barriers_of(gname));
            if !placed && !g.has_barrier_ptr() {
                self.bad(
                    K::BarrierStorage,
                    format!(
                        "group `{gname}` has no barrier storage: register it with the SmemAllocator \
                         (add_pipeline_group) or give its configuration a barrier_ptr"
                    ),
                );
            }
            let (close, what) = match g.shared_side() {
                Side::Consumer => (ScheduleStage::ConsumerRelease, "release"),
                Side::Producer => (ScheduleStage::ProducerCommit, "commit"),
            };
            for t in m.tasks() {
                walk_steps(&t.schedule().body, &mut |s, _| {
                    if s.stage == close && g.has_member(&s.resource) {
                        self.issues.push(VerifyIssue {
                            kind: K::BadGroup,
                            message: format!(
                                "task {}: #{} {}.{what}{}: `{}` shares its barriers in group `{gname}`; {what} \
                                 through the group, once for every member",
                                t.name(),
                                s.id,
                                s.resource,
                                at(s),
                                s.resource
                            ),
                        });
                    }
                });
            }
        }
    }

    fn memory(&mut self) {
        let m = self.m;
        if let Some(t) = m.tmem_allocator_ref()
            && let Err(e) = t.plan()
        {
            self.bad(K::TmemCapacity, format!("TMEM: {e}"));
        }
        let Some(alloc) = m.smem_allocator_ref() else { return };
        let layout = match alloc.plan() {
            Ok(l) => Some(l),
            Err(e) => {
                self.bad(K::SmemLayout, format!("shared memory: {e}"));
                None
            }
        };
        let grouped = |n: &str| m.resource_list().any(|r| r.as_group().is_some_and(|g| g.has_member(n)));
        for r in m.resource_list() {
            let Some(cfg) = r.pipeline_config() else { continue };
            if r.as_group().is_some() || grouped(r.name()) || cfg.barrier_ptr.is_some() {
                continue;
            }
            if !alloc.places_barriers_of(r.name()) {
                self.bad(
                    K::BarrierStorage,
                    format!(
                        "`{}` is pipelined but the SmemAllocator does not place its barriers, so it would be left \
                         without barrier storage in the unified block; add_resource it",
                        r.name()
                    ),
                );
            }
        }
        // Allocations sharing bytes need owners the graph orders by
        // control edges (one's use over before the other's begins).
        let mut ctrl: HashMap<&str, Vec<&str>> = HashMap::new();
        for (down, ups) in m.graph().nodes() {
            for e in ups.iter().filter(|e| e.is_control()) {
                ctrl.entry(e.upstream()).or_default().push(down.as_str());
            }
        }
        let reaches = |from: &str, to: &str| {
            let mut stack = vec![from];
            let mut seen = HashSet::new();
            while let Some(n) = stack.pop() {
                for &d in ctrl.get(n).into_iter().flatten() {
                    if d == to {
                        return true;
                    }
                    if seen.insert(d) {
                        stack.push(d);
                    }
                }
            }
            false
        };
        let find = |a: &crate::task_scheduling::SmemAllocation| {
            layout.as_ref().and_then(|l| {
                l.regions.iter().find(|r| r.name == a.name && r.alias_group.is_some()).map(|r| (r.offset, r.end()))
            })
        };
        for group in alloc.alias_groups() {
            for (pi, pa) in group.iter().enumerate() {
                for pb in &group[pi + 1..] {
                    for a in pa {
                        for b in pb {
                            let (Some((a0, a1)), Some((b0, b1))) = (find(a), find(b)) else { continue };
                            let (lo, hi) = (a0.max(b0), a1.min(b1));
                            if lo >= hi {
                                continue;
                            }
                            let (oa, ob) = (alloc.owner_of(a), alloc.owner_of(b));
                            let ordered = match (&oa, &ob) {
                                (Some(x), Some(y)) if x != y => reaches(x, y) || reaches(y, x),
                                _ => false,
                            };
                            if !ordered {
                                let who = |o: &Option<String>| match o {
                                    Some(o) => format!("`{o}`"),
                                    None => "no resource".to_string(),
                                };
                                self.bad(
                                    K::SmemAlias,
                                    format!(
                                        "shared memory bytes [{lo}, {hi}) hold both `{}` (of {}) and `{}` (of {}), \
                                         but the dependency graph has no control-edge path ordering {} and {}, so \
                                         their live ranges may overlap",
                                        a.name,
                                        who(&oa),
                                        b.name,
                                        who(&ob),
                                        who(&oa),
                                        who(&ob)
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn tasks(&mut self) {
        let m = self.m;
        let mut names = HashSet::new();
        for t in m.tasks() {
            if t.name().is_empty() || t.name().contains('/') || t.name().starts_with('#') {
                self.bad(K::BadTask, format!("`{}` cannot name a task (empty, or a '/' or leading '#')", t.name()));
            }
            if !names.insert(t.name()) {
                self.bad(K::BadTask, format!("two tasks are called `{}`", t.name()));
            }
            if t.num_warps() == 0 {
                self.bad(K::BadTask, format!("task {} has no warps", t.name()));
            }
            if let Some(rb) = t.registers() {
                let n = rb.count();
                if !(24..=256).contains(&n) || n % 8 != 0 {
                    self.bad(
                        K::BadTask,
                        format!("task {}: {n} registers is not a multiple of 8 from 24 to 256", t.name()),
                    );
                }
                if (t.warp_idx() % 4 != 0 || t.num_warps() % 4 != 0) && !super::manager::shares_budget(m.tasks(), t) {
                    self.bad(
                        K::BadTask,
                        format!(
                            "task {}: a register budget applies to whole warpgroups, but the task has warps {}..{}",
                            t.name(),
                            t.warp_idx(),
                            t.warp_idx() + t.num_warps()
                        ),
                    );
                }
            }
        }
        let tasks = m.tasks();
        for (i, a) in tasks.iter().enumerate() {
            for b in &tasks[i + 1..] {
                let (a0, a1) = (a.warp_idx(), a.warp_idx() + a.num_warps());
                let (b0, b1) = (b.warp_idx(), b.warp_idx() + b.num_warps());
                if a0 < b1 && b0 < a1 {
                    self.bad(K::BadTask, format!("tasks {} and {} share warps", a.name(), b.name()));
                }
            }
        }
        let mut seen = HashSet::new();
        for r in m.resource_list() {
            if !seen.insert(r.name()) {
                self.bad(K::BadTask, format!("two resources given to the TaskManager are called `{}`", r.name()));
            }
        }
    }

    fn roles(&mut self) {
        let m = self.m;
        for t in m.tasks() {
            let mut used = HashSet::new();
            for name in
                t.src_resources().iter().chain(t.dst_resources()).chain(t.schedule().resources.iter().map(|r| &r.name))
            {
                if m.find_resource(name).is_none() {
                    self.bad(
                        K::BarrierInit,
                        format!(
                            "task {}: the resource `{name}` is not given to the TaskManager, so it would be used \
                             without being created (its barriers never initialised)",
                            t.name()
                        ),
                    );
                }
            }
            walk_steps(&t.schedule().body, &mut |s, _| {
                used.insert(s.resource.clone());
                let group = m.find_resource(&s.resource).and_then(|r| r.as_group());
                let any = |f: &dyn Fn(&str) -> bool| {
                    f(&s.resource) || group.is_some_and(|g| g.member_names().iter().any(|x| f(x)))
                };
                let (ok, role) = match side_of(s.stage) {
                    Side::Producer => (any(&|x| t.produces(x)), "dst_resources"),
                    Side::Consumer => (any(&|x| t.consumes(x)), "src_resources"),
                };
                if !ok {
                    let what = s.label.as_deref().or(s.stage.op_name()).unwrap_or("?");
                    self.issues.push(VerifyIssue {
                        kind: K::RoleMismatch,
                        message: format!(
                            "task {}: #{} {}.{what}{} is a {:?}-side step, but `{}` is not in its {role}",
                            t.name(),
                            s.id,
                            s.resource,
                            at(s),
                            side_of(s.stage),
                            s.resource
                        ),
                    });
                }
            });
            for name in t.src_resources().iter().chain(t.dst_resources()) {
                if !used.contains(name) {
                    self.bad(
                        K::RoleMismatch,
                        format!("task {}: `{name}` is declared but no step of its schedule uses it", t.name()),
                    );
                }
            }
        }
    }

    fn graph(&mut self) {
        let m = self.m;
        let g = m.graph();
        let known = |n: &str| m.tasks().iter().any(|t| t.consumes(n) || t.produces(n));
        let barrier = |n: &str| m.find_resource(n).is_some_and(|r| r.is_barrier());
        for (down, ups) in g.nodes() {
            if !known(down) {
                self.bad(K::BadGraph, format!("the dependency graph names `{down}`, which no task uses"));
            }
            if m.find_resource(down).and_then(|r| r.pdl_role()) == Some(PdlRole::Launch) {
                self.bad(
                    K::BadGraph,
                    format!("`{down}` is a launch barrier and cannot be a downstream resource of the graph"),
                );
            }
            for e in ups {
                let up = e.upstream();
                if !known(up) {
                    self.bad(
                        K::BadGraph,
                        format!("the dependency graph names `{up}` (upstream of `{down}`), which no task uses"),
                    );
                    continue;
                }
                if up == down {
                    let src = m.tasks().iter().any(|t| t.consumes(up));
                    let dst = m.tasks().iter().any(|t| t.produces(up));
                    if !(src && dst) {
                        self.bad(
                            K::BadGraph,
                            format!("the self-edge on `{up}` needs a task consuming it and a task producing it"),
                        );
                    }
                    continue;
                }
                let strict = m.tasks().iter().any(|t| t.consumes(up) && t.produces(down));
                let relaxed = (barrier(up) || barrier(down))
                    && m.tasks()
                        .iter()
                        .any(|t| (t.consumes(up) || t.produces(up)) && (t.consumes(down) || t.produces(down)));
                if !strict && !relaxed {
                    self.bad(
                        K::BadGraph,
                        format!("no task consumes `{up}` and produces `{down}`, so nothing backs the edge `{up}` -> `{down}`"),
                    );
                }
            }
        }
        let mentioned = g.resources();
        for t in m.tasks() {
            for r in t.src_resources().iter().chain(t.dst_resources()) {
                // A launch barrier is never downstream and orders nothing
                // as an upstream, so the graph need not mention it.
                let launch = m.find_resource(r).and_then(|r| r.pdl_role()) == Some(PdlRole::Launch);
                if !launch && !mentioned.contains(r.as_str()) {
                    self.bad(K::BadGraph, format!("task {}: `{r}` does not appear in the dependency graph", t.name()));
                }
            }
        }
        // Cycles (self-edges aside): depth-first over downstream -> upstream.
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for (down, ups) in g.nodes() {
            for e in ups {
                if e.upstream() != down {
                    adj.entry(down.as_str()).or_default().push(e.upstream());
                }
            }
        }
        fn dfs<'g>(
            n: &'g str,
            adj: &HashMap<&'g str, Vec<&'g str>>,
            color: &mut HashMap<&'g str, u8>,
            stack: &mut Vec<&'g str>,
        ) -> Option<Vec<&'g str>> {
            color.insert(n, 1);
            stack.push(n);
            for &u in adj.get(n).into_iter().flatten() {
                match color.get(u).copied().unwrap_or(0) {
                    1 => {
                        let at = stack.iter().position(|s| *s == u).unwrap_or(0);
                        let mut cycle = stack[at..].to_vec();
                        cycle.push(u);
                        return Some(cycle);
                    }
                    0 => {
                        if let Some(c) = dfs(u, adj, color, stack) {
                            return Some(c);
                        }
                    }
                    _ => {}
                }
            }
            stack.pop();
            color.insert(n, 2);
            None
        }
        let mut color = HashMap::new();
        for (down, _) in g.nodes() {
            if color.get(down.as_str()).copied().unwrap_or(0) == 0
                && let Some(cycle) = dfs(down, &adj, &mut color, &mut vec![])
            {
                self.bad(K::CyclicGraph, format!("the dependency graph has a cycle: {}", cycle.join(" -> ")));
                break;
            }
        }
    }

    fn pipelines(&mut self) {
        let m = self.m;
        for r in m.resource_list() {
            let Some(cfg) = r.pipeline_config() else { continue };
            if r.as_group().is_some() {
                continue;
            }
            let name = r.name();
            // The group whose release (commit) closes this member's stages.
            let group = m.resource_list().find_map(|g| g.as_group().filter(|g| g.has_member(name)));
            let via_group = |s: &Step| {
                group.is_some_and(|g| {
                    s.resource == g.name()
                        && side_of(s.stage) == g.shared_side()
                        && matches!(s.stage, ScheduleStage::ProducerCommit | ScheduleStage::ConsumerRelease)
                })
            };
            // (task, sites of the closing operation)
            let mut producers: Vec<(&Task, Vec<String>)> = vec![];
            let mut consumers: Vec<(&Task, Vec<String>)> = vec![];
            for t in m.tasks() {
                let (mut p, mut c, mut p_sites, mut c_sites) = (false, false, vec![], vec![]);
                walk_steps(&t.schedule().body, &mut |s, path| {
                    if (s.resource != name && !via_group(s)) || !is_pipeline_op(s.stage) {
                        return;
                    }
                    match side_of(s.stage) {
                        Side::Producer => p = true,
                        Side::Consumer => c = true,
                    }
                    match s.stage {
                        ScheduleStage::ProducerCommit => p_sites.push(site(path)),
                        ScheduleStage::ConsumerRelease => c_sites.push(site(path)),
                        _ => {}
                    }
                });
                if p {
                    p_sites.sort();
                    producers.push((t, p_sites));
                }
                if c {
                    c_sites.sort();
                    consumers.push((t, c_sites));
                }
            }
            let names = |v: &[(&Task, Vec<String>)]| v.iter().map(|(t, _)| t.name()).collect::<Vec<_>>().join(", ");
            if producers.len() != 1 {
                let msg = if producers.is_empty() {
                    format!("the pipelined resource `{name}` has no producing task (no task acquires and commits it)")
                } else {
                    format!("the pipelined resource `{name}` has several producing tasks ({})", names(&producers))
                };
                self.bad(K::PipelineSides, msg);
            }
            if consumers.is_empty() || (consumers.len() > 1 && !r.is_work_queue()) {
                let msg = if consumers.is_empty() {
                    format!("the pipelined resource `{name}` has no consuming task (no task waits on and releases it)")
                } else {
                    format!("the pipelined resource `{name}` has several consuming tasks ({})", names(&consumers))
                };
                self.bad(K::PipelineSides, msg);
            }
            if let Some((pt, p_sites)) = producers.first() {
                for (ct, c_sites) in &consumers {
                    if p_sites != c_sites {
                        self.bad(
                            K::UnbalancedPipeline,
                            format!(
                                "`{name}`: task {} commits at [{}] but task {} releases at [{}]; the two sides would \
                                 not hand over the same number of stages",
                                pt.name(),
                                p_sites.join("; "),
                                ct.name(),
                                c_sites.join("; ")
                            ),
                        );
                    }
                }
            }
            // Thread counts of the sides whose threads arrive on the barrier.
            use PipelineType as P;
            let thread_producers = matches!(cfg.pipeline_type, P::AsyncAsync | P::AsyncUmma);
            let thread_consumers =
                matches!(cfg.pipeline_type, P::AsyncAsync | P::TmaAsync | P::UmmaAsync | P::ClcFetchAsync);
            let sides = [
                (thread_producers, &cfg.producer_group, cfg.producer_signaling_threads, &producers, "producer"),
                (thread_consumers, &cfg.consumer_group, cfg.consumer_signaling_threads, &consumers, "consumer"),
            ];
            for (threads_arrive, group, signaling, tasks, side) in sides {
                if !threads_arrive || signaling != SignalingThreads::ALL || tasks.is_empty() {
                    continue;
                }
                if !matches!(group.agent(), Agent::Thread | Agent::Warp) {
                    continue;
                }
                let BarrierParam::Static(want) = group.thread_arrive_count() else { continue };
                let have: u32 = tasks.iter().map(|(t, _)| t.num_threads()).sum();
                // A TMA pipeline's consumers release from lane 0 of each
                // warp (a count of warps) or from every thread.
                let warps: u32 = tasks.iter().map(|(t, _)| t.num_warps()).sum();
                let per_warp = side == "consumer" && cfg.pipeline_type == P::TmaAsync && want == warps as i64;
                if want != have as i64 && !per_warp {
                    self.bad(
                        K::ThreadCount,
                        format!(
                            "`{name}`: the {side} cooperative group counts {want} threads, but the {side} tasks ({}) \
                             have {have}",
                            names(tasks)
                        ),
                    );
                }
            }
        }
    }

    fn variables(&mut self) {
        let m = self.m;
        let g = m.graph();
        for t in m.tasks() {
            let steps: HashMap<usize, &Step> = t.schedule().steps().into_iter().map(|s| (s.id, s)).collect();
            walk_steps(&t.schedule().body, &mut |s, _| {
                for v in &s.outputs {
                    if !v.has_default() {
                        self.issues.push(VerifyIssue {
                            kind: K::VariableFlow,
                            message: format!(
                                "task {}: `{}` (returned by #{} {}) has no default, so the task cannot make its \
                                 storage; give it one",
                                t.name(),
                                v.name(),
                                s.id,
                                s.resource
                            ),
                        });
                    }
                }
                for route in &s.inputs {
                    let Source::Step { step, variable } = &route.source else { continue };
                    let Some(from) = steps.get(step) else { continue };
                    if from.resource == s.resource {
                        continue;
                    }
                    let data = g.upstreams(&s.resource).any(|e| {
                        e.upstream() == from.resource
                            && !e.is_control()
                            && m.find_resource(&from.resource).and_then(|r| r.pdl_role()).is_none()
                    });
                    if !data {
                        self.issues.push(VerifyIssue {
                            kind: K::VariableFlow,
                            message: format!(
                                "task {}: `{variable}` flows from #{} {} to #{} {}{}, but the dependency graph has no \
                                 data edge `{}` -> `{}`",
                                t.name(),
                                from.id,
                                from.resource,
                                s.id,
                                s.resource,
                                at(s),
                                from.resource,
                                s.resource
                            ),
                        });
                    }
                }
            });
        }
    }

    fn pdl(&mut self) {
        let m = self.m;
        let role = |n: &str| m.find_resource(n).and_then(|r| r.pdl_role());
        if !m.resource_list().any(|r| r.pdl_role().is_some()) {
            return;
        }
        // Resources depending on a wait barrier.
        let dependent: HashSet<&str> = m
            .graph()
            .nodes()
            .iter()
            .filter(|(_, ups)| ups.iter().any(|e| role(e.upstream()) == Some(PdlRole::Wait)))
            .map(|(d, _)| d.as_str())
            .collect();
        // Pipelined resources whose producer has waited before every commit:
        // a wait on them orders the waiter after a wait_griddep too.
        let mut safe: HashSet<String> = HashSet::new();
        loop {
            let mut pass =
                Pdl { m, dependent: &dependent, safe: &safe, issues: vec![], commits: HashMap::new(), task: None };
            for t in m.tasks() {
                pass.task = Some(t);
                pass.nodes(&t.schedule().body, State::default(), None);
            }
            let next: HashSet<String> = pass.commits.iter().filter(|(_, ok)| **ok).map(|(r, _)| r.clone()).collect();
            if next == safe {
                for i in pass.issues {
                    self.bad(i.kind, i.message);
                }
                break;
            }
            safe = next;
        }
    }
}

/// What a task has done on the path being analysed.
#[derive(Clone, Copy, Default)]
struct State {
    /// It ran `wait_griddep` itself.
    own: bool,
    /// A `wait_griddep` happened before (its own, or one ordered before
    /// through a pipeline).
    any: bool,
}

struct Pdl<'a> {
    m: &'a TaskManager,
    dependent: &'a HashSet<&'a str>,
    safe: &'a HashSet<String>,
    issues: Vec<VerifyIssue>,
    /// Resource -> whether every commit so far followed a wait.
    commits: HashMap<String, bool>,
    task: Option<&'a Task>,
}

impl<'a> Pdl<'a> {
    /// Analyse the first iteration of every loop (a wait only ever turns
    /// the state on, so the first iteration is the worst case). Loops with
    /// run-time bounds are assumed to run at least once.
    fn nodes(&mut self, nodes: &'a [Node], mut st: State, trip: Option<i64>) -> State {
        for n in nodes {
            st = match n {
                Node::Step(s) => self.step(s, st),
                Node::Block(b) => {
                    let first = match b.guard {
                        Guard::Iteration(IterationPredicate::FirstIter) => true,
                        Guard::Iteration(IterationPredicate::Every { start: 0, .. }) => true,
                        Guard::Iteration(IterationPredicate::LastIter) => trip == Some(1),
                        _ => false,
                    };
                    let after = self.nodes(&b.body, st, trip);
                    if first { after } else { st }
                }
                Node::DomainLoop(l) => match trip_count(l) {
                    Some(0) => st,
                    t => self.nodes(&l.body, st, t),
                },
                Node::WorkTileLoop(l) => self.nodes(&l.body, st, None),
            };
        }
        st
    }

    fn step(&mut self, s: &'a Step, mut st: State) -> State {
        let m = self.m;
        let t = self.task.expect("a task");
        let r = m.find_resource(&s.resource);
        let role = r.and_then(|r| r.pdl_role());
        let what = s.label.as_deref().or(s.stage.op_name()).unwrap_or("?");
        if role == Some(PdlRole::Wait) && s.stage.is_work() {
            st.own = true;
            st.any = true;
        }
        if s.stage == ScheduleStage::ConsumerWait && self.safe.contains(&s.resource) {
            st.any = true;
        }
        if role == Some(PdlRole::Launch) && s.stage.is_work() && !st.any {
            self.issues.push(VerifyIssue {
                kind: K::PdlLaunchBeforeWait,
                message: format!(
                    "task {}: #{} {}.{what}{} can run before any wait_griddep (in this task, or ordered before it \
                     through a pipeline)",
                    t.name(),
                    s.id,
                    s.resource,
                    at(s)
                ),
            });
        }
        if self.dependent.contains(s.resource.as_str()) && s.stage.is_work() && !st.own {
            let pipelined = r.is_some_and(|r| r.pipeline_config().is_some());
            let checked = match side_of(s.stage) {
                Side::Producer => t.produces(&s.resource),
                Side::Consumer => t.consumes(&s.resource) && !pipelined,
            };
            if checked {
                self.issues.push(VerifyIssue {
                    kind: K::PdlOrdering,
                    message: format!(
                        "task {}: #{} {}.{what}{} works on `{}`, which depends on the previous grid, before the task \
                         runs wait_griddep",
                        t.name(),
                        s.id,
                        s.resource,
                        at(s),
                        s.resource
                    ),
                });
            }
        }
        if s.stage == ScheduleStage::ProducerCommit {
            let e = self.commits.entry(s.resource.clone()).or_insert(true);
            *e &= st.any;
        }
        st
    }
}

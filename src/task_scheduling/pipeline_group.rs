// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Names and semantics follow CuTe DSL 4.8.1's
// cutlass.experimental.task_scheduling.pipeline_group (PipelineGroup,
// PipelineGroupMode; NVIDIA EULA); the code, the checks and the messages
// are written independently.

//! Pipeline groups: several pipelined resources sharing one side's
//! barriers, so that a consumer releases (or a producer commits) all of
//! them with one arrival.

use crate::pipeline::{Agent, CooperativeGroup, MbarrierArray};
use crate::tensor::Ptr;
use crate::value::Boolean;

use super::config::PipelineConfig;
use super::enums::{PipelineGroupMode, PipelineType};
use super::pipeline::{AgentKind, TsPipeline, empty_op, full_op, make_barriers, task_warp_leader};
use super::resource::{MemoryResource, Resource, ResourcePipeline, Side};

/// Pipelined resources ("members") that share barriers (CuTe DSL's
/// `PipelineGroup`), in one of three [`PipelineGroupMode`]s. With `S`
/// stages and `N` members the group's barrier block holds:
///
/// * `Merge` (`N` producers, one consumer): `N` full blocks of `S` (one per
///   member) then one shared empty block; the consumer waits on each member
///   and releases them all with the group's `release`;
/// * `Fork` (one producer, `N` consumers): one shared full block then `N`
///   empty blocks; the producer acquires each member (only the first arms
///   a TMA stage, with every member's bytes) and commits them all with the
///   group's `commit`;
/// * `FusedMerge`: one shared full block, counting every member's producer
///   arrivals and bytes, then one shared empty block; the consumer waits on
///   any member and releases through the group.
///
/// Its barriers come from the [`SmemAllocator`](super::SmemAllocator)
/// (register the group with it) or from the group configuration's
/// `barrier_ptr`. The group's own configuration is derived from the
/// members' (stages must agree; the shared side's cooperative group is the
/// members' common one, or for a fused merge the sum of the producers');
/// [`with_pipeline_config`](Self::with_pipeline_config) gives it
/// explicitly, and the TaskManager then checks it against the members.
/// The group is itself a [`Resource`] (a barrier-only one): give it to the
/// TaskManager with its members and pass it to the schedule that releases
/// (or commits) through it.
///
/// Members of a merge are consumed through one barrier, so the members'
/// thread-consumer conventions must agree: when a TMA member is merged with
/// thread-produced ones, every consumer thread releases (the TMA member's
/// consumer group may count warps or threads); a merge of TMA members only
/// releases from lane 0 of each consumer warp, unless the consumer group
/// counts the consuming task's threads.
pub struct PipelineGroup {
    memory: MemoryResource,
    mode: PipelineGroupMode,
    members: Vec<(String, Option<PipelineConfig>)>,
    member_names: Vec<String>,
    explicit: bool,
    problems: Vec<String>,
    shared: Option<TsPipeline>,
}

impl std::fmt::Debug for PipelineGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineGroup")
            .field("name", &self.memory.name())
            .field("mode", &self.mode)
            .field("members", &self.member_names)
            .finish()
    }
}

/// The static thread count of a group, if it has one.
fn threads(g: &CooperativeGroup) -> Option<i64> {
    match g.agent() {
        Agent::Thread | Agent::Warp => g.thread_arrive_count().as_static(),
        _ => None,
    }
}

impl PipelineGroup {
    /// The group `name` over `members` (in order) in `mode`.
    pub fn new(name: &str, members: &[&dyn Resource], mode: PipelineGroupMode) -> PipelineGroup {
        let members: Vec<(String, Option<PipelineConfig>)> =
            members.iter().map(|m| (m.name().to_string(), m.pipeline_config().cloned())).collect();
        let member_names = members.iter().map(|(n, _)| n.clone()).collect();
        let mut g = PipelineGroup {
            memory: MemoryResource::new(name),
            mode,
            members,
            member_names,
            explicit: false,
            problems: vec![],
            shared: None,
        };
        g.problems = g.structural_problems();
        if let Some(cfg) = g.derive() {
            g.memory = MemoryResource::new(name).with_pipeline_config(cfg);
        }
        g
    }

    /// With the group's configuration given explicitly (its cooperative
    /// groups and bytes are the arrival counts of the shared barriers).
    pub fn with_pipeline_config(mut self, cfg: PipelineConfig) -> PipelineGroup {
        let name = self.memory.name().to_string();
        self.memory = MemoryResource::new(&name).with_pipeline_config(cfg);
        self.explicit = true;
        self
    }

    /// Whether its configuration was given rather than derived.
    pub fn is_explicit(&self) -> bool {
        self.explicit
    }

    /// The mode.
    pub fn mode(&self) -> PipelineGroupMode {
        self.mode
    }

    /// The members' names, in order.
    pub fn member_names(&self) -> &[String] {
        &self.member_names
    }

    /// Whether `resource` is a member.
    pub fn has_member(&self, resource: &str) -> bool {
        self.member_names.iter().any(|m| m == resource)
    }

    /// The side whose barriers the members share and whose closing
    /// operation goes through the group: the consumers' release for a
    /// merge, the producer's commit for a fork.
    pub fn shared_side(&self) -> Side {
        match self.mode {
            PipelineGroupMode::Fork => Side::Producer,
            _ => Side::Consumer,
        }
    }

    /// Distinct mbarriers per stage (CuTe DSL's `num_barriers_per_stage`):
    /// `N + 1`, or 2 for a fused merge.
    pub fn num_barriers_per_stage(&self) -> usize {
        match self.mode {
            PipelineGroupMode::FusedMerge => 2,
            _ => self.members.len() + 1,
        }
    }

    /// The mbarriers of the group's block.
    pub fn num_barriers(&self) -> usize {
        self.num_barriers_per_stage() * self.stages()
    }

    fn stages(&self) -> usize {
        self.members.iter().find_map(|(_, c)| c.as_ref().map(|c| c.num_stages)).unwrap_or(1)
    }

    fn cfgs(&self) -> impl Iterator<Item = (&String, &PipelineConfig)> {
        self.members.iter().filter_map(|(n, c)| c.as_ref().map(|c| (n, c)))
    }

    fn mixed_tma_merge(&self) -> bool {
        let tma = self.cfgs().filter(|(_, c)| c.pipeline_type == PipelineType::TmaAsync).count();
        tma > 0 && tma < self.cfgs().count()
    }

    /// Problems with the members themselves (reported by the TaskManager's
    /// checks as [`VerifyIssueKind::BadGroup`](super::VerifyIssueKind)).
    pub fn structural_problems(&self) -> Vec<String> {
        let name = self.memory.name();
        let mut out = vec![];
        if self.members.len() < 2 {
            out.push(format!("group `{name}` needs at least two members"));
        }
        for (m, c) in &self.members {
            if c.is_none() {
                out.push(format!("group `{name}`: the member `{m}` has no pipeline configuration"));
            }
        }
        let cfgs: Vec<(&String, &PipelineConfig)> = self.cfgs().collect();
        let Some((first, f)) = cfgs.first().copied() else { return out };
        for (m, c) in &cfgs {
            if c.pipeline_type == PipelineType::ClcFetchAsync {
                out.push(format!(
                    "group `{name}`: `{m}` is a cluster launch control pipeline, which cannot be grouped"
                ));
            }
            if c.cta_layout_vmnk.as_ref().is_some_and(|l| l.size() != 1) {
                out.push(format!("group `{name}`: `{m}` spans a cluster; groups take one CTA for now"));
            }
            if c.num_stages != f.num_stages {
                out.push(format!(
                    "group `{name}`: `{m}` has {} stages but `{first}` has {}; members advance together",
                    c.num_stages, f.num_stages
                ));
            }
            if c.interleave_stride != f.interleave_stride
                || c.advance_on_wait != f.advance_on_wait
                || c.advance_on_acquire != f.advance_on_acquire
            {
                out.push(format!("group `{name}`: `{m}` and `{first}` advance their stages differently"));
            }
        }
        let kinds = |side: fn(PipelineType) -> Option<AgentKind>| {
            let mut v: Vec<Option<AgentKind>> = cfgs.iter().map(|(_, c)| side(c.pipeline_type)).collect();
            v.dedup();
            v.len()
        };
        match self.mode {
            PipelineGroupMode::Fork if kinds(AgentKind::producer_of) > 1 => out.push(format!(
                "group `{name}`: a fork shares its full barriers, so every member needs the same kind of producer"
            )),
            PipelineGroupMode::Merge | PipelineGroupMode::FusedMerge if kinds(AgentKind::consumer_of) > 1 => {
                out.push(format!(
                    "group `{name}`: a merge shares its empty barriers, so every member needs the same kind of consumer"
                ))
            }
            _ => {}
        }
        out
    }

    /// The configuration derived from the members'.
    fn derive(&self) -> Option<PipelineConfig> {
        let cfgs: Vec<&PipelineConfig> = self.cfgs().map(|(_, c)| c).collect();
        let first = (*cfgs.first()?).clone();
        let bytes: u32 = cfgs.iter().map(|c| c.num_bytes).sum();
        let consumer = cfgs
            .iter()
            .find(|c| c.pipeline_type != PipelineType::TmaAsync)
            .map_or(first.consumer_group, |c| c.consumer_group);
        let mut cfg = PipelineConfig { barrier_ptr: None, ..first.clone() };
        match self.mode {
            PipelineGroupMode::Merge => {
                cfg.consumer_group = consumer;
                cfg.num_bytes = 0;
            }
            PipelineGroupMode::Fork => cfg.num_bytes = bytes,
            PipelineGroupMode::FusedMerge => {
                cfg.consumer_group = consumer;
                cfg.num_bytes = bytes;
                let sum: Option<i64> = cfgs.iter().map(|c| threads(&c.producer_group)).sum();
                if let Some(n) = sum.filter(|n| *n > 0) {
                    cfg.producer_group = CooperativeGroup::new(Agent::Thread, n as i32);
                }
            }
        }
        Some(cfg)
    }

    /// Arrival counts of the shared barriers that the members do not add
    /// up to (reported as
    /// [`VerifyIssueKind::GroupArrivalCount`](super::VerifyIssueKind)).
    pub fn arrival_count_problems(&self) -> Vec<String> {
        let name = self.memory.name();
        let Some(cfg) = self.memory.pipeline_config() else { return vec![] };
        let mut out = vec![];
        let mixed = self.mixed_tma_merge();
        match self.mode {
            PipelineGroupMode::Merge | PipelineGroupMode::FusedMerge => {
                if let Some(want) = threads(&cfg.consumer_group) {
                    for (m, c) in self.cfgs() {
                        let Some(have) = threads(&c.consumer_group) else { continue };
                        let tma_warps = mixed && c.pipeline_type == PipelineType::TmaAsync && have * 32 == want;
                        if have != want && !tma_warps {
                            out.push(format!(
                                "group `{name}`: its shared empty barrier counts {want} consumer arrivals, but the \
                                 member `{m}`'s consumer group counts {have}"
                            ));
                        }
                    }
                }
            }
            PipelineGroupMode::Fork => {
                if let Some(want) = threads(&cfg.producer_group) {
                    for (m, c) in self.cfgs() {
                        if let Some(have) = threads(&c.producer_group)
                            && have != want
                        {
                            out.push(format!(
                                "group `{name}`: its shared full barrier counts {want} producer arrivals, but the \
                                 member `{m}`'s producer group counts {have}"
                            ));
                        }
                    }
                }
            }
        }
        if self.mode == PipelineGroupMode::FusedMerge {
            let sum: Option<i64> = self.cfgs().map(|(_, c)| threads(&c.producer_group)).sum();
            if let (Some(want), Some(sum)) = (threads(&cfg.producer_group), sum)
                && want != sum
            {
                out.push(format!(
                    "group `{name}`: its fused full barrier counts {want} producer arrivals, but its members' producer \
                     groups add up to {sum}"
                ));
            }
        }
        let tma = self.cfgs().any(|(_, c)| AgentKind::producer_of(c.pipeline_type) == Some(AgentKind::Tma));
        if tma && self.mode != PipelineGroupMode::Merge {
            let sum: u32 = self.cfgs().map(|(_, c)| c.num_bytes).sum();
            if cfg.num_bytes != sum {
                out.push(format!(
                    "group `{name}`: its shared full barrier expects {} bytes per stage, but its members load {sum}",
                    cfg.num_bytes
                ));
            }
        }
        out
    }

    pub(crate) fn derivation_problems(&self) -> &[String] {
        &self.problems
    }

    pub(crate) fn has_barrier_ptr(&self) -> bool {
        self.memory.pipeline_config().is_some_and(|c| c.barrier_ptr.is_some())
    }

    /// The pipeline the group's own operations use (in a kernel, after
    /// the TaskManager's setup).
    pub fn shared_pipeline(&self) -> Option<&TsPipeline> {
        self.shared.as_ref()
    }

    /// Build the shared barriers and every member's pipeline (in a kernel;
    /// the initialisation is published by the caller). `hint` gives a
    /// member's (or the group's) producer leader warp and whether every
    /// consumer thread releases.
    #[track_caller]
    pub(crate) fn build(&mut self, hint: &dyn Fn(&str) -> (Option<u32>, bool)) -> Vec<(String, ResourcePipeline)> {
        let name = self.memory.name().to_string();
        let cfg = self.memory.pipeline_config().cloned().expect("a group with members");
        let base: Ptr<u64> = cfg.barrier_ptr.clone().unwrap_or_else(|| {
            panic!("cute: task_scheduling: group `{name}` has no barrier storage (register it with the SmemAllocator)")
        });
        let s = cfg.num_stages;
        let n = self.members.len();
        let label = |what: &str| format!("{name}.{what}");
        let lane0 = || crate::arch::lane_idx().eq(0);
        let leader = |m: &str, c: &PipelineConfig| -> Option<Boolean> {
            hint(m).0.filter(|_| c.producer_signaling_threads.has_task_warp_leader()).map(task_warp_leader)
        };
        let release_pred = |c: &PipelineConfig, all: bool| -> Option<Boolean> {
            (c.pipeline_type == PipelineType::TmaAsync && !all).then(lane0)
        };
        let cfgs: Vec<(String, PipelineConfig)> =
            self.members.iter().map(|(m, c)| (m.clone(), c.clone().expect("checked by verify"))).collect();
        let t0 = cfgs[0].1.pipeline_type;
        let mut out = vec![];
        let shared = match self.mode {
            PipelineGroupMode::Merge | PipelineGroupMode::FusedMerge => {
                let fused = self.mode == PipelineGroupMode::FusedMerge;
                let empty_at = if fused { s } else { n * s };
                let empty = make_barriers(
                    &base.offset(empty_at as i32),
                    s,
                    empty_op(t0),
                    cfg.consumer_group,
                    0,
                    label("empty"),
                );
                let all_tma = cfgs.iter().all(|(_, c)| c.pipeline_type == PipelineType::TmaAsync);
                let group_release = (all_tma && !hint(&name).1).then(lane0);
                let fused_full = fused.then(|| {
                    make_barriers(
                        &base,
                        s,
                        full_op(t0, Some(cfgs[0].1.producer_op)),
                        cfg.producer_group,
                        0,
                        label("full"),
                    )
                });
                let mut first_full: Option<MbarrierArray> = None;
                for (i, (m, c)) in cfgs.iter().enumerate() {
                    let op = full_op(c.pipeline_type, Some(c.producer_op));
                    let full = match &fused_full {
                        Some(f) => f.recast_to_new_op_type(op),
                        None => make_barriers(
                            &base.offset((i * s) as i32),
                            s,
                            op,
                            c.producer_group,
                            c.num_bytes as i32,
                            label(&format!("full{i}")),
                        ),
                    };
                    first_full.get_or_insert_with(|| full.clone());
                    let p = TsPipeline::from_parts(
                        c.pipeline_type,
                        full,
                        empty.clone(),
                        c.num_bytes as i32,
                        leader(m, c),
                        true,
                        group_release,
                    );
                    out.push((m.clone(), ResourcePipeline::Ts(p)));
                }
                TsPipeline::from_parts(
                    t0,
                    first_full.expect("members"),
                    empty,
                    0,
                    None,
                    true,
                    if all_tma { group_release } else { None },
                )
            }
            PipelineGroupMode::Fork => {
                let c0 = &cfgs[0].1;
                let full = make_barriers(
                    &base,
                    s,
                    full_op(t0, Some(c0.producer_op)),
                    cfg.producer_group,
                    cfg.num_bytes as i32,
                    label("full"),
                );
                let mut first_empty: Option<MbarrierArray> = None;
                for (i, (m, c)) in cfgs.iter().enumerate() {
                    let empty = make_barriers(
                        &base.offset(((1 + i) * s) as i32),
                        s,
                        empty_op(c.pipeline_type),
                        c.consumer_group,
                        0,
                        label(&format!("empty{i}")),
                    );
                    first_empty.get_or_insert_with(|| empty.clone());
                    let p = TsPipeline::from_parts(
                        c.pipeline_type,
                        full.clone(),
                        empty,
                        cfg.num_bytes as i32,
                        leader(m, c),
                        i == 0,
                        release_pred(c, hint(m).1),
                    );
                    out.push((m.clone(), ResourcePipeline::Ts(p)));
                }
                TsPipeline::from_parts(t0, full, first_empty.expect("members"), cfg.num_bytes as i32, None, false, None)
            }
        };
        self.memory.create_with_pipeline(Some(ResourcePipeline::Ts(shared.clone())));
        self.shared = Some(shared);
        out
    }
}

impl Resource for PipelineGroup {
    fn memory(&self) -> &MemoryResource {
        &self.memory
    }

    fn memory_mut(&mut self) -> &mut MemoryResource {
        &mut self.memory
    }

    fn as_group(&self) -> Option<&PipelineGroup> {
        Some(self)
    }

    fn as_group_mut(&mut self) -> Option<&mut PipelineGroup> {
        Some(self)
    }

    fn is_barrier(&self) -> bool {
        true
    }

    #[track_caller]
    fn create_with(&mut self, _defer_sync: bool) {
        panic!("cute: task_scheduling: group `{}` is created by the TaskManager with its members", self.memory.name());
    }
}

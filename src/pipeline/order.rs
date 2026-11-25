// SPDX-License-Identifier: MIT OR Apache-2.0

//! Groups of threads that take turns ([`PipelineOrder`]).

use crate::control::Carry;
use crate::ir::ValueId;
use crate::tensor::Ptr;
use crate::value::{Int32, IntoValue};

use super::mbarrier::portable_waits;
use super::{
    Agent, CooperativeGroup, MbarrierArray, MbarrierArrayOptions, PipelineOp, PipelineState, agent_sync,
    alloc_reserved_mbarrier,
};
use crate::arch::mbarrier_init_fence;

/// The optional arguments of [`PipelineOrder::create`].
#[derive(Clone, Debug, Default)]
pub struct PipelineOrderOptions {
    /// Shared memory for the `depth * length` mbarriers. `None` allocates
    /// them with [`alloc_reserved_mbarrier`]`(depth * length, 2)`, which,
    /// as CuTe DSL's default multiplier does, reserves twice as many as the
    /// pipeline uses.
    pub barrier_storage: Option<Ptr<u64>>,
    /// Leave out the initialisation sync, for a kernel that publishes
    /// several pipelines' barriers at once
    /// ([`pipeline_init_arrive`](super::pipeline_init_arrive) and
    /// [`pipeline_init_wait`](super::pipeline_init_wait)).
    pub defer_sync: bool,
    /// A name for profiling, kept on the barrier array.
    pub name: String,
}

/// `length` groups of threads that take turns, in order, `depth` turns
/// ahead at most (CuTe DSL's `cutlass.pipeline.PipelineOrder`), as the
/// ping-pong math warpgroups of a persistent GEMM or attention kernel
/// alternate.
///
/// There are `depth * length` mbarriers, each counting the threads of
/// `producer_group` (one group's threads). Each group keeps a
/// [`PipelineState`] of `depth` stages, which starts at phase 1 for group 0
/// and at phase 0 for the others: group 0 goes first. At stage `i`, group
/// `g` [`wait`](Self::wait)s on mbarrier `i * length + g`, takes its turn,
/// and [`arrive`](Self::arrive)s on mbarrier `i * length + (g + 1) %
/// length`, handing the turn to the next group, then advances. Every
/// thread of the group waits and arrives.
///
/// The pipeline owns its state, so a loop must carry it ([`Carry`]: the
/// state's count, index and phase). Its waits follow
/// [`MbarrierArray::wait`]: a kernel built for a target below sm_90 spins on
/// `mbarrier.test_wait`, so it runs on SM80 to SM89.
///
/// ```
/// use cute::pipeline::{Agent, CooperativeGroup, PipelineOrder, PipelineOrderOptions};
/// use cute::prelude::*;
///
/// // Warps 0 and 1 take turns appending their id to a log, 3 rounds each.
/// let kernel = Kernel::builder("turns").block_dim([64, 1, 1]).target(Arch::Sm89).build(|| {
///     let log = param_tensor::<i32>("log", Layout::col_major(6));
///     let group = CooperativeGroup::new(Agent::Warp, 1);
///     let warp = make_warp_uniform(warp_idx());
///     let order = PipelineOrder::create(2, 2, warp, group, PipelineOrderOptions::default());
///     for_range_carry(0, 3, 1, order, |round, mut order| {
///         order.wait();
///         if_(lane_idx().eq(0), || log.store(round * 2 + warp, warp));
///         order.arrive();
///         order
///     });
/// });
/// let mut args = vec![KernelArg::buffer(&[-1i32; 6])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [64, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].to_vec::<i32>(), [0, 1, 0, 1, 0, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct PipelineOrder {
    sync_object_full: MbarrierArray,
    depth: usize,
    length: usize,
    group_id: Int32,
    state: PipelineState,
}

impl PipelineOrder {
    /// The turns of `length` groups, `depth` deep, as seen by group
    /// `group_id` (a constant or a runtime [`Int32`], such as a warpgroup
    /// index), each group being `producer_group`'s threads (CuTe DSL's
    /// `PipelineOrder.create`). The barriers are initialised from warp 0
    /// and, unless `defer_sync`, published to the block (every thread must
    /// reach it).
    ///
    /// Panics for a `depth` or `length` of 0.
    #[track_caller]
    pub fn create(
        depth: usize,
        length: usize,
        group_id: impl IntoValue<i32>,
        producer_group: CooperativeGroup,
        options: PipelineOrderOptions,
    ) -> PipelineOrder {
        assert!(
            depth > 0 && length > 0,
            "cute: PipelineOrder::create: depth and length must be at least 1, not {depth} and {length}"
        );
        let _at = crate::builder::LocationScope::new();
        let group_id = group_id.into_value();
        let num_stages = depth * length;
        crate::profiling::register_barrier(&options.name, num_stages, "PipelineOrder");
        let storage = options.barrier_storage.clone().unwrap_or_else(|| alloc_reserved_mbarrier(num_stages, 2));
        let sync_object_full = MbarrierArray::with_options(
            &storage,
            num_stages,
            (PipelineOp::AsyncThread, producer_group),
            0,
            MbarrierArrayOptions { name: options.name.clone(), ..Default::default() },
        );
        if !options.defer_sync {
            if !portable_waits() {
                mbarrier_init_fence();
            }
            agent_sync(Agent::ThreadBlock, false);
        }
        // Group 0 starts at phase 1, so its first `depth` waits pass at once.
        let phase = match group_id.as_constant() {
            Some(g) => Int32::from(i32::from(g == 0)),
            None => group_id.eq(0).to::<i32>(),
        };
        let state = PipelineState::new(depth, Int32::from(0), Int32::from(0), phase);
        PipelineOrder { sync_object_full, depth, length, group_id, state }
    }

    /// The barriers, `depth * length` of them.
    pub fn sync_object_full(&self) -> &MbarrierArray {
        &self.sync_object_full
    }

    /// The turns a group may be ahead of the next, the stages of its state.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The number of groups.
    pub fn length(&self) -> usize {
        self.length
    }

    /// This group's index.
    pub fn group_id(&self) -> Int32 {
        self.group_id
    }

    /// This group's position in its turns.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    fn length_i32(&self) -> i32 {
        i32::try_from(self.length).expect("cute: PipelineOrder's length fits in i32")
    }

    /// The barrier of group `group_id` at the stage of `state` (this
    /// pipeline's own state for `None`): `state.index() * length +
    /// group_id` (CuTe DSL's `get_barrier_for_current_stage_idx`).
    #[track_caller]
    pub fn get_barrier_for_current_stage_idx(
        &self,
        group_id: impl IntoValue<i32>,
        state: Option<&PipelineState>,
    ) -> Int32 {
        let state = state.unwrap_or(&self.state);
        state.index() * self.length_i32() + group_id.into_value()
    }

    /// Wait for this group's turn at its own state (CuTe DSL's `wait()`).
    #[track_caller]
    pub fn wait(&self) {
        self.wait_with(&self.state);
    }

    /// Wait for this group's turn at `state` (CuTe DSL's `wait(state)`).
    #[track_caller]
    pub fn wait_with(&self, state: &PipelineState) {
        let _at = crate::builder::LocationScope::new();
        let idx = self.get_barrier_for_current_stage_idx(self.group_id, Some(state));
        self.sync_object_full.wait(idx, state.phase());
    }

    /// End this group's turn: arrive on the next group's barrier at this
    /// stage, then advance the pipeline's own state (CuTe DSL's
    /// `arrive()`).
    #[track_caller]
    pub fn arrive(&mut self) {
        let state = self.state;
        self.state = self.arrive_with(state);
    }

    /// End this group's turn at `state`: arrive on the next group's barrier
    /// at its stage, and return `state` advanced (CuTe DSL's
    /// `arrive(state)`, which returns the state it was given, advanced).
    #[track_caller]
    pub fn arrive_with(&self, mut state: PipelineState) -> PipelineState {
        let _at = crate::builder::LocationScope::new();
        let next = (self.group_id + 1) % self.length_i32();
        let idx = self.get_barrier_for_current_stage_idx(next, Some(&state));
        self.sync_object_full.arrive(idx, None, None);
        state.advance();
        state
    }
}

impl Carry for PipelineOrder {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.state.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        PipelineOrder { state: self.state.rebuild(values), ..self.clone() }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        if (self.depth, self.length) != (other.depth, other.length) {
            return Some(format!(
                "a PipelineOrder of depth {} and length {} became one of depth {} and length {}",
                self.depth, self.length, other.depth, other.length
            ));
        }
        self.state.structure_mismatch(&other.state)
    }
}

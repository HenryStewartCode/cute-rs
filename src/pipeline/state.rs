//! Positions in a pipeline's circular buffer: [`PipelineState`] (one phase
//! for the whole ring), [`PerStagePipelineState`] (a phase and count per
//! stage) and the [`StageState`] interface the pipelines read them through.

use crate::control::Carry;
use crate::ir::ValueId;
use crate::layout::Layout;
use crate::tensor::{Tensor, make_rmem_tensor};
use crate::value::{Int32, select};

use super::BarrierParam;

/// What a pipeline reads from a position in its ring: the stage to use and
/// the phase (0 or 1) to wait for there, the parity that
/// [`MbarrierArray::wait`](super::MbarrierArray::wait) takes. The pipelines
/// accept any state that has both, so [`PipelineState`] and
/// [`PerStagePipelineState`] work with them alike.
pub trait StageState {
    /// The stage, from 0 to the number of stages less one.
    fn index(&self) -> Int32;
    /// The phase to wait for at that stage, 0 or 1.
    fn phase(&self) -> Int32;
}

/// Which side of a pipeline a [`PipelineState`] tracks (CuTe DSL's
/// `cutlass.pipeline.PipelineUserType`); [`make_pipeline_state`] starts the
/// phase from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineUserType {
    /// Fills stages: starts at phase 1, the ring being empty.
    Producer,
    /// Drains stages: starts at phase 0.
    Consumer,
    /// Both sides at once. It starts at phase 1, like a producer: CuTe DSL
    /// tests for it in the producer's branch first, so its consumer branch
    /// never sees it, and cute-rs keeps that.
    ProducerConsumer,
}

/// A position in a pipeline's ring of `stages` stages (CuTe DSL's
/// `cutlass.pipeline.PipelineState`): the stage `index`, the `phase` bit
/// that flips each time the index wraps, and the `count` of stages passed
/// so far. The three are runtime [`Int32`]s that loops carry ([`Carry`],
/// in CuTe DSL's order: count, index, phase); the number of stages is
/// static.
///
/// ```
/// use cute::pipeline::{PipelineUserType, make_pipeline_state};
/// use cute::prelude::*;
///
/// let kernel = Kernel::builder("ring").block_dim([32, 1, 1]).build(|| {
///     let out = param_tensor::<i32>("out", Layout::col_major(3));
///     let state = make_pipeline_state(PipelineUserType::Consumer, 4);
///     let state = for_range_carry(0, 6, 1, state, |_, mut s| {
///         s.advance();
///         s
///     });
///     if_(thread_idx().x.eq(0), || {
///         out.store(0, state.index());
///         out.store(1, state.phase());
///         out.store(2, state.count());
///     });
/// });
/// let mut args = vec![KernelArg::buffer(&[0i32; 3])];
/// kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
/// assert_eq!(args[0].to_vec::<i32>(), [2, 1, 6]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PipelineState {
    stages: usize,
    count: Int32,
    index: Int32,
    phase: Int32,
}

impl PipelineState {
    /// A state of `stages` stages at `index` and `phase`, `count` stages in.
    #[track_caller]
    pub fn new(
        stages: usize,
        count: impl Into<Int32>,
        index: impl Into<Int32>,
        phase: impl Into<Int32>,
    ) -> PipelineState {
        assert!(stages > 0, "cute: a PipelineState needs at least one stage");
        PipelineState { stages, count: count.into(), index: index.into(), phase: phase.into() }
    }

    /// A copy (CuTe DSL's `clone`; the state is `Copy`).
    pub fn clone_state(&self) -> PipelineState {
        *self
    }

    /// The current stage.
    pub fn index(&self) -> Int32 {
        self.index
    }

    /// The stages passed so far.
    pub fn count(&self) -> Int32 {
        self.count
    }

    /// The phase, 0 or 1.
    pub fn phase(&self) -> Int32 {
        self.phase
    }

    /// The number of stages in the ring.
    pub fn stages(&self) -> usize {
        self.stages
    }

    fn stages_i32(&self) -> i32 {
        i32::try_from(self.stages).expect("cute: a PipelineState's stage count fits in i32")
    }

    /// Set the count to 0 (CuTe DSL's `reset_count`).
    #[track_caller]
    pub fn reset_count(&mut self) {
        self.count = Int32::from(0);
    }

    /// Move to the next stage: the index and count go up by one, and when
    /// the index reaches the number of stages it wraps to 0 and the phase
    /// flips (CuTe DSL's `advance()`). Built with selects, not a branch, so
    /// that the values can leave the call (as CuTe DSL's `if_generate`
    /// yields them).
    #[track_caller]
    pub fn advance(&mut self) {
        let index = self.index + 1;
        self.count = self.count + 1;
        let wrap = index.eq(self.stages_i32());
        self.index = select(wrap, 0, index);
        self.phase = select(wrap, self.phase ^ 1, self.phase);
    }

    /// Move `n` stages on (CuTe DSL's `advance(count)`): a static `n` of at
    /// least 1, or an [`Int32`] computed at run time. A static 1 is
    /// [`advance`](Self::advance); otherwise the index and count go up by
    /// `n`, the phase flips once for each wrap (`(index / stages) % 2`) and
    /// the index is taken modulo the number of stages. Panics for a static
    /// `n` below 1.
    #[track_caller]
    pub fn advance_by(&mut self, n: impl Into<BarrierParam>) {
        let n = n.into();
        if let BarrierParam::Static(k) = n {
            assert!(k >= 1, "cute: PipelineState::advance_by: count must be >= 1, not {k}");
            if k == 1 {
                self.advance();
                return;
            }
        }
        let n = n.value();
        let stages = self.stages_i32();
        self.count = self.count + n;
        let index = self.index + n;
        let flips = index / stages;
        self.index = index % stages;
        self.phase = self.phase ^ (flips % 2);
    }

    /// Move back one stage (CuTe DSL's `reverse`): the index and count go
    /// down by one, and an index of -1 wraps to the last stage with the
    /// phase flipped.
    #[track_caller]
    pub fn reverse(&mut self) {
        let index = self.index - 1;
        self.count = self.count - 1;
        let wrap = index.eq(-1);
        self.index = select(wrap, self.stages_i32() - 1, index);
        self.phase = select(wrap, self.phase ^ 1, self.phase);
    }
}

impl StageState for PipelineState {
    fn index(&self) -> Int32 {
        self.index
    }
    fn phase(&self) -> Int32 {
        self.phase
    }
}

impl Carry for PipelineState {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.count.flatten(out);
        self.index.flatten(out);
        self.phase.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        PipelineState {
            stages: self.stages,
            count: self.count.rebuild(values),
            index: self.index.rebuild(values),
            phase: self.phase.rebuild(values),
        }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.stages != other.stages)
            .then(|| format!("a PipelineState of {} stages became one of {}", self.stages, other.stages))
    }
}

/// A fresh [`PipelineState`] of `stages` stages for a producer or consumer
/// (CuTe DSL's `make_pipeline_state`): index and count 0, phase 1 for a
/// producer (the ring starts empty, so its first wait for an empty stage
/// passes at once) and 0 for a consumer. A
/// [`ProducerConsumer`](PipelineUserType::ProducerConsumer) starts like a
/// producer.
#[track_caller]
pub fn make_pipeline_state(kind: PipelineUserType, stages: usize) -> PipelineState {
    let phase = match kind {
        PipelineUserType::Producer | PipelineUserType::ProducerConsumer => 1,
        PipelineUserType::Consumer => 0,
    };
    PipelineState::new(stages, Int32::from(0), Int32::from(0), Int32::from(phase))
}

/// A pipeline position whose stages each keep their own phase and count
/// (CuTe DSL's `cutlass.pipeline.PerStagePipelineState`), for stages that
/// complete out of order, such as a TMA load that must be issued again. The
/// phases and counts live in two per-thread arrays
/// ([`make_rmem_tensor`]), both 0 at first; only the stage index is carried
/// through loops ([`Carry`]), and every copy of the state shares the
/// arrays.
///
/// [`index`](Self::index), [`phase`](Self::phase) and
/// [`count`](Self::count) read the current stage; [`next`](Self::next)
/// moves to the next stage without changing either array;
/// [`mark_valid`](Self::mark_valid) and
/// [`mark_invalid`](Self::mark_invalid) record the outcome of the current
/// stage.
#[derive(Clone, Debug)]
pub struct PerStagePipelineState {
    num_stages: usize,
    index: Int32,
    phases: Tensor<i32>,
    counts: Tensor<i32>,
}

impl PerStagePipelineState {
    /// A state of `num_stages` stages at stage 0, every phase and count 0.
    #[track_caller]
    pub fn new(num_stages: usize) -> PerStagePipelineState {
        assert!(num_stages > 0, "cute: a PerStagePipelineState needs at least one stage");
        let phases = make_rmem_tensor::<i32>(Layout::col_major(num_stages));
        let counts = make_rmem_tensor::<i32>(Layout::col_major(num_stages));
        phases.fill(0);
        counts.fill(0);
        PerStagePipelineState { num_stages, index: Int32::from(0), phases, counts }
    }

    /// The number of stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// The current stage.
    pub fn index(&self) -> Int32 {
        self.index
    }

    /// The current stage's phase.
    #[track_caller]
    pub fn phase(&self) -> Int32 {
        self.phases.load(self.index)
    }

    /// The rounds the current stage has completed.
    #[track_caller]
    pub fn count(&self) -> Int32 {
        self.counts.load(self.index)
    }

    /// Move to the next stage, wrapping to 0 after the last; the phases and
    /// counts stay as they are.
    #[track_caller]
    pub fn next(&mut self) {
        let index = self.index + 1;
        let n = i32::try_from(self.num_stages).expect("cute: the stage count fits in i32");
        self.index = select(index.eq(n), 0, index);
    }

    /// The current stage's data arrived and is good: flip its phase and
    /// count the round.
    #[track_caller]
    pub fn mark_valid(&self) {
        self.phases.store(self.index, self.phases.load(self.index) ^ 1);
        self.counts.store(self.index, self.counts.load(self.index) + 1);
    }

    /// The current stage's barrier completed but its data must be loaded
    /// again: flip its phase without counting the round.
    #[track_caller]
    pub fn mark_invalid(&self) {
        self.phases.store(self.index, self.phases.load(self.index) ^ 1);
    }

    /// Set every stage's count to 0, for a new work tile.
    #[track_caller]
    pub fn reset_count(&self) {
        for i in 0..self.num_stages {
            self.counts.store(i, 0);
        }
    }
}

impl StageState for PerStagePipelineState {
    fn index(&self) -> Int32 {
        self.index
    }
    #[track_caller]
    fn phase(&self) -> Int32 {
        PerStagePipelineState::phase(self)
    }
}

impl Carry for PerStagePipelineState {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.index.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        PerStagePipelineState { index: self.index.rebuild(values), ..self.clone() }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.num_stages != other.num_stages).then(|| {
            format!("a PerStagePipelineState of {} stages became one of {}", self.num_stages, other.num_stages)
        })
    }
}

/// The k tile a stage holds (CuTe DSL's `compute_k_tile`): with k tile 0
/// in stage `stage_offset`, stage `stage_idx` after `stage_count` rounds
/// holds k tile `(stage_idx - stage_offset) mod num_stages + stage_count *
/// num_stages`. The difference is wrapped once, without a remainder, so
/// `stage_idx` and `stage_offset` must both be stages (0 to `num_stages -
/// 1`).
#[track_caller]
pub fn compute_k_tile(stage_idx: Int32, stage_count: Int32, stage_offset: Int32, num_stages: usize) -> Int32 {
    let n = i32::try_from(num_stages).expect("cute: the stage count fits in i32");
    let x = stage_idx - stage_offset + n;
    let pos = select(x.lt(n), x, x - n);
    pos + stage_count * n
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! cute_ext's TMEM accumulator buffering (CuTe DSL's
//! `cutlass.cute.experimental.accumulator_buffering`): how many accumulator
//! views a warp-specialised GEMM keeps in tensor memory, where they and the
//! kernel's other TMEM allocations go, and how the MMA warp and the epilogue
//! pick views so that the choice cannot drift from the pipeline.
//!
//! # Plans
//!
//! [`plan_accumulator_buffering`] takes one accumulator view's layout (its
//! columns: [`compute_tmem_cols_from_layout`]) and the TMEM requests packed
//! after it, and picks, in this order:
//!
//! 1. **Disjoint**, `d` views side by side, for `d` from
//!    `min(desired_disjoint_depth, max_auto_disjoint_depth)` down to 2: the
//!    first whose packed footprint fits the capacity. The pipeline has `d`
//!    stages and view `i` is stage `i`.
//! 2. **Overlapping** (only with `allow_overlap`, and a wanted depth of at
//!    least 2): two views of `C` columns sharing `O` (the *overlap*), so
//!    view 1 starts `C - O` columns after view 0 and the pair spans `2C - O`.
//!    The smallest `O` from 1 whose stride `C - O` is a multiple of
//!    `overlap_alignment_columns`, whose footprint fits, and whose release
//!    point falls inside the drain (the policy
//!    [`MinimumLegalFit`](AccumulatorOverlapPolicy::MinimumLegalFit)). The
//!    pipeline has one stage: the views alternate by round.
//! 3. **One view** (disjoint of depth 1), if that fits; otherwise the plan
//!    is refused.
//!
//! The footprint packs the accumulator at column 0 (aligned to
//! `accumulator_alignment_bytes / 4` columns), then each request in order,
//! each at the next multiple of its alignment.
//!
//! # The drain
//!
//! The epilogue reads a view in subtiles of `S` columns (from
//! `epilogue_tile`, the tile its `flat_divide` uses, or given directly). It
//! may release the view after the subtile of ordinal
//! `release_after_t2r_ordinal`: the last subtile (`ceil(C / S) - 1`) for a
//! disjoint plan, and `ceil(O / S) - 1` for an overlapping one, whose views
//! are drained starting from the shared columns: view 0 in reverse (its
//! last `O` columns are view 1's first), view 1 forwards. The early release
//! lets the MMA warp start the next round's view while the epilogue still
//! drains the unshared rest. [`AccumulatorDrain`] carries the pipeline and
//! the view selector through that drain.
//!
//! # Phases
//!
//! The views alternate by the round of the one-stage ring. CuTe DSL
//! derives the producer's selector from its phase flipped (its producer
//! phase starts at 1); cute-rs's producer and consumer phases agree (both
//! are the round's parity: see [`pipeline_ops`](super::pipeline_ops)), so
//! both sides select by the phase itself.

use std::fmt;

use super::pipeline::AsGenericPipeline;
use crate::control::Carry;
use crate::ir::{DType, ValueId};
use crate::layout::{IndexExpr, Layout, SliceCoord, Tiler, Tree};
use crate::memory::compute_tmem_cols_from_layout;
use crate::tensor::Tensor;
use crate::value::{Int32, Numeric, select};

/// How a plan's views relate (CuTe DSL's `AccumulatorBufferingKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccumulatorBufferingKind {
    /// Separate columns per view.
    Disjoint,
    /// Two views sharing columns.
    Overlapping,
}

/// How an overlap width is chosen (CuTe DSL's `AccumulatorOverlapPolicy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AccumulatorOverlapPolicy {
    /// The smallest overlap that fits (see the module docs).
    #[default]
    MinimumLegalFit,
}

/// A TMEM allocation the kernel makes after the accumulator, in this order
/// (CuTe DSL's `TmemAllocationRequest`): its columns and its alignment in
/// columns (32-bit TMEM columns, not bytes).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TmemAllocationRequest {
    pub name: String,
    pub columns: usize,
    pub alignment_columns: usize,
}

impl TmemAllocationRequest {
    /// A request. Panics for an empty name or an alignment of 0.
    #[track_caller]
    pub fn new(name: &str, columns: usize, alignment_columns: usize) -> TmemAllocationRequest {
        assert!(!name.is_empty(), "cute: experimental::TmemAllocationRequest: the request needs a name");
        assert!(
            alignment_columns > 0,
            "cute: experimental::TmemAllocationRequest: `{name}` has an alignment of 0 columns; it must be positive"
        );
        TmemAllocationRequest { name: name.to_string(), columns, alignment_columns }
    }

    /// The request for a tensor of `layout` and `dtype` allocated with
    /// `alignment_bytes` (CuTe DSL's `from_layout`): its columns
    /// ([`compute_tmem_cols_from_layout`]) and `alignment_bytes / 4`.
    /// Panics unless `alignment_bytes` is a positive multiple of 4 (a TMEM
    /// column's bytes).
    #[track_caller]
    pub fn from_layout(name: &str, layout: &Layout, dtype: DType, alignment_bytes: usize) -> TmemAllocationRequest {
        assert!(
            alignment_bytes > 0 && alignment_bytes.is_multiple_of(4),
            "cute: experimental::TmemAllocationRequest::from_layout: `{name}`'s alignment of {alignment_bytes} bytes \
             is not a positive multiple of a TMEM column's 4 bytes"
        );
        TmemAllocationRequest::new(name, compute_tmem_cols_from_layout(layout, dtype), alignment_bytes / 4)
    }
}

/// The options of [`plan_accumulator_buffering`] (CuTe DSL's keyword
/// arguments, with their defaults). Give exactly one of `epilogue_tile`
/// and `epilogue_subtile_columns`.
#[derive(Clone, Debug)]
pub struct AccumulatorBufferingOptions {
    /// The epilogue's tile in accumulator elements, `(M, N)` at least: the
    /// one its `flat_divide` uses.
    pub epilogue_tile: Option<Tiler>,
    /// The columns one epilogue subtile drains, given directly.
    pub epilogue_subtile_columns: Option<usize>,
    /// The kernel's other TMEM allocations, in allocation order.
    pub auxiliary_tmem_allocations: Vec<TmemAllocationRequest>,
    /// Default 512.
    pub tmem_capacity_columns: usize,
    /// Default 2.
    pub desired_disjoint_depth: usize,
    /// Default 2.
    pub max_auto_disjoint_depth: usize,
    /// Default false.
    pub allow_overlap: bool,
    /// Default 1.
    pub overlap_alignment_columns: usize,
    /// Default 16.
    pub accumulator_alignment_bytes: usize,
    pub overlap_policy: AccumulatorOverlapPolicy,
}

impl Default for AccumulatorBufferingOptions {
    fn default() -> Self {
        AccumulatorBufferingOptions {
            epilogue_tile: None,
            epilogue_subtile_columns: None,
            auxiliary_tmem_allocations: Vec::new(),
            tmem_capacity_columns: 512,
            desired_disjoint_depth: 2,
            max_auto_disjoint_depth: 2,
            allow_overlap: false,
            overlap_alignment_columns: 1,
            accumulator_alignment_bytes: 16,
            overlap_policy: AccumulatorOverlapPolicy::MinimumLegalFit,
        }
    }
}

/// The integer part of a plan (CuTe DSL's plan without its layout), as
/// [`plan_accumulator_buffering_columns`] picks it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AccumulatorPlanSelection {
    pub kind: AccumulatorBufferingKind,
    /// Views in storage.
    pub storage_depth: usize,
    /// Pipeline stages.
    pub synchronization_depth: usize,
    /// Columns of one view.
    pub one_stage_columns: usize,
    /// Columns from one view to the next.
    pub stage_stride_columns: usize,
    /// Columns all views span.
    pub physical_columns: usize,
    /// Columns two views share.
    pub overlap_columns: usize,
    /// The epilogue subtile after whose TMEM load the view is released.
    pub release_after_t2r_ordinal: usize,
    /// The start column of the accumulator, then of each request.
    pub allocation_offsets: Vec<usize>,
    /// The end of the last allocation.
    pub packed_columns: usize,
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// The packed end and the start of each allocation: the accumulator's
/// columns at 0 (aligned), then the requests.
fn pack(
    accumulator_columns: usize,
    accumulator_alignment: usize,
    aux: &[TmemAllocationRequest],
) -> (usize, Vec<usize>) {
    let mut offset = align_up(0, accumulator_alignment);
    let mut offsets = vec![offset];
    offset += accumulator_columns;
    for r in aux {
        offset = align_up(offset, r.alignment_columns);
        offsets.push(offset);
        offset += r.columns;
    }
    (offset, offsets)
}

/// The integer planner (CuTe DSL's pure planner, used by
/// [`plan_accumulator_buffering`] and testable on its own): see the
/// module docs. Panics for a zero count or alignment and when not even one
/// view fits.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn plan_accumulator_buffering_columns(
    one_stage_columns: usize,
    epilogue_subtile_columns: usize,
    auxiliary_tmem_allocations: &[TmemAllocationRequest],
    tmem_capacity_columns: usize,
    desired_disjoint_depth: usize,
    max_auto_disjoint_depth: usize,
    allow_overlap: bool,
    overlap_alignment_columns: usize,
    accumulator_alignment_columns: usize,
    overlap_policy: AccumulatorOverlapPolicy,
) -> AccumulatorPlanSelection {
    const WHAT: &str = "cute: experimental::plan_accumulator_buffering";
    for (v, name) in [
        (one_stage_columns, "one_stage_columns"),
        (epilogue_subtile_columns, "epilogue_subtile_columns"),
        (tmem_capacity_columns, "tmem_capacity_columns"),
        (desired_disjoint_depth, "desired_disjoint_depth"),
        (max_auto_disjoint_depth, "max_auto_disjoint_depth"),
        (overlap_alignment_columns, "overlap_alignment_columns"),
        (accumulator_alignment_columns, "accumulator_alignment_columns"),
    ] {
        assert!(v > 0, "{WHAT}: {name} is 0; it must be positive");
    }
    let AccumulatorOverlapPolicy::MinimumLegalFit = overlap_policy;
    let subtiles = |cols: usize| cols.div_ceil(epilogue_subtile_columns);
    let aux = auxiliary_tmem_allocations;
    let disjoint = |depth: usize, packed: usize, offsets: Vec<usize>| AccumulatorPlanSelection {
        kind: AccumulatorBufferingKind::Disjoint,
        storage_depth: depth,
        synchronization_depth: depth,
        one_stage_columns,
        stage_stride_columns: one_stage_columns,
        physical_columns: depth * one_stage_columns,
        overlap_columns: 0,
        release_after_t2r_ordinal: subtiles(one_stage_columns) - 1,
        allocation_offsets: offsets,
        packed_columns: packed,
    };

    let max_depth = desired_disjoint_depth.min(max_auto_disjoint_depth);
    for depth in (2..=max_depth).rev() {
        let (packed, offsets) = pack(depth * one_stage_columns, accumulator_alignment_columns, aux);
        if packed <= tmem_capacity_columns {
            return disjoint(depth, packed, offsets);
        }
    }

    if allow_overlap && max_depth >= 2 {
        for overlap in 1..one_stage_columns {
            let stride = one_stage_columns - overlap;
            if !stride.is_multiple_of(overlap_alignment_columns) {
                continue;
            }
            let physical = 2 * one_stage_columns - overlap;
            let (packed, offsets) = pack(physical, accumulator_alignment_columns, aux);
            if packed > tmem_capacity_columns {
                continue;
            }
            let release = subtiles(overlap) - 1;
            if release >= subtiles(one_stage_columns) {
                continue;
            }
            return AccumulatorPlanSelection {
                kind: AccumulatorBufferingKind::Overlapping,
                storage_depth: 2,
                synchronization_depth: 1,
                one_stage_columns,
                stage_stride_columns: stride,
                physical_columns: physical,
                overlap_columns: overlap,
                release_after_t2r_ordinal: release,
                allocation_offsets: offsets,
                packed_columns: packed,
            };
        }
    }

    let (packed, offsets) = pack(one_stage_columns, accumulator_alignment_columns, aux);
    if packed <= tmem_capacity_columns {
        return disjoint(1, packed, offsets);
    }
    let requests: Vec<String> =
        aux.iter().map(|r| format!("{} ({} columns, aligned to {})", r.name, r.columns, r.alignment_columns)).collect();
    panic!(
        "{WHAT}: one accumulator view of {one_stage_columns} columns and the other allocations [{}] need {packed} \
         TMEM columns once aligned, more than the capacity of {tmem_capacity_columns}",
        requests.join(", ")
    )
}

/// The columns one epilogue subtile covers: the view's `(M, N)` mode
/// (the other modes at 0), divided by `tile`, one tile of it.
#[track_caller]
fn epilogue_subtile_columns(one_stage_layout: &Layout, dtype: DType, tile: &Tiler) -> usize {
    const WHAT: &str = "cute: experimental::plan_accumulator_buffering";
    assert!(
        matches!(tile, Tree::Branch(m) if m.len() >= 2),
        "{WHAT}: the epilogue tile {tile:?} is not (M, N) or longer"
    );
    let mode0 = one_stage_layout.mode(0);
    assert!(
        one_stage_layout.rank() >= 1 && mode0.rank() == 2,
        "{WHAT}: the accumulator layout {one_stage_layout} does not start with an (M, N) mode"
    );
    let mut coord: Vec<SliceCoord> = vec![Tree::Branch(vec![Tree::Leaf(None), Tree::Leaf(None)])];
    coord.extend((1..one_stage_layout.rank()).map(|_| Tree::Leaf(Some(IndexExpr::Static(0)))));
    let (stage, _) = one_stage_layout.slice(&Tree::Branch(coord));
    let divided = stage.flat_divide(tile);
    let mut coord: Vec<SliceCoord> = vec![Tree::Leaf(None), Tree::Leaf(None)];
    coord.extend((2..divided.rank()).map(|_| Tree::Leaf(Some(IndexExpr::Static(0)))));
    let (sub, _) = divided.slice(&Tree::Branch(coord));
    let cols = compute_tmem_cols_from_layout(&sub, dtype);
    assert!(cols > 0, "{WHAT}: an epilogue subtile of {tile:?} spans no TMEM columns");
    cols
}

/// A checked plan with its storage layout (CuTe DSL's
/// `AccumulatorBufferingPlan`): the selection's numbers, which satisfy the
/// invariants of its kind (checked when it is made), and the storage
/// layout: the one-view layout with a last mode of `storage_depth` views,
/// `stage_stride_columns` columns apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccumulatorBufferingPlan {
    selection: AccumulatorPlanSelection,
    storage_layout: Layout,
}

impl AccumulatorBufferingPlan {
    /// A plan from a selection and its storage layout. Panics unless the
    /// selection is consistent: positive view columns; disjoint: positive
    /// depth, equal storage and synchronisation depths, no overlap, a stride
    /// of one view and a footprint of `depth` views; overlapping: storage
    /// depth 2, synchronisation depth 1, an overlap strictly between 0 and
    /// one view, a stride of one view minus the overlap and a footprint of
    /// two views minus it.
    #[track_caller]
    pub fn new(selection: AccumulatorPlanSelection, storage_layout: Layout) -> AccumulatorBufferingPlan {
        const WHAT: &str = "cute: experimental::AccumulatorBufferingPlan";
        let s = &selection;
        assert!(s.one_stage_columns > 0, "{WHAT}: a view must span at least one TMEM column");
        match s.kind {
            AccumulatorBufferingKind::Disjoint => {
                assert!(s.storage_depth > 0, "{WHAT}: a disjoint plan needs at least one view");
                assert!(
                    s.synchronization_depth == s.storage_depth,
                    "{WHAT}: a disjoint plan's pipeline has one stage per view ({} stages for {} views)",
                    s.synchronization_depth,
                    s.storage_depth
                );
                assert!(
                    s.overlap_columns == 0,
                    "{WHAT}: disjoint views share no columns ({} given)",
                    s.overlap_columns
                );
                assert!(
                    s.stage_stride_columns == s.one_stage_columns,
                    "{WHAT}: disjoint views are one view ({} columns) apart, not {}",
                    s.one_stage_columns,
                    s.stage_stride_columns
                );
                assert!(
                    s.physical_columns == s.storage_depth * s.one_stage_columns,
                    "{WHAT}: {} disjoint views of {} columns span {}, not {}",
                    s.storage_depth,
                    s.one_stage_columns,
                    s.storage_depth * s.one_stage_columns,
                    s.physical_columns
                );
            }
            AccumulatorBufferingKind::Overlapping => {
                assert!(
                    s.storage_depth == 2 && s.synchronization_depth == 1,
                    "{WHAT}: an overlapping plan has two views on a one-stage pipeline ({} views, {} stages given)",
                    s.storage_depth,
                    s.synchronization_depth
                );
                assert!(
                    0 < s.overlap_columns && s.overlap_columns < s.one_stage_columns,
                    "{WHAT}: the overlap ({}) must be more than 0 and less than a view's {} columns",
                    s.overlap_columns,
                    s.one_stage_columns
                );
                assert!(
                    s.stage_stride_columns == s.one_stage_columns - s.overlap_columns,
                    "{WHAT}: overlapping views are a view minus the overlap ({}) apart, not {}",
                    s.one_stage_columns - s.overlap_columns,
                    s.stage_stride_columns
                );
                assert!(
                    s.physical_columns == 2 * s.one_stage_columns - s.overlap_columns,
                    "{WHAT}: two views overlapping by {} span {}, not {}",
                    s.overlap_columns,
                    2 * s.one_stage_columns - s.overlap_columns,
                    s.physical_columns
                );
            }
        }
        AccumulatorBufferingPlan { selection, storage_layout }
    }

    /// The numbers.
    pub fn selection(&self) -> &AccumulatorPlanSelection {
        &self.selection
    }
    pub fn kind(&self) -> AccumulatorBufferingKind {
        self.selection.kind
    }
    pub fn storage_depth(&self) -> usize {
        self.selection.storage_depth
    }
    pub fn synchronization_depth(&self) -> usize {
        self.selection.synchronization_depth
    }
    pub fn one_stage_columns(&self) -> usize {
        self.selection.one_stage_columns
    }
    pub fn stage_stride_columns(&self) -> usize {
        self.selection.stage_stride_columns
    }
    pub fn physical_columns(&self) -> usize {
        self.selection.physical_columns
    }
    pub fn overlap_columns(&self) -> usize {
        self.selection.overlap_columns
    }
    pub fn release_after_t2r_ordinal(&self) -> usize {
        self.selection.release_after_t2r_ordinal
    }
    pub fn allocation_offsets(&self) -> &[usize] {
        &self.selection.allocation_offsets
    }
    pub fn packed_columns(&self) -> usize {
        self.selection.packed_columns
    }
    /// The views' layout (`(..view modes.., storage_depth)`).
    pub fn storage_layout(&self) -> &Layout {
        &self.storage_layout
    }

    /// Whether the views overlap (CuTe DSL's `uses_overlap`).
    pub fn uses_overlap(&self) -> bool {
        self.selection.kind == AccumulatorBufferingKind::Overlapping
    }

    #[track_caller]
    fn check<P: AsGenericPipeline, T: Numeric>(&self, pipeline: &P, buffer: &Tensor<T>, what: &str) {
        let stages = pipeline.base().num_stages();
        assert!(
            stages == self.selection.synchronization_depth,
            "cute: experimental::AccumulatorBufferingPlan::{what}: the plan's pipeline has {} stages, this one {stages}",
            self.selection.synchronization_depth
        );
        let rank = buffer.rank();
        let depth = if rank == 0 { 0 } else { buffer.layout().mode(rank - 1).size() };
        assert!(
            rank >= 2 && depth == self.selection.storage_depth,
            "cute: experimental::AccumulatorBufferingPlan::{what}: the buffer {} does not end in the plan's {} views",
            buffer.layout(),
            self.selection.storage_depth
        );
    }

    /// Acquire the producer's stage and return the view it writes (CuTe
    /// DSL's `producer_acquire_and_get_view`): the buffer (laid out by
    /// [`storage_layout`](Self::storage_layout)) at the view of the
    /// stage's index (disjoint) or the round's parity (overlapping). The
    /// view is handed out only after the acquire, so the MMA cannot start
    /// on columns the epilogue still drains.
    #[track_caller]
    pub fn producer_acquire_and_get_view<P: AsGenericPipeline, T: Numeric>(
        &self,
        pipeline: &mut P,
        buffer: &Tensor<T>,
    ) -> Tensor<T> {
        self.check(pipeline, buffer, "producer_acquire_and_get_view");
        let selector = if self.uses_overlap() {
            pipeline.base_mut().producer_acquire_and_get_phase(None)
        } else {
            pipeline.base_mut().producer_acquire_and_get_stage(None).1
        };
        select_last_mode(buffer, selector)
    }

    /// Wait for the consumer's stage and return the view to drain and the
    /// [`AccumulatorDrain`] that carries a copy of the pipeline through the
    /// drain (CuTe DSL's `consumer_wait_and_get_view`). Release through
    /// the drain ([`AccumulatorDrain::after_t2r`]) and take the pipeline
    /// back with [`AccumulatorDrain::into_pipeline`].
    #[track_caller]
    pub fn consumer_wait_and_get_view<P: AsGenericPipeline, T: Numeric>(
        &self,
        pipeline: &mut P,
        buffer: &Tensor<T>,
    ) -> (Tensor<T>, AccumulatorDrain<P>) {
        self.check(pipeline, buffer, "consumer_wait_and_get_view");
        let selector = if self.uses_overlap() {
            pipeline.base_mut().consumer_wait_and_get_phase(None)
        } else {
            pipeline.base_mut().consumer_wait_and_get_stage(None).1
        };
        let view = select_last_mode(buffer, selector);
        (view, AccumulatorDrain { plan: self.clone(), pipeline: pipeline.clone(), consumer_selector: selector })
    }
}

impl fmt::Display for AccumulatorBufferingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.selection;
        write!(
            f,
            "{:?} x{} (sync {}): {} columns a view, stride {}, overlap {}, {} physical, packed {} at {:?}, release \
             after subtile {}; {}",
            s.kind,
            s.storage_depth,
            s.synchronization_depth,
            s.one_stage_columns,
            s.stage_stride_columns,
            s.overlap_columns,
            s.physical_columns,
            s.packed_columns,
            s.allocation_offsets,
            s.release_after_t2r_ordinal,
            self.storage_layout
        )
    }
}

/// `buffer` at `selector` in its last mode.
#[track_caller]
fn select_last_mode<T: Numeric>(buffer: &Tensor<T>, selector: Int32) -> Tensor<T> {
    let rank = buffer.rank();
    let mut coord: Vec<SliceCoord> = (0..rank - 1).map(|_| Tree::Leaf(None)).collect();
    coord.push(Tree::Leaf(Some(IndexExpr::from(selector))));
    buffer.slice(Tree::Branch(coord))
}

/// Plan the accumulator views for a one-view layout `one_stage_layout` of
/// `accumulator_dtype` (CuTe DSL's `plan_accumulator_buffering`; see the
/// module docs), and build the storage layout. The layout's first mode is
/// `(M, N)`, its N stride being one TMEM column (the views' stride is a
/// multiple of it). Panics when both or neither of the epilogue tile and
/// the subtile columns are given, for an `accumulator_alignment_bytes`
/// that is not a positive multiple of 4, and as
/// [`plan_accumulator_buffering_columns`] does.
#[track_caller]
pub fn plan_accumulator_buffering(
    one_stage_layout: &Layout,
    accumulator_dtype: DType,
    options: &AccumulatorBufferingOptions,
) -> AccumulatorBufferingPlan {
    const WHAT: &str = "cute: experimental::plan_accumulator_buffering";
    let o = options;
    assert!(
        o.accumulator_alignment_bytes > 0 && o.accumulator_alignment_bytes.is_multiple_of(4),
        "{WHAT}: accumulator_alignment_bytes {} is not a positive multiple of 4",
        o.accumulator_alignment_bytes
    );
    let subtile = match (&o.epilogue_tile, o.epilogue_subtile_columns) {
        (None, Some(c)) => c,
        (Some(tile), None) => epilogue_subtile_columns(one_stage_layout, accumulator_dtype, tile),
        (None, None) => panic!("{WHAT}: give the epilogue tile or the epilogue subtile columns"),
        (Some(_), Some(_)) => panic!("{WHAT}: give the epilogue tile or the epilogue subtile columns, not both"),
    };
    let one_stage_columns = compute_tmem_cols_from_layout(one_stage_layout, accumulator_dtype);
    let selection = plan_accumulator_buffering_columns(
        one_stage_columns,
        subtile,
        &o.auxiliary_tmem_allocations,
        o.tmem_capacity_columns,
        o.desired_disjoint_depth,
        o.max_auto_disjoint_depth,
        o.allow_overlap,
        o.overlap_alignment_columns,
        o.accumulator_alignment_bytes / 4,
        o.overlap_policy,
    );
    let column_stride = match one_stage_layout.stride() {
        Tree::Branch(modes) if !modes.is_empty() => match &modes[0] {
            Tree::Branch(m0) if m0.len() >= 2 => match &m0[1] {
                Tree::Leaf(leaf) => leaf.clone(),
                _ => panic!("{WHAT}: the N stride of {one_stage_layout} (stride[0][1]) is not a single stride"),
            },
            _ => panic!("{WHAT}: the accumulator layout {one_stage_layout} has no N stride at stride[0][1]"),
        },
        _ => panic!("{WHAT}: the accumulator layout {one_stage_layout} has no N stride at stride[0][1]"),
    };
    let views = Layout::new(
        Tree::Leaf(IndexExpr::Static(selection.storage_depth)),
        Tree::Leaf(column_stride * IndexExpr::Static(selection.stage_stride_columns)),
    );
    let storage_layout = one_stage_layout.append(views);
    let actual = compute_tmem_cols_from_layout(&storage_layout, accumulator_dtype);
    assert!(
        actual == selection.physical_columns,
        "{WHAT}: the storage layout {storage_layout} spans {actual} TMEM columns, but the plan counts {}",
        selection.physical_columns
    );
    AccumulatorBufferingPlan::new(selection, storage_layout)
}

/// The accumulator pipeline and the view selector carried through an
/// epilogue's drain (CuTe DSL's `AccumulatorDrain`), from
/// [`AccumulatorBufferingPlan::consumer_wait_and_get_view`]. The epilogue
/// visits its subtiles in [`subtile_index`](Self::subtile_index) order and
/// calls [`after_t2r`](Self::after_t2r) after each subtile's TMEM load,
/// which releases the view at the plan's ordinal. Implements [`Carry`] (the
/// pipeline's states and the selector).
#[derive(Clone, Debug)]
pub struct AccumulatorDrain<P> {
    plan: AccumulatorBufferingPlan,
    pipeline: P,
    consumer_selector: Int32,
}

impl<P: AsGenericPipeline> AccumulatorDrain<P> {
    /// The plan.
    pub fn plan(&self) -> &AccumulatorBufferingPlan {
        &self.plan
    }

    /// The pipeline (released and advanced once [`after_t2r`](Self::after_t2r)
    /// passed the release ordinal).
    pub fn pipeline(&self) -> &P {
        &self.pipeline
    }

    /// The pipeline, to carry on with after the drain.
    pub fn into_pipeline(self) -> P {
        self.pipeline
    }

    /// The view being drained (the stage index, or the round's parity).
    pub fn consumer_selector(&self) -> Int32 {
        self.consumer_selector
    }

    /// The subtile the epilogue's `ordinal`-th load reads, of `count`: the
    /// ordinal itself, except for view 0 of an overlapping plan, which is
    /// drained in reverse (its shared columns are its last).
    #[track_caller]
    pub fn subtile_index(&self, ordinal: usize, count: usize) -> Int32 {
        assert!(
            ordinal < count,
            "cute: experimental::AccumulatorDrain::subtile_index: ordinal {ordinal} of {count} subtiles"
        );
        if self.plan.uses_overlap() {
            select(self.consumer_selector.eq(0), (count - 1 - ordinal) as i32, ordinal as i32)
        } else {
            Int32::from(ordinal as i32)
        }
    }

    /// Call after the TMEM load of the epilogue's `ordinal`-th subtile:
    /// at the plan's [`release_after_t2r_ordinal`](AccumulatorBufferingPlan::release_after_t2r_ordinal)
    /// the view is released and the consumer moves on
    /// (`consumer_release_and_advance`; a T2R consumer waits for its loads
    /// first). Returns whether it released.
    #[track_caller]
    pub fn after_t2r(&mut self, ordinal: usize) -> bool {
        let release = ordinal == self.plan.release_after_t2r_ordinal();
        if release {
            self.pipeline.base_mut().consumer_release_and_advance();
        }
        release
    }
}

impl<P: AsGenericPipeline> Carry for AccumulatorDrain<P> {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.pipeline.flatten(out);
        self.consumer_selector.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        AccumulatorDrain {
            plan: self.plan.clone(),
            pipeline: self.pipeline.rebuild(values),
            consumer_selector: self.consumer_selector.rebuild(values),
        }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        if self.plan != other.plan {
            return Some("an accumulator drain changed its plan".to_string());
        }
        self.pipeline.structure_mismatch(&other.pipeline)
    }
}

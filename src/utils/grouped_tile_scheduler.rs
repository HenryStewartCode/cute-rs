// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/helpers/grouped_gemm_persistent_tile_scheduler.py
//   (the warp-wide group search of `_group_search` and `_prefix_sum`, the
//   double-buffered problem shapes of `prefetch_problem_shapes`, and the
//   tile decode of `delinearize_z` and `_compute_cta_tile_coord`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The tile scheduler of a persistent grouped GEMM (CuTe DSL 4.8.1's
//! `StaticPersistentGroupTileScheduler`, from the examples' helper module;
//! the deprecated `cutlass.utils` copy adds the switch between direct and
//! cached problem-shape loads, [`use_cached_problem_shapes`]).
//!
//! A grouped GEMM runs `group_count` GEMMs of different shapes in one
//! launch. Their cluster tiles are numbered one group after another, group
//! by group, M fastest within a group, and the static persistent scheduler
//! walks those numbers. To find a number's group, a warp searches the
//! groups 32 at a time: each lane counts one group's cluster tiles, a
//! prefix sum over the warp gives where each group ends, and a ballot finds
//! the first group that ends past the number. The search state carries
//! from one tile to the next, so each search starts at the previous tile's
//! group.
//!
//! The search is warp-collective (shuffles and a ballot over the full
//! warp): every lane of the warp must call
//! [`get_current_work`](StaticPersistentGroupTileScheduler::get_current_work)
//! together.
//!
//! [`use_cached_problem_shapes`]: StaticPersistentGroupTileScheduler::create

use crate::arch::{Dim3, WARP_SIZE, lane_idx, popc, shuffle_sync, shuffle_sync_up, vote_ballot_sync};
use crate::control::{Carry, if_else_yield, while_carry};
use crate::ir::ValueId;
use crate::layout::{IndexExpr, Layout};
use crate::tensor::{Tensor, make_rmem_tensor};
use crate::value::{Boolean, Int32, select};
use crate::{atom::copy::autovec_copy, layout::Shape};

use super::{PersistentTileSchedulerParams, StaticPersistentTileScheduler};

/// A problem shape `(M, N, K, L)` in registers.
type ProblemShape = [Int32; 4];

/// The group a tile belongs to and where the tile is in it (CuTe DSL's
/// `GroupSearchResult`): the group's index, the CTA tile's coordinate
/// along M and N within the group, the group's problem shape `(M, N, K)`,
/// and its number of tiles along K. Carried through loops ([`Carry`], 7
/// values).
#[derive(Clone, Copy, Debug)]
pub struct GroupSearchResult {
    pub group_idx: Int32,
    pub cta_tile_idx_m: Int32,
    pub cta_tile_idx_n: Int32,
    pub problem_shape_m: Int32,
    pub problem_shape_n: Int32,
    pub problem_shape_k: Int32,
    pub cta_tile_count_k: Int32,
}

impl GroupSearchResult {
    fn fields(&self) -> [Int32; 7] {
        [
            self.group_idx,
            self.cta_tile_idx_m,
            self.cta_tile_idx_n,
            self.problem_shape_m,
            self.problem_shape_n,
            self.problem_shape_k,
            self.cta_tile_count_k,
        ]
    }

    fn from_fields(f: [Int32; 7]) -> GroupSearchResult {
        GroupSearchResult {
            group_idx: f[0],
            cta_tile_idx_m: f[1],
            cta_tile_idx_n: f[2],
            problem_shape_m: f[3],
            problem_shape_n: f[4],
            problem_shape_k: f[5],
            cta_tile_count_k: f[6],
        }
    }
}

impl Carry for GroupSearchResult {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.fields().flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        GroupSearchResult::from_fields(self.fields().rebuild(values))
    }
}

/// How far the group search has got (CuTe DSL's
/// `GroupedGemmGroupSearchState`): the group to start the next search at,
/// the cluster tiles before that group, the tiles counted so far (to the
/// end of the matched group once one is found), and whether the last
/// search found its tile. Carried through loops ([`Carry`], 4 values).
#[derive(Clone, Copy, Debug)]
pub struct GroupedGemmGroupSearchState {
    pub start_group_idx: Int32,
    pub tile_count_prev_group: Int32,
    pub tile_count_searched: Int32,
    pub found: Boolean,
}

impl Carry for GroupedGemmGroupSearchState {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.start_group_idx.flatten(out);
        self.tile_count_prev_group.flatten(out);
        self.tile_count_searched.flatten(out);
        self.found.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        GroupedGemmGroupSearchState {
            start_group_idx: self.start_group_idx.rebuild(values),
            tile_count_prev_group: self.tile_count_prev_group.rebuild(values),
            tile_count_searched: self.tile_count_searched.rebuild(values),
            found: self.found.rebuild(values),
        }
    }
}

/// The search state before the first tile (CuTe DSL's
/// `create_initial_search_state`): group 0, nothing counted, not found.
#[track_caller]
pub fn create_initial_search_state() -> GroupedGemmGroupSearchState {
    GroupedGemmGroupSearchState {
        start_group_idx: Int32::from(0),
        tile_count_prev_group: Int32::from(0),
        tile_count_searched: Int32::from(0),
        found: Boolean::from(false),
    }
}

/// A grouped GEMM's work tile (CuTe DSL's `GroupedWorkTileInfo`): the
/// persistent scheduler's raw tile coordinate (the CTA's rank in its
/// cluster along M and N, and the linear cluster tile index), whether the
/// search found a group for it, and what it found. Carried through loops
/// ([`Carry`], 11 values).
#[derive(Clone, Copy, Debug)]
pub struct GroupedWorkTileInfo {
    tile_idx: [Int32; 3],
    is_valid_tile: Boolean,
    group_search_result: GroupSearchResult,
}

impl GroupedWorkTileInfo {
    /// A work tile.
    pub fn new(tile_idx: [Int32; 3], is_valid_tile: Boolean, group_search_result: GroupSearchResult) -> Self {
        GroupedWorkTileInfo { tile_idx, is_valid_tile, group_search_result }
    }

    /// The raw tile coordinate from the persistent scheduler.
    pub fn tile_idx(&self) -> [Int32; 3] {
        self.tile_idx
    }

    /// Whether the tile is work: false once the scheduler has run past the
    /// last group's tiles.
    pub fn is_valid_tile(&self) -> Boolean {
        self.is_valid_tile
    }

    /// The group and the tile's place in it.
    pub fn group_search_result(&self) -> GroupSearchResult {
        self.group_search_result
    }
}

impl Carry for GroupedWorkTileInfo {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.tile_idx.flatten(out);
        self.is_valid_tile.flatten(out);
        self.group_search_result.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        GroupedWorkTileInfo {
            tile_idx: self.tile_idx.rebuild(values),
            is_valid_tile: self.is_valid_tile.rebuild(values),
            group_search_result: self.group_search_result.rebuild(values),
        }
    }
}

/// The parts of the search that do not change: where the problem shapes
/// are, how many groups there are and the cluster tile.
#[derive(Clone, Debug)]
struct Groups {
    problem_shape_mnkl: Tensor<i32>,
    group_count: usize,
    cluster_tile_shape_mnk: [usize; 3],
    cluster_shape_mn: [usize; 2],
}

/// The values the search loop carries: the start group, the tiles before
/// the matched group, the tiles counted, whether the tile is still not
/// found, and the two cached problem shapes.
type SearchLoop = (Int32, Int32, Int32, Boolean, ProblemShape, ProblemShape);

impl Groups {
    fn group_count_i32(&self) -> i32 {
        i32::try_from(self.group_count).expect("cute: the group count fits in i32")
    }

    fn tile(&self, mode: usize) -> i32 {
        i32::try_from(self.cluster_tile_shape_mnk[mode]).expect("cute: the cluster tile fits in i32")
    }

    /// Group `group`'s `(M, N, K, L)`, from global memory into registers
    /// (CuTe DSL's `_get_problem_for_group`: an `autovec_copy` of the
    /// group's row into a 4-element register tensor).
    #[track_caller]
    fn load(&self, group: Int32) -> ProblemShape {
        let row = self.problem_shape_mnkl.slice((IndexExpr::from(group), ..));
        let regs = make_rmem_tensor::<i32>(Layout::col_major(4));
        autovec_copy(&row, &regs);
        [0, 1, 2, 3].map(|i| regs.load(i))
    }

    /// `load(group)` when `cond`, `otherwise` if not.
    #[track_caller]
    fn load_if(&self, cond: Boolean, group: Int32, otherwise: ProblemShape) -> ProblemShape {
        if_else_yield(cond, || self.load(group), || otherwise)
    }

    /// A problem's cluster tiles along M times along N (CuTe DSL's
    /// `_get_cluster_tile_count_mn`).
    #[track_caller]
    fn cluster_tile_count_mn(&self, problem: &ProblemShape) -> Int32 {
        let (tm, tn) = (self.tile(0), self.tile(1));
        ((problem[0] + (tm - 1)) / tm) * ((problem[1] + (tn - 1)) / tn)
    }

    /// The inclusive prefix sum of `value` over the warp (CuTe DSL's
    /// `_prefix_sum`): `shuffle_sync_up` by 1, 2, 4, 8 and 16, each lane
    /// adding when it is at least that far in.
    #[track_caller]
    fn prefix_sum(&self, value: Int32, lane: Int32) -> Int32 {
        let mut sum = value;
        let mut offset = 1;
        while offset < WARP_SIZE as i32 {
            let up = shuffle_sync_up(sum, offset);
            sum = select(lane.ge(offset), sum + up, sum);
            offset <<= 1;
        }
        sum
    }

    /// The group of cluster tile `linear_idx` (CuTe DSL's `_group_search`),
    /// starting at group `init_group_idx` with `init_tile_count_searched`
    /// tiles before it, and `prev` the tiles before the last matched group.
    /// Slides a window of 32 groups along until a group ends past the
    /// tile, or the groups run out. With `cached`, each lane's group comes
    /// from the cached shapes, which it refills as CuTe DSL's double
    /// buffer does.
    #[track_caller]
    fn search(
        &self,
        linear_idx: Int32,
        init_group_idx: Int32,
        init_tile_count_searched: Int32,
        prev: Int32,
        cached: Option<(ProblemShape, ProblemShape)>,
    ) -> (GroupedGemmGroupSearchState, Option<(ProblemShape, ProblemShape)>) {
        let lane = lane_idx();
        let group_count = self.group_count_i32();
        let warp = WARP_SIZE as i32;
        let not_found = linear_idx.ge(init_tile_count_searched);
        let start_not_found = not_found;
        let unused = [Int32::from(-1); 4];
        let (c0, c1) = cached.unwrap_or((unused, unused));
        let is_cached = cached.is_some();
        let init: SearchLoop = (init_group_idx, prev, init_tile_count_searched, not_found, c0, c1);
        let (start, prev, searched, not_found, c0, c1) = while_carry(
            init,
            |(start, _, _, not_found, _, _)| not_found & start.lt(group_count),
            |(start, _, searched, _, _, c1)| {
                let cur_group_idx = start + lane;
                let inside = cur_group_idx.lt(group_count);
                let (c0, mut c1) = (c1, c1);
                let ntile = if is_cached {
                    let next = start + warp + lane;
                    c1 = self.load_if(next.lt(group_count), next, c1);
                    select(inside, self.cluster_tile_count_mn(&c0), 0)
                } else {
                    if_else_yield(inside, || self.cluster_tile_count_mn(&self.load(cur_group_idx)), || Int32::from(0))
                };
                let end = self.prefix_sum(ntile, lane) + searched;
                let hit = popc(vote_ballot_sync(linear_idx.ge(end))).to::<i32>();
                let not_found = hit.eq(warp);
                let start = start + hit;
                // Every lane shuffles; the hit is the same in every lane.
                let before_hit = shuffle_sync(end, hit - 1);
                let prev = select(hit.eq(0), searched, before_hit);
                let searched = shuffle_sync(end, select(not_found, warp - 1, hit));
                if is_cached {
                    let first = start + lane;
                    c1 = self.load_if(!not_found & first.lt(group_count), first, c1);
                }
                (start, prev, searched, not_found, c0, c1)
            },
        );
        let state = GroupedGemmGroupSearchState {
            start_group_idx: start,
            tile_count_prev_group: prev,
            tile_count_searched: searched,
            found: start_not_found.ne(not_found),
        };
        (state, is_cached.then_some((c0, c1)))
    }

    /// The CTA tile of a group's cluster tile `cluster_tile_idx`, decoded
    /// M fastest over the group's `(cluster_count_m, cluster_count_n)`
    /// cluster tiles, plus the CTA's coordinate in its cluster (CuTe DSL's
    /// `_compute_cta_tile_coord`).
    #[track_caller]
    fn cta_tile_coord(
        &self,
        cluster_tile_idx: Int32,
        cta_in_cluster: [Int32; 2],
        cluster_count_m: Int32,
        cluster_count_n: Int32,
    ) -> (Int32, Int32) {
        // A tile without a group has no cluster tiles (or a negative
        // count); decoding over at least one keeps it free of a division by
        // zero. Found tiles have at least one along each mode.
        let shape = Shape::from((IndexExpr::from(cluster_count_m.max(1)), IndexExpr::from(cluster_count_n.max(1))));
        let coord = Layout::col_major(shape).get_hier_coord(cluster_tile_idx);
        let leaves = coord.leaves();
        let (mi, ni) = (Int32::from(leaves[0].clone()), Int32::from(leaves[1].clone()));
        let [cm, cn] = self.cluster_shape_mn.map(|c| i32::try_from(c).expect("cute: the cluster shape fits in i32"));
        (mi * cm + cta_in_cluster[0], ni * cn + cta_in_cluster[1])
    }

    /// `ceil_div` of a problem's M, N and K by the cluster tile.
    #[track_caller]
    fn cluster_counts(&self, problem: &ProblemShape) -> [Int32; 3] {
        [0, 1, 2].map(|i| {
            let t = self.tile(i);
            (problem[i] + (t - 1)) / t
        })
    }
}

/// A static persistent tile scheduler over the tiles of a grouped GEMM
/// (CuTe DSL's `StaticPersistentGroupTileScheduler`): the static persistent
/// scheduler ([`StaticPersistentTileScheduler`]) hands out linear cluster
/// tile indices, and a warp-wide group search turns each into a group and a
/// CTA tile in it ([`GroupSearchResult`]).
///
/// Make the [`PersistentTileSchedulerParams`] with the problem
/// `(cluster_m, cluster_n, total_cluster_tiles)` in tiles, so that the
/// work index is the linear cluster tile; the groups' shapes are a
/// `(group_count, 4)` tensor of `i32` `(M, N, K, L)` rows.
///
/// The search is warp-collective: every lane of the warp must call
/// [`get_current_work`](Self::get_current_work) (and
/// [`prefetch_problem_shapes`](Self::prefetch_problem_shapes)) together, in
/// converged code; a call from part of a warp reads lanes that did not
/// take part, which the simulator reports. Carried through loops
/// ([`Carry`]): the work index and tile count, the search state and, with
/// cached shapes, the two cached shapes.
#[derive(Clone, Debug)]
pub struct StaticPersistentGroupTileScheduler {
    inner: StaticPersistentTileScheduler,
    groups: Groups,
    search_state: GroupedGemmGroupSearchState,
    cached: Option<(ProblemShape, ProblemShape)>,
}

impl StaticPersistentGroupTileScheduler {
    /// The scheduler of the CTA at `block_idx` in a grid of `grid_dim`
    /// (CuTe DSL's `create`), for `group_count` groups whose shapes are the
    /// rows of `problem_shape_mnkl` (`(group_count, 4)`, `i32`), searched
    /// in cluster tiles of `cluster_tile_shape_mnk` from
    /// `initial_search_state` ([`create_initial_search_state`]).
    ///
    /// With `use_cached_problem_shapes` (the deprecated `cutlass.utils`
    /// copy's switch; the examples' helper always caches) each lane keeps
    /// its group's shape in registers, double-buffered: the window being
    /// searched and the next one, loaded while the warp counts the first.
    /// Call [`prefetch_problem_shapes`](Self::prefetch_problem_shapes) once
    /// before the first search then. Without it every search loads the
    /// shapes it needs. Both give the same tiles.
    ///
    /// Panics for 0 groups, for a tensor whose shape is not `(group_count,
    /// 4)`, and for a cluster tile with a 0 extent.
    #[track_caller]
    pub fn create(
        params: &PersistentTileSchedulerParams,
        block_idx: Dim3,
        grid_dim: Dim3,
        cluster_tile_shape_mnk: [usize; 3],
        initial_search_state: GroupedGemmGroupSearchState,
        group_count: usize,
        problem_shape_mnkl: &Tensor<i32>,
        use_cached_problem_shapes: bool,
    ) -> Self {
        let groups = make_groups(
            "StaticPersistentGroupTileScheduler",
            params,
            cluster_tile_shape_mnk,
            group_count,
            problem_shape_mnkl,
        );
        let unused = [Int32::from(-1); 4];
        StaticPersistentGroupTileScheduler {
            inner: StaticPersistentTileScheduler::create(params, block_idx, grid_dim),
            groups,
            search_state: initial_search_state,
            cached: use_cached_problem_shapes.then_some((unused, unused)),
        }
    }

    /// The underlying static persistent scheduler.
    pub fn scheduler(&self) -> &StaticPersistentTileScheduler {
        &self.inner
    }

    /// The schedule.
    pub fn params(&self) -> &PersistentTileSchedulerParams {
        self.inner.params()
    }

    /// The number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.group_count
    }

    /// The cluster tile the groups are counted in.
    pub fn cluster_tile_shape_mnk(&self) -> [usize; 3] {
        self.groups.cluster_tile_shape_mnk
    }

    /// The groups' `(M, N, K, L)`.
    pub fn problem_shape_mnkl(&self) -> &Tensor<i32> {
        &self.groups.problem_shape_mnkl
    }

    /// The search state after the last search.
    pub fn search_state(&self) -> GroupedGemmGroupSearchState {
        self.search_state
    }

    /// Whether the problem shapes are cached in registers.
    pub fn use_cached_problem_shapes(&self) -> bool {
        self.cached.is_some()
    }

    /// Load the first 32 groups' shapes into the cache, one per lane (CuTe
    /// DSL's `prefetch_problem_shapes`): call it once, from the whole warp,
    /// before the first search when the shapes are cached. Does nothing
    /// otherwise.
    #[track_caller]
    pub fn prefetch_problem_shapes(&mut self) {
        let _at = crate::builder::LocationScope::new();
        if let Some((c0, c1)) = self.cached {
            let lane = lane_idx();
            let c1 = self.groups.load_if(lane.lt(self.groups.group_count_i32()), lane, c1);
            self.cached = Some((c0, c1));
        }
    }

    /// The tile at the current work index (CuTe DSL's `get_current_work`):
    /// the raw tile of the static scheduler, run through
    /// [`delinearize_z`](Self::delinearize_z). Updates the search state.
    #[track_caller]
    pub fn get_current_work(&mut self) -> GroupedWorkTileInfo {
        let _at = crate::builder::LocationScope::new();
        let tile = self.inner.get_current_work().tile_idx();
        self.delinearize_z(tile)
    }

    /// The first tile (CuTe DSL's `initial_work_tile_info`).
    #[track_caller]
    pub fn initial_work_tile_info(&mut self) -> GroupedWorkTileInfo {
        self.get_current_work()
    }

    /// Search the group of the current work index, from the group the last
    /// search matched, and decode the tile in it (CuTe DSL's
    /// `delinearize_z`): the tile index within the group is the work index
    /// less the tiles before the group, decoded M fastest over the group's
    /// cluster tiles, and `cta_tile_coord`'s first two entries (the CTA's
    /// rank in its cluster) are added. A work index past the last group's
    /// tiles is invalid, with group -1 and shape -1.
    #[track_caller]
    pub fn delinearize_z(&mut self, cta_tile_coord: [Int32; 3]) -> GroupedWorkTileInfo {
        let _at = crate::builder::LocationScope::new();
        let linear_idx = self.inner.current_work_linear_idx();
        let prev = self.search_state;
        let (state, cached) = self.groups.search(
            linear_idx,
            prev.start_group_idx,
            prev.tile_count_prev_group,
            prev.tile_count_prev_group,
            self.cached,
        );
        self.search_state = state;
        self.cached = cached;
        let found = state.found;
        let group_idx = select(found, state.start_group_idx, -1);
        let problem = self.groups.load_if(found, state.start_group_idx, [Int32::from(-1); 4]);
        let result = decode(&self.groups, linear_idx, &state, group_idx, &problem, cta_tile_coord);
        GroupedWorkTileInfo { tile_idx: cta_tile_coord, is_valid_tile: found, group_search_result: result }
    }

    /// Move to the next tile (the static scheduler's
    /// [`advance_to_next_work`](StaticPersistentTileScheduler::advance_to_next_work)).
    #[track_caller]
    pub fn advance_to_next_work(&mut self) {
        self.inner.advance_to_next_work();
    }

    /// Move `advance_count` tiles on.
    #[track_caller]
    pub fn advance_to_next_work_by(&mut self, advance_count: impl crate::value::IntoValue<i32>) {
        self.inner.advance_to_next_work_by(advance_count);
    }

    /// The number of advances so far.
    pub fn num_tiles_executed(&self) -> Int32 {
        self.inner.num_tiles_executed()
    }

    /// The grid of a persistent launch
    /// ([`PersistentTileSchedulerParams::get_grid_shape`]).
    #[track_caller]
    pub fn get_grid_shape(params: &PersistentTileSchedulerParams, max_active_clusters: u32) -> [u32; 3] {
        params.get_grid_shape(max_active_clusters)
    }
}

/// The checked search constants of a grouped scheduler.
#[track_caller]
fn make_groups(
    what: &str,
    params: &PersistentTileSchedulerParams,
    cluster_tile_shape_mnk: [usize; 3],
    group_count: usize,
    problem_shape_mnkl: &Tensor<i32>,
) -> Groups {
    assert!(group_count >= 1, "cute: {what}: a grouped GEMM needs at least one group, not 0");
    assert!(
        cluster_tile_shape_mnk.iter().all(|&t| t >= 1),
        "cute: {what}: the cluster tile {cluster_tile_shape_mnk:?} has a 0 extent"
    );
    let shape = problem_shape_mnkl.layout().shape();
    let extents: Vec<Option<usize>> = (0..shape.rank()).map(|i| crate::layout::size(shape, &[i]).as_static()).collect();
    assert!(
        shape.rank() == 2 && extents[1] == Some(4) && extents[0].is_none_or(|g| g == group_count),
        "cute: {what}: the problem shapes must be a (group_count, 4) = ({group_count}, 4) tensor, not {shape}"
    );
    Groups {
        problem_shape_mnkl: problem_shape_mnkl.clone(),
        group_count,
        cluster_tile_shape_mnk,
        cluster_shape_mn: params.cluster_shape_mn(),
    }
}

/// The tile's [`GroupSearchResult`] once the search has run.
#[track_caller]
fn decode(
    groups: &Groups,
    linear_idx: Int32,
    state: &GroupedGemmGroupSearchState,
    group_idx: Int32,
    problem: &ProblemShape,
    cta_tile_coord: [Int32; 3],
) -> GroupSearchResult {
    let in_group = linear_idx - state.tile_count_prev_group;
    let [count_m, count_n, count_k] = groups.cluster_counts(problem);
    let (m, n) = groups.cta_tile_coord(in_group, [cta_tile_coord[0], cta_tile_coord[1]], count_m, count_n);
    GroupSearchResult {
        group_idx,
        cta_tile_idx_m: m,
        cta_tile_idx_n: n,
        problem_shape_m: problem[0],
        problem_shape_n: problem[1],
        problem_shape_k: problem[2],
        cta_tile_count_k: count_k,
    }
}

impl Carry for StaticPersistentGroupTileScheduler {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.inner.flatten(out);
        self.search_state.flatten(out);
        if let Some(cached) = &self.cached {
            cached.0.flatten(out);
            cached.1.flatten(out);
        }
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let inner = self.inner.rebuild(values);
        let search_state = self.search_state.rebuild(values);
        let cached = self.cached.map(|(c0, c1)| (c0.rebuild(values), c1.rebuild(values)));
        StaticPersistentGroupTileScheduler { inner, groups: self.groups.clone(), search_state, cached }
    }
}

/// The deprecated helper that turns a persistent scheduler's raw tile into
/// a grouped GEMM's tile (CuTe DSL's `GroupedGemmTileSchedulerHelper`),
/// kept as a thin wrapper over the search of
/// [`StaticPersistentGroupTileScheduler`]: the linear cluster tile index is
/// the raw tile's third entry, and the shapes are loaded directly. Where
/// CuTe DSL's helper searches on without bound past the last group, this
/// stops there, which only changes tiles that have no group. Carried
/// through loops ([`Carry`]: the search state).
#[deprecated(note = "use StaticPersistentGroupTileScheduler instead")]
#[derive(Clone, Debug)]
pub struct GroupedGemmTileSchedulerHelper {
    group_count: usize,
    tile_sched_params: PersistentTileSchedulerParams,
    cluster_tile_shape_mnk: [usize; 3],
    search_state: GroupedGemmGroupSearchState,
}

// The deprecated helper's own implementation uses its deprecated fields.
#[allow(deprecated)]
impl GroupedGemmTileSchedulerHelper {
    /// The helper for `group_count` groups, a schedule made with
    /// `tile_sched_params`, cluster tiles of `cluster_tile_shape_mnk` and
    /// a starting `search_state` ([`create_initial_search_state`]).
    pub fn new(
        group_count: usize,
        tile_sched_params: &PersistentTileSchedulerParams,
        cluster_tile_shape_mnk: [usize; 3],
        search_state: GroupedGemmGroupSearchState,
    ) -> Self {
        GroupedGemmTileSchedulerHelper {
            group_count,
            tile_sched_params: tile_sched_params.clone(),
            cluster_tile_shape_mnk,
            search_state,
        }
    }

    /// The search state after the last search.
    pub fn search_state(&self) -> GroupedGemmGroupSearchState {
        self.search_state
    }

    #[track_caller]
    fn groups(&self, problem_shape_mnkl: &Tensor<i32>) -> Groups {
        make_groups(
            "GroupedGemmTileSchedulerHelper",
            &self.tile_sched_params,
            self.cluster_tile_shape_mnk,
            self.group_count,
            problem_shape_mnkl,
        )
    }

    /// Search the group of `linear_idx` and load its shape.
    #[track_caller]
    fn search(&mut self, groups: &Groups, linear_idx: Int32) -> (Int32, ProblemShape) {
        let prev = self.search_state;
        let (state, _) = groups.search(
            linear_idx,
            prev.start_group_idx,
            prev.tile_count_prev_group,
            prev.tile_count_prev_group,
            None,
        );
        // The helper reports every search as found, as CuTe DSL's does.
        self.search_state = GroupedGemmGroupSearchState { found: Boolean::from(true), ..state };
        let problem = groups.load_if(state.found, state.start_group_idx, [Int32::from(-1); 4]);
        (state.start_group_idx, problem)
    }

    /// The group and tile of the raw tile `cta_tile_coord`, whose third
    /// entry is the linear cluster tile index, in the groups whose shapes
    /// are the rows of `problem_shape_mnkl` (CuTe DSL's `delinearize_z`).
    /// Warp-collective.
    #[track_caller]
    pub fn delinearize_z(&mut self, cta_tile_coord: [Int32; 3], problem_shape_mnkl: &Tensor<i32>) -> GroupSearchResult {
        let _at = crate::builder::LocationScope::new();
        let groups = self.groups(problem_shape_mnkl);
        let linear_idx = cta_tile_coord[2];
        let (group_idx, problem) = self.search(&groups, linear_idx);
        let state = self.search_state;
        decode(&groups, linear_idx, &state, group_idx, &problem, cta_tile_coord)
    }

    /// The matched group's tiles along K and its index (CuTe DSL's
    /// `search_cluster_tile_count_k`), for warps that need only those.
    /// Warp-collective.
    #[track_caller]
    pub fn search_cluster_tile_count_k(
        &mut self,
        cta_tile_coord: [Int32; 3],
        problem_shape_mnkl: &Tensor<i32>,
    ) -> (Int32, Int32) {
        let _at = crate::builder::LocationScope::new();
        let groups = self.groups(problem_shape_mnkl);
        let (group_idx, problem) = self.search(&groups, cta_tile_coord[2]);
        let t = groups.tile(2);
        ((problem[2] + (t - 1)) / t, group_idx)
    }
}

// The deprecated helper's own implementation uses its deprecated fields.
#[allow(deprecated)]
impl Carry for GroupedGemmTileSchedulerHelper {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.search_state.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        GroupedGemmTileSchedulerHelper { search_state: self.search_state.rebuild(values), ..self.clone() }
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! The group search of a grouped GEMM over one concatenated tensor (CuTe
//! DSL 4.8.1's `contiguous_group_search` and its state, from
//! `cutlass.utils.mixed_input_helpers`), as MoE kernels run: the groups lie
//! one after another along one mode (the search mode), and a `cumsum`
//! tensor of `group_count + 1` entries gives their boundaries, `cumsum[0] =
//! 0` and `cumsum[g + 1]` the end of group `g`. Each group is cut into
//! tiles of the cluster tile's extent along that mode, the last one
//! partial, so no tile crosses a boundary.

use crate::atom::copy::autovec_copy;
use crate::control::{Carry, while_carry};
use crate::ir::ValueId;
use crate::layout::Layout;
use crate::tensor::{Tensor, make_rmem_tensor};
use crate::value::{Boolean, Int32, IntoValue, select};

/// How far a contiguous group search has got along its mode (CuTe DSL's
/// `ContiguousGGSearchState`): the tiles before the current group, the
/// current group's end, the tiles counted to that end, the current group's
/// index **plus one** (an index into `cumsum`), the current group's start,
/// and, once the search is done, where the tile starts. Carried through
/// loops ([`Carry`], 6 values).
#[derive(Clone, Copy, Debug)]
pub struct ContiguousGGSearchState {
    pub last_tile_count: Int32,
    pub cur_boundary: Int32,
    pub cur_tile_count: Int32,
    pub cur_group_idx: Int32,
    pub cur_offset: Int32,
    pub cur_start: Int32,
}

impl ContiguousGGSearchState {
    fn fields(&self) -> [Int32; 6] {
        [
            self.last_tile_count,
            self.cur_boundary,
            self.cur_tile_count,
            self.cur_group_idx,
            self.cur_offset,
            self.cur_start,
        ]
    }

    fn from_fields(f: [Int32; 6]) -> Self {
        ContiguousGGSearchState {
            last_tile_count: f[0],
            cur_boundary: f[1],
            cur_tile_count: f[2],
            cur_group_idx: f[3],
            cur_offset: f[4],
            cur_start: f[5],
        }
    }
}

impl Carry for ContiguousGGSearchState {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.fields().flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        ContiguousGGSearchState::from_fields(self.fields().rebuild(values))
    }
}

/// The state before the first search (CuTe DSL's
/// `create_initial_contiguous_group_search_state`): every field 0.
#[track_caller]
pub fn create_initial_contiguous_group_search_state() -> ContiguousGGSearchState {
    ContiguousGGSearchState::from_fields([Int32::from(0); 6])
}

/// Find the group of tile `linear_idx` along mode `search_mode` (0 for M, 1
/// for N, 2 for K) of the cluster tile (CuTe DSL's
/// `contiguous_group_search`), carrying on from `search_state`, which must
/// come from a search of a smaller or equal `linear_idx` (or be the initial
/// state).
///
/// Group `g` (`cur_group_idx = g + 1`) spans `cumsum[g]` to `cumsum[g + 1]`
/// and holds `ceil_div(cumsum[g + 1] - cumsum[g], tile)` tiles. When the
/// tile is past the current group, the search walks the boundaries on until
/// a group's tiles reach past `linear_idx`, or past group `group_count`;
/// then `cur_start = cur_offset + tile * (linear_idx - last_tile_count)`
/// is where the tile starts along the mode. A `cur_group_idx` above
/// `group_count` means the tile is past the last group.
///
/// Every thread that calls it reads `cumsum` itself; there is no warp
/// collective. Panics for a `search_mode` above 2 and a 0 tile extent.
#[track_caller]
pub fn contiguous_group_search(
    cluster_tile_shape_mnk: [usize; 3],
    group_count: impl IntoValue<i32>,
    linear_idx: impl IntoValue<i32>,
    search_state: ContiguousGGSearchState,
    cumsum: &Tensor<i32>,
    search_mode: usize,
) -> ContiguousGGSearchState {
    assert!(search_mode <= 2, "cute: contiguous_group_search: search_mode is 0 (M), 1 (N) or 2 (K), not {search_mode}");
    let tile = cluster_tile_shape_mnk[search_mode];
    assert!(tile >= 1, "cute: contiguous_group_search: the cluster tile's extent along mode {search_mode} is 0");
    let tile = i32::try_from(tile).expect("cute: the cluster tile fits in i32");
    let _at = crate::builder::LocationScope::new();
    let (group_count, linear_idx) = (group_count.into_value(), linear_idx.into_value());
    let s = search_state;
    let not_found = linear_idx.ge(s.cur_tile_count);
    let cur_group_idx = select(not_found, s.cur_group_idx + 1, s.cur_group_idx);
    let init = (not_found, s.last_tile_count, s.cur_boundary, s.cur_tile_count, cur_group_idx, s.cur_offset);
    let (_, last_tile_count, cur_boundary, cur_tile_count, cur_group_idx, cur_offset) = while_carry(
        init,
        |(not_found, _, _, _, group, _): (Boolean, Int32, Int32, Int32, Int32, Int32)| {
            not_found & group.le(group_count)
        },
        |(_, _, cur_boundary, cur_tile_count, group, _)| {
            let next_boundary = cumsum.load(group);
            let next_tile_count = (next_boundary - cur_boundary + (tile - 1)) / tile + cur_tile_count;
            let not_found = linear_idx.ge(next_tile_count);
            let group = select(not_found, group + 1, group);
            (not_found, cur_tile_count, next_boundary, next_tile_count, group, cur_boundary)
        },
    );
    let cur_start = cur_offset + (linear_idx - last_tile_count) * tile;
    ContiguousGGSearchState { last_tile_count, cur_boundary, cur_tile_count, cur_group_idx, cur_offset, cur_start }
}

/// A tile of a contiguous grouped GEMM as a scheduling warp wrote it to
/// shared memory (CuTe DSL's `ContiguousGroupWorkTileInfo`): the CTA's
/// coordinate along M, the tile's start along N, its group (from 0), and
/// how far the group's end is from the tile's start. Carried through loops
/// ([`Carry`], 4 values; the group count is fixed).
#[derive(Clone, Copy, Debug)]
pub struct ContiguousGroupWorkTileInfo {
    pub cta_coord_m: Int32,
    pub coord_n: Int32,
    pub group_idx: Int32,
    pub distance_to_boundary: Int32,
    group_count: usize,
}

impl ContiguousGroupWorkTileInfo {
    /// A tile of `group_count` groups.
    pub fn new(
        group_count: usize,
        cta_coord_m: Int32,
        coord_n: Int32,
        group_idx: Int32,
        distance_to_boundary: Int32,
    ) -> Self {
        ContiguousGroupWorkTileInfo { cta_coord_m, coord_n, group_idx, distance_to_boundary, group_count }
    }

    /// The number of groups.
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Whether the tile is work: its group is below the group count.
    #[track_caller]
    pub fn is_valid_tile(&self) -> Boolean {
        self.group_idx.lt(i32::try_from(self.group_count).expect("cute: the group count fits in i32"))
    }
}

impl Carry for ContiguousGroupWorkTileInfo {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        [self.cta_coord_m, self.coord_n, self.group_idx, self.distance_to_boundary].flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let [cta_coord_m, coord_n, group_idx, distance_to_boundary] =
            [self.cta_coord_m, self.coord_n, self.group_idx, self.distance_to_boundary].rebuild(values);
        ContiguousGroupWorkTileInfo { cta_coord_m, coord_n, group_idx, distance_to_boundary, ..*self }
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        (self.group_count != other.group_count).then(|| {
            format!("a ContiguousGroupWorkTileInfo of {} groups became one of {}", self.group_count, other.group_count)
        })
    }
}

/// Read a tile's 4-entry record (`cta_coord_m`, `coord_n`, `group_idx`,
/// `distance_to_boundary`) from shared memory into registers with
/// [`autovec_copy`] (CuTe DSL's `make_contiguous_group_work_tile_info`).
/// Panics unless `s_tile_info` has 4 elements.
#[track_caller]
pub fn make_contiguous_group_work_tile_info(
    group_count: usize,
    s_tile_info: &Tensor<i32>,
) -> ContiguousGroupWorkTileInfo {
    let _at = crate::builder::LocationScope::new();
    let shape = s_tile_info.layout().shape().clone();
    assert_eq!(
        crate::layout::size(&shape, &[]).as_static(),
        Some(4),
        "cute: make_contiguous_group_work_tile_info: the tile record must have 4 entries, not {shape}"
    );
    let regs = make_rmem_tensor::<i32>(Layout::col_major(shape));
    autovec_copy(s_tile_info, &regs);
    ContiguousGroupWorkTileInfo::new(group_count, regs.load(0), regs.load(1), regs.load(2), regs.load(3))
}

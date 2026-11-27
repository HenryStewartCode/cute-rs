//! Kernels of the grouped GEMM schedulers that take arguments
//! of their own: the grouped scheduler's enumeration of every group's
//! tiles and the contiguous search's tile records, with their plain-Rust
//! models. Shared by `tests/grouped_schedulers.rs` and
//! `tests/gpu_cpasync_grouped.rs`.

// Tests the deprecated `GroupedGemmTileSchedulerHelper`, which still has to work.
#![allow(deprecated)]

use cute::prelude::*;
use cute::utils::{
    GroupedGemmTileSchedulerHelper, PersistentTileSchedulerParams, StaticPersistentGroupTileScheduler,
    StaticPersistentTileScheduler, contiguous_group_search, create_initial_contiguous_group_search_state,
    create_initial_search_state,
};

/// A small deterministic generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A grouped problem: the groups' `(M, N, K, L)`, the CTA tile, the
/// cluster shape along M and N, and the number of persistent clusters to
/// launch.
#[derive(Clone, Debug)]
pub struct GroupedProblem {
    pub shapes: Vec<[i32; 4]>,
    pub cta_tile: [usize; 3],
    pub cluster: [usize; 2],
    pub clusters: u32,
}

/// Fields of a tile record: group, CTA tile along M and N, the group's M,
/// N and K, its tiles along K, and the persistent cluster that visited it.
pub const RECORD: usize = 8;

impl GroupedProblem {
    /// `groups` random groups (M up to 70, N up to 50, K from 1 to 100),
    /// with about one in five having M = 0 or N = 0 when `empty_groups`.
    pub fn random(seed: u64, groups: usize, empty_groups: bool, cluster: [usize; 2], clusters: u32) -> Self {
        let mut rng = Rng::new(seed);
        let shapes = (0..groups)
            .map(|_| {
                let mut m = rng.below(71) as i32;
                let mut n = rng.below(51) as i32;
                if empty_groups && rng.below(5) == 0 {
                    if rng.below(2) == 0 {
                        m = 0;
                    } else {
                        n = 0;
                    }
                } else {
                    m = m.max(1);
                    n = n.max(1);
                }
                [m, n, 1 + rng.below(100) as i32, 1]
            })
            .collect();
        GroupedProblem { shapes, cta_tile: [16, 16, 32], cluster, clusters }
    }

    pub fn group_count(&self) -> usize {
        self.shapes.len()
    }

    /// The cluster tile: the CTA tile times the cluster along M and N.
    pub fn cluster_tile(&self) -> [usize; 3] {
        [self.cta_tile[0] * self.cluster[0], self.cta_tile[1] * self.cluster[1], self.cta_tile[2]]
    }

    fn ceil(a: i32, b: usize) -> i32 {
        (a + b as i32 - 1) / b as i32
    }

    /// Every group's cluster tiles, in order: (group, tile along M, tile
    /// along N) of each linear cluster tile.
    pub fn cluster_tiles(&self) -> Vec<(usize, i32, i32)> {
        let t = self.cluster_tile();
        let mut out = Vec::new();
        for (g, s) in self.shapes.iter().enumerate() {
            let (cm, cn) = (Self::ceil(s[0], t[0]), Self::ceil(s[1], t[1]));
            for i in 0..cm * cn {
                out.push((g, i % cm, i / cm));
            }
        }
        out
    }

    pub fn cta_per_cluster(&self) -> usize {
        self.cluster[0] * self.cluster[1]
    }

    /// The record each CTA tile should get: rows of linear cluster tile
    /// times CTAs per cluster, the CTA's rank in its cluster M fastest.
    pub fn model(&self) -> Vec<[i32; RECORD]> {
        let t = self.cluster_tile();
        let [cm, cn] = self.cluster;
        let mut rows = Vec::new();
        for (linear, (g, mi, ni)) in self.cluster_tiles().into_iter().enumerate() {
            let s = self.shapes[g];
            for rn in 0..cn {
                for rm in 0..cm {
                    rows.push([
                        g as i32,
                        mi * cm as i32 + rm as i32,
                        ni * cn as i32 + rn as i32,
                        s[0],
                        s[1],
                        s[2],
                        Self::ceil(s[2], t[2]),
                        (linear % self.clusters as usize) as i32,
                    ]);
                }
            }
        }
        rows
    }

    /// The schedule: `(cluster_m, cluster_n, total cluster tiles)` tiles,
    /// so that the work index is the linear cluster tile.
    pub fn params(&self) -> PersistentTileSchedulerParams {
        let total = self.cluster_tiles().len().max(1);
        PersistentTileSchedulerParams::new(
            (self.cluster[0], self.cluster[1], total),
            [self.cluster[0], self.cluster[1], 1],
            Default::default(),
        )
    }

    /// The grid: the clusters' CTAs along x and y, the persistent clusters
    /// along z.
    pub fn grid(&self) -> [u32; 3] {
        [self.cluster[0] as u32, self.cluster[1] as u32, self.clusters]
    }

    pub fn ctas(&self) -> usize {
        self.grid().iter().product::<u32>() as usize
    }

    /// The `(group_count, 4)` problem-shape tensor's data.
    pub fn shape_data(&self) -> Vec<i32> {
        self.shapes.iter().flatten().copied().collect()
    }
}

/// How the enumeration kernel finds each tile's group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// `StaticPersistentGroupTileScheduler`, loading shapes directly.
    Direct,
    /// `StaticPersistentGroupTileScheduler` with cached shapes.
    Cached,
    /// The deprecated `GroupedGemmTileSchedulerHelper` over a
    /// `StaticPersistentTileScheduler`.
    Helper,
}

impl SearchMode {
    pub const ALL: [SearchMode; 3] = [SearchMode::Direct, SearchMode::Cached, SearchMode::Helper];
}

/// Threads per block of [`grouped_enumeration_kernel`]: two warps, each
/// running its own search.
pub const ENUM_THREADS: u32 = 64;

/// Every CTA walks its persistent tiles with the grouped scheduler of
/// `mode`, in both warps, until a tile has no group; thread 0 records each
/// tile ([`RECORD`] fields) at row `linear * ctas_per_cluster + rank`, and
/// at the end the tiles its CTA executed into `count`.
pub fn grouped_enumeration_kernel(p: &GroupedProblem, mode: SearchMode) -> Kernel {
    let groups = p.group_count();
    let rows = p.model().len().max(1);
    let ctas = p.ctas();
    let params = p.params();
    let cluster_tile = p.cluster_tile();
    let [cm, cn] = p.cluster;
    let name = match mode {
        SearchMode::Direct => "grouped_enumeration_direct",
        SearchMode::Cached => "grouped_enumeration_cached",
        SearchMode::Helper => "grouped_enumeration_helper",
    };
    Kernel::builder(name).block_dim([ENUM_THREADS, 1, 1]).build(move || {
        let shapes = param_tensor::<i32>("problem_shape_mnkl", Layout::row_major((groups, 4)));
        let records = param_tensor::<i32>("records", Layout::row_major((rows, RECORD)));
        let count = param_tensor::<i32>("count", Layout::col_major(ctas));
        let b = block_idx();
        let cta = b.x + b.y * cm as i32 + b.z * (cm * cn) as i32;
        let rank = b.x + b.y * cm as i32;
        let record = |linear: Int32, r: &cute::utils::GroupSearchResult| {
            if_(thread_idx().x.eq(0), || {
                let row = linear * (cm * cn) as i32 + rank;
                let fields = [
                    r.group_idx,
                    r.cta_tile_idx_m,
                    r.cta_tile_idx_n,
                    r.problem_shape_m,
                    r.problem_shape_n,
                    r.problem_shape_k,
                    r.cta_tile_count_k,
                    b.z,
                ];
                for (i, f) in fields.into_iter().enumerate() {
                    records.store((row, i), f);
                }
            });
        };
        let executed = match mode {
            SearchMode::Direct | SearchMode::Cached => {
                let mut sched = StaticPersistentGroupTileScheduler::create(
                    &params,
                    block_idx(),
                    grid_dim(),
                    cluster_tile,
                    create_initial_search_state(),
                    groups,
                    &shapes,
                    mode == SearchMode::Cached,
                );
                sched.prefetch_problem_shapes();
                let work = sched.initial_work_tile_info();
                let (_, sched) = while_carry(
                    (work, sched),
                    |(work, _)| work.is_valid_tile(),
                    |(work, mut sched)| {
                        record(sched.scheduler().current_work_linear_idx(), &work.group_search_result());
                        sched.advance_to_next_work();
                        let work = sched.get_current_work();
                        (work, sched)
                    },
                );
                sched.num_tiles_executed()
            }
            SearchMode::Helper => {
                let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
                let helper =
                    GroupedGemmTileSchedulerHelper::new(groups, &params, cluster_tile, create_initial_search_state());
                let work = sched.initial_work_tile_info();
                let (_, sched, _) = while_carry(
                    (work, sched, helper),
                    |(work, _, _)| work.is_valid_tile(),
                    |(work, mut sched, mut helper)| {
                        let r = helper.delinearize_z(work.tile_idx(), &shapes);
                        record(work.tile_idx()[2], &r);
                        sched.advance_to_next_work();
                        (sched.get_current_work(), sched, helper)
                    },
                );
                sched.num_tiles_executed()
            }
        };
        if_(thread_idx().x.eq(0), || count.store(cta, executed));
    })
}

/// The launch and arguments of [`grouped_enumeration_kernel`]: the shapes,
/// records of -1 and counts of -1.
pub fn grouped_enumeration_launch(p: &GroupedProblem) -> (LaunchConfig, Vec<KernelArg>) {
    let rows = p.model().len().max(1);
    let args = vec![
        KernelArg::buffer(&p.shape_data()),
        KernelArg::buffer(&vec![-1i32; rows * RECORD]),
        KernelArg::buffer(&vec![-1i32; p.ctas()]),
    ];
    (LaunchConfig::new(p.grid(), [ENUM_THREADS, 1, 1]), args)
}

/// Check an enumeration run against [`GroupedProblem::model`]: every CTA
/// tile of every group was visited once, by the right persistent cluster,
/// with the right group, coordinates, shape and K tiles, and each CTA
/// executed its share of the cluster tiles.
pub fn check_grouped_enumeration(p: &GroupedProblem, args: &[KernelArg]) {
    let model = p.model();
    let records = args[1].to_vec::<i32>();
    for (i, want) in model.iter().enumerate() {
        assert_eq!(records[i * RECORD..(i + 1) * RECORD], want[..], "{p:?}: row {i}");
    }
    if model.is_empty() {
        assert!(records.iter().all(|&r| r == -1), "{p:?}: no tile to record");
    }
    let tiles = p.cluster_tiles().len();
    let counts = args[2].to_vec::<i32>();
    for (cta, &c) in counts.iter().enumerate() {
        let z = cta / p.cta_per_cluster();
        let want = (tiles + p.clusters as usize - 1 - z) / p.clusters as usize;
        assert_eq!(c as usize, want, "{p:?}: tiles executed by CTA {cta}");
    }
}

/// The problems the enumeration runs: 1, 5, 33 and 70 groups (the window
/// slides past 32 and 64), some with empty groups, a problem whose groups
/// all are empty, clusters of 1 and of 2 CTAs along M, and 1 to 7
/// persistent clusters.
pub fn grouped_problems() -> Vec<GroupedProblem> {
    let mut out = vec![
        GroupedProblem::random(1, 1, false, [1, 1], 3),
        GroupedProblem::random(2, 5, true, [1, 1], 4),
        GroupedProblem::random(3, 33, true, [1, 1], 7),
        GroupedProblem::random(4, 70, true, [1, 1], 5),
        GroupedProblem::random(5, 40, true, [2, 1], 3),
        GroupedProblem::random(6, 9, false, [1, 2], 1),
    ];
    let mut empty = GroupedProblem::random(7, 37, false, [1, 1], 2);
    for (i, s) in empty.shapes.iter_mut().enumerate() {
        if i != 36 {
            s[i % 2] = 0;
        }
    }
    out.push(empty);
    out
}

/// Words of a contiguous search record: the search state after the tile
/// (6 fields), then the tile's start and distance to its boundary.
pub const CONTIG_RECORD: usize = 8;

/// A contiguous grouped problem: the boundaries `cumsum` (`group_count + 1`
/// entries from 0), the tile extent along the search mode, and the number
/// of tiles each thread searches.
#[derive(Clone, Debug)]
pub struct ContiguousProblem {
    pub cumsum: Vec<i32>,
    pub tile: usize,
    pub tiles: usize,
}

impl ContiguousProblem {
    /// `groups` random groups of 0 to `max` rows (some empty).
    pub fn random(seed: u64, groups: usize, max: u64, tile: usize) -> Self {
        let mut rng = Rng::new(seed);
        let mut cumsum = vec![0];
        for _ in 0..groups {
            let rows = if rng.below(6) == 0 { 0 } else { rng.below(max + 1) as i32 };
            cumsum.push(cumsum.last().unwrap() + rows);
        }
        let mut p = ContiguousProblem { cumsum, tile, tiles: 0 };
        p.tiles = p.total_tiles() + 3;
        p
    }

    pub fn group_count(&self) -> usize {
        self.cumsum.len() - 1
    }

    pub fn total_tiles(&self) -> usize {
        self.cumsum.windows(2).map(|w| (w[1] - w[0]) as usize).map(|r| r.div_ceil(self.tile)).sum()
    }

    /// A plain-Rust transcription of the search, run tile after tile from
    /// the initial state: each tile's state (last tile count, boundary,
    /// tile count, group + 1, offset, start), then its start and distance
    /// to the boundary.
    pub fn model(&self) -> Vec<[i32; CONTIG_RECORD]> {
        let t = self.tile as i32;
        let gc = self.group_count() as i32;
        let mut s = [0i32; 6];
        let mut out = Vec::new();
        for linear in 0..self.tiles as i32 {
            let [mut last, mut boundary, mut count, mut group, mut offset, _] = s;
            let mut not_found = linear >= count;
            if not_found {
                group += 1;
            }
            while not_found && group <= gc {
                let next_boundary = self.cumsum[group as usize];
                let next_count = (next_boundary - boundary + t - 1) / t + count;
                not_found = linear >= next_count;
                last = count;
                offset = boundary;
                boundary = next_boundary;
                count = next_count;
                if not_found {
                    group += 1;
                }
            }
            let start = offset + t * (linear - last);
            s = [last, boundary, count, group, offset, start];
            out.push([last, boundary, count, group, offset, start, start, boundary - start]);
        }
        out
    }

    /// The tiles a direct reading of the boundaries gives: for each tile
    /// of each group, its group, start and rows to the group's end.
    pub fn direct_tiles(&self) -> Vec<(usize, i32, i32)> {
        let mut out = Vec::new();
        for g in 0..self.group_count() {
            let (a, b) = (self.cumsum[g], self.cumsum[g + 1]);
            let mut start = a;
            while start < b {
                out.push((g, start, b - start));
                start += self.tile as i32;
            }
        }
        out
    }
}

/// Each of 32 threads searches tiles `lane, lane + 32, ...` (an increasing
/// sequence, as a persistent scheduler hands them out) from the initial
/// state, through a loop that carries the state, and records each tile's
/// state, start and distance to its boundary.
pub fn contiguous_search_kernel(p: &ContiguousProblem) -> Kernel {
    let groups = p.group_count();
    let tiles = p.tiles;
    let tile = p.tile;
    Kernel::builder("contiguous_search").block_dim([32, 1, 1]).build(move || {
        let cumsum = param_tensor::<i32>("cumsum", Layout::col_major(groups + 1));
        let records = param_tensor::<i32>("records", Layout::row_major((tiles, CONTIG_RECORD)));
        let lane = lane_idx();
        let n = tiles as i32;
        let start = (lane, create_initial_contiguous_group_search_state());
        while_carry(
            start,
            |(linear, _)| linear.lt(n),
            |(linear, state)| {
                let s = contiguous_group_search([tile, 1, 1], groups as i32, linear, state, &cumsum, 0);
                let fields = [
                    s.last_tile_count,
                    s.cur_boundary,
                    s.cur_tile_count,
                    s.cur_group_idx,
                    s.cur_offset,
                    s.cur_start,
                    s.cur_start,
                    s.cur_boundary - s.cur_start,
                ];
                for (i, f) in fields.into_iter().enumerate() {
                    records.store((linear, i), f);
                }
                (linear + 32, s)
            },
        );
    })
}

/// The launch and arguments of [`contiguous_search_kernel`].
pub fn contiguous_search_launch(p: &ContiguousProblem) -> (LaunchConfig, Vec<KernelArg>) {
    let args = vec![KernelArg::buffer(&p.cumsum), KernelArg::buffer(&vec![-7i32; p.tiles * CONTIG_RECORD])];
    (LaunchConfig::new([1, 1, 1], [32, 1, 1]), args)
}

/// Check a [`contiguous_search_kernel`] run: each thread's search, carried
/// from its own earlier tiles, agrees with the transcription run tile by
/// tile for the tiles in a group (past the last group, only in that the
/// group is past it); and the valid tiles (group + 1 at most the group
/// count) are the
/// groups' tiles of a direct reading of the boundaries, with their starts
/// and distances to the boundary.
pub fn check_contiguous_search(p: &ContiguousProblem, args: &[KernelArg]) {
    let records = args[1].to_vec::<i32>();
    let model = p.model();
    let direct = p.direct_tiles();
    let gc = p.group_count() as i32;
    for (linear, want) in model.iter().enumerate() {
        let got = &records[linear * CONTIG_RECORD..(linear + 1) * CONTIG_RECORD];
        if want[3] <= gc {
            // A tile in a group gets the same state whichever earlier tiles
            // the thread searched.
            assert_eq!(got, &want[..], "{p:?}: tile {linear}");
            let (g, start, distance) = direct[linear];
            assert_eq!((want[3] - 1, want[5], want[7]), (g as i32, start, distance), "{p:?}: tile {linear}");
        } else {
            assert!(linear >= direct.len(), "{p:?}: tile {linear} is past the last group");
            assert!(got[3] > gc, "{p:?}: tile {linear} is past the last group, not in group {}", got[3]);
        }
    }
}

/// The contiguous problems: 1, 6 and 50 groups of up to 40 rows in tiles
/// of 8, and 20 groups in tiles of 16.
pub fn contiguous_problems() -> Vec<ContiguousProblem> {
    vec![
        ContiguousProblem::random(11, 1, 40, 8),
        ContiguousProblem::random(12, 6, 40, 8),
        ContiguousProblem::random(13, 50, 40, 8),
        ContiguousProblem::random(14, 20, 60, 16),
    ]
}

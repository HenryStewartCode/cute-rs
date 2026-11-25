// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Ported from CUTLASS 4.8.1's examples/python/CuTeDSL/helpers/dynamic_persistent_tile_scheduler.py
// (ClcDynamicPersistentTileSchedulerParams, ClcDynamicPersistentTileScheduler), Copyright (c)
// 2025 - 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause; see THIRD-PARTY-NOTICES.md.

//! The cluster launch control dynamic persistent tile scheduler: the grid covers the whole problem, each cluster starts
//! on the tile of its own position in the grid, and then claims the tiles of
//! still-queued clusters, which it takes off the launch queue
//! (`clusterlaunchcontrol.try_cancel`), until the queue is empty.

use crate::arch::{Dim3, elect_one, fence_view_async_shared, issue_clc_query};
use crate::control::{Carry, if_};
use crate::fast_divmod::FastDivmod;
use crate::ir::ValueId;
use crate::layout::{IndexExpr, Shape, size};
use crate::tensor::{Ptr, recast_ptr};
use crate::value::{Int32, Numeric};

use super::tile_scheduler::decode_cluster;
use super::{PersistentTileSchedulerOptions, PersistentTileSchedulerParams, WorkTileInfo};

/// The optional arguments of [`ClcDynamicPersistentTileSchedulerParams::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClcDynamicPersistentTileSchedulerOptions {
    /// Clusters per band of the visiting order
    /// ([`swizzle_and_rasterize`](super::swizzle_and_rasterize)); 1, the
    /// default, is no swizzle.
    pub swizzle_size: usize,
    /// Whether the clusters go along M first (the default) or along N.
    pub raster_along_m: bool,
    /// The fallback cluster shape of a launch with preferred and fallback
    /// clusters (m6's `LaunchConfig::with_fallback_cluster`): the params
    /// keep a second schedule for it
    /// ([`fallback`](ClcDynamicPersistentTileSchedulerParams::fallback)),
    /// which [`ClcDynamicPersistentTileScheduler::create`] takes in the
    /// fallback arm of [`mixed_cluster_arms`](super::mixed_cluster_arms).
    pub fallback_cluster_shape_mnk: Option<[usize; 3]>,
}

impl Default for ClcDynamicPersistentTileSchedulerOptions {
    fn default() -> Self {
        ClcDynamicPersistentTileSchedulerOptions {
            swizzle_size: 1,
            raster_along_m: true,
            fallback_cluster_shape_mnk: None,
        }
    }
}

/// What the cluster launch control scheduler knows of the problem (CuTe
/// DSL's `ClcDynamicPersistentTileSchedulerParams`): the tiles along M, N
/// and L, the cluster shape (one CTA deep along K) and the visiting order.
/// It shares the static persistent scheduler's layout of the problem's
/// clusters ([`PersistentTileSchedulerParams`], which it holds), so the two
/// visit the clusters in the same order.
#[derive(Clone, Debug)]
pub struct ClcDynamicPersistentTileSchedulerParams {
    schedule: PersistentTileSchedulerParams,
    fallback: Option<Box<ClcDynamicPersistentTileSchedulerParams>>,
}

impl ClcDynamicPersistentTileSchedulerParams {
    /// The schedule of `problem_shape_ntile_mnl` tiles (rank 3; static or
    /// runtime extents) in clusters of `cluster_shape_mnk` CTAs.
    ///
    /// Panics as [`PersistentTileSchedulerParams::new`] does (a cluster
    /// deeper than one CTA along K, a `swizzle_size` of 0), also for the
    /// fallback shape.
    #[track_caller]
    pub fn new(
        problem_shape_ntile_mnl: impl Into<Shape>,
        cluster_shape_mnk: [usize; 3],
        options: ClcDynamicPersistentTileSchedulerOptions,
    ) -> ClcDynamicPersistentTileSchedulerParams {
        let problem = problem_shape_ntile_mnl.into();
        let order = PersistentTileSchedulerOptions {
            swizzle_size: options.swizzle_size,
            raster_along_m: options.raster_along_m,
        };
        let fallback = options.fallback_cluster_shape_mnk.map(|shape| {
            let options = ClcDynamicPersistentTileSchedulerOptions { fallback_cluster_shape_mnk: None, ..options };
            Box::new(ClcDynamicPersistentTileSchedulerParams::new(problem.clone(), shape, options))
        });
        ClcDynamicPersistentTileSchedulerParams {
            schedule: PersistentTileSchedulerParams::new(problem, cluster_shape_mnk, order),
            fallback,
        }
    }

    /// The underlying schedule: tile counts, cluster shape, visiting order
    /// and the layout of the problem's clusters.
    pub fn schedule(&self) -> &PersistentTileSchedulerParams {
        &self.schedule
    }

    /// The schedule for the fallback cluster shape, if one was given.
    pub fn fallback(&self) -> Option<&ClcDynamicPersistentTileSchedulerParams> {
        self.fallback.as_deref()
    }

    /// The cluster shape (M, N, K).
    pub fn cluster_shape_mnk(&self) -> [usize; 3] {
        self.schedule.cluster_shape_mnk()
    }

    /// Whether the grid is the problem's CTA grid itself (no swizzle,
    /// rastered along M), so that a CTA's block index is its tile.
    fn grid_is_problem(&self) -> bool {
        self.schedule.swizzle_size() == 1 && self.schedule.raster_along_m()
    }

    /// The grid that launches the whole problem (CuTe DSL's
    /// `get_grid_shape`): without a swizzle and rastered along M, the tiles
    /// rounded up to whole clusters, `(M, N, L)`; otherwise `(cluster_m,
    /// cluster_n, clusters)`, the clusters flattened into z in visiting
    /// order.
    ///
    /// With a swizzle, CuTe DSL launches the clusters of the unpadded
    /// problem, which `get_flat_coord` of the padded layout then maps partly
    /// onto padding clusters, leaving tiles unvisited when the swizzle does
    /// not divide the problem's clusters. cute-rs launches the padded
    /// layout's clusters instead: every tile is visited, and the padding
    /// clusters' tiles, past the problem, are valid work that the kernel
    /// must skip (as with the static scheduler). Host code: panics for a
    /// runtime tile count.
    #[track_caller]
    pub fn get_grid_shape(&self) -> [u32; 3] {
        let [cm, cn, _] = self.cluster_shape_mnk();
        let extent = |shape: &Shape, i: usize| {
            size(shape, &[i]).as_static().unwrap_or_else(|| {
                panic!("cute: ClcDynamicPersistentTileSchedulerParams::get_grid_shape needs a static problem shape")
            }) as u32
        };
        if self.grid_is_problem() {
            let tiles = self.schedule.problem_shape_ntile_mnl();
            let round = |e: u32, c: usize| e.div_ceil(c as u32) * c as u32;
            [round(extent(tiles, 0), cm), round(extent(tiles, 1), cn), extent(tiles, 2)]
        } else {
            let shape = self.schedule.problem_layout_ncluster_mnl().shape();
            [cm as u32, cn as u32, extent(shape, 0) * extent(shape, 1) * extent(shape, 2)]
        }
    }
}

/// The cluster launch control dynamic persistent tile scheduler (CuTe DSL's
/// `ClcDynamicPersistentTileScheduler`). Launched with
/// [`get_grid_shape`](ClcDynamicPersistentTileSchedulerParams::get_grid_shape),
/// each CTA's first tile is that of its own cluster
/// ([`initial_work_tile_info`](Self::initial_work_tile_info)). The cluster's
/// scheduler warp then asks for the next one
/// ([`advance_to_next_work`](Self::advance_to_next_work): a multicast
/// `try_cancel` into the response at `clc_response_ptr`, through a
/// [`PipelineClcFetchAsync`](crate::pipeline::PipelineClcFetchAsync) stage),
/// and every CTA decodes it
/// ([`get_current_work`](Self::get_current_work)): the first CTA of the
/// cancelled cluster, put through the visiting order, plus the CTA's rank in
/// its cluster. Work stops when the response says nothing was cancelled.
///
/// Carried through loops ([`Carry`]): only the count of tiles executed
/// changes.
#[derive(Clone, Debug)]
pub struct ClcDynamicPersistentTileScheduler {
    params: ClcDynamicPersistentTileSchedulerParams,
    divisors: Option<(FastDivmod, FastDivmod)>,
    cta_id_in_cluster: [Int32; 3],
    num_tiles_executed: Int32,
    clc_response_ptr: Ptr<i32>,
    block_idx: Dim3,
    insert_fence: bool,
}

impl ClcDynamicPersistentTileScheduler {
    /// The scheduler of the CTA at `block_idx` (CuTe DSL's `create`), whose
    /// responses land at `clc_response_ptr` (16-byte aligned shared memory).
    /// Its rank in its cluster is `(block_idx.x % cluster_m, block_idx.y %
    /// cluster_n, 0)`. With `insert_fence`, decoding a response is followed
    /// by `fence.proxy.async.shared::cta`, which orders the generic read
    /// before the next response's async-proxy write (leave it out only if
    /// the pipeline fences itself). `grid_dim` is unused, as in CuTe DSL.
    ///
    /// In an arm of [`mixed_cluster_arms`](super::mixed_cluster_arms) the
    /// scheduler uses the schedule of the arm's cluster shape: `params` itself
    /// or its [`fallback`](ClcDynamicPersistentTileSchedulerParams::fallback)
    /// (CuTe DSL's `_select_active_params`, decided when the arm is built);
    /// panics when neither has the arm's shape.
    #[track_caller]
    pub fn create<T: Numeric>(
        params: &ClcDynamicPersistentTileSchedulerParams,
        block_idx: Dim3,
        grid_dim: Dim3,
        clc_response_ptr: &Ptr<T>,
        insert_fence: bool,
    ) -> Self {
        let _ = grid_dim;
        let params = match (super::current_cluster_arm(), params.fallback()) {
            (Some(arm), Some(f)) if f.cluster_shape_mnk() == arm.cluster_shape_mnk => f,
            _ => params,
        };
        super::check_cluster_shape_in_arm("ClcDynamicPersistentTileScheduler's schedule", params.cluster_shape_mnk());
        let [cm, cn, _] = params.cluster_shape_mnk();
        let cta_id_in_cluster = [
            Int32::from(IndexExpr::from(block_idx.x) % cm),
            Int32::from(IndexExpr::from(block_idx.y) % cn),
            Int32::from(0),
        ];
        let schedule = &params.schedule;
        let divisors = (schedule.swizzle_size() == 1 && !schedule.raster_along_m()).then(|| schedule.divisors());
        ClcDynamicPersistentTileScheduler {
            params: params.clone(),
            divisors,
            cta_id_in_cluster,
            num_tiles_executed: Int32::from(0),
            clc_response_ptr: recast_ptr::<i32, T>(clc_response_ptr, None),
            block_idx,
            insert_fence,
        }
    }

    /// The grid that launches the whole problem (CuTe DSL's static
    /// `get_grid_shape`).
    #[track_caller]
    pub fn get_grid_shape(params: &ClcDynamicPersistentTileSchedulerParams) -> [u32; 3] {
        params.get_grid_shape()
    }

    /// The schedule.
    pub fn params(&self) -> &ClcDynamicPersistentTileSchedulerParams {
        &self.params
    }

    /// This CTA's rank in its cluster along M, N and K.
    pub fn cta_id_in_cluster(&self) -> [Int32; 3] {
        self.cta_id_in_cluster
    }

    /// The number of queries issued so far.
    pub fn num_tiles_executed(&self) -> Int32 {
        self.num_tiles_executed
    }

    /// The first tile of the cluster whose first CTA is at `(x, y, z)` in the
    /// grid (CuTe DSL's `_swizzle_and_rasterize`), without the CTA's rank:
    /// the coordinates themselves when the grid is the problem's, and the
    /// visiting order's cluster of linear index `z` otherwise, scaled by the
    /// cluster shape.
    #[track_caller]
    fn swizzle_and_rasterize(&self, x: Int32, y: Int32, z: Int32) -> [Int32; 3] {
        if self.params.grid_is_problem() {
            return [x, y, z];
        }
        let [cm, cn, _] = self.params.cluster_shape_mnk();
        let [m, n, l] = decode_cluster(&self.params.schedule, self.divisors, z);
        [m * cm as i32, n * cn as i32, l]
    }

    /// The tile of this CTA in the cluster at `(x, y, z)`.
    #[track_caller]
    fn tile_of(&self, x: Int32, y: Int32, z: Int32) -> [Int32; 3] {
        let [m, n, l] = self.swizzle_and_rasterize(x, y, z);
        [m + self.cta_id_in_cluster[0], n + self.cta_id_in_cluster[1], l]
    }

    /// The tile of this CTA's own cluster, always valid (CuTe DSL's
    /// `initial_work_tile_info`).
    #[track_caller]
    pub fn initial_work_tile_info(&self) -> WorkTileInfo {
        let b = self.block_idx;
        let tile = self.tile_of(b.x - self.cta_id_in_cluster[0], b.y - self.cta_id_in_cluster[1], b.z);
        WorkTileInfo::new(tile, crate::value::Boolean::from(true))
    }

    /// The work the response at `result_addr` names (CuTe DSL's
    /// `work_tile_info_from_clc_response`): [`clc_response`](crate::arch::clc_response),
    /// then the proxy fence with `insert_fence`, then the visiting order.
    /// Valid when a cluster was cancelled.
    #[track_caller]
    pub fn work_tile_info_from_clc_response<T: Numeric>(&self, result_addr: &Ptr<T>) -> WorkTileInfo {
        let (x, y, z, valid) = crate::arch::clc_response(result_addr);
        if self.insert_fence {
            fence_view_async_shared();
        }
        WorkTileInfo::new(self.tile_of(x, y, z), valid.ne(0))
    }

    /// The work of the latest response (CuTe DSL's `get_current_work`):
    /// read it after waiting on its pipeline stage.
    #[track_caller]
    pub fn get_current_work(&self) -> WorkTileInfo {
        self.work_tile_info_from_clc_response(&self.clc_response_ptr.clone())
    }

    /// Ask for the next cluster's work (CuTe DSL's `advance_to_next_work`):
    /// one elected thread of the calling warp issues a multicast
    /// [`issue_clc_query`] reporting to `mbarrier_addr` (the acquired
    /// stage's full barrier), and one more tile counts as executed. Called
    /// by the scheduler warp of CTA 0 after
    /// [`producer_acquire`](crate::pipeline::Pipeline::producer_acquire).
    #[track_caller]
    pub fn advance_to_next_work(&mut self, mbarrier_addr: &Ptr<u64>) {
        let response = self.clc_response_ptr.clone();
        if_(elect_one(), || issue_clc_query(mbarrier_addr, &response, true));
        self.num_tiles_executed = self.num_tiles_executed + 1;
    }
}

impl ClcDynamicPersistentTileScheduler {
    /// Count one more tile as executed, for a caller that issues the query
    /// itself (Task Scheduling's work queue, whose responses land in one
    /// buffer per pipeline stage).
    pub(crate) fn count_tile_executed(&mut self) {
        self.num_tiles_executed = self.num_tiles_executed + 1;
    }
}

impl Carry for ClcDynamicPersistentTileScheduler {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.num_tiles_executed.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        ClcDynamicPersistentTileScheduler {
            num_tiles_executed: self.num_tiles_executed.rebuild(values),
            ..self.clone()
        }
    }
}

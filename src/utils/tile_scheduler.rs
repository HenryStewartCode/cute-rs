// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static persistent tile schedulers (CuTe DSL 4.8.1's
//! `StaticPersistentTileScheduler`, from the examples' helper module that
//! the deprecated `cutlass.utils` copy moved to): a persistent kernel
//! launches about one cluster per SM, and each cluster walks the problem's
//! tiles with a stride of the number of clusters launched.
//!
//! * [`PersistentTileSchedulerParams`] describes the problem in tiles and
//!   the cluster shape, sizes the grid on the host
//!   ([`get_grid_shape`](PersistentTileSchedulerParams::get_grid_shape))
//!   and reaches the kernel as a few scalar parameters
//!   ([`param`](PersistentTileSchedulerParams::param) and
//!   [`kernel_args`](PersistentTileSchedulerParams::kernel_args)), or is
//!   built in the kernel from a static or runtime problem shape.
//! * [`StaticPersistentTileScheduler`] hands each CTA its [`WorkTileInfo`]s;
//!   [`StaticPersistentRuntimeTileScheduler`] decodes the tile index
//!   without a validity bound, for kernels that decide themselves when to
//!   stop.
//! * [`swizzle_and_rasterize`] is the order in which clusters visit the
//!   tiles, shared with the cluster launch control scheduler.

use crate::arch::{Dim3, make_warp_uniform};
use crate::control::Carry;
use crate::fast_divmod::{FastDivmod, HostFastDivmod, param_fast_divmod};
use crate::ir::ValueId;
use crate::layout::{IndexExpr, Layout, Shape, StrideLeaf, Tree, ceil_div, make_layout, round_up, size};
use crate::sim::KernelArg;
use crate::value::{Boolean, Int32, IntoValue};

/// `INT32_MAX`, the extent of the unbounded mode of
/// [`StaticPersistentRuntimeTileScheduler`]'s decode.
const UNBOUNDED: usize = i32::MAX as usize;

fn leaf(e: IndexExpr) -> Shape {
    Tree::Leaf(e)
}

fn stride(e: IndexExpr) -> Tree<StrideLeaf> {
    Tree::Leaf(StrideLeaf::Int(e))
}

/// The layout over a problem's clusters that orders their visits (CuTe DSL's
/// `_swizzle_and_rasterize`): `ncluster_mnl` is the number of clusters along
/// M, N and L. A layout maps a cluster's `(m, n, l)` coordinate to its
/// position in the visiting order, so `get_flat_coord` of a linear work
/// index gives the cluster to work on.
///
/// With `swizzle_size` 1 the order is column-major, M fastest
/// (`raster_along_m` does not matter: the FastDivmod decode of
/// [`StaticPersistentTileScheduler`] handles N-major order itself). With a
/// larger `swizzle_size`, the clusters go in bands of `swizzle_size` along
/// one mode: for `raster_along_m`, N is rounded up to a multiple of `swz`
/// and the layout is `(M, (swz, N/swz), L):(swz, (1, swz*M), M*N)`, so
/// `swz` neighbouring N columns are visited together while M advances;
/// otherwise M is rounded up and the layout is
/// `((swz, M/swz), N, L):((1, swz*N), swz, M*N)`. The rounding adds
/// padding clusters past the problem, which the scheduler reports as valid
/// work (as CuTe DSL does): a kernel without out-of-bounds handling must
/// skip them.
///
/// Extents may be runtime values. Panics unless `ncluster_mnl` has rank 3,
/// and for a `swizzle_size` of 0.
#[track_caller]
pub fn swizzle_and_rasterize(ncluster_mnl: &Shape, swizzle_size: usize, raster_along_m: bool) -> Layout {
    assert!(swizzle_size >= 1, "cute: swizzle_and_rasterize: swizzle_size must be >= 1, not 0");
    assert_eq!(
        ncluster_mnl.rank(),
        3,
        "cute: swizzle_and_rasterize: the cluster counts {ncluster_mnl} are not (M, N, L)"
    );
    if swizzle_size == 1 {
        return Layout::col_major(ncluster_mnl.clone());
    }
    let swz = IndexExpr::Static(swizzle_size);
    let multiple: Shape =
        if raster_along_m { (1usize, swizzle_size, 1usize).into() } else { (swizzle_size, 1usize, 1usize).into() };
    let padded = round_up(ncluster_mnl.clone(), multiple);
    let ext = |i: usize| size(&padded, &[i]);
    let (m, n, l) = (ext(0), ext(1), ext(2));
    let mn = m.clone() * n.clone();
    if raster_along_m {
        let shape =
            Tree::Branch(vec![leaf(m.clone()), Tree::Branch(vec![leaf(swz.clone()), leaf(n / swizzle_size)]), leaf(l)]);
        let strides = Tree::Branch(vec![
            stride(swz.clone()),
            Tree::Branch(vec![stride(IndexExpr::Static(1)), stride(swz * m)]),
            stride(mn),
        ]);
        make_layout(shape, strides)
    } else {
        let shape =
            Tree::Branch(vec![Tree::Branch(vec![leaf(swz.clone()), leaf(m / swizzle_size)]), leaf(n.clone()), leaf(l)]);
        let strides = Tree::Branch(vec![
            Tree::Branch(vec![stride(IndexExpr::Static(1)), stride(swz.clone() * n)]),
            stride(swz),
            stride(mn),
        ]);
        make_layout(shape, strides)
    }
}

/// The optional arguments of [`PersistentTileSchedulerParams::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentTileSchedulerOptions {
    /// Clusters per band of the visiting order ([`swizzle_and_rasterize`]);
    /// 1, the default, is no swizzle.
    pub swizzle_size: usize,
    /// With a swizzle, whether the bands advance along M (the default) or
    /// along N.
    pub raster_along_m: bool,
}

impl Default for PersistentTileSchedulerOptions {
    fn default() -> Self {
        PersistentTileSchedulerOptions { swizzle_size: 1, raster_along_m: true }
    }
}

/// Where the FastDivmod divisors of an unswizzled schedule come from.
#[derive(Clone, Copy, Debug)]
enum Divisors {
    /// Computed when the scheduler is made: as constants for static
    /// cluster counts, in the kernel otherwise.
    Derived,
    /// Kernel parameters ([`PersistentTileSchedulerParams::param`]):
    /// major, then minor.
    Params(FastDivmod, FastDivmod),
}

/// What a persistent tile scheduler knows of the problem (CuTe DSL's
/// `PersistentTileSchedulerParams`): the number of tiles along M, N and L,
/// the cluster shape (one CTA deep along K) and the visiting order, from
/// which it derives the layout of the problem's clusters
/// ([`problem_layout_ncluster_mnl`](Self::problem_layout_ncluster_mnl):
/// `ceil_div` of the tiles by the cluster shape, then
/// [`swizzle_and_rasterize`]).
///
/// On the host, with a static tile count, it sizes the grid
/// ([`get_grid_shape`](Self::get_grid_shape)) and gives the kernel
/// arguments ([`kernel_args`](Self::kernel_args)) for the kernel's
/// [`param`](Self::param); a kernel can also make it from a static or
/// runtime tile count with [`new`](Self::new). An unswizzled schedule
/// decodes work indices with two [`FastDivmod`]s: constants for a static
/// tile count, host-prepared divisors with [`param`](Self::param), and
/// divisors prepared in the kernel otherwise.
#[derive(Clone, Debug)]
pub struct PersistentTileSchedulerParams {
    problem_shape_ntile_mnl: Shape,
    cluster_shape_mnk: [usize; 3],
    options: PersistentTileSchedulerOptions,
    problem_layout_ncluster_mnl: Layout,
    divisors: Divisors,
    fallback_cluster_shape_mnk: Option<[usize; 3]>,
}

impl PersistentTileSchedulerParams {
    /// The schedule of `problem_shape_ntile_mnl` tiles (rank 3: M, N, L;
    /// static or runtime extents) in clusters of `cluster_shape_mnk` CTAs.
    ///
    /// Panics unless the cluster is one CTA deep along K
    /// (`cluster_shape_mnk[2] == 1`) with at least one CTA along M and N,
    /// for a `swizzle_size` of 0, and unless the problem shape has rank 3.
    #[track_caller]
    pub fn new(
        problem_shape_ntile_mnl: impl Into<Shape>,
        cluster_shape_mnk: [usize; 3],
        options: PersistentTileSchedulerOptions,
    ) -> PersistentTileSchedulerParams {
        let problem_shape_ntile_mnl = problem_shape_ntile_mnl.into();
        let [cm, cn, ck] = cluster_shape_mnk;
        assert!(
            ck == 1,
            "cute: PersistentTileSchedulerParams: unsupported cluster_shape_k {ck}: the cluster must be one CTA deep \
             along K"
        );
        assert!(
            cm >= 1 && cn >= 1,
            "cute: PersistentTileSchedulerParams: the cluster shape {cluster_shape_mnk:?} has no CTA along M or N"
        );
        assert!(
            options.swizzle_size >= 1,
            "cute: PersistentTileSchedulerParams: swizzle_size must be >= 1, not {}",
            options.swizzle_size
        );
        assert_eq!(
            problem_shape_ntile_mnl.rank(),
            3,
            "cute: PersistentTileSchedulerParams: the problem shape in tiles {problem_shape_ntile_mnl} is not (M, N, L)"
        );
        let ncluster = ceil_div(problem_shape_ntile_mnl.clone(), Shape::from((cm, cn)));
        let problem_layout_ncluster_mnl =
            swizzle_and_rasterize(&ncluster, options.swizzle_size, options.raster_along_m);
        PersistentTileSchedulerParams {
            problem_shape_ntile_mnl,
            cluster_shape_mnk,
            options,
            problem_layout_ncluster_mnl,
            divisors: Divisors::Derived,
            fallback_cluster_shape_mnk: None,
        }
    }

    /// The same schedule for a launch with preferred clusters of this
    /// schedule's shape and fallback clusters of `fallback_cluster_shape_mnk`
    /// (CuTe DSL's `fallback_cluster_shape_mnk=`; roadmap part
    /// m10-port-clc-clusters). A scheduler made from it may then be created in
    /// either arm of [`mixed_cluster_arms`](super::mixed_cluster_arms).
    ///
    /// The fallback arm keeps the preferred shape's work decomposition: each
    /// CTA's tiles depend on its block index alone (the work index is the
    /// preferred cluster's slot, `block_idx.z`, and the rank is taken in the
    /// preferred shape), so preferred and fallback clusters of one launch,
    /// whose grid is sized for preferred clusters, cover the tiles exactly once
    /// between them. (CuTe DSL's fallback clone strides by fallback clusters
    /// from the same work index, which only fits a grid of fallback clusters;
    /// its tutorial passes the preferred params to both arms for this reason.)
    /// Only the cluster-dependent parts of the kernel (cluster layout,
    /// multicast, pipelines) differ between the arms.
    ///
    /// Panics when the shapes fail
    /// [`check_mixed_cluster_shapes`](super::check_mixed_cluster_shapes).
    #[track_caller]
    pub fn with_fallback_cluster_shape_mnk(mut self, fallback_cluster_shape_mnk: [usize; 3]) -> Self {
        if let Err(e) = super::check_mixed_cluster_shapes(self.cluster_shape_mnk, fallback_cluster_shape_mnk, None) {
            panic!("cute: PersistentTileSchedulerParams::with_fallback_cluster_shape_mnk: {e}");
        }
        self.fallback_cluster_shape_mnk =
            (fallback_cluster_shape_mnk != self.cluster_shape_mnk).then_some(fallback_cluster_shape_mnk);
        self
    }

    /// The fallback cluster shape, if the schedule has a distinct one.
    pub fn fallback_cluster_shape_mnk(&self) -> Option<[usize; 3]> {
        self.fallback_cluster_shape_mnk
    }

    /// The schedule as kernel parameters: the tiles along M, N and L as
    /// `{name}_ntile_m`, `{name}_ntile_n` and `{name}_ntile_l` (`i32`), and
    /// for an unswizzled schedule the host-prepared divisors of the major
    /// and minor cluster counts ([`param_fast_divmod`] `{name}_major` and
    /// `{name}_minor`). The host passes [`kernel_args`](Self::kernel_args)
    /// of the same schedule. Panics as [`new`](Self::new) does.
    #[track_caller]
    pub fn param(
        name: &str,
        cluster_shape_mnk: [usize; 3],
        options: PersistentTileSchedulerOptions,
    ) -> PersistentTileSchedulerParams {
        let extent = |mode: &str| IndexExpr::from(crate::param_scalar::<i32>(&format!("{name}_ntile_{mode}")));
        let shape = Tree::Branch(vec![leaf(extent("m")), leaf(extent("n")), leaf(extent("l"))]);
        let mut params = PersistentTileSchedulerParams::new(shape, cluster_shape_mnk, options);
        if options.swizzle_size == 1 {
            let major = param_fast_divmod(&format!("{name}_major"));
            let minor = param_fast_divmod(&format!("{name}_minor"));
            params.divisors = Divisors::Params(major, minor);
        }
        params
    }

    /// The arguments of the kernel's [`param`](Self::param) for this
    /// schedule, in declaration order: the three tile counts, then (when
    /// unswizzled) the major and minor divisors. Panics for a runtime tile
    /// count.
    #[track_caller]
    pub fn kernel_args(&self) -> Vec<KernelArg> {
        let [m, n, l] = self.static_extents(&self.problem_shape_ntile_mnl, "kernel_args");
        let mut args: Vec<KernelArg> = [m, n, l].iter().map(|&e| KernelArg::scalar(e as i32)).collect();
        if let Some((major, minor)) = self.host_divisors() {
            args.extend(major.kernel_args());
            args.extend(minor.kernel_args());
        }
        args
    }

    #[track_caller]
    fn static_extents(&self, shape: &Shape, what: &str) -> [usize; 3] {
        [0, 1, 2].map(|i| {
            size(shape, &[i]).as_static().unwrap_or_else(|| {
                panic!("cute: PersistentTileSchedulerParams::{what} needs a static problem shape, not {shape}")
            })
        })
    }

    /// The cluster counts along the decode's major and minor modes (M then
    /// N when rastering along M), as host divisors, for an unswizzled
    /// schedule of static extent.
    fn host_divisors(&self) -> Option<(HostFastDivmod, HostFastDivmod)> {
        if self.options.swizzle_size != 1 {
            return None;
        }
        let shape = self.problem_layout_ncluster_mnl.shape();
        let m = size(shape, &[0]).as_static()?;
        let n = size(shape, &[1]).as_static()?;
        let (major, minor) = if self.options.raster_along_m { (m, n) } else { (n, m) };
        // A problem with no tiles along a mode has no valid work index, so
        // the decode's result does not matter; dividing by 1 keeps it
        // defined.
        let divisor = |d: usize| FastDivmod::host(i32::try_from(d.max(1)).expect("cute: a cluster count fits in i32"));
        Some((divisor(major), divisor(minor)))
    }

    /// The problem's tiles along M, N and L.
    pub fn problem_shape_ntile_mnl(&self) -> &Shape {
        &self.problem_shape_ntile_mnl
    }

    /// The cluster shape (M, N, K).
    pub fn cluster_shape_mnk(&self) -> [usize; 3] {
        self.cluster_shape_mnk
    }

    /// The cluster shape along M and N.
    pub fn cluster_shape_mn(&self) -> [usize; 2] {
        [self.cluster_shape_mnk[0], self.cluster_shape_mnk[1]]
    }

    /// The swizzle size.
    pub fn swizzle_size(&self) -> usize {
        self.options.swizzle_size
    }

    /// Whether the swizzled bands advance along M.
    pub fn raster_along_m(&self) -> bool {
        self.options.raster_along_m
    }

    /// The layout of the problem's clusters, in visiting order (padded to
    /// a multiple of the swizzle size).
    pub fn problem_layout_ncluster_mnl(&self) -> &Layout {
        &self.problem_layout_ncluster_mnl
    }

    /// The grid of a persistent launch that runs at most
    /// `max_active_clusters` clusters at once (CuTe DSL's `get_grid_shape`):
    /// `(cluster_m, cluster_n, clusters)`, where `clusters` is the number
    /// of the problem's clusters (padding included) or of the clusters
    /// that fit, whichever is fewer. Host code: panics for a runtime tile
    /// count.
    #[track_caller]
    pub fn get_grid_shape(&self, max_active_clusters: u32) -> [u32; 3] {
        let [m, n, l] = self.static_extents(self.problem_layout_ncluster_mnl.shape(), "get_grid_shape");
        let [cm, cn] = self.cluster_shape_mn();
        let per_cluster = (cm * cn) as u64;
        let problem_ctas = (m * cm) as u64 * (n * cn) as u64 * l as u64;
        let wave_ctas = max_active_clusters as u64 * per_cluster;
        let clusters = problem_ctas.min(wave_ctas) / per_cluster;
        [cm as u32, cn as u32, u32::try_from(clusters).expect("cute: the cluster count fits in u32")]
    }

    /// The decode's divisors: the major and minor cluster counts.
    #[track_caller]
    pub(crate) fn divisors(&self) -> (FastDivmod, FastDivmod) {
        match self.divisors {
            Divisors::Params(major, minor) => (major, minor),
            Divisors::Derived => match self.host_divisors() {
                Some((major, minor)) => {
                    let constant = |h: HostFastDivmod| FastDivmod {
                        divisor: Int32::from(h.divisor),
                        multiplier: h.multiplier.into_value(),
                        shift: h.shift.into_value(),
                    };
                    (constant(major), constant(minor))
                }
                None => {
                    let shape = self.problem_layout_ncluster_mnl.shape();
                    let (m, n) = (size(shape, &[0]), size(shape, &[1]));
                    let (major, minor) = if self.options.raster_along_m { (m, n) } else { (n, m) };
                    // As for the host divisors: a count of 0 divides by 1
                    // (no work index is valid then), not by 0.
                    let divisor = |d: IndexExpr| FastDivmod::new(Int32::from(d).max(1));
                    (divisor(major), divisor(minor))
                }
            },
        }
    }
}

/// The cluster coordinate of work index `idx` of `params`' visiting order:
/// the FastDivmod decode with `divisors` (major, minor; an unswizzled
/// schedule), `get_flat_coord` of the problem layout otherwise. Shared with
/// the cluster launch control scheduler (m10-clc-scheduling).
#[track_caller]
pub(crate) fn decode_cluster(
    params: &PersistentTileSchedulerParams,
    divisors: Option<(FastDivmod, FastDivmod)>,
    idx: Int32,
) -> [Int32; 3] {
    match divisors {
        Some((major, minor)) => {
            let (minor_and_batch, major_idx) = major.divmod(idx);
            let (batch, minor_idx) = minor.divmod(minor_and_batch);
            if params.raster_along_m() { [major_idx, minor_idx, batch] } else { [minor_idx, major_idx, batch] }
        }
        None => {
            let coord = params.problem_layout_ncluster_mnl().get_flat_coord(idx);
            let leaves = coord.leaves();
            [0, 1, 2].map(|i| Int32::from(leaves[i].clone()))
        }
    }
}

/// A tile to work on (CuTe DSL's `WorkTileInfo`): its coordinate in tiles
/// along M, N and L, and whether it is work at all. Carried through loops
/// ([`Carry`]).
#[derive(Clone, Copy, Debug)]
pub struct WorkTileInfo {
    tile_idx: [Int32; 3],
    is_valid_tile: Boolean,
}

impl WorkTileInfo {
    /// A work tile at `tile_idx`, valid or not.
    pub fn new(tile_idx: [Int32; 3], is_valid_tile: Boolean) -> WorkTileInfo {
        WorkTileInfo { tile_idx, is_valid_tile }
    }

    /// The tile's coordinate along M, N and L.
    pub fn tile_idx(&self) -> [Int32; 3] {
        self.tile_idx
    }

    /// Whether there is work: false once the scheduler has run past the
    /// problem.
    pub fn is_valid_tile(&self) -> Boolean {
        self.is_valid_tile
    }
}

impl Carry for WorkTileInfo {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.tile_idx.flatten(out);
        self.is_valid_tile.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        WorkTileInfo { tile_idx: self.tile_idx.rebuild(values), is_valid_tile: self.is_valid_tile.rebuild(values) }
    }
}

/// A static persistent tile scheduler (CuTe DSL's
/// `StaticPersistentTileScheduler`): launched with the grid of
/// [`PersistentTileSchedulerParams::get_grid_shape`], each cluster starts
/// at the work index of its cluster index in the grid (`block_idx.z`) and
/// moves on by the number of clusters launched, so that together they
/// visit every cluster of the problem once. A CTA's tile is its cluster's
/// coordinate times the cluster shape plus its rank in the cluster.
///
/// Carried through loops ([`Carry`]): only the work index and the count of
/// tiles executed change; the rest is computed once, before the loop.
///
/// ```
/// use cute::prelude::*;
/// use cute::utils::{PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentTileScheduler};
///
/// // 5 x 3 tiles, two persistent CTAs: each records the tiles it visits.
/// let params = PersistentTileSchedulerParams::new((5, 3, 1), [1, 1, 1], Default::default());
/// let grid = params.get_grid_shape(2);
/// assert_eq!(grid, [1, 1, 2]);
/// let kernel = Kernel::builder("visit").block_dim([32, 1, 1]).build(|| {
///     let out = param_tensor::<i32>("out", Layout::col_major(15));
///     let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
///     let work = sched.initial_work_tile_info();
///     while_carry(
///         (work, sched),
///         |(work, _)| work.is_valid_tile(),
///         |(work, mut sched)| {
///             let [m, n, _] = work.tile_idx();
///             if_(thread_idx().x.eq(0), || out.store(m + n * 5, block_idx().z));
///             sched.advance_to_next_work();
///             (sched.get_current_work(), sched)
///         },
///     );
/// });
/// let mut args = vec![KernelArg::buffer(&[-1i32; 15])];
/// let cfg = LaunchConfig::new(grid, [32, 1, 1]);
/// kernel.simulate(cfg, &mut args).unwrap();
/// assert_eq!(args[0].to_vec::<i32>(), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
/// ```
#[derive(Clone, Debug)]
pub struct StaticPersistentTileScheduler {
    params: PersistentTileSchedulerParams,
    major: Option<FastDivmod>,
    minor: Option<FastDivmod>,
    num_persistent_clusters: Int32,
    cta_id_in_cluster: [Int32; 3],
    current_work_linear_idx: Int32,
    num_tiles_executed: Int32,
}

impl StaticPersistentTileScheduler {
    /// The scheduler of the CTA at `block_idx` in a grid of `grid_dim`
    /// (CuTe DSL's `create`): the grid holds `size(grid_dim) /
    /// size(cluster_mn)` persistent clusters, the first work index is
    /// `block_idx.z`, and the CTA's rank in its cluster is `(block_idx.x %
    /// cluster_m, block_idx.y % cluster_n, 0)`.
    ///
    /// In an arm of [`mixed_cluster_arms`](super::mixed_cluster_arms), the
    /// arm's cluster shape must be the schedule's or its
    /// [fallback](PersistentTileSchedulerParams::with_fallback_cluster_shape_mnk)
    /// shape: panics otherwise.
    #[track_caller]
    pub fn create(params: &PersistentTileSchedulerParams, block_idx: Dim3, grid_dim: Dim3) -> Self {
        let in_fallback_arm = super::current_cluster_arm()
            .is_some_and(|a| params.fallback_cluster_shape_mnk() == Some(a.cluster_shape_mnk));
        if !in_fallback_arm {
            super::check_cluster_shape_in_arm("StaticPersistentTileScheduler's schedule", params.cluster_shape_mnk());
        }
        let [cm, cn] = params.cluster_shape_mn();
        let grid = IndexExpr::from(grid_dim.x) * grid_dim.y * grid_dim.z;
        let num_persistent_clusters = Int32::from(grid / (cm * cn));
        let cta_id_in_cluster = [
            Int32::from(IndexExpr::from(block_idx.x) % cm),
            Int32::from(IndexExpr::from(block_idx.y) % cn),
            Int32::from(0),
        ];
        let (major, minor) = if params.swizzle_size() == 1 {
            let (major, minor) = params.divisors();
            (Some(major), Some(minor))
        } else {
            (None, None)
        };
        StaticPersistentTileScheduler {
            params: params.clone(),
            major,
            minor,
            num_persistent_clusters,
            cta_id_in_cluster,
            current_work_linear_idx: block_idx.z,
            num_tiles_executed: Int32::from(0),
        }
    }

    /// The grid of a persistent launch (CuTe DSL's static `get_grid_shape`):
    /// [`PersistentTileSchedulerParams::get_grid_shape`].
    #[track_caller]
    pub fn get_grid_shape(params: &PersistentTileSchedulerParams, max_active_clusters: u32) -> [u32; 3] {
        params.get_grid_shape(max_active_clusters)
    }

    /// The schedule.
    pub fn params(&self) -> &PersistentTileSchedulerParams {
        &self.params
    }

    /// The number of clusters launched.
    pub fn num_persistent_clusters(&self) -> Int32 {
        self.num_persistent_clusters
    }

    /// This CTA's rank in its cluster along M, N and K.
    pub fn cta_id_in_cluster(&self) -> [Int32; 3] {
        self.cta_id_in_cluster
    }

    /// The current work index, a cluster's position in the visiting order.
    pub fn current_work_linear_idx(&self) -> Int32 {
        self.current_work_linear_idx
    }

    /// The cluster coordinate of work index `idx`: the FastDivmod decode
    /// for an unswizzled schedule, `get_flat_coord` of the problem layout
    /// otherwise.
    #[track_caller]
    fn cluster_coord(&self, idx: Int32) -> [Int32; 3] {
        decode_cluster(&self.params, self.major.zip(self.minor), idx)
    }

    /// The work tile of work index `idx`.
    #[track_caller]
    fn work_for_linear_idx(&self, idx: Int32) -> WorkTileInfo {
        let total = Int32::from(size(self.params.problem_layout_ncluster_mnl(), &[]));
        let is_valid = idx.lt(total);
        let cluster = self.params.cluster_shape_mnk();
        let coord = self.cluster_coord(idx);
        let tile_idx = [0, 1, 2].map(|i| {
            let tile = coord[i] * cluster[i] as i32 + self.cta_id_in_cluster[i];
            make_warp_uniform(tile)
        });
        WorkTileInfo { tile_idx, is_valid_tile: make_warp_uniform(is_valid.to::<i32>()).ne(0) }
    }

    /// The tile at the current work index (CuTe DSL's `get_current_work`).
    #[track_caller]
    pub fn get_current_work(&self) -> WorkTileInfo {
        self.work_for_linear_idx(self.current_work_linear_idx)
    }

    /// The first tile (CuTe DSL's `initial_work_tile_info`), the same as
    /// [`get_current_work`](Self::get_current_work) before any advance.
    #[track_caller]
    pub fn initial_work_tile_info(&self) -> WorkTileInfo {
        self.get_current_work()
    }

    /// Move to the next tile: the work index goes up by the number of
    /// clusters launched, and one more tile counts as executed.
    #[track_caller]
    pub fn advance_to_next_work(&mut self) {
        self.advance_to_next_work_by(1);
    }

    /// Move `advance_count` tiles on (CuTe DSL's
    /// `advance_to_next_work(advance_count=...)`): the work index goes up
    /// by `advance_count` times the number of clusters launched, and the
    /// count of tiles executed by one.
    #[track_caller]
    pub fn advance_to_next_work_by(&mut self, advance_count: impl IntoValue<i32>) {
        let step = advance_count.into_value() * self.num_persistent_clusters;
        self.current_work_linear_idx = self.current_work_linear_idx + step;
        self.num_tiles_executed = self.num_tiles_executed + 1;
    }

    /// The number of advances so far.
    pub fn num_tiles_executed(&self) -> Int32 {
        self.num_tiles_executed
    }
}

impl Carry for StaticPersistentTileScheduler {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.current_work_linear_idx.flatten(out);
        self.num_tiles_executed.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        StaticPersistentTileScheduler {
            current_work_linear_idx: self.current_work_linear_idx.rebuild(values),
            num_tiles_executed: self.num_tiles_executed.rebuild(values),
            ..self.clone()
        }
    }
}

/// A persistent scheduler for kernels that stop on their own (CuTe DSL's
/// `StaticPersistentRuntimeTileScheduler`): the grid usually covers every
/// SM (`cuda::HardwareInfo`'s SM count), every work
/// index is valid, and the kernel compares the tile against a runtime
/// bound of its own. The work index is decoded over an unbounded mode:
/// `get_hier_coord` on `(INT32_MAX, n):(n, 1)` for `inner_mode` 1 (N
/// fastest) or `(m, INT32_MAX):(1, m)` for `inner_mode` 0 (M fastest),
/// where `m` and `n` are the problem's clusters along M and N (the sizes of
/// those modes of the problem layout). The tile is `(m, n, 0)`, without the
/// CTA's rank in its cluster, as in CuTe DSL.
#[derive(Clone, Debug)]
pub struct StaticPersistentRuntimeTileScheduler {
    inner: StaticPersistentTileScheduler,
    inner_mode: usize,
}

impl StaticPersistentRuntimeTileScheduler {
    /// The scheduler of the CTA at `block_idx` in a grid of `grid_dim`,
    /// decoding along `inner_mode` first (CuTe DSL's `create`, whose
    /// default `inner_mode` is 1). Panics for an `inner_mode` other than 0
    /// and 1.
    #[track_caller]
    pub fn create(params: &PersistentTileSchedulerParams, block_idx: Dim3, grid_dim: Dim3, inner_mode: usize) -> Self {
        assert!(
            inner_mode <= 1,
            "cute: StaticPersistentRuntimeTileScheduler: inner_mode is 0 (M first) or 1 (N first), not {inner_mode}"
        );
        StaticPersistentRuntimeTileScheduler {
            inner: StaticPersistentTileScheduler::create(params, block_idx, grid_dim),
            inner_mode,
        }
    }

    /// The mode decoded first: 0 for M, 1 for N.
    pub fn inner_mode(&self) -> usize {
        self.inner_mode
    }

    /// The underlying static scheduler (its work index and counters).
    pub fn scheduler(&self) -> &StaticPersistentTileScheduler {
        &self.inner
    }

    /// The tile at the current work index, always valid.
    #[track_caller]
    pub fn get_current_work(&self) -> WorkTileInfo {
        let shape = self.inner.params.problem_layout_ncluster_mnl().shape();
        let (m, n) = (size(shape, &[0]), size(shape, &[1]));
        let unbounded = IndexExpr::Static(UNBOUNDED);
        let layout = if self.inner_mode == 1 {
            make_layout(
                Tree::Branch(vec![leaf(unbounded), leaf(n.clone())]),
                Tree::Branch(vec![stride(n), stride(IndexExpr::Static(1))]),
            )
        } else {
            make_layout(
                Tree::Branch(vec![leaf(m.clone()), leaf(unbounded)]),
                Tree::Branch(vec![stride(IndexExpr::Static(1)), stride(m)]),
            )
        };
        let coord = layout.get_hier_coord(self.inner.current_work_linear_idx);
        let leaves = coord.leaves();
        let tile_idx = [Int32::from(leaves[0].clone()), Int32::from(leaves[1].clone()), Int32::from(0)];
        WorkTileInfo { tile_idx, is_valid_tile: Boolean::from(true) }
    }

    /// The first tile.
    #[track_caller]
    pub fn initial_work_tile_info(&self) -> WorkTileInfo {
        self.get_current_work()
    }

    /// Move to the next tile.
    #[track_caller]
    pub fn advance_to_next_work(&mut self) {
        self.inner.advance_to_next_work();
    }

    /// Move `advance_count` tiles on.
    #[track_caller]
    pub fn advance_to_next_work_by(&mut self, advance_count: impl IntoValue<i32>) {
        self.inner.advance_to_next_work_by(advance_count);
    }

    /// The number of advances so far.
    pub fn num_tiles_executed(&self) -> Int32 {
        self.inner.num_tiles_executed()
    }

    /// The number of clusters launched.
    pub fn num_persistent_clusters(&self) -> Int32 {
        self.inner.num_persistent_clusters()
    }
}

impl Carry for StaticPersistentRuntimeTileScheduler {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.inner.flatten(out);
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        StaticPersistentRuntimeTileScheduler { inner: self.inner.rebuild(values), inner_mode: self.inner_mode }
    }
}

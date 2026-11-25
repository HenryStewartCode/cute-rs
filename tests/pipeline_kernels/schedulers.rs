//! Kernels that record the tiles the static persistent tile schedulers
//! hand out, and a plain-Rust model of the schedules to hold them to.

use cute::prelude::*;
use cute::utils::{
    PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentRuntimeTileScheduler,
    StaticPersistentTileScheduler,
};

/// How an enumeration kernel gets its schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamsFrom {
    /// Built in the kernel from a static tile count (constant divisors).
    Static,
    /// Kernel parameters with host-prepared divisors
    /// (`PersistentTileSchedulerParams::param` and `kernel_args`).
    Host,
    /// Built in the kernel from runtime tile counts read from scalar
    /// parameters (divisors prepared in the kernel).
    Runtime,
}

/// One schedule to enumerate.
#[derive(Clone, Copy, Debug)]
pub struct TileSchedule {
    pub tiles: [usize; 3],
    pub cluster: [usize; 3],
    pub swizzle: usize,
    pub raster_along_m: bool,
    pub max_active_clusters: u32,
}

impl TileSchedule {
    pub fn options(&self) -> PersistentTileSchedulerOptions {
        PersistentTileSchedulerOptions { swizzle_size: self.swizzle, raster_along_m: self.raster_along_m }
    }

    /// The host-side schedule.
    pub fn params(&self) -> PersistentTileSchedulerParams {
        PersistentTileSchedulerParams::new((self.tiles[0], self.tiles[1], self.tiles[2]), self.cluster, self.options())
    }

    /// The persistent grid.
    pub fn grid(&self) -> [u32; 3] {
        self.params().get_grid_shape(self.max_active_clusters)
    }

    /// The problem's clusters along M, N and L, before padding.
    pub fn clusters(&self) -> [usize; 3] {
        [self.tiles[0].div_ceil(self.cluster[0]), self.tiles[1].div_ceil(self.cluster[1]), self.tiles[2]]
    }

    /// The cluster counts padded to the swizzle.
    pub fn padded(&self) -> [usize; 3] {
        let [m, n, l] = self.clusters();
        match (self.swizzle, self.raster_along_m) {
            (1, _) => [m, n, l],
            (s, true) => [m, n.div_ceil(s) * s, l],
            (s, false) => [m.div_ceil(s) * s, n, l],
        }
    }

    /// The cluster coordinate of work index `idx`, decoded by hand.
    pub fn decode(&self, idx: usize) -> [usize; 3] {
        let [m, n, _] = self.padded();
        let (l, r) = (idx / (m * n), idx % (m * n));
        let s = self.swizzle;
        match (s, self.raster_along_m) {
            (1, true) => [r % m, r / m, l],
            (1, false) => [r / n, r % n, l],
            (_, true) => {
                let (outer, rest) = (r / (s * m), r % (s * m));
                [rest / s, outer * s + rest % s, l]
            }
            (_, false) => {
                let (outer, rest) = (r / (s * n), r % (s * n));
                [outer * s + rest % s, rest / s, l]
            }
        }
    }

    /// The tiles each CTA of the grid visits, in order, by the CTA's linear
    /// index `x + y * gx + z * gx * gy`.
    pub fn model(&self) -> Vec<Vec<[usize; 3]>> {
        let [gx, gy, gz] = self.grid().map(|g| g as usize);
        let total: usize = self.padded().iter().product();
        let clusters = gx * gy * gz / (self.cluster[0] * self.cluster[1]);
        let mut visits = vec![Vec::new(); gx * gy * gz];
        for z in 0..gz {
            for y in 0..gy {
                for x in 0..gx {
                    let cta = x + y * gx + z * gx * gy;
                    let mut idx = z;
                    while idx < total {
                        let c = self.decode(idx);
                        let rank = [x % self.cluster[0], y % self.cluster[1], 0];
                        visits[cta].push([0, 1, 2].map(|i| c[i] * self.cluster[i] + rank[i]));
                        idx += clusters;
                    }
                }
            }
        }
        visits
    }

    /// Visits a CTA may record.
    pub fn max_visits(&self) -> usize {
        self.model().iter().map(Vec::len).max().unwrap_or(0) + 1
    }
}

/// Each CTA walks its tiles with `StaticPersistentTileScheduler` and
/// records them: row `(cta, k)` of `visits` is the k-th tile's `(m, n, l,
/// 1)`, and `count[cta]` the number of tiles executed.
pub fn enumeration_kernel(s: &TileSchedule, from: ParamsFrom) -> Kernel {
    let rows = s.max_visits();
    Kernel::builder("enumerate_tiles").block_dim([32, 1, 1]).build(|| {
        let [gx, gy, gz] = s.grid().map(|g| g as usize);
        let ctas = gx * gy * gz;
        let visits = param_tensor::<i32>("visits", Layout::row_major((ctas * rows, 4)));
        let count = param_tensor::<i32>("count", Layout::col_major(ctas));
        let params = match from {
            ParamsFrom::Static => s.params(),
            ParamsFrom::Host => PersistentTileSchedulerParams::param("sched", s.cluster, s.options()),
            ParamsFrom::Runtime => {
                let e = |n: &str| IndexExpr::from(param_scalar::<i32>(n));
                let shape = Shape::from((e("tiles_m"), e("tiles_n"), e("tiles_l")));
                PersistentTileSchedulerParams::new(shape, s.cluster, s.options())
            }
        };
        let b = block_idx();
        let cta = b.x + b.y * gx as i32 + b.z * (gx * gy) as i32;
        let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
        let work = sched.initial_work_tile_info();
        let (_, sched) = while_carry(
            (work, sched),
            |(work, _)| work.is_valid_tile(),
            |(work, mut sched)| {
                let row = cta * rows as i32 + sched.num_tiles_executed();
                if_(thread_idx().x.eq(0), || {
                    let [m, n, l] = work.tile_idx();
                    visits.store((row, 0), m);
                    visits.store((row, 1), n);
                    visits.store((row, 2), l);
                    visits.store((row, 3), 1);
                });
                sched.advance_to_next_work();
                (sched.get_current_work(), sched)
            },
        );
        if_(thread_idx().x.eq(0), || count.store(cta, sched.num_tiles_executed()));
    })
}

/// The launch and zeroed arguments of [`enumeration_kernel`].
pub fn enumeration_launch(s: &TileSchedule, from: ParamsFrom) -> (LaunchConfig, Vec<KernelArg>) {
    let grid = s.grid();
    let ctas = grid.iter().product::<u32>() as usize;
    let mut args =
        vec![KernelArg::buffer(&vec![0i32; ctas * s.max_visits() * 4]), KernelArg::buffer(&vec![-1i32; ctas])];
    match from {
        ParamsFrom::Static => {}
        ParamsFrom::Host => args.extend(s.params().kernel_args()),
        ParamsFrom::Runtime => args.extend(s.tiles.map(|t| KernelArg::scalar(t as i32))),
    }
    let cfg = LaunchConfig::new(grid, [32, 1, 1]);
    let cfg =
        if s.cluster == [1, 1, 1] { cfg } else { cfg.with_cluster([s.cluster[0] as u32, s.cluster[1] as u32, 1]) };
    (cfg, args)
}

/// Check an enumeration run against [`TileSchedule::model`]: the same tiles in
/// the same order for each CTA, every real tile exactly once, and the
/// padding tiles (past the problem, from the swizzle's rounding) valid.
/// Returns the number of padding tiles visited.
pub fn check_enumeration(s: &TileSchedule, args: &[KernelArg]) -> usize {
    let (visits, count) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
    let rows = s.max_visits();
    let model = s.model();
    let mut seen = std::collections::HashMap::new();
    let mut padding = 0;
    for (cta, want) in model.iter().enumerate() {
        assert_eq!(count[cta] as usize, want.len(), "{s:?}: CTA {cta} executed {} tiles", count[cta]);
        for (k, tile) in want.iter().enumerate() {
            let row = (cta * rows + k) * 4;
            let got = [visits[row], visits[row + 1], visits[row + 2]].map(|v| v as usize);
            assert_eq!(got, *tile, "{s:?}: CTA {cta}, visit {k}");
            assert_eq!(visits[row + 3], 1, "{s:?}: CTA {cta}, visit {k}");
            if (0..3).all(|i| tile[i] < s.tiles[i]) {
                *seen.entry(*tile).or_insert(0) += 1;
            } else {
                padding += 1;
            }
        }
        let past = (cta * rows + want.len()) * 4;
        assert_eq!(visits[past + 3], 0, "{s:?}: CTA {cta} recorded a visit past its last tile");
    }
    let [m, n, l] = s.tiles;
    for t in (0..m).flat_map(|i| (0..n).flat_map(move |j| (0..l).map(move |k| [i, j, k]))) {
        assert_eq!(seen.get(&t).copied().unwrap_or(0), 1, "{s:?}: tile {t:?}");
    }
    padding
}

/// Rows per CTA of [`runtime_scheduler_kernel`]'s record.
pub const RUNTIME_ROWS: usize = 64;

/// Each CTA walks tiles with `StaticPersistentRuntimeTileScheduler` (from a
/// schedule of `tiles` with host-passed parameters, decoding along
/// `inner_mode`) until the tile's outer coordinate reaches the runtime
/// `limit` (M for inner mode 1, N for inner mode 0), recording `(m, n, l)`
/// per visit and the count of tiles executed.
pub fn runtime_scheduler_kernel(ctas: usize, inner_mode: usize) -> Kernel {
    Kernel::builder("runtime_tile_scheduler").block_dim([32, 1, 1]).build(|| {
        let visits = param_tensor::<i32>("visits", Layout::row_major((ctas * RUNTIME_ROWS, 3)));
        let count = param_tensor::<i32>("count", Layout::col_major(ctas));
        let params = PersistentTileSchedulerParams::param("sched", [1, 1, 1], Default::default());
        let limit = param_scalar::<i32>("limit");
        let cta = block_idx().z;
        let sched = StaticPersistentRuntimeTileScheduler::create(&params, block_idx(), grid_dim(), inner_mode);
        let work = sched.initial_work_tile_info();
        let outer = 1 - inner_mode;
        let (_, sched) = while_carry(
            (work, sched),
            move |(work, _)| work.is_valid_tile() & work.tile_idx()[outer].lt(limit),
            |(work, mut sched)| {
                let row = cta * RUNTIME_ROWS as i32 + sched.num_tiles_executed();
                if_(thread_idx().x.eq(0), || {
                    let [m, n, l] = work.tile_idx();
                    visits.store((row, 0), m);
                    visits.store((row, 1), n);
                    visits.store((row, 2), l);
                });
                sched.advance_to_next_work();
                (sched.get_current_work(), sched)
            },
        );
        if_(thread_idx().x.eq(0), || count.store(cta, sched.num_tiles_executed()));
    })
}

/// The launch and arguments of [`runtime_scheduler_kernel`]: `ctas`
/// persistent CTAs over `tiles` (M, N) of which the kernel visits the first
/// `limit` along the outer mode.
pub fn runtime_scheduler_launch(ctas: usize, tiles: [usize; 2], limit: usize) -> (LaunchConfig, Vec<KernelArg>) {
    let params = PersistentTileSchedulerParams::new((tiles[0], tiles[1], 1), [1, 1, 1], Default::default());
    let mut args =
        vec![KernelArg::buffer(&vec![-1i32; ctas * RUNTIME_ROWS * 3]), KernelArg::buffer(&vec![-1i32; ctas])];
    args.extend(params.kernel_args());
    args.push(KernelArg::scalar(limit as i32));
    (LaunchConfig::new([1, 1, ctas as u32], [32, 1, 1]), args)
}

/// Check a [`runtime_scheduler_kernel`] run: CTA `c` visits work indices
/// `c, c + ctas, ...` decoded along the inner mode, up to the limit.
pub fn check_runtime_scheduler(ctas: usize, tiles: [usize; 2], limit: usize, inner_mode: usize, args: &[KernelArg]) {
    let (visits, count) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
    let [m, n] = tiles;
    let decode = |idx: usize| if inner_mode == 1 { [idx / n, idx % n] } else { [idx % m, idx / m] };
    let outer = 1 - inner_mode;
    assert_eq!(count.len(), ctas);
    for (c, &executed) in count.iter().enumerate() {
        let want: Vec<[usize; 2]> = (0..).map(|k| decode(c + k * ctas)).take_while(|t| t[outer] < limit).collect();
        assert_eq!(executed as usize, want.len(), "CTA {c}");
        assert!(want.len() <= RUNTIME_ROWS);
        for (k, t) in want.iter().enumerate() {
            let row = (c * RUNTIME_ROWS + k) * 3;
            assert_eq!(visits[row..row + 3], [t[0] as i32, t[1] as i32, 0], "CTA {c}, visit {k}");
        }
    }
}

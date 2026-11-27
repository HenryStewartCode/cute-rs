// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/rmsnorm/rmsnorm.py
//   (`RMSNormConfig`, `RMSNormKernel`, `predicate_k`, `rmsnorm_ref`, the
//   cluster-size table and the comments) and
//   examples/python/CuTeDSL/cute/blackwell/kernel/reduce/reduce.py
//   (`row_reduce`, `block_reduce`, `cluster_reduce`, `store_shared_remote`,
//   `elem_pointer` and their comments),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! RMSNorm with cluster-based reduction for large hidden dimensions, ported
//! from the CuTe DSL example `rmsnorm.py` and its helper `reduce.py`.
//!
//! RMSNorm computes `y = x / sqrt(mean(x²) + eps) * weight` for each row of
//! an `M x N` matrix. Each CTA copies `rows_per_block` rows of its share of
//! the columns global → shared with 128-bit `cp.async`, sums the squares
//! (TensorSSA `x * x` and `reduce`), and reduces the sums hierarchically
//! (`row_reduce`):
//!
//! 1. **warp**: a butterfly `warp_reduction` over the threads of a row;
//! 2. **block** (`block_reduce`): each warp's sum through shared memory;
//! 3. **cluster** (`cluster_reduce`, SM90+): when a row is split across the
//!    `cluster_n` CTAs of a thread-block cluster, each warp `st.async`s its
//!    sum into the same slot of every CTA's reduction buffer (its own CTA's
//!    included) with `mbarrier::complete_tx`, and each CTA waits on its own
//!    mbarrier, whose phase expects every peer's bytes
//!    (`mbarrier_arrive_and_expect_tx`).
//!
//! Then `rstd = rsqrt(sum / N + eps)` and a second pass over shared memory
//! writes `x * rstd * weight`. The cluster size follows the example's
//! table (SM90+: one CTA up to 16K elements of Float16 or BFloat16, up to
//! 32K of Float32, then 2, 4, 8 and 16 CTAs); SM80 runs a row on one CTA.
//!
//! One built kernel serves every `M` (a runtime extent, `param_extent`), as
//! the example's compiled kernel takes `M` at each call; `N` is static, as
//! in the example's compile cache keyed by `(dtype, N, has_weight)`.
//!
//! Differences from the example:
//!
//! * `set_block_rank` (`mapa`) is inside cute-rs's `store_async_dsmem`, which
//!   maps both pointers and stores the bits of the Float32 with
//!   `st.async...complete_tx::bytes.b32` (the example's inline PTX stores
//!   `.f32`; the bytes are the same);
//! * `N_per_cta` is `ceil(N / cluster_n)` rather than `N // cluster_n`, and
//!   `num_vec_blocks` rounds `N_per_cta / vec_size` up too, so that the
//!   tiles of the cluster always cover the row (with the floors, a row of
//!   `N = 16392` on two CTAs would leave 8 columns unread);
//! * rows past `M` are predicated off the `cp.async` too, which zero-fills
//!   their shared memory, where the example skips the copy (`if
//!   row_in_bounds`) and then reads the never-written shared memory, which
//!   the simulator rejects; their results are not stored, as there;
//! * the weight fragment is zero-filled before its predicated copy, for the
//!   same reason;
//! * `fastmath` (the example's `rsqrt(..., fastmath=True)`) is a field of the
//!   configuration, true by default; with it false `rsqrt` is precise, and
//!   the GPU matches the simulator bit for bit (the reductions' order is
//!   fixed);
//! * `N` has to divide into whole 128-bit vectors (8 elements of 16 bits, 4
//!   of 32), as the example's vector copies need for aligned rows.
//!
//! Run with `cargo run --example rmsnorm_cluster` (on the simulator).

use cute::memory::{SmemAllocator, SmemPartition};
use cute::prelude::*;

// =============================================================================
// Predicate Utility
// =============================================================================

/// The predicate tensor for bounds checking (`predicate_k`): one flag per
/// application of the copy atom, `column < limit`, from the first
/// coordinate of each 128-bit vector of `t_xc_x`, a partitioned identity
/// tensor `((vec, num_vec_blocks), 1, 1)`. `row_in_bounds` is and-ed in
/// (see the module docs).
pub fn predicate_k(t_xc_x: &Tensor<Ident>, limit: usize, row_in_bounds: Boolean) -> Tensor<bool> {
    let rest_v_n = size(t_xc_x, &[0, 1]).as_static().expect("static");
    let rest_m = size(t_xc_x, &[1]).as_static().expect("static");
    let rest_k_n = size(t_xc_x, &[2]).as_static().expect("static");
    let t_xp_x = make_rmem_tensor::<bool>(Layout::with_strides((rest_v_n, rest_m, rest_k_n), (rest_k_n, 0, 1)));
    for rest_v in 0..rest_v_n {
        for rest_k in 0..rest_k_n {
            let crd = t_xc_x.coord(((0, rest_v), 0, rest_k));
            let col_in_bounds = elem_less(crd.mode(1).clone(), limit);
            t_xp_x.store((rest_v, 0, rest_k), col_in_bounds & row_in_bounds);
        }
    }
    t_xp_x
}

// =============================================================================
// RMSNorm Configuration
// =============================================================================

/// Configuration for the RMSNorm kernel, computed once from the element
/// type, `N` and the target's SM version (`RMSNormConfig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RMSNormConfig {
    pub dtype: DType,
    pub n: usize,
    pub has_weight: bool,
    pub sm_version: u32,
    /// `rsqrt(..., fastmath=True)` as in the example; false for the precise
    /// `rsqrt`.
    pub fastmath: bool,
    /// Elements per 128-bit vector.
    pub vec_size: usize,
    /// CTAs per cluster (1 below SM90).
    pub cluster_n: usize,
    /// Columns per CTA of the cluster.
    pub n_per_cta: usize,
    pub threads_per_row: usize,
    pub num_threads: usize,
    pub num_vec_blocks: usize,
    pub rows_per_block: usize,
    pub cols_per_tile: usize,
    pub warps_per_row: usize,
}

impl RMSNormConfig {
    /// 128-bit vectorised loads.
    pub const COPY_BITS: usize = 128;

    pub fn new(dtype: DType, n: usize, has_weight: bool, sm_version: u32) -> RMSNormConfig {
        let vec_size = Self::COPY_BITS / dtype.bits();
        assert!(
            n > 0 && n.is_multiple_of(vec_size),
            "RMSNormConfig: N = {n} is not a positive whole number of 128-bit vectors of {vec_size} elements"
        );
        // Compute cluster size (SM90+ only).
        let cluster_n = Self::compute_cluster_n(n, dtype, sm_version);
        // N per CTA for cluster case (rounded up, see the module docs).
        let n_per_cta = n.div_ceil(cluster_n);
        let threads_per_row = Self::compute_threads_per_row(n_per_cta);
        let num_threads = Self::compute_num_threads(n_per_cta);
        let num_vec_blocks = n_per_cta.div_ceil(vec_size).div_ceil(threads_per_row).max(1);
        let rows_per_block = num_threads / threads_per_row;
        let cols_per_tile = vec_size * num_vec_blocks * threads_per_row;
        let warps_per_row = (threads_per_row / 32).max(1);
        assert!(cols_per_tile * cluster_n >= n, "RMSNormConfig: the cluster's tiles cover the row");
        RMSNormConfig {
            dtype,
            n,
            has_weight,
            sm_version,
            fastmath: true,
            vec_size,
            cluster_n,
            n_per_cta,
            threads_per_row,
            num_threads,
            num_vec_blocks,
            rows_per_block,
            cols_per_tile,
            warps_per_row,
        }
    }

    /// The same configuration with the precise `rsqrt`.
    pub fn precise(self) -> RMSNormConfig {
        RMSNormConfig { fastmath: false, ..self }
    }

    /// Compute optimal cluster size based on N and architecture.
    pub fn compute_cluster_n(n: usize, dtype: DType, sm_version: u32) -> usize {
        if sm_version < 90 {
            return 1;
        }
        const K: usize = 1024;
        // FP16/BF16 switch at half the row length of FP32.
        let unit = if dtype.bits() == 16 { 16 * K } else { 32 * K };
        match n {
            _ if n <= unit => 1,
            _ if n <= 2 * unit => 2,
            _ if n <= 4 * unit => 4,
            _ if n <= 8 * unit => 8,
            _ => 16,
        }
    }

    /// Compute optimal threads per row based on N per CTA.
    pub fn compute_threads_per_row(n_per_cta: usize) -> usize {
        match n_per_cta {
            0..=64 => 8,
            65..=128 => 16,
            129..=3072 => 32,
            3073..=6144 => 64,
            6145..=16384 => 128,
            _ => 256,
        }
    }

    /// Compute total threads per block.
    pub fn compute_num_threads(n_per_cta: usize) -> usize {
        if n_per_cta <= 16384 { 128 } else { 256 }
    }

    /// The thread-value layout for coalesced vectorised memory access:
    /// `((threads_per_row, rows_per_block), (vec_size, num_vec_blocks))`,
    /// each thread's vectors `threads_per_row` vectors apart along the row.
    pub fn make_tv_layout(
        threads_per_row: usize,
        rows_per_block: usize,
        vec_size: usize,
        num_vec_blocks: usize,
    ) -> Layout {
        Layout::with_strides(
            ((threads_per_row, rows_per_block), (vec_size, num_vec_blocks)),
            ((vec_size * rows_per_block, 1), (rows_per_block, rows_per_block * vec_size * threads_per_row)),
        )
    }

    /// The tile of one CTA: `(rows_per_block, cols_per_tile)`.
    pub fn tiler_mn(&self) -> (usize, usize) {
        (self.rows_per_block, self.cols_per_tile)
    }

    /// Calculate shared memory requirement in bytes.
    pub fn smem_size_in_bytes(&self) -> usize {
        let tile_bytes = self.rows_per_block * self.cols_per_tile * self.dtype.bytes();
        let reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4;
        let mbar_bytes = if self.cluster_n > 1 { 8 } else { 0 };
        tile_bytes + reduction_bytes + mbar_bytes
    }

    /// The launch for `m` rows: `ceil(M / rows_per_block) x cluster_n` CTAs,
    /// in clusters of `1 x cluster_n` when `cluster_n > 1`.
    pub fn launch(&self, m: usize) -> LaunchConfig {
        let grid = [m.div_ceil(self.rows_per_block) as u32, self.cluster_n as u32, 1];
        let cfg = LaunchConfig::new(grid, [self.num_threads as u32, 1, 1]);
        if self.cluster_n > 1 { cfg.with_cluster([1, self.cluster_n as u32, 1]) } else { cfg }
    }
}

// =============================================================================
// Hierarchical reductions (reduce.py)
// =============================================================================

/// Asynchronously store a Float32 value to shared memory on a remote CTA
/// within the cluster, with mbarrier completion tracking
/// (`store_shared_remote`): `mapa` translates both local addresses to the
/// peer's, and `st.async.shared::cluster.mbarrier::complete_tx::bytes`
/// stores and signals the peer's mbarrier when the store completes.
pub fn store_shared_remote(val: Float32, smem_ptr: &Ptr<f32>, mbar_ptr: &Ptr<u64>, peer_cta_rank_in_cluster: Int32) {
    store_async_dsmem(smem_ptr, [val], mbar_ptr, peer_cta_rank_in_cluster);
}

/// A pointer to the element at `coord` of `x` (`elem_pointer`), such as the
/// shared-memory address of a hierarchical `(row, (col, cta_rank))` slot.
pub fn elem_pointer<T: Numeric>(x: &Tensor<T>, coord: impl Into<Coord>) -> Ptr<T> {
    x.ptr().offset(crd2idx(coord, x.layout()))
}

/// Reduce values across all warps within a thread block using shared
/// memory (`block_reduce`). Each warp contributes its warp-reduced `val`
/// (lane 0's), at `reduction_buffer[warp_idx / warps_per_row, warp_idx %
/// warps_per_row]`; after a barrier the lanes of each warp read their row's
/// values and reduce them. The buffer is `(rows_per_block, warps_per_row)`.
pub fn block_reduce(
    val: Float32,
    op: impl Fn(Float32, Float32) -> Float32,
    reduction_buffer: &Tensor<f32>,
    init_val: f32,
) -> Float32 {
    let lane_idx = lane_idx();
    let warp_idx = warp_idx();
    let warps_per_row = size(reduction_buffer, &[1]).as_static().expect("static") as i32;
    let row_idx = warp_idx / warps_per_row;
    let col_idx = warp_idx % warps_per_row;

    // Lane 0 of each warp writes its value to shared memory.
    if_(lane_idx.eq(0), || reduction_buffer.store((row_idx, col_idx), val));
    sync_threads();

    // All lanes participate in reading and reducing; only lanes <
    // warps_per_row have valid data.
    let block_reduce_val = if_else_yield(
        lane_idx.lt(warps_per_row),
        || reduction_buffer.load((row_idx, lane_idx)),
        || Float32::constant(init_val),
    );
    warp_reduction(block_reduce_val, op, 32)
}

/// What [`cluster_reduce`] gets wrong on purpose, to show the simulator's
/// diagnostics (the tests use it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Misuse {
    /// The kernel as the example has it.
    None,
    /// The phase's `expect_tx` counts one Float32 partial fewer than the
    /// peers store.
    ExpectTxShort,
    /// The last CTA of the cluster exits right after its own stores, without
    /// waiting on its mbarrier or arriving at the cluster barrier, while its
    /// peers still store into its shared memory.
    ExitBeforeClusterBarrier,
}

/// Reduce values across all CTAs within a cluster using distributed shared
/// memory (`cluster_reduce`). The buffer is `(rows_per_block,
/// (warps_per_row, cluster_n))`: lanes `0..cluster_n` of each warp store the
/// warp's `val` into slot `(row, (col, cta_rank))` of CTA `lane` of the
/// cluster; warp 0 sets up this CTA's mbarrier to expect every peer's
/// bytes, every thread waits on it, then each warp reduces its row's
/// `warps_per_row * cluster_n` values.
pub fn cluster_reduce(
    val: Float32,
    op: impl Fn(Float32, Float32) -> Float32,
    reduction_buffer: &Tensor<f32>,
    mbar_ptr: &Ptr<u64>,
    cluster_n: usize,
    init_val: f32,
    misuse: Misuse,
) -> Float32 {
    let cta_rank_in_cluster = block_idx_in_cluster();
    let lane_idx = lane_idx();
    let warp_idx = warp_idx();

    let rows_per_block = size(reduction_buffer, &[0]).as_static().expect("static");
    let warps_per_row = size(reduction_buffer, &[1, 0]).as_static().expect("static");

    let row_idx = warp_idx / warps_per_row as i32;
    let col_idx = warp_idx % warps_per_row as i32;

    // Warp 0, lane 0 sets up the mbarrier with the expected transaction
    // count: each warp sends cluster_n stores (one to each CTA) of 4 bytes.
    if_(warp_idx.eq(0), || {
        if_(elect_one(), || {
            let num_warps = rows_per_block * warps_per_row;
            let mut expected_bytes = num_warps * cluster_n * 4; // 4 bytes per Float32
            if misuse == Misuse::ExpectTxShort {
                expected_bytes -= 4;
            }
            mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes as i32);
        });
    });

    // Each lane < cluster_n writes to a different CTA's shared memory,
    // distributing the warp's value to all CTAs in the cluster.
    if_(lane_idx.lt(cluster_n as i32), || {
        let slot = elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster)));
        store_shared_remote(val, &slot, mbar_ptr, lane_idx);
    });

    let gather = || {
        // Wait for all cross-CTA stores to complete.
        mbarrier_wait(mbar_ptr, 0);

        // Now each CTA has all values from all CTAs in the cluster; reduce
        // across all collected values.
        let num_total = warps_per_row * cluster_n;
        let num_iter = num_total.div_ceil(32);
        let mut block_reduce_val = Float32::constant(init_val);
        for i in 0..num_iter {
            let idx = lane_idx + (i * 32) as i32;
            let acc = block_reduce_val;
            let slot = (row_idx, (idx % warps_per_row as i32, idx / warps_per_row as i32));
            block_reduce_val = if_else_yield(idx.lt(num_total as i32), || op(acc, reduction_buffer.load(slot)), || acc);
        }
        warp_reduction(block_reduce_val, &op, 32)
    };
    if misuse == Misuse::ExitBeforeClusterBarrier {
        // The last CTA does not wait for its peers' stores (and leaves).
        let last = cta_rank_in_cluster.eq(cluster_n as i32 - 1);
        if_else_yield(last, || Float32::constant(init_val), gather)
    } else {
        gather()
    }
}

/// Hierarchical row reduction with the reduction strategy chosen from
/// `threads_per_row` and `cluster_n` (`row_reduce`): each thread folds its
/// values, a warp reduction over the row's lanes, then a block reduction
/// (more than one warp per row) or a cluster reduction (`cluster_n > 1`).
pub fn row_reduce(
    x: &TensorSsa<f32>,
    op: ReductionOp,
    threads_per_row: usize,
    reduction_buffer: &Tensor<f32>,
    mbar_ptr: Option<&Ptr<u64>>,
    cluster_n: usize,
    init_val: f32,
    misuse: Misuse,
) -> Float32 {
    // Step 1: local reduction, each thread reduces its register values.
    let local_val = x.reduce_all(op, init_val);

    // Map the ReductionOp to a binary operator for warp/block reductions.
    let warp_op = move |a: Float32, b: Float32| match op {
        ReductionOp::Add => a + b,
        ReductionOp::Max => fmax(a, b),
        _ => panic!("row_reduce: {op:?} is not a row reduction (Add or Max)"),
    };

    // Step 2: warp reduction over at most threads_per_row lanes.
    let warp_width = threads_per_row.min(32);
    let warp_val = warp_reduction(local_val, warp_op, warp_width);

    // Steps 3 and 4: block or cluster reduction, if needed.
    let warps_per_row = (threads_per_row / 32).max(1);
    if warps_per_row > 1 || cluster_n > 1 {
        if cluster_n == 1 {
            block_reduce(warp_val, warp_op, reduction_buffer, init_val)
        } else {
            let mbar_ptr = mbar_ptr.expect("a cluster reduction needs an mbarrier");
            cluster_reduce(warp_val, warp_op, reduction_buffer, mbar_ptr, cluster_n, init_val, misuse)
        }
    } else {
        // A single warp handles the entire row.
        warp_val
    }
}

// =============================================================================
// RMSNorm Kernel
// =============================================================================

/// The RMSNorm kernel of `cfg` on `T` elements (`RMSNormKernel`):
/// parameters `x` (`M x N` row-major), `w` (`N`, if `has_weight`), the
/// output `o` (`M x N`), the runtime row count `M` and `eps`. Launch it
/// with [`RMSNormConfig::launch`].
pub fn rmsnorm<T: Numeric>(cfg: RMSNormConfig) -> Kernel {
    rmsnorm_with_misuse::<T>(cfg, Misuse::None)
}

/// [`rmsnorm`] with a deliberate [`Misuse`] (for the diagnostics tests).
pub fn rmsnorm_with_misuse<T: Numeric>(cfg: RMSNormConfig, misuse: Misuse) -> Kernel {
    assert_eq!(cfg.dtype, T::DTYPE, "rmsnorm: the configuration is for {}, not {}", cfg.dtype, T::DTYPE);
    let mut name = format!("rmsnorm_{}_n{}_c{}", T::DTYPE.mlir(), cfg.n, cfg.cluster_n);
    if !cfg.has_weight {
        name.push_str("_noweight");
    }
    if !cfg.fastmath {
        name.push_str("_precise");
    }
    if misuse != Misuse::None {
        name.push_str(&format!("_{misuse:?}").to_lowercase());
    }
    Kernel::builder(&name).block_dim([cfg.num_threads as u32, 1, 1]).build(|| {
        // The host function: tensors from the raw pointers.
        let x_ptr = param_ptr::<T>("x");
        let w_ptr = cfg.has_weight.then(|| param_ptr::<T>("w"));
        let o_ptr = param_ptr::<T>("o");
        let m = param_extent("M", 1);
        let eps = param_scalar::<f32>("eps");
        let n = cfg.n;
        let m_x = make_tensor(&x_ptr, Layout::with_strides((m, n), (n, 1)));
        let m_o = make_tensor(&o_ptr, Layout::with_strides((m, n), (n, 1)));
        let m_w = w_ptr.map(|w| make_tensor(&w, Layout::with_strides(n, 1)));
        let tv_layout =
            RMSNormConfig::make_tv_layout(cfg.threads_per_row, cfg.rows_per_block, cfg.vec_size, cfg.num_vec_blocks);
        let tiler_mn = cfg.tiler_mn();
        kernel::<T>(&cfg, &m_x, m_w.as_ref(), &m_o, eps, &tv_layout, tiler_mn, misuse);
    })
}

/// The device kernel implementing RMSNorm with cluster support.
fn kernel<T: Numeric>(
    cfg: &RMSNormConfig,
    m_x: &Tensor<T>,
    m_w: Option<&Tensor<T>>,
    m_o: &Tensor<T>,
    eps: Float32,
    tv_layout: &Layout,
    tiler_mn: (usize, usize),
    misuse: Misuse,
) {
    let tidx = thread_idx().x;
    let bidx = block_idx().x;
    let cluster_y = if cfg.cluster_n > 1 { block_idx().y } else { Int32::constant(0) };

    let m = Int32::from(size(m_x, &[0]));
    let threads_per_row = tv_layout.mode(0).mode(0).size();
    let warps_per_row = (threads_per_row / 32).max(1);
    let rows_per_block = tiler_mn.0;

    // =====================================================================
    // Allocate shared memory
    // =====================================================================
    let smem = SmemAllocator::new();
    let s_x = smem.allocate_tensor::<T>(make_ordered_layout(tiler_mn, &[1, 0]), 16, None, SmemPartition::User);
    let (reduction_buffer, mbar_ptr) = if cfg.cluster_n == 1 {
        let buffer = smem.allocate_tensor::<f32>(
            Layout::col_major((rows_per_block, warps_per_row)),
            4,
            None,
            SmemPartition::User,
        );
        (buffer, None)
    } else {
        let buffer = smem.allocate_tensor::<f32>(
            Layout::col_major((rows_per_block, (warps_per_row, cfg.cluster_n))),
            4,
            None,
            SmemPartition::User,
        );
        (buffer, Some(smem.allocate_array::<u64>(1, 8, SmemPartition::User)))
    };

    // =====================================================================
    // Initialize cluster
    // =====================================================================
    if let Some(mbar) = &mbar_ptr {
        if_(tidx.eq(0), || mbarrier_init(mbar, 1));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
    }

    // =====================================================================
    // Create identity tensor and partition
    // =====================================================================
    let id_x = make_identity_tensor(m_x.shape().clone());
    let g_x = local_tile(m_x, tiler_mn, (bidx, cluster_y));
    let g_o = local_tile(m_o, tiler_mn, (bidx, cluster_y));
    let c_x = local_tile(&id_x, tiler_mn, (bidx, cluster_y));
    // The weight row, broadcast over the tile's rows (stride 0).
    let g_w = m_w.map(|w| {
        let expanded = w.layout().prepend(Layout::with_strides(tiler_mn.0, 0));
        let m_w_2d = make_tensor(&w.ptr(), expanded);
        local_tile(&m_w_2d, tiler_mn, (0, cluster_y))
    });

    // =====================================================================
    // Create TiledCopy operations
    // =====================================================================
    let copy_atom_load_async = make_copy_atom::<T>(CopyOp::CpAsync { bytes: 16, bypass_l1: false });
    let copy_atom_load_w = make_copy_atom::<T>(CopyOp::Universal { bits: RMSNormConfig::COPY_BITS });
    let copy_atom_store = make_copy_atom::<T>(CopyOp::Universal { bits: RMSNormConfig::COPY_BITS });

    let tiled_copy_load = make_tiled_copy(&copy_atom_load_async, tv_layout, tiler_mn);
    let tiled_copy_w = make_tiled_copy(&copy_atom_load_w, tv_layout, tiler_mn);
    let tiled_copy_store = make_tiled_copy(&copy_atom_store, tv_layout, tiler_mn);

    let thr_copy_x = tiled_copy_load.get_slice(tidx);
    let thr_copy_w = tiled_copy_w.get_slice(tidx);
    let thr_copy_o = tiled_copy_store.get_slice(tidx);

    // Partition tensors.
    let t_xg_x = thr_copy_x.partition_s(&g_x);
    let t_xs_x = thr_copy_x.partition_d(&s_x);
    let t_xg_o = thr_copy_o.partition_d(&g_o);
    let t_xc_x = thr_copy_x.partition_s(&c_x);

    // Register fragments.
    let t_xr_x = make_fragment_like::<T, _>(&t_xg_x);
    let t_xr_o = make_fragment_like::<T, _>(&t_xg_o);

    let weight = g_w.as_ref().map(|g_w| {
        let t_wg_w = thr_copy_w.partition_s(g_w);
        let t_wr_w = make_fragment_like::<T, _>(&t_wg_w);
        let t_xr_w = thr_copy_x.retile_d(&t_wr_w);
        (t_wg_w, t_wr_w, t_xr_w)
    });

    // =====================================================================
    // Bounds checking
    // =====================================================================
    let row_coord = t_xc_x.coord(((0, 0), 0, 0));
    let row_in_bounds = Int32::from(row_coord.mode(0).as_leaf().expect("a row coordinate").clone()).lt(m);
    let t_xp_x = predicate_k(&t_xc_x, cfg.n, row_in_bounds);

    // =====================================================================
    // Async copy global -> shared
    // =====================================================================
    copy_if(&copy_atom_load_async, &t_xp_x, &t_xg_x, &t_xs_x);
    cp_async_commit_group();

    // Load weight while waiting.
    if let Some((t_wg_w, t_wr_w, _)) = &weight {
        let t_wp_w = predicate_k(&thr_copy_w.partition_s(&c_x), cfg.n, Boolean::constant(true));
        t_wr_w.store_ssa(&zeros_like::<T>(t_wr_w));
        copy_if(&copy_atom_load_w, &t_wp_w, t_wg_w, t_wr_w);
    }

    cp_async_wait_group(0);

    // =====================================================================
    // Pass 1: Compute sum of squares with cluster reduction
    // =====================================================================
    autovec_copy(&t_xs_x, &t_xr_x);
    let x = t_xr_x.load_ssa().to::<f32>();

    let x_sq = &x * &x;
    let sum_sq = row_reduce(
        &x_sq,
        ReductionOp::Add,
        threads_per_row,
        &reduction_buffer,
        mbar_ptr.as_ref(),
        cfg.cluster_n,
        0.0,
        misuse,
    );

    let finish = || {
        // rstd = 1 / sqrt(mean(x²) + eps)
        let mean_sq = sum_sq / cfg.n as f32;
        let rstd = math::rsqrt(mean_sq + eps, cfg.fastmath);

        // Sync after reduction.
        if cfg.cluster_n > 1 {
            cluster_arrive_relaxed();
            cluster_wait();
        } else {
            sync_threads();
        }

        // =================================================================
        // Pass 2: Normalize and output
        // =================================================================
        autovec_copy(&t_xs_x, &t_xr_x);
        let x = t_xr_x.load_ssa().to::<f32>();
        let mut y = x * rstd;

        // Apply weight if present.
        if let Some((_, _, t_xr_w)) = &weight {
            let w = t_xr_w.load_ssa().to::<f32>();
            y = y * w;
        }

        // Store to global memory.
        t_xr_o.store_ssa(&y.to::<T>());
        if_(row_in_bounds, || copy_if(&copy_atom_store, &t_xp_x, &t_xr_o, &t_xg_o));
    };
    if misuse == Misuse::ExitBeforeClusterBarrier {
        // The last CTA leaves the reduction early: its peers' stores to it
        // may still be in flight.
        if_(block_idx_in_cluster().ne(cfg.cluster_n as i32 - 1), finish);
    } else {
        finish();
    }
}

// =============================================================================
// Reference and run
// =============================================================================

/// Deterministic data in [-2, 2).
pub fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Reference RMSNorm in f64 (`rmsnorm_ref`): `x / sqrt(mean(x²) + eps) * w`.
pub fn rmsnorm_ref(x: &[f64], weight: Option<&[f64]>, n: usize, eps: f64) -> Vec<f64> {
    x.chunks(n)
        .flat_map(|row| {
            let rms = (row.iter().map(|v| v * v).sum::<f64>() / n as f64 + eps).sqrt();
            row.iter().enumerate().map(move |(k, v)| v / rms * weight.map_or(1.0, |w| w[k]))
        })
        .collect()
}

pub const EPS: f32 = 1e-6;

/// The arguments of the kernel of `cfg` for `m` rows, as `to` converts
/// them: `x`, `w` (if any), `o` (zeros), `M` and `eps`.
pub fn inputs<T: Numeric>(cfg: &RMSNormConfig, m: usize, to: impl Fn(f32) -> T) -> Vec<KernelArg> {
    let conv = |v: Vec<f32>| v.into_iter().map(&to).collect::<Vec<T>>();
    let mut args = vec![KernelArg::buffer(&conv(data(m * cfg.n, 1 + m as u64)))];
    if cfg.has_weight {
        args.push(KernelArg::buffer(&conv(data(cfg.n, 2))));
    }
    args.push(KernelArg::buffer(&conv(vec![0.0; m * cfg.n])));
    args.push(KernelArg::ScalarI32(m as i32));
    args.push(KernelArg::ScalarF32(EPS));
    args
}

/// The index of the output `o` in [`inputs`].
pub fn output_index(cfg: &RMSNormConfig) -> usize {
    if cfg.has_weight { 2 } else { 1 }
}

/// Check the output in `args` against [`rmsnorm_ref`] within `tol`
/// (absolute plus relative): the output's rounding dominates (2^-11 for
/// Float16, 2^-8 for BFloat16); Float32 is held to 1e-4.
pub fn check<T: Numeric>(name: &str, cfg: &RMSNormConfig, args: &[KernelArg], from: impl Fn(T) -> f32, tol: f64) {
    let f64s = |a: &KernelArg| a.to_vec::<T>().into_iter().map(|v| from(v) as f64).collect::<Vec<f64>>();
    let x = f64s(&args[0]);
    let w = cfg.has_weight.then(|| f64s(&args[1]));
    let want = rmsnorm_ref(&x, w.as_deref(), cfg.n, EPS as f64);
    let got = f64s(&args[output_index(cfg)]);
    for (i, (&g, &r)) in got.iter().zip(&want).enumerate() {
        assert!((g - r).abs() <= tol * (1.0 + r.abs()), "{name}: element {i} (row {}): {g} against {r}", i / cfg.n);
    }
}

/// Simulate `kernel` (built for `cfg`) on `m` rows and check it.
pub fn run<T: Numeric>(
    kernel: &Kernel,
    cfg: &RMSNormConfig,
    m: usize,
    to: impl Fn(f32) -> T,
    from: impl Fn(T) -> f32,
    tol: f64,
) -> SimReport {
    let mut args = inputs(cfg, m, to);
    let report =
        kernel.simulate(cfg.launch(m), &mut args).unwrap_or_else(|e| panic!("{} (M = {m}): {e}", kernel.name()));
    check(kernel.name(), cfg, &args, from, tol);
    report
}

pub fn run_f16(kernel: &Kernel, cfg: &RMSNormConfig, m: usize) -> SimReport {
    run::<F16>(kernel, cfg, m, F16::from_f32, F16::to_f32, 4e-3)
}

pub fn run_bf16(kernel: &Kernel, cfg: &RMSNormConfig, m: usize) -> SimReport {
    run::<BF16>(kernel, cfg, m, BF16::from_f32, BF16::to_f32, 2e-2)
}

pub fn run_f32(kernel: &Kernel, cfg: &RMSNormConfig, m: usize) -> SimReport {
    run::<f32>(kernel, cfg, m, |x| x, |x| x, 1e-4)
}

fn main() {
    // One CTA per row on SM80 and for short rows; a cluster of two and of
    // four CTAs on SM90 for rows of 20000 and 40000 Float16 elements (ragged:
    // the cluster's tiles cover 20480 and 40960 columns).
    for (n, sm) in [(4096, 80), (20000, 90), (40000, 90)] {
        let cfg = RMSNormConfig::new(DType::F16, n, true, sm);
        let kernel = rmsnorm::<F16>(cfg);
        for m in [1, 2] {
            run_f16(&kernel, &cfg, m);
        }
        println!(
            "N = {n} on SM{sm}: cluster_n {}, threads_per_row {}, rows_per_block {}: Float16 within tolerance",
            cfg.cluster_n, cfg.threads_per_row, cfg.rows_per_block
        );
    }
}

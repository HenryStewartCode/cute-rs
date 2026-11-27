// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// The mainloop is tensorop_gemm.rs's (by way of runtime_gemm.rs), so it
// follows the design of a CuTe DSL example of NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py,
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// and takes its tile size, its MMA, copy and shared-memory layouts and its
// swizzle from it. Those parts are under the BSD-3-Clause licence, and the rest
// of the file is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md
// has the licence text and describes what is taken.

//! A persistent tensor-core GEMM: the grid holds fewer CTAs than the problem
//! has tiles, and each CTA loops over the tiles a
//! [`StaticPersistentTileScheduler`] hands it, running `tensorop_gemm`'s
//! mainloop on each.
//!
//! `M`, `N` and `K` are runtime values, as in `runtime_gemm`; the schedule
//! reaches the kernel as parameters ([`PersistentTileSchedulerParams::param`]:
//! the tile counts and, unswizzled, two host-prepared FastDivmod divisors).
//! The host sizes the grid with
//! [`get_grid_shape`](PersistentTileSchedulerParams::get_grid_shape) from the
//! number of CTAs that fit at once, which on a GPU comes from
//! `HardwareInfo::get_max_active_clusters(1)`. A swizzled schedule visits
//! padding tiles past the problem, which the kernel skips.
//!
//! Run with `cargo run --release --example persistent_gemm`.

use cute::prelude::*;
use cute::utils::{PersistentTileSchedulerOptions, PersistentTileSchedulerParams, StaticPersistentTileScheduler};

const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 32;
const STAGES: usize = 3;

/// The persistent kernel for a schedule of the given options: A is `m x
/// k`, B is `n x k` and C is `m x n`, all row-major, of runtime extents.
pub fn build(options: PersistentTileSchedulerOptions) -> Kernel {
    Kernel::builder("persistent_gemm").block_dim([128, 1, 1]).build(|| {
        let a = param_ptr::<F16>("a");
        let b = param_ptr::<F16>("b");
        let c = param_ptr::<f32>("c");
        let m = param_extent("m", BM);
        let n = param_extent("n", BN);
        let k = param_extent("k", BK);
        let a = make_tensor(&a, Layout::row_major((m, k)));
        let b = make_tensor(&b, Layout::row_major((n, k)));
        let c = make_tensor(&c, Layout::row_major((m, n)));
        let params = PersistentTileSchedulerParams::param("sched", [1, 1, 1], options);
        let tiles = params.problem_shape_ntile_mnl().clone();
        let (tiles_m, tiles_n) = (Int32::from(size(&tiles, &[0])), Int32::from(size(&tiles, &[1])));
        let sched = StaticPersistentTileScheduler::create(&params, block_idx(), grid_dim());
        let work = sched.initial_work_tile_info();
        while_carry(
            (work, sched),
            |(work, _)| work.is_valid_tile(),
            |(work, mut sched)| {
                let [tile_m, tile_n, _] = work.tile_idx();
                // A swizzled schedule also hands out the padding tiles.
                if_(tile_m.lt(tiles_m) & tile_n.lt(tiles_n), || gemm_tile(&a, &b, &c, tile_m, tile_n));
                sched.advance_to_next_work();
                (sched.get_current_work(), sched)
            },
        );
    })
}

/// One output tile, `(tile_m, tile_n)`: `tensorop_gemm`'s mainloop and
/// epilogue. It starts and ends with every thread at a CTA barrier and no
/// copy in flight, so that the next tile may reuse shared memory.
fn gemm_tile(a: &Tensor<F16>, b: &Tensor<F16>, c: &Tensor<f32>, tile_m: Int32, tile_n: Int32) {
    let tid = thread_idx().x;
    let cta = (tile_m, tile_n, ..);
    let g_a = local_tile_proj(a, (BM, BN, BK), cta, &[true, false, true]);
    let g_b = local_tile_proj(b, (BM, BN, BK), cta, &[false, true, true]);
    let g_c = local_tile_proj(c, (BM, BN, BK), cta, &[true, true, false]);
    let k_tiles = Int32::from(size(&g_a, &[2]));

    let atom = Layout::row_major((8, 32));
    let sw = make_swizzle(2, 3, 3);
    let s_a = make_smem_tensor::<F16>(make_composed_layout(sw, 0, tile_to_shape(&atom, (BM, BK, STAGES))));
    let s_b = make_smem_tensor::<F16>(make_composed_layout(sw, 0, tile_to_shape(&atom, (BN, BK, STAGES))));

    let g2s_atom = make_copy_atom::<F16>(CopyOp::CpAsync { bytes: 16, bypass_l1: true });
    let g2s = make_tiled_copy_tv(&g2s_atom, &Layout::row_major((32, 4)), &Layout::row_major((1, 8)));
    let thr_g2s = g2s.get_slice(tid);
    let t_a_g = thr_g2s.partition_s(&g_a);
    let t_a_s = thr_g2s.partition_d(&s_a);
    let t_b_g = thr_g2s.partition_s(&g_b);
    let t_b_s = thr_g2s.partition_d(&s_b);

    let mma = make_tiled_mma(make_mma_atom(MmaOp::Sm80(MmaKind::M16N8K16F32F16)), Layout::col_major((2, 2, 1)));
    let thr_mma = mma.get_slice(tid);
    let r_a = thr_mma.partition_fragment_a::<F16, _>(&s_a.slice((.., .., 0)));
    let r_b = thr_mma.partition_fragment_b::<F16, _>(&s_b.slice((.., .., 0)));
    let t_c_g = thr_mma.partition_c(&g_c);
    let acc = thr_mma.partition_fragment_c::<f32, _>(&g_c);
    acc.fill(0.0);

    let s2r_a = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
    let s2r_b = make_tiled_copy_b(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 2, trans: false }), &mma);
    let thr_s2r_a = s2r_a.get_slice(tid);
    let thr_s2r_b = s2r_b.get_slice(tid);
    let t_a_s_view = thr_s2r_a.partition_s(&s_a);
    let t_b_s_view = thr_s2r_b.partition_s(&s_b);
    let r_a_view = thr_s2r_a.retile_d(&r_a);
    let r_b_view = thr_s2r_b.retile_d(&r_b);
    let k_blocks = r_a.shape().mode(2).size();

    for s in 0..STAGES - 1 {
        if_(Int32::from(s).lt(k_tiles), || {
            copy(&g2s, &t_a_g.slice((.., .., .., s)), &t_a_s.slice((.., .., .., s)));
            copy(&g2s, &t_b_g.slice((.., .., .., s)), &t_b_s.slice((.., .., .., s)));
        });
        cp_async_commit_group();
    }

    let acc = for_range_carry(0, k_tiles, 1, acc, |k_tile, acc| {
        cp_async_wait_group((STAGES - 2) as u32);
        sync_threads();

        let next = k_tile + (STAGES - 1) as i32;
        if_(next.lt(k_tiles), || {
            let stage = next % STAGES as i32;
            copy(&g2s, &t_a_g.slice((.., .., .., next)), &t_a_s.slice((.., .., .., stage)));
            copy(&g2s, &t_b_g.slice((.., .., .., next)), &t_b_s.slice((.., .., .., stage)));
        });
        cp_async_commit_group();

        let stage = k_tile % STAGES as i32;
        for kb in 0..k_blocks {
            copy(&s2r_a, &t_a_s_view.slice((.., .., kb, stage)), &r_a_view.slice((.., .., kb)));
            copy(&s2r_b, &t_b_s_view.slice((.., .., kb, stage)), &r_b_view.slice((.., .., kb)));
            gemm(&mma, &acc, &r_a.slice((.., .., kb)), &r_b.slice((.., .., kb)), &acc);
        }
        acc
    });

    autovec_copy(&acc, &t_c_g);
    // The next tile's prologue overwrites the stages this one read.
    cp_async_wait_group(0);
    sync_threads();
}

/// The launch and arguments for an `m x n x k` problem on `a` (`m x k`) and
/// `b` (`n x k`), with at most `max_active_clusters` CTAs at once: the grid
/// of `get_grid_shape`, and the schedule's parameters.
pub fn problem_on(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    max_active_clusters: u32,
    options: PersistentTileSchedulerOptions,
) -> (LaunchConfig, Vec<KernelArg>) {
    assert!(m.is_multiple_of(BM) && n.is_multiple_of(BN) && k.is_multiple_of(BK), "size must be a tile multiple");
    assert_eq!((a.len(), b.len()), (m * k, n * k), "A is m x k and B is n x k");
    let params = PersistentTileSchedulerParams::new((m / BM, n / BN, 1), [1, 1, 1], options);
    let grid = params.get_grid_shape(max_active_clusters);
    let mut args = vec![
        KernelArg::f16_from_f32(a),
        KernelArg::f16_from_f32(b),
        KernelArg::f32(vec![0.0; m * n]),
        KernelArg::ScalarI32(m as i32),
        KernelArg::ScalarI32(n as i32),
        KernelArg::ScalarI32(k as i32),
    ];
    args.extend(params.kernel_args());
    (LaunchConfig::new(grid, [128, 1, 1]), args)
}

/// [`problem_on`] with small-integer inputs, so that the result is exact.
pub fn problem(
    m: usize,
    n: usize,
    k: usize,
    max_active_clusters: u32,
    options: PersistentTileSchedulerOptions,
) -> (LaunchConfig, Vec<KernelArg>, Vec<f32>, Vec<f32>) {
    let (a, b) = (small_integers(m * k, 1), small_integers(n * k, 2));
    let (launch, args) = problem_on(m, n, k, &a, &b, max_active_clusters, options);
    (launch, args, a, b)
}

/// `len` integers from -8 to 8, exact in f16, whose sums of products are
/// exact in f32.
pub fn small_integers(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 17) as f32 - 8.0
        })
        .collect()
}

/// Check `c` against a CPU GEMM.
pub fn check(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
            assert_eq!(c[i * n + j], expected, "C[{i},{j}] of {m}x{n}x{k}");
        }
    }
}

/// The schedules [`main`] and the tests try: unswizzled, and swizzled by 2
/// rastering along M and along N.
pub fn variants() -> [PersistentTileSchedulerOptions; 3] {
    [
        PersistentTileSchedulerOptions::default(),
        PersistentTileSchedulerOptions { swizzle_size: 2, raster_along_m: true },
        PersistentTileSchedulerOptions { swizzle_size: 2, raster_along_m: false },
    ]
}

fn main() {
    for options in variants() {
        let kernel = build(options);
        for (m, n, k, ctas) in [(256, 384, 64, 4), (384, 128, 96, 2), (128, 128, 32, 8)] {
            let (launch, mut args, a, b) = problem(m, n, k, ctas, options);
            let report = kernel.simulate(launch, &mut args).unwrap_or_else(|e| panic!("{e}"));
            check(m, n, k, &a, &b, args[2].as_f32());
            println!(
                "{m}x{n}x{k} on {} CTAs ({options:?}): correct, {} mma.sync",
                launch.grid[2], report.mma_instructions
            );
        }
    }
}

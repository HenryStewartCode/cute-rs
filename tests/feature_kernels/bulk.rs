// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/cp_async_bulk.py (the
//   structure of the `cp_async_bulk` kernel),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Bulk copies from shared to global memory, bulk async-groups and proxy
//! fences (sm_90), and `fence.proxy.alias` (sm_70), held to plain Rust.

use cute::prelude::*;

use super::Case;

/// Elements per tile of [`cp_async_bulk_round_trip`].
const BULK_TILE: usize = 256;
/// Tiles per block of [`cp_async_bulk_round_trip`].
const BULK_TILES: usize = 4;
/// Blocks of [`cp_async_bulk_round_trip`].
const BULK_BLOCKS: usize = 2;

/// A port of CuTe DSL's `experimental/primitives/cp_async_bulk.py`: a
/// warp-specialised round trip through a two-stage ring of shared memory.
/// The producer warp's elected lane waits for a stage to be empty, then
/// loads a tile into it with a bulk copy completing on the stage's `full`
/// mbarrier; the consumer warp's elected lane waits on `full`, stores the
/// tile with `cp.async.bulk.global.shared::cta.bulk_group`, commits it,
/// waits for the group (`cp_async_bulk_wait_group(0, false)`) and arrives on
/// `empty`. The source shared memory was written by the async proxy, so it
/// needs no proxy fence.
pub fn cp_async_bulk_round_trip() -> Case {
    const TILE_BYTES: i32 = (BULK_TILE * 2) as i32;
    let kernel = Kernel::builder("cp_async_bulk_round_trip").block_dim([64, 1, 1]).build(|| {
        let n = BULK_BLOCKS * BULK_TILES * BULK_TILE;
        let src = param_tensor::<F16>("src", Layout::col_major(n));
        let dst = param_tensor::<F16>("dst", Layout::col_major(n));
        let base = block_idx().x * (BULK_TILES * BULK_TILE) as i32;
        let smem = alloc_smem::<F16>(2 * BULK_TILE, 128);
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let t = thread_idx().x;
        let (warp, issuer) = (warp_idx(), lane_idx().eq(0));
        if_(t.eq(0), || {
            for s in 0..2usize {
                mbarrier_init(&full.offset(s), 1);
                mbarrier_init(&empty.offset(s), 1);
            }
        });
        mbarrier_init_fence();
        sync_threads();
        if_(t.eq(0), || (0..2usize).for_each(|s| mbarrier_arrive(&empty.offset(s))));
        sync_threads();
        if_(warp.eq(0), || {
            for_range(0, BULK_TILES as i32, 1, |k| {
                let (s, parity) = (k % 2, (k / 2) & 1);
                mbarrier_wait(&empty.offset(s), parity);
                if_(issuer, || {
                    mbarrier_arrive_and_expect_tx(&full.offset(s), TILE_BYTES);
                    let from = src.ptr().offset(base + k * BULK_TILE as i32);
                    cp_async_bulk(&from, &smem.offset(s * BULK_TILE as i32), TILE_BYTES, &full.offset(s));
                });
            });
        });
        if_(warp.eq(1), || {
            for_range(0, BULK_TILES as i32, 1, |k| {
                let (s, parity) = (k % 2, (k / 2) & 1);
                mbarrier_wait(&full.offset(s), parity);
                if_(issuer, || {
                    let to = dst.ptr().offset(base + k * BULK_TILE as i32);
                    cp_async_bulk_global_shared_cta(&to, &smem.offset(s * BULK_TILE as i32), TILE_BYTES);
                    cp_async_bulk_commit_group();
                    cp_async_bulk_wait_group(0, false);
                    mbarrier_arrive(&empty.offset(s));
                });
            });
        });
    });
    fn check(args: &[KernelArg]) {
        let (src, dst) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (s, d)) in src.iter().zip(&dst).enumerate() {
            assert_eq!(s.0, d.0, "cp_async_bulk_round_trip: element {i}");
        }
    }
    let cfg = LaunchConfig::new([BULK_BLOCKS as u32, 1, 1], [64, 1, 1]);
    Case { name: "cp_async_bulk_round_trip", kernel, cfg, small_only: false, check }
}

/// Elements of [`bulk_store_reuse`]'s shared tile.
const REUSE_TILE: usize = 512;
/// Threads of [`bulk_store_reuse`].
const REUSE_THREADS: usize = 128;

/// Shared memory written by ordinary stores and stored to global memory
/// with a `CopyBulkS2GOp` atom, twice through the same buffer: every thread
/// writes its part of the tile, makes it visible to the async proxy
/// (`fence_view_async_shared`), and after a barrier one thread copies the
/// whole tile (`group_bulk_copy_modes`), commits, and waits only for the
/// source to be read (`cp_async_bulk_wait_group(0, true)`) before the
/// barrier that lets the others overwrite it. The final wait completes the
/// writes.
pub fn bulk_store_reuse() -> Case {
    let kernel = Kernel::builder("bulk_store_reuse").block_dim([REUSE_THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(REUSE_TILE));
        let out = param_tensor::<f32>("out", Layout::col_major((REUSE_TILE, 2)));
        let smem = make_smem_tensor::<f32>(Layout::col_major((REUSE_THREADS, REUSE_TILE / REUSE_THREADS)));
        let t = thread_idx().x;
        let atom = make_copy_atom::<f32>(cpasync::CopyBulkS2GOp);
        for round in 0..2usize {
            for j in 0..REUSE_TILE / REUSE_THREADS {
                let v = x.load(t + (j * REUSE_THREADS) as i32);
                smem.store((t, j), v * 2.0 + round as f32);
            }
            fence_view_async_shared();
            sync_threads();
            if_(t.eq(0), || {
                let (s, d) = cpasync::group_bulk_copy_modes(&smem, &out.slice((.., round)));
                copy(&atom, &s, &d);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, true);
            });
            sync_threads();
        }
        if_(t.eq(0), || cp_async_bulk_wait_group(0, false));
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for round in 0..2 {
            for (i, &v) in x.iter().enumerate() {
                assert_eq!(out[round * REUSE_TILE + i], v * 2.0 + round as f32, "round {round}, element {i}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [REUSE_THREADS as u32, 1, 1]);
    Case { name: "bulk_store_reuse", kernel, cfg, small_only: false, check }
}

/// The bulk-copy kernels, which need sm_90.
pub fn bulk_cases_sm90() -> Vec<Case> {
    vec![cp_async_bulk_round_trip(), bulk_store_reuse()]
}

/// `fence.proxy.alias` (sm_70) between a store and a load of the same
/// element, which the SM89 GPU runs.
pub fn fence_proxy_alias() -> Case {
    let kernel = Kernel::builder("fence_proxy_alias").block_dim([64, 1, 1]).build(|| {
        let x = param_tensor::<i32>("x", Layout::col_major(64));
        let y = param_tensor::<i32>("y", Layout::col_major(64));
        let t = thread_idx().x;
        y.store(t, x.load(t) * 3);
        fence_proxy(FenceProxyKind::Alias);
        y.store(t, y.load(t) + 1);
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<i32>(), args[1].to_vec::<i32>());
        for i in 0..64 {
            assert_eq!(y[i], x[i].wrapping_mul(3).wrapping_add(1), "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name: "fence_proxy_alias", kernel, cfg, small_only: false, check }
}

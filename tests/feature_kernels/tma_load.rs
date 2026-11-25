// SPDX-License-Identifier: MIT OR Apache-2.0

//! TMA tile loads (roadmap part m7-tma-load), held to plain Rust: single tiles
//! into each SM90 shared-memory layout atom (128-, 64- and 32-byte swizzles and
//! none, K- and M-major), a K loop through a two-stage ring with phase-tracked
//! mbarriers, edge tiles (zero fill counted in the transaction bytes), a batched
//! `(M, K, L)` tensor, runtime extents, `unroll_factor` over a memory rest mode, and
//! a kernel above 48 KB of shared memory (the dynamic window). Every kernel reads
//! the tiles back through the CuTe shared-memory layout, so the simulator's
//! descriptor model (box traversal, swizzle by address) must agree with the
//! layouts. They need sm_90 and run in the simulator.

use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// Threads of every kernel here: warp 0 issues, all read back.
const THREADS: usize = 128;

/// The `(m, k)` element of a row-major `(rows, cols)` matrix of `F16`s.
fn at(v: &[F16], cols: usize, m: usize, k: usize) -> u16 {
    v[m * cols + k].0
}

/// One tile of `smem_layout`'s shape, loaded by TMA from the tile of `a` at
/// `tile` (a coordinate of `tma_tensor`'s tiles) and written back to `out` at
/// `(row0, col0)` through the shared-memory layout. `a` and `out` are row-major
/// (K-major) or column-major (M-major) `(rows, cols)` matrices, as `k_major` says.
#[track_caller]
fn load_one_tile(
    a: &Tensor<F16>,
    out: &Tensor<F16>,
    smem_layout: &ComposedLayout,
    tile: (usize, usize),
    bm: usize,
    bk: usize,
) {
    let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
    let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, a, smem_layout.clone(), (bm, bk), 1).into();
    let s = make_smem_tensor_aligned::<F16>(smem_layout.clone(), 1024);
    let mbar = alloc_smem::<u64>(1, 8);
    let t = thread_idx().x;
    let (bx, by) = (block_idx().x, block_idx().y);
    let tile_m = bx * tile.0 as i32;
    let tile_k = by * tile.1 as i32;
    if_(t.eq(0), || mbarrier_init(&mbar, 1));
    mbarrier_init_fence();
    sync_threads();
    let g = local_tile(&tma_tensor, (bm, bk), (bx, by));
    let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
    if_(warp_idx().eq(0), || {
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (bm * bk * 2) as i32));
        copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
    });
    mbarrier_wait(&mbar, 0);
    for_range(0, (bm * bk / THREADS) as i32, 1, |j| {
        let idx = j * THREADS as i32 + t;
        let (m, k) = (idx % bm as i32, idx / bm as i32);
        out.store((tile_m + m, tile_k + k), s.load((m, k)));
    });
}

/// A 128x128 f16 matrix, one 64x64 tile per block, through `kind`'s atom tiled to the
/// tile (`tile_to_shape`; M-major atoms tiled along K first).
fn single_tile(name: &'static str, kind: SmemLayoutAtomKind) -> Case {
    const N: usize = 128;
    const B: usize = 64;
    let k_major = kind.major_mode() == cute::OperandMajorMode::K;
    let kernel = Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let layout = if k_major { Layout::new((N, N), (N, 1)) } else { Layout::new((N, N), (1, N)) };
        let a = param_tensor_aligned::<F16>("a", layout.clone(), 16);
        let out = param_tensor::<F16>("out", layout);
        let atom = make_smem_layout_atom::<F16>(kind);
        let smem = if k_major {
            tile_to_shape_swizzled(&atom, (B, B))
        } else {
            tile_to_shape_swizzled_ordered(&atom, (B, B), &[1, 0])
        };
        load_one_tile(&a, &out, &smem, (B, B), B, B);
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (x, y)) in a.iter().zip(&out).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([2, 2, 1], [THREADS as u32, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

pub fn tma_tile_k_sw128() -> Case {
    single_tile("tma_tile_k_sw128", SmemLayoutAtomKind::K_SW128)
}

pub fn tma_tile_k_sw64() -> Case {
    single_tile("tma_tile_k_sw64", SmemLayoutAtomKind::K_SW64)
}

pub fn tma_tile_k_sw32() -> Case {
    single_tile("tma_tile_k_sw32", SmemLayoutAtomKind::K_SW32)
}

pub fn tma_tile_k_inter() -> Case {
    single_tile("tma_tile_k_inter", SmemLayoutAtomKind::K_INTER)
}

pub fn tma_tile_mn_sw128() -> Case {
    single_tile("tma_tile_mn_sw128", SmemLayoutAtomKind::MN_SW128)
}

/// Rows and columns of [`tma_edge_tiles`]' matrix: not multiples of the tile, so the
/// boxes of the last row and column of tiles run past the tensor and are zero-filled.
const EDGE_M: usize = 100;
const EDGE_K: usize = 72;

/// Edge tiles: a `(100, 72)` K-major matrix in 64x64 tiles (2x2 blocks). The boxes past
/// the tensor's end are filled with zeros, and their bytes still count in the
/// transaction (expect_tx takes the whole box).
pub fn tma_edge_tiles() -> Case {
    let kernel = Kernel::builder("tma_edge_tiles").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((EDGE_M, EDGE_K), (EDGE_K, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((128, 128), (128, 1)));
        let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64));
        load_one_tile(&a, &out, &smem, (64, 64), 64, 64);
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for m in 0..128 {
            for k in 0..128 {
                let want = if m < EDGE_M && k < EDGE_K { at(&a, EDGE_K, m, k) } else { 0 };
                assert_eq!(out[m * 128 + k].0, want, "element ({m}, {k})");
            }
        }
    }
    let cfg = LaunchConfig::new([2, 2, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_edge_tiles", kernel, cfg, small_only: false, check }
}

/// A batched `(M, K, L)` tensor, `(64, 64, 3)` with the batch outermost: block `l`
/// loads batch `l`. The batch mode is a third TMA dimension with a box of 1.
pub fn tma_batched() -> Case {
    const L: usize = 3;
    let kernel = Kernel::builder("tma_batched").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 64, L), (64, 1, 64 * 64)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, 64, L), (64, 1, 64 * 64)));
        let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW64), (64, 64));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, smem.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem, 1024);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let l = block_idx().x;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let g = tma_tensor.slice((.., .., l));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 64 * 64 * 2));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        let o = out.slice((.., .., l));
        for_range(0, (64 * 64 / THREADS) as i32, 1, |j| {
            let idx = j * THREADS as i32 + t;
            let (m, k) = (idx % 64, idx / 64);
            o.store((m, k), s.load((m, k)));
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (x, y)) in a.iter().zip(&out).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([L as u32, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_batched", kernel, cfg, small_only: false, check }
}

/// Runtime extents and leading dimension: `(m, k)` with `m` and `k` launch values (13
/// in the tests' inputs) and a row stride `ld` (16), one 16x16 tile through the
/// 32-byte swizzle. The descriptor is encoded at the launch; the box's rows and
/// columns past 13 are zero-filled.
pub fn tma_runtime_extents() -> Case {
    let kernel = Kernel::builder("tma_runtime_extents").block_dim([THREADS as u32, 1, 1]).build(|| {
        let m = param_extent("m", 1);
        let k = param_extent("k", 1);
        let ld = param_extent("ld", 8);
        let a = param_tensor_aligned::<F16>("a", Layout::new((m, k), (ld, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((16, 16), (16, 1)));
        let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW32), (16, 16));
        load_one_tile(&a, &out, &smem, (16, 16), 16, 16);
    });
    fn check(args: &[KernelArg]) {
        let scalar = |a: &KernelArg| match a {
            KernelArg::ScalarI32(x) => *x as usize,
            other => panic!("not an i32 scalar: {other:?}"),
        };
        let (m, k, ld) = (scalar(&args[0]), scalar(&args[1]), scalar(&args[2]));
        let (a, out) = (args[3].to_vec::<F16>(), args[4].to_vec::<F16>());
        for r in 0..16 {
            for c in 0..16 {
                let want = if r < m && c < k { a[r * ld + c].0 } else { 0 };
                assert_eq!(out[r * 16 + c].0, want, "element ({r}, {c}) of a {m}x{k} matrix with ld {ld}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_runtime_extents", kernel, cfg, small_only: false, check }
}

/// Columns of [`tma_k_loop`]'s matrix: 8 k-tiles of 64.
const LOOP_K: usize = 512;

/// A K loop through a two-stage ring: warp 0's elected lane waits for stage `k % 2`
/// to be empty, arms its `full` mbarrier with the tile's bytes and loads k-tile `k`
/// into it (the TMA atom bound to the stage's mbarrier with `with_`); warp 1 waits on
/// `full` with the phase's parity, copies the tile to `out` through the swizzled
/// layout, adds it to a per-lane checksum weighted by `k + 1`, and releases the
/// stage. The checksums of small integers are exact.
pub fn tma_k_loop() -> Case {
    let kernel = Kernel::builder("tma_k_loop").block_dim([64, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, LOOP_K), (LOOP_K, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, LOOP_K), (LOOP_K, 1)));
        let sums = param_tensor::<f32>("sums", Layout::col_major(32));
        let atom_layout = make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
        let staged = tile_to_shape_swizzled(&atom_layout, (64, 64, 2));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, staged.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let (full, empty) = (alloc_smem::<u64>(2, 8), alloc_smem::<u64>(2, 8));
        let t = thread_idx().x;
        if_(t.eq(0), || {
            for st in 0..2usize {
                mbarrier_init(&full.offset(st), 1);
                mbarrier_init(&empty.offset(st), 1);
            }
        });
        mbarrier_init_fence();
        sync_threads();
        if_(t.eq(0), || (0..2usize).for_each(|st| mbarrier_arrive(&empty.offset(st))));
        sync_threads();
        // (64, 64, k-tiles) of the TMA tensor and (64, 64, stage) of shared memory.
        let g = local_tile(&tma_tensor, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let tiles = (LOOP_K / 64) as i32;
        if_(warp_idx().eq(0), || {
            for_range(0, tiles, 1, |k| {
                let (st, parity) = (k % 2, (k / 2) & 1);
                mbarrier_wait(&empty.offset(st), parity);
                let bar = full.offset(st);
                if_(elect_one(), || mbarrier_arrive_and_expect_tx(&bar, 64 * 64 * 2));
                let bound = atom.with_(CopyArgs::new().tma_bar_ptr(&bar));
                copy(&bound, &tg.slice((.., k)), &ts.slice((.., st)));
            });
        });
        if_(warp_idx().eq(1), || {
            let lane = lane_idx();
            let total = for_range_carry(0, tiles, 1, Float32::from(0.0f32), |k, acc| {
                let (st, parity) = (k % 2, (k / 2) & 1);
                mbarrier_wait(&full.offset(st), parity);
                let stage = s.slice((.., .., st));
                let w = (k + 1).to::<f32>();
                let acc = for_range_carry(0, 64 * 64 / 32, 1, acc, |j, acc| {
                    let idx = j * 32 + lane;
                    let (m, c) = (idx % 64, idx / 64);
                    let v = stage.load((m, c));
                    out.store((m, k * 64 + c), v);
                    acc + v.to::<f32>() * w
                });
                sync_warp();
                if_(elect_one(), || mbarrier_arrive(&empty.offset(st)));
                acc
            });
            sums.store(lane, total);
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, out, sums) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
        for (i, (x, y)) in a.iter().zip(&out).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
        for (lane, &got) in sums.iter().enumerate() {
            let mut want = 0.0f32;
            for k in 0..LOOP_K / 64 {
                for j in 0..64 * 64 / 32 {
                    let idx = j * 32 + lane;
                    let (m, c) = (idx % 64, idx / 64);
                    want += F16(at(&a, LOOP_K, m, k * 64 + c)).to_f32() * (k + 1) as f32;
                }
            }
            assert_eq!(got, want, "checksum of lane {lane}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name: "tma_k_loop", kernel, cfg, small_only: true, check }
}

/// Columns of [`tma_unrolled_stages`]' matrix: four 128x64 k-tiles.
const STAGED_K: usize = 256;

/// Four k-tiles of 128x64 loaded at once into four stages (64 KB of shared memory,
/// so the kernel's shared memory is the dynamic window, 1024-byte aligned), by one copy
/// whose rest mode (the stages) is a loop unrolled twice (`unroll_factor`): the
/// coordinates and the destinations are runtime values.
pub fn tma_unrolled_stages() -> Case {
    let kernel = Kernel::builder("tma_unrolled_stages").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((128, STAGED_K), (STAGED_K, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((128, STAGED_K), (STAGED_K, 1)));
        let staged = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (128, 64, 4));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, staged.clone(), (128, 64), 1).into();
        let mbar = alloc_smem::<u64>(1, 8);
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let g = local_tile(&tma_tensor, (128, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (128 * STAGED_K * 2) as i32));
            copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).unroll_factor(2));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, (128 * STAGED_K / THREADS) as i32, 1, |j| {
            let idx = j * THREADS as i32 + t;
            let (m, c) = (idx % 128, idx / 128);
            out.store((m, c), s.load((m, c % 64, c / 64)));
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (x, y)) in a.iter().zip(&out).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_unrolled_stages", kernel, cfg, small_only: false, check }
}

/// The TMA load kernels, which need sm_90.
pub fn tma_load_cases_sm90() -> Vec<Case> {
    vec![
        tma_tile_k_sw128(),
        tma_tile_k_sw64(),
        tma_tile_k_sw32(),
        tma_tile_k_inter(),
        tma_tile_mn_sw128(),
        tma_edge_tiles(),
        tma_batched(),
        tma_runtime_extents(),
        tma_k_loop(),
        tma_unrolled_stages(),
    ]
}

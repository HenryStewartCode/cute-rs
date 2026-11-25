// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py (the design
//   of the A/B multicast with masks from the cluster layout) and
//   examples/python/CuTeDSL/cute/blackwell/kernel/reduce/reduce.py (the design of
//   cluster_reduce),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Copies across the CTAs of a cluster, held to plain
//! Rust: multicast masks from a cluster layout (`make_layout_image_mask`,
//! `create_tma_multicast_mask`); a GEMM-style load in 2x1 and 2x2 clusters where A is
//! multicast along N and B along M, each CTA loading its share of the tiles it shares
//! with its cluster row or column (dense_gemm.py's pattern without WGMMA) and
//! checksumming them; a bulk G2S multicast; a ring of shared-to-shared bulk copies; and a
//! cluster reduction modelled on CuTe DSL's `cluster_reduce`, each CTA `st.async`-ing its
//! partial into every CTA. They need sm_90 and run in the simulator.

use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// Threads of the cluster kernels.
const THREADS: usize = 128;

/// The `(cm, cn, 1)` cluster shapes the mask kernel and its reference cover, with the
/// cluster layout `(1, cm, cn, 1)` (V, M, N, K), column-major.
pub const MASK_CLUSTERS: [(usize, usize); 3] = [(2, 1), (2, 2), (4, 2)];

/// The multicast mask of the CTA at `(m, n)` of a column-major `(cm, cn)` cluster along
/// `mode` (1: M, the CTAs of its column; 2: N, those of its row), in plain Rust.
pub fn reference_mask(cm: usize, cn: usize, m: usize, n: usize, mode: usize) -> u16 {
    let mut mask = 0u16;
    for i in 0..cm {
        for j in 0..cn {
            let same = if mode == 1 { j == n } else { i == m };
            if same {
                mask |= 1 << (i + cm * j);
            }
        }
    }
    mask
}

/// For a `(cm, cn)` cluster: every CTA writes the masks along M and N of its own
/// coordinate (`get_flat_coord` of its rank: runtime values) from
/// `create_tma_multicast_mask`, then those of the static coordinate of every CTA of the
/// cluster from `make_layout_image_mask` (masks known when the kernel is built), as
/// Int16s: `2 + 2 * size` of them per CTA.
pub fn image_masks(cm: usize, cn: usize) -> Kernel {
    let name = format!("image_masks_{cm}x{cn}");
    Kernel::builder(&name).block_dim([32, 1, 1]).build(move || {
        let size = cm * cn;
        let per = 2 + 2 * size;
        let out = param_tensor::<i16>("out", Layout::col_major(per * size));
        let vmnk = Layout::col_major((1, cm, cn, 1));
        let rank = block_idx_in_cluster();
        if_(thread_idx().x.eq(0), || {
            let at = rank * per as i32;
            let coord = vmnk.get_flat_coord(rank);
            for (k, mode) in [1usize, 2].into_iter().enumerate() {
                out.store(at + k as i32, cpasync::create_tma_multicast_mask(&vmnk, coord.clone(), mode));
            }
            for r in 0..size {
                let c: Coord = Tree::Branch(
                    [0, r % cm, r / cm, 0].into_iter().map(|x| Tree::Leaf(IndexExpr::Static(x))).collect(),
                );
                for (k, mode) in [1usize, 2].into_iter().enumerate() {
                    out.store(at + (2 + 2 * r + k) as i32, make_layout_image_mask(&vmnk, c.clone(), mode));
                }
            }
        });
    })
}

/// The values of [`image_masks`]' output for a `(cm, cn)` cluster, in plain Rust.
pub fn image_masks_reference(cm: usize, cn: usize) -> Vec<i16> {
    let size = cm * cn;
    let per = 2 + 2 * size;
    let mut out = vec![0i16; per * size];
    for rank in 0..size {
        for (k, mode) in [1usize, 2].into_iter().enumerate() {
            out[rank * per + k] = reference_mask(cm, cn, rank % cm, rank / cm, mode) as i16;
            for r in 0..size {
                out[rank * per + 2 + 2 * r + k] = reference_mask(cm, cn, r % cm, r / cm, mode) as i16;
            }
        }
    }
    out
}

/// K of the multicast GEMM loads: two k-tiles of 64.
const MC_K: usize = 128;

/// A GEMM's loads in a `(cm, cn)` cluster: CTA `(m, n)` of the grid needs the A tile of
/// row block `m` and the B tile of column block `n` for each of two k-tiles. A is shared
/// by the CTAs of a cluster row (multicast along N: each of the `cn` loads its share of
/// the tile, `num_multicast = cn`) and B by those of a cluster column (along M). Each
/// CTA's mbarrier for a k-tile expects both whole tiles, which its own and its peers'
/// shares fill. The cluster's mbarrier initialisation is published with a cluster
/// barrier before any copy; each CTA then checksums its tiles (weights by position) per
/// thread.
pub fn multicast_gemm_loads(cm: usize, cn: usize) -> Case {
    let name: &'static str = Box::leak(format!("multicast_gemm_loads_{cm}x{cn}").into_boxed_str());
    let kernel = Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let (m_total, n_total) = (64 * cm, 64 * cn);
        let a = param_tensor_aligned::<F16>("a", Layout::new((m_total, MC_K), (MC_K, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((n_total, MC_K), (MC_K, 1)), 16);
        let sums = param_tensor::<f32>("sums", Layout::col_major(cm * cn * THREADS));
        let staged = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, 2));
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
        let (atom_a, tma_a) = cpasync::make_tiled_tma_atom(mc, &a, staged.clone(), (64, 64), cn).into();
        let (atom_b, tma_b) = cpasync::make_tiled_tma_atom(mc, &b, staged.clone(), (64, 64), cm).into();
        let sa = make_smem_tensor_aligned::<F16>(staged.clone(), 1024);
        let sb = make_smem_tensor_aligned::<F16>(staged, 1024);
        let full = alloc_smem::<u64>(2, 8);
        let t = thread_idx().x;
        let (bm, bn) = (block_idx().x, block_idx().y);
        let cta = block_in_cluster_idx();
        let (cta_m, cta_n) = (cta.x, cta.y);
        let cluster_layout = Layout::col_major((cm, cn));
        let mask_a = make_layout_image_mask(&cluster_layout, (cta_m, cta_n), 1);
        let mask_b = make_layout_image_mask(&cluster_layout, (cta_m, cta_n), 0);
        if_(t.eq(0), || (0..2usize).for_each(|k| mbarrier_init(&full.offset(k), 1)));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
        let ga = local_tile(&tma_a, (64, 64), (bm, ..));
        let gb = local_tile(&tma_b, (64, 64), (bn, ..));
        let (tsa, tga) =
            cpasync::tma_partition(&atom_a, cta_n, &make_layout(cn, 1), &sa.group_modes(0, 2), &ga.group_modes(0, 2));
        let (tsb, tgb) =
            cpasync::tma_partition(&atom_b, cta_m, &make_layout(cm, 1), &sb.group_modes(0, 2), &gb.group_modes(0, 2));
        if_(warp_idx().eq(0), || {
            for k in 0..MC_K / 64 {
                let bar = full.offset(k);
                if_(elect_one(), || mbarrier_arrive_and_expect_tx(&bar, 2 * 64 * 64 * 2));
                copy_with(
                    &atom_a,
                    &tga.slice((.., k)),
                    &tsa.slice((.., k)),
                    CopyArgs::new().tma_bar_ptr(&bar).mcast_mask(mask_a),
                );
                copy_with(
                    &atom_b,
                    &tgb.slice((.., k)),
                    &tsb.slice((.., k)),
                    CopyArgs::new().tma_bar_ptr(&bar).mcast_mask(mask_b),
                );
            }
        });
        let mut acc = Float32::from(0.0f32);
        for k in 0..MC_K / 64 {
            mbarrier_wait(&full.offset(k), 0);
            let (ta, tb) = (sa.slice((.., .., k)), sb.slice((.., .., k)));
            acc = for_range_carry(0, 32, 1, acc, |j, acc| {
                let idx = j * THREADS as i32 + t;
                let (r, c) = (idx % 64, idx / 64);
                let wa = (r % 3 + 1).to::<f32>();
                let wb = (c % 5 + 2).to::<f32>();
                acc + ta.load((r, c)).to::<f32>() * wa + tb.load((r, c)).to::<f32>() * wb
            });
        }
        let linear = bm + bn * cm as i32;
        sums.store(linear * THREADS as i32 + t, acc);
        // No CTA leaves while its peers may still use its shared memory.
        cluster_arrive();
        cluster_wait();
    });
    let check: fn(&[KernelArg]) = match (cm, cn) {
        (2, 1) => |args| check_multicast_gemm(args, 2, 1),
        (2, 2) => |args| check_multicast_gemm(args, 2, 2),
        _ => panic!("multicast_gemm_loads: the 2x1 and 2x2 clusters"),
    };
    let cfg =
        LaunchConfig::new([cm as u32, cn as u32, 1], [THREADS as u32, 1, 1]).with_cluster([cm as u32, cn as u32, 1]);
    Case { name, kernel, cfg, small_only: true, check }
}

/// [`multicast_gemm_loads`]' checksums, in plain Rust.
fn check_multicast_gemm(args: &[KernelArg], cm: usize, cn: usize) {
    let (a, b, sums) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
    for bm in 0..cm {
        for bn in 0..cn {
            for t in 0..THREADS {
                let mut acc = 0.0f32;
                for k in 0..MC_K / 64 {
                    for j in 0..32 {
                        let idx = j * THREADS + t;
                        let (r, c) = (idx % 64, idx / 64);
                        let av = a[(bm * 64 + r) * MC_K + k * 64 + c].to_f32();
                        let bv = b[(bn * 64 + r) * MC_K + k * 64 + c].to_f32();
                        acc = acc + av * (r % 3 + 1) as f32 + bv * (c % 5 + 2) as f32;
                    }
                }
                let got = sums[(bm + bn * cm) * THREADS + t];
                assert_eq!(got, acc, "CTA ({bm}, {bn}), thread {t}");
            }
        }
    }
}

/// CTAs of the bulk multicast, ring and reduction kernels' cluster.
const RING: usize = 4;

/// Elements each CTA of [`bulk_multicast`] loads and multicasts: its quarter of 256.
const SLICE: usize = 64;

/// A bulk G2S multicast in a cluster of four: CTA `r` loads elements `64r..64r+64` of `x`
/// with `CopyBulkG2SMulticastOp` (one elected thread, mask 0b1111) into the same offset
/// of every CTA, whose mbarrier expects all 1024 bytes; each CTA then writes the whole
/// row to its row of `out`.
pub fn bulk_multicast() -> Case {
    let kernel = Kernel::builder("bulk_multicast").block_dim([64, 1, 1]).build(|| {
        let x = param_tensor_aligned::<f32>("x", Layout::col_major(SLICE * RING), 16);
        let out = param_tensor::<f32>("out", Layout::new((SLICE * RING, RING), (1, SLICE * RING)));
        let s = make_smem_tensor_aligned::<f32>(Layout::col_major(SLICE * RING), 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
        let atom = make_copy_atom::<f32>(cpasync::CopyBulkG2SMulticastOp);
        if_(t.eq(0), || {
            mbarrier_arrive_and_expect_tx(&mbar, (SLICE * RING * 4) as i32);
            let at = IndexExpr::from(rank) * SLICE;
            let src = make_tensor(&x.ptr().offset(at.clone()), Layout::col_major(SLICE));
            let dst = make_tensor(&s.ptr().offset(at), Layout::col_major(SLICE));
            let mask = Int16::from(((1u32 << RING) - 1) as i16);
            copy_with(&atom, &src, &dst, CopyArgs::new().mbar_ptr(&mbar).mcast_mask(mask));
        });
        mbarrier_wait(&mbar, 0);
        let cluster = block_idx().x / RING as i32;
        if_(cluster.eq(0), || {
            for_range(0, (SLICE * RING / 64) as i32, 1, |j| {
                let i = j * 64 + t;
                out.store((i, rank), s.load(i));
            });
        });
        cluster_arrive();
        cluster_wait();
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for r in 0..RING {
            for i in 0..SLICE * RING {
                assert_eq!(out[r * SLICE * RING + i].to_bits(), x[i].to_bits(), "CTA {r}, element {i}");
            }
        }
    }
    let cfg = LaunchConfig::new([RING as u32, 1, 1], [64, 1, 1]).with_cluster([RING as u32, 1, 1]);
    Case { name: "bulk_multicast", kernel, cfg, small_only: false, check }
}

/// A ring of shared-to-shared bulk copies in a cluster of four: CTA `r` writes 128
/// floats derived from `x` to its `send` buffer, fences them for the async proxy, and
/// one thread copies them into `recv` of CTA `(r + 1) % 4` with `CopyBulkS2SOp`
/// (`cta_rank`), counting down that CTA's mbarrier, which expects 512 bytes. Each CTA
/// writes what it received to `out`; a cluster barrier keeps every CTA (and the source
/// its copy reads) alive until all have.
pub fn s2s_ring() -> Case {
    const N: usize = 128;
    let kernel = Kernel::builder("s2s_ring").block_dim([64, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(N * RING));
        let out = param_tensor::<f32>("out", Layout::col_major(N * RING));
        let send = make_smem_tensor_aligned::<f32>(Layout::col_major(N), 128);
        let recv = make_smem_tensor_aligned::<f32>(Layout::col_major(N), 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        let base = (block_idx().x / RING as i32) * (N * RING) as i32;
        if_(t.eq(0), || {
            mbarrier_init(&mbar, 1);
        });
        mbarrier_init_fence();
        for_range(0, (N / 64) as i32, 1, |j| {
            let i = j * 64 + t;
            send.store(i, x.load(base + rank * N as i32 + i) * 2.0f32 + 1.0f32);
        });
        fence_view_async_shared();
        cluster_arrive();
        cluster_wait();
        let atom = make_copy_atom::<f32>(cpasync::CopyBulkS2SOp);
        if_(t.eq(0), || {
            mbarrier_arrive_and_expect_tx(&mbar, (N * 4) as i32);
            let next = (rank + 1) % RING as i32;
            copy_with(&atom, &send, &recv, CopyArgs::new().mbar_ptr(&mbar).cta_rank(next));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, (N / 64) as i32, 1, |j| {
            let i = j * 64 + t;
            out.store(base + rank * N as i32 + i, recv.load(i));
        });
        cluster_arrive();
        cluster_wait();
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for r in 0..RING {
            let from = (r + RING - 1) % RING;
            for i in 0..N {
                assert_eq!(out[r * N + i], x[from * N + i] * 2.0 + 1.0, "CTA {r}, element {i}");
            }
        }
    }
    let cfg = LaunchConfig::new([RING as u32, 1, 1], [64, 1, 1]).with_cluster([RING as u32, 1, 1]);
    Case { name: "s2s_ring", kernel, cfg, small_only: false, check }
}

/// A cluster reduction modelled on CuTe DSL's `cluster_reduce`: each of the four CTAs
/// of a cluster sums its 512 inputs per warp into four partials, and warp 0's lane 0
/// stores them (`store_async_dsmem`, a `.v4` of Float32) into slot `rank` of every
/// CTA's (its own included) `slots`, whose mbarrier expects 4 x 16 bytes. After the
/// wait each CTA adds the slots up and writes the four totals.
pub fn cluster_reduce_st_async() -> Case {
    const PER_CTA: usize = 512;
    let kernel = Kernel::builder("cluster_reduce_st_async").block_dim([THREADS as u32, 1, 1]).build(|| {
        let x = param_tensor::<f32>("x", Layout::col_major(PER_CTA * RING));
        let out = param_tensor::<f32>("out", Layout::col_major(4 * RING));
        let partial = make_smem_tensor_aligned::<f32>(Layout::col_major(4), 16);
        let slots = make_smem_tensor_aligned::<f32>(Layout::col_major(4 * RING), 16);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        cluster_arrive_relaxed();
        cluster_wait();
        // Each warp's sum of its 128 inputs (lane 0 adds them up).
        let w = warp_idx();
        if_(lane_idx().eq(0), || {
            let s = for_range_carry(0, (PER_CTA / 4) as i32, 1, Float32::from(0.0f32), |i, acc| {
                acc + x.load(rank * PER_CTA as i32 + w * (PER_CTA / 4) as i32 + i)
            });
            partial.store(w, s);
        });
        sync_threads();
        if_(t.eq(0), || {
            mbarrier_arrive_and_expect_tx(&mbar, (RING * 16) as i32);
            let v = [partial.load(0), partial.load(1), partial.load(2), partial.load(3)];
            for peer in 0..RING as i32 {
                store_async_dsmem(&slots.ptr().offset(IndexExpr::from(rank) * 4), v, &mbar, peer);
            }
        });
        mbarrier_wait(&mbar, 0);
        if_(t.lt(4), || {
            let total = (0..RING as i32).fold(Float32::from(0.0f32), |acc, p| acc + slots.load(p * 4 + t));
            out.store(rank * 4 + t, total);
        });
        cluster_arrive();
        cluster_wait();
    });
    fn check(args: &[KernelArg]) {
        const PER_CTA: usize = 512;
        let (x, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for r in 0..RING {
            for w in 0..4 {
                let total = (0..RING).fold(0.0f32, |acc, p| {
                    let s = (0..PER_CTA / 4).fold(0.0f32, |s, i| s + x[p * PER_CTA + w * (PER_CTA / 4) + i]);
                    acc + s
                });
                assert_eq!(out[r * 4 + w], total, "CTA {r}, warp {w}");
            }
        }
    }
    let cfg = LaunchConfig::new([RING as u32, 1, 1], [THREADS as u32, 1, 1]).with_cluster([RING as u32, 1, 1]);
    Case { name: "cluster_reduce_st_async", kernel, cfg, small_only: true, check }
}

/// The cluster-copy kernels, which need sm_90.
pub fn cluster_copy_cases_sm90() -> Vec<Case> {
    vec![
        multicast_gemm_loads(2, 1),
        multicast_gemm_loads(2, 2),
        bulk_multicast(),
        s2s_ring(),
        cluster_reduce_st_async(),
    ]
}

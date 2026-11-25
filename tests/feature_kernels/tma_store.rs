// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py (the design
//   of the TMA-store epilogue),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! TMA stores, reduce stores, L2 cache hints and prefetches (roadmap part m7-tma-store),
//! held to plain Rust: an epilogue that writes tiles to shared memory with plain stores,
//! fences them for the async proxy, synchronises on a named barrier and has one warp
//! TMA-store them through a two-stage buffer (waiting `.read` before each reuse), over a
//! tensor whose edge tiles TMA clips; reduce stores from the four CTAs of a cluster into
//! a tile the first CTA TMA-stores first (float and integer adds, min, max, inc, dec,
//! xor); a K loop whose loads are prefetched to L2 by one `cute.prefetch` and whose
//! descriptor is prefetched at kernel start; and G2R/R2G copy atoms with an L2 cache
//! policy (sm_80, which the GPU runs). The TMA kernels need sm_90 and run in the
//! simulator.

use cute::cpasync::{CacheHintSm90, ReductionKind};
use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// Threads of the epilogue kernels: four warps write, warp 0 stores.
const THREADS: usize = 128;

/// The value the epilogue writes at `(m, n)`: small integers, exact in f16.
fn epilogue_value(m: usize, n: usize) -> f32 {
    ((m * 7 + n * 3) % 61) as f32 - 30.0
}

/// An epilogue over an `(rows, cols)` row-major f16 output in 64x64 tiles, one block per
/// row of tiles: for each column tile `j`, all threads write the tile into stage `j % 2`
/// with plain stores, `fence_view_async_shared`, a named barrier, then warp 0 TMA-stores
/// the stage (with an EVICT_FIRST cache hint), commits, and waits until at most one
/// store still reads shared memory (`.read`) before the next barrier lets the stage be
/// rewritten; a full wait ends the kernel. Tiles past the tensor's edge are clipped.
fn epilogue(name: &'static str, rows: usize, cols: usize) -> Kernel {
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let out = param_tensor_aligned::<F16>("out", Layout::new((rows, cols), (cols, 1)), 16);
        let staged = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, 2));
        let s2g = cpasync::CopyBulkTensorTileS2GOp;
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(s2g, &out, staged.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let t = thread_idx().x;
        let bm = block_idx().x;
        // (64, 64, column tiles) of the TMA tensor for this block's row of tiles.
        let g = local_tile(&tma_tensor, (64, 64), (bm, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let tiles = cols.div_ceil(64);
        for j in 0..tiles {
            let st = j % 2;
            let stage = s.slice((.., .., st));
            for_range(0, (64 * 64 / THREADS) as i32, 1, |k| {
                let idx = k * THREADS as i32 + t;
                let (m, n) = (idx / 64, idx % 64);
                let (gm, gn) = (bm * 64 + m, n + (j * 64) as i32);
                let v = ((gm * 7 + gn * 3) % 61 - 30).to::<f32>().to::<F16>();
                stage.store((m, n), v);
            });
            fence_view_async_shared();
            cute::pipeline::sync(1);
            if_(warp_idx().eq(0), || {
                let hint = CopyArgs::new().cache_policy(CacheHintSm90::EVICT_FIRST);
                copy_with(&atom, &ts.slice((.., st)), &tg.slice((.., j)), hint);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(1, true);
            });
            cute::pipeline::sync(1);
        }
        if_(warp_idx().eq(0), || cp_async_bulk_wait_group(0, false));
    })
}

/// The epilogue's output against [`epilogue_value`].
fn check_epilogue(out: &[F16], rows: usize, cols: usize) {
    for m in 0..rows {
        for n in 0..cols {
            assert_eq!(out[m * cols + n].to_f32(), epilogue_value(m, n), "element ({m}, {n})");
        }
    }
}

/// [`epilogue`] over a `(128, 256)` output: two blocks of four column tiles.
pub fn tma_store_epilogue() -> Case {
    let kernel = epilogue("tma_store_epilogue", 128, 256);
    fn check(args: &[KernelArg]) {
        check_epilogue(&args[0].to_vec::<F16>(), 128, 256);
    }
    let cfg = LaunchConfig::new([2, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_store_epilogue", kernel, cfg, small_only: false, check }
}

/// [`epilogue`] over a `(100, 200)` output: the last row and column of tiles run past
/// the tensor, and TMA writes only their elements inside it.
pub fn tma_store_edge_tiles() -> Case {
    let kernel = epilogue("tma_store_edge_tiles", 100, 200);
    fn check(args: &[KernelArg]) {
        check_epilogue(&args[0].to_vec::<F16>(), 100, 200);
    }
    let cfg = LaunchConfig::new([2, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_store_edge_tiles", kernel, cfg, small_only: false, check }
}

/// CTAs of the reduction kernels' cluster.
pub const REDUCE_CTAS: usize = 4;

/// Elements per side of the reduction kernels' tile.
const RT: usize = 32;

/// The initial value of the reduction kernels' tile at `(m, n)`, and CTA `r`'s operand.
pub fn reduce_init(m: usize, n: usize) -> u64 {
    ((m * 5 + n) % 23) as u64 + 3
}

/// CTA `r`'s operand of the reduction kernels at `(m, n)`. The operands of `inc` and
/// `dec`, whose results depend on the order of the CTAs, are the same in every CTA.
pub fn reduce_operand(kind: ReductionKind, r: usize, m: usize, n: usize) -> u64 {
    let r = if matches!(kind, ReductionKind::INC | ReductionKind::DEC) { 0 } else { r };
    ((m * 3 + n * 11 + r * 7) % 29) as u64 + 1
}

/// A `(32, 32)` row-major tile of `T` reduced by the four CTAs of a cluster: CTA 0
/// TMA-stores [`reduce_init`] into it and waits for the store to complete, a cluster
/// barrier orders that before the others, then every CTA writes its operand
/// ([`reduce_operand`]) to shared memory and reduce-stores it with `kind`
/// (`CopyReduceBulkTensorTileS2GOp`). `to_t` makes a `T` from a small Int32.
fn reduce_kernel<T: Numeric>(name: &'static str, kind: ReductionKind, to_t: fn(Int32) -> Value<T>) -> Kernel {
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let acc = param_tensor_aligned::<T>("acc", Layout::new((RT, RT), (RT, 1)), 16);
        let smem = Layout::new((RT, RT), (RT, 1));
        let store_op = cpasync::CopyBulkTensorTileS2GOp;
        let (store, tma_tensor) = cpasync::make_tiled_tma_atom(store_op, &acc, smem.clone(), (RT, RT), 1).into();
        let red_op = cpasync::CopyReduceBulkTensorTileS2GOp { reduction_kind: kind };
        let (reduce, _) = cpasync::make_tiled_tma_atom(red_op, &acc, smem.clone(), (RT, RT), 1).into();
        let s = make_smem_tensor_aligned::<T>(smem, 128);
        let t = thread_idx().x;
        let rank = block_idx_in_cluster();
        let g = local_tile(&tma_tensor, (RT, RT), (0, 0));
        let (ts, tg) =
            cpasync::tma_partition(&store, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let fill = |value: &dyn Fn(Int32, Int32) -> Int32| {
            for_range(0, (RT * RT / THREADS) as i32, 1, |k| {
                let idx = k * THREADS as i32 + t;
                let (m, n) = (idx / RT as i32, idx % RT as i32);
                s.store((m, n), to_t(value(m, n)));
            });
            fence_view_async_shared();
            sync_threads();
        };
        if_(rank.eq(0), || {
            fill(&|m, n| (m * 5 + n) % 23 + 3);
            if_(warp_idx().eq(0), || {
                copy(&store, &ts, &tg);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, false);
            });
        });
        cluster_arrive();
        cluster_wait();
        let vary = if matches!(kind, ReductionKind::INC | ReductionKind::DEC) { 0 } else { 7 };
        fill(&|m, n| (m * 3 + n * 11 + rank * vary) % 29 + 1);
        if_(warp_idx().eq(0), || {
            copy(&reduce, &ts, &tg);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    })
}

/// The reduction kernels: `(name, kind, element type)`, each checked against a fold of
/// the CTAs' operands into [`reduce_init`] in plain Rust.
pub fn tma_reduce_cases() -> Vec<Case> {
    fn fold(kind: ReductionKind, m: usize, n: usize) -> u64 {
        (0..REDUCE_CTAS).fold(reduce_init(m, n), |a, r| {
            let x = reduce_operand(kind, r, m, n);
            match kind {
                ReductionKind::ADD => a + x,
                ReductionKind::MIN => a.min(x),
                ReductionKind::MAX => a.max(x),
                ReductionKind::INC => {
                    if a >= x {
                        0
                    } else {
                        a + 1
                    }
                }
                ReductionKind::DEC => {
                    if a == 0 || a > x {
                        x
                    } else {
                        a - 1
                    }
                }
                ReductionKind::AND => a & x,
                ReductionKind::OR => a | x,
                ReductionKind::XOR => a ^ x,
            }
        })
    }
    macro_rules! case {
        ($name:literal, $kind:expr, $t:ty, $to:expr, $read:expr) => {{
            fn check(args: &[KernelArg]) {
                let got: Vec<u64> = args[0].to_vec::<$t>().into_iter().map($read).collect();
                for m in 0..RT {
                    for n in 0..RT {
                        assert_eq!(got[m * RT + n], fold($kind, m, n), "{}: element ({m}, {n})", $name);
                    }
                }
            }
            let cfg = LaunchConfig::new([REDUCE_CTAS as u32, 1, 1], [THREADS as u32, 1, 1]).with_cluster([
                REDUCE_CTAS as u32,
                1,
                1,
            ]);
            Case { name: $name, kernel: reduce_kernel::<$t>($name, $kind, $to), cfg, small_only: false, check }
        }};
    }
    vec![
        case!("tma_reduce_add_f32", ReductionKind::ADD, f32, |v| v.to::<f32>(), |x: f32| x as u64),
        case!("tma_reduce_add_f16", ReductionKind::ADD, F16, |v| v.to::<f32>().to::<F16>(), |x: F16| x.to_f32() as u64),
        case!("tma_reduce_add_u32", ReductionKind::ADD, u32, |v| v.to::<u32>(), |x: u32| x as u64),
        case!("tma_reduce_min_u32", ReductionKind::MIN, u32, |v| v.to::<u32>(), |x: u32| x as u64),
        case!("tma_reduce_max_i32", ReductionKind::MAX, i32, |v| v, |x: i32| x as u64),
        case!("tma_reduce_inc_u32", ReductionKind::INC, u32, |v| v.to::<u32>(), |x: u32| x as u64),
        case!("tma_reduce_dec_u32", ReductionKind::DEC, u32, |v| v.to::<u32>(), |x: u32| x as u64),
        case!("tma_reduce_xor_u64", ReductionKind::XOR, u64, |v| v.to::<u64>(), |x: u64| x),
    ]
}

/// Columns of [`tma_prefetch_k_loop`]'s matrix: four k-tiles of 64.
const PREFETCH_K: usize = 256;

/// A K loop of TMA loads through two stages whose every k-tile is prefetched to L2 by
/// one `cute.prefetch` (`prefetch_atom`) on the whole TMA-tensor partition before the
/// loop, after warp 0 prefetched the atom's descriptor (`prefetch_descriptor`) at
/// kernel start. Each k-tile lands in its stage and is copied to `out`.
pub fn tma_prefetch_k_loop() -> Case {
    let kernel = Kernel::builder("tma_prefetch_k_loop").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, PREFETCH_K), (PREFETCH_K, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((64, PREFETCH_K), (PREFETCH_K, 1)));
        let staged = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, 2));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, staged.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let full = alloc_smem::<u64>(2, 8);
        let t = thread_idx().x;
        if_(warp_idx().eq(0), || cpasync::prefetch_descriptor(&atom));
        if_(t.eq(0), || (0..2usize).for_each(|st| mbarrier_init(&full.offset(st), 1)));
        mbarrier_init_fence();
        sync_threads();
        let g = local_tile(&tma_tensor, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(warp_idx().eq(0), || prefetch_atom(&atom, &tg));
        for k in 0..PREFETCH_K / 64 {
            let (st, parity) = (k % 2, (k / 2) as i32 & 1);
            let bar = full.offset(st);
            if_(warp_idx().eq(0), || {
                if_(elect_one(), || mbarrier_arrive_and_expect_tx(&bar, 64 * 64 * 2));
                copy_with(&atom, &tg.slice((.., k)), &ts.slice((.., st)), CopyArgs::new().tma_bar_ptr(&bar));
            });
            mbarrier_wait(&bar, parity);
            let stage = s.slice((.., .., st));
            for_range(0, (64 * 64 / THREADS) as i32, 1, |j| {
                let idx = j * THREADS as i32 + t;
                let (m, c) = (idx % 64, idx / 64);
                out.store((m, c + (k * 64) as i32), stage.load((m, c)));
            });
            sync_threads();
        }
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (x, y)) in a.iter().zip(&out).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_prefetch_k_loop", kernel, cfg, small_only: false, check }
}

/// Elements of [`bulk_atom_cache_hints`]' rows.
const BULK_ROW: usize = 256;

/// The non-tensor bulk-copy atoms with L2 cache hints: one thread prefetches the source
/// rows to L2 (`prefetch_atom` with the `CopyBulkG2SOp` atom,
/// `cp.async.bulk.prefetch.L2`), copies them into shared memory with `CopyBulkG2SOp`
/// (an mbarrier and an EVICT_LAST hint, one 512-byte copy per row), and after the
/// mbarrier wait copies them out with `CopyBulkS2GOp` (an EVICT_FIRST hint), committing
/// and waiting for the bulk group.
pub fn bulk_atom_cache_hints() -> Case {
    let kernel = Kernel::builder("bulk_atom_cache_hints").block_dim([64, 1, 1]).build(|| {
        let x = param_tensor_aligned::<f32>("x", Layout::new((BULK_ROW, 2), (1, BULK_ROW)), 16);
        let y = param_tensor_aligned::<f32>("y", Layout::new((BULK_ROW, 2), (1, BULK_ROW)), 16);
        let s = make_smem_tensor_aligned::<f32>(Layout::new((BULK_ROW, 2), (1, BULK_ROW)), 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let g2s = make_copy_atom::<f32>(cpasync::CopyBulkG2SOp);
        let s2g = make_copy_atom::<f32>(cpasync::CopyBulkS2GOp);
        let t = thread_idx().x;
        if_(t.eq(0), || {
            mbarrier_init(&mbar, 1);
            prefetch_atom(&g2s, &x);
        });
        mbarrier_init_fence();
        sync_threads();
        if_(t.eq(0), || {
            mbarrier_arrive_and_expect_tx(&mbar, (BULK_ROW * 2 * 4) as i32);
            let args = CopyArgs::new().mbar_ptr(&mbar).cache_policy(CacheHintSm90::EVICT_LAST);
            copy_with(&g2s, &x, &s, args);
        });
        mbarrier_wait(&mbar, 0);
        if_(t.eq(0), || {
            copy_with(&s2g, &s, &y, CopyArgs::new().cache_policy(CacheHintSm90::EVICT_FIRST));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    });
    fn check(args: &[KernelArg]) {
        let (x, y) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>());
        for (i, (a, b)) in x.iter().zip(&y).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [64, 1, 1]);
    Case { name: "bulk_atom_cache_hints", kernel, cfg, small_only: false, check }
}

/// The TMA store, reduction and prefetch kernels, which need sm_90.
pub fn tma_store_cases_sm90() -> Vec<Case> {
    let mut v = vec![tma_store_epilogue(), tma_store_edge_tiles(), tma_prefetch_k_loop(), bulk_atom_cache_hints()];
    v.extend(tma_reduce_cases());
    v
}

/// `CopyG2ROp` and `CopyR2GOp` atoms with an L2 cache policy (`ld.global.L2::cache_hint`,
/// `st.global.L2::cache_hint`; sm_80 and PTX 7.4) from `createpolicy`: each thread
/// copies four u32s and two f32 vectors of its own through registers, so the outputs
/// are the inputs, bit for bit, on the GPU too.
pub fn copy_atom_cache_hints() -> Case {
    const N: usize = 64;
    let kernel = Kernel::builder("copy_atom_cache_hints").block_dim([N as u32, 1, 1]).build(|| {
        let xs = param_tensor::<u32>("xs", Layout::col_major(N * 4));
        let fs = param_tensor::<f32>("fs", Layout::col_major(N * 4));
        let out = param_tensor::<u32>("out", Layout::col_major(N * 4));
        let of = param_tensor::<f32>("of", Layout::col_major(N * 4));
        let t = thread_idx().x;
        let policy: Int64 = inline_ptx_with_sim("createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", &[], |_| {
            vec![CacheHintSm90::EVICT_LAST as u64]
        });
        let g2r = make_copy_atom::<u32>(CopyOp::G2R { bits: 32, attrs: G2RAttrs::default() });
        let r2g = make_copy_atom::<u32>(CopyOp::R2G { bits: 32, attrs: R2GAttrs::default() });
        let frag = make_fragment::<u32>(4);
        let src = make_tensor(&xs.ptr().offset(IndexExpr::from(t) * 4), Layout::new(4, 1));
        let dst = make_tensor(&out.ptr().offset(IndexExpr::from(t) * 4), Layout::new(4, 1));
        copy_with(&g2r, &src, &frag, CopyArgs::new().cache_policy(policy));
        copy_with(&r2g, &frag, &dst, CopyArgs::new().cache_policy(policy));
        let g2r = make_copy_atom::<f32>(CopyOp::G2R { bits: 64, attrs: G2RAttrs::default() });
        let r2g = make_copy_atom::<f32>(CopyOp::R2G { bits: 64, attrs: R2GAttrs::default() });
        let frag = make_fragment::<f32>(4);
        let src = make_tensor(&fs.ptr().offset(IndexExpr::from(t) * 4), Layout::new(4, 1));
        let dst = make_tensor(&of.ptr().offset(IndexExpr::from(t) * 4), Layout::new(4, 1));
        copy_with(&g2r, &src, &frag, CopyArgs::new().cache_policy(policy));
        copy_with(&r2g, &frag, &dst, CopyArgs::new().cache_policy(policy));
    });
    fn check(args: &[KernelArg]) {
        assert_eq!(args[0].to_vec::<u32>(), args[2].to_vec::<u32>());
        let (f, o) = (args[1].to_vec::<f32>(), args[3].to_vec::<f32>());
        for (i, (x, y)) in f.iter().zip(&o).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [N as u32, 1, 1]);
    Case { name: "copy_atom_cache_hints", kernel, cfg, small_only: false, check }
}

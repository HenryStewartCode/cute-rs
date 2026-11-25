// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py (the design
//   of the TMA mainloop pipeline with multicast A/B loads and its consumer arrival
//   count, and of the TMA-store epilogue through a staging ring),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The TMA pipelines, held to plain Rust: a producer
//! warp streaming K tiles through a 4-stage `PipelineTmaAsync` ring to a consumer
//! warpgroup (with and without try-wait tokens); multicast A/B loads in 2x1, 1x2 and
//! 2x2 clusters whose consumers release the stages of their multicast partners (with
//! the hand-computed consumer count and with `enable_multicast_signaling`); and a
//! `PipelineTmaStore` epilogue storing 6 tiles through a 2-stage staging ring. They
//! need sm_90 and run in the simulator.

use cute::pipeline::{
    Agent, CooperativeGroup, Pipeline, PipelineTmaAsync, PipelineTmaAsyncOptions, PipelineTmaStore,
    pipeline_init_arrive, pipeline_init_wait,
};
use cute::prelude::*;
use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

use super::Case;

/// Threads of the mainloop kernels: a producer warp and a consumer warpgroup.
const THREADS: usize = 160;
/// Consumer threads (warps 1 to 4).
const CONSUMERS: usize = 128;
/// Rows of the single-CTA ring's matrix (one tile high).
const RING_M: usize = 64;
/// K tiles of the single-CTA ring: three passes over its 4 stages.
const RING_TILES: usize = 12;
/// Stages of the single-CTA ring.
const RING_STAGES: usize = 4;

/// The consumers' sum over their 32 elements of a 64x64 tile `t`.
#[track_caller]
fn sum_tile(t: &Tensor<F16>, ct: Int32) -> Float32 {
    for_range_carry(0, (64 * 64 / CONSUMERS) as i32, 1, Float32::from(0.0f32), move |j, acc| {
        let idx = j * CONSUMERS as i32 + ct;
        let (r, c) = (idx % 64, idx / 64);
        acc + t.load((r, c)).to::<f32>() * (r % 3 + 1).to::<f32>()
    })
}

/// The same sum in plain Rust, over a row-major `(rows, cols)` matrix at rows `r0..`, columns `c0..`.
fn sum_tile_ref(a: &[F16], cols: usize, r0: usize, c0: usize, ct: usize) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..64 * 64 / CONSUMERS {
        let idx = j * CONSUMERS + ct;
        let (r, c) = (idx % 64, idx / 64);
        acc += a[(r0 + r) * cols + c0 + c].to_f32() * (r % 3 + 1) as f32;
    }
    acc
}

/// A producer warp streams the [`RING_TILES`] 64x64 K tiles of a `(64, 768)` K-major
/// f16 matrix into a [`RING_STAGES`]-stage ring with TMA loads through
/// `PipelineTmaAsync` (acquire, copy to the handle's barrier, commit); a consumer
/// warpgroup (consumer count 4: one signalling lane per warp) waits, sums each tile
/// and releases. With `tokens`, both sides probe first and pass the token.
pub fn tma_pipeline_ring(tokens: bool) -> Case {
    let name = if tokens { "tma_pipeline_ring_tokens" } else { "tma_pipeline_ring" };
    let kernel = Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let k_total = 64 * RING_TILES;
        let a = param_tensor_aligned::<F16>("a", Layout::new((RING_M, k_total), (k_total, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::col_major(RING_TILES * CONSUMERS));
        let staged =
            tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, RING_STAGES));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma) = cpasync::make_tiled_tma_atom(g2s, &a, staged.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let pipeline = PipelineTmaAsync::create(
            RING_STAGES,
            CooperativeGroup::new(Agent::Thread, 1),
            CooperativeGroup::new(Agent::Thread, (CONSUMERS / 32) as i32),
            64 * 64 * 2,
            PipelineTmaAsyncOptions::default(),
        );
        let g = local_tile(&tma, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, RING_TILES as i32, 1, producer, |k, mut p| {
                    let token = tokens.then(|| p.try_acquire());
                    let h = p.acquire_and_advance(token);
                    copy_with(
                        &atom,
                        &tg.slice((.., k)),
                        &ts.slice((.., h.index())),
                        CopyArgs::new().tma_bar_ptr(&h.barrier()),
                    );
                    h.commit();
                    p
                });
                p.tail();
            },
            || {
                let ct = thread_idx().x - 32;
                for_range_carry(0, RING_TILES as i32, 1, consumer, |k, mut c| {
                    let token = tokens.then(|| c.try_wait());
                    let h = c.wait_and_advance(token);
                    let sum = sum_tile(&s.slice((.., .., h.index())), ct);
                    h.release();
                    out.store(k * CONSUMERS as i32 + ct, sum);
                    c
                });
            },
        );
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<f32>());
        for k in 0..RING_TILES {
            for ct in 0..CONSUMERS {
                let want = sum_tile_ref(&a, 64 * RING_TILES, 0, 64 * k, ct);
                assert_eq!(out[k * CONSUMERS + ct], want, "tile {k}, consumer thread {ct}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name, kernel, cfg, small_only: true, check }
}

/// K tiles of the multicast pipeline: two passes over its 2 stages.
const MC_TILES: usize = 4;
/// Stages of the multicast pipeline.
const MC_STAGES: usize = 2;

/// A GEMM mainloop's loads in a `(cm, cn)` cluster through `PipelineTmaAsync`: CTA
/// `(m, n)` needs the A tile of row block `m` (multicast along N to the CTAs of its
/// cluster row, each loading a `1/cn` share) and the B tile of column block `n`
/// (multicast along M) for each of [`MC_TILES`] k-tiles, through a 2-stage ring. The
/// consumers of every CTA release each stage to every CTA that writes into it (the
/// `cm + cn - 1` CTAs of its row and column). `auto_count` asks the pipeline to
/// count those arrivals (`enable_multicast_signaling`, the consumer group given as
/// the 128 consumer threads); otherwise the group is the hand-computed
/// `mcast_size * num_warps` threads, as dense_gemm.py has it. The barriers are
/// published with `defer_sync` and `pipeline_init_arrive` / `pipeline_init_wait`.
pub fn tma_pipeline_multicast(cm: usize, cn: usize, auto_count: bool) -> Case {
    let name: &'static str = Box::leak(
        format!("tma_pipeline_multicast_{cm}x{cn}{}", if auto_count { "_auto" } else { "" }).into_boxed_str(),
    );
    let kernel = Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(move || {
        let (m_total, n_total, k_total) = (64 * cm, 64 * cn, 64 * MC_TILES);
        let a = param_tensor_aligned::<F16>("a", Layout::new((m_total, k_total), (k_total, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((n_total, k_total), (k_total, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::col_major(cm * cn * MC_TILES * CONSUMERS));
        let staged =
            tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, MC_STAGES));
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
        let (atom_a, tma_a) = cpasync::make_tiled_tma_atom(mc, &a, staged.clone(), (64, 64), cn).into();
        let (atom_b, tma_b) = cpasync::make_tiled_tma_atom(mc, &b, staged.clone(), (64, 64), cm).into();
        let sa = make_smem_tensor_aligned::<F16>(staged.clone(), 1024);
        let sb = make_smem_tensor_aligned::<F16>(staged, 1024);
        let vmnk = Layout::col_major((1usize, cm, cn, 1usize));
        let consumer_group = if auto_count {
            CooperativeGroup::new(Agent::Thread, CONSUMERS as i32)
        } else {
            CooperativeGroup::new(Agent::Thread, ((cm + cn - 1) * CONSUMERS / 32) as i32)
        };
        let pipeline = PipelineTmaAsync::create(
            MC_STAGES,
            CooperativeGroup::new(Agent::Thread, 1),
            consumer_group,
            2 * 64 * 64 * 2,
            PipelineTmaAsyncOptions {
                cta_layout_vmnk: Some(vmnk),
                enable_multicast_signaling: auto_count,
                defer_sync: true,
                ..Default::default()
            },
        );
        let cluster = Layout::col_major((cm, cn, 1usize)).shape().clone();
        pipeline_init_arrive(Some(cluster.clone()), true);
        pipeline_init_wait(Some(cluster));
        let (bm, bn) = (block_idx().x, block_idx().y);
        let cta = block_in_cluster_idx();
        let (cta_m, cta_n) = (cta.x, cta.y);
        let cluster_layout = Layout::col_major((cm, cn));
        let mask_a = make_layout_image_mask(&cluster_layout, (cta_m, cta_n), 1);
        let mask_b = make_layout_image_mask(&cluster_layout, (cta_m, cta_n), 0);
        let ga = local_tile(&tma_a, (64, 64), (bm, ..));
        let gb = local_tile(&tma_b, (64, 64), (bn, ..));
        let (tsa, tga) =
            cpasync::tma_partition(&atom_a, cta_n, &make_layout(cn, 1), &sa.group_modes(0, 2), &ga.group_modes(0, 2));
        let (tsb, tgb) =
            cpasync::tma_partition(&atom_b, cta_m, &make_layout(cm, 1), &sb.group_modes(0, 2), &gb.group_modes(0, 2));
        let (producer, consumer) = pipeline.make_participants();
        if_else(
            warp_idx().eq(0),
            || {
                let mut p = for_range_carry(0, MC_TILES as i32, 1, producer, |k, mut p| {
                    let h = p.acquire_and_advance(None);
                    let bar = h.barrier();
                    let st = h.index();
                    copy_with(
                        &atom_a,
                        &tga.slice((.., k)),
                        &tsa.slice((.., st)),
                        CopyArgs::new().tma_bar_ptr(&bar).mcast_mask(mask_a),
                    );
                    copy_with(
                        &atom_b,
                        &tgb.slice((.., k)),
                        &tsb.slice((.., st)),
                        CopyArgs::new().tma_bar_ptr(&bar).mcast_mask(mask_b),
                    );
                    h.commit();
                    p
                });
                p.tail();
            },
            || {
                let ct = thread_idx().x - 32;
                let linear = bm + bn * cm as i32;
                for_range_carry(0, MC_TILES as i32, 1, consumer, |k, mut c| {
                    let h = c.wait_and_advance(None);
                    let st = h.index();
                    let sum = sum_tile(&sa.slice((.., .., st)), ct) + sum_tile(&sb.slice((.., .., st)), ct) * 2.0f32;
                    h.release();
                    out.store((linear * MC_TILES as i32 + k) * CONSUMERS as i32 + ct, sum);
                    c
                });
            },
        );
    });
    let check: fn(&[KernelArg]) = match (cm, cn) {
        (2, 1) => |args| check_multicast(args, 2, 1),
        (1, 2) => |args| check_multicast(args, 1, 2),
        (2, 2) => |args| check_multicast(args, 2, 2),
        _ => panic!("tma_pipeline_multicast: the 2x1, 1x2 and 2x2 clusters"),
    };
    let cfg =
        LaunchConfig::new([cm as u32, cn as u32, 1], [THREADS as u32, 1, 1]).with_cluster([cm as u32, cn as u32, 1]);
    Case { name, kernel, cfg, small_only: true, check }
}

/// [`tma_pipeline_multicast`]'s sums, in plain Rust.
fn check_multicast(args: &[KernelArg], cm: usize, cn: usize) {
    let (a, b, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>(), args[2].to_vec::<f32>());
    let k_total = 64 * MC_TILES;
    for bm in 0..cm {
        for bn in 0..cn {
            for k in 0..MC_TILES {
                for ct in 0..CONSUMERS {
                    let want = sum_tile_ref(&a, k_total, 64 * bm, 64 * k, ct)
                        + sum_tile_ref(&b, k_total, 64 * bn, 64 * k, ct) * 2.0;
                    let got = out[((bm + bn * cm) * MC_TILES + k) * CONSUMERS + ct];
                    assert_eq!(got, want, "CTA ({bm}, {bn}), k-tile {k}, consumer thread {ct}");
                }
            }
        }
    }
}

/// Tiles of the store epilogue.
const EPI_TILES: usize = 6;
/// Staging buffers of the store epilogue.
const EPI_STAGES: usize = 2;
/// Threads of the store epilogue.
const EPI_THREADS: usize = 128;

/// An epilogue through `PipelineTmaStore`: for each of [`EPI_TILES`] 64x64 tiles of a
/// `(64, 384)` row-major f16 output, every thread writes `2x + 1` of the input into
/// staging buffer `tile % 2`, `fence_view_async_shared`, named barrier 1; warp 0
/// TMA-stores the buffer, commits and acquires (waits until the other buffer is no
/// longer read); named barrier 1 again. Warp 0 runs `producer_tail` at the end.
pub fn tma_store_pipeline_epilogue() -> Case {
    let kernel = Kernel::builder("tma_store_pipeline_epilogue").block_dim([EPI_THREADS as u32, 1, 1]).build(|| {
        let cols = 64 * EPI_TILES;
        let x = param_tensor::<F16>("x", Layout::new((64, cols), (cols, 1)));
        let out = param_tensor_aligned::<F16>("out", Layout::new((64, cols), (cols, 1)), 16);
        let staged =
            tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128), (64, 64, EPI_STAGES));
        let (atom, tma) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &out, staged.clone(), (64, 64), 1).into();
        let s = make_smem_tensor_aligned::<F16>(staged, 1024);
        let g = local_tile(&tma, (64, 64), (0, ..));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let pipeline = PipelineTmaStore::create(EPI_STAGES, CooperativeGroup::new(Agent::Thread, EPI_THREADS as i32));
        let t = thread_idx().x;
        for_range(0, EPI_TILES as i32, 1, |j| {
            let st = j % EPI_STAGES as i32;
            let stage = s.slice((.., .., st));
            for_range(0, (64 * 64 / EPI_THREADS) as i32, 1, |k| {
                let idx = k * EPI_THREADS as i32 + t;
                let (m, n) = (idx / 64, idx % 64);
                let v = x.load((m, j * 64 + n)).to::<f32>() * 2.0f32 + 1.0f32;
                stage.store((m, n), v.to::<F16>());
            });
            fence_view_async_shared();
            cute::pipeline::sync(1);
            if_(warp_idx().eq(0), || {
                copy(&atom, &ts.slice((.., st)), &tg.slice((.., j)));
                pipeline.producer_commit();
                pipeline.producer_acquire();
            });
            cute::pipeline::sync(1);
        });
        if_(warp_idx().eq(0), || pipeline.producer_tail());
    });
    fn check(args: &[KernelArg]) {
        let (x, out) = (args[0].to_vec::<F16>(), args[1].to_vec::<F16>());
        for (i, (x, y)) in x.iter().zip(&out).enumerate() {
            assert_eq!(y.to_f32(), x.to_f32() * 2.0 + 1.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [EPI_THREADS as u32, 1, 1]);
    Case { name: "tma_store_pipeline_epilogue", kernel, cfg, small_only: true, check }
}

/// Every TMA pipeline kernel (sm_90).
pub fn tma_pipeline_cases_sm90() -> Vec<Case> {
    vec![
        tma_pipeline_ring(false),
        tma_pipeline_ring(true),
        tma_pipeline_multicast(2, 1, false),
        tma_pipeline_multicast(1, 2, false),
        tma_pipeline_multicast(2, 2, false),
        tma_pipeline_multicast(2, 1, true),
        tma_pipeline_multicast(1, 2, true),
        tma_pipeline_multicast(2, 2, true),
        tma_store_pipeline_epilogue(),
    ]
}

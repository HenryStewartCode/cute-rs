// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/warp_named_barrier.py
//   (the kernel `kernel`, the launch of `host` and the host data of `run`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Named-barrier warp specialisation, ported from CuTe DSL's
//! `experimental/primitives/warp_named_barrier.py`: 96 threads in three
//! warps. Warp 0 (the producer) fills `STAGES` shared-memory stages from
//! global memory through a ring of `full` / `empty` mbarriers (one arrival
//! on each empty one up front lets the first round of waits pass; parity
//! `(k / STAGES) & 1`) and flushes its lanes' stores with barrier 0 over its
//! 32 threads before its elected lane arrives on `full`. Warp 1 (the
//! consumer) accumulates the tiles into a result tile and arrives at named
//! barrier 1 (64 threads) without waiting; warp 2 (the epilogue) waits at
//! barrier 1 and stores the result. `num_tiles` is a runtime parameter.
//!
//! The example's `prims` front end maps onto `cute::arch`: `mbarrier_init`,
//! `mbarrier_arrive`, `fence_mbarrier_init` (`mbarrier_init_fence`),
//! `barrier_cta_sync(0)` (`sync_threads`), `barrier_cta_sync(id,
//! thread_count=)` (`barrier`), `barrier_cta_arrive` (`barrier_arrive`) and
//! `elect_sync` (`elect_one`). `mbarrier_try_wait_parity` in a loop needs
//! SM90, as does the fence: [`build`] takes the target, and an SM80/SM89
//! build spins on `mbarrier_test_wait` and drops the fence (the
//! `sync_threads` after the inits publishes them to the CTA).
//!
//! Run with `cargo run --release --example warp_named_barrier`.

use cute::prelude::*;

/// Three warps: producer, consumer, epilogue (`_BLOCK`).
pub const BLOCK: u32 = 96;
/// One element per lane (`_TILE`).
pub const TILE: usize = 32;
/// Barrier 0 over the producer warp's 32 threads (`_BAR_PROD_FLUSH`,
/// `_T_PROD`).
const BAR_PROD_FLUSH: i32 = 0;
const T_PROD: i32 = 32;
/// Barrier 1 over warps 1 and 2 (`_BAR_CONS_EPI`, `_T_CONS_EPI`).
const BAR_CONS_EPI: i32 = 1;
const T_CONS_EPI: i32 = 64;
/// The example's default pipeline depth and tile count.
pub const DEFAULT_STAGES: usize = 2;
pub const DEFAULT_NUM_TILES: usize = 8;
/// The seeds [`run`] simulates under, besides the built-in schedules.
pub const SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Whether `arch` has `mbarrier.try_wait` and `fence.mbarrier_init` (SM90).
pub fn has_try_wait(arch: Arch) -> bool {
    !matches!(arch, Arch::Sm80 | Arch::Sm86 | Arch::Sm89)
}

/// `while not mbarrier_try_wait_parity(bar, parity): pass`, or a
/// `mbarrier_test_wait` spin before SM90.
fn wait_parity(bar: &Ptr<u64>, parity: Int32, sm90: bool) {
    if sm90 {
        while_carry((), |()| !mbarrier_try_wait(bar, parity), |()| ());
    } else {
        while_carry((), |()| !mbarrier_test_wait(bar, parity), |()| ());
    }
}

/// The kernel for a pipeline of `stages` stages, for `arch`.
pub fn build(stages: usize, arch: Arch) -> Kernel {
    let sm90 = has_try_wait(arch);
    Kernel::builder("warp_named_barrier").block_dim([BLOCK, 1, 1]).build(move || {
        let src = param_ptr::<f32>("src");
        let dst = param_tensor::<f32>("dst", Layout::col_major(TILE));
        let num_tiles = param_scalar::<i32>("num_tiles");
        let tidx = thread_idx().x;
        let bidx = block_idx().x;
        let lane = tidx % 32;
        let warp = warp_idx();
        let smem_buf = make_smem_tensor::<f32>(Layout::col_major(stages * TILE));
        let smem_result = make_smem_tensor::<f32>(Layout::col_major(TILE));
        let mbar = alloc_smem::<u64>(2 * stages, 8);
        let full_bar = |s: Int32| mbar.offset(s);
        let empty_bar = |s: Int32| mbar.offset(s + stages as i32);

        if_(warp.eq(0), || {
            if_(elect_one(), || {
                for i in 0..stages as i32 {
                    mbarrier_init(&full_bar(i.into_value()), 1);
                    mbarrier_init(&empty_bar(i.into_value()), 1);
                }
            });
            smem_result.store(lane, 0.0f32);
        });
        if sm90 {
            mbarrier_init_fence();
        }
        sync_threads();
        // Pre-signal the empty barriers (one arrival completes phase 0).
        if_(warp.eq(0), || {
            if_(elect_one(), || {
                for i in 0..stages as i32 {
                    mbarrier_arrive(&empty_bar(i.into_value()));
                }
            })
        });
        sync_threads();

        // Producer (warp 0).
        if_(warp.eq(0), || {
            for_range(0, num_tiles, 1, |k| {
                let s = k % stages as i32;
                let phase = (k / stages as i32) & 1;
                wait_parity(&empty_bar(s), phase, sm90);
                let global_idx = bidx * (num_tiles * TILE as i32) + k * TILE as i32 + lane;
                smem_buf.store(s * TILE as i32 + lane, src.offset(global_idx).load());
                barrier(BAR_PROD_FLUSH, T_PROD);
                if_(elect_one(), || mbarrier_arrive(&full_bar(s)));
            });
        });

        // Consumer (warp 1).
        if_(warp.eq(1), || {
            for_range(0, num_tiles, 1, |k| {
                let s = k % stages as i32;
                let phase = (k / stages as i32) & 1;
                wait_parity(&full_bar(s), phase, sm90);
                smem_result.store(lane, smem_result.load(lane) + smem_buf.load(s * TILE as i32 + lane));
                if_(elect_one(), || mbarrier_arrive(&empty_bar(s)));
            });
            barrier_arrive(BAR_CONS_EPI, T_CONS_EPI);
        });

        // Epilogue (warp 2).
        if_(warp.eq(2), || {
            barrier(BAR_CONS_EPI, T_CONS_EPI);
            dst.store(bidx * TILE as i32 + lane, smem_result.load(lane));
        });
    })
}

/// The launch: one block (`host`).
pub fn launch() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [BLOCK, 1, 1])
}

/// `run`'s data: `src` is `0, 1, 2, ...` over `num_tiles` tiles, `dst`
/// zeros.
pub fn args(num_tiles: usize) -> Vec<KernelArg> {
    let src: Vec<f32> = (0..TILE * num_tiles).map(|i| i as f32).collect();
    vec![KernelArg::f32(src), KernelArg::f32(vec![0.0; TILE]), KernelArg::ScalarI32(num_tiles as i32)]
}

/// `verify`'s check: `dst` is the sum of the tiles of `src`.
pub fn check(args: &[KernelArg]) {
    let (src, dst) = (args[0].as_f32(), args[1].as_f32());
    let tiles = src.len() / TILE;
    for lane in 0..TILE {
        let want: f32 = (0..tiles).map(|k| src[k * TILE + lane]).sum();
        assert_eq!(dst[lane], want, "dst[{lane}]");
    }
}

/// Build for `arch`, simulate under every schedule and [`SEEDS`], and check.
pub fn run(stages: usize, num_tiles: usize, arch: Arch) -> SimReport {
    let mut args = args(num_tiles);
    let report = build(stages, arch)
        .simulate_schedules(launch(), &mut args, &SimOptions::default(), &SEEDS)
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    report
}

fn main() {
    for stages in [2, 4] {
        for num_tiles in [DEFAULT_NUM_TILES, 2 * DEFAULT_NUM_TILES] {
            run(stages, num_tiles, Arch::Sm89);
            run(stages, num_tiles, Arch::Sm90);
            println!("warp_named_barrier (STAGES={stages}, num_tiles={num_tiles}): PASS");
        }
    }
}

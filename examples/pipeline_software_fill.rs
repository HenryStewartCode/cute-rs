// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/pipeline_software_fill.py
//   (the kernel `kernel`, the launch of `host` and the host data of `run`),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A warp-specialised software-fill pipeline (no TMA), ported from CuTe
//! DSL's `experimental/primitives/pipeline_software_fill.py`: global →
//! shared → global through `STAGES` stages of `TILE` (32) Float32 elements.
//! Warp 0 (the producer) waits on `empty[s]`, fills the stage with direct
//! stores, flushes its lanes with barrier 0 over its 32 threads and arrives
//! on `full[s]` from an elected lane; warp 1 (the consumer) waits on
//! `full[s]`, writes the stage back to global memory and arrives on
//! `empty[s]`. Every `empty` mbarrier is arrived once in the prologue (the
//! pre-signal trick), so both sides wait with parity `(k / STAGES) & 1`.
//!
//! As in [`warp_named_barrier`](../warp_named_barrier/index.html), the
//! example's `prims` calls map onto `cute::arch`, and [`build`] takes the
//! target: SM90 waits with `mbarrier_try_wait` and fences the inits, an
//! SM80/SM89 build spins on `mbarrier_test_wait` without the fence.
//!
//! Run with `cargo run --release --example pipeline_software_fill`.

use cute::prelude::*;

/// Producer and consumer warps (`_BLOCK`).
pub const BLOCK: u32 = 64;
/// Float32 elements per stage, one per lane (`_DEFAULT_TILE`, which the
/// example requires to be 32).
pub const TILE: usize = 32;
/// The example's default pipeline depth and tile count.
pub const DEFAULT_STAGES: usize = 2;
pub const DEFAULT_NUM_TILES: usize = 16;
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
    Kernel::builder("pipeline_software_fill").block_dim([BLOCK, 1, 1]).build(move || {
        let src = param_ptr::<f32>("src");
        let dst = param_ptr::<f32>("dst");
        let num_tiles = param_scalar::<i32>("num_tiles");
        let tidx = thread_idx().x;
        let bidx = block_idx().x;
        let warp_idx = tidx / 32;
        let lane = tidx % 32;
        let tile = TILE as i32;

        // Step 1: the shared-memory stages and the two mbarrier arrays.
        let smem_buf = make_smem_tensor::<f32>(Layout::col_major(stages * TILE));
        let mbar = alloc_smem::<u64>(2 * stages, 8);
        let full_bar = |s: Int32| mbar.offset(s);
        let empty_bar = |s: Int32| mbar.offset(s + stages as i32);

        // Step 2: one elected thread initialises the barriers.
        if_(warp_idx.eq(0), || {
            if_(elect_one(), || {
                for i in 0..stages as i32 {
                    mbarrier_init(&full_bar(i.into_value()), 1);
                    mbarrier_init(&empty_bar(i.into_value()), 1);
                }
            })
        });
        if sm90 {
            mbarrier_init_fence();
        }
        sync_threads();

        // Step 3: pre-signal the empty barriers.
        if_(warp_idx.eq(0), || {
            if_(elect_one(), || {
                for i in 0..stages as i32 {
                    mbarrier_arrive(&empty_bar(i.into_value()));
                }
            })
        });
        sync_threads();

        // Step 4: the producer warp fills the pipeline.
        if_(warp_idx.eq(0), || {
            for_range(0, num_tiles, 1, |k| {
                let s = k % stages as i32;
                let phase = (k / stages as i32) & 1;
                wait_parity(&empty_bar(s), phase, sm90);
                let global_idx = bidx * (num_tiles * tile) + k * tile + lane;
                smem_buf.store(s * tile + lane, src.offset(global_idx).load());
                // All 32 lanes' stores are done before the consumer is told.
                barrier(0, 32);
                if_(elect_one(), || mbarrier_arrive(&full_bar(s)));
            });
        });

        // Step 5: the consumer warp drains it.
        if_(warp_idx.eq(1), || {
            for_range(0, num_tiles, 1, |k| {
                let s = k % stages as i32;
                let phase = (k / stages as i32) & 1;
                wait_parity(&full_bar(s), phase, sm90);
                let global_idx = bidx * (num_tiles * tile) + k * tile + lane;
                dst.offset(global_idx).store(smem_buf.load(s * tile + lane));
                if_(elect_one(), || mbarrier_arrive(&empty_bar(s)));
            });
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
    let n = TILE * num_tiles;
    vec![
        KernelArg::f32((0..n).map(|i| i as f32).collect()),
        KernelArg::f32(vec![0.0; n]),
        KernelArg::ScalarI32(num_tiles as i32),
    ]
}

/// `verify`'s check: `dst == src`.
pub fn check(args: &[KernelArg]) {
    assert_eq!(args[1].as_f32(), args[0].as_f32());
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

#[allow(dead_code)]
fn main() {
    for stages in [2, 4] {
        for num_tiles in [DEFAULT_NUM_TILES, 2 * DEFAULT_NUM_TILES] {
            run(stages, num_tiles, Arch::Sm89);
            run(stages, num_tiles, Arch::Sm90);
            println!("pipeline_software_fill (STAGES={stages}, num_tiles={num_tiles}): PASS");
        }
    }
}

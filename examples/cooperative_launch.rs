// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/dsl_tutorials/cooperative_launch.py (the
//   `GlobalBarrier` class's device methods, the kernel
//   `cooperative_kernel` and the launch of `run_cooperative_kernel`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A grid-wide barrier, ported from CuTe DSL's
//! `dsl_tutorials/cooperative_launch.py`: [`GlobalBarrier`] keeps one 32-bit
//! word in global memory, a phase bit (bit 31) over an arrival counter.
//! Each block's leader thread arrives with an `atom.add.release.gpu` after
//! a `sync_threads`; the last block to arrive flips the phase and resets
//! the counter with a second release add; the leaders spin on an
//! `ld.acquire.gpu` until the phase flips, and a `sync_threads` releases
//! the rest of the block. The kernel crosses the barrier ten times with a
//! grid of 2×2×2 blocks of 32×2×2 threads. The example's inline-asm
//! acquire load and release add are `cute::arch`'s `load` and
//! `atomic_add_with` here.
//!
//! The barrier only works if every block of the grid is resident at once,
//! which CUDA guarantees with a cooperative launch (the example's
//! `cooperative=True`, here `LaunchConfig::with_cooperative(true)`, which
//! sets `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` and is checked against the
//! device's occupancy before the launch). The simulator runs a cooperative
//! launch with the whole grid resident; launched without it, one block at a
//! time, it reports the hang, with a hint to launch cooperatively. The example prints a line per
//! iteration from block 0; instead, as a check of the barrier, each leader
//! counts its block into `arrived[i]` before the `i`-th crossing and reads
//! the count back after it into `seen`, which must hold the whole grid.
//!
//! Run with `cargo run --release --example cooperative_launch`.

use cute::prelude::*;

/// The example's grid and block (`run_cooperative_kernel`).
pub const GRID: [u32; 3] = [2, 2, 2];
pub const BLOCK: [u32; 3] = [32, 2, 2];
/// Barrier crossings (`for i in range(10)`).
pub const ITERATIONS: usize = 10;
/// Blocks in the grid.
pub const BLOCKS: usize = 8;

/// The device side of `GlobalBarrier`: the barrier word, whether this
/// thread is its block's leader, the phase it expects to leave, and the
/// grid's block count.
pub struct GlobalBarrier {
    barrier_ptr: Ptr<u32>,
    is_leader: Boolean,
    phase: Uint32,
    number_of_thread_blocks: Uint32,
}

impl GlobalBarrier {
    /// `GlobalBarrier(barrier_ptr)`: the leader is thread (0, 0, 0), the
    /// block count is the grid's.
    pub fn new(barrier_ptr: Ptr<u32>) -> GlobalBarrier {
        let t = thread_idx();
        let g = grid_dim();
        GlobalBarrier {
            barrier_ptr,
            is_leader: t.x.eq(0) & t.y.eq(0) & t.z.eq(0),
            phase: Uint32::from(0u32),
            number_of_thread_blocks: (g.x * g.y * g.z).to::<u32>(),
        }
    }

    /// `_read_barrier`: `ld.global.acquire.gpu.b32`.
    fn read_barrier(&self) -> Uint32 {
        load(
            &self.barrier_ptr,
            LoadOptions { sem: MemoryOrder::Acquire, scope: Some(MemoryScope::Gpu), ..Default::default() },
        )
    }

    /// `_increment_barrier`: `atom.add.release.gpu.u32`, returning the old
    /// value.
    fn increment_barrier(&self, value: Uint32) -> Uint32 {
        atomic_add_with(&self.barrier_ptr, value, AtomicOptions::new(MemoryOrder::Release, MemoryScope::Gpu))
    }

    /// `arrive`: after the block's threads agree, the leader counts the
    /// block in; the last block flips the phase bit and clears the counter.
    pub fn arrive(&self) {
        sync_threads();
        if_(self.is_leader, || {
            let barrier_value = self.increment_barrier(1u32.into_value()) + 1u32;
            if_((barrier_value & 0x7fff_ffffu32).eq(self.number_of_thread_blocks), || {
                let _ = self.increment_barrier(Uint32::from(1u32 << 31) - self.number_of_thread_blocks);
            });
        });
    }

    /// `wait`: the leader spins until the phase bit differs from the phase
    /// it entered with, then the block goes on together.
    pub fn wait(&mut self) {
        if_(self.is_leader, || {
            let expected = self.phase ^ 1u32;
            let first = self.read_barrier();
            while_carry(first, |v| (v >> 31u32).ne(expected), |_| self.read_barrier());
        });
        sync_threads();
        self.phase = self.phase ^ 1u32;
    }

    /// `arrive_and_wait`.
    pub fn arrive_and_wait(&mut self) {
        self.arrive();
        self.wait();
    }
}

/// `cooperative_kernel`, with the per-iteration check described above.
pub fn build() -> Kernel {
    Kernel::builder("cooperative_kernel").block_dim(BLOCK).build(|| {
        let barrier_ptr = param_tensor::<u32>("barrier", Layout::col_major(1));
        let arrived = param_tensor::<u32>("arrived", Layout::col_major(ITERATIONS));
        let seen = param_tensor::<u32>("seen", Layout::col_major(ITERATIONS * BLOCKS));
        let mut barrier = GlobalBarrier::new(barrier_ptr.ptr());
        let b = block_idx();
        let g = grid_dim();
        let block = b.x + g.x * (b.y + g.y * b.z);
        let relaxed = AtomicOptions::new(MemoryOrder::Relaxed, MemoryScope::Gpu);
        for i in 0..ITERATIONS {
            if_(barrier.is_leader, || {
                let _ = atomic_add_with(&arrived.ptr().offset(i), 1u32, relaxed);
            });
            barrier.arrive_and_wait();
            if_(barrier.is_leader, || {
                let opts =
                    LoadOptions { sem: MemoryOrder::Relaxed, scope: Some(MemoryScope::Gpu), ..Default::default() };
                seen.store(block * ITERATIONS as i32 + i as i32, load(&arrived.ptr().offset(i), opts));
            });
        }
    })
}

/// The launch: the example's grid and block, cooperative.
pub fn launch() -> LaunchConfig {
    LaunchConfig::new(GRID, BLOCK).with_cooperative(true)
}

/// The same launch without the cooperative attribute, which the simulator runs one block
/// at a time (and reports as a hang).
pub fn launch_not_cooperative() -> LaunchConfig {
    LaunchConfig::new(GRID, BLOCK)
}

/// The barrier word (`GlobalBarrier.allocate` zeroes it), the counts and
/// what each block saw.
pub fn args() -> Vec<KernelArg> {
    vec![
        KernelArg::buffer(&[0u32]),
        KernelArg::buffer(&[0u32; ITERATIONS]),
        KernelArg::buffer(&[0u32; ITERATIONS * BLOCKS]),
    ]
}

/// Every block saw every block arrive at every crossing, and the barrier
/// word is back to its counter of 0 with the phase it ended in (ten flips).
pub fn check(args: &[KernelArg]) {
    assert_eq!(args[0].to_vec::<u32>(), [0], "barrier word after {ITERATIONS} phase flips");
    assert_eq!(args[1].to_vec::<u32>(), [BLOCKS as u32; ITERATIONS], "arrivals");
    assert_eq!(args[2].to_vec::<u32>(), [BLOCKS as u32; ITERATIONS * BLOCKS], "what each block saw");
}

/// Simulate the cooperative launch (the whole grid resident) under every schedule and a few
/// seeds, and check.
pub fn run() -> SimReport {
    let mut args = args();
    let opts = SimOptions::default();
    let report =
        build().simulate_schedules(launch(), &mut args, &opts, &[0, 1, 2, 3]).unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    report
}

fn main() {
    run();
    println!("cooperative_launch: all {BLOCKS} blocks crossed the grid barrier {ITERATIONS} times");
    let opts = SimOptions { max_idle_steps: 1 << 14, ..SimOptions::default() };
    let e = build().simulate_with(launch_not_cooperative(), &mut args(), &opts).unwrap_err();
    println!("launched without the cooperative attribute, the simulator reports:\n{e}");
}

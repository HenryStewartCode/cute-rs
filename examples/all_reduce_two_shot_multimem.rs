// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py
//   (`all_reduce_multimem_kernel` and the tiling of `all_reduce_multimem`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! A two-shot all-reduce over multicast memory, ported from CuTe DSL's
//! `all_reduce_two_shot_multimem.py`.
//!
//! Every rank (GPU) holds an `M x N` Float32 input in multicast memory. The
//! `M x N` matrix is cut into tiles, and each rank's grid takes `1 /
//! world_size` of them. Per tile, each thread loads 4 contiguous elements of
//! a row with `multimem_ld_reduce_4xf32` (the sum over every rank's input)
//! and stores the sum to every rank's output with `multimem_st_4xb32`. After
//! `sync_threads`, thread 0 adds one to its block's flag on every rank
//! (`multimem_red_add1`, release at system scope) and spins on its own rank's
//! flag (`spin_lock_atom_cas_relaxed_wait`) until every rank has done so,
//! resetting it to zero for the next call.
//!
//! The thread layout is CuTe DSL's: 4 x 32 threads (row-major), each taking
//! a `rows_per_thread x 4` block of the tile, so a tile is `4 *
//! rows_per_thread` rows by 128 columns (the original uses 32 rows per
//! thread: 128 x 128 tiles).
//!
//! Multicast memory needs NVLink-connected sm_90 GPUs, so this runs one
//! rank's grid in the simulator: the multicast parameters get every rank's
//! replica ([`MulticastReplicas`]), and the rank's flags start at
//! `world_size - 1`, standing in for the other ranks' increments. Multi-GPU
//! execution is untested.
//!
//! Run with `cargo run --release --example all_reduce_two_shot_multimem`.

use cute::prelude::*;
use cute::utils::distributed::{
    multimem_ld_reduce_4xf32, multimem_red_add1, multimem_st_4xb32, param_tensor_multicast,
    spin_lock_atom_cas_relaxed_wait,
};

/// The threads of a block: 4 x 32.
pub const THREADS: u32 = 128;

/// An all-reduce problem: the matrix, the tile height and the ranks.
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub m: usize,
    pub n: usize,
    /// Rows each thread reduces (CuTe DSL's 32): the tile is `4 * rows_per_thread x 128`.
    pub rows_per_thread: usize,
    pub world_size: usize,
}

impl Problem {
    pub fn tile_m(&self) -> usize {
        4 * self.rows_per_thread
    }

    /// The number of tiles, which `world_size` divides.
    pub fn tiles(&self) -> usize {
        assert!(self.m.is_multiple_of(self.tile_m()) && self.n.is_multiple_of(128), "{self:?}: whole tiles");
        let t = (self.m / self.tile_m()) * (self.n / 128);
        assert!(t.is_multiple_of(self.world_size), "{self:?}: {t} tiles for {} ranks", self.world_size);
        t
    }

    /// Blocks of one rank's grid.
    pub fn blocks_per_rank(&self) -> usize {
        self.tiles() / self.world_size
    }
}

/// The kernel of rank `local_rank` (CuTe DSL's `all_reduce_multimem_kernel`, whose rank and
/// world size are compile-time constants too). Parameters: `in_mc` and `out_mc` (multicast
/// `M x N` row-major Float32), `flag` (the rank's own flags, one per block) and `flag_mc`
/// (the same flags, multicast).
pub fn all_reduce_kernel(p: Problem, local_rank: usize) -> Kernel {
    let tiles_m = p.m / p.tile_m();
    let chunk = p.blocks_per_rank();
    Kernel::builder("all_reduce_multimem").block_dim([THREADS, 1, 1]).build(move || {
        let g_in = param_tensor_multicast::<f32>("in_mc", Layout::row_major((p.m, p.n)));
        let g_out = param_tensor_multicast::<f32>("out_mc", Layout::row_major((p.m, p.n)));
        let flag = param_tensor::<i32>("flag", Layout::col_major(chunk));
        let flag_mc = param_tensor_multicast::<i32>("flag_mc", Layout::col_major(chunk));
        let tidx = thread_idx().x;
        let bidx = block_idx().x;
        // The tile this block reduces: zipped_divide's tile mode is column-major over tiles.
        let blk_idx = bidx + (local_rank * chunk) as i32;
        let (tile_row, tile_col) = (blk_idx % tiles_m as i32, blk_idx / tiles_m as i32);
        // The thread's rows_per_thread x 4 block (thread layout 4 x 32, row-major).
        let (thr_m, thr_n) = (tidx / 32, tidx % 32);
        let row0 = tile_row * p.tile_m() as i32 + thr_m * p.rows_per_thread as i32;
        let col = tile_col * 128 + thr_n * 4;
        for i in 0..p.rows_per_thread {
            let offset = (row0 + i as i32) * p.n as i32 + col;
            let [x, y, z, w] = multimem_ld_reduce_4xf32(&g_in.ptr().offset(offset));
            multimem_st_4xb32(&g_out.ptr().offset(offset), [x, y, z, w]);
        }
        // Every thread of the block has issued its multimem instructions.
        sync_threads();
        if_(tidx.eq(0), || {
            let sm_id_linear =
                block_idx().x + block_idx().y * grid_dim().x + block_idx().z * grid_dim().x * grid_dim().y;
            multimem_red_add1(&flag_mc.ptr().offset(sm_id_linear), MemoryOrder::Release, MemoryScope::Sys);
            spin_lock_atom_cas_relaxed_wait(&flag.ptr().offset(sm_id_linear), p.world_size as i32, 0, MemoryScope::Sys);
        });
    })
}

/// Rank `r`'s input: `in[r][i] = (i % 97) + 1000 r` (sums of these are exact in Float32).
pub fn rank_input(p: Problem, r: usize) -> Vec<f32> {
    (0..p.m * p.n).map(|i| (i % 97) as f32 + 1000.0 * r as f32).collect()
}

/// Run rank `local_rank`'s grid in the simulator: every rank's input and output replica, and
/// the rank's flags preset to `world_size - 1`. Returns the output replicas and the rank's
/// flags after the launch.
pub fn simulate_rank(p: Problem, local_rank: usize) -> (Vec<Vec<f32>>, Vec<i32>) {
    let kernel = all_reduce_kernel(p, local_rank);
    let chunk = p.blocks_per_rank();
    let inputs: Vec<KernelArg> = (0..p.world_size).map(|r| KernelArg::f32(rank_input(p, r))).collect();
    let outputs: Vec<KernelArg> = (0..p.world_size).map(|_| KernelArg::f32(vec![0.0; p.m * p.n])).collect();
    let others = vec![p.world_size as i32 - 1; chunk];
    let remote_flags: Vec<KernelArg> = (0..p.world_size).map(|_| KernelArg::i32(vec![0; chunk])).collect();
    let mut args = vec![
        KernelArg::f32(rank_input(p, local_rank)),
        KernelArg::f32(vec![0.0; p.m * p.n]),
        KernelArg::i32(others.clone()),
        KernelArg::i32(others),
    ];
    let mut groups = vec![
        MulticastReplicas::new("in_mc", inputs, local_rank),
        MulticastReplicas::new("out_mc", outputs, local_rank),
        MulticastReplicas::new("flag_mc", remote_flags, local_rank).alias("flag"),
    ];
    let cfg = LaunchConfig::new([chunk as u32, 1, 1], [THREADS, 1, 1]).with_cooperative(true);
    kernel.simulate_multicast(cfg, &mut args, &mut groups, &SimOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    let outs = groups[1].replicas.iter().map(|r| r.as_f32().to_vec()).collect();
    (outs, args[2].as_i32().to_vec())
}

/// What every output replica holds in rank `local_rank`'s tiles after its grid: the sum of
/// every rank's input there, and 0 elsewhere.
pub fn expected_output(p: Problem, local_rank: usize) -> Vec<f32> {
    let tiles_m = p.m / p.tile_m();
    let chunk = p.blocks_per_rank();
    let mut out = vec![0.0f32; p.m * p.n];
    for blk in local_rank * chunk..(local_rank + 1) * chunk {
        let (tr, tc) = (blk % tiles_m, blk / tiles_m);
        for r in tr * p.tile_m()..(tr + 1) * p.tile_m() {
            for c in tc * 128..(tc + 1) * 128 {
                let i = r * p.n + c;
                out[i] = (0..p.world_size).map(|k| rank_input(p, k)[i]).sum();
            }
        }
    }
    out
}

fn main() {
    let p = Problem { m: 32, n: 256, rows_per_thread: 2, world_size: 2 };
    for rank in 0..p.world_size {
        let (outs, flags) = simulate_rank(p, rank);
        let want = expected_output(p, rank);
        for (r, out) in outs.iter().enumerate() {
            assert_eq!(*out, want, "rank {rank}'s grid, output replica {r}");
        }
        assert!(flags.iter().all(|&f| f == 0), "the flags are reset");
        println!("rank {rank}: {} tiles reduced into all {} output replicas", p.blocks_per_rank(), p.world_size);
    }
    if let Some(toolchain) = Toolchain::discover() {
        let ptx = all_reduce_kernel(p, 0).compile_ptx(&toolchain, Arch::Sm90a).unwrap_or_else(|e| panic!("{e}"));
        let lines = ptx.lines().filter(|l| l.contains("multimem.")).count();
        println!("sm_90a PTX: {lines} multimem instructions");
    }
}

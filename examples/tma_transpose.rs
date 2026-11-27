// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v1.py
//   (the kernel `Sm100MatrixTransposeKernelV1`, its warp roles, barriers and
//   comments, and the problem sizes of `run_transpose`),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! TMA matrix transpose with a producer-consumer pattern (TMA load, S2R, R2S,
//! TMA store), ported from CuTe DSL's TMA tutorial `tma_v1.py`.
//!
//! Warp roles:
//! - producer: the TMA load warp (warp 4) loads a tile from global memory
//!   into `sA`;
//! - consumers: the transpose warps (warps 0 to 3) wait for the load, then
//!   copy `sA` to `sB` through registers;
//! - consumer: the TMA store warp (warp 5) waits for the transpose, then
//!   stores `sB` to global memory.
//!
//! Synchronisation: `load_mbar_ptr` hands the tile from the TMA load to the
//! transpose warps, `store_mbar_ptr` from the transpose warps to the TMA
//! store. The warps run different branches and wait on each other through
//! the mbarriers, which needs independent warp scheduling (and the
//! simulator's model of it).
//!
//! The transpose happens in the tensors' views, not in which thread copies
//! what: `sA` is the `(M, N)` tile laid out row-major (K-major, as the
//! source), `sB` the same `(M, N)` tile laid out column-major, as the
//! transposed view of the `(N, M)` destination that the store's TMA atom
//! describes. Both use the 128-byte swizzle atom, and the transpose warps'
//! universal tiled copy moves one element per thread and step with the same
//! thread-value layout from `sA` to registers and from registers to `sB`.
//!
//! The port keeps the kernel's structure and names. `@cute.struct` shared
//! storage becomes `alloc_smem` and `make_smem_tensor_aligned`;
//! `sm100_utils.make_smem_layout` is [`make_smem_layout`] here (an SM90
//! layout atom for the operand's majorness and major extent, tiled to the
//! tile and one stage). The TMA store's bulk async-group is waited for at
//! the end, which the Python kernel leaves to the end of the CTA. `M` and
//! `N` are launch values, multiples of 16. Every instruction is SM90's: it
//! compiles for sm_90 and is verified by the simulator (bit for bit) and
//! `ptxas`, as no SM90 GPU is available.
//!
//! Run with `cargo run --release --example tma_transpose`.

use cute::OperandMajorMode;
use cute::cpasync;
use cute::pipeline::NamedBarrier;
use cute::prelude::*;
use cute::utils::sm90::get_smem_layout_atom;
use cute::warpgroup::make_smem_layout_atom_dtype;

/// A staged shared-memory layout for an operand of `leading_mode`
/// majorness (CuTe DSL's `sm100_utils.make_smem_layout(leading_mode,
/// smem_tile_shape, dtype, num_stages)`): the widest swizzle atom whose row
/// fits the major extent, tiled to `(M, N, num_stages)` with the major mode
/// contiguous.
pub fn make_smem_layout(
    leading_mode: OperandMajorMode,
    smem_tile_shape: (usize, usize),
    dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let (m, n) = smem_tile_shape;
    let is_k_major = leading_mode == OperandMajorMode::K;
    let (layout, major) = if is_k_major { (LayoutEnum::ROW_MAJOR, n) } else { (LayoutEnum::COL_MAJOR, m) };
    let atom = make_smem_layout_atom_dtype(get_smem_layout_atom(layout, dtype, major), dtype);
    let order: &[usize] = if is_k_major { &[0, 1, 2] } else { &[1, 0, 2] };
    tile_to_shape_ordered(&atom, (m, n, num_stages), order)
}

/// The tutorial's kernel and its configuration (CuTe DSL's
/// `Sm100MatrixTransposeKernelV1`).
#[derive(Clone, Copy, Debug)]
pub struct Sm100MatrixTransposeKernelV1 {
    pub tile_shape: (usize, usize),
    pub tile_m: usize,
    pub tile_n: usize,
    pub cluster_shape_mn: (usize, usize),
    /// The transpose warps, 0 to `num_trans_warps - 1`.
    pub num_trans_warps: usize,
    pub tma_load_warp_id: i32,
    pub tma_store_warp_id: i32,
    pub threads_per_cta: u32,
    pub num_trans_threads: usize,
    /// Barrier 1: the transpose warps' own synchronisation.
    pub trans_sync_barrier: (i32, i32),
    /// Barrier 2: kept from the tutorial, which declares but does not use it
    /// (the TMA store warp waits on `store_mbar_ptr`).
    pub store_barrier: (i32, i32),
    pub buffer_align_bytes: usize,
}

impl Default for Sm100MatrixTransposeKernelV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sm100MatrixTransposeKernelV1 {
    pub fn new() -> Sm100MatrixTransposeKernelV1 {
        let tile_shape = (128, 128);
        // Specialized warp ids.
        let num_trans_warps = 4; // Maximum number of transpose warps
        let tma_load_warp_id = num_trans_warps as i32;
        let tma_store_warp_id = num_trans_warps as i32 + 1;
        Sm100MatrixTransposeKernelV1 {
            tile_shape,
            tile_m: tile_shape.0,
            tile_n: tile_shape.1,
            cluster_shape_mn: (1, 1),
            num_trans_warps,
            tma_load_warp_id,
            tma_store_warp_id,
            threads_per_cta: 32 * (num_trans_warps as u32 + 2),
            num_trans_threads: 32 * num_trans_warps,
            trans_sync_barrier: (1, 32 * num_trans_warps as i32),
            store_barrier: (2, 32),
            buffer_align_bytes: 1024,
        }
    }

    /// The kernel (CuTe DSL's `__call__` and `kernel`): parameters `m`, `n`
    /// (launch values, multiples of 16), then `src`, a row-major `(m, n)`
    /// Float16 tensor, and `dst`, a row-major `(n, m)` one.
    pub fn build(&self) -> Kernel {
        let cfg = *self;
        Kernel::builder("tma_transpose").block_dim([cfg.threads_per_cta, 1, 1]).target(Arch::Sm90).build(move || {
            let m = param_extent("m", 16);
            let n = param_extent("n", 16);
            let src = param_tensor_aligned::<F16>("src", Layout::new((m, n), (n, 1)), 16);
            let dst = param_tensor_aligned::<F16>("dst", Layout::new((n, m), (m, 1)), 16);

            // Create a transposed view of dst for the TMA descriptor:
            // dst is (N, M), viewed as (M, N) transposed.
            let dst_layout = dst.layout();
            let transed_dst = make_tensor(
                &dst.ptr(),
                Layout::new(
                    (dst_layout.shape().mode(1).clone(), dst_layout.shape().mode(0).clone()),
                    (dst_layout.stride().mode(1).clone(), dst_layout.stride().mode(0).clone()),
                ),
            );

            // row-major smem layout for sA (tile_m, tile_n)
            let smem_layout_s_a = make_smem_layout(
                LayoutEnum::from_tensor(&src).mma_major_mode(),
                (cfg.tile_m, cfg.tile_n),
                DType::F16,
                1,
            );
            // col-major smem layout for sB (tile_m, tile_n): sB matches the transposed
            // destination layout.
            let smem_layout_s_b = make_smem_layout(
                LayoutEnum::from_tensor(&transed_dst).mma_major_mode(),
                (cfg.tile_m, cfg.tile_n),
                DType::F16,
                1,
            );

            // TMA atoms. The swizzled layouts make the TMA load and store swizzle.
            let (tma_atom_src, tma_tensor_src) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SOp::default(),
                &src,
                smem_layout_s_a.clone(),
                (cfg.tile_m, cfg.tile_n),
                1,
            )
            .into();
            let (tma_atom_dst, tma_tensor_dst) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileS2GOp,
                &transed_dst,
                smem_layout_s_b.clone(),
                (cfg.tile_m, cfg.tile_n),
                1,
            )
            .into();
            cfg.kernel(
                &tma_atom_src,
                &tma_tensor_src,
                &tma_atom_dst,
                &tma_tensor_dst,
                &smem_layout_s_a,
                &smem_layout_s_b,
            );
        })
    }

    /// The kernel body (CuTe DSL's `kernel`).
    fn kernel(
        &self,
        tma_atom_load: &CopyAtom<F16>,
        tma_tensor_src: &Tensor<Ident>,
        tma_atom_store: &CopyAtom<F16>,
        tma_tensor_dst: &Tensor<Ident>,
        smem_layout_s_a: &ComposedLayout,
        smem_layout_s_b: &ComposedLayout,
    ) {
        let (bidx, bidy) = (block_idx().x, block_idx().y);
        let tidx = thread_idx().x;

        let warp_idx = make_warp_uniform(warp_idx());

        // Allocate shared memory. Two buffers for the transpose:
        // sA: the source tile (swizzled), sB: the destination tile (swizzled) for the TMA store.
        let s_a = make_smem_tensor_aligned::<F16>(smem_layout_s_a.clone(), self.buffer_align_bytes);
        let s_b = make_smem_tensor_aligned::<F16>(smem_layout_s_b.clone(), self.buffer_align_bytes);
        let num_tma_load_bytes =
            size_in_bytes::<F16>(smem_layout_s_a).as_static().expect("the shared-memory layout is static") as i32;
        let load_mbar_ptr = alloc_smem::<u64>(1, 8);
        let store_mbar_ptr = alloc_smem::<u64>(1, 8);

        // ------------------------------------------------------------------
        // Initialize Barriers
        // ------------------------------------------------------------------
        if_(tidx.eq(0), || {
            // Barrier for TMA Load: expect 1 arrive (from the TMA warp after it issues the load)
            mbarrier_init(&load_mbar_ptr, 1);
            mbarrier_expect_tx(&load_mbar_ptr, num_tma_load_bytes);

            // Barrier for TMA Store: expect arrival from the Trans warps
            mbarrier_init(&store_mbar_ptr, self.num_trans_warps as i32);
        });
        mbarrier_init_fence();

        // Sync all warps after barrier initialization
        sync_threads();

        // ------------------------------------------------------------------
        // PRODUCER: TMA Load Warp (G -> sA)
        // ------------------------------------------------------------------
        if_(warp_idx.eq(self.tma_load_warp_id), || {
            // Issue TMA Load
            // ((TileM, TileK), loopM, LoopK)
            let g_a = local_tile(tma_tensor_src, self.tile_shape, (.., ..));
            let (t_a_s_a, t_a_g_a) = cpasync::tma_partition(
                tma_atom_load,
                0,
                &make_layout(1, 0),
                &s_a.group_modes(0, 2),
                &g_a.group_modes(0, 2),
            );

            copy_with(
                tma_atom_load,
                &t_a_g_a.slice((.., bidx, bidy)),
                &t_a_s_a.slice((.., 0)),
                CopyArgs::new().tma_bar_ptr(&load_mbar_ptr),
            );

            // Arrive on mbarrier to satisfy the init count of 1
            if_(elect_one(), || mbarrier_arrive(&load_mbar_ptr));
        });

        // ------------------------------------------------------------------
        // CONSUMER: Transpose Warps (sA -> Reg -> sB)
        // ------------------------------------------------------------------
        if_(warp_idx.lt(self.tma_load_warp_id), || {
            let trans_tid = tidx % self.num_trans_threads as i32;

            // Wait for TMA Load to complete (consumer wait on load_mbar)
            mbarrier_wait(&load_mbar_ptr, 0);

            // Copy one element at a time.
            let atom = make_copy_atom::<F16>(CopyOp::Universal { bits: 16 });

            let copy_elems = 1;

            // Use the SAME thread layout for both read and write: the transpose happens
            // through the sB layout view.
            //
            // TV layout notation: T = thread-id, V = value-lane within that thread.
            // Here `copy_elems = 1` and `thread_layout` has shape (T, V) =
            // (num_trans_threads, 1), so V is always 0 (one value-lane per thread).
            //
            // Linearization rule (row-major by strides):
            //   idx(Ti, Vj) = i + j * num_trans_threads
            // Therefore here:
            //   idx(Ti, V0) = i
            //
            // Diagram (V is the column, T is the row; showing the first two threads):
            //
            //            V0
            //          ┌──────┐
            //   T0     │ T0V0 │  -> idx 0
            //   T1     │ T1V0 │  -> idx 1
            //   ...    │  ... │
            //          └──────┘
            //
            let thread_layout = Layout::new((self.num_trans_threads, 1), (1, self.num_trans_threads));
            let value_layout = Layout::col_major((1, copy_elems));
            // A "tiled copy" defines the per-thread copy mapping (T, V) for this warp group.
            // It is used twice below, via partition_S and partition_D, to create matching
            // per-thread views of the source and destination tensors. With the SAME (T, V)
            // mapping, the transpose comes from the tensor views (sA vs sB), not from which
            // threads copy what.
            let tiled_copy = make_tiled_copy_tv(&atom, &thread_layout, &value_layout);
            let thr_copy = tiled_copy.get_slice(trans_tid);

            // Partition sA (source) for reading
            let t_c_s_a = thr_copy.partition_s(&s_a);
            // make_fragment_like(tCsA) gives a register fragment with the same per-thread
            // shape as tCsA, so it matches the tiled copy's view without a retile.
            let t_c_r_a: Tensor<F16> = make_fragment_like(&t_c_s_a);
            copy(&tiled_copy, &t_c_s_a, &t_c_r_a);

            // Partition sB for writing
            let t_c_s_b = thr_copy.partition_d(&s_b);

            // Write from register to sB
            copy(&tiled_copy, &t_c_r_a, &t_c_s_b);

            // Fence and barrier to make the shared-memory stores visible to the TMA store
            fence_proxy_with(
                FenceProxyKind::AsyncShared,
                FenceProxyOptions { space: Some(SharedSpace::Cta), ..Default::default() },
            );
            NamedBarrier::new(self.trans_sync_barrier.0, self.trans_sync_barrier.1).arrive_and_wait();

            // Trans warps signal the TMA Store warp: "sB is ready!"
            if_(elect_one(), || mbarrier_arrive(&store_mbar_ptr));
        });

        // ------------------------------------------------------------------
        // CONSUMER: TMA Store Warp (sB -> G)
        // ------------------------------------------------------------------
        if_(warp_idx.eq(self.tma_store_warp_id), || {
            // Wait for the Trans warps to complete (consumer wait on store_mbar)
            mbarrier_wait(&store_mbar_ptr, 0);

            let g_dst_cta = local_tile(tma_tensor_dst, (self.tile_m, self.tile_n), (.., ..));
            let (t_b_s_b, t_b_g_b) = cpasync::tma_partition(
                tma_atom_store,
                0,
                &make_layout(1, 0),
                &s_b.group_modes(0, 2),
                &g_dst_cta.group_modes(0, 2),
            );
            copy(tma_atom_store, &t_b_s_b.slice((.., 0)), &t_b_g_b.slice((.., bidx, bidy)));
            // cute-rs: wait until the store has read shared memory before the CTA exits.
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, true);
        });
    }

    /// The launch for an `(m, n)` source: grid `(ceil(m / tile_m), ceil(n /
    /// tile_n))`, six warps per CTA.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let grid = [m.div_ceil(self.tile_m) as u32, n.div_ceil(self.tile_n) as u32, 1];
        LaunchConfig::new(grid, [self.threads_per_cta, 1, 1])
    }
}

/// The tutorial's default problem (`--M 128 --N 128`).
pub const DEFAULT_M: usize = 128;
pub const DEFAULT_N: usize = 128;

/// The arguments for an `(m, n)` source: `src` filled with distinct Float16
/// values, `dst` (`(n, m)`) with zeros.
pub fn args(m: usize, n: usize) -> Vec<KernelArg> {
    let a: Vec<F16> = (0..m * n).map(|i| F16::from_f32(((i * 53) % 2011) as f32 - 1005.0)).collect();
    vec![
        KernelArg::ScalarI32(m as i32),
        KernelArg::ScalarI32(n as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&vec![F16::from_f32(0.0); m * n]),
    ]
}

/// `run_transpose`'s verification, bit for bit: `dst` is `src` transposed.
pub fn check(args: &[KernelArg]) {
    let (m, n) = match (&args[0], &args[1]) {
        (KernelArg::ScalarI32(m), KernelArg::ScalarI32(n)) => (*m as usize, *n as usize),
        _ => unreachable!("m and n are the first arguments"),
    };
    let (a, b) = (args[2].to_vec::<F16>(), args[3].to_vec::<F16>());
    for i in 0..m {
        for j in 0..n {
            assert_eq!(b[j * m + i].0, a[i * n + j].0, "dst[{j}, {i}] against src[{i}, {j}]");
        }
    }
}

/// Simulate the kernel on an `(m, n)` source under every built-in schedule
/// (and the seeded ones given) and check it.
pub fn run(m: usize, n: usize, seeds: &[u64]) -> SimReport {
    let transpose_kernel = Sm100MatrixTransposeKernelV1::new();
    let kernel = transpose_kernel.build();
    let mut args = args(m, n);
    let report = kernel
        .simulate_all_orders(transpose_kernel.launch(m, n), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    if !seeds.is_empty() {
        let mut args = self::args(m, n);
        kernel
            .simulate_schedules(transpose_kernel.launch(m, n), &mut args, &SimOptions::default(), seeds)
            .unwrap_or_else(|e| panic!("{e}"));
        check(&args);
    }
    report
}

fn main() {
    for (m, n) in [(DEFAULT_M, DEFAULT_N), (256, 128), (144, 208)] {
        let report = run(m, n, &[]);
        println!("Matrix size: {m}x{n}");
        println!("Tile shape: {:?}", Sm100MatrixTransposeKernelV1::new().tile_shape);
        println!("TMA instructions: {}, bytes: {}", report.tma_instructions, report.tma_bytes);
        println!("Verification: PASSED (simulated)");
    }
}

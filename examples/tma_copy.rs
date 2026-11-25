// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py
//   (the kernel `Sm100SimpleCopyKernel`, its comments and diagrams, and the
//   problem sizes of `run_tma_copy`),
//     Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! TMA V0: understanding `tma_partition`, the foundation of TMA operations,
//! ported from CuTe DSL's TMA tutorial `tma_v0.py`.
//!
//! The tutorial demonstrates TMA (Tensor Memory Accelerator) operations
//! through a simple copy kernel, focusing on the `tma_partition` interface:
//!
//! 1. TMA load (global memory to shared memory);
//! 2. TMA store (shared memory to global memory);
//! 3. barrier synchronisation for TMA (`elect_one`, `mbarrier_init`,
//!    `mbarrier_arrive`, `mbarrier_wait`);
//! 4. what `tma_partition` does, with diagrams (in [`Sm100SimpleCopyKernel::kernel`]).
//!
//! Key points: `tma_partition` transforms tensors for TMA operations;
//! `group_modes` groups tensor modes to define the TMA atom's shape;
//! indexing selects a CTA's tile from the partitioned tensors.
//!
//! The port keeps the kernel's structure and names. CuTe DSL's
//! `@cute.struct` shared storage becomes `alloc_smem` for the barrier and
//! `make_smem_tensor_aligned` for the 1024-byte-aligned tile, and
//! `product_each(smem_layout.shape)` is the tile shape itself (the layout is
//! plain). The kernel ends
//! with a commit and wait of the TMA store's bulk async-group, which the
//! Python kernel leaves to the end of the CTA. `M` and `N` are launch values
//! (the tutorial marks the layout dynamic, `N` divisible by 16).
//!
//! The tutorial is written for Blackwell, but every instruction it uses is
//! SM90's: it compiles for sm_90. No SM90 GPU is available to cute-rs's
//! tests, so it is verified by the simulator (bit for bit against the input)
//! and by `ptxas`; `main` simulates the tutorial's default problem and two
//! more.
//!
//! Run with `cargo run --release --example tma_copy`.

use cute::cpasync;
use cute::prelude::*;

/// The tutorial's kernel and its configuration (CuTe DSL's
/// `Sm100SimpleCopyKernel`).
#[derive(Clone, Copy, Debug)]
pub struct Sm100SimpleCopyKernel {
    pub tile_shape: (usize, usize),
    pub tile_m: usize,
    pub tile_n: usize,
    pub cluster_shape_mn: (usize, usize),
    pub threads_per_cta: u32,
    pub buffer_align_bytes: usize,
}

impl Default for Sm100SimpleCopyKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Sm100SimpleCopyKernel {
    /// The configuration of a TMA copy kernel: 128x128 tiles, one warp.
    pub fn new() -> Sm100SimpleCopyKernel {
        let tile_shape = (128, 128);
        Sm100SimpleCopyKernel {
            tile_shape,
            tile_m: tile_shape.0,
            tile_n: tile_shape.1,
            cluster_shape_mn: (1, 1),
            threads_per_cta: 32,
            buffer_align_bytes: 1024,
        }
    }

    /// The kernel (CuTe DSL's `__call__`, which makes the atoms, and
    /// `kernel`): parameters `m`, `n` (launch values, `n` a multiple of 16),
    /// then `src` and `dst`, row-major `(m, n)` Float16 tensors.
    pub fn build(&self) -> Kernel {
        let cfg = *self;
        Kernel::builder("tma_copy").block_dim([cfg.threads_per_cta, 1, 1]).target(Arch::Sm90).build(move || {
            let m = param_extent("m", 1);
            let n = param_extent("n", 16);
            let src = param_tensor_aligned::<F16>("src", Layout::new((m, n), (n, 1)), 16);
            let dst = param_tensor_aligned::<F16>("dst", Layout::new((m, n), (n, 1)), 16);

            // layout for each cta: (tile_m, tile_n):(tile_n, 1)
            let smem_layout = Layout::new((cfg.tile_m, cfg.tile_n), (cfg.tile_n, 1));
            let num_tma_load_bytes =
                size_in_bytes::<F16>(&smem_layout).as_static().expect("the shared-memory layout is static");

            // cta_tiler: the per-CTA tile extents (M, N) used by TMA. smem_layout may be
            // swizzled or composed; here its shape is the tile, (tile_m, tile_n).
            let cta_tiler = (cfg.tile_m, cfg.tile_n);

            let (tma_atom_src, tma_tensor_src) = cpasync::make_tiled_tma_atom(
                cpasync::CopyBulkTensorTileG2SOp::default(),
                &src,
                smem_layout.clone(),
                cta_tiler,
                1,
            )
            .into();
            let (tma_atom_dst, tma_tensor_dst) =
                cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileS2GOp, &dst, smem_layout.clone(), cta_tiler, 1)
                    .into();
            cfg.kernel(
                &tma_atom_src,
                &tma_tensor_src,
                &tma_atom_dst,
                &tma_tensor_dst,
                &smem_layout,
                num_tma_load_bytes,
            );
        })
    }

    /// The kernel body (CuTe DSL's `kernel`).
    fn kernel(
        &self,
        tma_atom_src: &CopyAtom<F16>,
        tma_tensor_src: &Tensor<Ident>,
        tma_atom_dst: &CopyAtom<F16>,
        tma_tensor_dst: &Tensor<Ident>,
        smem_layout: &Layout,
        num_tma_load_bytes: usize,
    ) {
        let (bidx, bidy) = (block_idx().x, block_idx().y);

        // Allocate shared memory: the barrier and the tile.
        let barrier_ptr = alloc_smem::<u64>(1, 8);
        let smem_tensor = make_smem_tensor_aligned::<F16>(smem_layout.clone(), self.buffer_align_bytes);

        // Initialize the barrier: elect_one ensures only one thread executes this.
        // Note: we use elect_one() instead of "if tid == 0" because it picks exactly one
        // thread among those that are actually executing, with the warp converged.
        if_(elect_one(), || {
            mbarrier_init(&barrier_ptr, 1);
            mbarrier_expect_tx(&barrier_ptr, num_tma_load_bytes as i32);
        });

        // Fence ensures init/expect_tx are visible before proceeding
        mbarrier_init_fence();
        sync_threads();

        // Tile the (M, N) tensor: ((TileM, TileN), M/TileM, N/TileN)
        let g_src_tiled = local_tile(tma_tensor_src, (self.tile_m, self.tile_n), (.., ..));
        let g_dst_tiled = local_tile(tma_tensor_dst, (self.tile_m, self.tile_n), (.., ..));

        // ======================================================================
        // TMA Partition: Tensor Preparation for TMA Operations
        // ======================================================================
        //
        // tma_partition prepares tensors for TMA copy by partitioning them
        // according to the TMA atom's internal layout requirements.
        //
        // Signature:
        //   tma_partition(atom, cta_id, cta_layout, smem_tensor, gmem_tensor)
        //     -> (smem_view, gmem_view)
        //
        // Key Requirement: Mode 0 of both tensors must represent the TMA atom
        //
        // Example: M=512, N=128, TileM=128, TileN=64
        //
        // Input Tensors:
        //   gSrc_tiled:  (128, 64, 4, 2)      # 4 separate modes
        //                 └──┬──┘ └──┬──┘
        //                   Tile    Grid
        //
        //   smem_tensor: (128, 64)            # 2 separate modes
        //                 └──┬──┘
        //                   Tile
        //
        // Apply group_modes(tensor, 0, 2) to group first 2 modes:
        //
        //   group_modes(gSrc_tiled, 0, 2)   =>  ((128, 64), 4, 2)
        //                                         └───┬───┘
        //                                         Mode 0 = Atom
        //
        //   group_modes(smem_tensor, 0, 2)  =>  ((128, 64),)
        //                                         └───┬───┘
        //                                         Mode 0 = Atom
        //
        // After tma_partition:
        //
        //   tAsA: SMEM view with TMA internal layout
        //     Shape: ((TMA_Layout),)
        //     - TMA_Layout: Swizzled/banked layout for efficient SMEM access
        //
        //   tAgA: Global view preserving rest modes
        //     Shape: ((TMA_Layout), 4, 2)
        //            └─────┬─────┘ └──┬──┘
        //            TMA atom      Rest modes (grid)
        //
        // Usage Pattern:
        //   1. Group modes to define atom: group_modes(tensor, 0, 2)
        //   2. Call tma_partition: tAsA, tAgA = tma_partition(...)
        //   3. Select tile for CTA: tAgA_cta = tAgA[(None, bidx, bidy)]
        //      - None: keep entire atom
        //      - bidx, bidy: index into rest modes
        //   4. Issue TMA copy: cute.copy(atom, tAgA_cta, tAsA)
        //
        // Visual Data Flow:
        //
        //   Global Memory (512x128)           Shared Memory (128x64)
        //   ┌────────────────────┐            ┌──────────────┐
        //   │  ┌───┬───┐         │            │              │
        //   │  │0,0│0,1│         │            │  smem_tensor │
        //   │  ├───┼───┤         │            │  (128, 64)   │
        //   │  │1,0│1,1│ 4x2     │            │              │
        //   │  ├───┼───┤ tiles   │            └──────────────┘
        //   │  │2,0│2,1│         │                    │
        //   │  ├───┼───┤         │                    │ group_modes
        //   │  │3,0│3,1│         │                    ↓
        //   │  └───┴───┘         │            ((128, 64),)
        //   └────────────────────┘                    │
        //            │                                 │
        //            │ gSrc_tiled                      │
        //            │ (128, 64, 4, 2)                │
        //            ↓                                 │
        //     group_modes(_, 0, 2)                    │
        //            ↓                                 │
        //     ((128, 64), 4, 2)                       │
        //            │                                 │
        //            └────────┬────────────────────────┘
        //                     ↓
        //              tma_partition
        //                     ↓
        //      ┌──────────────┴──────────────┐
        //      │                             │
        //   tAgA                          tAsA
        //   ((TMA_Layout), 4, 2)         ((TMA_Layout),)
        //      │
        //      ↓ tAgA[(None, bidx, bidy)]
        //   tAgA_cta
        //   ((TMA_Layout),)
        //
        // ======================================================================

        // TMA Load partition
        // Here we only use a 1x1 cluster, so cta_id is 0 and cta_layout is (1).
        // Note: smem and gmem must have the same size (the atom's elements) in the first mode.
        let (t_a_s_a, t_a_g_a) = cpasync::tma_partition(
            tma_atom_src,
            0,
            &make_layout(1, 0),
            &smem_tensor.group_modes(0, 2),
            &g_src_tiled.group_modes(0, 2),
        );

        // TMA Store partition
        // Same process as TMA Load, but for the destination tensor:
        // partitions gDst_tiled and smem_tensor according to the TMA Store atom.
        let (_, t_b_g_b) = cpasync::tma_partition(
            tma_atom_dst,
            0,
            &make_layout(1, 0),
            &smem_tensor.group_modes(0, 2),
            &g_dst_tiled.group_modes(0, 2),
        );

        // Select the tile of this CTA from the partitioned global views
        // Input:  tAgA with shape ((TMA_Layout), M/TileM, N/TileN)
        // Output: tAgA_cta with shape ((TMA_Layout),)
        // The (None, bidx, bidy) indexing:
        //   - None: keeps the entire TMA atom layout (mode 0)
        //   - bidx: selects from rest mode 1 (M dimension grid)
        //   - bidy: selects from rest mode 2 (N dimension grid)
        let t_a_g_a_cta = t_a_g_a.slice((.., bidx, bidy));
        let t_b_g_b_cta = t_b_g_b.slice((.., bidx, bidy));

        // ---------- TMA Load: Global -> Shared ----------
        copy_with(
            tma_atom_src,
            &t_a_g_a_cta, // Source (TMA Tensor View)
            &t_a_s_a,     // Dest (SMEM Tensor View)
            CopyArgs::new().tma_bar_ptr(&barrier_ptr),
        );

        // Signal arrival on the barrier after TMA is issued
        if_(elect_one(), || mbarrier_arrive(&barrier_ptr));

        // Wait for TMA to complete
        mbarrier_wait(&barrier_ptr, 0);

        // ---------- TMA Store: Shared -> Global ----------
        copy(
            tma_atom_dst,
            &t_a_s_a,     // Source (SMEM Tensor View)
            &t_b_g_b_cta, // Dest (Global Tensor View)
        );
        // cute-rs: wait until the store has read shared memory before the CTA exits.
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group(0, true);
    }

    /// The launch for an `(m, n)` problem: grid `(ceil(m / tile_m),
    /// ceil(n / tile_n))`, one warp per CTA.
    pub fn launch(&self, m: usize, n: usize) -> LaunchConfig {
        let grid = [m.div_ceil(self.tile_m) as u32, n.div_ceil(self.tile_n) as u32, 1];
        LaunchConfig::new(grid, [self.threads_per_cta, 1, 1])
    }
}

/// The tutorial's default problem (`--M 512 --N 128`).
pub const DEFAULT_M: usize = 512;
pub const DEFAULT_N: usize = 128;

/// The arguments for an `(m, n)` problem: `src` filled with distinct
/// Float16 values, `dst` with zeros.
pub fn args(m: usize, n: usize) -> Vec<KernelArg> {
    let a: Vec<F16> = (0..m * n).map(|i| F16::from_f32(((i * 37) % 2003) as f32 - 1001.0)).collect();
    vec![
        KernelArg::ScalarI32(m as i32),
        KernelArg::ScalarI32(n as i32),
        KernelArg::buffer(&a),
        KernelArg::buffer(&vec![F16::from_f32(0.0); m * n]),
    ]
}

/// `run_tma_copy`'s verification, bit for bit: `dst` is `src`.
pub fn check(args: &[KernelArg]) {
    let (a, b) = (args[2].to_vec::<F16>(), args[3].to_vec::<F16>());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.0, y.0, "element {i}");
    }
}

/// Simulate the kernel on an `(m, n)` problem under every built-in schedule
/// and check it.
pub fn run(m: usize, n: usize) -> SimReport {
    let copy_kernel = Sm100SimpleCopyKernel::new();
    let mut args = args(m, n);
    let report = copy_kernel
        .build()
        .simulate_all_orders(copy_kernel.launch(m, n), &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    check(&args);
    report
}

#[allow(dead_code)]
fn main() {
    for (m, n) in [(DEFAULT_M, DEFAULT_N), (256, 256), (200, 144)] {
        let report = run(m, n);
        println!("Matrix size: {m}x{n}");
        println!("Tile shape: {:?}", Sm100SimpleCopyKernel::new().tile_shape);
        println!("TMA instructions: {}, bytes: {}", report.tma_instructions, report.tma_bytes);
        println!("Verification: PASSED (simulated)");
    }
}

// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   examples/python/CuTeDSL/experimental/primitives/tma/tma_tensormap_replace.py (the
//   descriptor-replacement kernel: copy a descriptor to a global workspace, replace its
//   global address and its second extent, release, acquire, TMA-load the box),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! In-kernel tensor-map copies and updates (roadmap part m7-tensormap-update),
//! held to plain Rust: a descriptor copied from a TMA atom to a global workspace,
//! pointed at a later row of a second tensor and shrunk to fewer rows (the rows past
//! them zero-filled), then used by a TMA load; and a grouped copy in which each CTA
//! rewrites its group's load and store descriptors with `TensorMapManager` (in global
//! memory, and through shared memory) for groups of different shapes, then TMA-loads
//! and TMA-stores every tile of its group. They need sm_90a (`tensormap.replace`) and
//! run in the simulator.

use cute::cpasync::{self, TensormapField};
use cute::prelude::*;
use cute::utils::{TensorMapManager, TensorMapUpdateMode};

use super::Case;

/// Rows and columns of the replace kernel's tensors, and its box.
const ROWS: usize = 128;
const COLS: usize = 64;
const BOX_M: usize = 16;
const BOX_K: usize = 32;

/// The scalar parameters' value in the tests' inputs (`inputs_with_len`): the row
/// offset and the active rows.
const SCALAR: usize = 13;

/// After `tma_tensormap_replace.py`: copy the TMA atom's descriptor (made for `a`) to
/// the global workspace `desc`, replace its global address with row `row_offset` of
/// `b` and its extent along M (TMA dimension 1) with `active_rows`, release, acquire,
/// and TMA-load the first box through it. Rows past `active_rows` are zero-filled.
pub fn tensormap_replace_load() -> Case {
    let kernel = Kernel::builder("tensormap_replace_load").block_dim([32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((ROWS, COLS), (COLS, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((ROWS, COLS), (COLS, 1)), 16);
        let out = param_tensor::<F16>("out", Layout::new((BOX_M, BOX_K), (BOX_K, 1)));
        let desc = param_tensor_aligned::<u64>("desc", Layout::col_major(16), 128);
        let row_offset = param_scalar::<i32>("row_offset");
        let active_rows = param_scalar::<i32>("active_rows");
        let smem_layout = Layout::new((BOX_M, BOX_K), (BOX_K, 1));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(g2s, &a, smem_layout.clone(), (BOX_M, BOX_K), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        if_(elect_one(), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let desc_ptr = desc.ptr();
        if_(elect_one(), || {
            cpasync::copy_tensormap(&atom, &desc_ptr);
            let row_base = b.ptr() + IndexExpr::from(row_offset * COLS as i32);
            // With the new address, the box's coordinates are relative to the row offset.
            cpasync::tensormap_replace(&desc_ptr, TensormapField::GlobalAddress, row_base.toint_i64());
            // With the smaller extent, TMA zero-fills the rows past the active ones.
            cpasync::tensormap_replace(&desc_ptr, TensormapField::GlobalDim(1), active_rows);
            cpasync::fence_tma_desc_release();
        });
        // The issuing lane is elected inside the copy: every lane acquires.
        cpasync::fence_tma_desc_acquire(&desc_ptr);
        let g = local_tile(&tma_tensor, (BOX_M, BOX_K), (0, 0));
        let (ts, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (BOX_M * BOX_K * 2) as i32));
        copy_with(&atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(desc_ptr.toint_i64()));
        mbarrier_wait(&mbar, 0);
        let t = thread_idx().x;
        for_range(0, (BOX_M * BOX_K / 32) as i32, 1, |j| {
            let idx = j * 32 + t;
            let (m, k) = (idx / BOX_K as i32, idx % BOX_K as i32);
            out.store((m, k), s.load((m, k)));
        });
    });
    fn check(args: &[KernelArg]) {
        let (b, out) = (args[1].to_vec::<F16>(), args[2].to_vec::<F16>());
        for m in 0..BOX_M {
            for k in 0..BOX_K {
                let want = if m < SCALAR { b[(SCALAR + m) * COLS + k].0 } else { 0 };
                assert_eq!(out[m * BOX_K + k].0, want, "element ({m}, {k})");
            }
        }
        // The workspace holds the replaced descriptor (the simulator's encoding).
        let words = args[3].to_vec::<u64>();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let d = cute::tma::TmaDesc::from_sim_bytes(&bytes).expect("a descriptor");
        assert_eq!(d.global_dims[1], SCALAR as u64);
        assert_eq!(d.global_dims[0], COLS as u64);
        assert_eq!(d.global_strides[0], (COLS * 2) as u64);
    }
    Case {
        name: "tensormap_replace_load",
        kernel,
        cfg: LaunchConfig::new([1, 1, 1], [32, 1, 1]),
        small_only: false,
        check,
    }
}

/// The groups of the grouped copy: `(rows, cols)` of each, row-major and packed one
/// after the other in one buffer. Columns are multiples of 8 (16-byte rows) and every
/// group starts 16-byte aligned; some are smaller than a tile, some cover several
/// tiles with partial edge tiles.
pub const GROUPS: [(usize, usize); 4] = [(20, 40), (16, 32), (33, 72), (7, 16)];

/// The tile of the grouped copy.
const TILE_M: usize = 16;
const TILE_K: usize = 32;

/// Elements of all the groups.
pub fn grouped_elements() -> usize {
    GROUPS.iter().map(|(m, k)| m * k).sum()
}

/// A constant of the group table, chosen by the group index `g` (a chain of selects).
fn pick(g: Int32, values: [usize; 4]) -> Int32 {
    let mut v = Int32::from(values[3] as i32);
    for i in (0..3).rev() {
        v = select(g.eq(i as i32), values[i] as i32, v);
    }
    v
}

/// One CTA per group: the CTA builds its group's input and output tensors from the
/// packed buffers, rewrites the load and store descriptors in its two workspace slots
/// with `TensorMapManager` in `mode` (from copies of the atoms' descriptors, made for
/// the template tensors `a` and `o`), acquires them, and copies each tile of the group
/// with a TMA load and a TMA store through them. The output is the input, bit for bit,
/// so the descriptors must describe each group exactly (the loads zero-fill past the
/// group's edge and the stores clip there).
fn grouped(name: &'static str, mode: TensorMapUpdateMode) -> Case {
    let kernel = Kernel::builder(name).block_dim([32, 1, 1]).build(move || {
        // Templates for the atoms: the largest group fits them, so their TMA tensors
        // have every tile coordinate a group uses.
        let a = param_tensor_aligned::<F16>("a", Layout::new((64, 128), (128, 1)), 16);
        let o = param_tensor_aligned::<F16>("o", Layout::new((64, 128), (128, 1)), 16);
        let total = grouped_elements();
        let a_all = param_tensor_aligned::<F16>("a_all", Layout::col_major(total), 16);
        let o_all = param_tensor_aligned::<F16>("o_all", Layout::col_major(total), 16);
        let workspace = param_tensor_aligned::<u64>("workspace", Layout::col_major(GROUPS.len() * 2 * 16), 128);
        let smem_layout = Layout::new((TILE_M, TILE_K), (TILE_K, 1));
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let (load_atom, tma_a) = cpasync::make_tiled_tma_atom(g2s, &a, smem_layout.clone(), (TILE_M, TILE_K), 1).into();
        let s2g = cpasync::CopyBulkTensorTileS2GOp;
        let (store_atom, tma_o) =
            cpasync::make_tiled_tma_atom(s2g, &o, smem_layout.clone(), (TILE_M, TILE_K), 1).into();
        let s = make_smem_tensor_aligned::<F16>(smem_layout, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let g = block_idx().x;
        let mut offsets = [0usize; 4];
        for i in 1..4 {
            offsets[i] = offsets[i - 1] + GROUPS[i - 1].0 * GROUPS[i - 1].1;
        }
        let m = pick(g, GROUPS.map(|(m, _)| m));
        let k = pick(g, GROUPS.map(|(_, k)| k));
        let offset = pick(g, offsets);
        let layout = Layout::new((IndexExpr::from(m), IndexExpr::from(k)), (IndexExpr::from(k), 1));
        let a_g = make_tensor(&(a_all.ptr() + IndexExpr::from(offset)), layout.clone());
        let o_g = make_tensor(&(o_all.ptr() + IndexExpr::from(offset)), layout);

        let manager = TensorMapManager::new(mode, 128);
        let slot = |i: i32| {
            manager.get_tensormap_ptr(&(workspace.ptr() + IndexExpr::from(g * 32 + i * 16)), AddressSpace::Gmem)
        };
        let (load_desc, store_desc) = (slot(0), slot(1));
        let smem_descs = alloc_smem::<u64>(2 * 16, 128);
        let (smem_load, smem_store) = (smem_descs.clone(), &smem_descs + 16);
        // The descriptors start as the atoms' own: in the shared copies (SMEM mode) or the
        // global slots (GMEM mode).
        let (init_load, init_store) = match mode {
            TensorMapUpdateMode::SMEM => (smem_load.clone(), smem_store.clone()),
            TensorMapUpdateMode::GMEM => (load_desc.clone(), store_desc.clone()),
        };
        manager.init_tensormap_from_atom(&load_atom, &init_load, 0);
        manager.init_tensormap_from_atom(&store_atom, &init_store, 0);
        manager.fence_tensormap_initialization();
        manager.update_tensormap(
            &[&a_g, &o_g],
            &[&load_atom, &store_atom],
            &[load_desc.clone(), store_desc.clone()],
            0,
            &[smem_load, smem_store],
        );
        manager.fence_tensormap_update(&load_desc);
        manager.fence_tensormap_update(&store_desc);

        if_(elect_one(), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let tiles_m = (m + (TILE_M as i32 - 1)) / TILE_M as i32;
        let tiles_k = (k + (TILE_K as i32 - 1)) / TILE_K as i32;
        for_range(0, tiles_m * tiles_k, 1, |i| {
            let (tm, tk) = (i / tiles_k, i % tiles_k);
            let ga = local_tile(&tma_a, (TILE_M, TILE_K), (tm, tk));
            let go = local_tile(&tma_o, (TILE_M, TILE_K), (tm, tk));
            let (ts, tga) =
                cpasync::tma_partition(&load_atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &ga.group_modes(0, 2));
            let (_, tgo) =
                cpasync::tma_partition(&store_atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &go.group_modes(0, 2));
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (TILE_M * TILE_K * 2) as i32));
            copy_with(&load_atom, &tga, &ts, CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(load_desc.toint_i64()));
            mbarrier_wait(&mbar, i % 2);
            copy_with(&store_atom, &ts, &tgo, CopyArgs::new().tma_desc_ptr(store_desc.toint_i64()));
            // The next load overwrites the tile: the store must have read it.
            if_(elect_one(), || {
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, true);
            });
            sync_warp();
        });
        if_(elect_one(), || cp_async_bulk_wait_group(0, false));
    });
    fn check(args: &[KernelArg]) {
        let (a, o) = (args[2].to_vec::<F16>(), args[3].to_vec::<F16>());
        for (i, (x, y)) in a.iter().zip(&o).enumerate() {
            assert_eq!(x.0, y.0, "element {i}");
        }
    }
    let cfg = LaunchConfig::new([GROUPS.len() as u32, 1, 1], [32, 1, 1]);
    Case { name, kernel, cfg, small_only: false, check }
}

/// [`grouped`] editing the descriptors in place in global memory.
pub fn tensormap_grouped_gmem() -> Case {
    grouped("tensormap_grouped_gmem", TensorMapUpdateMode::GMEM)
}

/// [`grouped`] editing copies in shared memory and publishing them with
/// `tensormap.cp_fenceproxy`.
pub fn tensormap_grouped_smem() -> Case {
    grouped("tensormap_grouped_smem", TensorMapUpdateMode::SMEM)
}

/// The kernels of this file (sm_90a: `tensormap.replace`).
pub fn tensormap_cases_sm90a() -> Vec<Case> {
    vec![tensormap_replace_load(), tensormap_grouped_gmem(), tensormap_grouped_smem()]
}

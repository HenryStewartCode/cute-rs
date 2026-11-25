// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SM100 TMA and bulk-copy forms, held to plain
//! Rust: gather4 loads of rows an index tensor names (some past the tensor, zero-filled)
//! and scatter4 stores to them, byte-masked bulk stores with several masks, FP4 and FP6
//! loads through the packed (`.b4x16`) and padded (`.b4x16_p64`, `.b6x16_p32`) formats
//! compared byte for byte with a hand-written packer, and the sm_107f override loads and
//! stores over a tensor with ragged edges. They run in the simulator; the sm_100a ones
//! compile for sm_100a, the override ones only to MLIR text (no toolchain has sm_107f).

use cute::prelude::*;

use super::Case;

/// Threads of every kernel here.
const THREADS: usize = 128;

/// Rows of the gather4/scatter4 tensors: the indices (0 to 16 on small inputs, anything
/// on random ones) past it are out of bounds.
const G_ROWS: usize = 12;
/// Columns of the gather4/scatter4 tensors (an f32 row is 128 bytes).
const G_COLS: usize = 32;
/// Rows of the gathered tile: two instructions of four rows.
const G_TILE: usize = 8;

/// A gather4 load (`CopyBulkTensor2DGather4G2SOp`): rows `idx[0..8]` of a `(12, 32)` f32
/// tensor, from an index tensor broadcast along the columns, into an `(8, 32)` tile
/// written back to `out`. Rows past the tensor (or negative) are zero-filled.
pub fn tma_gather4() -> Case {
    let kernel = Kernel::builder("tma_gather4").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<f32>("a", Layout::new((G_ROWS, G_COLS), (G_COLS, 1)), 16);
        let idx = param_tensor::<i32>("idx", Layout::new((G_ROWS, G_COLS), (1, 0)));
        let out = param_tensor::<f32>("out", Layout::new((G_TILE, G_COLS), (G_COLS, 1)));
        let smem = Layout::new((G_TILE, G_COLS), (G_COLS, 1));
        let opts = cpasync::TmaAtomOptions::new().gmem_coord_tensor(&idx);
        let op = cpasync::CopyBulkTensor2DGather4G2SOp::default();
        let (atom, tma_tensor) =
            cpasync::make_tiled_tma_atom_with(op, &a, smem.clone(), (G_TILE, G_COLS), 1, &opts).into();
        let s = make_smem_tensor_aligned::<f32>(smem, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let g = local_tile(&tma_tensor, (G_TILE, G_COLS), (0, 0));
        let gi = local_tile(&idx, (G_TILE, G_COLS), (0, 0));
        let (ts, tg, ti) = cpasync::tma_partition_indexed(
            &atom,
            0,
            &make_layout(1, 0),
            &s.group_modes(0, 2),
            &g.group_modes(0, 2),
            &gi.group_modes(0, 2),
        );
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, (G_TILE * G_COLS * 4) as i32));
            copy_aux(&atom, (&tg, &[(&ti).into()]), (&ts, &[]), CopyArgs::new().tma_bar_ptr(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, (G_TILE * G_COLS / THREADS) as i32, 1, |j| {
            let i = j * THREADS as i32 + t;
            let (r, c) = (i / G_COLS as i32, i % G_COLS as i32);
            out.store((r, c), s.load((r, c)));
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, idx, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<i32>(), args[2].to_vec::<f32>());
        for r in 0..G_TILE {
            for c in 0..G_COLS {
                let row = idx[r];
                let want = if (0..G_ROWS as i32).contains(&row) { a[row as usize * G_COLS + c] } else { 0.0 };
                assert_eq!(out[r * G_COLS + c].to_bits(), want.to_bits(), "row {r} (index {row}), column {c}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_gather4", kernel, cfg, small_only: false, check }
}

/// The value the scatter4 kernel writes to row `row`, column `c` (the same from every
/// tile row that names `row`, so repeated indices agree).
fn scatter_value(row: i32, c: i32) -> f32 {
    (row * 7 + c * 3 % 41) as f32
}

/// The rows the scatter4 kernel writes to: distinct, two past the tensor's end. (None is
/// negative: a store takes no negative coordinate, except scatter4 on sm_107f, and a B200
/// stops with an illegal-instruction error on one.)
const SCATTER_ROWS: [i32; G_TILE] = [3, 13, 0, 7, 11, 12, 5, 9];

/// A scatter4 store (`CopyBulkTensor2DScatter4S2GOp`): an `(8, 32)` tile whose row `r`
/// holds [`scatter_value`] of row `idx[r]` (the kernel writes [`SCATTER_ROWS`] there),
/// stored to rows `idx[0..8]` of a `(12, 32)` f32 tensor; rows past it are skipped.
pub fn tma_scatter4() -> Case {
    let kernel = Kernel::builder("tma_scatter4").block_dim([THREADS as u32, 1, 1]).build(|| {
        let out = param_tensor_aligned::<f32>("out", Layout::new((G_ROWS, G_COLS), (G_COLS, 1)), 16);
        let idx = param_tensor::<i32>("idx", Layout::new((G_ROWS, G_COLS), (1, 0)));
        let smem = Layout::new((G_TILE, G_COLS), (G_COLS, 1));
        let opts = cpasync::TmaAtomOptions::new().gmem_coord_tensor(&idx);
        let op = cpasync::CopyBulkTensor2DScatter4S2GOp;
        let (atom, tma_tensor) =
            cpasync::make_tiled_tma_atom_with(op, &out, smem.clone(), (G_TILE, G_COLS), 1, &opts).into();
        let s = make_smem_tensor_aligned::<f32>(smem, 128);
        let t = thread_idx().x;
        // Distinct rows (two past the tensor), written over the index argument.
        if_(t.eq(0), || {
            for (r, &v) in SCATTER_ROWS.iter().enumerate() {
                idx.store((r as i32, 0), v);
            }
        });
        sync_threads();
        for_range(0, (G_TILE * G_COLS / THREADS) as i32, 1, |j| {
            let i = j * THREADS as i32 + t;
            let (r, c) = (i / G_COLS as i32, i % G_COLS as i32);
            let row = idx.load((r, 0));
            s.store((r, c), (row * 7 + c * 3 % 41).to::<f32>());
        });
        fence_view_async_shared();
        sync_threads();
        let g = local_tile(&tma_tensor, (G_TILE, G_COLS), (0, 0));
        let gi = local_tile(&idx, (G_TILE, G_COLS), (0, 0));
        let (ts, tg, ti) = cpasync::tma_partition_indexed(
            &atom,
            0,
            &make_layout(1, 0),
            &s.group_modes(0, 2),
            &g.group_modes(0, 2),
            &gi.group_modes(0, 2),
        );
        if_(warp_idx().eq(0), || {
            copy_aux(&atom, (&ts, &[]), (&tg, &[(&ti).into()]), CopyArgs::new());
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    });
    fn check(args: &[KernelArg]) {
        let (out, idx) = (args[0].to_vec::<f32>(), args[1].to_vec::<i32>());
        for &row in &idx[..G_TILE] {
            if !(0..G_ROWS as i32).contains(&row) {
                continue;
            }
            for c in 0..G_COLS {
                assert_eq!(out[row as usize * G_COLS + c], scatter_value(row, c as i32), "row {row}, column {c}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_scatter4", kernel, cfg, small_only: false, check }
}

/// The byte masks of [`bulk_store_byte_mask`].
pub const BYTE_MASKS: [u16; 4] = [0xffff, 0x00ff, 0xa5c3, 0x0001];

/// Bytes per masked store.
const MASK_BYTES: usize = 64;

/// Byte-masked bulk stores (`CopyBulkS2GByteMaskOp`): for each mask of [`BYTE_MASKS`],
/// 64 bytes of `x` go to `out` with a plain bulk store, then 64 bytes of `y` with the
/// mask, which writes only the bytes whose bit (byte index mod 16) is set.
pub fn bulk_store_byte_mask() -> Case {
    let kernel = Kernel::builder("bulk_store_byte_mask").block_dim([THREADS as u32, 1, 1]).build(|| {
        let n = MASK_BYTES * BYTE_MASKS.len();
        let x = param_tensor::<u8>("x", Layout::col_major(n));
        let y = param_tensor::<u8>("y", Layout::col_major(n));
        let out = param_tensor_aligned::<u8>("out", Layout::col_major(n), 16);
        let sx = make_smem_tensor_aligned::<u8>(Layout::col_major(n), 128);
        let sy = make_smem_tensor_aligned::<u8>(Layout::col_major(n), 128);
        let t = thread_idx().x;
        for_range(0, (n / THREADS) as i32, 1, |j| {
            let i = j * THREADS as i32 + t;
            sx.store(i, x.load(i));
            sy.store(i, y.load(i));
        });
        fence_view_async_shared();
        sync_threads();
        let plain = make_copy_atom::<u8>(cpasync::CopyBulkS2GOp);
        let masked = make_copy_atom::<u8>(cpasync::CopyBulkS2GByteMaskOp);
        if_(t.eq(0), || {
            for (k, &mask) in BYTE_MASKS.iter().enumerate() {
                let tile = (k * MASK_BYTES) as i32;
                let part = |v: &Tensor<u8>| local_tile(v, MASK_BYTES, tile / MASK_BYTES as i32);
                copy(&plain, &part(&sx), &part(&out));
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, false);
                copy_with(&masked, &part(&sy), &part(&out), CopyArgs::new().byte_mask(mask as i16));
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group(0, false);
            }
        });
    });
    fn check(args: &[KernelArg]) {
        let (x, y, out) = (args[0].to_vec::<u8>(), args[1].to_vec::<u8>(), args[2].to_vec::<u8>());
        for (k, &mask) in BYTE_MASKS.iter().enumerate() {
            for b in 0..MASK_BYTES {
                let i = k * MASK_BYTES + b;
                let want = if mask >> (b % 16) & 1 == 1 { y[i] } else { x[i] };
                assert_eq!(out[i], want, "mask {mask:#06x}, byte {b}");
            }
        }
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "bulk_store_byte_mask", kernel, cfg, small_only: false, check }
}

/// Pack 4-bit values (the low nibble of each byte) two per byte, the first in the low
/// bits: global memory's FP4 layout, and the `.b4x16` format's shared-memory image.
pub fn pack_fp4(v: &[u8]) -> Vec<u8> {
    v.chunks(2).map(|p| (p[0] & 0xf) | (p.get(1).copied().unwrap_or(0) & 0xf) << 4).collect()
}

/// Pack 6-bit values four per three bytes, the first in the low bits.
pub fn pack_fp6(v: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; (v.len() * 6).div_ceil(8)];
    for (i, &x) in v.iter().enumerate() {
        for k in 0..6 {
            let bit = 6 * i + k;
            out[bit / 8] |= ((x >> k) & 1) << (bit % 8);
        }
    }
    out
}

/// The padded image of `.b4x16_p64` (`bits` 4) or `.b6x16_p32` (6): each group of 16
/// values packed into the start of a 16-byte slot, the rest of the slot left as it was
/// (zeros here, as the kernels clear shared memory first).
pub fn pad_groups(v: &[u8], bits: usize) -> Vec<u8> {
    v.chunks(16)
        .flat_map(|g| {
            let mut slot = if bits == 4 { pack_fp4(g) } else { pack_fp6(g) };
            slot.resize(16, 0);
            slot
        })
        .collect()
}

/// Rows of the FP4/FP6 tensors.
const SUB_ROWS: usize = 4;
/// Values per row: one 128-value box row.
const SUB_COLS: usize = 128;

/// The raw bits of the sub-byte values `args[0]` holds (one byte each on the host).
fn raw<E: Copy>(v: &[E]) -> Vec<u8> {
    v.iter()
        .map(|x| {
            // SAFETY: the narrow float types are one byte on the host.
            unsafe { *(x as *const E as *const u8) }
        })
        .collect()
}

/// An FP4 tensor of `(4, 128)` loaded by one TMA load through the packed `.b4x16` format
/// (`f416u4_align8b`, the default for FP4), its shared-memory bytes copied to `out`.
pub fn tma_fp4_packed() -> Case {
    let kernel = Kernel::builder("tma_fp4_packed").block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<F4E2M1FN>("a", Layout::new((SUB_ROWS, SUB_COLS), (SUB_COLS, 1)), 32);
        let out = param_tensor::<u8>("out", Layout::col_major(SUB_ROWS * SUB_COLS / 2));
        let smem = Layout::new((SUB_ROWS, SUB_COLS), (SUB_COLS, 1));
        let op = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) = cpasync::make_tiled_tma_atom(op, &a, smem.clone(), (SUB_ROWS, SUB_COLS), 1).into();
        // The box's 256 bytes as bytes: a u8 view of the same shared memory.
        let bytes = make_smem_tensor_aligned::<u8>(Layout::col_major(SUB_ROWS * SUB_COLS / 2), 128);
        let s = make_tensor(recast_ptr::<F4E2M1FN, u8>(&bytes.ptr(), None), smem);
        load_and_dump(&atom, &tma_tensor, &s, &bytes, &out, SUB_ROWS * SUB_COLS / 2);
    });
    fn check(args: &[KernelArg]) {
        let (a, out) = (raw(&args[0].to_vec::<F4E2M1FN>()), args[1].to_vec::<u8>());
        assert_eq!(out, pack_fp4(&a));
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_fp4_packed", kernel, cfg, small_only: false, check }
}

/// Load the one box of `tma_tensor` into `s` and copy the shared-memory bytes `bytes`
/// (`n` of them, cleared first) to `out`.
#[track_caller]
fn load_and_dump<T: Numeric, D: CopyElem<T> + Numeric>(
    atom: &CopyAtom<T>,
    tma_tensor: &Tensor<Ident>,
    s: &Tensor<D>,
    bytes: &Tensor<u8>,
    out: &Tensor<u8>,
    n: usize,
) {
    let mbar = alloc_smem::<u64>(1, 8);
    let t = thread_idx().x;
    for_range(0, n.div_ceil(THREADS) as i32, 1, |j| {
        let i = j * THREADS as i32 + t;
        if_(i.lt(n as i32), || bytes.store(i, 0u8));
    });
    fence_view_async_shared();
    if_(t.eq(0), || mbarrier_init(&mbar, 1));
    mbarrier_init_fence();
    sync_threads();
    let g = local_tile(tma_tensor, (SUB_ROWS, SUB_COLS), (0, 0));
    let tx = atom.tma_info().unwrap().spec().box_dims.iter().map(|&b| b as usize).product::<usize>();
    let tx = tx * T::DTYPE.bits() / 8;
    let (ts, tg) = cpasync::tma_partition(atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
    if_(warp_idx().eq(0), || {
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, tx as i32));
        copy_with(atom, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar));
    });
    mbarrier_wait(&mbar, 0);
    for_range(0, n.div_ceil(THREADS) as i32, 1, |j| {
        let i = j * THREADS as i32 + t;
        if_(i.lt(n as i32), || out.store(i, bytes.load(i)));
    });
}

/// An FP4 (or FP6, `E`) tensor of `(4, 128)` loaded through the padded format of an 8-bit
/// `internal_type` (`f416u4_align16b`, `.b4x16_p64`; `f416u6_align16b`, `.b6x16_p32`),
/// into a `u8` tensor of one byte slot per value whose 512 bytes are copied to `out`.
fn unpacked<E: Numeric>(name: &'static str) -> Kernel
where
    u8: CopyElem<E>,
{
    Kernel::builder(name).block_dim([THREADS as u32, 1, 1]).build(|| {
        let a = param_tensor_aligned::<E>("a", Layout::new((SUB_ROWS, SUB_COLS), (SUB_COLS, 1)), 32);
        let out = param_tensor::<u8>("out", Layout::col_major(SUB_ROWS * SUB_COLS));
        let smem = Layout::new((SUB_ROWS, SUB_COLS), (SUB_COLS, 1));
        let opts = cpasync::TmaAtomOptions::new().internal_type(DType::U8);
        let op = cpasync::CopyBulkTensorTileG2SOp::default();
        let (atom, tma_tensor) =
            cpasync::make_tiled_tma_atom_with(op, &a, smem.clone(), (SUB_ROWS, SUB_COLS), 1, &opts).into();
        let bytes = make_smem_tensor_aligned::<u8>(smem, 128);
        let flat = make_tensor(bytes.ptr(), Layout::col_major(SUB_ROWS * SUB_COLS));
        load_and_dump(&atom, &tma_tensor, &bytes, &flat, &out, SUB_ROWS * SUB_COLS);
    })
}

pub fn tma_fp4_unpacked() -> Case {
    fn check(args: &[KernelArg]) {
        let (a, out) = (raw(&args[0].to_vec::<F4E2M1FN>()), args[1].to_vec::<u8>());
        assert_eq!(out, pad_groups(&a, 4));
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_fp4_unpacked", kernel: unpacked::<F4E2M1FN>("tma_fp4_unpacked"), cfg, small_only: false, check }
}

pub fn tma_fp6_unpacked() -> Case {
    fn check(args: &[KernelArg]) {
        let (a, out) = (raw(&args[0].to_vec::<F6E2M3FN>()), args[1].to_vec::<u8>());
        assert_eq!(out, pad_groups(&a, 6));
    }
    let cfg = LaunchConfig::new([1, 1, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_fp6_unpacked", kernel: unpacked::<F6E2M3FN>("tma_fp6_unpacked"), cfg, small_only: false, check }
}

/// The SM100 cases that compile for sm_100a.
pub fn tma_sm100_cases() -> Vec<Case> {
    vec![
        tma_gather4(),
        tma_scatter4(),
        bulk_store_byte_mask(),
        tma_fp4_packed(),
        tma_fp4_unpacked(),
        tma_fp6_unpacked(),
    ]
}

/// Rows and columns of the override kernel's tensors: tiles of 16x32 leave ragged edges.
const O_ROWS: usize = 20;
const O_COLS: usize = 40;

/// Override copies (sm_107f; simulated, and compiled to MLIR text only): each block
/// loads its `(16, 32)` tile of a `(20, 40)` f32 tensor with
/// `CopyBulkTensorTileG2SOverrideOp`, giving the tile's address, the extent left of
/// each mode (from a residue tensor) and the strides, writes it to `out` (`(32, 64)`,
/// zero past the tensor, as the load fills), and stores it to `b` with
/// `CopyBulkTensorTileS2GOverrideOp`, which clips the ragged edge.
pub fn tma_override() -> Case {
    let kernel = Kernel::builder("tma_override").block_dim([THREADS as u32, 1, 1]).target(Arch::Sm107f).build(|| {
        let a = param_tensor_aligned::<f32>("a", Layout::new((O_ROWS, O_COLS), (O_COLS, 1)), 16);
        let b = param_tensor_aligned::<f32>("b", Layout::new((O_ROWS, O_COLS), (O_COLS, 1)), 16);
        let out = param_tensor::<f32>("out", Layout::new((32, 64), (64, 1)));
        let smem = Layout::new((16, 32), (32, 1));
        let load_op = cpasync::CopyBulkTensorTileG2SOverrideOp::default();
        let (load, _) = cpasync::make_tiled_tma_atom(load_op, &a, smem.clone(), (16, 32), 1).into();
        let store_op = cpasync::CopyBulkTensorTileS2GOverrideOp::default();
        let (store, _) = cpasync::make_tiled_tma_atom(store_op, &b, smem.clone(), (16, 32), 1).into();
        let s = make_smem_tensor_aligned::<f32>(smem, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        let t = thread_idx().x;
        let (bx, by) = (block_idx().x, block_idx().y);
        // What remains of each mode: the extents, counting down.
        let residue = make_coord_tensor(
            (O_ROWS as i32, O_COLS as i32),
            Layout::new((O_ROWS, O_COLS), (E(0) * IndexExpr::from_signed(-1), E(1) * IndexExpr::from_signed(-1))),
        );
        let r = local_tile(&residue, (16, 32), (bx, by));
        let ga = local_tile(&a, (16, 32), (bx, by));
        let gb = local_tile(&b, (16, 32), (bx, by));
        if_(t.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let (ts, tga, tr) = cpasync::tma_partition_override(
            &load,
            0,
            &make_layout(1, 0),
            &s.group_modes(0, 2),
            &ga.group_modes(0, 2),
            &r.group_modes(0, 2),
        );
        let (_, tgb, _) = cpasync::tma_partition_override(
            &store,
            0,
            &make_layout(1, 0),
            &s.group_modes(0, 2),
            &gb.group_modes(0, 2),
            &r.group_modes(0, 2),
        );
        if_(warp_idx().eq(0), || {
            if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 16 * 32 * 4));
            copy_aux(&load, (&tga, &[(&tr).into()]), (&ts, &[]), CopyArgs::new().tma_bar_ptr(&mbar));
        });
        mbarrier_wait(&mbar, 0);
        for_range(0, (16 * 32 / THREADS) as i32, 1, |j| {
            let i = j * THREADS as i32 + t;
            let (m, k) = (i / 32, i % 32);
            out.store((bx * 16 + m, by * 32 + k), s.load((m, k)));
        });
        if_(warp_idx().eq(0), || {
            copy_aux(&store, (&ts, &[]), (&tgb, &[(&tr).into()]), CopyArgs::new());
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group(0, false);
        });
    });
    fn check(args: &[KernelArg]) {
        let (a, b, out) = (args[0].to_vec::<f32>(), args[1].to_vec::<f32>(), args[2].to_vec::<f32>());
        for m in 0..32 {
            for k in 0..64 {
                let want = if m < O_ROWS && k < O_COLS { a[m * O_COLS + k] } else { 0.0 };
                assert_eq!(out[m * 64 + k].to_bits(), want.to_bits(), "out ({m}, {k})");
            }
        }
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "b element {i}");
        }
    }
    let cfg = LaunchConfig::new([2, 2, 1], [THREADS as u32, 1, 1]);
    Case { name: "tma_override", kernel, cfg, small_only: false, check }
}

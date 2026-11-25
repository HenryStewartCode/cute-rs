// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cutlass/detail/sm100_blockscaled_layout.hpp
//   (Sm1xxBlockScaledBasicChunk, Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA/B) and
//   include/cutlass/detail/sm103_blockscaled_layout.hpp (Sm103BlockScaledBasicChunk),
//   include/cute/atom/mma_traits_sm100_frag.hpp (the atom and tiling of UMMA::tmem_sf_frg,
//   in make_tmem_layout_sfa/sfb),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//   examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/
//   sm103_dense_blockscaled_gemm_persistent.py (sm103_make_smem_layout_sfa/sfb),
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The global-memory layout of block-scaled MMA scale factors (CuTe DSL's
//! `cutlass.utils.BlockScaledBasicChunk`, `Sm103BlockScaledBasicChunk` and
//! `tile_atom_to_shape_SF`).
//!
//! A scale-factor atom covers 128 rows (M of A, N of B) by 4 scale factors
//! along K, each shared by `sf_vec_size` consecutive K elements: 512 bytes
//! of 8-bit scale factors, the 4 of a row adjacent (one 32-bit TMEM word).

use crate::layout::{Layout, Shape, tile_to_shape_ordered};
use crate::nvgpu::OperandMajorMode;

/// The SM100 scale-factor atom (CuTe DSL's
/// `cutlass.utils.BlockScaledBasicChunk(sf_vec_size, major_mode)`, CuTe
/// C++'s `Sm1xxBlockScaledBasicChunk`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockScaledBasicChunk {
    pub sf_vec_size: usize,
    pub major_mode: OperandMajorMode,
    layout: Layout,
}

impl BlockScaledBasicChunk {
    /// The atom for scale-factor vectors of `sf_vec_size` elements: K-major
    /// `((32,4),(sf_vec_size,4)):((16,4),(0,1))`, MN-major the two modes
    /// swapped.
    pub fn new(sf_vec_size: usize, major_mode: OperandMajorMode) -> BlockScaledBasicChunk {
        let mn = Layout::new((32, 4), (16, 4));
        let k = Layout::new((sf_vec_size, 4), (0, 1));
        let layout = match major_mode {
            OperandMajorMode::K => Layout::from_modes([mn, k]),
            OperandMajorMode::MN => Layout::from_modes([k, mn]),
        };
        BlockScaledBasicChunk { sf_vec_size, major_mode, layout }
    }

    /// The atom's layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// The SM103 scale-factor atom (CuTe DSL's
/// `cutlass.utils.Sm103BlockScaledBasicChunk`, CuTe C++'s
/// `Sm103BlockScaledBasicChunk`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sm103BlockScaledBasicChunk {
    pub sf_vec_size: usize,
    pub major_mode: OperandMajorMode,
    layout: Layout,
}

impl Sm103BlockScaledBasicChunk {
    /// K-major `((8,4,4),(sf_vec_size,4)):((16,128,4),(0,1))`, MN-major the
    /// two modes swapped.
    pub fn new(sf_vec_size: usize, major_mode: OperandMajorMode) -> Sm103BlockScaledBasicChunk {
        let mn = Layout::new((8, 4, 4), (16, 128, 4));
        let k = Layout::new((sf_vec_size, 4), (0, 1));
        let layout = match major_mode {
            OperandMajorMode::K => Layout::from_modes([mn, k]),
            OperandMajorMode::MN => Layout::from_modes([k, mn]),
        };
        Sm103BlockScaledBasicChunk { sf_vec_size, major_mode, layout }
    }

    /// The atom's layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// The scale-factor layout of an A or B operand of shape `(MN, K)` or `(MN,
/// K, L)` (CuTe DSL's `cutlass.utils.tile_atom_to_shape_SF`, CuTe C++'s
/// `tile_atom_to_shape_SFA/SFB`): the K-major [`BlockScaledBasicChunk`]
/// tiled over the shape along K first, then MN, then L. The result is
/// `((Atom_MN, Rest_MN), (Atom_K, Rest_K), L)` and maps an operand
/// element's coordinate to its scale factor's offset. MN must be a
/// multiple of 128 and K of `4 * sf_vec_size` for the layout to be exact
/// (the tiling rounds up otherwise).
#[track_caller]
pub fn tile_atom_to_shape_sf(shape: impl Into<Shape>, sf_vec_size: usize) -> Layout {
    let shape = shape.into();
    let order: &[usize] = match shape.rank() {
        2 => &[1, 0],
        3 => &[1, 0, 2],
        r => panic!("tile_atom_to_shape_SF: the shape {shape} has rank {r}; it must be (MN, K) or (MN, K, L)"),
    };
    let atom = BlockScaledBasicChunk::new(sf_vec_size, OperandMajorMode::K);
    tile_to_shape_ordered(atom.layout(), shape, order)
}

// The SM120 shared-memory scale-factor layouts.

/// CuTe DSL's `blockscaled_layout.sm120_make_smem_layout_sfa(tiled_mma,
/// tile_shape_mnk, sf_vec_size, num_stages)`: the shared-memory layout of
/// one CTA tile's A scale factors, `(((32, 4), M / 128), ((sf_vec_size,
/// nsf), 4 / nsf, K / sf_vec_size / 4), STAGE)` with strides `(((16, 4),
/// 512), ((0, 1), nsf, M / 128 * 512), stage)`, where `nsf` is the scale
/// factors per instruction (K of the tiled MMA's atom over `sf_vec_size`):
/// [`BlockScaledBasicChunk`]s of 128 rows by 4 scale factors, the K-blocks
/// of one instruction adjacent. M must be a multiple of 128 and K of `4 *
/// sf_vec_size`.
#[track_caller]
pub fn sm120_make_smem_layout_sfa(
    tiled_mma: &crate::TiledMma,
    tile_shape_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    sm120_smem_layout_sf(
        "sm120_make_smem_layout_sfa",
        tiled_mma,
        tile_shape_mnk.0,
        tile_shape_mnk.2,
        sf_vec_size,
        num_stages,
    )
}

/// CuTe DSL's `blockscaled_layout.sm120_make_smem_layout_sfb`: as
/// [`sm120_make_smem_layout_sfa`] over N.
#[track_caller]
pub fn sm120_make_smem_layout_sfb(
    tiled_mma: &crate::TiledMma,
    tile_shape_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    sm120_smem_layout_sf(
        "sm120_make_smem_layout_sfb",
        tiled_mma,
        tile_shape_mnk.1,
        tile_shape_mnk.2,
        sf_vec_size,
        num_stages,
    )
}

#[track_caller]
fn sm120_smem_layout_sf(
    what: &str,
    tiled_mma: &crate::TiledMma,
    mn: usize,
    k: usize,
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    assert!(sf_vec_size == 16 || sf_vec_size == 32, "cute: {what}: 'sf_vec_size' is {sf_vec_size}, not 16 or 32");
    let nsf = tiled_mma.atom().shape_mnk()[2] / sf_vec_size;
    assert!(nsf >= 1 && 4 % nsf == 0, "cute: {what}: the atom's K is not 1, 2 or 4 scale-factor vectors");
    assert!(mn.is_multiple_of(128), "cute: {what}: the tile's M or N extent {mn} is not a multiple of 128");
    assert!(
        k.is_multiple_of(4 * sf_vec_size),
        "cute: {what}: the tile's K extent {k} is not a multiple of 4 scale-factor vectors ({})",
        4 * sf_vec_size
    );
    let blocks = mn / 128;
    let m_mode = Layout::new(((32, 4), blocks), ((16, 4), 512));
    let k_mode = Layout::new(((sf_vec_size, nsf), 4 / nsf, k / sf_vec_size / 4), ((0, 1), nsf, blocks * 512));
    let tile = Layout::from_modes([m_mode.clone(), k_mode.clone()]);
    let stage = tile.filter_zeros().cosize();
    Layout::from_modes([m_mode, k_mode, Layout::new(num_stages, stage)])
}

// The SM100 shared-memory and tensor-memory scale-factor layouts of
// a tcgen05 block-scaled MMA (CuTe DSL's `blockscaled_layout.make_smem_layout_sf`,
// `make_smem_layout_sfa/sfb`, `make_tmem_layout_sfa/sfb` and
// `sm103_make_smem_layout_sfa/sfb`).
//
// The shared-memory layouts are BlockScaledBasicChunks (512 bytes: 128 rows of 4 scale
// factors) tiled over the CTA's tile K-chunk first, divided into one MMA instruction's
// tile, `(((128, MN_inst / 128), (sf_vec_size, nsf)), MMA_MN, MMA_K)`, plus a stage
// mode. A Cp4x32x128bOp copy moves one chunk into TMEM (32 lanes of 16 bytes, written to
// all four 32-lane sub-partitions).
//
// The tensor-memory layout. CuTe DSL computes it in a closed-source MLIR op, so it is
// rederived here from CuTe C++'s `UMMA::tmem_sf_frg` (include/cute/atom/
// mma_traits_sm100_frag.hpp) and the PTX ISA's tcgen05 scale-factor layouts (9.7.16.10.7):
// - tmem_sf_frg's atom is `((((32, MN/32), 4), (VS, NSF))` with logical strides
//   `((((1, 512), 32), (0, 128)))`, composed with `(128, 16384):(DP, 1)` (data path lane,
//   then 8-bit column element). Row `32 c + l` thus lands in lane `l` (and its replicas
//   `l + 32 p` of the four sub-partitions, the `4` mode), byte `4 c + s` for factor `s`:
//   32-bit column `c`, byte `s`, the four factors of a row in one column (the PTX
//   ISA's "scale factor A/B" figures for `.scale_vec::4X`, and CUTLASS's
//   `sm100_blockscaled_layout`'s 128 x 4 chunk).
// - The MMA_MN and MMA_K modes are tmem_sf_frg's `tiled_product` of the atom: each tile
//   starts one atom cosize later, `4 MN/32 + NSF - 4` bytes in the composed space (16
//   bytes, four columns, for MN = 128 and NSF = 4); MMA_MN before MMA_K.
// cute-rs element offsets of an 8-bit TMEM tensor put a lane at `1 << 18` (65536
// columns of 4 bytes). The derivation is checked by simulation: the block-scaled MMA of
// the simulator reads each factor where `nvgpu::tcgen05::mma_blockscaled::sf_place`
// (written independently from the PTX ISA) says, and `examples/blackwell_nvfp4_gemm.rs`
// gives exact results only when these layouts agree with it.

use crate::layout::Tiler;

/// The TMEM lane stride of an 8-bit element offset.
const SF_TMEM_LANE: usize = 1 << 18;

/// CuTe DSL's `blockscaled_layout.make_smem_layout_sf(tile_shape, sf_vec_size,
/// num_stages)`: the [`BlockScaledBasicChunk`] tiled over the `(MN, K)` tile shape, along
/// K first, with a stage mode whose stride is the tile's cosize without its zero-stride
/// (vector) modes.
#[track_caller]
pub fn make_smem_layout_sf(tile_shape: impl Into<Shape>, sf_vec_size: usize, num_stages: usize) -> Layout {
    let chunk = BlockScaledBasicChunk::new(sf_vec_size, OperandMajorMode::K);
    let tile = tile_to_shape_ordered(chunk.layout(), tile_shape, &[1, 0]);
    let stage = tile.filter_zeros().cosize();
    tile.append(Layout::new(num_stages, stage))
}

/// The SM100 shared-memory SF layout of a `(mn, k)` CTA tile read by `(inst_mn, inst_k)`
/// MMA tiles.
#[track_caller]
fn smem_layout_sf_mma(
    what: &str,
    (mn, k): (usize, usize),
    (inst_mn, inst_k): (usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    assert!(sf_vec_size == 16 || sf_vec_size == 32, "cute: {what}: 'sf_vec_size' is {sf_vec_size}, not 16 or 32");
    assert!(
        inst_mn > 0 && inst_mn.is_multiple_of(128) && mn.is_multiple_of(inst_mn),
        "cute: {what}: one MMA's M or N extent {inst_mn} (of a {mn}-row tile) is not a multiple of 128 rows of \
         scale factors"
    );
    assert!(
        inst_k > 0 && inst_k.is_multiple_of(sf_vec_size) && k.is_multiple_of(inst_k),
        "cute: {what}: one MMA's K extent {inst_k} is not a multiple of the scale-factor vector ({sf_vec_size}) \
         dividing the tile's K extent {k}"
    );
    assert!(
        k.is_multiple_of(4 * sf_vec_size),
        "cute: {what}: the tile's K extent {k} is not a multiple of 4 scale-factor vectors ({})",
        4 * sf_vec_size
    );
    let chunk = BlockScaledBasicChunk::new(sf_vec_size, OperandMajorMode::K);
    // ((Atom_MN, Rest_MN), (Atom_K, Rest_K))
    let tile = tile_to_shape_ordered(chunk.layout(), (mn, k), &[1, 0]);
    // ((Inst_MN, Inst_K), MMA_MN, MMA_K)
    let inst: Tiler = (inst_mn, inst_k).into();
    let tile = tile.tiled_divide(&inst);
    // (((128, Inst_MN / 128), (sf_vec_size, NSF)), MMA_MN, MMA_K)
    let per_row: Tiler = ((128, sf_vec_size),).into();
    let tile = tile.logical_divide(&per_row);
    let stage = tile.filter_zeros().cosize();
    tile.append(Layout::new(num_stages, stage))
}

/// CuTe DSL's `blockscaled_layout.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk,
/// sf_vec_size, num_stages)`: the shared-memory layout of one CTA's A scale factors for a
/// tcgen05 block-scaled MMA, `(((128, M_inst / 128), (sf_vec_size, nsf)), MMA_M, MMA_K,
/// STAGE)` of 8-bit elements (`nsf` scale factors per row per instruction): the
/// [`BlockScaledBasicChunk`]s of the CTA's `(M / CTAs, K)` tile, K-chunks first, divided
/// into the MMA's tiles as [`TiledMma::partition_shape_a`](crate::TiledMma::partition_shape_a)
/// does A. The stage stride is the tile's compact size. The CTA's M must be a multiple
/// of 128 and the tile's K of `4 * sf_vec_size`.
#[track_caller]
pub fn make_smem_layout_sfa(
    tiled_mma: &crate::TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    let ctas = tiled_mma.thr_id().size();
    let [im, _, ik] = tiled_mma.atom().shape_mnk();
    let (m, _, k) = mma_tiler_mnk;
    let tile = (m / ctas, k);
    let inst = (tile.0 / (m / im).max(1), tile.1 / (k / ik).max(1));
    smem_layout_sf_mma("make_smem_layout_sfa", tile, inst, sf_vec_size, num_stages)
}

/// CuTe DSL's `blockscaled_layout.make_smem_layout_sfb`: [`make_smem_layout_sfa`] for B's
/// scale factors over N, rounded up to 128 rows (an MMA with N below 128 still reads a
/// whole 128-row block), each CTA holding the whole N of the tile.
#[track_caller]
pub fn make_smem_layout_sfb(
    tiled_mma: &crate::TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    let [_, inn, ik] = tiled_mma.atom().shape_mnk();
    let (_, n, k) = mma_tiler_mnk;
    let tile = (n.div_ceil(128) * 128, k);
    let inst = (tile.0 / (n / inn).max(1), tile.1 / (k / ik).max(1));
    smem_layout_sf_mma("make_smem_layout_sfb", tile, inst, sf_vec_size, num_stages)
}

/// The TMEM layout of the scale factors whose one-stage shared-memory layout is
/// `smem_layout` (`(((128, MN_inst / 128), (VS, NSF)), MMA_MN, MMA_K)`).
#[track_caller]
fn tmem_layout_sf(what: &str, sf_vec_size: usize, smem_layout: &Layout) -> Layout {
    assert!(
        smem_layout.rank() >= 3 && smem_layout.mode(0).rank() == 2,
        "cute: {what}: the shared-memory layout {smem_layout} is not one stage of ((MN, K), MMA_MN, MMA_K)"
    );
    let mma = smem_layout.mode(0);
    let (mn, inst_k) = (mma.mode(0).size(), mma.mode(1).size());
    assert!(
        mn.is_multiple_of(128) && inst_k.is_multiple_of(sf_vec_size),
        "cute: {what}: the MMA mode {mma} is not whole 128-row blocks of {sf_vec_size}-element vectors"
    );
    let nsf = inst_k / sf_vec_size;
    assert!((1..=4).contains(&nsf), "cute: {what}: {nsf} scale factors per row per MMA; a TMEM column holds 4");
    let (rest_mn, rest_k) = (smem_layout.mode(1).size(), smem_layout.mode(2).size());
    // tmem_sf_frg's atom, (((32, MN/32), 4 replicas), (VS, NSF)), in its logical
    // (128 lanes, byte) space, where index `lane + 128 * byte`: a 32-row group four bytes
    // (one column) on, the four replicas in the four sub-partitions, factor `j` in byte `j`.
    let atom = Layout::new((((32, mn / 32), 4), (sf_vec_size, nsf)), (((1, 512), 32), (0, 128)));
    // tiled_product with the (MMA_MN, MMA_K) tiles: the tiles fill the atom's complement, so
    // with fewer than four factors per row the next tiles take the column's free bytes
    // before moving past the atom's columns.
    let tiles = Layout::new((rest_mn, rest_k), (1, rest_mn));
    let rest = crate::layout::composition(&crate::layout::complement(&atom, atom.size() * tiles.cosize()), &tiles);
    let logical = Layout::from_modes([atom, rest.mode(0), rest.mode(1)]);
    // Then into TMEM 8-bit element offsets: lane at `SF_TMEM_LANE`, byte at 1.
    crate::layout::composition(&Layout::new((128, 1 << 14), (SF_TMEM_LANE, 1)), &logical)
}

/// CuTe DSL's `blockscaled_layout.make_tmem_layout_sfa(tiled_mma, mma_tiler_mnk,
/// sf_vec_size, smem_layout)`: the tensor-memory layout `(MMA, MMA_M, MMA_K)` of A's
/// scale factors (8-bit element offsets from the scale factors' first column) for the
/// one-stage shared-memory layout `smem_layout` of [`make_smem_layout_sfa`], as CuTe
/// C++'s `UMMA::tmem_sf_frg` lays it out (the derivation is above this function in the
/// source). Mode 0 is `((((32, MN/32), 4), (VS, NSF)))`: row `32 c + l` in lane `l` of each
/// sub-partition (the replica mode `4`, which a `Cp4x32x128bOp` copy writes at once) and
/// column `c`, its `NSF` factors in consecutive bytes. Slice it at `(.., .., k_block)` for
/// the MMA's `Field::SFA`.
#[track_caller]
pub fn make_tmem_layout_sfa(
    tiled_mma: &crate::TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    smem_layout: &Layout,
) -> Layout {
    let _ = mma_tiler_mnk;
    check_tmem_sf_mma("make_tmem_layout_sfa", tiled_mma);
    tmem_layout_sf("make_tmem_layout_sfa", sf_vec_size, smem_layout)
}

/// CuTe DSL's `blockscaled_layout.make_tmem_layout_sfb`: [`make_tmem_layout_sfa`] for B's
/// scale factors (`(MMA, MMA_N, MMA_K)`, an N of 256 taking eight columns per k-block).
/// One CTA's MMAs only: a CTA pair's SFB is split across both CTAs' TMEM (CuTe C++'s
/// 2x2 data path), which cute-rs does not lay out yet.
#[track_caller]
pub fn make_tmem_layout_sfb(
    tiled_mma: &crate::TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    sf_vec_size: usize,
    smem_layout: &Layout,
) -> Layout {
    let _ = mma_tiler_mnk;
    check_tmem_sf_mma("make_tmem_layout_sfb", tiled_mma);
    assert!(
        tiled_mma.thr_id().size() == 1,
        "cute: make_tmem_layout_sfb: a CTA pair's B scale factors (the 2x2 data path) are not supported yet; use a \
         CtaGroup::ONE MMA"
    );
    tmem_layout_sf("make_tmem_layout_sfb", sf_vec_size, smem_layout)
}

#[track_caller]
fn check_tmem_sf_mma(what: &str, tiled_mma: &crate::TiledMma) {
    let atom = tiled_mma.atom();
    assert!(
        matches!(atom.op(), crate::atom::mma::MmaOp::Sm100(crate::ir::UmmaKind { scale: Some(_), .. })),
        "cute: {what}: the tiled MMA's atom {:?} is not a tcgen05 block-scaled MMA",
        atom.op()
    );
}

/// `sm103_dense_blockscaled_gemm_persistent.py`'s `sm103_make_smem_layout_sfa(tiled_mma,
/// mma_tiler, sf_vec_size, num_stages)`: the [`Sm103BlockScaledBasicChunk`] tiled
/// over `(MMA_M * M_atom, MMA_K * K_atom / k_divisor)` (`k_divisor` 4 for 16-element
/// vectors, else 2), plus a stage mode of the tile's compact size.
#[track_caller]
pub fn sm103_make_smem_layout_sfa(
    tiled_mma: &crate::TiledMma,
    mma_tiler: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    let shape = tiled_mma.partition_shape_a((mma_tiler.0, mma_tiler.2));
    let ext = |s: &Shape| s.product().as_static().expect("cute: sm103_make_smem_layout_sfa: a static MMA tile");
    let (atom_m, atom_k) = (ext(shape.mode(0).mode(0)), ext(shape.mode(0).mode(1)));
    let (rest_m, rest_k) = (ext(shape.mode(1)), ext(shape.mode(2)));
    let atom = Sm103BlockScaledBasicChunk::new(sf_vec_size, a_major_mode(tiled_mma));
    let k_divisor = if sf_vec_size == 16 { 4 } else { 2 };
    let tiler = (atom_m * rest_m, atom_k * rest_k / k_divisor);
    sm103_staged(atom.layout(), tiler, num_stages)
}

/// `sm103_make_smem_layout_sfb`: the chunk tiled over `(N, K / k_divisor)`, with a
/// 256-row atom (its second 128 rows one K extent of chunks on) when N is not 128.
#[track_caller]
pub fn sm103_make_smem_layout_sfb(
    tiled_mma: &crate::TiledMma,
    mma_tiler: (usize, usize, usize),
    sf_vec_size: usize,
    num_stages: usize,
) -> Layout {
    let atom = Sm103BlockScaledBasicChunk::new(sf_vec_size, a_major_mode(tiled_mma));
    let k_divisor = if sf_vec_size == 16 { 4 } else { 2 };
    let tiler = (mma_tiler.1, mma_tiler.2 / k_divisor);
    if tiler.0 == 128 {
        sm103_staged(atom.layout(), tiler, num_stages)
    } else {
        let atom256 = Layout::new(((32, 4, 2), (sf_vec_size, 4)), ((16, 4, tiler.1 / sf_vec_size / 4 * 512), (0, 1)));
        sm103_staged(&atom256, tiler, num_stages)
    }
}

#[track_caller]
fn sm103_staged(atom: &Layout, tiler: (usize, usize), num_stages: usize) -> Layout {
    let ashape = atom.shape().product_each();
    let ext = |i: usize| ashape.mode(i).product().as_static().expect("a static atom");
    let reps = Layout::col_major((tiler.0 / ext(0), tiler.1 / ext(1)));
    let tile = crate::layout::tiled_product(atom, &reps);
    let stage = tile.filter_zeros().size();
    tile.append(Layout::new(num_stages, stage))
}

/// The A operand's major mode of a tcgen05 tiled MMA (`tiled_mma.op.a_major_mode`).
fn a_major_mode(tiled_mma: &crate::TiledMma) -> OperandMajorMode {
    match tiled_mma.atom().op() {
        crate::atom::mma::MmaOp::Sm100(crate::ir::UmmaKind { trans_a: true, .. }) => OperandMajorMode::MN,
        _ => OperandMajorMode::K,
    }
}

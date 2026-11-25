// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm100_frag.hpp (UMMA::tmem_e_frg, the TMEM layouts of
//   the sparsity metadata, in make_sparse_tmem_layout_e),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! CuTe DSL's SM100 sparse GEMM layout helpers (roadmap part
//! m10-sparse-umma; `cutlass.utils` from `blackwell_helpers`): the element
//! types of compressed A and of the metadata E ([`make_sparse_a_dtype`],
//! [`make_sparse_e_dtype`]), the metadata atom ([`make_tensor_e_atom_layout`]),
//! the metadata layouts in global, shared and tensor memory
//! ([`make_sparse_gmem_layout_e`], [`make_sparse_smem_layout_e`],
//! [`make_sparse_tmem_layout_e`]) and the staged shared-memory layout of
//! compressed A ([`make_sparse_smem_layout_a`]). Re-exported from
//! [`utils::sm100`](super::sm100).
//!
//! The metadata layouts count logical metadata elements of
//! [`make_sparse_e_dtype`] (one bit each for 16- and 8-bit A: the 4-bit
//! field of a group of four K elements is four logical elements, in order
//! of K): `(M, K) ->` offset. Recast them to Uint8 with
//! [`recast_layout_from_sparse`](crate::recast_layout_from_sparse) for
//! byte tensors. One metadata atom (128 rows by 128 logical K for 16- and
//! 8-bit A) is 16 bytes per row in shared memory, which a 128x128b
//! `tcgen05.cp` ([`Cp128x128bOp`](crate::tcgen05::Cp128x128bOp)) moves to
//! four TMEM columns.

use crate::atom::mma::{MmaOp, TiledMma};
use crate::ir::DType;
use crate::layout::{ComposedLayout, IntTuple, Layout, Shape, Swizzle, Tree, make_composed_layout};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::tcgen05::tile_to_mma_shape;
use crate::sparse_elem::{SparseElemType, get_sparse_elem_type};

/// The element type of compressed A for raw A of `a_raw_dtype` (CuTe
/// DSL's `make_sparse_a_dtype`): two logical elements per stored one, or
/// for Float4E2M1FN four per Uint8.
#[track_caller]
pub fn make_sparse_a_dtype(a_raw_dtype: DType) -> SparseElemType {
    if a_raw_dtype == DType::F4E2M1FN {
        get_sparse_elem_type(4, 1, DType::U8)
    } else {
        get_sparse_elem_type(2, 1, a_raw_dtype)
    }
}

/// The element type of the metadata E for raw A of `a_raw_dtype` (CuTe
/// DSL's `make_sparse_e_dtype`): logical elements per Uint8, 16 for
/// Float4E2M1FN, 8 for the 16- and 8-bit types, 4 for TFloat32 and
/// Float32. Panics for any other type.
#[track_caller]
pub fn make_sparse_e_dtype(a_raw_dtype: DType) -> SparseElemType {
    use DType::*;
    let n = match a_raw_dtype {
        F4E2M1FN => 16,
        F16 | BF16 | I8 | U8 | F8E4M3FN | F8E5M2 => 8,
        TF32 | F32 => 4,
        t => panic!("cute: make_sparse_e_dtype: no sparse metadata type for A of {}", t.cute_name()),
    };
    get_sparse_elem_type(n, 1, U8)
}

/// The metadata atom for raw A of `a_raw_dtype` (CuTe DSL's
/// `make_tensor_e_atom_layout`): `(M, K) ->` offset in logical metadata
/// elements of one 128-row block (row-major `(128, 256)` for Float4E2M1FN,
/// `(128, 128)` for the 8-bit types, and for TFloat32 and the 16-bit types
/// the interleaving the TMEM layout reads in rows of 16 bytes).
#[track_caller]
pub fn make_tensor_e_atom_layout(a_raw_dtype: DType) -> Layout {
    use DType::*;
    match a_raw_dtype {
        F4E2M1FN => Layout::new((128, 256), (256, 1)),
        I8 | U8 | F8E4M3FN | F8E5M2 => Layout::new((128, 128), (128, 1)),
        TF32 | F32 => Layout::new(((8, 2, 8), (8, 2, 4)), ((64, 8, 1024), (1, 512, 16))),
        F16 | BF16 => Layout::new(((8, 2, 8), (16, 2, 4)), ((128, 16, 2048), (1, 1024, 32))),
        t => panic!("cute: make_tensor_e_atom_layout: no sparse metadata atom for A of {}", t.cute_name()),
    }
}

/// The global-memory layout of the metadata of an `(M, N, K, L)` problem
/// (CuTe DSL's `make_sparse_gmem_layout_e(problem_shape_mnkl,
/// a_raw_dtype)`): M and K rounded up to whole atoms, the atoms tiled
/// M-block first, then K-blocks, then batches: `((ATOM_M, M / ATOM_M),
/// (ATOM_K, K / ATOM_K), L) ->` logical metadata element.
#[track_caller]
pub fn make_sparse_gmem_layout_e(problem_shape_mnkl: (usize, usize, usize, usize), a_raw_dtype: DType) -> Layout {
    let (m, _, k, l) = problem_shape_mnkl;
    let atom = make_tensor_e_atom_layout(a_raw_dtype);
    let (am, ak) = (atom.mode(0).size(), atom.mode(1).size());
    let cosize = atom.cosize();
    let (m_al, k_al) = (m.div_ceil(am) * am, k.div_ceil(ak) * ak);
    let mode = |a: &Layout, n: usize, stride: usize| Layout::from_modes([a.clone(), Layout::new(n, stride)]);
    Layout::from_modes([
        mode(&atom.mode(0), m_al / am, cosize),
        mode(&atom.mode(1), k_al / ak, m_al * ak),
        Layout::new(l, m_al * k_al),
    ])
}

/// The logical (uncompressed) partition shape `((MMA_M, MMA_K), REST_M,
/// REST_K)` of an `(m, k)` A tile of a sparse tiled MMA, whose
/// [`partition_shape_a`](TiledMma::partition_shape_a) counts compressed K.
fn logical_a_partition(tiled_mma: &TiledMma, (m, k): (usize, usize), what: &str) -> Shape {
    assert!(
        tiled_mma.atom().is_sparse(),
        "cute: {what}: the tiled MMA {:?} is not a sparse MMA",
        tiled_mma.atom().op()
    );
    let s = tiled_mma.partition_shape_a((m, k / 2));
    let s = s.as_static().unwrap_or_else(|| panic!("cute: {what}: the partition shape {s} is not static"));
    assert!(s.rank() == 3 && s.mode(0).rank() == 2, "cute: {what}: the partition shape {s} is not ((M, K), M', K')");
    let first = Tree::tuple([s.mode(0).mode(0).clone(), IntTuple::from(2 * s.mode(0).mode(1).size())]);
    Shape::from(&Tree::tuple([first, s.mode(1).clone(), s.mode(2).clone()]))
}

fn with_stages(shape: &Shape, num_stages: usize) -> Shape {
    let s = shape.as_static().expect("a static shape");
    Shape::from(&Tree::tuple(s.modes().iter().cloned().chain([IntTuple::from(num_stages)])))
}

/// The staged shared-memory layout of the metadata (CuTe DSL's
/// `make_sparse_smem_layout_e(tiled_mma, mma_tiler_e_mnk, a_raw_dtype,
/// num_stages)`): the metadata atom ([`make_tensor_e_atom_layout`], under
/// the identity swizzle `Swizzle(0, 4, 3)`) tiled K-block first to A's
/// logical partition of the `(M, K)` tile with `num_stages` appended:
/// `((MMA_M, MMA_K), REST_M, REST_K, STAGES) ->` logical metadata element.
#[track_caller]
pub fn make_sparse_smem_layout_e(
    tiled_mma: &TiledMma,
    mma_tiler_e_mnk: (usize, usize, usize),
    a_raw_dtype: DType,
    num_stages: usize,
) -> ComposedLayout {
    let (m, _, k) = mma_tiler_e_mnk;
    let shape = logical_a_partition(tiled_mma, (m, k), "make_sparse_smem_layout_e");
    let atom = make_composed_layout(Swizzle::new(0, 4, 3), 0, make_tensor_e_atom_layout(a_raw_dtype));
    tile_to_mma_shape(&atom, with_stages(&shape, num_stages), Some(&[0, 1, 2]))
}

/// The TMEM layout of the metadata (CuTe DSL's
/// `make_sparse_tmem_layout_e(tmem_shape, a_raw_dtype)`, CuTe C++'s
/// `UMMA::tmem_e_frg<T>`): `tmem_shape` is `((M_MMA, K_MMA), REST...)`,
/// one MMA's 128 rows by its logical K (16 for 32-bit A, 32 for 16-bit, 64
/// for 8-bit, 128 for FP4), tiled over `REST`; the result maps it to TMEM
/// offsets in logical metadata elements (lane at `2^21` for one-bit
/// elements, `2^20` for TFloat32's two-bit ones, `2^22` for FP4's half-bit
/// ones). Recast to Uint8 for a TMEM tensor of bytes. For 16-bit A each
/// MMA's metadata is one TMEM column, rows `16 q + r` and `16 q + 8 + r` in
/// lanes `16 q + 8 h + r` (the halves `h` of K); for 8-bit A two columns,
/// row `i` in lane `i`. Panics unless `M_MMA` is 128 and `K_MMA` matches.
#[track_caller]
pub fn make_sparse_tmem_layout_e(tmem_shape: impl Into<Shape>, a_raw_dtype: DType) -> Layout {
    let shape: Shape = tmem_shape.into();
    let s =
        shape.as_static().unwrap_or_else(|| panic!("cute: make_sparse_tmem_layout_e: the shape {shape} is not static"));
    assert!(
        s.rank() >= 1 && !s.mode(0).is_leaf() && s.mode(0).rank() == 2,
        "cute: make_sparse_tmem_layout_e: the shape {shape} is not ((M_MMA, K_MMA), ...)"
    );
    let (mm, nn) = (s.mode(0).mode(0).size(), s.mode(0).mode(1).size());
    assert!(mm == 128, "cute: make_sparse_tmem_layout_e: M_MMA is {mm}; the metadata layout has 128 rows");
    let bits = a_raw_dtype.bits();
    let (want, atom, dp) = match bits {
        32 => (16, Layout::new(((8, 2, 8), (8, 2)), ((1, 1024, 16), (128, 8))), 1usize << 20),
        16 => (32, Layout::new(((8, 2, 8), (16, 2)), ((1, 2048, 16), (128, 8))), 1 << 21),
        8 => (64, Layout::new((128, 64), (1, 128)), 1 << 21),
        4 => (128, Layout::new((128, 128), (1, 128)), 1 << 22),
        _ => panic!("cute: make_sparse_tmem_layout_e: no metadata layout for A of {}", a_raw_dtype.cute_name()),
    };
    assert!(
        nn == want,
        "cute: make_sparse_tmem_layout_e: K_MMA is {nn}; for {}-bit A one MMA's metadata covers K = {want}",
        bits
    );
    let rest: Vec<usize> = (1..s.rank()).map(|i| s.mode(i).size()).collect();
    let rest = if rest.is_empty() { vec![1] } else { rest };
    let logical = crate::layout::tiled_product(&atom, &Layout::col_major(rest));
    // "Virtual" addresses (lane stride 1, column stride 128) to lane and element offsets.
    let restride = Layout::new((128, 1usize << 15), (dp, 1));
    restride.compose(&logical)
}

/// The staged shared-memory layout of compressed A (CuTe DSL's
/// `make_sparse_smem_layout_a(tiled_mma, mma_tiler_mnk, a_raw_dtype,
/// num_stages, sparsity=2)`): A's partition of the `(M, K)` tile, the
/// layout atom [`get_smem_layout_atom_ab`](super::sm100::get_smem_layout_atom_ab)
/// picks with the sparsity, tiled in the order of A's majorness with
/// `num_stages` appended. cute-rs counts compressed A in stored elements
/// (CuTe DSL in logical ones through a sparse pointer): the result is
/// `((MMA_M, MMA_K / sparsity), REST_M, REST_K, STAGES) ->` stored element,
/// the dense atom on the compressed tile, which holds the same bytes.
#[track_caller]
pub fn make_sparse_smem_layout_a(
    tiled_mma: &TiledMma,
    mma_tiler_mnk: (usize, usize, usize),
    a_raw_dtype: DType,
    num_stages: usize,
    sparsity: usize,
) -> ComposedLayout {
    let what = "make_sparse_smem_layout_a";
    let MmaOp::Sm100(kind) = tiled_mma.atom().op() else {
        panic!("cute: {what}: the tiled MMA {:?} is not a tcgen05 MMA", tiled_mma.atom().op())
    };
    assert!(
        kind.sparse && sparsity == 2,
        "cute: {what}: sparsity {sparsity} with {:?}; cute-rs's sparse MMAs are 2:4 (sparsity 2)",
        tiled_mma.atom().op()
    );
    let major = if kind.trans_a { OperandMajorMode::MN } else { OperandMajorMode::K };
    let (m, _, k) = mma_tiler_mnk;
    let shape = tiled_mma.partition_shape_a((m, k / sparsity));
    let s = shape.as_static().unwrap_or_else(|| panic!("cute: {what}: the partition shape {shape} is not static"));
    let ext = (s.mode(0).mode(0).size() * s.mode(1).size(), s.mode(0).mode(1).size() * s.mode(2).size());
    // The atom of the logical tile with the sparsity is the dense atom of the compressed one.
    let kind_atom = super::sm100::get_smem_layout_atom_ab(major, a_raw_dtype, (ext.0, ext.1 * sparsity), sparsity);
    let atom = crate::nvgpu::tcgen05::make_smem_layout_atom_dtype(kind_atom, a_raw_dtype);
    let order: &[usize] = if major == OperandMajorMode::K { &[0, 1, 2] } else { &[1, 0, 2] };
    tile_to_mma_shape(&atom, with_stages(&shape, num_stages), Some(order))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_layouts_agree_on_bytes() {
        // 16-bit A: the bit of (m, k) in the shared-memory atom, read as 16-byte rows, is the
        // bit the TMEM layout puts in lane (row of 16 bytes) and column (4-byte word).
        let atom = make_tensor_e_atom_layout(DType::F16);
        let tmem = make_sparse_tmem_layout_e(((128, 32), 1, 4), DType::F16);
        for (m, k) in [(0, 0), (9, 17), (37, 70), (127, 127), (64, 33)] {
            let smem_bit = atom.call(m + 128 * k);
            let t = tmem.call(m + 128 * (k % 32) + 128 * 32 * (k / 32));
            let (lane, bit) = (t >> 21, t & ((1 << 21) - 1));
            assert_eq!(smem_bit, lane * 128 + bit, "({m}, {k})");
        }
        // 8-bit A: row i in lane i, K in order through two columns per MMA.
        let tmem = make_sparse_tmem_layout_e(((128, 64), 1, 2), DType::F8E4M3FN);
        assert_eq!(tmem.call(5 + 128 * 70), (5 << 21) + 70);
        let g = make_sparse_gmem_layout_e((256, 64, 256, 1), DType::F16);
        assert_eq!(g.cosize(), 256 * 256);
    }
}

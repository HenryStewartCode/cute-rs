// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_atom.hpp, include/cute/atom/mma_traits.hpp,
//   include/cute/atom/mma_traits_sm80.hpp, include/cute/atom/mma_traits_sm89.hpp,
//   include/cute/atom/mma_traits_sm90_gmma.hpp (the WGMMA fragment layouts),
//   include/cute/atom/mma_traits_sm100.hpp (the tcgen05 UMMA layouts),
//   include/cute/atom/mma_traits_sm120.hpp (the SM120 block-scaled fragment and
//   scale-factor layouts) and
//   include/cute/algorithm/gemm.hpp,
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! MMA atoms and tiled MMAs.
//!
//! An [`MmaAtom`] describes one multiply-accumulate instruction: its
//! `M x N x K` shape and, for each operand, a `(thread, value)` layout giving
//! the coordinate (colexicographic index into `M x K`, `N x K` or `M x N`)
//! that each thread's value holds. A [`TiledMma`] repeats an atom over a
//! layout of atoms (one per warp, for example); [`ThrMma`] is one thread's
//! view, which partitions operand tensors into per-thread fragments.

use std::panic::Location;

use crate::builder;
use crate::ir::{DType, MmaKind, Op};
use crate::layout::{Coord, IndexExpr, Layout, Shape, Tiler, Tree};
use crate::tensor::{Tensor, make_compact_fragment_like, require_static_fragment_shape};
use crate::value::{Numeric, Value};

/// The instruction an MMA atom issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmaOp {
    /// One thread computing `d = a * b + c` on scalars of the given float
    /// type (CuTe's `UniversalFMA`).
    UniversalFma(DType),
    /// A warp-wide tensor core `mma.sync` (SM80, or SM89 for FP8) or sparse
    /// `mma.sp`. [`MmaF16BF16Op`], [`MmaTF32Op`], [`MmaFP8Op`] and
    /// [`MmaF16BF16SparseOp`] pick the kind from CuTe DSL's parameters.
    Sm80(MmaKind),
    /// A warpgroup MMA (`wgmma.mma_async`, sm_90a) with A and B in shared
    /// memory; [`warpgroup::MmaF16BF16Op`](crate::nvgpu::warpgroup::MmaF16BF16Op)
    /// picks it.
    Sm90(crate::ir::WgmmaKind),
    /// A tcgen05 MMA (`tcgen05.mma`, sm_100a) with A and B in shared memory
    /// and D in tensor memory;
    /// [`tcgen05::MmaF16BF16Op`](crate::nvgpu::tcgen05::MmaF16BF16Op) picks it.
    Sm100(crate::ir::UmmaKind),
    /// A warp-level block-scaled `mma.sync` (sm_120a); the `cute::warp` operations
    /// ([`MmaMXF4Op`](crate::warp::MmaMXF4Op) and the others) pick it.
    Sm120(crate::ir::BlockScaledMmaKind),
}

/// CuTe DSL's `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)`:
/// a warp-level `mma.sync` (SM80 and newer) on f16 or bf16 operands, with A
/// row-major and B column-major (both K-major).
///
/// | A/B  | accumulator | shape_mnk              |
/// |------|-------------|------------------------|
/// | F16  | F16, F32    | (16, 8, 8), (16, 8, 16) |
/// | BF16 | F32         | (16, 8, 8), (16, 8, 16) |
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16Op {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
}

impl MmaF16BF16Op {
    /// The operation, or a panic naming the unsupported parameter (as CuTe
    /// DSL raises at construction).
    #[track_caller]
    pub fn new(ab_dtype: DType, acc_dtype: DType, shape_mnk: (usize, usize, usize)) -> MmaF16BF16Op {
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: MmaF16BF16Op takes Float16 or BFloat16 A and B operands, but 'ab_dtype' is {ab_dtype}"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: MmaF16BF16Op accumulates in Float16 or Float32, but 'acc_dtype' is {acc_dtype}"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: MmaF16BF16Op with BFloat16 operands ('ab_dtype') accumulates in Float32 only, but 'acc_dtype' is \
             {acc_dtype}"
        );
        assert!(
            matches!(shape_mnk, (16, 8, 8) | (16, 8, 16)),
            "cute: MmaF16BF16Op has no instruction of 'shape_mnk' {shape_mnk:?}; its shapes are (16,8,8) and (16,8,16)"
        );
        MmaF16BF16Op { ab_dtype, acc_dtype, shape_mnk }
    }

    pub fn ab_dtype(&self) -> DType {
        self.ab_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    /// The `mma.sync` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        match (self.ab_dtype, self.acc_dtype, self.shape_mnk.2) {
            (DType::F16, DType::F32, 16) => MmaKind::M16N8K16F32F16,
            (DType::F16, DType::F32, _) => MmaKind::M16N8K8F32F16,
            (DType::F16, _, 16) => MmaKind::M16N8K16F16F16,
            (DType::F16, _, _) => MmaKind::M16N8K8F16F16,
            (_, _, 16) => MmaKind::M16N8K16F32BF16,
            _ => MmaKind::M16N8K8F32BF16,
        }
    }
}

/// CuTe DSL's `cute.nvgpu.warp.MmaTF32Op(shape_mnk)`: a warp-level
/// `mma.sync` (SM80 and newer) on TF32 operands with an f32 accumulator.
/// The A and B fragments hold [`TFloat32`](crate::TFloat32) values (or, as
/// before, Float32 values, whose low 13 mantissa bits the tensor core
/// ignores); both are the same `f32` registers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaTF32Op {
    shape_mnk: (usize, usize, usize),
}

impl MmaTF32Op {
    /// The operation for `shape_mnk` `(16, 8, 4)` or `(16, 8, 8)`.
    #[track_caller]
    pub fn new(shape_mnk: (usize, usize, usize)) -> MmaTF32Op {
        assert!(
            matches!(shape_mnk, (16, 8, 4) | (16, 8, 8)),
            "cute: MmaTF32Op has no instruction of 'shape_mnk' {shape_mnk:?}; its shapes are (16,8,4) and (16,8,8)"
        );
        MmaTF32Op { shape_mnk }
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    /// The `mma.sync` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        if self.shape_mnk.2 == 4 { MmaKind::M16N8K4Tf32 } else { MmaKind::M16N8K8Tf32 }
    }
}

/// CuTe DSL's `cute.nvgpu.warp.MmaFP8Op(ab_dtype, acc_dtype, shape_mnk)`: a
/// warp-level `mma.sync` (SM89 and newer) on FP8 operands, A row-major and B
/// column-major (both K-major). A and B have the same type, Float8E4M3FN
/// or Float8E5M2; the accumulators are Float16 or Float32; the shape is
/// m16n8k16 or m16n8k32.
///
/// The m16n8k32 shape with an f32 accumulator needs PTX ISA 8.4; the m16n8k16
/// shape or an f16 accumulator PTX 8.7. Its `nvvm.mma.sync` form needs the
/// `mlir-opt` of LLVM 23 (or [`CompileOptions::mma_inline_ptx`](crate::CompileOptions::mma_inline_ptx)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaFP8Op {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
}

impl MmaFP8Op {
    /// The operation, or a panic naming the unsupported parameter (as CuTe
    /// DSL raises at construction).
    #[track_caller]
    pub fn new(ab_dtype: DType, acc_dtype: DType, shape_mnk: (usize, usize, usize)) -> MmaFP8Op {
        assert!(
            matches!(ab_dtype, DType::F8E4M3FN | DType::F8E5M2),
            "cute: MmaFP8Op takes Float8E4M3FN or Float8E5M2 A and B operands, but 'ab_dtype' is {ab_dtype}"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: MmaFP8Op accumulates in Float16 or Float32, but 'acc_dtype' is {acc_dtype}"
        );
        assert!(
            matches!(shape_mnk, (16, 8, 16) | (16, 8, 32)),
            "cute: MmaFP8Op has no instruction of 'shape_mnk' {shape_mnk:?}; its shapes are (16,8,16) and (16,8,32)"
        );
        MmaFP8Op { ab_dtype, acc_dtype, shape_mnk }
    }

    pub fn ab_dtype(&self) -> DType {
        self.ab_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    /// The `mma.sync` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        use MmaKind::*;
        let e4m3 = self.ab_dtype == DType::F8E4M3FN;
        match (e4m3, self.acc_dtype == DType::F32, self.shape_mnk.2 == 32) {
            (true, true, true) => M16N8K32F32E4M3,
            (true, false, true) => M16N8K32F16E4M3,
            (true, true, false) => M16N8K16F32E4M3,
            (true, false, false) => M16N8K16F16E4M3,
            (false, true, true) => M16N8K32F32E5M2,
            (false, false, true) => M16N8K32F16E5M2,
            (false, true, false) => M16N8K16F32E5M2,
            (false, false, false) => M16N8K16F16E5M2,
        }
    }
}

/// CuTe DSL's `cute.nvgpu.warp.SparseMetadataFormat`: how a sparse MMA's
/// metadata is laid out over the threads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SparseMetadataFormat {
    /// By thread id: sparsity selector 0, so in each group of four lanes the
    /// first (m16n8k16) or the first two (m16n8k32) supply the metadata
    /// ([`sparse_metadata_lane`]).
    Tid,
}

/// CuTe DSL's `cute.nvgpu.warp.MmaF16BF16SparseOp(ab_dtype, acc_dtype,
/// shape_mnk, sparse_metadata_format)`: a warp-level sparse
/// `mma.sp::ordered_metadata.sync` (SM80 and newer, PTX ISA 8.5) on f16 or
/// bf16 operands whose A is 2:4 structured sparse: of every four values
/// along K, A holds the two kept ones (a `(M, K / 2)` compressed tile,
/// partitioned with [`ThrMma::partition_a`]), and a Uint32 metadata register
/// per thread names their positions ([`ThrMma::partition_fragment_e`],
/// [`sparse_compress`]). [`gemm_sparse`] issues it. Its `nvvm.mma.sp.sync`
/// form needs the `mlir-opt` of LLVM 22 (or
/// [`CompileOptions::mma_inline_ptx`](crate::CompileOptions::mma_inline_ptx)).
///
/// Float16 operands accumulate in Float16 or Float32, BFloat16 ones in
/// Float32 only; the shape is m16n8k16 or m16n8k32 (K counting the dense A).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16SparseOp {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
    format: SparseMetadataFormat,
}

impl MmaF16BF16SparseOp {
    /// The operation, or a panic naming the unsupported parameter (as CuTe
    /// DSL raises at construction).
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        shape_mnk: (usize, usize, usize),
        sparse_metadata_format: SparseMetadataFormat,
    ) -> MmaF16BF16SparseOp {
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: MmaF16BF16SparseOp takes Float16 or BFloat16 A and B operands, but 'ab_dtype' is {ab_dtype}"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: MmaF16BF16SparseOp accumulates in Float16 or Float32, but 'acc_dtype' is {acc_dtype}"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: MmaF16BF16SparseOp with BFloat16 operands ('ab_dtype') accumulates in Float32 only, but \
             'acc_dtype' is {acc_dtype}"
        );
        assert!(
            matches!(shape_mnk, (16, 8, 16) | (16, 8, 32)),
            "cute: MmaF16BF16SparseOp has no instruction of 'shape_mnk' {shape_mnk:?}; its shapes are (16,8,16) and \
             (16,8,32)"
        );
        MmaF16BF16SparseOp { ab_dtype, acc_dtype, shape_mnk, format: sparse_metadata_format }
    }

    pub fn ab_dtype(&self) -> DType {
        self.ab_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    pub fn sparse_metadata_format(&self) -> SparseMetadataFormat {
        self.format
    }

    /// The `mma.sp` instruction it issues.
    pub fn kind(&self) -> MmaKind {
        use MmaKind::*;
        match (self.ab_dtype, self.acc_dtype, self.shape_mnk.2) {
            (DType::F16, DType::F32, 16) => SpM16N8K16F32F16,
            (DType::F16, DType::F32, _) => SpM16N8K32F32F16,
            (DType::F16, _, 16) => SpM16N8K16F16F16,
            (DType::F16, _, _) => SpM16N8K32F16F16,
            (_, _, 16) => SpM16N8K16F32BF16,
            _ => SpM16N8K32F32BF16,
        }
    }
}

/// The lane and nibble holding the metadata of the 4-wide chunk `chunk`
/// (columns `4 * chunk` to `4 * chunk + 3` of the dense A) of row `row` of a
/// sparse MMA of K extent `k` (16 or 32), in the
/// [`SparseMetadataFormat::Tid`] format: lane `4 * (row % 8)` (m16n8k16) or
/// lane `4 * (row % 8) + chunk / 4` (m16n8k32) holds rows `row % 8` in
/// nibbles 0 to 3 and `row % 8 + 8` in nibbles 4 to 7, one nibble per chunk
/// (PTX ISA, "Sparse matrix storage" and the sparse `.f16`/`.bf16` metadata
/// layouts, checked on an SM89 GPU). A nibble holds the positions in the
/// chunk of the two kept values, the first in its low two bits, increasing:
/// `0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101` or `0b1110`.
pub fn sparse_metadata_lane(k: usize, row: usize, chunk: usize) -> (usize, usize) {
    assert!(k == 16 || k == 32, "sparse MMAs have K = 16 or 32, not {k}");
    crate::sim::exec_sparse_metadata_slot(k, row, chunk)
}

/// Host side: a 2:4 sparse `16 x k` A tile (`dense`, row-major; `k` 16 or
/// 32) as the compressed `16 x k / 2` tile a sparse MMA takes (row-major)
/// and the metadata register of each of the 32 lanes, for
/// [`SparseMetadataFormat::Tid`]. Each chunk of four values along a row may
/// hold at most two values other than `zero`; a chunk with fewer keeps
/// `zero`s in the first free positions. Lanes that supply no metadata get 0.
#[track_caller]
pub fn sparse_compress<T: Copy + PartialEq>(dense: &[T], k: usize, zero: T) -> (Vec<T>, [u32; 32]) {
    assert!(k == 16 || k == 32, "sparse MMAs have K = 16 or 32, not {k}");
    assert_eq!(dense.len(), 16 * k, "a sparse A tile has 16 x {k} values");
    let mut compressed = Vec::with_capacity(8 * k);
    let mut meta = [0u32; 32];
    for row in 0..16 {
        for chunk in 0..k / 4 {
            let vals = &dense[row * k + 4 * chunk..row * k + 4 * chunk + 4];
            let mut keep: Vec<usize> = (0..4).filter(|&i| vals[i] != zero).collect();
            assert!(
                keep.len() <= 2,
                "row {row}, columns {} to {}: more than two of four values are kept",
                4 * chunk,
                4 * chunk + 3
            );
            for i in 0..4 {
                if keep.len() < 2 && !keep.contains(&i) {
                    keep.push(i);
                }
            }
            keep.sort();
            compressed.extend([vals[keep[0]], vals[keep[1]]]);
            let (lane, nibble) = sparse_metadata_lane(k, row, chunk);
            meta[lane] |= ((keep[0] | keep[1] << 2) as u32) << (4 * nibble);
        }
    }
    (compressed, meta)
}

impl From<MmaFP8Op> for MmaOp {
    fn from(op: MmaFP8Op) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

impl From<MmaF16BF16SparseOp> for MmaOp {
    fn from(op: MmaF16BF16SparseOp) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

impl From<MmaFP8Op> for MmaAtom {
    fn from(op: MmaFP8Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

impl From<MmaF16BF16SparseOp> for MmaAtom {
    fn from(op: MmaF16BF16SparseOp) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

impl From<MmaKind> for MmaOp {
    fn from(kind: MmaKind) -> MmaOp {
        MmaOp::Sm80(kind)
    }
}

impl From<MmaF16BF16Op> for MmaOp {
    fn from(op: MmaF16BF16Op) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

impl From<MmaTF32Op> for MmaOp {
    fn from(op: MmaTF32Op) -> MmaOp {
        MmaOp::Sm80(op.kind())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Operand {
    A,
    B,
    C,
}

/// One MMA instruction's shape and fragment layouts.
#[derive(Clone, Debug)]
pub struct MmaAtom {
    op: MmaOp,
    shape_mnk: [usize; 3],
    thr_id: Layout,
    a: Layout,
    b: Layout,
    c: Layout,
    /// The runtime fields ([`MmaField`](crate::MmaField)) the atom takes and
    /// their values.
    pub(crate) fields: super::fields::MmaFields,
}

impl MmaAtom {
    /// The atom for `op`.
    pub fn new(op: MmaOp) -> MmaAtom {
        match op {
            MmaOp::UniversalFma(t) => {
                assert!(t.is_float(), "UniversalFma needs a float type, not {t}");
                let one = Layout::new((1, 1), (0, 0));
                MmaAtom {
                    op,
                    shape_mnk: [1, 1, 1],
                    thr_id: Layout::new(1, 0),
                    a: one.clone(),
                    b: one.clone(),
                    c: one,
                    fields: Default::default(),
                }
            }
            MmaOp::Sm80(kind) => {
                let (m, n, k) = kind.shape_mnk();
                // CuTe's `MMA_Traits<SM80_16x8x*_*_TN>`: thread `(t, g)` is
                // lane `t + 4 * g`; layouts map to the colexicographic index
                // into `M x K`, `N x K` and `M x N`.
                let (a, b) = match kind {
                    MmaKind::M16N8K16F32F16 | MmaKind::M16N8K16F16F16 | MmaKind::M16N8K16F32BF16 => (
                        Layout::new(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
                        Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    ),
                    MmaKind::M16N8K8F32F16 | MmaKind::M16N8K8F16F16 | MmaKind::M16N8K8F32BF16 => {
                        (Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))), Layout::new(((4, 8), 2), ((16, 1), 8)))
                    }
                    MmaKind::M16N8K4Tf32 => {
                        (Layout::new(((4, 8), 2), ((16, 1), 8)), Layout::new(((4, 8), 1), ((8, 1), 0)))
                    }
                    MmaKind::M16N8K8Tf32 => {
                        (Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))), Layout::new(((4, 8), 2), ((8, 1), 32)))
                    }
                    MmaKind::M8N8K4F64 => {
                        (Layout::new(((4, 8), 1), ((8, 1), 0)), Layout::new(((4, 8), 1), ((8, 1), 0)))
                    }
                    // CuTe's `MMA_Traits<SM89_16x8x32_*_TN>` (the .s8 layouts
                    // of SM80_16x8x32), and the PTX ISA's m16n8k16 FP8
                    // fragments.
                    MmaKind::M16N8K32F32E4M3
                    | MmaKind::M16N8K32F16E4M3
                    | MmaKind::M16N8K32F32E5M2
                    | MmaKind::M16N8K32F16E5M2 => (
                        Layout::new(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
                        Layout::new(((4, 8), (4, 2)), ((32, 1), (8, 128))),
                    ),
                    MmaKind::M16N8K16F32E4M3
                    | MmaKind::M16N8K16F16E4M3
                    | MmaKind::M16N8K16F32E5M2
                    | MmaKind::M16N8K16F16E5M2 => {
                        (Layout::new(((4, 8), (4, 2)), ((64, 1), (16, 8))), Layout::new(((4, 8), 4), ((32, 1), 8)))
                    }
                    // Sparse: A is the compressed `16 x K/2` tile, held as
                    // the dense f16 A of half the K (PTX ISA's sparse
                    // fragment tables); B as the dense m16n8k16 B, with
                    // four pairs of rows for m16n8k32.
                    MmaKind::SpM16N8K16F32F16 | MmaKind::SpM16N8K16F16F16 | MmaKind::SpM16N8K16F32BF16 => (
                        Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))),
                        Layout::new(((4, 8), (2, 2)), ((16, 1), (8, 64))),
                    ),
                    MmaKind::SpM16N8K32F32F16 | MmaKind::SpM16N8K32F16F16 | MmaKind::SpM16N8K32F32BF16 => (
                        Layout::new(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
                        Layout::new(((4, 8), (2, 4)), ((16, 1), (8, 64))),
                    ),
                };
                let c = match kind {
                    MmaKind::M8N8K4F64 => Layout::new(((4, 8), 2), ((16, 1), 8)),
                    _ => Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))),
                };
                MmaAtom { op, shape_mnk: [m, n, k], thr_id: Layout::new(32, 1), a, b, c, fields: Default::default() }
            }
            // CuTe's `MMA_Traits<SM90_64xNx16_*_SS>`: every thread of the
            // warpgroup sees the whole of A and B (descriptors), `ABLayout<M,
            // K> = (128, (M, K)) : (0, (1, M))`, and holds `N / 2` accumulators
            // in `CLayout_64xN = ((4, 8, 4), (2, 2, N / 8)) : ((128, 1, 16), (64,
            // 8, 512))`.
            //
            // With A in registers (`SM90_64xNxK_*_RS`), A is `ALayout_64x16 =
            // ((4, 8, 4), (2, 2, 2)) : ((128, 1, 16), (64, 8, 512))` for
            // 16-bit types and `ALayout_64x32 = ((4, 8, 4), (4, 2, 2)) : ((256,
            // 1, 16), (64, 8, 1024))` for 8-bit ones: four stacked `mma.sync`
            // A fragments, one 16-row slab per warp. 8-bit types have K = 32.
            MmaOp::Sm90(kind) => {
                let (n, k) = (kind.n as usize, kind.k());
                let a = match (kind.a_regs, k) {
                    (false, _) => Layout::new((128, (64, k)), (0, (1, 64))),
                    (true, 16) => Layout::new(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512))),
                    (true, _) => Layout::new(((4, 8, 4), (4, 2, 2)), ((256, 1, 16), (64, 8, 1024))),
                };
                MmaAtom {
                    op,
                    shape_mnk: [64, n, k],
                    thr_id: Layout::new(128, 1),
                    a,
                    b: Layout::new((128, (n, k)), (0, (1, n))),
                    c: Layout::new(((4, 8, 4), (2, 2, n / 8)), ((128, 1, 16), (64, 8, 512))),
                    fields: super::fields::MmaFields::taking(&[crate::MmaField::ACCUMULATE]),
                }
            }
            // CuTe's `MMA_Traits<SM100_MMA_F16BF16_SS>`: one "thread" (the
            // CTA), `ALayout = (1, (M, K)) : (0, (1, M))`, and the same for B
            // and C.
            // CuTe's `MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS>` (FP4,
            // packed) and `SM120_16x8x32_TN_VS` (8-bit containers: the layouts
            // of `SM80_16x8x32_S32S8S8S32_TN`).
            MmaOp::Sm120(kind) => {
                let (m, n, k) = kind.shape_mnk();
                let (a, b) = if kind.is_fp4_packed() {
                    (
                        Layout::new(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
                        Layout::new(((4, 8), (8, 2)), ((64, 1), (8, 256))),
                    )
                } else {
                    (
                        Layout::new(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
                        Layout::new(((4, 8), (4, 2)), ((32, 1), (8, 128))),
                    )
                };
                MmaAtom {
                    op,
                    shape_mnk: [m, n, k],
                    thr_id: Layout::new(32, 1),
                    a,
                    b,
                    c: Layout::new(((4, 8), (2, 2)), ((32, 1), (16, 8))),
                    fields: super::fields::MmaFields::taking(&[crate::MmaField::SFA, crate::MmaField::SFB]),
                }
            }
            MmaOp::Sm100(kind) => {
                let (m, n, k) = kind.shape_mnk();
                // A CTA pair (ThrID 2) holds the upper and lower halves of A's,
                // B's and C's rows (CUTLASS's SM100_MMA_*_2x1SM_SS traits).
                let v = if kind.two_cta { 2 } else { 1 };
                let tv = |rows: usize, cols: usize| {
                    let r = rows / v;
                    Layout::new((v, (r, cols)), (if v == 1 { 0 } else { r }, (1, rows)))
                };
                MmaAtom {
                    op,
                    shape_mnk: [m, n, k],
                    thr_id: Layout::new(v, if v == 1 { 0 } else { 1 }),
                    // A sparse A is partitioned compressed (half of K).
                    a: tv(m, kind.a_k()),
                    b: tv(n, k),
                    c: tv(m, n),
                    // The runtime negate bits and disable-output-lane mask;
                    // The scale factors instead of the mask;
                    // And the metadata.
                    fields: if kind.sparse {
                        super::fields::MmaFields::taking(&[
                            crate::MmaField::ACCUMULATE,
                            crate::MmaField::NEGATE_A,
                            crate::MmaField::NEGATE_B,
                            crate::MmaField::METADATA,
                            crate::MmaField::DISABLE_OUTPUT_LANE,
                        ])
                    } else if kind.scale.is_some() {
                        super::fields::MmaFields::taking(&[
                            crate::MmaField::ACCUMULATE,
                            crate::MmaField::NEGATE_A,
                            crate::MmaField::NEGATE_B,
                            crate::MmaField::SFA,
                            crate::MmaField::SFB,
                        ])
                    } else {
                        super::fields::MmaFields::taking(&[
                            crate::MmaField::ACCUMULATE,
                            crate::MmaField::NEGATE_A,
                            crate::MmaField::NEGATE_B,
                            crate::MmaField::DISABLE_OUTPUT_LANE,
                        ])
                    },
                }
            }
        }
    }

    /// Whether the atom is a tcgen05 MMA (sm_100), whose A and B are
    /// shared-memory descriptor fragments and whose accumulators are in
    /// tensor memory.
    pub fn is_tcgen05(&self) -> bool {
        matches!(self.op, MmaOp::Sm100(_))
    }

    /// Whether the atom is a warpgroup MMA (WGMMA), whose B (and A, unless
    /// it comes from registers) are shared-memory descriptor fragments.
    pub fn is_warpgroup(&self) -> bool {
        matches!(self.op, MmaOp::Sm90(_))
    }

    /// Whether the atom is a warpgroup MMA whose A comes from registers
    /// ([`OperandSource::RMEM`](crate::nvgpu::warpgroup::OperandSource::RMEM)).
    pub fn is_warpgroup_register_a(&self) -> bool {
        matches!(self.op, MmaOp::Sm90(kind) if kind.a_regs)
    }

    /// Whether the atom is a tcgen05 MMA whose A comes from tensor memory
    /// ([`OperandSource::TMEM`](crate::nvgpu::tcgen05::OperandSource::TMEM)).
    pub fn is_tcgen05_tmem_a(&self) -> bool {
        matches!(self.op, MmaOp::Sm100(kind) if kind.a_tmem)
    }

    /// Whether operand `which` of this atom is a descriptor fragment.
    fn takes_descriptor(&self, which: Operand) -> bool {
        (self.is_tcgen05() && which != Operand::C && !(which == Operand::A && self.is_tcgen05_tmem_a()))
            || self.is_warpgroup() && (which == Operand::B || which == Operand::A && !self.is_warpgroup_register_a())
    }

    /// The instruction this atom issues.
    pub fn op(&self) -> MmaOp {
        self.op
    }

    /// `[M, N, K]` of one instruction.
    pub fn shape_mnk(&self) -> [usize; 3] {
        self.shape_mnk
    }

    /// Threads that cooperate in one instruction (1, or a warp).
    pub fn num_threads(&self) -> usize {
        self.thr_id.size()
    }

    /// The atom's thread id layout (CuTe DSL's `thr_id`): `1:0` for one
    /// thread or a 1-CTA `tcgen05` MMA, `2:1` for a CTA pair's.
    pub fn thr_id(&self) -> &Layout {
        &self.thr_id
    }

    /// `(thread, value) -> (m, k)` index layout of the A operand.
    pub fn layout_a(&self) -> &Layout {
        &self.a
    }

    /// `(thread, value) -> (n, k)` index layout of the B operand.
    pub fn layout_b(&self) -> &Layout {
        &self.b
    }

    /// `(thread, value) -> (m, n)` index layout of the C/D operand.
    pub fn layout_c(&self) -> &Layout {
        &self.c
    }

    /// Element types of A, B and C/D.
    pub fn dtypes(&self) -> (DType, DType, DType) {
        match self.op {
            MmaOp::UniversalFma(t) => (t, t, t),
            MmaOp::Sm80(kind) => kind.dtypes(),
            MmaOp::Sm90(kind) => (kind.a, kind.b, kind.acc),
            MmaOp::Sm100(kind) => (kind.a, kind.b, kind.acc),
            MmaOp::Sm120(kind) => (kind.a, kind.b, DType::F32),
        }
    }

    /// Whether an A or B operand of type `ty` fits where the instruction
    /// takes `expected`: the type itself, or TFloat32 for the f32 operands of
    /// a TF32 MMA. A tcgen05 TF32 MMA also takes
    /// Float32 tensors, and FP6 and FP4 operands of a tcgen05 F8F6F4 MMA are
    /// Uint8 tensors holding the padded image (16 values per 16-byte slot,
    /// as a TMA load with an 8-bit `internal_type` writes it: CuTe C++'s
    /// `float_e2m1_unpacksmem_t` and its FP6 peers).
    fn takes_operand(&self, expected: DType, ty: DType) -> bool {
        ty == expected
            || ty == DType::TF32 && matches!(self.op, MmaOp::Sm80(kind) if kind.is_tf32())
            || match self.op {
                MmaOp::Sm100(kind) => match kind.kind {
                    crate::ir::UmmaDataKind::Tf32 => ty == DType::F32,
                    crate::ir::UmmaDataKind::F8F6F4 | crate::ir::UmmaDataKind::Mxf8f6f4 => {
                        expected.bits() < 8 && ty == DType::U8
                    }
                    _ => false,
                },
                _ => false,
            }
    }

    /// Whether the atom is a sparse `mma.sp` (or
    /// `tcgen05.mma.sp`), whose A operand is the compressed `M x K/2` tile.
    pub fn is_sparse(&self) -> bool {
        matches!(self.op, MmaOp::Sm80(kind) if kind.is_sparse()) || matches!(self.op, MmaOp::Sm100(kind) if kind.sparse)
    }

    fn operand(&self, which: Operand) -> (&Layout, usize, usize) {
        let [m, n, k] = self.shape_mnk;
        match which {
            Operand::A if self.is_sparse() => (&self.a, m, k / 2),
            Operand::A => (&self.a, m, k),
            Operand::B => (&self.b, n, k),
            Operand::C => (&self.c, m, n),
        }
    }
}

/// `make_mma_atom(op)`: `op` is an [`MmaOp`], [`MmaF16BF16Op`], [`MmaTF32Op`],
/// [`MmaFP8Op`] or [`MmaF16BF16SparseOp`].
pub fn make_mma_atom(op: impl Into<MmaOp>) -> MmaAtom {
    MmaAtom::new(op.into())
}

impl From<MmaOp> for MmaAtom {
    fn from(op: MmaOp) -> MmaAtom {
        MmaAtom::new(op)
    }
}

impl From<MmaKind> for MmaAtom {
    fn from(kind: MmaKind) -> MmaAtom {
        MmaAtom::new(MmaOp::Sm80(kind))
    }
}

impl From<MmaF16BF16Op> for MmaAtom {
    fn from(op: MmaF16BF16Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

impl From<MmaTF32Op> for MmaAtom {
    fn from(op: MmaTF32Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

/// An MMA atom repeated over a layout of atoms.
#[derive(Clone, Debug)]
pub struct TiledMma {
    pub(crate) atom: MmaAtom,
    /// `(atom_thread, atom_m, atom_n, atom_k) -> thread index`.
    thr_layout_vmnk: Layout,
    permutation: [Option<Layout>; 3],
}

/// Tile `atom` (an [`MmaAtom`] or any op [`make_mma_atom`] accepts) over
/// `atom_layout` (an `M x N x K` layout of atoms, e.g.
/// `Layout::col_major((2, 2, 1))` for four warps).
#[track_caller]
pub fn make_tiled_mma(atom: impl Into<MmaAtom>, atom_layout: Layout) -> TiledMma {
    atom_layout.require_static("make_tiled_mma: the atom layout", "an MMA is tiled over a static layout of atoms");
    let atom: MmaAtom = atom.into();
    let atom_layout = atom_layout.pad_to_rank(3);
    assert_eq!(atom_layout.rank(), 3, "atom layout must have rank 3 (M, N, K)");
    let t = atom.num_threads();
    let mut modes = vec![atom.thr_id.clone()];
    for i in 0..3 {
        let m = atom_layout.mode(i);
        let stride = m.static_stride().map(&mut |&d| d * t);
        modes.push(Layout::new(m.shape().clone(), stride));
    }
    TiledMma { atom, thr_layout_vmnk: Layout::from_modes(modes), permutation: [None, None, None] }
}

impl TiledMma {
    /// Override the tile size per mode with a permutation layout (CuTe's
    /// `permutation_mnk`), e.g. `Layout::new(32, 1)` to cover 32 rows of M
    /// with one 16-row atom per warp repeated twice.
    pub fn with_permutation_mnk(mut self, perm: [Option<Layout>; 3]) -> TiledMma {
        self.permutation = perm;
        self
    }

    /// The tiled atom.
    pub fn atom(&self) -> &MmaAtom {
        &self.atom
    }

    /// This tiling (thread layout, permutations and the atom's runtime
    /// fields) of the atom of `op` instead (the B
    /// collector variants of a Rubin tiled MMA).
    pub(crate) fn with_op(&self, op: MmaOp) -> TiledMma {
        let mut atom = MmaAtom::new(op);
        atom.fields = self.atom.fields.clone();
        TiledMma { atom, thr_layout_vmnk: self.thr_layout_vmnk.clone(), permutation: self.permutation.clone() }
    }

    /// The permutation layouts of M, N and K given with
    /// [`with_permutation_mnk`](Self::with_permutation_mnk) (CuTe DSL's
    /// `tiled_mma.permutation_mnk`), `None` where none was.
    pub fn permutation_mnk(&self) -> &[Option<Layout>; 3] {
        &self.permutation
    }

    /// Threads taking part in the tiled MMA.
    pub fn num_threads(&self) -> usize {
        self.thr_layout_vmnk.size()
    }

    /// The atom's thread id layout (CuTe DSL's `tiled_mma.thr_id`), as
    /// [`MmaAtom::thr_id`].
    pub fn thr_id(&self) -> &Layout {
        self.atom.thr_id()
    }

    /// `(atom thread, atom m, atom n, atom k) ->` thread index.
    pub fn thr_layout_vmnk(&self) -> &Layout {
        &self.thr_layout_vmnk
    }

    /// Tile size of mode `i` (0 = M, 1 = N, 2 = K).
    pub fn tile_size(&self, i: usize) -> usize {
        match &self.permutation[i] {
            Some(p) => p.size(),
            None => self.atom.shape_mnk[i] * self.thr_layout_vmnk.mode(i + 1).size(),
        }
    }

    fn modes_of(which: Operand) -> (usize, usize) {
        match which {
            Operand::A => (0, 2),
            Operand::B => (1, 2),
            Operand::C => (0, 1),
        }
    }

    /// `((ThrV, (ThrM, ThrK)), (FrgV, (RestM, RestK, ...)))` for A, and the
    /// equivalents for B and C (CuTe's `thrfrg_A/B/C`).
    pub(crate) fn thrfrg(&self, layout: &Layout, which: Operand) -> Layout {
        let (atom_tv, ai, aj) = self.atom.operand(which);
        self.thrfrg_with_atom(layout, which, &atom_tv.clone(), ai, aj)
    }

    /// [`thrfrg`](Self::thrfrg) with another `(thread, value)` atom layout
    /// over an `ai x aj` atom tile: the SM120 scale-factor partitions.
    pub(crate) fn thrfrg_with_atom(
        &self,
        layout: &Layout,
        which: Operand,
        atom_tv: &Layout,
        ai: usize,
        aj: usize,
    ) -> Layout {
        let (i, j) = Self::modes_of(which);
        let perm = |p: &Option<Layout>| Tree::Leaf(p.clone());
        let t = layout.logical_divide(&Tree::Branch(vec![perm(&self.permutation[i]), perm(&self.permutation[j])]));
        let a = t.zipped_divide(&(ai, aj).into());
        let tv = a.compose_tiler(&Tree::Branch(vec![Tree::Leaf(Some(atom_tv.clone())), Tree::Leaf(None)]));
        let thr_i = self.thr_layout_vmnk.mode(i + 1).size();
        let thr_j = self.thr_layout_vmnk.mode(j + 1).size();
        let thr_tile: Tiler = Tree::Branch(vec![
            Tree::Leaf(None),
            Tree::Branch(vec![Tree::Leaf(Some(Layout::new(thr_i, 1))), Tree::Leaf(Some(Layout::new(thr_j, 1)))]),
        ]);
        tv.zipped_divide(&thr_tile)
    }

    /// `(thread index, value) -> ` index into the operand tile, for building
    /// matching tiled copies (CuTe's `get_layoutA_TV` and friends).
    fn layout_tv(&self, which: Operand) -> Layout {
        let (i, j) = Self::modes_of(which);
        // A sparse A tile is compressed: half the K.
        let halve = if which == Operand::A && self.atom.is_sparse() { 2 } else { 1 };
        let reference = Layout::col_major((self.tile_size(i), self.tile_size(j) / halve));
        let frg = self.thrfrg(&reference, which);
        self.layout_tv_of(frg, which)
    }

    /// The `(thread index, value)` layout of the thread-fragment layout `frg`
    /// of operand `which` (also the SM120 scale-factor layouts).
    pub(crate) fn layout_tv_of(&self, frg: Layout, which: Operand) -> Layout {
        let (i, j) = Self::modes_of(which);
        // Map a (v, m, n, k) thread coordinate onto this operand's (v, i, j).
        let shape: Vec<usize> = (0..4).map(|d| self.thr_layout_vmnk.mode(d).size()).collect();
        let mut stride = vec![0; 4];
        stride[0] = 1;
        stride[i + 1] = shape[0];
        stride[j + 1] = shape[0] * shape[i + 1];
        let to_operand = Layout::new(shape, stride);
        let thread_map = to_operand.compose(&self.thr_layout_vmnk.right_inverse());
        let thr = frg.mode(0).compose(&thread_map);
        Layout::from_modes([thr, frg.mode(1)])
    }

    /// `(thread, value) ->` index into the `(tile_m, tile_k)` A tile.
    pub fn layout_a_tv(&self) -> Layout {
        self.layout_tv(Operand::A)
    }

    /// `(thread, value) ->` index into the `(tile_n, tile_k)` B tile.
    pub fn layout_b_tv(&self) -> Layout {
        self.layout_tv(Operand::B)
    }

    /// `(thread, value) ->` index into the `(tile_m, tile_n)` C tile.
    pub fn layout_c_tv(&self) -> Layout {
        self.layout_tv(Operand::C)
    }

    /// The descriptor fragment of a warpgroup MMA's operand `which` from its
    /// partition `p`.
    #[track_caller]
    fn make_desc<T: Numeric>(&self, p: &Tensor<T>, which: Operand, what: &str) -> Tensor<T> {
        use crate::nvgpu::OperandMajorMode::{K, MN};
        if let MmaOp::Sm100(kind) = self.atom.op {
            // Each CTA of a pair holds half of A's and of B's rows.
            let v = if kind.two_cta { 2 } else { 1 };
            let (mn, trans, name) = if which == Operand::A {
                (kind.m as usize / v, kind.trans_a, "a")
            } else {
                (kind.n as usize / v, kind.trans_b, "b")
            };
            let what = format!("{what}_{name} of {:?}", self.atom.op);
            // FP6/FP4 operands are read from their padded image
            // (except the packed FP4 of .kind::mxf4 and mxf4nvf4).
            let packed = matches!(kind.kind, crate::ir::UmmaDataKind::Mxf4 | crate::ir::UmmaDataKind::Mxf4Nvf4);
            assert!(
                packed || T::DTYPE.bits() >= 8,
                "cute: {what}: the operand is a packed {} tensor, but a tcgen05 MMA reads FP6 and FP4 operands from \
                 the padded shared-memory image (each 16 values in a 16-byte slot): hold it in a Uint8 tensor, as a \
                 TMA load with internal_type Uint8 writes it",
                T::DTYPE
            );
            let halve_k = kind.sparse && which == Operand::B;
            return crate::nvgpu::tcgen05::mma::make_umma_desc_fragment(
                p,
                mn,
                if trans { MN } else { K },
                &what,
                halve_k,
                kind.sm107.is_some(),
            );
        }
        let MmaOp::Sm90(kind) = self.atom.op else { unreachable!("a warpgroup MMA") };
        let (mn, trans, name) = match which {
            Operand::A => (64, kind.trans_a, "A"),
            _ => (kind.n as usize, kind.trans_b, "B"),
        };
        let what = format!("{what}_{} of {:?}", name.to_lowercase(), self.atom.op);
        crate::nvgpu::warpgroup::make_desc_fragment(p, mn, if trans { MN } else { K }, &what)
    }

    /// The A fragment of `input`, this thread's partition of A
    /// ([`ThrMma::partition_a`]: `(MMA, MMA_M, MMA_K, ...)`), for
    /// [`gemm`] (CuTe DSL's `tiled_mma.make_fragment_A(input)`): for a
    /// warpgroup MMA, whose A is in shared memory, a tensor of shared-memory
    /// matrix descriptors, one per MMA tile (`(1, MMA_M, MMA_K, ...)`);
    /// otherwise (including a warpgroup MMA with A in registers) a register
    /// fragment shaped like `input`, whose shape must be static.
    #[track_caller]
    pub fn make_fragment_a<T: Numeric>(&self, input: &Tensor<T>) -> Tensor<T> {
        self.make_fragment_ab(input, Operand::A)
    }

    /// The B fragment of `input` (`(MMA, MMA_N, MMA_K, ...)`), as
    /// [`make_fragment_a`](Self::make_fragment_a) for A.
    #[track_caller]
    pub fn make_fragment_b<T: Numeric>(&self, input: &Tensor<T>) -> Tensor<T> {
        self.make_fragment_ab(input, Operand::B)
    }

    #[track_caller]
    fn make_fragment_ab<T: Numeric>(&self, input: &Tensor<T>, which: Operand) -> Tensor<T> {
        let (a, b, _) = self.atom.dtypes();
        let want = if which == Operand::A { a } else { b };
        assert!(
            self.atom.takes_operand(want, T::DTYPE),
            "{which:?} fragments of {:?} hold {want}, not {}",
            self.atom.op,
            T::DTYPE
        );
        if self.atom.takes_descriptor(which) {
            return self.make_desc(input, which, "make_fragment");
        }
        if which == Operand::A && self.atom.is_tcgen05_tmem_a() {
            return self.make_tmem_fragment_a(input);
        }
        let what = if which == Operand::A { "make_fragment_a" } else { "make_fragment_b" };
        require_static_fragment_shape(input, what);
        let frag = make_compact_fragment_like::<T, T>(input);
        if self.atom.is_warpgroup() {
            crate::nvgpu::warpgroup::mark_a_fragment(frag.register_storage().expect("a register fragment"), T::DTYPE);
        }
        frag
    }

    /// The accumulator fragment of `input` (CuTe DSL's
    /// `tiled_mma.make_fragment_C(input)`): a register fragment of the
    /// atom's accumulator type `T`, shaped like `input` (a partition of C,
    /// `(MMA, MMA_M, MMA_N, ...)`, or a [`Layout`] or shape), which must be
    /// static. The accumulators of a warpgroup MMA are fenced by
    /// [`warpgroup::fence`](crate::nvgpu::warpgroup::fence) and
    /// [`wait_group`](crate::nvgpu::warpgroup::wait_group).
    #[track_caller]
    pub fn make_fragment_c<T: Numeric>(&self, input: &impl MmaFragmentInput) -> Tensor<T> {
        let (_, _, c) = self.atom.dtypes();
        assert!(T::DTYPE == c, "C fragments of {:?} hold {c}, not {}", self.atom.op, T::DTYPE);
        if self.atom.is_tcgen05() {
            return self.make_tmem_fragment::<T>(&input.fragment_layout("make_fragment_c"), "make_fragment_c");
        }
        let frag = crate::tensor::make_fragment_with_layout::<T>(input.fragment_layout("make_fragment_c"));
        if self.atom.is_warpgroup() {
            crate::nvgpu::warpgroup::mark_accumulators(frag.register_storage().expect("a register fragment"), T::DTYPE);
        }
        frag
    }

    /// The TMEM A tensor of a tcgen05 MMA with A in tensor memory (CuTe
    /// DSL's `make_fragment_A(shape)` with `a_src` TMEM), shaped like `input` (`((M, K), MMA_M, MMA_K, ...)`,
    /// a partition of A, a [`Layout`] or a shape): CuTe C++'s
    /// `UMMA::tmem_frg_1sm<T, T, NonInterleaved>` layout over a TMEM pointer
    /// at address 0. Row `i` of each MMA's A tile is in the lane of row `i`
    /// of D (the PTX ISA's data-path layouts) and its K elements are packed
    /// into consecutive 32-bit columns (two 16-bit or four 8-bit values per
    /// column); the MMA tiles follow each other in columns. Put the layout
    /// over the TMEM columns that hold A (`make_tensor(a_ptr,
    /// frag.layout())`), write them with `tcgen05.st` and wait for the stores
    /// (`fence_view_async_tmem_store`) before the MMA.
    #[track_caller]
    pub fn make_tmem_fragment_a<T: Numeric>(&self, input: &impl MmaFragmentInput) -> Tensor<T> {
        let what = "make_fragment_a";
        let MmaOp::Sm100(kind) = self.atom.op else {
            panic!("cute: {what}: {:?} is not a tcgen05 MMA with A in tensor memory", self.atom.op)
        };
        assert!(kind.a_tmem, "cute: {what}: {} takes A from shared memory (a_src=OperandSource.SMEM)", kind.ptx());
        assert!(
            self.atom.takes_operand(kind.a, T::DTYPE) && T::DTYPE.bits() >= 8,
            "A fragments of {:?} hold {}, not {}",
            self.atom.op,
            kind.a,
            T::DTYPE
        );
        let shape = input.fragment_layout(what);
        let (m, k) = (kind.m as usize, kind.k());
        assert!(
            shape.mode(0).size() == m * k,
            "cute: {what} of {:?}: the shape {} is not ((M, K), MMA_M, MMA_K, ...) for the {m} x {k} A tile (use \
             partition_shape_a)",
            self.atom.op,
            shape.shape()
        );
        let rest: Vec<usize> = (1..shape.rank()).map(|i| shape.mode(i).size()).collect();
        let rest = if rest.is_empty() { vec![1] } else { rest };
        let layout = crate::nvgpu::tcgen05::mma::tmem_frg_a_layout(m, k, &rest, T::DTYPE.storage_bits());
        let ptr = crate::make_ptr::<T>(crate::value::Int32::from(0), crate::tensor::AddressSpace::Tmem, Some(16));
        crate::make_tensor(ptr, layout)
    }

    /// The TMEM accumulator tensor of a tcgen05 MMA shaped like `shape`
    /// (`((M, N), MMA_M, MMA_N, ...)`): CuTe C++'s `UMMA::tmem_frg_1sm`
    /// layout over a TMEM pointer at address 0, whose layout a kernel puts
    /// over the pointer of its TMEM allocation (`make_tensor(tmem_ptr,
    /// frag.layout())`), as CuTe DSL kernels do.
    #[track_caller]
    fn make_tmem_fragment<T: Numeric>(&self, shape: &Layout, what: &str) -> Tensor<T> {
        let MmaOp::Sm100(kind) = self.atom.op else { unreachable!("a tcgen05 MMA") };
        // One CTA's half of a pair's accumulator, M / 2 rows.
        let m = kind.m as usize / if kind.two_cta { 2 } else { 1 };
        let n = kind.n as usize;
        let first = shape.mode(0);
        assert!(
            first.size() == m * n,
            "cute: {what} of {:?}: the shape {} is not ((M, N), MMA_M, MMA_N, ...) for the {m} x {n} instruction              (use partition_shape_c)",
            self.atom.op,
            shape.shape()
        );
        let rest: Vec<usize> = (1..shape.rank()).map(|i| shape.mode(i).size()).collect();
        let rest = if rest.is_empty() { vec![1] } else { rest };
        let bits = T::DTYPE.storage_bits();
        let layout = if kind.two_cta {
            crate::nvgpu::tcgen05::mma::tmem_frg_2sm_layout(m, n, &rest, bits)
        } else {
            crate::nvgpu::tcgen05::mma::tmem_frg_1sm_layout(m, n, &rest, bits)
        };
        let ptr = crate::make_ptr::<T>(crate::value::Int32::from(0), crate::tensor::AddressSpace::Tmem, Some(16));
        crate::make_tensor(ptr, layout)
    }

    /// The shape of thread 0's partition of an `(M, K)` A tile of shape
    /// `shape` (CuTe DSL's `tiled_mma.partition_shape_A`): `(MMA, MMA_M,
    /// MMA_K, ...)`.
    #[track_caller]
    pub fn partition_shape_a(&self, shape: impl Into<Shape>) -> Shape {
        self.partition_shape(shape.into(), Operand::A)
    }

    /// The shape of thread 0's partition of an `(N, K)` B tile
    /// (`partition_shape_B`).
    #[track_caller]
    pub fn partition_shape_b(&self, shape: impl Into<Shape>) -> Shape {
        self.partition_shape(shape.into(), Operand::B)
    }

    /// The shape of thread 0's partition of an `(M, N)` C tile
    /// (`partition_shape_C`): for a tcgen05 MMA, `((M, N), MMA_M, MMA_N)`,
    /// what [`make_fragment_c`](Self::make_fragment_c) takes.
    #[track_caller]
    pub fn partition_shape_c(&self, shape: impl Into<Shape>) -> Shape {
        self.partition_shape(shape.into(), Operand::C)
    }

    #[track_caller]
    fn partition_shape(&self, shape: Shape, which: Operand) -> Shape {
        let t = crate::tensor::make_identity_tensor(shape);
        self.get_slice(0).partition(&t, which).shape().clone()
    }

    /// Thread `thr_idx`'s view.
    pub fn get_slice(&self, thr_idx: impl Into<IndexExpr>) -> ThrMma {
        let coord = self.thr_layout_vmnk.coord_of(thr_idx);
        ThrMma { mma: self.clone(), thr_vmnk: coord }
    }
}

/// One thread's slice of a [`TiledMma`].
#[derive(Clone, Debug)]
pub struct ThrMma {
    mma: TiledMma,
    thr_vmnk: Coord,
}

impl ThrMma {
    /// The tiled MMA this is a slice of.
    pub fn tiled_mma(&self) -> &TiledMma {
        &self.mma
    }

    #[track_caller]
    fn partition<U>(&self, t: &Tensor<U>, which: Operand) -> Tensor<U> {
        let mma = self.mma.clone();
        let layout = mma.thrfrg(t.layout(), which);
        let viewed = crate::layout::Tileable::map_layout(t, layout, move |l| mma.thrfrg(l, which));
        let coord = self.thrfrg_coord(viewed.layout(), which);
        viewed.slice(coord)
    }

    /// This thread's slice of `t` viewed through the thread-fragment layout
    /// `l` of operand `which` (`((ThrV, (ThrI, ThrJ)), (FrgV, Rest...))`).
    #[track_caller]
    pub(crate) fn slice_thrfrg<U>(&self, t: &Tensor<U>, l: Layout, which: Operand) -> Tensor<U> {
        let coord = self.thrfrg_coord(&l, which);
        crate::layout::Tileable::with_layout(t, l).slice(coord)
    }

    /// The slice of this thread in a thread-fragment layout `l` of operand
    /// `which`.
    fn thrfrg_coord(&self, l: &Layout, which: Operand) -> crate::layout::SliceCoord {
        let (i, j) = TiledMma::modes_of(which);
        let rest = l.get(&[1, 1]).rank();
        let c = |d: usize| Tree::Leaf(Some(self.thr_vmnk.mode(d).as_leaf().unwrap().clone()));
        Tree::Branch(vec![
            Tree::Branch(vec![c(0), Tree::Branch(vec![c(i + 1), c(j + 1)])]),
            Tree::Branch(vec![Tree::Leaf(None), Tree::Branch(vec![Tree::Leaf(None); rest])]),
        ])
    }

    /// This thread's part of an `(M, K, ...)` tile: `(MMA, MMA_M, MMA_K, ...)`.
    /// The tensor may have runtime extents, which the modes after `MMA`
    /// then have.
    #[track_caller]
    pub fn partition_a<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::A)
    }

    /// This thread's part of an `(N, K, ...)` tile: `(MMA, MMA_N, MMA_K, ...)`.
    #[track_caller]
    pub fn partition_b<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::B)
    }

    /// This thread's part of an `(M, N, ...)` tile: `(MMA, MMA_M, MMA_N, ...)`.
    #[track_caller]
    pub fn partition_c<U>(&self, t: &Tensor<U>) -> Tensor<U> {
        self.partition(t, Operand::C)
    }

    #[track_caller]
    fn fragment<T: Numeric, U>(&self, t: &Tensor<U>, which: Operand) -> Tensor<T> {
        if which == Operand::C && self.mma.atom.is_tcgen05() {
            let (_, _, c) = self.mma.atom.dtypes();
            assert!(T::DTYPE == c, "C fragments of {:?} hold {c}, not {}", self.mma.atom.op, T::DTYPE);
            let p = self.partition(t, which);
            return self.mma.make_tmem_fragment::<T>(&Layout::col_major(p.shape().clone()), "partition_fragment_c");
        }
        if self.mma.atom.takes_descriptor(which) {
            // A warpgroup MMA's A and B are descriptors of the operand in
            // shared memory, which must hold the operand's type.
            let (a, b, _) = self.mma.atom.dtypes();
            let want = if which == Operand::A { a } else { b };
            assert!(
                T::DTYPE == want,
                "{which:?} fragments of {:?} describe {want} operands, not {}",
                self.mma.atom.op,
                T::DTYPE
            );
            let p = self.partition(&t.retyped::<T>(), which);
            return self.mma.make_desc(&p, which, "partition_fragment");
        }
        let (a, b, c) = self.mma.atom.dtypes();
        let expected = match which {
            Operand::A => a,
            Operand::B => b,
            Operand::C => c,
        };
        let fits =
            if which == Operand::C { T::DTYPE == expected } else { self.mma.atom.takes_operand(expected, T::DTYPE) };
        assert!(fits, "{which:?} fragments of {:?} hold {expected}, not {}", self.mma.atom.op, T::DTYPE);
        let p = self.partition(t, which);
        let what = match which {
            Operand::A => "partition_fragment_a",
            Operand::B => "partition_fragment_b",
            Operand::C => "partition_fragment_c",
        };
        require_static_fragment_shape(&p, what);
        let frag = make_compact_fragment_like::<T, U>(&p);
        if which == Operand::C && self.mma.atom.is_warpgroup() {
            crate::nvgpu::warpgroup::mark_accumulators(frag.register_storage().expect("a register fragment"), T::DTYPE);
        }
        if which == Operand::A && self.mma.atom.is_warpgroup() {
            crate::nvgpu::warpgroup::mark_a_fragment(frag.register_storage().expect("a register fragment"), T::DTYPE);
        }
        frag
    }

    /// A register fragment for this thread's A operand of tile `t`, which
    /// must have static extents.
    #[track_caller]
    pub fn partition_fragment_a<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::A)
    }

    /// A register fragment for this thread's B operand of tile `t`, which
    /// must have static extents.
    #[track_caller]
    pub fn partition_fragment_b<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::B)
    }

    /// A Uint32 register fragment for this thread's sparse-MMA metadata over
    /// the compressed A tile `t` (`(M, K/2, ...)`, static): `(1, MMA_M,
    /// MMA_K, ...)`, one register per MMA, for [`gemm_sparse`].
    #[track_caller]
    pub fn partition_fragment_e<U>(&self, t: &Tensor<U>) -> Tensor<u32> {
        assert!(self.mma.atom.is_sparse(), "partition_fragment_e: {:?} is not a sparse MMA", self.mma.atom.op);
        let p = self.partition(t, Operand::A);
        require_static_fragment_shape(&p, "partition_fragment_e");
        let rest: Vec<usize> = (1..p.shape().rank()).map(|i| p.shape().mode(i).size()).collect();
        let shape: Vec<usize> = std::iter::once(1).chain(rest).collect();
        crate::tensor::make_fragment::<u32>(shape)
    }

    /// A register fragment for this thread's accumulators of tile `t`,
    /// which must have static extents.
    #[track_caller]
    pub fn partition_fragment_c<T: Numeric, U>(&self, t: &Tensor<U>) -> Tensor<T> {
        self.fragment(t, Operand::C)
    }
}

/// `d = a * b + c` over per-thread fragments: `a` is `(MMA, M, K)`, `b` is
/// `(MMA, N, K)`, `c` and `d` are `(MMA, M, N)`. `d` may be `c`. The MMAs
/// are unrolled when the kernel is built, so the extents of the operands
/// must be static.
///
/// `a` and `b` are fragments (`&Tensor`), or for a block-scaled MMA
/// ([`warp::MmaMXF4Op`](crate::warp::MmaMXF4Op) and the others) each a
/// fragment with its scale factors, `(&a, &sfa)` and `(&b, &sfb)` (CuTe
/// DSL's `[a, sfa]` operand lists; see [`gemm_block_scaled`]).
#[track_caller]
pub fn gemm<A: GemmOperand, B: GemmOperand, TC: Numeric>(mma: &TiledMma, d: &Tensor<TC>, a: A, b: B, c: &Tensor<TC>) {
    let ((a, sfa), (b, sfb)) = (a.split(), b.split());
    if let MmaOp::Sm120(_) = mma.atom.op {
        return gemm_block_scaled(mma, d, &a, sfa, &b, sfb, c);
    }
    // A's companion is the metadata.
    if let MmaOp::Sm100(kind) = mma.atom.op
        && (kind.scale.is_some() || kind.sparse)
    {
        return gemm_tcgen05(mma, d, &a, &b, c, (sfa, sfb));
    }
    assert!(
        sfa.is_none() && sfb.is_none(),
        "gemm: {:?} takes no scale factors: pass the A and B fragments alone",
        mma.atom.op
    );
    let (a, b) = (&a, &b);
    assert!(
        !mma.atom.is_sparse(),
        "gemm: {:?} is a sparse MMA, which takes a metadata operand: call gemm_sparse",
        mma.atom.op
    );
    if mma.atom.is_warpgroup() {
        return gemm_warpgroup(mma, d, a, b, c);
    }
    if mma.atom.is_tcgen05() {
        return gemm_tcgen05(mma, d, a, b, c, (None, None));
    }
    gemm_impl(mma, d, a, None, b, c)
}

/// The shape of a [`TiledMma::make_fragment_c`] input: a tensor (a
/// partition of C), a [`Layout`] or a shape.
pub trait MmaFragmentInput {
    #[doc(hidden)]
    fn fragment_layout(&self, what: &str) -> Layout;
}

impl<U> MmaFragmentInput for Tensor<U> {
    #[track_caller]
    fn fragment_layout(&self, what: &str) -> Layout {
        require_static_fragment_shape(self, what);
        crate::layout::make_layout_like(self.layout())
    }
}

impl MmaFragmentInput for Layout {
    #[track_caller]
    fn fragment_layout(&self, what: &str) -> Layout {
        self.require_static(what, "register fragments have static shapes");
        Layout::col_major(self.shape().clone())
    }
}

impl MmaFragmentInput for Shape {
    #[track_caller]
    fn fragment_layout(&self, what: &str) -> Layout {
        let l = Layout::col_major(self.clone());
        l.require_static(what, "register fragments have static shapes");
        l
    }
}

/// [`gemm`] for a warpgroup MMA: `a` and `b` are descriptor fragments
/// (`(1, MMA_M, MMA_K)` and `(1, MMA_N, MMA_K)`; with A in registers, `a`
/// is a register fragment `(8 or 16, MMA_M, MMA_K)`), `c` and `d` accumulator
/// fragments (`(MMA, MMA_M, MMA_N)`). Each MMA takes the atom's
/// `ACCUMULATE` field as it is (as CuTe's `gemm` does: a kernel sets it to
/// true after its first k-block). Accumulators never written are given as
/// zeros, which an MMA that does not accumulate ignores.
#[track_caller]
fn gemm_warpgroup<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    b: &Tensor<TB>,
    c: &Tensor<TC>,
) {
    let loc = Location::caller();
    let MmaOp::Sm90(kind) = mma.atom.op else { unreachable!("a warpgroup MMA") };
    let (ta, tb, tc) = mma.atom.dtypes();
    assert!(
        TA::DTYPE == ta && TB::DTYPE == tb && TC::DTYPE == tc,
        "gemm operand types do not match {:?}: {:?} given, {:?} expected",
        mma.atom.op,
        (TA::DTYPE, TB::DTYPE, TC::DTYPE),
        (ta, tb, tc)
    );
    let a_regs = kind.a_regs;
    if a_regs {
        assert!(
            a.is_register(),
            "gemm: operand A of the warpgroup MMA {:?} comes from registers (OperandSource::RMEM): pass the register \
             fragment TiledMma::make_fragment_a (or ThrMma::partition_fragment_a) makes, filled before \
             warpgroup::fence",
            mma.atom.op
        );
        crate::nvgpu::warpgroup::mark_a_fragment(a.register_storage().expect("a register fragment"), TA::DTYPE);
    }
    let descs: &[(&str, bool)] =
        if a_regs { &[("B", b.engine_is_desc())] } else { &[("A", a.engine_is_desc()), ("B", b.engine_is_desc())] };
    for &(name, t) in descs {
        assert!(
            t,
            "gemm: operand {name} of the warpgroup MMA {:?} must be the descriptor fragment that \
             TiledMma::make_fragment_{} (or ThrMma::partition_fragment_{}) makes of its shared-memory tile",
            mma.atom.op,
            name.to_lowercase(),
            name.to_lowercase()
        );
    }
    for (name, t) in [("D", d), ("C", c)] {
        assert!(t.is_register(), "gemm: the accumulators {name} of a warpgroup MMA are a register fragment");
        require_static_gemm_operand(name, t.layout());
    }
    for (name, t) in [("A", a.layout()), ("B", b.layout())] {
        require_static_gemm_operand(name, t);
    }
    let size = |t: &Shape, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
    let (m, n, k) = (size(c.shape(), 1), size(c.shape(), 2), size(a.shape(), 2));
    assert_eq!(size(a.shape(), 1), m, "gemm: A has {} M-blocks but C has {m}", size(a.shape(), 1));
    assert_eq!(size(b.shape(), 1), n, "gemm: B has {} N-blocks but C has {n}", size(b.shape(), 1));
    assert_eq!(size(b.shape(), 2), k, "gemm: A has {k} K-blocks but B has {}", size(b.shape(), 2));
    let acc = kind.accumulators();
    assert!(
        size(c.shape(), 0) == acc && size(d.shape(), 0) == acc,
        "gemm: each thread holds {acc} accumulators of {}, but C is {} and D {}",
        kind.ptx(),
        c.layout(),
        d.layout()
    );
    for t in [c, d] {
        crate::nvgpu::warpgroup::mark_accumulators(t.register_storage().expect("a register fragment"), TC::DTYPE);
    }
    let scale_d = mma
        .field_values()
        .into_iter()
        .find(|(f, _)| *f == crate::MmaField::ACCUMULATE)
        .and_then(|(_, v)| v)
        .unwrap_or_else(|| crate::value::IntoValue::into_value(false))
        .checked();
    let at = |t: &Shape, idx: [usize; 2]| -> crate::layout::SliceCoord {
        let mut modes = vec![Tree::Leaf(None)];
        for (i, &x) in idx.iter().enumerate() {
            if i + 1 < t.rank() {
                modes.push(Tree::Leaf(Some(IndexExpr::Static(x))));
            }
        }
        Tree::Branch(modes)
    };
    let frag_a = kind.a_fragment();
    if a_regs {
        assert!(
            size(a.shape(), 0) == frag_a,
            "gemm: each thread holds {frag_a} A elements of {} in registers, but A is {}",
            kind.ptx(),
            a.layout()
        );
    }
    for kk in 0..k {
        // A: a descriptor per M-block, or the register fragment's elements.
        let da: Vec<Vec<_>> = (0..m)
            .map(|mm| {
                let s = a.slice(at(a.shape(), [mm, kk]));
                if a_regs {
                    let storage = s.register_storage().expect("a register fragment").clone();
                    (0..frag_a)
                        .map(|i| {
                            let slot = s.element_offset(&i.into()).as_static().expect("static A index");
                            storage.read(slot).unwrap_or_else(|| {
                                panic!(
                                    "gemm: register {slot} of the A fragment of the warpgroup MMA {:?} was never \
                                     written: fill the fragment (before warpgroup::fence) first",
                                    mma.atom.op
                                )
                            })
                        })
                        .collect()
                } else {
                    vec![s.smem_desc(0).checked()]
                }
            })
            .collect();
        let db: Vec<_> = (0..n).map(|nn| b.slice(at(b.shape(), [nn, kk])).smem_desc(0).checked()).collect();
        for (mm, a_operand) in da.iter().enumerate() {
            for step in 0..n {
                let nn = if mm % 2 == 0 { step } else { n - 1 - step };
                let cs = c.slice(at(c.shape(), [mm, nn]));
                let storage = cs.register_storage().expect("a register fragment").clone();
                let cv: Vec<_> = (0..acc)
                    .map(|i| {
                        let slot = cs.element_offset(&i.into()).as_static().expect("static accumulator index");
                        storage.read(slot).unwrap_or_else(|| {
                            builder::with(|bs| bs.constant(crate::ir::Literal::zero(TC::DTYPE), loc))
                        })
                    })
                    .collect();
                let dv = builder::with(|bs| {
                    let dv: Vec<_> = (0..acc).map(|_| bs.new_value(TC::DTYPE)).collect();
                    let op = Op::Wgmma { kind, d: dv.clone(), c: cv, a: a_operand.clone(), desc_b: db[nn], scale_d };
                    bs.push(op, loc);
                    dv
                });
                crate::nvgpu::warpgroup::issued(&dv, if a_regs { a_operand } else { &[] });
                let ds = d.slice(at(d.shape(), [mm, nn]));
                for (i, v) in dv.into_iter().enumerate() {
                    ds.store(i, Value::<TC>::from_id(v));
                }
            }
        }
    }
}

/// [`gemm`] for a tcgen05 MMA: `a` and `b`
/// are descriptor fragments (`(1, MMA_M, MMA_K)` and `(1, MMA_N, MMA_K)`),
/// `d` the TMEM accumulator tensor (`((M, N), MMA_M, MMA_N)`, from
/// [`TiledMma::make_fragment_c`] over the TMEM allocation), which `c` must
/// be too (tcgen05 MMAs accumulate in place). Called by a whole warp: it
/// elects one lane (`elect.sync`), which issues each `tcgen05.mma` (CuTe
/// DSL: issue `cute.gemm` warp-uniformly, not inside `elect_one`). Each MMA
/// takes the atom's `ACCUMULATE` field as it is (false: D = A B).
#[track_caller]
fn gemm_tcgen05<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    b: &Tensor<TB>,
    c: &Tensor<TC>,
    (sfa, sfb): (Option<ScaleOperand>, Option<ScaleOperand>),
) {
    let loc = Location::caller();
    let MmaOp::Sm100(kind) = mma.atom.op else { unreachable!("a tcgen05 MMA") };
    // No published source gives the descriptor format of UE5M3 operands.
    assert!(
        kind.a != DType::F8E5M3FNU && kind.b != DType::F8E5M3FNU,
        "gemm: {} with a FloatNV8E5M3FNU operand: cute-rs does not know that operand's instruction-descriptor \
         format code (no published source gives it), so it cannot issue this MMA",
        kind.ptx()
    );
    let (ta, tb, tc) = mma.atom.dtypes();
    assert!(
        mma.atom.takes_operand(ta, TA::DTYPE) && mma.atom.takes_operand(tb, TB::DTYPE) && TC::DTYPE == tc,
        "gemm operand types do not match {:?}: {:?} given, {:?} expected",
        mma.atom.op,
        (TA::DTYPE, TB::DTYPE, TC::DTYPE),
        (ta, tb, tc)
    );
    let tmem_a = kind.a_tmem;
    if tmem_a {
        assert!(
            a.memory_space() == Some(crate::ir::MemSpace::Raw(crate::tensor::AddressSpace::Tmem)),
            "gemm: operand A of the tcgen05 MMA {:?} comes from tensor memory (a_src=OperandSource.TMEM): pass the \
             TMEM tensor make_fragment_a's layout gives over the TMEM columns holding A",
            mma.atom.op
        );
    }
    for (name, t) in [("A", tmem_a || a.engine_is_desc()), ("B", b.engine_is_desc())] {
        assert!(
            t,
            "gemm: operand {name} of the tcgen05 MMA {:?} must be the descriptor fragment that TiledMma::make_fragment_{} makes of its shared-memory tile",
            mma.atom.op,
            name.to_lowercase()
        );
    }
    let tmem = |t: &Tensor<TC>| t.memory_space() == Some(crate::ir::MemSpace::Raw(crate::tensor::AddressSpace::Tmem));
    assert!(
        tmem(d) && tmem(c),
        "gemm: the accumulators of a tcgen05 MMA are in tensor memory: pass the TMEM tensor make_fragment_c's layout gives over the TMEM allocation as C and D"
    );
    assert!(d.layout() == c.layout(), "gemm: a tcgen05 MMA accumulates in place, so C and D are the same TMEM tensor");
    for (name, t) in [("D", d.layout()), ("A", a.layout()), ("B", b.layout())] {
        require_static_gemm_operand(name, t);
    }
    let size = |t: &Shape, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
    let (m, n, k) = (size(d.shape(), 1), size(d.shape(), 2), size(a.shape(), 2));
    assert_eq!(size(a.shape(), 1), m, "gemm: A has {} M-blocks but D has {m}", size(a.shape(), 1));
    assert_eq!(size(b.shape(), 1), n, "gemm: B has {} N-blocks but D has {n}", size(b.shape(), 1));
    assert_eq!(size(b.shape(), 2), k, "gemm: A has {k} K-blocks but B has {}", size(b.shape(), 2));
    // Each CTA of a pair holds M / 2 rows of D.
    let rows = kind.m / if kind.two_cta { 2 } else { 1 };
    assert_eq!(
        size(d.shape(), 0),
        (rows * kind.n) as usize,
        "gemm: D's first mode {} is not the {} x {} accumulator tile of {}{}",
        d.layout(),
        rows,
        kind.n,
        kind.ptx(),
        if kind.two_cta { " in each CTA of the pair" } else { "" }
    );
    let enable = mma
        .field_values()
        .into_iter()
        .find(|(f, _)| *f == crate::MmaField::ACCUMULATE)
        .and_then(|(_, v)| v)
        .unwrap_or_else(|| crate::value::IntoValue::into_value(false))
        .checked();
    let at = |t: &Shape, idx: [usize; 2]| -> crate::layout::SliceCoord {
        let mut modes = vec![Tree::Leaf(None)];
        for (i, &x) in idx.iter().enumerate() {
            if i + 1 < t.rank() {
                modes.push(Tree::Leaf(Some(IndexExpr::Static(x))));
            }
        }
        Tree::Branch(modes)
    };
    // The runtime negate bits OR'd into the descriptor, and the lane mask.
    let field = |f: crate::MmaField| mma.field_values().into_iter().find(|(g, _)| *g == f).and_then(|(_, v)| v);
    let (neg_a, neg_b) = (field(crate::MmaField::NEGATE_A), field(crate::MmaField::NEGATE_B));
    let negated = (neg_a.is_some() || neg_b.is_some()).then(|| {
        let mut v: Value<u32> = crate::value::IntoValue::into_value(kind.idesc());
        for (neg, bit) in [(neg_a, crate::ir::UmmaKind::NEGATE_A_BIT), (neg_b, crate::ir::UmmaKind::NEGATE_B_BIT)] {
            if let Some(b) = neg {
                v = v | (b.to::<u32>() << bit);
            }
        }
        v
    });
    let idesc = if kind.scale.is_some() { None } else { negated.map(|v| v.checked()) };
    // The metadata's TMEM tensor (A's companion, or the METADATA field).
    let metadata = kind.sparse.then(|| {
        let Some(e) = sfa.clone().or_else(|| mma.atom.fields.tensors[0].clone()) else {
            panic!(
                "gemm: the sparse MMA {} reads A's sparsity metadata from tensor memory, and none was given: pass \
                 (&a, &e) as A, or set the atom's METADATA field with set_tensor",
                kind.ptx()
            )
        };
        assert!(
            e.tensor.memory_space() == Some(crate::ir::MemSpace::Raw(crate::tensor::AddressSpace::Tmem)),
            "gemm: the metadata of the sparse tcgen05 MMA {} is read from tensor memory: pass a TMEM tensor \
             (make_sparse_tmem_layout_e over the metadata's columns), not {:?}",
            kind.ptx(),
            e.tensor
        );
        e
    });
    if !kind.sparse && kind.scale.is_none() {
        assert!(sfa.is_none() && sfb.is_none(), "gemm: {} takes no scale factors or metadata", kind.ptx());
    }
    if kind.sparse {
        assert!(sfb.is_none(), "gemm: the sparse MMA {} takes no companion tensor with B", kind.ptx());
    }
    // The scale factors' TMEM tensors.
    let scales = kind.scale.map(|sc| {
        let field = |i: usize| mma.atom.fields.tensors[i].clone();
        let (Some(sfa), Some(sfb)) = (sfa.or_else(|| field(0)), sfb.or_else(|| field(1))) else {
            panic!(
                "gemm: the block-scaled MMA {} needs scale factors in tensor memory for A and B: pass (&a, &sfa) and \
                 (&b, &sfb), or set the atom's SFA and SFB fields with set_tensor",
                kind.ptx()
            )
        };
        for (name, s) in [("SFA", &sfa), ("SFB", &sfb)] {
            assert!(
                s.dtype == sc.sf,
                "gemm: {} scales each {} K elements by a {} scale factor, but {name} holds {}",
                kind.ptx(),
                sc.vec,
                sc.sf,
                s.dtype
            );
            assert!(
                s.tensor.memory_space() == Some(crate::ir::MemSpace::Raw(crate::tensor::AddressSpace::Tmem)),
                "gemm: {name} of the block-scaled tcgen05 MMA {} is read from tensor memory: pass a TMEM tensor \
                 (tmem_sf_frg_layout over the scale factors' columns), not {:?}",
                kind.ptx(),
                s.tensor
            );
        }
        (sfa, sfb)
    });
    let disable_lane: Vec<_> = mma.atom.disable_output_lane().map(<[_]>::to_vec).unwrap_or_default();
    let elect = crate::arch::elect_one().checked();
    // The TMEM address and the scale-factor ID (the byte within the column) of a scale-factor
    // tensor at the MMA's (MN, K) block: sliced when the tensor has those modes.
    let sf_operand = |s: &ScaleOperand, mn: usize, kk: usize| -> (crate::ir::ValueId, Value<u32>) {
        let t = &s.tensor;
        let t = if t.shape().rank() >= 3 { t.slice(at(t.shape(), [mn, kk])) } else { t.clone() };
        let p = t.ptr();
        let id = crate::layout::IndexExpr::and(p.offset.clone(), crate::layout::IndexExpr::Static(3));
        let id: Value<u32> = Value::<i32>::from_id(crate::tensor::index_operand(id)).to::<u32>();
        (p.toint_i32().checked(), id)
    };
    for kk in 0..k {
        for mm in 0..m {
            let a_mk = a.slice(at(a.shape(), [mm, kk]));
            let desc_a = if tmem_a { a_mk.ptr().toint_i32().checked() } else { a_mk.smem_desc(0).checked() };
            for nn in 0..n {
                let desc_b = b.slice(at(b.shape(), [nn, kk])).smem_desc(0).checked();
                let taddr = d.slice(at(d.shape(), [mm, nn])).ptr().toint_i32().checked();
                // The metadata address with its column's lowest bit in the
                // descriptor's selector (CuTe C++'s sparse MMA_Traits).
                let (meta, idesc) = match &metadata {
                    None => (None, idesc),
                    Some(e) => {
                        let t = &e.tensor;
                        let t = if t.shape().rank() >= 3 { t.slice(at(t.shape(), [mm, kk])) } else { t.clone() };
                        let addr: Value<u32> = t.ptr().toint_i32().to::<u32>();
                        let id2 = addr & 1u32;
                        let meta = (addr & !1u32).to::<i32>().checked();
                        let base = negated.unwrap_or_else(|| crate::value::IntoValue::into_value(kind.idesc()));
                        (Some(meta), Some((base | id2).checked()))
                    }
                };
                let (sf, idesc) = match &scales {
                    None => (None, idesc),
                    Some((sfa, sfb)) => {
                        let (a_addr, a_id) = sf_operand(sfa, mm, kk);
                        let (b_addr, b_id) = sf_operand(sfb, nn, kk);
                        let base = negated.unwrap_or_else(|| crate::value::IntoValue::into_value(kind.idesc()));
                        let desc = base
                            | (a_id << crate::ir::UmmaKind::A_SF_ID_BIT)
                            | (b_id << crate::ir::UmmaKind::B_SF_ID_BIT);
                        (Some((a_addr, b_addr)), Some(desc.checked()))
                    }
                };
                let op = Op::Tcgen05Mma {
                    kind,
                    taddr,
                    desc_a,
                    desc_b,
                    enable_input_d: enable,
                    elect,
                    idesc,
                    disable_lane: disable_lane.clone(),
                    sf,
                    meta,
                    zero_col_mask: None,
                };
                builder::with(|bs| bs.push(op, loc));
            }
        }
    }
}

/// [`gemm`] with a `{disable-output-lane}` mask (CuTe DSL's `cute.gemm(...,
/// disable_output_lane=mask)`): the MMAs of a
/// tcgen05 `mma` leave the TMEM lanes of D whose bits `mask` sets
/// unwritten (see [`MmaAtom::set_disable_output_lane`]).
#[track_caller]
pub fn gemm_disable_output_lane<A: GemmOperand, B: GemmOperand, TC: Numeric, I: crate::value::IntoValue<i32> + Copy>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: A,
    b: B,
    c: &Tensor<TC>,
    mask: &[I],
) {
    let mut mma = mma.clone();
    mma.set_disable_output_lane(mask);
    gemm(&mma, d, a, b, c)
}

/// `d = a * b + c` for a sparse MMA ([`MmaF16BF16SparseOp`]): `a` is the
/// compressed `(MMA, M, K)` fragment (half of each K tile), `e` the
/// `(1, M, K)` Uint32 metadata ([`ThrMma::partition_fragment_e`]), `b`, `c`
/// and `d` as for [`gemm`]. In CuTe DSL `cute.gemm` takes `[a, e]` as its A
/// operand.
#[track_caller]
pub fn gemm_sparse<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    e: &Tensor<u32>,
    b: &Tensor<TB>,
    c: &Tensor<TC>,
) {
    assert!(mma.atom.is_sparse(), "gemm_sparse: {:?} is not a sparse MMA: call gemm", mma.atom.op);
    // A tcgen05 sparse MMA's metadata tensor is in TMEM.
    if mma.atom.is_tcgen05() {
        return gemm_tcgen05(mma, d, a, b, c, (Some(ScaleOperand::of(e)), None));
    }
    gemm_impl(mma, d, a, Some(e), b, c)
}

#[track_caller]
fn gemm_impl<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    e: Option<&Tensor<u32>>,
    b: &Tensor<TB>,
    c: &Tensor<TC>,
) {
    let loc = Location::caller();
    let (ta, tb, tc) = mma.atom.dtypes();
    assert!(
        mma.atom.takes_operand(ta, TA::DTYPE) && mma.atom.takes_operand(tb, TB::DTYPE) && TC::DTYPE == tc,
        "gemm operand types do not match {:?}: {:?} given, {:?} expected",
        mma.atom.op,
        (TA::DTYPE, TB::DTYPE, TC::DTYPE),
        (ta, tb, tc)
    );
    for (name, t) in [("D", d.layout()), ("A", a.layout()), ("B", b.layout()), ("C", c.layout())] {
        require_static_gemm_operand(name, t);
    }
    if let Some(e) = e {
        require_static_gemm_operand("E", e.layout());
        let size = |t: &Shape, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
        assert!(
            size(e.shape(), 0) == 1
                && size(e.shape(), 1) == size(a.shape(), 1)
                && size(e.shape(), 2) == size(a.shape(), 2),
            "gemm_sparse: the metadata E {} must be (1, MMA_M, MMA_K) for A {}",
            e.layout(),
            a.layout()
        );
    }
    let size = |t: &Shape, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
    let (m, n, k) = (size(c.shape(), 1), size(c.shape(), 2), size(a.shape(), 2));
    assert_eq!(size(a.shape(), 1), m, "gemm: A has {} M-blocks but C has {m}", size(a.shape(), 1));
    assert_eq!(size(b.shape(), 1), n, "gemm: B has {} N-blocks but C has {n}", size(b.shape(), 1));
    assert_eq!(size(b.shape(), 2), k, "gemm: A has {k} K-blocks but B has {}", size(b.shape(), 2));
    let at = |t: &Shape, idx: [usize; 2]| -> crate::layout::SliceCoord {
        let mut modes = vec![Tree::Leaf(None)];
        for (i, &x) in idx.iter().enumerate() {
            if i + 1 < t.rank() {
                modes.push(Tree::Leaf(Some(IndexExpr::Static(x))));
            }
        }
        Tree::Branch(modes)
    };
    for kk in 0..k {
        // Read each operand fragment once per k-block (these are loads when
        // the operands live in shared memory).
        let av: Vec<_> = (0..m).map(|mm| a.slice(at(a.shape(), [mm, kk])).values()).collect();
        let ev: Vec<_> = (0..m).map(|mm| e.map(|e| e.slice(at(e.shape(), [mm, kk])).values()[0])).collect();
        let bv: Vec<_> = (0..n).map(|nn| b.slice(at(b.shape(), [nn, kk])).values()).collect();
        for (mm, a_frag) in av.iter().enumerate() {
            for step in 0..n {
                // Serpentine over N so consecutive MMAs share an A fragment.
                let nn = if mm % 2 == 0 { step } else { n - 1 - step };
                let cv = c.slice(at(c.shape(), [mm, nn])).values();
                let dv = emit_mma(&mma.atom, a_frag, ev[mm], &bv[nn], &cv, loc);
                d.slice(at(d.shape(), [mm, nn])).store_values(&dv);
            }
        }
    }
}

/// Panic, naming the leaf, if `layout`, the layout of gemm's operand
/// `name`, has a runtime extent: gemm unrolls its modes when the kernel is
/// built.
#[track_caller]
fn require_static_gemm_operand(name: &str, layout: &Layout) {
    if let Some((path, leaf)) = layout.shape().first_dynamic() {
        panic!(
            "gemm: operand {name} ({layout}) has the dynamic extent {leaf:?} at mode {path:?}, but gemm unrolls the \
             per-thread values and the MMA_M, MMA_N and MMA_K modes when the kernel is built, so they must be static: \
             for a runtime number of K tiles, write a for_range over them and call gemm on one tile per iteration"
        );
    }
}

fn emit_mma<TA: Numeric, TB: Numeric, TC: Numeric>(
    atom: &MmaAtom,
    a: &[Value<TA>],
    e: Option<Value<u32>>,
    b: &[Value<TB>],
    c: &[Value<TC>],
    loc: crate::ir::Loc,
) -> Vec<Value<TC>> {
    match atom.op {
        MmaOp::UniversalFma(_) => {
            assert!(a.len() == 1 && b.len() == 1 && c.len() == 1, "UniversalFma fragments hold one value");
            let (a, b, c) = (a[0].checked(), b[0].checked(), c[0].checked());
            vec![Value::from_id(builder::with(|bs| bs.fma(a, b, c, loc)))]
        }
        MmaOp::Sm90(_) => unreachable!("gemm_warpgroup issues warpgroup MMAs"),
        MmaOp::Sm100(_) => unreachable!("gemm_tcgen05 issues tcgen05 MMAs"),
        MmaOp::Sm120(_) => unreachable!("gemm_block_scaled issues block-scaled MMAs"),
        MmaOp::Sm80(kind) => {
            let (na, nb, nc) = kind.fragment_sizes();
            assert_eq!(
                (a.len(), b.len(), c.len()),
                (na, nb, nc),
                "{kind:?} fragments must hold {na}, {nb} and {nc} values"
            );
            let (a, b, c): (Vec<_>, Vec<_>, Vec<_>) = (
                a.iter().map(|v| v.checked()).collect(),
                b.iter().map(|v| v.checked()).collect(),
                c.iter().map(|v| v.checked()).collect(),
            );
            let e = e.map(|e| e.checked());
            let d = builder::with(|bs| {
                // TFloat32 operands are their f32 bits (no instruction).
                let as_f32 = |bs: &mut crate::builder::BuilderState, v: Vec<crate::ir::ValueId>| -> Vec<_> {
                    v.into_iter()
                        .map(|x| if bs.dtype(x) == DType::TF32 { bs.bitcast(x, DType::F32, loc) } else { x })
                        .collect()
                };
                let (a, b) = (as_f32(bs, a), as_f32(bs, b));
                let d: Vec<_> = (0..nc).map(|_| bs.new_value(TC::DTYPE)).collect();
                bs.push(Op::Mma { kind, d: d.clone(), a, b, c, e }, loc);
                d
            });
            d.into_iter().map(Value::from_id).collect()
        }
    }
}

// Gemm operands with scale factors, and the
// block-scaled warp MMA.

/// A [`gemm`] A or B operand: a fragment (`&Tensor<T>`), or a fragment and
/// its scale factors (`(&Tensor<T>, &Tensor<S>)`, CuTe DSL's `[a, sfa]`).
pub trait GemmOperand {
    /// The operand's element type.
    type Elem: Numeric;
    #[doc(hidden)]
    fn split(self) -> (Tensor<Self::Elem>, Option<ScaleOperand>);
}

/// The scale factors of a block-scaled [`gemm`] operand (any element type;
/// its values are passed on as they are).
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ScaleOperand {
    pub(crate) dtype: DType,
    pub(crate) tensor: Tensor<u8>,
}

impl ScaleOperand {
    pub(crate) fn of<S: Numeric>(t: &Tensor<S>) -> ScaleOperand {
        ScaleOperand { dtype: S::DTYPE, tensor: t.retyped::<u8>() }
    }
}

impl<T: Numeric> GemmOperand for &Tensor<T> {
    type Elem = T;
    fn split(self) -> (Tensor<T>, Option<ScaleOperand>) {
        (self.clone(), None)
    }
}

impl<T: Numeric, S: Numeric> GemmOperand for (&Tensor<T>, &Tensor<S>) {
    type Elem = T;
    fn split(self) -> (Tensor<T>, Option<ScaleOperand>) {
        (self.0.clone(), Some(ScaleOperand::of(self.1)))
    }
}

impl MmaAtom {
    /// Set a tensor field (CuTe DSL's `atom.set(warp.Field.SFA, sfa)`, which
    /// takes the fragment's pointer): [`MmaField::SFA`](crate::MmaField::SFA)
    /// or `SFB` of a block-scaled warp MMA, the scale-factor fragment
    /// ([`partition_fragment_sfa`](crate::utils::sm120::partition_fragment_sfa))
    /// that [`gemm`] then uses for operands given without one. Unlike the
    /// boolean fields, tensor fields are not carried through loops.
    #[track_caller]
    pub fn set_tensor<S: Numeric>(&mut self, field: crate::MmaField, value: &Tensor<S>) {
        // The METADATA of a sparse tcgen05 MMA (in the first slot).
        if let MmaOp::Sm100(crate::ir::UmmaKind { sparse: true, .. }) = self.op {
            assert!(
                field == crate::MmaField::METADATA,
                "cute: set_tensor({field}) on the sparse MMA atom {:?}: its tensor field is METADATA",
                self.op
            );
            self.fields.tensors[0] = Some(ScaleOperand::of(value));
            return;
        }
        assert!(
            matches!(self.op, MmaOp::Sm120(_) | MmaOp::Sm100(crate::ir::UmmaKind { scale: Some(_), .. }))
                && matches!(field, crate::MmaField::SFA | crate::MmaField::SFB),
            "cute: set_tensor({field}) on the {:?} MMA atom: only the SFA and SFB fields of a block-scaled warp or tcgen05 MMA \
             take tensors here",
            self.op
        );
        self.fields.tensors[(field == crate::MmaField::SFB) as usize] = Some(ScaleOperand::of(value));
    }
}

impl TiledMma {
    /// [`MmaAtom::set_tensor`] on the tiled atom.
    #[track_caller]
    pub fn set_tensor<S: Numeric>(&mut self, field: crate::MmaField, value: &Tensor<S>) {
        self.atom.set_tensor(field, value)
    }
}

/// [`gemm`] for a block-scaled warp MMA (sm_120a): `a` is `(MMA, MMA_M, MMA_K)`, `b` `(MMA, MMA_N,
/// MMA_K)`, `c` and `d` Float32 `(MMA, MMA_M, MMA_N)`, and `sfa` and `sfb`
/// the scale-factor fragments of [`partition_fragment_sfa`] and
/// [`partition_fragment_sfb`] (`(MMA_SF, MMA_M, MMA_K)` and `(MMA_SF, MMA_N,
/// MMA_K)`, whose first mode holds one scale per K element of the atom,
/// repeated over each vector) or, when not given, the atom's `SFA` and `SFB`
/// fields. For each MMA the scale of K-block `j` (its element `j *
/// sf_vec_size`) goes into byte `j` of the scale register. A FP4 operand
/// of [`MmaMXF8F6F4Op`](crate::warp::MmaMXF8F6F4Op) is put in 8-bit
/// containers (bits 5:2).
///
/// [`partition_fragment_sfa`]: crate::utils::sm120::partition_fragment_sfa
/// [`partition_fragment_sfb`]: crate::utils::sm120::partition_fragment_sfb
#[track_caller]
pub fn gemm_block_scaled<TA: Numeric, TB: Numeric, TC: Numeric>(
    mma: &TiledMma,
    d: &Tensor<TC>,
    a: &Tensor<TA>,
    sfa: Option<ScaleOperand>,
    b: &Tensor<TB>,
    sfb: Option<ScaleOperand>,
    c: &Tensor<TC>,
) {
    let loc = Location::caller();
    let MmaOp::Sm120(kind) = mma.atom.op else {
        panic!("gemm_block_scaled: {:?} is not a block-scaled warp MMA", mma.atom.op)
    };
    assert!(
        TA::DTYPE == kind.a && TB::DTYPE == kind.b && TC::DTYPE == DType::F32,
        "gemm operand types do not match {:?}: {:?} given, {:?} expected",
        mma.atom.op,
        (TA::DTYPE, TB::DTYPE, TC::DTYPE),
        (kind.a, kind.b, DType::F32)
    );
    let field = |i: usize| mma.atom.fields.tensors[i].clone();
    let (Some(sfa), Some(sfb)) = (sfa.or_else(|| field(0)), sfb.or_else(|| field(1))) else {
        panic!(
            "gemm: the block-scaled MMA {} needs scale factors for A and B: pass (&a, &sfa) and (&b, &sfb), or set \
             the atom's SFA and SFB fields with set_tensor",
            kind.ptx()
        )
    };
    let sf = kind.sf_dtype();
    for (name, s) in [("SFA", &sfa), ("SFB", &sfb)] {
        assert!(
            s.dtype == sf,
            "gemm: {} takes {sf} scale factors (vectors of {} elements), but {name} holds {}",
            kind.ptx(),
            kind.sf_vec_size,
            s.dtype
        );
    }
    for (name, t) in [("D", d.layout()), ("A", a.layout()), ("B", b.layout()), ("C", c.layout())] {
        require_static_gemm_operand(name, t);
    }
    for (name, t) in [("SFA", sfa.tensor.layout()), ("SFB", sfb.tensor.layout())] {
        require_static_gemm_operand(name, t);
    }
    let size = |t: &Shape, i: usize| if i < t.rank() { t.mode(i).size() } else { 1 };
    let (m, n, k) = (size(c.shape(), 1), size(c.shape(), 2), size(a.shape(), 2));
    assert_eq!(size(a.shape(), 1), m, "gemm: A has {} M-blocks but C has {m}", size(a.shape(), 1));
    assert_eq!(size(b.shape(), 1), n, "gemm: B has {} N-blocks but C has {n}", size(b.shape(), 1));
    assert_eq!(size(b.shape(), 2), k, "gemm: A has {k} K-blocks but B has {}", size(b.shape(), 2));
    let atom_k = kind.shape_mnk().2;
    for (name, s, mn, other) in [("SFA", &sfa, m, "A"), ("SFB", &sfb, n, "B")] {
        let sh = s.tensor.shape();
        assert!(
            size(sh, 0) == atom_k && size(sh, 1) == mn && size(sh, 2) == k,
            "gemm: the scale factors {name} {} must be (MMA_SF, MMA_{}, MMA_K) with {atom_k} values per MMA, for {other} \
             {}",
            s.tensor.layout(),
            if name == "SFA" { "M" } else { "N" },
            if name == "SFA" { a.layout() } else { b.layout() }
        );
    }
    let (na, nb, _) = kind.fragment_sizes();
    assert!(
        size(a.shape(), 0) == na && size(b.shape(), 0) == nb,
        "gemm: each thread holds {na} A and {nb} B values of {}, but A is {} and B {}",
        kind.ptx(),
        a.layout(),
        b.layout()
    );
    let at = |t: &Shape, idx: [usize; 2]| -> crate::layout::SliceCoord {
        let mut modes = vec![Tree::Leaf(None)];
        for (i, &x) in idx.iter().enumerate() {
            if i + 1 < t.rank() {
                modes.push(Tree::Leaf(Some(IndexExpr::Static(x))));
            }
        }
        Tree::Branch(modes)
    };
    let per = kind.values_per_register();
    // One operand's registers: packed FP4 nibbles, or 8-bit containers (an
    // FP4 value in bits 5:2: pack it into the high nibble of a byte and
    // shift the word right by two).
    let registers = |values: Vec<crate::ir::ValueId>, ty: DType| -> Vec<crate::ir::ValueId> {
        builder::with(|bs| {
            values
                .chunks(per)
                .map(|chunk| {
                    if ty == DType::F4E2M1FN && !kind.is_fp4_packed() {
                        let zero = bs.constant(crate::ir::Literal::from_bits128(DType::F4E2M1FN, 0), loc);
                        let parts = chunk.iter().flat_map(|&v| [zero, v]).collect();
                        let word = bs.pack(parts, DType::U32, loc);
                        let two = bs.constant(crate::ir::Literal::from_i128(DType::U32, 2), loc);
                        bs.binary(crate::ir::BinaryOp::ShrU, word, two, loc)
                    } else {
                        bs.pack(chunk.to_vec(), DType::U32, loc)
                    }
                })
                .collect()
        })
    };
    // A scale register: the scale of K-block j in byte j, the rest zero.
    let nsf = kind.scales_per_mma();
    let scale_register = |s: &Tensor<u8>| -> crate::ir::ValueId {
        let ids: Vec<_> = (0..nsf).map(|j| s.load(j * kind.sf_vec_size as usize).checked()).collect();
        builder::with(|bs| {
            let bytes: Vec<_> = ids.iter().map(|&v| bs.bitcast(v, DType::U8, loc)).collect();
            match bytes.len() {
                1 => bs.cast(bytes[0], DType::U32, loc),
                2 => {
                    let half = bs.pack(bytes, DType::U16, loc);
                    bs.cast(half, DType::U32, loc)
                }
                _ => bs.pack(bytes, DType::U32, loc),
            }
        })
    };
    for kk in 0..k {
        let ra: Vec<_> = (0..m)
            .map(|mm| {
                let vals = a.slice(at(a.shape(), [mm, kk])).values().iter().map(|v| v.checked()).collect();
                registers(vals, kind.a)
            })
            .collect();
        let rb: Vec<_> = (0..n)
            .map(|nn| {
                let vals = b.slice(at(b.shape(), [nn, kk])).values().iter().map(|v| v.checked()).collect();
                registers(vals, kind.b)
            })
            .collect();
        let sa: Vec<_> = (0..m).map(|mm| scale_register(&sfa.tensor.slice(at(sfa.tensor.shape(), [mm, kk])))).collect();
        let sb: Vec<_> = (0..n).map(|nn| scale_register(&sfb.tensor.slice(at(sfb.tensor.shape(), [nn, kk])))).collect();
        for mm in 0..m {
            for step in 0..n {
                let nn = if mm % 2 == 0 { step } else { n - 1 - step };
                let cs = c.slice(at(c.shape(), [mm, nn]));
                let cv: Vec<_> = cs.values().iter().map(|v| v.checked()).collect();
                let dv = builder::with(|bs| {
                    let dv: Vec<_> = (0..4).map(|_| bs.new_value(DType::F32)).collect();
                    let op = Op::MmaBlockScaled {
                        kind,
                        d: dv.clone(),
                        a: ra[mm].clone(),
                        b: rb[nn].clone(),
                        c: cv,
                        sfa: sa[mm],
                        sfb: sb[nn],
                    };
                    bs.push(op, loc);
                    dv
                });
                let dv: Vec<Value<TC>> = dv.into_iter().map(Value::from_id).collect();
                d.slice(at(d.shape(), [mm, nn])).store_values(&dv);
            }
        }
    }
}

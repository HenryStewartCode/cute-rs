// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm100_umma.hpp (the shape checks of SM100_MMA_F16BF16_SS_SPARSE,
//   SM100_MMA_F16BF16_2x1SM_SS_SPARSE, SM100_MMA_F8F6F4_SS_SPARSE and
//   SM100_MMA_F8F6F4_2x1SM_SS_SPARSE, and their PTX forms),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The 2:4 structured-sparse tcgen05 MMAs:
//! CuTe DSL's `tcgen05.MmaF16BF16SparseOp` and `tcgen05.MmaFP8SparseOp`,
//! `tcgen05.mma.sp.cta_group::N.kind::{f16,f8f6f4}` (PTX 8.6, sm_100a and
//! its family).
//!
//! Of every four consecutive K elements of a row of A at most two are
//! nonzero. A is stored compressed in shared memory, the two kept values
//! of each group of four in order, so an A tile has half the
//! instruction's K: cute-rs's sparse atoms partition A in these compressed
//! coordinates (`partition_shape_a` of an `(M, K / 2)` tile; CuTe DSL
//! indexes A in logical K through a sparse element pointer, which cute-rs
//! does not model), and the descriptor fragment is the dense one of the
//! compressed tile. The metadata, one 4-bit field per group of four (the
//! position of the first kept value in bits 0-1, of the second in bits
//! 2-3, the first lower), lives in tensor memory in the layout of
//! [`make_sparse_tmem_layout_e`](crate::utils::sm100::make_sparse_tmem_layout_e)
//! (staged there from shared memory with `tcgen05.cp`); it is passed to
//! [`gemm`](crate::gemm) with A as `(&a, &e)` (CuTe DSL's `[a, e]`) or set
//! with [`TiledMma::set_tensor`](crate::TiledMma::set_tensor)`(Field::METADATA, ..)`.
//! For each MMA `gemm` passes the TMEM address of the metadata tensor
//! sliced at the MMA's `(MMA_M, MMA_K)` with its column's lowest bit
//! cleared, and puts that bit into the instruction descriptor's sparse
//! selector (bits 0-1), as CuTe C++'s `MMA_Traits` of the sparse atoms do;
//! the metadata address is therefore two-column aligned.
//!
//! cute-rs models M = 128 on one CTA and M = 256 on a CTA pair (CuTe C++'s
//! metadata TMEM layout `tmem_e_frg` is written for 128 rows per CTA), A in
//! shared memory.

use super::mma::{CtaGroup, OperandSource};
use crate::atom::mma::{MmaAtom, MmaOp, SparseMetadataFormat};
use crate::ir::{DType, UmmaDataKind, UmmaKind};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::warpgroup::InstructionShape;

/// The checks of both sparse MMAs, and their instruction.
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn sparse_kind(
    name: &str,
    kind: UmmaDataKind,
    ab: DType,
    acc: DType,
    instruction_shape: impl InstructionShape,
    cta_group: CtaGroup,
    a_src: OperandSource,
    (a_major_mode, b_major_mode): (OperandMajorMode, OperandMajorMode),
) -> UmmaKind {
    let (m, n, k) = instruction_shape.mnk();
    let want_k = 2 * kind.k();
    let got = k.unwrap_or(want_k);
    assert!(
        got == want_k,
        "cute: tcgen05.{name}: the instruction shape's K is {got}; this sparse MMA has K = {want_k} (A stores half)"
    );
    assert!(
        a_src == OperandSource::SMEM,
        "cute: tcgen05.{name}: a_src=OperandSource.TMEM: the sparse MMAs read A from shared memory \
         (OperandSource.SMEM)"
    );
    let pair = cta_group == CtaGroup::TWO;
    let want_m = if pair { 256 } else { 128 };
    assert!(
        m == want_m,
        "cute: tcgen05.{name}: the instruction shape's M is {m}; cute-rs models sparse MMAs with 128 rows per CTA: M = \
         128 with CtaGroup.ONE, 256 with CtaGroup.TWO"
    );
    super::mma::check_instruction_shape(name, (m, n), cta_group, ab, b_major_mode);
    // The 8-bit forms step N by 16 (32 for a pair) at M = 128 (256).
    if kind == UmmaDataKind::F8F6F4 {
        let step = if pair { 32 } else { 16 };
        assert!(
            n.is_multiple_of(step),
            "cute: tcgen05.{name}: the instruction shape's N is {n}; the sparse 8-bit MMA with M = {m} has N a \
             multiple of {step}"
        );
    }
    UmmaKind {
        kind,
        m: m as u32,
        n: n as u32,
        a: ab,
        b: ab,
        acc,
        trans_a: a_major_mode == OperandMajorMode::MN,
        trans_b: b_major_mode == OperandMajorMode::MN,
        two_cta: pair,
        a_tmem: false,
        scale: None,
        sparse: true,
        sm107: None,
        prims: Default::default(),
    }
}

macro_rules! sparse_accessors {
    ($t:ident) => {
        impl $t {
            /// The type of A (as stored, compressed) and of B.
            pub fn ab_dtype(&self) -> DType {
                self.kind.a
            }

            pub fn acc_dtype(&self) -> DType {
                self.kind.acc
            }

            /// `(M, N, K)`, K the logical (uncompressed) extent.
            pub fn shape_mnk(&self) -> (usize, usize, usize) {
                self.kind.shape_mnk()
            }

            pub fn cta_group(&self) -> CtaGroup {
                if self.kind.two_cta { CtaGroup::TWO } else { CtaGroup::ONE }
            }

            /// Always shared memory.
            pub fn a_src(&self) -> OperandSource {
                OperandSource::SMEM
            }

            pub fn a_major_mode(&self) -> OperandMajorMode {
                if self.kind.trans_a { OperandMajorMode::MN } else { OperandMajorMode::K }
            }

            pub fn b_major_mode(&self) -> OperandMajorMode {
                if self.kind.trans_b { OperandMajorMode::MN } else { OperandMajorMode::K }
            }

            /// CuTe DSL's `sparse_metadata_format`.
            pub fn sparse_metadata_format(&self) -> SparseMetadataFormat {
                self.format
            }

            /// The instruction it issues.
            pub fn kind(&self) -> UmmaKind {
                self.kind
            }
        }

        impl From<$t> for MmaOp {
            fn from(op: $t) -> MmaOp {
                MmaOp::Sm100(op.kind)
            }
        }

        impl From<$t> for MmaAtom {
            fn from(op: $t) -> MmaAtom {
                MmaAtom::new(op.into())
            }
        }
    };
}

/// CuTe DSL's `tcgen05.MmaF16BF16SparseOp(ab_dtype, acc_dtype,
/// instruction_shape, cta_group, a_src, a_major_mode, b_major_mode,
/// sparse_metadata_format)`: `tcgen05.mma.sp.cta_group::N.kind::f16`,
/// K = 32, 2:4-sparse Float16 or BFloat16 A (16 stored values per row
/// per instruction) times dense B, accumulated in Float16 or Float32 in
/// TMEM (BFloat16 into Float32 only). M is 128 (`CtaGroup::ONE`, N a
/// multiple of 8 up to 256) or 256 (`CtaGroup::TWO`, N a multiple of 16);
/// A and B may each be K-major or MN-major. Takes the runtime fields
/// `ACCUMULATE`, `NEGATE_A`, `NEGATE_B`, `METADATA` and
/// `DISABLE_OUTPUT_LANE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16SparseOp {
    kind: UmmaKind,
    format: SparseMetadataFormat,
}

impl MmaF16BF16SparseOp {
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
        sparse_metadata_format: SparseMetadataFormat,
    ) -> MmaF16BF16SparseOp {
        let name = "MmaF16BF16SparseOp";
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: tcgen05.{name}: 'ab_dtype' is {ab_dtype}; its A and B operands are Float16 or BFloat16"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: tcgen05.{name}: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: tcgen05.{name}: BFloat16 operands accumulate in Float32 only, and 'acc_dtype' is {acc_dtype}"
        );
        let kind = sparse_kind(
            name,
            UmmaDataKind::F16,
            ab_dtype,
            acc_dtype,
            instruction_shape,
            cta_group,
            a_src,
            (a_major_mode, b_major_mode),
        );
        MmaF16BF16SparseOp { kind, format: sparse_metadata_format }
    }
}

sparse_accessors!(MmaF16BF16SparseOp);

/// CuTe DSL's `tcgen05.MmaFP8SparseOp(ab_dtype, acc_dtype,
/// instruction_shape, cta_group, a_src, a_major_mode, b_major_mode,
/// sparse_metadata_format)`: `tcgen05.mma.sp.cta_group::N.kind::f8f6f4`,
/// K = 64, 2:4-sparse A and dense B both Float8E4M3FN or both Float8E5M2,
/// accumulated in Float16 or Float32. M is 128 (`CtaGroup::ONE`, N a
/// multiple of 16) or 256 (`CtaGroup::TWO`, N a multiple of 32). The
/// fields and metadata are those of [`MmaF16BF16SparseOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaFP8SparseOp {
    kind: UmmaKind,
    format: SparseMetadataFormat,
}

impl MmaFP8SparseOp {
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
        sparse_metadata_format: SparseMetadataFormat,
    ) -> MmaFP8SparseOp {
        let name = "MmaFP8SparseOp";
        assert!(
            matches!(ab_dtype, DType::F8E4M3FN | DType::F8E5M2),
            "cute: tcgen05.{name}: 'ab_dtype' is {ab_dtype}; A and B are both Float8E4M3FN or both Float8E5M2"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: tcgen05.{name}: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        let kind = sparse_kind(
            name,
            UmmaDataKind::F8F6F4,
            ab_dtype,
            acc_dtype,
            instruction_shape,
            cta_group,
            a_src,
            (a_major_mode, b_major_mode),
        );
        MmaFP8SparseOp { kind, format: sparse_metadata_format }
    }
}

sparse_accessors!(MmaFP8SparseOp);

#[cfg(test)]
mod tests {
    use super::*;
    use OperandMajorMode::K;

    #[test]
    fn sparse_descriptors() {
        let (s, tid) = (OperandSource::SMEM, SparseMetadataFormat::Tid);
        // F16 -> F32, 128 x 64: the sparse flag (bit 2), K = 32.
        let f = MmaF16BF16SparseOp::new(DType::F16, DType::F32, (128, 64), CtaGroup::ONE, s, K, K, tid).kind();
        assert_eq!((f.k(), f.a_k()), (32, 16));
        assert_eq!(f.idesc(), 1 << 2 | 1 << 4 | 8 << 17 | 8 << 24);
        assert_eq!(f.ptx(), "tcgen05.mma.sp.cta_group::1.kind::f16");
        // E5M2 -> F16 on a pair, 256 x 128: K = 64.
        let e = MmaFP8SparseOp::new(DType::F8E5M2, DType::F16, (256, 128), CtaGroup::TWO, s, K, K, tid).kind();
        assert_eq!((e.k(), e.a_k()), (64, 32));
        assert_eq!(e.idesc(), 1 << 2 | 1 << 7 | 1 << 10 | 16 << 17 | 16 << 24);
        assert_eq!(e.ptx(), "tcgen05.mma.sp.cta_group::2.kind::f8f6f4");
    }
}

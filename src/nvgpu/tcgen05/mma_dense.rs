// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm100_umma.hpp (the shape checks of SM100_MMA_TF32_SS,
//   SM100_MMA_S8_SS and SM100_MMA_F8F6F4_SS) and include/cute/arch/mma_sm100_desc.hpp
//   (UMMA::to_UMMAFormat's TF32, S8 and MXF8F6F4 format codes, in
//   UmmaDataKind::operand_format),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The dense tcgen05 MMAs other than F16/BF16 (roadmap part
//! m10-umma-dense-kinds): CuTe DSL's `tcgen05.MmaTF32Op`, `MmaI8Op`,
//! `MmaF8F6F4Op` and its deprecated FP8-only form `MmaFP8Op`, on the
//! machinery of [`MmaF16BF16Op`](super::MmaF16BF16Op) (descriptor
//! fragments, TMEM accumulators, [`gemm`](crate::gemm),
//! [`commit`](super::commit)).

use super::mma::OperandSource;
use crate::atom::mma::{MmaAtom, MmaOp};
use crate::ir::{DType, UmmaDataKind, UmmaKind};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::nvgpu::warpgroup::InstructionShape;

/// The checks every dense tcgen05 MMA shares (CuTe DSL's `MmaOp` base):
/// the instruction shape for the `cta_group`
/// ([`check_instruction_shape`](super::mma::check_instruction_shape)) and,
/// with `CtaGroup::TWO`, A in shared memory. Returns `(M, N)`.
#[track_caller]
fn check_common(
    name: &str,
    instruction_shape: (usize, usize),
    cta_group: CtaGroup,
    a_src: OperandSource,
    b_dtype: DType,
    b_major_mode: OperandMajorMode,
) -> (usize, usize) {
    super::mma::check_pair_a_source(name, cta_group, a_src);
    super::mma::check_instruction_shape(name, instruction_shape, cta_group, b_dtype, b_major_mode)
}

/// `kind` issued by a CTA pair when `cta_group` is `CtaGroup::TWO`.
fn grouped(mut kind: UmmaKind, cta_group: CtaGroup) -> UmmaKind {
    kind.two_cta = cta_group == CtaGroup::TWO;
    kind
}

/// The instruction shape's K, defaulted to `k` when the shape has rank 2.
#[track_caller]
fn check_k(name: &str, given: Option<usize>, k: usize) -> usize {
    let got = given.unwrap_or(k);
    assert!(got == k, "cute: tcgen05.{name}: the instruction shape's K is {got}, but this MMA has K = {k}");
    k
}

macro_rules! umma_accessors {
    ($t:ident) => {
        impl $t {
            pub fn a_dtype(&self) -> DType {
                self.kind.a
            }

            pub fn b_dtype(&self) -> DType {
                self.kind.b
            }

            pub fn acc_dtype(&self) -> DType {
                self.kind.acc
            }

            pub fn shape_mnk(&self) -> (usize, usize, usize) {
                self.kind.shape_mnk()
            }

            pub fn cta_group(&self) -> CtaGroup {
                if self.kind.two_cta { CtaGroup::TWO } else { CtaGroup::ONE }
            }

            pub fn a_src(&self) -> OperandSource {
                if self.kind.a_tmem { OperandSource::TMEM } else { OperandSource::SMEM }
            }

            pub fn a_major_mode(&self) -> OperandMajorMode {
                if self.kind.trans_a { OperandMajorMode::MN } else { OperandMajorMode::K }
            }

            pub fn b_major_mode(&self) -> OperandMajorMode {
                if self.kind.trans_b { OperandMajorMode::MN } else { OperandMajorMode::K }
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

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn umma_kind(
    name: &str,
    kind: UmmaDataKind,
    (m, n): (usize, usize),
    a: DType,
    b: DType,
    acc: DType,
    a_src: OperandSource,
    a_major_mode: OperandMajorMode,
    b_major_mode: OperandMajorMode,
) -> UmmaKind {
    super::mma::check_a_source(name, a_src, a_major_mode, a);
    UmmaKind {
        kind,
        m: m as u32,
        n: n as u32,
        a,
        b,
        acc,
        trans_a: a_major_mode == OperandMajorMode::MN,
        trans_b: b_major_mode == OperandMajorMode::MN,
        two_cta: false,
        a_tmem: a_src == OperandSource::TMEM,
        scale: None,
        sparse: false,
        sm107: None,
        prims: Default::default(),
    }
}

/// CuTe DSL's `tcgen05.MmaTF32Op(instruction_shape, cta_group, a_src,
/// a_major_mode, b_major_mode)`: `tcgen05.mma.cta_group::1.kind::tf32`
/// (sm_100a, sm_110a; the sm_100f, sm_110f families from PTX 8.8) on
/// TFloat32 operands, K = 8, into a Float32 TMEM accumulator. The operands
/// are 32-bit in shared memory (Float32 or TFloat32 tensors); the tensor
/// core uses the top 19 bits of each (the low 13 mantissa bits are
/// dropped, as by the warp-level [`MmaTF32Op`](crate::MmaTF32Op)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaTF32Op {
    kind: UmmaKind,
}

impl MmaTF32Op {
    #[track_caller]
    pub fn new(
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaTF32Op {
        let (m, n, k) = instruction_shape.mnk();
        check_k("MmaTF32Op", k, 8);
        let mn = check_common("MmaTF32Op", (m, n), cta_group, a_src, DType::TF32, b_major_mode);
        let kind = umma_kind(
            "MmaTF32Op",
            UmmaDataKind::Tf32,
            mn,
            DType::TF32,
            DType::TF32,
            DType::F32,
            a_src,
            a_major_mode,
            b_major_mode,
        );
        MmaTF32Op { kind: grouped(kind, cta_group) }
    }
}

umma_accessors!(MmaTF32Op);

/// CuTe DSL's `tcgen05.MmaI8Op(ab_dtype, instruction_shape, cta_group,
/// a_src, a_major_mode, b_major_mode)`: `tcgen05.mma.cta_group::1.kind::i8`
/// on Int8 or Uint8 operands (`ab_dtype`, both), K = 32, into an Int32 TMEM
/// accumulator, with exact (wrapping) integer arithmetic.
///
/// N is 8, 16, 24 or 32, or a multiple of 16 from 48 to 256 (a multiple of
/// 16 from 16 to 256 with an MN-major B). The PTX ISA allows `.kind::i8`
/// only on sm_100a, sm_101a and sm_110a, not on the `f` families, so
/// [`check_target`](crate::validate::check_target) refuses sm_100f and its
/// peers although CuTe DSL lists them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaI8Op {
    kind: UmmaKind,
}

impl MmaI8Op {
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaI8Op {
        assert!(
            matches!(ab_dtype, DType::I8 | DType::U8),
            "cute: tcgen05.MmaI8Op: 'ab_dtype' is {ab_dtype}; its operands are Int8 or Uint8"
        );
        let (m, n, k) = instruction_shape.mnk();
        check_k("MmaI8Op", k, 32);
        let (m, n) = check_common("MmaI8Op", (m, n), cta_group, a_src, ab_dtype, b_major_mode);
        assert!(
            n <= 32 || n.is_multiple_of(16),
            "cute: tcgen05.MmaI8Op: the instruction shape's N is {n}; above 32, an Int8/Uint8 MMA's N goes in steps of \
             16 (48, 64, ..., 256)"
        );
        let kind = umma_kind(
            "MmaI8Op",
            UmmaDataKind::I8,
            (m, n),
            ab_dtype,
            ab_dtype,
            DType::I32,
            a_src,
            a_major_mode,
            b_major_mode,
        );
        MmaI8Op { kind: grouped(kind, cta_group) }
    }
}

umma_accessors!(MmaI8Op);

/// The operand types of `.kind::f8f6f4`.
const F8F6F4_TYPES: [DType; 5] = [DType::F8E4M3FN, DType::F8E5M2, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN];

/// CuTe DSL's `tcgen05.MmaF8F6F4Op(a_dtype, b_dtype, acc_dtype,
/// instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`:
/// `tcgen05.mma.cta_group::1.kind::f8f6f4` (the targets of
/// [`MmaTF32Op`]), K = 32, A and B each Float8E4M3FN, Float8E5M2,
/// Float6E2M3FN, Float6E3M2FN or Float4E2M1FN, into a Float16 or Float32
/// TMEM accumulator.
///
/// FP6 and FP4 operands are K-major and read from the padded shared-memory
/// image, each 16 values in a 16-byte slot (PTX's `.b6x16_p32` and
/// `.b4x16_p64`, as a TMA load with an 8-bit `internal_type` writes them):
/// pass that image as a Uint8 tensor to
/// [`make_fragment_a`](crate::TiledMma::make_fragment_a) and
/// [`gemm`](crate::gemm).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF8F6F4Op {
    kind: UmmaKind,
}

impl MmaF8F6F4Op {
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn new(
        a_dtype: DType,
        b_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaF8F6F4Op {
        Self::checked(
            "MmaF8F6F4Op",
            a_dtype,
            b_dtype,
            acc_dtype,
            instruction_shape,
            cta_group,
            a_src,
            a_major_mode,
            b_major_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    fn checked(
        name: &str,
        a_dtype: DType,
        b_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaF8F6F4Op {
        for (what, t, major) in [("a_dtype", a_dtype, a_major_mode), ("b_dtype", b_dtype, b_major_mode)] {
            assert!(
                F8F6F4_TYPES.contains(&t),
                "cute: tcgen05.{name}: '{what}' is {t}, which .kind::f8f6f4 does not take (it takes Float8E4M3FN, \
                 Float8E5M2, Float6E2M3FN, Float6E3M2FN and Float4E2M1FN)"
            );
            assert!(
                t.bits() == 8 || major == OperandMajorMode::K,
                "cute: tcgen05.{name}: '{what}' is {t} and MN-major, but FP6 and FP4 operands (read from the padded \
                 16-byte slots) must be K-major; only 8-bit operands may be MN-major"
            );
        }
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: tcgen05.{name}: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        let (m, n, k) = instruction_shape.mnk();
        check_k(name, k, 32);
        let mn = check_common(name, (m, n), cta_group, a_src, b_dtype, b_major_mode);
        let kind =
            umma_kind(name, UmmaDataKind::F8F6F4, mn, a_dtype, b_dtype, acc_dtype, a_src, a_major_mode, b_major_mode);
        MmaF8F6F4Op { kind: grouped(kind, cta_group) }
    }
}

umma_accessors!(MmaF8F6F4Op);

/// CuTe DSL's deprecated `tcgen05.MmaFP8Op(ab_dtype, acc_dtype,
/// instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`: the
/// FP8-only form of [`MmaF8F6F4Op`], with A and B both Float8E4M3FN or both
/// Float8E5M2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaFP8Op {
    kind: UmmaKind,
}

impl MmaFP8Op {
    #[deprecated(note = "use tcgen05::MmaF8F6F4Op")]
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaFP8Op {
        assert!(
            matches!(ab_dtype, DType::F8E4M3FN | DType::F8E5M2),
            "cute: tcgen05.MmaFP8Op: 'ab_dtype' is {ab_dtype}; its operands are both Float8E4M3FN or both Float8E5M2 \
             (MmaF8F6F4Op takes the other formats and mixed pairs)"
        );
        let op = MmaF8F6F4Op::checked(
            "MmaFP8Op",
            ab_dtype,
            ab_dtype,
            acc_dtype,
            instruction_shape,
            cta_group,
            a_src,
            a_major_mode,
            b_major_mode,
        );
        MmaFP8Op { kind: op.kind }
    }
}

umma_accessors!(MmaFP8Op);

#[cfg(test)]
mod tests {
    use super::*;
    use OperandMajorMode::{K, MN};

    #[test]
    fn instruction_descriptor_fields() {
        let s = OperandSource::SMEM;
        // TF32 -> F32, 128 x 64: a_format = b_format = 2.
        let tf = MmaTF32Op::new((128, 64), CtaGroup::ONE, s, K, MN).kind();
        assert_eq!(tf.k(), 8);
        assert_eq!(tf.idesc(), 1 << 4 | 2 << 7 | 2 << 10 | 1 << 16 | 8 << 17 | 8 << 24);
        // S8 -> S32: c_format 2, a_format = b_format = 1; U8: 0.
        let i8 = MmaI8Op::new(DType::I8, (64, 48), CtaGroup::ONE, s, K, K).kind();
        assert_eq!((i8.k(), i8.idesc()), (32, 2 << 4 | 1 << 7 | 1 << 10 | 6 << 17 | 4 << 24));
        let u8 = MmaI8Op::new(DType::U8, (128, 16), CtaGroup::ONE, s, MN, MN).kind();
        assert_eq!(u8.idesc(), 2 << 4 | 1 << 15 | 1 << 16 | 2 << 17 | 8 << 24);
        // F8F6F4: E2M1 (5) x E3M2 (4) -> F16, E5M2 (1) x E2M3 (3) -> F32.
        let f = |a, b, acc| MmaF8F6F4Op::new(a, b, acc, (128, 256), CtaGroup::ONE, s, K, K).kind().idesc();
        assert_eq!(f(DType::F4E2M1FN, DType::F6E3M2FN, DType::F16), 5 << 7 | 4 << 10 | 32 << 17 | 8 << 24);
        assert_eq!(f(DType::F8E5M2, DType::F6E2M3FN, DType::F32), 1 << 4 | 1 << 7 | 3 << 10 | 32 << 17 | 8 << 24);
        assert_eq!(f(DType::F8E4M3FN, DType::F8E4M3FN, DType::F32), 1 << 4 | 32 << 17 | 8 << 24);
    }
}

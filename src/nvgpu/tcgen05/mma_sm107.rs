// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm107_umma.hpp (the shape checks of SM107_MMA_F8F6F4_SS,
//   SM107_MMA_MXF8F6F4_SS, SM107_MMA_MXF4NVF4_SS and their 2x1SM forms, and their PTX
//   forms: K = 64 and 128, `.block16`/`.block32`, `.collector::b::fill`/`lastuse`),
//   include/cute/arch/mma_sm107_desc.hpp (Sm107MXF8F6F4InstrDescriptor and
//   Sm107MXF4NVF4InstrDescriptor: the K-size bits and the UE5M3 scale format, in
//   UmmaKind::idesc) and include/cute/atom/mma_traits_sm107.hpp (the K and N rules of
//   the traits),
//     Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The Rubin (sm_107) tcgen05 MMAs: CuTe
//! DSL's `tcgen05.SM107MmaFP8Op` (`.kind::f8f6f4`, K = 32 or 64),
//! `SM107BlockScaledMmaMXF8F6F4Op` (`.kind::mxf8f6f4.block_scale`, K = 64)
//! and `SM107MmaMXF4NVF4Op` (`.kind::mxf4nvf4.block_scale.block16` or
//! `.block32`, K = 128), each with the collector-buffer qualifiers of A and
//! B ([`CollectorOp`]).
//!
//! These are PTX ISA 9.4 instructions of the sm_107f family (sm_107f and
//! sm_107a). No installed LLVM or ptxas knows sm_107, so they are emitted
//! as inline PTX and checked on the simulator and as MLIR text only. Their
//! shared-memory descriptors use the sm_107 encoding (15-bit start address
//! and LBO, [`UmmaDescEncoding::Sm107`](super::UmmaDescEncoding::Sm107)).
//!
//! **Collectors.** With `CollectorOp::FILL` the MMA reads the operand from
//! shared memory and keeps it in the operand's collector buffer; later MMAs
//! with `USE` (or `LASTUSE`, which then releases the buffer) read it from
//! the buffer instead of shared memory. The descriptor is still passed and
//! names the same tile. The simulator keeps each CTA's A and B collectors
//! and reports a `USE`/`LASTUSE` with nothing filled, of a different tile,
//! or of a tile whose shared memory has changed since the `FILL`. A
//! collector of A needs A in shared memory (cute-rs's reading of the PTX
//! ISA: a TMEM-sourced A is not collected).

use super::mma::OperandSource;
use crate::atom::mma::{MmaAtom, MmaOp};
use crate::ir::{CollectorOp, DType, Sm107Umma, UmmaDataKind, UmmaKind, UmmaScale};
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::nvgpu::warpgroup::InstructionShape;

/// The checks of the collector usage: A's needs A in shared memory.
#[track_caller]
fn check_collectors(name: &str, a_src: OperandSource, a_collector_op: CollectorOp) {
    assert!(
        a_collector_op == CollectorOp::DISCARD || a_src == OperandSource::SMEM,
        "cute: tcgen05.{name}: a_collector_op={a_collector_op:?} with a_src=OperandSource.TMEM: only an A read from \
         shared memory goes through A's collector buffer"
    );
}

/// The block-scaled Rubin MMAs' shape: M = 128 on one CTA, 256 on a pair;
/// N a multiple of 8 (16 on a pair) up to 256, doubled for an 8-bit
/// MN-major B (CuTe DSL's constraints).
#[track_caller]
fn check_block_scaled_shape(
    name: &str,
    (m, n): (usize, usize),
    cta_group: CtaGroup,
    b: DType,
    b_major: OperandMajorMode,
) {
    let pair = cta_group == CtaGroup::TWO;
    let want_m = if pair { 256 } else { 128 };
    assert!(
        m == want_m,
        "cute: tcgen05.{name}: the instruction shape's M is {m}; this block-scaled MMA has M = {want_m} with {}",
        if pair { "CtaGroup.TWO" } else { "CtaGroup.ONE" }
    );
    super::mma::check_instruction_shape(name, (m, n), cta_group, b, b_major);
}

/// A Rubin `UmmaKind`.
#[allow(clippy::too_many_arguments)]
fn rubin_kind(
    kind: UmmaDataKind,
    (m, n): (usize, usize),
    (a, b, acc): (DType, DType, DType),
    scale: Option<UmmaScale>,
    wide_k: bool,
    cta_group: CtaGroup,
    a_src: OperandSource,
    (a_major, b_major): (OperandMajorMode, OperandMajorMode),
    (collector_a, collector_b): (CollectorOp, CollectorOp),
) -> UmmaKind {
    UmmaKind {
        kind,
        m: m as u32,
        n: n as u32,
        a,
        b,
        acc,
        trans_a: a_major == OperandMajorMode::MN,
        trans_b: b_major == OperandMajorMode::MN,
        two_cta: cta_group == CtaGroup::TWO,
        a_tmem: a_src == OperandSource::TMEM,
        scale,
        sparse: false,
        sm107: Some(Sm107Umma { wide_k, collector_a, collector_b }),
        prims: Default::default(),
    }
}

macro_rules! rubin_accessors {
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

            /// CuTe DSL's `a_collector_op`.
            pub fn a_collector_op(&self) -> CollectorOp {
                self.kind.sm107.expect("a Rubin MMA").collector_a
            }

            /// CuTe DSL's `b_collector_op`.
            pub fn b_collector_op(&self) -> CollectorOp {
                self.kind.sm107.expect("a Rubin MMA").collector_b
            }

            /// This operation with other collector usages (CuTe C++'s
            /// `MMA_Traits::with(b_reuse)`): the same instruction, reading or
            /// filling the collector buffers as `a` and `b` say.
            #[track_caller]
            pub fn with_collectors(self, a: CollectorOp, b: CollectorOp) -> $t {
                check_collectors(stringify!($t), self.a_src(), a);
                let mut kind = self.kind;
                kind.sm107 = Some(Sm107Umma { collector_a: a, collector_b: b, ..kind.sm107.expect("a Rubin MMA") });
                $t { kind }
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

/// CuTe DSL's `tcgen05.SM107MmaFP8Op(a_dtype, b_dtype, acc_dtype,
/// instruction_shape, cta_group, a_src, a_major_mode, b_major_mode,
/// a_collector_op, b_collector_op)`: `tcgen05.mma.cta_group::N.kind::f8f6f4`
/// on sm_107f and sm_107a (PTX 9.4), A and B each Float8E4M3FN or
/// Float8E5M2, a Float16 or Float32 accumulator, K = 64 (the default; the
/// instruction descriptor's K-size bit 29) or 32. The shape rules are
/// [`MmaF8F6F4Op`](super::MmaF8F6F4Op)'s (M of 64 or 128 on one CTA, 128 or
/// 256 on a pair), and A and B may be MN-major.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SM107MmaFP8Op {
    kind: UmmaKind,
}

impl SM107MmaFP8Op {
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
        a_collector_op: CollectorOp,
        b_collector_op: CollectorOp,
    ) -> SM107MmaFP8Op {
        let name = "SM107MmaFP8Op";
        for (what, t) in [("a_dtype", a_dtype), ("b_dtype", b_dtype)] {
            assert!(
                matches!(t, DType::F8E4M3FN | DType::F8E5M2),
                "cute: tcgen05.{name}: '{what}' is {t}; this MMA takes Float8E4M3FN or Float8E5M2"
            );
        }
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: tcgen05.{name}: 'acc_dtype' is {acc_dtype}; this MMA accumulates in Float16 or Float32"
        );
        let (m, n, k) = instruction_shape.mnk();
        let k = k.unwrap_or(64);
        assert!(
            k == 32 || k == 64,
            "cute: tcgen05.{name}: the instruction shape's K is {k}; this MMA has K = 32 or K = 64 (the default)"
        );
        super::mma::check_pair_a_source(name, cta_group, a_src);
        super::mma::check_instruction_shape(name, (m, n), cta_group, b_dtype, b_major_mode);
        super::mma::check_a_source(name, a_src, a_major_mode, a_dtype);
        check_collectors(name, a_src, a_collector_op);
        let kind = rubin_kind(
            UmmaDataKind::F8F6F4,
            (m, n),
            (a_dtype, b_dtype, acc_dtype),
            None,
            k == 64,
            cta_group,
            a_src,
            (a_major_mode, b_major_mode),
            (a_collector_op, b_collector_op),
        );
        SM107MmaFP8Op { kind }
    }
}

rubin_accessors!(SM107MmaFP8Op);

/// The operand types of the Rubin `.kind::mxf8f6f4`: SM100's five and
/// FloatNV8E5M3FNU.
const SM107_MXF8F6F4_TYPES: [DType; 6] =
    [DType::F8E4M3FN, DType::F8E5M2, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN, DType::F8E5M3FNU];

/// CuTe DSL's `tcgen05.SM107BlockScaledMmaMXF8F6F4Op(a_dtype, b_dtype,
/// sf_dtype, sf_vec_size, instruction_shape, cta_group, a_src,
/// a_major_mode, b_major_mode, a_collector_op, b_collector_op)`:
/// `tcgen05.mma.cta_group::N.kind::mxf8f6f4.block_scale` with K = 64 (the
/// block-scaled descriptor's bit 31) on sm_107f and sm_107a (PTX 9.4): A
/// and B each Float8E4M3FN, Float8E5M2, Float6E2M3FN, Float6E3M2FN,
/// Float4E2M1FN or FloatNV8E5M3FNU, one UE8M0 (Float8E8M0FNU) scale factor
/// per 32 K elements (two per row per instruction), into Float32. M = 128
/// on one CTA, 256 on a pair.
///
/// As for [`MmaMXF8F6F4Op`](super::MmaMXF8F6F4Op), FP6 and FP4 operands are
/// K-major and read from the padded image (a Uint8 tensor).
/// FloatNV8E5M3FNU operands are accepted, as CuTe DSL accepts them, but
/// [`gemm`](crate::gemm) refuses to issue them: no published source gives
/// their instruction-descriptor format code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SM107BlockScaledMmaMXF8F6F4Op {
    kind: UmmaKind,
}

impl SM107BlockScaledMmaMXF8F6F4Op {
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn new(
        a_dtype: DType,
        b_dtype: DType,
        sf_dtype: DType,
        sf_vec_size: usize,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
        a_collector_op: CollectorOp,
        b_collector_op: CollectorOp,
    ) -> SM107BlockScaledMmaMXF8F6F4Op {
        let name = "SM107BlockScaledMmaMXF8F6F4Op";
        for (what, t, major) in [("a_dtype", a_dtype, a_major_mode), ("b_dtype", b_dtype, b_major_mode)] {
            assert!(
                SM107_MXF8F6F4_TYPES.contains(&t),
                "cute: tcgen05.{name}: '{what}' is {t}, which the Rubin .kind::mxf8f6f4 does not take (it takes \
                 Float8E4M3FN, Float8E5M2, Float6E2M3FN, Float6E3M2FN, Float4E2M1FN and FloatNV8E5M3FNU)"
            );
            assert!(
                t.bits() == 8 || major == OperandMajorMode::K,
                "cute: tcgen05.{name}: '{what}' is {t} and MN-major; FP6 and FP4 operands are read K-major from their \
                 padded 16-byte slots"
            );
        }
        assert!(
            sf_dtype == DType::F8E8M0FNU,
            "cute: tcgen05.{name}: 'sf_dtype' is {sf_dtype}; this MMA's scale factors are Float8E8M0FNU (UE8M0)"
        );
        assert!(
            sf_vec_size == 32,
            "cute: tcgen05.{name}: 'sf_vec_size' is {sf_vec_size}; this MMA scales each 32 K elements"
        );
        let (m, n, k) = instruction_shape.mnk();
        let k = k.unwrap_or(64);
        assert!(k == 64, "cute: tcgen05.{name}: the instruction shape's K is {k}, but this MMA has K = 64");
        check_block_scaled_shape(name, (m, n), cta_group, b_dtype, b_major_mode);
        super::mma::check_pair_a_source(name, cta_group, a_src);
        super::mma::check_a_source(name, a_src, a_major_mode, a_dtype);
        check_collectors(name, a_src, a_collector_op);
        let scale = UmmaScale { sf: sf_dtype, vec: 32, k96: false };
        let kind = rubin_kind(
            UmmaDataKind::Mxf8f6f4,
            (m, n),
            (a_dtype, b_dtype, DType::F32),
            Some(scale),
            true,
            cta_group,
            a_src,
            (a_major_mode, b_major_mode),
            (a_collector_op, b_collector_op),
        );
        SM107BlockScaledMmaMXF8F6F4Op { kind }
    }

    /// Float8E8M0FNU.
    pub fn sf_dtype(&self) -> DType {
        self.kind.scale.expect("a block-scaled MMA").sf
    }

    /// 32.
    pub fn sf_vec_size(&self) -> usize {
        32
    }
}

rubin_accessors!(SM107BlockScaledMmaMXF8F6F4Op);

/// CuTe DSL's `tcgen05.SM107MmaMXF4NVF4Op(sf_dtype, sf_vec_size,
/// instruction_shape, cta_group, a_src, a_collector_op, b_collector_op)`:
/// `tcgen05.mma.cta_group::N.kind::mxf4nvf4.block_scale.block16` (or
/// `.block32` for 32-element vectors) with K = 128 (the descriptor's K-size
/// value 2: bit 3 set, bit 31 clear) on sm_107f and sm_107a (PTX 9.4):
/// packed K-major E2M1 operands, one scale factor per `sf_vec_size` (16 or
/// 32) K elements (eight or four per row per instruction), UE8M0
/// (Float8E8M0FNU), UE4M3 (Float8E4M3FN) or UE5M3 (FloatNV8E5M3FNU, scale
/// format 2), into Float32. M = 128 on one CTA, 256 on a pair.
///
/// A row's eight factors of a K = 128, 16-element MMA take two groups of
/// four bytes in TMEM, the second `max(4, rows / 32)` columns after the
/// first (CuTe C++'s `sm107_tmem_sf_frg` tiles the factors past the fourth
/// that way; the PTX ISA calls this `.scale_vec::8X` data).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SM107MmaMXF4NVF4Op {
    kind: UmmaKind,
}

impl SM107MmaMXF4NVF4Op {
    #[track_caller]
    pub fn new(
        sf_dtype: DType,
        sf_vec_size: usize,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_collector_op: CollectorOp,
        b_collector_op: CollectorOp,
    ) -> SM107MmaMXF4NVF4Op {
        let name = "SM107MmaMXF4NVF4Op";
        assert!(
            matches!(sf_dtype, DType::F8E8M0FNU | DType::F8E4M3FN | DType::F8E5M3FNU),
            "cute: tcgen05.{name}: 'sf_dtype' is {sf_dtype}; this MMA's scale factors are Float8E8M0FNU (UE8M0), \
             Float8E4M3FN (UE4M3) or FloatNV8E5M3FNU (UE5M3)"
        );
        assert!(
            sf_vec_size == 16 || sf_vec_size == 32,
            "cute: tcgen05.{name}: 'sf_vec_size' is {sf_vec_size}; this MMA scales each 16 or 32 K elements"
        );
        let (m, n, k) = instruction_shape.mnk();
        let k = k.unwrap_or(128);
        assert!(k == 128, "cute: tcgen05.{name}: the instruction shape's K is {k}, but this MMA has K = 128");
        let fp4 = DType::F4E2M1FN;
        let k_major = (OperandMajorMode::K, OperandMajorMode::K);
        check_block_scaled_shape(name, (m, n), cta_group, fp4, OperandMajorMode::K);
        super::mma::check_pair_a_source(name, cta_group, a_src);
        super::mma::check_a_source(name, a_src, OperandMajorMode::K, fp4);
        check_collectors(name, a_src, a_collector_op);
        let scale = UmmaScale { sf: sf_dtype, vec: sf_vec_size as u32, k96: false };
        let kind = rubin_kind(
            UmmaDataKind::Mxf4Nvf4,
            (m, n),
            (fp4, fp4, DType::F32),
            Some(scale),
            true,
            cta_group,
            a_src,
            k_major,
            (a_collector_op, b_collector_op),
        );
        SM107MmaMXF4NVF4Op { kind }
    }

    /// The scale factors' type.
    pub fn sf_dtype(&self) -> DType {
        self.kind.scale.expect("a block-scaled MMA").sf
    }

    /// 16 or 32.
    pub fn sf_vec_size(&self) -> usize {
        self.kind.scale.expect("a block-scaled MMA").vec as usize
    }
}

rubin_accessors!(SM107MmaMXF4NVF4Op);

#[cfg(test)]
mod tests {
    use super::*;
    use CollectorOp::{DISCARD, FILL, LASTUSE, USE};
    use OperandMajorMode::K;

    #[test]
    fn rubin_descriptors_and_ptx() {
        let (one, two, s) = (CtaGroup::ONE, CtaGroup::TWO, OperandSource::SMEM);
        // FP8 K = 64: the dense descriptor with bit 29 (CuTe C++'s make_sm107_instr_desc).
        let f = SM107MmaFP8Op::new(DType::F8E4M3FN, DType::F8E5M2, DType::F32, (128, 64), one, s, K, K, DISCARD, FILL)
            .kind();
        assert_eq!(f.k(), 64);
        assert_eq!(f.idesc(), 1 << 4 | 1 << 10 | 8 << 17 | 8 << 24 | 1 << 29);
        assert_eq!(f.ptx(), "tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::fill");
        // K = 32 keeps SM100's descriptor.
        let f32k =
            SM107MmaFP8Op::new(DType::F8E4M3FN, DType::F8E4M3FN, DType::F16, (64, 32, 32), one, s, K, K, USE, DISCARD)
                .kind();
        assert_eq!((f32k.k(), f32k.idesc() >> 29 & 1), (32, 0));
        assert_eq!(f32k.ptx(), "tcgen05.mma.cta_group::1.kind::f8f6f4.collector::a::use");
        // MXF8F6F4 K = 64: bit 31, UE8M0 (format 1 at bit 23), M = 128 at bit 27.
        let m = SM107BlockScaledMmaMXF8F6F4Op::new(
            DType::F4E2M1FN,
            DType::F8E4M3FN,
            DType::F8E8M0FNU,
            32,
            (128, 128),
            one,
            s,
            K,
            K,
            DISCARD,
            LASTUSE,
        )
        .kind();
        assert_eq!((m.k(), m.scale_factors()), (64, 2));
        assert_eq!(m.idesc(), 5 << 7 | 16 << 17 | 1 << 23 | 1 << 27 | 1 << 31);
        assert_eq!(m.ptx(), "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.collector::b::lastuse");
        // MXF4NVF4 K = 128 with UE5M3 (format 2): K-size 2 (bit 3), a pair's M = 256 at bit 28.
        let n = SM107MmaMXF4NVF4Op::new(DType::F8E5M3FNU, 16, (256, 64), two, s, DISCARD, DISCARD).kind();
        assert_eq!((n.k(), n.scale_factors()), (128, 8));
        assert_eq!(n.idesc(), 1 << 3 | 1 << 7 | 1 << 10 | 8 << 17 | 2 << 23 | 1 << 28);
        assert_eq!(n.ptx(), "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16");
        let n32 = SM107MmaMXF4NVF4Op::new(DType::F8E8M0FNU, 32, (128, 8), one, s, FILL, USE).kind();
        assert_eq!(n32.scale_factors(), 4);
        assert_eq!(
            n32.ptx(),
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill.collector::b::use"
        );
        // with_collectors changes only the qualifiers.
        let op = SM107MmaMXF4NVF4Op::new(DType::F8E8M0FNU, 32, (128, 8), one, s, DISCARD, DISCARD);
        assert_eq!(op.with_collectors(FILL, USE).kind(), n32);
    }
}

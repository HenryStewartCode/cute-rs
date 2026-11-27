// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm100_umma.hpp (the shape checks of SM100_MMA_MXF8F6F4_SS,
//   SM100_MMA_MXF8F6F4_2x1SM_SS, SM100_MMA_MXF4_SS and SM100_MMA_MXF4_2x1SM_SS, and
//   their PTX forms) and include/cute/atom/mma_traits_sm100_frag.hpp (the atom of
//   UMMA::tmem_sf_frg, the TMEM scale-factor layout, in tmem_sf_frg_layout),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! The block-scaled tcgen05 MMAs:
//! CuTe DSL's `tcgen05.MmaMXF8F6F4Op` (and its deprecated FP8 form
//! `MmaMXF8Op`), `MmaMXF4Op`, `MmaMXF4NVF4Op`, and the sm_103a K = 96 forms
//! `SM103MmaMXF4Op` and `SM103MmaMXF4NVF4Op`.
//!
//! Each multiplies A and B, each K-block of `sf_vec_size` elements scaled
//! by a scale factor held in tensor memory, into a Float32 TMEM
//! accumulator. The scale factors are staged into TMEM (by `tcgen05.cp`
//! from shared memory, or `tcgen05.st`) in the layout of
//! [`tmem_sf_frg_layout`] and passed to [`gemm`](crate::gemm) as `(&a,
//! &sfa)` and `(&b, &sfb)` (CuTe DSL's `[a, sfa]`), or set with
//! [`TiledMma::set_tensor`](crate::TiledMma::set_tensor)`(Field::SFA, ..)`.
//! For each MMA, `gemm` passes the TMEM address of the scale-factor tensor
//! (sliced at the MMA's `(MMA_MN, MMA_K)` coordinates when it has those
//! modes) and puts the byte of the 32-bit column its first element is in
//! into the instruction descriptor's scale-factor ID (CuTe C++'s
//! `make_runtime_instr_desc_block_scaled`).
//!
//! With `CtaGroup::ONE` M is 128; with `CtaGroup::TWO` M is 128 or 256,
//! N a multiple of 16. The block-scaled MMAs take no
//! `DISABLE_OUTPUT_LANE`.

use super::mma::OperandSource;
use crate::atom::mma::{MmaAtom, MmaOp};
use crate::ir::{DType, UmmaDataKind, UmmaKind, UmmaScale};
use crate::layout::Layout;
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::cpasync::CtaGroup;
use crate::nvgpu::warpgroup::InstructionShape;

/// The operand types of `.kind::mxf8f6f4`.
const MXF8F6F4_TYPES: [DType; 5] = [DType::F8E4M3FN, DType::F8E5M2, DType::F6E2M3FN, DType::F6E3M2FN, DType::F4E2M1FN];

/// The checks of every block-scaled MMA, and its instruction.
#[track_caller]
fn block_scaled(
    name: &str,
    kind: UmmaDataKind,
    (a, b): (DType, DType),
    scale: UmmaScale,
    instruction_shape: impl InstructionShape,
    cta_group: CtaGroup,
    a_src: OperandSource,
    (a_major_mode, b_major_mode): (OperandMajorMode, OperandMajorMode),
) -> UmmaKind {
    let (m, n, k) = instruction_shape.mnk();
    let want_k = if scale.k96 { 96 } else { kind.k() };
    let got = k.unwrap_or(want_k);
    assert!(got == want_k, "cute: tcgen05.{name}: the instruction shape's K is {got}, but this MMA has K = {want_k}");
    if cta_group == CtaGroup::ONE {
        assert!(
            m == 128,
            "cute: tcgen05.{name}: the instruction shape's M is {m}; a block-scaled MMA on one CTA (CtaGroup.ONE) has \
             M = 128"
        );
    }
    super::mma::check_pair_a_source(name, cta_group, a_src);
    super::mma::check_instruction_shape(name, (m, n), cta_group, b, b_major_mode);
    super::mma::check_a_source(name, a_src, a_major_mode, a);
    UmmaKind {
        kind,
        m: m as u32,
        n: n as u32,
        a,
        b,
        acc: DType::F32,
        trans_a: a_major_mode == OperandMajorMode::MN,
        trans_b: b_major_mode == OperandMajorMode::MN,
        two_cta: cta_group == CtaGroup::TWO,
        a_tmem: a_src == OperandSource::TMEM,
        scale: Some(scale),
        sparse: false,
        sm107: None,
        prims: Default::default(),
    }
}

/// The scale-factor type of an NVF4 MMA: UE4M3 or UE8M0.
#[track_caller]
fn check_nvf4_sf(name: &str, sf_dtype: DType) {
    assert!(
        matches!(sf_dtype, DType::F8E4M3FN | DType::F8E8M0FNU),
        "cute: tcgen05.{name}: 'sf_dtype' is {sf_dtype}; NVF4 scale factors are Float8E4M3FN (UE4M3) or \
         Float8E8M0FNU (UE8M0)"
    );
}

macro_rules! block_scaled_accessors {
    ($t:ident) => {
        impl $t {
            pub fn a_dtype(&self) -> DType {
                self.kind.a
            }

            pub fn b_dtype(&self) -> DType {
                self.kind.b
            }

            /// Always Float32.
            pub fn acc_dtype(&self) -> DType {
                self.kind.acc
            }

            /// The scale factors' type (CuTe DSL's `sf_dtype`).
            pub fn sf_dtype(&self) -> DType {
                self.kind.scale.expect("a block-scaled MMA").sf
            }

            /// The K elements each scale factor scales (CuTe DSL's `sf_vec_size`).
            pub fn sf_vec_size(&self) -> usize {
                self.kind.scale.expect("a block-scaled MMA").vec as usize
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

/// CuTe DSL's `tcgen05.MmaMXF8F6F4Op(a_dtype, b_dtype, instruction_shape,
/// cta_group, a_src, a_major_mode, b_major_mode)`:
/// `tcgen05.mma.cta_group::N.kind::mxf8f6f4.block_scale` (sm_100a and its
/// family), K = 32, A and B each Float8E4M3FN, Float8E5M2, Float6E2M3FN,
/// Float6E3M2FN or Float4E2M1FN, one UE8M0 (Float8E8M0FNU) scale factor per
/// 32 K elements of each row of A and column of B, into Float32.
///
/// As for [`MmaF8F6F4Op`](super::MmaF8F6F4Op), FP6 and FP4 operands are
/// K-major and read from the padded shared-memory image (a Uint8 tensor),
/// and 8-bit operands may be MN-major.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaMXF8F6F4Op {
    kind: UmmaKind,
}

impl MmaMXF8F6F4Op {
    #[track_caller]
    pub fn new(
        a_dtype: DType,
        b_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaMXF8F6F4Op {
        Self::checked(
            "MmaMXF8F6F4Op",
            a_dtype,
            b_dtype,
            instruction_shape,
            cta_group,
            a_src,
            a_major_mode,
            b_major_mode,
        )
    }

    #[track_caller]
    fn checked(
        name: &str,
        a_dtype: DType,
        b_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaMXF8F6F4Op {
        for (what, t, major) in [("a_dtype", a_dtype, a_major_mode), ("b_dtype", b_dtype, b_major_mode)] {
            assert!(
                MXF8F6F4_TYPES.contains(&t),
                "cute: tcgen05.{name}: '{what}' is {t}, which .kind::mxf8f6f4 does not take (it takes Float8E4M3FN, \
                 Float8E5M2, Float6E2M3FN, Float6E3M2FN and Float4E2M1FN)"
            );
            assert!(
                t.bits() == 8 || major == OperandMajorMode::K,
                "cute: tcgen05.{name}: '{what}' is {t} and MN-major; FP6 and FP4 operands are read K-major from their \
                 padded 16-byte slots, so only 8-bit operands may be MN-major"
            );
        }
        let scale = UmmaScale { sf: DType::F8E8M0FNU, vec: 32, k96: false };
        let kind = block_scaled(
            name,
            UmmaDataKind::Mxf8f6f4,
            (a_dtype, b_dtype),
            scale,
            instruction_shape,
            cta_group,
            a_src,
            (a_major_mode, b_major_mode),
        );
        MmaMXF8F6F4Op { kind }
    }
}

block_scaled_accessors!(MmaMXF8F6F4Op);

/// CuTe DSL's deprecated `tcgen05.MmaMXF8Op(ab_dtype, instruction_shape,
/// cta_group, a_src, a_major_mode, b_major_mode)`: [`MmaMXF8F6F4Op`] with A
/// and B both Float8E4M3FN or both Float8E5M2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaMXF8Op {
    kind: UmmaKind,
}

impl MmaMXF8Op {
    #[deprecated(note = "use tcgen05::MmaMXF8F6F4Op")]
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaMXF8Op {
        assert!(
            matches!(ab_dtype, DType::F8E4M3FN | DType::F8E5M2),
            "cute: tcgen05.MmaMXF8Op: 'ab_dtype' is {ab_dtype}; its operands are both Float8E4M3FN or both Float8E5M2 \
             (MmaMXF8F6F4Op takes the other formats and mixed pairs)"
        );
        let op = MmaMXF8F6F4Op::checked(
            "MmaMXF8Op",
            ab_dtype,
            ab_dtype,
            instruction_shape,
            cta_group,
            a_src,
            a_major_mode,
            b_major_mode,
        );
        MmaMXF8Op { kind: op.kind }
    }
}

block_scaled_accessors!(MmaMXF8Op);

/// The FP4 block-scaled MMAs' common construction.
#[track_caller]
fn fp4(
    name: &str,
    kind: UmmaDataKind,
    scale: UmmaScale,
    instruction_shape: impl InstructionShape,
    cta_group: CtaGroup,
    a_src: OperandSource,
) -> UmmaKind {
    let fp4 = DType::F4E2M1FN;
    let k_major = (OperandMajorMode::K, OperandMajorMode::K);
    block_scaled(name, kind, (fp4, fp4), scale, instruction_shape, cta_group, a_src, k_major)
}

/// CuTe DSL's `tcgen05.MmaMXF4Op(instruction_shape, cta_group, a_src)`:
/// `tcgen05.mma.cta_group::N.kind::mxf4.block_scale.scale_vec::2X`
/// (sm_100a, sm_103a, sm_107a, sm_110a; not the `f` families), K = 64,
/// E2M1 (Float4E2M1FN) A and B packed two per byte in shared memory (as a
/// TMA load with the 16U4_ALIGN8B format leaves them) and K-major, one UE8M0
/// scale factor per 32 K elements (two per row per instruction).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaMXF4Op {
    kind: UmmaKind,
}

impl MmaMXF4Op {
    #[track_caller]
    pub fn new(instruction_shape: impl InstructionShape, cta_group: CtaGroup, a_src: OperandSource) -> MmaMXF4Op {
        let scale = UmmaScale { sf: DType::F8E8M0FNU, vec: 32, k96: false };
        MmaMXF4Op { kind: fp4("MmaMXF4Op", UmmaDataKind::Mxf4, scale, instruction_shape, cta_group, a_src) }
    }
}

block_scaled_accessors!(MmaMXF4Op);

/// CuTe DSL's `tcgen05.MmaMXF4NVF4Op(sf_dtype, instruction_shape,
/// cta_group, a_src)`:
/// `tcgen05.mma.cta_group::N.kind::mxf4nvf4.block_scale.scale_vec::4X`
/// (PTX 8.7; the targets of [`MmaMXF4Op`]), K = 64, packed K-major E2M1
/// operands, one scale factor per 16 K elements (four per row per
/// instruction): NVFP4 with UE4M3 (Float8E4M3FN, `sf_dtype`) scales, or
/// UE8M0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaMXF4NVF4Op {
    kind: UmmaKind,
}

impl MmaMXF4NVF4Op {
    #[track_caller]
    pub fn new(
        sf_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
    ) -> MmaMXF4NVF4Op {
        check_nvf4_sf("MmaMXF4NVF4Op", sf_dtype);
        let scale = UmmaScale { sf: sf_dtype, vec: 16, k96: false };
        MmaMXF4NVF4Op { kind: fp4("MmaMXF4NVF4Op", UmmaDataKind::Mxf4Nvf4, scale, instruction_shape, cta_group, a_src) }
    }
}

block_scaled_accessors!(MmaMXF4NVF4Op);

/// CuTe DSL's `tcgen05.SM103MmaMXF4Op(instruction_shape, cta_group,
/// a_src)`: [`MmaMXF4Op`] with K = 96
/// (`tcgen05.mma.cta_group::N.kind::mxf4.block_scale.block32`, PTX 8.8,
/// sm_103a and sm_107a), three UE8M0 scale factors per row per instruction.
/// Each operand tile is 48 bytes of K.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SM103MmaMXF4Op {
    kind: UmmaKind,
}

impl SM103MmaMXF4Op {
    #[track_caller]
    pub fn new(instruction_shape: impl InstructionShape, cta_group: CtaGroup, a_src: OperandSource) -> SM103MmaMXF4Op {
        let scale = UmmaScale { sf: DType::F8E8M0FNU, vec: 32, k96: true };
        SM103MmaMXF4Op { kind: fp4("SM103MmaMXF4Op", UmmaDataKind::Mxf4, scale, instruction_shape, cta_group, a_src) }
    }
}

block_scaled_accessors!(SM103MmaMXF4Op);

/// CuTe DSL's `tcgen05.SM103MmaMXF4NVF4Op(sf_dtype, instruction_shape,
/// cta_group, a_src)`: [`MmaMXF4NVF4Op`] with K = 96
/// (`...kind::mxf4nvf4.block_scale.block16`, PTX 8.8, sm_103a and sm_107a),
/// six scale factors per row per instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SM103MmaMXF4NVF4Op {
    kind: UmmaKind,
}

impl SM103MmaMXF4NVF4Op {
    #[track_caller]
    pub fn new(
        sf_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
    ) -> SM103MmaMXF4NVF4Op {
        check_nvf4_sf("SM103MmaMXF4NVF4Op", sf_dtype);
        let scale = UmmaScale { sf: sf_dtype, vec: 16, k96: true };
        SM103MmaMXF4NVF4Op {
            kind: fp4("SM103MmaMXF4NVF4Op", UmmaDataKind::Mxf4Nvf4, scale, instruction_shape, cta_group, a_src),
        }
    }
}

block_scaled_accessors!(SM103MmaMXF4NVF4Op);

/// The TMEM placement of one scale factor: factor `j` of row `row` (of A's rows, or B's
/// columns, in one CTA) for an instruction reading `nsf` factors per row,
/// starting at byte `sf_id` of each column. Returns the lane within a
/// 32-lane sub-partition, the column after the scale factors' address and
/// the byte of that column; the factor is replicated in the same lane of
/// all four sub-partitions.
///
/// Row `32 c + l` is in lane `l` and column `c`, its factors in
/// consecutive bytes (CuTe C++'s `UMMA::tmem_sf_frg`, whose 128-row atom
/// takes four columns); factors past the fourth byte of a column (from
/// a nonzero `sf_id`, or the six of a K = 96 `.block16` MMA) continue in
/// the same lane in the next block of `max(4, rows / 32)` columns, as
/// CUTLASS's `sm107_tmem_sf_frg` tiles them. (cute-rs's model: the PTX
/// ISA's K = 96 figures were not available when this was written.)
pub(crate) fn sf_place(rows: usize, row: usize, sf_id: usize, j: usize) -> (usize, usize, usize) {
    let byte = sf_id + j;
    // The factors of a row block of up to 128 rows take four columns (CUTLASS pads N to 128).
    let groups = rows.div_ceil(32).max(4);
    (row % 32, row / 32 + groups * (byte / 4), byte % 4)
}

/// The TMEM layout of one CTA's scale factors for a block-scaled MMA (the
/// atom of CuTe C++'s `UMMA::tmem_sf_frg<T, sf_vec_size>` for SFA, and SFB
/// with the 4x1 data path): `((MN, (sf_vec_size, nsf)), REST...) ->` TMEM
/// element offset of 8-bit elements, `mn` rows (a multiple of 32) by `nsf`
/// factors (at most 4) of `sf_vec_size` elements each, tiled over `rest`
/// (MMA_MN, MMA_K, ...). Row `32 c + l` is in lane `l` (of each of the four
/// sub-partitions, whose copies the layout does not name: write them with
/// a broadcast `tcgen05.cp`, [`Cp4x32x128bOp`](super::Cp4x32x128bOp)) and
/// column `c`, its factors in consecutive bytes from byte 0. The tiles of
/// `rest` follow each other (column-major) `mn / 32` columns apart, each
/// starting at byte 0 (cute-rs's tiling; CuTe C++ tiles the atom with
/// `tiled_product`). A tensor over this layout, on the TMEM pointer of the
/// scale factors' columns, is what [`gemm`](crate::gemm) takes as SFA or
/// SFB.
pub fn tmem_sf_frg_layout(mn: usize, sf_vec_size: usize, nsf: usize, rest: &[usize]) -> Layout {
    assert!(mn.is_multiple_of(32) && mn > 0, "tmem_sf_frg_layout: {mn} rows is not a multiple of 32");
    assert!((1..=4).contains(&nsf), "tmem_sf_frg_layout: {nsf} scale factors per row (1 to 4 fit a column)");
    // Element offsets of 8-bit elements: lane `l` at `l << 18` (`1 << 16` columns of 4 bytes).
    let atom = Layout::new(((32, mn / 32), (sf_vec_size, nsf)), ((1usize << 18, 4), (0, 1)));
    let rest = if rest.is_empty() { vec![1] } else { rest.to_vec() };
    let mut modes = vec![atom];
    let mut stride = 4 * (mn / 32);
    for &r in &rest {
        modes.push(Layout::new(r, stride));
        stride *= r;
    }
    Layout::from_modes(modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use OperandMajorMode::K;

    #[test]
    fn block_scaled_descriptors() {
        let s = OperandSource::SMEM;
        // MXF8F6F4, E2M1 (5) x E4M3 (0), 128 x 256: scale format UE8M0 (bit 23), no c_format.
        let f = MmaMXF8F6F4Op::new(DType::F4E2M1FN, DType::F8E4M3FN, (128, 256), CtaGroup::ONE, s, K, K).kind();
        assert_eq!(f.idesc(), 5 << 7 | 32 << 17 | 1 << 23 | 8 << 24);
        assert_eq!(f.ptx(), "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale");
        // MXF4: E2M1 format 1, K = 64.
        let m = MmaMXF4Op::new((128, 64), CtaGroup::ONE, s).kind();
        assert_eq!((m.k(), m.scale_factors()), (64, 2));
        assert_eq!(m.idesc(), 1 << 7 | 1 << 10 | 8 << 17 | 1 << 23 | 8 << 24);
        // NVF4 with UE4M3: scale format 0; a pair's M = 256.
        let n = MmaMXF4NVF4Op::new(DType::F8E4M3FN, (256, 128), CtaGroup::TWO, s).kind();
        assert_eq!(n.idesc(), 1 << 7 | 1 << 10 | 16 << 17 | 16 << 24);
        assert_eq!(n.ptx(), "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X");
        // SM103 K = 96: bit 31.
        let k = SM103MmaMXF4NVF4Op::new(DType::F8E8M0FNU, (128, 32), CtaGroup::ONE, s).kind();
        assert_eq!((k.k(), k.scale_factors()), (96, 6));
        assert_eq!(k.idesc(), 1 << 7 | 1 << 10 | 4 << 17 | 1 << 23 | 8 << 24 | 1 << 31);
        assert_eq!(k.ptx(), "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16");
    }

    #[test]
    fn sf_layout() {
        // 128 rows, 4 factors of 16: row 32 c + l in lane l, column c, factor j in byte j.
        let l = tmem_sf_frg_layout(128, 16, 4, &[1, 2]);
        let lane = 1 << 18;
        for (row, j) in [(0, 0), (5, 3), (37, 1), (127, 2)] {
            let (ln, col, byte) = sf_place(128, row, 0, j);
            assert_eq!(l.call(row + 128 * 16 * j), ln * lane + 4 * col + byte);
        }
        // The second K tile after the first's four columns.
        assert_eq!(l.call(128 * 64), 16);
    }
}

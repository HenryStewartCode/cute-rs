// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/mma_sm100_umma.hpp (SM100_MMA_F16BF16_SS's shape checks
//   and its issue), include/cute/atom/mma_traits_sm100.hpp (the
//   MMA_Traits<SM100_MMA_F16BF16_SS> layouts and fragment kinds) and
//   include/cute/atom/mma_traits_sm100_frag.hpp (UMMA::tmem_frg_1sm, the TMEM
//   accumulator layout); m10-umma-kinds: the _TS forms' K-major A rule and
//   UMMA::tmem_frg_1sm<T, T, TmemAllocMode::NonInterleaved> (the TMEM A layout);
//   m10-cta-pair: the SM100_MMA_*_2x1SM_SS shape checks and UMMA::tmem_frg_2sm
//   (the CTA-pair accumulator layouts),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! tcgen05 MMAs: CuTe DSL's
//! `tcgen05.MmaF16BF16Op` with its [`CtaGroup`], [`OperandSource`] and
//! [`Field`] enums, and [`commit`].
//!
//! A tcgen05 MMA is issued by one thread for the whole CTA, so the
//! "threads" of a tcgen05 [`TiledMma`](crate::TiledMma) are CTAs: its
//! `thr_id` has size 1, and `get_slice(0)` partitions whole per-CTA tiles;
//! with `CtaGroup::TWO` it has size 2, and
//! `get_slice(rank_in_pair)` partitions each CTA's half.
//! A and B are shared-memory descriptor fragments
//! ([`TiledMma::make_fragment_a`](crate::TiledMma::make_fragment_a)), the
//! accumulator a TMEM tensor ([`TiledMma::make_fragment_c`](crate::TiledMma::make_fragment_c),
//! CuTe C++'s `UMMA::tmem_frg_1sm` layout), and [`gemm`](crate::gemm),
//! called by a whole warp, elects one lane to issue each MMA. The MMAs are
//! asynchronous: [`commit`] makes an mbarrier arrival once they complete,
//! and D may be read (`tcgen05.ld`) only after a wait on that mbarrier.

pub use super::super::cpasync::CtaGroup;
pub use crate::atom::fields::MmaField as Field;
use crate::atom::mma::{MmaAtom, MmaOp};
use crate::builder;
use crate::ir::{DType, Op, UmmaDataKind, UmmaKind};
use crate::layout::Layout;
use crate::nvgpu::OperandMajorMode;
use crate::nvgpu::warpgroup::InstructionShape;
use crate::tensor::{AddressSpace, Engine, Ptr, Tensor, TensorInner};
use crate::value::{Int16, Numeric, Value};

/// Where the A operand of a tcgen05 MMA comes from (CuTe DSL's
/// `tcgen05.OperandSource`): shared memory through a descriptor, or tensor
/// memory (the `_TS` forms).
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperandSource {
    TMEM,
    SMEM,
}

/// CuTe DSL's `tcgen05.MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape,
/// cta_group, a_src, a_major_mode, b_major_mode)`: the sm_100a MMA
/// `tcgen05.mma.cta_group::1.kind::f16` on Float16 or BFloat16 operands in
/// shared memory, accumulated in tensor memory.
///
/// | A/B  | accumulator | shape (cta_group::1)                     |
/// |------|-------------|------------------------------------------|
/// | F16  | F16, F32    | M of 64 or 128, N of 8 to 256 in steps of 8, K 16 |
/// | BF16 | F32         | (the same)                               |
///
/// A and B may each be K-major or MN-major. The atom takes the runtime
/// field [`Field::ACCUMULATE`] (`false` until set: the first MMA of a tile
/// overwrites D). With `CtaGroup::TWO` the even
/// CTA of a cluster pair issues one MMA for both CTAs: M of 128 or 256, N
/// of 16 to 256 in steps of 16, each CTA holding half of A's rows and half
/// of B's rows in its shared memory and its half of D in its TMEM. With `OperandSource::TMEM` A is
/// a K-major TMEM tensor ([`TiledMma::make_fragment_a`](crate::TiledMma::make_fragment_a)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16Op {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
    cta_group: CtaGroup,
    a_src: OperandSource,
    a_major_mode: OperandMajorMode,
    b_major_mode: OperandMajorMode,
}

impl MmaF16BF16Op {
    /// The operation, or a panic naming the parameter no instruction takes
    /// (as CuTe DSL raises at construction).
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        cta_group: CtaGroup,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaF16BF16Op {
        let (m, n, k) = instruction_shape.mnk();
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: tcgen05.MmaF16BF16Op: 'ab_dtype' is {ab_dtype}; its A and B operands are Float16 or BFloat16"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: tcgen05.MmaF16BF16Op: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: tcgen05.MmaF16BF16Op: BFloat16 operands ('ab_dtype') accumulate in Float32 only, and 'acc_dtype' \
             is {acc_dtype}"
        );
        let k = k.unwrap_or(16);
        assert!(
            k == 16,
            "cute: tcgen05.MmaF16BF16Op: the instruction shape's K is {k}; Float16 and BFloat16 MMAs have K = 16"
        );
        check_pair_a_source("MmaF16BF16Op", cta_group, a_src);
        check_instruction_shape("MmaF16BF16Op", (m, n), cta_group, ab_dtype, b_major_mode);
        check_a_source("MmaF16BF16Op", a_src, a_major_mode, ab_dtype);
        MmaF16BF16Op { ab_dtype, acc_dtype, shape_mnk: (m, n, k), cta_group, a_src, a_major_mode, b_major_mode }
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

    pub fn cta_group(&self) -> CtaGroup {
        self.cta_group
    }

    pub fn a_src(&self) -> OperandSource {
        self.a_src
    }

    pub fn a_major_mode(&self) -> OperandMajorMode {
        self.a_major_mode
    }

    pub fn b_major_mode(&self) -> OperandMajorMode {
        self.b_major_mode
    }

    /// The instruction it issues.
    pub fn kind(&self) -> UmmaKind {
        UmmaKind {
            kind: UmmaDataKind::F16,
            m: self.shape_mnk.0 as u32,
            n: self.shape_mnk.1 as u32,
            a: self.ab_dtype,
            b: self.ab_dtype,
            acc: self.acc_dtype,
            trans_a: self.a_major_mode == OperandMajorMode::MN,
            trans_b: self.b_major_mode == OperandMajorMode::MN,
            two_cta: self.cta_group == CtaGroup::TWO,
            a_tmem: self.a_src == OperandSource::TMEM,
            scale: None,
            sparse: false,
            sm107: None,
            prims: Default::default(),
        }
    }
}

/// The instruction shape `(M, N)` of a tcgen05 MMA for its `cta_group`
/// (the PTX ISA's shapes for `.kind::f16`, `tf32`, `i8` and `f8f6f4`):
/// with `CtaGroup::ONE` M is 64 or 128 and N a multiple of 8 from 8 to 256;
/// with `CtaGroup::TWO` (a CTA pair) M is 128 or
/// 256 and N a multiple of 16 from 16 to 256. An 8-bit MN-major B doubles
/// N's step (16, or 32 for a pair). Returns `(M, N)`.
#[track_caller]
pub(super) fn check_instruction_shape(
    name: &str,
    (m, n): (usize, usize),
    cta_group: CtaGroup,
    b_dtype: DType,
    b_major_mode: OperandMajorMode,
) -> (usize, usize) {
    let pair = cta_group == CtaGroup::TWO;
    let (ms, group) = if pair { ([128, 256], "CtaGroup.TWO") } else { ([64, 128], "CtaGroup.ONE") };
    assert!(
        ms.contains(&m),
        "cute: tcgen05.{name}: the instruction shape's M is {m}; with {group}, M is {} or {}",
        ms[0],
        ms[1]
    );
    let wide_b = b_dtype.bits() == 8 && b_major_mode == OperandMajorMode::MN;
    let step = (if pair { 16 } else { 8 }) * if wide_b { 2 } else { 1 };
    assert!(
        (step..=256).contains(&n) && n.is_multiple_of(step),
        "cute: tcgen05.{name}: the instruction shape's N is {n}; with {group}{}, N is a multiple of {step} from \
         {step} to 256",
        if wide_b { " and an 8-bit MN-major B" } else { "" }
    );
    (m, n)
}

/// A CTA pair's MMA reads A from shared memory. (CUTLASS's
/// 2-SM MMAs with A in tensor memory duplicate A in both CTAs' TMEM, a
/// layout cute-rs does not model yet.)
#[track_caller]
pub(super) fn check_pair_a_source(name: &str, cta_group: CtaGroup, a_src: OperandSource) {
    assert!(
        cta_group == CtaGroup::ONE || a_src == OperandSource::SMEM,
        "cute: tcgen05.{name}: cta_group=CtaGroup.TWO with a_src=OperandSource.TMEM: cute-rs models CTA-pair MMAs \
         with A in shared memory only (OperandSource.SMEM)"
    );
}

/// The checks of A in tensor memory (`a_src` TMEM, the
/// `[a-tmem]` form): A is K-major there (a transposed A comes from shared
/// memory), and of the F8F6F4 formats only 8-bit A is modelled in TMEM.
#[track_caller]
pub(super) fn check_a_source(name: &str, a_src: OperandSource, a_major_mode: OperandMajorMode, a_dtype: DType) {
    if a_src != OperandSource::TMEM {
        return;
    }
    assert!(
        a_major_mode == OperandMajorMode::K,
        "cute: tcgen05.{name}: a_src=OperandSource.TMEM with an MN-major A: A in tensor memory is always K-major, so \
         a transposed A must come from shared memory (OperandSource.SMEM)"
    );
    assert!(
        a_dtype.bits() >= 8,
        "cute: tcgen05.{name}: a_src=OperandSource.TMEM with a {a_dtype} A: cute-rs models FP6 and FP4 A operands in \
         shared memory only; pass A through OperandSource.SMEM"
    );
}

impl From<MmaF16BF16Op> for MmaOp {
    fn from(op: MmaF16BF16Op) -> MmaOp {
        MmaOp::Sm100(op.kind())
    }
}

impl From<MmaF16BF16Op> for MmaAtom {
    fn from(op: MmaF16BF16Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

/// CuTe DSL's `tcgen05.commit(mbar_ptr, mask=None, cta_group=CtaGroup.ONE)`
/// (`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster{.multicast::cluster}.b64`,
/// sm_100a): one arrival on the mbarrier at `mbar_ptr` once every tcgen05
/// MMA the executing thread issued before has completed; with `mask` (a
/// 16-bit mask of CTA ranks, from
/// `create_tma_multicast_mask`) on the
/// mbarrier at the same offset of each CTA it names. It is not elected:
/// every thread that executes it commits, so wrap it in
/// [`elect_one`](crate::elect_one), in the warp that called
/// [`gemm`](crate::gemm). With `CtaGroup::TWO` it tracks
/// the CTA-pair MMAs the thread issued; the leader CTA commits with a mask
/// naming both CTAs of the pair so that each CTA's mbarrier hears of the
/// completion (without the peer's bit the peer's waiters wait forever).
#[track_caller]
pub fn commit(mbar_ptr: &Ptr<u64>, mask: Option<Int16>, cta_group: CtaGroup) {
    let loc = std::panic::Location::caller();
    assert!(
        mbar_ptr.memspace() == AddressSpace::Smem,
        "cute: tcgen05.commit: the mbarrier pointer is in {:?}, not shared memory",
        mbar_ptr.memspace()
    );
    let (buffer, offset) = mbar_ptr.parts();
    let index = crate::tensor::index_operand(offset);
    let mask = mask.map(|m| m.checked());
    let two_cta = cta_group == CtaGroup::TWO;
    builder::with(|b| b.push(Op::Tcgen05Commit { buffer, index, mask, two_cta }, loc));
}

/// The descriptor fragment of a tcgen05 MMA operand (CuTe DSL's
/// `make_fragment_A/B` of a tcgen05 MMA, CuTe C++'s `UMMA::smem_desc`):
/// `input` is the partition of the shared-memory operand, `((MN, K),
/// MMA_MN, MMA_K, ...)`. The result has the same shape with the first mode
/// `1`, and each element is the UMMA descriptor of one MMA tile (the modes
/// after the first in 16-byte units of the start address, as
/// [`Tensor::smem_desc`] advances them).
#[track_caller]
pub(crate) fn make_umma_desc_fragment<T: Numeric>(
    input: &Tensor<T>,
    mn: usize,
    major: OperandMajorMode,
    what: &str,
    halve_k: bool,
    sm107: bool,
) -> Tensor<T> {
    use super::smem_desc::{UmmaDescEncoding, UmmaLayoutType, umma_desc_fields};
    let fail = |why: String| -> ! { panic!("cute: {what}: {why}") };
    let Engine::Memory { buffer, space: crate::ir::MemSpace::Shared, peer: None } = input.engine else {
        fail(format!(
            "a tcgen05 MMA reads this operand from shared memory through a descriptor, but the tensor {:?} is not in \
             shared memory",
            input
        ))
    };
    let layout_type = match &input.inner {
        TensorInner::Identity => UmmaLayoutType::SwizzleNone,
        TensorInner::Swizzle { bytes: Some(sw), elem, .. } => {
            UmmaLayoutType::from_byte_swizzle(*sw).unwrap_or_else(|| {
                fail(format!(
                    "the operand is swizzled by {elem} (on elements), which is no UMMA swizzle: build it from a \
                     tcgen05::make_smem_layout_atom atom"
                ))
            })
        }
        TensorInner::Swizzle { elem, .. } => fail(format!("the operand's swizzle {elem} acts within bytes")),
        TensorInner::Layout { .. } => fail("the operand is over a composed layout with a layout inner".into()),
    };
    let layout = input.layout();
    let tile = layout.mode(0);
    if tile.size() > 0 && tile.rank() == 2 && tile.mode(0).size() != mn {
        fail(format!("the operand's MMA tile {tile} does not have the instruction's {mn} rows"));
    }
    // B of a sparse MMA spans twice the canonical K (64 bytes); its descriptor
    // fields are those of the first half.
    let tile = if halve_k && tile.rank() == 2 {
        let k = tile.mode(1);
        Layout::from_modes([tile.mode(0), k.compose(&Layout::new(k.size() / 2, 1))])
    } else {
        tile
    };
    // A Rubin MMA's descriptors have the sm_107 encoding whatever the target.
    let enc = if sm107 {
        UmmaDescEncoding::Sm107
    } else {
        builder::target_arch().map(UmmaDescEncoding::for_arch).unwrap_or_default()
    };
    let fields = umma_desc_fields(&tile, major, layout_type, T::DTYPE.bits(), enc)
        .unwrap_or_else(|why| fail(format!("the operand's MMA tile is not a canonical UMMA layout: {why}")));
    let align = builder::with(|b| b.buffer(buffer).align);
    let repeat = layout_type.repeat_bytes();
    if align < repeat {
        fail(format!(
            "its shared-memory buffer is aligned to {align} bytes, but {layout_type:?} repeats every {repeat} bytes, \
             so the buffer must be aligned to {repeat} (make_smem_tensor_aligned(.., {repeat}))"
        ));
    }
    let bits = T::DTYPE.bits();
    let mut modes = vec![Layout::new(1, 0)];
    for i in 1..layout.rank() {
        let m = layout.mode(i);
        if m.shape().first_dynamic().is_some() {
            fail(format!("mode {i} of the operand ({m}) has a runtime extent"));
        }
        let stride = m.static_stride().map(&mut |&d| {
            let bytes = d * bits / 8;
            if !bytes.is_multiple_of(16) {
                fail(format!(
                    "mode {i} of the operand ({m}) has a stride of {d} elements, which does not start a 16-byte unit"
                ));
            }
            bytes / 16
        });
        modes.push(Layout::new(m.shape().clone(), stride));
    }
    let addr: Value<u32> = input.ptr().toint_i32().to::<u32>();
    let mask = (1u64 << enc.address_bits()) - 1;
    let start = (addr.to::<u64>() >> 4u64) & mask;
    let desc = start | fields.encode(enc);
    Tensor::new(Engine::SmemDesc { desc: desc.checked() }, Layout::from_modes(modes))
}

/// The TMEM accumulator layout of a `cta_group::1` MMA of `m` rows (64 or
/// 128) and `n` columns tiled over `rest` (MMA_M, MMA_N, ...), on elements
/// of `bits` bits each held in a 32-bit column (CuTe C++'s
/// `UMMA::tmem_frg_1sm<T>` with the interleaved allocation): `((M, N),
/// MMA_M, MMA_N, ...) ->` TMEM element offset (lane at a stride of `2^16`
/// 32-bit columns).
pub(crate) fn tmem_frg_1sm_layout(m: usize, n: usize, rest: &[usize], bits: usize) -> Layout {
    // "Virtual" addresses: data path (lane) stride 1, column stride 128.
    let atom = if m == 64 {
        Layout::new(((16, 4), n), ((1, 32), 128))
    } else {
        assert_eq!(m, 128, "tcgen05 MMAs with cta_group::1 have M of 64 or 128");
        Layout::new((128, n), (1, 128))
    };
    let rest_layout = Layout::col_major(rest.to_vec());
    let logical = crate::layout::tiled_product(&atom, &rest_layout);
    let restride = Layout::new((128, 16384), ((1usize << 21) / bits, 32 / bits));
    restride.compose(&logical)
}

/// The TMEM accumulator layout of one CTA's half of a
/// `cta_group::2` MMA, `m` (64 or 128, half the pair's M) rows by `n`
/// columns tiled over `rest` (CuTe C++'s `UMMA::tmem_frg_2sm<T>`, the
/// interleaved allocation): for `m` = 128 (a pair M of 256, the PTX ISA's
/// Layout A) row `i` in lane `i` and column `j` in column `j`; for `m` = 64
/// (a pair M of 128, Layout B) columns `0..n/2` in lanes 0 to 63 and
/// columns `n/2..n` in lanes 64 to 127, each half in `n / 2` columns.
pub(crate) fn tmem_frg_2sm_layout(m: usize, n: usize, rest: &[usize], bits: usize) -> Layout {
    let atom = if m == 64 {
        Layout::new((64, (n / 2, 2)), (1, (128, 64)))
    } else {
        assert_eq!(m, 128, "a CTA of a cta_group::2 MMA holds 64 or 128 rows of D");
        Layout::new((128, n), (1, 128))
    };
    let rest_layout = Layout::col_major(rest.to_vec());
    let logical = crate::layout::tiled_product(&atom, &rest_layout);
    let restride = Layout::new((128, 16384), ((1usize << 21) / bits, 32 / bits));
    restride.compose(&logical)
}

/// The TMEM layout of a `cta_group::1` MMA's A operand of
/// `m` rows (64 or 128) and `k` columns tiled over `rest` (MMA_M, MMA_K,
/// ...), on elements of `bits` bits packed `32 / bits` to a 32-bit column
/// (CuTe C++'s `UMMA::tmem_frg_1sm<T, T, TmemAllocMode::NonInterleaved>`):
/// `((M, K), MMA_M, MMA_K, ...) ->` TMEM element offset. For M = 64 the
/// tiles are not interleaved into the other half of each sub-partition
/// (CuTe C++ tiles with a stride of 2), so every A tile starts at lane 0,
/// as the D it multiplies into does.
pub(crate) fn tmem_frg_a_layout(m: usize, k: usize, rest: &[usize], bits: usize) -> Layout {
    let atom = if m == 64 {
        Layout::new(((16, 4), k), ((1, 32), 128))
    } else {
        assert_eq!(m, 128, "tcgen05 MMAs with cta_group::1 have M of 64 or 128");
        Layout::new((128, k), (1, 128))
    };
    let tile_stride = if m == 64 { 2 } else { 1 };
    let mut strides = Vec::with_capacity(rest.len());
    let mut acc = tile_stride;
    for &r in rest {
        strides.push(acc);
        acc *= r;
    }
    let rest_layout = Layout::new(rest.to_vec(), strides);
    let logical = crate::layout::tiled_product(&atom, &rest_layout);
    let restride = Layout::new((128, 16384), ((1usize << 21) / bits, 1));
    restride.compose(&logical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_descriptor_fields() {
        let op = |ab, acc, m, n, a, b| {
            MmaF16BF16Op::new(ab, acc, (m, n), CtaGroup::ONE, OperandSource::SMEM, a, b).kind().idesc()
        };
        use OperandMajorMode::{K, MN};
        // F16 x F16 -> F32, 128 x 256, both K-major: c_format 1, N>>3 = 32, M>>4 = 8.
        assert_eq!(op(DType::F16, DType::F32, 128, 256, K, K), 1 << 4 | 32 << 17 | 8 << 24);
        // BF16 -> F32, 64 x 8, A MN-major.
        assert_eq!(op(DType::BF16, DType::F32, 64, 8, MN, K), 1 << 4 | 1 << 7 | 1 << 10 | 1 << 15 | 1 << 17 | 4 << 24);
        // F16 -> F16, B MN-major.
        assert_eq!(op(DType::F16, DType::F16, 128, 128, K, MN), 1 << 16 | 16 << 17 | 8 << 24);
    }

    #[test]
    fn tmem_a_layouts() {
        // 16-bit A, M = 128, K = 16, two K tiles: row i in lane i, two values per column.
        let lane = 1 << 17;
        let l = tmem_frg_a_layout(128, 16, &[1, 2], 16);
        for (row, col) in [(0, 0), (5, 7), (127, 15)] {
            assert_eq!(l.call(row + 128 * col), row * lane + col);
        }
        assert_eq!(l.call(128 * 16), 16);
        // M = 64, 8-bit, K = 32: rows 16 q to 16 q + 15 in lanes 32 q onwards; the second K
        // tile after the first's columns, not in lanes 16 to 31.
        let lane = 1 << 18;
        let l = tmem_frg_a_layout(64, 32, &[1, 2], 8);
        for (row, col) in [(0, 0), (17, 3), (63, 31)] {
            assert_eq!(l.call(row + 64 * col), (32 * (row / 16) + row % 16) * lane + col);
        }
        assert_eq!(l.call(64 * 32), 32);
    }

    #[test]
    fn pair_accumulator_layouts() {
        let lane = 1 << 16;
        // Layout A (pair M = 256): each CTA's 128 rows in lanes 0 to 127.
        let l = tmem_frg_2sm_layout(128, 64, &[1, 1], 32);
        for (row, col) in [(0, 0), (5, 7), (127, 63)] {
            assert_eq!(l.call(row + 128 * col), row * lane + col);
        }
        // Layout B (pair M = 128): the second half of the columns in lanes 64 to 127.
        let l = tmem_frg_2sm_layout(64, 128, &[1, 1], 32);
        for (row, col) in [(0, 0), (5, 7), (63, 63), (0, 64), (17, 100), (63, 127)] {
            assert_eq!(l.call(row + 64 * col), (row + 64 * (col / 64)) * lane + col % 64);
        }
        // Two N tiles of 128: the second after the first's 64 columns.
        let l = tmem_frg_2sm_layout(64, 128, &[1, 2], 32);
        assert_eq!(l.call(64 * 128), 64);
    }

    #[test]
    fn tmem_accumulator_layouts() {
        let lane = 1 << 16;
        // M = 128: row i in lane i, column j in column j.
        let l = tmem_frg_1sm_layout(128, 64, &[1, 1], 32);
        for (row, col) in [(0, 0), (5, 7), (127, 63)] {
            assert_eq!(l.call(row + 128 * col), row * lane + col);
        }
        // Two N tiles: the second after the first's columns.
        let l = tmem_frg_1sm_layout(128, 64, &[1, 2], 32);
        assert_eq!(l.call(128 * 64), 64);
        // M = 64: rows 16 q to 16 q + 15 in lanes 32 q onwards.
        let l = tmem_frg_1sm_layout(64, 32, &[1, 1], 32);
        for (row, col) in [(0, 0), (17, 3), (63, 31)] {
            assert_eq!(l.call(row + 64 * col), (32 * (row / 16) + row % 16) * lane + col);
        }
        // A 16-bit accumulator: one element per 32-bit column (every other half).
        let l = tmem_frg_1sm_layout(128, 16, &[1, 1], 16);
        assert_eq!(l.call(3 + 128 * 5), 3 * 2 * lane + 2 * 5);
    }
}

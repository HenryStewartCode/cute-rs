// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/mma_traits_sm90_gmma.hpp (the layout atoms, and the
//   canonical-layout checks and descriptor fields of make_gmma_desc) and
//   include/cute/arch/mma_sm90_desc.hpp (GmmaDescriptor's bit fields),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! CuTe DSL's `cute.nvgpu.warpgroup` (SM90 WGMMA): its shared-memory
//! layout atoms ([`SmemLayoutAtomKind`], [`make_smem_layout_atom`]), the
//! deprecated alias of [`OperandMajorMode`], and the F16/BF16 warpgroup MMA ([`MmaF16BF16Op`]) with its
//! [`fence`], [`commit_group`] and [`wait_group`] protocol, and the FP8 and 8-bit integer MMAs ([`MmaF8Op`],
//! [`MmaI8Op`]) and the register-sourced A ([`OperandSource::RMEM`]).
//!
//! The atoms are C++ CuTe's `GMMA::Layout_{MN,K}_{INTER,SW32,SW64,SW128}_Atom`
//! (`include/cute/atom/mma_traits_sm90_gmma.hpp`): an 8-row block of 128,
//! 256, 512 or 1024 contiguous bits, under `Swizzle<0,4,3>` (none),
//! `Swizzle<1,4,3>`, `Swizzle<2,4,3>` or `Swizzle<3,4,3>` on byte
//! addresses.

pub use super::common::OperandMajorMode;
use crate::ir::DType;
use crate::layout::{ComposedLayout, Layout, Swizzle, make_composed_layout};
use crate::value::Numeric;

/// The kinds of SM90 shared-memory layout atom (CuTe DSL's
/// `warpgroup.SmemLayoutAtomKind`): the smallest compact layouts from which
/// TMA- and WGMMA-legal operand layouts are tiled. `MN_*` atoms are
/// contiguous along M (A) or N (B), `K_*` along K; `INTER` is unswizzled
/// (16-byte interleaved), `SW32`, `SW64` and `SW128` are swizzled over rows
/// of 32, 64 and 128 bytes.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SmemLayoutAtomKind {
    MN_INTER,
    MN_SW32,
    MN_SW64,
    MN_SW128,
    K_INTER,
    K_SW32,
    K_SW64,
    K_SW128,
}

impl SmemLayoutAtomKind {
    pub const ALL: [SmemLayoutAtomKind; 8] = [
        SmemLayoutAtomKind::MN_INTER,
        SmemLayoutAtomKind::MN_SW32,
        SmemLayoutAtomKind::MN_SW64,
        SmemLayoutAtomKind::MN_SW128,
        SmemLayoutAtomKind::K_INTER,
        SmemLayoutAtomKind::K_SW32,
        SmemLayoutAtomKind::K_SW64,
        SmemLayoutAtomKind::K_SW128,
    ];

    /// The major mode the atom is for.
    pub fn major_mode(self) -> OperandMajorMode {
        use SmemLayoutAtomKind::*;
        match self {
            MN_INTER | MN_SW32 | MN_SW64 | MN_SW128 => OperandMajorMode::MN,
            K_INTER | K_SW32 | K_SW64 | K_SW128 => OperandMajorMode::K,
        }
    }

    /// Contiguous bits along the major mode: 128, 256, 512 or 1024.
    pub fn contiguous_bits(self) -> usize {
        128 << self.swizzle_bits()
    }

    fn swizzle_bits(self) -> u32 {
        use SmemLayoutAtomKind::*;
        match self {
            MN_INTER | K_INTER => 0,
            MN_SW32 | K_SW32 => 1,
            MN_SW64 | K_SW64 => 2,
            MN_SW128 | K_SW128 => 3,
        }
    }

    /// The atom's swizzle on byte addresses, as CuTe writes it:
    /// `Swizzle<B,4,3>` with `B` 0 (INTER), 1, 2 or 3 (SW128).
    pub fn byte_swizzle(self) -> Swizzle {
        Swizzle { bits: self.swizzle_bits(), base: 4, shift: 3 }
    }
}

/// The SM90 shared-memory layout atom of `kind` for elements of `T` (CuTe
/// DSL's `warpgroup.make_smem_layout_atom(kind, element_type)`), a composed
/// layout in units of elements. See [`make_smem_layout_atom_dtype`].
#[track_caller]
pub fn make_smem_layout_atom<T: Numeric>(kind: SmemLayoutAtomKind) -> ComposedLayout {
    make_smem_layout_atom_dtype(kind, T::DTYPE)
}

/// [`make_smem_layout_atom`] for an element type given as a [`DType`]. With
/// `c` the atom's contiguous bits over the element width, an `MN_*` atom is
/// `sw o 0 o (c,8):(1,c)` and a `K_*` atom `sw o 0 o (8,c):(c,1)`.
///
/// cute-rs's composed layouts swizzle element offsets, where CuTe's atoms
/// swizzle byte addresses, so `sw` is the byte swizzle
/// ([`SmemLayoutAtomKind::byte_swizzle`]) restated for the element width
/// ([`Swizzle::to_elements`]): `Swizzle<B,4-k,3>` for elements of `2^k`
/// bytes (`Swizzle<3,3,3>` for a 16-bit SW128 atom) and `Swizzle<B,5,3>`
/// for 4-bit elements. Panics for an element width that is not a power of
/// two (the 6-bit floats), wider than 128 bits, or a Boolean (1 bit, which
/// the atoms do not hold).
#[track_caller]
pub fn make_smem_layout_atom_dtype(kind: SmemLayoutAtomKind, element_type: DType) -> ComposedLayout {
    let bits = element_type.bits();
    assert!(
        bits.is_power_of_two() && (4..=128).contains(&bits),
        "make_smem_layout_atom: {} is {bits} bits wide; an SM90 shared-memory layout atom takes elements of 4, 8, 16, \
         32, 64 or 128 bits",
        element_type.cute_name()
    );
    let c = kind.contiguous_bits() / bits;
    let sw = kind.byte_swizzle().to_elements(bits).unwrap_or_else(|why| panic!("make_smem_layout_atom: {why}"));
    let outer = match kind.major_mode() {
        OperandMajorMode::MN => Layout::new((c, 8), (1, c)),
        OperandMajorMode::K => Layout::new((8, c), (c, 1)),
    };
    make_composed_layout(sw, 0, outer)
}

// ---------------------------------------------------------------------------
// Warpgroup MMA.
//
// A WGMMA (`wgmma.mma_async`, sm_90a) multiplies a 64 x K tile of A by a
// K x N tile of B into 64 x N accumulators spread over the 128 threads of a
// warpgroup. A and B are read from shared memory through 64-bit matrix
// descriptors (GmmaDescriptor): the operand's start address, two byte
// strides (the leading and stride byte offsets) and its swizzle, for one of
// the PTX ISA's canonical K-major or MN-major layouts. The instruction is
// asynchronous: `fence` orders earlier register writes before it,
// `commit_group` closes a group of them and `wait_group(n)` waits until at
// most `n` groups are in flight, after which the accumulators hold the
// results.
//
// In cute-rs the accumulators are SSA values: a WGMMA's results are only
// valid after a wait covers it, so `wait_group` rebinds every accumulator
// fragment of the current region to the outputs of a `WgmmaWait`, which
// code after it reads, and the validator refuses any other read of a
// result before then.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::panic::Location;
use std::rc::{Rc, Weak};

use crate::atom::mma::{MmaAtom, MmaOp};
use crate::builder;
use crate::ir::{Op, ValueId, WgmmaKind};
use crate::tensor::{Engine, RegisterStorage, Tensor, TensorInner};

/// Where the A operand of a warpgroup MMA comes from (CuTe DSL's
/// `warpgroup.OperandSource`): shared memory through a descriptor
/// ([`OperandSource::SMEM`]), or registers ([`OperandSource::RMEM`], the
/// PTX ISA's register form: A is then a register fragment laid out like
/// four stacked `mma.sync` A fragments, one 16-row slab per warp, which
/// [`TiledMma::make_fragment_a`](crate::TiledMma::make_fragment_a) makes
/// and a kernel fills, by `ldmatrix` or from an earlier accumulator, before
/// [`fence`]).
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperandSource {
    RMEM,
    SMEM,
}

/// An instruction shape for [`MmaF16BF16Op::new`]: `(M, N)` or `(M, N, K)`.
pub trait InstructionShape {
    #[doc(hidden)]
    fn mnk(self) -> (usize, usize, Option<usize>);
}

impl InstructionShape for (usize, usize) {
    fn mnk(self) -> (usize, usize, Option<usize>) {
        (self.0, self.1, None)
    }
}

impl InstructionShape for (usize, usize, usize) {
    fn mnk(self) -> (usize, usize, Option<usize>) {
        (self.0, self.1, Some(self.2))
    }
}

/// CuTe DSL's `cute.nvgpu.warpgroup.MmaF16BF16Op(ab_dtype, acc_dtype,
/// instruction_shape, a_src, a_major_mode, b_major_mode)`: the sm_90a
/// warpgroup MMA `wgmma.mma_async.sync.aligned.m64nNk16` on Float16 or
/// BFloat16 operands.
///
/// | A/B  | accumulator | shape           |
/// |------|-------------|-----------------|
/// | F16  | F16, F32    | (64, N, 16)     |
/// | BF16 | F32         | (64, N, 16)     |
///
/// N is a multiple of 8 from 8 to 256 (K may be left out: it is 16). A and
/// B may each be K-major or MN-major (transposed). The atom spans a
/// warpgroup of 128 threads, each holding `N / 2` accumulators; its
/// operands are descriptor fragments ([`TiledMma::make_fragment_a`](crate::TiledMma::make_fragment_a)),
/// and it takes the runtime field [`MmaField::ACCUMULATE`](crate::MmaField::ACCUMULATE)
/// (`false` until set: the first MMA of a tile overwrites the
/// accumulators). Kernels using it compile for sm_90a only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF16BF16Op {
    ab_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
    a_src: OperandSource,
    a_major_mode: OperandMajorMode,
    b_major_mode: OperandMajorMode,
}

impl MmaF16BF16Op {
    /// The operation, or a panic naming the parameter that no instruction
    /// takes (as CuTe DSL raises at construction).
    #[track_caller]
    pub fn new(
        ab_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaF16BF16Op {
        let (m, n, k) = instruction_shape.mnk();
        assert!(
            matches!(ab_dtype, DType::F16 | DType::BF16),
            "cute: warpgroup.MmaF16BF16Op: 'ab_dtype' is {ab_dtype}; its A and B operands are Float16 or BFloat16"
        );
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: warpgroup.MmaF16BF16Op: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        assert!(
            ab_dtype != DType::BF16 || acc_dtype == DType::F32,
            "cute: warpgroup.MmaF16BF16Op: BFloat16 operands ('ab_dtype') accumulate in Float32 only, and \
             'acc_dtype' is {acc_dtype}"
        );
        assert!(m == 64, "cute: warpgroup.MmaF16BF16Op: the instruction shape's M is {m}; a warpgroup MMA has M = 64");
        assert!(
            (8..=256).contains(&n) && n % 8 == 0,
            "cute: warpgroup.MmaF16BF16Op: the instruction shape's N is {n}; N must be a multiple of 8 from 8 to 256"
        );
        let k = k.unwrap_or(16);
        assert!(
            k == 16,
            "cute: warpgroup.MmaF16BF16Op: the instruction shape's K is {k}; Float16 and BFloat16 MMAs have K = 16"
        );
        check_register_a("MmaF16BF16Op", a_src, a_major_mode);
        MmaF16BF16Op { ab_dtype, acc_dtype, shape_mnk: (m, n, k), a_src, a_major_mode, b_major_mode }
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
    pub fn kind(&self) -> WgmmaKind {
        WgmmaKind {
            n: self.shape_mnk.1 as u32,
            a: self.ab_dtype,
            b: self.ab_dtype,
            acc: self.acc_dtype,
            trans_a: self.a_major_mode == OperandMajorMode::MN,
            trans_b: self.b_major_mode == OperandMajorMode::MN,
            a_regs: self.a_src == OperandSource::RMEM,
        }
    }
}

/// A from registers is K-major: PTX's register form has no `imm-trans-a`.
#[track_caller]
fn check_register_a(op: &str, a_src: OperandSource, a_major_mode: OperandMajorMode) {
    assert!(
        a_src == OperandSource::SMEM || a_major_mode == OperandMajorMode::K,
        "cute: warpgroup.{op}: 'a_src' is OperandSource::RMEM (A in registers) with 'a_major_mode' MN: an A operand \
         in registers is K-major (the register form of wgmma.mma_async has no A transpose)"
    );
}

/// Checks shared by the 8-bit warpgroup MMAs: M = 64, N a multiple of 8
/// from 8 to 256, K = 32 (or left out), both operands K-major. Returns
/// `(M, N, K)`.
#[track_caller]
fn check_8bit_shape(
    op: &str,
    instruction_shape: (usize, usize, Option<usize>),
    a_major_mode: OperandMajorMode,
    b_major_mode: OperandMajorMode,
) -> (usize, usize, usize) {
    let (m, n, k) = instruction_shape;
    assert!(m == 64, "cute: warpgroup.{op}: the instruction shape's M is {m}; a warpgroup MMA has M = 64");
    assert!(
        (8..=256).contains(&n) && n % 8 == 0,
        "cute: warpgroup.{op}: the instruction shape's N is {n}; 8-bit warpgroup MMAs take N of 8, 16, ..., 256 (steps \
         of 8)"
    );
    let k = k.unwrap_or(32);
    assert!(k == 32, "cute: warpgroup.{op}: the instruction shape's K is {k}; 8-bit MMAs have K = 32");
    for (which, mode) in [("a_major_mode", a_major_mode), ("b_major_mode", b_major_mode)] {
        assert!(
            mode == OperandMajorMode::K,
            "cute: warpgroup.{op}: '{which}' is MN; 8-bit warpgroup MMAs take K-major operands only (PTX has no \
             transpose for them)"
        );
    }
    (m, n, k)
}

/// CuTe DSL's `cute.nvgpu.warpgroup.MmaF8Op(a_dtype, b_dtype, acc_dtype,
/// instruction_shape, a_src, a_major_mode, b_major_mode)`: the sm_90a
/// warpgroup MMA `wgmma.mma_async.sync.aligned.m64nNk32` on FP8 operands.
///
/// | A           | B           | accumulator | shape       |
/// |-------------|-------------|-------------|-------------|
/// | E4M3, E5M2  | E4M3, E5M2  | F16, F32    | (64, N, 32) |
///
/// A and B are each [`Float8E4M3FN`](crate::Float8E4M3FN) or
/// [`Float8E5M2`](crate::Float8E5M2) (they may differ); N is a multiple of
/// 8 from 8 to 256; both operands are K-major. Otherwise it is used as
/// [`MmaF16BF16Op`] is.
///
/// Hopper accumulates FP8 products with reduced internal precision (the
/// reason for CUTLASS's two-level accumulation in its FP8 GEMMs); the
/// simulator instead adds each MMA's 32 exact products to C in Float32 in
/// order of K and rounds once, so results agree with hardware only within a
/// tolerance unless the sums are exact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaF8Op {
    a_dtype: DType,
    b_dtype: DType,
    acc_dtype: DType,
    shape_mnk: (usize, usize, usize),
    a_src: OperandSource,
}

impl MmaF8Op {
    /// The operation, or a panic naming the parameter that no instruction
    /// takes.
    #[track_caller]
    pub fn new(
        a_dtype: DType,
        b_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaF8Op {
        for (which, t) in [("a_dtype", a_dtype), ("b_dtype", b_dtype)] {
            assert!(
                matches!(t, DType::F8E4M3FN | DType::F8E5M2),
                "cute: warpgroup.MmaF8Op: '{which}' is {t}; its operands are Float8E4M3FN or Float8E5M2"
            );
        }
        assert!(
            matches!(acc_dtype, DType::F16 | DType::F32),
            "cute: warpgroup.MmaF8Op: 'acc_dtype' is {acc_dtype}; it accumulates in Float16 or Float32"
        );
        let shape_mnk = check_8bit_shape("MmaF8Op", instruction_shape.mnk(), a_major_mode, b_major_mode);
        MmaF8Op { a_dtype, b_dtype, acc_dtype, shape_mnk, a_src }
    }

    pub fn a_dtype(&self) -> DType {
        self.a_dtype
    }

    pub fn b_dtype(&self) -> DType {
        self.b_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        self.acc_dtype
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    pub fn a_src(&self) -> OperandSource {
        self.a_src
    }

    /// The instruction it issues.
    pub fn kind(&self) -> WgmmaKind {
        WgmmaKind {
            n: self.shape_mnk.1 as u32,
            a: self.a_dtype,
            b: self.b_dtype,
            acc: self.acc_dtype,
            trans_a: false,
            trans_b: false,
            a_regs: self.a_src == OperandSource::RMEM,
        }
    }
}

/// CuTe DSL's `cute.nvgpu.warpgroup.MmaI8Op(a_dtype, b_dtype, acc_dtype,
/// instruction_shape, a_src, a_major_mode, b_major_mode)`: the sm_90a
/// warpgroup MMA `wgmma.mma_async.sync.aligned.m64nNk32.s32` on 8-bit
/// integers.
///
/// | A        | B        | accumulator | shape       |
/// |----------|----------|-------------|-------------|
/// | S8, U8   | S8, U8   | S32         | (64, N, 32) |
///
/// A and B are each [`Int8`](crate::Int8) or [`Uint8`](crate::Uint8) (the
/// signedness may differ: the mixed forms need PTX ISA 8.4); N is 8, 24 or
/// a multiple of 16 up to 256; both operands are K-major. The sums are
/// exact and wrap modulo 2^32 (PTX's `.satfinite`, which CuTe DSL does not
/// expose, is not used).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MmaI8Op {
    a_dtype: DType,
    b_dtype: DType,
    shape_mnk: (usize, usize, usize),
    a_src: OperandSource,
}

impl MmaI8Op {
    /// The operation, or a panic naming the parameter that no instruction
    /// takes.
    #[track_caller]
    pub fn new(
        a_dtype: DType,
        b_dtype: DType,
        acc_dtype: DType,
        instruction_shape: impl InstructionShape,
        a_src: OperandSource,
        a_major_mode: OperandMajorMode,
        b_major_mode: OperandMajorMode,
    ) -> MmaI8Op {
        for (which, t) in [("a_dtype", a_dtype), ("b_dtype", b_dtype)] {
            assert!(
                matches!(t, DType::I8 | DType::U8),
                "cute: warpgroup.MmaI8Op: '{which}' is {t}; its operands are Int8 or Uint8"
            );
        }
        assert!(
            acc_dtype == DType::I32,
            "cute: warpgroup.MmaI8Op: 'acc_dtype' is {acc_dtype}; it accumulates in Int32"
        );
        let shape_mnk = check_8bit_shape("MmaI8Op", instruction_shape.mnk(), a_major_mode, b_major_mode);
        let n = shape_mnk.1;
        assert!(
            n == 8 || n == 24 || n.is_multiple_of(16),
            "cute: warpgroup.MmaI8Op: the instruction shape's N is {n}; an 8-bit integer MMA has N of 8, 24 or a \
             multiple of 16 (32 to 256)"
        );
        MmaI8Op { a_dtype, b_dtype, shape_mnk, a_src }
    }

    pub fn a_dtype(&self) -> DType {
        self.a_dtype
    }

    pub fn b_dtype(&self) -> DType {
        self.b_dtype
    }

    pub fn acc_dtype(&self) -> DType {
        DType::I32
    }

    pub fn shape_mnk(&self) -> (usize, usize, usize) {
        self.shape_mnk
    }

    pub fn a_src(&self) -> OperandSource {
        self.a_src
    }

    /// The instruction it issues.
    pub fn kind(&self) -> WgmmaKind {
        WgmmaKind {
            n: self.shape_mnk.1 as u32,
            a: self.a_dtype,
            b: self.b_dtype,
            acc: DType::I32,
            trans_a: false,
            trans_b: false,
            a_regs: self.a_src == OperandSource::RMEM,
        }
    }
}

macro_rules! sm90_op_into {
    ($($t:ty),*) => {$(
        impl From<$t> for MmaOp {
            fn from(op: $t) -> MmaOp {
                MmaOp::Sm90(op.kind())
            }
        }

        impl From<$t> for MmaAtom {
            fn from(op: $t) -> MmaAtom {
                MmaAtom::new(op.into())
            }
        }
    )*};
}

sm90_op_into!(MmaF8Op, MmaI8Op);

impl From<MmaF16BF16Op> for MmaOp {
    fn from(op: MmaF16BF16Op) -> MmaOp {
        MmaOp::Sm90(op.kind())
    }
}

impl From<MmaF16BF16Op> for MmaAtom {
    fn from(op: MmaF16BF16Op) -> MmaAtom {
        MmaAtom::new(op.into())
    }
}

/// `wgmma.fence.sync.aligned` (CuTe DSL's `warpgroup.fence()`): orders the
/// warpgroup's earlier register accesses (to accumulators) before the
/// warpgroup MMAs after it. The PTX ISA requires one before the first MMA
/// of a warpgroup and between a write of an accumulator (a fill, say) and
/// an MMA that uses it. Every thread of the warpgroup executes it.
///
/// The accumulator fragments of the current region (those an MMA has used,
/// or [`TiledMma::make_fragment_c`](crate::TiledMma::make_fragment_c) and
/// [`ThrMma::partition_fragment_c`](crate::ThrMma::partition_fragment_c)
/// made for a warpgroup MMA) are passed through an operand fence first, so
/// that the compiler materialises their values before the fence; registers
/// not yet written are set to zero there (an MMA that does not accumulate
/// ignores them, but must not find them written after the fence). A
/// fragment made otherwise ([`make_fragment`](crate::make_fragment)) joins
/// at its first [`gemm`](crate::gemm): fill it before the fence.
#[track_caller]
pub fn fence() {
    let loc = Location::caller();
    let mut slots = live_accumulators(true, loc);
    slots.extend(live_registers(Marked::A, false, loc));
    let inputs: Vec<ValueId> = slots.iter().map(|&(_, _, v)| v).collect();
    let results = builder::with(|b| {
        let results: Vec<ValueId> = inputs.iter().map(|&v| b.new_value(b.dtype(v))).collect();
        b.push(Op::WgmmaFence { results: results.clone(), inputs: inputs.clone() }, loc);
        results
    });
    rebind(&slots, &results, loc);
    with_groups(|g| g.rename(&inputs, &results));
}

/// `wgmma.commit_group.sync.aligned` (CuTe DSL's
/// `warpgroup.commit_group()`): the warpgroup MMAs issued since the last
/// commit become a group, which [`wait_group`] waits for.
#[track_caller]
pub fn commit_group() {
    let loc = Location::caller();
    builder::with(|b| b.push(Op::WgmmaCommitGroup, loc));
    with_groups(|g| {
        let group = std::mem::take(&mut g.uncommitted);
        g.groups.push_back(group);
        let group_a = std::mem::take(&mut g.uncommitted_a);
        g.groups_a.push_back(group_a);
    });
}

/// `wgmma.wait_group.sync.aligned n` (CuTe DSL's
/// `warpgroup.wait_group(n)`): waits until at most `n` of this warp's
/// committed groups of warpgroup MMAs are in flight. The accumulators of
/// the groups it completes may then be read; with `n > 0` the newest `n`
/// groups may still be writing theirs (the pipelined main loop of a dense
/// GEMM waits for all but the last few). The accumulator fragments of the
/// current region are rebound to the values after the wait (see [`fence`]).
#[track_caller]
pub fn wait_group(n: u32) {
    let loc = Location::caller();
    let slots = live_accumulators(false, loc);
    let mut inputs: Vec<ValueId> = slots.iter().map(|&(_, _, v)| v).collect();
    let path = builder::with(|b| b.region_path());
    let (mut complete, a_values) = with_groups(|g| g.wait(n, &inputs, &path));
    // The register A operands of the MMAs the wait completes go through its
    // operand fence too, so that their registers stay live (and unchanged)
    // until the tensor cores have read them.
    for v in a_values {
        if !inputs.contains(&v) {
            inputs.push(v);
            complete.push(true);
        }
    }
    let results = builder::with(|b| {
        let results: Vec<ValueId> = inputs.iter().map(|&v| b.new_value(b.dtype(v))).collect();
        b.push(Op::WgmmaWait { pending: n, results: results.clone(), inputs: inputs.clone(), complete }, loc);
        results
    });
    rebind(&slots, &results[..slots.len()], loc);
    with_groups(|g| g.rename(&inputs[..slots.len()], &results[..slots.len()]));
}

/// What the builder knows about the warpgroup MMAs of the kernel being
/// built: its accumulator and register-A fragments, and which results (and
/// register A operands) are in which committed group (in the order the
/// kernel is built, which a loop body is once).
#[derive(Default)]
struct Groups {
    kernel: u32,
    accumulators: Vec<(Weak<RegisterStorage>, crate::ir::DType)>,
    a_fragments: Vec<(Weak<RegisterStorage>, crate::ir::DType)>,
    uncommitted: Vec<ValueId>,
    groups: VecDeque<Vec<ValueId>>,
    /// The register A operands of the uncommitted MMAs and of each group,
    /// with the region path where the MMA was issued.
    uncommitted_a: Vec<(ValueId, Vec<u64>)>,
    groups_a: VecDeque<Vec<(ValueId, Vec<u64>)>>,
}

impl Groups {
    /// `wait_group(n)` on `inputs` at region path `path`: whether each is
    /// known complete, and the register A operands of the groups it
    /// completes that are visible at `path`. The groups it completes are
    /// forgotten.
    fn wait(&mut self, n: u32, inputs: &[ValueId], path: &[u64]) -> (Vec<bool>, Vec<ValueId>) {
        let done = self.groups.len().saturating_sub(n as usize);
        let completed: Vec<ValueId> = self.groups.drain(..done).flatten().collect();
        let done_a = self.groups_a.len().saturating_sub(n as usize);
        let mut a_values: Vec<ValueId> = Vec::new();
        for (v, at) in self.groups_a.drain(..done_a).flatten() {
            if path.starts_with(&at) && !a_values.contains(&v) {
                a_values.push(v);
            }
        }
        let in_flight = |v: &ValueId| self.uncommitted.contains(v) || self.groups.iter().any(|g| g.contains(v));
        let complete = inputs.iter().map(|v| !in_flight(v) && (n == 0 || completed.contains(v))).collect();
        (complete, a_values)
    }

    /// Results that take the place of inputs (operand fences and waits).
    fn rename(&mut self, inputs: &[ValueId], results: &[ValueId]) {
        for list in std::iter::once(&mut self.uncommitted).chain(self.groups.iter_mut()) {
            for v in list.iter_mut() {
                if let Some(i) = inputs.iter().position(|x| x == v) {
                    *v = results[i];
                }
            }
        }
    }
}
thread_local! {
    static GROUPS: RefCell<Groups> = RefCell::new(Groups::default());
}

/// Run `f` on the state of the kernel being built (a new kernel starts
/// afresh).
fn with_groups<R>(f: impl FnOnce(&mut Groups) -> R) -> R {
    let kernel = builder::with(|b| b.kernel);
    GROUPS.with(|g| {
        let mut g = g.borrow_mut();
        if g.kernel != kernel {
            *g = Groups { kernel, ..Groups::default() };
        }
        f(&mut g)
    })
}

/// Flag `storage`, of `dtype` registers, as holding warpgroup-MMA
/// accumulators.
pub(crate) fn mark_accumulators(storage: &Rc<RegisterStorage>, dtype: crate::ir::DType) {
    if storage.wgmma_acc.replace(true) {
        return;
    }
    with_groups(|g| g.accumulators.push((Rc::downgrade(storage), dtype)));
}

/// Flag `storage`, of `dtype` registers, as holding register-sourced A
/// operands of warpgroup MMAs.
pub(crate) fn mark_a_fragment(storage: &Rc<RegisterStorage>, dtype: crate::ir::DType) {
    if storage.wgmma_a.replace(true) {
        return;
    }
    with_groups(|g| g.a_fragments.push((Rc::downgrade(storage), dtype)));
}

/// Results of a warpgroup MMA just issued, uncommitted, and its register A
/// operands (none for a shared-memory A).
pub(crate) fn issued(results: &[ValueId], a_registers: &[ValueId]) {
    let path = builder::with(|b| b.region_path());
    with_groups(|g| {
        g.uncommitted.extend_from_slice(results);
        g.uncommitted_a.extend(a_registers.iter().map(|&v| (v, path.clone())));
    });
}

type Slot = (Rc<RegisterStorage>, usize, ValueId);

/// Which fragments [`live_registers`] looks at.
#[derive(Clone, Copy)]
enum Marked {
    Accumulators,
    A,
}

/// The written registers of the accumulator fragments of the current
/// region, and those of fragments from outside an open `if` or loop that
/// the region wrote; with `zero_fill`, the unwritten registers of the
/// current region's fragments are first set to zero.
#[track_caller]
fn live_accumulators(zero_fill: bool, loc: crate::ir::Loc) -> Vec<Slot> {
    live_registers(Marked::Accumulators, zero_fill, loc)
}

/// [`live_accumulators`] for the accumulator or register-A fragments.
#[track_caller]
fn live_registers(which: Marked, zero_fill: bool, loc: crate::ir::Loc) -> Vec<Slot> {
    let storages: Vec<(Rc<RegisterStorage>, crate::ir::DType)> = with_groups(|g| {
        let list = match which {
            Marked::Accumulators => &mut g.accumulators,
            Marked::A => &mut g.a_fragments,
        };
        list.retain(|(w, _)| w.strong_count() > 0);
        list.iter().filter_map(|(w, t)| w.upgrade().map(|s| (s, *t))).collect()
    });
    let mut out = Vec::new();
    for (s, dtype) in storages {
        // A fragment from outside an open `if` or loop: only the registers
        // the region itself wrote (an MMA in it, say) are rebound, since a
        // value defined in the region must not leak out of it.
        let outer = builder::with(|b| b.region_opened_since(s.created));
        for i in 0..s.len() {
            match s.peek(i) {
                Some(v) if outer && !builder::with(|b| b.defined_since_region_opened(s.created, v)) => {}
                Some(v) => out.push((s.clone(), i, v)),
                None if zero_fill && !outer => {
                    let zero = builder::with(|b| b.constant(crate::ir::Literal::zero(dtype), loc));
                    s.write(i, zero);
                    out.push((s.clone(), i, zero));
                }
                None => {}
            }
        }
    }
    out
}

#[track_caller]
fn rebind(slots: &[Slot], results: &[ValueId], _loc: crate::ir::Loc) {
    for ((s, i, _), &r) in slots.iter().zip(results) {
        s.write(*i, r);
    }
}

/// The GMMA layout types (`GmmaDescriptor::layout_type`, bits 62-63): no
/// swizzle (interleaved), and the 128-, 64- and 32-byte swizzles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GmmaLayoutType {
    Interleave = 0,
    B128 = 1,
    B64 = 2,
    B32 = 3,
}

impl GmmaLayoutType {
    /// 16-byte units in a swizzle row: 1, 2, 4 or 8.
    fn width(self) -> usize {
        match self {
            GmmaLayoutType::Interleave => 1,
            GmmaLayoutType::B32 => 2,
            GmmaLayoutType::B64 => 4,
            GmmaLayoutType::B128 => 8,
        }
    }
}

/// The fields of a GMMA descriptor fixed when the kernel is built (all but
/// the start address): CuTe's `make_gmma_desc` for one MMA tile of
/// `mn x K` elements (K of 32 bytes: 16 or 32 elements) whose layout (`(mn, k) ->` element offset, a column-major
/// index into the tile) is `tile`, `major` the operand's major mode,
/// `swizzle_bits` the B of the tensor's `Swizzle<B,4,3>` on bytes, `bits`
/// the element width. Returns the high bits of the descriptor (leading and
/// stride byte offsets and layout type; the start address is added at
/// run time) or why the layout is not a canonical GMMA layout.
pub(crate) fn gmma_desc_fields(
    tile: &crate::layout::Layout,
    mn: usize,
    major: OperandMajorMode,
    swizzle_bits: u32,
    bits: usize,
) -> Result<u64, String> {
    let ty = match swizzle_bits {
        0 => GmmaLayoutType::Interleave,
        1 => GmmaLayoutType::B32,
        2 => GmmaLayoutType::B64,
        3 => GmmaLayoutType::B128,
        b => return Err(format!("a {b}-bit swizzle, which no GMMA layout has")),
    };
    // K covers 32 bytes: 16 elements of 16 bits, 32 of 8.
    let k = 256 / bits;
    if tile.size() != mn * k {
        return Err(format!("the MMA tile {tile} has {} elements, not {mn} x {k}", tile.size()));
    }
    // Elements per 16-byte unit, and per swizzle row.
    let t = 128 / bits;
    let w = ty.width();
    let f = |m: usize, kk: usize| tile.call(m + mn * kk);
    // The strides that define the canonical layout, in elements, then a
    // check of every element against it.
    type Canonical = Box<dyn Fn(usize, usize) -> usize>;
    let (lbo, sbo, canonical): (usize, usize, Canonical) = match major {
        // ((8, m), (T, 2)) : ((W T, SBO), (1, LBO or T)).
        OperandMajorMode::K => {
            let sbo = if mn > 8 { f(8, 0) } else { 0 };
            let lbo = if ty == GmmaLayoutType::Interleave { f(0, t) } else { t };
            let canon = move |m: usize, kk: usize| (m / 8) * sbo + (m % 8) * w * t + (kk / t) * lbo + kk % t;
            (lbo, sbo, Box::new(canon))
        }
        // Interleaved: ((T, 1, m), (8, k)) : ((1, T, SBO), (T, LBO)).
        // Swizzled: ((T, W, m), (8, k)) : ((1, T, LBO), (W T, SBO)).
        OperandMajorMode::MN => {
            let block = w * t;
            let mn_step = if mn > block { f(block, 0) } else { 0 };
            let k_step = f(0, 8);
            let canon =
                move |m: usize, kk: usize| (m / block) * mn_step + m % block + (kk / 8) * k_step + (kk % 8) * block;
            let (lbo, sbo) = if ty == GmmaLayoutType::Interleave { (k_step, mn_step) } else { (mn_step, k_step) };
            (lbo, sbo, Box::new(canon))
        }
    };
    if !mn.is_multiple_of(8) || (major == OperandMajorMode::MN && !mn.is_multiple_of(w * t)) {
        return Err(format!(
            "its {mn} {} rows do not fill whole {}-element blocks of the layout",
            if major == OperandMajorMode::K { "M/N" } else { "MN-major" },
            if major == OperandMajorMode::K { 8 } else { w * t }
        ));
    }
    for kk in 0..k {
        for m in 0..mn {
            let (got, want) = (f(m, kk), canonical(m, kk));
            if got != want {
                return Err(format!(
                    "element ({m}, {kk}) is at offset {got}, where the canonical {major:?}-major layout with {} puts it \
                     at {want}",
                    match ty {
                        GmmaLayoutType::Interleave => "no swizzle".to_string(),
                        _ => format!("the {}-byte swizzle", 16 * w),
                    }
                ));
            }
        }
    }
    let bytes = |elems: usize| -> Result<u32, String> {
        u32::try_from(elems * bits / 8).map_err(|_| format!("a byte offset of {} elements", elems))
    };
    // The fields are encoded by the public
    // builder (`prims::WgmmaSmemDesc`; base offset 0). A swizzled K-major
    // layout has no leading offset, which CuTe sets to 1 unit (16 bytes).
    let lbo = if major == OperandMajorMode::K && ty != GmmaLayoutType::Interleave { 16 } else { bytes(lbo)? };
    let desc = crate::prims::WgmmaSmemDesc::build(0u32, lbo, bytes(sbo)?, 0, ty.into())?;
    Ok(desc.bits().as_static().expect("a descriptor of constants"))
}

/// The descriptor fragment of a WGMMA operand (CuTe DSL's
/// `make_fragment_A/B` of a warpgroup MMA): `input` is this thread's
/// partition of the shared-memory operand, `((MN, K), MMA_MN, MMA_K,
/// ...)`. The result has the same shape with the first mode `1`, and each
/// element is the descriptor of one MMA tile.
#[track_caller]
pub(crate) fn make_desc_fragment<T: crate::value::Numeric>(
    input: &Tensor<T>,
    mn: usize,
    major: OperandMajorMode,
    what: &str,
) -> Tensor<T> {
    let fail = |why: String| -> ! { panic!("cute: {what}: {why}") };
    let Engine::Memory { buffer, space: crate::ir::MemSpace::Shared, peer: None } = input.engine else {
        fail(format!(
            "a warpgroup MMA reads this operand from shared memory through a descriptor, but the tensor {:?} is not \
             in shared memory",
            input
        ))
    };
    let bits = T::DTYPE.bits();
    let swizzle_bits = match &input.inner {
        TensorInner::Identity => 0,
        TensorInner::Swizzle { bytes: Some(sw), .. } if sw.base == 4 && sw.shift == 3 && sw.bits <= 3 => sw.bits,
        TensorInner::Swizzle { elem, .. } => fail(format!(
            "the operand is swizzled by {elem} (on elements), which is no GMMA swizzle: build it from a \
             warpgroup::make_smem_layout_atom atom (Swizzle<B,4,3> on bytes)"
        )),
        TensorInner::Layout { .. } => fail("the operand is over a composed layout with a layout inner".into()),
    };
    let layout = input.layout();
    let tile = layout.mode(0);
    let fields = gmma_desc_fields(&tile, mn, major, swizzle_bits, bits)
        .unwrap_or_else(|why| fail(format!("the operand's MMA tile {tile} is not a canonical GMMA layout: {why}")));
    // The swizzle acts on the absolute shared-memory address, so the buffer
    // must start on a repeat of the pattern (8 rows of 16, 32, 64 or 128
    // bytes).
    let repeat = 128usize << swizzle_bits;
    let align = builder::with(|b| b.buffer(buffer).align);
    if swizzle_bits > 0 && align < repeat {
        fail(format!(
            "its shared-memory buffer is aligned to {align} bytes, but the {}-byte swizzle repeats every {repeat} \
             bytes, so the buffer must be aligned to {repeat} (make_smem_tensor_aligned(.., {repeat}))",
            16 << swizzle_bits
        ));
    }
    // The other modes, in 16-byte units.
    let mut modes = vec![crate::layout::Layout::new(1, 0)];
    for i in 1..layout.rank() {
        let m = layout.mode(i);
        if m.shape().first_dynamic().is_some() {
            fail(format!("mode {i} of the operand ({m}) has a runtime extent"));
        }
        let stride = m.static_stride().map(&mut |&d| {
            let bytes = d * bits / 8;
            if bytes % 16 != 0 {
                fail(format!(
                    "mode {i} of the operand ({m}) has a stride of {d} elements, which does not start a 16-byte unit"
                ));
            }
            bytes / 16
        });
        modes.push(crate::layout::Layout::new(m.shape().clone(), stride));
    }
    let addr: crate::value::Value<u32> = input.ptr().toint_i32().to::<u32>();
    let desc = crate::prims::WgmmaSmemDesc::from_bits(fields).with_start_address(addr).value();
    Tensor::new(Engine::SmemDesc { desc: desc.checked() }, crate::layout::Layout::from_modes(modes))
}

impl<T> Tensor<T> {
    /// Whether this is a WGMMA operand fragment of matrix descriptors.
    pub(crate) fn engine_is_desc(&self) -> bool {
        matches!(self.engine, Engine::SmemDesc { .. })
    }

    /// The shared-memory matrix descriptor at `coord` of a WGMMA operand
    /// fragment ([`TiledMma::make_fragment_a`](crate::TiledMma::make_fragment_a)):
    /// the fragment's descriptor advanced by the coordinate's offset, in
    /// 16-byte units (CuTe's `DescriptorIterator`). Panics for other tensors.
    #[track_caller]
    pub fn smem_desc(&self, coord: impl Into<crate::layout::Coord>) -> crate::value::Value<u64> {
        let Engine::SmemDesc { desc } = self.engine else {
            panic!("cute: Tensor::smem_desc: {self:?} is not a WGMMA operand fragment of matrix descriptors")
        };
        let off = self.offset.clone() + self.layout.offset_checked(&coord.into(), "Tensor::smem_desc");
        let base = crate::value::Value::<u64>::from_id(desc);
        if off.as_static() == Some(0) {
            return base;
        }
        let loc = Location::caller();
        let off = builder::with(|b| {
            let v = b.materialize_at(&off, 64, loc);
            b.cast(v, crate::ir::DType::U64, loc)
        });
        base + crate::value::Value::<u64>::from_id(off)
    }
}

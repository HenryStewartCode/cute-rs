// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/atom/copy_traits_sm100.hpp (the SM100_TMEM_LOAD_* and
//   SM100_TMEM_STORE_* copy traits' ValID, source, destination and reference
//   layouts, and make_tmem_copy) and include/cute/arch/copy_sm100.hpp (the
//   instructions those traits wrap, and the .16x32bx2 half-split offsets),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! TMEM load and store copy atoms (roadmap part m10-tmem-ld-st): CuTe DSL's
//! `tcgen05.Ld16x64bOp`, `Ld16x128bOp`, `Ld16x256bOp`, `Ld16x32bx2Op`,
//! `Ld32x32bOp` and their `St..Op` counterparts, used through
//! [`make_copy_atom`](crate::make_copy_atom) and [`copy`](crate::copy), and
//! [`make_tmem_copy`], which tiles one over the four warps of a warpgroup.
//!
//! The atoms' layouts are CuTe C++'s `SM100_TMEM_LOAD_*` traits, computed
//! from the shape, the repetition and packing rather than tabulated: in
//! bits, `ValID` maps a bit of the atom (each lane's bits, then the lanes)
//! to its TMEM bit address (`lane * 2^21 + column * 32 + bit`), and the
//! register side gives each thread's 32-bit registers. They are converted
//! to the atom's element type as CuTe's `Copy_Atom` converts them.

use crate::atom::copy::{CopyAtom, CopyOp, TiledCopy, make_cotiled_copy};
use crate::ir::TmemShape;
use crate::layout::Layout;
use crate::tensor::{AddressSpace, Tensor};
use crate::value::Numeric;

/// The repetition of a TMEM load or store (CuTe DSL's
/// `tcgen05.Repetition`): `.x1` to `.x128`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Repetition {
    #[default]
    X1,
    X2,
    X4,
    X8,
    X16,
    X32,
    X64,
    X128,
}

impl Repetition {
    /// The repetition count.
    pub fn value(self) -> u32 {
        1 << (self as u32)
    }
}

/// Packing of a TMEM load (CuTe DSL's `tcgen05.Pack`): `Pack16bIn32b` loads
/// the low 16 bits of two adjacent columns into one 32-bit register
/// (`.pack::16b`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Pack {
    #[default]
    None,
    Pack16bIn32b,
}

/// Unpacking of a TMEM store (CuTe DSL's `tcgen05.Unpack`): `Unpack32bIn16b`
/// stores the halves of a 32-bit register to the low 16 bits of two
/// adjacent columns (`.unpack::16b`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Unpack {
    #[default]
    None,
    Unpack32bIn16b,
}

macro_rules! tmem_ops {
    ($($(#[$doc:meta])* $ld:ident, $st:ident, $shape:expr;)*) => {$(
        $(#[$doc])*
        ///
        /// The load (CuTe DSL's op of the same name): `repeat` and `pack`.
        /// Admissible on the sm_100f and sm_110f families (sm_100a, sm_110a).
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $ld {
            pub repeat: Repetition,
            pub pack: Pack,
        }

        impl $ld {
            /// The load of `repeat` repetitions, unpacked.
            pub fn new(repeat: Repetition) -> Self {
                $ld { repeat, pack: Pack::None }
            }

            /// The load with `pack`.
            pub fn with_pack(self, pack: Pack) -> Self {
                $ld { pack, ..self }
            }
        }

        impl From<$ld> for CopyOp {
            fn from(op: $ld) -> CopyOp {
                CopyOp::TmemLoad { shape: $shape, num: op.repeat.value(), pack: op.pack == Pack::Pack16bIn32b }
            }
        }

        $(#[$doc])*
        ///
        /// The store (CuTe DSL's op of the same name): `repeat` and `unpack`.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $st {
            pub repeat: Repetition,
            pub unpack: Unpack,
        }

        impl $st {
            /// The store of `repeat` repetitions, not unpacked.
            pub fn new(repeat: Repetition) -> Self {
                $st { repeat, unpack: Unpack::None }
            }

            /// The store with `unpack`.
            pub fn with_unpack(self, unpack: Unpack) -> Self {
                $st { unpack, ..self }
            }
        }

        impl From<$st> for CopyOp {
            fn from(op: $st) -> CopyOp {
                CopyOp::TmemStore {
                    shape: $shape,
                    num: op.repeat.value(),
                    unpack: op.unpack == Unpack::Unpack32bIn16b,
                }
            }
        }
    )*};
}

tmem_ops! {
    /// `tcgen05.ld/st.16x64b`: 16 TMEM lanes of 64 bits per repetition.
    Ld16x64bOp, St16x64bOp, TmemShape::S16x64b;
    /// `tcgen05.ld/st.16x128b`: 16 lanes of 128 bits (up to `.x64`).
    Ld16x128bOp, St16x128bOp, TmemShape::S16x128b;
    /// `tcgen05.ld/st.16x256b`: 16 lanes of 256 bits (up to `.x32`).
    Ld16x256bOp, St16x256bOp, TmemShape::S16x256b;
    /// `tcgen05.ld/st.16x32bx2`: two accesses of 16 lanes of 32 bits, the
    /// second as many columns after the first as the first spans.
    Ld16x32bx2Op, St16x32bx2Op, TmemShape::S16x32bx2;
    /// `tcgen05.ld/st.32x32b`: 32 lanes of 32 bits.
    Ld32x32bOp, St32x32bOp, TmemShape::S32x32b;
}

pub use crate::ir::TmemLoadRedOp;

macro_rules! tmem_red_ops {
    ($($(#[$doc:meta])* $ld:ident, $shape:expr $(, $split:ident)?;)*) => {$(
        $(#[$doc])*
        ///
        /// CuTe DSL's op of the same name (roadmap part m10-ldred-packed-math):
        /// `repeat` (`.x2` to `.x128`), the reduction `red_op` and `nan`
        /// (`.NaN`: any NaN is the result; otherwise NaNs are ignored). Used
        /// on Float32 atoms through [`copy_atom_call`](crate::copy_atom_call)
        /// with the data and the reduction as destinations. PTX ISA 8.8:
        /// sm_110a, sm_103f and sm_110f (not sm_100a).
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $ld {
            pub repeat: Repetition,
            pub red_op: TmemLoadRedOp,
            pub nan: bool,
            $(pub $split: u32,)?
        }

        impl $ld {
            /// The load of `repeat` repetitions reducing with `red_op`, NaNs ignored.
            pub fn new(repeat: Repetition, red_op: TmemLoadRedOp) -> Self {
                $ld { repeat, red_op, nan: false $(, $split: 0)? }
            }

            /// The load with `nan` propagation.
            pub fn with_nan(self, nan: bool) -> Self {
                $ld { nan, ..self }
            }
        }

        impl From<$ld> for CopyOp {
            fn from(op: $ld) -> CopyOp {
                CopyOp::TmemLoadRed {
                    shape: $shape,
                    num: op.repeat.value(),
                    red: op.red_op,
                    nan: op.nan,
                    half_split_off: 0 $(+ op.$split)?,
                }
            }
        }
    )*};
}

tmem_red_ops! {
    /// `tcgen05.ld.red.16x32bx2`: two accesses of 16 lanes of 32 bits. The
    /// second access's column offset `half_split_off` is 0 for as many
    /// columns as the first spans.
    LdRed16x32bx2Op, TmemShape::S16x32bx2, half_split_off;
    /// `tcgen05.ld.red.32x32b`: 32 lanes of 32 bits.
    LdRed32x32bOp, TmemShape::S32x32b;
}

/// `tcgen05.ld.spcompress.32x32b` (roadmap part m10-rubin-extras; CuTe DSL's
/// `tcgen05.LdSPCompress32x32bOp(repeat, redOp)`): 32 lanes of 32 bits, `.x4`
/// to `.x128`, loading Float32 values and keeping two of every four of each
/// thread's as `red_op` prefers (the larger for `Max`, the smaller for
/// `Min`, by absolute value for the `Abs` forms). Issued with
/// [`copy_atom_call_sp`](crate::copy_atom_call_sp), which returns the
/// compressed data and the 2:4 metadata the sparse MMAs consume. PTX ISA
/// 9.4: sm_107a only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LdSPCompress32x32bOp {
    pub repeat: Repetition,
    pub red_op: TmemLoadRedOp,
}

impl LdSPCompress32x32bOp {
    /// The load of `repeat` repetitions choosing with `red_op`.
    pub fn new(repeat: Repetition, red_op: TmemLoadRedOp) -> Self {
        LdSPCompress32x32bOp { repeat, red_op }
    }
}

impl From<LdSPCompress32x32bOp> for CopyOp {
    fn from(op: LdSPCompress32x32bOp) -> CopyOp {
        CopyOp::TmemLoadSpCompress { num: op.repeat.value(), red: op.red_op, with_red: false, nan: false }
    }
}

/// `tcgen05.ld.red.spcompress.32x32b` (CuTe DSL's
/// `tcgen05.LdRedSPCompress32x32bOp(repeat, redOp, nan)`): as
/// [`LdSPCompress32x32bOp`], and also the thread's reduction over all its
/// loaded values with `red_op` (as [`LdRed32x32bOp`] reduces; `nan` makes
/// any NaN the result, and the most preferred value when choosing).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LdRedSPCompress32x32bOp {
    pub repeat: Repetition,
    pub red_op: TmemLoadRedOp,
    pub nan: bool,
}

impl LdRedSPCompress32x32bOp {
    /// The load of `repeat` repetitions, NaNs ignored.
    pub fn new(repeat: Repetition, red_op: TmemLoadRedOp) -> Self {
        LdRedSPCompress32x32bOp { repeat, red_op, nan: false }
    }

    /// The load with `nan` propagation.
    pub fn with_nan(self, nan: bool) -> Self {
        LdRedSPCompress32x32bOp { nan, ..self }
    }
}

impl From<LdRedSPCompress32x32bOp> for CopyOp {
    fn from(op: LdRedSPCompress32x32bOp) -> CopyOp {
        CopyOp::TmemLoadSpCompress { num: op.repeat.value(), red: op.red_op, with_red: true, nan: op.nan }
    }
}

/// A data path lane's stride in bits (CuTe's `TMEM::DP_b`): a TMEM address
/// has the lane in bits 31:16 and the 32-bit column in bits 15:0.
const DP_BITS: usize = 1 << 21;

/// The `ValID` layout (in bits) of an `.x<num>` access: bit of the atom ->
/// TMEM bit address. With packing a register's 32 bits come from the low
/// halves of two columns.
pub(crate) fn val_id_bits(shape: TmemShape, num: u32, packed: bool) -> Layout {
    let lanes = shape.lanes() as usize;
    let per_lane = 1024 * shape.registers(num) as usize / lanes;
    if packed {
        Layout::new(((16, per_lane / 16), lanes), ((1, 32), DP_BITS))
    } else {
        Layout::new((per_lane, lanes), (1, DP_BITS))
    }
}

/// The register side (in bits) of an `.x<num>` access: (thread, bit) -> bit
/// of the atom, as `ValID` numbers them.
pub(crate) fn registers_tv_bits(shape: TmemShape, num: u32) -> Layout {
    let n = num as usize;
    match (shape, n) {
        (TmemShape::S32x32b, _) => Layout::new((32, 32 * n), (32 * n, 1)),
        (TmemShape::S16x64b, 1) => Layout::new(((2, 2, 8), 32), ((512, 32, 64), 1)),
        (TmemShape::S16x64b, _) => Layout::new(((2, 2, 8), (32, n)), ((512 * n, 32, 64 * n), (1, 64))),
        (TmemShape::S16x128b, 1) => Layout::new(((4, 8), (32, 2)), ((32, 128), (1, 1024))),
        (TmemShape::S16x128b, _) => Layout::new(((4, 8), (32, 2, n)), ((32, 128 * n), (1, 1024 * n, 128))),
        (TmemShape::S16x256b, 1) => Layout::new(((4, 8), (64, 2)), ((64, 256), (1, 2048))),
        (TmemShape::S16x256b, _) => Layout::new(((4, 8), (64, 2, n)), ((64, 256 * n), (1, 2048 * n, 256))),
        (TmemShape::S16x32bx2, _) => Layout::new(((16, 2), 32 * n), ((64 * n, 32 * n), 1)),
    }
}

/// The `(thread, value)` layouts of a TMEM atom on elements of `bits` bits:
/// `(thr_id, src, dst, reference)`. `load` puts the TMEM side (every thread
/// sees the whole atom, as CuTe's `SrcLayout` `(32, N):(0, 1)`) at the
/// source, a store at the destination; the reference is the TMEM side.
#[track_caller]
pub(crate) fn tmem_atom_layouts(op: CopyOp, elem: crate::ir::DType) -> (Layout, Layout, Layout, Layout) {
    let (shape, num, packed, load) = match op {
        CopyOp::TmemLoad { shape, num, pack } => (shape, num, pack, true),
        CopyOp::TmemLoadRed { shape, num, .. } => (shape, num, false, true),
        CopyOp::TmemLoadSpCompress { num, .. } => (TmemShape::S32x32b, num, false, true),
        CopyOp::TmemStore { shape, num, unpack } => (shape, num, unpack, false),
        _ => unreachable!("a TMEM load or store"),
    };
    let name = tmem_op_name(shape, load);
    let bits = elem.storage_bits();
    if let CopyOp::TmemLoadSpCompress { with_red, .. } = op {
        assert!(
            elem == crate::ir::DType::F32 && (4..=128).contains(&num),
            "cute: make_copy_atom({}, {elem}): tcgen05.ld{}.spcompress loads Float32 values, repeated .x4 to .x128 (x{num} is not one)",
            if with_red { "LdRedSPCompress32x32bOp" } else { "LdSPCompress32x32bOp" },
            if with_red { ".red" } else { "" }
        );
    }
    if matches!(op, CopyOp::TmemLoadRed { .. }) {
        assert!(
            elem == crate::ir::DType::F32 && num >= 2,
            "cute: make_copy_atom({name}, {elem}): tcgen05.ld.red loads Float32 values, repeated .x2 to .x128 (x{num} is not one)"
        );
    }
    assert!(
        !elem.is_word_packed() && bits <= 32 && 32 % bits == 0,
        "cute: make_copy_atom({name}, {elem}): TMEM holds 32-bit cells, so the element type has 8, 16 or 32 bits"
    );
    assert!(
        num.is_power_of_two() && num <= shape.max_repetition(),
        "cute: make_copy_atom({name}, {elem}): repetition x{num} moves {} registers per thread, but \
         tcgen05.{}.{} takes at most 128 (up to x{})",
        shape.registers(num),
        if load { "ld" } else { "st" },
        shape.ptx(),
        shape.max_repetition()
    );
    assert!(
        !packed || bits <= 16,
        "cute: make_copy_atom({name}, {elem}): {} moves 16-bit halves, so the element type has at most 16 bits",
        if load { "Pack.PACK_16b_IN_32b" } else { "Unpack.UNPACK_32b_IN_16b" }
    );
    let total = 1024 * shape.registers(num) as usize;
    let tmem_side = Layout::new((32, total / bits), (0, 1));
    let regs = registers_tv_bits(shape, num).upcast_by(bits);
    let thr = Layout::new(32, 1);
    if load { (thr, tmem_side.clone(), regs, tmem_side) } else { (thr, regs, tmem_side.clone(), tmem_side) }
}

/// The TMEM layout (in elements of `bits` bits) a TMEM atom's TMEM operand
/// has: `ValID` converted, coalesced.
pub(crate) fn tmem_val_layout(op: CopyOp, bits: usize) -> Layout {
    let (shape, num, packed) = match op {
        CopyOp::TmemLoad { shape, num, pack } => (shape, num, pack),
        CopyOp::TmemLoadRed { shape, num, .. } => (shape, num, false),
        CopyOp::TmemLoadSpCompress { num, .. } => (TmemShape::S32x32b, num, false),
        CopyOp::TmemStore { shape, num, unpack } => (shape, num, unpack),
        _ => unreachable!("a TMEM load or store"),
    };
    val_id_bits(shape, num, packed).upcast_by(bits).coalesce()
}

/// CuTe DSL's name for the op of `shape`.
pub(crate) fn tmem_op_name(shape: TmemShape, load: bool) -> String {
    let s = match shape {
        TmemShape::S16x64b => "16x64b",
        TmemShape::S16x128b => "16x128b",
        TmemShape::S16x256b => "16x256b",
        TmemShape::S16x32bx2 => "16x32bx2",
        TmemShape::S32x32b => "32x32b",
    };
    format!("{}{s}Op", if load { "Ld" } else { "St" })
}

/// A tiled copy of a TMEM load or store atom over the TMEM tensor `tmem`
/// (CuTe DSL's `tcgen05.make_tmem_copy`, CuTe C++'s `make_tmem_copy`): the
/// atom's warp is repeated over the four warps of a warpgroup, warp `i` at
/// TMEM lanes `32 i` onwards (its lane quadrant), and over the columns of
/// `tmem`, giving a 128-thread copy. `partition_s(tmem)` (for a load) gives
/// each thread the TMEM view its warp's instructions address, and
/// `partition_d` of a register fragment, or of a global or shared tensor of
/// the same shape (for its shape), its registers; a store is the other way
/// round. The atom's element type has `tmem`'s width.
#[track_caller]
pub fn make_tmem_copy<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tmem: &Tensor<U>) -> TiledCopy<T> {
    assert!(
        matches!(atom.op(), CopyOp::TmemLoad { .. } | CopyOp::TmemStore { .. }),
        "cute: make_tmem_copy takes a TMEM load or store atom (tcgen05.Ld..Op or St..Op), not {:?}",
        atom.op()
    );
    assert!(
        tmem.memory_space().is_some() && tmem.memspace() == AddressSpace::Tmem,
        "cute: make_tmem_copy: the tensor {} is not in tensor memory",
        tmem.layout()
    );
    let bits = T::DTYPE.storage_bits();
    assert_eq!(
        bits,
        U::DTYPE.storage_bits(),
        "cute: make_tmem_copy: the atom's {} and the TMEM tensor's {} differ in width",
        T::DTYPE,
        U::DTYPE
    );
    // Atom thread -> TMEM address: the four warps at the same place of their own lane quadrants.
    let atom_t = Layout::new((32, 4), (0, 32 * (DP_BITS / bits)));
    // Atom value -> TMEM address.
    let atom_v = tmem_val_layout(atom.op(), bits);
    make_cotiled_copy(atom, &Layout::from_modes([atom_t, atom_v]), tmem.layout())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A few of CuTe C++'s tables, as copied from copy_traits_sm100.hpp.
    #[test]
    fn layouts_match_cute_tables() {
        let dp = DP_BITS;
        assert_eq!(
            registers_tv_bits(TmemShape::S16x64b, 2),
            Layout::new(((2, 2, 8), (32, 2)), ((1024, 32, 128), (1, 64)))
        );
        assert_eq!(registers_tv_bits(TmemShape::S16x128b, 1), Layout::new(((4, 8), (32, 2)), ((32, 128), (1, 1024))));
        assert_eq!(
            registers_tv_bits(TmemShape::S16x256b, 32),
            Layout::new(((4, 8), (64, 2, 32)), ((64, 8192), (1, 65536, 256)))
        );
        assert_eq!(registers_tv_bits(TmemShape::S16x32bx2, 2), Layout::new(((16, 2), 64), ((128, 64), 1)));
        assert_eq!(registers_tv_bits(TmemShape::S32x32b, 128), Layout::new((32, 4096), (4096, 1)));
        assert_eq!(val_id_bits(TmemShape::S16x256b, 1, true), Layout::new(((16, 16), 16), ((1, 32), dp)));
        assert_eq!(val_id_bits(TmemShape::S16x32bx2, 1, true), Layout::new(((16, 4), 16), ((1, 32), dp)));
        assert_eq!(val_id_bits(TmemShape::S32x32b, 1, true), Layout::new(((16, 2), 32), ((1, 32), dp)));
        assert_eq!(val_id_bits(TmemShape::S16x64b, 2, false), Layout::new((128, 16), (1, dp)));
    }

    /// The layouts and the simulator's fragment map (`TmemShape::cells`)
    /// agree: register `r` of thread `i` is at the TMEM cells the layouts
    /// put its bits at.
    #[test]
    fn layouts_match_fragments() {
        for shape in
            [TmemShape::S16x64b, TmemShape::S16x128b, TmemShape::S16x256b, TmemShape::S16x32bx2, TmemShape::S32x32b]
        {
            for num in [1, 2, 4, 8] {
                for packed in [false, true] {
                    let tv = registers_tv_bits(shape, num);
                    let val = val_id_bits(shape, num, packed);
                    for i in 0..32u32 {
                        for r in 0..shape.registers(num) {
                            let cells = shape.cells(i, r, num, packed);
                            for (h, &(l, c)) in cells.iter().enumerate() {
                                let bit = 32 * r as usize + 16 * h;
                                let atom_bit = tv.call(i as usize + 32 * bit);
                                let addr = val.call(atom_bit);
                                assert_eq!(
                                    (addr / DP_BITS, addr % DP_BITS),
                                    (l as usize, 32 * c as usize),
                                    "{shape:?} x{num} packed {packed}: thread {i} register {r} half {h}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// m10-umma-pipelines: TMEM copy introspection (CuTe DSL's tcgen05 helpers
// is_tmem_load, is_tmem_store and get_tmem_copy_properties).

/// The packing of a TMEM copy atom, as [`get_tmem_copy_properties`]
/// returns it: a load's [`Pack`] or a store's [`Unpack`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmemPacking {
    Pack(Pack),
    Unpack(Unpack),
}

impl CopyOp {
    /// Whether this is a TMEM load (`tcgen05.ld`).
    pub fn is_tmem_load(&self) -> bool {
        matches!(self, CopyOp::TmemLoad { .. } | CopyOp::TmemLoadRed { .. } | CopyOp::TmemLoadSpCompress { .. })
    }

    /// Whether this is a TMEM store (`tcgen05.st`).
    pub fn is_tmem_store(&self) -> bool {
        matches!(self, CopyOp::TmemStore { .. })
    }

    /// A TMEM copy's data paths (16 or 32 lanes), bits per lane and
    /// repetition (32, 64, 128 or 256), repetition count and packing;
    /// `None` for any other operation.
    pub fn tmem_copy_properties(&self) -> Option<(u32, u32, u32, TmemPacking)> {
        let (shape, num, packing) = match *self {
            CopyOp::TmemLoad { shape, num, pack } => {
                (shape, num, TmemPacking::Pack(if pack { Pack::Pack16bIn32b } else { Pack::None }))
            }
            CopyOp::TmemLoadRed { shape, num, .. } => (shape, num, TmemPacking::Pack(Pack::None)),
            CopyOp::TmemLoadSpCompress { num, .. } => (TmemShape::S32x32b, num, TmemPacking::Pack(Pack::None)),
            CopyOp::TmemStore { shape, num, unpack } => {
                (shape, num, TmemPacking::Unpack(if unpack { Unpack::Unpack32bIn16b } else { Unpack::None }))
            }
            _ => return None,
        };
        let bits = match shape {
            TmemShape::S16x64b => 64,
            TmemShape::S16x128b => 128,
            TmemShape::S16x256b => 256,
            TmemShape::S16x32bx2 | TmemShape::S32x32b => 32,
        };
        Some((shape.lanes(), bits, num, packing))
    }
}

/// Whether `atom` is a TMEM load (CuTe DSL's `tcgen05.is_tmem_load`).
pub fn is_tmem_load<T: Numeric>(atom: &CopyAtom<T>) -> bool {
    atom.op().is_tmem_load()
}

/// Whether `atom` is a TMEM store (CuTe DSL's `tcgen05.is_tmem_store`).
pub fn is_tmem_store<T: Numeric>(atom: &CopyAtom<T>) -> bool {
    atom.op().is_tmem_store()
}

/// The properties of a TMEM copy atom (CuTe DSL's
/// `tcgen05.get_tmem_copy_properties`): its data paths (16 or 32), bits
/// (32, 64, 128 or 256), repetition count and packing. Panics for an atom
/// that is not a TMEM copy.
#[track_caller]
pub fn get_tmem_copy_properties<T: Numeric>(atom: &CopyAtom<T>) -> (u32, u32, u32, TmemPacking) {
    let op = atom.op();
    op.tmem_copy_properties()
        .unwrap_or_else(|| panic!("cute: get_tmem_copy_properties: {op:?} is not a TMEM load or store"))
}

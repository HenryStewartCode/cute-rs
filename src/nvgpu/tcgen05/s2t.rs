// SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
//
// Portions of this file are ported from NVIDIA CUTLASS 4.8.1:
//   include/cute/arch/copy_sm100.hpp (the SM100_UTCCP_* instructions) and
//   include/cute/atom/copy_traits_sm100.hpp (their copy traits' ValID,
//   source and destination layouts, and make_utccp_copy), and
//   include/cute/atom/mma_traits_sm100.hpp (get_utccp_smem_desc_tensor),
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//     (mma_traits_sm100.hpp: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES.)
// Those portions are under the BSD-3-Clause licence, and the rest of the file
// is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the
// licence text and lists what each part ports.

//! Shared-memory-to-TMEM copies: CuTe DSL's
//! `tcgen05.Cp128x256bOp`, `Cp128x128bOp`, `Cp4x256bOp`, `Cp4x32x128bOp`,
//! `Cp2x64x128b0213Op` and `Cp2x64x128b0123Op` (`tcgen05.cp`), used through
//! [`make_copy_atom`](crate::make_copy_atom), [`make_s2t_copy`],
//! [`get_s2t_smem_desc_tensor`] and [`copy`](crate::copy), with
//! [`append_s2t_broadcast_mode`] for the broadcast forms.
//!
//! An S2T copy reads its source through shared-memory matrix descriptors,
//! one per atom: the SMEM side of the copy is partitioned like any other,
//! then turned into a tensor of descriptors ([`get_s2t_smem_desc_tensor`]),
//! which the copy consumes with the partitioned TMEM destination. A whole
//! converged warp calls the copy and one lane, elected by the copy, issues
//! each `tcgen05.cp` (do not wrap it in `elect_one`). The copies run on the
//! CTA's tensor-core pipe, in order with its MMAs, and complete through
//! [`commit`](super::commit).
//!
//! The atoms' TMEM side is CuTe C++'s `ValID` (in bits, a bit of the atom ->
//! its TMEM bit address `lane * 2^21 + column * 32 + bit`): `(rows, row
//! bits)` for the plain shapes, with a broadcast mode of TMEM-lane stride for
//! `.warpx4` and `.warpx2`. An atom holds that many values of its element
//! type: the TMEM footprint, so for `Cp4x256bOp` the 128 x 256-bit view CuTe
//! C++ calls the refresh view.

use crate::atom::copy::{CopyAtom, CopyOp, TiledCopy, make_cotiled_copy};
use crate::ir::TmemCpShape;
use crate::layout::{Layout, Tileable};
use crate::nvgpu::OperandMajorMode;
use crate::tensor::{AddressSpace, Engine, Tensor, TensorInner};
use crate::value::Numeric;

pub use crate::nvgpu::cpasync::CtaGroup;

use super::smem_desc::{UmmaDescEncoding, UmmaLayoutType, umma_desc_fields};

/// A data path lane's stride in bits (CuTe's `TMEM::DP_b`).
const DP_BITS: usize = 1 << 21;

macro_rules! s2t_ops {
    ($($(#[$doc:meta])* $name:ident, $shape:expr;)*) => {$(
        $(#[$doc])*
        ///
        /// Takes the CTA group (`CtaGroup::TWO`: the pair's even CTA issues
        /// the copy and each CTA of the pair copies its own shared memory into
        /// its own TMEM). Admissible on the sm_100f and sm_110f families
        /// (sm_100a, sm_110a).
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name {
            pub cta_group: CtaGroup,
        }

        impl $name {
            /// The copy for `cta_group`.
            pub fn new(cta_group: CtaGroup) -> Self {
                $name { cta_group }
            }

            /// TMEM core matrices written per core matrix read (CuTe DSL's
            /// `broadcast_factor`): 4 for `.warpx4`, 2 for `.warpx2`, else 1.
            pub fn broadcast_factor(&self) -> usize {
                $shape.broadcast()
            }
        }

        impl From<$name> for CopyOp {
            fn from(op: $name) -> CopyOp {
                CopyOp::TmemCp { shape: $shape, two_cta: op.cta_group == CtaGroup::TWO }
            }
        }
    )*};
}

s2t_ops! {
    /// `tcgen05.cp.128x256b`: 128 rows of 256 bits into 128 TMEM lanes.
    Cp128x256bOp, TmemCpShape::S128x256b;
    /// `tcgen05.cp.128x128b`: 128 rows of 128 bits into 128 TMEM lanes.
    Cp128x128bOp, TmemCpShape::S128x128b;
    /// `tcgen05.cp.4x256b`: 4 rows of 256 bits into TMEM lanes 0, 32, 64
    /// and 96.
    Cp4x256bOp, TmemCpShape::S4x256b;
    /// `tcgen05.cp.32x128b.warpx4`: 32 rows of 128 bits broadcast into each
    /// of the four 32-lane quadrants.
    Cp4x32x128bOp, TmemCpShape::S32x128bWarpx4;
    /// `tcgen05.cp.64x128b.warpx2::02_13`: 64 rows of 128 bits, row `r` into
    /// lanes `r` and `64 + r` (quadrants 0 and 2, 1 and 3).
    Cp2x64x128b0213Op, TmemCpShape::S64x128bWarpx2_02_13;
    /// `tcgen05.cp.64x128b.warpx2::01_23`: 64 rows of 128 bits, rows 0 to 31
    /// into quadrants 0 and 1, rows 32 to 63 into quadrants 2 and 3.
    Cp2x64x128b0123Op, TmemCpShape::S64x128bWarpx2_01_23;
}

impl CopyOp {
    /// Whether this is a shared-memory-to-TMEM copy (`tcgen05.cp`).
    pub fn is_s2t(&self) -> bool {
        matches!(self, CopyOp::TmemCp { .. })
    }

    /// An S2T copy's broadcast factor (1, 2 or 4); `None` for any other
    /// operation.
    pub fn s2t_broadcast_factor(&self) -> Option<usize> {
        match self {
            CopyOp::TmemCp { shape, .. } => Some(shape.broadcast()),
            _ => None,
        }
    }
}

/// CuTe C++'s `Copy_Traits<SM100_UTCCP_*>::ValID`, in bits: bit of the atom
/// -> TMEM bit address, ordered (core-matrix rows, row bits, repeat,
/// broadcast).
pub(crate) fn val_id_bits(shape: TmemCpShape) -> Layout {
    match shape {
        TmemCpShape::S128x256b | TmemCpShape::S4x256b => Layout::new((128, 256), (DP_BITS, 1)),
        TmemCpShape::S128x128b => Layout::new((128, 128), (DP_BITS, 1)),
        TmemCpShape::S32x128bWarpx4 => Layout::new((32, 128, 4), (DP_BITS, 1, 32 * DP_BITS)),
        TmemCpShape::S64x128bWarpx2_02_13 => Layout::new((64, 128, 2), (DP_BITS, 1, 64 * DP_BITS)),
        TmemCpShape::S64x128bWarpx2_01_23 => Layout::new((32, 128, 2, 2), (DP_BITS, 1, 64 * DP_BITS, 32 * DP_BITS)),
    }
}

/// CuTe DSL's name for the op of `shape`.
pub(crate) fn s2t_op_name(shape: TmemCpShape) -> &'static str {
    match shape {
        TmemCpShape::S128x256b => "Cp128x256bOp",
        TmemCpShape::S128x128b => "Cp128x128bOp",
        TmemCpShape::S4x256b => "Cp4x256bOp",
        TmemCpShape::S32x128bWarpx4 => "Cp4x32x128bOp",
        TmemCpShape::S64x128bWarpx2_02_13 => "Cp2x64x128b0213Op",
        TmemCpShape::S64x128bWarpx2_01_23 => "Cp2x64x128b0123Op",
    }
}

/// The `(thr_id, src, dst, reference)` layouts of an S2T atom on elements
/// of type `elem`: one thread per CTA of the group, each holding the atom's
/// TMEM footprint.
#[track_caller]
pub(crate) fn s2t_atom_layouts(op: CopyOp, elem: crate::ir::DType) -> (Layout, Layout, Layout, Layout) {
    let CopyOp::TmemCp { shape, two_cta } = op else { unreachable!("an S2T copy") };
    let bits = elem.storage_bits();
    assert!(
        !elem.is_word_packed() && bits <= 32 && 32 % bits == 0,
        "cute: make_copy_atom({}, {elem}): tensor memory holds 32-bit cells, so the element type has 8, 16 or 32 bits",
        s2t_op_name(shape)
    );
    let n = if two_cta { 2 } else { 1 };
    let values = val_id_bits(shape).size() / bits;
    let side = Layout::new((n, values), (0, 1));
    (Layout::new(n, 1), side.clone(), side.clone(), side)
}

/// The TMEM layout (in elements of `bits` bits) of an S2T atom: `ValID`
/// upcast, coalesced.
pub(crate) fn s2t_val_layout(shape: TmemCpShape, bits: usize) -> Layout {
    val_id_bits(shape).upcast_by(bits).coalesce()
}

/// A tiled copy of an S2T atom over the TMEM tensor `tmem` (CuTe DSL's
/// `tcgen05.make_s2t_copy`, CuTe C++'s `make_utccp_copy`): zero-stride modes
/// of `tmem` are filtered out first; the copy has one thread per CTA of the
/// atom's group, all at the same TMEM addresses, and the atom's values tile
/// `tmem`. Partition with `get_slice(0)`: `partition_d(tmem)` gives the TMEM
/// destination, and `partition_s` of the shared-memory tensor holding the
/// same coordinates the source, which [`get_s2t_smem_desc_tensor`] turns
/// into descriptors. The atom's element type has `tmem`'s width.
#[track_caller]
pub fn make_s2t_copy<T: Numeric, U: Numeric>(atom: &CopyAtom<T>, tmem: &Tensor<U>) -> TiledCopy<T> {
    let CopyOp::TmemCp { shape, two_cta } = atom.op() else {
        panic!(
            "cute: make_s2t_copy takes an SMEM-to-TMEM copy atom (tcgen05.Cp..Op), not {:?}: TMEM loads and stores \
             tile with make_tmem_copy",
            atom.op()
        )
    };
    assert!(
        tmem.memory_space().is_some() && tmem.memspace() == AddressSpace::Tmem,
        "cute: make_s2t_copy: the tensor {} is not in tensor memory",
        tmem.layout()
    );
    let bits = T::DTYPE.storage_bits();
    assert_eq!(
        bits,
        U::DTYPE.storage_bits(),
        "cute: make_s2t_copy: the atom's {} and the TMEM tensor's {} differ in width",
        T::DTYPE,
        U::DTYPE
    );
    let compact = tmem.layout().filter_zeros();
    let atom_t = Layout::new(if two_cta { 2 } else { 1 }, 0);
    let atom_v = s2t_val_layout(shape, bits);
    make_cotiled_copy(atom, &Layout::from_modes([atom_t, atom_v]), &compact)
}

/// The shared-memory descriptor view of an S2T copy's partitioned source
/// (CuTe DSL's `tcgen05.get_s2t_smem_desc_tensor`, CuTe C++'s
/// `get_utccp_smem_desc_tensor`): `smem` is `partition_s` of the
/// shared-memory tensor by the S2T tiled copy's slice 0, whose mode 0 is the
/// atom's values `(V, 1)`. The first `rows x row-bits` values of each atom
/// (the core matrices, without the broadcast and repeat modes) must be a
/// canonical K-major UMMA layout (8-row groups of 16-byte core matrices,
/// unswizzled or swizzled as the tensor's pointer is); the result is a
/// tensor of descriptors with `smem`'s layout counted in 16-byte units, the
/// descriptor of each atom at its first element. The encoding follows the
/// target (15-bit fields on sm_107).
#[track_caller]
pub fn get_s2t_smem_desc_tensor<T: Numeric, U: Numeric>(
    atom: &impl crate::atom::copy::AsCopyAtom<T>,
    smem: &Tensor<U>,
) -> Tensor<U> {
    let op = atom.copy_atom().op();
    let CopyOp::TmemCp { shape, .. } = op else {
        panic!("cute: get_s2t_smem_desc_tensor takes an SMEM-to-TMEM copy atom (tcgen05.Cp..Op), not {op:?}")
    };
    let fail = |why: String| -> ! { panic!("cute: get_s2t_smem_desc_tensor({}): {why}", s2t_op_name(shape)) };
    let Engine::Memory { buffer, space: crate::ir::MemSpace::Shared, peer: None } = smem.engine else {
        fail(format!("the source {smem:?} is not a tensor in this CTA's shared memory"))
    };
    let layout = smem.layout();
    let vec = layout.mode(0);
    if vec.rank() != 2 || vec.mode(1).shape().product().as_static() != Some(1) {
        fail(format!(
            "mode 0 of the partitioned source {layout} is not the atom's (values, 1): partition the shared-memory \
             tensor with the make_s2t_copy tiled copy's get_slice(0).partition_s"
        ));
    }
    if !vec.is_static() {
        fail(format!("mode 0 of the partitioned source {layout} has runtime extents or strides"));
    }
    let bits = U::DTYPE.storage_bits();
    let rows = shape.rows_logical();
    let k = shape.row_bits() / bits;
    let flat = vec.mode(0).flatten();
    if shape.broadcast() > 1 {
        let strides = flat.static_stride().flat_leaves();
        let extents = flat.static_shape().flat_leaves();
        if !strides.iter().zip(&extents).any(|(&d, &e)| d == 0 && e > 1) {
            fail(format!(
                "the atom's values {} have no zero-stride broadcast mode, but the copy writes each core matrix {} \
                 times: add it to the shared-memory tensor with tcgen05::append_s2t_broadcast_mode (after \
                 filter_zeros) before partitioning",
                vec.mode(0),
                shape.broadcast()
            ));
        }
    }
    if flat.size() < rows * k {
        fail(format!("the atom's values {} are fewer than its {rows} x {k} core-matrix elements", vec.mode(0)));
    }
    let tile = flat.with_shape((rows, k));
    let layout_type = match &smem.inner {
        TensorInner::Identity => UmmaLayoutType::SwizzleNone,
        TensorInner::Swizzle { bytes: Some(sw), .. } => UmmaLayoutType::from_byte_swizzle(*sw)
            .unwrap_or_else(|| fail(format!("the source's swizzle {sw} (on bytes) is none of UMMA's"))),
        _ => fail("the source is over a composed layout UMMA descriptors do not describe".into()),
    };
    let enc = crate::builder::target_arch().map(UmmaDescEncoding::for_arch).unwrap_or_default();
    // A 128-bit row is one core matrix along K: stand in a second one where the canonical form puts
    // it, so that the K-major check (32 bytes of K) applies; the hardware reads no LBO for it.
    let t = 128 / bits;
    let tile = if k == t {
        if tile.mode(1).coalesce() != Layout::new(t, 1) {
            fail(format!("a core-matrix row of {tile} is not {t} contiguous elements (16 bytes)"));
        }
        let lbo = if layout_type == UmmaLayoutType::SwizzleNone { 0 } else { t };
        Layout::from_modes([tile.mode(0), Layout::new((t, 2), (1, lbo))])
    } else {
        tile
    };
    let fields = umma_desc_fields(&tile, OperandMajorMode::K, layout_type, bits, enc)
        .unwrap_or_else(|why| fail(format!("the core matrices are not a canonical K-major UMMA layout: {why}")));
    let align = crate::builder::with(|b| b.buffer(buffer).align);
    let repeat = layout_type.repeat_bytes();
    if align < repeat {
        fail(format!("the source's buffer is aligned to {align} bytes, but {layout_type:?} needs {repeat}"));
    }
    for i in 1..layout.rank() {
        let m = layout.mode(i);
        if let Some(bad) = m.static_stride().flat_leaves().into_iter().find(|&d| (d * bits) % 128 != 0) {
            fail(format!(
                "mode {i} of the partitioned source ({m}) has a stride of {bad} elements, which does not start a \
                 16-byte unit"
            ));
        }
    }
    let units = crate::layout::recast_layout(128, bits, layout);
    let addr: crate::value::Value<u32> = smem.ptr().toint_i32().to::<u32>();
    let start = (addr.to::<u64>() >> 4u64) & ((1u64 << enc.address_bits()) - 1);
    let desc = start | fields.encode(enc);
    Tensor::new(Engine::SmemDesc { desc: desc.checked() }, units)
}

/// CuTe DSL's `tcgen05.append_s2t_broadcast_mode(op, smem_tensor)`: for an
/// S2T op with a broadcast factor `b` above 1, `smem`'s mode 0 `(MN, K)`
/// becomes `((MN..., b:0), K)` (the broadcast as a zero-stride mode, which
/// only TMEM has), the other modes kept, so that the shared-memory operand
/// partitions as the TMEM side does; other S2T ops return `smem` as it is.
/// Panics for an op that is not an S2T copy, or a mode 0 that is not
/// `(MN, K)` followed by at least one more mode.
#[track_caller]
pub fn append_s2t_broadcast_mode<T>(op: impl Into<CopyOp>, smem: &Tensor<T>) -> Tensor<T> {
    let op = op.into();
    let Some(factor) = op.s2t_broadcast_factor() else {
        panic!("cute: append_s2t_broadcast_mode takes an SMEM-to-TMEM copy op (tcgen05.Cp..Op), not {op:?}")
    };
    if factor == 1 {
        return smem.clone();
    }
    let layout = smem.layout();
    assert!(
        layout.rank() >= 2 && layout.mode(0).rank() == 2,
        "cute: append_s2t_broadcast_mode: the shared-memory operand {layout} of a broadcast copy (factor {factor}) \
         needs mode 0 as (MN, K) and further modes after it"
    );
    let mn = layout.mode(0).mode(0).append(Layout::new(factor, 0));
    let mma = Layout::from_modes([mn, layout.mode(0).mode(1)]);
    let mut modes = vec![mma];
    modes.extend((1..layout.rank()).map(|i| layout.mode(i)));
    smem.with_layout(Layout::from_modes(modes))
}

impl TmemCpShape {
    /// Rows of the core-matrix part of the atom's `ValID` (its first mode):
    /// the source rows, except 128 for `.4x256b` (CuTe's refresh view) and
    /// 32 for `.64x128b.warpx2::01_23` (whose rows repeat).
    pub(crate) fn rows_logical(self) -> usize {
        match self {
            TmemCpShape::S4x256b => 128,
            TmemCpShape::S64x128bWarpx2_01_23 => 32,
            s => s.rows(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TMEM footprint of each `ValID` matches the shapes' lane map.
    #[test]
    fn val_ids_match_lanes() {
        for shape in TmemCpShape::ALL {
            let v = val_id_bits(shape);
            let mut lanes: Vec<usize> = (0..v.size()).map(|i| v.call(i) / DP_BITS).collect();
            lanes.sort_unstable();
            lanes.dedup();
            let mut want: Vec<usize> = (0..shape.rows()).flat_map(|r| shape.lanes_of_row(r)).collect();
            want.sort_unstable();
            want.dedup();
            if shape != TmemCpShape::S4x256b {
                assert_eq!(lanes, want, "{shape:?}");
            }
            let cols = (0..v.size()).map(|i| v.call(i) % DP_BITS).max().unwrap() + 1;
            assert_eq!(cols, shape.row_bits(), "{shape:?}");
        }
    }
}

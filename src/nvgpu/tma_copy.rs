// SPDX-License-Identifier: MIT OR Apache-2.0

//! Issuing TMA tile copies: [`copy`](crate::copy) with an atom from
//! [`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom) on
//! `CopyBulkTensorTileG2SOp`, its multicast form
//! `CopyBulkTensorTileG2SMulticastOp`,
//! `CopyBulkTensorTileS2GOp` and `CopyReduceBulkTensorTileS2GOp`, and the prefetches of TMA boxes and descriptors
//! ([`prefetch_atom`], [`prefetch_descriptor`]).
//!
//! The global side is the TMA tensor's partition from
//! [`tma_partition`](crate::cpasync::tma_partition) (a coordinate tensor),
//! the other the shared-memory partition. Both are canonicalised as
//! `cute.copy` does (rank 2, the rest modes grouped); mode 0 holds whole
//! boxes, and each box (an application of the atom) is one instruction whose
//! coordinates are the coordinate tensor's element at the box's start, one
//! per TMA dimension in the descriptor's order, and whose shared-memory
//! operand is the box's start, unswizzled (the descriptor's swizzle places
//! the bytes, by address). One elected lane of each warp that executes the
//! copy issues every box, as CuTe DSL wraps the issue in `elect_one` itself:
//! call the copy from one warp, without electing a thread. A load's
//! mbarrier initialisation and `expect_tx`, and a store's bulk async-group
//! commit and wait, stay the program's own work.

use std::panic::Location;

use crate::atom::copy::{CopyArgs, CopyAtom, CopyElem, CopyOp};
use crate::builder;
use crate::control::if_;
use crate::cpasync::{CtaGroup, ReductionKind};
use crate::ir::{DType, MemSpace, Op, ValueId};
use crate::layout::{IndexExpr, Layout};
use crate::tensor::{Ident, Tensor, index_operand};
use crate::value::{IntoValue, Numeric};

/// The element types PTX's `cp.reduce.async.bulk.tensor` takes for each
/// reduction (the PTX ISA's table for the instruction): `add` on `.f16`,
/// `.bf16` (both `.noftz`), `.f32`, `.u32`, `.s32` and `.u64`; `min` and
/// `max` on `.f16`, `.bf16`, `.u32`, `.s32`, `.u64` and `.s64`; `inc` and
/// `dec` on `.u32`; `and`, `or` and `xor` on `.b32` and `.b64`. Returns why
/// not.
pub(crate) fn tma_reduce_type_rule(kind: ReductionKind, dt: DType) -> Result<(), String> {
    use DType::*;
    let (ok, rule) = match kind {
        ReductionKind::ADD => (
            matches!(dt, F16 | BF16 | F32 | U32 | I32 | U64),
            ".add takes .f16, .bf16 (.noftz), .f32, .u32, .s32 and .u64",
        ),
        ReductionKind::MIN | ReductionKind::MAX => (
            matches!(dt, F16 | BF16 | U32 | I32 | U64 | I64),
            ".min and .max take .f16, .bf16, .u32, .s32, .u64 and .s64",
        ),
        ReductionKind::INC | ReductionKind::DEC => (matches!(dt, U32), ".inc and .dec take .u32"),
        ReductionKind::AND | ReductionKind::OR | ReductionKind::XOR => {
            (matches!(dt, U32 | I32 | U64 | I64), ".and, .or and .xor take .b32 and .b64 (32- and 64-bit integers)")
        }
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "cp.reduce.async.bulk.tensor .{} does not take {} elements: in PTX {rule}",
            kind.ptx_name(),
            dt.cute_name()
        ))
    }
}

/// The boxes of a TMA copy between the coordinate tensor `coords` and the
/// shared-memory tensor `smem`: for each, its start in `smem` (an element
/// offset) and its coordinates, one i32 per TMA dimension (and the im2col
/// offsets, or nothing, by the atom's mode). Checks the canonical shapes, the
/// box's contiguity and the static alignment.
#[track_caller]
fn boxes<T: Numeric>(
    atom: &CopyAtom<T>,
    coords: &Tensor<Ident>,
    smem: &Tensor<T>,
    what: &str,
    func: &str,
) -> Vec<(IndexExpr, Vec<ValueId>, crate::ir::TmaMode)> {
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    let store = !atom.op().is_tma_load();
    box_leaves(atom, coords, smem, what, func)
        .into_iter()
        .map(|(o, leaves, coord)| {
            let (c, mode) = tma_operands(info, leaves, store, &format!("{coord:?}"), func);
            (o, c, mode)
        })
        .collect()
}

/// The instruction's coordinates (i32 values) and access mode for the TMA
/// tensor's coordinate `leaves` of a box: the
/// leaves themselves for a tiled atom, padded to the descriptor's rank; an
/// im2col atom's GEMM coordinate turned into the filter base and offsets.
#[track_caller]
pub(crate) fn tma_operands(
    info: &crate::cpasync::TmaAtomInfo,
    mut leaves: Vec<IndexExpr>,
    store: bool,
    coord: &str,
    func: &str,
) -> (Vec<ValueId>, crate::ir::TmaMode) {
    use crate::ir::TmaMode;
    use crate::nvgpu::tma_atom::TmaAtomMode;
    match &info.mode {
        TmaAtomMode::Im2col(g) => {
            let (c, offsets) = g.operands(&leaves);
            (c, if store { TmaMode::Im2colNoOffs } else { TmaMode::Im2col { offsets } })
        }
        _ => {
            let rank = info.spec().global_dims.len();
            if leaves.len() > rank {
                panic!(
                    "{func}: the TMA tensor's coordinates {coord} have {} entries, but the atom's descriptor has \
                     {rank} dimensions: the coordinate tensor is not the atom's TMA tensor",
                    leaves.len()
                );
            }
            leaves.resize(rank, IndexExpr::Static(0));
            (leaves.into_iter().map(|c| IntoValue::<i32>::into_value(c).checked()).collect(), TmaMode::Tile)
        }
    }
}

/// [`boxes`] before the coordinates become operands: each box's start in
/// `smem`, the TMA tensor's coordinate leaves at its start, and the
/// coordinate (for messages).
#[track_caller]
pub(crate) fn box_leaves<T: Numeric>(
    atom: &CopyAtom<T>,
    coords: &Tensor<Ident>,
    smem: &Tensor<T>,
    what: &str,
    func: &str,
) -> Vec<(IndexExpr, Vec<IndexExpr>, crate::layout::Coord)> {
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    // cute.copy's canonical form: (V, REST).
    let canon = |l: &Layout| {
        let l = l.pad_to_rank(2);
        if l.rank() == 2 {
            l
        } else {
            Layout::from_modes([l.mode(0), Layout::from_modes((1..l.rank()).map(|i| l.mode(i)))])
        }
    };
    let (cl, sl) = (canon(coords.layout()), canon(smem.layout()));
    for (l, role) in [(&cl, "TMA tensor"), (&sl, "shared-memory")] {
        if let Some((path, leaf)) = l.shape().first_dynamic() {
            panic!(
                "{func}: the {role} partition {l} has the dynamic extent {leaf:?} at mode {path:?}: a TMA copy unrolls \
                 its boxes when the kernel is built (pass CopyArgs::unroll_factor to loop over a runtime rest mode)"
            );
        }
    }
    let v = atom.num_val_src();
    let (c0, s0) = (cl.mode(0).size(), sl.mode(0).size());
    if c0 != s0 || !c0.is_multiple_of(v) {
        panic!(
            "{func}: mode 0 of the TMA tensor's partition {cl} ({c0} elements) and of the shared-memory partition {sl} \
             ({s0}) must be the same whole number of the atom's {v}-element boxes (partition both with tma_partition)"
        );
    }
    let (cr, sr) = (cl.mode(1).size(), sl.mode(1).size());
    if cr != sr {
        panic!(
            "{func}: the TMA tensor's partition {cl} and the shared-memory partition {sl} have different numbers of \
             rest elements (mode 1): {cr} and {sr}"
        );
    }
    // Each box is contiguous in the (unswizzled) shared-memory layout.
    let box_layout = sl.mode(0).logical_divide(&crate::layout::Tree::Leaf(Some(Layout::new(v, 1)))).mode(0);
    if box_layout.coalesce() != Layout::new(v, 1) {
        panic!(
            "{func}: a box of the atom is {v} contiguous elements of shared memory, but the shared-memory partition \
             lays them out as {box_layout} (partition it with tma_partition)"
        );
    }
    let _ = info;
    let n0 = c0 / v;
    let apps = n0 * cr;
    let elem_bytes = T::DTYPE.bytes();
    let (buffer, _) = smem.ptr().parts_unswizzled();
    let align = builder::with(|b| b.buffer(buffer).align);
    let mut out = Vec::with_capacity(apps);
    for a in 0..apps {
        let logical = (a % n0) * v + (a / n0) * c0;
        let coord = coords.coord(IndexExpr::Static(logical));
        let leaves: Vec<IndexExpr> = coord.leaves().into_iter().cloned().collect();
        let offset = smem.base_offset().clone() + sl.offset_at(logical);
        if let Some(o) = offset.as_static() {
            let byte = o * elem_bytes;
            if align >= 128 && !byte.is_multiple_of(128) {
                panic!(
                    "{func}: box {a} starts at byte {byte} of its shared-memory buffer, which is not 128-byte aligned, \
                     as the shared-memory side of {what} must be"
                );
            }
        }
        out.push((offset, leaves, coord));
    }
    out
}

/// A shared-memory element offset as the op's index: for packed sub-byte
/// elements, the byte (a box starts at a whole byte).
fn packed_index<T: Numeric>(o: IndexExpr) -> IndexExpr {
    if T::DTYPE.is_sub_byte() {
        IndexExpr::quotient(IndexExpr::product(o, IndexExpr::Static(T::DTYPE.bits())), IndexExpr::Static(8))
    } else {
        o
    }
}

/// Check that the shared-memory tensor `smem` of a TMA copy is plain or
/// swizzled shared memory with the descriptor's swizzle.
#[track_caller]
fn check_smem<T: Numeric>(atom: &CopyAtom<T>, smem: &Tensor<T>, what: &str, role: &str, func: &str) {
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    // This CTA's shared memory: not a map_dsmem_ptr tensor, whose
    // peer a TMA copy would drop.
    match (smem.memory_space(), smem.dsmem_peer()) {
        (Some(MemSpace::Shared), None) => {}
        _ => panic!(
            "{func}: {what} {} shared memory, but the {role} {} is {}",
            if role == "destination" { "writes" } else { "reads" },
            smem.layout(),
            if smem.is_register() { "a register fragment".to_string() } else { format!("{}", smem.memspace()) }
        ),
    }
    if let Some(inner) = smem.inner_layout() {
        panic!("{func}: {what} takes plain or swizzled shared memory, not a tensor over the inner layout {inner}");
    }
    // The descriptor places the bytes with its swizzle: the tensor must lay them out with
    // the same one, or the tile moves permuted without any error.
    let swizzle = smem.swizzle().unwrap_or_else(crate::layout::Swizzle::identity);
    match crate::nvgpu::tma_atom::tma_swizzle(swizzle, T::DTYPE.bits()) {
        Ok(s) if s == info.swizzle() => {}
        _ => panic!(
            "{func}: the atom's descriptor swizzles shared memory with {}, but the {role} {} is swizzled by {} (on \
             elements): partition the shared-memory tensor the atom was made for",
            info.swizzle().name(),
            smem.layout(),
            swizzle
        ),
    }
}

/// [`copy`](crate::copy) and its siblings with a TMA atom.
#[track_caller]
pub(crate) fn tma_copy<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: &CopyArgs,
    func: &str,
) {
    match atom.op() {
        // Cta_group=CtaGroup::TWO loads for a CTA pair (`.cta_group::2`).
        CopyOp::TmaLoad { multicast, cta_group } => {
            tma_load(atom, src, dst, args, multicast, cta_group == CtaGroup::TWO, func)
        }
        CopyOp::TmaStore => tma_store(atom, src, dst, args, None, func),
        CopyOp::TmaReduce(kind) => tma_store(atom, src, dst, args, Some(kind), func),
        // m7-tma-im2col.
        CopyOp::TmaLoadMode { mode: crate::cpasync::TmaOpMode::Im2col, multicast, cta_group } => {
            tma_load(atom, src, dst, args, multicast, cta_group == CtaGroup::TWO, func)
        }
        CopyOp::TmaStoreMode { mode: crate::cpasync::TmaOpMode::Im2col } => tma_store(atom, src, dst, args, None, func),
        // Gather4, scatter4 and the overrides take auxiliary operands.
        op @ (CopyOp::TmaLoadMode { .. } | CopyOp::TmaStoreMode { .. }) => panic!(
            "{func}: the {op:?} atom takes an auxiliary operand (the index tensor of gather4 and scatter4, the residue \
             tensor of an override copy): copy with copy_aux(atom, (src, &[aux]), (dst, &[]), args), the partitions \
             of tma_partition_indexed or tma_partition_override"
        ),
        op => unreachable!("{op:?} is not a TMA operation"),
    }
}

/// A TMA load, plain or multicast; with `two_cta` for a CTA
/// pair, its bytes counted on the mbarrier of the pair's even CTA.
#[track_caller]
fn tma_load<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: &CopyArgs,
    multicast: bool,
    two_cta: bool,
    func: &str,
) {
    let loc = Location::caller();
    let what = if multicast { "a multicast TMA load" } else { "a TMA load" };
    if !S::COORDINATES {
        panic!(
            "{func}: {what} reads the box at the coordinates its source gives, so the source is the TMA tensor's \
             partition (the second result of tma_partition, a coordinate tensor), not a data tensor"
        );
    }
    if D::COORDINATES {
        panic!("{func}: {what} writes shared memory: the destination is the SMEM partition, not coordinates");
    }
    if args.pred.is_some() {
        panic!(
            "{func}: {what} is not predicated: out-of-bounds elements of the box are filled (with zeros) and still \
             counted in its transaction bytes, so copy whole boxes"
        );
    }
    let src: Tensor<Ident> = src.retyped();
    let dst: Tensor<T> = dst.retyped();
    // The padded formats write one byte slot per value (a u8 tensor).
    let padded = atom.tma_info().is_some_and(|i| i.data_type().is_padded());
    if padded != D::UNPACKED {
        panic!(
            "{func}: {}",
            if padded {
                "the atom's descriptor unpacks FP4/FP6 values into 16-byte slots of 16 values: its destination is a u8 \
                 shared-memory tensor with one element per value"
            } else {
                "a u8 destination stands for unpacked FP4/FP6 values, but the atom's descriptor does not unpack (make \
                 it with internal_type 8-bit)"
            }
        );
    }
    let elem = if padded || T::DTYPE.is_sub_byte() { DType::U8 } else { T::DTYPE };
    check_smem(atom, &dst, what, "destination", func);
    let mbar = args.tma_bar_ptr.as_ref().expect("checked: a TMA load needs tma_bar_ptr");
    if mbar.memory_space() != MemSpace::Shared {
        panic!("{func}: tma_bar_ptr points to {:?} memory; an mbarrier lives in shared memory", mbar.memory_space());
    }
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    let issues = boxes(atom, &src, &dst, what, func);
    let (dst_buffer, _) = dst.ptr().parts_unswizzled();
    // Everything the instructions take, computed before the election.
    let desc = match &args.tma_desc_ptr {
        Some(p) => p.checked(),
        None => info.descriptor().get_ptr().checked(),
    };
    let (mbar_buffer, mbar_offset) = mbar.parts();
    let mbar_index = index_operand(mbar_offset);
    let cache_policy = args.cache_policy.map(|p| p.checked());
    // A plain load ignores a mask, as CuTe DSL's does (one body serves both).
    let mcast_mask = if multicast {
        Some(args.mcast_mask.expect("checked: a multicast TMA load needs mcast_mask").checked())
    } else {
        None
    };
    // Each CTA of a multicast loads its share of the box (the descriptor's box).
    let box_bytes = (atom.num_val_src() / info.num_multicast() * T::DTYPE.bytes()) as u32;
    let issues: Vec<(ValueId, Vec<ValueId>, crate::ir::TmaMode)> = issues
        .into_iter()
        .map(|(o, c, m)| (index_operand(if padded { o } else { packed_index::<T>(o) }), c, m))
        .collect();
    let issue = || {
        for (dst_index, coords, mode) in issues {
            builder::with(|b| {
                b.push(
                    Op::TmaLoad {
                        dst: dst_buffer,
                        dst_index,
                        elem,
                        desc,
                        coords,
                        mbar: mbar_buffer,
                        mbar_index,
                        cache_policy,
                        box_bytes,
                        mcast_mask,
                        mode,
                        overrides: None,
                        two_cta,
                    },
                    loc,
                )
            });
        }
    };
    if_(crate::arch::elect_one(), issue);
}

/// A TMA store (`reduce` `None`) or reduce store.
#[track_caller]
fn tma_store<T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    atom: &CopyAtom<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: &CopyArgs,
    reduce: Option<ReductionKind>,
    func: &str,
) {
    let loc = Location::caller();
    let what = if reduce.is_some() { "a TMA reduce store" } else { "a TMA store" };
    if let Some(kind) = reduce
        && let Err(e) = tma_reduce_type_rule(kind, T::DTYPE)
    {
        panic!(
            "{func}: CopyReduceBulkTensorTileS2GOp(ReductionKind.{kind:?}) on {} elements: {e}",
            T::DTYPE.cute_name()
        );
    }
    if S::COORDINATES {
        panic!("{func}: {what} reads shared memory: the source is the SMEM partition, not coordinates");
    }
    if !D::COORDINATES {
        panic!(
            "{func}: {what} writes the box at the coordinates its destination gives, so the destination is the TMA \
             tensor's partition (the second result of tma_partition, a coordinate tensor), not a data tensor"
        );
    }
    if args.pred.is_some() {
        panic!(
            "{func}: {what} is not predicated: out-of-bounds elements of the box are not written (TMA clips them), \
             so copy whole boxes"
        );
    }
    let src: Tensor<T> = src.retyped();
    let dst: Tensor<Ident> = dst.retyped();
    check_smem(atom, &src, what, "source", func);
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    let issues = boxes(atom, &dst, &src, what, func);
    let (src_buffer, _) = src.ptr().parts_unswizzled();
    let desc = match &args.tma_desc_ptr {
        Some(p) => p.checked(),
        None => info.descriptor().get_ptr().checked(),
    };
    let cache_policy = args.cache_policy.map(|p| p.checked());
    let box_bytes = (atom.num_val_src() * T::DTYPE.bytes()) as u32;
    let issues: Vec<(ValueId, Vec<ValueId>, crate::ir::TmaMode)> =
        issues.into_iter().map(|(o, c, m)| (index_operand(packed_index::<T>(o)), c, m)).collect();
    let issue = || {
        for (src_index, coords, mode) in issues {
            builder::with(|b| {
                b.push(
                    Op::TmaStore {
                        src: src_buffer,
                        src_index,
                        elem: if T::DTYPE.is_sub_byte() { DType::U8 } else { T::DTYPE },
                        desc,
                        coords,
                        reduce,
                        cache_policy,
                        box_bytes,
                        mode,
                        overrides: None,
                    },
                    loc,
                )
            });
        }
    };
    if_(crate::arch::elect_one(), issue);
}

/// Prefetch into L2 what a copy with `atom` from `src` would read (CuTe
/// DSL's `cute.prefetch(atom, src)`): every box of a TMA atom's TMA-tensor
/// partition `src` (`cp.async.bulk.prefetch.tensor.{N}d.L2.global.tile`,
/// one elected lane of each warp that executes it issuing them all, as for
/// the copy), or with a non-tensor bulk-copy atom (`CopyBulkG2SOp`, ...)
/// each bulk copy's source in the global tensor `src`
/// (`cp.async.bulk.prefetch.L2.global`, issued by every thread that
/// executes it: elect one). `src` is the copy's own source, rest modes
/// (stages, k tiles) included, so one call outside the stage loop
/// prefetches them all. A hint: nothing is read into shared memory, and no
/// mbarrier or mask is involved. An im2col atom prefetches in `.im2col` mode;
/// gather4 atoms prefetch through [`prefetch_atom_aux`](crate::cpasync::prefetch_atom_aux).
#[track_caller]
pub fn prefetch_atom<T: Numeric, S: CopyElem<T>>(atom: &CopyAtom<T>, src: &Tensor<S>) {
    let loc = Location::caller();
    let func = "prefetch_atom";
    match atom.op() {
        CopyOp::TmaLoad { .. } | CopyOp::TmaLoadMode { mode: crate::cpasync::TmaOpMode::Im2col, .. } => {}
        CopyOp::BulkG2S { bits, .. } => {
            if S::COORDINATES {
                panic!("{func}: a bulk copy's prefetch reads the global source tensor, not coordinates");
            }
            let src: Tensor<T> = src.retyped();
            crate::atom::copy::bulk_prefetch(&src, bits, loc, func);
            return;
        }
        op @ CopyOp::TmaLoadMode { mode: crate::cpasync::TmaOpMode::Rows4, .. } => panic!(
            "{func}: a gather4 atom's prefetch takes its index tensor too: prefetch_atom_aux(&atom, &tAgA, &tAgI) \
             ({op:?})"
        ),
        op => panic!(
            "{func}: cute.prefetch takes TMA load atoms (tiled or im2col; gather4 through prefetch_atom_aux) and \
             non-tensor bulk-copy atoms, not {op:?}"
        ),
    }
    if !S::COORDINATES {
        panic!(
            "{func}: a TMA prefetch reads the boxes at the coordinates its source gives, so the source is the TMA \
             tensor's partition (the second result of tma_partition), not a data tensor"
        );
    }
    let src: Tensor<Ident> = src.retyped();
    let info = atom.tma_info().expect("a TMA atom has its descriptor");
    // The boxes, as a copy into a plain tile of the source's shape would issue them.
    let canon = src.layout().pad_to_rank(2);
    let canon = if canon.rank() == 2 {
        canon
    } else {
        Layout::from_modes([canon.mode(0), Layout::from_modes((1..canon.rank()).map(|i| canon.mode(i)))])
    };
    if let Some((path, leaf)) = canon.shape().first_dynamic() {
        panic!("{func}: the TMA tensor's partition {canon} has the dynamic extent {leaf:?} at mode {path:?}");
    }
    let v = atom.num_val_src();
    let (c0, cr) = (canon.mode(0).size(), canon.mode(1).size());
    if !c0.is_multiple_of(v) {
        panic!(
            "{func}: mode 0 of the TMA tensor's partition {canon} is not a whole number of the atom's {v}-element boxes"
        );
    }
    let desc = info.descriptor().get_ptr().checked();
    let n0 = c0 / v;
    let mut issues: Vec<(Vec<ValueId>, crate::ir::TmaMode)> = Vec::new();
    for a in 0..n0 * cr {
        let logical = (a % n0) * v + (a / n0) * c0;
        let coord = src.coord(IndexExpr::Static(logical));
        let leaves: Vec<IndexExpr> = coord.leaves().into_iter().cloned().collect();
        issues.push(tma_operands(info, leaves, false, &format!("{coord:?}"), func));
    }
    if_(crate::arch::elect_one(), || {
        for (coords, mode) in issues {
            builder::with(|b| b.push(Op::TmaPrefetch { desc, coords, mode }, loc));
        }
    });
}

/// Prefetch a TMA atom's descriptor (CuTe DSL's
/// `cpasync.prefetch_descriptor(tma_atom)`, `prefetch.tensormap` on the
/// generic address of its grid-constant parameter; sm_90). CuTe DSL's GEMMs
/// call it for each TMA atom from one warp at kernel start. Every thread
/// that executes it issues it.
#[track_caller]
pub fn prefetch_descriptor<T: Numeric>(atom: &CopyAtom<T>) {
    let loc = Location::caller();
    let Some(info) = atom.tma_info() else {
        panic!("prefetch_descriptor: {:?} is not a TMA atom (make one with cpasync::make_tiled_tma_atom)", atom.op())
    };
    let desc = info.descriptor().get_ptr().checked();
    builder::with(|b| b.push(Op::PrefetchTensormap { desc }, loc));
}

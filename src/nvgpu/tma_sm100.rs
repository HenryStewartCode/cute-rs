// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SM100 TMA modes with auxiliary operands:
//! gather4 loads and scatter4 stores, which take an index tensor, and the override
//! copies, which take the global tensor's tile and a residue tensor.
//!
//! A gather4 atom ([`make_tiled_tma_atom_with`](crate::cpasync::make_tiled_tma_atom_with)
//! with [`TmaAtomOptions::gmem_coord_tensor`](crate::cpasync::TmaAtomOptions)) copies a
//! tile of a 2-D tensor whose rows (the tensor's non-contiguous mode, the gather mode)
//! come from an `Int32` index tensor of the tensor's shape, broadcast (stride 0) along
//! the contiguous mode: each instruction moves four rows, `box_dims[0]` elements from
//! column `x` of rows `index[r..r + 4]`. [`tma_partition_indexed`] partitions the index
//! tensor as the TMA tensor, and [`copy_aux`](crate::copy_aux) issues the instructions,
//! the elected lane reading the four indices of each.
//!
//! An override atom ([`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom) on an
//! override operation; sm_107f) issues each box with the global address of its first
//! element, the remaining extents (from a residue tensor: a coordinate tensor of the
//! tensor's shape whose strides are negated, so that its element at a box's start is
//! what remains of each mode, clamped to the box) and the tensor's byte strides, at
//! coordinates zero: [`tma_partition_override`] partitions the global tile and the
//! residue, and [`copy_aux`](crate::copy_aux) copies.

use std::panic::Location;

use crate::atom::copy::{CopyArgs, CopyAtom, CopyOp};
use crate::builder::{self, user_panic};
use crate::control::if_;
use crate::ir::{DType, MemSpace, Op, TmaMode, TmaOverride, ValueId};
use crate::layout::{Coord, IndexExpr, Layout, StrideLeaf};
use crate::nvgpu::tma_atom::{TmaAtomMode, TmaPlan};
use crate::tensor::{Ident, Tensor, index_operand};
use crate::value::{Int32, IntoValue, Numeric, select};

/// Check a gather4/scatter4 atom's tensor and index tensor, and turn the tile plan's box
/// into four-row boxes: returns the gather mode (the tensor's mode the index tensor
/// gives rows of).
pub(crate) fn check_rows4(glayout: &Layout, index: &Layout, plan: &mut TmaPlan) -> Result<usize, String> {
    let gshape = glayout.shape().flat_leaves();
    let istride = index.stride().flat_leaves();
    let ishape = index.shape().flat_leaves();
    if gshape.len() != 2 || ishape.len() != 2 {
        return Err(format!(
            "the gather4 and scatter4 modes copy 2-D tensors with a 2-D index tensor, not {glayout} and {index}"
        ));
    }
    let zero: Vec<bool> = istride.iter().map(|s| matches!(s, StrideLeaf::Int(e) if e.as_static() == Some(0))).collect();
    if zero.iter().filter(|z| **z).count() != 1 {
        return Err(format!(
            "the index tensor {index} needs exactly one broadcast mode (stride 0), the tensor's contiguous one; the \
             other gives the rows"
        ));
    }
    let gather = if zero[0] { 1 } else { 0 };
    if let Some(n) = ishape[gather].as_static()
        && n < 4
    {
        return Err(format!(
            "the index tensor {index}'s gather mode (mode {gather}) has {n} indices; an instruction gathers four rows"
        ));
    }
    if plan.box_dims.len() != 2 || plan.dim_modes[0] != [1 - gather] || plan.dim_modes[1] != [gather] {
        return Err(format!(
            "the index tensor broadcasts mode {} and gathers mode {gather}, but the tensor's contiguous mode (TMA's \
             dimension 0) is {:?}: broadcast the contiguous mode",
            1 - gather,
            plan.dim_modes.first()
        ));
    }
    let rows = plan.box_dims[1];
    if !rows.is_multiple_of(4) {
        return Err(format!("the tile has {rows} rows, not a multiple of the four an instruction moves"));
    }
    plan.box_dims[1] = 1;
    plan.box_elements = 4 * plan.box_dims[0] as usize;
    Ok(gather)
}

/// Partition a gather4/scatter4 atom's shared-memory tile, TMA-tensor tile and index
/// tensor tile (CuTe DSL's `cpasync.tma_partition(atom, cta_coord, cta_layout, smem,
/// [coord_tensor, index_tensor])`), returning `(tXsX, tXgX, tXgI)`: the index tensor is
/// partitioned as the TMA tensor is, so that a box's four rows are its elements at
/// `0`, `box_dims[0]`, `2 * box_dims[0]` and `3 * box_dims[0]` in mode 0.
#[track_caller]
pub fn tma_partition_indexed<T: Numeric, G>(
    atom: &CopyAtom<T>,
    cta_coord: impl Into<Coord>,
    cta_layout: &Layout,
    smem_tensor: &Tensor<T>,
    coord_tensor: &Tensor<G>,
    index_tensor: &Tensor<i32>,
) -> (Tensor<T>, Tensor<G>, Tensor<i32>) {
    let coord: Coord = cta_coord.into();
    if !atom.tma_info().is_some_and(|i| matches!(i.mode, TmaAtomMode::Rows4 { .. })) {
        user_panic(
            Location::caller(),
            format!("cute: tma_partition with an index tensor takes a gather4/scatter4 atom, not {:?}", atom.op()),
        );
    }
    let (s, g) = crate::cpasync::tma_partition(atom, coord.clone(), cta_layout, smem_tensor, coord_tensor);
    let (_, i) = crate::cpasync::tma_partition(atom, coord, cta_layout, smem_tensor, index_tensor);
    (s, g, i)
}

/// Partition an override atom's shared-memory tile, global tile and residue tile (CuTe
/// DSL's `cpasync.tma_partition` with `(gmem_tensor, residue_tensor)`), returning
/// `(tXsX, tXgX, tXrX)`. The global tile is the tensor's own (in memory), the residue a
/// coordinate tensor over its shape whose element at each box's start is what remains
/// of each mode from there (a coordinate tensor of origin the extents with negated
/// strides, tiled as the tensor is).
#[track_caller]
pub fn tma_partition_override<T: Numeric>(
    atom: &CopyAtom<T>,
    cta_coord: impl Into<Coord>,
    cta_layout: &Layout,
    smem_tensor: &Tensor<T>,
    gmem_tensor: &Tensor<T>,
    residue_tensor: &Tensor<Ident>,
) -> (Tensor<T>, Tensor<T>, Tensor<Ident>) {
    let coord: Coord = cta_coord.into();
    if !atom.tma_info().is_some_and(|i| i.mode == TmaAtomMode::Override) {
        user_panic(
            Location::caller(),
            format!("cute: tma_partition with a residue tensor takes an override atom, not {:?}", atom.op()),
        );
    }
    let (s, g) = crate::cpasync::tma_partition(atom, coord.clone(), cta_layout, smem_tensor, gmem_tensor);
    let (_, r) = crate::cpasync::tma_partition(atom, coord, cta_layout, smem_tensor, residue_tensor);
    (s, g, r)
}

/// A layout in `cute.copy`'s canonical form, `(V, REST)`.
fn canon(l: &Layout) -> Layout {
    let l = l.pad_to_rank(2);
    if l.rank() == 2 {
        l
    } else {
        Layout::from_modes([l.mode(0), Layout::from_modes((1..l.rank()).map(|i| l.mode(i)))])
    }
}

/// The issuing data of a four-row or override copy: per box, the shared-memory index and
/// the instruction's coordinates and overrides.
type Issue = (ValueId, Vec<ValueId>, Option<TmaOverride>);

/// [`copy_aux`](crate::copy_aux) with a gather4 or scatter4 atom: `gmem` is the TMA
/// tensor's partition and `index` the index tensor's, `smem` the shared-memory one.
#[track_caller]
pub(crate) fn rows4_copy<T: Numeric>(
    atom: &CopyAtom<T>,
    gmem: &Tensor<Ident>,
    index: &Tensor<i32>,
    smem: &Tensor<T>,
    args: &CopyArgs,
    load: bool,
    func: &str,
) {
    let info = atom.tma_info().expect("a TMA atom");
    let row = info.spec().box_dims[0] as usize;
    let leaves = crate::nvgpu::tma_copy::box_leaves(atom, gmem, smem, "a gather4/scatter4 copy", func);
    let ic = canon(index.layout());
    let gc = canon(gmem.layout());
    if ic.mode(0).size() != gc.mode(0).size() || ic.mode(1).size() != gc.mode(1).size() {
        panic!(
            "{func}: the index tensor's partition {} does not match the TMA tensor's {}: partition both with \
             tma_partition_indexed",
            index.layout(),
            gmem.layout()
        );
    }
    let it = crate::layout::Tileable::with_layout(index, ic);
    let (v, c0) = (atom.num_val_src(), gc.mode(0).size());
    let n0 = c0 / v;
    let mut issues: Vec<Issue> = Vec::new();
    for (a, (offset, leaves, _)) in leaves.into_iter().enumerate() {
        let logical = (a % n0) * v + (a / n0) * c0;
        let x = IntoValue::<i32>::into_value(leaves.first().cloned().unwrap_or(IndexExpr::Static(0))).checked();
        let mut coords = vec![x];
        for j in 0..4 {
            coords.push(it.load(IndexExpr::Static(logical + j * row)).checked());
        }
        issues.push((index_operand(offset), coords, None));
    }
    let mode = if load { TmaMode::Gather4 } else { TmaMode::Scatter4 };
    issue(atom, smem, args, issues, mode, load, func);
}

/// [`copy_aux`](crate::copy_aux) with an override atom: `gmem` is the global tile's
/// partition (memory), `residue` the residue's, `smem` the shared-memory one.
#[track_caller]
pub(crate) fn override_copy<T: Numeric>(
    atom: &CopyAtom<T>,
    gmem: &Tensor<T>,
    residue: &Tensor<Ident>,
    smem: &Tensor<T>,
    args: &CopyArgs,
    load: bool,
    func: &str,
) {
    let loc = Location::caller();
    let info = atom.tma_info().expect("a TMA atom");
    let spec = info.spec().clone();
    if gmem.memory_space() != Some(MemSpace::Global) {
        panic!("{func}: an override copy takes the global tensor's tile (in memory) as its global operand");
    }
    let leaves = crate::nvgpu::tma_copy::box_leaves(atom, residue, smem, "an override copy", func);
    let gc = canon(gmem.layout());
    let (buffer, _) = gmem.ptr().parts_unswizzled();
    let (v, c0) = (atom.num_val_src(), gc.mode(0).size());
    let n0 = c0 / v;
    let rank = spec.global_dims.len();
    // The byte strides: lower 32 bits (after the 16-byte scale) and the packed 4-bit uppers.
    let strides: Vec<IndexExpr> = spec.global_strides.clone();
    let lower: Vec<ValueId> = strides
        .iter()
        .map(|s| {
            let l =
                IndexExpr::and(IndexExpr::shift_right(s.clone(), IndexExpr::Static(4)), IndexExpr::Static(0xffff_ffff));
            IntoValue::<i64>::into_value(l).to::<i32>().checked()
        })
        .collect();
    let upper = (rank > 1).then(|| {
        let mut u = IndexExpr::Static(0);
        for (k, s) in strides.iter().enumerate() {
            let nib = IndexExpr::and(IndexExpr::shift_right(s.clone(), IndexExpr::Static(36)), IndexExpr::Static(0xf));
            u = IndexExpr::sum(u, IndexExpr::product(nib, IndexExpr::Static(1 << (4 * k))));
        }
        IntoValue::<i32>::into_value(u).to::<i16>().checked()
    });
    let zero = IntoValue::<i32>::into_value(IndexExpr::Static(0)).checked();
    let mut issues: Vec<Issue> = Vec::new();
    for (a, (offset, rleaves, coord)) in leaves.into_iter().enumerate() {
        let logical = (a % n0) * v + (a / n0) * c0;
        let goff = gmem.base_offset().clone() + gc.offset_at(logical);
        let index = index_operand(goff);
        let address = builder::with(|b| {
            let result = b.new_value(DType::I64);
            b.push(Op::PtrToInt { result, buffer, index, elem: T::DTYPE }, loc);
            result
        });
        if rleaves.len() < info.flat_modes {
            panic!("{func}: the residue's coordinate {coord:?} has fewer entries than the tensor's modes");
        }
        let dims = info
            .dim_modes
            .iter()
            .enumerate()
            .map(|(j, m)| {
                let r = Int32::from(rleaves[m[0]].clone());
                let b = spec.box_dims[j] as i32;
                let clamped: Int32 = select(r.lt(0), 0, r.min(b));
                clamped.to::<i16>().checked()
            })
            .collect();
        let over = TmaOverride { global_address: address, dims, lower_strides: lower.clone(), upper_strides: upper };
        let over =
            if rank == 1 { TmaOverride { lower_strides: Vec::new(), upper_strides: None, ..over } } else { over };
        issues.push((index_operand(offset), vec![zero; rank], Some(over)));
    }
    issue(atom, smem, args, issues, TmaMode::Tile, load, func);
}

/// Issue the boxes from the elected lane of each warp that executes the copy.
#[track_caller]
fn issue<T: Numeric>(
    atom: &CopyAtom<T>,
    smem: &Tensor<T>,
    args: &CopyArgs,
    issues: Vec<Issue>,
    mode: TmaMode,
    load: bool,
    func: &str,
) {
    let loc = Location::caller();
    let info = atom.tma_info().expect("a TMA atom");
    if smem.memory_space() != Some(MemSpace::Shared) {
        panic!("{func}: a TMA copy's shared-memory operand is {}", smem.memspace());
    }
    let (sbuf, _) = smem.ptr().parts_unswizzled();
    let desc = match &args.tma_desc_ptr {
        Some(p) => p.checked(),
        None => info.descriptor().get_ptr().checked(),
    };
    let cache_policy = args.cache_policy.map(|p| p.checked());
    let box_bytes = (atom.num_val_src() / info.num_multicast() * T::DTYPE.bytes()) as u32;
    let multicast = matches!(atom.op(), CopyOp::TmaLoadMode { multicast: true, .. });
    let mbar = if load {
        let m = args.tma_bar_ptr.as_ref().expect("checked: a TMA load needs tma_bar_ptr");
        let (b, o) = m.parts();
        Some((b, index_operand(o)))
    } else {
        None
    };
    let mcast_mask = if multicast { args.mcast_mask.map(|m| m.checked()) } else { None };
    if_(crate::arch::elect_one(), || {
        for (index, coords, overrides) in issues {
            let mode = mode.clone();
            let op = match mbar {
                Some((mbar, mbar_index)) => Op::TmaLoad {
                    dst: sbuf,
                    dst_index: index,
                    elem: T::DTYPE,
                    desc,
                    coords,
                    mbar,
                    mbar_index,
                    cache_policy,
                    box_bytes,
                    mcast_mask,
                    two_cta: false,
                    mode,
                    overrides,
                },
                None => Op::TmaStore {
                    src: sbuf,
                    src_index: index,
                    elem: T::DTYPE,
                    desc,
                    coords,
                    reduce: None,
                    cache_policy,
                    box_bytes,
                    mode,
                    overrides,
                },
            };
            builder::with(|b| b.push(op, loc));
        }
    });
}

/// Prefetch into L2 the rows a gather4 copy from `src` (the TMA tensor's partition) with
/// the index tensor's partition `index` would read (CuTe DSL's `cute.prefetch(atom,
/// [src, index])`, `cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4`, sm_100a),
/// from the elected lane of each warp that executes it.
#[track_caller]
pub fn prefetch_atom_aux<T: Numeric>(atom: &CopyAtom<T>, src: &Tensor<Ident>, index: &Tensor<i32>) {
    let loc = Location::caller();
    let func = "prefetch_atom_aux";
    let Some(info) = atom.tma_info().filter(|i| matches!(i.mode, TmaAtomMode::Rows4 { .. })) else {
        panic!("{func}: takes a gather4 atom, not {:?}", atom.op())
    };
    if !atom.op().is_tma_load() {
        panic!("{func}: a scatter4 atom stores; only gather4 loads prefetch");
    }
    let row = info.spec().box_dims[0] as usize;
    let sc = canon(src.layout());
    let it = crate::layout::Tileable::with_layout(index, canon(index.layout()));
    let (v, c0, cr) = (atom.num_val_src(), sc.mode(0).size(), sc.mode(1).size());
    let n0 = c0 / v;
    let desc = info.descriptor().get_ptr().checked();
    let mut issues = Vec::new();
    for a in 0..n0 * cr {
        let logical = (a % n0) * v + (a / n0) * c0;
        let coord = src.coord(IndexExpr::Static(logical));
        let x = coord.leaves().first().map(|c| (*c).clone()).unwrap_or(IndexExpr::Static(0));
        let mut coords = vec![IntoValue::<i32>::into_value(x).checked()];
        for j in 0..4 {
            coords.push(it.load(IndexExpr::Static(logical + j * row)).checked());
        }
        issues.push(coords);
    }
    if_(crate::arch::elect_one(), || {
        for coords in issues {
            builder::with(|b| b.push(Op::TmaPrefetch { desc, coords, mode: TmaMode::Gather4 }, loc));
        }
    });
}

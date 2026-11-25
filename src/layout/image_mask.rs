// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layout image masks: the bit mask of the
//! offsets a layout's mode reaches from a coordinate, as the multicast masks
//! of cluster copies use ([`make_layout_image_mask`],
//! [`create_tma_multicast_mask`]).

use std::panic::Location;

use super::{Coord, IndexExpr, Layout, SliceCoord, Tree};
use crate::value::{Int16, Int32, IntoValue};

/// CuTe DSL's `cute.make_layout_image_mask(lay, coord, mode)`: the 16-bit
/// mask with bit `lay(c)` set for every coordinate `c` that agrees with
/// `coord` except in mode `mode`, which runs over its whole extent. `coord`
/// has `lay`'s rank (its entry for `mode` is ignored) and may hold runtime
/// values; a static layout's cosize must be at most 16. The layout is sliced
/// at `coord` with `mode` kept (`slice_and_offset`): the kept mode's offsets
/// give the mask's bits, and the slice's offset shifts it. With a static
/// layout the bits are known when the kernel is built, so only the shift is
/// computed at run time; a layout with runtime strides sets each bit at run
/// time. With a cluster's CTA layout and a CTA's coordinate in it, this is
/// the mask of the CTAs that share the CTA's coordinates outside `mode`
/// ([`create_tma_multicast_mask`]).
#[track_caller]
pub fn make_layout_image_mask(lay: &Layout, coord: impl Into<Coord>, mode: usize) -> Int16 {
    let loc = Location::caller();
    let coord: Coord = coord.into();
    let rank = lay.rank();
    let entries: Vec<IndexExpr> = match &coord {
        Tree::Leaf(c) => vec![c.clone()],
        Tree::Branch(modes) => modes
            .iter()
            .map(|m| match m {
                Tree::Leaf(c) => c.clone(),
                _ => crate::builder::user_panic(
                    loc,
                    format!(
                        "cute: make_layout_image_mask: the coordinate {coord} has a nested entry; give one integer per \
                         mode of the layout {lay}"
                    ),
                ),
            })
            .collect(),
    };
    if entries.len() != rank {
        crate::builder::user_panic(
            loc,
            format!(
                "cute: make_layout_image_mask: the coordinate {coord} has rank {}, the layout {lay} rank {rank}: they \
                 must be equal",
                entries.len()
            ),
        );
    }
    if mode >= rank {
        crate::builder::user_panic(
            loc,
            format!("cute: make_layout_image_mask: mode {mode} is not a mode of the rank-{rank} layout {lay}"),
        );
    }
    if lay.is_static() && lay.cosize() > 16 {
        crate::builder::user_panic(
            loc,
            format!(
                "cute: make_layout_image_mask: the layout {lay} has cosize {}, but the mask has 16 bits",
                lay.cosize()
            ),
        );
    }
    let slice: SliceCoord = if rank == 1 {
        Tree::Leaf(None)
    } else {
        Tree::Branch(
            entries
                .iter()
                .enumerate()
                .map(|(i, c)| Tree::Leaf(if i == mode { None } else { Some(c.clone()) }))
                .collect(),
        )
    };
    let (sliced, offset) = lay.slice(&slice);
    let static_bits: Option<u32> = sliced
        .is_static()
        .then(|| (0..sliced.size()).fold(0u32, |m, i| m | (1u32 << sliced.offset_at(i).as_static().expect("static"))));
    let bits: Int32 = match static_bits {
        Some(b) => (b as i32).into_value(),
        None => {
            // Runtime strides (or extent): set each bit at run time.
            let n = sliced.shape().product();
            let sliced2 = sliced.clone();
            crate::control::for_range_carry(0, n, 1, 0i32.into_value(), move |i, acc: Int32| {
                let at: Int32 = IntoValue::<i32>::into_value(sliced2.offset(&Tree::Leaf(IndexExpr::from(i))));
                acc | (Int32::from(1) << at)
            })
        }
    };
    let shifted: Int32 = match (static_bits, offset.as_static()) {
        (Some(b), Some(o)) => ((b << o) as i32).into_value(),
        _ => bits << IntoValue::<i32>::into_value(offset),
    };
    shifted.to::<i16>()
}

/// CuTe DSL's `cpasync.create_tma_multicast_mask(cta_layout_vmnk,
/// cta_coord_vmnk, mcast_mode)` (C++ CuTe's `create_tma_multicast_mask`): the
/// CTAs of the cluster that share this CTA's coordinates outside mode
/// `mcast_mode`, as a mask of their ranks in the cluster, for
/// [`CopyArgs::mcast_mask`](crate::CopyArgs::mcast_mask).
/// `cta_layout_vmnk` maps a CTA's `(V, M, N, K)` coordinate to its rank in
/// the cluster and `cta_coord_vmnk` is this CTA's coordinate (both rank 4);
/// it is [`make_layout_image_mask`].
#[track_caller]
pub fn create_tma_multicast_mask(
    cta_layout_vmnk: &Layout,
    cta_coord_vmnk: impl Into<Coord>,
    mcast_mode: usize,
) -> Int16 {
    let loc = Location::caller();
    let coord: Coord = cta_coord_vmnk.into();
    let coord_rank = match &coord {
        Tree::Leaf(_) => 1,
        Tree::Branch(m) => m.len(),
    };
    if cta_layout_vmnk.rank() != 4 || coord_rank != 4 {
        crate::builder::user_panic(
            loc,
            format!(
                "cute: create_tma_multicast_mask: the CTA layout {cta_layout_vmnk} and coordinate {coord} are \
                 (V, M, N, K): both rank 4"
            ),
        );
    }
    make_layout_image_mask(cta_layout_vmnk, coord, mcast_mode)
}

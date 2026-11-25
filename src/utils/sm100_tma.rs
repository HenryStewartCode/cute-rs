// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TMA load operation of a Blackwell GEMM operand for a cluster shape
//! and an MMA (CuTe DSL's
//! `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A`, `_B` and
//! `_SFB`): whether the load multicasts and whether it serves a CTA pair.
//! Plain host code, run when a kernel is configured.

use crate::atom::copy::CopyOp;
use crate::layout::{Layout, Shape};
use crate::nvgpu::cpasync::{CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp, CtaGroup};

/// The TMA load chosen for an operand: a plain tile load or its multicast
/// form, each with the MMA's CTA group. Converts into a [`CopyOp`] for
/// [`make_tiled_tma_atom_a`](crate::nvgpu::make_tiled_tma_atom_a) and its
/// kin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sm100TmaLoadOp {
    /// `CopyBulkTensorTileG2SOp(cta_group)`.
    Tile(CopyBulkTensorTileG2SOp),
    /// `CopyBulkTensorTileG2SMulticastOp(cta_group)`.
    Multicast(CopyBulkTensorTileG2SMulticastOp),
}

impl Sm100TmaLoadOp {
    fn new(multicast: bool, cta_group: CtaGroup) -> Sm100TmaLoadOp {
        if multicast {
            Sm100TmaLoadOp::Multicast(CopyBulkTensorTileG2SMulticastOp { cta_group })
        } else {
            Sm100TmaLoadOp::Tile(CopyBulkTensorTileG2SOp { cta_group })
        }
    }

    /// Whether it is the multicast form.
    pub fn is_multicast(self) -> bool {
        matches!(self, Sm100TmaLoadOp::Multicast(_))
    }

    /// The CTA group: [`CtaGroup::TWO`] for a CTA pair's MMA.
    pub fn cta_group(self) -> CtaGroup {
        match self {
            Sm100TmaLoadOp::Tile(op) => op.cta_group,
            Sm100TmaLoadOp::Multicast(op) => op.cta_group,
        }
    }
}

impl From<Sm100TmaLoadOp> for CopyOp {
    fn from(op: Sm100TmaLoadOp) -> CopyOp {
        match op {
            Sm100TmaLoadOp::Tile(op) => op.into(),
            Sm100TmaLoadOp::Multicast(op) => op.into(),
        }
    }
}

/// Which operand the load is for.
#[derive(Clone, Copy)]
enum Which {
    A,
    B,
    Sfb,
}

/// The static size of mode `i` of a shape, or of the whole shape.
fn static_size(shape: &Shape, mode: Option<usize>) -> Option<usize> {
    let part = match mode {
        Some(i) => shape.modes().get(i)?.clone(),
        None => shape.clone(),
    };
    part.flat_leaves().iter().map(|e| e.as_static()).product()
}

fn select(which: Which, cluster_shape_mnk: Shape, atom_thr_id: &Layout) -> Result<Sm100TmaLoadOp, String> {
    let shape = &cluster_shape_mnk;
    let sms = atom_thr_id.shape().product().as_static();
    let (Some(sms), Some(_)) = (sms, static_size(shape, None)) else {
        return Err(format!(
            "the cluster shape {shape} and the MMA's thr_id {atom_thr_id} must be static to pick a TMA load"
        ));
    };
    let mode = |i: usize| {
        static_size(shape, Some(i)).ok_or_else(|| format!("the cluster shape {shape} has no static mode {i}"))
    };
    let cluster_m = mode(0)?;
    if sms == 0 || cluster_m % sms != 0 {
        return Err(format!(
            "the cluster's M extent ({cluster_m}, of {shape}) is not a multiple of the MMA's {sms} SM(s) ({atom_thr_id})"
        ));
    }
    let group = match sms {
        1 => CtaGroup::ONE,
        2 => CtaGroup::TWO,
        n => return Err(format!("an SM100 MMA spans 1 or 2 SMs, not {n} (thr_id {atom_thr_id})")),
    };
    let multicast = match which {
        // A is shared by the CTAs along N, B by the MMAs along M.
        Which::A => mode(1)? != 1,
        Which::B => cluster_m != sms,
        // B's scale factors: every CTA pair multicasts them.
        Which::Sfb => sms == 2 || cluster_m != 1,
    };
    Ok(Sm100TmaLoadOp::new(multicast, group))
}

/// The TMA load for the A operand (CuTe DSL's
/// `cluster_shape_to_tma_atom_A(cluster_shape_mnk, atom_thr_id)`): the CTA
/// group from the size of the MMA's `thr_id` (1 or 2 SMs), multicast when
/// the cluster's N extent is above 1 (the CTAs along N share an A tile).
///
/// Errs for a dynamic cluster shape or `thr_id`, an MMA of another SM count,
/// and a cluster whose M extent is not a multiple of the SM count.
pub fn cluster_shape_to_tma_atom_a(
    cluster_shape_mnk: impl Into<Shape>,
    atom_thr_id: &Layout,
) -> Result<Sm100TmaLoadOp, String> {
    select(Which::A, cluster_shape_mnk.into(), atom_thr_id)
}

/// The TMA load for the B operand (CuTe DSL's `cluster_shape_to_tma_atom_B`):
/// as [`cluster_shape_to_tma_atom_a`], but multicast when the cluster's M
/// extent differs from the SM count (several MMAs along M share a B tile).
pub fn cluster_shape_to_tma_atom_b(
    cluster_shape_mnk: impl Into<Shape>,
    atom_thr_id: &Layout,
) -> Result<Sm100TmaLoadOp, String> {
    select(Which::B, cluster_shape_mnk.into(), atom_thr_id)
}

/// The TMA load for B's scale factors in a block-scaled GEMM (CuTe DSL's
/// `cluster_shape_to_tma_atom_SFB`): always multicast for a 2-SM MMA;
/// for a 1-SM MMA, multicast when the cluster's M extent is above 1.
pub fn cluster_shape_to_tma_atom_sfb(
    cluster_shape_mnk: impl Into<Shape>,
    atom_thr_id: &Layout,
) -> Result<Sm100TmaLoadOp, String> {
    select(Which::Sfb, cluster_shape_mnk.into(), atom_thr_id)
}

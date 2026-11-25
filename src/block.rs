// SPDX-License-Identifier: MIT OR Apache-2.0

//! CuTe DSL's block API, `cutlass.block` (roadmap part m10-s2t-copies):
//! [`block_copy`], one call for a CTA's whole TMA tile or S2T copy, which
//! partitions the tile-shaped operands itself.

use crate::atom::copy::{AsCopyAtom, CopyArgs, CopyAtom, CopyElem, CopyOp, TiledCopy, copy_with};
use crate::layout::{Layout, Tileable};
use crate::tensor::Tensor;
use crate::value::Numeric;

/// What [`block_copy`] takes as its copy: a TMA atom
/// ([`make_tiled_tma_atom`](crate::cpasync::make_tiled_tma_atom)) or an S2T
/// tiled copy ([`make_s2t_copy`](crate::tcgen05::make_s2t_copy)).
pub trait BlockCopy<T>: AsCopyAtom<T> {
    /// The tiled copy, when this is one.
    #[doc(hidden)]
    fn tiled(&self) -> Option<&TiledCopy<T>>;
}

impl<T> BlockCopy<T> for CopyAtom<T> {
    fn tiled(&self) -> Option<&TiledCopy<T>> {
        None
    }
}

impl<T> BlockCopy<T> for TiledCopy<T> {
    fn tiled(&self) -> Option<&TiledCopy<T>> {
        Some(self)
    }
}

fn filter_zeros<X>(t: &Tensor<X>) -> Tensor<X> {
    t.with_layout(t.layout().filter_zeros())
}

/// Copy a CTA's tile with one call (CuTe DSL's `cutlass.block.block_copy`):
///
/// * With a TMA atom, `src` and `dst` are the tiles whose mode 0 is the
///   atom's TMA tile (the global side the TMA tensor's coordinate tile, the
///   other the shared-memory tile): they are partitioned with
///   [`tma_partition`](crate::cpasync::tma_partition) for a single CTA
///   (`cta_coord` 0, `cta_layout` `1:0`), zero-stride modes filtered out,
///   and copied with `args` (a TMA load needs `tma_bar_ptr`).
/// * With an S2T tiled copy, `src` is the shared-memory tensor and `dst`
///   the TMEM tensor: both are filtered of zero-stride modes (for a
///   broadcast copy the source then gets its broadcast mode back, as
///   [`append_s2t_broadcast_mode`](crate::tcgen05::append_s2t_broadcast_mode)
///   adds it, when its mode 0 is `(MN, K)`), partitioned by the copy's slice
///   0, the source turned into descriptors
///   ([`get_s2t_smem_desc_tensor`](crate::tcgen05::get_s2t_smem_desc_tensor))
///   and copied (a converged warp calls it; one lane issues each copy).
///
/// `tma_multicast` (the block API's multicast derived from a cluster
/// shape) is not supported yet.
#[track_caller]
pub fn block_copy<'a, T: Numeric, S: CopyElem<T>, D: CopyElem<T>>(
    tiled_copy: &impl BlockCopy<T>,
    src: &Tensor<S>,
    dst: &Tensor<D>,
    args: impl Into<CopyArgs<'a>>,
) {
    let atom = tiled_copy.copy_atom();
    let args = args.into();
    match atom.op() {
        op if op.is_tma() => {
            let cta_layout = Layout::new(1, 0);
            if op.is_tma_load() {
                let (s, g) = crate::cpasync::tma_partition(atom, 0, &cta_layout, dst, src);
                copy_with(atom, &filter_zeros(&g), &filter_zeros(&s), args);
            } else {
                let (s, g) = crate::cpasync::tma_partition(atom, 0, &cta_layout, src, dst);
                copy_with(atom, &filter_zeros(&s), &filter_zeros(&g), args);
            }
        }
        CopyOp::TmemCp { .. } => {
            let Some(tiled) = tiled_copy.tiled() else {
                panic!(
                    "cute: block_copy: an S2T copy is given as its tiled copy (tcgen05::make_s2t_copy over the TMEM \
                     tensor), not the bare atom {:?}",
                    atom.op()
                )
            };
            let thr = tiled.get_slice(0);
            // The broadcast mode, which filtering would remove, goes in after it.
            let mut s = filter_zeros(&src.retyped::<T>());
            let l = s.layout();
            if atom.op().s2t_broadcast_factor() > Some(1) && l.rank() >= 2 && l.mode(0).rank() == 2 {
                s = crate::tcgen05::append_s2t_broadcast_mode(atom.op(), &s);
            }
            let s = thr.partition_s(&s);
            let d = thr.partition_d(&filter_zeros(&dst.retyped::<T>()));
            let desc = crate::tcgen05::get_s2t_smem_desc_tensor(tiled, &s);
            copy_with(atom, &desc, &d, args);
        }
        op => panic!(
            "cute: block_copy supports TMA atoms and S2T (tcgen05.Cp..Op) tiled copies, not {op:?}: copy it with \
             cute::copy"
        ),
    }
}

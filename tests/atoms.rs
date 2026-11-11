//! Copy and MMA atoms and the tiled copies built from them: introspection,
//! and the layouts CuTe's constructors produce.

use cute::prelude::*;

/// A layout as a function: its value at every index.
fn values(l: &Layout) -> Vec<usize> {
    (0..l.size()).map(|i| l.call(i)).collect()
}

fn single_warp_mma() -> TiledMma {
    make_tiled_mma(MmaF16BF16Op::new(DType::F16, DType::F32, (16, 8, 16)), Layout::col_major(1))
}

#[test]
fn copy_atom_introspection() {
    let ldsm = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
    assert_eq!(ldsm.value_type(), DType::F16);
    assert_eq!(ldsm.thr_id(), &Layout::new(32, 1));
    // CuTe's `SM75_U32x4_LDSM_N` traits, recast from bits to 16-bit values.
    assert_eq!(ldsm.layout_src_tv(), &Layout::new((32, 8), (8, 1)));
    assert_eq!(ldsm.layout_dst_tv(), &Layout::new((32, (2, 4)), (2, (1, 64))));
    let u = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
    assert_eq!((u.value_type(), u.num_threads(), u.layout_src_tv().size()), (DType::F32, 1, 4));
    assert_eq!(u.thr_id(), &Layout::new(1, 0));
}

#[test]
fn tiled_copy_from_an_mma() {
    let mma = single_warp_mma();
    let ldsm = make_tiled_copy_a(&make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false }), &mma);
    assert_eq!((ldsm.size(), ldsm.tile_shape(), ldsm.value_type()), (32, IntTuple::from((16, 16)), DType::F16));
    assert_eq!(ldsm.thr_id(), &Layout::new(32, 1));
    assert_eq!(values(ldsm.layout_tv_tiled()), values(&mma.layout_a_tv()));
    // Each thread receives its MMA A values ...
    assert_eq!(values(&ldsm.layout_dst_tv_tiled()), values(&mma.layout_a_tv()));
    // ... and supplies the 8-element row of A that its lane addresses:
    // row `l % 8` of matrix `l / 8`, the matrices being A[0:8, 0:8],
    // A[8:16, 0:8], A[0:8, 8:16] and A[8:16, 8:16].
    let src = ldsm.layout_src_tv_tiled();
    for l in 0..32 {
        for v in 0..8 {
            let (m, k) = (l % 8 + 8 * ((l / 8) % 2), 8 * (l / 16) + v);
            assert_eq!(src.call(l + 32 * v), m + 16 * k, "lane {l} value {v}");
        }
    }
    // make_tiled_copy_S/D reuse those layouts and the tiler with another atom.
    let s = make_tiled_copy_s(&make_copy_atom::<F16>(CopyOp::Universal { bits: 128 }), &ldsm);
    assert_eq!((values(s.layout_tv_tiled()), s.tiler_mn()), (values(&src), ldsm.tiler_mn()));
    let d = make_tiled_copy_d(&make_copy_atom::<F16>(CopyOp::Universal { bits: 32 }), &ldsm);
    assert_eq!(values(d.layout_tv_tiled()), values(&mma.layout_a_tv()));
}

#[test]
fn tiled_copy_c_atom_is_the_smallest_c_copy() {
    let c = make_tiled_copy_c_atom(&make_copy_atom::<f32>(CopyOp::Universal { bits: 64 }), &single_warp_mma());
    // CuTe's make_tiled_copy_C_atom, worked by hand for an m16n8 accumulator
    // and a two-value atom: each thread keeps c0 and c1 (row g, columns 2t
    // and 2t + 1), so the tile is rows 0..8 and all 8 columns, in the order
    // (t, c0/c1).
    assert_eq!(c.layout_tv_tiled(), &Layout::new(((4, 8), 2), ((8, 1), 32)));
    let tiler: Tiler =
        Tree::Branch(vec![Tree::Leaf(Some(Layout::new(8, 1))), Tree::Leaf(Some(Layout::new((4, 2), (2, 1))))]);
    assert_eq!(c.tiler_mn(), &tiler);
    assert_eq!(c.tile_shape(), IntTuple::from((8, (4, 2))));
}

#[test]
fn retile_follows_the_copy_value_order() {
    let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
    // Threads 8 x 4, each with a 4 x 2 block (columns fastest): copy value
    // `w` of a thread is row `w / 2`, column `w % 2` of its block.
    let tiled = make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::row_major((4, 2)));
    // A (2, 4) fragment of (column, row): the copy's values are its slots in order.
    let frag = make_fragment::<F16>((2, 4));
    let r = tiled.retile(&frag);
    assert_eq!(r.size(), 8);
    assert_eq!(values(r.layout()), (0..8).collect::<Vec<_>>());
    // The same fragment stored row by row: value `w` is slot `4 * (w % 2) + w / 2`.
    let frag = make_fragment_with_layout::<F16>(Layout::new((2, 4), (4, 1)));
    assert_eq!(values(tiled.retile(&frag).layout()), vec![0, 4, 1, 5, 2, 6, 3, 7]);

    // An MMA fragment retiled for the ldmatrix copy built from the same MMA
    // keeps its value order and its (MMA_M, MMA_K) modes.
    let mma = single_warp_mma();
    let ldsm = make_tiled_copy_a(&atom, &mma);
    let r_a = make_fragment::<F16>((8, 4, 2));
    let view = ldsm.retile(&r_a);
    assert_eq!(view.rank(), 3);
    assert_eq!((view.shape().mode(1).size(), view.shape().mode(2).size()), (4, 2));
    assert_eq!(values(view.layout()), (0..64).collect::<Vec<_>>());
}

#[test]
fn tiled_copy_misuse_is_rejected() {
    let atom = make_copy_atom::<F16>(CopyOp::LdMatrix { num: 4, trans: false });
    let msg = std::panic::catch_unwind(|| make_tiled_copy_tv(&atom, &Layout::row_major((4, 4)), &Layout::col_major(8)))
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("16 threads") && msg.contains("32 threads"), "{msg}");
    let msg = std::panic::catch_unwind(|| make_tiled_copy_tv(&atom, &Layout::row_major((8, 4)), &Layout::col_major(4)))
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("multiple of"), "{msg}");
}

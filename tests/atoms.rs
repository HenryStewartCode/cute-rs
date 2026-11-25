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
    // The `SM75_U32x4_LDSM_N` traits of CuTe's
    // include/cute/atom/copy_traits_sm75.hpp, recast from bits to 16-bit
    // values.
    assert_eq!(ldsm.layout_src_tv(), &Layout::new((32, 8), (8, 1)));
    assert_eq!(ldsm.layout_dst_tv(), &Layout::new((32, (2, 4)), (2, (1, 64))));
    // One thread, as `Copy_Traits<UniversalCopy>` of
    // include/cute/atom/copy_traits.hpp has it.
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

/// `make_cotiled_copy` against CuTe's C++ algorithm, worked by hand: `inv =
/// left_inverse(data)` with `1:0` appended, `layout_tv_data = inv ∘ atom_tv`,
/// a tiler per data mode `filter(make_layout(shape, e_i) ∘ layout_tv_data)`,
/// and `layout_tv = left_inverse(make_layout(shape) ∘ tiler) ∘ layout_tv_data`.
#[test]
fn cotiled_copy_follows_cute() {
    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 128 });
    let by_mode = |ls: Vec<Layout>| -> Tiler { Tree::Branch(ls.into_iter().map(|l| Tree::Leaf(Some(l))).collect()) };
    // A row-major 8 x 4 tile, eight threads with a row of four each: thread t
    // holds addresses 4t..4t + 4, which are row t. The inverse of the data is
    // (4,8):(8,1), so layout_tv_data = (8,4):(1,8), the tilers are 8:1 and
    // 4:1, and the tile is the whole data with layout_tv = (8,4):(1,8).
    let rows = make_cotiled_copy(&atom, &Layout::new((8, 4), (4, 1)), &Layout::new((8, 4), (4, 1)));
    assert_eq!(rows.layout_tv_tiled(), &Layout::new((8, 4), (1, 8)));
    assert_eq!(rows.tiler_mn(), &by_mode(vec![Layout::new(8, 1), Layout::new(4, 1)]));
    assert_eq!(rows.tile_shape(), IntTuple::from((8, 4)));
    // The same threads on a column-major 8 x 4 tile: thread t's addresses
    // are rows 4(t % 2)..4(t % 2) + 4 of column t / 2. The inverse is 32:1,
    // so layout_tv_data = (8,4):(4,1); mode 0's tiler is filter of
    // ((2,4),4):((4,0),1), which is (2,4):(4,1) (row order 0, 4, 1, 5, ...),
    // mode 1's is 4:1, tile2data = ((2,4),4):((4,1),8), its left inverse
    // (4,2,4):(2,1,8), and layout_tv = ((2,4),4):((1,8),2).
    let cols = make_cotiled_copy(&atom, &Layout::new((8, 4), (4, 1)), &Layout::new((8, 4), (1, 8)));
    assert_eq!(cols.layout_tv_tiled(), &Layout::new(((2, 4), 4), ((1, 8), 2)));
    assert_eq!(cols.tiler_mn(), &by_mode(vec![Layout::new((2, 4), (4, 1)), Layout::new(4, 1)]));
    assert_eq!(cols.tile_shape(), IntTuple::from(((2, 4), 4)));
    // A hierarchical thread mode with the same offsets gives the same tiled
    // copy, up to the thread mode's shape.
    let nested = make_cotiled_copy(&atom, &Layout::new(((2, 4), 4), ((4, 8), 1)), &Layout::new((8, 4), (1, 8)));
    assert_eq!(nested.layout_tv_tiled(), &Layout::new(((2, 4), 4), ((1, 8), 2)));
    assert_eq!(nested.tiler_mn(), cols.tiler_mn());
    // Four threads cover the first four rows of the row-major tile: the
    // tilers are 4:1 and 4:1, tile2data = (4,4):(1,8), whose left inverse is
    // (4,2,4):(1,16,4), and layout_tv = (4,4):(1,4).
    let half = make_cotiled_copy(&atom, &Layout::new((4, 4), (4, 1)), &Layout::new((8, 4), (4, 1)));
    assert_eq!(half.layout_tv_tiled(), &Layout::new((4, 4), (1, 4)));
    assert_eq!(half.tiler_mn(), &by_mode(vec![Layout::new(4, 1), Layout::new(4, 1)]));
    // In every case the copy hands thread t exactly the addresses
    // atom_layout_tv gives it: data(tile2data(layout_tv(t, v))) = atom(t, v).
    for (tiled, atom_tv, data) in [
        (&rows, Layout::new((8, 4), (4, 1)), Layout::new((8, 4), (4, 1))),
        (&cols, Layout::new((8, 4), (4, 1)), Layout::new((8, 4), (1, 8))),
        (&nested, Layout::new(((2, 4), 4), ((4, 8), 1)), Layout::new((8, 4), (1, 8))),
        (&half, Layout::new((4, 4), (4, 1)), Layout::new((8, 4), (4, 1))),
    ] {
        let tile2data = Layout::col_major(data.shape().product_each()).compose_tiler(tiled.tiler_mn());
        let addr = data.compose(&tile2data).compose(tiled.layout_tv_tiled());
        assert_eq!(values(&addr), values(&atom_tv), "{atom_tv} on {data}");
    }
}

/// `make_cotiled_copy` rejects an atom whose values reach memory the data
/// does not hold: past its end, or in its holes, for any thread.
#[test]
fn cotiled_copy_checks_the_atom_against_the_data() {
    let atom = make_copy_atom::<f32>(CopyOp::Universal { bits: 32 });
    let fail = |atom_tv: Layout, data: Layout| {
        std::panic::catch_unwind(|| make_cotiled_copy(&atom, &atom_tv, &data))
            .err()
            .and_then(|e| e.downcast_ref::<String>().cloned())
            .unwrap_or_default()
    };
    // Eight values at addresses 0..8 of four elements.
    let msg = fail(Layout::new((1, 8), (0, 1)), Layout::new(4, 1));
    assert!(msg.contains("every value of atom_layout_tv must be at an address that data_layout holds"), "{msg}");
    assert!(
        msg.contains("value 4 of thread 0 of (1,8):(0,1) is at address 4, which data_layout 4:1 does not hold"),
        "{msg}"
    );
    // Addresses 0..4 of a layout holding only the even ones.
    let msg = fail(Layout::new((1, 4), (0, 1)), Layout::new(4, 2));
    assert!(
        msg.contains("value 1 of thread 0 of (1,4):(0,1) is at address 1, which data_layout 4:2 does not hold"),
        "{msg}"
    );
    // CuTe checks only the value mode, the offsets of one thread's values.
    // Threads 8 to 15 here read addresses 32 to 63, past the data, which
    // CuTe's tiled copy would wrap onto threads 0 to 7.
    let msg = fail(Layout::new((16, 4), (4, 1)), Layout::new((8, 4), (4, 1)));
    assert!(msg.contains("must be at an address that data_layout holds"), "{msg}");
    assert!(
        msg.contains("value 0 of thread 8 of (16,4):(4,1) is at address 32, which data_layout (8,4):(4,1)"),
        "{msg}"
    );
    // Thread 7's last four values are past the data (the tiler's complement
    // would fail on this one).
    let msg = fail(Layout::new((8, 8), (4, 1)), Layout::new((8, 4), (4, 1)));
    assert!(msg.contains("value 4 of thread 7 of (8,8):(4,1) is at address 32"), "{msg}");
    // A runtime stride cannot be inverted when the kernel is built.
    let msg = std::panic::catch_unwind(|| {
        Kernel::build("runtime_stride", || {
            let stride = IndexExpr::from(param_scalar::<i32>("stride"));
            let data = Layout::with_strides(4, stride);
            drop(make_cotiled_copy(
                &make_copy_atom::<f32>(CopyOp::Universal { bits: 32 }),
                &Layout::new((1, 4), (0, 1)),
                &data,
            ));
        })
    })
    .err()
    .and_then(|e| e.downcast_ref::<String>().cloned())
    .unwrap_or_default();
    assert!(
        msg.contains(
            "make_cotiled_copy: both layouts need static strides, but atom_layout_tv is (1,4):(0,1) and data_layout is 4:"
        ),
        "{msg}"
    );
}

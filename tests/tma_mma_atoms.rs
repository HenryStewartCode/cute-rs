// SPDX-License-Identifier: MIT OR Apache-2.0

//! The MMA-aware TMA atoms of SM100 (`cute.nvgpu.make_tiled_tma_atom_A/B/C`), at build time: the atom's tiling (box,
//! descriptor dimensions, gmem basis, TMA tensor) is that of `make_tiled_tma_atom` on
//! the equivalent explicit CTA tile, the multicast counts follow the cluster's N mode
//! (A) and M mode (B), the im2col A atom tiles as `make_im2col_tma_atom` on the
//! explicit tile, and the operations each takes are checked.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::nvgpu::{make_im2col_tma_atom_a, make_tiled_tma_atom_a, make_tiled_tma_atom_b, make_tiled_tma_atom_c};
use cute::prelude::*;
use cute::tcgen05::{self, CtaGroup, MmaF16BF16Op, OperandSource, SmemLayoutAtomKind};
use cute::utils::sm100::{make_smem_layout_a, make_smem_layout_b, make_smem_layout_c};

const TILER: (usize, usize, usize) = (128, 256, 64);

fn in_kernel<R>(f: impl FnOnce() -> R) -> R {
    let mut out = None;
    Kernel::builder("tma_mma_atom_case").target(Arch::Sm100a).build(|| out = Some(f()));
    out.unwrap()
}

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(|| in_kernel(f))).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn mma() -> TiledMma {
    let op = MmaF16BF16Op::new(
        DType::F16,
        DType::F32,
        (128, 256, 16),
        CtaGroup::ONE,
        OperandSource::SMEM,
        OperandMajorMode::K,
        OperandMajorMode::K,
    );
    make_tiled_mma(op, Layout::col_major((1, 1, 1)))
}

/// What the tests compare of two atoms: the descriptor's box, extents and strides, the
/// gmem basis, the multicast, the values per instruction and the TMA tensor's layout.
fn tiling<T: Numeric>(info: &cpasync::TmaInfo<T>) -> String {
    let ti = info.atom.tma_info().unwrap();
    let s = ti.spec();
    format!(
        "box {:?} dims {:?} strides {:?} gbasis {} mc {} vals {} tensor {}",
        s.box_dims,
        s.global_dims,
        s.global_strides,
        ti.tma_gbasis(),
        ti.num_multicast(),
        info.atom.num_val_src(),
        info.tma_tensor.layout()
    )
}

/// The explicit equivalent: the K-major 128-byte-swizzle atom tiled over the
/// `(rows, 64)` tile, with that tile as the tiler.
fn explicit<T: Numeric>(g: &Tensor<T>, rows: usize, mc: usize) -> String {
    let atom = tcgen05::make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
    let smem = tile_to_shape_swizzled(&atom, (rows, 64));
    let op: CopyOp = if mc == 1 {
        cpasync::CopyBulkTensorTileG2SOp::default().into()
    } else {
        cpasync::CopyBulkTensorTileG2SMulticastOp::default().into()
    };
    tiling(&cpasync::make_tiled_tma_atom(op, g, smem, (rows, 64), mc))
}

/// A `(512, 256)` and B `(768, 256)`, K-major f16, under the `(128, 256, 64)` tile of a
/// `128x256x16` tcgen05 MMA: the A atom's box is `[64, 128]` and B's `[64, 256]`, as
/// `make_tiled_tma_atom` gives for `(128, 64)` and `(256, 64)` CTA tiles, from one stage
/// of the SMEM layout or the staged one alike.
#[test]
fn mma_aware_atoms_tile_as_the_explicit_cta_tile() {
    in_kernel(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((768, 256), (256, 1)), 16);
        let mma = mma();
        let sa = make_smem_layout_a(&mma, TILER, DType::F16, 4, None);
        let sb = make_smem_layout_b(&mma, TILER, DType::F16, 4, None);
        assert_eq!(sa.rank(), 4);
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let ia = make_tiled_tma_atom_a(g2s, &a, sa.select(&[0, 1, 2]), TILER, &mma, None);
        let ia4 = make_tiled_tma_atom_a(g2s, &a, sa.clone(), TILER, &mma, None);
        let ib = make_tiled_tma_atom_b(g2s, &b, sb.select(&[0, 1, 2]), TILER, &mma, None);
        assert_eq!(ia.atom.tma_info().unwrap().spec().box_dims, vec![64, 128]);
        assert_eq!(ib.atom.tma_info().unwrap().spec().box_dims, vec![64, 256]);
        assert_eq!(tiling(&ia), explicit(&a, 128, 1));
        assert_eq!(tiling(&ia4), tiling(&ia));
        assert_eq!(tiling(&ib), explicit(&b, 256, 1));
        assert_eq!(ia4.smem_layout, sa, "the staged layout is returned as given");
    });
}

/// Multicast counts: on a `(2, 1)` cluster (`cluster_shape_vmnk` `(1, 2, 1, 1)`) A is
/// not shared (the cluster's N mode is 1) and B is split across the two M CTAs; on a
/// `(2, 2)` cluster both are split in two. The non-multicast operation ignores the
/// cluster.
#[test]
fn multicast_counts_follow_the_cluster() {
    in_kernel(|| {
        let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16);
        let b = param_tensor_aligned::<F16>("b", Layout::new((768, 256), (256, 1)), 16);
        let mma = mma();
        let sa = make_smem_layout_a(&mma, TILER, DType::F16, 4, None).select(&[0, 1, 2]);
        let sb = make_smem_layout_b(&mma, TILER, DType::F16, 4, None).select(&[0, 1, 2]);
        let mc = cpasync::CopyBulkTensorTileG2SMulticastOp::default();
        for (cluster, (mca, mcb)) in [((2, 1), (1, 2)), ((2, 2), (2, 2))] {
            let vmnk: Shape = (1usize, cluster.0, cluster.1, 1usize).into();
            let ia = make_tiled_tma_atom_a(mc, &a, sa.clone(), TILER, &mma, Some(vmnk.clone()));
            let ib = make_tiled_tma_atom_b(mc, &b, sb.clone(), TILER, &mma, Some(vmnk));
            assert_eq!(ia.atom.tma_info().unwrap().num_multicast(), mca, "{cluster:?}");
            assert_eq!(ib.atom.tma_info().unwrap().num_multicast(), mcb, "{cluster:?}");
            assert_eq!(tiling(&ia), explicit(&a, 128, mca), "{cluster:?}");
            assert_eq!(tiling(&ib), explicit(&b, 256, mcb), "{cluster:?}");
        }
        let g2s = cpasync::CopyBulkTensorTileG2SOp::default();
        let ib = make_tiled_tma_atom_b(g2s, &b, sb.clone(), TILER, &mma, Some((1usize, 2usize, 2usize, 1usize).into()));
        assert_eq!(ib.atom.tma_info().unwrap().num_multicast(), 1);
    });
}

/// The C atom: a TMA store of the `(128, 256)` tile as the MMA partitions C, tiled as
/// the explicit CTA tile with the same SMEM layout.
#[test]
fn c_atom_stores_the_mma_tile() {
    in_kernel(|| {
        let c = param_tensor_aligned::<F16>("c", Layout::new((256, 512), (512, 1)), 16);
        let mma = mma();
        let sc = make_smem_layout_c(&mma, TILER, DType::F16, 1, false);
        let s2g = cpasync::CopyBulkTensorTileS2GOp;
        let ic = make_tiled_tma_atom_c(s2g, &c, sc.select(&[0, 1, 2]), TILER, &mma);
        let flat = make_composed_layout(sc.swizzle().unwrap(), 0, sc.outer.select(&[0, 1, 2]).mode(0));
        let ex = cpasync::make_tiled_tma_atom(s2g, &c, flat, (128, 256), 1);
        assert_eq!(tiling(&ic), tiling(&ex));
        let red = cpasync::CopyReduceBulkTensorTileS2GOp::default();
        make_tiled_tma_atom_c(red, &c, sc.select(&[0, 1, 2]), TILER, &mma);
    });
}

/// The operations each atom takes, and the SMEM layout's rank.
#[test]
fn mma_aware_atoms_check_their_arguments() {
    let setup = || {
        let a = param_tensor_aligned::<F16>("a", Layout::new((512, 256), (256, 1)), 16);
        let mma = mma();
        let sa = make_smem_layout_a(&mma, TILER, DType::F16, 4, None);
        (a, mma, sa)
    };
    let e = build_panic(|| {
        let (a, mma, sa) = setup();
        make_tiled_tma_atom_a(cpasync::CopyBulkTensorTileS2GOp, &a, sa, TILER, &mma, None);
    });
    assert!(e.contains("make_tiled_tma_atom_A") && e.contains("not a TMA load"), "{e}");
    let e = build_panic(|| {
        let (a, mma, sa) = setup();
        make_tiled_tma_atom_c(cpasync::CopyBulkTensorTileG2SOp::default(), &a, sa, TILER, &mma);
    });
    assert!(e.contains("make_tiled_tma_atom_C") && e.contains("not a TMA store"), "{e}");
    let e = build_panic(|| {
        let (a, mma, sa) = setup();
        make_tiled_tma_atom_a(cpasync::CopyBulkTensorTileG2SOp::default(), &a, sa.select(&[0, 1]), TILER, &mma, None);
    });
    assert!(e.contains("has rank 2"), "{e}");
}

/// `make_im2col_tma_atom_A`: a 3x3 convolution (padding 1) over a `(16, 16, 2)` NHWC
/// activation of 64 channels, read as the GEMM's `(512, (64, 3, 3))` A through the
/// `(128, 64)` tile of the MMA tiler: the descriptor (with its im2col fields) and the TMA
/// tensor are `make_im2col_tma_atom`'s for the explicit `(128, 64)` tile; the multicast
/// splits the 128 pixels across the cluster's N mode.
#[test]
fn im2col_a_atom_tiles_as_the_explicit_cta_tile() {
    in_kernel(|| {
        let (w, h, n, c) = (16usize, 16usize, 2usize, 64usize);
        let layout = Layout::new(((w, h, n), c), ((c, w * c, w * h * c), 1));
        let x = param_tensor_aligned::<F16>("x", layout, 16);
        let mma = mma();
        let sa = make_smem_layout_a(&mma, TILER, DType::F16, 4, None);
        let conv = cpasync::Im2colOptions::conv(&[3, 3], &[1, 1], &[1, 1], &[1, 1], &[1, 1]);
        let op = cpasync::CopyBulkTensorIm2ColG2SOp::default();
        let ia = make_im2col_tma_atom_a(op, &x, sa.clone(), TILER, &mma, &conv, None);
        let atom = tcgen05::make_smem_layout_atom::<F16>(SmemLayoutAtomKind::K_SW128);
        let ex = cpasync::make_im2col_tma_atom(op, &x, tile_to_shape_swizzled(&atom, (128, 64)), (128, 64), &conv);
        let spec = |i: &cpasync::TmaInfo<F16>| format!("{:?}", i.atom.tma_info().unwrap().spec());
        assert_eq!(spec(&ia), spec(&ex));
        assert_eq!(ia.tma_tensor.layout().to_string(), ex.tma_tensor.layout().to_string());
        assert_eq!(ia.atom.num_val_src(), ex.atom.num_val_src());
        let mc = cpasync::CopyBulkTensorIm2ColG2SMulticastOp::default();
        let ia = make_im2col_tma_atom_a(mc, &x, sa, TILER, &mma, &conv, Some((1usize, 1usize, 2usize, 1usize).into()));
        let ex = cpasync::make_im2col_tma_atom(
            mc,
            &x,
            tile_to_shape_swizzled(&atom, (128, 64)),
            (128, 64),
            &conv.clone().num_multicast(2),
        );
        assert_eq!(ia.atom.tma_info().unwrap().num_multicast(), 2);
        assert_eq!(spec(&ia), spec(&ex));
    });
}

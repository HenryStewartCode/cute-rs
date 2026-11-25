//! TMA's im2col mode and the SM100 modes:
//! the feature kernels in the simulator, their descriptors, MLIR and diagnostics.

mod feature_kernels;

use cute::prelude::*;
use feature_kernels::Case;

fn simulate(case: &Case, small: bool) -> Vec<KernelArg> {
    let mut args = feature_kernels::inputs_with_len(&case.kernel, small, 256);
    case.kernel
        .simulate_all_orders(case.cfg, &mut args, &SimOptions::default())
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    args
}

#[test]
fn im2col_kernels_match_the_rust_reference() {
    for case in feature_kernels::tma_im2col_cases_sm90() {
        for small in [true, false] {
            (case.check)(&simulate(&case, small));
        }
    }
}

#[test]
fn sm100_kernels_match_the_rust_reference() {
    for case in feature_kernels::tma_sm100_cases().into_iter().chain([feature_kernels::tma_override()]) {
        for small in [true, false] {
            (case.check)(&simulate(&case, small));
        }
    }
}

use std::panic::{AssertUnwindSafe, catch_unwind};

use cute::tma::{Im2colParams, TensorMapDataType, TensorMapSwizzle};
use feature_kernels::{conv_2d, conv_2d_strided};

fn build_panic(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected the kernel build to panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// The descriptor of an im2col load atom, as C++ CuTe's `make_im2col_tma_copy_desc` derives
/// it: dimensions (C, W, H, N) with byte strides, the traversal strides, corners
/// `-lower_padding` and `upper_padding - (filter - 1) * dilation`, a pixel's channels
/// (the SMEM layout's contiguous run) and the pixels (its next run); and the TMA
/// tensor's GEMM shape `(Q * P * N, (C, S, R))`.
#[test]
fn im2col_descriptor_fields_follow_cute() {
    for (conv, stride, lower, upper, swizzle, tile_k) in [
        (conv_2d(), 1, -1, -1, TensorMapSwizzle::s64b, 32usize),
        (conv_2d_strided(), 2, -2, -3, TensorMapSwizzle::s32b, 16),
    ] {
        let c = conv.c;
        let (w, h, n) = (conv.extent[0], conv.extent[1], conv.n);
        let out = conv.out();
        let m = conv.m();
        let kind = if tile_k == 32 { SmemLayoutAtomKind::K_SW64 } else { SmemLayoutAtomKind::K_SW32 };
        Kernel::builder("im2col_desc").block_dim([32, 1, 1]).build(move || {
            let layout = Layout::new(((w, h, n), c), ((c, w * c, w * h * c), 1));
            let x = param_tensor_aligned::<F16>("x", layout, 16);
            let smem = tile_to_shape_swizzled(&make_smem_layout_atom::<F16>(kind), (64, tile_k));
            let dil = if stride == 2 { 2 } else { 1 };
            let pad_hi = conv.pad_hi[0] as i32;
            let opts = cpasync::Im2colOptions::conv(&[3, 3], &[-lower; 2], &[pad_hi; 2], &[stride; 2], &[dil; 2]);
            let info = cpasync::make_im2col_tma_atom(
                cpasync::CopyBulkTensorIm2ColG2SOp::default(),
                &x,
                smem,
                (64, tile_k),
                &opts,
            );
            let spec = info.atom.tma_info().unwrap().spec().clone();
            assert_eq!(spec.data_type, TensorMapDataType::float16);
            let dims: Vec<usize> = spec.global_dims.iter().map(|e| e.as_static().unwrap()).collect();
            assert_eq!(dims, vec![c, w, h, n]);
            let strides: Vec<usize> = spec.global_strides.iter().map(|e| e.as_static().unwrap()).collect();
            assert_eq!(strides, vec![2 * c, 2 * w * c, 2 * w * h * c]);
            assert_eq!(spec.element_strides, vec![1, stride as u32, stride as u32, 1]);
            assert_eq!(spec.swizzle, swizzle);
            assert_eq!(
                spec.im2col,
                Some(Im2colParams {
                    lower_corner: [lower, lower, 0],
                    upper_corner: [upper, upper, 0],
                    channels_per_pixel: tile_k as u32,
                    pixels_per_column: 64,
                })
            );
            assert_eq!(info.tma_tensor.layout().to_string(), format!("({m},({c},3,3)):(1@0,(1@1,1@2,1@3))"));
            assert_eq!(m, n * out[0] * out[1]);
            assert_eq!(info.atom.num_val_src(), 64 * tile_k);
        });
    }
}

use cute::warpgroup::{SmemLayoutAtomKind, make_smem_layout_atom};

/// The instructions' PTX text, and the targets that take them.
#[test]
fn modes_emit_their_ptx_and_need_their_targets() {
    use cute::validate::check_target;
    let text = |c: &Case| c.kernel.to_mlir();
    let im2col = text(&feature_kernels::tma_im2col_2d_fprop());
    assert!(
        im2col.contains(
            "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, \
             $6}], [$2], {$7, $8};"
        ),
        "{im2col}"
    );
    assert!(im2col.contains("\"r,l,r,r,r,r,r,h,h,~{memory}\""), "{im2col}");
    let im3 = text(&feature_kernels::tma_im2col_3d_fprop());
    assert!(im3.contains("cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"));
    assert!(im3.contains("{$8, $9, $10};"), "{im3}");
    let store = text(&feature_kernels::tma_im2col_store());
    assert!(
        store.contains(
            "cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group [$1, {$2, $3, $4, $5}], [$0];"
        ),
        "{store}"
    );
    let gather = text(&feature_kernels::tma_gather4());
    assert!(
        gather.contains(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [$0], [$1, {$3, \
             $4, $5, $6, $7}], [$2];"
        ),
        "{gather}"
    );
    let scatter = text(&feature_kernels::tma_scatter4());
    assert!(
        scatter.contains(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [$1, {$2, $3, $4, $5, $6}], [$0];"
        ),
        "{scatter}"
    );
    let mask = text(&feature_kernels::bulk_store_byte_mask());
    assert!(mask.contains("cp.async.bulk.global.shared::cta.bulk_group.cp_mask [$0], [$1], $2, $3;"), "{mask}");
    let over = text(&feature_kernels::tma_override());
    assert!(
        over.contains(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.override::global_address.\
             override::global_dim_stride [$0], [$1, $5, {$6, $7}, {$8}, $9, {$3, $4}], [$2];"
        ),
        "{over}"
    );
    assert!(
        over.contains("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride [$1, $4, {$5, $6}, {$7}, $8, {$2, $3}], [$0];"),
        "{over}"
    );
    // Targets.
    let ok = |c: &Case, a: Arch| check_target(c.kernel.ir(), a).map(|_| ()).map_err(|e| e.message);
    let im = feature_kernels::tma_im2col_2d_fprop();
    ok(&im, Arch::Sm90).unwrap();
    assert!(ok(&im, Arch::Sm89).is_err());
    let g = feature_kernels::tma_gather4();
    ok(&g, Arch::Sm100a).unwrap();
    ok(&g, Arch::Sm100f).unwrap();
    ok(&g, Arch::Sm100).unwrap_err();
    let e = ok(&g, Arch::Sm90a).unwrap_err();
    assert!(e.contains("gather4"), "{e}");
    let e = ok(&feature_kernels::tma_scatter4(), Arch::Sm120a).unwrap_err();
    assert!(e.contains("scatter4"), "{e}");
    ok(&feature_kernels::tma_scatter4(), Arch::Sm110a).unwrap();
    let e = ok(&feature_kernels::bulk_store_byte_mask(), Arch::Sm90a).unwrap_err();
    assert!(e.contains("cp_mask"), "{e}");
    ok(&feature_kernels::bulk_store_byte_mask(), Arch::Sm100).unwrap();
    let o = feature_kernels::tma_override();
    ok(&o, Arch::Sm107f).unwrap();
    for a in [Arch::Sm100a, Arch::Sm100f, Arch::Sm110f, Arch::Sm90a] {
        let e = ok(&o, a).unwrap_err();
        assert!(e.contains("override"), "{a:?}: {e}");
    }
    let e = ok(&feature_kernels::tma_fp4_unpacked(), Arch::Sm90a).unwrap_err();
    assert!(e.contains("f416u4_align16b") && e.contains("sm_100"), "{e}");
    ok(&feature_kernels::tma_fp6_unpacked(), Arch::Sm100a).unwrap();
}

/// What the atom makers and copies refuse.
#[test]
fn diagnostics() {
    // A rank-5 (3-D) convolution whose filter offsets reach 4 * 8 = 32, past the 31 its
    // offsets hold.
    let e = build_panic(|| {
        Kernel::builder("bad_offsets").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 8, 1), 16), ((16, 128, 1024, 8192), 1)), 16);
            let opts = cpasync::Im2colOptions::conv(&[5, 1, 1], &[0; 3], &[0; 3], &[1; 3], &[8, 1, 1]);
            let smem = Layout::new((64, 16), (16, 1));
            cpasync::make_im2col_tma_atom(cpasync::CopyBulkTensorIm2ColG2SOp::default(), &x, smem, (64, 16), &opts);
        });
    });
    assert!(e.contains("from 0 to 32") && e.contains("0 to 31"), "{e}");
    // Padding of 20 puts a rank-5 corner at -20, past -16.
    let e = build_panic(|| {
        Kernel::builder("bad_corner").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 8, 1), 16), ((16, 128, 1024, 8192), 1)), 16);
            let opts = cpasync::Im2colOptions::conv(&[1, 1, 1], &[20, 0, 0], &[0; 3], &[1; 3], &[1; 3]);
            let smem = Layout::new((64, 16), (16, 1));
            cpasync::make_im2col_tma_atom(cpasync::CopyBulkTensorIm2ColG2SOp::default(), &x, smem, (64, 16), &opts);
        });
    });
    assert!(e.contains("lower corner") && e.contains("outside -16 to 15"), "{e}");
    // A load without the im2col parameters, and an M-major SMEM layout.
    let e = build_panic(|| {
        Kernel::builder("no_params").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 1), 16), ((16, 128, 1024), 1)), 16);
            let smem = Layout::new((64, 16), (16, 1));
            let opts = cpasync::Im2colOptions::new();
            cpasync::make_im2col_tma_atom(cpasync::CopyBulkTensorIm2ColG2SOp::default(), &x, smem, (64, 16), &opts);
        });
    });
    assert!(e.contains("needs lower_corner_whd"), "{e}");
    let e = build_panic(|| {
        Kernel::builder("m_major").block_dim([32, 1, 1]).build(|| {
            let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 1), 16), ((16, 128, 1024), 1)), 16);
            let smem = Layout::new((64, 16), (1, 64));
            let opts = cpasync::Im2colOptions::conv(&[3, 3], &[1; 2], &[1; 2], &[1; 2], &[1; 2]);
            cpasync::make_im2col_tma_atom(cpasync::CopyBulkTensorIm2ColG2SOp::default(), &x, smem, (64, 16), &opts);
        });
    });
    assert!(e.contains("K-major"), "{e}");
    // An index tensor without a broadcast mode, and one with fewer than four rows.
    for (index, rows, needle) in [((1usize, 32usize), 12usize, "exactly one broadcast"), ((1, 0), 3, "has 3 indices")] {
        let e = build_panic(|| {
            Kernel::builder("bad_index").block_dim([32, 1, 1]).build(|| {
                let a = param_tensor_aligned::<f32>("a", Layout::new((rows, 32), (32, 1)), 16);
                let idx = param_tensor::<i32>("idx", Layout::new((rows, 32), index));
                let opts = cpasync::TmaAtomOptions::new().gmem_coord_tensor(&idx);
                let op = cpasync::CopyBulkTensor2DGather4G2SOp::default();
                cpasync::make_tiled_tma_atom_with(op, &a, Layout::new((4, 32), (32, 1)), (4, 32), 1, &opts);
            });
        });
        assert!(e.contains(needle), "{e}");
    }
    // An FP6 box of 24 values (not a whole number of 16-value groups), and an unpacked FP4
    // box of 64 values (32 bytes, not the 64 .b4x16_p64 needs).
    let e = build_panic(|| {
        Kernel::builder("fp6_box").block_dim([32, 1, 1]).build(|| {
            let a = param_tensor_aligned::<F6E2M3FN>("a", Layout::new((4, 128), (128, 1)), 32);
            let op = cpasync::CopyBulkTensorTileG2SOp::default();
            cpasync::make_tiled_tma_atom(op, &a, Layout::new((4, 24), (24, 1)), (4, 24), 1);
        });
    });
    assert!(e.contains("groups of 16"), "{e}");
    let e = build_panic(|| {
        Kernel::builder("p64_box").block_dim([32, 1, 1]).build(|| {
            let a = param_tensor_aligned::<F4E2M1FN>("a", Layout::new((4, 128), (128, 1)), 32);
            let opts = cpasync::TmaAtomOptions::new().internal_type(DType::U8);
            let op = cpasync::CopyBulkTensorTileG2SOp::default();
            cpasync::make_tiled_tma_atom_with(op, &a, Layout::new((4, 64), (64, 1)), (4, 64), 1, &opts);
        });
    });
    assert!(e.contains("exactly 64 bytes"), "{e}");
    // A byte-masked store needs its mask.
    let e = build_panic(|| {
        Kernel::builder("no_mask").block_dim([32, 1, 1]).build(|| {
            let out = param_tensor_aligned::<u8>("out", Layout::col_major(64), 16);
            let s = make_smem_tensor_aligned::<u8>(Layout::col_major(64), 128);
            copy(&make_copy_atom::<u8>(cpasync::CopyBulkS2GByteMaskOp), &s, &out);
        });
    });
    assert!(e.contains("needs `byte_mask`"), "{e}");
}

/// An im2col atom's descriptor used by a tiled copy (tile coordinates): the simulator
/// refuses the access.
#[test]
fn im2col_descriptor_with_tile_coordinates_is_reported() {
    let kernel = Kernel::builder("tile_on_im2col").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 1), 16), ((16, 128, 1024), 1)), 16);
        let smem = Layout::new((64, 16), (16, 1));
        let opts = cpasync::Im2colOptions::conv(&[3, 3], &[1; 2], &[1; 2], &[1; 2], &[1; 2]);
        let im = cpasync::make_im2col_tma_atom(
            cpasync::CopyBulkTensorIm2ColG2SOp::default(),
            &x,
            smem.clone(),
            (64, 16),
            &opts,
        );
        let (tile, tma_tensor) =
            cpasync::make_tiled_tma_atom(cpasync::CopyBulkTensorTileG2SOp::default(), &x, smem.clone(), (64, 16), 1)
                .into();
        let s = make_smem_tensor_aligned::<F16>(smem, 128);
        let mbar = alloc_smem::<u64>(1, 8);
        if_(thread_idx().x.eq(0), || mbarrier_init(&mbar, 1));
        mbarrier_init_fence();
        sync_threads();
        let g = local_tile(&tma_tensor, (64, 16), (0, 0));
        let (ts, tg) = cpasync::tma_partition(&tile, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        let desc = im.atom.tma_info().unwrap().descriptor().get_ptr();
        if_(elect_one(), || mbarrier_arrive_and_expect_tx(&mbar, 2048));
        copy_with(&tile, &tg, &ts, CopyArgs::new().tma_bar_ptr(&mbar).tma_desc_ptr(desc));
        mbarrier_wait(&mbar, 0);
    });
    let mut args = feature_kernels::inputs(&kernel, true);
    let e = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap_err();
    assert!(e.message.contains("a tile-mode access with an im2col tensor map"), "{e}");
}

/// `cute.prefetch` of im2col boxes (`.im2col` with offsets) and gather4 rows
/// (`prefetch_atom_aux`, `.tile::gather4`): checked in the simulator (the filter base in
/// the bounding box) and as PTX text.
#[test]
fn im2col_and_gather4_prefetches() {
    let kernel = Kernel::builder("tma_mode_prefetches").block_dim([32, 1, 1]).build(|| {
        let x = param_tensor_aligned::<F16>("x", Layout::new(((8, 8, 1), 16), ((16, 128, 1024), 1)), 16);
        let smem = Layout::new((64, 16), (16, 1));
        let opts = cpasync::Im2colOptions::conv(&[3, 3], &[1; 2], &[1; 2], &[1; 2], &[1; 2]);
        let (atom, tma_tensor) = cpasync::make_im2col_tma_atom(
            cpasync::CopyBulkTensorIm2ColG2SOp::default(),
            &x,
            smem.clone(),
            (64, 16),
            &opts,
        )
        .into();
        let s = make_smem_tensor_aligned::<F16>(smem, 128);
        let g = local_tile(&tma_tensor, (64, 16), (0, 4));
        let (_, tg) = cpasync::tma_partition(&atom, 0, &make_layout(1, 0), &s.group_modes(0, 2), &g.group_modes(0, 2));
        prefetch_atom(&atom, &tg);
        let a = param_tensor_aligned::<f32>("a", Layout::new((12, 32), (32, 1)), 16);
        let idx = param_tensor::<i32>("idx", Layout::new((12, 32), (1, 0)));
        let gs = Layout::new((8, 32), (32, 1));
        let gopts = cpasync::TmaAtomOptions::new().gmem_coord_tensor(&idx);
        let (gatom, gt) = cpasync::make_tiled_tma_atom_with(
            cpasync::CopyBulkTensor2DGather4G2SOp::default(),
            &a,
            gs.clone(),
            (8, 32),
            1,
            &gopts,
        )
        .into();
        let sa = make_smem_tensor_aligned::<f32>(gs, 128);
        let (_, tga, tgi) = cpasync::tma_partition_indexed(
            &gatom,
            0,
            &make_layout(1, 0),
            &sa.group_modes(0, 2),
            &local_tile(&gt, (8, 32), (0, 0)).group_modes(0, 2),
            &local_tile(&idx, (8, 32), (0, 0)).group_modes(0, 2),
        );
        cpasync::prefetch_atom_aux(&gatom, &tga, &tgi);
    });
    let mlir = kernel.to_mlir();
    assert!(
        mlir.contains("cp.async.bulk.prefetch.tensor.4d.L2.global.im2col [$0, {$1, $2, $3, $4}], {$5, $6};"),
        "{mlir}"
    );
    assert!(
        mlir.contains("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [$0, {$1, $2, $3, $4, $5}];"),
        "{mlir}"
    );
    let mut args = feature_kernels::inputs(&kernel, true);
    let report = kernel.simulate(LaunchConfig::new([1, 1, 1], [32, 1, 1]), &mut args).unwrap();
    assert_eq!(report.prefetches, 3);
    cute::validate::check_target(kernel.ir(), Arch::Sm100a).unwrap();
}
